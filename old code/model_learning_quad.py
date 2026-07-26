# model_learning_quad.py -- multi-step learning of the Koopman observables (paper Sec. III).
#
# Pipeline order:  load_data.m -> export_config.m -> THIS FILE -> Terminal_set_quad.m -> KDPC_quad_sim.m
#
# Solves problem (7): learn phi, Psi~, Gamma~ jointly by minimising the
# MULTI-STEP output prediction error, so errors do not propagate with horizon.
#
# CHANGES vs. the previous version
#  * Hyperparameters come from data/config.mat (written by export_config.m).
#    They were previously hard-coded here and re-declared in two .m files.
#  * ALL parameters are exported, biases included, to data/net.mat.  The old
#    script saved only `.weight`, so MATLAB's tanh(W2*tanh(W1*x)) evaluated a
#    different function than the trained network as soon as bias=True.
#  * Equilibrium-shifted observables phi~(x) = phi(x) - phi(0) restore
#    Assumption 2.4 (phi(0)=0) while keeping the biases.
#  * Per-channel loss weighting so yaw does not dominate the MSE.
#  * Held-out R^2 reported per channel AND per prediction step.

import json
import os

import matplotlib
import numpy as np

# WSL usually has no display; fall back to a file-only backend so a headless run
# still saves data/training_curve.png instead of dying at plt.show().
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import scipy.io
import torch
import torch.nn as nn
from scipy.io import savemat

# ---------------------------------------------------------------- config
cfg = scipy.io.loadmat("data/config.mat")


def _scalar(key):
    """loadmat wraps MATLAB scalars as (1,1) arrays; numpy >= 2.0 refuses to
    convert those implicitly, so unwrap explicitly."""
    return int(np.asarray(cfg[key]).ravel()[0])


def _cellstr(key):
    """MATLAB cellstr -> list[str], robust to how loadmat nests the object array."""
    return [str(np.asarray(s).ravel()[0]) for s in np.asarray(cfg[key]).ravel()]


Tini = _scalar("Tini"); N = _scalar("N"); n_basis = _scalar("n_basis")
ny_cfg = _scalar("ny"); nu_cfg = _scalar("nu")
out_labels = _cellstr("out_labels")

epochs = 6000
lr = 1e-3
weight_decay = 1e-5
val_frac = 0.2          # held-out slice of the TRAIN set for early-stopping
torch.manual_seed(41)

# Relative weight per output channel in the loss. Position is what we control;
# yaw is weighted down because it is the weakest-excited, weakest-actuated
# channel (see CASCADE_DEBATE.md) and otherwise dominates the gradient.
channel_weight = {"x": 1.0, "y": 1.0, "z": 1.0, "yaw": 0.3}

# ---------------------------------------------------------------- data
u_data = scipy.io.loadmat("data/u_data.mat")["u_data"]
y_data = scipy.io.loadmat("data/y_data.mat")["y_data"]
u_test = scipy.io.loadmat("data/u_data_test.mat")["u_data_test"]
y_test = scipy.io.loadmat("data/y_data_test.mat")["y_data_test"]

nu, ny = u_data.shape[1], y_data.shape[1]
assert ny == ny_cfg, f"y_data has {ny} cols but config says ny={ny_cfg}. Re-run load_data.m."
assert nu == nu_cfg, f"u_data has {nu} cols but config says nu={nu_cfg}."
assert len(out_labels) == ny

rl = scipy.io.loadmat("data/run_lengths.mat")
run_lengths = rl["run_lengths"].ravel().astype(int)
run_lengths_test = rl["run_lengths_test"].ravel().astype(int)

in1 = (Tini - 1) * nu + Tini * ny
in2 = n_basis + N * nu
out = N * ny
print(f"nu={nu}, ny={ny}, Tini={Tini}, N={N}, n_basis={n_basis}")


def build_regressors(u, y, lens, Tini, N):
    """Hankel windows, eq. (6). Time-major flattening -- must match build_regressors.m."""
    X1s, X2s, Ys = [], [], []
    start = 0
    for L in lens:
        uu, yy = u[start : start + L], y[start : start + L]
        for i in range(L - Tini - N):
            X1s.append(np.concatenate([uu[i + 1 : i + Tini].ravel(),
                                       yy[i + 1 : i + Tini + 1].ravel()]))
            X2s.append(uu[i + Tini : i + Tini + N].ravel())
            Ys.append(yy[i + Tini + 1 : i + Tini + 1 + N].ravel())
        start += L
    return tuple(torch.FloatTensor(np.array(a)) for a in (X1s, X2s, Ys))


X1, X2, Y = build_regressors(u_data, y_data, run_lengths, Tini, N)
X1_te, X2_te, Y_te = build_regressors(u_test, y_test, run_lengths_test, Tini, N)
print(f"regressors: X1 {tuple(X1.shape)}, X2 {tuple(X2.shape)}, Y {tuple(Y.shape)}")

n_val = int(val_frac * len(X1))
perm = torch.randperm(len(X1))
tr_i, va_i = perm[:-n_val], perm[-n_val:]
# Materialise the splits once. Indexing with a permutation tensor copies the
# whole array, and doing that inside a 6000-iteration full-batch loop is pure
# overhead.
X1_tr, X2_tr, Y_tr = X1[tr_i].contiguous(), X2[tr_i].contiguous(), Y[tr_i].contiguous()
X1_va, X2_va, Y_va = X1[va_i].contiguous(), X2[va_i].contiguous(), Y[va_i].contiguous()

# Loss weights, tiled over the N future steps (Y is time-major: [y(k+1); ...]).
w_chan = torch.FloatTensor([channel_weight.get(l, 1.0) for l in out_labels])
W_loss = w_chan.repeat(N)


class Model(nn.Module):
    """phi = tanh(W2 tanh(W1 x + b1) + b2);  y_pred = W3 [phi; u_f] + b3.

    Biases are ON: they fit materially better. Assumption 2.4 (phi(0)=0) is
    restored downstream by subtracting z_eq = phi(0) -- see load_net.m. The
    readout is *linear in the future inputs*, which is what makes the MPC a QP.
    """

    def __init__(self):
        super().__init__()
        self.l_1 = nn.Linear(in1, n_basis, bias=True)
        self.l_2 = nn.Linear(n_basis, n_basis, bias=True)
        self.l_3 = nn.Linear(in2, out, bias=True)

    def phi(self, x1):
        return torch.tanh(self.l_2(torch.tanh(self.l_1(x1))))

    def forward(self, x1, x2):
        return self.l_3(torch.cat((self.phi(x1), x2), 1))


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=200)


def wmse(pred, targ):
    return (W_loss * (pred - targ) ** 2).mean()


losses, val_losses = [], []
best, best_state = float("inf"), None
for i in range(epochs):
    model.train()
    loss = wmse(model(X1_tr, X2_tr), Y_tr)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())

    with torch.no_grad():
        vl = wmse(model(X1_va, X2_va), Y_va).item()
    val_losses.append(vl); sched.step(vl)
    if vl < best:
        best = vl
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if i % 200 == 0:
        with torch.no_grad():
            tl = wmse(model(X1_te, X2_te), Y_te).item()
        print(f"epoch {i:5d} | train {loss.item():.6f} | val {vl:.6f} | test {tl:.6f}")

model.load_state_dict(best_state)   # early stopping on the validation split
print(f"restored best validation loss {best:.6f}")

# ---------------------------------------------------------------- evaluation
with torch.no_grad():
    Yp = model(X1_te, X2_te).numpy()
Yt = Y_te.numpy()


def r2(a, b):
    return 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)


print("\nHeld-out (test) R^2 per channel, over all N steps:")
for c, lab in enumerate(out_labels):
    idx = np.arange(c, out, ny)
    print(f"  {lab:>4}: R^2 = {r2(Yt[:, idx], Yp[:, idx]):.4f}")

print("\nHeld-out R^2 per prediction step (degradation with horizon):")
for j in [1, 5, 10, N]:
    idx = np.arange((j - 1) * ny, j * ny)
    print(f"  step {j:2d}: R^2 = {r2(Yt[:, idx], Yp[:, idx]):.4f}")

# ---------------------------------------------------------------- export
sd = {k: v.detach().numpy() for k, v in model.state_dict().items()}
z_eq = model.phi(torch.zeros(1, in1)).detach().numpy().ravel()
print(f"\n||phi(0)|| = {np.linalg.norm(z_eq):.4f}  "
      "(subtracted downstream to satisfy Assumption 2.4)")

savemat("data/net.mat", {
    "W1": sd["l_1.weight"], "b1": sd["l_1.bias"],
    "W2": sd["l_2.weight"], "b2": sd["l_2.bias"],
    "W3": sd["l_3.weight"], "b3": sd["l_3.bias"],
    "z_eq": z_eq,
    "Tini": Tini, "N": N, "n_basis": n_basis, "nu": nu, "ny": ny,
})
savemat("data/params.mat",
        {"Tini": Tini, "N": N, "n_basis": n_basis, "nu": nu, "ny": ny})

# First 500 regressor columns, so check_pipeline.m can verify that
# build_regressors.m produces byte-identical windows to the ones trained on.
savemat("data/X1_py.mat", {"X1_py": X1[:500].numpy().T})

with open("data/train_report.json", "w") as f:
    json.dump({"best_val": best,
               "r2_channel": {l: float(r2(Yt[:, np.arange(c, out, ny)],
                                          Yp[:, np.arange(c, out, ny)]))
                              for c, l in enumerate(out_labels)}}, f, indent=2)

plt.semilogy(losses, label="train"); plt.semilogy(val_losses, label="val")
plt.xlabel("epoch"); plt.ylabel("weighted MSE"); plt.grid(True); plt.legend()
plt.savefig("data/training_curve.png", dpi=110)
plt.show()

print("\nSaved data/net.mat + params.mat. Next: Terminal_set_quad.m")
