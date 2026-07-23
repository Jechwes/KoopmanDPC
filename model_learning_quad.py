# model_learning_quad.py — MIMO version of model_learning_tanh.py (pendulum example)
#
# Learns the Koopman lifting for KDPC on the nano-drone data:
#   two bias-free tanh layers lift (u_ini, y_ini) -> n_basis observables,
#   a final bias-free linear layer maps [observables ; future inputs] -> future outputs.
#
# Run AFTER load_data.m, from the project folder:  python model_learning_quad.py
# Saves weight1/2/3.mat, X.mat, y.mat, params.mat into data/.
#
# Regressor layout (must match the MATLAB controller side!):
#   x1 = [ u(k-Tini+1..k-1) ; y(k-Tini+1..k) ]  flattened TIME-MAJOR:
#        [u1..u4 @ t, u1..u4 @ t+1, ...]  then same for y
#   In MATLAB, with u_ini of size (Tini-1) x nu:  reshape(u_ini', [], 1)

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.io import savemat
import torch
import torch.nn as nn

# ----------------------------- hyperparameters -----------------------------
Tini = 5        # past window (u and y)
N = 20          # prediction horizon (0.2 s at 100 Hz)
n_basis = 40    # number of Koopman observables
epochs = 3000
lr = 1e-2
weight_decay = 1e-5
torch.manual_seed(41)

# ----------------------------- load data -----------------------------------
u_data = scipy.io.loadmat("data/u_data.mat")["u_data"]            # (T, 4) normalized
y_data = scipy.io.loadmat("data/y_data.mat")["y_data"]            # (T, 4) normalized
u_test = scipy.io.loadmat("data/u_data_test.mat")["u_data_test"]
y_test = scipy.io.loadmat("data/y_data_test.mat")["y_data_test"]
rl = scipy.io.loadmat("data/run_lengths.mat")
run_lengths = rl["run_lengths"].ravel().astype(int)
run_lengths_test = rl["run_lengths_test"].ravel().astype(int)

nu = u_data.shape[1]
ny = y_data.shape[1]
in1 = (Tini - 1) * nu + Tini * ny   # lifting-network input
in2 = n_basis + N * nu              # readout input
out = N * ny                        # predicted future outputs

print(f"nu={nu}, ny={ny}, Tini={Tini}, N={N}, n_basis={n_basis}")
print(f"train samples={len(u_data)}, test samples={len(u_test)}")


# ------------------- Hankel regressors (per run, no boundary crossing) -----
def build_regressors(u, y, lens, Tini, N):
    X1s, X2s, Ys = [], [], []
    start = 0
    for L in lens:
        uu, yy = u[start : start + L], y[start : start + L]
        n_samp = L - Tini - N
        for i in range(n_samp):
            u_ini = uu[i + 1 : i + Tini].ravel()          # (Tini-1)*nu
            y_ini = yy[i + 1 : i + Tini + 1].ravel()      # Tini*ny
            u_f = uu[i + Tini : i + Tini + N].ravel()     # N*nu
            y_f = yy[i + Tini + 1 : i + Tini + 1 + N].ravel()  # N*ny
            X1s.append(np.concatenate([u_ini, y_ini]))
            X2s.append(u_f)
            Ys.append(y_f)
        start += L
    return (
        torch.FloatTensor(np.array(X1s)),
        torch.FloatTensor(np.array(X2s)),
        torch.FloatTensor(np.array(Ys)),
    )


X1, X2, Y = build_regressors(u_data, y_data, run_lengths, Tini, N)
X1_test, X2_test, Y_test = build_regressors(u_test, y_test, run_lengths_test, Tini, N)
print(f"regressors: X1 {tuple(X1.shape)}, X2 {tuple(X2.shape)}, Y {tuple(Y.shape)}")


# ----------------------------- model ---------------------------------------
class Model(nn.Module):
    def __init__(self, in_1, nb, in_2, n_out):
        super().__init__()
        self.l_1 = nn.Linear(in_1, nb, bias=False)
        self.l_2 = nn.Linear(nb, nb, bias=False)
        self.l_3 = nn.Linear(in_2, n_out, bias=False)

    def forward(self, x1, x2):
        z = torch.tanh(self.l_1(x1))
        z = torch.tanh(self.l_2(z))
        return self.l_3(torch.cat((z, x2), 1))


model = Model(in1, n_basis, in2, out)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# ----------------------------- training ------------------------------------
losses = []
for i in range(epochs):
    y_pred = model(X1, X2)
    loss = criterion(y_pred, Y)
    losses.append(loss.item())
    if i % 100 == 0:
        with torch.no_grad():
            test_loss = criterion(model(X1_test, X2_test), Y_test).item()
        print(f"epoch {i:5d} | train {loss.item():.6f} | test {test_loss:.6f}")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.semilogy(losses)
plt.xlabel("epoch"); plt.ylabel("MSE loss"); plt.grid(True)
plt.show()

# --------------------- multi-step R^2 on test set (per channel) ------------
with torch.no_grad():
    Yp = model(X1_test, X2_test).numpy()
Yt = Y_test.numpy()
print("\nTest R^2 per output channel (over all N future steps):")
labels = ["x", "y", "z", "yaw", "vx", "vy", "vz", "qx", "qy"][:ny]
for c in range(ny):
    idx = np.arange(c, out, ny)  # columns of channel c across horizon
    ss_res = np.sum((Yt[:, idx] - Yp[:, idx]) ** 2)
    ss_tot = np.sum((Yt[:, idx] - Yt[:, idx].mean()) ** 2)
    print(f"  {labels[c]:>4}: R^2 = {1 - ss_res / ss_tot:.4f}")

# ----------------------------- export --------------------------------------
savemat("data/weight1.mat", {"weight1": model.l_1.weight.detach().numpy()})
savemat("data/weight2.mat", {"weight2": model.l_2.weight.detach().numpy()})
savemat("data/weight3.mat", {"weight3": model.l_3.weight.detach().numpy()})
savemat("data/X.mat", {"X": torch.cat((X1, X2), 1).numpy()})
savemat("data/y.mat", {"y": Y.numpy()})
savemat("data/params.mat",
        {"Tini": Tini, "N": N, "n_basis": n_basis, "nu": nu, "ny": ny})
print("\nSaved weights + params to data/. Next: Terminal_set_quad.m")
