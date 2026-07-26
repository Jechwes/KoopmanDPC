# model_learning_tanh.py -- MIMO version of KDPC_RSRFG/Pendulum/model_learning_tanh.py
#
# Learns the observables phi by minimising the MULTI-STEP output prediction
# error (paper problem (7)):
#
#     y(k+1..k+N)  ~  W3 * [ phi(u_ini, y_ini) ; u(k..k+N-1) ]
#
# Same network as the reference (two tanh layers + linear readout, no biases),
# only the dimensions change: nu = 4 motors, ny = 4 outputs.

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
from scipy.io import savemat

# ---- hyperparameters (reference used Tini=3, n_basis=10, N=15 for 1 state) --
Tini = 10
N = 20
n_basis = 40
epochs = 3000
torch.manual_seed(41)

u_data = scipy.io.loadmat("data/u_data.mat")["u_data"]
y_data = scipy.io.loadmat("data/y_data.mat")["y_data"]
u_test = scipy.io.loadmat("data/u_data_test.mat")["u_data_test"]
y_test = scipy.io.loadmat("data/y_data_test.mat")["y_data_test"]
rl = scipy.io.loadmat("data/run_lengths.mat")

nu, ny = u_data.shape[1], y_data.shape[1]
in1 = (Tini - 1) * nu + Tini * ny        # width of [u_ini; y_ini]
in2 = n_basis + N * nu                   # width of [z; u_f]
out = N * ny                             # stacked future outputs
print(f"nu={nu} ny={ny} Tini={Tini} N={N} n_basis={n_basis}")


def build(u, y, lens):
    """Hankel windows, eq. (6). Never crosses a run boundary.
    Flattening is TIME-MAJOR and must match tanh_nn.m / Terminal_set.m."""
    X1, X2, Y = [], [], []
    s = 0
    for L in lens:
        uu, yy = u[s : s + L], y[s : s + L]
        for i in range(L - Tini - N):
            X1.append(np.r_[uu[i + 1 : i + Tini].ravel(), yy[i + 1 : i + Tini + 1].ravel()])
            X2.append(uu[i + Tini : i + Tini + N].ravel())
            Y.append(yy[i + Tini + 1 : i + Tini + 1 + N].ravel())
        s += L
    return [torch.FloatTensor(np.array(a)) for a in (X1, X2, Y)]


X1, X2, Y = build(u_data, y_data, rl["run_lengths"].ravel().astype(int))
X1t, X2t, Yt = build(u_test, y_test, rl["run_lengths_test"].ravel().astype(int))
print(f"regressors: X1 {tuple(X1.shape)}, X2 {tuple(X2.shape)}, Y {tuple(Y.shape)}")


class Model(nn.Module):
    """Two tanh layers + linear readout.

    BIASES ARE ON, unlike the reference. Without them tanh(Wx) is an ODD
    function, phi(-x) = -phi(x), so the whole predictor is forced to be odd
    about the operating point. That is exactly right for the pendulum, whose
    only nonlinearity is sin(x2). It is wrong for a quadrotor: rotation-matrix
    products, one-sided thrust and Omega^2 have no such symmetry. Measured on
    this data, y(-x) vs -y(x) shows a 66% relative mismatch -- a component a
    bias-free network cannot represent at any n_basis.

    Consequence handled in tanh_nn.m: phi(0) != 0 now, so the observables are
    shifted by phi(0) to restore Assumption 2.4 of the paper.
    """

    def __init__(self):
        super().__init__()
        self.l_1 = nn.Linear(in1, n_basis, bias=True)
        self.l_2 = nn.Linear(n_basis, n_basis, bias=True)
        self.l_3 = nn.Linear(in2, out, bias=True)

    def forward(self, x1, x2):
        z = torch.tanh(self.l_2(torch.tanh(self.l_1(x1))))
        return self.l_3(torch.cat((z, x2), 1))


model = Model()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

losses = []
for i in range(epochs):
    loss = criterion(model(X1, X2), Y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    losses.append(loss.item())
    if i % 200 == 0:
        with torch.no_grad():
            print(f"epoch {i:5d} | train {loss.item():.6f} | test {criterion(model(X1t, X2t), Yt).item():.6f}")

# ---- R^2 on the test set, per channel and per prediction step --------------
with torch.no_grad():
    Yp = model(X1t, X2t).numpy()
Ytn = Yt.numpy()
r2 = lambda a, b: 1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)

print("\ntest R^2 per output channel:")
for c, lab in enumerate(["x", "y", "z", "yaw"]):
    idx = np.arange(c, out, ny)
    print(f"  {lab:>4}: {r2(Ytn[:, idx], Yp[:, idx]):.4f}")
print("test R^2 per prediction step:")
for j in [1, 5, 10, N]:
    idx = np.arange((j - 1) * ny, j * ny)
    print(f"  step {j:2d}: {r2(Ytn[:, idx], Yp[:, idx]):.4f}")

# Export weights AND biases. The old script saved only `.weight`, so MATLAB
# evaluated tanh(W2*tanh(W1*x)) -- a different function from the trained one.
nz = lambda t: t.detach().numpy()
savemat("data/weight1.mat", {"weight1": nz(model.l_1.weight), "bias1": nz(model.l_1.bias)})
savemat("data/weight2.mat", {"weight2": nz(model.l_2.weight), "bias2": nz(model.l_2.bias)})
savemat("data/weight3.mat", {"weight3": nz(model.l_3.weight), "bias3": nz(model.l_3.bias)})
savemat("data/params.mat", {"Tini": Tini, "N": N, "n_basis": n_basis, "nu": nu, "ny": ny})

with torch.no_grad():
    z_eq = torch.tanh(model.l_2(torch.tanh(model.l_1(torch.zeros(1, in1))))).numpy().ravel()
print(f"\n||phi(0)|| = {np.linalg.norm(z_eq):.4f}  (subtracted in tanh_nn.m so phi~(0) = 0)")

plt.semilogy(losses); plt.xlabel("epoch"); plt.ylabel("MSE"); plt.grid(True); plt.show()
print("\nsaved weights to data/. Next: Terminal_set.m")
