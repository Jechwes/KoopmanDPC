"""Eq. (6) data matrices. Port of hankel_koopman.m.

  Zp(:,c) = z(k)                       1 deep
  Uf(:,c) = col(u(k)   .. u(k+N-1))    N deep
  Yf(:,c) = col(y(k+1) .. y(k+N))      N deep

Up and Yp are not built separately: with phi keeping x_ini on top they are the
first rows of Zp. Zf is not built, this fits eq. (7) not eq. (8)  (D1).

CAUSAL ALIGNMENT. This assumes the logger recorded y(k) BEFORE the input u(k)
was sent, so that u(k) drives y(k+1). The flight loop in fly.py and collect.py
reads Vicon, then computes, then transmits, then stores, which produces exactly
that ordering. Logging the pose after transmitting instead would put u(k) and
y(k) in the same row and leave the predictor one step ahead of anything a
controller can actually do.

Windows never span two flights (D3).
"""

import numpy as np

from .lift import lift


def build(u, y, att, lens, par):
    """u (S,m), y (S,p) mean-removed; att (S,3) raw. lens: samples per flight."""
    Tini, N, m, p = par["Tini"], par["N"], par["m"], par["p"]
    lens = np.asarray(lens, dtype=int).ravel()

    Zp, Uf, Yf = [], [], []
    start = 0
    for L in lens:
        uu, yy, aa = u[start:start+L], y[start:start+L], att[start:start+L]
        k = np.arange(Tini, L - N)                 # 0-based; MATLAB k-1
        for k0 in k:
            Zp.append(lift(uu[k0-Tini+1:k0], yy[k0-Tini+1:k0+1],
                           aa[k0-Tini+1:k0+1], par))
            Uf.append(uu[k0:k0+N].reshape(-1))
            Yf.append(yy[k0+1:k0+N+1].reshape(-1))
        start += L

    assert Zp, f"every flight is shorter than Tini+N = {Tini+N}"
    # columns are windows, matching the MATLAB orientation
    return (np.asarray(Zp).T, np.asarray(Uf).T, np.asarray(Yf).T)


def fit(Zp, Uf, Yf, par):
    """Theta = Yf * pinv([Zp; Uf]), eq. (7) with phi fixed. Returns Psi, Gam."""
    X = np.vstack([Zp, Uf])
    need = par["L"] + par["N"] * par["m"]
    assert X.shape[1] >= need, \
        f"{X.shape[1]} columns, need >= {need} for full row rank"
    Theta = Yf @ np.linalg.pinv(X)
    return Theta[:, :par["L"]], Theta[:, par["L"]:]
