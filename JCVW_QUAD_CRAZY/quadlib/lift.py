"""Observables and the lifted state. Port of phi_nl / lift from KDPC.m.

  z = phi(x_ini) = [x_ini ; phi_nl],   x_ini = col(u_ini, y_ini)

Keeping x_ini on top makes y a sub-vector of z, so C is a selection matrix
rather than something to estimate, and use_phi = False reduces exactly to SPC.

phi_nl, eight terms:

  alpha_T*a                [3]  the bilinear part of alpha_T*b3
  a_x^2, a_y^2, a_x*a_y    [3]  the thrust-projection sag, to second order
  yaw*w_x, yaw*w_y         [2]  body-rate command rotated into the world frame

The first six are derived, not guessed. Since b3 is a unit vector,

  a_z = sqrt(alpha_T^2 - a_x^2 - a_y^2) - mean(alpha_T)
      ~ alpha_T~ - (a_x^2 + a_y^2) / (2*mean(alpha_T)),

so the altitude sag a purely linear predictor has to average over IS a linear
combination of exactly these terms. That is why the benefit lands on z.

SELECTION. An observable earns its place only if it is a product the window
cannot already form. Screen a candidate by regressing it on [x_ini; Uf]: R^2
near 1 means a near-duplicate column, which least squares uses to fit the
identification data while making the closed loop worse. A 22-term dictionary of
sin/cos of the Euler angles predicted 15.9 % better on held-out data and then
diverged in closed loop.
"""

import numpy as np

N_PHI = 8
G = 9.81


def a_of(aT, ang, g=G):
    """a = alpha_T*b3 - g*e3, b3 the third column of R for the ZYX convention.
    aT and ang = [roll, pitch, yaw] are PHYSICAL."""
    r, p, y = ang
    b3 = np.array([np.cos(y)*np.sin(p)*np.cos(r) + np.sin(y)*np.sin(r),
                   np.sin(y)*np.sin(p)*np.cos(r) - np.cos(y)*np.sin(r),
                   np.cos(p)*np.cos(r)])
    return aT * b3 - np.array([0.0, 0.0, g])


def phi_nl(Ubuf, Ybuf, Abuf, par):
    """Ubuf (Tini-1, m) and Ybuf (Tini, p) are MEAN-REMOVED; Abuf (Tini, 3) is
    raw [roll, pitch, yaw] in rad."""
    if not par["use_phi"]:
        return np.zeros(0)

    u1 = Ubuf[-1]
    aT, w = u1[0], u1[1:4]
    yw = Ybuf[-1, 3]

    # offset by the hover trim so the observables are centred
    a = a_of(aT + par["um1"], Abuf[-1], par["g"]) - a_of(par["um1"], np.zeros(3), par["g"])

    return np.concatenate([aT * a,
                           [a[0]**2, a[1]**2, a[0]*a[1], yw*w[0], yw*w[1]]])


def lift(Ubuf, Ybuf, Abuf, par):
    """z = [u_ini ; y_ini ; phi_nl].

    Row-major flattening matches MATLAB's reshape(A.', [], 1).
    """
    return np.concatenate([Ubuf.reshape(-1), Ybuf.reshape(-1),
                           phi_nl(Ubuf, Ybuf, Abuf, par)])


def make_par(cfg, u_mean, Ts):
    """Dimension bookkeeping, eq. (3)."""
    m, p = 4, 4
    Tini = int(cfg["model"]["Tini"])
    use_phi = bool(cfg["model"]["use_phi"])
    par = {
        "Tini": Tini, "N": int(cfg["model"]["N"]), "Ts": float(Ts),
        "m": m, "p": p, "g": G,
        "use_phi": use_phi,
        "n_phi": N_PHI if use_phi else 0,
        "um1": float(u_mean[0]),
    }
    par["n_ini"] = (Tini - 1) * m + Tini * p
    par["L"] = par["n_ini"] + par["n_phi"]
    return par
