"""Does the Python port reproduce the MATLAB reference? No drone, no MATLAB.

Reads what KDPC.m already writes -- data/*.mat, KDPCTheta.mat, KDPC.mat -- and
checks four things in increasing scope:

  1. identification   Python hankel + pinv  vs  MATLAB Theta
  2. plant            quadlib/sim.py driven by MATLAB's u_log  vs  MATLAB x_log
  3. QP, per step     every QP MATLAB solved, re-solved with OSQP
                      vs  YALMIP/QUADPROG, and vs an exact active-set solver
  4. closed loop      Python identify + lift + OSQP + plant  vs  MATLAB run

Check 3 rebuilds each step's z(k) and r(k) from MATLAB's own logged state, so
every QP is the one YALMIP actually saw.

    python verify_matlab.py --matlab ../JCVW_QUAD

Needs KDPC.m run once with the analytic observables. Optional: pip install
quadprog, for the exact reference in check 3.
"""

import argparse
import os
import sys
import time

import numpy as np
import scipy.io as sio

from quadlib import hankel
from quadlib.lift import lift
from quadlib.sim import quad_params, quad_plant
from quadlib.spc import SPCController


def loadm(d, f):
    return sio.loadmat(os.path.join(d, "data", f), squeeze_me=True,
                       struct_as_record=False)


def measure(x, yaw, y_mean):
    """measure() in KDPC.m. MATLAB state: [p v q(w,x,y,z) om]."""
    q = x[6:10]
    roll = np.arctan2(2*(q[0]*q[1] + q[2]*q[3]), 1 - 2*(q[1]**2 + q[2]**2))
    pitch = np.arcsin(np.clip(2*(q[0]*q[2] - q[3]*q[1]), -1, 1))
    return np.r_[x[:3], yaw] - y_mean, np.r_[roll, pitch, yaw]


def yaw_of(x):
    q = x[6:10]
    return np.arctan2(2*(q[0]*q[3] + q[1]*q[2]), 1 - 2*(q[2]**2 + q[3]**2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matlab", default="../JCVW_QUAD")
    ap.add_argument("--Qy", type=float, nargs=4, default=[10, 10, 10, 1])
    ap.add_argument("--Ru", type=float, nargs=4, default=[0.1, 2, 2, 2])
    ap.add_argument("--fig8", type=float, nargs=4, default=[1.2, 0.8, 0.0, 1.0],
                    metavar=("A", "B", "C", "w"), help="the A_sp B_sp C_sp w_sp KDPC.m used")
    args = ap.parse_args()
    D = args.matlab

    u = loadm(D, "u_data.mat")["u_data"]; y = loadm(D, "y_data.mat")["y_data"]
    att = loadm(D, "att_data.mat")["att_data"]
    lens = loadm(D, "run_lengths.mat")["run_lengths"]
    sc = loadm(D, "scalers.mat"); T = loadm(D, "KDPCTheta.mat"); K = loadm(D, "KDPC.mat")
    pm = T["par"]
    if getattr(pm, "phi_src", "analytic") != "analytic":
        sys.exit("KDPCTheta.mat was made with learned observables; rerun KDPC.m analytic")

    par = dict(Tini=int(pm.Tini), N=int(pm.N), Ts=float(pm.Ts), m=4, p=4,
               g=float(pm.g), use_phi=bool(pm.use_phi), n_phi=int(pm.n_phi),
               um1=float(pm.um1), n_ini=int(pm.n_ini), L=int(pm.L))
    u_mean, y_mean = sc["u_mean"], sc["y_mean"]
    Tini, N, m, Ts, g = par["Tini"], par["N"], 4, par["Ts"], par["g"]
    x_ml, u_ml, p_ref = K["x_log"], K["u_log"].T, K["p_ref"]
    n = len(u_ml)
    ok = True
    print(f"MATLAB run: Tini {Tini}, N {N}, Ts {Ts}, L {par['L']}, {n} steps\n")

    # ---------------------------------------------------------------- 1
    Zp, Uf, Yf = hankel.build(u, y, att, lens, par)
    Psi, Gam = hankel.fit(Zp, Uf, Yf, par)
    e1 = np.abs(np.hstack([Psi, Gam]) - T["Theta"]).max() / np.abs(T["Theta"]).max()
    print(f"1. identification   {Zp.shape[1]} windows, Theta relative error {e1:.1e}")
    ok &= e1 < 1e-8

    # ---------------------------------------------------------------- 2
    prm, lim, inner = quad_params()
    x, st, e2 = x_ml[0].copy(), None, 0.0
    for i in range(n):
        x, st = quad_plant(x, u_ml[i] + u_mean, prm, lim, inner, Ts, 0.002, st)
        e2 = max(e2, np.abs(x - x_ml[i + 1]).max())
    print(f"2. plant            max state error over {n} steps {e2:.1e}")
    ok &= e2 < 1e-8

    # ---------------------------------------------------------------- 3
    A, B, C, w = args.fig8
    ph = sc["p_hover"]
    r_pos = lambda t: ph + np.array([A*np.sin(w*t), B*np.sin(2*w*t), C*np.sin(w*t)])
    e_ref = max(np.abs(r_pos(k*Ts) - p_ref[k]).max() for k in range(len(p_ref)))
    if e_ref > 1e-9:
        sys.exit(f"--fig8 does not reproduce KDPC.mat's p_ref (err {e_ref:.1e}); "
                 f"pass the A_sp B_sp C_sp w_sp that run used")

    u_lo = np.r_[lim["T_spec"][0], -3, -3, -3] - u_mean
    u_hi = np.r_[lim["T_spec"][1], 3, 3, 3] - u_mean
    ctrl = SPCController(T["Psi_t"], T["Gam_t"], par, args.Qy, args.Ru, u_lo, u_hi)
    try:
        import quadprog
        Qb = np.kron(np.eye(N), np.diag(args.Qy)); Rb = np.kron(np.eye(N), np.diag(args.Ru))
        H = T["Gam_t"].T @ Qb @ T["Gam_t"] + Rb
        Cq = np.hstack([np.eye(N*m), -np.eye(N*m)])
        bq = np.r_[np.tile(u_lo, N), -np.tile(u_hi, N)]
        exact = lambda z, r: quadprog.solve_qp(
            H, -(T["Gam_t"].T @ Qb @ (T["Psi_t"] @ z - r)), Cq, bq)[0][:m]
    except ImportError:
        exact = None

    def r_at(t):
        return np.concatenate([np.r_[r_pos(t + j*Ts), 0.0] - y_mean
                               for j in range(1, N + 1)])

    yaw = 0.0
    y0, a0 = measure(x_ml[0], yaw, y_mean)
    Yb, Ab, Ub = np.tile(y0, (Tini, 1)), np.tile(a0, (Tini, 1)), np.zeros((Tini-1, m))
    e3, e3x, ms = [], [], []
    for i in range(n):
        z, r = lift(Ub, Yb, Ab, par), r_at(i * Ts)
        ts = time.perf_counter(); uo, _ = ctrl.step(z, r); ms.append(time.perf_counter() - ts)
        e3.append(np.abs(uo - u_ml[i]).max())
        if exact is not None:
            e3x.append(np.abs(uo - exact(z, r)).max())
        x = x_ml[i + 1]
        yaw = yaw + np.arctan2(np.sin(yaw_of(x) - yaw), np.cos(yaw_of(x) - yaw))
        yk, ak = measure(x, yaw, y_mean)
        Yb, Ab = np.vstack([Yb[1:], yk]), np.vstack([Ab[1:], ak])
        Ub = np.vstack([Ub[1:], u_ml[i]])
    e3 = np.array(e3)
    print(f"3. QP per step      OSQP vs YALMIP: max {e3.max():.1e}, median "
          f"{np.median(e3):.1e}  (|u| up to {np.abs(u_ml).max():.2f})")
    if e3x:
        print(f"                    OSQP vs exact:  max {max(e3x):.1e}"
              f"   -> the residual vs YALMIP is QUADPROG's tolerance, not OSQP's")
    print(f"                    OSQP {1e3*np.mean(ms):.2f} ms mean, "
          f"{ctrl.n_fail} not solved")
    ok &= e3.max() < 1e-4 and ctrl.n_fail == 0

    # ---------------------------------------------------------------- 4
    ctrl = SPCController(Psi, Gam, par, args.Qy, args.Ru, u_lo, u_hi)   # Python Theta
    x, st, yaw = x_ml[0].copy(), None, 0.0
    y0, a0 = measure(x, yaw, y_mean)
    Yb, Ab, Ub = np.tile(y0, (Tini, 1)), np.tile(a0, (Tini, 1)), np.zeros((Tini-1, m))
    X = [x.copy()]
    for i in range(n):
        uo, _ = ctrl.step(lift(Ub, Yb, Ab, par), r_at(i * Ts))
        x, st = quad_plant(x, uo + u_mean, prm, lim, inner, Ts, 0.002, st)
        yaw = yaw + np.arctan2(np.sin(yaw_of(x) - yaw), np.cos(yaw_of(x) - yaw))
        yk, ak = measure(x, yaw, y_mean)
        Yb, Ab = np.vstack([Yb[1:], yk]), np.vstack([Ab[1:], ak])
        Ub = np.vstack([Ub[1:], uo])
        X.append(x.copy())
    X = np.array(X)

    def rmse(xl):
        e = xl[:, :3] - p_ref
        return np.r_[np.sqrt((e**2).mean(0)), np.sqrt((e**2).sum(1).mean())]
    rm, rp = rmse(x_ml), rmse(X)
    gap = np.linalg.norm(X[:, :3] - x_ml[:, :3], axis=1).max()
    print("4. closed loop       RMSE      x        y        z        3D")
    print("                     MATLAB  " + "  ".join(f"{v:.5f}" for v in rm))
    print("                     Python  " + "  ".join(f"{v:.5f}" for v in rp))
    print(f"                    max trajectory gap {gap:.1e} m over {n*Ts:.0f} s")
    ok &= gap < 1e-4

    print("\nPASS -- the Python pipeline reproduces the MATLAB reference" if ok
          else "\nFAIL -- see above")


if __name__ == "__main__":
    main()
