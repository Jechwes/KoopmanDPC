"""Acceptance test: does the OSQP port reproduce the YALMIP solution?

The OCP was canonicalised by hand, so the transcription needs checking against
something trusted before it flies. Save a few parameter/solution triples from
MATLAB and compare.

In MATLAB, inside the KDPC.m closed loop:

    sol = ctrl({z_k, r_k});
    cases(end+1) = struct('z0', z_k, 'r', r_k, 'u', sol{1});   %#ok
    ...
    save('qp_cases.mat', 'cases', 'Psi_t', 'Gam_t', 'Qy_phys', 'Ru_phys', ...
         'u_lo', 'u_hi', 'par');

then copy qp_cases.mat here and run

    python verify_qp.py --cases qp_cases.mat

Agreement to ~1e-6 means the port is exact. Anything larger is a transcription
error, not a solver difference.
"""

import argparse

import numpy as np
import scipy.io as sio

from quadlib.spc import SPCController


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="qp_cases.mat")
    ap.add_argument("--tol", type=float, default=1e-5)
    args = ap.parse_args()

    d = sio.loadmat(args.cases, squeeze_me=True, struct_as_record=False)
    par_m = d["par"]
    par = {"N": int(par_m.N), "m": int(par_m.m), "p": int(par_m.p),
           "L": int(par_m.L)}

    ctrl = SPCController(np.atleast_2d(d["Psi_t"]), np.atleast_2d(d["Gam_t"]),
                         par, np.ravel(d["Qy_phys"]), np.ravel(d["Ru_phys"]),
                         np.ravel(d["u_lo"]), np.ravel(d["u_hi"]))

    cases = np.atleast_1d(d["cases"])
    print(f"{len(cases)} cases, L {par['L']}, N {par['N']}\n")
    print(f"{'case':>5} {'max |du|':>12} {'rel':>10}   status")

    worst = 0.0
    for i, c in enumerate(cases):
        u_py, _, status = ctrl.solve_full(np.ravel(c.z0), np.ravel(c.r))
        u_ml = np.ravel(c.u)
        e = np.abs(u_py - u_ml).max()
        rel = e / max(np.abs(u_ml).max(), 1e-12)
        worst = max(worst, e)
        print(f"{i:5d} {e:12.3e} {rel:10.3e}   {status}")

    print(f"\nworst absolute difference: {worst:.3e}")
    if worst < args.tol:
        print("PASS -- the OSQP port matches YALMIP to solver tolerance.")
    else:
        print("FAIL -- check, in this order:")
        print("  1. P is upper triangular (OSQP doubles off-diagonals otherwise)")
        print("  2. the factor of 2 in P and q from the 1/2 x'Px convention")
        print("  3. the equality sign: Gam*u - y = -Psi*z0")
        print("  4. u_lo/u_hi are mean-removed in BOTH places")
        print("  5. eps_abs/eps_rel are tight enough (default 1e-3 is not)")


if __name__ == "__main__":
    main()
