"""The OCP, eq. (10), canonicalised for OSQP.

This is the YALMIP problem from KDPC.m with nothing eliminated:

    min (y - r)' Q (y - r) + u' R u
    s.t. y = Psi z0 + Gam u
         u_lo <= u <= u_hi

YALMIP's optimizer() does the canonicalisation and the parametric patching
internally. Here it is done once, by hand, so there is no modelling layer at
runtime. OSQP wants

    min 1/2 x' P x + q' x   s.t.   l <= A x <= u

Stack x = [u ; y], expand (y-r)'Q(y-r) = y'Qy - 2r'Qy + const, and move the
equality's variables to the left:

    P = blkdiag(2R, 2Q)           q = [0 ; -2 Q r]
    A = [[Gam, -I], [I, 0]]       l = [-Psi z0 ; u_lo]   u = [-Psi z0 ; u_hi]

P and A depend only on the weights and Gam, so the KKT factorisation is computed
once by setup() and reused. Every control step touches only q and the first
block of l/u, and warm-starts from the previous solution.
"""

import numpy as np
import osqp
import scipy.sparse as sp


class SPCController:
    def __init__(self, Psi, Gam, par, Qy, Ru, u_lo, u_hi,
                 eps=1e-6, verbose=False):
        self.Psi, self.Gam, self.par = Psi, Gam, par
        N, m, p = par["N"], par["m"], par["p"]
        self.nu, self.ny, self.m = N * m, N * p, m

        self.Q = sp.kron(sp.eye(N), sp.diags(np.asarray(Qy, dtype=float)))
        R = sp.kron(sp.eye(N), sp.diags(np.asarray(Ru, dtype=float)))

        # P must be upper triangular: OSQP doubles the off-diagonals otherwise
        P = sp.triu(sp.block_diag([2 * R, 2 * self.Q]), format="csc")
        A = sp.vstack([
            sp.hstack([sp.csc_matrix(Gam), -sp.eye(self.ny)]),      # equality
            sp.hstack([sp.eye(self.nu), sp.csc_matrix((self.nu, self.ny))]),
        ], format="csc")

        self.lo = np.tile(np.asarray(u_lo, dtype=float), N)
        self.hi = np.tile(np.asarray(u_hi, dtype=float), N)

        self.prob = osqp.OSQP()
        self.prob.setup(P, np.zeros(self.nu + self.ny), A,
                        np.concatenate([np.zeros(self.ny), self.lo]),
                        np.concatenate([np.zeros(self.ny), self.hi]),
                        warm_starting=True, polishing=True, verbose=verbose,
                        eps_abs=eps, eps_rel=eps)

        self.u_prev = np.zeros(m)
        self.n_fail = 0

    def step(self, z0, r):
        """z0 = phi(x_ini(k)); r = mean-removed reference over the horizon.

        Returns (u(k), ok). On a non-'solved' status the previous input is held
        rather than flying an unconverged iterate.
        """
        b = -self.Psi @ z0
        self.prob.update(
            q=np.concatenate([np.zeros(self.nu), -2.0 * (self.Q @ r)]),
            l=np.concatenate([b, self.lo]),
            u=np.concatenate([b, self.hi]))
        res = self.prob.solve()

        if res.info.status != "solved":
            self.n_fail += 1
            return self.u_prev.copy(), False

        self.u_prev = res.x[:self.m].copy()
        return self.u_prev.copy(), True

    def solve_full(self, z0, r):
        """Whole horizon, for verify_qp.py. Returns (u, y, status)."""
        b = -self.Psi @ z0
        self.prob.update(
            q=np.concatenate([np.zeros(self.nu), -2.0 * (self.Q @ r)]),
            l=np.concatenate([b, self.lo]),
            u=np.concatenate([b, self.hi]))
        res = self.prob.solve()
        return res.x[:self.nu], res.x[self.nu:], res.info.status


def input_box(box):
    """[alpha_T, wx, wy, wz] bounds from a config section with alpha_T, omega."""
    w = float(box["omega"])
    return (np.array([box["alpha_T"][0], -w, -w, -w], dtype=float),
            np.array([box["alpha_T"][1], w, w, w], dtype=float))


def load(path, cfg):
    """Build a controller from models/theta_spc.npz or theta_kspc.npz.

    The input box is spc_box when given -- its thrust band taken around the
    identified hover trim u_mean[0], as the thesis' box is around hover, so it
    stays centred when thrust_max_N is off -- intersected with the actuator clip
    in limits: an input the QP plans outside the clip would be cut on the wire
    while the input window records it as flown.
    """
    d = np.load(path, allow_pickle=True)
    par = d["par"].item()
    u_mean = d["u_mean"]

    lo, hi = input_box(cfg["limits"])
    if "spc_box" in cfg:
        b = cfg["spc_box"]
        w = float(b["omega"])
        lo = np.maximum(lo, np.r_[b["alpha_T_rel"][0] * u_mean[0], -w, -w, -w])
        hi = np.minimum(hi, np.r_[b["alpha_T_rel"][1] * u_mean[0], w, w, w])

    ctrl = SPCController(d["Psi"], d["Gam"], par,
                         cfg["weights"]["Qy"], cfg["weights"]["Ru"],
                         lo - u_mean, hi - u_mean,
                         eps=float(cfg.get("solver", {}).get("osqp_eps", 1e-6)))
    return ctrl, par, u_mean, d["y_mean"]
