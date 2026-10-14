"""Run the controller over a recorded flight. No drone, no radio.

Feeds the logged Vicon stream through exactly the lift -> solve path fly.py
uses, and reports what the controller WOULD have commanded. Catches sign
errors, unit errors, NaNs, infeasible QPs and absurd inputs on the ground,
where they cost nothing.

It is not a closed-loop simulation: the drone followed whatever flew it at the
time, so the commands will drift from the recorded ones. What you are checking
is that the outputs are finite, inside the actuator box, smooth, and pointed
the same way as the recorded inputs.

    python replay.py --flight data/flights/baseline_*_r00.npz
"""

import argparse
import glob

import numpy as np

from quadlib import reference, spc
from quadlib.lift import lift
from quadlib.util import load_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flight", required=True)
    ap.add_argument("--model", default=None,
                    help="default: theta_spc / theta_kspc as model.use_phi says")
    ap.add_argument("--traj", default=None, help="default: the logged p_ref")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    model = args.model or f"models/theta_{'kspc' if cfg['model']['use_phi'] else 'spc'}.npz"
    ctrl, par, u_mean, y_mean = spc.load(model, cfg)
    Tini, m, p, Ts = par["Tini"], par["m"], par["p"], par["Ts"]

    f = sorted(glob.glob(args.flight))[0]
    d = np.load(f)
    pos, rpy, u_rec = d["pos"], d["rpy"], d["u"]
    n = len(pos)
    print(f"{f}: {n} samples, Ts {float(d['Ts']):.4f} s")

    traj = reference.make(args.traj, cfg["collect"]["hover"]) if args.traj else None

    y0 = np.concatenate([pos[0], [rpy[0, 2]]]) - y_mean
    Ybuf = np.tile(y0, (Tini, 1))
    Abuf = np.tile(rpy[0], (Tini, 1))
    Ubuf = np.zeros((Tini - 1, m))

    U, nfail = [], 0
    for k in range(n):
        y_k = np.concatenate([pos[k], [rpy[k, 2]]]) - y_mean
        Ybuf = np.vstack([Ybuf[1:], y_k])
        Abuf = np.vstack([Abuf[1:], rpy[k]])

        z_k = lift(Ubuf, Ybuf, Abuf, par)
        if traj is not None:
            r_k = reference.horizon(traj, k * Ts, par, y_mean)
        else:
            # hold the logged reference across the horizon
            r_k = np.tile(np.concatenate([d["p_ref"][k], [0.0]]) - y_mean, par["N"])

        u_k, ok = ctrl.step(z_k, r_k)
        nfail += (not ok)
        U.append(u_k + u_mean)
        # replay the RECORDED input, so the window stays on the real trajectory
        Ubuf = np.vstack([Ubuf[1:], u_rec[k] - u_mean])

    U = np.asarray(U)
    lo, hi = ctrl.lo[:m] + u_mean, ctrl.hi[:m] + u_mean       # the QP's own box
    print(f"\nnot solved: {nfail}/{n}")
    print(f"non-finite: {(~np.isfinite(U)).sum()}")
    print("commanded u   min / mean / max        SPC box")
    for i, nm in enumerate(["alpha_T", "wx", "wy", "wz"]):
        print(f"  {nm:8s} {U[:,i].min():8.3f} {U[:,i].mean():8.3f} {U[:,i].max():8.3f}"
              f"    [{lo[i]:.3f}, {hi[i]:.3f}]")

    edge = (U <= lo + 1e-3) | (U >= hi - 1e-3)
    print("on the box edge (% of steps): " + "  ".join(
        f"{nm} {100 * edge[:, i].mean():.0f}" for i, nm in enumerate(["aT", "wx", "wy", "wz"])))

    c = [np.corrcoef(U[:, i], u_rec[:, i])[0, 1] for i in range(4)]
    print("correlation with the recorded input (sign check, not accuracy):")
    print("  " + "  ".join(f"{nm} {v:+.2f}" for nm, v in
                           zip(["aT", "wx", "wy", "wz"], c)))
    if min(c[1:]) < 0:
        print("  NEGATIVE on a rate channel -- check the TX signs in config.yaml")


if __name__ == "__main__":
    main()
