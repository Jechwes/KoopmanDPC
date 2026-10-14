"""Frame check -- MOTORS STAY OFF. Hold the drone and rotate it by hand.

The TX signs in config.yaml are derived from the firmware source and assume the
Vicon rigid body's frame is the Crazyflie body frame (x forward, y left, z up).
That alignment is defined in Vicon Tracker, not in any code, so it has to be
measured. This compares the onboard gyro with body rates derived from the
Vicon rotation:

    R(t+h) = R(t-h) exp([w] 2h)   ->   w_body = log(R(t-h)' R(t+h)) / 2h

Pass = each Vicon axis correlates > 0.8 with the SAME gyro axis, positively.
Then set convert.frame_verified: true in config.yaml.

The Vicon orientation is resampled on a 10 ms grid and differenced over
+/-25 ms rather than frame to frame: frames reach the laptop in bursts, and a
frame-to-frame difference divides each rotation by the gap between ARRIVAL
times, so a burst (two frames 1 ms apart) becomes a 10x spike in the rate.
Grid points next to a tracking gap or a Vicon glitch are left out.

Every run saves its raw recording to logs/; --replay re-runs the analysis on
one without the drone.

    python check_frame.py            # ~20 s: roll, pitch, yaw, a few times each
    python check_frame.py --replay logs/check_frame_<stamp>.npz
"""

import argparse
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from quadlib.util import load_cfg, rot_from_rpy

AX = "xyz"
GRID = 0.01         # s
HALF = 0.025        # s, half the differencing baseline
W_MAX = 15.0        # rad/s, faster than a hand turns a drone: a Vicon glitch
GAP = 0.05          # s, a tracking gap that no differencing window may span
LAGS = range(-15, 16)   # grid steps, +/-150 ms of radio vs Vicon latency


def record(cfg, duration):
    """Vicon pose and onboard gyro, both stamped with laptop arrival time."""
    from quadlib.link import Link
    from quadlib.vicon import ViconThread

    with Link(cfg) as link:
        vic = ViconThread(cfg["vicon"]["hostname"], cfg["vicon"]["object_name"],
                          cf=link.cf)
        vic.start()
        if not vic.wait_for_pose():
            raise RuntimeError("no Vicon pose")

        tv, rpy, tg, gyro = [], [], [], []
        last = None
        t0 = time.time()
        while time.time() - t0 < duration:
            pose = vic.latest()
            if pose[0] != last:
                last = pose[0]
                tv.append(pose[0]); rpy.append(pose[2])
            tg.append(link.telemetry["t"]); gyro.append(link.telemetry["gyro"].copy())
            time.sleep(0.005)
        vic.stop()

    rec = dict(tv=np.array(tv), rpy=np.array(rpy), tg=np.array(tg), gyro=np.array(gyro))
    os.makedirs("logs", exist_ok=True)
    path = f"logs/check_frame_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(path, **rec)
    print(f"  recording saved -> {path}")
    return rec


def best_corr(a, b, ok):
    """Correlation of a with b at the lag that maximises |r|, over ok points of a.
    A positive lag means a (Vicon) trails b (gyro)."""
    best, best_lag, n = 0.0, 0, len(a)
    for L in LAGS:
        x, y, m = (a[L:], b[:n - L], ok[L:]) if L >= 0 else (a[:L], b[-L:], ok[:L])
        if m.sum() < 50 or x[m].std() == 0 or y[m].std() == 0:
            continue
        c = np.corrcoef(x[m], y[m])[0, 1]
        if abs(c) > abs(best):
            best, best_lag = c, L
    return best, best_lag


def analyze(rec):
    """Returns the 3x3 correlation matrix (Vicon rows, gyro columns), the lag of
    each pair in grid steps, the Vicon rate std per axis and the kept fraction."""
    tv, rpy, tg, G = rec["tv"], rec["rpy"], rec["tg"], rec["gyro"]
    k = np.r_[True, np.diff(tv) > 0]
    tv, rpy = tv[k], rpy[k]
    k = np.r_[True, np.diff(tg) > 0]                 # one gyro sample per log packet
    tg, G = tg[k], G[k]

    dt = np.diff(tv)
    print(f"  vicon: {len(tv)} frames, {1 / np.median(dt):.0f} Hz; arrival gaps median "
          f"{1e3 * np.median(dt):.1f} ms, {100 * np.mean(dt < 0.3 * np.median(dt)):.0f} % "
          f"bunched (< 0.3 x median), longest {1e3 * dt.max():.0f} ms")
    print(f"  gyro : {len(tg)} log packets, {1 / np.median(np.diff(tg)):.0f} Hz")

    grid = np.arange(max(tv[0], tg[0]) + 0.2, min(tv[-1], tg[-1]) - 0.2, GRID)
    s = Slerp(tv, Rotation.from_matrix(np.array([rot_from_rpy(a) for a in rpy])))
    V = (s(grid - HALF).inv() * s(grid + HALF)).as_rotvec() / (2 * HALF)
    ok = np.linalg.norm(V, axis=1) < W_MAX
    for i in np.flatnonzero(dt > GAP):
        ok &= ~((grid + HALF > tv[i]) & (grid - HALF < tv[i + 1]))
    print(f"  kept {100 * ok.mean():.0f} % of the grid "
          f"({(dt > GAP).sum()} tracking gaps > {1e3 * GAP:.0f} ms, "
          f"{(np.linalg.norm(V, axis=1) >= W_MAX).sum()} glitch points)")

    Gy = np.column_stack([np.interp(grid, tg, G[:, i]) for i in range(3)])
    C, L = np.zeros((3, 3)), np.zeros((3, 3), dtype=int)
    for i in range(3):
        for j in range(3):
            C[i, j], L[i, j] = best_corr(V[:, i], Gy[:, j], ok)
    sd = np.array([V[ok, i].std() for i in range(3)])
    return C, L, sd, ok.mean()


def verdict(C, L, sd):
    print("\ncorrelation, Vicon body rate (rows) vs gyro (cols):")
    print("         gyro.x  gyro.y  gyro.z")
    for i in range(3):
        print(f"  vicon.{AX[i]}  " + "  ".join(f"{C[i,j]:+.2f}" for j in range(3)))

    ok = True
    for i in range(3):
        j = int(np.argmax(np.abs(C[i])))
        if sd[i] < 0.5:
            print(f"  {AX[i]}: not enough rotation (std {sd[i]:.2f} rad/s) -- rotate more")
            ok = False
        elif j != i or C[i, j] < 0.8:
            print(f"  {AX[i]}: FAIL -- Vicon {AX[i]} matches gyro.{AX[j]} with "
                  f"{C[i,j]:+.2f}. The Vicon object frame is not the drone's body "
                  f"frame; fix the object orientation in Vicon Tracker.")
            ok = False
        else:
            print(f"  {AX[i]}: ok ({C[i,i]:+.2f}, Vicon trails the gyro by "
                  f"{1e3 * GRID * L[i, i]:.0f} ms)")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--replay", default=None, help="analyze a saved recording, no drone")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    if args.replay:
        rec = dict(np.load(args.replay))
    else:
        cfg = load_cfg(args.config)
        print("\n  MOTORS STAY OFF -- this script never arms.")
        print("  Rotate the drone by hand: tip it right-side-down and back (roll),")
        print("  nose-down and back (pitch), turn it left and right (yaw). Repeat.\n")
        rec = record(cfg, args.duration)

    ok = verdict(*analyze(rec)[:3])
    print("\nPASS -- set convert.frame_verified: true in config.yaml" if ok
          else "\nNOT VERIFIED -- do not fly")


if __name__ == "__main__":
    main()
