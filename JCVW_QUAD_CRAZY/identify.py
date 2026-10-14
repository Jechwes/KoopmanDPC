"""Fit the SPC prediction matrices from collected flights. No drone needed.

    Theta = Yf * pinv([Zp; Uf])      eq. (7) with phi fixed
    Psi   = Theta(:, 1:L)            Gam = Theta(:, L+1:end)

Always fits SPC; with model.use_phi also KSPC, from the same flights and the
same mean removal, so the ablation is one flag in fly.py:
    models/theta_kspc.npz   z = [x_ini ; phi_nl]
    models/theta_spc.npz    z = x_ini            (phi = identity)

Each flight is one run for the Hankel builder, so windows never span a landing
(D3). Data is mean-removed because the predictor has no intercept and a
quadrotor hovers at alpha_T ~ 9.81, not 0.

    python identify.py --flights "data/flights/baseline_*.npz"
    python identify.py --tini 10 --outdir models/tini10    # a setting from sweep_tini.py
"""

import argparse
import copy
import glob
import os

import numpy as np

from quadlib import hankel
from quadlib.lift import make_par
from quadlib.util import load_cfg


def load_flights(pattern, overrun=1.25):
    """Load flights and cut them into segments at every loop overrun.

    The Hankel matrices assume every step is exactly Ts. A step that took
    longer -- scheduler hiccup, radio stall, GC pause -- corrupts every window
    that spans it, and with Tini + N = 23 a single overrun spoils 23 windows. So
    each overrun is treated like a landing (D3): a segment boundary no window
    may cross. Measured in the threaded simulator, 2.5 % overruns touched 57 %
    of the windows and the resulting model could not fly.

    Steps much SHORTER than Ts are cut too. The flight loops now restart their
    grid after an overrun, but flights logged before that caught up with
    back-to-back steps a few ms apart, which are just as wrong for the model.
    """
    files = sorted(glob.glob(pattern))
    assert files, f"no flights matched {pattern}"
    Ts = float(np.load(files[0])["Ts"])
    u, y, att, lens, dts = [], [], [], [], []
    n_cut = 0
    for f in files:
        d = np.load(f)
        dt = np.diff(d["t"])
        dts.append(dt)
        bad = (dt > overrun * Ts) | (dt < Ts / overrun)
        cuts = np.flatnonzero(bad) + 1                        # first index after a gap
        n_cut += len(cuts)
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(d["u"])]):
            u.append(d["u"][a:b])
            y.append(np.column_stack([d["pos"][a:b], d["rpy"][a:b, 2]]))  # x y z yaw
            att.append(d["rpy"][a:b])                                     # raw
            lens.append(b - a)
        print(f"  {os.path.basename(f)}: {len(d['u'])} samples, "
              f"{len(cuts)} steps outside [{1/overrun:.2f}, {overrun:.2f}] Ts")
    print(f"  -> {len(lens)} segments from {len(files)} flights ({n_cut} cuts)")
    return (np.vstack(u), np.vstack(y), np.vstack(att), np.array(lens), Ts,
            np.concatenate(dts))


def speed_p99(y, lens, Ts):
    """99th percentile of the horizontal and vertical speed in the data, from
    central differences of position within each segment. fly.py warns when a
    reference asks for more: the model has hardly seen the drone that fast."""
    vh, vz, s = [], [], 0
    for L in lens:
        p = y[s:s + L, :3]
        s += L
        if L < 3:
            continue
        v = (p[2:] - p[:-2]) / (2 * Ts)
        vh.append(np.hypot(v[:, 0], v[:, 1]))
        vz.append(np.abs(v[:, 2]))
    return np.percentile(np.concatenate(vh), 99), np.percentile(np.concatenate(vz), 99)


def r2_vs_window(Zp, Uf, par):
    """R^2 of each observable against what [x_ini; Uf] can already form.
    Near 1 means a near-duplicate column: least squares will use it to fit the
    identification data while making the closed loop worse."""
    W = np.vstack([Zp[:par["n_ini"]], Uf])
    Wc = W - W.mean(1, keepdims=True)
    P = np.linalg.pinv(Wc)
    out = []
    for i in range(par["n_phi"]):
        o = Zp[par["n_ini"] + i] - Zp[par["n_ini"] + i].mean()
        e = o - (o @ P) @ Wc
        out.append(1 - e @ e / max(o @ o, 1e-12))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flights", default="data/flights/*.npz")
    ap.add_argument("--test", default=None, help="held-out pattern")
    ap.add_argument("--outdir", default="models")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tini", type=int, default=None, help="override model.Tini")
    ap.add_argument("--n", type=int, default=None, help="override model.N")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.tini:
        cfg["model"]["Tini"] = args.tini
    if args.n:
        cfg["model"]["N"] = args.n
    u_raw, y_raw, att, lens, Ts, dt = load_flights(args.flights)

    # the model is only valid at the period the loop actually ran at
    med = float(np.median(dt))
    print(f"\nnominal Ts {1e3*Ts:.1f} ms, achieved median {1e3*med:.1f} ms, "
          f"p99 {1e3*np.percentile(dt, 99):.1f} ms")
    if abs(med - Ts) > 0.05 * Ts:
        print("  WARNING: achieved period is >5% off nominal. Theta will plan for a "
              "cadence the loop does not deliver. Fix the loop rate and re-collect.")

    u_mean, y_mean = u_raw.mean(0), y_raw.mean(0)
    u, y = u_raw - u_mean, y_raw - y_mean
    tilt = np.degrees(np.arccos(np.clip(np.cos(att[:, 0]) * np.cos(att[:, 1]), -1, 1)))
    print(f"tilt: median {np.median(tilt):.1f}, p90 {np.percentile(tilt, 90):.1f}, "
          f"max {tilt.max():.1f} deg")
    vh_p99, vz_p99 = speed_p99(y_raw, lens, Ts)
    print(f"speed p99: horizontal {vh_p99:.2f} m/s, vertical {vz_p99:.2f} m/s")
    print(f"u std: {np.array2string(u.std(0), precision=3)}")

    os.makedirs(args.outdir, exist_ok=True)
    fits = [(True, "kspc"), (False, "spc")] if cfg["model"]["use_phi"] else [(False, "spc")]
    for use_phi, name in fits:
        c = copy.deepcopy(cfg)
        c["model"]["use_phi"] = use_phi
        par = make_par(c, u_mean, Ts)

        Zp, Uf, Yf = hankel.build(u, y, att, lens, par)
        Psi, Gam = hankel.fit(Zp, Uf, Yf, par)
        print(f"\n[{name}] L {par['L']} | columns {Zp.shape[1]} | "
              f"cond([Zp;Uf]) {np.linalg.cond(np.vstack([Zp, Uf])):.2e}")

        if par["n_phi"]:
            r2 = r2_vs_window(Zp, Uf, par)
            print(f"  observable R^2 vs window: max {r2.max():.3f}, "
                  f"{(r2 > 0.9).sum()} above 0.9")
            if r2.max() > 0.9:
                print("  WARNING: near-duplicate observable, expect closed-loop trouble")

        if args.test:
            ut, yt, at, lt, _, _ = load_flights(args.test)
            Zt, Ut, Yt = hankel.build(ut - u_mean, yt - y_mean, at, lt, par)
            E = Yt - (Psi @ Zt + Gam @ Ut)
            print(f"  held-out multi-step RMSE: {np.sqrt((E**2).mean()):.5f}")

        out = os.path.join(args.outdir, f"theta_{name}.npz")
        np.savez(out, Psi=Psi, Gam=Gam, u_mean=u_mean, y_mean=y_mean,
                 par=par, Ts=Ts, n_windows=Zp.shape[1],
                 tilt_p99_deg=np.percentile(tilt, 99),   # fly.py warns beyond these
                 vh_p99=vh_p99, vz_p99=vz_p99)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
