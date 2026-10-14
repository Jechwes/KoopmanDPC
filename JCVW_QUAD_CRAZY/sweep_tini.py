"""T_ini sweep: held-out prediction error of SPC and KSPC per window length.

No drone needed. For each T_ini (N fixed), the flights are split into folds; each
fold is predicted by a model identified on the other flights, with the same mean
removal as identify.py. The score is the RMSE of the position predicted over the
whole horizon, so it measures what the controller relies on, before flying.

    python sweep_tini.py                          # T_ini 5 10 15 20 30 40, N from config
    python sweep_tini.py --tini 3 5 8 10 --n 30
    python sweep_tini.py --flights "data/flights/[ab]_*.npz"

The x_ini window is a time-delay embedding, so T_ini chooses observables, not just
a horizon: the SPC can only see the drone's attitude through the window, the KSPC
also through phi_nl. A shorter window may therefore suit the KSPC better.

Folds take every k-th flight (sorted by name), so with sessions a and b each fold
holds one run of each. To fly a setting afterwards:

    python identify.py --tini 10 --outdir models/tini10
    python fly.py --traj fig8 --model models/tini10/theta_kspc.npz
"""

import argparse
import contextlib
import copy
import csv
import glob
import io
import os

import numpy as np

from identify import load_flights
from plot_compare import COLOR, INK, INK2, MUTED, NAME, SURFACE, style
from quadlib import hankel
from quadlib.lift import make_par
from quadlib.util import load_cfg


def load(files):
    """load_flights over several files, quietly; segments stay separate."""
    parts = []
    for f in files:
        with contextlib.redirect_stdout(io.StringIO()):
            parts.append(load_flights(f))
    cat = lambda i: np.concatenate([p[i] for p in parts])
    return cat(0), cat(1), cat(2), cat(3), parts[0][4]


def fold_errors(train, test, tini, n, cfg):
    """Position prediction RMSE over the horizon on the test flights [cm]:
    {"spc"|"kspc": (3D, z)}. One Hankel build serves both models, as the SPC's
    lifted state is the first n_ini rows of the KSPC's."""
    u, y, att, lens, Ts = load(train)
    ut, yt, at, lt, _ = load(test)
    um, ym = u.mean(0), y.mean(0)
    c = copy.deepcopy(cfg)
    c["model"].update(Tini=tini, N=n, use_phi=True)
    par = make_par(c, um, Ts)
    Zp, Uf, Yf = hankel.build(u - um, y - ym, att, lens, par)
    Zt, Ut, Yt = hankel.build(ut - um, yt - ym, at, lt, par)
    out = {}
    for name, rows in (("spc", par["n_ini"]), ("kspc", par["L"])):
        Psi, Gam = hankel.fit(Zp[:rows], Uf, Yf, dict(par, L=rows))
        E = (Yt - (Psi @ Zt[:rows] + Gam @ Ut)).reshape(n, par["p"], -1)[:, :3]
        out[name] = (100 * np.sqrt((E ** 2).sum(1).mean()), 100 * np.sqrt((E[:, 2] ** 2).mean()))
    return out, par["L"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tini", type=int, nargs="+", default=[5, 10, 15, 20, 30, 40])
    ap.add_argument("--n", type=int, default=None, help="horizon, default model.N")
    ap.add_argument("--flights", default="data/flights/*.npz")
    ap.add_argument("--folds", type=int, default=4)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--out", default="figures/tini_sweep", help="without extension")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    n = args.n or int(cfg["model"]["N"])
    files = sorted(glob.glob(args.flights))
    assert len(files) >= args.folds >= 2, f"{len(files)} flights for {args.folds} folds"
    assert min(args.tini) >= 2, "T_ini >= 2: phi_nl needs the input at k-1"
    folds = [files[i::args.folds] for i in range(args.folds)]
    print(f"{len(files)} flights, {args.folds} folds, N = {n}; "
          f"position RMSE over the horizon on held-out flights, mean (std) over folds\n")
    print(f"{'T_ini':>5} {'L':>4} | {'SPC 3D':>13} {'KSPC 3D':>13} {'gain':>6} | "
          f"{'SPC z':>6} {'KSPC z':>7}")

    rows = []
    for tini in sorted(args.tini):
        res = [fold_errors([f for f in files if f not in test], test, tini, n, cfg)
               for test in folds]
        L = res[0][1]
        e = {c: np.array([r[0][c] for r in res]) for c in ("spc", "kspc")}   # folds x (3D, z)
        gain = 100 * (1 - e["kspc"][:, 0] / e["spc"][:, 0])
        rows.append(dict(tini=tini, L=L,
                         spc=e["spc"][:, 0].mean(), spc_sd=e["spc"][:, 0].std(),
                         kspc=e["kspc"][:, 0].mean(), kspc_sd=e["kspc"][:, 0].std(),
                         gain=gain.mean(), gain_sd=gain.std(),
                         spc_z=e["spc"][:, 1].mean(), kspc_z=e["kspc"][:, 1].mean()))
        r = rows[-1]
        print(f"{tini:5d} {L:4d} | {r['spc']:6.2f} ({r['spc_sd']:4.2f}) {r['kspc']:6.2f} "
              f"({r['kspc_sd']:4.2f}) {r['gain']:5.1f}% | {r['spc_z']:6.2f} {r['kspc_z']:7.2f}")

    best = min(rows, key=lambda r: r["kspc"])
    print(f"\nlowest KSPC error at T_ini = {best['tini']}; config.yaml uses "
          f"{cfg['model']['Tini']}. Prediction is a proxy: fly the best few to decide.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(f"{args.out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(rows, int(cfg["model"]["Tini"]), n, args.out)
    print(f"saved {args.out}.png, .pdf and .csv")


def plot(rows, current, n, out):
    import matplotlib.pyplot as plt
    t = np.array([r["tini"] for r in rows])
    fig, (ax, ag) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.4, 1]))
    fig.patch.set_facecolor(SURFACE)
    shown = t.min() <= current <= t.max()          # the config's T_ini, if swept over
    for a in (ax, ag):
        style(a)
        if shown:
            a.axvline(current, color=MUTED, linewidth=0.8, linestyle=":", zorder=1)
        a.set_xlabel("T_ini [samples]", color=INK2, fontsize=9)
    for c in ("spc", "kspc"):
        m = np.array([r[c] for r in rows])
        sd = np.array([r[f"{c}_sd"] for r in rows])
        ax.fill_between(t, m - sd, m + sd, color=COLOR[c], alpha=0.15, linewidth=0)
        ax.plot(t, m, color=COLOR[c], marker="o", markersize=4, linewidth=1.6, label=NAME[c])
    ax.set_ylabel("held-out position error [cm]", color=INK2, fontsize=9)
    ax.set_title(f"Prediction error over the horizon (N = {n})", color=INK, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    g = np.array([r["gain"] for r in rows])
    gs = np.array([r["gain_sd"] for r in rows])
    ag.fill_between(t, g - gs, g + gs, color=COLOR["kspc"], alpha=0.15, linewidth=0)
    ag.plot(t, g, color=COLOR["kspc"], marker="o", markersize=4, linewidth=1.6)
    ag.axhline(0.0, color=MUTED, linewidth=0.8)
    ag.set_ylabel("KSPC improvement over SPC [%]", color=INK2, fontsize=9)
    ag.set_title("Gain of the observables", color=INK, fontsize=11, loc="left")
    if shown:
        ax.text(current, 1.0, " config", transform=ax.get_xaxis_transform(), color=MUTED,
                fontsize=8, va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=170, facecolor=SURFACE)


if __name__ == "__main__":
    main()
