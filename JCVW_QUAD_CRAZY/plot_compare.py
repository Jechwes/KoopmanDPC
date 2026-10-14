"""SPC vs KSPC on the figure 8s: top view and tracking error, from logs/.

One row per figure 8 flown with both controllers, any --speed, slowest first.
Per trajectory and controller it shows the run whose RMSE is closest to that
controller's mean (ties: the latest), so the plot is representative when a
trajectory was flown more than once. RMSE counts from 5 s after SPC engages:
the first seconds are the PID handover and the 4 s ramp-in.

    python plot_compare.py                        # -> figures/spc_vs_kspc.png, .pdf
    python plot_compare.py --exclude fig8_x2.5    # all but that one
    python plot_compare.py --traj fig8 fig8_fast --out figures/fig8_pair
    python plot_compare.py --z --traj fig8 fig8_planes   # height: sag in the turns

--z swaps the top view for the height: height over time and the signed height
error, with the turns shaded (the reference in the outer half of a lobe). Holding
height while banked takes extra thrust that grows with the square of the tilt,
which a linear predictor cannot form, so the SPC sags there. It prints the z error
in the turns and through the crossing, and the mean error in the turns (the sag).

Trajectory names are as in the log file names; run without options to see them.
"""

import argparse
import glob
import os
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

BASE_W = {"fig8_slow": 0.4, "fig8": 0.8, "fig8_fast": 1.1,   # reference.py
          "fig8_planes_slow": 0.4, "fig8_planes": 0.8, "fig8_planes_fast": 1.1,
          "fig8_planes_xfast": 1.21}
SKIP_S = 5.0

# reference palette (dataviz): slots 1 and 2, neutral reference, quiet chrome
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
COLOR = {"spc": "#2a78d6", "kspc": "#eb6834"}
NAME = {"spc": "SPC", "kspc": "KSPC"}


def load_runs():
    runs = defaultdict(list)
    for f in glob.glob("logs/fly_*.npz"):
        m = re.match(r"fly_(.+)_(spc|kspc)_(\d+_\d+)\.npz", os.path.basename(f))
        if not m:
            continue
        traj, ctrl, stamp = m.groups()
        d = np.load(f)
        t, e = d["t"], d["pos"] - d["p_ref"]
        k = t - t[0] > SKIP_S
        rmse = np.sqrt((e[k] ** 2).sum(1).mean())
        runs[(traj, ctrl)].append(dict(stamp=stamp, rmse=rmse, t=t - t[0],
                                       pos=d["pos"], ref=d["p_ref"],
                                       err=np.linalg.norm(e, axis=1)))
    return runs


def omega(traj):
    """Figure-8 angular rate from a log name (fig8_x1.75 -> 1.4), else None."""
    m = re.fullmatch(r"(fig8(?:_planes)?(?:_slow|_fast|_xfast)?)(?:_x([0-9.]+))?"
                     r"(?:_tini\d+)?(?:_n\d+)?", traj)
    return BASE_W[m.group(1)] * float(m.group(2) or 1.0) if m else None


def turns(ref):
    """True where the reference is in a turn: the outer half of a lobe in x."""
    dx = ref[:, 0] - 0.5 * (ref[:, 0].max() + ref[:, 0].min())
    return np.abs(dx) > 0.5 * np.abs(dx).max()


def z_stats(r):
    """z RMSE in the turns and through the crossing, and the mean z error in the
    turns (negative = sagging), all in cm, from SKIP_S on."""
    k = r["t"] > SKIP_S
    ez = 100 * (r["pos"][:, 2] - r["ref"][:, 2])
    turn = turns(r["ref"])
    rms = lambda m: np.sqrt(np.mean(ez[m] ** 2))
    return rms(turn & k), rms(~turn & k), ez[turn & k].mean()


def shade(ax, t, mask):
    edges = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(t[a], t[b - 1], color=BAND, zorder=0, linewidth=0)


def draw_z(ax_h, ax_e, traj, sub, picked, runs, line):
    """--z: height over time and signed height error, turns shaded, from SKIP_S."""
    ref = picked["spc"][0]
    t, k = ref["t"], ref["t"] > SKIP_S
    for ax in (ax_h, ax_e):
        shade(ax, t, turns(ref["ref"]) & k)
        ax.set_xlim(SKIP_S, max(picked[c][0]["t"][-1] for c in picked))
        ax.set_xlabel("time since SPC engaged [s]", color=INK2, fontsize=9)
    ax_h.plot(t[k], ref["ref"][k, 2], color=MUTED, linewidth=1.2, zorder=1, label="Reference")
    peak = 0.0
    for c in ("spc", "kspc"):
        r = picked[c][0]
        kk = r["t"] > SKIP_S
        ez = 100 * (r["pos"][kk, 2] - r["ref"][kk, 2])
        ax_h.plot(r["t"][kk], r["pos"][kk, 2], color=COLOR[c], zorder=2, label=NAME[c], **line)
        ax_e.plot(r["t"][kk], ez, color=COLOR[c], zorder=2, **line)
        peak = max(peak, np.abs(ez).max())
    ax_e.axhline(0.0, color=AXIS, linewidth=0.8, zorder=1)
    ax_e.set_ylim(-1.15 * peak, 1.15 * peak)
    ax_h.set_ylabel("height [m]", color=INK2, fontsize=9)
    ax_e.set_ylabel("height error [cm]", color=INK2, fontsize=9)
    ax_h.set_title(f"{traj}  ({sub}), turns shaded", color=INK, fontsize=11, loc="left", pad=8)
    st = {c: np.mean([z_stats(r) for r in runs[(traj, c)]], axis=0) for c in ("spc", "kspc")}
    ax_e.set_title("z RMSE in turns / through crossing (mean error in turns)\n" + "   ".join(
        f"{NAME[c]} {st[c][0]:.1f} / {st[c][1]:.1f} cm ({st[c][2]:+.1f})" for c in ("spc", "kspc")),
        color=INK2, fontsize=9, loc="left", pad=8)


def representative(rs):
    mean = np.mean([r["rmse"] for r in rs])
    # rounded, so runs equally far from the mean (always so with two) really
    # tie, and the latest wins instead of floating-point noise
    return min(sorted(rs, key=lambda r: r["stamp"], reverse=True),
               key=lambda r: round(abs(r["rmse"] - mean), 6)), mean


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9, length=3, width=0.8)
    ax.grid(True, color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", nargs="+", default=None,
                    help="plot only these trajectories, e.g. fig8 fig8_fast fig8_x1.75")
    ap.add_argument("--exclude", nargs="+", default=[], help="leave these trajectories out")
    ap.add_argument("--out", default=None,
                    help="output path without extension (.png and .pdf are written); "
                         "default figures/spc_vs_kspc, or figures/spc_vs_kspc_z with --z")
    ap.add_argument("--z", action="store_true",
                    help="height over time and height error instead of the top view")
    args = ap.parse_args()
    args.out = args.out or ("figures/spc_vs_kspc_z" if args.z else "figures/spc_vs_kspc")

    runs = load_runs()
    both = sorted({tr for tr, _ in runs
                   if omega(tr) and (tr, "spc") in runs and (tr, "kspc") in runs}, key=omega)
    unknown = (set(args.traj or []) | set(args.exclude)) - set(both)
    if unknown:
        print(f"  not in logs/ with both SPC and KSPC, ignored: {', '.join(sorted(unknown))}")
    rows = [(tr, f"w = {omega(tr):g} rad/s" + "".join(         # + ", Tini 10" etc.
                 f", {'Tini' if k == 'tini' else 'N'} {v}"
                 for k, v in re.findall(r"_(tini|n)(\d+)(?=_|$)", tr))) for tr in both
            if (args.traj is None or tr in args.traj) and tr not in args.exclude]
    if not rows:
        raise SystemExit(f"nothing to plot; with both controllers in logs/: {', '.join(both)}")

    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 3.6 * len(rows) + 0.5),
                             gridspec_kw=dict(width_ratios=[1, 1] if args.z else [1, 1.6]),
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    line = dict(linewidth=1.4, solid_capstyle="round", solid_joinstyle="round")

    for row, (traj, sub) in enumerate(rows):
        ax_xy, ax_e = axes[row]
        style(ax_xy); style(ax_e)
        picked = {c: representative(runs[(traj, c)]) for c in ("spc", "kspc")}
        if args.z:
            draw_z(ax_xy, ax_e, traj, sub, picked, runs, line)
            continue

        ref = picked["spc"][0]["ref"]
        ax_xy.plot(ref[:, 0], ref[:, 1], color=MUTED, linewidth=1.2, zorder=1,
                   label="Reference")
        for c in ("spc", "kspc"):
            r = picked[c][0]
            ax_xy.plot(r["pos"][:, 0], r["pos"][:, 1], color=COLOR[c], zorder=2,
                       label=NAME[c], **line)
            ax_e.plot(r["t"], 100 * r["err"], color=COLOR[c], zorder=2, **line)
        ax_xy.set_aspect("equal", adjustable="datalim")
        ax_xy.set_xlabel("x [m]", color=INK2, fontsize=9)
        ax_xy.set_ylabel("y [m]", color=INK2, fontsize=9)
        ax_xy.set_title(f"{traj}  ({sub})", color=INK, fontsize=11, loc="left", pad=8)

        # scale to the tracking phase; the handover peak goes off scale, noted
        tracking = max(100 * r["err"][r["t"] > 2.0].max() for r, _ in picked.values())
        peak = max(100 * r["err"].max() for r, _ in picked.values())
        ax_e.axvspan(0, SKIP_S, color=BAND, zorder=0, linewidth=0)
        ax_e.text(0.6 * SKIP_S, 0.97, f"handover,\nramp-in\npeak {peak:.0f} cm",   # clear of the spike
                  transform=ax_e.get_xaxis_transform(), ha="center", va="top",
                  fontsize=8, color=MUTED)
        ax_e.set_xlim(0, max(picked[c][0]["t"][-1] for c in picked))
        ax_e.set_ylim(0, 1.3 * tracking)
        ax_e.set_xlabel("time since SPC engaged [s]", color=INK2, fontsize=9)
        ax_e.set_ylabel("position error [cm]", color=INK2, fontsize=9)
        n = {c: len(runs[(traj, c)]) for c in picked}
        stats = "   ".join(
            f"{NAME[c]} {100 * picked[c][1]:.1f} cm" + (f" (mean of {n[c]})" if n[c] > 1 else "")
            for c in ("spc", "kspc"))
        ax_e.set_title(f"RMSE after {SKIP_S:.0f} s:  {stats}", color=INK2, fontsize=10,
                       loc="left", pad=8)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=10,
               labelcolor=INK, bbox_to_anchor=(0.985, 0.995))
    fig.suptitle("SPC vs KSPC on the Crazyflie 2.1 Brushless", x=0.012, y=0.995,
                 ha="left", va="top", fontsize=13, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.32 / fig.get_figheight()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=170, facecolor=SURFACE)
    print(f"saved {args.out}.png and .pdf  (with both controllers in logs/: {', '.join(both)})")
    for traj, _ in rows:
        for c in ("spc", "kspc"):
            r, mean = representative(runs[(traj, c)])
            print(f"  {traj:9s} {NAME[c]:4s}: showing run {r['stamp']} "
                  f"({100 * r['rmse']:.1f} cm; mean of {len(runs[(traj, c)])} runs {100 * mean:.1f} cm)")
            if args.z:
                zt, zc, sag = np.mean([z_stats(x) for x in runs[(traj, c)]], axis=0)
                print(f"  {'':9s} {'':4s}  z RMSE in turns {zt:.1f} cm, through the crossing "
                      f"{zc:.1f} cm, mean z error in turns {sag:+.1f} cm")


if __name__ == "__main__":
    main()
