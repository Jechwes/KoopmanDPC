"""Identification flight: laptop PID + white dither, logged at the SPC rate.

A quadrotor cannot be excited open loop, so it is flown by a stabilising PID
with white noise added on top of its output. That dither is what makes the
model identifiable: under pure feedback the input is a deterministic function
of the state, and least squares cannot separate "the input caused this
response" from "the state caused this input".

Everything runs in RATE mode at the SPC rate, so the logged u is exactly the
model input with no conversion, and the radio and solve latency present here is
the same latency the controller will face. Identifying on this data therefore
bakes the loop delay into Theta.

Ctrl+C at any point lands (or cuts the motors if Vicon is gone).

    python collect.py --runs 8 --tag baseline
"""

import argparse
import time

import numpy as np

from quadlib.link import Link, require_frame_verified
from quadlib.safety import Safety
from quadlib.util import AbortFlight, FlightLog, VelEstimator, abort, land, \
    load_cfg, pid_fly
from quadlib.vicon import ViconThread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default="data/flights")
    ap.add_argument("--seed0", type=int, default=1000,
                    help="run r uses seed seed0+r; change it between sessions so "
                         "the excitation is not repeated")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    require_frame_verified(cfg)
    col = cfg["collect"]
    hover = np.asarray(col["hover"], dtype=float)
    Ts = 1.0 / cfg["loop"]["spc_hz"]
    stamp = time.strftime("%Y%m%d_%H%M%S")

    with Link(cfg) as link:
        vic = ViconThread(cfg["vicon"]["hostname"], cfg["vicon"]["object_name"],
                          cf=link.cf)
        vic.start()
        if not vic.wait_for_pose():
            raise RuntimeError(f"no Vicon pose for '{cfg['vicon']['object_name']}'")
        link.reset_estimator()
        print(f"  vicon locked at {np.round(vic.latest()[1], 3)}")

        vel = VelEstimator()
        safe = Safety(cfg, hover)
        link.arm()
        link.unlock()

        try:
            for r in range(args.runs):
                print(f"\n--- run {r+1}/{args.runs} ---")
                rng = np.random.default_rng(args.seed0 + r)

                # climb and settle, no dither, so the transient stays out of the data
                p0 = vic.latest()[1].copy()
                pid_fly(link, vic, lambda t: (p0 + (hover - p0) * min(t / 3.0, 1.0), 0.0),
                        3.5, cfg, vel, safe)
                pid_fly(link, vic, lambda t: (hover, 0.0), col["settle_s"], cfg, vel, safe)

                # excitation: random setpoint steps + white dither on the command
                nhold = max(1, int(round(col["ref_hold_s"] / Ts)))
                box = np.asarray(col["ref_box"], dtype=float)
                K = int(round(col["duration_s"] / Ts))
                dT = col["dither_T"] * rng.standard_normal(K + 100)
                dw = col["dither_w"] * rng.standard_normal((K + 100, 3))
                seg = {"i": -1, "p": hover.copy(), "yaw": 0.0}

                def target(t):
                    # new setpoint whenever t enters a new hold segment, robust
                    # to loop overruns skipping individual steps
                    i = int(t / (nhold * Ts))
                    if i != seg["i"]:
                        seg.update(i=i, p=hover + (2 * rng.random(3) - 1) * box,
                                   yaw=(2 * rng.random() - 1) * col["yaw_amp"])
                    return seg["p"], seg["yaw"]

                log = FlightLog()
                pid_fly(link, vic, target, col["duration_s"], cfg, vel, safe, log=log,
                        dither=lambda k: np.concatenate([[dT[k]], dw[k]]))

                a = log.save(f"{args.outdir}/{args.tag}_{stamp}_r{r:02d}.npz",
                             dict(Ts=Ts, hover=hover, tag=args.tag,
                                  dither=[col["dither_T"], col["dither_w"]],
                                  ref_box=box, ref_hold_s=col["ref_hold_s"]))
                dt = np.diff(a[:, 0])
                print(f"  loop period: median {1e3*np.median(dt):.1f} ms "
                      f"(target {1e3*Ts:.1f}), max {1e3*dt.max():.1f} ms")

            land(link, vic, cfg, vel, safe)

        except (AbortFlight, KeyboardInterrupt) as e:
            abort(link, vic, cfg, vel, e if str(e) else "Ctrl+C")
        finally:
            vic.stop()
            print(f"\nvicon frames {vic.n_frames}, missing {vic.n_missing}")


if __name__ == "__main__":
    main()
