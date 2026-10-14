"""Closed-loop SPC on the drone (Koopman SPC when model.use_phi is true).

Loop order, every step, and it matters:

    read Vicon  ->  lift  ->  solve  ->  transmit  ->  push into buffers

so the pose used to compute u(k) is logged before u(k) goes out, and u(k)
drives y(k+1). That is the same alignment hankel.build assumes and the same one
collect.py records, so the identified model and the live loop agree.

Takeoff and the climb to the trajectory start are flown by the collection PID;
SPC takes over at the start point and hands back to the PID for landing. The
SPC's past window is filled from the PID's last Tini steps, so the first
prediction starts from the drone's real history, not a replicated sample.
Ctrl+C at any point lands (or cuts the motors if Vicon is gone).

    python fly.py --traj fig8                # model.use_phi picks SPC or KSPC
    python fly.py --traj fig8 --no-phi       # SPC,  models/theta_spc.npz
    python fly.py --traj fig8 --speed 1.75   # same path 1.75x faster (w = 1.4 rad/s)
"""

import argparse
import time

import numpy as np

from quadlib import reference, spc
from quadlib.lift import lift
from quadlib.link import Link, require_frame_verified
from quadlib.safety import Safety
from quadlib.util import AbortFlight, FlightLog, VelEstimator, abort, land, \
    load_cfg, pid_fly
from quadlib.vicon import ViconThread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", default="fig8")
    ap.add_argument("--no-phi", action="store_true", help="SPC baseline")
    ap.add_argument("--model", default=None, help="override the theta file")
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--ramp-s", type=float, default=4.0,
                    help="ease the reference in from rest over this long, 0 = off")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="fly the trajectory this many times faster, e.g. 1.75")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    require_frame_verified(cfg)
    use_phi = cfg["model"]["use_phi"] and not args.no_phi
    model = args.model or f"models/theta_{'kspc' if use_phi else 'spc'}.npz"

    ctrl, par, u_mean, y_mean = spc.load(model, cfg)
    Ts, Tini = par["Ts"], par["Tini"]
    hover = np.asarray(cfg["collect"]["hover"], dtype=float)
    traj = reference.make(args.traj, hover, args.ramp_s, args.speed)
    name = "kspc" if par["use_phi"] else "spc"

    print(f"{model}: {name.upper()}, L {par['L']}, Ts {1e3*Ts:.1f} ms")
    if (Tini, par["N"]) != (cfg["model"]["Tini"], cfg["model"]["N"]):
        if args.model:      # chosen on purpose, e.g. a setting from sweep_tini.py
            print(f"  note: this model uses Tini/N {Tini}/{par['N']}, config.yaml "
                  f"{cfg['model']['Tini']}/{cfg['model']['N']}; flying the model's")
        else:
            print(f"  WARNING: model identified with Tini/N {Tini}/{par['N']}, config.yaml "
                  f"says {cfg['model']['Tini']}/{cfg['model']['N']}. Re-run identify.py.")
    tilt = reference.peak_tilt_deg(traj, args.duration)
    rate = reference.peak_tilt_rate(traj, args.duration)
    print(f"reference {traj['name']}, peak tilt {tilt:.1f} deg, tilt rate {rate:.2f} rad/s")
    box = cfg.get("spc_box", cfg["limits"])["omega"]
    if rate > 0.9 * box:            # fig8_fast flew at 71 %, fig8 x1.75 diverged at 148 %
        print(f"  WARNING: the reference alone needs {rate:.2f} of the +/-{box} rad/s the "
              f"SPC may command, leaving little for feedback. Raise spc_box.omega: in "
              f"simulation +/-0.785 diverged at fig8 x1.75 and +/-1.5 flew it.")
    t_lo, t_hi = reference.thrust_range(traj, args.duration)
    b_lo, b_hi = ctrl.lo[0] + u_mean[0], ctrl.hi[0] + u_mean[0]     # the QP's thrust box
    print(f"reference thrust {t_lo:.2f}..{t_hi:.2f} m/s2, box {b_lo:.2f}..{b_hi:.2f}")
    if u_mean[0] - t_lo > 0.9 * (u_mean[0] - b_lo) or t_hi - u_mean[0] > 0.9 * (b_hi - u_mean[0]):
        print(f"  WARNING: the reference alone needs most of the thrust box, leaving little "
              f"for feedback. Widen spc_box.alpha_T_rel, within the thrust the "
              f"identification data covers.")
    saved = np.load(model, allow_pickle=True)
    if "tilt_p99_deg" in saved.files and tilt > float(saved["tilt_p99_deg"]):
        print(f"  WARNING: peak tilt {tilt:.0f} deg is beyond 99 % of the identification "
              f"data ({float(saved['tilt_p99_deg']):.0f} deg): the model extrapolates. "
              f"Collect more aggressive data (larger collect.ref_box, shorter ref_hold_s).")
    vh, vz = reference.peak_speed(traj, args.duration)
    print(f"reference speed: horizontal {vh:.2f} m/s, vertical {vz:.2f} m/s")
    if "vz_p99" in saved.files and (vh > float(saved["vh_p99"]) or vz > float(saved["vz_p99"])):
        print(f"  WARNING: beyond 99 % of the identification data ({float(saved['vh_p99']):.2f} m/s "
              f"horizontal, {float(saved['vz_p99']):.2f} m/s vertical): the model has hardly "
              f"seen the drone this fast. Collect faster data (shorter collect.ref_hold_s).")

    log = FlightLog()
    solve_ms, n_fail, K = [], 0, int(round(args.duration / Ts))

    with Link(cfg) as link:
        vic = ViconThread(cfg["vicon"]["hostname"], cfg["vicon"]["object_name"],
                          cf=link.cf)
        vic.start()
        if not vic.wait_for_pose():
            raise RuntimeError("no Vicon pose")
        link.reset_estimator()

        vel = VelEstimator()
        safe = Safety(cfg, hover)
        link.arm()
        link.unlock()

        try:
            # climb to the trajectory start on the PID, settle, then hand over
            start = traj["pos"](0.0)
            p0 = vic.latest()[1].copy()
            pid_fly(link, vic, lambda t: (p0 + (start - p0) * min(t / 4.0, 1.0), 0.0),
                    4.5, cfg, vel, safe)
            hold = FlightLog()
            pid_fly(link, vic, lambda t: (start, 0.0), max(2.0, 2 * Tini * Ts),
                    cfg, vel, safe, log=hold)

            # past window = the PID's last Tini steps: Ybuf the last Tini poses
            # (the first push below drops the oldest), Ubuf the last Tini-1
            # inputs, the newest of which drives the first pose SPC reads
            h = np.asarray(hold.rows[-Tini:])
            Ybuf = np.column_stack([h[:, 1:4], h[:, 6]]) - y_mean
            Abuf = h[:, 4:7]
            Ubuf = h[1:, 7:11] - u_mean

            print(f"\n--- {name.upper()} engaged ---")
            t0 = t_next = time.perf_counter()
            t_prev = 0.0
            for k in range(K):
                t = time.perf_counter() - t0

                pose_t, pos, rpy = vic.latest()
                why = safe.check(pose_t, pos, rpy, link.telemetry["vbat"], t - t_prev)
                if why:
                    raise AbortFlight(why)
                t_prev = t

                y_k = np.concatenate([pos, [rpy[2]]]) - y_mean
                Ybuf = np.vstack([Ybuf[1:], y_k])
                Abuf = np.vstack([Abuf[1:], rpy])

                z_k = lift(Ubuf, Ybuf, Abuf, par)
                r_k = reference.horizon(traj, t, par, y_mean)

                ts = time.perf_counter()
                u_k, ok = ctrl.step(z_k, r_k)
                solve_ms.append(1e3 * (time.perf_counter() - ts))
                n_fail += (not ok)

                link.send_u(u_k + u_mean)                  # <- on the wire
                Ubuf = np.vstack([Ubuf[1:], u_k])
                log.append(t, pos, rpy, u_k + u_mean, link.telemetry, traj["pos"](t))

                # After an overrun, restart the grid from now instead of
                # catching up: back-to-back catch-up steps put samples a few ms
                # apart, which the fixed-Ts model reads as Ts apart. With 3 %
                # of steps stalled 20-80 ms, catching up diverged fig8_fast in
                # simulation; restarting the grid flew it.
                t_next += Ts
                s = t_next - time.perf_counter()
                if s > 0:
                    time.sleep(s)
                else:
                    t_next = time.perf_counter()

            print("--- trajectory done, landing ---")
            land(link, vic, cfg, vel, safe)

        except (AbortFlight, KeyboardInterrupt) as e:
            abort(link, vic, cfg, vel, e if str(e) else "Ctrl+C")
        finally:
            vic.stop()

    if not log.rows:
        return
    # a window other than config.yaml's goes into the name, so plot_compare.py and
    # plot_fig8_3d.m keep it apart from the default runs instead of averaging them
    tag = (f"_tini{Tini}" if Tini != cfg["model"]["Tini"] else "") + \
          (f"_n{par['N']}" if par["N"] != cfg["model"]["N"] else "")
    a = log.save(f"logs/fly_{traj['name']}{tag}_{name}_{time.strftime('%Y%m%d_%H%M%S')}.npz",
                 dict(Ts=Ts, traj=traj["name"], use_phi=par["use_phi"], model=model))

    e = a[:, 1:4] - a[:, 15:18]
    print(f"\n{traj['name']} | {name.upper()} | {len(a)} steps")
    print("RMSE x %.4f  y %.4f  z %.4f m   |  3D %.4f m" %
          (*np.sqrt((e**2).mean(0)), np.sqrt((e**2).sum(1).mean())))
    print(f"solve: mean {np.mean(solve_ms):.2f} ms, max {np.max(solve_ms):.2f} ms, "
          f"{n_fail} not solved")
    dt = np.diff(a[:, 0])
    if len(dt):
        print(f"loop period: median {1e3*np.median(dt):.1f} ms (target {1e3*Ts:.1f}), "
              f"max {1e3*dt.max():.1f}, {safe.n_overrun} overruns")
        if abs(np.median(dt) - Ts) > 0.1 * Ts:
            print("  WARNING: achieved period != Ts. The model plans for a cadence "
                  "the loop does not deliver; re-identify at the achieved rate.")


if __name__ == "__main__":
    main()
