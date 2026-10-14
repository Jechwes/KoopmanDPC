"""Config loading, velocity estimation, the PID flight helper, abort handling,
and the flight logger."""

import os
import time

import numpy as np
import yaml

from .convert import clip_u
from .pid import outer_pid


def load_cfg(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


class AbortFlight(Exception):
    """Raised by any safety check. Caught by the scripts, which land if Vicon
    is still valid and cut the motors otherwise."""


class VelEstimator:
    """Low-passed finite difference of Vicon position. Feeds the collection PID
    only; the model never sees it."""

    def __init__(self, tau=0.08):
        self.tau, self.v, self.p, self.t = tau, np.zeros(3), None, None

    def update(self, t, p):
        if self.p is not None and t > self.t:
            dt = t - self.t
            a = dt / (self.tau + dt)
            self.v = (1 - a) * self.v + a * (p - self.p) / dt
        self.p, self.t = p.copy(), t
        return self.v.copy()


def rot_from_rpy(rpy):
    """R = Rz(yaw) Ry(pitch) Rx(roll), body-to-world, same as quat_to_rot."""
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                              np.sin(p), np.cos(y), np.sin(y))
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp,   cp*sr,            cp*cr]])


def pid_fly(link, vic, target_fn, duration, cfg, vel, safety=None,
            log=None, dither=None):
    """Run the laptop PID in rate mode for `duration` seconds.

    target_fn(t) -> (p_des, yaw_des). Used for takeoff, settling, landing and
    the excitation flight, so there is one input convention everywhere.
    Raises AbortFlight if `safety` trips. The geofence floor is not enforced
    here: takeoff and landing pass below it by design.
    """
    Ts = 1.0 / cfg["loop"]["spc_hz"]
    gains = cfg["pid"]
    t0 = t_next = time.perf_counter()
    k, t_prev = 0, 0.0
    while True:
        t = time.perf_counter() - t0
        if t >= duration:
            return

        pose = vic.latest()
        if pose is None:
            raise AbortFlight("no Vicon pose")
        pose_t, pos, rpy = pose
        if safety is not None:
            why = safety.check(pose_t, pos, rpy, link.telemetry["vbat"],
                               t - t_prev, enforce_floor=False)
            if why:
                raise AbortFlight(why)
        t_prev = t

        v = vel.update(t, pos)
        p_des, yaw_des = target_fn(t)
        u = outer_pid(pos, v, rot_from_rpy(rpy), p_des, np.zeros(3), yaw_des, gains)
        if dither is not None:
            u = u + dither(k)
        u = clip_u(u, cfg)              # log what the drone gets, not the request

        link.send_u(u)
        if log is not None:
            log.append(t, pos, rpy, u, link.telemetry, p_des)

        k += 1
        t_next += Ts                    # after an overrun, restart the grid
        s = t_next - time.perf_counter()  # rather than catch up (see fly.py)
        if s > 0:
            time.sleep(s)
        else:
            t_next = time.perf_counter()


def land(link, vic, cfg, vel, safety=None):
    """PID descent at 0.35 m/s from the current position, then motors off."""
    pz = vic.latest()[1].copy()
    pid_fly(link, vic,
            lambda t: (np.array([pz[0], pz[1], max(0.05, pz[2] - 0.35 * t)]), 0.0),
            pz[2] / 0.35 + 1.5, cfg, vel, safety)
    link.stop()


def abort(link, vic, cfg, vel, reason):
    """Land under control if Vicon is still fresh, otherwise cut the motors.

    Without a valid pose there is no way to land under control; at <= 2 m in a
    netted cage, dropping is the safer failure."""
    print(f"\n*** ABORT: {reason} ***")
    pose = vic.latest()
    fresh = pose is not None and time.time() - pose[0] < cfg["limits"]["vicon_timeout_s"]
    if fresh:
        print("  Vicon valid -- landing")
        try:
            land(link, vic, cfg, vel)
            return
        except (AbortFlight, KeyboardInterrupt) as e:
            print(f"  landing aborted too ({e}) -- motors off")
    else:
        print("  no valid Vicon -- motors off")
    link.stop()


class FlightLog:
    """One row per control step.

    ORDERING: the pose is the one READ BEFORE u was computed and sent, so u(k)
    drives y(k+1). hankel.build assumes exactly that. Appending the pose after
    transmitting would put u(k) and y(k) in the same row and leave the
    identified predictor one step ahead of anything a controller can do.
    """

    def __init__(self):
        self.rows = []

    def append(self, t, pos, rpy, u, tel, p_ref=None):
        self.rows.append(np.concatenate([
            [t], pos, rpy, u, [tel["vbat"]], tel["gyro"],
            p_ref if p_ref is not None else np.full(3, np.nan)]))

    def save(self, path, meta):
        a = np.asarray(self.rows)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path,
                 t=a[:, 0], pos=a[:, 1:4], rpy=a[:, 4:7], u=a[:, 7:11],
                 vbat=a[:, 11], gyro=a[:, 12:15], p_ref=a[:, 15:18],
                 **meta)
        print(f"  saved {len(a)} samples -> {path}")
        return a
