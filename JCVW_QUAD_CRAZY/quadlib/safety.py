"""Flight envelope checks. Anything that returns a reason string aborts.

Checked every control step, before the setpoint is transmitted.
"""

import time

import numpy as np


class Safety:
    def __init__(self, cfg, hover):
        lim = cfg["limits"]
        self.hover = np.asarray(hover, dtype=float)
        self.xy = float(lim["geofence_xy"])
        self.z_lo, self.z_hi = lim["geofence_z"]
        self.tilt_max = np.radians(lim["tilt_deg"])
        self.vicon_timeout = float(lim["vicon_timeout_s"])
        self.vbat_min = float(lim["vbat_min"])
        self.period_warn = 1.5 / float(cfg["loop"]["spc_hz"])
        self.n_overrun = 0

    def check(self, pose_t, pos, rpy, vbat, dt, enforce_floor=True):
        """enforce_floor=False during takeoff and landing, which legitimately
        pass below geofence_z[0]."""
        now = time.time()

        if now - pose_t > self.vicon_timeout:
            return f"vicon dropout ({now - pose_t:.3f} s)"

        d = pos - self.hover
        if abs(d[0]) > self.xy or abs(d[1]) > self.xy:
            return f"geofence xy ({d[0]:+.2f}, {d[1]:+.2f})"
        if pos[2] > self.z_hi or (enforce_floor and pos[2] < self.z_lo):
            return f"geofence z ({pos[2]:.2f} m)"

        tilt = np.arccos(np.clip(np.cos(rpy[0]) * np.cos(rpy[1]), -1, 1))
        if tilt > self.tilt_max:
            return f"tilt {np.degrees(tilt):.0f} deg"

        if vbat < self.vbat_min:
            return f"battery {vbat:.2f} V"

        if dt > self.period_warn:
            self.n_overrun += 1

        return None
