"""Reference trajectories. Port of ref_traj.m.

pos/vel/acc are used to build a consistent initial state; the controller only
tracks pos and yaw.
"""

import numpy as np


def make(name, hover, ramp_s=4.0, speed=1.0):
    """The periodic references are eased in from rest over ramp_s seconds
    (ramp_in); ramp_s = 0 gives the raw trajectory, as ref_traj.m has it.
    speed != 1 flies the same path faster or slower (faster)."""
    traj = _make(name, np.asarray(hover, dtype=float))
    if speed != 1.0:
        traj = faster(traj, speed)
    periodic = name.startswith("fig8") or name == "circle_z"
    return ramp_in(traj, ramp_s) if periodic and ramp_s > 0 else traj


def faster(traj, s):
    """traj(s t): the same path, s times as fast. Tilt grows ~ s^2 and the body
    rates it takes ~ s^3. The speed goes into the name, so logs of different
    speeds never share a name."""
    return {"name": f"{traj['name']}_x{s:g}",
            "pos": lambda t: traj["pos"](s * t),
            "vel": lambda t: s * traj["vel"](s * t),
            "acc": lambda t: s ** 2 * traj["acc"](s * t),
            "yaw": lambda t: traj["yaw"](s * t)}


def ramp_in(traj, T):
    """Time-warp a reference so it starts from rest.

    tau'(t) = s(t) = (1 - cos(pi t/T)) / 2 rises from 0 to 1 over T seconds, so
    the reference leaves traj(0) with zero velocity and acceleration, and from
    t = T on it is exactly traj delayed by T/2. Unramped, the figure 8 starts at
    ~1 m/s the instant SPC takes over from a hover, and the first second is a
    catch-up at several times the tilt the trajectory itself needs.
    """
    def tau(t):
        return t - T / 2 if t >= T else t / 2 - T / (2 * np.pi) * np.sin(np.pi * t / T)

    def s(t):
        return 1.0 if t >= T else 0.5 * (1 - np.cos(np.pi * t / T))

    def ds(t):
        return 0.0 if t >= T else np.pi / (2 * T) * np.sin(np.pi * t / T)

    return {"name": traj["name"],
            "pos": lambda t: traj["pos"](tau(t)),
            "vel": lambda t: traj["vel"](tau(t)) * s(t),
            "acc": lambda t: traj["acc"](tau(t)) * s(t) ** 2 + traj["vel"](tau(t)) * ds(t),
            "yaw": lambda t: traj["yaw"](tau(t))}


def _make(name, h):
    if name == "hover":
        return {"name": name,
                "pos": lambda t: h.copy(),
                "vel": lambda t: np.zeros(3),
                "acc": lambda t: np.zeros(3),
                "yaw": lambda t: 0.0}

    if name == "step":
        d = np.array([0.5, 0.0, 0.0])
        return {"name": name,
                "pos": lambda t: h + d * (t >= 5.0),
                "vel": lambda t: np.zeros(3),
                "acc": lambda t: np.zeros(3),
                "yaw": lambda t: 0.0}

    if name.startswith("fig8_planes"):
        # Figure 8 with flat turns: each lobe is flown at constant height,
        # hover + H and hover - H, and the climb happens around the crossing,
        # z = H tanh(kap sin wt) / tanh(kap). The turns then need more thrust
        # only to hold height while banked -- the sag the observables model --
        # instead of sharing it with a vertical sine as fig8 does. Larger kap
        # gives flatter turns but a steeper climb, which drives the turn rate.
        # xfast: the biggest step the c/d identification data covers in thrust
        # (+/-3.6 of +/-4.2 m/s^2), tilt (19 of 24 deg) and turn rate (1.3 of
        # 1.55 rad/s); needs spc_box.alpha_T_rel of about [0.58, 1.42].
        A, B = 1.2, 0.5
        w, H, kap = {"fig8_planes_slow": (0.4, 0.4, 3.0),
                     "fig8_planes": (0.8, 0.4, 3.0),
                     "fig8_planes_fast": (1.1, 0.4, 3.0),
                     "fig8_planes_xfast": (1.21, 0.5, 2.5)}[name]
        hz = H / np.tanh(kap)

        def z_terms(t):
            s, c = np.sin(w * t), np.cos(w * t)
            T = np.tanh(kap * s)
            S2 = 1.0 - T ** 2
            return (hz * T, hz * kap * w * S2 * c,
                    hz * kap * w ** 2 * S2 * (-2.0 * kap * T * c ** 2 - s))

        return {"name": name,
                "pos": lambda t: h + np.array([A*np.sin(w*t),
                                               B*np.sin(2*w*t),
                                               z_terms(t)[0]]),
                "vel": lambda t: np.array([A*w*np.cos(w*t),
                                           2*B*w*np.cos(2*w*t),
                                           z_terms(t)[1]]),
                "acc": lambda t: np.array([-A*w**2*np.sin(w*t),
                                           -4*B*w**2*np.sin(2*w*t),
                                           z_terms(t)[2]]),
                "yaw": lambda t: 0.0}

    if name.startswith("fig8"):
        A, B, C = 0.8, 0.5, 0.25
        w = {"fig8_slow": 0.4, "fig8": 0.8, "fig8_fast": 1.1}[name]
        return {"name": name,
                "pos": lambda t: h + np.array([A*np.sin(w*t),
                                               B*np.sin(2*w*t),
                                               C*np.sin(w*t)]),
                "vel": lambda t: np.array([A*w*np.cos(w*t),
                                           2*B*w*np.cos(2*w*t),
                                           C*w*np.cos(w*t)]),
                "acc": lambda t: np.array([-A*w**2*np.sin(w*t),
                                           -4*B*w**2*np.sin(2*w*t),
                                           -C*w**2*np.sin(w*t)]),
                "yaw": lambda t: 0.0}

    if name == "circle_z":
        # constant bank angle with an oscillating altitude, so thrust and tilt
        # are off trim at the same time -- the hardest case for a linear map
        R, w, C, wz = 0.8, 1.2, 0.3, 0.5
        return {"name": name,
                "pos": lambda t: h + np.array([R*np.sin(w*t),
                                               -R*np.cos(w*t),
                                               C*np.sin(wz*t)]),
                "vel": lambda t: np.array([R*w*np.cos(w*t),
                                           R*w*np.sin(w*t),
                                           C*wz*np.cos(wz*t)]),
                "acc": lambda t: np.array([-R*w**2*np.sin(w*t),
                                           R*w**2*np.cos(w*t),
                                           -C*wz**2*np.sin(wz*t)]),
                "yaw": lambda t: 0.0}

    raise ValueError(f"unknown trajectory '{name}'")


def peak_tilt_deg(traj, t_end=40.0, g=9.81):
    """Tilt the reference demands: tan(tilt) = |a_xy| / (g + a_z).
    Check this against the tilt your identification data actually covers."""
    t = np.arange(0.0, t_end, 0.02)
    out = 0.0
    for ti in t:
        a = traj["acc"](ti)
        out = max(out, np.degrees(np.arctan2(np.linalg.norm(a[:2]), g + a[2])))
    return out


def peak_speed(traj, t_end=40.0, dt=0.02):
    """Largest horizontal and vertical speed [m/s] the reference asks for, to
    hold against the speeds the identification data covers."""
    v = np.array([traj["vel"](ti) for ti in np.arange(0.0, t_end, dt)])
    return np.hypot(v[:, 0], v[:, 1]).max(), np.abs(v[:, 2]).max()


def thrust_range(traj, t_end=40.0, g=9.81, dt=0.01):
    """Smallest and largest specific thrust [m/s^2] the reference alone asks
    for, |a + g e3|, to hold against the SPC's thrust box."""
    a = np.array([traj["acc"](ti) for ti in np.arange(0.0, t_end, dt)])
    T = np.linalg.norm(a + np.array([0.0, 0.0, g]), axis=1)
    return T.min(), T.max()


def peak_tilt_rate(traj, t_end=40.0, g=9.81, dt=0.01):
    """Body rate [rad/s] the reference alone demands: how fast the thrust
    direction b3 = (a + g e3) / |a + g e3| has to turn. Feed-forward only, so
    the SPC's rate box needs headroom above it for feedback."""
    t = np.arange(0.0, t_end, dt)
    b = np.array([traj["acc"](ti) + np.array([0.0, 0.0, g]) for ti in t])
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    return np.linalg.norm(np.diff(b, axis=0), axis=1).max() / dt


def horizon(traj, t, par, y_mean):
    """Mean-removed reference over the prediction horizon, stacked as Yf is."""
    N, p, Ts = par["N"], par["p"], par["Ts"]
    r = np.empty(N * p)
    for j in range(1, N + 1):
        tj = t + j * Ts
        r[(j-1)*p:j*p] = np.concatenate([traj["pos"](tj), [traj["yaw"](tj)]]) - y_mean
    return r
