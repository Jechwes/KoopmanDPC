"""Python port of the MATLAB plant: quad_params.m + quad_plant.m.

Line-for-line: 13-state rigid body, 500 Hz rate PID with conditional
anti-windup, mixer inversion with rotor-speed clamping, RK4. Used only by
verify_matlab.py to check that the Python controller reproduces the MATLAB
closed loop -- nothing on the flight path imports this.
"""

import numpy as np


def quad_params():
    prm = {"m": 0.045, "g": 9.81,
           "J": np.diag([2.3951e-5, 2.3951e-5, 3.2347e-6]), "cd": 0.0}
    prm["Jin"] = np.linalg.inv(prm["J"])
    Kt, Kc, arm = 3.72e-8, 7.74e-12, 0.0353
    prm["Alloc"] = np.array([Kt * np.array([1, 1, 1, 1]),
                             Kt * arm * np.array([-1, -1, 1, 1]),
                             Kt * arm * np.array([-1, 1, 1, -1]),
                             -Kc * np.array([1, -1, 1, -1])])
    prm["omega_min"], prm["omega_max"] = 400.0, 2800.0

    lim = {"T_min": 0.30 * prm["m"] * prm["g"], "T_max": 1.80 * prm["m"] * prm["g"],
           "tau_max": np.array([4.0e-3, 4.0e-3, 2.0e-4]),
           "T_spec": [0.5 * prm["g"], 1.6 * prm["g"]], "w_cmd": 3.0}

    Jd = np.diag(prm["J"])
    inner = {"Kp_rate": Jd * np.array([60, 60, 30]),
             "Ki_rate": Jd * np.array([200, 200, 100]),
             "Kd_rate": np.zeros(3), "i_max_att": 50.0}
    return prm, lim, inner


def _rate_pid(om_sp, x, inner, Ts, st, lim):
    if st is None:
        st = {"e_int": np.zeros(3), "om_prev": x[10:13].copy()}
    om_act = x[10:13]
    e_rate = om_sp - om_act
    p_term = inner["Kp_rate"] * e_rate
    d_raw = -(om_act - st["om_prev"]) / Ts
    st["om_prev"] = om_act.copy()
    d_term = inner["Kd_rate"] * d_raw

    e_old = st["e_int"]
    e_try = np.clip(e_old + e_rate * Ts, -inner["i_max_att"], inner["i_max_att"])
    tau_try = p_term + d_term + inner["Ki_rate"] * e_try
    hold = (np.abs(tau_try) > lim["tau_max"]) & (np.sign(tau_try) == np.sign(e_rate))
    e_new = e_try.copy()
    e_new[hold] = e_old[hold]
    st["e_int"] = e_new

    tau = p_term + d_term + inner["Ki_rate"] * e_new
    return np.clip(tau, -lim["tau_max"], lim["tau_max"]), st


def _actuator_sat(T_cmd, tau_cmd, prm, lim):
    T_cmd = min(max(T_cmd, lim["T_min"]), lim["T_max"])
    tau_cmd = np.clip(tau_cmd, -lim["tau_max"], lim["tau_max"])
    om2 = np.linalg.solve(prm["Alloc"], np.r_[T_cmd, tau_cmd])
    om2 = np.clip(om2, prm["omega_min"] ** 2, prm["omega_max"] ** 2)
    w = prm["Alloc"] @ om2
    return w[0], w[1:4]


def _dynamics(x, T, tau, prm):
    v, om = x[3:6], x[10:13]
    q = x[6:10] / np.linalg.norm(x[6:10])
    q0, q1, q2, q3 = q
    R = np.array([[1-2*(q2**2+q3**2), 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2)],
                  [2*(q1*q2+q0*q3),   1-2*(q1**2+q3**2), 2*(q2*q3-q0*q1)],
                  [2*(q1*q3-q0*q2),   2*(q2*q3+q0*q1),   1-2*(q1**2+q2**2)]])
    vdot = R @ np.array([0, 0, T]) / prm["m"] - np.array([0, 0, prm["g"]]) \
        - prm["cd"] * v / prm["m"]
    Om = np.array([[0, -om[0], -om[1], -om[2]],
                   [om[0], 0, om[2], -om[1]],
                   [om[1], -om[2], 0, om[0]],
                   [om[2], om[1], -om[0], 0]])
    omdot = prm["Jin"] @ (tau - np.cross(om, prm["J"] @ om))
    return np.concatenate([v, vdot, 0.5 * Om @ q, omdot])


def quad_plant(x, v_cmd, prm, lim, inner, Ts_outer, Ts_inner, st):
    """Advance one OUTER step under v_cmd = [alpha_T, wx, wy, wz] (physical)."""
    n_inner = int(round(Ts_outer / Ts_inner))
    T_cmd = min(max(prm["m"] * v_cmd[0], lim["T_min"]), lim["T_max"])
    om_ref = np.asarray(v_cmd[1:4], dtype=float)
    h = Ts_inner
    for _ in range(n_inner):
        tau_cmd, st = _rate_pid(om_ref, x, inner, h, st, lim)
        T_app, tau_app = _actuator_sat(T_cmd, tau_cmd, prm, lim)
        f = lambda xx: _dynamics(xx, T_app, tau_app, prm)
        k1 = f(x); k2 = f(x + h/2*k1); k3 = f(x + h/2*k2); k4 = f(x + h*k3)
        x = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        x[6:10] = x[6:10] / np.linalg.norm(x[6:10])
    return x, st
