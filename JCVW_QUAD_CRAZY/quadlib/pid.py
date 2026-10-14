"""Cascaded position PID. Port of outer_pid.m.

Used only for data collection: it flies the drone while a dither is added on
top, which is what makes the model identifiable. Under pure feedback the input
is a deterministic function of the state and least squares cannot separate
"the input caused this" from "the state caused this".

Outputs the same u = [alpha_T, wx, wy, wz] the SPC does, so the logged inputs
are in the model's coordinates with no conversion.
"""

import numpy as np

G = 9.81


def quat_to_rot(qx, qy, qz, qw):
    q0, q1, q2, q3 = qw, qx, qy, qz
    return np.array([
        [1-2*(q2**2+q3**2), 2*(q1*q2-q3*q0),   2*(q1*q3+q2*q0)],
        [2*(q1*q2+q3*q0),   1-2*(q1**2+q3**2), 2*(q2*q3-q1*q0)],
        [2*(q1*q3-q2*q0),   2*(q2*q3+q1*q0),   1-2*(q1**2+q2**2)]])


def outer_pid(p, v, R, p_des, v_des, yaw_des, gains):
    """p, v: measured position and velocity. R: body-to-world rotation."""
    Kp = np.asarray(gains["Kp_pos"], dtype=float)
    Kd = np.asarray(gains["Kd_pos"], dtype=float)
    tilt_max = np.radians(gains["tilt_max_deg"])

    # 1. desired acceleration in the world frame
    a_des = Kp * (np.asarray(p_des) - p) + Kd * (np.asarray(v_des) - v) + np.array([0, 0, G])
    a_des[2] = max(a_des[2], 0.3 * G)               # never command negative lift

    # 2. clamp commanded tilt
    hor = np.linalg.norm(a_des[:2])
    hmax = a_des[2] * np.tan(tilt_max)
    if hor > hmax > 0 and hor > 1e-9:
        a_des[:2] *= hmax / hor

    # 3. specific thrust projected on the CURRENT body z, so the drone does not
    #    over-thrust while it is still rotating towards the desired attitude
    alpha_T = float(a_des @ R[:, 2])

    # 4. desired attitude
    z_des = a_des / np.linalg.norm(a_des)
    x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
    y_des = np.cross(z_des, x_c)
    n = np.linalg.norm(y_des)
    y_des = np.array([0.0, 1.0, 0.0]) if n < 1e-6 else y_des / n
    R_des = np.column_stack([np.cross(y_des, z_des), y_des, z_des])

    # 5. attitude error -> body-rate command
    M = R_des.T @ R - R.T @ R_des
    eR = 0.5 * np.array([M[2, 1], M[0, 2], M[1, 0]])
    return np.concatenate([[alpha_T], -gains["K_att"] * eR])
