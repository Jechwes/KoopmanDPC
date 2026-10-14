"""Unit and frame conversions between the model and the radio.

The model works in specific thrust [m/s^2] and body rates [rad/s], mean-removed.
The radio takes body rates in deg/s and thrust as a uint16 PWM. Everything that
crosses that boundary lives here, so there is exactly one place to get the signs
and units wrong.
"""

import numpy as np

G = 9.81


# ---------------------------------------------------------------- attitude
def quat_to_rpy(qx, qy, qz, qw):
    """Quaternion -> (roll, pitch, yaw) in rad, ZYX convention.

    Same formulas as measure() in KDPC.m, with the MATLAB q = [w x y z] order
    mapped onto Vicon's (x, y, z, w).
    """
    q0, q1, q2, q3 = qw, qx, qy, qz
    roll  = np.arctan2(2*(q0*q1 + q2*q3), 1 - 2*(q1**2 + q2**2))
    pitch = np.arcsin(np.clip(2*(q0*q2 - q3*q1), -1.0, 1.0))
    yaw   = np.arctan2(2*(q0*q3 + q1*q2), 1 - 2*(q2**2 + q3**2))
    return roll, pitch, yaw


def unwrap_yaw(yaw_new, yaw_prev):
    """Continuous yaw, so the linear regression never sees a +/-pi jump."""
    return yaw_prev + np.arctan2(np.sin(yaw_new - yaw_prev),
                                 np.cos(yaw_new - yaw_prev))


# ------------------------------------------------------------------ thrust
def thrust_to_pwm(thrust_N, thrust_max_N, pwm_lim=(10001, 60000)):
    """Thrust -> PWM, inverting  F = (pwm/65535) * thrust_max.

    No battery term: the Crazyflie 2.1 Brushless firmware compensates the
    battery itself. Eight PID flights between 3.5 and 3.9 V fitted thrust ~
    V^0.0 at fixed PWM, and the V^2 correction this function used to apply on
    top drifted the thrust gain 25 % over one battery.

    thrust_max_N is per drone; hover.py prints it.
    """
    pwm = thrust_N / thrust_max_N * 65535.0
    return int(np.clip(pwm, pwm_lim[0], pwm_lim[1]))


def alpha_to_thrust(alpha_T, mass_kg):
    """Specific thrust [m/s^2] -> total thrust [N].

    Only mass_kg / thrust_max_N reaches the PWM, and hover.py calibrates
    thrust_max_N with the configured mass, so a mass that is a few grams off is
    absorbed by the calibration. Change the two together, never one alone."""
    return alpha_T * mass_kg


# --------------------------------------------------------------- setpoints
def clip_u(u_phys, cfg):
    """The input the drone actually receives, in model units.

    Anything that LOGS an input must log this, not the unclipped request:
    identifying on the request when the actuator box was active teaches the
    model an input the plant never saw. The collection PID's yaw loop alone
    exceeds 3 rad/s on every yaw setpoint step.
    """
    lim = cfg["limits"]
    u = np.array(u_phys, dtype=float)
    u[0] = np.clip(u[0], lim["alpha_T"][0], lim["alpha_T"][1])
    u[1:4] = np.clip(u[1:4], -lim["omega"], lim["omega"])
    return u


def u_to_setpoint(u_phys, cfg, thrust_max_N=None):
    """Model input -> the four arguments of cf.commander.send_setpoint.

    u_phys = [alpha_T, wx, wy, wz] in PHYSICAL units (m/s^2, rad/s), i.e. the
    controller output with u_mean already added back.

    Returns (roll_dps, pitch_dps, yawrate_dps, pwm). With flightmode.stabModeRoll
    and stabModePitch set to 0 the firmware reads roll/pitch as RATES in deg/s.
    cflib negates pitch on the wire, which is what pitch_tx_sign absorbs.
    """
    c = cfg["convert"]
    lim = cfg["limits"]
    tmax = thrust_max_N if thrust_max_N is not None else c["thrust_max_N"]

    u = clip_u(u_phys, cfg)
    alpha_T, w = float(u[0]), u[1:4]

    pwm = thrust_to_pwm(alpha_to_thrust(alpha_T, cfg["crazyflie"]["mass_kg"]),
                        tmax, tuple(lim["pwm"]))

    d = np.degrees(w)
    return (c["roll_tx_sign"] * d[0],
            c["pitch_tx_sign"] * d[1],
            c["yawrate_tx_sign"] * d[2],
            pwm)


def hover_alpha():
    """Specific thrust that holds altitude at zero tilt."""
    return G
