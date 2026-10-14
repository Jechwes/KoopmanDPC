"""Hover test with plain cflib commands. The drone flies itself.

Checks the chain the SPC scripts depend on, with nothing of the SPC in the
loop: radio link, battery, supervisor state, the Vicon stream for the rigid
body, the Vicon pose reaching the onboard Kalman filter, arming, and a real
takeoff / hover / landing. The flying is done onboard by the high-level
commander and the stock PID, on the Kalman filter fed with Vicon pose -- the
standard Crazyflie mocap setup. If this does not hover, nothing laptop-side
will fly either.

It also measures the hover thrust. convert.thrust_max_N maps specific thrust to
PWM for the laptop PID; the value that makes that map exact at hover, for the
configured mass, is printed at the end.

    python hover.py --no-fly            # connection checks only, never arms
    python hover.py                     # take off to 0.5 m, hover 10 s, land
    python hover.py --height 1.0 --hover-s 20

Ctrl+C lands. Ctrl+C again while landing cuts the motors.
"""

import argparse
import os
import time
import warnings

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from quadlib.convert import G
from quadlib.safety import Safety
from quadlib.util import AbortFlight, load_cfg
from quadlib.vicon import ViconThread

# Flight modes are RAM params: fly.py and collect.py leave roll/pitch in RATE
# mode until the drone is power-cycled. The high-level commander ignores them,
# but a gamepad in cfclient does not, so put back the firmware defaults.
PARAMS = [("stabilizer.estimator", "2"),       # Kalman, fed with Vicon pose
          ("stabilizer.controller", "1"),      # PID
          ("locSrv.extPosStdDev", "0.01"),
          ("locSrv.extQuatStdDev", "0.05"),
          ("flightmode.stabModeRoll", "1"),    # 1 = ANGLE, firmware default
          ("flightmode.stabModePitch", "1")]

LOG_VARS = ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
            "stabilizer.thrust", "pm.vbat"]

VBAT_TAKEOFF = 3.7      # V at rest; below this the in-flight sag trips vbat_min
EKF_TOL = 0.05          # m, Kalman vs Vicon before takeoff is allowed
LEVEL_TOL_DEG = 10.0    # Vicon tilt of the drone sitting on the floor
CLIMB_MPS = 0.25        # takeoff and landing speed

# Firmware older than CRTP protocol 12 has no supervisor port: arming still
# works through cflib's legacy fallback, the state flags just read as unset.
warnings.filterwarnings("ignore", message="The supervisor subsystem requires")


def supervisor_state(cf):
    if cf.platform.get_protocol_version() < 12:
        return "n/a (CRTP protocol < 12, update the firmware to see it)"
    return ", ".join(cf.supervisor.read_state_list()) or "no flags set"


def connect_hint(uri, err):
    """Scan for the configured address, so a URI mismatch is obvious."""
    print(f"\n  could not connect to {uri}: {err}")
    try:
        found = cflib.crtp.scan_interfaces(int(uri.split("/")[-1], 16))
        print("  scan found: " + (", ".join(u for u, _ in found) or "nothing"))
    except Exception as e:
        print(f"  scan failed too: {e}")
    print("  check: drone on, Crazyradio plugged in (WSL: usbipd attach --wsl), "
          "cfclient closed, crazyflie.uri in config.yaml")


def set_params(cf):
    for name, val in PARAMS:
        try:
            cf.param.set_value(name, val)
        except KeyError:
            raise SystemExit(f"param {name} does not exist on this firmware")

    def bad():                                  # cache updates on the firmware's ack
        return [n for n, v in PARAMS
                if abs(float(cf.param.get_value(n)) - float(v)) > 1e-6]
    t0 = time.time()
    while bad() and time.time() - t0 < 3.0:
        time.sleep(0.05)
    if bad():
        raise SystemExit(f"firmware did not accept params: {bad()}")


def start_log(cf, tel, rec):
    lc = LogConfig("hover", period_in_ms=20)
    for name in LOG_VARS:
        lc.add_variable(name, "float")

    def on_data(ts, data, _):
        tel.update(data)
        tel["t"] = time.time()
        if rec["on"]:
            rec["rows"].append([tel["t"]] + [data[n] for n in LOG_VARS])

    lc.data_received_cb.add_callback(on_data)
    cf.log.add_config(lc)
    lc.start()
    t0 = time.time()
    while "pm.vbat" not in tel:
        if time.time() - t0 > 3.0:
            raise SystemExit("no telemetry from the drone -- logging not running")
        time.sleep(0.02)


def wait_for_ekf(vic, tel, timeout=5.0):
    """Kalman estimate within EKF_TOL of Vicon for 0.5 s straight."""
    t0, t_ok = time.time(), None
    while time.time() - t0 < timeout:
        est = np.array([tel[n] for n in LOG_VARS[:3]])
        if np.linalg.norm(est - vic.latest()[1]) < EKF_TOL:
            t_ok = t_ok or time.time()
            if time.time() - t_ok > 0.5:
                return est
        else:
            t_ok = None
        time.sleep(0.02)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=float, default=0.5, help="hover height [m, world z]")
    ap.add_argument("--hover-s", type=float, default=10.0)
    ap.add_argument("--no-fly", action="store_true", help="checks only, motors stay off")
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    uri = cfg["crazyflie"]["uri"]
    host, body = cfg["vicon"]["hostname"], cfg["vicon"]["object_name"]
    lim = cfg["limits"]
    if not 0.2 <= args.height <= lim["geofence_z"][1] - 0.3:
        raise SystemExit(f"--height {args.height} outside [0.2, {lim['geofence_z'][1] - 0.3}] m")

    cflib.crtp.init_drivers()
    print(f"[1] radio      connecting to {uri}")
    try:
        scf = SyncCrazyflie(uri, cf=Crazyflie(rw_cache="./cache"))
        scf.open_link()
    except Exception as e:
        connect_hint(uri, e)
        raise SystemExit(1)

    cf = scf.cf
    tel, rec = {}, {"on": False, "rows": []}
    vic = None
    try:
        print(f"               connected, CRTP protocol {cf.platform.get_protocol_version()}")
        set_params(cf)
        start_log(cf, tel, rec)
        print(f"[2] battery    {tel['pm.vbat']:.2f} V")
        print(f"[3] supervisor {supervisor_state(cf)}")

        print(f"[4] vicon      {host}, rigid body '{body}'")
        vic = ViconThread(host, body, cf=cf)             # also feeds the Kalman filter
        vic.start()
        if not vic.wait_for_pose(5.0):
            raise SystemExit(
                f"no pose: Vicon streamed {vic.n_missing} frames without '{body}'"
                if vic.n_missing else f"no frames from {host} -- network / DataStream?")
        n0 = vic.n_frames
        time.sleep(1.0)
        _, p0, rpy0 = vic.latest()
        tilt0 = np.degrees(np.arccos(np.cos(rpy0[0]) * np.cos(rpy0[1])))
        print(f"               {vic.n_frames - n0} Hz, pos {np.round(p0, 3)} m, "
              f"yaw {np.degrees(rpy0[2]):+.1f} deg, tilt {tilt0:.1f} deg")
        if tilt0 > LEVEL_TOL_DEG:
            raise SystemExit("drone not level according to Vicon -- on flat ground? "
                             "rigid body z-axis up?")

        print("[5] kalman     reset, waiting for it to lock onto Vicon")
        cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        cf.param.set_value("kalman.resetEstimation", "0")
        est = wait_for_ekf(vic, tel)
        if est is None:
            raise SystemExit("Kalman estimate did not converge to the Vicon pose -- "
                             "is the pose reaching the drone?")
        print(f"               estimate {np.round(est, 3)} m -- ok")

        if not cfg["convert"].get("frame_verified", False):
            print("\n  NOTE: convert.frame_verified is false. The Kalman filter uses the "
                  "Vicon orientation;\n  if the flight drifts or circles, run "
                  "check_frame.py before anything else.")
        low_bat = tel["pm.vbat"] < VBAT_TAKEOFF
        if args.no_fly:
            print("\n--no-fly: all checks passed, motors never armed"
                  + (f" (battery below {VBAT_TAKEOFF} V: charge before flying)" if low_bat else ""))
            return
        if low_bat:
            raise SystemExit(f"battery {tel['pm.vbat']:.2f} V < {VBAT_TAKEOFF} V at rest -- charge it")

        fly(cf, vic, tel, rec, cfg, args, p0)
    finally:
        if vic is not None:
            vic.stop()
        scf.close_link()

    report(rec["rows"], cfg)


def fly(cf, vic, tel, rec, cfg, args, p0):
    hl = cf.high_level_commander
    safe = Safety(cfg, p0)
    target = np.array([p0[0], p0[1], args.height])
    errs, gaps = [], []

    def watch(duration, log_err=False):
        t_end, t_prev = time.time() + duration, time.time()
        while time.time() < t_end:
            pose_t, pos, rpy = vic.latest()
            now = time.time()
            why = safe.check(pose_t, pos, rpy, tel["pm.vbat"], now - t_prev,
                             enforce_floor=False)
            gaps.append(now - t_prev)
            t_prev = now
            if why:
                raise AbortFlight(why)
            if now - tel["t"] > 0.5:
                raise AbortFlight("no telemetry for 0.5 s -- radio link")
            if log_err:
                errs.append(pos - target)
            time.sleep(0.02)

    print("\n[6] arming")
    cf.supervisor.send_arming_request(True)
    time.sleep(1.0)
    print(f"               supervisor: {supervisor_state(cf)}")

    up_s = max(2.0, (args.height - p0[2]) / CLIMB_MPS)
    try:
        print(f"[7] takeoff    to {args.height:.2f} m in {up_s:.1f} s")
        hl.takeoff(args.height, up_s, yaw=None)          # keep the current heading
        watch(up_s + 1.5)
        if vic.latest()[1][2] < 0.5 * args.height:
            raise AbortFlight("did not climb -- not armed, or supervisor locked "
                              "(power-cycle after a crash)")
        print(f"[8] hover      {args.hover_s:.0f} s at {np.round(target, 2)}")
        rec["on"] = True
        watch(args.hover_s, log_err=True)
        rec["on"] = False
    except (Exception, KeyboardInterrupt) as e:
        # anything at all: the high-level commander would keep hovering on its
        # own after this script exits, so always land
        rec["on"] = False
        print(f"\n*** {e if str(e) else 'Ctrl+C'} -- landing ***")

    try:
        z = vic.latest()[1][2]
        down_s = max(2.0, z / CLIMB_MPS)
        print(f"[9] landing    from {z:.2f} m in {down_s:.1f} s")
        hl.land(0.0, down_s, yaw=None)
        time.sleep(down_s + 1.0)
    except KeyboardInterrupt:
        print("\n*** Ctrl+C while landing -- motors off ***")
    finally:
        hl.stop()
        time.sleep(0.1)
        cf.supervisor.send_arming_request(False)
        time.sleep(0.2)

    if errs:
        e = np.array(errs)
        print(f"\n    hover error (Vicon): RMS xy {np.sqrt((e[:, :2]**2).sum(1).mean()):.3f} m, "
              f"RMS z {np.sqrt((e[:, 2]**2).mean()):.3f} m, "
              f"max {np.linalg.norm(e, axis=1).max():.3f} m")
    print(f"    vicon frames {vic.n_frames}, frames without '{vic.object_name}' {vic.n_missing}")
    if gaps:
        g = np.array(gaps)          # this loop sleeps 20 ms: > 60 ms is a stall
        print(f"    laptop stalls: {(g > 0.06).sum()} over 60 ms, longest {1e3 * g.max():.0f} ms"
              + ("  <- SPC needs its 40 ms steps on time: close other programs"
                 if (g > 0.06).any() else ""))


def report(rows, cfg):
    """Thrust calibration from the steady part of the hover."""
    if len(rows) < 100:
        return
    a = np.asarray(rows)[50:]                   # drop the first second
    pwm, vbat = a[:, 4].mean(), a[:, 5].mean()
    if pwm < 1000:
        return
    c, m = cfg["convert"], cfg["crazyflie"]["mass_kg"]
    tmax = m * G / (pwm / 65535.0)                    # inverts thrust_to_pwm
    print(f"\n    hover thrust {pwm:.0f} PWM (std {a[:, 4].std():.0f}) at {vbat:.2f} V")
    print(f"    -> convert.thrust_max_N: {tmax:.3f}  (config has {c['thrust_max_N']}, "
          f"for crazyflie.mass_kg {m})")

    os.makedirs("logs", exist_ok=True)
    path = f"logs/hover_{time.strftime('%Y%m%d_%H%M%S')}.npz"
    np.savez(path, t=a[:, 0], est=a[:, 1:4], thrust=a[:, 4], vbat=a[:, 5])
    print(f"    saved -> {path}")


if __name__ == "__main__":
    main()
