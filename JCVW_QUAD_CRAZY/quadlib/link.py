"""Crazyflie radio link: connect, put the firmware in RATE mode, arm, transmit.

The setting that makes this project different from an attitude-setpoint
controller:

    flightmode.stabModeRoll  = 0     (0 = rate, 1 = angle)
    flightmode.stabModePitch = 0

With those set, cf.commander.send_setpoint(roll, pitch, yawrate, thrust) reads
roll and pitch as BODY RATES in deg/s, which is what the SPC outputs. The values
are read back after writing; if the firmware did not take them the link refuses
to open, because angles interpreted as rates (or vice versa) will crash.

What the firmware does with each channel (crtp_commander_rpyt.c and
controller_pid.c, firmware master):
  roll, pitch  pure rate loop at 500 Hz on the gyro
  yaw          NOT a pure rate loop: the rate setpoint is integrated into a yaw
               ANGLE setpoint and an attitude PID closes around it. Identifying
               on real flight data absorbs this; the MATLAB sim does not model it
  thrust       passed straight to the mixer, no PID

The onboard estimator is not in the control path: roll/pitch rates use the gyro
and the laptop uses Vicon directly. It is still set to the Kalman filter fed with
Vicon pose, so the supervisor's tumble check and the stateEstimate logs are
meaningful.
"""

import time

import numpy as np
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from .convert import u_to_setpoint

# onboard signals. pm.vbat feeds the low-battery abort; the rest are
# diagnostics. None is in the control path.
LOG_FAST = [("gyro.x", "float"), ("gyro.y", "float"), ("gyro.z", "float"),
            ("stabilizer.thrust", "uint16_t")]
LOG_SLOW = [("pm.vbat", "float"),
            ("stateEstimate.x", "float"), ("stateEstimate.y", "float"),
            ("stateEstimate.z", "float")]

PARAMS = [("stabilizer.controller", "1"),      # PID: the only one with RPYT rate mode
          ("stabilizer.estimator", "2"),       # Kalman, fed with Vicon pose
          ("locSrv.extPosStdDev", "0.01"),
          ("locSrv.extQuatStdDev", "0.05"),
          ("flightmode.stabModeRoll", "0"),    # 0 = RATE
          ("flightmode.stabModePitch", "0"),
          ("flightmode.stabModeYaw", "0")]


def require_frame_verified(cfg):
    if not cfg["convert"].get("frame_verified", False):
        raise RuntimeError(
            "convert.frame_verified is false. Run check_frame.py (motors off, "
            "rotate the drone by hand) and set it to true once all three axes pass.")


class Link:
    def __init__(self, cfg):
        self.cfg = cfg
        self.uri = cfg["crazyflie"]["uri"]
        self.telemetry = {"vbat": None, "gyro": np.zeros(3),
                          "thrust": 0, "est": np.zeros(3), "t": 0.0}
        self._scf = None
        self._armed = False

    def __enter__(self):
        cflib.crtp.init_drivers()
        self._scf = SyncCrazyflie(self.uri, cf=Crazyflie(rw_cache="./cache"))
        self._scf.__enter__()
        self.cf = self._scf.cf
        time.sleep(1.0)
        self._set_params()
        self._start_logging()
        self._wait_for_telemetry()
        return self

    def __exit__(self, *exc):
        try:
            self.stop()
            if self._armed:
                self.cf.supervisor.send_arming_request(False)
                time.sleep(0.1)
        finally:
            self._scf.__exit__(*exc)

    # ------------------------------------------------------------------ setup
    def _set_params(self):
        p = self.cf.param
        for name, val in PARAMS:
            p.set_value(name, val)

        # get_value returns cflib's cache, which only updates on the firmware's
        # ack, so poll until every value has come back. Compare numerically:
        # float params echo back in float32 ("0.009999999776...").
        def bad():
            return [(n, v, p.get_value(n)) for n, v in PARAMS
                    if abs(float(p.get_value(n)) - float(v)) > 1e-6]
        t0 = time.time()
        while bad() and time.time() - t0 < 3.0:
            time.sleep(0.05)
        if bad():
            raise RuntimeError(f"firmware did not accept params: {bad()}")
        print("  PID controller, Kalman estimator, RATE mode on roll/pitch/yaw -- verified")

    def reset_estimator(self):
        """Call once Vicon is streaming, so the EKF restarts at the Vicon pose."""
        self.cf.param.set_value("kalman.resetEstimation", "1")
        time.sleep(0.1)
        self.cf.param.set_value("kalman.resetEstimation", "0")
        time.sleep(1.0)

    def _start_logging(self):
        ms = int(self.cfg["loop"]["log_period_ms"])
        self._lc_fast = LogConfig("fast", period_in_ms=ms)
        for n, t in LOG_FAST:
            self._lc_fast.add_variable(n, t)
        self._lc_fast.data_received_cb.add_callback(self._on_fast)

        self._lc_slow = LogConfig("slow", period_in_ms=max(ms * 5, 50))
        for n, t in LOG_SLOW:
            self._lc_slow.add_variable(n, t)
        self._lc_slow.data_received_cb.add_callback(self._on_slow)

        for lc in (self._lc_fast, self._lc_slow):
            self.cf.log.add_config(lc)
            lc.start()

    def _wait_for_telemetry(self, timeout=3.0):
        t0 = time.time()
        while self.telemetry["vbat"] is None:
            if time.time() - t0 > timeout:
                raise RuntimeError("no pm.vbat from the drone -- logging not running")
            time.sleep(0.02)
        print(f"  battery {self.telemetry['vbat']:.2f} V")

    def _on_fast(self, ts, data, _):
        self.telemetry["gyro"] = np.radians([data["gyro.x"], data["gyro.y"],
                                             data["gyro.z"]])
        self.telemetry["thrust"] = data["stabilizer.thrust"]
        self.telemetry["t"] = time.time()

    def _on_slow(self, ts, data, _):
        self.telemetry["vbat"] = data["pm.vbat"]
        self.telemetry["est"] = np.array([data["stateEstimate.x"],
                                          data["stateEstimate.y"],
                                          data["stateEstimate.z"]])

    # ------------------------------------------------------------------- arm
    def arm(self):
        require_frame_verified(self.cfg)
        self.cf.supervisor.send_arming_request(True)
        self._armed = True
        time.sleep(0.5)

    def unlock(self):
        """The RPYT commander keeps thrust locked until it sees one zero-thrust
        setpoint after connecting."""
        self.cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.1)

    # -------------------------------------------------------------- transmit
    def send_u(self, u_phys, thrust_max_N=None):
        """u_phys = [alpha_T, wx, wy, wz] in PHYSICAL units, FLU body rates.
        Returns what was actually put on the wire, for logging."""
        roll, pitch, yawrate, pwm = u_to_setpoint(u_phys, self.cfg, thrust_max_N)
        self.cf.commander.send_setpoint(roll, pitch, yawrate, pwm)
        return roll, pitch, yawrate, pwm

    def stop(self):
        """Motors off. The drone falls from wherever it is."""
        try:
            for _ in range(3):
                self.cf.commander.send_setpoint(0, 0, 0, 0)
                time.sleep(0.02)
            self.cf.commander.send_stop_setpoint()
        except Exception:
            pass
