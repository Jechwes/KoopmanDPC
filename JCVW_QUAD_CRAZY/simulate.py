"""Dry run: the real collect.py / identify.py / fly.py against a simulated drone.

Nothing in the flight scripts is changed or bypassed. This installs stand-ins
for `cflib` and `motioncapture` that reproduce their behaviour as the flight
code sees it, backed by the MATLAB plant ported to Python (quadlib/sim.py)
running in real time:

  cflib   send_setpoint packs (roll, -pitch, yawrate, thrust) as cflib 0.1.32
          does; the "firmware" then applies crtp_commander_rpyt.c in RATE mode
          (thrust lock, yaw input inverted) and controller_pid.c's rate-loop
          conventions (tracks -gyro.y on pitch), and maps PWM to thrust with the
          same linear model convert.py inverts. Refuses rate commands unless
          flightmode.stabMode* were set to RATE, like a real drone would do the
          wrong thing silently.
  vicon   rigid body with .position and .rotation.{x,y,z,w} at 100 Hz

So a sign error in config.yaml shows up here as the drone flipping, exactly as
it would on the real one -- which is the point.

    python simulate.py all              # collect -> identify -> fly, ~6 min
    python simulate.py collect --runs 3
    python simulate.py fly --traj fig8 [--no-phi]

`all` collects 4 x 60 s: with Tini = N = 30, 4 x 30 s was too little data and
the model diverged.

Flights and models go to data/sim/, never mixed with real ones.
"""

import os
import sys
import tempfile
import threading
import time
import types

import numpy as np
import yaml

from quadlib.sim import quad_params, quad_plant

SIM_DIR = "data/sim"


# ============================================================ physics thread
class _Physics(threading.Thread):
    DT = 0.01                               # publish period, 100 Hz

    def __init__(self, cfg):
        super().__init__(daemon=True)
        self.prm, self.lim, self.inner = quad_params()
        self.cfg = cfg
        self.x = np.zeros(13); self.x[6] = 1.0      # on the ground, level, yaw 0
        self.st = None
        self.vbat = 4.0
        self.lock = threading.Lock()
        self.frame = threading.Event()
        self.logs = []
        self.params = {}
        self.armed, self.thrust_locked = False, True
        self.pkt = (0.0, 0.0, 0.0, 0)
        self.running = True

    # ---- firmware: crtp_commander_rpyt.c + controller_pid.c, RATE mode
    def command(self):
        roll, pitch, yaw, thrust = self.pkt
        if thrust == 0:
            self.thrust_locked = False
        if not self.armed or self.thrust_locked or thrust < 1000:
            return None
        # only matters once the motors are producing thrust, as on the real drone
        for ax in ("Roll", "Pitch", "Yaw"):
            if self.params.get(f"flightmode.stabMode{ax}") != "0":
                raise RuntimeError(f"sim firmware: stabMode{ax} is not RATE")
        pwm = min(thrust, 60000)
        c = self.cfg["convert"]
        F = pwm / 65535.0 * c["thrust_max_N"]      # firmware compensates the battery
        alpha_T = F / self.prm["m"]
        # rateDesired.roll = roll ; rateDesired.pitch = pitch tracks -gyro.y ;
        # attitudeRate.yaw = -yaw ("legacy rate input is inverted")
        w = np.radians([roll, -pitch, -yaw])
        return np.r_[alpha_T, w]

    def run(self):
        t_next = time.perf_counter()
        t0 = t_next
        while self.running:
            with self.lock:
                v = self.command()
                if v is None:                        # motors off: fall
                    self.x[3:6] += np.array([0, 0, -self.prm["g"]]) * self.DT
                    self.x[0:3] += self.x[3:6] * self.DT
                    self.st = None
                else:
                    self.x, self.st = quad_plant(self.x, v, self.prm, self.lim,
                                                 self.inner, self.DT, 0.002, self.st)
                if self.x[2] <= 0.0:                 # ground contact
                    yaw = np.arctan2(2*(self.x[6]*self.x[9] + self.x[7]*self.x[8]),
                                     1 - 2*(self.x[8]**2 + self.x[9]**2))
                    self.x[2] = 0.0; self.x[3:6] = 0.0; self.x[10:13] = 0.0
                    self.x[6:10] = [np.cos(yaw/2), 0, 0, np.sin(yaw/2)]
                    self.st = None
                x = self.x.copy()
            self.frame.set()

            now = time.perf_counter()
            for lc in list(self.logs):
                if now >= lc._next:
                    lc._next = now + lc.period_in_ms / 1000.0
                    lc._fire(x, self, int(1e3 * (now - t0)))

            t_next += self.DT
            s = t_next - time.perf_counter()
            if s > 0:
                time.sleep(s)


_PHYS = None


# ================================================================ cflib mock
def _install_mocks(cfg):
    global _PHYS
    _PHYS = _Physics(cfg)

    class Param:
        def set_value(self, name, value):
            _PHYS.params[name] = str(value)

        def get_value(self, name, timeout=60):
            return _PHYS.params[name]

    class Callback:
        def __init__(self):
            self.cbs = []

        def add_callback(self, cb):
            self.cbs.append(cb)

    class LogConfig:
        def __init__(self, name, period_in_ms):
            self.name, self.period_in_ms = name, period_in_ms
            self.vars, self.data_received_cb = [], Callback()
            self._next = 0.0

        def add_variable(self, name, fetch_as=None):
            self.vars.append(name)

        def start(self):
            _PHYS.logs.append(self)

        def stop(self):
            if self in _PHYS.logs:
                _PHYS.logs.remove(self)

        def _fire(self, x, ph, ts):
            src = {"gyro.x": np.degrees(x[10]), "gyro.y": np.degrees(x[11]),
                   "gyro.z": np.degrees(x[12]), "stabilizer.thrust": ph.pkt[3],
                   "pm.vbat": ph.vbat, "stateEstimate.x": x[0],
                   "stateEstimate.y": x[1], "stateEstimate.z": x[2]}
            data = {v: src[v] for v in self.vars}
            for cb in self.data_received_cb.cbs:
                cb(ts, data, self)

    class Log:
        def add_config(self, lc):
            pass

    class Commander:
        def send_setpoint(self, roll, pitch, yawrate, thrust):
            if thrust > 0xFFFF or thrust < 0:            # as cflib
                raise ValueError("Thrust must be between 0 and 0xFFFF")
            with _PHYS.lock:
                _PHYS.pkt = (float(roll), -float(pitch), float(yawrate), int(thrust))

        def send_stop_setpoint(self):
            with _PHYS.lock:
                _PHYS.pkt, _PHYS.thrust_locked = (0.0, 0.0, 0.0, 0), True

    class Supervisor:
        def send_arming_request(self, do_arm):
            _PHYS.armed = bool(do_arm)

    class Extpos:
        def send_extpose(self, *a):
            pass

    class Crazyflie:
        def __init__(self, rw_cache=None):
            self.param, self.log = Param(), Log()
            self.commander, self.supervisor, self.extpos = Commander(), Supervisor(), Extpos()

    class SyncCrazyflie:
        def __init__(self, uri, cf=None):
            self.cf = cf or Crazyflie()

        def __enter__(self):
            if not _PHYS.is_alive():
                _PHYS.start()
            return self

        def __exit__(self, *exc):
            time.sleep(0.2)

    class RigidBody:
        def __init__(self, x):
            self.position = x[0:3].copy()
            self.rotation = types.SimpleNamespace(w=x[6], x=x[7], y=x[8], z=x[9])

    class MotionCapture:
        def __init__(self, name):
            self.name = name

        def waitForNextFrame(self):
            _PHYS.frame.wait(); _PHYS.frame.clear()

        @property
        def rigidBodies(self):
            with _PHYS.lock:
                return {self.name: RigidBody(_PHYS.x)}

    name = cfg["vicon"]["object_name"]
    mods = {
        "cflib": types.ModuleType("cflib"),
        "cflib.crtp": types.SimpleNamespace(init_drivers=lambda: None),
        "cflib.crazyflie": types.SimpleNamespace(Crazyflie=Crazyflie),
        "cflib.crazyflie.log": types.SimpleNamespace(LogConfig=LogConfig),
        "cflib.crazyflie.syncCrazyflie": types.SimpleNamespace(SyncCrazyflie=SyncCrazyflie),
        "motioncapture": types.SimpleNamespace(connect=lambda kind, opts: MotionCapture(name)),
    }
    mods["cflib"].crtp = mods["cflib.crtp"]
    sys.modules.update(mods)


# ================================================================== driver
def _sim_config(base="config.yaml", **over):
    with open(base) as f:
        cfg = yaml.safe_load(f)
    cfg["crazyflie"]["mass_kg"] = quad_params()[0]["m"]   # the simulated airframe
    cfg["convert"]["frame_verified"] = True               # true by construction here
    for k, v in over.items():
        sec, key = k.split("__")
        cfg[sec][key] = v
    fd, path = tempfile.mkstemp(suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return cfg, path


def _run(module, argv, cfg):
    _install_mocks(cfg)
    sys.argv = [module + ".py"] + argv
    mod = __import__(module)
    mod.main()
    _PHYS.running = False


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("collect", "identify", "fly", "all"):
        sys.exit(__doc__)
    what, rest = sys.argv[1], sys.argv[2:]
    os.makedirs(f"{SIM_DIR}/flights", exist_ok=True)

    if what in ("collect", "all"):
        # --duration S shortens each run (sim only); a real collect uses config
        dur = None
        if "--duration" in rest:
            i = rest.index("--duration"); dur = float(rest[i + 1]); rest = rest[:i] + rest[i+2:]
        cfg, path = _sim_config(collect__duration_s=dur) if dur else _sim_config()
        args = ["--config", path, "--tag", "sim", "--outdir", f"{SIM_DIR}/flights"]
        args += ["--runs", "4"] if what == "all" else rest
        _run("collect", args, cfg)

    if what in ("identify", "all"):
        sys.argv = ["identify.py", "--flights", f"{SIM_DIR}/flights/*.npz",
                    "--outdir", f"{SIM_DIR}/models"]
        __import__("identify").main()

    if what in ("fly", "all"):
        cfg, path = _sim_config()
        phi = "kspc" if cfg["model"]["use_phi"] and "--no-phi" not in rest else "spc"
        args = ["--config", path, "--model", f"{SIM_DIR}/models/theta_{phi}.npz"]
        args += ["--traj", "fig8", "--duration", "20"] if what == "all" else rest
        _run("fly", args, cfg)


if __name__ == "__main__":
    main()
