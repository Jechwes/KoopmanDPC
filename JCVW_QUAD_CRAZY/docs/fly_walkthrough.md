# fly.py, line by line

This walkthrough follows `fly.py` in the order Python runs it when you type

```bash
python fly.py --traj fig8
```

When `fly.py` calls into another file for the first time, a **↪ Detour** explains
that function, then the walkthrough continues in `fly.py`. Later calls to the same
function only point back to the detour.

Line numbers match the code on 5 October 2026. They shift when you edit a file.

## The flight in one picture

1. Read the options, the config and the model; build the controller and the reference.
2. Connect the radio, switch the firmware to rate mode, start telemetry and the Vicon
   thread, reset the drone's estimator, arm.
3. Laptop PID: take off and climb to the start point (4.5 s), then hold it (2.4 s)
   while logging.
4. SPC/KSPC: 1000 steps of 40 ms, each one *read Vicon → build z → solve the QP →
   send → store*.
5. Laptop PID: land, motors off. Save the log and print the statistics.

## Contents

1. [Start-up and options](#1-start-up-and-options): fly.py 1–47, 180–181
2. [Config, model and reference](#2-config-model-and-reference): fly.py 49–58
3. [Checks before flying](#3-checks-before-flying): fly.py 60–79
4. [Connecting to the drone and Vicon](#4-connecting-to-the-drone-and-vicon): fly.py 81–92
5. [Takeoff on the PID, and the first commands](#5-takeoff-on-the-pid-and-the-first-commands): fly.py 94–102
6. [Handover to the SPC](#6-handover-to-the-spc): fly.py 104–114
7. [The control loop](#7-the-control-loop): fly.py 115–150
8. [Landing and errors](#8-landing-and-errors): fly.py 152–158
9. [After the flight](#9-after-the-flight): fly.py 160–177

---

## 1. Start-up and options

- **L1–20** Docstring: documentation only. It summarises the loop order, the
  PID → SPC → PID handover and three example commands.

**Imports (L22–33).** Importing a module runs its top level, which only defines
functions, classes and constants. Nothing connects yet.

- **L22** `import argparse`: reads command-line options such as `--traj fig8_fast`.
- **L23** `import time`: `perf_counter` (a precise clock) and `sleep` for the 25 Hz
  loop, `strftime` for the log file name.
- **L25** `import numpy as np`: arrays and maths.
- **L27** `from quadlib import reference, spc`: `reference.py` provides the reference
  trajectories, `spc.py` the controller and its QP.
- **L28** `from quadlib.lift import lift`: builds the lifted state z(k).
- **L29** `from quadlib.link import Link, require_frame_verified`: the radio link to
  the drone, and the frame-check guard.
- **L30** `from quadlib.safety import Safety`: the abort checks.
- **L31–32** `from quadlib.util import ...`:
  - `AbortFlight`: the error that ends a flight;
  - `FlightLog`: records every step;
  - `VelEstimator`: velocity for the PID;
  - `abort`, `land`: the emergency stop and the landing;
  - `load_cfg`: reads `config.yaml`;
  - `pid_fly`: flies with the laptop PID.
- **L33** `from quadlib.vicon import ViconThread`: receives Vicon poses in a
  background thread.

These imports also load the libraries behind them:
- `util.py` imports `convert.py` and `pid.py`;
- `link.py` imports `cflib` (the Crazyflie library);
- `vicon.py` imports `motioncapture` (the Vicon client);
- `spc.py` imports `osqp` and `scipy`.

**Starting main.**
- **L36** `def main():` defines the main function. Its body doesn't run yet.
- **L180–181** `if __name__ == "__main__": main()` is true when the file is run
  directly rather than imported, so `main()` starts, at line 37.

**Command-line options (L37–47).**
- **L37** `ap = argparse.ArgumentParser()`: creates the option parser.
- **L38** `--traj`, default `fig8`: the reference. Options are `hover`, `step`,
  `fig8_slow`, `fig8`, `fig8_fast` and `circle_z`.
- **L39** `--no-phi`: forces plain SPC even if the config says KSPC.
- **L40** `--model`: uses a different model file than the default.
- **L41** `--duration`, default 40 s: how long the SPC/KSPC flies.
- **L42–43** `--ramp-s`, default 4 s: eases the reference in from standstill;
  0 switches this off.
- **L44–45** `--speed`, default 1.0: flies the same path faster or slower.
- **L46** `--config`, default `config.yaml`: which config file to read.
- **L47** `args = ap.parse_args()`: reads what you typed into `args`. Dashes become
  underscores, so `--ramp-s` becomes `args.ramp_s`.

---

## 2. Config, model and reference

- **L49** `cfg = load_cfg(args.config)`

  #### ↪ Detour: `load_cfg`, quadlib/util.py 14–16
  - **L15** Opens the file.
  - **L16** `yaml.safe_load` turns it into nested dictionaries, so for example
    `cfg["model"]["Tini"]` is 30.

- **L50** `require_frame_verified(cfg)`

  #### ↪ Detour: `require_frame_verified`, quadlib/link.py 55–59
  - **L56–59** If `convert.frame_verified` in the config is not `true`, it stops
    with an error that says to run `check_frame.py` first. This prevents flying with
    unverified sign conventions.

- **L51** `use_phi = cfg["model"]["use_phi"] and not args.no_phi`: KSPC if the
  config says so, unless you passed `--no-phi`.
- **L52** `model = ...`: the model file. It is `--model` if given, otherwise
  `models/theta_kspc.npz` or `models/theta_spc.npz`, both written by `identify.py`.
- **L54** `ctrl, par, u_mean, y_mean = spc.load(model, cfg)`

  #### ↪ Detour: `load`, quadlib/spc.py 99–123
  - **L108** Opens the model file. It holds Ψ (`Psi`), Γ (`Gam`), the data means
    `u_mean` and `y_mean`, and `par`. `allow_pickle` is needed because `par` is a
    dictionary.
  - **L109** `par`: the dimensions `Tini`, `N`, `Ts`, `m`, `p`, `use_phi`, `n_phi`,
    `um1` (the hover thrust), `n_ini` and `L`.
  - **L110** `u_mean`: the mean input of the data, about [9.87, 0, 0, 0]. Its first
    entry is the hover trim ᾱ_T.
  - **L112** `input_box(cfg["limits"])` (spc.py **92–96**) builds the actuator clip:
    α_T between 4.9 and 15.7 m/s² and each rate within ±3 rad/s.
  - **L113–117** Intersects that clip with `spc_box`, the box from the thesis:
    - α_T between 0.683 and 1.317 times the hover trim, so 6.74 to 13.0 m/s²;
    - rates within ±0.785 rad/s.

    `lo` and `hi` are the tighter of the two.
  - **L119–122** Builds the controller. The box is shifted by `u_mean`, because the
    QP works with mean-removed inputs. The tolerance comes from `solver.osqp_eps`
    (1e-3).

    #### ↪ Detour inside the detour: `SPCController.__init__`, spc.py 32–59
    The QP is solved for **x = [u; y]**: all 30 future inputs u (30 × 4 = 120
    numbers) and all 30 predicted outputs y (120 numbers). It is written as
    *minimise ½ xᵀPx + qᵀx with l ≤ Ax ≤ u*.
    - **L34** Stores Ψ, Γ and `par`.
    - **L35–36** N = 30 and m = p = 4, so there are 120 inputs and 120 outputs over
      the horizon.
    - **L38** Q = I_N ⊗ diag(Qy): the output weight [10, 10, 10, 1], repeated for
      each of the 30 steps. It is a sparse 120 × 120 matrix.
    - **L39** R = I_N ⊗ diag(Ru): the same for the input weight.
    - **L42** P = blkdiag(2R, 2Q). Only the upper triangle is kept, because OSQP
      expects that.
    - **L43–46** A = [[Γ, −I], [I, 0]].
      - The first row block writes the prediction y = Ψz + Γu as Γu − y = −Ψz.
      - The second row block picks out u for the input box.
    - **L48–49** `lo` and `hi` are repeated for all 30 steps.
    - **L51–56** OSQP is set up once, with warm-starting, polishing and the
      tolerance. The zeros are placeholders that every step overwrites. OSQP
      factorises its system matrix here, so each later solve only changes vectors.
    - **L58–59** `u_prev` (held if a solve fails) and a failure counter.

    ↩ back to `load`.
  - **L123** Returns `ctrl`, `par`, `u_mean` and `y_mean`.

  ↩ back to fly.py.

- **L55** `Ts, Tini`: 0.04 s and 30, taken from the model (the values it was
  identified with).
- **L56** `hover = [0, 0, 1.0]` from `collect.hover`: the centre of the reference
  and of the geofence.
- **L57** `traj = reference.make(args.traj, hover, args.ramp_s, args.speed)`

  #### ↪ Detour: `make`, quadlib/reference.py 10–18
  - **L14** `_make(name, hover)` (reference.py **57–104**) returns a dictionary of
    functions of time: `pos(t)`, `vel(t)`, `acc(t)`, `yaw(t)`, plus a `name`.
    - For a figure 8 (**L73–86**): A = 0.8, B = 0.5, C = 0.25 m, and ω = 0.4, 0.8 or
      1.1 rad/s depending on the name.
      - pos = hover + (A sin ωt, B sin 2ωt, C sin ωt);
      - `vel` and `acc` are its derivatives;
      - yaw stays 0.
    - The other references are `hover` (**L58–63**), `step` (**L65–71**, a 0.5 m step
      in x at t = 5 s) and `circle_z` (**L88–102**). An unknown name raises an error
      (**L104**).
  - **L15–16** If `--speed` isn't 1, `faster` (**L21–29**) plays the same path s
    times as fast:
    - pos(s·t), with velocity × s and acceleration × s²;
    - the name gets `_x<s>`, so logs of different speeds never mix.
  - **L17–18** Periodic references (the figure 8s and `circle_z`) get `ramp_in`
    (**L32–54**), unless `--ramp-s 0`.
    - It is a time warp: the playback speed s(t) = (1 − cos(πt/T))/2 rises from 0 to
      1 over T = 4 s.
    - So the reference starts at standstill and gently speeds up.
    - After 4 s it is exactly the figure 8, delayed by 2 s. Velocity and
      acceleration follow from the chain rule (**L52–53**).

  ↩ back to fly.py.

- **L58** `name`: `"kspc"` or `"spc"`, read from the model. It is used in the
  prints and the log file name.

---

## 3. Checks before flying

- **L60** Prints the model file, the type, the size L of z, and Ts.
- **L61–63** Warns if the model was identified with a different `Tini`/`N` than
  `config.yaml` now says. In that case you should re-run `identify.py`.
- **L64** `tilt = reference.peak_tilt_deg(...)`

  #### ↪ Detour: `peak_tilt_deg`, reference.py 107–115
  Samples the reference acceleration every 20 ms over the flight. The tilt the drone
  needs is atan2(|a_xy|, g + a_z); it returns the maximum, in degrees.

- **L65** `rate = reference.peak_tilt_rate(...)`

  #### ↪ Detour: `peak_tilt_rate`, reference.py 118–125
  Computes the thrust direction (a + g·e₃)/|a + g·e₃| every 10 ms, and returns the
  fastest it turns, in rad/s. That is the body rate the reference alone needs.

- **L66** Prints both.
- **L67** `box`: the rate limit the SPC may command (0.785 rad/s).
- **L68–71** Warns if the reference alone needs more than 90 % of that limit,
  which leaves no room for feedback.
- **L72–76** Warns if the reference tilts more than 99 % of the identification
  data, because the model would then extrapolate. The current model files predate
  the `tilt_p99_deg` field, so for them this check is skipped.
- **L78** `log = FlightLog()` (util.py **137–138**): an empty list of rows.
- **L79** `solve_ms` (the solve times), `n_fail = 0`, and K = 40 / 0.04 = 1000
  steps.

---

## 4. Connecting to the drone and Vicon

- **L81** `with Link(cfg) as link:`
  - This creates the link and calls its `__enter__`.
  - When the block ends, normally or through an error, `__exit__` runs (see
    section 8).

  #### ↪ Detour: `Link`, quadlib/link.py
  - **L63–69** `__init__`:
    - Stores the config and the address `radio://0/80/2M/E7E7E7E7B4`, which means
      Crazyradio 0, channel 80, 2 Mbit/s, and the drone's address.
    - Creates an empty telemetry dictionary.
    - Marks the drone as not armed.
  - **L71–80** `__enter__`:
    - **L72** Finds the Crazyradio USB dongle.
    - **L73–75** Connects. This waits until the drone's parameter and log tables are
      downloaded; they are cached in `./cache` to speed up the next connection.
    - **L76** Waits 1 s.
    - **L77** `_set_params` (**L92–108**):
      - Writes `PARAMS` (**L46–52**): the PID controller, the Kalman estimator, the
        trust in the Vicon data, and rate mode for roll, pitch and yaw.
      - Polls for up to 3 s until the drone echoes every value back, and stops with
        an error if it doesn't. If roll and pitch weren't switched to rates, the
        drone would read the rate commands as angles and crash.
    - **L78** `_start_logging` (**L117–131**) sets up two telemetry streams. They
      arrive in cflib's background thread, so `link.telemetry` always holds the
      latest values.
      - "fast", every 10 ms: gyro and thrust, handled by `_on_fast` (**L141–145**),
        which stores the gyro in rad/s.
      - "slow", every 50 ms: battery voltage and the onboard position estimate,
        handled by `_on_slow` (**L147–151**).
    - **L79** `_wait_for_telemetry` (**L133–139**) waits up to 3 s for the first
      battery voltage and prints it.
    - **L80** Returns itself, which becomes `link`.

  ↩ back to fly.py.

- **L82–83** `vic = ViconThread(...)`

  #### ↪ Detour: `ViconThread.__init__`, quadlib/vicon.py 21–31
  - A daemon thread, so it stops with the program.
  - It stores the Vicon host (`10.128.7.250`) and the object name (`cfb4`).
  - It will also feed the drone's Kalman filter, because it got `cf`.
  - It also creates a lock, an empty pose and frame counters.

- **L84** `vic.start()` runs `run()` in the background:

  #### ↪ Detour: `ViconThread.run`, vicon.py 33–56
  - **L34** Connects to the Vicon server.
  - **L35–36** Loops, each time waiting for the next frame (100 Hz).
  - **L37–39** Skips the frame if `cfb4` isn't in it, and counts it as missing.
  - **L40–42** Reads the position in m and the orientation as a quaternion.
  - **L44–49** Sends the pose to the drone's Kalman filter (`send_extpose`). Errors
    are ignored, so a radio hiccup never kills the thread.
  - **L51** Quaternion → roll, pitch, yaw (convert.py **15–25**, ZYX convention).
  - **L52–56** Under the lock, so the main loop never reads half a pose:
    - unwraps the yaw (convert.py **28–31**): it adds the smallest angle change, so
      yaw never jumps between +π and −π;
    - stores (time, position, roll/pitch/yaw).

- **L85–86** Waits up to 5 s for the first pose (vicon.py **63–69**, using `latest`
  at **58–61**). Otherwise it stops with "no Vicon pose".
- **L87** `link.reset_estimator()` (link.py **110–115**): restarts the drone's
  Kalman filter at the current Vicon pose, then waits 1 s.
- **L89** `vel = VelEstimator()` (util.py **24–29**): starts at zero velocity, with
  a 0.08 s filter time constant.
- **L90** `safe = Safety(cfg, hover)` (safety.py **12–21**) stores the limits:
  - a geofence of ±1.5 m around hover in x and y, and z between 0.3 and 2.0 m;
  - a maximum tilt of 40°;
  - a Vicon timeout of 0.15 s;
  - a minimum battery voltage of 3.3 V;
  - 60 ms (1.5 × Ts) as the threshold for an overrun.
- **L91** `link.arm()` (link.py **154–158**): checks the frame flag again, asks the
  firmware's supervisor to arm (it won't spin the motors otherwise), and waits
  0.5 s.
- **L92** `link.unlock()` (link.py **160–164**): sends one all-zero command. The
  firmware ignores thrust until it has seen a zero-thrust command after connecting,
  which is a safety lock.

---

## 5. Takeoff on the PID, and the first commands

- **L94** `try:`: an abort or Ctrl+C from here on is handled at L155.
- **L96** `start = traj["pos"](0.0)`: where the reference starts. For a figure 8
  that is the hover point (0, 0, 1.0).
- **L97** `p0`: the current position, on the ground.
- **L98–99** `pid_fly(...)` flies the PID for 4.5 s. Its target moves in a straight
  line from `p0` to `start` over 4 s, with yaw 0.

  #### ↪ Detour: `pid_fly`, quadlib/util.py 50–96. The first commands go out here.
  - **L59–62** Ts = 1/25 = 0.04 s, the PID gains, the clock and a step counter.
  - **L63–66** Loops until `duration` has passed.
  - **L68–71** Gets the latest Vicon pose, and aborts if there is none.
  - **L72–76** Runs the safety check, without the floor check because takeoff
    starts below 0.3 m. It raises `AbortFlight` if anything is wrong.

    #### ↪ Detour: `Safety.check`, quadlib/safety.py 23–47
    It returns a reason to abort (or `None` if all is fine) in these cases:
    - the Vicon pose is older than 0.15 s;
    - x or y is more than 1.5 m from hover;
    - z is above 2.0 m, or below 0.3 m when the floor check is on;
    - tilt = arccos(cos roll · cos pitch) is more than 40°;
    - the battery is below 3.3 V.

    A step longer than 60 ms is only counted as an overrun; it doesn't abort.
  - **L77** Remembers this step's time.
  - **L79** `vel.update` (util.py **31–37**): the position difference divided by the
    time step, low-pass filtered with weight a = dt/(τ + dt).
  - **L80** Gets the target position and yaw from the lambda.
  - **L81** Computes the PID command. `rot_from_rpy` (util.py **40–47**) first builds
    the body-to-world rotation R = Rz(yaw)·Ry(pitch)·Rx(roll).

    #### ↪ Detour: `outer_pid`, quadlib/pid.py 25–56
    - **L27–29** Gains: Kp = [3, 3, 6], Kd = [2.6, 2.6, 4], tilt limit 35°.
    - **L32** Desired acceleration = Kp·(position error) + Kd·(velocity error)
      + gravity.
    - **L33** Never less than 0.3 g upwards.
    - **L36–39** Limits the horizontal part, so the commanded tilt stays under 35°.
    - **L43** Specific thrust α_T = the desired acceleration projected on the
      drone's *current* body z-axis. It only thrusts what helps in the current
      attitude.
    - **L46–51** Desired attitude: body z along the desired acceleration, heading
      from the yaw target.
    - **L54–55** Attitude error e_R, from the difference between the two rotation
      matrices.
    - **L56** Returns u = [α_T, −K_att·e_R] with K_att = 4. So the body-rate command
      is proportional to the attitude error.
  - **L82–83** Adds dither (only `collect.py` uses this).
  - **L84** `clip_u` (convert.py **59–71**): clips α_T to between 4.9 and
    15.7 m/s², and the rates to ±3 rad/s.
  - **L86** `link.send_u(u)` sends the command:

    #### ↪ Detour: `send_u`, link.py 167–172 → `u_to_setpoint`, convert.py 74–98
    - **L84–86** The config sections, and T_max = 0.78 N.
    - **L88–89** Clips again (a safety net) and splits α_T and ω.
    - **L91–92** Thrust command:
      - thrust = α_T·m with m = 0.0375 kg (`alpha_to_thrust`, **L49–55**);
      - PWM = thrust / T_max · 65535, clipped to 10001–60000 (`thrust_to_pwm`,
        **L35–46**). Hover comes out at about 31 100.
    - **L94** rad/s → deg/s.
    - **L95–98** Signs:
      - roll +1;
      - pitch +1: cflib negates pitch and the firmware tracks −gyro.y, and the two
        cancel;
      - yaw −1: the firmware inverts the yaw input.
    - link.py **L171** `cf.commander.send_setpoint(roll, pitch, yawrate, pwm)`:
      cflib packs these four numbers into one radio packet. On the drone, roll and
      pitch rates go to the rate PID, the yaw rate is integrated into a heading
      setpoint, and the PWM goes to the mixer.
  - **L87–88** Logs the step, if a log was passed in.
  - **L90–96** Timing:
    - the next step is due 40 ms later, and it sleeps until then;
    - if it is already late, it restarts the timing grid from now instead of
      catching up.

  ↩ back to fly.py.

- **L100** `hold = FlightLog()`: a second, separate log.
- **L101–102** `pid_fly` holds the start point for max(2.0, 2·Tini·Ts) = 2.4 s,
  logging every step. `FlightLog.append` (util.py **140–143**) stores one row of 18
  numbers:

  | Columns | Content |
  |---|---|
  | 0 | t |
  | 1–3 | position |
  | 4–6 | roll, pitch, yaw |
  | 7–10 | u |
  | 11 | battery voltage |
  | 12–14 | gyro |
  | 15–17 | reference position |

---

## 6. Handover to the SPC

The SPC needs a past window x_ini(k) of 29 inputs and 30 outputs. It is filled
from the PID's last steps, so the first prediction starts from the drone's real
history.

- **L107** `h`: the last 30 rows of the hold log.
- **L108** `Ybuf`: [x, y, z, yaw] of those 30 poses, minus `y_mean`. A 30 × 4 array.
- **L109** `Abuf`: their raw roll, pitch and yaw, for the observables.
- **L110** `Ubuf`: the inputs of the last 29 rows, minus `u_mean`. A 29 × 4 array.
  Together with the first new pose pushed at L125, this is exactly
  x_ini(k) = [u(k−29) … u(k−1), y(k−29) … y(k)].
- **L112** Prints "SPC engaged" (or KSPC).
- **L113–114** Restarts the clock for this phase.

---

## 7. The control loop

Each pass takes 40 ms. **L115** runs 1000 passes (k = 0 … 999), so 40 s.

- **L116** `t`: the time since the SPC engaged. The reference is evaluated at this
  t.
- **L118** The latest Vicon pose: a time stamp, the position, and roll/pitch/yaw.
- **L119–121** `safe.check` (see section 5), now *with* the floor check. It also
  gets the loop period, to count overruns. It aborts if there is a reason.
- **L122** Remembers this step's time.
- **L124** `y_k = [x, y, z, yaw] − y_mean`.
- **L125** `Ybuf`: drops the oldest row and appends `y_k`.
- **L126** `Abuf`: the same for the raw attitude.
- **L128** `z_k = lift(Ubuf, Ybuf, Abuf, par)`

  #### ↪ Detour: quadlib/lift.py
  - **`lift`, L63–69**: puts Ubuf (116 numbers, oldest first), then Ybuf (120
    numbers), then `phi_nl` in one row. z_k has length 236 for the SPC and 244 for
    the KSPC. The order matches how `identify.py` built Z_p.
  - **`phi_nl`, L46–60**: the 8 KSPC observables.
    - **L49–50** Empty for the SPC.
    - **L52–54** Uses the newest input (α̃_T and ω̃) and the newest yaw ψ̃.
    - **L57** a = a_of(ᾱ_T + α̃_T, newest attitude) − a_of(ᾱ_T, level). That is the
      acceleration the last thrust command gives at the current attitude, with the
      hover part removed.
    - **L59–60** Returns α̃_T·a (3 numbers), a_x², a_y², a_x·a_y, ψ̃·ω̃_x and
      ψ̃·ω̃_y.
  - **`a_of`, L36–43**: b₃ = the body z-axis in world coordinates (the third column
    of R). It returns a = α_T·b₃ − g·e₃.

- **L129** `r_k = reference.horizon(traj, t, par, y_mean)`

  #### ↪ Detour: `horizon`, reference.py 128–135
  For j = 1 … 30 it takes [pos(t + j·Ts), yaw(t + j·Ts)] − `y_mean`. That gives 120
  numbers, in the same order as Y_f: y(k+1) … y(k+N).

- **L131–132** `u_k, ok = ctrl.step(z_k, r_k)`, timed.

  #### ↪ Detour: `SPCController.step`, spc.py 61–79
  - **L67** b = −Ψ·z_k: what the outputs would do if the future inputs stayed at
    the mean. It moves to the right-hand side of the equality.
  - **L68–71** Updates the vectors only:
    - q = [0; −2Q·r], which brings the reference into the cost;
    - the equality bounds l = u = b, so Γu − y = −Ψz;
    - the input box stays as it was.
  - **L72** Solves, warm-started from the previous solution.
  - **L74–76** If it isn't solved, it keeps the previous input and returns
    `ok = False`.
  - **L78–79** The first 4 numbers of the solution are u(0|k), the input to apply
    now (mean-removed). It is stored in case the next solve fails.

- **L133–134** Stores the solve time in ms, and counts failures.
- **L136** `link.send_u(u_k + u_mean)`: adds the hover trim back and sends it. This
  is the same path as in `pid_fly` (see section 5).
- **L137** `Ubuf`: drops the oldest input and appends `u_k`. The window is now ready
  for k + 1.
- **L138** Logs the step: the time, the pose read *before* sending, the applied
  input, the telemetry, and the reference position at t.
- **L145–150** Timing, as in `pid_fly`:
  - the next pass is due 40 ms later, and it sleeps until then;
  - if it is late, it restarts the grid instead of catching up. The model assumes
    exactly 40 ms between samples, and catch-up steps a few ms apart would break
    that.

---

## 8. Landing and errors

- **L152–153** `land(...)`

  #### ↪ Detour: `land`, util.py 99–105
  - Flies `pid_fly` from the current position down at 0.35 m/s to 5 cm, holding x
    and y, for (height / 0.35 + 1.5) s.
  - Then `link.stop()` (link.py **174–182**) sends three zero commands 20 ms apart
    and a stop command, so the motors are off.

- **L155–156** On `AbortFlight` or Ctrl+C it calls `abort(...)`, with the reason
  (or "Ctrl+C").

  #### ↪ Detour: `abort`, util.py 108–125
  - Prints the reason.
  - If the latest Vicon pose is younger than 0.15 s, it lands as above. It does this
    without safety checks, so a geofence violation can't abort the landing.
  - If that landing fails too, or there is no fresh pose, it switches the motors
    off. The drone drops, which in a netted cage is safer than flying blind.

- **L157–158** `finally: vic.stop()` (vicon.py **71–72**) always ends the Vicon
  thread's loop, also after errors.
- When the `with` block ends, `Link.__exit__` (link.py **82–89**) switches the
  motors off, disarms, and closes the radio link. This also happens after an
  unexpected error (a bug), but then there is no landing first: the motors are cut.

---

## 9. After the flight

- **L160–161** If nothing was logged (an abort before the SPC engaged), it stops
  here.
- **L162–163** `log.save(...)`

  #### ↪ Detour: `FlightLog.save`, util.py 145–153
  - Turns the rows into an array (steps × 18) and creates `logs/` if needed.
  - Saves `t`, `pos`, `rpy`, `u`, `vbat`, `gyro`, `p_ref` and the metadata (Ts,
    trajectory name, `use_phi`, model path).
  - The file is `logs/fly_<traj>_<spc|kspc>_<date_time>.npz`.
  - Prints, and returns the array.

- **L165** `e`: the position minus the reference, for every step.
- **L166–168** Prints the RMSE per axis and in 3D, over the *whole* SPC phase,
  including the handover and the ramp-in. `plot_compare.py` and `plot_fig8_3d.m`
  leave out the first 5 s, so their numbers are lower.
- **L169–170** The mean and maximum solve time, and the number of failed solves.
- **L171–177** The median and maximum loop period, and the number of overruns. It
  warns if the median is more than 10 % off Ts, because the model then assumes a
  rhythm the loop doesn't deliver.
