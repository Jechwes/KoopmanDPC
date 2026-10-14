# JCVW_QUAD_CRAZY — dependencies

Subspace predictive control (SPC; Koopman SPC with `model.use_phi: true`) on a
Crazyflie 2.1 Brushless. SPC runs on the laptop at 25 Hz, the rate PID runs on
the drone.

## Python

Python 3.12, Linux / WSL2.

```bash
python3 -m venv quad_env
source quad_env/bin/activate
pip install -r requirements.txt
```

| package | used for |
|---|---|
| `cflib` | Crazyflie radio link, setpoints, logging, params |
| `motioncapture` | Vicon DataStream client |
| `numpy` | everything |
| `scipy` | quaternions (`spatial.transform`), sparse matrices, `pinv` |
| `osqp` | the QP in the control loop |
| `pyyaml` | `config.yaml` |
| `matplotlib` | plots |

Deliberately **not** using `cvxpy`: its canonicalisation layer recompiles the
problem whenever the parametric expression is not DPP-compliant, which cost
~70 ms per solve in an earlier project. The QP here is set up once against OSQP
directly.

## System

```bash
sudo apt install libusb-1.0-0
```

Crazyradio udev rules, or run as root:
https://www.bitcraze.io/documentation/repository/crazyflie-lib-python/master/installation/usb_permissions/

## WSL2 only — USB passthrough

The Crazyradio is not visible to WSL by default. On the **Windows** side:

```powershell
winget install usbipd
usbipd list                          # find the Crazyradio bus id
usbipd bind   --busid <X-Y>          # once, as admin
usbipd attach --wsl --busid <X-Y>    # after every replug / WSL restart
```

Vicon needs nothing special — it is an outbound TCP connection, so WSL2's NAT
is fine.

## Hardware

- Crazyflie 2.1 Brushless with Vicon markers
- Crazyradio PA
- Vicon system streaming a rigid body whose name matches `vicon.object_name`

## Workflow (SPC)

From this folder, in this order. Every flying script lands on Ctrl+C.

| step | command | motors | establishes |
|---|---|---|---|
| 1 | `python hover.py --no-fly` | off | radio, battery, supervisor, Vicon body, Vicon pose reaching the Kalman filter |
| 2 | `python hover.py` | on | onboard takeoff / hover / land on Vicon; prints the `thrust_max_N` to put in `config.yaml` |
| 3 | `python check_frame.py` | off | Vicon body frame = drone body frame, then set `frame_verified: true` |
| 4 | `python collect.py --runs 4 --tag a` | on | identification data; one battery each, repeat with `--seed0 2000 --tag b` |
| 5 | `python identify.py` | – | `models/theta_spc.npz` |
| 6 | `python replay.py --flight "data/flights/a_*_r00.npz"` | – | controller output finite, in the box, right signs |
| 7 | `python fly.py --traj fig8_slow`, then `--traj fig8` | on | SPC flight, laptop at 25 Hz, rate PID onboard |

`python simulate.py all` runs steps 4, 5 and 7 against a simulated drone with the
same scripts. `model.use_phi: true` switches everything to Koopman SPC.

## SPC settings

`config.yaml` carries the real-flight SPC of Aalders (2024), *Data-Driven
Predictive Control of Crazyflie Quadcopters*, Sec. V: N = Tini = 30 at 25 Hz,
its weights and its input box, converted to this project's inputs (specific
thrust, body rates in rad/s). The conversion is in the comments there. Figure 8
in simulation, model identified on 8 x 60 s:

| | 10 ms loop delay | 20 ms | 20 ms + Vicon-like noise |
|---|---|---|---|
| previous (Tini 3, N 20, Ru 0.1 / 2) | 0.16 m RMSE | diverges | – |
| thesis horizon, previous weights | 0.12 m | 0.13 m | 0.17 m |
| thesis horizon, weights and box | 0.04 m | 0.04 m | 0.08 m |

Real Vicon + radio delay is 10-20 ms. A model identified at 10 ms delay flew
the same at 0-30 ms. On 4 x 30 s of data (about the thesis' T = 3000) the
previous settings diverged; the thesis settings did as well on 4 x 60 s as on
8 x 60 s. Collect at least 4 x 60 s, better 8 x 60 s over two batteries.

Loop timing matters as much as the gains: the model assumes every step is
40 ms, and a 30-step window remembers a late step for 1.2 s. The loops restart
their schedule after an overrun instead of catching up, and `identify.py` cuts
the data there, but keep the laptop otherwise idle while flying. `hover.py`
reports laptop stalls over 60 ms; `fly.py` prints the overruns it saw.

## Faster trajectories

`--speed` flies the same path faster: `python fly.py --traj fig8 --speed 1.75`
is a figure 8 at w = 1.4 rad/s. Tilt grows with speed^2 and the body rates it
takes with speed^3, and `fly.py` warns about both limits before taking off:

- the reference alone needs more than 90 % of `spc_box.omega`. In simulation,
  +/-0.785 rad/s (the thesis) diverged at `fig8 --speed 1.75` and +/-1.5 flew it.
- the peak tilt, or the peak horizontal or vertical speed, is beyond 99 % of
  the identification data (stored by `identify.py`). Collect more aggressive
  data (larger `collect.ref_box`, shorter `ref_hold_s`) and re-identify.

`python plot_compare.py` plots every speed flown with both controllers.

## T_ini sweep

`python sweep_tini.py` scores T_ini = 5 ... 40 (N from the config) by the
held-out prediction error of SPC and KSPC: models identified on all flights but
one fold, predicting the position over the horizon on that fold. No drone
needed; it writes `figures/tini_sweep.png/.pdf/.csv`. To fly a setting:
`python identify.py --tini 20 --outdir models/tini20`, then
`python fly.py --traj fig8 --model models/tini20/theta_kspc.npz`. Leave
`config.yaml` as it is: a window other than its `Tini`/`N` goes into the log name
(`fly_fig8_tini20_kspc_*.npz`), so these runs are plotted apart from the default ones.

## Figure 8 with flat turns

`--traj fig8_planes` (also `_slow`, `_fast`, `_xfast`) is a figure 8 of
+/-1.2 m by +/-0.5 m whose lobes are flown at constant height, 1.0 m + H and
1.0 m - H, with the climb around the crossing. Holding height while banked needs
extra thrust that grows with the square of the tilt, which a linear predictor
cannot form; the SPC shows it as sag in the turns, so compare the z error in the
turns: `python plot_compare.py --z --traj fig8_planes_fast fig8_planes_xfast`
plots the height with the turns shaded and prints the sag per controller.

| | w [rad/s] | H [m] | kap | tilt | turn rate | thrust [m/s^2] |
|---|---|---|---|---|---|---|
| `fig8_planes` | 0.8 | 0.4 | 3 | 8 deg | 0.28 rad/s | 8.0 .. 11.6 |
| `fig8_planes_fast` | 1.1 | 0.4 | 3 | 15 deg | 0.97 rad/s | 6.5 .. 13.3 |
| `fig8_planes_xfast` | 1.21 | 0.5 | 2.5 | 19 deg | 1.33 rad/s | 6.3 .. 13.6 |

`fly.py` checks the turn rate and the thrust the reference needs against the
SPC's input box, and the speed against the identification data. `_xfast` needs
`spc_box.alpha_T_rel: [0.58, 1.42]` and data as aggressive as sessions c/d.
