# KoopmanDPC — quadrotor

Koopman data-driven predictive control of a Crazyflie-class nano-drone, following
de Jong, Breschi, Schoukens & Lazar, *"Koopman Data-Driven Predictive Control with
Robust Stability and Recursive Feasibility Guarantees"*, CDC 2024.

- **`REVIEW.md`** — what was wrong with the previous version and what changed.
- **`CASCADE_DEBATE.md`** — inner-loop rate PID vs. flat motor-level KDPC, with
  numbers.

## First run

Your `data/` folder still holds artifacts from the old pipeline that are actively
wrong (`invariant.mat` built at `n_basis = 80`, `weight*.mat` with no bias terms,
`scalers.mat` with 9 outputs against `ny = 4`). Clear them once:

```matlab
cd C:\TUe\Job\JCVW_QUAD
clean_stale          % deletes stale .mat files; your CSVs are untouched
```

Then run the pipeline below from step 2. Everything after that is a normal re-run.

## Pipeline

Run in this order, from the project folder:

| # | step | produces |
|---|---|---|
| 1 | `koopman_config` | *edit this file to change anything* |
| 2 | `load_data` | `data/{u,y}_data*.mat`, `scalers.mat`, `run_lengths.mat` |
| 3 | `export_config` | `data/config.mat` (so Python sees the same settings) |
| 4 | `python model_learning_quad.py` | `data/net.mat` (all weights **and biases**) |
| 5 | `calibrate_plant` | `data/plant_params.mat` |
| 6 | `Terminal_set_quad` | `data/invariant.mat` (causal Θ, A, B, C, D, P, K, α) |
| 7 | `check_pipeline` | **run before every simulation** |
| 8 | `KDPC_quad_sim` | closed-loop results |

`check_pipeline` is not optional bookkeeping: it catches stale `.mat` files,
dimension drift between the Python and MATLAB halves, and violations of the
paper's Assumption 2.4 and LMI (13a). All of those had silently occurred.

### Cascade variant

```matlab
calibrate_plant        % needs c_t
make_cascade_data      % rebuilds the dataset as [T, wx, wy, wz] -> [x y z yaw]
```
Then point the pipeline at `data/cascade/` and use `quad_plant_cascade.m` as the
plant. See `CASCADE_DEBATE.md`.

## Files

**Configuration**
- `koopman_config.m` — single source of truth for every hyperparameter
- `export_config.m` — mirrors it to `data/config.mat` for Python

**Shared primitives** (each previously duplicated across 3 files)
- `build_regressors.m` — Hankel windows, eq. (6); twin of the Python function
- `lift.m` — the only place the observable network is evaluated
- `load_net.m` — loads the network and the equilibrium shift `z_eq = φ(0)`

**Pipeline**
- `load_data.m`, `model_learning_quad.py`, `Terminal_set_quad.m`, `KDPC_quad_sim.m`
- `check_pipeline.m` — consistency and theory-assumption guard

**Plant** (independent of the predictor, for closed-loop testing)
- `calibrate_plant.m`, `quad_plant.m`, `quad_plant_cascade.m`, `make_cascade_data.m`

## Requirements

MATLAB with Control System Toolbox (`idare`), [YALMIP](https://yalmip.github.io/),
and `quadprog`; SeDuMi if `use_terminal_set = true` (the SOC terminal constraint).
Python: `numpy`, `scipy`, `matplotlib`, `torch`.

## Environment (WSL)

The Python venv lives in the WSL home directory (`~/jcvw-venv`), **not** in this
folder — keeping it out of `/mnt/c` avoids path/symlink issues and is faster.

```bash
cd /mnt/c/TUe/Job/JCVW_QUAD
source ~/jcvw-venv/bin/activate
python model_learning_quad.py
```

In VS Code (WSL-connected): Ctrl+Shift+P → "Python: Select Interpreter" →
`~/jcvw-venv/bin/python`.

Recreating it:

```bash
python3 -m venv ~/jcvw-venv
source ~/jcvw-venv/bin/activate
pip install numpy scipy matplotlib torch
# CPU-only: append --index-url https://download.pytorch.org/whl/cpu
```

## Data

`data/` is git-ignored and kept local. The scripts expect `data/train/*.csv` and
`data/test/*.csv` (Busetto et al. benchmark, Crazyflie 2.1 Brushless, 100 Hz) with
columns `t,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz,m1_rads..m4_rads,ax_body,ay_body,az_body`.

> **Known limitation.** The supplied flights are closed-loop and carry almost no
> independent input excitation (`u(k)` is predictable from its own past 5 samples
> at R² = 0.9996). See `REVIEW.md` — this bounds how much any model identified
> from this data can be trusted for control.
