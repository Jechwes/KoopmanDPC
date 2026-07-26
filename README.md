# KDPC on a nano-drone

Koopman data-driven predictive control (de Jong, Breschi, Schoukens & Lazar,
CDC 2024) applied to a Crazyflie 2.1 Brushless.

Structure mirrors `KDPC_RSRFG/Pendulum` file-for-file, so you can read the two
side by side.

| this project | reference | does |
|---|---|---|
| `Generate_data.m` | `Generate_data.m` | data → `u_data`, `y_data`, scalers |
| `model_learning_tanh.py` | `model_learning_tanh.py` | learns the observables φ |
| `tanh_nn.m` | `tanh_nn.m` | evaluates φ, builds `[z; u_f]` |
| `Terminal_set.m` | `Terminal_set.m` | multi-step `Theta`, `A`, `B`, `P`, `K` |
| `simulation_Xi.m` | `simulation_Xi.m` | closed-loop KDPC |

Run in that order, from the project folder.

```matlab
Generate_data
```
```bash
source ~/jcvw-venv/bin/activate && python model_learning_tanh.py
```
```matlab
Terminal_set
simulation_Xi
```

## What differs from the reference, and why

| | reference | here |
|---|---|---|
| system | pendulum, 1 in / 1 out | drone, 4 motors / 4 outputs |
| data | simulated, open-loop multisine | logged flights, `data/train`, `data/test` |
| scaling | none needed | normalised (motor speeds ~1740 rad/s) |
| Hankel windows | one continuous run | per run, never crossing a boundary |
| `Tini`, `N`, `n_basis` | 3, 15, 10 | 10, 20, 40 |
| terminal set | LMI (mosek) + polyhedron (MPT3) | omitted; terminal cost `P` from `idare` |

The terminal set is left out on purpose. It needs mosek and MPT3, and neither
scales to `n_basis = 40`. Running with terminal cost only means Theorem 4.2's
recursive feasibility guarantee does not apply — worth stating in a write-up,
but it is not needed to get the scheme working.

The plant in `simulation_Xi.m` is the learned lifted model itself, so there is
no plant/model mismatch — the same situation the reference is in. Get this
working first; a rigid-body simulator goes in the marked block afterwards.

## Requirements

MATLAB with Control System Toolbox (`idare`), YALMIP, `quadprog`.
Python: `numpy`, `scipy`, `matplotlib`, `torch`.

## Data

`data/` is git-ignored. Expects `data/train/*.csv` and `data/test/*.csv` with
columns `t,x,y,z,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz,m1_rads..m4_rads,ax_body,ay_body,az_body`.

## `old code/`

The previous, much larger version. Contains analysis worth keeping —
`WHY_IT_FAILS.md` compares this project against the reference in detail, and
`CASCADE_DEBATE.md` argues for an inner-loop rate PID. The headline findings:

- the training data is closed-loop and barely excited (`u(k)` is predictable
  from its own past 5 samples at R² = 0.9996), which is why `Theta` puts 100 %
  of its first block row's energy on inputs that have not happened yet;
- with `N = 20` at 100 Hz the drone can only move 1.9 cm within the horizon.

Neither is fixed by the code here. Both matter once you move past the
zero-mismatch case.
