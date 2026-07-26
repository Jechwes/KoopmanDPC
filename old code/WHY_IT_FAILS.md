# Why KDPC works on the pendulum and chart, but not (yet) on the drone

Comparison of `KDPC_RSRFG/{Pendulum,Chart}` against `JCVW_QUAD`. Every number
below is measured, not asserted.

---

## The controlled experiment

The most useful thing I can show you is a comparison where **only the data
differs**. I applied the *same* random tanh lifting, the *same* estimator, and
the *same* equation — the paper's unstructured eq. (8) — to both datasets, and
looked at the block row of `Theta` that predicts `z(k+1)`:

| dataset | ‖[A B]‖ | ‖acausal‖ | acausal share | cond(Φₚ Φₚᵀ) |
|---|---|---|---|---|
| pendulum, open-loop multisine | 6.56 | **0.05** | **0.0 %** | 7.2e+03 |
| drone, closed-loop flight | 48.44 | **47944** | **100.0 %** | 2.1e+14 |

"Acausal" = coefficients placed on inputs `u(k+1 … k+N−1)` that have not
happened yet when `z(k+1)` occurs.

**On the pendulum, the block-triangular structure of eq. (4) emerges by itself.**
The multisine is white, so future inputs are uncorrelated with `z(k+1)` and least
squares assigns them ~zero weight. `Theta = Phi_f*pinv(Phi_p)` is therefore
causal *for free*, and the reference code can use it directly in the MPC.

On your data it does not emerge, because the input is not informative:

| lags of `u` used | pendulum: R² predicting `u(k)` | drone: R² predicting `u(k)` |
|---|---|---|
| 1 | 0.0000 | 0.9804 |
| 3 | 0.0020 | 0.9969 |
| 9 | 0.0068 | 0.9977 |

The reference's input is *designed*: `idinput([1500 1 1],'sine',[0 1],[-4 4],[25 40 1])`
— a 25-line multisine spanning the full band, applied **open loop** to the plant.
That is persistency of excitation by construction, exactly what Section III of
the paper requires ("a persistently exciting input of sufficient order", ref [19]).

Your input is whatever the Crazyflie's onboard controller happened to command
while flying chirp/random/square *trajectories*. Exciting the **reference** is
not the same as exciting the **input**: the controller absorbs the trajectory
variation and emits a smooth, state-determined motor signal. Least squares then
cannot separate "the input moved the state" from "the state moved the input".

**This is the single biggest reason it doesn't work.** It is also the cheapest to
fix, and it is upstream of nearly everything else you have been fighting.

---

## Full structural comparison

| | Pendulum / Chart (works) | Your drone (doesn't) |
|---|---|---|
| **Input design** | `idinput` multisine, open loop, ±4 | onboard controller output, closed loop |
| **PE** (`u(k)` from own past) | R² = 0.00 | R² = 0.998 |
| **cond(Φₚ Φₚᵀ)** | 7.2e3 | 2.1e14 |
| **Acausal energy in Θ** | 0.0 % | 100.0 % |
| **Plant in simulation** | *the exact difference equations that generated the data* | a different grey-box model; data came from real hardware |
| **Actuation** | SISO, directly actuated, relative degree 2 | 4 inputs / 6 DOF, underactuated, relative degree 4 |
| **Horizon vs. task** | N=15 @ 1/30 s = 0.5 s ≈ ⅓ of a natural period | N=20 @ 0.01 s = 0.2 s ≈ 2 % of the manoeuvre |
| **Equilibrium** | origin exactly, with `u = 0` | hover at `u = u_mean`; trim only approximate |
| **φ(0) = 0** | automatic (`bias=False`) | broken by `bias=True` (now fixed via `φ̃ = φ − φ(0)`) |
| **Scaling** | none needed, `y ~ O(1)` | required; yaw heavy-tailed, ±14σ before the fix |
| **n_basis** | 10 | 40 |
| **Θ size** | 150 × 25 | 800 × 120 |
| **Terminal set** | MPT3 polyhedron, exact | ellipsoid, **empty in practice** |
| **Solver** | quadprog, small QP | SeDuMi for the SOC, 200–300 ms/step |

---

## Ranked causes

### 1. The data (dominant)
Covered above. Everything else is secondary until this is fixed. Note it also
explains three symptoms you had already worked around by hand:

- the acausal `Theta` you correctly diagnosed and bypassed with `A^j` chaining;
- the yaw-torque direction the QP kept exciting "for free" (your `rho_yaw` hack);
- R² > 0.99 that a *plain linear* predictor nearly matches (0.9993 on x/y/z).

**Fix:** re-fly with a dither — PRBS or multisine, a few percent of hover
amplitude, flat to ~20 Hz — superimposed on the stabilising controller's output.
That converts a closed-loop log into an identification experiment. One afternoon.

### 2. Underactuation and horizon
The pendulum's input is a torque acting directly on the coordinate being
controlled: relative degree 2, and `N` covers a third of a natural period. Your
input is four rotor speeds, and position responds only through
thrust → attitude → tilt → lateral acceleration → velocity → position. Within
`N = 20` (0.2 s) the drone can move **1.9 cm**, of which 79 ms goes just to
establishing tilt. Your test offset is 40 cm.

This is not tunable. It needs either `N ≈ 100` or, better, the cascade
(`CASCADE_DEBATE.md`), which changes the input to `[T, ω_ref]` and makes 0.2–1 s
of look-ahead meaningful.

### 3. Plant/model mismatch
In the reference, `simulation_Xi.m` propagates **the same two lines of difference
equation** that `Generate_data.m` used. The only error is the Koopman
approximation itself — precisely the regime the paper's ISS result was designed
for.

You have real flight data on one side and an independent grey-box model on the
other, so `e(k)` contains genuine physics error too: `calibrate_plant.m` explains
only 24 % / 13 % / 8 % of the angular accelerations, and `quad_plant.m` has no
aerodynamic damping. Your `'lifted_state'` mode is the right way to separate
these — it reproduces the reference's zero-mismatch condition.

### 4. Equilibrium and trim
The pendulum's origin is an exact equilibrium at `u = 0`, and `bias=False` gives
`φ(0) = 0` automatically, so Assumptions 2.2 and 2.4 hold with no effort.

Hover needs `u = u_mean ≠ 0`, and `u_mean` is the mean of *manoeuvring* flight,
which is not exactly a trim point. `calibrate_plant.m` papers over this by
forcing the plant's trim to match — the right call, but the residual shows up as
a constant disturbance the Koopman model must absorb.

### 5. Dimension and terminal-set conservatism
`n_basis` 10 → 40 turns a well-conditioned 150×25 regression into an
ill-conditioned 800×120 one, and makes the ellipsoidal terminal set exponentially
conservative — `X_T` contains **0.00 %** of your training lifts.

Worth trying: drop `n_basis` to 15–20. Given that a linear predictor already
reaches R² = 0.9993 on position, 40 tanh observables are almost certainly
overkill, and the extra dimensions are actively hurting the terminal ingredients.

---

## Things in the reference worth copying

- **`|z_i| ≤ 1` as the state constraint** (`Pendulum/Terminal_set.m`, `Ax=[I;-I]`,
  `bx=1`). The observables are tanh outputs, so this bound is *exact and valid
  everywhere*, not estimated from a sample. Adopted — `Terminal_set_quad.m` now
  uses `1 + |z_eq|` (the shifted equivalent) instead of a data percentile.
- **`pinv` rather than `\`** for `Theta`. `pinv` truncates small singular values,
  which is implicit regularisation. It would have damped your blow-up — though
  imposing the causal structure directly, as we now do, is strictly better.
- **A much smaller lifting.** `n_basis = 10` for a 2-state system.

## Things in the reference NOT worth copying

- **`R = 1` in `Pendulum/Terminal_set.m` but `R = 0.01` in `simulation_Xi.m`.**
  The reference has the same `Q,R` inconsistency I flagged in your code. Lemma 4.3
  needs the same weights in the stage cost and in LMI (13a). It survives on the
  pendulum; it will not survive here.
- **`lambda = 1e10`.** The ISS gain scales linearly with λ (Lemma 4.3). You
  inherited `1e6`; both are far larger than needed.
- **Re-solving the LMI for `P` inside `simulation_Xi.m`** with a *different*
  objective (`Pin = 1/10000`) than `Terminal_set.m` used (`10*eye`). Two
  different `P` matrices in one pipeline.
- **`Tini` and `N` inferred from weight-matrix shapes.** Clever, but it is how a
  stale weight file silently redefines your horizon.

---

## Recommended order

1. **Run `plant_mode = 'lifted_state'`.** This reproduces the reference's
   zero-mismatch condition. If it converges, the controller machinery is sound
   and your problem is data + horizon. If it doesn't, stop and fix that first —
   nothing downstream can work.
2. **Re-fly with dither.** Biggest single improvement available, cheapest to do.
3. **Move to the cascade** (`CASCADE_DEBATE.md`), which fixes the horizon and the
   yaw-authority problems together.
4. **Shrink `n_basis`** to 15–20 and see whether anything gets worse.
5. Keep the flat motor-level design as a documented negative result. You now have
   hard numbers for every part of it: 100 % acausal energy, 1.9 cm reachable
   motion, 23× yaw authority deficit, empty terminal set.
