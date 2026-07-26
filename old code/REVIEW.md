# Code sweep — findings and what changed

Everything below was verified against your actual files and data, not inferred.
Ordered by severity. `CASCADE_DEBATE.md` covers the architecture question.

---

## Blocking bugs — the pipeline could not have run correctly

### B1. Stale artifacts, silently mismatched dimensions
`data/invariant.mat` held `A (80×80)`, `P (80×80)`, `Theta (1600×160)` — i.e. it
was built at `n_basis = 80`, while `params.mat` said `n_basis = 40`.
Simultaneously `scalers.mat` held 9-element `y_mean`/`y_std` (from a run with
`include_extra_states = true`) against `ny = 4`, so
`(y_phys - y_mean)./y_std` in `KDPC_quad_sim.m` was a 4-vector minus a 9-vector.

Nothing in the pipeline checked this. **Fix:** `check_pipeline.m` validates every
artifact against every other one and against `koopman_config.m`, and
`KDPC_quad_sim.m` asserts on entry. `load_data.m` now trims the saved scalers to
`ny` so a 9-column scaler can never reach a 4-output model.

### B2. Network biases were never exported — MATLAB evaluated a different function
`model_learning_quad.py` saved only `model.l_N.weight`. Every `.m` file then computed

```matlab
tanh(weight2*tanh(weight1*X1))       % no b1, no b2
```

With `bias=False` that is correct. **You switched `bias=True`** — so from that
moment MATLAB's lifting was a different function from the one PyTorch trained, and
every downstream object (`Theta`, `A`, `B`, `P`, `K`, `alpha`) was fitted to
observables the network never produced. Same for the readout: `C` and `D` were
sliced out of `weight3` while `b3` was discarded.

**Fix:** all parameters export to `data/net.mat`; `lift.m` is the single place the
network is evaluated; `C`/`D` are now fitted from data rather than sliced.

### B3. `bias=True` breaks Assumption 2.4 — and with it the control objective
The paper requires `φ(0) = 0` (Assumption 2.4) with the origin as the equilibrium
(Assumption 2.2). Biases make `φ(0) = tanh(W₂ tanh(b₁) + b₂) ≠ 0`. Then:

- `z'Qz` no longer measures distance to hover, so the controller regulates to the
  wrong point;
- the terminal ellipsoid `z'Pz ≤ α` is centred on a non-equilibrium;
- Theorem 4.4 concludes `‖φ(x_ini)‖ → 0`, which is no longer the hover condition.

**You should keep `bias=True`** — it does fit better. The fix that keeps both is the
shifted observable

```
φ̃(x) := φ(x) − φ(0)
```

which satisfies `φ̃(0) = 0` exactly. Because the multi-step regression is then run
**without an intercept**, a linear map necessarily sends 0 → 0, so `(δz, u) = (0,0)`
is an exact fixed point of the lifted model. Assumptions 2.2 and 2.4 hold as
stated. Implemented in `load_net.m` / `lift.m`; `check_pipeline.m` verifies
`‖φ̃(0)‖ < 1e-10`.

---

## Major — correct code, wrong formulation

### M1. `Theta` was computed, then thrown away (this is the big one)
`Terminal_set_quad.m` solved the paper's unstructured eq. (8), `Theta = Phi_f/Phi_p`.
`KDPC_quad_sim.m` then discarded it and rebuilt predictions by chaining `A^j`,
with the comment that regressing `Theta` "destabilized the loop … acausal terms".

**Your diagnosis was right, and it is worse than you thought.** Measured on your
data, in the block row of `Theta` that predicts `z(k+1)`:

```
‖coefficients on u(k+1 … k+N−1)‖ = 9510.9      <- inputs that have not happened
‖coefficients on [z(k), u(k)]‖   =    9.8
acausal share of the row's energy = 100.0 %
```

But the workaround gives up the paper's entire contribution: multi-step prediction
exists *precisely* to stop error propagating with horizon, and `A^j` chaining
reintroduces it. Remark 3.1 anticipates this and states the correct remedy — so
the refactor takes it: regress each block row `j` on `[z(k); u(k…k+j−1)]` only.
This is causal by construction, preserves the nested structure of eq. (4) that
Theorem 4.2's shifted-sequence argument depends on, and remains a genuine
multi-step fit.

Held-out comparison on your data:

| | multi-step lifted R² | rel. err @ step 5 | @ step 10 | @ step 20 |
|---|---|---|---|---|
| causal block-triangular `Theta` | **0.9557** | **0.1297** | **0.1950** | **0.2827** |
| `A^j` chaining (your current code) | 0.9432 | 0.1439 | 0.2206 | 0.3215 |

Strictly better at every horizon, *and* theoretically sound. `KDPC_quad_sim.m` now
predicts with `Theta` again.

### M2. The `rho_yaw` hack
```matlab
d_yaw = [1;-1;1;-1]/2;  rho_yaw = 1000;  Rc_eff = Rc + rho_yaw*(d_yaw*d_yaw');
```
This penalised the yaw-torque input direction by 10⁵× the nominal `R`, to stop the
QP spinning the drone. Two problems: it treats a symptom of M1 and of the yaw
scaling issue (below), and `Rc_eff` was used in the MPC while plain `Rc` was used
in the DARE that produced `P` and `K`. Lemma 4.3 requires the *same* `Q, R` in the
stage cost and in LMI (13a); they were different, so the Lyapunov decrease
argument did not apply to the controller you were running.

**Fix:** removed. `R` comes from `koopman_config()` and is used identically in both
places. `check_pipeline.m` verifies LMI (13a) numerically.

### M3. Yaw scaling was the direct cause of `R²(yaw) ≈ 0.8`
Yaw is heavy-tailed — mostly held near constant with occasional large turns. Its
std is 0.055 rad (3.2°) while individual runs span 0.5–0.95 rad, so after
z-scoring yaw ranges over **±14σ** while x/y/z sit inside ±2.5σ. Yaw therefore
*dominated the training MSE* while being the least predictable channel.

**Fix:** robust scaling (half the 2–98 percentile span) plus explicit per-channel
loss weights in the training script. See `CASCADE_DEBATE.md` §2 for the physical
reason yaw is hard regardless (23× less control authority than roll/pitch).

### M4. Input bounds were not constraints
```matlab
u_bnd = min(abs(min(u_data)), abs(max(u_data)))';
```
This gives `u_bnd ≈ [5.37, 4.71, 5.41, 4.96]` in normalised units — a ±5σ box that
can never activate, so `α = 5215` and the terminal set was effectively the whole
space. Meanwhile the physics plant separately clipped with `max(u_phys, 0)`, a real
limit the QP knew nothing about.

**Fix:** bounds come from physical motor limits in `koopman_config.m`, mapped
through the scalers. They are *asymmetric* about hover, so the MPC gets the true
box and `α` uses the largest symmetric box inscribed in it (the support-function
bound assumes symmetry).

### M5. Regulating to the training-data mean
The target was `y = 0` in normalised coordinates, i.e. the mean position of the
training set — not a setpoint you can choose. **Fix:** quadrotor dynamics are
exactly translation-invariant, so a position setpoint is free by shifting the
measured history before lifting. Yaw setpoints are only approximately free (the
x/y dynamics depend on yaw through the rotation matrix); noted in the code.

### M6. `lambda = 1e6`
Lemma 4.3 gives `V(k+1) − V(k) ≤ −λ_min(Q)‖z₀‖² + λ‖e(k+1)‖²`, so the ISS gain
grows **linearly in λ**. `λ = 1e6` against `Q = 10I` also pins `ξ ≈ 0` at all times,
which makes the interpolated-initial-state machinery — the paper's actual novelty —
dead code. λ only needs to be large enough that ξ collapses once constraints go
inactive (paper Fig. 2). Default now `1e3`, documented, in `koopman_config.m`.

### M7. Terminal set disabled — and it turns out it *has* to be
`use_terminal_set = false`, and constraint (10d) on the observables (`X_z`) was
absent entirely. Without (10e) there is no Theorem 4.2, so the scheme had no
recursive feasibility guarantee. Both are now implemented — but implementing them
surfaced a genuine problem you should report rather than hide:

**`X_T` is empty in practice. 0.00 % of your training lifts satisfy `z'Pz ≤ α`**,
for every `(Qy, q_reg)` combination I tried. Two compounding causes:

1. `max|eig(A)| ≈ 1.002`, so the LQR `P` has very large eigenvalues and
   `{z'Pz ≤ α}` is a thin sliver;
2. an ellipsoid inscribed in a box is exponentially conservative in dimension,
   and `n_basis = 40` is a lot of dimensions. The `X_z` half of (13b) alone
   shrinks `α` by ~20×.

On top of that, `N = 20` (0.2 s) cannot steer the drone anywhere near the origin
regardless — see `CASCADE_DEBATE.md` §1. So a **hard** terminal constraint makes
the QP infeasible at step 1.

`cfg.terminal_mode` now offers `'off'` / `'soft'` / `'hard'`, defaulting to
`'soft'` (slack variable with a large penalty: keeps the QP feasible while
pushing the terminal state toward `X_T`). `Terminal_set_quad.m` warns loudly when
`X_T` is empty, and `check_pipeline.m` flags the `'hard'` + empty combination.

**Write-up implication:** with `'soft'` or `'off'` you do *not* have Theorem 4.2's
recursive feasibility guarantee. State that explicitly. Getting it back needs a
horizon long enough to reach `X_T`, which is the cascade argument.

---

## Redundancy

| what | where | fix |
|---|---|---|
| Hankel windowing implemented **3×** (Python, inline in `Terminal_set_quad.m`, implicitly in the sim), kept in sync by a comment | all | single `build_regressors.m` + Python twin; `check_pipeline.m` compares them numerically via `data/X1_py.mat` |
| `tanh(weight2*tanh(weight1*·))` copy-pasted 3× | 3 files | `lift.m` |
| `Q = 10*eye(n_basis)` duplicated with a "must match" comment; `Rc`/`R`, `Tini`, `N`, `n_basis` re-declared per file | 3 files | `koopman_config.m`, mirrored to Python via `export_config.m` |
| `include_extra_states` flag | `load_data.m` | dead — Python sliced to 4 outputs regardless. Removed; `ny` now comes from config and the full 9 are always stored |
| `data/X.mat`, `data/y.mat` (≈ 70 MB) | written by Python | never read by any `.m` file. No longer written |
| `_wtest.txt` | repo root | deleted |
| growing arrays in the sim loop (`Y_cl = [Y_cl; …]`) | `KDPC_quad_sim.m` | preallocated |
| `data/params.mat` vs `data/config.mat` | both | `params.mat` kept only for backward compatibility |

---

## Kept as-is (these were right)

- **The `l_r` reformulation.** Your rewrite of the pendulum's `ξ²` penalty into
  `λ‖z₀ − φ(x_ini)‖²` is exactly eq. (11), and by eq. (17) it is *identically*
  eq. (18) `λξ²‖e(k)‖²`. Writing it your way keeps the Hessian constant, which is
  what keeps this a fast parametric QP. Good call; I added the derivation as a
  comment so a reader does not "fix" it back.
- **DARE instead of the LMI** for `P`, `K`. The DARE solution satisfies (13a) with
  equality and is far better conditioned than an SDP at `n_basis = 40` with
  `max|eig(A)| ≈ 1.002`.
- **Ellipsoidal terminal set instead of MPT3 polyhedra.** Correct at this dimension.
  Your `α` derivation (`max_{z'Pz≤α}|K_i z| = √(α K_i P⁻¹ K_iᵀ)`) is right; I added
  the second half of (13b), the `X_T ⊆ X_z` check.
- **Per-run Hankel windows** that never cross run boundaries. Easy to get wrong.
- **Trim enforcement in `calibrate_plant.m`.** Forcing the plant to share the
  controller's equilibrium is the right call and well reasoned in your comment.
- **Online yaw unwrapping** in the physics loop to match the unwrapped training yaw.

---

## The finding that matters most

Your training data is closed-loop and carries almost no independent excitation:

```
u(k) predicted from the previous 5 input samples alone : R² = 0.9996
one-step change in u is 14 % of the signal magnitude
```

Least squares therefore cannot separate "input caused state" from "state caused
input" — which is the root cause of the acausal `Theta` (M1), of the 1e17 condition
number in the regression, and of R² numbers that look better than the model is. A
plain **linear** multi-step predictor with no lifting at all reaches R² = 0.9993 on
x/y/z and 0.916 on yaw on your held-out set; a 40-dimensional tanh lifting should
not be that close to it.

Before more modelling work, log flights with a dither (PRBS or multisine at a few
percent of hover, flat to ~20 Hz) superimposed on the stabilising controller. This
is what the paper's Section III means by "a persistently exciting input of
sufficient order". It is an afternoon of flying and it will improve your results
more than any change to the lifting.

---

## New pipeline order

```
koopman_config.m      <- edit settings here, nowhere else
  ↓
load_data.m           CSVs -> normalised data + scalers
export_config.m       mirror config to data/config.mat for Python
model_learning_quad.py   multi-step learning of φ (exports ALL parameters)
calibrate_plant.m     grey-box plant for closed-loop testing
Terminal_set_quad.m   causal Θ, A, B, C, D, P, K, α
check_pipeline.m      <- run this before every simulation
KDPC_quad_sim.m       closed loop
```

Cascade variant: `make_cascade_data.m` → same pipeline pointed at `data/cascade/`,
with `quad_plant_cascade.m` as the plant.
