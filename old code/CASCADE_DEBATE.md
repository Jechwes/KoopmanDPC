# Cascade or flat? Rate PID inner loop vs. motor-level Koopman DPC

**The question.** Standard quadrotor practice puts an attitude-rate PID on the inner
loop (500–1000 Hz, on-board) and an optimiser on the outer loop (50–100 Hz). Your
current design is flat: the KDPC outputs four motor speeds directly at 100 Hz.
Is the standard architecture also right here?

**Short answer: yes, cascade — and for this project the argument is not stylistic,
it is quantitative.** Three independent numbers from your own data and hardware
constants say the flat design cannot work as configured. But the reason is subtler
than "cascades are conventional", and one of the three reasons is a data problem
that a cascade only partly fixes. All figures below are computed from your CSVs and
the benchmark's published constants.

---

## 1. The horizon argument (decisive)

Your prediction horizon is `N = 20` at 100 Hz = **0.20 s**. A quadrotor is
underactuated: to move laterally it must first tilt, and tilting costs time.

Using the benchmark constants (`Kt = 3.72e-8`, `arm = 0.0353`, `J_xy = 2.40e-5`)
and a 3σ differential motor command:

| | value |
|---|---|
| Time just to establish a 15° tilt | **79 ms — 40 % of the entire horizon** |
| Lateral distance reachable within the horizon | **1.9 cm** |
| Yaw reachable within the horizon | **4.2°** |

Your simulation initial condition is a **40 cm** offset. The controller can see
1.9 cm ahead. Effectively the entire manoeuvre lives beyond the horizon, so the
terminal cost `z'Pz` — a linear-quadratic extrapolation of a lifted model that was
never validated that far out — is doing 95 % of the work. That is not predictive
control; it is an LQR with a QP wrapped around it.

This is not a hypothetical. Building the terminal ingredients properly and
measuring, **0.00 % of your training lifts satisfy `z'Pz ≤ α`** — the terminal set
is empty in practice, so a hard terminal constraint (10e) makes the QP infeasible
at step 1, and Theorem 4.2's recursive feasibility guarantee is unavailable. The
horizon simply cannot reach the terminal region.

For reference, the horizon a *flat* design would actually need:

| step size | bang-bang time @15° | required N @100 Hz | required N @20 Hz |
|---|---|---|---|
| 0.1 m | 0.39 s | 39 | 7 |
| 0.4 m | 0.78 s | 78 | 15 |
| 1.0 m | 1.23 s | 123 | 24 |

`N = 100` at `n_basis = 40` means an 800-dimensional prediction stack and a
440-variable QP every 10 ms. That is the crux: **a flat controller must be fast
(motor bandwidth) and long-sighted (position settling) at the same time, and those
requirements are two orders of magnitude apart in timescale.** A cascade splits
them — the rate loop takes the fast requirement, the KDPC takes the slow one and
can then run at 20–50 Hz with a modest `N`.

## 2. The actuation asymmetry argument (explains your yaw R²)

From the identified allocation, per unit differential command:

```
roll / pitch authority :  55.9 rad/s²
yaw authority          :   2.44 rad/s²      -> 23x weaker
```

Yaw torque comes from rotor *drag* (`Kc = 7.74e-12`), roll/pitch from rotor
*thrust* × arm (`Kt·arm = 1.31e-9`). Meanwhile the input *energy* your data spends
in each direction is comparable (normalised σ: collective 1.77, roll 0.44, pitch
0.65, yaw 0.51). So yaw gets similar excitation and produces ~23× less response —
the yaw signal-to-noise ratio in the training data is roughly 23× worse than
roll/pitch. **That, not your network capacity, is why yaw sits at R² ≈ 0.8 while
x/y/z sit above 0.99.** Adding basis functions will not fix a channel whose
response is buried in gyro noise.

A rate PID fixes this the way high-gain feedback always does: it closes a stiff
loop around the weak channel, so the outer loop commands `ψ̇_ref` and gets it,
regardless of how badly the yaw *torque* map is identified. The 23× disadvantage
is absorbed by the inner loop instead of appearing in your Koopman model.

This is also what makes the `rho_yaw = 1000` hack in your old `KDPC_quad_sim.m`
unnecessary. That penalty existed to stop the QP exciting the `[+,−,+,−]`
direction "for free" — a direct symptom of a badly identified, weakly excited yaw
channel. Suppressing the input direction suppresses the symptom and your yaw
authority along with it.

## 3. The identifiability argument (the one a cascade only *partly* fixes)

This is the finding I would put first in your report, because it undermines the
R² numbers you are currently reporting as success.

Your training data is **closed-loop**: the Crazyflie's own controller generated
the motor commands. Measured on your CSVs:

```
u(k) predicted from the previous 1 input sample alone : R^2 = 0.9808
                            ... 3 samples             : R^2 = 0.9980
                            ... 5 samples             : R^2 = 0.9996
                            ... 9 samples             : R^2 = 0.99996
one-step change in u is only 14 % of the signal magnitude
```

The input is, to four decimal places, a smooth deterministic continuation of its
own past. There is almost no independent excitation anywhere in the dataset.

Three consequences, all of which you have already run into:

1. **The unstructured `Theta` blows up.** Regressing `z(k+1)` on `[z(k); u(k..k+N-1)]`,
   the coefficients on inputs that *have not happened yet* carry **100.0 %** of the
   block row's energy (‖acausal‖ = 9511 vs ‖[A B]‖ = 9.8). Least squares cannot
   tell "the input caused this" from "this caused the input", so it uses future
   inputs as a proxy for the state. This is exactly the instability you worked
   around by discarding `Theta` and chaining `A^j`. The refactor fixes it properly
   via Remark 3.1 (causal block-triangular regression), but the underlying data
   deficiency remains.
2. **Your R² is optimistic.** A plain *linear* multi-step predictor — no Koopman
   lifting at all — gets R² = 0.9993 on x, y, z and 0.916 on yaw on your held-out
   test set. Your 40-dimensional tanh lifting is barely beating a linear model.
   On genuinely informative data it should not be close. High open-loop R² on
   closed-loop data mostly measures how predictable the *pilot* was.
3. **The model is only valid along the trajectories the old controller flew.**
   MPC deliberately applies inputs that controller would not have chosen. That is
   the regime where the model has no support.

**A cascade helps here, but does not cure it.** Re-identifying the outer-loop map
`[T_spec, ωx, ωy, ωz] → [x, y, z, yaw]` on the same data gives a modest but real
improvement (plain linear predictor, identical windows):

| predictor input | overall | x | y | z | yaw | step 20 |
|---|---|---|---|---|---|---|
| motor speeds | 0.9823 | 0.9993 | 0.9995 | 0.9993 | 0.9160 | 0.931 |
| `[T, ωx, ωy, ωz]` | **0.9878** | 0.9994 | 0.9995 | 0.9993 | **0.9424** | **0.956** |

Yaw improves most (0.916 → 0.942), and long-horizon accuracy improves (0.931 →
0.956) — consistent with the cascade removing the fastest, worst-identified
dynamics from the model. But the gains are incremental because the *excitation*
problem is untouched.

**What actually fixes it:** log new flights with a dither signal — a PRBS or
multisine superimposed on the stabilising controller's output, amplitude a few
percent of hover, spectrum flat over 0–20 Hz for the outer loop. This is what
Section III of the paper means by "a persistently exciting input of sufficient
order" (their ref. [19]). It is a one-afternoon experiment and it will do more for
your model quality than any change to the lifting.

## 4. The honest case *against* the cascade

I do not think these win, but your committee will raise them and they deserve real
answers.

- **"The cascade throws away the point of the paper."** Partly true. The scheme's
  contribution is handling plant/model mismatch of a *linear-in-control* Koopman
  model. Wrapping a PID inside means you now identify a partly-closed-loop plant,
  and the ISS result is with respect to *that* plant's model error, not the bare
  drone's. The theorems still hold verbatim — the inner loop is just part of `f`
  in eq. (1) — but the demonstration is less pure.
  *Answer:* the paper's own Remark 2.6 concedes linear-in-control Koopman models
  are only valid "for short prediction horizons... where trajectories remain close
  to current initial conditions". A cascade is the standard way to *create* that
  condition. You are satisfying the paper's stated hypothesis, not evading it.

- **"Cascade control is conservative — bandwidth separation costs performance."**
  True in general, and it matters for aggressive flight (flips, high-speed
  trajectories) where the small-angle assumption of the outer loop breaks.
  *Answer:* irrelevant at your operating point. Your data is hover ±0.5 m at
  modest tilt. The cascade's assumptions hold comfortably.

- **"Constraint handling degrades."** Real. A flat design can impose true motor
  limits inside the QP. Cascaded, the outer loop constrains tilt and thrust, and
  the mixer saturates motors afterwards — so motor saturation is invisible to the
  optimiser and the recursive-feasibility guarantee is with respect to the
  *surrogate* constraint set.
  *Answer:* this is the genuine theoretical cost. Mitigate by making the outer
  loop's tilt/thrust box conservative enough that the mixer never saturates, and
  say so explicitly in the write-up. Note your *current* code has a worse version
  of the same problem: `u_bnd` was derived from the training-data range, giving a
  ±5σ "limit" that can never activate, while the physics plant separately clipped
  at zero — so the QP already had no idea what the real limits were.

- **"Bare-motor KDPC is the more novel result."** It is. If the thesis
  contribution is *demonstrating the paper's scheme on a drone*, the cascade is
  the right call. If the contribution is *pushing Koopman DPC down to the actuator
  level*, then the flat design is the point — but then you need `N ≈ 100`, a much
  richer lifting, and properly excited data, and you should budget accordingly.

## 5. Recommendation

```
  KDPC  (20-50 Hz, offboard)         inner loop (500 Hz, on-board)
  ┌─────────────────────────┐        ┌──────────────────────┐
  │ y = [x y z yaw]         │  T,ω_ref │ rate PID + mixer     │  Ω
  │ Koopman lift + QP       │ ───────► │                      │ ───► drone
  └─────────────────────────┘        └──────────────────────┘
```

- Outer loop: `u = [T_spec, ωx_ref, ωy_ref, ωz_ref]`, `y = [x, y, z, yaw]`,
  `Ts = 0.02–0.05 s`, `N = 15–25` (0.3–1.25 s of look-ahead — finally meaningful).
- Inner loop: the Crazyflie's stock rate PID. Do not re-derive it.
- Constraints become physically meaningful and roughly symmetric: tilt-rate limits
  and a thrust box, instead of an arbitrary multiple of a motor-speed std.
- `n_basis` can likely drop from 40; the outer-loop plant is far closer to linear
  (it is nearly a double integrator per axis plus a rotation), so the lifting has
  less work to do.

**Two files in this refactor let you test the claim rather than take my word:**

- `make_cascade_data.m` — rebuilds the dataset in outer-loop coordinates **from the
  CSVs you already have**. Because a fast rate loop makes `ω_ref ≈ ω`, the measured
  body rates are a valid proxy for the outer-loop input, so no new flights are
  needed for a first comparison. (State the caveat in your report: this assumes
  perfect inner-loop tracking and is therefore optimistic. Re-log with real
  setpoints once a rate loop is flying.)
- `quad_plant_cascade.m` — `quad_plant` wrapped in a 500 Hz rate PI loop with the
  identified allocation, so `KDPC_quad_sim.m` can be pointed at a cascaded plant.

Run the pipeline both ways and compare closed-loop tracking. That comparison is a
far stronger thesis result than either architecture alone.

## 6. Suggested order of work

1. **Fix the data first.** Log dithered flights. Nothing else you do matters as
   much, and it is the cheapest item on this list.
2. Rebuild in cascade coordinates (`make_cascade_data.m`), retrain, compare R²
   *and* closed-loop performance against the flat design.
3. Keep the flat motor-level design as the ablation. Reporting "we tried it at the
   actuator level, here is precisely why it fails — 1.9 cm of reachable motion in
   the horizon, 23× yaw authority deficit, 100 % acausal energy in the
   unstructured `Theta`" is a genuinely useful negative result, and you now have
   the numbers to back every part of it.
