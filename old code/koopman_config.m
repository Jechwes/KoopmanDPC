function cfg = koopman_config()
%KOOPMAN_CONFIG  Single source of truth for the whole KDPC pipeline.
%
% Every MATLAB script and model_learning_quad.py reads its settings from here
% (Python via data/config.mat, written by export_config).  Nothing downstream
% is allowed to redefine Tini, N, n_basis, Q, R or lambda locally -- that is
% what silently desynchronised the old pipeline.
%
% Run  export_config  after changing anything here.

%% ---- Sampling / model structure -----------------------------------------
cfg.Ts      = 0.01;      % 100 Hz (benchmark logging rate)
cfg.Tini    = 10;        % past window feeding the observables (0.09 s of u, 0.10 s of y)
cfg.N       = 20;        % prediction horizon (0.20 s)  <-- see CASCADE_DEBATE.md
cfg.n_basis = 40;        % lifted dimension L

% Controlled outputs, in order. y is ALWAYS stored with the full ordering
% below; ny = numel(out_labels) selects the leading prefix actually used.
cfg.out_full   = {'x','y','z','yaw','vx','vy','vz','qx','qy'};
cfg.out_labels = {'x','y','z','yaw'};
cfg.ny = numel(cfg.out_labels);
cfg.nu = 4;              % 4 motor speeds

%% ---- Cost ---------------------------------------------------------------
% The paper penalises the observables directly (z'Qz).  For a drone we care
% about OUTPUT error, so the default builds Q from the learned readout:
%     Q = C'*Qy*C + q_reg*I
% The q_reg*I term is not cosmetic: Lemma 4.3 needs Q > 0, and C'*Qy*C is
% rank ny = 4 out of n_basis = 40.  Set use_output_cost=false to recover the
% paper's plain Q = q_iso*I.
cfg.use_output_cost = true;
cfg.Qy    = diag([10, 10, 10, 2]);   % [x y z yaw] weights, physical-ish units
cfg.q_reg = 1e-2;                    % regularisation keeping Q positive definite
cfg.q_iso = 10;                      % used only if use_output_cost = false
cfg.R     = 0.01*eye(cfg.nu);

% Regularisation weight on the interpolated initial state, eq. (11)/(18).
% NOTE: 1e6 (the old value) is far too large.  Lemma 4.3 gives
%     V(k+1) - V(k) <= -lambda_min(Q)||z0||^2 + lambda*||e(k+1)||^2,
% so the ISS gain grows LINEARLY in lambda.  lambda only needs to be big
% enough that xi collapses to 0 once constraints are inactive (paper Fig. 2).
cfg.lambda = 1e3;

%% ---- Actuator limits (physical, rad/s) ----------------------------------
% Real motors are bounded [omega_min, omega_max], asymmetric about hover
% (~1740 rad/s).
cfg.omega_min = 400;     % idle, not 0: the mixer never fully stops a rotor
cfg.omega_max = 2800;

% MODEL-VALIDITY BOX -- and this matters more than the actuator box.
% The applied input constraint is the INTERSECTION of the two:
%     u in [omega_min, omega_max]  AND  u within the data support
% 99 % of the training data lives in roughly [1340, 2230] rad/s (~+/-1.5 sigma).
% The actuator box [400, 2800] is [-4.2, +3.3] sigma -- about 3x wider than
% anything ever observed. A tanh lifting saturates outside its data support,
% so out there the model reports almost no sensitivity to u and the QP happily
% commands bang-bang inputs it believes are free. That is precisely the
% motor-rail chattering and loss of control seen in the first physics run.
%
% This is the standard rule for data-driven predictive control: an input
% constraint must encode where the MODEL is valid, not just where the actuator
% can go. Widen only as far as your data support genuinely extends.
cfg.u_validity_pct = [1 99];   % percentile of the training data taken as valid
cfg.u_validity_pad = 1.25;     % mild extrapolation allowance beyond that

%% ---- Move suppression ---------------------------------------------------
% Penalise du = u(k) - u(k-1). Without it a short-horizon MPC on a
% marginally-stable lifted model chatters between the rails: nothing in the
% cost objects to a 2400 rad/s step every 10 ms. Also a proxy for the rotor
% inertia the plant model omits.
cfg.R_du   = 0.5*eye(cfg.nu);   % weight on du; raise if the input still chatters
cfg.du_max = 250;               % hard slew limit [rad/s per sample]; inf to disable

%% ---- Scaling ------------------------------------------------------------
% Yaw is heavy-tailed (peaked near hold, occasional large turns): its std is
% 0.055 rad while single runs span 0.5-0.95 rad, i.e. +/-14 sigma after
% z-scoring.  That makes yaw dominate the training MSE while being the least
% predictable channel -- the direct cause of R^2(yaw) ~ 0.8.  Robust scaling
% (half the 2nd-98th percentile span) fixes the imbalance.
cfg.robust_scaling = true;
cfg.robust_pct     = [2 98];

%% ---- Terminal ingredients (Assumption 4.1) ------------------------------
% MEASURED ON YOUR DATA: 0.0% of the training lifts satisfy z'Pz <= alpha, for
% every (Qy, q_reg) combination tried. The terminal set is empty in practice.
%
% Two compounding reasons:
%   1. max|eig(A)| ~ 1.002, so the LQR P has very large eigenvalues and the
%      sublevel set {z'Pz <= alpha} is a thin sliver;
%   2. an ellipsoid inscribed in a box is exponentially conservative in
%      dimension, and n_basis = 40 is a lot of dimensions.
% Compounding both, N = 20 (0.2 s) cannot steer the drone anywhere near the
% origin anyway -- see CASCADE_DEBATE.md section 1.
%
% So a HARD terminal constraint (10e) makes the QP infeasible at step 1.
% Options, in increasing order of theoretical strength:
%   'off'  - terminal cost only. Fast, always feasible, but Theorem 4.2 does
%            NOT apply: you have no recursive feasibility guarantee. Say so.
%   'soft' - slack variable with a large penalty. Keeps the QP feasible while
%            driving the terminal state toward X_T. Recommended default: it is
%            honest about the situation and still behaves sensibly.
%   'hard' - as in the paper. Only viable once N is long enough to reach X_T
%            (i.e. after moving to the cascade, or N >~ 100).
% Default 'off': 'soft'/'hard' need a CONIC solver (SeDuMi) for the SOC
% constraint, which measured 200-300 ms per step against a 10 ms budget --
% 20-30x too slow to be a controller. 'off' uses quadprog. Since X_T is empty
% anyway (below), 'soft' buys little here. Switch it on for offline studies.
cfg.terminal_mode  = 'off';
cfg.terminal_slack = 1e4;      % penalty on the slack when terminal_mode='soft'
cfg.use_state_box  = false;    % (10d) on the observables: very conservative at
                               % n_basis=40; it alone shrinks alpha by ~20x
cfg.state_box_pct  = [1 99];   % percentile span of phi over the training data

%% ---- Simulation ---------------------------------------------------------
cfg.Tsim       = 600;                      % 6 s
cfg.plant_mode = 'lifted_state';                % 'lifted_state' | 'network_lift' | 'physics'
cfg.setpoint   = [0.5; -0.3; 1.2; 0.0];    % [x y z yaw] target, PHYSICAL units
cfg.x0_offset  = [-0.4; 0.4; -0.3; 0.0];   % initial error from the setpoint
end
