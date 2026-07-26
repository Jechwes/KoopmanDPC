%% Terminal_set_quad.m -- lifted predictor + terminal ingredients (paper Sec. III, IV)
%
%  1) evaluate the equilibrium-shifted observables along the training data
%  2) regress the multi-step predictor Theta : [z(k); u_f] -> [z(k+1); ...; z(k+N)]
%     *** with the CAUSAL block-triangular structure of eq. (4) enforced ***
%  3) extract A, B and the readout C, D
%  4) terminal cost P and gain K from the DARE (satisfies LMI (13a) with equality)
%  5) ellipsoidal terminal set z'Pz <= alpha satisfying (13b)
%
% ============================ THE MAIN FIX =================================
% The previous version solved the UNSTRUCTURED least squares of eq. (8),
%       Theta = Phi_f / Phi_p,
% which is what the paper literally writes -- but Remark 3.1 warns that this
% "does not ensure the desired structure for Gamma and Psi in (4)".  On this
% dataset that warning is catastrophic.  Measured on your data, the block row
% of Theta that predicts z(k+1) puts
%       ||coefficients on u(k+1..k+N-1)|| = 9511
%       ||coefficients on [z(k), u(k)]||  =    9.8
% i.e. 100.0% of that row's energy sits on inputs that have not happened yet.
% Flight data is strongly autocorrelated, so LS happily "predicts" z(k+1) from
% future inputs.  No plant honours that, and the MPC exploits it -- which is
% exactly the instability the old KDPC_quad_sim.m worked around by throwing
% Theta away and chaining A^j instead.
%
% Remark 3.1 offers the correct fix, and we take it: regress each block row j
% ONLY on [z(k); u(k..k+j-1)].  This
%   (a) is causal by construction, so it is usable in the MPC;
%   (b) preserves the nested structure of (4) that Theorem 4.2's shifted-
%       sequence argument relies on -- recursive feasibility needs it;
%   (c) still minimises MULTI-STEP error, which is the whole point of the
%       paper, unlike A^j chaining which reintroduces error propagation.
% Held-out check on your data: multi-step lifted R^2 0.9557 (causal) vs
% 0.9432 (A^j chaining); at step 20 the relative error is 0.283 vs 0.322.
% ===========================================================================
clc; close all; clear;

cfg = koopman_config();
net = load_net();
load('data/u_data.mat');  load('data/y_data.mat');  load('data/run_lengths.mat');
load('data/scalers.mat', 'u_mean','u_std','y_mean','y_std');

Tini = cfg.Tini; N = cfg.N; nu = cfg.nu; ny = cfg.ny;
n_basis = net.n_basis;
assert(n_basis == cfg.n_basis, ...
    'net.mat has n_basis=%d but config says %d -- stale network, retrain.', ...
    n_basis, cfg.n_basis);
assert(size(y_data,2) == ny, 'y_data has %d cols, config ny=%d. Re-run load_data.m.', ...
    size(y_data,2), ny);
assert(net.in1 == (Tini-1)*nu + Tini*ny, 'network input width disagrees with Tini/nu/ny.');

stride = 3;   % subsample regression columns (speed/memory); windows overlap heavily

%% 1) lifted observables along the training data
[X1, UF, YF, col_lens] = build_regressors(u_data, y_data, run_lengths, Tini, N);
Z = lift(X1, net);                       % n_basis x M, zero at the equilibrium
fprintf('lifted %d windows | ||z||_rms = %.3f | max|z| = %.3f\n', ...
    size(Z,2), sqrt(mean(Z(:).^2)), max(abs(Z(:))));

%% 2) causal multi-step regression, per block row
% Assemble, per run, the pairs ( [z(k); u_f(k)] , [z(k+1); ...; z(k+N)] ).
Pp = {}; Pf = {};
off = 0;
for rr = 1:numel(col_lens)
    M   = col_lens(rr);
    Zr  = Z(:,  off+(1:M));
    Ur  = UF(:, off+(1:M));
    idx = 1:stride:(M-N);
    Pp{rr} = [Zr(:,idx); Ur(:,idx)];                                    %#ok<SAGROW>
    Pfr = zeros(N*n_basis, numel(idx));
    for jj = 1:numel(idx)
        zf = Zr(:, idx(jj)+1 : idx(jj)+N);
        Pfr(:,jj) = zf(:);
    end
    Pf{rr} = Pfr;                                                       %#ok<SAGROW>
    off = off + M;
end
Pp = cell2mat(Pp);  Pf = cell2mat(Pf);
fprintf('regression: %d samples, Theta is %dx%d\n', size(Pp,2), N*n_basis, n_basis+N*nu);

Theta = zeros(N*n_basis, n_basis + N*nu);
lam_ridge = 1e-8 * size(Pp,2);      % tiny ridge: Z columns are near-collinear
for j = 1:N
    cols = [1:n_basis, n_basis+(1:j*nu)];       % z(k) and u(k..k+j-1) ONLY
    Ppj  = Pp(cols,:);
    rows = (j-1)*n_basis + (1:n_basis);
    Theta(rows, cols) = ((Ppj*Ppj' + lam_ridge*eye(numel(cols))) \ (Ppj*Pf(rows,:)'))';
end
res = norm(Theta*Pp - Pf,'fro')/norm(Pf,'fro');
fprintf('causal Theta relative residual: %.4f\n', res);
assert(nnz(Theta(1:n_basis, n_basis+nu+1:end)) == 0, 'causality structure violated');

A = Theta(1:n_basis, 1:n_basis);
B = Theta(1:n_basis, n_basis+(1:nu));
fprintf('open-loop max|eig(A)| = %.4f | rank(ctrb) = %d/%d\n', ...
    max(abs(eig(A))), rank(ctrb(A,B)), n_basis);

%% 3) output readout  y(k+1) ~ C z(k) + D u(k)
% Fitted directly from data rather than sliced out of the network's third
% layer.  The old code took C = weight3(1:ny,1:n_basis) and silently dropped
% the layer's bias, so C*z + D*u did not reproduce the network's own output.
Yp1 = YF(1:ny, :);                                  % y(k+1) for every window
Reg = [Z; UF(1:nu,:)];
CD  = (Yp1 * Reg') / (Reg*Reg' + 1e-8*eye(n_basis+nu));
C = CD(:, 1:n_basis);   D = CD(:, n_basis+(1:nu));
fprintf('readout fit R^2 per channel:');
for c = 1:ny
    e = Yp1(c,:) - CD(c,:)*Reg;
    fprintf('  %s %.4f', cfg.out_labels{c}, 1 - sum(e.^2)/sum((Yp1(c,:)-mean(Yp1(c,:))).^2));
end
fprintf('\n');

%% 4) stage cost, terminal cost P and gain K
if cfg.use_output_cost
    Q = C'*cfg.Qy*C + cfg.q_reg*eye(n_basis);   % penalise OUTPUT error, not observables
else
    Q = cfg.q_iso*eye(n_basis);
end
Q = (Q+Q')/2;
R = cfg.R;
assert(min(eig(Q)) > 0, 'Q must be positive definite (Lemma 4.3); raise cfg.q_reg.');

% The DARE solution satisfies LMI (13a) with equality and is numerically far
% more robust than an SDP at n_basis=40 with max|eig(A)| ~ 1.
[P, K_dare] = idare(A, B, Q, R);
assert(~isempty(P), 'idare failed: (A,B) not stabilisable / too ill-conditioned.');
P = (P+P')/2;
K = -K_dare;                                   % convention u = K z
fprintf('closed-loop max|eig(A+BK)| = %.4f | cond(P) = %.2e\n', ...
    max(abs(eig(A+B*K))), cond(P));
assert(max(abs(eig(A+B*K))) < 1, 'terminal controller is not stabilising.');

%% 5) terminal set  X_T = { z : z'Pz <= alpha }   satisfying (13b)
% PHYSICAL actuator box, mapped to normalised coordinates. This is asymmetric
% about hover, so the MPC gets the true box while alpha uses the largest
% SYMMETRIC box inscribed in it (the support-function bound below assumes
% symmetry).  The old code derived u_bnd from the training-data range, giving
% a +/-5 sigma "limit" that could never activate.
u_lb_act = ((cfg.omega_min - u_mean(:)) ./ u_std(:));
u_ub_act = ((cfg.omega_max - u_mean(:)) ./ u_std(:));

% Model-validity box from the data support (see koopman_config.m). The applied
% constraint is the INTERSECTION with the actuator box -- almost always the
% validity box binds, and it is what stops the QP bang-banging into a region
% where the tanh lifting has saturated and reports free control authority.
qv   = pctl(u_data, cfg.u_validity_pct);
u_lb_val = cfg.u_validity_pad * qv(1,:).';
u_ub_val = cfg.u_validity_pad * qv(2,:).';
u_lb = max(u_lb_act, u_lb_val);
u_ub = min(u_ub_act, u_ub_val);
assert(all(u_lb < 0 & u_ub > 0), 'hover trim is outside the admissible input box.');
u_bnd = min(abs(u_lb), u_ub);                  % inscribed symmetric box
fprintf('input box: actuator [%.2f %.2f], validity [%.2f %.2f] -> applied [%.2f %.2f] (sigma)\n', ...
    mean(u_lb_act), mean(u_ub_act), mean(u_lb_val), mean(u_ub_val), mean(u_lb), mean(u_ub));
fprintf('           applied box in physical units: [%.0f, %.0f] rad/s\n', ...
    mean(u_lb.*u_std(:) + u_mean(:)), mean(u_ub.*u_std(:) + u_mean(:)));

% K X_T subseteq U :  max_{z'Pz<=alpha} |K_i z| = sqrt(alpha * K_i P^-1 K_i')
alpha_u = min( (u_bnd.^2) ./ diag(K*(P\K')) );

% X_T subseteq X_z :  max_{z'Pz<=alpha} |e_i'z| = sqrt(alpha * (P^-1)_ii)
if cfg.use_state_box
    % EXACT bound, borrowed from the reference implementation (Pendulum/
    % Terminal_set.m uses Ax=[I;-I], bx=1, i.e. |z_i| <= 1). The observables
    % are tanh outputs, so |phi_i| <= 1 by construction; after the
    % equilibrium shift phi~ = phi - z_eq the exact bound is 1 + |z_eq_i|.
    % This is strictly better than a percentile of the data: it is a property
    % of the network, valid everywhere, not an estimate from a finite sample.
    z_box_exact = 1 + abs(net.z_eq);
    z_box_data  = max(abs(pctl(Z.', cfg.state_box_pct)), [], 1).';
    z_box       = z_box_exact;                  % exact, network-derived bound
    fprintf('X_z box: exact tanh bound mean %.3f (data p%d-p%d span mean %.3f)\n', ...
        mean(z_box_exact), cfg.state_box_pct(1), cfg.state_box_pct(2), mean(z_box_data));
    Pinv_d = diag(inv(P));
    alpha_z = min( (z_box.^2) ./ Pinv_d );
else
    z_box = []; alpha_z = inf;
end
alpha = min(alpha_u, alpha_z);
fprintf('alpha = %.4f  (input-limited %.4f, state-box-limited %.4f)\n', ...
    alpha, alpha_u, alpha_z);
fprintf('u box (normalised): lb %s  ub %s\n', ...
    mat2str(round(u_lb',2)), mat2str(round(u_ub',2)));

% How much of the training data actually sits inside X_T?
frac = mean(sum((Z'*P).*Z', 2) <= alpha);
fprintf('fraction of training lifts inside X_T: %.2f%%\n', 100*frac);
if frac < 0.01
    warning(['X_T is effectively empty (%.2f%% of lifts inside).\n' ...
        '  A HARD terminal constraint will make the QP infeasible at step 1.\n' ...
        '  Causes: max|eig(A)| = %.4f (large P), and an ellipsoid inscribed in\n' ...
        '  a %d-dimensional box is very conservative. Use cfg.terminal_mode =\n' ...
        '  ''soft'' or ''off'', and note that Theorem 4.2 then does not apply.\n' ...
        '  See CASCADE_DEBATE.md section 1 -- N=%d is too short to reach X_T.'], ...
        100*frac, max(abs(eig(A))), n_basis, N);
end

save('data/invariant.mat', 'Theta','A','B','C','D','K','P','Q','R', ...
     'alpha','u_bnd','u_lb','u_ub','z_box','n_basis','N','nu','ny','Tini');
disp('Saved data/invariant.mat -- next: check_pipeline, then KDPC_quad_sim.m');
