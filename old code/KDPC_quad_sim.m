%% KDPC_quad_sim.m -- Koopman DPC of the nano-drone, problem (10) of the paper
%
% plant_mode:
%   'lifted_state' Phase 1a. Plant IS the lifted model and the controller sees
%                  z exactly. Zero mismatch -> MUST converge. Machinery check.
%   'network_lift' Phase 1b. Plant is the lifted model, controller re-lifts the
%                  measured i/o history. Tests the output-feedback path.
%   'physics'      Phase 2. Grey-box rigid-body quadrotor (quad_plant.m).
%
% SETPOINT HANDLING (a deliberate departure from the paper)
% ---------------------------------------------------------
% The paper regulates to the origin. The old code inherited that literally, so
% the target was whatever the training-data MEAN position happened to be --
% not something you can fly to. Quadrotor dynamics are exactly invariant to
% translation in (x,y,z), so a position setpoint is obtained for free by
% shifting the measured history before lifting:
%       y_ini <- (y_meas - r)/y_std
% Yaw is only APPROXIMATELY free: the x,y dynamics depend on yaw through the
% rotation matrix, so a large yaw offset moves you off the identified
% operating point. Small yaw setpoints are fine; large ones want the cascade
% (see CASCADE_DEBATE.md).
clc; close all; clear;

cfg = koopman_config();
net = load_net();
load('data/invariant.mat');                 % Theta,A,B,C,D,K,P,Q,R,alpha,u_lb,u_ub,z_box
load('data/scalers.mat');                   % u_mean,u_std,y_mean,y_std,Ts

Tini = cfg.Tini; N = cfg.N; nu = cfg.nu; ny = cfg.ny;
n_basis = net.n_basis;

%% ---- Stale-artifact guards -------------------------------------------
% Every one of these fired on the previous state of the repo: invariant.mat
% had been built with n_basis=80 against a params.mat saying 40, and
% scalers.mat still held 9-element y_mean/y_std against ny=4.
assert(size(A,1) == n_basis, ...
    'invariant.mat has n_basis=%d, net.mat has %d. Re-run Terminal_set_quad.m.', ...
    size(A,1), n_basis);
assert(size(Theta,1) == N*n_basis && size(Theta,2) == n_basis+N*nu, ...
    'Theta is %dx%d, expected %dx%d.', size(Theta,1), size(Theta,2), N*n_basis, n_basis+N*nu);
assert(numel(y_mean) == ny && numel(y_std) == ny, ...
    'scalers hold %d outputs but ny=%d. Re-run load_data.m.', numel(y_mean), ny);
assert(nnz(Theta(1:n_basis, n_basis+nu+1:end)) == 0, ...
    'Theta is not causal -- re-run Terminal_set_quad.m.');

% Validate up front. Previously a typo here fell through both switch
% statements and surfaced 15 lines later as "Unrecognized variable 'y_next'".
plant_mode = cfg.plant_mode;
valid_modes = {'lifted_state','network_lift','physics'};
assert(ischar(plant_mode) || isstring(plant_mode), 'cfg.plant_mode must be a string.');
plant_mode = char(strtrim(plant_mode));
assert(ismember(plant_mode, valid_modes), ...
    ['cfg.plant_mode = ''%s'' is not recognised.\n' ...
     'Valid options (exact spelling, underscore not space): %s'], ...
    plant_mode, strjoin(valid_modes, ', '));
fprintf('plant_mode = %s\n', plant_mode);

Tsim   = cfg.Tsim;
lambda = cfg.lambda;
r_phys = cfg.setpoint(:);                  % [x y z yaw] target, physical units

%% ---- MPC problem (10), built on the CAUSAL multi-step Theta ------------
% Z = Theta*[z0; u]. Because Theta is block lower-triangular in u, this is a
% legitimate multi-step predictor: no future input influences an earlier
% state. The old code discarded Theta and rebuilt predictions from A^j, which
% reintroduces exactly the error propagation the paper set out to remove.
%
% There is also no rho_yaw penalty any more. That hack existed to stop the QP
% exciting the yaw-torque direction "for free" -- a symptom of the acausal
% Theta and of yaw being blown up by z-scoring. Both are fixed upstream.
Psi   = kron(eye(N), R);                   % same R the DARE used
Omega = blkdiag(kron(eye(N-1), Q), P);     % stage cost on z1..z_{N-1}, P on z_N
G     = Theta'*Omega*Theta;  G = (G+G')/2;
Theta1 = Theta(1:n_basis, :);              % predicts z(k+1)
ThetaN = Theta(end-n_basis+1:end, :);      % predicts z(k+N)

u       = sdpvar(N*nu,1);
nl_part = sdpvar(n_basis,1);               % phi(x_ini(k)), from measurement
z_part  = sdpvar(n_basis,1);               % z*_{1|k-1}, predicted last step
u_prev  = sdpvar(nu,1);                    % last applied input, for move suppression
z0      = sdpvar(n_basis,1);
xi      = sdpvar(1,1);
w = [z0; u];

% Move-suppression operator: du = Dm*u - Em*u_prev
Dm = kron(eye(N),eye(nu)) - kron(diag(ones(N-1,1),-1), eye(nu));
Em = [eye(nu); zeros((N-1)*nu, nu)];
du = Dm*u - Em*u_prev;

% Cost (10a): l_N(z_N) + sum_{i=0}^{N-1} l(z_i,u_i) + l_r(z_0).
% Omega covers z_1..z_{N-1} and z_N; z_0 is added explicitly.
% l_r uses eq. (11). By eq. (17), z_0 - phi(x_ini) = xi*e(k), so this is
% identically eq. (18) lambda*xi^2*||e||^2 -- but written this way the
% Hessian is CONSTANT, which keeps it a standard parametric QP.
Psi_du = kron(eye(N), cfg.R_du);
objective = w'*G*w + u'*Psi*u + du'*Psi_du*du + z0'*Q*z0 ...
          + lambda*(z0-nl_part)'*(z0-nl_part);

constraints = [ z0 == (1-xi)*nl_part + xi*z_part;      % (10c)
                0 <= xi <= 1 ];                        % (10f)
constraints = [constraints, repmat(u_lb,N,1) <= u <= repmat(u_ub,N,1) ];   % (10d), inputs
if isfinite(cfg.du_max)                                % slew-rate limit
    du_n = cfg.du_max ./ u_std(:);                     % rad/s per sample -> normalised
    constraints = [constraints, -repmat(du_n,N,1) <= du <= repmat(du_n,N,1)];
end

% Terminal constraint (10e). See koopman_config.m: on this model X_T contains
% 0.0% of the training lifts, so 'hard' is infeasible from step 1 and 'soft'
% is the honest default. With 'off' or 'soft', Theorem 4.2 does not apply --
% report that rather than claiming recursive feasibility.
qp_solver = 'quadprog';
switch cfg.terminal_mode
    case 'hard'
        Psqrt = real(sqrtm(P)); Psqrt = (Psqrt+Psqrt')/2;
        constraints = [constraints, norm(Psqrt*(ThetaN*w)) <= sqrt(alpha)];
        qp_solver = 'sedumi';                          % SOC -> conic solver
    case 'soft'
        Psqrt = real(sqrtm(P)); Psqrt = (Psqrt+Psqrt')/2;
        slack = sdpvar(1,1);
        constraints = [constraints, slack >= 0, ...
                       norm(Psqrt*(ThetaN*w)) <= sqrt(alpha) + slack];
        objective  = objective + cfg.terminal_slack*slack^2;
        qp_solver  = 'sedumi';
    case 'off'
        % terminal cost P only
    otherwise
        error('cfg.terminal_mode must be ''hard'', ''soft'' or ''off''.');
end
if cfg.use_state_box && ~isempty(z_box)                 % (10d), states
    constraints = [constraints, ...
        repmat(-z_box,N,1) <= Theta*w <= repmat(z_box,N,1)];
end

options    = sdpsettings('solver', qp_solver, 'verbose', 0);
disp('compiling controller...');
controller = optimizer(constraints, objective, options, ...
                       {nl_part, z_part, u_prev}, {u, Theta1*w, xi, z0});
disp('controller ready, simulating...');

%% ---- Initial condition -------------------------------------------------
y0_phys = r_phys + cfg.x0_offset(:);           % start offset from the target
y0      = (y0_phys - r_phys) ./ y_std(:);      % error coordinates, normalised
u_ini   = zeros((Tini-1)*nu, 1);               % 0 deviation = hover trim
y_ini   = repmat(y0, Tini, 1);

z_plant = lift([u_ini; y_ini], net);           % lifted-plant modes

if strcmp(plant_mode,'physics')
    load('data/plant_params.mat');             % c_t, Mw, w_bias, g
    prm = struct('c_t',c_t,'Mw',Mw,'w_bias',w_bias,'g',g);
    q0  = [cos(y0_phys(4)/2); 0; 0; sin(y0_phys(4)/2)];
    x_plant  = [y0_phys(1:3); zeros(3,1); q0; zeros(3,1)];
    yaw_prev = y0_phys(4);
end

%% ---- Closed loop -------------------------------------------------------
Y_cl = y0.';  U_cl = zeros(Tsim,nu);  Xi_cl = zeros(Tsim,1);
E_k  = zeros(n_basis, Tsim);  th = zeros(Tsim,1);  Zkm1 = [];
uk_prev = zeros(nu,1);        % hover trim
k_end = Tsim;
for k = 1:Tsim
    switch plant_mode
        case 'lifted_state'
            Nl_part = z_plant;                  % exact state feedback
        case {'network_lift','physics'}
            Nl_part = lift([u_ini; y_ini], net);  % output feedback: re-lift history
    end
    if k < Tini
        Z_part = Nl_part;                       % no prediction available yet
    else
        Z_part = Zkm1;
        E_k(:,k) = Zkm1 - Nl_part;              % e(k), eq. (12)
    end

    tic;
    [OUT, errorcode] = controller({Nl_part, Z_part, uk_prev});
    th(k) = toc;
    assert(errorcode == 0, 'step %d solver failure: %s', k, yalmiperror(errorcode));
    Uk = OUT{1};  Zkm1 = OUT{2};  Xi_cl(k) = OUT{3};
    uk = Uk(1:nu);  uk_prev = uk;
    if mod(k,100)==0, fprintf('step %d/%d\n', k, Tsim); end

    % ------------------------------- PLANT -------------------------------
    switch plant_mode
        case {'lifted_state','network_lift'}
            y_next  = C*z_plant + D*uk;
            z_plant = A*z_plant + B*uk;
        case 'physics'
            u_phys  = uk.*u_std(:) + u_mean(:);
            u_phys  = min(max(u_phys, cfg.omega_min), cfg.omega_max);
            x_plant = quad_plant(x_plant, u_phys, prm, Ts);
            q = x_plant(7:10);
            yaw_raw  = atan2(2*(q(1)*q(4)+q(2)*q(3)), 1-2*(q(3)^2+q(4)^2));
            yaw_phys = yaw_prev + atan2(sin(yaw_raw-yaw_prev), cos(yaw_raw-yaw_prev));
            yaw_prev = yaw_phys;
            y_phys   = [x_plant(1:3); yaw_phys];
            y_next   = (y_phys - r_phys) ./ y_std(:);   % ERROR coordinates
        otherwise
            error('unreachable: plant_mode validated at startup');
    end
    % ---------------------------------------------------------------------

    u_ini = [u_ini(nu+1:end); uk];
    y_ini = [y_ini(ny+1:end); y_next];
    Y_cl(k+1,:) = y_next.';   U_cl(k,:) = uk.';

    % Divergence guard. Without it the rigid-body integrator keeps running
    % after the drone has tumbled, and the plots show -2500 m of "position
    % error" that is really just RK4 breaking down at several hundred rad/s.
    if strcmp(plant_mode,'physics')
        pos_err = norm(y_next(1:3).*y_std(1:3).');
        if pos_err > 10 || norm(x_plant(11:13)) > 50
            fprintf(2, ['\nDIVERGED at t = %.2f s (step %d): |pos err| = %.1f m, ' ...
                '|omega| = %.0f rad/s\n'], k*Ts, k, pos_err, norm(x_plant(11:13)));
            fprintf(2, 'Motor commands were at the rails for %.0f%% of the run.\n', ...
                100*mean(any(abs(U_cl(1:k,:) - u_ub.'/2 - u_lb.'/2) > ...
                             0.98*(u_ub.'-u_lb.')/2, 2)));
            fprintf(2, ['Debug order: (1) set plant_mode=''lifted_state'' -- that MUST\n' ...
                'converge; (2) if it does, this is model extrapolation, so tighten\n' ...
                'cfg.u_validity_pct / raise cfg.R_du; (3) see CASCADE_DEBATE.md on N.\n']);
            k_end = k;  break
        end
    end
end
Tsim = k_end;  Y_cl = Y_cl(1:Tsim+1,:);  U_cl = U_cl(1:Tsim,:);
Xi_cl = Xi_cl(1:Tsim);  E_k = E_k(:,1:Tsim);  th = th(1:Tsim);
t_vec = (0:Tsim)*Ts;
fprintf('mean/max solve time: %.1f / %.1f ms  (budget at %d Hz: %.1f ms)\n', ...
    1e3*mean(th), 1e3*max(th), round(1/Ts), 1e3*Ts);
fprintf('final position error: %.3f m | final yaw error: %.3f rad\n', ...
    norm(Y_cl(end,1:3).*y_std(1:3)), Y_cl(end,4)*y_std(4));

%% ---- Plots (physical units, error from the setpoint) -------------------
Y_phys = Y_cl .* y_std(:).';
U_phys = U_cl .* u_std(:).' + u_mean(:).';

figure('Name','KDPC closed loop');
subplot(3,1,1); plot(t_vec, Y_phys(:,1:3),'LineWidth',1.5); yline(0,'k--'); grid on
ylabel('position error [m]'); legend('x','y','z'); title(sprintf('setpoint = %s', mat2str(r_phys',3)));
subplot(3,1,2); plot(t_vec, Y_phys(:,4),'LineWidth',1.5); yline(0,'k--'); grid on
ylabel('yaw error [rad]');
subplot(3,1,3); plot(t_vec(1:end-1), U_phys,'LineWidth',1); grid on
yline(cfg.omega_min,'r--'); yline(cfg.omega_max,'r--');
ylabel('\Omega [rad/s]'); xlabel('t [s]');

figure('Name','KDPC internals');
subplot(3,1,1); plot(t_vec(1:end-1), Xi_cl,'LineWidth',1.5); grid on
ylabel('\xi'); title('interpolation variable (should fall to 0 once constraints are inactive)');
subplot(3,1,2); plot(t_vec(Tini:end-1), vecnorm(E_k(:,Tini:end)),'LineWidth',1.5); grid on
ylabel('||e(k)||'); title('Koopman one-step model error, eq. (12)');
subplot(3,1,3); plot(t_vec(1:end-1), 1e3*th,'LineWidth',1); grid on
yline(1e3*Ts,'r--'); ylabel('solve time [ms]'); xlabel('t [s]');
