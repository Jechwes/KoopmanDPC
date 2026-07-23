%% KDPC_quad_sim.m — Koopman data-driven predictive control of the nano-drone
% MIMO adaptation of Pendulum/simulation_Xi.m.
%
% PHASE 1 (this file, plant_mode='koopman'): the plant is the learned lifted
%   model itself. This is a sanity check of the entire MPC machinery
%   (dimensions, Theta, P, terminal set, xi-interpolation) with zero
%   plant/model mismatch from physics.
% PHASE 2: set plant_mode='physics' once a Crazyflie rigid-body simulator is
%   added (replace the plant block in the loop).
%
% Task: start hovering at an offset from the target and regulate back.
% Target = the training-data mean position (y=0 in normalized coordinates).
clc; close all; clear all;

%% Load pipeline results
load('data/weight1.mat'); load('data/weight2.mat'); load('data/weight3.mat');
load('data/params.mat');    % Tini, N, n_basis, nu, ny
load('data/invariant.mat'); % Theta, A, B, K, P, alpha, u_bnd
load('data/scalers.mat');   % u_mean,u_std,y_mean,y_std,Ts
weight1 = double(weight1); weight2 = double(weight2); weight3 = double(weight3);
Tini = double(Tini); N = double(N); n_basis = double(n_basis);
nu = double(nu); ny = double(ny);

%% Controller settings (mirror simulation_Xi.m)
Q  = 10*eye(n_basis);          % must match Terminal_set_quad.m
Rc = 0.01*eye(nu);
lambda = 1e+6;                 % penalty coupling the two initial conditions
                               % (1e10 as in the pendulum wrecks conditioning here)
Tsim = 400;                    % simulation steps (4 s at 100 Hz)
use_terminal_set = false;      % true -> SOC terminal constraint (needs sedumi,
                               % much slower per step); false -> terminal cost only
plant_mode = 'physics';   % 'lifted_state' : plant = lifted model, controller
                               %                  sees z_plant exactly (Phase 1a —
                               %                  machinery check, MUST converge)
                               % 'network_lift' : plant = lifted model, controller
                               %                  re-lifts measured history via the
                               %                  network (Phase 1b — output feedback)
                               % 'physics'      : Phase 2, TODO

% initial offset from target in the full output ordering
% [x y z yaw vx vy vz qx qy]; sliced to the active ny (extra states default to 0)
offset_full = [-0.4; 0.4; -0.3; zeros(6,1)];
offset_phys = offset_full(1:ny);

% Approximate linear readout y(k+1) ~ C z(k) + D u(k) from the readout layer
C = weight3(1:ny, 1:n_basis);
D = weight3(1:ny, n_basis+1:n_basis+nu);

%% MPC problem (YALMIP optimizer) — same structure as simulation_Xi.m, but:
%  - Z is eliminated: Z = Theta*[z0;u], with the Hessian Theta'*Omega*Theta
%    precomputed NUMERICALLY (symbolic expansion hangs at this size)
%  - the pendulum's xi^2 penalty is rewritten exactly: the z0-constraint gives
%    xi*(z_part-nl_part) = z0-nl_part, so
%    lambda*||z_part-nl_part||^2*xi^2  ==  lambda*||z0-nl_part||^2
%    -> constant Hessian, standard parametric QP, fast to compile and solve
% Penalize the yaw-torque input direction [+,-,+,-]: yaw is the worst-identified
% channel (weak excitation in the data), so without this the QP excites yaw
% torque "for free", spins the drone out of the training distribution, and the
% estimator collapses.
d_yaw   = [1;-1;1;-1]/2;
rho_yaw = 1000;
Rc_eff  = Rc + rho_yaw*(d_yaw*d_yaw');

Psi   = kron(eye(N), Rc_eff);                    % input cost
Omega = blkdiag(kron(eye(N-1), Q), P);           % state cost + terminal cost P

% Prediction matrices from the ONE-STEP model (A,B), chained over the horizon:
%   Z = calN*z0 + calM*u,   z(k+j) = A^j z0 + sum_i A^(j-i) B u(i-1)
% NOTE: predicting directly with the regressed multistep Theta destabilized the
% loop — regression on correlated flight data gives Theta acausal terms (future
% inputs "affecting" z(k+1)) that the controller exploits but no plant honors.
calN = zeros(N*n_basis, n_basis);
calM = zeros(N*n_basis, N*nu);
for j = 1:N
    calN((j-1)*n_basis+1:j*n_basis, :) = A^j;
    for i = 1:j
        calM((j-1)*n_basis+1:j*n_basis, (i-1)*nu+1:i*nu) = A^(j-i)*B;
    end
end
Hcal  = [calN, calM];                            % [z0;u] -> Z
G     = Hcal'*Omega*Hcal;  G = (G+G')/2;         % numeric Hessian part
Theta1 = Hcal(1:n_basis, :);                     % block predicting z(k+1)
ThetaN = Hcal(end-n_basis+1:end, :);             % block predicting z(k+N)

u       = sdpvar(N*nu,1);          % future inputs (time-major: u(k),u(k+1),...)
nl_part = sdpvar(n_basis,1);       % z from measured history (network)
z_part  = sdpvar(n_basis,1);       % z predicted at previous step
z0      = sdpvar(n_basis,1);       % interpolated initial condition
xi      = sdpvar(1,1);             % interpolation variable
w = [z0; u];

objective = w'*G*w + u'*Psi*u + z0'*Q*z0 ...
          + lambda*(z0-nl_part)'*(z0-nl_part);

constraints = [z0 == (1-xi)*nl_part + xi*z_part;
               0 <= xi <= 1];
U_bnd = repmat(u_bnd(:), N, 1);
constraints = [constraints, -U_bnd <= u <= U_bnd];

if use_terminal_set
    Psqrt = sqrtm(P);  Psqrt = real((Psqrt+Psqrt')/2);
    constraints = [constraints, norm(Psqrt*(ThetaN*w)) <= sqrt(alpha)];
    qp_solver = 'sedumi';          % SOC constraint -> conic solver
else
    qp_solver = 'quadprog';        % pure QP, fast
end
options = sdpsettings('solver', qp_solver, 'verbose', 0);
disp('compiling controller...');
controller = optimizer(constraints, objective, options, {nl_part, z_part}, {u, Theta1*w, xi, z0});
disp('controller ready, simulating...');

%% Initial condition: hovering at an offset, history = constant hover
y0 = offset_phys ./ y_std(:);                  % normalized initial output
u_ini = zeros((Tini-1)*nu, 1);                 % deviation 0 = mean (~hover) input
y_ini = repmat(y0, Tini, 1);                   % time-major stacked history

% plant init (koopman modes): lifted state consistent with that history
z_plant = tanh(weight2*tanh(weight1*[u_ini; y_ini]));

% plant init (physics mode): hovering level at the offset position
if strcmp(plant_mode, 'physics')
    load('data/plant_params.mat');                 % c_t, Mw, w_bias, g
    prm = struct('c_t',c_t, 'Mw',Mw, 'w_bias',w_bias, 'g',g);
    p0   = y_mean(1:3)' + offset_phys(1:3);
    yaw0 = y_mean(4)    + offset_phys(4);
    q0   = [cos(yaw0/2); 0; 0; sin(yaw0/2)];
    x_plant = [p0; zeros(3,1); q0; zeros(3,1)];    % [p; v; q; w]
    yaw_prev = yaw0;                               % for online yaw unwrapping
end

%% Closed-loop simulation
y  = y0;
Y_cl = y0';  U_cl = [];  Xi_cl = [];  E_k = [];  th = [];
Zkm1 = [];
for k = 1:Tsim
    % lifted initial condition seen by the controller
    switch plant_mode
        case 'lifted_state'    % exact state feedback (predictor == plant)
            Nl_part = z_plant;
        otherwise              % output feedback: re-lift measured history
            Nl_part = tanh(weight2*tanh(weight1*[u_ini; y_ini]));
    end
    if k < Tini
        Z_part = Nl_part;
    else
        Z_part = Zkm1;
        E_k = [E_k, Zkm1 - Nl_part];           % prediction/measurement gap
    end

    tic;
    [OUT, errorcode] = controller({Nl_part, Z_part});
    th   = [th; toc];
    Uk   = OUT{1};
    assert(errorcode == 0, 'Step %d solver failure: %s', k, yalmiperror(errorcode));
    Zkm1 = OUT{2};                             % predicted z(k+1) (shifted i.c.)
    Xi_cl(k) = OUT{3};
    uk   = Uk(1:nu);                           % first input of the sequence
    if mod(k,50)==0, fprintf('step %d/%d\n', k, Tsim); end

    % ----------------- PLANT -----------------
    switch plant_mode
        case {'lifted_state','network_lift'}    % Phase 1: plant = lifted model
            y_next  = C*z_plant + D*uk;
            z_plant = A*z_plant + B*uk;
        case 'physics'                          % Phase 2: grey-box quadrotor
            u_phys  = uk .* u_std(:) + u_mean(:);          % de-normalize
            u_phys  = max(u_phys, 0);                      % motors can't reverse
            x_plant = quad_plant(x_plant, u_phys, prm, Ts);
            q = x_plant(7:10);
            yaw_raw  = atan2(2*(q(1)*q(4)+q(2)*q(3)), 1-2*(q(3)^2+q(4)^2));
            yaw_phys = yaw_prev + atan2(sin(yaw_raw-yaw_prev), cos(yaw_raw-yaw_prev));
            yaw_prev = yaw_phys;                           % online unwrap (training used unwrapped yaw)
            % full candidate output ordering [x y z yaw vx vy vz qx qy], sliced to ny
            y_full  = [x_plant(1:3); yaw_phys; x_plant(4:6); q(2); q(3)];
            y_phys  = y_full(1:ny);
            y_next  = (y_phys - y_mean(:)) ./ y_std(:);    % normalize
    end
    % ------------------------------------------

    % shift history buffers (time-major)
    u_ini = [u_ini(nu+1:end); uk];
    y_ini = [y_ini(ny+1:end); y_next];
    y = y_next;

    Y_cl = [Y_cl; y'];
    U_cl = [U_cl; uk'];
end
t_vec = (0:Tsim)*Ts;
fprintf('mean/max solve time: %.1f / %.1f ms\n', 1e3*mean(th), 1e3*max(th));

%% Plots (physical units: deviations from target in m / rad)
Y_phys = Y_cl .* y_std;                        % deviation from target
U_phys = U_cl .* u_std + u_mean;               % motor speeds [rad/s]

figure('Name','KDPC closed loop');
subplot(3,1,1); hold on; grid on
plot(t_vec, Y_phys(:,1:3), 'LineWidth', 1.5); yline(0,'k--');
ylabel('position error [m]'); legend('x','y','z');
subplot(3,1,2); hold on; grid on
plot(t_vec, Y_phys(:,4), 'LineWidth', 1.5); yline(0,'k--');
ylabel('yaw error [rad]');
subplot(3,1,3); hold on; grid on
plot(t_vec(1:end-1), U_phys, 'LineWidth', 1);
ylabel('\Omega [rad/s]'); xlabel('t [s]');

figure('Name','KDPC internals');
subplot(2,1,1); plot(t_vec(1:end-1), Xi_cl, 'LineWidth', 1.5); grid on
ylabel('\xi'); xlabel('t [s]');
subplot(2,1,2); hold on; grid on
if ~isempty(E_k), plot(t_vec(Tini:end-1), vecnorm(E_k), 'LineWidth', 1.5); end
ylabel('||z_{k|k-1} - \phi(data)||'); xlabel('t [s]');
