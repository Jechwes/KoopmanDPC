%% calibrate_plant.m -- grey-box quadrotor plant for closed-loop testing.
%
% Deliberately NOT the model the predictor was trained on: it is an
% independent, physics-structured plant so that 'physics' mode in
% KDPC_quad_sim.m exercises real plant/model mismatch.
%
%  1) thrust:  az_body ~ c_t * sum(Omega_i^2)   (c_t = Kt/m absorbs mass)
%  2) torques: wdot    ~ w_bias + Mw*[Omega_i^2]   (Mw = J^-1 * allocation)
%
% Neglected: gyroscopic coupling w x Jw, rotor drag, lateral body forces,
% motor lag. See CASCADE_DEBATE.md -- motor lag in particular is why a
% 100 Hz motor-level controller is optimistic here.
%
% CHANGE: constants now come from koopman_config()/scalers.mat rather than
% being duplicated; the regression is kept only as a SIGN check on the
% published allocation, which is what it was actually being used for.
clc; close all; clear;

cfg = koopman_config();
Ts = cfg.Ts;  g = 9.81;
files = dir(fullfile('data','train','*.csv'));
assert(~isempty(files), 'run from the project folder');

AZ = []; S2 = []; WD = []; OM2 = [];
for k = 1:numel(files)
    T   = readtable(fullfile(files(k).folder, files(k).name));
    Om2 = [T.m1_rads, T.m2_rads, T.m3_rads, T.m4_rads].^2;
    w   = [T.wx, T.wy, T.wz];
    AZ  = [AZ;  T.az_body];        %#ok<AGROW>
    S2  = [S2;  sum(Om2,2)];       %#ok<AGROW>
    WD  = [WD;  diff(w)/Ts];       %#ok<AGROW>
    OM2 = [OM2; Om2(1:end-1,:)];   %#ok<AGROW>
end

%% regressed estimates (used for validation only)
c_t_reg = S2 \ AZ;
sol     = [OM2, ones(size(OM2,1),1)] \ WD;
Mw_reg  = sol(1:4,:)';
w_bias_reg = sol(5,:)';

%% published physics constants (Busetto et al. benchmark, models/models.py)
m_q = 0.045;  Kt = 3.72e-8;  Kc = 7.74e-12;  arm = 0.0353;
Jq  = diag([2.3951e-5, 2.3951e-5, 3.2347e-6]);
Alloc = [ Kt*arm*[-1 -1  1  1];      % tau_x
          Kt*arm*[-1  1  1 -1];      % tau_y
          -Kc*   [ 1 -1  1 -1]];     % tau_z (sign matched to the data)
Mw  = Jq\Alloc;
c_t = Kt/m_q;
fprintf('c_t: published %.4e vs regressed %.4e (%.1f%% apart)\n', ...
    c_t, c_t_reg, 100*abs(c_t-c_t_reg)/c_t_reg);
fprintf('allocation sign agreement (regressed vs published): %d/12\n', ...
    nnz(sign(Mw_reg) == sign(Mw)));

% Yaw authority is intrinsically ~23x weaker than roll/pitch here:
fprintf('|Mw| row norms  roll %.2e  pitch %.2e  yaw %.2e  (yaw/roll = %.3f)\n', ...
    norm(Mw(1,:)), norm(Mw(2,:)), norm(Mw(3,:)), norm(Mw(3,:))/norm(Mw(1,:)));

%% trim enforcement
% The controller's equilibrium is u = u_mean (normalised 0). Force the plant
% to share that trim exactly, otherwise it drifts at "hover" and the
% controller fights a constant disturbance it can barely act on.
load('data/scalers.mat', 'u_mean');
c_t    = g / sum(u_mean.^2);
w_bias = -Mw * (u_mean(:).^2);
fprintf('trim-enforced c_t = %.4e | w_bias = %s\n', c_t, mat2str(w_bias',3));
fprintf('thrust fit R^2 = %.3f\n', 1 - norm(AZ - S2*c_t_reg)^2/norm(AZ-mean(AZ))^2);
res = WD - [OM2, ones(size(OM2,1),1)]*sol;
lbl = 'xyz';
for a = 1:3
    fprintf('wdot_%c fit R^2 = %.3f\n', lbl(a), 1 - norm(res(:,a))^2/norm(WD(:,a)-mean(WD(:,a)))^2);
end

save('data/plant_params.mat', 'c_t', 'Mw', 'w_bias', 'g');
disp('Saved data/plant_params.mat');
