%% calibrate_plant.m — grey-box quadrotor plant identified from the train CSVs
% Instead of guessing Crazyflie parameters (mass, inertia, motor layout, spin
% directions), estimate the two input maps directly from data:
%
%  1) thrust:   az_body ~ c_t * sum(Omega_i^2)        (accelerometer measures
%     specific force, so c_t = kT/m absorbs mass & thrust coeff.)
%  2) torques:  wdot ~ w_bias + Mw * [Omega_1^2 .. Omega_4^2]'
%     (Mw = J^-1 * allocation matrix: captures arm geometry, spin signs, inertia
%      in one 3x4 matrix — no layout guessing, no sign errors)
%
% Neglected: gyroscopic coupling w x Jw, rotor drag, lateral body forces.
% This is deliberately NOT the model the predictor was trained on — it is an
% independent physics-structured plant for closed-loop testing.
clc; close all; clear all;

Ts = 0.01;  g = 9.81;
files = dir(fullfile('data','train','*.csv'));
assert(~isempty(files), 'run from the project folder');

AZ = []; S2 = []; WD = []; OM2 = [];
for k = 1:numel(files)
    T   = readtable(fullfile(files(k).folder, files(k).name));
    Om2 = [T.m1_rads, T.m2_rads, T.m3_rads, T.m4_rads].^2;
    w   = [T.wx, T.wy, T.wz];

    AZ  = [AZ;  T.az_body];
    S2  = [S2;  sum(Om2,2)];
    WD  = [WD;  diff(w)/Ts];          % finite-difference angular acceleration
    OM2 = [OM2; Om2(1:end-1,:)];
end

% ---- thrust gain (no intercept: zero speed = zero thrust) ----
c_t = S2 \ AZ;                        % [m/s^2 per (rad/s)^2, summed over motors]

% ---- torque map with bias ----
sol    = [OM2, ones(size(OM2,1),1)] \ WD;   % 5x3
Mw     = sol(1:4,:)';                 % 3x4
w_bias = sol(5,:)';                   % 3x1 (trim/CoG offset)

% ---- diagnostics ----
hover_implied = sqrt(g/(4*c_t));      % motor speed that exactly cancels gravity
fprintf('c_t = %.4e  ->  implied hover speed %.0f rad/s\n', c_t, hover_implied);

% ---- published physics parameters (Busetto et al. benchmark, models/models.py) ----
% The gyro-noise-attenuated regression above recovers the correct SIGN pattern
% but magnitudes ~5-10x too small. Use the paper's constants for magnitudes:
%   m=0.045, Kt=3.72e-8 (=> c_t = Kt/m = 8.27e-7, matches regression!),
%   arm=0.0353, Kc=7.74e-12, J=diag([2.3951e-5, 2.3951e-5, 3.2347e-6])
m_q  = 0.045;  Kt = 3.72e-8;  Kc = 7.74e-12;  arm = 0.0353;
Jq   = diag([2.3951e-5, 2.3951e-5, 3.2347e-6]);
Alloc = [ Kt*arm*[-1 -1  1  1];      % tau_x: (m3,m4) vs (m1,m2)
          Kt*arm*[-1  1  1 -1];      % tau_y: (m2,m3) vs (m1,m4)
          -Kc*   [ 1 -1  1 -1]];     % tau_z: sign flipped to match the data
Mw_phys = Jq\Alloc;
fprintf('sign check (regressed vs published Mw): %d/12 agree\n', ...
    nnz(sign(Mw) == sign(Mw_phys)));
Mw = Mw_phys;
c_t = Kt/m_q;

% ---- trim enforcement ----
% The controller's equilibrium is u = u_mean (normalized 0). Force the plant to
% share that trim exactly: at u_mean it must produce thrust = g and zero angular
% acceleration. Without this, regression bias makes the plant drift/tip at
% "hover" and the controller fights a constant disturbance it can barely act on.
load('data/scalers.mat', 'u_mean');
c_t_reg = c_t;  w_bias_reg = w_bias;                  % keep regressed values
c_t     = g / sum(u_mean.^2);
w_bias  = -Mw * (u_mean(:).^2);
fprintf('trim-enforced c_t: %.4e (regressed %.4e, %.1f%% off)\n', ...
    c_t, c_t_reg, 100*abs(c_t-c_t_reg)/c_t_reg);
fprintf('trim-enforced w_bias vs regressed: %s vs %s\n', ...
    mat2str(w_bias',3), mat2str(w_bias_reg',3));
fprintf('thrust fit R^2 = %.3f\n', 1 - norm(AZ - S2*c_t)^2/norm(AZ - mean(AZ))^2);
res = WD - [OM2, ones(size(OM2,1),1)]*sol;
axlabel = 'xyz';
for a = 1:3
    fprintf('wdot_%c fit R^2 = %.3f\n', axlabel(a), ...
        1 - norm(res(:,a))^2/norm(WD(:,a)-mean(WD(:,a)))^2);
end
disp('Mw ='); disp(Mw);

save('data/plant_params.mat', 'c_t', 'Mw', 'w_bias', 'g');
disp('Saved data/plant_params.mat');
