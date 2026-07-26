%% load_data.m -- nano-drone benchmark CSVs -> normalised data for the KDPC pipeline
% Reads data/train/*.csv and data/test/*.csv (Busetto et al. benchmark,
% Crazyflie 2.1 Brushless, 100 Hz).
%
% Inputs  u : 4 motor speeds [rad/s]
% Outputs y : [x y z yaw vx vy vz qx qy]; only the first ny are used, but the
%             full set is always stored so the scalers stay consistent.
%
% CHANGES vs. the previous version
%  * ny comes from koopman_config(), and the saved scalers are TRIMMED to ny.
%    Before, scalers.mat held 9-element y_mean/y_std while ny was 4, so
%    KDPC_quad_sim.m computed (y_phys - y_mean)./y_std with mismatched sizes.
%  * Robust (percentile) scaling by default -- see the yaw discussion in
%    koopman_config.m.
%  * Yaw is unwrapped PER RUN, not per concatenated stream.
clc; close all; clear;

cfg = koopman_config();
Ts  = cfg.Ts;

train_files = dir(fullfile('data','train','*.csv'));
test_files  = dir(fullfile('data','test','*.csv'));
assert(~isempty(train_files), 'No CSVs in data/train -- run from the project folder.');
assert(~isempty(test_files),  'No CSVs in data/test.');

[u_train, y_train, run_lengths]      = load_runs(train_files);
[u_test,  y_test,  run_lengths_test] = load_runs(test_files);

%% Scaling -- statistics from TRAIN data only
u_mean = mean(u_train);
y_mean = mean(y_train);
if cfg.robust_scaling
    % Half the central percentile span. For a Gaussian this is ~2.05*sigma, so
    % weights stay comparable, but heavy-tailed channels (yaw) are no longer
    % blown up to +/-14 "sigma" by a tiny std.
    p = cfg.robust_pct;
    qu = pctl(u_train, p);   u_std = (qu(2,:) - qu(1,:))/2;
    qy = pctl(y_train, p);   y_std = (qy(2,:) - qy(1,:))/2;
else
    u_std = std(u_train);
    y_std = std(y_train);
end
u_std(u_std < 1e-9) = 1;   % guard against a constant channel
y_std(y_std < 1e-9) = 1;

u_data      = (u_train - u_mean)./u_std;
y_data      = (y_train - y_mean)./y_std;
u_data_test = (u_test  - u_mean)./u_std;
y_data_test = (y_test  - y_mean)./y_std;

% Trim outputs to the ny actually used, so no downstream file can pick up a
% stale 9-column scaler against a 4-column model.
ny = cfg.ny;
y_data      = y_data(:, 1:ny);
y_data_test = y_data_test(:, 1:ny);
y_mean      = y_mean(1:ny);
y_std       = y_std(1:ny);

if ~exist('data','dir'), mkdir('data'); end
save('data/u_data.mat','u_data');
save('data/y_data.mat','y_data');
save('data/u_data_test.mat','u_data_test');
save('data/y_data_test.mat','y_data_test');
save('data/run_lengths.mat','run_lengths','run_lengths_test');
save('data/scalers.mat','u_mean','u_std','y_mean','y_std','Ts','ny');

fprintf('Train: %d samples / %d runs | Test: %d samples / %d runs\n', ...
    size(u_data,1), numel(run_lengths), size(u_data_test,1), numel(run_lengths_test));
fprintf('Hover (mean) motor speeds [rad/s]: %s\n', mat2str(round(u_mean,1)));
fprintf('Output scale (%s): %s\n', ternary(cfg.robust_scaling,'robust','std'), ...
    mat2str(round(y_std,4)));

% Normalised actuator range actually spanned by the data -- sanity only; the
% MPC bounds come from the PHYSICAL limits in koopman_config.m.
fprintf('u_data spans [%.2f, %.2f]; config box maps to [%.2f, %.2f]\n', ...
    min(u_data(:)), max(u_data(:)), ...
    min((cfg.omega_min - u_mean)./u_std), max((cfg.omega_max - u_mean)./u_std));

%% Quick visual check (physical units)
figure('Name','Training data');
subplot(3,1,1); plot(y_train(:,1:3)); ylabel('pos [m]'); legend('x','y','z'); grid on
subplot(3,1,2); plot(y_train(:,4));   ylabel('yaw [rad]'); grid on
subplot(3,1,3); plot(u_train);        ylabel('\Omega [rad/s]'); xlabel('sample'); grid on

disp('Saved data/*.mat -- next: export_config, then model_learning_quad.py');

%% ------------------------------------------------------------------------
function [U, Y, lens] = load_runs(files)
U = []; Y = []; lens = [];
for k = 1:numel(files)
    T = readtable(fullfile(files(k).folder, files(k).name));
    u = [T.m1_rads, T.m2_rads, T.m3_rads, T.m4_rads];
    % yaw from quaternion (ZYX), unwrapped WITHIN the run only
    yaw = atan2(2*(T.qw.*T.qz + T.qx.*T.qy), 1 - 2*(T.qy.^2 + T.qz.^2));
    yaw = unwrap(yaw);
    y = [T.x, T.y, T.z, yaw, T.vx, T.vy, T.vz, T.qx, T.qy];
    U = [U; u];  Y = [Y; y];  lens = [lens; height(T)];  %#ok<AGROW>
end
lens = double(lens);
end

function out = ternary(c,a,b)
if c, out = a; else, out = b; end
end
