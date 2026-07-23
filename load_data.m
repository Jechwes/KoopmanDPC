%% load_data.m — nano-drone benchmark CSVs -> normalized data for KDPC pipeline
% Replaces Generate_data.m from the pendulum example.
% Reads data/train/*.csv and data/test/*.csv (Busetto et al., 2026 benchmark,
% Crazyflie 2.1 Brushless, 100 Hz).
%
% Inputs  u: 4 motor speeds [rad/s]           -> normalized deviation from mean (~hover)
% Outputs y: position x,y,z [m] + yaw [rad]   -> normalized deviation from train mean
%           optionally extended with [vx vy vz qx qy] (see include_extra_states)
%
% Saves to data/:
%   u_data.mat, y_data.mat            train data (T x 4 each), normalized
%   u_data_test.mat, y_data_test.mat  test data (melon), normalized w/ TRAIN scalers
%   run_lengths.mat                   samples per run -> Hankel matrices must not
%                                     cross run boundaries (model_learning_quad.py)
%   scalers.mat                       means/stds + Ts to map back to physical units
clc; close all; clear all;

Ts = 0.01;   % 100 Hz

% ----- Output selection --------------------------------------------------
% y always starts with the controlled outputs [x y z yaw]. Set the flag to true
% to append the extra states [vx vy vz qx qy], which make the internal dynamics
% observable to the Koopman lifting. false -> optimize on the 4 controlled
% outputs only. The ordering is fixed as [x y z yaw vx vy vz qx qy] so every
% downstream file just uses the first ny entries.
include_extra_states = false;

train_files = dir(fullfile('data','train','*.csv'));
test_files  = dir(fullfile('data','test','*.csv'));
assert(~isempty(train_files),'No CSVs found in data/train — run from the project folder.')

[u_train, y_train, run_lengths]      = load_runs(train_files, include_extra_states);
[u_test,  y_test,  run_lengths_test] = load_runs(test_files,  include_extra_states);

%% Scaling — statistics from TRAIN data only
u_mean = mean(u_train);    % per-motor mean ~ hover speed
u_std  = std(u_train);
y_mean = mean(y_train);
y_std  = std(y_train);

u_data      = (u_train - u_mean)./u_std;
y_data      = (y_train - y_mean)./y_std;
u_data_test = (u_test  - u_mean)./u_std;
y_data_test = (y_test  - y_mean)./y_std;

save('data/u_data.mat','u_data');
save('data/y_data.mat','y_data');
save('data/u_data_test.mat','u_data_test');
save('data/y_data_test.mat','y_data_test');
save('data/run_lengths.mat','run_lengths','run_lengths_test');
save('data/scalers.mat','u_mean','u_std','y_mean','y_std','Ts');

fprintf('Train: %d samples over %d runs | Test: %d samples over %d runs\n', ...
    size(u_data,1), numel(run_lengths), size(u_data_test,1), numel(run_lengths_test));
fprintf('Hover (mean) motor speeds [rad/s]: %s\n', mat2str(round(u_mean,1)));

%% Quick visual check (physical units)
figure('Name','Training data');
subplot(3,1,1); plot(y_train(:,1:3)); ylabel('pos [m]'); legend('x','y','z'); grid on
subplot(3,1,2); plot(y_train(:,4));   ylabel('yaw [rad]'); grid on
subplot(3,1,3); plot(u_train);        ylabel('\Omega [rad/s]'); xlabel('sample'); grid on

%% ------------------------------------------------------------------------
function [U, Y, lens] = load_runs(files, include_extra)
U = []; Y = []; lens = [];
for k = 1:length(files)
    T = readtable(fullfile(files(k).folder, files(k).name));
    u = [T.m1_rads, T.m2_rads, T.m3_rads, T.m4_rads];
    % yaw from quaternion (ZYX convention), unwrapped to avoid +/-pi jumps
    yaw = atan2(2*(T.qw.*T.qz + T.qx.*T.qy), 1 - 2*(T.qy.^2 + T.qz.^2));
    yaw = unwrap(yaw);
    % controlled outputs: position + yaw
    y = [T.x, T.y, T.z, yaw];
    if include_extra
        % extra states: velocities and roll/pitch quaternion components
        % (make the internal state observable to the lifting)
        y = [y, T.vx, T.vy, T.vz, T.qx, T.qy];
    end
    U = [U; u];  Y = [Y; y];  lens = [lens; height(T)];
end
end
