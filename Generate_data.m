%% Generate_data.m  --  drone counterpart of KDPC_RSRFG/Pendulum/Generate_data.m
% The reference simulates a pendulum with a multisine. We instead read logged
% Crazyflie flights, but produce exactly the same two variables: u_data, y_data.
%
%   u : 4 motor speeds [rad/s]
%   y : [x y z yaw]    [m, m, m, rad]
%
% Two differences from the reference, both forced by real data:
%   1. we normalise (the pendulum was already O(1); motor speeds are ~1740)
%   2. we record run_lengths, because these are 15 separate flights and a
%      Hankel window must never span two of them
clc; close all; clear all;

Ts = 0.01;      % 100 Hz

train_files = dir(fullfile('data','train','*.csv'));
test_files  = dir(fullfile('data','test','*.csv'));
assert(~isempty(train_files),'no CSVs in data/train — run from the project folder');

[u_train, y_train, run_lengths]      = load_runs(train_files);
[u_test,  y_test,  run_lengths_test] = load_runs(test_files);

% scaling from TRAIN data only
u_mean = mean(u_train);  u_std = std(u_train);
y_mean = mean(y_train);  y_std = std(y_train);

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

fprintf('train %d samples / %d runs, test %d samples / %d runs\n', ...
    size(u_data,1), numel(run_lengths), size(u_data_test,1), numel(run_lengths_test));
fprintf('hover motor speeds [rad/s]: %s\n', mat2str(round(u_mean,1)));

figure
subplot(3,1,1); plot(y_train(:,1:3)); ylabel('pos [m]'); legend('x','y','z'); grid on
subplot(3,1,2); plot(y_train(:,4));   ylabel('yaw [rad]'); grid on
subplot(3,1,3); plot(u_train);        ylabel('\Omega [rad/s]'); xlabel('sample'); grid on

%% ------------------------------------------------------------------------
function [U, Y, lens] = load_runs(files)
U = []; Y = []; lens = [];
for k = 1:numel(files)
    T   = readtable(fullfile(files(k).folder, files(k).name));
    u   = [T.m1_rads, T.m2_rads, T.m3_rads, T.m4_rads];
    yaw = unwrap(atan2(2*(T.qw.*T.qz + T.qx.*T.qy), 1 - 2*(T.qy.^2 + T.qz.^2)));
    y   = [T.x, T.y, T.z, yaw];
    U = [U; u];  Y = [Y; y];  lens = [lens; height(T)];
end
lens = double(lens);
end
