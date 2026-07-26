%% make_cascade_data.m -- rebuild the dataset for the CASCADE architecture.
%
% Identifies the OUTER-loop plant instead of the motor-level plant:
%
%     u = [T_spec, wx, wy, wz]        (specific thrust + body rates)
%     y = [x, y, z, yaw]
%
% WHY THIS WORKS WITH THE FLIGHTS YOU ALREADY HAVE
% ------------------------------------------------
% You have no logged rate SETPOINTS, only motor speeds and measured states.
% But if the inner rate loop is fast relative to the outer loop (30-50 Hz
% rate bandwidth vs a 100 Hz outer loop), then w_ref ~ w, so the MEASURED
% body rates are a valid proxy for the outer-loop input. Specific thrust is
% reconstructed from the motor speeds via the identified c_t. So the existing
% CSVs are re-usable as-is -- no new flights needed to try the cascade.
%
% Caveat, state it in your report: using measured w as the input assumes
% perfect inner-loop tracking. It biases the identified model optimistically.
% Once you have a rate loop flying, re-log with the actual setpoints.
clc; close all; clear;

cfg = koopman_config();
assert(exist('data/plant_params.mat','file')==2, 'run calibrate_plant.m first');
load('data/plant_params.mat', 'c_t');

train_files = dir(fullfile('data','train','*.csv'));
test_files  = dir(fullfile('data','test','*.csv'));
[u_tr, y_tr, run_lengths]      = load_runs_casc(train_files, c_t);
[u_te, y_te, run_lengths_test] = load_runs_casc(test_files,  c_t);

u_mean = mean(u_tr);   y_mean = mean(y_tr);
p  = cfg.robust_pct;
qu = pctl(u_tr, p);   u_std = (qu(2,:) - qu(1,:))/2;
qy = pctl(y_tr, p);   y_std = (qy(2,:) - qy(1,:))/2;
u_std(u_std<1e-9) = 1;   y_std(y_std<1e-9) = 1;

u_data      = (u_tr - u_mean)./u_std;
y_data      = (y_tr - y_mean)./y_std;
u_data_test = (u_te - u_mean)./u_std;
y_data_test = (y_te - y_mean)./y_std;
Ts = cfg.Ts;  ny = size(y_data,2);

if ~exist('data/cascade','dir'), mkdir('data/cascade'); end
save('data/cascade/u_data.mat','u_data');
save('data/cascade/y_data.mat','y_data');
save('data/cascade/u_data_test.mat','u_data_test');
save('data/cascade/y_data_test.mat','y_data_test');
save('data/cascade/run_lengths.mat','run_lengths','run_lengths_test');
save('data/cascade/scalers.mat','u_mean','u_std','y_mean','y_std','Ts','ny');

fprintf('Cascade dataset: %d train / %d test samples\n', size(u_data,1), size(u_data_test,1));
fprintf('u_mean = %s  (T_spec [m/s^2], rates [rad/s])\n', mat2str(round(u_mean,3)));
fprintf('u_std  = %s\n', mat2str(round(u_std,3)));
fprintf('y_std  = %s\n', mat2str(round(y_std,4)));
disp('To train on this: point load paths at data/cascade/ and re-run the pipeline.');

%% ------------------------------------------------------------------------
function [U, Y, lens] = load_runs_casc(files, c_t)
U = []; Y = []; lens = [];
for k = 1:numel(files)
    T = readtable(fullfile(files(k).folder, files(k).name));
    Om2 = [T.m1_rads, T.m2_rads, T.m3_rads, T.m4_rads].^2;
    T_spec = c_t * sum(Om2,2);                  % specific thrust [m/s^2]
    u = [T_spec, T.wx, T.wy, T.wz];             % outer-loop input
    yaw = unwrap(atan2(2*(T.qw.*T.qz + T.qx.*T.qy), 1 - 2*(T.qy.^2 + T.qz.^2)));
    y = [T.x, T.y, T.z, yaw];
    U = [U; u];  Y = [Y; y];  lens = [lens; height(T)];  %#ok<AGROW>
end
lens = double(lens);
end
