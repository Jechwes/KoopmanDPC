close all; clear all; clc;

%% N-step ahead predictor -- identification data
% A quadrotor cannot be excited open loop, so the drone is flown by a
% stabilising PID (outer_pid.m) with a white-noise added on top of its
% output. This is what makes the gains identifiable: under pure feedback
% u = -K*x exactly, and least squares cannot separate "the input caused this
% response" from "the state caused this input". 

[prm, lim, inner] = quad_params();
Ts       = 0.05;                 % 20 Hz outer loop
Ts_inner = 0.002;                % 500 Hz rate loop
g        = prm.g;

Period   = 1200;                 % samples per run (60 s)
NumRuns  = 12;
NumTest  = 6;
settle   = 100;                  % hover settling steps before logging

Range_T  = 0.8;                  % dither std, specific thrust [m/s^2]
Range_w  = 0.35;                 % dither std, body rate [rad/s]
Band     = [0.2, 8.0];           % multisine only, unused for white noise
SineData = 40;                   % multisine only, unused for white noise

p_hover  = [0; 0; 1.4];
ref_box  = [1.2; 1.2; 0.5];      % +- m about hover; the difficulty knob
ref_hold = 2.0;                  % s each setpoint is held
yaw_amp  = 0.6;                  % rad

T_clip   = [0.4*g, 1.8*g];
w_clip   = 4.0;

% y = [x y z yaw], the differentially flat outputs. 

cfg = struct('Ts',Ts,'Ts_inner',Ts_inner,'K',Period,'settle',settle, ...
             'p_hover',p_hover,'ref_box',ref_box,'ref_hold',ref_hold, ...
             'yaw_amp',yaw_amp,'amp_T',Range_T,'amp_w',Range_w, ...
             'f_lo',Band(1),'f_hi',Band(2),'n_lines',SineData, ...
             'T_clip',T_clip,'w_clip',w_clip, ...
             'prm',prm,'lim',lim,'inner',inner);

%% ------------------------------------------------------------------------
[u_train, y_full,   run_lengths]      = gather_set(NumRuns, 1,    cfg);
[u_test,  y_full_t, run_lengths_test] = gather_set(NumTest, 1000, cfg);

y_train = y_full(:,   1:4);
y_test  = y_full_t(:, 1:4);

u_mean = mean(u_train);
y_mean = mean(y_train);

u_data      = u_train - u_mean;
y_data      = y_train - y_mean;
u_data_test = u_test  - u_mean;
y_data_test = y_test  - y_mean;

t_train = (0:size(y_data,1)-1)*Ts;

%% ------------------------------------------------------------------------
curr_fig = figure;
curr_axes1=axes('Parent',curr_fig,'FontSize',11,'FontName','Times New Roman');
box(curr_axes1,'on');
hold(curr_axes1,'all');
subplot(2,1,1)
hold on
plot(t_train,u_train,'LineWidth',1)
ylabel('$u(k)$',Interpreter='latex')
legend('T','\omega_x','\omega_y','\omega_z')
axis tight
subplot(2,1,2)
hold on
plot(t_train,y_train(:,1:3),'LineWidth',1)
ylabel('$y(k)$',Interpreter='latex')
xlabel('$t$[s]',Interpreter='latex')
legend('x','y','z')
axis tight
set(gca,'TickLabelInterpreter','Latex')
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
savefig('figures/identification_quad.fig')

%% ------------------------------------------------------------------------
save('data/u_data',"u_data")
save('data/y_data',"y_data")
save('data/u_data_test',"u_data_test")
save('data/y_data_test',"y_data_test")
save('data/run_lengths',"run_lengths","run_lengths_test")
source = 'sim';
save('data/scalers',"u_mean","y_mean","Ts","source","p_hover")

tilt = 2*asin(min(1, hypot(y_full(:,5), y_full(:,6))))*180/pi;
fprintf('gathered %d+%d runs | tilt median %.1f deg, max %.1f deg\n', ...
        NumRuns, NumTest, median(tilt), max(tilt));

%% ------------------------------------------------------------------------
function [U, Y, lens] = gather_set(n_runs, seed0, cfg)
U = []; Y = []; lens = [];
for r = 1:n_runs
    fprintf('gathering data\n')
    [u, y] = gather_run(seed0 + r, cfg);
    U = [U; u];  Y = [Y; y];  lens = [lens; size(y,1)];
end
lens = double(lens);
end

function [U, Y] = gather_run(seed, cfg)
rng(seed);
K = cfg.K;  Ts = cfg.Ts;  prm = cfg.prm;

dT = cfg.amp_T * randn(K,1);
dw = cfg.amp_w * randn(K,3);

% band-limited multisine, kept for comparison. Correlated in time, which lets
% the dense fit of eq. (12) move u(k)'s effect onto later columns of Gamma.
% dT = multisine(K, Ts, cfg.amp_T, cfg.f_lo, cfg.f_hi, cfg.n_lines);
% dw = [multisine(K,Ts,cfg.amp_w,cfg.f_lo,cfg.f_hi,cfg.n_lines), ...
%       multisine(K,Ts,cfg.amp_w,cfg.f_lo,cfg.f_hi,cfg.n_lines), ...
%       multisine(K,Ts,cfg.amp_w,cfg.f_lo,cfg.f_hi,cfg.n_lines)];

x  = [cfg.p_hover; zeros(3,1); 1;0;0;0; zeros(3,1)];
st = [];
for k = 1:cfg.settle
    v0 = outer_pid(x, cfg.p_hover, zeros(3,1), 0, prm);
    [x, st] = quad_plant(x, v0, 0, prm, cfg.lim, cfg.inner, Ts, cfg.Ts_inner, st);
end

nhold = max(1, round(cfg.ref_hold/Ts));
U = zeros(K,4);  Y = zeros(K,6);
p_des = cfg.p_hover;  yaw_des = 0;  yaw_prev = 0;
for k = 1:K
    if mod(k-1, nhold) == 0
        p_des   = cfg.p_hover + (2*rand(3,1)-1).*cfg.ref_box(:);
        yaw_des = (2*rand-1)*cfg.yaw_amp;
    end
    v_pid = outer_pid(x, p_des, zeros(3,1), yaw_des, prm);
    v_cmd = v_pid + [dT(k); dw(k,:).'];
    v_cmd(1)   = min(max(v_cmd(1), cfg.T_clip(1)), cfg.T_clip(2));
    v_cmd(2:4) = min(max(v_cmd(2:4), -cfg.w_clip), cfg.w_clip);

    [x, st] = quad_plant(x, v_cmd, (k-1)*Ts, prm, cfg.lim, cfg.inner, Ts, cfg.Ts_inner, st);

    q  = x(7:10);
    yr = atan2(2*(q(1)*q(4)+q(2)*q(3)), 1-2*(q(3)^2+q(4)^2));
    yaw_prev = yaw_prev + atan2(sin(yr-yaw_prev), cos(yr-yaw_prev));
    U(k,:) = v_cmd.';
    Y(k,:) = [x(1:3).', yaw_prev, q(2), q(3)];
    assert(all(isfinite(x)) && norm(x(1:3)-cfg.p_hover) < 30, ...
           'run diverged at k = %d -- reduce the dither or the reference box', k);
end
end

function s = multisine(K, Ts, amp, f_lo, f_hi, nf) %#ok<DEFNU>
t  = (0:K-1).'*Ts;
f  = linspace(f_lo, f_hi, nf);
ph = 2*pi*rand(1,nf);
s  = sum(sin(2*pi*t*f + ph), 2);
s  = amp * s / std(s);
end
