%% KDPC with an inner-loop attitude rate PID on simulated flight data
clc; clear;
% close all;
 
%% load params and data, set weights and time
% THE SWITCH. false -> phi = identity, i.e. plain subspace predictive control in
% this same formulation. true -> append the physics observables of phi_nl below.
% Everything else is untouched, so any difference is due to the lifting alone.
use_phi = true;
 
% load params, set sim rate and time
[prm, lim, inner] = quad_params();
Ts_inner = 0.002;                         % inner rate loop, 500 Hz
t_end    = 40;
 
% Generate_data logs y = [x y z yaw], the differentially flat outputs. Attitude
% is not measured, so phi_nl reconstructs the tilt from the position lags: the
% acceleration a ~ alpha_T*b3 - g*e3 stands in for the thrust direction.
% That costs a second difference, whose noise gain is sqrt(6)/Ts^2 ~ 980 -- fine
% here because the simulated positions are exact, but it will not survive real
% flight data, where the same estimate has SNR ~ 1.1. If you ever want this on
% hardware, log the attitude the flight controller already estimates and build
% a from that instead.
idx_w = [1 2 3 4];                       % channels in the window and predicted
 
% cost weights, physical units, one entry per PREDICTED channel
Qy_phys = [10; 10; 10; 1];               % x y z yaw
Ru_phys = [0.1; 2.0; 2.0; 2.0];          % T wx wy wz
N    = 20;
Tini = 6;
 
load('data/u_data.mat')
load('data/y_data.mat')
load('data/run_lengths.mat')
load('data/scalers.mat')
 
assert(strcmp(source, 'sim'), ...
       'scalers were saved with source = ''%s'', expected ''sim''. Rerun Generate_data.m', source);
 
w_term = 10;                              % weight multiplier on the last output block
 
nu  = size(u_data,2);
ny  = size(y_data,2);                     % MEASURED channels, 6
nyw = numel(idx_w);                       % windowed / predicted channels, 4
now = nyw - 3;                            % non-position channels in the window, 1
nU  = (Tini-1)*nu;                        % past inputs   u(k-Tini+1..k-1)
nP  = (Tini-1)*3;                         % past position differences
nO  = Tini*now;                           % past yaw
n_ini = nU + nP + nO;
L     = n_ini + nphi(use_phi);
k_sim = round(t_end/Ts);
 
assert(ny == 4, 'expected y = [x y z yaw] from Generate_data.m, got ny = %d', ny);
assert(numel(Qy_phys) == nyw, 'Qy_phys has %d entries but %d channels are predicted', ...
       numel(Qy_phys), nyw);
assert(Tini >= 3, 'phi_nl needs three position lags for the second difference');
 
fprintf('KDPC | use_phi %d | Tini %d | N %d | measured %d, predicted %d | n_ini %d | L %d\n', ...
        use_phi, Tini, N, ny, nyw, n_ini, L);
 
%% set ref
% reference, figure 8 increasing in speed
% Note: Increase in frequency requires more tilt thus more nonlinearities.
A_sp = 1.2;  B_sp = 0.8;  C_sp = 0;  w_sp = 0.4;
r_pos = @(t) p_hover(:) + [A_sp*sin(w_sp*t); B_sp*sin(2*w_sp*t); C_sp*sin(w_sp*t)];
 
% derivatives, used to give drone correct initial condition
r_vel = @(t) [ A_sp*w_sp  *cos(w_sp*t);  2*B_sp*w_sp  *cos(2*w_sp*t);  C_sp*w_sp  *cos(w_sp*t)];
r_acc = @(t) [-A_sp*w_sp^2*sin(w_sp*t); -4*B_sp*w_sp^2*sin(2*w_sp*t); -C_sp*w_sp^2*sin(w_sp*t)];
r_jrk = @(t) [-A_sp*w_sp^3*cos(w_sp*t); -8*B_sp*w_sp^3*cos(2*w_sp*t); -C_sp*w_sp^3*cos(w_sp*t)];
 
%% load X,Y, learn the theta state model and calc Pz and Gamma
[Zini, UF, YF] = koopman_windows(u_data, y_data, run_lengths, Tini, N, use_phi, Ts, idx_w);
 
X  = [Zini; UF];
Yt = YF;
 
% learn the theta state model
Theta_KDPC = Yt*pinv(X);
save('data/KDPCTheta', 'Theta_KDPC', 'Tini', 'N', 'L', 'use_phi')
 
% end state definition
Pz    = Theta_KDPC(:, 1:L);
Gamma = Theta_KDPC(:, L+1:end);
 
%% Simulate the system
Q = diag(Qy_phys(:));
R = diag(Ru_phys(:));
 
Psi   = kron(eye(N), R);
Omega = blkdiag(kron(eye(N-1), Q), w_term*Q);
 
uVar = sdpvar(N*nu,1);
yVar = sdpvar(N*nyw,1);
zini = sdpvar(L,1);                       % parameter: phi(x_ini(k))
rVar = sdpvar(N*nyw,1);                   % parameter: reference over the horizon
% uini = sdpvar(nU,1);
 
objective   = (yVar - rVar)'*Omega*(yVar - rVar) + uVar'*Psi*uVar;
% objective = yVar'*Omega*yVar + uVar'*Psi*uVar
constraints = [yVar == Pz*zini + Gamma*uVar];
 
% input box: the actuator
u_lo = [lim.T_spec(1); -lim.w_cmd*ones(3,1)] - u_mean(:);
u_hi = [lim.T_spec(2);  lim.w_cmd*ones(3,1)] - u_mean(:);
constraints = [constraints, repmat(u_lo,N,1) <= uVar <= repmat(u_hi,N,1)];
 
Parameters = {zini, rVar};
% Parameters = {uini,yini};
Outputs    = {uVar, yVar};
 
options = sdpsettings('solver','QUADPROG','verbose',0,'debug',0);
controller = optimizer(constraints, objective, options, Parameters, Outputs);
 
%% initialize
% start on the trajectory
a0 = r_acc(0);  j0 = r_jrk(0);
zb = [a0(1); a0(2); a0(3) + prm.g];  zb = zb/norm(zb);   % body z axis
sq = sqrt(2*(1 + zb(3)));                                % q rotating [0;0;1] -> zb
q0 = [sq/2; cross([0;0;1], zb)/sq];  q0 = q0/norm(q0);   % zero yaw
om0 = [-j0(2); j0(1); 0]/prm.g;                          % small-angle jerk -> rate
x  = [r_pos(0); r_vel(0); q0; om0];
 
st = [];
yaw_prev = 0;
 
% Buffers hold the last raw measurements; the window is formed from them each
% step. Seeded from the reference, which is exact since the drone starts on it.
Ybuf = zeros(Tini, ny);  Ubuf = zeros(Tini-1, nu);
for b = 1:Tini
    Ybuf(b,:) = ref_meas(-(Tini-b)*Ts, r_pos, r_acc, prm.g).' - y_mean(:).';
end
for b = 1:Tini-1
    Ubuf(b,:) = ref_u(-(Tini-b)*Ts, r_acc, r_jrk, prm.g).' - u_mean(:).';
end
 
uKDPC = zeros(nu, k_sim);
yKDPC = zeros(ny, k_sim);
th    = zeros(k_sim,1);
 
% DIAGNOSTIC. yhat1 is what the model said y(k+1) would be, logged at k; yact1 is
% what actually happened. If these agree, the model and the wiring are sound and
% the fault is in the cost or the reference. If they disagree, the fault is
% upstream and the channel will show which. usat counts how often the command
% sits on the input box -- a saturated controller cannot track no matter how good
% the model is.
yhat1 = zeros(nyw, k_sim);
yact1 = zeros(nyw, k_sim);
usat  = zeros(nu,  k_sim);
X_log = zeros(k_sim+1,13);  X_log(1,:) = x.';
P_ref = zeros(k_sim+1,3);   P_ref(1,:) = r_pos(0).';
t_KDPC = (0:k_sim)*Ts;
n_ok  = k_sim;
 
%% main loop
for i = 1:k_sim
    fprintf("running step %d of %d \n", i , k_sim)
 
    tic;
    t = (i-1)*Ts;
 
    x_ini = ti_window(Ubuf, Ybuf, idx_w);
    z_k   = [x_ini; phi_nl(Ubuf, Ybuf, Ts, use_phi)];
 
    % reference in the coordinates the predictor returns: displacement from the
    % current position for the position channels, mean-removed absolute for the
    % rest. p(k) and y_mean cancel out of the position part entirely.
    p_now = x(1:3);
    rvec  = zeros(N*nyw,1);
    for j = 1:N
        rj = ref_meas(t + j*Ts, r_pos, r_acc, prm.g);
        rj = rj(idx_w);
        rvec((j-1)*nyw + (1:3))   = rj(1:3) - p_now;
        rvec((j-1)*nyw + (4:nyw)) = rj(4:nyw) - y_mean(idx_w(4:nyw)).';
    end
 
    OUT = controller({z_k, rvec});
 
    Uk = OUT{1};
    Yp = OUT{2};                                    % predicted y(k+1..k+N)
    yhat1(:,i) = Yp(1:nyw);                         % the one-step prediction
 
    u_mpc     = Uk(1:nu);
    uKDPC(:,i) = u_mpc;
    usat(:,i) = (u_mpc <= u_lo + 1e-9) - (u_mpc >= u_hi - 1e-9);
    th(i)     = toc;
 
    % output update
    v_cmd = u_mpc + u_mean(:);                      % [T_spec; om_ref]
    [x, st] = quad_plant(x, v_cmd, t, prm, lim, inner, Ts, Ts_inner, st);
 
    q  = x(7:10);
    yr = atan2(2*(q(1)*q(4)+q(2)*q(3)), 1-2*(q(3)^2+q(4)^2));
    yaw_prev = yaw_prev + atan2(sin(yr-yaw_prev), cos(yr-yaw_prev));
 
    y_next = measure(x, yaw_prev, y_mean);
 
    % what actually happened, in the SAME coordinates the predictor returns:
    % displacement from p(k) for position, mean-removed absolute for yaw
    yw_next    = y_next(idx_w);
    yact1(:,i) = [x(1:3) - p_now; yw_next(4:nyw)];
 
    Ybuf = [Ybuf(2:end,:); y_next.'];
    Ubuf = [Ubuf(2:end,:); u_mpc.'];
 
    yKDPC(:,i)   = y_next;
    X_log(i+1,:) = x.';
    P_ref(i+1,:) = r_pos(i*Ts).';
 
    % sim stops when the drone diverges to much
    if ~all(isfinite(x)) || norm(x(1:3) - r_pos(i*Ts)) > 50
        warning('diverged at t = %.2f s', i*Ts);
        n_ok = i;  break
    end
end
 
yhat1 = yhat1(:,1:n_ok);  yact1 = yact1(:,1:n_ok);  usat = usat(:,1:n_ok);
uKDPC = uKDPC(:,1:n_ok);  yKDPC = yKDPC(:,1:n_ok);  th = th(1:n_ok);
X_log = X_log(1:n_ok+1,:);  P_ref = P_ref(1:n_ok+1,:);  t_KDPC = t_KDPC(1:n_ok+1);
 
%% New plot
e = X_log(:,1:3) - P_ref;
fprintf('\nuse_phi = %d | Tini = %d | N = %d | Ts = %.3f s\n', use_phi, Tini, N, Ts);
fprintf('RMSE x %.4f  y %.4f  z %.4f m   |  3D %.4f m\n', ...
        sqrt(mean(e(:,1).^2)), sqrt(mean(e(:,2).^2)), sqrt(mean(e(:,3).^2)), ...
        sqrt(mean(sum(e.^2,2))));
fprintf('solve: mean %.1f ms, max %.1f ms\n', 1e3*mean(th), 1e3*max(th));
 
curr_fig = figure;
curr_axes1=axes('Parent',curr_fig,'FontSize',11,'FontName','Times New Roman');
box(curr_axes1,'on');
hold(curr_axes1,'all');
subplot(2,1,1)
hold on;
plot(t_KDPC,P_ref,'LineWidth',2,'LineStyle','--');
plot(t_KDPC,X_log(:,1:3),'LineWidth',2);
legend('x_{ref}','y_{ref}','z_{ref}','x','y','z','Location','northeast');
ylabel('$p$ [m]',Interpreter='latex')
axis tight
grid on
subplot(2,1,2)
hold on;
plot(t_KDPC(1:n_ok),(uKDPC.') + u_mean,'LineWidth',2);
ylabel('$u$',Interpreter='latex')
xlabel('$t$[s]',Interpreter='latex')
legend('T','\omega_x','\omega_y','\omega_z','Location','southeast');
axis tight
grid on;
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
savefig('figures/KDPC_quad.fig')
 
figure; hold on; grid on; axis equal
plot3(X_log(:,1),X_log(:,2),X_log(:,3),'LineWidth',2);
plot3(P_ref(:,1),P_ref(:,2),P_ref(:,3),'--','LineWidth',1.5);
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
legend('flown','reference'); view(3);
 
%% DIAGNOSTIC: did the model predict what actually happened?
lbl = {'dx','dy','dz','yaw'};
fprintf('\none-step prediction check (predicted y(k+1) vs measured)\n');
fprintf('%-6s %12s %12s %10s %9s\n','chan','rms actual','rms error','err/act','slope');
for c = 1:nyw
    A = yact1(c,:);  P = yhat1(c,:);
    sl = (P(:)\A(:));                     % actual = slope * predicted
    fprintf('%-6s %12.5f %12.5f %10.3f %9.3f\n', lbl{c}, ...
            sqrt(mean(A.^2)), sqrt(mean((P-A).^2)), ...
            sqrt(mean((P-A).^2))/max(1e-12,sqrt(mean(A.^2))), sl);
end
fprintf('slope > 1 means the plant moves MORE than the model predicted,\n');
fprintf('which is what makes a predictive controller over-command and overshoot.\n');
fprintf('input on the box: T %.1f%%  wx %.1f%%  wy %.1f%%  wz %.1f%%\n', ...
        100*mean(abs(usat),2));
 
 
save('data/KDPC', 't_KDPC','yKDPC','uKDPC','X_log','P_ref','use_phi','yhat1','yact1','usat')
 
%% helpers
function y = measure(x, yaw, y_mean)
% Mean-removed outputs y = [p; yaw], matching what Generate_data.m writes. Yaw is
% passed in already unwrapped. No reference is subtracted -- it lives in the cost,
% and ti_window differences the positions.
y = [x(1:3); yaw] - y_mean(:);
end
 
function r = ref_meas(t, r_pos, r_acc, g) %#ok<INUSD>
% Reference outputs in PHYSICAL units, y = [p; yaw]. Yaw is held at zero.
r = [r_pos(t); 0];
end
 
function u = ref_u(t, r_acc, r_jrk, g)
% Reference input in PHYSICAL units, used ONLY to seed the buffers at t < 0.
% The body rate uses the small-angle jerk relation, which is exact only near
% hover -- fine for initialisation, which is why it is kept out of the cost.
a  = r_acc(t);  j = r_jrk(t);
zb = [a(1); a(2); a(3) + g];
u  = [norm(zb); -j(2)/g; j(1)/g; 0];
end
 
function x = ti_window(Ubuf, Ybuf, idx_w)
% x_ini = [u(k-Tini+1..k-1); p(k-Tini+1..k-1) - p(k); o(k-Tini+1..k)]
%   Ubuf : (Tini-1) x nu     Ybuf : Tini x 6 (all measured channels)
% idx_w selects which measured channels reach the window; here that is all of
% them, since Generate_data logs only [x y z yaw].
% The j = 0 position entry is identically zero and is dropped; keeping it would
% make the regressor rank deficient.
Yw = Ybuf(:, idx_w);
ui = reshape(Ubuf.', [], 1);
pk = Yw(end, 1:3);
dp = reshape((Yw(1:end-1, 1:3) - pk).', [], 1);
oo = reshape(Yw(:, 4:end).', [], 1);
x  = [ui; dp; oo];
end
 
function n = nphi(use_phi)
if use_phi, n = 14; else, n = 0; end
end
 
function p = phi_nl(Ubuf, Ybuf, Ts, use_phi)
% Nonlinear observables, 14 of them.
%
% A linear predictor can already form any linear combination of the window, so an
% observable earns its dimension only if it is a PRODUCT the window cannot
% produce. From
%       vdot  = alpha_T*b3(q) - g*e3,      b3dot = (R*w) x b3
% the products that matter pair the commanded input with the current attitude.
%
%   w x a       [3] thrust axis rotating under the commanded body rate: the
%                   attitude propagation, which dominates the multi-step residual
%   alpha_T*a   [3] the bilinear part of alpha_T*b3 surviving linearisation
%   a_x^2, a_y^2, a_x*a_y
%               [3] tilt geometry, second order
%   yaw*a_x, yaw*a_y
%               [2] body tilt rotated into the world frame, first order in yaw
%   w x v       [3] rate against velocity
%
% Attitude is not measured (y = [x y z yaw]), so the acceleration stands in for
% the tilt: a ~ alpha_T*b3 - g*e3, recovered as a second difference of position.
% That is exact here because the simulated positions are exact, but the noise
% gain is sqrt(6)/Ts^2 ~ 980, which on the real flight data in data/real_*.mat
% gives SNR ~ 1.1. Log the flight controller's attitude if this ever goes to
% hardware, and build a from that instead.
%
% Takes the raw buffers rather than x_ini so it can reach three position lags.
if ~use_phi, p = zeros(0,1); return; end
u1 = Ubuf(end,:).';                              % u(k-1) = [alpha_T; wx; wy; wz]
aT = u1(1);  w = u1(2:4);
yw = Ybuf(end,4);                                % yaw at k, mean-removed
v  = (Ybuf(end,1:3) - Ybuf(end-1,1:3)).'/Ts;     % FIRST  difference, gain sqrt(2)/Ts
a  = (Ybuf(end,1:3) - 2*Ybuf(end-1,1:3) + Ybuf(end-2,1:3)).'/Ts^2;   % SECOND, gain sqrt(6)/Ts^2
 
% every term is a product of window entries that are themselves zero at
% x_ini = 0, so phi_nl(0) = 0 exactly (Assumption 2.4) with no offset needed.
p = [cross(w,a); aT*a; a(1)^2; a(2)^2; a(1)*a(2); yw*a(1); yw*a(2); cross(w,v)];
end
 
function [Zini, UF, YF] = koopman_windows(u, y, lens, Tini, N, use_phi, Ts, idx_w)
% Zini(:,c) = phi(x_ini(k)),  UF(:,c) = u(k..k+N-1),
% YF(:,c)   = [p(k+j)-p(k); o(k+j)] for j = 1..N, over the idx_w channels only.
% Windows never cross a run boundary: runs are separate flights.
nu = size(u,2);  nyw = numel(idx_w);  now = nyw - 3;  lens = double(lens(:));
n_ini = (Tini-1)*nu + (Tini-1)*3 + Tini*now;
L = n_ini + nphi(use_phi);
 
ncols = 0;
for rr = 1:numel(lens), ncols = ncols + max(0, lens(rr)-Tini-N); end
assert(ncols > 0, 'every run is shorter than Tini+N = %d samples', Tini+N);
 
Zini = zeros(L, ncols);  UF = zeros(N*nu, ncols);  YF = zeros(N*nyw, ncols);
 
start = 0;  col = 0;
for rr = 1:numel(lens)
    len = lens(rr);  uu = u(start+(1:len),:);  yy = y(start+(1:len),:);
    for i = 1:(len-Tini-N)
        col = col + 1;
        k   = i + Tini;                              % current time, run-local
        Ub  = uu(k-Tini+1 : k-1, :);   Yb = yy(k-Tini+1 : k, :);
        xw  = ti_window(Ub, Yb, idx_w);
        Zini(:,col) = [xw; phi_nl(Ub, Yb, Ts, use_phi)];
        UF(:,col)   = reshape(uu(k : k+N-1, :).', [], 1);
        yw          = yy(:, idx_w);
        blk         = [yw(k+1:k+N, 1:3) - yw(k,1:3),  yw(k+1:k+N, 4:end)];
        YF(:,col)   = reshape(blk.', [], 1);
    end
    start = start + len;
end
end