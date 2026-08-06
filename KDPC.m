%% KDPC with an inner-loop attitude rate PID on simulated or on real flight data
clc; clear;
% close all;

%% load params and data, set weights and time
% set data type
% 'sim'  -> simulated data from Generate_data.m
% 'real' -> real data from Load_flight_data.m
data = 'sim';

% THE SWITCH. false -> phi = identity, i.e. plain subspace predictive control in
% this same formulation. true -> append the physics observables of phi_nl below.
% Everything else is untouched, so any difference is due to the lifting alone.
use_phi = true;

% load params, set sim rate and time
[prm, lim, inner] = quad_params();
Ts_inner = 0.002;                         % inner rate loop, 500 Hz
t_end    = 40;

switch data
case 'sim'
    % cost weights, physical units
    Qy_phys = [10; 10; 10; 1];           % x y z yaw
    Ru_phys = [0.1; 2.0; 2.0; 2.0];      % T wx wy wz
    N    = 20;
    Tini = 6;

    load('data/u_data.mat')
    load('data/y_data.mat')
    load('data/run_lengths.mat')
    load('data/scalers.mat')
    tag = '';
case 'real'
    Qy_phys = [10; 10; 10; 1];
    Ru_phys = [0.1; 2.0; 2.0; 200];
    N    = 20;
    Tini = 15;

    load('data/real_u_data.mat')
    load('data/real_y_data.mat')
    load('data/real_run_lengths.mat')
    load('data/real_scalers.mat')
    u_data = real_u_data;  y_data = real_y_data;  run_lengths = real_run_lengths;
    tag = '_real';
otherwise
    error('data must be ''sim'' or ''real'', got ''%s''', data);
end
assert(strcmp(source, data), ...
       'scalers were saved with source = ''%s'' but data = ''%s''', source, data);

w_term = 10;                              % weight multiplier on the last output block

nu = size(u_data,2);  ny = size(y_data,2);
no = ny - 3;                              % non-position outputs (yaw, ...)
nU = (Tini-1)*nu;                         % past inputs   u(k-Tini+1..k-1)
nP = (Tini-1)*3;                          % past position differences
nO = Tini*no;                             % past non-position outputs
n_ini = nU + nP + nO;
L     = n_ini + nphi(use_phi);
k_sim = round(t_end/Ts);

fprintf('KDPC | data %s | use_phi %d | Tini %d | N %d | n_ini %d | L %d\n', ...
        data, use_phi, Tini, N, n_ini, L);

%% set ref
% reference, figure 8 increasing in speed
% Note: Increase in frequency requires more tilt thus more nonlinearities.
A_sp = 1.2;  B_sp = 0.8;  C_sp = 0.4;  w_sp = 1.0;
r_pos = @(t) p_hover(:) + [A_sp*sin(w_sp*t); B_sp*sin(2*w_sp*t); C_sp*sin(w_sp*t)];

% derivatives, used to give drone correct initial condition
r_vel = @(t) [ A_sp*w_sp  *cos(w_sp*t);  2*B_sp*w_sp  *cos(2*w_sp*t);  C_sp*w_sp  *cos(w_sp*t)];
r_acc = @(t) [-A_sp*w_sp^2*sin(w_sp*t); -4*B_sp*w_sp^2*sin(2*w_sp*t); -C_sp*w_sp^2*sin(w_sp*t)];
r_jrk = @(t) [-A_sp*w_sp^3*cos(w_sp*t); -8*B_sp*w_sp^3*cos(2*w_sp*t); -C_sp*w_sp^3*cos(w_sp*t)];

%% load X,Y, learn the theta state model and calc Pz and Gamma
% One departure from SPC.m, and it is required rather than cosmetic.
%
% SPC.m tracks by feeding the predictor y = p - p_ref and driving the prediction
% to zero. That is valid only if the fitted model is exactly invariant to a
% constant position shift. It is not -- measured on this data, a 1 m shift
% produces up to 0.12 m of spurious bias at j = N -- and SPC.m survives only
% because it operates within ~2 cm of the origin. It also breaks outright once
% phi is nonlinear, since phi would then be evaluated on error coordinates while
% it was identified on absolute ones.
%
% So the window here holds position DIFFERENCES p(k-j) - p(k) and the predictor
% returns DISPLACEMENTS p(k+j) - p(k). A constant shift then changes no regressor
% entry at all, exactly, phi's arguments are invariant, and the reference goes in
% the cost where it can be previewed over the horizon.
[Zini, UF, YF] = koopman_windows(u_data, y_data, run_lengths, Tini, N, use_phi, Ts);

X  = [Zini; UF];
Yt = YF;

% learn the theta state model
Theta_KDPC = Yt*pinv(X);
save(['data/KDPCTheta' tag], 'Theta_KDPC', 'Tini', 'N', 'L', 'use_phi')

% end state definition
Pz    = Theta_KDPC(:, 1:L);
Gamma = Theta_KDPC(:, L+1:end);

%% Simulate the system
Q = diag(Qy_phys(:));
R = diag(Ru_phys(:));

Psi   = kron(eye(N), R);
Omega = blkdiag(kron(eye(N-1), Q), w_term*Q);

% No terminal cost from eq. (13a) here. Assumption 4.1 places P and X_T around
% the ORIGIN for a regulation problem; with a moving reference that construction
% does not carry over. w_term is an ordinary heavier terminal weight and claims
% none of the guarantees of Theorem 4.2 or 4.4.

uVar = sdpvar(N*nu,1);
yVar = sdpvar(N*ny,1);
zini = sdpvar(L,1);                       % parameter: phi(x_ini(k))
rVar = sdpvar(N*ny,1);                    % parameter: reference over the horizon
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
    Ybuf(b,:) = ref_out(-(Tini-b)*Ts, r_pos, r_acc, prm.g, ny).' - y_mean(:).';
end
for b = 1:Tini-1
    Ubuf(b,:) = ref_u(-(Tini-b)*Ts, r_acc, r_jrk, prm.g).' - u_mean(:).';
end

uKDPC = zeros(nu, k_sim);
yKDPC = zeros(ny, k_sim);
th    = zeros(k_sim,1);
X_log = zeros(k_sim+1,13);  X_log(1,:) = x.';
P_ref = zeros(k_sim+1,3);   P_ref(1,:) = r_pos(0).';
t_KDPC = (0:k_sim)*Ts;
n_ok  = k_sim;

%% main loop
for i = 1:k_sim
    fprintf("running step %d of %d \n", i , k_sim)

    tic;
    t = (i-1)*Ts;

    x_ini = ti_window(Ubuf, Ybuf, Tini, nu, ny);
    z_k   = [x_ini; phi_nl(x_ini, Tini, nu, no, Ts, use_phi)];

    % reference in the coordinates the predictor returns: displacement from the
    % current position for the position channels, mean-removed absolute for the
    % rest. p(k) and y_mean cancel out of the position part entirely.
    p_now = x(1:3);
    rvec  = zeros(N*ny,1);
    for j = 1:N
        rj = ref_out(t + j*Ts, r_pos, r_acc, prm.g, ny);
        rvec((j-1)*ny + (1:3))  = rj(1:3) - p_now;
        rvec((j-1)*ny + (4:ny)) = rj(4:ny) - y_mean(4:ny).';
    end

    OUT = controller({z_k, rvec});

    Uk = OUT{1};

    u_mpc     = Uk(1:nu);
    uKDPC(:,i) = u_mpc;
    th(i)     = toc;

    % output update
    v_cmd = u_mpc + u_mean(:);                      % [T_spec; om_ref]
    [x, st] = quad_plant(x, v_cmd, t, prm, lim, inner, Ts, Ts_inner, st);

    q  = x(7:10);
    yr = atan2(2*(q(1)*q(4)+q(2)*q(3)), 1-2*(q(3)^2+q(4)^2));
    yaw_prev = yaw_prev + atan2(sin(yr-yaw_prev), cos(yr-yaw_prev));

    y_next = measure(x, yaw_prev, y_mean, ny);
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

uKDPC = uKDPC(:,1:n_ok);  yKDPC = yKDPC(:,1:n_ok);  th = th(1:n_ok);
X_log = X_log(1:n_ok+1,:);  P_ref = P_ref(1:n_ok+1,:);  t_KDPC = t_KDPC(1:n_ok+1);

%% New plot
e = X_log(:,1:3) - P_ref;
fprintf('\ndata = %s | use_phi = %d | Tini = %d | N = %d | Ts = %.3f s\n', data, use_phi, Tini, N, Ts);
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
savefig(['figures/KDPC_quad' tag '.fig'])

figure; hold on; grid on; axis equal
plot3(X_log(:,1),X_log(:,2),X_log(:,3),'LineWidth',2);
plot3(P_ref(:,1),P_ref(:,2),P_ref(:,3),'--','LineWidth',1.5);
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
legend('flown','reference'); view(3);

save(['data/KDPC' tag], 't_KDPC','yKDPC','uKDPC','X_log','P_ref','use_phi')

%% helpers
function y = measure(x, yaw, y_mean, ny)
% Absolute mean-removed outputs [p; yaw; b3](1:ny). No reference is subtracted --
% the reference lives in the cost, and ti_window differences the positions.
q  = x(7:10);
b3 = [2*(q(2)*q(4) + q(1)*q(3));
      2*(q(3)*q(4) - q(1)*q(2));
      1 - 2*(q(2)^2 + q(3)^2)];
y = [x(1:3); yaw; b3];
y = y(1:ny) - y_mean(:);
end

function r = ref_out(t, r_pos, r_acc, g, ny)
% Reference outputs in PHYSICAL units. The b3 part is only used when ny > 4.
a  = r_acc(t);
zb = [a(1); a(2); a(3) + g];  zb = zb/norm(zb);
r  = [r_pos(t); 0; zb];
r  = r(1:ny);
end

function u = ref_u(t, r_acc, r_jrk, g)
% Reference input in PHYSICAL units, used ONLY to seed the buffers at t < 0.
% The body rate uses the small-angle jerk relation, which is exact only near
% hover -- fine for initialisation, which is why it is kept out of the cost.
a  = r_acc(t);  j = r_jrk(t);
zb = [a(1); a(2); a(3) + g];
u  = [norm(zb); -j(2)/g; j(1)/g; 0];
end

function x = ti_window(Ubuf, Ybuf, Tini, nu, ny)
% x_ini = [u(k-Tini+1..k-1); p(k-Tini+1..k-1) - p(k); o(k-Tini+1..k)]
%   Ubuf : (Tini-1) x nu     Ybuf : Tini x ny
% The j = 0 position entry is identically zero and is dropped; keeping it would
% make the regressor rank deficient.
ui = reshape(Ubuf.', [], 1);
pk = Ybuf(end, 1:3);
dp = reshape((Ybuf(1:end-1, 1:3) - pk).', [], 1);
oo = reshape(Ybuf(:, 4:ny).', [], 1);
x  = [ui; dp; oo];
end

function n = nphi(use_phi)
if use_phi, n = 14; else, n = 0; end
end

function [u1, v, a, o] = ti_parts(x, Tini, nu, no, Ts)
% Physical quantities recoverable from the invariant window. Each is zero at
% x_ini = 0, so every product built from them vanishes at the origin and
% Assumption 2.4 holds by construction, with no offset subtracted anywhere.
nU = (Tini-1)*nu;  nP = (Tini-1)*3;
u1 = x(nU-nu+1 : nU);                         % u(k-1) = [alpha_T; wx; wy; wz]
dp = reshape(x(nU+1 : nU+nP), 3, Tini-1).';   % rows p(k-j)-p(k), j = Tini-1..1
v  = -dp(end,:).'/Ts;                         % (p(k) - p(k-1))/Ts
a  = (dp(end-1,:).' - 2*dp(end,:).')/Ts^2;    % (p(k) - 2p(k-1) + p(k-2))/Ts^2
o  = x(nU+nP + (Tini-1)*no + (1:no));         % non-position outputs at k
end

function p = phi_nl(x, Tini, nu, no, Ts, use_phi)
% Nonlinear observables, 14 of them.
%
% A linear predictor can already form any linear combination of the window, so an
% observable is only worth its dimension if it is a PRODUCT the window cannot
% produce. From
%       vdot  = alpha_T*b3(q) - g*e3,      b3dot = (R*w) x b3
% the products that matter pair the commanded input with the current attitude.
% Attitude is not measured here, but near hover the acceleration carries it,
% a ~ alpha_T*b3 - g*e3, so a stands in for the tilt.
%
%   w x a       [3] thrust axis rotating under the commanded body rate: the
%                   attitude propagation, which dominates the multi-step residual
%   alpha_T*a   [3] the bilinear part of alpha_T*b3 surviving linearisation
%   a_x^2, a_y^2, a_x*a_y
%               [3] tilt geometry: |b3| = 1 makes the vertical component second
%                   order in the horizontal ones
%   yaw*a_x, yaw*a_y
%               [2] body tilt rotated into the world frame, first order in yaw
%   w x v       [3] rate against velocity
if ~use_phi, p = zeros(0,1); return; end
[u1, v, a, o] = ti_parts(x, Tini, nu, no, Ts);
w  = u1(2:4);  aT = u1(1);
p  = [cross(w,a); aT*a; a(1)^2; a(2)^2; a(1)*a(2); o(1)*a(1); o(1)*a(2); cross(w,v)];
end

function [Zini, UF, YF] = koopman_windows(u, y, lens, Tini, N, use_phi, Ts)
% Zini(:,c) = phi(x_ini(k)),  UF(:,c) = u(k..k+N-1),
% YF(:,c)   = [p(k+j)-p(k); o(k+j)] for j = 1..N.
% Windows never cross a run boundary: runs are separate flights.
nu = size(u,2);  ny = size(y,2);  no = ny-3;  lens = double(lens(:));
n_ini = (Tini-1)*nu + (Tini-1)*3 + Tini*no;
L = n_ini + nphi(use_phi);

ncols = 0;
for rr = 1:numel(lens), ncols = ncols + max(0, lens(rr)-Tini-N); end
assert(ncols > 0, 'every run is shorter than Tini+N = %d samples', Tini+N);

Zini = zeros(L, ncols);  UF = zeros(N*nu, ncols);  YF = zeros(N*ny, ncols);

start = 0;  col = 0;
for rr = 1:numel(lens)
    len = lens(rr);  uu = u(start+(1:len),:);  yy = y(start+(1:len),:);
    for i = 1:(len-Tini-N)
        col = col + 1;
        k   = i + Tini;                              % current time, run-local
        xw  = ti_window(uu(k-Tini+1:k-1,:), yy(k-Tini+1:k,:), Tini, nu, ny);
        Zini(:,col) = [xw; phi_nl(xw, Tini, nu, no, Ts, use_phi)];
        UF(:,col)   = reshape(uu(k : k+N-1, :).', [], 1);
        blk         = [yy(k+1:k+N, 1:3) - yy(k,1:3),  yy(k+1:k+N, 4:ny)];
        YF(:,col)   = reshape(blk.', [], 1);
    end
    start = start + len;
end
end
