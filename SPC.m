%% SPC with an inner-loop attitude rate PID on simulated or on real flight data
clc; clear;
% close all;

%% load params and data, set weights and time
% set data type
% 'sim'  -> simulated data from Generate_data.m
% 'real' -> real data from Load_flight_data.m
data = 'sim';

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
    Tini  = 6;

    % load sim data
    load('data/u_data.mat')
    load('data/y_data.mat')
    load('data/u_data_test.mat')
    load('data/y_data_test.mat')
    load('data/run_lengths.mat')
    load('data/scalers.mat')
    tag = '';
case 'real'
    % cost weights, physical units
    % Note: 'real' needs higher RU_wz (due to yaw underexcited) cost and Tini
    Qy_phys = [10; 10; 10; 1];           % x y z yaw
    Ru_phys = [0.1; 2.0; 2.0; 200];
    N    = 20;
    Tini = 15;

    % load real data
    load('data/real_u_data.mat')
    load('data/real_y_data.mat')
    load('data/real_u_data_test.mat')
    load('data/real_y_data_test.mat')
    load('data/real_run_lengths.mat')
    load('data/real_scalers.mat')
    tag = '_real';
otherwise
    error('data must be ''sim'' or ''real'', got ''%s''', data);
end
assert(strcmp(source, data), ...
       'scalers were saved with source = ''%s'' but data = ''%s''', source, data);

% Generate_data now logs y = [x y z yaw b3x b3y b3z]. The b3 channels exist for
% the KDPC lifting; SPC uses only the four tracked outputs.
y_data      = y_data(:,      1:4);
y_data_test = y_data_test(:, 1:4);
y_mean      = y_mean(1:4);

nu = size(u_data,2);  ny = size(y_data,2);
n_ini = (Tini-1)*nu + Tini*ny;
nU    = (Tini-1)*nu;
k_sim = round(t_end/Ts);

%% set ref
% reference, figure 8 increasing in speed
% Note: Increase in frequency requires more tilt thus more nonlinearities.
% This results in less acurate tracking. Increase w_sp to increase velocity
A_sp = 1.2;  B_sp = 0.8;  C_sp = 0.4;  w_sp = 0.4;
r_pos = @(t) p_hover(:) + [A_sp*sin(w_sp*t); B_sp*sin(2*w_sp*t); C_sp*sin(w_sp*t)];

% derivatives, used to give drone correct initial condition
r_vel = @(t) [ A_sp*w_sp  *cos(w_sp*t);  2*B_sp*w_sp  *cos(2*w_sp*t);  C_sp*w_sp  *cos(w_sp*t)];
r_acc = @(t) [-A_sp*w_sp^2*sin(w_sp*t); -4*B_sp*w_sp^2*sin(2*w_sp*t); -C_sp*w_sp^2*sin(w_sp*t)];
r_jrk = @(t) [-A_sp*w_sp^3*cos(w_sp*t); -8*B_sp*w_sp^3*cos(2*w_sp*t); -C_sp*w_sp^3*cos(w_sp*t)];

%% load X,Y, learn the theta state model and calc P1, P2 and Gamma
[Uini,Yini,UF,YF]     = hankel_windows(u_data,      y_data,      run_lengths,      Tini, N);
[Ui_t,Yi_t,UF_t,YF_t] = hankel_windows(u_data_test, y_data_test, run_lengths_test, Tini, N);

X   = [Uini; Yini; UF];
Yt  = YF;
X_t = [Ui_t; Yi_t; UF_t];

% learn the theta state model
Theta_SPC = Yt*pinv(X);
save(['data/SPCTheta' tag], 'Theta_SPC')

% end state definition
P1    = Theta_SPC(:, 1:nU);
P2    = Theta_SPC(:, nU + (1:Tini*ny));
Gamma = Theta_SPC(:, n_ini+1:end);

%% Simulate the system
Q = diag(Qy_phys(:));
R = diag(Ru_phys(:));

Psi   = kron(eye(N), R);
Omega = kron(eye(N), Q);

% Terminal cost based on DARE
Az = zeros(n_ini);  Bz = zeros(n_ini,nu);
Az(1:nU-nu, nu+1:nU)         = eye(nU-nu);                 % shift past inputs
Bz(nU-nu+1:nU, :)            = eye(nu);                    % append u(k)
Az(nU+1:end-ny, nU+ny+1:end) = eye(Tini*ny-ny);            % shift past outputs
Az(end-ny+1:end, :)          = Theta_SPC(1:ny, 1:n_ini);   % append y(k+1)
Bz(end-ny+1:end, :)          = Theta_SPC(1:ny, n_ini+(1:nu));
Cz = [zeros(ny, n_ini-ny), eye(ny)];

P_term = idare(Az, Bz, Cz.'*Q*Cz, R);
assert(~isempty(P_term), 'idare failed -- (Az,Bz) not stabilisable');
P_term = (P_term + P_term.')/2;
AN = Az^N;  Sn = zeros(n_ini, N*nu);
for j = 0:N-1, Sn(:, j*nu+(1:nu)) = Az^(N-1-j)*Bz; end

uVar = sdpvar(N*nu,1);
yVar = sdpvar(N*ny,1);
yini = sdpvar(Tini*ny,1);
uini = sdpvar(nU,1);

zN        = AN*[uini; yini] + Sn*uVar;
objective = yVar'*Omega*yVar + uVar'*Psi*uVar; %+ zN'*P_term*zN;

constraints = [yVar == P1*uini + P2*yini + Gamma*uVar];

% input box: the actuator 
u_lo = [lim.T_spec(1); -lim.w_cmd*ones(3,1)] - u_mean(:);
u_hi = [lim.T_spec(2);  lim.w_cmd*ones(3,1)] - u_mean(:);
constraints = [constraints, repmat(u_lo,N,1) <= uVar <= repmat(u_hi,N,1)];

Parameters = {uini,yini};
Outputs    = {uVar,yVar};

options = sdpsettings('solver','QUADPROG','verbose',0,'debug',0);
controller = optimizer(constraints, objective, options, Parameters, Outputs);

%% initialize
% start from rest 
% x  = [r_pos(0); zeros(3,1); 1;0;0;0; zeros(3,1)]

% start on the trajectory
a0 = r_acc(0);  j0 = r_jrk(0);
zb = [a0(1); a0(2); a0(3) + prm.g];  zb = zb/norm(zb);   % body z axis
sq = sqrt(2*(1 + zb(3)));                                % q rotating [0;0;1] -> zb
q0 = [sq/2; cross([0;0;1], zb)/sq];  q0 = q0/norm(q0);   % zero yaw
om0 = [-j0(2); j0(1); 0]/prm.g;                          % small-angle jerk -> rate
x  = [r_pos(0); r_vel(0); q0; om0];

st = [];
yaw_prev = 0;

Y_ini = repmat(measure(x, yaw_prev, r_pos(0), y_mean), Tini, 1);
U_ini = zeros(nU,1);                      
assert(isequal(size(Y_ini),[Tini*ny 1]) && isequal(size(U_ini),[nU 1]), ...
       'buffer shape wrong: Y_ini %s, U_ini %s', mat2str(size(Y_ini)), mat2str(size(U_ini)));

uSPC  = zeros(nu, k_sim);
ySPC  = zeros(ny, k_sim);
th    = zeros(k_sim,1);
X_log = zeros(k_sim+1,13);  X_log(1,:) = x.';
P_ref = zeros(k_sim+1,3);   P_ref(1,:) = r_pos(0).';
t_SPC = (0:k_sim)*Ts;
n_ok  = k_sim;

%% main loop
for i = 1:k_sim
    fprintf("running step %d of %d \n", i , k_sim)
    
    OUT = controller({U_ini,Y_ini});

    Uk = OUT{1};

    u_mpc     = Uk(1:nu);
    uSPC(:,i) = u_mpc;
    th(i)     = toc;

    % output update
    v_cmd = u_mpc + u_mean(:);                      % [T_spec; om_ref]
    [x, st] = quad_plant(x, v_cmd, (i-1)*Ts, prm, lim, inner, Ts, Ts_inner, st);

    q  = x(7:10);
    yr = atan2(2*(q(1)*q(4)+q(2)*q(3)), 1-2*(q(3)^2+q(4)^2));
    yaw_prev = yaw_prev + atan2(sin(yr-yaw_prev), cos(yr-yaw_prev));

    p_ref_next = r_pos(i*Ts);
    y_next     = measure(x, yaw_prev, p_ref_next, y_mean);

    Y_ini = [Y_ini(ny+1:end); y_next];
    U_ini = [U_ini(nu+1:end); u_mpc];

    ySPC(:,i)    = y_next;
    X_log(i+1,:) = x.';
    P_ref(i+1,:) = p_ref_next.';

    % sim stops when the drone diverges to much
    if ~all(isfinite(x)) || norm(x(1:3) - p_ref_next) > 50
        warning('diverged at t = %.2f s', i*Ts);
        n_ok = i;  break
    end
end

uSPC = uSPC(:,1:n_ok);  ySPC = ySPC(:,1:n_ok);  th = th(1:n_ok);
X_log = X_log(1:n_ok+1,:);  P_ref = P_ref(1:n_ok+1,:);  t_SPC = t_SPC(1:n_ok+1);

%% New plot
e = X_log(:,1:3) - P_ref;
fprintf('\ndata = %s | Tini = %d | N = %d | Ts = %.3f s\n', data, Tini, N, Ts);
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
plot(t_SPC,P_ref,'LineWidth',2,'LineStyle','--');
plot(t_SPC,X_log(:,1:3),'LineWidth',2);
legend('x_{ref}','y_{ref}','z_{ref}','x','y','z','Location','northeast');
ylabel('$p$ [m]',Interpreter='latex')
axis tight
grid on
subplot(2,1,2)
hold on;
plot(t_SPC(1:n_ok),(uSPC.') + u_mean,'LineWidth',2);
ylabel('$u$',Interpreter='latex')
xlabel('$t$[s]',Interpreter='latex')
legend('T','\omega_x','\omega_y','\omega_z','Location','southeast');
axis tight
grid on;
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
savefig(['figures/SPC_quad' tag '.fig'])

figure; hold on; grid on; axis equal
plot3(X_log(:,1),X_log(:,2),X_log(:,3),'LineWidth',2);
plot3(P_ref(:,1),P_ref(:,2),P_ref(:,3),'--','LineWidth',1.5);
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
legend('flown','reference'); view(3);

save(['data/SPC' tag], 't_SPC','ySPC','uSPC','X_log','P_ref')

%% helpers
function y = measure(x, yaw, p_sp, y_mean)
y_ref = y_mean(:);  y_ref(1:3) = p_sp(:);
y     = [x(1:3); yaw] - y_ref;
end

function [Uini, Yini, UF, YF] = hankel_windows(u, y, lens, Tini, N)
% u_ini = [u(k-Tini+1) ... u(k-1)]   excludes u(k), which is a variable
% y_ini = [y(k-Tini+1) ... y(k)  ]   includes y(k), which is measured
% u_f   = [u(k)        ... u(k+N-1)]
% y_f   = [y(k+1)      ... y(k+N)  ]
% Windows never cross a run boundary: runs are separate flights.
nu = size(u,2);  ny = size(y,2);  lens = double(lens(:));

ncols = 0;
for rr = 1:numel(lens), ncols = ncols + max(0, lens(rr)-Tini-N); end
assert(ncols > 0, 'every run is shorter than Tini+N = %d samples', Tini+N);

Uini = zeros((Tini-1)*nu, ncols);  Yini = zeros(Tini*ny, ncols);
UF   = zeros(N*nu,        ncols);  YF   = zeros(N*ny,    ncols);

start = 0;  col = 0;
for rr = 1:numel(lens)
    L = lens(rr);  uu = u(start+(1:L),:);  yy = y(start+(1:L),:);
    for i = 1:(L-Tini-N)
        col = col + 1;
        Uini(:,col) = reshape(uu(i+1      : i+Tini-1,   :).', [], 1);
        Yini(:,col) = reshape(yy(i+1      : i+Tini,     :).', [], 1);
        UF(:,col)   = reshape(uu(i+Tini   : i+Tini+N-1, :).', [], 1);
        YF(:,col)   = reshape(yy(i+Tini+1 : i+Tini+N,   :).', [], 1);
    end
    start = start + L;
end
end
