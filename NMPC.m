%% Nonlinear MPC baseline (acados)
% Same role as KDPC_RSRFG/Pendulum/NMPC.m, but solved with acados rather than
% YALMIP+fmincon. A 20-stage, 13-state multiple-shooting NLP has a block-banded
% KKT system that HPIPM factorises stage by stage; fmincon treats it as one
% dense block and reports RCOND ~ 1e-21 on this problem. The pendulum gets away
% with fmincon at 3 variables per stage; this has 17.
%
% This is the ceiling for the comparison: it uses the TRUE model, so
%   RMSE_NMPC <= RMSE_KDPC <= RMSE_SPC
% Run it before any Koopman work to see how much headroom a better model buys.
%
% Requires acados on the path (run acados_startup.m / env.sh first).
clc; clear all;
% close all;

[prm, lim, inner] = quad_params();
Ts       = 0.05;                          % 20 Hz outer loop, matching SPC
Ts_inner = 0.002;                         % 500 Hz rate loop
g        = prm.g;
t_end    = 40;

N     = 20;                               % same horizon as SPC
k_sim = round(t_end/Ts);

% Weights. Position and yaw match SPC.m; velocity and body rate get small
% damping terms, which SPC gets implicitly through its terminal DARE cost on
% the whole delay embedding.
Wp = [10 10 10];                          % position
Wv = [0.1 0.1 0.1];                       % velocity damping
Ww = [0.05 0.05 0.05];                    % body-rate damping
WT = 0.1;                                 % thrust deviation from hover
Wt = [1e3 1e3 1e4];                       % torque effort (tau ~ 1e-3 N*m)

p_hover = [0; 0; 1.4];

% Reference, matching SPC.m. Keep w_sp in step with SPC.m or the two baselines
% are flying different trajectories and the comparison is meaningless.
% NOTE p_hover: SPC.m takes it from the scalers, so with data = 'real' it sits
% at the flight data's mean position, ~7 cm from [0;0;1.4]. Compare like for
% like, or set the value below to match.
A_sp = 1.2;  B_sp = 0.8;  C_sp = 0.4;  w_sp = 0.4;
r_pos = @(t) p_hover + [A_sp*sin(w_sp*t); B_sp*sin(2*w_sp*t); C_sp*sin(w_sp*t)];
r_vel = @(t) [ A_sp*w_sp  *cos(w_sp*t);  2*B_sp*w_sp  *cos(2*w_sp*t);  C_sp*w_sp  *cos(w_sp*t)];
r_acc = @(t) [-A_sp*w_sp^2*sin(w_sp*t); -4*B_sp*w_sp^2*sin(2*w_sp*t); -C_sp*w_sp^2*sin(w_sp*t)];
r_jrk = @(t) [-A_sp*w_sp^3*cos(w_sp*t); -8*B_sp*w_sp^3*cos(2*w_sp*t); -C_sp*w_sp^3*cos(w_sp*t)];

% Start ON the trajectory, as SPC.m does: velocity from the reference, attitude
% from the required acceleration, body rate from the jerk (differential
% flatness). Otherwise NMPC spends the first second catching up and the RMSE
% is dominated by a start-up transient SPC does not have.
a0 = r_acc(0);  j0 = r_jrk(0);
zb = [a0(1); a0(2); a0(3) + g];  zb = zb/norm(zb);       % body z axis
sq = sqrt(2*(1 + zb(3)));                                % q rotating [0;0;1] -> zb
q0 = [sq/2; cross([0;0;1], zb)/sq];  q0 = q0/norm(q0);   % zero yaw
om0 = [-j0(2); j0(1); 0]/g;                              % small-angle jerk -> rate

% x0_traj = [r_pos(0); zeros(3,1); 1;0;0;0; zeros(3,1)]; % uncomment: from rest
x0_traj = [r_pos(0); r_vel(0); q0; om0];

assert(~isempty(which('AcadosOcp')), 'acados not on the path -- run acados_startup.m');

%% Setup the solver
% state x = [p(3); v(3); q(4, w first); om(3)]   (13)
% input u = [T(1); tau(3)]                        (4)   -- the WRENCH
%
% NMPC plans over the true rigid body with torque as the input, then hands the
% inner loop only [T/m; om_des], where om_des is taken from the optimised state
% at node 1. So the plant sees exactly the signal SPC commands and nothing
% below the outer loop differs between the two.
import casadi.*
nx = 13;  nu = 4;

p = SX.sym('p',3);  v = SX.sym('v',3);  q = SX.sym('q',4);  om = SX.sym('om',3);
Th = SX.sym('Th');  tau = SX.sym('tau',3);
x = [p; v; q; om];
u = [Th; tau];

q0=q(1); q1=q(2); q2=q(3); q3=q(4);
Rwb=[1-2*(q2^2+q3^2), 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2);
     2*(q1*q2+q0*q3), 1-2*(q1^2+q3^2),   2*(q2*q3-q0*q1);
     2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1),   1-2*(q1^2+q2^2)];
Omq=[0,-om(1),-om(2),-om(3); om(1),0,om(3),-om(2);
     om(2),-om(3),0,om(1);   om(3),om(2),-om(1),0];

k_norm = 1.0;                             % quaternion drift correction
xdot = [ v;
         (Rwb*[0;0;Th])/prm.m - [0;0;g];
         0.5*Omq*q + k_norm*(1-(q.'*q))*q;
         prm.Jin*(tau - cross(om, prm.J*om)) ];

model = AcadosModel();
model.name        = 'quad13w';
model.x           = x;
model.u           = u;
model.f_expl_expr = xdot;
model.xdot        = SX.sym('xdot',nx);
model.f_impl_expr = model.xdot - xdot;

ocp = AcadosOcp();
ocp.model = model;

y_expr   = [p; v; om; Th - prm.m*g; tau];  % 3+3+3+1+3 = 13
y_expr_e = [p; v];                         % 6

ocp.cost.cost_type   = 'NONLINEAR_LS';
ocp.cost.cost_type_0 = 'NONLINEAR_LS';
ocp.cost.cost_type_e = 'NONLINEAR_LS';
ocp.model.cost_y_expr   = y_expr;
ocp.model.cost_y_expr_0 = y_expr;
ocp.model.cost_y_expr_e = y_expr_e;

W = diag([Wp(:); Wv(:); Ww(:); WT; Wt(:)]);
ocp.cost.W   = W;
ocp.cost.W_0 = W;
ocp.cost.W_e = diag([Wp(:); Wv(:)]);
ocp.cost.yref   = zeros(13,1);
ocp.cost.yref_0 = zeros(13,1);
ocp.cost.yref_e = zeros(6,1);

% wrench box: the real actuator envelope, the same one actuator_sat enforces
ocp.constraints.lbu   = [lim.T_min; -lim.tau_max(:)];
ocp.constraints.ubu   = [lim.T_max;  lim.tau_max(:)];
ocp.constraints.idxbu = (0:nu-1)';
ocp.constraints.x0    = x0_traj;          % overwritten each step by constr_x0

ocp.solver_options.N_horizon             = N;
ocp.solver_options.tf                    = N*Ts;
ocp.solver_options.qp_solver             = 'PARTIAL_CONDENSING_HPIPM';
ocp.solver_options.nlp_solver_type       = 'SQP_RTI';
ocp.solver_options.integrator_type       = 'ERK';
ocp.solver_options.regularize_method     = 'CONVEXIFY';
ocp.solver_options.sim_method_num_stages = 4;
ocp.solver_options.sim_method_num_steps  = 2;
ocp.solver_options.qp_solver_warm_start  = 1;
ocp.solver_options.hpipm_mode            = 'ROBUST';
ocp.solver_options.qp_solver_iter_max    = 100;

solver = AcadosOcpSolver(ocp);

%% initial conditions
xp = x0_traj;
st = [];

uNMPC = zeros(nu, k_sim);                 % [T_spec; om_ref] handed to the plant
th    = zeros(k_sim,1);
X_log = zeros(k_sim+1,13);  X_log(1,:) = xp.';
P_ref = zeros(k_sim+1,3);   P_ref(1,:) = r_pos(0).';
t_NMPC = (0:k_sim)*Ts;
n_ok  = k_sim;

%%
for i = 1:k_sim
tic;
t_k = (i-1)*Ts;

solver.set('constr_x0', xp);
for jj = 0:N-1
    tj = t_k + jj*Ts;
    solver.set('cost_y_ref', [r_pos(tj); r_vel(tj); zeros(7,1)], jj);
end
tN = t_k + N*Ts;
solver.set('cost_y_ref_e', [r_pos(tN); r_vel(tN)], N);

solver.solve();                            % returns nothing; poll the status
if solver.get('status') ~= 0
    warning('acados status %d at t = %.2f s', solver.get('status'), t_k);
end

u0 = solver.get('u', 0);
x1 = solver.get('x', 1);
v_cmd = [u0(1)/prm.m; x1(11:13)];          % [T_spec; om_des] from node 1
uNMPC(:,i) = v_cmd;
th(i)      = toc;

% output update -- the SAME plant SPC is controlled against
[xp, st] = quad_plant(xp, v_cmd, t_k, prm, lim, inner, Ts, Ts_inner, st);

X_log(i+1,:) = xp.';
P_ref(i+1,:) = r_pos(i*Ts).';

if ~all(isfinite(xp)) || norm(xp(1:3) - r_pos(i*Ts)) > 50
    warning('diverged at t = %.2f s', i*Ts);
    n_ok = i;  break
end
end

uNMPC = uNMPC(:,1:n_ok);  th = th(1:n_ok);
X_log = X_log(1:n_ok+1,:);  P_ref = P_ref(1:n_ok+1,:);  t_NMPC = t_NMPC(1:n_ok+1);

%% New plot
e = X_log(:,1:3) - P_ref;
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
plot(t_NMPC,P_ref,'LineWidth',2,'LineStyle','--');
plot(t_NMPC,X_log(:,1:3),'LineWidth',2);
legend('x_{ref}','y_{ref}','z_{ref}','x','y','z','Location','northeast');
ylabel('$p$ [m]',Interpreter='latex')
axis tight
grid on
subplot(2,1,2)
hold on;
plot(t_NMPC(1:n_ok),uNMPC.','LineWidth',2);
ylabel('$u$',Interpreter='latex')
xlabel('$t$[s]',Interpreter='latex')
legend('T','\omega_x','\omega_y','\omega_z','Location','southeast');
axis tight
grid on;
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
savefig('figures/NMPC_quad.fig')

figure; hold on; grid on; axis equal
plot3(X_log(:,1),X_log(:,2),X_log(:,3),'LineWidth',2);
plot3(P_ref(:,1),P_ref(:,2),P_ref(:,3),'--','LineWidth',1.5);
xlabel('x [m]'); ylabel('y [m]'); zlabel('z [m]');
legend('flown','reference'); view(3);

save('data/NMPC','t_NMPC','uNMPC','X_log','P_ref')
