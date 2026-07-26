%% simulation_Xi.m  --  MIMO version of KDPC_RSRFG/Pendulum/simulation_Xi.m
%
% Koopman DPC with the interpolated initial state, problem (10) of the paper:
%
%   min  Z'*Omega*Z + u'*Psi*u + z0'*Q*z0 + lambda*||e(k)||^2*xi^2
%   s.t. Z  = Theta*[z0; u]                       (10b) multi-step prediction
%        z0 = (1-xi)*phi(x_ini) + xi*z*_{1|k-1}   (10c) interpolated i.c.
%        0 <= xi <= 1                             (10f)
%        input bounds                             (10d)
%
% The plant here is the LIFTED MODEL itself, so there is no plant/model
% mismatch — the same situation the reference is in, where the simulated
% pendulum is exactly the system the data came from. This is the configuration
% that must work before anything harder is attempted. Swap in a rigid-body
% simulator later by replacing the marked block.
clc; close all; clear all;

load('data/weight1.mat'); load('data/weight2.mat'); load('data/weight3.mat');
load('data/params.mat'); load('data/invariant.mat'); load('data/scalers.mat');
weights = struct('weight1',weight1,'bias1',bias1,'weight2',weight2,'bias2',bias2);
Tini = double(Tini); N = double(N); n_basis = double(n_basis);
nu = double(nu); ny = double(ny);

Tlength = 300;              % simulation steps (3 s at 100 Hz)
lambda  = 1e+6;             % cost on the initial condition (reference: 1e10)
u_bnd   = 2.0;              % input bound, in normalised units (~2 sigma)

% Readout y(k+1) = C z + D u, from the first block row of weight3.
% The network computes  y = W3*[phi; u_f] + b3  with the UNSHIFTED phi, while
% tanh_nn.m returns phi~ = phi - phi(0). Substituting phi = phi~ + z_eq:
%     y = C*phi~ + D*u + ( C*z_eq + b3(1:ny) )
% The bracket is the model's predicted output at the equilibrium; it should be
% ~0 if the training mean really is a trim point. It is subtracted so that
% (z,u) = (0,0) gives y = 0, i.e. the plant used below has its equilibrium
% where the controller thinks it is. Its size tells you how far the data mean
% is from a true hover trim.
C = weight3(1:ny, 1:n_basis);
D = weight3(1:ny, n_basis+(1:nu));
z_eq  = tanh(weight2*tanh(bias1(:)) + bias2(:));
y_off = C*z_eq + bias3(1:ny).';
fprintf('readout offset at equilibrium ||y_off|| = %.4f (normalised)\n', norm(y_off));

%% Controller
Psi   = kron(eye(N), R);                                 % input cost
Omega = kron(eye(N), Q);
Omega = blkdiag(Omega(1:end-n_basis,1:end-n_basis), P);  % terminal cost on z_N

u       = sdpvar(N*nu,1);       % future inputs (time-major)
Z       = sdpvar(N*n_basis,1);  % predicted lifted states
nl_part = sdpvar(n_basis,1);    % phi(x_ini(k)) from measurement
z_part  = sdpvar(n_basis,1);    % z*_{1|k-1}, predicted at the previous step
z0      = sdpvar(n_basis,1);    % initial condition used by the MPC
xi      = sdpvar(1,1);          % interpolation variable

objective = Z'*Omega*Z + u'*Psi*u + z0'*Q*z0 ...
          + lambda*(z_part-nl_part)'*(z_part-nl_part)*xi^2;      % eq. (18)

constraints = [ Z  == Theta*[z0; u];
                z0 == (1-xi)*nl_part + xi*z_part;
                0 <= xi <= 1;
                -u_bnd <= u <= u_bnd ];

options    = sdpsettings('solver','quadprog','verbose',0);
controller = optimizer(constraints, objective, options, ...
                       {nl_part, z_part}, {u, Z, xi, z0});
disp('controller ready');

%% Initial condition: offset from the target, history held constant
y0    = [-0.4; 0.4; -0.3; 0] ./ y_std(:);     % start error, normalised
u_ini = zeros((Tini-1)*nu, 1);                % 0 deviation = hover
y_ini = repmat(y0, Tini, 1);

z_plant = tanh_nn(weights, u_ini, [], y_ini);   % plant state (lifted, shifted)

Y = y0.';  U = [];  Xi = [];  E_k = [];  Zkm1 = [];
for i = 1:Tlength
    Nl_part = tanh_nn(weights, u_ini, [], y_ini);       % phi~(x_ini(k))

    if i < Tini
        Z_part = Nl_part;
    else
        Z_part = Zkm1;
        E_k = [E_k, Zkm1 - Nl_part];          % e(k), eq. (12)
    end

    OUT  = controller({Nl_part, Z_part});
    Uk   = OUT{1};
    Zkm1 = OUT{2}(1:n_basis);                 % z*_{1|k}, shifted i.c. for next step
    Xi(i) = OUT{3};
    uk   = Uk(1:nu);                          % apply only the first input

    % ------------------------------ PLANT ------------------------------
    % y_off subtracted (see above) so that z = 0, u = 0 gives y = 0
    y_next  = C*z_plant + D*uk;
    z_plant = A*z_plant + B*uk;
    % -------------------------------------------------------------------

    u_ini = [u_ini(nu+1:end); uk];            % shift the history buffers
    y_ini = [y_ini(ny+1:end); y_next];
    Y = [Y; y_next.'];  U = [U; uk.'];
end
t = (0:Tlength)*Ts;

%% Plots
Y_phys = Y .* y_std;                          % back to metres / radians
U_phys = U .* u_std + u_mean;

figure
subplot(3,1,1); plot(t, Y_phys(:,1:3),'LineWidth',1.5); yline(0,'k--'); grid on
ylabel('position error [m]'); legend('x','y','z')
subplot(3,1,2); plot(t, Y_phys(:,4),'LineWidth',1.5); yline(0,'k--'); grid on
ylabel('yaw error [rad]')
subplot(3,1,3); plot(t(1:end-1), U_phys,'LineWidth',1); grid on
ylabel('\Omega [rad/s]'); xlabel('t [s]')

figure
subplot(2,1,1); plot(t(1:end-1), Xi,'LineWidth',1.5); grid on
ylabel('\xi'); title('interpolation variable')
subplot(2,1,2); plot(t(Tini:end-1), vecnorm(E_k),'LineWidth',1.5); grid on
ylabel('||e(k)||'); xlabel('t [s]'); title('Koopman model error')
