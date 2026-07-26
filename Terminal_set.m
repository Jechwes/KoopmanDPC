%% Terminal_set.m  --  MIMO version of KDPC_RSRFG/Pendulum/Terminal_set.m
%
% Two jobs, exactly as in the reference:
%   1. build the multi-step predictor  Theta : [z(k); u_f] -> [z(k+1)..z(k+N)]
%      by least squares on the lifted data              (paper eq. (8))
%   2. extract A, B from its first block row, and compute the terminal cost P
%      and gain K                                       (Assumption 4.1)
%
% Difference from the reference: it solves an LMI with mosek and then builds a
% polyhedral terminal set with MPT3. At n_basis = 40 neither scales, so P and K
% come from the Riccati equation instead — the LQR solution satisfies LMI (13a)
% with equality, which is all Assumption 4.1 needs. The terminal SET is left
% out for now; we run with terminal cost only (see simulation_Xi.m).
clc; close all; clear all;

load('data/u_data.mat'); load('data/y_data.mat'); load('data/run_lengths.mat');
load('data/weight1.mat'); load('data/weight2.mat'); load('data/weight3.mat');
load('data/params.mat');
weights = struct('weight1',weight1,'bias1',bias1,'weight2',weight2,'bias2',bias2);
Tini = double(Tini); N = double(N); n_basis = double(n_basis);
nu = double(nu); ny = double(ny);

Q = 10*eye(n_basis);        % observable cost  (same value as the reference)
R = 0.01*eye(nu);           % input cost       (nu x nu now, was scalar)
stride = 5;                 % keep every 5th window. The reference has ~1000
                            % windows total; we have ~71000, and Phi_f would
                            % be 800 x 71000 (457 MB). Neighbouring windows
                            % overlap in 9 of 10 samples anyway, so they carry
                            % almost no extra information.

%% 1) lift the data and stack the regressors, run by run
ncols = 0;                                       % count first, then preallocate
for rr = 1:numel(run_lengths)
    ncols = ncols + numel(1:stride:(run_lengths(rr)-Tini-2*N));
end
Phi_p = zeros(n_basis + N*nu, ncols);
Phi_f = zeros(N*n_basis,      ncols);

start = 0; col = 0;
for rr = 1:numel(run_lengths)
    L = run_lengths(rr);
    u = u_data(start+(1:L),:);  y = y_data(start+(1:L),:);
    M = L - Tini - N;

    % assemble every window of this run, then lift them all in one product
    U_ini = zeros((Tini-1)*nu, M);
    Y_ini = zeros(Tini*ny,     M);
    UF    = zeros(N*nu,        M);
    for i = 1:M
        U_ini(:,i) = reshape(u(i+1:i+Tini-1,      :).', [], 1);   % time-major
        Y_ini(:,i) = reshape(y(i+1:i+Tini,        :).', [], 1);
        UF(:,i)    = reshape(u(i+Tini:i+Tini+N-1, :).', [], 1);
    end
    Z = tanh_nn(weights, U_ini, [], Y_ini);      % n_basis x M lifted states

    % pair z(k) with the N lifted states that follow it
    for j = 1:stride:(M-N)
        col = col + 1;
        Phi_p(:,col) = [Z(:,j); UF(:,j)];
        zf = Z(:, j+1:j+N);
        Phi_f(:,col) = zf(:);
    end
    start = start + L;
end
Phi_p = Phi_p(:,1:col);  Phi_f = Phi_f(:,1:col);
fprintf('regression: %d samples, Theta is %dx%d\n', ...
    col, N*n_basis, n_basis+N*nu);

%% 2) least squares for Theta, then A and B from the first block row
% No intercept in this regression: that is what makes z = 0, u = 0 an exact
% fixed point of the lifted model (a linear map sends 0 -> 0), so the shift
% applied in tanh_nn.m gives a genuine equilibrium at the origin.
Theta = Phi_f*pinv(Phi_p);          % pinv, as in the reference
A = Theta(1:n_basis, 1:n_basis);
B = Theta(1:n_basis, n_basis+(1:nu));

% sanity: the shifted observables must vanish at the equilibrium
z0chk = tanh_nn(weights, zeros((Tini-1)*nu,1), [], zeros(Tini*ny,1));
fprintf('||phi~(0)||        = %.2e  (must be ~0, Assumption 2.4)\n', norm(z0chk));

fprintf('relative residual  = %.4f\n', norm(Theta*Phi_p-Phi_f,'fro')/norm(Phi_f,'fro'));
fprintf('max|eig(A)|        = %.4f\n', max(abs(eig(A))));
fprintf('rank(ctrb(A,B))    = %d / %d\n', rank(ctrb(A,B)), n_basis);

%% 3) terminal cost P and gain K
[P, K_dare] = idare(A, B, Q, R);
assert(~isempty(P), 'idare failed: (A,B) not stabilisable');
P = (P+P')/2;
K = -K_dare;                                   % convention u = K z
fprintf('max|eig(A+BK)|     = %.4f\n', max(abs(eig(A+B*K))));

save('data/invariant.mat','Theta','A','B','K','P','Q','R');
disp('saved data/invariant.mat. Next: simulation_Xi.m');
