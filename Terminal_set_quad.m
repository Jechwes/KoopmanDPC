%% Terminal_set_quad.m — MIMO adaptation of Chart/Terminal_set.m
% Steps:
%  1) evaluate the learned observables z(k) along the training data
%  2) regress the lifted linear predictor  Theta : [z(k); u_f] -> [z(k+1); ...; z(k+N)]
%     and extract the one-step model  z(k+1) = A z(k) + B u(k)
%  3) terminal cost P and terminal gain K via idare (replaces the Chart LMI,
%     which is numerically intractable at n_basis = 40 with near-unit eigenvalues)
%  4) ELLIPSOIDAL terminal set  z'Pz <= alpha  (replaces MPT3 polyhedral set,
%     which does not scale to n_basis = 40 dimensions)
%
% Requires: Control System Toolbox (idare). Run AFTER model_learning_quad.py.
clc; close all; clear all;

load('data/weight1.mat'); load('data/weight2.mat'); load('data/weight3.mat');
load('data/params.mat');   % Tini, N, n_basis, nu, ny
load('data/u_data.mat');   load('data/y_data.mat'); load('data/run_lengths.mat');
weight1 = double(weight1); weight2 = double(weight2);
Tini = double(Tini); N = double(N); n_basis = double(n_basis);
nu = double(nu); ny = double(ny);

Q = 10*eye(n_basis);      % observable-state cost (as in Chart)
R = 0.01*eye(nu);            % input cost
stride = 3;               % subsample regression columns (speed/memory)

%% 1) + 2) build regression data per run (windows never cross run boundaries)
Phi_p = {}; Phi_f = {};
start = 0;
for rr = 1:numel(run_lengths)
    L = run_lengths(rr);
    u = u_data(start+(1:L),:);   y = y_data(start+(1:L),:);
    M = L - Tini - N;
    X1 = zeros((Tini-1)*nu + Tini*ny, M);
    UF = zeros(N*nu, M);
    for i = 1:M
        u_ini = u(i+1:i+Tini-1, :);          % (Tini-1) x nu
        y_ini = y(i+1:i+Tini,   :);          % Tini x ny
        ufm   = u(i+Tini:i+Tini+N-1, :);     % N x nu
        % time-major flattening — must match model_learning_quad.py
        X1(:,i) = [reshape(u_ini',[],1); reshape(y_ini',[],1)];
        UF(:,i) = reshape(ufm',[],1);
    end
    Zr = tanh(weight2*tanh(weight1*X1));     % n_basis x M observables

    idx = 1:stride:(M-N);
    Pp = zeros(n_basis + N*nu, numel(idx));
    Pf = zeros(N*n_basis,      numel(idx));
    for jj = 1:numel(idx)
        j = idx(jj);
        zf = Zr(:, j+1:j+N);
        Pf(:,jj) = zf(:);                    % [z(j+1); ...; z(j+N)]
        Pp(:,jj) = [Zr(:,j); UF(:,j)];       % [z(j); u_f(j)]
    end
    Phi_p{rr} = Pp;  Phi_f{rr} = Pf;
    start = start + L;
end
Phi_p = cell2mat(Phi_p);  Phi_f = cell2mat(Phi_f);
fprintf('Regression: %d samples, Theta is %dx%d\n', size(Phi_p,2), N*n_basis, n_basis+N*nu);

Theta = Phi_f/Phi_p;                         % least squares Theta*Phi_p ~ Phi_f
res = norm(Theta*Phi_p - Phi_f,'fro')/norm(Phi_f,'fro');
fprintf('Theta relative residual: %.4f\n', res);

A = Theta(1:n_basis, 1:n_basis);
B = Theta(1:n_basis, n_basis+1:n_basis+nu);
fprintf('open-loop  max|eig(A)|     = %.4f\n', max(abs(eig(A))));
fprintf('rank(ctrb) = %d / %d\n', rank(ctrb(A,B)), n_basis);

%% 3) terminal cost P and gain K via the discrete-time Riccati equation
% The Chart LMI encodes (A+BK)'P(A+BK) - P <= -(Q + K'RK); the LQR solution
% from the DARE satisfies this with equality and is numerically robust even
% for near-marginally-stable lifted dynamics (max|eig(A)| ~ 1), where SDP
% solvers blow up. No YALMIP/SDP solver needed.
[P, K_dare, ~] = idare(A, B, Q, R);
assert(~isempty(P), 'idare failed: (A,B) not stabilizable or too ill-conditioned');
P = (P+P')/2;
K = -K_dare;                    % convention u = K z  =>  A+BK = A - B*K_dare
fprintf('closed-loop max|eig(A+BK)| = %.4f\n', max(abs(eig(A+B*K))));
fprintf('cond(P) = %.2e, max eig(P) = %.2e\n', cond(P), max(eig(P)));

%% 4) ellipsoidal terminal set  z'Pz <= alpha
% Invariance of {z'Pz<=alpha} under z+ = (A+BK)z follows from the LMI.
% alpha is the largest level set on which the terminal controller u = Kz
% respects the input bounds:  |K_i z| <= u_bnd_i  for all z'Pz <= alpha
%   max_{z'Pz<=alpha} |K_i z| = sqrt(alpha * K_i inv(P) K_i')
% Input bounds: taken from the range seen in the (normalized) training data —
% the predictor is only trusted there anyway.
u_bnd = min(abs(min(u_data)), abs(max(u_data)))';   % nu x 1, symmetric box
alpha = min( (u_bnd.^2) ./ diag(K*(P\K')) );
fprintf('alpha = %.4f | u_bnd (normalized) = %s\n', alpha, mat2str(round(u_bnd,2)'));

save('data/invariant.mat', 'Theta', 'A', 'B', 'K', 'P', 'alpha', 'u_bnd');
disp('Saved data/invariant.mat — next: KDPC_quad_sim.m');
