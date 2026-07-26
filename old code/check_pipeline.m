%% check_pipeline.m -- consistency guard for the whole pipeline.
% Run after any retrain, before KDPC_quad_sim.m. This exists because the repo
% silently reached a state where invariant.mat had been built at n_basis=80,
% params.mat said 40, and scalers.mat carried 9 outputs against ny=4 -- none of
% which announced itself until a dimension error deep inside the simulation.
clc; clear;
fprintf('=== KDPC pipeline check ===\n');

cfg = koopman_config();
fails = {};

%% 0) artifacts present
need = {'data/config.mat','data/net.mat','data/scalers.mat','data/u_data.mat', ...
        'data/y_data.mat','data/run_lengths.mat','data/invariant.mat'};
missing = need(cellfun(@(f) exist(f,'file')~=2, need));
if ~isempty(missing)
    fprintf(2,'  [FAIL] missing artifacts: %s\n', strjoin(missing,', '));
    error('Pipeline incomplete -- run the stages in order (see README.md).');
end

net = load_net();
load('data/scalers.mat');   load('data/invariant.mat');
load('data/u_data.mat');    load('data/y_data.mat');   load('data/run_lengths.mat');
S = load('data/config.mat');
nb = cfg.n_basis; Ti = cfg.Tini; Nh = cfg.N; nu = cfg.nu; ny = cfg.ny;

%% 1) dimensions agree across every artifact
fails = add(fails, net.n_basis==nb,   sprintf('net.mat n_basis=%d vs config %d',net.n_basis,nb));
fails = add(fails, size(A,1)==nb,     sprintf('invariant A is %dx%d vs n_basis %d',size(A,1),size(A,2),nb));
fails = add(fails, size(y_data,2)==ny,sprintf('y_data has %d cols vs ny=%d',size(y_data,2),ny));
fails = add(fails, numel(y_std)==ny,  sprintf('scalers hold %d outputs vs ny=%d',numel(y_std),ny));
fails = add(fails, double(S.Tini)==Ti && double(S.N)==Nh, 'config.mat stale -- re-run export_config.m');
fails = add(fails, net.in1==(Ti-1)*nu+Ti*ny, 'network input width disagrees with Tini/nu/ny');
fails = add(fails, isequal(size(Theta),[Nh*nb, nb+Nh*nu]), 'Theta has the wrong size');

%% 2) properties the paper's theorems depend on
fails = add(fails, nnz(Theta(1:nb, nb+nu+1:end))==0, ...
    'Theta block row 1 is NOT causal -- future inputs predict z(k+1) (Remark 3.1)');
fails = add(fails, norm(lift(zeros(net.in1,1), net))<1e-10, ...
    'phi(0) ~= 0 -- Assumption 2.4 violated, the z_eq shift is broken');
fails = add(fails, min(eig((Q+Q')/2))>0, 'Q is not positive definite (needed by Lemma 4.3)');
fails = add(fails, min(eig((P+P')/2))>0, 'P is not positive definite');
fails = add(fails, max(abs(eig(A+B*K)))<1, 'terminal gain K is not stabilising (Assumption 4.1)');
lmi = (A+B*K)'*P*(A+B*K) - P + Q + K'*R*K;
fails = add(fails, max(eig((lmi+lmi')/2))<1e-6, ...
    sprintf('LMI (13a) violated, max eig = %.2e', max(eig((lmi+lmi')/2))));
fails = add(fails, alpha>0, sprintf('terminal set empty (alpha=%.3g)', alpha));

%% 3) MATLAB and Python must build identical regressors
[X1m,~,~] = build_regressors(u_data, y_data, run_lengths, Ti, Nh);
if exist('data/X1_py.mat','file')==2
    Xp = double(getfield(load('data/X1_py.mat'),'X1_py'));  %#ok<GFLD>
    d  = norm(X1m(:,1:size(Xp,2)) - Xp, 'fro');
    fails = add(fails, d<1e-4, sprintf('MATLAB/Python regressors differ, ||d||=%.3e', d));
else
    fprintf('  [skip] data/X1_py.mat absent -- cross-language regressor check not run\n');
end

%% 4) diagnostics (informational, never fatal)
Zs = lift(X1m(:, 1:min(5000, size(X1m,2))), net);
inT = mean(sum((Zs'*P).*Zs', 2) <= alpha);
fprintf('\n--- diagnostics ---\n');
fprintf('  max|eig(A)|          = %.4f%s\n', max(abs(eig(A))), ...
    tern(max(abs(eig(A)))>1.02, '   <- lifted model unstable (expected for a drone)',''));
fprintf('  cond(P)              = %.2e%s\n', cond(P), tern(cond(P)>1e10,'   <- ill-conditioned',''));
fprintf('  alpha                = %.4f\n', alpha);
fprintf('  lifts inside X_T     = %.2f%%%s\n', 100*inT, ...
    tern(inT<0.01, sprintf('   <- effectively empty; use terminal_mode=''%s'', not ''hard''', 'soft'),''));
if inT < 0.01 && strcmp(cfg.terminal_mode,'hard')
    fprintf(2,'  [WARN] terminal_mode=''hard'' with an empty X_T -> the QP will be infeasible.\n');
end
fprintf('  ||phi||_rms on data  = %.3f\n', sqrt(mean(Zs(:).^2)));

%% verdict
fprintf('\n');
if isempty(fails)
    fprintf('=== ALL CHECKS PASSED ===\n');
else
    for i = 1:numel(fails), fprintf(2,'  [FAIL] %s\n', fails{i}); end
    error('%d check(s) failed -- fix before running KDPC_quad_sim.m.', numel(fails));
end

%% ------------------------------------------------------------------------
function f = add(f, cond, msg)
if ~logical(cond), f{end+1} = msg; end
end

function s = tern(c,a,b)
if c, s = a; else, s = b; end
end
