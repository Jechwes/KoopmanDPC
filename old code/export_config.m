%% export_config.m -- mirror koopman_config() into data/config.mat for Python.
% model_learning_quad.py reads this instead of hard-coding hyperparameters.
% Run whenever koopman_config.m changes.
cfg = koopman_config();
if ~exist('data','dir'), mkdir('data'); end

Ts = cfg.Ts; Tini = cfg.Tini; N = cfg.N; n_basis = cfg.n_basis;
ny = cfg.ny; nu = cfg.nu;
out_labels = cfg.out_labels;
save('data/config.mat', 'Ts','Tini','N','n_basis','ny','nu','out_labels');
fprintf('Wrote data/config.mat  (Tini=%d, N=%d, n_basis=%d, ny=%d, nu=%d)\n', ...
        Tini, N, n_basis, ny, nu);
