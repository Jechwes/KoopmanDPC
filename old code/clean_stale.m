%% clean_stale.m -- remove artifacts from the pre-refactor pipeline.
% Run ONCE before the first full re-run. These files are not just outdated,
% they are actively dangerous: invariant.mat was built at n_basis=80 against a
% params.mat saying 40, and weight1/2/3.mat carry no bias terms.
% data/train and data/test (your CSVs) are never touched.
stale = { ...
    'data/weight1.mat', 'data/weight2.mat', 'data/weight3.mat', ... % no biases
    'data/invariant.mat', ...                                      % n_basis=80
    'data/scalers.mat', ...                                        % 9 outputs vs ny=4
    'data/params.mat', 'data/config.mat', ...
    'data/u_data.mat', 'data/y_data.mat', ...
    'data/u_data_test.mat', 'data/y_data_test.mat', ...
    'data/run_lengths.mat', 'data/plant_params.mat', ...
    'data/X.mat', 'data/y.mat', ...                                % ~52 MB, unused
    'data/X1_py.mat'};

n = 0; freed = 0;
for i = 1:numel(stale)
    if exist(stale{i},'file') == 2
        d = dir(stale{i});  freed = freed + d.bytes;
        delete(stale{i});   n = n + 1;
        fprintf('  removed %s\n', stale{i});
    end
end
fprintf('\nRemoved %d file(s), freed %.1f MB.\n', n, freed/1e6);
fprintf('Now run, in order:\n');
fprintf('   load_data -> export_config -> python model_learning_quad.py\n');
fprintf('   -> calibrate_plant -> Terminal_set_quad -> check_pipeline -> KDPC_quad_sim\n');
