%% SPC vs KSPC on the figure 8 (slow, normal, fast): 3D paths and error
% Reads the flight logs converted by export_mat.py (logs/mat/fly_*.mat), so in
% WSL run that first:  python export_mat.py
clc; clear; close all;

%% settings
% show     = {'slow', 'normal', 'fast'};   % any of these, e.g. {'fast'} or {'slow', 'fast'}
show     = {'xfast'};
skip_s   = 5;           % s after SPC engages, left out of the RMSE
n_loops  = 2;           % last loops drawn in 3D (the RMSE uses all); Inf = all
settle_k = 1;           % settled once the error drops below settle_k x the RMSE
logs_in  = 'logs/good'; % flights to use, converted by: python export_mat.py logs/good

speeds = struct('label', {'fast', 'xfast', 'tini5', 'tini10', 'tini15'}, ...
                'traj',  {'fig8_planes_fast', 'fig8_planes_xfast', 'fig8_planes_fast_tini5', ...
                          'fig8_planes_fast_tini10', 'fig8_planes_fast_tini15'}, ...   % as in the log names
                'w',     {1.1, 1.21, 1.1, 1.1, 1.1});                          % rad/s, reference.py
bad = setdiff(show, {speeds.label});
assert(isempty(bad), 'unknown in show: %s (use slow, normal, fast)', strjoin(bad, ', '));
speeds = speeds(ismember({speeds.label}, show));

ctrls = {'spc', 'kspc'};
names = {'SPC', 'KSPC'};
c     = [0.165 0.471 0.839;          % SPC,  blue    (as in plot_compare.py)
         0.922 0.408 0.204];         % KSPC, orange
c_ref = [0 0 0];         % reference, gray

here   = fileparts(mfilename('fullpath'));
logdir = fullfile(here, logs_in, 'mat');
assert(isfolder(logdir), 'no %s -- run "python export_mat.py %s" first', logdir, logs_in);

%% pick a representative run per speed and controller
runs = struct();
for i = 1:numel(speeds)
    for j = 1:numel(ctrls)
        files = dir(fullfile(logdir, sprintf('fly_%s_%s_*.mat', speeds(i).traj, ctrls{j})));
        assert(~isempty(files), 'no %s flight of %s in %s', names{j}, speeds(i).traj, logdir);
        [~, o] = sort({files.name});  files = files(o);  % by name = by time
        rmse = zeros(numel(files), 1);
        dur  = zeros(numel(files), 1);
        for f = 1:numel(files)
            d = load(fullfile(logdir, files(f).name), 't', 'pos', 'p_ref');
            k = d.t - d.t(1) > skip_s;
            rmse(f) = sqrt(mean(sum((d.pos(k,:) - d.p_ref(k,:)).^2, 2)));
            dur(f)  = d.t(end) - d.t(1);
        end
        full = dur >= 0.9*max(dur);                  % aborted runs stop early: left out
        if any(~full)
            fprintf('%-7s %-4s: left out, stopped early: %s\n', speeds(i).label, ...
                    names{j}, strjoin({files(~full).name}, ', '));
        end
        files = files(full);  rmse = rmse(full);
        dist = round(abs(rmse - mean(rmse)), 6);     % two runs always tie
        pick = find(dist == min(dist), 1, 'last');   % -> the latest
        d = load(fullfile(logdir, files(pick).name), 't', 'pos', 'p_ref');

        t   = d.t - d.t(1);
        err = 100*vecnorm(d.pos - d.p_ref, 2, 2);                 % cm
        k0  = find(err < settle_k*100*rmse(pick), 1);              % handover over
        if isempty(k0), k0 = find(t > skip_s, 1); end
        runs(i, j).t        = t;
        runs(i, j).err      = err;
        runs(i, j).t_settle = t(k0);

        k = t > skip_s & d.t >= d.t(end) - n_loops*2*pi/speeds(i).w;   % last n_loops
        runs(i, j).pos  = d.pos(k, :);
        runs(i, j).ref  = d.p_ref(k, :);
        runs(i, j).rmse = mean(rmse);
        runs(i, j).n    = numel(files);
        fprintf('%-7s %-4s: %s  (%.1f cm; mean of %d runs %.1f cm), settled at %.1f s\n', ...
                speeds(i).label, names{j}, files(pick).name, 100*rmse(pick), ...
                numel(files), 100*mean(rmse), t(k0));
    end
end

% one set of limits per row, so the speeds are directly comparable
P    = vertcat(runs.pos, runs.ref);
lim  = [min(P); max(P)] + [-1; 1]*0.05;
t0   = arrayfun(@(i) max([runs(i, :).t_settle]), 1:numel(speeds));  % both settled
emax = 0;
for i = 1:numel(speeds)
    for j = 1:numel(ctrls)
        emax = max(emax, max(runs(i, j).err(runs(i, j).t >= t0(i))));
    end
end
emax = 1.15*emax;

%% plot
fh = gobjects(1, numel(speeds));
fe = gobjects(1, numel(speeds));
for i = 1:numel(speeds)
    ttl = sprintf('%s (omega = %.1f rad/s)', speeds(i).label, speeds(i).w);
    n = arrayfun(@(rr) sprintf(' (n = %d)', rr.n), runs(i, :), 'UniformOutput', false);
    n([runs(i, :).n] == 1) = {''};
    rmse_txt = sprintf('RMSE after %g s:  SPC %.1f cm%s,  KSPC %.1f cm%s', skip_s, ...
                       100*runs(i,1).rmse, n{1}, 100*runs(i,2).rmse, n{2});

    % 3D path, last loops
    fh(i) = figure;
    ax = axes(fh(i), 'FontSize', 11, 'FontName', 'Times New Roman');
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');

    r = runs(i, 1).ref;
    h = plot3(ax, r(:,1), r(:,2), r(:,3), '--', 'LineWidth', 1.2, 'Color', c_ref);
    for j = 1:numel(ctrls)
        p = runs(i, j).pos;
        h(end+1) = plot3(ax, p(:,1), p(:,2), p(:,3), 'LineWidth', 1.5, ...
                         'Color', c(j,:)); %#ok<SAGROW>
    end

    axis(ax, 'equal');
    axis(ax, lim(:)');
    view(ax, -35, 25);
    xlabel(ax, 'x [m]'); ylabel(ax, 'y [m]'); zlabel(ax, 'z [m]');
    legend(ax, h, [{'reference'}, names], 'Location', 'southoutside', ...
           'Orientation', 'horizontal', 'FontSize', 11, 'FontName', 'Times New Roman');

    % error, from settled on the trajectory to the end
    fe(i) = figure;
    ax = axes(fe(i), 'FontSize', 11, 'FontName', 'Times New Roman');
    hold(ax, 'on'); grid(ax, 'on'); box(ax, 'on');
    he = gobjects(1, numel(ctrls));
    for j = 1:numel(ctrls)
        k = runs(i, j).t >= t0(i);
        he(j) = plot(ax, runs(i, j).t(k), runs(i, j).err(k), 'LineWidth', 1.2, 'Color', c(j,:));
    end
    xlim(ax, [t0(i) max(cellfun(@(t) t(end), {runs(i, :).t}))]);
    ylim(ax, [0 emax]);
    xlabel(ax, 'time since SPC engaged [s]');
    ylabel(ax, 'position error [cm]');
    legend(ax, he, names, 'Location', 'southoutside', 'Orientation', 'horizontal', ...
           'FontSize', 11, 'FontName', 'Times New Roman');

    fprintf('%s, figure %d (3D) and figure %d (error): %s\n', ...
            ttl, fh(i).Number, fe(i).Number, rmse_txt);
end

%% save, the full set as fig8_3d, a selection as e.g. fig8_3d_fast
% name = 'fig8_3d';
% if numel(speeds) < 3, name = [name '_' strjoin({speeds.label}, '_')]; end
% outdir = fullfile(here, 'figures');
% if ~isfolder(outdir), mkdir(outdir); end
% savefig(fh, fullfile(outdir, [name '.fig']));
% exportgraphics(fh, fullfile(outdir, [name '.png']), 'Resolution', 200);
% % vector export last: once in batch mode it left the figure handle invalid
% ws = warning('off', 'MATLAB:print:ContentTypeImageSuggested');   % vector is wanted
% exportgraphics(fh, fullfile(outdir, [name '.pdf']), 'ContentType', 'vector');
% warning(ws);
% fprintf('saved figures/%s.fig, .png and .pdf\n', name);
