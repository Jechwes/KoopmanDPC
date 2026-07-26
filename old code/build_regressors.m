function [X1, UF, YF, col_lens] = build_regressors(u, y, lens, Tini, N)
%BUILD_REGRESSORS  Hankel windows for the Koopman SPC problem, eq. (6).
%
%   u    : T x nu normalised inputs      (all runs concatenated)
%   y    : T x ny normalised outputs
%   lens : samples per run -- windows never cross a run boundary
%
%   X1       : ((Tini-1)*nu + Tini*ny) x M   past window  [u_ini; y_ini]
%   UF       : (N*nu) x M                    future inputs u(k..k+N-1)
%   YF       : (N*ny) x M                    future outputs y(k+1..k+N)
%   col_lens : columns contributed by each run. Consecutive columns within a
%              run are consecutive time steps; across runs they are NOT.
%              Terminal_set_quad.m needs this to build z(k+1..k+N) stacks
%              without splicing two different flights together.
%
% Per the paper's definitions:
%   u_ini(k) = col(u(k-Tini+1), ..., u(k-1))   -> (Tini-1)*nu
%   y_ini(k) = col(y(k-Tini+1), ..., y(k))     -> Tini*ny
%
% FLATTENING IS TIME-MAJOR and must match build_regressors() in
% model_learning_quad.py exactly.  This file used to exist in three
% independent copies (here, the .py, and inline in Terminal_set_quad.m), kept
% in sync only by a comment.  check_pipeline.m now verifies they agree.

nu = size(u,2);  ny = size(y,2);
tot = sum(double(lens)) - numel(lens)*(Tini + N);
X1 = zeros((Tini-1)*nu + Tini*ny, tot);
UF = zeros(N*nu, tot);
YF = zeros(N*ny, tot);

col_lens = zeros(numel(lens),1);
start = 0;  col = 0;
for rr = 1:numel(lens)
    L  = double(lens(rr));
    uu = u(start+(1:L), :);
    yy = y(start+(1:L), :);
    col_lens(rr) = L - Tini - N;
    for i = 1:(L - Tini - N)
        col = col + 1;
        u_ini = uu(i+1 : i+Tini-1,       :);   % (Tini-1) x nu
        y_ini = yy(i+1 : i+Tini,         :);   % Tini x ny
        u_f   = uu(i+Tini : i+Tini+N-1,  :);   % N x nu
        y_f   = yy(i+Tini+1 : i+Tini+N,  :);   % N x ny
        X1(:,col) = [reshape(u_ini.',[],1); reshape(y_ini.',[],1)];
        UF(:,col) = reshape(u_f.',[],1);
        YF(:,col) = reshape(y_f.',[],1);
    end
    start = start + L;
end
assert(col == tot, 'regressor count mismatch');
end
