function q = pctl(X, p)
%PCTL  Column-wise percentiles, without the Statistics Toolbox.
%
%   q = pctl(X, p)   X: n x m data, p: vector of percentiles in [0,100]
%                    q: numel(p) x m
%
% Drop-in replacement for prctile(X,p) on matrices, using the same linear
% interpolation convention (MATLAB's midpoint/"Hazen" definition), so results
% match prctile to floating-point tolerance. Exists only so the pipeline does
% not require the Statistics and Machine Learning Toolbox.

if isvector(X), X = X(:); end
n = size(X,1);
Xs = sort(X, 1);
% sample percentiles of the sorted values: (i-0.5)/n * 100
pos = (0.5:1:(n-0.5)).' / n * 100;
q = zeros(numel(p), size(X,2));
for c = 1:size(X,2)
    q(:,c) = interp1(pos, Xs(:,c), p(:), 'linear', NaN);
    % clamp the tails, matching prctile's extrapolation to the extremes
    q(p(:) <= pos(1),   c) = Xs(1,c);
    q(p(:) >= pos(end), c) = Xs(end,c);
end
end
