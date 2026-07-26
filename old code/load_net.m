function net = load_net()
%LOAD_NET  Load the trained observable network exported by model_learning_quad.py.
%
% Returns a struct with W1,b1,W2,b2 (the lifting) and the equilibrium offset
% z_eq required to satisfy Assumption 2.4 of the paper.
%
% WHY z_eq EXISTS
% ---------------
% Assumption 2.4 requires phi(0) = 0, and Assumption 2.2 fixes the origin as
% the equilibrium.  With bias=False that is automatic: tanh(W*0) = 0.
% Training with biases (which fits noticeably better) breaks it --
% phi(0) = tanh(W2*tanh(b1) + b2) is some arbitrary nonzero vector.  Every
% downstream result then collapses: z'Qz no longer measures distance to
% hover, the terminal set is centred on the wrong point, and the ISS argument
% of Theorem 4.4 concludes convergence of ||phi(x_ini)|| to zero -- which is
% no longer hover.
%
% The fix keeps the biases AND the theory: use the shifted observable
%     phi~(x) := phi(x) - phi(0),
% so phi~(0) = 0 exactly.  Because the multi-step predictor is then regressed
% WITHOUT an intercept, a linear map necessarily sends 0 -> 0, so
% (dz, u) = (0, 0) is an exact fixed point of the lifted model.  Assumptions
% 2.2 and 2.4 hold as stated.

f = 'data/net.mat';
assert(exist(f,'file')==2, ...
    ['%s not found. Run model_learning_quad.py first.\n' ...
     'If you only have the legacy weight1/2/3.mat files, they are STALE and ' ...
     'missing all bias terms -- retrain.'], f);
S = load(f);

req = {'W1','b1','W2','b2'};
for i = 1:numel(req)
    assert(isfield(S,req{i}), 'data/net.mat is missing %s (retrain).', req{i});
end

net.W1 = double(S.W1);  net.b1 = double(S.b1(:));
net.W2 = double(S.W2);  net.b2 = double(S.b2(:));

% phi(0): lift of an all-zero past window (u_ini = 0, y_ini = 0), which in
% normalised coordinates is exactly the trim / setpoint condition.
net.z_eq = tanh(net.W2*tanh(net.b1) + net.b2);
net.n_basis = size(net.W2,1);
net.in1     = size(net.W1,2);
end
