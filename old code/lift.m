function Z = lift(X1, net)
%LIFT  Equilibrium-shifted Koopman observables  phi~(x_ini) = phi(x_ini) - phi(0).
%
%   X1  : in1 x M matrix of stacked past windows [u_ini; y_ini] (time-major),
%         normalised, and already expressed RELATIVE TO THE SETPOINT.
%   net : struct from load_net().
%   Z   : n_basis x M observables, satisfying Z = 0 at the equilibrium.
%
% This is the ONLY place the network is evaluated.  Previously the expression
%   tanh(weight2*tanh(weight1*X1))
% was copy-pasted into three files and silently dropped every bias term, so
% MATLAB computed a different lifting than the one PyTorch trained.

Z = tanh(net.W2*tanh(net.W1*X1 + net.b1) + net.b2) - net.z_eq;
end
