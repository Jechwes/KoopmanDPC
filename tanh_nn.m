function phi = tanh_nn(param, u_ini, u_f, y_ini)
%TANH_NN  Koopman observables, MIMO version of KDPC_RSRFG/Pendulum/tanh_nn.m
%
%   phi = [ z ; u_f ]   with   z = phi~(x_ini) = phi(x_ini) - phi(0)
%   phi = z             if u_f is empty
%
%   u_ini : (Tini-1)*nu x K   past inputs,  time-major
%   y_ini :  Tini*ny    x K   past outputs, time-major
%   u_f   :  N*nu       x K   future inputs, time-major   (may be [])
%
% Accepts K windows at once, one per column.
%
% TWO DIFFERENCES FROM THE REFERENCE
% ----------------------------------
% 1. Biases are used. Without them tanh(Wx) is odd, phi(-x) = -phi(x), which
%    forces the whole predictor to be odd about the operating point. That is
%    correct for the pendulum (its only nonlinearity is sin(x2)) but wrong for
%    a quadrotor. The reference drops the bias terms; we must not.
%
% 2. The observables are shifted by z_eq = phi(0).
%    Assumption 2.4 of the paper requires the observables to VANISH AT THE
%    EQUILIBRIUM. With biases, phi(0) = tanh(W2*tanh(b1)+b2) is some arbitrary
%    nonzero vector, so z'Qz would no longer measure distance to hover and the
%    controller would regulate to the wrong point. Subtracting z_eq gives
%    phi~(0) = 0 exactly. Because Theta is then regressed with no intercept, a
%    linear map necessarily sends 0 -> 0, so the origin is a true fixed point
%    of the lifted model and the theory holds as written.

W1 = double(param.weight1);  b1 = double(param.bias1(:));
W2 = double(param.weight2);  b2 = double(param.bias2(:));

z_eq = tanh(W2*tanh(b1) + b2);                        % phi(0)
z    = tanh(W2*tanh(W1*[u_ini; y_ini] + b1) + b2) - z_eq;

if nargin < 3 || isempty(u_f)
    phi = z;
else
    phi = [z; u_f];
end
end
