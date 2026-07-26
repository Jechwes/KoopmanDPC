function [x_next, st, Omega] = quad_plant_cascade(x, v, prm, st, cfg)
%QUAD_PLANT_CASCADE  Quadrotor with an on-board attitude-RATE PID inner loop.
%
%   v = [T_spec; wx_ref; wy_ref; wz_ref]
%       T_spec  : commanded specific thrust [m/s^2]  (i.e. thrust/mass)
%       w*_ref  : commanded body rates [rad/s]
%
% The inner PI rate loop runs at cfg.f_inner (default 500 Hz), the outer KDPC
% at 1/cfg.Ts (100 Hz). This is the standard quadrotor architecture and the
% one CASCADE_DEBATE.md recommends for this project.
%
% st is the controller state (rate-error integral); pass [] to initialise.

if isempty(st), st = struct('eint', zeros(3,1)); end
if ~isfield(cfg,'f_inner'), cfg.f_inner = 500; end
n_sub = max(1, round(cfg.f_inner * cfg.Ts));
h     = cfg.Ts / n_sub;

% Rate-loop gains. Crazyflie-class: the rate loop is deliberately stiff --
% closed-loop rate bandwidth ~30-50 Hz, i.e. 5-10x faster than the 100 Hz
% outer loop, which is what makes the timescale separation legitimate.
Kp = [220; 220; 120];
Ki = [500; 500; 200];
i_max = 200;

Alloc4 = [prm.c_t*ones(1,4); prm.Mw];      % [T_spec; wdot] <- Omega.^2
om2_min = cfg.omega_min^2;  om2_max = cfg.omega_max^2;

for s = 1:n_sub
    w   = x(11:13);
    err = v(2:4) - w;
    st.eint = max(min(st.eint + err*h, i_max), -i_max);
    wdot_des = Kp.*err + Ki.*st.eint;

    % allocation: solve for squared motor speeds, then saturate
    om2 = Alloc4 \ [v(1); wdot_des - prm.w_bias];
    om2 = max(min(om2, om2_max), om2_min);
    Omega = sqrt(om2);

    x = quad_plant(x, Omega, prm, h);
end
x_next = x;
end
