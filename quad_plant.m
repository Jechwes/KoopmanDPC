function [x, st, w_app] = quad_plant(x, v_cmd, t, prm, lim, inner, Ts_outer, Ts_inner, st)
% QUAD_PLANT  Advance the plant one OUTER step under a [T_spec; om_ref] command.
%
%   x      13x1 state [p(3); v(3); q(4, w first); om(3)]
%   v_cmd  4x1 [T_spec (m/s^2); om_ref (rad/s)]      <- what SPC outputs
%   st     inner-PID state; [] to initialise
%   w_app  4x1 wrench [T; tau] applied on the last sub-step

n_inner = round(Ts_outer/Ts_inner);

T_cmd  = max(min(prm.m*v_cmd(1), lim.T_max), lim.T_min);
om_ref = v_cmd(2:4);

w_app = zeros(4,1);
for j = 1:n_inner
    [tau_cmd, st]    = rate_pid(om_ref, x, inner, Ts_inner, st, lim);
    [T_app, tau_app] = actuator_sat(T_cmd, tau_cmd, prm, lim);
    w_app = [T_app; tau_app];
    x = rk4(@(tt,xx) quad_dynamics(tt, xx, w_app, prm), t + (j-1)*Ts_inner, x, Ts_inner);
    x(7:10) = x(7:10)/norm(x(7:10));
end
end

%% ------------------------------------------------------------------------
function [tau_cmd, st] = rate_pid(om_sp, x, inner, Ts, st, lim)
% inner attitude-rATE loop: body-rate setpoint -> body torque, 500 Hz.
if isempty(st)
    st = struct('e_int', zeros(3,1), 'om_prev', x(11:13));
end
om_act = x(11:13);
e_rate = om_sp(:) - om_act;

p_term = inner.Kp_rate(:).*e_rate;

d_raw      = -(om_act - st.om_prev)/Ts;
st.om_prev = om_act;
d_term     = inner.Kd_rate(:).*d_raw;

e_int_old = st.e_int;
e_int_try = max(min(e_int_old + e_rate*Ts, inner.i_max_att), -inner.i_max_att);
tau_try   = p_term + d_term + inner.Ki_rate(:).*e_int_try;

hold           = (abs(tau_try) > lim.tau_max(:)) & (sign(tau_try) == sign(e_rate));
st.e_int       = e_int_try;
st.e_int(hold) = e_int_old(hold);

tau_cmd = p_term + d_term + inner.Ki_rate(:).*st.e_int;
tau_cmd = max(min(tau_cmd, lim.tau_max(:)), -lim.tau_max(:));
end

%% ------------------------------------------------------------------------
function [T_app, tau_app] = actuator_sat(T_cmd, tau_cmd, prm, lim)
% apply actuator saturation
T_cmd   = max(min(T_cmd,   lim.T_max), lim.T_min);
tau_cmd = max(min(tau_cmd, lim.tau_max(:)), -lim.tau_max(:));

om2 = prm.Alloc \ [T_cmd; tau_cmd(:)];
om2 = max(min(om2, prm.omega_max^2), prm.omega_min^2);

w = prm.Alloc * om2;
T_app   = w(1);
tau_app = w(2:4);
end

%% ------------------------------------------------------------------------
function xdot = quad_dynamics(t, x, u, prm)
% continuous-time 13-state rigid body. x = [x;v;q;om]
v  = x(4:6);
q  = x(7:10)/norm(x(7:10));
om = x(11:13);
T  = u(1);
tau = u(2:4);

q0=q(1); q1=q(2); q2=q(3); q3=q(4);
Rwb=[1-2*(q2^2+q3^2), 2*(q1*q2-q0*q3),   2*(q1*q3+q0*q2);
     2*(q1*q2+q0*q3), 1-2*(q1^2+q3^2),   2*(q2*q3-q0*q1);
     2*(q1*q3-q0*q2), 2*(q2*q3+q0*q1),   1-2*(q1^2+q2^2)];

vdot = (Rwb*[0;0;T])/prm.m - [0;0;prm.g] - prm.cd*v/prm.m;
Om   = [0,-om(1),-om(2),-om(3); om(1),0,om(3),-om(2);
        om(2),-om(3),0,om(1);   om(3),om(2),-om(1),0];

xdot = [v; vdot; 0.5*Om*q; prm.Jin*(tau - cross(om, prm.J*om))];
end

%% ------------------------------------------------------------------------
function x = rk4(f, t, x, h)
k1 = f(t,     x);
k2 = f(t+h/2, x+h/2*k1);
k3 = f(t+h/2, x+h/2*k2);
k4 = f(t+h,   x+h*k3);
x  = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
end
