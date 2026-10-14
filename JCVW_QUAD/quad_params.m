function [prm, lim, inner] = quad_params()
%QUAD_PARAMS  Crazyflie 2.1 Brushless constants, actuator limits, inner gains.
%
% Single source of truth, so main_job.m and data/gather_data_sim.m identify on
% and control against exactly the same plant. Published values from the
% Busetto et al. benchmark (models/models.py).

%% == plant ==
prm.m   = 0.045;                                % kg
prm.g   = 9.81;
prm.J   = diag([2.3951e-5, 2.3951e-5, 3.2347e-6]);
prm.Jin = inv(prm.J);
prm.cd  = 0;                                    % linear drag (unmodelled)

Kt  = 3.72e-8;      % thrust coeff [N/(rad/s)^2]
Kc  = 7.74e-12;     % rotor-drag torque coeff (yaw)
arm = 0.0353;       % m

% mixer:  [T; tau] = Alloc * Omega.^2
prm.Alloc = [ Kt*[1 1 1 1];
              Kt*arm*[-1 -1  1  1];             % roll
              Kt*arm*[-1  1  1 -1];             % pitch
             -Kc*    [ 1 -1  1 -1]];            % yaw
prm.omega_min = 400;                            % rotor limits [rad/s]
prm.omega_max = 2800;

% external disturbances (kept for signature compatibility with quad_dynamics)
prm.F_ext   = [0;0;0];
prm.Tau_ext = [0;0;0];

%% == actuator limits ==
% Thrust and torque envelopes implied by the rotor limits, with a margin so
% the mixer is not permanently on a rail.
lim.T_min   = 0.30*prm.m*prm.g;
lim.T_max   = 1.80*prm.m*prm.g;
lim.tau_max = [4.0e-3; 4.0e-3; 2.0e-4];         % N*m

% Outer-loop command box, in the units SPC/NMPC/KDPC actually output
lim.T_spec = [0.5*prm.g, 1.6*prm.g];            % specific thrust [m/s^2]
lim.w_cmd  = 3.0;                               % body-rate command [rad/s]

%% == inner attitude-rate PID ==
% ~20 ms rate time constant (≈8 Hz closed bandwidth), comfortably faster than
% the 20 Hz outer loop -- that separation is what justifies the cascade.
inner.Kp_rate  = prm.J*[60; 60; 30];
inner.Ki_rate  = prm.J*[200; 200; 100];
inner.Kd_rate  = [0; 0; 0];
inner.i_max_att = 50;

%% == outer position PID, DATA GATHERING ONLY ==
% Used by helpers/outer_pid.m to fly the drone around while a dither is added
% on top. Not a baseline controller and not tuned for performance -- it only
% has to stay stable over the gathering envelope (+-1.2 m steps, ~48 deg peak
% tilt). Verified stable there.
prm.Kp_pos   = [3.0; 3.0; 6.0];
prm.Kd_pos   = [2.6; 2.6; 4.0];
prm.K_att    = 8.0;                             % attitude error -> body rate
prm.tilt_max = 35*pi/180;                       % clamp on the COMMANDED tilt
end
