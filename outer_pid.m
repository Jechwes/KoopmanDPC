function v_cmd = outer_pid(x, p_des, v_des, yaw_des, prm)
% OUTER_PID  Position controller -> [T_spec; wx_ref; wy_ref; wz_ref]
%
%   v_cmd = OUTER_PID(x, p_des, v_des, yaw_des, prm)
%
%   only used for data gathering to have a cascaded PID as controller

%% load data
p = x(1:3);  
v = x(4:6);
R = quat2rot_local(x(7:10));

%% cascaded PID loop
% 1. desired acceleration in the world frame
a_des = prm.Kp_pos(:).*(p_des(:) - p) + prm.Kd_pos(:).*(v_des(:) - v) + [0;0;prm.g];
a_des(3) = max(a_des(3), 0.3*prm.g);            % never command negative lift

% 2. clamp commanded tilt
hor  = norm(a_des(1:2));
hmax = a_des(3)*tan(prm.tilt_max);
if hor > hmax && hor > 1e-9
    a_des(1:2) = a_des(1:2) * (hmax/hor);
end

% 3. specific thrust: project on the current body z, so the drone does not
%    over-thrust while it is still rotating towards the desired attitude
T = a_des.' * R(:,3);

% 4. desired attitude
z_des = a_des / norm(a_des);
x_c   = [cos(yaw_des); sin(yaw_des); 0];
y_des = cross(z_des, x_c);
ny_   = norm(y_des);
if ny_ < 1e-6                                    % degenerate: pointing along x_c
    y_des = [0;1;0];
else
    y_des = y_des / ny_;
end
R_des = [cross(y_des, z_des), y_des, z_des];

% 5. attitude error -> body-rate command
M  = R_des.'*R - R.'*R_des;
eR = 0.5*[M(3,2); M(1,3); M(2,1)];               % vee(.)
v_cmd = [T; -prm.K_att*eR];
end

%% helpers
function R = quat2rot_local(q)
qw=q(1); qx=q(2); qy=q(3); qz=q(4);
R = [1-2*(qy^2+qz^2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw);
     2*(qx*qy+qz*qw),  1-2*(qx^2+qz^2),  2*(qy*qz-qx*qw);
     2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx^2+qy^2)];
end
