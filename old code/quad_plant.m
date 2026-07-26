function x_next = quad_plant(x, Omega, prm, Ts)
% One Ts step of the grey-box quadrotor plant (see calibrate_plant.m).
% State x (13x1): [p(3); v(3); q(4, qw first); w(3, body rates)]
% Omega (4x1): motor speeds [rad/s]
% RK4 with substeps; quaternion renormalized each substep.
nsub = 2;  h = Ts/nsub;
for s = 1:nsub
    k1 = f(x,          Omega, prm);
    k2 = f(x + h/2*k1, Omega, prm);
    k3 = f(x + h/2*k2, Omega, prm);
    k4 = f(x + h*k3,   Omega, prm);
    x  = x + h/6*(k1 + 2*k2 + 2*k3 + k4);
    x(7:10) = x(7:10)/norm(x(7:10));
end
x_next = x;
end

function dx = f(x, Omega, prm)
v = x(4:6);  q = x(7:10);  w = x(11:13);
Rm = quat2rot(q);
a  = Rm*[0; 0; prm.c_t*sum(Omega.^2)] + [0; 0; -prm.g];  % thrust + gravity
wd = prm.w_bias + prm.Mw*(Omega.^2);                     % identified torque map
qd = 0.5*quatmul(q, [0; w]);
dx = [v; a; qd; wd];
end

function R = quat2rot(q)
qw=q(1); qx=q(2); qy=q(3); qz=q(4);
R = [1-2*(qy^2+qz^2),  2*(qx*qy-qz*qw),  2*(qx*qz+qy*qw);
     2*(qx*qy+qz*qw),  1-2*(qx^2+qz^2),  2*(qy*qz-qx*qw);
     2*(qx*qz-qy*qw),  2*(qy*qz+qx*qw),  1-2*(qx^2+qy^2)];
end

function r = quatmul(q, p)
r = [q(1)*p(1) - q(2:4)'*p(2:4);
     q(1)*p(2:4) + p(1)*q(2:4) + cross(q(2:4), p(2:4))];
end
