close all; clear all; clc;

%% Identification data from the IDSIA nano-drone benchmark
[prm, lim, inner] = quad_params();

decimate = 5;                    % 100 -> 20 Hz, matching Generate_data.m
Ts = 0.01 * decimate;

train_files = dir(fullfile('data','train','*.csv'));
test_files  = dir(fullfile('data','test','*.csv'));
assert(~isempty(train_files), 'no CSVs in data/train -- run from the JCVW_QUAD_ folder');

[u_train, y_train, run_lengths]      = load_runs(train_files, decimate, prm);
[u_test,  y_test,  run_lengths_test] = load_runs(test_files,  decimate, prm);

u_mean = mean(u_train);
y_mean = mean(y_train);

u_data      = u_train - u_mean;
y_data      = y_train - y_mean;
u_data_test = u_test  - u_mean;
y_data_test = y_test  - y_mean;

t_train = (0:size(y_data,1)-1)*Ts;

%% ------------------------------------------------------------------------
curr_fig = figure;
curr_axes1=axes('Parent',curr_fig,'FontSize',11,'FontName','Times New Roman');
box(curr_axes1,'on');
hold(curr_axes1,'all');
subplot(2,1,1)
hold on
plot(t_train,u_train,'LineWidth',1)
ylabel('$u(k)$',Interpreter='latex')
legend('T/m','\omega_x','\omega_y','\omega_z')
axis tight
subplot(2,1,2)
hold on
plot(t_train,y_train(:,1:3),'LineWidth',1)
ylabel('$y(k)$',Interpreter='latex')
xlabel('$t$[s]',Interpreter='latex')
legend('x','y','z')
axis tight
set(gca,'TickLabelInterpreter','Latex')
set(curr_fig,'Units','centimeters','PaperSize',[20.98 29.68],'PaperUnits','centimeters','PaperPosition',[0 0 12 8])
savefig('figures/identification_real.fig')

%% ------------------------------------------------------------------------
save('data/real_u_data',"u_data")
save('data/real_y_data',"y_data")
save('data/real_u_data_test',"u_data_test")
save('data/real_y_data_test',"y_data_test")
save('data/real_run_lengths',"run_lengths","run_lengths_test")

source  = 'real';
p_hover = mean(y_train(:,1:3)).';
save('data/real_scalers',"u_mean","y_mean","Ts","source","p_hover")

%% ------------------------------------------------------------------------
fprintf('loaded %d train / %d test runs at %.0f Hz (%d / %d samples)\n', ...
        numel(run_lengths), numel(run_lengths_test), 1/Ts, ...
        size(y_train,1), size(y_test,1));

spd = vecnorm(diff(y_train(:,1:3))/Ts, 2, 2);
fprintf('  speed       median %.2f m/s, max %.2f m/s\n', median(spd), max(spd));
fprintf('  T/m         [%.2f, %.2f] m/s^2   (SPC box %.2f .. %.2f)\n', ...
        min(u_train(:,1)), max(u_train(:,1)), lim.T_spec(1), lim.T_spec(2));
fprintf('  |omega|     max %.2f rad/s        (SPC box %.1f)\n', ...
        max(max(abs(u_train(:,2:4)))), lim.w_cmd);
outside = mean(u_train(:,1) < lim.T_spec(1) | u_train(:,1) > lim.T_spec(2) | ...
               any(abs(u_train(:,2:4)) > lim.w_cmd, 2));
fprintf('  %.1f%% of samples sit outside the SPC input box (training on a wider\n', 100*outside);
fprintf('     envelope than the controller may command is fine, not a fault)\n');

% EXCITATION -- the number that decides whether Gamma is identifiable at all:
% how much of u(k) is already a deterministic continuation of the past. There
% is no dither, so this comes entirely from the inner loop's bandwidth.
% Near 1.0 would mean least squares has nothing left to attribute to u(k).
H = 6;
nrow = sum(max(0, run_lengths - H));
Xr = zeros(nrow, H*(size(u_data,2)+size(y_data,2)));  Tg = zeros(nrow,size(u_data,2));
st = 0;  row = 0;
for rr = 1:numel(run_lengths)
    L = run_lengths(rr);  uu = u_data(st+(1:L),:);  yy = y_data(st+(1:L),:);
    for i = H+1:L
        row = row+1;
        Xr(row,:) = [reshape(uu(i-H:i-1,:).',1,[]), reshape(yy(i-H:i-1,:).',1,[])];
        Tg(row,:) = uu(i,:);
    end
    st = st + L;
end
Xr = [Xr(1:row,:) ones(row,1)];  Tg = Tg(1:row,:);
Rr = Tg - Xr*(Xr\Tg);
fprintf('  R2(u(k) <- past)  %s\n', ...
        mat2str(round(1 - sum(Rr.^2)./sum((Tg-mean(Tg)).^2), 3)));
fprintf('     ^ 1.0 would mean Gamma is unidentifiable. The wrench form gave 0.9997.\n');
fprintf('\nsaved data/real_*.mat. Next: SPC_real.m\n');

%% ------------------------------------------------------------------------
function [U, Y, lens] = load_runs(files, dec, prm)
U = []; Y = []; lens = [];
for k = 1:numel(files)
    T = readtable(fullfile(files(k).folder, files(k).name));

    % INPUT: motor speeds -> wrench (eq. 5 = prm.Alloc), thrust as SPECIFIC
    % thrust so u matches Generate_data.m; body rate stands in for w_ref.
    Om2 = [T.m1_rads, T.m2_rads, T.m3_rads, T.m4_rads].^2;
    w   = (prm.Alloc * Om2.').';
    u   = [w(:,1)/prm.m, T.wx, T.wy, T.wz];

    % OUTPUT: the flat outputs, as Generate_data.m. Quaternion is scalar-LAST.
    yaw = unwrap(atan2(2*(T.qw.*T.qz + T.qx.*T.qy), 1 - 2*(T.qy.^2 + T.qz.^2)));
    y   = [T.x, T.y, T.z, yaw];

    if dec > 1     % anti-alias before decimating; same filter on u and y
        u = movmean(u, dec, 1);  y = movmean(y, dec, 1);
        u = u(1:dec:end, :);     y = y(1:dec:end, :);
    end

    U = [U; u];  Y = [Y; y];  lens = [lens; size(y,1)];
end
lens = double(lens);
end
