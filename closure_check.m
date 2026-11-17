% closure_check.m
%
% Checks the validity of the mean-field moment closure for the
% two-compartment free/bound (F <-> B) reaction-transport model by
% comparing an SSA (Gillespie) ensemble against the mean-field ODE
% system, and by directly measuring the size of the Var/Cov terms that
% the closure drops.
%
% Exact (unclosed) moment equations (from the CME, derived by summing
% k*[gain-loss] over states for each species k):
%
%   d<F1>/dt = -beta*<F1(F1-1)> - beta*<F1*B1^gamma> + kappa*<B1> + rho*(<F2>-<F1>)
%   d<B1>/dt =  beta*<F1(F1-1)> + beta*<F1*B1^gamma> - kappa*<B1>
%   d<F2>/dt = -beta*<F2(F2-1)> - beta*<F2*B2^gamma> + kappa*<B2> + rho*(<F1>-<F2>)
%   d<B2>/dt =  beta*<F2(F2-1)> + beta*<F2*B2^gamma> - kappa*<B2>
%
% Mean-field closure sets:
%   <Fj(Fj-1)>       ~ <Fj>^2 - <Fj>                 
%   <Fj*Bj^gamma>    ~ <Fj>*<Bj>^gamma
% i.e. drop covariance and variance from exact eqn
% for both j=1,2

clear; clc; close all;

%% 0 %% initialization
%  ---- Parameters (must match those hard-coded inside gillespie.m) ----
params.beta=0.015;
params.kappa=0.08;
params.rho=0.13;

svpath='/Users/katie/Desktop/plasmids/figs/';
% ---- Initial condition and ensemble size ----
Nplasmids=42;

F0    = 5;    % initial free molecules, compartment 1
B0    = Nplasmids-F0;     % initial bound molecules, compartment 1
Ntraj = 1e2;  % number of independent SSA realizations
N0 = [F0 B0 0 0];
gvals=[0 2/3 1];

%% 1 %% Run SSA ensemble

for i=1:4
tic
    params.gamma=gvals(i);

traj = cell(Ntraj,1);
for n = 1:Ntraj
    [f1,b1,f2,b2,tt] = copyno_ssa(N0,params);
    traj{n}.t  = tt;
    traj{n}.f1 = f1;
    traj{n}.b1 = b1;
    traj{n}.f2 = f2;
    traj{n}.b2 = b2;
end

%% ---- Put every trajectory on a common time grid ----
% Each SSA run stops after a fixed number of reaction EVENTS (maxitr),
% so different runs end at different physical times. Use the shortest
% final time across the ensemble so every grid point has full coverage.
Tend  = min(cellfun(@(s) s.t(end), traj));
Ngrid = 400;
tgrid = linspace(0, Tend, Ngrid);

F1s = zeros(Ntraj,Ngrid); B1s = zeros(Ntraj,Ngrid);
F2s = zeros(Ntraj,Ngrid); B2s = zeros(Ntraj,Ngrid);

for n = 1:Ntraj
    s = traj{n};
    % state is piecewise-constant between reaction events -> 'previous'
    F1s(n,:) = interp1(s.t, s.f1, tgrid, 'previous');
    B1s(n,:) = interp1(s.t, s.b1, tgrid, 'previous');
    F2s(n,:) = interp1(s.t, s.f2, tgrid, 'previous');
    B2s(n,:) = interp1(s.t, s.b2, tgrid, 'previous');
end
toc
%% ---- Empirical (SSA) moments at every grid time ----
mF1 = mean(F1s,1); mB1 = mean(B1s,1);
mF2 = mean(F2s,1); mB2 = mean(B2s,1);

stdF1 = std(F1s,1); stdB1 = std(B1s,1);
stdF2 = std(F2s,1); stdB2 = std(B2s,1);

varF1 = var(F1s,0,1);
varF2 = var(F2s,0,1);

% Cov(B^gamma, F) computed directly from the ensemble at each grid time
covB1gF1 = zeros(1,Ngrid);
covB2gF2 = zeros(1,Ngrid);
for k = 1:Ngrid
    C = cov(B1s(:,k).^params.gamma, F1s(:,k)); covB1gF1(k) = C(1,2);
    C = cov(B2s(:,k).^params.gamma, F2s(:,k)); covB2gF2(k) = C(1,2);
end

%% ---- Mean-field ODE (Var/Cov terms dropped) ----
x0 = [F0; B0; 0; 0];
odefun = @(t,x) meanfieldode(t,x,params);
[tode, xode] = ode45(odefun, tgrid, x0);

%% ---- Plot 1: SSA ensemble mean vs mean-field ODE ----
figure(1);

labels  = {'F_1','B_1','F_2','B_2'};
titles = {'$$F_1$$','$$B_1$$','$$F_2$$','$$B_2$$'};
simdata = {mF1,mB1,mF2,mB2};
stddata = {stdF1,stdB1,stdF2,stdB2};
clrstr={'[0.9 0 0]';'[0.1 0.1 0.9]';'[0.2 0.6 0.2]';'[0.9 0.7 0.1]'};
clrvec=[0.9 0 0; 0.1 0.1 0.9; 0.2 0.6 0.2; 0.9 0.7 0.1];
for k = 1:4
    subplot(2,2,k); 
    hold on;
    set(gca,'TickLabelInterpreter','latex')
    fontsize(16,'points')
    box on
    plot(tgrid, simdata{k}, '-', 'Color',clrstr{i},  'LineWidth', 1.5);
    plot(tode,  xode(:,k),  '--','Color','k', 'LineWidth', 2);
    
    std1 = simdata{k}+stddata{k};
    std2 = simdata{k}-stddata{k};
    x2 = [tgrid, fliplr(tgrid)];
    inBetween = [std1, fliplr(std2),];
    %fstd=fill(x2, inBetween, clrvec(i,:));
    %fstd.LineWidth=0.1;
    %fstd.FaceAlpha=0.2;


    %legend('SSA ensemble mean','mean-field ODE','std','Location','best','Interpreter','latex','fontsize',16);
    xlim([0,500])
    ylim([0,35])
    title(titles{k},'Interpreter','latex','fontsize',16);
end

xlabel('time (min)','Interpreter','latex','fontsize',16);
    
ylabel(['$$\langle ' labels{k} ' \rangle$$'],'Interpreter','latex','fontsize',16);



%% ---- Plot 2 & 3: magnitude of the dropped Var/Cov terms ----
leading_1 = params.beta*(mF1.^2) + params.beta*(mF1.*mB1.^params.gamma);   % mean-field part
dropped_1 = params.beta*(params.beta*mF1+varF1 + covB1gF1);                        % what closure drops

leading_2 = params.beta*(mF2.^2) + params.beta*(mF2.*mB2.^params.gamma);
dropped_2 = params.beta*(params.beta*mF2+varF2 + covB2gF2);

saveas(gcf,append(svpath,'mfa_means'),'epsc')
saveas(gcf,append(svpath,'mfa_means'),'fig')
saveas(gcf,append(svpath,'mfa_means'),'png')

figure(2);

subplot(1,2,1); hold on;
set(gca,'TickLabelInterpreter','latex')
fontsize(16,'points')
box on
plot(tgrid, leading_1, 'Color',clrstr{i}, 'LineWidth', 1.5);
plot(tgrid, dropped_1, '--','Color',clrstr{i}, 'LineWidth', 1.5);
xlabel('time (min)','Interpreter','latex','fontsize',16); 
ylabel('rate contribution','Interpreter','latex','fontsize',16);
xlim([0,500])
legend('leading (mean-field) term','dropped term','Location','best','Interpreter','latex','fontsize',16);
title('Compartment 1','Interpreter','latex','fontsize',16);

subplot(1,2,2); hold on;
set(gca,'TickLabelInterpreter','latex')
fontsize(16,'points')
box on
plot(tgrid, leading_2, 'Color',clrstr{i}, 'LineWidth', 1.5);
plot(tgrid, dropped_2, '--', 'Color',clrstr{i}, 'LineWidth', 1.5);
xlabel('time (min)','Interpreter','latex','fontsize',16); 
ylabel('rate contribution','Interpreter','latex','fontsize',16);
xlim([0,500])
%legend('leading (mean-field) term','dropped term','Location','best','Interpreter','latex','fontsize',16);
title('Compartment 2','Interpreter','latex','fontsize',16);

saveas(gcf,append(svpath,'mfa_terms'),'epsc')
saveas(gcf,append(svpath,'mfa_terms'),'fig')
saveas(gcf,append(svpath,'mfa_terms'),'png')


%% ---- Report a single summary number ----
rel_err_1 = abs(dropped_1) ./ (abs(leading_1)+abs(dropped_1));
rel_err_2 = abs(dropped_2) ./ (abs(leading_2)+abs(dropped_2));

figure(3)
subplot(1,2,1); hold on;
set(gca,'TickLabelInterpreter','latex')
fontsize(16,'points')
box on
plot(tgrid,rel_err_1,'linewidth',1.5,'Color',clrstr{i})
xlabel('time (min)','Interpreter','latex','fontsize',16); 
ylabel('relative error','Interpreter','latex','fontsize',16)
title('Compartment 1','Interpreter','latex','fontsize',16);
ylim([0,0.5])
xlim([0,500])
subplot(1,2,2); hold on;
set(gca,'TickLabelInterpreter','latex')
fontsize(16,'points')
box on
plot(tgrid,rel_err_2,'linewidth',1.5,'Color',clrstr{i})
xlabel('time (min)','Interpreter','latex','fontsize',16); 
ylabel('relative error','Interpreter','latex','fontsize',16)
ylim([0,0.5])
xlim([0,500])
%legend('compartment 1','compartment 2','Location','best','Interpreter','latex','fontsize',16);
title('Compartment 2','Interpreter','latex','fontsize',16);

legend('$\gamma=0$','$\gamma=2/3$','$\gamma=1$','fontsize',16,'interpreter','latex','location','best')
saveas(gcf,append(svpath,'mfa_relerr'),'epsc')
saveas(gcf,append(svpath,'mfa_relerr'),'fig')
saveas(gcf,append(svpath,'mfa_relerr'),'png')

%fprintf('Compartment 1: max |dropped/leading| over time = %.2f%%\n', 100*max(rel_err_1));
%fprintf('Compartment 2: max |dropped/leading| over time = %.2f%%\n', 100*max(rel_err_2));
%fprintf(['If this is small (a few %%) across the whole trajectory, dropping\n', ...
 %        'the Var/Cov terms (mean-field closure) is well justified for these\n', ...
  %       'parameters/initial conditions. If it is large, or grows over time,\n', ...
   %      'the full Gaussian closure (or SSA itself) should be used instead.\n']);
toc


end






%% ---- Local functions ---- 

% function dxdt = meanfieldode(~,x,params)
% % mean-field RHS / ODE
% f1 = x(1); b1 = x(2); f2 = x(3); b2 = x(4);
% dxdt = zeros(4,1);
% dxdt(1) =  -params.beta*(f1^2 ) - params.beta*f1*b1^params.gamma + params.kappa*b1 + params.rho*(f2-f1);
% dxdt(2) =  params.beta*(f1^2 ) + params.beta*f1*b1^params.gamma - params.kappa*b1;
% dxdt(3) =  -params.beta*(f2^2 ) - params.beta*f2*b2^params.gamma + params.kappa*b2 + params.rho*(f1-f2);
% dxdt(4) =   params.beta*(f2^2 ) + params.beta*f2*b2^params.gamma - params.kappa*b2;
% end

