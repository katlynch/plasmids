%% timescales

% plots effective timescale (t1/2 = ln(2)/lambda), where lambda determines
% the asymmetry decay rate after some initial transient behavior

% compares this across gamma and Ntot values


%% set up
clear all
tic


sv=0;
svpath='/Users/katie/Desktop/plasmids/';

% number of points to consider for each gamma/Ntot
M=10;
% Ntots
Nvals=round(linspace(10,100,M));
% gammas to consider
gvals=linspace(0,1,M);

tic
% constant parameter values
p.beta=0.015;
p.kappa=0.08;
p.rho=0.13;
p.gamma=0;
tend=500;
t_transient=50;
tgrid = linspace(0, tend, 5000);


% preallocation for rates
l_single_vals=zeros(M,M);
l_multi_vals=zeros(M,M);


%% collect timeseries across Ntot values
% and estimate decay rate
% suppress lsqcurve fit output display
options=optimoptions('lsqcurvefit','Display','off');

for j=1:M
    %set current value of lambda
    p.gamma=gvals(j);
for i=1:M
    % set current value of Ntot
    p.Ntot=Nvals(i);

    y0 = [p.Ntot,0,0];% ics
    
    % call and solve asymmetry ODES for current parameter set
    odeasym = @(t,y) asymmetryode(t,y,p);
    [tode, yode] = ode45(odeasym, tgrid, y0);
    
    % ics
    z0=zeros(p.Ntot*2,1);
    z0(p.Ntot)=1;

    % call and solve multi cluster ODEs
    odemc = @(t,y) multicluster(t,y,p);
    [tode2, zode] = ode23s(odemc, tgrid, z0);

    sizes = 1:p.Ntot;
    c1 = zode(:, 1:p.Ntot);        % population 1, all sizes, all times
    c2 = zode(:, p.Ntot+1:2*p.Ntot);    % population 2, all sizes, all times
    mass1 = c1 * sizes';       % (nt x N) * (N x 1)
    mass2 = c2 * sizes';
    multi_dN = mass1-mass2;
    
    
    % fit curve to find decay rate
    %fdecay=@(lambda,t) p.Ntot*exp(-lambda*t);
    % we will only fit data after the transient

    mask = tode >= t_transient;
    A0 = yode(find(mask, 1), 1);          % amplitude at t=10 (first post-transient point)
    tshift = tode(mask) - tode(find(mask, 1));  % shift time so fit starts at 0
    fdecay = @(lambda, t) A0 * exp(-lambda * t);
    l_single_vals(j,i) = lsqcurvefit(fdecay, 1, tshift, yode(mask, 1),[],[],options);

    % repeat for multicluster
    mask = tode2 >= t_transient;
    B0 = zode(find(mask, 1), 1);          % amplitude at t=10 (first post-transient point)
    tshift = tode2(mask) - tode2(find(mask, 1));  % shift time so fit starts at 0
    gdecay = @(lambda, t) B0 * exp(-lambda * t);
    l_multi_vals(j,i) = lsqcurvefit(gdecay, 1, tshift, multi_dN(mask,1),[],[],options);
    

end
end

slvals=real(l_single_vals);
single_t12_vals=log(2)./slvals;

mlvals=real(l_multi_vals);
multi_t12_vals=log(2)./mlvals;




%% plot exponential decay rates (from slow manifold)
figure(11)
clf
subplot(1,3,1)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
imagesc(gvals,Nvals, slvals);
ylabel('$\log(N_{tot})$','interpreter','latex','fontsize',16); 
xlabel('$\gamma$','interpreter','latex','fontsize',16);
title('$\lambda$ -- Data','interpreter','latex','fontsize',16); axis xy;
%yscale('log')
c = colorbar('location','EastOutside');
c.TickLabelInterpreter = 'latex';
c.Label.String='asymmetry decay rate';
c.Label.Interpreter ='latex';
c.Label.FontSize = 16;


subplot(1,3,2)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
hold on
c1map = parula(M);  % or try 'jet', 'turbo', 'cool', etc.
for k = 1:M
    plot(Nvals, slvals(k,:), 'linewidth', 2, 'Color', c1map(k,:))
end
xlabel('$N_{tot}$','interpreter','latex','fontsize',16)
ylabel('$\lambda$ (decay rate)','interpreter','latex','fontsize',16)
% Add colorbar
colormap(c1map)
cb1 = colorbar;
cb1.Label.String = '$\gamma$';
cb1.Label.Interpreter = 'latex';
cb1.Label.FontSize = 16;

% Set colorbar ticks to correspond to actual Nvals
clim([min(gvals), max(gvals)]);
cb1.Ticks = linspace(min(gvals), max(gvals), M);
cb1.TickLabels = arrayfun(@num2str, gvals, 'UniformOutput', false);
cb1.TickLabelInterpreter = 'Latex';

xscale('log')
yscale('log')

subplot(1,3,3)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
hold on

% Define colormap for the curves
c2map = parula(M);  % or try 'jet', 'turbo', 'cool', etc.

for k = 1:M
    plot(gvals, slvals(:,k), 'linewidth', 2, 'Color', c2map(k,:))
end

xlabel('$\gamma$','interpreter','latex','fontsize',20)
%ylabel('$\lambda$ (decay rate)','interpreter','latex','fontsize',20)


% Add colorbar
colormap(c2map)
cb2 = colorbar;
cb2.Label.String = '$N_{tot}$';
cb2.Label.Interpreter = 'latex';
cb2.Label.FontSize = 16;

% Set colorbar ticks to correspond to actual Nvals
clim([min(Nvals), max(Nvals)]);
cb2.Ticks = linspace(min(Nvals), max(Nvals), M);
cb2.TickLabels = arrayfun(@num2str, Nvals, 'UniformOutput', false);
cb2.TickLabelInterpreter = 'Latex';

sgtitle('exponential decay rate (slow manifold): single cluster')



% repeat for multi cluster
figure(21)
clf
subplot(1,3,1)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
imagesc(gvals,Nvals, mlvals);
ylabel('$\log(N_{tot})$','interpreter','latex','fontsize',16); 
xlabel('$\gamma$','interpreter','latex','fontsize',16);
title('$\lambda$ -- Data','interpreter','latex','fontsize',16); axis xy;
%yscale('log')
c = colorbar('location','EastOutside');
c.TickLabelInterpreter = 'latex';
c.Label.String='asymmetry decay rate';
c.Label.Interpreter ='latex';
c.Label.FontSize = 16;


subplot(1,3,2)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
hold on
c1map = parula(M);  % or try 'jet', 'turbo', 'cool', etc.
for k = 1:M
    plot(Nvals, mlvals(k,:), 'linewidth', 2, 'Color', c1map(k,:))
end
xlabel('$N_{tot}$','interpreter','latex','fontsize',16)
ylabel('$\lambda$ (decay rate)','interpreter','latex','fontsize',16)
% Add colorbar
colormap(c1map)
cb1 = colorbar;
cb1.Label.String = '$\gamma$';
cb1.Label.Interpreter = 'latex';
cb1.Label.FontSize = 16;

% Set colorbar ticks to correspond to actual Nvals
clim([min(gvals), max(gvals)]);
cb1.Ticks = linspace(min(gvals), max(gvals), M);
cb1.TickLabels = arrayfun(@num2str, gvals, 'UniformOutput', false);
cb1.TickLabelInterpreter = 'Latex';

yscale('log')

subplot(1,3,3)
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
hold on

% Define colormap for the curves
c2map = parula(M);  % or try 'jet', 'turbo', 'cool', etc.

for k = 1:M
    plot(gvals, mlvals(:,k), 'linewidth', 2, 'Color', c2map(k,:))
end

xlabel('$\gamma$','interpreter','latex','fontsize',20)
%ylabel('$\lambda$ (decay rate)','interpreter','latex','fontsize',20)


% Add colorbar
colormap(c2map)
cb2 = colorbar;
cb2.Label.String = 'N';
cb2.Label.Interpreter = 'latex';
cb2.Label.FontSize = 16;

% Set colorbar ticks to correspond to actual Nvals
clim([min(Nvals), max(Nvals)]);
cb2.Ticks = linspace(min(Nvals), max(Nvals), M);
cb2.TickLabels = arrayfun(@num2str, Nvals, 'UniformOutput', false);
cb2.TickLabelInterpreter = 'Latex';
yscale('log')
sgtitle('exponential decay rate (slow manifold): multi cluster')

return
%% now repeat plots for timescale (t1/2)


if sv==0
figure(12)
clf
subplot(1,3,1)
elseif sv==1
figure(12)
clf
end
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
imagesc(gvals,Nvals, log(t12_vals));
set(gca, 'YDir', 'normal')
ylabel('$N_{tot}$','interpreter','latex','fontsize',16); 
xlabel('$\gamma$','interpreter','latex','fontsize',16);
%yscale('log')
c = colorbar('location','EastOutside');
c.TickLabelInterpreter = 'latex';
c.Label.String='$\log(t_{1/2})$';
c.Label.Interpreter ='latex';
c.Label.FontSize = 9;

if sv==0
figure(12)
clf
subplot(1,3,2)
elseif sv==1
figure(13)
clf
end
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
hold on
c1map = parula(M);  % or try 'jet', 'turbo', 'cool', etc.
for k = 1:M
    plot(Nvals, t12_vals(k,:), 'linewidth', 2, 'Color', c1map(k,:))
end
xlabel('$N_{tot}$','interpreter','latex','fontsize',16)
ylabel('$t_{1/2}$','interpreter','latex','fontsize',16)
xscale('log')
yscale('log')
% Add colorbar
colormap(c1map)
cb1 = colorbar;
cb1.Label.String = '$\gamma$';
cb1.Label.Interpreter = 'latex';
cb1.Label.FontSize = 9;
% Set colorbar ticks to correspond to actual Nvals
clim([min(gvals), max(gvals)]);
cb1.Ticks = linspace(min(gvals), max(gvals), M);
cb1.TickLabels = arrayfun(@num2str, t12_vals, 'UniformOutput', false);
cb1.TickLabelInterpreter = 'Latex';







if sv==0
figure(12)
clf
subplot(1,3,3)
elseif sv==1
figure(14)
clf
end
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
hold on
% Define colormap for the curves
c2map = parula(M);  % or try 'jet', 'turbo', 'cool', etc.
for k = 1:M
    plot(gvals, t12_vals(:,k), 'linewidth', 2, 'Color', c2map(k,:))
end
xlabel('$\gamma$','interpreter','latex','fontsize',20)
xscale('log')
yscale('log')

% Add colorbar
colormap(c2map)
cb2 = colorbar;
cb2.Label.String = '$N_{tot}$';
cb2.Label.Interpreter = 'latex';
cb2.Label.FontSize = 9;
% Set colorbar ticks to correspond to actual Nvals
clim([min(Nvals), max(Nvals)]);
cb2.Ticks = linspace(min(Nvals), max(Nvals), M);
cb2.TickLabels = arrayfun(@num2str, Nvals, 'UniformOutput', false);
cb2.TickLabelInterpreter = 'Latex';


if sv==0
    sgtitle('timescale (half decay on slow manifold)')
elseif sv==1

figure(12)
saveas(gcf,append(svpath,'t12_heat'),'png')

figure(13)
saveas(gcf,append(svpath,'t12_N'),'png')

figure(14)
ylabel('$t_{1/2}$','interpreter','latex','fontsize',16)
saveas(gcf,append(svpath,'t12_g'),'png')

end

disp('finished finding decay rates \n')

