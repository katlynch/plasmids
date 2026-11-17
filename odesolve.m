% plasmid simple Ode solve

% solves ODEs and plots trajectories for 
% (1) free/bound plasmid system
% (2) asymmetry ode system

% for both, uses initial condition drawn from one pole simulations as
% initial condition for ratio of bound/free plasmids


sv=0; % set =1 to save png of all graphs
svpath='/Users/katie/Desktop/plasmids/figs/';

params.Ntot=50;

% parameter values
params.beta=0.015;
params.kappa=0.08;
params.rho=0.13;

%params.gamma=1;
gvals=[ 0 2/3 1 2];
tend=250;
titles = {'$$F_1$$','$$B_1$$','$$F_2$$','$$B_2$$'};
titles2= {'$\gamma=0$','$\gamma=2/3$','$\gamma=1$','$\gamma=2$'};


for fg = 1:length(gvals)+1
    figure(fg)
    clf
    set(gca,'TickLabelInterpreter','latex','fontsize',16)
    box on
end


tgrid = linspace(0, tend, 5000);


clr={'[0.9 0 0]';'[0.1 0.1 0.9]';'[0.2 0.6 0.2]';'[0.9 0.7 0.1]'};

for i=1:length(gvals)
    params.gamma=gvals(i);
    [F0,B0]=sc_1pole_ssa(params);

    x0 = [F0; B0; 0; 0];
    y0 = [F0+B0,F0,F0];

    odemean = @(t,x) meanfieldode(t,x,params);
    [tode, xode] = ode45(odemean, tgrid, x0);

    figure(5)
    set(gca,'TickLabelInterpreter','latex','fontsize',16)

    for k=1:4
        subplot(2,2,k)
        box on
        hold on
        plot(tode,  xode(:,k),'Color',clr{i},'LineWidth', 1.5);
        xlabel('time','Interpreter','latex','fontsize',16)
        ylabel('plasmids','Interpreter','latex','fontsize',16)
        ylim([0,F0+B0])
        xlim([0,tend])
        title(titles{k},'Interpreter','latex','fontsize',16);
    end
    
    
    odeasym = @(t,y) asymmetryode(t,y,params);
    [tode, yode] = ode15s(odeasym, tgrid, y0);

    figure(i)
    hold on
    plot(tode,  yode(:,1),'Color',clr{i},'LineWidth', 2);
    plot(tode,  yode(:,2), '--', 'Color',clr{i},'LineWidth', 2);
    plot(tode,  yode(:,3), '-.','Color',clr{i},'LineWidth', 2);
    xlabel('time (min)','Interpreter','latex','fontsize',20)
    ylabel('plasmid copy no.','Interpreter','latex','fontsize',20)
    %ylim([0,params.Ntot])
    legend('$\Delta N$','$\Delta F','$F_{tot}$','Interpreter','latex','fontsize',16,'location','best')
    title(titles2{i},'Interpreter','latex','fontsize',20);

end

if sv==1

figure(5)
saveas(gcf,append(svpath,'2pole_sims_copyno'),'png')

figure(1)
saveas(gcf,append(svpath,'2pole_sims_g0'),'png')


figure(2)
saveas(gcf,append(svpath,'2pole_sims_g23'),'png')

figure(3)
saveas(gcf,append(svpath,'2pole_sims_g1'),'png')

figure(4)
saveas(gcf,append(svpath,'2pole_sims_g2'),'png')
end