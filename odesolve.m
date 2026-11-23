% plasmid simple Ode solve

% solves ODEs and plots trajectories for 
% (1) free/bound plasmid system
% (2) asymmetry ode system

% for both, uses initial condition drawn from one pole simulations as
% initial condition for ratio of bound/free plasmids


sv=0; % set =1 to save png of all graphs
svpath='/Users/katie/Desktop/plasmids/figs/';

params.Ntot=200;

% parameter values
params.beta=0.015;
params.kappa=0.08;
params.rho=0.13;

%params.gamma=1;
gvals=1;%[ 0 2/3 1 2];
tend=5000;
titles = {'$$F_1$$','$$B_1$$','$$F_2$$','$$B_2$$'};
titles2= {'$\gamma=0$','$\gamma=2/3$','$\gamma=1$','$\gamma=2$'};


for fg = 1:length(gvals)+1
    figure(fg)
    clf
    set(gca,'TickLabelInterpreter','latex','fontsize',16)
    box on
end

z0=zeros(params.Ntot*2,1);
z0(params.Ntot)=1;


tgrid = linspace(0, tend, 5000);


clr={'[0.9 0 0]';'[0.1 0.1 0.9]';'[0.2 0.6 0.2]';'[0.9 0.7 0.1]'};

for i=1:length(gvals)
    params.gamma=gvals(i);
    [F0,B0]=sc_1pole_ssa(params);

    x0 = [F0; B0; 0; 0];
    y0 = [F0+B0,F0,F0];
    
    % first solves the meanfield ODEs/ i.e., looks at free and bound
    odemean = @(t,x) meanfieldode(t,x,params);
    [tode, xode] = ode45(odemean, tgrid, x0);

    figure(5)
    set(gca,'TickLabelInterpreter','latex','fontsize',16)
    % generates a single plot containing f1, f2, b1, b2 
    % all gammas compared
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

    % solves asymmetry odes
    odeasym = @(t,y) asymmetryode(t,y,params);
    [tode, yode] = ode45(odeasym, tgrid, y0);
    
    % plot asymmetry ode solutions (each gamma on separate plot)
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


    % solves multi cluster ODEs
    odemc = @(t,y) multicluster(t,y,params);
    [tode, zode] = ode45(odemc, tgrid, z0);

    sizes = 1:params.Ntot;
    c1 = zode(:, 1:params.Ntot);        % population 1, all sizes, all times
    c2 = zode(:, params.Ntot+1:2*params.Ntot);    % population 2, all sizes, all times
    mass1 = c1 * sizes';       % (nt x N) * (N x 1)
    mass2 = c2 * sizes';
    multi_dN = mass1-mass2;

    figure(10)
    clf
    hold on
    plot(tode,mass1,'linewidth',2)
    plot(tode,mass2,'linewidth',2)
   
     
    figure(i+length(gvals))
    clf
    hold on
    plot(tode,  yode(:,1),'--','Color',clr{i},'LineWidth', 2);
    plot(tode,  multi_dN,'Color',clr{i},'LineWidth', 2);
    xlabel('time (min)','Interpreter','latex','fontsize',20)
    ylabel('$\Delta N$ [plasmid copy no.]','Interpreter','latex','fontsize',20)
    legend('net cluster size','multicluster','Interpreter','latex','fontsize',16,'location','best')
    ylim([0,params.Ntot])
    %yscale('log')
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