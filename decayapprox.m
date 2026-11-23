% gamma = 0 approximation graphs

svpath='/Users/katie/Desktop/';


clr={'[0.9 0 0]';'[0.1 0.1 0.9]';'[0.2 0.6 0.2]'};

p.beta=0.015;
p.kappa=0.08;
p.rho=0.13;

p.Ntot=42;

scale=p.beta*p.Ntot;

tend=200;
tgrid = linspace(0, tend, 1000);


%% gamma=0

% initial conditions 

p.gamma=0;
[F0,B0]=sc_1pole_ssa(p);
x0=[F0+B0,F0,F0];

ode1 = @(t,x) asymmetryode(t,x,p);
[t_exact_g0, x_exact_g0] = ode45(ode1, tgrid, x0);

% slow manifold, with ~FT*
% re dimensionalized
Ftotstar=(-p.beta - p.kappa + sqrt(2*p.Ntot*p.beta*p.kappa + p.beta^2 + 2*p.beta*p.kappa + p.kappa^2))/p.beta;
% ode2 = @(t,N) -2*p.rho*N*(p.kappa)/(p.kappa+p.beta*(1+Ftotstar));
% [t_slow_g0, x_slow_g0] = ode45(ode2, tgrid, 39);
R0 = 2*p.rho*(p.kappa)/(p.kappa+p.beta*(1+Ftotstar)+2*p.rho);
etest = 35*exp(-R0*tgrid);


figure(8)
clf
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
hold on
plot(t_exact_g0,x_exact_g0(:,1),'Color',clr{1},'linewidth',2)

plot(tgrid,etest,'--','Color','k','linewidth',2)

t_g0 = 1/R0;
fprintf('for gamma=0, tau = %f \n',t_g0)

plot([t_g0 t_g0], [0 13], '-','color','[0.7 0.7 0.7]', 'linewidth', 1.5)
plot([0 t_g0], [13 13], '--','color','[0.7 0.7 0.7]','linewidth', 1.5)
text(26,2,'$\tau$','interpreter','latex','fontsize',16)

plot([50,50], [0 6], '-','color','[0.7 0.7 0.7]', 'linewidth', 1.5)
plot([0 50], [6,6], '--','color','[0.7 0.7 0.7]', 'linewidth', 1.5)
text(50,2,'division','interpreter','latex','fontsize',16)

legend('exact','slow manifold','interpreter','latex','fontsize',16,'location','best')

xlim([0,100])

%yscale('log')

%legend('$\gamma=0$ exact','$\gamma=1$ exact','$\gamma=0$ approx','$\gamma=0$ approx',...
 %   'interpreter','latex','fontsize',16,'location','best')
xlabel('time (min)','interpreter','latex','fontsize',16)
ylabel('$\Delta N$','interpreter','latex','fontsize',16)

% saveas(gcf,append(svpath,'approx_comp_g0'),'epsc')
% saveas(gcf,append(svpath,'approx_comp_g0'),'fig')
saveas(gcf,append(svpath,'approx_comp_g0'),'png')



%% gamma = 1

p.gamma=1;
[F0,B0]=sc_1pole_ssa(p);
x0=[F0+B0,F0,F0];


ode1 = @(t,x) asymmetryode(t,x,p);
[t_exact_g1, x_exact_g1] = ode45(ode1, tgrid, x0);

R1=2*p.rho*4*p.kappa^2/((p.Ntot*p.beta + 2*p.kappa)*(p.Ntot*p.beta + 2*p.kappa + 4*p.rho));
etest1 = 33*exp(-tgrid*R1);

t_g1=1/R1;
fprintf('for gamma=1, t_1/2 = %d \n',t_g1)


figure(9)
clf
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on
hold on
plot(t_exact_g1,x_exact_g1(:,1),'Color',clr{3},'linewidth',2)
plot(tgrid,etest1,'--','Color','k','linewidth',2)

t_g1 = log(2)/R1;
plot([t_g1 t_g1], [0 16.5], '-', 'color','[0.7 0.7 0.7]', 'linewidth', 1.5)
plot([0 t_g1], [16.5 16.5], '--','color','[0.7 0.7 0.7]', 'linewidth', 1.5)
text(98,5,'$t_{1/2}$','interpreter','latex','fontsize',16)

plot([50,50], [0 25.6], '-','color','[0.7 0.7 0.7]', 'linewidth', 1.5)
plot([0 50], [25.7,25.6], '--','color','[0.7 0.7 0.7]', 'linewidth', 1.5)
text(25,5,'division','interpreter','latex','fontsize',16)


legend('exact','slow manifold','interpreter','latex','fontsize',16,'location','best')


ylim([0,p.Ntot])

xlabel('time (min)','interpreter','latex','fontsize',16)
ylabel('$\Delta N$','interpreter','latex','fontsize',16)

% saveas(gcf,append(svpath,'approx_comp_g1'),'epsc')
% saveas(gcf,append(svpath,'approx_comp_g1'),'fig')
saveas(gcf,append(svpath,'approx_comp_g1'),'png')


