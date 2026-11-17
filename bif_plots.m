% has code for plotting bifurcation related things
% currently: the gamma=0 cusp and labels to the 1 parameter bif diagram (in
% del F) from xpp and post the usual converter

svpath='/Users/katie/Desktop/plasmids/figs/';

% Parameters
beta  = 0.015;
kappa = 0.08;
rho   = 0.13;





%% gamma = 0 fast system saddle node

close all

openfig(append(svpath,'fast_g0bif.fig'));

box on
grid on
set(gca,'TickLabelInterpreter','latex')
fontsize(16,"points")
xlabel('$\Delta N$','Interpreter','Latex','FontSize',16)
ylabel('$\Delta F$','Interpreter','Latex','FontSize',16)
leg=legend();
leg.FontSize=16;
leg.Location='best';
leg.Interpreter='Latex';
xlim([-55,55])
ylim([-55,55])

% saveas(gcf,append(svpath,'fast_g0bif_label'),'epsc')
% saveas(gcf,append(svpath,'fast_g0bif_label'),'fig')
saveas(gcf,append(svpath,'fast_g0bif_label'),'png')


% Grid
N_vec    = linspace(-100, 100, 1000);
Ntot_vec = linspace(0,  100, 1000);
[N, Ntot] = meshgrid(N_vec, Ntot_vec);

% Expression R2
R2=(2*N*beta*kappa + 2*Ntot*beta*kappa + beta^2 + 2*beta*kappa + kappa^2).*(2*N*beta*kappa - 2*Ntot*beta*kappa - beta^2 - 2*beta*kappa - kappa^2);

% plotting
figure(17);
clf
box on
contour(N_vec, Ntot_vec, R2, [0 0], 'r', 'LineWidth', 1.5);
axis tight;
hold on
plot(N_vec,N_vec,'--','linewidth',1.5,'Color','k')
ylim([0,100])
set(gca,'TickLabelInterpreter','latex','fontsize',16)
xlabel('$\Delta N$','fontsize',16,'interpreter','latex');
ylabel('$N_{tot}$','fontsize',16,'interpreter','latex');
grid on;
text(-20,50,'slow manifold exists','interpreter','latex','fontsize',16)
text(35,10,'no solution','interpreter','latex','fontsize',16)
text(-65,10,'no solution','interpreter','latex','fontsize',16)


legend('limit points','$\Delta N=N_{tot}$','fontsize',16,'interpreter','latex','location','north')
% saveas(gcf,append(svpath,'fast_g0bif_cusp'),'epsc')
% saveas(gcf,append(svpath,'fast_g0bif_cusp'),'fig')
saveas(gcf,append(svpath,'fast_g0bif_cusp'),'png')



%% 2 Pole Stuff

% 1 parameter bifurcation plot
% in gamma and Delta N
X=load(append(svpath,'2pole_bif1gammaN.dat'));

% data sorting to separate out stable / unstable branches

Xstable = X(X(:,4)==1,:);   % extract stable branch
Xunstable = X(X(:,4)==2,:); % extract stablebranch
Xu_p=Xunstable(2:end-3,:);
Xs_p=sortrows(Xstable,2);
    
figure(1);
clf
hold on;
set(gca, 'TickLabelInterpreter', 'latex')
fontsize(16, 'points')
box on
plot(Xs_p(:,1),Xs_p(:,2),'linewidth',1.5,'color','r')
plot(Xu_p(:,1),Xu_p(:,2),'linewidth',1.5,'color','k')
xlabel('$\gamma$','Interpreter','latex','fontsize',16)
ylabel('$\Delta N$','Interpreter','latex','fontsize',16)
xlim([0,2.5])
ylim([-55,55])
        
legend('stable','unstable','location','best','interpreter','latex','fontsize',16)
    
% saveas(gcf,append(svpath,'2pole_pitchfork_gammaDelN'),'fig')
% saveas(gcf,append(svpath,'2pole_pitchfork_gammaDelN'),'epsc')
saveas(gcf,append(svpath,'2pole_pitchfork_gammaDelN'),'png')




% 2 parameter plot in gamma and Ntot 
% implicit solution for the curve in Ntot / gamma space

Nt_vals   = linspace(0.9, 80, 500);
gamma_vals = nan(size(Nt_vals));


for i = 1:numel(Nt_vals)
    Nt = Nt_vals(i);
    % From eliminating kappa: (g-1)*((Nt-Ft)/2)^g = Ft/2
    % Use this to solve for Ft given g, then enforce eq(1) for kappa consistency.
    % Combine both: solve eqs (1)&(2) for Ft and g simultaneously.
    fun = @(x) residuals(x, Nt, beta, kappa);
    x0  = [0.5*Nt; 1.5];
    opts = optimoptions('fsolve','Display','off','TolFun',1e-10,'TolX',1e-10);
    [sol, ~, flag] = fsolve(fun, x0, opts);
    if flag > 0 && sol(1) > 0 && sol(2) > 1
        gamma_vals(i) = sol(2);
    end
end



bp = load(append(svpath,'2pole_bif2gammaNtot.dat'));
bpsort = sortrows(bp);
figure(1)
clf
set(gca,'TickLabelInterpreter','latex','fontsize',16)
box on; hold on
% plot(bpsort(:,1), bpsort(:,2), 'c','Linewidth',2); 
plot(gamma_vals, Nt_vals, 'c', 'LineWidth', 2);

text(0.12,15,'symmetric plasmid distribution','interpreter','latex','fontsize',16)
text(1.5,50,'plasmids remain in old pole','interpreter','latex','fontsize',16)
  

xlabel('$\gamma$','interpreter','latex','fontsize',16); ylabel('$N_{tot}$','interpreter','latex','fontsize',16);
xlim([0,3])
ylim([0,80])
%legend('continuation method','implicit solution','interpreter','latex','fontsize',20)

% saveas(gcf,append(svpath,'2pole_pitchfork_gammaNtot'),'fig')
% saveas(gcf,append(svpath,'2pole_pitchfork_gammaNtot'),'epsc')
saveas(gcf,append(svpath,'2pole_pitchfork_gammaNtot'),'png')

function r = residuals(x, Nt, beta, kappa)
    Ft = x(1); g = x(2);
    h  = ((Nt - Ft)/2)^g;
    r(1) = -beta*Ft*h*g + kappa*(Nt - Ft);   % eq (1) unchanged
    r(2) = (g - 1)*h - Ft/2;                  % reduced relation
end



