% Half-system SSA: f1 and b1 only (rho=0, no f2/b2)
% Reactions:
%   r1: F+F -> B   rate = 0.5*beta*f1*(f1-1)
%   r2: F+B -> B   rate = beta*f1*b1^gamma
%   r3: B -> F     rate = kappa*b1
function [f0,b0] = onecompartment_ssa(p)


% Settings
T      = 1000;    % end time (change here)
Nsims  = 500;    % number of simulations
f1_0   = p.Ntot/2;
b1_0   = p.Ntot/2;

% Storage for final values
f1_end = zeros(1, Nsims);
b1_end = zeros(1, Nsims);

for s = 1:Nsims
    f1 = f1_0;  b1 = b1_0;  t = 0;
    while t < T
        a = [0.5*p.beta*f1*(f1-1), ...
             p.beta*f1*b1^p.gamma, ...
             p.kappa*b1];
        a0 = sum(a);
        if a0 == 0, break; end
        r  = rand(1,2);
        t  = t - log(r(1))/a0;
        mu = find(cumsum(a) >= r(2)*a0, 1);
        switch mu
            case 1, f1=f1-2; b1=b1+2;   % r1: 2F consumed, 1B produced
            case 2, f1=f1-1; b1=b1+1;   % r2: 1F consumed, 1B produced
            case 3, f1=f1+1; b1=b1-1;   % r3: 1F produced, 1B consumed
        end
    end
    f1_end(s) = f1;
    b1_end(s) = b1;
end

f0=round(mean(f1_end));
b0=round(mean(b1_end));

% fprintf('Mean f1 = %.3f\n', round(mean(f1_end)));
% fprintf('Mean b1 = %.3f\n', round(mean(b1_end)));
% fprintf('Std  f1 = %.3f\n', std(f1_end));
% fprintf('Std  b1 = %.3f\n', std(b1_end));
end