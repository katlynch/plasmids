
function dxdt = meanfieldode(~,x,params)
% mean-field RHS / ODE
% for use integrating using built in ode solver
% ode45 is sufficient
dxdt=zeros(4,1);
f1 = x(1); b1 = x(2); f2 = x(3); b2 = x(4);
dxdt(1) =  -params.beta*(f1^2 ) - params.beta*f1*b1^params.gamma + params.kappa*b1 + params.rho*(f2-f1);
dxdt(2) =  params.beta*(f1^2 ) + params.beta*f1*b1^params.gamma - params.kappa*b1;
dxdt(3) =  -params.beta*(f2^2 ) - params.beta*f2*b2^params.gamma + params.kappa*b2 + params.rho*(f1-f2);
dxdt(4) =   params.beta*(f2^2 ) + params.beta*f2*b2^params.gamma - params.kappa*b2;
end