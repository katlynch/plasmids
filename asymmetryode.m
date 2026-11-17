function dxdt = asymmetryode(~, x, params)
    
    N  = x(1);
    F  = x(2);
    Ft = x(3);

    Lambda1 = ((Ft + F)*(params.Ntot + N - F - Ft)^params.gamma)/4;
    Lambda2 = ((Ft - F)*(params.Ntot - N + F - Ft)^params.gamma)/4;

    
    dN  = -2*(params.rho)*F;
    dF  = -(params.beta)*(F*Ft + Lambda1 - Lambda2) + params.kappa*(N - F) - 2*(params.rho)*F;
    dFt = -(params.beta)*((F^2)/2 + (Ft^2)/2 + Lambda1 + Lambda2) + params.kappa*(params.Ntot - Ft);

    dxdt = [dN; dF; dFt];
end