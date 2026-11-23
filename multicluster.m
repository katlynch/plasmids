function dy = multicluster(t, y, params)
N = params.Ntot;
c = reshape(y, N, 2)';

jj = (1:N)';
jg = jj.^params.gamma;  jg(1) = 0;

dc = zeros(2, N);
for p = 1:2
    f = c(p,1);
    J = zeros(N,1);
    J(1) = 0.5*params.beta*f^2 - 2*params.kappa*c(p,2);
    js = (2:N-1)';
    J(js) = params.beta*jg(js)*f.*c(p,js)' - params.kappa*(js+1).*c(p,js+1)';
    dc(p, 2:N) = J(1:N-1)' - J(2:N)';
    dc(p,1) = -sum(jj(2:N) .* dc(p,2:N)');
end
dc(1,1) = dc(1,1) + params.rho*(c(2,1) - c(1,1));
dc(2,1) = dc(2,1) + params.rho*(c(1,1) - c(2,1));

dy = reshape(dc', [], 1);
end