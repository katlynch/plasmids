
function [f1_vals,b1_vals,f2_vals,b2_vals,tvals] = copyno_ssa(N0,params)
% input: initial plasmids (free and bound) in compartment 1 (old pole)
% output: full time-course arrays
% required parameters: beta, kappa, rho, gamma

% initialization
t=0;
maxitr=4000;


j=1;
% initialization
f1_vals=zeros(1,maxitr);
f2_vals=zeros(1,maxitr);
b1_vals=zeros(1,maxitr);
b2_vals=zeros(1,maxitr);
tvals=zeros(1,maxitr);
f1_vals(1)=N0(1);
b1_vals(1)=N0(2);
f2_vals(1)=N0(3);
b2_vals(1)=N0(4);
% algorithm
while j<maxitr
%%%%%%%%%% SYSTEM %%%%%%%%%%%%%%%%
%---------------------------------------------
% rxn  |  equation              | rate
%------|------------------------|------------
% r1   | F + F -> 2B            | 1/2*beta*f1*(f1-1)
% r2   | F + B -> B+B           | beta*f1*b1^gamma
% r3   | B -> F                 | kappa*b1
%---------------------------------------------
% r4   | F + F -> B             | 1/2*beta*b2*f2*(f2-1)
% r5   | F + B -> B             | beta*f2*b2^gamma
% r6   | B -> F                 | kappa*b2
%---------------------------------------------
% r7   | F_1 -> F_2,            | rho f1
% r7   | F_2 -> F_1,            | rho f2
%---------------------------------------------
%% Step 1: Calculate propensities
f1=f1_vals(j);
b1=b1_vals(j);
f2=f2_vals(j);
b2=b2_vals(j);
a(1)=1/2*params.beta*f1*(f1-1);
a(2)=params.beta*f1*b1^params.gamma;
a(3)=params.kappa*b1;
a(4)=1/2*params.beta*f2*(f2-1);
a(5)=params.beta*f2*b2^params.gamma;
a(6)=params.kappa*b2;
a(7)=params.rho*f1;
a(8)=params.rho*f2;
% a0 is total propensity that anything happens. This number emerges
% more out of mathematical necessity than physical intuition.
a0 = sum(a);
r = rand(2,1);
tau = -log(r(1))/a0;
% Note 0<=mu<=M. This decides which reaction occurs. If mu=0, no
% reaction occurs. The switch statement does nothing in this case.
mu = sum(r(2)*a0 <= cumsum(a));
t=t+tau;
%% Step 2: determine which reaction occurs
%reactions
% --> do the step
switch mu
case 8         % c1 initialization
        f1=f1-2;
        b1=b1+2;
case 7         % c1 binding
        f1=f1-1;
        b1=b1+1;
case 6         % c1 unbinding
        f1=f1+1;
        b1=b1-1;
case 5         % c2 initialization
        f2=f2-2;
        b2=b2+2;
case 4         % c2 binding
        f2=f2-1;
        b2=b2+1;
case 3         % c2 unbinding
        f2=f2+1;
        b2=b2-1;
case 2          % transfer c1->c2
        f1=f1-1;
        f2=f2+1;
case 1         % transfer c2->c1
        f1=f1+1;
        f2=f2-1;
end
tvals(j+1)=t;
f1_vals(j+1)=f1;
b1_vals(j+1)=b1;
f2_vals(j+1)=f2;
b2_vals(j+1)=b2;
j=j+1;
end
end