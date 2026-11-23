"""
gen_homotopy.py
Generate homotopy.ode for XPP/AUTO.
Usage: python gen_homotopy.py [N]  (default N=40)

State variables: DeltaF=(f1-f2), bvar1, bvar2, c1j, c2j for j=2..N  (2*N+1 total)

Conservation: SumF = f1+f2 = Ntot - bvar1 - bvar2 - csum1 - csum2
              f1 = (SumF + DeltaF)/2
              f2 = (SumF - DeltaF)/2

This eliminates the zero eigenvector (uniform monomer shift) without
creating artificial cross-compartment coupling. DeltaF is the natural
bifurcation observable (asymmetry between compartments).

At theta=0: reduces to 3-variable system (DeltaF, bvar1, bvar2)
At theta=1: reduces to full multicluster (DeltaF + 2*(N-1) cluster variables)
"""
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40

lines = []
lines.append("# homotopy.ode  (auto-generated, N={})".format(N))
lines.append("# State: DeltaF=f1-f2, bvar1, bvar2, c1j, c2j (j=2..N)")
lines.append("")

# Named expressions: csums, SumF, f1, f2
csum1 = "+".join("{}*c1{}".format(j,j) for j in range(2,N+1))
csum2 = "+".join("{}*c2{}".format(j,j) for j in range(2,N+1))

lines.append("# Conservation and monomer reconstruction")
lines.append("csum1={}\n".format(csum1))
lines.append("csum2={}\n".format(csum2))
lines.append("SumF=Ntot-bvar1-bvar2-csum1-csum2")
lines.append("f1=(SumF+DeltaF)/2")
lines.append("f2=(SumF-DeltaF)/2\n")
lines.append("")

# Partial sums for DeltaF equation
# dcsum1 = theta*(beta*f1^2 - 4*kappa*c12 + beta*f1*pg1 - kappa*pk1)
# dcsum2 = theta*(beta*f2^2 - 4*kappa*c22 + beta*f2*pg2 - kappa*pk2)
# pg = sum_{j=2}^{N-1} j^gamma * cj  (excludes j=N)
# pk = sum_{k=3}^{N}   k * ck
lines.append("# Partial sums for DeltaF equation")
lines.append("pg1={}".format("+".join("{}^gamma*c1{}".format(j,j) for j in range(2,N))))
lines.append("pk1={}".format("+".join("{}*c1{}".format(k,k) for k in range(3,N+1))))
lines.append("pg2={}".format("+".join("{}^gamma*c2{}".format(j,j) for j in range(2,N))))
lines.append("pk2={}".format("+".join("{}*c2{}".format(k,k) for k in range(3,N+1))))
lines.append("")

# bvar equations
lines.append("# Bulk variables from asymmetry model (freeze at theta=1)")
lines.append("dbvar1/dt=(1-theta)*(beta*f1*(f1+bvar1^gamma)-kappa*bvar1)")
lines.append("dbvar2/dt=(1-theta)*(beta*f2*(f2+bvar2^gamma)-kappa*bvar2)")
lines.append("")

# Cluster distribution equations
lines.append("# Cluster distribution equations (theta-scaled)")
for p, f, cx in [(1,"f1","c1"),(2,"f2","c2")]:
    for j in range(2, N+1):
        if j == 2:
            Jjm1 = "0.5*beta*{}^2-2*kappa*{}2".format(f, cx)
        else:
            Jjm1 = "beta*{}^gamma*{}*{}{}-kappa*{}*{}{}".format(j-1, f, cx, j-1, j, cx, j)
        if j == N:
            Jj = "0"
        else:
            Jj = "beta*{}^gamma*{}*{}{}-kappa*{}*{}{}".format(j, f, cx, j, j+1, cx, j+1)
        rhs = "theta*({})".format(Jjm1) if Jj=="0" else "theta*(({})-({}))".format(Jjm1,Jj)
        lines.append("d{}{}/dt={}".format(cx, j, rhs))
    lines.append("")

# DeltaF equation: d(DeltaF)/dt = df1/dt - df2/dt
# df1 = -dbv1 - dcsum1 + rho*(f2-f1)
# df2 = -dbv2 - dcsum2 + rho*(f1-f2)
# d(DeltaF)/dt = -(dbv1-dbv2) - (dcsum1-dcsum2) + 2*rho*(f2-f1)
# dcsum1 = theta*(beta*f1^2 - 4*kappa*c12 + beta*f1*pg1 - kappa*pk1)
# dcsum2 = theta*(beta*f2^2 - 4*kappa*c22 + beta*f2*pg2 - kappa*pk2)
lines.append("# DeltaF = f1-f2 equation (d(DeltaF)/dt = df1/dt - df2/dt)")
lines.append("dDeltaF/dt=-((1-theta)*(beta*f1*(f1+bvar1^gamma)-kappa*bvar1)-(1-theta)*(beta*f2*(f2+bvar2^gamma)-kappa*bvar2))-theta*(beta*f1^2-4*kappa*c12+beta*f1*pg1-kappa*pk1-beta*f2^2+4*kappa*c22-beta*f2*pg2+kappa*pk2)-2*rho*DeltaF")
lines.append("")

# Auxiliary outputs
lines.append("# Auxiliary outputs")
lines.append("aux f1=f1")
lines.append("aux f2=f2")
lines.append("aux bvar1=bvar1")
lines.append("aux bvar2=bvar2")
lines.append("aux csum1=csum1")
lines.append("aux csum2=csum2")
lines.append("aux DeltaM=(f1+bvar1+csum1)-(f2+bvar2+csum2)")
lines.append("")

# Parameters
lines.append("par gamma=0.2,Ntot={},beta=0.015,kappa=0.08,rho=0.13,theta=0.1".format(N))
lines.append("")

# Initial conditions: symmetric (DeltaF=0)
lines.append("init DeltaF=0")
lines.append("init bvar1=0,bvar2=0")
for cx in ["c1","c2"]:
    for j in range(2, N+1):
        lines.append("init {}{}=0".format(cx, j))
lines.append("")

# Numerics
lines.append("@ total=500,dt=0.1,meth=cvode")
lines.append("@ xplot=t,yplot=DeltaF,xlo=0,xhi=500,ylo={},yhi={}".format(-N, N))
lines.append("@ autoxmin=0,autoxmax=3,autoymin={},autoymax={}".format(-N, N))
lines.append("@ autovar=DeltaF,parmin=0,parmax=3")
lines.append("@ nmax=500,ds=0.005,dsmin=0.0001,dsmax=0.05")
lines.append("@ epsl=1e-6,epsu=1e-6,epss=1e-4")
lines.append("@ ntst=40,ncol=4")
lines.append("")
lines.append("done")

out = "\n".join(lines)
with open("homotopy.ode","w") as fh:
    fh.write(out)
print("Wrote homotopy.ode ({} state variables)".format(2*N+1))
