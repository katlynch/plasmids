"""
Mean-field equations for the two-pole, many-cluster plasmid model.

State: c[p, j] = (mean) number of clusters of exactly j plasmids in pole p,
p = 0, 1 and j = 1..N.  c[p, 1] = f_p is the number of free plasmids.
Total mass sum_{p,j} j c[p,j] = N is conserved.

Reactions in each pole (stochastic model):
    nucleation     f + f     -> c_2          (1/2) beta f (f-1)
    growth  j>=2   f + c_j   -> c_{j+1}      beta j^gamma f c_j
    shedding j>=3  c_j       -> c_{j-1} + f  kappa j c_j
    dimer breakup  c_2       -> f + f        2 kappa c_2
Transfer between poles (free plasmids only):
    f_p -> f_q                               rho f_p

Mean-field limit (drop Var(f) - f and Cov(f, c_j)): in each pole a
Becker-Doering chain
    dc_j/dt = J_{j-1} - J_j,                         j = 2..N
    J_1 = (1/2) beta f^2 - 2 kappa c_2               (nucleation flux)
    J_j = beta j^gamma f c_j - kappa (j+1) c_{j+1},  j = 2..N-1
    J_N = 0                                          (no growth past size N)
    df_p/dt = -beta f_p^2 - beta f_p sum_{j=2}^{N-1} j^gamma c_j
              + kappa sum_{j=3}^{N} j c_j + 4 kappa c_2 + rho (f_q - f_p)
The two chains are coupled only through the rho term.
"""
import numpy as np
from scipy.integrate import solve_ivp


gamma, N, beta, kappa, rho, T = 0, 60, 0.015, 0.08, 0.13, 1000.0

def mf_rhs(t, y, N, beta, kappa, gamma, rho):
    c = y.reshape(2, N + 2)                # index 0 and N+1 are unused padding
    dc = np.zeros_like(c)
    jj = np.arange(N + 2, dtype=float)
    jg = np.where(jj >= 2, jj ** gamma, 0.0)
    for p in (0, 1):
        f = c[p, 1]
        J = np.zeros(N + 2)
        J[1] = 0.5 * beta * f * f - 2.0 * kappa * c[p, 2]
        js = np.arange(2, N)
        J[js] = beta * jg[js] * f * c[p, js] - kappa * (js + 1.0) * c[p, js + 1]
        J[N] = 0.0
        dc[p, 2:N + 1] = J[1:N] - J[2:N + 1]
        dc[p, 1] = (-beta * f * f
                    - beta * f * (jg[2:N] * c[p, 2:N]).sum()
                    + kappa * (jj[3:N + 1] * c[p, 3:N + 1]).sum()
                    + 4.0 * kappa * c[p, 2])
    dc[0, 1] += rho * (c[1, 1] - c[0, 1])
    dc[1, 1] += rho * (c[0, 1] - c[1, 1])
    return dc.ravel()


def mf_solve(N, beta, kappa, gamma, rho, T, c0, n_record=400,
             rtol=1e-9, atol=1e-11):
    """Integrate from c0 (shape (2, N+2)). Returns (t, C) with C[t, p, j]."""
    ts = np.linspace(0.0, T, n_record)
    sol = solve_ivp(mf_rhs, (0, T), np.asarray(c0, float).ravel(), t_eval=ts,
                    method="LSODA", args=(N, beta, kappa, gamma, rho),
                    rtol=rtol, atol=atol)
    return ts, sol.y.T.reshape(-1, 2, N + 2)


def observables(C):
    """Free f_p, bound b_p and cluster number C_p per pole, each (..., 2)."""
    jj = np.arange(C.shape[-1], dtype=float)
    mask = jj >= 2
    return C[..., 1], (C * jj * mask).sum(-1), (C * mask).sum(-1)


def ic_one_cluster(N, pole=0):
    """All N plasmids in a single cluster in one pole."""
    c = np.zeros((2, N + 2))
    c[pole, N] = 1.0
    return 



    
t, C = mf_solve(N, beta, kappa, gamma, rho, T, ic_one_cluster(N))
f, b, nc = observables(C)
dN = (f[:, 0] + b[:, 0]) - (f[:, 1] + b[:, 1])
print(f"gamma={gamma:.3f}  t={T:g}:  f={f[-1].round(2)}  "
        f"b={b[-1].round(2)}  C={nc[-1].round(2)}  dN={dN[-1]:+.3f}")



