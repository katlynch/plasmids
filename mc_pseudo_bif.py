"""Mean-field two-pole plasmid model — pseudo-bifurcation diagram."""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def mf_rhs(t, y, N, beta, kappa, gamma, rho):
    c = y.reshape(2, N + 2)
    dc = np.zeros_like(c)
    jj = np.arange(N + 2, dtype=float)
    jg = np.where(jj >= 2, jj ** gamma, 0.0)
    for p in (0, 1):
        f = c[p, 1]
        J = np.zeros(N + 2)
        J[1] = 0.5 * beta * f * f - 2.0 * kappa * c[p, 2]
        js = np.arange(2, N)
        J[js] = beta * jg[js] * f * c[p, js] - kappa * (js + 1.0) * c[p, js + 1]
        dc[p, 2:N + 1] = J[1:N] - J[2:N + 1]
        dc[p, 1] = (-beta * f * f
                    - beta * f * (jg[2:N] * c[p, 2:N]).sum()
                    + kappa * (jj[3:N + 1] * c[p, 3:N + 1]).sum()
                    + 4.0 * kappa * c[p, 2])
    dc[0, 1] += rho * (c[1, 1] - c[0, 1])
    dc[1, 1] += rho * (c[0, 1] - c[1, 1])
    return dc.ravel()

def run(gamma, ic_pole, N=40, beta=0.015, kappa=0.08, rho=0.13, T=1000000):
    c0 = np.zeros((2, N + 2)); c0[ic_pole, N] = 1.0
    sol = solve_ivp(mf_rhs, (0, T), c0.ravel(), method="LSODA",
                    args=(N, beta, kappa, gamma, rho), rtol=1e-9, atol=1e-11)
    c = sol.y[:, -1].reshape(2, N + 2)
    jj = np.arange(N + 2, dtype=float)
    total = (c * jj).sum(-1)
    return total[1] - total[0]   # ΔN = pole1 - pole0


gammas_outer = np.linspace(0.0, 3.0, 20)
gammas_inner = np.linspace(0,1,4)
gammas = np.unique(np.concatenate([gammas_outer, gammas_inner]))

fig, ax = plt.subplots(figsize=(8, 5))
for ic_pole, label, color, marker in (
        (0, "IC: all in pole 0", "#2166ac", "o"),
        (1, "IC: all in pole 1", "#d6604d", "s")):
    dNs = [run(g, ic_pole) for g in gammas]
    ax.plot(gammas, dNs, marker=marker, lw=1.8, ms=6, color=color, label=label)

ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.set_xlabel("γ", fontsize=12)
ax.set_ylabel("ΔN", fontsize=12)
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mc_pseudo_bif.png")
fig.savefig(out, dpi=150)
print(f"Saved to {out}")
plt.show()