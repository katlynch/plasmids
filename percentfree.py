"""percent free comparison"""
import numpy as np
from scipy.integrate import solve_ivp
import os

# ── Mean-field (distribution over cluster sizes) ──────────────────────────────

def mf_rhs(t, y, N, beta, kappa, gamma):
    c = y.copy()
    dc = np.zeros(N + 2)
    jj = np.arange(N + 2, dtype=float)
    jg = np.where(jj >= 2, jj ** gamma, 0.0)
    f = c[1]
    J = np.zeros(N + 2)
    J[1] = 0.5 * beta * f * f - 2.0 * kappa * c[2]
    js = np.arange(2, N)
    J[js] = beta * jg[js] * f * c[js] - kappa * (js + 1.0) * c[js + 1]
    dc[2:N + 1] = J[1:N] - J[2:N + 1]
    dc[1] = (-beta * f * f
             - beta * f * (jg[2:N] * c[2:N]).sum()
             + kappa * (jj[3:N + 1] * c[3:N + 1]).sum()
             + 4.0 * kappa * c[2])
    return dc

def run_mf(N, beta, kappa, gamma, T=10000.0):
    c0 = np.zeros(N + 2); c0[N] = 1.0
    sol = solve_ivp(mf_rhs, (0, T), c0,
                    method="LSODA", args=(N, beta, kappa, gamma),
                    rtol=1e-9, atol=1e-11)
    c = sol.y[:, -1]
    jj = np.arange(N + 2, dtype=float)
    return c[1], (c[2:] * jj[2:]).sum()

# ── Simple mean-field ODE (scalar f, b) ──────────────────────────────────────

def mf2_rhs(t, y, beta, kappa, gamma):
    f, b = y
    return [-beta*f**2 - beta*f*b**gamma + kappa*b,
             beta*f**2 + beta*f*b**gamma - kappa*b]

def run_mf2(N, beta, kappa, gamma, T=10000.0):
    sol = solve_ivp(mf2_rhs, (0, T), [float(N), 0.0],
                    method="LSODA", args=(beta, kappa, gamma),
                    rtol=1e-9, atol=1e-11)
    return sol.y[0, -1], sol.y[1, -1]

# ── Two-compartment cluster MF ────────────────────────────────────────────────

def mf2p_rhs(t, y, N, beta, kappa, gamma, rho):
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

def run_mf2p(N, beta, kappa, gamma, rho, T=10000.0):
    c0 = np.zeros((2, N + 2)); c0[0, N] = 1.0
    sol = solve_ivp(mf2p_rhs, (0, T), c0.ravel(),
                    method="LSODA", args=(N, beta, kappa, gamma, rho),
                    rtol=1e-9, atol=1e-11)
    c = sol.y[:, -1].reshape(2, N + 2)
    jj = np.arange(N + 2, dtype=float)
    f1 = c[0, 1]; b1 = (c[0, 2:] * jj[2:]).sum()
    f2 = c[1, 1]; b2 = (c[1, 2:] * jj[2:]).sum()
    return f1, b1, f2, b2

# ── Two-compartment scalar MF ─────────────────────────────────────────────────

def mf2p2_rhs(t, y, beta, kappa, gamma, rho):
    f1, b1, f2, b2 = y
    return [-beta*f1**2 - beta*f1*b1**gamma + kappa*b1 + rho*(f2-f1),
             beta*f1**2 + beta*f1*b1**gamma - kappa*b1,
            -beta*f2**2 - beta*f2*b2**gamma + kappa*b2 + rho*(f1-f2),
             beta*f2**2 + beta*f2*b2**gamma - kappa*b2]

def run_mf2p2(N, beta, kappa, gamma, rho, T=10000.0):
    sol = solve_ivp(mf2p2_rhs, (0, T), [float(N), 0.0, 0.0, 0.0],
                    method="LSODA", args=(beta, kappa, gamma, rho),
                    rtol=1e-9, atol=1e-11)
    f1, b1, f2, b2 = sol.y[:, -1]
    return f1, b1, f2, b2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    beta, kappa, rho = 0.015, 0.08, 0.13
    Ntot_vals = np.linspace(5, 200, 20).astype(int)
    here = os.path.dirname(os.path.abspath(__file__))

    gammas = [0, 2/3, 1, 2]
    colors = ['red', 'blue', 'green', 'darkorange']

    # collect results per gamma
    results = {}
    for gamma in gammas:
        r1c, r1s = [], []
        r2c_1, r2c_2, r2s_1, r2s_2 = [], [], [], []
        for N in Ntot_vals:
            f, b = run_mf(N, beta, kappa, gamma)
            r1c.append(f / N)
            f, b = run_mf2(N, beta, kappa, gamma)
            r1s.append(f / N)
            f1, b1, f2, b2 = run_mf2p(N, beta, kappa, gamma, rho)
            r2c_1.append(f1 / N); r2c_2.append(f2 / N)
            f1, b1, f2, b2 = run_mf2p2(N, beta, kappa, gamma, rho)
            r2s_1.append(f1 / N); r2s_2.append(f2 / N)
        results[gamma] = (r1c, r1s, r2c_1, r2c_2, r2s_1, r2s_2)

        # (A) 1compartment_comp.png — one panel per gamma, both models on each
    fig, axes1 = plt.subplots(1, 4, figsize=(18, 4))
    for ax, gamma, color in zip(axes1, gammas, colors):
        r1c, r1s, *_ = results[gamma]
        ax.plot(Ntot_vals, r1c, 'o-',  color=color, label='multi cluster')
        ax.plot(Ntot_vals, r1s, 's--', color=color, label='net bound')
        ax.set_title(f'γ={gamma:.4g}')
        ax.set_xlabel('Ntot'); ax.set_ylabel('fraction free plasmids (f/Ntot)')
        ax.legend()
    fig.suptitle(f'1-compartment: fraction free vs Ntot  (β={beta}, κ={kappa})')
    fig.tight_layout()
    fig.savefig(os.path.join(here, '1compartment_comp.png'), dpi=150)
    plt.show()

    # (B) 2compartment_comp.png — one panel per gamma, total free only, both models on each
    fig, axes2 = plt.subplots(1, 4, figsize=(18, 4))
    for ax, gamma, color in zip(axes2, gammas, colors):
        r1c, r1s, r2c_1, r2c_2, r2s_1, r2s_2 = results[gamma]
        r2c_tot = [a + b for a, b in zip(r2c_1, r2c_2)]
        r2s_tot = [a + b for a, b in zip(r2s_1, r2s_2)]
        ax.plot(Ntot_vals, r2c_tot, 'o-',  color=color, label='multi cluster')
        ax.plot(Ntot_vals, r2s_tot, 's--', color=color, label='net bound')
        ax.set_title(f'γ={gamma:.4g}')
        ax.set_xlabel('Ntot'); ax.set_ylabel('fraction free plasmids (f/Ntot)')
        ax.legend()
    fig.suptitle(f'2-compartment (total): fraction free vs Ntot  (β={beta}, κ={kappa}, ρ={rho})')
    fig.tight_layout()
    fig.savefig(os.path.join(here, '2compartment_comp.png'), dpi=150)
    plt.close(fig)
    print("Saved 1compartment_comp.png and 2compartment_comp.png")
    plt.show()
