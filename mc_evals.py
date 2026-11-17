"""Leading eigenvalue at symmetric fixed point vs gamma, for a range of N."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
import os

# set up parameters and functions
Nvals = np.linspace(10, 200, 11, dtype=int)
gvals = np.linspace(0, 3, 19)

beta, kappa, rho = 0.015, 0.08, 0.13


def mf_rhs(t, y, N, gamma):
    # meanfield equations for becker doring multi cluster model
    c = y.reshape(2, N + 2)
    dc = np.zeros_like(c)
    jj = np.arange(N + 2, dtype=float)
    jg = np.where(jj >= 2, jj ** gamma, 0.0)
    for p in (0, 1):
        f = c[p, 1]
        J = np.zeros(N + 2, dtype=c.dtype)
        J[1] = 0.5 * beta * f * f - 2.0 * kappa * c[p, 2]
        js = np.arange(2, N)
        J[js] = beta * jg[js] * f * c[p, js] - kappa * (js + 1.0) * c[p, js + 1]
        dc[p, 2:N+1] = J[1:N] - J[2:N+1]
        dc[p, 1] = (-beta * f * f
                    - beta * f * (jg[2:N] * c[p, 2:N]).sum()
                    + kappa * (jj[3:N+1] * c[p, 3:N+1]).sum()
                    + 4.0 * kappa * c[p, 2])
    dc[0, 1] += rho * (c[1, 1] - c[0, 1])
    dc[1, 1] += rho * (c[0, 1] - c[1, 1])
    return dc.ravel()

def ic_symmetric(N):
    # defines a symmetric initial condition close to the symmetric steady state
    c = np.zeros((2, N + 2))
    c[0, N] = 0.5
    c[1, N] = 0.5
    return c

#def leading_eigenvalue(N, gamma):
#    y0 = ic_symmetric(N).ravel()
#    fp = solve_ivp(mf_rhs, (0, 1000), y0, args=(N, gamma),
#                   method="LSODA", rtol=1e-10, atol=1e-12).y[:, -1]
#    n = len(fp)
#    eps = 1e-6
#    f0 = mf_rhs(0, fp, N, gamma)
#    J = np.zeros((n, n))
#    for i in range(n):
#        fp_p = fp.copy(); fp_p[i] += eps
#        J[:, i] = (mf_rhs(0, fp_p, N, gamma) - f0) / eps
#    real = np.linalg.eigvals(J).real
#    # note: need extra filter here because 0>-1
#    return real[np.abs(real) > eps].max()




def leading_eigenvalue(N, gamma, return_components=False):
    # Integrate to obtain steady state
    y0 = ic_symmetric(N).ravel()
    sol = solve_ivp(
        mf_rhs, (0, 1000), y0,
        args=(N, gamma),
        method="LSODA",
        rtol=1e-10,
        atol=1e-12
    )
    fp = sol.y[:, -1]

    # Check steady-state residual
    residual = np.linalg.norm(mf_rhs(0, fp, N, gamma))

    if residual > 1e-5:
        print(f"Warning: residual = {residual:.3e}, "
              f"N={N}, gamma={gamma}")

    # Active coordinates
    active = np.concatenate([
        np.arange(1, N + 1),
        np.arange(N + 3, 2*N + 3)
    ])

    # Complex-step Jacobian
    n = 2 * N
    h = 1e-30
    J = np.zeros((n, n))

    for i in range(n):
        fp_c = fp.astype(complex)
        fp_c[active[i]] += 1j * h

        J[:, i] = np.imag(
            mf_rhs(0, fp_c, N, gamma)[active]
        ) / h

    # Decompose into population blocks
    A1 = J[:N, :N]
    A2 = J[N:, N:]
    B12 = J[:N, N:]
    B21 = J[N:, :N]

    # Check symmetry of the Jacobian
    block_error = max(
        np.linalg.norm(A1 - A2),
        np.linalg.norm(B12 - B21)
    )

    if block_error > 1e-8:
        print(f"Warning: block symmetry error = {block_error:.3e}")

    A = 0.5 * (A1 + A2)
    B = 0.5 * (B12 + B21)

    # Symmetric and antisymmetric blocks
    J_sym = A + B
    J_anti = A - B

    # Restrict symmetric block to fixed-mass manifold
    w = np.arange(1, N + 1, dtype=float)
    Q = null_space(w.reshape(1, -1))

    J_sym_red = Q.T @ J_sym @ Q

    # Calculate spectra
    evals_sym = np.linalg.eigvals(J_sym_red)
    evals_anti = np.linalg.eigvals(J_anti)

    # Spectral abscissae
    lambda_sym = evals_sym.real.max()
    lambda_anti = evals_anti.real.max()

    # Overall leading eigenvalue on fixed-mass manifold
    lambda_lead = max(lambda_sym, lambda_anti)

    if return_components:
        return lambda_lead, lambda_sym, lambda_anti

    return lambda_lead


# Calculate eigenvalues
results = {}

for N in Nvals:
    results[N] = np.array([
        leading_eigenvalue(N, g, return_components=True)
        for g in gvals
    ])



cmap = plt.cm.viridis
norm = plt.Normalize(Nvals.min(), Nvals.max())


# plotting
# Figure 1: Symmetric and Antisymmetric
fig1, axes1 = plt.subplots(1, 2, figsize=(11, 5), sharex=True)

titles1 = ["Symmetric perturbations", "Antisymmetric perturbations"]

for N in Nvals:
    color = cmap(norm(N))
    sym = results[N][:, 1]
    anti = results[N][:, 2]
    axes1[0].plot(gvals, sym, color=color, lw=2, marker='o', ms=3)
    axes1[1].plot(gvals, anti, color=color, lw=2, marker='o', ms=3)

for ax, title in zip(axes1, titles1):
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$\gamma$", fontsize=13)
    ax.axhline(0, color="k", lw=0.8, ls=":", alpha=0.6)
    ax.grid(True, alpha=0.25)
axes1[0].set_ylabel("Leading eigenvalue (Re)", fontsize=12)

fig1.tight_layout()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig1.colorbar(sm, ax=axes1, label="N (total plasmids)", shrink=0.85)


# Figure 2: Leading eigenvalue only
fig2, ax2 = plt.subplots(figsize=(7, 5))

for N in Nvals:
    color = cmap(norm(N))
    lead = results[N][:, 0]
    ax2.plot(gvals, lead, color=color, lw=2)

ax2.set_title("Overall leading eigenvalue", fontsize=12)
ax2.set_xlabel(r"$\gamma$", fontsize=13)
ax2.set_ylabel("Leading eigenvalue (Re)", fontsize=12)
ax2.axhline(0, color="k", lw=0.8, ls=":", alpha=0.6)
ax2.grid(True, alpha=0.25)

fig2.tight_layout()
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
fig2.colorbar(sm, ax=ax2, label="N (total plasmids)", shrink=0.85)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mc_eigval.png")
fig2.savefig(out, dpi=150)
print(f"Saved to {out}")

plt.show()
