#!/usr/bin/env python3
"""
check_one_cluster.py
====================
Does the net-bound (one-cluster) model of github.com/katlynch/plasmids agree
with the one-cluster reduction of our Becker-Doring (Part IV) and
Smoluchowski (Part V) models?

1. Pointwise: our chain right-hand sides (plasmid.chain_rhs, both forms),
   evaluated at one-cluster states c_j = delta_{j,b} (integer b, any f),
   give df/dt and db/dt.  These are compared with Kat's meanfieldode.m.

2. Trajectories, N = 50, gamma = 0, 2/3, 1, 2, from Kat's odesolve.m start
   (pole 1 at the rounded mean of her one-pole SSA, pole 2 empty):
     - Kat meanfieldode.m             (f1, b1, f2, b2)
     - Kat asymmetry.ode  (XPP)       (Delta N, Delta F, F_tot)
     - Kat asymmetryode.m (MATLAB)    (Delta N, Delta F, F_tot), as written
     - our reduction, eqs. (bd2-one) = (sm-one)   (plasmid.one_rhs)

Writes ../one_cluster_check/one_cluster_models_agree.png and prints the
numbers.
"""
import os
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import plasmid as pl                                   # noqa: E402

OUT = os.path.join(HERE, "..", "one_cluster_check")
BETA, KAPPA, RHO, NTOT = 0.015, 0.08, 0.13, 50
GAMMAS = [0.0, 2 / 3, 1.0, 2.0]
LABS = [r"$\gamma=0$", r"$\gamma=2/3$", r"$\gamma=1$", r"$\gamma=2$"]


# ----------------------------------------------------------------------
#  Kat's equations, transcribed literally
# ----------------------------------------------------------------------

def _pw(x, g):
    """x**g with the base clipped at 0 (round-off can make an empty pole's
    b slightly negative, where x**g is NaN for non-integer g)."""
    return max(x, 0.0) ** g


def kat_meanfieldode(t, x, g):                       # meanfieldode.m
    f1, b1, f2, b2 = x
    return [-BETA * f1 ** 2 - BETA * f1 * _pw(b1, g) + KAPPA * b1 + RHO * (f2 - f1),
            BETA * f1 ** 2 + BETA * f1 * _pw(b1, g) - KAPPA * b1,
            -BETA * f2 ** 2 - BETA * f2 * _pw(b2, g) + KAPPA * b2 + RHO * (f1 - f2),
            BETA * f2 ** 2 + BETA * f2 * _pw(b2, g) - KAPPA * b2]


def kat_asym_matlab(t, x, g):                        # asymmetryode.m
    N, F, Ft = x
    L1 = ((Ft + F) * _pw(NTOT + N - F - Ft, g)) / 4
    L2 = ((Ft - F) * _pw(NTOT - N + F - Ft, g)) / 4
    return [-2 * RHO * F,
            -BETA * (F * Ft + L1 - L2) + KAPPA * (N - F) - 2 * RHO * F,
            -BETA * (F ** 2 / 2 + Ft ** 2 / 2 + L1 + L2) + KAPPA * (NTOT - Ft)]


def kat_asym_xpp(t, x, g):                           # asymmetry.ode
    N, F, Ft = x
    L1 = (Ft + F) / 2 * _pw((NTOT + N - F - Ft) / 2, g)
    L2 = (Ft - F) / 2 * _pw((NTOT - N + F - Ft) / 2, g)
    return [-2 * RHO * F,
            -BETA * (F * Ft + L1 - L2) + KAPPA * (N - F) - 2 * RHO * F,
            -BETA * (F ** 2 / 2 + Ft ** 2 / 2 + L1 + L2) + KAPPA * (NTOT - Ft)]


def kat_one_pole_ssa(g, rng, nsims=500, T=1000.0):   # sc_1pole_ssa.m
    fe, be = np.zeros(nsims), np.zeros(nsims)
    for s in range(nsims):
        f, b, t = NTOT / 2, NTOT / 2, 0.0
        while True:
            a = np.array([.5 * BETA * f * (f - 1), BETA * f * b ** g, KAPPA * b])
            a0 = a.sum()
            if a0 == 0:
                break
            t -= np.log(rng.random()) / a0
            if t >= T:
                break
            mu = np.searchsorted(np.cumsum(a), rng.random() * a0)
            if mu == 0:
                f, b = f - 2, b + 2
            elif mu == 1:
                f, b = f - 1, b + 1
            else:
                f, b = f + 1, b - 1
        fe[s], be[s] = f, b
    return round(fe.mean()), round(be.mean())


# ----------------------------------------------------------------------
#  1. pointwise check of our chain at one-cluster states
# ----------------------------------------------------------------------

def pointwise(model):
    P = pl.Params(model=model, N=NTOT, beta=BETA, kappa=KAPPA, rho=RHO,
                  betam=BETA)
    jj = np.arange(NTOT + 1)
    worst = 0.0
    for g in GAMMAS + [0.5, 1.5]:
        _, w = pl.weights(NTOT, g)
        for b1 in range(2, NTOT - 1):
            for b2 in (0, 2, 7, 20):
                for f1 in (0.0, 0.7, 3.0, 11.0):
                    f2 = 2.5
                    if b1 + b2 + f1 + f2 > NTOT:
                        continue
                    c = np.zeros((2, NTOT + 1))
                    c[0, 1], c[1, 1] = f1, f2
                    c[0, b1] = 1
                    if b2:
                        c[1, b2] = 1
                    d = pl.chain_rhs(0, c.ravel(), P, g, w, jj).reshape(2, -1)
                    ours = [d[0, 1], (jj[2:] * d[0, 2:]).sum(),
                            d[1, 1], (jj[2:] * d[1, 2:]).sum()]
                    # Kat's formula, with an empty pole offering no sites
                    kat = kat_meanfieldode(0, [f1, b1, f2, b2], g)
                    if b2 == 0:
                        kat[2] += BETA * f2 * 0 ** g
                        kat[3] -= BETA * f2 * 0 ** g
                    worst = max(worst, np.abs(np.subtract(ours, kat)).max())
    return worst


# ----------------------------------------------------------------------
#  2. trajectories
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUT, exist_ok=True)
    print("1. our chain at one-cluster states vs Kat's meanfieldode.m "
          "(max |difference| in df/dt, db/dt):")
    for m in ("bd", "smol"):
        print(f"   {m:5s}: {pointwise(m):.2e}")

    rng = np.random.default_rng(2024)
    T = 250.0
    ts = np.linspace(0, T, 2501)
    P = pl.Params(model="bd", N=NTOT, beta=BETA, kappa=KAPPA, rho=RHO)
    rows = [("f", r"free  $F_1$, $F_2$"), ("b", r"bound  $B_1$, $B_2$"),
            ("dN", r"$\Delta N = N_1-N_2$")]
    sty = {"kat": dict(color="#bbbbbb", lw=5, solid_capstyle="round"),
           "xpp": dict(color="#2c4fa8", lw=1.6, ls=(0, (1, 1.5))),
           "ours": dict(color="#c0392b", lw=1.4, ls=(0, (5, 3))),
           "mat": dict(color="#2e8b3d", lw=1.4, ls="-.")}
    fig, axs = plt.subplots(3, 4, figsize=(18, 11), sharex=True)
    print("\n2. trajectories, N = 50, t <= 250; max |difference| from Kat "
          "meanfieldode.m over the run")
    print("   gamma   start F0,B0   ours(f,b)   ours(f,b) b^0=1   XPP(dN)"
          "   MATLAB asymmetryode.m(dN)   dN(250): kat / ours / MATLAB")
    for c, (g, lab) in enumerate(zip(GAMMAS, LABS)):
        F0, B0 = kat_one_pole_ssa(g, rng)
        x0 = [F0, B0, 0.0, 0.0]
        y0 = [F0 + B0, F0, F0]
        kw = dict(t_eval=ts, method="Radau", rtol=1e-10, atol=1e-12)
        k = solve_ivp(kat_meanfieldode, (0, T), x0, args=(g,), **kw).y
        x = solve_ivp(kat_asym_xpp, (0, T), y0, args=(g,), **kw).y
        a = solve_ivp(kat_asym_matlab, (0, T), y0, args=(g,), **kw).y
        c0 = np.zeros((2, NTOT + 1))
        c0[0, 1] = F0
        c0[0, B0] = 1                                    # one cluster of B0
        o = pl.one_solve(P, g, ts, c0)
        ours = np.c_[o["f"][:, 0], o["b"][:, 0], o["f"][:, 1], o["b"][:, 1]].T
        kdN = k[0] + k[1] - k[2] - k[3]
        odN = ours[0] + ours[1] - ours[2] - ours[3]
        # same with b^0 = 1 for an empty pole (Kat's convention)
        o0 = solve_ivp(lambda t, y: [
            -BETA * y[0] * (y[0] + _pw(y[2], g)) + KAPPA * y[2] + RHO * (y[1] - y[0]),
            -BETA * y[1] * (y[1] + _pw(y[3], g)) + KAPPA * y[3] + RHO * (y[0] - y[1]),
            BETA * y[0] * (y[0] + _pw(y[2], g)) - KAPPA * y[2],
            BETA * y[1] * (y[1] + _pw(y[3], g)) - KAPPA * y[3]],
            (0, T), [F0, 0, B0, 0], **kw).y
        e_ours = np.abs(ours - k).max()
        e_ours0 = np.abs(o0[[0, 2, 1, 3]] - k).max()
        e_xpp = np.abs(x[0] - kdN).max()
        e_mat = np.abs(a[0] - kdN).max()
        print(f"   {g:5.3f}   {F0:3d},{B0:3d}      {e_ours:.1e}     "
              f"{e_ours0:.1e}           {e_xpp:.1e}   {e_mat:9.2e}"
              f"                   {kdN[-1]:.2f} / {odN[-1]:.2f} / {a[0][-1]:.2f}")

        for r, (key, ylab) in enumerate(rows):
            ax = axs[r, c]
            if key == "dN":
                ax.plot(ts, kdN, **sty["kat"])
                ax.plot(ts, x[0], **sty["xpp"])
                ax.plot(ts, odN, **sty["ours"])
                ax.plot(ts, a[0], **sty["mat"])
            else:
                i = 0 if key == "f" else 1
                for p, alpha in ((0, 1.0), (1, .55)):
                    ax.plot(ts, k[i + 2 * p], **sty["kat"], alpha=alpha)
                    ax.plot(ts, ours[i + 2 * p], **sty["ours"], alpha=alpha)
            ax.set_xlim(0, T)
            ax.grid(alpha=.3)
            if r == 0:
                ax.set_title(lab + f"   (start $F_0={F0}$, $B_0={B0}$)",
                             fontsize=13)
            if c == 0:
                ax.set_ylabel(ylab, fontsize=12)
            if r == 2:
                ax.set_xlabel("time (min)")
                ax.text(.97, .95, f"max |our − Kat| = {e_ours:.0e}\n"
                        f"max |XPP − Kat| = {e_xpp:.0e}\n"
                        f"max |MATLAB asym − Kat| = {e_mat:.1f}",
                        transform=ax.transAxes, ha="right", va="top",
                        fontsize=9.5, bbox=dict(fc="white", ec="#cccccc"))
    h = [plt.Line2D([], [], **sty[s]) for s in ("kat", "ours", "xpp", "mat")]
    fig.legend(h, ["Kat meanfieldode.m (net-bound model)",
                   "our one-cluster reduction of Becker–Döring / Smoluchowski "
                   "(eqs. bd2-one = sm-one)",
                   "Kat asymmetry.ode (XPP)", "Kat asymmetryode.m (MATLAB), "
                   "as written"], loc="upper center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(.5, 1.0))
    fig.suptitle(r"One-cluster model: Kat's equations vs our reduction, "
                 r"$N=50$, $\beta=0.015$, $\kappa=0.08$, $\rho=0.13$ "
                 "(pole 2 lighter in rows 1–2)", fontsize=14, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, .95])
    path = os.path.join(OUT, "one_cluster_models_agree.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
