"""
kat_figures.py
==============
The figure set of github.com/katlynch/plasmids (the MATLAB/Python files at
the root of this repository), made for our many-cluster models instead of the
net-bound (one-cluster) model:

  model 'bd'    Becker-Doring (Part IV)   -> ../beckerdoring/katlynch_figures
  model 'smol'  Smoluchowski  (Part V)    -> ../smoluchowski/katlynch_figures

beta = 0.015, kappa = 0.08, rho = 0.13 (beta_m = beta) and the plasmid number of
each original script.  Detachment per the notes (dimer dissolution kappa c_2;
the repo's python scripts use the old 2 kappa c_2).

  figure                                   repo original              source
  ---------------------------------------  -------------------------  ---------------
  poles_vs_time_N50.png                    2pole_sims_copyno          odesolve.m
  asymmetry_vs_time_N50.png                2pole_sims_g0/g23/g1/g2    odesolve.m
  closure_ssa_vs_meanfield_N42.png         mfa_means                  closure_check.m
  closure_dropped_terms_N42.png            mfa_terms                  closure_check.m
  closure_relative_error_N42.png           mfa_relerr                 closure_check.m
  slow_decay_vs_exact_N42.png              approx_comp_g0 / _g1       decayapprox.m
  fraction_free_vs_N_one_pole.png          1compartment_comp          percentfree.py
  fraction_free_vs_N_two_poles.png         2compartment_comp          percentfree.py
  leading_eigenvalue_vs_gamma.png          mc_eigval                  mc_evals.py
  sym_antisym_eigenvalues_vs_gamma.png     (figure 1, not saved)      mc_evals.py
  pseudo_bifurcation_N25.png               mc_pseudo_bif              mc_pseudo_bif.py
  bifurcation_dN_vs_gamma_N50.png          2pole_pitchfork_gammaDelN  bif_plots.m (XPP)
  pitchfork_locus_N_vs_gamma.png           2pole_pitchfork_gammaNtot  bif_plots.m

The two bifurcation figures are computed here (pseudo-arclength continuation
and eigenvalues of the full chain) instead of read from XPP/AUTO output.  The
gamma = 0 fast-subsystem cusp of bif_plots.m has no counterpart and is left
out.
"""
from __future__ import annotations

import argparse
import dataclasses
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import null_space
from scipy.optimize import brentq, fsolve

import plasmid as pl

OUT = "katlynch_figures"
MODEL = "bd"
CLR = ["#e60000", "#1a1ae6", "#339933", "#e6b31a"]       # repo's colours
G4 = [0.0, 2 / 3, 1.0, 2.0]
G4_LAB = [r"$\gamma=0$", r"$\gamma=2/3$", r"$\gamma=1$", r"$\gamma=2$"]


def Pn(N, **kw):
    return pl.Params(model=MODEL, N=int(N), **kw)


def _name():
    return Pn(10).name


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def _pool(fn, args):
    with ProcessPoolExecutor(pl.NPROC,
                             mp_context=multiprocessing.get_context("fork")) as ex:
        return list(ex.map(fn, args))


def _rates(P):
    s = rf"$\beta={P.beta}$, $\kappa={P.kappa}$, $\rho={P.rho}$"
    return s + (rf", $\beta_m={P.betam}$" if P.smol else "")


# ======================================================================
#  Symmetric steady state, Jacobian, eigenvalues
# ======================================================================


def sym_state(P, g):
    """Symmetric mean-field steady state, each pole holding N/2."""
    with np.errstate(all="ignore"):
        return _sym_state(P, g)


def _sym_state(P, g):
    ce = np.tile(pl.eq_pole(P, g, pl.eq_free(P, g, P.N / 2)), (2, 1))
    if not P.smol or (g == 1.0 and P.betam == P.beta):
        return ce
    # one pole holding N/2 (the poles decouple when f1 = f2): Newton in log
    # variables from the BD product form, falling back to a short
    # integration first if Newton does not converge
    j = np.arange(1, P.N + 1)                  # log of the product form
    u = pl.logq(P, g)[1:] + j * np.log(pl.eq_free(P, g, P.N / 2))
    for attempt in range(2):
        for _ in range(60):
            J = _H1jac(P, u, g)
            try:
                du = np.linalg.solve(J, -_H1(P, u, g))
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(du)):
                break
            lam, r0 = 1.0, np.abs(_H1(P, u, g)).max()
            while lam > 1e-4 and not np.abs(_H1(P, u + lam * du, g)).max() < r0:
                lam *= .5
            u = u + lam * du
            if np.abs(du).max() < 1e-12 or r0 < 1e-14:
                break
        if np.abs(_H1(P, u, g)).max() < 1e-11:
            break
        jj, w = pl.weights(P.N, g)
        Q = dataclasses.replace(P, rho=0.0)
        s = solve_ivp(pl.chain_rhs, (0, 2e4), ce.ravel(), method="BDF",
                      args=(Q, g, w, jj), rtol=1e-8, atol=1e-12)
        u = np.log(np.maximum(s.y[:, -1].reshape(2, -1)[0, 1:], 1e-60))
    c = np.zeros(P.N + 1)
    c[1:] = np.exp(u)
    return np.tile(c, (2, 1))


def _H1(P, u, g):
    """One pole holding N/2 at a symmetric state: rows d(log c_j)/dt, the
    f row replaced by the relative mass error."""
    N = P.N
    x = np.exp(u)
    jj, w = pl.weights(N, g)
    y = np.zeros((2, N + 1), dtype=np.result_type(u, g))
    y[0, 1:] = x
    y[1, 1:] = x
    r = pl.chain_rhs(0, y.ravel(), P, g, w, jj)[1:N + 1] / x
    r[0] = ((np.arange(1, N + 1) * x).sum() - N / 2) / (N / 2)
    return r


def _H1jac(P, u, g):
    n, h = len(u), 1e-30
    J = np.zeros((n, n))
    for i in range(n):
        uc = u.astype(complex)
        uc[i] += 1j * h
        J[:, i] = np.imag(_H1(P, uc, g)) / h
    return J


def _active(N):
    """state indices j = 1..N of both poles (index 0 of each pole unused)"""
    return np.r_[np.arange(1, N + 1), np.arange(N + 2, 2 * N + 2)]


def jac(P, g, c):
    """Jacobian of the chain on the active coordinates, by complex step."""
    N = P.N
    jj, w = pl.weights(N, g)
    act = _active(N)
    y = np.asarray(c, float).ravel()
    J = np.zeros((2 * N, 2 * N))
    h = 1e-30
    for i, k in enumerate(act):
        yc = y.astype(complex)
        yc[k] += 1j * h
        J[:, i] = np.imag(pl.chain_rhs(0, yc, P, g, w, jj)[act]) / h
    return J


def sym_anti_eigs(P, g):
    """Spectral abscissae of the symmetric (on the fixed-mass manifold) and
    antisymmetric blocks at the symmetric steady state (as mc_evals.py)."""
    N = P.N
    J = jac(P, g, sym_state(P, g))
    A = 0.5 * (J[:N, :N] + J[N:, N:])
    B = 0.5 * (J[:N, N:] + J[N:, :N])
    Q = null_space(np.arange(1, N + 1, dtype=float)[None])
    ls = np.linalg.eigvals(Q.T @ (A + B) @ Q).real.max()
    la = np.linalg.eigvals(A - B).real.max()
    return ls, la


def _eig_job(a):
    N, g = a
    return sym_anti_eigs(Pn(N), g)


# ======================================================================
#  odesolve.m:  2pole_sims_copyno, 2pole_sims_g*
# ======================================================================


def odesolve(quick=False):
    print("odesolve: poles_vs_time_N50, asymmetry_vs_time_N50")
    P = Pn(50)
    T = 250.0
    ts = np.linspace(0, T, 2001)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    fig2, axs2 = plt.subplots(1, 4, figsize=(20, 4.8))
    for g, col, lab, ax2 in zip(G4, CLR, G4_LAB, axs2):
        _, mean = pl.newborn_pool(P, g)
        _, Y = pl.chain_solve(P, g, T, mean, ts=ts)
        m = pl.moments(Y, g)
        one = pl.netbound_solve(P, g, ts, mean)
        print(f"  gamma={g:.3f}  start f1 {mean[0, 1]:.2f} b1 "
              f"{(mean[0] * np.arange(P.N + 1))[2:].sum():.2f};  t={T:g}: "
              f"f1 {m['f'][-1, 0]:.2f} b1 {m['b'][-1, 0]:.2f} f2 "
              f"{m['f'][-1, 1]:.2f} b2 {m['b'][-1, 1]:.2f}")
        for k, (key, p) in enumerate([("f", 0), ("b", 0), ("f", 1), ("b", 1)]):
            ax = axs.flat[k]
            ax.plot(ts, m[key][:, p], color=col, lw=1.8,
                    label=f"{lab}" if k == 0 else None)
            ax.plot(ts, one[key][:, p], color=col, lw=1.0, ls=(0, (4, 2.5)),
                    alpha=.8)
        sd = pl.sum_diff(Y, g)
        ax2.plot(ts, sd["dN"], color=col, lw=2, label=r"$\Delta N$")
        ax2.plot(ts, sd["dF"], color=col, lw=2, ls="--", label=r"$\Delta F$")
        ax2.plot(ts, sd["Ftot"], color=col, lw=2, ls="-.",
                 label=r"$F_{tot}$")
        ax2.set_title(lab, fontsize=15)
        ax2.set_xlabel("time (min)")
        ax2.set_xlim(0, T)
        ax2.legend(frameon=False, fontsize=11)
        ax2.grid(alpha=.25)
    for ax, ttl in zip(axs.flat, ["$F_1$", "$B_1$", "$F_2$", "$B_2$"]):
        ax.set_title(ttl, fontsize=15)
        ax.set_ylim(0, P.N)
        ax.set_xlim(0, T)
        ax.set_ylabel("plasmids")
        ax.grid(alpha=.25)
    for ax in axs[1]:
        ax.set_xlabel("time")
    h, l = axs[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="#555555", lw=1.8),
          plt.Line2D([], [], color="#555555", lw=1.0, ls=(0, (4, 2.5)))]
    l += [f"{_name()} mean-field chain", "Kat's net-bound model (meanfieldode.m)"]
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, .965))
    fig.suptitle(f"{_name()} — free and bound plasmids per pole, N = {P.N}, "
                 + _rates(P) + "; newborn start (all in pole 1)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    _save(fig, "poles_vs_time_N50.png")
    axs2[0].set_ylabel("plasmid copy no.")
    fig2.suptitle(f"{_name()} — pole asymmetry variables (mean-field), "
                  f"N = {P.N}, " + _rates(P), fontsize=13)
    fig2.tight_layout(rect=[0, 0, 1, .92])
    _save(fig2, "asymmetry_vs_time_N50.png")


# ======================================================================
#  closure_check.m:  mfa_means, mfa_terms, mfa_relerr
# ======================================================================


def closure(quick=False):
    print("closure_check: closure_ssa_vs_meanfield / dropped_terms / "
          "relative_error, N = 42")
    P = Pn(42)
    T, NREC = 500.0, 401
    nrep = 200 if quick else 500
    G3, C3 = G4[:3], CLR[:3]
    runs = []
    for g in G3:
        draws, mean = pl.newborn_pool(P, g)
        e = pl.ensemble(P, g, T, NREC, draws[:nrep], seed=4242, tag="kssa")
        _, Y = pl.chain_solve(P, g, T, mean, ts=e["t"])
        e["ode"] = pl.moments(Y, g)
        runs.append(e)

    # (a) SSA mean vs mean-field ODE
    fig, axs = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    for e, g, col, lab in zip(runs, G3, C3, G4_LAB):
        t = e["t"]
        for k, (key, p) in enumerate([("f", 0), ("b", 0), ("f", 1), ("b", 1)]):
            ax = axs.flat[k]
            X = e[key][:, :, p]
            mu, se = X.mean(0), X.std(0, ddof=1) / np.sqrt(len(X))
            ax.fill_between(t, mu - 2 * se, mu + 2 * se, color=col, alpha=.25,
                            lw=0)
            ax.plot(t, mu, color=col, lw=1.5, label=lab if k == 0 else None)
            ax.plot(t, e["ode"][key][:, p], color="k", lw=1.6, ls="--")
    for ax, ttl in zip(axs.flat, ["$F_1$", "$B_1$", "$F_2$", "$B_2$"]):
        ax.set_title(ttl, fontsize=15)
        ax.set_xlim(0, T)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.25)
    for ax in axs[1]:
        ax.set_xlabel("time (min)")
    h, l = axs[0, 0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="k", lw=1.6, ls="--"),
          mpl.patches.Patch(color="#999999", alpha=.35)]
    l += ["mean-field ODE", r"SSA mean $\pm$2 SE"]
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, .965))
    fig.suptitle(f"{_name()} — SSA ensemble mean ({nrep} runs) vs mean-field "
                 f"ODE, N = {P.N}, newborn start", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    _save(fig, "closure_ssa_vs_meanfield_N42.png")

    # (b), (c) binding flux: exact beta<f(f-1)> + beta<f A> vs its mean-field
    # value beta fbar^2 + beta fbar Abar; dropped = beta (Var f - fbar + Cov(f,A))
    fig, axs = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig3, axs3 = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for e, g, col, lab in zip(runs, G3, C3, G4_LAB):
        t = e["t"]
        line = f"  gamma={g:.3f}"
        for p in (0, 1):
            f, A = e["f"][:, :, p], e["A"][:, :, p]
            fb, Ab = f.mean(0), A.mean(0)
            cov = ((f - fb) * (A - Ab)).sum(0) / (len(f) - 1)
            lead = P.beta * (fb ** 2 + fb * Ab)
            drop = P.beta * (f.var(0, ddof=1) - fb + cov)
            with np.errstate(invalid="ignore", divide="ignore"):
                rel = np.abs(drop) / (np.abs(lead) + np.abs(drop))
            axs[p].plot(t, lead, color=col, lw=1.6,
                        label=lab if p == 0 else None)
            axs[p].plot(t, drop, color=col, lw=1.4, ls="--")
            axs3[p].plot(t, rel, color=col, lw=1.4, label=lab)
            late = t > T / 2
            line += (f"  pole {p + 1}: lead {lead[late].mean():.3f} dropped "
                     f"{drop[late].mean():+.4f} relerr {np.nanmean(rel[late]):.3f}")
        print(line)
    for p in (0, 1):
        for ax in (axs[p], axs3[p]):
            ax.set_title(f"Compartment {p + 1}", fontsize=14)
            ax.set_xlabel("time (min)")
            ax.set_xlim(0, T)
            ax.grid(alpha=.25)
        axs[p].axhline(0, color="k", lw=.6)
    axs[0].set_ylabel("rate contribution")
    h, l = axs[0].get_legend_handles_labels()
    h += [plt.Line2D([], [], color="#555555", lw=1.6),
          plt.Line2D([], [], color="#555555", lw=1.4, ls="--")]
    l += ["leading (mean-field) term", "dropped term"]
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=11,
               bbox_to_anchor=(.5, 1.02))
    fig.suptitle(rf"{_name()} — binding flux: mean-field part $\beta\bar f^2"
                 r"+\beta\bar f\bar A$ vs dropped part $\beta(\mathrm{Var}\,f"
                 r"-\bar f+\mathrm{Cov}(f,A))$, $A=\sum_j j^\gamma c_j$",
                 fontsize=12, y=1.09)
    fig.tight_layout()
    _save(fig, "closure_dropped_terms_N42.png")
    axs3[0].set_ylabel("relative error")
    axs3[0].set_ylim(0, .5)
    axs3[1].legend(frameon=False, fontsize=12)
    fig3.suptitle(rf"{_name()} — |dropped| / (|leading| + |dropped|), "
                  f"N = {P.N}, {nrep} SSA runs", fontsize=13)
    fig3.tight_layout(rect=[0, 0, 1, .94])
    _save(fig3, "closure_relative_error_N42.png")


# ======================================================================
#  decayapprox.m:  approx_comp_g0, approx_comp_g1
# ======================================================================


def decay(quick=False):
    """Exact mean-field Delta N(t) vs the slow-mode exponential.  The repo's
    rate comes from the one-cluster slow manifold; here it is the slowest
    antisymmetric eigenvalue of the full chain at the symmetric state, and
    the amplitude is fitted to the tail (the repo set it by hand)."""
    print("decayapprox: slow_decay_vs_exact_N42")
    P = Pn(42)
    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    for ax, g, col, xmax in zip(axs, (0.0, 1.0), (CLR[0], CLR[2]),
                                (100.0, 200.0)):
        lam = -sym_anti_eigs(P, g)[1]
        _, mean = pl.newborn_pool(P, g)
        tl = np.linspace(0, 40 / lam, 4001)
        _, Y = pl.chain_solve(P, g, tl[-1], mean, ts=tl)
        dN = pl.sum_diff(Y, g)["dN"]
        a = dN / dN[0]
        sel = (a < .3) & (a > 1e-3)
        amp = np.exp(np.mean(np.log(dN[sel]) + lam * tl[sel]))
        t = np.linspace(0, xmax * 2, 2001)
        exact = np.interp(t, tl, dN)
        tau = 1 / lam
        print(f"  gamma={g:.0f}: slowest antisymmetric rate {lam:.5f} "
              f"(tau {tau:.1f}, t1/2 {np.log(2) * tau:.1f}); fitted amplitude "
              f"{amp:.2f} of Delta N(0) {dN[0]:.2f}; Delta N(50) {np.interp(50, tl, dN):.2f}")
        ax.plot(t, exact, color=col, lw=2, label="exact (mean-field chain)")
        ax.plot(t, amp * np.exp(-lam * t), "k--", lw=2,
                label="slow mode  $A e^{-t/\\tau}$")
        yt = amp * np.exp(-1)
        ax.plot([tau, tau], [0, yt], color="#b3b3b3", lw=1.5)
        ax.plot([0, tau], [yt, yt], "--", color="#b3b3b3", lw=1.5)
        ax.text(tau, -.06 * P.N, r"$\tau$", ha="center", fontsize=14)
        yd = np.interp(50, tl, dN)
        ax.plot([50, 50], [0, yd], color="#b3b3b3", lw=1.5)
        ax.plot([0, 50], [yd, yd], "--", color="#b3b3b3", lw=1.5)
        ax.text(50, .02 * P.N, " division", fontsize=12)
        ax.set_xlim(0, xmax)
        ax.set_ylim(0, P.N)
        ax.set_xlabel("time (min)")
        ax.set_ylabel(r"$\Delta N$")
        ax.set_title(rf"$\gamma={g:g}$:  $\tau={tau:.1f}$ min", fontsize=14)
        ax.legend(frameon=False, fontsize=11)
        ax.grid(alpha=.25)
    fig.suptitle(f"{_name()} — pole imbalance vs slowest antisymmetric mode, "
                 f"N = {P.N}, newborn start", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    _save(fig, "slow_decay_vs_exact_N42.png")


# ======================================================================
#  percentfree.py:  1compartment_comp, 2compartment_comp
# ======================================================================

NVALS_FREE = np.linspace(5, 200, 20).astype(int)


def _free_job(a):
    N, g = a
    T = 1e4
    out = []
    for rho in (0.0, 0.13):
        P = Pn(N, rho=rho)
        c0 = pl.one_cluster_ic(P)                      # one N-cluster, pole 1
        ts = np.array([0.0, T])
        _, Y = pl.chain_solve(P, g, T, c0, ts=ts, method="BDF")
        fm = Y[-1, :, 1].sum() / N
        c1 = np.zeros((2, N + 1))
        c1[0, 1] = N                                   # net bound: all free
        one = pl.netbound_solve(P, g, ts, c1)        # Kat's meanfieldode.m
        out += [fm, one["f"][-1].sum() / N]
    return out                 # [1p multi, 1p net, 2p multi, 2p net]


def percent_free(quick=False):
    print("percentfree: fraction_free_vs_N_one_pole / two_poles")
    nv = NVALS_FREE[::2] if quick else NVALS_FREE
    jobs = [(N, g) for g in G4 for N in nv]
    d = pl.cached("kfree", dict(model=MODEL, N=list(map(int, nv)), g=G4,
                                v=3), lambda: dict(r=np.array(_pool(_free_job,
                                                                    jobs))))
    r = d["r"].reshape(len(G4), len(nv), 4)
    for k, (fname, ttl) in enumerate([
            ("fraction_free_vs_N_one_pole.png", "one pole"),
            ("fraction_free_vs_N_two_poles.png",
             r"two poles (total free), $\rho=0.13$")]):
        fig, axs = plt.subplots(1, 4, figsize=(19, 4.4), sharey=True)
        for i, (ax, g, col, lab) in enumerate(zip(axs, G4, CLR, G4_LAB)):
            ax.plot(nv, r[i, :, 2 * k], "o-", color=col, label="multi cluster")
            ax.plot(nv, r[i, :, 2 * k + 1], "s--", color=col, mfc="white",
                    label="net bound (Kat's ODE)")
            ax.set_title(lab, fontsize=14)
            ax.set_xlabel(r"$N_{tot}$")
            ax.grid(alpha=.25)
            ax.legend(frameon=False)
        axs[0].set_ylabel(r"fraction free plasmids ($f/N_{tot}$)")
        fig.suptitle(f"{_name()} — fraction free vs $N_{{tot}}$, {ttl}: "
                     f"steady state at t = 10$^4$ "
                     r"($\beta=0.015$, $\kappa=0.08$"
                     + (r", $\beta_m=0.015$" if MODEL == "smol" else "") + ")",
                     fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, .92])
        _save(fig, fname)
        for i, g in enumerate(G4):
            print(f"  {ttl[:9]:9s} gamma={g:.3f}  N={nv[0]}: multi "
                  f"{r[i, 0, 2 * k]:.3f} net {r[i, 0, 2 * k + 1]:.3f}   "
                  f"N={nv[-1]}: multi {r[i, -1, 2 * k]:.3f} net "
                  f"{r[i, -1, 2 * k + 1]:.3f}")


# ======================================================================
#  mc_evals.py:  mc_eigval (+ symmetric / antisymmetric panels)
# ======================================================================


def eigenvalues(quick=False):
    print("mc_evals: leading_eigenvalue_vs_gamma, sym_antisym_eigenvalues")
    nv = np.linspace(10, 200, 6 if quick else 11, dtype=int)
    gv = np.linspace(0, 3, 10 if quick else 19)
    jobs = [(N, g) for N in nv for g in gv]
    d = pl.cached("keig", dict(model=MODEL, N=list(map(int, nv)),
                               g=list(gv), v=1),
                  lambda: dict(r=np.array(_pool(_eig_job, jobs))))
    r = d["r"].reshape(len(nv), len(gv), 2)
    cmap, norm = plt.cm.viridis, plt.Normalize(nv.min(), nv.max())
    for i, N in enumerate(nv):
        la = r[i, :, 1]
        cross = np.where(np.diff(np.sign(la)) > 0)[0]
        gs = gv[cross[0]] if len(cross) else np.nan
        print(f"  N={N:3d}: antisymmetric eigenvalue at gamma=0 {la[0]:+.4f},"
              f" changes sign between gamma {gs:.2f} and "
              f"{gs + gv[1] - gv[0]:.2f}" if len(cross) else
              f"  N={N:3d}: antisymmetric eigenvalue stays negative "
              f"(max {la.max():+.4f})")

    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    for i, N in enumerate(nv):
        for k in (0, 1):
            axs[k].plot(gv, r[i, :, k], color=cmap(norm(N)), lw=2, marker="o",
                        ms=3)
    for ax, ttl in zip(axs, ["Symmetric perturbations",
                             "Antisymmetric perturbations"]):
        ax.set_title(ttl, fontsize=12)
        ax.set_xlabel(r"$\gamma$", fontsize=13)
        ax.axhline(0, color="k", lw=.8, ls=":")
        ax.grid(alpha=.25)
    axs[0].set_ylabel("Leading eigenvalue (Re)", fontsize=12)
    note = ("" if MODEL == "smol" else "\n(detailed balance: all "
            "eigenvalues are negative; values that reach 0 at large "
            r"$\gamma$ are below round-off)")
    fig.suptitle(f"{_name()} — eigenvalues at the symmetric steady state"
                 + note, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=axs,
                 label="N (total plasmids)", shrink=.85)
    _save(fig, "sym_antisym_eigenvalues_vs_gamma.png")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for i, N in enumerate(nv):
        ax.plot(gv, r[i].max(1), color=cmap(norm(N)), lw=2)
    ax.set_title(f"{_name()} — overall leading eigenvalue", fontsize=12)
    ax.set_xlabel(r"$\gamma$", fontsize=13)
    ax.set_ylabel("Leading eigenvalue (Re)", fontsize=12)
    ax.axhline(0, color="k", lw=.8, ls=":")
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax,
                 label="N (total plasmids)", shrink=.85)
    _save(fig, "leading_eigenvalue_vs_gamma.png")


# ======================================================================
#  mc_pseudo_bif.py
# ======================================================================


def _pb_job(a):
    g, pole = a
    P = Pn(25)
    _, Y = pl.chain_solve(P, g, 1e4, pl.one_cluster_ic(P, pole),
                          ts=np.array([0.0, 1e4]), method="BDF")
    return pl.sum_diff(Y, g)["dN"][-1]


def pseudo_bif(quick=False):
    print("mc_pseudo_bif: pseudo_bifurcation_N25")
    gam = np.unique(np.concatenate([np.linspace(0, 3, 20),
                                    np.linspace(0, 1, 4)]))
    d = pl.cached("kpb", dict(model=MODEL, g=list(gam), v=1), lambda: dict(
        r=np.array(_pool(_pb_job, [(g, p) for p in (0, 1) for g in gam]))))
    r = d["r"].reshape(2, len(gam))
    gstar = pitchfork_gamma(Pn(25))
    print(f"  pitchfork of the symmetric state at gamma* = {gstar:.4f}")
    for g, a in zip(gam, r[0]):
        print(f"    gamma={g:.3f}  dN(1e4) {a:+.2f}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gam, r[0], "o-", lw=1.8, ms=6, color="#2166ac",
            label="IC: all in pole 1")
    ax.plot(gam, r[1], "s-", lw=1.8, ms=6, color="#d6604d",
            label="IC: all in pole 2")
    ax.axhline(0, color="k", lw=.8, ls="--", alpha=.5)
    if np.isfinite(gstar):
        ax.axvline(gstar, color="#888888", lw=1, ls=":")
        ax.text(gstar, .9 * 25, rf" pitchfork $\gamma^*={gstar:.2f}$",
                fontsize=10, color="#555555")
    else:
        ax.text(.02, .04, "no pitchfork: the symmetric state is the only "
                "steady state (detailed balance);\nnonzero $\\Delta N$ here "
                "is a transient that has not relaxed by $t=10^4$",
                transform=ax.transAxes, fontsize=9, color="#555555")
    ax.set_xlabel(r"$\gamma$", fontsize=12)
    ax.set_ylabel(r"$\Delta N = N_1 - N_2$ at $t=10^4$", fontsize=12)
    ax.set_ylim(-27, 27)
    ax.legend(fontsize=10)
    ax.grid(alpha=.3)
    ax.set_title(f"{_name()} mean-field, N = 25, start: one N-cluster",
                 fontsize=12)
    fig.tight_layout()
    _save(fig, "pseudo_bifurcation_N25.png")


# ======================================================================
#  bif_plots.m:  bifurcation diagram and pitchfork locus (computed here)
# ======================================================================


def pitchfork_gamma(P, glo=0.0, ghi=4.0, n=41):
    """gamma at which the antisymmetric eigenvalue of the symmetric state
    first crosses zero (NaN if it does not in [glo, ghi]).

    Becker-Doring: always NaN.  Every steady state has J_j = 0 for all j
    (J_N = 0), i.e. detailed balance in each pole, and then the f equations
    give f_1 = f_2: the symmetric state is the only steady state and, the
    network being reversible and detailed balanced, it is stable for every
    gamma.  Its antisymmetric eigenvalue only becomes too small to resolve.
    """
    if not P.smol:
        return np.nan
    gs = np.linspace(glo, ghi, n)
    la = [sym_anti_eigs(P, g)[1] for g in gs]
    for k in range(n - 1):
        if la[k] < 0 <= la[k + 1]:
            return brentq(lambda g: sym_anti_eigs(P, g)[1], gs[k], gs[k + 1],
                          xtol=1e-6)
    return np.nan


def _stable(P, x, g):
    """Largest real eigenvalue on the fixed-mass manifold."""
    N = P.N
    J = jac(P, g, _to_c(P, x))
    w = np.r_[np.arange(1, N + 1), np.arange(1, N + 1)].astype(float)
    Q = null_space(w[None])
    return np.linalg.eigvals(Q.T @ J @ Q).real.max()


def _to_c(P, x):
    c = np.zeros(2 * (P.N + 1))
    c[_active(P.N)] = x
    return c.reshape(2, P.N + 1)


def _dN(P, x):
    j = np.arange(1, P.N + 1)
    return (j * x[:P.N]).sum() - (j * x[P.N:]).sum()


def _H(P, u, g):
    """Fixed-point residual in log variables u = log c (active coordinates):
    rows d(log c_j)/dt, with the f_1 row replaced by the relative mass
    error.  All rows are O(rate), whatever the size of c_j."""
    N = P.N
    x = np.exp(u)
    jj, w = pl.weights(N, g)
    y = np.zeros(2 * (N + 1), dtype=np.result_type(u, g))
    act = _active(N)
    y[act] = x
    r = pl.chain_rhs(0, y, P, g, w, jj)[act] / x
    r[0] = ((np.r_[np.arange(1, N + 1), np.arange(1, N + 1)] * x).sum()
            - N) / N
    return r


def _Hjac(P, u, g):
    n, h = len(u), 1e-30
    J = np.zeros((n, n + 1))
    for i in range(n):
        uc = u.astype(complex)
        uc[i] += 1j * h
        J[:, i] = np.imag(_H(P, uc, g)) / h
    J[:, n] = np.imag(_H(P, u.astype(complex), g + 1j * h)) / h
    return J


def _log_rhs(t, u, P, g, w, jj, act):
    x = np.exp(u)
    y = np.zeros(2 * (P.N + 1))
    y[act] = x
    return pl.chain_rhs(0, y, P, g, w, jj)[act] / x


def relax_log(P, g, c0, T):
    """Integrate the chain in log variables (no absolute-tolerance floor, so
    nearly empty slots are resolved); returns the active state at T."""
    jj, w = pl.weights(P.N, g)
    act = _active(P.N)
    u0 = np.log(np.maximum(np.asarray(c0, float).ravel()[act], 1e-60))
    s = solve_ivp(_log_rhs, (0, T), u0, method="BDF",
                  args=(P, g, w, jj, act), rtol=1e-10, atol=1e-10)
    return np.exp(s.y[:, -1])


GW = 20.0          # weight of gamma in the arclength (u changes are O(10))


def _corrector(P, yp, tan, maxit=20, umax=1.0):
    """Damped Newton for H(y) = 0, tan . (y - yp) = 0 (y = (u, gamma));
    a step that moves any log-concentration by more than umax is refused,
    so the corrector cannot jump to another branch."""
    n = len(yp) - 1
    y = yp.copy()
    W = np.r_[np.ones(n), GW]
    for it in range(maxit):
        J = _Hjac(P, y[:n], y[n])
        F = np.r_[_H(P, y[:n], y[n]), (W * tan) @ (y - yp)]
        dy = np.linalg.solve(np.vstack([J, W * tan]), -F)
        if np.abs(dy[:n]).max() > umax or not np.all(np.isfinite(dy)):
            return None, it
        y = y + dy
        if np.abs(dy[:n]).max() < 1e-9 and \
                np.abs(_H(P, y[:n], y[n])).max() < 1e-10:
            return y, it
    return None, maxit


def continue_branch(P, x0, g0, direction=-1.0, ds=0.05, gmin=0.0, gmax=2.6,
                    max_steps=4000):
    """Pseudo-arclength continuation of fixed points in gamma (log
    variables), from (x0, g0) with gamma initially moving in `direction`.
    Stops on leaving [gmin, gmax] or on reaching the symmetric branch."""
    n = len(x0)
    W = np.r_[np.ones(n), GW]
    u = np.log(np.maximum(x0, 1e-300))
    for _ in range(30):                        # polish at fixed gamma
        J = _Hjac(P, u, g0)[:, :n]
        du = np.linalg.solve(J, -_H(P, u, g0))
        u += np.clip(du, -1, 1)
        if np.abs(du).max() < 1e-11:
            break
    y = np.r_[u, g0]

    def tangent(y, prev):
        J = _Hjac(P, y[:n], y[n])
        t = np.linalg.solve(np.vstack([J, W * prev]), np.r_[np.zeros(n), 1.0])
        return t / np.sqrt((W * t) @ t)

    tan = tangent(y, np.r_[np.zeros(n), 1.0])
    tan *= direction * np.sign(tan[n])
    pts = [y.copy()]
    for _ in range(max_steps):
        for _try in range(30):
            yc, it = _corrector(P, y + ds * tan, tan)
            if yc is not None:
                break
            ds *= .5
        else:
            print(f"    continuation: corrector failed at gamma {y[n]:.4f}")
            break
        t_new = tangent(yc, tan)
        if (W * t_new) @ tan < 0:
            t_new = -t_new
        y, tan = yc, t_new
        pts.append(y.copy())
        ds = min(ds * (1.5 if it < 4 else .7), 2.0)
        if not (gmin <= y[n] <= gmax) or abs(_dN(P, np.exp(y[:n]))) < 1e-3:
            break
    pts = np.array(pts)
    return np.c_[np.exp(pts[:, :n]), pts[:, n]]


TDIV = 50.0        # cell-cycle time (min) marked in decayapprox.m


def bifurcation(quick=False):
    print("bif_plots: bifurcation_dN_vs_gamma_N50")
    P = Pn(50)
    N = P.N
    gmax = 2.5
    gs = np.linspace(0, gmax, 51 if quick else 126)

    def compute():
        sym = np.array(_pool(_eig_job, [(N, g) for g in gs]))
        out = dict(gs=gs, sym=sym)
        if P.smol:
            out["gstar"] = np.array(pitchfork_gamma(P, 0.5, gmax, 21))
            x0 = relax_log(P, gmax, pl.one_cluster_ic(P), 1e4)
            br = continue_branch(P, x0, gmax, direction=-1.0, gmax=gmax + .05)
            out["g"] = br[:, -1]
            out["dN"] = np.array([_dN(P, y[:-1]) for y in br])
            out["stab"] = np.array([_stable(P, y[:-1], y[-1]) for y in br])
        else:                    # no asymmetric steady states: transients
            Ts = np.array([TDIV, 1e3, 1e4])
            dn = []
            for g in gs[::5]:
                _, Y = pl.chain_solve(P, g, Ts[-1], pl.one_cluster_ic(P),
                                      ts=np.r_[0, Ts], method="BDF")
                dn.append(pl.sum_diff(Y, g)["dN"][1:])
            out.update(Ts=Ts, gt=gs[::5], dNt=np.array(dn))
        return out

    d = pl.cached("kbif", dict(model=MODEL, N=N, gmax=gmax, n=len(gs), v=3),
                  compute)
    lam_a = d["sym"][:, 1]
    if P.smol:
        gstar, g, dN, stab = float(d["gstar"]), d["g"], d["dN"], d["stab"]
        print(f"  pitchfork gamma* = {gstar:.4f}; asymmetric branch: "
              f"{len(g)} points, gamma {g.min():.4f}..{g.max():.3f}, "
              f"stable points {np.mean(stab < 0):.0%}")
        for gg in (1.25, 1.5, 2.0, 2.5):
            k = np.argmin(np.abs(g - gg) + 1e3 * (dN < 0))
            print(f"    gamma={g[k]:.3f}: dN = {dN[k]:+.2f}, leading "
                  f"eigenvalue {stab[k]:+.2e}")
        fig, ax = plt.subplots(figsize=(8, 5.5))
        s = stab < 0
        for mask, col, ls in ((s, "r", "-"), (~s, "k", "--")):
            ax.plot(g, np.where(mask, dN, np.nan), color=col, ls=ls, lw=1.8)
        ss = d["sym"].max(1) < 0
        ax.plot(gs, np.where(ss, 0, np.nan), "r-", lw=1.8)
        ax.plot(gs, np.where(~ss, 0, np.nan), "k--", lw=1.8)
        ax.plot([gstar], [0], "o", color="#444444", ms=5)
        ax.annotate(rf"$\gamma^*={gstar:.3f}$", (gstar, 0), (gstar + .15, -12),
                    fontsize=11, arrowprops=dict(arrowstyle="-", lw=.6))
        # second sign change of lambda_anti: the symmetric state becomes
        # (barely) stable again; |lambda_anti| < 1e-5 per min around it
        k2 = np.where((gs[:-1] > gstar + .1) & (lam_a[:-1] > 0)
                      & (lam_a[1:] <= 0))[0]
        if len(k2):
            g2 = gs[k2[0]] - lam_a[k2[0]] * (gs[1] - gs[0]) / (
                lam_a[k2[0] + 1] - lam_a[k2[0]])
            print(f"  symmetric state re-stabilises at gamma2 ~ {g2:.3f}; "
                  f"max lambda_anti on [gamma*, gamma2] "
                  f"{lam_a[(gs > gstar) & (gs < g2)].max():.1e}, at gamma 2: "
                  f"{np.interp(2.0, gs, lam_a):.1e}")
            ax.plot([g2], [0], "o", color="#444444", ms=5)
            ax.annotate(rf"$\gamma_2={g2:.2f}$: symmetric state stable again,"
                        "\nbut nearly neutral (|$\\lambda_{anti}$| < "
                        "$10^{-5}$ min$^{-1}$);\nthe branch it emits was not "
                        "resolved", (g2, 0), (1.45, 14), fontsize=9,
                        color="#444444",
                        arrowprops=dict(arrowstyle="-", lw=.6))
        ax.plot([], [], "r-", lw=1.8, label="stable")
        ax.plot([], [], "k--", lw=1.8, label="unstable")
        ax.set_xlabel(r"$\gamma$", fontsize=14)
        ax.set_ylabel(r"$\Delta N$", fontsize=14)
        ax.set_xlim(0, gmax)
        ax.set_ylim(-55, 55)
        ax.grid(alpha=.3)
        ax.legend(frameon=False, fontsize=12, loc="upper left")
        ax.set_title(f"{_name()} mean-field steady states, N = {N} "
                     "(pseudo-arclength continuation)", fontsize=12)
        fig.tight_layout()
        _save(fig, "bifurcation_dN_vs_gamma_N50.png")
        return

    # Becker-Doring: the symmetric state is the only steady state
    J0 = max(np.abs(jac(P, g, sym_state(P, g))).max() for g in (0.0, gmax))
    res = 1e-12 * J0
    tau = np.where(lam_a < -res, -1 / np.minimum(lam_a, -1e-300), np.nan)
    ok = np.isfinite(tau)
    print(f"  no asymmetric steady states (detailed balance); slowest "
          f"antisymmetric time tau: gamma 0 {tau[0]:.1f}, 1 "
          f"{np.interp(1, gs[ok], tau[ok]):.3g}; resolvable up to gamma "
          f"{gs[ok].max():.2f} (|lambda| > {res:.0e})")
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.3))
    ax = axs[0]
    ax.plot(gs, 0 * gs, "r-", lw=2, label="steady state (unique, stable)")
    for k, (T, mk, col) in enumerate(zip(d["Ts"], "o^s",
                                         ("#bbbbbb", "#888888", "#444444"))):
        lab = rf"$\Delta N$ at $t={T:g}$ (transient)"
        ax.plot(d["gt"], d["dNt"][:, k], mk, color=col, ms=5, label=lab)
        ax.plot(d["gt"], -d["dNt"][:, k], mk, color=col, ms=5, mfc="white")
    ax.set_xlabel(r"$\gamma$", fontsize=14)
    ax.set_ylabel(r"$\Delta N$", fontsize=14)
    ax.set_xlim(0, gmax)
    ax.set_ylim(-55, 55)
    ax.grid(alpha=.3)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.set_title(r"(a) steady states, and $\Delta N(t)$ from one $N$-cluster"
                 " in pole 1 (filled) / pole 2 (open)", fontsize=11)
    ax = axs[1]
    ax.semilogy(gs[ok], tau[ok], "-", color="#2c4fa8", lw=2)
    ax.axhline(TDIV, color="#888888", ls="--", lw=1.2)
    ax.text(.05, TDIV * 1.3, f"cell cycle ({TDIV:g} min)", fontsize=10,
            color="#555555")
    if (~ok).any():
        ax.axvspan(gs[ok].max(), gmax, color="#eeeeee")
        ax.text(gs[ok].max() + .03, tau[ok].max() / 30,
                "below\nround-off", fontsize=9, color="#777777")
    ax.set_xlabel(r"$\gamma$", fontsize=14)
    ax.set_ylabel(r"pole-balancing time $\tau=1/|\lambda_{anti}|$ (min)",
                  fontsize=12)
    ax.set_xlim(0, gmax)
    ax.grid(alpha=.3, which="both")
    ax.set_title("(b) slowest antisymmetric relaxation time of the "
                 "symmetric state", fontsize=11)
    fig.suptitle(f"{_name()}, N = {N}: no pitchfork — detailed balance makes "
                 r"the symmetric state the only steady state; for large "
                 r"$\gamma$ it is reached only after a very long time",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, .93])
    _save(fig, "bifurcation_dN_vs_gamma_N50.png")


def _one_cluster_locus(Nt, beta=0.015, kappa=0.08):
    """bif_plots.m's implicit pitchfork curve of the net-bound model."""
    def res(x):
        Ft, g = x
        h = ((Nt - Ft) / 2) ** g
        return [-beta * Ft * h * g + kappa * (Nt - Ft), (g - 1) * h - Ft / 2]
    with np.errstate(all="ignore"):
        sol, info, flag, _ = fsolve(res, [0.5 * Nt, 1.5], full_output=True,
                                    xtol=1e-12)
    return sol[1] if flag == 1 and sol[0] > 0 and sol[1] > 1 else np.nan


def pitchfork_locus(quick=False):
    """(gamma, N) map of the antisymmetric eigenvalue of the symmetric state:
    its zero contour is the pitchfork (Smoluchowski only), its -1/TDIV
    contour is where pole balancing takes one cell cycle."""
    print("bif_plots: pitchfork_locus_N_vs_gamma")
    nv = np.arange(4, 81, 4 if quick else 2)
    gv = np.linspace(0, 3, 31 if quick else 61)
    jobs = [(N, g) for N in nv for g in gv]
    d = pl.cached("kpfl", dict(model=MODEL, N=list(map(int, nv)),
                               g=list(gv), v=2),
                  lambda: dict(r=np.array(_pool(_eig_job, jobs))))
    la = d["r"][:, 1].reshape(len(nv), len(gv))
    Nt = np.linspace(0.9, 80, 500)
    g1 = np.array([_one_cluster_locus(x) for x in Nt])
    fig, ax = plt.subplots(figsize=(8, 5.5))
    G, NN = np.meshgrid(gv, nv)
    hs, ls = [], []
    if MODEL == "smol":
        cs = ax.contour(G, NN, la, [0.0], colors="c", linewidths=2.4)
        ax.annotate("second crossing: symmetric state\nstable again, "
                    "nearly neutral\n(|$\\lambda_{anti}$| < $10^{-5}$ "
                    "min$^{-1}$)", (2.47, 62), (1.75, 70), fontsize=8.5,
                    color="#407f7f", arrowprops=dict(arrowstyle="-", lw=.6,
                                                     color="#407f7f"))
        hs.append(plt.Line2D([], [], color="c", lw=2.4))
        ls.append(r"pitchfork ($\lambda_{anti}=0$), "
                  f"{_name()} mean-field chain")
        for N in (10, 20, 40, 60, 80):
            i = np.argmin(np.abs(nv - N))
            row = la[i]
            k = np.where((row[:-1] < 0) & (row[1:] >= 0))[0]
            gx = (gv[k[0]] - row[k[0]] * (gv[1] - gv[0]) / (row[k[0] + 1]
                  - row[k[0]])) if len(k) else np.nan
            print(f"  N={nv[i]:3d}: pitchfork gamma* ~ {gx:.3f}")
    ax.contour(G, NN, la, [-1 / TDIV], colors="#2c4fa8", linewidths=1.6,
               linestyles="--")
    hs.append(plt.Line2D([], [], color="#2c4fa8", lw=1.6, ls="--"))
    ls.append(rf"pole balancing takes one cell cycle ($\tau={TDIV:g}$ min)")
    ax.plot(g1, Nt, ":", color="#777777", lw=1.8)
    hs.append(plt.Line2D([], [], color="#777777", lw=1.8, ls=":"))
    ls.append("net-bound (one-cluster) pitchfork, bif_plots.m")
    ax.text(.12, 15, "symmetric plasmid distribution", fontsize=13)
    ax.text(1.5, 50, "plasmids remain in old pole", fontsize=13)
    ax.set_xlabel(r"$\gamma$", fontsize=14)
    ax.set_ylabel(r"$N_{tot}$", fontsize=14)
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 80)
    ax.grid(alpha=.3)
    ax.legend(hs, ls, frameon=True, framealpha=.9, fontsize=9,
              loc="upper left")
    ttl = (f"{_name()}: pitchfork of the symmetric state" if MODEL == "smol"
           else f"{_name()}: no pitchfork (detailed balance); contour of the "
           "pole-balancing time")
    ax.set_title(ttl, fontsize=11.5)
    fig.tight_layout()
    _save(fig, "pitchfork_locus_N_vs_gamma.png")
    for N in (10, 20, 40, 60, 80):
        i = np.argmin(np.abs(nv - N))
        row = la[i] + 1 / TDIV
        k = np.where((row[:-1] < 0) & (row[1:] >= 0))[0]
        gx = gv[k[0]] if len(k) else np.nan
        print(f"  N={nv[i]:3d}: tau reaches {TDIV:g} min at gamma ~ {gx:.2f}")


# ======================================================================

FIGS = {"odesolve": odesolve, "closure": closure, "decay": decay,
        "percent_free": percent_free, "eigenvalues": eigenvalues,
        "pseudo_bif": pseudo_bif, "bifurcation": bifurcation,
        "pitchfork_locus": pitchfork_locus}


def main(model, here):
    global MODEL, OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", nargs="+", choices=list(FIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    if not (args.fig or args.all):
        ap.print_help()
        return
    MODEL = model
    OUT = os.path.join(here, "katlynch_figures")
    pl.CACHE_DIR = os.path.join(here, "cache")
    pl.RECOMPUTE = args.recompute
    t0 = time.time()
    for k in (list(FIGS) if args.all else args.fig):
        t1 = time.time()
        FIGS[k](quick=args.quick)
        print(f"  [{time.time() - t1:.1f} s]", flush=True)
    print(f"\ntotal {time.time() - t0:.1f} s")
