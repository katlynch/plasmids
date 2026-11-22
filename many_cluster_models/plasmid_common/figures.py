"""
figures.py
==========
The part4 figure set, made for either form of the two-pole model:

  model 'bd'    Becker-Doering (Part IV)      -> ../beckerdoring/figures
  model 'smol'  Smoluchowski   (Part V)       -> ../smoluchowski/figures

Rates and initial conditions are those of part4/fig.py and part4/reduced.py:
N = 60, beta = 0.015, kappa = 0.08, rho = 0.13 (beta_m = beta = 0.015);
detachment per the notes (dimer dissolution kappa c_2).

  figure name                              part4 original   start
  ---------------------------------------  ---------------  -----------------
  ssa_vs_meanfield_time_course.png         fig5             newborn cell
  plasmids_per_cluster_size_t1000.png/.csv fig5b            newborn cell
  meanfield_error_ssa_minus_ode.png        fig5d            newborn cell
  chain_vs_kat_netbound.png                fig5e            newborn cell (vs net-bound)
  pole_imbalance_vs_gamma_ssa.png          fig6             newborn, mirrored
  pole_sum_difference_time_course.png      fig7             one N-cluster, pole 1
  pole_imbalance_vs_gamma_meanfield.png    fig8             one N-cluster, mirrored
  pole_sum_difference_gamma2_long_run.png  fig10            one N-cluster, pole 1

Newborn cell: all N plasmids in pole 1, drawn from that pole's stationary
cluster distribution, pole 2 empty; the ODE starts from the mean of those
draws.  See plasmid.newborn_pool.

Run through ../beckerdoring/make_figures.py or ../smoluchowski/make_figures.py.
"""
from __future__ import annotations

import argparse
import functools
import os
import time

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns

import plasmid as pl

OUT = "figures"
GAMMAS = [0.0, 2.0 / 3.0, 1.0]
LABELS = [r"$\gamma=0$", r"$\gamma=2/3$", r"$\gamma=1$"]
COLORS = ["#c0392b", "#2c4fa8", "#2e8b3d"]
POLE_COL = ["#2b2b2b", "#d9822b"]
P = pl.Params()                         # set in main()


def _save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def _half_time(t, d):
    i = np.argmax(d <= d[0] / 2)
    return t[i] if d[i] <= d[0] / 2 else np.nan


def _ic_note():
    return ("newborn start: pole 1 holds all N at its stationary law"
            + (f" (burn-in t={pl.T_BURN:g})" if P.smol else "")
            + ", pole 2 empty")


# ======================================================================
#  Shared runs: SSA ensemble + mean-field chain, newborn start
# ======================================================================

TC = dict(T=1000.0, NREC=601, SEED=8080)


@functools.lru_cache(None)
def _runs(quick):
    nrep = 150 if quick else 500
    runs = []
    for g in GAMMAS:
        draws, mean = pl.newborn_pool(P, g)
        e = pl.ensemble(P, g, TC["T"], TC["NREC"], draws[:nrep], TC["SEED"])
        _, Y = pl.chain_solve(P, g, TC["T"], mean, ts=e["t"])
        M = pl.moments(Y, g)
        e.update(fd=M["f"], bd=M["b"], Cd=M["C"], Y=Y, c0=mean)
        runs.append(e)
    return runs, nrep


# ======================================================================
#  SSA vs mean-field time course  (part4 fig5)
# ======================================================================


def time_course(quick=False):
    print("ssa_vs_meanfield_time_course: gamma effect at fixed rates")
    T = TC["T"]
    runs, nrep = _runs(quick)
    steady = []
    for g, e in zip(GAMMAS, runs):
        st = pl.moments(pl.mf_equilibrium(P, g), g)
        steady.append({k: st[k][0] for k in ("f", "b", "C")})
        print(f"  gamma={g:.3f}  pole 1 at t=T: f SSA {e['f'][:, -1, 0].mean():5.2f}"
              f" ODE {e['fd'][-1, 0]:5.2f} | C SSA {e['C'][:, -1, 0].mean():5.2f}"
              f" ODE {e['Cd'][-1, 0]:5.2f} | steady state f* {steady[-1]['f']:5.2f}"
              f" C* {steady[-1]['C']:5.2f}")

    ode_kw = dict(color="#111111", lw=1.1, dashes=(4, 2.5), zorder=4)
    with sns.axes_style("ticks"), sns.plotting_context("notebook"):
        fig, axs = plt.subplots(2, 3, figsize=(15, 8.4), sharex=True)
        for p in (0, 1):
            q = p + 1
            for j, (ks, kd, ttl) in enumerate([
                    ("f", "fd", rf"free plasmids  $f_{q}$"),
                    ("b", "bd", rf"bound plasmids  $b_{q}$"),
                    ("C", "Cd", rf"number of clusters  $C_{q}$")]):
                ax = axs[p, j]
                for r, col, st in zip(runs, COLORS, steady):
                    X = r[ks][:, :, p]
                    mu = X.mean(0)
                    se = X.std(0, ddof=1) / np.sqrt(X.shape[0])
                    ax.fill_between(r["t"], mu - 2 * se, mu + 2 * se,
                                    color=col, alpha=.3, lw=0, zorder=1)
                    ax.plot(r["t"], mu, color=col, lw=1.6, alpha=.9,
                            solid_capstyle="round", zorder=3)
                    ax.plot(r["t"], r[kd][:, p], **ode_kw)
                    ax.plot(T * 1.015, st[ks], marker="<", ms=8, color=col,
                            mec="white", mew=.8, clip_on=False, zorder=5)
                ax.set_title(f"({'abcdef'[3 * p + j]})  " + ttl, loc="left",
                             fontsize=12)
                ax.set_xlim(0, T)
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", color="#e6e6e6", lw=.8)
                ax.set_axisbelow(True)
                if p == 1:
                    ax.set_xlabel("time")
                if j == 0:
                    ax.set_ylabel(f"Pole {q}\ncount", fontsize=12)
                sns.despine(ax=ax)

        h = [plt.Line2D([], [], color=c, lw=2.6) for c in COLORS]
        labs = list(LABELS)
        h += [plt.Line2D([], [], color="#999999", lw=1.6),
              mpl.patches.Patch(color="#999999", alpha=.35, lw=0),
              plt.Line2D([], [], **{k: v for k, v in ode_kw.items()
                                    if k != "zorder"}),
              plt.Line2D([], [], color="#666666", marker="<", ms=8, lw=0)]
        labs += [f"SSA mean ({nrep} runs)", r"SSA mean $\pm$2 SE",
                 "mean-field ODE", "mean-field steady state"]
        fig.legend(h, labs, loc="upper center", ncol=7, frameon=False,
                   bbox_to_anchor=(.5, .96), fontsize=11, handlelength=2.2)
        fig.suptitle(rf"{P.name} — only $\gamma$ varies:  " + P.rates_tex(),
                     fontsize=14, y=.995)
        fig.text(.5, .005, _ic_note(), ha="center", fontsize=10,
                 color="#555555")
        fig.tight_layout(rect=[0, .015, 1, .935])
        _save(fig, "ssa_vs_meanfield_time_course.png")


# ======================================================================
#  Mean-field ODE error  (part4 fig5d)
# ======================================================================


def ode_error(quick=False):
    print("meanfield_error_ssa_minus_ode")
    runs, nrep = _runs(quick)
    rows = [("f", "fd", "free  $f$"), ("b", "bd", "bound  $b$"),
            ("C", "Cd", "clusters  $C$")]
    print("  ODE error at t > T/2:  mean(SSA - ODE) [SE] per pole, "
          "and share of time points with |SSA - ODE| > 2 SE")
    with sns.axes_style("ticks"), sns.plotting_context("notebook"):
        fig, axs = plt.subplots(3, 3, figsize=(14, 9.5), sharex=True,
                                sharey="row")
        for c, (r, lab, g) in enumerate(zip(runs, LABELS, GAMMAS)):
            t = r["t"]
            late = t > t[-1] / 2
            for i, (ks, kd, name) in enumerate(rows):
                ax = axs[i, c]
                ax.axhline(0, color="#888888", lw=.9, zorder=1)
                line = f"    gamma={g:.3f} {ks}:"
                for p in (0, 1):
                    X = r[ks][:, :, p]
                    se = X.std(0, ddof=1) / np.sqrt(X.shape[0])
                    d = X.mean(0) - r[kd][:, p]
                    ax.fill_between(t, d - 2 * se, d + 2 * se,
                                    color=POLE_COL[p], alpha=.18, lw=0,
                                    zorder=2)
                    ax.plot(t, d, color=POLE_COL[p], lw=1.2, zorder=3)
                    out = np.mean(np.abs(d[1:]) > 2 * se[1:])
                    line += (f"  pole {p + 1} {d[late].mean():+.3f}"
                             f" [{se[late].mean():.3f}], {out:4.0%} out")
                print(line)
                ax.grid(axis="y", color="#e6e6e6", lw=.8)
                ax.set_axisbelow(True)
                ax.set_xlim(0, t[-1])
                if i == 0:
                    ax.set_title(lab, fontsize=13, color=COLORS[c],
                                 fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"{name}\nSSA $-$ ODE", fontsize=12)
                if i == 2:
                    ax.set_xlabel("time")
                sns.despine(ax=ax)
        h = [plt.Line2D([], [], color=col, lw=1.2) for col in POLE_COL]
        h.append(mpl.patches.Patch(color="#888888", alpha=.3, lw=0))
        fig.legend(h, ["pole 1", "pole 2", r"$\pm$2 SE of SSA mean"],
                   loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(.5, .955), fontsize=11)
        fig.suptitle(f"{P.name} — mean-field ODE error: SSA mean ({nrep} runs)"
                     " $-$ ODE  (0 = exact agreement)", fontsize=14, y=.995)
        fig.tight_layout(rect=[0, 0, 1, .93])
        _save(fig, "meanfield_error_ssa_minus_ode.png")


# ======================================================================
#  Plasmids per cluster size at t = T  (part4 fig5b)
# ======================================================================


def cluster_sizes(quick=False):
    print("plasmids_per_cluster_size_t1000")
    runs, nrep = _runs(quick)
    N, T = P.N, TC["T"]
    js = np.arange(N + 1)
    rows, summary = [], []
    fig, axs = plt.subplots(2, 3, figsize=(16, 8.5), sharey="col")
    for c, (r, g, col, lab) in enumerate(zip(runs, GAMMAS, COLORS, LABELS)):
        cf = r["cfinal"]                                   # (nrep, 2, N+1)
        x_ssa = (cf * js).astype(float)
        x_ode = r["Y"][-1] * js
        jmax = int(max(js[2:][(x_ssa.mean((0, 1))[2:] > 0)
                              | (x_ode.max(0)[2:] > .1)].max(), 4))
        k = np.arange(2, jmax + 1)
        for p in (0, 1):
            ax = axs[p, c]
            mu = x_ssa[:, p].mean(0)
            se = x_ssa[:, p].std(0, ddof=1) / np.sqrt(len(cf))
            ax.bar(k, mu[k], .8, color=col, lw=0, label="SSA")
            ax.errorbar(k, mu[k], yerr=[np.minimum(2 * se[k], mu[k]), 2 * se[k]],
                        fmt="none", ecolor="#333333", elinewidth=.8,
                        capsize=1.5)
            ax.plot(k, x_ode[p, k], "o", ms=4.5, mfc="white", mec="#222222",
                    mew=1.2, label="ODE")
            fs, fd = cf[:, p, 1].mean(), r["Y"][-1, p, 1]
            bs, bd = mu[2:].sum(), x_ode[p, 2:].sum()
            ax.text(.97, .95, f"bound  SSA {bs:.1f} / ODE {bd:.1f}\n"
                    f"free  SSA {fs:.1f} / ODE {fd:.1f}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=9)
            ax.set_xlim(1, jmax + 1)
            ax.grid(axis="y", alpha=.25)
            ax.set_axisbelow(True)
            if p == 0:
                ax.set_title(lab, fontsize=13)
            else:
                ax.set_xlabel("cluster size $j$ (plasmids per cluster)")
            if c == 0:
                ax.set_ylabel(rf"plasmids in clusters of size $j$  "
                              rf"($j\,c_{{{p + 1},j}}$)")
                ax.text(-.25, .5, f"POLE {p + 1}", transform=ax.transAxes,
                        rotation=90, va="center", ha="center", fontsize=13,
                        fontweight="bold")
                if p == 0:
                    ax.legend(fontsize=9, frameon=False, loc="center right")
            for j in range(2, N + 1):
                if mu[j] > 0 or x_ode[p, j] > 1e-3:
                    rows.append((g, p + 1, j, mu[j], se[j], x_ode[p, j]))
            summary.append((g, p + 1, fs, fd, bs, bd))

    fig.suptitle(rf"{P.name} — plasmids in clusters of each size at $t={T:g}$,"
                 f" per pole (N = {N}, {nrep} runs; same runs as the time"
                 " course)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .95])
    _save(fig, "plasmids_per_cluster_size_t1000.png")

    path = os.path.join(OUT, "plasmids_per_cluster_size_t1000.csv")
    with open(path, "w") as fh:
        fh.write("gamma,pole,cluster_size,ssa_plasmids,ssa_se,ode_plasmids\n")
        for row in rows:
            fh.write("{:.4f},{},{},{:.4f},{:.4f},{:.4f}\n".format(*row))
    print(f"  -> {path}")
    for g, p, fs, fd, bs, bd in summary:
        print(f"  gamma={g:5.3f} pole {p}: free SSA {fs:5.1f} | ODE {fd:5.1f}"
              f"   bound SSA {bs:5.1f} | ODE {bd:5.1f}")


# ======================================================================
#  One-cluster ODE vs full mean-field  (part4 fig5e)
# ======================================================================


def one_cluster_vs_full(quick=False):
    print("chain_vs_kat_netbound")
    T = TC["T"]
    runs, _ = _runs(quick)
    with sns.axes_style("ticks"), sns.plotting_context("notebook"):
        fig, axs = plt.subplots(3, 3, figsize=(14, 10.6), sharex=True,
                                sharey="row")
        for c, (r, g, lab) in enumerate(zip(runs, GAMMAS, LABELS)):
            t = r["t"]
            red = pl.netbound_solve(P, g, t, r["c0"])
            print(f"  gamma={g:.3f}  t=T  full f1,f2,b1,b2 = "
                  f"{r['fd'][-1, 0]:.2f} {r['fd'][-1, 1]:.2f} "
                  f"{r['bd'][-1, 0]:.2f} {r['bd'][-1, 1]:.2f} | Kat net-bound "
                  f"{red['f'][-1, 0]:.2f} {red['f'][-1, 1]:.2f} "
                  f"{red['b'][-1, 0]:.2f} {red['b'][-1, 1]:.2f}")
            for i, (ks, kd, name) in enumerate([
                    ("f", "fd", "free plasmids  $f_p$"),
                    ("b", "bd", "bound plasmids  $b_p$")]):
                ax = axs[i, c]
                for p in (0, 1):
                    ax.plot(t, r[kd][:, p], color=POLE_COL[p], lw=2.2,
                            solid_capstyle="round", zorder=2)
                    ax.plot(t, red[ks][:, p], color=POLE_COL[p], lw=1.5,
                            dashes=(4, 2.5), zorder=3, path_effects=[
                                pe.withStroke(linewidth=3.4,
                                              foreground="white")])
                ax.set_xlim(0, T)
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", color="#e6e6e6", lw=.8)
                ax.set_axisbelow(True)
                if i == 0:
                    ax.set_title(lab, fontsize=13, color=COLORS[c],
                                 fontweight="bold")
                if c == 0:
                    ax.set_ylabel(f"{name}\ncount", fontsize=12)
                sns.despine(ax=ax)

            ax = axs[2, c]
            dfull = (r["fd"][:, 0] + r["bd"][:, 0]
                     - r["fd"][:, 1] - r["bd"][:, 1])
            dred = red["f"][:, 0] + red["b"][:, 0] - red["f"][:, 1] \
                - red["b"][:, 1]
            ax.semilogy(t, dfull, color=COLORS[c], lw=2.2, zorder=2)
            ax.semilogy(t, dred, color=COLORS[c], lw=1.5, dashes=(4, 2.5),
                        zorder=3, path_effects=[
                            pe.withStroke(linewidth=3.4, foreground="white")])
            ax.text(.97, .95, r"$t_{1/2}$:  full "
                    f"{_half_time(t, dfull):.0f},  net-bound "
                    f"{_half_time(t, dred):.0f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=10.5, color="#333333")
            ax.set_ylim(1e-2, 100)
            ax.grid(axis="y", color="#e6e6e6", lw=.8)
            ax.set_axisbelow(True)
            ax.set_xlabel("time")
            if c == 0:
                ax.set_ylabel("pole imbalance\n"
                              r"$\Delta N = N_1 - N_2$ (log)", fontsize=12)
            sns.despine(ax=ax)
        h = [plt.Line2D([], [], color=col, lw=2.4) for col in POLE_COL]
        h += [plt.Line2D([], [], color="#777777", lw=2.2),
              plt.Line2D([], [], color="#777777", lw=1.5, dashes=(4, 2.5))]
        fig.legend(h, ["pole 1", "pole 2", f"full mean-field ODE ({P.name})",
                       "Kat's net-bound model (meanfieldode.m)"],
                   loc="upper center", ncol=4, frameon=False,
                   bbox_to_anchor=(.5, .965), fontsize=11, handlelength=2.4)
        fig.suptitle(f"{P.name} mean-field chain vs Kat's net-bound model:  "
                     + P.rates_tex(), fontsize=14, y=.995)
        fig.tight_layout(rect=[0, 0, 1, .94])
        _save(fig, "chain_vs_kat_netbound.png")


# ======================================================================
#  Pole imbalance vs gamma, SSA  (part4 fig6)
# ======================================================================


def imbalance_ssa(quick=False):
    print("pole_imbalance_vs_gamma_ssa")
    T, NREC = 5000.0, 3001
    nrep = 60 if quick else 200
    gammas = np.unique(np.round(np.concatenate(
        [np.linspace(0.0, 2.0, 10 if quick else 13), GAMMAS]), 12))
    ics = [(0, "-"), (1, "--")]
    ts, dN, final = None, {0: [], 1: []}, {0: [], 1: []}
    for g in gammas:
        draws, _ = pl.newborn_pool(P, g)
        starts = {0: draws[:nrep], 1: pl.mirror(draws[nrep:2 * nrep])}
        for p0, _ in ics:
            e = pl.ensemble(P, g, T, NREC, starts[p0], seed=4242 + p0)
            Np = e["f"].mean(0) + e["b"].mean(0)
            d = Np[:, 0] - Np[:, 1]
            ts = e["t"]
            dN[p0].append(d)
            final[p0].append(d[-1])
        print(f"  gamma={g:.3f}  final dN(+N)={final[0][-1]:+7.3f}  "
              f"final dN(-N)={final[1][-1]:+7.3f}")

    fig, axs = plt.subplots(1, 2, figsize=(13, 5))
    ax = axs[0]
    for g, col, lab in zip(GAMMAS, COLORS, LABELS):
        i = int(np.argmin(np.abs(gammas - g)))
        for p0, ls in ics:
            ax.plot(ts, dN[p0][i], color=col, lw=1.4, ls=ls,
                    label=lab if p0 == 0 else None)
    ax.axhline(0, color="k", lw=.8, ls=":")
    ax.plot([], [], "k-", lw=1.4, label=r"start in pole 1  ($+N$)")
    ax.plot([], [], "k--", lw=1.4, label=r"start in pole 2  ($-N$)")
    ax.set_xlabel("time")
    ax.set_ylabel(r"$\Delta N = (f_1+b_1)-(f_2+b_2)$")
    ax.set_title(r"(a)  $\Delta N$ vs time (SSA mean)", fontsize=12)
    ax.set_xlim(0, T)
    ax.legend(fontsize=8, frameon=False, loc="upper right")

    ax = axs[1]
    ax.plot(gammas, final[0], "o-", color="#2c4fa8", lw=1.6, ms=5,
            label=r"start in pole 1  ($+N$)")
    ax.plot(gammas, final[1], "s--", color="#c0392b", lw=1.4, ms=4, alpha=.85,
            label=r"start in pole 2  ($-N$)")
    ax.axhline(0, color="k", lw=.8, ls=":")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(rf"$\Delta N$  at  $t={T:g}$")
    ax.set_title(r"(b)  $\Delta N$ vs $\gamma$", fontsize=12)
    ax.set_ylim(-P.N * 1.1, P.N * 1.1)
    ax.grid(alpha=.3)
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle(f"{P.name} — pole imbalance in total plasmids, "
                 + P.rates_tex() + rf" $-$ {len(gammas)} values of $\gamma$, "
                 f"mirror ICs, {nrep} SSA runs each", fontsize=12.5)
    fig.text(.5, -.01, _ic_note().replace("pole 1", "one pole")
             .replace(", pole 2 empty", ", the other empty"), ha="center",
             fontsize=10, color="#555555")
    fig.tight_layout(rect=[0, 0, 1, .93])
    _save(fig, "pole_imbalance_vs_gamma_ssa.png")


# ======================================================================
#  Pole sum / difference equations  (part4 fig7, fig10)
# ======================================================================

SD_STY = [("Ntot", "#222222", "-", r"$N_{tot}$"),
          ("dN", "#222222", "--", r"$\Delta N$"),
          ("Ftot", "#c0392b", "-", r"$F_{tot}$"),
          ("dF", "#c0392b", "--", r"$\Delta F$"),
          ("Ctot", "#2c4fa8", "-", r"$C_{tot}$"),
          ("dC", "#2c4fa8", "--", r"$\Delta C$")]


def _plot_sd(ax, t, red):
    for k, col, ls, lab in SD_STY:
        ax.plot(t, red[k], color=col, ls=ls, lw=1.8, label=lab)
    ax.axhline(0, color="k", lw=.6, ls=":")
    ax.grid(alpha=.3)


def _report_sd(g, red, full):
    err = max(np.abs(red[k] - full[k]).max()
              for k in ("dN", "dF", "Ftot", "dC", "Ctot"))
    print(f"  gamma={g:.3f}  max |reduced - chain| = {err:.1e}   t=T: "
          f"dN {red['dN'][-1]:.2f}  Ftot {red['Ftot'][-1]:.2f}  "
          f"Ctot {red['Ctot'][-1]:.2f}")


def sum_difference(quick=False):
    print("pole_sum_difference_time_course")
    T = 1000.0
    ts = np.linspace(0, T, 2001)
    c0 = pl.one_cluster_ic(P)
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
    for ax, g, lab in zip(axs, GAMMAS, LABELS):
        red, full = pl.reduced_solve(P, g, T, c0, ts)
        _report_sd(g, red, full)
        _plot_sd(ax, ts, red)
        ax.set_title(lab, fontsize=14)
        ax.set_xlabel("time")
        ax.set_xlim(0, T)
    axs[0].set_ylabel("total plasmids")
    axs[0].legend(ncol=3, fontsize=10, frameon=False, loc="center right")
    fig.suptitle(f"{P.name} — pole sum/difference equations:  "
                 + P.rates_tex() + r" $-$ start: one cluster of $N$ in pole 1",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .93])
    _save(fig, "pole_sum_difference_time_course.png")


def sum_difference_gamma2(quick=False):
    print("pole_sum_difference_gamma2_long_run")
    g = 2.0
    c0 = pl.one_cluster_ic(P)
    ta = np.linspace(0, 1000.0, 2001)
    tb = np.concatenate([[0.0], np.logspace(-2, 5, 1500)])
    ra, fa = pl.reduced_solve(P, g, ta[-1], c0, ta)
    rb, fb = pl.reduced_solve(P, g, tb[-1], c0, tb)
    _report_sd(g, ra, fa)
    _report_sd(g, rb, fb)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9.5), sharex="col")
    for col, (t, red) in enumerate([(ta, ra), (tb, rb)]):
        for row in (0, 1):
            ax = axs[row, col]
            _plot_sd(ax, t, red)
            if col == 1:
                ax.set_xscale("log")
                ax.set_xlim(1e-2, 1e5)
            else:
                ax.set_xlim(0, 1000)
    small = np.concatenate([np.abs(r[k]) for r in (ra, rb)
                            for k in ("Ftot", "dF", "Ctot", "dC")])
    top = 1.1 * small.max()
    for ax in axs[1]:
        ax.set_ylim(-0.1 * top, top)
        ax.set_xlabel("time")
    for ax in axs[:, 0]:
        ax.set_ylabel("total plasmids")
    axs[0, 0].set_title(r"(a)  $0 \leq t \leq 1000$", fontsize=12)
    axs[0, 1].set_title(r"(b)  long run to $t=10^5$, log time", fontsize=12)
    axs[1, 0].set_title(r"(c)  as (a), zoomed on $F$ and $C$", fontsize=12)
    axs[1, 1].set_title(r"(d)  as (b), zoomed on $F$ and $C$", fontsize=12)
    axs[0, 0].legend(ncol=3, fontsize=10, frameon=False, loc="center right")
    fig.suptitle(rf"{P.name} — pole sum/difference equations, $\gamma=2$, "
                 + P.rates_tex() + r" $-$ start: one cluster of $N$ in pole 1",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, .95])
    _save(fig, "pole_sum_difference_gamma2_long_run.png")


# ======================================================================
#  Pole imbalance vs gamma, mean-field  (part4 fig8)
# ======================================================================


def imbalance_meanfield(quick=False):
    print("pole_imbalance_vs_gamma_meanfield")
    T = 1000.0
    gammas = np.unique(np.round(np.concatenate(
        [np.arange(0, 2.0001, .1 if quick else .05), [2 / 3]]), 12))
    final = {0: [], 1: []}
    for g in gammas:
        for p0 in (0, 1):
            _, Y = pl.chain_solve(P, g, T, pl.one_cluster_ic(P, p0),
                                  n_record=2)
            final[p0].append(pl.sum_diff(Y, g)["dN"][-1])
    final = {k: np.array(v) for k, v in final.items()}
    for g, a, b in zip(gammas, final[0], final[1]):
        if np.isclose(g % .25, 0) or np.isclose(g % .25, .25):
            print(f"  gamma={g:.2f}  dN(T) {a:+7.2f} {b:+7.2f}")
    print(f"  max antisymmetry error {np.abs(final[0] + final[1]).max():.1e}")

    fig, ax = plt.subplots(figsize=(7.2, 5))
    ax.plot(gammas, final[0], "o-", color="#2c4fa8", lw=1.6, ms=4,
            label=r"start in pole 1  ($+N$)")
    ax.plot(gammas, final[1], "s--", color="#c0392b", lw=1.4, ms=4,
            label=r"start in pole 2  ($-N$)")
    for g in GAMMAS:
        ax.axvline(g, color="#888888", lw=.6, ls=":")
    ax.axhline(0, color="k", lw=.8, ls=":")
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(rf"$\Delta N$  at  $t={T:g}$")
    ax.set_ylim(-P.N * 1.1, P.N * 1.1)
    ax.grid(alpha=.3)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    ax.set_title(rf"{P.name}: $\Delta N$ vs $\gamma$ (mean-field equations)"
                 + "\n" + P.rates_tex() + r", start: one $N$-cluster",
                 fontsize=11)
    fig.tight_layout()
    _save(fig, "pole_imbalance_vs_gamma_meanfield.png")


# ======================================================================
#  Numerical checks
# ======================================================================


def checks():
    jj = np.arange(P.N + 1)
    print(f"\n=== {P.name}: conservation ===")
    for g in (0.0, 1.0, 2.0):
        rec, c, _ = pl.ssa_run(P, g, 2000.0, np.linspace(0, 2000, 201),
                               pl.one_cluster_ic(P).astype(np.int64), 7)
        m = (rec * jj).sum((1, 2))
        _, Y = pl.chain_solve(P, g, 2000.0, pl.one_cluster_ic(P), n_record=201)
        md = (Y * jj).sum((1, 2))
        print(f"  gamma={g:.1f}  SSA mass {m.min()}..{m.max()}   ODE mass "
              f"drift {np.ptp(md):.1e}  min c {Y.min():+.1e}")

    print("\n=== mean-field steady state is a fixed point of the chain ===")
    for g in GAMMAS + [2.0]:
        ce = pl.mf_equilibrium(P, g)
        _, Y = pl.chain_solve(P, g, 2e4, ce, n_record=2)
        cf = pl.eq_pole(P, g, pl.eq_free(P, g, P.N / 2))
        print(f"  gamma={g:.3f}  drift over t=2e4 {np.abs(Y[-1] - ce).max():.1e}"
              f"   distance to product form {np.abs(ce[0] - cf).max():.2e}")

    if P.smol:
        print("\n=== newborn burn-in: pool at T_BURN vs a longer run ===")
        for g in GAMMAS + [2.0]:
            draws, mean = pl.newborn_pool(P, g)
            n = 400
            e = pl._parallel(P, g, pl.T_BURN, np.array([pl.T_BURN]),
                             draws[:n], pl._seeds(55, n), rho=0.0)
            a = pl.moments(draws[:n].astype(float), g)
            b = pl.moments(e["cfinal"].astype(float), g)
            line = f"  gamma={g:.3f}"
            for k in ("f", "C"):
                x, y = a[k][:, 0], b[k][:, 0]
                se = np.sqrt(x.var() / n + y.var() / n)
                line += (f"  {k}: {x.mean():6.3f} -> {y.mean():6.3f}"
                         f" (z={(y.mean() - x.mean()) / se:+.2f})")
            if g == 1.0:
                _, ex = pl.product_form(P, g, 1, 0)
                m = pl.moments(ex, g)
                line += (f"   exact product form f {m['f']:.3f}"
                         f" C {m['C']:.3f}")
            print(line)

    print("\n=== exact CME (N=8) vs SSA, t=3 ===")
    from scipy.linalg import expm
    Q_ = pl.Params(model=P.model, N=8, beta=.15, kappa=.4, rho=.2, betam=.15)
    S = _states(Q_.N)
    idx = {s.tobytes(): i for i, s in enumerate(S)}
    for g in (0.0, 2 / 3, 1.5):
        Q = np.zeros((len(S), len(S)))
        for i, s in enumerate(S):
            for d, r in _jumps(Q_, g, s):
                Q[i, idx[d.tobytes()]] += r
                Q[i, i] -= r
        c0 = pl.one_cluster_ic(Q_).astype(np.int64)
        p0 = np.zeros(len(S))
        p0[idx[c0.tobytes()]] = 1
        pT = p0 @ expm(3.0 * Q)
        M = pl.moments(np.array(S), g)
        n = 100000
        e = pl._parallel(Q_, g, 3.0, np.array([3.0]),
                         np.repeat(c0[None], n, 0), pl._seeds(3, n))
        z = [(e[k][:, -1, p].mean() - (pT * M[k][:, p]).sum())
             / (e[k][:, -1, p].std() / np.sqrt(n)) for k in ("f", "b", "C")
             for p in (0, 1)]
        print(f"  gamma={g:.2f}  {len(S)} states, max |z| over f,b,C x 2 poles"
              f" = {np.max(np.abs(z)):.2f}")


def _states(N):
    def parts(n, mx):
        if n == 0:
            yield ()
            return
        for k in range(min(n, mx), 0, -1):
            for r in parts(n - k, k):
                yield (k,) + r
    S = []
    for n1 in range(N + 1):
        for a in parts(n1, n1):
            for b in parts(N - n1, N - n1):
                c = np.zeros((2, N + 1), np.int64)
                for k in a:
                    c[0, k] += 1
                for k in b:
                    c[1, k] += 1
                S.append(c)
    return S


def _jumps(Q, g, c):
    """Reaction table of the notes, written out independently of ssa.c."""
    N, out = Q.N, []

    def J(rate, *mods):
        if rate > 0:
            d = c.copy()
            for pp, j, dv in mods:
                d[pp, j] += dv
            out.append((d, rate))
    for p in (0, 1):
        q, cp = 1 - p, c[p]
        f = cp[1]
        J(.5 * Q.beta * f * (f - 1), (p, 1, -2), (p, 2, 1))
        for j in range(2, N):
            J(Q.beta * j ** g * f * cp[j], (p, 1, -1), (p, j, -1), (p, j + 1, 1))
        for j in range(3, N + 1):
            J(Q.kappa * j * cp[j], (p, j, -1), (p, j - 1, 1), (p, 1, 1))
        J(Q.kappa * cp[2], (p, 2, -1), (p, 1, 2))
        J(Q.rho * f, (p, 1, -1), (q, 1, 1))
        if Q.smol:
            for i in range(2, N + 1):
                for j in range(i, N + 1 - i):
                    if i < j:
                        J(Q.betam * (i * j) ** g * cp[i] * cp[j],
                          (p, i, -1), (p, j, -1), (p, i + j, 1))
                    else:
                        J(.5 * Q.betam * i ** (2 * g) * cp[i] * (cp[i] - 1),
                          (p, i, -2), (p, 2 * i, 1))
            for m in range(4, N + 1):
                for i in range(2, m // 2 + 1):
                    if 2 * i == m:
                        J(.5 * Q.kappa * m * cp[m], (p, m, -1), (p, i, 2))
                    else:
                        J(Q.kappa * m * cp[m], (p, m, -1), (p, i, 1),
                          (p, m - i, 1))
    return out


# ======================================================================

FIGS = {"time_course": time_course,
        "cluster_sizes": cluster_sizes,
        "ode_error": ode_error,
        "one_cluster": one_cluster_vs_full,
        "imbalance_ssa": imbalance_ssa,
        "sum_difference": sum_difference,
        "imbalance_meanfield": imbalance_meanfield,
        "sum_difference_gamma2": sum_difference_gamma2}


def main(model, here):
    global P, OUT
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fig", nargs="+", choices=list(FIGS))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--checks", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="fewer replicates / gamma values")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore the cache and rerun the SSA")
    args = ap.parse_args()
    if not (args.fig or args.all or args.checks):
        ap.print_help()
        return
    P = pl.Params(model=model)
    OUT = os.path.join(here, "figures")
    pl.CACHE_DIR = os.path.join(here, "cache")
    pl.RECOMPUTE = args.recompute
    t0 = time.time()
    if args.checks:
        checks()
    for k in (list(FIGS) if args.all else (args.fig or [])):
        FIGS[k](quick=args.quick)
    print(f"\ntotal {time.time() - t0:.1f} s")
