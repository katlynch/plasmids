"""
plasmid.py
==========
Engine for the two-pole, many-cluster plasmid model in its two forms:

  'bd'    Becker-Doering (Part IV): nucleation, growth, shedding, dimer
          dissolution, transfer.
  'smol'  Smoluchowski (Part V): the same plus merging of two clusters
          (beta_m (ij)^gamma) and breaking into two clusters (rate kappa m per
          unequal split, kappa m / 2 for the equal split).

Detachment follows the per-label rule r(j) = kappa j / 2 of the notes, so a
dimer dissolves at kappa c_2 and the total release is kappa b in every
configuration.

Contents
  * exact SSA (C kernel ssa.c via ctypes, run in parallel processes)
  * mean-field chains (eq. bd2-chain / sm-chain), truncated at size N
  * one-cluster-per-pole ODE (Part III), identical for both forms
  * pole sum / difference equations (Delta N, Delta F, F_tot, Delta C, C_tot)
  * newborn-cell initial condition: all N plasmids in one pole, drawn from
    that pole's stationary law; the other pole empty
  * disk cache for SSA ensembles
"""
from __future__ import annotations

import ctypes
import dataclasses
import hashlib
import json
import multiprocessing
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import gammaln, logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, "cache")     # overridden by the figure script
RECOMPUTE = False
CACHE_VERSION = 1
NPROC = os.cpu_count() or 4
T_BURN = 2000.0           # one-pole burn-in for the Smoluchowski newborn IC


@dataclasses.dataclass(frozen=True)
class Params:
    model: str = "bd"            # 'bd' or 'smol'
    N: int = 60
    beta: float = 0.015
    kappa: float = 0.08
    rho: float = 0.13
    betam: float = 0.015         # merging constant (Smoluchowski only)

    @property
    def smol(self):
        return self.model == "smol"

    @property
    def name(self):
        return "Smoluchowski" if self.smol else "Becker–Döring"

    def rates_tex(self):
        s = (rf"$\beta={self.beta}$, $\kappa={self.kappa}$, "
             rf"$\rho={self.rho}$, $N={self.N}$")
        if self.smol:
            s += rf", $\beta_m={self.betam}$"
        return s

    def key(self):
        d = dataclasses.asdict(self)
        if not self.smol:
            d.pop("betam")
        return d


# ======================================================================
#  C kernel
# ======================================================================

_LIB = None


def _lib():
    global _LIB
    if _LIB is not None:
        return _LIB
    src = os.path.join(HERE, "ssa.c")
    so = os.path.join(HERE, "libssa.so")
    if not os.path.exists(so) or os.path.getmtime(so) < os.path.getmtime(src):
        subprocess.run(["cc", "-O3", "-shared", "-fPIC", "-o", so, src, "-lm"],
                       check=True)
    lib = ctypes.CDLL(so)
    lib.ssa_run.restype = ctypes.c_long
    D, I, P = ctypes.c_double, ctypes.c_int, np.ctypeslib.ndpointer
    lib.ssa_run.argtypes = [
        I, D, D, D, D, D, I, D, I,
        P(np.float64, flags="C"), P(np.int64, flags="C"),
        P(np.int16, flags="C"), ctypes.c_uint64]
    _LIB = lib
    return lib


def ssa_run(P: Params, gamma, T, trec, c0, seed, rho=None):
    """One trajectory. c0: int state (2, N+1). Returns (rec (nrec,2,N+1) int16,
    final state, number of events)."""
    lib = _lib()
    c = np.ascontiguousarray(c0, dtype=np.int64).copy()
    trec = np.ascontiguousarray(trec, dtype=np.float64)
    rec = np.zeros((len(trec), 2, P.N + 1), dtype=np.int16)
    nev = lib.ssa_run(P.N, P.beta, P.betam, P.kappa,
                      P.rho if rho is None else rho, float(gamma),
                      int(P.smol), float(T), len(trec), trec, c, rec,
                      ctypes.c_uint64(int(seed)))
    return rec, c, nev


# ======================================================================
#  Moments
# ======================================================================


def weights(N, gamma):
    jj = np.arange(N + 1, dtype=float)
    return jj, np.where(jj >= 2, jj ** gamma, 0.0)


def moments(C, gamma):
    """C[..., p, j] (j = 0..N) -> f, b, C, A, c2, each shaped (..., 2)."""
    jj, w = weights(C.shape[-1] - 1, gamma)
    big = jj >= 2
    return dict(f=C[..., 1].astype(float),
                b=(C * jj * big).sum(-1),
                C=(C * big).sum(-1).astype(float),
                A=(C * w).sum(-1),
                c2=C[..., 2].astype(float))


# ======================================================================
#  Cache
# ======================================================================


def _hash_array(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def cached(tag, params, compute):
    key = json.dumps(dict(params, version=CACHE_VERSION), sort_keys=True,
                     default=str)
    path = os.path.join(CACHE_DIR,
                        f"{tag}_{hashlib.sha1(key.encode()).hexdigest()[:12]}.npz")
    if os.path.exists(path) and not RECOMPUTE:
        with np.load(path) as d:
            return {k: d[k] for k in d.files if k != "_params"}
    out = compute()
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez_compressed(path, _params=np.array(key), **out)
    print(f"    cached {os.path.basename(path)}")
    return out


# ======================================================================
#  SSA ensembles (parallel)
# ======================================================================


def _worker(args):
    P, gamma, T, trec, c0s, seeds, rho = args
    nrec = len(trec)
    out = {k: np.zeros((len(c0s), nrec, 2), dtype=np.float32)
           for k in ("f", "b", "C", "A")}
    cfin = np.zeros((len(c0s), 2, P.N + 1), dtype=np.int16)
    s1 = np.zeros((nrec, 2, P.N + 1))
    nev = 0
    for r, (c0, sd) in enumerate(zip(c0s, seeds)):
        rec, c, n = ssa_run(P, gamma, T, trec, c0, sd, rho=rho)
        m = moments(rec, gamma)
        for k in out:
            out[k][r] = m[k]
        cfin[r] = c
        s1 += rec
        nev += n
    return out, cfin, s1, nev


def _parallel(P, gamma, T, trec, c0s, seeds, rho=None):
    n = len(c0s)
    nchunk = min(n, 4 * NPROC)
    idx = np.array_split(np.arange(n), nchunk)
    jobs = [(P, gamma, T, trec, c0s[i], seeds[i], rho) for i in idx]
    if NPROC > 1 and n > 1:
        with ProcessPoolExecutor(
                NPROC, mp_context=multiprocessing.get_context("fork")) as ex:
            res = list(ex.map(_worker, jobs))
    else:
        res = [_worker(j) for j in jobs]
    out = {k: np.concatenate([r[0][k] for r in res]) for k in res[0][0]}
    out["cfinal"] = np.concatenate([r[1] for r in res])
    out["cmean"] = sum(r[2] for r in res) / n
    out["nev"] = np.array(sum(r[3] for r in res))
    return out


def _seeds(seed, n):
    return np.random.SeedSequence(seed).generate_state(n, dtype=np.uint64)


def ensemble(P, gamma, T, nrec, c0s, seed, tag="ssa"):
    """SSA replicates from the given initial states c0s (nrep, 2, N+1)."""
    c0s = np.asarray(c0s, dtype=np.int64)
    trec = np.linspace(0.0, T, nrec)

    def compute():
        out = _parallel(P, gamma, T, trec, c0s, _seeds(seed, len(c0s)))
        out["t"] = trec
        return out

    params = dict(P.key(), gamma=float(gamma), T=T, nrec=nrec, seed=seed,
                  c0=_hash_array(c0s))
    out = cached(tag, params, compute)
    for k in ("f", "b", "C", "A"):
        out[k] = out[k].astype(float)
    return out


# ======================================================================
#  Stationary law of one pole and the newborn initial condition
# ======================================================================
#
#  Becker-Doering: every channel is reversible with detailed balance, so one
#  isolated pole holding M plasmids has the product-form law
#      pi(c) ~ prod_j q_j^{c_j} / c_j!,
#      q_j = (beta/kappa)^{j-1} ((j-1)!)^gamma / j!   (q_1 = 1),
#  and the mean-field equilibrium is c_j = q_j f^j.  (With dimer dissolution
#  at kappa c_2 there is no extra 1/2 in q_j.)
#
#  Smoluchowski: merging/breaking satisfy detailed balance with the same q_j
#  only if gamma = 1 and beta_m = beta.  For other gamma the stationary law
#  has no closed form, so the newborn state is drawn by running pole 1 alone
#  (rho = 0) for T_BURN from a product-form draw.


def logq(P, gamma):
    j = np.arange(P.N + 1, dtype=float)
    lq = np.full(P.N + 1, -np.inf)
    js = j[1:]
    lq[1:] = (js - 1) * np.log(P.beta / P.kappa) + gamma * gammaln(js) \
        - gammaln(js + 1)
    return lq


def _logZ(n, logw):
    """m Z_m = sum_j j w_j Z_{m-j}  (log scale), m = 0..n."""
    lz = np.zeros(n + 1)
    lj = np.log(np.arange(1, n + 1))
    for m in range(1, n + 1):
        k = np.arange(1, m + 1)
        lz[m] = logsumexp(lj[:m] + logw[k] + lz[m - k]) - np.log(m)
    return lz


def _sample_partition(n, logw, lz, rng):
    c = np.zeros(n + 1, dtype=np.int64)
    m = n
    while m > 0:
        k = np.arange(1, m + 1)
        lp = np.log(k) + logw[k] + lz[m - k] - np.log(m) - lz[m]
        p = np.exp(lp - lp.max())
        j = rng.choice(k, p=p / p.sum())
        c[j] += 1
        m -= j
    return c


def product_form(P, gamma, n, seed):
    """n exact draws of one pole holding all N plasmids under the product-form
    law, and its exact mean (both shaped for one pole, (.., N+1))."""
    lw = logq(P, gamma)
    lz = _logZ(P.N, lw)
    rng = np.random.default_rng(seed)
    draws = np.array([_sample_partition(P.N, lw, lz, rng) for _ in range(n)])
    j = np.arange(1, P.N + 1)
    mean = np.zeros(P.N + 1)
    mean[1:] = np.exp(lw[j] + lz[P.N - j] - lz[P.N])
    return draws, mean


NPOOL = 1000


def newborn_pool(P, gamma, seed=2024):
    """NPOOL draws of pole 1 holding all N plasmids at its own stationary law,
    pole 2 empty: (draws (NPOOL, 2, N+1) int, mean (2, N+1))."""
    def compute():
        d1, mean1 = product_form(P, gamma, NPOOL, seed)
        if P.smol:
            c0s = np.zeros((NPOOL, 2, P.N + 1), dtype=np.int64)
            c0s[:, 0] = d1
            res = _parallel(P, gamma, T_BURN, np.array([T_BURN]), c0s,
                            _seeds(seed + 7, NPOOL), rho=0.0)
            d1 = res["cfinal"][:, 0].astype(np.int64)
            mean1 = d1.mean(0)
        draws = np.zeros((NPOOL, 2, P.N + 1), dtype=np.int64)
        draws[:, 0] = d1
        mean = np.zeros((2, P.N + 1))
        mean[0] = mean1
        return dict(draws=draws, mean=mean)

    params = dict(P.key(), gamma=float(gamma), npool=NPOOL, seed=seed,
                  tburn=T_BURN if P.smol else 0)
    d = cached("pool", params, compute)
    return d["draws"], d["mean"]


def mirror(c):
    """Swap the poles."""
    return np.asarray(c)[..., ::-1, :].copy()


def one_cluster_ic(P, pole=0):
    c = np.zeros((2, P.N + 1))
    c[pole, P.N] = 1
    return c


# ======================================================================
#  Mean-field chains
# ======================================================================


def chain_rhs(t, y, P, gamma, w, jj):
    N, beta, kappa = P.N, P.beta, P.kappa
    c = y.reshape(2, N + 1)
    dc = np.zeros_like(c)
    for p in (0, 1):
        cp = c[p]
        f = cp[1]
        J = np.zeros(N + 1, dtype=c.dtype)        # J[j]: flux j -> j+1
        J[1] = 0.5 * beta * f * f - kappa * cp[2]
        J[2:N] = beta * w[2:N] * f * cp[2:N] - kappa * jj[3:] * cp[3:]
        d = dc[p]
        d[2:] = J[1:N] - J[2:]
        d[1] = -2.0 * J[1] - J[2:N].sum()
        if P.smol:
            wc = w * cp                           # zero at j = 0, 1
            gain = 0.5 * P.betam * np.convolve(wc, wc)[:N + 1]
            S = np.cumsum(wc)                     # partners k <= N - j
            loss = P.betam * wc * S[N - np.arange(N + 1)]
            mc = jj * cp
            mc[:2] = 0.0
            tail = np.concatenate([np.cumsum(mc[::-1])[::-1], [0.0, 0.0]])
            bgain = kappa * tail[np.arange(N + 1) + 2]
            bloss = 0.5 * kappa * jj * (jj - 3.0) * cp
            bloss[:4] = 0.0                       # j = 2, 3 do not break
            d[2:] += gain[2:] - loss[2:] + bgain[2:] - bloss[2:]
    dc[0, 1] += P.rho * (c[1, 1] - c[0, 1])
    dc[1, 1] += P.rho * (c[0, 1] - c[1, 1])
    return dc.ravel()


def chain_solve(P, gamma, T, c0, ts=None, n_record=400, dense=False,
                rtol=1e-9, atol=1e-11, method="LSODA"):
    jj, w = weights(P.N, gamma)
    ts = np.linspace(0, T, n_record) if ts is None else ts
    sol = solve_ivp(chain_rhs, (0, T), np.asarray(c0, float).ravel(),
                    t_eval=ts, method=method, args=(P, gamma, w, jj),
                    rtol=rtol, atol=atol, dense_output=dense)
    if not sol.success:
        raise RuntimeError(sol.message)
    Y = sol.y.T.reshape(-1, 2, P.N + 1)
    return (ts, Y, sol) if dense else (ts, Y)


def eq_pole(P, gamma, f):
    lq = logq(P, gamma)
    c = np.zeros(P.N + 1)
    j = np.arange(1, P.N + 1)
    c[1:] = np.exp(lq[1:] + j * np.log(f))
    return c


def eq_free(P, gamma, M):
    """Free level at which one pole's product-form equilibrium holds mass M."""
    lq = logq(P, gamma)[1:]
    j = np.arange(1, P.N + 1)
    h = lambda lf: logsumexp(np.log(j) + lq + j * lf) - np.log(M)
    return np.exp(brentq(h, -60.0, np.log(M), xtol=1e-14))


def mf_equilibrium(P, gamma):
    """Symmetric mean-field steady state (2, N+1).  Closed form for BD (and for
    Smoluchowski at gamma = 1, beta_m = beta, where detailed balance holds);
    otherwise a long run of the chain from the BD equilibrium."""
    ce = np.tile(eq_pole(P, gamma, eq_free(P, gamma, P.N / 2.0)), (2, 1))
    if not P.smol or (gamma == 1.0 and P.betam == P.beta):
        return ce
    _, Y = chain_solve(P, gamma, 2e5, ce, n_record=2, rtol=1e-10, atol=1e-12)
    return Y[-1]


# ======================================================================
#  One-cluster-per-pole ODE (Part III); the same for both forms
# ======================================================================


def one_rhs(t, y, P, gamma):
    f1, f2, b1, b2 = y
    g1 = b1 ** gamma if b1 > 0 else 0.0
    g2 = b2 ** gamma if b2 > 0 else 0.0
    k1 = P.beta * f1 * (f1 + g1) - P.kappa * b1
    k2 = P.beta * f2 * (f2 + g2) - P.kappa * b2
    return [-k1 + P.rho * (f2 - f1), -k2 + P.rho * (f1 - f2), k1, k2]


def one_solve(P, gamma, ts, c0):
    jj = np.arange(P.N + 1)
    f0 = c0[:, 1]
    b0 = (c0 * jj).sum(1) - f0
    sol = solve_ivp(one_rhs, (0, ts[-1]), [f0[0], f0[1], b0[0], b0[1]],
                    t_eval=ts, method="Radau", args=(P, gamma), rtol=1e-9,
                    atol=1e-11)
    return dict(f=sol.y[0:2].T, b=sol.y[2:4].T)


def netbound_rhs(t, x, P, gamma):
    """Kat Lynch's net-bound model, transcribed from meanfieldode.m
    (github.com/katlynch/plasmids), state x = (f1, b1, f2, b2).  As there,
    b^gamma is evaluated also for an empty pole (0^0 = 1); the base is
    clipped at 0 only to keep round-off from producing NaN."""
    f1, b1, f2, b2 = x
    g1, g2 = max(b1, 0.0) ** gamma, max(b2, 0.0) ** gamma
    be, ka, rh = P.beta, P.kappa, P.rho
    return [-be * f1 ** 2 - be * f1 * g1 + ka * b1 + rh * (f2 - f1),
            be * f1 ** 2 + be * f1 * g1 - ka * b1,
            -be * f2 ** 2 - be * f2 * g2 + ka * b2 + rh * (f1 - f2),
            be * f2 ** 2 + be * f2 * g2 - ka * b2]


def netbound_solve(P, gamma, ts, c0):
    """Kat's net-bound model from the f_p, b_p of the chain state c0."""
    jj = np.arange(P.N + 1)
    f0 = c0[:, 1]
    b0 = (c0 * jj).sum(1) - f0
    sol = solve_ivp(netbound_rhs, (0, ts[-1]), [f0[0], b0[0], f0[1], b0[1]],
                    t_eval=ts, method="Radau", args=(P, gamma), rtol=1e-9,
                    atol=1e-11)
    return dict(f=sol.y[[0, 2]].T, b=sol.y[[1, 3]].T)


# ======================================================================
#  Pole sum / difference equations
# ======================================================================
#
#  Delta N = N_1 - N_2, Delta F = f_1 - f_2, F_tot = f_1 + f_2,
#  Delta C = C_1 - C_2, C_tot = C_1 + C_2;  with f_1,2 = (F_tot +- Delta F)/2
#
#  dDelta N/dt = -2 rho Delta F
#  dDelta F/dt = -beta [Delta F F_tot + (f_1 A_1 - f_2 A_2)]
#                + kappa (Delta N - Delta F) - 2 rho Delta F
#  dF_tot/dt   = -beta [(F_tot^2 + Delta F^2)/2 + (f_1 A_1 + f_2 A_2)]
#                + kappa (N - F_tot)
#  dC_p/dt     = (1/2) beta f_p^2 - kappa c_2^(p)
#                [Smoluchowski:  - (1/2) beta_m sum_{i+k<=N} w_i c_i w_k c_k
#                                + (1/2) kappa sum_{m>=4} m (m-3) c_m]
#
#  A_p, c_2^(p) and the merging/breaking sums depend on the size distribution;
#  they are read off the mean-field chain and the five equations are then
#  integrated on their own.  The merging term is the chain-consistent one
#  (the sum of M_j over the truncated chain), so C_p agrees with sum_j c_j of
#  the chain; the self-pair-excluded form of eq. (sm-mom) would drift away
#  from it by (1/2) beta_m sum_j w_j^2 c_j per unit time.


def closure_inputs(P, gamma, c):
    """c (..., 2, N+1) -> A, c2, merge sum, break sum, each (..., 2).

    A = sum_{j=2}^{N-1} w_j c_j: on the truncated chain size N cannot grow
    (J_N = 0), so the capture sum stops at N-1."""
    jj, w = weights(P.N, gamma)
    A = (c[..., :P.N] * w[:P.N]).sum(-1)
    c2 = c[..., 2]
    if not P.smol:
        z = np.zeros_like(A)
        return A, c2, z, z
    wc = c * w
    S = np.cumsum(wc, -1)
    mrg = (wc * S[..., P.N - np.arange(P.N + 1)]).sum(-1)
    brk = (np.where(jj >= 4, jj * (jj - 3.0), 0.0) * c).sum(-1)
    return A, c2, mrg, brk


def reduced_solve(P, gamma, T, c0, ts):
    """Integrate the five sum/difference equations with closure from the
    chain.  Returns (red, full): dicts of dN, dF, Ftot, dC, Ctot (and Ntot)
    from the reduced ODEs and evaluated directly on the chain.

    The five equations are integrated together with the chain in one system,
    so the closure inputs are the chain's exact current state (interpolating
    a separate chain solution lets the large, nearly cancelling merge/break
    fluxes of the Smoluchowski form accumulate interpolation error)."""
    N, beta, kappa, rho = P.N, P.beta, P.kappa, P.rho
    jj, w = weights(N, gamma)
    nc = 2 * (N + 1)

    def rhs(t, y):
        c = y[:nc]
        dN, dF, Ft, dC, Ct = y[nc:]
        A, c2, mrg, brk = closure_inputs(P, gamma, c.reshape(2, N + 1))
        f = np.array([(Ft + dF) / 2, (Ft - dF) / 2])
        dCp = 0.5 * beta * f ** 2 - kappa * c2 \
            - 0.5 * P.betam * mrg + 0.5 * kappa * brk
        red = [-2 * rho * dF,
               -beta * (dF * Ft + f[0] * A[0] - f[1] * A[1])
               + kappa * (dN - dF) - 2 * rho * dF,
               -beta * ((Ft ** 2 + dF ** 2) / 2 + f[0] * A[0] + f[1] * A[1])
               + kappa * (N - Ft),
               dCp[0] - dCp[1], dCp[0] + dCp[1]]
        return np.concatenate([chain_rhs(t, c, P, gamma, w, jj), red])

    c0 = np.asarray(c0, float)
    s0 = sum_diff(c0[None], gamma)
    y0 = np.concatenate([c0.ravel(), [s0[k][0] for k in
                                      ("dN", "dF", "Ftot", "dC", "Ctot")]])
    # tolerances are looser than for the chain alone: in the Smoluchowski form
    # dC/dt is a difference of merge/break fluxes up to ~1e5 (gamma = 2), so
    # round-off alone is ~1e-6 and a tighter atol only shrinks the steps
    r = solve_ivp(rhs, (ts[0], ts[-1]), y0, t_eval=ts, method="BDF",
                  rtol=1e-8, atol=1e-9)
    if not r.success:
        raise RuntimeError(r.message)
    full = sum_diff(r.y[:nc].T.reshape(-1, 2, N + 1), gamma)
    red = dict(zip(("dN", "dF", "Ftot", "dC", "Ctot"), r.y[nc:]))
    red["Ntot"] = full["Ntot"]
    return red, full


def sum_diff(Y, gamma):
    m = moments(Y, gamma)
    Np = m["f"] + m["b"]
    return dict(dN=Np[:, 0] - Np[:, 1], Ntot=Np.sum(1),
                dF=m["f"][:, 0] - m["f"][:, 1], Ftot=m["f"].sum(1),
                dC=m["C"][:, 0] - m["C"][:, 1], Ctot=m["C"].sum(1))
