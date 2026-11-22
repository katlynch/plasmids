# Many-cluster models: Becker–Döring and Smoluchowski

This folder extends the net-bound (one-cluster) model at the root of this
repository. Each pole now holds any number of clusters of any size, $c_j^{(p)}$.
There are two forms:

- **Becker–Döring** (`beckerdoring/`): clusters grow and shrink one plasmid at
  a time. The channels are nucleation, growth at rate β j^γ f c_j, shedding at
  κ j c_j, dimer dissolution at κ c_2, and transfer of free plasmids between
  the poles at rate ρ.
- **Smoluchowski** (`smoluchowski/`): the same, plus merging of two clusters at
  β_m (ij)^γ and breaking of a cluster into two (κ m per unequal split), with
  β_m = β.

Rates: β = 0.015, κ = 0.08, ρ = 0.13 per minute. Each model has an exact
stochastic simulation (Gillespie, in C) and a mean-field size chain truncated
at size N.

The net-bound model used for every comparison is **`meanfieldode.m` from this
repository**, transcribed line for line (`plasmid_common/plasmid.py`,
`netbound_rhs`).

## Layout

| path | contents |
|---|---|
| `plasmid_common/plasmid.py` | the model: stochastic simulation driver, mean-field chains, net-bound ODE, initial conditions, cache |
| `plasmid_common/ssa.c` | Gillespie kernel, compiled automatically with `cc` on first use |
| `plasmid_common/figures.py` | time course, cluster sizes, closure error, pole imbalance (N = 60) |
| `plasmid_common/kat_figures.py` | the figures of this repository's scripts, redone for the many-cluster models |
| `plasmid_common/check_one_cluster.py` | check that the one-cluster reduction of both models equals `meanfieldode.m` |
| `beckerdoring/`, `smoluchowski/` | `make_figures.py`, `make_kat_figures.py`, figures, and a README mapping each figure to its counterpart |
| `one_cluster_check/` | output of `check_one_cluster.py` |

## Running

```
cd beckerdoring            # or smoluchowski
python make_figures.py --all          # figures/
python make_kat_figures.py --all      # katlynch_figures/
python make_figures.py --checks       # exact master equation vs simulation, conservation, equilibria
cd ../plasmid_common && python check_one_cluster.py
```

Requirements: Python 3 with numpy, scipy, matplotlib and seaborn, plus a C
compiler (`cc`). Simulation results are cached in `<model>/cache/`, which is
not tracked. A full run takes about 5 minutes for Becker–Döring and 10 for
Smoluchowski.

## Main results

- **The one-cluster reduction of both forms is the net-bound model.** Our
  chains evaluated at one-cluster states reproduce `meanfieldode.m` to 1e-11,
  and the trajectories agree to 3e-9 (`one_cluster_check/`). It is exact at
  γ = 1. For γ < 1 the net-bound model keeps too many plasmids free and
  balances the poles about twice too fast.
- **`asymmetryode.m` differs from `meanfieldode.m` and `asymmetry.ode`.** Its
  capture term
  `((Ft + F)*(Ntot + N - F - Ft)^gamma)/4` equals 2^(γ−1) f₁ b₁^γ, not
  f₁ b₁^γ, so it is correct only at γ = 1 (ΔN off by up to 1.5 at γ = 2/3).
  The XPP file `asymmetry.ode` divides by 2 inside the power and is correct.
- **Becker–Döring has no pitchfork.** Every steady state is in detailed
  balance, which forces f₁ = f₂. The symmetric state is therefore the only
  steady state and is stable for every γ and N. Plasmids staying in the old
  pole is a transient, but its relaxation time grows from ~50 min at γ = 0 to
  over 10¹⁰ min at γ ≈ 2.2 (N = 50).
- **Smoluchowski has a genuine supercritical pitchfork**, at
  γ* = 1.165 for N = 50. Across N it runs from 1.99 (N = 10) to 1.11
  (N = 80), with no instability below N ≈ 10. Merging and breaking break
  detailed balance except at γ = 1 with β_m = β.
- **The mean-field closure is accurate.** It drops about 2% of the binding
  flux.
