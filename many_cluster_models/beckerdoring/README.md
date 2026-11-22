# Becker–Döring form (Part IV)

Part4 figure set redone for this model. Rates N = 60, β = 0.015, κ = 0.08, ρ = 0.13
and initial conditions as in `part4/`; detachment per the notes (dimer
dissolution κ c₂).

| figure | part4 original | start |
|---|---|---|
| `ssa_vs_meanfield_time_course.png` | fig5_gamma_effect | newborn cell |
| `plasmids_per_cluster_size_t1000.png` / `.csv` | fig5b_cluster_sizes | newborn cell |
| `meanfield_error_ssa_minus_ode.png` | fig5d_ode_error | newborn cell |
| `chain_vs_kat_netbound.png` | fig5e_reduced_vs_full (now against Kat's net-bound `meanfieldode.m`) | newborn cell |
| `pole_imbalance_vs_gamma_ssa.png` | fig6_delta_vs_gamma | newborn, mirrored |
| `pole_sum_difference_time_course.png` | fig7_reduced | one N-cluster in pole 1 |
| `pole_imbalance_vs_gamma_meanfield.png` | fig8_delta_vs_gamma_reduced | one N-cluster, mirrored |
| `pole_sum_difference_gamma2_long_run.png` | fig10_reduced_gamma2 | one N-cluster in pole 1 |

    python make_figures.py --all      # SSA results are cached in ./cache
    python make_figures.py --checks   # CME, conservation, equilibrium checks

Code shared with the other model: `../plasmid_common/` (plasmid.py, ssa.c, figures.py).

## katlynch/plasmids figure set (`katlynch_figures/`)

The figures of github.com/katlynch/plasmids (the files at the root of this repository),
computed for this model instead of the net-bound one, with that repo's
β, κ, ρ and per-script N. Run `python make_kat_figures.py --all`.

| figure | repo original | script |
|---|---|---|
| `poles_vs_time_N50.png` | 2pole_sims_copyno | odesolve.m |
| `asymmetry_vs_time_N50.png` | 2pole_sims_g0 / g23 / g1 / g2 | odesolve.m |
| `closure_ssa_vs_meanfield_N42.png` | mfa_means | closure_check.m |
| `closure_dropped_terms_N42.png` | mfa_terms | closure_check.m |
| `closure_relative_error_N42.png` | mfa_relerr | closure_check.m |
| `slow_decay_vs_exact_N42.png` | approx_comp_g0 / g1 | decayapprox.m |
| `fraction_free_vs_N_one_pole.png` | 1compartment_comp | percentfree.py |
| `fraction_free_vs_N_two_poles.png` | 2compartment_comp | percentfree.py |
| `leading_eigenvalue_vs_gamma.png` | mc_eigval | mc_evals.py |
| `sym_antisym_eigenvalues_vs_gamma.png` | (unsaved figure 1) | mc_evals.py |
| `pseudo_bifurcation_N25.png` | mc_pseudo_bif | mc_pseudo_bif.py |
| `bifurcation_dN_vs_gamma_N50.png` | 2pole_pitchfork_gammaDelN | bif_plots.m (XPP) |
| `pitchfork_locus_N_vs_gamma.png` | 2pole_pitchfork_gammaNtot | bif_plots.m |

Differences from the repo: newborn start (pole 1 at its stationary cluster
distribution) wherever the repo used a one-pole SSA mean or 5 free / 37 bound;
500 SSA runs in the closure check (repo: 100); dropped closure term
β(Var f − f̄ + Cov(f, A)) (the repo's formula has an extra β); slow-mode rate =
slowest antisymmetric eigenvalue of the full chain, amplitude fitted. The γ = 0
fast-subsystem cusp of bif_plots.m has no counterpart and is omitted.

**No pitchfork in this model.** At any mean-field steady state all fluxes J_j vanish (J_N = 0), so each pole is in detailed balance and the f equations force f₁ = f₂: the symmetric state is the only steady state and it is stable for every γ. The pole asymmetry seen at finite times is a slow transient; its relaxation time grows from ~50 min (γ=0, N=50) to >10¹⁰ min (γ≈2.2). The two bif_plots figures therefore show that relaxation time, and the contour where it equals one cell cycle (50 min), instead of a pitchfork.
