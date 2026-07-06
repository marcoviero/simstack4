---
name: pah-forward-model
description: Context for pah-forward-model-* branches — measuring PAH-line contamination of MIPS 24/70 µm stacked fluxes via dithered stacking. Use when working on PAH dithering strategy, pseudo-spectrum deconvolution, or the pah_bandpass/pah_dither/pah_spectrum modules.
---

# PAH Forward Model (dithered stacking)

## The problem

At 24 µm the MIPS bandpass is so broad (R ≈ 2.4) that stacked fluxes mix
PAH features with warm-dust continuum. **Dithered stacking** runs the
simultaneous-stacking pipeline many times with offset fine redshift bins,
so rest-frame PAH features (6.2, 7.7, 8.6, 11.3, 12.7, 16.4/17.0 µm)
sweep through the bandpass and modulate the stacked flux vs z — a
pseudo-spectrum from broadband photometry. Two questions define the
branch family:

1. **Strategy** — what bin widths / stagger counts / property splits
   (M*, σ_SFR) maximize the recoverable spectral information?
2. **Deconvolution** — how to invert the pseudo-spectra into the unique
   line strengths that produced them?

## History — what was tried and why it was unsatisfying

**Frozen reference code — do not edit these functions:**

- `analyze_pah.py::measure_pah_excess` — excess over the greybody
  extrapolation at 24 µm. Failed: warm dust contaminates the excess at
  z > 2, and L_IR-based covariates are circular.
- `pah_model.py::fit_forward_model` — detrend-then-fit per bin. Failed:
  free smooth trend absorbs the 7.7 µm bump (spans Δz ≈ 1.8).
- `pah_model.py::fit_bayesian_forward_model` — hierarchical pooling
  helped but stayed 24 µm-only and ignored stagger covariance.

**Active code in the same files** (`pah_model.py::PAHModel` class):
- `fit_forward_model_multibin` — joint fitter across mass bins with shared
  template shape; `include_silicate=True` adds 9.7 µm Drude nuisance
  parameter τ_sil ≥ 0. This is what runs on real data.
- `analyze_pah.py::combine_pah_spectra` — assembles the tomographic
  f₂₄/f_peak DataFrame from multiple stacking wrappers.

## Current architecture (pah-forward-model-2)

Linear kernel formulation, per dither bin i, property bin m, band b:

```
F_i,m,b = C_m · W_b(z_i; T_w, β_w) + Σ_g Ã_g,m · K_ib,g      (A = Ã/C)
K_ib,g  = Σ_k p_i(z_k) · T_g,b(z_k)
```

- `p_i(z)`: true-z probability of bin i — encodes bin width, dN/dz,
  photo-z smearing σ_z0(1+z), and a catastrophic-outlier pedestal f_cat.
- Continuum = **physical warm modified blackbody** (not a polynomial —
  that was the detrend failure mode); MIPS 70 pins it where 24 µm cannot.
- Amplitudes are linear → GLS; MCMC samples only globals
  (T_w, log A0_g, β_M, β_σ) with C_m profiled analytically — the
  `DustEvolutionModel` pattern.
- **Fisher/CRLB strategy evaluation**: the model is linear, so
  (XᵀΣ⁻¹X)⁻¹ gives exact bounds for any scheme without fitting.
- **Shared-source covariance**: staggered runs reuse sources;
  Cov_ij = σ_i σ_j N_shared/√(N_i N_j) per band. This is what makes the
  resolution claims honest.

### Key findings from simulations (2026-06-11 notebooks; re-derive if assumptions change)

- The bandpass is the resolution floor: dz < 0.1 and σ_z0 < 0.03 buy
  almost nothing.
- Staggering is **sub-additive** at the kernel floor (5 staggers of
  dz=0.1 gain ≲5%) but **rescues coarse bins** (up to ~7× at dz=0.4).
- Ignoring photo-z in the kernel biases amplitudes by tens of percent;
  matched kernels are percent-level even at σ_z0=0.06, f_cat=0.05.
- Detection SNR (feature flux Ã) is noise-limited; ratio precision
  (A = Ã/C) is continuum-limited — quote both (`FisherResult.crlb_flux`
  vs `.crlb`).
- Simulation winning baseline: dz=0.10, n_stagger=2, MIPS 24+70 jointly.
- Property splits cost √M per bin; measure evolution slopes with the
  pooled `fit_mcmc`, not per-bin fits.

### Key findings from real data (`PAHModel.fit_forward_model_multibin`)

**Actual stacking scheme used**: dz=0.15, **4 dither runs** (offsets 0, Δz/4, Δz/2, 3Δz/4),
not the dz=0.10 simulation baseline. Δz=0.15 gave ~140 Tier B sources/bin; dz=0.10 gave ~75
(mostly Tier C), confirming coarser bins are better for this catalog depth.

**Results — Run 2 (2026-06-12, 197 Tier B points, 3 mass bins, λ_rest = 5.4–16.0 μm)**:

| log M*/M☉ | α (PAH amplitude) | σ_α | SNR |
|-----------|-------------------|-----|-----|
| 8.50–10.30 | 1.077 | 0.833 | 1.3 |
| 10.30–10.70 | 0.871 | 0.675 | 1.3 |
| 10.70–12.00 | 0.694 | 0.539 | 1.3 |

- Mass slope: d log α / d log M* = −0.10/dex (PAH/FIR decreases with M*)
- τ_sil = 0.000 ± 0.081 — no silicate absorption detected
- 70 μm null test: PASS (all bins)
- Bump SNR: 2.0–2.6× (marginal visual detection)
- χ²_red = 3.365 (baseline scatter ~1.8× formal errors; rescale σ_α × 1.83 for honest errors)
- Median f₂₄/f_peak = 2.0%, range [0.5%, 4.8%]

Updated `greybody.py` empirical coefficients: `_pah_coeffs = [-0.10, -0.206, 0.066, -0.349]`
(mass slope sign-flipped vs prior calibration; normalization preserved at log M*=10.5).

**Additional runs (notebooks set up, pending execution)**:
- **Run 2b (2026-06-12)**: 4-bin mass scheme [8.5, 10.2, 10.6, 11.0, 12.0], same uniform Δz=0.15.
- **Run 2c (2026-06-14)**: σ_SFR cross-cut, 2 mass × 3 σ_SFR bins; 3/4 runs complete. Expected direction: α decreases with σ_SFR (UV field → grain destruction).
- **Run 2d (2026-06-15)**: accordion vs uniform z-bin comparison (same 4-bin mass scheme). Accordion stagger 0.025/run, narrow bins (Δz=0.10) in high-gradient PAH zones.

Full summary including all runs: `docs/pah-forward-model-2-summary.md`.
Publication strategy and talk figure set: `docs/pah-forward-model-3-brief.md`.

## Code map

| Path | Role |
|------|------|
| `src/simstack4/pah_bandpass.py` | MIPS 24 + MIPS 70 response curves (`get_bandpass`); 24 µm arrays mirror frozen `pah_model` (guarded by test) |
| `src/simstack4/pah_dither.py` | `DitherScheme` (uniform/adaptive, `to_toml_bins()`), `TruthSpectrum` (independent direct-integration path; sSFR evolution via `eta_ssfr_amp`/`eta_ssfr_ratio`; hot/VSG MIR power-law continuum via `mir_plaw_amp` — without it simulated 24 µm flux is pure PAH and C_m is unidentifiable; `flux_envelope(z, prop_bin)` observed-flux dimming envelope, calibrate to the real smoothed f24_cold(z, M*)), `compute_pz_matrix`, `NoiseModel` + shared-source covariance, `simulate_dithered_fluxes`, `fisher_for_scheme`, `fisher_evolution`, `evolution_recovery_sweep`, `sweep_strategies`, `injection_recovery_sweep` |
| `src/simstack4/pah_spectrum.py` | `build_design_matrix`, `warm_continuum_kernel`, `solve_linear_amplitudes` (GLS), `PAHSpectrumModel.fit_lstsq/.fit_shared/.fit_evolving/.fit_evolving_mcmc/.fit_with_alpha/.fit_mcmc/.pseudo_spectrum`, `evolving_flux_decomposition` (posterior baseline + per-group flux split for overlays). Multi-band evolving fits normalize all bands by ONE per-bin scalar (2026-07-02 fix — per-band medians silently forced equal 24/70 continuum levels through the shared C_m; single-band unchanged). Anchor the reference feature group on 7.7+8.6 µm or η_A floats. On real observed-flux data pass `feature_envelope="baseline"` (2026-07-03): features must dim with the source or the ~10× envelope leaks into a spurious negative η_A |
| `src/simstack4/plots.py` (end of file) | `plot_dither_kernels`, `plot_fisher_summary`, `plot_strategy_sweep`, `plot_pseudo_spectrum_overlay`, `plot_pah_spectrum_corner`, `plot_pah_flux_decomposition` (f_band vs z per mass bin with stacked shaded feature-group contributions) |
| `src/simstack4/scripts/pah_dither_endtoend.py` | Map-level spot check through the real stacking pipeline (`--quick` for smoke) |
| `tests/test_pah_dither_strategy.py` | Tier 1 simulator/kernel/Fisher + Tier 2 GLS recovery (incl. photo-z negative control) |
| `tests/test_pah_spectrum_recovery.py` | Tier 1 math, Tier 2 conditioning/band leverage, Tier 3 MCMC recovery |
| `tests/test_pah_endtoend.py` | `@pytest.mark.slow` end-to-end smoke |
| `notebooks/build_pah_dither_strategy_notebook.py` | → `2026-06-11-pah-dither-strategy-explorer.ipynb` (problem 1: scheme design) |
| `notebooks/build_pah_spectrum_notebook.py` | → `2026-06-11-pah-forward-model-sanity.ipynb` (problem 2: GLS deconvolution) |
| `notebooks/build_pah_evolving_mcmc_notebook.py` | → `2026-07-02-pah-evolving-template-mcmc-simulation.ipynb` (evolving-truth injection → MCMC flexibility ladder L1–L4 → f24(z) shaded feature-group decomposition; 2026-07-02) |
| `notebooks/build_pah_money_plots_notebook.py` | → `2026-07-03-pah-money-plots.ipynb` (money plots with POOLED centrals + fold errors; §2b intrinsic band ratios; §2c sSFR-coloured + standard decompositions; §3c z-sliced L_PAH/L_IR crossing pattern — per-mass-bin templates REQUIRED, a global template flattens it; §3d slice-slope sweep vs branch bands) |
| `notebooks/2026-06-12-load-json-fit-seds-redshift-stellar-mass-PAH-dithered-dz015.ipynb` | Real-data analysis: 3-bin and 4-bin mass runs → `combine_pah_spectra` → `PAHModel.fit_forward_model_multibin` → α(M*), τ_sil, null test |
| `notebooks/2026-06-14-load-json-fit-seds-redshift-stellar-mass-sigma_sfr-PAH-dithered-dz015.ipynb` | σ_SFR cross-cut: 2 mass × 3 σ_SFR bins; group_col="sigma_sfr" (alias for log_sigma_sfr) |
| `notebooks/2026-06-15-pah-3bin-vs-4bin-mass-comparison.ipynb` | Accordion vs uniform z-bin comparison; same 4-bin mass scheme both schemes |
| `config/cosmos25_PAH_dithered.toml` | All dither schemes as commented reference blocks; set active `bins =` to desired run |
| `config/cosmos25_PAH_dithered_3d.toml` | σ_SFR config (2 mass × 3 σ_SFR bins) |
| `docs/pah-forward-model-2-summary.md` | Full measurement summary including all runs, coefficient derivation, forward path |
| `docs/pah-forward-model-7-summary.md` | Branch-7 summary: band-ratio mechanism signal (envelope-aware calibration), Narayanan confrontation, evolution-required result, machinery delivered |
| `docs/pah-forward-model-8-brief.md` | Branch 8: talk figures → pivoted to figure corrections (pooled centrals, §3c/§3d); styling tasks still open |
| `docs/pah-forward-model-9-brief.md` | Branch 9: stress tests + literature confrontation (template systematic first) |
| `docs/pah-forward-model-3-brief.md` | Branch 3 goals: publication figures |
| `docs/pah-forward-model-4-brief.md` | Branch 4 goals: shared-slope baseline + K-fold catalog splitting |
| `docs/pah-forward-model-5-brief.md` | Branch 5 goals: error rescaling, robustness, talk figures, T_dust correction |
| `src/simstack4/scripts/prepare_cosmos2020_catalog.py` | K-fold COSMOS2020 parquet generator; `uv run prepare-cosmos2020-catalog` |

## How to run

```bash
uv run pytest tests/test_pah_dither_strategy.py tests/test_pah_spectrum_recovery.py -q
uv run pytest tests/test_pah_endtoend.py -m slow -q       # ~10 s smoke
uv run python -m simstack4.scripts.pah_dither_endtoend --quick
uv run python notebooks/build_pah_dither_strategy_notebook.py   # regenerate
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/2026-06-11-pah-dither-strategy-explorer.ipynb
```

Typical analysis loop:

```python
from simstack4.pah_dither import DitherScheme, TruthSpectrum, \
    simulate_dithered_fluxes, fisher_for_scheme
from simstack4.pah_spectrum import PAHSpectrumModel

scheme = DitherScheme.uniform(dz=0.10, n_stagger=2)
fr = fisher_for_scheme(scheme)                    # strategy FoM, no fitting
sim = simulate_dithered_fluxes(scheme, TruthSpectrum(beta_mass=0.35))
model = PAHSpectrumModel(sigma_z0=0.01)
res = model.fit_lstsq(sim["df"], cov=sim["cov"], scheme=scheme)
```

For real stacking output: build a DataFrame with columns
(run_id, z_lo, z_hi, z_mid, prop_bin_id, n_sources, MIPS_24, MIPS_24_err, …)
— the fitter reconstructs the scheme from it (`_scheme_from_df`) or accepts
`scheme=` explicitly.

## Conventions

- 3-tier tests (simulator consistency → MAP recovery → MCMC), PARAM_TOL=0.20,
  MCMC presets with 32 walkers — see `tests/test_dust_evolution_recovery.py`.
- Notebooks are generated by `notebooks/build_*.py` scripts (nbformat),
  date-prefixed, committed with outputs.
- `analyze_pah.py` and `pah_model.py` stay frozen;
  `DitherScheme.adaptive` imports `staggered_pah_zbins` read-only.

## Key references

Full citations with one-line roles: `docs/pah-refs.md`

Quick lookup:

| When you need... | Paper |
|---|---|
| PAH physics from first principles (single-photon heating, band ratios, U) | Tielens (2008) ARA&A 46, 289 |
| Local PAH deficit slope vs sSFR/L_IR to compare α(M*) against | Smith et al. (2007) ApJ 656, 770 |
| Multi-variate local anchor (M*, metallicity, sSFR) | Galliano et al. (2021) A&A 649, A18 |
| Radiation field → PAH destruction mechanism (U anti-correlation, PHANGS-JWST) | Egorov et al. (2025) A&A 703, A103 |
| Resolved PAH destruction in HII regions (PHANGS framework) | Leroy et al. (2023) ApJS 264, 10 |
| Theoretical α(M*, z) prediction to compare our slope against | arXiv:2606.20809 (PAH lifecycle sims, 2026) |
| Direct JWST MIRI PAH spectroscopy at z~1–3 (potential tension/comparison) | arXiv:2606.18244 (PAHSPECS, 2026) |
| T_dust(z) prior and motivation for PAH correction | Schreiber et al. (2018) A&A 609, A30 |
| Our own T_dust(z) stacking result (predecessor) | Viero et al. (2022) MNRAS 516, L30 |

---

## Open questions

**Answered in pah-forward-model-2:**
- ✓ `NoiseModel.sigma_ref`: real bootstrap errors validated at ~9 SNR/point median for Tier B.
- ✓ Stagger resolution: confirmed dz=0.15 × 4 runs outperforms dz=0.10 × 2 runs for this catalog depth.
- ✓ τ_sil: no silicate absorption detected (τ_sil = 0.000 ± 0.081) in MS galaxies at z~0.5–3.5.
- ✓ Mass trend direction: α decreases with M* (PAH deficit; consistent with GOALS/Spitzer IRS literature).

**For pah-forward-model-4 (current branch):**
- [ ] **Shared-slope baseline**: does fixing γ from off-feature windows stabilise the mass ordering of α?
  Compare α(M*) from `baseline_method="independent"` vs `"shared_slope"` on the same data.
- [ ] **χ²_red diagnosis**: does the shared-slope baseline reduce χ²_red, or does it stay ~8–9?
  If it drops, the elevation was partly baseline misfit. If not, it's pure astrophysical scatter.
- [ ] **K-fold independence**: do 3 independent pseudo-spectra (K=3 COSMOS2020 splits) give
  consistent α values? std(α_A, α_B, α_C) / √(K−1) should agree with rescaled bootstrap σ_α.
- [ ] **γ value**: what power-law slope does the off-feature continuum give? Expected 1.5–2.5.
  Lower-mass galaxies should have higher A_m (more stochastic VSG heating at high sSFR).

**For pah-forward-model-5:**
- [ ] **Rescaled-error significance**: joint α(M*) slope SNR after ×√χ²_red rescaling.
- [ ] **σ_SFR direction**: does α decrease with σ_SFR at fixed M*? Confirms radiation-field mechanism.
- [ ] **T_dust bias**: mean ΔT_dust at z=1.5–2.5 from un-corrected 24 μm; fraction of bins promoted Tier C→B.
- [ ] **Robustness suite**: baseline method variation, ±0.1 dex bin edge shifts, jackknife over runs.
- [ ] **12.7 μm r₂ constraint**: more runs at z~0.7–1.1 would pin 12.7/6.2 ratio.
- [ ] **AGN contamination**: is (T_w, β_w) single continuum sufficient at high σ_SFR?
