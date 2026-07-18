# CLAUDE.md

## What This Is

Simstack4 is a generalized **simultaneous stacking** pipeline for astronomy. It bins a source catalog into populations (by redshift, stellar mass, UV slope, etc.), builds a PSF-convolved layer matrix, and regresses all populations jointly against a set of sky maps — deblending confused sources that can't be resolved individually. Error bars come from bootstrap A/B splitting. Post-stacking, the code fits modified blackbody SEDs and infers physical properties: infrared luminosities (L_IR), star-formation rates (SFR), dust temperatures, and spectral features (PAH emission, [CII] line strength).

**Why it matters**: At far-infrared and submillimeter wavelengths, deep fields are source-confused — individual galaxies are unresolvable, but their mean properties can still be extracted statistically. This code pushes to new depths by stacking the latest multiwavelength catalogs (COSMOS, JWST-era) against Herschel/SCUBA/ALMA maps.

**Design goals for community release**: general enough for science cases we haven't thought of; straightforward for students to configure and run; easy to debug with synthetic data tests.

## Commands

```bash
# Install dependencies
uv sync --extra dev --extra notebooks

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_stacking_recovery.py -v

# Lint / format
uv run ruff check src/
uv run black src/
```

## Environment Variables

Required; paths are expanded with `os.expandvars()`:

```bash
export MAPSPATH="/path/to/fits/maps"
export CATSPATH="/path/to/catalogs"
export PICKLESPATH="/path/to/output"
```

## Pipeline Flow

```
TOML config (config/*.toml)
  → load_config()                   → SimstackConfig
  → SkyCatalogs.load_catalog()      → catalog DataFrame + PopulationManager
  → load_maps()                     → SkyMaps (dict of MapData)
  → SimstackAlgorithm.run_stacking()
      → _build_per_bin_cache()      → cached layer matrices per map
      → _run_*_error_estimation()   → bootstrap A/B split iterations
      → _solve_linear_system()      → scipy.linalg.lstsq (WLS)
      → _compile_results()          → StackingResults with bin_properties
  → SimstackResults                 → SED fitting, L_IR, SFR, line fluxes
  → SimstackPlots / wrapper JSON save
```

## Key Design Decisions

**Simultaneous fitting**: All populations fit at once via a `(N_pop × N_pix)` layer matrix. One `lstsq` call deblends everything. Never fit populations individually.

**Two bootstrap methods** (`config.error_estimator.bootstrap.method`):
- `"all_bins"`: A/B-split all populations each iteration — captures joint variance including cross-population confusion
- `"per_bin"`: split one population at a time, hold others fixed — captures isolated variance; slightly overshoots (~1.1–1.2×) in the confused regime due to A/B layer anti-correlation
Both record `std((x_A − x_B) / 2)` across iterations — the **half-difference estimator**. The former sum `x_A + x_B` collapsed to near-zero for non-overlapping PSFs due to an algebraic cancellation identity (`x_A + x_B = const` when A/B layers are orthogonal with a 50:50 split). Both use all sources (A+B = full set). Fluxes come from the full solve; iterations only estimate uncertainty.

**Generalized binning**: `ClassificationConfig.binning` is `dict[str, BinConfig]`. Any catalog column can be a bin dimension. `PopulationManager` enumerates combinations via `itertools.product`.

**SED fitting in rest frame**: `Greybody.fit_sed()` converts λ_obs → λ_rest = λ_obs/(1+z). T is T_rest with bounds [15, 60] K. L_IR integrates 8–1000 μm with a 1/(1+z) factor. Always use `CosmologyCalculator` (Planck18) for D_L — the old Hubble-law fallback in `Greybody` is wrong at z > 0.3.

**Fit quality tiers**: Tier A (SNR ≥ `snr_high` in ≥ 3 bands, data-driven), Tier B (≥ `snr_low` in ≥ 2 bands, prior-assisted), Tier C (≤ 1 qualifying band, prior-dominated). Thresholds are set per analysis call (`snr_high=5.0, snr_low=2.0` in PAH runs). Default for plots: `min_tier="A"`.

**Self-contained output**: `wrapper.py` embeds full config + catalog metadata in JSON so results load without the original TOML or catalog.

## Module Map

| Module | Role |
|--------|------|
| `algorithm.py` | Core stacking: layer matrix, PSF stamping, Gram matrix solve, per-bin caching |
| `wrapper.py` | Pipeline orchestration, JSON save/load, bootstrap covariance |
| `results.py` | SED fitting across populations, L_IR/SFR/dust mass, I/O |
| `greybody.py` | Modified blackbody model, temperature priors, MCMC (emcee); `_inflate_band_errors` supports redshift-dependent inflation (`{(z_lo, z_hi): factor}` dict); `_pah_flux_0` / `_physical_wien_flux` for Wien-side PAH+warm-dust (coefficients calibrated from PAH tomography 2026-06-12) |
| `sed_fitting.py` | `CovarianceGreybodyFitter` (Cholesky), `RegressionGreybodyFitter` |
| `pah_model.py` | `PAHModel`: joint multibin PAH forward model (`fit_forward_model_multibin`), optional 9.7 μm silicate absorption (`include_silicate=True`, Drude profile), simulation/plotting helpers. **Old standalone functions** (`fit_forward_model`, `fit_bayesian_forward_model`) are frozen reference — do not edit. |
| `analyze_pah.py` | `combine_pah_spectra`: assembles f₂₄/f_peak tomographic spectrum from multiple stacking wrappers; `fit_pah_model` (empirical, diagnostic only); `staggered_pah_zbins`, `adaptive_pah_zbins` |
| `pah_bandpass.py` | MIPS 24 + MIPS 70 bandpass response curves (`get_bandpass`); 24 μm arrays stay in sync with `pah_model` frozen arrays (guarded by test) |
| `pah_dither.py` | `DitherScheme` (uniform/adaptive, `to_toml_bins()`), `TruthSpectrum` (PAH+warm-MBB injection; optional sSFR-driven within-bin evolution via `eta_ssfr_amp`/`eta_ssfr_ratio`; optional hot/VSG MIR power-law continuum via `mir_plaw_amp` — without it the simulated 24 µm flux is pure PAH and the continuum amplitude unidentifiable; optional `flux_envelope(z, prop_bin)` observed-flux envelope — without it band fluxes are constant-amplitude with z, unlike real stacks that dim ~10× over z=0.5–3.5), `compute_pz_matrix`, `NoiseModel` + shared-source covariance, `simulate_dithered_fluxes`, `fisher_for_scheme`, `fisher_evolution` (CRLB on the evolution slopes; 24-only vs 24+70 identifiability), `evolution_recovery_sweep` (Monte-Carlo bias/scatter/rail-rate of `fit_evolving` vs SNR/bands/prior — sets the minimum detectable η_A), `sweep_strategies` |
| `pah_spectrum.py` | `PAHSpectrumModel`: theoretical GLS deconvolution via design matrix + shared-source covariance; `fit_lstsq` (warm-MBB continuum + free per-group amplitudes), `fit_shared` (cold-greybody baseline + shared feature ratios `r_g` + per-bin `alpha_m`, alternating WLS), `fit_evolving` (multi-band, sSFR-anchored amplitude + line-ratio evolution with shared global slopes `η_A`/`η_g`; 70 µm breaks the amplitude/ratio degeneracy), `fit_with_alpha` (profiles the cold-baseline Wien slope `α` — the `(1+z)^(−α)` power-law splice in `greybody_model` — with a Gaussian prior, default strong at 2; wraps `fit_shared`/`fit_evolving`, re-tilting the baseline; ≥2 bands make α data-driven, 1 band leans on the prior), `fit_evolving_mcmc` (emcee posterior over the evolving-template parameters θ=[η_A, η_g, log r-block] with per-bin (C_m, α_m) profiled analytically; `per_bin_ratios=True` gives every mass bin its own ratio vector — the flexibility knob; anchor the reference group on 7.7+8.6 µm or η_A floats), `feature_envelope="baseline"` (features dim with the source via the reference band's cold baseline — REQUIRED on real observed-flux data or the ~10× dimming envelope leaks into a spurious negative η_A; the branch-5 railed η_A=−2.37 was partly this), `evolving_flux_decomposition` (posterior per-point baseline + per-group contributions for overlay plots), `fit_mcmc`, `pseudo_spectrum`; `smoothed_ms_baseline` helper (Wien-tail baseline stabilization). Multi-band evolving fits normalize all bands by ONE per-bin scalar (2026-07-02 fix: per-band medians forced equal 24/70 continuum levels through the shared C_m; single-band fits unchanged). `DEFAULT_FEATURES` now has 3.3 µm appended at **index 7** (out of wavelength order, deliberately — features are referenced by explicit index so 0–6 and every group/coeff keyed to them stay fixed); only sampled once MIPS 24 reaches rest < 3.3 µm, i.e. z ≳ 6.3 (needs the z→8 binning). `_ratio_block_solve` (2026-07-17) makes the shared-ratio WLS drop feature columns whose weighted power is negligible and pin `r_g=0` — without it a structurally-unsampled feature (3.3 µm at z<6.3, kernel underflows to ~1e-243, NOT exact zero so `solve` doesn't raise and pinv never fires) rails `r_g` to ±1e130; existing fits whose every feature is sampled over their z-range are unchanged. Distinct from `PAHModel` — see PAH Tomographic Stacking section below. |
| `populations.py` | `PopulationManager`: arbitrary-dimension binning |
| `config.py` | TOML → dataclasses: `SplitType`, `BootstrapConfig`, `BinConfig`, `BeamConfig` |
| `sky_maps.py` | FITS loading, WCS, PSF convolution, `MapData` |
| `sky_catalogs.py` | Catalog loading, NUVrJ classification, computed columns (β_UV, L_UV) |
| `cosmology.py` | `CosmologyCalculator`: Planck15/18 D_L, flux↔luminosity, SFR |
| `bin_optimizer.py` | SNR-equalizing bin edge optimization |
| `plots.py` | SED grids, TRF-redshift, L_IR–L_UV–β plots, T_dust–DTG (Parente+2026) |
| `dust_evolution.py` | Two-component dust SED fitter: `DustEvolutionModel`, hierarchical MCMC over (z, M*, σ_SFR) bins; `main_sequence_ssfr(z, log_mstar, relation)` (Speagle+2014 / Schreiber+2015) — the sSFR(z, M*) proxy that drives PAH evolution |

## DustEvolutionModel

**Science question**: Is the steep T_dust(z) rise seen in single-component stacking (Viero+22) real, or an artifact of warm-dust contamination inflating the apparent temperature at high-z?

**Model** (`src/simstack4/dust_evolution.py`, class `DustEvolutionModel`):

```
F_ν = A_c · [GB(λ, T_c, β_c=1.8) + f_w · GB(λ, T_w, β_w=1.5)]

T_c(z)          = T_c0 + b_z · z                              (cold component)
T_w(log_σ_SFR) = T_w0 + c_sigma · log_σ_SFR                  (warm component)
f_w(z,M*,σ)    = 10^(a0 + a_z·z + a_M·log_M* + a_sigma·log_σ)
```

**8 global params** `theta = [T_c0, b_z, T_w0, c_sigma, a0, a_z, a_M, a_sigma]`:
- `T_c0` — cold dust anchor at z=0 (~24 K, Schreiber+18 implies ≈23.7 K)
- `b_z` — T_c slope with z (Schreiber+18: 4.6 K/z); prior [0, 7], σ=5 (weak)
- `T_w0`, `c_sigma` — warm component anchor and σ_SFR slope
- `a0, a_z, a_M, a_sigma` — log warm fraction coefficients

**Key design decisions**:
- `A_c` solved **analytically** at every MCMC step — eliminates M amplitude dimensions from chain
- **Wien power law off in MCMC**: `_greybody_nu(alpha=None)` is pure modified blackbody in the likelihood. Wien (`alpha=2.0`) is used only in `_per_bin_fits` (T_c0 prior estimation) and plotting. Real MIPS 24µm is PAH-dominated, not thermal continuum — including Wien in the likelihood drives f_w→0.
- **MIPS 24µm excluded for z > 2.5** (`_MIPS_ZMAX = 2.5`); rest-frame < 9.6µm is PAH forest
- **Data-driven T_c0 prior**: before MCMC, per-bin greybody fits at low-z set the prior center (20th percentile of low-z T_apparent, σ=3 K)
- `fix_a_M=True` / `fix_a_sigma=True` reduce free params when grid dimensions are degenerate

**Reference T_dust(z) relations** (all in `greybody.py` and `plots.py`):
- Viero+22 (MNRAS 516, L30): `T = 23.8 + 2.7z + 0.9z²` — quadratic, from stacking COSMOS2020
- Schreiber+18 (A&A 609, A30): `T = 32.9 + 4.6(z−2)` — linear, CANDELS+Herschel+ALMA
- Parente+2026 / Sommovigo+2022: radiative equilibrium `T^(4+β) ∝ Σ_SFR^0.286 / DTG` — plotted in `create_tdust_dtg_plot()`

**Adapter**: `stacking_results_to_dust_df(wrapper)` converts `SimstackWrapper` output → `DustEvolutionModel` DataFrame (in the science notebook).

## PAH Tomographic Stacking

**Science question**: What is the PAH emission amplitude as a function of stellar mass (and potentially σ_SFR), and how much does it bias MIPS 24 μm stacked fluxes?

**Method** (`analyze_pah.py` + `pah_model.py`): Each stacking run uses fine redshift bins (Δz=0.15). As z varies, the MIPS 24 μm bandpass sweeps rest-frame 5–16 μm, producing a tomographic pseudo-spectrum of f₂₄/f_peak vs λ_rest = 24/(1+z). Four dither runs with offsets Δz×{0, ¼, ½, ¾} give dense wavelength sampling at ~R=53 while keeping ~140 sources per bin.

**Workflow**:
```python
# 1. Stack with inflation_factors={24: 10000, 70: {(0.0,0.8): 1.0, (0.8,99): 10000}}
wrapper_N.run_analysis_only(**ANALYSIS_KWARGS)

# 2. Combine runs into tomographic spectrum
df = combine_pah_spectra([wrapper_0, wrapper_1, wrapper_2, wrapper_3], split_filter=[0])

# 3. Fit multibin forward model
result = PAHModel(include_silicate=True).fit_forward_model_multibin(
    df, group_col="stellar_mass", feature_groups=[[0],[1,2],[4]])
# → result["alpha_per_bin"], result["tau_sil"], result["group_ratios"]
```

**Forward model** (per mass bin m, shared feature template):
```
flux_m(z) = baseline_m(z) × [1 + α_m · Σ_g r_g · T_g(z)] × exp(−τ_sil · S(z))
```
- `α_m` — per-bin PAH amplitude (≥ 0); the science output
- `r_g` — shared feature-group ratios (r₀ ≡ 1, fitted globally across all mass bins)
- `T_g(z)` — MIPS bandpass-integrated PAH feature template for group g
- `τ_sil · S(z)` — optional 9.7 μm Drude silicate absorption (Drude: λ₀=9.7 μm, γ=3.3 μm)

**Measured values (2026-06-12, 4 runs, Δz=0.15, 197 Tier B points)**:

| log M*/M☉ | α | σ_α | τ_sil |
|-----------|---|-----|-------|
| 8.5–10.3 | 1.077 | 0.833 | 0.000 ± 0.081 |
| 10.3–10.7 | 0.871 | 0.675 | (global) |
| 10.7–12.0 | 0.694 | 0.539 | (global) |

PAH/FIR amplitude decreases with M* at −0.10 dex/dex (PAH deficit). No silicate absorption detected (τ_sil consistent with zero). 70 μm null test passes. See `docs/pah-forward-model-2-summary.md`.

**Four PAH fitting paths** (don't confuse them):
- `PAHModel.fit_forward_model_multibin` (`pah_model.py`) — practical joint fitter on **f₂₄/f_peak** vs z; shared group ratios `r_g`, `independent`/`shared_slope` (empirical λ-power-law) baseline. f_peak-normalized.
- `PAHSpectrumModel.fit_lstsq` (`pah_spectrum.py`) — GLS deconvolution on **absolute flux** with a warm-MBB continuum and **free** per-group amplitudes; full photo-z kernel, Fisher/CRLB strategy evaluation.
- `PAHSpectrumModel.fit_shared` (`pah_spectrum.py`) — **physical-baseline** path: absolute flux, **cold-greybody** continuum (`C_m·f_cold`, optionally `smoothed_ms_baseline`-stabilized) + **shared** ratios `r_g` + per-bin `alpha_m`; reports `A = alpha/C_m` (PAH/continuum, no stray `median(f_cold)` factor). This is the 2026-06 physical-baseline notebook method ported into the library.
- `PAHSpectrumModel.fit_evolving` (`pah_spectrum.py`) — **branch-5 evolving** path: extends `fit_shared` so the amplitude and ratios drift *within* a mass bin with sSFR(z, M*). `alpha_i = alpha_m·10^(η_A·ŝ_i)`, `r_g(ŝ_i) = r_g0·10^(η_g·ŝ_i)`, `ŝ_i = log_ssfr(z_i, M_m) − s_pivot`; shared global slopes `η_A`/`η_g` fit by an outer optimizer wrapping the alternating WLS. **Multi-band** (24+70): MIPS 70 probes a different rest λ at the same z and breaks the amplitude/ratio degeneracy. Per-point `log_ssfr` from the data, else `main_sequence_ssfr` fallback. Errors from the disjoint-fold ensemble (same as `fit_shared`). `eta_prior_sigma` (Gaussian prior on the slopes, ~1 dex) is needed on real data: the slopes are degenerate with the per-bin amplitude when the sSFR lever arm is short and otherwise rail to the bounds with unphysical pivot amplitudes. **On COSMOS2020 (2026-06-29) the evolving fit is scatter-limited** (χ²_red≈6 is real galaxy-to-galaxy PAH scatter, NOT baseline error). A controlled sim with the *correct* baseline + scatter tuned to χ²_red≈6 reproduces a spurious η_A (truth 0 → recovered ~+1.5), so η_A=−2.37 on the real data is an artifact: within-bin evolution is an **upper limit** (min detectable η_A ≈ 0.8 with 24+70+prior / ≈1.4 with 24-only). Separately, the baseline is a power law `f_ν ∝ (1+z)^(−α)` with `α=2` (the greybody Wien-side splice in `greybody_model`, not an exponential tail); **A_pah is strongly α-sensitive** (Δα=±0.5 → A_pah ×3–4 and flips the mass slope), so quote the EW slope (differential, more robust) with an α systematic. Trustworthy result stays the fold-based mass slopes (L_PAH/L_IR≈flat, EW slope +0.37, 3.3σ). Notebook §3d (α-sweep) and §3e (working-sim-vs-real overlay) make both visible.

**Physical-baseline error budget**: the formal instrumental+confusion covariance omits cosmic/sample variance and a diagonal χ² over source-correlated tomographic points over-counts DOF (χ²_red≈5–9). Use the **disjoint-fold ensemble** (`split{N}of3`): refit each fold with `fit_shared` and take the scatter as the error (jackknife) — it both captures cosmic variance and sidesteps the correlated χ². The `√χ²` rescale is over-conservative; do not use it.

**Inflation factors**: 24 μm is always inflated to 10000 (excluded from SED fit) during PAH runs so f_peak is determined by FIR bands only. 70 μm uses redshift-dependent inflation `{(0.0, 0.8): 1.0, (0.8, 99.0): 10000}` — included as FIR anchor at z < 0.8, excluded above.

**Applying PAH correction to greybody fits**: use measured α(M*) to correct f₂₄ before SED fitting (two-pass: fit greybody with 24 excluded → f_peak → subtract f₂₄_PAH = α·T_m(z)·f_peak → re-fit with reduced inflation ~3–5×). This promotes Tier C → Tier B at z~1.5–2.5 where MIPS probes the 7.7+8.6 μm features. Empirical coefficients in `greybody.py::_pah_flux_0` and `_physical_wien_flux` are calibrated from these measurements: `_pah_coeffs = [-0.10, -0.206, 0.066, -0.349]`.

## TOML Configuration Reference

Key config sections:
- `[binning]` — `stack_all_z_at_once`, `add_foreground`, `crop_circles`
- `[error_estimator.bootstrap]` — `enabled`, `method`, `iterations`, `split_fraction`
- `[catalog.classification]` — `split_type` ("labels"/"nuvrj"/"uvj"), `bin_property_columns`
- `[catalog.classification.binning.<dim>]` — `id`, `label`, `bins` (any catalog column). **cosmos2020 3pop catalogs (branch-10) make `starburst` a binning dimension** (`bins=[-0.5,0.5,1.5]`, 0/1 flag) rather than a 4th population class — so the sfg_signal (`split_0`) layer splits into non-SB (`starburst_-0.5_0.5`) + SB (`starburst_0.5_1.5`). This DIVERGES from the cosmos2025 builder (SB = class 3, excluded from `sf_keep`). To reproduce the sf_keeper convention downstream, filter `bin_properties["starburst"]==0` (e.g. `build_pah_spectrum_df(..., starburst_filter=0)` in the money-plots notebook) — else non-SB and SB enter as duplicate rows per (z, mass) cell and distort per-bin fits.
- `[maps.<name>]` — `wavelength`, `path_map`, `path_noise`, `color_correction`, `[maps.<name>.beam]`

## Test Suite

Tests use synthetic data with known injected fluxes — no real FITS or catalogs required (except `test_integration.py`).

| File | Tests | Validates |
|------|-------|-----------|
| `test_stacking_recovery.py` | 16 | Linear algebra: single/multi-source/population recovery, noise, confusion |
| `test_luminosity.py` | 35 | Greybody model, L_IR integration, D_L accuracy, SFR |
| `test_per_bin_errors.py` | 14 | Per-bin vs all-bins bootstrap, diff estimator calibration, sum-estimator regression guard |
| `test_sed_fitting.py` | 51 | SED fitting with covariance and MCMC |
| `test_integration.py` | 10 | Full pipeline: TOML → catalog → maps → stacking → bootstrap |
| `test_dust_evolution_recovery.py` | 14 | DustEvolutionModel: parameter recovery, MIPS dropout, warm-fraction ordering |
| `test_pah_dither_strategy.py` | 28 | `DitherScheme`, `NoiseModel`, Fisher/CRLB bounds, shared-source covariance, photo-z kernel |
| `test_pah_spectrum_recovery.py` | 25 | `PAHSpectrumModel` math, GLS conditioning, band leverage, MCMC recovery |
| `test_pah_shared_baseline.py` | 4 | `fit_shared` injection-recovery (shared ratios + per-bin α), `smoothed_ms_baseline` scatter reduction |
| `test_pah_evolution_recovery.py` | 9 | `fit_evolving` sSFR-evolution recovery (shared η_A/η_g), static-model bias under evolution, 70 µm breaks the ratio degeneracy, null → reduces to `fit_shared`, `eta_prior_sigma` tames runaway, `evolution_recovery_sweep` unbiased + 70 µm helps, `fit_evolving_mcmc` slope recovery + decomposition consistency, per-bin-ratio flexibility, `feature_envelope` recovery under source dimming |
| `test_pah_bayesian_recovery.py` | 11 | Bayesian forward model parameter recovery |
| `test_pah_endtoend.py` | 1 | `@pytest.mark.slow` smoke: map-level spot check through real stacking pipeline |
