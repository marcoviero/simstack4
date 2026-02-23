# Simstack4 Refactor — Persistent Context File

> **Purpose**: This file preserves project state across Claude conversation windows.
> Copy-paste the *entire* contents of this file at the start of each new conversation,
> along with any specific file you want to discuss. Update this file at the end of each session.

---

## 1. Project Overview

**Simstack4** is a simultaneous stacking code for infrared/submillimeter astronomy.
Given a catalog of galaxy positions (with redshift, stellar mass, and optionally
other properties like β_UV) and a set of far-IR/submm maps (e.g., Herschel SPIRE),
it bins galaxies into populations, builds a "layer" image per population (delta
functions at source positions convolved with the map PSF), then solves a linear
system to extract the mean flux density per population per wavelength.

**Key science**: Measuring the cosmic infrared background (CIB) by decomposing
confused far-IR maps into contributions from galaxy populations, enabling SED
fitting and star-formation-rate estimation for sources too faint to detect individually.

### Pipeline Flow
```
TOML config
  → SkyCatalogs.load_catalog()
    → PopulationManager (bins galaxies by arbitrary dimensions: z, M★, β_UV, etc.)
  → SkyMaps / load_maps() (reads FITS maps + noise + PSFs)
  → SimstackAlgorithm.run_stacking()
    → per map:
       _create_layer_matrix()  (source layers convolved with PSF)
       _crop_to_circles()      (optional: restrict to regions near sources)
       _solve_linear_system()  (weighted least-squares → flux densities)
  → StackingResults → SimstackResults (post-processing, SEDs, luminosities)
  → GreybodyFitter / CovarianceGreybodyFitter (SED fitting, optionally with MCMC)
  → SimstackPlots (visualization)
```

### Active Branch: `streamline-code-2`

This branch has significant upgrades over main. All work should be done on this branch.

### Repo Layout (with line counts — streamline-code-2, as of Session 14)
```
src/simstack4/
  __init__.py            (82)   - Public API, placeholder fallbacks
  algorithm.py          (~1100) - Core stacking + per_bin cache + Gram solve [was 734]
  cli.py                 (194)  - Command-line interface
  config.py             (~480)  - TOML config parsing, generalized binning, bin_property_columns, psf_file
  cosmology.py           (344)  - Luminosity distance, SFR calculations
  populations.py         (719)  - Generalized PopulationManager (arbitrary bin dims)
  results.py           (~2600)  - Results, GreybodyFitter, MCMC, bin_properties, HDF5 save
  sky_catalogs.py        (402)  - Catalog loading (pandas/polars)
  sky_maps.py            (505)  - FITS map loading, PSF convolution, WCS
  toolbox.py             (638)  - Math/coordinate utilities (overlaps with utils.py)
  utils.py               (275)  - Logging, memory, environment checks
  wrapper.py            (1865)  - Pipeline orchestration, JSON save/load, metadata
  plots.py               (748)  - Matplotlib visualization
  exceptions/             (72)  - Custom exception classes
  scripts/
    cli_stacking_script.py    (412)  - CLI runner with argparse
    cosmos_stacking_clean.py  (198)  - Streamlined COSMOS pipeline
    cosmos_stacking.py        (274)  - Original COSMOS script
    simmap_stacking.py        (277)  - Simulation stacking script
    simstack_simulation_generator.py (602) - Simulation generator
    clean_cosmos_merger.py    (432)  - COSMOS catalog merger
    clean_cosmos_wijesekera.py(280)  - Wijesekera+2026 selection catalog builder
    irx_plots.py             (~400)  - IRX-β and IRX-M* plotting with reference curves
    analyze_results.py         (39)  - Results analysis stub
```
Total: ~13,000 lines Python

---

## 2. Key Changes on streamline-code-2 (vs main)

### Already done:
- **Generalized binning**: ClassificationConfig now uses `binning: dict[str, BinConfig]`
  instead of hardcoded z + M★. Supports arbitrary dimensions (β_UV, L_UV, etc.)
- **PopulationBin generalized**: `bin_ranges: dict` + `medians: dict` with legacy
  `@property` accessors for backward compatibility
- **Bootstrap A/B split approach**: algorithm.py splits each population into A and B
  halves, builds doubled layer matrix, solves simultaneously, sums flux_A + flux_B
- **Per-bin caching + Gram matrix solve**: 10.4× speedup via cached layers, 130× faster
  linear algebra via Gram matrix updates instead of full lstsq
- **PSF stamping**: 1000× faster layer creation for per_bin iterations
- **GreybodyFitter** and **CovarianceGreybodyFitter** with optional MCMC (emcee)
- **Tier-stratified SED fitting**: A (data, SNR≥3), B (prior-assisted), C (prior-dominated)
- **bin_properties**: per-population median catalog statistics stored in StackingResults
- **Self-contained JSON output**: wrapper.py embeds config + catalog metadata so
  results can be loaded without the original TOML/catalog files
- **Measured PSF file support**: `beam.psf_file` in TOML config
- **Hardcoded 8×8 correlation matrix** (24–850μm) in wrapper.py
- **Dead code from main removed** from algorithm.py (get_population_data_method,
  enhanced_wrapper_integration, string literals)
- **New scripts**: cli_stacking_script.py, cosmos_stacking_clean.py,
  clean_cosmos_wijesekera.py (Wijesekera+2026 selection), irx_plots.py

### Known debris / issues on this branch:
- ~~`sky_catalogs.py`: COSMOS-specific `load_cosmos_catalog` commented out~~ **FIXED**
- ~~`sky_catalogs.py`: Validation commented out with `# pdb.set_trace()`~~ **FIXED**
- ~~`algorithm.py`: Duplicated `_crop_to_circles_bootstrap` / `_crop_to_circles_standard`~~ **FIXED**
- ~~`algorithm.py`: ProgressTracker excessive emoji logging~~ **FIXED**
- ~~`populations.py`: Column validation wrapped in docstring~~ **FIXED**
- `wrapper.py`: 1865 lines — serialize/deserialize/reconstruct logic is very long.
- `results.py`: ~2600 lines — GreybodyFitter + CovarianceGreybodyFitter + MCMC +
  bin_properties + HDF5 save. Functional but long.
- `results.py`: ~~**BUG #1**~~ **FIXED** — `GreybodyFitter.luminosity_distance()` now delegates
  to `CosmologyCalculator` (Planck18 via astropy). Previously used Hubble-law `c*z/H0`.
  Injected via `cosmology_calc` parameter at init.
- `results.py`: ~~**BUG #2**~~ **FIXED** — SED fitting now operates in rest frame.
  `fit_sed()` transforms λ_obs → λ_rest = λ_obs/(1+z) before fitting, so the
  temperature parameter is T_rest directly. Bounds [15, 60]K are stable at all z.
  `calculate_LIR()` integrates rest-frame model over 8–1000μm with 1/(1+z) factor.
- `results.py`: **Both bugs verified fixed** — `test_luminosity.py` (32 tests) confirms
  D_L matches astropy to machine precision, and `calculate_LIR` matches independent
  rest-frame integration to < 0.5% at all redshifts z=0.1–3.0.
- `results.py`: ~~**T_dust(z) investigation**~~ **FIXED** (Session 8)
- `results.py`: ~~**N_src=0 display**~~ **FIXED** (Session 13) — uses `raw_results.n_sources`
- `results.py`: ~~**bin_properties type error**~~ **FIXED** (Session 13) — isinstance guards
- ~~`algorithm.py`: **OOM crash in per_bin cache**~~ **FIXED** (Session 13) — 5× memory reduction
- ~~`algorithm.py`: **Progress tracker 494%**~~ **FIXED** (Session 13)
- **χ²_red pathological values** (median 15,543, max ~10³¹): Observed in production run.
  Likely cause: bootstrap errors underestimated for some populations, or greybody model
  mismatch (e.g., PAH contamination at 24μm). Needs investigation.

---

## 3. Prioritized Plan

### Priority 1: Dead Code Cleanup ✅ COMPLETE
Goal: Remove debris so the codebase is honest about what it does.

- [x] `sky_catalogs.py`: Remove `'''`-commented `load_cosmos_catalog` method
- [x] `sky_catalogs.py`: Restore column validation (removed pdb ref, uncommented raise)
- [x] `populations.py`: Fix the docstring-wrapped validation code (restored active raise)
- [x] `algorithm.py`: Unified `_crop_to_circles_bootstrap` / `_crop_to_circles_standard`
      into single `_crop_to_circles()` (mask built from full population source lists,
      correct for both standard and bootstrap since A ∪ B = full set)
- [x] `algorithm.py`: Trimmed ProgressTracker (removed step_times, emoji, _format_time,
      _get_memory_usage; now just elapsed/ETA/memory in one log line)
- [x] `algorithm.py`: Cleaned up emoji-heavy logging throughout (run_stacking,
      _run_bootstrap_stacking, _create_bootstrap_splits, _stack_single_map_with_bootstrap_splits,
      _compile_results, print_results_summary)
- [x] `algorithm.py`: Removed unused imports (os, psutil)
- [x] `algorithm.py`: Removed speculative runtime estimation from __init__
- [x] `algorithm.py`: Unified `_create_bootstrap_layer_matrix` + `_create_standard_layer_matrix`
      into single `_create_layer_matrix(map_name, layer_specs)` taking `(label, indices)` pairs
- [x] `algorithm.py`: Unified `_stack_single_map_standard` + `_stack_single_map_with_bootstrap_splits`
      via shared `_stack_single_map(map_name, layer_specs)` pipeline (foreground → crop → solve)
- [x] `sky_maps.py`: Removed `import pdb` and `pdb.set_trace()` from WCS error handler

**Line count changes**:
- algorithm.py: 1013 → 734 (-279 lines, -28%)
- sky_catalogs.py: 492 → 402 (-90 lines, -18%)
- populations.py: 721 → 719 (-2 lines, validation restored)
- sky_maps.py: 508 → 505 (-3 lines, pdb removed)

**NOTE**: `_run_all_bins_error_estimation()` correctly uses full-solve fluxes (verified
Session 8). The full solve IS run (line 244) for flux estimates; bootstrap iterations
provide errors only.

### Priority 2: Escalating Test Suite — IN PROGRESS
Goal: Validate the core linear algebra with known-answer tests, building up complexity.

Each test generates synthetic data (catalog + map) with **known injected fluxes**,
runs the solver, and asserts recovered fluxes match within tolerance.

**Tests 1–6: Core stacking recovery** — `tests/test_stacking_recovery.py` ✅ 16/16 PASS

Tests operate at the linear algebra level: synthetic Gaussian PSF layers +
standalone lstsq solver, no TOML/WCS/FITS infrastructure needed.

| Test class | # tests | What it validates |
|---|---|---|
| `TestSingleSourceSingleLayer` | 3 | Single source recovery across positions and flux scales |
| `TestManySourcesSingleLayer` | 3 | N-source stacking, scaling with N, overlapping sources |
| `TestMultipleLayersDeblending` | 3 | Simultaneous deblending (2, 5 pops), foreground layer |
| `TestNoisyRecovery` | 3 | Recovery within 3σ, error consistency via Monte Carlo (200 realizations) |
| `TestMeanSubtraction` | 2 | Mean subtraction doesn't bias recovery |
| `TestConfusionRegime` | 2 | Large beam (FWHM=12–15 pix), 10 confused populations |

Helper functions (reusable for future tests):
- `gaussian_psf_layer()` — synthetic layer from source positions + Gaussian PSF
- `build_observed_map()` — construct observed = Σ flux_k × layer_k + noise
- `solve_for_fluxes()` — standalone lstsq solver matching `_solve_linear_system`

**Remaining tests** (not yet implemented):

**Test 7–12: Luminosity estimator validation** → Priority 3

**Future integration tests** (need config/catalog/map infrastructure):
- [ ] Full pipeline test: TOML config → catalog → maps → stacking → results
- [ ] Bootstrap error estimation validation (all_bins and per_bin methods)

### Priority 3: Luminosity Estimator Validation ✅ COMPLETE
Goal: Verify greybody model → L_IR integration → D_L → SFR chain.

`tests/test_luminosity.py` — 32 tests, all passing.

| Test class | # tests | What it validates |
|---|---|---|
| `TestGreybodyModel` | 5 | Wien peak, RJ slope (ν^{2+β}), power-law transition, amplitude scaling, positivity |
| `TestLIRIntegration` | 3 | Amplitude scaling, T monotonicity, direct integration cross-check |
| `TestLuminosityDistance` | 6 | Hubble-law divergence (documents BUG #1), low-z accuracy, H0 mismatch, astropy match |
| `TestFrameMixing` | 6 | Rest-frame L_IR integration consistency across z, monotonicity |
| `TestFluxLuminosityRoundTrip` | 4 | CosmologyCalculator flux↔luminosity inverse, monotonicity |
| `TestSFRConversion` | 2 | Kennicutt relation, plausible SFR from greybody fit |
| `TestCombinedLIRErrors` | 2 | Full error budget table, D_L always underestimates |

**Key finding**: the two L_IR bugs partially cancel:
- D_L (BUG #1): underestimates L_IR by 1.9–4.1× at z=0.5–3.0
- Frame (BUG #2): overestimates L_IR by 1.04–1.31× at z=0.5–3.0
- Net: L_IR underestimated by 1.78–3.14× (D_L dominates)

**Final resolution**: All fitting moved to rest frame (Session 8). The observed-frame
approach was mathematically self-consistent but made temperature bounds management
fragile at high z. Rest-frame fitting eliminates the problem entirely:
- λ_rest = λ_obs / (1+z), T parameter = T_rest, bounds [15, 60]K stable at all z
- L_IR = 4πD_L²/(1+z) × ∫ model(λ_rest) dν_rest over 8–1000μm
- No frame conversions needed in the fitting chain

### Priority 4: Per-Bin Error Estimation ✅ COMPLETE
Goal: Add a `per_bin` error estimation method alongside the existing `all_bins`
approach. Both always use all sources (A + B = full set, no sources dropped).
The only difference is *scope of splitting per iteration*.

**Implementation summary**:

Config (`config.py`):
- Added `method` field to `BootstrapConfig`: `"all_bins"` (default) or `"per_bin"`
- Fixed `split_fraction` default mismatch (was 0.8 in parser, 0.5 in dataclass)

Algorithm (`algorithm.py`):
- Renamed `_run_bootstrap_stacking` → `_run_all_bins_error_estimation`
- Fixed all_bins to use full-solve fluxes (was using bootstrap mean — subtle bias)
- Added `_run_per_bin_error_estimation()`: loops over populations, splits one at a
  time, holds others fixed, collects A+B sum per iteration, takes std
- Added routing in `run_stacking()` based on `bootstrap_method`
- Added `bootstrap_method` validation in `__init__`
- Handles foreground layer correctly (appends 0 error for foreground)

`tests/test_per_bin_errors.py` — 10 tests, all passing:

| Test class | # tests | What it validates |
|---|---|---|
| `TestPerBinBasics` | 3 | Flux recovery, positive/finite errors, identical fluxes vs all_bins |
| `TestPerBinVsAllBins` | 3 | Per-bin ≤ all-bins for separated pops, noise scaling, source count scaling |
| `TestPerBinMultiPopulation` | 2 | 3-pop overlapping scenario, variance isolation (pop A error independent of pop B size) |
| `TestPerBinConfigIntegration` | 2 | Config defaults and per_bin setting |

Key insight from testing: per-bin gives *smaller* errors than all-bins for well-separated
populations because it doesn't introduce artificial cross-population covariance from
simultaneous splitting.

#### Design

**Both methods split sources into A/B halves and solve for flux = A + B.**
**No sources are ever dropped. Split fraction is configurable (50/50, 80/20, etc.).**

| | `error_method="all_bins"` (current) | `error_method="per_bin"` (new) |
|---|---|---|
| **What gets split** | Every bin simultaneously | One bin at a time |
| **Layer matrix** | 2 × N_pop rows (A_all + B_all) | N_pop + 1 rows (full layers + 1 split pair for bin_k) |
| **Iterations** | M total iterations | M iterations × N_bins (but each is cheaper) |
| **Memory** | 2 × N_pop × N_pix | (N_pop + 1) × N_pix |
| **What it measures** | Joint variance across all bins | Isolated variance per bin |

**`per_bin` workflow**:
```
1. FULL SOLVE (all sources, all bins, no splitting)
   → flux_densities[pop][map]                    ← THE FLUX ESTIMATES
   → cache the full layer_matrix (N_pop × N_pix) ← reused in step 2

2. ERROR ESTIMATION (sequential, one bin at a time):
   for each population bin k = 1..N_bins:
       for each map:
           for iteration i = 1..M:
               split bin_k's sources into A/B (using split_fraction)
               replace row k in cached layer_matrix with A_layer and B_layer
                 → layer_matrix becomes (N_pop + 1) × N_pix
               solve full system
               flux_k[i] = flux_A_k + flux_B_k
           uncertainty_k = std(flux_k[1..M])

3. COMBINE:
   Final result per bin = (flux from step 1, uncertainty from step 2)

4. DIAGNOSTICS (optional):
   Flag bins where per_bin uncertainty >> formal lstsq error
   → suggests bright-source contamination or layer degeneracy
```

**Implementation plan**:
- [x] Add `method` config option to `BootstrapConfig`: `"all_bins"` (default), `"per_bin"`
- [x] Implement `_run_per_bin_error_estimation()` in algorithm.py
- [ ] Cache the full layer matrix from step 1; currently rebuilds layers each iteration
      (correct but slower — optimization for later)
- [x] Renamed `_run_bootstrap_stacking()` → `_run_all_bins_error_estimation()`
- [x] `split_fraction` parameter applies to both methods identically
- [x] Validated with `test_per_bin_errors.py` (10 tests)

---

## 4. Session Log

### Session 1 — 2026-02-15
**Goal**: Set up persistent context system; initial codebase audit.
**Done**:
- Cloned repo (main branch), analyzed full structure
- Created initial REFACTOR_CONTEXT.md
**Notes**: Was initially working on main branch.

### Session 2 — 2026-02-15
**Goal**: Switch to streamline-code-2 branch; audit changes; plan priorities.
**Done**:
- Switched to `streamline-code-2` branch
- Full diff analysis vs main (~5,450 lines changed)
- Cataloged all changes: generalized binning, bootstrap A/B split, GreybodyFitter,
  JSON output, new scripts
- Identified debris: commented-out code in sky_catalogs.py, duplicated methods in
  algorithm.py, dead validation in populations.py
- Designed escalating test suite (Tests 1–6)
- Designed `per_bin` error estimation method alongside existing `all_bins`
- Clarified: both methods always use all sources (A + B = full set, no sources
  dropped). Difference is scope: `all_bins` splits every bin simultaneously,
  `per_bin` splits one bin at a time while holding others fixed.
- Agreed: full solve gives fluxes, iterations give errors only
- Named methods `all_bins` / `per_bin` (describes scope, not technique)
- Added Priority 3: luminosity estimator validation (Tests 7–12)
- **Found bug**: GreybodyFitter.luminosity_distance() uses Hubble-law approximation,
  not proper cosmology. All L_IR at z > ~0.3 are systematically biased.
- **Found bug #2**: calculate_LIR() evaluates observed-frame greybody model at
  rest-frame wavelengths — frame mixing that compounds with the D_L error.
- **Corrected analysis**: Neither bug directly affects fitted T_dust (fit is
  self-consistent in observed frame). But both corrupt L_IR in z-dependent ways.
  The T_obs bounds [12,55]K mean T_rest = T_obs*(1+z) mechanically scales with z.
  Need to investigate whether observed T_dust(z) trends are physical, artifacts of
  bounds, or consequences of corrupted T–L relationship.
- Per-bin error estimation moved to Priority 4
- Updated REFACTOR_CONTEXT.md with revised priorities
**Next**:
- Begin Priority 2: implement Tests 1–3 (core linear algebra validation)

### Session 3 — 2026-02-15
**Goal**: Execute Priority 1 dead code cleanup.
**Done**:
- `sky_catalogs.py`: Removed 90 lines — deleted commented-out COSMOS loader,
  restored column validation (removed pdb, uncommented raise)
- `populations.py`: Restored active validation (was wrapped in docstring)
- `algorithm.py`: Removed 238 lines — unified `_crop_to_circles` (eliminated
  bootstrap vs standard duplication), trimmed ProgressTracker to essentials,
  cleaned all emoji logging, removed unused imports (os, psutil), removed
  speculative runtime estimation, cleaned print_results_summary
- `_create_bootstrap_layer_matrix` / `_create_standard_layer_matrix` deferred
- All three files verified with `ast.parse()`
**Net reduction**: 330 lines removed across 3 files

### Session 4 — 2026-02-15
**Goal**: Continue cleanup; finalize Priority 3 (luminosity tests); plan remaining work.
**Done**:
- Clarified `all_bins` / `per_bin` naming (both use A/B splits, differ in scope only)
- Added Priority 3: luminosity estimator validation (Tests 7–12)
- Found **BUG #1**: `GreybodyFitter.luminosity_distance()` uses Hubble-law `D_L = c*z/H0`
- Found **BUG #2**: `calculate_LIR()` evaluates observed-frame model at rest-frame
  wavelengths (frame mixing), compounding with D_L error
- Corrected analysis: neither bug directly affects fitted T_dust (fit is self-consistent
  in observed frame), but both corrupt L_IR in z-dependent ways. T_obs bounds [12,55]K
  cause T_rest = T_obs*(1+z) to mechanically scale with z.
- Completed remaining cleanup from Priority 1:
  - `algorithm.py`: Unified `_create_bootstrap_layer_matrix` + `_create_standard_layer_matrix`
    into single `_create_layer_matrix(map_name, layer_specs)` (-42 lines)
  - `algorithm.py`: Created shared `_stack_single_map(map_name, layer_specs)` pipeline
  - `sky_maps.py`: Removed `import pdb` and `pdb.set_trace()`
- Flagged: `_run_bootstrap_stacking()` uses bootstrap mean for flux estimates, not
  full solve — inconsistent with agreed design (needs fixing in Priority 4)
- All 4 modified files verified with `ast.parse()`
**Total cleanup**: algorithm.py 1013→734, sky_catalogs.py 492→402, sky_maps.py 508→505
**Next**:
- Priority 2: implement escalating test suite (Tests 1–6)
- Priority 3: luminosity estimator tests (Tests 7–12)

### Session 5 — 2026-02-15
**Goal**: Implement Priority 2 — stacking recovery test suite.
**Done**:
- Built `tests/test_stacking_recovery.py` — 16 tests, all passing in 1.1s
- Tests validate core linear algebra at the solver level using synthetic
  Gaussian PSF layers + standalone lstsq solver (no TOML/WCS/FITS needed)
- Six test classes covering escalating complexity:
  - Single source recovery (3 tests)
  - Many-source stacking (3 tests, including overlapping sources)
  - Multi-population deblending (3 tests: 2-pop, 5-pop, with foreground)
  - Noisy recovery (3 tests: 3σ check, residual consistency, Monte Carlo
    error validation across 200 realizations)
  - Mean subtraction invariance (2 tests)
  - High-confusion regime (2 tests: large beam, 10 mixed populations)
- Created reusable test helpers: gaussian_psf_layer, build_observed_map,
  solve_for_fluxes (standalone lstsq matching _solve_linear_system)
- Fixed: replaced unweighted reduced-χ² assertion with residual-std check
**Next**:
- Priority 3: luminosity estimator validation (Tests 7–12)

### Session 5 continued — Luminosity Tests
**Done**:
- Built `tests/test_luminosity.py` — 32 tests, all passing in 5.8s
- Corrected understanding of BUG #2 (frame mixing):
  - Method A (T_rest at rest-frame wavelengths) is NOT valid — amplitude was
    fitted in observed frame, so changing T invalidates the normalization
  - Correct fix: integrate observed-frame model over λ_obs = 8(1+z) to 1000(1+z)μm
  - Frame mixing effect is 4–24% (z=0.5–3), not order-of-magnitude as originally thought
  - Frame mixing OVERESTIMATES L_IR (opposite direction from D_L bug)
- Quantified combined error budget:
  - D_L (BUG #1): underestimates by 1.86× to 4.10× at z=0.5–3.0
  - Frame (BUG #2): overestimates by 1.04× to 1.31× at z=0.5–3.0
  - Net: L_IR underestimated by 1.78× to 3.14× (bugs partially cancel)
- Discovered H0 mismatch: fitter uses H0=70, Planck18 has H0=67.7 → 3.4% baseline
  D_L offset even at z→0. Separate issue from Hubble-law breakdown.
- Total test suite: 48 tests (16 stacking + 32 luminosity), all passing
**Next**:
- Fix the bugs (replace GreybodyFitter.luminosity_distance with CosmologyCalculator,
  fix calculate_LIR integration range) — tests become regression guards
- Priority 4: per_bin error estimation

### Session 5 continued — Bug Fixes
**Done**:
- **BUG #1 fixed**: Added `cosmology_calc` parameter to `GreybodyFitter.__init__`.
  `luminosity_distance()` now delegates to `CosmologyCalculator` (Planck18/astropy).
  Falls back to Hubble-law with warning if astropy unavailable. Removed hardcoded
  `self.H0 = 70` / `self.Om0 = 0.3`. Wired through at `ResultsProcessor` instantiation
  site (both `GreybodyFitter` and `CovarianceGreybodyFitter` via `**kwargs`).
- **BUG #2 fixed**: Changed integration range in `calculate_LIR()` and
  `_calculate_LIR_single()` from `[8, 1000]μm` to `[8*(1+z), 1000*(1+z)]μm`.
- **Tests updated**: Flipped 12 bug-documenting tests into regression tests that
  assert errors are now *absent*. All 48 tests pass (16 stacking + 32 luminosity).
- Code changes in `results.py` only; verified with `ast.parse()`.
**Next**:
- Priority 4: per_bin error estimation

### Session 5 continued — Priority 4: Per-Bin Error Estimation
**Done**:
- Added `method` field to `BootstrapConfig` (`"all_bins"` / `"per_bin"`)
- Fixed `split_fraction` default mismatch in TOML parser (0.8 → 0.5)
- Renamed `_run_bootstrap_stacking` → `_run_all_bins_error_estimation`
- Fixed all_bins flux estimate: now uses full-solve fluxes instead of bootstrap mean
- Implemented `_run_per_bin_error_estimation()` — splits one population per iteration,
  holds others fixed, collects A+B sum, takes std
- Added routing in `run_stacking()`, validation in `__init__`
- 10 tests in `test_per_bin_errors.py`, all passing
- Total: 58 project tests passing (16 stacking + 32 luminosity + 10 per_bin)

### Session 6 — 2026-02-16: Integration Tests + Bug Fix Verification

**Done**:
- Verified all 8 integration tests pass (test_integration.py, already on GitHub)
  - TOML config → load_config → SkyMaps (FITS) → PopulationManager (CSV)
  - Noiseless flux recovery (exact match to injected truth)
  - Foreground layer ≈ 0 when no background present
  - Population source counts preserved through pipeline
  - Noisy recovery within expected errors
  - Bootstrap all_bins completes without error
  - Bootstrap per_bin completes without error
- Verified both L_IR bugs are FIXED in code (done in prior session, confirmed here):
  - BUG #1 (D_L): `GreybodyFitter.luminosity_distance()` now uses `CosmologyCalculator`
    (Hubble-law c*z/H0 retained only as fallback if astropy unavailable)
  - BUG #2 (frame mixing): `_calculate_LIR_single()` now integrates over
    λ_obs = 8*(1+z) to 1000*(1+z) μm with T_obs (not rest-frame 8–1000μm)
  - Key insight: Method A (T_rest at rest-frame wavelengths) is NOT valid because
    amplitude was fitted in observed frame. Only Method B (observed-frame λ with T_obs)
    correctly computes L_IR from fitted parameters.
  - Tests updated: `test_fitter_matches_cosmology_calculator`, `test_frame_consistency_at_high_z`
- 3 bonus tests in luminosity suite (69 vs 66 from last count)
- Total: **69 project tests passing** (16 stacking + 35 luminosity + 10 per_bin + 8 integration)

### Session 7 — 2026-02-16: Mean Subtraction Bias Fix

**Problem**: Layer mean subtraction used global `nanmean` (all pixels including
unobserved zeros), while map mean subtraction used only non-zero pixels. This
mismatch caused 5-25% systematic flux underestimation on maps with unobserved
border regions (zeros at edges, typical for Herschel).

**Analysis**: Three mean subtractions in the pipeline:
1. Map load (`sky_maps.py:286`): mean of non-zero, non-NaN pixels only
2. Layer creation (`algorithm.py:548`): `nanmean` of entire layer (including zeros) ← BUG
3. Crop circles (`algorithm.py:596`): re-mean-subtract cropped region

The defaults (`crop_circles=True`, `add_foreground=True`) masked the bug — the
crop region re-mean-subtraction + constant foreground absorbed the DC offset.
But `crop_circles=False` gave biased results (-7% to -24%).

**Fix**:
- `MapData` gets `valid_pixel_mask` field, set during map loading (before mean sub)
- `_create_and_convolve_layer` mean-subtracts layers over `valid_pixel_mask`
- Unobserved pixels zeroed out in layers

**Tests added**: `TestPartialCoverage` in `test_integration.py`:
- `test_crop_circles_true`: recovery exact with 15% zero-pixel borders
- `test_crop_circles_false`: regression test — previously failed, now passes

**Total: 119 tests passing** (16 stacking + 32 luminosity + 10 per_bin + 10 integration + 51 SED fitting)

### Session 9 — 2026-02-21/22: Per-Bin Caching + Performance Optimization

**Problem**: per_bin bootstrap was rebuilding all layers from scratch every iteration
(614K redundant PSF convolutions for 135 pops × 7 maps × 100 iters). Also had
progress tracker bugs (showing 494% completion) and memory issues.

**Fixes**:
- **Layer matrix caching**: Pre-build full layer matrix once per map, replace only the
  split row each iteration. 10.4× speedup.
- **Gram matrix solve**: Pre-compute A^T W A and update only the split row's contribution
  per iteration → lstsq on (N_pop × N_pop) Gram matrix instead of (N_pop × N_pix).
  130× faster linear algebra per iteration.
- **PSF stamping**: Place PSF stamps at source positions instead of full-map FFT
  convolution → 1000× faster layer creation.
- **Sequential map processing**: Process maps one at a time → 7× less peak memory.
- **Progress tracker**: Fixed per_bin to use n_populations × n_maps as total steps
  (was using bootstrap_iterations, giving "iter 82/10" at 494%).

### Session 10 — 2026-02-22: PSF Normalization Fix

**Problem**: per_bin producing enormous error bars (100× too large).
**Root cause**: PSF kernel normalization mismatch — stamping code used peak=1
convention but the layer builder expected sum-normalized kernels.
**Fix**: Use `sky_maps.create_psf_kernel()` directly; match peak=1 convention.
Added optional measured PSF file loading via `config.beam.psf_file`.

### Session 11 — 2026-02-22: SED Fitter Cleanup + Fit Quality Tiers

**Refactored** `results.py` GreybodyFitter:
- SNR-aware two-pass fitting: Tier A (data-driven, SNR≥3 in ≥3 bands),
  Tier B (prior-assisted, 2 bands), Tier C (prior-dominated, ≤1 band)
- Tier B uses analytical amplitude solve (fixes inflation on Wien side)
- Configurable temperature bounds, SNR thresholds, amplitude bounds
- Replaced print statements with logging, removed hardcoded constants

### Session 12 — 2026-02-22: results.py Cleanup

- Removed duplicate code in GreybodyFitter
- Replaced hardcoded constants with class attributes
- Improved numerical integration
- Cleaned user-specific comments
- Fixed empirical T(z) lookup fragility

### Session 13 — 2026-02-23: Bug Fixes + bin_properties Feature

**Bugs fixed**:
- **N_src=0 display**: `_create_sed_for_population` was reading `pop_bin.n_sources`
  (returning 0) instead of `StackingResults.n_sources` dict. Fixed with fallback.
- **OOM crash in per_bin cache**: Peak 10.6 GB from building full layer list then
  copying to matrix. Refactored `_build_per_bin_cache` to compute crop mask first,
  then build+crop one layer at a time. Peak reduced to ~2.2 GB (5× reduction).
- **bin_properties type error**: `sed_result.bin_properties` was sometimes a string
  instead of dict. Added isinstance guards at assignment, summary, and HDF5 save sites.

**Feature — bin_properties**:
- `config.py`: `ClassificationConfig.bin_property_columns` — extra catalog columns
  to summarize per population bin (e.g., `["l_nuv", "sfr"]`). Binning axis columns
  auto-included. `all_property_columns` property merges and dedupes.
- `algorithm.py`: Computes median of each property column per population using
  `catalog_df.iloc[pop_bin.indices]`. Stored in `StackingResults.bin_properties`.
- `results.py`: `SEDResults.bin_properties` field. Flows into `get_population_summary()`
  as `median_{col}` columns and into `print_results_summary()` dynamic formatting.
- `config.py`: Also merged `BeamConfig.psf_file` with `__post_init__` env var expansion.

### Session 14 — 2026-02-23: Wijesekera+2026 Catalog Builder + IRX Plots

**Context**: Preparing to reproduce Wijesekera, Koprowski et al. (2026) A&A —
"Evolution of dust attenuation in star-forming galaxies with UV slope, stellar
mass, and redshift out to z~5" — using simstack4 on COSMOS data.

**Catalog builder** (`clean_cosmos_wijesekera.py`):
- NUVrJ quiescent removal (Ilbert+2013): `(NUV−r) > 3(r−J)+1` AND `(NUV−r) > 3.1`,
  using rest-frame magnitudes, applied only at z < 4
- Mass-completeness cuts per z-bin (Table A.1, McLeod+2021)
- Starburst removal: SFR/SFR_MS > 3 (Schreiber+2015 main sequence reference)
- Star removal, χ² quality cut
- Output: parquet file with standardized column names for simstack4

**TOML configs** matching the paper's binning:
- `cosmos_irx_mstar.toml`: 8 z-bins × 8 M*-bins = 64 pops (Table A.3, Fig 6/8)
- `cosmos_irx_beta.toml`: 5 z-bins × 7 β-bins = 35 pops (Table A.1, Fig 1)

**IRX plotting** (`irx_plots.py`):
- Single `create_irx_plot()` function, `x_axis` selects IRX-β or IRX-M* mode
- Reference curves: Wijesekera+2026 Eqs 9/10/11, SMC (dA/dβ=0.91), Meurer+99
- Mass-dependent reddening curves when `color_by='stellar_mass'` (Fig 3)
- z-dependent IRX-M* model curves when `x_axis='stellar_mass'` (Fig 6)
- L_UV from `bin_properties` (`median_l_nuv`); falls back to L_IR-only plot
- Adaptive y-axis limits for IRX vs L_IR modes

**Current stacking run**: 3-axis binning (β × z × M*) with `bin_property_columns = ["l_nuv"]`.
Waiting for completion to enable IRX computation.

**Remaining work (backlog)**:
- Per-bin diagnostics: flag bins where per_bin >> formal error
- test_basic.py stale import (ClassificationBins no longer exists)
- wrapper.py: 1865 lines of serialize/deserialize/reconstruct, untested
- ~~Cache full layer matrix in per_bin path~~ **DONE** (Session 9)

**New backlog items**:

- **Template / main-sequence SED fitting**: The smooth greybody model cannot capture
  PAH emission features (3.3, 6.2, 7.7, 8.6, 11.3, 12.7 μm). At z~1–2, the 7.7μm
  PAH shifts into MIPS 24μm, causing excess flux the greybody can't fit → biases
  temperature and amplitude. Candidate approach: fit empirical template libraries
  (Chary & Elbaz 2001, Dale & Helou 2002, Schreiber+2018) parameterized by L_IR or
  heating intensity α. One-parameter families give L_IR directly without overfitting.
  Architecture: `TemplateFitter` class with same interface as `GreybodyFitter`,
  selectable via config. Everything upstream unchanged — fitter only called in
  results.py on N flux points per population. Could also run as hybrid: template
  for full SED (capturing PAH), greybody for FIR-only subset (PACS+SPIRE+SCUBA).
  Open decisions: which library, replacement vs complement to greybody.

- **Optimal binning for signal-to-noise**: Current bins are manually defined (e.g.,
  Δ log M* = 0.25, Δz = 0.5). Some high-z / low-mass bins have very few sources
  and low SNR stacked fluxes, while other bins oversample the parameter space
  (many sources, high SNR, little physical variation across adjacent bins).
  Goal: develop a method to find bin edges that equalize SNR across populations,
  or that maximize information content (minimize total uncertainty on derived
  quantities like L_IR or SFR) subject to a constraint on number of bins.
  Approaches: (a) adaptive binning — iteratively merge low-SNR adjacent bins
  until all exceed a threshold; (b) quantile-based — bin edges at equal source
  counts; (c) information-theoretic — optimize bin edges to minimize total
  posterior uncertainty on the SED parameters. Could be a standalone utility
  that takes a catalog + map noise level and proposes bin edges before stacking.

---

## 5. Key Decisions & Conventions

| Decision | Rationale |
|----------|-----------|
| Branch: `streamline-code-2` | Active development branch; main is stale |
| Full solve for fluxes, iterations for errors only | Best flux estimate uses all data; iterations probe uncertainty |
| `all_bins` / `per_bin` naming | Describes scope of splitting, not technique (both use A/B splits) |
| No sources ever dropped | A + B = full set always; split fraction is configurable |
| `per_bin` approach for memory savings | Memory scales as (N_pop + 1) × N_pix, not 2 × N_pop × N_pix |
| Use CosmologyCalculator for all D_L | GreybodyFitter's Hubble-law approx is wrong at z > 0.3; also fix frame mixing in L_IR |
| Fit SED in rest frame (T parameter = T_rest) | Stable bounds [15, 60]K at all z; avoids T_obs clamping that biased high-z fits |
| Support both `all_bins` and `per_bin` | Can compare joint vs isolated variance; user chooses |
| Test framework: pytest with synthetic data | Known-answer tests; no dependency on real FITS files |
| Keep generalized binning (arbitrary dimensions) | Already implemented and working on this branch |
| bin_properties for catalog stats per population | Results self-contained, no dependency on population manager at analysis time |
| Fitter interface: `GreybodyFitter` / future `TemplateFitter` | Same API, selectable via config; upstream stacking unchanged |
| Wijesekera+2026 as reference analysis | Well-defined selection, binning, and published results to validate against |
| This file is the cross-session "memory" | Copy into new conversations verbatim |

---

## 6. How to Use This File

### Starting a new conversation:
```
Paste the full contents of REFACTOR_CONTEXT.md, then say:

"Continuing simstack4 refactor. Here's our context file.
[paste REFACTOR_CONTEXT.md]
Today I want to work on [specific task]."
```

### Ending a session:
Ask Claude to update the Session Log section and any completed checklist items,
then save the updated file.

### If providing code for review:
Paste the specific file(s) along with the context file. No need to paste the
entire repo every time — the context file has the architecture map.

---

## 7. File-Level Notes

### algorithm.py (~1100 lines) — Core stacking
- `BootstrapSplit` dataclass: container for A/B split indices
- `ProgressTracker`: lightweight progress logging (elapsed/ETA/memory)
- `StackingResults`: expanded dataclass with bootstrap fields + `bin_properties`
- `SimstackAlgorithm`: main class
  - `run_stacking()` → dispatches to bootstrap or single path
  - `_run_all_bins_error_estimation()` → all-bins bootstrap (full-solve fluxes, bootstrap std for errors)
  - `_run_single_stacking()` → non-bootstrap path
  - `_stack_single_map(map_name, layer_specs)` → shared pipeline: layers → foreground → crop → solve
  - `_create_layer_matrix(map_name, layer_specs)` → unified: takes `(label, indices)` pairs
  - `_create_and_convolve_layer()` → PSF convolution for a single layer
  - `_crop_to_circles()` → spatial masking around source positions
  - `_compute_crop_geometry()` → crop mask extraction (used by per_bin cache builder)
  - `_build_per_bin_cache()` → memory-efficient one-layer-at-a-time cache construction
  - `_solve_linear_system()` → scipy.linalg.lstsq (WLS if noise map available)
  - `_compile_results()` → assembles StackingResults + computes bin_properties
  - `_run_per_bin_error_estimation()` → per-bin bootstrap with Gram matrix solve + PSF stamping

### cosmology.py (344 lines)
- `CosmologyCalculator`: proper astropy-based cosmology (Planck15/18)
  - `luminosity_distance(z)`: the CORRECT D_L (should replace GreybodyFitter's)
  - `flux_to_luminosity()`: includes K-correction with hardcoded (1+z)^2
  - `luminosity_to_flux()`: inverse of above
  - `comoving_volume_element()`: for number density calculations
- Note: `flux_to_luminosity` K-correction assumes S_ν ~ ν^(-1) power law.
  For stacking (where we fit a greybody), the K-correction should come from
  the fitted SED shape, not a fixed power law. Worth testing.

### config.py (~480 lines)
- `BinConfig`: generic bin definition (id, label, bins, optional formula_ref)
- `ClassificationConfig.binning`: `dict[str, BinConfig]` — arbitrary dimensions
- `ClassificationConfig.bin_property_columns`: extra catalog columns to summarize per bin
- `ClassificationConfig.all_property_columns`: merged binning axes + explicit extras (deduped)
- `BeamConfig.psf_file`: optional path to measured PSF FITS file (env var expansion)
- `FormulaParams`: for calculated variables (β_UV, etc.)
- `BootstrapConfig`: now has `split_fraction` field + `method` ("all_bins"/"per_bin")
- `SplitType`: added `FORMULA` option

### populations.py (721 lines)
- `PopulationBin.bin_ranges`: `dict[str, tuple[float, float]]`
- `PopulationManager`: uses `itertools.product` over arbitrary dimensions
- Legacy `@property` accessors: `redshift_range`, `stellar_mass_range`, etc.

### results.py (~2600 lines)
- `GreybodyFitter`: modified blackbody SED fitting (T_dust, β, amplitude)
  - `greybody_model()`: modified BB + Wien-side power law
  - `fit_sed()`: fits in REST FRAME; λ_obs → λ_rest = λ_obs/(1+z); T_rest bounded [15, 60]K
  - Tier-stratified fitting: A (data-driven, SNR≥3 in ≥3 bands), B (prior-assisted),
    C (prior-dominated). Tier B uses analytical amplitude solve.
  - `calculate_LIR()`: integrates rest-frame model over 8–1000μm with 1/(1+z) factor → L_sun
  - `luminosity_distance()`: delegates to `CosmologyCalculator` (Planck18 via astropy)
  - `calculate_dust_mass()`: from rest-frame fit parameters + proper D_L
  - `schreiber_temperature_prior()`: returns (T_rest, σ), z-dependent σ (3–5K)
  - `log_prior()`: rest-frame T bounds [15, 60]K + optional Schreiber Gaussian prior
- `CovarianceGreybodyFitter`: correlated errors via Cholesky decomposition
- MCMC support via emcee (optional)
- `SEDResults.bin_properties`: per-population median catalog statistics
- `SimstackResults`: post-processing, luminosity integration, SFR calculation
  - `_create_sed_for_population()`: uses `raw_results.n_sources` (with fallback),
    pulls bin_properties from raw_results
  - `get_population_summary()`: adds `median_{col}` columns from bin_properties
  - `print_results_summary()`: dynamic bin_property column detection
  - HDF5 save: bin_properties stored as per-SED attrs

### wrapper.py (1865 lines)
- `EnhancedJSONEncoder`: numpy/pandas/enum serialization
- JSON save/load with embedded config + catalog metadata
- `estimate_bootstrap_covariance()`: covariance from bootstrap samples
- Hardcoded 8×8 wavelength correlation matrix (24–850μm)
- Config reconstruction from saved JSON
