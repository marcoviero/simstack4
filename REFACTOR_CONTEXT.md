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

### Repo Layout (with line counts — streamline-code-2)
```
src/simstack4/
  __init__.py            (82)   - Public API, placeholder fallbacks
  algorithm.py           (734)  - Core stacking + bootstrap split logic [was 1013]
  cli.py                 (194)  - Command-line interface
  config.py              (449)  - TOML config parsing, generalized binning
  cosmology.py           (344)  - Luminosity distance, SFR calculations
  populations.py         (719)  - Generalized PopulationManager (arbitrary bin dims) [was 721]
  results.py            (2155)  - Results, GreybodyFitter, CovarianceGreybodyFitter, MCMC
  sky_catalogs.py        (402)  - Catalog loading (pandas/polars) [was 492]
  sky_maps.py            (505)  - FITS map loading, PSF convolution, WCS [was 508]
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
    analyze_results.py         (39)  - Results analysis stub
```
Total: ~11,800 lines Python

---

## 2. Key Changes on streamline-code-2 (vs main)

### Already done:
- **Generalized binning**: ClassificationConfig now uses `binning: dict[str, BinConfig]`
  instead of hardcoded z + M★. Supports arbitrary dimensions (β_UV, L_UV, etc.)
- **PopulationBin generalized**: `bin_ranges: dict` + `medians: dict` with legacy
  `@property` accessors for backward compatibility
- **Bootstrap A/B split approach**: algorithm.py splits each population into A and B
  halves, builds doubled layer matrix, solves simultaneously, sums flux_A + flux_B
- **GreybodyFitter** and **CovarianceGreybodyFitter** with optional MCMC (emcee)
- **Self-contained JSON output**: wrapper.py embeds config + catalog metadata so
  results can be loaded without the original TOML/catalog files
- **Hardcoded 8×8 correlation matrix** (24–850μm) in wrapper.py
- **Dead code from main removed** from algorithm.py (get_population_data_method,
  enhanced_wrapper_integration, string literals)
- **New scripts**: cli_stacking_script.py, cosmos_stacking_clean.py

### Known debris / issues on this branch:
- ~~`sky_catalogs.py`: COSMOS-specific `load_cosmos_catalog` commented out~~ **FIXED**
- ~~`sky_catalogs.py`: Validation commented out with `# pdb.set_trace()`~~ **FIXED**
- ~~`algorithm.py`: Duplicated `_crop_to_circles_bootstrap` / `_crop_to_circles_standard`~~ **FIXED**
- ~~`algorithm.py`: ProgressTracker excessive emoji logging~~ **FIXED**
- ~~`populations.py`: Column validation wrapped in docstring~~ **FIXED**
- `wrapper.py`: 1865 lines — serialize/deserialize/reconstruct logic is very long.
- `results.py`: 2155 lines — GreybodyFitter + CovarianceGreybodyFitter + MCMC is
- `results.py`: 2155 lines — GreybodyFitter + CovarianceGreybodyFitter + MCMC is
  a lot of new code without any tests.
- `results.py`: ~~**BUG #1**~~ **FIXED** — `GreybodyFitter.luminosity_distance()` now delegates
  to `CosmologyCalculator` (Planck18 via astropy). Previously used Hubble-law `c*z/H0`.
  Injected via `cosmology_calc` parameter at init.
- `results.py`: ~~**BUG #2**~~ **FIXED** — `calculate_LIR()` and `_calculate_LIR_single()` now
  integrate over observed-frame wavelengths `[8*(1+z), 1000*(1+z)]μm` with T_obs.
  Previously used rest-frame `[8, 1000]μm`.
- `results.py`: **Both bugs verified fixed** — `test_luminosity.py` regression tests confirm
  fitter D_L matches astropy to machine precision, and calculate_LIR matches independent
  observed-frame integration to < 0.5% at all redshifts z=0.1–3.0.
- `results.py`: **NOTE on T_dust(z)** — Neither bug directly affects the fitted T_dust
  (the fit is self-consistent in observed frame). However:
  - Fit bounds `T_obs ∈ [12, 55]K` mean `T_rest = T_obs*(1+z)` mechanically scales
    with z. At z=3, T_rest is bounded to [48, 220]K.
  - If T_obs hits the upper bound, T_rest rises linearly with z — which could mimic
    rapid T_dust evolution.
  - Both L_IR bugs corrupt the T–L relationship, which could cause misinterpretation
    of T_dust trends.
  - Needs investigation: are the observed T_dust(z) trends real, or artifacts of
    bounds/L_IR errors?

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

**NOTE for Priority 4**: `_run_bootstrap_stacking()` currently uses bootstrap mean as
flux estimates, not the full solve. The full solve IS run (line 228) but only for
systematic errors. Should be changed to match agreed design (full solve = fluxes,
iterations = errors only).

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
| `TestFrameMixing` | 6 | Frame mixing at z≈0, divergence vs z (documents BUG #2), monotonic error growth |
| `TestFluxLuminosityRoundTrip` | 4 | CosmologyCalculator flux↔luminosity inverse, monotonicity |
| `TestSFRConversion` | 2 | Kennicutt relation, plausible SFR from greybody fit |
| `TestCombinedLIRErrors` | 2 | Full error budget table, D_L always underestimates |

**Key finding**: the two L_IR bugs partially cancel:
- D_L (BUG #1): underestimates L_IR by 1.9–4.1× at z=0.5–3.0
- Frame (BUG #2): overestimates L_IR by 1.04–1.31× at z=0.5–3.0
- Net: L_IR underestimated by 1.78–3.14× (D_L dominates)

**Corrected understanding of BUG #2**: Method A (T_rest at rest-frame wavelengths)
is NOT a valid correction — amplitude was fitted in observed frame, so changing T
while keeping A gives wrong normalization. Only Method B (observed-frame wavelengths
8*(1+z) to 1000*(1+z) μm with T_obs) is correct.

### Priority 4: Per-Bin Error Estimation
Goal: Add a `per_bin` error estimation method alongside the existing `all_bins`
approach. Both always use all sources (A + B = full set, no sources dropped).
The only difference is *scope of splitting per iteration*.

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
- [ ] Add `error_method` config option: `"all_bins"` (current), `"per_bin"` (new), `"none"`
- [ ] Implement `_run_per_bin_error_estimation()` in algorithm.py
- [ ] Cache the full layer matrix from step 1; in step 2, only recompute the
      one (or two) rows corresponding to the targeted bin_k
- [ ] Keep `_run_bootstrap_stacking()` as the `"all_bins"` option (rename for clarity)
- [ ] `split_fraction` parameter applies to both methods identically
- [ ] Validate with Test 6 from Priority 2

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
| Support both `all_bins` and `per_bin` | Can compare joint vs isolated variance; user chooses |
| Test framework: pytest with synthetic data | Known-answer tests; no dependency on real FITS files |
| Keep generalized binning (arbitrary dimensions) | Already implemented and working on this branch |
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

### algorithm.py (734 lines) — Core stacking
- `BootstrapSplit` dataclass: container for A/B split indices
- `ProgressTracker`: lightweight progress logging (elapsed/ETA/memory)
- `StackingResults`: expanded dataclass with bootstrap fields
- `SimstackAlgorithm`: main class
  - `run_stacking()` → dispatches to bootstrap or single path
  - `_run_bootstrap_stacking()` → `all_bins` approach (split every bin simultaneously)
    - NOTE: currently uses bootstrap mean as flux estimate — needs Priority 4 fix
  - `_run_single_stacking()` → non-bootstrap path
  - `_stack_single_map_with_bootstrap_splits()` → builds A/B layer specs, delegates
  - `_stack_single_map_standard()` → builds standard layer specs, delegates
  - `_stack_single_map(map_name, layer_specs)` → shared pipeline: layers → foreground → crop → solve
  - `_create_layer_matrix(map_name, layer_specs)` → unified: takes `(label, indices)` pairs
  - `_create_and_convolve_layer()` → PSF convolution for a single layer
  - `_crop_to_circles()` → spatial masking around source positions
  - `_solve_linear_system()` → scipy.linalg.lstsq (WLS if noise map available)
  - `_compile_results()` → assembles StackingResults dataclass
  - **TO ADD**: `_run_per_bin_error_estimation()` → new `per_bin` approach

### cosmology.py (344 lines)
- `CosmologyCalculator`: proper astropy-based cosmology (Planck15/18)
  - `luminosity_distance(z)`: the CORRECT D_L (should replace GreybodyFitter's)
  - `flux_to_luminosity()`: includes K-correction with hardcoded (1+z)^2
  - `luminosity_to_flux()`: inverse of above
  - `comoving_volume_element()`: for number density calculations
- Note: `flux_to_luminosity` K-correction assumes S_ν ~ ν^(-1) power law.
  For stacking (where we fit a greybody), the K-correction should come from
  the fitted SED shape, not a fixed power law. Worth testing.

### config.py (449 lines)
- `BinConfig`: generic bin definition (id, label, bins, optional formula_ref)
- `ClassificationConfig.binning`: `dict[str, BinConfig]` — arbitrary dimensions
- `FormulaParams`: for calculated variables (β_UV, etc.)
- `BootstrapConfig`: now has `split_fraction` field
- `SplitType`: added `FORMULA` option

### populations.py (721 lines)
- `PopulationBin.bin_ranges`: `dict[str, tuple[float, float]]`
- `PopulationManager`: uses `itertools.product` over arbitrary dimensions
- Legacy `@property` accessors: `redshift_range`, `stellar_mass_range`, etc.

### results.py (2155 lines)
- `GreybodyFitter`: modified blackbody SED fitting (T_dust, β, amplitude)
  - `greybody_model()`: modified BB + Wien-side power law
  - `fit_sed()`: fits in observed frame; T_obs bounded [12, 55]K; T_rest = T_obs*(1+z)
  - `calculate_LIR()`: integrates model SED over 8–1000μm → L_sun
    - **BUG #1**: uses Hubble-law D_L (`c*z/H0`), not proper cosmology
    - **BUG #2**: evaluates observed-frame model at rest-frame wavelengths (frame mixing)
  - `luminosity_distance()`: Hubble-law only — must be replaced with CosmologyCalculator
  - `calculate_dust_mass()`: from fit parameters (also uses broken D_L)
- `CovarianceGreybodyFitter`: correlated errors via Cholesky decomposition
- MCMC support via emcee (optional)
- `SimstackResults`: post-processing, luminosity integration, SFR calculation
  - `_create_sed_for_population()`: uses observed-frame wavelengths from map config
    (correct), but note: also uses `CosmologyCalculator.flux_to_luminosity()` for
    rest-frame luminosities, which has a hardcoded K-correction — inconsistent with
    the greybody-based K-correction implicit in the SED fit

### wrapper.py (1865 lines)
- `EnhancedJSONEncoder`: numpy/pandas/enum serialization
- JSON save/load with embedded config + catalog metadata
- `estimate_bootstrap_covariance()`: covariance from bootstrap samples
- Hardcoded 8×8 wavelength correlation matrix (24–850μm)
- Config reconstruction from saved JSON
