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
  algorithm.py          (1013)  - Core stacking + bootstrap split logic
  cli.py                 (194)  - Command-line interface
  config.py              (449)  - TOML config parsing, generalized binning
  cosmology.py           (344)  - Luminosity distance, SFR calculations
  populations.py         (721)  - Generalized PopulationManager (arbitrary bin dims)
  results.py            (2155)  - Results, GreybodyFitter, CovarianceGreybodyFitter, MCMC
  sky_catalogs.py        (491)  - Catalog loading (pandas/polars)
  sky_maps.py            (507)  - FITS map loading, PSF convolution, WCS
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
- `sky_catalogs.py`: COSMOS-specific `load_cosmos_catalog` commented out with `'''`
  block quotes (lines 150–232). Should be cleanly removed or moved to a script.
- `sky_catalogs.py`: Validation of missing columns commented out with
  `# pdb.set_trace()` / `# raise ValidationError(...)` (lines 370-371).
  Must be restored or replaced.
- `algorithm.py`: Duplicated code paths — `_crop_to_circles_bootstrap` /
  `_crop_to_circles_standard` and `_create_bootstrap_layer_matrix` /
  `_create_standard_layer_matrix` share ~80% identical logic.
- `algorithm.py`: ProgressTracker has excessive emoji logging; runtime estimation
  logic in __init__ is long and speculative.
- `wrapper.py`: 1865 lines — serialize/deserialize/reconstruct logic is very long.
- `populations.py`: Column validation wrapped in docstring `"""..."""` (dead code
  that should either be restored or removed).
- `results.py`: 2155 lines — GreybodyFitter + CovarianceGreybodyFitter + MCMC is
  a lot of new code without any tests.
- `results.py`: **BUG #1** — `GreybodyFitter.luminosity_distance(z)` (line 840) uses
  Hubble-law approximation `D_L = c*z/H0`. Diverges ~30% at z=1 from the proper
  cosmological calculation in `CosmologyCalculator`. Used in `calculate_LIR`, so
  L_IR ∝ D_L² is systematically wrong at z > ~0.3.
- `results.py`: **BUG #2** — `calculate_LIR()` (line 761-766) evaluates the greybody
  model at rest-frame wavelengths (8–1000μm) but with the observed-frame temperature
  `T_obs`. This is frame mixing: the model was fitted in observed frame, so should
  either use rest-frame wavelengths with `T_rest`, or observed-frame wavelengths
  `[8*(1+z), 1000*(1+z)]μm` with `T_obs`. This error is z-dependent and compounds
  with the D_L bug.
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

### Priority 1: Dead Code Cleanup
Goal: Remove debris so the codebase is honest about what it does.

- [ ] `sky_catalogs.py`: Remove `'''`-commented `load_cosmos_catalog` method
- [ ] `sky_catalogs.py`: Restore or properly handle column validation (remove pdb ref)
- [ ] `populations.py`: Fix the docstring-wrapped validation code
- [ ] `algorithm.py`: Unify `_crop_to_circles_bootstrap` / `_crop_to_circles_standard`
      into a single `_crop_to_circles(layer_matrix, observed_map, map_name, source_coords)`
- [ ] `algorithm.py`: Unify `_create_bootstrap_layer_matrix` / `_create_standard_layer_matrix`
- [ ] `algorithm.py`: Trim ProgressTracker (remove speculative runtime estimation)
- [ ] General: Remove any remaining `# TODO`, `# HACK`, commented-out blocks

### Priority 2: Escalating Test Suite
Goal: Validate the core linear algebra with known-answer tests, building up complexity.

Each test generates synthetic data (catalog + map) with **known injected fluxes**,
runs the stacking algorithm, and asserts recovered fluxes match within tolerance.

**Test 1 — Single source, single layer, single map**
- 1 population with 1 source at a known position
- 1 map: a Gaussian PSF placed at that position, scaled to a known flux
- Assert: recovered flux matches injected flux to < 1%

**Test 2 — Many sources, same flux, single layer, single map**
- 1 population with ~100 sources at random positions
- 1 map: sum of Gaussian PSFs at all positions, each with the same flux
- Assert: recovered mean flux matches injected flux

**Test 3 — Many sources, same flux, multiple layers, single map**
- N populations (e.g., 5) each with ~100 sources at random positions
- 1 map: sum of all layers, each population has a different known flux
- Assert: recovered flux per population matches its injected flux
- This tests the linear algebra deblending (the core of simstack)

**Test 4 — Many sources, mass-function distributed fluxes, single layer, single map**
- 1 population, sources with flux proportional to stellar mass (Schechter-like)
- Map built from individual source fluxes convolved with PSF
- Assert: recovered MEAN flux matches the population mean of injected fluxes

**Test 5 — Many sources, mass-function fluxes, multiple layers, multiple maps**
- N populations × M maps, each with different beam sizes / noise levels
- Full pipeline test: recover per-population, per-map fluxes
- Assert: all recovered fluxes within expected uncertainties
- Test noise-weighted solve vs unweighted

**Test 6 — Bootstrap / jackknife error estimation**
- Use Test 3 setup, run with bootstrap/jackknife enabled
- Assert: reported uncertainties are consistent with scatter across realizations
- Assert: uncertainties scale approximately as 1/√N_sources

### Priority 3: Luminosity Estimator Validation
Goal: Verify the full chain from stacked flux densities → greybody fit → L_IR → SFR.

**Known issue (found during audit)**:
`GreybodyFitter.luminosity_distance()` (results.py line 840) uses a Hubble-law
approximation `D_L = c*z/H0`, which diverges ~30% at z=1 from the proper cosmological
calculation in `CosmologyCalculator.luminosity_distance()` (cosmology.py, uses astropy).
Must decide: use the astropy calculator everywhere, or accept the approximation and
document it. Recommendation: inject `CosmologyCalculator` into `GreybodyFitter`.

**Test 7 — Greybody model sanity**
- Generate a greybody SED with known (T_dust, β, amplitude)
- Assert: peak wavelength consistent with Wien's law (λ_peak ∝ 1/T)
- Assert: Rayleigh-Jeans slope matches ν^(2+β) at long wavelengths
- Assert: Wien-side power law kicks in at the expected transition frequency

**Test 8 — L_IR integration (analytical cross-check)**
- For a pure Planck function (β=0, no power-law tail), L_IR = σ T^4 × (4π R^2)
  integrated over 8–1000μm. Compare numerical `calculate_LIR` against the
  analytical fraction of the Stefan-Boltzmann result in that wavelength range.
- Test at multiple temperatures (20K, 35K, 50K)

**Test 9 — Luminosity distance consistency**
- Compare `GreybodyFitter.luminosity_distance(z)` vs
  `CosmologyCalculator.luminosity_distance(z)` at z = 0.1, 0.5, 1.0, 2.0, 4.0
- Quantify the divergence (expected: ~30% at z=1, worse at higher z)
- Assert: `CosmologyCalculator` matches astropy to machine precision

**Test 9b — L_IR frame-mixing bug**
- Generate a known greybody SED (T_rest=35K, β=1.8) at z=0 and z=2
- Compute L_IR three ways:
  (a) Current code: `greybody(8-1000μm, A, T_obs, β)` × `4π D_L²` ← WRONG
  (b) Correct method A: `greybody(8-1000μm, A, T_rest, β)` with proper K-correction
  (c) Correct method B: `greybody(8*(1+z)-1000*(1+z)μm, A, T_obs, β)` with conversion
- Assert: methods (b) and (c) agree; quantify how far (a) diverges vs z
- This directly tests whether the frame mixing explains anomalous L_IR values

**Test 10 — Flux ↔ luminosity round-trip**
- Pick a known L_IR at a known redshift
- Convert L_IR → flux via `luminosity_to_flux`
- Convert flux → L_IR via `flux_to_luminosity`
- Assert: round-trip recovery to < 0.1%
- Also test: generate a greybody SED at rest-frame, redshift it, "observe" it,
  fit it, recover L_IR — should match the input

**Test 11 — SFR from L_IR (Kennicutt relation)**
- Inject a known L_IR → compute SFR
- Assert: SFR = L_IR × 1.0e-10 M_sun/yr (Kennicutt 1998) or whichever
  calibration is implemented
- Verify error propagation is self-consistent

**Test 12 — End-to-end: synthetic catalog → stacking → SED fit → L_IR**
- Combine Test 5 (multi-population, multi-map flux recovery) with greybody fitting
- Inject populations with known greybody SEDs at known redshifts
- Run full pipeline: stacking → fit SEDs → compute L_IR
- Assert: recovered L_IR within expected uncertainties of injected values

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
- Begin Priority 1: dead code cleanup
- Begin Priority 2: implement Tests 1–3 (core linear algebra validation)

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

### algorithm.py (1013 lines) — Core stacking
- `BootstrapSplit` dataclass: container for A/B split indices
- `ProgressTracker`: progress logging with ETA (needs trimming)
- `StackingResults`: expanded dataclass with bootstrap fields
- `SimstackAlgorithm`: main class
  - `run_stacking()` → dispatches to bootstrap or single path
  - `_run_bootstrap_stacking()` → current `all_bins` approach (split every bin simultaneously)
  - `_run_single_stacking()` → non-bootstrap path
  - `_create_bootstrap_layer_matrix()` → builds [A_all, B_all] doubled matrix
  - `_create_standard_layer_matrix()` → builds standard N_pop × N_pix
  - `_crop_to_circles_bootstrap()` / `_crop_to_circles_standard()` → duplicated
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
