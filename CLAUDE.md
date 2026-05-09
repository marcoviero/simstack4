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
- `"all_bins"`: A/B-split all populations each iteration — measures joint variance
- `"per_bin"`: split one population at a time, hold others fixed — measures isolated variance (smaller errors for well-separated bins)
Both use all sources (A+B = full set). Fluxes come from the full solve; iterations only estimate uncertainty.

**Generalized binning**: `ClassificationConfig.binning` is `dict[str, BinConfig]`. Any catalog column can be a bin dimension. `PopulationManager` enumerates combinations via `itertools.product`.

**SED fitting in rest frame**: `Greybody.fit_sed()` converts λ_obs → λ_rest = λ_obs/(1+z). T is T_rest with bounds [15, 60] K. L_IR integrates 8–1000 μm with a 1/(1+z) factor. Always use `CosmologyCalculator` (Planck18) for D_L — the old Hubble-law fallback in `Greybody` is wrong at z > 0.3.

**Fit quality tiers**: Tier A (SNR ≥ 3 in ≥ 3 bands, data-driven), Tier B (2 bands, prior-assisted), Tier C (≤ 1 band, prior-dominated). Default for plots: `min_tier="A"`.

**Self-contained output**: `wrapper.py` embeds full config + catalog metadata in JSON so results load without the original TOML or catalog.

## Module Map

| Module | Role |
|--------|------|
| `algorithm.py` | Core stacking: layer matrix, PSF stamping, Gram matrix solve, per-bin caching |
| `wrapper.py` | Pipeline orchestration, JSON save/load, bootstrap covariance |
| `results.py` | SED fitting across populations, L_IR/SFR/dust mass, I/O |
| `greybody.py` | Modified blackbody model, temperature priors, MCMC (emcee) |
| `sed_fitting.py` | `CovarianceGreybodyFitter` (Cholesky), `RegressionGreybodyFitter` |
| `pah_model.py` | PAH feature emission (sum of Gaussians) for Wien-side SED |
| `populations.py` | `PopulationManager`: arbitrary-dimension binning |
| `config.py` | TOML → dataclasses: `SplitType`, `BootstrapConfig`, `BinConfig`, `BeamConfig` |
| `sky_maps.py` | FITS loading, WCS, PSF convolution, `MapData` |
| `sky_catalogs.py` | Catalog loading, NUVrJ classification, computed columns (β_UV, L_UV) |
| `cosmology.py` | `CosmologyCalculator`: Planck15/18 D_L, flux↔luminosity, SFR |
| `bin_optimizer.py` | SNR-equalizing bin edge optimization |
| `plots.py` | SED grids, TRF-redshift, L_IR–L_UV–β plots |

## TOML Configuration Reference

Key config sections:
- `[binning]` — `stack_all_z_at_once`, `add_foreground`, `crop_circles`
- `[error_estimator.bootstrap]` — `enabled`, `method`, `iterations`, `split_fraction`
- `[catalog.classification]` — `split_type` ("labels"/"nuvrj"/"uvj"), `bin_property_columns`
- `[catalog.classification.binning.<dim>]` — `id`, `label`, `bins` (any catalog column)
- `[maps.<name>]` — `wavelength`, `path_map`, `path_noise`, `color_correction`, `[maps.<name>.beam]`

## Test Suite

Tests use synthetic data with known injected fluxes — no real FITS or catalogs required (except `test_integration.py`).

| File | Tests | Validates |
|------|-------|-----------|
| `test_stacking_recovery.py` | 16 | Linear algebra: single/multi-source/population recovery, noise, confusion |
| `test_luminosity.py` | 35 | Greybody model, L_IR integration, D_L accuracy, SFR |
| `test_per_bin_errors.py` | 10 | Per-bin vs all-bins bootstrap, variance isolation |
| `test_sed_fitting.py` | 51 | SED fitting with covariance and MCMC |
| `test_integration.py` | 10 | Full pipeline: TOML → catalog → maps → stacking → bootstrap |
