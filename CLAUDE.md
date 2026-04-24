# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --extra dev --extra notebooks

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_stacking_recovery.py -v

# Run a single test
uv run pytest tests/test_luminosity.py::TestLIRIntegration::test_amplitude_scaling -v

# Lint
uv run ruff check src/

# Format
uv run black src/

# Build catalog
uv run python src/simstack4/scripts/clean_cosmos_wijesekera.py
```

## Active Branch

All active development is on `streamline-code-2`. The `main` branch is stale. Always work on `streamline-code-2`.

## Environment Variables

Three variables must be set; paths are expanded with `os.expandvars()` throughout the config parser:

```bash
export MAPSPATH="/path/to/fits/maps"
export CATSPATH="/path/to/catalogs"
export PICKLESPATH="/path/to/output"
```

## Architecture

### Pipeline Flow

```
TOML config (config/*.toml)
  → load_config()                   → SimstackConfig (config.py)
  → SkyCatalogs.load_catalog()      → catalog DataFrame + PopulationManager
  → load_maps()                     → SkyMaps (dict of MapData)
  → SimstackAlgorithm.run_stacking()
      → _build_per_bin_cache()      → cached layer matrices per map
      → _run_*_error_estimation()   → bootstrap A/B split iterations
      → _solve_linear_system()      → scipy.linalg.lstsq (WLS)
      → _compile_results()          → StackingResults with bin_properties
  → SimstackResults                 → SED fitting, luminosities, SFR
      → Greybody / CovarianceGreybodyFitter
  → SimstackPlots / wrapper JSON save
```

### Key Design Decisions

**Simultaneous fitting**: All galaxy populations are fit simultaneously, not individually. The layer matrix is `(N_pop × N_pix)` — each row is a PSF-convolved template image. The solver extracts per-population mean flux densities in one lstsq call, deblending clustered populations.

**Two error methods** (`config.error_estimator.bootstrap.method`):
- `"all_bins"`: splits every population A/B simultaneously each iteration — measures joint variance
- `"per_bin"`: splits one population at a time, holds others fixed — measures isolated variance (smaller errors for separated populations)
Both methods always use all sources (A+B = full set, no sources dropped). Fluxes come from the full solve; iterations only estimate uncertainty.

**Generalized binning**: `ClassificationConfig.binning` is `dict[str, BinConfig]` — arbitrary dimensions (redshift, stellar_mass, beta_uv, l_uv, etc.). `PopulationManager` uses `itertools.product` to enumerate all combinations. Legacy `@property` accessors (`redshift_range`, `stellar_mass_range`) exist for backward compatibility.

**SED fitting in rest frame**: `Greybody.fit_sed()` transforms λ_obs → λ_rest = λ_obs/(1+z) before fitting. The T parameter is T_rest with bounds [15, 60]K, stable at all redshifts. L_IR integrates the rest-frame model over 8–1000μm with a 1/(1+z) factor.

**Fit quality tiers**: Tier A (data-driven, SNR≥3 in ≥3 bands), Tier B (prior-assisted, 2 bands), Tier C (prior-dominated, ≤1 band). Tier B uses an analytical amplitude solve. `min_tier="A"` is the default for publication plots.

**Luminosity distance**: Always use `CosmologyCalculator` (Planck18 via astropy). `Greybody`'s old Hubble-law `D_L = c*z/H0` is wrong at z > 0.3 and is retained only as an import fallback.

**bin_properties**: Median catalog statistics per population bin (configured via `bin_property_columns` in TOML). Stored in `StackingResults` and `SEDResults`, surfaced as `median_{col}` columns in `get_population_summary()`.

**Self-contained JSON output**: `wrapper.py` embeds the full config + catalog metadata in the JSON output so results can be loaded and analyzed without the original TOML or catalog files.

### Module Responsibilities

| Module | Role |
|--------|------|
| `wrapper.py` (~1865 lines) | Pipeline orchestration, JSON save/load, bootstrap covariance estimation, hardcoded 8×8 band correlation matrix |
| `algorithm.py` (~1100 lines) | Core stacking: layer matrix construction, PSF stamping, Gram matrix solve, per-bin caching |
| `results.py` (~2600 lines) | SED fitting orchestration across populations, L_IR/SFR/dust mass, I/O |
| `greybody.py` | `Greybody` class: modified blackbody model, temperature priors, MCMC via emcee |
| `sed_fitting.py` | `CovarianceGreybodyFitter` (Cholesky-decomposed covariance), `RegressionGreybodyFitter` |
| `populations.py` | `PopulationManager`: arbitrary-dimension binning via itertools.product |
| `config.py` | TOML parsing into dataclasses; `SplitType`, `BootstrapConfig`, `BinConfig`, `BeamConfig` |
| `sky_maps.py` | FITS loading, WCS, PSF convolution, `MapData` dataclass with `valid_pixel_mask` |
| `sky_catalogs.py` | Catalog loading (pandas/polars), NUVrJ classification, computed columns (β_UV, L_UV) |
| `pah_model.py` | PAH feature emission model (sum of Gaussians) for Wien-side SED |
| `bin_optimizer.py` | `BinOptimizer`: SNR-equalizing bin edge optimization |
| `plots.py` | `plot_sed_grid`, `create_trf_redshift_plot`, `create_lir_luv_beta_plot` |
| `cosmology.py` | `CosmologyCalculator`: Planck15/18 D_L, flux↔luminosity, SFR |

### Performance Notes

The per-bin bootstrap uses three optimizations implemented in Session 9 (see `REFACTOR_CONTEXT.md`):
1. **Layer matrix caching** (`_build_per_bin_cache`): pre-build all layers, replace only the split row per iteration — 10.4× speedup
2. **Gram matrix solve**: update only the split row's contribution, solve on (N_pop × N_pop) instead of (N_pop × N_pix) — 130× faster linear algebra
3. **PSF stamping**: place PSF stamps at source positions instead of full-map FFT convolution — 1000× faster layer creation

### TOML Configuration

Key sections in config files:
- `[binning]` — `stack_all_z_at_once`, `add_foreground`, `crop_circles`
- `[error_estimator.bootstrap]` — `enabled`, `method` ("all_bins"/"per_bin"), `iterations`, `split_fraction`
- `[catalog.classification]` — `split_type` ("labels"/"nuvrj"/"uvj"), `bin_property_columns`
- `[catalog.classification.binning.<dim>]` — arbitrary dimension with `id`, `label`, `bins`
- `[catalog.classification.beta_uv_formula]` / `[catalog.classification.l_uv_formula]` — computed columns
- `[maps.<name>]` — per-band: `wavelength`, `path_map`, `path_noise`, `color_correction`, `[maps.<name>.beam]`

### Test Suite

Tests use synthetic data (known injected fluxes) with no dependency on real FITS files or catalogs, except `test_integration.py` which uses `test_data/`.

| File | Tests | What it validates |
|------|-------|-------------------|
| `test_stacking_recovery.py` | 16 | Core linear algebra: single/multi-source/population recovery, noise, confusion |
| `test_luminosity.py` | 35 | Greybody model, L_IR integration, D_L (astropy match), SFR |
| `test_per_bin_errors.py` | 10 | Per-bin vs all-bins bootstrap, variance isolation |
| `test_integration.py` | 10 | Full pipeline: TOML → catalog → maps → stacking → bootstrap |
| `test_sed_fitting.py` | 51 | SED fitting including covariance and MCMC |

### Known Issues / Backlog

- `results.py` and `wrapper.py` are very long (~2600 and ~1865 lines); candidates for splitting
- `test_basic.py` has a stale import (`ClassificationBins` no longer exists)
- χ²_red pathological values (median ~15,000) observed in some production runs — likely PAH contamination at 24μm or underestimated bootstrap errors for some populations
- `pdb` import still present in `results.py` (leftover debug aid)
- wrapper.py serialize/deserialize/reconstruct logic is largely untested
