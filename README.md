# Simstack4: Simultaneous Infrared Galaxy Stacking

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Package Manager](https://img.shields.io/badge/package%20manager-uv-green.svg)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Simstack4** is a simultaneous stacking pipeline for measuring average infrared flux densities from populations of galaxies. It accounts for source clustering bias through simultaneous fitting of multiple population layers, with integrated SED fitting, bootstrap error estimation, and publication-quality plotting.

Building on [Viero et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...779...32V) and [Viero et al. 2022](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516L..30V), simstack4 replaces the nested loop structure of simstack3 with a matrix-based approach and adds greybody SED fitting with MCMC support, covariance-aware fitting, and automated bin optimization.

## Quick Start

### Installation

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/marcoviero/simstack4.git
cd simstack4
uv sync --extra dev --extra notebooks
```

### Environment Variables

Add to your shell config (`~/.zshrc` or `~/.bashrc`):

```bash
export MAPSPATH="/path/to/your/maps"
export CATSPATH="/path/to/your/catalogs"
export PICKLESPATH="/path/to/your/pickles"
```

### Run a Stacking Pipeline

```python
from simstack4.wrapper import SimstackWrapper

wrapper = SimstackWrapper(
    "config/cosmos25.toml",
    read_maps=True,
    read_catalog=True,
    stack_automatically=True,
)

# Run SED analysis
results = wrapper.run_analysis_only(
    use_mcmc=False,
    temperature_prior="flat",       # "flat", "schreiber", or "viero"
    use_covariance=True,
    inflation_factors={24: 10},     # downweight 24µm
)

# Get summary table
summary = results.get_population_summary()
summary.to_csv("results.csv")
```

## How Simstack Works

When measuring infrared emission from faint galaxy populations, individual sources are often below the detection threshold. Traditional stacking averages flux at source positions but ignores clustering bias — nearby galaxies contribute correlated signal.

Simstack solves this by fitting all populations simultaneously:

1. **Population binning**: split galaxies by redshift, stellar mass, UV slope, etc.
2. **Layer creation**: build 2D templates with unit sources at galaxy positions
3. **PSF convolution**: convolve each template with the instrument beam
4. **Simultaneous fitting**: solve `observed_map = Σ (N × F_avg × template) + noise` via least-squares

This yields unbiased average flux densities per population, which are then fit with greybody SEDs to derive dust temperatures, IR luminosities, and star formation rates.

## Module Structure

```
src/simstack4/
├── wrapper.py          # Main entry point and pipeline orchestration
├── config.py           # TOML configuration parsing
├── sky_catalogs.py     # Catalog loading and column validation
├── populations.py      # Population binning, formulas (β_UV, L_UV)
├── sky_maps.py         # Map loading and PSF handling
├── algorithm.py        # Simultaneous stacking algorithm
├── greybody.py         # Greybody model, Planck function, priors, L_IR
├── sed_fitting.py      # Covariance-aware fitting, regression fitting
├── results.py          # Results orchestration, I/O, derived quantities
├── plots.py            # Publication-quality plotting
├── bin_optimizer.py    # Automated bin edge optimization
└── cosmology.py        # Cosmological calculations
```

### SED Fitting Architecture

The fitting pipeline has three layers:

- **`greybody.py`** — `Greybody` class: the core modified blackbody model with Wien-side power-law extension, temperature priors, curve_fit + MCMC fitting, and L_IR integration. Backwards-compatible alias `GreybodyFitter = Greybody`.

- **`sed_fitting.py`** — `CovarianceGreybodyFitter(Greybody)`: extends the base fitter with inter-band covariance matrices (instrumental + bootstrap), Cholesky-decomposed likelihood. `RegressionGreybodyFitter`: multi-population polynomial regression fitting.

- **`results.py`** — `SimstackResults`: orchestrates fitting across all populations, manages I/O, computes derived quantities (L_IR, SFR, dust mass). Re-exports all fitting classes for backwards compatibility.

## Configuration

Configuration uses TOML format. Key sections:

### Catalog and Binning

```toml
[catalog]
path = "$CATSPATH/cosmos"
file = "COSMOSWeb_wijesekera_sfg.parquet"

[catalog.astrometry]
ra = "ra"
dec = "dec"

[catalog.classification]
# Columns whose median values are stored per population bin
# (e.g., median L_UV within each redshift × mass × β bin)
bin_property_columns = ["calculated_l_uv"]

# Optional: NUVrJ star-forming / quiescent split
split_type = "nuvrj"
[catalog.classification.split_params]
id = "split_label"
[catalog.classification.split_params.bins]
"NUV-r" = "mabs_nuv"
"r-J" = "mabs_r"
"J" = "mabs_j"

# Binning dimensions
[catalog.classification.binning.redshift]
id = "zpdf_med"
label = "Redshift"
bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0]

[catalog.classification.binning.stellar_mass]
id = "mass_med"
label = "Stellar Mass"
bins = [8.5, 10.0, 10.3, 10.6, 10.9, 11.2, 12.0]

[catalog.classification.binning.beta_uv]
id = "calculated_beta_uv"
label = "UV Slope β"
bins = [-2.0, -0.95, -0.55, -0.19, 0.33, 1.5]
```

### Computed Columns (Formulas)

Simstack4 can compute derived columns from catalog data before binning. Formulas are evaluated in order, so later formulas can reference earlier outputs.

```toml
# β_UV from E(B-V) and dust attenuation law
# β = β_intrinsic + k_λ × E(B-V)
# where k_λ depends on the attenuation law (Calzetti/Arnouts/Salim)
[catalog.classification.beta_uv_formula]
formula = "beta_uv"
bins = {"E(B-V)" = "ebv_minchi2", "dust_law" = "law_minchi2"}
single_law = "calzetti"    # or use per-source law from catalog

# L_UV at 1600Å from L_NUV(2300Å), corrected using β_UV
# L(1600) = L(2300) × (1600/2300)^β
# Using L_NUV as L_UV underestimates IRX by 44-130%!
[catalog.classification.l_uv_formula]
formula = "l_uv_1600"
bins = {"l_nuv" = "l_nuv", "beta_uv" = "calculated_beta_uv"}
```

### Maps

```toml
[maps.mips_24]
wavelength = 24.0
color_correction = 1.0
path_map = "$MAPSPATH/cosmos/mips_24_GO3_sci_10.fits"
path_noise = "$MAPSPATH/cosmos/mips_24_GO3_unc_10.fits"
[maps.mips_24.beam]
fwhm = 6.32

[maps.spire_psw]
wavelength = 250.0
color_correction = 1.018
path_map = "$MAPSPATH/cosmos/cosmos_spire_psw.fits"
path_noise = "$MAPSPATH/cosmos/cosmos_spire_psw_noise.fits"
[maps.spire_psw.beam]
fwhm = 18.1
```

### Bootstrap Error Estimation

```toml
[error_estimator.bootstrap]
enabled = true
iterations = 100
initial_seed = 42
```

## SED Analysis

### Temperature Priors

Three options for the dust temperature prior in greybody fitting:

| Prior | Relation | Source |
|-------|----------|--------|
| `"flat"` | Uniform over [T_min, T_max] | Default — let the data decide |
| `"schreiber"` | T = 32.9 + 4.60(z − 2) | Schreiber et al. 2018 (linear, valid z~0–4) |
| `"viero"` | T = 23.8 + 2.7z + 0.9z² | Viero et al. 2022 (quadratic, ~105K at z~8) |

The prior sigma scales with SNR: high-SNR fits are data-driven, low-SNR fits are pulled toward the prior. Use `"flat"` to see what the data say independently.

```python
results = wrapper.run_analysis_only(
    temperature_prior="flat",     # no assumed T-z relation
    use_covariance=True,
)
```

### Covariance-Aware Fitting

Supports three covariance sources, combinable:

- **Instrumental**: inter-band Pearson correlation matrix (e.g., PACS-SPIRE correlations of 0.04–0.37)
- **Bootstrap**: per-population covariance from bootstrap iterations
- **Combined**: sum of instrumental + bootstrap

```python
results = wrapper.run_analysis_only(
    use_covariance=True,
    inflation_factors={24: 10, 100: 5},  # downweight noisy bands
)
```

## Plotting

Three main plotting functions:

### SED Grid

```python
from simstack4.plots import plot_sed_grid

fig = plot_sed_grid(
    wrapper,
    show_prior=True,          # overlay temperature prior SED
    show_model=True,          # show fitted greybody
    show_tier=True,           # show fit quality tier (A/B/C)
    fontsize_legend=7,        # smaller legend text for dense grids
)
```

### T_dust vs Redshift

```python
from simstack4.plots import create_trf_redshift_plot

fig, df = create_trf_redshift_plot(
    wrapper,
    min_tier="A",             # only Tier A fits (default)
    color_by="stellar_mass",
    fit_data=True,            # quadratic fit to data
    show_literature=True,     # Schreiber+18, Viero+22, Drew&Casey+22
)
```

### IRX–β

```python
from simstack4.plots import create_lir_luv_beta_plot

fig = create_lir_luv_beta_plot(
    wrapper,
    color_by="redshift",
    min_tier="A",
)
```

L_UV is pulled from `bin_property_columns` (median per population) or from binning dimensions. Reference curves: Meurer+99 (MW-like) and SMC-like (Prevot+84).

## Bin Optimization

Automatically optimize bin edges to equalize signal power across bins:

```python
from simstack4.bin_optimizer import optimize_binning

opt = optimize_binning(
    wrapper,
    dims=["redshift", "stellar_mass"],
    n_bins={"redshift": 8, "stellar_mass": 5},
    fixed_edges={"redshift": [0.01, 0.5, 1.5, 3.0, 6.0, 10.0]},  # fix one dim
)
```

Uses equal-power CDF inversion from SED fits: `power_density(z) = sps² × dn/dz`, where `sps = peak_snr / √N`. Edges placed at equal cumulative power intervals — narrow bins where signal is concentrated (low z), wide where dilute (high z).

## Catalog Preparation

The `scripts/clean_cosmos_wijesekera.py` script builds stacking-ready catalogs from COSMOS2025:

- NUVrJ star-forming selection (Ilbert+2013)
- Mass-completeness cuts per redshift bin (Wijesekera+2026 Table A.1)
- Starburst removal (SFR/SFR_MS > 3, Elbaz+2018)
- **β_UV computation** from E(B-V) and attenuation law
- **L_UV(1600Å)** corrected from L_NUV using β_UV (critical for IRX-β)

```bash
uv run python scripts/clean_cosmos_wijesekera.py
```

## Scientific Background

### Key Papers

- **Viero et al. 2013** — Original SIMSTACK algorithm
- **Duivenvoorden et al. 2020** — Foreground subtraction and masking improvements
- **Viero et al. 2022** — COSMOS2020 stacking, dust temperature evolution to z~10
- **Schreiber et al. 2018** — Dust SED library, T_dust-z relation
- **Wijesekera, Koprowski et al. 2026** — IRX-β evolution with UV slope, mass, and redshift

### Citation

```bibtex
@software{simstack4,
  author = {Marco Viero and Contributors},
  title = {Simstack4: Simultaneous Infrared Galaxy Stacking},
  url = {https://github.com/marcoviero/simstack4},
  year = {2025}
}
```

## Development

```bash
uv sync --extra dev
uv run pytest
uv run ruff check src/
```

## License

MIT License — see [LICENSE](LICENSE).
