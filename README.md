# Simstack4: Next-Generation Infrared Galaxy Stacking

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![Package Manager](https://img.shields.io/badge/package%20manager-uv-green.svg)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Simstack4** is a modernized simultaneous stacking pipeline for measuring average infrared flux densities from populations of galaxies. It accounts for source clustering bias through simultaneous fitting of multiple population layers, replacing the nested loop structure of simstack3 with a more efficient matrix-based approach.

## 🌟 What's New in Simstack4

- **Modernized Infrastructure**: Python 3.13+ with [uv](https://github.com/astral-sh/uv) package management
- **Efficient Algorithm**: Matrix-based simultaneous fitting replaces nested population loops
- **Flexible Binning**: Dynamic population binning without hardcoded limits  
- **TOML Configuration**: User-friendly configuration files replace INI format
- **Type Safety**: Full type hints and Pydantic data validation
- **Comprehensive Testing**: Automated testing and error handling
- **Bootstrap Support**: Built-in bootstrap error estimation

## 🚀 Quick Start

### Installation

We use [**uv**](https://github.com/astral-sh/uv) as our package manager because it's:
- **Fast**: 10-100x faster than pip for dependency resolution
- **Reliable**: Deterministic dependency resolution with lock files  
- **Modern**: Built in Rust with excellent Python integration
- **Simple**: Drop-in replacement for pip/conda workflows

#### 1. Install uv

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.sh | iex"

# Or with pip
pip install uv
```

#### 2. Clone and Install Simstack4

```bash
git clone https://github.com/marcoviero/simstack4.git
cd simstack4

# Install with development dependencies
uv sync --extra dev --extra notebooks

# Or install just the basics
uv sync
```

#### 3. Set Environment Variables

Add these to your shell configuration file (`~/.zshrc`, `~/.bashrc`, etc.):

```bash
# Path to astronomical maps (FITS files)
export MAPSPATH="/path/to/your/maps"

# Path to source catalogs (Parquet/CSV files)  
export CATSPATH="/path/to/your/catalogs"

# Path for output pickles and results
export PICKLESPATH="/path/to/your/pickles"
```

Then reload your shell:
```bash
source ~/.zshrc  # or ~/.bashrc
```

#### 4. Verify Installation

```bash
# Check system status
uv run simstack4 --check-system

# Run basic example
uv run python examples/basic_usage.py
```

## 📖 How Simstack Works

### The Physics Problem

When measuring infrared emission from faint galaxy populations, individual sources are often below the detection threshold. Traditional stacking averages flux at source positions, but **ignores clustering bias** - nearby galaxies contribute correlated signal that biases measurements.

### The Simstack Solution

Simstack solves this through **simultaneous fitting**:

1. **Population Binning**: Split galaxies into populations by stellar mass, redshift, and type
2. **Layer Creation**: Create 2D "templates" for each population with unit sources at galaxy positions  
3. **PSF Convolution**: Convolve each template with the instrument point spread function
4. **Simultaneous Fitting**: Fit all templates simultaneously to observed maps using least-squares

This accounts for clustering by fitting all correlated populations together, yielding unbiased average flux densities.

### Mathematical Framework

The observed map is modeled as:
```
observed_map = Σ_populations (N_pop × F_avg_pop × PSF_convolved_template) + noise
```

Where:
- `N_pop` = number of sources in each population
- `F_avg_pop` = average flux per source (what we solve for)
- `PSF_convolved_template` = template with sources convolved by PSF

## 📋 What You Need

### Required Data

1. **Astronomical Maps** (FITS format)
   - Signal maps at multiple wavelengths (e.g., 100, 160, 250, 350, 500 μm)
   - Noise/uncertainty maps (optional but recommended)
   - Proper WCS coordinates and beam information

2. **Source Catalog** (Parquet/CSV format)
   - Right Ascension (`ra`) and Declination (`dec`) coordinates
   - Redshift measurements
   - Stellar mass estimates  
   - Population classification (star-forming vs quiescent)

3. **Configuration File** (TOML format)
   - Map paths and properties
   - Catalog column mappings
   - Population binning scheme
   - Algorithm parameters

### Example Data Structure

```
data/
├── maps/
│   ├── cosmos/
│   │   ├── COSMOS_PACS100_signal.fits
│   │   ├── COSMOS_PACS100_noise.fits
│   │   ├── COSMOS_SPIRE250_signal.fits
│   │   └── ...
├── catalogs/
│   ├── cosmos/
│   │   └── COSMOSWeb_clean.parquet
└── pickles/
    └── simstack/
        └── results/
```

## ⚙️ Configuration Files

Configuration files use TOML format and are stored in the `config/` directory. Here's how to create one:

### Basic Structure

```toml
# config/your_project.toml

# Algorithm settings
[binning]
stack_all_z_at_once = true    # Stack all redshifts simultaneously (recommended)
add_foreground = true         # Include foreground subtraction layer
crop_circles = true           # Fit only pixels around sources

# Output settings  
[output]
folder = "$PICKLESPATH/simstack/results"
shortname = "your_project_name"

# Catalog configuration
[catalog]
path = "$CATSPATH/your_field"
file = "your_catalog.parquet"

[catalog.astrometry]
ra = "ra"              # Column name for right ascension
dec = "dec"            # Column name for declination

# Population binning scheme
[catalog.classification]
split_type = "uvj"     # Options: "labels", "uvj", "nuvrj"

[catalog.classification.redshift]
id = "z_best"          # Redshift column name
bins = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # Redshift bin edges

[catalog.classification.stellar_mass]  
id = "mass_best"       # Stellar mass column name (in log solar masses)
bins = [9.0, 9.5, 10.0, 10.5, 11.0, 12.0]   # Mass bin edges

# Star-forming vs quiescent classification
[catalog.classification.split_params]
id = "sfg_flag"        # Population type column
[catalog.classification.split_params.bins]
"U-V" = "U_V_color"    # U-V color column (for UVJ classification)
"V-J" = "V_J_color"    # V-J color column

# Map configurations
[maps.pacs_100]
wavelength = 100.0
color_correction = 1.0
path_map = "$MAPSPATH/your_field/PACS_100_signal.fits"
path_noise = "$MAPSPATH/your_field/PACS_100_noise.fits"

[maps.pacs_100.beam]
fwhm = 7.0            # Beam FWHM in arcsec
area = 1.0            # Beam area (1.0 = calculate from FWHM)

[maps.spire_250]
wavelength = 250.0
color_correction = 1.02
path_map = "$MAPSPATH/your_field/SPIRE_250_signal.fits" 
path_noise = "$MAPSPATH/your_field/SPIRE_250_noise.fits"

[maps.spire_250.beam]
fwhm = 18.1
area = 1.0
```

### Configuration Options

#### Population Classification (`split_type`)

- **`"labels"`**: Use pre-existing labels in catalog
- **`"uvj"`**: UVJ color-color classification (star-forming vs quiescent)  
- **`"nuvrj"`**: NUV-r-J color classification

#### Algorithm Settings

- **`stack_all_z_at_once`**: Fit all redshift bins simultaneously (recommended, but memory intensive)
- **`add_foreground`**: Include constant foreground layer to account for background offsets
- **`crop_circles`**: Only fit pixels within circles around source positions (reduces computation)

#### Error Estimation

```toml
[error_estimator.bootstrap]
enabled = true        # Enable bootstrap error estimation  
iterations = 100      # Number of bootstrap iterations
initial_seed = 42     # Random seed for reproducibility
```

## 🎯 Running Simstack4

### Basic Usage

```python
import simstack4

# Load configuration
config = simstack4.load_config("config/cosmos.toml")

# Run stacking pipeline
wrapper = simstack4.SimstackWrapper(
    config, 
    read_maps=True, 
    read_catalog=True, 
    stack_automatically=True
)

# Access results
results = wrapper.processed_results
summary_df = results.get_population_summary()
```

### Command Line Interface

```bash
# Run complete pipeline
uv run simstack4 run config/cosmos.toml

# Check system and validate config
uv run simstack4 validate config/cosmos.toml

# Run with bootstrap error estimation
uv run simstack4 run config/cosmos.toml --bootstrap --iterations 100
```

### Example Script

```python
#!/usr/bin/env python3
"""Example COSMOS stacking pipeline"""

from simstack4.config import load_config
from simstack4.wrapper import SimstackWrapper
from pathlib import Path

def run_cosmos_stacking():
    # Load configuration
    config = load_config("config/cosmos.toml")
    
    # Enable bootstrap errors
    config.error_estimator.bootstrap.enabled = True
    config.error_estimator.bootstrap.iterations = 50
    
    # Run stacking
    wrapper = SimstackWrapper(
        config,
        read_maps=True,
        read_catalog=True, 
        stack_automatically=True
    )
    
    # Save results
    summary = wrapper.processed_results.get_population_summary()
    summary.to_csv("cosmos_stacking_results.csv")
    
    # Print summary
    print(f"Completed stacking of {len(summary)} populations")
    detected = summary[summary['total_ir_luminosity_lsun'] > 0]
    print(f"Detected IR emission in {len(detected)} populations")

if __name__ == "__main__":
    run_cosmos_stacking()
```

## 📊 Understanding Results

Simstack4 outputs several data products:

### Population Summary Table
CSV table with one row per population containing:
- Population identifiers (redshift, mass, type)
- Measured flux densities and uncertainties for each map
- Derived quantities (IR luminosity, SFR, etc.)
- Source counts and completeness information

### Detailed Results Object
Pickle file containing:
- Raw stacking results for all populations and maps
- Bootstrap error estimates (if enabled)
- Configuration parameters used
- Diagnostic information

### Key Output Columns

- `total_ir_luminosity_lsun`: Total infrared luminosity (8-1000 μm) in solar units
- `sfr_msun_yr`: Star formation rate in solar masses per year
- `n_sources`: Number of sources in population
- `median_redshift`: Median redshift of population
- `median_stellar_mass`: Median stellar mass of population

## 🔧 Troubleshooting

### Common Issues

**Import Errors**
```bash
# Ensure you're in the right environment
uv run python -c "import simstack4; print('Success!')"
```

**Environment Variable Issues**
```bash
# Check variables are set
echo $MAPSPATH $CATSPATH $PICKLESPATH

# Validate paths exist  
uv run simstack4 --check-system
```

**Memory Issues**
```toml
# In your config file, try:
[binning] 
stack_all_z_at_once = false  # Process redshift bins separately
crop_circles = true           # Reduce number of fitted pixels
```

**Configuration Errors**
```bash
# Validate your config file
uv run simstack4 validate config/your_config.toml
```

### Getting Help

- **Documentation**: Full API docs at [https://simstack4.readthedocs.io](https://simstack4.readthedocs.io)
- **Issues**: Report bugs at [https://github.com/marcoviero/simstack4/issues](https://github.com/marcoviero/simstack4/issues)
- **Discussions**: Ask questions at [https://github.com/marcoviero/simstack4/discussions](https://github.com/marcoviero/simstack4/discussions)

## 📚 Scientific Background

### Key Papers

- **Viero et al. 2013**: Original SIMSTACK algorithm and methodology
- **Duivenvoorden et al. 2020**: Improvements with foreground subtraction and masking
- **Your Paper**: If you publish results using Simstack4, we'd love to hear about it!

### Citation

If you use Simstack4 in your research, please cite:

```bibtex
@software{simstack4,
  author = {Marco Viero and Contributors},
  title = {Simstack4: Next-Generation Infrared Galaxy Stacking},
  url = {https://github.com/marcoviero/simstack4},
  version = {1.0.0},
  year = {2024}
}
```

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/marcoviero/simstack4.git
cd simstack4
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install

# Run tests
uv run pytest

# Format code  
uv run black src/
uv run ruff check src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original SIMSTACK algorithm by Marco Viero
- Improvements by the HELP survey team
- Beta testing by the astronomical community
- Built with modern Python tools: [uv](https://github.com/astral-sh/uv), [Pydantic](https://pydantic.dev/), [Astropy](https://www.astropy.org/)

---

**Happy Stacking!** 🌌📡✨