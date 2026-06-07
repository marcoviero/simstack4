# COSMOS Catalog Preparation

This document describes how to build a stacking-ready parquet catalog from the
COSMOSWeb master FITS files using `prepare-cosmos-catalog`.

---

## Overview

Simstack4 requires a pre-processed source catalog (parquet format) that lists
sky positions, redshifts, and population labels for every source to be stacked.
The `prepare-cosmos-catalog` script performs all necessary filtering, flagging,
and derived-quantity computation from the raw COSMOSWeb FITS files.

**Key design principle**: simstack stacks ALL populations simultaneously as
separate label dimensions.  Sources are only removed from the output if their
population class appears in `exclude_classes`.  Aggressive completeness
filtering should be treated as a paper-specific option, not the default — if a
population has correlated IR emission and is removed, the stacked fluxes for
the remaining populations will be biased.

---

## Input Files (COSMOSWeb v1)

Located in `$CATSPATH/cosmos/`:

| File | Contents |
|------|---------|
| `COSMOSWeb_mastercatalog_v1.fits` | Photometry: RA, Dec, flags, fluxes (20+ filters), Sersic morphology |
| `COSMOSWeb_mastercatalog_v1_lephare.fits` | LePhare SED fits: z_phot, M*, SFR, rest-frame magnitudes, E(B-V) |
| `COSMOSWeb_mastercatalog_v1_cigale.fits` | CIGALE SED fits: metallicity, independent stellar mass (optional) |

Catalogs are merged by row index — the standard COSMOS convention.

---

## Usage

```bash
# Default settings (z=0.01-9, β from both methods, all populations)
prepare-cosmos-catalog

# Wijesekera+2026 preset
prepare-cosmos-catalog --paper w26

# Parente+2026 preset (includes Sersic + sigma_SFR)
prepare-cosmos-catalog --paper p26

# Explicit config file
prepare-cosmos-catalog --config config/catalog/custom.toml

# Override specific settings
prepare-cosmos-catalog --paper w26 --keep-all        # don't exclude QT
prepare-cosmos-catalog --paper p26 --no-morphology   # skip Sersic
prepare-cosmos-catalog --exclude 2 3                 # drop QT and SB

# Override catalog path
prepare-cosmos-catalog --catspath /path/to/catalogs/cosmos
```

### Output files

| Config | Output |
|--------|--------|
| defaults | `$CATSPATH/cosmos/COSMOSWeb_cosmos25_catalog.parquet` |
| `--paper w26` | `COSMOSWeb_w26_catalog.parquet` |
| `--paper p26` | `COSMOSWeb_p26_catalog.parquet` |
| `--output foo.parquet` | `foo.parquet` |

---

## Config System

Paper presets live in `config/catalog/` and override only the keys that differ
from `defaults.toml`.  The merge order is:

```
defaults.toml → paper preset (--paper) → explicit config (--config) → CLI flags
```

### defaults.toml structure

```toml
[paper]       # id, description
[inputs]      # phot_file, sed_file, cigale_file
[columns]     # ra, dec, redshift, stellar_mass, sfr, nuv, r_band, j_band, ...
[selection]   # redshift_min/max, mass_min, chi2_max
[populations] # flag_quiescent/starburst/mass_completeness, exclude_classes
[beta_luv]    # method ("template"|"photometric"|"both"), flux_prefix, ...
[morphology]  # sersic, sigma_sfr
[output]      # filename
```

---

## Population Classification

Each source is assigned a `population_class` integer based on three binary flags:

| Class | Label | Criteria |
|-------|-------|---------|
| 0 | `complete_sfg` | star-forming, mass-complete, not starburst |
| 1 | `incomplete_sfg` | star-forming, mass-incomplete, not starburst |
| 2 | `qt` | quiescent (any mass completeness) |
| 3 | `sb` | starburst: SFR/SFR_MS > 3 (Elbaz+2018), SF, mass-complete |

Priority order: SB (3) > QT (2) > mass-completeness (0 vs 1).

### NUVrJ classification (Ilbert+2013)

A source is **quiescent** if both:
- (NUV − r) > 3 × (r − J) + 1
- (NUV − r) > 3.1

Applied only at z < 4 (rest-frame J unreliable at higher z).
Sources with missing colors default to star-forming.

### Mass completeness (Wijesekera+2026 Table A.1)

90% completeness limits per redshift bin:

| z range | log(M*/Msun) limit |
|---------|-------------------|
| 0.01–0.5 | 8.50 |
| 0.5–1.5  | 9.25 |
| 1.5–2.5  | 9.50 |
| 2.5–3.5  | 9.75 |
| 3.5–5.0  | 10.00–10.25 |
| 5.0–8.0+ | 10.50–11.00+ (extrapolated) |

### Starburst identification (Schreiber+2015)

SFR_MS from Schreiber+2015 Eq 9:
`log(SFR_MS) = m − 0.5 + 1.5r − 0.3[max(0, m − 0.36 − 2.5r)]²`
where `m = log(M*/10⁹ M☉)`, `r = log(1+z)`.

Sources with SFR / SFR_MS > 3 are flagged as starbursts.

---

## UV Slope and Luminosity

Two independent methods are available (configured via `beta_luv.method`):

### Template method (`"template"`)

β = β_intrinsic + k_λ × E(B-V)

- β_intrinsic = −2.3 (young SF population)
- k_λ: 4.43 (Calzetti), 4.20 (Arnouts), 3.80 (Salim)
- Output columns: `beta_uv`, `log_l_uv`

L_UV(1600 Å) is corrected from L_NUV(2300 Å):
`log L(1600) = log L(2300) + (β + 1) × log₁₀(1600/2300)`

**Warning**: template β and L_UV share the E(B-V) parameter with the SED fit.
For IRX-β analysis, using template quantities collapses the IRX-β plane to a
near-flat line because β and L_IR are correlated through the same dust model.
Use `"photometric"` for IRX-β science.

### Photometric method (`"photometric"`)

Power-law fit to observed fluxes sampling rest-frame 1300–2600 Å:
`log(f_ν) = (β + 2) × log(λ_rest) + const`

- Uses: CFHT-u, HSC grizy, HST/F814W, JWST/F115W/F150W, Subaru IA/IB
- Requires ≥ 2 bands with SNR ≥ 2 in UV window
- Requires z ≥ 0.75 (insufficient UV coverage below)
- Output columns: `beta_uv_phot`, `log_l_uv_phot`

This method is independent of SED fitting and is the correct choice for
IRX-β analysis (Meurer+1999, Reddy+2018).

---

## Output Columns

| Column | Description |
|--------|------------|
| `ra`, `dec` | Astrometry (degrees) |
| `redshift` | Photometric redshift (from LePhare `zfinal`) |
| `log_stellar_mass` | log₁₀(M*/M☉) from LePhare |
| `log_sfr` | log₁₀(SFR / [M☉/yr]) from LePhare |
| `log_ssfr` | log₁₀(sSFR) = log_sfr − log_stellar_mass |
| `log_delta_ms` | log(SFR/SFR_MS); offset from Schreiber+2015 |
| `beta_uv` | UV slope from E(B-V) + dust law (template) |
| `log_l_uv` | log₁₀(νL_ν(1600Å)/L☉), corrected from L_NUV (template) |
| `beta_uv_phot` | UV slope from broadband photometry |
| `log_l_uv_phot` | log₁₀(νL_ν(1600Å)/L☉), from broadband photometry |
| `ebv_minchi2` | E(B-V) from LePhare best-fit |
| `star_forming` | 1 = SF, 0 = quiescent (NUVrJ) |
| `mass_complete` | 1 = above 90% mass-completeness limit |
| `starburst` | 1 = SFR/SFR_MS > 3 |
| `population_class` | Integer: 0=complete_sfg, 1=incomplete_sfg, 2=qt, 3=sb |
| `sersic_reliable` | 1 = reliable Sersic fit (if `morphology.sersic=true`) |
| `log_sigma_sfr` | log₁₀(Σ_SFR / [M☉/yr/kpc²]) (if `morphology.sigma_sfr=true`) |
| `metallicity` | Gas-phase metallicity from CIGALE (mass fraction; solar ≈ 0.02) |

All original catalog columns are also retained.

---

## Paper Presets

### `--paper w26` — Wijesekera+2026

"Evolution of dust attenuation in star-forming galaxies with UV slope,
stellar mass, and redshift out to z~5"

- z range: 0.5 – 5.0
- β method: `photometric` (critical for unbiased IRX-β)
- exclude_classes: [2] (QT excluded to reduce label count)

### `--paper p26` — Parente+2026

"Dust temperature evolution and two-component SED modeling in COSMOS"

- z range: 0.5 – 5.0
- Includes CIGALE (metallicity)
- Morphology: Sersic + σ_SFR enabled
- All population classes kept

### `--paper a26` — Agrawal+2026

Placeholder — configuration to be filled in when paper parameters are finalized.

---

## Pointing Simstack at the New Catalog

Update `[catalog]` in your stacking TOML:

```toml
[catalog]
path = "$CATSPATH/cosmos"
file = "COSMOSWeb_w26_catalog.parquet"

[catalog.classification.binning.redshift]
id = "redshift"
...
```

---

## Legacy Scripts

The following scripts are superseded by `prepare-cosmos-catalog`:

| Old script | Replacement |
|------------|------------|
| `clean_cosmos_wijesekera_phot.py` | `prepare-cosmos-catalog --paper w26` |
| `clean_cosmos_sersic.py` | `prepare-cosmos-catalog --paper p26` |

The old scripts remain in `src/simstack4/scripts/` as reference but should
be considered deprecated.
