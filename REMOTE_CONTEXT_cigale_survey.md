# Remote context: CIGALE SED survey

This file is a self-contained prompt for the machine that holds `CIGAL_SEDs_v1.tar.gz`.
It does not require reading `CLAUDE.md` — all necessary context is here.

## What this machine needs to produce

We are building a three-component dust SED model (cold modified blackbody + hot dust
continuum + PAH features) for the simstack4 stacking pipeline. The CIGALE SED catalog
is the physics ground-truth for the hot dust and PAH components.

Produce four small summary files (< 10 MB total) that can be transferred back to the
main development machine:

```
cigale_sed_grid.npz          # median L_ν/L_IR per (z, M*) bin over rest-frame λ grid
cigale_pah_amplitudes.csv    # L_PAH/L_IR vs log_L_IR, z, log_M* per bin
cigale_hot_dust_shapes.npz   # continuum residual 15–60 µm after PAH subtraction, per bin
cigale_t_hd_grid.csv         # effective T_hd(z, M*, log_SFR) from Draine+Li Umin mapping
```

## Step 1: Explore the archive

First, peek inside the tarball without extracting everything:

```bash
tar -tzf CIGAL_SEDs_v1.tar.gz | head -50
tar -tzf CIGAL_SEDs_v1.tar.gz | wc -l
```

Determine:
- File format of individual SEDs (FITS, HDF5, ASCII, npz?)
- Directory structure (one file per source? per bin? one large file?)
- Column names / array keys for: wavelength, flux, redshift, stellar mass, SFR,
  L_IR, and any CIGALE-specific outputs (qPAH, Umin, Umax, f_AGN)

Extract a single example file and inspect it before extracting the full archive.

## Step 2: Map the SED structure

For each source SED, identify:
- **Wavelength array**: rest-frame or observed? Units (µm, Å, Hz)?
- **Flux array**: L_ν/L_IR normalised, or physical units? Which component columns
  exist (total, stellar, dust, PAH, AGN)?
- **Scalar properties**: z, log_M*, log_SFR, log_L_IR, qPAH, Umin, Umax

Goal: confirm the SEDs extend from UV (~0.1 µm) to submm (~1000 µm) in rest frame,
covering the PAH region (6–15 µm) and FIR dust peak (80–500 µm).

## Step 3: Bin definitions

Use these stacking bin edges — they match the COSMOS stacking grid on the main machine:

```python
z_bins    = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]       # 6 z bins
mass_bins = [8.5, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0]   # 6 M* bins
```

Bin centre redshifts: [0.75, 1.25, 1.75, 2.25, 2.75, 3.5]

For each (z, M*) bin, select all sources within the bin boundaries and compute the
**median SED** — i.e., median L_ν/L_IR at each wavelength point on a common
rest-frame grid.

Common rest-frame λ grid (log-spaced, 200 points):
```python
import numpy as np
lam_rest_um = np.logspace(np.log10(0.5), np.log10(1200.0), 200)
```

## Step 4: Extract PAH amplitudes

For each source (or bin median), measure:

```
L_PAH / L_IR  where L_PAH = integral of SED from 5–15 µm above the underlying
              continuum (fit a modified blackbody to 20–60 µm and extrapolate under
              the PAH bump)
```

If CIGALE provides a dedicated PAH luminosity column, use that directly.

Output `cigale_pah_amplitudes.csv` with columns:
```
z_bin, mass_bin, z_med, log_mass_med, log_l_ir_med, log_sfr_med,
log_l_pah_over_l_ir, log_l_pah_over_l_ir_std, n_sources
```

## Step 5: Extract hot dust continuum shapes

For each bin, subtract the PAH template from the median SED in the 5–20 µm range,
leaving the hot-dust continuum. Then fit a modified blackbody `ν^β · B_ν(T)` with
β=1.5 to the residual in the 15–60 µm window.

Output `cigale_hot_dust_shapes.npz`:
```python
np.savez('cigale_hot_dust_shapes.npz',
    lam_um       = lam_rest_um,           # shape (200,)
    sed_grid     = sed_grid,              # shape (n_z, n_mass, 200)  median L_ν/L_IR
    residual_grid= residual_grid,         # shape (n_z, n_mass, 200)  hot-dust residual
    z_bins       = z_bin_centres,
    mass_bins    = mass_bin_centres,
)
```

Output `cigale_sed_grid.npz` (same structure, full SED not just residual).

## Step 6: Extract T_hd grid

If Draine+Li 2007 dust module was used in the CIGALE run, each source has a best-fit
`Umin` value. Convert to effective dust temperature via:

```python
# T_hd ∝ U^(1/(4+β)) with β=1.5, anchored to T=18.3 K at U=1 (Draine+07)
T_hd = 18.3 * Umin**(1.0 / (4.0 + 1.5))
```

Bin by (z, M*, log_SFR) and report median + scatter.

Output `cigale_t_hd_grid.csv` with columns:
```
z_bin, mass_bin, sfr_bin, z_med, log_mass_med, log_sfr_med,
T_hd_med, T_hd_std, Umin_med, n_sources
```

If the CIGALE run used Dale+14 instead of Draine+Li, report the best-fit α parameter
instead of Umin, and leave T_hd_med/Umin_med as NaN.

## Step 7: Transfer outputs

Once the four files are produced, transfer them to the main machine:

```bash
# Example — adjust paths as needed
rsync -av cigale_sed_grid.npz cigale_pah_amplitudes.csv \
          cigale_hot_dust_shapes.npz cigale_t_hd_grid.csv \
          mviero@<main-machine-ip>:/Users/mviero/Repositories/simstack4-2026/templates/cigale/
```

Or copy to a shared location and pull from the main machine.

## Notes

- The simstack4-2026 repo is at the same relative path on both machines.
- The stacking results JSON for cross-referencing:
  `/Users/mviero/data/Astronomy/pickles/simstack/stacked_flux_densities/cosmos25_stacking_20260317_201727.json`
- If the CIGALE run used AGN templates, add an `f_agn` column to
  `cigale_pah_amplitudes.csv` so we can flag AGN-contaminated bins.
- Keep intermediate per-source arrays in memory only — do not write per-source files,
  only the binned summary outputs listed above.
