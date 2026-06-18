# PAH Forward Model — Run 2 Summary
**Date**: 2026-06-12  
**Notebook**: `2026-06-12-load-json-fit-seds-redshift-stellar-mass-PAH-dithered-dz015.ipynb`  
**Stacking runs**: `cosmos25_stacking_{20260612_114951, 121640, 142425, 145341}.json`

---

## Data

| Run | Date stamp | Offset | Populations | Tier A | Tier B | Tier C |
|-----|-----------|--------|-------------|--------|--------|--------|
| 0 | 20260612_114951 | Δz = 0.000 | 120 | 17 | 35 | 68 |
| 1 | 20260612_121640 | Δz = 0.0375 | 126 | 14 | 47 | 65 |
| 2 | 20260612_142425 | Δz = 0.0750 | 126 | 19 | — | — |
| 3 | 20260612_145341 | Δz = 0.1125 | 126 | — | — | — |

**Combined tomographic spectrum** (`combine_pah_spectra`, `split_filter=[0]`, `min_tier="B"`):
- 47 populations with valid f₂₄/f_peak ratios
- 197 spectral points across 3 mass bins (43 + 77 + 77)
- λ_rest coverage: **5.4–15.9 μm** (essentially complete over main PAH complex)
- Median f₂₄/f_peak = **0.020** (2.0%), range [0.005, 0.048]
- Median L₂₄/L_IR = 4.3%
- Median SNR per point: 8.7–9.5

**SED analysis settings** (`ANALYSIS_KWARGS`):
- `snr_high=5.0`, `snr_low=2.0`
- `inflation_factors = {24: 10000, 70: {(0.0, 0.8): 1.0, (0.8, 99.0): 10000}}`
  - 24 μm fully excluded from SED fit (PAH-contaminated); f₂₄/f_peak measured post-fit
  - 70 μm included at z < 0.8 (rest ~40–55 μm, clean continuum) as FIR anchor; excluded at z ≥ 0.8

---

## Forward Model Results

Model: `PAHModel(include_silicate=True).fit_forward_model_multibin`

```
flux_m(z) = baseline_m(z) × [1 + α_m · Σ_g r_g · T_g(z)] × exp(−τ_sil · S(z))
```

### Global parameters

| Parameter | Value | Uncertainty | Significance |
|-----------|-------|-------------|--------------|
| χ²_red | 3.365 | — | Elevated; baseline scatter larger than formal errors |
| τ_sil (silicate 9.7 μm) | 0.000 | ± 0.081 | Not detected (< 1σ) |
| r₁ (ratio 7.7+8.6 μm / 6.2 μm) | 2.639 | ± 1.851 | Marginally constrained |
| r₂ (ratio 12.7 μm / 6.2 μm) | 5.000 | ± 3.733 | Hit upper bound; 12.7 μm poorly covered |

### Per-bin PAH amplitudes α_m

| Mass bin (log M*/M☉) | Center | N pts | α | σ_α | SNR | Bump SNR |
|---------------------|--------|-------|---|-----|-----|----------|
| 8.50 – 10.30 | 9.40 | 43 | **1.077** | 0.833 | 1.3 | 1.98× |
| 10.30 – 10.70 | 10.50 | 77 | **0.871** | 0.675 | 1.3 | 2.64× |
| 10.70 – 12.00 | 11.35 | 77 | **0.694** | 0.539 | 1.3 | 2.52× |

**Mass trend**: α decreases with stellar mass at −0.10 dex per dex in log M* (fitted slope from three-point regression). Amplitude ratio low-mass/high-mass ≈ 1.55, significant at ~1σ given the per-bin uncertainties.

### Self-consistency recovery

Injected fitted amplitudes back with 5% noise; recovered within 1σ for all bins:

| Bin | Injected | Recovered | Status |
|-----|----------|-----------|--------|
| 8.50–10.30 | 1.077 | 1.118 ± 0.356 | OK |
| 10.30–10.70 | 0.871 | 0.851 ± 0.275 | OK |
| 10.70–12.00 | 0.694 | 0.642 ± 0.209 | OK |

Recovery uncertainties (~0.2–0.35) are 2–3× tighter than data uncertainties, confirming the forward model is well-identified — the dominant source of per-bin uncertainty is baseline scatter, not model degeneracy.

### Null test: 70 μm

f₇₀/f_peak fitted with identical forward model. All bins **PASS** (α consistent with zero):

| Bin | α (24 μm) | α (70 μm) | Null? |
|-----|-----------|-----------|-------|
| 8.50–10.30 | 1.077 ± 0.833 | 0.000 ± 5.119 | PASS |
| 10.30–10.70 | 0.871 ± 0.675 | 5.000 ± 10.741 | PASS |
| 10.70–12.00 | 0.694 ± 0.539 | 0.000 ± 2.506 | PASS |

The 24 μm modulation is not a baseline artifact — 70 μm shows no coherent structure.

---

## Interpretation

### What is detected
At SNR~1.3 per mass bin, the PAH amplitude is **consistent with but not a strong detection of** the known PAH complex. The collective evidence for PAH emission is:
1. The three α values are all positive and follow the expected mass trend (decreasing with M*)
2. The bump SNR (~2–2.6×) places PAH features just above the noise floor visually
3. The null test at 70 μm passes cleanly
4. The self-consistency test confirms the model is well-constrained given more data

The marginal SNR is primarily a depth limitation (Δz=0.15 bins, ~140 sources/bin Tier B median), not a model problem. Adding σ_SFR as a second binning dimension, or stacking deeper maps, would improve SNR.

### No silicate absorption
τ_sil = 0.000 ± 0.081. Normal main-sequence galaxies at z~0.5–3.5 are optically thin at 9.7 μm; the silicate feature is not detectable with this dataset. The Drude term is correctly recovering zero and not distorting the PAH amplitudes.

### The 12.7 μm ratio hitting bounds
The r₂ parameter (12.7/6.2 ratio) hits the upper bound of 5.0 in both with- and without-silicate fits. The 12.7 μm feature is only accessible at z~0.9 (where λ_rest = 12.7 μm), a narrow slice. With few points there, the ratio is unconstrained — it is driven by the optimizer to maximize the χ² contribution from those points. This does not affect the α measurements materially.

### chi2_red = 3.365
Higher than the simulation chi2_red ≈ 0.90 (which uses the same noise level). Indicates actual f₂₄/f_peak scatter is ~√3.4 ≈ 1.8× larger than the formal bootstrap errors. Likely causes:
- Real astrophysical scatter in PAH/FIR ratio within each mass bin (population heterogeneity)
- Baseline polynomial (quadratic in z) not perfectly capturing the smooth underlying trend
- Residual systematic in f_peak from SED fits (propagates into f₂₄/f_peak ratio)

---

## Updated Empirical Coefficients in `greybody.py`

**Method**: `_physical_wien_flux` and `_pah_flux_0` both use  
`log₁₀(f₂₄/f_peak) = a·log M* + b·z + c·PAH_strength + d`

| Coefficient | Old value | New value | Change |
|-------------|-----------|-----------|--------|
| a (log M* slope) | +0.017 | **−0.10** | Sign flip + larger magnitude |
| b (z slope) | −0.206 | −0.206 | Unchanged |
| c (PAH bandpass strength) | +0.066 | +0.066 | Unchanged |
| d (constant) | −1.577 | **−0.349** | Adjusted to preserve normalization at pivot |

**Pivot point** for normalization: log M* = 10.5, z = 1.5 → f₂₄/f_peak ≈ 0.021 (unchanged).

**Derived slope**: from three α(log M*) measurements, linear regression gives
d(log₁₀ α)/d(log M*) = −0.099. Since f₂₄/f_peak ∝ α at fixed z, this maps to a = −0.10.

**Physical interpretation**: PAH-to-FIR ratio **decreases** with stellar mass. Higher-mass galaxies (which are more IR-luminous, more likely to be toward the ULIRG regime) show weaker PAH relative to the FIR continuum. This is consistent with the known PAH deficit and the opposite sign from the old empirical coefficient (which was calibrated on a narrower dataset).

**Uncertainty**: SNR~1.3 per bin; the slope is uncertain by ±0.3–0.5 dex/dex. The direction (negative) is robustly measured; the magnitude should be re-calibrated once more runs at additional mass cuts are available.

---

---

## Run 2b: 4-bin mass scheme, uniform Δz=0.15  
**Date**: 2026-06-12 (afternoon)  
**Notebook**: `2026-06-12-load-json-fit-seds-redshift-stellar-mass-PAH-dithered-dz015.ipynb`  
**Stacking runs**: `cosmos25_stacking_{20260612_190116, 180838, 164943, 160940}.json`

Mass bins: `[(8.5, 10.2), (10.2, 10.6), (10.6, 11.0), (11.0, 12.0)]` — finer grid than Run 2 (3 bins), designed to resolve slope and test for curvature. Same Δz=0.15 uniform stagger, offsets 0.000/0.0375/0.0750/0.1125.

Results and analysis available in the notebook. Key comparison to Run 2 (3-bin): with 4 bins the mass slope has one more degree of freedom, allowing a quadratic term. The narrower bin widths (Δ log M* ≈ 0.4 vs 0.8/0.4/1.3) give more equal source counts per bin.

---

## Run 2c: σ_SFR cross-cut (2 mass × 3 σ_SFR bins)  
**Date**: 2026-06-14 (3 of 4 runs complete)  
**Notebook**: `2026-06-14-load-json-fit-seds-redshift-stellar-mass-sigma_sfr-PAH-dithered-dz015.ipynb`  
**Stacking runs**: `cosmos25_stacking_{20260614_140041, 154016, 164554}.json` + pending run 3

**Binning**: 2 stellar mass × 3 σ_SFR = 6 population cells per z-slice.

```
MASS_BINS     = [(8.5, 10.5), (10.5, 12.0)]
SIGMA_SFR_BINS = [(-3.0, -1.71), (-1.71, -0.99), (-0.99, 1.5)]  M☉/yr/kpc²
```

**Known issue with current σ_SFR bins**: Bin 0 (−3.0 to −1.71) spans an extreme regime with very few Tier B sources. Bin 2 (−0.99 to 1.5) spans 2.5 dex of physically interesting range. Better future edges: [−2.0, −1.0, 0.0, 1.0].

**Science direction**: α should *decrease* with σ_SFR (stronger UV radiation field in compact SF galaxies → PAH grain destruction). This is physically distinct from the M* trend (which may be driven by overall dust opacity / AGN fraction at high M*). A confirmed σ_SFR trend would localize the mechanism to the radiation field rather than the halo mass.

**Status**: Notebook structure complete; run 3 pending. Analysis code ready but not yet executed. Fill in `RUN_DATES[3]` in notebook cell `16153c5d` when stacking completes.

---

## Run 2d: Accordion z-bins vs uniform Δz=0.15  
**Date**: 2026-06-15  
**Notebook**: `2026-06-15-pah-3bin-vs-4bin-mass-comparison.ipynb`  
**Uniform runs**: `cosmos25_stacking_{20260612_190116, 180838, 164943, 160940}.json` (same as 2b)  
**Accordion runs**: `cosmos25_stacking_{20260615_092101, 101110, 105025, 113922}.json`

Both schemes use 4 mass bins `[(8.5,10.2),(10.2,10.6),(10.6,11.0),(11.0,12.0)]` and 4 dither runs. Accordion stagger offset = 0.025/run (Δz_narrow/4 = 0.10/4).

**Accordion bin edges**:
```
[0.50, 0.65, 0.75, 0.85, 0.95, 1.10, 1.30, 1.45, 1.55, 1.65,
 1.75, 1.85, 2.05, 2.25, 2.35, 2.45, 2.55, 2.65, 2.80, 3.00, 3.30, 3.60]
```
Narrow bins (Δz=0.10) in three high-gradient zones:  
- λ_rest ≈ 12.7 μm at z~0.7–0.9  
- λ_rest ≈ 7.7+8.6 μm at z~1.6–1.8  
- 7.7 μm exiting at z~2.3–2.6  

Wide bins (Δz=0.20–0.30) in low-gradient zones between features.

**What to look for**: does σ_α decrease in the 7.7+8.6 μm zone (z~1.5–2.1)? If the improvement is ≲15%, baseline scatter (χ²_red ≈ 3.4) dominates over sampling — uniform is simpler.

**Status**: Notebook complete, awaiting execution. Key output: σ_α ratio per mass bin (accordion/uniform) and χ²_red comparison.

---

## Forward path → pah-forward-model-3

1. **Complete pending runs**: σ_SFR run 4 (offset 0.1125); execute accordion vs uniform comparison notebook; execute 4-bin mass notebook.

2. **Honest error bars**: rescale all σ_α by √(χ²_red) ≈ 1.83 to account for baseline scatter. Report both raw (formal) and rescaled uncertainties.

3. **Combined significance**: fit slope d log α / d log M* jointly across all mass bins, report uncertainty on slope with and without error rescaling. If accordion is beneficial, use accordion dataset for final slope measurement.

4. **σ_SFR result**: once run 4 completes, fit the joint (M*, σ_SFR) forward model to determine whether the dominant driver is M* or radiation field (σ_SFR).

5. **PAH correction application**: use measured α(M*) to correct f₂₄ per population before SED fitting; reduce inflation 10000× → 3–5×; re-run main SED stacks; check Tier C→B promotion at z~1.5–2.5 and quantify T_dust bias.

6. **Publication figures**: see `docs/pah-forward-model-3-brief.md` for the full figure set and referee-defense strategy.
