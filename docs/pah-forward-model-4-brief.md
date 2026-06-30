# PAH Forward Model — Branch 4 Brief

**Goal**: Improve the fidelity of the baseline model and validate spectral independence
of pseudo-spectrum points, so that the forward-model fit has a defensible χ² and
physically motivated continuum.

Two objectives, nothing else in this branch.

---

## Objective 1 — Shared power-law baseline from off-feature windows

### Problem

The current baseline is a per-bin power law with independent slope n_m and normalisation
log10(A_m) — two free parameters per mass bin (2M total). With 3 bins this gives 6
baseline parameters on top of 3 α values and 2 ratios = 11 free parameters. The fitter
can trade baseline curvature against α, producing the inverted mass ordering observed in
the K-fold run and masking a physically real trend.

### Physics

Between PAH features, the MIPS 24 μm bandpass samples warm/VSG stochastically-heated
dust. The continuum is smooth and follows a power law in λ_rest = 24/(1+z):

```
f₂₄/f_peak(z) ≈ A_m × (24/(1+z))^γ   [off-feature windows]
```

The slope γ reflects the mid-IR SED shape of the warm component, which should be
**universal** across mass bins (it depends on dust grain size distribution, not M*).
Only the normalisation A_m carries the mass dependence (higher sSFR → more stochastic
heating → higher baseline).

### Off-feature windows where T_PAH < 5%

| λ_rest (μm) | Observed z (MIPS 24 μm) | Description |
|-------------|------------------------|-------------|
| 9.0–10.5 | 1.3–1.7 | Gap between 8.6 and 11.3 μm features |
| 11.8–12.5 | 0.9–1.0 | Gap between 11.3 and 12.7 μm features |
| > 13.5 | < 0.78 | Long-wavelength continuum tail |
| 6.5–7.2 | 2.3–2.7 | Short of the 7.7 μm complex |

### Implementation

`PAHModel._estimate_shared_slope_baseline(bins_data, z_union, T_grid, feature_threshold=0.05)`:

1. Compute T_total(z) = Σ_g T_g(z) at every data point.
2. Flag off-feature points: T_total(z) < 0.05 × max(T_total).
3. Fit the design matrix [log(λ_rest), bin_0_indicator, ..., bin_{M-1}_indicator]
   via WLS → returns γ (shared slope) and log10(A_m) per bin.

`PAHModel.fit_forward_model_multibin(..., baseline_method="shared_slope")`:

- Pre-fits γ from off-feature points (step above); fixes it for the main optimizer.
- Only log10(A_m) per bin is free in the optimizer → n_base = M (not 2M).
- Parameter count for 3-bin run: 3 α + 2 ratios + 1 τ_sil + 3 logA = 9 params
  (vs 12 with independent baseline).

### Fallback

If fewer than M+2 off-feature points are found (data doesn't cover the off-feature
windows), the method reverts to `baseline_method="independent"` automatically.

### Expected outcome

- γ ≈ 1.5–2.5 (warm SED slope in mid-IR; dustier/hotter galaxies → steeper)
- A_m inversely ordered with mass: lower-mass galaxies have higher warm continuum
  (higher sSFR → more stochastic VSG heating) → A_low-M > A_high-M
- α ordering should stabilise (no longer free to invert to absorb baseline curvature)
- χ²_red reduction: if the elevated χ²_red was partly from baseline misfit, it will
  drop; if purely astrophysical scatter, it will stay ~8–9 (both outcomes are informative)

---

## Objective 2 — K-fold source partitioning

### Problem

Every dither run uses the same source catalog. All spectral points in the pseudo-spectrum
share the same galaxy population → residuals are correlated. The χ²_red = 8–9 is a mix of
real astrophysical scatter and inter-point correlation that are hard to disentangle.

### Approach

Split the SFG catalog into K=3 non-overlapping subsets. Run K stacking jobs per dither
scheme, each promoting a distinct 1/K fraction of SFGs to `population_class=0` (signal)
and demoting the remaining (K−1)/K to `population_class=1` (nuisance deblender).
QTs (class=2) are copied unchanged into all K catalogs.

Stacking all three against the same map produces three pseudo-spectra whose class-0 flux
values at every z-bin are drawn from **disjoint** galaxy populations → independent.

### Implementation

Script: `uv run prepare-cosmos2020-catalog`

Usage:
```bash
uv run prepare-cosmos2020-catalog \
  --catalog /path/to/COSMOS2020_LePhare_v2.0.1.fits \
  --splits 3 \
  --split-type labels \
  --stem cosmos2020_mass \
  --output-dir /path/to/output/
```

Outputs `cosmos2020_mass_split0of3.parquet`, `split1of3.parquet`, `split2of3.parquet`.

TOML usage: `split_type = "labels"` reads `population_class` column directly
(0=sfg_signal, 1=sfg_nuisance, 2=qt).

### 3-run dither scheme for K=3

4-run scheme (Δz=0.15, offsets 0, 0.0375, 0.075, 0.1125) → 3-run equivalent:
```toml
bins = [[0.3, 0.45], [0.45, 0.60], [0.60, 0.75], ...]   # offset 0.00
bins = [[0.35, 0.50], [0.50, 0.65], [0.65, 0.80], ...]  # offset 0.05
bins = [[0.40, 0.55], [0.55, 0.70], [0.70, 0.85], ...]  # offset 0.10
```

3 runs × K=3 catalogs = 9 total stacking jobs (each job loads 1/3 of SFGs as signal).
With ~450 SFGs per mass bin in COSMOS2020, K=3 gives ~150 sources/z-bin → SNR ~2.

### What this buys

1. **Valid χ²**: points from different K-splits at the same z-bin are independent →
   the denominator of χ²_red is honest for the first time.
2. **Empirical σ_α**: std(α_A, α_B, α_C) / √(K−1) gives σ_α directly from data,
   no bootstrap assumptions.
3. **Correlation diagnosis**: compare χ²_red from pooled (all K) vs single-run fits.
   If χ²_red(pooled) << χ²_red(single), the elevation was correlation not scatter.

### Noise budget

With K=3, each split carries 1/3 of the sources → per-point stacking noise ×√3.
But K independent measurements combine to the same total information.

| Setup | Sources/z-bin | Noise/pt | Points | Combined SNR |
|-------|---------------|----------|--------|-------------|
| 1-run, no split | 450 | σ₀ | n | √n × 450/σ_gal |
| K=3, 3 runs | 150 | σ₀√3 | 3n | √(3n) × 150√3/σ_gal = √n × 450/σ_gal |

Same total SNR, but now with independent realisations.

---

## Code changes (this branch only)

1. **`PAHModel._estimate_shared_slope_baseline`** — new method (implemented ✓)
2. **`PAHModel.fit_forward_model_multibin`** — new `baseline_method` and `off_feature_threshold`
   parameters; returns `gamma` and `baseline_method` keys (implemented ✓)
3. **`prepare_cosmos2020_catalog.py`** — script for K-fold parquet generation (implemented ✓)

---

## What is NOT in this branch

The following are real and important but deferred to pah-forward-model-5:

- **Error rescaling** by √(χ²_red): `rescale_alpha_errors(result, chi2_red)`
- **Robustness suite**: baseline degree variation (1–3), ±0.1 dex bin edge shifts,
  jackknife over dither runs
- **Talk figure set**: 5-panel figure builder (`create_pah_talk_figures`)
- **T_dust correction figure**: ΔT_dust vs z before/after PAH correction
- **2D (M*, σ_SFR) joint fitter**: extend `fit_forward_model_multibin` to 2D bin grid
- **Combined slope significance**: fit α(M*) = α₀ × (M*/10^{10.5})^β with rescaled errors

See `pah-forward-model-5-brief.md` for the deferred scope.
