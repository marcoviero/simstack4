# PAH Forward Model — Branch 7 Brief

**Goal**: Bring the PAH tomographic stacking result to talk-quality and referee-quality.
The primary deliverable is a measurement of α(M*) that an audience can trust and a referee
can challenge and fail to break. Builds on branches 4–6 (baseline fix + K-folding, and the
sSFR-evolution + α-fitting model). Start once branch 6 has delivered the α-pinned,
multi-band z>4 measurement.

---

## Prerequisites from branches 4–6

- Shared-slope baseline implemented and validated (`baseline_method="shared_slope"`)
- K=3 COSMOS2020 catalogs generated and stacking runs complete
- sSFR-evolution + `fit_with_alpha` model available and applied to the z>4 / multi-band
  stacks (branch 6); α pinned, evolution reported as detection-or-upper-limit
- χ²_red re-assessed (astrophysical scatter vs baseline; see branch-5 findings)

---

## Objective 1 — Honest error budget

### 1a. Error rescaling utility

```python
def rescale_alpha_errors(result, chi2_red):
    """Multiply σ_α and ratio_errors by √(chi2_red). Returns updated result dict."""
```

Add to `PAHModel` or `analyze_pah.py`. Report both raw and rescaled errors in the paper
table. Physical interpretation: χ²_red > 1 means scatter exceeds formal errors — the
extra factor is real astrophysical variance within each (mass, z) bin.

### 1b. Robustness suite

```python
def run_robustness_tests(df, group_col, bin_edges, feature_groups, n_runs_all):
    """
    Sweep over systematic perturbations and return DataFrame of (α, σ_α) per bin.
    
    Tests:
      - baseline_method: "independent" vs "shared_slope"
      - bin edge shifts: ±0.1 dex in stellar mass
      - jackknife over dither runs: drop one run at a time
    
    Returns DataFrame: columns = [test_name, bin_label, alpha, alpha_err, chi2_red]
    """
```

Add to `pah_model.py` or `analyze_pah.py`. If α(M*) trend survives all perturbations,
report as significant at the rescaled-error SNR.

### 1c. Combined slope significance

Fit α(M*) = α₀ × (M*/10^{10.5})^β simultaneously across all mass bins:
- Report β with 1σ uncertainty (rescaled errors).
- Physical claim: "PAH/FIR amplitude decreases with stellar mass at −β ± σ_β dex/dex."

---

## Objective 2 — σ_SFR cross-cut

Once the (M*, σ_SFR) 2D stacking runs are complete:

- **2D fitter extension**: extend `PAHModel.fit_forward_model_multibin` to accept a 2D
  bin structure (M* × σ_SFR), fit all cells jointly with shared group ratios and τ_sil.
  Report partial correlation coefficients.
- **Science question**: does α decrease with σ_SFR at fixed M*?
- **Physical interpretation**: UV radiation field ∝ σ_SFR → PAH grain destruction.
  If confirmed, the mechanism is radiation field intensity, not halo mass.
- **Cross-check**: partial correlations — hold M* fixed, vary σ_SFR and vice versa.

---

## Objective 3 — PAH correction to T_dust

Two-pass SED fitting:
1. Exclude 24 μm (inflation=10000) → fit greybody → get f_peak
2. Compute f₂₄_PAH = α(M*,z) × T_m(z) × f_peak (using measured α from forward model)
3. Reduce 24 μm inflation to 3–5× (residual uncertainty ≈ few percent) → re-fit

Deliverable: `create_pah_correction_tdust_plot(wrapper_corrected, wrapper_uncorrected, z_range)` in `plots.py`:
- Left panel: SED fits for the highest-z bins (z~1.5–2.5) before and after correction
- Right panel: ΔT_dust vs z, showing the bias from uncorrected MIPS 24 μm contamination
- Report: ΔT_dust bias in K, fraction of bins promoted Tier C → Tier B

This speaks directly to the Viero+22 T_dust evolution claim.

---

## Objective 4 — Talk figure set

Five figures for a conference talk / paper:

### Figure 1: The method — why dithering works
- Left: MIPS 24 μm bandpass overlaid with PAH template spectrum at z=0, 1, 2
- Right: z-bin layout showing how 3+ dither runs tile λ_rest space
- Message: the bandpass sweeps rest-frame wavelength; dithering gives dense sampling

### Figure 2: Raw pseudo-spectra
- One panel per mass bin, f₂₄/f_peak vs λ_rest, all dither runs combined
- Overplot: best-fit baseline (shared power law) as smooth curve
- Overplot: 70 μm null test (should be flat)

### Figure 3: Detrended residuals — the detection
- Same panels, (f₂₄/f_peak)/baseline − 1 vs λ_rest
- Overplot: model PAH template with fitted α_m
- Shade the 7.7+8.6 μm zone (z ≈ 1.6–2.0)
- Annotate each panel: α ± σ_α (rescaled), bump SNR

### Figure 4: α(M*) — the science result
- α vs log M* with 1σ error bars (rescaled)
- Overplot: literature points (Smith+2007 SINGS IRS z~0, Shi+2011 GOALS)
- Inset: α vs σ_SFR for the two mass bins (if 2D run complete)

### Figure 5: Impact on T_dust
- Left: SED fits at z~1.5–2.5 before and after PAH correction
- Right: ΔT_dust vs z

Builder: `create_pah_talk_figures(result, df_combined, mass_bins, out_dir)` in `plots.py`.

---

## Referee defense strategy

**Q: "The χ²_red = 8–9 means your formal errors are too small."**
A: We report rescaled errors explicitly (×√χ²_red). With rescaled errors, α is positive
in all bins and the slope is non-zero at Xσ. The elevated χ²_red is astrophysical scatter
(galaxy-to-galaxy PAH/FIR ratio variation within each mass–z bin). The 70 μm null test
shows χ²_red ~1, proving bootstrap errors correctly calibrated for noise; the elevation
is PAH-signal-specific. K-fold validation shows χ²_red barely changes (9.35 → 8.72)
when sources are independent across runs, confirming astrophysical origin.

**Q: "The 12.7 μm ratio r₂ hits the prior boundary."**
A: r₂ only affects points near z~0.9. We show α is stable when r₂ is fixed to any value
in [2, 5]. The 7.7+8.6 μm zone (z~1.6–2.0) provides most constraining power and is
insensitive to r₂.

**Q: "How do you know the modulation is not continuum evolution?"**
A: (1) 70 μm null test gives α consistent with zero; (2) modulation pattern matches
T(z) template in shape, not a smooth z-trend; (3) varying the baseline method
(independent vs shared-slope) does not change the residual modulation amplitude.

**Q: "Your mass bins are broad. This is really an L_IR trend."**
A: σ_SFR cross-cut: at fixed M*, α decreases with σ_SFR — opposite to what an L_IR
trend (more luminous = more PAH) predicts. The trend is radiation field intensity.

---

## Config and data notes

- `config/cosmos25_PAH_dithered.toml`: active 3-run scheme (Δz=0.15, offsets 0, 0.05, 0.10)
- `config/cosmos25_PAH_dithered_3d.toml`: σ_SFR config (2 mass × 3 σ_SFR bins)
- COSMOS2020 K=3 catalogs: `cosmos2020_mass_split{0,1,2}of3.parquet`
- For paper Table 1: report both raw σ_α and rescaled σ_α × √χ²_red
