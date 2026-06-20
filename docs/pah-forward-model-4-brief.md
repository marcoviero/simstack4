# PAH Forward Model — Branch 4 Brief

**Goal**: Bring the PAH tomographic stacking result to talk-quality and referee-quality.  
The primary deliverable is a measurement of α(M*) — the PAH/FIR amplitude as a function of stellar mass — that an audience can trust and a referee can challenge and fail to break.

---

## What pah-forward-model-2 established

- **Method validated**: MIPS 24 μm dithered stacking produces a coherent pseudo-spectrum of f₂₄/f_peak vs λ_rest. The 70 μm null test passes; the forward model self-consistency test passes.
- **Marginal detection**: SNR ~1.3 per mass bin (after formal bootstrap errors). Collective evidence is stronger: all three α values are positive, follow the expected direction, and the bump SNR is 2–2.6×.
- **Trend direction confirmed**: α decreases with M* at −0.10 dex/dex. This is the PAH deficit direction (consistent with GOALS / Spitzer IRS literature) and opposite to what a simple luminosity bias would produce.
- **No silicate absorption**: τ_sil = 0.000 ± 0.081 in normal MS galaxies at z~0.5–3.5.
- **χ²_red = 3.365**: scatter is 1.83× formal errors. The elevated χ²_red is understood (baseline polynomial does not capture real continuum evolution; each point is a different galaxy population at different z).
- **Pending**: 4-bin mass run (20260612_190116…), σ_SFR cross-cut (3/4 runs), accordion vs uniform comparison. All notebooks are set up and waiting for execution.

---

## What pah-forward-model-3 must do

### 1. Complete and consolidate measurements

- Execute pending notebooks (`2026-06-12-…PAH-dithered-dz015.ipynb`, `2026-06-14-…sigma_sfr…`, `2026-06-15-…accordion…`).
- Complete σ_SFR stacking run 4 (offset 0.1125); fill in `RUN_DATES[3]` in notebook cell `16153c5d`.
- Accept or reject accordion binning based on σ_α ratio. If accordion ≲5% better → discard (simpler wins); if >10% better → re-run 4-bin mass scheme with accordion.

### 2. Honest error budget

Formal bootstrap errors underestimate true scatter by 1.83×. For publication:

- **Rescaled errors**: multiply all σ_α by √(χ²_red). Report both in the table.
- **Baseline robustness test**: refit with 1st, 2nd (current), and 3rd-order polynomial baseline per bin. Show α values are stable across baseline order.
- **Bin edge robustness test**: shift all mass bin edges by ±0.1 dex; refit; show slope is unchanged.
- **Jackknife over runs**: drop one of the 4 dither runs at a time; show α values are stable.
- **Summary**: if α(M*) trend survives all of the above, report as significant at the rescaled-error SNR.

### 3. Combined slope significance

Fit a power law α(M*) = α₀ × (M*/10^{10.5})^β simultaneously across all mass bins:
- Report β with 1σ uncertainty (rescaled errors).
- If the 4-bin run is better than 3-bin (more bins, similar per-bin SNR), use 4-bin for the slope.
- Physical claim: "PAH/FIR amplitude decreases with stellar mass at −0.10 ± 0.XX dex/dex."

### 4. σ_SFR cross-cut

Once run 4 is complete, fit the joint forward model:
- `alpha(M*, sigma_sfr)` — does α decrease with σ_SFR at fixed M*?
- Physical interpretation: UV radiation field ∝ σ_SFR → PAH grain destruction. If confirmed, the mechanism is not halo mass but radiation field intensity.
- Cross-check: partial correlations — hold M* fixed, vary σ_SFR and vice versa. Report which drives the trend.

### 5. PAH correction to T_dust

- Two-pass SED fitting: exclude 24 μm (inflation=10000) → get f_peak → compute f₂₄_PAH = α(M*,z) × f_peak → reduce inflation to 3–5× (residual uncertainty ≈ few percent) → refit.
- Compare T_dust posteriors before and after correction at z = 1.5–2.5 (where MIPS probes 7.7+8.6 μm).
- Report: ΔT_dust bias in K, fraction of bins promoted Tier C → Tier B.
- This result speaks directly to the dust temperature evolution claim (Viero+22) which is one of the branch's primary science goals.

---

## Talk figure set (5 slides / figures)

### Figure 1: The method — why dithering works
- Left: MIPS 24 μm bandpass overlaid with PAH template spectrum at z=0, 1, 2.
- Right: z-bin layout showing how 4 dither runs tile λ_rest space (use the z-bin width diagram from `2026-06-15` notebook).
- Message: we are not detecting individual PAH lines — we are measuring the bandpass-modulated envelope as a function of redshift.

### Figure 2: Raw pseudo-spectra
- 4 panels (one per mass bin), f₂₄/f_peak vs λ_rest, all 4 runs combined.
- Overplot: best-fit baseline (polynomial) as a smooth curve.
- Message: the modulation is real and coherent across runs; the 70 μm null test shows it is not a baseline artifact.

### Figure 3: Detrended residuals — the detection
- Same 4 panels, (f₂₄/f_peak)/baseline − 1 vs λ_rest.
- Overplot: model PAH template with fitted α_m.
- Shade the 7.7+8.6 μm zone (z ≈ 1.6–2.0) — the main feature.
- Annotate each panel: α ± σ_α (rescaled), bump SNR.
- Message: the PAH complex is detected at 2–3× above the noise floor. The amplitude decreases with mass.

### Figure 4: α(M*) — the science result
- α vs log M* with 1σ error bars (rescaled).
- Overplot: literature points where available (Smith+2007 SINGS IRS at z~0, Shi+2011 GOALS LIRGs, Galliano+2021 review trend).
- Inset or lower panel: α vs σ_SFR for the two mass bins, if the σ_SFR run is complete.
- Overplot: the PAH deficit trend (Smith+2007: PAH EW decreases with sSFR/L_IR).
- Message: stacked z~0.5–3.5 galaxies follow the same PAH deficit direction as local resolved samples.

### Figure 5: Impact on T_dust
- Left: SED fits for the highest-z bins (z~1.5–2.5) before and after PAH correction.
- Right: ΔT_dust vs z, showing the bias introduced by un-corrected 24 μm contamination.
- Message: ignoring PAH contamination biases T_dust upward by X K at z~1.5–2.0; the PAH correction is necessary for accurate T_dust evolution measurements.

---

## Referee defense strategy

**Q: "The χ²_red = 3.36 means your formal errors are too small. The result is not significant."**  
A: We report rescaled errors explicitly (×1.83). With rescaled errors, α is positive in all bins and the slope is non-zero at Xσ. The elevated χ²_red is expected (see §5 of the paper): each data point is a different galaxy population at different redshift, so astrophysical scatter (in PAH/FIR ratio within each mass–z bin) is real and irreducible with more runs. We show the result is stable to baseline order variation, bin edge shifts, and jackknife over runs.

**Q: "The 12.7 μm ratio r₂ hits the prior boundary. How do you know your α values are unbiased?"**  
A: r₂ only affects points near z~0.9 (a narrow slice). We show α is stable when r₂ is fixed to any value in [2, 5]. The 7.7+8.6 μm zone (z~1.6–2.0) — which provides most of the constraining power — is insensitive to r₂.

**Q: "How do you know the f₂₄/f_peak modulation is not continuum evolution?"**  
A: Three lines of evidence: (1) the 70 μm null test — the same forward model applied to f₇₀/f_peak gives α consistent with zero; (2) the modulation pattern matches the MIPS 24 μm PAH template T(z) in shape, not a smooth z-trend; (3) varying the baseline polynomial order does not change the residual modulation amplitude.

**Q: "Your mass bins are broad. This is really an L_IR trend."**  
A: We show the σ_SFR cross-cut (Figure 4 inset): at fixed M*, α decreases with σ_SFR. This is opposite to what a simple L_IR trend (more IR-luminous = more PAH) would predict. The trend is driven by radiation field intensity, not total luminosity.

**Q: "24 μm at z>2 is sampling rest-frame <8 μm — you're in the PAH forest, not a single feature."**  
A: Correct — at z>2 the bandpass integrates over the 6.2+7.7+8.6 μm complex simultaneously. The forward model accounts for this via the template kernel T(z), which includes all three features weighted by the bandpass response. The measured α is the amplitude of the full PAH complex relative to the FIR peak; it is not a single-line measurement.

---

## New code needed in pah-forward-model-3

1. **Error rescaling utility**: `rescale_alpha_errors(result, chi2_red)` — multiply σ_α and ratio_errors by √(chi2_red); add to `PAHModel` or `analyze_pah.py`.

2. **Robustness suite**: `run_robustness_tests(df, group_col, bin_edges, feature_groups)` — sweeps baseline degree (1–3), ±0.1 dex bin edge shifts, and jackknife-over-runs. Returns a DataFrame of (α, σ_α) per perturbation for each bin. Add to `pah_model.py` or `analyze_pah.py`.

3. **Talk figure builder**: `create_pah_talk_figures(result, df_combined, mass_bins, out_dir)` in `plots.py` — generates Figures 1–4 above in a single call; designed for direct use in keynote/beamer.

4. **T_dust bias figure**: `create_pah_correction_tdust_plot(wrapper_corrected, wrapper_uncorrected, z_range)` in `plots.py` — already has the SED grid plotter; add a ΔT_dust panel.

5. **Joint (M*, σ_SFR) fitter**: extend `PAHModel.fit_forward_model_multibin` to accept a 2D bin structure (M* × σ_SFR), fit all 6 cells jointly with shared group ratios and τ_sil. Report partial correlation coefficients.

---

## Config and data notes

- `config/cosmos25_PAH_dithered.toml` now contains all four dither schemes as commented reference blocks. The active `bins =` line (last uncommented) should be set to the scheme you want to run.
- `config/cosmos25_PAH_dithered_3d.toml` is the σ_SFR config (2 mass × 3 σ_SFR bins). Run 4 needs `bins` updated to the offset 0.1125 block.
- COSMOS catalog must have `log_sigma_sfr` and `sersic_reliable` columns (produced by `prepare-cosmos-catalog --paper p26`).

---

## K-fold Source Partitioning — Making Spectral Points Independent

**Problem**: Every dither run uses the same source catalog. All spectral points in the pseudo-spectrum share the same galaxy population, so residuals are correlated. The elevated χ²_red = 3.36 is a mix of real astrophysical scatter and this correlation — they are hard to disentangle without independent realisations of the data.

**Proposed approach**: Split the SFG catalog into K non-overlapping subsets. Run K stacking jobs per dither scheme, each promoting a distinct 1/K fraction of sources to the `sfg_a` signal class and demoting the remaining (K−1)/K to a `sfg_b` deblending nuisance class. **All sources remain in every catalog** — simstack must account for every galaxy to deblend the map correctly.

**Example (K=3)**:
```
Catalog A:  sfg_a = source_id % 3 == 0  (1/3 of SFGs)
            sfg_b = remaining 2/3         (deblend nuisance)
            qt    = all QTs
Catalog B:  sfg_a = source_id % 3 == 1  (1/3 of SFGs, non-overlapping with A)
            sfg_b = remaining 2/3
            qt    = all QTs
Catalog C:  sfg_a = source_id % 3 == 2
            sfg_b = remaining 2/3
            qt    = all QTs
```

Stacking all three against the same map produces three pseudo-spectra whose `sfg_a` flux values at every z-bin are drawn from disjoint galaxy populations → independent (to the extent that the dominant variance is source shot noise, not pixel noise).

**What this buys**:
1. **Valid χ²**: spectral points from different K-splits at the same z-bin are independent, so χ²_red has a correct denominator for the first time.
2. **Empirical covariance**: the K×K scatter matrix of α values is the true covariance — no bootstrap assumptions needed.
3. **Empirical σ_α**: std(α_A, α_B, α_C) / √(K−1) gives σ_α directly from data.
4. **Model selection**: AIC/BIC between forward models (free per-bin α vs linear evolution prior) becomes meaningful with independent data.

**Noise cost and feasibility**:
Each sfg_a carries 1/K of the sources → per-point stacking noise ×√K larger. But K independent measurements combine back to the same total information.

| Catalog | Sources/z-bin (M*=10.6–11) | K=2 → N/split | K=3 → N/split |
|---------|---------------------------|---------------|---------------|
| COSMOS-Web | ~100 | ~50 (marginal) | ~33 (too noisy) |
| COSMOS2020 | ~446 | ~223 (good) | ~149 (viable) |

**COSMOS-Web can only support K=2.** COSMOS2020 can support K=3 comfortably, K=4 at push. This is one more reason to run the COSMOS2020 stacking.

**Independence caveat**: The K catalogs share the same map pixel noise, so the noise floor is correlated. The `sfg_a` and `sfg_b` layers within each catalog are also mildly anti-correlated through the PSF confusion matrix (the same algebraic anti-correlation as the bootstrap A/B split). In practice, at MIPS 24 μm depths where source shot noise >> pixel noise, these correlations are second-order.

**Combination strategies**:
- **Pool** (preferred for χ² fitting): stack all K pseudo-spectra into one DataFrame (K × n_dithers × n_zbins rows); fit the forward model once. Points from different splits at the same z are independent → the χ² is valid.
- **Ensemble** (preferred for σ_α): fit K times independently; the mean and std of α across splits gives the result and a fully empirical error bar with no model assumptions. Cleanest for paper Table 1.

**Implementation** (catalog prep only — no changes to stacking algorithm):

Add `n_catalog_splits` parameter to `prepare_cosmos_catalog`:
1. Assign each SFG a split_id = `hash(galaxy_id) % n_catalog_splits`
2. For split i: promote sources with `split_id == i` to `sfg_a`, relabel rest as `sfg_b`
3. QTs and other classes copied unchanged into all K output catalogs
4. Write `catname_split{i}.parquet` + corresponding TOML for each i

The TOML just needs a different catalog path and updated `[catalog.classification.binning]` label scheme; the stacking config (z-bins, maps, beam) is identical across all K runs.

---

## SKILL.md to update at branch start

Add to `open questions`:
- [ ] Rescaled-error significance: what is the joint α(M*) slope SNR after ×1.83 rescaling?
- [ ] Accordion verdict: is σ_α(accordion)/σ_α(uniform) < 0.95 in any mass bin?
- [ ] σ_SFR direction: does α decrease with σ_SFR at fixed M*?
- [ ] T_dust bias: what is the mean ΔT_dust at z=1.5–2.5 from un-corrected 24 μm PAH?

Remove from `open questions` once resolved:
- σ_z0 and f_cat trade study (answered: dz=0.15 × 4 outperforms dz=0.10 × 2 for this depth)
- τ_sil detection (answered: not detected)
- NoiseModel sigma_ref (validated at ~9 SNR/point median Tier B)
