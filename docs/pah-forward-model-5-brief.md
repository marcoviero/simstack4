# PAH Forward Model — Branch 5 Brief

**Goal**: Add redshift-evolution capability to the PAH tomographic forward model. The current
model ties the short- and long-wavelength PAH behaviour of a stellar-mass bin to a single
amplitude and a single set of band ratios. Because the MIPS 24 µm bandpass sweeps rest-frame
~7→18 µm *as a function of redshift within that bin*, and because specific star-formation rate
(sSFR) rises steeply with z along the main sequence, the high-z (short-λ) and low-z (long-λ)
points are drawn from physically different galaxy populations. The model must evolve too.

This branch is about the physics ingredient, not the talk figures (those moved to
`pah-forward-model-6-brief.md`).

---

## The problem

`PAHSpectrumModel.fit_shared` (`src/simstack4/pah_spectrum.py`) describes mass bin *m* as

```
f_obs(z) = C_m · f_cold_norm(z) + alpha_m · Σ_g r_g · K_g(z)
```

with `alpha_m` (PAH/continuum amplitude) and `r_g` (shared feature-group ratios, `r_0 ≡ 1`)
**constant in z**. But the method depends on the bandpass sweeping rest wavelength with z:

- high-z points probe the 7.7/8.6 µm **ionized** PAH complex,
- low-z points probe the 11.3/12.7 µm **neutral** bands.

Along that sweep, sSFR(z, M*) climbs by ~1–1.5 dex across z = 0.5→3.5 at fixed M*. The
radiation-field hardness and PAH/continuum ratio track sSFR, so both `alpha_m` and the
ionized/neutral band ratios should **drift across the bin**. Forcing them constant can bias the
recovered amplitude and inflate χ² (consistent with the χ²_red ≈ 8–9 seen in branch 4).

---

## The model

For pseudo-spectrum point *i* in mass bin *m*, with centered specific SFR
`ŝ_i = log_ssfr(z_i, M_m) − s_pivot`:

```
alpha_i      = alpha_m · 10^(η_A · ŝ_i)                    # amplitude evolution
r_g(ŝ_i)     = r_g0 · 10^(η_g · ŝ_i)   (g ≥ 1; r_0 ≡ 1, η_0 ≡ 0)   # ratio evolution
f_obs_{i,b}  = C_m · f_cold_norm_{i,b}
             + alpha_m · 10^(η_A ŝ_i) · [ K_{0,i,b} + Σ_{g≥1} r_g0 · 10^(η_g ŝ_i) · K_{g,i,b} ]
```

Design decisions (set 2026-06-28):

- **Anchor**: the within-bin evolution is tied to **sSFR(z, M*)** (physical), not to redshift
  directly. Per-point `log_ssfr` comes from the assembled spectrum (derived `sfr_med − M*`), with
  a Speagle+2014 main-sequence fallback. One global slope then applies coherently across mass bins.
- **What evolves**: **both** the amplitude (`η_A`) and the band ratios (`η_g` per non-reference
  group) — the ionized/neutral drift is the deeper effect and the reason short/long λ decouple.
- **Slope sharing**: the evolution slopes `η_A`, `η_g` are **single global parameters shared across
  mass bins** (like the existing shared `r_g`), keeping the fit stable at ~SNR 2 per point.
- **70 µm leverage**: MIPS 70 probes a *different* rest wavelength at the same z (15–47 µm: the
  16.4/17.0 µm bands and FIR continuum), giving the independent handle that separates amplitude
  evolution from ratio evolution. Both bands are fit jointly, sharing `alpha_m`, `r_g0`, `η`.

The per-bin `[C_m, alpha_m]` and global `r_g0` stay **linear** given the slopes → the existing
alternating-WLS core is reused; only the few `η` slopes go through an outer nonlinear optimizer.

---

## Work plan

### Phase A — Quantify the concern (simulation + Fisher)
1. `main_sequence_ssfr(z, log_mstar, relation="speagle2014")` in `dust_evolution.py` (alongside
   `warm_temperature`/`cold_temperature`).
2. Extend `TruthSpectrum` (`pah_dither.py`) with `eta_ssfr_amp`, `eta_ssfr_ratio`, `s_pivot`,
   `ssfr_fn`; make amplitudes/ratios z-aware and thread through `simulate_dithered_fluxes`.
3. **Bias quantification**: fit static `fit_shared` to evolving truth — report the bias in
   recovered `alpha_m` (and mass ordering) and the χ² inflation.
4. **Identifiability**: evolution-aware Fisher diagnostic; CRLB on `η` for 24-only vs 24+70 →
   confirm 70 µm is required to pin the ratio slopes.

### Phase B — Evolving fitter
5. `PAHSpectrumModel.fit_evolving` (new sibling of `fit_shared`): multi-band, sSFR-anchored
   amplitude + ratio evolution, shared global slopes; per-point sSFR resolver with MS fallback.
   Returns `fit_shared`'s keys plus `eta_amp`, `eta_amp_err`, `eta_ratio`, `eta_ratio_err`,
   `s_pivot`.
6. Ensure the combined spectrum carries `f70_cold` and per-point `log_ssfr` (`analyze_pah.py`).
7. Errors via the **disjoint-fold ensemble** (`split{N}of3`): refit each fold, take the scatter on
   `alpha_m`, `η_A`, `η_g`, `r_g0`. No √χ² rescaling.

### Phase C — Validation
8. `tests/test_pah_evolution_recovery.py`: recover injected `η`; static-model bias guard;
   24-only degeneracy vs 24+70; shared-slope sanity; null test (zero evolution → `fit_shared`).
9. Wire `fit_evolving` into
   `notebooks/2026-06-28-load-json-fit-seds-redshift-stellar-mass-PAH-dithered-ssfr-line-evolution-3-parts.ipynb`
   over the three folds. Update CLAUDE.md module map + test table.

---

## Files

- `src/simstack4/dust_evolution.py` — `main_sequence_ssfr`.
- `src/simstack4/pah_dither.py` — `TruthSpectrum` evolution fields; evolution-aware Fisher.
- `src/simstack4/pah_spectrum.py` — `fit_evolving`.
- `src/simstack4/analyze_pah.py` — `f70_cold`, `log_ssfr` in combined df.
- `tests/test_pah_evolution_recovery.py` — new.
- the ssfr-line-evolution notebook; `CLAUDE.md`.

## Reused machinery (no rewrite)

`_prepare` per-band kernel slicing, `_wls` alternating core, delta-method errors
(`pah_spectrum.py`); `build_design_matrix`, `feature_band_curves`, `warm_continuum_kernel`;
`compute_pz_matrix`, `make_dndz`, `NoiseModel`, `shared_fraction_matrix`,
`simulate_dithered_fluxes`, `fisher_for_scheme` (`pah_dither.py`); `smoothed_ms_baseline`.

## Findings (2026-06-28)

- **Method validated on synthetic data.** `fit_evolving` recovers injected η_A/η_g
  unbiased; the static fit is biased under evolution; 70 µm halves the η_A scatter
  (`evolution_recovery_sweep`, `fisher_evolution`).
- **On real COSMOS2020 the evolving fit is scatter-limited, not a detection.**
  η_A railed to −2.37 on the real data. A *controlled* sim with the correct baseline
  + astrophysical scatter tuned to χ²_red≈6 reproduces a spurious η_A (truth 0 →
  recovered ~+1.5), so the runaway is driven by **galaxy-to-galaxy PAH scatter**, NOT
  baseline error. **η is an upper limit**; min detectable η_A ≈ 0.8 (24+70 +
  `eta_prior_sigma≈1`) / ≈1.4 (24-only).
- **Correction to an earlier wrong claim:** the baseline is NOT an exponential Wien
  tail that collapses — `greybody_model` splices a power law `f_ν ∝ ν^(−α) ∝
  (1+z)^(−α)` with `α=2` short-ward of rest ~71 µm. It is a gentle, reasonable warm
  continuum shape. But **A_pah is strongly sensitive to the assumed slope α**
  (Δα=±0.5 → A_pah ×3–4 and flips the mass slope) — a real systematic on absolute PAH
  amplitudes; the EW *slope* is more robust.
- **α can now be fit (not fixed at 2):** `PAHSpectrumModel.fit_with_alpha` profiles the
  Wien slope with a Gaussian prior (default strong at 2) by re-tilting the baseline by
  `(1+z)^(α_ref−α)` around the inner fit. Injection-recovery: 24+70 recovers an injected
  α=2.4 to ±0.05 (70 µm is pure continuum at z<3.5 → pins the slope); single-band leans
  on the prior. Use with the new z>4 / multi-band data so α is data-driven.
- **Mitigation / diagnostics added:** `eta_prior_sigma` Gaussian slope prior; notebook
  §3d (α-sensitivity sweep) and §3e (working-sim-vs-real overlay, which shows the real
  pseudo-spectrum scatters around the PAH model exactly like a working sim — normal,
  just scatter-limited). Report η via the disjoint-fold ensemble, never formal errors.
- **Next:** pin α from external mid-IR continuum constraints (or `wien_mode="physical"`)
  for absolute amplitudes. The trustworthy science is unchanged: L_PAH/L_IR≈flat, EW
  slope +0.37 (3.3σ).

## Out of scope (→ branch 6)

Error-rescaling utility, robustness sweep, α(M*) talk/referee figures, T_dust correction figure,
σ_SFR cross-cut — see `pah-forward-model-6-brief.md`.
