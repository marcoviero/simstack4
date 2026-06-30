# PAH Forward Model — Branch 5 Summary

**Goal.** Add **redshift-evolution capability** to the PAH tomographic forward model — let
the PAH/continuum amplitude and the line ratios drift *within* a stellar-mass bin with
specific SFR as the MIPS bandpass sweeps rest wavelength with z. The diagnostics this
required also pinned down two things that limit the measurement: the **baseline Wien-slope
(α) systematic** and the **astrophysical-scatter floor**.

See `pah-forward-model-5-brief.md` for the design rationale and `pah-forward-model-6-brief.md`
for deferred (talk/referee) scope.

---

## New library capabilities

| Where | What |
|-------|------|
| `dust_evolution.py` | `main_sequence_ssfr(z, log_mstar, relation)` — Speagle+2014 / Schreiber+2015 sSFR(z, M*) proxy; the physical "evolution axis". |
| `pah_dither.py` | `TruthSpectrum` gains `eta_ssfr_amp` / `eta_ssfr_ratio` / `s_pivot` (z-aware injection; zero-η reproduces the static spectrum). `fisher_evolution` — CRLB on the evolution slopes, 24-only vs 24+70. `evolution_recovery_sweep` — Monte-Carlo bias/scatter/rail-rate vs SNR/bands/prior. |
| `pah_spectrum.py` | `_prepare` extended with per-band baselines + per-row `z_mid`/`log_ssfr`. `fit_evolving` — multi-band, sSFR-anchored amplitude + line-ratio evolution with shared global slopes `η_A`/`η_g`; `eta_prior_sigma` Gaussian slope prior. `fit_with_alpha` — profiles the cold-baseline Wien slope α with a Gaussian prior (default strong at 2), wrapping `fit_shared`/`fit_evolving`. |

**Model (per point i, mass bin m, band b):**
```
ŝ_i          = log_ssfr(z_i, M_m) − s_pivot
alpha_i      = alpha_m · 10^(η_A · ŝ_i)
r_g(ŝ_i)     = r_g0 · 10^(η_g · ŝ_i)          (g ≥ 1; r_0 ≡ 1, η_0 ≡ 0)
f_obs_{i,b}  = C_m · f_cold_norm_{i,b} + alpha_m·10^(η_A ŝ_i)·Σ_g r_g0·10^(η_g ŝ_i)·K_{g,i,b}
```
Per-bin `[C_m, alpha_m]` and shared `r_g0` stay linear (alternating WLS); the few global
slopes (and optionally α) are profiled by an outer optimizer. MIPS 70 — a different rest λ
at the same z — breaks the amplitude/ratio degeneracy.

---

## Key findings

1. **Method validated in simulation.** `fit_evolving` recovers injected `η_A`/`η_g`
   unbiased; the static fit is biased under evolution; 70 µm halves the η scatter
   (`evolution_recovery_sweep`, `fisher_evolution`).
2. **On real COSMOS2020 the within-bin evolution is scatter-limited → an upper limit.**
   A controlled sim with the *correct* baseline + scatter tuned to χ²_red≈6 reproduces a
   spurious η_A, so the real η_A=−2.37 is an artifact of galaxy-to-galaxy PAH scatter, not
   a detection. Minimum detectable η_A ≈ 0.8 (24+70 + `eta_prior_sigma≈1`) / ≈1.4 (24-only).
3. **The baseline is a power law of slope α=2** (the `(1+z)^(−α)` Wien-side splice in
   `greybody_model`, not an exponential tail — this corrects an earlier wrong "collapse"
   claim). **A_pah is strongly α-sensitive** (Δα=±0.5 → A_pah ×3–4 and flips the mass
   slope); the EW *slope* (differential) is more robust.
4. **α can now be fit, not assumed.** `fit_with_alpha` recovers an injected α=2.4 to ±0.05
   with 24+70 (70 µm is pure continuum at z<3.5 → pins the slope); single-band leans on the
   prior. Use with multi-band / z>4 data so α is data-driven.
5. **Trustworthy headline unchanged:** L_PAH/L_IR ≈ flat; EW slope +0.37 (3.3σ).

---

## Notebook (local only — `notebooks/` is git-excluded)

`2026-06-28-…-ssfr-line-evolution-3-parts.ipynb`:
- **§3c** evolving fit + an honest detection/upper-limit verdict.
- **§3d** Wien-slope (α) sensitivity sweep.
- **§3e** working-simulation-vs-real overlay (real pseudo-spectrum scatters around the PAH
  model exactly like a working sim → normal, just scatter-limited).
- **§9b** evolving fold-ensemble (cosmic-variance errors on η).

---

## Tests

`tests/test_pah_evolution_recovery.py` (7): shared-η recovery, static-model bias, 70 µm
breaks the ratio degeneracy, null → reduces to `fit_shared`, `eta_prior_sigma` tames the
runaway, `evolution_recovery_sweep` unbiased + 70 µm helps, `fit_with_alpha` recovers the
Wien slope. No regressions in the existing PAH/dust suites.

---

## Recommendations / next steps

- **Push to z > 4** (proposed accordion bins: Δz 0.15→0.30→0.45 over z 0.5→5.0): MIPS 70
  then sweeps 16.4/17 (z≈3.1–3.3) and 12.7/11.3 (z≈4.5–5.2) — an independent re-measurement
  of those features and a z≈1→5 lever on the shared ratios; 24 (short) + 70/100 (long)
  bracket the warm continuum.
- **Pin α with multi-band** (`fit_with_alpha`, 24+70[+100]) to shrink the ±3–4× absolute
  baseline systematic instead of assuming α=2.
- **Quote the EW slope** (differential) with an α systematic; report η via the disjoint-fold
  ensemble, never the formal curvature errors.
- Deferred to branch 6: error-rescaling utility, robustness suite, α(M*)/talk figures,
  T_dust-correction figure, σ_SFR cross-cut.

---

## Files changed (tracked)

- `src/simstack4/dust_evolution.py`, `pah_dither.py`, `pah_spectrum.py`
- `tests/test_pah_evolution_recovery.py`
- `docs/pah-forward-model-{5-brief,5-summary,6-brief}.md`, `pah-forward-model-4-brief.md`
- `config/cosmos20_PAH_dithered_3cats.toml`, `src/simstack4/scripts/prepare_cosmos2020_catalog.py`,
  `pyproject.toml`, `CLAUDE.md`

(Notebook edits and `.claude/` working files are intentionally not tracked.)
