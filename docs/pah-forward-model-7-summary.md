# PAH Forward Model — Branch 7 Summary

**Goal.** Turn the branch-6 measurement (PAH-to-continuum ratios rise with stellar mass,
α-robust) into a defensible physical narrative: identify the mechanism signal (band ratios),
confront it with theory (Narayanan+26), stress-test the evolving-template machinery on
realistic simulations, and audit the money plots on the corrected pipeline.

See `pah-forward-model-7-brief.md` for the running detail; `pah-forward-model-8-brief.md`
(talk figures) picks up from here.

---

## Headline results

**1. The band-ratio mechanism signal stands, with a corrected absolute calibration.**
The 12.7 µm/6.2 µm (neutral/ionized) ratio falls monotonically with stellar mass in all
three independent K-folds — the ionization-state mechanism signature. The 2026-07-03
envelope-aware re-derivation showed the original absolute values were window-envelope
contaminated (each feature group's amplitude absorbed the source-dimming envelope of its
own redshift window, a ×4.3–6.5 bias): the **intrinsic ratios are 13.4±3.8 → 1.96±0.52 →
0.88±0.06 → 0.19±0.14** (adjacent-bin separations 3.0/2.1/4.5σ). Use these for literature
comparisons; the flux-amplitude values (72.4 → 0.8) are the internal-trend version only.

**2. The L_PAH/L_IR mass slope disfavors the density/shattering channel — with an
estimator systematic.** Fold-ensemble slope +0.234 ± 0.077 (free α; survives the multi-band
normalization fix essentially unchanged vs the documented +0.236 ± 0.076). Confronted with
the two mass-axis channels derived from Narayanan+26 (which publishes no q_PAH(M\*) at
fixed z): 1.8σ above the density/shattering band [−0.35, +0.09], inside the enrichment/PZR
band [−0.10, +0.55]. Envelope-aware estimators keep the slope positive (+0.13…+0.23) but
weaken the density-chain tension to ~0.7σ — quote the estimator spread as a systematic
alongside α_wien. Key physics: f_mol is saturated (0.90–0.99) at cosmic noon over
logM\*=10–11.3, so the shattering mechanism has almost no f_mol lever arm on the mass axis.

**3. Evolution is required to fit the real stacks — and the branch-5 artifact is
explained.** With the new envelope-aware feature term, one shared amplitude slope buys
Δχ² = 253 pooled (44–65 per fold, all three independent folds agree): **η_A = +0.844 ±
0.026** — positive (feature/continuum rising with sSFR within each mass bin), far beyond a
scatter-null calibration (max spurious Δχ² ≈ 16, |η_A| ≈ 0.5 at matched χ²_red). The
branch-5 railed η_A = −2.37 is now understood: the constant-flux feature term let the ~10×
source-dimming envelope leak into the evolution slope (sim rung L2n reproduces the
mechanism, ~−0.85 dex on η_A). Claim to quote: "evolution is required to fit the stacks";
its physical decomposition still carries the α systematic (§6 tilt test).

## Machinery delivered (all guarded by tests; 10/10 evolution tests pass)

- `PAHSpectrumModel.fit_evolving_mcmc` — emcee over (η_A, η_g, log r-block) with per-bin
  (C_m, α_m) profiled analytically; `per_bin_ratios=True` flexibility knob;
  `evolving_flux_decomposition` + `plots.plot_pah_flux_decomposition` for the shaded
  feature-contribution overlays (sim and real-data versions).
- **`feature_envelope="baseline"`** in `fit_evolving`/`fit_evolving_mcmc` (passes through
  `fit_with_alpha`): features dim with the source via the reference band's cold baseline.
  Required on real observed-flux data.
- **Multi-band normalization fix** in `_evolving_data`: all bands share one per-bin scalar
  (per-band medians silently forced equal 24/70 continuum levels through the shared C_m).
  Single-band fits — all branch-6/7 headline paths — numerically unchanged.
- `TruthSpectrum.mir_plaw_amp` (hot/VSG MIR continuum; without it simulated f24 is pure
  PAH and C_m unidentifiable) and `TruthSpectrum.flux_envelope` (observed-flux dimming,
  calibrated to the real smoothed f24_cold(z, M\*)).

## Simulation findings (flexibility ladder, evolving-template MCMC notebook)

- Anchor the reference feature group on 7.7+8.6 µm or η_A floats; only per-group total
  slopes e_g = η_A + η_g are identifiable; 16.4+17.0 always rails (no bandpass leverage).
- Per-mass-bin ratio freedom (up to 21 dims) converges; weak-group posteriors inflate ~2×.
- The truth sits off-center in high-SNR corners because of the z_mid-vs-p(z) kernel
  approximation, not a code bug (posterior-predictive replica refits centered); upgrade
  path = integrate the sSFR modulation + envelope over p(z) inside the kernel (not needed
  at real-data SNR).
- 70 µm at its real depth (per-point SNR ~0.2) only bounds the 16.4+17.0 posterior; its
  degeneracy-breaking power needs depth.

## Notebooks (executed; under gitignored `notebooks/`, build scripts tracked)

| Notebook | Content |
|---|---|
| `2026-07-01-pah-forward-model-letter-candidates.ipynb` | Candidate questions; §4a band-ratio money plot |
| `2026-07-02-pah-narayanan-confrontation.ipynb` | Caveats 2a/2b closed; §6 confrontation money plot |
| `2026-07-02-pah-evolving-template-mcmc-simulation.ipynb` | Flexibility ladder L1–L4, §4c PPC diagnosis, §7 real-data static-vs-evolving + decomposition figures |
| `2026-07-03-pah-money-plots.ipynb` | Both money plots reproduced standalone; §2b/§3b envelope-aware re-derivations |

## Decisions and supersessions

- **No AGN-based interpretation** (user decision 2026-07-02; sample AGN fraction unknown).
- June-28 flat L_PAH/L_IR slope (+0.019) superseded (raw-baseline Tier-C artifact).
- §1a absolute band ratios superseded by the §2b envelope-aware values (trend unchanged).
- Branch-5 within-bin evolution "upper limit" framing superseded: with the envelope-aware
  term the evolution preference is real at the fit level; interpretation stays guarded.

## Open questions carried to branch 8+

- Envelope-aware η_A runs to ~2 in free-α fits (α_wien↔η_A degeneracy) — needs a joint
  treatment before η_A is quoted as physics.
- Evolve-in-kernel upgrade (integrate modulation over p(z)) for deep-SNR honesty.
- Tier A/B-only and bootstrap-over-sources cross-checks on the band-ratio fold errors.
- Narayanan-team contact for a sim band-ratio-vs-mass prediction (no-AGN framing).
