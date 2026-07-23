# PAH Forward Model — Branch 11 Summary

**Goal.** Explain the high-mass (log M*>11) z≈2.5–5 (rest 4–6.8 µm) offset — the stacked
flux rides *above* the cold-dust + PAH model — by adding an explicit hot-dust/AGN continuum
component, and settle the long-running Wien-slope (α) ambiguity that was silently inflating
the PAH amplitude. Catalog masking of AGN was off the table (needs builder-side labels), so
the excess is *modeled*, not removed.

This branch is primarily **infrastructure + methodology**; the physical hot-MBB fit on the
residual is the continuing work (see NEXT). Started off `dc00596` on 2026-07-18.

---

## What merged (code + config)

**1. Drude line profiles for the PAH kernels** (`4e7e3f4`). `PAHSpectrumModel(profile="drude")`
switches `feature_band_curves`/`build_design_matrix` to Drude profiles (PAHFIT / Smith+2007);
Gaussian stays the default and is byte-identical. The Drude wings survive bandpass integration:
in-band peaks scale ×1.19–1.40 (group-dependent → `r_g` shifts ~10–15 %), an ~8–10 %-of-peak
wing floor leaks into the cold baseline, and `feature_profile_area` corrects the L_PAH
conversion — **Gaussian areas under-quote L_PAH by ×1.46** relative to the Drude convention all
literature (incl. the SINGS anchor and the LIM bridge) uses. Mass *slopes* are unaffected;
absolute L_PAH/L_IR moves to 13–23 % at z~1–2 (SINGS-consistent).

**2. PAHFIT-style hot-dust ladder** (`337151e`). `hot_ladder=(T1,T2,…)` + `hot_beta` adds
fixed-T hot-dust MBB rungs as non-negative **linear** columns in `fit_shared`/`fit_evolving`.
Because the temperatures are never fit, the rungs can't rail the way a free-T nonlinear hot-MBB
would; they dim with the source under `feature_envelope="baseline"`, and results carry
`hot_T`/`hot_amp`/`hot_amp_err`. `fit_lstsq`/`fit_mcmc`/`fit_evolving_mcmc` raise if a ladder is
set (the MCMC/window paths are degenerate with it). This **replaces** the originally-planned
free-T hot-MBB fit. First runs show the excess is *hotter* than PAHFIT's default ladder
(rest 4–7 µm ⇒ T~600–1400 K, i.e. AGN torus), so the working ladder moved to (600, 1000) K.

**3. Fisher-final dithered binning** (`335f129`, `0f02f06`). `z=0.2–6.0`, 3 staggered runs
(offsets 0/0.05/0.10, paired with the `split{0,1,2}of3` folds). Established with an
**honest-envelope** Fisher analysis (`fisher_evolution`/`fisher_for_scheme`) after the earlier
z→7.5 case was found to rest on a quadratic flux envelope that exploded above its z≤3.5 fit
range (f70 extrapolated to ~4500 mJy at z~7 vs a real ~0.7 mJy plateau). Honest conclusions:
- **`z_low 0.5→0.2` is the real win** — 16.4+17 group SNR +58 %, 12.7 +14 %, baseline-tilt
  (α-proxy) CRLB ×1.3–3.4 tighter from bright low-z anchors (SNR24 15–45/bin).
- Widening z>3.5 bins to Δz≈0.5–0.7 costs **zero** Fisher info and lifts per-bin SNR ~√2 to
  solid Tier B; the tail is placed so 70 µm rides **12.7 µm at z≈4.5** and **11.3 µm at z≈5.2**
  (24 µm on bare continuum there — the continuum-vs-feature split that breaks the
  amplitude/ratio degeneracy).
- **z>5 buys ~nothing** (statics saturate by z≈4.5; SNR70~1); 3.3 µm is undetectable (SNR~0.2).
  Split parquets cut at z=6.0 so no catalog rebuild.

**4. Top mass bin capped 11.0–11.5** (`0f02f06`, was 11.0–12.0). The 11.5–12.0 tail is where
the hot/AGN continuum and the sparsest counts live; dropping it keeps the cleanest PAH-amplitude
bin clean. Fisher penalty is ~10 % on the evolution-slope CRLBs (negligible); the 4-bin
`[9.9,10.6,10.9,11.5]` variant is marginally *tighter* than 5-bin.

---

## Key decisions (do not re-litigate)

**α is NOT cleanly measurable here — it is degenerate with the very MIR excess we model.** The
free clean-masked 24+70 fit rails to a stable ~3.0 (3.049 weighted vs 3.015 unweighted — not a
weighting bug; even masked, high-z 70 µm retains hot flux). **DECISION: fix α = 2.0** (physical
cold-dust Wien for β≈1.8) in the analysis; the free-α value is printed only as a diagnostic of
the contamination magnitude. The excess a free α would absorb now goes into the explicit hot
component. A_pah is Δα≈0.5 → ×3–4 sensitive, so this matters.

**FEATURE_GROUPS = `[[1,2],[0],[3,4],[5,6]]`** (7.7+8.6 reference | 6.2 | **11.3+12.7 merged** |
16.4+17). 11.3 alone railed (MCMC forces r>0 via 10^logr while WLS wanted r<0); merging it with
12.7 also dropped χ²_red (missing-11.3 was partly misspecification). The neutral/ionized band
ratio is therefore r[2] = (11.3+12.7)/(7.7+8.6) directly — 7.7+8.6 is the standard low-noise
ionized anchor.

**Baseline provenance is explicit.** `build_pah_spectrum_df`/`smooth_baseline` take a
**required** `alpha` (no silent default), train T(z,M*)/logA(z,M*) on Tier A/B ∩ SNR_FIR≥3
detections **unweighted** (SNR²-weighting was numerically unstable), and flag
`baseline_prior_dominated = max_snr_fir<3`. SED-QA by z: z<3.5 solid (Tier A/B, SNR 6–16);
3.5–5.5 marginal-real; z>5.5 prior-dominated.

**Trust Viero+22 for T, but do not cite high-z baseline/T as an independent result** — Viero's
steep T(z) is *assumed*, so quoting it back is circular. The MIR excess is measured *relative to*
that baseline. Prior only affects Tier B/C (z≳3.5); the QA tier/SNR table is data-driven.

---

## Science state at merge (in progress, not final)

Infrastructure is done and the new Fisher-final stacks exist. First Drude+ladder pass:
- **z~3 crossing slope is rock-solid** (−0.61…−0.93 across all three estimators, below both
  interpretation branch-bands) — the robust headline.
- z~1 positive end is now **estimator-dependent** (+0.14 pooled vs −0.17 combined) → reframed as
  a "high-z inversion" rather than a symmetric crossing.
- All-z slope is not significant (1.3–1.6σ; the old 3.1σ does not reproduce on the new data).
- Money-plot-1 band ratio compressed; bin-3 r(6.2)≈3.5 is suspected hot-continuum leakage into
  the 6.2 kernel — **do not quote until the (600,1000) K rerun**.

## NEXT (continues on interpretation / letter branches)

Notebook pass on the new stacks: swap RUN_DATES, run §2b/§3c/§3d with `profile="drude"` as the
systematic row and the hot ladder in the §2b fits (watch the free-α diagnostic drop toward 2 now
that the z<0.5 anchor exists), then the explicit hot-MBB fit on the log M*>11 z~2.5–4 residual
(T free ~few×100 K, α fixed 2, T_cold Viero; needs 70 µm). χ²≈5 persists until the hot component
is in. See `pah-forward-model-6-summary.md` (the old α≈2.9 was this excess contamination) and the
letter draft (headline is z<3.5-driven, should be prior/α-robust).
