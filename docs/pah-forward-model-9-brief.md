# PAH Forward Model — Branch 9 Brief: stress tests and literature confrontation

**Goal.** The branch-7/8 results (intrinsic band-ratio mass trend, the L_PAH/L_IR crossing
pattern, the z-resolved branch sweep) are new and presentation-ready but not yet
defense-ready. This branch decides, test by test, which survive at what confidence — before
anything is quoted externally. Numbers may change here; that is the point.

Framing constraints carried forward: NO AGN-based interpretation (user decision, branch 7);
quote free-α as primary; pooled fits are centrals, fold scatter is the error.

---

## Objective 1 — the template systematic (the dominant unquantified error)

The §3c/§3d results convert single-window slice amplitudes to total L_PAH via the measured
per-mass-bin templates. That conversion is common-mode across folds, so the §3d fold errors
(±0.004 at z~1!) exclude it entirely.

- [ ] Propagate template uncertainty: per-fold templates (breaks the common mode), or sample
      template vectors from the §2b fold spread, or a joint (template + slice-amplitude) fit.
- [ ] Envelope-choice stress: features scaled by f_peak (greybody peak) instead of f24_cold —
      the physically motivated alternative; does the crossing pattern survive?
- [ ] Slice-boundary robustness: shift the z-slice edges ±0.2; 4-slice variant.
- [ ] Does the §3d sweep hold per fold (sign pattern in each fold separately)?
- [ ] **Smooth global model as an alternative to per-bin templates** (would eliminate the
      common-mode template-uncertainty problem above if it holds up): fit
      `log(L_PAH/L_IR) = a0 + a_z*z + a_M*logM* + a_Mz*z*logM*` -- a z x log M* interaction
      term, still linear-in-parameters, same GLS machinery as everything else here. An
      additive model (no interaction) has one fixed mass slope and structurally cannot
      reproduce the §3d sign flip (+0.602 -> +0.113 -> -1.088); the interaction term lets the
      effective mass slope (`a_M + a_Mz*z`) cross zero on its own. Note the three slice slopes
      aren't evenly spaced (drop rate roughly doubles from z~1->2 to z~2->3), so a pure linear
      interaction may still undershoot the z~3 point -- only add curvature (z^2 term) if that
      residual is real. New function, needs its own recovery test before touching real data.

## Objective 2 — literature confrontation (use §2b intrinsic values ONLY)

- [ ] **Absolute L_PAH/L_IR scale**: our 3.5–15% (partial 3-group template) vs Smith+07 SINGS
      total-PAH/L_TIR ~10–13% — build the partial→total template conversion and check we're
      not over/under by construction.
- [ ] **PAHSPECS (arXiv:2606.18244)**: direct read (still pending from branch 7); JWST/MIRI
      band ratios at z~1–3 are the closest external check on the 12.7/6.2 mass trend and the
      crossing pattern's z~2 slice.
- [ ] **Local band-ratio anchors**: Smith+07, Galliano+21; Whitcomb+24 calibrations. Mind the
      convention gap (our kernel-blended groups vs spectroscopic band fluxes) and the no-AGN
      framing (Xie & Ho ionization interpretation is fine; AGN-fraction claims are not).
- [ ] **Branch-band construction uncertainty**: the §3d z~1 slope (+0.60) sits just above
      branch B's ceiling (+0.55) — is that tension real or within our own band-derivation
      slack (MZR slope range, S_PZR range, G0 systematic)? Same for the z~3 overshoot below
      branch A (extrapolated from its z=2.5 row).

## Objective 3 — internal sanity checks (carried debts)

- [ ] Tier A/B-only refits of §2b and §3c (Tier C Eddington-bias check).
- [ ] Bootstrap-over-sources error as an independent cross-check on the fold-ensemble errors
      (3 folds → the scatter estimate itself is imprecise).
- [ ] Mass-bin edge shifts ±0.1 dex.
- [ ] α_wien ↔ η_A joint treatment (envelope-aware free-α fits rail η_A to ~2; the two are
      degenerate — profile jointly or put a physical prior on both before quoting either).
- [ ] Quantify the §1b ↔ §3c agreement (the two independent estimates of the within-bin
      z-trend) instead of calling it "matching" by eye.
- [ ] Scatter-limited χ²_red: repeat the branch-7 scatter-null calibration for the SLICE fits
      (fewer points per fit → more vulnerable to spurious structure).

## Deliverable

One stress-test notebook (build script per repo convention) ending in a verdict table:
result × test → pass / soft / fail, with the surviving numbers restated. The talk figure set
(branch-8 styling tasks, still open) should be regenerated only after this table exists.

---

## Outcome (2026-07-11) — combined stack added; paper scope narrowed to the crossing pattern

Deliverable exists: `notebooks/2026-07-10-pah-money-plots.ipynb` (tracked build script
`build_pah_money_plots_2026-07-10_notebook.py`), a clean rebuild of the hand-edited
`2026-07-07-pah-money-plots.ipynb`, ending in a §4 Verdict cell. Full detail in memory
`pah-forward-model-9-status`; summary here.

**What happened:**
- Widened the z-bins locally (z>2.6, where 6.2 µm enters) and the mass bins (bin0 →
  9.9–10.6, top bin edge → 11.0) to address the bin0 reference-amplitude instability.
- Added a third estimator — the **combined (non-K-fold) stack**: same catalog, not split,
  3 dither-offset runs, 3x the sources per (z, mass) cell of any one fold, no independent
  error bar of its own.
- **Objective 1, first bullet: DONE.** Self-consistent-per-fold on the *original* narrow
  bins found the z~1 slice unstable (pooled +0.602 → self-consistent −0.026, sign flip) —
  traced via the §3f exact-covariance diagnostic to bin0's A(6.2) SNR=−3.6, a
  near-zero-denominator ratio instability, not a parameter-covariance degeneracy. Widened
  bins fixed the z~1 *slope* stability (now +0.30/+0.35/+0.33 across
  pooled/self-consistent/combined) but **not** the raw K-fold-pooled A(6.2) SNR, still
  −4.17 after widening — only the combined stack resolves it (SNR=+8.19). Quote the
  combined-stack bin0 band ratio (1.6), not the K-fold-pooled one (6.4).
- **Objective 2's branch-band tension bullet: RESOLVED.** The z~1-above-branch-B-ceiling
  concern doesn't survive the stress test on the narrow bins (it wasn't a real measurement)
  and doesn't recur on the widened bins (+0.30ish sits comfortably inside branch B).
- **The all-z L_PAH/L_IR slope collapsed to zero** on the widened bins (fold-ensemble
  −0.004±0.038, pooled −0.006, combined +0.001 — all agree) from the documented +0.234±0.077.
  Not a contradiction of the crossing pattern — a slope running +0.3→0→−0.6/−0.7 across z
  integrates to ~0 under one global amplitude; this is what you'd expect if the crossing is
  real, not evidence against it.
- **The crossing pattern survives and strengthens**: three independent estimators agree in
  sign at every slice; correcting the wide bins' naive center to the N-weighted mean mass
  (bin3: 11.5→11.19) widens the swing ~25% rather than shrinking it; a nested model
  comparison (1 global slope vs. 3 per-slice slopes) gives Δχ²=328 for 4 extra params
  (ΔAIC=+320, ΔBIC=+318, F(4,6)=12.0, p=0.005) — **caveated**: fold-scatter errors are
  correlated/undersized (one field), so the qualitative call is robust (~6.4x uniform
  underestimate needed to flip it) but p=0.005 is not independently calibrated and should
  not be quoted externally as-is.
- **New open item**: η_A differs ~2x between K-fold pooled (+0.421±0.069) and combined
  (+0.860±0.041); combined matches the documented branch-7 headline (+0.844±0.026), pooled
  does not. Feeds the existing α_wien↔η_A joint-treatment item in Objective 3, now more
  urgent. Also noted: α_wien values this session (1.8–2.8) run systematically below the
  branch-6-documented ≈2.9–3.3, likely because that calibration predates the branch-7
  multi-band-normalization and feature-envelope fixes — `greybody.py`'s `_pah_coeffs` are
  very likely stale and this is a correctness issue, separate from the deferred error work.

**Paper scope decision:**
- **Headline: the z-resolved crossing pattern**, not the all-z Narayanan-confrontation
  slope (moves to a footnote explaining why it's flat).
- **Secondary: the band-ratio-vs-mass decline**, quoting the combined-stack lowest-mass-bin
  value, not the K-fold-pooled one.
- Combined-stack numbers are primary wherever the K-fold-pooled fit has a known defect
  (bin0 band ratio); three-way agreement is the robustness argument everywhere else.
- η_A evolution claim stays out of the headline pending pooled-vs-combined reconciliation.

**Still open**: envelope-choice stress and slice-boundary robustness (Objective 1's
remaining bullets — lower priority now given the combined-stack cross-check independently
corroborates the pattern), Smith+07 absolute-scale check (Objective 2), Tier A/B-only
refit, bootstrap-over-sources, mass-bin-edge shifts, and the α_wien↔η_A joint treatment
(Objective 3, more urgent given the η_A discrepancy above).

---

## Addendum (2026-07-11) — interpretation phase: the two-arms tests

Objective 2 moved from reading to testing. New deliverables:

- `docs/pah-interpretation-candidates.md` — ranked physical interpretations (C1
  metallicity-supply + destruction vs C2 shattering/density) for the crossing and the
  band ratio, the D1–D6 discriminating-test programme, tiered reading list, venue call
  (ApJL). PAHSPECS I+II read (five z≈1.1 galaxies, no mass slopes, no z~3 — **crossing
  novelty confirmed**); `docs/pah-refs.md` entry updated.
- `notebooks/build_two_arms_tests_2026-07-11_notebook.py` →
  `2026-07-11-two-arms-tests.ipynb` — D1–D6 executed. Headline outcomes: **D1/D5 kill the
  metallicity arm at both ends** (z~1 measured +0.357±0.071 vs step-function ceiling 0.00,
  +5.0σ; z~3 −0.874±0.122 vs a flat-to-positive track, −7.1σ); D2 rules out pure gas
  tracing (t_dep swing ≤0.10 vs measured −1.23); D3 inconclusive (SNR-limited above
  z=2.4); D4 first pass (A_pah rises with σ_SFR at fixed mass — consistent with η_A>0,
  below-threshold regime; doesn't separate mediators); **D6: no equilibrium
  scaling-relation arm pair fits — the crossing demands a threshold/nonlinear response**.
  Full numbers in the candidates doc's Results section.
- Letter framing consequence: the z~1 positive arm is unexplained by any published
  mean-relation mechanism; direct confrontation with the Narayanan+26 simulation outputs
  is the decisive next test.
