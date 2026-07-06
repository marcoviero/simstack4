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
