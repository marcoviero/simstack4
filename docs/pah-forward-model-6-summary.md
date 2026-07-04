# PAH Forward Model — Branch 6 Summary

**Goal.** Stop *assuming* the Wien-side warm-dust slope (α=2) and instead measure it, then
check whether every downstream PAH result (mass slope, EW slope, L_PAH/L_IR) survives having
α free instead of fixed. Push the validated result into the shared SED-fitting library
(`greybody.py`) so the actual per-population SED fit — not just a standalone notebook — carries
the correction.

See `pah-forward-model-6-brief.md` for the original design rationale;
`pah-forward-model-7-brief.md` (revised alongside this summary) picks up from here.

---

## Headline result — **the PAH deficit is strongest at low stellar mass, and this is α-robust**

**Wording correction (branch 7, 2026-07-01, advisor-flagged):** both ratios below *rise* with
M\* — more PAH per unit continuum/L_IR at high mass, i.e. the deficit is a *low-mass*
phenomenon that weakens toward high mass, not a trend that "rises with mass." Numbers/slopes
below are unchanged; see `docs/pah-forward-model-7-brief.md` §0.

Two independent ratios both rise with M\* and both survive the branch's own α uncertainty
range (α=2.0 pinned → α≈2.86-3.3 best-fit):

| Quantity | α=2.0 (pinned) | α=2.86 (best-fit) | Direction |
|---|---|---|---|
| PAH equivalent width, `A=α_m/C_m` (mass slope) | +0.351 ± 0.159 (fold) | +0.619 ± 0.033 (fold) | rises, both ≥2σ |
| **L_PAH/L_IR** (this branch, new) | **+0.146 dex/dex** | **+0.212 dex/dex** | rises, sign-stable, 1.5× swing |

This **overturns the 2026-06-28 "Letter 01" framing** ("PAH tracks L_IR, invariant of M\*",
flat L_PAH/L_IR ≈ 2.4%, `.claude/skills/pah-forward-model/letters/01-pah-tracks-lir-invariant.md`).
That result was measured at the *assumed* α=2 on a single catalog split with an unsmoothed
baseline; this branch's α=2.0-pinned re-measurement on the K-fold-pooled, smoothed-baseline
sample already gives a clearly rising +0.146 dex/dex, not flat +0.019. The old measurement
is superseded, not just α-corrected — **the discrepancy at matched α=2 is a methodology
difference (sample pooling / baseline smoothing / feature-ratio treatment), not yet isolated**.
Branch 7 should resolve which specific change drove it before quoting a final number (see
Open Questions below) — but the *rising* direction itself is now established across three
independent checks (EW slope at α=2 and α=2.86, and L_PAH/L_IR at α=2 and α=2.86), so the
Letter can proceed on that direction while the exact normalization is tightened.

User decision (2026-07-01): pivot the Letter headline to **"the PAH deficit rises with stellar
mass"**, with the EW slope as the most fold-tested supporting evidence and the new L_PAH/L_IR
slope as a second, α-checked line of evidence. Retire the "invariant with mass" framing.

---

## Wien slope α is not 2

`PAHSpectrumModel.fit_with_alpha` (24+70 µm, weak Gaussian prior at α=2.0/σ=0.3) on the pooled
K-fold z=0.5-5 sample gives **α=2.86±0.06** (fast re-tilt baseline) or **α=3.26±0.09** (exact
`greybody_model`-rebuilt baseline) — both far from the textbook 2, and both give essentially
the same fit quality (χ²_red 5.50 vs 5.51), so the ~0.4 spread between the two baseline
treatments is a real systematic, not a preference. Fold-ensemble error: **α=2.95±0.28** (3
independent K-fold refits, wider bounds than the library default to avoid an artificial
ceiling that initially railed 2/3 folds). z-cut robustness (z<2.5/3.5/5.0) gives statistically
identical α. **Correct citable claim**: "α significantly > 2 (direction bulletproof), exact
value 2.9-3.3 depending on baseline-construction method." Full derivation, all robustness
checks, and the two bugs that had to be fixed to trust this number:
`pah-forward-model-6-brief.md` execution log / project memory `pah-forward-model-6-status.md`.

**Consequence**: any absolute PAH/continuum amplitude from this pipeline must state its α —
moving α 2→2.86 inflates the raw amplitude 2-4×. L_PAH/L_IR is comparatively α-robust (1.5×
swing over the same α range, not 3-4×) — see the headline table above.

---

## New library capability — the correction now lives in `greybody.py`, not just a notebook

| Addition | What |
|---|---|
| `Greybody.alpha_wien` | Wien-side slope, was hardcoded `alpha=2.0` inside `greybody_model`'s default arg; now an overridable instance attribute (`greybody_model(..., alpha=None)` falls back to `self.alpha_wien`). Every existing call site is unaffected (all omit the arg). |
| `Greybody._pah_flux_lir` / `wien_mode="lir_pah"` | Predicts PAH flux from a measured `log10(L_PAH/L_IR) = a·logM* + d` relation — using the properly 8-1000 µm-integrated `calculate_LIR`, **not** the discredited f₂₄/f_peak proxy — converted to in-band flux via the real MIPS bandpass kernel (`feature_band_curves`). Evaluated **sharp-z** (population's actual z), matching how every other quantity in `greybody_model` is evaluated (deliberately *not* the dither-tomography's bin/photo-z-integrated kernel — see round-trip methodology below). |
| `SimstackResults` / `run_analysis_only(**kwargs)` | New passthrough kwargs: `alpha_wien`, `pah_lir_coeffs`, `pah_feature_groups`, `pah_r_ratios`, `pah_bands`. No `wrapper.py` changes needed — flows through the existing catch-all-kwargs path. |

Full test suite: **257/257 passing**, no regressions from either change.

### Round-trip validation — two real bugs found, one non-bug finding

1. **Amplitude placeholder bug.** An early check passed `amplitude=0.0` instead of each
   point's real fitted amplitude → `A=10⁰=1` instead of `A≈10⁻³⁵` → ~10³⁷% disagreement.
   Obvious once seen.
2. **`alpha_wien` state mismatch (subtler).** After fixing (1), a uniform ~18-21% residual
   remained at *every* point regardless of z or mass — the flatness (not the expected
   point-to-point scatter pattern) was the signature of a structural bug. Root cause: the
   notebook's independent `Greybody()` instance used both to derive `(a,d)` and to
   cross-check the library call still had default `alpha_wien=2.0`, while the library call
   under test used the fitted `alpha_wien≈2.86`. `calculate_LIR` integrates through the
   Wien-side power law, so one un-set attribute biased the whole derived relation by a
   near-constant factor. **Lesson for future work**: a uniform, z/mass-independent residual
   in a round-trip check is the signature of a silent state mismatch between two model
   instances, not real astrophysical scatter — check every non-default attribute is copied
   before chasing a normalization bug. Fixed → round-trip passes at exactly **0.0000%**
   (16/16 points, machine precision).
3. **Non-bug**: the *first* version of the check compared the library's sharp-z evaluation
   against the dither-tomography measurement's bin-integrated kernel and found 48-100%
   disagreement, worst at high z. Traced (not patched around) to two expected, physical
   differences: the true in-band response varies steeply with z (bin-integrating pulls in
   much more area than a point sample), and the dither fit's small (3%) catastrophic-outlier
   redshift pedestal (`f_cat=0.03`) leaks low-z-consistent flux into nominally high-z bins
   where the true response is genuinely ~0 (feature swept out of the 24 µm band by z~3.5-4).
   Confirmed numerically. The sharp-z convention is the physically correct choice for SED-fit
   injection; the round-trip check was rebuilt to test the library's documented formula
   against an independent computation of that *same* formula, not the definitionally
   different tomography estimate.

### Two-pass refit result (`wien_mode="lir_pah"`, α=2.86, redshift-dependent 24µm inflation)

24 µm inflation made **redshift-dependent** (4× at z<3 where the correction is active and
validated, 10000× above z=3 where the PAH bump has swept out of band — mirrors the existing
70 µm z=0.8 split).

- **Tier counts unchanged** (83 A / 95 B / 102 C, before and after) — tier is set by *median*
  SNR across all 8 bands, and 24 µm isn't usually the median-determining band even once
  un-suppressed, so this correction alone doesn't promote Tier C→B on this dataset. This
  **supersedes the more optimistic C→B promotion language** in the general PAH-correction
  recipe in `CLAUDE.md`, which predates this specific check.
- **T_dust does shift**: +0.81 K mean at z<3 (correction active) vs +0.43 K at z≥3 (PAH term
  is off there; that residual is from `alpha_wien=2.86` retilting the general Wien-side
  continuum for *all* bands, a separate, already-validated result — not the PAH correction
  itself). The z<3 shift being roughly double the z≥3 one is consistent with the PAH
  correction adding a real, modest effect on top of that baseline retilt.

---

## Deliverable

`notebooks/2026-07-01-pah-forward-model-letter.ipynb` (built by
`notebooks/build_pah_letter_notebook.py`) — 66 cells, 0 errors, all assertions pass. Sections:
data-consistency checks; α=2 vs α-free forward-model fit; fold-ensemble + z-cut + exact-baseline
robustness; literature context; model-overlay and deconvolved-pseudo-spectrum figures at the
best-fit α; intrinsic PAH line template by mass bin × redshift slice; the α-robust
L_PAH/L_IR(M*) derivation and its α-sensitivity check (9a-ii); round-trip validation; the
two-pass refit and resulting `plot_sed_grid` with the PAH correction live in the fit. Companion
living summary: `docs/pah-forward-model-6-letter-draft.html`.

---

## Open questions for branch 7

- **Isolate the old-vs-new L_PAH/L_IR discrepancy at matched α=2** (+0.019 flat, June 28,
  single split, unsmoothed baseline vs +0.146 rising, this branch, K-fold pooled, smoothed
  baseline). Candidates: catalog pooling (3 folds vs 1), `smooth_baseline`'s BIC-selected
  T(z,M*)/logA(z,M*) fit, or the feature-ratio (`r_g`) treatment. Needed before quoting a
  final normalization in the paper.
- **Fold-ensemble error on the L_PAH/L_IR mass slope** — 9a/9a-ii used the pooled sample only
  (point-to-point scatter, not a fold-ensemble error); redo with per-fold refits for an honest
  error bar, matching how the EW slope error is already handled.
- Update `.claude/skills/pah-forward-model/letters/` to reflect the pivoted headline (done
  alongside this summary — see `03-pah-deficit-rises-with-mass.md`, `01-...` marked superseded).
- σ_SFR cross-cut (2 mass × 3 σ_SFR bins) still open from branch 4/5 — would test whether the
  radiation-field mechanism (not halo mass) explanation holds at fixed M\*.
