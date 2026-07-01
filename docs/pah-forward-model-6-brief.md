# PAH Forward Model — Branch 6 Brief

**Goal**: Turn the branch-5 *upper limit* into a real measurement. Branch 5 built the
sSFR-evolution + Wien-slope machinery and showed that, on the z≤3.5 24-µm data, within-bin
evolution is scatter-limited and the absolute PAH amplitude is α-limited. Branch 6 applies
that machinery to the **z>4, multi-band (24+70[+100]) K-fold stacks** — where MIPS 70
re-measures the strong PAH features and the longer bands pin the continuum slope — to get a
defensible detection-or-upper-limit on evolution and an α-pinned A_pah(M*).

This is the science follow-through only; talk/referee-quality packaging is `pah-forward-model-7-brief.md`.

---

## Prerequisites from branch 5 (done)

- `PAHSpectrumModel.fit_evolving` (sSFR-anchored amplitude + line-ratio evolution),
  `fit_with_alpha` (Wien slope with a Gaussian prior), `eta_prior_sigma`.
- `pah_dither.fisher_evolution` / `evolution_recovery_sweep`; `dust_evolution.main_sequence_ssfr`.
- Findings that gate this branch (`docs/pah-forward-model-5-summary.md`): evolution is
  scatter-limited → upper limit; baseline is `(1+z)^(−α)` slope-2; A_pah is ±3–4× α-sensitive;
  headline L_PAH/L_IR ≈ flat, EW slope +0.37 (3.3σ).

---

## Objective 1 — Fix the z>4 dither binning

The z>4 accordion scheme (Δz 0.15→0.30→0.45 over z 0.5→5.0) must interleave properly:
**dither offsets scale with local Δz (≈ Δz_local/3)**, not a flat 0.05/0.10. Flat offsets on
the wide coarse bins make the staggered runs sample the same sparse high-z sources →
overlapping/redundant points (seen in the 2026-06-30 run). Regenerate the 3-run edge lists
(and don't force every run to end exactly at z=5.00). Verify with `n_sources` per bin.

## Objective 2 — Multi-band z>4 K-fold stacks

Stack z 0.5→5.0 in the fixed scheme against **24 + 70 (+ 100 if usable)**, K=3 disjoint SFG
folds. At z≈3–5 MIPS 70 sweeps rest 17→12 µm (16.4/17 at z≈3.1–3.3; 12.7 at z≈4.5; 11.3 at
z≈5.2); 24 (rest 4–6 µm) and 70/100 (rest 12–25 µm) bracket the warm continuum. Expect the
z>4 bins to be low-SNR individually — their value is the aggregate feature constraint.

## Objective 3 — Pin α (kill the ±3–4× systematic)

Run `fit_with_alpha` with 24+70(+100) and a strong prior (`alpha_prior=(2.0, 0.3)` or tighter)
so α is data-driven (70 µm is near-pure continuum → pins the slope) instead of assumed = 2.
Report α_wien ± (fold) error and the α-pinned A_pah(M*). Flag if α still rails (→ not
identifiable; keep the prior).

## Objective 4 — Evolution over the z≈1→5 baseline

Refit `fit_evolving` with the extended baseline: the 11.3/12.7/16.4 features that MIPS 24
measured at z≈0.4–1.1 are re-measured by MIPS 70 at z≈3–5. This (a) cross-checks the feature
ratios r_g with a different band, and (b) gives a long lever on η. Errors from the disjoint-fold
ensemble (never formal). Report **detection or honest upper limit** on η_A/η_g against the
`evolution_recovery_sweep` minimum-detectable η.

## Objective 5 — Deliverables

- α-pinned A_pah(M*) with an explicit α systematic.
- η_A/η_g: detection or upper limit, with the fold-ensemble error and the sensitivity floor.
- Updated `pah-forward-model-6-summary.md`; hand the trustworthy result to branch 7 for figures.

---

## Risks / notes

- **Source counts at z>4** (per K-fold) may be too low; check the `lp_zBEST` histogram (split
  in thirds) before committing the top bins — merge or cap at z≈4.5 if <~70–80 sources/bin.
- **70/100 µm at high z is faint** (cold peak > 250 µm at z~2); the extension is an aggregate
  constraint, not per-bin detections.
- Keep adding new functions, never replacing the working z≤3.5 path (project convention).
