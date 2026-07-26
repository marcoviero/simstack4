# PAH Interpretation — Branch 1 Brief: solve the two arms, then write the letter

**Goal.** Branch-9 established the measurement (the z-resolved crossing of the
L_PAH/L_IR mass slope: +0.36 → 0 → −0.87, three estimators agreeing) and the two-arms
tests established what does NOT explain it (`docs/pah-interpretation-candidates.md`,
Results section; `notebooks/2026-07-11-two-arms-tests.ipynb`): the metallicity step is
dead at both ends (±5–7σ), pure gas tracing is dead, and no equilibrium
scaling-relation arm pair fits at face value (D6). This branch's job is to determine —
or honestly bracket — what carries each arm, and assemble the letter.

**Priority order**: the two-arms resolution is Objective 1 and gets the effort. But
"solved" for letter purposes means *identified or defensibly bracketed*, not a complete
mechanism theory — the measurement is the letter's spine and stands on its own. One
measurement debt (α_wien↔η_A) gates quotable L_PAH numbers regardless of
interpretation, so it runs in parallel, not after.

Framing constraints carried forward: NO AGN-based headline; combined-stack bin0
numbers over K-fold-pooled; fold-scatter errors are for sign/ordering, the nested-model
p=0.005 is never quoted externally; N-weighted mean bin masses everywhere.

---

## Objective 1 — solve (or bracket) the two arms

The D6 finding sets the strategy: every equilibrium arm has a near-z-independent mass
slope, so the crossing requires either a **threshold/nonlinear response** or physics
outside the mean relations. Three attacks, in order of decisiveness-per-effort:

- [ ] **1a. Explicit threshold model (the "solving it" deliverable).** Extend the D6
      machinery from linear arms to a two-parameter threshold response:
      `log(L_PAH/L_IR) = P(M*) − D · max(0, log Σ_SFR(M*,z) − log Σ_crit)`, with the
      positive arm P and the threshold pair (Σ_crit, D) fit jointly to the 12
      (mass, z-slice) points. Questions it answers: does ONE threshold reproduce all
      three slice slopes? Is the fitted Σ_crit consistent with where the local PAH
      deficit turns on (compact (U)LIRG cores, Díaz-Santos+11)? Same GLS spirit as
      everything else; needs a recovery test on synthetic slopes before real data.
      A fitted, physically-sited Σ_crit is a letter-grade figure and the closest thing
      to "solved" this branch can produce internally.
- [ ] **1b. The Narayanan+26 arbiter (decisive external test).** Extract
      d q_PAH / d log M* at z = 1, 2, 3 from the simulation *outputs* (full
      nonlinearity + scatter, not our linearized reconstruction). Path: check for
      public data release / galaxy tables first; else digitize Fig 7/9 relations and
      sample their galaxy population; else email the authors (J.-D. Smith, Whitcomb
      are on both sides of this literature). If their sims produce the crossing, C2 is
      sufficient and the letter cites it as the mechanism; if they fail, the threshold
      destruction arm (1a) is required and the sims get a confrontation figure.
- [ ] **1c. The z~1 positive-arm identity check (cheap, do first).** Read Shivaei+24's
      q_PAH(M*) restricted to log M* = 9.9–11.2: if flat there, D1 is decisive as
      computed and the positive arm needs gas-structure physics; if rising, quantify
      the intra-plateau q_PAH(Z) slope their data allow and re-run D1 with it. An
      afternoon with their figures/tables; closes the last open caveat on D1.
- [ ] **1d. Internal consistency: within-bin vs across-bin.** Branch-7 measured
      η_A > 0 (PAH/continuum rises with sSFR *within* mass bins); at fixed z, sSFR
      falls with M*, which naively pulls the cross-bin slope negative — yet z~1
      measures +0.36. A single joint (M*, z, sSFR) amplitude model must reconcile
      both. This is the branch-9 smooth-global-model item (a0 + a_z·z + a_M·logM* +
      a_Mz·z·logM*) upgraded with the sSFR term; if the reconciliation fails, one of
      the two measurements is contaminated and the letter needs to know which.
- [ ] **1e. D3/D4 sharpening — decide, don't drift.** D3 (windowed band ratios) is
      SNR-limited above z=2.4: try combined-stack-only with a widened high window; if
      still inconclusive, it becomes a stated caveat, not a result. D4 needs a
      COSMOS2020 σ_SFR cross-cut with the widened bins to be more than directional —
      decide whether the stacking cost is justified BEFORE the letter, or defer to the
      follow-up paper and say so.

## Objective 2 — measurement debts that gate the letter (run in parallel)

- [ ] **2a. α_wien ↔ η_A joint treatment** (carried from branch 9, most urgent): the
      two are degenerate; η_A differs 2× between pooled (+0.42) and combined (+0.86),
      and α_wien drifted (1.8–2.8 vs documented 2.9–3.3). Profile jointly or put a
      physical prior on both. Every quoted L_PAH/L_IR inherits this systematic.
- [ ] **2b. Stale `_pah_coeffs` in `greybody.py`** (correctness): recalibrate from the
      current pipeline (post multi-band-normalization + feature-envelope fixes) or
      clearly mark as pre-branch-7 calibration.
- [ ] **2c. Remaining robustness row for the verdict table**: Tier A/B-only refit,
      bootstrap-over-sources error cross-check, mass-bin-edge shifts ±0.1 dex,
      envelope-choice stress (features scaled by f_peak instead of f24_cold),
      slice-boundary shifts ±0.2. None expected to move the sign; the letter's
      robustness paragraph quotes this table.

## Objective 3 — the letter

- [ ] Outline: measurement (crossing + band-ratio decline) → what it rules out
      (metallicity step ±5–7σ, gas tracing, all equilibrium mean-relation mechanisms)
      → what survives (threshold response; charge reading of the band ratio; Narayanan
      confrontation outcome from 1b) → implications and JWST-testable predictions.
- [ ] Figure set (≤4): crossing with three estimators; band ratio vs mass
      (combined-stack bin0); two-arms/threshold confrontation (1a/1b output); compact
      robustness/verdict panel.
- [ ] Venue: ApJL/A&A Letters (per candidates doc). Nature/Science pivot only if 1b
      yields a spectacular quantitative match or failure AND a calibrated significance
      exists (bootstrap-over-sources or second field) — both, not either.
- [ ] Discussion must own, explicitly: the Whitcomb+24 opposite-direction size
      prediction; the PAHSPECS resolved-vs-integrated two-arrow tension; the [NeII]
      blending caveat (works in our favor); the AGN paragraph (selection + Shivaei+24
      agreement); the α_wien systematic on absolute L_PAH.

## Deliverable

A threshold-model notebook (build script per convention, recovery test first) and an
updated verdict table folding in 1a–1e; then the letter draft. The letter is the
branch's exit criterion — if the two arms resist identification after 1a–1c, the
bracketed version ships with the threshold requirement itself as the interpretive
claim ("no equilibrium mechanism reproduces this; the response must be nonlinear"),
which is already a publishable statement.
