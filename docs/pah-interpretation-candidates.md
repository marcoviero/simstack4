# Interpretation candidates and targeted reading list (branch 9, 2026-07-11)

Scope: the two letter results — (1) the z-resolved crossing of the L_PAH/L_IR mass slope
(+0.3 at z~1 → 0 at z~2 → −0.6/−0.7 at z~3) and (2) the 12.7/6.2 band-ratio decline with
M*. Framing constraints: NO AGN-based headline (branch-7 decision); combined-stack bin0
numbers; the nested-model p=0.005 is directional only.

## Where the letter sits in the literature

- **The z~1 positive slope is already published physics**: Shivaei+24 (SMILES, JWST/MIRI,
  443 non-AGN galaxies) find q_PAH *rising* with M* at z=0.7–2, driven by metallicity
  (q_PAH flat above ~0.5 Z☉, collapsing below). Our z~1 slice reproduces this with an
  independent method — that is the validation anchor, not the news.
- **No published measurement of L_PAH/L_IR vs M* at z~3 exists** (JWST censuses stop at
  z≈2; spectroscopy above that is few-object). No published claim of a sign inversion.
  **The crossing is the hook.** PAHSPECS read 2026-07-11 (both papers, see pah-refs.md):
  five galaxies at z≈1.1, no mass slopes, nothing at z~3 — novelty confirmed. Their
  integrated band ratios (ionized, small-weighted PAH mix vs local LIRGs) support our
  charge interpretation B1; their resolved UV-hardness trend pulls the other way and
  belongs in the B1-vs-B2 discussion.
- One-line physics framing that survives the candidate ranking below: *PAH abundance is
  chemistry-limited at low z and environment-limited at high z; the mass slope inverts
  because the controlling variable changes.*

## Ranked candidates — crossing pattern

**C1. Two-mechanism balance: metallicity-regulated abundance vs. destruction/dilution in
compact, intense star formation (leading interpretation).**
z~1 positive slope = the MZR through the q_PAH(Z) step function (Shivaei+17, Shivaei+24,
Engelbracht+05, Whitcomb+24). z~3 negative slope = massive z~3 galaxies are compact,
gas-rich, high-Σ_SFR systems where the PAH deficit seen locally in (U)LIRGs (PAH EW
anti-correlated with mid-IR compactness / IR8: Díaz-Santos+11, Elbaz+11, Stierwalt+14;
resolved high-z example: arXiv:2505.09728) takes over. Sharpening point for the letter:
at z~3 the MZR pushes the *other way* (the log M*~10 bin sits near/below the 0.5 Z☉
threshold), so the inversion happens *despite* metallicity — the destruction/dilution
term must beat it. Internal supporting evidence: the band-ratio trend (below) carries the
same ionization/processing fingerprint, which pure abundance changes would not produce.

**C2. Single-mechanism ISM-density regulation (Narayanan+26 shattering).**
q_PAH is set by grain-grain shattering, efficient only in diffuse gas: gas fraction
falling with M* pulls the slope positive; Σ_SFR rising with M* pulls it negative. At z~1
(low gas fractions) the first lever dominates; at z~3 (everything gas-rich, massive
galaxies extremely dense) the second wins. The crossing then falls out of one mechanism
with no fine-tuning — and is *extractable from their simulation outputs* at z=1/2/3,
which is the confrontation figure this branch already planned. Kin result: the tight
PAH–CO correlation to z~4 (arXiv:2409.05710; Cortzen+19) — if L_PAH tracks molecular gas,
L_PAH/L_IR ~ t_dep and the crossing maps onto the evolving t_dep(M*) relation.

**C3. Pure dilution (G₀ decoupling), no abundance change — disfavored internally.**
Massive high-z galaxies could simply add warm-dust L_IR (higher ⟨U⟩, higher T_dust)
without losing PAHs; Narayanan+26's own "q_PAH and L_PAH/L_FIR do not evolve in lockstep"
warning. Discriminant we already hold: pure dilution leaves band ratios unchanged; we
observe the 12.7/6.2 ratio moving with M*, so processing (charge and/or size) is
happening, not just dilution. Keep as the caveat that maps L_PAH/L_IR → q_PAH claims
through Draine & Li 2007 / Draine+21.

**C4. AGN contamination — excluded as headline, mandatory one-paragraph alternative.**
AGN fraction rises with both M* and z: torus continuum inflates L_IR and hard radiation
destroys small grains — qualitatively reproduces a z~3 negative slope. Mitigations to
state: NUVrJ star-forming selection; agreement of our z~1 slice with the explicitly
non-AGN Shivaei+24 sample; and check the sign of the Xie & Ho differential-suppression
prediction for 12.7/6.2 before using the band ratio as an AGN discriminant (their
pattern — neutral bands preserved, ionized suppressed — may actually predict the
*opposite* of our trend, which would be exculpatory; needs the direct read).

**C5. Technical/selection residuals.**
Template systematic (addressed: three estimators agree), bin0 A(6.2) SNR (addressed:
combined stack), Tier-C Eddington bias (open), α_wien↔η_A degeneracy (open), and the
per-z-slice rest-λ coverage difference (each slice leans on different features — the
per-mass-bin templates carry this; the envelope-choice stress test is the remaining
guard). Not interpretations, but the referee's first alternatives — cite the verdict
table.

## Separating C1 from C2 (added 2026-07-11)

They are not mutually exclusive — C2's shattering is a *production* mechanism, C1's
photo-destruction operates on whatever exists — and both are two-variable stories. The
question decomposes into two independent arms, each with its own discriminant:

**Positive arm (z~1): metallicity supply (C1) vs. diffuse-gas production (C2).**
Nearly all sign-level predictions are degenerate; the leverage is quantitative.

- **D1 — the plateau test (cheapest, do first).** Our mass range (log M* = 9.9–11.2) at
  z~1 sits entirely at/above the ~0.5 Z☉ threshold via the MZR (Sanders+21: even the
  lowest bin is ~0.7 Z☉ at z~1), i.e. ON the q_PAH(Z) plateau where Shivaei+24 find
  q_PAH ≈ const (~3.4%). The metallicity step function therefore predicts a ~FLAT mass
  slope at z~1 across our bins — we measure +0.30. Check Shivaei+24's q_PAH(M*) *within
  9.9–11.2 specifically* (their correlation may be carried by their log M* < 10 tail,
  below our range). If their trend is flat in our window, the z~1 slope already requires
  the C2 production arm (or a gradual q_PAH(Z) rise above threshold — quantify from
  their figure). Literature-only test, no new stacking.
- **D2 — gas-tracer normalization.** Under pure "PAH traces molecular gas"
  (PAH–CO), L_PAH/L_IR ∝ t_dep, whose Tacconi+18 mass slope is weak (bracket
  ≈ [−0.15, +0.09]: composite μ_gas−sSFR route vs their direct fit) and nearly
  z-independent — too shallow for +0.30 and structurally unable to flip. So even C2
  needs its density/shattering-efficiency term beyond gas mass; fitting the slice
  amplitudes against t_dep(M*,z) and reading the residual isolates that term.

**Negative arm (z~3): photo-destruction (C1) vs. density-suppressed production (C2).**

- **D3 — windowed band ratios (best internal test).** Destruction and charging are the
  same photons: if C1's arm drives the inversion, the population mix should trend with
  mass in lockstep with the amplitude slope; density-suppressed production removes
  molecules without processing the survivors (ratio trend flat while amplitude inverts).
  Coupling = destruction fingerprint; decoupling = production fingerprint.
  *Implementation constraint*: one broad band sees each feature at its own z (12.7 µm at
  z~0.9, 7.7+8.6 at z~1.6–2.6, 6.2 only at z>2.4), so a per-slice 12.7/6.2 is impossible.
  Instead split at z=2.1 (the 7.7 µm crossing anchors both sides) and fit each window's
  co-constrained pair: r(12.7/7.7+8.6) below, r(6.2/7.7+8.6) above; compare each window's
  ratio mass-slope to the same window's amplitude mass-slope. Implemented in
  `notebooks/2026-07-11-two-arms-tests.ipynb` §3.
- **D4 — mediator separation at fixed (M*, z).** Both arms predict deficit ~ Σ_SFR, but
  the causal chains differ: C1 runs through radiation (T_dust, IR8 as proxies — T_dust is
  free from every greybody fit), C2 through gas density (σ_SFR, t_dep proxies). Partial
  correlations of the per-bin amplitude residuals: does T_dust predict the deficit at
  fixed σ_SFR (C1), or σ_SFR at fixed T_dust (C2)? The run-2c σ_SFR cross-cut (2 mass ×
  3 σ_SFR, 3/4 runs done) is the data for this.
- **D5 — the metallicity-track violation.** If the low-mass z~3 amplitude sits *above*
  the q_PAH(Z(M*, z=3)) track (that bin is at/below threshold, so C1-supply predicts
  strong suppression), metallicity is demonstrably not the controller there → C2.

### Results (2026-07-11, `notebooks/2026-07-11-two-arms-tests.ipynb`, first execution)

Measured slice slopes (N-weighted mean-mass x, pooled K-fold with fold errors; combined
stack in parentheses): z~1 **+0.357±0.071** (+0.388), z~2 +0.073±0.052 (+0.004), z~3
**−0.874±0.122** (−0.702).

- **D1: DECISIVE against the metallicity step at z~1.** All four bins sit ON the q_PAH(Z)
  plateau at z~1 (12+log(O/H) = 8.69–8.92 vs thresholds 8.39/8.51) → predicted slope
  ≡ 0.00 across the full (γ, threshold, s_dec) bracket; measured +0.357 is **+5.0σ above
  the ceiling**. The supply arm cannot carry the z~1 positive slope. (Manual check still
  open: Shivaei+24's q_PAH(M*) within 9.9–11.2 specifically.)
- **D5: DECISIVE — Z is not the controller at z~3.** Even at z~3 our bins are at/above
  threshold (floor of the bracket ≈ 0.0, ceiling +0.15); measured −0.874 is **−7.1σ below
  the track**. The negative arm overwhelms a flat-to-opposing metallicity gradient.
- **D2: pure gas tracing ruled out.** t_dep mass slope ∈ [−0.13, +0.09], z~1→z~3 swing
  ≤ 0.10 vs measured swing −1.23.
- **D3: INCONCLUSIVE (SNR-limited).** High-window (z≥2.1) r(6.2/7.7+8.6) mass slope:
  combined +0.42 vs pooled −0.01±0.15 — estimators disagree; 6.2 µm leverage exists only
  at z>2.4. Low-window r(12.7/ref) slope is robustly negative (−0.27 combined,
  −0.14±0.04 pooled) — the band-ratio mass decline is confirmed at z<2.1 independently.
- **D4: first pass — does not separate the mediators yet.** At fixed mass, A_pah *rises*
  with σ_SFR (+1.2/+1.5 dex/dex) and T_dust rises alongside — both proxies move together.
  Direction is consistent with branch-7's η_A > 0 (sSFR boosts PAH/continuum on the main
  sequence), i.e. the sampled σ_SFR range sits *below* any destruction threshold. Two
  low-σ_SFR cells have unphysical negative amplitudes (unstable fits); COSMOS25 catalog,
  wide mass bins — a dedicated COSMOS2020 cross-cut is the follow-up.
- **D6: NO arm pair fits at face value** (best: PZR power-law + Σ_SFR, χ²=29.7 with
  coefficients 10–17×). All equilibrium arms have near-z-independent mass slopes because
  scaling-relation *exponents* barely evolve — only normalizations do (Σ_SFR of a massive
  MS galaxy rises ~30× from z~1→3). **The crossing therefore demands a threshold/nonlinear
  response** (locally: no deficit in normal disks, strong deficit in compact (U)LIRG
  cores) — at z~1 massive bins sit below threshold (positive arm rules), by z~3 they cross
  it (slope inverts). Both C1-destruction and C2-suppression admit threshold versions;
  D3/D4 remain the separators. Direct confrontation with the Narayanan+26 simulation
  *outputs* (full nonlinearity + scatter) is now the decisive external arbiter.

**Net after first execution**: the supply-side C1 arm is quantitatively dead at both ends
(D1, D5); C2 survives only in threshold/nonlinear form; the z~1 positive arm is the
unexplained piece (needs a mass-correlated driver still rising above half-solar Z). For
the letter this *sharpens* the novelty claim: no published mean-relation mechanism
reproduces the crossing at face value.

**External arbiter — D6, the Narayanan slope extraction (already planned).** Get
d q_PAH/d log M* at z = 1, 2, 3 from their sims (via Fig 9 + Tacconi/van der Wel
scalings). Quantitative reproduction of the crossing → C2 sufficient, C1's step emergent.
Failure to invert → C2 insufficient, destruction arm required. Also check whether their
z≳3 galaxies *deviate* from the PZR — the signature that Z is correlate, not cause.

**Structural note for the letter**: no single controlling variable (Z, t_dep, Σ_SFR) has
a mass slope that flips sign with z under standard scaling relations — the crossing
*forces* a two-term model with opposite pulls and different z-scalings. That is a strong,
assumption-light statement worth making explicitly; C1 and C2 are then the two candidate
identifications of the terms. A GLS regression of the 12 slice points on
{log Z, log t_dep, log Σ_SFR}(M*, z) with AIC model comparison (mind collinearity — all
predictors are smooth in M*, z; quote the condition number) operationalizes this; it is
the smooth-global-model item from the branch-9 brief with physical predictors.

If C2 dominates, the letter connects directly to the L_IR/T_dust program: q_PAH and
T_dust become two hands of one ISM-density variable (denser → warmer dust AND fewer
PAHs), unifying this result with Viero+22 and the DustEvolutionModel σ_SFR axis.

## The disentangling statement (2026-07-16): C1 vs C2, fleshed out per arm

Start-from-scratch framing for the consolidated notebook. Two facts to explain —
the crossing (+0.36 → 0 → −0.87) and the 12.7/6.2 band-ratio decline with M* — and
two candidate explanations, **each a two-arm story**:

- **C1 — two mechanisms, shifting balance.** PAH abundance is *chemistry-limited*
  where metallicity is low (supply arm: the q_PAH(Z) step, Shivaei+24/Whitcomb+24)
  and *radiation-limited* where star formation is compact and intense (destruction
  arm: the local (U)LIRG deficit, Díaz-Santos+11). z~1 positive slope = the MZR
  read through q_PAH(Z); z~3 negative slope = photo-destruction beating a
  metallicity gradient that points the other way. The crossing is the *balance of
  two mechanisms* shifting with z.
- **C2 — one mechanism, ISM density (Narayanan+26 shattering).** q_PAH is set by
  grain-grain shattering, efficient only in diffuse gas: gas fraction falling with
  M* pulls the slope positive; density (Σ_SFR) rising with M* pulls it negative.
  At z~1 (gas-poor) the first lever dominates; by z~3 everything is gas-rich and
  the massive galaxies are extremely dense, so suppressed *production* wins. The
  crossing falls out of one variable with no tuning — and is extractable from
  their simulation outputs.

The hypotheses pair off arm by arm, and each pairing has its own discriminant:

| | C1 says | C2 says | Discriminant |
|---|---|---|---|
| **+ arm (z~1)** | metallicity supply: slope = q_PAH(Z(M*)) through the MZR; our bins sit ON the plateau → predicts ~flat | diffuse-gas production: slope follows falling f_gas / shattering efficiency | D1 (plateau bracket), D2 (gas-tracing null), 1c (Shivaei in-window check) |
| **− arm (z~3)** | photo-destruction: survivors are *processed* → band-ratio mass trend coupled to the amplitude trend; mediator is radiation (T_dust) | suppressed production: molecules removed without processing → mix ~flat while amplitude inverts; mediator is density (σ_SFR) | D3 (fingerprint), D4 (mediator at fixed M*, z), 1a (fitted Σ_crit vs local deficit onset) |
| **the package** | two mechanisms whose balance must be tuned to cross | crossing emerges from one variable | D6 (arm decomposition), 1b (Narayanan slope extraction — the decider) |

Scoreboard after the D1–D6 first execution, read within this frame:

- **C1's supply arm as published fails quantitatively at both ends** (D1: measured
  +0.357 is +5.0σ above the step ceiling at z~1; D5: −0.874 is −7.1σ below the
  track at z~3). If C1 is right, its + arm needs more than the step — 1c tests
  whether Shivaei+24's own data allow an intra-plateau rise in our mass window.
- **D6: neither hypothesis reproduces the swing with equilibrium mean-relation
  arms** — both survive only in threshold/nonlinear form (locally true for the
  deficit anyway: normal disks show none, compact (U)LIRG cores do).
- **The − arm is the live C1-vs-C2 contest** (photo-destruction vs suppressed
  production), currently degenerate in Σ_SFR; D3/D4/1a/1b are the separators.
  D3 first pass: inconclusive above z=2.4 (SNR); D4 first pass: directional only.
- **The + arm is the open flank for both**: C1's candidate is dead as published,
  C2's is underpowered ~10× at face value (D6 coefficients).
- **1b is the overall decider**: sims reproduce the crossing → C2 sufficient (one
  mechanism, C1's step emergent within it); sims fail → C1's destruction arm is
  required on top of production physics (a C1-style hybrid).

Consolidated narrative notebook: `notebooks/2026-07-16-pah-crossing-two-arms.ipynb`
(build script `notebooks/build_pah_crossing_two_arms_2026-07-16_notebook.py`),
organized measurement → eliminations → the C1-vs-C2 contest per arm → verdict →
plan, superseding the test-ordered `2026-07-11-two-arms-tests.ipynb` as the
reading copy (that notebook remains the D1–D6 execution record).

## Ranked candidates — 12.7/6.2 decline with M*

**B1. Charge: PAHs in massive galaxies are more ionized.** 6.2 µm is a C–C mode enhanced
in cations; 12.7 µm is C–H out-of-plane, strongest in neutrals. Ionized fraction scales
~G₀√T/n_e, so denser + more intense star formation in massive galaxies raises it.
Coherent with C1/C2. Anchors: Draine & Li 2001/2007, Maragkoudakis+20 charge/size grids,
Egorov+25 (U anti-correlation), Leroy+23.

**B2. Size: smaller mean PAHs at high mass — in tension with the metallicity route.**
Whitcomb+24: *low* metallicity (→ low mass) shifts power to short-wavelength bands, i.e.
predicts long/short *rising* with M* — opposite to what we measure. This standing tension
(already in pah-refs.md) is why B1 (charge) outranks B2 (size); the letter should say so
explicitly rather than hide it. Cosmic-noon precedent that band ratios differ from local:
enhanced 11.3/3.3 in the MIRI LRS 37-galaxy survey (arXiv:2510.07365).

**B3. [NeII] 12.81 µm blending — a caveat that works in our favor.** The broadband kernel
cannot separate [NeII] from the 12.7 feature. [NeII] strengthens with SFR/mass, which
would push 12.7/6.2 *up* with M*; we observe it going down, so the contaminant dilutes
rather than creates the trend. One caveat sentence, not a section.

## Targeted reading list

Tier 1 — read before drafting (each changes what the letter can claim):

| Paper | Why |
|---|---|
| Shivaei+24, A&A (SMILES), arXiv:2402.07989 | q_PAH–M* at z=0.7–2, non-AGN; our z~1 anchor and the baseline the crossing departs from. Read their Fig. on q_PAH(M*,Z) closely. |
| Shivaei+17, ApJ 837, arXiv:1609.04814 | L_7.7/L_IR vs Z at z~2 from MIPS 24 stacking — the direct methodological predecessor; also their 24µm-SFR bias framing. |
| PAHSPECS, arXiv:2606.18230 (+18244) | READ 2026-07-11: five z≈1.1 galaxies, no mass slopes, no z~3 — novelty safe. Integrated ratios support B1 (charge); resolved UV-hardness trend supports B2-direction destruction. Cite both sides. |
| Díaz-Santos+11 (ApJ 741, 32) + Elbaz+11 (A&A 533, A119) + Stierwalt+14 (arXiv:1406.3891) | The compactness/IR8 PAH-deficit mechanism carrying the z~3 end of C1. |
| Narayanan+26, arXiv:2606.20809 (re-read §Fig 7/9) | Extract d q_PAH/d log M* at z=1/2/3 via Tacconi+18 + van der Wel+14 — the C2 confrontation figure. |

Tier 2 — interpretation depth:

| Paper | Why |
|---|---|
| Draine & Li 2007 (ApJ 657, 810); Draine+21 (ApJ 917, 3) | q_PAH formalism and U-scaling — required to state C3 correctly and convert L_PAH/L_IR → q_PAH. |
| Maragkoudakis+20 (MNRAS 494, 642) | Charge/size diagnostic grids for band pairs — puts B1 vs B2 on quantitative footing for 12.7/6.2 specifically. |
| Whitcomb+24, arXiv:2405.09685 | The opposite-direction size/metallicity prediction — the tension the letter must own. |
| Xie & Ho 2022, arXiv:2110.09705 | Direct read for the 12.7 differential-suppression sign (C4 discriminant). |
| arXiv:2510.07365 (MIRI LRS cosmic noon) | Evolving band ratios at cosmic noon — precedent that local templates don't hold at z~2. |
| Egorov+25, Leroy+23 (already in pah-refs) | Resolved U-driven destruction — mechanism citations for C1/B1. |

Tier 3 — context/discussion citations only: arXiv:2409.05710 + Cortzen+19 (PAH–CO,
gas-tracer framing), Spilker+23 Nature (3.3 µm at z=4.2 — PAHs exist in massive dusty
systems beyond our range), arXiv:2505.09728 (resolved high-z starburst-core deficit),
Smith+07 (absolute-scale check, already planned), Tielens 2008 (band physics),
arXiv:2506.13863 (PAH intensity mapping — methodological kin for tomographic stacking).

## Venue call

ApJL / A&A Letters. The crossing + band ratio + one robustness panel is a clean 4-figure
letter. Nature/Science requires a calibrated significance we cannot currently supply
(single field, fold-scatter errors, p=0.005 explicitly uncalibrated) — revisit only if
the Narayanan confrontation yields a spectacular quantitative match or failure AND a
second field or bootstrap-over-sources delivers a defensible p-value.
