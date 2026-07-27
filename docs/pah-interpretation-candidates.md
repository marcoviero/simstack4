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

---

## D4 feasibility — measured 2026-07-26 (before building anything)

**The lever exists.** `notebooks/build_d4_lever_check.py` on
`cosmos25_stacking_20260317_201727` (5 z × 4 mass × 5 σ_SFR): projecting
(z, log M*) out of both mediators leaves partial corr(T_dust, log σ_SFR | z, logM)
= **+0.12 to +0.19** — only 1–4 % shared variance. The two are *separable*; the
worry that they are collinear on the main sequence is not supported. Independent
T_dust lever 5.1–5.6 K vs a 2.2–2.3 K median error → **S/N ≈ 2.3**.

**Why the first pass was inconclusive — probably cell count, not collinearity.**
The PAH amplitude comes from the tomographic z-sweep, so it is measured per
*(mass, σ_SFR)* cell, not per (z, mass, σ_SFR) cell. A 2×3 grid gives **6
amplitudes**; a 3×3 grid gives 9. A partial correlation over 6–9 points has no
power. The fix is the z-sliced amplitude machinery the crossing already uses
(`zslice_ratios`), which lifts this to 18–27 cells.

**COSMOS2020 cannot do D4 at all**: its PAH catalogs carry 11–13 columns and no
size/Sérsic information, so Σ_SFR = SFR/(2πR_e²) is not computable. COSMOSWeb has
`radius_sersic` and a ready-made `log_sigma_sfr`. D4 is a COSMOS25 experiment.

**The existing 4 dithered June-15 runs are nearly right but mis-binned.**
`20260614_203609 / 220935 / 230839 / 20260615_000803` (21 z-bins × 2 mass × 3
σ_SFR, ¼-bin offsets) give excellent tomography — 75–83 z-points per (mass,σ)
cell, T_dust median error 1.69 K, 472/489 cells passing physicality+error cuts.
But their σ_SFR edges `[-2.5, -1.25, 0.5, 2.5]` split the sample **29 / 67 / 4 %**:
two-thirds in one bin, 4 % in the top bin (median 50 sources/cell). That washes
out the σ_SFR contrast and drops the lever to **S/N 1.62**.

**Recommended re-stack** (COSMOSWeb, `COSMOSWeb_stacking_catalog_all_sf_qt`,
84 892 usable in z = 0.2–6, logM = 9–11.5). Terciles lift the binding
(top-mass × top-σ) cell from 27 → 67 sources; a 4th mass bin collapses it to 13:

```toml
[catalog.classification.binning.redshift]
bins = [0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1, 1.25, 1.4,
        1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.9, 3.2, 3.5, 4.2, 5.0, 6.0]
[catalog.classification.binning.stellar_mass]
bins = [9.0, 10.0, 10.7, 11.5]
[catalog.classification.binning.sigma_sfr]     # catalog terciles — the key fix
bins = [-9.0, -1.07, -0.32, 9.0]
```

**Expect a marginal result**: lever S/N ≈ 2.3, ~18–27 correlation cells, ~42 % of
cells under 100 sources. Worth running — it is the test that separates destruction
from suppressed production — but plan for "suggestive, not decisive".

**GOTCHA that nearly inverted the verdict.** The fit-quality TIER grades band SNR,
not whether the SED fit converged. One cell in the 20260317 grid has T_dust = 140 K
with a **1514 K** error bar and is labelled **Tier A**. Left in, it alone doubles
the apparent lever (S/N 5.3 vs the true 2.3). Anything consuming T_dust needs an
explicit physicality + error cut; Tier A/B filtering does not protect against
railed or unconstrained temperature fits.

## D4 first execution on the 20260726 re-stack — NOT answerable, needs dithering

`cosmos25_stacking_20260726_105516` (COSMOSWeb sersic, 20 graded z-bins, mass
[9.0, 10.0, 10.7, 11.5], σ_SFR at catalog terciles [-9, -0.96, -0.27, 9], SINGLE
un-dithered run). Nuisance mass bin (9.0–10.0) excluded per convention → 91 rows,
6 (mass, σ_SFR) cells, 13–16 z-points each.

**The ratio block cannot be fit here.** Free group ratios rail: r(6.2) = −0.83
overall, r(11.3+12.7) = **+28.6** in one σ bin, r(6.2) = **+6.35** in one mass bin.
This is *not* a model problem — the identical model on COSMOS2020 gives
r(6.2) = +0.56 to +1.14, well determined. Nor is it the low-z anchor: cutting
COSMOS2020 to z ≥ 0.57 (this stack's floor) leaves r(6.2) = +1.04 unchanged. It is
simply too little data spread over too many cells.

**Fix: fix the shape, fit only the amplitude.** D4 needs α per cell, not the
template. `notebooks/build_d4_amplitudes.py` bakes the COSMOS2020-measured ratios
into the feature strengths and welds all features into one group, so r ≡ 1 with
nothing to fit and each cell contributes only (C_m, α_m). That fit is clean:
χ²_red = 4.88, τ_sil = 0.45, all six amplitudes positive.

**Result: inconclusive.** partial corr(logA, T_dust | logM, logσ) = −0.694 (C1
radiation); partial corr(logA, logσ | logM, T_dust) = +0.699 (C2 density). With
N = 6 and two controls, dof = 2 and the critical |r| is **≈0.95** — neither is
close. The two partials being near-mirror-images while the raw correlations are
tiny (−0.069, +0.134) is the signature of two correlated predictors splitting
noise, and the amplitudes run non-monotonically in σ_SFR in *opposite* directions
in the two mass bins.

**z-slicing to 18 cells makes it worse, not better.** Window solves get only 3–6
points: 6 of 16 cells return negative amplitudes and the positives span 0.015 to
23.8. A partial correlation computed on the 10 positive cells returns "significant"
(−0.85) but that is **selection on the dependent variable** and must not be quoted.

**What is actually needed: DITHER OFFSETS.** The advice to use a single run was
wrong for this application. Fisher/CRLB prices the *global* fit and correctly says
oversampling is near information-neutral there — but it says nothing about whether
each z-window has enough points to solve. The June-15 runs (4 offsets) had 75–83
z-points per (mass,σ) cell; this one has 15, so each window gets 3–6. Re-run the
same bins with **3–4 offsets** (stagger the z edges by ⅓–¼ of the local bin width)
→ 45–60 z-points per cell, 15–20 per window, dof ≈ 12 and a critical |r| of ~0.55.

**Rule to carry:** when an analysis solves inside sub-windows, Fisher on the global
fit is the wrong figure of merit. Count points per window.

## D4 EXECUTED — inconclusive, and structurally so (2026-07-26)

Run on the three dithered COMBINED stacks (`20260726_141318 / 152037 / 161906`,
COSMOSWeb sersic, 20 graded z-bins × 3 mass × 3 σ_SFR at catalog terciles, offsets
0 / ⅓ / ⅔ of the local bin width). `population_class == 0` = sfg_keep; the
9.0–10.0 nuisance mass bin excluded; cells with `n_sources < 20` cut (that cut
removes the 16 % of cells whose MIPS-24 SNR median is 1.24, including 9 % with
*negative* 24 µm flux).

**The whole-range fit is good.** Template shape fixed to the COSMOS2020-measured
group ratios (all features welded into one group, so r ≡ 1 and only (C_m, α_m) are
free): χ²_red = 2.98, τ_sil = 0.17, all six amplitudes positive —
A = 2.73, 1.06, 3.22 (logM ≈ 10.25, σ_SFR low→high) and 1.03, 2.77, 3.01
(logM ≈ 10.90). But six cells with two controls is **dof = 2**, where the critical
|r| is 0.95. Partials: **+0.795** (C1 radiation, T_dust) and **−0.579** (C2 density,
σ_SFR). Leaning C1, nowhere near significance, not quotable.

**The z-sliced route to 18 cells still fails.** The dithering did exactly what it
was designed to do — 5–18 points per (cell, z-window) instead of the 3–6 that broke
the un-dithered attempt — and the window solves were *still* degenerate: 6 of 17
cells returned negative amplitudes, the positives spanned 0.54 → 13.5, and single
cells swung −0.92 → +9.77 between adjacent z-slices.

**Why, and this is the transferable part: z-slicing the amplitude is
self-defeating for a tomographic measurement.** The PAH amplitude is identified *by*
the bandpass sweeping rest wavelength as z varies. Inside a narrow z-window the
feature template and the cold baseline are nearly parallel, so the two-parameter
(C, α) solve is degenerate however many points it contains. **More points do not fix
collinearity.** The COSMOS2020 crossing survives the same slicing only because its
cells are ~4× richer (no σ_SFR split, 4 mass bins), leaving residual leverage.

**Net: D4 needs many cells AND well-determined per-cell amplitudes, and those fight
— cells come from splitting, amplitudes come from not splitting.** With ~34 000
sfg_keep galaxies you cannot have both. This is the same conclusion D3 reached by a
different route: the mediator question is not answerable with existing MIR data.

**What still stands from the feasibility work:** the mediators *are* separable —
partial corr(T_dust, log σ_SFR | z, logM) = +0.12 to +0.19, only 1–4 % shared
variance. The obstacle is cell count and amplitude precision, not collinearity
between the mediators.

**Do not re-attempt without new data.** Neither finer binning nor more dither
offsets will help — both were tried. What would help: a deeper MIR sample (more
sfg_keep galaxies per cell), or a direct per-galaxy PAH measurement that removes
the need to split at all.

## Threshold-model fit — the destruction hypothesis TESTED, and it fails (2026-07-26)

D6 concluded the crossing "demands a threshold/nonlinear response". That was never
fitted, only asserted. It has now been fitted, to the 12 (4 mass × 3 z) pooled
L_PAH/L_IR points with fold errors, and **the natural implementation does not work.**

**Model.** log(L_PAH/L_IR) = a₀ + a₁(logM\*−10.5) − d·f(Σ_SFR), i.e. a z-independent
production slope plus a destruction term triggered above a Σ_SFR threshold, with
Σ_SFR(M\*,z) from Speagle+14 × van der Wel+14 (the same relations as §3g).

**Model comparison (12 points).** Hard threshold: χ² = 77.5, χ²_red = 9.7, AIC = 85.5,
versus production-only (AIC 350.2) and production + free normalisation per z
(AIC 130.6, same parameter count). **ΔAIC = −45 — the threshold wins.** But it wins
while failing:

| z | measured slope | hard threshold | soft (scatter) threshold |
|---|---|---|---|
| 0.95 | +0.380 | +0.184 | +0.176 |
| 1.90 | −0.009 | −0.069 | +0.058 |
| **2.95** | **−0.699** | **−0.096** | −0.503 |

**Why it fails — Σ_SFR is too flat in mass.** d log Σ_SFR/d log M\* is 0.242 / 0.310 /
0.343 at z = 0.95 / 1.90 / 2.95: it grows only **1.11×** from z=1.9 to 3.0. Taking the
z~1 slope as the destruction-free production value (+0.380), the destruction term must
supply a tilt of 0.000 / −0.389 / **−1.079** — a growth of **2.8×**. A Σ_SFR trigger
falls short by **2.5×**. sSFR is worse (its gradient *shrinks*, 0.87×).

**Adding scatter (threshold → sigmoid) helps but is degenerate.** The mass gradient of
the above-threshold fraction peaks where the population crosses Σ_crit, which is the
right mechanism, and it improves χ²_red 9.7 → 5.4 and the z~3 slope −0.10 → −0.50.
But the destruction depth **rails at every bound tested** (0.5, 1.0, 1.5, 2.0, 3.0,
12.0 dex), χ² falling monotonically as the bound loosens: only the *product*
d × P(Σ>Σ_crit) is constrained. At d ≤ 1.0 it reproduces the z~3 slope (−0.647) but
at χ²_red 7.6, i.e. by breaking the other two slices. χ²_red never falls below 5.4.

**SPECIFICATION for any viable trigger X(M\*,z)** — the useful output of this exercise:
its **mass gradient must grow ≈2.8× between z=2 and z=3**. No mean scaling relation
built from the main sequence plus sizes has that property, because their exponents are
near-constant and only normalisations evolve — the D6 result, now sharpened from "power
laws fail" to "thresholds in Σ_SFR fail too, and here is the quantitative requirement".

**So destruction is not ruled out — the Σ_SFR-triggered implementation of it is.**
What survives needs either a trigger whose *mass dependence itself* evolves (radiation
hardness? gas-phase geometry? merger fraction?), or a genuinely non-mean-relation
effect (the massive z~3 population being a different galaxy mix, not the same galaxies
shifted along a scaling relation).

Reproduce: `notebooks/build_threshold_fit.py`.

## Direct Σ_SFR test + the G₀ hypothesis (2026-07-27) — the thread for branch 3

**Direct test of the destruction trigger.** The 2-σ_SFR-bin dithered COMBINED stacks
(`20260726_210408 / 203157 / 192719`, split at log Σ_SFR = −0.75) let us ask the
threshold question *directly* rather than inferring Σ_crit from the (M\*, z) pattern:
at fixed mass and z, is the PAH amplitude lower in the high-Σ_SFR half?

**It is not.** The raw 24 µm excess *before any fitting* rises with Σ_SFR in both mass
bins: f24/f24_cold = 1.01 → 1.64 (logM ≈ 10.28) and 0.96 → 2.13 (logM ≈ 10.91). Fitted
amplitudes: A = 2.25 → 1.91 and 0.29 → 3.07. This is the **third** independent statement
of the same thing (D4 first pass +1.2/+1.5 dex/dex; the 3-σ-bin run; this), and it is
the opposite sign to destruction.

*Caveat*: the low-σ cells have Tier-A fractions of 0.13–0.16 vs 0.38–0.51 for high-σ,
so their `f24_cold` baseline is prior-dominated exactly where the excess vanishes — the
same ambiguity as the 9.0–9.9 nuisance bin. And the split at −0.75 sits *below* the
Σ_crit ≈ −0.30 the fit wanted, so the upper bin straddles rather than clears the
threshold. **This constrains destruction below Σ_SFR ≈ 1 M☉/yr/kpc²; it does not
exclude a threshold above the range COSMOSWeb's main-sequence sample reaches.**

### The reframing this forces: are we measuring q_PAH at all?

Narayanan+26's density term is **suppressed shattering** (a *production* channel), so
their q_PAH *falls* with Σ_SFR — the opposite sign to what we measure. But they state
explicitly that *"the physical q_PAH and the observed L_PAH/L_FIR do **not** evolve in
lockstep"*, because L_PAH/M_PAH ∝ G₀ while q_PAH anti-correlates with Σ_SFR. So

  **L_PAH/L_IR ∝ q_PAH × (G₀ / ⟨U⟩)**

and our rising L_PAH/L_IR is *consistent* with their falling q_PAH provided the G₀ boost
wins. **We have been fitting abundance mechanisms (metallicity supply, shattering
suppression, destruction thresholds) to an observable that may be radiation-dominated.**

This matters because G₀/⟨U⟩ is a **geometry** factor — how concentrated young stars are
relative to the bulk dust — not a mean scaling relation. It therefore **escapes the D6
argument entirely**: D6's force is that scaling-relation *exponents* barely evolve, so
nothing built from the MS + sizes can flip a mass slope. A radiation-geometry term is
not so constrained.

### Evidence so far — promising, NOT established

Mass slope of the ionised (7.7+8.6) vs neutral (11.3+12.7) luminosity per L_IR, z<2
(the only window where the neutral group has leverage, §1b):

| variant | pooled ion / neu | combined ion / neu |
|---|---|---|
| 3 groups (6.2 railed) | +0.224 / −0.040 | +0.296 / −0.048 |
| 6.2 dropped, z<2 | +0.168 / −0.059 | **+0.070 / −0.175** |
| 6.2 dropped, z<1.6 | +0.296 / +0.043 | +0.033 / −0.098 |
| 6.2 dropped, z<2.4 | **+0.257 / +0.010** | +0.137 / −0.116 |

**Pooled supports G₀** — the trend lives in the ionised bands, neutral flat, stable
across every window and grouping — and it reproduces §3h's independent inference that
d log(α·r_neutral)/d log M\* ≈ 0. **Combined inverts** once 6.2 is dropped (neutral slope
exceeds ionised). The clean-looking 3-group agreement (−0.040 / −0.048) is suspect:
r(6.2) railed to −11…−60 in those fits, and a railing column absorbs its neighbours'
flux. **Two estimators disagree — this is not yet a result.**

**→ Chased explicitly in `docs/pah-interpretation-3-brief.md`.**

## Step 0-bis executed: ⟨U⟩ from T_dust, and the search for a G₀ proxy (2026-07-27)

`notebooks/build_qpah_radiation.py` (supersedes `build_qpah_backout.py`). Four results,
in increasing order of how much they change the branch.

### 1. "T_dust correction only" is not a null assumption — it is g = 0

Write the decoupling as a power law, `G₀ ∝ ⟨U⟩^g`, so that

    log q_PAH = log(L_PAH/L_IR) + (1 − g) · p · log T_dust ,   p = 4 + β = 5.8

`g = 1` is fixed geometry: the field terms cancel exactly and there is **no correction at
all**. `g = 0` — what the first pass called "T_dust only" — asserts the PAH-heating field
is constant across 1 dex of M\* and z = 0.5–3.5, the most extreme decoupling available.
The first pass presented the maximal correction as the conservative one.

| g | z~1 | z~2 | z~3 | swing | crossing |
|---|---|---|---|---|---|
| 1.00 (fixed geometry) | +0.380 | −0.009 | −0.699 | −1.079 | YES |
| 0.50 | +0.217 | −0.149 | −0.719 | −0.935 | YES |
| 0.00 (max decoupling) | +0.053 | −0.289 | −0.739 | −0.791 | YES |

The **crossing is present at every g**. Only the z~1 positive arm is at stake.

### 2. The first-pass z~1 headline is estimator-dependent, and the fold errors are large

The committed table (z~1 → −0.010, swing −0.644) reproduces **exactly** with a plain
median of T within each (M\*, z) cell. With the source-weighted mean of the same cells it
is +0.053 / −0.791; with `T_dust_smooth`, +0.138 / −1.035. The physicality cut does
nothing (every cell already has 15 < T < 45 K).

Disjoint-fold ensemble (T *and* L_PAH/L_IR re-measured in each fold), at g = 0:

| estimator | z~1 | z~2 | z~3 | swing |
|---|---|---|---|---|
| weighted mean | +0.026 ± 0.023 | −0.316 ± 0.086 | −0.777 ± 0.169 | −0.803 ± 0.173 (4.6σ) |
| median | −0.008 ± 0.039 | −0.324 ± 0.115 | −0.693 ± 0.191 | −0.685 ± 0.203 (3.4σ) |

So the brief's worry is **confirmed in one direction and refuted in the other**: at g = 0
the z~1 positive arm *is* erased (≤1.1σ from zero under either estimator), but the swing
is **not** — it survives at 3.4–4.6σ. The correction converts a sign change into a
monotonic decline that steepens with redshift. It does not remove the phenomenon.
β sensitivity is negligible (swing −0.806/−0.791/−0.781 for β = 1.5/1.8/2.0); the fold
scatter on T is ≤1.5 K, and 1 K at 30 K is 0.083 dex in ⟨U⟩.

### 3. Viero+2013 Eq 19 — an external T(M\*,z) contributes exactly zero swing

Viero+2013 (ApJ 779, 32) Eq 19–21 fit `T = A_z (log M\*)^{α_T}` with
`A_z = A₀ + A₁(1+z)^{A₂}`, `A = [−439.83, 578.93, 0.11]`, and
`α_T = α₀ + α₁(1+z)^{α₂}`, `α = [−0.81, 2.84×10⁻⁵, 3.55]` (their β = 2, ours 1.8).

Because the fitted `α₁ = 2.8×10⁻⁵`, **their mass gradient is z-independent to four
decimals** — the data preferred no evolution of the T–M\* slope. Substituting it shifts
all three slopes by the same −0.19 dex/dex and changes the swing by **exactly zero**.

| d log T / d log M\* | z~1 | z~2 | z~3 | (z3 − z1) |
|---|---|---|---|---|
| ours, weighted mean | −0.057 | −0.048 | −0.007 | **+0.050** |
| ours, median | −0.067 | −0.040 | +0.008 | **+0.075** |
| Viero+2013 Eq 19–21 | −0.033 | −0.033 | −0.033 | **+0.0001** |

Every bit of swing the ⟨U⟩ correction buys comes from *our own fits'* z-evolving T–M\*
gradient. **It is not a prior-pull artifact** — restricting to Tier A only, or to
`max_snr_fir > 10`, leaves it intact and slightly stronger (+0.057, +0.070), even though
it is the *low*-mass z~3 cells that are the weakest (Tier-A fraction 0.50, SNR 8.5, vs
1.00 / 12.4 for the top mass bin). It is a genuine measurement, and a genuine departure
from Viero+2013 on a mass-selected Herschel stack of the same kind. That the T–M\*
anti-correlation *vanishes by z~3* is a result in its own right — but it is also the
single unverified link in the whole correction, so quote it as such.

### 4. The inversion — no power-law G₀ proxy can work either

Turn it around. If q_PAH's mass gradient does not itself evolve (the D1/D6 result: the
metallicity-supply arm gives a near-constant gradient), the entire swing must come from
the radiation term, so `d log(G₀/⟨U⟩)/d log M\*` must equal the observed slopes, and

    required d log G₀ / d log M\*  =  +0.053 / −0.289 / −0.739 ,  change −0.791

A proxy X with `G₀ ∝ X^b` can only supply that if **its own mass gradient evolves**:

| proxy | z~1 | z~2 | z~3 | d(slope) | b required |
|---|---|---|---|---|---|
| log Σ_SFR (branch-2, COSMOSWeb sizes) | +0.242 | +0.310 | +0.343 | +0.101 | **−7.8** |
| log sSFR (measured) | −1.057 | −1.176 | −1.121 | −0.064 | **+12.5** |
| log sSFR_MS(z, M\*) | −0.318 | −0.250 | −0.217 | +0.101 | −7.9 |
| log Δ_MS | −0.740 | −0.926 | −0.904 | −0.164 | +4.8 |
| log T_dust | −0.056 | −0.048 | −0.007 | +0.050 | −15.9 |

Σ_SFR needs `b = −7.8` — **negative**, i.e. G₀ would have to *fall* as star formation
concentrates. sSFR needs `b = +12.5`. Both are unphysical in sign or in size.

**This is the D6 wall reached from the radiation side.** The brief opened branch 3 on the
argument that "G₀/⟨U⟩ is not a scaling relation, so it escapes D6". That is true only if
G₀ is *measured*. Every G₀ **proxy** we can build out of the main sequence plus sizes is
a scaling relation, inherits the near-constant exponents, and fails for exactly the
reason every abundance arm failed. Together with the sSFR result already in the brief,
this closes the cheap route.

**Consequence for the branch.** There is no way around task 1. The ionised/neutral band
ratio is the only G₀ estimator that is not a scaling relation in disguise, and it is
precisely the one that is unstable. Stabilising it is not the first task among several —
it is the *only* remaining task on the G₀ hypothesis.
