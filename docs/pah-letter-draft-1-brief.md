# PAH Letter Draft — Branch 1 Brief: assemble the ApJL letter

**Goal.** Turn the branch-9 measurement into a submitted ApJL letter whose **discussion is
built around the two-arm interpretation of the crossing**. This branch owns the manuscript
(structure, figures, prose, journal formatting, submission).

**Dependency / workflow.** This letter **follows `pah-interpretation-*`** (which build off
branch-9). **The two arms are NOT yet resolved** — that interpretation work happens first on
the `pah-interpretation-*` branches, and letter development resumes once it lands. So *for now*
this branch sets up the manuscript spine (the measurement, which stands on its own) and the
two-arm discussion **scaffold**; the discussion's mechanism claims are filled in from the
interpretation branches as they resolve, not asserted here. (The `pah-interpretation-1` brief
and the D1–D6 first pass read as if the arms are mostly settled — they are not; treat those as
constraints, not conclusions.)

**Headline (user decision):** the z-resolved **L_PAH/L_IR crossing** — the mass slope inverts
with redshift: **+0.36 (z~1) → 0 (z~2) → −0.87 (z~3)**, three independent estimators
(pooled-template, self-consistent-per-fold, combined stack) agreeing in sign at every slice.
**Secondary:** the **12.7/6.2 band-ratio decline with M\***. Measured by dithered simultaneous
stacking of COSMOS2020 against MIPS 24 (Δz=0.15 × 3–4 dither offsets → an R≈53 rest-frame
5–16 µm pseudo-spectrum).

**Framing constraints (from branch 7–9, non-negotiable):**
- **NO AGN-based headline.** AGN is one mandatory alternative paragraph (NUVrJ SF selection;
  z~1 slice agrees with the non-AGN Shivaei+24; Xie & Ho sign check). Ionization interpretation
  fine; AGN-*fraction* claims are not.
- Free-α results primary; combined-stack numbers where the K-fold-pooled fit has a known defect
  (bin0 A(6.2) SNR); N-weighted mean bin masses everywhere.
- **The nested-model p=0.005 is NEVER quoted externally** — "crossing beats flat" is robust,
  the p-value is directional only.
- Venue **ApJL** (single field, fold-scatter errors → not Nature/Science).

---

## The two arms — high-level (the spine of the discussion)

**The crossing forces a two-arm decomposition.** The mass slope of L_PAH/L_IR *inverts sign*
with redshift, so the discussion is organised around explaining two opposite behaviours that no
single variable produces:

- **Positive arm (z~1): L_PAH/L_IR RISES with M\*.** At cosmic-noon's low-z side, more massive
  galaxies are *more* PAH-rich per unit L_IR.
- **Negative arm (z~3): L_PAH/L_IR FALLS with M\*.** By z~3 the trend inverts — massive
  galaxies are *more* PAH-poor per unit L_IR.

Each arm can in principle be carried by *production* (build more/fewer PAHs) or *destruction*
(preserve/destroy existing PAHs), and by a *metallicity*, *gas-structure/density*, or
*radiation* driver. **Two candidate frameworks ("paths") span both arms:**

- **Path A — two-mechanism balance.** Metallicity-regulated PAH *abundance* sets the positive
  arm (higher-Z massive galaxies build more PAHs), while *destruction/dilution* in compact,
  intense star formation sets the negative arm (the local (U)LIRG PAH deficit taking over at
  high z). Two distinct mechanisms with different z-scalings.
- **Path B — single ISM-density mechanism.** One variable (gas density / Σ_SFR, via grain-grain
  shattering that only works in diffuse gas): gas fraction falling with M\* pulls the slope
  positive at z~1, Σ_SFR rising with M\* pulls it negative at z~3 — the crossing falls out of
  one density lever with no fine-tuning.

**The assumption-light claim that is already safe (state it explicitly):** *no single
controlling variable (Z, t_dep, Σ_SFR) has a mass slope that flips sign with z under standard
scaling relations* — so the crossing forces a **two-term or threshold** model regardless of
which path wins. Everything past that (which driver carries which arm; whether one Σ_SFR
threshold does the whole job) is the **open question resolved on `pah-interpretation-*`**, and
the letter's mechanism paragraphs are written from that resolution.

**What the first-pass tests (D1–D6, `pah-interpretation-candidates.md`) currently *constrain*
(not conclude):** they disfavour the simplest metallicity-*step* (our bins sit at/above the
0.5 Z☉ plateau at both ends) and pure gas-*tracing*, and lean toward a threshold/nonlinear
response — but which mechanism carries each arm is not settled. Do NOT write the discussion as
if it is.

---

## The measurement (the spine — settled, stands on its own)

| Result | Value | Source |
|---|---|---|
| L_PAH/L_IR mass slope, z~1 | +0.36 (3 estimators agree, ~+0.30–0.36) | money-plots §3c/§3d |
| L_PAH/L_IR mass slope, z~2 | ≈ 0 | " |
| L_PAH/L_IR mass slope, z~3 | −0.87 (N-weighted; −0.6/−0.7 on naive centres) | " |
| all-z slope | ≈ 0 (integrates the crossing; a discussion point, NOT a result) | " |
| 12.7/6.2 band ratio vs M* | declines (envelope-aware calibration) | money-plots §2b |

Robustness: three independent central-value estimators agree in sign at every slice; the
N-weighted-mass correction *widens* the swing ~25%; nested model (3 per-slice slopes vs 1) beats
flat by Δχ²=328 for 4 params (quote the qualitative verdict, not p). Novelty confirmed: no
L_PAH/L_IR-vs-M* at z~3 exists (JWST censuses stop at z≈2; PAHSPECS = 5 galaxies at z≈1.1); the
z~1 slice reproduces Shivaei+24 by an independent method (validation, not news). **The crossing
is the hook.**

For the **band ratio** discussion: charge (6.2 cation C–C vs 12.7 neutral C–H) is the leading
read; Whitcomb+24 size/metallicity predicts the *opposite* trend — own that tension; [NeII]
12.81 blend dilutes rather than creates it. (This too follows the interpretation branch's read
of whether charge or size wins.)

---

## Letter structure (ApJL, ~4 figures, ~3500 words)

1. **Intro.** PAH as SF/ISM diagnostic + the 24 µm-SFR bias; what is unmeasured (L_PAH/L_IR vs
   M\* beyond z~2); dithered stacking reaches the confused faint population.
2. **Data & method.** COSMOS2020 SF (NUVrJ); dithered simultaneous stacking vs MIPS 24 (+70
   null/anchor); the f₂₄/f_peak tomographic pseudo-spectrum; the forward model; three estimators
   + fold-scatter errors. (Compress hard — ApJL.)
3. **Results.** The crossing (Fig 1) + the band ratio (Fig 2).
4. **Discussion — the two arms (the focus).** State the two arms; the assumption-light
   two-term/threshold claim; then the mechanism read (Path A vs B, which driver per arm) **from
   `pah-interpretation-*`**; the band-ratio charge-vs-size read; the mandatory AGN paragraph.
   Fig 3 = the two-arm / two-term argument; Fig 4 = the external confrontation (Narayanan+26)
   — both sourced from the interpretation branches.
5. **Implications.** The 24 µm-SFR correction (mass/z-dependent); one line on LIM (the companion
   forecast); the T_dust connection (Viero+22 / DustEvolutionModel σ_SFR axis) if the density
   path wins.

## Figure plan

| Fig | Content | Source |
|---|---|---|
| 1 | **The crossing:** L_PAH/L_IR vs M\* in 3 z-slices (or slope vs z), 3 estimators; the +0.36→0→−0.87 inversion | `2026-07-10-pah-money-plots` §3c/§3d |
| 2 | **12.7/6.2 band ratio vs M\*** (envelope-aware) | money-plots §2b |
| 3 | **The two-arm / two-term argument** (crossing vs single-mechanism predictions; threshold model if it lands) | `pah-interpretation-*` |
| 4 | **External confrontation** (Narayanan+26 d q_PAH/d logM* at z=1/2/3) | `pah-interpretation-*` |

## Dependencies (what the letter waits on)

- **The two-arm resolution** (`pah-interpretation-*`) — the whole Discussion. Letter development
  RESUMES after it. **User is doing this next.**
- **α_wien ↔ η_A** (measurement debt, interpretation-1 Obj 2a): degenerate; gates quotable
  *absolute* L_PAH numbers. The *slopes* (differential) are robust regardless — that's why the
  crossing is the headline.
- **Partial→total PAH** — only for an absolute L_PAH/L_TIR statement, not the crossing.

## First actions on this branch (measurement-side, runnable now)

1. Draft the Intro + Method + Results text around the settled measurement (narrative spine).
2. Publication-quality Fig 1 (crossing) and Fig 2 (band ratio), re-styled from the money-plots
   notebook.
3. Write the Discussion **scaffold** (the two arms stated at high level + the safe two-term
   claim) with mechanism paragraphs and Fig 3/4 left as placeholders pending
   `pah-interpretation-*`.
4. Assemble `pah-refs.md` citations into the bibliography.

Refs: `docs/pah-refs.md`. Interpretation: `docs/pah-interpretation-candidates.md` +
`docs/pah-interpretation-1-brief.md` (follows onto this letter). Measurement notebooks:
`2026-07-10-pah-money-plots`, `2026-07-11-two-arms-tests`.
