# Letter possibility 03 — The PAH deficit is strongest at low stellar mass (α-robust)

**Status:** current headline (decided 2026-07-01, user choice after α-sensitivity check).
**CORRECTED 2026-07-01 (branch 7, advisor-flagged):** the working title/thesis below originally
read "the deficit rises/strengthens with stellar mass." That is backwards — both ratios in the
table below **rise** with M\*, meaning MORE PAH per unit continuum/L_IR at high mass, i.e. LESS
deficit at high mass. The deficit (defined as suppressed PAH emission relative to L_IR/continuum)
is strongest at LOW mass and weakens toward high mass. Only the wording is corrected here; the
measured numbers, slopes, and errors are unchanged. See `docs/pah-forward-model-7-brief.md` §0.
**Supersedes:** `01-pah-tracks-lir-invariant.md` (flat/invariant framing, retired).
**Source:** `notebooks/2026-07-01-pah-forward-model-letter.ipynb` §5-7 (Wien slope), §9a/§9a-ii
(L_PAH/L_IR + its α-check). Summary: `docs/pah-forward-model-6-summary.md`.

---

## Working title

*The mid-infrared PAH deficit is strongest in low-mass star-forming galaxies at cosmic noon and
weakens toward high stellar mass, independent of the assumed warm-dust continuum slope*

## One-line thesis

From far-infrared tomographic stacking, two independent PAH-to-continuum ratios both rise
with stellar mass — i.e. the PAH deficit is a low-mass phenomenon that eases at high M\*, not
a warm-continuum artifact — and this direction survives replacing the textbook Wien-slope
assumption (α=2) with the branch's own measured, data-driven value (α≈2.9-3.3).

## Headline measurements

| Quantity | value / slope | robustness |
|---|---|---|
| Wien slope α (warm-dust continuum, 24+70 µm, fold-ensemble) | **2.95 ± 0.28** (α=2.86-3.26 depending on baseline method) | z-cut invariant; not a bound/bug artifact (both triple-checked) |
| EW slope, `A=α_m/C_m` (mass slope, fold-ensemble) | α=2: **+0.351±0.159**; α≈2.86: **+0.619±0.033** | positive at both α, steepens under free α |
| **L_PAH/L_IR mass slope** (fold-ensemble, 2026-07-02) | free α: **+0.236±0.076 (3.1σ)**; pinned α=2: +0.135±0.136 (1.0σ); pooled α=2.86: +0.212 | sign-stable, 1.5× swing (vs 3-4× for raw amplitude) — the decisive α-robustness check; citable error bar now in place |
| L_PAH/L_IR normalization (pivot logM\*=10.75, α=2.86) | **≈8.0%** | scale is method-dependent (see caveats); direction is not |
| Feature ratios 6.2 : 7.7+8.6 : 12.7 (α=2 fit) | 1 : 1.5 : 4.5 | global, shared across mass bins |

## Why it's novel / publishable

- First measurement of the Wien-side warm-dust continuum slope from the data itself (24+70 µm
  jointly), rather than assumed — and a demonstration that the standard assumption (α=2) is
  wrong by a significant, robustly-bounded margin.
- A PAH-deficit mass trend that is explicitly stress-tested against the largest known
  systematic in this kind of measurement (the α-amplitude degeneracy) and survives — most
  broadband PAH-deficit claims do not report this check.
- Delivers the same concrete systematic the community needs (24 µm → T_dust/SFR contamination)
  but now derived self-consistently at the correct α, and wired directly into the shared
  SED-fitting library (`wien_mode="lir_pah"`), not just a one-off notebook calculation.

## Figure set (existing, in the letter notebook)

1. **§5-7**: α forward-model fit comparison (α=2 assumed vs α free), fold-ensemble +
   z-cut + exact-baseline robustness panels — establishes α≈2.9-3.3 is real.
2. **§8**: model overlay on raw stacked 24 µm flux vs z (continuum-only vs continuum+PAH),
   per mass bin, at best-fit α.
3. **§8b**: deconvolved PAH-excess pseudo-spectrum vs rest-frame λ, colour-coded by 4 redshift
   slices — the visual "different epochs, same rest-λ bump" consistency check.
4. **§8c**: intrinsic PAH line template (pre-bandpass, pre-photo-z), 4 mass-bin panels × 4
   redshift-slice curves — flags amplitude evolution as scatter-limited (not a claim) while
   isolating a comparatively better-behaved line-ratio pattern.
5. **§9a**: L_PAH/L_IR(M\*) at best-fit α — the headline figure.
6. **§9a-ii**: the α-robustness check itself (α=2.0 pinned vs α=2.86 best-fit slope
   comparison) — candidate as a referee-defense figure or supplementary panel, since it is the
   single strongest piece of evidence against the "this is just the α systematic" objection.
7. **§9c**: `plot_sed_grid` with the PAH correction live in the fit (`wien_mode="lir_pah"`) —
   shows 24 µm as a real, model-informed point at z<3 instead of an excluded placeholder.

## Caveats to resolve before submission

1. **RESOLVED (2026-07-02, `notebooks/2026-07-02-pah-narayanan-confrontation.ipynb` §3):**
   the old-vs-new discrepancy at matched α=2 is the **baseline treatment**. A 16-variant grid
   (sample × baseline × fit path × L_PAH definition) shows the old-like configuration
   reproduces the June-28 flat slope (+0.035 vs +0.019); raw→smoothed baseline moves the slope
   by +0.29 dex/dex while every other factor moves it ≤0.03. All 8 smoothed variants rise
   (+0.13 to +0.21); all raw variants are flat-or-negative AND sample-unstable (fold0 +0.035
   vs pooled −0.13/−0.19). June-28's flat value is superseded as a raw-baseline (Tier-C-noise)
   artifact. Remaining paper-level cross-check: Tier-A/B-only raw-baseline variant.
2. **RESOLVED (same notebook, §4):** fold-ensemble L_PAH/L_IR mass slope =
   **+0.236 ± 0.076 dex/dex (3.1σ) at free α** (per-fold α_wien 3.17/3.27/2.40);
   **+0.135 ± 0.136 (1.0σ) at pinned α=2**. Quote free-α as primary, α=2 as conservative
   bound. Pooled reference at α=2.86: +0.212, normalization 7.99% at pivot logM*=10.75.
3. **α itself is method-limited** (2.9-3.3 depending on fast-retilt vs exact-baseline
   reconstruction) — always quote the L_PAH/L_IR slope's range across that band, not a single
   point estimate, mirroring how §9a-ii already reports it.
4. Low-mass reliability: check whether any bin's FIR-fit L_IR is Eddington-bias-prone (flagged
   as a specific concern for the *lowest* mass bin in the superseded Letter 01 doc — re-check
   under the current K-fold/smoothed-baseline pipeline, not assumed inherited).
5. σ_SFR cross-cut (open since branch 4) would test whether the mechanism is radiation-field
   intensity (predicts deficit at fixed M\* rising with σ_SFR) rather than a pure mass effect.

## Positioning against cited papers (`docs/pah-refs.md`)

- **Smith+07, Galliano+21** — local PAH/FIR ratio declines with sSFR/L_IR/mass; this result is
  now *consistent in direction* (deficit rises with mass) after the α pivot, unlike the
  superseded flat framing which was in tension with them. Frame as broad agreement with a
  methodologically distinct (broadband stacking vs resolved/spectroscopic) high-z extension.
- **Tielens 08; Egorov+25; Leroy+23** — resolved radiation-field PAH-destruction mechanism;
  now the natural physical explanation to test via the still-open σ_SFR cross-cut.
- **Narayanan et al. 2026 (arXiv:2606.20809)** — read + confronted 2026-07-02
  (`notebooks/2026-07-02-pah-narayanan-confrontation.ipynb`): the paper publishes **no**
  q_PAH(M\*) at fixed z, so we derived the two mass-axis channels its shattering mechanism
  implies. Density/shattering chain predicts [−0.35, +0.09] dex/dex (f_mol is saturated
  0.90–0.99 at cosmic noon, so the Σ_SFR channel dominates, negative); enrichment/PZR chain
  predicts [−0.10, +0.55]. Our +0.236±0.076 sits **1.8σ above the density chain's upper
  edge** and inside the enrichment band — first observational constraint on this model along
  the stellar-mass axis; the low-mass deficit behaves like an abundance (PZR-like) effect,
  not dense-ISM suppression. (At pinned α=2 the discrimination weakens to 0.3σ — it rests on
  the branch-6 α measurement.)
- **arXiv:2606.18244 (PAHSPECS)** — JWST MIRI spectroscopy, bright tail; complementary
  independent check, no continuum-slope assumption needed.
- **Viero+22, Schreiber+18** — T_dust(z) evolution; the §9c two-pass refit (+0.81 K at z<3) is
  a modest, not dramatic, correction — weaker than Letter 02's original flat-framing
  hypothesis anticipated. Report honestly; do not oversell the T_dust story alongside this one.
