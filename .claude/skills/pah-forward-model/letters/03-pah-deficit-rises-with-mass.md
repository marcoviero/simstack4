# Letter possibility 03 — The PAH deficit rises with stellar mass (α-robust)

**Status:** current headline (decided 2026-07-01, user choice after α-sensitivity check).
**Supersedes:** `01-pah-tracks-lir-invariant.md` (flat/invariant framing, retired).
**Source:** `notebooks/2026-07-01-pah-forward-model-letter.ipynb` §5-7 (Wien slope), §9a/§9a-ii
(L_PAH/L_IR + its α-check). Summary: `docs/pah-forward-model-6-summary.md`.

---

## Working title

*The mid-infrared PAH deficit strengthens with stellar mass in normal star-forming galaxies
at cosmic noon, independent of the assumed warm-dust continuum slope*

## One-line thesis

From far-infrared tomographic stacking, two independent PAH-to-continuum ratios both rise
with stellar mass — a genuine PAH deficit at high M\*, not a warm-continuum artifact — and
this direction survives replacing the textbook Wien-slope assumption (α=2) with the branch's
own measured, data-driven value (α≈2.9-3.3).

## Headline measurements

| Quantity | value / slope | robustness |
|---|---|---|
| Wien slope α (warm-dust continuum, 24+70 µm, fold-ensemble) | **2.95 ± 0.28** (α=2.86-3.26 depending on baseline method) | z-cut invariant; not a bound/bug artifact (both triple-checked) |
| EW slope, `A=α_m/C_m` (mass slope, fold-ensemble) | α=2: **+0.351±0.159**; α≈2.86: **+0.619±0.033** | positive at both α, steepens under free α |
| **L_PAH/L_IR mass slope** (this branch, pooled sample) | α=2: **+0.146**; α≈2.86: **+0.212** dex/dex | sign-stable, 1.5× swing (vs 3-4× for raw amplitude) — the decisive α-robustness check |
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

1. **Old-vs-new discrepancy at matched α=2** (this branch's K-fold-pooled, smoothed-baseline
   pipeline gives +0.146 dex/dex; the June-28 single-split, unsmoothed pipeline gave +0.019,
   flat). Not yet isolated to a single cause (sample pooling / baseline smoothing /
   feature-ratio treatment) — see `pah-forward-model-6-summary.md` Open Questions. Must be
   resolved before quoting a final L_PAH/L_IR normalization or slope value in the paper.
2. **No fold-ensemble error yet on the L_PAH/L_IR slope** — §9a/9a-ii used the pooled sample's
   point-to-point scatter, not independent per-fold refits (unlike the EW slope, which has a
   proper fold-ensemble error). Needed for a citable significance.
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
- **arXiv:2606.20809** — sim L_PAH(M\*,z) prediction; direct overplot against the α-checked
  slope, not the old flat one.
- **arXiv:2606.18244 (PAHSPECS)** — JWST MIRI spectroscopy, bright tail; complementary
  independent check, no continuum-slope assumption needed.
- **Viero+22, Schreiber+18** — T_dust(z) evolution; the §9c two-pass refit (+0.81 K at z<3) is
  a modest, not dramatic, correction — weaker than Letter 02's original flat-framing
  hypothesis anticipated. Report honestly; do not oversell the T_dust story alongside this one.
