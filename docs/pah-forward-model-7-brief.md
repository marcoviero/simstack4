# PAH Forward Model — Branch 7 Brief

**Goal**: take the settled Letter headline — **the PAH deficit rises with stellar mass, and
this direction survives replacing the assumed warm-dust slope (α=2) with the branch's own
measured value (α≈2.9–3.3)** — and turn it into a submittable Letter + talk. This brief was
substantially rewritten 2026-07-01; the previous version (error-rescaling utility, σ_α×√χ²_red,
a from-scratch T_dust-correction plan, and an α(M\*) figure framed around the old flat/EW-only
result) is **superseded** — several of those objectives are done, one is explicitly rejected
project convention, and the headline itself changed. See `docs/pah-forward-model-6-summary.md`
and `.claude/skills/pah-forward-model/letters/03-pah-deficit-rises-with-mass.md` for the full
history of how the headline got here — read those before touching this brief again.

---

## What branch 6 already delivered (don't redo)

- **α measured, not assumed**: 2.86–3.26 depending on baseline-construction method,
  fold-ensemble 2.95±0.28. Triple-checked (z-cut, bound-widening, exact-vs-fast baseline).
- **Two independent PAH/continuum ratios both rise with mass and both survive the α check**:
  the EW-style ratio `A=α_m/C_m` (fold-ensemble, α=2: +0.351±0.159; α≈2.86: +0.619±0.033) and
  the new `L_PAH/L_IR` ratio (pooled sample, α=2.0 pinned: +0.146; α≈2.86: +0.212 dex/dex,
  sign-stable, only 1.5× swing vs the 3–4× swing already documented for the raw amplitude).
- **The correction is wired into the shared library**, not a one-off notebook calculation:
  `Greybody.alpha_wien` (overridable Wien slope), `wien_mode="lir_pah"` /
  `Greybody._pah_flux_lir` (L_PAH/L_IR-based correction via the real bandpass kernel), threaded
  through `SimstackResults`/`run_analysis_only(**kwargs)`. Round-trip validated to 0.0000%.
  Full test suite 257/257, no regressions.
- **A first two-pass refit** (`wien_mode="lir_pah"`, α=2.86, z-dependent 24µm inflation):
  T_dust shifts +0.81 K at z<3 (small, not dramatic), tier counts unchanged (median-SNR tier
  logic isn't sensitive to one previously-suppressed band — a real finding, don't expect
  Tier C→B promotions from this correction alone).
- **Rejected, do not resurrect**: the `√χ²_red` error-rescaling in the old brief's Objective 1.
  Established project convention (branches 5–6) is the **fold-ensemble error** (independent
  K-fold catalog-split refits, scatter = error) — it is *tighter* than the rescaled formal
  error here, meaning the rescale was over-conservative, not under. Any new error budget in
  this branch uses the fold-ensemble method, full stop.

---

## Objective 1 — Close branch 6's two open items (prerequisite for a citable number)

### 1a. Isolate the old-vs-new L_PAH/L_IR discrepancy

At *matched* α=2.0, the June-28 pipeline gave a flat slope (+0.019, single catalog split,
unsmoothed baseline) and branch 6's pipeline gives a clearly rising one (+0.146, K-fold-pooled,
`smooth_baseline`-smoothed). Isolate which change is responsible by toggling one factor at a
time on the branch-6 pipeline:
- Single split (`split_filter=[0]` only, no pooling) vs pooled 3-fold — does the slope drop
  toward flat with one split alone?
- Smoothed (`smooth_baseline`) vs raw per-point baseline — does turning smoothing off change
  the slope?
- Feature-group ratio treatment (`FEATURE_GROUPS`/`r_g` fit vs whatever June-28 used).

This gates how confidently the paper can quote a single L_PAH/L_IR slope value — right now it
can only cite "rising, sign-stable, magnitude 0.15–0.21 dex/dex depending on method," which is
honest but weaker than a paper wants.

### 1b. Fold-ensemble error on the L_PAH/L_IR slope

`§9a`/`§9a-ii` in the branch-6 notebook used the pooled sample's point-to-point scatter, not
independent per-fold refits. Redo exactly like the EW slope already is: refit `§9a`'s
`lir_pah_ratio_by_bin` machinery on each of the 3 disjoint K-fold splits separately, report
mean ± scatter/√N. This is the number that actually goes in the paper table.

---

## Objective 2 — σ_SFR cross-cut (open since branch 4)

Tests the physical mechanism directly: if the deficit is radiation-field-driven (Tielens 08,
Egorov+25, Leroy+23 — harder UV field → PAH destruction), it should strengthen with σ_SFR at
**fixed** M\*, not just track M\* itself. Requires the 2 mass × 3 σ_SFR stacking runs (config
scaffold exists, `group_col="sigma_sfr"`). Extend `§9a`'s L_PAH/L_IR machinery — already
factored into a reusable function (`lir_pah_ratio_by_bin`) — to the 2D bin structure. If it
holds, this is the mechanism figure; if it doesn't, the deficit is closer to a pure mass/depth
effect and the paper should say so rather than assume the radiation-field story.

---

## Objective 3 — Finish the T_dust correction companion result

Branch 6 §9 did a first pass (two-pass refit, ΔT_dust at z<3). Not yet done:
- Refit the actual **T_dust(z) relation** (not just a per-bin ΔT) with the correction applied,
  and compare its shape to Viero+22 (`T=23.8+2.7z+0.9z²`) and Schreiber+18 (`T=32.9+4.6(z−2)`).
- Given the branch-6 shift was modest (+0.81 K, not the +4 K upper bound the superseded Letter
  01/02 framing anticipated), set expectations accordingly: this is likely a **secondary,
  supporting result** for the Letter, not a second headline. Report it as such — don't oversell.
- Reuse `wien_mode="lir_pah"` as-is; no new library work expected here, just more analysis on
  top of what §9 already built.

---

## Objective 4 — Referee defense strategy (rewritten for the current headline)

**Q: "Isn't the rising L_PAH/L_IR slope just your own known α–amplitude systematic?"**
A: Checked directly (§9a-ii): pinning α at the old assumed value (2.0) instead of the fitted
2.86 changes the slope by only 1.5× (0.146→0.212 dex/dex) and never changes sign — far smaller
than the 3–4× swing already documented for the raw (non-ratio) PAH amplitude. The ratio's
denominator (L_IR) absorbs most of the α-dependence that the numerator alone would carry.

**Q: "You had a flat result in an earlier version of this analysis. What changed?"**
A: State plainly: the earlier (flat) measurement used the assumed α=2, a single catalog split,
and an unsmoothed baseline; **the direction (rising) is robust to the α question specifically**
(§9a-ii), and Objective 1a above pins down the remaining methodology difference before
submission. Do not hide this history — the pivot itself, done via an explicit discriminating
check rather than by assumption, is a strength of the methods section, not a weakness.

**Q: "α is only known to ±0.3ish (2.9–3.3 across methods) — how does that propagate?"**
A: Every headline number is quoted across that band, not as a point estimate (see EW slope and
L_PAH/L_IR slope tables in `docs/pah-forward-model-6-summary.md`), and the round-trip-validated
library correction (`wien_mode="lir_pah"`) makes it trivial to regenerate every downstream
figure at any α if a referee wants a specific value tested.

**Q: "Why trust the fold-ensemble error over a formal covariance / χ² rescale?"**
A: The formal covariance omits cosmic/sample variance, and a diagonal χ² over
source-correlated tomographic points over-counts degrees of freedom. K=3 independent catalog
splits sidestep both: refit each fold separately, use the ensemble scatter. Established this
project's convention in branch 5 (the √χ²_red rescale was checked and found *over*-conservative
relative to the fold error, not under).

**Q: "Radiation field or just stellar mass?"**
A: Objective 2 (σ_SFR cross-cut) is the direct test — report whichever way it comes out.

---

## Objective 5 — Talk + paper figure set

Figures already exist in the branch-6 notebook (`notebooks/2026-07-01-pah-forward-model-letter.ipynb`)
and mostly need reformatting for publication, not rebuilding from scratch:

1. **The method**: MIPS 24 µm bandpass swept across rest-frame PAH features vs z (existing
   figure-1-style asset from branch 3, `build_pah_fig1.py` — reuse/update).
2. **α is not 2**: fold-ensemble + z-cut + exact-vs-fast-baseline robustness panel (notebook
   §5–7) — establishes the measurement the rest of the paper leans on.
3. **The headline**: L_PAH/L_IR(M\*) at best-fit α (notebook §9a) *and* the α-robustness check
   (§9a-ii, α=2.0 pinned vs 2.86 slope comparison) as a paired panel — this pairing **is** the
   referee-defense figure for the "isn't this just your α systematic" question, worth
   promoting to a main-text figure rather than supplementary.
4. **Model against the data**: model-overlay (continuum vs continuum+PAH on raw 24 µm flux,
   §8) and the deconvolved pseudo-spectrum coloured by redshift slice (§8b) — the visual
   "different epochs, same rest-λ bump" consistency argument.
5. **The correction in the actual SED fit**: `plot_sed_grid` with `wien_mode="lir_pah"` live
   (§9c) — shows this isn't just a standalone measurement, it changes the pipeline's own fits.
6. **(if Objective 2 completes)** σ_SFR cross-cut: deficit vs σ_SFR at fixed M\*, mechanism test.
7. **(if Objective 3 completes)** Corrected T_dust(z) vs Viero+22/Schreiber+18 — secondary
   figure, sized/framed as supporting evidence given the modest (+0.81 K) shift found so far.

Builder pattern: follow branch 3's `notebooks/build_pah_fig{1,2,3}.py` structure (these are the
only PAH notebook files currently committed to the repo — `notebooks/` is otherwise gitignored;
force-add (`git add -f`) any new builder script intended to ship with the paper, same as those).

---

## Config and data notes

- Dataset: `cosmos20_stacking_20260630_{193627,211122,222635}.json` — 3 disjoint K-fold splits
  (`cosmos2020_PAH_split{0,1,2}of3`), each also offset in z-binning (dither). Config:
  `config/cosmos20_PAH_dithered_3cats.toml` (toggle `[catalog].file` + which `bins=` block is
  uncommented to regenerate any of the 3 runs — last-used state committed at the end of branch
  6 was split2of3 / run 2, 26-bin scheme; check before assuming it's still what you want).
- 4 science mass bins (10.0–10.5, 10.5–10.8, 10.8–11.1, >11.1); 8.5–10.0 stays an unanalysed
  low-mass nuisance layer.
- `FEATURE_GROUPS = [[0],[1,2],[4]]` (6.2 | 7.7+8.6 | 12.7, 11.3 blind) — keep consistent with
  branch 6 for any comparison; changing it invalidates a direct slope comparison.
