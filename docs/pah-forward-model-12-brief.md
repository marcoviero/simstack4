# pah-forward-model-12 — brief

**Branch:** `pah-forward-model-12` (open fresh off `main`; the work below was scoped out
during `lim-talk-figs-1` but belongs in its own branch). Commit-msg hook prepends `[branch]`
— verify the branch before committing.

## One-line thesis

Rebuild the tomographic PAH forward model as a **confusion-aware forward model** — "a
simstack for the lines": place the PAH lines at fixed rest wavelengths with **fixed physical
relative strengths for the features the tomography cannot resolve**, convolve with the Drude
profile *once*, then band-integrate — instead of fitting a free amplitude per feature group.
Then add the two missing physical components (a **plateau/continuum-under-features** term and
the **9.7 µm silicate dip**) and see whether they absorb the coherent residual that currently
inflates χ².

## Why (what branch-11/lim-talk-figs-1 established)

- The old welded 7.7+8.6 group asserted the **backwards** `DEFAULT_FEATURES` strengths
  (8.6=0.6089 > 7.7=0.4577; peak 8.6/7.7 = 1.33). Freeing 8.6 dropped χ²_red 6.5→5.4 and
  collapsed the within-bin sSFR slope η_A +1.11→+0.15 — but that low η_A is an **artifact**.
- **η_A is degenerate with the assumed 8.6/7.7 ratio** (near-linear, welded sweep, drude,
  the 2026-07-23 K-fold stacks):

  | peak 8.6/7.7 | integ | χ²_red | η_A |
  |---|---|---|---|
  | 0.10 | 0.05 | 5.38 | +0.20 |
  | 0.25 | 0.12 | 5.44 | +0.38 |
  | **0.50 (physical)** | **0.24** | **5.65** | **+0.62** |
  | 1.00 | 0.49 | 6.17 | +0.96 |
  | 1.33 (default) | 0.65 | 6.50 | +1.11 |
  | free split (fits 0.13) | 0.06 | 5.40 | +0.15 |

  η_A ≈ 0.15 + 0.7·(peak ratio). The free split and η_A are **the same degree of freedom**:
  with 8.6 as a free knob it floats down to absorb the 7.7 Drude wing that already covers the
  8.6 region (the "confusion"), dragging η_A with it. So η_A is **not independently
  measurable**; it is an upper limit set by whatever 8.6 prior you adopt.

- **Correct physical ratio:** integrated 8.6/7.7 ≈ **0.2–0.3** (Smith+2007 SINGS, Draine & Li;
  ionization-dependent, 8.6 always weaker). Code parametrizes by *peak*: integ = peak ×
  (area₈.₆/area₇.₇) = peak × 0.486, so integ 0.25 → **peak ≈ 0.5**. Already encoded as
  `pah_spectrum.FEATURES_86_CALIBRATED` (`_R86_FIXED=0.5`) via `rescale_feature_strength()`.

## The architecture (the actual task)

Tie the **unresolvable** features at a fixed physical ratio; free only the **separable** ones:

- **Ionized complex 7.7+8.6** → ONE component, fixed internal ratio (peak 0.5). No independent
  8.6 knob → nothing for η_A to soak up. This is the welded `[[1,2],...]` config with
  `features=FEATURES_86_CALIBRATED`. Under it the *honest* η_A ≈ **+0.6 upper limit**, not ~0.
- **6.2** (z≈2.9) and **11.3+12.7** (z≈1) → free amplitudes; the redshift lever arm resolves
  them, so their ratios carry the real band-ratio science.
- Optional single rigid **ionization** parameter tilting neutral↔ionized as a block.

This is largely a grouping/tying change to existing machinery (`feature_band_curves` already
convolves lines with the profile), not a new module — but it should be the **default**, with
the free split kept only as a printed systematic.

## Missing physical components

1. **Plateau / continuum-under-features** (PAHFIT has it; `PAHSpectrumModel` does not).
   `warm_continuum_kernel` (warm MBB) and `hot_ladder` (fixed-T MBB rungs) partly stand in but
   are not the ~5–10 µm PAH plateau. The Drude wings currently *are* the plateau proxy, done
   badly — hence the residual. Add a proper broad plateau term and test whether it pulls
   χ²_red toward 1 and stabilizes the feature/8.6/η_A trade-off.

2. **The 9.7 µm silicate dip — REVISIT.** There is a real trough in the stacked flux at rest
   ~9–10 µm (visible between the 7.7 peak and the 11.3+12.7 rise in the decomposition figure).
   The residual anatomy shows the model **over-predicts** it (data below model): pulls
   ~ −4.7σ (9.0–9.8 µm) and −2.2σ (9.8–10.6 µm), while feature *peaks* are under-fit
   (7.7–8.6 +2.4σ, 11.3+12.7 +2.3σ) — i.e. the model is too smooth, wings fill the trough.
   9.7 µm silicate absorption (Drude, λ₀=9.7, γ=3.3; already in `pah_model.py` via
   `include_silicate=True`) deepens exactly that trough. Earlier silicate tests found τ≈0
   **per fold**, but that was with free-per-group amplitudes and no plateau — **confounded**.
   Re-test silicate *jointly* with the plateau + physically-tied features; the dip may then
   require τ_sil > 0. Port silicate into `PAHSpectrumModel` (it is only in `PAHModel`).

## Concrete experiments

1. Make welded-at-physical (`FEATURES_86_CALIBRATED`, groups `[[1,2],[0],[3,4]]`) the fit
   default; free split → printed systematic. Reframe η_A as a **prior-bounded upper limit**
   (≈ +0.15 weak-8.6 … +0.6 physical-8.6).
2. Add a plateau term to `fit_shared`/`fit_evolving`; measure Δχ² and whether the 9–10.6 µm
   and feature-peak residuals flatten.
3. Port 9.7 µm silicate (Drude) into `PAHSpectrumModel`; fit τ_sil jointly with the plateau;
   re-decide detection (the confounded per-fold τ≈0 does not settle it).
4. Confirm the **science is invariant** to all of the above (it should be — see below).

## What is ROBUST — do not re-litigate

- The **mass slope / crossing** are set by the total PAH-to-continuum amplitude (the
  well-constrained 7.7-complex-as-a-whole), invariant to the 7.7↔8.6 split and to profile:
  all-z slope ≈ +0.10; z-slice crossing **+0.42 / +0.11 / −0.63** (z~1/2/3); F-test rejects
  flat (p=0.004). These survive; branch-12 is about χ²/η_A/absolute-L_PAH, not the crossing.
- **Drude is the correct profile** (physical shape + literature L_PAH convention; ×1.46 area
  vs gaussian). Keep it. Gaussian's lower χ² is it discarding real wing/plateau flux — not a
  reason to switch. See `feature_profile_area`.

## Data & code pointers

- **Stacks:** K-fold `20260723_181128 / _183744 / _190437` (`cosmos20_PAH_wide.toml`, Δz=0.30,
  z=0.2–6.0, offsets 0/0.10/0.20); combined `20260723_193310` (FARMER). Star-forming
  `split_0`, Tier C, mass bins `[9.9,10.6,10.8,11.0,11.5]`, `starburst_filter=0`.
- **Build the pseudo-spectrum df:** `pah_money_helpers.build_pah_spectrum_df(WRAPPERS,
  MASS_BINS, split_filter=[0], min_tier="C", starburst_filter=0)` → `smooth_baseline` →
  ×1e3 to mJy. (Mirrors `2026-07-23-pah-money-plots-wider-z-bins` cell 4 /
  `2026-07-22-lim-lines-to-footprints` cell 14.) `ANALYSIS_KWARGS`: `use_pah=False`,
  `temperature_prior="viero"`, `inflation_factors={24:10000, 70:{(0,0.8):1,(0.8,99):10000}}`,
  `use_covariance=True`.
- **Modules:** `pah_spectrum.py` (`fit_shared`, `fit_evolving`, `fit_evolving_mcmc`,
  `feature_band_curves`, `_profile_spectrum`, `rescale_feature_strength`,
  `FEATURES_86_CALIBRATED`, `warm_continuum_kernel`, `hot_ladder`); `pah_model.py`
  (`include_silicate=True`, Drude λ₀=9.7 γ=3.3 — the silicate reference to port).
- **Diagnostic:** `2026-07-22-lim-lines-to-footprints` §5c (relaxation ladder + residual
  anatomy). Notebooks dir is gitignored — `git add -f` build scripts.
- **Tests:** `test_pah_spectrum_recovery.py::TestCalibratedFeatureStrengths` (5, green);
  full suite 275 green. Add plateau + silicate recovery tests.
- **Memory:** `pah-template-rigidity`, `lim-via-pah-measured-inputs`.
