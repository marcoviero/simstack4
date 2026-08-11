# LIM Talk Slides — working template

Source sketch: `Note Jul 27, 2026.pdf` (11-slide handwritten draft). This markdown is the
factual scaffold for notebook `2026-07-27-lim-talk-slides`, which in turn is a template
for Keynote slides (built by hand, not generated). Built collaboratively, slide by slide.

**Rule**: every bullet is either a fact we've established in this project (with a doc/notebook
pointer) or a literature citation. Nothing invented. Where a citation's bibliographic detail
couldn't be fully verified, that's flagged explicitly rather than guessed.

**Figure rule**: every figure section says "reproduce in the new notebook" and names the
source cell/section — never "pull from" the old notebook. The point of this doc is that
Keynote-building (and notebook-building) don't require chasing anything down separately.

**Slide map** (talk is 12+3 min — main deck stays lean; sketch's original numbering in
parens where it shifted):

| # | Title | Status |
|---|-------|--------|
| 1 | *(was sketch #2)* What are PAHs? | done |
| 2 | *(was sketch #4)* Stacked fluxes in FIR / PAH contamination | done |
| 3 | *(was #5)* MIPS bandpasses | done |
| 4 | *(was #6)* Dithering | done |
| aside | Simstack philosophy / personal story / teasers | done, not numbered |
| 5 | *(was #7)* Forward Model | done |
| 6 | *(was #8)* Band ratio vs M* | done |
| 7 | *(was #9)* Interpret (M*, z trends) | done |
| 8 | *(was #10)* Modeling [CII] | done |
| 9 (backup) | *(was #11)* CO | done, held out of main deck |
| backup | Traditional PAH measurement methods (spectroscopy) | done, held out of main deck |

**2026-07-29: the original Slide 1 ("Relating [CII] to PAH", the PDR-bridge framing) was
dropped from the deck** — judged not to be pulling its weight; the talk now opens directly
on "what are PAHs" (creation/destruction/ionization vocabulary). The PDR-bridge physics
(Helou+01, Croxall+12, Smith+17, McKinney+20) is still needed later, for the [CII] modeling
slide's bridge argument — reuse those citations there rather than re-deriving them; not yet
re-checked that they're still present in that slide's bullets.

**⚠ Open caveat, flagged 2026-07-28, not yet fixed in the slides below.** The K-fold
`split{0,1,2}of3` runs used for "fold-scatter" error bars on the Dithering, Band-ratio,
Interpret, and Modeling [CII] slides turned out NOT to be disjoint galaxy subsamples — they
stack the same ~237k galaxies on offset redshift-bin grids (`kfold-runs-not-disjoint`
project memory, verified 2026-07-28). Where a slide says the fold scatter reflects
"independent," "disjoint," or "uncorrelated" samples, that's currently wrong — it's really a
z-binning sensitivity check. Revisit before the talk is finalized; not fixed yet per explicit
instruction to keep moving for now.

---


## Slide 1 — What are PAHs?

**Purpose**: high-level PAH primer — composition, origin, radiation mechanism, energy
budget, formation/destruction, and the ionization-state band tracers — before getting into
how we measure them.

### Bullets

- **What**: Polycyclic Aromatic Hydrocarbons — planar molecules of fused benzene-like
  carbon rings with peripheral H atoms (~20–100+ C atoms); a large-molecule/small-grain
  population, not a mineral dust species.
- **How they radiate**: single-photon (stochastic) heating — absorption of one UV/optical
  photon transiently spikes the internal (vibrational) temperature far above equilibrium;
  the energy comes back out as IR fluorescence in C–C and C–H stretching/bending modes,
  producing the characteristic emission features rather than a smooth thermal continuum
  (Draine & Li 2007, ApJ 657, 810).
- **Energy budget**: PAH emission carries **~10% of total L_IR** (8–1000 μm) in normal
  star-forming galaxies (Smith et al. 2007) — already the project's adopted total-PAH/L_IR
  scale (`docs/pah-refs.md`, `_pah_coeffs` calibration in `greybody.py`).
- **Formation**: two channels discussed in the literature —
  - classical picture: injected into the ISM from carbon-rich AGB star outflows /
    photo-processing of larger carbonaceous grains (Tielens 2008, ARA&A 46, 289);
  - in-situ ISM growth via grain–grain shattering in diffuse gas, efficient where density
    is low (Narayanan et al. 2026, arXiv:2606.20809 — already read into this project,
    `docs/pah-refs.md`).
- **Destruction**: photodissociation/ionization by hard UV and X-ray photons in intense
  radiation fields — HII regions, AGN — quantified as PAH fraction anti-correlating with
  ionization parameter U (Egorov et al. 2025, A&A 703, A103; Leroy et al. 2023, ApJS 264,
  10; low-metallicity small-grain destruction channel: Engelbracht et al. 2005, ApJ 628,
  L29) — all already in this project's reference set.
- **Ionization-state tracers** — the band ratio that does the diagnostic work:
  - neutral-PAH-dominated: **11.3, 12.7 μm** (C–H out-of-plane bending modes)
  - ionized(cation)-PAH-dominated: **6.2, 7.7, 8.6 μm** (C–C stretching modes)
  - the ionized/neutral ratio (e.g. 6.2/11.3 or 7.7/11.3) tracks the ionization parameter
    U = G₀/n_e (radiation field intensity over electron density) — Tielens (2008) §1–4,
    the canonical review already anchoring this project's band-ratio physics.

### Figure

**Reproduce directly in the new notebook** — port the §1 cell from
`notebooks/2026-07-25-pah-fingerprints.ipynb` ("The lines — the intrinsic PAH spectrum
(Drude profiles)", output `fig1_pah_drude_spectrum.png`) into `2026-07-27-lim-talk-slides`
as its own cell, not a link back to the old notebook: the model PAH emission spectrum built
from Drude profiles at 3.3, 6.2, 7.7, 8.6, 11.3, 12.7, 16.4, 17.0 μm, each feature
colour-coded and labelled. Reuse the same colour key for the rest of this talk's figures.

### Speaker notes

- This is the "vocabulary" slide — composition, mechanism, energy budget, birth/death,
  and the neutral/ionized read-out — so that later slides (measuring the band ratio, the
  M*/z trends) land as *changes in something the audience already understands*, not as
  unexplained jargon.
- The ~10% L_IR number is worth lingering on: it's small enough that PAHs are a detail for
  bolometric IR luminosity, but large enough (and concentrated in a few narrow bands) to
  matter a great deal for any single-band flux — which is exactly the contamination problem
  the talk is about.
- Two formation channels (AGB stars vs. in-situ shattering) are both live in the literature;
  don't present it as fully settled — flag it as an open mechanism question if asked.
- The neutral/ionized band split is the single fact the audience needs to carry into the
  band-ratio evolution result later — it's worth a beat of emphasis and maybe a pointer
  gesture at the 6.2/7.7/8.6 vs 11.3/12.7 features on the spectrum.

---

## Slide 2 — Stacked fluxes in FIR: PAHs contaminate the greybody

**Purpose**: acknowledge the methodological inspiration, then show *our own* stacked SED
with the MIPS 24 μm point sitting above the fitted greybody — the excess this whole talk
is about.

### Bullets

- Inspiration acknowledgment: measuring a hidden line/feature as an **excess in broadband
  photometry**, statistically, without resolving individual sources —
  **Agrawal, Aguirre & Keenan (2026, A&A 705, A246)** and **Pullen, Serra, Chang, Doré & Ho
  (2018, MNRAS 478, 1911)** (both added to `docs/pah-refs.md` this session).
- The personal-frustration framing: fitting a single modified-blackbody (greybody) to a
  stacked FIR SED, MIPS 24 μm routinely sits *above* the fitted curve — not noise, not a
  bad fit, but PAH emission riding on top of the thermal dust continuum.
- This excess is exactly why the PAH tomography program exists: 24 μm has to be
  down-weighted/excluded from greybody fits (`inflation_factors={24: 10000, ...}`,
  `CLAUDE.md` PAH Tomographic Stacking section) or it biases the fitted dust temperature.

### Figure

**Built, executed, and verified in `notebooks/2026-07-27-lim-talk-slides.ipynb`** (real
data, not the Viero+22 published figure). Two things were found and fixed while building
this, both worth knowing if this cell is ever touched again:

1. **Frame bug**: `SEDFitResult.model_wavelengths` is stored in the **rest frame**
   (`greybody.py` ~line 972), while `.wavelengths`/`.flux_densities` are observed-frame.
   Plotting them together directly compares the wrong x-axis and produces a spurious,
   sign-inconsistent multi-sigma offset that looks like noise, not a PAH signature. Fix:
   `model_wavelengths * (1+z)` before plotting.
2. **Wrong baseline**: a single stacking bin's own greybody fit (`model_fluxes`) is a
   noisy, high-variance baseline — across every bin in the dataset it gave at best a
   sub-1σ "excess," indistinguishable from zero. The project's own established tool for
   this exact comparison is **`f24_cold`** (`notebooks/pah_money_helpers.py`'s
   `smooth_baseline()`): a Wien-side cold-continuum extrapolation from a T(z,M\*)/
   amplitude(z,M\*) relation trained jointly across many Tier A/B bins — already computed
   in `df_pool_sm` (loaded once in the shared setup cell, reused across Slides 3/5/6/7/8).
   Using that as the "no-PAH" reference instead of a single bin's own fit is what
   `pah_money_helpers.py`'s `pop_id` column (added this session) exists to support.

- **Data source**: the same 3 K-fold stacking runs (`cosmos2020_PAH_split{0,1,2}of3`,
  `RUN_DATES = {0: "20260723_181128", 1: "20260723_183744", 2: "20260723_190437"}`) already
  loaded once in the notebook's shared setup cell and reused across Slides 3, 5, 6, 7, 8 —
  see the caveat at the top of this doc: these are z-dithers of the same galaxies, not
  disjoint K-folds, despite the directory-name convention.
- **Bin choice**: among Tier A/B, log M\*≥10.3 populations, rank by `(MIPS_24 − f24_cold) /
  MIPS_24_err` and take the largest. The notebook run picked
  `redshift_1.9_2.2__stellar_mass_10.8_11.0__split_0` (z≈2.05, log M\*≈10.9, N=424, Tier A,
  **34.9σ** excess) — squarely in the mid-mass, z~2 regime where MIPS 24 μm samples the
  strong 7.7+8.6 μm complex, and physically sensible in magnitude: real 0.165 mJy vs. a
  0.053 mJy smoothed baseline, a genuine ~3× excess (the huge σ comes from MIPS 24's sharp
  PSF giving low per-source confusion, hence a tight formal stacking error, not from
  anything suspicious).
- **Plot**: real measured flux points at all 8 bands with error bars, overlaid with the
  smoothed cold-continuum curve (not a per-bin greybody fit) swept across the full
  observed-wavelength range, log-log axes. **Color by instrument**:
  - MIPS (24, 70 μm) — **red**
  - PACS (100, 160 μm) — **yellow**
  - SPIRE (250, 350, 500 μm) — **green**
  - SCUBA2 (850 μm) — **blue**
- Confirmed visually: MIPS 24 (and 70) sit clearly above the curve; PACS/SPIRE track it
  closely as expected for bands that constrain the fit.

**Second figure, added 2026-08-02 — animation**: same idea, made dynamic. Fixes ONE mass
bin (10.6 < log M\* < 10.8, a single K-fold run for a clean monotonic z sequence) and
cycles through its real stacked SEDs from low to high z. Each frame shows the real
multi-band data, the smoothed cold-continuum curve, and the MIPS 24 μm point (large star
marker) tracking above/below the continuum as different rest-frame PAH features sweep
through the band with z. A second, dashed curve overlays a Drude PAH template on top of
the continuum — its **shape** is the same template used everywhere else in this deck
(`rest_spectrum`/`FEATURES`), but its **amplitude is set, per frame, by the real measured
excess** (`MIPS_24 − f24_cold`), not independently fit — illustrative of *why* the point
moves, anchored to real numbers rather than an arbitrary bump. Saved as
`talk_slide2_mips24_tracking.gif`.

### Speaker notes

- Credit the inspiration briefly and by name — this is a "someone else had the same idea
  in a different tracer" moment, not a competing claim.
- Then pivot to first person: "when I fit a simple greybody to our stacked SEDs, this is
  what I kept seeing" — walk the audience's eye from the well-behaved FIR points up to the
  24 μm point sticking out.
- Land the sentence that sets up the rest of the talk: that excess *is* PAH emission, and
  the rest of the talk is about turning "there's a bump" into a measurement of how much
  PAH, in which feature, as a function of mass and redshift.

---

## Backup slide — Traditional PAH measurement methods

**Status: held out of the main deck.** 12+3 min doesn't leave room for a methods detour;
keep this ready in case a question opens the door (e.g. "why stack instead of just taking
spectra?").

### Bullets

- Traditional route: mid-IR spectroscopy (Spitzer/IRS locally, JWST/MIRI-MRS at high z) —
  directly resolves individual PAH features per galaxy.
- Shortcomings driving this talk's method instead:
  - **Expensive per source** → spectroscopic PAH samples at cosmic noon and beyond are
    small (tens, not thousands, of galaxies).
  - **Selection bias**: spectroscopic follow-up targets are necessarily bright/massive
    IR-detected galaxies — not representative of the confused, low-mass population that
    dominates the LIM signal (this is the same population-match argument that motivates
    stacking generally, `docs/forecast-lim-via-pah-1-brief.md` §"Stacking reaches exactly
    LIM's population").
  - Concretely: PAHSPECS (5 z≈1.1 galaxies, JWST/MIRI-MRS) is the kind of sample size this
    method produces — already read into this project (`docs/pah-refs.md`) as a comparison
    point, not a competitor at these numbers.

### Figure

None planned — this is a backup slide, likely text-only if it gets used.

### Speaker notes

- Only pull this up if asked directly why we don't "just get spectra" — the honest answer
  is sample size and selection, not that spectroscopy is wrong.
- If used, land it back on the stacking pitch: simultaneous stacking measures the mean over
  the *unresolved* population, which is the population LIM actually cares about.

---

## Slide 3 — MIPS bandpasses: the blending problem

**Purpose**: show the actual measurement challenge — MIPS 24/70 μm are broad bandpasses,
not narrow spectral points, so a redshift-swept PAH feature gets smoothed into a broad
"fingerprint" rather than read off directly. Set up why tomography (next slides) is needed
at all.

### Bullets

- MIPS 24 μm and MIPS 70 μm are **broad** photometric bands, not spectrometers: at z≈2 the
  MIPS 24 μm band spans roughly rest-frame 6.5–9.7 μm — wide enough to contain *multiple*
  PAH features (e.g. 7.7 and 8.6 μm) simultaneously.
- A single broadband flux measurement is therefore a **bandpass-weighted integral** over
  whatever intrinsic spectrum currently falls in-band — not a spectral point. You cannot
  read off one feature's amplitude directly from one broadband number.
- As redshift increases, a given rest-frame PAH line sweeps through the band: entering,
  peaking, then leaving. The bandpass response smooths this sweep into a broad, smooth
  curve in z-space — the **"fingerprint" `K_g(z)`** for that feature (or feature group).
  This is the same quantity computed in §2 of `notebooks/2026-07-25-pah-fingerprints.ipynb`
  ("Each line's fingerprint — the templates that get fit"): sharp intrinsic Drude lines in,
  broad coloured z-space templates out.
- The bandpasses themselves are **real, calibrated instrument response curves**, not
  schematic shapes: MIPS 24 μm from IRSA calibration, MIPS 70 μm from the SVO Filter
  Profile Service (`Spitzer/MIPS.70mu`) — both tabulated in `pah_bandpass.py`
  (`get_bandpass`), cross-checked against the frozen reference arrays by a guarding test.

### Figure

**Reproduce/rebuild directly in the new notebook, as an animation.** Two existing
notebooks contain the right pieces, but neither should be reused verbatim:

- `notebooks/2026-04-29-animated-figures-for-lim-talk.ipynb` ("Animation 1 — The Blending
  Problem in Motion") already built exactly this concept — a two-panel sweep over
  z = 0.5–3.5, left panel showing which PAH features are currently inside the MIPS 24 μm
  band (observed frame), right panel building up the per-feature template curve `T_j(z)`
  point by point. **Reuse the animation structure**, but it was built on the **old, frozen**
  `pah_model.PAH_FEATURES` template, whose 8.6 μm strength is backwards relative to every
  observed PAH spectrum (larger than 7.7 μm) — a since-corrected error (branch-12, see
  `pah-template-rigidity` project history). **Do not reuse those template numbers.**
- `notebooks/2026-07-25-pah-fingerprints.ipynb` §1–§3 has the **current, correct** pieces:
  the real Drude-profile intrinsic spectrum (`FEATURES_CALIBRATED`/`PHYSICAL_GROUPS`, the
  branch-12 welded-ratio template), the real bandpass (`get_bandpass`), and the fingerprint
  construction (`feature_band_curves`, plus the notebook-local `group_fingerprint`/
  `band_footprint`/`zcolored_line` helpers) — but these cells are static, not animated.
- **Build the animation on the branch-12 template + real bandpass, TWO panels — rebuilt
  2026-07-30 (v3, superseding an intermediate 3-panel attempt)**:
  1. **Individual Drude lines + real bandpass sweep**: the rest-frame spectrum decomposed
     into its individual features (same `FEAT_COLOR` palette as Slide 1's Drude plot, each
     line drawn with a bolder/darker stroke of its own fill colour — not the summed grey
     total), with the **actual MIPS 24/70 response curves**
     (`get_bandpass(b).lam_fine`/`resp_fine`, real tapered/non-top-hat shapes, IRSA
     calibration for MIPS 24, SVO Filter Profile Service for MIPS 70) redshifted and swept
     across it as z increases.
  2. **One fingerprint curve per PAH feature** (not just per band or per group) — built on
     a **rest-wavelength-probed** x-axis (`BAND_EFF[b]/(1+z)`), matching
     `2026-07-25-pah-fingerprints.ipynb` §2's static reference figure, instead of a
     redshift axis — so this panel visually converges toward that same reference plot by
     the end of the sweep. MIPS 24 solid, MIPS 70 dashed, same `FEAT_COLOR` per feature as
     panel 1. **Band totals overlaid in the band colours** (bold red/blue,
     `sum(_single_fp[(b, j)] for j in _feat_idx)`) so the additivity is visible directly in
     the figure — the total is exactly the sum of the per-feature curves underneath it (the
     band integral is linear; verified numerically to 1e-4 agreement). Added 2026-07-31 after
     the static reference figure (`talk_slide6_fingerprints.png`) was misread as showing a
     "summed fingerprint" bigger than its parts — that plot's grey background is the raw,
     un-integrated intrinsic 7.7 μm line, not a total fingerprint, so nothing there was
     actually being summed; this panel now shows the real total explicitly instead. The two
     panels together tell the literal story: sharp individual lines (1) become broad z-swept
     fingerprints (2) — "what they combine to become."
  Include **both MIPS 24 and MIPS 70** (the sketch shows both). Slowed down (fps 16→10).
  Two earlier passes are superseded: a flat shaded rectangle over `band_rest_window`'s
  2%-threshold edges (threw away the real bandpass shape), then a 3-panel version with a
  separate "total spectrum" panel and a per-band (not per-feature) redshift-axis
  fingerprint panel (didn't converge toward the reference figure's look).
- Export as a GIF/MP4 for embedding in Keynote (matplotlib `PillowWriter`, already used in
  both source notebooks) rather than relying on a live kernel during the talk.

### Speaker notes

- Lead with the physical picture, not the math: "the band doesn't know which feature it's
  looking at — it just adds up everything in its window."
- The z≈1.7–2.3 double-feature overlap (7.7 and 8.6 μm both in-band) from the original
  Animation 1 is a good concrete moment to pause on: this is *why* a single MIPS 24 μm
  number can't separate two features, and *why* the fix is tomography across many z (next
  slide) rather than a cleverer single-band trick.
- This slide is the "here's the problem" beat — the dithering slide right after is "here's
  how we solve it," so keep this one focused on diagnosis, not solution.

---

## Slide 4 — Dithering: enhanced resolving power

**Purpose**: show how staggered redshift-bin offsets across multiple runs build up dense
effective spectral sampling without losing per-bin SNR, and how the source catalog is
split to get an independent, uncorrelated error bar on top of that. Then the aside: why
this project exists and who's telling you about it.

**"Dithering" is the right word** — it's the term already established throughout this
project (`pah_dither.py`'s `DitherScheme`, "dithered z-binning" in `CLAUDE.md`), consistent
with the general astronomical usage (offset sampling patterns, as in dithered imaging).

### Bullets

- **The idea**: run the same stacking analysis multiple times with redshift-bin edges
  offset by a fraction of the bin width (e.g. 3 runs at offsets 0, Δz/3, 2Δz/3). Each run
  individually has normal per-bin source counts and SNR; *together* they interleave into
  an effective sampling several times finer than any single run's Δz — turning broadband
  photometry into low-resolution tomographic spectroscopy.
- **K-fold source partitioning** (the sketch's grid): independently, split the star-forming
  catalog into K non-overlapping subsets. Each of K stacking runs promotes a distinct 1/K
  of galaxies to the **signal** layer (`population_class=0`, "sf_keepers" — what's actually
  measured) and demotes the remaining (K−1)/K to a **nuisance** deblending layer
  (`population_class=1`, "sf_nuisance" — still modelled and subtracted so it doesn't
  contaminate the signal layer, but not the science output of that run). Quiescent
  galaxies are copied unchanged into all K catalogs. (`docs/pah-forward-model-4-brief.md`
  Objective 2.)
- Because the K runs draw their signal layer from **disjoint** galaxy subsets, their
  flux measurements at every z-bin are statistically independent — the fold-to-fold
  scatter is a legitimate, uncorrelated error bar, not just a repeat of the same sample.
- Put together: dithering buys spectral resolution, K-folding buys an honest error bar —
  and both ride on the same underlying stacking machinery, at no extra observing cost.

### Figure

**Reproduce directly in the new notebook**:

1. **"Stagger run consistency"** — port the cell you just added to §0 of
   `notebooks/2026-07-24-pah-money-plots-wider-z-bins.ipynb` (the `f_24` vs z panel,
   one subplot per mass bin, points coloured/labelled by `Run 0/1/2`): shows the three
   staggered dither runs interleaving consistently to build up dense z-sampling. This is
   the direct evidence for "enhanced resolving power," not a schematic.
2. **K-fold keep/nuisance grid** — a simple schematic (not data-driven, build directly):
   a 3×3 grid, rows = fold {0,1,2}, columns = catalog subset {A,B,C}, each cell marked
   **Keep** (that subset is the signal layer for that fold) or **Nuisance** (demoted,
   still modelled). This is redrawing the sketch's table, not reproducing a data figure.

### Speaker notes

- Two distinct mechanisms doing two distinct jobs — don't let them blur together: dithering
  is about *resolution* (denser sampling of the same signal), K-folding is about
  *independence* (a real error bar). The grid diagram is entirely about the second one.
- "Enhanced resolving power" is the right framing for the audience: broadband photometry
  becomes a crude spectrograph once you dither finely enough across enough redshift.

---

## Aside — the philosophy of Simstack, and why this talk exists

**Marked with a star in the sketch.** Not a numbered main-deck slide — a personal beat
placed here because the dithering/K-fold machinery is a natural "how does this actually
work under the hood" moment to pull back and explain the tool and the story.

### Bullets

- **Simstack's core philosophy**: every population is fit *simultaneously* — one
  `(N_pop × N_pix)` layer matrix, one linear solve (`scipy.linalg.lstsq`), deblending all
  confused populations at once rather than fitting them one at a time (`CLAUDE.md`, "Key
  Design Decisions: Simultaneous fitting"). This is *the* idea the rest of the talk's
  method rests on.
- **Personal aside**: not formally trained as an astronomer; built this rewrite
  (Simstack3 → Simstack4) partly as a deliberate project to learn to work with Claude/
  AI-assisted coding on a real research codebase.
- **Why papers are coming out now**: the COSMOS2025 catalog release is what opened up the
  current run of results — the PAH tomography program and this talk's forecast both sit
  downstream of that.
- **Tease, don't explain** — three other real results from this pipeline, previewed by
  name/thumbnail only, full versions out of scope for this talk:
  - `plot_sed_residual_grid` (`src/simstack4/analyze_cii_lines.py`) — SED-fit grids with a
    residual sub-panel per bin highlighting line excess in a chosen band; used in
    `notebooks/2026-06-15-load-json-fit-seds-redshift-sigma_sfr-CII-high-photoz.ipynb` to
    search for [CII] excess directly in high-photo-z SED residuals — the same "excess
    above the model" logic as Slide 2, applied to a different line/regime.
  - **Dust-to-gas ratio vs T_dust**, testing the Parente+2026/Sommovigo+2022 radiative-
    equilibrium relation (`create_tdust_dtg_plot`, cell 12 of
    `notebooks/2026-03-17-load-json-fit-seds-redshift-mass-sigma_sfr.ipynb`).
  - **IRX–β**: L_IR/L_UV vs UV slope β, coloured by redshift (`create_lir_luv_beta_plot`,
    cell 9 of `notebooks/2026-03-15-load-json-fit-seds-l_uv_beta-plot-sfms.ipynb`) — the
    dust-attenuation-law diagnostic.

### Figure

Small teaser strip: regenerate compact/thumbnail versions of the three figures above
directly from their source notebooks/cells (real output, just resized/simplified for a
few-second flash) — not full annotated versions, since they're not being explained here.

### Speaker notes

- Keep this genuinely brief — it's a breather beat between the technical dithering slide
  and whatever comes next, not a second talk. A minute, maybe less.
- The self-deprecating "not an astronomer" framing works *because* it's followed
  immediately by real results — let the teased figures carry the credibility, don't
  over-explain them.
- If time is short, this whole aside is the first thing to cut — it's explicitly a
  personal/motivational beat, not load-bearing for the science argument.

---

## Slide 5 — Forward Model

**Purpose, two-fold**: (1) show in pictures how the forward model works, including how
sSFR-driven evolution is engineered into it; (2) convince the audience it's real and
trustworthy — real data, a real posterior, real evidence the evolving term is needed.

### Bullets

- **The picture, in two steps** (mirrors the sketch): sharp intrinsic Drude "fingerprints"
  (Slide 1/4's spectrum) get bandpass-smoothed into broad z-space templates `K_g(z)`
  (Slide 3) — the model then fits *how much* of each template is in the real stacked data.
- **The static forward model** (per mass bin `m`, shared feature template):
  ```
  flux_m(z) = baseline_m(z) × [1 + α_m · Σ_g r_g · T_g(z)] × exp(−τ_sil · S(z))
  ```
  `α_m` = per-bin PAH amplitude (the science output), `r_g` = shared feature-group ratios
  fit globally, `T_g(z)` = the bandpass-integrated fingerprint, `τ_sil·S(z)` = optional
  9.7 μm silicate absorption. (`CLAUDE.md`, PAH Tomographic Stacking section.)
- **How sSFR evolution is engineered in** (`fit_evolving`/`fit_evolving_mcmc`,
  `pah_spectrum.py`): the amplitude and ratios are allowed to drift *within* a mass bin
  with sSFR, via shared global slopes fit by an outer optimizer wrapping the alternating
  WLS:
  ```
  ŝ_i     = log_sSFR(z_i, M_m) − s_pivot
  α_i     = α_m  · 10^(η_A · ŝ_i)
  r_g(ŝ_i) = r_g0 · 10^(η_g · ŝ_i)
  ```
  `η_A`, `η_g` are shared across all mass bins — two extra global parameters buy the model
  a within-bin evolution axis, on top of the across-bin mass trend `α_m` already carries.
- **"Is it real" evidence #1 — the decomposition**: overlay the fitted model's posterior
  (cold baseline + stacked feature contributions + 68% credible band) directly on the real
  stacked flux points. If the band tracks the data across every mass bin and redshift
  without being tuned to, that's the trust-building check — not a fit to a single number,
  a fit to the whole tomographic sweep at once.
- **"Is it real" evidence #2 — evolving vs. non-evolving χ²**: fit the *same* real pooled
  K-fold data with the evolution term off (`fit_evolving` with `evolve_amp=False,
  evolve_ratios=False` — η's pinned to 0) and with it on (`fit_evolving_mcmc`, η_A and one
  η_g per non-reference feature group free — **3 evolution parameters for this template**,
  not 2: η_A, η(6.2), η(11.3+12.7), confirmed by the actual fit, not assumed in advance).
  Report a nested F-test on the improvement — the same significance-testing approach
  already used elsewhere in this project for the crossing result
  (`docs/pah-forward-model-12-summary.md`: F(4,6)=26.27, p=0.0006). **Run 2026-07-28 in
  `notebooks/2026-07-27-lim-talk-slides.ipynb`**: static χ²_red=6.36 (dof=294) vs evolving
  χ²_red=4.84 (dof=291) — F(3,291)=31.78, p<0.0001. Real numbers; re-run with a fresh MCMC
  seed before quoting a final value in the talk, since MCMC seed/config can shift it modestly.

### Figure

**Reproduce directly in the new notebook**, three pieces:

1. **Fingerprints panel** — reuse the same Drude-spectrum → bandpass-fingerprint figure
   built for Slide 1/4 (`2026-07-25-pah-fingerprints.ipynb` §1–2); this slide is where it
   gets explicitly captioned as "what the forward model fits against."
2. **Fit to real data** — port the exact pattern from
   `notebooks/2026-07-24-pah-money-plots-wider-z-bins.ipynb` cells 8–9: run
   `model.fit_evolving_mcmc(df_pool_sm, feature_envelope="baseline", eta_prior_sigma=1.0,
   ...)` on the pooled K-fold dataframe, build `dec = evolving_flux_decomposition(evolving,
   n_draws=100)`, and plot with `plot_pah_flux_decomposition(dec, band="MIPS_24",
   mass_labels=...)` (`src/simstack4/plots.py`) — real stacked points, posterior-median
   baseline, stacked feature-group wedges, 68% credible band, one panel per mass bin.
3. **Corner plot #1 (evolution only)** — new, not a copy of an existing one. The existing
   corner plot in `2026-07-25-pah-fingerprints.ipynb` §5b shows the line ratios + τ_sil, but
   **not** the evolution parameters. Built from the *same* `fit_evolving_mcmc` run's
   `evolving["chain"]`/`evolving["names"]`, selecting `"eta_A"` and the `"eta_{group}"`
   entries (confirmed present in `names`, `pah_spectrum.py` line ~2340) — this is the
   posterior that actually demonstrates the sSFR-evolution measurement, which is this
   slide's point.
4. **Corner plot #2 (all six parameters)**, added 2026-07-29 — the full joint posterior:
   `eta_A`, `eta_6.2`, `eta_(11.3+12.7)` (evolution), `logr_6.2`, `logr_(11.3+12.7)`
   (the static feature-ratio block), and `tau_sil`. Surfaces a real, already-documented
   correlation: τ_sil trades against the 7.7↔11.3 ratio because 8.6 μm sits on the blue
   shoulder of the 9.7 μm silicate trough. Kept as a *second*, separate corner rather than
   folded into #1, so the evolution-only story stays uncluttered and this fuller one is
   available if a question calls for it.
5. **sSFR-colored residual decomposition**, added 2026-07-29 — a different, purpose-built
   figure from `notebooks/2026-07-24-pah-money-plots-wider-z-bins.ipynb` cell 12 (verbatim
   logic, not `plot_pah_flux_decomposition`): plots `stacked flux − cold baseline` (the
   residual, not the raw flux) vs z, per mass bin, with data points **colored by measured
   sSFR**. This is the one figure in the deck that visually shows the evolution result
   itself — the color gradient tracking the model is direct evidence sSFR is doing
   something, which piece #2 above (no sSFR encoding at all) cannot show. Found by
   comparing our reproduction against the source notebook's own rendered output — they
   looked different because this cell had been skipped, not because of a bug.

### Speaker notes

- Structure the slide around the two-fold motivation explicitly: first "here's how it
  works" (fingerprints → equations), then "here's why you should believe it" (decomposition
  overlay → corner plot → χ² comparison). Don't let the equations come first if the picture
  can go first — the picture is the intuition, the equations are the bookkeeping.
- The χ²(evolving) vs χ²(static) comparison is the single most persuasive number on this
  slide for a skeptical audience — it's a direct, quantitative answer to "couldn't you fit
  that without the sSFR term?"
- If someone asks what η_A actually means physically: it's how fast the PAH amplitude
  moves *within* a mass bin as sSFR moves away from the main sequence at fixed (z, M*) —
  a second axis of variation beyond the across-bin mass trend.

---

## Slide 6 — Band ratio vs stellar mass (Results begin)

**Purpose**: first results slide. The neutral/ionized PAH band-ratio trend with stellar
mass — what it shows, what it means physically, and whether it corroborates anything
already in the literature. **Do not call it a "money plot" on the slide** — that's this
project's internal nickname, not talk language.

### Bullets

- **The measurement**: `r_(11.3+12.7 μm) / r_(7.7+8.6 μm)` — the neutral-PAH-band group
  over the ionized-PAH-band group — vs stellar mass, K-fold pooled (red, fold-scatter
  error bars), individual folds shown as small grey points
  (`notebooks/2026-07-24-pah-money-plots-wider-z-bins.ipynb`, "Money plot 1").
- **The ratio declines with mass**: slope ≈ **−0.21 dex/dex** — higher-mass galaxies have
  relatively *less* neutral, *more* ionized PAH emission.
- **Why this split is the right diagnostic**: 6.2, 7.7, 8.6 μm (C–C stretch, C–H in-plane
  bend) are ionized(cation)-PAH-dominated; 11.3, 12.7 μm (C–H out-of-plane bend) are
  neutral-PAH-dominated — the canonical PAH ionization-state diagnostic (**Li & Draine
  2001, ApJ 554, 778**; **Hudgins & Allamandola 1999, ApJ 516, L41**, both added to
  `docs/pah-refs.md` this session).
- **Physical reading**: a declining neutral/ionized ratio with mass means the PAH
  population is relatively more *ionized* toward higher stellar mass — consistent with a
  harder/more intense radiation field (rising ionization parameter U) in more actively
  star-forming, higher-mass galaxies, the same U-driven framework introduced on Slide 1
  (Tielens 2008) and used for the destruction citations (Egorov+25; Leroy+23).
- **Does it corroborate anything?** Yes, with a caveat worth stating explicitly. PAHSPECS'
  *integrated* sample (z≈1.1 cosmic-noon SFGs, arXiv:2606.18230, already in
  `docs/pah-refs.md`) found *higher* 6.2/7.7 and *lower* 11.3/7.7 than local LIRGs in
  massive systems — the **same direction** as this declining trend. But PAHSPECS'
  *resolved* companion paper (arXiv:2606.18244) found the **opposite** sign *within* a
  galaxy (harder UV locally raises 11.3/7.7, via small-grain photo-destruction, not
  charging) — the two PAHSPECS papers bracket both possible physical channels, and this
  galaxy-integrated result lands on the **charge/ionization** side, not the **grain-size**
  side. State it as "consistent with the charge channel," not as an unqualified
  confirmation.
- **Known systematics, disclose all three**:
  - 8.6 μm sits on the blue shoulder of the 9.7 μm silicate trough → a common-mode
    τ_sil systematic (~±13% on the ratio) that doesn't change sign across mass bins; the
    declining trend survives it (`docs/pah-forward-model-12-summary.md` §3a).
  - [Ne II] 12.81 μm blends with the 12.7 μm PAH feature at MIPS resolution; a constant
    contamination fraction cancels in the slope, but if it scales with sSFR (which falls
    with mass) it flattens the slope — quote a **one-sided** systematic of +0.04 to +0.08
    dex/dex (can only flatten, never steepen).
  - Cross-epoch caveat: with MIPS 24 alone, the numerator (11.3/12.7) and denominator
    (7.7/8.6) are actually measured at different redshifts (z≈0.9–1.1 vs. z≈1.8–2.1);
    turning that into one "ratio" assumes the ratio is z-invariant over that span. Tested
    directly: restricting the fit to the z<2 window where both groups carry real template
    leverage reproduces the same mass slope within errors — the all-z number was
    mislabelled, not wrong.

### Figure

**Reproduce directly in the new notebook** — port `2026-07-24-pah-money-plots-wider-z-bins.ipynb`
cells 11 + 13 exactly: `bandratio_env()` computing the envelope-aware `fit_evolving` ratio
per mass bin (evolution off, `feature_envelope="baseline"`, MIPS 24 only), then the
errorbar plot — K-fold pooled (red) with fold-scatter error bars, the 3 individual folds
as small grey points, x-axis `log M*/M☉`, y-axis the neutral/ionized ratio.

### Speaker notes

- This is the pivot from "here's the method" to "here's what we found" — say that
  transition out loud so the audience recalibrates what kind of slide is coming.
- Lead with the physical statement (more ionized at higher mass) before the systematics
  list — get the result across cleanly first, then show the rigor.
- The PAHSPECS corroboration is a nice moment but needs the caveat said out loud: it's
  directionally consistent with the *integrated* measurement, and the *resolved* companion
  paper found the opposite sign for a different physical reason — don't let it sound like
  an unqualified confirmation.
- Keep the three systematics tight — a few words each, not three sub-bullets read verbatim.
  The point is "we checked, and it holds," not a methods digression.

---

## Slide 7 — Interpret: the crossing pattern

**Purpose**: the headline result. The mass slope of L_PAH/L_IR isn't fixed-sign — it flips
sign with redshift. Show it two ways (§3c and §3d are the same data, axes swapped), state
what it means, and say plainly how it differs from the standard picture.

### Bullets

- **The measurement**: L_PAH/L_IR mass slope by redshift slice —
  **+0.442 ± 0.035** (z~1) → **+0.027 ± 0.061** (z~2, consistent with zero) →
  **−0.680 ± 0.117** (z~3). The sign flip is statistically significant: nested F-test
  F(4,6) = 26.27, p = 0.0006 (`docs/pah-forward-model-12-summary.md`, the exact dataset
  §3c/§3d draw from).
- **Two views of the same result**: §3c plots L_PAH/L_IR vs stellar mass with one line per
  redshift slice (mass on the x-axis — the slope is what you're reading off). §3d plots
  the identical data with axes swapped — L_PAH/L_IR vs redshift, one line per mass bin —
  so the low-mass and high-mass lines visibly **cross** rather than asking the audience to
  infer a sign change from three separate slope numbers.
- **How this differs from the canonical picture**: the standard PAH-deficit literature
  (Engelbracht+05, Whitcomb+24, Smith+07 — already cited on Slides 2 and 7) describes a
  **fixed-sign** relationship between PAH strength and a driving property (metallicity,
  L_IR, Σ_SFR) — more of the property, consistently less (or more) PAH, at any epoch
  examined. A mass slope that **changes sign** with redshift is not something any tested
  single-epoch, scaling-relation-driven mechanism produces: metallicity supply, gas
  tracing, shattering suppression, and Σ_SFR-threshold triggers were all tested directly
  against this pattern and all fail to reproduce the sign flip (`threshold-model-fails`,
  `pah-interpretation-3-status` project history) — mass-metallicity-type relations have
  near-constant exponents across z=0–3.3 (Sanders+21), so anything built from them predicts
  a near-z-independent mass slope, not a flip.
- **Not a contradiction of the established deficit relations — an extra axis they don't
  have an opinion about.** The classic [CII]/PAH "deficit" trends are mirror images of
  each other along the L_IR/Σ_SFR/⟨U⟩ (radiation-intensity) axis *at fixed epoch*
  (`docs/forecast-lim-via-pah-1-brief.md`, "bridge" section). The crossing lives on a
  different axis — how the *mass* slope itself evolves with z — that a monotonic,
  single-epoch deficit relation is simply silent on. This is compatible with, not
  falsifying of, the standard picture; it's the part standard forecasts miss.
- **Literature comparison — appears to be a new measurement, stated carefully**: the most
  relevant modern high-z comparison, PAHSPECS (already in `docs/pah-refs.md`), doesn't
  reach the mass range or go above z~1.3, so it can't confirm or contradict this pattern —
  there's no direct precedent to cite, and no known measurement to compare against, rather
  than a confirmed absence of the phenomenon elsewhere.
- **What's still open** (one line, not a rabbit hole for this talk): whether the physical
  driver is molecule abundance (q_PAH) or radiation-field geometry (G₀/⟨U⟩) is unresolved
  with this dataset — flagged honestly as future work (JWST/MIRI), not glossed over.

### Figure

**Reproduce directly in the new notebook** — port both from
`notebooks/2026-07-24-pah-money-plots-wider-z-bins.ipynb`:

- **§3c** (cells 26–27): L_PAH/L_IR [%] vs log M*, K-fold pooled, one line per redshift
  slice (z~1/z~2/z~3, blue colour ramp), log y-axis.
- **§3d** (cells 29–30): the same data, axes swapped — L_PAH/L_IR [%] vs redshift, one
  line per mass bin (colour-coded by mass bin), log y-axis — the panel where the lines
  visibly cross.
- Use the **K-fold pooled** panel only for each (skip the "combined stack" side-by-side
  cross-check panel — that's a robustness check for the paper, not needed for the talk).

### Speaker notes

- **Lead with §3d**, not §3c — "watch these lines cross" is the most intuitive read; the
  mass-slope framing (§3c) is the more rigorous but less immediately legible version, good
  as a follow-up for whoever wants to see the slope directly.
- Say the punchline in plain words, not just numbers: *"if I only showed you z~1, you'd
  conclude PAH content rises with mass. If I only showed you z~3, you'd conclude the
  opposite. Both are correct — just at different epochs."*
- When you get to "how this differs from canonical," be concrete: name the kind of relation
  people expect (single-sign, e.g. "PAH deficit rises with L_IR") and say directly that a
  *sign-flipping* mass slope isn't in that family of models — that's the surprise.
- Do **not** get pulled into the abundance-vs-radiation-field debate live if asked — it's
  genuinely open and the honest answer is "we don't know yet, and here's specifically what
  would resolve it (MIRI)." One sentence, then move on; don't let it eat the talk.

---

## Slide 8 — The crossing, in spectral space

**Purpose**: the same crossing result as the previous slide (L_PAH/L_IR vs M*/z), now shown
as the actual fitted PAH template (Drude profiles) — one panel per mass bin, three
overlapping redshift slices per panel. **Two versions, kept side by side** (2026-07-29):
the static-ratio-per-bin version (matches the crossing headline number exactly) and the
sSFR-driven version (a more flexible model, useful as a robustness check).

### Bullets

- **Version A — static ratio per bin** (`per_bin_template`/`zslice_ratios`,
  `evolve_ratios=False`): each mass bin's feature ratios are fixed across all z; only the
  amplitude varies by z-slice. This is literally the same fit that produces the quoted
  crossing headline number — **the one to show if you're claiming that number**.
- **Version B — sSFR-driven** (`fit_evolving_mcmc(..., per_bin_ratios=True)`): both
  amplitude and ratios evolve continuously with measured sSFR. Refit 2026-07-29: found
  `eta_A` consistent with zero (`-0.10±0.11`) once each mass bin gets its own ratio pivot
  — i.e. curves nearly on top of each other within a panel. **Not a weaker result** — it's
  a genuine finding that the within-bin sSFR response is weak once mass and sSFR are
  properly disentangled (consistent with this project's prior "evolution is scatter-limited"
  finding, `pah-forward-model-7`), and it *supports* Version A's simplifying assumption
  rather than undermining it.
- Together, these two figures make an honest point: the headline crossing number is not
  sensitive to whether within-bin sSFR evolution is real or not — both models agree there
  isn't much of it.

### Figure

**Reproduce directly in the new notebook**, building on the previous slide's already-fitted
`AW_POOL`/`R_BINS`:

1. **Version B first** (as currently ordered): refit `evolving_perbin =
   model_dec.fit_evolving_mcmc(df_mjy, feature_envelope="baseline", eta_prior_sigma=1.0,
   per_bin_ratios=True, n_walkers=32, n_steps=800, n_burn=300, seed=2)` (separate MCMC run
   from the Forward Model slide's `evolving`), derive `ŝ` per (mass bin, z-slice) from
   median measured sSFR (fallback: `main_sequence_ssfr`), plot `alpha(ŝ) *
   fitted_rest_spectrum(lam, r(ŝ))`.
2. **Version A second**: `zslice_amplitude(dff, r_by_bin, aw, z_windows)`, a companion to
   `zslice_ratios` that returns the fitted amplitude `a_s` directly instead of converting
   to L_PAH/L_IR; plot `A_BINS[i,k] * fitted_rest_spectrum(lam, R_BINS[i])`.
3. Both use the same 2×2 grid layout and the same Blues z-slice colour ramp as the previous
   slide's §3c panel, for visual continuity.

### Speaker notes

- Lead with whichever version matches the number you just quoted — if you said "+0.38 at
  z~1," show Version A; don't let the two get swapped in the talk.
- If you show both: "we checked this with a more flexible model that lets the shape evolve
  too, and it agrees there isn't much within-bin evolution — which is reassuring, not a
  weaker result."
- Don't present Version B's near-zero eta as itself a finding worth dwelling on — it's a
  robustness check for Version A, not a new headline.

---

## Slide 9 — Modeling [CII]

**Purpose**: turn the PAH measurement into a [CII] LIM forecast and check it against the
real measurement (Chiang+2026).

### Bullets — the PAH → [CII] chain

- PDR bridge (physics dropped from the deck with the old Slide 1 — restate it briefly here
  since this is now its only appearance): [CII] is the dominant PDR coolant, PAHs the
  dominant photoelectric heater of the same gas, so L_CII/L_PAH ≈ constant across resolved
  regions/galaxies (Helou+01; Croxall+12; Smith+17; Sutter+19).
- Definition trap, resolved: the commonly-quoted L_CII/L_PAH≈0.1 (Croxall+12) is
  L_CII/**PAH-subset** (≈7.7 μm complex only). Against our **total**-PAH luminosity the
  correct ratio is **0.048** (range 0.027–0.069), derived from Smith+2017's
  L_CII/L_TIR=0.48±0.21% ÷ Smith+2007's L_PAH/L_TIR≈10% — fixed *before* any comparison,
  not tuned to match anything.
- Under a constant ratio, our measured crossing (Slide 7) predicts L_CII/L_IR crosses with
  mass the same way — the non-monotonic behaviour a fixed-slope forecast can't produce.
- Comparison curves are **published** L_CII–SFR relations (De Looze+14; Lagache+18) and
  SAM/halo models (Yang+22; Silva+15) pushed through the same population machinery — only
  the L-assignment differs, so it's an apples-to-apples test. (Padmanabhan+19 was in an
  earlier pass of this figure; dropped 2026-07-28.)
- **Result**: we sit above Chiang+2026 (the actual measurement) by 1.7–2.4× across
  z=0.5–3, flattest near z~1.5 — inside the combined bridge (×0.56–1.44) + Chiang (±35%)
  systematic band, so consistent, on the high side.
- The crossing's isolated effect: relative to a flat (non-crossing) L_PAH/L_IR, it shifts
  ⟨I_CII⟩ by −20% (z~0.5–1) to +111% (z~3) — a distinct shape signature standard forecasts
  don't produce.

### Figure

**Reproduce directly in the new notebook**, from `notebooks/2026-07-26-lim-via-pah.ipynb`:

1. **3-panel calibration-ladder figure (cell 12), panels (a) and (c) only** — skip panel
   (b), CO: (a) L_CII vs L_PAH (PDR bridge, ±dex scatter band); (c) L_CII/L_PAH vs sSFR,
   the intensity-drift systematic (naive constant-ratio bridge vs the drift-corrected one).
2. **Chiang comparison, mean intensity + power spectrum (cell 17, adapted 2026-07-28)**:
   (a) ⟨I_CII⟩(z) — Chiang+2026 measurement (black, measured) vs. published models
   (De Looze+14, Lagache+18, Yang+22, Silva+15 — **Padmanabhan+19 dropped**) vs. **one**
   "ours" curve (full integral, floor log M\*=9.0, matching the real catalog's coverage
   floor — the separate "log M\*>9.9 stacking-reach" curve was dropped: with matching
   floors it was numerically identical to the full-integral curve) vs. a "canonical PAH, no
   crossing" curve for contrast, also at floor 9.0; (b) predicted P(k) at z=2.5, same model
   set, no [CII] P(k) measurement exists yet to compare against.
3. **Why crossing > canonical at z>2, in absolute L_PAH (added 2026-08-02)** — the previous
   two figures are all in ratio/intensity space; this one drops the ratio and plots
   `L_PAH_ext = L_CII_ext / R_CII_PAH` directly. (a) L_PAH vs mass at z=1,2,3,4 (solid =
   crossing, dashed = canonical): at z≳2 the crossing curve sits *above* canonical at low
   mass and *below* it at high mass — the negative crossing slope pivots around log
   M\*≈10.5. (b) the population-integrated PAH luminosity density ρ_L_PAH(z)
   (`L.luminosity_density`, the same SMF-weighted integral behind ⟨I_CII⟩, with no [CII]
   conversion) — crossing sits above canonical from z~2 on. **The resolution**: the steep
   low-mass end of the Weaver+23 SMF means the low-mass boost from the negative slope wins
   the integral over the high-mass suppression — the crossing redistributes L_PAH across
   the mass function, it doesn't add PAH luminosity overall. Printed ratios: crossing/
   canonical = 0.98 (z=2.0) → 1.22 (z=3.0) → 1.46 (z=4.0).

### Speaker notes

- State the definition-trap fix explicitly and briefly — it's the kind of detail that
  builds trust. **What it is, spelled out**: the literature commonly quotes
  L_CII/L_PAH ≈ 0.1 (Croxall+2012, Sutter+2019), but their "L_PAH" there is only the
  7.7 μm complex (~49% of total PAH power, Smith+2007) — not total PAH luminosity. Our
  L_PAH is the total. Applying 0.1 to a total-PAH number silently treats the whole as if
  it were just that one subset, which **overpredicts L_CII by ~2×**. The fix: derive the
  ratio in matched units instead — L_CII/L_TIR (Smith+2017, 0.48±0.21%) divided by
  L_PAH-total/L_TIR (Smith+2007, ≈10%) — giving the correct **0.048**, about half of the
  naively-applied 0.1. This is a **definition mismatch in how the field's ratio gets
  applied**, not an error in the literature itself; say it that way, not as "a bug."
- Land on "1.7–2.4× above Chiang, consistent within combined systematics" as an honest
  result, not a triumphant exact match — the bridge band is real and should be shown.
- The crossing's isolated ⟨I_CII⟩ effect (−20% to +111%) is the sentence that connects
  this slide back to the whole talk's thesis: bring it up explicitly rather than letting
  it sit silently in a figure.

---

## Slide 10 (backup) — Modeling CO

**Status: backup, likely no time for it.** The CO equivalent of Slide 8. Include only if
time allows or if CO comes up in Q&A.

### Bullets — the PAH → CO chain

- L_PAH tracks L′_CO out to z~4 (Cortzen+19; **Shivaei & Boogaard 2024**, A&A 691, L2 —
  14 z=1–3 CO-detected galaxies + a z=0–4 literature compilation, 0.21 dex scatter) — PAH
  luminosity is effectively a molecular-gas-mass tracer. (Note: a *different* Shivaei+24
  paper, SMILES/arXiv:2402.07989, is cited elsewhere in this deck for q_PAH–metallicity —
  same first author, unrelated topic, don't conflate them.)
- Two ways to carry CO, and the gap between them *is* the assumption: **"gas"** route
  (L′_CO = L_IR/70, Sargent+2014 MS locus) uses no PAH input and carries **no crossing**;
  **"PAH"** route (L′_CO = L_PAH/(L_PAH/L′_CO), assumed constant) is what propagates our
  measured crossing into CO. The two differ by only 0.63–0.84× — for CO the crossing is a
  ~20–40% effect, much smaller than for [CII]. **CO is not where this result has leverage;
  [CII] is.**
- **Mean intensity agrees well**: our ⟨T_CO(1–0)⟩(z) tracks ASPECS LP, COLDz, and COMAP
  ρ(H₂)-derived points across 0.3<z<4, within a factor ~2, comfortably under COMAP's 95%
  upper limit.
- **Shot power: isolated to COPSS, not a uniform deficit.** Corrected 2026-07-31 — the
  previously quoted "~90× below COPSS, ~5–9× below mmIME on all three transitions" didn't
  reproduce from the code (nor from its own source notebook's printed output) and was
  wrong. Recomputed directly (`lim_via_pah_helpers.py`, `sig_co_dex=0.21` from Shivaei &
  Boogaard 2024): our CO(1–0) shot power under-predicts **COPSS II by ~15×** at z=2.8, but
  the **mmIME ladder comparison is actually consistent** — CO(2-1) ours/data=1.7×
  (over-predicts), CO(3-2)=0.70×, CO(4-3)=0.80× (all within mmIME's own error bars). So the
  bright-end deficit this population model has is specific to the COPSS CO(1-0) point, not
  a general shot-power shortfall across the ladder.
- Caveat: COPSS II is itself a 2σ detection in tension with COMAP's own limits — read the
  ~15× gap as bounded by that disagreement, not as a clean model failure. Given mmIME
  (higher-J, different z) is consistent, the honest read is "COPSS may be anomalously high,
  not that our model is missing a whole population of bright emitters."
- **CIB closure check** (does the L_IR route over-predict?): running the forecast's own
  ingredients (Weaver+23 star-forming SMF × Speagle+14 main sequence × greybody at the
  measured T_dust(z)) through the emissivity integral recovers **78%** of the 250 μm
  cosmic infrared background — landing between the catalogue-limited **Viero et al. (2013,
  ApJ 779, 32)** stack (67%, the like-for-like match to our own stacking method) and
  **Viero et al. (2015, ApJL 809, L22)**'s smoothed total (94%, whole population including
  faint companions) — exactly where a full-SMF integral should land. No hidden
  normalization factor sitting in the forecast's amplitude.

### Figure

**Reproduce directly in the new notebook**, from `notebooks/2026-07-26-lim-via-pah.ipynb`:

1. **CO comparison (cells 20–21)**: 3-panel figure — (a) ⟨T_CO(1–0)⟩ vs z against
   ρ(H₂)-derived points (ASPECS LP, COLDz, COMAP S2); (b) CO(1–0) shot power against COPSS
   II and the COMAP upper limit; (c) our model converted up the excitation ladder
   (r₃₁=0.84±0.26) to compare against mmIME on mmIME's own higher-J transitions. Both
   "gas" and "PAH" routes shown throughout.
2. **CIB closure, stripped down** (from cell 42, panel (a)): **only the §8c curve**
   (SMF × main-sequence L_IR, the forecast's own route) against the Fixsen+98 FIRAS CIB
   curve and the Viero+2013 CIB data points — drop the 8a (stacked-flux×counts) and 8b
   (through-SED) routes shown in the full notebook panel; this slide only needs "does the
   forecast's own amplitude break the CIB," not the full closure-test methodology.

### Speaker notes

- Two independent, separable results — don't blur them: (1) CO's *mean* intensity checks
  out everywhere, and its *shot noise* checks out too against mmIME — the one outlier is
  COPSS II's CO(1-0) point (~15×), which is itself a marginal 2σ detection in tension with
  COMAP; (2) separately, the whole forecast's amplitude closes against the measured CIB,
  so there's no hidden normalization inflating any of this — including the [CII] numbers
  from the previous slide.
- If asked "so is your forecast right or not," the honest answer in one line: "the shape
  and mean amplitude check out against independent data, and so does the shot power at
  every higher-J transition mmIME provides — the one discrepancy is a single, disputed
  COPSS point, not a systematic population-model shortfall."
- Given this is a backup slide, don't over-invest prep time here — if it comes up, the
  CIB-closure bullet is the strongest, most self-contained thing to lead with.

---
