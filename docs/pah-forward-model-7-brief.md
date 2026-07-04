# PAH Forward Model — Branch 7 Brief

**Goal**: turn a robust but not-yet-convincing measurement — PAH-to-continuum ratios
increase with stellar mass, and this direction survives the branch-6 α re-measurement — into
a defensible physical narrative, positioned against the literature this project cites.
Companion exploratory notebook (real code, executed against the 3 K-fold stacking runs):
`notebooks/2026-07-01-pah-forward-model-letter-candidates.ipynb`.

---

## 0. Definitions, stated once

A PAH **deficit** means suppressed PAH emission relative to the continuum/L_IR — a **low**
value of the PAH-to-continuum or L_PAH/L_IR ratio. Both measured ratios (EW-style
`A=α_m/C_m`, and `L_PAH/L_IR`) **rise** with stellar mass.  This is easy to invert by accident.

---

## 1. The physical picture

The literature usually cited here (Smith+07, Galliano+21, Egorov+25, Leroy+23) measures the
PAH deficit **at fixed stellar mass**, as a function of sSFR / radiation-field intensity
(harder fields at higher sSFR destroy small grains). Our measured trend is a **stellar-mass**
axis result. These are different independent variables, and comparing them directly is why a
convincing narrative has been hard to land — not a shortage of results.

Two distinct mechanisms could drive the mass axis:

- **Abundance / metallicity**: more metals at high M\* (mass–metallicity relation) → more PAH
  carbon → higher PAH-to-IR fraction, with the PAH template's internal **band ratios staying
  fixed** across mass.
- **Ionization state** (Tielens 2008): the balance between ionized-PAH bands (6.2, 7.7, 8.6
  µm) and neutral-PAH bands (11.3, 12.7, 17 µm) shifts with the radiation environment — e.g.
  a rising AGN fraction at high M\* — with the **band ratios shifting systematically** with
  mass.

Three results below, all already computable from the existing K-fold stacking runs (no new
stacking needed): §1a tests the mass axis, §1b tests the fixed-mass sSFR axis, and §1c tests
whether those two are actually the same mechanism (they are not shown to be).

### 1a. Band ratios shift with mass — a mechanism signal, not just an amplitude trend

Every forward-model fit run so far shares one global `r_g` across all mass bins — only the
overall amplitude was ever allowed to vary. Relaxing that (candidates notebook §4: refit
`fit_shared` independently per mass bin, independently per K-fold split) gives the
12.7 µm-to-6.2 µm ratio:

| log M\*/M☉ | fold 0 | fold 1 | fold 2 |
|---|---|---|---|
| 10.0–10.5 | 31.9 | 73.5 | 111.9 |
| 10.5–10.8 | 15.7 | 17.8 | 4.8 |
| 10.8–11.1 | 4.2 | 4.2 | 2.8 |
| >11.1 | 0.76 | 1.87 | −0.10 (unconstrained) |

The neutral-PAH feature falls from ~30–110× the ionized-PAH feature at low mass to roughly
parity at high mass, **monotonically in all three independent K-fold catalog splits**. This
points to an ionization-state mechanism, not pure abundance.

**With a fold-ensemble error bar** (candidates notebook §4a — mean ± scatter/√3 across the 3
folds, this project's standard error convention, not a new or looser one): 72.4±23.1 →
12.8±4.0 → 3.7±0.5 → 0.8±0.6. Every adjacent mass bin is separated by **2.2–4.0σ**, and the
trend still spans two orders of magnitude top to bottom — it does not evaporate once an error
bar is attached. **This is a reasonable result to build the paper's mechanism case around.**
Remaining caveats before it's fully citable: only 3 folds means the scatter estimate itself is
imprecise (a std from n=3 draws carries large uncertainty), and the highest-mass bin is the
shakiest point (one fold gave an unphysical negative ratio). **Checked and held up** (§4d):
an unlinked feature-group scheme and an unsmoothed baseline — every variant still declines
monotonically with 1.9–4.7σ between adjacent bins. Still open: Tier A/B-only, and a
bootstrap-over-sources error as an independent cross-check on the fold-ensemble number. None of
that is expected to change the direction — this is about tightening a result that already
looks solid, not testing whether it exists.

**The lowest-mass bin has its own instability, and the cause is now identified** (candidates
notebook §4c). The independent free-amplitude cross-check (§4b) returns a significantly
negative 6.2 µm amplitude there (−3.20±0.42, ~7.7σ from zero — unphysical for an emission
feature). Tested and ruled out: splitting the linked 7.7+8.6 group into two independent
amplitudes doesn't fix it (6.2 µm stays negative, −3.93±0.42, if anything slightly worse; the
6.2-vs-other-group kernel correlations are modest, |corr| ≤ 0.37, not the near-unity
collinearity that would indicate a real grouping-driven degeneracy — 7.7 and 8.6 *are*
strongly correlated with each other, ~0.6, matching why they're linked in the first place, but
that's a self-contained 7.7-vs-8.6 issue, not the source of 6.2 going negative). What does
matter: swapping the smoothed main-sequence baseline for the raw per-bin one collapses the
significance from −7.7σ to −0.7σ (consistent with noise). This points at the smoothed
T(z,M\*)/logA(z,M\*) relation fitting the lowest-mass bin poorly, not a feature-grouping
problem — reinforces, with direct evidence, the "check against an unsmoothed baseline" item
already above.

*On relaxing shared ratios in general*: doing this for a **static**, cross-mass-bin comparison
(as above) costs nothing extra in parameters — it's exactly the same as fitting 4 mass bins
independently instead of pooling them, and each bin has ~70–100 tomographic points, a large
lever arm (2 dex in mass) to constrain 2 extra ratios per bin. It gets expensive only if you
also let each bin's ratios *evolve* with z/sSFR independently — that needs each bin's own,
much shorter within-bin z/sSFR range to constrain both a ratio vector and an evolution rate
simultaneously, and is a real identifiability risk. §1b below hits exactly that tension.

### 1b. The classic deficit signature — but only in the highest-mass bin

At fixed stellar mass, higher-z galaxies sit at higher sSFR (main-sequence evolution), so the
sharpest test of the radiation-field mechanism is whether `L_PAH/L_IR` **falls** with z/sSFR
*within* a single mass bin (candidates notebook §5b, on the correctly-normalized L_PAH/L_IR,
not the EW/continuum ratio — see `pah-signflip-diagnosis` for why those answer different
questions).

A parametric version of this (`fit_evolving` per mass bin, amplitude drifting as a power law
in sSFR) fits *worse* than assuming no evolution at all in every bin, and produces implausibly
wide point-to-point ratios — a single power-law slope chasing individual noisy points,
amplified by Tier C's known Eddington-bias-prone per-point fits. A non-parametric version
(Tier A/B only, weighted by `n_sources`, median-split on sSFR within each mass bin — no
evolution-rate parameter to over-fit) gives a cleaner and more interesting answer:

| mass bin | low-sSFR half | high-sSFR half |
|---|---|---|
| 10.0–10.5 | 4.9–5.0% | 8.6–8.7% |
| 10.5–10.8 | 5.0% | 6.3–6.4% |
| 10.8–11.1 | 6.2–6.3% | 6.6–6.7% |
| >11.1 | 6.8–6.9% | 5.4–5.9% |

L_PAH/L_IR **rises** with sSFR in the two lower-mass bins (opposite of the classic deficit),
is flat in the third, and only **falls** with sSFR — the classic deficit signature — in the
highest-mass bin. Real sSFR (`lp_sSFR_med`) and the (z, M\*) main-sequence proxy give nearly
identical splits, so within a fixed mass bin this is effectively a redshift trend, not an
independent sSFR lever arm. This uses each bin's *fixed* ratios from §1a — it shows the overall
amplitude responding to sSFR, and says nothing on its own about whether the ratios themselves
shift with sSFR.

This is a first pass — no propagated errors, a single two-way split, Tier A/B only (~50–70
points/bin) — not a citable result yet.

### 1c. Do the ratios themselves shift with sSFR at fixed mass? Tested directly — no.

§1a and §1b are two separate results (ratios shift with mass; amplitude, at each bin's fixed
ratios, shifts with sSFR in a mass-dependent way). Whether the *ratios* shift with sSFR at
fixed mass — the claim that would actually connect them through a shared ionization mechanism
— is a distinct, testable question. Splitting each mass bin into low/high real-sSFR halves and
refitting `fit_shared` with the ratios floating (candidates notebook §5c) does **not** support
it: several halves return enormous ratio uncertainties (one is 175 ± 273 on ~26 points trying
to constrain 2 ratios plus 2 per-bin parameters), the direction is inconsistent across mass
bins, and the one marginally significant shift (highest mass bin: 12.7/6.2 goes from
1.15±0.09 to 4.52±1.39, low to high sSFR) moves **opposite** to the simple "harder field → more
ionization → lower neutral/ionized ratio" story. **We have not shown that band ratios track
sSFR/radiation environment** — only that they track mass (§1a), and that overall amplitude at
fixed ratios tracks sSFR in a mass-dependent way (§1b). Do not connect these into one mechanism
narrative without a dedicated, better-powered test.

### 1d. Literature check (2026-07-01): the "obvious" explanation doesn't fit — that's useful

Checked directly (advisor was unavailable this session; this is my own analysis, verified
against real sources via web search, not from memory alone — see `docs/pah-refs.md` for full
citations): the most standard explanation a referee would reach for first is the well-known
**low-metallicity PAH deficit** (Engelbracht et al. 2005), acting through the
mass–metallicity relation. A recent, more detailed version of that result (Whitcomb et al.
2024) measures exactly the kind of quantity we do — a long-to-short-wavelength PAH band ratio
— as a function of metallicity, and finds long-wavelength bands (17 µm) decline steeply below
~2/3 solar metallicity while short-wavelength bands dominate more at low metallicity.

**Via the standard mass–metallicity relation, this predicts the opposite mass trend from what
we measure.** Low mass → low metallicity → their mechanism suppresses the long-wavelength
band → long/short ratio should be *low* at low mass. We find long/short (12.7/6.2) is
*highest* at low mass (72×) and *lowest* at high mass (0.8×) — the reverse. This isn't dodged
by a mass-range technicality either: even our highest-mass bin at z~2–2.5 sits close to
Whitcomb+24's ~2/3-solar threshold, so their mechanism's regime does overlap with our sample.

This is a genuinely useful negative result, not a dead end: it argues *against* the most
standard alternative explanation for our mass trend, which strengthens (by elimination, not by
direct confirmation) the case that something else — plausibly AGN incidence, which does rise
with M\*, and which Xie & Ho (2022) show can shift PAH band ratios differentially in quasar
hosts, similar in kind to what we see — is a better candidate than a straightforward
metallicity/grain-size story. **This tension should be stated explicitly in the paper's
discussion section**, not smoothed over; "our result runs opposite to the low-metallicity
grain-size effect seen in dwarf galaxies, disfavoring that mechanism at the massive end and
pointing toward AGN or another mass-specific process" is a stronger, more specific claim than
a vague appeal to "ionization state (Tielens 2008)" alone.

---

## Objective 1 — §1a is the lead result; finish making it airtight

§1a (band ratios fall ~2 orders of magnitude from low to high mass, fold-robust, 2.2–4.0σ
between every adjacent bin with the project's standard fold-ensemble error) is the strongest,
most directly mechanistic result of this branch and the one to build the paper's case around.
Remaining before it's fully citable:

- **Done: `fit_shared` itself (not just the free-amplitude cross-check) rechecked against
  unlinked feature groups and an unsmoothed baseline** (candidates notebook §4d). Both hold up:
  every variant (current linked+smoothed, unlinked+smoothed, linked+unsmoothed) declines
  monotonically across all 4 mass bins, with every adjacent-bin step separated by 1.9–4.7σ (one
  step dips just under 2σ under the unsmoothed baseline; nothing reverses or flattens).
  Unlinking barely moves the `fit_shared` numbers at all. The unsmoothed baseline actually
  *tightens* the lowest-mass bin's error (23→5) rather than exploding it — the opposite of what
  happened to the free-amplitude method in §4c, confirming that instability was specific to
  that side-cross-check's parametrization, not a property of the method the headline is built
  on. **This is the strongest evidence yet that §1a is not an artifact of one baseline choice
  or one feature-grouping decision.**
- **Bootstrap-over-sources error**, as an independent cross-check on the fold-ensemble number
  (only 3 folds makes that scatter estimate imprecise on its own) — still open.
- Check it survives **Tier A/B-only** (currently includes Tier C) — still open.
- Look more closely at the **highest-mass bin** specifically (one fold gives an unphysical
  negative ratio) — is this instability, or a real sign the effect saturates/reverses there?
  Still open, and now the most likely remaining weak point given everything else has checked out.
- MCMC over `r_g,m` (not just the WLS point estimate + fold scatter) would give a proper joint
  posterior and a cleaner significance statement than the adjacent-bin σ comparison above.

§1b (fixed-mass sSFR trend) and §1c (ratios don't demonstrably shift with sSFR) stay as
secondary, exploratory findings — real, and worth a sentence in the paper's discussion, but
**do not build the mechanism narrative around them**, and do not claim §1a and §1b share a
single ionization mechanism; §1c tested that directly and it didn't hold up at current sample
sizes. §1a stands on its own as a mass-axis result and doesn't need §1b/§1c to be interesting.

## Objective 2 — Close the remaining branch-6 items — **DONE 2026-07-02**

Both items closed in `notebooks/2026-07-02-pah-narayanan-confrontation.ipynb` (built by
`notebooks/build_pah_narayanan_confrontation_notebook.py`):

**2a. RESOLVED — the discrepancy is the baseline treatment.** 16-variant grid at matched α=2
(sample × baseline × fit path × L_PAH definition). The old-like configuration (single fold,
raw baseline, 24-only `fit_shared`, in-band L_PAH) reproduces June-28's flat slope (+0.035 vs
+0.019). Raw→smoothed baseline moves the slope +0.29 dex/dex; every other factor ≤0.03. All 8
smoothed variants rise (+0.13..+0.21); raw variants are flat-or-negative and sample-unstable
(sign flips between fold0 and pooled). June-28's flat value is superseded as a raw-baseline
(Tier-C noise in `C_m`) artifact. Remaining paper-level cross-check: Tier-A/B-only raw
baseline.

**2b. DONE — citable fold-ensemble slope.** L_PAH/L_IR mass slope = **+0.236 ± 0.076 dex/dex
(3.1σ) at free α** (per-fold α_wien 3.17/3.27/2.40); +0.135 ± 0.136 (1.0σ) at pinned α=2.
Pooled at α=2.86: +0.212, normalization 7.99% at pivot logM\*=10.75. Quote free-α as primary,
α=2 as conservative bound.

## Objective 3 — External anchors (bibliography gap partly closed; two items still need a real read)

**Done (2026-07-01)**: the mass–metallicity–PAH literature gap flagged in the previous version
of this brief is filled — `docs/pah-refs.md` now has Engelbracht et al. (2005) and Whitcomb
et al. (2024) for the metallicity/grain-size mechanism, and Xie & Ho (2022) added to the AGN
section. §1d works through what they imply for our result (a real tension with the
metallicity mechanism, not confirmation) — this was found via live web search this session,
not from memory, but none of these three have been read in full; the summaries used are
good enough to state the tension exists, not to quote exact numbers from them in the paper.

**User decision 2026-07-02: no AGN-based interpretation** anywhere in the analysis or paper
framing while the sample's AGN fraction is unknown. §1d's by-elimination argument stands (the
metallicity mechanism is in tension with §1a), but its "AGN is the better candidate"
conclusion is parked — the Xie & Ho follow-up below is deprioritized accordingly.

**DONE 2026-07-02 — Narayanan et al. (2026), arXiv:2606.20809, read + confronted**
(`notebooks/2026-07-02-pah-narayanan-confrontation.ipynb` §5–6; entry expanded in
`docs/pah-refs.md`). Key facts: shattering-driven in-situ PAH formation; inverse q_PAH–f_mol
(their Fig 7); q_PAH–Σ_SFR anti-correlation (Fig 9); q_PAH 5×10⁻⁴→10⁻² over z=4→0 (Fig 5);
L_PAH/M_PAH ∝ G₀ decoupling; **no q_PAH(M\*) at fixed z and no band-ratio predictions
published** — the answer to this item's original ask is "the paper doesn't contain it," so we
derived it. The two mass-axis channels their mechanism implies: density chain [−0.35, +0.09]
dex/dex (f_mol is saturated 0.90–0.99 at cosmic noon so the Σ_SFR channel dominates,
negative); enrichment/PZR chain [−0.10, +0.55]. Confronted with 2b's measurement:
**+0.236±0.076 sits 1.8σ above the density chain's upper edge, inside the enrichment band**
— the first observational constraint on this model along the mass axis at cosmic noon. The
low-mass PAH deficit behaves like an abundance (PZR-like) effect, not dense-ISM suppression.
(At pinned α=2 the discrimination weakens to 0.3σ — it rests on the branch-6 α measurement.)
Their sims *could* also predict the §1a band-ratio trend (per-galaxy pahfit band luminosities
and a size-dependent ionization prescription exist in their pipeline) but the paper doesn't
report it — concrete follow-up/contact item; J.-D. Smith and C. Whitcomb are coauthors, the
same group behind the §1d metallicity band-ratio tension.

Still open, need a direct read (not just search-result summaries) before citing specifics:
- **arXiv:2606.18244** (PAHSPECS, JWST MIRI MRS spectroscopy z~1–3): extract their PAH
  EW/deficit vs M\* and any band-ratio-vs-mass information, as an independent,
  continuum-assumption-free cross-check on both §1a and §1b.
- **Xie & Ho (2022)**: confirm their exact 12.7 µm (or nearest reported band) prediction
  before using it to support or oppose §1a's specific ratio — the summary used this session
  only confirmed 6.2/7.7, 8.6/7.7, and 11.3/7.7 numbers, not 12.7 directly.
- **Whitcomb et al. (2024)**: confirm the tension in §1d holds up against their actual
  quantitative M*/Z-binned data (not just the qualitative summary used here) before stating it
  as a settled discussion point in the paper.

sSFR (SFR/M\*) is the correct available proxy for radiation-field intensity in this dataset:
COSMOS2020 (`prepare_cosmos2020_catalog.py`) carries `lp_mass_med`, `lp_SFR_med`,
`lp_sSFR_med` but no size-derived Σ_SFR (that column exists only in COSMOS2025, not in use
here) — and it isn't a downgrade, since Smith+07 and Galliano+21 plot PAH fraction against
sSFR/L_IR themselves, not literal Σ_SFR.

---

## Money plots reproduced standalone — 2026-07-03

`notebooks/2026-07-03-pah-money-plots.ipynb` (built by
`notebooks/build_pah_money_plots_notebook.py`; user-directed) regenerates the branch's two
headline figures from the 3 K-fold stacks, with verbatim data cells and printed
reproduction checks. PNGs: `pah_money_bandratio_vs_mass.png`,
`pah_money_narayanan_confrontation.png`.

- **§1a band-ratio-vs-mass figure: strict MATCH.** All four fold-ensemble values reproduce
  (72.4±23.1 → 12.8±4.0 → 3.7±0.5 → 0.8±0.6; adjacent separations 2.55/2.23/3.97σ). Expected
  — its path (24 µm-only `fit_shared`) is untouched by the 2026-07-03 fixes.
- **Narayanan confrontation figure: headline number SURVIVES the multi-band normalization
  fix.** The 2b slope runs through `fit_with_alpha` (24+70), i.e. through the fixed
  `_evolving_data`; re-derived fold ensemble = **+0.234 ± 0.077 (3.0σ)** vs documented
  +0.236 ± 0.076 (per-fold α_wien 3.23/3.27/2.44 vs 3.17/3.27/2.40) — the z<0.8 70 µm rows
  carry too little weight to move it. Verdict unchanged: 1.8σ above the density chain's upper
  edge, inside the enrichment band. Pooled normalization at the pivot shifts slightly
  (7.99% → 7.30%; pooled α_wien 2.86 → 2.85, slope +0.212 → +0.214).
- **§2b/§3b (user-directed follow-up, executed same day): envelope-aware re-derivations.**
  (a) *Band ratios*: `fit_shared`'s constant-flux feature term makes each group's amplitude
  absorb the mean dimming envelope of ITS OWN redshift window (12.7 µm measured at z≈0.9,
  6.2 µm at z≈2.9), so the §1a absolute ratios were window-envelope contaminated by a ×4.3–6.5
  factor. Intrinsic (envelope-aware) values: **13.4±3.8 → 1.96±0.52 → 0.88±0.06 → 0.19±0.14**;
  the mass trend, monotonicity and adjacent-bin separations survive (3.0/2.1/4.5σ), but
  "parity at high mass" moves to the 10.8–11.1 bin and the >11.1 bin sits at 0.19. **Use the
  §2b values for any literature band-ratio comparison (Xie & Ho, Whitcomb); §1a's table is the
  internal-trend version only.**
  (b) *L_PAH/L_IR slope*: envelope-static estimator +0.227±0.185 (fold-unstable — the static
  envelope model is mis-specified, per the §7 evolving-fit result); envelope+η_A estimator
  +0.131±0.055 (η_A runs to ~2 in these free-α fits). Slope stays positive under every
  estimator (+0.13…+0.23), but the density-chain tension is estimator-dependent: 1.8σ
  (original) → ~0.7σ (envelope-aware). Quote the estimator spread as a systematic alongside
  α_wien.

## Evolving-template study, part 2 — realism calibration + REAL-DATA comparison (2026-07-03)

User review of the first executed notebook raised three points; all three are now addressed in
the same notebook (regenerated in place):

1. **Sim flux scale was wrong** (f24 ~10× off, no decline with z): the first version had no
   flux envelope — band fluxes were comoving-luminosity-like while real stacked f24 falls
   ~7–10× over z=0.5–3.5. Fixed: `TruthSpectrum.flux_envelope` calibrated to the real 3-fold
   smoothed f24_cold(z, M\*) (quad-in-z + linear-in-mass, coefficients cross-checked live in
   §7 to 4 decimals). Sim now reproduces the measured mJy scale, z-shape, SNR(z) profile
   (~19 → ~2), and mass ordering. **Companion model fix**: `feature_envelope="baseline"` in
   `fit_evolving`/`fit_evolving_mcmc` — features must dim with the source; the old
   constant-flux feature term lets the envelope leak into the evolution slope (~−0.85 dex on
   the sim's η_A; rung L2n demonstrates it).
2. **Truth off-center in the MCMC corners**: §4c posterior-predictive check — a replica drawn
   from the fitted model itself refits centered (|pull| ≲ 2), so the sampler/profiled
   likelihood are self-consistent; the injected-truth offsets are the z_mid-vs-p(z) kernel
   approximation floor (the fit modulates the smeared kernel at z_mid; the truth evolves
   continuously across each bin's membership). At the real depth the floor is subdominant to
   noise; the upgrade path (integrate the sSFR modulation + envelope over p(z) inside the
   kernel) is only needed well above real SNR.
3. **Static vs evolving ON THE REAL DATA** (§7, 3 K-folds, letter-verbatim data cells,
   reference group moved to 7.7+8.6 µm): with the envelope-aware feature term, ONE shared
   amplitude slope buys Δχ² = 253 pooled (÷χ²_red=3.5 → 72 scatter-rescaled), **consistently
   in all 3 independent folds** (Δχ² = 65/44/52; η_A = 0.895/0.831/0.807 → fold ensemble
   **+0.844 ± 0.026**), far above the scatter-null calibration (same sim, zero evolution,
   scatter tuned to the real χ²_red). η_A is **positive** — feature/continuum rising with
   sSFR within each mass bin — opposite in sign to the branch-5 railed −2.37, which the L2n
   rung identifies as envelope absorption (the old fits had no feature dimming). With ratio
   evolution also free, η_A drops to +0.47±0.08 (amp↔ratio degeneracy; quote the model, not
   just the number). Defensible claim: *evolution is required to fit the stacks*; its physical
   decomposition still needs the α systematic quoted alongside (§6 tilt test).

## Evolving-template MCMC flexibility study — DONE 2026-07-02 (simulation)

`notebooks/2026-07-02-pah-evolving-template-mcmc-simulation.ipynb` (built by
`notebooks/build_pah_evolving_mcmc_notebook.py`; user-directed): inject a realistically
evolving truth (amplitude + line-ratio drift with sSFR(z, M\*), mass trend +0.35 dex/dex,
hot-dust MIR continuum) through the real scheme geometry (dz=0.15 × 3 staggers, 4 mass bins,
~140 sources/bin, bootstrap-calibrated noise, 24+70 µm), then MCMC the forward model at
increasing flexibility with the new `PAHSpectrumModel.fit_evolving_mcmc` (η_A, η_g, log r
sampled; per-bin (C_m, α_m) profiled; `per_bin_ratios=True` = §1a-style ratio freedom).
Headline figure: f₂₄(z) per mass bin with shaded posterior feature-group contributions
(`evolving_flux_decomposition` + `plot_pah_flux_decomposition`). Findings:

- **Anchor the reference feature group on 7.7+8.6 µm** — its slope (reported as η_A) is then
  recovered to ±0.01–0.04 at real depth; referenced to 6.2 µm instead, η_A floats and comes
  back consistent with zero against an injected +0.4.
- Per-group total slopes e_g = η_A+η_g are the identifiable quantities: 7.7+8.6 tight;
  6.2/11.3/12.7 recovered with 0.1–0.3 dex errors; 16.4+17.0 rails (no bandpass leverage).
- 70 µm at real depth (per-point SNR ~0.2) only bounds the 16.4+17.0 posterior; the
  degeneracy-breaking claim from equal-noise sims needs depth.
- Per-bin ratios (L3/L4, up to 21 dims) still converge; weak-group posteriors inflate ~2×.
  At 33× depth a kernel-systematics floor appears (χ²_red ≈ 2.2, e_g biases ≈ 0.1 dex).
- **Two structural fixes landed**: (a) multi-band evolving fits (`_evolving_data`) now
  normalize all bands by one per-bin scalar — the old per-band medians forced equal 24/70
  continuum levels through the shared C_m (mis-specified; masked in tests by identical
  baseline columns; single-band fits, i.e. all headline results, numerically unchanged);
  (b) `TruthSpectrum.mir_plaw_amp` adds a hot/VSG MIR continuum — without it simulated
  24 µm flux is pure PAH and C_m is unidentifiable. Guards: 2 new tests in
  `test_pah_evolution_recovery.py` (8 total).

## Not being pursued this branch

- **T_dust correction.** An earlier two-pass refit found a modest shift (+0.81 K at z<3).
  Reference for later: `letters/02-corrected-tdust-companion.md`.
- **Referee-defense writeup.** Depends on which mechanism narrative (Objective 1) and which
  normalization (Objective 2) end up in the paper; premature until both settle.

## Objective 4 — Talk + paper figure set (deferred until the mechanism question locks)

Reusable regardless of outcome, from `notebooks/2026-07-01-pah-forward-model-letter.ipynb`:
the method figure (MIPS 24 µm bandpass swept across rest-frame PAH features vs z), the
α-measurement robustness panel (§5–7, "α is not 2"), and the model-overlay + deconvolved
pseudo-spectrum (§8/§8b). Everything downstream — which ratio plot is the headline figure,
whether a band-ratio-vs-mass panel or a fixed-mass sSFR panel appears, whether both do —
waits on Objective 1.

---

## Config and data notes

- Dataset: `cosmos20_stacking_20260630_{193627,211122,222635}.json` — 3 disjoint K-fold splits
  (`cosmos2020_PAH_split{0,1,2}of3`), each also offset in z-binning (dither). Config:
  `config/cosmos20_PAH_dithered_3cats.toml`.
- 4 science mass bins (10.0–10.5, 10.5–10.8, 10.8–11.1, >11.1); 8.5–10.0 stays an unanalysed
  low-mass nuisance layer.
- `FEATURE_GROUPS = [[0],[1,2],[4]]` (6.2 | 7.7+8.6 | 12.7, 11.3 blind) — keep consistent for
  any slope comparison.
- `lp_sSFR_med` (real per-source median sSFR) is already stored in every stacked bin's
  `bin_properties`; no new stacking runs are needed to use it.
- Per-fold refits must build each fold's dataframe via `build_pah_spectrum_df([single_wrapper], ...)`
  (one wrapper at a time), not by filtering a pooled dataframe down to one `run_id` afterward —
  `PAHSpectrumModel._prepare`'s scheme/run_id reconstruction assumes the dataframe came from the
  wrapper list actually being fit, and silently mis-indexes (or raises `IndexError`) otherwise.
