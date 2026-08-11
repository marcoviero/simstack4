# PAH Reference Papers

Key literature for the PAH tomographic stacking analysis and letter.
One-line role for each — follow the ADS link for full text.

---

## Physics foundations

**Tielens (2008)** — *Interstellar Polycyclic Aromatic Hydrocarbon Molecules*
ARA&A 46, 289.
[ADS](https://ui.adsabs.harvard.edu/abs/2008ARA&A..46..289T/abstract)
The canonical review. Single-photon heating physics; how band ratios (6.2/7.7/8.6/11.3)
encode grain ionization state and size; ionization parameter U as the primary control
variable for PAH abundance; physical basis of the PAH deficit. Read §1–4 first.

**Li & Draine (2001)** — *Infrared Emission from Interstellar Dust. II. The Diffuse
Interstellar Medium*
ApJ 554, 778.
[ADS](https://iopscience.iop.org/article/10.1086/323147)
Quantitative dust/PAH IR emission model establishing that 6.2, 7.7, 8.6 μm (C–C stretch)
scale with the ionized-PAH population and 11.3 μm (C–H out-of-plane bend) with the neutral
population — the physical basis of the ionized/neutral band-ratio diagnostic used in
Slide 7 (band ratio vs M*). Note: notebook code comments in this project cite this as
"Draine & Li (2001)" — the correct author order is **Li & Draine**; worth fixing those
comments if noticed in passing.

**Hudgins & Allamandola (1999)** — *Interstellar PAH Emission in the 11–14 Micron Region:
New Insights from Laboratory Data and a Tracer of Ionized PAHs*
ApJ 516, L41.
[ADS](https://ui.adsabs.harvard.edu/abs/1999ApJ...516L..41H/abstract)
Laboratory (matrix-isolation) spectroscopy establishing 11.3 μm (and the 12.7 μm/[Ne II]-
adjacent region) as neutral-PAH tracers, complementing Li & Draine (2001)'s modelling.
Companion citation for the same band-ratio diagnostic.

---

## PAH deficit — local empirical anchors

**Smith et al. (2007)** — *The mid-infrared spectral properties of normal, starburst, and
active galaxies* (SINGS IRS survey)
ApJ 656, 770. arXiv:astro-ph/0610913.
[ADS](https://ui.adsabs.harvard.edu/abs/2007ApJ...656..770S/abstract)
First systematic measurement of PAH EW vs sSFR/L_IR in z~0 resolved galaxies.
The local benchmark slope that our α(M*) measurement at z~0.5–3.5 should be compared to.

**READ DIRECTLY 2026-08-03 — Table 7, the local composition benchmark.** *"PAH Band
Luminosity Ratios L(λ₁)/L(λ₂)"*, median with 10%–90% range. Two columns matter for the
line bridges; encoded verbatim as `SMITH07_T7` in `notebooks/lim_via_pah_helpers.py`:

| band | L/ΣPAH | L/L_TIR |
|---|---|---|
| 6.2 | 0.110 (0.086–0.150) | 0.0110 |
| **7.7 complex** (7.42+7.60+7.85, **8.6 NOT included**) | **0.420 (0.180–0.450)** | 0.0410 |
| 8.6 | 0.073 (0.049–0.088) | 0.0072 |
| 11.3 | 0.120 (0.092–0.180) | 0.0110 |
| 12.6 | 0.065 (0.047–0.087) | 0.0060 |
| 17 | 0.059 (0.039–0.097) | 0.0062 |

ΣPAH/L_TIR = 0.10. **ΣPAH is ALL dust features 5–19 µm, not just these six** — which is
why the six L/ΣPAH values sum to 0.85, not 1. Table 3 defines the blended complexes.

- **`F77 = 0.49` is NOT a Smith+2007 number.** The Table 7 median is **0.42**, and 0.49 is
  above the 90th percentile (0.45). It appears to be the prose — *"can contribute nearly
  one-half of the total PAH luminosity"* — which describes the top of the range.
  `build_co_shivaei_section.py` carried 0.49 as "(Smith+2007)"; corrected 2026-08-03.
- **Composition cross-check that our template passes.** Rescaled onto our 5-feature basis
  (drop 17 µm and the minor features; the five sum to 0.788 of ΣPAH), Table 7 gives
  L(7.7)/total = **0.533** and f_neutral = (0.120+0.065)/0.788 = **0.235**. Our fitted
  templates give 0.557 — **4% agreement**, an independent confirmation of both the template
  and the luminosity conversion.
- **The [CII] bridge anchor.** L(11.3)+L(12.6) = **1.7% of L_TIR**, so with Smith+2017's
  L_CII/L_TIR = 0.48±0.21% the *neutral-anchored* efficiency is
  **ε_neu = 0.0048/0.0170 = 0.282**. Doing it in L_TIR units sidesteps the ΣPAH definition
  entirely. Note the two routes to f_neutral,local disagree ~9% (0.170 via L_TIR, 0.185 via
  the ΣPAH column) because **medians of ratios are not transitive** — carry as a systematic.
- Smith's median 12.6/11.3 = **0.57** (10–90%: 0.31–0.65). Our welded template asserts
  **0.377** (Hernán-Caballero+2020); inside their range but on the low side, which depresses
  our f_neutral and therefore *overstates* the neutral-bridge correction.
- Their §5.6 states the physics we rely on: 11.3 from neutral PAHs, 7.7 from cations
  (citing Allamandola+1999, Li & Draine 2001), with harder fields also destroying grains.

**Galliano et al. (2021)** — *The dust-to-stellar mass ratio as a function of star formation
rate and stellar mass* (DustPedia)
A&A 649, A18.
[ADS](https://ui.adsabs.harvard.edu/abs/2021A&A...649A..18G/abstract)
PAH abundance vs galaxy properties (M*, metallicity, sSFR) across the DustPedia local
sample. Provides the multi-variate local comparison for α(M*) and α(σ_SFR) trends.

---

## PAH deficit — metallicity / grain-size mechanism (new, 2026-07-01, branch 7)

**Directly tests the same kind of observable as our §1a (long-to-short-wavelength PAH band
ratio vs. a galaxy property) but against metallicity, not stellar mass — added because it was
the most relevant literature missing from this list, not because it agrees with our result.**
Working through it with the standard mass–metallicity relation gives the **opposite** mass
trend from what we measure (see `pah-forward-model-7-brief.md` §1d) — flagged as an open
tension to resolve or explicitly discuss, not a confirmation.

**Engelbracht et al. (2005)** — *The PAH Emission Deficit in Low-Metallicity Galaxies — A
Spitzer View*
ApJ 628, L29. arXiv:astro-ph/0512404.
[arXiv](https://arxiv.org/abs/astro-ph/0512404)
Foundational Spitzer result: PAH emission (8/24 µm) collapses below ~1/4 solar metallicity.
Attributes this to preferential destruction of small PAH grains in the harder, less-shielded
ISM of low-metallicity systems — the origin of the "PAH deficit at low metallicity" paradigm.

**Whitcomb et al. (2024)** — *The Metallicity Dependence of PAH Emission in Galaxies I:
Insights from Deep Radial Spitzer Spectroscopy*
ApJ (2024). arXiv:2405.09685.
[arXiv](https://arxiv.org/abs/2405.09685)
Modern, spectroscopic (not just photometric) follow-up: PAH-to-dust luminosity is flat above
~2/3 solar metallicity and declines steeply below it. Critically, the decline is **band
dependent** — long-wavelength features (17 µm especially) decline steeply at low metallicity
while short-wavelength bands (6.2, 7.7 µm) carry an increasingly large fraction of the power —
attributed to an evolving grain-size distribution (fewer large PAHs survive at low
metallicity), not a pure ionization effect. **The direction check**: via the standard
mass–metallicity relation (low M* → low Z), this predicts long/short PAH ratios should be
*suppressed* at low mass — we measure the opposite (§1a: long/short is *highest* at low mass,
lowest at high mass). Even our highest-mass bin at z~2–2.5 sits close to their ~2/3-solar
threshold (12+log(O/H)≈8.5–8.7 vs. their ~8.51), so this isn't avoided by a mass-range
technicality — the tension is real and should be addressed directly in the paper.

---

## PAH destruction — radiation field mechanism

**Egorov et al. (2025)** — *PAH destruction in star-forming regions across 42 nearby galaxies*
A&A 703, A103. arXiv:2509.13845.
[ADS](https://ui.adsabs.harvard.edu/abs/2025A&A...703A.103E/abstract)
Most direct empirical anchor for the radiation-field interpretation. Quantifies PAH
fraction anti-correlation with ionization parameter U across thousands of HII regions
(PHANGS-JWST + MUSE). The physical mechanism linking M* → harder UV field → PAH destruction.

**Leroy et al. (2023)** — *PHANGS-JWST First Results: Destruction of the PAH molecules
in HII regions probed by JWST and MUSE*
ApJS 264, 10. arXiv:2212.09159.
[ADS](https://ui.adsabs.harvard.edu/abs/2023ApJS..264...10L/abstract)
Companion PHANGS paper establishing the resolved-galaxy framework. Shows PAH/continuum
ratio suppressed inside HII regions; recovery outside. Context for why U is the right
proxy for destruction.

**Xie & Ho (2022)** — *The Ionization and Destruction of Polycyclic Aromatic Hydrocarbons in
Powerful Quasars*
ApJ. arXiv:2110.09705.
[arXiv](https://arxiv.org/abs/2110.09705)
86 low-z quasars (Spitzer). Shows AGN suppress PAH bands **differentially**, not just via
uniform continuum dilution: 6.2/7.7 and 8.6/7.7 are suppressed relative to normal galaxies
while 11.3/7.7 is unchanged, attributed to AGN radiation preferentially destroying small
grains and raising the ionization fraction. Directly relevant to whether a rising AGN
fraction at high M* could drive §1a's band-ratio trend — but their specific 12.7 µm number
wasn't confirmed from a summary read; needs a direct read of the paper before citing a
predicted sign for our exact band pair.

---

## High-redshift PAH / cosmological context

**Narayanan et al. (2026)** — *The Lifecycle and Emission Properties of PAHs in
Cosmological Hydrodynamic Galaxy Formation Simulations*
arXiv:2606.20809. 19 authors incl. Torrey, Parente, J.-D. Smith, Hensley, Sandstrom,
Shivaei, Whitcomb.
[arXiv](https://arxiv.org/abs/2606.20809)
40 zoom-in galaxies (log M\*/M☉ ≈ 8.2–10.9) with on-the-fly dust evolution + single-photon
excitation MIR spectra. **Read directly 2026-07-02** (targeted extraction from the HTML full
text, not a page-by-page read; core statements cross-checked between abstract and body):
- Central mechanism: PAHs form **in situ via grain-grain shattering**, efficient only in
  diffuse gas (collision velocity ∝ 1/ρ). Hence an **inverse q_PAH–f_mol relation** (Fig 7)
  and lower q_PAH at higher Σ_SFR (Fig 9: "higher-SFR galaxies have a denser ISM that
  suppresses shattering"). q_PAH rises ~5×10⁻⁴ (z~4) → ~10⁻² (z~0); reproduces the PZR at
  z=0–2 as an emergent byproduct.
- Key decoupling statement: "the physical q_PAH and the observed L_PAH/L_FIR do **not**
  evolve in lockstep" — L_PAH/M_PAH ∝ G₀ while q_PAH anti-correlates with Σ_SFR. Any
  comparison to our L_PAH/L_IR(M\*) must go through this G₀ correction.
- **They publish no q_PAH(M\*) or L_PAH/L_IR(M\*) at fixed z, and no band-ratio predictions**
  (pahfit gives per-band luminosities per galaxy but only total L_PAH is analyzed). Both are
  gaps our measurements (amplitude slope; §1a 12.7/6.2 ratio vs mass) can directly confront —
  the branch-7 supporting-result target. The sign their mechanism implies for
  d q_PAH/d log M\* at fixed z is **not obvious**: at cosmic noon, gas fraction falls with M\*
  (→ q_PAH rises with mass, matching our trend) but Σ_SFR is flat-to-rising with M\*
  (→ opposite pull); needs an explicit derivation via Fig 9 + Tacconi+18 scaling relations
  before claiming support or tension.

**PAHSPECS I & II (2026)** — arXiv:2606.18230 (integrated) + arXiv:2606.18244 (resolved).
**Read directly 2026-07-11** (abstract-level extraction of both).
Five z≈1.1 star-forming galaxies (ASPECS/HUDF, JWST MIRI MRS, CAFE decomposition; one AGN,
ASPECS-15). **No mass slopes, no L_PAH/L_IR(M*), nothing above z~1.3 — no threat to the
crossing pattern's novelty.** Findings relevant to us:
- Integrated (18230): vs local LIRGs, *higher* 6.2/7.7 and *lower* 11.3/7.7 — an ionized
  PAH mix weighted to smaller grains in massive cosmic-noon SFGs. Same direction as our
  12.7/6.2 decline with M* (supports the charge interpretation, B1).
- Resolved (18244): PAHs become larger/more neutral with galactocentric radius; within
  galaxies, harder UV **raises** 11.3/7.7 (photo-destruction of small/ionized PAHs) —
  the small-grain-destruction channel pulls the ratio the *opposite* way from the charge
  channel. The two PAHSPECS papers thus bracket both arrows; our galaxy-integrated trend
  lands on the charge side.
- 7.7 µm stays a robust SFR tracer at z~1.1 — consistent with our well-behaved z~1 slice.

---

## Scaling relations used in the Narayanan+26 confrontation (2026-07-02, branch 7)

Used by `notebooks/2026-07-02-pah-narayanan-confrontation.ipynb` §5 to map the
shattering mechanism onto the stellar-mass axis at fixed z. Not PAH papers; listed here so
the derivation's inputs are traceable.

**Tacconi et al. (2018)** — ApJ 853, 179. arXiv:1702.01140.
Molecular gas scaling: log μ_gas = 0.12 − 3.62(log(1+z) − 0.66)² + 0.53 log δMS
− 0.35(log M* − 10.7). Coefficient forms cross-checked against the arXiv abstract
(μ_gas ∝ δMS^0.52 M*^−0.36) 2026-07-02.

**van der Wel et al. (2014)** — ApJ 788, 28.
Late-type size–mass relation R_e ≈ 8.9 kpc (M*/5×10¹⁰)^0.22 (1+z)^−0.75 → Σ_H2, Σ_SFR.

**Sanders et al. (2021)** — ApJ 914, 19.
MZR: O/H ∝ M*^0.30 (low-mass slope, invariant z=0–3.3; flattens near the ~10^10.2
turnover). Basis of the γ_MZR ∈ [0.15, 0.30] bracket over log M* = 10–11.3.

**Bigiel et al. (2008)** — AJ 136, 2846.
Σ_HI saturation at ~10 M☉/pc² — sets the f_mol = Σ_H2/(Σ_H2+Σ_HI) proxy; bracketed 5–20.

---

## Dust temperature context

**Viero et al. (2022)** — *A Surprising Lack of Dust Evolution at z < 5 Observed with
Herschel and Spitzer*
MNRAS 516, L30.
[ADS](https://ui.adsabs.harvard.edu/abs/2022MNRAS.516L..30V/abstract)
T_dust(z) = 23.8 + 2.7z + 0.9z² from stacking COSMOS2020 (this work's predecessor).
The PAH correction is needed to avoid biasing T_dust upward at z~1.5–2.5.

**Schreiber et al. (2018)** — *Dust temperature and mid-to-total infrared color
distributions for star-forming galaxies at 0 < z < 4*
A&A 609, A30.
[ADS](https://ui.adsabs.harvard.edu/abs/2018A&A...609A..30S/abstract)
T_dust(z) = 32.9 + 4.6(z−2) linear relation. Used as the Schreiber temperature prior
in `greybody.py`. Sets the FIR peak anchor that f₂₄/f_peak normalises against.

## Methodological inspiration — line/feature excess hidden in broadband photometry (added 2026-07-27, lim-talk-figs-1)

**Agrawal, Aguirre & Keenan (2026)** — *Far-infrared lines hidden in archival deep
multi-wavelength surveys: Limits on [CII]-158 μm at z ∼ 0.3–2.9*
A&A 705, A246.
[A&A](https://www.aanda.org/10.1051/0004-6361/202556503)
Tomographic stacking of archival FIR/submm survey data to constrain a hidden emission
line via its imprint on broadband photometry — the same statistical philosophy as this
project's PAH tomography. Direct inspiration slide-1/slide-3 acknowledgment.

**Pullen, Serra, Chang, Doré & Ho (2018)** — *Search for C II emission on cosmological
scales at redshift z ∼ 2.6*
MNRAS 478, 1911.
[ADS](https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.1911P/abstract)
Cross-correlates SDSS quasars/CMASS galaxies with Planck 545 GHz broadband maps to search
for excess line emission statistically, without resolving individual sources. Companion
inspiration citation alongside Agrawal+26 for "measure the excess in broadband photometry."

## Line-intensity mapping (LIM) — [CII]/CO forecasting (branch forecast-lim-via-pah-1, 2026-07-12)

**Chiang et al. (2026)** *Cosmic CO and [CII] backgrounds and the fuelling of star
formation over 12 Gyr*
Nature Astronomy; arXiv:2602.02658.
[arXiv](https://arxiv.org/abs/2602.02658)
First MEASUREMENT of the mean cosmic [CII] (3σ) and CO (7σ, full ladder) line backgrounds,
0<z<4.2, via tomographic clustering of diffuse broadband intensities with reference galaxies
— the same statistical philosophy as our dithered stacking. Measured [CII] comoving
luminosity density fit: ρ_CII(z) ≈ 5.9×10³⁸ (1+z)^3.2 / [1+((1+z)/2.9)^6.6] erg s⁻¹ Mpc⁻³;
Ω_H2(z) ≈ 1.9×10⁻⁴ (1+z)^2.3 / [1+((1+z)/2.8)^6.5] (≈2× that resolved in galaxy surveys);
t_dep ≈ 3(1+z)⁻¹ Gyr. Used as the real ⟨I_CII⟩(z) anchor in the LIM notebook; our
De-Looze-anchored logM*>9.9 forecast lands at ~1/3 of it (the massive-galaxy contribution).

**De Looze et al. (2014)** A&A 568, A62 — L_CII–SFR calibration, whole-sample
**log L_CII = 7.06 + log SFR**. Computed [CII] model curve in the LIM notebook. ALPINE
(Schaerer+2020, A&A 643, A3) finds slope 0.96 / offset −0.03 dex vs this → effectively the
same line.

**Lagache et al. (2018)** A&A 609, A130 — z-evolving SAM L_CII–SFR:
**log L_CII = (1.4 − 0.07 z) log SFR + (7.1 − 0.07 z)**, σ = 0.5 dex. Computed [CII] model
curve (steeper, evolves; near Chiang at low z).

**Padmanabhan (2019)** MNRAS 488, 3014 — halo-mass-based [CII] model,
L_CII(M,z) = (M/M₁)^β exp(−N₁/M) × [(1+z)^2.7/(1+((1+z)/2.9)^5.6)]^α. One of several
published [CII] model curves overlaid on the Chiang comparison figure (Slide 9).

**Yang, Popping, Somerville, Pullen & Maniyar (2022)** ApJ 929, 140 —
empirical representation of a physical ISM model for [CII]/CO/[CI] at 1≤z≤9 (Popping SAM).
Another comparison model curve on the same figure.

**Silva et al. (2015)** ApJ 806, 209 — *Prospects for Detecting [CII] Emission during the
Epoch of Reionization*; four model variants (m1–m4), **m2** used here as a comparison curve.

**Li et al. (2016)** ApJ 817, 169 — COMAP CO(1-0) forecast model. SFR = δ_MF·10⁻¹⁰·L_IR
(Kennicutt, δ_MF=1); **log L_IR = 1.37 log L'_CO − 1.74** (Carilli & Walter 2013); L_CO =
4.9e-5 L'_CO. Computed CO model curve.

**mmIME** Keating et al. (2020) ApJ 901, 141 — CO shot-noise **detection**
P_shot = 2.0(+1.1/−1.2)×10³ μK²(Mpc/h)³ (higher-J summed). **COMAP ES-V** Chung et al. (2022)
ApJ 933, 186 — CO(1-0) z~3 upper limit ⟨T_b⟩² < 50 μK²; also carries a **fiducial CO model**
(UM+COLDz+COPSS: UniverseMachine halo–CO with COLDz/COPSS priors) — add as a model curve
(digitize from the paper), not just the limit.

**Chung, Viero, Church & Wechsler (2020)** ApJ 892, 51; arXiv:1812.08135 — [CII] LIM forecast,
primarily z~4–8 (UniverseMachine galaxy–halo × SFR–[CII] correlation). A comparison [CII] model
to add (grazes our z~1–4 only at z~4); digitize its L_CII prescription / intensity. Co-author MV.

**CO real-data comparisons used in §3b of the LIM notebook (rebuilt 2026-07-25, each at its
own transition/z — the earlier version mismatched transitions):**
Keating et al. (2016) ApJ 830, 34 — COPSS II, CO(1–0) shot noise at z=2.8 (a 2σ detection in
tension with COMAP's own limits — read the ~90× gap accordingly); Stutzer et al. (2024) —
COMAP Season 2, CO(1–0) 95% upper limit at z=2.4–3.4; Decarli et al. (2020) ApJ 902, 110 —
ASPECS LP, ρ(H₂) in 5 z-bins over 0.3–4.5, α_CO=3.6; Riechers et al. (2019) ApJ 872, 7 — COLDz,
ρ(H₂) at z=2.0–2.8; Chung et al. (2024) — COMAP Season 2, ρ(H₂) and ⟨T_b⟩ upper limit at z~3;
Riechers et al. (2020, VLASPECS) — CO excitation ladder r₃₁=L'_CO(3–2)/L'_CO(1–0)=0.84±0.26,
used to convert our CO(1–0) model onto mmIME's higher-J transitions for a like-for-like check.

**Bridge calibrations** (PAH ↔ line). **[CII]: use L_CII/TOTAL-PAH ≈ 0.05, NOT 0.1.** The
Croxall+12 / Sutter+19 "L_CII/L_PAH ≈ 0.1" is L_CII/PAH-*subset* (their PAH ≈ the 7.7 µm
complex, ≈49% of total PAH per Smith+07). The total-PAH bridge, from the same local galaxies,
is (L_CII/L_TIR)/(L_PAH/L_TIR) = 0.48%/10% ≈ 0.05 — see **Herrera-Camus et al. (2015)** ApJ 800,
1 (KINGFISH L_CII/L_TIR = 0.48±0.21%) and **Smith et al. (2007)** ApJ 656, 770 (L_PAH/L_TIR ≈
10%; 7.7 complex ≈ 49% of total PAH). Applying 0.1 to a total-PAH template double-counts by ~2×.
**CO:** Cortzen+19 & **Shivaei & Boogaard (2024)** A&A 691, L2; arXiv:2409.05710 —
*"The tight correlation between PAH and CO emission from z~0 to 4"* — 14 z=1–3
main-sequence galaxies with CO detections + a z=0–4 literature compilation; L_PAH(7.7 µm)
vs. L'_CO(1–0) holds with **0.21 dex scatter**. This is the direct measurement behind the
"L_PAH tracks L'_CO" claim (previously cited here only by its bare arXiv number — same
first author, unrelated topic, as the SMILES **Shivaei+24** metallicity paper above; do not
conflate the two). MS L_IR/L'_CO ≈ 70 (Sargent+14).

**READ DIRECTLY 2026-08-03 — the 7.7 definition, which the bridge depends on.** Verbatim:
*"We first integrated the flux density from 6.9 to 9.7 µm, assuming the feature strength to
be zero on either side... As the derived values are for the continuum-subtracted 7.7+8.6 µm
PAH complex luminosity, we then applied a **15% correction for the 8.6 µm feature
contamination**, to estimate the 7.7 µm luminosity alone."*
So **their L(PAH 7.7) excludes 8.6** — any conversion from a total-PAH template must use the
7.7-only share, not the welded 7.7+8.6 group. Our template puts **16.4%** of the complex in
8.6, matching their 15% to 1.4 points, so the definitions are compatible.
Table 1 (L(PAH₇.₇)–L′(CO), all sources): slope **1.07±0.04**, intercept **−0.52±0.36**,
intrinsic scatter **0.21±0.02 dex**, median ratio **1.40±0.49**. Also tabulated for non-AGN
and z>1 subsets, and for L(PAH₇.₇)–L(IR) and L′(CO)–L(IR).
Their decomposition is **Draine & Li (2007) models inside PROSPECTOR**, not PAHFIT, though
they describe the continuum treatment as closest to PAHFIT-style decomposition.

## Cosmic infrared background — the closure test on n × L_IR (added 2026-07-28, lim-talk-figs-1)

Used by §8 of `notebooks/2026-07-26-lim-via-pah.ipynb` (built by
`notebooks/build_cib_closure_section.py`) to check that the (abundance × L_IR) product setting
the absolute amplitude of the PAH-anchored [CII]/CO forecast also reproduces the measured CIB.

**Viero et al. (2013)** ApJ 779, 32; arXiv:1304.0446 — *HerMES: The Contribution to the CIB from
Galaxies Selected by Mass and Redshift*. **This is the like-for-like reference**, not the 2015
letter: same technique (simultaneous stacking of a mass/z-binned catalog), reported as νI_ν split
by redshift slice and mass bin. **Figure 7** is the figure §8 reproduces; **Table 3** is the
number table. UDS, K_AB < 24, 0.63 deg², logM 9–12, z 0–4.

Table 3 "Total Stacking" (catalogue-limited, nW m⁻² sr⁻¹) and the absolute CIB it adopts:

| λ (µm) | 24 | 70 | 100 | 160 | 250 | 350 | 500 | 1100 |
|---|---|---|---|---|---|---|---|---|
| stacking | 1.84±0.05 | 3.31±0.20 | 8.74±0.38 | 9.43±0.63 | 7.00±0.34 | 4.38±0.22 | 1.84±0.10 | 0.06±0.01 |
| absolute CIB | 2.86±0.17 | 6.60±0.70 | 12.60±4.00 | 13.60±2.50 | 10.40±2.30 | 6.50±1.60 | 2.60±0.60 | 0.19±0.04 |
| recovered | 64% | 50% | 69% | 69% | 67% | 67% | 70% | 34% |

Absolute-CIB references per band: **Béthermin et al. (2010)** at 24/70; **Berta et al. (2011)** at
100/160 (note the 100 µm value carries a ±32% error — do not read a 100 µm "deficit" as physical);
**Lagache et al. (2000)** at 250–1100. Tables 6/7/8 give the same split by z bin and by mass bin.

**Viero et al. (2015)** ApJL 809, L22; arXiv:1505.06242 — *HerMES: Current CIB Estimates Can Be
Explained by Known Galaxies and their Faint Companions at z < 4*. The **follow-up**: smooths the
maps before stacking so faint sources clustered around catalogued ones are swept in, recovering
9.82±0.78 / 5.77±0.43 / 2.32±0.19 nW m⁻² sr⁻¹ at 250/350/500 µm = 94/107/97% of the CIB.
UltraVISTA K_S < 23.4, 1.62 deg² of COSMOS. **This is the upper reference line, not our analogue**
— a catalogue-limited stack like ours should land near the 2013 numbers; the gap between the two
papers *is* the below-catalogue population.

**Fixsen et al. (1998)** ApJ 508, 123; arXiv:astro-ph/9803021 — COBE/FIRAS FIRB spectrum, the grey
curve in the §8 figures. Analytic fit, valid ν = 5–80 cm⁻¹ (2000–125 µm):
`I_ν = (1.3±0.4)e-5 (ν/ν₀)^(0.64±0.12) P_ν(18.5±1.2 K)`, ν₀ = 100 cm⁻¹ (λ₀ = 100 µm), P the Planck
function. Evaluates to 10.28 / 5.63 / 2.37 nW m⁻² sr⁻¹ at 250/350/500 µm — reproduces the
Lagache+2000 values Viero+13 adopts to within their errors. Total over the fit range: 14 nW m⁻² sr⁻¹.

**Emissivity integral (the formula, since one factor of (1+z) is easy to lose):**
`ν₀I_ν₀ = (c/4π) ∫dz  ν_e ε_ν(ν_e,z) / [(1+z)² H(z)]`, ν_e = (1+z)ν₀, ε **comoving**. Two powers of
(1+z): photon-energy loss, plus a comoving shell subtending D_C²dD_C while flux falls as
D_L⁻² = [(1+z)D_C]⁻². Getting it wrong by one power inflates the answer by ⟨1+z⟩ ≈ 2 — the size of
the effect being tested. §8b validates it against a cosmology-free flux sum (agrees to <10%).
