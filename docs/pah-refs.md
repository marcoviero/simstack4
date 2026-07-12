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

---

## PAH deficit — local empirical anchors

**Smith et al. (2007)** — *The mid-infrared spectral properties of normal, starburst, and
active galaxies* (SINGS IRS survey)
ApJ 656, 770.
[ADS](https://ui.adsabs.harvard.edu/abs/2007ApJ...656..770S/abstract)
First systematic measurement of PAH EW vs sSFR/L_IR in z~0 resolved galaxies.
The local benchmark slope that our α(M*) measurement at z~0.5–3.5 should be compared to.

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
