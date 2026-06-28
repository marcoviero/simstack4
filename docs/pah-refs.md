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

---

## High-redshift PAH / cosmological context

**arXiv:2606.20809** (2026) — *The Lifecycle and Emission Properties of PAHs in
Cosmological Hydrodynamic Galaxy Formation Simulations*
[arXiv](https://arxiv.org/abs/2606.20809)
Models PAH formation/destruction as a function of radiation field in a full cosmological
simulation. Directly comparable to our α(M*, z) measurement; tests whether the slope
persists or evolves with redshift. Most relevant theoretical counterpart.

**arXiv:2606.18244** (2026) — *PAHSPECS: Spatially Resolved PAH Spectroscopy at Cosmic
Noon with JWST MIRI MRS*
[arXiv](https://arxiv.org/abs/2606.18244)
Direct PAH spectroscopy at z~1–3 with JWST MIRI. Potential direct comparison or tension
with our stacking result. Check for α(M*) values or PAH EW vs stellar mass at z~1–2.

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
