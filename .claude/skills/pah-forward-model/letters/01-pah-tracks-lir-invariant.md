# Letter possibility 01 — PAH emission tracks L_IR, independent of stellar mass

**Status:** SUPERSEDED (2026-07-01) — see `03-pah-deficit-rises-with-mass.md`. This branch's
α-free re-measurement (`pah-forward-model-6-summary.md`) found L_PAH/L_IR **rises** with M*
(+0.15 to +0.21 dex/dex, sign-stable across α=2.0-2.86), not flat. At matched α=2 the old
result here (+0.019, flat) and the new one (+0.146, rising) still disagree — a methodology
difference (single catalog split + unsmoothed baseline here vs K-fold-pooled + smoothed
baseline there), not (only) the α assumption. Kept for the record / to document what changed
and why; do not cite the "invariant" claim below.
**Source notebook:** `notebooks/2026-06-27-load-json-fit-seds-redshift-stellar-mass-PAH-dithered-physical-baseline.ipynb` (§8 / §8b / §8c)
**Origin:** asked to find a novel result to complement the α(M\*) PAH work; computing
L_PAH/L_IR (which had never been done) overturned the assumed deficit and produced a
cleaner claim. See memory `pah-signflip-diagnosis.md`.

---

## Working title

*PAH emission tracks the infrared luminosity of main-sequence galaxies independent of
stellar mass at cosmic noon*

## One-line thesis

From far-infrared **tomographic stacking** alone (no MIR spectroscopy), the PAH-to-total-IR
luminosity ratio of normal star-forming galaxies is **constant (≈2.4%) across
10¹⁰–10¹² M⊙ at 0.5 < z < 3.5** — PAH luminosity tracks L_IR (hence SFR), not stellar mass.
The stellar-mass trend seen in MIPS-24 mid-IR colour is a **warm-continuum** effect, not PAH
destruction; and naively including 24 µm as a thermal point biases stacked T_dust high by ~4 K.

## Headline measurements (5-bin run, Δz=0.15 × 4 dithers, COSMOS2020, split=0)

| Quantity | Value | Significance |
|---|---|---|
| L_PAH/L_IR, science bins (logM\* 10–12) | 2.14 / 2.62 / 2.43 / 2.41 % → mean **2.40%** | — |
| d log(L_PAH/L_IR)/d log M\* | **+0.019 ± 0.034 dex/dex** | 0.2σ (rescaled) → **flat** |
| PAH equivalent width A=α/C_m slope | **+0.31 dex/dex** | **7.4σ** (rescaled by √χ²_red=3.07) |
| Global feature ratios 6.2 : 7.7+8.6 : 12.7 | 1 : 1.68 : 4.32 | well-constrained |
| PAH fraction of f₂₄ (science bins) | ~46–48% | (rest ~half warm continuum) |
| 24 µm contamination f₂₄/f_cold, z=1.5–2.5 | median **2.3×** (IQR 1.5–2.6) | — |
| Upper-bound T_dust bias, z=1.5–2.5 | **+4.2 K** (IQR +2.1 to +5.1) | if 24 µm taken as thermal |

**Key interpretive point:** the rising EW (7.4σ) and flat L_PAH/L_IR (0.2σ) together mean the
**warm mid-IR continuum drops faster than the PAH** with stellar mass. The trend is in the
continuum, not the PAH abundance.

## Why it's novel / publishable

- First **statistical** L_PAH/L_IR vs M\* for *normal* (not just bright/spectroscopic)
  galaxies at cosmic noon, measured from the broadband data itself rather than assumed from a
  template (DH02/DL07).
- A clean, falsifiable claim (**invariance**) that runs against the naive extrapolation of the
  local resolved PAH-destruction picture to integrated high-z galaxies.
- Delivers a concrete systematic the community uses constantly: the 24 µm → T_dust/SFR bias.

## Figure set

1. **Headline:** L_PAH/L_IR vs log M\* (flat, science bins) beside A=α/C_m vs log M\* (rising).
   The "trend is in the continuum" two-panel. *(notebook §8)*
2. Tomographic pseudo-spectrum (f₂₄ − baseline)/baseline vs λ_rest with the deconvolved
   intrinsic PAH spectrum overlaid. *(notebook §4c/§4d)*
3. T_dust bias: 24 µm contamination factor + implied ΔT histograms (z=1.5–2.5). *(§8c)*
4. (optional) L_PAH/L_IR vs the arXiv:2606.20809 sim prediction and the Smith+07/Galliano+21
   local relation, showing high-z invariance vs local decline.

## Positioning against cited papers (`docs/pah-refs.md`)

- **Smith+07, Galliano+21** — local L_PAH/L_IR (and EW) *declines* with sSFR/L_IR/mass; we show
  the integrated high-z ratio is *flat*. Tension to discuss: averaging / selection / SF-dominated.
- **Tielens 08; Egorov+25; Leroy+23** — resolved radiation-field PAH destruction; we argue it
  does not survive galaxy-integration for normal SF galaxies at these masses.
- **arXiv:2606.20809** — sim L_PAH(M\*,z); direct overplot, test invariance.
- **arXiv:2606.18244 (PAHSPECS)** — JWST MIRI spectroscopy of the bright tail; complementary,
  check their L_PAH/L_IR or EW vs M\* at z~1–2.
- **Viero+22; Schreiber+18** — the +4 K 24 µm T_dust bias revises the stacked T_dust(z) slope.

## Caveats to nail before submission

1. **Low-mass L_IR** unreliable (<10¹⁰; the 9.0–10.0 bin returns 374%). State the mass range
   honestly; do not claim below 10¹⁰ without a better L_IR.
2. **χ²_red ≈ 9** → all significances quoted after ×√χ²_red error inflation (done in §8b).
3. Per-feature band ratios are **not** individually identifiable (SNR≈1.2, §7) — quote only the
   global ratio and the pooled amplitude.
4. **K-fold independence** (3 COSMOS2020 splits) still needed to back the error bars.
5. **σ_SFR cross-cut** would test whether invariance holds at fixed M\* across the MS.
6. AGN / warm-dust contamination at the high-σ_SFR end (single (T_w,β_w) continuum assumption).

## Open follow-ups (other notebooks/configs)

- K-fold parquets → 3 independent pseudo-spectra; std(α) vs rescaled bootstrap σ.
- 2 mass × 3 σ_SFR run → L_PAH/L_IR at fixed M\* vs σ_SFR.
- Shared-slope baseline (branch-4) → does fixing γ change the EW slope?
