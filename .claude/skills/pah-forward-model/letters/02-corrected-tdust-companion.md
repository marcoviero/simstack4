# Letter possibility 02 (companion to 01) — Corrected T_dust(z) after PAH removal

**Status:** partially executed 2026-07-01 (branch 6, §9 of the letter notebook) using the
**pivoted** headline (`03-pah-deficit-rises-with-mass.md`), not the original flat-L_PAH/L_IR
framing this doc was written under. The library-level correction (`wien_mode="lir_pah"`,
`alpha_wien` override) now exists and was run once: **tier counts unchanged** (median-SNR
tier logic isn't sensitive to a single previously-suppressed band), **T_dust shifts +0.81 K**
at z<3 (small, not the dramatic flattening this doc's original hypothesis expected) vs +0.43 K
at z≥3 (a separate α_wien-retilt effect, correction itself is off there). This is a first pass,
not the full companion-paper treatment below (no T(z) relation refit or Viero+22 comparison
yet) — still open for branch 7.
**Pairs with:** `03-pah-deficit-rises-with-mass.md` (supersedes `01-pah-tracks-lir-invariant.md`)
**Science question:** Is the steep T_dust(z) rise (Viero+22: 23.8+2.7z+0.9z²) real, or
partly an artifact of PAH leakage into the warm side of 24 µm-anchored fits? (= the
`DustEvolutionModel` question, now testable with a measured PAH correction.)

## Scope (what makes it a separate paper, not §8c)

§8c only computed an **upper-bound** ΔT ≈ +4 K (z=1.5–2.5) by inverting a single 24 µm point.
The companion paper must **propagate the measured PAH correction through the SED fits**:

1. Use measured L_PAH/L_IR ≈ 2.4% (flat; Letter 01) + the global feature template (r_g) to
   predict f₂₄_PAH(M\*, z) per bin.
2. Subtract f₂₄_PAH from the stacked 24 µm, then **re-fit greybody with 24 µm re-included**
   at reduced inflation (two-pass scheme already sketched in `CLAUDE.md` PAH section).
3. Compare corrected vs uncorrected T_dust(z); report the revised T–z relation and the
   fraction of bins promoted Tier C→B.

## Inputs / dependencies

- Measured α/template from Letter 01 (in hand).
- `greybody.py::_pah_flux_0` / `_physical_wien_flux` correction path (exists; coefficients
  `_pah_coeffs` calibrated — re-check against the flat-L_PAH/L_IR result, which differs from
  the −0.10 deficit those coeffs assume).
- A new notebook: load wrappers → apply correction → two-pass SED fit → T_dust(z).

## Headline (hypothesis)

Corrected T_dust(z) has a **shallower** high-z slope than Viero+22; the residual rise (if any)
is the real cold-dust evolution. Compare to Schreiber+18 linear relation.

## Caveat

The `_pah_coeffs` in `greybody.py` encode a mass *deficit* slope; Letter 01 finds L_PAH/L_IR
**flat**. Reconcile before applying — the correction normalization (per L_IR) is what matters,
and that is well-measured (~2.4%), but the mass dependence baked into the coeffs may need
flattening.
