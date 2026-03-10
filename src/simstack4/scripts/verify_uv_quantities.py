#!/usr/bin/env python3
"""
Verify β_UV and L_UV(1600Å) computations in the catalog.

Run: uv run python scripts/verify_uv_quantities.py

Checks:
  1. β_UV = β_intrinsic + k_λ × E(B-V) is self-consistent
  2. L_UV(1600) = L_NUV × (1600/2300)^β is applied correctly
  3. Distributions are physically reasonable
  4. Cross-checks against independent estimates
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────────────────
CATALOG_PATH = Path.home() / "data/Astronomy/catalogs/cosmos/COSMOSWeb_wijesekera_sfg.parquet"
#CATALOG_PATH = Path.home() / "data/Astronomy/catalogs/cosmos/COSMOSWeb_mastercatalog_v1_galaxies_mag27.0_clean.parquet"

# Column names (adjust if your parquet uses different names)
EBV_COL = "ebv_minchi2"
LAW_COL = "law_minchi2"
L_NUV_COL = "l_nuv"           # log10(L_NUV / L_sun) from LePhare
BETA_COL = "beta_uv"          # computed β_UV
L_UV_COL = "log_l_uv"         # computed log10(L_UV(1600) / L_sun)
Z_COL = "redshift"
MASS_COL = "log_stellar_mass"

# Physics constants
BETA_INTRINSIC = -2.3
K_LAMBDA = {0: 4.43, 1: 4.20, 2: 3.80}  # Calzetti, Arnouts, Salim
LAW_NAMES = {0: "Calzetti", 1: "Arnouts", 2: "Salim"}
LOG10_1600_2300 = np.log10(1600.0 / 2300.0)  # = -0.1576


def main():
    print("=" * 70)
    print("VERIFICATION: β_UV and L_UV(1600Å)")
    print("=" * 70)

    # ── Load catalog ─────────────────────────────────────────────────
    if not CATALOG_PATH.exists():
        print(f"\nCatalog not found: {CATALOG_PATH}")
        print("Edit CATALOG_PATH at the top of this script.")
        sys.exit(1)

    df = pd.read_parquet(CATALOG_PATH)
    print(f"\nLoaded {len(df):,} sources from {CATALOG_PATH.name}")
    print(f"Columns of interest:")
    for col in [EBV_COL, LAW_COL, L_NUV_COL, BETA_COL, L_UV_COL, Z_COL, MASS_COL]:
        present = col in df.columns
        print(f"  {col:>20}: {'FOUND' if present else 'MISSING'}")

    # ── Check 1: β_UV computation ────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("CHECK 1: β_UV = β_intrinsic + k_λ × E(B-V)")
    print(f"{'─' * 70}")

    if EBV_COL not in df.columns:
        print("  SKIP: E(B-V) column not found")
    elif BETA_COL not in df.columns:
        print("  SKIP: beta_uv column not found (run catalog builder first)")
    else:
        ebv = df[EBV_COL].values.astype(float)
        beta = df[BETA_COL].values.astype(float)
        valid = np.isfinite(ebv) & np.isfinite(beta) & (ebv >= 0)

        # Recompute β from E(B-V) and check
        if LAW_COL in df.columns:
            law = df[LAW_COL].values
        else:
            law = np.zeros(len(df))
            print("  (No dust law column; assuming Calzetti for verification)")

        beta_recomputed = np.full_like(ebv, BETA_INTRINSIC)
        for law_idx, k_val in K_LAMBDA.items():
            mask = (law == law_idx) & valid
            beta_recomputed[mask] = BETA_INTRINSIC + k_val * ebv[mask]

        # Compare
        diff = beta[valid] - beta_recomputed[valid]
        print(f"\n  E(B-V) distribution ({np.sum(valid):,} valid):")
        print(f"    min={ebv[valid].min():.3f}  median={np.median(ebv[valid]):.3f}"
              f"  max={ebv[valid].max():.3f}")

        if LAW_COL in df.columns:
            for law_idx, name in LAW_NAMES.items():
                n = np.sum(law[valid] == law_idx)
                print(f"    {name}: {n:,} sources ({100*n/np.sum(valid):.0f}%)")

        print(f"\n  β_UV distribution ({np.sum(valid):,} valid):")
        print(f"    min={beta[valid].min():.2f}  median={np.median(beta[valid]):.2f}"
              f"  max={beta[valid].max():.2f}")

        print(f"\n  Recomputed vs catalog β_UV:")
        print(f"    max |diff| = {np.max(np.abs(diff)):.6f}")
        print(f"    mean diff  = {np.mean(diff):.6f}")
        if np.max(np.abs(diff)) < 1e-4:
            print("    ✓ PASS: β_UV matches E(B-V) computation exactly")
        else:
            print("    ✗ FAIL: β_UV does not match recomputation!")
            # Show where they differ
            bad = np.abs(diff) > 0.01
            if np.any(bad):
                idx = np.where(valid)[0][bad][:5]
                for i in idx:
                    print(f"      row {i}: E(B-V)={ebv[i]:.3f}, law={law[i]}, "
                          f"catalog β={beta[i]:.3f}, recomputed={beta_recomputed[i]:.3f}")

        # Sanity: β should be in [-2.3, ~2] for star-forming galaxies
        frac_blue = np.mean(beta[valid] < -2.0)
        frac_red = np.mean(beta[valid] > 1.0)
        print(f"\n  Sanity checks:")
        print(f"    β < -2.0 (very blue):  {100*frac_blue:.1f}%"
              f"  {'(OK, mostly unattenuated)' if frac_blue < 0.3 else '(SUSPICIOUS — too many)'}")
        print(f"    β > +1.0 (very red):   {100*frac_red:.1f}%"
              f"  {'(OK, heavily attenuated)' if frac_red < 0.1 else '(CHECK — very dusty)'}")
        print(f"    β at E(B-V)=0:         {BETA_INTRINSIC:.1f} (intrinsic)")
        print(f"    β at E(B-V)=0.3:       {BETA_INTRINSIC + 4.43*0.3:.2f} (Calzetti)")
        print(f"    β at E(B-V)=0.5:       {BETA_INTRINSIC + 4.43*0.5:.2f} (Calzetti)")

    # ── Check 2: L_UV(1600Å) computation ─────────────────────────────
    print(f"\n{'─' * 70}")
    print("CHECK 2: log L_UV(1600) = log L_NUV + β × log₁₀(1600/2300)")
    print(f"{'─' * 70}")

    if L_NUV_COL not in df.columns:
        print("  SKIP: L_NUV column not found")
    elif L_UV_COL not in df.columns:
        print("  SKIP: log_l_uv column not found")
    elif BETA_COL not in df.columns:
        print("  SKIP: beta_uv column not found")
    else:
        l_nuv = df[L_NUV_COL].values.astype(float)
        l_uv = df[L_UV_COL].values.astype(float)
        beta = df[BETA_COL].values.astype(float)
        valid = np.isfinite(l_nuv) & np.isfinite(l_uv) & np.isfinite(beta)

        # Recompute
        l_uv_recomputed = l_nuv + beta * LOG10_1600_2300

        diff = l_uv[valid] - l_uv_recomputed[valid]
        correction = beta[valid] * LOG10_1600_2300

        print(f"\n  L_NUV distribution ({np.sum(valid):,} valid):")
        print(f"    min={l_nuv[valid].min():.2f}  median={np.median(l_nuv[valid]):.2f}"
              f"  max={l_nuv[valid].max():.2f}")

        print(f"\n  L_UV(1600) distribution:")
        print(f"    min={l_uv[valid].min():.2f}  median={np.median(l_uv[valid]):.2f}"
              f"  max={l_uv[valid].max():.2f}")

        print(f"\n  Correction (β × log₁₀(1600/2300)):")
        print(f"    log₁₀(1600/2300) = {LOG10_1600_2300:.4f}")
        print(f"    min correction  = {correction.min():+.4f} dex"
              f" (factor {10**correction.min():.2f}×)")
        print(f"    median correction = {np.median(correction):+.4f} dex"
              f" (factor {10**np.median(correction):.2f}×)")
        print(f"    max correction  = {correction.max():+.4f} dex"
              f" (factor {10**correction.max():.2f}×)")

        print(f"\n  Recomputed vs catalog L_UV(1600):")
        print(f"    max |diff| = {np.max(np.abs(diff)):.6f}")
        if np.max(np.abs(diff)) < 1e-4:
            print("    ✓ PASS: L_UV(1600) matches β-corrected L_NUV exactly")
        else:
            print("    ✗ FAIL: L_UV(1600) does not match recomputation!")

        # Key diagnostic: what was the OLD L_UV (uncorrected)?
        print(f"\n  Impact of correction (old L_NUV vs new L_UV(1600)):")
        print(f"    If you used L_NUV as L_UV, IRX was off by:")
        for b in [-2.0, -1.5, -1.0, -0.5, 0.0]:
            corr = b * LOG10_1600_2300
            frac_at_beta = np.mean(np.abs(beta[valid] - b) < 0.25)
            print(f"      β≈{b:+.1f}: L_1600/L_NUV = {10**corr:.2f}×"
                  f"  (IRX was {(1/10**corr - 1)*100:+.0f}% wrong)"
                  f"  [{100*frac_at_beta:.0f}% of sources near this β]")

    # ── Check 3: L_UV units ──────────────────────────────────────────
    print(f"\n{'─' * 70}")
    print("CHECK 3: Units sanity — is l_nuv in log₁₀(L/L_sun)?")
    print(f"{'─' * 70}")

    if L_NUV_COL in df.columns and Z_COL in df.columns:
        l_nuv = df[L_NUV_COL].values.astype(float)
        z = df[Z_COL].values.astype(float)
        valid = np.isfinite(l_nuv) & np.isfinite(z) & (z > 0.5)

        print(f"\n  L_NUV range: {l_nuv[valid].min():.1f} to {l_nuv[valid].max():.1f}")

        if l_nuv[valid].max() > 20:
            print("    → Values > 20: likely log₁₀(L/L_sun) ✓")
            print(f"    → L_NUV = 10^{np.median(l_nuv[valid]):.1f} L_sun"
                  f" = {10**np.median(l_nuv[valid]):.1e} L_sun at median")
        elif l_nuv[valid].max() > 0 and l_nuv[valid].max() < 20:
            print("    ⚠ Values in [0, 20]: could be log₁₀(L/erg/s/Hz) or Watts?")
            print("    Check LePhare documentation for l_nuv units")
        else:
            print("    ⚠ Unexpected range — check units")

        # Cross-check: at z~1, typical SF galaxy has log(L_UV) ~ 9.5-10.5
        z_mask = (z > 0.8) & (z < 1.2) & valid
        if np.any(z_mask):
            med_luv = np.median(l_nuv[z_mask])
            print(f"\n  At z~1 ({np.sum(z_mask):,} sources):")
            print(f"    median log L_NUV = {med_luv:.2f}")
            if 9.0 < med_luv < 11.0:
                print(f"    → Consistent with log₁₀(L/L_sun) for SF galaxies ✓")
            else:
                print(f"    ⚠ Unexpected for z~1 SF galaxies (expect ~9.5-10.5)")

    # ── Check 4: β vs E(B-V) relationship ────────────────────────────
    print(f"\n{'─' * 70}")
    print("CHECK 4: β_UV vs E(B-V) slope (should be ~4.4 for Calzetti)")
    print(f"{'─' * 70}")

    if EBV_COL in df.columns and BETA_COL in df.columns:
        ebv = df[EBV_COL].values.astype(float)
        beta = df[BETA_COL].values.astype(float)
        valid = np.isfinite(ebv) & np.isfinite(beta) & (ebv > 0) & (ebv < 1.0)

        if np.sum(valid) > 10:
            # Linear fit: β = a + b × E(B-V)
            coeffs = np.polyfit(ebv[valid], beta[valid], 1)
            slope, intercept = coeffs
            print(f"\n  Linear fit: β = {intercept:.2f} + {slope:.2f} × E(B-V)")
            print(f"    Expected: β = {BETA_INTRINSIC:.1f} + 4.43 × E(B-V) [Calzetti]")
            print(f"    Intercept: {intercept:.2f} (expect {BETA_INTRINSIC:.1f})"
                  f"  {'✓' if abs(intercept - BETA_INTRINSIC) < 0.1 else '⚠'}")
            print(f"    Slope:     {slope:.2f} (expect 4.43)"
                  f"  {'✓' if abs(slope - 4.43) < 0.2 else '⚠'}")

    # ── Check 5: IRX sanity at known β values ────────────────────────
    print(f"\n{'─' * 70}")
    print("CHECK 5: Expected IRX at known β values (Meurer+99 reference)")
    print(f"{'─' * 70}")

    if BETA_COL in df.columns and L_UV_COL in df.columns:
        beta = df[BETA_COL].values.astype(float)
        l_uv = df[L_UV_COL].values.astype(float)
        valid = np.isfinite(beta) & np.isfinite(l_uv)

        print(f"\n  Meurer+99: IRX = 10^(0.4 × (4.43 + 1.99β)) - 1")
        print(f"  At your median β = {np.median(beta[valid]):.2f}:")
        med_beta = np.median(beta[valid])
        expected_irx = 10**(0.4 * (4.43 + 1.99 * med_beta)) - 1
        print(f"    Expected IRX = {expected_irx:.1f}")
        print(f"    Expected log(IRX) = {np.log10(max(expected_irx, 0.01)):.2f}")
        print(f"\n  This is what your stacked L_IR / L_UV should approximately be")
        print(f"  (will scatter due to geometry, metallicity, age variations)")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    if BETA_COL in df.columns:
        beta = df[BETA_COL].values.astype(float)
        valid = np.isfinite(beta)
        print(f"  β_UV: {np.sum(valid):,} valid, "
              f"median={np.median(beta[valid]):.2f}, "
              f"range=[{beta[valid].min():.2f}, {beta[valid].max():.2f}]")
    if L_UV_COL in df.columns:
        l_uv = df[L_UV_COL].values.astype(float)
        valid = np.isfinite(l_uv)
        print(f"  L_UV(1600): {np.sum(valid):,} valid, "
              f"median={np.median(l_uv[valid]):.2f}, "
              f"range=[{l_uv[valid].min():.2f}, {l_uv[valid].max():.2f}]")
    if L_NUV_COL in df.columns and L_UV_COL in df.columns:
        diff = df[L_UV_COL].values - df[L_NUV_COL].values
        valid = np.isfinite(diff)
        print(f"  L_UV - L_NUV correction: median={np.median(diff[valid]):+.3f} dex "
              f"(factor {10**np.median(diff[valid]):.2f}×)")


if __name__ == "__main__":
    main()
