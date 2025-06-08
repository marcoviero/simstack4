# =============================================================================
# FIXED COSMOS MERGER - Handles Multidimensional Columns
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


def create_clean_cosmos_catalog_fixed(photometry_path, lephare_path, output_path):
    """
    Create a clean COSMOS catalog with only essential columns for stacking
    FIXED to handle multidimensional columns in FITS files

    Args:
        photometry_path: Path to COSMOSWeb_mastercatalog_v1.fits (with RA/Dec)
        lephare_path: Path to COSMOSWeb_mastercatalog_v1_lephare.fits (with z, mass)
        output_path: Path to save clean catalog

    Returns:
        Clean pandas DataFrame with only essential columns
    """

    print("=== Creating Clean COSMOS Catalog (FIXED) ===")

    # Define exactly what columns we need
    ESSENTIAL_PHOTOMETRY_COLUMNS = [
        "id",  # Source ID for matching
        "ra",  # Right Ascension
        "dec",  # Declination
        "mag_model_cfht-u",  # U-band magnitude (for UVJ)
        "mag_model_hsc-r",  # R-band magnitude (V-proxy for UVJ)
        "mag_model_uvista-j",  # J-band magnitude (for UVJ)
        "mag_err_model_cfht-u",  # U-band error
        "mag_err_model_hsc-r",  # R-band error
        "mag_err_model_uvista-j",  # J-band error
        "flag_star",  # Star flag
        "flag_blend",  # Blending flag
    ]

    ESSENTIAL_LEPHARE_COLUMNS = [
        "zfinal",  # Final redshift
        "mass_med",  # Median stellar mass
        "zpdf_med",  # Photometric redshift median
        "zpdf_l68",  # Photometric redshift lower 68%
        "zpdf_u68",  # Photometric redshift upper 68%
        "chi2_best",  # Best-fit chi-squared
        "age_med",  # Median age
        "sfr_med",  # Median star formation rate
        "ssfr_med",  # Median specific star formation rate
    ]

    print(f"Loading photometry catalog: {photometry_path}")
    with fits.open(photometry_path) as hdul:
        table = Table(hdul[1].data)

        # FIXED: Filter out multidimensional columns before converting to pandas
        print(f"  - Original columns: {len(table.colnames)}")

        # Get only 1D columns
        good_colnames = [name for name in table.colnames if len(table[name].shape) <= 1]
        print(f"  - 1D columns: {len(good_colnames)}")
        print(
            f"  - Filtered out {len(table.colnames) - len(good_colnames)} multidimensional columns"
        )

        # Create filtered table with only 1D columns
        filtered_table = table[good_colnames]
        phot_df = filtered_table.to_pandas()

    print(f"  - Total sources: {len(phot_df):,}")
    print(f"  - Available columns: {len(phot_df.columns)}")

    # Check which essential photometry columns are available
    available_phot_cols = []
    missing_phot_cols = []

    for col in ESSENTIAL_PHOTOMETRY_COLUMNS:
        if col in phot_df.columns:
            available_phot_cols.append(col)
        else:
            missing_phot_cols.append(col)

    print(
        f"  - Essential columns found: {len(available_phot_cols)}/{len(ESSENTIAL_PHOTOMETRY_COLUMNS)}"
    )

    if missing_phot_cols:
        print(f"  - Missing essential columns: {missing_phot_cols}")

        # Try to find alternative columns
        print("  - Looking for alternative columns...")

        # Check for alternative U, V, J bands
        available_mags = [col for col in phot_df.columns if "mag_model" in col]
        print(f"    Available magnitude columns: {len(available_mags)}")

        # Look for U-band alternatives
        u_alternatives = [
            col
            for col in available_mags
            if any(u in col.lower() for u in ["_u_", "-u_", "_u-", "cfht-u", "hsc-u"])
        ]
        if u_alternatives and "mag_model_cfht-u" not in available_phot_cols:
            print(f"    U-band alternatives: {u_alternatives[:3]}")
            if u_alternatives:
                available_phot_cols.append(u_alternatives[0])
                ESSENTIAL_PHOTOMETRY_COLUMNS[3] = u_alternatives[0]  # Update mapping

        # Look for V/R-band alternatives
        v_alternatives = [
            col
            for col in available_mags
            if any(
                v in col.lower() for v in ["_r_", "-r_", "_r-", "hsc-r", "_v_", "-v_"]
            )
        ]
        if v_alternatives and "mag_model_hsc-r" not in available_phot_cols:
            print(f"    V/R-band alternatives: {v_alternatives[:3]}")
            if v_alternatives:
                available_phot_cols.append(v_alternatives[0])
                ESSENTIAL_PHOTOMETRY_COLUMNS[4] = v_alternatives[0]  # Update mapping

        # Look for J-band alternatives
        j_alternatives = [
            col
            for col in available_mags
            if any(
                j in col.lower() for j in ["_j_", "-j_", "_j-", "uvista-j", "vista-j"]
            )
        ]
        if j_alternatives and "mag_model_uvista-j" not in available_phot_cols:
            print(f"    J-band alternatives: {j_alternatives[:3]}")
            if j_alternatives:
                available_phot_cols.append(j_alternatives[0])
                ESSENTIAL_PHOTOMETRY_COLUMNS[5] = j_alternatives[0]  # Update mapping

    # Extract available essential photometry columns
    final_phot_cols = [col for col in available_phot_cols if col in phot_df.columns]
    if not final_phot_cols:
        print("  ❌ No essential photometry columns found!")
        return None

    clean_phot_df = phot_df[final_phot_cols].copy()
    print(f"  ✓ Extracted {len(final_phot_cols)} essential columns")

    print(f"\nLoading LePhare catalog: {lephare_path}")
    with fits.open(lephare_path) as hdul:
        lephare_table = Table(hdul[1].data)

        # Filter 1D columns for LePhare too (just in case)
        lephare_good_cols = [
            name
            for name in lephare_table.colnames
            if len(lephare_table[name].shape) <= 1
        ]
        lephare_filtered = lephare_table[lephare_good_cols]
        lephare_df = lephare_filtered.to_pandas()

    print(f"  - Total sources: {len(lephare_df):,}")
    print(f"  - Available columns: {len(lephare_df.columns)}")

    # Check which essential LePhare columns are available
    available_lephare_cols = [
        col for col in ESSENTIAL_LEPHARE_COLUMNS if col in lephare_df.columns
    ]
    missing_lephare_cols = [
        col for col in ESSENTIAL_LEPHARE_COLUMNS if col not in lephare_df.columns
    ]

    print(
        f"  - Essential columns found: {len(available_lephare_cols)}/{len(ESSENTIAL_LEPHARE_COLUMNS)}"
    )
    if missing_lephare_cols:
        print(f"  - Missing LePhare columns: {missing_lephare_cols}")

    if not available_lephare_cols:
        print("  ❌ No essential LePhare columns found!")
        return None

    clean_lephare_df = lephare_df[available_lephare_cols].copy()

    print("\nMerging catalogs...")

    # Check if catalogs have same length (most common case for COSMOS)
    if len(clean_phot_df) == len(clean_lephare_df):
        print(
            f"  ✓ Same length ({len(clean_phot_df):,} sources) - merging by row index"
        )

        # Merge by concatenating along columns (assuming same order)
        clean_merged_df = pd.concat(
            [
                clean_phot_df.reset_index(drop=True),
                clean_lephare_df.reset_index(drop=True),
            ],
            axis=1,
        )

    else:
        print("  ❌ Different lengths:")
        print(f"     Photometry: {len(clean_phot_df):,}")
        print(f"     LePhare: {len(clean_lephare_df):,}")
        print("  ⚠️  This might still work - trying to merge anyway...")

        # Take the minimum length and merge
        min_length = min(len(clean_phot_df), len(clean_lephare_df))
        clean_phot_df = clean_phot_df.iloc[:min_length].reset_index(drop=True)
        clean_lephare_df = clean_lephare_df.iloc[:min_length].reset_index(drop=True)

        clean_merged_df = pd.concat([clean_phot_df, clean_lephare_df], axis=1)
        print(
            f"  ✓ Merged {len(clean_merged_df):,} sources (truncated to minimum length)"
        )

    # Try to calculate UVJ colors
    print("\nCalculating UVJ colors...")

    # Find the magnitude columns we actually have
    mag_cols = [col for col in clean_merged_df.columns if "mag_model" in col]

    # Look for U, V/R, J bands
    u_col = None
    v_col = None
    j_col = None

    for col in mag_cols:
        if any(u in col.lower() for u in ["cfht-u", "hsc-u", "_u_"]):
            u_col = col
        elif any(v in col.lower() for v in ["hsc-r", "_r_", "_v_"]):
            v_col = col
        elif any(j in col.lower() for j in ["uvista-j", "vista-j", "_j_"]):
            j_col = col

    print(f"  - U-band: {u_col}")
    print(f"  - V-band: {v_col}")
    print(f"  - J-band: {j_col}")

    if u_col and v_col and j_col:
        clean_merged_df["U-V"] = clean_merged_df[u_col] - clean_merged_df[v_col]
        clean_merged_df["V-J"] = clean_merged_df[v_col] - clean_merged_df[j_col]

        # UVJ classification (Whitaker et al. 2011)
        uv = clean_merged_df["U-V"]
        vj = clean_merged_df["V-J"]

        quiescent_mask = (uv > 1.3) & (vj < 1.6) & (uv > 0.88 * vj + 0.59)
        clean_merged_df["UVJ_class"] = quiescent_mask.astype(int)

        n_sf = np.sum(~quiescent_mask & np.isfinite(uv) & np.isfinite(vj))
        n_q = np.sum(quiescent_mask & np.isfinite(uv) & np.isfinite(vj))
        print(f"  ✓ UVJ classification: {n_sf:,} star-forming, {n_q:,} quiescent")
    else:
        print("  ❌ Cannot calculate UVJ - missing magnitude columns")
        # Add dummy UVJ columns
        clean_merged_df["U-V"] = np.nan
        clean_merged_df["V-J"] = np.nan
        clean_merged_df["UVJ_class"] = 0

    # Add quality flags
    print("\nAdding quality flags...")

    # Basic quality cuts
    good_coords = (clean_merged_df.get("ra", pd.Series([0])) > 0) & (
        clean_merged_df.get("dec", pd.Series([0])) != 0
    )

    if "zfinal" in clean_merged_df.columns:
        good_redshift = (clean_merged_df["zfinal"] > 0.01) & (
            clean_merged_df["zfinal"] < 6.0
        )
    else:
        good_redshift = pd.Series([True] * len(clean_merged_df))

    if "mass_med" in clean_merged_df.columns:
        good_mass = (clean_merged_df["mass_med"] > 7.0) & (
            clean_merged_df["mass_med"] < 13.0
        )
    else:
        good_mass = pd.Series([True] * len(clean_merged_df))

    clean_merged_df["quality_flag"] = (good_coords & good_redshift & good_mass).astype(
        int
    )

    n_good = np.sum(clean_merged_df["quality_flag"] == 1)
    print(
        f"  ✓ Quality flag: {n_good:,}/{len(clean_merged_df):,} sources pass basic cuts"
    )

    # Final summary
    print("\n=== Clean Catalog Summary ===")
    print(f"Total sources: {len(clean_merged_df):,}")
    print(f"Total columns: {len(clean_merged_df.columns)}")
    print(
        f"File size estimate: ~{len(clean_merged_df) * len(clean_merged_df.columns) * 8 / 1024 ** 2:.1f} MB"
    )

    print("\nFinal columns:")
    for i, col in enumerate(clean_merged_df.columns, 1):
        print(f"  {i:2d}. {col}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save clean catalog
    print(f"\nSaving clean catalog to: {output_path}")

    if output_path.suffix == ".parquet":
        clean_merged_df.to_parquet(output_path, index=False)
    elif output_path.suffix == ".fits":
        # Convert to FITS
        clean_table = Table.from_pandas(clean_merged_df)
        clean_table.write(output_path, format="fits", overwrite=True)
    else:
        # Default to CSV
        clean_merged_df.to_csv(output_path, index=False)

    print("✅ Clean COSMOS catalog created successfully!")

    return clean_merged_df


# =============================================================================
# QUICK EXECUTION SCRIPT (FIXED)
# =============================================================================


def run_clean_cosmos_merger_fixed():
    """
    Execute the FIXED clean COSMOS catalog merger
    """

    # Your file paths
    photometry_path = Path(
        "/Users/mviero/data/Astronomy/catalogs/cosmos/COSMOSWeb_mastercatalog_v1.fits"
    )
    lephare_path = Path(
        "/Users/mviero/data/Astronomy/catalogs/cosmos/COSMOSWeb_mastercatalog_v1_lephare.fits"
    )
    output_path = Path(
        "/Users/mviero/data/Astronomy/catalogs/cosmos/COSMOSWeb_clean.parquet"
    )

    # Check files exist
    if not photometry_path.exists():
        print(f"❌ Photometry catalog not found: {photometry_path}")
        return None

    if not lephare_path.exists():
        print(f"❌ LePhare catalog not found: {lephare_path}")
        return None

    # Create clean catalog
    clean_df = create_clean_cosmos_catalog_fixed(
        photometry_path, lephare_path, output_path
    )

    if clean_df is not None:
        print("\n🎉 SUCCESS!")
        print(f"Clean catalog saved to: {output_path}")
        print(f"Reduced from 330+ columns to {len(clean_df.columns)} essential columns")
        print("Ready for Simstack4!")

        # Show what columns we actually got
        essential_cols = ["id", "ra", "dec", "zfinal", "mass_med"]
        print("\nEssential columns check:")
        for col in essential_cols:
            status = "✓" if col in clean_df.columns else "❌"
            print(f"  {status} {col}")

        print("\nTo use with Simstack4:")
        print("1. Update your cosmos25.toml:")
        print('   file = "COSMOSWeb_clean.parquet"')
        print("2. Update column mapping in cosmos.py based on actual columns found")

        return output_path

    return None


# Execute if run directly
if __name__ == "__main__":
    output_path = run_clean_cosmos_merger_fixed()

# Run the fixed version
run_clean_cosmos_merger_fixed()
