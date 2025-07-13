# =============================================================================
# COSMOS CATALOG CLEANER - Fixed and Simplified
# =============================================================================

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


def create_clean_cosmos_catalog(
    catalog_path,
    output_path,
    sn_cut=None,
    mag_cut=None,
    keep_all=False,
    keep_stars=False,
    verbose=True,
):
    """
    Create a clean COSMOS catalog from the master catalog FITS file

    Args:
        catalog_path: Path to COSMOSWeb_mastercatalog_v1.fits
        output_path: Path to save clean catalog
        sn_cut: Optional S/N cut (e.g., 5.0 for 5-sigma detection)
        mag_cut: Optional magnitude cut (e.g., 22.0 for mag < 22)
        keep_all: If True, keep all objects of selected type regardless of quality flags
        keep_stars: If True, select stars (type==1) instead of galaxies (type==0)
        verbose: Print detailed information

    Selection Logic:
        Default: Clean galaxies only (type==0, good quality flags)
        --keep_all: All galaxies, ignore quality flags
        --keep_stars: Stars only (type==1), with quality cuts applied
        --sn X: Apply S/N threshold to whatever objects are selected
        --mag X: Apply magnitude cut (mag < X) to whatever objects are selected

    Returns:
        Clean pandas DataFrame
    """

    if verbose:
        print("=== Creating Clean COSMOS Catalog ===")
        print(f"Input catalog: {catalog_path}")

        if keep_stars:
            print("SELECTION: Stars only (type==1)")
        else:
            print("SELECTION: Galaxies only (type==0)")

        if keep_all:
            print("QUALITY: Keep all objects, ignore quality flags")
        else:
            print("QUALITY: Apply quality cuts")

        if sn_cut:
            print(f"S/N CUT: {sn_cut}")
        if mag_cut:
            print(f"MAG CUT: < {mag_cut}")

        print()  # Empty line for readability

    # Load the master catalog with all HDUs
    with fits.open(catalog_path) as hdul:
        if verbose:
            print("\nFITS file structure:")
            hdul.info()

        # Extract catalog extensions
        print("\nLoading catalog extensions...")

        # HDU 1: Main photometry catalog
        cat_photom = Table(hdul[1].data)
        print(
            f"  - Photometry: {len(cat_photom):,} sources, {len(cat_photom.colnames)} columns"
        )

        # HDU 2: LePhare catalog (SED fitting results)
        cat_lephare = Table(hdul[2].data)
        print(
            f"  - LePhare: {len(cat_lephare):,} sources, {len(cat_lephare.colnames)} columns"
        )

        # Filter out multidimensional columns before converting to pandas
        # (aperture photometry columns often have multiple aperture sizes)
        print("  - Filtering multidimensional columns...")

        # For photometry catalog
        photom_1d_cols = [
            name for name in cat_photom.colnames if len(cat_photom[name].shape) <= 1
        ]
        print(
            f"    Photometry: {len(photom_1d_cols)}/{len(cat_photom.colnames)} columns are 1D"
        )
        cat_photom_filtered = cat_photom[photom_1d_cols]

        # For LePhare catalog
        lephare_1d_cols = [
            name for name in cat_lephare.colnames if len(cat_lephare[name].shape) <= 1
        ]
        print(
            f"    LePhare: {len(lephare_1d_cols)}/{len(cat_lephare.colnames)} columns are 1D"
        )
        cat_lephare_filtered = cat_lephare[lephare_1d_cols]

        # Convert filtered tables to pandas
        df_photom = cat_photom_filtered.to_pandas()
        df_lephare = cat_lephare_filtered.to_pandas()

    # Define essential columns
    PHOTOM_ESSENTIAL = [
        "id",
        "ra",
        "dec",  # Basic info
        "sersic",
        "axis_ratio",
        "position_angle",  # Morphology
        "warn_flag",
        "flag_star_hsc",  # Quality flags
        "flux_model_f444w",
        "flux_err-cal_model_f444w",  # For S/N
        "mag_model_f444w",
        "mag_err_model_f444w",  # Magnitudes
        "area_arcsec2",
        "flux_radius_arcsec",  # Size info
    ]

    LEPHARE_ESSENTIAL = [
        "type",  # Object type
        "zfinal",
        "zpdf_med",
        "zpdf_l68",
        "zpdf_u68",  # Redshift
        "mass_med",
        "sfr_med",
        "ssfr_med",
        "age_med",  # Physical properties
        "ebv_minchi2",
        "law_minchi2",
        "chi2_best",  # Dust/fitting
        # REST-FRAME ABSOLUTE MAGNITUDES
        "mabs_nuv",
        "mabs_u",
        "mabs_b",
        "mabs_v",
        "mabs_r",
        "mabs_i",
        "mabs_j",
        "mabs_h",
        "mabs_k",
        "l_nuv",
        "beta_best",  # UV properties
    ]

    # Check available columns
    available_photom = [col for col in PHOTOM_ESSENTIAL if col in df_photom.columns]
    available_lephare = [col for col in LEPHARE_ESSENTIAL if col in df_lephare.columns]

    if verbose:
        print("\nColumn availability:")
        print(
            f"  - Photometry: {len(available_photom)}/{len(PHOTOM_ESSENTIAL)} available"
        )
        print(
            f"  - LePhare: {len(available_lephare)}/{len(LEPHARE_ESSENTIAL)} available"
        )

    # Extract available columns and merge
    clean_photom = df_photom[available_photom].copy()
    clean_lephare = df_lephare[available_lephare].copy()

    # Check catalog lengths
    if len(clean_photom) != len(clean_lephare):
        print("❌ WARNING: Catalog lengths don't match!")
        print(f"   Photometry: {len(clean_photom):,}")
        print(f"   LePhare: {len(clean_lephare):,}")
        min_len = min(len(clean_photom), len(clean_lephare))
        clean_photom = clean_photom.iloc[:min_len]
        clean_lephare = clean_lephare.iloc[:min_len]
        print(f"   Using minimum: {min_len:,} sources")

    # Merge catalogs
    df = pd.concat(
        [clean_photom.reset_index(drop=True), clean_lephare.reset_index(drop=True)],
        axis=1,
    )

    if verbose:
        print(f"Merged catalog: {len(df):,} sources, {len(df.columns)} columns")

    # Apply selection criteria
    print("\n=== APPLYING SELECTION ===")
    original_count = len(df)

    # Step 1: Object type selection
    if "type" in df.columns:
        if keep_stars:
            type_mask = df["type"] == 1  # Stars
            object_type = "stars"
        else:
            type_mask = df["type"] == 0  # Galaxies
            object_type = "galaxies"

        n_galaxies = np.sum(df["type"] == 0)
        n_stars = np.sum(df["type"] == 1)
        n_qsos = np.sum(df["type"] == 2)

        print(
            f"Object types: {n_galaxies:,} galaxies, {n_stars:,} stars, {n_qsos:,} QSOs"
        )
        print(f"Selected: {np.sum(type_mask):,} {object_type}")
    else:
        type_mask = pd.Series([True] * len(df))
        object_type = "stars" if keep_stars else "galaxies"
        print(f"No type column - assuming all are {object_type}")

    # Step 2: Quality cuts (if not keeping all)
    if keep_all:
        quality_mask = pd.Series([True] * len(df))
        print(f"Keeping all {object_type} regardless of quality")
    else:
        print(f"Applying quality cuts to {object_type}:")

        # Basic coordinate check
        good_coords = (df["ra"] > 0) & (df["dec"] != 0)
        print(f"  - Good coordinates: {np.sum(good_coords):,}/{len(df):,}")

        # Warning flags
        if "warn_flag" in df.columns:
            good_warn = df["warn_flag"] == 0
            print(f"  - Good warning flags: {np.sum(good_warn):,}/{len(df):,}")
        else:
            good_warn = pd.Series([True] * len(df))

        # Star classification flag (different logic for stars vs galaxies)
        if "flag_star_hsc" in df.columns:
            if keep_stars:
                good_star_flag = (
                    df["flag_star_hsc"] == 1
                )  # Want objects flagged as stars
                print(f"  - Flagged as stars: {np.sum(good_star_flag):,}/{len(df):,}")
            else:
                good_star_flag = (
                    df["flag_star_hsc"] == 0
                )  # Want objects NOT flagged as stars
                print(
                    f"  - Not flagged as stars: {np.sum(good_star_flag):,}/{len(df):,}"
                )
        else:
            good_star_flag = pd.Series([True] * len(df))

        # For galaxies: redshift, mass, and magnitude cuts
        if not keep_stars:
            if "zfinal" in df.columns:
                good_z = (df["zfinal"] > 0) & (df["zfinal"] < 7)
                print(f"  - Good redshifts: {np.sum(good_z):,}/{len(df):,}")
            else:
                good_z = pd.Series([True] * len(df))

            if "mass_med" in df.columns:
                good_mass = (df["mass_med"] > 7) & (df["mass_med"] < 13)
                print(f"  - Good masses: {np.sum(good_mass):,}/{len(df):,}")
            else:
                good_mass = pd.Series([True] * len(df))

            # Basic magnitude sanity check (reject obvious bad values)
            mag_columns = ["mag_model_f444w", "mag_model_f277w", "mag_model_f150w"]
            available_mag_cols = [col for col in mag_columns if col in df.columns]
            if available_mag_cols:
                # Use the first available magnitude column for sanity check
                mag_col = available_mag_cols[0]
                good_mag_sanity = np.abs(df[mag_col]) < 30  # Reject crazy magnitudes
                print(
                    f"  - Reasonable {mag_col}: {np.sum(good_mag_sanity):,}/{len(df):,}"
                )
            else:
                good_mag_sanity = pd.Series([True] * len(df))
                print("  - No magnitude columns for sanity check")
        else:
            # For stars, skip redshift/mass cuts but still check magnitudes
            good_z = pd.Series([True] * len(df))
            good_mass = pd.Series([True] * len(df))

            # Basic magnitude sanity check for stars too
            mag_columns = ["mag_model_f444w", "mag_model_f277w", "mag_model_f150w"]
            available_mag_cols = [col for col in mag_columns if col in df.columns]
            if available_mag_cols:
                mag_col = available_mag_cols[0]
                good_mag_sanity = np.abs(df[mag_col]) < 30
                print(
                    f"  - Reasonable {mag_col}: {np.sum(good_mag_sanity):,}/{len(df):,}"
                )
            else:
                good_mag_sanity = pd.Series([True] * len(df))

        # Combine quality cuts
        quality_mask = (
            good_coords
            & good_warn
            & good_star_flag
            & good_z
            & good_mass
            & good_mag_sanity
        )
        print(f"  - Combined quality: {np.sum(quality_mask):,}/{len(df):,}")

    # Step 3: S/N cut (if requested)
    if (
        sn_cut
        and "flux_model_f444w" in df.columns
        and "flux_err-cal_model_f444w" in df.columns
    ):
        sn_ratio = df["flux_model_f444w"] / df["flux_err-cal_model_f444w"]
        sn_mask = sn_ratio > sn_cut
        print(f"S/N > {sn_cut}: {np.sum(sn_mask):,}/{len(df):,}")
    else:
        sn_mask = pd.Series([True] * len(df))
        if sn_cut:
            print("S/N cut requested but flux columns not available")

    # Step 4: Magnitude cut (if requested)
    if mag_cut:
        # Try different magnitude columns in order of preference
        mag_columns = [
            "mag_model_f277w",
            "mag_model_f444w",
            "mag_model_f150w",
            "mag_model_f115w",
        ]
        mag_mask = pd.Series([True] * len(df))
        mag_col_used = None

        for mag_col in mag_columns:
            if mag_col in df.columns:
                mag_mask = df[mag_col] < mag_cut
                mag_col_used = mag_col
                print(
                    f"Magnitude {mag_col} < {mag_cut}: {np.sum(mag_mask):,}/{len(df):,}"
                )
                break

        if mag_col_used is None:
            print("Magnitude cut requested but no suitable magnitude columns found")
            print(
                f"Available columns: {[col for col in df.columns if 'mag_model' in col][:5]}"
            )
    else:
        mag_mask = pd.Series([True] * len(df))

    # Combine all selection criteria
    final_mask = type_mask & quality_mask & sn_mask & mag_mask

    print("\nFINAL SELECTION:")
    print(f"  - Type selection: {np.sum(type_mask):,}")
    print(f"  - Quality cuts: {np.sum(quality_mask):,}")
    print(f"  - S/N cuts: {np.sum(sn_mask):,}")
    print(f"  - Magnitude cuts: {np.sum(mag_mask):,}")
    print(f"  - FINAL: {np.sum(final_mask):,}/{original_count:,} {object_type}")

    # Apply selection
    df_selected = df[final_mask].copy().reset_index(drop=True)

    # Add flag columns for transparency
    df_selected["quality_flag"] = 1
    df_selected["object_type"] = object_type

    # Calculate rest-frame colors (galaxies only)
    if not keep_stars and len(df_selected) > 0:
        print("\nCalculating rest-frame colors...")

        # UVJ colors
        if all(col in df_selected.columns for col in ["mabs_u", "mabs_v", "mabs_j"]):
            df_selected["rest_U_V"] = df_selected["mabs_u"] - df_selected["mabs_v"]
            df_selected["rest_V_J"] = df_selected["mabs_v"] - df_selected["mabs_j"]

            # UVJ classification
            uv = df_selected["rest_U_V"]
            vj = df_selected["rest_V_J"]
            quiescent_uvj = (uv > 1.3) & (vj < 1.6) & (uv > 0.88 * vj + 0.59)
            df_selected["UVJ_class"] = quiescent_uvj.astype(int)

            n_sf = np.sum(~quiescent_uvj & np.isfinite(uv) & np.isfinite(vj))
            n_q = np.sum(quiescent_uvj & np.isfinite(uv) & np.isfinite(vj))
            print(f"  UVJ classification: {n_sf:,} star-forming, {n_q:,} quiescent")
        else:
            df_selected["rest_U_V"] = np.nan
            df_selected["rest_V_J"] = np.nan
            df_selected["UVJ_class"] = 0
            print("  Cannot calculate UVJ - missing columns")

        # NUVRJ colors
        if all(col in df_selected.columns for col in ["mabs_nuv", "mabs_r", "mabs_j"]):
            df_selected["rest_NUV_R"] = df_selected["mabs_nuv"] - df_selected["mabs_r"]
            df_selected["rest_R_J"] = df_selected["mabs_r"] - df_selected["mabs_j"]

            # NUVRJ classification (your criteria)
            nuv_r = df_selected["rest_NUV_R"]
            r_j = df_selected["rest_R_J"]
            criterion_1 = nuv_r > (3 * r_j + 1)
            criterion_2 = nuv_r > 3.1
            quiescent_nuvrj = criterion_1 & criterion_2
            df_selected["NUVRJ_class"] = quiescent_nuvrj.astype(int)

            n_sf = np.sum(~quiescent_nuvrj & np.isfinite(nuv_r) & np.isfinite(r_j))
            n_q = np.sum(quiescent_nuvrj & np.isfinite(nuv_r) & np.isfinite(r_j))
            print(f"  NUVRJ classification: {n_sf:,} star-forming, {n_q:,} quiescent")
        else:
            df_selected["rest_NUV_R"] = np.nan
            df_selected["rest_R_J"] = np.nan
            df_selected["NUVRJ_class"] = 0
            print("  Cannot calculate NUVRJ - missing columns")
    else:
        # Add dummy columns for stars or empty selection
        df_selected["rest_U_V"] = np.nan
        df_selected["rest_V_J"] = np.nan
        df_selected["UVJ_class"] = 0
        df_selected["rest_NUV_R"] = np.nan
        df_selected["rest_R_J"] = np.nan
        df_selected["NUVRJ_class"] = 0

    # Add S/N column if available
    if (
        "flux_model_f444w" in df_selected.columns
        and "flux_err-cal_model_f444w" in df_selected.columns
    ):
        df_selected["sn_f444w"] = (
            df_selected["flux_model_f444w"] / df_selected["flux_err-cal_model_f444w"]
        )

    # Generate filename
    output_path = Path(output_path)
    suffix_parts = []

    if keep_stars:
        suffix_parts.append("stars")
    else:
        suffix_parts.append("galaxies")

    if sn_cut:
        suffix_parts.append(f"sn{sn_cut}")

    if mag_cut:
        suffix_parts.append(f"mag{mag_cut}")

    if keep_all:
        suffix_parts.append("keepall")
    else:
        suffix_parts.append("clean")

    # Update filename
    stem = output_path.stem
    suffix = output_path.suffix
    new_name = f"{stem}_{'_'.join(suffix_parts)}{suffix}"
    output_path = output_path.parent / new_name

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save catalog
    print(f"\nSaving to: {output_path}")

    if output_path.suffix == ".parquet":
        df_selected.to_parquet(output_path, index=False)
    elif output_path.suffix == ".fits":
        clean_table = Table.from_pandas(df_selected)
        clean_table.write(output_path, format="fits", overwrite=True)
    else:
        df_selected.to_csv(output_path, index=False)

    # Summary
    print("\n=== SUMMARY ===")
    print(f"Selected: {len(df_selected):,} {object_type}")
    print(
        f"Reduction: {original_count:,} → {len(df_selected):,} ({100 * len(df_selected) / original_count:.1f}% kept)"
    )
    print(f"Columns: {len(df_selected.columns)}")

    print(f"\n✅ SUCCESS! Clean {object_type} catalog ready for analysis")

    return df_selected


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description="Clean COSMOS catalog for stacking analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: Clean galaxies with S/N cut
  python script.py catalog.fits output.parquet --sn 5.0

  # Magnitude cut only (F277W < 22)
  python script.py catalog.fits output.parquet --mag 22.0

  # Combined S/N and magnitude cuts
  python script.py catalog.fits output.parquet --sn 5.0 --mag 22.0

  # All galaxies (ignore quality flags)
  python script.py catalog.fits output.parquet --keep_all

  # Clean stars with magnitude cut
  python script.py catalog.fits output.parquet --keep_stars --mag 20.0
        """,
    )

    parser.add_argument("catalog_path", help="Path to COSMOSWeb_mastercatalog_v1.fits")
    parser.add_argument("output_path", help="Path for output clean catalog")
    parser.add_argument("--sn", type=float, help="S/N cut threshold (e.g., 5.0)")
    parser.add_argument(
        "--mag", type=float, help="Magnitude cut threshold (e.g., 22.0 for mag < 22)"
    )
    parser.add_argument(
        "--keep_all", action="store_true", help="Keep all objects, ignore quality flags"
    )
    parser.add_argument(
        "--keep_stars", action="store_true", help="Select stars instead of galaxies"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Run the cleaning
    clean_df = create_clean_cosmos_catalog(
        catalog_path=args.catalog_path,
        output_path=args.output_path,
        sn_cut=args.sn,
        mag_cut=args.mag,
        keep_all=args.keep_all,
        keep_stars=args.keep_stars,
        verbose=not args.quiet,
    )

    return clean_df


if __name__ == "__main__":
    clean_df = main()
