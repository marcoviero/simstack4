"""
COSMOS Catalog Builder for Wijesekera+2026 IRX Analysis
========================================================

Builds a stacking-ready catalog matching the selection criteria of
Wijesekera, Koprowski et al. (2026), A&A — "Evolution of dust attenuation
in star-forming galaxies with UV slope, stellar mass, and redshift out to z~5"

Selection (Section 3.1):
  - Star-forming galaxies only (NUVrJ quiescent removal, Ilbert+2013 Eq 1)
  - NUVrJ cut applied only at z < 4 (rest-frame J hard to trace above)
  - Mass-complete: 90% completeness limits per redshift bin (Table A.1)
  - Starbursts removed: SFR/SFR_MS > 3 (Elbaz+2018 definition)
  - Stars removed via flag_star
  - Redshift range: 0.5 < z <= 5.0

Required catalog columns (from LePhare / CIGALE):
  - Rest-frame absolute magnitudes: M_NUV, M_r, M_J  (for NUVrJ cut)
  - Stellar mass: log(M*/Msun)
  - Photometric redshift
  - SFR (for starburst removal)
  - UV slope beta (from SED fitting, 125-250nm rest-frame)
  - UV luminosity at 1600A (for IRX computation downstream)

Outputs a parquet file ready for simstack4 with the TOML configs below.
"""
import pdb
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table


# =========================================================================
# Mass-completeness limits from Wijesekera+2026, Table A.1
# log(M*/Msun) at 90% completeness per redshift bin
# =========================================================================
MASS_COMPLETENESS = {
    # (z_low, z_high): log_mass_limit
    (0.5, 1.0): 9.25,   # Extrapolated from McLeod+2021
    (1.0, 1.5): 9.25,   # Extrapolated from McLeod+2021
    (1.5, 2.0): 9.50,   # Extrapolated from McLeod+2021
    (2.0, 2.5): 9.50,   # Table A.1
    (2.5, 3.0): 9.75,   # Table A.1
    (3.0, 3.5): 9.75,   # Table A.1
    (3.5, 4.0): 10.00,  # Table A.1
    (4.0, 5.0): 10.25,  # Table A.1
}


def schreiber_main_sequence_sfr(log_mass, z):
    """
    Star-forming main sequence SFR from Schreiber+2015.

    Used as reference for starburst identification (SFR/SFR_MS > 3).
    Eq 9 from Schreiber+2015:
        log(SFR_MS) = m - m0 + a0*r - a1*[max(0, m-m1-a2*r)]^2
    where m = log(M*/Msun), r = log(1+z)
    """
    m0 = 0.5
    a0 = 1.5
    a1 = 0.3
    m1 = 0.36
    a2 = 2.5

    r = np.log10(1 + z)
    m = np.asarray(log_mass, dtype=float)

    term = np.maximum(0, m - m1 - a2 * r)
    log_sfr = m - m0 + a0 * r - a1 * term**2

    return 10**log_sfr


def apply_nuvrj_cut(df, nuv_col, r_col, j_col, z_col):
    """
    NUVrJ quiescent galaxy removal (Ilbert+2013, Eq 1 in Wijesekera+2026).

    Quiescent if BOTH:
        (NUV - r) > 3 * (r - J) + 1
        (NUV - r) > 3.1

    Applied only at z < 4.  At z >= 4, all galaxies are kept (assumed SF).

    Parameters
    ----------
    df : DataFrame
        Must contain rest-frame absolute magnitudes (from SED fitting).
    nuv_col, r_col, j_col : str
        Column names for rest-frame NUV, r, J absolute magnitudes.
    z_col : str
        Redshift column.

    Returns
    -------
    is_starforming : boolean Series
    """
    nuv_r = df[nuv_col] - df[r_col]
    r_j = df[r_col] - df[j_col]

    quiescent = (nuv_r > 3.0 * r_j + 1.0) & (nuv_r > 3.1)

    # Only apply at z < 4
    z = df[z_col]
    quiescent = quiescent & (z < 4.0)

    # Need valid colors to classify
    has_colors = np.isfinite(nuv_r) & np.isfinite(r_j)

    # If colors unavailable, keep the source (don't discard unknowns)
    is_starforming = ~quiescent | ~has_colors

    #pdb.set_trace()
    return is_starforming


def apply_mass_completeness(df, z_col, mass_col):
    """
    Apply redshift-dependent mass-completeness cuts from Table A.1.

    Sources below the 90% completeness limit for their redshift bin
    are removed.
    """
    mask = pd.Series(False, index=df.index)

    for (z_lo, z_hi), mass_lim in MASS_COMPLETENESS.items():
        in_bin = (df[z_col] > z_lo) & (df[z_col] <= z_hi)
        above_limit = df[mass_col] >= mass_lim
        mask |= (in_bin & above_limit)

    return mask


def remove_starbursts(df, mass_col, z_col, sfr_col, threshold=3.0):
    """
    Remove starbursts: SFR/SFR_MS > threshold (Elbaz+2018 definition).

    Uses Schreiber+2015 main sequence as reference.
    """
    sfr_ms = schreiber_main_sequence_sfr(df[mass_col], df[z_col])
    sfr = df[sfr_col]

    # Keep if SFR/SFR_MS <= threshold, or if SFR data unavailable
    is_main_seq = (sfr / sfr_ms) <= threshold
    no_sfr_data = ~np.isfinite(sfr) | (sfr <= 0)

    #pdb.set_trace()

    return is_main_seq | no_sfr_data


def create_wijesekera_catalog(
    photometry_path,
    lephare_path,
    output_path,
    # ---- Column name mappings (adjust to your catalog) ----
    z_col="zfinal",
    mass_col="mass_med",            # log(M*/Msun)
    sfr_col="sfr_med",              # log(SFR) or linear SFR
    sfr_is_log=True,                # True if sfr_col is log10(SFR)
    ra_col="ra",
    dec_col="dec",
    # Rest-frame absolute magnitudes from SED fitting
    # Set to None if not available (NUVrJ cut will be skipped)
    nuv_col="mabs_nuv",                   # e.g. "MNUV" or "restframe_NUV"
    r_col="mabs_r",                     # e.g. "Mr" or "restframe_r"
    j_col_rf="mabs_j",                  # e.g. "MJ" or "restframe_J"
    # UV properties from SED fitting (CIGALE)
    luv_col="l_nuv",                   # e.g. "L_UV" or "LUV_1600"
    ebv_minchi2="ebv_minchi2",
    law_minchi2="law_minchi2",
    star_flag_col="flag_star",
    # Quality
    chi2_col="chi2_best",
    chi2_max=100.0,
):
    """
    Build a stacking-ready catalog matching Wijesekera+2026 selection.

    Parameters
    ----------
    photometry_path : Path
        FITS photometry catalog (RA, Dec, flags).
    lephare_path : Path
        FITS LePhare/CIGALE output (z, mass, SFR, rest-frame mags).
    output_path : Path
        Output parquet file.
    *_col : str or None
        Column name mappings — adjust to match your actual catalog columns.
    """
    print("=" * 70)
    print("Wijesekera+2026 Catalog Builder")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load catalogs
    # ------------------------------------------------------------------
    print(f"\nLoading photometry: {photometry_path}")
    with fits.open(photometry_path) as hdul:
        table = Table(hdul[1].data)
        good_cols = [n for n in table.colnames if len(table[n].shape) <= 1]
        phot_df = table[good_cols].to_pandas()
    print(f"  {len(phot_df):,} sources, {len(phot_df.columns)} columns")

    print(f"Loading LePhare: {lephare_path}")
    with fits.open(lephare_path) as hdul:
        table = Table(hdul[1].data)
        good_cols = [n for n in table.colnames if len(table[n].shape) <= 1]
        lephare_df = table[good_cols].to_pandas()
    print(f"  {len(lephare_df):,} sources, {len(lephare_df.columns)} columns")

    # Merge by row index (standard for COSMOS catalogs)
    assert len(phot_df) == len(lephare_df), (
        f"Catalog length mismatch: {len(phot_df)} vs {len(lephare_df)}"
    )
    df = pd.concat(
        [phot_df.reset_index(drop=True), lephare_df.reset_index(drop=True)],
        axis=1,
    )
    # Remove duplicate columns
    df = df.loc[:, ~df.columns.duplicated()]
    n_initial = len(df)
    print(f"\nMerged catalog: {n_initial:,} sources, {len(df.columns)} columns")

    # ------------------------------------------------------------------
    # 2. Verify required columns
    # ------------------------------------------------------------------
    required = {
        "redshift": z_col,
        "stellar_mass": mass_col,
        "ra": ra_col,
        "dec": dec_col,
    }
    missing = {k: v for k, v in required.items() if v not in df.columns}
    if missing:
        print(f"\n  MISSING REQUIRED COLUMNS: {missing}")
        print(f"  Available columns: {sorted(df.columns.tolist())}")
        raise ValueError(f"Missing columns: {missing}")

    # Report optional columns
    optional = {
        "SFR": sfr_col,
        "NUV (rest)": nuv_col,
        "r (rest)": r_col,
        "J (rest)": j_col_rf,
        "L_UV": luv_col,
        "star_flag": star_flag_col,
        "chi2": chi2_col,
        "ebv_minchi2": ebv_minchi2,
        "law_minchi2": law_minchi2,
    }
    print("\nColumn availability:")
    for label, col in optional.items():
        status = "found" if col and col in df.columns else "MISSING"
        print(f"  {label:>12}: {col or 'not specified':30} [{status}]")

    # ------------------------------------------------------------------
    # 3. Basic quality cuts
    # ------------------------------------------------------------------
    print(f"\n--- Selection cuts ---")
    mask = pd.Series(True, index=df.index)

    # Valid coordinates
    coord_mask = np.isfinite(df[ra_col]) & np.isfinite(df[dec_col])
    mask &= coord_mask
    print(f"  Valid coords:         {mask.sum():>8,} / {n_initial:,}")

    # Valid redshift: 0.5 < z <= 5.0
    z_mask = (df[z_col] > 0.5) & (df[z_col] <= 5.0) & np.isfinite(df[z_col])
    mask &= z_mask
    print(f"  0.5 < z <= 5.0:      {mask.sum():>8,} / {n_initial:,}")

    # Valid stellar mass
    mass_mask = np.isfinite(df[mass_col]) & (df[mass_col] > 7.0)
    mask &= mass_mask
    print(f"  Valid mass:           {mask.sum():>8,} / {n_initial:,}")

    # Star removal
    if star_flag_col and star_flag_col in df.columns:
        star_mask = df[star_flag_col] == 0
        mask &= star_mask
        print(f"  Star removal:         {mask.sum():>8,} / {n_initial:,}")

    # Chi2 quality cut
    if chi2_col and chi2_col in df.columns:
        chi2_mask = df[chi2_col] < chi2_max
        mask &= chi2_mask
        print(f"  chi2 < {chi2_max}:          {mask.sum():>8,} / {n_initial:,}")

    # ------------------------------------------------------------------
    # 4. NUVrJ quiescent removal (Ilbert+2013, Eq 1)
    # ------------------------------------------------------------------
    if nuv_col and r_col and j_col_rf:
        if all(c in df.columns for c in [nuv_col, r_col, j_col_rf]):
            sf_mask = apply_nuvrj_cut(df, nuv_col, r_col, j_col_rf, z_col)
            mask &= sf_mask
            n_quiescent = (~sf_mask & mask).sum()
            print(f"  NUVrJ SF only:        {mask.sum():>8,} / {n_initial:,}"
                  f"  (removed {n_quiescent:,} quiescent)")
        else:
            print(f"  NUVrJ: SKIPPED (columns not found)")
    else:
        print(f"  NUVrJ: SKIPPED (rest-frame mag columns not specified)")
        print(f"         Set nuv_col, r_col, j_col_rf to enable")

    #pdb.set_trace()
    # ------------------------------------------------------------------
    # 5. Mass-completeness cuts (Table A.1)
    # ------------------------------------------------------------------
    mc_mask = apply_mass_completeness(df, z_col, mass_col)
    mask &= mc_mask
    print(f"  Mass-complete:        {mask.sum():>8,} / {n_initial:,}")

    # ------------------------------------------------------------------
    # 6. Starburst removal (SFR/SFR_MS > 3)
    # ------------------------------------------------------------------
    if sfr_col and sfr_col in df.columns:
        if sfr_is_log:
            df["_sfr_linear"] = 10**df[sfr_col]
        else:
            df["_sfr_linear"] = df[sfr_col]

        sb_mask = remove_starbursts(df, mass_col, z_col, "_sfr_linear")
        #mask &= sb_mask
        n_sb = (~sb_mask).sum()
        print(f"  Starburst removal:    {mask.sum():>8,} / {n_initial:,}"
              f"  (flagged {n_sb:,} starbursts)")
        df.drop("_sfr_linear", axis=1, inplace=True)
    else:
        print(f"  Starburst removal: SKIPPED (SFR column not found)")

    # ------------------------------------------------------------------
    # 7. Build output catalog
    # ------------------------------------------------------------------
    df_sel = df[mask].copy()
    print(f"\n  FINAL SAMPLE: {len(df_sel):,} star-forming galaxies"
          f" ({100*len(df_sel)/n_initial:.1f}% of input)")

    # Standardize column names for simstack4
    output_cols = {
        ra_col: "ra",
        dec_col: "dec",
        z_col: "redshift",
        mass_col: "log_stellar_mass",
    }

    if sfr_col and sfr_col in df_sel.columns:
        output_cols[sfr_col] = "log_sfr" if sfr_is_log else "sfr"

    if luv_col and luv_col in df_sel.columns:
        output_cols[luv_col] = "log_l_uv"

    # Keep all original columns plus standardized names
    for old_name, new_name in output_cols.items():
        if old_name in df_sel.columns and old_name != new_name:
            df_sel[new_name] = df_sel[old_name]

    # Compute log(L_UV) if not present (from L_UV or magnitude)
    if "log_l_uv" not in df_sel.columns:
        print("\n  NOTE: log_l_uv column not available.")
        print("        You'll need to add UV luminosity from CIGALE output.")

    if "beta_uv" not in df_sel.columns:
        print("\n  NOTE: beta_uv column not available.")
        print("        You'll need to add UV slope from CIGALE SED fitting")
        print("        (power-law fit to best-fit SED in 125-250nm rest-frame).")

    # ------------------------------------------------------------------
    # 8. Summary statistics
    # ------------------------------------------------------------------
    print(f"\n--- Sample Summary ---")
    for (z_lo, z_hi), mass_lim in MASS_COMPLETENESS.items():
        in_bin = (df_sel["redshift"] > z_lo) & (df_sel["redshift"] <= z_hi)
        n = in_bin.sum()
        if n > 0:
            med_mass = df_sel.loc[in_bin, "log_stellar_mass"].median()
            print(f"  {z_lo:.1f} < z <= {z_hi:.1f}: {n:>7,} sources, "
                  f"mass_lim={mass_lim:.2f}, median_mass={med_mass:.2f}")

    # ------------------------------------------------------------------
    # 9. Save
    # ------------------------------------------------------------------
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_sel.to_parquet(output_path, index=False)
    print(f"\nSaved: {output_path}")
    print(f"  {len(df_sel):,} sources, {len(df_sel.columns)} columns")
    print(f"  Size: {output_path.stat().st_size / 1e6:.1f} MB")

    return df_sel


# =========================================================================
# Execution
# =========================================================================

if __name__ == "__main__":
    base = Path("/Users/mviero/data/Astronomy/catalogs/cosmos")

    df = create_wijesekera_catalog(
        photometry_path=base / "COSMOSWeb_mastercatalog_v1.fits",
        lephare_path=base / "COSMOSWeb_mastercatalog_v1_lephare.fits",
        output_path=base / "COSMOSWeb_wijesekera_sfg.parquet",
        # ---- Adjust these to your actual catalog column names ----
        z_col="zfinal",
        mass_col="mass_med",          # log(M*/Msun)
        sfr_col="sfr_med",            # log(SFR/Msun/yr)
        sfr_is_log=True,
        ra_col="ra",
        dec_col="dec",
        # Rest-frame magnitudes - check your LePhare output!
        # Common names: "MNUV", "Mr", "MJ", "restframe_NUV", etc.
        # Run: Table(fits.open(lephare_path)[1].data).colnames to check
        nuv_col="mabs_nuv",                 # TODO: set to your NUV rest-frame column
        r_col="mabs_r",                   # TODO: set to your r rest-frame column
        j_col_rf="mabs_j",                # TODO: set to your J rest-frame column
        # UV properties from CIGALE - set once available
        luv_col="l_nuv",                 # TODO: set to L_UV column
        ebv_minchi2="ebv_minchi2",
        law_minchi2="law_minchi2",
        star_flag_col="flag_star",
        chi2_col="chi2_best",
        chi2_max=100.0,
    )
