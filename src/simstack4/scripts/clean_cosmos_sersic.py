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
    # z < 0.5: very deep, nearly complete
    (0.01, 0.5): 8.50,  # Conservative for COSMOS depth
    # Wijesekera+2026 Table A.1 + extrapolations
    (0.5, 1.0): 9.25,
    (1.0, 1.5): 9.25,
    (1.5, 2.0): 9.50,
    (2.0, 2.5): 9.50,
    (2.5, 3.0): 9.75,
    (3.0, 3.5): 9.75,
    (3.5, 4.0): 10.00,
    (4.0, 4.5): 10.25,
    (4.5, 5.0): 10.25,
    # High-z: extrapolated, JWST-era estimates
    (5.0, 6.0): 10.50,
    (6.0, 7.0): 10.75,
    (7.0, 8.0): 11.00,
    (8.0, 10.0): 11.25,
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
    m = np.asarray(log_mass, dtype=float) - 9.0  # Schreiber uses m = log(M/10⁹ M☉)

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

    # pdb.set_trace()
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
        mask |= in_bin & above_limit

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

    # pdb.set_trace()

    return is_main_seq | no_sfr_data


# ── Photometric β_UV and L_UV(1600Å) ────────────────────────────────────

# Effective wavelengths (Å) for COSMOS2025 filters
# Broadband: CFHT MegaCam, HSC, HST/ACS, JWST/NIRCam
# Intermediate: Subaru Suprime-Cam IA/IB filters
_BAND_WAVELENGTHS = {
    "cfht-u": 3709,
    "sc-ib427": 4270,
    "sc-ia484": 4847,
    "sc-ib505": 5050,
    "sc-ia527": 5270,
    "sc-ib574": 5740,
    "sc-ia624": 6240,
    "sc-ia679": 6790,
    "sc-ib709": 7090,
    "sc-ia738": 7380,
    "sc-ia767": 7670,
    "sc-ib827": 8270,
    "hsc-g": 4770,
    "hsc-r": 6175,
    "hsc-i": 7711,
    "hsc-z": 8898,
    "hsc-y": 9762,
    "hst-f814w": 8140,
    "f115w": 11534,
    "f150w": 15007,
}


def compute_photometric_beta_luv(
    df,
    z_col="redshift",
    flux_prefix="flux_model",
    err_prefix="flux_err-cal_model",
    uv_window=(1300, 2600),
    target_lambda=1600.0,
    min_bands=2,
    min_snr=2.0,
    z_min=0.75,
):
    """
    Compute β_UV and L_UV(1600Å) directly from broadband photometry,
    independent of SED fitting.

    Method
    ------
    The UV spectral slope β is defined as f_λ ∝ λ^β (Calzetti+1994).
    For broadband photometry, β is measured by fitting a power law to
    observed flux densities whose effective wavelengths fall within a
    rest-frame UV window (default 1300–2600 Å).

    Since f_ν ∝ λ^(β+2), we fit:
        log₁₀(f_ν) = (β + 2) × log₁₀(λ_rest) + const

    L_UV(1600Å) is obtained by evaluating the best-fit power law at
    rest-frame 1600 Å and converting to luminosity:
        L_ν = 4π d_L² (1+z) f_ν
        νL_ν = L_ν × ν_1600 = L_ν × c / 1600Å

    This decouples β and L_UV from the SED-fit E(B-V), which is
    critical for IRX-β analysis (Meurer+1999). When both quantities
    come from the same E(B-V), they are correlated by construction,
    collapsing the IRX-β relation to a flat line.

    Bands used (COSMOS2025): CFHT-u, HSC grizy, HST/F814W,
    JWST F115W/F150W, and Subaru intermediate bands (IA/IB).
    At z ≥ 1, typically 5–13 bands sample the UV window.

    References
    ----------
    Calzetti+1994 (ApJ 429, 582): Defined β from IUE UV spectral windows
    Meurer+1999 (ApJ 521, 64): IRX-β relation; β from UV spectroscopy
    Bouwens+2009 (ApJ 705, 936): Photometric β at z~2.5–6
    Bouwens+2012 (ApJ 754, 83): β measurement methodology from photometry
    Castellano+2012 (A&A 540, A39): Multi-band β fitting
    Rogers+2014 (MNRAS 440, 3714): Comparison of photometric β methods
    Reddy+2018 (ApJ 853, 56): IRX-β at z~1.5–2.5, photometric β

    Parameters
    ----------
    df : DataFrame
        Must contain flux columns (e.g. flux_model_hsc-g) and redshifts.
    z_col : str
        Redshift column name.
    flux_prefix : str
        Prefix for flux columns (e.g. "flux_auto" → "flux_auto_hsc-g").
    err_prefix : str
        Prefix for flux error columns.
    uv_window : (float, float)
        Rest-frame wavelength window for β fitting (Å).
    target_lambda : float
        Rest-frame wavelength for L_UV (Å). Default 1600.
    min_bands : int
        Minimum number of bands in UV window to attempt fit.
    min_snr : float
        Minimum SNR in a band to include it.
    z_min : float
        Minimum redshift (below this, insufficient UV coverage).

    Returns
    -------
    beta_uv : array
        UV spectral slope. NaN where fit not possible.
    log_l_uv : array
        log₁₀(νL_ν(1600Å) / L_☉). NaN where fit not possible.
    """
    from astropy.cosmology import Planck18
    import warnings

    z = df[z_col].values.astype(float)
    n = len(df)
    beta_out = np.full(n, np.nan)
    log_luv_out = np.full(n, np.nan)

    # Constants
    c_cgs = 2.998e10  # cm/s
    Mpc_to_cm = 3.0857e24
    uJy_to_cgs = 1e-29  # μJy → erg/s/cm²/Hz
    L_sun = 3.828e33  # erg/s
    nu_target = c_cgs / (target_lambda * 1e-8)  # Hz

    # Identify available flux columns
    available_bands = {}
    for band, lam_obs in _BAND_WAVELENGTHS.items():
        flux_col = f"{flux_prefix}_{band}"
        err_col = f"{err_prefix}_{band}"
        if flux_col in df.columns:
            available_bands[band] = {
                "lam_obs": lam_obs,
                "flux_col": flux_col,
                "err_col": err_col if err_col in df.columns else None,
            }

    if not available_bands:
        print(
            "  No flux columns found with prefix "
            f"'{flux_prefix}_*'. Available: "
            f"{[c for c in df.columns if 'flux' in c.lower()][:10]}"
        )
        return beta_out, log_luv_out

    band_names = list(available_bands.keys())
    band_lam_obs = np.array([available_bands[b]["lam_obs"] for b in band_names])
    print(f"  Photometric β: {len(available_bands)} bands available")

    # Load all fluxes into a matrix (n_sources × n_bands)
    flux_matrix = np.column_stack(
        [df[available_bands[b]["flux_col"]].values.astype(float) for b in band_names]
    )
    # Load errors if available
    has_errors = all(available_bands[b]["err_col"] is not None for b in band_names)
    if has_errors:
        err_matrix = np.column_stack(
            [df[available_bands[b]["err_col"]].values.astype(float) for b in band_names]
        )
    else:
        err_matrix = None

    # Process in redshift bins for efficiency
    z_valid = (z >= z_min) & (z < 10) & np.isfinite(z)
    dz_bin = 0.05
    z_edges = np.arange(z_min, 10.0, dz_bin)

    # Luminosity distances — precompute for z bin midpoints
    z_mids = np.arange(z_min + dz_bin / 2, 10.0, dz_bin)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dl_mids = {
            round(zm, 3): Planck18.luminosity_distance(zm).value for zm in z_mids
        }

    n_fit = 0
    n_skip_bands = 0
    n_skip_snr = 0
    n_skip_fit = 0

    for z_lo in z_edges:
        z_hi = z_lo + dz_bin
        z_mid = z_lo + dz_bin / 2
        in_bin = z_valid & (z >= z_lo) & (z < z_hi)
        if not np.any(in_bin):
            continue

        # d_L for this bin (< 1% error within dz=0.05)
        dl_Mpc = dl_mids.get(round(z_mid, 3))
        if dl_Mpc is None:
            dl_Mpc = Planck18.luminosity_distance(z_mid).value
        dl_cm = dl_Mpc * Mpc_to_cm

        # Which bands are in UV window at this redshift?
        lam_rest = band_lam_obs / (1 + z_mid)
        uv_mask = (lam_rest >= uv_window[0]) & (lam_rest <= uv_window[1])
        n_uv = uv_mask.sum()

        if n_uv < min_bands:
            n_skip_bands += in_bin.sum()
            continue

        uv_idx = np.where(uv_mask)[0]
        log_lam_rest = np.log10(lam_rest[uv_idx])

        # Get fluxes for sources in this z-bin
        idx_sources = np.where(in_bin)[0]
        fluxes = flux_matrix[np.ix_(idx_sources, uv_idx)]  # (M, N_uv)

        if err_matrix is not None:
            errors = err_matrix[np.ix_(idx_sources, uv_idx)]
        else:
            errors = None

        # Per-source fitting
        for j, src_idx in enumerate(idx_sources):
            f = fluxes[j]
            zi = z[src_idx]

            # SNR filter
            if errors is not None:
                e = errors[j]
                snr = np.where((e > 0) & np.isfinite(e), f / e, 0)
                good = (f > 0) & np.isfinite(f) & (snr >= min_snr)
            else:
                good = (f > 0) & np.isfinite(f)

            if good.sum() < min_bands:
                n_skip_snr += 1
                continue

            # Fit: log(f_ν) = (β+2) × log(λ_rest) + const
            log_f = np.log10(f[good])
            log_l = log_lam_rest[good]

            try:
                coeffs = np.polyfit(log_l, log_f, 1)
                slope = coeffs[0]  # β + 2
                intercept = coeffs[1]
                beta = slope - 2.0

                # Sanity check: β should be in [-3, 4]
                if beta < -3.5 or beta > 5.0:
                    n_skip_fit += 1
                    continue

                beta_out[src_idx] = beta

                # Interpolate f_ν at target wavelength
                log_f_target = slope * np.log10(target_lambda) + intercept
                f_target_uJy = 10**log_f_target  # μJy

                # Convert to νL_ν / L_sun
                # L_ν(ν_rest) = 4π d_L² f_ν(ν_obs) / (1+z)
                # The (1+z) accounts for bandwidth compression
                f_cgs = f_target_uJy * uJy_to_cgs
                L_nu = 4 * np.pi * dl_cm**2 * f_cgs / (1 + zi)
                nuLnu = L_nu * nu_target
                log_luv_out[src_idx] = np.log10(nuLnu / L_sun)

                n_fit += 1

            except (np.linalg.LinAlgError, ValueError):
                n_skip_fit += 1
                continue

    # Summary
    valid = np.isfinite(beta_out)
    print(f"  Photometric β results:")
    print(f"    Fitted: {n_fit:,} sources")
    print(f"    Skipped (too few UV bands): {n_skip_bands:,}")
    print(f"    Skipped (low SNR): {n_skip_snr:,}")
    print(f"    Skipped (bad fit): {n_skip_fit:,}")
    if np.any(valid):
        print(
            f"    β range: [{beta_out[valid].min():.2f}, "
            f"{beta_out[valid].max():.2f}], "
            f"median = {np.median(beta_out[valid]):.2f}"
        )
        print(
            f"    log(L_UV) range: [{log_luv_out[valid].min():.1f}, "
            f"{log_luv_out[valid].max():.1f}], "
            f"median = {np.median(log_luv_out[valid]):.1f}"
        )

    return beta_out, log_luv_out


# ── Sersic reliability flag ──────────────────────────────────────────────


def flag_sersic_reliable(
    df,
    z_col="redshift",
    radius_col="radius_sersic",
    chi2_col="fmf_chi2",
    snr_col="snr_f444w",
    z_max=4.0,
    min_radius_arcsec=0.05,
    max_chi2=5.0,
    min_snr=10.0,
):
    """
    Flag sources with reliable Sersic profile measurements.

    COSMOS2025 fits Sersic profiles in F444W (4.4 um) using SE++.
    Reliability depends on:
      - Redshift: at z < 3, F444W samples rest-frame > 1 um (stellar mass
        morphology, robust). At z ~ 4, rest-frame ~ 0.9 um (OK). At z > 5,
        rest-frame UV where galaxies are clumpy (unreliable).
      - Resolution: NIRCam F444W PSF FWHM ~ 0.15". Sources need
        r_e > PSF/2 ~ 0.075" to be resolved. We use 0.05" as minimum
        to be conservative.
      - Fit quality: SE++ chi^2 should be reasonable.
      - Detection SNR: need sufficient F444W SNR for profile fitting.

    Parameters
    ----------
    df : DataFrame
    z_col : str
        Redshift column.
    radius_col : str
        Sersic effective radius column (in degrees, as in COSMOS2025).
    chi2_col : str
        SE++ model chi^2 column.
    snr_col : str
        F444W SNR column.
    z_max : float
        Maximum redshift for reliable Sersic (default 4.0).
    min_radius_arcsec : float
        Minimum r_e in arcsec (0.05" ~ 0.4 kpc at z=2).
    max_chi2 : float
        Maximum acceptable reduced chi^2.
    min_snr : float
        Minimum F444W detection SNR.

    Returns
    -------
    reliable : boolean array
    """
    n = len(df)
    reliable = np.ones(n, dtype=bool)

    # Redshift cut
    if z_col in df.columns:
        reliable &= df[z_col].values < z_max

    # Radius cut (column is in degrees, convert to arcsec)
    if radius_col in df.columns:
        r_arcsec = df[radius_col].values * 3600.0
        reliable &= (r_arcsec > min_radius_arcsec) & np.isfinite(r_arcsec)
    else:
        print(f"  WARNING: Sersic radius column '{radius_col}' not found")

    # Chi2 cut
    if chi2_col in df.columns:
        chi2 = df[chi2_col].values
        reliable &= (chi2 < max_chi2) & (chi2 > 0) & np.isfinite(chi2)

    # SNR cut
    if snr_col in df.columns:
        snr = df[snr_col].values
        reliable &= (snr > min_snr) & np.isfinite(snr)

    return reliable


def compute_sigma_sfr(df, sfr_col, radius_col, z_col, sfr_is_log=True):
    """
    Compute SFR surface density: Sigma_SFR = SFR / (2 pi r_e^2).

    Uses the Sersic effective radius from SE++ (in degrees) converted
    to physical kpc using angular diameter distance.

    Parameters
    ----------
    df : DataFrame
    sfr_col : str
        SFR column (log or linear).
    radius_col : str
        Sersic effective radius in degrees.
    z_col : str
        Redshift column.
    sfr_is_log : bool
        Whether SFR is log10(SFR).

    Returns
    -------
    log_sigma_sfr : array
        log10(Sigma_SFR / [Msun/yr/kpc^2]). NaN where unavailable.
    """
    from astropy.cosmology import Planck18
    import warnings

    n = len(df)
    log_sigma = np.full(n, np.nan)

    if sfr_col not in df.columns or radius_col not in df.columns:
        print(
            f"  WARNING: Missing columns for Sigma_SFR "
            f"(need {sfr_col} and {radius_col})"
        )
        return log_sigma

    if sfr_is_log:
        sfr = 10 ** df[sfr_col].values.astype(float)
    else:
        sfr = df[sfr_col].values.astype(float)

    r_deg = df[radius_col].values.astype(float)
    z = df[z_col].values.astype(float)

    # Convert radius from degrees to physical kpc
    # Angular diameter distance, then r_kpc = r_rad * d_A
    # r_rad = r_deg * pi/180
    valid = (
        np.isfinite(sfr)
        & (sfr > 0)
        & np.isfinite(r_deg)
        & (r_deg > 0)
        & np.isfinite(z)
        & (z > 0.01)
    )

    # Precompute kpc_per_arcsec at z bin midpoints for speed
    z_bins = np.arange(0.01, 10.0, 0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpc_per_arcsec = {
            round(zb, 3): Planck18.kpc_proper_per_arcmin(zb).value / 60.0
            for zb in z_bins
        }

    for i in np.where(valid)[0]:
        zi = z[i]
        zbin = round(round(zi / 0.05) * 0.05, 3)
        kpc_as = kpc_per_arcsec.get(zbin)
        if kpc_as is None:
            kpc_as = Planck18.kpc_proper_per_arcmin(zi).value / 60.0

        r_arcsec = r_deg[i] * 3600.0
        r_kpc = r_arcsec * kpc_as

        if r_kpc > 0:
            area_kpc2 = 2 * np.pi * r_kpc**2  # within r_e
            sigma = sfr[i] / area_kpc2
            if sigma > 0:
                log_sigma[i] = np.log10(sigma)

    n_valid = np.isfinite(log_sigma).sum()
    if n_valid > 0:
        vals = log_sigma[np.isfinite(log_sigma)]
        print(
            f"  Sigma_SFR: {n_valid:,} sources, "
            f"median = {np.median(vals):.2f}, "
            f"range [{vals.min():.2f}, {vals.max():.2f}] "
            f"log(Msun/yr/kpc^2)"
        )

    return log_sigma


def create_wijesekera_catalog(
    photometry_path,
    lephare_path,
    cigale_path,
    output_path,
    # ---- Column name mappings (adjust to your catalog) ----
    z_col="zfinal",
    mass_col="mass_med",  # log(M*/Msun)
    sfr_col="sfr_med",  # log(SFR) or linear SFR
    sfr_is_log=True,  # True if sfr_col is log10(SFR)
    ra_col="ra",
    dec_col="dec",
    # Rest-frame absolute magnitudes from SED fitting
    nuv_col="mabs_nuv",
    r_col="mabs_r",
    j_col_rf="mabs_j",
    # UV properties
    luv_col="l_nuv",
    ebv_minchi2="ebv_minchi2",
    law_minchi2="law_minchi2",
    star_flag_col="flag_star",
    # Morphology
    radius_col="radius_sersic",  # Sersic r_e in degrees (SE++)
    sersic_chi2_col="fmf_chi2",  # SE++ model chi^2
    # CIGALE columns (extension 4)
    metallicity_col="metallicity",  # gas-phase metallicity from CIGALE
    cigale_mass_col="mass",  # stellar mass from CIGALE (linear)
    # Quality
    chi2_col="chi2_best",
    chi2_max=100.0,
    # Population class options
    exclude_classes=None,
    flag_quiescent=True,
    flag_starburst=True,
    flag_mass_complete=True,
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
    exclude_classes : list of int, optional
        Population classes to exclude from output.
        0=complete_sfg, 1=incomplete_sfg, 2=all_qt, 3=all_sb.
    flag_quiescent : bool
        Compute NUVrJ star-forming/quiescent flag. Default True.
    flag_starburst : bool
        Compute starburst flag (SFR/SFR_MS > 3). Default True.
    flag_mass_complete : bool
        Compute mass-completeness flag. Default True.

    Output columns include
    ----------------------
    Standard: ra, dec, redshift, log_stellar_mass, log_sfr
    UV: beta_uv (template), log_l_uv (template), beta_uv_phot, log_l_uv_phot
    Dust: ebv_minchi2, E(B-V) from LePhare
    Metallicity: metallicity (from CIGALE, mass fraction)
    Derived: log_ssfr, log_sigma_sfr, sersic_reliable
    Flags: star_forming, mass_complete, starburst, population_class
    """

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
    assert len(phot_df) == len(
        lephare_df
    ), f"Catalog length mismatch: {len(phot_df)} vs {len(lephare_df)}"
    df = pd.concat(
        [phot_df.reset_index(drop=True), lephare_df.reset_index(drop=True)],
        axis=1,
    )

    print(f"Loading Cigale: {cigale_path}")
    if cigale_path and cigale_path.exists():
        with fits.open(cigale_path) as hdul:
            cigale_df = Table(hdul[1].data).to_pandas()
        df = pd.concat([df, cigale_df], axis=1)

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
    # 4. NUVrJ star-forming / quiescent flag (Ilbert+2013, Eq 1)
    # ------------------------------------------------------------------
    # Don't remove quiescent — simstack needs all sources modeled to
    # avoid clustering bias. Flag them for use as a split dimension.
    if nuv_col and r_col and j_col_rf:
        if all(c in df.columns for c in [nuv_col, r_col, j_col_rf]):
            sf_mask = apply_nuvrj_cut(df, nuv_col, r_col, j_col_rf, z_col)
            n_quiescent = (~sf_mask & mask).sum()
            df["star_forming"] = sf_mask.astype(int)  # 1 = SF, 0 = quiescent
            # Do NOT apply to mask — keep all for stacking
            print(
                f"  NUVrJ flag:           {mask.sum():>8,} sources, "
                f"{n_quiescent:,} quiescent flagged (both kept)"
            )
        else:
            print(f"  NUVrJ: SKIPPED (columns not found)")
    else:
        print(f"  NUVrJ: SKIPPED (rest-frame mag columns not specified)")
        print(f"         Set nuv_col, r_col, j_col_rf to enable")

    # pdb.set_trace()
    # ------------------------------------------------------------------
    # 5. Mass-completeness flag (Table A.1)
    # ------------------------------------------------------------------
    # Instead of removing mass-incomplete sources (which biases simstack
    # by leaving correlated sources unmodeled), we flag them.
    # Simstack stacks ALL sources together; the flag can be used as a
    # split dimension to separate mass-complete from incomplete.
    mc_mask = apply_mass_completeness(df, z_col, mass_col)
    df["mass_complete"] = mc_mask.astype(int)  # 1 = complete, 0 = incomplete
    n_complete = mc_mask.sum()
    n_incomplete = len(df) - n_complete
    # Do NOT apply to mask — keep all sources for stacking
    print(
        f"  Mass-complete flag:   {n_complete:>8,} complete, "
        f"{n_incomplete:,} incomplete (both kept)"
    )

    # ------------------------------------------------------------------
    # 6. Starburst flag (SFR/SFR_MS > 3, Elbaz+2018)
    # ------------------------------------------------------------------
    if sfr_col and sfr_col in df.columns:
        if sfr_is_log:
            df["_sfr_linear"] = 10 ** df[sfr_col]
        else:
            df["_sfr_linear"] = df[sfr_col]

        sb_mask = remove_starbursts(df, mass_col, z_col, "_sfr_linear")
        df["starburst"] = (~sb_mask).astype(int)  # 1 = starburst, 0 = normal
        n_sb = (~sb_mask).sum()
        print(
            f"  Starburst flag:       {n_sb:>8,} starbursts flagged "
            f"(both kept for stacking)"
        )
        df.drop("_sfr_linear", axis=1, inplace=True)
    else:
        print(f"  Starburst flag: SKIPPED (SFR column not found)")

    # ------------------------------------------------------------------
    # 6b. Combined population class for simstack splitting
    # ------------------------------------------------------------------
    # Encode the binary flags into a single integer label:
    #   0 = complete_sfg   (SF, mass-complete, not starburst)
    #   1 = incomplete_sfg  (SF, mass-incomplete, not starburst)
    #   2 = all_qt          (quiescent, any mass completeness)
    #   3 = all_sb          (starburst, any SF/QT status)
    #
    # Priority: starburst > quiescent > mass-completeness
    # Each flag is only applied if the corresponding option is enabled.

    has_sf = flag_quiescent and "star_forming" in df.columns
    has_mc = flag_mass_complete and "mass_complete" in df.columns
    has_sb = flag_starburst and "starburst" in df.columns

    pop_class = np.zeros(len(df), dtype=int)  # default: all are class 0

    if has_mc:
        pop_class[df["mass_complete"] == 0] = 1  # incomplete_sfg
    if has_sf:
        pop_class[df["star_forming"] == 0] = 2  # all_qt
    if has_sb and has_mc and has_sf:
        sb_reliable = (
            (df["starburst"] == 1)
            & (df["mass_complete"] == 1)
            & (df["star_forming"] == 1)
        )
        pop_class[sb_reliable] = 3  # all_sb
    elif has_sb:
        pop_class[df["starburst"] == 1] = 3

    df["population_class"] = pop_class

    _pop_labels = {0: "complete_sfg", 1: "incomplete_sfg", 2: "all_qt", 3: "all_sb"}
    active_classes = sorted(set(pop_class[mask]))
    print(
        f"\n  Population classes "
        f"(mc={flag_mass_complete}, qt={flag_quiescent}, sb={flag_starburst}):"
    )
    for code in active_classes:
        label = _pop_labels.get(code, f"class_{code}")
        n = (pop_class[mask] == code).sum()
        print(f"    {code} ({label:>16}): {n:>8,}")

    # ------------------------------------------------------------------
    # 7. Build output catalog
    # ------------------------------------------------------------------
    df_sel = df[mask].copy()
    print(
        f"\n  TOTAL (quality cuts): {len(df_sel):,} galaxies"
        f" ({100*len(df_sel)/n_initial:.1f}% of input)"
    )

    # Optionally exclude population classes
    if exclude_classes:
        _pop_labels = {0: "complete_sfg", 1: "incomplete_sfg", 2: "all_qt", 3: "all_sb"}
        for cls in exclude_classes:
            n_excl = (df_sel["population_class"] == cls).sum()
            label = _pop_labels.get(cls, f"class_{cls}")
            df_sel = df_sel[df_sel["population_class"] != cls]
            print(f"  Excluded class {cls} ({label}): -{n_excl:,} sources")
            if cls not in (2,):  # warn if excluding anything other than QT
                print(
                    f"    ⚠ WARNING: excluding {label} may bias stacking "
                    f"if these sources have correlated IR emission"
                )
        print(f"  FINAL SAMPLE: {len(df_sel):,} galaxies")

    # Standardize column names for simstack4
    output_cols = {
        ra_col: "ra",
        dec_col: "dec",
        z_col: "redshift",
        mass_col: "log_stellar_mass",
    }

    if sfr_col and sfr_col in df_sel.columns:
        output_cols[sfr_col] = "log_sfr" if sfr_is_log else "sfr"

    # Keep all original columns plus standardized names
    for old_name, new_name in output_cols.items():
        if old_name in df_sel.columns and old_name != new_name:
            df_sel[new_name] = df_sel[old_name]

    # ------------------------------------------------------------------
    # 7b. Compute β_UV and L_UV(1600Å) from E(B-V) and L_NUV
    # ------------------------------------------------------------------
    # β_UV from E(B-V) and attenuation law:
    #   β = β_intrinsic + dβ/dE(B-V) × E(B-V)
    #   where dβ/dE(B-V) comes from the slope of k(λ) between 1600-2300Å
    #   Calzetti: 4.43, Arnouts: 4.20, Salim: 3.80

    has_ebv = ebv_minchi2 and ebv_minchi2 in df_sel.columns
    has_law = law_minchi2 and law_minchi2 in df_sel.columns
    has_luv = luv_col and luv_col in df_sel.columns

    if has_ebv:
        BETA_INTRINSIC = -2.3  # young SF population
        K_LAMBDA = {0: 4.43, 1: 4.20, 2: 3.80}  # Calzetti, Arnouts, Salim

        ebv = df_sel[ebv_minchi2].values.astype(float)
        if has_law:
            dust_law = df_sel[law_minchi2].values
        else:
            dust_law = np.zeros(len(df_sel))
            print("  No dust law column; assuming Calzetti for all")

        beta_uv = np.full_like(ebv, BETA_INTRINSIC)
        for law_idx, k_val in K_LAMBDA.items():
            mask = dust_law == law_idx
            if np.any(mask):
                beta_uv[mask] = BETA_INTRINSIC + k_val * ebv[mask]

        # Sanitize
        bad = ~np.isfinite(beta_uv) | ~np.isfinite(ebv) | (ebv < 0)
        beta_uv[bad] = BETA_INTRINSIC

        df_sel["beta_uv"] = beta_uv
        valid = np.isfinite(beta_uv)
        print(
            f"\n  β_UV computed from E(B-V): "
            f"range [{beta_uv[valid].min():.2f}, {beta_uv[valid].max():.2f}], "
            f"median {np.median(beta_uv[valid]):.2f}"
        )
    else:
        print("\n  WARNING: E(B-V) column not found — cannot compute β_UV")
        beta_uv = None

    # L_UV(1600Å) from L_NUV(2300Å) corrected using β_UV:
    #   l_nuv is νL_ν (= λL_λ); for f_λ ∝ λ^β, νL_ν ∝ λ^(β+1)
    #   log L(1600) = log L(2300) + (β+1) × log10(1600/2300)
    #   Correction vanishes at β=-1 (flat νL_ν).
    #   At β=-2.0: L_1600 = 1.44 × L_NUV (+0.16 dex)

    if has_luv and beta_uv is not None:
        LOG10_RATIO = np.log10(1600.0 / 2300.0)  # = -0.1576
        l_nuv_vals = df_sel[luv_col].values.astype(float)
        log_l_1600 = l_nuv_vals + (beta_uv + 1) * LOG10_RATIO

        # Fall back to uncorrected where β is invalid
        bad_corr = ~np.isfinite(log_l_1600)
        log_l_1600[bad_corr] = l_nuv_vals[bad_corr]

        df_sel["log_l_uv"] = log_l_1600

        correction = (beta_uv[~bad_corr] + 1) * LOG10_RATIO
        print(
            f"  L_UV(1600) from L_NUV + (β+1) correction: "
            f"median correction = {np.median(correction):+.3f} dex "
            f"(factor {10**np.median(correction):.2f}×)"
        )
        print(
            f"  WARNING: previous catalog used L_NUV as L_UV — "
            f"IRX was underestimated by ~{(10**np.median(correction) - 1)*100:.0f}%"
        )

    elif has_luv:
        # No β available — use L_NUV uncorrected (with warning)
        df_sel["log_l_uv"] = df_sel[luv_col]
        print(f"\n  WARNING: Using L_NUV as L_UV (no β correction)")
        print(f"           IRX will be systematically underestimated")

    else:
        print("\n  WARNING: No UV luminosity column — L_UV not computed")

    # ------------------------------------------------------------------
    # 7c. Photometric β_UV and L_UV(1600Å) from observed fluxes
    # ------------------------------------------------------------------
    # Independent of SED-fit E(B-V): fits power law to observed bands
    # sampling rest-frame 1300-2600Å.  Requires z ≥ 0.75 for coverage.
    # These are the quantities needed for an unbiased IRX-β analysis.
    print("\n  Computing photometric β_UV and L_UV(1600Å) from observed fluxes...")
    beta_phot, log_luv_phot = compute_photometric_beta_luv(
        df_sel,
        z_col="redshift",
        flux_prefix="flux_model",
        err_prefix="flux_err-cal_model",
    )
    df_sel["beta_uv_phot"] = beta_phot
    df_sel["log_l_uv_phot"] = log_luv_phot

    # Compare with template-derived values where both exist
    if "beta_uv" in df_sel.columns:
        both = np.isfinite(beta_phot) & np.isfinite(df_sel["beta_uv"].values)
        if np.any(both):
            diff = beta_phot[both] - df_sel["beta_uv"].values[both]
            r = np.corrcoef(beta_phot[both], df_sel["beta_uv"].values[both])[0, 1]
            print(
                f"  β comparison (phot vs template): "
                f"median offset = {np.median(diff):+.2f}, "
                f"scatter = {np.std(diff):.2f}, r = {r:.3f}"
            )
    if "log_l_uv" in df_sel.columns:
        both = np.isfinite(log_luv_phot) & np.isfinite(df_sel["log_l_uv"].values)
        if np.any(both):
            diff = log_luv_phot[both] - df_sel["log_l_uv"].values[both]
            print(
                f"  L_UV comparison (phot vs template): "
                f"median offset = {np.median(diff):+.3f} dex, "
                f"scatter = {np.std(diff):.3f} dex"
            )

    # ------------------------------------------------------------------
    # 7d. Derived quantities: sSFR, Sigma_SFR, Sersic flag
    # ------------------------------------------------------------------

    # sSFR = SFR / M*  (in log: log_ssfr = log_sfr - log_mass)
    if sfr_col and sfr_col in df_sel.columns:
        if sfr_is_log:
            df_sel["log_ssfr"] = (
                df_sel[sfr_col].values - df_sel["log_stellar_mass"].values
            )
        else:
            df_sel["log_ssfr"] = (
                np.log10(np.maximum(df_sel[sfr_col].values, 1e-20))
                - df_sel["log_stellar_mass"].values
            )
        valid_ssfr = np.isfinite(df_sel["log_ssfr"])
        print(
            f"\n  sSFR: {valid_ssfr.sum():,} valid, "
            f"median = {df_sel.loc[valid_ssfr, 'log_ssfr'].median():.2f} "
            f"log(yr^-1)"
        )

    # Delta_MS = SFR / SFR_MS (offset from Schreiber+2015 main sequence)
    if sfr_col and sfr_col in df_sel.columns:
        log_sfr_ms = np.array(
            [
                schreiber_main_sequence_sfr(m, z)
                for m, z in zip(
                    df_sel["log_stellar_mass"].values, df_sel["redshift"].values
                )
            ]
        )
        log_sfr_ms = np.log10(np.maximum(log_sfr_ms, 1e-20))
        if sfr_is_log:
            df_sel["log_delta_ms"] = df_sel[sfr_col].values - log_sfr_ms
        else:
            df_sel["log_delta_ms"] = (
                np.log10(np.maximum(df_sel[sfr_col].values, 1e-20)) - log_sfr_ms
            )
        valid_dms = np.isfinite(df_sel["log_delta_ms"])
        print(
            f"  Delta_MS: {valid_dms.sum():,} valid, "
            f"median = {df_sel.loc[valid_dms, 'log_delta_ms'].median():.2f} dex"
        )

    # Sersic reliability flag
    if radius_col in df_sel.columns:
        sersic_ok = flag_sersic_reliable(
            df_sel,
            z_col="redshift",
            radius_col=radius_col,
            chi2_col=sersic_chi2_col if sersic_chi2_col in df_sel.columns else None,
            snr_col="snr_f444w" if "snr_f444w" in df_sel.columns else None,
        )
        df_sel["sersic_reliable"] = sersic_ok.astype(int)
        print(
            f"  Sersic reliable: {sersic_ok.sum():,} / {len(df_sel):,} "
            f"({100*sersic_ok.mean():.0f}%)"
        )
    else:
        print(f"  Sersic: SKIPPED ('{radius_col}' not found)")

    # Sigma_SFR (SFR surface density)
    if radius_col in df_sel.columns and sfr_col in df_sel.columns:
        log_sigma = compute_sigma_sfr(
            df_sel,
            sfr_col=sfr_col if sfr_is_log else sfr_col,
            radius_col=radius_col,
            z_col="redshift",
            sfr_is_log=sfr_is_log,
        )
        df_sel["log_sigma_sfr"] = log_sigma
    else:
        print(f"  Sigma_SFR: SKIPPED (need both {sfr_col} and {radius_col})")

    # Ensure E(B-V) is in output
    if ebv_minchi2 and ebv_minchi2 in df_sel.columns:
        print(f"  E(B-V): present as '{ebv_minchi2}'")
    else:
        print(f"  E(B-V): not found")

    # Ensure metallicity is in output (from CIGALE extension)
    if metallicity_col and metallicity_col in df_sel.columns:
        valid_z_gas = np.isfinite(df_sel[metallicity_col]) & (
            df_sel[metallicity_col] > 0
        )
        print(
            f"  Metallicity: {valid_z_gas.sum():,} valid, "
            f"median = {df_sel.loc[valid_z_gas, metallicity_col].median():.4f} "
            f"(solar = 0.02)"
        )
    else:
        print(f"  Metallicity: '{metallicity_col}' not found")

    # ------------------------------------------------------------------
    # 8. Summary statistics
    # ------------------------------------------------------------------
    print(f"\n--- Sample Summary ---")
    for (z_lo, z_hi), mass_lim in MASS_COMPLETENESS.items():
        in_bin = (df_sel["redshift"] > z_lo) & (df_sel["redshift"] <= z_hi)
        n = in_bin.sum()
        if n > 0:
            med_mass = df_sel.loc[in_bin, "log_stellar_mass"].median()
            print(
                f"  {z_lo:.1f} < z <= {z_hi:.1f}: {n:>7,} sources, "
                f"mass_lim={mass_lim:.2f}, median_mass={med_mass:.2f}"
            )

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
    import argparse

    parser = argparse.ArgumentParser(
        description="Build stacking-ready catalog from COSMOS2025 (Wijesekera+2026 selection)"
    )
    parser.add_argument(
        "--exclude",
        type=int,
        nargs="*",
        default=None,
        help="Population classes to exclude: 0=complete_sfg, 1=incomplete_sfg, "
        "2=all_qt, 3=all_sb. Default: None (keep all).",
    )
    parser.add_argument(
        "--no-qt",
        action="store_true",
        help="Disable quiescent (NUVrJ) flagging — all sources treated as SF.",
    )
    parser.add_argument(
        "--no-sb", action="store_true", help="Disable starburst flagging."
    )
    parser.add_argument(
        "--no-mc",
        action="store_true",
        help="Disable mass-completeness flagging — all sources class 0.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output parquet path. Default: COSMOSWeb_stacking_catalog.parquet",
    )
    args = parser.parse_args()

    base = Path("/Users/mviero/data/Astronomy/catalogs/cosmos")

    if args.output:
        output_path = Path(args.output)
    else:
        output_path = base / "COSMOSWeb_sersic_stacking_catalog.parquet"

    df = create_wijesekera_catalog(
        photometry_path=base / "COSMOSWeb_mastercatalog_v1.fits",
        lephare_path=base / "COSMOSWeb_mastercatalog_v1_lephare.fits",
        cigale_path=base / "COSMOSWeb_mastercatalog_v1_cigale.fits",
        output_path=output_path,
        z_col="zfinal",
        mass_col="mass_med",
        sfr_col="sfr_med",
        sfr_is_log=True,
        ra_col="ra",
        dec_col="dec",
        nuv_col="mabs_nuv",
        r_col="mabs_r",
        j_col_rf="mabs_j",
        luv_col="l_nuv",
        ebv_minchi2="ebv_minchi2",
        law_minchi2="law_minchi2",
        star_flag_col="flag_star",
        radius_col="radius_sersic",
        sersic_chi2_col="fmf_chi2",
        metallicity_col="metallicity",
        chi2_col="chi2_best",
        chi2_max=100.0,
        exclude_classes=args.exclude,
        flag_quiescent=not args.no_qt,
        flag_starburst=not args.no_sb,
        flag_mass_complete=not args.no_mc,
    )
