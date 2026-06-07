"""
Unified COSMOS Catalog Preparation Script
==========================================

Builds a stacking-ready parquet catalog from COSMOSWeb master FITS files.
Supports paper-specific presets (w26, p26, a26) and full CLI customization.

Usage
-----
    prepare-cosmos-catalog                      # defaults
    prepare-cosmos-catalog --paper w26          # Wijesekera+2026
    prepare-cosmos-catalog --paper p26          # Parente+2026
    prepare-cosmos-catalog --config my.toml     # custom config
    prepare-cosmos-catalog --paper w26 --keep-all  # override exclude_classes
    prepare-cosmos-catalog --paper p26 --no-morphology

Population Classes
------------------
    0  complete_sfg   : SF, mass-complete, not starburst
    1  incomplete_sfg : SF, mass-incomplete, not starburst
    2  qt             : quiescent (any mass completeness)
    3  sb             : starburst (SF + SFR/SFR_MS > 3)

Note: simstack stacks ALL classes simultaneously as separate label dimensions.
Excluding a class (exclude_classes) removes those sources from the output
parquet — it does NOT change the stacking algorithm.  Quiescent galaxies
(class 2) are safe to exclude because they are IR-faint; excluding classes
0, 1, or 3 will bias the stacked fluxes.
"""

from __future__ import annotations

import operator
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

# =========================================================================
# Mass-completeness limits — Wijesekera+2026, Table A.1 (+extrapolations)
# log(M*/Msun) at 90% completeness per redshift bin
# =========================================================================

_MASS_COMPLETENESS_W26: dict[tuple[float, float], float] = {
    # (z_lo, z_hi): log_mass_limit
    (0.01, 0.5): 8.50,   # conservative (COSMOS depth)
    (0.5,  1.0): 9.25,
    (1.0,  1.5): 9.25,
    (1.5,  2.0): 9.50,
    (2.0,  2.5): 9.50,
    (2.5,  3.0): 9.75,
    (3.0,  3.5): 9.75,
    (3.5,  4.0): 10.00,
    (4.0,  4.5): 10.25,
    (4.5,  5.0): 10.25,
    # high-z: extrapolated (JWST-era estimates)
    (5.0,  6.0): 10.50,
    (6.0,  7.0): 10.75,
    (7.0,  8.0): 11.00,
    (8.0, 10.0): 11.25,
}

_MASS_COMPLETENESS_TABLES = {
    "wijesekera2026": _MASS_COMPLETENESS_W26,
}

# Effective wavelengths (Å) for COSMOS2025 broadband + intermediate filters
_BAND_WAVELENGTHS: dict[str, int] = {
    "cfht-u":   3709,
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
    "hsc-g":    4770,
    "hsc-r":    6175,
    "hsc-i":    7711,
    "hsc-z":    8898,
    "hsc-y":    9762,
    "hst-f814w": 8140,
    "f115w":   11534,
    "f150w":   15007,
}

_PAPER_IDS = {"w26": "wijesekera2026", "p26": "parente2026", "a26": "agrawal2026"}

# Python-side fallback — mirrors what defaults.toml defines.
# The TOML value takes precedence once load_config() runs.
_DEFAULT_CLASSES = [
    {"code": 3, "label": "sb",             "requires": ["starburst", "star_forming", "mass_complete"], "requires_not": [],              "warn_if_excluded": True},
    {"code": 2, "label": "qt",             "requires": [],                                              "requires_not": ["star_forming"], "warn_if_excluded": False},
    {"code": 1, "label": "incomplete_sfg", "requires": ["star_forming"],                                "requires_not": ["mass_complete"],"warn_if_excluded": True},
    {"code": 0, "label": "complete_sfg",   "requires": ["star_forming"],                                "requires_not": [],              "warn_if_excluded": True},
]


def _class_labels(config: dict) -> dict[int, str]:
    """Return {code: label} from TOML class definitions."""
    classes = config.get("populations", {}).get("classes", _DEFAULT_CLASSES)
    return {c["code"]: c["label"] for c in classes}


# =========================================================================
# Configuration
# =========================================================================


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, val in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(val, dict)
        ):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_toml(path: Path) -> dict:
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        import tomli as tomllib  # type: ignore[no-reattr]
    with open(path, "rb") as f:
        return tomllib.load(f)


def load_config(
    paper: str | None = None,
    config_path: str | Path | None = None,
    cli_overrides: dict | None = None,
) -> dict:
    """
    Load catalog preparation config.

    Merging order (later overrides earlier):
        defaults.toml → paper preset (if --paper) → explicit config (if --config)
        → cli_overrides

    Parameters
    ----------
    paper : str or None
        Short paper ID: "w26", "p26", "a26".
    config_path : Path or None
        Explicit TOML file to load.  Merged on top of defaults (+ paper preset).
    cli_overrides : dict or None
        Key-value overrides from the command line (nested dicts allowed).

    Returns
    -------
    config : dict
        Merged configuration.
    """
    catalog_config_dir = Path(__file__).parent.parent.parent.parent / "config" / "catalog"
    defaults_path = catalog_config_dir / "defaults.toml"

    if not defaults_path.exists():
        # Fallback: search relative to CWD
        defaults_path = Path("config/catalog/defaults.toml")
    if not defaults_path.exists():
        raise FileNotFoundError(
            f"defaults.toml not found. Expected at {defaults_path}. "
            "Run from the repository root or install the package."
        )

    config = _load_toml(defaults_path)

    if paper is not None:
        name = _PAPER_IDS.get(paper, paper)
        preset_path = catalog_config_dir / f"{name}.toml"
        if not preset_path.exists():
            raise FileNotFoundError(f"Paper preset not found: {preset_path}")
        config = _deep_merge(config, _load_toml(preset_path))

    if config_path is not None:
        config = _deep_merge(config, _load_toml(Path(config_path)))

    if cli_overrides:
        config = _deep_merge(config, cli_overrides)

    return config


# =========================================================================
# Step 1: Load FITS catalogs and merge
# =========================================================================


def _fits_to_df(path: Path) -> pd.DataFrame:
    """Load a FITS binary table, dropping multidimensional columns."""
    with fits.open(path) as hdul:
        table = Table(hdul[1].data)
        good_cols = [n for n in table.colnames if len(table[n].shape) <= 1]
        dropped = set(table.colnames) - set(good_cols)
        if dropped:
            print(f"    Dropped {len(dropped)} multidimensional columns: "
                  f"{sorted(dropped)[:5]}{'...' if len(dropped) > 5 else ''}")
        return table[good_cols].to_pandas()


def load_fits_catalogs(config: dict, catspath: Path | None = None) -> pd.DataFrame:
    """
    Load and merge phot + SED + optional CIGALE FITS catalogs.

    Parameters
    ----------
    config : dict
    catspath : Path or None
        Root directory for catalogs.  Falls back to $CATSPATH/cosmos.

    Returns
    -------
    df : DataFrame
        Merged catalog; row order preserved (COSMOS standard).
    """
    if catspath is None:
        cats_env = os.environ.get("CATSPATH", "")
        catspath = Path(cats_env) / "cosmos" if cats_env else None
    if catspath is None or not catspath.exists():
        raise FileNotFoundError(
            "Catalog directory not found. Set $CATSPATH or pass --catspath."
        )

    inp = config["inputs"]
    phot_path = catspath / inp["phot_file"]
    sed_path  = catspath / inp["sed_file"]
    cig_path  = catspath / inp["cigale_file"] if inp.get("cigale_file") else None

    print(f"  Loading photometry: {phot_path.name}")
    phot_df = _fits_to_df(phot_path)
    print(f"    {len(phot_df):,} sources, {len(phot_df.columns)} columns")

    print(f"  Loading SED fits:   {sed_path.name}")
    sed_df = _fits_to_df(sed_path)
    print(f"    {len(sed_df):,} sources, {len(sed_df.columns)} columns")

    if len(phot_df) != len(sed_df):
        raise ValueError(
            f"Catalog length mismatch: phot={len(phot_df)} sed={len(sed_df)}"
        )

    df = pd.concat(
        [phot_df.reset_index(drop=True), sed_df.reset_index(drop=True)], axis=1
    )
    df = df.loc[:, ~df.columns.duplicated()]

    if cig_path is not None:
        print(f"  Loading CIGALE:     {cig_path.name}")
        cig_df = _fits_to_df(cig_path)
        print(f"    {len(cig_df):,} sources, {len(cig_df.columns)} columns")
        df = pd.concat([df, cig_df.reset_index(drop=True)], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

    print(f"  Merged: {len(df):,} sources, {len(df.columns)} columns")
    return df


# =========================================================================
# Step 2: Quality cuts
# =========================================================================


def apply_quality_cuts(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Apply basic quality selections; return boolean mask (True = keep).

    Cuts applied:
      - Valid astrometry
      - Redshift in [z_min, z_max]
      - Stellar mass > mass_min
      - Not a star (star_flag == 0)
      - chi2 < chi2_max
    """
    cols = config["columns"]
    sel  = config["selection"]
    n    = len(df)

    ra_col    = cols["ra"]
    dec_col   = cols["dec"]
    z_col     = cols["redshift"]
    mass_col  = cols["stellar_mass"]

    mask = pd.Series(True, index=df.index)

    coord_ok = np.isfinite(df[ra_col]) & np.isfinite(df[dec_col])
    mask &= coord_ok
    print(f"  Valid coords:   {mask.sum():>8,} / {n:,}")

    z_ok = (
        (df[z_col] > sel["redshift_min"])
        & (df[z_col] <= sel["redshift_max"])
        & np.isfinite(df[z_col])
    )
    mask &= z_ok
    print(f"  Redshift range: {mask.sum():>8,} / {n:,}  "
          f"({sel['redshift_min']:.2f} < z <= {sel['redshift_max']:.2f})")

    mass_ok = np.isfinite(df[mass_col]) & (df[mass_col] > sel["mass_min"])
    mask &= mass_ok
    print(f"  Valid mass:     {mask.sum():>8,} / {n:,}  "
          f"(log M* > {sel['mass_min']:.1f})")

    star_col = cols.get("star_flag")
    if star_col and star_col in df.columns:
        mask &= df[star_col] == 0
        print(f"  Star removal:   {mask.sum():>8,} / {n:,}")

    chi2_col = cols.get("chi2")
    if chi2_col and chi2_col in df.columns:
        mask &= df[chi2_col] < sel["chi2_max"]
        print(f"  chi2 quality:   {mask.sum():>8,} / {n:,}  "
              f"(chi2 < {sel['chi2_max']:.0f})")

    snr_min = sel.get("snr_min", 0.0)
    if snr_min and snr_min > 0:
        snr_col = sel.get("snr_col", "snr_f444w")
        if snr_col in df.columns:
            mask &= df[snr_col].values >= snr_min
            print(f"  SNR cut:        {mask.sum():>8,} / {n:,}  "
                  f"({snr_col} >= {snr_min:.1f})")
        else:
            print(f"  SNR cut: SKIPPED ('{snr_col}' not found)")

    mag_max = sel.get("mag_max", 0.0)
    if mag_max and mag_max > 0:
        mag_col = sel.get("mag_col", "mag_model_uvista-ks")
        if mag_col in df.columns:
            mask &= df[mag_col].values <= mag_max
            print(f"  Mag cut:        {mask.sum():>8,} / {n:,}  "
                  f"({mag_col} <= {mag_max:.2f})")
        else:
            print(f"  Mag cut: SKIPPED ('{mag_col}' not found)")

    return mask


# =========================================================================
# Step 3: NUVrJ star-forming / quiescent classification
# =========================================================================


def classify_nuvrj(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    NUVrJ quiescent classification (Ilbert+2013, Eq 1).

    Quiescent if BOTH:
      (NUV - r) > 3*(r - J) + 1
      (NUV - r) > 3.1
    Only applied at z < 4 (rest-frame J unreliable beyond).
    Sources with missing colors are kept as star-forming.

    Returns
    -------
    star_forming : Series (bool)
        True = star-forming, False = quiescent.
    """
    cols    = config["columns"]
    nuv_col = cols.get("nuv")
    r_col   = cols.get("r_band")
    j_col   = cols.get("j_band")
    z_col   = cols["redshift"]

    n = len(df)
    star_forming = pd.Series(True, index=df.index)  # default: keep all

    if not all([nuv_col, r_col, j_col]):
        print("  NUVrJ: SKIPPED (nuv/r_band/j_band not specified in config)")
        return star_forming

    if not all(c in df.columns for c in [nuv_col, r_col, j_col]):
        missing = [c for c in [nuv_col, r_col, j_col] if c not in df.columns]
        print(f"  NUVrJ: SKIPPED (columns not found: {missing})")
        return star_forming

    nuv_r = df[nuv_col] - df[r_col]
    r_j   = df[r_col]   - df[j_col]
    z     = df[z_col]

    quiescent = (nuv_r > 3.0 * r_j + 1.0) & (nuv_r > 3.1) & (z < 4.0)
    has_colors = np.isfinite(nuv_r) & np.isfinite(r_j)

    star_forming = ~quiescent | ~has_colors
    n_qt = (~star_forming).sum()
    print(f"  NUVrJ: {n_qt:,} / {n:,} classified quiescent "
          f"({100*n_qt/max(n,1):.1f}%); both kept for stacking")
    return star_forming


# =========================================================================
# Step 4: Mass completeness
# =========================================================================


def _schreiber_ms_sfr(log_mass: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Schreiber+2015 Eq 9 main-sequence SFR (linear, Msun/yr)."""
    m0, a0, a1, m1, a2 = 0.5, 1.5, 0.3, 0.36, 2.5
    with np.errstate(invalid="ignore"):  # NaN z → NaN SFR_MS (intentional)
        r = np.log10(1.0 + np.asarray(z, dtype=float))
    m = np.asarray(log_mass, dtype=float) - 9.0
    term = np.maximum(0.0, m - m1 - a2 * r)
    return 10.0 ** (m - m0 + a0 * r - a1 * term**2)


def apply_mass_completeness(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Flag mass-complete sources (True = above 90% completeness limit).

    Table reference: Wijesekera+2026 Table A.1.  Sources below the limit
    are flagged as incomplete (class 1), NOT removed.
    """
    cols     = config["columns"]
    pop_cfg  = config["populations"]
    z_col    = cols["redshift"]
    mass_col = cols["stellar_mass"]

    table_key = pop_cfg.get("mass_completeness_table", "wijesekera2026")
    table = _MASS_COMPLETENESS_TABLES.get(table_key)
    if table is None:
        raise ValueError(
            f"Unknown mass_completeness_table '{table_key}'. "
            f"Available: {list(_MASS_COMPLETENESS_TABLES)}"
        )

    complete = pd.Series(False, index=df.index)
    for (z_lo, z_hi), mass_lim in table.items():
        in_bin = (df[z_col] > z_lo) & (df[z_col] <= z_hi)
        complete |= in_bin & (df[mass_col] >= mass_lim)

    n_c = complete.sum()
    n_i = len(df) - n_c
    print(f"  Mass completeness: {n_c:,} complete, {n_i:,} incomplete "
          f"(table: {table_key})")
    return complete


# =========================================================================
# Step 5: Starburst flag
# =========================================================================


def flag_starbursts(
    df: pd.DataFrame, config: dict, threshold: float = 3.0
) -> pd.Series:
    """
    Identify starbursts: SFR / SFR_MS > threshold (Elbaz+2018 definition).

    SFR_MS from Schreiber+2015 Eq 9.
    Sources with missing SFR data are flagged as non-starburst.

    Returns
    -------
    is_starburst : Series (bool)
    """
    cols     = config["columns"]
    sfr_col  = cols.get("sfr")
    z_col    = cols["redshift"]
    mass_col = cols["stellar_mass"]
    sfr_log  = cols.get("sfr_is_log", True)

    n = len(df)
    is_sb = pd.Series(False, index=df.index)

    if not sfr_col or sfr_col not in df.columns:
        print("  Starburst flag: SKIPPED (SFR column not found)")
        return is_sb

    sfr = 10.0 ** df[sfr_col].values.astype(float) if sfr_log else df[sfr_col].values.astype(float)
    sfr_ms = _schreiber_ms_sfr(df[mass_col].values, df[z_col].values)

    valid_sfr = np.isfinite(sfr) & (sfr > 0)
    is_sb_arr = np.zeros(n, dtype=bool)
    is_sb_arr[valid_sfr] = (sfr[valid_sfr] / sfr_ms[valid_sfr]) > threshold

    is_sb = pd.Series(is_sb_arr, index=df.index)
    n_sb = is_sb.sum()
    print(f"  Starburst flag: {n_sb:,} / {n:,} starbursts "
          f"(SFR/SFR_MS > {threshold}); both kept for stacking")
    return is_sb


# =========================================================================
# Step 6a: β_UV and L_UV(1600Å) from template (E(B-V) + dust law)
# =========================================================================

# β = β_intrinsic + k_λ * E(B-V)
# k_λ = d(β)/d(E(B-V)) at 1600–2300 Å for each attenuation law
_BETA_INTRINSIC = -2.3
_K_LAMBDA = {0: 4.43, 1: 4.20, 2: 3.80}   # Calzetti, Arnouts, Salim


def compute_beta_luv_template(
    df: pd.DataFrame, config: dict
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute β_UV and log L_UV(1600Å) from LePhare E(B-V) and L_NUV.

    β = β_intrinsic + k_λ(dust_law) × E(B-V)
    log L(1600) = log L(2300) + (β + 1) × log10(1600/2300)

    Returns
    -------
    beta_uv, log_l_uv : (n,) arrays, NaN where unavailable
    """
    cols     = config["columns"]
    ebv_col  = cols.get("ebv")
    law_col  = cols.get("dust_law")
    luv_col  = cols.get("l_nuv")
    n        = len(df)

    beta_out  = np.full(n, np.nan)
    log_luv_out = np.full(n, np.nan)

    if not ebv_col or ebv_col not in df.columns:
        print("  Template β: SKIPPED (ebv column not found)")
        return beta_out, log_luv_out

    ebv = df[ebv_col].values.astype(float)
    dust_law = (
        df[law_col].values if law_col and law_col in df.columns
        else np.zeros(n, dtype=int)
    )

    beta = np.full(n, _BETA_INTRINSIC, dtype=float)
    for law_idx, k_val in _K_LAMBDA.items():
        where = dust_law == law_idx
        beta[where] = _BETA_INTRINSIC + k_val * ebv[where]
    bad = ~np.isfinite(beta) | ~np.isfinite(ebv) | (ebv < 0)
    beta[bad] = _BETA_INTRINSIC
    beta_out = beta

    if luv_col and luv_col in df.columns:
        LOG_RATIO = np.log10(1600.0 / 2300.0)
        l_nuv = df[luv_col].values.astype(float)
        log_l = l_nuv + (beta + 1.0) * LOG_RATIO
        bad_l = ~np.isfinite(log_l)
        log_l[bad_l] = l_nuv[bad_l]
        log_luv_out = log_l

        corr = (beta[~bad_l] + 1.0) * LOG_RATIO
        print(
            f"  Template β: median={np.nanmedian(beta):.2f}, "
            f"L_UV correction={np.nanmedian(corr):+.3f} dex"
        )
    else:
        valid = np.isfinite(beta)
        print(f"  Template β: median={np.nanmedian(beta[valid]):.2f} "
              f"(L_UV skipped — l_nuv column not found)")

    return beta_out, log_luv_out


# =========================================================================
# Step 6b: β_UV and L_UV(1600Å) from broadband photometry
# =========================================================================


def compute_beta_luv_photometric(
    df: pd.DataFrame,
    config: dict,
    uv_window: tuple[float, float] = (1300.0, 2600.0),
    target_lambda: float = 1600.0,
    min_bands: int = 2,
    min_snr: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute β_UV and log L_UV(1600Å) by fitting a power law to observed fluxes.

    f_ν ∝ λ^(β+2) → fit log(f_ν) = (β+2)*log(λ_rest) + const
    in rest-frame window [1300, 2600] Å.

    This is independent of SED-fit E(B-V), which is critical for IRX-β
    analysis (Meurer+1999): template β and L_UV are correlated with E(B-V)
    by construction, collapsing the IRX-β relation to a flat line.

    Returns
    -------
    beta_uv, log_l_uv : (n,) arrays, NaN where fit not possible
    """
    from astropy.cosmology import Planck18

    beta_cfg  = config["beta_luv"]
    z_col     = config["columns"]["redshift"]
    z_min     = beta_cfg.get("min_redshift_phot", 0.75)
    flux_pfx  = beta_cfg.get("flux_prefix", "flux_model")
    err_pfx   = beta_cfg.get("err_prefix", "flux_err-cal_model")

    n = len(df)
    beta_out    = np.full(n, np.nan)
    log_luv_out = np.full(n, np.nan)
    z = df[z_col].values.astype(float)

    # Physical constants
    c_cgs       = 2.998e10       # cm/s
    Mpc_to_cm   = 3.0857e24
    uJy_to_cgs  = 1e-29          # μJy → erg/s/cm²/Hz
    L_sun       = 3.828e33       # erg/s
    nu_target   = c_cgs / (target_lambda * 1e-8)

    # Discover available flux columns
    available: dict[str, dict] = {}
    for band, lam_obs in _BAND_WAVELENGTHS.items():
        flux_col = f"{flux_pfx}_{band}"
        err_col  = f"{err_pfx}_{band}"
        if flux_col in df.columns:
            available[band] = {
                "lam_obs":  lam_obs,
                "flux_col": flux_col,
                "err_col":  err_col if err_col in df.columns else None,
            }

    if not available:
        print(f"  Photometric β: SKIPPED (no columns matching '{flux_pfx}_*')")
        return beta_out, log_luv_out

    band_names  = list(available)
    band_lam    = np.array([available[b]["lam_obs"] for b in band_names])
    print(f"  Photometric β: {len(available)} bands found")

    flux_mat = np.column_stack(
        [df[available[b]["flux_col"]].values.astype(float) for b in band_names]
    )
    has_err = all(available[b]["err_col"] is not None for b in band_names)
    err_mat = (
        np.column_stack(
            [df[available[b]["err_col"]].values.astype(float) for b in band_names]
        )
        if has_err else None
    )

    # Precompute d_L at z-bin midpoints (dz=0.05 gives < 1% error)
    dz        = 0.05
    z_lo_arr  = np.arange(z_min, 10.0, dz)
    z_mids    = z_lo_arr + dz / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dl_map = {round(zm, 3): Planck18.luminosity_distance(zm).value for zm in z_mids}

    z_valid = np.isfinite(z) & (z >= z_min) & (z < 10.0)
    n_fit = n_skip_bands = n_skip_snr = n_skip_fit = 0

    for z_lo in z_lo_arr:
        z_hi  = z_lo + dz
        z_mid = z_lo + dz / 2
        in_bin = z_valid & (z >= z_lo) & (z < z_hi)
        if not in_bin.any():
            continue

        dl_Mpc = dl_map.get(round(z_mid, 3)) or Planck18.luminosity_distance(z_mid).value
        dl_cm  = dl_Mpc * Mpc_to_cm

        lam_rest = band_lam / (1.0 + z_mid)
        uv_mask  = (lam_rest >= uv_window[0]) & (lam_rest <= uv_window[1])
        if uv_mask.sum() < min_bands:
            n_skip_bands += int(in_bin.sum())
            continue

        uv_idx      = np.where(uv_mask)[0]
        log_lam_uv  = np.log10(lam_rest[uv_idx])
        src_idx_arr = np.where(in_bin)[0]
        fluxes      = flux_mat[np.ix_(src_idx_arr, uv_idx)]
        errors      = err_mat[np.ix_(src_idx_arr, uv_idx)] if err_mat is not None else None

        for j, src_idx in enumerate(src_idx_arr):
            f = fluxes[j]
            if errors is not None:
                e   = errors[j]
                snr = np.where((e > 0) & np.isfinite(e), f / e, 0.0)
                good = (f > 0) & np.isfinite(f) & (snr >= min_snr)
            else:
                good = (f > 0) & np.isfinite(f)

            if good.sum() < min_bands:
                n_skip_snr += 1
                continue

            try:
                coeffs = np.polyfit(log_lam_uv[good], np.log10(f[good]), 1)
                slope, intercept = coeffs
                beta = slope - 2.0

                if beta < -3.5 or beta > 5.0:
                    n_skip_fit += 1
                    continue

                beta_out[src_idx] = beta

                log_f_tgt  = slope * np.log10(target_lambda) + intercept
                f_uJy      = 10.0 ** log_f_tgt
                f_cgs      = f_uJy * uJy_to_cgs
                L_nu       = 4.0 * np.pi * dl_cm**2 * f_cgs / (1.0 + z[src_idx])
                log_luv_out[src_idx] = np.log10(L_nu * nu_target / L_sun)
                n_fit += 1

            except (np.linalg.LinAlgError, ValueError):
                n_skip_fit += 1

    valid = np.isfinite(beta_out)
    print(
        f"    fitted={n_fit:,}  skip_bands={n_skip_bands:,}  "
        f"skip_snr={n_skip_snr:,}  skip_fit={n_skip_fit:,}"
    )
    if valid.any():
        print(
            f"    β median={np.median(beta_out[valid]):.2f}  "
            f"log L_UV median={np.median(log_luv_out[valid]):.2f}"
        )
    return beta_out, log_luv_out


# =========================================================================
# Step 7: Sersic reliability and Sigma_SFR (optional)
# =========================================================================


def compute_sersic_columns(
    df: pd.DataFrame,
    config: dict,
    z_max: float = 4.0,
    min_radius_arcsec: float = 0.05,
    max_chi2: float = 5.0,
    min_snr: float = 10.0,
) -> pd.DataFrame:
    """
    Add sersic_reliable and (if morphology.sigma_sfr) log_sigma_sfr columns.

    Sersic reliability criteria:
      - z < z_max (F444W samples stellar mass morphology)
      - r_e > min_radius_arcsec (resolved)
      - SE++ chi2 < max_chi2
      - F444W SNR > min_snr
    """
    from astropy.cosmology import Planck18

    cols        = config["columns"]
    morph_cfg   = config["morphology"]
    radius_col  = cols.get("sersic_radius", "radius_sersic")
    chi2_col    = cols.get("sersic_chi2", "fmf_chi2")
    snr_col     = "snr_f444w"
    z_col       = cols["redshift"]
    sfr_col     = cols.get("sfr")
    sfr_log     = cols.get("sfr_is_log", True)

    n = len(df)
    reliable = np.ones(n, dtype=bool)

    if radius_col not in df.columns:
        print(f"  Sersic: SKIPPED ('{radius_col}' not found)")
        return df

    reliable &= df[z_col].values < z_max

    r_arcsec = df[radius_col].values * 3600.0
    reliable &= (r_arcsec > min_radius_arcsec) & np.isfinite(r_arcsec)

    if chi2_col in df.columns:
        c2 = df[chi2_col].values
        reliable &= (c2 < max_chi2) & (c2 > 0) & np.isfinite(c2)

    if snr_col in df.columns:
        snr = df[snr_col].values
        reliable &= (snr > min_snr) & np.isfinite(snr)

    df = df.copy()
    df["sersic_reliable"] = reliable.astype(int)
    pct = 100.0 * reliable.mean()
    print(f"  Sersic reliable: {reliable.sum():,} / {n:,} ({pct:.0f}%)")

    if not morph_cfg.get("sigma_sfr", False):
        return df
    if not sfr_col or sfr_col not in df.columns:
        print("  Sigma_SFR: SKIPPED (sfr column not found)")
        return df

    sfr = (
        10.0 ** df[sfr_col].values.astype(float)
        if sfr_log
        else df[sfr_col].values.astype(float)
    )
    r_deg = df[radius_col].values.astype(float)
    z     = df[z_col].values.astype(float)

    valid = (
        np.isfinite(sfr) & (sfr > 0)
        & np.isfinite(r_deg) & (r_deg > 0)
        & np.isfinite(z) & (z > 0.01)
    )

    # Precompute kpc/arcsec at bin midpoints
    z_bins = np.arange(0.01, 10.0, 0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpc_map = {
            round(zb, 3): Planck18.kpc_proper_per_arcmin(zb).value / 60.0
            for zb in z_bins
        }

    log_sigma = np.full(n, np.nan)
    for i in np.where(valid)[0]:
        zi      = z[i]
        zbin    = round(round(zi / 0.05) * 0.05, 3)
        kpc_as  = kpc_map.get(zbin) or Planck18.kpc_proper_per_arcmin(zi).value / 60.0
        r_kpc   = r_deg[i] * 3600.0 * kpc_as
        if r_kpc > 0:
            sigma = sfr[i] / (2.0 * np.pi * r_kpc**2)
            if sigma > 0:
                log_sigma[i] = np.log10(sigma)

    df["log_sigma_sfr"] = log_sigma
    n_valid = np.isfinite(log_sigma).sum()
    if n_valid > 0:
        vals = log_sigma[np.isfinite(log_sigma)]
        print(
            f"  Sigma_SFR: {n_valid:,} valid, "
            f"median={np.median(vals):.2f} log(Msun/yr/kpc^2)"
        )
    return df


# =========================================================================
# Step 8a: Custom flag columns
# =========================================================================

_CMP_OPS: dict[str, object] = {
    "==": operator.eq, "!=": operator.ne,
    ">":  operator.gt, ">=": operator.ge,
    "<":  operator.lt, "<=": operator.le,
}


def apply_custom_flags(df: pd.DataFrame, config: dict) -> None:
    """
    Add boolean columns to df for each entry in [populations.custom_flags].

    Each entry: name = {column = "col", op = "==", value = scalar}
    The resulting column is written as int (0/1) so it behaves like the
    built-in star_forming / mass_complete / starburst columns.
    """
    custom = config.get("populations", {}).get("custom_flags", {})
    if not custom:
        return
    for name, spec in custom.items():
        col    = spec.get("column")
        op_str = spec.get("op", "==")
        val    = spec.get("value", 1)
        op_fn  = _CMP_OPS.get(op_str)
        if op_fn is None:
            raise ValueError(f"Unknown operator '{op_str}' for custom flag '{name}'")
        if col not in df.columns:
            print(f"  Custom flag '{name}': SKIPPED (column '{col}' not in catalog)")
            continue
        df[name] = op_fn(df[col].values, val).astype(int)
        n = int(df[name].sum())
        print(f"  Custom flag '{name}': {n:,} sources  ({col} {op_str} {val})")


# =========================================================================
# Step 8b: Population class encoding
# =========================================================================


def encode_populations(df: pd.DataFrame, config: dict) -> pd.Series:
    """
    Assign each source to a population class using the TOML class definitions.

    Class rules are evaluated in list order (first match wins = highest priority).
    Each rule specifies:
      requires     : flag columns that must be True  (missing → rule fails)
      requires_not : flag columns that must be False (missing → ignored)

    Sources not matching any rule are assigned code -1 and a warning is printed.

    The default rules reproduce the previous hardcoded scheme:
      3 sb             (starburst + SF + mass-complete)
      2 qt             (quiescent)
      1 incomplete_sfg (SF + mass-incomplete)
      0 complete_sfg   (SF, catch-all)
    """
    classes   = config.get("populations", {}).get("classes", _DEFAULT_CLASSES)
    labels    = {c["code"]: c["label"] for c in classes}
    n         = len(df)
    pop_class = np.full(n, -1, dtype=int)

    for cls in classes:
        code         = cls["code"]
        requires     = cls.get("requires", [])
        requires_not = cls.get("requires_not", [])

        mask = pop_class == -1  # only consider still-unclassified sources
        if not mask.any():
            break

        ok = True
        for col in requires:
            if col not in df.columns:
                warnings.warn(
                    f"Flag column '{col}' required by class {code} ('{cls['label']}') "
                    f"not found — class will be empty. "
                    f"Check flag_quiescent/starburst/mass_completeness settings."
                )
                ok = False
                break
            mask = mask & df[col].values.astype(bool)
        if not ok:
            continue

        for col in requires_not:
            if col in df.columns:
                mask = mask & ~df[col].values.astype(bool)

        pop_class[mask] = code

    result = pd.Series(pop_class, index=df.index)

    for code in sorted(set(pop_class)):
        label  = labels.get(code, f"class_{code}") if code >= 0 else "unclassified"
        n_code = (pop_class == code).sum()
        print(f"    class {code} ({label:>16}): {n_code:>8,}")

    n_unclassified = (pop_class == -1).sum()
    if n_unclassified:
        warnings.warn(
            f"{n_unclassified:,} sources not matched by any class rule "
            f"(assigned code -1). Check [[populations.classes]] ordering."
        )

    return result


# =========================================================================
# Step 9: Save
# =========================================================================


def save_catalog(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved: {out_path}")
    print(f"  {len(df):,} sources, {len(df.columns)} columns, {size_mb:.1f} MB")


# =========================================================================
# Orchestrator
# =========================================================================


def build_catalog(
    config: dict,
    catspath: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Run the full catalog preparation pipeline."""
    print("=" * 70)
    paper = config.get("paper", {})
    print(f"  Catalog:   {paper.get('description', 'COSMOSWeb')}")
    if paper.get("reference"):
        print(f"  Reference: {paper['reference']}")
    print("=" * 70)

    # ── 1. Load ────────────────────────────────────────────────────────────
    print("\n[1] Loading FITS catalogs")
    df = load_fits_catalogs(config, catspath)
    n_initial = len(df)

    # ── 2. Quality cuts ────────────────────────────────────────────────────
    print("\n[2] Quality cuts")
    quality_mask = apply_quality_cuts(df, config)

    # ── 3. Population flags (computed on full catalog before subsetting) ───
    print("\n[3] Population flags")

    pop_cfg = config["populations"]

    if pop_cfg.get("flag_quiescent", True):
        star_forming = classify_nuvrj(df, config)
    else:
        star_forming = pd.Series(True, index=df.index)
        print("  NUVrJ: SKIPPED (flag_quiescent=false)")

    if pop_cfg.get("flag_mass_completeness", True):
        mass_complete = apply_mass_completeness(df, config)
    else:
        mass_complete = pd.Series(True, index=df.index)
        print("  Mass completeness: SKIPPED (flag_mass_completeness=false)")

    if pop_cfg.get("flag_starburst", True):
        is_starburst = flag_starbursts(df, config)
    else:
        is_starburst = pd.Series(False, index=df.index)
        print("  Starburst flag: SKIPPED (flag_starburst=false)")

    # Attach built-in flags to full catalog
    df["star_forming"]  = star_forming.astype(int)
    df["mass_complete"] = mass_complete.astype(int)
    df["starburst"]     = is_starburst.astype(int)

    # Apply any custom flag columns defined in [populations.custom_flags]
    apply_custom_flags(df, config)

    print("  Population classes:")
    pop_class = encode_populations(df, config)
    df["population_class"] = pop_class

    # ── 4. Subset to quality-passing sources ──────────────────────────────
    df_sel = df[quality_mask].copy()
    print(
        f"\n  After quality cuts: {len(df_sel):,} / {n_initial:,} "
        f"({100*len(df_sel)/max(n_initial,1):.1f}%)"
    )

    # ── 5. Standardize column names ────────────────────────────────────────
    cols     = config["columns"]
    z_col    = cols["redshift"]
    mass_col = cols["stellar_mass"]
    sfr_col  = cols.get("sfr")
    sfr_log  = cols.get("sfr_is_log", True)
    ra_col   = cols["ra"]
    dec_col  = cols["dec"]

    rename = {
        ra_col:   "ra",
        dec_col:  "dec",
        z_col:    "redshift",
        mass_col: "log_stellar_mass",
    }
    if sfr_col and sfr_col in df_sel.columns:
        rename[sfr_col] = "log_sfr" if sfr_log else "sfr"
    for old, new in rename.items():
        if old in df_sel.columns and old != new:
            df_sel[new] = df_sel[old]

    # ── 6. UV β and L_UV ──────────────────────────────────────────────────
    method = config["beta_luv"].get("method", "both")
    print(f"\n[4] UV β and L_UV (method={method})")

    if method in ("template", "both"):
        beta_tmpl, log_luv_tmpl = compute_beta_luv_template(df_sel, config)
        df_sel["beta_uv"]    = beta_tmpl
        df_sel["log_l_uv"]   = log_luv_tmpl

    if method in ("photometric", "both"):
        print("  Photometric fit (this may take a few minutes)...")
        beta_phot, log_luv_phot = compute_beta_luv_photometric(df_sel, config)
        df_sel["beta_uv_phot"]  = beta_phot
        df_sel["log_l_uv_phot"] = log_luv_phot

        if method == "both" and "beta_uv" in df_sel.columns:
            both = np.isfinite(beta_phot) & np.isfinite(df_sel["beta_uv"].values)
            if both.any():
                diff = beta_phot[both] - df_sel["beta_uv"].values[both]
                r    = np.corrcoef(beta_phot[both], df_sel["beta_uv"].values[both])[0, 1]
                print(
                    f"  β comparison (phot vs template): "
                    f"offset={np.median(diff):+.2f}  σ={np.std(diff):.2f}  r={r:.3f}"
                )

    # ── 7. sSFR and Δ_MS ──────────────────────────────────────────────────
    if "log_sfr" in df_sel.columns and "log_stellar_mass" in df_sel.columns:
        df_sel["log_ssfr"] = df_sel["log_sfr"].values - df_sel["log_stellar_mass"].values
        log_sfr_ms = np.log10(np.maximum(
            _schreiber_ms_sfr(df_sel["log_stellar_mass"].values, df_sel["redshift"].values),
            1e-20
        ))
        df_sel["log_delta_ms"] = df_sel["log_sfr"].values - log_sfr_ms

    # ── 8. Morphology ─────────────────────────────────────────────────────
    morph = config["morphology"]
    if morph.get("sersic", False):
        print("\n[5] Sersic morphology + Sigma_SFR")
        df_sel = compute_sersic_columns(df_sel, config)
    else:
        print("\n[5] Sersic: SKIPPED (morphology.sersic=false)")

    # ── 9. Exclude classes ────────────────────────────────────────────────
    exclude = pop_cfg.get("exclude_classes", [])
    if exclude:
        labels = _class_labels(config)
        # Build {code: warn_if_excluded} from class defs
        warn_lookup = {
            c["code"]: c.get("warn_if_excluded", True)
            for c in config.get("populations", {}).get("classes", _DEFAULT_CLASSES)
        }
        print(f"\n[6] Excluding population classes: {exclude}")
        for cls in exclude:
            n_excl = (df_sel["population_class"] == cls).sum()
            label  = labels.get(cls, f"class_{cls}")
            df_sel = df_sel[df_sel["population_class"] != cls]
            print(f"  Removed class {cls} ({label}): -{n_excl:,} sources")
            if warn_lookup.get(cls, True):
                print(f"    WARNING: excluding '{label}' may bias stacked fluxes")
        print(f"  Final sample: {len(df_sel):,} sources")

    # ── 10. Summary ───────────────────────────────────────────────────────
    print("\n--- Redshift bin summary ---")
    table = _MASS_COMPLETENESS_TABLES.get(
        pop_cfg.get("mass_completeness_table", "wijesekera2026"), {}
    )
    for (z_lo, z_hi), mass_lim in table.items():
        in_bin = (df_sel["redshift"] > z_lo) & (df_sel["redshift"] <= z_hi)
        n_bin  = in_bin.sum()
        if n_bin > 0:
            med_m = df_sel.loc[in_bin, "log_stellar_mass"].median()
            print(
                f"  {z_lo:.1f} < z <= {z_hi:.1f}: {n_bin:>6,} sources  "
                f"mass_lim={mass_lim:.2f}  med_mass={med_m:.2f}"
            )

    # ── 11. Save ──────────────────────────────────────────────────────────
    if output_path is None:
        paper_id = config.get("paper", {}).get("id", "cosmos25")
        catspath_out = catspath or Path(
            os.environ.get("CATSPATH", "."), "cosmos"
        )
        output_path = catspath_out / f"COSMOSWeb_{paper_id}_catalog.parquet"

    save_catalog(df_sel, output_path)
    return df_sel


# =========================================================================
# CLI entry point
# =========================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="prepare-cosmos-catalog",
        description="Build a stacking-ready COSMOSWeb parquet catalog.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Paper presets:
  --paper w26   Wijesekera+2026 (z=0.5-5, photometric β, exclude QT)
  --paper p26   Parente+2026    (z=0.5-5, Sersic+sigma_SFR, all classes)
  --paper a26   Agrawal+2026    (placeholder)

Population classes:
  0  complete_sfg   : SF, mass-complete, not SB
  1  incomplete_sfg : SF, mass-incomplete, not SB
  2  qt             : quiescent
  3  sb             : starburst (SFR/SFR_MS > 3)
""",
    )
    parser.add_argument(
        "--paper", choices=["w26", "p26", "a26"],
        help="Paper preset (overrides defaults)"
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="TOML config file (merged on top of defaults + paper preset)"
    )
    parser.add_argument(
        "--catspath", metavar="DIR",
        help="Catalog directory (overrides $CATSPATH/cosmos)"
    )
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output parquet path (default: auto-named)"
    )
    parser.add_argument(
        "--exclude", type=int, nargs="*", metavar="CLASS",
        help="Population classes to exclude (e.g. --exclude 2 3)"
    )
    parser.add_argument(
        "--keep-all", action="store_true",
        help="Keep all population classes (overrides --exclude and config)"
    )
    parser.add_argument(
        "--no-morphology", action="store_true",
        help="Disable Sersic + Sigma_SFR even if config enables it"
    )
    parser.add_argument(
        "--no-qt", action="store_true",
        help="Disable NUVrJ quiescent flagging"
    )
    parser.add_argument(
        "--no-sb", action="store_true",
        help="Disable starburst flagging"
    )
    parser.add_argument(
        "--no-mc", action="store_true",
        help="Disable mass-completeness flagging"
    )
    parser.add_argument(
        "--mass-min", type=float, metavar="FLOAT",
        help="Minimum log10(M*/Msun) threshold (e.g. 9.0); overrides config"
    )
    parser.add_argument(
        "--snr-min", type=float, metavar="FLOAT",
        help="Minimum detection SNR (e.g. 5.0); overrides config"
    )
    parser.add_argument(
        "--snr-col", metavar="COL",
        help="Column name for detection SNR (default: snr_f444w)"
    )
    parser.add_argument(
        "--mag-max", type=float, metavar="FLOAT",
        help="K-band magnitude limit, brighter-end (e.g. 24.7); overrides config"
    )
    parser.add_argument(
        "--mag-col", metavar="COL",
        help="Column name for magnitude cut (default: mag_model_uvista-ks)"
    )
    args = parser.parse_args()

    # Build CLI overrides dict
    overrides: dict = {}
    if args.keep_all:
        overrides.setdefault("populations", {})["exclude_classes"] = []
    elif args.exclude is not None:
        overrides.setdefault("populations", {})["exclude_classes"] = args.exclude
    if args.no_morphology:
        overrides.setdefault("morphology", {})["sersic"] = False
        overrides.setdefault("morphology", {})["sigma_sfr"] = False
    if args.no_qt:
        overrides.setdefault("populations", {})["flag_quiescent"] = False
    if args.no_sb:
        overrides.setdefault("populations", {})["flag_starburst"] = False
    if args.no_mc:
        overrides.setdefault("populations", {})["flag_mass_completeness"] = False
    if args.mass_min is not None:
        overrides.setdefault("selection", {})["mass_min"] = args.mass_min
    if args.snr_min is not None:
        overrides.setdefault("selection", {})["snr_min"] = args.snr_min
    if args.snr_col is not None:
        overrides.setdefault("selection", {})["snr_col"] = args.snr_col
    if args.mag_max is not None:
        overrides.setdefault("selection", {})["mag_max"] = args.mag_max
    if args.mag_col is not None:
        overrides.setdefault("selection", {})["mag_col"] = args.mag_col

    config = load_config(
        paper=args.paper,
        config_path=args.config,
        cli_overrides=overrides or None,
    )

    catspath  = Path(args.catspath) if args.catspath else None
    out_path  = Path(args.output) if args.output else None

    build_catalog(config, catspath=catspath, output_path=out_path)


if __name__ == "__main__":
    main()
