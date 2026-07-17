"""
COSMOS2020 Catalog Preparation with K-fold Source Partitioning
==============================================================

Builds K stacking-ready parquet catalogs from a COSMOS2020 FITS or CSV file.
In each output parquet a distinct 1/K subset of SFGs carries population_class=0
(the signal partition) while the remaining (K-1)/K SFGs carry population_class=1
(deblending nuisance). Quiescent galaxies (class 2) are copied unchanged into
every split.

The result: K pseudo-spectra whose sfg_a fluxes at every z-bin come from
disjoint galaxy populations → valid χ² and fully empirical σ_α.
See docs/pah-forward-model-4-brief.md §"K-fold Source Partitioning".

Usage
-----
    prepare-cosmos2020-catalog --catalog cosmos2020_FARMER.csv --splits 3
    prepare-cosmos2020-catalog --catalog cosmos2020_FARMER.fits --splits 2
    prepare-cosmos2020-catalog --catalog cosmos2020_FARMER.csv --splits 1  # single, no splitting
    prepare-cosmos2020-catalog --catalog ... --split-type labels           # use existing population_class col
    prepare-cosmos2020-catalog --catalog ... --splits 3 --stem cosmos2020_PAH_3pop  # + starburst columns

Population classes in output parquets
--------------------------------------
    0  sfg_signal  : SFGs in this split's signal partition (1/K of SFGs)
    1  sfg_nuisance: SFGs in the other (K-1) partitions  (deblending layer)
    2  qt          : quiescent — identical in every split

Starburst labeling (optional, on by default)
---------------------------------------------
Mirrors `prepare_cosmos2025_catalog.py` / `prepare_cosmos_catalog.flag_starbursts`:
starbursts are NOT a separate `population_class` code (that would K-fold-split an
already-rare subsample three ways on top of the mass/z grid). Instead a `starburst`
(0/1) and `log_delta_ms` (log10(SFR/SFR_MS)) column are added to the output parquet
for use as a downstream `[catalog.classification.binning.starburst]` dimension —
`population_class` × `starburst` × mass × z then gives the "3 populations" (SFG-MS,
SFG-SB, QT) via simstack4's generic itertools.product binning.

SFR/SFR_MS > `--starburst-threshold` (default 3.0, Elbaz+2018) against a
Schreiber+2015 Eq. 9 main-sequence SFR, RECENTERED to this catalog's own median
SF-population offset from the literature relation before applying the threshold
(same self-calibration principle as `pah_spectrum.py`'s `s_pivot`) — COSMOS2020's
LePhare SFRs run ~0.2 dex above the literature Schreiber+2015 ridge, and without
recentering the literal cut flags ~30% of SFGs instead of a rare tail.

Matching TOML stanza for the output parquets
--------------------------------------------
    [catalog.astrometry]
    ra  = "ALPHA_J2000"
    dec = "DELTA_J2000"

    [catalog.classification]
    split_type = "labels"
    bin_property_columns = ["lp_mass_med", "lp_SFR_best", "lp_sSFR_med", "starburst", "log_delta_ms"]

    [catalog.classification.split_params]
    id = "population_class"
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_cosmos_catalog import _schreiber_ms_sfr


# ---------------------------------------------------------------------------
# Load catalog
# ---------------------------------------------------------------------------

def _load_catalog(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in (".fits", ".fit"):
        from astropy.io import fits
        from astropy.table import Table
        print(f"  Loading FITS: {path.name}")
        with fits.open(path) as hdul:
            tbl = Table(hdul[1].data)
            good = [n for n in tbl.colnames if len(tbl[n].shape) <= 1]
            dropped = len(tbl.colnames) - len(good)
            if dropped:
                print(f"    Dropped {dropped} multi-dim columns")
            df = tbl[good].to_pandas()
    elif suffix == ".csv":
        print(f"  Loading CSV: {path.name}")
        df = pd.read_csv(path, low_memory=False)
    elif suffix == ".parquet":
        print(f"  Loading Parquet: {path.name}")
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}")
    print(f"    {len(df):,} rows, {len(df.columns)} columns")
    return df


# ---------------------------------------------------------------------------
# Quality cuts
# ---------------------------------------------------------------------------

def _apply_quality_cuts(
    df: pd.DataFrame,
    z_col: str,
    mass_col: str,
    ra_col: str,
    dec_col: str,
    z_min: float,
    z_max: float,
    mass_min: float,
    star_col: str | None,
    star_galaxy_value: int,
) -> pd.Series:
    n = len(df)
    mask = pd.Series(True, index=df.index)

    coord_ok = np.isfinite(df[ra_col]) & np.isfinite(df[dec_col])
    mask &= coord_ok
    print(f"  Valid coords:   {mask.sum():>8,} / {n:,}")

    z_ok = (
        np.isfinite(df[z_col])
        & (df[z_col] > z_min)
        & (df[z_col] <= z_max)
    )
    mask &= z_ok
    print(f"  Redshift range: {mask.sum():>8,} / {n:,}  ({z_min} < z <= {z_max})")

    mass_ok = np.isfinite(df[mass_col]) & (df[mass_col] > mass_min)
    mask &= mass_ok
    print(f"  Valid mass:     {mask.sum():>8,} / {n:,}  (log M* > {mass_min})")

    if star_col and star_col in df.columns:
        gal_mask = df[star_col] == star_galaxy_value
        mask &= gal_mask
        print(f"  Galaxy select:  {mask.sum():>8,} / {n:,}  ({star_col} == {star_galaxy_value})")
    else:
        if star_col:
            print(f"  Star cut: SKIPPED ('{star_col}' not found)")

    return mask


# ---------------------------------------------------------------------------
# NUVrJ classification
# ---------------------------------------------------------------------------

def _classify_nuvrj(
    df: pd.DataFrame,
    nuv_r_col: str,
    r_j_col: str,
    z_col: str,
) -> np.ndarray:
    """
    Returns population_class array: 0 = star-forming, 2 = quiescent.
    Ilbert+2013 criterion: quiescent if (NUV-r) > 3*(r-J)+1 AND (NUV-r) > 3.1,
    applied only at z < 4 where rest-frame J is reliable.
    Sources with missing colors are classified as star-forming.
    """
    nuv_r = df[nuv_r_col].values.astype(float)
    r_j   = df[r_j_col].values.astype(float)
    z     = df[z_col].values.astype(float)

    has_colors = np.isfinite(nuv_r) & np.isfinite(r_j)
    quiescent  = (
        has_colors
        & (nuv_r > 3.0 * r_j + 1.0)
        & (nuv_r > 3.1)
        & (z < 4.0)
    )

    pop = np.where(quiescent, 2, 0).astype(int)
    n_qt = quiescent.sum()
    n_sf = (~quiescent).sum()
    print(f"  NUVrJ: {n_sf:,} star-forming (class 0), {n_qt:,} quiescent (class 2)")
    return pop


# ---------------------------------------------------------------------------
# Starburst flagging (downstream binning column, not a population_class code)
# ---------------------------------------------------------------------------

def _flag_starbursts(
    df: pd.DataFrame,
    sf_mask: np.ndarray,
    sfr_col: str,
    mass_col: str,
    z_col: str,
    sfr_is_log: bool = True,
    threshold: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Identify starbursts: SFR / SFR_MS > threshold (Elbaz+2018 definition),
    against a Schreiber+2015 Eq. 9 main-sequence SFR (`_schreiber_ms_sfr`,
    imported from prepare_cosmos_catalog.py — same function, same threshold
    convention as prepare_cosmos2025_catalog.py's COSMOSWeb pipeline).

    The comparison is done in log10(SFR/SFR_MS) space, RECENTERED by the
    median offset among star-forming (sf_mask) sources before applying the
    threshold — literature main-sequence relations are calibrated against a
    specific SFR indicator/IMF/SED code and rarely land exactly on another
    catalog's zero point. Mirrors the `s_pivot` self-calibration already used
    in `pah_spectrum.py`.

    Only sf_mask sources are eligible to be flagged (quiescent galaxies are
    never starbursts by construction). Sources with missing/invalid SFR are
    flagged as non-starburst.

    Returns
    -------
    is_starburst : int array (0/1)
    log_delta_ms : float array, log10(SFR/SFR_MS) after recentering (NaN where
        SFR is missing/invalid)
    """
    n = len(df)
    log_delta_ms = np.full(n, np.nan)
    is_sb = np.zeros(n, dtype=int)

    if sfr_col not in df.columns:
        print(f"  Starburst flag: SKIPPED ('{sfr_col}' not found)")
        return is_sb, log_delta_ms

    log_sfr = (
        df[sfr_col].values.astype(float)
        if sfr_is_log
        else np.log10(np.clip(df[sfr_col].values.astype(float), 1e-20, None))
    )
    sfr_ms = _schreiber_ms_sfr(df[mass_col].values.astype(float), df[z_col].values.astype(float))
    valid = np.isfinite(log_sfr) & np.isfinite(sfr_ms) & (sfr_ms > 0)

    raw_delta = np.full(n, np.nan)
    raw_delta[valid] = log_sfr[valid] - np.log10(sfr_ms[valid])

    recenter_mask = valid & sf_mask
    offset = float(np.nanmedian(raw_delta[recenter_mask])) if recenter_mask.any() else 0.0

    log_delta_ms[valid] = raw_delta[valid] - offset
    eligible = valid & sf_mask
    is_sb[eligible] = (log_delta_ms[eligible] >= np.log10(threshold)).astype(int)

    n_sb = int(is_sb.sum())
    n_sf = int(sf_mask.sum())
    print(
        f"  Starburst flag: {n_sb:,} / {n_sf:,} SFGs "
        f"(SFR/SFR_MS > {threshold}, recentered by {offset:+.3f} dex to this "
        f"catalog's own SF ridge; {n_sb / max(n_sf, 1):.1%} of SFGs)"
    )
    return is_sb, log_delta_ms


# ---------------------------------------------------------------------------
# K-fold split assignment
# ---------------------------------------------------------------------------

def _assign_kfold_splits(
    pop_class: np.ndarray,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    """
    Assign split_id (0..n_splits-1) to every SFG (pop_class == 0).
    Uses a shuffled round-robin so splits are balanced and randomised.
    QTs (pop_class == 2) receive split_id = -1 (not used for splitting).
    """
    sfg_idx  = np.where(pop_class == 0)[0]
    split_id = np.full(len(pop_class), -1, dtype=int)

    if len(sfg_idx) == 0 or n_splits <= 1:
        split_id[sfg_idx] = 0
        return split_id

    rng   = np.random.default_rng(seed)
    order = rng.permutation(len(sfg_idx))           # shuffle SFG indices
    assigned = np.arange(len(sfg_idx))[order] % n_splits  # balanced round-robin
    # un-shuffle so split_id[i] corresponds to the original SFG order
    split_id_sfg = np.empty(len(sfg_idx), dtype=int)
    split_id_sfg[order] = assigned
    split_id[sfg_idx] = split_id_sfg

    for k in range(n_splits):
        n_k = (split_id_sfg == k).sum()
        print(f"    split {k}: {n_k:,} SFGs")
    return split_id


def _make_split_population_class(
    pop_class: np.ndarray,
    split_id: np.ndarray,
    target_split: int,
) -> np.ndarray:
    """
    For split i:
      SFGs with split_id == i  → class 0 (signal)
      SFGs with split_id != i  → class 1 (nuisance deblender)
      QTs  (pop_class == 2)    → class 2 (unchanged)
    """
    out = pop_class.copy()
    sfg_mask    = pop_class == 0
    signal_mask = sfg_mask & (split_id == target_split)
    nuisance_mask = sfg_mask & (split_id != target_split)
    out[signal_mask]   = 0
    out[nuisance_mask] = 1
    return out


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    mb = path.stat().st_size / 1e6
    print(f"  Saved: {path.name}  ({len(df):,} sources, {mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def build_cosmos2020_catalogs(
    catalog_path: Path,
    n_splits: int = 3,
    split_type: str = "nuvrj",
    z_col: str = "lp_zBEST",
    mass_col: str = "lp_mass_med",
    ra_col: str = "ALPHA_J2000",
    dec_col: str = "DELTA_J2000",
    nuv_r_col: str = "restNUV-R",
    r_j_col: str = "restR-J",
    z_min: float = 0.1,
    z_max: float = 6.0,
    mass_min: float = 8.0,
    star_col: str | None = "lp_type",
    star_galaxy_value: int = 0,
    seed: int = 42,
    output_dir: Path | None = None,
    stem: str | None = None,
    flag_starburst: bool = True,
    sfr_col: str = "lp_SFR_best",
    sfr_is_log: bool = True,
    starburst_threshold: float = 3.0,
) -> list[Path]:
    """
    Build K COSMOS2020 parquet catalogs for K-fold PAH stacking analysis.

    Parameters
    ----------
    catalog_path : Path
        Input COSMOS2020 FITS, CSV, or parquet file.
    n_splits : int
        Number of independent source partitions (K). K=3 is the default.
        Use K=1 to produce a single unsplit catalog (all SFGs as class 0).
    split_type : str
        "nuvrj"  — run NUVrJ classification to assign population_class (default).
        "labels" — use existing population_class column in the input catalog.
    z_col, mass_col, ra_col, dec_col : str
        Column names for redshift, stellar mass, RA, Dec.
    nuv_r_col, r_j_col : str
        Column names for NUV-R and R-J rest-frame colors (used when split_type="nuvrj").
    z_min, z_max, mass_min : float
        Quality cut thresholds.
    star_col : str or None
        Column for star/galaxy separation. Set to None to skip. Default "lp_type"
        (the real COSMOS2020 FARMER column; NOT "type", which never matches).
    star_galaxy_value : int
        Value of star_col that indicates a galaxy (default 0).
    seed : int
        Random seed for reproducible K-fold assignment.
    output_dir : Path or None
        Directory for output parquets. Defaults to catalog_path.parent.
    stem : str or None
        Base name for output files. Defaults to catalog_path.stem.
    flag_starburst : bool
        Add `starburst` (0/1) and `log_delta_ms` columns (see module docstring).
        Set False to skip entirely.
    sfr_col : str
        SFR column for starburst flagging (default "lp_SFR_best", log10(SFR)).
    sfr_is_log : bool
        Whether sfr_col is log10(SFR) (default True).
    starburst_threshold : float
        SFR/SFR_MS threshold, Elbaz+2018 definition (default 3.0).

    Returns
    -------
    out_paths : list[Path]
        Paths to the K written parquet files (or 1 file when n_splits=1).
    """
    print("=" * 70)
    print(f"  COSMOS2020 Catalog Preparation  (K={n_splits}, split_type={split_type})")
    print("=" * 70)

    if output_dir is None:
        output_dir = catalog_path.parent
    if stem is None:
        stem = catalog_path.stem

    # ── 1. Load ────────────────────────────────────────────────────────────
    print("\n[1] Loading catalog")
    df_raw = _load_catalog(catalog_path)

    # Assign split_id to every raw row NOW so it is stable w.r.t. quality cuts
    # (placeholder; will be re-computed after classification on filtered catalog)
    n_raw = len(df_raw)

    # ── 2. Quality cuts ────────────────────────────────────────────────────
    print("\n[2] Quality cuts")
    for col, label in [(z_col, "Redshift"), (mass_col, "Stellar mass"),
                        (ra_col, "RA"), (dec_col, "Dec")]:
        if col not in df_raw.columns:
            raise ValueError(f"Required column '{col}' ({label}) not found in catalog. "
                             f"Available: {list(df_raw.columns[:10])}...")

    quality_mask = _apply_quality_cuts(
        df_raw, z_col, mass_col, ra_col, dec_col,
        z_min, z_max, mass_min, star_col, star_galaxy_value,
    )
    df = df_raw[quality_mask].copy()
    print(f"\n  After cuts: {len(df):,} / {n_raw:,} sources retained")

    # ── 3. Population classification ──────────────────────────────────────
    print(f"\n[3] Classification (split_type={split_type})")

    if split_type == "nuvrj":
        for col, label in [(nuv_r_col, "NUV-R"), (r_j_col, "R-J")]:
            if col not in df.columns:
                raise ValueError(
                    f"NUVrJ requires column '{col}' ({label}). "
                    "Pass --split-type labels if population_class already exists."
                )
        pop_class = _classify_nuvrj(df, nuv_r_col, r_j_col, z_col)
        df["population_class"] = pop_class

    elif split_type == "labels":
        if "population_class" not in df.columns:
            raise ValueError(
                "split_type='labels' requires a 'population_class' column in the input catalog. "
                "Run with --split-type nuvrj to compute it."
            )
        pop_class = df["population_class"].values.astype(int)
        n_sf = (pop_class == 0).sum()
        n_qt = (pop_class == 2).sum()
        print(f"  Using existing population_class: {n_sf:,} SFG (0), {n_qt:,} QT (2)")

    else:
        raise ValueError(f"Unknown split_type '{split_type}'. Choose 'nuvrj' or 'labels'.")

    n_sfg = (pop_class == 0).sum()
    n_qt  = (pop_class == 2).sum()
    print(f"  Total: {n_sfg:,} SFGs, {n_qt:,} QTs, {len(df):,} sources")

    # ── 3a. Starburst flagging (downstream binning column) ─────────────────
    if flag_starburst:
        print(f"\n[3a] Starburst flagging")
        is_sb, log_delta_ms = _flag_starbursts(
            df, pop_class == 0, sfr_col, mass_col, z_col,
            sfr_is_log=sfr_is_log, threshold=starburst_threshold,
        )
        df["starburst"] = is_sb
        df["log_delta_ms"] = log_delta_ms

    # ── 4. K-fold split assignment ─────────────────────────────────────────
    if n_splits > 1:
        print(f"\n[4] K-fold split assignment (K={n_splits}, seed={seed})")
        split_id = _assign_kfold_splits(pop_class, n_splits, seed)
    else:
        # n_splits == 1: single output, all SFGs are class 0
        split_id = np.zeros(len(df), dtype=int)
        split_id[pop_class == 2] = -1  # QTs

    # ── 5. Write output parquets ──────────────────────────────────────────
    print(f"\n[5] Writing {n_splits} output parquet(s)")
    out_paths: list[Path] = []

    for k in range(n_splits):
        df_out = df.copy()
        df_out["population_class"] = _make_split_population_class(
            pop_class, split_id, target_split=k
        )
        df_out["catalog_split_id"] = k  # convenience: which split this is

        n_sig = (df_out["population_class"] == 0).sum()
        n_nui = (df_out["population_class"] == 1).sum()
        n_qt_ = (df_out["population_class"] == 2).sum()

        if n_splits == 1:
            fname = f"{stem}_catalog.parquet"
        else:
            fname = f"{stem}_split{k}of{n_splits}.parquet"
        out_path = output_dir / fname

        print(f"\n  Split {k}: class 0 (signal)={n_sig:,}  "
              f"class 1 (nuisance)={n_nui:,}  class 2 (QT)={n_qt_:,}")
        _save(df_out, out_path)
        out_paths.append(out_path)

    print("\n" + "=" * 70)
    print(f"  Done.  {n_splits} catalog(s) written to {output_dir}")
    print("=" * 70)
    print("\nTOML stanza for output catalogs:")
    print("  [catalog.astrometry]")
    print(f'  ra  = "{ra_col}"')
    print(f'  dec = "{dec_col}"')
    print("  [catalog.classification]")
    print('  split_type = "labels"')
    if flag_starburst:
        print(
            '  bin_property_columns = '
            f'["{mass_col}", "{sfr_col}", "starburst", "log_delta_ms"]'
        )
    print("  [catalog.classification.split_params]")
    print('  id = "population_class"')
    print(f"  [catalog.classification.binning.redshift]")
    print(f'  id = "{z_col}"')
    print(f"  [catalog.classification.binning.stellar_mass]")
    print(f'  id = "{mass_col}"')
    if flag_starburst:
        print(f"  [catalog.classification.binning.starburst]")
        print(f'  id = "starburst"')
        print(f"  bins = [-0.5, 0.5, 1.5]")

    return out_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prepare-cosmos2020-catalog",
        description="Build K-fold COSMOS2020 parquet catalogs for PAH stacking analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 3-fold split using NUVrJ classification
  prepare-cosmos2020-catalog --catalog $CATSPATH/cosmos/cosmos2020_FARMER.fits --splits 3

  # 2-fold split, output to specific directory
  prepare-cosmos2020-catalog --catalog cosmos2020_FARMER.csv --splits 2 --output-dir $CATSPATH/cosmos

  # Single catalog (no splitting), NUVrJ classification → labels-compatible output
  prepare-cosmos2020-catalog --catalog cosmos2020_FARMER.fits --splits 1

  # Use existing population_class column (already classified externally)
  prepare-cosmos2020-catalog --catalog cosmos2020_classified.parquet --split-type labels --splits 3
""",
    )
    parser.add_argument(
        "--catalog", required=True, metavar="PATH",
        help="Input COSMOS2020 catalog (FITS, CSV, or parquet)",
    )
    parser.add_argument(
        "--splits", type=int, default=3, metavar="K",
        help="Number of independent source partitions (default: 3)",
    )
    parser.add_argument(
        "--split-type", choices=["nuvrj", "labels"], default="nuvrj",
        help="Classification method: 'nuvrj' (compute from colors, default) or "
             "'labels' (use existing population_class column)",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR",
        help="Output directory (default: same directory as --catalog)",
    )
    parser.add_argument(
        "--stem", metavar="NAME",
        help="Base name for output files (default: input filename stem)",
    )
    parser.add_argument(
        "--z-col", default="lp_zBEST", metavar="COL",
        help="Redshift column name (default: lp_zBEST)",
    )
    parser.add_argument(
        "--mass-col", default="lp_mass_med", metavar="COL",
        help="Stellar mass column name (default: lp_mass_med)",
    )
    parser.add_argument(
        "--ra-col", default="ALPHA_J2000", metavar="COL",
        help="RA column name (default: ALPHA_J2000)",
    )
    parser.add_argument(
        "--dec-col", default="DELTA_J2000", metavar="COL",
        help="Dec column name (default: DELTA_J2000)",
    )
    parser.add_argument(
        "--nuv-r-col", default="restNUV-R", metavar="COL",
        help="NUV-R color column for NUVrJ classification (default: restNUV-R)",
    )
    parser.add_argument(
        "--r-j-col", default="restR-J", metavar="COL",
        help="R-J color column for NUVrJ classification (default: restR-J)",
    )
    parser.add_argument(
        "--z-min", type=float, default=0.1, metavar="FLOAT",
        help="Minimum redshift (default: 0.1)",
    )
    parser.add_argument(
        "--z-max", type=float, default=6.0, metavar="FLOAT",
        help="Maximum redshift (default: 6.0)",
    )
    parser.add_argument(
        "--mass-min", type=float, default=8.0, metavar="FLOAT",
        help="Minimum log10(M*/Msun) (default: 8.0)",
    )
    parser.add_argument(
        "--star-col", default="lp_type", metavar="COL",
        help="Star/galaxy column (default: 'lp_type'; set to '' to skip)",
    )
    parser.add_argument(
        "--galaxy-value", type=int, default=0, metavar="INT",
        help="Value of --star-col indicating a galaxy (default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="INT",
        help="Random seed for K-fold assignment (default: 42)",
    )
    parser.add_argument(
        "--no-starburst", action="store_true",
        help="Skip starburst flagging (no 'starburst'/'log_delta_ms' columns)",
    )
    parser.add_argument(
        "--sfr-col", default="lp_SFR_best", metavar="COL",
        help="SFR column for starburst flagging (default: lp_SFR_best, log10(SFR))",
    )
    parser.add_argument(
        "--sfr-linear", action="store_true",
        help="--sfr-col is linear SFR, not log10(SFR) (default: log10)",
    )
    parser.add_argument(
        "--starburst-threshold", type=float, default=3.0, metavar="FLOAT",
        help="SFR/SFR_MS threshold for starburst flag, Elbaz+2018 (default: 3.0)",
    )
    args = parser.parse_args()

    build_cosmos2020_catalogs(
        catalog_path=Path(args.catalog),
        n_splits=args.splits,
        split_type=args.split_type,
        z_col=args.z_col,
        mass_col=args.mass_col,
        ra_col=args.ra_col,
        dec_col=args.dec_col,
        nuv_r_col=args.nuv_r_col,
        r_j_col=args.r_j_col,
        z_min=args.z_min,
        z_max=args.z_max,
        mass_min=args.mass_min,
        star_col=args.star_col or None,
        star_galaxy_value=args.galaxy_value,
        seed=args.seed,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        stem=args.stem,
        flag_starburst=not args.no_starburst,
        sfr_col=args.sfr_col,
        sfr_is_log=not args.sfr_linear,
        starburst_threshold=args.starburst_threshold,
    )


if __name__ == "__main__":
    main()
