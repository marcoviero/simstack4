"""
COSMOS2025 (COSMOSWeb) Catalog Preparation with K-fold Source Partitioning
===========================================================================

Builds K stacking-ready parquet catalogs from the COSMOSWeb master FITS files,
combining two existing pipelines:

  - `prepare_cosmos_catalog.py`   : the full COSMOSWeb science pipeline (quality
    cuts, NUVrJ, mass completeness, beta_UV/L_UV, sSFR/Delta_MS, Sersic morphology
    + Sigma_SFR). Reused here unmodified via import.
  - `prepare_cosmos2020_catalog.py`: the generic K-fold source-partitioning helpers
    (balanced round-robin split + signal/nuisance relabeling). Reused here
    unmodified via import.

Why a new script instead of extending either: COSMOS2020 (prepare_cosmos2020_catalog.py)
has no JWST-quality Sersic sizes, so it cannot produce Sigma_SFR. The K-fold
disjoint-fold ensemble (docs/pah-forward-model-4-brief.md "Objective 2") was only
ever implemented against COSMOS2020. This script produces the same K-fold outputs
against COSMOSWeb (2025) so PAHSpectrumModel.fit_evolving can be extended to use
Sigma_SFR (log_sigma_sfr) as an additional/alternative evolution driver alongside
sSFR.

Population classes in output parquets (matches prepare_cosmos2020_catalog.py)
-------------------------------------------------------------------------------
    0  sfg_signal  : star-forming galaxies in this split's signal partition (1/K)
    1  sfg_nuisance: star-forming galaxies in the other (K-1) partitions
    2  qt          : quiescent — identical in every split

All star-forming sources (complete_sfg + incomplete_sfg + starburst, i.e. the
`star_forming` flag from the COSMOSWeb pipeline) are pooled before splitting;
mass-completeness/starburst sub-classification is not preserved in
`population_class` but survives as continuous columns (log_stellar_mass,
log_delta_ms, starburst, mass_complete) for downstream binning.

Usage
-----
    prepare-cosmos2025-catalog --splits 3
    prepare-cosmos2025-catalog --splits 3 --paper p26   # Sersic+sigma_SFR preset
    prepare-cosmos2025-catalog --splits 1               # single catalog, no split
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .prepare_cosmos2020_catalog import (
    _assign_kfold_splits,
    _make_split_population_class,
    _save,
)
from .prepare_cosmos_catalog import (
    _schreiber_ms_sfr,
    apply_custom_flags,
    apply_mass_completeness,
    apply_photoz_quality,
    apply_quality_cuts,
    classify_nuvrj,
    compute_beta_luv_photometric,
    compute_beta_luv_template,
    compute_sersic_columns,
    flag_starbursts,
    load_config,
    load_fits_catalogs,
)

# =========================================================================
# Step 1: Full COSMOSWeb science pipeline (through population flags)
# =========================================================================


def build_cosmos2025_base(
    config: dict,
    catspath: Path | None = None,
) -> pd.DataFrame:
    """
    Run the COSMOSWeb pipeline through science-column computation and collapse
    population to the 2-class SFG/QT scheme used for K-fold partitioning.

    This mirrors `prepare_cosmos_catalog.build_catalog()` step-for-step (quality
    cuts, NUVrJ, mass completeness, photo-z quality, starburst, beta_UV/L_UV,
    sSFR/Delta_MS, Sersic + Sigma_SFR) but skips the 4-class `encode_populations`
    step in favor of `population_class = 0 if star_forming else 2`, matching the
    convention `prepare_cosmos2020_catalog.py` and the production
    `cosmos25_PAH_dithered_3d.toml` catalog already use for K-fold / label splits.

    Returns
    -------
    df_sel : DataFrame
        Quality-cut catalog with all science columns and a 2-class
        `population_class` (0=star_forming, 2=quiescent), NOT yet K-fold split.
    """
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

    if pop_cfg.get("flag_photoz_quality", False):
        photoz_reliable = apply_photoz_quality(df, config)
    else:
        photoz_reliable = pd.Series(True, index=df.index)
        print("  Photo-z quality:   SKIPPED (flag_photoz_quality=false)")

    if pop_cfg.get("flag_starburst", True):
        is_starburst = flag_starbursts(df, config)
    else:
        is_starburst = pd.Series(False, index=df.index)
        print("  Starburst flag: SKIPPED (flag_starburst=false)")

    df["star_forming"] = star_forming.astype(int)
    df["mass_complete"] = mass_complete.astype(int)
    df["photoz_reliable"] = photoz_reliable.astype(int)
    df["starburst"] = is_starburst.astype(int)

    apply_custom_flags(df, config)

    # 2-class population scheme for K-fold splitting (0=sfg pool, 2=qt),
    # instead of the 4-class complete_sfg/incomplete_sfg/qt/sb scheme.
    print("\n  Population classes (2-class SFG/QT scheme for K-fold splitting):")
    pop_class = np.where(star_forming.values, 0, 2).astype(int)
    df["population_class"] = pop_class
    n_sfg = int((pop_class == 0).sum())
    n_qt = int((pop_class == 2).sum())
    print(f"    class 0 (            sfg): {n_sfg:>8,}")
    print(f"    class 2 (             qt): {n_qt:>8,}")

    # ── 4. Subset to quality-passing sources ──────────────────────────────
    df_sel = df[quality_mask].copy()
    print(
        f"\n  After quality cuts: {len(df_sel):,} / {n_initial:,} "
        f"({100 * len(df_sel) / max(n_initial, 1):.1f}%)"
    )

    # ── 5. Standardize column names ────────────────────────────────────────
    cols = config["columns"]
    z_col = cols["redshift"]
    mass_col = cols["stellar_mass"]
    sfr_col = cols.get("sfr")
    sfr_log = cols.get("sfr_is_log", True)
    ra_col = cols["ra"]
    dec_col = cols["dec"]

    rename = {
        ra_col: "ra",
        dec_col: "dec",
        z_col: "redshift",
        mass_col: "log_stellar_mass",
    }
    if sfr_col and sfr_col in df_sel.columns:
        rename[sfr_col] = "log_sfr" if sfr_log else "sfr"
    for old, new in rename.items():
        if old in df_sel.columns and old != new:
            df_sel[new] = df_sel[old]

    # ── 6. UV beta and L_UV ─────────────────────────────────────────────────
    method = config["beta_luv"].get("method", "both")
    print(f"\n[4] UV beta and L_UV (method={method})")

    if method in ("template", "both"):
        beta_tmpl, log_luv_tmpl = compute_beta_luv_template(df_sel, config)
        df_sel["beta_uv"] = beta_tmpl
        df_sel["log_l_uv"] = log_luv_tmpl

    if method in ("photometric", "both"):
        print("  Photometric fit (this may take a few minutes)...")
        beta_phot, log_luv_phot = compute_beta_luv_photometric(df_sel, config)
        df_sel["beta_uv_phot"] = beta_phot
        df_sel["log_l_uv_phot"] = log_luv_phot

    # ── 7. sSFR and Delta_MS ────────────────────────────────────────────────
    if "log_sfr" in df_sel.columns and "log_stellar_mass" in df_sel.columns:
        df_sel["log_ssfr"] = (
            df_sel["log_sfr"].values - df_sel["log_stellar_mass"].values
        )
        log_sfr_ms = np.log10(
            np.maximum(
                _schreiber_ms_sfr(
                    df_sel["log_stellar_mass"].values, df_sel["redshift"].values
                ),
                1e-20,
            )
        )
        df_sel["log_delta_ms"] = df_sel["log_sfr"].values - log_sfr_ms

    # ── 8. Morphology (Sersic + Sigma_SFR) ──────────────────────────────────
    morph = config["morphology"]
    if morph.get("sersic", False):
        print("\n[5] Sersic morphology + Sigma_SFR")
        df_sel = compute_sersic_columns(df_sel, config)
    else:
        print(
            "\n[5] Sersic: SKIPPED (morphology.sersic=false) "
            "— log_sigma_sfr will NOT be available"
        )

    return df_sel


# =========================================================================
# Step 2: K-fold source partitioning + save
# =========================================================================


def build_cosmos2025_folds(
    config: dict,
    n_splits: int = 3,
    seed: int = 42,
    catspath: Path | None = None,
    output_dir: Path | None = None,
    stem: str | None = None,
) -> list[Path]:
    """
    Build K COSMOSWeb (2025) parquet catalogs for K-fold PAH forward-model analysis.

    Parameters
    ----------
    config : dict
        Merged config from `prepare_cosmos_catalog.load_config()`.
    n_splits : int
        Number of independent source partitions (K). K=3 is the default.
        Use K=1 to produce a single unsplit catalog (all SFGs as class 0).
    seed : int
        Random seed for reproducible K-fold assignment.
    catspath : Path or None
        Root directory for input catalogs. Falls back to $CATSPATH/cosmos.
    output_dir : Path or None
        Directory for output parquets. Defaults to catspath.
    stem : str or None
        Base name for output files. Defaults to f"COSMOSWeb_{paper_id}_PAH".

    Returns
    -------
    out_paths : list[Path]
        Paths to the K written parquet files (or 1 file when n_splits=1).
    """
    df_sel = build_cosmos2025_base(config, catspath=catspath)
    pop_class = df_sel["population_class"].values.astype(int)

    if catspath is None:
        import os

        cats_env = os.environ.get("CATSPATH", "")
        catspath = Path(cats_env) / "cosmos" if cats_env else Path(".")
    if output_dir is None:
        output_dir = catspath
    if stem is None:
        paper_id = config.get("paper", {}).get("id", "cosmos25")
        stem = f"COSMOSWeb_{paper_id}_PAH"

    # ── K-fold split assignment ────────────────────────────────────────────
    if n_splits > 1:
        print(f"\n[6] K-fold split assignment (K={n_splits}, seed={seed})")
        split_id = _assign_kfold_splits(pop_class, n_splits, seed)
    else:
        split_id = np.zeros(len(df_sel), dtype=int)
        split_id[pop_class == 2] = -1  # QTs

    # ── Write output parquets ──────────────────────────────────────────────
    print(f"\n[7] Writing {n_splits} output parquet(s)")
    out_paths: list[Path] = []

    for k in range(n_splits):
        df_out = df_sel.copy()
        df_out["population_class"] = _make_split_population_class(
            pop_class, split_id, target_split=k
        )
        df_out["catalog_split_id"] = k

        n_sig = int((df_out["population_class"] == 0).sum())
        n_nui = int((df_out["population_class"] == 1).sum())
        n_qt_ = int((df_out["population_class"] == 2).sum())

        if n_splits == 1:
            fname = f"{stem}_catalog.parquet"
        else:
            fname = f"{stem}_split{k}of{n_splits}.parquet"
        out_path = output_dir / fname

        print(
            f"\n  Split {k}: class 0 (signal)={n_sig:,}  "
            f"class 1 (nuisance)={n_nui:,}  class 2 (QT)={n_qt_:,}"
        )
        _save(df_out, out_path)
        out_paths.append(out_path)

    print("\n" + "=" * 70)
    print(f"  Done.  {n_splits} catalog(s) written to {output_dir}")
    print("=" * 70)
    print("\nTOML stanza for output catalogs:")
    print("  [catalog.astrometry]")
    print('  ra  = "ra"')
    print('  dec = "dec"')
    print("  [catalog.classification]")
    print('  split_type = "labels"')
    print(
        '  bin_property_columns = ["log_stellar_mass", "beta_uv_phot", '
        '"log_l_uv_phot", "log_sfr", "log_ssfr", "log_sigma_sfr"]'
    )
    print("  [catalog.classification.split_params]")
    print('  id = "population_class"')
    print("  # 0 = sfg_signal, 1 = sfg_nuisance, 2 = qt")

    return out_paths


# =========================================================================
# CLI entry point
# =========================================================================


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="prepare-cosmos2025-catalog",
        description=(
            "Build K-fold COSMOSWeb (2025) parquet catalogs for PAH forward-model "
            "analysis, including Sigma_SFR (requires JWST-quality Sersic sizes, "
            "unavailable in COSMOS2020)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: 3-fold split, defaults.toml preset
  prepare-cosmos2025-catalog --splits 3

  # 3-fold split with the Parente+2026 preset (Sersic+sigma_SFR already on)
  prepare-cosmos2025-catalog --splits 3 --paper p26

  # Single catalog (no splitting)
  prepare-cosmos2025-catalog --splits 1

Population classes in output parquets:
  0  sfg_signal   : SFGs in this split's signal partition (1/K)
  1  sfg_nuisance : SFGs in the other (K-1) partitions
  2  qt           : quiescent (identical across all splits)
""",
    )
    parser.add_argument(
        "--paper", choices=["w26", "p26", "a26"],
        help="Paper preset (overrides defaults); morphology is forced on regardless "
             "(see --no-morphology)",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="TOML config file (merged on top of defaults + paper preset)",
    )
    parser.add_argument(
        "--catspath", metavar="DIR",
        help="Catalog directory (overrides $CATSPATH/cosmos)",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR",
        help="Output directory for split parquets (default: same as --catspath)",
    )
    parser.add_argument(
        "--stem", metavar="NAME",
        help='Base name for output files (default: "COSMOSWeb_{paper_id}_PAH")',
    )
    parser.add_argument(
        "--splits", type=int, default=3, metavar="K",
        help="Number of independent source partitions (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, metavar="INT",
        help="Random seed for K-fold assignment (default: 42)",
    )
    parser.add_argument(
        "--no-morphology", action="store_true",
        help="Disable Sersic + Sigma_SFR (log_sigma_sfr will not be computed)",
    )
    parser.add_argument(
        "--no-qt", action="store_true",
        help="Disable NUVrJ quiescent flagging (all sources treated as SFG)",
    )
    parser.add_argument(
        "--mass-min", type=float, metavar="FLOAT",
        help="Minimum log10(M*/Msun) threshold; overrides config",
    )
    args = parser.parse_args()

    overrides: dict = {}
    # Morphology defaults to ON in this script (that's the point of the K-fold
    # catalogs) regardless of the paper preset's morphology settings.
    overrides.setdefault("morphology", {})["sersic"] = not args.no_morphology
    overrides.setdefault("morphology", {})["sigma_sfr"] = not args.no_morphology
    if args.no_qt:
        overrides.setdefault("populations", {})["flag_quiescent"] = False
    if args.mass_min is not None:
        overrides.setdefault("selection", {})["mass_min"] = args.mass_min

    config = load_config(
        paper=args.paper,
        config_path=args.config,
        cli_overrides=overrides,
    )

    catspath = Path(args.catspath) if args.catspath else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    build_cosmos2025_folds(
        config,
        n_splits=args.splits,
        seed=args.seed,
        catspath=catspath,
        output_dir=output_dir,
        stem=args.stem,
    )


if __name__ == "__main__":
    main()
