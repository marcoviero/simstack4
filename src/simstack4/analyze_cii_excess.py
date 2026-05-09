#!/usr/bin/env python3
"""
Detect [CII] 158um and CO line contamination in stacked FIR SEDs.

Method
------
For each stacked population:
1. Identify which photometric band(s) contain [CII] 157.74um (and other
   bright FIR lines) at that population's median redshift.
2. Compute the residual: excess = (data - model) in that band, where
   the model is the best-fit greybody continuum.
3. Express excess as:
   - fractional: excess / model  (expect ~0.1-1% for [CII])
   - line luminosity: L_line = excess * bandwidth * 4*pi*dL^2
   - L_line / L_IR ratio (the "[CII] deficit" diagnostic)
4. Correlate with Sigma_SFR and L_IR.

Expected signals
----------------
- [CII]/L_FIR ~ 0.1-1% for main sequence galaxies (Diaz-Santos+2017)
- [CII]/L_FIR ~ 0.01-0.1% for starbursts (the "deficit")
- CO lines ~10x fainter than [CII]
- Signal strongest in bins where [CII] falls near band center

References
----------
Diaz-Santos+2017 (ApJL 846, L2): [CII] deficit vs Sigma_IR
Smith+2017 (ApJ 834, 183): [CII]/FIR in local galaxies
Gullberg+2015 (MNRAS 449, 2883): [CII] in z~4.5 SMGs
Stacey+2010 (ApJ 724, 957): [CII] at high-z

Run
---
    from simstack4.plots import _parse_bins
    from analyze_cii_excess import analyze_line_excess
    fig, df = analyze_line_excess(wrapper)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ── Line catalog ─────────────────────────────────────────────────────
# rest wavelength in microns
LINES = {
    "[CII] 158": 157.74,
    "[NII] 205": 205.18,
    "[OI] 145": 145.53,
    "[NII] 122": 121.90,
    "[OIII] 88": 88.36,
    "[OI] 63": 63.18,
    "CO(3-2)": 866.96,
    "CO(4-3)": 650.25,
    "CO(5-4)": 520.23,
    "CO(6-5)": 433.56,
    "CO(7-6)": 371.65,
    "CI(1-0)": 609.14,
    "CI(2-1)": 370.42,
}

# Band edges (50% transmission, microns)
BAND_EDGES = {
    "PACS_100": (85, 125),
    "PACS_160": (130, 210),
    "SPIRE_250": (194, 313),
    "SPIRE_350": (283, 413),
    "SPIRE_500": (383, 693),
    "SCUBA2_850": (770, 940),
}


def _lines_in_band(z, band_lo, band_hi):
    """Return list of (line_name, rest_lam) for lines in band at redshift z."""
    hits = []
    for name, lam_rest in LINES.items():
        lam_obs = lam_rest * (1 + z)
        if band_lo <= lam_obs <= band_hi:
            hits.append((name, lam_rest, lam_obs))
    return hits


def _band_for_wavelength(wave_um):
    """Return band name and edges for an observed wavelength."""
    for band, (lo, hi) in BAND_EDGES.items():
        center = (lo + hi) / 2
        if abs(wave_um - center) / center < 0.3:  # within 30% of center
            return band, lo, hi
    return None, None, None


def analyze_line_excess(
    wrapper,
    *,
    primary_line="[CII] 158",
    min_tier="B",
    split_filter=None,
    save_path=None,
):
    """
    Analyze FIR line contamination in stacked SEDs.

    For each population, computes the excess flux in the band
    containing [CII] (or other specified line) relative to the
    greybody continuum fit.

    Parameters
    ----------
    wrapper : SimstackWrapper
    primary_line : str
        Line to focus on. Default "[CII] 158".
    min_tier : str
        Minimum fit quality tier.
    split_filter : list of int, optional
        Population class filter.
    save_path : str or Path, optional

    Returns
    -------
    fig, df : figure and DataFrame of excess measurements
    """
    try:
        from simstack4.plots import _parse_bins, _extract_pop_type
    except ImportError:
        from plots import _parse_bins, _extract_pop_type

    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed results found")
        return None, None

    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

    # Map names and wavelengths from config
    sr = wrapper.stacking_results
    wavelengths = np.array([wrapper.config.maps[m].wavelength for m in sr.map_names])
    map_names = list(sr.map_names)

    line_rest_lam = LINES.get(primary_line)
    if line_rest_lam is None:
        print(f"Unknown line '{primary_line}'. Available: {list(LINES.keys())}")
        return None, None

    print(f"Analyzing {primary_line} ({line_rest_lam:.1f} um) excess in stacked SEDs")
    print(f"Tier >= {min_tier}, {len(pr.sed_results)} populations")

    # ── extract excess per population ────────────────────────────────
    rows = []

    for pop_id, sed in pr.sed_results.items():
        if not sed.greybody_fit_success:
            continue
        tier = getattr(sed, "fit_quality_tier", None) or "C"
        if tier_rank.get(tier, 2) > min_rank:
            continue

        pop_type = _extract_pop_type(pop_id)
        if split_filter is not None:
            allowed = {f"split_{i}" for i in split_filter}
            if pop_type not in allowed and pop_type != "_all_":
                continue
        elif pop_type == "split_2":
            continue

        derived = pr.derived_quantities.get(pop_id)
        if not derived or derived.total_ir_luminosity <= 0:
            continue

        z = sed.median_redshift
        l_ir = derived.total_ir_luminosity

        # Which band has the primary line?
        lam_line_obs = line_rest_lam * (1 + z)

        # Find the matching band in our data
        line_band_idx = None
        line_band_name = None
        for i, w in enumerate(sed.wavelengths):
            band, blo, bhi = _band_for_wavelength(w)
            if band and blo <= lam_line_obs <= bhi:
                line_band_idx = i
                line_band_name = band
                break

        if line_band_idx is None:
            # Line not in any observed band at this z — record as control
            rows.append(
                {
                    "pop_id": pop_id,
                    "z": z,
                    "l_ir": l_ir,
                    "line_in_band": False,
                    "line_band": "none",
                    "lam_line_obs": lam_line_obs,
                    "excess_frac": np.nan,
                    "excess_jy": np.nan,
                    "excess_snr": np.nan,
                    "n_lines_in_band": 0,
                    "lines_in_band": "",
                    "tier": tier,
                }
            )
            # Get bin_properties for sigma_sfr etc
            props = sed.bin_properties or {}
            if isinstance(props, str):
                import ast

                try:
                    props = ast.literal_eval(props)
                except (ValueError, SyntaxError):
                    props = {}
            rows[-1]["log_sigma_sfr"] = props.get("log_sigma_sfr", np.nan)
            rows[-1]["stellar_mass"] = props.get("mass_med", sed.median_mass)
            rows[-1]["n_sources"] = sed.n_sources
            continue

        # Data and model in the line band
        flux_data = sed.flux_densities[line_band_idx]  # Jy
        flux_err = sed.flux_errors[line_band_idx]  # Jy

        # Evaluate model at the data wavelength
        if sed.model_wavelengths is not None and sed.model_fluxes is not None:
            # Interpolate model at the observed wavelength
            flux_model = np.interp(
                sed.wavelengths[line_band_idx],
                sed.model_wavelengths,
                sed.model_fluxes,
            )
        else:
            continue

        if flux_model <= 0:
            continue

        excess_jy = flux_data - flux_model
        excess_frac = excess_jy / flux_model
        excess_snr = excess_jy / flux_err if flux_err > 0 else 0

        # What OTHER lines are also in this band?
        _, blo, bhi = _band_for_wavelength(sed.wavelengths[line_band_idx])
        all_lines = _lines_in_band(z, blo, bhi) if blo else []
        other_lines = [name for name, _, _ in all_lines if name != primary_line]

        # Residuals in ALL bands (for comparison)
        all_excess_frac = []
        for j in range(len(sed.wavelengths)):
            if sed.model_wavelengths is not None:
                fm = np.interp(
                    sed.wavelengths[j], sed.model_wavelengths, sed.model_fluxes
                )
                if fm > 0:
                    all_excess_frac.append((sed.flux_densities[j] - fm) / fm)
                else:
                    all_excess_frac.append(np.nan)
            else:
                all_excess_frac.append(np.nan)

        # Mean excess in bands WITHOUT the primary line (baseline)
        baseline_excess = []
        for j in range(len(sed.wavelengths)):
            if j == line_band_idx:
                continue
            _, blo_j, bhi_j = _band_for_wavelength(sed.wavelengths[j])
            if blo_j is not None:
                lines_j = _lines_in_band(z, blo_j, bhi_j)
                has_primary = any(n == primary_line for n, _, _ in lines_j)
                if not has_primary and np.isfinite(all_excess_frac[j]):
                    baseline_excess.append(all_excess_frac[j])

        baseline_mean = np.mean(baseline_excess) if baseline_excess else 0

        # Get bin_properties
        props = sed.bin_properties or {}
        if isinstance(props, str):
            import ast

            try:
                props = ast.literal_eval(props)
            except (ValueError, SyntaxError):
                props = {}

        rows.append(
            {
                "pop_id": pop_id,
                "z": z,
                "l_ir": l_ir,
                "line_in_band": True,
                "line_band": line_band_name,
                "lam_line_obs": lam_line_obs,
                "excess_frac": excess_frac,
                "excess_frac_corrected": excess_frac - baseline_mean,
                "excess_jy": excess_jy,
                "excess_snr": excess_snr,
                "baseline_frac": baseline_mean,
                "flux_data": flux_data,
                "flux_model": flux_model,
                "flux_err": flux_err,
                "n_lines_in_band": len(all_lines),
                "lines_in_band": " + ".join([n for n, _, _ in all_lines]),
                "log_sigma_sfr": props.get("log_sigma_sfr", np.nan),
                "stellar_mass": props.get("mass_med", sed.median_mass),
                "n_sources": sed.n_sources,
                "tier": tier,
            }
        )

    if not rows:
        print("No populations found")
        return None, None

    df = pd.DataFrame(rows)
    df_line = df[df["line_in_band"]].copy()
    df_ctrl = df[~df["line_in_band"]].copy()

    print(f"\n{'='*60}")
    print(f"{primary_line} EXCESS ANALYSIS")
    print(f"{'='*60}")
    print(f"  Populations with {primary_line} in a band: {len(df_line)}")
    print(f"  Control (line not in any band): {len(df_ctrl)}")

    if len(df_line) > 0:
        ef = df_line["excess_frac"]
        efc = df_line.get("excess_frac_corrected", ef)
        valid = np.isfinite(ef)
        if np.any(valid):
            print(f"\n  Raw fractional excess (data-model)/model:")
            print(
                f"    median = {ef[valid].median():+.4f} "
                f"({ef[valid].median()*100:+.2f}%)"
            )
            print(f"    mean   = {ef[valid].mean():+.4f}")
            print(f"    std    = {ef[valid].std():.4f}")
        valid_c = np.isfinite(efc)
        if np.any(valid_c):
            print(f"\n  Baseline-corrected excess:")
            print(
                f"    median = {efc[valid_c].median():+.4f} "
                f"({efc[valid_c].median()*100:+.2f}%)"
            )

        # By band
        print(f"\n  By band:")
        for band in df_line["line_band"].unique():
            bm = df_line["line_band"] == band
            n = bm.sum()
            med = ef[bm & valid].median() if (bm & valid).any() else np.nan
            print(
                f"    {band:>12}: {n:>3} pops, "
                f"median excess = {med:+.4f} ({med*100:+.2f}%)"
            )

        # Detection significance
        if np.any(valid):
            mean_excess = ef[valid].mean()
            sem = ef[valid].std() / np.sqrt(valid.sum())
            significance = mean_excess / sem if sem > 0 else 0
            print(
                f"\n  Detection significance: {significance:.1f} sigma "
                f"(mean excess / SEM)"
            )

    # ── plotting ─────────────────────────────────────────────────────
    has_sigma = np.isfinite(df_line.get("log_sigma_sfr", pd.Series())).any()
    n_panels = 4 + (1 if has_sigma else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    cmap = "plasma"

    # Panel 1: excess vs redshift, colored by band
    ax = axes[0]
    band_colors = {
        "PACS_160": "C0",
        "SPIRE_250": "C1",
        "SPIRE_350": "C2",
        "SPIRE_500": "C3",
        "SCUBA2_850": "C4",
    }
    for band, color in band_colors.items():
        bm = df_line["line_band"] == band
        if bm.any():
            ax.scatter(
                df_line.loc[bm, "z"],
                df_line.loc[bm, "excess_frac"] * 100,
                c=color,
                s=30,
                alpha=0.7,
                label=band.replace("_", " "),
            )
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(f"{primary_line} excess (%)")
    ax.set_title(f"Excess vs redshift")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.2)

    # Panel 2: excess vs L_IR (the deficit relation)
    ax = axes[1]
    valid = df_line["line_in_band"] & np.isfinite(df_line["excess_frac"])
    if valid.any():
        sc = ax.scatter(
            df_line.loc[valid, "l_ir"],
            df_line.loc[valid, "excess_frac"] * 100,
            c=df_line.loc[valid, "z"],
            cmap=cmap,
            s=30,
            alpha=0.7,
        )
        plt.colorbar(sc, ax=ax, label="Redshift")
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    # Reference: CII/FIR ~ 0.3% for L_IR < 1e11, dropping to 0.03% at 1e13
    lir_ref = np.logspace(10, 13, 50)
    cii_frac_ref = 0.3 * (lir_ref / 1e11) ** (-0.5)  # approximate deficit
    ax.plot(
        lir_ref, cii_frac_ref, "r--", lw=1.5, alpha=0.6, label="[CII] deficit (approx)"
    )
    ax.set_xscale("log")
    ax.set_xlabel("L_IR (L_sun)")
    ax.set_ylabel(f"{primary_line} excess (%)")
    ax.set_title("Excess vs L_IR\n([CII] deficit)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 3: baseline-corrected excess vs L_IR
    ax = axes[2]
    if "excess_frac_corrected" in df_line.columns:
        valid_c = np.isfinite(df_line["excess_frac_corrected"])
        if valid_c.any():
            sc = ax.scatter(
                df_line.loc[valid_c, "l_ir"],
                df_line.loc[valid_c, "excess_frac_corrected"] * 100,
                c=df_line.loc[valid_c, "z"],
                cmap=cmap,
                s=30,
                alpha=0.7,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")
    ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
    ax.plot(
        lir_ref, cii_frac_ref, "r--", lw=1.5, alpha=0.6, label="[CII] deficit (approx)"
    )
    ax.set_xscale("log")
    ax.set_xlabel("L_IR (L_sun)")
    ax.set_ylabel(f"Corrected {primary_line} excess (%)")
    ax.set_title("Baseline-corrected\nexcess vs L_IR")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    # Panel 4: histogram of excess (line-in-band vs control)
    ax = axes[3]
    valid_line = df_line["excess_frac"].dropna() * 100
    if len(valid_line) > 0:
        bins_h = np.linspace(
            max(valid_line.min(), -5),
            min(valid_line.max(), 5),
            30,
        )
        ax.hist(
            valid_line,
            bins=bins_h,
            alpha=0.6,
            color="C3",
            label=f"{primary_line} in band (N={len(valid_line)})",
            density=True,
        )
    # Compare: residuals from non-line bands
    if "baseline_frac" in df_line.columns:
        bl = df_line["baseline_frac"].dropna() * 100
        if len(bl) > 0:
            ax.hist(
                bl,
                bins=bins_h,
                alpha=0.4,
                color="gray",
                label=f"Baseline (other bands)",
                density=True,
            )
    ax.axvline(0, color="k", ls="--", lw=1, alpha=0.5)
    if len(valid_line) > 0:
        ax.axvline(
            valid_line.median(),
            color="C3",
            ls="-",
            lw=2,
            alpha=0.7,
            label=f"Median = {valid_line.median():.2f}%",
        )
    ax.set_xlabel("Fractional excess (%)")
    ax.set_ylabel("Density")
    ax.set_title("Excess distribution")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # Panel 5 (if Sigma_SFR available): excess vs Sigma_SFR
    if has_sigma:
        ax = axes[4]
        valid_s = np.isfinite(df_line["log_sigma_sfr"]) & np.isfinite(
            df_line["excess_frac"]
        )
        if valid_s.any():
            sc = ax.scatter(
                df_line.loc[valid_s, "log_sigma_sfr"],
                df_line.loc[valid_s, "excess_frac"] * 100,
                c=df_line.loc[valid_s, "z"],
                cmap=cmap,
                s=30,
                alpha=0.7,
            )
            plt.colorbar(sc, ax=ax, label="Redshift")
        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("log Sigma_SFR (M_sun/yr/kpc^2)")
        ax.set_ylabel(f"{primary_line} excess (%)")
        ax.set_title("Excess vs Sigma_SFR\n(expect deficit at high Sigma)")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"{primary_line} line excess in stacked SEDs "
        f"({len(df_line)} detections, {len(df_ctrl)} controls, "
        f"tier >= {min_tier})",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\n  Saved: {save_path}")

    return fig, df


def analyze_all_lines(wrapper, min_tier="B", split_filter=None):
    """
    Quick summary of excess for ALL bright FIR lines.

    Prints a table of median excess per line, sorted by strength.
    """
    try:
        from simstack4.plots import _parse_bins, _extract_pop_type
    except ImportError:
        from plots import _parse_bins, _extract_pop_type

    results = []
    for line_name in [
        "[CII] 158",
        "[NII] 205",
        "[OI] 145",
        "[NII] 122",
        "[OIII] 88",
        "[OI] 63",
        "CO(5-4)",
        "CO(6-5)",
        "CO(7-6)",
    ]:
        _, df = analyze_line_excess(
            wrapper,
            primary_line=line_name,
            min_tier=min_tier,
            split_filter=split_filter,
        )
        plt.close()  # don't show individual plots

        if df is not None:
            df_det = df[df["line_in_band"]]
            valid = np.isfinite(df_det["excess_frac"])
            if valid.any():
                results.append(
                    {
                        "line": line_name,
                        "n_pops": valid.sum(),
                        "median_excess_pct": df_det.loc[valid, "excess_frac"].median()
                        * 100,
                        "mean_excess_pct": df_det.loc[valid, "excess_frac"].mean()
                        * 100,
                        "std_pct": df_det.loc[valid, "excess_frac"].std() * 100,
                        "significance": (
                            (
                                df_det.loc[valid, "excess_frac"].mean()
                                / (
                                    df_det.loc[valid, "excess_frac"].std()
                                    / np.sqrt(valid.sum())
                                )
                            )
                            if valid.sum() > 1
                            else 0
                        ),
                    }
                )

    if results:
        summary = pd.DataFrame(results).sort_values(
            "median_excess_pct", ascending=False
        )
        print("\n" + "=" * 70)
        print("LINE EXCESS SUMMARY (all bright FIR lines)")
        print("=" * 70)
        print(summary.to_string(index=False, float_format="%.3f"))
        return summary
    return None


# ── CLI ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Import and call analyze_line_excess(wrapper) or analyze_all_lines(wrapper)")
    print()
    print("Example:")
    print("  from analyze_cii_excess import analyze_line_excess, analyze_all_lines")
    print()
    print("  # Single line analysis with plots")
    print("  fig, df = analyze_line_excess(wrapper)")
    print("  fig, df = analyze_line_excess(wrapper, primary_line='CO(5-4)')")
    print()
    print("  # Summary table of all lines")
    print("  summary = analyze_all_lines(wrapper)")
