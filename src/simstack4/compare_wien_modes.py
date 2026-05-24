"""
Wien-mode comparison: quantify warm dust bias on T_dust evolution.

Run the same stacking with three Wien-side models and compare
T_dust(z) to determine how much of the apparent temperature
evolution at z > 2 is real vs warm dust contamination.

Usage
-----
    from compare_wien_modes import compare_wien_modes

    fig, df = compare_wien_modes(
        wrapper_powerlaw,    # wien_mode = "powerlaw"
        wrapper_additive,    # wien_mode = "additive"
        wrapper_physical,    # wien_mode = "physical"
    )

    # Or from saved pickles:
    fig, df = compare_wien_modes_from_pickles(
        "results_powerlaw.pkl",
        "results_additive.pkl",
        "results_physical.pkl",
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from simstack4.plots import _parse_bins, _extract_pop_type
except ImportError:
    from plots import _parse_bins, _extract_pop_type


def _extract_temperatures(pr, label, split_filter=None, min_tier="B"):
    """Extract T_dust, z, M* from processed results."""
    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

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
        l_ir = derived.total_ir_luminosity if derived else 0

        rows.append(
            {
                "pop_id": pop_id,
                "z": sed.median_redshift,
                "T_dust": sed.dust_temperature_rest_frame,
                "T_err": sed.dust_temperature_error,
                "stellar_mass": sed.median_mass,
                "log_l_ir": np.log10(l_ir) if l_ir > 0 else np.nan,
                "l_ir": l_ir,
                "n_sources": sed.n_sources,
                "tier": tier,
                "chi2": sed.chi2_reduced,
                "amplitude": sed.amplitude,
                "mode": label,
            }
        )

    return pd.DataFrame(rows)


def compare_wien_modes(
    wrapper_powerlaw,
    wrapper_additive=None,
    wrapper_physical=None,
    *,
    split_filter=None,
    min_tier="B",
    z_bin_width=0.5,
    save_path=None,
):
    """
    Compare T_dust across Wien-side models.

    Parameters
    ----------
    wrapper_powerlaw : SimstackWrapper
        Results with wien_mode = "powerlaw" (baseline).
    wrapper_additive : SimstackWrapper, optional
        Results with wien_mode = "additive" (PAH on top of power-law).
    wrapper_physical : SimstackWrapper, optional
        Results with wien_mode = "physical" (warm dust replaces power-law).
    split_filter : list of int, optional
    min_tier : str
    z_bin_width : float
        Bin width for median T_dust(z) curves.
    save_path : str or Path, optional

    Returns
    -------
    fig, df_all : figure and combined DataFrame
    """
    # Extract temperatures from each mode
    modes = []

    pr_pl = getattr(wrapper_powerlaw, "processed_results", None)
    if pr_pl:
        df_pl = _extract_temperatures(pr_pl, "powerlaw", split_filter, min_tier)
        modes.append(("powerlaw", df_pl, "#1f77b4", "-", "ν⁻ᵅ power-law (baseline)"))

    if wrapper_additive is not None:
        pr_ad = getattr(wrapper_additive, "processed_results", None)
        if pr_ad:
            df_ad = _extract_temperatures(pr_ad, "additive", split_filter, min_tier)
            modes.append(("additive", df_ad, "#ff7f0e", "--", "Power-law + PAH"))

    if wrapper_physical is not None:
        pr_ph = getattr(wrapper_physical, "processed_results", None)
        if pr_ph:
            df_ph = _extract_temperatures(pr_ph, "physical", split_filter, min_tier)
            modes.append(
                ("physical", df_ph, "#2ca02c", "-.", "Warm dust + PAH (physical)")
            )

    if not modes:
        print("No valid results found")
        return None, None

    df_all = pd.concat([df for _, df, _, _, _ in modes], ignore_index=True)

    n_panels = 3 if len(modes) > 1 else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6))

    # ── Panel 1: T_dust vs z, all modes overlaid ─────────────────
    ax = axes[0]

    z_max = df_all["z"].max()
    z_edges = np.arange(0, z_max + z_bin_width, z_bin_width)

    for mode_name, df_mode, color, ls, label in modes:
        # Individual points
        ax.scatter(df_mode["z"], df_mode["T_dust"], c=color, s=15, alpha=0.3, zorder=2)

        # Binned medians with IQR
        z_mids, t_meds, t_lo, t_hi = [], [], [], []
        for i in range(len(z_edges) - 1):
            mask = (df_mode["z"] >= z_edges[i]) & (df_mode["z"] < z_edges[i + 1])
            if mask.sum() >= 3:
                temps = df_mode.loc[mask, "T_dust"].values
                z_mids.append((z_edges[i] + z_edges[i + 1]) / 2)
                t_meds.append(np.median(temps))
                t_lo.append(np.median(temps) - np.percentile(temps, 16))
                t_hi.append(np.percentile(temps, 84) - np.median(temps))

        if z_mids:
            ax.errorbar(
                z_mids,
                t_meds,
                yerr=[t_lo, t_hi],
                fmt="o" + ls[0],
                color=color,
                ms=8,
                mfc="white",
                mew=2,
                capsize=4,
                lw=2,
                label=label,
                zorder=5,
            )
            ax.plot(z_mids, t_meds, ls, color=color, lw=2, alpha=0.7, zorder=4)

    # Literature
    z_lit = np.linspace(0, min(z_max * 1.1, 10), 100)
    ax.plot(
        z_lit,
        32.9 + 4.60 * (z_lit - 2.0),
        ":",
        color="gray",
        lw=1.5,
        alpha=0.5,
        label="Schreiber+18",
    )

    ax.set_xlabel("Redshift", fontsize=13)
    ax.set_ylabel("T$_{dust}$ (K)", fontsize=13)
    ax.set_title("T$_{dust}$(z) by Wien model", fontsize=14)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)

    # ── Panel 2: ΔT_dust vs z (difference from baseline) ────────
    if len(modes) > 1:
        ax = axes[1]

        # Match populations by pop_id across modes
        df_baseline = modes[0][1].set_index("pop_id")

        for mode_name, df_mode, color, ls, label in modes[1:]:
            df_m = df_mode.set_index("pop_id")

            # Find common populations
            common = df_baseline.index.intersection(df_m.index)
            if len(common) == 0:
                continue

            z_common = df_baseline.loc[common, "z"].values
            dT = (
                df_m.loc[common, "T_dust"].values
                - df_baseline.loc[common, "T_dust"].values
            )

            ax.scatter(z_common, dT, c=color, s=20, alpha=0.4, zorder=2)

            # Binned medians
            z_mids, dt_meds, dt_lo, dt_hi = [], [], [], []
            for i in range(len(z_edges) - 1):
                mask = (z_common >= z_edges[i]) & (z_common < z_edges[i + 1])
                if mask.sum() >= 3:
                    dts = dT[mask]
                    z_mids.append((z_edges[i] + z_edges[i + 1]) / 2)
                    dt_meds.append(np.median(dts))
                    dt_lo.append(np.median(dts) - np.percentile(dts, 16))
                    dt_hi.append(np.percentile(dts, 84) - np.median(dts))

            if z_mids:
                ax.errorbar(
                    z_mids,
                    dt_meds,
                    yerr=[dt_lo, dt_hi],
                    fmt="o",
                    color=color,
                    ms=8,
                    mfc="white",
                    mew=2,
                    capsize=4,
                    lw=2,
                    label=label,
                    zorder=5,
                )
                ax.plot(z_mids, dt_meds, ls, color=color, lw=2, alpha=0.7, zorder=4)

            # Statistics
            valid = np.isfinite(dT)
            if valid.sum() > 0:
                med = np.median(dT[valid])
                mad = 1.4826 * np.median(np.abs(dT[valid] - med))
                ax.text(
                    0.95,
                    0.95
                    - 0.08 * (modes.index((mode_name, df_mode, color, ls, label)) - 1),
                    f"median ΔT = {med:+.1f}K, σ = {mad:.1f}K",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=9,
                    color=color,
                    bbox=dict(fc="white", alpha=0.8),
                )

        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        ax.set_xlabel("Redshift", fontsize=13)
        ax.set_ylabel("ΔT$_{dust}$ (mode − powerlaw) [K]", fontsize=13)
        ax.set_title("Warm dust bias on T$_{dust}$", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    # ── Panel 3: ΔT vs T_dust (does bias scale with temperature?) ─
    if len(modes) > 1 and n_panels >= 3:
        ax = axes[2]

        df_baseline = modes[0][1].set_index("pop_id")

        for mode_name, df_mode, color, ls, label in modes[1:]:
            df_m = df_mode.set_index("pop_id")
            common = df_baseline.index.intersection(df_m.index)
            if len(common) == 0:
                continue

            T_base = df_baseline.loc[common, "T_dust"].values
            dT = df_m.loc[common, "T_dust"].values - T_base
            z_common = df_baseline.loc[common, "z"].values

            sc = ax.scatter(
                T_base, dT, c=z_common, cmap="plasma", s=25, alpha=0.6, zorder=2
            )

            # Fit slope
            valid = np.isfinite(T_base) & np.isfinite(dT)
            if valid.sum() > 5:
                try:
                    coeffs = np.polyfit(T_base[valid], dT[valid], 1)
                    T_grid = np.linspace(T_base[valid].min(), T_base[valid].max(), 50)
                    ax.plot(
                        T_grid,
                        np.polyval(coeffs, T_grid),
                        ls,
                        color=color,
                        lw=2,
                        alpha=0.7,
                        label=f"{label}\nslope={coeffs[0]:.2f} K/K",
                    )
                except Exception:
                    pass

        ax.axhline(0, color="k", ls="--", lw=1, alpha=0.5)
        plt.colorbar(sc, ax=ax, label="Redshift", shrink=0.8)
        ax.set_xlabel("T$_{dust}$ (powerlaw) [K]", fontsize=13)
        ax.set_ylabel("ΔT$_{dust}$ [K]", fontsize=13)
        ax.set_title("Bias vs temperature\n(slope = fractional bias)", fontsize=14)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Wien-side model comparison  |  {len(df_all)} populations "
        f"(tier ≥ {min_tier})",
        fontsize=14,
        y=1.02,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    # ── Print summary table ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("WIEN-SIDE MODEL COMPARISON")
    print(f"{'='*70}")

    for mode_name, df_mode, _, _, label in modes:
        valid = np.isfinite(df_mode["T_dust"])
        print(f"\n  {label}:")
        print(f"    N = {valid.sum()}")
        print(
            f"    T_dust: {df_mode.loc[valid, 'T_dust'].median():.1f}K "
            f"(median), {df_mode.loc[valid, 'T_dust'].min():.0f}–"
            f"{df_mode.loc[valid, 'T_dust'].max():.0f}K (range)"
        )
        if "log_l_ir" in df_mode.columns:
            vlir = np.isfinite(df_mode["log_l_ir"])
            if vlir.any():
                print(
                    f"    log L_IR: {df_mode.loc[vlir, 'log_l_ir'].median():.2f} (median)"
                )

    if len(modes) > 1:
        df_base = modes[0][1].set_index("pop_id")
        print(f"\n  Differences from {modes[0][4]}:")
        for mode_name, df_mode, _, _, label in modes[1:]:
            df_m = df_mode.set_index("pop_id")
            common = df_base.index.intersection(df_m.index)
            if len(common) > 0:
                dT = (
                    df_m.loc[common, "T_dust"].values
                    - df_base.loc[common, "T_dust"].values
                )
                z_c = df_base.loc[common, "z"].values
                print(f"\n    {label}:")
                print(f"      N matched = {len(common)}")
                print(
                    f"      ΔT overall: {np.median(dT):+.1f}K "
                    f"(median), σ = {np.std(dT):.1f}K"
                )

                for z_lo, z_hi in [(0, 1), (1, 2), (2, 3), (3, 5), (5, 10)]:
                    zmask = (z_c >= z_lo) & (z_c < z_hi)
                    if zmask.sum() >= 3:
                        print(
                            f"      z={z_lo}–{z_hi}: ΔT = {np.median(dT[zmask]):+.1f}K "
                            f"(N={zmask.sum()})"
                        )

    return fig, df_all


def compare_wien_modes_from_pickles(
    path_powerlaw,
    path_additive=None,
    path_physical=None,
    **kwargs,
):
    """Load results from pickle files and compare."""
    import pickle

    class _FakeWrapper:
        def __init__(self, pkl_path):
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            # SimstackResults.load_results returns a dict
            if isinstance(data, dict):
                self.processed_results = type(
                    "PR",
                    (),
                    {
                        "sed_results": data.get("sed_results", {}),
                        "derived_quantities": data.get("derived_quantities", {}),
                    },
                )()
            else:
                self.processed_results = data

    wrappers = [_FakeWrapper(path_powerlaw)]
    if path_additive:
        wrappers.append(_FakeWrapper(path_additive))
    else:
        wrappers.append(None)
    if path_physical:
        wrappers.append(_FakeWrapper(path_physical))
    else:
        wrappers.append(None)

    return compare_wien_modes(*wrappers, **kwargs)


if __name__ == "__main__":
    print("Usage:")
    print("  from compare_wien_modes import compare_wien_modes")
    print()
    print("  # Run stacking three times with different wien_mode:")
    print("  # wrapper_pl:  wien_mode = 'powerlaw'")
    print("  # wrapper_add: wien_mode = 'additive'")
    print("  # wrapper_phy: wien_mode = 'physical'")
    print()
    print("  fig, df = compare_wien_modes(wrapper_pl, wrapper_add, wrapper_phy)")
    print()
    print("  # Or compare just two:")
    print("  fig, df = compare_wien_modes(wrapper_pl, wrapper_add)")
