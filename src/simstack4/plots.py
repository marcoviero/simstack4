"""
Complete Plotting and visualization for Simstack4

This module creates publication-quality plots from saved stacking results,
maintaining modularity so plot generation is separate from the main algorithm.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from matplotlib.patches import Rectangle
    import matplotlib.gridspec as gridspec

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    sns = None

from .results import SimstackResults, SEDResults
from .utils import setup_logging
from .exceptions import PlotError

logger = setup_logging()

"""
SED Grid Plot for Simstack4

Plots stacked flux densities and fitted greybody models on a
(column x row) grid of binning dimensions.  Handles both
population-type splits (UVJ, NuVRJ → sfg / quiescent / …) and
non-split runs automatically.

Usage:
    from simstack4.plots import plot_sed_grid

    fig = plot_sed_grid(wrapper, save_path='sed_grid.pdf')

    # With temperature-prior overlay for diagnostics:
    fig = plot_sed_grid(wrapper, show_prior=True, save_path='sed_diag.pdf')
"""

import logging
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Centralised plot configuration — edit here, not in the code below
# ---------------------------------------------------------------------------

# ── Font sizes ────────────────────────────────────────────────────────────
FONT = {
    "suptitle": 14,
    "col_header": 12,
    "row_label": 12,
    "axis_label": 12,
    "tick": 11,
    "legend": 15,
    "empty_cell": 14,
    "colorbar": 12,
    "colorbar_tick": 11,
}

# ── Marker / line styling ────────────────────────────────────────────────
STYLE = {
    "marker_size": 10,
    "marker_alpha": 0.85,
    "marker_fillstyle": "none",  # open circles
    "marker_edgewidth": 0.7,
    "capsize": 2,
    "marker_lw": 1.2,
    "model_lw": 1.8,
    "model_alpha": 0.6,
    "prior_lw": 1.4,
    "prior_alpha": 0.7,
    "prior_color": "0.45",
    "upper_limit_ms": 3,
    "upper_limit_alpha": 0.4,
    "legend_handle_lw": 2,
    "grid_alpha": 0.15,
}

# ── Layout ───────────────────────────────────────────────────────────────
LAYOUT = {
    "panel_width": 3.2,  # inches per column
    "panel_height": 2.8,  # inches per row
    "fig_pad_w": 1.0,  # extra width (inches)
    "fig_pad_h": 0.8,  # extra height (inches)
    "hspace": 0.04,
    "wspace": 0.04,
    "col_header_pad": 4,
    "row_label_pad": 18,
    "suptitle_y": 1.01,
    "lam_model_pad_lo": 0.3,  # model lam grid extends to lam_min x this
    "lam_model_pad_hi": 1.5,  # model lam grid extends to lam_max x this
    "xlim_pad_lo": 0.5,  # x-axis extends to lam_min x this
    "xlim_pad_hi": 1.5,  # x-axis extends to lam_max x this
    "n_model_points": 120,
}

# ── Colorbar ─────────────────────────────────────────────────────────────
CBAR = {
    "fraction": 0.02,
    "pad": 0.02,
    "aspect": 30,
    "right": 0.88,  # subplots_adjust right when cbar present
}

# ── Type colours for population splits ───────────────────────────────────
TYPE_COLORS = {
    "sfg": "#1f77b4",  # blue (star-forming)
    "quiescent": "#d62728",  # red  (quiescent)
    "agn": "#2ca02c",  # green
    "dusty": "#ff7f0e",  # orange
    "split_0": "#1f77b4",  # blue  (complete SF)
    "split_1": "#ff7f0e",  # orange (incomplete SF)
    "split_2": "#d62728",  # red   (quiescent)
    "split_3": "#2ca02c",  # green (starburst)
}
_EXTRA_COLORS = [
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

# Type-specific sequential cmaps for extra-dimension gradients
_TYPE_CMAPS = {
    "sfg": cm.winter,
    "quiescent": cm.autumn,
    "agn": cm.Greens,
    "dusty": cm.Oranges,
    "split_0": cm.winter,
    "split_1": cm.Wistia,
    "split_2": cm.autumn,
    "split_3": cm.summer,
}

# Short display names for type labels
_TYPE_SHORT = {
    "split_0": "cSF",
    "split_1": "iSF",
    "split_2": "QT",
    "split_3": "SB",
    "sfg": "SF",
    "quiescent": "Q",
    "agn": "AGN",
}

# Dimension label pretty-printing
_DIM_LABELS = {
    "redshift": "z",
    "stellar_mass": "log M★",
    "sigma_sfr": "log σ_SFR",
    "beta_uv": "β_UV",
    "l_uv": "log L_UV",
    "l_nuv": "log L_NUV",
}

# Grid dimension auto-detection priority (first match → columns)
_DIM_PRIORITY = ["redshift", "l_uv", "l_nuv", "beta_uv", "stellar_mass", "sigma_sfr"]


# ---------------------------------------------------------------------------
# Population ID parsing helpers
# ---------------------------------------------------------------------------
def _extract_pop_type(pop_id: str) -> str:
    """
    Extract population type from ID.  Handles both prefix and suffix
    styles that PopulationManager can produce:

    Prefix:  'sfg__stellar_mass_9.5_10.0__redshift_0.5_1.0'      → 'sfg'
    Suffix:  'redshift_0.01_0.5__stellar_mass_8.5_10.0__split_0'  → 'split_0'
    Neither: 'stellar_mass_9.5_10.0__redshift_0.5_1.0'            → '_all_'
    """
    segments = pop_id.split("__")

    # Check first segment (prefix style: sfg__...)
    first = segments[0]
    if not any(c.isdigit() for c in first):
        return first

    # Check last segment (suffix style: ...__split_0)
    if len(segments) > 1:
        last = segments[-1]
        parts = last.split("_")
        # A bin-range segment has >=3 parts ending in two floats
        # e.g. "stellar_mass_8.5_10.0"  →  skip
        # A type-label segment does not, e.g. "split_0"
        if len(parts) >= 3:
            try:
                float(parts[-2])
                float(parts[-1])
                # Successfully parsed as bin range → not a type label
                return "_all_"
            except ValueError:
                return last
        else:
            return last

    return "_all_"


def _parse_bins(pop_id: str) -> dict[str, tuple[float, float]]:
    """
    'sfg__stellar_mass_9.5_10.0__redshift_0.5_1.0'
    → {'stellar_mass': (9.5, 10.0), 'redshift': (0.5, 1.0)}
    """
    bins = {}
    for segment in pop_id.split("__"):
        parts = segment.split("_")
        if len(parts) >= 3:
            try:
                lo, hi = float(parts[-2]), float(parts[-1])
                name = "_".join(parts[:-2])
                bins[name] = (lo, hi)
            except ValueError:
                continue
    return bins


def _dim_label(dim: str) -> str:
    """Pretty-print a dimension name."""
    return _DIM_LABELS.get(dim, dim.replace("_", " "))


def _solve_amplitude_at_T(greybody_fitter, wave_rest, fluxes, errors, T_rest, beta):
    """
    Analytically solve for log10(A) at fixed T and beta.

    Returns (log10_A, success_bool).
    """
    template = greybody_fitter.greybody_model(
        wave_rest,
        0.0,
        T_rest,
        beta,
    )
    w = 1.0 / errors**2
    denom = np.sum(template**2 * w)
    if denom <= 0:
        return -35.0, False
    x_opt = np.sum(fluxes * template * w) / denom
    if x_opt > 0:
        return np.log10(x_opt), True
    return -35.0, False


def _set_pah_state(fitter, z, log_stellar_mass):
    """Set PAH state on fitter for the current population."""
    if hasattr(fitter, "_pah_z"):
        fitter._pah_z = z
    if hasattr(fitter, "_pah_log_stellar_mass"):
        fitter._pah_log_stellar_mass = log_stellar_mass


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------
def plot_sed_grid(
    wrapper,
    *,
    col_dim: str | None = None,
    row_dim: str | None = None,
    split_filter: list[int] | None = None,
    show_errors: bool = True,
    show_model: bool = True,
    show_prior: bool = True,
    show_tier: bool = False,
    flux_unit: str = "mJy",
    ylim: tuple[float, float] = (1e-2, 5e2),
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    fontsize_legend: int | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Create a grid plot of SEDs from SimstackWrapper results.

    Parameters
    ----------
    wrapper : SimstackWrapper
        Must have ``stacking_results`` and ``processed_results``.
    col_dim, row_dim : str, optional
        Binning dimension names for columns / rows.
        Auto-detected if None (prefers redshift → columns).
    split_filter : list of int, optional
        Which split classes to include, e.g. [0] for complete_sfg only,
        [0, 1] for both SF classes, [0, 3] for SF + starbursts.
        None = show all splits.
    show_errors : bool
        Plot error bars on data points.
    show_model : bool
        Overlay the fitted greybody curve (coloured to match data).
    show_prior : bool
        Overlay the temperature-prior SED (grey dashed) with
        amplitude scaled to best-fit the data.
    show_tier : bool
        Append the fit quality tier (A/B/C) to each legend entry.
    flux_unit : {'mJy', 'Jy'}
        Flux-density unit for the y-axis.
    ylim : (float, float)
        y-axis limits (in flux_unit).
    figsize : (float, float), optional
        Figure size; auto-sized if None.
    save_path : str or Path, optional
        Save the figure to this path.
    dpi : int
        Resolution for saved raster output.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # ── validate inputs ───────────────────────────────────────────────────
    sr = getattr(wrapper, "stacking_results", None)
    pr = getattr(wrapper, "processed_results", None)
    if sr is None or pr is None:
        raise ValueError("wrapper must have stacking_results and processed_results")

    sed_results = pr.sed_results
    if not sed_results:
        raise ValueError("No SED results in processed_results")

    flux_scale = 1e3 if flux_unit == "mJy" else 1.0
    flux_label = f"Flux Density ({flux_unit})"

    # ── wavelengths from config ───────────────────────────────────────────
    wavelengths = np.array([wrapper.config.maps[m].wavelength for m in sr.map_names])
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    map_names_sorted = [sr.map_names[i] for i in sort_idx]

    print(f"📡 Wavelengths: {wavelengths} µm  ({len(wavelengths)} bands)")

    # ── population label → stacking array index ──────────────────────────
    pop_label_to_idx = {label: i for i, label in enumerate(sr.population_labels)}

    # ── parse every population ID ────────────────────────────────────────
    pop_ids = list(sed_results.keys())

    # Apply split filter if specified
    if split_filter is not None:
        allowed = {f"split_{i}" for i in split_filter}
        pop_ids = [
            pid
            for pid in pop_ids
            if _extract_pop_type(pid) in allowed or _extract_pop_type(pid) == "_all_"
        ]
        print(f"Split filter: keeping {allowed} → {len(pop_ids)} populations")

    parsed = {pid: _parse_bins(pid) for pid in pop_ids}
    types = {pid: _extract_pop_type(pid) for pid in pop_ids}
    unique_types = sorted(set(types.values()))
    has_type_split = unique_types != ["_all_"]

    if has_type_split:
        type_counts = {
            t: sum(1 for v in types.values() if v == t) for t in unique_types
        }
        print(
            f"🏷️  Population types: "
            f"{', '.join(f'{t}: {n}' for t, n in type_counts.items())}"
        )
    else:
        print(f"📊 {len(pop_ids)} populations (no type split)")

    # ── discover binning dimensions ──────────────────────────────────────
    all_dims: set[str] = set()
    for b in parsed.values():
        all_dims.update(b.keys())

    print(f"🗂️  Binning dimensions: {sorted(all_dims)}")

    if len(all_dims) < 2:
        print("⚠️  < 2 binning dimensions — falling back to simple plot")
        return _plot_sed_simple(
            wrapper,
            sr,
            pr,
            wavelengths,
            map_names_sorted,
            pop_label_to_idx,
            flux_scale,
            flux_label,
            show_errors,
            show_model,
            show_prior,
            ylim,
        )

    if col_dim is None or row_dim is None:
        ranked = sorted(
            all_dims,
            key=lambda d: (_DIM_PRIORITY.index(d) if d in _DIM_PRIORITY else 99),
        )
        if col_dim is None:
            col_dim = ranked[0]
        if row_dim is None:
            row_dim = [d for d in ranked if d != col_dim][0]

    print(f"📐 Grid: {col_dim} (columns) x {row_dim} (rows)")

    # ── unique bins per axis ─────────────────────────────────────────────
    def _unique_bins(dim):
        vals = set()
        for b in parsed.values():
            if dim in b:
                vals.add(b[dim])
        return sorted(vals)

    col_bins = _unique_bins(col_dim)
    row_bins = _unique_bins(row_dim)
    n_cols, n_rows = len(col_bins), len(row_bins)

    if not col_bins or not row_bins:
        raise ValueError(f"No bins found for {col_dim} or {row_dim}")

    print(f"📊 Grid size: {n_cols} x {n_rows} = {n_cols * n_rows} panels")

    # ── extra dimensions (anything beyond col & row → colour) ────────────
    extra_dims = sorted(all_dims - {col_dim, row_dim})
    if extra_dims:
        print(f"🎨 Colour dimension(s): {extra_dims}")

    def _colour_key(pid):
        ptype = types[pid]
        extras = tuple(parsed[pid].get(d, (None, None)) for d in extra_dims)
        return (ptype, extras)

    colour_keys = sorted(set(_colour_key(p) for p in pop_ids))

    # ── build colour map ─────────────────────────────────────────────────
    colour_map: dict = {}

    if has_type_split and not extra_dims:
        for i, t in enumerate(unique_types):
            colour_map[(t, ())] = TYPE_COLORS.get(
                t, _EXTRA_COLORS[i % len(_EXTRA_COLORS)]
            )
    elif has_type_split and extra_dims:
        for t in unique_types:
            keys_t = sorted(k for k in colour_keys if k[0] == t)
            n = max(len(keys_t), 1)
            cmap = _TYPE_CMAPS.get(t, cm.Purples)
            for j, k in enumerate(keys_t):
                colour_map[k] = cmap(0.3 + 0.6 * j / max(n - 1, 1))
    else:
        n = max(len(colour_keys), 1)
        for j, k in enumerate(colour_keys):
            colour_map[k] = cm.viridis(0.1 + 0.85 * j / max(n - 1, 1))

    # ── create figure ────────────────────────────────────────────────────
    if figsize is None:
        figsize = (
            n_cols * LAYOUT["panel_width"] + LAYOUT["fig_pad_w"],
            n_rows * LAYOUT["panel_height"] + LAYOUT["fig_pad_h"],
        )

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    # fine lam grid for model curves (observed frame)
    lam_obs_model = np.logspace(
        np.log10(wavelengths.min() * LAYOUT["lam_model_pad_lo"]),
        np.log10(wavelengths.max() * LAYOUT["lam_model_pad_hi"]),
        LAYOUT["n_model_points"],
    )

    # ── fill panels ──────────────────────────────────────────────────────
    for ri, rbin in enumerate(row_bins):
        for ci, cbin in enumerate(col_bins):
            ax = axes[ri, ci]

            cell_pops = [
                pid
                for pid in pop_ids
                if parsed[pid].get(col_dim) == cbin and parsed[pid].get(row_dim) == rbin
            ]

            if not cell_pops:
                ax.text(
                    0.5,
                    0.5,
                    "—",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=FONT["empty_cell"],
                    color="0.6",
                )
            else:
                legend_handles, legend_labels = [], []
                prior_plotted = False  # one prior per panel is enough

                for pid in cell_pops:
                    sed = sed_results[pid]
                    idx = pop_label_to_idx.get(pid)
                    if idx is None:
                        continue

                    colour = colour_map.get(_colour_key(pid), "0.4")

                    # ── data points ──────────────────────────────────
                    fluxes = (
                        np.array([sr.flux_densities[m][idx] for m in map_names_sorted])
                        * flux_scale
                    )
                    errors = (
                        np.array([sr.flux_errors[m][idx] for m in map_names_sorted])
                        * flux_scale
                    )

                    det = fluxes > 0

                    if np.any(det):
                        if show_errors:
                            ax.errorbar(
                                wavelengths[det],
                                fluxes[det],
                                yerr=errors[det],
                                fmt="o",
                                color=colour,
                                ms=STYLE["marker_size"],
                                mfc="none",
                                mew=STYLE["marker_edgewidth"],
                                capsize=STYLE["capsize"],
                                lw=STYLE["marker_lw"],
                                alpha=STYLE["marker_alpha"],
                                zorder=3,
                            )
                        else:
                            ax.plot(
                                wavelengths[det],
                                fluxes[det],
                                "o",
                                color=colour,
                                ms=STYLE["marker_size"],
                                mfc="none",
                                mew=STYLE["marker_edgewidth"],
                                alpha=STYLE["marker_alpha"],
                                zorder=3,
                            )

                    # ── upper limits ─────────────────────────────────
                    nondet = (~det) & (errors > 0)
                    if np.any(nondet):
                        ax.plot(
                            wavelengths[nondet],
                            2 * errors[nondet],
                            "v",
                            color=colour,
                            ms=STYLE["upper_limit_ms"],
                            alpha=STYLE["upper_limit_alpha"],
                        )

                    # ── fitted model curve ───────────────────────────
                    if (
                        show_model
                        and sed.amplitude is not None
                        and sed.dust_temperature_rest_frame is not None
                        and np.isfinite(sed.dust_temperature_rest_frame)
                    ):
                        z = sed.median_redshift
                        beta = sed.emissivity_index or 1.8
                        try:
                            lam_rest = lam_obs_model / (1.0 + z)
                            _set_pah_state(pr.greybody_fitter, z, sed.median_mass)
                            model_flux = (
                                pr.greybody_fitter.greybody_model(
                                    lam_rest,
                                    sed.amplitude,
                                    sed.dust_temperature_rest_frame,
                                    beta,
                                )
                                * flux_scale
                            )
                            ax.plot(
                                lam_obs_model,
                                model_flux,
                                "-",
                                color=colour,
                                lw=STYLE["model_lw"],
                                alpha=STYLE["model_alpha"],
                                zorder=2,
                            )
                        except Exception as e:
                            logger.debug(f"Model failed for {pid}: {e}")

                    # ── Temperature-prior overlay ──────────────────────
                    if show_prior and not prior_plotted:
                        z = sed.median_redshift
                        beta = sed.emissivity_index or 1.8
                        try:
                            T_prior, _ = pr.greybody_fitter.temperature_prior_relation(
                                z
                            )

                            # Solve amplitude analytically at T_prior
                            # using this population's data (in Jy)
                            fluxes_jy = np.array(
                                [sr.flux_densities[m][idx] for m in map_names_sorted]
                            )
                            errors_jy = np.array(
                                [sr.flux_errors[m][idx] for m in map_names_sorted]
                            )
                            wave_rest_data = wavelengths / (1.0 + z)

                            # Only use positive detections for scaling
                            pos = fluxes_jy > 0
                            if np.sum(pos) >= 1:
                                A_prior, ok = _solve_amplitude_at_T(
                                    pr.greybody_fitter,
                                    wave_rest_data[pos],
                                    fluxes_jy[pos],
                                    errors_jy[pos],
                                    T_prior,
                                    beta,
                                )

                                if ok:
                                    lam_rest = lam_obs_model / (1.0 + z)
                                    _set_pah_state(
                                        pr.greybody_fitter, z, sed.median_mass
                                    )
                                    prior_flux = (
                                        pr.greybody_fitter.greybody_model(
                                            lam_rest,
                                            A_prior,
                                            T_prior,
                                            beta,
                                        )
                                        * flux_scale
                                    )
                                    ax.plot(
                                        lam_obs_model,
                                        prior_flux,
                                        "--",
                                        color=STYLE["prior_color"],
                                        lw=STYLE["prior_lw"],
                                        alpha=STYLE["prior_alpha"],
                                        zorder=1,
                                    )
                                    prior_plotted = True
                        except Exception as e:
                            logger.debug(f"Prior overlay failed: {e}")

                    # ── legend entry ─────────────────────────────────
                    T = sed.dust_temperature_rest_frame

                    # Build informative label with physical quantities
                    parts = []

                    # L_IR (from derived quantities, not SEDResults)
                    derived = pr.derived_quantities.get(pid)
                    lir = derived.total_ir_luminosity if derived else 0
                    if lir is not None and lir > 0:
                        parts.append(f"L_IR={np.log10(lir):.1f}")

                    # T_rf
                    if T is not None and np.isfinite(T):
                        parts.append(f"T_rf={T:.0f}K")

                    # Tier (optional)
                    if show_tier:
                        tier = sed.fit_quality_tier or "?"
                        parts.append(tier)

                    label = " ".join(parts) if parts else "—"

                    legend_handles.append(
                        Line2D([0], [0], color=colour, lw=STYLE["legend_handle_lw"])
                    )
                    legend_labels.append(label)

                # Add prior to legend once
                if show_prior and prior_plotted:
                    # Retrieve the prior T for the label
                    _first_sed = sed_results[cell_pops[0]]
                    _T_prior, _ = pr.greybody_fitter.temperature_prior_relation(
                        _first_sed.median_redshift
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=STYLE["prior_color"],
                            lw=STYLE["prior_lw"],
                            ls="--",
                        )
                    )
                    _pname = getattr(pr.greybody_fitter, "temperature_prior", "prior")
                    legend_labels.append(f"{_pname} T_rf={_T_prior:.0f}K")

                if legend_handles:
                    if fontsize_legend is None:
                        fontsize = FONT["legend"]
                    else:
                        fontsize = fontsize_legend
                    ax.legend(
                        legend_handles,
                        legend_labels,
                        loc="upper left",
                        fontsize=fontsize,
                        framealpha=0.7,
                        edgecolor="0.8",
                        handlelength=1.2,
                        borderpad=0.3,
                        labelspacing=0.3,
                    )

            # ── axes formatting ──────────────────────────────────────
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(ylim)
            ax.set_xlim(
                wavelengths.min() * LAYOUT["xlim_pad_lo"],
                wavelengths.max() * LAYOUT["xlim_pad_hi"],
            )
            ax.grid(True, alpha=STYLE["grid_alpha"], which="both")
            ax.tick_params(
                labelsize=FONT["tick"],
                which="both",
                direction="in",
            )

            if ri == 0:
                lo, hi = cbin
                ax.set_title(
                    f"{_dim_label(col_dim)}: {lo:.1f}–{hi:.1f}",
                    fontsize=FONT["col_header"],
                    pad=LAYOUT["col_header_pad"],
                )

            if ci == n_cols - 1:
                lo, hi = rbin
                ax2 = ax.twinx()
                ax2.set_ylabel(
                    f"{_dim_label(row_dim)}: {lo:.1f}–{hi:.1f}",
                    rotation=270,
                    labelpad=LAYOUT["row_label_pad"],
                    fontsize=FONT["row_label"],
                )
                ax2.set_yticks([])

            if ri == n_rows - 1:
                ax.set_xlabel("Observed lam (µm)", fontsize=FONT["axis_label"])
            if ci == 0:
                ax.set_ylabel(flux_label, fontsize=FONT["axis_label"])

    # ── title & layout ───────────────────────────────────────────────────
    n_pops = len(pop_ids)
    title = f"SED Grid — {n_pops} populations"
    if has_type_split:
        counts = {t: sum(1 for v in types.values() if v == t) for t in unique_types}
        title += (
            "  ("
            + ", ".join(f"{_TYPE_SHORT.get(t, t)}: {n}" for t, n in counts.items())
            + ")"
        )
    fig.suptitle(title, fontsize=FONT["suptitle"], y=LAYOUT["suptitle_y"])

    fig.tight_layout()

    # ── colorbar for extra-dimension gradient ────────────────────────────
    if extra_dims:
        from matplotlib.colors import Normalize

        ed = extra_dims[0]
        ed_bins = sorted(
            set(
                parsed[pid].get(ed)
                for pid in pop_ids
                if parsed[pid].get(ed) is not None
            )
        )

        if len(ed_bins) >= 2:
            mids = [0.5 * (lo + hi) for lo, hi in ed_bins]
            vmin, vmax = mids[0], mids[-1]

            if has_type_split:
                first_type = unique_types[0]
                cbar_cmap = _TYPE_CMAPS.get(first_type, cm.viridis)
            else:
                cbar_cmap = cm.viridis

            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = plt.cm.ScalarMappable(cmap=cbar_cmap, norm=norm)
            sm.set_array([])

            plt.subplots_adjust(
                hspace=LAYOUT["hspace"],
                wspace=LAYOUT["wspace"],
                right=CBAR["right"],
            )
            cbar = fig.colorbar(
                sm,
                ax=axes,
                fraction=CBAR["fraction"],
                pad=CBAR["pad"],
                aspect=CBAR["aspect"],
            )
            cbar.set_label(_dim_label(ed), fontsize=FONT["colorbar"])
            cbar.ax.tick_params(labelsize=FONT["colorbar_tick"])
            cbar.set_ticks(mids)
            cbar.set_ticklabels([f"{m:.1f}" for m in mids])
        else:
            plt.subplots_adjust(
                hspace=LAYOUT["hspace"],
                wspace=LAYOUT["wspace"],
            )
    else:
        plt.subplots_adjust(
            hspace=LAYOUT["hspace"],
            wspace=LAYOUT["wspace"],
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )
        print(f"💾 Saved to {save_path}")

    return fig


def plot_sed_grid_sigma_sfr_rows(
    wrapper,
    *,
    split_filter: list[int] | None = None,
    **kwargs,
) -> plt.Figure:
    """
    SED grid with σ_SFR bins as rows, redshift as columns, and a stellar-mass
    colorbar — the complement of the default plot_sed_grid layout.

    All keyword arguments are forwarded to plot_sed_grid.
    """
    return plot_sed_grid(
        wrapper,
        col_dim="redshift",
        row_dim="sigma_sfr",
        split_filter=split_filter,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Fallback: simple single-panel plot
# ---------------------------------------------------------------------------
def _plot_sed_simple(
    wrapper,
    sr,
    pr,
    wavelengths,
    map_names,
    pop_label_to_idx,
    flux_scale,
    flux_label,
    show_errors,
    show_model,
    show_prior,
    ylim,
) -> plt.Figure:
    """Single-panel SED plot when < 2 binning dimensions."""
    fig, ax = plt.subplots(figsize=(9, 5))
    pop_ids = list(pr.sed_results.keys())
    n = len(pop_ids)
    colors = cm.tab10(np.linspace(0, 1, min(n, 10)))

    lam_obs_model = np.logspace(
        np.log10(wavelengths.min() * LAYOUT["lam_model_pad_lo"]),
        np.log10(wavelengths.max() * LAYOUT["lam_model_pad_hi"]),
        LAYOUT["n_model_points"],
    )

    for k, pid in enumerate(pop_ids):
        sed = pr.sed_results[pid]
        idx = pop_label_to_idx.get(pid)
        if idx is None:
            continue

        c = colors[k % len(colors)]

        fluxes = np.array([sr.flux_densities[m][idx] for m in map_names]) * flux_scale
        errors = np.array([sr.flux_errors[m][idx] for m in map_names]) * flux_scale

        det = fluxes > 0
        label = f"{pid.split('__')[0]}… N={sed.n_sources}"

        if np.any(det):
            if show_errors:
                ax.errorbar(
                    wavelengths[det],
                    fluxes[det],
                    yerr=errors[det],
                    fmt="o",
                    color=c,
                    ms=STYLE["marker_size"],
                    mfc="none",
                    mew=STYLE["marker_edgewidth"],
                    capsize=STYLE["capsize"] + 1,
                    lw=STYLE["marker_lw"] + 0.3,
                    alpha=STYLE["marker_alpha"],
                    label=label,
                )
            else:
                ax.plot(
                    wavelengths[det],
                    fluxes[det],
                    "o",
                    color=c,
                    ms=STYLE["marker_size"],
                    mfc="none",
                    mew=STYLE["marker_edgewidth"],
                    alpha=STYLE["marker_alpha"],
                    label=label,
                )

            # Fitted model
            if (
                show_model
                and sed.amplitude is not None
                and sed.dust_temperature_rest_frame is not None
                and np.isfinite(sed.dust_temperature_rest_frame)
            ):
                z = sed.median_redshift
                beta = sed.emissivity_index or 1.8
                try:
                    lam_rest = lam_obs_model / (1.0 + z)
                    _set_pah_state(pr.greybody_fitter, z, sed.median_mass)
                    mf = (
                        pr.greybody_fitter.greybody_model(
                            lam_rest,
                            sed.amplitude,
                            sed.dust_temperature_rest_frame,
                            beta,
                        )
                        * flux_scale
                    )
                    ax.plot(
                        lam_obs_model,
                        mf,
                        "-",
                        color=c,
                        lw=STYLE["model_lw"],
                        alpha=STYLE["model_alpha"],
                    )
                except Exception:
                    pass

            # Prior overlay
            if show_prior:
                z = sed.median_redshift
                beta = sed.emissivity_index or 1.8
                try:
                    T_prior, _ = pr.greybody_fitter.temperature_prior_relation(z)
                    fluxes_jy = np.array([sr.flux_densities[m][idx] for m in map_names])
                    errors_jy = np.array([sr.flux_errors[m][idx] for m in map_names])
                    wave_rest_data = wavelengths / (1.0 + z)
                    pos = fluxes_jy > 0
                    if np.sum(pos) >= 1:
                        A_prior, ok = _solve_amplitude_at_T(
                            pr.greybody_fitter,
                            wave_rest_data[pos],
                            fluxes_jy[pos],
                            errors_jy[pos],
                            T_prior,
                            beta,
                        )
                        if ok:
                            lam_rest = lam_obs_model / (1.0 + z)
                            _set_pah_state(pr.greybody_fitter, z, sed.median_mass)
                            pf = (
                                pr.greybody_fitter.greybody_model(
                                    lam_rest,
                                    A_prior,
                                    T_prior,
                                    beta,
                                )
                                * flux_scale
                            )
                            ax.plot(
                                lam_obs_model,
                                pf,
                                "--",
                                color=c,
                                lw=STYLE["prior_lw"],
                                alpha=STYLE["prior_alpha"] - 0.3,
                            )
                except Exception:
                    pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ylim)
    ax.set_xlabel("Observed lam (µm)", fontsize=FONT["axis_label"])
    ax.set_ylabel(flux_label, fontsize=FONT["axis_label"])
    ax.tick_params(labelsize=FONT["tick"])
    ax.grid(True, alpha=STYLE["grid_alpha"] + 0.05)
    ax.legend(fontsize=FONT["legend"], bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    return fig


# =============================================================================
# ── T_RF VS REDSHIFT ──────────────────────────────────────────────────────


def create_trf_redshift_plot(
    wrapper,
    *,
    color_by="stellar_mass",
    show_errors=True,
    show_literature=True,
    add_qt=False,
    split_filter=None,
    fit_data=True,
    min_tier="A",
    figsize=(10, 8),
    save_path=None,
):
    """
    Create T_rest vs redshift plot from SimstackWrapper results.

    Parameters
    ----------
    wrapper : SimstackWrapper
        Must have processed_results with SED fits.
    color_by : str
        'stellar_mass', 'l_ir', 'l_uv', 'beta_uv', 'chi2', 'population_type'.
        Uses bin_properties medians when available, falls back to bin centers.
    show_errors : bool
        Show temperature error bars.
    show_literature : bool
        Show Schreiber+18, Viero+22, Drew & Casey 22 relations.
    add_qt : bool
        Include quiescent galaxies.
    fit_data : bool
        Fit and show quadratic T-z relation.
    min_tier : str
        Minimum fit quality tier ('A', 'B', 'C').
    figsize : tuple
        Figure size.
    save_path : str or Path, optional

    Returns
    -------
    fig, df_plot
    """
    import matplotlib.cm as mcm
    from matplotlib.colors import BoundaryNorm, ListedColormap

    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed SED results found")
        return None, None

    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

    # ── extract data ─────────────────────────────────────────────────
    data_list = []
    for pop_id, sed in pr.sed_results.items():
        if not sed.greybody_fit_success:
            continue
        if sed.dust_temperature_rest_frame is None:
            continue
        tier = getattr(sed, "fit_quality_tier", None) or "C"
        if tier_rank.get(tier, 2) > min_rank:
            continue

        row = {
            "population_id": pop_id,
            "redshift": sed.median_redshift,
            "T_rest_frame": sed.dust_temperature_rest_frame,
            "T_error": sed.dust_temperature_error,
            "n_sources": sed.n_sources,
            "chi2_reduced": sed.chi2_reduced,
            "fit_quality_tier": tier,
        }

        # Derived quantities
        derived = pr.derived_quantities.get(pop_id)
        if derived:
            row["l_ir"] = derived.total_ir_luminosity or 0
        else:
            row["l_ir"] = 0

        # Population type
        if "split_0" in pop_id:
            row["pop_type_code"] = 0  # complete_sfg
        elif "split_1" in pop_id:
            row["pop_type_code"] = 1  # incomplete_sfg
        elif "split_2" in pop_id:
            row["pop_type_code"] = 2  # quiescent
        elif "split_3" in pop_id:
            row["pop_type_code"] = 3  # starburst
        else:
            row["pop_type_code"] = -1

        # ── bin_properties (preferred: actual medians) ────────────
        props = sed.bin_properties
        if isinstance(props, str):
            import ast as _ast

            try:
                props = _ast.literal_eval(props)
            except (ValueError, SyntaxError):
                props = {}
        if not isinstance(props, dict):
            props = {}

        for key, val in props.items():
            kl = key.lower()
            if "mass" in kl:
                row["stellar_mass_prop"] = val
            elif "l_uv" in kl or "luv" in kl:
                row["l_uv_prop"] = val
            elif "beta" in kl:
                row["beta_uv_prop"] = val

        # ── bin centers (fallback) ────────────────────────────────
        bins = _parse_bins(pop_id)
        for dim, (lo, hi) in bins.items():
            dl = dim.lower()
            if "mass" in dl and "stellar_mass_prop" not in row:
                row["stellar_mass_bin"] = (lo + hi) / 2
            elif ("l_uv" in dl or "luv" in dl) and "l_uv_prop" not in row:
                row["l_uv_bin"] = (lo + hi) / 2
            elif "beta" in dl and "beta_uv_prop" not in row:
                row["beta_uv_bin"] = (lo + hi) / 2

        # Resolve: prefer prop over bin center
        row["stellar_mass"] = row.get(
            "stellar_mass_prop",
            row.get("stellar_mass_bin", getattr(sed, "median_mass", np.nan)),
        )
        row["l_uv"] = row.get("l_uv_prop", row.get("l_uv_bin", np.nan))
        row["beta_uv"] = row.get("beta_uv_prop", row.get("beta_uv_bin", np.nan))

        data_list.append(row)

    if not data_list:
        print("No valid temperature measurements found")
        return None, None

    df_plot = pd.DataFrame(data_list)

    # Filter by population split
    if split_filter is not None:
        allowed_codes = set(split_filter)
        n_pre = len(df_plot)
        df_plot = df_plot[df_plot["pop_type_code"].isin(allowed_codes)].copy()
        if len(df_plot) < n_pre:
            print(f"Split filter {list(allowed_codes)}: {n_pre} → {len(df_plot)}")
    elif not add_qt:
        # Default: skip quiescent (class 2)
        n_pre = len(df_plot)
        df_plot = df_plot[df_plot["pop_type_code"] != 2].copy()
        n_qt = n_pre - len(df_plot)
        if n_qt > 0:
            print(f"Filtered {n_qt} quiescent galaxies")

    if len(df_plot) == 0:
        print("No data after filtering")
        return None, None

    print(f"T-z plot: {len(df_plot)} populations (tier ≥ {min_tier})")
    print(f"  z: {df_plot['redshift'].min():.2f} - {df_plot['redshift'].max():.2f}")
    print(
        f"  T: {df_plot['T_rest_frame'].min():.0f} - {df_plot['T_rest_frame'].max():.0f} K"
    )

    # ── color setup ──────────────────────────────────────────────────
    color_configs = {
        "stellar_mass": ("stellar_mass", "log10(M*/M_sun)", "viridis"),
        "l_ir": ("l_ir", "log10(L_IR/L_sun)", "plasma"),
        "l_uv": ("l_uv", "log10(L_UV/L_sun)", "viridis"),
        "beta_uv": ("beta_uv", "β_UV", "coolwarm"),
        "chi2": ("chi2_reduced", "chi^2_red", "coolwarm"),
    }

    if color_by in color_configs:
        col_name, color_label, cmap_name = color_configs[color_by]
        if col_name == "l_ir":
            color_data = np.log10(df_plot[col_name].replace(0, np.nan)).values
        else:
            color_data = df_plot[col_name].values.astype(float)
    elif color_by == "population_type":
        color_data = df_plot["pop_type_code"].values.astype(float)
        color_label = "Population Type"
        cmap_name = "RdBu"
    else:
        color_data = None
        color_label = None
        cmap_name = "plasma"

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    if color_data is not None:
        valid_mask = np.isfinite(color_data)
    else:
        valid_mask = np.ones(len(df_plot), dtype=bool)

    if color_data is not None and np.any(valid_mask):
        valid_vals = color_data[valid_mask]

        # ── discrete colorbar ────────────────────────────────────
        if color_by == "stellar_mass":
            boundaries = np.array([8.5, 9.5, 10.0, 10.3, 10.6, 11.0, 12.0])
        elif color_by in ("l_ir", "l_uv"):
            boundaries = np.arange(
                np.floor(np.nanmin(valid_vals)),
                np.ceil(np.nanmax(valid_vals)) + 0.5,
                0.5,
            )
        elif color_by == "beta_uv":
            boundaries = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5])
        else:
            n_disc = min(6, len(np.unique(valid_vals)))
            boundaries = np.linspace(
                np.nanmin(valid_vals), np.nanmax(valid_vals), n_disc + 1
            )

        # Trim to data range
        boundaries = boundaries[
            (boundaries >= np.nanmin(valid_vals) - 0.5)
            & (boundaries <= np.nanmax(valid_vals) + 0.5)
        ]
        if len(boundaries) < 2:
            boundaries = np.linspace(np.nanmin(valid_vals), np.nanmax(valid_vals), 5)

        n_colors = max(len(boundaries) - 1, 1)
        base_cmap = mcm.get_cmap(cmap_name, n_colors)
        discrete_cmap = ListedColormap([base_cmap(i) for i in range(n_colors)])
        norm = BoundaryNorm(boundaries, discrete_cmap.N)
        bin_labels = [
            f"{boundaries[i]:.1f}-{boundaries[i+1]:.1f}"
            for i in range(len(boundaries) - 1)
        ]

        scatter = ax.scatter(
            df_plot["redshift"],
            df_plot["T_rest_frame"],
            c=color_data,
            cmap=discrete_cmap,
            norm=norm,
            s=80,
            alpha=0.85,
            edgecolors="k",
            linewidth=0.5,
            zorder=3,
        )

        if show_errors and "T_error" in df_plot.columns:
            ax.errorbar(
                df_plot["redshift"],
                df_plot["T_rest_frame"],
                yerr=df_plot["T_error"],
                fmt="none",
                color="gray",
                alpha=0.3,
                capsize=2,
                elinewidth=0.8,
                zorder=1,
            )

        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_label, fontsize=13)
        tick_positions = boundaries[:-1] + np.diff(boundaries) / 2
        cbar.set_ticks(tick_positions)
        cbar.set_ticklabels(bin_labels, fontsize=9)
        for b in boundaries[1:-1]:
            cbar.ax.axhline(b, color="white", linewidth=0.8)

    else:
        ax.scatter(
            df_plot["redshift"],
            df_plot["T_rest_frame"],
            s=80,
            alpha=0.85,
            edgecolors="k",
            linewidth=0.5,
            zorder=3,
        )
        if show_errors and "T_error" in df_plot.columns:
            ax.errorbar(
                df_plot["redshift"],
                df_plot["T_rest_frame"],
                yerr=df_plot["T_error"],
                fmt="none",
                color="gray",
                alpha=0.3,
                capsize=2,
                elinewidth=0.8,
                zorder=1,
            )

    # ── literature ───────────────────────────────────────────────────
    if show_literature:
        z_lit = np.linspace(0, 6, 100)
        z_ext = np.linspace(0, 10, 150)

        T_schreiber = 32.9 + 4.60 * (z_lit - 2.0)
        ax.plot(
            z_lit,
            T_schreiber,
            "--",
            color="green",
            linewidth=2,
            label="Schreiber+18: T=23.7+4.6z",
            alpha=0.8,
        )

        T_viero = 20.9 + 5.9 * z_ext + 0.5 * z_ext**2
        ax.plot(
            z_ext,
            T_viero,
            "--",
            color="purple",
            linewidth=2,
            label="Viero+22: T=20.9+5.9z+0.5z^2",
            alpha=0.8,
        )

        ax.axhline(
            32,
            color="gray",
            ls=":",
            lw=1,
            alpha=0.5,
            label="Drew & Casey 22: no evol. at fixed L_IR",
        )

    # ── fit data ─────────────────────────────────────────────────────
    if fit_data and len(df_plot) >= 3:
        try:
            from scipy.optimize import curve_fit

            z_data = df_plot["redshift"].values
            T_data = df_plot["T_rest_frame"].values

            T_errors_raw = df_plot.get("T_error")
            if T_errors_raw is not None:
                T_errors = T_errors_raw.values.copy()
                bad = ~np.isfinite(T_errors) | (T_errors <= 0)
                if np.all(bad):
                    T_errors = None
                elif np.any(bad):
                    T_errors[bad] = np.median(T_errors[~bad])
            else:
                T_errors = None

            def quadratic(z, a, b, c_coeff):
                return a + b * z + c_coeff * z**2

            sigma_kw = (
                dict(sigma=T_errors, absolute_sigma=True)
                if T_errors is not None
                else {}
            )
            popt, pcov = curve_fit(
                quadratic,
                z_data,
                T_data,
                p0=[25, 3, 0.5],
                bounds=([10, -5, -1], [50, 20, 5]),
                **sigma_kw,
            )
            a, b, c_coeff = popt
            a_err, b_err, c_err = np.sqrt(np.diag(pcov))

            T_pred = quadratic(z_data, *popt)
            ss_res = np.sum((T_data - T_pred) ** 2)
            ss_tot = np.sum((T_data - np.mean(T_data)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            z_fit = np.linspace(0, min(z_data.max() * 1.15, 11), 200)
            ax.plot(
                z_fit,
                quadratic(z_fit, *popt),
                "-",
                color="black",
                linewidth=2.5,
                alpha=0.85,
                label=f"This work: T={a:.0f}+{b:.1f}z+{c_coeff:.2f}z^2 (R^2={r2:.2f})",
            )

            print(
                f"Quadratic fit: T = {a:.1f}+-{a_err:.1f} + "
                f"{b:.1f}+-{b_err:.1f}*z + {c_coeff:.2f}+-{c_err:.2f}*z^2  "
                f"(R^2={r2:.3f})"
            )

        except Exception as e:
            print(f"Fit failed: {e}")

    # ── formatting ───────────────────────────────────────────────────
    ax.set_xlabel("Redshift", fontsize=14)
    ax.set_ylabel("Rest-frame Dust Temperature (K)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.2)

    ax.set_xlim(0, df_plot["redshift"].max() * 1.15)
    T_max = df_plot["T_rest_frame"].max()
    ax.set_ylim(max(df_plot["T_rest_frame"].min() * 0.8, 10), T_max * 1.2)

    title = f"T_dust vs Redshift ({len(df_plot)} populations, tier ≥ {min_tier})"
    ax.set_title(title, fontsize=15, pad=15)

    if show_literature or fit_data:
        ax.legend(loc="upper left", fontsize=10, framealpha=0.9)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path as _P

        save_path = _P(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 300 if save_path.suffix.lower() in [".png", ".jpg"] else 150
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

    return fig, df_plot


# ── IRX-beta reference curves ───────────────────────────────────────────────


def _meurer99_irx(beta):
    """Meurer+99 IRX-beta relation (starburst/MW-like)."""
    return 10 ** (0.4 * (4.43 + 1.99 * np.asarray(beta))) - 1


def _smc_irx(beta):
    """SMC-like IRX-beta relation (McLure+2018 parameterization, dA/dbeta=0.91)."""
    beta = np.asarray(beta)
    beta_int = -2.30
    BC_1600 = 1.51
    A1600 = 0.91 * (beta - beta_int)
    return (10 ** (0.4 * A1600) - 1) * BC_1600


# ── IRX-beta plot ───────────────────────────────────────────────────────────


def create_lir_luv_beta_plot(
    wrapper,
    *,
    color_by: str = "redshift",
    average_over: str | list[str] | None = None,
    weight_by: str | None = "n_sources",
    split_filter: list[int] | None = None,
    min_tier: str = "A",
    show_errors: bool = True,
    figsize: tuple = (10, 8),
    save_path=None,
):
    """
    Create L_IR/L_UV vs beta_UV plot from SimstackWrapper results.

    Parameters
    ----------
    wrapper : SimstackWrapper
        Must have stacking_results and processed_results.
    color_by : str
        'redshift' or 'l_uv'.
    average_over : str, list of str, or None
        Dimension(s) to average over, collapsing them.
        E.g. 'stellar_mass' averages over mass bins at fixed beta and z.
        'l_uv' averages over L_UV bins.
        ['stellar_mass', 'l_uv'] averages over both.
        None = no averaging (plot every population).
    weight_by : str or None
        Column to weight the average by.
        'n_sources' (default) weights by population size.
        'l_ir' weights by IR luminosity.
        None = unweighted mean.
    split_filter : list of int, optional
        Which split classes to include.
    min_tier : str
        Minimum fit quality tier ('A', 'B', 'C').
    show_errors : bool
        Show error bars (std or weighted std) when averaging.
    figsize : tuple
        Figure size.
    save_path : str or Path, optional

    Returns
    -------
    fig : matplotlib Figure
    """
    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed SED results found")
        return None

    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

    # ── extract data ─────────────────────────────────────────────────
    rows = []
    for pop_id, sed in pr.sed_results.items():
        if not sed.greybody_fit_success:
            continue
        tier = getattr(sed, "fit_quality_tier", None) or "C"
        if tier_rank.get(tier, 2) > min_rank:
            continue

        # Filter by population split class
        pop_type = _extract_pop_type(pop_id)
        if split_filter is not None:
            allowed = {f"split_{i}" for i in split_filter}
            if pop_type not in allowed and pop_type != "_all_":
                continue
        else:
            # Default: skip quiescent (split_2)
            if pop_type == "split_2":
                continue

        bins = _parse_bins(pop_id)
        beta_bin = None
        luv_val = None
        z_bin = None

        # ── bin_properties (preferred: actual medians) ────────────
        props = sed.bin_properties
        if isinstance(props, str):
            import ast

            try:
                props = ast.literal_eval(props)
            except (ValueError, SyntaxError):
                props = {}
        if not isinstance(props, dict):
            props = {}

        luv_source = None
        beta_source = None
        mass_val = None
        for key, val in props.items():
            kl = key.lower()
            if ("l_uv" in kl or "luv" in kl) and luv_val is None:
                luv_val = val
                luv_source = "bin_properties"
            elif "beta" in kl and beta_bin is None:
                beta_bin = val
                beta_source = "bin_properties"
            elif "mass" in kl and mass_val is None:
                mass_val = val

        # ── bin centers (fallback) ────────────────────────────────
        for dim, (lo, hi) in bins.items():
            dl = dim.lower()
            if "beta" in dl and beta_bin is None:
                beta_bin = (lo + hi) / 2
                beta_source = "bin_center"
            elif ("l_uv" in dl or "luv" in dl) and luv_val is None:
                luv_val = (lo + hi) / 2
                luv_source = "bin_center"
            elif ("redshift" in dl or dim == "z") and z_bin is None:
                z_bin = (lo + hi) / 2
            elif "mass" in dl and mass_val is None:
                mass_val = (lo + hi) / 2

        if beta_bin is None or luv_val is None:
            continue

        # L_IR from derived quantities
        derived = pr.derived_quantities.get(pop_id)
        l_ir = derived.total_ir_luminosity if derived else 0
        if l_ir <= 0:
            continue

        l_uv_linear = 10**luv_val
        irx = l_ir / l_uv_linear

        rows.append(
            {
                "beta_uv": beta_bin,
                "l_uv": luv_val,
                "redshift": z_bin if z_bin is not None else sed.median_redshift,
                "stellar_mass": mass_val if mass_val is not None else np.nan,
                "irx": irx,
                "l_ir": l_ir,
                "n_sources": sed.n_sources,
                "tier": tier,
                "_luv_source": luv_source,
                "_beta_source": beta_source,
                # Store bin edges for grouping
                **{f"_bin_{dim}": f"{lo}_{hi}" for dim, (lo, hi) in bins.items()},
            }
        )

    if not rows:
        # Diagnostic: what did we find?
        sample_id = list(pr.sed_results.keys())[0] if pr.sed_results else "N/A"
        sample_sed = list(pr.sed_results.values())[0] if pr.sed_results else None
        sample_bins = _parse_bins(sample_id) if sample_id != "N/A" else {}
        sample_props = (
            sample_sed.bin_properties
            if sample_sed and sample_sed.bin_properties
            else {}
        )
        print(f"No populations with both beta_UV and L_UV data")
        print(f"  Sample population ID: {sample_id}")
        print(f"  Parsed binning dims: {list(sample_bins.keys())}")
        print(
            f"  bin_properties keys: {list(sample_props.keys()) if isinstance(sample_props, dict) else type(sample_props).__name__}"
        )
        print(f"  Need: 'beta' in binning dims + 'l_uv' in bin_properties or binning")
        return None

    # Report data sources
    n_luv_props = sum(1 for r in rows if r.get("_luv_source") == "bin_properties")
    n_beta_props = sum(1 for r in rows if r.get("_beta_source") == "bin_properties")
    n_total = len(rows)
    print(
        f"  beta_UV source: {n_beta_props}/{n_total} from bin_properties, "
        f"{n_total - n_beta_props} from bin center"
    )
    print(
        f"  L_UV source: {n_luv_props}/{n_total} from bin_properties, "
        f"{n_total - n_luv_props} from bin center"
    )

    df = pd.DataFrame(rows)
    df = df.drop(columns=["_luv_source", "_beta_source"], errors="ignore")
    print(f"IRX-beta: {len(df)} populations (tier ≥ {min_tier})")

    # ── optional averaging over dimensions ───────────────────────────
    if average_over is not None:
        if isinstance(average_over, str):
            average_over = [average_over]

        # Identify grouping columns: all bin dimensions EXCEPT those
        # being averaged over.  Bin dimensions are stored as _bin_* cols.
        bin_cols = [c for c in df.columns if c.startswith("_bin_")]
        avg_dims = set(d.lower() for d in average_over)

        group_cols = []
        for bc in bin_cols:
            dim_name = bc.replace("_bin_", "").lower()
            if not any(ad in dim_name for ad in avg_dims):
                group_cols.append(bc)

        if not group_cols:
            # No bin dimensions left → average everything
            group_cols = ["_dummy"]
            df["_dummy"] = "all"

        print(f"  Averaging over: {average_over}")
        print(f"  Grouping by: {[c.replace('_bin_', '') for c in group_cols]}")

        def _weighted_agg(grp):
            w = (
                grp[weight_by].values
                if weight_by and weight_by in grp.columns
                else np.ones(len(grp))
            )
            w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
            w_sum = w.sum()

            result = {}
            for col in ["beta_uv", "l_uv", "redshift", "stellar_mass", "irx"]:
                if col in grp.columns:
                    vals = grp[col].values
                    valid = np.isfinite(vals)
                    if np.any(valid):
                        wv = w[valid]
                        result[col] = np.average(vals[valid], weights=wv)
                        if col == "irx":
                            # Weighted std for error bars
                            mean = result[col]
                            var = np.average((vals[valid] - mean) ** 2, weights=wv)
                            result["irx_std"] = np.sqrt(var)
                    else:
                        result[col] = np.nan
            result["n_pops"] = len(grp)
            result["n_sources"] = (
                grp["n_sources"].sum() if "n_sources" in grp.columns else len(grp)
            )
            return pd.Series(result)

        df_avg = (
            df.groupby(group_cols)
            .apply(_weighted_agg, include_groups=False)
            .reset_index(drop=True)
        )
        n_pre = len(df)
        df = df_avg
        df["irx_std"] = df["irx_std"].fillna(0)

        wt_label = f", weighted by {weight_by}" if weight_by else ""
        print(f"  {n_pre} → {len(df)} averaged points{wt_label}")

    # Clean up internal columns
    df = df.drop(
        columns=[c for c in df.columns if c.startswith("_bin_") or c == "_dummy"],
        errors="ignore",
    )

    # ── prepare plot arrays ──────────────────────────────────────────
    beta_vals = df["beta_uv"].values
    irx_vals = df["irx"].values
    irx_errs = df["irx_std"].values if "irx_std" in df.columns else None

    # Flexible color_by: check DataFrame columns
    _color_map = {
        "redshift": ("redshift", "Redshift", "plasma"),
        "l_uv": ("l_uv", "log L_UV/L☉", "viridis"),
        "l_uv_phot": ("l_uv", "log L_UV/L_sun (phot)", "viridis"),
        "log_l_uv_phot": ("l_uv", "log L_UV/L_sun (phot)", "viridis"),
        "beta_uv": ("beta_uv", "β_UV", "coolwarm"),
        "beta_uv_phot": ("beta_uv", "β_UV (phot)", "coolwarm"),
        "stellar_mass": ("stellar_mass", "log M*/M☉", "viridis"),
    }
    # Also accept column names directly (e.g. "beta_uv_phot")
    if color_by in _color_map:
        _cc, color_label, cmap_name = _color_map[color_by]
        color_vals = df[_cc].values if _cc in df.columns else df["redshift"].values
    elif color_by in df.columns:
        color_vals = df[color_by].values
        color_label = color_by
        cmap_name = "coolwarm" if "beta" in color_by else "viridis"
    else:
        print(f"  color_by='{color_by}' not found, falling back to redshift")
        print(f"  Available: {[c for c in df.columns if not c.startswith('_')]}")
        color_vals = df["redshift"].values
        color_label = "Redshift"
        cmap_name = "plasma"

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    valid = np.isfinite(color_vals)
    if np.any(valid):
        sc = ax.scatter(
            beta_vals,
            irx_vals,
            c=color_vals,
            cmap=cmap_name,
            s=80,
            alpha=0.85,
            edgecolors="k",
            linewidth=0.5,
            zorder=3,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(color_label, fontsize=13)

        if average_over and show_errors and irx_errs is not None:
            ax.errorbar(
                beta_vals,
                irx_vals,
                yerr=irx_errs,
                fmt="none",
                color="gray",
                alpha=0.4,
                capsize=2,
                zorder=1,
            )
    else:
        ax.scatter(
            beta_vals,
            irx_vals,
            s=80,
            alpha=0.85,
            edgecolors="k",
            linewidth=0.5,
            zorder=3,
        )

    # ── reference curves ─────────────────────────────────────────────
    beta_model = np.linspace(-2.5, 1.5, 100)
    ax.plot(
        beta_model,
        _meurer99_irx(beta_model),
        "k--",
        lw=2,
        alpha=0.7,
        label="Meurer+99 (MW-like)",
    )
    ax.plot(
        beta_model,
        _smc_irx(beta_model),
        "-.",
        color="brown",
        lw=2,
        alpha=0.7,
        label="SMC-like (McLure+18)",
    )

    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    ax.set_xlabel("β_UV", fontsize=14)
    ax.set_ylabel("IRX  (L_IR / L_UV)", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(-2.2, 1.2)
    ax.set_ylim(10 ** (-1.2), 10**3.5)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"IRX–beta  ({len(df)} populations, tier ≥ {min_tier})", fontsize=15)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path as _P

        save_path = _P(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 300 if save_path.suffix.lower() in [".png", ".jpg"] else 150
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

    return fig


# ── Schreiber+2015 Main Sequence ─────────────────────────────────────────


def _schreiber15_log_sfr(log_mass, z):
    """
    Schreiber+2015 star-forming main sequence (Eq 9, Table 2).

    Parameters
    ----------
    log_mass : float or array
        log10(M* / Msun)
    z : float
        Redshift.

    Returns
    -------
    log10(SFR / Msun/yr)
    """
    m0, a0, a1, m1, a2 = 0.5, 1.5, 0.3, 0.36, 2.5
    r = np.log10(1 + z)
    m = np.asarray(log_mass, dtype=float) - 9.0  # Schreiber uses m = log(M/10⁹)
    term = np.maximum(0, m - m1 - a2 * r)
    return m - m0 + a0 * r - a1 * term**2


def _sfr_to_lir(sfr):
    """Kennicutt+1998 (Chabrier IMF): L_IR = SFR / 1e-10 Lsun."""
    return np.asarray(sfr) / 1.0e-10


def _lir_to_sfr(lir):
    """Inverse Kennicutt: SFR = 1e-10 * L_IR."""
    return 1.0e-10 * np.asarray(lir)


# ── L_IR vs Stellar Mass plot ────────────────────────────────────────────


def create_lir_mass_plot(
    wrapper,
    *,
    quantity="lir",
    color_by="redshift",
    size_by="l_uv",
    edge_color_by=None,
    split_filter=None,
    min_tier="A",
    show_ms=True,
    show_errors=False,
    figsize=(10, 8),
    save_path=None,
):
    """
    Plot L_IR (or SFR) vs stellar mass, colored by redshift.

    Parameters
    ----------
    wrapper : SimstackWrapper
    quantity : str
        'lir' for L_IR vs M*, 'sfr' for SFR vs M* (same data, axis rescaled).
    color_by : str
        'redshift' (default), 'l_uv', 'beta_uv'.
    size_by : str or None
        'l_uv', 'beta_uv', 'n_sources', or None for uniform size.
    edge_color_by : str or None
        Optional second color dimension for marker edges.
        'beta_uv', 'l_uv', 'redshift', or None (black edges).
    split_filter : list of int, optional
        Which split classes to include, e.g. [0] for complete_sfg only,
        [0, 1] for both SF classes. None = show all except quiescent (split_2).
    min_tier : str
        Minimum fit quality tier ('A', 'B', 'C').
    show_ms : bool
        Overlay Schreiber+2015 main sequence at each redshift bin center.
    show_errors : bool
        Show L_IR error bars.
    figsize : tuple
    save_path : str or Path, optional

    Returns
    -------
    fig, df_plot
    """
    import matplotlib.cm as mcm
    from matplotlib.colors import BoundaryNorm, ListedColormap

    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed SED results found")
        return None, None

    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

    # ── extract data ─────────────────────────────────────────────────
    rows = []
    z_bin_edges = set()

    for pop_id, sed in pr.sed_results.items():
        if not sed.greybody_fit_success:
            continue
        tier = getattr(sed, "fit_quality_tier", None) or "C"
        if tier_rank.get(tier, 2) > min_rank:
            continue

        # Filter by population split class
        pop_type = _extract_pop_type(pop_id)
        if split_filter is not None:
            allowed = {f"split_{i}" for i in split_filter}
            if pop_type not in allowed and pop_type != "_all_":
                continue
        else:
            # Default: skip quiescent (split_2)
            if pop_type == "split_2":
                continue

        derived = pr.derived_quantities.get(pop_id)
        l_ir = derived.total_ir_luminosity if derived else 0
        l_ir_err = derived.total_ir_luminosity_error if derived else 0
        if l_ir <= 0:
            continue

        # ── bin_properties (preferred) ────────────────────────────
        props = sed.bin_properties
        if isinstance(props, str):
            import ast as _ast

            try:
                props = _ast.literal_eval(props)
            except (ValueError, SyntaxError):
                props = {}
        if not isinstance(props, dict):
            props = {}

        mass_val = None
        luv_val = None
        beta_val = None
        for key, val in props.items():
            kl = key.lower()
            if "mass" in kl and mass_val is None:
                mass_val = val
            elif ("l_uv" in kl or "luv" in kl) and luv_val is None:
                luv_val = val
            elif "beta" in kl and beta_val is None:
                beta_val = val

        # ── bin centers (fallback) ────────────────────────────────
        bins = _parse_bins(pop_id)
        z_val = None
        for dim, (lo, hi) in bins.items():
            dl = dim.lower()
            if "mass" in dl and mass_val is None:
                mass_val = (lo + hi) / 2
            elif ("l_uv" in dl or "luv" in dl) and luv_val is None:
                luv_val = (lo + hi) / 2
            elif "beta" in dl and beta_val is None:
                beta_val = (lo + hi) / 2
            elif "redshift" in dl or dim == "z":
                z_val = (lo + hi) / 2
                z_bin_edges.add((lo, hi))

        if z_val is None:
            z_val = sed.median_redshift
        if mass_val is None:
            mass_val = getattr(sed, "median_mass", np.nan)

        if not np.isfinite(mass_val) or mass_val < 7:
            continue

        rows.append(
            {
                "stellar_mass": mass_val,
                "l_ir": l_ir,
                "l_ir_err": l_ir_err or 0,
                "sfr": _lir_to_sfr(l_ir),
                "redshift": z_val,
                "l_uv": luv_val if luv_val is not None else np.nan,
                "beta_uv": beta_val if beta_val is not None else np.nan,
                "n_sources": sed.n_sources,
                "tier": tier,
            }
        )

    if not rows:
        print("No populations with L_IR and stellar mass")
        return None, None

    df = pd.DataFrame(rows)
    print(f"L_IR-M* plot: {len(df)} populations (tier ≥ {min_tier})")

    # ── y-axis quantity ──────────────────────────────────────────────
    if quantity == "sfr":
        y_col = "sfr"
        y_label = "SFR  (M☉ yr⁻¹)"
        y_err_col = None  # could add later
    else:
        y_col = "l_ir"
        y_label = "L_IR  (L☉)"
        y_err_col = "l_ir_err"

    # ── color setup (discrete, matching other plots) ─────────────────
    color_col_map = {
        "redshift": ("redshift", "Redshift", "plasma"),
        "l_uv": ("l_uv", "log10(L_UV/L_sun)", "viridis"),
        "beta_uv": ("beta_uv", "β_UV", "coolwarm"),
    }
    col_name, color_label, cmap_name = color_col_map.get(
        color_by, ("redshift", "Redshift", "plasma")
    )
    color_data = df[col_name].values.astype(float)
    valid_color = np.isfinite(color_data)

    # Discrete boundaries
    if color_by == "redshift":
        boundaries = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0])
    elif color_by == "l_uv":
        boundaries = np.arange(
            np.floor(np.nanmin(color_data[valid_color])),
            np.ceil(np.nanmax(color_data[valid_color])) + 0.5,
            0.5,
        )
    elif color_by == "beta_uv":
        boundaries = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5])
    else:
        boundaries = np.linspace(
            np.nanmin(color_data[valid_color]),
            np.nanmax(color_data[valid_color]),
            7,
        )

    # Trim to data
    if np.any(valid_color):
        boundaries = boundaries[
            (boundaries >= np.nanmin(color_data[valid_color]) - 0.5)
            & (boundaries <= np.nanmax(color_data[valid_color]) + 0.5)
        ]
    if len(boundaries) < 2:
        boundaries = np.linspace(
            np.nanmin(color_data[valid_color]),
            np.nanmax(color_data[valid_color]),
            5,
        )

    n_colors = max(len(boundaries) - 1, 1)
    base_cmap = mcm.get_cmap(cmap_name, n_colors)
    discrete_cmap = ListedColormap([base_cmap(i) for i in range(n_colors)])
    norm = BoundaryNorm(boundaries, discrete_cmap.N)
    bin_labels = [
        f"{boundaries[i]:.1f}–{boundaries[i+1]:.1f}" for i in range(len(boundaries) - 1)
    ]

    # ── size setup ───────────────────────────────────────────────────
    def _scale_sizes(data, s_min=30, s_max=200):
        valid = np.isfinite(data)
        if np.any(valid) and data[valid].max() > data[valid].min():
            return np.where(
                valid,
                s_min
                + (s_max - s_min)
                * (data - np.nanmin(data[valid]))
                / (np.nanmax(data[valid]) - np.nanmin(data[valid])),
                s_min,
            )
        return np.full_like(data, 60.0)

    size_label = None
    if size_by == "l_uv" and np.any(np.isfinite(df["l_uv"])):
        sizes = _scale_sizes(df["l_uv"].values.astype(float))
        size_label = "L_UV"
    elif size_by == "beta_uv" and np.any(np.isfinite(df["beta_uv"])):
        sizes = _scale_sizes(df["beta_uv"].values.astype(float))
        size_label = "β_UV"
    elif size_by == "n_sources":
        sizes = _scale_sizes(df["n_sources"].values.astype(float))
        size_label = "N_sources"
    else:
        sizes = 60
        size_label = None

    # ── edge color setup ─────────────────────────────────────────────
    edge_colors = "k"
    edge_cmap_obj = None
    edge_norm_obj = None
    edge_label = None
    edge_lw = 0.5

    if edge_color_by is not None:
        _edge_configs = {
            "beta_uv": (
                "beta_uv",
                "β_UV",
                "coolwarm",
                np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5]),
            ),
            "l_uv": ("l_uv", "log10(L_UV/L_sun)", "viridis", None),
            "redshift": (
                "redshift",
                "Redshift",
                "plasma",
                np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]),
            ),
        }
        if edge_color_by in _edge_configs:
            _ecol, edge_label, _ecmap, _ebounds = _edge_configs[edge_color_by]
            _edata = df[_ecol].values.astype(float)
            _evalid = np.isfinite(_edata)

            if np.any(_evalid):
                if _ebounds is None:
                    _ebounds = np.arange(
                        np.floor(np.nanmin(_edata[_evalid])),
                        np.ceil(np.nanmax(_edata[_evalid])) + 0.5,
                        0.5,
                    )
                _ebounds = _ebounds[
                    (_ebounds >= np.nanmin(_edata[_evalid]) - 0.5)
                    & (_ebounds <= np.nanmax(_edata[_evalid]) + 0.5)
                ]
                if len(_ebounds) >= 2:
                    _en = max(len(_ebounds) - 1, 1)
                    _ebase = mcm.get_cmap(_ecmap, _en)
                    edge_cmap_obj = ListedColormap([_ebase(i) for i in range(_en)])
                    edge_norm_obj = BoundaryNorm(_ebounds, edge_cmap_obj.N)
                    edge_colors = [edge_cmap_obj(edge_norm_obj(v)) for v in _edata]
                    edge_lw = 1.5

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        df["stellar_mass"],
        df[y_col],
        c=color_data,
        cmap=discrete_cmap,
        norm=norm,
        s=sizes,
        alpha=0.85,
        edgecolors=edge_colors,
        linewidth=edge_lw,
        zorder=3,
    )

    if show_errors and y_err_col and y_err_col in df.columns:
        y_err = df[y_err_col].values
        valid_err = np.isfinite(y_err) & (y_err > 0)
        if np.any(valid_err):
            ax.errorbar(
                df["stellar_mass"].values[valid_err],
                df[y_col].values[valid_err],
                yerr=y_err[valid_err],
                fmt="none",
                color="gray",
                alpha=0.3,
                capsize=2,
                elinewidth=0.8,
                zorder=1,
            )

    # Discrete colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label, fontsize=13)
    tick_positions = boundaries[:-1] + np.diff(boundaries) / 2
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(bin_labels, fontsize=9)
    for b in boundaries[1:-1]:
        cbar.ax.axhline(b, color="white", linewidth=0.8)

    # Edge color legend (if used) — save handle for later
    _edge_legend = None
    if edge_cmap_obj is not None and edge_norm_obj is not None and edge_label:
        _ebounds = edge_norm_obj.boundaries
        from matplotlib.lines import Line2D

        edge_handles = []
        for i in range(len(_ebounds) - 1):
            mid = (_ebounds[i] + _ebounds[i + 1]) / 2
            ec = edge_cmap_obj(edge_norm_obj(mid))
            edge_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="none",
                    markeredgecolor=ec,
                    markeredgewidth=2,
                    markerfacecolor="lightgray",
                    markersize=8,
                    label=f"{_ebounds[i]:.1f}–{_ebounds[i+1]:.1f}",
                )
            )
        _edge_legend = ax.legend(
            handles=edge_handles,
            title=f"Edge: {edge_label}",
            loc="lower left",
            fontsize=8,
            title_fontsize=9,
            framealpha=0.9,
            ncol=max(1, len(edge_handles) // 4),
        )

    # ── main sequence overlay ────────────────────────────────────────
    if show_ms:
        mass_grid = np.linspace(
            max(df["stellar_mass"].min() - 0.3, 8.5),
            min(df["stellar_mass"].max() + 0.3, 12.5),
            100,
        )

        # Get unique z bin centers
        z_centers = sorted(set(round((lo + hi) / 2, 2) for lo, hi in z_bin_edges))
        if not z_centers:
            z_centers = sorted(df["redshift"].unique())

        for z_c in z_centers:
            log_sfr_ms = _schreiber15_log_sfr(mass_grid, z_c)
            sfr_ms = 10**log_sfr_ms

            if quantity == "sfr":
                y_ms = sfr_ms
            else:
                y_ms = _sfr_to_lir(sfr_ms)

            # Color to match the redshift colorbar
            ms_color = discrete_cmap(norm(z_c))
            ax.plot(
                mass_grid,
                y_ms,
                "--",
                color=ms_color,
                linewidth=1.8,
                alpha=0.7,
                zorder=2,
                label=(
                    f"MS z={z_c:.1f}"
                    if z_c == z_centers[0] or z_c == z_centers[-1]
                    else None
                ),
            )

        # Add single legend entry for MS
        from matplotlib.lines import Line2D

        ms_handle = Line2D([0], [0], ls="--", color="gray", lw=1.8, alpha=0.7)
        ms_legend = ax.legend(
            [ms_handle],
            ["Schreiber+15 MS"],
            loc="lower right",
            fontsize=10,
            framealpha=0.9,
        )
        # If edge color legend exists, keep both via add_artist
        if _edge_legend is not None:
            ax.add_artist(_edge_legend)

    # ── size legend ──────────────────────────────────────────────────
    if size_label and not isinstance(sizes, (int, float)):
        ax.text(
            0.02,
            0.02,
            f"Marker size ~ {size_label}",
            transform=ax.transAxes,
            fontsize=9,
            alpha=0.7,
            verticalalignment="bottom",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    # ── formatting ───────────────────────────────────────────────────
    ax.set_xlabel("log10(M* / M_sun)", fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_yscale("log")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.2, which="both")

    ax.set_xlim(df["stellar_mass"].min() - 0.3, df["stellar_mass"].max() + 0.3)
    ax.set_ylim(df[y_col].min() * 0.3, df[y_col].max() * 3)

    qty_label = "SFR" if quantity == "sfr" else "L_IR"
    title = f"{qty_label} vs M* ({len(df)} populations, tier ≥ {min_tier})"
    ax.set_title(title, fontsize=15, pad=15)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path as _P

        save_path = _P(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 300 if save_path.suffix.lower() in [".png", ".jpg"] else 150
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

    return fig, df


# ── Gas mass scaling and DTG ─────────────────────────────────────────────


def _tacconi20_log_mu_gas(log_mass, z, delta_ms=1.0):
    """
    Tacconi+2020 molecular gas fraction scaling relation.

    log(mu_gas) = log(M_mol / M*) from PHIBSS2 (Tacconi+2020, ARA&A 58, 157).
    Eq 4 with best-fit coefficients from their Table 3 ("all methods").

    Parameters
    ----------
    log_mass : float or array
        log10(M* / M_sun)
    z : float or array
        Redshift.
    delta_ms : float or array
        SFR / SFR_MS (offset from main sequence; 1.0 = on the MS).

    Returns
    -------
    log10(M_mol / M*)
    """
    log_mass = np.asarray(log_mass, dtype=float)
    z = np.asarray(z, dtype=float)
    delta_ms = np.asarray(delta_ms, dtype=float)

    # Tacconi+2020 Table 3, "all methods" row
    A = 0.06
    B = -3.33  # curvature in (1+z)
    C = 0.66  # center of z curvature
    D = 0.53  # δMS slope
    E = -0.35  # M* slope
    F = 10.7  # M* pivot

    log_mu = (
        A
        + B * (np.log10(1 + z) - C) ** 2
        + D * np.log10(np.maximum(delta_ms, 0.01))
        + E * (log_mass - F)
    )

    return log_mu


def _estimate_gas_mass(log_mass, z, sfr, log_sfr_ms=None):
    """
    Estimate molecular gas mass from Tacconi+2020 scaling relation.

    Parameters
    ----------
    log_mass : float
        log10(M* / M_sun)
    z : float
        Redshift.
    sfr : float
        Star formation rate (M_sun/yr). Used to compute δMS.
    log_sfr_ms : float, optional
        log10(SFR_MS) at this mass and z. If None, computed from Schreiber+2015.

    Returns
    -------
    M_gas : float
        Molecular gas mass in M_sun.
    """
    if log_sfr_ms is None:
        log_sfr_ms = _schreiber15_log_sfr(log_mass, z)

    sfr_ms = 10**log_sfr_ms
    delta_ms = max(sfr / sfr_ms, 0.01) if sfr_ms > 0 else 1.0

    log_mu = _tacconi20_log_mu_gas(log_mass, z, delta_ms)
    m_gas = 10 ** (log_mu + log_mass)  # mu = M_gas/M*, so M_gas = mu x M*

    return m_gas


# ── T_dust vs DTG plot ───────────────────────────────────────────────────


def create_tdust_dtg_plot(
    wrapper,
    *,
    color_by="redshift",
    split_filter=None,
    min_tier="A",
    show_parente=True,
    figsize=(10, 8),
    save_path=None,
):
    """
    Plot T_dust vs dust-to-gas ratio (DTG), testing Parente+2026.

    DTG is estimated per population from:
        M_dust : greybody SED fit (from simstack)
        M_gas  : Tacconi+2020 scaling relation (from M*, z, SFR)
        DTG = M_dust / M_gas

    Parameters
    ----------
    wrapper : SimstackWrapper
    color_by : str
        'redshift', 'stellar_mass', 'l_ir', 'sigma_sfr'.
    min_tier : str
        Minimum fit quality tier.
    show_parente : bool
        Show Parente+2026 / Sommovigo+2022 radiative equilibrium model
        curves at constant Sigma_SFR (0.01, 0.1, 1, 10 M☉/yr/kpc²).
    figsize : tuple
    save_path : str or Path, optional

    Returns
    -------
    fig, df_plot
    """
    import matplotlib.cm as mcm
    from matplotlib.colors import BoundaryNorm, ListedColormap

    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed SED results found")
        return None, None

    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

    # ── extract data ─────────────────────────────────────────────────
    rows = []
    for pop_id, sed in pr.sed_results.items():
        if not sed.greybody_fit_success:
            continue
        if sed.dust_temperature_rest_frame is None:
            continue
        tier = getattr(sed, "fit_quality_tier", None) or "C"
        if tier_rank.get(tier, 2) > min_rank:
            continue

        # Filter by population split class
        pop_type = _extract_pop_type(pop_id)
        if split_filter is not None:
            allowed = {f"split_{i}" for i in split_filter}
            if pop_type not in allowed and pop_type != "_all_":
                continue
        else:
            if pop_type == "split_2":
                continue

        derived = pr.derived_quantities.get(pop_id)
        if not derived:
            continue

        m_dust = derived.dust_mass
        l_ir = derived.total_ir_luminosity or 0
        sfr = derived.star_formation_rate or 0
        if m_dust is None or m_dust <= 0 or l_ir <= 0:
            continue

        # Get M* and z — prefer bin_properties
        props = sed.bin_properties
        if isinstance(props, str):
            import ast as _ast

            try:
                props = _ast.literal_eval(props)
            except (ValueError, SyntaxError):
                props = {}
        if not isinstance(props, dict):
            props = {}

        mass_val = None
        luv_val = None
        beta_val = None
        _extra = {}
        for key, val in props.items():
            kl = key.lower()
            if "mass" in kl and "delta" not in kl and mass_val is None:
                mass_val = val
            elif ("l_uv" in kl or "luv" in kl) and luv_val is None:
                luv_val = val
            elif "beta" in kl and beta_val is None:
                beta_val = val
            # Extra columns for color_by
            if "metal" in kl:
                _extra["metallicity"] = val
            elif "ebv" in kl:
                _extra["ebv"] = val
            elif "sigma" in kl and "sfr" in kl:
                _extra["log_sigma_sfr"] = val
            elif "ssfr" in kl:
                _extra["log_ssfr"] = val
            elif "delta" in kl:
                _extra["log_delta_ms"] = val

        # Fallback to bin centers
        bins = _parse_bins(pop_id)
        z_val = None
        for dim, (lo, hi) in bins.items():
            dl = dim.lower()
            if "mass" in dl and mass_val is None:
                mass_val = (lo + hi) / 2
            elif "redshift" in dl or dim == "z":
                z_val = (lo + hi) / 2

        if z_val is None:
            z_val = sed.median_redshift
        if mass_val is None:
            mass_val = getattr(sed, "median_mass", np.nan)

        if not np.isfinite(mass_val) or not np.isfinite(z_val):
            continue

        # Estimate gas mass from Tacconi+2020
        m_gas = _estimate_gas_mass(mass_val, z_val, sfr)
        if m_gas <= 0 or not np.isfinite(m_gas):
            continue

        dtg = m_dust / m_gas

        rows.append(
            {
                "T_rest": sed.dust_temperature_rest_frame,
                "T_error": sed.dust_temperature_error,
                "log_dtg": np.log10(dtg),
                "dtg": dtg,
                "m_dust": m_dust,
                "m_gas": m_gas,
                "redshift": z_val,
                "stellar_mass": mass_val,
                "l_ir": l_ir,
                "sfr": sfr,
                "l_uv": luv_val if luv_val is not None else np.nan,
                "beta_uv": beta_val if beta_val is not None else np.nan,
                "n_sources": sed.n_sources,
                "tier": tier,
                **{k: v for k, v in _extra.items()},
            }
        )

    if not rows:
        print("No populations with both T_dust and M_dust")
        return None, None

    df = pd.DataFrame(rows)
    print(f"T_dust-DTG plot: {len(df)} populations (tier ≥ {min_tier})")
    print(f"  T_dust: {df['T_rest'].min():.0f} – {df['T_rest'].max():.0f} K")
    print(f"  log(DTG): {df['log_dtg'].min():.2f} – {df['log_dtg'].max():.2f}")
    print(f"  M_dust: {df['m_dust'].min():.1e} – {df['m_dust'].max():.1e} M_sun")
    print(f"  M_gas (Tacconi): {df['m_gas'].min():.1e} – {df['m_gas'].max():.1e} M_sun")

    # ── color setup ──────────────────────────────────────────────────
    color_configs = {
        "redshift": ("redshift", "Redshift", "plasma"),
        "stellar_mass": ("stellar_mass", "log10(M*/M_sun)", "viridis"),
        "l_ir": ("l_ir", "log10(L_IR/L_sun)", "plasma"),
        "metallicity": ("metallicity", "Metallicity (Z)", "viridis"),
        "ebv": ("ebv", "E(B-V)", "YlOrRd"),
        "log_sigma_sfr": ("log_sigma_sfr", "log Sigma_SFR", "inferno"),
        "sigma_sfr": ("log_sigma_sfr", "log Sigma_SFR", "inferno"),
        "log_ssfr": ("log_ssfr", "log sSFR (yr^-1)", "RdYlBu_r"),
        "ssfr": ("log_ssfr", "log sSFR (yr^-1)", "RdYlBu_r"),
        "log_delta_ms": ("log_delta_ms", "log delta_MS", "RdBu_r"),
        "delta_ms": ("log_delta_ms", "log delta_MS", "RdBu_r"),
        "beta_uv": ("beta_uv", "beta_UV", "coolwarm"),
    }

    if color_by in color_configs:
        col_name, color_label, cmap_name = color_configs[color_by]
        if col_name == "l_ir":
            color_data = np.log10(df[col_name].replace(0, np.nan)).values
        elif col_name in df.columns:
            color_data = df[col_name].values.astype(float)
        else:
            print(
                f"  color_by='{color_by}' maps to '{col_name}' "
                f"which is not in data. Need it in bin_property_columns."
            )
            print(f"  Available: {[c for c in df.columns if not c.startswith('_')]}")
            color_data = df["redshift"].values.astype(float)
            color_label = "Redshift"
            cmap_name = "plasma"
    elif color_by in df.columns:
        color_data = df[color_by].values.astype(float)
        color_label = color_by
        cmap_name = "viridis"
    else:
        print(f"  color_by='{color_by}' not found, falling back to redshift")
        color_data = df["redshift"].values.astype(float)
        color_label = "Redshift"
        cmap_name = "plasma"

    valid_color = np.isfinite(color_data)

    # Discrete boundaries
    _boundary_defaults = {
        "redshift": np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]),
        "stellar_mass": np.array([8.5, 9.5, 10.0, 10.3, 10.6, 11.0, 12.0]),
        "metallicity": np.array([0.002, 0.005, 0.01, 0.02, 0.04]),
        "ebv": np.array([0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8]),
        "log_sigma_sfr": np.array([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
        "log_ssfr": np.array([-11.0, -10.0, -9.5, -9.0, -8.5, -8.0]),
        "log_delta_ms": np.array([-1.0, -0.3, 0.0, 0.3, 0.6, 1.0]),
        "beta_uv": np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5]),
    }

    if color_by in _boundary_defaults:
        boundaries = _boundary_defaults[color_by]
    elif color_by in ("sigma_sfr", "ssfr", "delta_ms"):
        # Aliases
        alias_map = {
            "sigma_sfr": "log_sigma_sfr",
            "ssfr": "log_ssfr",
            "delta_ms": "log_delta_ms",
        }
        boundaries = _boundary_defaults.get(alias_map[color_by], None)
    else:
        boundaries = None

    if boundaries is None:
        boundaries = np.arange(
            np.floor(np.nanmin(color_data[valid_color])),
            np.ceil(np.nanmax(color_data[valid_color])) + 0.5,
            0.5,
        )

    boundaries = boundaries[
        (boundaries >= np.nanmin(color_data[valid_color]) - 0.5)
        & (boundaries <= np.nanmax(color_data[valid_color]) + 0.5)
    ]
    if len(boundaries) < 2:
        boundaries = np.linspace(
            np.nanmin(color_data[valid_color]),
            np.nanmax(color_data[valid_color]),
            5,
        )

    n_colors = max(len(boundaries) - 1, 1)
    base_cmap = mcm.get_cmap(cmap_name, n_colors)
    discrete_cmap = ListedColormap([base_cmap(i) for i in range(n_colors)])
    norm = BoundaryNorm(boundaries, discrete_cmap.N)
    bin_labels = [
        f"{boundaries[i]:.1f}–{boundaries[i+1]:.1f}" for i in range(len(boundaries) - 1)
    ]

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        df["log_dtg"],
        df["T_rest"],
        c=color_data,
        cmap=discrete_cmap,
        norm=norm,
        s=80,
        alpha=0.85,
        edgecolors="k",
        linewidth=0.5,
        zorder=3,
    )

    if "T_error" in df.columns:
        ax.errorbar(
            df["log_dtg"],
            df["T_rest"],
            yerr=df["T_error"],
            fmt="none",
            color="gray",
            alpha=0.3,
            capsize=2,
            elinewidth=0.8,
            zorder=1,
        )

    # Discrete colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label, fontsize=13)
    tick_positions = boundaries[:-1] + np.diff(boundaries) / 2
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(bin_labels, fontsize=9)
    for b in boundaries[1:-1]:
        cbar.ax.axhline(b, color="white", linewidth=0.8)

    # ── Radiative equilibrium model (Parente+2026, Sommovigo+2022) ─────
    # T_dust^(4+beta) ~ Sigma_SFR^0.286 / DTG  (KS law + greybody equilibrium)
    # Calibrated: T=25K at DTG=0.01, Sigma_SFR=0.01 M☉/yr/kpc²
    if show_parente:
        _beta_emis = 1.8
        _exp = 1.0 / (4 + _beta_emis)  # ~ 0.172
        _C_cal = 25.0 / (0.01**0.286 / 0.01) ** _exp  # calibration constant

        log_dtg_grid = np.linspace(
            df["log_dtg"].min() - 0.2,
            df["log_dtg"].max() + 0.2,
            100,
        )
        dtg_grid = 10**log_dtg_grid

        # Lines of constant Sigma_SFR
        sigma_sfr_values = [0.1, 1.0, 10.0, 100.0, 1000.0]
        sigma_sfr_labels = [
            "Σ=0.1  (MS at z~0)",
            "Σ=1  (MS at z~2)",
            "Σ=10  (compact SF, M82)",
            "Σ=100  (ULIRGs, SMGs)",
            "Σ=1000  (Arp 220, Eddington)",
        ]
        sigma_sfr_styles = [
            (":", 1.0),
            ("--", 1.2),
            ("-.", 1.5),
            ("-", 1.8),
            ("-", 2.2),
        ]
        for sig, label, (ls, lw) in zip(
            sigma_sfr_values, sigma_sfr_labels, sigma_sfr_styles
        ):
            T_model = _C_cal * (sig**0.286 / dtg_grid) ** _exp
            color = "darkred" if sig >= 1000 else "gray"
            alpha = 0.8 if sig >= 1000 else 0.6
            ax.plot(
                log_dtg_grid,
                T_model,
                ls=ls,
                lw=lw,
                color=color,
                alpha=alpha,
                zorder=2,
                label=label,
            )

        ax.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.9,
            title="Parente+26 model\nM☉/yr/kpc²",
            title_fontsize=8,
        )

    # ── formatting ───────────────────────────────────────────────────
    ax.set_xlabel("log10(DTG)  [M_dust / M_gas(Tacconi+20)]", fontsize=14)
    ax.set_ylabel("Rest-frame Dust Temperature (K)", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.2)

    x_pad = (df["log_dtg"].max() - df["log_dtg"].min()) * 0.1
    ax.set_xlim(df["log_dtg"].min() - x_pad, df["log_dtg"].max() + x_pad)
    ax.set_ylim(
        max(df["T_rest"].min() * 0.8, 10),
        df["T_rest"].max() * 1.15,
    )

    title = f"T_dust vs DTG ({len(df)} populations, tier ≥ {min_tier})"
    ax.set_title(title, fontsize=15, pad=15)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path as _P

        save_path = _P(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 300 if save_path.suffix.lower() in [".png", ".jpg"] else 150
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

    return fig, df


# ── NUVrJ diagnostic plot ────────────────────────────────────────────────


def create_nuvrj_plot(
    source,
    *,
    nuv_col=None,
    r_col=None,
    j_col=None,
    z_col=None,
    pop_class_col="population_class",
    z_bins=None,
    figsize=None,
    save_path=None,
):
    """
    NUVrJ color-color diagnostic plot, showing selection boundaries.

    Parameters
    ----------
    source : SimstackWrapper or DataFrame
        Either a wrapper (catalog extracted automatically) or a DataFrame.
    nuv_col, r_col, j_col : str or None
        Column names for NUV, r, J absolute magnitudes.
        If None, auto-detected from column names.
    z_col : str or None
        Redshift column. Auto-detected if None.
    pop_class_col : str or None
        Population class column (0=complete_sfg, 1=incomplete_sfg,
        2=all_qt, 3=all_sb). If None, colors by density.
    z_bins : list of tuples, optional
        Redshift bins to show, e.g. [(0.5,1), (1,1.5), ...].
        Default: 4 bins spanning the data.
    figsize : tuple, optional
    save_path : str or Path, optional

    Returns
    -------
    fig
    """
    # ── extract catalog DataFrame ────────────────────────────────────
    catalog_df = None

    if isinstance(source, pd.DataFrame):
        catalog_df = source
    else:
        # Try wrapper paths in order of preference
        for attr_chain in [
            ("population_manager", "catalog_df"),
            ("sky_catalogs", "catalog_df"),
        ]:
            obj = source
            for attr in attr_chain:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None and hasattr(obj, "columns"):
                catalog_df = obj
                break

        # Last resort: reload from the config catalog path
        if catalog_df is None and hasattr(source, "config"):
            try:
                cat_path = source.config.catalog.full_path
                if cat_path.exists():
                    print(f"Reloading catalog from {cat_path}")
                    catalog_df = pd.read_parquet(cat_path)
            except Exception as e:
                print(f"Could not reload catalog: {e}")

    if catalog_df is None:
        print("Could not extract catalog. Pass a DataFrame or SimstackWrapper.")
        return None
    if hasattr(catalog_df, "to_pandas"):
        catalog_df = catalog_df.to_pandas()

    cols = list(catalog_df.columns)
    cols_lower = {c.lower(): c for c in cols}

    # Auto-detect columns
    def _find_col(candidates, label):
        """Find first matching column from candidates."""
        for c in candidates:
            if c in catalog_df.columns:
                return c
            if c.lower() in cols_lower:
                return cols_lower[c.lower()]
        return None

    if nuv_col is None:
        nuv_col = _find_col(
            ["mabs_nuv", "MNUV", "restframe_NUV", "NUV_abs", "abs_mag_nuv"], "NUV"
        )
    if r_col is None:
        r_col = _find_col(["mabs_r", "Mr", "restframe_r", "r_abs", "abs_mag_r"], "r")
    if j_col is None:
        j_col = _find_col(["mabs_j", "MJ", "restframe_J", "j_abs", "abs_mag_j"], "J")
    if z_col is None:
        z_col = _find_col(["redshift", "zpdf_med", "zfinal", "z_best", "z"], "redshift")

    # Report what was found
    print(f"NUVrJ columns: NUV={nuv_col}, r={r_col}, J={j_col}, z={z_col}")

    # Check columns
    for col, label in [(nuv_col, "NUV"), (r_col, "r"), (j_col, "J")]:
        if col is None or col not in catalog_df.columns:
            # Show candidates to help debug
            mag_cols = [
                c
                for c in cols
                if any(
                    k in c.lower() for k in ["mag", "abs", "mabs", "nuv", "_r", "_j"]
                )
            ]
            print(f"Column for {label} not found (tried '{col}').")
            print(f"  Candidate columns: {mag_cols[:20]}")
            return None

    # Compute colors
    nuv_r = catalog_df[nuv_col] - catalog_df[r_col]
    r_j = catalog_df[r_col] - catalog_df[j_col]
    z = catalog_df[z_col].values if z_col in catalog_df.columns else None

    valid = np.isfinite(nuv_r) & np.isfinite(r_j)

    # Default z bins
    if z_bins is None:
        if z is not None:
            z_bins = [(0.5, 1.0), (1.0, 1.5), (1.5, 2.5), (2.5, 4.0)]
        else:
            z_bins = [(None, None)]

    n_panels = len(z_bins)
    ncols = min(n_panels, 4)
    nrows = (n_panels + ncols - 1) // ncols
    if figsize is None:
        figsize = (4.5 * ncols, 4.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    # Population class colors
    pop_colors = {
        0: ("C0", "complete SF", 0.15),
        1: ("C1", "incomplete SF", 0.08),
        2: ("C3", "quiescent", 0.5),
        3: ("C2", "starburst", 0.5),
    }
    has_pop = pop_class_col and pop_class_col in catalog_df.columns

    for i, (z_lo, z_hi) in enumerate(z_bins):
        if i >= len(axes):
            break
        ax = axes[i]

        # Redshift mask
        if z_lo is not None and z_hi is not None and z is not None:
            zmask = valid & (z >= z_lo) & (z < z_hi)
            ax.set_title(f"{z_lo:.1f} < z < {z_hi:.1f}", fontsize=12)
        else:
            zmask = valid
            ax.set_title("All z", fontsize=12)

        if zmask.sum() == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        x = r_j[zmask].values
        y = nuv_r[zmask].values

        if has_pop:
            pop = catalog_df[pop_class_col].values[zmask]
            # Plot each class in order (SF first as background)
            for cls in [1, 0, 3, 2]:
                clsmask = pop == cls
                if not np.any(clsmask):
                    continue
                color, label, alpha = pop_colors.get(cls, ("gray", f"class {cls}", 0.2))
                ax.scatter(
                    x[clsmask],
                    y[clsmask],
                    c=color,
                    s=2,
                    alpha=alpha,
                    rasterized=True,
                    label=f"{label} ({clsmask.sum():,})",
                )
        else:
            ax.scatter(x, y, c="gray", s=1, alpha=0.05, rasterized=True)

        # Selection boundaries: (NUV-r) > 3*(r-J) + 1 AND (NUV-r) > 3.1
        rj_grid = np.linspace(-1, 3, 200)
        boundary_line = 3.0 * rj_grid + 1.0
        ax.plot(rj_grid, boundary_line, "k--", lw=1.5, alpha=0.8)
        ax.axhline(3.1, color="k", ls="--", lw=1.5, alpha=0.8)

        # Shade quiescent region
        rj_fill = np.linspace(-1, 3, 100)
        y_fill = np.maximum(3.0 * rj_fill + 1.0, 3.1)
        ax.fill_between(rj_fill, y_fill, 8, color="red", alpha=0.05)
        ax.text(1.5, 5.5, "Quiescent", fontsize=9, color="red", alpha=0.7, ha="center")

        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(0, 7)
        ax.set_xlabel("r − J", fontsize=12)
        ax.set_ylabel("NUV − r", fontsize=12)
        ax.grid(True, alpha=0.15)

        if has_pop and i == 0:
            ax.legend(loc="upper left", fontsize=7, markerscale=4, framealpha=0.9)

    # Hide unused panels
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path as _P

        save_path = _P(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 200 if save_path.suffix.lower() in [".png", ".jpg"] else 150
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

    return fig


# ── SFR comparison plot ──────────────────────────────────────────────────


def create_sfr_comparison_plot(
    wrapper,
    *,
    catalog_sfr_key="sfr",
    catalog_sfr_is_log=True,
    color_by="redshift",
    split_filter=None,
    min_tier="A",
    figsize=(8, 8),
    save_path=None,
):
    """
    Compare stacked SFR (from L_IR) vs catalog SFR (from SED fitting).

    The stacked SFR comes from the greybody L_IR via Kennicutt+98:
        SFR_stacked = 1e-10 x L_IR

    The catalog SFR is the median LePhare/CIGALE SFR within each
    population bin (from bin_property_columns).

    Parameters
    ----------
    wrapper : SimstackWrapper
    catalog_sfr_key : str
        Key to match in bin_properties for the catalog SFR.
        Matches any key containing this string (case-insensitive).
        Default 'sfr' matches 'sfr_med', 'log_sfr', etc.
    catalog_sfr_is_log : bool
        Whether the catalog SFR in bin_properties is log10(SFR).
    color_by : str
        'redshift', 'stellar_mass', 'beta_uv'.
    min_tier : str
        Minimum fit quality tier.
    figsize : tuple
    save_path : str or Path, optional

    Returns
    -------
    fig, df_plot
    """
    import matplotlib.cm as mcm
    from matplotlib.colors import BoundaryNorm, ListedColormap

    pr = getattr(wrapper, "processed_results", None)
    if pr is None or not pr.sed_results:
        print("No processed SED results found")
        return None, None

    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank.get(min_tier.upper(), 2)

    rows = []
    for pop_id, sed in pr.sed_results.items():
        if not sed.greybody_fit_success:
            continue
        tier = getattr(sed, "fit_quality_tier", None) or "C"
        if tier_rank.get(tier, 2) > min_rank:
            continue

        # Filter by population split class
        pop_type = _extract_pop_type(pop_id)
        if split_filter is not None:
            allowed = {f"split_{i}" for i in split_filter}
            if pop_type not in allowed and pop_type != "_all_":
                continue
        else:
            if pop_type == "split_2":
                continue

        derived = pr.derived_quantities.get(pop_id)
        if not derived or not derived.star_formation_rate:
            continue
        sfr_stacked = derived.star_formation_rate
        if sfr_stacked <= 0:
            continue

        # Get catalog SFR from bin_properties
        props = sed.bin_properties
        if isinstance(props, str):
            import ast as _ast

            try:
                props = _ast.literal_eval(props)
            except (ValueError, SyntaxError):
                props = {}
        if not isinstance(props, dict):
            props = {}

        sfr_catalog = None
        mass_val = None
        luv_val = None
        beta_val = None
        for key, val in props.items():
            kl = key.lower()
            if catalog_sfr_key.lower() in kl and sfr_catalog is None:
                sfr_catalog = val
            if "mass" in kl and mass_val is None:
                mass_val = val
            if ("l_uv" in kl or "luv" in kl) and luv_val is None:
                luv_val = val
            if "beta" in kl and beta_val is None:
                beta_val = val

        if sfr_catalog is None:
            continue

        # Convert if log
        if catalog_sfr_is_log:
            sfr_catalog = 10**sfr_catalog

        if sfr_catalog <= 0 or not np.isfinite(sfr_catalog):
            continue

        # Get z from bins
        bins = _parse_bins(pop_id)
        z_val = None
        for dim, (lo, hi) in bins.items():
            dl = dim.lower()
            if "redshift" in dl or dim == "z":
                z_val = (lo + hi) / 2
        if z_val is None:
            z_val = sed.median_redshift
        if mass_val is None:
            mass_val = getattr(sed, "median_mass", np.nan)

        rows.append(
            {
                "sfr_stacked": sfr_stacked,
                "sfr_catalog": sfr_catalog,
                "redshift": z_val,
                "stellar_mass": mass_val if mass_val is not None else np.nan,
                "beta_uv": beta_val if beta_val is not None else np.nan,
                "l_uv": luv_val if luv_val is not None else np.nan,
                "tier": tier,
            }
        )

    if not rows:
        print("No populations with both stacked and catalog SFR")
        # Diagnostic
        sample_sed = list(pr.sed_results.values())[0] if pr.sed_results else None
        if sample_sed and sample_sed.bin_properties:
            print(
                f"  bin_properties keys: {list(sample_sed.bin_properties.keys()) if isinstance(sample_sed.bin_properties, dict) else type(sample_sed.bin_properties)}"
            )
            print(f"  Looking for key containing '{catalog_sfr_key}'")
        else:
            print(
                "  No bin_properties found — add sfr column to bin_property_columns in TOML"
            )
        return None, None

    df = pd.DataFrame(rows)
    print(f"SFR comparison: {len(df)} populations (tier ≥ {min_tier})")

    # ── color setup ──────────────────────────────────────────────────
    color_configs = {
        "redshift": (
            "redshift",
            "Redshift",
            "plasma",
            np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0]),
        ),
        "stellar_mass": (
            "stellar_mass",
            "log10(M*/M_sun)",
            "viridis",
            np.array([8.5, 9.5, 10.0, 10.3, 10.6, 11.0, 12.0]),
        ),
        "beta_uv": (
            "beta_uv",
            "β_UV",
            "coolwarm",
            np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5]),
        ),
    }

    col_name, color_label, cmap_name, boundaries = color_configs.get(
        color_by, color_configs["redshift"]
    )
    color_data = df[col_name].values.astype(float)
    valid_color = np.isfinite(color_data)

    boundaries = boundaries[
        (boundaries >= np.nanmin(color_data[valid_color]) - 0.5)
        & (boundaries <= np.nanmax(color_data[valid_color]) + 0.5)
    ]
    if len(boundaries) < 2:
        boundaries = np.linspace(
            np.nanmin(color_data[valid_color]),
            np.nanmax(color_data[valid_color]),
            5,
        )

    n_colors = max(len(boundaries) - 1, 1)
    base_cmap = mcm.get_cmap(cmap_name, n_colors)
    discrete_cmap = ListedColormap([base_cmap(i) for i in range(n_colors)])
    norm = BoundaryNorm(boundaries, discrete_cmap.N)
    bin_labels = [
        f"{boundaries[i]:.1f}–{boundaries[i+1]:.1f}" for i in range(len(boundaries) - 1)
    ]

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        df["sfr_catalog"],
        df["sfr_stacked"],
        c=color_data,
        cmap=discrete_cmap,
        norm=norm,
        s=60,
        alpha=0.85,
        edgecolors="k",
        linewidth=0.5,
        zorder=3,
    )

    # 1:1 line
    lims = [
        min(df["sfr_catalog"].min(), df["sfr_stacked"].min()) * 0.3,
        max(df["sfr_catalog"].max(), df["sfr_stacked"].max()) * 3,
    ]
    ax.plot(lims, lims, "k--", lw=1.5, alpha=0.5, label="1:1")
    ax.plot(lims, [l * 3 for l in lims], ":", color="gray", lw=1, alpha=0.4, label="3x")
    ax.plot(lims, [l / 3 for l in lims], ":", color="gray", lw=1, alpha=0.4)

    # Statistics
    ratio = df["sfr_stacked"] / df["sfr_catalog"]
    log_ratio = np.log10(ratio)
    valid_r = np.isfinite(log_ratio)
    if np.any(valid_r):
        med_ratio = np.median(ratio[valid_r])
        scatter_dex = np.std(log_ratio[valid_r])
        ax.text(
            0.05,
            0.95,
            f"median SFR_stack/SFR_cat = {med_ratio:.2f}\n"
            f"scatter = {scatter_dex:.2f} dex\n"
            f"N = {len(df)}",
            transform=ax.transAxes,
            fontsize=10,
            va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.9),
        )

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label, fontsize=13)
    tick_positions = boundaries[:-1] + np.diff(boundaries) / 2
    cbar.set_ticks(tick_positions)
    cbar.set_ticklabels(bin_labels, fontsize=9)
    for b in boundaries[1:-1]:
        cbar.ax.axhline(b, color="white", linewidth=0.8)

    ax.set_xlabel("SFR_catalog (LePhare)  [M_sun/yr]", fontsize=14)
    ax.set_ylabel("SFR_stacked (L_IR x 1e-10)  [M_sun/yr]", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.grid(True, alpha=0.2, which="both")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

    title = f"SFR comparison: stacked vs catalog ({len(df)} pops, tier ≥ {min_tier})"
    ax.set_title(title, fontsize=14, pad=15)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path as _P

        save_path = _P(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 300 if save_path.suffix.lower() in [".png", ".jpg"] else 150
        plt.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="none"
        )

    return fig, df


# ── PAH standalone plot functions ─────────────────────────────────────────────

def plot_pah_fit(fit_result, features, ax=None, show_components=True):
    """
    Plot a single PAH fit: raw + model (left), detrended + features (right).

    Parameters
    ----------
    fit_result : dict
        Output from PAHModel.fit_pah.
    features : list of (center, amplitude, fwhm) tuples
        PAH feature definitions (e.g. PAHModel.features).
    ax : matplotlib.axes.Axes, optional
    show_components : bool
    """
    has_detrend = "detrended_flux" in fit_result and fit_result["detrended_flux"] is not None

    if ax is not None:
        fig = ax.figure
        axes = [ax]
    else:
        n_panels = 2 if has_detrend else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

    lam = fit_result["wavelengths"]
    data = fit_result["data"]
    model = fit_result["model_spectrum"]

    ax0 = axes[0]
    ax0.scatter(lam, data, c="k", s=15, zorder=5, alpha=0.5, label="Data")
    ax0.plot(lam, model, "r-", lw=2, label="PAH + trend", zorder=4)
    if has_detrend:
        ax0.plot(lam, fit_result["smooth_trend"], "b--", lw=1.5, alpha=0.6,
                label="Smooth z-trend")
    ax0.set_xlabel("Rest-frame wavelength ($\\mu$m)")
    ax0.set_ylabel("L$_{24}$/L$_{IR}$")
    ax0.set_title("Raw data + model")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.2)

    if has_detrend and len(axes) > 1:
        ax1 = axes[1]
        flux_dt = fit_result["detrended_flux"]
        model_dt = fit_result["model_detrended"]

        ax1.scatter(lam, flux_dt, c="k", s=15, zorder=5, alpha=0.5, label="Detrended")
        ax1.plot(lam, model_dt, "r-", lw=2, label="PAH model", zorder=4)

        C0 = fit_result["continuum_level"]
        ax1.axhline(C0, color="gray", ls="--", lw=1, alpha=0.5, label=f"Baseline={C0:.3f}")

        if show_components:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
            for i, (A, fwhm, center) in enumerate(
                zip(fit_result["amplitudes"], fit_result["widths_fwhm"],
                    fit_result["centers"])):
                sigma = fwhm / 2.355
                feat = C0 + A * np.exp(-0.5 * ((lam - center) / sigma) ** 2)
                bump = A / C0 * 100 if C0 > 0 else 0
                ax1.plot(lam, feat, "-", color=colors[i % len(colors)],
                        lw=1.5, alpha=0.7, label=f"{center:.1f}μm ({bump:.0f}%)")

        for lam_c, _, _ in features:
            ax1.axvline(lam_c, color="gray", ls=":", alpha=0.2)

        ax1.set_xlabel("Rest-frame wavelength ($\\mu$m)")
        ax1.set_ylabel("Flux / smooth trend")
        ax1.set_title(f"PAH features ($\\chi^2_r$={fit_result['chi2_red']:.2f})")
        ax1.legend(fontsize=7, ncol=2, loc="upper right")
        ax1.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig


def plot_pah_forward_fit(fit_result, features, save_path=None):
    """
    Plot forward-model PAH fit results.

    Three panels:
      1. Flux vs redshift (measurement space) + model
      2. Flux vs rest wavelength + recovered intrinsic PAH spectrum
      3. MIPS 24um bandpass with PAH features at key redshifts

    Parameters
    ----------
    fit_result : dict
        Output from PAHModel.fit_forward_model.
    features : list of (center, amplitude, fwhm) tuples
        PAH feature definitions (e.g. PAHModel.features).
    save_path : str or Path, optional
    """
    from pathlib import Path as _Path

    z = fit_result["z_values"]
    flux = fit_result["data"]
    flux_dt = fit_result["detrended_flux"]
    model_dt = fit_result["model_detrended"]
    model_full = fit_result["model_flux_vs_z"]
    trend = fit_result["smooth_trend"]
    amps = fit_result["amplitudes"]
    rest_lam = 24.0 / (1 + z)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    ax.scatter(z, flux, c="k", s=15, alpha=0.5, zorder=3, label="Data")
    ax.plot(z, model_full, "r-", lw=2, label="Forward model", zorder=4)
    ax.plot(z, trend, "b--", lw=1.5, alpha=0.5, label="Smooth z-trend")

    feat_z = {0.9: "12.7", 1.8: "8.6", 2.1: "7.7", 2.9: "6.2"}
    for z_f, name in feat_z.items():
        if z.min() < z_f < z.max():
            ax.axvline(z_f, color="gray", ls=":", alpha=0.3)
            ax.text(z_f, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else flux.max() * 1.05,
                    f"{name}μm", fontsize=7, ha="center", va="bottom", color="red", alpha=0.7)

    ax.set_xlabel("Redshift")
    ax.set_ylabel("L$_{24}$/L$_{IR}$ (or flux ratio)")
    ax.set_title("Flux vs redshift\n(features enter/exit the bandpass)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.scatter(rest_lam, flux_dt, c="k", s=15, alpha=0.5, zorder=3, label="Detrended data")
    ax.plot(rest_lam, model_dt, "r-", lw=2, label="Forward model (detrended)", zorder=4)

    lam_fine = np.linspace(5, 16, 500)
    spec_intrinsic = np.ones_like(lam_fine)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    for j, (lc, _, fwhm) in enumerate(features):
        sigma = fwhm / 2.355
        if j < len(amps) and amps[j] > 0.001:
            feat = amps[j] * np.exp(-0.5 * ((lam_fine - lc) / sigma) ** 2)
            ax.plot(lam_fine, 1.0 + feat, "-", color=colors[j % len(colors)],
                    lw=1.5, alpha=0.6, label=f"{lc:.1f}μm A={amps[j]:.3f}")
            spec_intrinsic += feat

    ax.plot(lam_fine, spec_intrinsic, "k--", lw=1, alpha=0.3, label="Intrinsic PAH (recovered)")
    ax.axhline(1, color="gray", ls="--", lw=0.8, alpha=0.3)
    for lc, _, _ in features:
        ax.axvline(lc, color="gray", ls=":", alpha=0.2)

    ax.set_xlabel("Rest-frame wavelength (μm)")
    ax.set_ylabel("Flux / smooth trend")
    ax.set_title(f"Recovered intrinsic PAH spectrum\n($\\chi^2_r$ = {fit_result['chi2_red']:.2f})")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(5, 16)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    bp_lam = np.array([
        18.005, 19.134, 19.716, 19.944, 20.177, 20.415, 20.577, 20.742,
        20.909, 20.993, 21.079, 21.252, 21.427, 21.606, 21.787, 21.879,
        21.972, 22.160, 22.351, 22.545, 22.743, 22.944, 23.149, 23.358,
        23.570, 23.786, 24.006, 24.231, 24.459, 24.692, 24.930, 25.172,
        25.419, 25.670, 25.927, 26.189, 26.456, 26.729, 27.007, 27.292,
        27.582, 27.878, 28.181, 28.491, 28.808, 29.131, 29.462, 29.801,
        30.148, 30.865, 31.618, 32.207,
    ])
    bp_resp = np.array([
        0.000237, 0.000644, 0.004662, 0.012914, 0.024468, 0.076447,
        0.177656, 0.377777, 0.735924, 0.813463, 0.857335, 0.907091,
        0.957009, 0.984957, 0.997749, 1.000000, 0.998522, 0.970058,
        0.926258, 0.880798, 0.856114, 0.856779, 0.834520, 0.795473,
        0.752764, 0.777653, 0.839877, 0.911819, 0.924876, 0.897806,
        0.859096, 0.803216, 0.736609, 0.649479, 0.558191, 0.486209,
        0.428693, 0.381986, 0.326682, 0.261178, 0.195445, 0.148445,
        0.117945, 0.097827, 0.080512, 0.060997, 0.042525, 0.029764,
        0.021807, 0.011002, 0.003471, 0.000727,
    ])

    ax.fill_between(bp_lam, bp_resp, alpha=0.15, color="k")
    ax.plot(bp_lam, bp_resp, "k-", lw=2, label="MIPS 24μm response")

    for z_show, color in [(0.9, "C0"), (1.8, "C1"), (2.1, "C2"), (2.9, "C3")]:
        obs_lam = lam_fine * (1 + z_show)
        pah_scaled = (spec_intrinsic - 1.0) * 0.3
        in_band = (obs_lam > 18) & (obs_lam < 33)
        if in_band.any():
            ax.plot(obs_lam[in_band], pah_scaled[in_band], "-", color=color,
                    lw=1.5, alpha=0.7, label=f"PAH at z={z_show}")

    ax.set_xlabel("Observed wavelength (μm)")
    ax.set_ylabel("Response / PAH signal")
    ax.set_title("MIPS bandpass + PAH features\n(showing which features are in-band)")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_xlim(17, 33)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"PAH forward-model fit  |  {len(z)} spectral points  |  "
        f"$\\chi^2_r$ = {fit_result['chi2_red']:.2f}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_pah_vs_property(per_bin_results, features, save_path=None):
    """
    Science plot: recovered intrinsic PAH spectra per bin.

    Three panels:
      1. Flux vs z per bin (measurement space, shows the data)
      2. Recovered intrinsic PAH spectra overlaid (the result)
      3. Feature amplitudes vs bin center (the science)

    Parameters
    ----------
    per_bin_results : dict
        Output from PAHModel.fit_forward_per_bin.
    features : list of (center, amplitude, fwhm) tuples
        PAH feature definitions (e.g. PAHModel.features).
    save_path : str or Path, optional
    """
    import matplotlib.cm as cm
    from pathlib import Path as _Path

    group_by = per_bin_results.get("_group_by", "bin")
    df_evo = per_bin_results.get("_evolution")
    bin_labels = [k for k in per_bin_results
                  if not k.startswith("_") and isinstance(per_bin_results[k], dict)]

    if not bin_labels:
        print("No bins to plot")
        return None

    n_bins = len(bin_labels)
    colors = cm.viridis(np.linspace(0.15, 0.85, max(n_bins, 2)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    ax = axes[0]
    for i, label in enumerate(bin_labels):
        r = per_bin_results[label]
        z = r["z_values"]
        ax.scatter(z, r["detrended_flux"], c=[colors[i]], s=12, alpha=0.4, zorder=2)
        ax.plot(z, r["model_detrended"], "-", color=colors[i], lw=2, alpha=0.8,
                zorder=3, label=f"{group_by}={label}")
    ax.axhline(1, color="k", ls="--", lw=0.8, alpha=0.3)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Flux / smooth trend")
    ax.set_title("Detrended flux vs z per bin\n(bumps = PAH features)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    lam_fine = np.linspace(5, 16, 500)
    for i, label in enumerate(bin_labels):
        r = per_bin_results[label]
        amps = r["amplitudes"]
        spec = np.ones_like(lam_fine)
        for lc, amp, fwhm in zip(r["centers"], amps, r["widths_fwhm"]):
            sigma = fwhm / 2.355
            spec += amp * np.exp(-0.5 * ((lam_fine - lc) / sigma) ** 2)
        ax.plot(lam_fine, spec, "-", color=colors[i], lw=2.5, alpha=0.8,
                label=f"{group_by}={label}")

    ax.axhline(1, color="k", ls="--", lw=0.8, alpha=0.3)
    for lc, _, _ in features:
        ax.axvline(lc, color="gray", ls=":", alpha=0.2)
        ax.text(lc, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1.6,
                f"{lc}", fontsize=7, ha="center", va="bottom", color="gray")

    ax.set_xlabel("Rest-frame wavelength (μm)")
    ax.set_ylabel("1 + PAH emission")
    ax.set_title("Recovered intrinsic PAH spectra\n(deconvolved from MIPS bandpass)")
    ax.legend(fontsize=8)
    ax.set_xlim(5, 15)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    if df_evo is not None and len(df_evo) > 1:
        a_cols = [c for c in df_evo.columns if c.startswith("A_")]
        feat_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
        x = df_evo["bin_center"].values
        for j, ac in enumerate(a_cols):
            name = ac.replace("A_", "")
            y = df_evo[ac].values
            if np.max(y) < 0.001:
                continue
            ax.plot(x, y, "o-", color=feat_colors[j % len(feat_colors)],
                    ms=9, mfc="white", mew=2.5, lw=2.5, label=name)
        ax.set_xlabel(group_by.replace("_", " ").title())
        ax.set_ylabel("Intrinsic PAH amplitude")
        ax.set_title("PAH feature strength\nvs " + group_by.replace("_", " "))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    else:
        ax.text(0.5, 0.5, "Need ≥2 bins", ha="center", va="center", transform=ax.transAxes)

    fig.suptitle(
        f"PAH emission vs {group_by}  |  {n_bins} bins  |  "
        f"Forward model with real MIPS bandpass",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_pah_vs_property_bayesian(result, features, save_path=None):
    """
    Science summary plot for fit_bayesian_forward_model — mirrors plot_pah_vs_property.

    Three panels:
      1. Detrended flux vs z per bin (data bumps = PAH features)
      2. Recovered intrinsic PAH spectra per bin (rest-frame, deconvolved from bandpass)
      3. Per-bin alpha vs bin center — one series per feature group in group_amplitudes modes

    Dispatches on alpha_per_bin shape: (M,) for mode 1, (G, M) for modes 2/3.

    Parameters
    ----------
    result : dict
        Output from PAHModel.fit_bayesian_forward_model.
    features : list of (center, amplitude, fwhm) tuples
        PAH feature definitions (e.g. PAHModel.features).
    save_path : str or Path, optional
    """
    from pathlib import Path as _Path

    bin_labels       = result["bin_labels"]
    bin_centers      = result["bin_centers"]
    alpha_per_bin    = result["alpha_per_bin"]    # (M,) or (G, M)
    alpha_samples    = result["alpha_samples"]    # (n_flat, M) or (n_flat, G, M)
    medians          = result["medians"]
    groups           = result["feature_groups"]
    group_names      = result.get("group_names", [f"g{g}" for g in range(len(groups))])
    feat_scale       = result["_feat_scale"]
    widths_sigma     = result["_widths_sigma"]
    obs_per_bin      = result["_obs_per_bin"]
    z_union          = result["_z_union"]
    T_grid           = result["_T_grid"]
    group_amplitudes = result.get("group_amplitudes", False)

    M        = len(bin_labels)
    n_groups = T_grid.shape[0]
    colors_m = plt.cm.viridis(np.linspace(0.15, 0.85, max(M, 2)))
    colors_g = plt.cm.plasma(np.linspace(0.15, 0.85, max(n_groups, 2)))

    # For panel 1: compute pah_factor at each bin using posterior medians.
    # Mode 1: 1 + α_m × (r @ T); modes 2/3: 1 + Σ_g α_gm × T_g
    def _pah_factor_plot(m, T_arr):
        if not group_amplitudes:
            r_med = np.concatenate([[1.0], [
                medians.get(f"r_group{gi}", 1.0) for gi in range(1, n_groups)
            ]])
            return 1.0 + alpha_per_bin[m] * (r_med @ T_arr)
        else:
            return 1.0 + alpha_per_bin[:, m] @ T_arr   # (G,) @ (G, n_obs) = (n_obs,)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # ── Panel 1: detrended flux vs z ─────────────────────────────────────────
    ax = axes[0]
    for m, label in enumerate(bin_labels):
        obs    = obs_per_bin[label]
        z_m    = obs["z"]
        flux_m = obs["flux"]
        sig_m  = obs["sigma"]

        T_data = np.array([np.interp(z_m, z_union, T_grid[gi]) for gi in range(n_groups)])
        pah_m  = _pah_factor_plot(m, T_data)
        w_log  = flux_m / (sig_m * np.log(10))
        X      = np.column_stack([z_m, z_m**2, np.ones_like(z_m)])
        y_log  = np.log10(flux_m / pah_m)
        coeffs, _, _, _ = np.linalg.lstsq(X * w_log[:, None], y_log * w_log, rcond=None)
        baseline = 10.0 ** (X @ coeffs)

        detrended     = flux_m / baseline
        detrended_err = sig_m  / baseline

        z_fine = np.linspace(z_m.min(), z_m.max(), 150)
        T_fine = np.array([np.interp(z_fine, z_union, T_grid[gi]) for gi in range(n_groups)])
        pah_curve = _pah_factor_plot(m, T_fine)

        ax.errorbar(z_m, detrended, yerr=detrended_err,
                    fmt="o", ms=5, alpha=0.5, color=colors_m[m], ecolor=colors_m[m])
        ax.plot(z_fine, pah_curve, "-", color=colors_m[m], lw=2.0, alpha=0.85, label=label)

    ax.axhline(1, color="k", ls="--", lw=0.8, alpha=0.3)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Flux / smooth trend")
    ax.set_title("Detrended flux vs z per bin\n(bumps = PAH features)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.2)

    # ── Panel 2: recovered intrinsic PAH spectra ──────────────────────────────
    # 1 + Σ_{g,f} α_gm × feat_scale[f] × G(λ; lc_f, σ_f)
    # For mode 1: α_gm → α_m × r_g; for modes 2/3: α_gm directly.
    ax = axes[1]
    lam_fine = np.linspace(5, 16, 500)
    for m, label in enumerate(bin_labels):
        spec = np.ones_like(lam_fine)
        for gi, g in enumerate(groups):
            if group_amplitudes:
                a_gm = alpha_per_bin[gi, m]
            else:
                r_med_gi = medians.get(f"r_group{gi}", 1.0) if gi > 0 else 1.0
                a_gm = alpha_per_bin[m] * r_med_gi
            for j in g:
                lc  = features[j][0]
                sig = widths_sigma[j]
                spec += a_gm * feat_scale[j] * np.exp(-0.5 * ((lam_fine - lc) / sig) ** 2)
        ax.plot(lam_fine, spec, "-", color=colors_m[m], lw=2.5, alpha=0.8, label=label)

    ax.axhline(1, color="k", ls="--", lw=0.8, alpha=0.3)
    for lc, _, _ in features:
        ax.axvline(lc, color="gray", ls=":", alpha=0.2)
        ax.text(lc, max(ax.get_ylim()[1], 1.6),
                f"{lc}", fontsize=7, ha="center", va="bottom", color="gray")
    ax.set_xlabel("Rest-frame wavelength (μm)")
    ax.set_ylabel("1 + PAH emission")
    ax.set_title("Recovered intrinsic PAH spectra\n(deconvolved from MIPS bandpass)")
    ax.legend(fontsize=8)
    ax.set_xlim(5, 15)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: alpha vs bin center ──────────────────────────────────────────
    # Mode 1: one series; modes 2/3: G series (one per feature group)
    ax = axes[2]
    if not group_amplitudes:
        q16 = np.percentile(alpha_samples, 16, axis=0)  # (M,)
        q84 = np.percentile(alpha_samples, 84, axis=0)
        for m in range(M):
            ax.errorbar(
                bin_centers[m], alpha_per_bin[m],
                yerr=[[alpha_per_bin[m] - q16[m]], [q84[m] - alpha_per_bin[m]]],
                fmt="o", ms=9, color=colors_m[m], ecolor=colors_m[m],
                elinewidth=2, capsize=5, label=bin_labels[m], zorder=5,
            )
        ax.set_ylabel("PAH amplitude alpha")
        ax.set_title("PAH amplitude per bin\n(posterior median + 16–84 CI)")
    else:
        # alpha_samples shape (n_flat, G, M); alpha_per_bin shape (G, M)
        q16 = np.percentile(alpha_samples, 16, axis=0)  # (G, M)
        q84 = np.percentile(alpha_samples, 84, axis=0)
        for g in range(n_groups):
            ax.errorbar(
                bin_centers, alpha_per_bin[g],
                yerr=[alpha_per_bin[g] - q16[g], q84[g] - alpha_per_bin[g]],
                fmt="o-", ms=7, lw=1.5, color=colors_g[g], ecolor=colors_g[g],
                elinewidth=1.5, capsize=4, label=group_names[g], zorder=5,
            )
        ax.set_ylabel("PAH amplitude alpha_gm")
        ax.set_title("Per-group PAH amplitude vs bin\n(posterior median + 16–84 CI)")

    ax.set_xlabel(result.get("_group_col", "bin center").replace("_", " "))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    mode_label = "single-alpha" if not group_amplitudes else (
        "shared-beta" if not result.get("independent_betas", False) else "independent-beta"
    )
    group_col = result.get("_group_col", "bin property")
    fig.suptitle(
        f"PAH emission vs {group_col}  |  {M} bins  |  Bayesian [{mode_label}]",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_pah_multibin_forward_fit(result, features, save_path=None):
    """
    Plot multi-bin forward model results.

    Four panels:
      1. Flux vs z — all bins with model and baseline
      2. Detrended flux vs z — PAH modulation (shared template)
      3. Recovered intrinsic PAH spectra per bin (shared shape, different α)
      4. Per-bin α vs bin center — the science result

    Parameters
    ----------
    result : dict
        Output from PAHModel.fit_forward_model_multibin.
    features : list of (center, amplitude, fwhm) tuples
        PAH feature definitions (e.g. PAHModel.features).
    save_path : str or Path, optional
    """
    from pathlib import Path as _Path

    bin_labels = result["bin_labels"]
    bin_centers = result["bin_centers"]
    alpha = result["alpha_per_bin"]
    alpha_err = result["alpha_err_per_bin"]
    ratios = result["group_ratios"]
    groups = result["feature_groups"]
    feat_scale = result["feat_scale"]
    model_per_bin = result["model_per_bin"]
    group_col = result.get("_group_col", "bin")
    chi2_red = result["chi2_red"]
    M = len(bin_labels)

    colors = plt.cm.plasma(np.linspace(0.2, 0.85, M))
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5))

    ax = axes[0]
    for m, label in enumerate(bin_labels):
        d = model_per_bin[label]
        idx = np.argsort(d["z"])
        ax.scatter(d["z"], d["flux"], c=[colors[m]], s=12, alpha=0.4, zorder=3)
        ax.plot(d["z"][idx], d["model"][idx], "-", color=colors[m], lw=2, label=label, zorder=4)
        ax.plot(d["z"][idx], d["baseline"][idx], "--", color=colors[m], lw=1, alpha=0.35)
    ax.set_xlabel("Redshift")
    ax.set_ylabel(r"$f_{24}/f_{\rm peak}$")
    ax.set_title("Flux vs z\n(model — baseline)")
    ax.legend(fontsize=7, title=group_col.replace("_", " "))
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    for m, label in enumerate(bin_labels):
        d = model_per_bin[label]
        idx = np.argsort(d["z"])
        ax.scatter(d["z"], d["detrended_data"], c=[colors[m]], s=12, alpha=0.4, zorder=3)
        ax.plot(d["z"][idx], d["detrended_model"][idx], "-", color=colors[m], lw=2,
                label=label, zorder=4)
    ax.axhline(1, color="k", ls="--", lw=0.8, alpha=0.3)
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Flux / baseline")
    ax.set_title("PAH modulation\n(shared template, per-bin α)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    ax = axes[2]
    lam_fine = np.linspace(5, 16, 500)
    for m, label in enumerate(bin_labels):
        spec = np.ones_like(lam_fine)
        for gi, g in enumerate(groups):
            r_gi = ratios[gi]
            for j in g:
                lc, _, fwhm = features[j]
                sigma = fwhm / 2.355
                spec += alpha[m] * r_gi * feat_scale[j] * np.exp(
                    -0.5 * ((lam_fine - lc) / sigma) ** 2)
        ax.plot(lam_fine, spec, "-", color=colors[m], lw=2.5, alpha=0.9, label=label)
    ax.axhline(1, color="k", ls="--", lw=0.8, alpha=0.3)
    for lc, _, _ in features:
        ax.axvline(lc, color="gray", ls=":", alpha=0.2)
        ax.text(lc, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1.2,
                f"{lc}", fontsize=6, ha="center", va="bottom", color="gray")
    ax.set_xlabel("Rest-frame wavelength (μm)")
    ax.set_ylabel("1 + PAH emission")
    ax.set_title("Recovered intrinsic PAH spectra\n(jointly constrained)")
    ax.legend(fontsize=7)
    ax.set_xlim(5, 15)
    ax.grid(True, alpha=0.2)

    ax = axes[3]
    ax.errorbar(bin_centers, alpha, yerr=alpha_err, fmt="o-", color="navy",
                ms=9, mfc="white", mew=2.5, lw=2, capsize=4, elinewidth=1.5)
    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.3)
    ax.set_xlabel(group_col.replace("_", " ").title())
    ax.set_ylabel("PAH amplitude α")
    ax.set_title("PAH strength vs property\n(jointly constrained)")
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Multi-bin PAH forward model  |  {M} bins  |  χ²_r = {chi2_red:.2f}",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path = _Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


# ============================================================================
# PAH dithered-stacking diagnostics (pah_dither / pah_spectrum modules)
# ============================================================================


def plot_dither_kernels(
    z_grid,
    bands=("MIPS_24", "MIPS_70"),
    sigma_z0_list=None,
    features=None,
    feature_groups=None,
    save_path=None,
):
    """Feature-group kernels T_g,b(z) per band, optionally photo-z smeared.

    Shows which redshifts each PAH group modulates each bandpass at, and
    how σ_z smearing erodes the modulation contrast.
    """
    from scipy.ndimage import gaussian_filter1d

    from .pah_spectrum import DEFAULT_GROUPS, feature_band_curves

    groups = DEFAULT_GROUPS if feature_groups is None else feature_groups
    sigma_z0_list = sigma_z0_list or [0.0]
    fig, axes = plt.subplots(1, len(bands), figsize=(7 * len(bands), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(groups)))
    dz = z_grid[1] - z_grid[0]
    for ax, band in zip(axes, bands, strict=False):
        T = feature_band_curves(z_grid, band, features, groups)
        for g in range(T.shape[1]):
            for k, sz0 in enumerate(sigma_z0_list):
                curve = T[:, g]
                if sz0 > 0:
                    sig_pix = sz0 * (1 + np.median(z_grid)) / dz
                    curve = gaussian_filter1d(curve, sig_pix)
                ax.plot(
                    z_grid,
                    curve,
                    color=colors[g],
                    ls=["-", "--", ":"][k % 3],
                    lw=2 if k == 0 else 1.2,
                    alpha=1.0 if k == 0 else 0.7,
                )
        ax.set_xlabel("redshift")
        ax.set_title(band.replace("_", " ") + " µm")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("in-band response to unit-peak feature")
    from .pah_spectrum import DEFAULT_FEATURES

    feats = DEFAULT_FEATURES if features is None else features
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=colors[g],
            lw=2,
            label="+".join(f"{feats[j][0]:g}" for j in grp) + " µm",
        )
        for g, grp in enumerate(groups)
    ]
    for k, sz0 in enumerate(sigma_z0_list):
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color="gray",
                ls=["-", "--", ":"][k % 3],
                label=f"σ_z0={sz0:g}",
            )
        )
    axes[-1].legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_fisher_summary(fisher_result, save_path=None):
    """Three-panel autopsy of a FisherResult: correlation matrix, marginal
    CRLBs (ratio and flux flavors), and the worst degeneracy direction."""
    fr = fisher_result
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    im = axes[0].imshow(fr.corr, vmin=-1, vmax=1, cmap="RdBu_r")
    axes[0].set_xticks(range(len(fr.labels)))
    axes[0].set_yticks(range(len(fr.labels)))
    axes[0].set_xticklabels(fr.labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_yticklabels(fr.labels, fontsize=8)
    axes[0].set_title("parameter correlations")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    x = np.arange(len(fr.crlb))
    labels_A = fr.labels[1:]
    axes[1].bar(x - 0.2, fr.snr, 0.4, label="detection SNR (Ã/σ_Ã)", color="C0")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels_A, rotation=45, ha="right", fontsize=8)
    ax2 = axes[1].twinx()
    ax2.bar(x + 0.2, fr.crlb, 0.4, label="ratio CRLB σ(A)", color="C1")
    ax2.set_yscale("log")
    axes[1].set_ylabel("detection SNR", color="C0")
    ax2.set_ylabel("σ(A) incl. continuum", color="C1")
    axes[1].set_title(f"marginal bounds (C_err={fr.C_err:.3f})")

    worst = fr.eigvecs[:, 0]  # smallest eigenvalue = worst direction
    axes[2].bar(x, worst, color="C3")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels_A, rotation=45, ha="right", fontsize=8)
    axes[2].set_title(
        f"worst-constrained direction (λ_min/λ_max={fr.eigvals[0]/fr.eigvals[-1]:.1e})"
    )
    axes[2].axhline(0, color="k", lw=0.5)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_strategy_sweep(
    sweep_df, value="snr_min", x="dz", y="n_stagger", save_path=None
):
    """Heatmap of a sweep_strategies figure of merit over two grid axes.

    Extra grid dimensions present in the DataFrame are shown as separate
    panels (one per unique combination)."""
    others = [
        c
        for c in ("sigma_z0", "f_cat", "n_property_bins")
        if c not in (x, y) and sweep_df[c].nunique() > 1
    ]
    if others:
        combos = sweep_df[others].drop_duplicates()
        panels = [
            (
                ", ".join(f"{c}={row[c]:g}" for c in others),
                sweep_df[np.all([sweep_df[c] == row[c] for c in others], axis=0)],
            )
            for _, row in combos.iterrows()
        ]
    else:
        panels = [("", sweep_df)]
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.2), squeeze=False)
    for ax, (title, sub) in zip(axes[0], panels, strict=False):
        pivot = sub.pivot_table(index=y, columns=x, values=value)
        im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:g}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:g}" for v in pivot.index])
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{value} {title}")
        for (i, j), v in np.ndenumerate(pivot.to_numpy()):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="w", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pseudo_spectrum_overlay(
    spec_df, truth=None, prop_bin_id=None, value="ratio", save_path=None
):
    """Measured pseudo-spectrum vs rest wavelength, with the injected truth.

    spec_df comes from PAHSpectrumModel.pseudo_spectrum. value="ratio"
    plots the continuum-normalized excess; "excess_snr" the significance.
    """
    sub = (
        spec_df
        if prop_bin_id is None
        else spec_df[spec_df["prop_bin_id"] == prop_bin_id]
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    for band, marker, color in (("MIPS_24", "o", "C0"), ("MIPS_70", "s", "C2")):
        s = sub[sub["band"] == band]
        if not len(s):
            continue
        yerr = s["ratio_err"] if value == "ratio" else None
        ax.errorbar(
            s["lam_rest"],
            s[value],
            yerr=yerr,
            fmt=marker,
            ms=4,
            color=color,
            alpha=0.7,
            lw=1,
            label=band.replace("_", " "),
        )
    if truth is not None and value == "ratio":
        lam = np.linspace(sub["lam_rest"].min(), sub["lam_rest"].max(), 600)
        from .dust_evolution import _greybody_nu

        cont = _greybody_nu(2.998e14 / lam, truth.T_warm, truth.beta_warm)
        spec = truth.rest_spectrum(lam) / truth.continuum_amp
        ax.plot(
            lam,
            spec / cont - 1.0,
            "k-",
            lw=1,
            alpha=0.6,
            label="injected truth (sharp)",
        )
    ax.set_xlabel("rest wavelength [µm]")
    ax.set_ylabel("F / continuum − 1" if value == "ratio" else "excess SNR")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    if value == "ratio":
        ax.set_ylim(-1, max(8.0, np.nanpercentile(sub[value], 98) * 1.3))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pah_spectrum_corner(result, params=None, save_path=None):
    """Corner plot of the PAHSpectrumModel.fit_mcmc posterior."""
    import corner

    chain = result.sampler.get_chain(discard=0, flat=True)
    n_burn = chain.shape[0] // 3
    chain = chain[n_burn:]
    names = result.param_names
    if params is not None:
        idx = [names.index(p) for p in params]
        chain = chain[:, idx]
        names = [names[i] for i in idx]
    fig = corner.corner(
        chain,
        labels=names,
        show_titles=True,
        title_fmt=".3f",
        quantiles=[0.16, 0.5, 0.84],
    )
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
