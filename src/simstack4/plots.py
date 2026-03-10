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
from .exceptions.simstack_exceptions import PlotError

logger = setup_logging()

"""
SED Grid Plot for Simstack4

Plots stacked flux densities and fitted greybody models on a
(column × row) grid of binning dimensions.  Handles both
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
    "suptitle":     14,
    "col_header":   12,
    "row_label":    12,
    "axis_label":   12,
    "tick":         11,
    "legend":       15,
    "empty_cell":   14,
    "colorbar":     12,
    "colorbar_tick": 11,
}

# ── Marker / line styling ────────────────────────────────────────────────
STYLE = {
    "marker_size":          10,
    "marker_alpha":         0.85,
    "marker_fillstyle":     "none",     # open circles
    "marker_edgewidth":     0.7,
    "capsize":              2,
    "marker_lw":            1.2,
    "model_lw":             1.8,
    "model_alpha":          0.6,
    "prior_lw":             1.4,
    "prior_alpha":          0.7,
    "prior_color":          "0.45",
    "upper_limit_ms":       3,
    "upper_limit_alpha":    0.4,
    "legend_handle_lw":     2,
    "grid_alpha":           0.15,
}

# ── Layout ───────────────────────────────────────────────────────────────
LAYOUT = {
    "panel_width":   3.2,      # inches per column
    "panel_height":  2.8,      # inches per row
    "fig_pad_w":     1.0,      # extra width (inches)
    "fig_pad_h":     0.8,      # extra height (inches)
    "hspace":        0.04,
    "wspace":        0.04,
    "col_header_pad": 4,
    "row_label_pad":  18,
    "suptitle_y":     1.01,
    "lam_model_pad_lo": 0.3,   # model λ grid extends to λ_min × this
    "lam_model_pad_hi": 1.5,   # model λ grid extends to λ_max × this
    "xlim_pad_lo":      0.5,   # x-axis extends to λ_min × this
    "xlim_pad_hi":      1.5,   # x-axis extends to λ_max × this
    "n_model_points":   120,
}

# ── Colorbar ─────────────────────────────────────────────────────────────
CBAR = {
    "fraction":  0.02,
    "pad":       0.02,
    "aspect":    30,
    "right":     0.88,    # subplots_adjust right when cbar present
}

# ── Type colours for population splits ───────────────────────────────────
TYPE_COLORS = {
    "sfg":        "#1f77b4",   # blue (star-forming)
    "quiescent":  "#d62728",   # red  (quiescent)
    "agn":        "#2ca02c",   # green
    "dusty":      "#ff7f0e",   # orange
    "split_0":    "#1f77b4",   # blue (star-forming)
    "split_1":    "#d62728",   # red  (quiescent)
}
_EXTRA_COLORS = [
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Type-specific sequential cmaps for extra-dimension gradients
_TYPE_CMAPS = {
    "sfg":        cm.winter,
    "quiescent":  cm.autumn,
    "agn":        cm.Greens,
    "dusty":      cm.Oranges,
    "split_0":    cm.winter,
    "split_1":    cm.autumn,
}

# Short display names for type labels
_TYPE_SHORT = {
    "split_0": "SF", "split_1": "Q",
    "sfg": "SF", "quiescent": "Q", "agn": "AGN",
}

# Dimension label pretty-printing
_DIM_LABELS = {
    "redshift": "z",
    "stellar_mass": "log M★",
    "beta_uv": "β_UV",
    "l_uv": "log L_UV",
    "l_nuv": "log L_NUV",
}

# Grid dimension auto-detection priority (first match → columns)
_DIM_PRIORITY = ["redshift", "l_uv", "l_nuv", "beta_uv", "stellar_mass"]


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
        # A bin-range segment has ≥3 parts ending in two floats
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


def _solve_amplitude_at_T(greybody_fitter, wave_rest, fluxes, errors,
                           T_rest, beta):
    """
    Analytically solve for log10(A) at fixed T and β.

    Returns (log10_A, success_bool).
    """
    template = greybody_fitter.greybody_model(wave_rest, 0.0, T_rest, beta)
    w = 1.0 / errors**2
    denom = np.sum(template**2 * w)
    if denom <= 0:
        return -35.0, False
    x_opt = np.sum(fluxes * template * w) / denom
    if x_opt > 0:
        return np.log10(x_opt), True
    return -35.0, False


# ---------------------------------------------------------------------------
# Main plotting function
# ---------------------------------------------------------------------------
def plot_sed_grid(
    wrapper,
    *,
    col_dim: str | None = None,
    row_dim: str | None = None,
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
        raise ValueError(
            "wrapper must have stacking_results and processed_results"
        )

    sed_results = pr.sed_results
    if not sed_results:
        raise ValueError("No SED results in processed_results")

    flux_scale = 1e3 if flux_unit == "mJy" else 1.0
    flux_label = f"Flux Density ({flux_unit})"

    # ── wavelengths from config ───────────────────────────────────────────
    wavelengths = np.array([
        wrapper.config.maps[m].wavelength for m in sr.map_names
    ])
    sort_idx = np.argsort(wavelengths)
    wavelengths = wavelengths[sort_idx]
    map_names_sorted = [sr.map_names[i] for i in sort_idx]

    print(f"📡 Wavelengths: {wavelengths} µm  ({len(wavelengths)} bands)")

    # ── population label → stacking array index ──────────────────────────
    pop_label_to_idx = {
        label: i for i, label in enumerate(sr.population_labels)
    }

    # ── parse every population ID ────────────────────────────────────────
    pop_ids = list(sed_results.keys())
    parsed = {pid: _parse_bins(pid) for pid in pop_ids}
    types  = {pid: _extract_pop_type(pid) for pid in pop_ids}
    unique_types = sorted(set(types.values()))
    has_type_split = unique_types != ["_all_"]

    if has_type_split:
        type_counts = {t: sum(1 for v in types.values() if v == t)
                       for t in unique_types}
        print(f"🏷️  Population types: "
              f"{', '.join(f'{t}: {n}' for t, n in type_counts.items())}")
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
            wrapper, sr, pr, wavelengths, map_names_sorted,
            pop_label_to_idx, flux_scale, flux_label,
            show_errors, show_model, show_prior, ylim,
        )

    if col_dim is None or row_dim is None:
        ranked = sorted(all_dims, key=lambda d: (
            _DIM_PRIORITY.index(d) if d in _DIM_PRIORITY else 99
        ))
        if col_dim is None:
            col_dim = ranked[0]
        if row_dim is None:
            row_dim = [d for d in ranked if d != col_dim][0]

    print(f"📐 Grid: {col_dim} (columns) × {row_dim} (rows)")

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

    print(f"📊 Grid size: {n_cols} × {n_rows} = {n_cols * n_rows} panels")

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
        n_rows, n_cols, figsize=figsize,
        sharex=True, sharey=True, squeeze=False,
    )

    # fine λ grid for model curves (observed frame)
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
                pid for pid in pop_ids
                if parsed[pid].get(col_dim) == cbin
                and parsed[pid].get(row_dim) == rbin
            ]

            if not cell_pops:
                ax.text(
                    0.5, 0.5, "—", ha="center", va="center",
                    transform=ax.transAxes, fontsize=FONT["empty_cell"],
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
                    fluxes = np.array([
                        sr.flux_densities[m][idx] for m in map_names_sorted
                    ]) * flux_scale
                    errors = np.array([
                        sr.flux_errors[m][idx] for m in map_names_sorted
                    ]) * flux_scale

                    det = fluxes > 0

                    if np.any(det):
                        if show_errors:
                            ax.errorbar(
                                wavelengths[det], fluxes[det],
                                yerr=errors[det],
                                fmt="o", color=colour,
                                ms=STYLE["marker_size"],
                                mfc="none",
                                mew=STYLE["marker_edgewidth"],
                                capsize=STYLE["capsize"],
                                lw=STYLE["marker_lw"],
                                alpha=STYLE["marker_alpha"], zorder=3,
                            )
                        else:
                            ax.plot(
                                wavelengths[det], fluxes[det],
                                "o", color=colour,
                                ms=STYLE["marker_size"],
                                mfc="none",
                                mew=STYLE["marker_edgewidth"],
                                alpha=STYLE["marker_alpha"], zorder=3,
                            )

                    # ── upper limits ─────────────────────────────────
                    nondet = (~det) & (errors > 0)
                    if np.any(nondet):
                        ax.plot(
                            wavelengths[nondet], 2 * errors[nondet],
                            "v", color=colour,
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
                            model_flux = (
                                pr.greybody_fitter.greybody_model(
                                    lam_rest, sed.amplitude,
                                    sed.dust_temperature_rest_frame, beta,
                                )
                                * flux_scale
                            )
                            ax.plot(
                                lam_obs_model, model_flux,
                                "-", color=colour,
                                lw=STYLE["model_lw"],
                                alpha=STYLE["model_alpha"], zorder=2,
                            )
                        except Exception as e:
                            logger.debug(f"Model failed for {pid}: {e}")

                    # ── Temperature-prior overlay ──────────────────────
                    if show_prior and not prior_plotted:
                        z = sed.median_redshift
                        beta = (sed.emissivity_index or 1.8)
                        try:
                            T_prior, _ = (
                                pr.greybody_fitter
                                .temperature_prior_relation(z)
                            )

                            # Solve amplitude analytically at T_prior
                            # using this population's data (in Jy)
                            fluxes_jy = np.array([
                                sr.flux_densities[m][idx]
                                for m in map_names_sorted
                            ])
                            errors_jy = np.array([
                                sr.flux_errors[m][idx]
                                for m in map_names_sorted
                            ])
                            wave_rest_data = wavelengths / (1.0 + z)

                            # Only use positive detections for scaling
                            pos = fluxes_jy > 0
                            if np.sum(pos) >= 1:
                                A_prior, ok = _solve_amplitude_at_T(
                                    pr.greybody_fitter,
                                    wave_rest_data[pos],
                                    fluxes_jy[pos],
                                    errors_jy[pos],
                                    T_prior, beta,
                                )

                                if ok:
                                    lam_rest = lam_obs_model / (1.0 + z)
                                    prior_flux = (
                                        pr.greybody_fitter.greybody_model(
                                            lam_rest, A_prior,
                                            T_prior, beta,
                                        )
                                        * flux_scale
                                    )
                                    ax.plot(
                                        lam_obs_model, prior_flux,
                                        "--", color=STYLE["prior_color"],
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
                        Line2D([0], [0], color=colour,
                               lw=STYLE["legend_handle_lw"])
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
                        Line2D([0], [0], color=STYLE["prior_color"],
                               lw=STYLE["prior_lw"], ls="--")
                    )
                    _pname = getattr(pr.greybody_fitter, "temperature_prior", "prior")
                    legend_labels.append(f"{_pname} T_rf={_T_prior:.0f}K")

                if legend_handles:
                    if fontsize_legend is None:
                        fontsize = FONT["legend"]
                    else:
                        fontsize = fontsize_legend
                    ax.legend(
                        legend_handles, legend_labels,
                        loc="upper left", fontsize=fontsize,
                        framealpha=0.7, edgecolor="0.8",
                        handlelength=1.2, borderpad=0.3,
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
                labelsize=FONT["tick"], which="both", direction="in",
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
                ax.set_xlabel("Observed λ (µm)", fontsize=FONT["axis_label"])
            if ci == 0:
                ax.set_ylabel(flux_label, fontsize=FONT["axis_label"])

    # ── title & layout ───────────────────────────────────────────────────
    n_pops = len(pop_ids)
    title = f"SED Grid — {n_pops} populations"
    if has_type_split:
        counts = {t: sum(1 for v in types.values() if v == t)
                  for t in unique_types}
        title += "  (" + ", ".join(
            f"{_TYPE_SHORT.get(t, t)}: {n}" for t, n in counts.items()
        ) + ")"
    fig.suptitle(title, fontsize=FONT["suptitle"], y=LAYOUT["suptitle_y"])

    fig.tight_layout()

    # ── colorbar for extra-dimension gradient ────────────────────────────
    if extra_dims:
        from matplotlib.colors import Normalize

        ed = extra_dims[0]
        ed_bins = sorted(set(
            parsed[pid].get(ed) for pid in pop_ids
            if parsed[pid].get(ed) is not None
        ))

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
                hspace=LAYOUT["hspace"], wspace=LAYOUT["wspace"],
                right=CBAR["right"],
            )
            cbar = fig.colorbar(
                sm, ax=axes,
                fraction=CBAR["fraction"], pad=CBAR["pad"],
                aspect=CBAR["aspect"],
            )
            cbar.set_label(_dim_label(ed), fontsize=FONT["colorbar"])
            cbar.ax.tick_params(labelsize=FONT["colorbar_tick"])
            cbar.set_ticks(mids)
            cbar.set_ticklabels([f"{m:.1f}" for m in mids])
        else:
            plt.subplots_adjust(
                hspace=LAYOUT["hspace"], wspace=LAYOUT["wspace"],
            )
    else:
        plt.subplots_adjust(
            hspace=LAYOUT["hspace"], wspace=LAYOUT["wspace"],
        )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            save_path, dpi=dpi, bbox_inches="tight", facecolor="white",
        )
        print(f"💾 Saved to {save_path}")

    return fig


# ---------------------------------------------------------------------------
# Fallback: simple single-panel plot
# ---------------------------------------------------------------------------
def _plot_sed_simple(
    wrapper, sr, pr, wavelengths, map_names, pop_label_to_idx,
    flux_scale, flux_label, show_errors, show_model, show_prior, ylim,
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

        fluxes = np.array([
            sr.flux_densities[m][idx] for m in map_names
        ]) * flux_scale
        errors = np.array([
            sr.flux_errors[m][idx] for m in map_names
        ]) * flux_scale

        det = fluxes > 0
        label = f"{pid.split('__')[0]}… N={sed.n_sources}"

        if np.any(det):
            if show_errors:
                ax.errorbar(
                    wavelengths[det], fluxes[det], yerr=errors[det],
                    fmt="o", color=c,
                    ms=STYLE["marker_size"],
                    mfc="none",
                    mew=STYLE["marker_edgewidth"],
                    capsize=STYLE["capsize"] + 1,
                    lw=STYLE["marker_lw"] + 0.3,
                    alpha=STYLE["marker_alpha"], label=label,
                )
            else:
                ax.plot(
                    wavelengths[det], fluxes[det],
                    "o", color=c,
                    ms=STYLE["marker_size"],
                    mfc="none",
                    mew=STYLE["marker_edgewidth"],
                    alpha=STYLE["marker_alpha"], label=label,
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
                    mf = pr.greybody_fitter.greybody_model(
                        lam_rest, sed.amplitude,
                        sed.dust_temperature_rest_frame, beta,
                    ) * flux_scale
                    ax.plot(
                        lam_obs_model, mf, "-", color=c,
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
                    T_prior, _ = (
                        pr.greybody_fitter
                        .temperature_prior_relation(z)
                    )
                    fluxes_jy = np.array([
                        sr.flux_densities[m][idx] for m in map_names
                    ])
                    errors_jy = np.array([
                        sr.flux_errors[m][idx] for m in map_names
                    ])
                    wave_rest_data = wavelengths / (1.0 + z)
                    pos = fluxes_jy > 0
                    if np.sum(pos) >= 1:
                        A_prior, ok = _solve_amplitude_at_T(
                            pr.greybody_fitter,
                            wave_rest_data[pos], fluxes_jy[pos],
                            errors_jy[pos], T_prior, beta,
                        )
                        if ok:
                            lam_rest = lam_obs_model / (1.0 + z)
                            pf = pr.greybody_fitter.greybody_model(
                                lam_rest, A_prior, T_prior, beta,
                            ) * flux_scale
                            ax.plot(
                                lam_obs_model, pf,
                                "--", color=c,
                                lw=STYLE["prior_lw"],
                                alpha=STYLE["prior_alpha"] - 0.3,
                            )
                except Exception:
                    pass

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(ylim)
    ax.set_xlabel("Observed λ (µm)", fontsize=FONT["axis_label"])
    ax.set_ylabel(flux_label, fontsize=FONT["axis_label"])
    ax.tick_params(labelsize=FONT["tick"])
    ax.grid(True, alpha=STYLE["grid_alpha"] + 0.05)
    ax.legend(fontsize=FONT["legend"], bbox_to_anchor=(1.02, 1),
              loc="upper left")
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
            row["pop_type_code"] = 0
        elif "split_1" in pop_id:
            row["pop_type_code"] = 1
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
        row["stellar_mass"] = row.get("stellar_mass_prop",
                              row.get("stellar_mass_bin",
                              getattr(sed, "median_mass", np.nan)))
        row["l_uv"] = row.get("l_uv_prop", row.get("l_uv_bin", np.nan))
        row["beta_uv"] = row.get("beta_uv_prop", row.get("beta_uv_bin", np.nan))

        data_list.append(row)

    if not data_list:
        print("No valid temperature measurements found")
        return None, None

    df_plot = pd.DataFrame(data_list)

    # Filter quiescent
    if not add_qt:
        n_pre = len(df_plot)
        df_plot = df_plot[df_plot["pop_type_code"] == 0].copy()
        n_qt = n_pre - len(df_plot)
        if n_qt > 0:
            print(f"Filtered {n_qt} quiescent galaxies")

    if len(df_plot) == 0:
        print("No data after filtering")
        return None, None

    print(f"T-z plot: {len(df_plot)} populations (tier >= {min_tier})")
    print(f"  z: {df_plot['redshift'].min():.2f} - {df_plot['redshift'].max():.2f}")
    print(f"  T: {df_plot['T_rest_frame'].min():.0f} - {df_plot['T_rest_frame'].max():.0f} K")

    # ── color setup ──────────────────────────────────────────────────
    color_configs = {
        "stellar_mass": ("stellar_mass", "log₁₀(M*/M☉)", "viridis"),
        "l_ir":         ("l_ir",         "log₁₀(L_IR/L☉)", "plasma"),
        "l_uv":         ("l_uv",         "log₁₀(L_UV/L☉)", "viridis"),
        "beta_uv":      ("beta_uv",      "β_UV", "coolwarm"),
        "chi2":         ("chi2_reduced",  "χ²_red", "coolwarm"),
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
                np.ceil(np.nanmax(valid_vals)) + 0.5, 0.5,
            )
        elif color_by == "beta_uv":
            boundaries = np.array([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.5])
        else:
            n_disc = min(6, len(np.unique(valid_vals)))
            boundaries = np.linspace(np.nanmin(valid_vals), np.nanmax(valid_vals), n_disc + 1)

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
            df_plot["redshift"], df_plot["T_rest_frame"],
            c=color_data, cmap=discrete_cmap, norm=norm,
            s=80, alpha=0.85, edgecolors="k", linewidth=0.5, zorder=3,
        )

        if show_errors and "T_error" in df_plot.columns:
            ax.errorbar(
                df_plot["redshift"], df_plot["T_rest_frame"],
                yerr=df_plot["T_error"],
                fmt="none", color="gray", alpha=0.3,
                capsize=2, elinewidth=0.8, zorder=1,
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
            df_plot["redshift"], df_plot["T_rest_frame"],
            s=80, alpha=0.85, edgecolors="k", linewidth=0.5, zorder=3,
        )
        if show_errors and "T_error" in df_plot.columns:
            ax.errorbar(
                df_plot["redshift"], df_plot["T_rest_frame"],
                yerr=df_plot["T_error"],
                fmt="none", color="gray", alpha=0.3,
                capsize=2, elinewidth=0.8, zorder=1,
            )

    # ── literature ───────────────────────────────────────────────────
    if show_literature:
        z_lit = np.linspace(0, 6, 100)
        z_ext = np.linspace(0, 10, 150)

        T_schreiber = 32.9 + 4.60 * (z_lit - 2.0)
        ax.plot(z_lit, T_schreiber, "--", color="green", linewidth=2,
                label="Schreiber+18: T=23.7+4.6z", alpha=0.8)

        T_viero = 20.9 + 5.9 * z_ext + 0.5 * z_ext**2
        ax.plot(z_ext, T_viero, "--", color="purple", linewidth=2,
                label="Viero+22: T=20.9+5.9z+0.5z²", alpha=0.8)

        ax.axhline(32, color="gray", ls=":", lw=1, alpha=0.5,
                   label="Drew & Casey 22: no evol. at fixed L_IR")

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

            sigma_kw = dict(sigma=T_errors, absolute_sigma=True) if T_errors is not None else {}
            popt, pcov = curve_fit(
                quadratic, z_data, T_data, p0=[25, 3, 0.5],
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
            ax.plot(z_fit, quadratic(z_fit, *popt), "-", color="black",
                    linewidth=2.5, alpha=0.85,
                    label=f"This work: T={a:.0f}+{b:.1f}z+{c_coeff:.2f}z² (R²={r2:.2f})")

            print(f"Quadratic fit: T = {a:.1f}±{a_err:.1f} + "
                  f"{b:.1f}±{b_err:.1f}·z + {c_coeff:.2f}±{c_err:.2f}·z²  "
                  f"(R²={r2:.3f})")

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
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")

    return fig, df_plot



# ── IRX-β reference curves ───────────────────────────────────────────────

def _meurer99_irx(beta):
    """Meurer+99 IRX-β relation (starburst/MW-like)."""
    return 10 ** (0.4 * (4.43 + 1.99 * np.asarray(beta))) - 1


def _smc_irx(beta):
    """SMC-like IRX-β relation (McLure+2018 parameterization, dA/dβ=0.91)."""
    beta = np.asarray(beta)
    beta_int = -2.30
    BC_1600 = 1.51
    A1600 = 0.91 * (beta - beta_int)
    return (10 ** (0.4 * A1600) - 1) * BC_1600


# ── IRX-β plot ───────────────────────────────────────────────────────────

def create_lir_luv_beta_plot(
    wrapper,
    *,
    color_by: str = "redshift",
    average_luv: bool = False,
    min_tier: str = "A",
    show_errors: bool = True,
    figsize: tuple = (10, 8),
    save_path=None,
):
    """
    Create L_IR/L_UV vs β_UV plot from SimstackWrapper results.

    Parameters
    ----------
    wrapper : SimstackWrapper
        Must have stacking_results and processed_results.
    color_by : str
        'redshift' or 'l_uv'.
    average_luv : bool
        Average IRX over populations with the same β_UV and redshift
        (useful when L_UV varies within bins).
    min_tier : str
        Minimum fit quality tier ('A', 'B', 'C').
    show_errors : bool
        Show error bars when averaging.
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
        for key, val in props.items():
            kl = key.lower()
            if ("l_uv" in kl or "luv" in kl) and luv_val is None:
                luv_val = val
                luv_source = "bin_properties"
            elif "beta" in kl and beta_bin is None:
                beta_bin = val
                beta_source = "bin_properties"

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

        if beta_bin is None or luv_val is None:
            continue

        # L_IR from derived quantities
        derived = pr.derived_quantities.get(pop_id)
        l_ir = derived.total_ir_luminosity if derived else 0
        if l_ir <= 0:
            continue

        l_uv_linear = 10 ** luv_val
        irx = l_ir / l_uv_linear

        rows.append({
            "beta_uv": beta_bin,
            "l_uv": luv_val,
            "redshift": z_bin if z_bin is not None else sed.median_redshift,
            "irx": irx,
            "tier": tier,
            "_luv_source": luv_source,
            "_beta_source": beta_source,
        })

    if not rows:
        # Diagnostic: what did we find?
        sample_id = list(pr.sed_results.keys())[0] if pr.sed_results else "N/A"
        sample_sed = list(pr.sed_results.values())[0] if pr.sed_results else None
        sample_bins = _parse_bins(sample_id) if sample_id != "N/A" else {}
        sample_props = (sample_sed.bin_properties
                        if sample_sed and sample_sed.bin_properties else {})
        print(f"No populations with both β_UV and L_UV data")
        print(f"  Sample population ID: {sample_id}")
        print(f"  Parsed binning dims: {list(sample_bins.keys())}")
        print(f"  bin_properties keys: {list(sample_props.keys()) if isinstance(sample_props, dict) else type(sample_props).__name__}")
        print(f"  Need: 'beta' in binning dims + 'l_uv' in bin_properties or binning")
        return None

    # Report data sources
    n_luv_props = sum(1 for r in rows if r.get("_luv_source") == "bin_properties")
    n_beta_props = sum(1 for r in rows if r.get("_beta_source") == "bin_properties")
    n_total = len(rows)
    print(f"  β_UV source: {n_beta_props}/{n_total} from bin_properties, "
          f"{n_total - n_beta_props} from bin center")
    print(f"  L_UV source: {n_luv_props}/{n_total} from bin_properties, "
          f"{n_total - n_luv_props} from bin center")

    df = pd.DataFrame(rows)
    df = df.drop(columns=["_luv_source", "_beta_source"], errors="ignore")
    print(f"IRX-β: {len(df)} populations (tier ≥ {min_tier})")

    # ── optional L_UV averaging ──────────────────────────────────────
    if average_luv:
        grp = df.groupby(["beta_uv", "redshift"]).agg(
            irx_mean=("irx", "mean"),
            irx_std=("irx", "std"),
            l_uv=("l_uv", "mean"),
            n=("irx", "size"),
        ).reset_index()
        grp["irx_std"] = grp["irx_std"].fillna(0)
        beta_vals = grp["beta_uv"].values
        irx_vals = grp["irx_mean"].values
        irx_errs = grp["irx_std"].values
        color_vals = grp["redshift"].values if color_by == "redshift" else grp["l_uv"].values
        print(f"Averaged to {len(grp)} points over L_UV")
    else:
        beta_vals = df["beta_uv"].values
        irx_vals = df["irx"].values
        irx_errs = None
        color_vals = df["redshift"].values if color_by == "redshift" else df["l_uv"].values

    # ── plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    color_label = "Redshift" if color_by == "redshift" else "log L_UV/L☉"
    cmap_name = "plasma" if color_by == "redshift" else "viridis"

    valid = np.isfinite(color_vals)
    if np.any(valid):
        sc = ax.scatter(
            beta_vals, irx_vals, c=color_vals,
            cmap=cmap_name, s=80, alpha=0.85,
            edgecolors="k", linewidth=0.5, zorder=3,
        )
        cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(color_label, fontsize=13)

        if average_luv and show_errors and irx_errs is not None:
            ax.errorbar(
                beta_vals, irx_vals, yerr=irx_errs,
                fmt="none", color="gray", alpha=0.4,
                capsize=2, zorder=1,
            )
    else:
        ax.scatter(beta_vals, irx_vals, s=80, alpha=0.85,
                   edgecolors="k", linewidth=0.5, zorder=3)

    # ── reference curves ─────────────────────────────────────────────
    beta_model = np.linspace(-2.5, 1.5, 100)
    ax.plot(beta_model, _meurer99_irx(beta_model), "k--", lw=2, alpha=0.7,
            label="Meurer+99 (MW-like)")
    ax.plot(beta_model, _smc_irx(beta_model), "-.", color="brown", lw=2,
            alpha=0.7, label="SMC-like (McLure+18)")

    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)

    ax.set_xlabel("β_UV", fontsize=14)
    ax.set_ylabel("IRX  (L_IR / L_UV)", fontsize=14)
    ax.set_yscale("log")
    ax.set_xlim(-2.2, 1.2)
    ax.set_ylim(10 ** (-1.2), 10 ** 3.5)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"IRX–β  ({len(df)} populations, tier ≥ {min_tier})", fontsize=15)

    plt.tight_layout()

    if save_path is not None:
        from pathlib import Path as _P
        save_path = _P(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dpi = 300 if save_path.suffix.lower() in [".png", ".jpg"] else 150
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight",
                    facecolor="white", edgecolor="none")

    return fig
