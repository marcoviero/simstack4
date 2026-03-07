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

    # With Schreiber prior overlay for diagnostics:
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
    "marker_edgewidth":     1.5,
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
    "sfg":        cm.Blues,
    "quiescent":  cm.Reds,
    "agn":        cm.Greens,
    "dusty":      cm.Oranges,
    "split_0":    cm.Blues,
    "split_1":    cm.Reds,
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
    show_prior: bool = False,
    show_tier: bool = False,
    flux_unit: str = "mJy",
    ylim: tuple[float, float] = (1e-2, 5e2),
    figsize: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
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
        Overlay the Schreiber+2015 prior SED (grey dashed) with
        amplitude scaled to best-fit the data.  Useful for diagnosing
        whether the prior is sensible for each bin.
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
                colour_map[k] = cmap(0.35 + 0.55 * j / max(n - 1, 1))
    else:
        n = max(len(colour_keys), 1)
        for j, k in enumerate(colour_keys):
            colour_map[k] = cm.viridis(0.15 + 0.75 * j / max(n - 1, 1))

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

                    # ── Schreiber prior overlay ──────────────────────
                    if show_prior and not prior_plotted:
                        z = sed.median_redshift
                        beta = (sed.emissivity_index or 1.8)
                        try:
                            T_prior, _ = (
                                pr.greybody_fitter
                                .schreiber_temperature_prior(z)
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
                    # Retrieve the Schreiber T for the label
                    _first_sed = sed_results[cell_pops[0]]
                    _T_sch, _ = pr.greybody_fitter.schreiber_temperature_prior(
                        _first_sed.median_redshift
                    )
                    legend_handles.append(
                        Line2D([0], [0], color=STYLE["prior_color"],
                               lw=STYLE["prior_lw"], ls="--")
                    )
                    legend_labels.append(f"Schreiber T_rf={_T_sch:.0f}K")

                if legend_handles:
                    ax.legend(
                        legend_handles, legend_labels,
                        loc="upper left", fontsize=FONT["legend"],
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
                        .schreiber_temperature_prior(z)
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
# T_RF VS REDSHIFT PLOT FOR SIMSTACK4 RESULTS
# Extracts rest-frame dust temperatures and redshifts from wrapper results
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm


def create_trf_redshift_plot(wrapper,
                             color_by='l_ir',
                             size_by='n_sources',
                             show_errors=True,
                             show_literature=True,
                             add_qt=False,
                             fit_data=True,
                             figsize=(10, 8),
                             save_path=None):
    """
    Create T_rf vs redshift plot from SimstackWrapper results

    Parameters:
    -----------
    wrapper : SimstackWrapper
        Wrapper object with stacking_results and processed_results
    color_by : str
        What to color points by: 'l_ir', 'stellar_mass', 'beta_uv', 'population_type', 'chi2'
    size_by : str
        What to size points by: 'n_sources', 'l_ir', 'stellar_mass'
    show_errors : bool
        Whether to show error bars on temperature
    show_literature : bool
        Whether to show literature comparison lines
    add_qt : bool
        Whether to include quiescent galaxies (default: False, star-forming only)
    fit_data : bool
        Whether to fit and show power-law relation to the data
    figsize : tuple
        Figure size (width, height)
    save_path : str or Path or None
        Path to save the figure

    Returns:
    --------
    fig : matplotlib figure
        The created figure
    df_plot : pandas DataFrame
        The data used for plotting
    """

    # Check if we have the required data
    if not hasattr(wrapper, 'processed_results') or wrapper.processed_results is None:
        print("❌ No processed results found in wrapper")
        return None, None

    if not hasattr(wrapper.processed_results, 'sed_results'):
        print("❌ No SED results found in processed results")
        return None, None

    print("✅ Found processed results with SED data")

    # Extract data from SED results
    data_list = []

    for pop_id, sed_result in wrapper.processed_results.sed_results.items():
        # Skip if no greybody fit or no temperature
        if not sed_result.greybody_fit_success:
            continue

        if sed_result.dust_temperature_rest_frame is None:
            continue

        # Get basic data
        row = {
            'population_id': pop_id,
            'redshift': sed_result.median_redshift,
            'T_rest_frame': sed_result.dust_temperature_rest_frame,
            'T_observed_frame': sed_result.dust_temperature_observed_frame,
            'T_error': sed_result.dust_temperature_error,
            'n_sources': sed_result.n_sources,
            'stellar_mass': sed_result.median_mass,
            'chi2_reduced': sed_result.chi2_reduced,
            'beta': sed_result.emissivity_index,
            'amplitude': sed_result.amplitude,
        }

        # Get derived quantities
        if pop_id in wrapper.processed_results.derived_quantities:
            derived = wrapper.processed_results.derived_quantities[pop_id]
            row['l_ir'] = derived.total_ir_luminosity
            row['l_ir_error'] = derived.total_ir_luminosity_error
            row['sfr'] = derived.star_formation_rate
            row['dust_mass'] = derived.dust_mass
        else:
            row['l_ir'] = 0
            row['l_ir_error'] = 0
            row['sfr'] = 0
            row['dust_mass'] = None

        # Parse population type from ID
        if 'split_0' in pop_id:
            row['population_type'] = 'Star-forming'
            row['pop_type_code'] = 0
        elif 'split_1' in pop_id:
            row['population_type'] = 'Quiescent'
            row['pop_type_code'] = 1
        else:
            row['population_type'] = 'Unknown'
            row['pop_type_code'] = -1

        # Extract other bin information
        try:
            parts = pop_id.split('__')
            for part in parts:
                if 'stellar_mass' in part or 'mass' in part:
                    elements = part.split('_')
                    if len(elements) >= 3:
                        try:
                            mass_min = float(elements[-2])
                            mass_max = float(elements[-1])
                            row['mass_bin_center'] = (mass_min + mass_max) / 2
                        except ValueError:
                            pass
                elif 'redshift' in part or '_z_' in part:
                    elements = part.split('_')
                    if len(elements) >= 3:
                        try:
                            z_min = float(elements[-2])
                            z_max = float(elements[-1])
                            row['z_bin_center'] = (z_min + z_max) / 2
                        except ValueError:
                            pass
        except:
            pass

        data_list.append(row)

    if not data_list:
        print("❌ No valid temperature measurements found")
        return None, None

    # Create DataFrame
    df_plot = pd.DataFrame(data_list)

    # Filter by population type if requested
    if not add_qt:
        # Keep only star-forming galaxies (default behavior)
        initial_count = len(df_plot)
        df_plot = df_plot[df_plot['pop_type_code'] == 0].copy()
        quiescent_count = initial_count - len(df_plot)
        if quiescent_count > 0:
            print(f"🔍 Filtered out {quiescent_count} quiescent galaxies (use add_qt=True to include)")

    print(f"📊 Found {len(df_plot)} populations with valid T_rest measurements")
    if len(df_plot) == 0:
        print("❌ No data remaining after filtering")
        return None, None

    print(f"   Redshift range: {df_plot['redshift'].min():.2f} - {df_plot['redshift'].max():.2f}")
    print(f"   T_rest range: {df_plot['T_rest_frame'].min():.1f} - {df_plot['T_rest_frame'].max():.1f} K")

    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    # Set up coloring
    if color_by == 'l_ir':
        color_data = np.log10(df_plot['l_ir'].replace(0, np.nan))
        color_label = 'log₁₀(L_IR/L☉)'
        cmap = 'plasma'
        vmin, vmax = None, None
    elif color_by == 'stellar_mass':
        color_data = df_plot['stellar_mass']
        color_label = 'log₁₀(M*/M☉)'
        cmap = 'viridis'
        vmin, vmax = None, None
    elif color_by == 'population_type':
        color_data = df_plot['pop_type_code']
        color_label = 'Population Type'
        cmap = 'RdBu'
        vmin, vmax = -0.5, 1.5
    elif color_by == 'chi2':
        color_data = df_plot['chi2_reduced']
        color_label = 'χ²_reduced'
        cmap = 'coolwarm'
        vmin, vmax = 0.5, 2.0
    elif color_by == 'beta_uv':
        # Try to get beta_UV from population manager if available
        color_data = df_plot['beta']
        color_label = 'β (emissivity index)'
        cmap = 'coolwarm'
        vmin, vmax = 1.0, 2.5
    else:
        color_data = 'blue'
        color_label = None
        cmap = None
        vmin, vmax = None, None

    # Set up sizing
    if size_by == 'n_sources':
        size_data = df_plot['n_sources']
        # Scale sizes: min=20, max=200
        size_min, size_max = 20, 200
        sizes = size_min + (size_max - size_min) * (size_data - size_data.min()) / (size_data.max() - size_data.min())
        size_label = 'N_sources'
    elif size_by == 'l_ir':
        size_data = np.log10(df_plot['l_ir'].replace(0, 1e8))
        size_min, size_max = 20, 200
        sizes = size_min + (size_max - size_min) * (size_data - size_data.min()) / (size_data.max() - size_data.min())
        size_label = 'log₁₀(L_IR/L☉)'
    elif size_by == 'stellar_mass':
        size_data = df_plot['stellar_mass']
        size_min, size_max = 20, 200
        sizes = size_min + (size_max - size_min) * (size_data - size_data.min()) / (size_data.max() - size_data.min())
        size_label = 'log₁₀(M*/M☉)'
    else:
        sizes = 50
        size_label = None

    # pdb.set_trace()
    # Make the scatter plot
    if show_errors and 'T_error' in df_plot.columns:
        # Plot with error bars
        if isinstance(color_data, str):
            # Single color
            scatter = ax.errorbar(df_plot['redshift'], df_plot['T_rest_frame'],
                                  yerr=df_plot['T_error'],
                                  fmt='o', color=color_data, markersize=8,
                                  capsize=3, alpha=0.7, elinewidth=1)
        else:
            # Color-coded points with error bars (more complex)
            scatter = ax.scatter(df_plot['redshift'], df_plot['T_rest_frame'],
                                 c=color_data,  # s=sizes, alpha=0.7,
                                 cmap=cmap, vmin=vmin, vmax=vmax,
                                 edgecolors='black', linewidth=0.5)
            # pdb.set_trace()
            # Add error bars separately
            ax.errorbar(df_plot['redshift'], df_plot['T_rest_frame'],
                        yerr=df_plot['T_error'],
                        fmt='none', color='gray', alpha=0.3,
                        capsize=2, elinewidth=0.8)
    else:
        # Plot without error bars
        if isinstance(color_data, str):
            scatter = ax.scatter(df_plot['redshift'], df_plot['T_rest_frame'],
                                 s=sizes, color=color_data, alpha=0.7,
                                 edgecolors='black', linewidth=0.5)
        else:
            scatter = ax.scatter(df_plot['redshift'], df_plot['T_rest_frame'],
                                 c=color_data, s=sizes, alpha=0.7,
                                 cmap=cmap, vmin=vmin, vmax=vmax,
                                 edgecolors='black', linewidth=0.5)

    # Add colorbar if needed
    if not isinstance(color_data, str) and color_label:
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
        cbar.set_label(color_label, fontsize=12)

        # Special handling for population type
        if color_by == 'population_type':
            cbar.set_ticks([0, 1])
            cbar.set_ticklabels(['Star-forming', 'Quiescent'])

    # Add literature relations if requested
    if show_literature:
        z_lit = np.linspace(0, 6, 100)
        z_lit_v22 = np.linspace(0, 9, 100)

        # Casey et al. 2018: T_d = 25 * (1+z)^0.36 K
        T_casey = 25 * (1 + z_lit) ** 0.36
        ax.plot(z_lit, T_casey, '--', color='red', linewidth=2,
                label='Casey+18: T∝(1+z)^0.36', alpha=0.8)

        # Magnelli et al. 2014: T_d = 32.9 + 4.6*z K
        T_magnelli = 32.9 + 4.6 * z_lit
        ax.plot(z_lit, T_magnelli, '--', color='orange', linewidth=2,
                label='Magnelli+14: T=32.9+4.6z', alpha=0.8)

        # Schreiber et al. 2018: T_d ≈ 24 * (1+z)^0.32 K
        T_schreiber = 24 * (1 + z_lit) ** 0.32
        ax.plot(z_lit, T_schreiber, '--', color='green', linewidth=2,
                label='Schreiber+18: T∝(1+z)^0.32', alpha=0.8)

        # Viero et al. 2022: Near exponential increase reaching ~100K at z~7
        # From abstract: "increase of dust temperature with redshift, reaching 100±12 K at z ~ 7"
        # This suggests a steep rise. Let's approximate as T_d = 25 * (1+z)^0.6
        T_viero = 20.9 + (5.9 * z_lit_v22) + 0.5 * z_lit_v22 ** 2
        ax.plot(z_lit_v22, T_viero, '--', color='purple', linewidth=2,
                label='Viero+22: T∝(1+z)^0.6', alpha=0.8)

    # Fit data if requested
    if fit_data and len(df_plot) >= 3:
        try:
            from scipy.optimize import curve_fit

            # Define fitting functions
            def power_law(z, T0, alpha):
                return T0 * (1 + z) ** alpha

            def linear_law(z, T0, slope):
                return T0 + slope * z

            # Get data for fitting
            z_data = df_plot['redshift'].values
            T_data = df_plot['T_rest_frame'].values
            T_errors = df_plot['T_error'].values if 'T_error' in df_plot.columns else None

            # Fit power law: T = T0 * (1+z)^alpha
            try:
                if T_errors is not None and np.any(T_errors > 0):
                    popt_power, pcov_power = curve_fit(power_law, z_data, T_data,
                                                       sigma=T_errors, p0=[25, 0.4],
                                                       bounds=([10, 0], [50, 2]))
                else:
                    popt_power, pcov_power = curve_fit(power_law, z_data, T_data,
                                                       p0=[25, 0.4],
                                                       bounds=([10, 0], [50, 2]))

                T0_power, alpha_power = popt_power
                T0_err, alpha_err = np.sqrt(np.diag(pcov_power))

                # Calculate R-squared
                T_pred_power = power_law(z_data, T0_power, alpha_power)
                ss_res_power = np.sum((T_data - T_pred_power) ** 2)
                ss_tot = np.sum((T_data - np.mean(T_data)) ** 2)
                r2_power = 1 - (ss_res_power / ss_tot)

                # Plot fit
                z_fit = np.linspace(z_data.min(), z_data.max() * 1.1, 100)
                T_fit_power = power_law(z_fit, T0_power, alpha_power)
                ax.plot(z_fit, T_fit_power, '-', color='black', linewidth=3, alpha=0.8,
                        label=f'This work: T={T0_power:.1f}(1+z)^{alpha_power:.2f} (R²={r2_power:.2f})')

                print(f"🔬 Power-law fit: T = {T0_power:.1f}±{T0_err:.1f} × (1+z)^{alpha_power:.2f}±{alpha_err:.2f}")
                print(f"   R² = {r2_power:.3f}")

            except Exception as e:
                print(f"⚠️  Power-law fit failed: {e}")

            # Also try linear fit: T = T0 + slope*z
            try:
                if T_errors is not None and np.any(T_errors > 0):
                    popt_linear, pcov_linear = curve_fit(linear_law, z_data, T_data,
                                                         sigma=T_errors, p0=[30, 5])
                else:
                    popt_linear, pcov_linear = curve_fit(linear_law, z_data, T_data,
                                                         p0=[30, 5])

                T0_linear, slope_linear = popt_linear
                T0_linear_err, slope_linear_err = np.sqrt(np.diag(pcov_linear))

                # Calculate R-squared
                T_pred_linear = linear_law(z_data, T0_linear, slope_linear)
                ss_res_linear = np.sum((T_data - T_pred_linear) ** 2)
                r2_linear = 1 - (ss_res_linear / ss_tot)

                print(
                    f"🔬 Linear fit: T = {T0_linear:.1f}±{T0_linear_err:.1f} + {slope_linear:.1f}±{slope_linear_err:.1f}×z")
                print(f"   R² = {r2_linear:.3f}")

                # Add best fit info to legend
                if r2_power > r2_linear:
                    print(f"📈 Power-law fit is better (R² = {r2_power:.3f} vs {r2_linear:.3f})")
                else:
                    print(f"📈 Linear fit is better (R² = {r2_linear:.3f} vs {r2_power:.3f})")

            except Exception as e:
                print(f"⚠️  Linear fit failed: {e}")

        except ImportError:
            print("⚠️  scipy not available for fitting")
        except Exception as e:
            print(f"⚠️  Fitting failed: {e}")

    # Formatting
    ax.set_xlabel('Redshift', fontsize=14)
    ax.set_ylabel('Rest-frame Dust Temperature (K)', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3)

    # Set reasonable limits
    ax.set_xlim(0, df_plot['redshift'].max() * 1.1)
    ax.set_ylim(df_plot['T_rest_frame'].min() * 0.9,
                min(df_plot['T_rest_frame'].max() * 1.1, 150))

    # Title and legend
    title = f'Rest-frame Dust Temperature vs Redshift'
    if len(df_plot) > 1:
        title += f' ({len(df_plot)} populations)'
    ax.set_title(title, fontsize=16, pad=20)

    if show_literature:
        ax.legend(loc='upper left', fontsize=10, framealpha=0.8)

    # Add size legend if varying sizes
    if size_label and not isinstance(sizes, (int, float)):
        # Create size legend
        size_legend_values = [size_data.min(), size_data.median(), size_data.max()]
        size_legend_sizes = [size_min, (size_min + size_max) / 2, size_max]
        size_legend_labels = [f'{v:.0f}' if size_by == 'n_sources' else f'{v:.1f}'
                              for v in size_legend_values]

        # Add size legend as text
        ax.text(0.02, 0.98, f'Marker size: {size_label}',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        try:
            from pathlib import Path
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            dpi = 300 if save_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] else 150
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"💾 Figure saved to: {save_path}")

        except Exception as e:
            print(f"⚠️  Could not save figure: {e}")

    return fig, df_plot


def create_temperature_evolution_plot(wrapper,
                                      split_by_type=True,
                                      show_individual=True,
                                      show_literature=True,
                                      add_qt=False,
                                      fit_data=True,
                                      figsize=(12, 8),
                                      save_path=None):
    """
    Create a comprehensive temperature evolution plot with multiple panels

    Parameters:
    -----------
    wrapper : SimstackWrapper
        Wrapper object with results
    split_by_type : bool
        Whether to separate star-forming and quiescent galaxies
    show_individual : bool
        Whether to show individual population points
    show_literature : bool
        Whether to show literature comparison
    add_qt : bool
        Whether to include quiescent galaxies (default: False)
    fit_data : bool
        Whether to fit relations to the data
    figsize : tuple
        Figure size
    save_path : str or Path or None
        Path to save figure
    """

    # Get data using the main function
    fig_temp, df_plot = create_trf_redshift_plot(wrapper, color_by='population_type',
                                                 show_errors=True, show_literature=False,
                                                 add_qt=add_qt, fit_data=False)
    plt.close(fig_temp)  # Close the temporary figure

    if df_plot is None:
        print("❌ No data available for temperature evolution plot")
        return None, None

    # Create comprehensive plot
    if split_by_type and 'pop_type_code' in df_plot.columns:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)
        axes = [ax1, ax2]

        # Separate by population type
        sf_data = df_plot[df_plot['pop_type_code'] == 0]  # Star-forming
        q_data = df_plot[df_plot['pop_type_code'] == 1]  # Quiescent

        datasets = [sf_data, q_data]
        colors = ['#FF6B6B', '#4ECDC4']  # Red for SF, Teal for Q
        labels = ['Star-forming', 'Quiescent']

    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)
        axes = [ax1]
        datasets = [df_plot]
        colors = ['#45B7D1']
        labels = ['All galaxies']

    # Plot each dataset
    for ax, data, color, label in zip(axes, datasets, colors, labels):
        if len(data) == 0:
            ax.text(0.5, 0.5, f'No {label.lower()} data',
                    ha='center', va='center', transform=ax.transAxes)
            continue

        # Individual points
        if show_individual:
            sizes = 50 + 150 * (data['n_sources'] - data['n_sources'].min()) / (
                        data['n_sources'].max() - data['n_sources'].min())

            scatter = ax.scatter(data['redshift'], data['T_rest_frame'],
                                 s=sizes, color=color, alpha=0.7,
                                 edgecolors='black', linewidth=0.5,
                                 label=f'{label} (N={len(data)})')

            # Error bars
            if 'T_error' in data.columns:
                ax.errorbar(data['redshift'], data['T_rest_frame'],
                            yerr=data['T_error'],
                            fmt='none', color='gray', alpha=0.3,
                            capsize=2, elinewidth=0.8)

        # Bin and plot trends if enough data
        if len(data) >= 4:
            # Bin by redshift
            z_bins = np.linspace(data['redshift'].min(), data['redshift'].max(), 4)
            z_centers = []
            T_means = []
            T_errors = []

            for i in range(len(z_bins) - 1):
                mask = (data['redshift'] >= z_bins[i]) & (data['redshift'] < z_bins[i + 1])
                if np.sum(mask) > 0:
                    z_centers.append((z_bins[i] + z_bins[i + 1]) / 2)
                    T_means.append(data.loc[mask, 'T_rest_frame'].mean())
                    T_errors.append(data.loc[mask, 'T_rest_frame'].std() / np.sqrt(np.sum(mask)))

            if len(z_centers) > 1:
                ax.errorbar(z_centers, T_means, yerr=T_errors,
                            fmt='s-', color=color, markersize=8, linewidth=2,
                            capsize=4, label=f'{label} (binned trend)')

        # Literature comparison
        if show_literature:
            z_lit = np.linspace(0, 6, 100)

            # Casey et al. 2018
            T_casey = 25 * (1 + z_lit) ** 0.36
            ax.plot(z_lit, T_casey, '--', color='gray', linewidth=1.5,
                    alpha=0.8, label='Casey+18' if ax == axes[0] else '')

            # Magnelli et al. 2014
            T_magnelli = 32.9 + 4.6 * z_lit
            ax.plot(z_lit, T_magnelli, ':', color='gray', linewidth=1.5,
                    alpha=0.8, label='Magnelli+14' if ax == axes[0] else '')

            # Viero et al. 2022
            T_viero = 25 * (1 + z_lit) ** 0.6
            T_viero = 20.9 + (5.9 * z_lit) + 0.5 * z_lit ** 2
            print(T_viero)
            ax.plot(z_lit, T_viero, '-', color='purple', linewidth=1.5,
                    alpha=0.8, label='Viero+22' if ax == axes[0] else '')

        # Formatting
        ax.set_xlabel('Redshift', fontsize=14)
        if ax == axes[0]:
            ax.set_ylabel('Rest-frame Dust Temperature (K)', fontsize=14)
        ax.set_title(f'{label}', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        # Set limits
        if len(data) > 0:
            ax.set_xlim(0, max(6, data['redshift'].max() * 1.1))
            ax.set_ylim(20, 90)

    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        try:
            from pathlib import Path
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            dpi = 300 if save_path.suffix.lower() in ['.png', '.jpg', '.jpeg'] else 150
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"💾 Temperature evolution plot saved to: {save_path}")

        except Exception as e:
            print(f"⚠️  Could not save figure: {e}")

    return fig, df_plot


# Usage example:
def plot_temperature_evolution_example(wrapper, add_qt=False):
    """
    Example usage of the temperature plotting functions
    """
    print("Creating T_rf vs redshift plots...")

    # Basic plot colored by L_IR (star-forming only by default)
    fig1, df1 = create_trf_redshift_plot(wrapper,
                                         color_by='l_ir',
                                         size_by='n_sources',
                                         show_errors=True,
                                         add_qt=add_qt,
                                         fit_data=True,
                                         save_path='temperature_vs_redshift_basic.png')

    # Plot colored by population type (if including quiescent)
    if add_qt:
        fig2, df2 = create_trf_redshift_plot(wrapper,
                                             color_by='population_type',
                                             size_by='stellar_mass',
                                             add_qt=True,
                                             fit_data=True,
                                             save_path='temperature_vs_redshift_by_type.png')
    else:
        fig2, df2 = create_trf_redshift_plot(wrapper,
                                             color_by='stellar_mass',
                                             size_by='n_sources',
                                             add_qt=False,
                                             fit_data=True,
                                             save_path='temperature_vs_redshift_sf_only.png')

    # Comprehensive evolution plot
    fig3, df3 = create_temperature_evolution_plot(wrapper,
                                                  split_by_type=add_qt,
                                                  show_individual=True,
                                                  show_literature=True,
                                                  add_qt=add_qt,
                                                  fit_data=True,
                                                  save_path='temperature_evolution_comprehensive.png')

    print("✅ Temperature evolution plots created!")

    return fig1, fig2, fig3, df1


class SimstackPlots:
    """
    Create plots and visualizations from stacking results

    This class generates publication-quality plots from saved SimstackResults,
    maintaining modularity by working with saved results rather than requiring
    the full pipeline.
    """

    def __init__(self, results_object: Optional[SimstackResults] = None,
                 results_file: Optional[Path] = None, style: str = 'default'):
        """
        Initialize plotting class

        Args:
            results_object: Processed SimstackResults object
            results_file: Path to saved results file
            style: Plotting style ('default', 'publication', 'presentation')
        """
        if not HAS_MATPLOTLIB:
            raise PlotError("matplotlib is required for plotting")

        # Load results
        if results_object is not None:
            self.results = results_object
        elif results_file is not None:
            self.results = SimstackResults.load_results(results_file)
        else:
            raise PlotError("Either results_object or results_file must be provided")

        # Set up plotting style
        self.style = style
        self._setup_plotting_style()

        logger.info(f"SimstackPlots initialized with {len(self.results.sed_results)} populations")

    def _setup_plotting_style(self) -> None:
        """Set up matplotlib plotting style"""
        if self.style == 'publication':
            # Publication-ready style
            plt.style.use('seaborn-v0_8-whitegrid' if HAS_SEABORN else 'seaborn-whitegrid')
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 11,
                'figure.titlesize': 18,
                'lines.linewidth': 2,
                'lines.markersize': 6,
                'figure.figsize': [8, 6],
                'savefig.dpi': 300,
                'savefig.bbox': 'tight'
            })
        elif self.style == 'presentation':
            # Presentation style (larger fonts)
            plt.rcParams.update({
                'font.size': 16,
                'axes.labelsize': 18,
                'axes.titlesize': 20,
                'xtick.labelsize': 16,
                'ytick.labelsize': 16,
                'legend.fontsize': 14,
                'figure.titlesize': 22,
                'lines.linewidth': 3,
                'lines.markersize': 8,
                'figure.figsize': [10, 8]
            })

    def plot_stacked_seds(self, population_ids: Optional[List[str]] = None,
                         normalize_by_sources: bool = False,
                         show_errors: bool = True,
                         save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot stacked SEDs for selected populations

        Args:
            population_ids: List of population IDs to plot (None for all)
            normalize_by_sources: Whether to normalize by number of sources
            show_errors: Whether to show error bars
            save_path: Path to save plot

        Returns:
            Figure object
        """
        if population_ids is None:
            population_ids = list(self.results.sed_results.keys())

        fig, ax = plt.subplots(figsize=(10, 8))

        # Define colors for different populations
        colors = plt.cm.tab10(np.linspace(0, 1, len(population_ids)))

        for i, pop_id in enumerate(population_ids):
            if pop_id not in self.results.sed_results:
                logger.warning(f"Population {pop_id} not found in results")
                continue

            sed = self.results.sed_results[pop_id]

            # Get plotting data
            wavelengths = sed.wavelengths
            flux_densities = sed.flux_densities.copy()
            flux_errors = sed.flux_errors.copy()

            # Normalize by number of sources if requested
            if normalize_by_sources and sed.n_sources > 0:
                flux_densities /= sed.n_sources
                flux_errors /= sed.n_sources

            # Create label
            label = f"{pop_id} (N={sed.n_sources})"
            if len(label) > 40:
                label = label[:37] + "..."

            # Plot
            if show_errors:
                ax.errorbar(wavelengths, flux_densities, yerr=flux_errors,
                           color=colors[i], marker='o', linestyle='-',
                           label=label, capsize=3, capthick=1)
            else:
                ax.plot(wavelengths, flux_densities, color=colors[i],
                       marker='o', linestyle='-', label=label)

        # Formatting
        ax.set_xlabel('Observed Wavelength [μm]')
        ylabel = 'Flux Density [Jy'
        if normalize_by_sources:
            ylabel += '/source'
        ylabel += ']'
        ax.set_ylabel(ylabel)
        ax.set_title('Stacked Spectral Energy Distributions')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SED plot saved to {save_path}")

        return fig

    def plot_luminosity_functions(self, rest_wavelength: float = 100.0,
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot rest-frame luminosity functions

        Args:
            rest_wavelength: Rest-frame wavelength to use (microns)
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get luminosities at specified wavelength
        luminosities = []
        redshifts = []
        masses = []
        pop_labels = []

        for pop_id, sed in self.results.sed_results.items():
            # Find closest wavelength
            wave_idx = np.argmin(np.abs(sed.wavelengths - rest_wavelength))
            closest_wave = sed.wavelengths[wave_idx]

            if abs(closest_wave - rest_wavelength) < rest_wavelength * 0.5:  # Within 50%
                lum = sed.rest_luminosities[wave_idx]
                if lum > 0:
                    luminosities.append(lum)
                    redshifts.append(sed.median_redshift)
                    masses.append(sed.median_mass)
                    pop_labels.append(pop_id)

        if not luminosities:
            logger.warning(f"No valid luminosities found at {rest_wavelength}μm")
            return fig

        luminosities = np.array(luminosities)
        redshifts = np.array(redshifts)
        masses = np.array(masses)

        # Create scatter plot colored by redshift
        scatter = ax.scatter(masses, luminosities, c=redshifts,
                           s=60, alpha=0.7, cmap='viridis')

        # Formatting
        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel(f'L_{rest_wavelength:.0f}μm [L☉]')
        ax.set_title(f'Rest-frame {rest_wavelength:.0f}μm Luminosity vs Stellar Mass')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Median Redshift')

        # Add population labels for highest luminosities
        if len(luminosities) <= 10:  # Only if not too crowded
            for i, (mass, lum, label) in enumerate(zip(masses, luminosities, pop_labels)):
                ax.annotate(label[:10], (mass, lum), xytext=(5, 5),
                          textcoords='offset points', fontsize=8, alpha=0.7)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Luminosity function plot saved to {save_path}")

        return fig

    def plot_sfr_vs_mass(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot star formation rate vs stellar mass (main sequence)

        Args:
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get data from results
        summary_df = self.results.get_population_summary()

        # Filter for valid SFR measurements
        valid_mask = (summary_df['sfr_msun_yr'] > 0) & (summary_df['median_log_mass'] > 0)
        plot_data = summary_df[valid_mask]

        if len(plot_data) == 0:
            logger.warning("No valid SFR/mass data found")
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=16)
            return fig

        # Create scatter plot
        masses = plot_data['median_log_mass']
        sfrs = plot_data['sfr_msun_yr']
        n_sources = plot_data['n_sources']

        # Size points by number of sources
        sizes = 30 + 100 * (n_sources / n_sources.max())

        scatter = ax.scatter(masses, sfrs, s=sizes, alpha=0.7,
                           c=plot_data['median_redshift'], cmap='plasma')

        # Plot main sequence relation (Whitaker et al. 2012)
        mass_range = np.linspace(masses.min(), masses.max(), 100)
        ms_sfr = 10**(0.7 * (mass_range - 10.5) - 0.13)  # Simplified relation
        ax.plot(mass_range, ms_sfr, 'k--', alpha=0.5, linewidth=2,
               label='Main Sequence (z~1)')

        # Formatting
        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel('SFR [M☉/yr]')
        ax.set_title('Star Formation Rate vs Stellar Mass')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Median Redshift')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"SFR vs mass plot saved to {save_path}")

        return fig

    def plot_fit_quality(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot fit quality metrics across bands

        Args:
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        # Extract fit quality data
        map_names = list(self.results.band_results.keys())
        wavelengths = [self.results.band_results[name]['wavelength_um'] for name in map_names]
        chi2_values = [self.results.band_results[name]['reduced_chi_squared'] for name in map_names]
        chi2_abs = [self.results.band_results[name]['chi_squared'] for name in map_names]

        # Plot 1: Reduced chi-squared vs wavelength
        ax1.scatter(wavelengths, chi2_values, s=80, alpha=0.7, color='steelblue')
        ax1.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Perfect fit')
        ax1.axhline(y=2, color='orange', linestyle='--', alpha=0.7, label='Acceptable fit')

        ax1.set_xlabel('Wavelength [μm]')
        ax1.set_ylabel('Reduced χ²')
        ax1.set_title('Fit Quality by Wavelength')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Add band labels
        for wave, chi2, name in zip(wavelengths, chi2_values, map_names):
            ax1.annotate(name, (wave, chi2), xytext=(5, 5),
                        textcoords='offset points', fontsize=9, alpha=0.8)

        # Plot 2: Absolute chi-squared
        ax2.bar(range(len(map_names)), chi2_abs, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Band')
        ax2.set_ylabel('χ²')
        ax2.set_title('Absolute Chi-squared by Band')
        ax2.set_xticks(range(len(map_names)))
        ax2.set_xticklabels([f"{name}\n{wave:.0f}μm" for name, wave in zip(map_names, wavelengths)],
                           rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Fit quality plot saved to {save_path}")

        return fig

    def plot_population_comparison(self, populations: List[str],
                                 metric: str = 'flux_densities',
                                 save_path: Optional[Path] = None) -> plt.Figure:
        """
        Compare specific populations across bands

        Args:
            populations: List of population IDs to compare
            metric: Metric to compare ('flux_densities', 'rest_luminosities')
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        colors = plt.cm.Set1(np.linspace(0, 1, len(populations)))

        for i, pop_id in enumerate(populations):
            if pop_id not in self.results.sed_results:
                logger.warning(f"Population {pop_id} not found")
                continue

            sed = self.results.sed_results[pop_id]

            if metric == 'flux_densities':
                y_data = sed.flux_densities
                y_errors = sed.flux_errors
                ylabel = 'Flux Density [Jy]'
            elif metric == 'rest_luminosities':
                y_data = sed.rest_luminosities
                y_errors = sed.rest_luminosity_errors
                ylabel = 'Rest Luminosity [L☉]'
            else:
                raise PlotError(f"Unknown metric: {metric}")

            # Create label
            label = f"{pop_id} (z={sed.median_redshift:.2f}, N={sed.n_sources})"
            if len(label) > 50:
                label = label[:47] + "..."

            ax.errorbar(sed.wavelengths, y_data, yerr=y_errors,
                       color=colors[i], marker='o', linestyle='-',
                       label=label, capsize=3, markersize=6)

        ax.set_xlabel('Wavelength [μm]')
        ax.set_ylabel(ylabel)
        ax.set_title(f'Population Comparison: {metric.replace("_", " ").title()}')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Population comparison plot saved to {save_path}")

        return fig

    def plot_redshift_evolution(self, quantity: str = 'sfr',
                               save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot evolution of quantities with redshift

        Args:
            quantity: Quantity to plot ('sfr', 'total_ir_luminosity', 'specific_sfr')
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Get data
        summary_df = self.results.get_population_summary()

        # Map quantity names
        quantity_map = {
            'sfr': ('sfr_msun_yr', 'SFR [M☉/yr]'),
            'total_ir_luminosity': ('total_ir_luminosity_lsun', 'L_IR [L☉]'),
            'specific_sfr': ('specific_sfr_yr', 'sSFR [yr⁻¹]')
        }

        if quantity not in quantity_map:
            raise PlotError(f"Unknown quantity: {quantity}")

        col_name, ylabel = quantity_map[quantity]

        # Filter valid data
        valid_mask = (summary_df[col_name] > 0) & (summary_df['median_redshift'] > 0)
        plot_data = summary_df[valid_mask]

        if len(plot_data) == 0:
            logger.warning(f"No valid data for {quantity}")
            return fig

        # Color code by stellar mass
        masses = plot_data['median_log_mass']
        redshifts = plot_data['median_redshift']
        values = plot_data[col_name]
        n_sources = plot_data['n_sources']

        # Size by number of sources
        sizes = 30 + 100 * (n_sources / n_sources.max())

        scatter = ax.scatter(redshifts, values, c=masses, s=sizes,
                           alpha=0.7, cmap='viridis')

        ax.set_xlabel('Redshift')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel.split("[")[0].strip()} Evolution with Redshift')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

        # Colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('log(M*/M☉)')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Redshift evolution plot saved to {save_path}")

        return fig

    def create_summary_dashboard(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Create comprehensive summary dashboard

        Args:
            save_path: Path to save plot

        Returns:
            Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Plot 1: SEDs (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_seds_subplot(ax1)

        # Plot 2: SFR vs Mass (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_sfr_mass_subplot(ax2)

        # Plot 3: Fit quality (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_fit_quality_subplot(ax3)

        # Plot 4: Redshift evolution (middle left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_redshift_evolution_subplot(ax4)

        # Plot 5: Population source counts (middle middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_source_counts_subplot(ax5)

        # Plot 6: Luminosity distribution (middle right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_luminosity_distribution_subplot(ax6)

        # Results summary table (bottom)
        ax7 = fig.add_subplot(gs[2, :])
        self._create_results_table(ax7)

        # Main title
        fig.suptitle('Simstack4 Results Dashboard', fontsize=20, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Summary dashboard saved to {save_path}")

        return fig

    def _plot_seds_subplot(self, ax: plt.Axes) -> None:
        """Helper method for SEDs subplot"""
        # Plot a few representative SEDs
        pop_ids = list(self.results.sed_results.keys())[:5]  # Limit to 5
        colors = plt.cm.tab10(np.linspace(0, 1, len(pop_ids)))

        for i, pop_id in enumerate(pop_ids):
            sed = self.results.sed_results[pop_id]
            label = f"{pop_id[:15]}..." if len(pop_id) > 15 else pop_id
            ax.plot(sed.wavelengths, sed.flux_densities, 'o-',
                   color=colors[i], label=label, markersize=4)

        ax.set_xlabel('λ [μm]')
        ax.set_ylabel('S [Jy]')
        ax.set_title('Stacked SEDs')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    def _plot_sfr_mass_subplot(self, ax: plt.Axes) -> None:
        """Helper method for SFR vs mass subplot"""
        summary_df = self.results.get_population_summary()
        valid_mask = (summary_df['sfr_msun_yr'] > 0) & (summary_df['median_log_mass'] > 0)

        if np.sum(valid_mask) > 0:
            plot_data = summary_df[valid_mask]
            ax.scatter(plot_data['median_log_mass'], plot_data['sfr_msun_yr'],
                      alpha=0.7, s=30)
            ax.set_yscale('log')

        ax.set_xlabel('log(M*/M☉)')
        ax.set_ylabel('SFR [M☉/yr]')
        ax.set_title('Main Sequence')
        ax.grid(True, alpha=0.3)

    def _plot_fit_quality_subplot(self, ax: plt.Axes) -> None:
        """Helper method for fit quality subplot"""
        map_names = list(self.results.band_results.keys())
        chi2_values = [self.results.band_results[name]['reduced_chi_squared'] for name in map_names]
        wavelengths = [self.results.band_results[name]['wavelength_um'] for name in map_names]

        ax.scatter(wavelengths, chi2_values, s=60, alpha=0.7)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.set_xlabel('λ [μm]')
        ax.set_ylabel('χ²_red')
        ax.set_title('Fit Quality')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)

    def _plot_redshift_evolution_subplot(self, ax: plt.Axes) -> None:
        """Helper method for redshift evolution subplot"""
        summary_df = self.results.get_population_summary()
        valid_mask = (summary_df['sfr_msun_yr'] > 0) & (summary_df['median_redshift'] > 0)

        if np.sum(valid_mask) > 0:
            plot_data = summary_df[valid_mask]
            ax.scatter(plot_data['median_redshift'], plot_data['sfr_msun_yr'],
                      alpha=0.7, s=30)
            ax.set_yscale('log')

        ax.set_xlabel('Redshift')
        ax.set_ylabel('SFR [M☉/yr]')
        ax.set_title('SFR Evolution')
        ax.grid(True, alpha=0.3)

    def _plot_source_counts_subplot(self, ax: plt.Axes) -> None:
        """Helper method for source counts subplot"""
        summary_df = self.results.get_population_summary()

        # Show top 10 populations by source count
        top_pops = summary_df.nlargest(10, 'n_sources')
        pop_labels = [pid[:10] + "..." if len(pid) > 10 else pid for pid in top_pops['population_id']]

        bars = ax.bar(range(len(pop_labels)), top_pops['n_sources'], alpha=0.7)
        ax.set_xlabel('Population')
        ax.set_ylabel('N sources')
        ax.set_title('Source Counts')
        ax.set_xticks(range(len(pop_labels)))
        ax.set_xticklabels(pop_labels, rotation=45, ha='right', fontsize=8)

    def _plot_luminosity_distribution_subplot(self, ax: plt.Axes) -> None:
        """Helper method for luminosity distribution subplot"""
        summary_df = self.results.get_population_summary()
        valid_lums = summary_df[summary_df['total_ir_luminosity_lsun'] > 0]['total_ir_luminosity_lsun']

        if len(valid_lums) > 0:
            ax.hist(np.log10(valid_lums), bins=15, alpha=0.7, edgecolor='black')

        ax.set_xlabel('log(L_IR/L☉)')
        ax.set_ylabel('N populations')
        ax.set_title('L_IR Distribution')
        ax.grid(True, alpha=0.3)

    def _create_results_table(self, ax: plt.Axes) -> None:
        """Helper method to create results summary table"""
        ax.axis('off')

        # Get summary statistics
        summary_df = self.results.get_population_summary()

        if len(summary_df) == 0:
            ax.text(0.5, 0.5, 'No results to display', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return

        # Create summary statistics
        stats_data = [
            ['Total Populations', f"{len(summary_df)}"],
            ['Total Sources', f"{summary_df['n_sources'].sum():,}"],
            ['Redshift Range', f"{summary_df['median_redshift'].min():.2f} - {summary_df['median_redshift'].max():.2f}"],
            ['Mass Range', f"{summary_df['median_log_mass'].min():.1f} - {summary_df['median_log_mass'].max():.1f}"],
            ['Mean SFR', f"{summary_df['sfr_msun_yr'].mean():.2e} M☉/yr"],
            ['Mean L_IR', f"{summary_df['total_ir_luminosity_lsun'].mean():.2e} L☉"]
        ]

        # Create table
        table = ax.table(cellText=stats_data,
                        colLabels=['Metric', 'Value'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.3, 0.3])

        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)

        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#f1f1f2' if i % 2 == 0 else 'white')

        ax.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

    def save_all_plots(self, output_dir: Path, populations: Optional[List[str]] = None) -> None:
        """
        Generate and save all standard plots

        Args:
            output_dir: Directory to save plots
            populations: List of populations for comparison plots
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating all plots in {output_dir}")

        # Standard plots
        plots_to_generate = [
            ('seds', lambda: self.plot_stacked_seds()),
            ('sfr_vs_mass', lambda: self.plot_sfr_vs_mass()),
            ('fit_quality', lambda: self.plot_fit_quality()),
            ('redshift_evolution_sfr', lambda: self.plot_redshift_evolution('sfr')),
            ('luminosity_functions', lambda: self.plot_luminosity_functions()),
            ('dashboard', lambda: self.create_summary_dashboard())
        ]

        for plot_name, plot_func in plots_to_generate:
            try:
                fig = plot_func()
                save_path = output_dir / f"{plot_name}.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved {plot_name} to {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate {plot_name}: {e}")

        # Population comparison if populations specified
        if populations and len(populations) > 1:
            try:
                fig = self.plot_population_comparison(populations)
                save_path = output_dir / "population_comparison.png"
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                logger.info(f"Saved population comparison to {save_path}")
            except Exception as e:
                logger.error(f"Failed to generate population comparison: {e}")

        logger.info("All plots generated successfully")


def create_plots_from_results(results_path: Path, output_dir: Path,
                            style: str = 'publication') -> SimstackPlots:
    """
    Convenience function to create plots from saved results

    Args:
        results_path: Path to saved results file
        output_dir: Directory to save plots
        style: Plotting style

    Returns:
        SimstackPlots object
    """
    plotter = SimstackPlots(results_file=results_path, style=style)
    plotter.save_all_plots(output_dir)
    return plotter