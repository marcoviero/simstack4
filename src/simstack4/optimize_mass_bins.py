"""
Optimize stellar mass binning for PAH tomographic spectroscopy.

Uses the actual catalog source distribution to find mass bins
that give comparable S/N across all bins. The constraint is the
sparsest (z, M*) cell — if any z-bin within a mass bin has too
few sources, the greybody fit fails and the PAH measurement is lost.

Usage
-----
    from optimize_mass_bins import optimize_mass_bins, plot_optimization

    # From your catalog:
    mass = cat["mass_med"].values
    z = cat["zpdf_med"].values

    # Find optimal bins
    result = optimize_mass_bins(mass, z, n_mass_bins=3)
    plot_optimization(result)
"""

import numpy as np
import matplotlib.pyplot as plt


def _count_per_cell(mass, z, mass_edges, z_edges):
    """Count sources in each (z, M*) cell."""
    n_mass = len(mass_edges) - 1
    n_z = len(z_edges) - 1
    counts = np.zeros((n_mass, n_z), dtype=int)
    for i in range(n_mass):
        for j in range(n_z):
            counts[i, j] = np.sum(
                (mass >= mass_edges[i])
                & (mass < mass_edges[i + 1])
                & (z >= z_edges[j])
                & (z < z_edges[j + 1])
            )
    return counts


def _flux_weight_per_bin(mass_edges, ms_slope=0.7):
    """
    Relative stacked FIR flux per mass bin (main-sequence scaling).

    The stacked flux at the FIR peak scales as:
        fpeak ∝ L_IR ∝ SFR ∝ M*^ms_slope

    Normalized to logM*=10.5.

    Returns array of relative flux weights per mass bin.
    """
    n_mass = len(mass_edges) - 1
    weights = np.zeros(n_mass)
    for i in range(n_mass):
        mid = (mass_edges[i] + mass_edges[i + 1]) / 2
        weights[i] = 10 ** (ms_slope * (mid - 10.5))
    return weights


def _score_binning(mass, z, mass_edges, z_edges, min_sources=200, ms_slope=0.7):
    """
    Score a mass binning scheme.

    The PAH detection S/N scales as:
        S/N ∝ mean_fpeak(M*) × √N / σ_confusion
    where mean_fpeak ∝ M*^ms_slope (main sequence).

    Low-mass bins have fainter stacks and need MORE sources
    to achieve the same PAH S/N as high-mass bins.

    Returns
    -------
    score : float
        Higher is better. Negative if any cell is unusable.
    details : dict
    """
    counts = _count_per_cell(mass, z, mass_edges, z_edges)
    n_mass, n_z = counts.shape

    # Flux weight per mass bin
    flux_wt = _flux_weight_per_bin(mass_edges, ms_slope)

    # Per mass bin statistics
    median_per_mass = np.array([np.median(counts[i]) for i in range(n_mass)])
    min_per_mass = np.array([np.min(counts[i]) for i in range(n_mass)])

    # Flux-weighted S/N proxy: fpeak_rel × √N
    # This is what actually determines PAH detection quality
    snr_proxy = flux_wt * np.sqrt(median_per_mass)
    snr_min = flux_wt * np.sqrt(min_per_mass.astype(float).clip(1))

    # Effective min sources: how many sources in the low-mass bin
    # give the same S/N as min_sources in a logM*=10.5 bin?
    # N_effective = N × flux_wt²
    effective_min = min_per_mass * flux_wt**2

    # Usable z-bins: cells where effective S/N exceeds threshold
    # A cell is "usable" if N × flux_wt² >= min_sources
    usable_per_mass = np.array(
        [np.sum(counts[i] * flux_wt[i] ** 2 >= min_sources) for i in range(n_mass)]
    )

    # Uniformity: ratio of weakest to strongest S/N
    snr_ratio = snr_proxy.min() / snr_proxy.max() if snr_proxy.max() > 0 else 0

    # Penalty
    n_failed = np.sum(counts * flux_wt[:, None] ** 2 < min_sources)
    frac_usable = usable_per_mass.min() / n_z

    if effective_min.min() < min_sources * 0.3:
        score = -1000 + effective_min.min()
    else:
        score = (
            snr_ratio * 100  # uniformity (0-100)
            + frac_usable * 50  # z-coverage (0-50)
            - n_failed * 5  # penalty per failed cell
        )

    details = {
        "counts": counts,
        "median_per_mass": median_per_mass,
        "min_per_mass": min_per_mass,
        "flux_weight": flux_wt,
        "snr_proxy": snr_proxy,
        "snr_min": snr_min,
        "snr_ratio": snr_ratio,
        "effective_min": effective_min,
        "usable_per_mass": usable_per_mass,
        "n_failed": n_failed,
        "global_min": counts.min(),
        "frac_usable": frac_usable,
        "score": score,
    }
    return score, details


def optimize_mass_bins(
    mass,
    z,
    *,
    n_mass_bins=3,
    z_edges=None,
    mass_range=None,
    min_sources=200,
    n_trials=2000,
    method="equal_snr",
    verbose=True,
):
    """
    Find optimal stellar mass bin edges for PAH tomography.

    Parameters
    ----------
    mass : array
        log10(M*/Msun) values from catalog.
    z : array
        Redshifts from catalog.
    n_mass_bins : int
        Number of mass bins (2-5).
    z_edges : array, optional
        Redshift bin edges. Default: np.arange(0.5, 3.55, 0.15)
    mass_range : tuple, optional
        (min, max) mass range. Default: auto from data.
    min_sources : int
        Minimum sources per (z, M*) cell.
    n_trials : int
        Number of random trials for optimization.
    method : str
        'equal_snr': optimize for equal S/N across bins.
        'equal_count': simple equal-count quantile bins.
        'optimize': brute-force search.
    verbose : bool

    Returns
    -------
    result : dict
        'mass_edges': optimal bin edges
        'details': scoring details
        'all_trials': list of (edges, score) if method='optimize'
    """
    valid = np.isfinite(mass) & np.isfinite(z) & (z >= 0.5) & (z <= 3.5)
    mass = mass[valid]
    z = z[valid]

    if mass_range is None:
        mass_range = (np.percentile(mass, 1), np.percentile(mass, 99))

    if z_edges is None:
        z_edges = np.arange(0.5, 3.55, 0.15)
    else:
        z_edges = np.asarray(z_edges, dtype=float)

    n_z = len(z_edges) - 1
    m_lo, m_hi = mass_range

    if verbose:
        print(f"\nOptimizing {n_mass_bins} mass bins for PAH tomography:")
        print(f"  Sources: {len(mass):,} in z=0.5-3.5")
        print(f"  Mass range: {m_lo:.1f} - {m_hi:.1f}")
        print(f"  Z bins: {n_z} (dz={z_edges[1]-z_edges[0]:.2f})")
        print(f"  Min sources/cell: {min_sources}")

    # ── Method 1: Equal count (baseline) ─────────────────────────
    quantiles = np.linspace(0, 1, n_mass_bins + 1)
    mass_in_range = mass[(mass >= m_lo) & (mass <= m_hi)]
    eq_count_edges = np.percentile(mass_in_range, quantiles * 100)
    eq_count_edges[0] = m_lo
    eq_count_edges[-1] = m_hi + 0.01

    eq_score, eq_details = _score_binning(mass, z, eq_count_edges, z_edges, min_sources)

    if verbose:
        print(f"\n  Equal-count bins: {np.round(eq_count_edges, 2)}")
        print(f"    Score: {eq_score:.1f}")
        print(f"    Flux-weighted S/N ratio (min/max): {eq_details['snr_ratio']:.2f}")
        for i in range(n_mass_bins):
            fw = eq_details["flux_weight"][i]
            print(
                f"    M*={eq_count_edges[i]:.2f}-{eq_count_edges[i+1]:.2f}: "
                f"median N={eq_details['median_per_mass'][i]:.0f}, "
                f"min N={eq_details['min_per_mass'][i]}, "
                f"flux_wt={fw:.2f}, "
                f"S/N∝{eq_details['snr_proxy'][i]:.0f}, "
                f"usable={eq_details['usable_per_mass'][i]}/{n_z}"
            )

    if method == "equal_count":
        return {
            "mass_edges": eq_count_edges,
            "details": eq_details,
            "method": "equal_count",
            "z_edges": z_edges,
            "mass": mass,
            "z": z,
        }

    # ── Method 2: Equal S/N (sqrt(N) balanced) ──────────────────
    # Idea: wider bins for high mass (fewer sources) to equalize sqrt(N)
    # Find edges where sqrt(N_per_zbin) is balanced

    # For each trial mass edge set, compute the median sqrt(N) per mass bin
    best_score = eq_score
    best_edges = eq_count_edges.copy()
    all_trials = [(eq_count_edges.copy(), eq_score)]

    np.random.seed(42)
    for trial in range(n_trials):
        # Random perturbation of equal-count edges
        if trial < n_trials // 3:
            # Phase 1: explore around equal-count
            inner = eq_count_edges[1:-1].copy()
            inner += np.random.randn(len(inner)) * 0.3
        elif trial < 2 * n_trials // 3:
            # Phase 2: explore around best so far
            inner = best_edges[1:-1].copy()
            inner += np.random.randn(len(inner)) * 0.15
        else:
            # Phase 3: fine-tune
            inner = best_edges[1:-1].copy()
            inner += np.random.randn(len(inner)) * 0.05

        inner = np.sort(inner)
        # Enforce minimum bin width and range
        inner = np.clip(inner, m_lo + 0.3, m_hi - 0.3)
        for k in range(1, len(inner)):
            if inner[k] - inner[k - 1] < 0.2:
                inner[k] = inner[k - 1] + 0.2

        trial_edges = np.concatenate([[m_lo], inner, [m_hi + 0.01]])
        trial_score, trial_det = _score_binning(
            mass, z, trial_edges, z_edges, min_sources
        )
        all_trials.append((trial_edges.copy(), trial_score))

        if trial_score > best_score:
            best_score = trial_score
            best_edges = trial_edges.copy()

    _, best_details = _score_binning(mass, z, best_edges, z_edges, min_sources)

    if verbose:
        print(f"\n  Optimized bins: {np.round(best_edges, 2)}")
        print(f"    Score: {best_score:.1f} (was {eq_score:.1f})")
        print(f"    Flux-weighted S/N ratio (min/max): {best_details['snr_ratio']:.2f}")
        for i in range(n_mass_bins):
            fw = best_details["flux_weight"][i]
            print(
                f"    M*={best_edges[i]:.2f}-{best_edges[i+1]:.2f}: "
                f"median N={best_details['median_per_mass'][i]:.0f}, "
                f"min N={best_details['min_per_mass'][i]}, "
                f"flux_wt={fw:.2f}, "
                f"S/N∝{best_details['snr_proxy'][i]:.0f}, "
                f"usable={best_details['usable_per_mass'][i]}/{n_z}"
            )

        # TOML output
        edges_str = ", ".join(f"{e:.2f}" for e in best_edges)
        print(f"\n  For TOML:")
        print(f"    bins = [{edges_str}]")

    return {
        "mass_edges": best_edges,
        "details": best_details,
        "eq_count_edges": eq_count_edges,
        "eq_count_details": eq_details,
        "method": method,
        "z_edges": z_edges,
        "mass": mass,
        "z": z,
        "all_trials": all_trials,
    }


def plot_optimization(result, save_path=None):
    """
    Plot the optimization results.

    Four panels:
      1. Source count heatmap (z vs M*) with bin edges
      2. S/N proxy per mass bin across z
      3. Score landscape (if optimization was run)
      4. Comparison: equal-count vs optimized
    """
    from pathlib import Path

    mass = result["mass"]
    z = result["z"]
    z_edges = np.asarray(result["z_edges"])
    best_edges = np.asarray(result["mass_edges"])
    best_det = result["details"]
    n_mass = len(best_edges) - 1
    n_z = len(z_edges) - 1

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Panel 1: Source count heatmap ────────────────────────────
    ax = axes[0, 0]
    counts = best_det["counts"]
    im = ax.imshow(
        counts,
        aspect="auto",
        origin="lower",
        extent=[z_edges[0], z_edges[-1], 0, n_mass],
        cmap="YlOrRd",
        vmin=0,
    )
    plt.colorbar(im, ax=ax, label="N sources")

    # Label mass bins
    for i in range(n_mass):
        ax.text(
            -0.02,
            i + 0.5,
            f"M*={best_edges[i]:.1f}-{best_edges[i+1]:.1f}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8,
        )
        # Mark cells below threshold
        for j in range(n_z):
            if counts[i, j] < 200:
                ax.plot(
                    z_edges[j] + (z_edges[1] - z_edges[0]) / 2,
                    i + 0.5,
                    "x",
                    color="k",
                    ms=8,
                    mew=2,
                )

    ax.set_xlabel("Redshift")
    ax.set_ylabel("Mass bin")
    ax.set_yticks([])
    ax.set_title("Source counts per (z, M*) cell\n(× = below threshold)")

    # ── Panel 2: Flux-weighted S/N vs z per mass bin ───────────────
    ax = axes[0, 1]
    import matplotlib.cm as cm

    colors = cm.viridis(np.linspace(0.15, 0.85, n_mass))
    z_mid = 0.5 * (z_edges[:-1] + z_edges[1:])
    flux_wt = best_det["flux_weight"]

    for i in range(n_mass):
        # Raw √N
        snr_raw = np.sqrt(counts[i].astype(float))
        ax.plot(z_mid, snr_raw, ":", color=colors[i], ms=2, lw=1, alpha=0.4)
        # Flux-weighted S/N
        snr_weighted = flux_wt[i] * snr_raw
        ax.plot(
            z_mid,
            snr_weighted,
            "o-",
            color=colors[i],
            ms=4,
            lw=1.5,
            label=f"M*={best_edges[i]:.1f}-{best_edges[i+1]:.1f} "
            f"(fw={flux_wt[i]:.2f})",
        )

    ax.axhline(np.sqrt(200), color="r", ls="--", alpha=0.5, label=f"√{200} threshold")
    ax.set_xlabel("Redshift")
    ax.set_ylabel("Flux-weighted √N  (∝ PAH S/N)")
    ax.set_title(
        "PAH detection S/N vs redshift\n(dotted = raw √N, solid = flux-weighted)"
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.2)

    # ── Panel 3: Comparison of binning strategies ────────────────
    ax = axes[1, 0]

    if "eq_count_edges" in result:
        eq_edges = result["eq_count_edges"]
        eq_det = result["eq_count_details"]

        x = np.arange(n_mass)
        w = 0.35

        ax.bar(
            x - w / 2,
            eq_det["median_per_mass"],
            w,
            label="Equal-count",
            alpha=0.7,
            color="C0",
        )
        ax.bar(
            x + w / 2,
            best_det["median_per_mass"],
            w,
            label="Optimized",
            alpha=0.7,
            color="C1",
        )

        # Error bars showing min-max range
        for i in range(n_mass):
            ax.plot(
                [i - w / 2, i - w / 2],
                [eq_det["min_per_mass"][i], eq_det["median_per_mass"][i] * 1.5],
                "|-",
                color="C0",
                lw=1,
            )
            ax.plot(
                [i + w / 2, i + w / 2],
                [best_det["min_per_mass"][i], best_det["median_per_mass"][i] * 1.5],
                "|-",
                color="C1",
                lw=1,
            )

        eq_labels = [f"{eq_edges[i]:.1f}-{eq_edges[i+1]:.1f}" for i in range(n_mass)]
        opt_labels = [
            f"{best_edges[i]:.1f}-{best_edges[i+1]:.1f}" for i in range(n_mass)
        ]

        ax.set_xticks(x)
        ax.set_xticklabels([f"Bin {i+1}" for i in range(n_mass)])
        ax.axhline(200, color="r", ls="--", alpha=0.5, label="Min threshold")
        ax.set_ylabel("Median N per z-bin")
        ax.set_title("Equal-count vs optimized")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

        # Annotate with edges
        for i in range(n_mass):
            ax.text(
                i - w / 2,
                eq_det["median_per_mass"][i] + 50,
                eq_labels[i],
                fontsize=6,
                ha="center",
                color="C0",
                rotation=45,
            )
            ax.text(
                i + w / 2,
                best_det["median_per_mass"][i] + 50,
                opt_labels[i],
                fontsize=6,
                ha="center",
                color="C1",
                rotation=45,
            )

    # ── Panel 4: Mass distribution + bin edges ───────────────────
    ax = axes[1, 1]
    ax.hist(mass, bins=100, color="gray", alpha=0.5, density=True, label="All sources")

    for i in range(n_mass):
        ax.axvline(best_edges[i], color="C1", ls="-", lw=2, alpha=0.7)
    ax.axvline(
        best_edges[-1], color="C1", ls="-", lw=2, alpha=0.7, label="Optimized edges"
    )

    if "eq_count_edges" in result:
        for e in eq_edges[1:-1]:
            ax.axvline(e, color="C0", ls="--", lw=1.5, alpha=0.5)
        ax.axvline(
            eq_edges[1],
            color="C0",
            ls="--",
            lw=1.5,
            alpha=0.5,
            label="Equal-count edges",
        )

    ax.set_xlabel("log(M*/M☉)")
    ax.set_ylabel("Density")
    ax.set_title("Mass distribution + bin edges")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    fig.suptitle(
        f"Mass binning optimization  |  {len(mass):,} sources  |  "
        f"S/N ratio: {best_det['snr_ratio']:.2f}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


if __name__ == "__main__":
    print("Usage:")
    print("  from optimize_mass_bins import optimize_mass_bins, plot_optimization")
    print()
    print("  mass = cat['mass_med'].values")
    print("  z = cat['zpdf_med'].values")
    print()
    print("  result = optimize_mass_bins(mass, z, n_mass_bins=3)")
    print("  fig = plot_optimization(result)")
