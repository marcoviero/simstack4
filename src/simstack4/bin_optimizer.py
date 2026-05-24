"""
SNR-equalizing bin optimizer for simstack4.

Supports arbitrary binning dimensions (redshift, stellar_mass, beta_uv,
L_UV, etc.) and can optimize one or more dimensions simultaneously.

Physics:
    SNR ~ mean_flux_per_source × sqrt(N_sources) / noise_floor
    - Bright populations (high M*, low z) need fewer sources
    - Faint populations (low M*, high z) need more sources
    - Goal: bin edges where SNR is uniform across all bins

Usage:
    from bin_optimizer import BinOptimizer

    opt = BinOptimizer(wrapper)
    opt.diagnose()

    # Optimize mass edges only (z fixed)
    opt.optimize(dims=["stellar_mass"])

    # Optimize both z and mass
    opt.optimize(dims=["redshift", "stellar_mass"])

    # 3-way: z, L_UV, beta_UV with shared edges
    opt.optimize(dims=["redshift", "L_UV", "beta_uv"],
                 shared_edges=True)

    opt.plot_comparison()
    opt.print_config_snippet()
"""

import numpy as np
import pandas as pd
from itertools import product as iterproduct
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class BinOptimizer:
    """
    Optimize binning for equalized SNR across populations.

    Parameters
    ----------
    wrapper : SimstackWrapper
        Must have completed stacking + analysis.
    split_value : int
        Galaxy type to optimize for (0=star-forming, 1=quiescent).
    """

    def __init__(
        self,
        wrapper,
        split_value=0,
        photoz_cut=None,
        sigma_z_floor=None,
        photoz_cols=None,
    ):
        """
        Parameters
        ----------
        wrapper : SimstackWrapper
        split_value : int
            Galaxy type (0=star-forming, 1=quiescent).
        photoz_cut : float or None
            Drop sources with σ_z/(1+z) > photoz_cut before optimization.
            E.g. 0.05 removes sources whose 68% photo-z CI spans >5% of (1+z).
        sigma_z_floor : float or None
            Enforce minimum bin width = sigma_z_floor × local median σ_z.
            Applied after equal-power edge placement for the redshift dimension.
            E.g. 2.0 means no bin can be narrower than 2× the local photo-z scatter.
        photoz_cols : dict or None
            Column names for the photo-z PDF bounds. Defaults to COSMOSWeb columns:
            {"lo": "zpdf_l68", "hi": "zpdf_u68", "z": "zpdf_med"}.
        """
        self.wrapper = wrapper
        self.split_value = split_value
        self.photoz_cut = photoz_cut
        self.sigma_z_floor = sigma_z_floor
        self._photoz_cols = photoz_cols or {
            "lo": "zpdf_l68",
            "hi": "zpdf_u68",
            "z": "zpdf_med",
        }

        if wrapper.population_manager is None:
            raise RuntimeError("No population_manager — load catalog first")
        if wrapper.processed_results is None:
            raise RuntimeError("No processed_results — run analysis first")

        self.pm = wrapper.population_manager
        self.results = wrapper.processed_results

        # Discover binning dimensions from config
        self.dim_configs = {}  # dim_name -> {col, edges, label}
        for bin_name, bin_config in self.pm.bin_configs.items():
            self.dim_configs[bin_name] = {
                "col": bin_config.id,
                "edges": sorted(bin_config.bins),
                "label": getattr(bin_config, "label", bin_name),
            }

        self.dim_names = list(self.dim_configs.keys())
        self._catalog_df = getattr(wrapper.sky_catalogs, "catalog_df", None)

        # Build split mask (catalog indices belonging to this split_value)
        # photoz_cut is applied here if set.
        self._build_split_mask()

        # Precompute median σ_z(z) profile for floor enforcement.
        self._sigma_z_func = None
        if sigma_z_floor is not None:
            self._build_sigma_z_profile()

        # Extract current per-population data
        self._extract_current_data()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _build_split_mask(self):
        """Build boolean mask over catalog for the target split_value.

        If photoz_cut is set, additionally removes sources whose
        σ_z/(1+z) exceeds the threshold, where σ_z = (zpdf_u68 - zpdf_l68)/2.
        """
        if self._catalog_df is None:
            self._split_mask = None
            return
        n = len(self._catalog_df)
        idx_set = set()
        for pop in self.pm.populations.values():
            if pop.split_value == self.split_value:
                idx_set.update(pop.indices.tolist())
        self._split_mask = np.array([i in idx_set for i in range(n)])

        if self.photoz_cut is not None:
            cols = self._photoz_cols
            lo_col, hi_col, z_col = cols["lo"], cols["hi"], cols["z"]
            if (
                lo_col in self._catalog_df.columns
                and hi_col in self._catalog_df.columns
            ):
                lo = self._catalog_df[lo_col].values
                hi = self._catalog_df[hi_col].values
                z = (
                    self._catalog_df[z_col].values
                    if z_col in self._catalog_df.columns
                    else np.zeros(n, dtype=float)
                )
                sigma_z = (hi - lo) / 2.0
                dz_over_1pz = sigma_z / (1.0 + np.maximum(z, 0.0))
                good = np.isfinite(dz_over_1pz) & (dz_over_1pz <= self.photoz_cut)
                n_before = int(self._split_mask.sum())
                self._split_mask = self._split_mask & good
                n_after = int(self._split_mask.sum())
                print(
                    f"  Photo-z cut σ_z/(1+z) ≤ {self.photoz_cut}: "
                    f"{n_before - n_after:,} removed → {n_after:,} remain"
                )
            else:
                print(
                    f"  Photo-z cut requested but columns "
                    f"'{lo_col}'/'{hi_col}' not found — skipping"
                )

    def _build_sigma_z_profile(self):
        """Build median σ_z(z) profile from the (post-cut) catalog sample.

        Stores self._sigma_z_func: callable z → median σ_z at that redshift.
        Used by _enforce_sigma_z_floor to set minimum bin widths.
        """
        if self._catalog_df is None or self._split_mask is None:
            self._sigma_z_func = None
            return
        cols = self._photoz_cols
        lo_col, hi_col, z_col = cols["lo"], cols["hi"], cols["z"]
        if (
            lo_col not in self._catalog_df.columns
            or hi_col not in self._catalog_df.columns
        ):
            print(
                f"  σ_z floor: columns '{lo_col}'/'{hi_col}' not found — floor disabled"
            )
            self._sigma_z_func = None
            return

        df_cut = self._catalog_df[self._split_mask]
        lo = df_cut[lo_col].values
        hi = df_cut[hi_col].values
        z = df_cut[z_col].values if z_col in df_cut.columns else np.zeros(len(lo))
        sigma_z = (hi - lo) / 2.0

        valid = np.isfinite(sigma_z) & np.isfinite(z) & (sigma_z > 0)
        if valid.sum() < 10:
            self._sigma_z_func = None
            return

        z_v, sz_v = z[valid], sigma_z[valid]
        # Equal-count z bins so each has enough sources for a stable median
        pct = np.linspace(0, 100, 30)
        z_edges = np.unique(np.percentile(z_v, pct))
        z_centers, med_sz = [], []
        for i in range(len(z_edges) - 1):
            in_bin = (z_v >= z_edges[i]) & (z_v < z_edges[i + 1])
            if in_bin.sum() >= 5:
                z_centers.append((z_edges[i] + z_edges[i + 1]) / 2.0)
                med_sz.append(float(np.median(sz_v[in_bin])))

        if len(z_centers) < 2:
            self._sigma_z_func = None
            return

        self._sigma_z_func = interp1d(
            z_centers,
            med_sz,
            kind="linear",
            fill_value=(med_sz[0], med_sz[-1]),
            bounds_error=False,
        )

    def _enforce_sigma_z_floor(self, edges, target_dim):
        """Post-process edges to enforce min bin width = sigma_z_floor × local σ_z.

        Only applied to the redshift dimension. Uses a left-to-right sweep:
        when a bin is too narrow, its right edge is pushed outward, which
        propagates naturally to the next bin's check. The final edge is fixed.
        """
        if self._sigma_z_func is None or target_dim != "redshift":
            return edges
        edges = list(edges)
        for i in range(len(edges) - 1):
            z_center = (edges[i] + edges[i + 1]) / 2.0
            min_width = self.sigma_z_floor * float(self._sigma_z_func(z_center))
            if edges[i + 1] - edges[i] < min_width:
                new_right = edges[i] + min_width
                if i + 1 < len(edges) - 1:  # not the fixed final edge
                    edges[i + 1] = min(new_right, edges[-1] - 1e-6)
        return np.array(edges)

    @staticmethod
    def _parse_bin_ranges_from_id(id_label, dim_names):
        """
        Parse bin ranges from population id_label as fallback.

        Format: 'redshift_0.01_0.5__stellar_mass_8.5_9.5__split_0'
        Each segment is dim_lo_hi separated by '__'.
        """
        parsed = {}
        segments = id_label.split("__")
        for seg in segments:
            if seg.startswith("split_"):
                continue
            # Try to match each known dimension name
            for dim in dim_names:
                if seg.startswith(dim + "_"):
                    remainder = seg[len(dim) + 1 :]  # e.g. "0.01_0.5"
                    parts = remainder.split("_")
                    if len(parts) >= 2:
                        try:
                            lo = float(parts[0])
                            hi = float(parts[1])
                            parsed[dim] = (lo, hi)
                        except ValueError:
                            pass
                    break
        return parsed

    def _extract_current_data(self):
        """Pull per-population diagnostics from existing results."""
        rows = []

        for pop_id, sed in self.results.sed_results.items():
            pop = self.pm.populations.get(pop_id)
            if pop is None or pop.split_value != self.split_value:
                continue

            # Bin ranges for every dimension
            # Fallback: parse from id_label if bin_ranges is incomplete
            # (happens when populations are reconstructed from saved JSON)
            bin_ranges = dict(pop.bin_ranges)
            missing_dims = [d for d in self.dim_names if d not in bin_ranges]
            if missing_dims:
                parsed = self._parse_bin_ranges_from_id(pop_id, self.dim_names)
                for d in missing_dims:
                    if d in parsed:
                        bin_ranges[d] = parsed[d]

            row = {"pop_id": pop_id, "n_sources": sed.n_sources}

            for dim in self.dim_names:
                lo, hi = bin_ranges.get(dim, (np.nan, np.nan))
                row[f"{dim}_lo"] = lo
                row[f"{dim}_hi"] = hi
                # Try multiple sources for median value
                med = pop.medians.get(dim, np.nan)
                if np.isnan(med):
                    # Fallback to SED-level values
                    if dim == "redshift":
                        med = sed.median_redshift
                    elif dim == "stellar_mass":
                        med = sed.median_mass
                if np.isnan(med) and not (np.isnan(lo) or np.isnan(hi)):
                    med = (lo + hi) / 2
                row[f"{dim}_med"] = med

            # SNR metrics
            band_snrs = []
            if sed.flux_densities is not None and sed.flux_errors is not None:
                for f, e in zip(sed.flux_densities, sed.flux_errors):
                    if e > 0 and np.isfinite(f) and np.isfinite(e):
                        band_snrs.append(abs(f) / e)

            pos_snrs = (
                [s for s, f in zip(band_snrs, sed.flux_densities) if f > 0]
                if sed.flux_densities is not None
                else []
            )

            row["sed_snr"] = (
                sed.sed_snr
                if sed.sed_snr is not None
                else (np.median(pos_snrs) if pos_snrs else 0.0)
            )
            row["peak_snr"] = max(band_snrs) if band_snrs else 0.0
            row["fit_tier"] = sed.fit_quality_tier
            row["T_rest"] = sed.dust_temperature_rest_frame

            # Store median_redshift and median_mass from SED for display
            row["median_redshift"] = sed.median_redshift
            row["median_mass"] = sed.median_mass

            rows.append(row)

        self.df = pd.DataFrame(rows)
        if len(self.df) == 0:
            raise RuntimeError(
                f"No populations found with split_value={self.split_value}"
            )

        # Current edges per dimension
        self.current_edges = {
            dim: [float(x) for x in cfg["edges"]]
            for dim, cfg in self.dim_configs.items()
        }

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnose(self):
        """Print current binning diagnostics."""
        print("=" * 90)
        print("CURRENT BINNING DIAGNOSTICS")
        print("=" * 90)

        split_name = "Star-forming" if self.split_value == 0 else "Quiescent"
        print(f"Galaxy type: {split_name}")
        print(f"Populations: {len(self.df)}")
        print(f"Dimensions:  {self.dim_names}")
        for dim in self.dim_names:
            print(f"  {dim}: {self.current_edges[dim]}")

        # Table header
        dim_headers = "".join(f"{d:<13}" for d in self.dim_names)
        print(
            f"\n{dim_headers}{'N_src':>7} {'SNR':>7} {'Peak':>7} "
            f"{'Tier':>5} {'T_rest':>7}"
        )
        print("-" * (13 * len(self.dim_names) + 40))

        # Sort by first dim, then second, etc.
        sort_cols = [f"{d}_lo" for d in self.dim_names]
        df_sorted = self.df.sort_values(sort_cols)

        for _, row in df_sorted.iterrows():
            dim_strs = ""
            for d in self.dim_names:
                lo, hi = row[f"{d}_lo"], row[f"{d}_hi"]
                dim_strs += f"{lo:.1f}-{hi:.1f}    "
            t_str = f"{row['T_rest']:.0f}" if pd.notna(row.get("T_rest")) else "---"
            tier = row.get("fit_tier", "?") or "?"
            print(
                f"{dim_strs}{row['n_sources']:>7d} {row['sed_snr']:>7.1f} "
                f"{row['peak_snr']:>7.1f} {tier:>5} {t_str:>7}"
            )

        # Summary
        snrs = self.df["sed_snr"]
        print(f"\nSNR range: {snrs.min():.1f} – {snrs.max():.1f}")
        print(f"SNR mean:  {snrs.mean():.1f} ± {snrs.std():.1f}")
        ratio = snrs.max() / max(snrs.min(), 0.01)
        print(f"SNR ratio (max/min): {ratio:.1f}x")

        # Per-slice stats for first dimension
        primary = self.dim_names[0]
        print(f"\nPer {primary} slice:")
        print(
            f"  {'slice':<15} {'N_bins':>6} {'SNR_min':>8} {'SNR_max':>8} "
            f"{'ratio':>7} {'N_min':>7} {'N_max':>7}"
        )
        print("  " + "-" * 65)
        for lo in sorted(self.df[f"{primary}_lo"].unique()):
            sl = self.df[self.df[f"{primary}_lo"] == lo]
            hi = sl[f"{primary}_hi"].iloc[0]
            s, n = sl["sed_snr"], sl["n_sources"]
            r = s.max() / max(s.min(), 0.01)
            print(
                f"  {lo:.1f}-{hi:.1f}        {len(sl):>6} {s.min():>8.1f} "
                f"{s.max():>8.1f} {r:>6.1f}x {n.min():>7d} {n.max():>7d}"
            )

        return self.df

    # ------------------------------------------------------------------
    # SNR model
    # ------------------------------------------------------------------

    def _get_slices(self, target_dim, fixed_edges):
        """
        Generate all slices of fixed dimensions.

        Parameters
        ----------
        target_dim : str
            Dimension being optimized.
        fixed_edges : dict
            {dim_name: [edges]} for all fixed dimensions.

        Yields
        ------
        dict : {dim_name: (lo, hi)} for one combination of fixed dim bins.
        """
        fixed_dims = [d for d in self.dim_names if d != target_dim]
        if not fixed_dims:
            yield {}
            return

        ranges_per_dim = []
        for d in fixed_dims:
            edges = fixed_edges[d]
            ranges_per_dim.append(
                [(d, edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
            )

        for combo in iterproduct(*ranges_per_dim):
            yield {d: (lo, hi) for d, lo, hi in combo}

    def _get_populations_in_slice(self, target_dim, fixed_slice):
        """
        Get populations that fall within the given fixed-dimension slice.

        Returns dataframe rows sorted by target_dim median.
        """
        mask = pd.Series(True, index=self.df.index)
        for d, (lo, hi) in fixed_slice.items():
            mask &= self.df[f"{d}_lo"] >= lo - 1e-6
            mask &= self.df[f"{d}_hi"] <= hi + 1e-6

        subset = self.df[mask].sort_values(f"{target_dim}_med")
        return subset

    def _get_catalog_values_in_slice(self, target_dim, fixed_slice):
        """
        Get catalog values of target_dim for sources in the fixed-dim slice.
        Returns None if catalog is not available.
        """
        if self._catalog_df is None or self._split_mask is None:
            return None

        mask = self._split_mask.copy()
        for d, (lo, hi) in fixed_slice.items():
            col = self.dim_configs[d]["col"]
            vals = self._catalog_df[col].values
            mask &= (vals >= lo) & (vals < hi)

        target_col = self.dim_configs[target_dim]["col"]
        return self._catalog_df[target_col].values[mask]

    def _build_source_cdf(self, target_dim, fixed_slice):
        """
        Build cumulative source count function from existing bin data.

        Works without raw catalog access by interpolating the cumulative
        N_sources at each bin edge.

        Returns
        -------
        cdf_func : callable
            cdf_func(x) -> cumulative source count up to x
        total_sources : int
            Total sources in this slice
        """
        subset = self._get_populations_in_slice(target_dim, fixed_slice)
        if len(subset) == 0:
            return None, 0

        subset = subset.sort_values(f"{target_dim}_lo")

        # Build cumulative at bin edges
        edges = []
        cum = []
        running = 0.0
        for _, row in subset.iterrows():
            lo = row[f"{target_dim}_lo"]
            hi = row[f"{target_dim}_hi"]
            n = row["n_sources"]
            if not edges or lo != edges[-1]:
                edges.append(lo)
                cum.append(running)
            running += n
            edges.append(hi)
            cum.append(running)

        # Deduplicate (same edge from adjacent bins)
        edge_arr = np.array(edges)
        cum_arr = np.array(cum)
        # Keep last value at each unique edge
        unique_edges, unique_idx = np.unique(edge_arr, return_index=True)
        # For cumulative, we want the MAX cum value at each edge
        unique_cum = np.array([max(cum_arr[edge_arr == e]) for e in unique_edges])
        # Ensure monotonic
        for i in range(1, len(unique_cum)):
            unique_cum[i] = max(unique_cum[i], unique_cum[i - 1])

        if len(unique_edges) < 2:
            return None, 0

        cdf_func = interp1d(
            unique_edges,
            unique_cum,
            kind="linear",
            fill_value=(0, unique_cum[-1]),
            bounds_error=False,
        )
        return cdf_func, int(unique_cum[-1])

    def _predict_count_in_range(self, cdf_func, lo, hi):
        """Predict source count in [lo, hi) using CDF interpolation."""
        if cdf_func is None:
            return 0
        return max(0, int(round(cdf_func(hi) - cdf_func(lo))))

    def _predict_snr_for_edges(self, target_dim, edges, fixed_slice):
        """
        Predict SNR for candidate bin edges along target_dim.

        Model: SNR(bin) = snr_per_source(center) * sqrt(N_in_bin)

        Uses only reliably detected bins (SNR > 1).  Endpoint outliers
        (e.g., z=8-10 with T>100K) are pruned if their snr_per_source
        exceeds 5x the median of interior points.  Predictions outside
        the reliable data range are clamped to boundary values.
        """
        subset = self._get_populations_in_slice(target_dim, fixed_slice)
        if len(subset) < 2:
            return None

        m_med = subset[f"{target_dim}_med"].values
        snr_obs = subset["sed_snr"].values
        n_obs = subset["n_sources"].values

        # Only use reliably detected bins
        reliable = snr_obs > 1.0
        if np.sum(reliable) < 2:
            return None

        m_r = m_med[reliable]
        snr_r = snr_obs[reliable]
        n_r = n_obs[reliable]
        snr_per_src = snr_r / np.sqrt(np.maximum(n_r, 1))

        # Prune endpoint outliers: if the highest or lowest point has
        # snr_per_source > 5x the median of the rest, drop it.
        # This catches the z=8-10 anomaly (T>100K, different SED regime).
        if len(snr_per_src) >= 4:
            order = np.argsort(m_r)
            sps_sorted = snr_per_src[order]
            # Check high endpoint
            interior_median = np.median(sps_sorted[:-1])
            if interior_median > 0 and sps_sorted[-1] / interior_median > 5:
                keep = np.ones(len(m_r), dtype=bool)
                keep[order[-1]] = False
                m_r = m_r[keep]
                snr_per_src = snr_per_src[keep]
            # Check low endpoint
            if len(snr_per_src) >= 4:
                order = np.argsort(m_r)
                sps_sorted = snr_per_src[order]
                interior_median = np.median(sps_sorted[1:])
                if interior_median > 0 and sps_sorted[0] / interior_median > 5:
                    keep = np.ones(len(m_r), dtype=bool)
                    keep[order[0]] = False
                    m_r = m_r[keep]
                    snr_per_src = snr_per_src[keep]

        if len(m_r) < 2:
            return None

        interp_func = interp1d(
            m_r,
            np.log10(np.maximum(snr_per_src, 1e-10)),
            kind="linear",
            fill_value=(
                np.log10(max(snr_per_src[np.argmin(m_r)], 1e-10)),
                np.log10(max(snr_per_src[np.argmax(m_r)], 1e-10)),
            ),
            bounds_error=False,
        )

        # Source counts: catalog or CDF
        cat_vals = self._get_catalog_values_in_slice(target_dim, fixed_slice)
        use_catalog = cat_vals is not None and len(cat_vals) > 0

        if not use_catalog:
            cdf_func, _ = self._build_source_cdf(target_dim, fixed_slice)
            if cdf_func is None:
                return None

        predicted = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            center = (lo + hi) / 2.0

            if use_catalog:
                n_in_bin = int(np.sum((cat_vals >= lo) & (cat_vals < hi)))
            else:
                n_in_bin = self._predict_count_in_range(cdf_func, lo, hi)

            if n_in_bin < 1:
                predicted.append(0.01)
                continue
            log_snr_ps = interp_func(center)
            predicted.append(max(10**log_snr_ps * np.sqrt(n_in_bin), 0.01))

        return np.array(predicted)

    # ------------------------------------------------------------------
    # Single-dimension optimizer
    # ------------------------------------------------------------------

    def _optimize_dim(
        self,
        target_dim,
        n_bins=None,
        min_sources=30,
        dim_range=None,
        fixed_edges=None,
        shared_edges=True,
    ):
        """
        Optimize edges for one dimension.

        Parameters
        ----------
        target_dim : str
            Dimension to optimize.
        n_bins : int or None
            Number of bins.  None = keep current count.
        min_sources : int
            Minimum sources per bin.
        dim_range : tuple or None
            (min, max).  None = use current.
        fixed_edges : dict or None
            Edges for other dimensions.  None = use current.
        shared_edges : bool
            If True, find single set of edges across all fixed-dim slices.
            If False, per-slice edges (stored but can't be used in config).

        Returns
        -------
        dict : Per-slice results with edges and predictions.
        """
        current = self.current_edges[target_dim]
        if n_bins is None:
            n_bins = len(current) - 1
        if dim_range is None:
            dim_range = (current[0], current[-1])
        if fixed_edges is None:
            fixed_edges = {
                d: self.current_edges[d] for d in self.dim_names if d != target_dim
            }

        d_min, d_max = dim_range

        per_slice_results = {}

        for fixed_slice in self._get_slices(target_dim, fixed_edges):
            slice_key = tuple(sorted(fixed_slice.items()))

            # Get source count from populations
            slice_pops_check = self._get_populations_in_slice(target_dim, fixed_slice)
            n_total = int(slice_pops_check["n_sources"].sum())

            if n_total < n_bins * min_sources:
                per_slice_results[slice_key] = {
                    "edges": current[:],
                    "status": "insufficient_sources",
                    "n_sources": n_total,
                }
                continue

            # Signal quality for weighting
            slice_pops = self._get_populations_in_slice(target_dim, fixed_slice)
            slice_mean_snr = float(slice_pops["sed_snr"].mean())

            # Skip slices where fewer than half the bins are detected.
            # Undetected slices produce unreliable sps models that
            # corrupt the unified edge average.
            n_detected = int(np.sum(slice_pops["sed_snr"] > 1.0))
            n_bins_in_slice = len(slice_pops)
            if n_bins_in_slice > 0 and n_detected / n_bins_in_slice < 0.5:
                per_slice_results[slice_key] = {
                    "edges": current[:],
                    "status": "low_detection_rate",
                    "n_sources": n_total,
                    "n_detected": n_detected,
                    "n_bins_in_slice": n_bins_in_slice,
                }
                continue

            best_edges = self._compute_equal_power_edges(
                target_dim,
                fixed_slice,
                n_bins,
                d_min,
                d_max,
            )
            if best_edges is None:
                per_slice_results[slice_key] = {
                    "edges": current[:],
                    "status": "power_computation_failed",
                    "n_sources": n_total,
                }
                continue

            # Predict SNR with new edges
            pred_snr = self._predict_snr_for_edges(target_dim, best_edges, fixed_slice)

            # Count sources per bin via CDF
            cdf_func, _ = self._build_source_cdf(target_dim, fixed_slice)
            if cdf_func is not None:
                counts = [
                    self._predict_count_in_range(
                        cdf_func, best_edges[i], best_edges[i + 1]
                    )
                    for i in range(len(best_edges) - 1)
                ]
            else:
                counts = [0] * (len(best_edges) - 1)

            per_slice_results[slice_key] = {
                "edges": best_edges,
                "pred_snr": pred_snr,
                "counts": counts,
                "status": "optimized",
                "n_sources": n_total,
                "fixed_slice": fixed_slice,
                "mean_snr": slice_mean_snr,
            }

        # Build unified edges if shared
        if shared_edges:
            unified = self._average_edges(per_slice_results, n_bins, d_min, d_max)
        else:
            unified = None

        return {
            "per_slice": per_slice_results,
            "unified": unified,
            "n_bins": n_bins,
            "dim_range": dim_range,
        }

    def _compute_equal_power_edges(self, target_dim, fixed_slice, n_bins, d_min, d_max):
        """
        Place edges so each bin captures equal signal power (SNR**2).

        From SED fits:
          sps(z) = peak_snr / sqrt(N)   [per-source signal strength]
          power_density(z) = sps**2 * dn/dz

        Cumulative power P(z) = integral of power_density.
        Edges placed at P = k/n_bins * P_total.

        Deterministic: no optimization, no local minima.
        """
        subset = self._get_populations_in_slice(target_dim, fixed_slice)
        if len(subset) < 2:
            return None

        subset = subset.sort_values(f"{target_dim}_lo")

        z_lo = subset[f"{target_dim}_lo"].values
        z_hi = subset[f"{target_dim}_hi"].values
        peak = subset["peak_snr"].values
        n_src = subset["n_sources"].values

        # Per-source signal from SED fits (peak_snr always > 0)
        sps = peak / np.sqrt(np.maximum(n_src, 1))

        # Source density (per unit of dimension)
        dz = np.maximum(z_hi - z_lo, 0.01)
        n_density = n_src / dz

        # Power density: where detectable signal concentrates
        # SNR**2 = sps**2 * N, so d(SNR**2)/dz = sps**2 * n_density
        power_density = sps**2 * n_density

        # Cumulative power at existing bin edges
        bin_edges = np.concatenate([z_lo[:1], z_hi])
        cum_power = np.zeros(len(bin_edges))
        for i in range(len(subset)):
            cum_power[i + 1] = cum_power[i] + power_density[i] * dz[i]

        total_power = cum_power[-1]
        if total_power <= 0:
            return None

        # Interpolate cumulative power
        power_cdf = interp1d(
            bin_edges,
            cum_power,
            kind="linear",
            fill_value=(0, total_power),
            bounds_error=False,
        )

        # Invert on fine grid: find z where P(z) = k/n_bins * total
        z_fine = np.linspace(d_min, d_max, 10000)
        p_fine = power_cdf(z_fine)

        edges = [d_min]
        for k in range(1, n_bins):
            target_p = k * total_power / n_bins
            idx = int(np.searchsorted(p_fine, target_p))
            idx = min(idx, len(z_fine) - 1)
            edges.append(float(z_fine[idx]))
        edges.append(d_max)
        edges = np.array(edges)

        if self.sigma_z_floor is not None:
            edges = self._enforce_sigma_z_floor(edges, target_dim)

        return edges

    def _average_edges(self, per_slice_results, n_bins, d_min, d_max):
        """SNR-weighted average of per-slice optimal edges.

        Slices with higher mean SNR contribute more, so well-detected
        slices drive the result over noisy slices.
        """
        all_internal = []
        weights = []
        for info in per_slice_results.values():
            if info["status"] != "optimized":
                continue
            edges = info["edges"]
            internal = edges[1:-1]
            if len(internal) == n_bins - 1:
                all_internal.append(internal)
                w = info.get("mean_snr", 1.0)
                weights.append(max(w, 0.01))

        if not all_internal:
            return None

        all_internal = np.array(all_internal)
        weights = np.array(weights)
        weights /= weights.sum()

        avg = np.average(all_internal, axis=0, weights=weights)
        unified = np.round(np.concatenate([[d_min], avg, [d_max]]), 2)
        return unified

    # ------------------------------------------------------------------
    # Multi-dimension optimizer (coordinate descent)
    # ------------------------------------------------------------------

    def optimize(
        self,
        dims=None,
        n_bins=None,
        min_sources=30,
        dim_ranges=None,
        shared_edges=True,
        max_iterations=5,
        tol=0.01,
        fixed_edges=None,
    ):
        """
        Optimize bin edges for one or more dimensions.

        Parameters
        ----------
        dims : list[str] or None
            Dimensions to optimize. None = all dimensions.
        n_bins : dict[str, int] or int or None
            Number of bins per dimension.  Int applies to all.
            Dict example: {"redshift": 7, "stellar_mass": 5}
            None = keep current bin counts.
        min_sources : int
            Minimum sources per bin (across all dimensions).
        dim_ranges : dict[str, tuple] or None
            {dim: (min, max)} overrides.
        shared_edges : bool
            If True, each dimension gets one set of edges used everywhere.
        max_iterations : int
            Coordinate descent iterations (for multi-dim).
            Capped at 1 when no catalog is loaded (populations can't
            be re-sliced with shifted edges).
        tol : float
            Convergence tolerance on edge movement (relative).
        fixed_edges : dict[str, list] or None
            Fix specific dimensions at given edges (must align with
            original bin boundaries when no catalog is loaded).
            Example: {"redshift": [0.01, 0.5, 1.5, 3.0, 6.0, 10.0]}
            Dimensions in fixed_edges are excluded from optimization.

        Returns
        -------
        dict : {dim_name: optimized_edges} for each optimized dimension.
        """
        if dims is None:
            dims = self.dim_names[:]
        if isinstance(dims, str):
            dims = [dims]

        # Apply fixed_edges: override current and exclude from optimization
        if fixed_edges:
            for d, edges in fixed_edges.items():
                if d not in self.dim_configs:
                    avail = list(self.dim_configs.keys())
                    raise ValueError(f"Unknown dimension '{d}'. Available: {avail}")
                self.current_edges[d] = [float(x) for x in edges]
                if d in dims:
                    dims.remove(d)
                    print(f"  {d}: fixed at {edges}")

        if not dims:
            raise ValueError(
                "No dimensions left to optimize after " "applying fixed_edges"
            )

        # Validate
        for d in dims:
            if d not in self.dim_configs:
                avail = list(self.dim_configs.keys())
                raise ValueError(f"Unknown dimension '{d}'. Available: {avail}")

        # Parse n_bins
        if n_bins is None:
            n_bins_dict = {d: len(self.current_edges[d]) - 1 for d in dims}
        elif isinstance(n_bins, int):
            n_bins_dict = {d: n_bins for d in dims}
        else:
            n_bins_dict = {d: n_bins[d] for d in dims if d in n_bins}
            for d in dims:
                if d not in n_bins_dict:
                    n_bins_dict[d] = len(self.current_edges[d]) - 1

        # Parse dim_ranges
        if dim_ranges is None:
            dim_ranges = {}

        # Cap iterations: without a catalog, populations can't be re-sliced
        # with shifted edges, so coordinate descent iteration 2+ finds
        # empty slices and produces garbage.
        has_catalog = self._catalog_df is not None
        if not has_catalog and max_iterations > 1:
            max_iterations = 1

        # Working edges: start from current
        working_edges = {
            d: np.array(self.current_edges[d], dtype=float) for d in self.dim_names
        }

        self._optimization_log = []

        split_name = "SF" if self.split_value == 0 else "QT"
        print(f"\n{'=' * 70}")
        print(f"OPTIMIZING: {dims} | {split_name} | shared_edges={shared_edges}")
        print(f"{'=' * 70}")

        if len(dims) == 1:
            # Single dimension — one pass is enough
            target = dims[0]
            fixed = {d: list(working_edges[d]) for d in self.dim_names if d != target}

            result = self._optimize_dim(
                target,
                n_bins=n_bins_dict[target],
                min_sources=min_sources,
                dim_range=dim_ranges.get(target),
                fixed_edges=fixed,
                shared_edges=shared_edges,
            )

            self._report_dim_result(target, result)

            if result["unified"] is not None:
                working_edges[target] = result["unified"]

            self.optimized_results = {target: result}
            self.optimized_edges = {
                d: [round(float(x), 4) for x in working_edges[d]]
                for d in self.dim_names
            }
            return self.optimized_edges

        # Multi-dimension: coordinate descent
        for iteration in range(max_iterations):
            edges_before = {d: working_edges[d].copy() for d in dims}

            print(f"\n--- Coordinate descent iteration {iteration + 1} ---")

            for target in dims:
                fixed = {
                    d: list(working_edges[d]) for d in self.dim_names if d != target
                }

                result = self._optimize_dim(
                    target,
                    n_bins=n_bins_dict[target],
                    min_sources=min_sources,
                    dim_range=dim_ranges.get(target),
                    fixed_edges=fixed,
                    shared_edges=shared_edges,
                )

                self._report_dim_result(target, result, verbose=(iteration == 0))

                if result["unified"] is not None:
                    working_edges[target] = result["unified"]

            # Check convergence
            max_shift = 0.0
            for d in dims:
                old, new = edges_before[d], working_edges[d]
                if len(old) == len(new):
                    span = old[-1] - old[0]
                    if span > 0:
                        shift = np.max(np.abs(new - old)) / span
                        max_shift = max(max_shift, shift)

            self._optimization_log.append(
                {
                    "iteration": iteration + 1,
                    "max_shift": max_shift,
                    "edges": {d: working_edges[d].copy() for d in dims},
                }
            )

            print(f"  Max relative edge shift: {max_shift:.4f}")
            if max_shift < tol:
                print(f"  Converged at iteration {iteration + 1}")
                break

        # Store results
        self.optimized_results = {}
        for target in dims:
            fixed = {d: list(working_edges[d]) for d in self.dim_names if d != target}
            self.optimized_results[target] = self._optimize_dim(
                target,
                n_bins=n_bins_dict[target],
                min_sources=min_sources,
                dim_range=dim_ranges.get(target),
                fixed_edges=fixed,
                shared_edges=shared_edges,
            )

        self.optimized_edges = {
            d: [round(float(x), 4) for x in working_edges[d]] for d in self.dim_names
        }

        # Final report
        print(f"\n{'=' * 70}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'=' * 70}")
        for d in self.dim_names:
            tag = " (optimized)" if d in dims else " (fixed)"
            print(f"  {d}{tag}:")
            print(f"    current:   {self.current_edges[d]}")
            print(f"    optimized: {self.optimized_edges[d]}")

        # Predict final SNR distribution
        self._report_predicted_snr()

        return self.optimized_edges

    def _report_dim_result(self, dim, result, verbose=True):
        """Print optimization result for one dimension."""
        n_opt = sum(
            1 for v in result["per_slice"].values() if v["status"] == "optimized"
        )
        n_skip_src = sum(
            1
            for v in result["per_slice"].values()
            if v["status"] == "insufficient_sources"
        )
        n_skip_det = sum(
            1
            for v in result["per_slice"].values()
            if v["status"] == "low_detection_rate"
        )

        parts = [f"{n_opt} optimized"]
        if n_skip_src:
            parts.append(f"{n_skip_src} too few sources")
        if n_skip_det:
            parts.append(f"{n_skip_det} low detection rate")
        print(f"\n  {dim}: {', '.join(parts)}")

        if result["unified"] is not None:
            print(
                f"    unified edges: {[round(float(x), 2) for x in result['unified']]}"
            )
            print(f"    (was:          {self.current_edges[dim]})")

        if verbose:
            for slice_key, info in result["per_slice"].items():
                if info["status"] != "optimized":
                    continue
                slice_str = ", ".join(
                    f"{d}={lo:.1f}-{hi:.1f}"
                    for d, (lo, hi) in info.get("fixed_slice", {}).items()
                )
                if slice_str:
                    slice_str = f" [{slice_str}]"

                pred = info.get("pred_snr")
                counts = info.get("counts")
                if pred is not None and len(pred) > 0:
                    ratio = max(pred) / max(min(pred), 0.01)
                    print(f"    {slice_str}")
                    print(f"      N/bin:     {counts}")
                    print(f"      pred SNR:  {[f'{s:.1f}' for s in pred]}")
                    print(f"      SNR ratio: {ratio:.1f}x")

    def _report_predicted_snr(self):
        """Report predicted SNR distribution with optimized edges."""
        if not hasattr(self, "optimized_results"):
            return

        # Use first optimized dim's per-slice predictions as representative
        all_pred_snrs = []
        for dim, result in self.optimized_results.items():
            for info in result["per_slice"].values():
                pred = info.get("pred_snr")
                if pred is not None:
                    all_pred_snrs.extend(pred.tolist())

        if all_pred_snrs:
            arr = np.array([s for s in all_pred_snrs if s > 0])
            if len(arr) > 0:
                ratio = arr.max() / max(arr.min(), 0.01)
                print(f"\n  Predicted SNR distribution:")
                print(f"    range: {arr.min():.1f} – {arr.max():.1f}")
                print(
                    f"    ratio: {ratio:.1f}x  (was: "
                    f"{self.df['sed_snr'].max() / max(self.df['sed_snr'].min(), 0.01):.1f}x)"
                )

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_comparison(self, figsize=None):
        """
        Plot current vs optimized binning.

        Generates one row of panels per optimized dimension, showing
        SNR vs that dimension for each slice of the others.
        """
        if not hasattr(self, "optimized_results"):
            print("Run optimize() first")
            return None

        opt_dims = list(self.optimized_results.keys())
        n_opt = len(opt_dims)

        if figsize is None:
            figsize = (14, 5 * n_opt)

        fig, axes = plt.subplots(n_opt, 1, figsize=figsize, squeeze=False)

        for row, target_dim in enumerate(opt_dims):
            ax = axes[row, 0]
            result = self.optimized_results[target_dim]

            # Plot current data points per slice
            fixed_dims = [d for d in self.dim_names if d != target_dim]
            colors = plt.cm.tab10(np.linspace(0, 1, 10))
            color_idx = 0

            for slice_key, info in result["per_slice"].items():
                fixed_slice = info.get("fixed_slice", {})
                slice_label = (
                    ", ".join(
                        f"{d}={lo:.1f}-{hi:.1f}" for d, (lo, hi) in fixed_slice.items()
                    )
                    or "all"
                )

                # Current populations in this slice
                subset = self._get_populations_in_slice(target_dim, fixed_slice)
                if len(subset) == 0:
                    continue

                c = colors[color_idx % len(colors)]
                color_idx += 1

                # Current data
                ax.scatter(
                    subset[f"{target_dim}_med"],
                    subset["sed_snr"],
                    s=np.sqrt(subset["n_sources"]) * 3,
                    color=c,
                    edgecolors="black",
                    linewidth=0.5,
                    alpha=0.7,
                    zorder=5,
                    label=f"Current [{slice_label}]",
                )

                # Optimized prediction
                pred = info.get("pred_snr")
                if pred is not None and info["status"] == "optimized":
                    opt_edges = info["edges"]
                    centers = [
                        (opt_edges[j] + opt_edges[j + 1]) / 2
                        for j in range(len(opt_edges) - 1)
                    ]
                    ax.scatter(
                        centers,
                        pred,
                        s=80,
                        marker="D",
                        color=c,
                        edgecolors="black",
                        linewidth=0.5,
                        alpha=0.8,
                        zorder=6,
                    )
                    # Connect with dashed line
                    ax.plot(centers, pred, "--", color=c, alpha=0.5, lw=1)

            # Edge lines
            for e in self.current_edges[target_dim][1:-1]:
                ax.axvline(e, color="steelblue", alpha=0.3, ls="--", lw=0.8)
            if result["unified"] is not None:
                for e in result["unified"][1:-1]:
                    ax.axvline(e, color="orangered", alpha=0.4, ls=":", lw=1.2)

            label = self.dim_configs[target_dim].get("label", target_dim)
            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("SNR", fontsize=11)
            ax.set_title(
                f"Optimizing {target_dim}  "
                f"(blue dashed=current edges, red dotted=optimized)",
                fontsize=11,
            )
            ax.legend(fontsize=7, loc="best", ncol=2)
            ax.grid(alpha=0.2)

        plt.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Config output
    # ------------------------------------------------------------------

    def print_config_snippet(self):
        """Print YAML config snippet with optimized edges."""
        if not hasattr(self, "optimized_edges"):
            print("Run optimize() first")
            return

        print("\n# ── Optimized bin edges for config.yaml ──")
        print("# Paste into catalog.classification.binning.<dim>.bins\n")

        for dim in self.dim_names:
            edges = self.optimized_edges[dim]
            current = self.current_edges[dim]
            changed = len(edges) != len(current) or not np.allclose(
                edges, current, atol=0.01
            )
            tag = "  # CHANGED" if changed else "  # unchanged"
            print(f"# {dim}:")
            print(f"bins: {[round(e, 2) for e in edges]}{tag}")

            if changed and hasattr(self, "optimized_results"):
                result = self.optimized_results.get(dim)
                if result:
                    for slice_key, info in result["per_slice"].items():
                        if info["status"] != "optimized":
                            continue
                        fs = info.get("fixed_slice", {})
                        tag = ", ".join(
                            f"{d}={lo:.1f}-{hi:.1f}" for d, (lo, hi) in fs.items()
                        )
                        if tag:
                            tag = f"  [{tag}]"
                        print(
                            f"#   per-slice{tag}: "
                            f"{[round(float(e), 2) for e in info['edges']]}"
                        )
            print()

    def to_dict(self):
        """Export optimization results as a dictionary."""
        result = {
            "dimensions": self.dim_names,
            "current_edges": {d: list(e) for d, e in self.current_edges.items()},
            "diagnostics": self.df.to_dict(orient="records"),
        }
        if hasattr(self, "optimized_edges"):
            result["optimized_edges"] = self.optimized_edges
        if hasattr(self, "_optimization_log"):
            result["convergence"] = self._optimization_log
        return result


# ------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------


def diagnose_binning(wrapper, split_value=0):
    """One-liner: print current binning diagnostics."""
    opt = BinOptimizer(wrapper, split_value=split_value)
    return opt.diagnose()


def optimize_binning(
    wrapper,
    dims=None,
    split_value=0,
    photoz_cut=None,
    sigma_z_floor=None,
    photoz_cols=None,
    n_bins=None,
    min_sources=30,
    shared_edges=True,
    plot=True,
    fixed_edges=None,
):
    """
    One-liner: optimize and optionally plot.

    Parameters
    ----------
    wrapper : SimstackWrapper
    dims : list[str] or None
        Dimensions to optimize (None = all).
    split_value : int
        0=star-forming, 1=quiescent.
    n_bins : dict or int or None
        Bins per dimension.
    min_sources : int
        Minimum sources per bin.
    shared_edges : bool
        Same edges across all slices of other dimensions.
    plot : bool
        Show comparison plot.
    """
    opt = BinOptimizer(
        wrapper,
        split_value=split_value,
        photoz_cut=photoz_cut,
        sigma_z_floor=sigma_z_floor,
        photoz_cols=photoz_cols,
    )
    opt.diagnose()
    opt.optimize(
        dims=dims,
        n_bins=n_bins,
        min_sources=min_sources,
        shared_edges=shared_edges,
        fixed_edges=fixed_edges,
    )
    if plot:
        opt.plot_comparison()
    opt.print_config_snippet()
    return opt
