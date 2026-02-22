"""
Core stacking algorithm for Simstack4

This module implements the simultaneous stacking algorithm that fits
multiple population layers to observed maps, replacing the nested loop
structure from simstack3 with a more efficient matrix-based approach.
"""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from .config import SimstackConfig
from .exceptions.simstack_exceptions import (
    AlgorithmError,
    ValidationError,
)
from .populations import PopulationManager
from .sky_maps import MapData, SkyMaps
from .utils import memory_usage_gb, setup_logging

logger = setup_logging()


@dataclass
class BootstrapSplit:
    """Container for a single bootstrap split of a population"""

    population_id: str
    split_type: str  # "A" or "B"
    source_indices: np.ndarray  # Indices into the original catalog
    n_sources: int

    def __post_init__(self):
        self.n_sources = len(self.source_indices)


class ProgressTracker:
    """Track and report progress for bootstrap iterations"""

    def __init__(self, total_iterations: int, total_maps: int):
        self.total_iterations = total_iterations
        self.total_maps = total_maps
        self.total_steps = total_iterations * total_maps
        self.current_step = 0
        self.start_time = time.time()

    def update(self, iteration: int, map_name: str, step_description: str = ""):
        """Update progress and log status"""
        self.current_step += 1
        elapsed = time.time() - self.start_time

        progress_pct = (self.current_step / self.total_steps) * 100

        if self.current_step > 1:
            avg_time_per_step = elapsed / self.current_step
            remaining = (self.total_steps - self.current_step) * avg_time_per_step
            eta_str = f"{remaining:.0f}s" if remaining < 60 else f"{remaining / 60:.1f}m"
        else:
            eta_str = "..."

        memory_gb = memory_usage_gb()
        logger.info(
            f"  [{progress_pct:5.1f}%] iter {iteration}/{self.total_iterations}, "
            f"{map_name} | elapsed {elapsed:.0f}s, ETA {eta_str} | "
            f"mem {memory_gb:.2f}GB {step_description}"
        )


@dataclass
class StackingResults:
    """Container for stacking results"""

    flux_densities: dict[str, np.ndarray]  # [map_name][population_index] - mean values
    flux_errors: dict[
        str, np.ndarray
    ]  # [map_name][population_index] - bootstrap errors
    flux_errors_systematic: dict[str, np.ndarray]  # Traditional least-squares errors
    bootstrap_flux_samples: dict[
        str, np.ndarray
    ] | None  # [map_name][bootstrap_iter, population_index]
    population_labels: list[str]
    map_names: list[str]
    n_sources: dict[str, int]  # sources per population
    chi_squared: dict[str, float]  # chi2 per map
    degrees_of_freedom: dict[str, int]
    reduced_chi_squared: dict[str, float]
    execution_time: float
    memory_used_gb: float
    bootstrap_enabled: bool
    bootstrap_iterations: int
    bootstrap_split_fraction: float  # What fraction goes to "A" vs "B" (e.g., 0.8 for 80:20)
    bin_properties: dict[str, dict[str, float]] | None = None  # {pop_label: {col: median}}

    def __post_init__(self):
        """Calculate derived quantities"""
        for map_name in self.map_names:
            if map_name in self.chi_squared and map_name in self.degrees_of_freedom:
                dof = self.degrees_of_freedom[map_name]
                if dof > 0:
                    self.reduced_chi_squared[map_name] = (
                        self.chi_squared[map_name] / dof
                    )


class SimstackAlgorithm:
    """
    Core simultaneous stacking algorithm

    This class implements the matrix-based simultaneous fitting approach
    that replaces the nested population loops from simstack3.
    """

    def __init__(
        self,
        config: SimstackConfig,
        population_manager: PopulationManager,
        sky_maps: SkyMaps,
    ):
        self.config = config
        self.population_manager = population_manager
        self.sky_maps = sky_maps

        # Algorithm settings
        self.crop_circles = config.binning.crop_circles
        self.add_foreground = config.binning.add_foreground

        # Bootstrap settings
        self.bootstrap_enabled = config.error_estimator.bootstrap.enabled
        self.bootstrap_method = config.error_estimator.bootstrap.method
        self.bootstrap_iterations = config.error_estimator.bootstrap.iterations
        self.bootstrap_split_fraction = config.error_estimator.bootstrap.split_fraction
        self.bootstrap_seed = config.error_estimator.bootstrap.initial_seed

        # Validate method
        if self.bootstrap_method not in ("all_bins", "per_bin"):
            raise ValidationError(
                f"Unknown bootstrap method '{self.bootstrap_method}', "
                f"expected 'all_bins' or 'per_bin'"
            )

        # Progress tracking
        self.progress_tracker = None

        logger.info("SimstackAlgorithm initialized:")
        logger.info(f"  {len(population_manager)} populations, {len(sky_maps)} maps")
        logger.info(
            f"  crop_circles={self.crop_circles}, "
            f"add_foreground={self.add_foreground}, "
            f"bootstrap={self.bootstrap_enabled}"
        )
        if self.bootstrap_enabled:
            logger.info(
                f"  {self.bootstrap_iterations} iterations, "
                f"method={self.bootstrap_method}, "
                f"split_fraction={self.bootstrap_split_fraction}"
            )

    def run_stacking(self):
        """Execute stacking algorithm"""
        start_time = time.time()
        start_memory = memory_usage_gb()

        logger.info("Starting stacking analysis...")

        try:
            self._validate_inputs()

            if self.bootstrap_enabled:
                if self.bootstrap_method == "per_bin":
                    # per_bin progress: one update per (population, map) pair
                    n_pops = len(self.population_manager.populations)
                    self.progress_tracker = ProgressTracker(
                        n_pops, len(self.sky_maps.maps)
                    )
                    results_dict = self._run_per_bin_error_estimation()
                else:
                    self.progress_tracker = ProgressTracker(
                        self.bootstrap_iterations, len(self.sky_maps.maps)
                    )
                    results_dict = self._run_all_bins_error_estimation()
            else:
                logger.info("Running single stacking (no bootstrap)...")
                results_dict = self._run_single_stacking()

            # Compile results
            self.results = self._compile_results(results_dict, start_time, start_memory)

            # Final summary
            total_time = time.time() - start_time
            final_memory = memory_usage_gb()

            logger.info(f"Stacking completed in {total_time / 60:.1f} minutes")
            logger.info(f"Memory: {start_memory:.2f} -> {final_memory:.2f} GB")

            if self.bootstrap_enabled:
                total_computations = self.bootstrap_iterations * len(self.sky_maps.maps)
                logger.info(
                    f"Avg time per (iteration × map): "
                    f"{total_time / total_computations:.1f}s"
                )

            return self.results

        except Exception as e:
            logger.error(f"Stacking failed: {e}")
            raise AlgorithmError(f"Stacking failed: {e}") from e

    def _run_all_bins_error_estimation(self) -> dict[str, dict[str, Any]]:
        """
        All-bins bootstrap: split every population per iteration.

        Flux estimates come from the full solve (no splitting).
        Bootstrap std across iterations gives the error estimate.
        """
        bootstrap_results = {map_name: [] for map_name in self.sky_maps.maps.keys()}

        logger.info(
            f"Starting all_bins bootstrap: {self.bootstrap_iterations} iterations "
            f"across {len(self.sky_maps.maps)} maps..."
        )

        for bootstrap_iter in range(self.bootstrap_iterations):
            # Create bootstrap splits for this iteration
            bootstrap_splits = self._create_bootstrap_splits(
                seed=self.bootstrap_seed + bootstrap_iter
            )

            # Process each map
            for map_name in self.sky_maps.maps.keys():
                self.progress_tracker.update(bootstrap_iter + 1, map_name)

                map_results = self._stack_single_map_with_bootstrap_splits(
                    map_name, bootstrap_splits
                )
                bootstrap_results[map_name].append(map_results["total_flux_densities"])

        logger.info("Bootstrap iterations complete. Computing statistics...")

        # Full solve for flux estimates; bootstrap std for errors
        final_results = {}
        for map_name in self.sky_maps.maps.keys():
            bootstrap_fluxes = np.array(bootstrap_results[map_name])
            bootstrap_errors = np.std(bootstrap_fluxes, axis=0, ddof=1)

            # Full solve (all sources, no splitting) gives the flux estimates
            full_results = self._stack_single_map_standard(map_name)

            final_results[map_name] = {
                "flux_densities": full_results["flux_densities"],
                "flux_errors": bootstrap_errors,
                "flux_errors_systematic": full_results["flux_errors"],
                "bootstrap_flux_samples": bootstrap_fluxes,
                "population_labels": full_results["population_labels"],
                "fit_statistics": full_results["fit_statistics"],
                "n_valid_pixels": full_results["n_valid_pixels"],
            }

            logger.debug(
                f"  {map_name}: mean|flux|={np.mean(np.abs(full_results['flux_densities'])):.2e}, "
                f"mean_err={np.mean(bootstrap_errors):.2e}"
            )

        return final_results

    def _run_per_bin_error_estimation(self) -> dict[str, dict[str, Any]]:
        """
        Per-bin bootstrap using Gram matrix solve + direct PSF stamping.

        Avoids redundant PSF convolutions and enormous lstsq solves.

        Strategy per map (processed sequentially to limit peak memory):
          1. Build all base layers (one FFT convolution each), crop, cache.
          2. Compute Gram matrix G = AᵀA  (n_layers × n_layers — tiny).
             Compute projection  h = Aᵀb  (n_layers — tiny).
          3. Full-solve via G x = h.
          4. Pre-compute crop geometry + PSF kernel for direct stamping.
          5. For each target population k, for M iterations:
               - Split k into A/B source sets.
               - Stamp PSF at A-source pixel positions → cropped layer_A.
               - layer_B = cached_layer_k − layer_A  (no convolution).
               - Update G and h with rank-2 replace → solve tiny system.
               - Record flux_k = x_A + x_B.
          6. Free this map's cache.
        """
        population_ids = list(self.population_manager.populations.keys())
        n_populations = len(population_ids)

        logger.info(
            f"Starting per_bin bootstrap: {self.bootstrap_iterations} iterations × "
            f"{n_populations} populations × {len(self.sky_maps.maps)} maps..."
        )

        # Step 1: Full solve for flux estimates (reused for final output)
        full_solve_results = {}
        for map_name in self.sky_maps.maps.keys():
            full_solve_results[map_name] = self._stack_single_map_standard(map_name)

        # Step 2: Per-bin error estimation — one map at a time
        per_bin_errors = {
            map_name: np.zeros(n_populations)
            for map_name in self.sky_maps.maps.keys()
        }

        for map_name in self.sky_maps.maps.keys():
            t_map_start = time.time()
            map_data = self.sky_maps[map_name]

            # ---- 2a. Build & cache all cropped base layers ----
            cache, crop_info = self._build_per_bin_cache(
                map_name, population_ids
            )
            # cache: (n_layers, n_crop_pixels) — base layers + foreground
            # crop_info: dict with flat_indices, obs_vector, etc.

            n_layers = cache.shape[0]
            obs_vector = crop_info["obs_vector"]

            # ---- 2b. Gram matrix and projection ----
            G_base = cache @ cache.T          # (n_layers, n_layers)
            h_base = cache @ obs_vector       # (n_layers,)

            # ---- 2c. Get PSF kernel + crop geometry ----
            psf_kernel = self.sky_maps.create_psf_kernel(map_name)
            psf_half = psf_kernel.shape[0] // 2
            kernel_sum = np.sum(psf_kernel)
            flat_to_crop = crop_info["flat_to_crop"]
            flat_indices = crop_info["flat_indices"]
            map_shape = map_data.shape
            n_full = map_shape[0] * map_shape[1]

            logger.info(
                f"  {map_name}: cache {cache.shape[0]}×{cache.shape[1]} "
                f"({cache.nbytes / 1e9:.2f} GB), "
                f"PSF {psf_kernel.shape}, G {G_base.shape}"
            )

            # ---- 2d. Per-bin iterations ----
            for k in range(n_populations):
                pop = self.population_manager.populations[population_ids[k]]
                if pop.n_sources == 0:
                    continue

                self.progress_tracker.update(
                    k + 1, map_name, f"per_bin: {population_ids[k]}"
                )

                # Pre-compute pixel positions for this population's sources
                # (one WCS transform per population, not per iteration)
                ra_all, dec_all = self._get_coordinates_from_indices(pop.indices)
                x_pix_all, y_pix_all = self.sky_maps.world_to_pixel(
                    map_name, ra_all, dec_all
                )
                # y=row, x=col  (FITS convention)
                rows_all = np.round(y_pix_all).astype(np.intp)
                cols_all = np.round(x_pix_all).astype(np.intp)

                flux_k_samples = []

                for iteration in range(self.bootstrap_iterations):
                    rng = np.random.RandomState(
                        self.bootstrap_seed
                        + k * self.bootstrap_iterations
                        + iteration
                    )

                    # Split sources
                    n_sources = len(pop.indices)
                    n_A = int(n_sources * self.bootstrap_split_fraction)
                    perm = rng.permutation(n_sources)
                    idx_A = perm[:n_A]

                    # Build layer_A via direct PSF stamping with real kernel
                    # (exact match to convolve_with_psf, ~1000× faster)
                    layer_A = self._stamp_psf_cropped(
                        rows_all[idx_A], cols_all[idx_A],
                        psf_kernel, psf_half,
                        flat_to_crop, map_shape,
                        crop_info["n_crop"],
                    )
                    # Mean-subtract to match base layers (which were mean-
                    # subtracted over the full map before cropping).
                    # For peak-normalised PSF, full-map sum of stamped
                    # layer ≈ n_A × kernel_sum.
                    full_map_mean_A = len(idx_A) * kernel_sum / n_full
                    layer_A -= full_map_mean_A

                    # layer_B = base_k − layer_A  (no convolution)
                    layer_B = cache[k] - layer_A

                    # ---- Gram matrix update ----
                    # Cross-products with all base layers
                    A_dot_base = cache @ layer_A   # (n_layers,)
                    B_dot_base = cache @ layer_B   # (n_layers,)

                    A_dot_A = np.dot(layer_A, layer_A)
                    B_dot_B = np.dot(layer_B, layer_B)
                    A_dot_B = np.dot(layer_A, layer_B)

                    h_A = np.dot(layer_A, obs_vector)
                    h_B = np.dot(layer_B, obs_vector)

                    # Build (n_layers+1) × (n_layers+1) system:
                    # row/col k → A, insert new row/col k+1 → B
                    n_new = n_layers + 1
                    G_new = np.empty((n_new, n_new))
                    h_new = np.empty(n_new)

                    # Map: old indices [0..k-1, k+1..n_layers-1] → new [0..k-1, k+2..n_new-1]
                    old_keep = np.concatenate(
                        [np.arange(k), np.arange(k + 1, n_layers)]
                    )
                    new_keep = np.concatenate(
                        [np.arange(k), np.arange(k + 2, n_new)]
                    )

                    # Unchanged block
                    G_new[np.ix_(new_keep, new_keep)] = G_base[
                        np.ix_(old_keep, old_keep)
                    ]
                    h_new[new_keep] = h_base[old_keep]

                    # A row/col (index k)
                    G_new[k, new_keep] = A_dot_base[old_keep]
                    G_new[new_keep, k] = A_dot_base[old_keep]
                    G_new[k, k] = A_dot_A
                    G_new[k, k + 1] = A_dot_B
                    G_new[k + 1, k] = A_dot_B

                    # B row/col (index k+1)
                    G_new[k + 1, new_keep] = B_dot_base[old_keep]
                    G_new[new_keep, k + 1] = B_dot_base[old_keep]
                    G_new[k + 1, k + 1] = B_dot_B

                    h_new[k] = h_A
                    h_new[k + 1] = h_B

                    # Solve tiny system
                    try:
                        x_new = np.linalg.solve(G_new, h_new)
                    except np.linalg.LinAlgError:
                        x_new, _, _, _ = np.linalg.lstsq(
                            G_new, h_new, rcond=None
                        )

                    flux_k_total = x_new[k] + x_new[k + 1]
                    flux_k_samples.append(flux_k_total)

                per_bin_errors[map_name][k] = np.std(flux_k_samples, ddof=1)

            # Free this map's cache
            del cache, G_base, h_base, obs_vector
            elapsed = time.time() - t_map_start
            logger.info(
                f"  {map_name} per_bin complete: {elapsed:.0f}s "
                f"({elapsed / 60:.1f} min)"
            )

        # Step 3: Combine full-solve fluxes with per-bin errors
        final_results = {}
        for map_name in self.sky_maps.maps.keys():
            full = full_solve_results[map_name]

            errors = per_bin_errors[map_name]
            if self.add_foreground:
                errors = np.append(errors, 0.0)

            final_results[map_name] = {
                "flux_densities": full["flux_densities"],
                "flux_errors": errors,
                "flux_errors_systematic": full["flux_errors"],
                "bootstrap_flux_samples": None,
                "population_labels": full["population_labels"],
                "fit_statistics": full["fit_statistics"],
                "n_valid_pixels": full["n_valid_pixels"],
            }

            logger.debug(
                f"  {map_name}: mean|flux|={np.mean(np.abs(full['flux_densities'])):.2e}, "
                f"mean_err={np.mean(per_bin_errors[map_name]):.2e}"
            )

        return final_results

    # ------------------------------------------------------------------
    # Per-bin helpers
    # ------------------------------------------------------------------

    def _build_per_bin_cache(
        self,
        map_name: str,
        population_ids: list[str],
    ) -> tuple[np.ndarray, dict]:
        """
        Build cropped base layers for one map + crop geometry.

        Returns
        -------
        cache : (n_layers, n_crop_pixels) float64
            Cropped, mean-subtracted base layers (+ foreground if enabled).
        crop_info : dict
            flat_to_crop : (n_full_pixels,) int32 — maps flat pixel index to
                crop index (-1 if not in crop mask).
            obs_vector : (n_crop_pixels,) — cropped, mean-subtracted observed map.
            n_crop : int — number of cropped pixels.
        """
        map_data = self.sky_maps[map_name]
        map_shape = map_data.shape
        n_full = map_shape[0] * map_shape[1]
        n_pops = len(population_ids)

        # Build full layer matrix (one convolution per population)
        layer_list = []
        for pid in population_ids:
            pop = self.population_manager.populations[pid]
            if pop.n_sources > 0:
                ra, dec = self._get_coordinates_from_indices(pop.indices)
                layer = self._create_and_convolve_layer(
                    map_name, ra, dec, map_shape
                )
            else:
                layer = np.zeros(n_full)
            layer_list.append(layer)

        if self.add_foreground:
            layer_list.append(np.ones(n_full))

        full_matrix = np.array(layer_list)  # (n_layers, n_full)
        n_layers = full_matrix.shape[0]

        # Crop
        if self.crop_circles:
            full_matrix, obs_vector, flat_indices = self._crop_to_circles(
                full_matrix, map_data.data, map_name
            )
        else:
            obs_flat = map_data.data.ravel()
            valid = ~np.isnan(obs_flat)
            flat_indices = np.where(valid)[0]
            obs_vector = obs_flat[flat_indices]
            obs_vector -= np.nanmean(obs_vector)
            full_matrix = full_matrix[:, flat_indices]

        n_crop = len(flat_indices)

        # Build flat→crop lookup
        flat_to_crop = np.full(n_full, -1, dtype=np.int32)
        flat_to_crop[flat_indices] = np.arange(n_crop, dtype=np.int32)

        logger.info(
            f"  Cached {map_name}: {n_layers} layers × {n_crop} pixels "
            f"({full_matrix.nbytes / 1e9:.2f} GB)"
        )

        crop_info = {
            "flat_to_crop": flat_to_crop,
            "obs_vector": obs_vector,
            "n_crop": n_crop,
            "flat_indices": flat_indices,
        }

        return full_matrix, crop_info

    def _build_psf_kernel(self, map_name: str) -> tuple[np.ndarray, int]:
        """
        Build Gaussian PSF kernel from beam FWHM.

        Replicates sky_maps.create_psf_kernel() exactly:
        - Peak-normalised (peak=1)
        - kernel_size = int(6*sigma), forced odd
        - Gaussian2DKernel equivalent

        Used only in tests or if create_psf_kernel is unavailable.

        Returns (kernel_2d, half_size).
        """
        map_data = self.sky_maps[map_name]
        fwhm_pix = map_data.beam_fwhm_pixels
        sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Match sky_maps.py: kernel_size = int(6 * sigma), forced odd
        kernel_size = int(6 * sigma)
        if kernel_size % 2 == 0:
            kernel_size += 1
        half = kernel_size // 2

        y, x = np.mgrid[-half : half + 1, -half : half + 1]
        kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel /= kernel.max()  # peak = 1

        return kernel, half

    def _stamp_psf_cropped(
        self,
        src_rows: np.ndarray,
        src_cols: np.ndarray,
        psf_kernel: np.ndarray,
        psf_half: int,
        flat_to_crop: np.ndarray,
        map_shape: tuple[int, int],
        n_crop: int,
    ) -> np.ndarray:
        """
        Stamp PSF at source positions directly into a cropped-pixel buffer.

        ~1000× faster than FFT convolution for typical population sizes,
        because we only touch pixels within ±psf_half of each source.

        Parameters
        ----------
        src_rows, src_cols : (n_sources,) int
            Pixel positions (row, col) of sources to stamp.
        psf_kernel : (2*psf_half+1, 2*psf_half+1)
            Normalised PSF kernel.
        flat_to_crop : (n_full_pixels,) int32
            Maps flat pixel index → crop index (-1 if outside crop).
        map_shape : (n_rows, n_cols)
        n_crop : number of cropped pixels.

        Returns
        -------
        layer : (n_crop,) float64
        """
        n_rows, n_cols = map_shape
        layer = np.zeros(n_crop, dtype=np.float64)

        # PSF support as relative offsets + values (skip near-zero entries)
        psf_size = 2 * psf_half + 1
        threshold = psf_kernel.max() * 1e-4
        ky, kx = np.where(psf_kernel > threshold)
        kvals = psf_kernel[ky, kx]
        k_dr = ky - psf_half  # relative row offsets
        k_dc = kx - psf_half  # relative col offsets
        n_kpix = len(kvals)

        # Vectorised stamping: outer sums of (n_src, n_kpix) positions
        all_rows = src_rows[:, None] + k_dr[None, :]  # (n_src, n_kpix)
        all_cols = src_cols[:, None] + k_dc[None, :]

        # Bounds check
        valid = (
            (all_rows >= 0) & (all_rows < n_rows)
            & (all_cols >= 0) & (all_cols < n_cols)
        )

        all_flat = np.where(valid, all_rows * n_cols + all_cols, 0)
        all_crop = flat_to_crop[all_flat.ravel()].reshape(all_flat.shape)
        valid &= all_crop >= 0  # within crop mask

        # Scatter-add PSF values
        v_mask = valid.ravel()
        v_crop_idx = all_crop.ravel()[v_mask]
        # Broadcast PSF values across sources
        v_vals = np.broadcast_to(
            kvals[None, :], (len(src_rows), n_kpix)
        ).ravel()[v_mask]

        np.add.at(layer, v_crop_idx, v_vals)

        return layer

    def _create_bootstrap_splits(
        self, seed: int
    ) -> dict[str, dict[str, BootstrapSplit]]:
        """Create A/B source splits for all populations."""
        np.random.seed(seed)
        bootstrap_splits = {}

        for pop_id, population in self.population_manager.populations.items():
            if population.n_sources == 0:
                bootstrap_splits[pop_id] = {
                    "A": BootstrapSplit(pop_id, "A", np.array([]), 0),
                    "B": BootstrapSplit(pop_id, "B", np.array([]), 0),
                }
                continue

            original_indices = population.indices
            n_sources = len(original_indices)
            n_A = int(n_sources * self.bootstrap_split_fraction)

            shuffled_indices = np.random.permutation(original_indices)
            indices_A = shuffled_indices[:n_A]
            indices_B = shuffled_indices[n_A:]

            bootstrap_splits[pop_id] = {
                "A": BootstrapSplit(pop_id, "A", indices_A, n_A),
                "B": BootstrapSplit(pop_id, "B", indices_B, n_sources - n_A),
            }

        return bootstrap_splits

    def _stack_single_map_with_bootstrap_splits(
        self, map_name: str, bootstrap_splits: dict[str, dict[str, BootstrapSplit]]
    ) -> dict[str, Any]:
        """Stack single map using A/B split layers for all populations."""
        population_ids = list(bootstrap_splits.keys())
        n_populations = len(population_ids)

        # Build layer specs: A layers then B layers
        layer_specs = []
        for pop_id in population_ids:
            split_A = bootstrap_splits[pop_id]["A"]
            layer_specs.append((f"{pop_id}_A", split_A.source_indices))
        for pop_id in population_ids:
            split_B = bootstrap_splits[pop_id]["B"]
            layer_specs.append((f"{pop_id}_B", split_B.source_indices))

        # Use shared stacking pipeline
        solve_result = self._stack_single_map(map_name, layer_specs)

        # Extract A and B fluxes and sum to get total per population
        combined = solve_result["flux_densities"]
        if self.add_foreground:
            flux_A = combined[:n_populations]
            flux_B = combined[n_populations : 2 * n_populations]
            foreground_flux = combined[2 * n_populations]
        else:
            flux_A = combined[:n_populations]
            flux_B = combined[n_populations : 2 * n_populations]
            foreground_flux = 0.0

        total_flux_densities = flux_A + flux_B

        final_labels = population_ids.copy()
        if self.add_foreground:
            final_labels.append("foreground")
            total_flux_densities = np.append(total_flux_densities, foreground_flux)

        return {
            "total_flux_densities": total_flux_densities,
            "population_labels": final_labels,
            "fit_statistics": solve_result["fit_statistics"],
            "n_valid_pixels": solve_result["n_valid_pixels"],
        }

    def _create_layer_matrix(
        self,
        map_name: str,
        layer_specs: list[tuple[str, np.ndarray]],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Create layer matrix from a list of (label, source_indices) pairs.

        This is the single entry point for building layer matrices, used by both
        standard stacking (one layer per population) and bootstrap stacking
        (A/B layers per population).
        """
        map_data = self.sky_maps[map_name]
        map_shape = map_data.shape
        n_pixels = map_shape[0] * map_shape[1]
        n_layers = len(layer_specs)

        layer_matrix = np.zeros((n_layers, n_pixels))
        layer_labels = []

        for i, (label, source_indices) in enumerate(layer_specs):
            if len(source_indices) > 0:
                ra, dec = self._get_coordinates_from_indices(source_indices)
                layer = self._create_and_convolve_layer(map_name, ra, dec, map_shape)
                layer_matrix[i, :] = layer

            layer_labels.append(label)

        return layer_matrix, layer_labels

    def _get_coordinates_from_indices(
        self, source_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract RA, Dec from catalog using source indices

        This is the CLEAN way to get coordinates - no population manager lookup needed.
        """
        if len(source_indices) == 0:
            return np.array([]), np.array([])

        catalog_df = self.population_manager.catalog_df

        # Get column names
        if (
            hasattr(self.population_manager, "full_config")
            and self.population_manager.full_config
        ):
            ra_col = self.population_manager.full_config.catalog.astrometry["ra"]
            dec_col = self.population_manager.full_config.catalog.astrometry["dec"]
        else:
            ra_col = "ALPHA_J2000"
            dec_col = "DELTA_J2000"

        try:
            subset = catalog_df.iloc[source_indices]
            ra = subset[ra_col].values
            dec = subset[dec_col].values
            return ra, dec
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to extract coordinates: {e}")
            raise

    def _create_and_convolve_layer(
        self, map_name: str, ra: np.ndarray, dec: np.ndarray, map_shape: tuple[int, int]
    ) -> np.ndarray:
        """Create and convolve a source layer"""
        from .toolbox import SimstackToolbox

        if len(ra) == 0:
            return np.zeros(map_shape[0] * map_shape[1])

        map_data = self.sky_maps[map_name]

        # Create source layer
        source_layer = SimstackToolbox.create_source_layer(
            ra=ra, dec=dec, wcs=map_data.wcs, shape=map_shape
        )

        # Convolve with PSF
        convolved_layer = self.sky_maps.convolve_with_psf(source_layer, map_name)

        # Remove mean and flatten
        convolved_layer -= np.nanmean(convolved_layer)
        return convolved_layer.ravel()

    def _crop_to_circles(
        self,
        layer_matrix: np.ndarray,
        observed_map: np.ndarray,
        map_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crop fitting to circular regions around all source positions.

        The mask is built from the full population source lists (which is correct
        for both standard and bootstrap modes, since A ∪ B = full set).
        """
        map_data = self.sky_maps[map_name]

        mask = np.zeros(observed_map.shape, dtype=bool)
        circle_radius_pix = max(3.0, map_data.beam_fwhm_pixels)

        for pop_bin in self.population_manager.iter_populations():
            if pop_bin.n_sources == 0:
                continue

            try:
                ra, dec = self._get_coordinates_from_indices(pop_bin.indices)
                x_pix, y_pix = self.sky_maps.world_to_pixel(map_name, ra, dec)

                for x, y in zip(x_pix, y_pix, strict=True):
                    ix, iy = int(np.round(x)), int(np.round(y))
                    if (
                        0 <= ix < observed_map.shape[1]
                        and 0 <= iy < observed_map.shape[0]
                    ):
                        self._add_circle_to_mask(mask, x, y, circle_radius_pix)

            except Exception as e:
                logger.warning(f"Error processing population {pop_bin.id_label}: {e}")
                continue

        valid_pixels_2d = mask & ~np.isnan(observed_map)

        if np.sum(valid_pixels_2d) < layer_matrix.shape[0]:
            logger.warning("Too few pixels for fitting, using all valid pixels")
            valid_pixels_2d = ~np.isnan(observed_map)

        flat_indices = np.where(valid_pixels_2d.ravel())[0]
        cropped_observed = observed_map.ravel()[flat_indices]
        cropped_observed -= np.nanmean(cropped_observed)
        cropped_layers = layer_matrix[:, flat_indices]

        return cropped_layers, cropped_observed, flat_indices

    def _add_circle_to_mask(
        self, mask: np.ndarray, center_x: float, center_y: float, radius: float
    ) -> None:
        """Add circular region to mask"""
        ny, nx = mask.shape
        y_min = max(0, int(center_y - radius))
        y_max = min(ny, int(center_y + radius) + 1)
        x_min = max(0, int(center_x - radius))
        x_max = min(nx, int(center_x + radius) + 1)

        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
        circle_mask = distances <= radius
        mask[y_min:y_max, x_min:x_max][circle_mask] = True

    def _run_single_stacking(self) -> dict[str, dict[str, Any]]:
        """Run stacking without bootstrap"""
        results_dict = {}

        for map_name in self.sky_maps.maps.keys():
            map_results = self._stack_single_map_standard(map_name)
            map_results["bootstrap_flux_samples"] = None
            map_results["flux_errors_systematic"] = map_results["flux_errors"]
            results_dict[map_name] = map_results

        return results_dict

    def _stack_single_map_standard(self, map_name: str) -> dict[str, Any]:
        """Standard stacking without bootstrap splits"""
        # Build layer specs: one layer per population
        layer_specs = []
        for pop_bin in self.population_manager.iter_populations():
            layer_specs.append((pop_bin.id_label, pop_bin.indices))

        return self._stack_single_map(map_name, layer_specs)

    def _stack_single_map(
        self,
        map_name: str,
        layer_specs: list[tuple[str, np.ndarray]],
    ) -> dict[str, Any]:
        """
        Shared stacking pipeline: create layers → add foreground → crop → solve.

        Args:
            map_name: Name of the map to stack
            layer_specs: List of (label, source_indices) pairs for building layers
        """
        map_data = self.sky_maps[map_name]

        # Create layer matrix
        layer_matrix, population_labels = self._create_layer_matrix(
            map_name, layer_specs
        )

        # Add foreground if requested
        if self.add_foreground:
            foreground_layer = np.ones_like(layer_matrix[0])
            layer_matrix = np.vstack([layer_matrix, foreground_layer[np.newaxis, :]])
            population_labels.append("foreground")

        # Crop if requested
        if self.crop_circles:
            layer_matrix, observed_vector, valid_mask = self._crop_to_circles(
                layer_matrix, map_data.data, map_name
            )
        else:
            observed_vector = map_data.data.ravel()
            valid_mask = ~np.isnan(observed_vector)
            layer_matrix = layer_matrix[:, valid_mask]
            observed_vector = observed_vector[valid_mask]

        # Solve
        flux_densities, flux_errors, fit_stats = self._solve_linear_system(
            layer_matrix, observed_vector, map_data
        )

        return {
            "flux_densities": flux_densities,
            "flux_errors": flux_errors,
            "population_labels": population_labels,
            "fit_statistics": fit_stats,
            "n_valid_pixels": len(observed_vector),
        }

    def _solve_linear_system(
        self, layer_matrix: np.ndarray, observed_vector: np.ndarray, map_data: MapData
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Solve linear system for flux densities"""
        n_populations, n_pixels = layer_matrix.shape

        if n_pixels == 0:
            raise AlgorithmError("No valid pixels for fitting")

        try:
            flux_densities, residuals, rank, singular_values = linalg.lstsq(
                layer_matrix.T, observed_vector
            )

            # Calculate systematic errors (bootstrap errors calculated separately)
            if n_pixels > n_populations:
                try:
                    if hasattr(residuals, "__len__") and len(residuals) > 0:
                        mse = residuals[0] / (n_pixels - n_populations)
                    else:
                        model_prediction = layer_matrix.T @ flux_densities
                        mse = np.sum((observed_vector - model_prediction) ** 2) / (
                            n_pixels - n_populations
                        )

                    covariance = linalg.inv(layer_matrix @ layer_matrix.T) * mse
                    systematic_errors = np.sqrt(np.diag(covariance))
                except linalg.LinAlgError:
                    systematic_errors = np.abs(flux_densities) * 0.1
            else:
                systematic_errors = np.abs(flux_densities) * 0.1

            # Calculate fit statistics
            model_prediction = layer_matrix.T @ flux_densities
            chi_squared = np.sum((observed_vector - model_prediction) ** 2)
            dof = max(1, n_pixels - n_populations)

            fit_stats = {
                "chi_squared": chi_squared,
                "degrees_of_freedom": dof,
                "reduced_chi_squared": chi_squared / dof,
                "rank": rank,
            }

            return flux_densities, systematic_errors, fit_stats

        except Exception as e:
            logger.error(f"Linear system solution failed: {e}")
            return (
                np.zeros(n_populations),
                np.ones(n_populations),
                {
                    "chi_squared": np.inf,
                    "degrees_of_freedom": 1,
                    "reduced_chi_squared": np.inf,
                },
            )

    def _validate_inputs(self) -> None:
        """Validate inputs"""
        if len(self.population_manager) == 0:
            raise ValidationError("No populations found")
        if len(self.sky_maps) == 0:
            raise ValidationError("No maps loaded")

    def _compile_results(self, results_dict, start_time, start_memory):
        """Compile results into final StackingResults object"""

        map_names = list(results_dict.keys())
        population_labels = results_dict[map_names[0]]["population_labels"]

        # Extract results per map
        flux_densities = {}
        flux_errors = {}
        flux_errors_systematic = {}
        bootstrap_flux_samples = {} if self.bootstrap_enabled else None
        chi_squared = {}
        degrees_of_freedom = {}

        for map_name, map_results in results_dict.items():
            flux_densities[map_name] = map_results["flux_densities"]
            flux_errors[map_name] = map_results["flux_errors"]
            flux_errors_systematic[map_name] = map_results.get(
                "flux_errors_systematic", map_results["flux_errors"]
            )

            if (
                self.bootstrap_enabled
                and map_results.get("bootstrap_flux_samples") is not None
            ):
                bootstrap_flux_samples[map_name] = map_results["bootstrap_flux_samples"]

            chi_squared[map_name] = map_results["fit_statistics"]["chi_squared"]
            degrees_of_freedom[map_name] = map_results["fit_statistics"][
                "degrees_of_freedom"
            ]

        # Count sources and compute per-bin median properties
        n_sources = {}
        bin_properties = {}
        property_columns = self.config.catalog.classification.all_property_columns
        catalog_df = self.population_manager.catalog_df

        for pop_bin in self.population_manager.iter_populations():
            n_sources[pop_bin.id_label] = pop_bin.n_sources

            if pop_bin.n_sources > 0 and property_columns:
                props = {}
                subset = catalog_df.iloc[pop_bin.indices]
                for col in property_columns:
                    if col in subset.columns:
                        vals = pd.to_numeric(subset[col], errors="coerce")
                        median_val = vals.median()
                        if pd.notna(median_val):
                            props[col] = float(median_val)
                    else:
                        logger.debug(
                            f"Column '{col}' not in catalog, "
                            f"skipping for {pop_bin.id_label}"
                        )
                bin_properties[pop_bin.id_label] = props
            else:
                bin_properties[pop_bin.id_label] = {}

        # Calculate timing and memory
        execution_time = time.time() - start_time
        current_memory = memory_usage_gb()
        memory_used = max(0, current_memory - start_memory)

        # Create the results object
        results = StackingResults(
            flux_densities=flux_densities,
            flux_errors=flux_errors,
            flux_errors_systematic=flux_errors_systematic,
            bootstrap_flux_samples=bootstrap_flux_samples,
            population_labels=population_labels,
            map_names=map_names,
            n_sources=n_sources,
            chi_squared=chi_squared,
            degrees_of_freedom=degrees_of_freedom,
            reduced_chi_squared={},
            execution_time=execution_time,
            memory_used_gb=memory_used,
            bootstrap_enabled=self.bootstrap_enabled,
            bootstrap_iterations=self.bootstrap_iterations,
            bootstrap_split_fraction=getattr(self, "bootstrap_split_fraction", 0.5),
            bin_properties=bin_properties if bin_properties else None,
        )

        logger.info(
            f"StackingResults compiled: {len(map_names)} maps, "
            f"{len(population_labels)} populations"
        )

        return results

    def print_results_summary(self) -> None:
        """Print a summary of stacking results"""
        if self.results is None:
            print("No results available - run stacking first")
            return

        results = self.results

        print("=== Stacking Results Summary ===")
        print(f"Execution time: {results.execution_time:.1f}s")
        print(f"Memory used: {results.memory_used_gb:.2f}GB")
        print(f"Populations: {len(results.population_labels)}")
        print(f"Maps: {len(results.map_names)}")
        if results.bootstrap_enabled:
            print(
                f"Bootstrap: {results.bootstrap_iterations} iterations, "
                f"split {results.bootstrap_split_fraction:.0%}/"
                f"{1 - results.bootstrap_split_fraction:.0%}"
            )
        print()

        print("Flux Densities (Jy):")
        print(
            f"{'Population':<30} "
            + " ".join(f"{name:<15}" for name in results.map_names)
        )
        print("-" * (30 + 16 * len(results.map_names)))

        for i, pop_label in enumerate(results.population_labels):
            line = f"{pop_label:<30}"
            for map_name in results.map_names:
                flux = results.flux_densities[map_name][i]
                error = results.flux_errors[map_name][i]
                sys_error = results.flux_errors_systematic[map_name][i]

                if results.bootstrap_enabled:
                    line += f" {flux:>8.2e}±{error:.1e}"
                else:
                    line += f" {flux:>8.2e}±{sys_error:.1e}*"
            print(line)

        if not results.bootstrap_enabled:
            print("* Least-squares errors only (no bootstrap)")

        if results.bootstrap_enabled:
            print()
            print("Bootstrap vs least-squares error ratio:")
            for map_name in results.map_names:
                bootstrap_err = np.mean(results.flux_errors[map_name])
                systematic_err = np.mean(results.flux_errors_systematic[map_name])
                ratio = bootstrap_err / systematic_err if systematic_err > 0 else np.inf
                print(f"  {map_name}: {ratio:.2f}")


def run_stacking(
    config: SimstackConfig, population_manager: PopulationManager, sky_maps: SkyMaps
):
    """Run stacking with proper bootstrap design"""
    algorithm = SimstackAlgorithm(config, population_manager, sky_maps)
    return algorithm.run_stacking()
