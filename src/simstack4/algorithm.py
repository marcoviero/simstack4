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
                    n_populations = len(self.population_manager.populations)
                    self.progress_tracker = ProgressTracker(
                        n_populations, len(self.sky_maps.maps)
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
                if self.bootstrap_method == "per_bin":
                    n_populations = len(self.population_manager.populations)
                    total_solves = (
                        n_populations * self.bootstrap_iterations * len(self.sky_maps.maps)
                    )
                    logger.info(
                        f"Per-bin: {n_populations} pops × "
                        f"{self.bootstrap_iterations} iters × "
                        f"{len(self.sky_maps.maps)} maps = {total_solves} solves, "
                        f"avg {total_time / total_solves:.2f}s each"
                    )
                else:
                    total_computations = (
                        self.bootstrap_iterations * len(self.sky_maps.maps)
                    )
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
        Per-bin bootstrap: split one population at a time, hold others fixed.

        Caches the base layer matrix and crop mask per map so that each
        iteration only rebuilds the 2 layers (A/B) that actually change.

        1. Full solve (all sources, no splitting) → flux estimates
        2. For each map: build base layers + crop data (cached)
        3. For each population k, for each map:
             - For M iterations:
                 - Build only A and B layers for pop k (2 PSF convolutions)
                 - Splice into cached base → (N_pop+1) × N_pix matrix
                 - Solve
                 - flux_k[i] = flux_A + flux_B
             - uncertainty_k = std(flux_k[1..M])
        4. Combine: flux from step 1, uncertainty from step 3
        """
        population_ids = list(self.population_manager.populations.keys())
        n_populations = len(population_ids)

        logger.info(
            f"Starting per_bin bootstrap: {self.bootstrap_iterations} iterations × "
            f"{n_populations} populations × {len(self.sky_maps.maps)} maps..."
        )

        # Step 1: Full solve for flux estimates
        full_solve_results = {}
        for map_name in self.sky_maps.maps.keys():
            full_solve_results[map_name] = self._stack_single_map_standard(map_name)

        # Step 2: Cache base layers and crop data per map
        map_cache = {}
        for map_name in self.sky_maps.maps.keys():
            map_cache[map_name] = self._build_per_bin_cache(
                map_name, population_ids
            )

        # Step 3: Per-bin error estimation
        per_bin_errors = {
            map_name: np.zeros(n_populations)
            for map_name in self.sky_maps.maps.keys()
        }

        for k, pop_id in enumerate(population_ids):
            population = self.population_manager.populations[pop_id]

            if population.n_sources == 0:
                logger.debug(f"  Skipping {pop_id}: no sources")
                continue

            logger.info(
                f"  Estimating error for {pop_id} "
                f"({population.n_sources} sources)..."
            )

            for map_name in self.sky_maps.maps.keys():
                self.progress_tracker.update(k + 1, map_name,
                    f"per_bin: {pop_id}")

                cache = map_cache[map_name]
                flux_k_samples = []

                for iteration in range(self.bootstrap_iterations):
                    rng = np.random.RandomState(
                        self.bootstrap_seed + k * self.bootstrap_iterations + iteration
                    )

                    # Split only population k
                    original_indices = population.indices
                    n_sources = len(original_indices)
                    n_A = int(n_sources * self.bootstrap_split_fraction)

                    shuffled = rng.permutation(original_indices)
                    indices_A = shuffled[:n_A]
                    indices_B = shuffled[n_A:]

                    # Build only the 2 split layers (the only work per iteration)
                    layer_A = self._build_single_layer(map_name, indices_A)
                    layer_B = self._build_single_layer(map_name, indices_B)

                    # Crop to cached pixel mask
                    layer_A_cropped = layer_A[cache["pixel_indices"]]
                    layer_B_cropped = layer_B[cache["pixel_indices"]]

                    # Assemble iteration matrix: base with row k → [A, B]
                    # base_layers_cropped is (N_pop, N_cropped_pix)
                    # Result is (N_pop+1, N_cropped_pix)
                    iter_matrix = np.empty(
                        (n_populations + 1, cache["n_pixels"]),
                        dtype=cache["base_layers_cropped"].dtype,
                    )
                    # Rows before k: unchanged base layers
                    iter_matrix[:k] = cache["base_layers_cropped"][:k]
                    # Row k: A layer
                    iter_matrix[k] = layer_A_cropped
                    # Row k+1: B layer
                    iter_matrix[k + 1] = layer_B_cropped
                    # Rows after k: shifted by 1 from base
                    iter_matrix[k + 2:] = cache["base_layers_cropped"][k + 1:]

                    # Add foreground if requested
                    if self.add_foreground:
                        iter_matrix = np.vstack([
                            iter_matrix, cache["foreground_row"][np.newaxis, :]
                        ])

                    # Solve
                    flux_densities, _, _ = self._solve_linear_system(
                        iter_matrix, cache["observed_vector"], cache["map_data"]
                    )

                    # A is at index k, B at index k+1
                    flux_k_total = flux_densities[k] + flux_densities[k + 1]
                    flux_k_samples.append(flux_k_total)

                per_bin_errors[map_name][k] = np.std(flux_k_samples, ddof=1)

        # Step 4: Combine full-solve fluxes with per-bin errors
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

    def _build_per_bin_cache(
        self, map_name: str, population_ids: list[str]
    ) -> dict[str, Any]:
        """
        Build cached base layers and crop data for a single map.

        Returns dict with:
          base_layers_cropped: (N_pop, N_pix) array of cropped base layers
          pixel_indices: flat indices for cropping full layers
          observed_vector: cropped+mean-subtracted observed map
          foreground_row: cropped foreground layer (ones)
          map_data: MapData for the solver
          n_pixels: number of cropped pixels
        """
        map_data = self.sky_maps[map_name]

        # Build base layers (N_pop × N_full_pix)
        layer_specs = []
        for pid in population_ids:
            pop = self.population_manager.populations[pid]
            layer_specs.append((pid, pop.indices))

        base_layers_full, _ = self._create_layer_matrix(map_name, layer_specs)

        # Compute crop/valid pixel mask (identical for all iterations)
        if self.crop_circles:
            # _crop_to_circles returns (cropped_layers, cropped_obs, flat_indices)
            _, cropped_observed, pixel_indices = self._crop_to_circles(
                base_layers_full, map_data.data, map_name
            )
            base_layers_cropped = base_layers_full[:, pixel_indices]
        else:
            observed_flat = map_data.data.ravel()
            valid_mask = ~np.isnan(observed_flat)
            pixel_indices = np.where(valid_mask)[0]
            base_layers_cropped = base_layers_full[:, pixel_indices]
            cropped_observed = observed_flat[pixel_indices]

        n_pixels = len(pixel_indices)
        foreground_row = np.ones(n_pixels)

        logger.info(
            f"  Cached {map_name}: {len(population_ids)} base layers × "
            f"{n_pixels} pixels"
        )

        return {
            "base_layers_cropped": base_layers_cropped,
            "pixel_indices": pixel_indices,
            "observed_vector": cropped_observed,
            "foreground_row": foreground_row,
            "map_data": map_data,
            "n_pixels": n_pixels,
        }

    def _build_single_layer(
        self, map_name: str, source_indices: np.ndarray
    ) -> np.ndarray:
        """
        Build a single PSF-convolved, mean-subtracted layer from source indices.

        Returns flat array of length N_full_pixels (before cropping).
        """
        map_data = self.sky_maps[map_name]
        map_shape = map_data.shape

        if len(source_indices) == 0:
            return np.zeros(map_shape[0] * map_shape[1])

        ra, dec = self._get_coordinates_from_indices(source_indices)
        return self._create_and_convolve_layer(map_name, ra, dec, map_shape)

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

        # Remove mean over same valid pixels used for map mean subtraction.
        # Using a different pixel set (e.g., all pixels including unobserved
        # zeros) causes systematic flux underestimation.
        valid_mask = map_data.valid_pixel_mask
        if valid_mask is not None and np.any(valid_mask):
            convolved_layer[valid_mask] -= np.mean(convolved_layer[valid_mask])
            convolved_layer[~valid_mask] = 0.0
        else:
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

        # Count sources per population
        n_sources = {}
        for pop_bin in self.population_manager.iter_populations():
            n_sources[pop_bin.id_label] = pop_bin.n_sources

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
