"""
Core stacking algorithm for Simstack4

This module implements the simultaneous stacking algorithm that fits
multiple population layers to observed maps, replacing the nested loop
structure from simstack3 with a more efficient matrix-based approach.
"""

import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import psutil
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
        self.step_times = []

    def update(self, iteration: int, map_name: str, step_description: str = ""):
        """Update progress and log detailed status"""
        self.current_step += 1
        current_time = time.time()
        elapsed = current_time - self.start_time

        # Calculate progress
        progress_pct = (self.current_step / self.total_steps) * 100

        # Estimate time remaining
        if self.current_step > 0:
            avg_time_per_step = elapsed / self.current_step
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = remaining_steps * avg_time_per_step
            eta_str = self._format_time(eta_seconds)
        else:
            eta_str = "calculating..."

        # Memory usage
        memory_gb = self._get_memory_usage()

        # Log detailed progress
        logger.info(
            f"  📍 Iteration {iteration}/{self.total_iterations}, Map: {map_name}"
        )
        logger.info(
            f"     Progress: {progress_pct:.1f}% ({self.current_step}/{self.total_steps})"
        )
        logger.info(f"     Elapsed: {self._format_time(elapsed)}, ETA: {eta_str}")
        logger.info(f"     Memory: {memory_gb:.2f}GB {step_description}")

        # Store timing for better estimates
        self.step_times.append(
            current_time - (self.step_times[-1] if self.step_times else self.start_time)
        )

    def _format_time(self, seconds: float) -> str:
        """Format time in human readable format"""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def _get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        try:
            process = psutil.Process(os.getpid())
            memory_bytes = process.memory_info().rss
            return memory_bytes / (1024**3)  # Convert to GB
        except Exception:
            return memory_usage_gb()  # Fallback to existing method


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
        self.bootstrap_iterations = config.error_estimator.bootstrap.iterations
        self.bootstrap_split_fraction = 0.5  # 50:50 split for better balance
        self.bootstrap_seed = config.error_estimator.bootstrap.initial_seed

        # Progress tracking
        self.progress_tracker = None

        logger.info("Enhanced Bootstrap Stacking Algorithm initialized:")
        logger.info(f"  - {len(population_manager)} populations")
        logger.info(f"  - {len(sky_maps)} maps")
        logger.info(f"  - Bootstrap enabled: {self.bootstrap_enabled}")
        if self.bootstrap_enabled:
            logger.info(f"  - Bootstrap iterations: {self.bootstrap_iterations}")
            logger.info(
                f"  - Bootstrap split: {self.bootstrap_split_fraction:.1%}:{1 - self.bootstrap_split_fraction:.1%}"
            )
            total_computations = self.bootstrap_iterations * len(sky_maps)
            logger.info(
                f"  - Total computations: {total_computations} (iterations × maps)"
            )

            # Estimate computation time
            avg_sources_per_pop = np.mean(
                [
                    pop.n_sources
                    for pop in population_manager.populations.values()
                    if pop.n_sources > 0
                ]
            )
            total_sources = sum(
                pop.n_sources for pop in population_manager.populations.values()
            )
            logger.info(f"  - Total sources: {total_sources:,}")
            logger.info(f"  - Avg sources/population: {avg_sources_per_pop:.1f}")

            # Rough time estimate (very approximate)
            if total_sources > 100000:
                estimated_minutes = (
                    total_computations * 2
                )  # ~2 min per computation for large datasets
                logger.info(
                    f"  - Estimated runtime: ~{estimated_minutes:.0f} minutes (large dataset)"
                )
            elif total_sources > 10000:
                estimated_minutes = total_computations * 0.5  # ~30s per computation
                logger.info(
                    f"  - Estimated runtime: ~{estimated_minutes:.0f} minutes (medium dataset)"
                )
            else:
                estimated_minutes = total_computations * 0.1  # ~6s per computation
                logger.info(
                    f"  - Estimated runtime: ~{estimated_minutes:.1f} minutes (small dataset)"
                )

    def run_stacking(self):
        """Execute stacking with enhanced progress tracking"""
        start_time = time.time()
        start_memory = memory_usage_gb()

        logger.info("🚀 Starting enhanced bootstrap stacking analysis...")
        logger.info(f"📊 Initial memory usage: {start_memory:.2f}GB")

        try:
            self._validate_inputs()

            if self.bootstrap_enabled:
                # Initialize progress tracker
                self.progress_tracker = ProgressTracker(
                    self.bootstrap_iterations, len(self.sky_maps.maps)
                )

                results_dict = self._run_bootstrap_stacking()
            else:
                logger.info("Running single stacking (no bootstrap)...")
                results_dict = self._run_single_stacking()

            # Compile results
            self.results = self._compile_results(results_dict, start_time, start_memory)

            # Final summary
            total_time = time.time() - start_time
            final_memory = memory_usage_gb()
            memory_peak = max(final_memory, start_memory)

            logger.info("✅ Bootstrap stacking completed successfully!")
            logger.info(f"⏱️  Total execution time: {total_time / 60:.1f} minutes")
            logger.info(f"💾 Peak memory usage: {memory_peak:.2f}GB")
            logger.info(f"📈 Memory change: {final_memory - start_memory:+.2f}GB")

            if self.bootstrap_enabled:
                total_computations = self.bootstrap_iterations * len(self.sky_maps.maps)
                avg_time_per_computation = total_time / total_computations
                logger.info(
                    f"⚡ Average time per computation: {avg_time_per_computation:.1f}s"
                )

            return self.results

        except Exception as e:
            logger.error(f"❌ Bootstrap stacking failed: {e}")
            raise AlgorithmError(f"Bootstrap stacking failed: {e}") from e

    def _run_bootstrap_stacking(self) -> dict[str, dict[str, Any]]:
        """Run bootstrap stacking with detailed progress tracking"""
        bootstrap_results = {map_name: [] for map_name in self.sky_maps.maps.keys()}

        logger.info("🔄 Starting bootstrap iterations...")
        logger.info("=" * 60)

        for bootstrap_iter in range(self.bootstrap_iterations):
            iter_start_time = time.time()

            logger.info(
                f"🎯 BOOTSTRAP ITERATION {bootstrap_iter + 1}/{self.bootstrap_iterations}"
            )
            logger.info("-" * 40)

            # Create bootstrap splits for this iteration
            logger.info("📝 Creating bootstrap splits...")
            split_start_time = time.time()
            bootstrap_splits = self._create_bootstrap_splits(
                seed=self.bootstrap_seed + bootstrap_iter
            )
            split_time = time.time() - split_start_time
            logger.info(f"   ✓ Splits created in {split_time:.1f}s")

            # Log split statistics
            total_A = sum(splits["A"].n_sources for splits in bootstrap_splits.values())
            total_B = sum(splits["B"].n_sources for splits in bootstrap_splits.values())
            logger.info(
                f"   📊 Split A: {total_A:,} sources, Split B: {total_B:,} sources"
            )

            # Process each map
            for map_idx, map_name in enumerate(self.sky_maps.maps.keys()):
                map_start_time = time.time()

                # Update progress tracker
                self.progress_tracker.update(
                    bootstrap_iter + 1,
                    map_name,
                    f"(map {map_idx + 1}/{len(self.sky_maps.maps)})",
                )

                logger.info(f"🗺️  Processing map: {map_name}")

                # Run stacking for this map
                map_results = self._stack_single_map_with_bootstrap_splits(
                    map_name, bootstrap_splits
                )

                # Store results
                bootstrap_results[map_name].append(map_results["total_flux_densities"])

                map_time = time.time() - map_start_time
                logger.info(f"   ✓ Map {map_name} completed in {map_time:.1f}s")

                # Memory check
                current_memory = memory_usage_gb()
                logger.info(f"   💾 Current memory: {current_memory:.2f}GB")

            iter_time = time.time() - iter_start_time
            logger.info(
                f"✅ Iteration {bootstrap_iter + 1} completed in {iter_time / 60:.1f} minutes"
            )
            logger.info("-" * 40)

        logger.info("🔄 All bootstrap iterations completed!")
        logger.info("📊 Computing bootstrap statistics...")

        # Calculate bootstrap statistics
        final_results = {}
        for map_name in self.sky_maps.maps.keys():
            logger.info(f"   📈 Processing statistics for {map_name}...")

            bootstrap_fluxes = np.array(bootstrap_results[map_name])

            mean_fluxes = np.mean(bootstrap_fluxes, axis=0)
            bootstrap_errors = np.std(bootstrap_fluxes, axis=0, ddof=1)

            # Log bootstrap statistics
            non_fg_mask = ~np.array(
                [label == "foreground" for label in range(len(mean_fluxes))]
            )
            if np.any(non_fg_mask):
                mean_flux_science = np.mean(np.abs(mean_fluxes[non_fg_mask]))
                mean_error_science = np.mean(bootstrap_errors[non_fg_mask])
                snr = (
                    mean_flux_science / mean_error_science
                    if mean_error_science > 0
                    else 0
                )
                logger.info(f"      Mean |flux|: {mean_flux_science:.2e} Jy")
                logger.info(f"      Mean error: {mean_error_science:.2e} Jy")
                logger.info(f"      Typical S/N: {snr:.1f}")

            # Get systematic errors from full sample
            full_results = self._stack_single_map_standard(map_name)

            final_results[map_name] = {
                "flux_densities": mean_fluxes,
                "flux_errors": bootstrap_errors,
                "flux_errors_systematic": full_results["flux_errors"],
                "bootstrap_flux_samples": bootstrap_fluxes,
                "population_labels": full_results["population_labels"],
                "fit_statistics": full_results["fit_statistics"],
                "n_valid_pixels": full_results["n_valid_pixels"],
            }

        logger.info("✅ Bootstrap statistics computed!")
        return final_results

    def _create_bootstrap_splits(
        self, seed: int
    ) -> dict[str, dict[str, BootstrapSplit]]:
        """Create bootstrap splits with progress logging"""
        np.random.seed(seed)
        bootstrap_splits = {}

        n_pops_with_sources = 0
        total_sources_split = 0

        for pop_id, population in self.population_manager.populations.items():
            if population.n_sources == 0:
                bootstrap_splits[pop_id] = {
                    "A": BootstrapSplit(pop_id, "A", np.array([]), 0),
                    "B": BootstrapSplit(pop_id, "B", np.array([]), 0),
                }
                continue

            n_pops_with_sources += 1
            total_sources_split += population.n_sources

            # Get original source indices
            original_indices = population.indices
            n_sources = len(original_indices)

            # Calculate split sizes
            n_A = int(n_sources * self.bootstrap_split_fraction)
            n_B = n_sources - n_A

            # Randomly permute and split
            shuffled_indices = np.random.permutation(original_indices)
            indices_A = shuffled_indices[:n_A]
            indices_B = shuffled_indices[n_A:]

            # Create bootstrap splits
            bootstrap_splits[pop_id] = {
                "A": BootstrapSplit(pop_id, "A", indices_A, n_A),
                "B": BootstrapSplit(pop_id, "B", indices_B, n_B),
            }

            # Verify no sources lost
            assert len(indices_A) + len(indices_B) == n_sources
            assert len(np.intersect1d(indices_A, indices_B)) == 0

        logger.debug(
            f"   📊 Split {n_pops_with_sources} populations with {total_sources_split:,} total sources"
        )
        return bootstrap_splits

    def _stack_single_map_with_bootstrap_splits(
        self, map_name: str, bootstrap_splits: dict[str, dict[str, BootstrapSplit]]
    ) -> dict[str, Any]:
        """Stack single map with detailed sub-step logging"""

        logger.debug(f"      🔧 Creating layer matrix for {map_name}...")
        layer_start = time.time()

        map_data = self.sky_maps[map_name]
        population_ids = list(bootstrap_splits.keys())
        n_populations = len(population_ids)

        # Create layer matrix
        layer_matrix, layer_labels = self._create_bootstrap_layer_matrix(
            map_name, map_data, bootstrap_splits
        )

        layer_time = time.time() - layer_start
        logger.debug(
            f"         ✓ Layer matrix created ({layer_matrix.shape}) in {layer_time:.1f}s"
        )

        # Add foreground if requested
        if self.add_foreground:
            foreground_layer = np.ones_like(layer_matrix[0])
            layer_matrix = np.vstack([layer_matrix, foreground_layer[np.newaxis, :]])
            layer_labels.append("foreground")
            logger.debug("         ✓ Added foreground layer")

        # Crop to circles if requested
        if self.crop_circles:
            logger.debug("      ✂️  Cropping to circles around sources...")
            crop_start = time.time()
            layer_matrix, observed_vector, valid_mask = self._crop_to_circles_bootstrap(
                layer_matrix, map_data.data, map_name, bootstrap_splits
            )
            crop_time = time.time() - crop_start
            logger.debug(
                f"         ✓ Cropped to {len(observed_vector):,} pixels in {crop_time:.1f}s"
            )
        else:
            observed_vector = map_data.data.ravel()
            valid_mask = ~np.isnan(observed_vector)
            layer_matrix = layer_matrix[:, valid_mask]
            observed_vector = observed_vector[valid_mask]
            logger.debug(f"         ✓ Using all {len(observed_vector):,} valid pixels")

        # Solve linear system
        logger.debug("      🧮 Solving linear system...")
        solve_start = time.time()
        combined_flux_densities, flux_errors, fit_stats = self._solve_linear_system(
            layer_matrix, observed_vector, map_data
        )
        solve_time = time.time() - solve_start

        # Log fit quality
        chi2_red = fit_stats.get("reduced_chi_squared", np.inf)
        logger.debug(
            f"         ✓ System solved in {solve_time:.1f}s, χ²_red = {chi2_red:.2f}"
        )

        # Extract A and B fluxes and calculate totals
        if self.add_foreground:
            flux_A = combined_flux_densities[:n_populations]
            flux_B = combined_flux_densities[n_populations : 2 * n_populations]
            foreground_flux = combined_flux_densities[2 * n_populations]
        else:
            flux_A = combined_flux_densities[:n_populations]
            flux_B = combined_flux_densities[n_populations : 2 * n_populations]
            foreground_flux = 0.0

        # Calculate total flux for each population
        total_flux_densities = flux_A + flux_B

        # Create population labels
        final_labels = population_ids.copy()
        if self.add_foreground:
            final_labels.append("foreground")
            total_flux_densities = np.append(total_flux_densities, foreground_flux)

        # Log flux statistics
        non_fg_fluxes = (
            total_flux_densities[:-1] if self.add_foreground else total_flux_densities
        )
        if len(non_fg_fluxes) > 0:
            positive_fluxes = non_fg_fluxes[non_fg_fluxes > 0]
            negative_fluxes = non_fg_fluxes[non_fg_fluxes < 0]
            logger.debug(
                f"         📊 Flux range: [{np.min(non_fg_fluxes):.2e}, {np.max(non_fg_fluxes):.2e}] Jy"
            )
            logger.debug(
                f"             Positive: {len(positive_fluxes)}, Negative: {len(negative_fluxes)}"
            )

        return {
            "total_flux_densities": total_flux_densities,
            "population_labels": final_labels,
            "fit_statistics": fit_stats,
            "n_valid_pixels": len(observed_vector),
        }

    def _create_bootstrap_layer_matrix(
        self,
        map_name: str,
        map_data: MapData,
        bootstrap_splits: dict[str, dict[str, BootstrapSplit]],
    ) -> tuple[np.ndarray, list[str]]:
        """
        Create layer matrix from bootstrap splits

        Structure: [A_pop1, A_pop2, ..., B_pop1, B_pop2, ...]
        """
        map_shape = map_data.shape
        n_pixels = map_shape[0] * map_shape[1]
        population_ids = list(bootstrap_splits.keys())
        n_populations = len(population_ids)

        # Initialize layer matrix: 2 * n_populations layers (A and B for each)
        layer_matrix = np.zeros((2 * n_populations, n_pixels))
        layer_labels = []

        # Create A layers
        for i, pop_id in enumerate(population_ids):
            split_A = bootstrap_splits[pop_id]["A"]

            if split_A.n_sources > 0:
                ra, dec = self._get_coordinates_from_indices(split_A.source_indices)
                layer = self._create_and_convolve_layer(map_name, ra, dec, map_shape)
            else:
                layer = np.zeros(n_pixels)

            layer_matrix[i, :] = layer
            layer_labels.append(f"{pop_id}_A")

        # Create B layers
        for i, pop_id in enumerate(population_ids):
            split_B = bootstrap_splits[pop_id]["B"]

            if split_B.n_sources > 0:
                ra, dec = self._get_coordinates_from_indices(split_B.source_indices)
                layer = self._create_and_convolve_layer(map_name, ra, dec, map_shape)
            else:
                layer = np.zeros(n_pixels)

            layer_matrix[n_populations + i, :] = layer
            layer_labels.append(f"{pop_id}_B")

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

    def _crop_to_circles_bootstrap(
        self,
        layer_matrix: np.ndarray,
        observed_map: np.ndarray,
        map_name: str,
        bootstrap_splits: dict[str, dict[str, BootstrapSplit]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Crop to circles around all bootstrap split sources"""
        map_data = self.sky_maps[map_name]

        mask = np.zeros(observed_map.shape, dtype=bool)
        circle_radius_pix = max(3.0, map_data.beam_fwhm_pixels)

        # Add circles for all sources in all splits
        for pop_id, splits in bootstrap_splits.items():
            for split_type, split_data in splits.items():
                if split_data.n_sources == 0:
                    continue

                try:
                    ra, dec = self._get_coordinates_from_indices(
                        split_data.source_indices
                    )
                    x_pix, y_pix = self.sky_maps.world_to_pixel(map_name, ra, dec)

                    for x, y in zip(x_pix, y_pix, strict=True):
                        ix, iy = int(np.round(x)), int(np.round(y))
                        if (
                            0 <= ix < observed_map.shape[1]
                            and 0 <= iy < observed_map.shape[0]
                        ):
                            self._add_circle_to_mask(mask, x, y, circle_radius_pix)

                except Exception as e:
                    logger.warning(f"Error processing {pop_id}_{split_type}: {e}")
                    continue

        # Apply mask
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
        map_data = self.sky_maps[map_name]

        # Create standard layer matrix
        layer_matrix, population_labels = self._create_standard_layer_matrix(
            map_name, map_data
        )

        # Add foreground if requested
        if self.add_foreground:
            foreground_layer = np.ones_like(layer_matrix[0])
            layer_matrix = np.vstack([layer_matrix, foreground_layer[np.newaxis, :]])
            population_labels.append("foreground")

        # Crop if requested
        if self.crop_circles:
            layer_matrix, observed_vector, valid_mask = self._crop_to_circles_standard(
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

    def _create_standard_layer_matrix(
        self, map_name: str, map_data: MapData
    ) -> tuple[np.ndarray, list[str]]:
        """Create standard layer matrix (no bootstrap splits)"""
        populations = list(self.population_manager.iter_populations())
        n_populations = len(populations)
        map_shape = map_data.shape
        n_pixels = map_shape[0] * map_shape[1]

        layer_matrix = np.zeros((n_populations, n_pixels))
        population_labels = []

        for i, pop_bin in enumerate(populations):
            if pop_bin.n_sources == 0:
                population_labels.append(pop_bin.id_label)
                continue

            # Use the SAME coordinate extraction method
            ra, dec = self._get_coordinates_from_indices(pop_bin.indices)
            layer = self._create_and_convolve_layer(map_name, ra, dec, map_shape)

            layer_matrix[i, :] = layer
            population_labels.append(pop_bin.id_label)

        return layer_matrix, population_labels

    def _crop_to_circles_standard(
        self, layer_matrix: np.ndarray, observed_map: np.ndarray, map_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Standard cropping (no bootstrap)"""
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

        logger.info("✅ StackingResults object created successfully")
        logger.info(f"📊 {len(map_names)} maps, {len(population_labels)} populations")

        return results

    def print_results_summary(self) -> None:
        """Print a summary of unbiased bootstrap stacking results"""
        if self.results is None:
            print("No results available - run stacking first")
            return

        results = self.results

        print("=== UNBIASED Bootstrap Stacking Results Summary ===")
        print(f"Execution time: {results.execution_time:.1f}s")
        print(f"Memory used: {results.memory_used_gb:.2f}GB")
        print(f"Populations: {len(results.population_labels)}")
        print(f"Maps: {len(results.map_names)}")
        print(f"Bootstrap enabled: {results.bootstrap_enabled}")
        if results.bootstrap_enabled:
            print(f"Bootstrap iterations: {results.bootstrap_iterations}")
            print(
                f"Bootstrap split: {results.bootstrap_split_fraction:.1%}:{1 - results.bootstrap_split_fraction:.1%}"
            )
            print("✓ UNBIASED: All sources included in each iteration")
        print()

        print("Flux Densities with Unbiased Bootstrap Errors (Jy):")
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
            print("* Systematic errors only (no bootstrap)")

        print()
        print("Unbiased Bootstrap vs Systematic Error Comparison:")
        if results.bootstrap_enabled:
            for map_name in results.map_names:
                bootstrap_err = np.mean(results.flux_errors[map_name])
                systematic_err = np.mean(results.flux_errors_systematic[map_name])
                ratio = bootstrap_err / systematic_err if systematic_err > 0 else np.inf
                print(f"{map_name}: Bootstrap/Systematic error ratio = {ratio:.2f}")

            print("\n✓ All sources included in each bootstrap iteration")
            print("✓ No positive bias from missing sources")


def run_stacking(
    config: SimstackConfig, population_manager: PopulationManager, sky_maps: SkyMaps
):
    """Run stacking with proper bootstrap design"""
    algorithm = SimstackAlgorithm(config, population_manager, sky_maps)
    return algorithm.run_stacking()
