"""
Core stacking algorithm for Simstack4

This module implements the simultaneous stacking algorithm that fits
multiple population layers to observed maps, replacing the nested loop
structure from simstack3 with a more efficient matrix-based approach.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import time
from scipy import linalg
from sklearn.linear_model import LinearRegression
import warnings

from .populations import PopulationManager, PopulationBin
from .sky_maps import SkyMaps, MapData
from .config import SimstackConfig, BinningConfig
from .exceptions.simstack_exceptions import AlgorithmError, ValidationError
from .utils import setup_logging, memory_usage_gb

logger = setup_logging()


@dataclass
class StackingResults:
    """Container for stacking results"""
    flux_densities: Dict[str, np.ndarray]  # [map_name][population_index]
    flux_errors: Dict[str, np.ndarray]
    population_labels: List[str]
    map_names: List[str]
    n_sources: Dict[str, int]  # sources per population
    chi_squared: Dict[str, float]  # chi2 per map
    degrees_of_freedom: Dict[str, int]
    reduced_chi_squared: Dict[str, float]
    execution_time: float
    memory_used_gb: float

    def __post_init__(self):
        """Calculate derived quantities"""
        for map_name in self.map_names:
            if map_name in self.chi_squared and map_name in self.degrees_of_freedom:
                dof = self.degrees_of_freedom[map_name]
                if dof > 0:
                    self.reduced_chi_squared[map_name] = self.chi_squared[map_name] / dof


class SimstackAlgorithm:
    """
    Core simultaneous stacking algorithm

    This class implements the matrix-based simultaneous fitting approach
    that replaces the nested population loops from simstack3.
    """

    def __init__(self, config: SimstackConfig, population_manager: PopulationManager,
                 sky_maps: SkyMaps):
        """
        Initialize stacking algorithm

        Args:
            config: Simstack configuration
            population_manager: Populated catalog manager
            sky_maps: Loaded sky maps
        """
        self.config = config
        self.population_manager = population_manager
        self.sky_maps = sky_maps
        self.results: Optional[StackingResults] = None

        # Algorithm settings
        self.crop_circles = config.binning.crop_circles
        self.add_foreground = config.binning.add_foreground
        self.stack_all_z = config.binning.stack_all_z_at_once

        logger.info(f"SimstackAlgorithm initialized:")
        logger.info(f"  - {len(population_manager)} populations")
        logger.info(f"  - {len(sky_maps)} maps")
        logger.info(f"  - Crop circles: {self.crop_circles}")
        logger.info(f"  - Add foreground: {self.add_foreground}")

    def run_stacking(self) -> StackingResults:
        """
        Execute the simultaneous stacking algorithm

        Returns:
            StackingResults object with flux densities and uncertainties
        """
        start_time = time.time()
        start_memory = memory_usage_gb()

        logger.info("Starting simultaneous stacking...")

        try:
            # Validate inputs
            self._validate_inputs()

            # Main stacking loop over maps
            results_dict = {}

            for map_name in self.sky_maps.maps.keys():
                logger.info(f"Processing map: {map_name}")
                map_results = self._stack_single_map(map_name)
                results_dict[map_name] = map_results

            # Compile results
            self.results = self._compile_results(results_dict, start_time, start_memory)

            logger.info(f"Stacking completed in {self.results.execution_time:.1f}s")
            logger.info(f"Peak memory usage: {self.results.memory_used_gb:.2f}GB")

            return self.results

        except Exception as e:
            logger.error(f"Stacking failed: {e}")
            raise AlgorithmError(f"Stacking algorithm failed: {e}")

    def _validate_inputs(self) -> None:
        """Validate that inputs are compatible for stacking"""
        if len(self.population_manager) == 0:
            raise ValidationError("No populations found in catalog")

        if len(self.sky_maps) == 0:
            raise ValidationError("No maps loaded")

        # Check memory requirements
        n_populations = len(self.population_manager)
        n_maps = len(self.sky_maps)

        # Get typical map size
        first_map = next(iter(self.sky_maps.maps.values()))
        map_pixels = first_map.shape[0] * first_map.shape[1]

        # Estimate memory for layer matrix (n_populations x map_pixels)
        layer_memory_gb = (n_populations * map_pixels * 8) / 1e9  # 8 bytes per float64

        if layer_memory_gb > 8.0:  # More than 8GB
            logger.warning(f"Large memory requirement estimated: {layer_memory_gb:.1f}GB")
            if not self.stack_all_z:
                logger.info("Consider setting stack_all_z_at_once=False for memory efficiency")

    def _stack_single_map(self, map_name: str) -> Dict[str, Any]:
        """
        Stack a single map using simultaneous fitting

        Args:
            map_name: Name of the map to stack

        Returns:
            Dictionary with stacking results for this map
        """
        map_data = self.sky_maps[map_name]

        # Create layer matrix
        logger.debug(f"Creating layers for {map_name}...")
        layer_matrix, population_labels = self._create_layer_matrix(map_name, map_data)

        # Add foreground layer if requested
        if self.add_foreground:
            foreground_layer = np.ones_like(layer_matrix[0])
            layer_matrix = np.vstack([layer_matrix, foreground_layer[np.newaxis, :]])
            population_labels.append("foreground")

        # Crop to circles around sources if requested
        if self.crop_circles:
            layer_matrix, observed_vector, valid_mask = self._crop_to_circles(
                layer_matrix, map_data.data, map_name
            )
        else:
            # Use all pixels
            observed_vector = map_data.data.ravel()
            valid_mask = ~np.isnan(observed_vector)
            layer_matrix = layer_matrix[:, valid_mask]
            observed_vector = observed_vector[valid_mask]

        # Solve linear system
        logger.debug(f"Solving linear system: {layer_matrix.shape}")
        flux_densities, flux_errors, fit_stats = self._solve_linear_system(
            layer_matrix, observed_vector, map_data
        )

        return {
            "flux_densities": flux_densities,
            "flux_errors": flux_errors,
            "population_labels": population_labels,
            "fit_statistics": fit_stats,
            "n_valid_pixels": len(observed_vector)
        }

    def _create_layer_matrix(self, map_name: str, map_data: MapData) -> Tuple[np.ndarray, List[str]]:
        """
        Create the matrix of convolved population layers

        Args:
            map_name: Name of the map
            map_data: Map data object

        Returns:
            Layer matrix (n_populations x n_pixels) and population labels
        """
        populations = list(self.population_manager.iter_populations())
        n_populations = len(populations)
        map_shape = map_data.shape
        n_pixels = map_shape[0] * map_shape[1]

        # Initialize layer matrix
        layer_matrix = np.zeros((n_populations, n_pixels))
        population_labels = []

        for i, pop_bin in enumerate(populations):
            # Get population data
            pop_data = self.population_manager.populations[pop_bin.id_label]

            if pop_data.n_sources == 0:
                logger.warning(f"Population {pop_bin.id_label} has no sources")
                continue

            # Get source positions from catalog
            # This is a simplified approach - in practice you'd get RA/Dec from catalog
            # using the population indices
            ra, dec, weights = self._get_population_coordinates(pop_bin)

            # Create source layer
            source_layer = self._create_source_layer(map_name, ra, dec, weights, map_shape)

            # Convolve with PSF
            convolved_layer = self.sky_maps.convolve_with_psf(source_layer, map_name)

            # Store flattened layer
            layer_matrix[i, :] = convolved_layer.ravel()
            population_labels.append(pop_bin.id_label)

        return layer_matrix, population_labels

    def _get_population_coordinates(self, pop_bin: PopulationBin) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get RA, Dec, and weights for a population

        This is a placeholder - in practice this would interface with the catalog
        to get actual coordinates using pop_bin.indices

        Args:
            pop_bin: Population bin object

        Returns:
            ra, dec, weights arrays
        """
        # TODO: This needs to interface with the actual catalog loaded in SkyCatalogs
        # For now, create dummy data
        n_sources = pop_bin.n_sources

        # Create random positions around median redshift position
        # This is just for demonstration - real implementation would use catalog data
        ra = np.random.normal(0, 0.1, n_sources)  # Random RA around 0
        dec = np.random.normal(0, 0.1, n_sources)  # Random Dec around 0

        # Use stellar mass as weights (log mass converted to linear)
        weights = 10**(pop_bin.median_stellar_mass - 10)  # Normalize around 10^10 solar masses
        weights = np.full(n_sources, weights)

        logger.warning("Using dummy coordinates - needs catalog integration")
        return ra, dec, weights

    def _create_source_layer(self, map_name: str, ra: np.ndarray, dec: np.ndarray,
                           weights: np.ndarray, map_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create a 2D source layer from coordinates and weights

        Args:
            map_name: Name of the map (for coordinate system)
            ra: Source right ascensions in degrees
            dec: Source declinations in degrees
            weights: Source weights (e.g., stellar mass)
            map_shape: Shape of the output layer

        Returns:
            2D source layer array
        """
        # Convert world coordinates to pixel coordinates
        x_pix, y_pix = self.sky_maps.world_to_pixel(map_name, ra, dec)

        # Initialize layer
        layer = np.zeros(map_shape)

        # Add sources to layer
        for x, y, weight in zip(x_pix, y_pix, weights):
            # Check bounds and add source
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < map_shape[1] and 0 <= iy < map_shape[0]:
                layer[iy, ix] += weight

        return layer

    def _crop_to_circles(self, layer_matrix: np.ndarray, observed_map: np.ndarray,
                        map_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crop fitting to circular regions around sources

        This reduces the number of pixels used in fitting, focusing on regions
        where sources are expected to contribute signal.

        Args:
            layer_matrix: Population layer matrix
            observed_map: Observed map data
            map_name: Name of the map

        Returns:
            Cropped layer matrix, observed vector, and valid pixel mask
        """
        map_data = self.sky_maps[map_name]

        # Create mask for pixels within circles around sources
        mask = np.zeros(observed_map.shape, dtype=bool)

        # Get beam radius in pixels (2 x FWHM for conservative circle)
        circle_radius_pix = 2 * map_data.beam_fwhm_pixels

        # Add circles around all sources from all populations
        for pop_bin in self.population_manager.iter_populations():
            if pop_bin.n_sources == 0:
                continue

            ra, dec, _ = self._get_population_coordinates(pop_bin)
            x_pix, y_pix = self.sky_maps.world_to_pixel(map_name, ra, dec)

            # Add circles
            for x, y in zip(x_pix, y_pix):
                self._add_circle_to_mask(mask, x, y, circle_radius_pix)

        # Apply mask
        valid_pixels = mask.ravel() & ~np.isnan(observed_map.ravel())

        cropped_layers = layer_matrix[:, valid_pixels]
        cropped_observed = observed_map.ravel()[valid_pixels]

        logger.debug(f"Cropped to {np.sum(valid_pixels)} pixels from {len(valid_pixels)}")

        return cropped_layers, cropped_observed, valid_pixels

    def _add_circle_to_mask(self, mask: np.ndarray, center_x: float, center_y: float,
                           radius: float) -> None:
        """Add a circular region to the mask"""
        y_indices, x_indices = np.ogrid[:mask.shape[0], :mask.shape[1]]
        distances = np.sqrt((x_indices - center_x)**2 + (y_indices - center_y)**2)
        mask[distances <= radius] = True

    def _solve_linear_system(self, layer_matrix: np.ndarray, observed_vector: np.ndarray,
                           map_data: MapData) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Solve the linear system to get flux densities

        Solves: observed = layer_matrix @ flux_densities + noise

        Args:
            layer_matrix: Population layers (n_populations x n_pixels)
            observed_vector: Observed pixel values (n_pixels,)
            map_data: Map data for noise estimation

        Returns:
            flux_densities, flux_errors, fit_statistics
        """
        n_populations, n_pixels = layer_matrix.shape

        if n_pixels == 0:
            raise AlgorithmError("No valid pixels for fitting")

        if n_populations == 0:
            raise AlgorithmError("No populations to fit")

        # Check for degenerate cases
        if n_pixels < n_populations:
            logger.warning(f"Underdetermined system: {n_pixels} pixels, {n_populations} populations")

        try:
            # Use weighted least squares if we have noise information
            if map_data.noise is not None:
                weights = 1.0 / (map_data.noise.ravel()[~np.isnan(observed_vector)]**2)
                weights = weights[:len(observed_vector)]  # Match lengths

                # Weight the system
                weighted_layers = layer_matrix * np.sqrt(weights)
                weighted_observed = observed_vector * np.sqrt(weights)

                # Solve weighted system
                flux_densities, residuals, rank, singular_values = linalg.lstsq(
                    weighted_layers.T, weighted_observed
                )
            else:
                # Ordinary least squares
                flux_densities, residuals, rank, singular_values = linalg.lstsq(
                    layer_matrix.T, observed_vector
                )

            # Calculate uncertainties
            if len(residuals) > 0 and n_pixels > n_populations:
                # Standard approach: use residuals
                mse = residuals[0] / (n_pixels - n_populations)

                # Covariance matrix
                try:
                    if map_data.noise is not None:
                        # Use noise weights
                        weighted_layers = layer_matrix * np.sqrt(weights)
                        covariance = linalg.inv(weighted_layers @ weighted_layers.T)
                    else:
                        covariance = linalg.inv(layer_matrix @ layer_matrix.T) * mse

                    flux_errors = np.sqrt(np.diag(covariance))
                except linalg.LinAlgError:
                    logger.warning("Could not compute covariance matrix, using scaled errors")
                    flux_errors = np.abs(flux_densities) * 0.1  # 10% errors as fallback
            else:
                # Fallback error estimate
                flux_errors = np.abs(flux_densities) * 0.1

            # Calculate fit statistics
            model_prediction = layer_matrix.T @ flux_densities
            chi_squared = np.sum((observed_vector - model_prediction)**2)

            if map_data.noise is not None:
                # Proper chi-squared with noise weights
                noise_vector = map_data.noise.ravel()[~np.isnan(observed_vector)][:len(observed_vector)]
                chi_squared = np.sum(((observed_vector - model_prediction) / noise_vector)**2)

            dof = max(1, n_pixels - n_populations)
            reduced_chi_squared = chi_squared / dof

            fit_stats = {
                "chi_squared": chi_squared,
                "degrees_of_freedom": dof,
                "reduced_chi_squared": reduced_chi_squared,
                "rank": rank,
                "condition_number": np.max(singular_values) / np.min(singular_values) if len(singular_values) > 0 else np.inf
            }

            return flux_densities, flux_errors, fit_stats

        except Exception as e:
            logger.error(f"Linear system solution failed: {e}")
            # Return zeros as fallback
            return (np.zeros(n_populations), np.ones(n_populations),
                   {"chi_squared": np.inf, "degrees_of_freedom": 1, "reduced_chi_squared": np.inf})

    def _compile_results(self, results_dict: Dict[str, Dict], start_time: float,
                        start_memory: float) -> StackingResults:
        """Compile results from all maps into final StackingResults object"""
        map_names = list(results_dict.keys())

        # Get population labels (should be same for all maps)
        population_labels = results_dict[map_names[0]]["population_labels"]

        # Extract results per map
        flux_densities = {}
        flux_errors = {}
        chi_squared = {}
        degrees_of_freedom = {}

        for map_name, map_results in results_dict.items():
            flux_densities[map_name] = map_results["flux_densities"]
            flux_errors[map_name] = map_results["flux_errors"]
            chi_squared[map_name] = map_results["fit_statistics"]["chi_squared"]
            degrees_of_freedom[map_name] = map_results["fit_statistics"]["degrees_of_freedom"]

        # Count sources per population
        n_sources = {}
        for pop_bin in self.population_manager.iter_populations():
            n_sources[pop_bin.id_label] = pop_bin.n_sources

        # Calculate timing and memory
        execution_time = time.time() - start_time
        current_memory = memory_usage_gb()
        memory_used = max(0, current_memory - start_memory)

        return StackingResults(
            flux_densities=flux_densities,
            flux_errors=flux_errors,
            population_labels=population_labels,
            map_names=map_names,
            n_sources=n_sources,
            chi_squared=chi_squared,
            degrees_of_freedom=degrees_of_freedom,
            reduced_chi_squared={},  # Will be calculated in __post_init__
            execution_time=execution_time,
            memory_used_gb=memory_used
        )

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
        print()

        print("Flux Densities (Jy):")
        print(f"{'Population':<30} " + " ".join(f"{name:<12}" for name in results.map_names))
        print("-" * (30 + 13 * len(results.map_names)))

        for i, pop_label in enumerate(results.population_labels):
            line = f"{pop_label:<30}"
            for map_name in results.map_names:
                flux = results.flux_densities[map_name][i]
                error = results.flux_errors[map_name][i]
                line += f" {flux:>8.2e}±{error:.1e}"
            print(line)

        print()
        print("Fit Quality:")
        for map_name in results.map_names:
            chi2 = results.chi_squared[map_name]
            dof = results.degrees_of_freedom[map_name]
            reduced_chi2 = results.reduced_chi_squared.get(map_name, chi2/dof)
            print(f"{map_name}: χ² = {chi2:.1f}, DoF = {dof}, χ²_red = {reduced_chi2:.2f}")


def run_stacking(config: SimstackConfig, population_manager: PopulationManager,
                sky_maps: SkyMaps) -> StackingResults:
    """
    Convenience function to run simultaneous stacking

    Args:
        config: Simstack configuration
        population_manager: Populated catalog manager
        sky_maps: Loaded sky maps

    Returns:
        StackingResults object
    """
    algorithm = SimstackAlgorithm(config, population_manager, sky_maps)
    return algorithm.run_stacking()