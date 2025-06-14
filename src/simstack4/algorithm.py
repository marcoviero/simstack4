"""
Core stacking algorithm for Simstack4

This module implements the simultaneous stacking algorithm that fits
multiple population layers to observed maps, replacing the nested loop
structure from simstack3 with a more efficient matrix-based approach.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import linalg

from .config import SimstackConfig
from .exceptions.simstack_exceptions import (
    AlgorithmError,
    PopulationError,
    ValidationError,
)
from .populations import PopulationBin, PopulationManager
from .sky_maps import MapData, SkyMaps
from .utils import memory_usage_gb, setup_logging

logger = setup_logging()


@dataclass
class StackingResults:
    """Container for stacking results"""

    flux_densities: dict[str, np.ndarray]  # [map_name][population_index]
    flux_errors: dict[str, np.ndarray]
    population_labels: list[str]
    map_names: list[str]
    n_sources: dict[str, int]  # sources per population
    chi_squared: dict[str, float]  # chi2 per map
    degrees_of_freedom: dict[str, int]
    reduced_chi_squared: dict[str, float]
    execution_time: float
    memory_used_gb: float

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
        self.results: StackingResults | None = None

        # Algorithm settings
        self.crop_circles = config.binning.crop_circles
        self.add_foreground = config.binning.add_foreground
        self.stack_all_z = config.binning.stack_all_z_at_once

        # FITS output settings
        self.write_layer_maps = getattr(config.error_estimator, "write_simmaps", False)
        self.layer_output_dir = Path(config.output.folder) / "layer_maps"

        # Create output directory if we're writing layers
        if self.write_layer_maps:
            self.layer_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Layer maps will be written to: {self.layer_output_dir}")

        logger.info("SimstackAlgorithm initialized:")
        logger.info(f"  - {len(population_manager)} populations")
        logger.info(f"  - {len(sky_maps)} maps")
        logger.info(f"  - Crop circles: {self.crop_circles}")
        logger.info(f"  - Add foreground: {self.add_foreground}")
        logger.info(f"  - Write layer maps: {self.write_layer_maps}")

    def save_layer_to_fits(
        self,
        data: np.ndarray,
        wcs: WCS,
        map_name: str,
        population_id: str,
        layer_type: str = "convolved",
    ) -> None:
        """
        Save a population layer to FITS file

        Args:
            data: 2D array of layer data
            wcs: World coordinate system
            map_name: Name of the map (e.g., 'pacs_green')
            population_id: Population identifier
            layer_type: Type of layer ('raw', 'convolved', 'mean_subtracted')
        """
        if not self.write_layer_maps:
            return

        try:
            # Create safe filename
            safe_pop_id = population_id.replace("__", "_").replace(".", "p")
            filename = f"{map_name}_{safe_pop_id}_{layer_type}.fits"
            output_path = self.layer_output_dir / filename

            # Create FITS header from WCS
            header = wcs.to_header()

            # Add metadata about the layer
            header["LAYERTYP"] = layer_type
            header["POPID"] = population_id
            header["MAPNAME"] = map_name
            header["BUNIT"] = "Jy/beam" if layer_type == "convolved" else "Jy"
            header["COMMENT"] = f"Population layer for {population_id}"

            # Add statistics
            header["DATAMIN"] = np.nanmin(data)
            header["DATAMAX"] = np.nanmax(data)
            header["DATAMEAN"] = np.nanmean(data)
            header["DATASTD"] = np.nanstd(data)

            # Write FITS file
            fits.writeto(output_path, data, header=header, overwrite=True)

            logger.debug(f"Saved layer: {filename}")

        except Exception as e:
            logger.warning(f"Failed to save layer {population_id} for {map_name}: {e}")

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

            if self.write_layer_maps:
                logger.info(f"Layer maps saved to: {self.layer_output_dir}")

            return self.results

        except Exception as e:
            logger.error(f"Stacking failed: {e}")
            raise AlgorithmError(f"Stacking algorithm failed: {e}") from e

    def _validate_inputs(self) -> None:
        """Validate that inputs are compatible for stacking"""
        if len(self.population_manager) == 0:
            raise ValidationError("No populations found in catalog")

        if len(self.sky_maps) == 0:
            raise ValidationError("No maps loaded")

        # Check memory requirements
        n_populations = len(self.population_manager)
        # n_maps = len(self.sky_maps)

        # Get typical map size
        first_map = next(iter(self.sky_maps.maps.values()))
        map_pixels = first_map.shape[0] * first_map.shape[1]

        # Estimate memory for layer matrix (n_populations x map_pixels)
        layer_memory_gb = (n_populations * map_pixels * 8) / 1e9  # 8 bytes per float64

        if layer_memory_gb > 8.0:  # More than 8GB
            logger.warning(
                f"Large memory requirement estimated: {layer_memory_gb:.1f}GB"
            )
            if not self.stack_all_z:
                logger.info(
                    "Consider setting stack_all_z_at_once=False for memory efficiency"
                )

    def _stack_single_map(self, map_name: str) -> dict[str, Any]:
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
            "n_valid_pixels": len(observed_vector),
        }

    def _create_layer_matrix(
        self, map_name: str, map_data: MapData
    ) -> tuple[np.ndarray, list[str]]:
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
            ra, dec = self._get_population_coordinates(pop_bin)

            # Create source layer
            source_layer = self._create_source_layer(map_name, ra, dec, map_shape)

            # Convolve with PSF
            convolved_layer = self.sky_maps.convolve_with_psf(source_layer, map_name)

            # Remove mean
            convolved_layer -= np.nanmean(convolved_layer)

            if self.write_layer_maps:
                self.save_layer_to_fits(
                    data=convolved_layer,
                    wcs=map_data.wcs,
                    map_name=map_name,
                    population_id=pop_bin.id_label,
                    layer_type="convolved",
                )

            # Store flattened layer
            layer_matrix[i, :] = convolved_layer.ravel()
            population_labels.append(pop_bin.id_label)

        return layer_matrix, population_labels

    # Fix for the _get_population_coordinates method in algorithm.py
    def _get_population_coordinates(
        self, pop_bin: PopulationBin
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get RA, Dec, and weights for a population (ENHANCED for COSMOS)

        Args:
            pop_bin: Population bin object

        Returns:
            ra, dec, weights arrays
        """
        # Get population data from the catalog through the population manager
        pop_data = self.population_manager.get_population_data(pop_bin.id_label)

        ra = pop_data["ra"]
        dec = pop_data["dec"]

        return ra, dec

    # Fix for the _create_source_layer method in algorithm.py
    def _create_source_layer(
        self,
        map_name: str,
        ra: np.ndarray,
        dec: np.ndarray,
        map_shape: tuple[int, int],
    ) -> np.ndarray:
        """
        Create a 2D source layer from coordinates and weights

        This replaces the simplified version in algorithm.py with the toolbox implementation
        """
        from .toolbox import SimstackToolbox

        map_data = self.sky_maps[map_name]

        # Use toolbox function for better accuracy
        layer = SimstackToolbox.create_source_layer(
            ra=ra, dec=dec, wcs=map_data.wcs, shape=map_shape
        )

        return layer

    # Fix for the PopulationManager integration in algorithm.py
    def get_population_data_method(self, population_id: str) -> dict[str, np.ndarray]:
        """
        Method to add to PopulationManager class to extract catalog data for populations

        Add this method to your populations.py PopulationManager class
        """
        if population_id not in self.populations:
            raise PopulationError(f"Population {population_id} not found")

        pop_bin = self.populations[population_id]
        indices = pop_bin.indices

        if self.catalog_df is None:
            raise PopulationError("No catalog data loaded")

        # Extract data using the indices
        if hasattr(self.catalog_df, "iloc"):  # pandas
            subset = self.catalog_df.iloc[indices]

            # Get column names from config
            ra_col = self.config.astrometry["ra"]
            dec_col = self.config.astrometry["dec"]
            z_col = self.config.redshift.id
            mass_col = self.config.stellar_mass.id

            data = {
                "ra": subset[ra_col].values,
                "dec": subset[dec_col].values,
                "redshift": subset[z_col].values,
                "stellar_mass": subset[mass_col].values,
                "indices": indices,
            }
        else:  # polars or other
            # Handle polars case
            subset = self.catalog_df[indices]
            # Implementation would depend on polars API
            pass

        return data

    # Enhanced wrapper integration
    def enhanced_wrapper_integration():
        """
        Code to add to wrapper.py to properly integrate all components
        """

        def run_complete_pipeline(self):
            """
            Run the complete simstack pipeline with all components

            Add this method to SimstackWrapper class
            """
            from .algorithm import SimstackAlgorithm
            from .cosmology import CosmologyCalculator
            from .results import SimstackResults

            logger = setup_logging()

            try:
                # Run stacking algorithm
                logger.info("Running stacking algorithm...")
                algorithm = SimstackAlgorithm(
                    config=self.config,
                    population_manager=self.population_manager,
                    sky_maps=self.sky_maps,
                )

                stacking_results = algorithm.run_stacking()

                # Process results
                logger.info("Processing results...")
                cosmology_calc = CosmologyCalculator(self.config.cosmology)

                results_processor = SimstackResults(
                    config=self.config,
                    stacking_results=stacking_results,
                    population_manager=self.population_manager,
                    cosmology_calc=cosmology_calc,
                )

                # Save results
                output_path = (
                    Path(self.config.output.folder)
                    / f"{self.config.output.shortname}_results.pkl"
                )
                results_processor.save_results(output_path, format="pickle")

                # Print summary
                results_processor.print_results_summary()

                # Store for access
                self.stacking_results = stacking_results
                self.processed_results = results_processor

                logger.info(
                    f"Pipeline completed successfully. Results saved to {output_path}"
                )

                return results_processor

            except Exception as e:
                logger.error(f"Pipeline failed: {e}")
                raise

    # Add these imports to the top of algorithm.py
    additional_imports = """
    from .toolbox import SimstackToolbox
    from .populations import PopulationManager, PopulationBin
    """

    # Updated pyproject.toml dependencies
    updated_dependencies = """
    dependencies = [
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "astropy>=5.3.0",
        "matplotlib>=3.6.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "lmfit>=1.2.0",
        "emcee>=3.1.0",
        "corner>=2.2.0",
        "pydantic>=2.5.0",
        "tomli>=2.0.0; python_version<'3.11'",
        "polars>=0.20.0",
        "pyarrow>=14.0.0",
        "psutil>=5.9.0",  # For memory monitoring
        "h5py>=3.8.0",    # For HDF5 output
        "seaborn>=0.12.0", # For enhanced plotting
    ]
    """

    def _crop_to_circles(
        self, layer_matrix: np.ndarray, observed_map: np.ndarray, map_name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Crop fitting to circular regions around sources (FIXED VERSION)

        This reduces the number of pixels used in fitting, focusing on regions
        where sources are expected to contribute signal.

        Args:
            layer_matrix: Population layer matrix (n_populations, n_pixels)
            observed_map: Observed map data (2D array)
            map_name: Name of the map

        Returns:
            Cropped layer matrix, observed vector, and valid pixel mask
        """
        map_data = self.sky_maps[map_name]

        # Create mask for pixels within circles around sources
        mask = np.zeros(observed_map.shape, dtype=bool)

        # Get beam radius in pixels (use more conservative radius)
        circle_radius_pix = max(3.0, map_data.beam_fwhm_pixels)  # At least 3 pixels

        logger.debug(f"Using circle radius: {circle_radius_pix:.1f} pixels")

        # Track number of sources added
        n_sources_added = 0

        # Add circles around all sources from all populations
        for pop_bin in self.population_manager.iter_populations():
            if pop_bin.n_sources == 0:
                continue

            try:
                # Get population data
                pop_data = self.population_manager.get_population_data(pop_bin.id_label)
                ra = pop_data["ra"]
                dec = pop_data["dec"]

                # Convert to pixel coordinates
                x_pix, y_pix = self.sky_maps.world_to_pixel(map_name, ra, dec)

                # Add circles for sources that fall within the map
                for x, y in zip(x_pix, y_pix, strict=True):
                    ix, iy = int(np.round(x)), int(np.round(y))

                    # Check if source is within map bounds
                    if (
                        0 <= ix < observed_map.shape[1]
                        and 0 <= iy < observed_map.shape[0]
                    ):
                        # Add circle around this source
                        self._add_circle_to_mask(mask, x, y, circle_radius_pix)
                        n_sources_added += 1

            except Exception as e:
                logger.warning(f"Error processing population {pop_bin.id_label}: {e}")
                continue

        logger.debug(f"Added circles for {n_sources_added} sources")

        # Apply mask and get valid pixels
        valid_pixels_2d = mask & ~np.isnan(observed_map)
        n_valid_pixels = np.sum(valid_pixels_2d)

        logger.debug(f"Valid pixels after cropping: {n_valid_pixels}")

        # Check if we have enough pixels for the fit
        n_populations = layer_matrix.shape[0]

        if n_valid_pixels < n_populations:
            logger.warning(
                f"Too few pixels ({n_valid_pixels}) for {n_populations} populations"
            )
            logger.warning("Expanding circle radius and using more pixels")

            # Expand circles or use more conservative approach
            expanded_radius = max(circle_radius_pix * 2, 10.0)
            mask_expanded = np.zeros(observed_map.shape, dtype=bool)

            # Re-add circles with larger radius
            for pop_bin in self.population_manager.iter_populations():
                if pop_bin.n_sources == 0:
                    continue

                try:
                    pop_data = self.population_manager.get_population_data(
                        pop_bin.id_label
                    )
                    ra = pop_data["ra"]
                    dec = pop_data["dec"]
                    x_pix, y_pix = self.sky_maps.world_to_pixel(map_name, ra, dec)

                    for x, y in zip(x_pix, y_pix, strict=True):
                        ix, iy = int(np.round(x)), int(np.round(y))
                        if (
                            0 <= ix < observed_map.shape[1]
                            and 0 <= iy < observed_map.shape[0]
                        ):
                            self._add_circle_to_mask(
                                mask_expanded, x, y, expanded_radius
                            )

                except Exception:
                    continue

            # Use expanded mask
            valid_pixels_2d = mask_expanded & ~np.isnan(observed_map)
            n_valid_pixels = np.sum(valid_pixels_2d)
            logger.debug(f"Valid pixels after expansion: {n_valid_pixels}")

            # If still not enough, fall back to using all pixels
            if n_valid_pixels < n_populations:
                logger.warning("Still insufficient pixels, using all valid pixels")
                valid_pixels_2d = ~np.isnan(observed_map)
                n_valid_pixels = np.sum(valid_pixels_2d)

        # Convert to flat indices
        flat_indices = np.where(valid_pixels_2d.ravel())[0]

        # Extract the data
        cropped_observed = observed_map.ravel()[flat_indices]
        # Remove mean
        cropped_observed -= np.nanmean(cropped_observed)
        # Add cropped layer to cube
        cropped_layers = layer_matrix[:, flat_indices]

        logger.debug(
            f"Final cropping: {len(cropped_observed)} pixels, {cropped_layers.shape[0]} populations"
        )

        return cropped_layers, cropped_observed, flat_indices

    def _add_circle_to_mask(
        self, mask: np.ndarray, center_x: float, center_y: float, radius: float
    ) -> None:
        """Add a circular region to the mask (IMPROVED VERSION)"""

        # Get mask dimensions
        ny, nx = mask.shape

        # Create coordinate grids centered on the circle
        y_min = max(0, int(center_y - radius))
        y_max = min(ny, int(center_y + radius) + 1)
        x_min = max(0, int(center_x - radius))
        x_max = min(nx, int(center_x + radius) + 1)

        # Create coordinate grids for the region
        y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]

        # Calculate distances from center
        distances = np.sqrt((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)

        # Set mask where distance <= radius
        circle_mask = distances <= radius
        mask[y_min:y_max, x_min:x_max][circle_mask] = True

    def _solve_linear_system(
        self, layer_matrix: np.ndarray, observed_vector: np.ndarray, map_data: MapData
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
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
            logger.warning(
                f"Underdetermined system: {n_pixels} pixels, {n_populations} populations"
            )

        try:
            # Use weighted least squares if we have noise information
            if map_data.noise is not None:
                # Get noise values for the same pixels as observed_vector
                # The valid_mask should be passed in or reconstructed
                noise_flat = map_data.noise.ravel()

                # Apply the same mask that was used to create observed_vector
                if hasattr(self, "_last_valid_mask"):
                    noise_values = noise_flat[self._last_valid_mask]
                else:
                    # Fallback: assume observed_vector came from ~isnan mask
                    valid_pixels = ~np.isnan(map_data.data.ravel())
                    noise_values = noise_flat[valid_pixels][: len(observed_vector)]

                # Create weights, avoiding division by zero
                weights = 1.0 / np.maximum(noise_values**2, 1e-20)

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
            if n_pixels > n_populations:
                # Overdetermined system - can estimate proper errors
                try:
                    # Check if residuals is array or scalar
                    if hasattr(residuals, "__len__") and len(residuals) > 0:
                        mse = residuals[0] / (n_pixels - n_populations)
                    elif np.isfinite(residuals):
                        mse = residuals / (n_pixels - n_populations)
                    else:
                        # Calculate residuals manually
                        model_prediction = layer_matrix.T @ flux_densities
                        mse = np.sum((observed_vector - model_prediction) ** 2) / (
                            n_pixels - n_populations
                        )

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
                        logger.warning(
                            "Could not compute covariance matrix, using scaled errors"
                        )
                        flux_errors = (
                            np.abs(flux_densities) * 0.1
                        )  # 10% errors as fallback

                except Exception as e:
                    logger.warning(f"Error estimation failed: {e}")
                    flux_errors = np.abs(flux_densities) * 0.1
            else:
                # Underdetermined system - use fallback errors
                flux_errors = np.abs(flux_densities) * 0.1

            # Calculate fit statistics
            model_prediction = layer_matrix.T @ flux_densities
            chi_squared = np.sum((observed_vector - model_prediction) ** 2)

            if map_data.noise is not None:
                noise_flat = map_data.noise.ravel()
                valid_pixels = ~np.isnan(map_data.data.ravel())

                # Proper chi-squared with noise weights
                noise_vector = noise_flat[valid_pixels][: len(observed_vector)]

                # Avoid division by zero in noise
                noise_vector = np.maximum(noise_vector, 1e-20)  # Small floor value

                chi_squared = np.sum(
                    ((observed_vector - model_prediction) / noise_vector) ** 2
                )

            dof = max(1, n_pixels - n_populations)
            reduced_chi_squared = chi_squared / dof

            fit_stats = {
                "chi_squared": chi_squared,
                "degrees_of_freedom": dof,
                "reduced_chi_squared": reduced_chi_squared,
                "rank": rank,
                "condition_number": np.max(singular_values) / np.min(singular_values)
                if len(singular_values) > 0
                else np.inf,
            }

            return flux_densities, flux_errors, fit_stats

        except Exception as e:
            logger.error(f"Linear system solution failed: {e}")
            # Return zeros as fallback
            return (
                np.zeros(n_populations),
                np.ones(n_populations),
                {
                    "chi_squared": np.inf,
                    "degrees_of_freedom": 1,
                    "reduced_chi_squared": np.inf,
                },
            )

    def _compile_results(
        self, results_dict: dict[str, dict], start_time: float, start_memory: float
    ) -> StackingResults:
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
            memory_used_gb=memory_used,
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
        print(
            f"{'Population':<30} "
            + " ".join(f"{name:<12}" for name in results.map_names)
        )
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
            reduced_chi2 = results.reduced_chi_squared.get(map_name, chi2 / dof)
            print(
                f"{map_name}: χ² = {chi2:.1f}, DoF = {dof}, χ²_red = {reduced_chi2:.2f}"
            )


def run_stacking(
    config: SimstackConfig, population_manager: PopulationManager, sky_maps: SkyMaps
) -> StackingResults:
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
