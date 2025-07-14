"""
Complete SimstackWrapper implementation for Simstack4 with JSON save/load

Replace your current wrapper.py with this version
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .algorithm import run_stacking
from .config import SimstackConfig, load_config
from .cosmology import CosmologyCalculator
from .exceptions.simstack_exceptions import SimstackError
from .populations import PopulationManager
from .results import SimstackResults, create_results_processor
from .sky_catalogs import SkyCatalogs
from .sky_maps import SkyMaps, load_maps
from .utils import setup_logging

logger = setup_logging()


class EnhancedJSONEncoder(json.JSONEncoder):
    """Enhanced JSON encoder for numpy arrays and custom objects"""

    def default(self, obj):
        # Handle numpy types
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)

        # Handle custom enum/object types by converting to string
        if hasattr(obj, "__name__"):  # Class types
            return str(obj.__name__)
        if hasattr(obj, "name"):  # Enum types
            return str(obj.name)
        if hasattr(obj, "value"):  # Enum values
            return obj.value

        # Handle other custom objects by converting to string
        if hasattr(obj, "__dict__"):
            # Try to serialize the object's dictionary
            try:
                return {
                    k: v
                    for k, v in obj.__dict__.items()
                    if isinstance(v | (str, int, float, bool, list, dict, type(None)))
                }
            except Exception:
                return str(obj)

        # Final fallback - convert to string
        return str(obj)


class SimstackWrapper:
    """
    Main wrapper class for Simstack4 operations

    This class provides a simple interface to the complete simstack pipeline,
    handling configuration, data loading, stacking, and results processing.
    """

    def __init__(
        self,
        config: SimstackConfig | str | Path | None = None,
        read_maps: bool = False,
        read_catalog: bool = False,
        stack_automatically: bool = False,
        analyze_automatically: bool = False,  # New parameter
    ):
        """
        Initialize SimstackWrapper

        Args:
            config: Configuration object, file path, or None
            read_maps: Whether to load maps immediately
            read_catalog: Whether to load catalog immediately
            stack_automatically: Whether to run stacking immediately
            analyze_automatically: Whether to run analysis immediately (requires stacking)
        """

        # Handle config loading
        if isinstance(config, str | Path):
            self.config = load_config(config)
            self.config_path = str(config)  # Store original path
        else:
            self.config = config
            self.config_path = None

        # Store embedded config from JSON files
        self._embedded_config: dict | None = None

        # Store population info for reconstruction
        self._population_info: list = []
        self._population_details: dict = {}

        # Store initialization options
        self.read_maps = read_maps
        self.read_catalog = read_catalog
        self.stack_automatically = stack_automatically
        self.analyze_automatically = analyze_automatically

        # Initialize component containers
        self.sky_catalogs: SkyCatalogs | None = None
        self.sky_maps: SkyMaps | None = None
        self.population_manager: PopulationManager | None = None
        self.cosmology_calc: CosmologyCalculator | None = None

        # Results containers
        self.stacking_results = None
        self.processed_results: SimstackResults | None = None
        self.results_dict: dict[str, Any] = {}

        logger.info("SimstackWrapper initialized")

        # Load components if requested
        if read_catalog and self.config:
            self._load_catalog()

        if read_maps and self.config:
            self._load_maps()

        if stack_automatically and self.config:
            self._run_stacking()

        if analyze_automatically and self.config and self.stacking_results:
            self._run_analysis()

    def _load_catalog(self) -> None:
        """Load catalog and create population manager"""
        current_config = self._get_working_config()  # Use helper method

        if not current_config or not current_config.catalog:
            raise SimstackError("No catalog configuration available")

        logger.info("Loading catalog...")

        try:
            # Load catalog using SkyCatalogs
            self.sky_catalogs = SkyCatalogs(
                catalog_config=current_config.catalog,
                backend="auto",
                full_config=current_config,
            )
            self.sky_catalogs.load_catalog()

            # Create population manager from loaded catalog
            if self.sky_catalogs.population_manager:
                self.population_manager = self.sky_catalogs.population_manager
                logger.info(
                    f"✓ Catalog loaded: {len(self.population_manager)} populations created"
                )
            else:
                logger.warning("Catalog loaded but no population manager created")

        except Exception as e:
            logger.error(f"Failed to load catalog: {e}")
            raise SimstackError(f"Catalog loading failed: {e}") from e

    def _get_working_config(self) -> SimstackConfig:
        """Get the working config (either loaded or reconstructed from JSON)"""
        if self.config:
            return self.config
        elif self._embedded_config:
            return self._reconstruct_config_from_json()
        else:
            raise SimstackError("No configuration available")

    def _load_maps(self) -> None:
        """Load all configured maps"""
        if not self.config or not self.config.maps:
            raise SimstackError("No maps configuration provided")

        logger.info("Loading maps...")

        try:
            # Load maps using SkyMaps
            self.sky_maps = load_maps(self.config.maps)
            logger.info(f"✓ Maps loaded: {len(self.sky_maps)} maps available")

        except Exception as e:
            logger.error(f"Failed to load maps: {e}")
            raise SimstackError(f"Map loading failed: {e}") from e

    def _run_stacking(self) -> None:
        """Run ONLY the stacking computation (no analysis)"""
        if not self.population_manager:
            logger.info("Loading catalog for stacking...")
            self._load_catalog()

        if not self.sky_maps:
            logger.info("Loading maps for stacking...")
            self._load_maps()

        if not self.cosmology_calc:
            self.cosmology_calc = CosmologyCalculator(self.config.cosmology)

        logger.info("Starting stacking pipeline...")

        try:
            # Run ONLY stacking algorithm - no analysis yet
            self.stacking_results = run_stacking(
                config=self.config,
                population_manager=self.population_manager,
                sky_maps=self.sky_maps,
            )

            logger.info("✓ Stacking computation completed!")

        except Exception as e:
            logger.error(f"Stacking pipeline failed: {e}")
            raise SimstackError(f"Stacking failed: {e}") from e

    def _run_analysis(
        self,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        use_schreiber_prior: bool = False,
    ) -> None:
        """Run ONLY the analysis/SED fitting (requires stacking results)"""
        if not self.stacking_results:
            raise SimstackError("No stacking results available for analysis")

        if not self.population_manager:
            logger.info("Loading catalog for analysis...")
            self._load_catalog()

        fitting_method = "MCMC" if use_mcmc else "curve_fit"
        prior_type = "Schreiber+2015" if use_schreiber_prior else "flat"
        logger.info(
            f"Starting analysis pipeline with {fitting_method} fitting and {prior_type} priors..."
        )

        try:
            # Process results - SED fitting, error estimation, etc.
            self.processed_results = create_results_processor(
                config=self.config,
                stacking_results=self.stacking_results,
                population_manager=self.population_manager,
                use_mcmc=use_mcmc,
                mcmc_iterations=mcmc_iterations,
                mcmc_burn_in=mcmc_burn_in,
                use_schreiber_prior=use_schreiber_prior,
            )

            # Update results dict for backward compatibility
            self.results_dict = {
                "band_results_dict": self.processed_results.band_results,
                "population_summary": self.processed_results.get_population_summary(),
                "stacking_results": self.stacking_results,
                "processed_results": self.processed_results,
            }

            logger.info("✓ Analysis completed successfully!")

        except Exception as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise SimstackError(f"Analysis failed: {e}") from e

    def save_stacking_results(self, filepath: str | Path) -> None:
        """Save raw stacking results as JSON with embedded config"""
        if not self.stacking_results:
            raise SimstackError("No stacking results to save")

        filepath = Path(filepath).with_suffix(".json")

        try:
            # Extract stacking data as simple types
            stacking_data = self._extract_stacking_data()

            # Embed the full configuration in the JSON file
            if self.config:
                stacking_data["embedded_config"] = self._serialize_config(self.config)

            # Add metadata
            stacking_data["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "version": "simstack4_json_self_contained",
                "config_path": self.config_path,
                "n_populations": len(self.population_manager)
                if self.population_manager
                else 0,
                "n_maps": len(self.sky_maps) if self.sky_maps else 0,
            }

            # Use the enhanced encoder
            with open(filepath, "w") as f:
                json.dump(stacking_data, f, indent=2, cls=EnhancedJSONEncoder)

            logger.info(f"✓ Stacking results saved as self-contained JSON: {filepath}")

        except Exception as e:
            logger.error(f"Failed to save stacking results: {e}")

            # Try to save without embedded config as fallback
            try:
                logger.info("Attempting to save without embedded config...")
                stacking_data = self._extract_stacking_data()
                stacking_data["metadata"] = {
                    "timestamp": datetime.now().isoformat(),
                    "version": "simstack4_json_basic",
                    "config_path": self.config_path,
                    "n_populations": len(self.population_manager)
                    if self.population_manager
                    else 0,
                    "n_maps": len(self.sky_maps) if self.sky_maps else 0,
                    "note": "Config embedding failed, use original config file",
                }

                with open(filepath, "w") as f:
                    json.dump(stacking_data, f, indent=2, cls=EnhancedJSONEncoder)

                logger.info(
                    f"✓ Stacking results saved (without embedded config): {filepath}"
                )

            except Exception as e2:
                logger.error(f"Complete save failure: {e2}")
                raise SimstackError("Could not save stacking results.") from e2

    def load_stacking_results(self, filepath: str | Path) -> None:
        """Load stacking results from JSON with validation and automatic catalog loading"""
        filepath = Path(filepath)

        # Try .json extension if original doesn't exist
        if not filepath.exists() and filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")

        if not filepath.exists():
            raise SimstackError(f"Stacking results file not found: {filepath}")

        with open(filepath) as f:
            stacking_data = json.load(f)

        # Validate JSON structure (skip validation for now to handle legacy files)
        # if not self.validate_json_structure(stacking_data):
        #     raise SimstackError("Invalid JSON structure - missing required data for analysis")

        # Reconstruct stacking results object
        self.stacking_results = type("StackingResults", (), {})()

        for key, value in stacking_data.items():
            if key not in [
                "metadata",
                "populations",
                "population_details",
                "embedded_config",
            ]:
                if isinstance(value, list):
                    # Convert back to numpy arrays
                    setattr(self.stacking_results, key, np.array(value))
                else:
                    setattr(self.stacking_results, key, value)

        # Store embedded config if available
        self._embedded_config = stacking_data.get("embedded_config")

        # Store population info for reference and reconstruction
        self._population_info = stacking_data.get("populations", [])
        self._population_details = stacking_data.get("population_details", {})

        # Try to load the original config if available, otherwise use embedded
        metadata = stacking_data.get("metadata", {})
        original_config_path = metadata.get("config_path")

        if not self.config and original_config_path:
            try:
                logger.info(
                    f"Attempting to load original config: {original_config_path}"
                )
                self.config = load_config(original_config_path)
                self.config_path = original_config_path
            except Exception as e:
                logger.warning(
                    f"Original config not found ({e}), will use embedded config"
                )

        # Log loading status
        logger.info(f"✓ Stacking results loaded from JSON: {filepath}")
        logger.info(f"  - {len(self._population_info)} populations")
        logger.info(
            f"  - Config source: {'original file' if self.config else 'embedded in JSON'}"
        )

        # Handle legacy JSON files that don't have embedded config
        if not self.config and not self._embedded_config:
            logger.warning("Legacy JSON file detected - no embedded config found")
            logger.warning("You need to provide the config file path manually")
            raise SimstackError(
                "Legacy JSON file without embedded config. "
                "Please use: wrapper = SimstackWrapper(config='path/to/config.toml') "
                "before loading stacking results."
            )

        # Automatically load catalog if we have config
        if self.config or self._embedded_config:
            try:
                logger.info("Automatically loading catalog for analysis...")
                self._load_catalog()
            except Exception as e:
                logger.warning(f"Failed to auto-load catalog: {e}")
                logger.info("You can manually call wrapper.load_catalog() if needed")
        else:
            logger.warning(
                "No configuration available - cannot load catalog automatically"
            )

    def _serialize_config(self, config: SimstackConfig) -> dict:
        """Enhanced config serialization that handles custom objects safely"""

        def safe_serialize(obj):
            """Recursively serialize an object, converting custom types to strings"""
            if obj is None:
                return None
            elif isinstance(obj | (str, int, float, bool)):
                return obj
            elif isinstance(obj | (list, tuple)):
                return [safe_serialize(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: safe_serialize(v) for k, v in obj.items()}
            elif hasattr(obj, "__dict__"):
                # Handle objects with attributes
                return {k: safe_serialize(v) for k, v in obj.__dict__.items()}
            elif hasattr(obj, "name"):  # Enum types
                return str(obj.name)
            elif hasattr(obj, "value"):  # Enum values
                return obj.value
            else:
                # Convert everything else to string
                return str(obj)

        config_dict = {}

        # Safely serialize each major config section
        try:
            # Handle catalog config
            if hasattr(config, "catalog") and config.catalog:
                config_dict["catalog"] = safe_serialize(config.catalog)

            # Handle maps config
            if hasattr(config, "maps") and config.maps:
                config_dict["maps"] = safe_serialize(config.maps)

            # Handle other sections
            for attr in ["cosmology", "output", "binning", "error_estimator"]:
                if hasattr(config, attr):
                    config_dict[attr] = safe_serialize(getattr(config, attr))

        except Exception as e:
            logger.warning(f"Config serialization failed for some sections: {e}")
            # Fallback - just serialize what we can
            config_dict = safe_serialize(config)

        return config_dict

    def _reconstruct_config_from_json(self) -> SimstackConfig:
        """Enhanced config reconstruction from embedded JSON data"""
        if not self._embedded_config:
            raise SimstackError("No embedded config data available")

        try:
            # Import the config classes
            from .config import (
                AstrometryConfig,
                BeamConfig,
                BinningConfig,
                BootstrapConfig,
                CatalogConfig,
                ClassificationConfig,
                ErrorEstimatorConfig,
                MapConfig,
                OutputConfig,
                SimstackConfig,
            )

            # Reconstruct catalog config
            catalog_data = self._embedded_config.get("catalog", {})

            # Astrometry
            astrometry_data = catalog_data.get("astrometry", {})
            astrometry_config = AstrometryConfig(
                ra=astrometry_data.get("ra", "ra"),
                dec=astrometry_data.get("dec", "dec"),
            )

            # Classification with binning
            classification_data = catalog_data.get("classification", {})

            # Redshift binning
            redshift_binning_data = classification_data.get("binning", {}).get(
                "redshift", {}
            )
            redshift_binning = BinningConfig(
                id=redshift_binning_data.get("id", "zpdf_med"),
                bins=redshift_binning_data.get("bins", []),
            )

            # Stellar mass binning
            mass_binning_data = classification_data.get("binning", {}).get(
                "stellar_mass", {}
            )
            mass_binning = BinningConfig(
                id=mass_binning_data.get("id", "mass_med"),
                bins=mass_binning_data.get("bins", []),
            )

            # Split params
            split_params_data = classification_data.get("split_params", {})

            # Classification config
            classification_config = ClassificationConfig(
                split_type=classification_data.get("split_type", "labels"),
                split_params=split_params_data,
                binning={"redshift": redshift_binning, "stellar_mass": mass_binning},
            )

            # Full catalog config
            catalog_config = CatalogConfig(
                path=catalog_data.get("path", ""),
                file=catalog_data.get("file", ""),
                astrometry=astrometry_config,
                classification=classification_config,
            )

            # Reconstruct maps config
            maps_data = self._embedded_config.get("maps", {})
            maps_config = {}
            for map_name, map_data in maps_data.items():
                # Beam config
                beam_data = map_data.get("beam", {})
                beam_config = BeamConfig(
                    fwhm=beam_data.get("fwhm", 0), area_sr=beam_data.get("area_sr", 1.0)
                )

                # Map config
                maps_config[map_name] = MapConfig(
                    wavelength=map_data.get("wavelength", 0),
                    path_map=map_data.get("path_map", ""),
                    path_noise=map_data.get("path_noise", ""),
                    color_correction=map_data.get("color_correction", 1.0),
                    beam=beam_config,
                )

            # Other config sections
            output_data = self._embedded_config.get("output", {})
            output_config = OutputConfig(
                folder=output_data.get("folder", ""),
                shortname=output_data.get("shortname", ""),
            )

            binning_data = self._embedded_config.get("binning", {})
            binning_config = {
                "stack_all_z_at_once": binning_data.get("stack_all_z_at_once", True),
                "add_foreground": binning_data.get("add_foreground", True),
                "crop_circles": binning_data.get("crop_circles", True),
            }

            error_est_data = self._embedded_config.get("error_estimator", {})
            bootstrap_data = error_est_data.get("bootstrap", {})
            bootstrap_config = BootstrapConfig(
                enabled=bootstrap_data.get("enabled", True),
                iterations=bootstrap_data.get("iterations", 5),
                initial_seed=bootstrap_data.get("initial_seed", 1),
                cosmology=bootstrap_data.get("cosmology", "Planck18"),
            )

            error_estimator_config = ErrorEstimatorConfig(
                write_simmaps=error_est_data.get("write_simmaps", False),
                randomize=error_est_data.get("randomize", False),
                bootstrap=bootstrap_config,
            )

            # Create the full config
            config = SimstackConfig(
                catalog=catalog_config,
                maps=maps_config,
                cosmology=self._embedded_config.get("cosmology", "Planck18"),
                output=output_config,
                binning=binning_config,
                error_estimator=error_estimator_config,
            )

            return config

        except Exception as e:
            logger.error(f"Failed to reconstruct config from JSON: {e}")
            raise SimstackError(f"Config reconstruction failed: {e}") from e

    def _extract_stacking_data(self) -> dict:
        """Enhanced stacking data extraction with better population info"""
        data = {}

        # Extract stacking results
        if hasattr(self.stacking_results, "__dict__"):
            for key, value in self.stacking_results.__dict__.items():
                if isinstance(value, dict):
                    # Handle dictionaries (like flux_densities, flux_errors)
                    data[key] = {}
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, np.ndarray | list | float | int | str):
                            data[key][sub_key] = sub_value
                        else:
                            data[key][sub_key] = str(sub_value)
                elif isinstance(value, np.ndarray | list | float | int | str):
                    data[key] = value
                elif hasattr(value, "__dict__"):
                    # Handle objects with attributes
                    data[key] = {
                        k: v
                        for k, v in value.__dict__.items()
                        if isinstance(v, np.ndarray | list | float | int | str)
                    }
                else:
                    data[key] = str(value)

        # Enhanced population info extraction
        if self.population_manager:
            data["populations"] = []
            data["population_details"] = {}

            for i, (pop_id, pop) in enumerate(
                self.population_manager.populations.items()
            ):
                # Basic population data
                pop_data = {
                    "index": i,
                    "population_id": pop_id,
                    "n_sources": getattr(pop, "n_sources", 0),
                    "median_redshift": getattr(pop, "median_redshift", 0),
                    "median_stellar_mass": getattr(pop, "median_stellar_mass", 0),
                }

                # Try to get additional population properties
                for attr in ["redshift_range", "mass_range", "classification_value"]:
                    if hasattr(pop, attr):
                        pop_data[attr] = getattr(pop, attr)

                data["populations"].append(pop_data)

                # Store detailed population info (coordinates, etc.) separately
                # This allows for reconstruction of population manager
                pop_details = {}
                for attr in [
                    "source_indices",
                    "coordinates",
                    "redshifts",
                    "stellar_masses",
                ]:
                    if hasattr(pop, attr):
                        value = getattr(pop, attr)
                        if isinstance(value, np.ndarray):
                            pop_details[attr] = value.tolist()
                        else:
                            pop_details[attr] = value

                data["population_details"][pop_id] = pop_details

        return data

    # Additional method for better JSON structure validation
    def validate_json_structure(self, json_data: dict) -> bool:
        """Validate that JSON file has required structure for analysis"""
        required_keys = [
            "flux_densities",
            "flux_errors",
            "map_names",
            "population_labels",
        ]

        for key in required_keys:
            if key not in json_data:
                logger.warning(f"Missing required key in JSON: {key}")
                return False

        # Check for embedded config or metadata with config path
        has_config = "embedded_config" in json_data
        has_config_path = json_data.get("metadata", {}).get("config_path") is not None

        if not (has_config or has_config_path):
            logger.warning("JSON file missing both embedded config and config path")
            return False

        return True

    def save_analysis_results(self, filepath: str | Path) -> None:
        """Save analysis results"""
        if not self.processed_results:
            raise SimstackError("No analysis results to save")

        self.processed_results.save_results(filepath)
        logger.info(f"✓ Analysis results saved to: {filepath}")

    # Public methods
    def load_catalog(self) -> None:
        """Public method to load catalog"""
        self._load_catalog()

    def load_maps(self) -> None:
        """Public method to load maps"""
        self._load_maps()

    def run_stacking_only(self) -> None:
        """Public method to run ONLY stacking"""
        self._run_stacking()

    def run_analysis_only(
        self,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        use_schreiber_prior: bool = False,
    ) -> SimstackResults:
        """
        Public method to run ONLY analysis (requires stacking results)

        Args:
            use_mcmc: Whether to use MCMC fitting (requires emcee)
            mcmc_iterations: Number of MCMC iterations
            mcmc_burn_in: Number of burn-in iterations to discard
            use_schreiber_prior: Whether to use Schreiber+2015 T_dust vs redshift prior
        """
        # Reload catalog if needed for analysis
        if not self.population_manager and self.config:
            self._load_catalog()

        self._run_analysis(
            use_mcmc=use_mcmc,
            mcmc_iterations=mcmc_iterations,
            mcmc_burn_in=mcmc_burn_in,
            use_schreiber_prior=use_schreiber_prior,
        )
        return self.processed_results

    def run_complete_pipeline(
        self,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        use_schreiber_prior: bool = False,
    ) -> SimstackResults:
        """
        Run both stacking and analysis

        Args:
            use_mcmc: Whether to use MCMC fitting (requires emcee)
            mcmc_iterations: Number of MCMC iterations
            mcmc_burn_in: Number of burn-in iterations to discard
            use_schreiber_prior: Whether to use Schreiber+2015 T_dust vs redshift prior
        """
        self._run_stacking()
        self._run_analysis(
            use_mcmc=use_mcmc,
            mcmc_iterations=mcmc_iterations,
            mcmc_burn_in=mcmc_burn_in,
            use_schreiber_prior=use_schreiber_prior,
        )
        return self.processed_results

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current state"""
        summary = {
            "config_loaded": self.config is not None,
            "catalog_loaded": self.sky_catalogs is not None,
            "maps_loaded": self.sky_maps is not None,
            "populations_created": self.population_manager is not None,
            "stacking_completed": self.stacking_results is not None,
            "analysis_completed": self.processed_results is not None,
        }

        if self.population_manager:
            summary["n_populations"] = len(self.population_manager)

        if self.sky_maps:
            summary["n_maps"] = len(self.sky_maps)
            summary["map_names"] = list(self.sky_maps.maps.keys())

        if self.processed_results:
            summary["n_results"] = len(self.processed_results.sed_results)

        return summary

    def print_summary(self) -> None:
        """Print formatted summary"""
        summary = self.get_summary()

        print("=== SimstackWrapper Summary ===")
        print(f"Config loaded: {'✓' if summary['config_loaded'] else '✗'}")
        print(f"Catalog loaded: {'✓' if summary['catalog_loaded'] else '✗'}")
        print(f"Maps loaded: {'✓' if summary['maps_loaded'] else '✗'}")
        print(f"Populations created: {'✓' if summary['populations_created'] else '✗'}")
        print(f"Stacking completed: {'✓' if summary['stacking_completed'] else '✗'}")
        print(f"Analysis completed: {'✓' if summary['analysis_completed'] else '✗'}")

        if summary.get("n_populations"):
            print(f"Number of populations: {summary['n_populations']}")

        if summary.get("n_maps"):
            print(f"Number of maps: {summary['n_maps']}")
            print(f"Map names: {', '.join(summary['map_names'])}")

        if summary.get("n_results"):
            print(f"Results available for {summary['n_results']} populations")

    def __len__(self) -> int:
        """Return number of populations (for backward compatibility)"""
        return len(self.population_manager) if self.population_manager else 0

    def __repr__(self) -> str:
        """String representation"""
        status = []
        if self.sky_catalogs:
            status.append(f"{len(self.population_manager)} populations")
        if self.sky_maps:
            status.append(f"{len(self.sky_maps)} maps")
        if self.stacking_results:
            status.append("stacking complete")
        if self.processed_results:
            status.append("analysis complete")

        status_str = ", ".join(status) if status else "not initialized"
        return f"SimstackWrapper({status_str})"


# Convenience functions
def run_stacking_only(
    config_path: str | Path, save_path: str | Path = None
) -> SimstackWrapper:
    """
    Run ONLY stacking and save results

    Args:
        config_path: Path to configuration file
        save_path: Path to save stacking results (optional)

    Returns:
        Wrapper with stacking results
    """
    wrapper = SimstackWrapper(
        config=config_path,
        read_maps=True,
        read_catalog=True,
        stack_automatically=True,
        analyze_automatically=False,
    )

    if save_path:
        wrapper.save_stacking_results(save_path)

    return wrapper


def run_analysis_only(
    stacking_results_path: str | Path,
    save_path: str | Path = None,
    use_mcmc: bool = False,
    mcmc_iterations: int = 1000,
    mcmc_burn_in: int = 200,
    use_schreiber_prior: bool = False,
) -> SimstackResults:
    """
    Run analysis from self-contained JSON file (no config needed!)

    Args:
        stacking_results_path: Path to saved stacking results JSON
        save_path: Path to save analysis results (optional)
        use_mcmc: Whether to use MCMC fitting (requires emcee)
        mcmc_iterations: Number of MCMC iterations
        mcmc_burn_in: Number of burn-in iterations to discard
        use_schreiber_prior: Whether to use Schreiber+2015 T_dust vs redshift prior

    Returns:
        Processed results object
    """
    wrapper = SimstackWrapper()  # No config needed!
    wrapper.load_stacking_results(stacking_results_path)
    results = wrapper.run_analysis_only(
        use_mcmc=use_mcmc,
        mcmc_iterations=mcmc_iterations,
        mcmc_burn_in=mcmc_burn_in,
        use_schreiber_prior=use_schreiber_prior,
    )

    if save_path:
        wrapper.save_analysis_results(save_path)

    return results
