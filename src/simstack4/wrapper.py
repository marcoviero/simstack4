"""
Enhanced wrapper.py with comprehensive catalog metadata saving - RUFF COMPLIANT

This version saves complete catalog metadata to JSON files, making them fully self-contained
for analysis without requiring the original TOML config file.
"""

from __future__ import annotations

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
        if isinstance(obj, np.integer | np.int32 | np.int64):
            return int(obj)
        if isinstance(obj, np.floating | np.float32 | np.float64):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)

        # Handle pandas Series/DataFrame
        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict()
            except (AttributeError, TypeError, ValueError):
                pass

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
                    if isinstance(
                        v, str | int | float | bool | list | dict | type(None)
                    )
                }
            except (AttributeError, TypeError, ValueError):
                return str(obj)

        # Final fallback - convert to string
        return str(obj)


class SimstackWrapper:
    """
    Main wrapper class for Simstack4 operations with enhanced JSON metadata
    """

    def __init__(
        self,
        config: SimstackConfig | str | Path | None = None,
        read_maps: bool = False,
        read_catalog: bool = False,
        stack_automatically: bool = False,
        analyze_automatically: bool = False,
        use_covariance: bool = True,  # NEW
        correlation_matrix: dict | None = None,  # NEW
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
            self.config_path = str(config)
        else:
            self.config = config
            self.config_path = None

        # NEW: Covariance support
        self.use_covariance = use_covariance
        self.correlation_matrix = (
            correlation_matrix or self._get_default_correlation_matrix()
        )

        # Store embedded config and catalog metadata
        self._embedded_config: dict | None = None
        self._catalog_metadata: dict | None = None
        self._catalog_data_cache: dict | None = None

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

    def _get_default_correlation_matrix(self) -> dict:
        """Get the default correlation matrix from your data"""
        return {
            24: {
                24: 1.00,
                70: 0.23,
                100: 0.33,
                160: 0.32,
                250: 0.28,
                350: 0.17,
                500: 0.07,
                850: 0.10,
            },
            70: {
                24: 0.23,
                70: 1.00,
                100: 0.19,
                160: 0.24,
                250: 0.23,
                350: 0.14,
                500: 0.06,
                850: 0.08,
            },
            100: {
                24: 0.33,
                70: 0.19,
                100: 1.00,
                160: 0.28,
                250: 0.21,
                350: 0.11,
                500: 0.04,
                850: 0.05,
            },
            160: {
                24: 0.32,
                70: 0.24,
                100: 0.28,
                160: 1.00,
                250: 0.35,
                350: 0.23,
                500: 0.10,
                850: 0.13,
            },
            250: {
                24: 0.28,
                70: 0.23,
                100: 0.21,
                160: 0.35,
                250: 1.00,
                350: 0.37,
                500: 0.18,
                850: 0.28,
            },
            350: {
                24: 0.17,
                70: 0.14,
                100: 0.11,
                160: 0.23,
                250: 0.37,
                350: 1.00,
                500: 0.20,
                850: 0.33,
            },
            500: {
                24: 0.07,
                70: 0.06,
                100: 0.04,
                160: 0.10,
                250: 0.18,
                350: 0.20,
                500: 1.00,
                850: 0.23,
            },
            850: {
                24: 0.10,
                70: 0.08,
                100: 0.05,
                160: 0.13,
                250: 0.28,
                350: 0.33,
                500: 0.23,
                850: 1.00,
            },
        }

    def estimate_bootstrap_covariance(self):
        """
        Estimate bootstrap covariance matrices from loaded stacking results
        Only called if bootstrap data is available in the JSON

        Returns:
        --------
        bootstrap_covariances : dict
            Dictionary mapping population_id -> covariance matrix
        """
        if not self.stacking_results:
            logger.warning("No stacking results loaded")
            return {}

        # Check different possible locations for bootstrap data
        bootstrap_data = None

        # Try different attribute names for bootstrap data
        for attr_name in [
            "bootstrap_flux_samples",
            "bootstrap_iterations",
            "bootstrap_results",
            "bootstrap_data",
            "bootstrap",
        ]:
            if hasattr(self.stacking_results, attr_name):
                bootstrap_data = getattr(self.stacking_results, attr_name)
                logger.info(f"Found bootstrap data in attribute: {attr_name}")
                break

        if bootstrap_data is None:
            logger.info("No bootstrap data found in stacking results")
            return {}

        # Handle different data structures
        if isinstance(bootstrap_data, int):
            logger.info(
                f"Bootstrap iterations count: {bootstrap_data}, but no iteration data found"
            )
            return {}

        if isinstance(bootstrap_data, dict):
            # Check if it's the map-based structure like yours
            map_names = list(bootstrap_data.keys())
            if all(isinstance(bootstrap_data[key], list) for key in map_names):
                # This looks like: {'mips_24': [...], 'pacs_green': [...], ...}
                logger.info(
                    f"Found map-based bootstrap structure with {len(map_names)} bands"
                )

                # Get number of bootstrap iterations and populations
                first_map = map_names[0]
                bootstrap_samples = bootstrap_data[first_map]

                if not bootstrap_samples or len(bootstrap_samples) == 0:
                    logger.warning("Empty bootstrap samples")
                    return {}

                # Assume structure is: [iteration][population_index]
                n_iterations = len(bootstrap_samples)
                n_populations = len(bootstrap_samples[0]) if bootstrap_samples[0] else 0

                logger.info(
                    f"Bootstrap structure: {n_iterations} iterations, {n_populations} populations"
                )

                # Reconstruct iteration-based structure
                bootstrap_iterations = []
                for iter_idx in range(n_iterations):
                    iteration_data = {"flux_densities": {}}
                    for map_name in map_names:
                        try:
                            iteration_data["flux_densities"][map_name] = bootstrap_data[
                                map_name
                            ][iter_idx]
                        except (IndexError, KeyError) as e:
                            logger.warning(
                                f"Error accessing bootstrap data for {map_name}, iteration {iter_idx}: {e}"
                            )
                            continue
                    bootstrap_iterations.append(iteration_data)

            elif "iterations" in bootstrap_data:
                bootstrap_iterations = bootstrap_data["iterations"]
            elif "results" in bootstrap_data:
                bootstrap_iterations = bootstrap_data["results"]
            else:
                logger.warning(
                    f"Bootstrap data is dict but unrecognized structure. Keys: {list(bootstrap_data.keys())}"
                )
                return {}
        elif isinstance(bootstrap_data, list):
            # Bootstrap data is directly a list of iterations
            bootstrap_iterations = bootstrap_data
        else:
            logger.warning(f"Unknown bootstrap data type: {type(bootstrap_data)}")
            return {}

        if not bootstrap_iterations or len(bootstrap_iterations) == 0:
            logger.info("No bootstrap iterations found")
            return {}

        population_labels = self.stacking_results.population_labels
        map_names = self.stacking_results.map_names

        logger.info(f"Found {len(bootstrap_iterations)} bootstrap iterations")
        logger.info("Calculating bootstrap covariance matrices...")

        # Calculate covariance for each population
        bootstrap_covariances = {}

        for pop_idx, pop_id in enumerate(population_labels):
            if pop_id == "foreground":
                continue  # Skip foreground

            # Collect flux measurements across bootstrap iterations
            pop_fluxes = []

            for iteration in bootstrap_iterations:
                iter_fluxes = []
                for map_name in map_names:
                    try:
                        # Handle different possible structures for iteration data
                        if isinstance(iteration, dict):
                            if "flux_densities" in iteration:
                                flux = iteration["flux_densities"][map_name][pop_idx]
                            elif map_name in iteration:
                                flux = iteration[map_name][pop_idx]
                            else:
                                logger.warning(
                                    f"Cannot find flux data for {map_name} in iteration"
                                )
                                flux = 0.0
                        else:
                            logger.warning(
                                f"Unexpected iteration structure: {type(iteration)}"
                            )
                            flux = 0.0

                        iter_fluxes.append(flux)
                    except (KeyError, IndexError, TypeError) as e:
                        logger.warning(
                            f"Error extracting flux for {map_name}, pop {pop_idx}: {e}"
                        )
                        iter_fluxes.append(0.0)

                pop_fluxes.append(iter_fluxes)

            # Convert to numpy array and calculate covariance
            pop_fluxes = np.array(pop_fluxes)  # Shape: (n_iterations, n_bands)

            if len(pop_fluxes) > 1 and pop_fluxes.shape[1] > 1:
                # Calculate covariance matrix
                pop_cov = np.cov(pop_fluxes.T)  # Shape: (n_bands, n_bands)

                # Check for valid covariance matrix
                if np.all(np.isfinite(pop_cov)) and np.all(np.diag(pop_cov) > 0):
                    bootstrap_covariances[pop_id] = pop_cov

                    # Log diagonal vs off-diagonal elements
                    diag_mean = np.mean(np.diag(pop_cov))
                    off_diag_mean = np.mean(pop_cov[np.triu_indices_from(pop_cov, k=1)])
                    logger.info(
                        f"Population {pop_id}: diag={diag_mean:.2e}, off-diag={off_diag_mean:.2e}"
                    )
                else:
                    logger.warning(f"Invalid covariance matrix for population {pop_id}")
            else:
                logger.warning(
                    f"Insufficient bootstrap data for population {pop_id}: shape {pop_fluxes.shape}"
                )

        logger.info(
            f"Calculated bootstrap covariances for {len(bootstrap_covariances)} populations"
        )
        return bootstrap_covariances

    def _extract_catalog_metadata(self) -> dict:
        """Extract comprehensive catalog metadata for JSON embedding"""
        if not self.sky_catalogs or not self.population_manager:
            logger.warning("No catalog loaded - cannot extract metadata")
            return {}

        metadata = {}

        try:
            # Basic catalog info - FIX: Check for empty properly
            catalog_df = getattr(self.sky_catalogs, "catalog_df", None)
            n_total_sources = 0
            catalog_columns = []

            if catalog_df is not None:
                if hasattr(catalog_df, "empty"):
                    # It's a pandas DataFrame
                    if not catalog_df.empty:
                        n_total_sources = len(catalog_df)
                        catalog_columns = list(catalog_df.columns)
                elif hasattr(catalog_df, "__len__"):
                    # It's some other iterable
                    n_total_sources = len(catalog_df)
                    if hasattr(catalog_df, "columns"):
                        catalog_columns = list(catalog_df.columns)

            metadata["catalog_info"] = {
                "n_total_sources": n_total_sources,
                "n_populations": len(self.population_manager.populations),
                "catalog_columns": catalog_columns,
                "catalog_file": self.config.catalog.file if self.config else None,
                "catalog_path": self.config.catalog.path if self.config else None,
            }

            # Population statistics
            metadata["population_stats"] = {}
            for pop_id, population in self.population_manager.populations.items():
                stats = {
                    "n_sources": getattr(population, "n_sources", 0),
                    "median_redshift": getattr(population, "median_redshift", 0.0),
                    "median_stellar_mass": getattr(
                        population, "median_stellar_mass", 0.0
                    ),
                    "redshift_range": getattr(population, "redshift_range", [0.0, 0.0]),
                    "mass_range": getattr(population, "mass_range", [0.0, 0.0]),
                }

                # Add additional attributes if they exist
                for attr in ["classification_value", "bin_edges", "selection_criteria"]:
                    if hasattr(population, attr):
                        value = getattr(population, attr)
                        if isinstance(value, np.ndarray):
                            stats[attr] = value.tolist()
                        else:
                            stats[attr] = value

                metadata["population_stats"][pop_id] = stats

            # Binning configuration used
            if self.config and self.config.catalog.classification.binning:
                metadata["binning_config"] = {}
                for (
                    bin_name,
                    bin_config,
                ) in self.config.catalog.classification.binning.items():
                    metadata["binning_config"][bin_name] = {
                        "id": bin_config.id,
                        "label": bin_config.label,
                        "bins": bin_config.bins,
                        "formula_ref": bin_config.formula_ref,
                    }

            # Coordinate system info
            if self.config:
                metadata["astrometry"] = dict(self.config.catalog.astrometry)

            # Sample of catalog data for validation (first 100 sources) - FIX: Better handling
            if catalog_df is not None and n_total_sources > 0:
                try:
                    # Try to get sample data
                    if hasattr(catalog_df, "head"):
                        sample_df = catalog_df.head(100)
                    elif hasattr(catalog_df, "__getitem__"):
                        sample_df = (
                            catalog_df[:100] if len(catalog_df) > 100 else catalog_df
                        )
                    else:
                        sample_df = None

                    if sample_df is not None and hasattr(sample_df, "columns"):
                        # Convert to JSON-serializable format
                        catalog_sample = {}
                        for col in sample_df.columns:
                            try:
                                series = sample_df[col]
                                # Convert pandas nullable extension types (Int64,
                                # Float64, String, etc.) to numpy-compatible values.
                                # These raise TypeError with np.issubdtype.
                                try:
                                    values = series.to_numpy(
                                        dtype=float, na_value=np.nan
                                    )
                                    catalog_sample[col] = values.tolist()
                                except (ValueError, TypeError):
                                    # Non-numeric column — convert to strings
                                    catalog_sample[col] = [
                                        str(v) if v is not None and v == v else None
                                        for v in series
                                    ]
                            except Exception as e:
                                logger.warning(
                                    f"Could not serialize column {col}: {e}"
                                )
                                catalog_sample[col] = None

                        metadata["catalog_sample"] = catalog_sample
                except (AttributeError, TypeError, ValueError, KeyError) as e:
                    logger.warning(f"Could not create catalog sample: {e}")

            logger.info(
                f"Extracted catalog metadata for {metadata['catalog_info']['n_populations']} populations"
            )

        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"Failed to extract catalog metadata: {e}")
            metadata["extraction_error"] = str(e)

        return metadata

    def _extract_population_data(self) -> dict:
        """Extract detailed population data for reconstruction"""
        if not self.population_manager:
            return {}

        population_data = {
            "populations": [],
            "population_details": {},
            "reconstruction_info": {
                "total_sources": 0,
                "coordinate_system": "icrs",  # Default
                "required_columns": [],
            },
        }

        try:
            total_sources = 0
            required_columns = set()

            for i, (pop_id, population) in enumerate(
                self.population_manager.populations.items()
            ):
                # Basic population info
                pop_info = {
                    "index": i,
                    "population_id": pop_id,
                    "n_sources": getattr(population, "n_sources", 0),
                    "median_redshift": getattr(population, "median_redshift", 0.0),
                    "median_stellar_mass": getattr(
                        population, "median_stellar_mass", 0.0
                    ),
                }

                # Add range information
                for attr in ["redshift_range", "mass_range", "classification_value"]:
                    if hasattr(population, attr):
                        value = getattr(population, attr)
                        if isinstance(value, np.ndarray):
                            pop_info[attr] = value.tolist()
                        else:
                            pop_info[attr] = value

                population_data["populations"].append(pop_info)
                total_sources += pop_info["n_sources"]

                # Detailed data for reconstruction
                pop_details = {}

                # Source indices (critical for reconstruction)
                if hasattr(population, "source_indices"):
                    indices = population.source_indices
                    if isinstance(indices, np.ndarray):
                        pop_details["source_indices"] = indices.tolist()
                    else:
                        pop_details["source_indices"] = list(indices)

                # Coordinates (essential for stacking)
                if hasattr(population, "coordinates"):
                    coords = population.coordinates
                    if hasattr(coords, "ra") and hasattr(coords, "dec"):
                        pop_details["coordinates"] = {
                            "ra": coords.ra.degree.tolist()
                            if hasattr(coords.ra, "degree")
                            else coords.ra.tolist(),
                            "dec": coords.dec.degree.tolist()
                            if hasattr(coords.dec, "degree")
                            else coords.dec.tolist(),
                        }
                        required_columns.update(["ra", "dec"])

                # Physical properties
                for attr in ["redshifts", "stellar_masses"]:
                    if hasattr(population, attr):
                        values = getattr(population, attr)
                        if isinstance(values, np.ndarray):
                            pop_details[attr] = values.tolist()
                        elif values is not None:
                            pop_details[attr] = list(values)

                        if attr == "redshifts":
                            required_columns.add("redshift")
                        elif attr == "stellar_masses":
                            required_columns.add("stellar_mass")

                # Store detailed data
                population_data["population_details"][pop_id] = pop_details

            # Update reconstruction info
            population_data["reconstruction_info"]["total_sources"] = total_sources
            population_data["reconstruction_info"]["required_columns"] = list(
                required_columns
            )

            logger.info(
                f"Extracted population data for {len(population_data['populations'])} populations ({total_sources} total sources)"
            )

        except (AttributeError, TypeError, ValueError, KeyError) as e:
            logger.error(f"Failed to extract population data: {e}")
            population_data["extraction_error"] = str(e)

        return population_data

    def save_stacking_results(self, filepath: str | Path) -> None:
        """Save stacking results as fully self-contained JSON"""
        if not self.stacking_results:
            raise SimstackError("No stacking results to save")

        filepath = Path(filepath).with_suffix(".json")

        try:
            # Extract all necessary data
            stacking_data = self._extract_stacking_data()

            # Add comprehensive catalog metadata
            catalog_metadata = self._extract_catalog_metadata()
            stacking_data["catalog_metadata"] = catalog_metadata

            # Add detailed population data
            population_data = self._extract_population_data()
            stacking_data.update(population_data)

            # Embed the full configuration
            if self.config:
                stacking_data["embedded_config"] = self._serialize_config(self.config)

            # Enhanced metadata
            stacking_data["metadata"] = {
                "timestamp": datetime.now().isoformat(),
                "version": "simstack4_json_self_contained_v2",
                "config_path": self.config_path,
                "n_populations": len(self.population_manager)
                if self.population_manager
                else 0,
                "n_maps": len(self.sky_maps) if self.sky_maps else 0,
                "catalog_embedded": catalog_metadata is not None
                and len(catalog_metadata) > 0,
                "population_data_embedded": population_data is not None
                and len(population_data.get("populations", [])) > 0,
                "self_contained": True,  # Flag indicating this file should work standalone
            }

            # Save with enhanced encoder
            with open(filepath, "w") as f:
                json.dump(stacking_data, f, indent=2, cls=EnhancedJSONEncoder)

            # Validate the saved file
            self._validate_saved_json(filepath)

            logger.info(f"✓ Self-contained stacking results saved: {filepath}")
            logger.info(f"  - {stacking_data['metadata']['n_populations']} populations")
            logger.info(f"  - {stacking_data['metadata']['n_maps']} maps")
            logger.info(
                f"  - Catalog metadata: {'✓' if stacking_data['metadata']['catalog_embedded'] else '✗'}"
            )
            logger.info(
                f"  - Population data: {'✓' if stacking_data['metadata']['population_data_embedded'] else '✗'}"
            )

        except (OSError, ValueError, TypeError) as e:
            logger.error(f"Failed to save stacking results: {e}")
            raise SimstackError(f"Save failed: {e}") from e

    def _validate_saved_json(self, filepath: Path) -> None:
        """Validate that saved JSON contains all necessary data for analysis"""
        try:
            with open(filepath) as f:
                data = json.load(f)

            required_keys = [
                "flux_densities",
                "flux_errors",
                "map_names",
                "population_labels",
                "catalog_metadata",
                "populations",
                "population_details",
            ]

            missing_keys = [key for key in required_keys if key not in data]
            if missing_keys:
                logger.warning(
                    f"JSON validation warning - missing keys: {missing_keys}"
                )

            # Check catalog metadata
            catalog_meta = data.get("catalog_metadata", {})
            if not catalog_meta.get("catalog_info", {}).get("n_populations", 0):
                logger.warning("JSON validation warning - no population metadata found")

            # Check population data
            pop_data = data.get("populations", [])
            if not pop_data:
                logger.warning("JSON validation warning - no population data found")

            logger.info("✓ JSON validation completed")

        except (OSError, ValueError) as e:
            logger.warning(f"JSON validation failed: {e}")

    def load_stacking_results(self, filepath: str | Path) -> None:
        """Load stacking results from self-contained JSON"""
        filepath = Path(filepath)

        if not filepath.exists() and filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")

        if not filepath.exists():
            raise SimstackError(f"Stacking results file not found: {filepath}")

        with open(filepath) as f:
            stacking_data = json.load(f)

        # Check if this is a self-contained file
        metadata = stacking_data.get("metadata", {})
        is_self_contained = metadata.get("self_contained", False)

        logger.info(f"Loading stacking results from: {filepath}")
        logger.info(f"File type: {'self-contained' if is_self_contained else 'legacy'}")

        # Reconstruct stacking results object
        self.stacking_results = type("StackingResults", (), {})()

        # Load basic stacking data
        for key, value in stacking_data.items():
            if key not in [
                "metadata",
                "populations",
                "population_details",
                "embedded_config",
                "catalog_metadata",
            ]:
                if isinstance(value, list):
                    setattr(self.stacking_results, key, np.array(value))
                else:
                    setattr(self.stacking_results, key, value)

        # Store embedded data
        self._embedded_config = stacking_data.get("embedded_config")
        self._catalog_metadata = stacking_data.get("catalog_metadata", {})
        self._population_info = stacking_data.get("populations", [])
        self._population_details = stacking_data.get("population_details", {})

        # Try to reconstruct config
        if not self.config:
            if self._embedded_config:
                try:
                    self.config = self._reconstruct_config_from_json()
                    logger.info("✓ Configuration reconstructed from embedded data")
                except (ImportError, AttributeError, TypeError) as e:
                    logger.warning(f"Config reconstruction failed: {e}")
            elif metadata.get("config_path"):
                try:
                    self.config = load_config(metadata["config_path"])
                    logger.info(
                        f"✓ Configuration loaded from: {metadata['config_path']}"
                    )
                except OSError as e:
                    logger.warning(f"Original config not accessible: {e}")

        # Reconstruct population manager from embedded data
        if self._catalog_metadata and self._population_info:
            try:
                self._reconstruct_population_manager()
                logger.info("✓ Population manager reconstructed from embedded data")
            except (ImportError, AttributeError, TypeError, KeyError) as e:
                logger.warning(f"Population manager reconstruction failed: {e}")
                logger.info("Will attempt to load from catalog if config is available")

        # Fallback: load catalog from config if population manager reconstruction failed
        if not self.population_manager and self.config:
            try:
                logger.info("Attempting to load catalog from configuration...")
                self._load_catalog()
            except (OSError, ImportError) as e:
                logger.warning(f"Catalog loading failed: {e}")

        # Final validation
        if not self.population_manager:
            raise SimstackError(
                "Could not reconstruct population manager. "
                "The JSON file may be missing required metadata."
            )

        logger.info(
            f"✓ Successfully loaded stacking results with {len(self.population_manager)} populations"
        )

    def _get_cosmology_enum(self, cosmology_str: str):
        """Convert cosmology string to enum"""
        try:
            from .config import Cosmology

            # Handle both string formats
            cosmology_upper = cosmology_str.upper()
            if cosmology_upper in ["PLANCK18", "Planck18"]:
                return Cosmology.PLANCK18
            elif cosmology_upper in ["PLANCK15", "Planck15"]:
                return Cosmology.PLANCK15
            else:
                logger.warning(
                    f"Unknown cosmology string '{cosmology_str}', defaulting to Planck18"
                )
                return Cosmology.PLANCK18
        except ImportError:
            logger.warning("Could not import Cosmology enum, returning string")
            return cosmology_str

    def _reconstruct_population_manager(self) -> None:
        """Reconstruct population manager from embedded JSON data"""
        if not self._catalog_metadata or not self._population_info:
            raise SimstackError(
                "Insufficient metadata for population manager reconstruction"
            )

        try:
            # Import required classes
            import inspect

            from .populations import PopulationBin, PopulationManager

            # Create population bins from embedded data
            populations = {}

            # Inspect PopulationBin constructor to understand its signature
            try:
                sig = inspect.signature(PopulationBin.__init__)
                logger.debug(f"PopulationBin constructor signature: {sig}")
            except (AttributeError, TypeError) as e:
                logger.debug(f"Could not inspect PopulationBin constructor: {e}")

            for pop_data in self._population_info:
                pop_id = pop_data["population_id"]
                pop_details = self._population_details.get(pop_id, {})

                # Create population bin using the correct dataclass structure
                try:
                    # Extract bin ranges from population ID or use defaults
                    bin_ranges = {}
                    if "redshift_range" in pop_data:
                        redshift_range = pop_data["redshift_range"]
                        if (
                            isinstance(redshift_range, list)
                            and len(redshift_range) >= 2
                        ):
                            bin_ranges["redshift"] = (
                                redshift_range[0],
                                redshift_range[1],
                            )

                    if "mass_range" in pop_data:
                        mass_range = pop_data["mass_range"]
                        if isinstance(mass_range, list) and len(mass_range) >= 2:
                            bin_ranges["stellar_mass"] = (mass_range[0], mass_range[1])

                    # If no ranges found, try to parse from population ID
                    if not bin_ranges:
                        # Parse population ID like "redshift_0.01_0.5__l_uv_9.0_10.0__split_0"
                        id_parts = pop_id.split("__")
                        for part in id_parts:
                            if "_" in part and not part.startswith("split_"):
                                components = part.split("_")
                                if len(components) >= 3:
                                    try:
                                        var_name = components[0]
                                        min_val = float(components[1])
                                        max_val = float(components[2])
                                        bin_ranges[var_name] = (min_val, max_val)
                                    except (ValueError, IndexError):
                                        pass

                    # Extract split info
                    split_value = 0
                    split_label = "split_0"
                    if "split_" in pop_id:
                        try:
                            split_part = [
                                p for p in pop_id.split("__") if p.startswith("split_")
                            ][0]
                            split_value = int(split_part.split("_")[1])
                            split_label = split_part
                        except (IndexError, ValueError):
                            pass

                    # Create the PopulationBin with correct parameters
                    population = PopulationBin(
                        id_label=pop_id,
                        bin_ranges=bin_ranges,
                        split_label=split_label,
                        split_value=split_value,
                        indices=np.array(pop_details.get("source_indices", [])),
                        n_sources=pop_data.get("n_sources", 0),
                    )

                    # Set medians
                    medians = {}
                    if "median_redshift" in pop_data:
                        medians["redshift"] = pop_data["median_redshift"]
                    if "median_stellar_mass" in pop_data:
                        medians["stellar_mass"] = pop_data["median_stellar_mass"]
                    population.medians = medians

                except (TypeError, ValueError) as e:
                    logger.warning(f"Could not create PopulationBin for {pop_id}: {e}")
                    # Create a minimal compatible object
                    population = type(
                        "PopulationBin",
                        (),
                        {
                            "id_label": pop_id,
                            "bin_ranges": {},
                            "split_label": "split_0",
                            "split_value": 0,
                            "indices": np.array(pop_details.get("source_indices", [])),
                            "n_sources": pop_data.get("n_sources", 0),
                            "medians": {
                                "redshift": pop_data.get("median_redshift", 0.0),
                                "stellar_mass": pop_data.get(
                                    "median_stellar_mass", 0.0
                                ),
                            },
                            "population_id": pop_id,  # For legacy compatibility
                            "median_redshift": pop_data.get("median_redshift", 0.0),
                            "median_stellar_mass": pop_data.get(
                                "median_stellar_mass", 0.0
                            ),
                            "redshift_range": pop_data.get(
                                "redshift_range", [0.0, 0.0]
                            ),
                            "mass_range": pop_data.get("mass_range", [0.0, 0.0]),
                        },
                    )()

                # Add coordinates if available
                if "coordinates" in pop_details:
                    coord_data = pop_details["coordinates"]
                    try:
                        import astropy.units as u
                        from astropy.coordinates import SkyCoord

                        population.coordinates = SkyCoord(
                            ra=coord_data["ra"] * u.deg, dec=coord_data["dec"] * u.deg
                        )
                    except ImportError:
                        # Fallback: simple coordinate container
                        population.coordinates = type(
                            "Coordinates",
                            (),
                            {
                                "ra": type(
                                    "RA", (), {"degree": np.array(coord_data["ra"])}
                                )(),
                                "dec": type(
                                    "Dec", (), {"degree": np.array(coord_data["dec"])}
                                )(),
                            },
                        )()

                # Add other attributes for compatibility
                if "redshifts" in pop_details:
                    population.redshifts = np.array(pop_details["redshifts"])

                if "stellar_masses" in pop_details:
                    population.stellar_masses = np.array(pop_details["stellar_masses"])

                # Add legacy properties for backward compatibility
                if not hasattr(population, "population_id"):
                    population.population_id = pop_id
                if not hasattr(population, "median_redshift"):
                    population.median_redshift = pop_data.get("median_redshift", 0.0)
                if not hasattr(population, "median_stellar_mass"):
                    population.median_stellar_mass = pop_data.get(
                        "median_stellar_mass", 0.0
                    )

                populations[pop_id] = population

            # Create population manager with proper config
            self.population_manager = PopulationManager(self.config)

            # Manually set the populations dictionary
            self.population_manager.populations = populations

            # Set up additional manager attributes from metadata
            if self._catalog_metadata.get("binning_config"):
                # Reconstruct bin configs if needed
                try:
                    from .config import BinConfig

                    bin_configs = {}
                    for bin_name, bin_data in self._catalog_metadata[
                        "binning_config"
                    ].items():
                        bin_configs[bin_name] = BinConfig(
                            id=bin_data.get("id", bin_name),
                            label=bin_data.get("label", bin_name),
                            bins=bin_data.get("bins", []),
                            formula_ref=bin_data.get("formula_ref"),
                        )
                    self.population_manager.bin_configs = bin_configs
                except (ImportError, KeyError, TypeError) as e:
                    logger.warning(f"Could not reconstruct bin configs: {e}")

            # Set split labels if available
            unique_splits = set()
            for pop in populations.values():
                if hasattr(pop, "split_value"):
                    unique_splits.add(pop.split_value)
            self.population_manager.split_labels = [
                f"split_{i}" for i in sorted(unique_splits)
            ]

            logger.info(
                f"Reconstructed population manager with {len(populations)} populations"
            )

        except (ImportError, AttributeError, TypeError, KeyError) as e:
            logger.error(f"Population manager reconstruction failed: {e}")
            raise SimstackError(f"Population reconstruction failed: {e}") from e

    def _load_catalog(self) -> None:
        """Load catalog and create population manager"""
        current_config = self._get_working_config()

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

        except (OSError, ImportError) as e:
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
            self.sky_maps = load_maps(self.config.maps)
            logger.info(f"✓ Maps loaded: {len(self.sky_maps)} maps available")

        except (OSError, ImportError) as e:
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
            self.stacking_results = run_stacking(
                config=self.config,
                population_manager=self.population_manager,
                sky_maps=self.sky_maps,
            )

            logger.info("✓ Stacking computation completed!")

        except (RuntimeError, ValueError, ImportError) as e:
            logger.error(f"Stacking pipeline failed: {e}")
            raise SimstackError(f"Stacking failed: {e}") from e

    def _run_analysis(
        self,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        use_schreiber_prior: bool = False,
        save_csv_path: str | None = None,
        use_covariance: bool | None = None,
        correlation_matrix: dict | None = None,
        inflation_factors: dict | None = None,
        bootstrap_covariances: dict | None = None,
    ) -> None:
        """Run ONLY the analysis/SED fitting (requires stacking results)"""
        if not self.stacking_results:
            raise SimstackError("No stacking results available for analysis")

        if not self.population_manager:
            logger.info("Loading catalog for analysis...")
            self._load_catalog()

        # Use instance settings or overrides
        covariance_enabled = (
            use_covariance if use_covariance is not None else self.use_covariance
        )
        correlation_data = (
            correlation_matrix
            if correlation_matrix is not None
            else self.correlation_matrix
        )

        fitting_method = "MCMC" if use_mcmc else "curve_fit"
        prior_type = "Schreiber+2015" if use_schreiber_prior else "flat"
        covariance_status = (
            "with covariance" if covariance_enabled else "diagonal errors"
        )

        logger.info(
            f"Starting analysis pipeline with {fitting_method} fitting, {prior_type} priors, {covariance_status}..."
        )

        try:
            self.processed_results = create_results_processor(
                config=self.config,
                stacking_results=self.stacking_results,
                population_manager=self.population_manager,
                use_mcmc=use_mcmc,
                mcmc_iterations=mcmc_iterations,
                mcmc_burn_in=mcmc_burn_in,
                use_schreiber_prior=use_schreiber_prior,
                use_covariance=covariance_enabled,  # NEW
                correlation_matrix=correlation_data,  # NEW
                inflation_factors=inflation_factors,  # NEW
                bootstrap_covariances=bootstrap_covariances,  # NEW
            )

            # Update results dict for backward compatibility
            self.results_dict = {
                "band_results_dict": self.processed_results.band_results,
                "population_summary": self.processed_results.get_population_summary(),
                "stacking_results": self.stacking_results,
                "processed_results": self.processed_results,
                "covariance_used": covariance_enabled,  # NEW
            }

            if save_csv_path is not None:
                self._save_results_to_csv(save_csv_path)

            logger.info("✓ Analysis completed successfully!")
            if covariance_enabled:
                logger.info("✓ Covariance matrix was used in fitting")

        except (RuntimeError, ValueError, ImportError) as e:
            logger.error(f"Analysis pipeline failed: {e}")
            raise SimstackError(f"Analysis failed: {e}") from e

    def _serialize_config(self, config: SimstackConfig) -> dict:
        """Enhanced config serialization that handles custom objects safely - NON-RECURSIVE VERSION"""

        def safe_serialize(obj, depth=0, max_depth=10):
            """Non-recursive serialization with depth limiting"""
            if depth > max_depth:
                return f"<max_depth_reached_{type(obj).__name__}>"

            if obj is None:
                return None
            elif isinstance(obj, str | int | float | bool):
                return obj
            elif isinstance(obj, list | tuple):
                return [safe_serialize(item, depth + 1, max_depth) for item in obj]
            elif isinstance(obj, dict):
                return {
                    str(k): safe_serialize(v, depth + 1, max_depth)
                    for k, v in obj.items()
                }
            elif hasattr(obj, "name") and isinstance(obj.name, str):
                return str(obj.name)
            elif hasattr(obj, "value"):
                return obj.value
            elif hasattr(obj, "__dict__"):
                # Limit attribute serialization to avoid recursion
                result = {}
                for k, v in obj.__dict__.items():
                    if not k.startswith("_"):  # Skip private attributes
                        try:
                            result[str(k)] = safe_serialize(v, depth + 1, max_depth)
                        except (RecursionError, TypeError, AttributeError):
                            result[str(k)] = str(type(v).__name__)
                return result
            else:
                # Convert everything else to string representation
                try:
                    return str(obj)
                except (TypeError, AttributeError):
                    return f"<{type(obj).__name__}>"

        config_dict = {}

        try:
            # Manually serialize specific config sections to avoid recursion
            logger.info("Serializing configuration sections...")

            # Handle catalog config
            if hasattr(config, "catalog") and config.catalog:
                logger.debug("Serializing catalog config...")
                catalog_dict = {}

                # Basic catalog info
                catalog_dict["path"] = getattr(config.catalog, "path", "")
                catalog_dict["file"] = getattr(config.catalog, "file", "")

                # Astrometry
                if hasattr(config.catalog, "astrometry"):
                    catalog_dict["astrometry"] = dict(config.catalog.astrometry)

                # Classification
                if hasattr(config.catalog, "classification"):
                    classification_dict = {}
                    classification = config.catalog.classification

                    # Split type and params
                    if hasattr(classification, "split_type"):
                        classification_dict["split_type"] = (
                            str(classification.split_type)
                            if classification.split_type
                            else None
                        )

                    if hasattr(classification, "split_params"):
                        classification_dict["split_params"] = safe_serialize(
                            classification.split_params, 0, 5
                        )

                    # Binning configuration
                    if hasattr(classification, "binning"):
                        binning_dict = {}
                        for bin_name, bin_config in classification.binning.items():
                            binning_dict[bin_name] = {
                                "id": getattr(bin_config, "id", bin_name),
                                "label": getattr(bin_config, "label", bin_name),
                                "bins": getattr(bin_config, "bins", []),
                                "formula_ref": getattr(bin_config, "formula_ref", None),
                            }
                        classification_dict["binning"] = binning_dict

                    # Formulas
                    if hasattr(classification, "formulas"):
                        classification_dict["formulas"] = safe_serialize(
                            classification.formulas, 0, 3
                        )

                    catalog_dict["classification"] = classification_dict

                config_dict["catalog"] = catalog_dict

            # Handle maps config
            if hasattr(config, "maps") and config.maps:
                logger.debug("Serializing maps config...")
                maps_dict = {}
                for map_name, map_config in config.maps.items():
                    maps_dict[map_name] = {
                        "wavelength": getattr(map_config, "wavelength", 0),
                        "path_map": getattr(map_config, "path_map", ""),
                        "path_noise": getattr(map_config, "path_noise", ""),
                        "color_correction": getattr(
                            map_config, "color_correction", 1.0
                        ),
                        "beam": {
                            "fwhm": getattr(map_config.beam, "fwhm", 0)
                            if hasattr(map_config, "beam")
                            else 0,
                            "area_sr": getattr(map_config.beam, "area_sr", 1.0)
                            if hasattr(map_config, "beam")
                            else 1.0,
                        },
                    }
                config_dict["maps"] = maps_dict

            # Handle other simple sections
            simple_attrs = ["cosmology", "output", "binning", "error_estimator"]
            for attr in simple_attrs:
                if hasattr(config, attr):
                    logger.debug(f"Serializing {attr} config...")
                    try:
                        config_dict[attr] = safe_serialize(getattr(config, attr), 0, 5)
                    except (RecursionError, TypeError, AttributeError) as e:
                        logger.warning(f"Failed to serialize {attr}: {e}")
                        config_dict[attr] = f"<serialization_failed: {e}>"

            logger.info("✓ Configuration serialization completed")

        except (AttributeError, TypeError, ValueError, RecursionError) as e:
            logger.warning(f"Config serialization failed: {e}")
            # Minimal fallback config
            config_dict = {
                "serialization_error": str(e),
                "config_type": str(type(config).__name__),
                "fallback_data": {
                    "cosmology": getattr(config, "cosmology", "Planck18")
                    if hasattr(config, "cosmology")
                    else "Planck18",
                    "catalog_file": getattr(config.catalog, "file", "")
                    if hasattr(config, "catalog")
                    else "",
                    "n_maps": len(config.maps) if hasattr(config, "maps") else 0,
                },
            }

        return config_dict

    def _reconstruct_config_from_json(self) -> SimstackConfig:
        """Reconstruct config from embedded JSON data"""
        if not self._embedded_config:
            raise SimstackError("No embedded config data available")

        try:
            from .config import (
                BeamConfig,
                BinConfig,
                BootstrapConfig,
                CatalogConfig,
                ClassificationConfig,
                ErrorConfig,
                MapConfig,
                OutputConfig,
                SimstackConfig,
            )

            # Reconstruct catalog config
            catalog_data = self._embedded_config.get("catalog", {})

            # Astrometry
            astrometry_data = catalog_data.get("astrometry", {})
            astrometry_config = astrometry_data

            # Classification with binning
            classification_data = catalog_data.get("classification", {})

            # Binning configurations
            binning_config = {}
            if "binning" in classification_data:
                binning_dict = classification_data["binning"]
                for bin_name, bin_config_dict in binning_dict.items():
                    if not isinstance(bin_config_dict, dict):
                        continue

                    bin_config = BinConfig(
                        id=bin_config_dict.get("id", bin_name),
                        label=bin_config_dict.get(
                            "label", bin_name.replace("_", " ").title()
                        ),
                        bins=bin_config_dict.get("bins", []),
                        formula_ref=bin_config_dict.get("formula_ref"),
                    )
                    binning_config[bin_name] = bin_config

            # Classification config
            classification_config = ClassificationConfig(
                split_type=classification_data.get("split_type"),
                binning=binning_config,
                split_params=classification_data.get("split_params"),
                formulas=classification_data.get("formulas"),
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
            binning_config_dict = {
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
            )

            error_estimator_config = ErrorConfig(
                write_simmaps=error_est_data.get("write_simmaps", False),
                randomize=error_est_data.get("randomize", False),
                bootstrap=bootstrap_config,
            )

            # Create the full config
            config = SimstackConfig(
                binning=binning_config_dict,
                error_estimator=error_estimator_config,
                cosmology=self._get_cosmology_enum(
                    self._embedded_config.get("cosmology", "Planck18")
                ),
                output=output_config,
                catalog=catalog_config,
                maps=maps_config,
            )

            return config

        except (ImportError, AttributeError, TypeError, KeyError) as e:
            logger.error(f"Failed to reconstruct config from JSON: {e}")
            raise SimstackError(f"Config reconstruction failed: {e}") from e

    def _extract_stacking_data(self) -> dict:
        """Enhanced stacking data extraction"""
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

        return data

    # Public interface methods
    def run_analysis_only(
        self,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        use_schreiber_prior: bool = False,
        save_csv_path: str | None = None,
        use_covariance: bool | None = None,
        correlation_matrix: dict | None = None,
        inflation_factors: dict | None = None,
        use_bootstrap_covariance: bool = True,
        use_instrumental_covariance: bool = True,
    ) -> SimstackResults:
        """Public method to run ONLY analysis (requires stacking results)"""
        if not self.population_manager and self.config:
            self._load_catalog()

        # Handle covariance settings
        if not use_covariance:
            # Override: no covariance at all
            use_bootstrap_covariance = False
            use_instrumental_covariance = False
            logger.info("Covariance disabled - using diagonal errors only")

        # Estimate bootstrap covariance if requested and available
        bootstrap_covariances = {}
        if use_bootstrap_covariance:
            bootstrap_covariances = self.estimate_bootstrap_covariance()
            if bootstrap_covariances:
                logger.info(
                    f"Will use bootstrap covariance for {len(bootstrap_covariances)} populations"
                )
            else:
                logger.info("No bootstrap covariance available")

        # Set up instrumental correlation matrix
        instrumental_correlation = None
        if use_instrumental_covariance:
            instrumental_correlation = (
                correlation_matrix
                if correlation_matrix is not None
                else self.correlation_matrix
            )
            if instrumental_correlation:
                logger.info("Will use instrumental correlation matrix")
            else:
                logger.info("No instrumental correlation matrix available")

        # Log what we're using
        covariance_components = []
        if use_instrumental_covariance and instrumental_correlation:
            covariance_components.append("instrumental")
        if use_bootstrap_covariance and bootstrap_covariances:
            covariance_components.append("bootstrap")

        self._run_analysis(
            use_mcmc=use_mcmc,
            mcmc_iterations=mcmc_iterations,
            mcmc_burn_in=mcmc_burn_in,
            use_schreiber_prior=use_schreiber_prior,
            save_csv_path=save_csv_path,
            use_covariance=use_covariance,
            correlation_matrix=instrumental_correlation,
            inflation_factors=inflation_factors,
            bootstrap_covariances=bootstrap_covariances,
        )
        return self.processed_results

    def run_complete_pipeline(
        self,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        use_schreiber_prior: bool = False,
        save_path: str | None = None,
    ) -> SimstackResults:
        """Run both stacking and analysis"""
        self._run_stacking()
        self._run_analysis(
            use_mcmc=use_mcmc,
            mcmc_iterations=mcmc_iterations,
            mcmc_burn_in=mcmc_burn_in,
            use_schreiber_prior=use_schreiber_prior,
            save_csv_path=save_path,
        )
        return self.processed_results

    def _save_results_to_csv(self, csv_output_path: str | None = None) -> None:
        """Save analysis results to CSV file(s)"""
        try:
            if csv_output_path is None:
                if hasattr(self.config, "output") and hasattr(
                    self.config.output, "results_dir"
                ):
                    output_dir = Path(self.config.output.results_dir)
                else:
                    output_dir = Path(".")

                base_name = getattr(self.config.output, "shortname", "simstack_results")
                csv_output_path = output_dir / f"{base_name}_results.csv"
            else:
                csv_output_path = Path(csv_output_path)

            csv_output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save main population summary
            summary_df = self.processed_results.get_population_summary()
            summary_path = csv_output_path.with_suffix(".csv")
            summary_df.to_csv(summary_path, index=False)
            logger.info(f"✓ Population summary saved to: {summary_path}")

            # Save individual SEDs
            sed_dir = csv_output_path.parent / f"{csv_output_path.stem}_seds"
            sed_dir.mkdir(exist_ok=True)

            for pop_id in self.processed_results.sed_results.keys():
                try:
                    sed_df = self.processed_results.get_sed_table(pop_id)
                    safe_pop_id = (
                        pop_id.replace("__", "_").replace(".", "p").replace("/", "_")
                    )
                    sed_path = sed_dir / f"sed_{safe_pop_id}.csv"
                    sed_df.to_csv(sed_path, index=False)
                    logger.info(f"✓ SED for {pop_id} saved to: {sed_path}")
                except (AttributeError, TypeError, ValueError) as e:
                    logger.warning(f"Failed to save SED for population {pop_id}: {e}")

            logger.info(f"✓ All CSV files saved to: {csv_output_path.parent}")

        except (OSError, PermissionError) as e:
            logger.error(f"Failed to save CSV files: {e}")
            raise SimstackError(f"CSV export failed: {e}") from e

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
        """Return number of populations"""
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
    config_path: str | Path, save_path: str | Path | None = None
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
    save_path: str | Path | None = None,
    save_csv_path: str | Path | None = None,
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
        save_csv_path: Path to save CSV results (optional)
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
        save_csv_path=save_csv_path,
    )

    if save_path:
        wrapper._save_analysis_results(save_path)

    return results


def validate_json_self_contained(filepath: str | Path) -> dict:
    """
    Validate that a JSON file is self-contained and ready for analysis

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with validation results
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return {"valid": False, "error": f"File not found: {filepath}"}

    try:
        with open(filepath) as f:
            data = json.load(f)

        validation = {
            "valid": True,
            "version": data.get("metadata", {}).get("version", "unknown"),
            "self_contained": data.get("metadata", {}).get("self_contained", False),
            "n_populations": len(data.get("populations", [])),
            "n_maps": len(data.get("map_names", [])),
            "has_catalog_metadata": "catalog_metadata" in data,
            "has_population_data": "population_details" in data,
            "has_embedded_config": "embedded_config" in data,
            "missing_keys": [],
        }

        # Check required keys
        required_keys = [
            "flux_densities",
            "flux_errors",
            "map_names",
            "population_labels",
        ]

        for key in required_keys:
            if key not in data:
                validation["missing_keys"].append(key)

        if validation["missing_keys"]:
            validation["valid"] = False

        # Check catalog metadata quality
        catalog_meta = data.get("catalog_metadata", {})
        if catalog_meta:
            validation["catalog_info"] = {
                "n_total_sources": catalog_meta.get("catalog_info", {}).get(
                    "n_total_sources", 0
                ),
                "has_sample_data": "catalog_sample" in catalog_meta,
                "has_population_stats": "population_stats" in catalog_meta,
            }

        return validation

    except (OSError, ValueError) as e:
        return {"valid": False, "error": str(e)}
