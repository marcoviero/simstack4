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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder for numpy arrays"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


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
        if not self.config or not self.config.catalog:
            raise SimstackError("No catalog configuration provided")

        logger.info("Loading catalog...")

        try:
            # Load catalog using SkyCatalogs
            self.sky_catalogs = SkyCatalogs(
                catalog_config=self.config.catalog,
                backend="auto",
                full_config=self.config,
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

    def _run_analysis(self) -> None:
        """Run ONLY the analysis/SED fitting (requires stacking results)"""
        if not self.stacking_results:
            raise SimstackError("No stacking results available for analysis")

        if not self.population_manager:
            logger.info("Loading catalog for analysis...")
            self._load_catalog()

        logger.info("Starting analysis pipeline...")

        try:
            # Process results - SED fitting, error estimation, etc.
            self.processed_results = create_results_processor(
                config=self.config,
                stacking_results=self.stacking_results,
                population_manager=self.population_manager,
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
        """Save raw stacking results as JSON"""
        if not self.stacking_results:
            raise SimstackError("No stacking results to save")

        filepath = Path(filepath).with_suffix(".json")

        # Extract stacking data as simple types
        stacking_data = self._extract_stacking_data()

        # Add metadata
        stacking_data["metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "version": "simstack4_json",
            "config_path": self.config_path,
            "n_populations": len(self.population_manager)
            if self.population_manager
            else 0,
            "n_maps": len(self.sky_maps) if self.sky_maps else 0,
        }

        with open(filepath, "w") as f:
            json.dump(stacking_data, f, indent=2, cls=NumpyEncoder)

        logger.info(f"✓ Stacking results saved as JSON: {filepath}")

    def _extract_stacking_data(self) -> dict:
        """Extract stacking results as JSON-serializable data"""
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

        # Extract population info (minimal)
        """
        if self.population_manager:
            data["populations"] = []
            for i, pop in enumerate(self.population_manager):
                pop_data = {
                    "index": i,
                    "population_id": getattr(pop, 'population_id', f"pop_{i}"),
                    "n_sources": getattr(pop, 'n_sources', 0),
                }
                data["populations"].append(pop_data)
        """

        return data

    def load_stacking_results(self, filepath: str | Path) -> None:
        """Load stacking results from JSON"""
        filepath = Path(filepath)

        # Try .json extension if original doesn't exist
        if not filepath.exists() and filepath.suffix != ".json":
            filepath = filepath.with_suffix(".json")

        if not filepath.exists():
            raise SimstackError(f"Stacking results file not found: {filepath}")

        with open(filepath) as f:
            stacking_data = json.load(f)

        # Reconstruct stacking results object
        self.stacking_results = type("StackingResults", (), {})()

        for key, value in stacking_data.items():
            if key not in ["metadata", "populations"]:
                if isinstance(value, list):
                    # Convert back to numpy arrays
                    setattr(self.stacking_results, key, np.array(value))
                else:
                    setattr(self.stacking_results, key, value)

        # Store population info (we'll reload the full catalog for analysis)
        self._population_info = stacking_data.get("populations", [])

        # Load config if we don't have one
        if not self.config and stacking_data["metadata"].get("config_path"):
            config_path = stacking_data["metadata"]["config_path"]
            logger.info(f"Loading config from: {config_path}")
            try:
                self.config = load_config(config_path)
            except Exception:
                logger.warning(f"Config file {config_path} not found")

        logger.info(f"✓ Stacking results loaded from JSON: {filepath}")
        logger.info(f"  - {len(self._population_info)} populations")

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

    def run_analysis_only(self) -> SimstackResults:
        """Public method to run ONLY analysis (requires stacking results)"""
        # Reload catalog if needed for analysis
        if not self.population_manager and self.config:
            self._load_catalog()

        self._run_analysis()
        return self.processed_results

    def run_complete_pipeline(self) -> SimstackResults:
        """Run both stacking and analysis"""
        self._run_stacking()
        self._run_analysis()
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
    stacking_results_path: str | Path, save_path: str | Path = None
) -> SimstackResults:
    """
    Run analysis from self-contained JSON file (no config needed!)

    Args:
        stacking_results_path: Path to saved stacking results JSON
        save_path: Path to save analysis results (optional)

    Returns:
        Processed results object
    """
    wrapper = SimstackWrapper()  # No config needed!
    wrapper.load_stacking_results(stacking_results_path)
    results = wrapper.run_analysis_only()

    if save_path:
        wrapper.save_analysis_results(save_path)

    return results
