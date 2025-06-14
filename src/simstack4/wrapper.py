"""
Complete SimstackWrapper implementation for Simstack4

Replace your current wrapper.py with this complete version
"""

from pathlib import Path
from typing import Any

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
    ):
        """
        Initialize SimstackWrapper

        Args:
            config: Configuration object, file path, or None
            read_maps: Whether to load maps immediately
            read_catalog: Whether to load catalog immediately
            stack_automatically: Whether to run stacking immediately
        """

        # Handle config loading
        if isinstance(config, str | Path):
            self.config = load_config(config)
        else:
            self.config = config

        # Store initialization options
        self.read_maps = read_maps
        self.read_catalog = read_catalog
        self.stack_automatically = stack_automatically

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
        """Run the complete stacking pipeline"""
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
            # Run stacking algorithm
            self.stacking_results = run_stacking(
                config=self.config,
                population_manager=self.population_manager,
                sky_maps=self.sky_maps,
            )

            # Process results
            self.processed_results = create_results_processor(
                config=self.config,
                stacking_results=self.stacking_results,
                population_manager=self.population_manager,
            )

            # Save results
            output_path = (
                Path(self.config.output.folder)
                / f"{self.config.output.shortname}_results.pkl"
            )
            self.processed_results.save_results(output_path)

            # Update results dict for backward compatibility
            self.results_dict = {
                "band_results_dict": self.processed_results.band_results,
                "population_summary": self.processed_results.get_population_summary(),
                "stacking_results": self.stacking_results,
                "processed_results": self.processed_results,
            }

            logger.info("✓ Stacking completed successfully!")
            logger.info(f"Results saved to: {output_path}")

        except Exception as e:
            logger.error(f"Stacking pipeline failed: {e}")
            raise SimstackError(f"Stacking failed: {e}") from e

    def load_catalog(self) -> None:
        """Public method to load catalog"""
        self._load_catalog()

    def load_maps(self) -> None:
        """Public method to load maps"""
        self._load_maps()

    def run_stacking(self) -> SimstackResults:
        """Public method to run stacking"""
        self._run_stacking()
        return self.processed_results

    def get_summary(self) -> dict[str, Any]:
        """Get summary of current state"""
        summary = {
            "config_loaded": self.config is not None,
            "catalog_loaded": self.sky_catalogs is not None,
            "maps_loaded": self.sky_maps is not None,
            "populations_created": self.population_manager is not None,
            "stacking_completed": self.stacking_results is not None,
            "results_processed": self.processed_results is not None,
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

        status_str = ", ".join(status) if status else "not initialized"
        return f"SimstackWrapper({status_str})"


# Convenience functions for backward compatibility
def create_simstack_wrapper(
    config_path: str | Path, load_all: bool = True
) -> SimstackWrapper:
    """
    Convenience function to create and initialize wrapper

    Args:
        config_path: Path to configuration file
        load_all: Whether to load all components immediately

    Returns:
        Initialized SimstackWrapper
    """
    return SimstackWrapper(
        config=config_path,
        read_maps=load_all,
        read_catalog=load_all,
        stack_automatically=False,
    )


def run_complete_pipeline(config_path: str | Path) -> SimstackResults:
    """
    Run complete stacking pipeline from config file

    Args:
        config_path: Path to configuration file

    Returns:
        Processed results object
    """
    wrapper = SimstackWrapper(
        config=config_path, read_maps=True, read_catalog=True, stack_automatically=True
    )

    return wrapper.processed_results
