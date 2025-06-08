"""
Improved population management system for Simstack4

This replaces the jpop, lpop, mpop loop system from simstack3 with a more
flexible and efficient population management system.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import ClassificationConfig, SplitType
from .exceptions.simstack_exceptions import PopulationError


@dataclass
class PopulationBin:
    """Represents a single population bin with its properties"""

    id_label: str
    redshift_range: tuple[float, float]
    stellar_mass_range: tuple[float, float]
    split_label: str
    split_value: int
    indices: np.ndarray = field(default_factory=lambda: np.array([]))
    n_sources: int = 0
    median_redshift: float = 0.0
    median_stellar_mass: float = 0.0

    def __post_init__(self):
        self.n_sources = len(self.indices)


class PopulationManager:
    """
    Manages populations and binning for stacking analysis

    This class replaces the nested loop structure (jpop, lpop, mpop) from simstack3
    with a more flexible system that can handle arbitrary numbers of populations
    and mass bins.
    """

    def __init__(self, classification_config: ClassificationConfig):
        self.config = classification_config
        self.populations: dict[str, PopulationBin] = {}
        self.redshift_bins = self._create_bins(classification_config.redshift.bins)
        self.stellar_mass_bins = self._create_bins(
            classification_config.stellar_mass.bins
        )
        self.split_labels: list[str] = []

    def _create_bins(self, bin_edges: list[float]) -> list[tuple[float, float]]:
        """Create bin tuples from bin edges"""
        return [(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]

    def _validate_catalog_columns(self, catalog_df: pd.DataFrame) -> None:
        """Validate that catalog has required columns"""
        required_cols = [self.config.redshift.id, self.config.stellar_mass.id]

        if self.config.split_params:
            if self.config.split_type in [SplitType.UVJ, SplitType.NUVRJ]:
                required_cols.extend(self.config.split_params.bins.values())

        missing_cols = [col for col in required_cols if col not in catalog_df.columns]
        if missing_cols:
            raise PopulationError(f"Missing required columns: {missing_cols}")

    def _classify_by_labels(self, catalog_df: pd.DataFrame) -> np.ndarray:
        """Classify sources using predefined labels"""
        # For labels method, assume split_params.id contains the column with labels
        if not self.config.split_params:
            # No splitting, all sources get label 0
            return np.zeros(len(catalog_df), dtype=int)

        label_col = self.config.split_params.id
        if label_col not in catalog_df.columns:
            raise PopulationError(f"Label column '{label_col}' not found in catalog")

        return catalog_df[label_col].values

    def _classify_by_uvj(self, catalog_df: pd.DataFrame) -> np.ndarray:
        """Classify sources using UVJ criteria"""
        if not self.config.split_params:
            raise PopulationError("split_params required for UVJ classification")

        uv_col = self.config.split_params.bins.get("U-V")
        vj_col = self.config.split_params.bins.get("V-J")

        if not uv_col or not vj_col:
            raise PopulationError("U-V and V-J columns required for UVJ classification")

        uv = catalog_df[uv_col].values
        vj = catalog_df[vj_col].values

        # UVJ quiescent criteria (Whitaker et al. 2011)
        # Quiescent: U-V > 1.3 and V-J < 1.6 and U-V > 0.88 * V-J + 0.59
        quiescent = (uv > 1.3) & (vj < 1.6) & (uv > 0.88 * vj + 0.59)

        # 0 = star-forming, 1 = quiescent
        return quiescent.astype(int)

    def _classify_by_nuvrj(self, catalog_df: pd.DataFrame) -> np.ndarray:
        """Classify sources using NUVRJ criteria"""
        if not self.config.split_params:
            raise PopulationError("split_params required for NUVRJ classification")

        # Similar to UVJ but with NUV-R and R-J
        # Implementation would depend on specific criteria
        # Placeholder for now
        return np.zeros(len(catalog_df), dtype=int)

    def _create_populations(
        self,
        catalog_df: pd.DataFrame,
        z_col: str,
        mass_col: str,
        split_values: np.ndarray,
    ) -> None:
        """Create population bins for all combinations"""
        z_vals = catalog_df[z_col].values
        mass_vals = catalog_df[mass_col].values

        # Get unique split values and create labels
        unique_splits = np.unique(split_values)
        self.split_labels = [f"split_{i}" for i in unique_splits]

        # Create all combinations of bins
        for z_min, z_max in self.redshift_bins:
            for mass_min, mass_max in self.stellar_mass_bins:
                for split_val in unique_splits:
                    # Find sources in this bin
                    mask = (
                        (z_vals >= z_min)
                        & (z_vals < z_max)
                        & (mass_vals >= mass_min)
                        & (mass_vals < mass_max)
                        & (split_values == split_val)
                    )

                    indices = np.where(mask)[0]

                    if len(indices) > 0:  # Only create bin if it has sources
                        # Create ID label
                        z_label = f"redshift_{z_min:.2f}_{z_max:.2f}"
                        mass_label = f"stellar_mass_{mass_min:.1f}_{mass_max:.1f}"
                        split_label = f"split_{split_val}"
                        id_label = f"{z_label}__{mass_label}__{split_label}"

                        # Calculate medians
                        median_z = np.median(z_vals[indices])
                        median_mass = np.median(mass_vals[indices])

                        # Create population bin
                        pop_bin = PopulationBin(
                            id_label=id_label,
                            redshift_range=(z_min, z_max),
                            stellar_mass_range=(mass_min, mass_max),
                            split_label=split_label,
                            split_value=split_val,
                            indices=indices,
                            median_redshift=median_z,
                            median_stellar_mass=median_mass,
                        )

                        self.populations[id_label] = pop_bin

    def get_population_summary(self) -> pd.DataFrame:
        """Get summary statistics of all populations"""
        data = []
        for pop_bin in self.populations.values():
            data.append(
                {
                    "id_label": pop_bin.id_label,
                    "n_sources": pop_bin.n_sources,
                    "z_range": f"{pop_bin.redshift_range[0]:.2f}-{pop_bin.redshift_range[1]:.2f}",
                    "mass_range": f"{pop_bin.stellar_mass_range[0]:.1f}-{pop_bin.stellar_mass_range[1]:.1f}",
                    "split_label": pop_bin.split_label,
                    "median_z": pop_bin.median_redshift,
                    "median_mass": pop_bin.median_stellar_mass,
                }
            )

        return pd.DataFrame(data)

    def get_population_indices(self, id_label: str) -> np.ndarray:
        """Get catalog indices for a specific population"""
        if id_label not in self.populations:
            raise PopulationError(f"Population '{id_label}' not found")
        return self.populations[id_label].indices

    def iter_populations(self) -> Iterator[PopulationBin]:
        """Iterate over all population bins"""
        yield from self.populations.values()

    def get_bootstrap_populations(
        self, bootstrap_fraction: float = 0.8, seed: int | None = None
    ) -> "PopulationManager":
        """
        Create bootstrap version of populations

        Args:
            bootstrap_fraction: Fraction of sources to use in bootstrap sample
            seed: Random seed for reproducibility

        Returns:
            New PopulationManager with bootstrap populations
        """
        if seed is not None:
            np.random.seed(seed)

        bootstrap_manager = PopulationManager(self.config)
        bootstrap_manager.redshift_bins = self.redshift_bins
        bootstrap_manager.stellar_mass_bins = self.stellar_mass_bins
        bootstrap_manager.split_labels = self.split_labels

        # CRITICAL FIX: Copy catalog data reference
        bootstrap_manager.catalog_df = self.catalog_df
        bootstrap_manager.config = self.config

        for id_label, pop_bin in self.populations.items():
            n_bootstrap = int(pop_bin.n_sources * bootstrap_fraction)

            if n_bootstrap > 0:
                # Randomly select indices
                bootstrap_indices = np.random.choice(
                    pop_bin.indices, size=n_bootstrap, replace=False
                )

                # Create bootstrap population bin
                bootstrap_bin = PopulationBin(
                    id_label=f"{id_label}__bootstrap",
                    redshift_range=pop_bin.redshift_range,
                    stellar_mass_range=pop_bin.stellar_mass_range,
                    split_label=pop_bin.split_label,
                    split_value=pop_bin.split_value,
                    indices=bootstrap_indices,
                    median_redshift=pop_bin.median_redshift,
                    median_stellar_mass=pop_bin.median_stellar_mass,
                )

                bootstrap_manager.populations[bootstrap_bin.id_label] = bootstrap_bin

        return bootstrap_manager

    """
    Add this method to your PopulationManager class in populations.py

    Insert this method inside the PopulationManager class, after the existing methods
    """

    def get_population_data(self, population_id: str) -> dict[str, np.ndarray]:
        """
        Get catalog data for a specific population

        Args:
            population_id: Population identifier

        Returns:
            Dictionary with ra, dec, redshift, stellar_mass, and indices arrays

        Raises:
            PopulationError: If population not found or no catalog loaded
        """
        if population_id not in self.populations:
            raise PopulationError(f"Population {population_id} not found")

        pop_bin = self.populations[population_id]
        indices = pop_bin.indices

        if not hasattr(self, "catalog_df") or self.catalog_df is None:
            raise PopulationError("No catalog data loaded")

        # Get column names from config
        ra_col = None
        dec_col = None
        z_col = None
        mass_col = None

        # Try to get column names from the config
        # This assumes the catalog_df was loaded with the classification config
        if hasattr(self, "config"):
            if hasattr(self.config, "astrometry"):
                ra_col = self.config.astrometry.get("ra", "ra")
                dec_col = self.config.astrometry.get("dec", "dec")
            z_col = (
                self.config.redshift.id
                if hasattr(self.config, "redshift")
                else "z_peak"
            )
            mass_col = (
                self.config.stellar_mass.id
                if hasattr(self.config, "stellar_mass")
                else "lmass"
            )

        # Fallback to common column names if config not available
        if not ra_col:
            ra_col = "ra"
        if not dec_col:
            dec_col = "dec"
        if not z_col:
            z_col = "z_peak"
        if not mass_col:
            mass_col = "lmass"

        # Extract data using the indices
        try:
            if hasattr(self.catalog_df, "iloc"):  # pandas DataFrame
                subset = self.catalog_df.iloc[indices]

                data = {
                    "ra": subset[ra_col].values,
                    "dec": subset[dec_col].values,
                    "redshift": subset[z_col].values,
                    "stellar_mass": subset[mass_col].values,
                    "indices": indices,
                }
            else:
                # Handle other DataFrame types (like polars)
                subset = self.catalog_df[indices]
                data = {
                    "ra": subset[ra_col].to_numpy(),
                    "dec": subset[dec_col].to_numpy(),
                    "redshift": subset[z_col].to_numpy(),
                    "stellar_mass": subset[mass_col].to_numpy(),
                    "indices": indices,
                }

        except KeyError as e:
            raise PopulationError(f"Required column not found in catalog: {e}") from e
        except Exception as e:
            raise PopulationError(f"Error extracting population data: {e}") from e

        return data

    # Also add this method to store the catalog_df reference
    def set_catalog_data(self, catalog_df, config=None):
        """
        Store reference to catalog DataFrame for population data extraction

        Args:
            catalog_df: The loaded catalog DataFrame
            config: Optional config object with column mappings
        """
        self.catalog_df = catalog_df
        if config:
            self.config = config

    # And modify the classify_catalog method to store the reference
    def classify_catalog(self, catalog_df: pd.DataFrame) -> None:
        """
        Classify catalog sources into populations based on configuration
        UPDATED to store catalog reference

        Args:
            catalog_df: Catalog dataframe with required columns
        """
        # Store reference to catalog data
        self.catalog_df = catalog_df

        # Validate required columns
        self._validate_catalog_columns(catalog_df)

        # Get column names
        z_col = self.config.redshift.id
        mass_col = self.config.stellar_mass.id

        # Classify sources based on split type
        if self.config.split_type == SplitType.LABELS:
            split_values = self._classify_by_labels(catalog_df)
        elif self.config.split_type == SplitType.UVJ:
            split_values = self._classify_by_uvj(catalog_df)
        elif self.config.split_type == SplitType.NUVRJ:
            split_values = self._classify_by_nuvrj(catalog_df)
        else:
            raise PopulationError(f"Unknown split type: {self.config.split_type}")

        # Create populations for all combinations of bins
        self._create_populations(catalog_df, z_col, mass_col, split_values)

    def __len__(self) -> int:
        """Return number of populations"""
        return len(self.populations)

    def __iter__(self) -> Iterator[str]:
        """Iterate over population ID labels"""
        return iter(self.populations.keys())
