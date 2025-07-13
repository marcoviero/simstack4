"""
Improved population management system for Simstack4 (GENERALIZED VERSION)

This replaces the jpop, lpop, mpop loop system from simstack3 with a more
flexible and efficient population management system that supports arbitrary
binning dimensions (redshift, stellar mass, beta slope, L_UV, etc.).
"""
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from .config import SplitType
from .exceptions.simstack_exceptions import PopulationError
from .utils import setup_logging

logger = setup_logging()


@dataclass
class PopulationBin:
    """Represents a single population bin with its properties (GENERALIZED)"""

    id_label: str
    bin_ranges: dict[
        str, tuple[float, float]
    ]  # e.g., {"redshift": (0.5, 1.0), "beta": (-2.5, -2.0)}
    split_label: str
    split_value: int
    indices: np.ndarray = field(default_factory=lambda: np.array([]))
    n_sources: int = 0
    medians: dict[str, float] = field(
        default_factory=dict
    )  # Median values for each binned variable

    def __post_init__(self):
        self.n_sources = len(self.indices)

    # Legacy properties for backward compatibility
    @property
    def redshift_range(self) -> tuple[float, float]:
        """Legacy property - returns redshift range if available"""
        return self.bin_ranges.get("redshift", (0.0, 0.0))

    @property
    def stellar_mass_range(self) -> tuple[float, float]:
        """Legacy property - returns stellar mass range if available"""
        return self.bin_ranges.get("stellar_mass", (0.0, 0.0))

    @property
    def median_redshift(self) -> float:
        """Legacy property - returns median redshift if available"""
        return self.medians.get("redshift", 0.0)

    @property
    def median_stellar_mass(self) -> float:
        """Legacy property - returns median stellar mass if available"""
        return self.medians.get("stellar_mass", 0.0)


class PopulationManager:
    """
    Manages populations and binning for stacking analysis (GENERALIZED)

    This class handles arbitrary numbers of binning dimensions and population
    types, replacing the fixed redshift/mass structure from the original version.
    """

    def __init__(self, full_config, classification_config=None):
        """
        Initialize population manager

        Args:
            full_config: Complete SimstackConfig object (for astrometry access)
            classification_config: Legacy parameter for backward compatibility
        """
        # Handle both new and legacy initialization
        if classification_config and not hasattr(full_config, "catalog"):
            # Legacy mode: full_config is actually classification_config
            self.config = full_config
            self.full_config = None
        else:
            # Normal case: full_config provided
            self.full_config = full_config
            self.config = full_config.catalog.classification

        self.populations: dict[str, PopulationBin] = {}
        self.bin_configs = self.config.binning  # Dict of BinConfig objects
        self.split_labels: list[str] = []
        # NEW: Check if splitting is enabled
        self.has_splitting = self.config.split_type is not None
        if self.has_splitting:
            logger.info(f"Population splitting enabled: {self.config.split_type}")
        else:
            logger.info(
                "No population splitting - all sources in single population type"
            )

    def _create_bin_combinations(self) -> list[dict[str, tuple[float, float]]]:
        """Create all combinations of bin ranges from all binning dimensions"""
        bin_names = list(self.bin_configs.keys())
        bin_ranges_list = []

        for bin_name in bin_names:
            bin_config = self.bin_configs[bin_name]
            ranges = [
                (bin_config.bins[i], bin_config.bins[i + 1])
                for i in range(len(bin_config.bins) - 1)
            ]
            bin_ranges_list.append([(bin_name, r) for r in ranges])

        # Create all combinations using itertools.product
        combinations = []
        for combo in product(*bin_ranges_list):
            combinations.append(dict(combo))

        return combinations

    def _create_population_id(
        self, bin_ranges: dict[str, tuple[float, float]], split_value: int
    ) -> str:
        """Create population ID (UPDATED for optional splitting)"""
        parts = []

        # Sort by bin name for consistent ordering
        for bin_name in sorted(bin_ranges.keys()):
            min_val, max_val = bin_ranges[bin_name]
            parts.append(f"{bin_name}_{min_val}_{max_val}")

        # Only add split info if splitting is enabled
        if self.has_splitting:
            parts.append(f"split_{split_value}")

        return "__".join(parts)

    def _validate_catalog_columns(self, catalog_df: pd.DataFrame) -> None:
        """Validate that catalog has required columns"""
        required_cols = []

        # Add all binning columns
        for bin_config in self.bin_configs.values():
            required_cols.append(bin_config.id)

        # Add split parameter columns if needed
        if self.config.split_params:
            if self.config.split_type in [SplitType.UVJ, SplitType.NUVRJ]:
                required_cols.extend(self.config.split_params.bins.values())

        """
        missing_cols = [col for col in required_cols if col not in catalog_df.columns]
        if missing_cols:
            available_cols = list(catalog_df.columns)
            raise PopulationError(
                f"Missing required columns: {missing_cols}\n"
                f"Available columns: {available_cols[:10]}..."
            )
        """

    def _classify_by_labels(self, catalog_df: pd.DataFrame) -> np.ndarray:
        """Classify sources using predefined labels"""
        if not self.config.split_params:
            # No splitting, all sources get label 0
            return np.zeros(len(catalog_df), dtype=int)

        label_col = self.config.split_params.id
        if label_col not in catalog_df.columns:
            raise PopulationError(f"Label column '{label_col}' not found in catalog")

        return catalog_df[label_col].values

    def _classify_by_uvj(self, catalog_df) -> np.ndarray:
        """Classify sources using UVJ criteria"""
        if not self.config.split_params:
            raise PopulationError("split_params required for UVJ classification")

        # Check if we already have a UVJ_class column
        if hasattr(catalog_df, "columns"):  # pandas
            columns = catalog_df.columns
        else:  # polars
            columns = catalog_df.columns

        if "UVJ_class" in columns:
            logger.debug("Using existing UVJ_class column")
            if hasattr(catalog_df, "values"):  # pandas
                return catalog_df["UVJ_class"].values
            else:  # polars
                return catalog_df["UVJ_class"].to_numpy()

        # Calculate UVJ classification from colors
        uv_col = self.config.split_params.bins.get("U-V")
        vj_col = self.config.split_params.bins.get("V-J")

        if not uv_col or not vj_col:
            raise PopulationError("U-V and V-J columns required for UVJ classification")

        # Handle both pandas and polars
        if hasattr(catalog_df, "values"):  # pandas
            uv = catalog_df[uv_col].values
            vj = catalog_df[vj_col].values
        else:  # polars
            uv = catalog_df[uv_col].to_numpy()
            vj = catalog_df[vj_col].to_numpy()

        # UVJ quiescent criteria (Whitaker et al. 2011)
        quiescent = (uv > 1.3) & (vj < 1.6) & (uv > 0.88 * vj + 0.59)

        return quiescent.astype(int)

    def _classify_by_nuvrj(self, catalog_df: pd.DataFrame) -> np.ndarray:
        """
        Classify sources using NUVRJ criteria - UPDATED for exact Ilbert et al. 2013 implementation

        Reference: Ilbert et al. 2013, A&A, 556, A55, Section 3.3
        "Mass assembly in quiescent and star-forming galaxies since z≃4 from UltraVISTA"

        The paper uses a "slightly modified version of the two-colour selection
        technique proposed by Williams et al. (2009), following Ilbert et al. (2010)"
        using NUV - r+ versus r+ - J instead of U - V versus V - J.

        EXACT CRITERIA:
        1. M_NUV - M_r > 3(M_r - M_J) + 1  (diagonal line)
        2. M_NUV - M_r > 3.1                (horizontal line)

        Both conditions must be satisfied for quiescent classification.
        """
        if not self.config.split_params:
            raise PopulationError("split_params required for NUVRJ classification")

        # Check if we already have a NUVRJ_class column
        if hasattr(catalog_df, "columns"):  # pandas
            columns = catalog_df.columns
        else:  # polars
            columns = catalog_df.columns

        if "NUVRJ_class" in columns:
            logger.debug("Using existing NUVRJ_class column")
            if hasattr(catalog_df, "values"):  # pandas
                return catalog_df["NUVRJ_class"].values
            else:  # polars
                return catalog_df["NUVRJ_class"].to_numpy()

        # Calculate NUVRJ classification from colors
        # Get color column names from config
        nuv_r_col = self.config.split_params.bins.get("NUV-R")  # Should be NUV - r+
        r_j_col = self.config.split_params.bins.get("R-J")  # Should be r+ - J

        if not nuv_r_col or not r_j_col:
            raise PopulationError(
                "NUVRJ classification requires 'NUV-R' and 'R-J' columns in split_params.bins"
            )

        # Check if columns exist
        missing_cols = []
        if nuv_r_col not in catalog_df.columns:
            missing_cols.append(f"NUV-R color column '{nuv_r_col}'")
        if r_j_col not in catalog_df.columns:
            missing_cols.append(f"R-J color column '{r_j_col}'")

        if missing_cols:
            available_cols = [
                col
                for col in catalog_df.columns
                if any(c in col.lower() for c in ["nuv", "r-j", "r_j", "color"])
            ]
            error_msg = f"Missing columns: {', '.join(missing_cols)}"
            if available_cols:
                error_msg += f"\nAvailable color-like columns: {available_cols}"
            raise PopulationError(error_msg)

        # Get color values
        if hasattr(catalog_df, "values"):  # pandas
            nuv_r = catalog_df[nuv_r_col].values
            r_j = catalog_df[r_j_col].values
        else:  # polars
            nuv_r = catalog_df[nuv_r_col].to_numpy()
            r_j = catalog_df[r_j_col].to_numpy()

        # Apply Ilbert et al. 2013 NUVRJ classification criteria
        # Criterion 1: M_NUV - M_r > 3(M_r - M_J) + 1 (diagonal line)
        criterion_1 = nuv_r > (3 * r_j + 1)

        # Criterion 2: M_NUV - M_r > 3.1 (horizontal line)
        criterion_2 = nuv_r > 3.1

        # Quiescent classification: BOTH criteria must be satisfied
        # This is more restrictive than the simple NUV-R > 3.1 AND R-J > 0.9
        quiescent_mask = criterion_1 & criterion_2

        # Create classification array (0=star-forming, 1=quiescent)
        classification = np.zeros(len(catalog_df), dtype=int)
        classification[quiescent_mask] = 1

        # Handle NaN values (classify as star-forming by default)
        nan_mask = np.isnan(nuv_r) | np.isnan(r_j)
        classification[nan_mask] = 0

        # Statistics and validation
        valid_mask = ~nan_mask
        n_total = len(catalog_df)
        n_valid = np.sum(valid_mask)
        n_star_forming = np.sum((classification == 0) & valid_mask)
        n_quiescent = np.sum(classification == 1)
        n_nan = np.sum(nan_mask)

        logger.info("NUVRJ classification results (Ilbert et al. 2013 criteria):")
        logger.info(f"  Total sources: {n_total:,}")
        logger.info(f"  Valid colors: {n_valid:,} ({n_valid / n_total * 100:.1f}%)")
        logger.info(
            f"  Star-forming: {n_star_forming:,} ({n_star_forming / n_valid * 100:.1f}% of valid)"
        )
        logger.info(
            f"  Quiescent: {n_quiescent:,} ({n_quiescent / n_valid * 100:.1f}% of valid)"
        )
        logger.info(f"  NaN/invalid: {n_nan:,}")

        # Diagnostic: compare with simple NUVRJ criteria
        simple_quiescent = (nuv_r > 3.1) & (r_j > 0.9) & valid_mask
        n_simple_q = np.sum(simple_quiescent)

        logger.info("  Comparison with simple NUVRJ (NUV-R > 3.1 AND R-J > 0.9):")
        logger.info(f"    Simple criteria would find: {n_simple_q:,} quiescent")
        logger.info(f"    Ilbert 2013 criteria: {n_quiescent:,} quiescent")
        logger.info(
            f"    Difference: {n_simple_q - n_quiescent:,} fewer with Ilbert criteria"
        )

        # Additional diagnostic: show intersection point
        # The two lines intersect when 3.1 = 3*r_j + 1, so r_j = 0.7
        intersection_r_j = (3.1 - 1) / 3  # = 0.7
        logger.debug(f"  Ilbert criteria intersection at R-J = {intersection_r_j:.2f}")

        # Warn if results seem unusual
        quiescent_fraction = n_quiescent / n_valid * 100
        if quiescent_fraction < 10:
            logger.warning(
                f"Very low quiescent fraction ({quiescent_fraction:.1f}%) - check color definitions"
            )
            logger.warning(
                "Expected COSMOS quiescent fraction: ~15-25% with Ilbert 2013 criteria"
            )
        elif quiescent_fraction > 40:
            logger.warning(
                f"Very high quiescent fraction ({quiescent_fraction:.1f}%) - check color definitions"
            )

        return classification

    def _calculate_beta_uv_values(
        self, catalog_df: pd.DataFrame, formula_params
    ) -> np.ndarray:
        """Calculate β_UV values from E(B-V) and dust law"""

        # Get column names
        ebv_col = formula_params.bins.get("E(B-V)")
        law_col = formula_params.bins.get("dust_law")
        single_law = formula_params.single_law

        if not ebv_col:
            raise PopulationError("E(B-V) column required for beta_UV calculation")

        # Check columns exist
        if ebv_col not in catalog_df.columns:
            available_cols = [col for col in catalog_df.columns if "ebv" in col.lower()]
            raise PopulationError(
                f"E(B-V) column '{ebv_col}' not found. Available: {available_cols}"
            )

        # Get E(B-V) values
        ebv = catalog_df[ebv_col].values

        # Determine dust law to use
        if single_law:
            # Use single dust law for all galaxies
            law_mapping = {"calzetti": 0, "arnouts": 1, "salim": 2}
            if single_law.lower() in law_mapping:
                dust_law = np.full(len(catalog_df), law_mapping[single_law.lower()])
                logger.info(f"Using single dust law: {single_law}")
            else:
                logger.warning(
                    f"Unknown single_law '{single_law}', defaulting to Calzetti"
                )
                dust_law = np.zeros(len(catalog_df))
        elif law_col and law_col in catalog_df.columns:
            # Use existing dust law column
            dust_law = catalog_df[law_col].values
            logger.info("Using existing dust law column")
        else:
            # Default to Calzetti for all
            dust_law = np.zeros(len(catalog_df))
            logger.info("No dust law specified, defaulting to Calzetti")

        # Constants
        beta_intrinsic = -2.3
        k_lambda_values = {0: 4.43, 1: 4.20, 2: 3.80}  # Calzetti, Arnouts, Salim

        # Calculate β_UV
        beta_uv = np.full_like(ebv, beta_intrinsic)

        for law_idx, k_lambda in k_lambda_values.items():
            mask = dust_law == law_idx
            if np.any(mask):
                beta_uv[mask] = beta_intrinsic + k_lambda * ebv[mask]

        # Handle invalid E(B-V)
        invalid_mask = ~np.isfinite(ebv) | (ebv < 0)
        if np.any(invalid_mask):
            beta_uv[invalid_mask] = beta_intrinsic
            logger.warning(f"Found {np.sum(invalid_mask)} sources with invalid E(B-V)")

        # Log statistics
        valid_mask = np.isfinite(beta_uv) & np.isfinite(ebv)
        if np.any(valid_mask):
            logger.info(f"β_UV calculation: {np.sum(valid_mask):,} valid sources")
            logger.info(
                f"β_UV range: {np.min(beta_uv[valid_mask]):.2f} to {np.max(beta_uv[valid_mask]):.2f}"
            )
            logger.info(f"Median β_UV: {np.median(beta_uv[valid_mask]):.2f}")

        return beta_uv

    def _create_populations(
        self, catalog_df: pd.DataFrame, split_values: np.ndarray
    ) -> None:
        """Create population bins for all combinations (GENERALIZED)"""

        # Get column data for all binning dimensions
        column_data = {}
        for bin_name, bin_config in self.bin_configs.items():
            column_data[bin_name] = catalog_df[bin_config.id].values

        # Get unique split values
        unique_splits = np.unique(split_values)
        self.split_labels = [f"split_{i}" for i in unique_splits]

        # Create all bin combinations
        bin_combinations = self._create_bin_combinations()

        logger.info(
            f"Creating populations from {len(bin_combinations)} bin combinations × {len(unique_splits)} splits"
        )

        # Create populations for each combination
        total_populations = 0
        for bin_ranges in bin_combinations:
            for split_val in unique_splits:
                # Create mask for this bin combination
                mask = np.ones(len(catalog_df), dtype=bool)

                # Apply each binning dimension
                for bin_name, (min_val, max_val) in bin_ranges.items():
                    col_data = column_data[bin_name]
                    mask = mask & (col_data >= min_val) & (col_data < max_val)

                # Apply split condition
                mask = mask & (split_values == split_val)

                indices = np.where(mask)[0]

                if len(indices) > 0:  # Only create if has sources
                    # Calculate medians for each binned variable
                    medians = {}
                    for bin_name in bin_ranges.keys():
                        col_data = column_data[bin_name]
                        medians[bin_name] = np.median(col_data[indices])

                    # Create population ID
                    id_label = self._create_population_id(bin_ranges, split_val)

                    # Create population bin
                    pop_bin = PopulationBin(
                        id_label=id_label,
                        bin_ranges=bin_ranges,
                        split_label=f"split_{split_val}",
                        split_value=split_val,
                        indices=indices,
                        medians=medians,
                    )

                    self.populations[id_label] = pop_bin
                    total_populations += 1

        logger.info(f"Created {total_populations} populations with sources")

    def calculate_formula_values(
        self, catalog_df: pd.DataFrame, formula_name: str
    ) -> np.ndarray:
        """
        Calculate values for a formula-based binning variable

        Args:
            catalog_df: Input catalog
            formula_name: Name of the formula to calculate

        Returns:
            Array of calculated values
        """
        if not self.config.formulas or formula_name not in self.config.formulas:
            raise PopulationError(
                f"Formula '{formula_name}' not found in configuration"
            )

        formula_params = self.config.formulas[formula_name]

        if formula_params.formula == "beta_uv":
            return self._calculate_beta_uv_values(catalog_df, formula_params)
        elif formula_params.formula == "custom":
            return self._calculate_custom_formula(catalog_df, formula_params)
        else:
            raise PopulationError(f"Unknown formula type: {formula_params.formula}")

    def classify_catalog(self, catalog_df) -> None:
        """Classify catalog sources into populations (UPDATED for optional splitting)"""
        # Convert polars to pandas if needed
        if hasattr(catalog_df, "to_pandas"):
            logger.debug(
                "Converting Polars DataFrame to pandas for population classification"
            )
            pandas_df = catalog_df.to_pandas()
            self.catalog_df = pandas_df
            self._original_catalog_df = catalog_df
            self._catalog_backend = "polars"
        else:
            pandas_df = catalog_df
            self.catalog_df = catalog_df
            self._catalog_backend = "pandas"

        # Pre-calculate any formula-based variables
        if self.config.formulas:
            for formula_name, _formula_params in self.config.formulas.items():
                column_name = f"calculated_{formula_name}"
                calculated_values = self.calculate_formula_values(
                    pandas_df, formula_name
                )
                pandas_df[column_name] = calculated_values
                logger.info(f"Added calculated column: {column_name}")

        # Validate required columns
        self._validate_catalog_columns(pandas_df)

        # Handle splitting (or lack thereof)
        if self.has_splitting:
            # Traditional splitting approach
            if self.config.split_type == SplitType.LABELS:
                split_values = self._classify_by_labels(pandas_df)
            elif self.config.split_type == SplitType.UVJ:
                split_values = self._classify_by_uvj(pandas_df)
            elif self.config.split_type == SplitType.NUVRJ:
                split_values = self._classify_by_nuvrj(pandas_df)
            else:
                raise PopulationError(f"Unknown split type: {self.config.split_type}")
        else:
            # No splitting - all sources get the same value
            split_values = np.zeros(len(pandas_df), dtype=int)
            logger.info("No splitting applied - all sources in single population type")

        # Create populations for all combinations of bins
        self._create_populations(pandas_df, split_values)

    def get_population_data(self, population_id: str) -> dict[str, np.ndarray]:
        """Get catalog data for a specific population (GENERALIZED)"""
        if population_id not in self.populations:
            raise PopulationError(f"Population {population_id} not found")

        pop_bin = self.populations[population_id]
        indices = pop_bin.indices

        if not hasattr(self, "catalog_df") or self.catalog_df is None:
            raise PopulationError("No catalog data loaded")

        # Get astrometry columns from config
        if hasattr(self, "full_config") and self.full_config:
            ra_col = self.full_config.catalog.astrometry["ra"]
            dec_col = self.full_config.catalog.astrometry["dec"]
        else:
            # Fallback for legacy usage
            ra_col = "ALPHA_J2000"
            dec_col = "DELTA_J2000"

        # Initialize data dict
        data = {"indices": indices}

        try:
            subset = self.catalog_df.iloc[indices]

            # Always include RA/Dec
            data["ra"] = subset[ra_col].values
            data["dec"] = subset[dec_col].values

            # Include all binned variables
            for bin_name, bin_config in self.bin_configs.items():
                data[bin_name] = subset[bin_config.id].values

            # Legacy compatibility - add redshift and stellar_mass if they exist
            if "redshift" in self.bin_configs:
                data["redshift"] = data["redshift"]  # Already added above
            if "stellar_mass" in self.bin_configs:
                data["stellar_mass"] = data["stellar_mass"]  # Already added above

        except KeyError as e:
            available_cols = list(self.catalog_df.columns)
            expected_cols = {
                "ra": ra_col,
                "dec": dec_col,
            }
            expected_cols.update(
                {
                    bin_name: bin_config.id
                    for bin_name, bin_config in self.bin_configs.items()
                }
            )

            raise PopulationError(
                f"Required column '{e}' not found.\n"
                f"Expected columns: {expected_cols}\n"
                f"Available: {available_cols[:10]}..."
            ) from e

        return data

    def get_population_summary(self) -> pd.DataFrame:
        """Get summary statistics of all populations (GENERALIZED)"""
        data = []
        for pop_bin in self.populations.values():
            row = {
                "id_label": pop_bin.id_label,
                "n_sources": pop_bin.n_sources,
                "split_label": pop_bin.split_label,
            }

            # Add range information for each binning dimension
            for bin_name, (min_val, max_val) in pop_bin.bin_ranges.items():
                row[f"{bin_name}_range"] = f"{min_val:.2f}-{max_val:.2f}"
                row[f"{bin_name}_median"] = pop_bin.medians.get(bin_name, np.nan)

            # Legacy compatibility
            if "redshift" in pop_bin.bin_ranges:
                row["z_range"] = row["redshift_range"]
                row["median_z"] = row["redshift_median"]
            if "stellar_mass" in pop_bin.bin_ranges:
                row["mass_range"] = row["stellar_mass_range"]
                row["median_mass"] = row["stellar_mass_median"]

            data.append(row)

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
        """Create bootstrap version of populations"""
        if seed is not None:
            np.random.seed(seed)

        bootstrap_manager = PopulationManager(self.full_config or self.config)
        bootstrap_manager.bin_configs = self.bin_configs
        bootstrap_manager.split_labels = self.split_labels
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
                    bin_ranges=pop_bin.bin_ranges,
                    split_label=pop_bin.split_label,
                    split_value=pop_bin.split_value,
                    indices=bootstrap_indices,
                    medians=pop_bin.medians,
                )

                bootstrap_manager.populations[bootstrap_bin.id_label] = bootstrap_bin

        return bootstrap_manager

    def get_population_type_summary(self) -> dict[str, dict]:
        """Get summary of populations by type (star-forming vs quiescent)"""
        if not hasattr(self, "catalog_df") or self.catalog_df is None:
            return {}

        summary = {
            "star_forming": {"populations": [], "total_sources": 0},
            "quiescent": {"populations": [], "total_sources": 0},
        }

        for pop_id, pop_bin in self.populations.items():
            if pop_bin.split_value == 0:  # star-forming
                summary["star_forming"]["populations"].append(pop_id)
                summary["star_forming"]["total_sources"] += pop_bin.n_sources
            elif pop_bin.split_value == 1:  # quiescent
                summary["quiescent"]["populations"].append(pop_id)
                summary["quiescent"]["total_sources"] += pop_bin.n_sources

        return summary

    def __len__(self) -> int:
        """Return number of populations"""
        return len(self.populations)

    def __iter__(self) -> Iterator[str]:
        """Iterate over population ID labels"""
        return iter(self.populations.keys())
