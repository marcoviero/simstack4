"""
COSMOS catalog handling for Simstack4

This module provides specialized handling for COSMOS catalogs,
including automatic UVJ classification and integration with the Simstack4 pipeline.
Optimized for large (8GB+) COSMOS catalogs using Polars backend.
"""

import warnings
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
from astropy.io import fits
from astropy.table import Table

# Import both pandas and polars with fallback handling
try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

from .config import CatalogConfig, ClassificationConfig, SplitType
from .exceptions.simstack_exceptions import CatalogError, ValidationError
from .populations import PopulationManager
from .utils import setup_logging

logger = setup_logging()


class COSMOSCatalog:
    """
    Specialized handler for COSMOS catalogs

    This class automatically handles COSMOS catalog formats and provides
    built-in UVJ classification following Whitaker et al. (2011) criteria.
    Optimized for large catalogs using Polars backend by default.
    """

    # Standard COSMOS column mappings (flexible for different COSMOS releases)
    COSMOS_COLUMNS = {
        "ra": "ra",  # Right ascension ✓
        "dec": "dec",  # Declination ✓
        "id": "id",  # Source ID ✓
        "redshift": "zfinal",  # Final redshift ✓
        "stellar_mass": "mass_med",  # Median stellar mass ✓
        "u_mag": "mag_model_cfht-u",  # U-band magnitude ✓
        "v_mag": "mag_model_hsc-r",  # R-band (V-proxy) magnitude ✓
        "j_mag": "mag_model_uvista-j",  # J-band magnitude ✓
        "u_err": "mag_err_model_cfht-u",  # U-band error ✓
        "v_err": "mag_err_model_hsc-r",  # R-band error ✓
        "j_err": "mag_err_model_uvista-j",  # J-band error ✓
        "photoz": "zpdf_med",  # Photometric redshift ✓
        "photoz_err": "zpdf_l68",  # Photo-z lower 68% ✓
        "specz": "zfinal",  # Use zfinal as best redshift ✓
        "chi2": "chi2_best",  # SED fitting chi-squared ✓
        "flag": "quality_flag",  # Combined quality flag ✓
    }

    def __init__(
        self,
        catalog_path: Path,
        custom_column_mapping: Optional[dict] = None,
        backend: str = "auto",
    ):
        """
        Initialize COSMOS catalog handler

        Args:
            catalog_path: Path to COSMOS FITS file
            custom_column_mapping: Optional custom column name mappings
            backend: Data backend - "auto", "polars", or "pandas" (auto defaults to polars for large files)
        """
        self.catalog_path = Path(catalog_path)
        self.catalog_df: Optional[Union[pl.DataFrame, pd.DataFrame]] = None
        self.population_manager: Optional[PopulationManager] = None
        self.backend = self._select_backend(backend)

        # Use custom mappings if provided, otherwise use defaults
        self.column_mapping = self.COSMOS_COLUMNS.copy()
        if custom_column_mapping:
            self.column_mapping.update(custom_column_mapping)

        logger.info(f"COSMOS catalog initialized: {self.catalog_path}")
        logger.info(f"Using {self.backend} backend for large catalog processing")

    def _select_backend(self, backend: str) -> str:
        """Select appropriate data backend, preferring Polars for large files"""
        if backend == "auto":
            # For large COSMOS catalogs, prefer Polars
            if HAS_POLARS:
                return "polars"
            elif HAS_PANDAS:
                logger.warning(
                    "Polars not available for large catalog - falling back to pandas"
                )
                return "pandas"
            else:
                raise CatalogError("Neither pandas nor polars is available")

        elif backend == "polars":
            if not HAS_POLARS:
                if HAS_PANDAS:
                    warnings.warn(
                        "Polars not available, falling back to pandas", stacklevel=2
                    )
                    return "pandas"
                else:
                    raise CatalogError("Polars not available and no fallback")
            return "polars"

        elif backend == "pandas":
            if not HAS_PANDAS:
                raise CatalogError("Pandas not available")
            return "pandas"

        else:
            raise CatalogError(f"Unknown backend: {backend}")

    def load_catalog(
        self, quality_cuts: bool = True, redshift_range: tuple = (0.1, 4.0)
    ) -> Union[pl.DataFrame, pd.DataFrame]:
        """
        Load and process the COSMOS catalog with memory-efficient streaming

        Args:
            quality_cuts: Apply standard quality cuts
            redshift_range: Redshift range to keep (z_min, z_max)

        Returns:
            Processed catalog DataFrame (Polars or pandas)
        """
        if not self.catalog_path.exists():
            raise CatalogError(f"COSMOS catalog not found: {self.catalog_path}")

        logger.info("Loading COSMOS catalog with memory optimization...")

        try:
            # Load FITS file efficiently
            if self.backend == "polars":
                self.catalog_df = self._load_fits_polars()
            else:
                self.catalog_df = self._load_fits_pandas()

            logger.info(f"Loaded {len(self.catalog_df)} sources from COSMOS catalog")

            # Validate required columns exist
            self._validate_cosmos_columns()

            # Apply quality cuts if requested
            if quality_cuts:
                self._apply_quality_cuts_optimized(redshift_range)

            # Calculate UVJ colors
            self._calculate_uvj_colors_optimized()

            logger.info(
                f"Final catalog: {len(self.catalog_df)} sources after processing"
            )

            return self.catalog_df

        except Exception as e:
            raise CatalogError(f"Failed to load COSMOS catalog: {e}") from e

    def _load_fits_polars(self) -> pl.DataFrame:
        """Load FITS file using Polars for memory efficiency"""
        logger.debug("Loading FITS with Polars backend for memory efficiency")

        with fits.open(self.catalog_path, memmap=True) as hdul:
            # Find the main catalog table
            catalog_hdu = None
            for i, hdu in enumerate(hdul):
                if hasattr(hdu, "data") and hdu.data is not None:
                    if hasattr(hdu.data, "dtype") and hdu.data.dtype.names:
                        catalog_hdu = hdu
                        logger.debug(f"Found catalog data in HDU {i}")
                        break

            if catalog_hdu is None:
                raise CatalogError("No table data found in COSMOS FITS file")

            # Convert to astropy Table first (handles FITS efficiently)
            table = Table(catalog_hdu.data)

            # Convert to pandas then to polars (most reliable path)
            pandas_df = table.to_pandas()

            # Convert to polars with type optimization
            polars_df = pl.from_pandas(pandas_df).with_columns(
                [
                    # Optimize data types for memory efficiency
                    pl.col("ra").cast(pl.Float32),
                    pl.col("dec").cast(pl.Float32),
                ]
                + [
                    # Cast magnitude columns to Float32 for memory savings
                    pl.col(col).cast(pl.Float32)
                    for col in pandas_df.columns
                    if any(
                        mag_type in col.lower()
                        for mag_type in ["mag", "err", "flux", "mass", "z_"]
                    )
                    and col in pandas_df.columns
                ]
            )

            return polars_df

    def _load_fits_pandas(self) -> pd.DataFrame:
        """Load FITS file using pandas (fallback)"""
        logger.debug("Loading FITS with pandas backend")

        with fits.open(self.catalog_path, memmap=True) as hdul:
            catalog_hdu = None
            for _i, hdu in enumerate(hdul):
                if hasattr(hdu, "data") and hdu.data is not None:
                    if hasattr(hdu.data, "dtype") and hdu.data.dtype.names:
                        catalog_hdu = hdu
                        break

            if catalog_hdu is None:
                raise CatalogError("No table data found in COSMOS FITS file")

            # Convert to pandas with memory optimization
            table = Table(catalog_hdu.data)
            pandas_df = table.to_pandas()

            # Optimize dtypes for memory
            for col in pandas_df.columns:
                if pandas_df[col].dtype == "float64":
                    # Try to downcast to float32 if precision allows
                    if any(
                        mag_type in col.lower()
                        for mag_type in [
                            "mag",
                            "err",
                            "flux",
                            "ra",
                            "dec",
                            "mass",
                            "z_",
                        ]
                    ):
                        pandas_df[col] = pd.to_numeric(pandas_df[col], downcast="float")

            return pandas_df

    def _validate_cosmos_columns(self) -> None:
        """Validate that required COSMOS columns are present"""
        required_cols = [
            "ra",
            "dec",
            "redshift",
            "stellar_mass",
            "u_mag",
            "v_mag",
            "j_mag",
        ]
        missing_cols = []

        # Get column list based on backend
        if self.backend == "polars":
            available_columns = self.catalog_df.columns
        else:
            available_columns = list(self.catalog_df.columns)

        for req_col in required_cols:
            cosmos_col = self.column_mapping.get(req_col)
            if cosmos_col not in available_columns:
                missing_cols.append(f"{req_col} (expected: {cosmos_col})")

        if missing_cols:
            logger.error(
                f"Available columns: {available_columns[:20]}..."
            )  # Show first 20
            raise ValidationError(f"Missing required COSMOS columns: {missing_cols}")

        logger.debug("COSMOS column validation passed")

    def _apply_quality_cuts_optimized(self, redshift_range: tuple) -> None:
        """Apply standard quality cuts to COSMOS catalog with backend optimization"""
        initial_count = len(self.catalog_df)

        # Standard COSMOS quality cuts following the tutorial
        z_col = self.column_mapping["redshift"]
        mass_col = self.column_mapping["stellar_mass"]
        flag_col = self.column_mapping.get("flag")
        u_col = self.column_mapping["u_mag"]
        v_col = self.column_mapping["v_mag"]
        j_col = self.column_mapping["j_mag"]

        z_min, z_max = redshift_range

        if self.backend == "polars":
            # Use Polars lazy evaluation for memory efficiency
            self.catalog_df = (
                self.catalog_df.lazy()
                .filter(
                    # Redshift cuts
                    (pl.col(z_col) >= z_min)
                    & (pl.col(z_col) <= z_max)
                    &
                    # Stellar mass cut
                    (pl.col(mass_col) > 9.0)
                    &
                    # Photometry quality cuts
                    pl.col(u_col).is_finite()
                    & (pl.col(u_col) > 0)
                    & pl.col(v_col).is_finite()
                    & (pl.col(v_col) > 0)
                    & pl.col(j_col).is_finite()
                    & (pl.col(j_col) > 0)
                    &
                    # Flag cut if available
                    (
                        pl.col(flag_col) == 0
                        if flag_col and flag_col in self.catalog_df.columns
                        else pl.lit(True)
                    )
                )
                .collect()
            )

        else:
            # Pandas version with chained filtering
            mask = (
                (self.catalog_df[z_col] >= z_min)
                & (self.catalog_df[z_col] <= z_max)
                & (self.catalog_df[mass_col] > 9.0)
                & np.isfinite(self.catalog_df[u_col])
                & (self.catalog_df[u_col] > 0)
                & np.isfinite(self.catalog_df[v_col])
                & (self.catalog_df[v_col] > 0)
                & np.isfinite(self.catalog_df[j_col])
                & (self.catalog_df[j_col] > 0)
            )

            # Add flag cut if available
            if flag_col and flag_col in self.catalog_df.columns:
                mask = mask & (self.catalog_df[flag_col] == 0)

            self.catalog_df = self.catalog_df[mask].copy()

        logger.info(f"Quality cuts: {initial_count} -> {len(self.catalog_df)} sources")

    def _calculate_uvj_colors_optimized(self) -> None:
        """
        Calculate UVJ colors with smart detection for pre-calculated values
        UPDATED to handle both raw catalogs and pre-processed clean catalogs
        """

        # First, check if UVJ colors and classification already exist (clean catalog)
        if self.backend == "polars":
            columns = self.catalog_df.columns
            has_uvj = all(col in columns for col in ["U-V", "V-J", "UVJ_class"])

            if has_uvj:
                logger.info("✓ Found pre-calculated UVJ colors and classification")
                n_q = self.catalog_df.filter(pl.col("UVJ_class") == 1).height
                n_sf = len(self.catalog_df) - n_q
                logger.info(
                    f"  Using existing UVJ classification: {n_sf} star-forming, {n_q} quiescent"
                )
                return

        else:  # pandas
            columns = self.catalog_df.columns
            has_uvj = all(col in columns for col in ["U-V", "V-J", "UVJ_class"])

            if has_uvj:
                logger.info("✓ Found pre-calculated UVJ colors and classification")
                n_sf = np.sum(self.catalog_df["UVJ_class"] == 0)
                n_q = np.sum(self.catalog_df["UVJ_class"] == 1)
                logger.info(
                    f"  Using existing UVJ classification: {n_sf} star-forming, {n_q} quiescent"
                )
                return

        # If not pre-calculated, calculate from magnitude columns
        logger.info("UVJ colors not found - calculating from magnitude columns...")

        u_col = self.column_mapping["u_mag"]
        v_col = self.column_mapping["v_mag"]
        j_col = self.column_mapping["j_mag"]

        if self.backend == "polars":
            # Use Polars expressions for efficient computation
            self.catalog_df = self.catalog_df.with_columns(
                [
                    (pl.col(u_col) - pl.col(v_col)).alias("U-V"),
                    (pl.col(v_col) - pl.col(j_col)).alias("V-J"),
                ]
            ).with_columns(
                [
                    # UVJ classification using Polars when expressions
                    pl.when(
                        (pl.col("U-V") > 1.3)
                        & (pl.col("V-J") < 1.6)
                        & (pl.col("U-V") > 0.88 * pl.col("V-J") + 0.59)
                    )
                    .then(1)
                    .otherwise(0)
                    .alias("UVJ_class")
                ]
            )

            # Get classification statistics
            n_q = self.catalog_df.filter(pl.col("UVJ_class") == 1).height
            n_sf = len(self.catalog_df) - n_q

        else:  # pandas
            # Calculate colors
            self.catalog_df["U-V"] = self.catalog_df[u_col] - self.catalog_df[v_col]
            self.catalog_df["V-J"] = self.catalog_df[v_col] - self.catalog_df[j_col]

            # Classify using UVJ criteria (Whitaker et al. 2011)
            uv = self.catalog_df["U-V"].values
            vj = self.catalog_df["V-J"].values

            # Quiescent criteria
            quiescent_mask = (uv > 1.3) & (vj < 1.6) & (uv > 0.88 * vj + 0.59)

            # Create classification column: 0 = star-forming, 1 = quiescent
            self.catalog_df["UVJ_class"] = quiescent_mask.astype(int)

            n_sf = np.sum(~quiescent_mask)
            n_q = np.sum(quiescent_mask)

        logger.info(
            f"  Calculated UVJ classification: {n_sf} star-forming, {n_q} quiescent"
        )

    def create_simstack_config(
        self,
        redshift_bins: list,
        mass_bins: list,
    ) -> CatalogConfig:
        """
        Create a CatalogConfig object for use with Simstack4

        Args:
            redshift_bins: Redshift bin edges
            mass_bins: Stellar mass bin edges (log scale)

        Returns:
            CatalogConfig object ready for Simstack4 pipeline
        """
        if self.catalog_df is None:
            raise CatalogError("Must load catalog first")

        # Create classification config with UVJ splitting
        from .config import ClassificationBins, SplitParams

        classification_config = ClassificationConfig(
            split_type=SplitType.UVJ,
            redshift=ClassificationBins(
                id=self.column_mapping["redshift"], bins=redshift_bins
            ),
            stellar_mass=ClassificationBins(
                id=self.column_mapping["stellar_mass"], bins=mass_bins
            ),
            split_params=SplitParams(
                id="UVJ_class", bins={"U-V": "U-V", "V-J": "V-J"}  # Column we created
            ),
        )

        # Create catalog config
        catalog_config = CatalogConfig(
            path=str(self.catalog_path.parent),
            file=self.catalog_path.name,
            astrometry={
                "ra": self.column_mapping["ra"],
                "dec": self.column_mapping["dec"],
            },
            classification=classification_config,
        )

        return catalog_config

    def create_population_manager(
        self,
        redshift_bins: list = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0],
        mass_bins: list = [9.0, 9.5, 10.0, 10.5, 11.0, 12.0],
    ) -> PopulationManager:
        """
        Create and populate a PopulationManager for this catalog

        Args:
            redshift_bins: Redshift bin edges
            mass_bins: Stellar mass bin edges

        Returns:
            Populated PopulationManager instance
        """
        if self.catalog_df is None:
            raise CatalogError("Must load catalog first")

        # Create classification config
        config = self.create_simstack_config(redshift_bins, mass_bins)

        # Create population manager
        self.population_manager = PopulationManager(config.classification)

        # Convert to pandas if needed for population manager
        if self.backend == "polars":
            # PopulationManager currently expects pandas DataFrame
            pandas_df = self.catalog_df.to_pandas()
            self.population_manager.classify_catalog(pandas_df)
        else:
            self.population_manager.classify_catalog(self.catalog_df)

        logger.info(f"Created {len(self.population_manager)} populations")

        return self.population_manager

    def get_catalog_summary(self) -> dict[str, Any]:
        """Get summary information about the loaded catalog"""
        if self.catalog_df is None:
            return {"status": "not_loaded"}

        # Handle both Polars and pandas
        if self.backend == "polars":
            n_sources = self.catalog_df.height
            z_col = self.column_mapping["redshift"]
            mass_col = self.column_mapping["stellar_mass"]

            z_stats = self.catalog_df.select(
                [pl.col(z_col).min().alias("z_min"), pl.col(z_col).max().alias("z_max")]
            ).to_dicts()[0]

            mass_stats = self.catalog_df.select(
                [
                    pl.col(mass_col).min().alias("mass_min"),
                    pl.col(mass_col).max().alias("mass_max"),
                ]
            ).to_dicts()[0]

            redshift_range = (z_stats["z_min"], z_stats["z_max"])
            mass_range = (mass_stats["mass_min"], mass_stats["mass_max"])

        else:
            n_sources = len(self.catalog_df)
            redshift_range = (
                self.catalog_df[self.column_mapping["redshift"]].min(),
                self.catalog_df[self.column_mapping["redshift"]].max(),
            )
            mass_range = (
                self.catalog_df[self.column_mapping["stellar_mass"]].min(),
                self.catalog_df[self.column_mapping["stellar_mass"]].max(),
            )

        summary = {
            "n_sources": n_sources,
            "redshift_range": redshift_range,
            "mass_range": mass_range,
            "backend": self.backend,
        }

        # Add UVJ statistics if available
        if self.backend == "polars":
            if "UVJ_class" in self.catalog_df.columns:
                uvj_stats = (
                    self.catalog_df.group_by("UVJ_class")
                    .agg(pl.count().alias("count"))
                    .to_dicts()
                )

                n_sf = next((d["count"] for d in uvj_stats if d["UVJ_class"] == 0), 0)
                n_q = next((d["count"] for d in uvj_stats if d["UVJ_class"] == 1), 0)

                summary["uvj_classification"] = {
                    "star_forming": n_sf,
                    "quiescent": n_q,
                    "fraction_quiescent": n_q / (n_sf + n_q) if (n_sf + n_q) > 0 else 0,
                }
        else:
            if "UVJ_class" in self.catalog_df.columns:
                n_sf = np.sum(self.catalog_df["UVJ_class"] == 0)
                n_q = np.sum(self.catalog_df["UVJ_class"] == 1)
                summary["uvj_classification"] = {
                    "star_forming": n_sf,
                    "quiescent": n_q,
                    "fraction_quiescent": n_q / (n_sf + n_q) if (n_sf + n_q) > 0 else 0,
                }

        if self.population_manager:
            summary["n_populations"] = len(self.population_manager)

        return summary

    def get_memory_usage(self) -> dict[str, float]:
        """Get memory usage information for the catalog"""
        if self.catalog_df is None:
            return {"status": "no_catalog_loaded"}

        if self.backend == "polars":
            # Polars memory usage
            memory_usage = {
                "estimated_size_mb": self.catalog_df.estimated_size() / 1024 / 1024,
                "n_rows": self.catalog_df.height,
                "n_columns": len(self.catalog_df.columns),
                "backend": "polars",
            }
        else:
            # Pandas memory usage
            memory_usage = {
                "memory_usage_mb": self.catalog_df.memory_usage(deep=True).sum()
                / 1024
                / 1024,
                "n_rows": len(self.catalog_df),
                "n_columns": len(self.catalog_df.columns),
                "backend": "pandas",
            }

        return memory_usage

    def print_catalog_summary(self) -> None:
        """Print a detailed summary of the catalog"""
        summary = self.get_catalog_summary()

        print("=== COSMOS Catalog Summary ===")
        print(f"File: {self.catalog_path}")
        print(f"Sources: {summary['n_sources']:,}")
        print(f"Backend: {summary['backend']}")

        if "redshift_range" in summary:
            z_min, z_max = summary["redshift_range"]
            print(f"Redshift range: {z_min:.3f} - {z_max:.3f}")

        if "mass_range" in summary:
            m_min, m_max = summary["mass_range"]
            print(f"Stellar mass range: {m_min:.2f} - {m_max:.2f} (log M☉)")

        if "uvj_classification" in summary:
            uvj = summary["uvj_classification"]
            print(f"Star-forming: {uvj['star_forming']:,}")
            print(f"Quiescent: {uvj['quiescent']:,}")
            print(f"Quiescent fraction: {uvj['fraction_quiescent']:.1%}")

        if "n_populations" in summary:
            print(f"Populations: {summary['n_populations']}")


def load_cosmos_catalog(
    catalog_path: Path,
    backend: str = "auto",
    quality_cuts: bool = True,
    redshift_range: tuple = (0.1, 4.0),
    redshift_bins: list = [0.2, 0.5, 1.0, 1.5, 2.0, 3.0],
    mass_bins: list = [9.0, 9.5, 10.0, 10.5, 11.0, 12.0],
) -> tuple[COSMOSCatalog, PopulationManager]:
    """
    Convenience function to load COSMOS catalog and create population manager
    Optimized for large catalogs with Polars backend by default

    Args:
        catalog_path: Path to COSMOS FITS file
        backend: Data backend ("auto", "polars", "pandas") - auto defaults to polars
        quality_cuts: Apply standard quality cuts
        redshift_range: Redshift range to keep
        redshift_bins: Redshift bin edges
        mass_bins: Stellar mass bin edges

    Returns:
        Tuple of (COSMOSCatalog, PopulationManager)
    """
    # Load catalog with optimized backend
    cosmos_cat = COSMOSCatalog(catalog_path, backend=backend)
    cosmos_cat.load_catalog(quality_cuts=quality_cuts, redshift_range=redshift_range)

    # Print memory usage info
    memory_info = cosmos_cat.get_memory_usage()
    if "estimated_size_mb" in memory_info:
        logger.info(
            f"Catalog memory usage: {memory_info['estimated_size_mb']:.1f} MB ({memory_info['backend']})"
        )
    elif "memory_usage_mb" in memory_info:
        logger.info(
            f"Catalog memory usage: {memory_info['memory_usage_mb']:.1f} MB ({memory_info['backend']})"
        )

    # Create population manager
    pop_manager = cosmos_cat.create_population_manager(redshift_bins, mass_bins)

    return cosmos_cat, pop_manager


def create_cosmos_sky_catalog(
    catalog_path: Path, backend: str = "auto", **kwargs
):
    """
    Create a SkyCatalogs instance from COSMOS catalog with optimized backend

    This function can be imported and used in sky_catalogs.py to support COSMOS catalogs
    """
    from .sky_catalogs import SkyCatalogs

    # Load COSMOS catalog with optimized backend
    cosmos_cat = COSMOSCatalog(catalog_path, backend=backend)
    cosmos_cat.load_catalog(**kwargs)

    # Create catalog config
    catalog_config = cosmos_cat.create_simstack_config()

    # Create SkyCatalogs instance with matching backend
    sky_catalog_backend = "pandas" if backend == "pandas" else "polars"
    sky_catalog = SkyCatalogs(catalog_config, backend=sky_catalog_backend)

    # Set the loaded data directly instead of loading from file
    if cosmos_cat.backend == "polars" and sky_catalog.backend == "pandas":
        # Convert polars to pandas for sky_catalog
        sky_catalog.catalog_df = cosmos_cat.catalog_df.to_pandas()
    elif cosmos_cat.backend == "pandas" and sky_catalog.backend == "polars":
        # Convert pandas to polars for sky_catalog
        sky_catalog.catalog_df = pl.from_pandas(cosmos_cat.catalog_df)
    else:
        # Same backend, direct assignment
        sky_catalog.catalog_df = cosmos_cat.catalog_df

    sky_catalog._catalog_info = cosmos_cat.get_catalog_summary()

    # Create population manager
    sky_catalog.population_manager = cosmos_cat.create_population_manager()

    return sky_catalog
