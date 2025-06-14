"""
Sky catalogs handling for Simstack4

This module handles loading, validation, and processing of astronomical catalogs.
Supports both pandas and polars for different performance needs.
"""

import warnings
from pathlib import Path
from typing import Any

import numpy as np

# Try to import both pandas and polars
try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

from astropy.io import fits
from astropy.table import Table

from .config import CatalogConfig
from .cosmos import create_cosmos_sky_catalog
from .exceptions.simstack_exceptions import CatalogError, ValidationError
from .populations import PopulationManager
from .utils import setup_logging

logger = setup_logging()


class SkyCatalogs:
    """
    Handle astronomical catalogs with support for multiple data backends

    This class can use either pandas or polars as the backend, with automatic
    fallback and performance optimization for large datasets.
    """

    def __init__(self, catalog_config: CatalogConfig, backend: str = "auto"):
        """
        Initialize catalog handler

        Args:
            catalog_config: Configuration for catalog loading
            backend: Data backend - "pandas", "polars", or "auto"
        """
        self.config = catalog_config
        self.backend = self._select_backend(backend)
        self.catalog_df = None
        self.population_manager = None
        self._catalog_info = {}

        logger.info(f"SkyCatalogs initialized with {self.backend} backend")

    def _select_backend(self, backend: str) -> str:
        """Select appropriate data backend"""
        if backend == "auto":
            # Auto-select based on availability and data size
            if HAS_POLARS:
                return "polars"
            elif HAS_PANDAS:
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

    # In sky_catalogs.py, add Parquet support to the load_catalog method:

    def load_catalog(self) -> None:
        """Load catalog from file (MODIFIED to support COSMOS and Parquet)"""
        catalog_path = self.config.full_path

        if not catalog_path.exists():
            raise CatalogError(f"Catalog file not found: {catalog_path}")

        logger.info(f"Loading catalog: {catalog_path}")

        # Check if this is a COSMOS catalog
        if "COSMOS" in catalog_path.name.upper():
            logger.info("Detected COSMOS catalog, using specialized loader")
            self.load_cosmos_catalog()
            return

        # Original file format detection code with Parquet support
        suffix = catalog_path.suffix.lower()

        if suffix == ".parquet":
            self._load_parquet(catalog_path)
        elif suffix == ".csv":
            self._load_csv(catalog_path)
        elif suffix in [".fits", ".fit"]:
            self._load_fits(catalog_path)
        elif suffix in [".txt", ".dat"]:
            self._load_text(catalog_path)
        else:
            raise CatalogError(f"Unsupported file format: {suffix}")

        self._validate_catalog()
        self._create_population_manager()

        logger.info(f"Catalog loaded successfully: {len(self.catalog_df)} sources")

    def _load_parquet(self, catalog_path: Path) -> None:
        """Load Parquet file with appropriate backend"""
        try:
            if self.backend == "polars":
                self.catalog_df = pl.read_parquet(catalog_path)
                logger.debug("Loaded Parquet with polars")
            else:
                self.catalog_df = pd.read_parquet(catalog_path)
                # Convert to polars if that was the intended backend
                if self.backend == "polars" and HAS_POLARS:
                    self.catalog_df = pl.from_pandas(self.catalog_df)
                logger.debug("Loaded Parquet with pandas")
        except Exception as e:
            raise CatalogError(f"Failed to load Parquet file: {e}") from e

    def load_cosmos_catalog(self, **kwargs) -> None:
        """
        Load COSMOS catalog - SIMPLIFIED for pre-processed clean catalog

        Args:
            **kwargs: Arguments passed to catalog loader (mostly ignored now)
        """
        catalog_path = self.config.full_path

        if not catalog_path.exists():
            raise CatalogError(f"COSMOS catalog file not found: {catalog_path}")

        logger.info(f"Loading pre-processed COSMOS catalog: {catalog_path}")

        # Load the clean catalog directly (no need for special COSMOS processing)
        suffix = catalog_path.suffix.lower()

        if suffix == ".parquet":
            self._load_parquet(catalog_path)
        elif suffix in [".fits", ".fit"]:
            self._load_fits(catalog_path)
        elif suffix == ".csv":
            self._load_csv(catalog_path)
        else:
            raise CatalogError(f"Unsupported COSMOS catalog format: {suffix}")

        logger.info(f"COSMOS catalog loaded: {len(self.catalog_df)} sources")

        # Validate that we have the essential columns
        essential_cols = ["ra", "dec", "zfinal", "mass_med"]
        missing_cols = []

        if self.backend == "polars":
            available_cols = self.catalog_df.columns
        else:
            available_cols = list(self.catalog_df.columns)

        for col in essential_cols:
            if col not in available_cols:
                missing_cols.append(col)

        # if missing_cols:
        #    raise ValidationError(
        #        f"COSMOS catalog missing essential columns: {missing_cols}"
        #    )

        # Check if UVJ classification is already done
        if "UVJ_class" in available_cols:
            if self.backend == "polars":
                n_q = self.catalog_df.filter(pl.col("UVJ_class") == 1).height
                n_sf = len(self.catalog_df) - n_q
            else:
                n_sf = np.sum(self.catalog_df["UVJ_class"] == 0)
                n_q = np.sum(self.catalog_df["UVJ_class"] == 1)

            logger.info(
                f"✓ UVJ classification found: {n_sf} star-forming, {n_q} quiescent"
            )
        else:
            logger.warning(
                "No UVJ_class column found - will use split_type from config"
            )

        if "NUVRJ_class" in available_cols:
            if self.backend == "polars":
                n_q = self.catalog_df.filter(pl.col("NUVRJ_class") == 1).height
                n_sf = len(self.catalog_df) - n_q
            else:
                n_sf = np.sum(self.catalog_df["NUVRJ_class"] == 0)
                n_q = np.sum(self.catalog_df["NUVRJ_class"] == 1)

            logger.info(
                f"✓ NUVRJ classification found: {n_sf} star-forming, {n_q} quiescent"
            )
        else:
            logger.warning(
                "No NUVRJ_class column found - will use split_type from config"
            )

        # Standard catalog validation and population creation
        self._validate_catalog()
        self._create_population_manager()

        logger.info(
            f"✓ COSMOS catalog processing complete: {len(self.population_manager)} populations"
        )

    def _load_csv(self, catalog_path: Path) -> None:
        """Load CSV file with appropriate backend"""
        if self.backend == "polars":
            try:
                self.catalog_df = pl.read_csv(
                    catalog_path,
                    try_parse_dates=True,
                    null_values=["", "NULL", "null", "NaN", "nan"],
                )
                logger.debug("Loaded CSV with polars")
            except Exception as e:
                warnings.warn(
                    f"Polars failed to load CSV: {e}, trying pandas", stacklevel=2
                )
                self._load_csv_pandas(catalog_path)

        else:  # pandas
            self._load_csv_pandas(catalog_path)

    def _load_csv_pandas(self, catalog_path: Path) -> None:
        """Load CSV with pandas"""
        try:
            self.catalog_df = pd.read_csv(
                catalog_path,
                na_values=["", "NULL", "null", "NaN", "nan", "999", "-999"],
            )
            # Convert to polars if that was the intended backend
            if self.backend == "polars" and HAS_POLARS:
                self.catalog_df = pl.from_pandas(self.catalog_df)
            logger.debug("Loaded CSV with pandas")
        except Exception as e:
            raise CatalogError(f"Failed to load CSV file: {e}") from e

    def _load_fits(self, catalog_path: Path) -> None:
        """Load FITS file using astropy"""
        try:
            # Use astropy to read FITS
            with fits.open(catalog_path) as hdul:
                # Find the table HDU (usually HDU 1 for catalogs)
                table_hdu = None
                for _i, hdu in enumerate(hdul):
                    if hasattr(hdu, "data") and hdu.data is not None:
                        if hasattr(hdu.data, "dtype") and hdu.data.dtype.names:
                            table_hdu = hdu
                            break

                if table_hdu is None:
                    raise CatalogError("No table data found in FITS file")

                # Convert to astropy Table first
                table = Table(table_hdu.data)

                # Convert to desired backend
                if self.backend == "polars" and HAS_POLARS:
                    # Convert via pandas (astropy -> pandas -> polars)
                    pandas_df = table.to_pandas()
                    self.catalog_df = pl.from_pandas(pandas_df)
                else:
                    # Convert to pandas
                    self.catalog_df = table.to_pandas()

            logger.debug("Loaded FITS file")

        except Exception as e:
            raise CatalogError(f"Failed to load FITS file: {e}") from e

    def _load_text(self, catalog_path: Path) -> None:
        """Load space/tab-separated text file"""
        if self.backend == "polars":
            try:
                self.catalog_df = pl.read_csv(
                    catalog_path,
                    separator=None,  # Auto-detect separator
                    try_parse_dates=True,
                )
            except Exception as e:
                logger.warning(f"Polars failed to load text file: {e}, trying pandas")
                self._load_text_pandas(catalog_path)
        else:
            self._load_text_pandas(catalog_path)

    def _load_text_pandas(self, catalog_path: Path) -> None:
        """Load text file with pandas"""
        try:
            self.catalog_df = pd.read_csv(
                catalog_path, sep=None, engine="python"  # Auto-detect separator
            )
            if self.backend == "polars" and HAS_POLARS:
                self.catalog_df = pl.from_pandas(self.catalog_df)
        except Exception as e:
            raise CatalogError(f"Failed to load text file: {e}") from e

    def _validate_catalog(self) -> None:
        """Validate that catalog has required columns"""
        if self.catalog_df is None:
            raise ValidationError("No catalog data loaded")

        # Get column names based on backend
        if self.backend == "polars":
            columns = self.catalog_df.columns
        else:
            columns = list(self.catalog_df.columns)

        # Check required astrometry columns
        ra_col = self.config.astrometry.get("ra")
        dec_col = self.config.astrometry.get("dec")

        missing_cols = []
        if ra_col not in columns:
            missing_cols.append(f"RA column '{ra_col}'")
        if dec_col not in columns:
            missing_cols.append(f"Dec column '{dec_col}'")

        # Check required classification columns
        z_col = self.config.classification.redshift.id
        mass_col = self.config.classification.stellar_mass.id

        if z_col not in columns:
            missing_cols.append(f"Redshift column '{z_col}'")
        if mass_col not in columns:
            missing_cols.append(f"Stellar mass column '{mass_col}'")

        # Check split parameter columns if needed
        if self.config.classification.split_params:
            for (
                color_name,
                col_name,
            ) in self.config.classification.split_params.bins.items():
                if col_name not in columns:
                    missing_cols.append(f"Color column '{col_name}' for {color_name}")

        if missing_cols:
            raise ValidationError(
                f"Missing required columns: {', '.join(missing_cols)}"
            )

        logger.debug("Catalog validation passed")

    def _create_population_manager(self) -> None:
        """Create and initialize population manager"""
        self.population_manager = PopulationManager(self.config.classification)

        # Convert to pandas if population manager needs it
        if self.backend == "polars":
            # PopulationManager currently expects pandas
            pandas_df = self.catalog_df.to_pandas()
            self.population_manager.classify_catalog(pandas_df)
        else:
            self.population_manager.classify_catalog(self.catalog_df)

        logger.info(f"Created {len(self.population_manager)} populations")

    def get_catalog_info(self) -> dict[str, Any]:
        """Get information about the loaded catalog"""
        if self.catalog_df is None:
            return {"status": "not_loaded"}

        if self.backend == "polars":
            n_rows, n_cols = self.catalog_df.shape
            columns = self.catalog_df.columns
            # Get basic stats for numeric columns
            numeric_cols = [
                col
                for col in columns
                if self.catalog_df[col].dtype
                in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]
            ]
        else:
            n_rows, n_cols = self.catalog_df.shape
            columns = list(self.catalog_df.columns)
            numeric_cols = list(
                self.catalog_df.select_dtypes(include=[np.number]).columns
            )

        info = {
            "n_sources": n_rows,
            "n_columns": n_cols,
            "columns": columns,
            "numeric_columns": numeric_cols,
            "backend": self.backend,
            "file_path": str(self.config.full_path),
        }

        if self.population_manager:
            info["n_populations"] = len(self.population_manager)
            info[
                "population_summary"
            ] = self.population_manager.get_population_summary()

        return info

    def get_source_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Get RA and Dec positions as numpy arrays"""
        if self.catalog_df is None:
            raise CatalogError("No catalog loaded")

        ra_col = self.config.astrometry["ra"]
        dec_col = self.config.astrometry["dec"]

        if self.backend == "polars":
            ra = self.catalog_df[ra_col].to_numpy()
            dec = self.catalog_df[dec_col].to_numpy()
        else:
            ra = self.catalog_df[ra_col].values
            dec = self.catalog_df[dec_col].values

        return ra, dec

    def print_catalog_summary(self) -> None:
        """Print a summary of the catalog"""
        info = self.get_catalog_info()

        print("=== Catalog Summary ===")
        print(f"File: {info['file_path']}")
        print(f"Backend: {info['backend']}")
        print(f"Sources: {info['n_sources']:,}")
        print(f"Columns: {info['n_columns']}")

        if "n_populations" in info:
            print(f"Populations: {info['n_populations']}")

            # Print population summary
            if "population_summary" in info:
                pop_summary = info["population_summary"]
                if len(pop_summary) > 0:
                    print("\nTop 10 populations by source count:")
                    if self.backend == "polars":
                        top_pops = pop_summary.sort("n_sources", descending=True).head(
                            10
                        )
                        for row in top_pops.iter_rows(named=True):
                            print(f"  {row['id_label']}: {row['n_sources']} sources")
                    else:
                        top_pops = pop_summary.nlargest(10, "n_sources")
                        for _, row in top_pops.iterrows():
                            print(f"  {row['id_label']}: {row['n_sources']} sources")


def load_catalog(catalog_config: CatalogConfig, backend: str = "auto") -> SkyCatalogs:
    """
    Convenience function to load a catalog

    Args:
        catalog_config: Configuration for catalog
        backend: Data backend to use

    Returns:
        Loaded SkyCatalogs instance
    """
    catalogs = SkyCatalogs(catalog_config, backend=backend)
    catalogs.load_catalog()
    return catalogs


def load_cosmos_catalog_direct(
    catalog_path: Path, backend: str = "auto", **kwargs
) -> SkyCatalogs:
    """
    Convenience function to load COSMOS catalog directly with optimized backend

    Args:
        catalog_path: Path to COSMOSweb_master.fits file
        backend: Data backend ("auto", "polars", "pandas") - auto defaults to polars for large files
        **kwargs: Additional arguments for COSMOS loader

    Returns:
        Loaded SkyCatalogs instance with COSMOS data
    """
    return create_cosmos_sky_catalog(catalog_path, backend=backend, **kwargs)
