"""
Configuration management for Simstack4

This module handles loading and validating configuration files in TOML format,
replacing the INI format used in simstack3.
"""

import os
import tomllib
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from .exceptions.simstack_exceptions import ConfigError


class SplitType(Enum):
    """Enum for different catalog splitting methods (expanded)"""

    LABELS = "labels"  # Use existing column values
    FORMULA = "formula"  # Calculate from custom formula
    UVJ = "uvj"  # Built-in UVJ classification
    NUVRJ = "nuvrj"  # Built-in NUVRJ classification


class Cosmology(Enum):
    """Available cosmology models"""

    PLANCK15 = "Planck15"
    PLANCK18 = "Planck18"


@dataclass
class BinConfig:
    """Generic binning configuration for any variable"""

    id: str  # Column name in catalog
    label: str  # Human-readable label
    bins: list[float]  # Bin edges
    formula_ref: str | None = None  # Reference to formula for calculated variables


@dataclass
class BinningConfig:
    """Configuration for stacking binning options"""

    stack_all_z_at_once: bool = True
    add_foreground: bool = True
    crop_circles: bool = True


@dataclass
class BootstrapConfig:
    """Configuration for bootstrap error estimation"""

    enabled: bool = True
    method: str = "all_bins"  # "all_bins" or "per_bin"
    iterations: int = 150
    split_fraction: float = 0.5
    initial_seed: int = 1


@dataclass
class ErrorConfig:
    """Configuration for error estimation methods"""

    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    write_simmaps: bool = False
    randomize: bool = False


@dataclass
class BeamConfig:
    """Beam configuration with optional explicit beam area and PSF file"""

    fwhm: float
    area_sr: float | None = None  # Make this optional
    psf_file: str | None = None   # Optional path to measured PSF FITS file

    def __post_init__(self):
        """Expand environment variables in psf_file path"""
        if self.psf_file:
            self.psf_file = os.path.expandvars(self.psf_file)

    def get_beam_area_sr(self) -> float:
        """
        Returns:
        - If area_sr specified: return area_sr (for conversion)
        - If area_sr is None: calculate from FWHM assuming Gaussian beam
        """
        if self.area_sr is not None:
            return self.area_sr
        else:
            # Calculate beam area from FWHM for Gaussian beam
            # Area = 1.133 * (FWHM_arcsec)^2 * (π/180/3600)^2 steradians
            fwhm_arcsec = self.fwhm
            beam_area_arcsec2 = 1.133 * fwhm_arcsec**2
            beam_area_sr = beam_area_arcsec2 * (np.pi / (180 * 3600)) ** 2
            return beam_area_sr


@dataclass
class MapConfig:
    """Configuration for individual maps"""

    wavelength: float  # microns
    beam: BeamConfig
    color_correction: float = 1.0
    path_map: str = ""
    path_noise: str = ""

    def __post_init__(self):
        """Expand environment variables in paths"""
        self.path_map = os.path.expandvars(self.path_map)
        self.path_noise = os.path.expandvars(self.path_noise)


@dataclass
class SplitParams:
    """Parameters for population splitting (generalized)"""

    id: str  # Column name OR formula identifier
    formula: str | None = None  # Formula expression or built-in name
    bins: dict[str, str] | None = None  # Input columns for formulas (e.g., colors)


@dataclass
class FormulaParams:
    """Parameters for calculated binning variables (like β_UV)"""

    formula: str  # Formula type: "beta_uv", "custom", etc.
    bins: dict[str, str] | None = None  # Input columns for formulas
    single_law: str | None = None  # For dust law selection
    custom_expression: str | None = None  # For custom formulas


@dataclass
class ClassificationConfig:
    """Configuration for catalog classification and binning"""

    binning: dict[str, BinConfig]  # Multiple binning dimensions
    split_type: SplitType | None = None  # Optional population splitting
    split_params: SplitParams | None = None  # Only if split_type is specified
    formulas: dict[str, FormulaParams] | None = None  # Calculation formulas
    bin_property_columns: list[str] | None = None  # Extra columns to summarize per bin

    @property
    def all_property_columns(self) -> list[str]:
        """
        All columns to summarize per bin: binning axis IDs + explicit extras.

        Binning axes are always included automatically so the user
        doesn't need to list them again in bin_property_columns.
        """
        # Start with binning axis column IDs (always included)
        cols = [bc.id for bc in self.binning.values()]
        # Add explicit extras (deduped, order-preserving)
        if self.bin_property_columns:
            for c in self.bin_property_columns:
                if c not in cols:
                    cols.append(c)
        return cols


@dataclass
class CatalogConfig:
    """Configuration for catalog data"""

    path: str
    file: str
    astrometry: dict[str, str]  # {"ra": "ra", "dec": "dec"}
    classification: ClassificationConfig

    def __post_init__(self):
        """Expand environment variables in path"""
        self.path = os.path.expandvars(self.path)

    @property
    def full_path(self) -> Path:
        """Get full path to catalog file"""
        return Path(self.path) / self.file


@dataclass
class OutputConfig:
    """Configuration for output settings"""

    folder: str
    shortname: str

    def __post_init__(self):
        """Expand environment variables in folder path"""
        self.folder = os.path.expandvars(self.folder)


@dataclass
class SimstackConfig:
    """Main configuration class for Simstack4"""

    binning: BinningConfig
    error_estimator: ErrorConfig
    cosmology: Cosmology
    output: OutputConfig
    catalog: CatalogConfig
    maps: dict[str, MapConfig]

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "SimstackConfig":
        """Load configuration from TOML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "rb") as f:
                config_dict = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to parse TOML file: {e}") from e

        return cls._from_dict(config_dict)

    """
    Fixed configuration parser that matches your exact TOML structure

    Replace the _from_dict method in your config.py with this version
    """

    # Replace the classification parsing section in your config.py _from_dict method

    @classmethod
    def _from_dict(cls, config_dict: dict[str, Any]) -> "SimstackConfig":
        """Create configuration from dictionary (UPDATED for generalized binning)"""
        try:
            # Parse binning config (unchanged)
            binning_dict = config_dict.get("binning", {})
            binning = BinningConfig(
                stack_all_z_at_once=binning_dict.get("stack_all_z_at_once", True),
                add_foreground=binning_dict.get("add_foreground", True),
                crop_circles=binning_dict.get("crop_circles", True),
            )

            # Parse error estimator config (unchanged)
            error_dict = config_dict.get("error_estimator", {})
            bootstrap_dict = error_dict.get("bootstrap", {})
            bootstrap = BootstrapConfig(
                enabled=bootstrap_dict.get("enabled", True),
                method=bootstrap_dict.get("method", "all_bins"),
                iterations=bootstrap_dict.get("iterations", 150),
                split_fraction=bootstrap_dict.get("split_fraction", 0.5),
                initial_seed=bootstrap_dict.get("initial_seed", 1),
            )
            error_estimator = ErrorConfig(
                bootstrap=bootstrap,
                write_simmaps=error_dict.get("write_simmaps", False),
                randomize=error_dict.get("randomize", False),
            )

            # Parse cosmology (unchanged)
            cosmology_str = config_dict.get("cosmology", "Planck18")
            try:
                cosmology = Cosmology(cosmology_str)
            except ValueError as e:
                raise ConfigError(f"Unknown cosmology: {cosmology_str}") from e

            # Parse output config (unchanged)
            output_dict = config_dict.get("output", {})
            output = OutputConfig(
                folder=output_dict.get("folder", "./output"),
                shortname=output_dict.get("shortname", "simstack_run"),
            )

            # Parse catalog config
            catalog_dict = config_dict.get("catalog", {})
            if not catalog_dict:
                raise ConfigError("Catalog configuration is required")

            classification_dict = catalog_dict.get("classification", {})
            if not classification_dict:
                raise ConfigError("Classification configuration is required")

            # ===== UPDATED CLASSIFICATION PARSING =====

            # Parse split type (NOW OPTIONAL)
            split_type = None
            split_params = None

            if "split_type" in classification_dict:
                split_type_str = classification_dict["split_type"]
                try:
                    split_type = SplitType(split_type_str)
                except ValueError:
                    raise ConfigError(f"Unknown split_type: {split_type_str}") from None

                # Parse split params only if split_type is specified
                if "split_params" in classification_dict:
                    split_params_dict = classification_dict["split_params"]
                    split_params = SplitParams(
                        id=split_params_dict.get("id", "population_split"),
                        formula=split_params_dict.get("formula"),
                        bins=split_params_dict.get("bins", {}),
                    )

            # Parse formula configurations (NEW)
            formulas = {}
            for key, value in classification_dict.items():
                if key.endswith("_formula"):
                    formula_name = key.replace("_formula", "")
                    formulas[formula_name] = FormulaParams(
                        formula=value.get("formula", formula_name),
                        bins=value.get("bins", {}),
                        single_law=value.get("single_law"),
                        custom_expression=value.get("custom_expression"),
                    )

            # Parse binning configurations
            binning_config = {}
            if "binning" in classification_dict:
                binning_dict = classification_dict["binning"]
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

                    # Validate bins
                    if len(bin_config.bins) < 2:
                        raise ConfigError(
                            f"At least 2 bin edges required for {bin_name}"
                        )

                    if bin_config.bins != sorted(bin_config.bins):
                        raise ConfigError(
                            f"Bin edges must be in ascending order for {bin_name}"
                        )

                    binning_config[bin_name] = bin_config
            else:
                raise ConfigError("Binning configuration is required")

            # Create classification config
            classification = ClassificationConfig(
                split_type=split_type,  # Can be None
                binning=binning_config,
                split_params=split_params,  # Can be None
                formulas=formulas if formulas else None,
                bin_property_columns=classification_dict.get(
                    "bin_property_columns", None
                ),
            )

            # Parse astrometry (unchanged)
            astrometry = catalog_dict.get("astrometry", {})
            if not astrometry:
                raise ConfigError("Astrometry configuration is required")

            catalog = CatalogConfig(
                path=catalog_dict.get("path", ""),
                file=catalog_dict.get("file", ""),
                astrometry=astrometry,
                classification=classification,
            )

            # Parse maps config (unchanged)
            maps = {}
            maps_dict = config_dict.get("maps", {})
            if not maps_dict:
                raise ConfigError("At least one map configuration is required")

            for map_name, map_config in maps_dict.items():
                if not isinstance(map_config, dict):
                    continue

                beam_dict = map_config.get("beam", {})
                if not beam_dict:
                    raise ConfigError(f"Beam configuration required for map {map_name}")

                beam = BeamConfig(
                    fwhm=beam_dict.get("fwhm", 6.0),
                    area_sr=beam_dict.get("area_sr", 1.0),
                    psf_file=beam_dict.get("psf_file", None),
                )

                wavelength = map_config.get("wavelength")
                if wavelength is None:
                    raise ConfigError(f"Wavelength required for map {map_name}")

                path_map = map_config.get("path_map")
                if not path_map:
                    raise ConfigError(f"path_map required for map {map_name}")

                maps[map_name] = MapConfig(
                    wavelength=wavelength,
                    beam=beam,
                    color_correction=map_config.get("color_correction", 1.0),
                    path_map=path_map,
                    path_noise=map_config.get("path_noise", ""),
                )

            return cls(
                binning=binning,
                error_estimator=error_estimator,
                cosmology=cosmology,
                output=output,
                catalog=catalog,
                maps=maps,
            )

        except ConfigError:
            raise
        except Exception as e:
            raise ConfigError(f"Failed to parse configuration: {e}") from e

    def validate(self) -> None:
        """Validate configuration settings (FIXED for generalized binning)"""
        errors = []

        # Check that catalog file exists (unchanged)
        if self.catalog.path and self.catalog.file:
            catalog_path = self.catalog.full_path
            if not catalog_path.exists():
                if "$" not in str(catalog_path):
                    errors.append(f"Catalog file not found: {catalog_path}")

        # Check that map files exist (unchanged)
        for map_name, map_config in self.maps.items():
            if map_config.path_map:
                map_path = Path(map_config.path_map)
                if not map_path.exists():
                    if "$" not in map_config.path_map:
                        errors.append(f"Map file not found for {map_name}: {map_path}")

            if map_config.path_noise:
                noise_path = Path(map_config.path_noise)
                if not noise_path.exists():
                    if "$" not in map_config.path_noise:
                        errors.append(
                            f"Noise file not found for {map_name}: {noise_path}"
                        )

        # Check output directory can be created (unchanged)
        try:
            output_path = Path(self.output.folder)
            if "$" not in self.output.folder:
                output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if "$" not in self.output.folder:
                errors.append(f"Cannot create output directory: {e}")

        # FIXED: Validate bin configurations for generalized binning
        for bin_name, bin_config in self.catalog.classification.binning.items():
            if len(bin_config.bins) < 2:
                errors.append(f"At least 2 bin edges required for {bin_name}")

            if bin_config.bins != sorted(bin_config.bins):
                errors.append(f"Bin edges must be in ascending order for {bin_name}")

        if errors:
            raise ConfigError(
                "Configuration validation failed:\n"
                + "\n".join(f"  - {error}" for error in errors)
            )


def load_config(config_path: str | Path) -> SimstackConfig:
    """Convenience function to load and validate configuration"""
    config = SimstackConfig.from_toml(config_path)
    config.validate()
    return config
