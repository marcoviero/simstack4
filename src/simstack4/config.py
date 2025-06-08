"""
Configuration management for Simstack4

This module handles loading and validating configuration files in TOML format,
replacing the INI format used in simstack3.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import tomllib
import os
from dataclasses import dataclass, field
from enum import Enum

from .exceptions.simstack_exceptions import ConfigError


class SplitType(Enum):
    """Enum for different catalog splitting methods"""
    LABELS = "labels"
    UVJ = "uvj"
    NUVRJ = "nuvrj"


class Cosmology(Enum):
    """Available cosmology models"""
    PLANCK15 = "Planck15"
    PLANCK18 = "Planck18"


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
    iterations: int = 150
    initial_seed: int = 1


@dataclass
class ErrorConfig:
    """Configuration for error estimation methods"""
    bootstrap: BootstrapConfig = field(default_factory=BootstrapConfig)
    write_simmaps: bool = False
    randomize: bool = False


@dataclass
class BeamConfig:
    """Configuration for telescope beam properties"""
    fwhm: float  # arcsec
    area: float  # solid angle (1.0 if already in Jy/beam)


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
class ClassificationBins:
    """Configuration for classification binning"""
    id: str  # Column name in catalog
    bins: List[float]  # Bin edges


@dataclass
class SplitParams:
    """Parameters for UVJ/NUVRJ splitting"""
    id: str  # Label for this split
    bins: Dict[str, str]  # e.g., {"U-V": "rf_U_V", "V-J": "rf_V_J"}


@dataclass
class ClassificationConfig:
    """Configuration for catalog classification and binning"""
    split_type: SplitType
    redshift: ClassificationBins
    stellar_mass: ClassificationBins
    split_params: Optional[SplitParams] = None


@dataclass
class CatalogConfig:
    """Configuration for catalog data"""
    path: str
    file: str
    astrometry: Dict[str, str]  # {"ra": "ra", "dec": "dec"}
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
    maps: Dict[str, MapConfig]

    @classmethod
    def from_toml(cls, config_path: Union[str, Path]) -> 'SimstackConfig':
        """Load configuration from TOML file"""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, 'rb') as f:
                config_dict = tomllib.load(f)
        except Exception as e:
            raise ConfigError(f"Failed to parse TOML file: {e}")

        return cls._from_dict(config_dict)

    """
    Fixed configuration parser that matches your exact TOML structure

    Replace the _from_dict method in your config.py with this version
    """

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'SimstackConfig':
        """Create configuration from dictionary (CORRECTED VERSION)"""
        try:
            # Parse binning config
            binning_dict = config_dict.get('binning', {})
            binning = BinningConfig(
                stack_all_z_at_once=binning_dict.get('stack_all_z_at_once', True),
                add_foreground=binning_dict.get('add_foreground', True),
                crop_circles=binning_dict.get('crop_circles', True)
            )

            # Parse error estimator config
            error_dict = config_dict.get('error_estimator', {})

            # Parse bootstrap config - only extract the expected keys
            bootstrap_dict = error_dict.get('bootstrap', {})
            bootstrap = BootstrapConfig(
                enabled=bootstrap_dict.get('enabled', True),
                iterations=bootstrap_dict.get('iterations', 150),
                initial_seed=bootstrap_dict.get('initial_seed', 1)
            )

            error_estimator = ErrorConfig(
                bootstrap=bootstrap,
                write_simmaps=error_dict.get('write_simmaps', False),
                randomize=error_dict.get('randomize', False)
            )

            # Parse cosmology - TOP-LEVEL key in your TOML
            cosmology_str = config_dict.get('cosmology', 'Planck18')
            try:
                cosmology = Cosmology(cosmology_str)
            except ValueError:
                raise ConfigError(f"Unknown cosmology: {cosmology_str}")

            # Parse output config
            output_dict = config_dict.get('output', {})
            output = OutputConfig(
                folder=output_dict.get('folder', './output'),
                shortname=output_dict.get('shortname', 'simstack_run')
            )

            # Parse catalog config
            catalog_dict = config_dict.get('catalog', {})
            if not catalog_dict:
                raise ConfigError("Catalog configuration is required")

            classification_dict = catalog_dict.get('classification', {})
            if not classification_dict:
                raise ConfigError("Classification configuration is required")

            # Parse classification config
            split_type_str = classification_dict.get('split_type', 'labels')
            try:
                split_type = SplitType(split_type_str)
            except ValueError:
                raise ConfigError(f"Unknown split_type: {split_type_str}")

            # Parse redshift bins
            redshift_dict = classification_dict.get('redshift', {})
            if not redshift_dict:
                raise ConfigError("Redshift configuration is required")

            redshift = ClassificationBins(
                id=redshift_dict.get('id', 'z'),
                bins=redshift_dict.get('bins', [])
            )

            # Parse stellar mass bins
            stellar_mass_dict = classification_dict.get('stellar_mass', {})
            if not stellar_mass_dict:
                raise ConfigError("Stellar mass configuration is required")

            stellar_mass = ClassificationBins(
                id=stellar_mass_dict.get('id', 'stellar_mass'),
                bins=stellar_mass_dict.get('bins', [])
            )

            # Parse split params (optional)
            split_params = None
            if 'split_params' in classification_dict:
                split_params_dict = classification_dict['split_params']
                split_params = SplitParams(
                    id=split_params_dict.get('id', 'split_id'),
                    bins=split_params_dict.get('bins', {})
                )

            classification = ClassificationConfig(
                split_type=split_type,
                redshift=redshift,
                stellar_mass=stellar_mass,
                split_params=split_params
            )

            # Parse astrometry
            astrometry = catalog_dict.get('astrometry', {})
            if not astrometry:
                raise ConfigError("Astrometry configuration is required")

            catalog = CatalogConfig(
                path=catalog_dict.get('path', ''),
                file=catalog_dict.get('file', ''),
                astrometry=astrometry,
                classification=classification
            )

            # Parse maps config
            maps = {}
            maps_dict = config_dict.get('maps', {})
            if not maps_dict:
                raise ConfigError("At least one map configuration is required")

            for map_name, map_config in maps_dict.items():
                if not isinstance(map_config, dict):
                    continue  # Skip non-dict entries

                # Parse beam config
                beam_dict = map_config.get('beam', {})
                if not beam_dict:
                    raise ConfigError(f"Beam configuration required for map {map_name}")

                beam = BeamConfig(
                    fwhm=beam_dict.get('fwhm', 6.0),
                    area=beam_dict.get('area', 1.0)
                )

                # Validate required map fields
                wavelength = map_config.get('wavelength')
                if wavelength is None:
                    raise ConfigError(f"Wavelength required for map {map_name}")

                path_map = map_config.get('path_map', '')
                if not path_map:
                    raise ConfigError(f"path_map required for map {map_name}")

                maps[map_name] = MapConfig(
                    wavelength=wavelength,
                    beam=beam,
                    color_correction=map_config.get('color_correction', 1.0),
                    path_map=path_map,
                    path_noise=map_config.get('path_noise', '')
                )

            return cls(
                binning=binning,
                error_estimator=error_estimator,
                cosmology=cosmology,
                output=output,
                catalog=catalog,
                maps=maps
            )

        except ConfigError:
            # Re-raise ConfigError as-is
            raise
        except Exception as e:
            raise ConfigError(f"Failed to parse configuration: {e}")

    def validate(self) -> None:
        """Validate configuration settings (IMPROVED VERSION)"""
        errors = []

        # Check that catalog file exists (if path is set)
        if self.catalog.path and self.catalog.file:
            catalog_path = self.catalog.full_path
            if not catalog_path.exists():
                # Don't fail validation if using environment variables that aren't set yet
                if '$' not in str(catalog_path):
                    errors.append(f"Catalog file not found: {catalog_path}")

        # Check that map files exist (if paths are set)
        for map_name, map_config in self.maps.items():
            if map_config.path_map:
                map_path = Path(map_config.path_map)
                if not map_path.exists():
                    # Don't fail validation if using environment variables
                    if '$' not in map_config.path_map:
                        errors.append(f"Map file not found for {map_name}: {map_path}")

            if map_config.path_noise:
                noise_path = Path(map_config.path_noise)
                if not noise_path.exists():
                    if '$' not in map_config.path_noise:
                        errors.append(f"Noise file not found for {map_name}: {noise_path}")

        # Check output directory can be created
        try:
            output_path = Path(self.output.folder)
            # Don't try to create if it has environment variables
            if '$' not in self.output.folder:
                output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            if '$' not in self.output.folder:
                errors.append(f"Cannot create output directory: {e}")

        # Validate bin configurations
        if len(self.catalog.classification.redshift.bins) < 2:
            errors.append("At least 2 redshift bin edges required")

        if len(self.catalog.classification.stellar_mass.bins) < 2:
            errors.append("At least 2 stellar mass bin edges required")

        # Check bins are in ascending order
        z_bins = self.catalog.classification.redshift.bins
        if z_bins != sorted(z_bins):
            errors.append("Redshift bins must be in ascending order")

        m_bins = self.catalog.classification.stellar_mass.bins
        if m_bins != sorted(m_bins):
            errors.append("Stellar mass bins must be in ascending order")

        if errors:
            raise ConfigError("Configuration validation failed:\n" + "\n".join(f"  - {error}" for error in errors))


def load_config(config_path: Union[str, Path]) -> SimstackConfig:
    """Convenience function to load and validate configuration"""
    config = SimstackConfig.from_toml(config_path)
    config.validate()
    return config