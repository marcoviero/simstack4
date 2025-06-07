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

    @classmethod
    def _from_dict(cls, config_dict: Dict[str, Any]) -> 'SimstackConfig':
        """Create configuration from dictionary"""
        try:
            # Parse binning config
            binning_dict = config_dict.get('binning', {})
            binning = BinningConfig(**binning_dict)

            # Parse error estimator config
            error_dict = config_dict.get('error_estimator', {})
            bootstrap_dict = error_dict.get('bootstrap', {})
            bootstrap = BootstrapConfig(**bootstrap_dict)
            error_estimator = ErrorConfig(
                bootstrap=bootstrap,
                write_simmaps=error_dict.get('write_simmaps', False),
                randomize=error_dict.get('randomize', False)
            )

            # Parse cosmology
            cosmology_str = config_dict.get('cosmology', 'Planck18')
            cosmology = Cosmology(cosmology_str)

            # Parse output config
            output_dict = config_dict.get('output', {})
            output = OutputConfig(**output_dict)

            # Parse catalog config
            catalog_dict = config_dict.get('catalog', {})
            classification_dict = catalog_dict.get('classification', {})

            # Parse classification config
            split_type = SplitType(classification_dict.get('split_type', 'labels'))

            redshift_dict = classification_dict.get('redshift', {})
            redshift = ClassificationBins(**redshift_dict)

            stellar_mass_dict = classification_dict.get('stellar_mass', {})
            stellar_mass = ClassificationBins(**stellar_mass_dict)

            split_params = None
            if 'split_params' in classification_dict:
                split_params_dict = classification_dict['split_params']
                split_params = SplitParams(**split_params_dict)

            classification = ClassificationConfig(
                split_type=split_type,
                redshift=redshift,
                stellar_mass=stellar_mass,
                split_params=split_params
            )

            catalog = CatalogConfig(
                path=catalog_dict.get('path', ''),
                file=catalog_dict.get('file', ''),
                astrometry=catalog_dict.get('astrometry', {}),
                classification=classification
            )

            # Parse maps config
            maps = {}
            maps_dict = config_dict.get('maps', {})
            for map_name, map_config in maps_dict.items():
                beam_dict = map_config.get('beam', {})
                beam = BeamConfig(**beam_dict)

                maps[map_name] = MapConfig(
                    wavelength=map_config.get('wavelength', 0.0),
                    beam=beam,
                    color_correction=map_config.get('color_correction', 1.0),
                    path_map=map_config.get('path_map', ''),
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

        except Exception as e:
            raise ConfigError(f"Failed to parse configuration: {e}")

    def validate(self) -> None:
        """Validate configuration settings"""
        errors = []

        # Check that catalog file exists
        if not self.catalog.full_path.exists():
            errors.append(f"Catalog file not found: {self.catalog.full_path}")

        # Check that map files exist
        for map_name, map_config in self.maps.items():
            map_path = Path(map_config.path_map)
            if not map_path.exists():
                errors.append(f"Map file not found for {map_name}: {map_path}")

            if map_config.path_noise:
                noise_path = Path(map_config.path_noise)
                if not noise_path.exists():
                    errors.append(f"Noise file not found for {map_name}: {noise_path}")

        # Check output directory is writable
        output_path = Path(self.output.folder)
        try:
            output_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")

        if errors:
            raise ConfigError("Configuration validation failed:\n" + "\n".join(errors))


def load_config(config_path: Union[str, Path]) -> SimstackConfig:
    """Convenience function to load and validate configuration"""
    config = SimstackConfig.from_toml(config_path)
    config.validate()
    return config