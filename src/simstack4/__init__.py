"""
Simstack4: Simultaneous stacking code for astrophysical sources
"""

__version__ = "0.1.0"
__author__ = "Marco Viero"

from .config import SimstackConfig, load_config
from .populations import PopulationManager, PopulationBin
from .wrapper import SimstackWrapper
from .results import SimstackResults
from .utils import setup_logging, validate_environment

from .exceptions.simstack_exceptions import (
    SimstackError,
    ConfigError,
    CatalogError,
    MapError,
    CosmologyError,
    PopulationError,
    ValidationError,
)

__all__ = [
    # Core functionality
    "SimstackConfig",
    "load_config",
    "PopulationManager",
    "PopulationBin",
    "SimstackWrapper",
    "SimstackResults",
    # Utilities
    "setup_logging",
    "validate_environment",
    # Exceptions
    "SimstackError",
    "ConfigError",
    "CatalogError",
    "MapError",
    "CosmologyError",
    "PopulationError",
    "ValidationError",
]
