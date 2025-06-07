"""
Simstack4: Simultaneous stacking code for astrophysical sources

An improved version of simstack3 with:
- Python 3.13 support
- Modern package management with uv
- TOML configuration files
- Improved population management system
- Support for polars/vaex for large datasets
- Better modular architecture
"""

__version__ = "0.1.0"
__author__ = "Marco Viero"

# Core classes
from .config import SimstackConfig, load_config
from .populations import PopulationManager, PopulationBin

# Import with fallback for development
try:
    from .wrapper import SimstackWrapper
    from .results import SimstackResults
    from .plots import SimstackPlots
    from .utils import setup_logging, validate_environment
except ImportError as e:
    import warnings
    warnings.warn(f"Some modules not fully implemented yet: {e}")

    # Create placeholder classes
    class SimstackWrapper:
        def __init__(self, *args, **kwargs):
            print("SimstackWrapper placeholder - not yet implemented")

    class SimstackResults:
        def __init__(self, *args, **kwargs):
            print("SimstackResults placeholder - not yet implemented")

    class SimstackPlots:
        def __init__(self, *args, **kwargs):
            print("SimstackPlots placeholder - not yet implemented")

    def setup_logging(*args, **kwargs):
        import logging
        return logging.getLogger("simstack4")

    def validate_environment():
        return {"status": "placeholder"}

# Exceptions
from .exceptions.simstack_exceptions import (
    SimstackError,
    ConfigError,
    CatalogError,
    MapError,
    StackingError,
    CosmologyError,
    PopulationError,
    ValidationError
)

__all__ = [
    # Core functionality
    "SimstackConfig",
    "load_config",
    "PopulationManager",
    "PopulationBin",
    "SimstackWrapper",
    "SimstackResults",
    "SimstackPlots",

    # Utilities
    "setup_logging",
    "validate_environment",

    # Exceptions
    "SimstackError",
    "ConfigError",
    "CatalogError",
    "MapError",
    "StackingError",
    "CosmologyError",
    "PopulationError",
    "ValidationError",
]