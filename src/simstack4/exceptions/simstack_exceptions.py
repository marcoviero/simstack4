"""
Custom exceptions for Simstack4
"""


class SimstackError(Exception):
    """Base exception for Simstack4 errors"""
    pass


class ConfigError(SimstackError):
    """Raised when there's an error in configuration"""
    pass


class CatalogError(SimstackError):
    """Raised when there's an error with catalog data"""
    pass


class MapError(SimstackError):
    """Raised when there's an error with map data"""
    pass


class StackingError(SimstackError):
    """Raised when there's an error during stacking process"""
    pass


class CosmologyError(SimstackError):
    """Raised when there's an error with cosmological calculations"""
    pass


class PopulationError(SimstackError):
    """Raised when there's an error with population management"""
    pass


class ValidationError(SimstackError):
    """Raised when data validation fails"""
    pass