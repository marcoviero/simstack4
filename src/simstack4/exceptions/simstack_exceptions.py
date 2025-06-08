"""
Complete exception classes for Simstack4
"""


class SimstackError(Exception):
    """Base exception for Simstack4 errors"""
    pass


class ConfigError(SimstackError):
    """Configuration related errors"""
    pass


class CatalogError(SimstackError):
    """Catalog loading and processing errors"""
    pass


class CatalogValidationError(CatalogError):
    """Catalog validation specific errors"""
    pass


class MapError(SimstackError):
    """Map loading and processing errors"""
    pass


class MapValidationError(MapError):
    """Map validation specific errors"""
    pass


class PopulationError(SimstackError):
    """Population management errors"""
    pass


class AlgorithmError(SimstackError):
    """Stacking algorithm errors"""
    pass


class StackingError(SimstackError):
    """General stacking errors (alias for AlgorithmError)"""
    pass


class ValidationError(SimstackError):
    """General validation errors"""
    pass


class CosmologyError(SimstackError):
    """Cosmological calculation errors"""
    pass


class ResultsError(SimstackError):
    """Results processing errors"""
    pass


class PlotError(SimstackError):
    """Plotting and visualization errors"""
    pass


class ToolboxError(SimstackError):
    """Toolbox function errors"""
    pass