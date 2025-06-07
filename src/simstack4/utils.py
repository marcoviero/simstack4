"""
Utility functions for Simstack4
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import warnings

from .exceptions.simstack_exceptions import ValidationError


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """
    Set up logging for Simstack4

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file to write logs to

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("simstack4")
    logger.setLevel(getattr(logging, level.upper()))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if requested
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_environment() -> Dict[str, Any]:
    """
    Validate that required environment variables are set

    Returns:
        Dictionary with environment status and paths

    Raises:
        ValidationError: If required environment variables are missing
    """
    required_vars = ["MAPSPATH", "CATSPATH", "PICKLESPATH"]
    env_status = {}
    missing_vars = []

    for var in required_vars:
        value = os.environ.get(var)
        if value:
            path = Path(value)
            env_status[var] = {
                "value": value,
                "exists": path.exists(),
                "is_dir": path.is_dir() if path.exists() else False,
                "writable": os.access(path, os.W_OK) if path.exists() else False
            }
        else:
            missing_vars.append(var)
            env_status[var] = {"value": None, "exists": False}

    if missing_vars:
        raise ValidationError(
            f"Missing required environment variables: {missing_vars}\n"
            f"Please set them in your shell profile (.zshrc for zsh):\n"
            f"export MAPSPATH=/path/to/your/maps/\n"
            f"export CATSPATH=/path/to/your/catalogs/\n"
            f"export PICKLESPATH=/path/to/your/pickles/"
        )

    return env_status


def memory_usage_gb() -> float:
    """Get current memory usage in GB"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 ** 3
    except ImportError:
        warnings.warn("psutil not available, cannot check memory usage")
        return 0.0


def estimate_memory_requirements(n_sources: int, n_populations: int,
                                 n_maps: int, map_size_mb: float = 100) -> Dict[str, float]:
    """
    Estimate memory requirements for stacking

    Args:
        n_sources: Number of sources in catalog
        n_populations: Number of population bins
        n_maps: Number of maps
        map_size_mb: Typical map size in MB

    Returns:
        Dictionary with memory estimates in GB
    """
    # Rough estimates based on simstack3 experience
    catalog_gb = n_sources * 50 * 1e-9  # ~50 bytes per source for typical catalog
    maps_gb = n_maps * map_size_mb / 1024  # Convert MB to GB
    layers_gb = n_populations * map_size_mb / 1024  # One layer per population

    # Bootstrap multiplies memory requirements
    bootstrap_multiplier = 2.0  # Conservative estimate

    base_memory = catalog_gb + maps_gb + layers_gb
    with_bootstrap = base_memory * bootstrap_multiplier

    return {
        "catalog_gb": catalog_gb,
        "maps_gb": maps_gb,
        "layers_gb": layers_gb,
        "base_total_gb": base_memory,
        "with_bootstrap_gb": with_bootstrap,
        "recommended_ram_gb": with_bootstrap * 1.5  # 50% buffer
    }


def check_disk_space(path: str, required_gb: float) -> bool:
    """
    Check if sufficient disk space is available

    Args:
        path: Path to check
        required_gb: Required space in GB

    Returns:
        True if sufficient space available
    """
    try:
        import shutil
        free_bytes = shutil.disk_usage(path).free
        free_gb = free_bytes / 1024 ** 3
        return free_gb >= required_gb
    except Exception:
        warnings.warn(f"Could not check disk space for {path}")
        return True  # Assume OK if we can't check


def format_time(seconds: float) -> str:
    """Format time duration in human-readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def format_memory(bytes_val: int) -> str:
    """Format memory size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024:
            return f"{bytes_val:.1f}{unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f}PB"


def create_output_directory(path: str) -> Path:
    """
    Create output directory with proper structure

    Args:
        path: Base output path

    Returns:
        Created directory path
    """
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for organization
    (output_path / "results").mkdir(exist_ok=True)
    (output_path / "plots").mkdir(exist_ok=True)
    (output_path / "logs").mkdir(exist_ok=True)

    return output_path


def validate_catalog_file(catalog_path: Path) -> Dict[str, Any]:
    """
    Validate catalog file and return basic info

    Args:
        catalog_path: Path to catalog file

    Returns:
        Dictionary with catalog information
    """
    if not catalog_path.exists():
        raise ValidationError(f"Catalog file not found: {catalog_path}")

    # Try to determine file format and basic properties
    file_size_mb = catalog_path.stat().st_size / 1024 ** 2

    try:
        # Try reading first few lines to determine format
        if catalog_path.suffix.lower() == '.csv':
            import pandas as pd
            # Read just the header and first row
            sample_df = pd.read_csv(catalog_path, nrows=1)
            n_columns = len(sample_df.columns)
            column_names = list(sample_df.columns)
        else:
            # For other formats, we'll need more specific handling
            n_columns = None
            column_names = []
    except Exception as e:
        warnings.warn(f"Could not read catalog file for validation: {e}")
        n_columns = None
        column_names = []

    return {
        "path": str(catalog_path),
        "exists": True,
        "size_mb": file_size_mb,
        "n_columns": n_columns,
        "column_names": column_names
    }


def print_system_info() -> None:
    """Print system information useful for debugging"""
    import platform
    import sys

    print("=== System Information ===")
    print(f"Python version: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.architecture()[0]}")
    print(f"Processor: {platform.processor()}")

    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total RAM: {format_memory(memory.total)}")
        print(f"Available RAM: {format_memory(memory.available)}")
        print(f"CPU cores: {psutil.cpu_count()}")
    except ImportError:
        print("psutil not available - install for detailed system info")

    print("\n=== Environment Variables ===")
    for var in ["MAPSPATH", "CATSPATH", "PICKLESPATH"]:
        value = os.environ.get(var, "NOT SET")
        print(f"{var}: {value}")

    print("\n=== Key Package Versions ===")
    packages = ["numpy", "pandas", "astropy", "matplotlib", "scipy"]
    for package in packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"{package}: {version}")
        except ImportError:
            print(f"{package}: NOT INSTALLED")