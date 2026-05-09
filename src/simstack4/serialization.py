"""
JSON serialization and deserialization utilities for SimstackWrapper.

These are module-level functions so they can be imported and tested
independently of SimstackWrapper.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .config import SimstackConfig

logger = logging.getLogger(__name__)


class EnhancedJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy arrays and common Python/astropy objects"""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer | np.int32 | np.int64):
            return int(obj)
        if isinstance(obj, np.floating | np.float32 | np.float64):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)

        if hasattr(obj, "to_dict"):
            try:
                return obj.to_dict()
            except (AttributeError, TypeError, ValueError):
                pass

        if hasattr(obj, "__name__"):
            return str(obj.__name__)
        if hasattr(obj, "name"):
            return str(obj.name)
        if hasattr(obj, "value"):
            return obj.value

        if hasattr(obj, "__dict__"):
            try:
                return {
                    k: v
                    for k, v in obj.__dict__.items()
                    if isinstance(v, str | int | float | bool | list | dict | type(None))
                }
            except (AttributeError, TypeError, ValueError):
                return str(obj)

        return str(obj)


def get_cosmology_enum(cosmology_str: str):
    """Convert a cosmology string (from JSON) to the Cosmology enum value."""
    from .config import Cosmology

    upper = cosmology_str.upper()
    if upper in ("PLANCK18",):
        return Cosmology.PLANCK18
    if upper in ("PLANCK15",):
        return Cosmology.PLANCK15
    logger.warning(f"Unknown cosmology string '{cosmology_str}', defaulting to Planck18")
    return Cosmology.PLANCK18


def serialize_config(config: SimstackConfig) -> dict:
    """Serialise a SimstackConfig to a JSON-compatible dict.

    Uses a depth-limited traversal so deeply nested dataclasses don't cause
    recursion errors.
    """

    def safe_serialize(obj, depth=0, max_depth=10):
        if depth > max_depth:
            return f"<max_depth_reached_{type(obj).__name__}>"
        if obj is None:
            return None
        if isinstance(obj, str | int | float | bool):
            return obj
        if isinstance(obj, list | tuple):
            return [safe_serialize(item, depth + 1, max_depth) for item in obj]
        if isinstance(obj, dict):
            return {str(k): safe_serialize(v, depth + 1, max_depth) for k, v in obj.items()}
        if hasattr(obj, "name") and isinstance(obj.name, str):
            return str(obj.name)
        if hasattr(obj, "value"):
            return obj.value
        if hasattr(obj, "__dict__"):
            result = {}
            for k, v in obj.__dict__.items():
                if not k.startswith("_"):
                    try:
                        result[str(k)] = safe_serialize(v, depth + 1, max_depth)
                    except (RecursionError, TypeError, AttributeError):
                        result[str(k)] = str(type(v).__name__)
            return result
        try:
            return str(obj)
        except (TypeError, AttributeError):
            return f"<{type(obj).__name__}>"

    config_dict: dict = {}

    try:
        logger.info("Serializing configuration sections...")

        if hasattr(config, "catalog") and config.catalog:
            catalog_dict: dict = {}
            catalog_dict["path"] = getattr(config.catalog, "path", "")
            catalog_dict["file"] = getattr(config.catalog, "file", "")

            if hasattr(config.catalog, "astrometry"):
                catalog_dict["astrometry"] = dict(config.catalog.astrometry)

            if hasattr(config.catalog, "classification"):
                classification = config.catalog.classification
                classification_dict: dict = {}

                if hasattr(classification, "split_type"):
                    classification_dict["split_type"] = (
                        str(classification.split_type) if classification.split_type else None
                    )
                if hasattr(classification, "split_params"):
                    classification_dict["split_params"] = safe_serialize(
                        classification.split_params, 0, 5
                    )
                if hasattr(classification, "binning"):
                    binning_dict = {}
                    for bin_name, bin_cfg in classification.binning.items():
                        binning_dict[bin_name] = {
                            "id": getattr(bin_cfg, "id", bin_name),
                            "label": getattr(bin_cfg, "label", bin_name),
                            "bins": getattr(bin_cfg, "bins", []),
                            "formula_ref": getattr(bin_cfg, "formula_ref", None),
                        }
                    classification_dict["binning"] = binning_dict
                if hasattr(classification, "formulas"):
                    classification_dict["formulas"] = safe_serialize(
                        classification.formulas, 0, 3
                    )
                catalog_dict["classification"] = classification_dict

            config_dict["catalog"] = catalog_dict

        if hasattr(config, "maps") and config.maps:
            maps_dict = {}
            for map_name, map_cfg in config.maps.items():
                maps_dict[map_name] = {
                    "wavelength": getattr(map_cfg, "wavelength", 0),
                    "path_map": getattr(map_cfg, "path_map", ""),
                    "path_noise": getattr(map_cfg, "path_noise", ""),
                    "color_correction": getattr(map_cfg, "color_correction", 1.0),
                    "beam": {
                        "fwhm": (
                            getattr(map_cfg.beam, "fwhm", 0) if hasattr(map_cfg, "beam") else 0
                        ),
                        "area_sr": (
                            getattr(map_cfg.beam, "area_sr", 1.0)
                            if hasattr(map_cfg, "beam")
                            else 1.0
                        ),
                    },
                }
            config_dict["maps"] = maps_dict

        for attr in ("cosmology", "output", "binning", "error_estimator"):
            if hasattr(config, attr):
                try:
                    config_dict[attr] = safe_serialize(getattr(config, attr), 0, 5)
                except (RecursionError, TypeError, AttributeError) as e:
                    logger.warning(f"Failed to serialize {attr}: {e}")
                    config_dict[attr] = f"<serialization_failed: {e}>"

        logger.info("Configuration serialization completed")

    except (AttributeError, TypeError, ValueError, RecursionError) as e:
        logger.warning(f"Config serialization failed: {e}")
        config_dict = {
            "serialization_error": str(e),
            "config_type": str(type(config).__name__),
            "fallback_data": {
                "cosmology": (
                    getattr(config, "cosmology", "Planck18") if hasattr(config, "cosmology") else "Planck18"
                ),
                "catalog_file": (
                    getattr(config.catalog, "file", "") if hasattr(config, "catalog") else ""
                ),
                "n_maps": len(config.maps) if hasattr(config, "maps") else 0,
            },
        }

    return config_dict


def reconstruct_config_from_json(embedded_config: dict) -> SimstackConfig:
    """Reconstruct a SimstackConfig from the dict that was produced by serialize_config."""
    from .config import (
        BeamConfig,
        BinConfig,
        BootstrapConfig,
        CatalogConfig,
        ClassificationConfig,
        ErrorConfig,
        MapConfig,
        OutputConfig,
        SimstackConfig,
        SplitParams,
        SplitType,
    )
    from .exceptions import SimstackError

    try:
        catalog_data = embedded_config.get("catalog", {})
        astrometry_config = catalog_data.get("astrometry", {})
        classification_data = catalog_data.get("classification", {})

        binning_config: dict = {}
        for bin_name, bin_cfg_dict in classification_data.get("binning", {}).items():
            if not isinstance(bin_cfg_dict, dict):
                continue
            binning_config[bin_name] = BinConfig(
                id=bin_cfg_dict.get("id", bin_name),
                label=bin_cfg_dict.get("label", bin_name.replace("_", " ").title()),
                bins=bin_cfg_dict.get("bins", []),
                formula_ref=bin_cfg_dict.get("formula_ref"),
            )

        _split_type_str = classification_data.get("split_type")
        _split_type = None
        if _split_type_str:
            _split_map = (
                {st.value: st for st in SplitType}
                | {st.name: st for st in SplitType}
                | {str(st): st for st in SplitType}
            )
            _split_type = _split_map.get(_split_type_str)
            if _split_type is None:
                raise ValueError(f"Cannot parse split_type from JSON: {_split_type_str!r}")

        _sp_dict = classification_data.get("split_params")
        _split_params = None
        if isinstance(_sp_dict, dict):
            _split_params = SplitParams(
                id=_sp_dict.get("id", "population_split"),
                formula=_sp_dict.get("formula"),
                bins=_sp_dict.get("bins", {}),
            )

        classification_config = ClassificationConfig(
            split_type=_split_type,
            binning=binning_config,
            split_params=_split_params,
            formulas=classification_data.get("formulas"),
        )

        catalog_config = CatalogConfig(
            path=catalog_data.get("path", ""),
            file=catalog_data.get("file", ""),
            astrometry=astrometry_config,
            classification=classification_config,
        )

        maps_config: dict = {}
        for map_name, map_data in embedded_config.get("maps", {}).items():
            beam_data = map_data.get("beam", {})
            maps_config[map_name] = MapConfig(
                wavelength=map_data.get("wavelength", 0),
                path_map=map_data.get("path_map", ""),
                path_noise=map_data.get("path_noise", ""),
                color_correction=map_data.get("color_correction", 1.0),
                beam=BeamConfig(
                    fwhm=beam_data.get("fwhm", 0),
                    area_sr=beam_data.get("area_sr", 1.0),
                ),
            )

        output_data = embedded_config.get("output", {})
        output_config = OutputConfig(
            folder=output_data.get("folder", ""),
            shortname=output_data.get("shortname", ""),
        )

        binning_data = embedded_config.get("binning", {})
        binning_config_dict = {
            "stack_all_z_at_once": binning_data.get("stack_all_z_at_once", True),
            "add_foreground": binning_data.get("add_foreground", True),
            "crop_circles": binning_data.get("crop_circles", True),
        }

        error_est_data = embedded_config.get("error_estimator", {})
        bootstrap_data = error_est_data.get("bootstrap", {})
        bootstrap_config = BootstrapConfig(
            enabled=bootstrap_data.get("enabled", True),
            iterations=bootstrap_data.get("iterations", 5),
            initial_seed=bootstrap_data.get("initial_seed", 1),
        )
        error_estimator_config = ErrorConfig(
            write_simmaps=error_est_data.get("write_simmaps", False),
            randomize=error_est_data.get("randomize", False),
            bootstrap=bootstrap_config,
        )

        return SimstackConfig(
            binning=binning_config_dict,
            error_estimator=error_estimator_config,
            cosmology=get_cosmology_enum(embedded_config.get("cosmology", "Planck18")),
            output=output_config,
            catalog=catalog_config,
            maps=maps_config,
        )

    except (AttributeError, TypeError, KeyError, ValueError) as e:
        raise SimstackError(f"Config reconstruction failed: {e}") from e


def validate_saved_json(filepath: Path) -> None:
    """Warn if the saved JSON is missing keys needed for standalone analysis."""
    try:
        with open(filepath) as f:
            data = json.load(f)

        required = [
            "flux_densities", "flux_errors", "map_names", "population_labels",
            "catalog_metadata", "populations", "population_details",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            logger.warning(f"JSON validation warning - missing keys: {missing}")

        if not data.get("catalog_metadata", {}).get("catalog_info", {}).get("n_populations", 0):
            logger.warning("JSON validation warning - no population metadata found")
        if not data.get("populations"):
            logger.warning("JSON validation warning - no population data found")

        logger.info("JSON validation completed")

    except (OSError, ValueError) as e:
        logger.warning(f"JSON validation failed: {e}")
