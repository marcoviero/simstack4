"""
Sky maps handling for Simstack4

This module handles loading, processing, and manipulating astronomical maps.
Supports FITS files with proper WCS handling, PSF convolution, and coordinate transformations.
"""
import pdb
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy import units as u
from astropy.convolution import Gaussian2DKernel, convolve, convolve_fft
from astropy.coordinates import SkyCoord

# Core dependencies
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS

from .config import MapConfig
from .exceptions.simstack_exceptions import MapError
from .utils import setup_logging

logger = setup_logging()


@dataclass
class MapData:
    """Container for map data and metadata"""

    data: np.ndarray
    noise: np.ndarray | None
    wcs: WCS
    header: fits.Header
    wavelength: float
    beam_fwhm: float
    beam_area: float
    color_correction: float
    map_name: str
    units: str = "Jy/beam"

    @property
    def shape(self) -> tuple[int, int]:
        """Get map dimensions"""
        return self.data.shape

    @property
    def pixel_scale(self) -> float:
        """Get pixel scale in arcsec"""
        if hasattr(self, "header") and self.header is not None:
            header = self.header

            # Check for CD matrix elements
            if "CD2_2" in header:
                # Use CD2_2 for Y-axis pixel scale (like simstack3)
                pix_scale = abs(header["CD2_2"]) * 3600.0
                # logger.info(f"pixel size = {pix_scale:.2e}")
                return pix_scale

            # Check for CDELT values
            elif "CDELT2" in header:
                # Use CDELT2 for Y-axis (like simstack3)
                pix_scale = abs(header["CDELT2"]) * 3600.0
                # logger.info(f"pixel size = {pix_scale:.2e}")
                return pix_scale

    @property
    def beam_fwhm_pixels(self) -> float:
        """Get beam FWHM in pixels"""
        return self.beam_fwhm / self.pixel_scale

    def get_rms_noise(self, mask: np.ndarray | None) -> float:
        """Calculate RMS noise in map"""
        if self.noise is not None:
            # Use provided noise map
            if mask is not None:
                return np.std(self.noise[mask])
            else:
                return np.std(self.noise)
        else:
            # Estimate from data using sigma clipping
            data_to_use = self.data if mask is None else self.data[mask]
            _, _, std = sigma_clipped_stats(data_to_use, sigma=3.0)
            return std


class SkyMaps:
    """
    Handle astronomical maps for stacking analysis

    This class manages loading, processing, and manipulating FITS maps,
    including coordinate transformations, PSF handling, and noise estimation.
    """

    def __init__(self, maps_config: dict[str, MapConfig]):
        """
        Initialize SkyMaps handler

        Args:
            maps_config: Dictionary of map configurations
        """
        self.config = maps_config
        self.maps: dict[str, MapData] = {}
        self._coordinate_grids = {}

        logger.info(f"SkyMaps initialized with {len(maps_config)} map configurations")

    def load_maps(self) -> None:
        """Load all configured maps"""
        logger.info("Loading maps...")

        for map_name, map_config in self.config.items():
            try:
                self._load_single_map(map_name, map_config)
                logger.info(f"✓ Loaded {map_name} ({self.maps[map_name].wavelength}μm)")
            except Exception as e:
                logger.error(f"✗ Failed to load {map_name}: {e}")
                raise MapError(f"Failed to load map {map_name}: {e}") from None

        logger.info(f"Successfully loaded {len(self.maps)} maps")
        self._validate_map_compatibility()

    def _load_single_map(self, map_name: str, map_config: MapConfig) -> None:
        """Load a single map from FITS file"""
        map_path = Path(map_config.path_map)

        if not map_path.exists():
            raise MapError(f"Map file not found: {map_path}")

        # Load main map
        with fits.open(map_path) as hdul:
            # Find the image HDU
            image_hdu = None
            for hdu in hdul:
                if hasattr(hdu, "data") and hdu.data is not None:
                    if len(hdu.data.shape) >= 2:  # 2D or higher
                        image_hdu = hdu
                        break

            if image_hdu is None:
                raise MapError(f"No valid image data found in {map_path}")

            # Get data and header
            data = image_hdu.data.copy()
            header = image_hdu.header.copy()

            # Handle multi-dimensional data (take first 2D slice)
            while len(data.shape) > 2:
                data = data[0]

            # Create WCS
            try:
                wcs = WCS(header, naxis=2)
            except Exception as e:
                logger.warning(f"WCS creation failed for {map_name}: {e}")
                # Create dummy WCS
                wcs = self._create_dummy_wcs(data.shape)
                pdb.set_trace()

        # Load noise map if specified
        noise_data = None
        if map_config.path_noise:
            noise_path = Path(map_config.path_noise)
            if noise_path.exists() and noise_path != map_path:
                # Separate noise file
                try:
                    with fits.open(noise_path) as noise_hdul:
                        noise_hdu = (
                            noise_hdul[0] if len(noise_hdul) == 1 else noise_hdul[1]
                        )
                        noise_data = noise_hdu.data.copy()
                        while len(noise_data.shape) > 2:
                            noise_data = noise_data[0]
                except Exception as e:
                    logger.warning(f"Failed to load noise map for {map_name}: {e}")
            else:
                # Same file - assume noise is in another HDU
                try:
                    with fits.open(map_path) as hdul:
                        if len(hdul) > 1:
                            noise_data = hdul[1].data.copy()
                            while len(noise_data.shape) > 2:
                                noise_data = noise_data[0]
                except Exception as e:
                    logger.warning(f"Failed to load noise from second HDU: {e}")

        # Get units from header
        units = header.get("BUNIT", "unknown")
        if "MJy/sr" in units:
            units = "MJy/sr"
        elif "Jy/beam" in units:
            units = "Jy/beam"
        elif "mJy/beam" in units:
            units = "mJy/beam"

        # beam_area_sr = map_config.beam.get_beam_area_sr()

        # Create MapData object
        map_data = MapData(
            data=data,
            noise=noise_data,
            wcs=wcs,
            header=header,
            wavelength=map_config.wavelength,
            beam_fwhm=map_config.beam.fwhm,
            beam_area=map_config.beam.area_sr,
            color_correction=map_config.color_correction,
            map_name=map_name,
            units=units,
        )

        # Convert units if needed
        self._convert_units(map_data)

        # Color correct if needed
        self._apply_color_correction(map_data)

        # Remove mean
        self._apply_mean_subtraction(map_data)

        # Store map
        self.maps[map_name] = map_data

    def _create_dummy_wcs(self, shape: tuple[int, int]) -> WCS:
        """Create a dummy WCS for maps without proper coordinates"""
        wcs = WCS(naxis=2)
        wcs.wcs.crpix = [shape[1] / 2, shape[0] / 2]
        wcs.wcs.crval = [0.0, 0.0]
        wcs.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]  # 1 arcsec pixels
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        return wcs

    def _convert_units(self, map_data: MapData) -> None:
        """Convert map units to Jy/beam if needed"""
        if ("MJy/sr" in map_data.units) or (abs(map_data.beam_area - 1.0) > 1e-10):
            beam_area_sr = map_data.beam_area  # From BeamConfig.get_beam_area_sr()
            # Only apply conversion if beam_area_sr != 1.0
            if abs(beam_area_sr - 1.0) > 1e-10:
                conversion_factor = beam_area_sr * 1e6
                map_data.data *= conversion_factor
                if map_data.noise is not None:
                    map_data.noise *= conversion_factor
                map_data.units = "Jy/beam"
                logger.info(
                    f"Converted {map_data.map_name}: MJy/sr → Jy/beam (factor: {conversion_factor:.2e})"
                )
            else:
                logger.info(
                    f"{map_data.map_name}: No conversion applied (calculated beam area)"
                )
                map_data.units = "Jy/beam"

        elif "mJy/beam" in map_data.units:
            # Convert mJy/beam to Jy/beam
            map_data.data *= 1e-3
            if map_data.noise is not None:
                map_data.noise *= 1e-3

            map_data.units = "Jy/beam"
            logger.debug(f"Converted {map_data.map_name} from mJy/beam to Jy/beam")

    def _apply_color_correction(self, map_data: MapData) -> None:
        """Apply color correction if needed"""
        if abs(map_data.color_correction - 1.0) > 1e-10:
            color_correction = map_data.color_correction
            map_data.data *= color_correction
            logger.info(
                f"Color corrected {map_data.map_name} (factor: {color_correction:.2e})"
            )

    def _apply_mean_subtraction(self, map_data: MapData) -> None:
        """Subtract mean from maps using only non-zero, non-NaN pixels"""
        # Identify signal pixels
        signal_mask = ~np.isnan(map_data.data) & (map_data.data != 0.0)
        n_signal_pixels = np.sum(signal_mask)

        if n_signal_pixels == 0:
            logger.warning(
                f"No signal pixels found in {map_data.map_name} - skipping mean subtraction"
            )
            return

        # Calculate and subtract mean from signal pixels only
        signal_mean = np.mean(map_data.data[signal_mask])
        map_data.data[signal_mask] -= signal_mean

        logger.info(
            f"Mean subtracted from {map_data.map_name}: "
            f"mean={signal_mean:.6e}, signal_pixels={n_signal_pixels}/{map_data.data.size} "
            f"({100 * n_signal_pixels / map_data.data.size:.1f}%)"
        )

    def _validate_map_compatibility(self) -> None:
        """Check that all maps are compatible for stacking"""
        if len(self.maps) < 2:
            return

        # Get reference map
        ref_map = next(iter(self.maps.values()))
        ref_shape = ref_map.shape
        # ref_wcs = ref_map.wcs

        for map_name, map_data in self.maps.items():
            # Check shapes are similar (allow small differences)
            if (
                abs(map_data.shape[0] - ref_shape[0]) > 10
                or abs(map_data.shape[1] - ref_shape[1]) > 10
            ):
                logger.warning(
                    f"Map {map_name} has significantly different shape: {map_data.shape} vs {ref_shape}"
                )

            # Check pixel scales are similar
            try:
                if (
                    abs(map_data.pixel_scale - ref_map.pixel_scale) > 0.5
                ):  # 0.5 arcsec tolerance
                    logger.warning(
                        f"Map {map_name} has different pixel scale: {map_data.pixel_scale:.2f} vs {ref_map.pixel_scale:.2f}"
                    )
            except Exception:
                pass

    def world_to_pixel(
        self, map_name: str, ra: np.ndarray, dec: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert sky coordinates to pixel coordinates for creating model layers

        Args:
            map_name: Name of the map
            ra: Right ascension in degrees
            dec: Declination in degrees

        Returns:
            x_pixel, y_pixel arrays
        """
        if map_name not in self.maps:
            raise MapError(f"Map '{map_name}' not loaded")

        map_data = self.maps[map_name]

        # Convert sky coordinates to pixel coordinates
        try:
            coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
            x_pix, y_pix = map_data.wcs.world_to_pixel(coords)
        except Exception as e:
            logger.warning(f"WCS transformation failed: {e}, using dummy coordinates")
            # Fallback: assume simple linear mapping centered on map
            x_pix = (ra - 0) * 3600 / map_data.pixel_scale + map_data.shape[1] / 2
            y_pix = (dec - 0) * 3600 / map_data.pixel_scale + map_data.shape[0] / 2

        return x_pix, y_pix

    def create_psf_kernel(self, map_name: str, normalize: bool = True) -> np.ndarray:
        """
        Create PSF kernel for the specified map

        Args:
            map_name: Name of the map
            normalize: Whether to normalize the kernel

        Returns:
            2D PSF kernel array
        """
        if map_name not in self.maps:
            raise MapError(f"Map '{map_name}' not loaded")

        map_data = self.maps[map_name]

        # Get beam FWHM in pixels
        beam_fwhm_pix = map_data.beam_fwhm_pixels

        # Convert FWHM to sigma
        sigma_pix = beam_fwhm_pix / (2 * np.sqrt(2 * np.log(2)))

        # Create Gaussian kernel
        # Kernel size should be at least 6*sigma, make it odd
        kernel_size = int(6 * sigma_pix)
        if kernel_size % 2 == 0:
            kernel_size += 1

        kernel = Gaussian2DKernel(
            x_stddev=sigma_pix,
            y_stddev=sigma_pix,
            x_size=kernel_size,
            y_size=kernel_size,
        )

        if normalize:
            # kernel = kernel.array / np.sum(kernel.array)
            kernel = kernel.array / np.max(kernel.array)
        else:
            kernel = kernel.array

        return kernel

    def convolve_with_psf(
        self, data: np.ndarray, map_name: str, method: str = "fft"
    ) -> np.ndarray:
        """
        Convolve data with map PSF

        Args:
            data: 2D array to convolve
            map_name: Name of map (for PSF properties)
            method: Convolution method ('fft' or 'direct')

        Returns:
            Convolved data array
        """
        kernel = self.create_psf_kernel(map_name)
        # logger.info(f"PSF sum: {np.sum(kernel)}")  # Should be 1.0

        if method == "fft":
            return convolve_fft(
                data,
                kernel,
                boundary="wrap",
                nan_treatment="interpolate",
                normalize_kernel=False,
            )
        else:
            return convolve(
                data,
                kernel,
                boundary="extend",
                nan_treatment="interpolate",
                normalize_kernel=False,
            )

    def get_map_summary(self) -> dict[str, Any]:
        """Get summary information about loaded maps"""
        summary = {
            "n_maps": len(self.maps),
            "wavelengths": [map_data.wavelength for map_data in self.maps.values()],
            "map_details": {},
        }

        for map_name, map_data in self.maps.items():
            summary["map_details"][map_name] = {
                "wavelength_um": map_data.wavelength,
                "shape": map_data.shape,
                "beam_fwhm_arcsec": map_data.beam_fwhm,
                "pixel_scale_arcsec": map_data.pixel_scale,
                "units": map_data.units,
                "has_noise": map_data.noise is not None,
                "rms_noise": map_data.get_rms_noise(),
            }

        return summary

    def print_map_summary(self) -> None:
        """Print a formatted summary of loaded maps"""
        summary = self.get_map_summary()

        print("=== Sky Maps Summary ===")
        print(f"Number of maps: {summary['n_maps']}")
        print(f"Wavelengths: {sorted(summary['wavelengths'])} μm")
        print()

        print("Map Details:")
        print(
            f"{'Name':<15} {'λ(μm)':<8} {'Shape':<12} {'Beam(\")':<10} {'Pixel(\")':<10} {'RMS':<10}"
        )
        print("-" * 80)

        for map_name, details in summary["map_details"].items():
            print(
                f"{map_name:<15} "
                f"{details['wavelength_um']:<8.1f} "
                f"{str(details['shape']):<12} "
                f"{details['beam_fwhm_arcsec']:<10.2f} "
                f"{details['pixel_scale_arcsec']:<10.2f} "
                f"{details['rms_noise']:<10.2e}"
            )

    def __len__(self) -> int:
        """Return number of loaded maps"""
        return len(self.maps)

    def __contains__(self, map_name: str) -> bool:
        """Check if map is loaded"""
        return map_name in self.maps

    def __getitem__(self, map_name: str) -> MapData:
        """Get map data by name"""
        if map_name not in self.maps:
            raise MapError(f"Map '{map_name}' not loaded")
        return self.maps[map_name]


def load_maps(maps_config: dict[str, MapConfig]) -> SkyMaps:
    """
    Convenience function to load maps

    Args:
        maps_config: Dictionary of map configurations

    Returns:
        Loaded SkyMaps instance
    """
    sky_maps = SkyMaps(maps_config)
    sky_maps.load_maps()
    return sky_maps
