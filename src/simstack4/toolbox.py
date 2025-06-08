"""
Complete Toolbox implementation for Simstack4

This consolidates utility functions and mathematical operations needed for stacking,
replacing the minimal toolbox.py and extending utils.py functionality.
"""

import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import logging
from dataclasses import dataclass
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u

logger = logging.getLogger(__name__)


@dataclass
class CircularRegion:
    """Defines a circular region on the sky"""
    center_ra: float  # degrees
    center_dec: float  # degrees
    radius_arcsec: float
    weight: float = 1.0


class SimstackToolbox:
    """
    Core mathematical and utility functions for stacking

    This class provides coordinate transformations, PSF handling,
    statistical functions, and other mathematical operations needed
    throughout the stacking pipeline.
    """

    @staticmethod
    def world_to_pixel_coords(wcs: WCS, ra: np.ndarray, dec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert world coordinates to pixel coordinates with error handling

        Args:
            wcs: World coordinate system
            ra: Right ascension array (degrees)
            dec: Declination array (degrees)

        Returns:
            x_pixel, y_pixel arrays
        """
        try:
            coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            x_pix, y_pix = wcs.world_to_pixel(coords)
            return np.asarray(x_pix), np.asarray(y_pix)
        except Exception as e:
            logger.warning(f"WCS transformation failed: {e}")
            # Fallback: assume tangent projection centered at (0,0)
            pixel_scale = 1.0  # arcsec/pixel default
            try:
                pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600
            except:
                pass

            x_pix = ra * 3600 / pixel_scale  # Convert to pixels from origin
            y_pix = dec * 3600 / pixel_scale
            return x_pix, y_pix

    @staticmethod
    def create_circular_mask(shape: Tuple[int, int], center: Tuple[float, float],
                           radius: float) -> np.ndarray:
        """
        Create circular mask array

        Args:
            shape: Array shape (ny, nx)
            center: Center coordinates (x, y) in pixels
            radius: Radius in pixels

        Returns:
            Boolean mask array (True inside circle)
        """
        ny, nx = shape
        y, x = np.ogrid[:ny, :nx]
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        return dist_from_center <= radius

    @staticmethod
    def add_circles_to_mask(mask: np.ndarray, centers: List[Tuple[float, float]],
                          radii: Union[float, List[float]]) -> np.ndarray:
        """
        Add multiple circles to existing mask

        Args:
            mask: Existing boolean mask
            centers: List of (x, y) center coordinates
            radii: Single radius or list of radii

        Returns:
            Updated mask
        """
        if isinstance(radii, (int, float)):
            radii = [radii] * len(centers)

        for center, radius in zip(centers, radii):
            circle_mask = SimstackToolbox.create_circular_mask(mask.shape, center, radius)
            mask |= circle_mask

        return mask

    @staticmethod
    def gaussian_psf_kernel(fwhm_pixels: float, kernel_size: Optional[int] = None) -> np.ndarray:
        """
        Create normalized Gaussian PSF kernel

        Args:
            fwhm_pixels: FWHM in pixels
            kernel_size: Size of kernel (odd integer). If None, auto-determine

        Returns:
            2D normalized Gaussian kernel
        """
        # Convert FWHM to sigma
        sigma = fwhm_pixels / (2 * np.sqrt(2 * np.log(2)))

        # Auto-determine kernel size if not provided
        if kernel_size is None:
            kernel_size = int(6 * sigma)
            if kernel_size % 2 == 0:
                kernel_size += 1

        # Create coordinate grids
        center = kernel_size // 2
        x, y = np.mgrid[0:kernel_size, 0:kernel_size]

        # Calculate Gaussian
        dist_sq = (x - center)**2 + (y - center)**2
        kernel = np.exp(-0.5 * dist_sq / sigma**2)

        # Normalize
        kernel /= np.sum(kernel)

        return kernel

    @staticmethod
    def convolve_with_kernel(data: np.ndarray, kernel: np.ndarray,
                           mode: str = 'constant') -> np.ndarray:
        """
        Convolve data with kernel using scipy

        Args:
            data: Input 2D array
            kernel: Convolution kernel
            mode: Boundary condition ('constant', 'nearest', 'reflect', 'wrap')

        Returns:
            Convolved array
        """
        from scipy.ndimage import convolve
        return convolve(data, kernel, mode=mode, cval=0.0)

    @staticmethod
    def fast_gaussian_convolve(data: np.ndarray, sigma: float) -> np.ndarray:
        """
        Fast Gaussian convolution using separable kernels

        Args:
            data: Input 2D array
            sigma: Gaussian sigma in pixels

        Returns:
            Convolved array
        """
        return gaussian_filter(data, sigma=sigma, mode='constant', cval=0.0)

    @staticmethod
    def create_source_layer(ra: np.ndarray, dec: np.ndarray, weights: np.ndarray,
                          wcs: WCS, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create 2D source layer from coordinates and weights

        Args:
            ra: Source RA coordinates (degrees)
            dec: Source Dec coordinates (degrees)
            weights: Source weights (e.g., stellar mass)
            wcs: World coordinate system for pixel conversion
            shape: Output array shape (ny, nx)

        Returns:
            2D source layer with sources placed at pixel locations
        """
        # Convert to pixel coordinates
        x_pix, y_pix = SimstackToolbox.world_to_pixel_coords(wcs, ra, dec)

        # Initialize layer
        layer = np.zeros(shape, dtype=np.float64)

        # Add sources to layer
        for x, y, weight in zip(x_pix, y_pix, weights):
            # Round to nearest pixel
            ix, iy = int(np.round(x)), int(np.round(y))

            # Check bounds
            if 0 <= ix < shape[1] and 0 <= iy < shape[0]:
                layer[iy, ix] += weight

        return layer

    @staticmethod
    def bilinear_interpolate_sources(ra: np.ndarray, dec: np.ndarray, weights: np.ndarray,
                                   wcs: WCS, shape: Tuple[int, int]) -> np.ndarray:
        """
        Create source layer using bilinear interpolation for sub-pixel accuracy

        Args:
            ra: Source RA coordinates (degrees)
            dec: Source Dec coordinates (degrees)
            weights: Source weights
            wcs: World coordinate system
            shape: Output array shape (ny, nx)

        Returns:
            2D source layer with bilinear interpolation
        """
        # Convert to pixel coordinates
        x_pix, y_pix = SimstackToolbox.world_to_pixel_coords(wcs, ra, dec)

        # Initialize layer
        layer = np.zeros(shape, dtype=np.float64)

        for x, y, weight in zip(x_pix, y_pix, weights):
            # Get integer pixel coordinates
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = x0 + 1, y0 + 1

            # Calculate fractional parts
            fx, fy = x - x0, y - y0

            # Bilinear weights
            w00 = (1 - fx) * (1 - fy)
            w01 = (1 - fx) * fy
            w10 = fx * (1 - fy)
            w11 = fx * fy

            # Add to pixels if within bounds
            if 0 <= x0 < shape[1] and 0 <= y0 < shape[0]:
                layer[y0, x0] += weight * w00
            if 0 <= x1 < shape[1] and 0 <= y0 < shape[0]:
                layer[y0, x1] += weight * w10
            if 0 <= x0 < shape[1] and 0 <= y1 < shape[0]:
                layer[y1, x0] += weight * w01
            if 0 <= x1 < shape[1] and 0 <= y1 < shape[0]:
                layer[y1, x1] += weight * w11

        return layer

    @staticmethod
    def calculate_overlap_matrix(layers: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        """
        Calculate overlap matrix between population layers

        Args:
            layers: Layer matrix (n_populations, n_pixels)
            valid_mask: Boolean mask for valid pixels

        Returns:
            Overlap matrix (n_populations, n_populations)
        """
        # Use only valid pixels
        valid_layers = layers[:, valid_mask]

        # Calculate overlap matrix (A^T @ A)
        overlap = np.dot(valid_layers, valid_layers.T)

        return overlap

    @staticmethod
    def robust_linear_solve(A: np.ndarray, b: np.ndarray,
                          rcond: float = 1e-12) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Robust linear system solver with diagnostics

        Args:
            A: Coefficient matrix (n_pixels, n_populations)
            b: Observation vector (n_pixels,)
            rcond: Relative condition number cutoff

        Returns:
            solution, diagnostics_dict
        """
        from scipy.linalg import lstsq, LinAlgError

        diagnostics = {}

        try:
            # Solve using least squares
            solution, residuals, rank, singular_values = lstsq(A, b, rcond=rcond)

            # Calculate condition number
            if len(singular_values) > 0:
                cond_num = singular_values[0] / singular_values[-1]
            else:
                cond_num = np.inf

            diagnostics = {
                'residuals': residuals,
                'rank': rank,
                'singular_values': singular_values,
                'condition_number': cond_num,
                'well_conditioned': cond_num < 1e12,
                'n_pixels': len(b),
                'n_populations': A.shape[1]
            }

            # Check for numerical issues
            if np.any(~np.isfinite(solution)):
                logger.warning("Non-finite values in solution")
                solution = np.nan_to_num(solution)

        except LinAlgError as e:
            logger.error(f"Linear algebra error: {e}")
            solution = np.zeros(A.shape[1])
            diagnostics['error'] = str(e)

        return solution, diagnostics

    @staticmethod
    def bootstrap_resample_indices(n_total: int, n_sample: int,
                                 seed: Optional[int] = None) -> np.ndarray:
        """
        Generate bootstrap resampled indices

        Args:
            n_total: Total number of items
            n_sample: Number of items to sample
            seed: Random seed

        Returns:
            Array of resampled indices
        """
        if seed is not None:
            np.random.seed(seed)

        return np.random.choice(n_total, size=n_sample, replace=True)

    @staticmethod
    def calculate_error_from_bootstrap(bootstrap_results: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate errors from bootstrap results

        Args:
            bootstrap_results: List of bootstrap flux arrays

        Returns:
            mean_flux, flux_errors (standard deviation)
        """
        if not bootstrap_results:
            return np.array([]), np.array([])

        # Stack results
        stacked = np.vstack(bootstrap_results)

        # Calculate statistics
        mean_flux = np.mean(stacked, axis=0)
        flux_errors = np.std(stacked, axis=0, ddof=1)

        return mean_flux, flux_errors

    @staticmethod
    def percentile_errors(bootstrap_results: List[np.ndarray],
                         percentiles: Tuple[float, float] = (16, 84)) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate percentile-based errors from bootstrap

        Args:
            bootstrap_results: List of bootstrap flux arrays
            percentiles: Lower and upper percentiles for errors

        Returns:
            median_flux, lower_error, upper_error
        """
        if not bootstrap_results:
            return np.array([]), np.array([]), np.array([])

        stacked = np.vstack(bootstrap_results)

        median_flux = np.median(stacked, axis=0)
        lower = np.percentile(stacked, percentiles[0], axis=0)
        upper = np.percentile(stacked, percentiles[1], axis=0)

        lower_error = median_flux - lower
        upper_error = upper - median_flux

        return median_flux, lower_error, upper_error

    @staticmethod
    def validate_layer_matrix(layers: np.ndarray, population_labels: List[str]) -> Dict[str, Any]:
        """
        Validate layer matrix for numerical issues

        Args:
            layers: Layer matrix (n_populations, n_pixels)
            population_labels: Labels for populations

        Returns:
            Validation results dictionary
        """
        n_pops, n_pixels = layers.shape

        results = {
            'shape': (n_pops, n_pixels),
            'total_sources': np.sum(layers),
            'populations': {}
        }

        for i, label in enumerate(population_labels):
            layer = layers[i]
            pop_results = {
                'total_flux': np.sum(layer),
                'n_nonzero_pixels': np.sum(layer > 0),
                'max_value': np.max(layer),
                'has_nan': np.any(np.isnan(layer)),
                'has_inf': np.any(np.isinf(layer))
            }
            results['populations'][label] = pop_results

        return results

    @staticmethod
    def spatial_clustering_analysis(ra: np.ndarray, dec: np.ndarray,
                                  max_separation_arcmin: float = 5.0) -> Dict[str, Any]:
        """
        Analyze spatial clustering of sources

        Args:
            ra: Source RA coordinates (degrees)
            dec: Source Dec coordinates (degrees)
            max_separation_arcmin: Maximum separation to consider for clustering

        Returns:
            Clustering analysis results
        """
        # Convert to cartesian coordinates for distance calculation
        coords = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)

        # Calculate pairwise separations (this can be memory intensive for large catalogs)
        if len(ra) > 10000:
            logger.warning(f"Large catalog ({len(ra)} sources) - spatial analysis may be slow")

        separations = coords[:, np.newaxis].separation(coords).to(u.arcmin).value

        # Count pairs within max_separation
        close_pairs = np.sum(separations < max_separation_arcmin) - len(ra)  # Subtract diagonal

        # Calculate clustering statistics
        mean_separation = np.mean(separations[separations > 0])
        median_separation = np.median(separations[separations > 0])

        results = {
            'n_sources': len(ra),
            'close_pairs': close_pairs // 2,  # Each pair counted twice
            'clustering_fraction': (close_pairs / 2) / len(ra),
            'mean_separation_arcmin': mean_separation,
            'median_separation_arcmin': median_separation
        }

        return results

    @staticmethod
    def estimate_confusion_noise(map_data: np.ndarray, beam_fwhm_arcsec: float,
                               pixel_scale_arcsec: float) -> float:
        """
        Estimate confusion noise in map

        Args:
            map_data: 2D map array
            beam_fwhm_arcsec: Beam FWHM in arcseconds
            pixel_scale_arcsec: Pixel scale in arcseconds/pixel

        Returns:
            Estimated confusion noise (same units as map)
        """
        # Calculate beam area in pixels
        beam_fwhm_pixels = beam_fwhm_arcsec / pixel_scale_arcsec
        beam_area_pixels = np.pi * (beam_fwhm_pixels / 2.355)**2  # Gaussian beam

        # Use structure function approach to estimate confusion
        # This is a simplified version - full implementation would be more sophisticated
        valid_data = map_data[~np.isnan(map_data)]

        if len(valid_data) == 0:
            return 0.0

        # Calculate local variance on beam scales
        kernel_size = int(beam_fwhm_pixels)
        if kernel_size < 3:
            kernel_size = 3
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Smooth map and calculate difference
        smoothed = gaussian_filter(map_data, sigma=beam_fwhm_pixels/2.355, mode='constant')
        difference = map_data - smoothed

        # Estimate confusion as RMS of difference
        confusion_noise = np.nanstd(difference)

        return confusion_noise

    @staticmethod
    def color_correction_factor(observed_wavelength: float, rest_wavelength: float,
                              redshift: float, spectral_index: float = -2.0) -> float:
        """
        Calculate color correction factor for flux measurements

        Args:
            observed_wavelength: Observed wavelength (microns)
            rest_wavelength: Rest-frame wavelength (microns)
            redshift: Source redshift
            spectral_index: Assumed spectral index (S_nu ~ nu^alpha)

        Returns:
            Color correction factor
        """
        # Frequency ratio
        freq_ratio = rest_wavelength / observed_wavelength

        # Color correction assuming power-law spectrum
        color_correction = freq_ratio**spectral_index

        return color_correction


# Additional utility functions that were in utils.py but fit better here
def create_coordinate_grids(shape: Tuple[int, int], wcs: WCS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create RA/Dec coordinate grids for a map

    Args:
        shape: Map shape (ny, nx)
        wcs: World coordinate system

    Returns:
        ra_grid, dec_grid in degrees
    """
    ny, nx = shape
    x_pixels, y_pixels = np.meshgrid(np.arange(nx), np.arange(ny))

    try:
        coords = wcs.pixel_to_world(x_pixels, y_pixels)
        ra_grid = coords.ra.degree
        dec_grid = coords.dec.degree
    except Exception as e:
        logger.warning(f"Could not create coordinate grids: {e}")
        # Fallback to linear approximation
        pixel_scale = 1.0  # arcsec/pixel
        try:
            pixel_scale = abs(wcs.wcs.cdelt[0]) * 3600
        except:
            pass

        ra_grid = x_pixels * pixel_scale / 3600  # Convert to degrees
        dec_grid = y_pixels * pixel_scale / 3600

    return ra_grid, dec_grid


def safe_log10(values: np.ndarray, min_value: float = 1e-30) -> np.ndarray:
    """
    Safe logarithm that handles zeros and negative values

    Args:
        values: Input array
        min_value: Minimum value to use instead of zeros/negatives

    Returns:
        log10(values) with safe handling
    """
    safe_values = np.where(values > 0, values, min_value)
    return np.log10(safe_values)


def angular_separation(ra1: float, dec1: float, ra2: float, dec2: float) -> float:
    """
    Calculate angular separation between two points on the sky

    Args:
        ra1, dec1: First point coordinates (degrees)
        ra2, dec2: Second point coordinates (degrees)

    Returns:
        Angular separation in degrees
    """
    coord1 = SkyCoord(ra=ra1*u.deg, dec=dec1*u.deg)
    coord2 = SkyCoord(ra=ra2*u.deg, dec=dec2*u.deg)
    return coord1.separation(coord2).degree