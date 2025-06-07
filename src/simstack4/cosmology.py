"""
Cosmological calculations for Simstack4

This module handles cosmological distance calculations and conversions
needed for stacking analysis, including luminosity distances and volume calculations.
"""

from typing import Union, Dict, Any
import numpy as np
from dataclasses import dataclass
import warnings

from astropy.cosmology import Planck15, Planck18, FlatLambdaCDM
from astropy import units as u
from astropy import constants as const

from .config import Cosmology
from .exceptions.simstack_exceptions import CosmologyError
from .utils import setup_logging

logger = setup_logging()


@dataclass
class CosmologyResults:
    """Container for cosmological calculation results"""
    redshift: Union[float, np.ndarray]
    luminosity_distance: Union[float, np.ndarray]  # Mpc
    angular_diameter_distance: Union[float, np.ndarray]  # Mpc
    comoving_distance: Union[float, np.ndarray]  # Mpc
    lookback_time: Union[float, np.ndarray]  # Gyr
    age_at_z: Union[float, np.ndarray]  # Gyr
    h_factor: Union[float, np.ndarray]  # h^-2 correction factor

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy access"""
        return {
            'redshift': self.redshift,
            'luminosity_distance_mpc': self.luminosity_distance,
            'angular_diameter_distance_mpc': self.angular_diameter_distance,
            'comoving_distance_mpc': self.comoving_distance,
            'lookback_time_gyr': self.lookback_time,
            'age_at_z_gyr': self.age_at_z,
            'h_factor': self.h_factor
        }


class CosmologyCalculator:
    """
    Handle cosmological calculations for stacking analysis

    This class provides standardized cosmological distance calculations
    using Astropy's cosmology modules with support for Planck15 and Planck18.
    """

    def __init__(self, cosmology: Cosmology = Cosmology.PLANCK18):
        """
        Initialize cosmology calculator

        Args:
            cosmology: Cosmology model to use (Planck15 or Planck18)
        """
        self.cosmology_name = cosmology
        self.cosmo = self._get_cosmology(cosmology)

        # Store key parameters for easy access
        self.h0 = self.cosmo.H0.value  # km/s/Mpc
        self.h = self.h0 / 100.0  # Dimensionless Hubble parameter
        self.omega_m = self.cosmo.Om0
        self.omega_lambda = self.cosmo.Ode0

        logger.info(f"Cosmology initialized: {cosmology.value}")
        logger.info(f"H0 = {self.h0:.1f} km/s/Mpc, Ωm = {self.omega_m:.3f}, ΩΛ = {self.omega_lambda:.3f}")

    def _get_cosmology(self, cosmology: Cosmology) -> FlatLambdaCDM:
        """Get astropy cosmology object"""
        if cosmology == Cosmology.PLANCK15:
            return Planck15
        elif cosmology == Cosmology.PLANCK18:
            return Planck18
        else:
            raise CosmologyError(f"Unknown cosmology: {cosmology}")

    def calculate_distances(self, redshift: Union[float, np.ndarray]) -> CosmologyResults:
        """
        Calculate all cosmological distances for given redshift(s)

        Args:
            redshift: Redshift value(s)

        Returns:
            CosmologyResults object with all calculated distances
        """
        z = np.atleast_1d(redshift)

        # Validate redshift values
        if np.any(z < 0):
            raise CosmologyError("Negative redshift values not allowed")
        if np.any(z > 20):
            warnings.warn("Very high redshift values (z > 20) may be unreliable")

        # Calculate distances using astropy
        try:
            lum_dist = self.cosmo.luminosity_distance(z).to(u.Mpc).value
            ang_dist = self.cosmo.angular_diameter_distance(z).to(u.Mpc).value
            com_dist = self.cosmo.comoving_distance(z).to(u.Mpc).value
            lookback = self.cosmo.lookback_time(z).to(u.Gyr).value
            age_at_z = self.cosmo.age(z).to(u.Gyr).value

        except Exception as e:
            raise CosmologyError(f"Distance calculation failed: {e}")

        # Calculate h-factor correction (commonly used in IR luminosity studies)
        # This accounts for the h^-2 scaling of luminosity
        h_factor = (self.h / 0.7)**(-2)

        # Return scalar if input was scalar
        if np.isscalar(redshift):
            return CosmologyResults(
                redshift=float(z[0]),
                luminosity_distance=float(lum_dist[0]),
                angular_diameter_distance=float(ang_dist[0]),
                comoving_distance=float(com_dist[0]),
                lookback_time=float(lookback[0]),
                age_at_z=float(age_at_z[0]),
                h_factor=float(h_factor)
            )
        else:
            return CosmologyResults(
                redshift=z,
                luminosity_distance=lum_dist,
                angular_diameter_distance=ang_dist,
                comoving_distance=com_dist,
                lookback_time=lookback,
                age_at_z=age_at_z,
                h_factor=np.full_like(z, h_factor)
            )

    def luminosity_distance(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate luminosity distance in Mpc

        Args:
            redshift: Redshift value(s)

        Returns:
            Luminosity distance(s) in Mpc
        """
        results = self.calculate_distances(redshift)
        return results.luminosity_distance

    def angular_diameter_distance(self, redshift: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Calculate angular diameter distance in Mpc

        Args:
            redshift: Redshift value(s)

        Returns:
            Angular diameter distance(s) in Mpc
        """
        results = self.calculate_distances(redshift)
        return results.angular_diameter_distance

    def flux_to_luminosity(self, flux_jy: Union[float, np.ndarray],
                          redshift: Union[float, np.ndarray],
                          rest_wavelength_um: float) -> Union[float, np.ndarray]:
        """
        Convert observed flux to rest-frame luminosity

        Args:
            flux_jy: Observed flux in Jy
            redshift: Source redshift
            rest_wavelength_um: Rest-frame wavelength in microns

        Returns:
            Rest-frame luminosity in L_sun
        """
        # Get luminosity distance
        d_l = self.luminosity_distance(redshift)  # Mpc

        # Convert to cm
        d_l_cm = d_l * 1e6 * const.pc.cgs.value  # cm

        # K-correction factor (assumes S_nu ~ nu^-alpha with alpha ~ 1-2)
        # For IR: K = (1+z)^(1+alpha), with alpha ~ 1 for typical SED shapes
        k_correction = (1 + redshift)**2

        # Convert flux to luminosity
        # L = 4π D_L^2 * S_obs * K / (1+z)
        # Factor of (1+z) corrects for cosmological time dilation

        flux_cgs = flux_jy * 1e-23  # Convert Jy to erg/s/cm²/Hz

        # Calculate luminosity in erg/s/Hz
        lum_cgs = 4 * np.pi * d_l_cm**2 * flux_cgs * k_correction / (1 + redshift)

        # Convert to L_sun
        # For IR wavelengths, need to convert from per Hz to bolometric
        # This is wavelength-dependent - simplified here
        freq_hz = const.c.cgs.value / (rest_wavelength_um * 1e-4)  # Hz

        # Convert to L_sun (rough conversion for IR luminosities)
        l_sun_cgs = 3.828e33  # erg/s
        lum_l_sun = lum_cgs / l_sun_cgs

        return lum_l_sun

    def luminosity_to_flux(self, luminosity_l_sun: Union[float, np.ndarray],
                          redshift: Union[float, np.ndarray],
                          rest_wavelength_um: float) -> Union[float, np.ndarray]:
        """
        Convert rest-frame luminosity to observed flux

        Args:
            luminosity_l_sun: Rest-frame luminosity in L_sun
            redshift: Source redshift
            rest_wavelength_um: Rest-frame wavelength in microns

        Returns:
            Observed flux in Jy
        """
        # Get luminosity distance
        d_l = self.luminosity_distance(redshift)  # Mpc
        d_l_cm = d_l * 1e6 * const.pc.cgs.value  # cm

        # Convert luminosity to erg/s/Hz
        l_sun_cgs = 3.828e33  # erg/s
        lum_cgs = luminosity_l_sun * l_sun_cgs

        # K-correction (inverse of flux_to_luminosity)
        k_correction = (1 + redshift)**2

        # Calculate observed flux
        flux_cgs = lum_cgs * (1 + redshift) / (4 * np.pi * d_l_cm**2 * k_correction)

        # Convert to Jy
        flux_jy = flux_cgs / 1e-23

        return flux_jy

    def comoving_volume_element(self, redshift: Union[float, np.ndarray],
                               sky_area_deg2: float = 1.0) -> Union[float, np.ndarray]:
        """
        Calculate comoving volume element dV/dz per unit sky area

        Args:
            redshift: Redshift value(s)
            sky_area_deg2: Sky area in square degrees

        Returns:
            Comoving volume element in Mpc³ per unit redshift per deg²
        """
        # Get comoving distance and angular diameter distance
        results = self.calculate_distances(redshift)
        d_c = results.comoving_distance  # Mpc

        # Convert sky area to steradians
        sky_area_sr = sky_area_deg2 * (np.pi / 180)**2

        # Hubble parameter at redshift z
        h_z = self.cosmo.H(redshift).to(u.km/u.s/u.Mpc).value

        # Volume element: dV = c * D_C²(z) * dΩ * dz / H(z)
        c_km_s = const.c.to(u.km/u.s).value

        dv_dz = c_km_s * d_c**2 * sky_area_sr / h_z

        return dv_dz

    def get_cosmology_summary(self) -> Dict[str, Any]:
        """Get summary of cosmological parameters"""
        return {
            "cosmology": self.cosmology_name.value,
            "H0_km_s_Mpc": self.h0,
            "h": self.h,
            "Omega_m": self.omega_m,
            "Omega_lambda": self.omega_lambda,
            "Omega_k": getattr(self.cosmo, 'Ok0', 0.0),
            "age_universe_Gyr": self.cosmo.age(0).to(u.Gyr).value
        }

    def print_cosmology_summary(self) -> None:
        """Print a formatted summary of cosmological parameters"""
        summary = self.get_cosmology_summary()

        print("=== Cosmology Summary ===")
        print(f"Model: {summary['cosmology']}")
        print(f"H₀ = {summary['H0_km_s_Mpc']:.1f} km/s/Mpc (h = {summary['h']:.3f})")
        print(f"Ωₘ = {summary['Omega_m']:.3f}")
        print(f"ΩΛ = {summary['Omega_lambda']:.3f}")
        if summary['Omega_k'] != 0:
            print(f"Ωₖ = {summary['Omega_k']:.3f}")
        print(f"Age of Universe = {summary['age_universe_Gyr']:.2f} Gyr")


def calculate_redshift_bins_distances(redshift_bins: list,
                                     cosmology: Cosmology = Cosmology.PLANCK18) -> Dict[str, np.ndarray]:
    """
    Calculate distances for redshift bin centers and edges

    Args:
        redshift_bins: List of redshift bin edges
        cosmology: Cosmology model to use

    Returns:
        Dictionary with bin centers, edges, and corresponding distances
    """
    cosmo_calc = CosmologyCalculator(cosmology)

    # Calculate bin centers
    z_edges = np.array(redshift_bins)
    z_centers = (z_edges[:-1] + z_edges[1:]) / 2

    # Calculate distances for centers and edges
    results_centers = cosmo_calc.calculate_distances(z_centers)
    results_edges = cosmo_calc.calculate_distances(z_edges)

    return {
        "z_edges": z_edges,
        "z_centers": z_centers,
        "d_l_centers_mpc": results_centers.luminosity_distance,
        "d_l_edges_mpc": results_edges.luminosity_distance,
        "d_a_centers_mpc": results_centers.angular_diameter_distance,
        "d_a_edges_mpc": results_edges.angular_diameter_distance,
        "lookback_centers_gyr": results_centers.lookback_time,
        "lookback_edges_gyr": results_edges.lookback_time
    }


# Convenience function for quick distance calculations
def quick_luminosity_distance(redshift: Union[float, np.ndarray],
                             cosmology: Cosmology = Cosmology.PLANCK18) -> Union[float, np.ndarray]:
    """
    Quick luminosity distance calculation

    Args:
        redshift: Redshift value(s)
        cosmology: Cosmology model

    Returns:
        Luminosity distance(s) in Mpc
    """
    cosmo_calc = CosmologyCalculator(cosmology)
    return cosmo_calc.luminosity_distance(redshift)