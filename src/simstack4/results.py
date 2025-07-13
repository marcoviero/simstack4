"""
Complete Results processing and analysis for Simstack4

This module handles processing of stacking results, applying cosmological corrections,
calculating derived quantities like luminosities and star formation rates.
"""
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .algorithm import StackingResults
from .config import SimstackConfig
from .cosmology import CosmologyCalculator
from .exceptions.simstack_exceptions import ResultsError
from .populations import PopulationManager
from .utils import setup_logging

logger = setup_logging()


@dataclass
class SEDResults:
    """Spectral Energy Distribution results for a population"""

    population_id: str
    wavelengths: np.ndarray  # microns
    flux_densities: np.ndarray  # Jy
    flux_errors: np.ndarray  # Jy
    luminosity_distances: np.ndarray  # Mpc
    rest_luminosities: np.ndarray  # L_sun
    rest_luminosity_errors: np.ndarray  # L_sun
    median_redshift: float
    median_mass: float
    n_sources: int

    # Greybody fit results
    greybody_fit_success: bool = False
    dust_temperature_rest_frame: float | None = None  # K (rest-frame)
    dust_temperature_observed_frame: float | None = None  # K (observed-frame)
    dust_temperature_error: float | None = None  # K (error on rest-frame)
    emissivity_index: float | None = None  # beta
    emissivity_index_error: float | None = None
    amplitude: float | None = None  # Jy (at reference frequency)
    amplitude_error: float | None = None  # Jy
    chi2_reduced: float | None = None
    model_wavelengths: np.ndarray | None = None  # microns
    model_fluxes: np.ndarray | None = None  # Jy


@dataclass
class DerivedQuantities:
    """Container for derived astrophysical quantities"""

    total_ir_luminosity: float  # L_sun
    total_ir_luminosity_error: float  # L_sun
    star_formation_rate: float  # M_sun/yr
    star_formation_rate_error: float  # M_sun/yr
    specific_sfr: float  # yr^-1
    dust_temperature_rest_frame: float | None = None  # K (rest-frame)
    dust_temperature_observed_frame: float | None = None  # K (observed-frame)
    dust_mass: float | None = None  # M_sun


class GreybodyFitter:
    """Greybody fitter with proper L_IR calculation and fixed beta option"""

    def __init__(self, fix_beta: bool = True, beta_fixed: float = 1.8):
        """
        Initialize fitter

        Parameters:
        -----------
        fix_beta : bool
            Whether to fix the emissivity index beta
        beta_fixed : float
            Fixed value of beta if fix_beta=True
        """
        # Physical constants
        self.h = 6.62607015e-34  # Planck constant (J⋅s)
        self.c = 299792458  # Speed of light (m/s)
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.L_sun = 3.828e26  # Solar luminosity (W)

        self.fix_beta = fix_beta
        self.beta_fixed = beta_fixed

        # Setup cosmology (simplified for now)
        self.H0 = 70  # km/s/Mpc
        self.Om0 = 0.3

    def luminosity_distance(self, z: float) -> float:
        """Calculate luminosity distance in meters"""
        if z <= 0:
            z = 0.01  # Small default for nearby sources

        # Simplified calculation for low-z
        c_km_s = 299792.458  # km/s
        D_L = c_km_s * z / self.H0  # Mpc
        # D_L_m = D_L * 3.086e22  # Convert Mpc to meters
        return D_L  # _m

    def planck_function(self, nu: np.ndarray, T: float) -> np.ndarray:
        """Planck function B_ν(T) in SI units (W m⁻² sr⁻¹ Hz⁻¹)"""
        # The Planck function needs to be normalized properly for the greybody model
        # We need B_ν(T) but normalized for the greybody amplitude
        exponent = self.h * nu / (self.k_B * T)

        # Avoid overflow for large exponents
        exponent = np.clip(exponent, 0, 700)

        return (2 * self.h * nu**3 / self.c**2) / (np.exp(exponent) - 1)

    def greybody_model(
        self,
        wavelength_um: np.ndarray,
        amplitude: float,
        temperature: float,
        beta: float = 1.8,
        alpha: float = 2.0,
    ) -> np.ndarray:
        """
        Greybody model with power law extensions.

        Parameters:
        -----------
        wavelength_um : array of wavelengths in micrometers
        amplitude : log10 of the amplitude parameter (typical values around -35)
        temperature : dust temperature in Kelvin
        beta : emissivity index for long wavelengths (default 1.8)
        alpha : power law index for short wavelengths (default 2.0)

        Returns:
        --------
        flux_density : array of flux densities in Jy
        """

        # Convert wavelength to frequency
        c_light = 299792458.0  # m/s
        nu_in = c_light * 1.0e6 / wavelength_um  # Hz

        # Linear amplitude
        A = 10**amplitude

        # Calculate transition frequency
        nu_cut = (3.0 + beta + alpha) * 0.208367e11 * temperature

        # Constants for power law normalization (from your existing code)
        base = (
            2.0
            * (6.626) ** (-2.0 - beta - alpha)
            * (1.38) ** (3.0 + beta + alpha)
            / (2.99792458) ** 2.0
        )
        expo = 34.0 * (2.0 + beta + alpha) - 23.0 * (3.0 + beta + alpha) - 16.0 + 26.0
        K = base * 10.0**expo
        w_num = A * K * (temperature * (3.0 + beta + alpha)) ** (3.0 + beta + alpha)
        w_den = np.exp(3.0 + beta + alpha) - 1.0
        w_div = w_num / w_den

        # Modified blackbody (long wavelength side)
        graybody = A * nu_in**beta * self.black(nu_in, temperature)[0] / 1000.0

        # Power law (short wavelength side)
        powerlaw = w_div * nu_in ** (-alpha)

        # Apply transition
        flux_density = graybody.copy()
        ind_cut = nu_in >= nu_cut
        flux_density[ind_cut] = powerlaw[ind_cut]

        return flux_density

    def black(self, nu_in, T):
        # h = 6.623e-34     ; Joule*s
        # k = 1.38e-23      ; Joule/K
        # c = 3e8           ; m/s
        # (2*h*nu_in^3/c^2)*(1/( exp(h*nu_in/k*T) - 1 )) * 10^29

        a0 = 1.4718e-21  # 2*h*10^29/c^2
        a1 = 4.7993e-11  # h/k

        num = a0 * nu_in**3.0
        den = np.exp(a1 * np.outer(1.0 / T, nu_in)) - 1.0
        ret = num / den

        return ret

    def blackbody_model(self, nu_in, T):
        # h = 6.623e-34     ; Joule*s
        # k = 1.38e-23      ; Joule/K
        # c = 3e8           ; m/s
        # (2*h*nu_in^3/c^2)*(1/( exp(h*nu_in/k*T) - 1 )) * 10^29

        a0 = 1.4718e-21  # 2*h*10^29/c^2
        a1 = 4.7993e-11  # h/k

        num = a0 * nu_in**3.0
        den = np.exp(a1 * nu_in / T) - 1.0
        ret = num / den

        return ret

    def calculate_LIR(
        self, amplitude: float, temperature: float, beta: float, redshift: float
    ) -> tuple[float, float]:
        """
        Calculate L_IR by integrating L_ν over frequency (8-1000 μm rest-frame)

        Parameters:
        -----------
        amplitude : SED normalization
        temperature : dust temperature in Kelvin
        beta : emissivity index for long wavelengths
        alpha : power law index for short wavelengths
        redshift : power law index for short wavelengths

        Returns:
        --------
        flux_density : array of flux densities in Jy
        """
        if np.isnan(redshift) or redshift <= 0:
            redshift = 0.01

        # Get luminosity distance (in Mpc, matching simstack3)
        D_L = self.luminosity_distance(redshift)

        # Generate wavelength range in microns (8-1000 μm rest-frame)
        wavelength_range = np.logspace(np.log10(8), np.log10(1000), 1000)

        # Get model SED in same units as simstack3
        # greybody_model returns flux in Jy
        model_sed_jy = self.greybody_model(
            wavelength_range, amplitude, temperature, beta
        )

        # Convert wavelength to frequency (Hz)
        c_light = 299792458.0  # m/s (matching simstack3)
        nu_in = c_light * 1.0e6 / wavelength_range  # Hz

        # Calculate frequency intervals (matching simstack3 exactly)
        dnu = (
            nu_in[:-1] - nu_in[1:]
        )  # Note: frequencies decrease as wavelength increases
        dnu = np.append(dnu[0], dnu)  # Extend first interval to match array length

        # Integrate: L_IR = ∫ S_ν dν in Jy⋅Hz (matching simstack3)
        Lir_jy_hz = np.sum(model_sed_jy * dnu)

        # Convert to solar luminosities
        # conversion = 4π D_L² / L_sun with proper unit handling
        conversion = (
            4.0 * np.pi * (1.0e-13 * D_L * 3.08568025e22) ** 2.0 / self.L_sun
        )  # Units: L_sun/(Jy⋅Hz)

        L_IR_solar = Lir_jy_hz * conversion

        # Error estimate (20% uncertainty)
        L_IR_error = L_IR_solar * 0.2
        print(np.log10(L_IR_solar))
        return L_IR_solar, L_IR_error

    def calculate_dust_mass(
        self, amplitude: float, temperature: float, beta: float, redshift: float
    ) -> tuple[float, float]:
        """
        Calculate dust mass from greybody fit parameters

        Returns:
        --------
        M_dust : float
            Dust mass in solar masses
        M_dust_error : float
            Error estimate
        """
        if np.isnan(temperature) or np.isnan(amplitude):
            return np.nan, np.nan

        # Get luminosity distance
        D_L = self.luminosity_distance(redshift)

        # Reference wavelength and opacity
        lambda_ref = 250e-6  # 250 μm in meters
        kappa_ref = 0.4  # m²/kg at 250 μm (typical value)

        # Convert amplitude to luminosity at reference wavelength
        nu_ref = self.c / lambda_ref

        # Get flux at reference wavelength
        flux_ref_jy = self.greybody_model(
            np.array([250.0]), amplitude, temperature, beta
        )[0]
        flux_ref_si = flux_ref_jy * 1e-26  # W/m²/Hz

        # Calculate dust mass (simplified)
        # M_dust = F_ν * D_L² / (κ * B_ν(T))
        B_nu = self.planck_function(np.array([nu_ref]), temperature)[0]
        M_dust_kg = flux_ref_si * D_L**2 / (kappa_ref * B_nu)

        # Convert to solar masses
        M_sun_kg = 1.989e30
        M_dust_solar = M_dust_kg / M_sun_kg

        # Error estimate
        M_dust_error = M_dust_solar * 0.3  # 30% error

        return M_dust_solar, M_dust_error

    def fit_sed(
        self,
        wavelengths: np.ndarray,
        fluxes: np.ndarray,
        flux_errors: np.ndarray,
        redshift: float,
    ) -> dict[str, Any]:
        """
        Fit greybody model to SED data with proper L_IR calculation

        Returns both rest-frame and observed-frame dust temperatures
        """
        valid = (
            (fluxes > 0)
            & (flux_errors > 0)
            & np.isfinite(fluxes)
            & np.isfinite(flux_errors)
        )

        if np.sum(valid) < 3:
            return {"fit_success": False, "reason": "insufficient_data"}

        wave_fit = wavelengths[valid]
        flux_fit = fluxes[valid]
        error_fit = flux_errors[valid]

        try:
            # Initial guess based on Wien's displacement law
            peak_idx = np.argmax(flux_fit)
            T_guess = 2898 / wave_fit[peak_idx]  # Wien's displacement
            T_guess = np.clip(T_guess, 8, 80)  # Reasonable temperature range
            T_guess = 15
            T_guess = (23.8 + 2.7 * redshift + 0.9 * redshift**2) / (1 + redshift)
            amplitude_guess = -35

            # Fit the model
            if self.fix_beta:
                # Wrapper for fixed beta
                def model_func(wave, amp, temp):
                    return self.greybody_model(wave, amp, temp, self.beta_fixed)

                popt, pcov = curve_fit(
                    model_func,
                    wave_fit,
                    flux_fit,
                    sigma=error_fit,
                    p0=[amplitude_guess, T_guess],
                    # bounds=([amplitude_guess * 0.01, 8], [amplitude_guess * 100, 80]),
                    bounds=([-50, 8], [-20, 80]),
                    maxfev=5000,
                )
                amplitude, temperature_observed = popt
                beta = self.beta_fixed
                param_errors = np.sqrt(np.diag(pcov))
                amplitude_err, temperature_err = param_errors
                beta_err = 0.0

            else:
                popt, pcov = curve_fit(
                    self.greybody_model,
                    wave_fit,
                    flux_fit,
                    sigma=error_fit,
                    p0=[amplitude_guess, T_guess, 1.8],
                    # bounds=([amplitude_guess * 0.01, 8, 0.5], [amplitude_guess * 100, 80, 2.5]),
                    bounds=([-50, 8], [-20, 80]),
                    maxfev=5000,
                )
                amplitude, temperature_observed, beta = popt
                param_errors = np.sqrt(np.diag(pcov))
                amplitude_err, temperature_err, beta_err = param_errors

            # Calculate rest-frame temperature from observed-frame
            temperature_rest_frame = temperature_observed * (1 + redshift)

            # Calculate goodness of fit
            model_fluxes = self.greybody_model(
                wave_fit, amplitude, temperature_rest_frame, beta
            )
            chi2 = np.sum(((flux_fit - model_fluxes) / error_fit) ** 2)
            dof = len(wave_fit) - (2 if self.fix_beta else 3)
            chi2_reduced = chi2 / max(1, dof)

            # Calculate L_IR and dust mass
            L_IR, L_IR_error = self.calculate_LIR(
                amplitude, temperature_observed, beta, redshift
            )
            M_dust, M_dust_error = self.calculate_dust_mass(
                amplitude, temperature_observed, beta, redshift
            )

            # Generate smooth model for plotting
            wave_model = np.logspace(
                np.log10(np.min(wave_fit) * 0.5), np.log10(np.max(wave_fit) * 2), 200
            )
            flux_model = self.greybody_model(
                wave_model, amplitude, temperature_observed, beta
            )

            return {
                "fit_success": True,
                "amplitude": amplitude,
                "amplitude_error": amplitude_err,
                "temperature_rest_frame": temperature_rest_frame,
                "temperature_observed_frame": temperature_observed,
                "temperature_error": temperature_err,
                "beta": beta,
                "beta_error": beta_err,
                "chi2_reduced": chi2_reduced,
                "L_IR": L_IR,
                "L_IR_error": L_IR_error,
                "dust_mass": M_dust,
                "dust_mass_error": M_dust_error,
                "n_points": len(wave_fit),
                "wavelengths_fit": wave_fit,
                "fluxes_fit": flux_fit,
                "flux_errors_fit": error_fit,
                "model_wavelengths": wave_model,
                "model_fluxes": flux_model,
                "redshift_used": redshift,
            }

        except Exception as e:
            logger.warning(f"Greybody fit failed: {e}")
            return {"fit_success": False, "reason": str(e)}


class SimstackResults:
    """
    Process and analyze stacking results

    This class takes raw stacking results and applies cosmological corrections,
    calculates luminosities, star formation rates, and other derived quantities.
    """

    def __init__(
        self,
        config: SimstackConfig,
        stacking_results: StackingResults,
        population_manager: PopulationManager,
        cosmology_calc: CosmologyCalculator | None = None,
        fix_beta: bool = True,
        beta_fixed: float = 1.8,
    ):
        """
        Initialize results processor

        Args:
            config: Simstack configuration
            stacking_results: Raw stacking results from algorithm
            population_manager: Population manager with catalog info
            cosmology_calc: Cosmology calculator (created if None)
            fix_beta: Whether to fix emissivity index beta in greybody fitting
            beta_fixed: Fixed value of beta if fix_beta=True
        """
        self.config = config
        self.raw_results = stacking_results
        self.population_manager = population_manager

        # Set up cosmology
        if cosmology_calc is None:
            self.cosmology_calc = CosmologyCalculator(
                config.cosmology
            )  # Cosmology("Planck18"))
        else:
            self.cosmology_calc = cosmology_calc

        # Initialize greybody fitter
        self.greybody_fitter = GreybodyFitter(fix_beta=fix_beta, beta_fixed=beta_fixed)

        # Initialize processed results containers
        self.sed_results: dict[str, SEDResults] = {}
        self.derived_quantities: dict[str, DerivedQuantities] = {}
        self.band_results: dict[str, dict[str, Any]] = {}

        # Process results
        self._process_results()

    def _process_results(self) -> None:
        """Process raw results into SEDs and derived quantities"""
        logger.info("Processing stacking results...")

        # Create SEDs for each population
        for i, pop_label in enumerate(self.raw_results.population_labels):
            if pop_label == "foreground":
                continue  # Skip foreground layer

            try:
                sed_result = self._create_sed_for_population(pop_label, i)
                self.sed_results[pop_label] = sed_result

                # Calculate derived quantities
                derived = self._calculate_derived_quantities(sed_result)
                self.derived_quantities[pop_label] = derived

            except Exception as e:
                logger.warning(f"Failed to process population {pop_label}: {e}")

        # Process band-by-band results
        self._process_band_results()

        logger.info(f"Processed results for {len(self.sed_results)} populations")

    def _create_sed_for_population(self, pop_label: str, pop_index: int) -> SEDResults:
        """Create SED results for a single population"""

        # Get population info
        if pop_label not in self.population_manager.populations:
            raise ResultsError(f"Population {pop_label} not found in manager")

        pop_bin = self.population_manager.populations[pop_label]

        # Extract fluxes and errors for this population
        wavelengths = []
        flux_densities = []
        flux_errors = []

        for map_name in self.raw_results.map_names:
            # Get wavelength from config
            map_config = self.config.maps[map_name]
            wavelengths.append(map_config.wavelength)

            # Get flux and error
            flux = self.raw_results.flux_densities[map_name][pop_index]
            error = self.raw_results.flux_errors[map_name][pop_index]

            # Apply color correction
            # flux *= map_config.color_correction
            # error *= map_config.color_correction

            flux_densities.append(flux)
            flux_errors.append(error)

        # Convert to arrays and sort by wavelength
        wavelengths = np.array(wavelengths)
        flux_densities = np.array(flux_densities)
        flux_errors = np.array(flux_errors)

        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        flux_densities = flux_densities[sort_idx]
        flux_errors = flux_errors[sort_idx]

        # Calculate luminosity distance
        z_median = pop_bin.median_redshift
        cosmo_results = self.cosmology_calc.calculate_distances(z_median)
        luminosity_distance = cosmo_results.luminosity_distance

        # Convert to rest-frame luminosities
        rest_luminosities = np.zeros_like(flux_densities)
        rest_luminosity_errors = np.zeros_like(flux_errors)

        for i, (wave, flux, flux_err) in enumerate(
            zip(wavelengths, flux_densities, flux_errors, strict=True)
        ):
            if flux > 0:  # Only convert positive fluxes
                rest_lum = self.cosmology_calc.flux_to_luminosity(flux, z_median, wave)
                rest_lum_err = self.cosmology_calc.flux_to_luminosity(
                    flux_err, z_median, wave
                )
                rest_luminosities[i] = rest_lum
                rest_luminosity_errors[i] = rest_lum_err

        # Fit greybody model
        greybody_results = self.greybody_fitter.fit_sed(
            wavelengths, flux_densities, flux_errors, z_median
        )

        # Create SED result
        sed_result = SEDResults(
            population_id=pop_label,
            wavelengths=wavelengths,
            flux_densities=flux_densities,
            flux_errors=flux_errors,
            luminosity_distances=np.full_like(wavelengths, luminosity_distance),
            rest_luminosities=rest_luminosities,
            rest_luminosity_errors=rest_luminosity_errors,
            median_redshift=z_median,
            median_mass=pop_bin.median_stellar_mass,
            n_sources=pop_bin.n_sources,
        )

        # Add greybody fit results
        if greybody_results["fit_success"]:
            sed_result.greybody_fit_success = True
            sed_result.dust_temperature_rest_frame = greybody_results[
                "temperature_rest_frame"
            ]
            sed_result.dust_temperature_observed_frame = greybody_results[
                "temperature_observed_frame"
            ]
            sed_result.dust_temperature_error = greybody_results["temperature_error"]
            sed_result.emissivity_index = greybody_results["beta"]
            sed_result.emissivity_index_error = greybody_results["beta_error"]
            sed_result.amplitude = greybody_results["amplitude"]
            sed_result.amplitude_error = greybody_results["amplitude_error"]
            sed_result.chi2_reduced = greybody_results["chi2_reduced"]
            sed_result.model_wavelengths = greybody_results["model_wavelengths"]
            sed_result.model_fluxes = greybody_results["model_fluxes"]

        return sed_result

    def _calculate_derived_quantities(
        self, sed_result: SEDResults
    ) -> DerivedQuantities:
        """Calculate derived astrophysical quantities from SED"""

        # Get L_IR and dust mass from greybody model if available
        total_ir_lum = 0.0
        total_ir_lum_err = 0.0
        dust_temp_rest = None
        dust_temp_obs = None
        dust_mass = None

        if sed_result.greybody_fit_success:
            # Get L_IR directly from the greybody fit results that are already stored
            # The fit_sed method already calculated L_IR and stored it in the results

            # Use the stored fit parameters to calculate L_IR
            amplitude = sed_result.amplitude
            # temperature_rest = sed_result.dust_temperature_rest_frame
            temperature_observed = sed_result.dust_temperature_observed_frame
            beta = sed_result.emissivity_index or self.greybody_fitter.beta_fixed
            redshift = sed_result.median_redshift

            if amplitude is not None and temperature_observed is not None:
                try:
                    # Calculate L_IR using the greybody fitter's method
                    total_ir_lum, total_ir_lum_err = self.greybody_fitter.calculate_LIR(
                        amplitude, temperature_observed, beta, redshift
                    )

                    # Also get dust mass
                    dust_mass, _ = self.greybody_fitter.calculate_dust_mass(
                        amplitude, temperature_observed, beta, redshift
                    )

                    # Store temperature values
                    dust_temp_rest = sed_result.dust_temperature_rest_frame
                    dust_temp_obs = sed_result.dust_temperature_observed_frame

                    print(
                        f"✓ Population {sed_result.population_id}: L_IR = {total_ir_lum:.2e} L_sun"
                    )

                except Exception as e:
                    print(
                        f"⚠️  L_IR calculation failed for {sed_result.population_id}: {e}"
                    )
                    total_ir_lum = 0.0
                    total_ir_lum_err = 0.0
            else:
                print(f"⚠️  Missing fit parameters for {sed_result.population_id}")
        else:
            print(f"⚠️  No successful greybody fit for {sed_result.population_id}")

        """# Fallback: if L_IR is still zero, try direct integration of flux measurements
        if total_ir_lum <= 0:
            total_ir_lum, total_ir_lum_err = self._calculate_direct_LIR(sed_result)
            if total_ir_lum > 0:
                print(f"✓ Used direct integration for {sed_result.population_id}: L_IR = {total_ir_lum:.2e} L_sun")
        """

        # Convert IR luminosity to star formation rate
        # Use Kennicutt (1998) relation: SFR [M_sun/yr] = L_IR [L_sun] / 1e10
        sfr = total_ir_lum / 1e10
        sfr_err = total_ir_lum_err / 1e10

        # Calculate specific star formation rate
        stellar_mass = 10**sed_result.median_mass  # Convert from log mass
        specific_sfr = sfr / stellar_mass if stellar_mass > 0 else 0.0

        return DerivedQuantities(
            total_ir_luminosity=total_ir_lum,
            total_ir_luminosity_error=total_ir_lum_err,
            star_formation_rate=sfr,
            star_formation_rate_error=sfr_err,
            specific_sfr=specific_sfr,
            dust_temperature_rest_frame=dust_temp_rest,
            dust_temperature_observed_frame=dust_temp_obs,
            dust_mass=dust_mass,
        )

    def _calculate_direct_LIR(self, sed_result: SEDResults) -> tuple[float, float]:
        """Calculate L_IR using direct integration method (fallback)"""

        valid_mask = (sed_result.rest_luminosities > 0) & np.isfinite(
            sed_result.rest_luminosities
        )

        if np.sum(valid_mask) < 2:
            return 0.0, 0.0

        # Get valid data
        valid_waves = sed_result.wavelengths[valid_mask]
        valid_lums = sed_result.rest_luminosities[valid_mask]
        valid_errs = sed_result.rest_luminosity_errors[valid_mask]

        # IR range integration (8-1000 μm)
        ir_mask = (valid_waves >= 8) & (valid_waves <= 1000)

        if np.sum(ir_mask) >= 2:
            # Trapezoid integration in frequency space
            ir_waves = valid_waves[ir_mask]
            ir_lums = valid_lums[ir_mask]
            ir_errs = valid_errs[ir_mask]

            # Convert to frequency space for integration
            freq_hz = 2.998e14 / ir_waves  # c/lambda in Hz
            lum_freq = ir_lums * ir_waves / 2.998e14  # Convert to L_sun/Hz

            # Sort by frequency for integration
            sort_idx = np.argsort(freq_hz)
            freq_hz_sorted = freq_hz[sort_idx]
            lum_freq_sorted = lum_freq[sort_idx]

            # Integrate using trapezoid rule
            total_ir_lum = np.trapz(lum_freq_sorted, freq_hz_sorted)

            # Simple error propagation
            rel_errors = ir_errs / ir_lums
            avg_rel_error = np.mean(rel_errors[ir_lums > 0])
            total_ir_lum_err = total_ir_lum * avg_rel_error

        else:
            # Fall back to sum of available IR points
            ir_band_mask = valid_waves >= 24  # At least include mid-IR
            if np.sum(ir_band_mask) > 0:
                total_ir_lum = np.sum(valid_lums[ir_band_mask])
                total_ir_lum_err = np.sqrt(np.sum(valid_errs[ir_band_mask] ** 2))
            else:
                total_ir_lum = 0.0
                total_ir_lum_err = 0.0

        return total_ir_lum, total_ir_lum_err

    def _process_band_results(self) -> None:
        """Process results for individual bands"""
        for map_name in self.raw_results.map_names:
            band_result = {
                "wavelength_um": self.config.maps[map_name].wavelength,
                "flux_densities_jy": self.raw_results.flux_densities[map_name],
                "flux_errors_jy": self.raw_results.flux_errors[map_name],
                "population_labels": self.raw_results.population_labels,
                "chi_squared": self.raw_results.chi_squared[map_name],
                "reduced_chi_squared": self.raw_results.reduced_chi_squared.get(
                    map_name, np.nan
                ),
                "n_sources_per_pop": [
                    self.raw_results.n_sources.get(pop, 0)
                    for pop in self.raw_results.population_labels
                ],
            }
            self.band_results[map_name] = band_result

    def get_population_summary(self) -> pd.DataFrame:
        """Get summary table of all population results"""
        data = []

        for pop_id, sed_result in self.sed_results.items():
            derived = self.derived_quantities.get(pop_id)

            row = {
                "population_id": pop_id,
                "n_sources": sed_result.n_sources,
                "median_redshift": sed_result.median_redshift,
                "median_log_mass": sed_result.median_mass,
                "n_bands": len(sed_result.wavelengths),
                "greybody_fit_success": sed_result.greybody_fit_success,
                "dust_temperature_rest_frame_K": sed_result.dust_temperature_rest_frame,
                "dust_temperature_observed_frame_K": sed_result.dust_temperature_observed_frame,
                "dust_temperature_error_K": sed_result.dust_temperature_error,
                "emissivity_index": sed_result.emissivity_index,
                "chi2_reduced": sed_result.chi2_reduced,
                "total_ir_luminosity_lsun": derived.total_ir_luminosity
                if derived
                else 0,
                "total_ir_luminosity_error_lsun": derived.total_ir_luminosity_error
                if derived
                else 0,
                "dust_mass_msun": derived.dust_mass if derived else None,
                "sfr_msun_yr": derived.star_formation_rate if derived else 0,
                "specific_sfr_yr": derived.specific_sfr if derived else 0,
            }
            data.append(row)

        return pd.DataFrame(data)

    def get_sed_table(self, population_id: str) -> pd.DataFrame:
        """Get SED data table for a specific population"""
        if population_id not in self.sed_results:
            raise ResultsError(f"Population {population_id} not found in results")

        sed = self.sed_results[population_id]

        data = {
            "wavelength_um": sed.wavelengths,
            "flux_density_jy": sed.flux_densities,
            "flux_error_jy": sed.flux_errors,
            "rest_luminosity_lsun": sed.rest_luminosities,
            "rest_luminosity_error_lsun": sed.rest_luminosity_errors,
        }

        return pd.DataFrame(data)

    def save_results(self, output_path: Path, format: str = "pickle") -> None:
        """
        Save processed results to file

        Args:
            output_path: Output file path
            format: Output format ('pickle', 'hdf5', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "pickle":
            self._save_pickle(output_path)
        elif format == "hdf5":
            self._save_hdf5(output_path)
        elif format == "csv":
            self._save_csv(output_path)
        else:
            raise ResultsError(f"Unknown output format: {format}")

        logger.info(f"Results saved to {output_path}")

    def _save_pickle(self, output_path: Path) -> None:
        """Save results as pickle file"""
        results_dict = {
            "config": self.config,
            "raw_results": self.raw_results,
            "sed_results": self.sed_results,
            "derived_quantities": self.derived_quantities,
            "band_results": self.band_results,
            "population_summary": self.get_population_summary(),
            "cosmology_summary": self.cosmology_calc.get_cosmology_summary(),
        }

        with open(output_path, "wb") as f:
            pickle.dump(results_dict, f)

    def _save_hdf5(self, output_path: Path) -> None:
        """Save results as HDF5 file"""
        try:
            import h5py
        except ImportError as err:
            raise ResultsError("h5py required for HDF5 output") from err

        with h5py.File(output_path, "w") as f:
            # Save population summary
            summary_df = self.get_population_summary()
            summary_grp = f.create_group("population_summary")
            for col in summary_df.columns:
                summary_grp.create_dataset(col, data=summary_df[col].values)

            # Save SEDs
            seds_grp = f.create_group("seds")
            for pop_id, sed in self.sed_results.items():
                sed_grp = seds_grp.create_group(pop_id)
                sed_grp.create_dataset("wavelengths", data=sed.wavelengths)
                sed_grp.create_dataset("flux_densities", data=sed.flux_densities)
                sed_grp.create_dataset("flux_errors", data=sed.flux_errors)
                sed_grp.create_dataset("rest_luminosities", data=sed.rest_luminosities)
                sed_grp.create_dataset(
                    "rest_luminosity_errors", data=sed.rest_luminosity_errors
                )

                # Add metadata
                sed_grp.attrs["median_redshift"] = sed.median_redshift
                sed_grp.attrs["median_mass"] = sed.median_mass
                sed_grp.attrs["n_sources"] = sed.n_sources

    def _save_csv(self, output_path: Path) -> None:
        """Save results as CSV files"""
        base_path = output_path.with_suffix("")

        # Save population summary
        summary_df = self.get_population_summary()
        summary_df.to_csv(f"{base_path}_summary.csv", index=False)

        # Save individual SEDs
        for pop_id, _sed in self.sed_results.items():
            sed_df = self.get_sed_table(pop_id)
            safe_pop_id = pop_id.replace("__", "_").replace(".", "p")
            sed_df.to_csv(f"{base_path}_sed_{safe_pop_id}.csv", index=False)

    def print_results_summary(self) -> None:
        """Print a formatted summary of results"""
        print("=== Simstack4 Results Summary ===")
        print(f"Processed {len(self.sed_results)} populations")
        print(f"Bands: {len(self.raw_results.map_names)}")
        print(
            f"Greybody fitting: β {'fixed' if self.greybody_fitter.fix_beta else 'free'} = {self.greybody_fitter.beta_fixed if self.greybody_fitter.fix_beta else 'fitted'}"
        )
        print()

        # Print greybody fit quality
        successful_fits = sum(
            1 for sed in self.sed_results.values() if sed.greybody_fit_success
        )
        print(
            f"Greybody Fit Success Rate: {successful_fits}/{len(self.sed_results)} ({100*successful_fits/len(self.sed_results):.1f}%)"
        )

        if successful_fits > 0:
            temps_rest = [
                sed.dust_temperature_rest_frame
                for sed in self.sed_results.values()
                if sed.greybody_fit_success and sed.dust_temperature_rest_frame
            ]
            temps_obs = [
                sed.dust_temperature_observed_frame
                for sed in self.sed_results.values()
                if sed.greybody_fit_success and sed.dust_temperature_observed_frame
            ]
            chi2s = [
                sed.chi2_reduced
                for sed in self.sed_results.values()
                if sed.greybody_fit_success and sed.chi2_reduced
            ]

            if temps_rest:
                print(
                    f"Rest-frame temperature range: {np.min(temps_rest):.1f} - {np.max(temps_rest):.1f} K (median: {np.median(temps_rest):.1f} K)"
                )
            if temps_obs:
                print(
                    f"Observed-frame temperature range: {np.min(temps_obs):.1f} - {np.max(temps_obs):.1f} K (median: {np.median(temps_obs):.1f} K)"
                )
            if chi2s:
                print(
                    f"χ²_red range: {np.min(chi2s):.2f} - {np.max(chi2s):.2f} (median: {np.median(chi2s):.2f})"
                )

        print()

        # Print fit quality
        print("Fit Quality:")
        for map_name in self.raw_results.map_names:
            # chi2 = self.raw_results.chi_squared[map_name]
            red_chi2 = self.raw_results.reduced_chi_squared.get(map_name, np.nan)
            wave = self.config.maps[map_name].wavelength
            print(f"  {map_name} ({wave}μm): χ²_red = {red_chi2:.2f}")
        print()

        # Print population results
        summary_df = self.get_population_summary()
        if len(summary_df) > 0:
            print("Population Results:")
            print(
                f"{'Population':<30} {'N_src':<8} {'z_med':<8} {'T_rest[K]':<10} {'T_obs[K]':<10} {'L_IR[L☉]':<12} {'SFR[M☉/yr]':<12}"
            )
            print("-" * 110)

            for _, row in summary_df.iterrows():
                pop_id = row["population_id"][:29]  # Truncate long names
                n_src = int(row["n_sources"])
                z_med = row["median_redshift"]
                t_rest = (
                    row["dust_temperature_rest_frame_K"]
                    if pd.notna(row["dust_temperature_rest_frame_K"])
                    else 0
                )
                t_obs = (
                    row["dust_temperature_observed_frame_K"]
                    if pd.notna(row["dust_temperature_observed_frame_K"])
                    else 0
                )
                l_ir = row["total_ir_luminosity_lsun"]
                sfr = row["sfr_msun_yr"]

                print(
                    f"{pop_id:<30} {n_src:<8} {z_med:<8.2f} {t_rest:<10.1f} {t_obs:<10.1f} {l_ir:<12.2e} {sfr:<12.1f}"
                )

    def plot_results_overview(self, save_path: Path | None = None) -> None:
        """Create overview plots of results (requires matplotlib)"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not available for plotting")
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: SEDs
        ax1 = axes[0, 0]
        for pop_id, sed in self.sed_results.items():
            if sed.n_sources > 0:
                ax1.errorbar(
                    sed.wavelengths,
                    sed.flux_densities,
                    yerr=sed.flux_errors,
                    label=pop_id[:20],
                    marker="o",
                )
        ax1.set_xlabel("Wavelength [μm]")
        ax1.set_ylabel("Flux Density [Jy]")
        ax1.set_title("Stacked SEDs")
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot 2: SFR vs stellar mass
        ax2 = axes[0, 1]
        summary_df = self.get_population_summary()
        if len(summary_df) > 0:
            mask = summary_df["sfr_msun_yr"] > 0
            if np.sum(mask) > 0:
                ax2.scatter(
                    summary_df.loc[mask, "median_log_mass"],
                    summary_df.loc[mask, "sfr_msun_yr"],
                )
        ax2.set_xlabel("log(M*/M☉)")
        ax2.set_ylabel("SFR [M☉/yr]")
        ax2.set_title("Star Formation Rate vs Stellar Mass")
        ax2.set_yscale("log")
        ax2.grid(True, alpha=0.3)

        # Plot 3: Fit quality
        ax3 = axes[1, 0]
        map_names = list(self.band_results.keys())
        chi2_values = [
            self.band_results[name]["reduced_chi_squared"] for name in map_names
        ]
        wavelengths = [self.band_results[name]["wavelength_um"] for name in map_names]

        ax3.scatter(wavelengths, chi2_values)
        ax3.axhline(y=1, color="r", linestyle="--", alpha=0.5, label="Perfect fit")
        ax3.set_xlabel("Wavelength [μm]")
        ax3.set_ylabel("Reduced χ²")
        ax3.set_title("Fit Quality by Band")
        ax3.set_xscale("log")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot 4: Number of sources per population
        ax4 = axes[1, 1]
        if len(summary_df) > 0:
            pop_labels = [pid[:15] for pid in summary_df["population_id"]]
            # n_sources = summary_df["n_sources"]

            # bars = ax4.bar(range(len(pop_labels)), n_sources)
            ax4.set_xlabel("Population")
            ax4.set_ylabel("Number of Sources")
            ax4.set_title("Sources per Population")
            ax4.set_xticks(range(len(pop_labels)))
            ax4.set_xticklabels(pop_labels, rotation=45, ha="right")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Overview plot saved to {save_path}")
        else:
            plt.show()

    @classmethod
    def load_results(cls, file_path: Path) -> "SimstackResults":
        """Load results from saved file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise ResultsError(f"Results file not found: {file_path}")

        if file_path.suffix == ".pkl":
            with open(file_path, "rb") as f:
                results_dict = pickle.load(f)

            # Reconstruct the results object
            # This is a simplified version - full implementation would
            # properly reconstruct all components
            logger.info(f"Loaded results from {file_path}")
            return results_dict

        else:
            raise ResultsError(f"Unsupported file format: {file_path.suffix}")


def create_results_processor(
    config: SimstackConfig,
    stacking_results: StackingResults,
    population_manager: PopulationManager,
    fix_beta: bool = True,
    beta_fixed: float = 1.8,
) -> SimstackResults:
    """
    Convenience function to create results processor

    Args:
        config: Simstack configuration
        stacking_results: Raw stacking results
        population_manager: Population manager
        fix_beta: Whether to fix the emissivity index beta in greybody fitting
        beta_fixed: Fixed value of beta if fix_beta=True (recommended: 1.8 for galaxies)

    Returns:
        Processed SimstackResults object
    """
    return SimstackResults(
        config,
        stacking_results,
        population_manager,
        fix_beta=fix_beta,
        beta_fixed=beta_fixed,
    )
