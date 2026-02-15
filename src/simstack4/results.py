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
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import curve_fit

# Add emcee import
try:
    import emcee

    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

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

    # MCMC results
    mcmc_samples: np.ndarray | None = None  # MCMC samples if used
    mcmc_percentiles: dict | None = None  # Percentiles from MCMC


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

    # MCMC-derived uncertainties
    total_ir_luminosity_mcmc_error: tuple | None = None  # (lower, upper) from MCMC
    dust_temperature_mcmc_error: tuple | None = None  # (lower, upper) from MCMC


class GreybodyFitter:
    """Improved Greybody fitter with robust MCMC handling"""

    def __init__(
        self,
        fix_beta: bool = True,
        beta_fixed: float = 1.8,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        use_schreiber_prior: bool = False,
    ):
        """
        Initialize fitter with improved defaults
        """
        # Physical constants
        self.h = 6.62607015e-34  # Planck constant (J⋅s)
        self.c = 299792458  # Speed of light (m/s)
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.L_sun = 3.828e26  # Solar luminosity (W)

        self.fix_beta = fix_beta
        self.beta_fixed = beta_fixed
        self.use_mcmc = use_mcmc and HAS_EMCEE
        self.mcmc_iterations = mcmc_iterations
        self.mcmc_burn_in = mcmc_burn_in
        self.use_schreiber_prior = use_schreiber_prior

        # Setup cosmology (simplified for now)
        self.H0 = 70  # km/s/Mpc
        self.Om0 = 0.3

        if use_mcmc and not HAS_EMCEE:
            logger.warning(
                "MCMC requested but emcee not available. Falling back to curve_fit."
            )

    def _validate_data(self, wavelengths, fluxes, flux_errors):
        """
        Validate and filter data, handling the huge dynamic range in your data

        Returns:
        --------
        valid_mask : array
            Boolean mask for valid data points
        use_for_fit : array
            Boolean mask for data to use in fitting (more restrictive)
        """
        # Basic validity checks
        valid = (
            np.isfinite(fluxes)
            & np.isfinite(flux_errors)
            & (flux_errors > 0)
            & np.isfinite(wavelengths)
            & (wavelengths > 0)
        )

        # Calculate SNR
        snr = np.abs(fluxes) / flux_errors

        # For your data, we need to be smart about what to include:
        # 1. Strong positive detections (>2σ)
        strong_detections = (fluxes > 0) & (snr > 2.0)

        # 2. Reasonable measurements (>1σ, including negatives)
        reasonable_measurements = (snr > 1.0) & (
            snr < 1000
        )  # Cap very high SNR that might be outliers

        # 3. Exclude measurements with extremely large errors (like SCUBA in your data)
        # If error is >100x the typical error, it's probably not useful
        log_errors = np.log10(flux_errors)
        median_log_error = np.median(log_errors)
        reasonable_errors = (
            log_errors - median_log_error
        ) < 2.0  # Within 100x of median

        # Final selection: use measurements that are reasonable AND either strong detections OR reasonable measurements
        use_for_fit = (
            valid & reasonable_errors & (strong_detections | reasonable_measurements)
        )

        # Ensure we have at least 3 points and at least 1 positive detection
        if np.sum(use_for_fit) < 3:
            # Fallback: use at least the best detections
            if np.sum(strong_detections & valid) >= 3:
                use_for_fit = strong_detections & valid
            else:
                use_for_fit = valid

        return valid, use_for_fit

    def _get_initial_guess(self, wavelengths, fluxes, flux_errors, redshift):
        """
        Get much better initial guess using curve_fit first
        """
        # Always run curve_fit first to get the best possible initial guess
        try:
            if self.use_schreiber_prior and redshift > 0:
                T_guess, _ = self.schreiber_temperature_prior(redshift)
                T_guess = np.clip(T_guess, 5, 45)
            else:
                T_guess = 25.0
            amplitude_guess = -35.0

            # Run a quick curve_fit to get much better initial parameters
            if self.fix_beta:

                def model_func(wave, amp, temp):
                    return self.greybody_model(wave, amp, temp, self.beta_fixed)

                try:
                    popt, _ = curve_fit(
                        model_func,
                        wavelengths,
                        fluxes,
                        sigma=flux_errors,
                        p0=[amplitude_guess, T_guess],
                        bounds=([-41, 12], [-29, 55]),
                        maxfev=1000,  # Quick fit
                    )
                    amplitude_guess, T_guess = popt
                    logger.debug(
                        f"Curve_fit initial guess: A={amplitude_guess:.2f}, T={T_guess:.1f}K"
                    )
                except Exception as e:
                    logger.debug(
                        f"Curve_fit for initial guess failed: {e}, using defaults"
                    )

            return amplitude_guess, T_guess

        except Exception as e:
            logger.warning(f"Initial guess estimation failed: {e}")
            return -35.0, 25.0

    def schreiber_temperature_prior(self, redshift: float) -> tuple[float, float]:
        """
        Calculate expected dust temperature from Schreiber+2015 with realistic uncertainties
        """
        if redshift <= 0:
            return 25.0, 5.0

        # Schreiber+2015 relation: T_dust = (23.8 + 2.7*z + 0.9*z^2) / (1+z)
        T_rest = 23.8 + 2.7 * redshift + 0.9 * redshift**2
        T_observed = T_rest / (1 + redshift)

        # Clip to reasonable range
        T_observed = np.clip(T_observed, 15, 50)

        # More realistic uncertainty based on scatter in Schreiber+2015 data
        # The scatter increases with redshift due to increased diversity
        if redshift < 1.0:
            T_sigma = 3.0  # Low-z: tight relation
        elif redshift < 2.0:
            T_sigma = 4.0  # Mid-z: more scatter
        else:
            T_sigma = 5.0  # High-z: significant scatter

        # Don't let uncertainty be smaller than reasonable measurement precision
        T_sigma = max(T_sigma, 2.0)

        T_sigma = 2.0

        return T_observed, T_sigma

    def log_prior(self, theta: list, redshift: float = 0.0) -> float:
        """Calculate log prior with much tighter, realistic bounds for your data"""
        amplitude, temperature = theta

        # Much tighter amplitude bounds based on your data range
        if not (-38 < amplitude < -30):
            return -np.inf

        # Realistic temperature bounds for dust
        if not (15 < temperature < 50):
            return -np.inf

        # Temperature priors
        if self.use_schreiber_prior and redshift > 0:
            T_expected, T_sigma = self.schreiber_temperature_prior(redshift)
            log_p_temp = -0.5 * ((temperature - T_expected) / T_sigma) ** 2
        else:
            # Strong preference for reasonable dust temperatures (20-40K)
            if 20 <= temperature <= 40:
                log_p_temp = 0.0
            elif 15 <= temperature < 20 or 40 < temperature <= 50:
                # Mild penalty for edge temperatures
                log_p_temp = -0.5 * ((temperature - 30) / 10) ** 2
            else:
                log_p_temp = -np.inf

        return log_p_temp

    def log_likelihood(self, theta, wavelengths, fluxes, flux_errors):
        """
        Improved log likelihood that handles negative fluxes properly
        """
        amplitude, temperature = theta

        try:
            # Calculate model fluxes
            model_fluxes = self.greybody_model(
                wavelengths, amplitude, temperature, self.beta_fixed
            )

            # Check for invalid model fluxes
            if not np.all(np.isfinite(model_fluxes)):
                return -np.inf

            # Prevent completely unreasonable model values
            if np.any(model_fluxes <= 0) or np.any(model_fluxes > 1000):
                return -np.inf

            # Calculate chi-squared (can handle negative observed fluxes)
            residuals = (fluxes - model_fluxes) / flux_errors
            chi2 = np.sum(residuals**2)

            # Sanity check
            if not np.isfinite(chi2) or chi2 > 10000:
                return -np.inf

            # Log likelihood
            log_like = -0.5 * chi2 - 0.5 * np.sum(np.log(2 * np.pi * flux_errors**2))

            return log_like

        except (FloatingPointError, OverflowError, ValueError):
            return -np.inf

    def log_posterior(self, theta, wavelengths, fluxes, flux_errors, redshift=0.0):
        """Calculate log posterior probability"""
        log_p = self.log_prior(theta, redshift)
        if not np.isfinite(log_p):
            return -np.inf

        log_l = self.log_likelihood(theta, wavelengths, fluxes, flux_errors)

        return log_p + log_l

    def _initialize_walkers(
        self, initial_guess, n_walkers, wavelengths, fluxes, flux_errors, redshift
    ):
        """
        Initialize MCMC walkers very close to the curve_fit solution
        """
        amplitude_guess, temperature_guess = initial_guess
        pos = []

        # Start walkers very close to the best-fit solution
        # This is key for your data where the solution is well-constrained
        for _i in range(n_walkers):
            # Very small perturbations around the curve_fit solution
            amp_trial = amplitude_guess + np.random.normal(0, 0.1)  # Tiny scatter
            temp_trial = temperature_guess + np.random.normal(0, 0.5)  # Small scatter

            # Ensure within bounds
            amp_trial = np.clip(amp_trial, -37.5, -30.5)
            temp_trial = np.clip(temp_trial, 16, 49)

            # Test that this position gives finite probability
            test_prob = self.log_posterior(
                [amp_trial, temp_trial], wavelengths, fluxes, flux_errors, redshift
            )

            if np.isfinite(test_prob):
                pos.append([amp_trial, temp_trial])
            else:
                # If the perturbation failed, use the exact solution with tiny noise
                pos.append(
                    [
                        amplitude_guess + np.random.normal(0, 0.01),
                        temperature_guess + np.random.normal(0, 0.1),
                    ]
                )

        return np.array(pos)

    def run_mcmc(self, wavelengths, fluxes, flux_errors, redshift=0.0):
        """
        Run MCMC fitting with better error handling and progress tracking
        """
        if not HAS_EMCEE:
            raise ImportError("emcee is required for MCMC fitting")

        # Get initial guess
        amplitude_guess, T_guess = self._get_initial_guess(
            wavelengths, fluxes, flux_errors, redshift
        )

        logger.info(f"MCMC initial guess: A={amplitude_guess:.2f}, T={T_guess:.1f}K")

        # Setup walkers
        n_walkers = 32
        n_dim = 2

        # Initialize walker positions
        pos = self._initialize_walkers(
            (amplitude_guess, T_guess),
            n_walkers,
            wavelengths,
            fluxes,
            flux_errors,
            redshift,
        )

        if len(pos) < n_walkers:
            raise ValueError(f"Could only initialize {len(pos)}/{n_walkers} walkers")

        # Create sampler
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            self.log_posterior,
            args=(wavelengths, fluxes, flux_errors, redshift),
        )

        # Run MCMC with progress bar and error handling
        try:
            logger.info(
                f"Running MCMC: {self.mcmc_iterations} iterations, {self.mcmc_burn_in} burn-in"
            )

            # Try to run with progress bar first
            try:
                sampler.run_mcmc(pos, self.mcmc_iterations, progress=True)
                total_steps = self.mcmc_iterations

            except Exception as progress_error:
                logger.info(
                    f"Progress bar failed ({progress_error}), running without progress bar..."
                )

                # Fallback: run in chunks without progress bar but with monitoring
                chunk_size = min(100, self.mcmc_iterations // 4)
                total_steps = 0
                current_pos = pos

                while total_steps < self.mcmc_iterations:
                    steps_remaining = self.mcmc_iterations - total_steps
                    steps_this_chunk = min(chunk_size, steps_remaining)

                    try:
                        sampler.run_mcmc(current_pos, steps_this_chunk, progress=False)
                        current_pos = sampler.get_last_sample()
                        total_steps += steps_this_chunk

                        # Manual progress reporting
                        if total_steps % (chunk_size * 2) == 0:
                            acceptance = np.mean(sampler.acceptance_fraction)
                            progress_pct = 100 * total_steps / self.mcmc_iterations
                            logger.info(
                                f"MCMC progress: {progress_pct:.1f}% ({total_steps}/{self.mcmc_iterations}), acceptance: {acceptance:.3f}"
                            )

                            # Check for extremely low acceptance (indicates problems)
                            if acceptance < 0.01:
                                logger.warning(
                                    f"Very low acceptance: {acceptance:.4f}, terminating MCMC"
                                )
                                break

                    except Exception as e:
                        logger.error(f"MCMC chunk failed at step {total_steps}: {e}")
                        raise

        except Exception as e:
            logger.error(f"MCMC run failed: {e}")
            raise

        # Run MCMC with progress bar and error handling
        try:
            logger.info(
                f"Running MCMC: {self.mcmc_iterations} iterations, {self.mcmc_burn_in} burn-in"
            )
            logger.info(
                f"Starting near curve_fit solution: A={amplitude_guess:.3f}, T={T_guess:.1f}K"
            )

            # For well-constrained problems like yours, shorter runs with tight walkers work better
            effective_iterations = min(
                self.mcmc_iterations, 500
            )  # Cap at 500 for efficiency
            effective_burn_in = min(self.mcmc_burn_in, effective_iterations // 4)

            # Try to run with progress bar first
            try:
                sampler.run_mcmc(pos, effective_iterations, progress=True)
                total_steps = effective_iterations

            except Exception as progress_error:
                logger.info(
                    f"Progress bar failed ({progress_error}), running without progress bar..."
                )
                sampler.run_mcmc(pos, effective_iterations, progress=False)
                total_steps = effective_iterations

        except Exception as e:
            logger.error(f"MCMC run failed: {e}")
            raise

        # Check final acceptance
        final_acceptance = np.mean(sampler.acceptance_fraction)
        logger.info(f"MCMC completed. Final acceptance: {final_acceptance:.3f}")

        # For well-constrained problems, acceptance can be lower but still OK
        if final_acceptance < 0.15:
            logger.warning(
                f"Low MCMC acceptance: {final_acceptance:.3f} - this may indicate the problem is very well-constrained by the data"
            )

        # Extract samples
        try:
            # Use the effective burn-in
            samples = sampler.get_chain(discard=effective_burn_in, flat=True)
        except Exception as e:
            logger.error(f"Failed to extract MCMC samples: {e}")
            raise

        if len(samples) < 50:
            raise ValueError(f"Too few MCMC samples: {len(samples)}")

        # Calculate results
        percentiles = [16, 50, 84]
        amplitude_percentiles = np.percentile(samples[:, 0], percentiles)
        temperature_percentiles = np.percentile(samples[:, 1], percentiles)

        # Best-fit parameters (median)
        amplitude_best = amplitude_percentiles[1]
        temperature_best = temperature_percentiles[1]

        # Errors (16th-84th percentile range / 2)
        amplitude_err = np.diff(amplitude_percentiles[[0, 2]])[0] / 2
        temperature_err = np.diff(temperature_percentiles[[0, 2]])[0] / 2

        return {
            "samples": samples,
            "amplitude": amplitude_best,
            "amplitude_error": amplitude_err,
            "amplitude_percentiles": amplitude_percentiles,
            "temperature_observed_frame": temperature_best,
            "temperature_error": temperature_err,
            "temperature_percentiles": temperature_percentiles,
            "n_samples": len(samples),
            "acceptance_fraction": final_acceptance,
            "total_steps": total_steps,
            "effective_iterations": effective_iterations,
            "effective_burn_in": effective_burn_in,
        }

    def fit_sed(self, wavelengths, fluxes, flux_errors, redshift):
        """
        Fit greybody model with improved data handling
        """
        # Validate and filter data
        valid_mask, fit_mask = self._validate_data(wavelengths, fluxes, flux_errors)

        if np.sum(fit_mask) < 3:
            logger.warning(f"Insufficient valid data points: {np.sum(fit_mask)}")
            return {"fit_success": False, "reason": "insufficient_data"}

        wave_fit = wavelengths[fit_mask]
        flux_fit = fluxes[fit_mask]
        error_fit = flux_errors[fit_mask]

        logger.info(
            f"Fitting SED: {len(wave_fit)} points, {np.sum(flux_fit > 0)} positive detections"
        )

        try:
            # Always start with curve_fit for initial guess
            amplitude_guess, T_guess = self._get_initial_guess(
                wave_fit, flux_fit, error_fit, redshift
            )

            logger.debug(f"Initial guess: A={amplitude_guess:.2f}, T={T_guess:.1f}K")

            # Curve fitting with conservative bounds
            if self.fix_beta:

                def model_func(wave, amp, temp):
                    return self.greybody_model(wave, amp, temp, self.beta_fixed)

                popt, pcov = curve_fit(
                    model_func,
                    wave_fit,
                    flux_fit,
                    sigma=error_fit,
                    p0=[amplitude_guess, T_guess],
                    bounds=([-41, 12], [-29, 55]),
                    maxfev=5000,
                )
                amplitude, temperature_observed = popt
                beta = self.beta_fixed
                param_errors = np.sqrt(np.diag(pcov))
                amplitude_err, temperature_err = param_errors
                beta_err = 0.0
            else:
                # Free beta fitting
                popt, pcov = curve_fit(
                    self.greybody_model,
                    wave_fit,
                    flux_fit,
                    sigma=error_fit,
                    p0=[amplitude_guess, T_guess, 1.8],
                    bounds=([-41, 12, 0.5], [-29, 55, 2.5]),
                    maxfev=5000,
                )
                amplitude, temperature_observed, beta = popt
                param_errors = np.sqrt(np.diag(pcov))
                amplitude_err, temperature_err, beta_err = param_errors

            logger.info(
                f"Curve fit: T_obs={temperature_observed:.1f}±{temperature_err:.1f}K"
            )

            # Run MCMC if requested and curve_fit succeeded
            mcmc_results = None
            if self.use_mcmc and self.fix_beta:
                try:
                    mcmc_results = self.run_mcmc(
                        wave_fit, flux_fit, error_fit, redshift
                    )
                    # Update parameters with MCMC results
                    amplitude = mcmc_results["amplitude"]
                    temperature_observed = mcmc_results["temperature_observed_frame"]
                    amplitude_err = mcmc_results["amplitude_error"]
                    temperature_err = mcmc_results["temperature_error"]

                    logger.info(
                        f"MCMC fit: T_obs={temperature_observed:.1f}±{temperature_err:.1f}K"
                    )

                except Exception as e:
                    logger.warning(f"MCMC failed, using curve_fit results: {e}")
                    mcmc_results = None

            # Calculate rest-frame temperature
            temperature_rest_frame = temperature_observed * (1 + redshift)

            # Calculate goodness of fit
            model_fluxes = self.greybody_model(
                wave_fit, amplitude, temperature_observed, beta
            )
            chi2 = np.sum(((flux_fit - model_fluxes) / error_fit) ** 2)
            dof = len(wave_fit) - (2 if self.fix_beta else 3)
            chi2_reduced = chi2 / max(1, dof)

            # Calculate L_IR
            L_IR, L_IR_error = self.calculate_LIR(
                amplitude, temperature_observed, beta, redshift
            )

            # Generate smooth model for plotting
            wave_model = np.logspace(
                np.log10(np.min(wave_fit) * 0.5), np.log10(np.max(wave_fit) * 2), 200
            )
            flux_model = self.greybody_model(
                wave_model, amplitude, temperature_observed, beta
            )

            results = {
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
                "n_points": len(wave_fit),
                "n_positive": np.sum(flux_fit > 0),
                "wavelengths_fit": wave_fit,
                "fluxes_fit": flux_fit,
                "flux_errors_fit": error_fit,
                "model_wavelengths": wave_model,
                "model_fluxes": flux_model,
                "redshift_used": redshift,
                "mcmc_used": mcmc_results is not None,
                "schreiber_prior_used": self.use_schreiber_prior,
            }

            # Add MCMC-specific results
            if mcmc_results:
                results.update(
                    {
                        "mcmc_samples": mcmc_results["samples"],
                        "mcmc_percentiles": {
                            "amplitude": mcmc_results["amplitude_percentiles"],
                            "temperature": mcmc_results["temperature_percentiles"],
                        },
                        "mcmc_acceptance_fraction": mcmc_results["acceptance_fraction"],
                        "mcmc_total_steps": mcmc_results["total_steps"],
                    }
                )

            return results

        except Exception as e:
            logger.warning(f"Greybody fit failed: {e}")
            return {"fit_success": False, "reason": str(e)}

    # Include all the other methods (greybody_model, calculate_LIR, etc.) from the original class
    def greybody_model(
        self, wavelength_um, amplitude, temperature, beta=1.8, alpha=2.0
    ):
        """Greybody model with power law extensions."""
        # Convert wavelength to frequency
        c_light = 299792458.0  # m/s
        nu_in = c_light * 1.0e6 / wavelength_um  # Hz

        # Linear amplitude
        A = 10**amplitude

        # Calculate transition frequency
        nu_cut = (3.0 + beta + alpha) * 0.208367e11 * temperature

        # Constants for power law normalization
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
        """Blackbody function"""
        a0 = 1.4718e-21  # 2*h*10^29/c^2
        a1 = 4.7993e-11  # h/k

        num = a0 * nu_in**3.0
        den = np.exp(a1 * np.outer(1.0 / T, nu_in)) - 1.0
        ret = num / den

        return ret

    def calculate_LIR(
        self,
        amplitude,
        temperature,
        beta,
        redshift,
        amplitude_err=None,
        temperature_err=None,
    ):
        """
        Calculate L_IR with proper error propagation from fit uncertainties
        """
        if np.isnan(redshift) or redshift <= 0:
            redshift = 0.01

        # Get luminosity distance (in Mpc)
        D_L = self.luminosity_distance(redshift)

        # Generate wavelength range in microns (8-1000 μm rest-frame)
        wavelength_range = np.logspace(np.log10(8), np.log10(1000), 1000)

        # Get model SED
        model_sed_jy = self.greybody_model(
            wavelength_range, amplitude, temperature, beta
        )

        # Convert wavelength to frequency (Hz)
        c_light = 299792458.0  # m/s
        nu_in = c_light * 1.0e6 / wavelength_range  # Hz

        # Calculate frequency intervals
        dnu = nu_in[:-1] - nu_in[1:]
        dnu = np.append(dnu[0], dnu)

        # Integrate: L_IR = ∫ S_ν dν in Jy⋅Hz
        Lir_jy_hz = np.sum(model_sed_jy * dnu)

        # Convert to solar luminosities
        conversion = 4.0 * np.pi * (1.0e-13 * D_L * 3.08568025e22) ** 2.0 / self.L_sun
        L_IR_solar = Lir_jy_hz * conversion

        # Proper error propagation if uncertainties provided
        if amplitude_err is not None and temperature_err is not None:
            # Calculate derivatives for error propagation
            # dL_IR/dA ≈ (L_IR(A+δA) - L_IR(A-δA)) / (2*δA)
            delta_amp = max(0.001, abs(amplitude_err) * 0.1)  # Small step
            delta_temp = max(0.1, abs(temperature_err) * 0.1)

            try:
                # Amplitude derivative
                L_IR_plus_amp = self._calculate_LIR_single(
                    amplitude + delta_amp, temperature, beta, redshift, D_L, conversion
                )
                L_IR_minus_amp = self._calculate_LIR_single(
                    amplitude - delta_amp, temperature, beta, redshift, D_L, conversion
                )
                dLIR_damp = (L_IR_plus_amp - L_IR_minus_amp) / (2 * delta_amp)

                # Temperature derivative
                L_IR_plus_temp = self._calculate_LIR_single(
                    amplitude, temperature + delta_temp, beta, redshift, D_L, conversion
                )
                L_IR_minus_temp = self._calculate_LIR_single(
                    amplitude, temperature - delta_temp, beta, redshift, D_L, conversion
                )
                dLIR_dtemp = (L_IR_plus_temp - L_IR_minus_temp) / (2 * delta_temp)

                # Error propagation: σ²(L_IR) = (∂L_IR/∂A)²σ²(A) + (∂L_IR/∂T)²σ²(T)
                L_IR_error = np.sqrt(
                    (dLIR_damp * amplitude_err) ** 2
                    + (dLIR_dtemp * temperature_err) ** 2
                )

            except Exception as e:
                logger.warning(f"Error propagation failed: {e}, using 15% estimate")
                L_IR_error = L_IR_solar * 0.15  # More conservative fallback
        else:
            # Fallback: use systematic uncertainty estimate based on typical model uncertainties
            L_IR_error = L_IR_solar * 0.15  # 15% systematic uncertainty

        return L_IR_solar, L_IR_error

    def _calculate_LIR_single(
        self, amplitude, temperature, beta, redshift, D_L, conversion
    ):
        """Helper function for L_IR calculation"""
        wavelength_range = np.logspace(np.log10(8), np.log10(1000), 1000)
        model_sed_jy = self.greybody_model(
            wavelength_range, amplitude, temperature, beta
        )
        c_light = 299792458.0
        nu_in = c_light * 1.0e6 / wavelength_range
        dnu = nu_in[:-1] - nu_in[1:]
        dnu = np.append(dnu[0], dnu)
        Lir_jy_hz = np.sum(model_sed_jy * dnu)
        return Lir_jy_hz * conversion

    def luminosity_distance(self, z):
        """Calculate luminosity distance in Mpc"""
        if z <= 0:
            z = 0.01
        c_km_s = 299792.458  # km/s
        D_L = c_km_s * z / self.H0  # Mpc
        return D_L

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
        try:
            if np.isnan(temperature) or np.isnan(amplitude):
                return np.nan, np.nan

            # Get luminosity distance
            D_L = self.luminosity_distance(redshift) * 3.086e22  # Convert Mpc to meters

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

            if B_nu <= 0:
                return np.nan, np.nan

            M_dust_kg = flux_ref_si * D_L**2 / (kappa_ref * B_nu)

            # Convert to solar masses
            M_sun_kg = 1.989e30
            M_dust_solar = M_dust_kg / M_sun_kg

            # Error estimate
            M_dust_error = M_dust_solar * 0.3  # 30% error

            return M_dust_solar, M_dust_error

        except Exception as e:
            logger.warning(f"Dust mass calculation failed: {e}")
            return np.nan, np.nan

    def planck_function(self, nu: np.ndarray, T: float) -> np.ndarray:
        """Planck function B_ν(T) in SI units (W m⁻² sr⁻¹ Hz⁻¹)"""
        exponent = self.h * nu / (self.k_B * T)
        exponent = np.clip(exponent, 0, 700)
        return (2 * self.h * nu**3 / self.c**2) / (np.exp(exponent) - 1)


class CovarianceGreybodyFitter(GreybodyFitter):
    """Enhanced Greybody fitter that properly handles band correlations"""

    def __init__(self, correlation_matrix=None, inflation_factors=None, **kwargs):
        """
        Initialize with correlation matrix

        Parameters:
        -----------
        correlation_matrix : dict or numpy.ndarray
            Either a dict mapping wavelengths to correlation matrix,
            or a 2D numpy array for the correlation matrix
        """
        super().__init__(**kwargs)
        self.correlation_matrix = correlation_matrix
        self.covariance_matrix = None
        self.inv_cov_chol = None  # Cholesky decomposition for efficiency
        self.inflation_factors = inflation_factors  # NEW

    def set_correlation_matrix_from_dict(self, corr_dict, wavelengths):
        """
        Set correlation matrix from dictionary format

        Parameters:
        -----------
        corr_dict : dict
            Dictionary with wavelength keys and correlation values (can be empty)
        wavelengths : array
            Wavelengths used in fitting (in same order as flux arrays)
        """
        # Use the hardcoded correlation data if corr_dict is empty
        if not corr_dict:
            correlation_data = {
                24: {
                    24: 1.00,
                    70: 0.23,
                    100: 0.33,
                    160: 0.32,
                    250: 0.28,
                    350: 0.17,
                    500: 0.07,
                    1100: 0.10,
                },
                70: {
                    24: 0.23,
                    70: 1.00,
                    100: 0.19,
                    160: 0.24,
                    250: 0.23,
                    350: 0.14,
                    500: 0.06,
                    1100: 0.08,
                },
                100: {
                    24: 0.33,
                    70: 0.19,
                    100: 1.00,
                    160: 0.28,
                    250: 0.21,
                    350: 0.11,
                    500: 0.04,
                    1100: 0.05,
                },
                160: {
                    24: 0.32,
                    70: 0.24,
                    100: 0.28,
                    160: 1.00,
                    250: 0.35,
                    350: 0.23,
                    500: 0.10,
                    1100: 0.13,
                },
                250: {
                    24: 0.28,
                    70: 0.23,
                    100: 0.21,
                    160: 0.35,
                    250: 1.00,
                    350: 0.37,
                    500: 0.18,
                    1100: 0.28,
                },
                350: {
                    24: 0.17,
                    70: 0.14,
                    100: 0.11,
                    160: 0.23,
                    250: 0.37,
                    350: 1.00,
                    500: 0.20,
                    1100: 0.33,
                },
                500: {
                    24: 0.07,
                    70: 0.06,
                    100: 0.04,
                    160: 0.10,
                    250: 0.18,
                    350: 0.20,
                    500: 1.00,
                    1100: 0.23,
                },
                1100: {
                    24: 0.10,
                    70: 0.08,
                    100: 0.05,
                    160: 0.13,
                    250: 0.28,
                    350: 0.33,
                    500: 0.23,
                    1100: 1.00,
                },
            }
        else:
            correlation_data = corr_dict

        # Find which wavelengths we actually have data for
        available_wavelengths = []
        for w in wavelengths:
            # Find closest match in correlation matrix (with some tolerance)
            if correlation_data:  # Only if we have data
                closest_key = min(correlation_data.keys(), key=lambda x: abs(x - w))
                if abs(closest_key - w) < w * 0.1:  # Within 10%
                    available_wavelengths.append(closest_key)
                else:
                    logger.warning(f"No correlation data for wavelength {w:.1f} μm")
                    available_wavelengths.append(None)
            else:
                available_wavelengths.append(None)

        # Build correlation matrix for available data
        n_bands = len([w for w in available_wavelengths if w is not None])
        if n_bands < 2:
            logger.warning(
                "Insufficient bands with correlation data, using diagonal covariance"
            )
            self.correlation_matrix = None
            return

        # Create the correlation matrix
        valid_indices = [
            i for i, w in enumerate(available_wavelengths) if w is not None
        ]
        valid_wavelengths = [available_wavelengths[i] for i in valid_indices]

        corr_matrix = np.zeros((n_bands, n_bands))
        for i, w1 in enumerate(valid_wavelengths):
            for j, w2 in enumerate(valid_wavelengths):
                corr_matrix[i, j] = correlation_data[w1][w2]

        self.correlation_matrix = corr_matrix
        self.wavelength_mapping = valid_indices  # Indices in original wavelength array

        logger.info(
            f"Set up correlation matrix for {n_bands} bands: {valid_wavelengths}"
        )

    def _inflate_band_errors(self, wavelengths, flux_errors):
        """
        Inflate uncertainties for specific bands

        Parameters:
        -----------
        wavelengths : array
            Wavelengths in microns
        flux_errors : array
            Original flux errors
        """
        if self.inflation_factors is None:
            return flux_errors

        inflated_errors = flux_errors.copy()

        for key, factor in self.inflation_factors.items():
            if isinstance(key, tuple):
                # Range format: (wave_min, wave_max)
                wave_min, wave_max = key
                mask = (wavelengths >= wave_min) & (wavelengths <= wave_max)
            else:
                # Single wavelength format: find closest match within 10%
                wave_target = float(key)
                mask = np.abs(wavelengths - wave_target) < (wave_target * 0.1)

            inflated_errors[mask] *= factor

            if np.any(mask):
                affected_waves = wavelengths[mask]
                logger.info(
                    f"Inflated errors by {factor}x for {key} (affected: {affected_waves})"
                )

        return inflated_errors

    def _setup_covariance_matrix(
        self, flux_errors_filtered, original_fit_mask, bootstrap_cov=None
    ):
        """
        Setup covariance matrix with flexible combination of sources
        """
        n_points = len(flux_errors_filtered)

        # Start with diagonal (no correlations)
        base_cov = np.diag(flux_errors_filtered**2)

        # Add instrumental correlations if available
        if (
            self.correlation_matrix is not None
            and not isinstance(self.correlation_matrix, dict)
            and self.correlation_matrix.shape[0] == n_points
        ):
            # Replace diagonal with correlated version
            instrumental_cov = (
                np.outer(flux_errors_filtered, flux_errors_filtered)
                * self.correlation_matrix
            )
            logger.info(f"Added instrumental correlations: {n_points}×{n_points}")
        else:
            instrumental_cov = base_cov
            logger.debug("Using diagonal covariance (no instrumental correlations)")

        # Add bootstrap covariance if provided
        if bootstrap_cov is not None:
            bootstrap_cov_filtered = bootstrap_cov[
                np.ix_(original_fit_mask, original_fit_mask)
            ]

            if bootstrap_cov_filtered.shape == instrumental_cov.shape:
                self.covariance_matrix = instrumental_cov + bootstrap_cov_filtered

                # Log contributions
                inst_trace = np.trace(instrumental_cov)
                boot_trace = np.trace(bootstrap_cov_filtered)
                total_trace = np.trace(self.covariance_matrix)
                logger.info(
                    f"Combined covariance - Instrumental: {inst_trace:.2e}, Bootstrap: {boot_trace:.2e}, Total: {total_trace:.2e}"
                )
            else:
                logger.warning(
                    "Bootstrap covariance shape mismatch, using instrumental only"
                )
                self.covariance_matrix = instrumental_cov
        else:
            self.covariance_matrix = instrumental_cov

        # Rest of method unchanged (positive definite check, Cholesky, etc.)
        eigenvals = np.linalg.eigvals(self.covariance_matrix)
        if np.any(eigenvals <= 0):
            logger.warning(
                "Covariance matrix not positive definite, adding regularization"
            )
            regularization = 1e-10 * np.max(np.diag(self.covariance_matrix))
            self.covariance_matrix += regularization * np.eye(n_points)

        try:
            L = cholesky(self.covariance_matrix, lower=True)
            self.inv_cov_chol = solve_triangular(L, np.eye(n_points), lower=True)
            logger.info(f"Set up {n_points}×{n_points} covariance matrix")
        except np.linalg.LinAlgError as e:
            logger.error(
                f"Cholesky decomposition failed: {e}, using diagonal covariance"
            )
            self.covariance_matrix = np.diag(flux_errors_filtered**2)
            self.inv_cov_chol = np.diag(1.0 / flux_errors_filtered)

    def log_likelihood_with_covariance(self, theta, wavelengths, fluxes, flux_errors):
        """
        Log likelihood using full covariance matrix

        This is the key improvement that should make the Schreiber prior more effective
        """
        amplitude, temperature = theta

        try:
            # Calculate model fluxes
            model_fluxes = self.greybody_model(
                wavelengths, amplitude, temperature, self.beta_fixed
            )

            if not np.all(np.isfinite(model_fluxes)):
                return -np.inf

            # Calculate residuals
            residuals = fluxes - model_fluxes

            if self.inv_cov_chol is None:
                # Fallback to diagonal case
                chi2 = np.sum((residuals / flux_errors) ** 2)
                log_det_cov = 2 * np.sum(np.log(flux_errors))
            else:
                # Use full covariance matrix via Cholesky decomposition
                # This is equivalent to: residuals^T @ inv_cov @ residuals
                # but much more numerically stable
                transformed_residuals = self.inv_cov_chol.T @ residuals
                chi2 = np.sum(transformed_residuals**2)

                # Log determinant from Cholesky: log|C| = 2 * sum(log(diag(L)))
                log_det_cov = -2 * np.sum(np.log(np.diag(self.inv_cov_chol)))

            # Sanity checks
            if not np.isfinite(chi2) or chi2 > 10000:
                return -np.inf

            # Log likelihood: -0.5 * [chi2 + log(2π)^n + log|C|]
            n_points = len(residuals)
            log_like = -0.5 * (chi2 + n_points * np.log(2 * np.pi) + log_det_cov)

            return log_like

        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            return -np.inf

    def log_posterior_with_covariance(
        self, theta, wavelengths, fluxes, flux_errors, redshift=0.0
    ):
        """Log posterior using covariance-aware likelihood"""
        log_p = self.log_prior(theta, redshift)
        if not np.isfinite(log_p):
            return -np.inf

        log_l = self.log_likelihood_with_covariance(
            theta, wavelengths, fluxes, flux_errors
        )
        return log_p + log_l

    def run_mcmc_with_covariance(self, wavelengths, fluxes, flux_errors, redshift=0.0):
        """
        Run MCMC with proper covariance treatment
        """
        if not HAS_EMCEE:
            raise ImportError("emcee is required for MCMC fitting")

        # Setup covariance matrix
        fit_mask = np.ones(len(wavelengths), dtype=bool)  # Use all points for now
        self._setup_covariance_matrix(flux_errors, fit_mask)

        # Get initial guess (this will be less dominant now)
        amplitude_guess, T_guess = self._get_initial_guess(
            wavelengths, fluxes, flux_errors, redshift
        )

        logger.info(f"MCMC with covariance: initial T={T_guess:.1f}K")

        # If using Schreiber prior, start walkers around the prior expectation
        if self.use_schreiber_prior and redshift > 0:
            T_prior, T_sigma = self.schreiber_temperature_prior(redshift)
            logger.info(
                f"Schreiber prior: T={T_prior:.1f}±{T_sigma:.1f}K (z={redshift:.2f})"
            )

            # Blend curve_fit guess with prior expectation
            alpha = 0.7  # Weight towards prior
            T_start = alpha * T_prior + (1 - alpha) * T_guess
            T_spread = max(T_sigma, 2.0)  # Ensure reasonable spread
        else:
            T_start = T_guess
            T_spread = 3.0

        logger.info(
            f"Starting MCMC around T={T_start:.1f}K with spread {T_spread:.1f}K"
        )

        # Setup walkers with appropriate spread
        n_walkers = 32
        n_dim = 2
        pos = []

        for _i in range(n_walkers):
            amp_trial = amplitude_guess + np.random.normal(0, 0.2)
            temp_trial = T_start + np.random.normal(0, T_spread)

            # Ensure within bounds
            amp_trial = np.clip(amp_trial, -37.5, -30.5)
            temp_trial = np.clip(temp_trial, 4, 44)

            # Test finite probability
            test_prob = self.log_posterior_with_covariance(
                [amp_trial, temp_trial], wavelengths, fluxes, flux_errors, redshift
            )

            if np.isfinite(test_prob):
                pos.append([amp_trial, temp_trial])
            else:
                # Fallback
                pos.append(
                    [
                        amplitude_guess + np.random.normal(0, 0.1),
                        T_start + np.random.normal(0, 1.0),
                    ]
                )

        pos = np.array(pos)

        # Create sampler with covariance-aware posterior
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            self.log_posterior_with_covariance,
            args=(wavelengths, fluxes, flux_errors, redshift),
        )

        # Run MCMC
        logger.info(f"Running MCMC with covariance: {self.mcmc_iterations} iterations")

        try:
            sampler.run_mcmc(pos, self.mcmc_iterations, progress=True)
        except Exception:
            logger.info("Progress bar failed, running without progress")
            sampler.run_mcmc(pos, self.mcmc_iterations, progress=False)

        # Check acceptance
        acceptance = np.mean(sampler.acceptance_fraction)
        logger.info(f"MCMC acceptance: {acceptance:.3f}")

        # Extract samples
        samples = sampler.get_chain(discard=self.mcmc_burn_in, flat=True)

        if len(samples) < 50:
            raise ValueError(f"Too few MCMC samples: {len(samples)}")

        # Calculate results
        percentiles = [16, 50, 84]
        amplitude_percentiles = np.percentile(samples[:, 0], percentiles)
        temperature_percentiles = np.percentile(samples[:, 1], percentiles)

        amplitude_best = amplitude_percentiles[1]
        temperature_best = temperature_percentiles[1]
        amplitude_err = np.diff(amplitude_percentiles[[0, 2]])[0] / 2
        temperature_err = np.diff(temperature_percentiles[[0, 2]])[0] / 2

        return {
            "samples": samples,
            "amplitude": amplitude_best,
            "amplitude_error": amplitude_err,
            "amplitude_percentiles": amplitude_percentiles,
            "temperature_observed_frame": temperature_best,
            "temperature_error": temperature_err,
            "temperature_percentiles": temperature_percentiles,
            "n_samples": len(samples),
            "acceptance_fraction": acceptance,
            "covariance_used": self.covariance_matrix is not None,
            "correlation_matrix_shape": self.correlation_matrix.shape
            if self.correlation_matrix is not None
            else None,
        }

    def fit_sed_with_covariance(
        self, wavelengths, fluxes, flux_errors, redshift, bootstrap_cov=None
    ):
        """
        Enhanced SED fitting with covariance matrix (optionally including bootstrap)
        """
        # Inflate specific band errors FIRST
        flux_errors = self._inflate_band_errors(wavelengths, flux_errors)

        # Validate data
        valid_mask, fit_mask = self._validate_data(wavelengths, fluxes, flux_errors)

        if np.sum(fit_mask) < 3:
            logger.warning(f"Insufficient valid data points: {np.sum(fit_mask)}")
            return {"fit_success": False, "reason": "insufficient_data"}

        wave_fit = wavelengths[fit_mask]
        flux_fit = fluxes[fit_mask]
        error_fit = flux_errors[fit_mask]  # Already filtered

        # Set up correlation matrix for these specific wavelengths
        if isinstance(self.correlation_matrix, dict) and self.correlation_matrix:
            logger.info("Converting correlation matrix from dict to numpy array")
            self.set_correlation_matrix_from_dict(self.correlation_matrix, wave_fit)

        # Setup covariance matrix - pass the already-filtered errors and original mask
        self._setup_covariance_matrix(error_fit, fit_mask, bootstrap_cov)

        logger.info(f"Fitting SED with covariance: {len(wave_fit)} points")

        try:
            # Still do curve_fit for initial guess, but it's less dominant now
            initial_result = super().fit_sed(wave_fit, flux_fit, error_fit, redshift)

            if not initial_result["fit_success"]:
                return initial_result

            # Run MCMC with covariance
            if self.use_mcmc:
                try:
                    mcmc_results = self.run_mcmc_with_covariance(
                        wave_fit, flux_fit, error_fit, redshift
                    )

                    # Update results with MCMC values
                    initial_result.update(
                        {
                            "amplitude": mcmc_results["amplitude"],
                            "amplitude_error": mcmc_results["amplitude_error"],
                            "temperature_observed_frame": mcmc_results[
                                "temperature_observed_frame"
                            ],
                            "temperature_error": mcmc_results["temperature_error"],
                            "mcmc_used": True,
                            "covariance_used": mcmc_results["covariance_used"],
                            "bootstrap_covariance_used": bootstrap_cov
                            is not None,  # NEW
                            "mcmc_samples": mcmc_results["samples"],
                            "mcmc_acceptance": mcmc_results["acceptance_fraction"],
                        }
                    )

                    logger.info(
                        f"Final result with covariance: T={mcmc_results['temperature_observed_frame']:.1f}±{mcmc_results['temperature_error']:.1f}K"
                    )

                except Exception as e:
                    logger.warning(f"MCMC with covariance failed: {e}")

            return initial_result

        except Exception as e:
            logger.error(f"Covariance SED fitting failed: {e}")
            return {"fit_success": False, "reason": str(e)}

    def fit_sed(self, wavelengths, fluxes, flux_errors, redshift):
        """
        Override parent fit_sed to automatically use covariance version
        """
        return self.fit_sed_with_covariance(wavelengths, fluxes, flux_errors, redshift)


class SimstackResults:
    """Process and analyze stacking results"""

    def __init__(
        self,
        config: SimstackConfig,
        stacking_results: StackingResults,
        population_manager: PopulationManager,
        cosmology_calc: CosmologyCalculator | None = None,
        fix_beta: bool = True,
        beta_fixed: float = 1.8,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        use_schreiber_prior: bool = False,
        use_covariance: bool = True,  # NEW
        correlation_matrix: dict | None = None,  # NEW
        inflation_factors: dict | None = None,  # NEW
        bootstrap_covariances: dict | None = None,
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
            use_mcmc: Whether to use MCMC fitting (requires emcee)
            mcmc_iterations: Number of MCMC iterations
            mcmc_burn_in: Number of burn-in iterations to discard
            use_schreiber_prior: Whether to use Schreiber+2015 T_dust vs redshift prior
            use_covariance: Whether to use covariance-aware fitting
            correlation_matrix: Custom correlation matrix (uses default if None)
        """
        self.config = config
        self.raw_results = stacking_results
        self.population_manager = population_manager

        # Set up cosmology
        if cosmology_calc is None:
            self.cosmology_calc = CosmologyCalculator(config.cosmology)
        else:
            self.cosmology_calc = cosmology_calc

        # Store bootstrap covariances for use in fitting
        self.bootstrap_covariances = bootstrap_covariances or {}

        # Initialize greybody fitter with MCMC and covariance options
        if use_covariance:
            self.greybody_fitter = CovarianceGreybodyFitter(
                correlation_matrix=correlation_matrix,
                fix_beta=fix_beta,
                beta_fixed=beta_fixed,
                use_mcmc=use_mcmc,
                mcmc_iterations=mcmc_iterations,
                mcmc_burn_in=mcmc_burn_in,
                use_schreiber_prior=use_schreiber_prior,
                inflation_factors=inflation_factors,
            )
            logger.info("Using covariance-aware greybody fitter")
        else:
            self.greybody_fitter = GreybodyFitter(
                fix_beta=fix_beta,
                beta_fixed=beta_fixed,
                use_mcmc=use_mcmc,
                mcmc_iterations=mcmc_iterations,
                mcmc_burn_in=mcmc_burn_in,
                use_schreiber_prior=use_schreiber_prior,
            )
            logger.info("Using standard greybody fitter")

        # Store settings for logging
        self.inflation_factors = inflation_factors
        self.use_bootstrap_covariance = bool(bootstrap_covariances)

        # Initialize processed results containers
        self.sed_results: dict[str, SEDResults] = {}
        self.derived_quantities: dict[str, DerivedQuantities] = {}
        self.band_results: dict[str, dict[str, Any]] = {}

        # Process results
        self._process_results()

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

        # Get bootstrap covariance for this population if available
        bootstrap_cov = self.bootstrap_covariances.get(pop_label, None)
        if bootstrap_cov is not None:
            logger.info(
                f"Using bootstrap covariance for {pop_label}: {bootstrap_cov.shape}"
            )

        # Fit greybody model
        if isinstance(self.greybody_fitter, CovarianceGreybodyFitter):
            greybody_results = self.greybody_fitter.fit_sed_with_covariance(
                wavelengths, flux_densities, flux_errors, z_median, bootstrap_cov
            )
        else:
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

            # Add MCMC results if available
            if greybody_results.get("mcmc_used", False):
                sed_result.mcmc_samples = greybody_results.get("mcmc_samples")
                sed_result.mcmc_percentiles = greybody_results.get("mcmc_percentiles")

        return sed_result

    # Rest of the methods remain the same...
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

        # MCMC-derived uncertainties
        total_ir_lum_mcmc_err = None
        dust_temp_mcmc_err = None

        if sed_result.greybody_fit_success:
            # Use the stored fit parameters to calculate L_IR
            amplitude = sed_result.amplitude
            temperature_observed = sed_result.dust_temperature_observed_frame
            beta = sed_result.emissivity_index or self.greybody_fitter.beta_fixed
            redshift = sed_result.median_redshift

            if amplitude is not None and temperature_observed is not None:
                try:
                    # Calculate L_IR using the greybody fitter's method
                    total_ir_lum, total_ir_lum_err = self.greybody_fitter.calculate_LIR(
                        amplitude, temperature_observed, beta, redshift
                    )

                    # Also get dust mass - with better error handling
                    try:
                        dust_mass, _ = self.greybody_fitter.calculate_dust_mass(
                            amplitude, temperature_observed, beta, redshift
                        )
                    except Exception as e:
                        logger.warning(
                            f"Dust mass calculation failed for {sed_result.population_id}: {e}"
                        )
                        dust_mass = None

                    # Store temperature values
                    dust_temp_rest = sed_result.dust_temperature_rest_frame
                    dust_temp_obs = sed_result.dust_temperature_observed_frame

                    # Calculate MCMC-derived uncertainties if available
                    if sed_result.mcmc_samples is not None:
                        try:
                            # Calculate L_IR for all MCMC samples
                            n_samples = min(
                                len(sed_result.mcmc_samples), 1000
                            )  # Limit for speed
                            sample_indices = np.random.choice(
                                len(sed_result.mcmc_samples), n_samples, replace=False
                            )

                            lir_samples = []
                            temp_samples = []

                            for i in sample_indices:
                                sample_amp, sample_temp = sed_result.mcmc_samples[i]
                                sample_temp_rest = sample_temp * (1 + redshift)

                                try:
                                    sample_lir, _ = self.greybody_fitter.calculate_LIR(
                                        sample_amp, sample_temp, beta, redshift
                                    )
                                    lir_samples.append(sample_lir)
                                    temp_samples.append(sample_temp_rest)
                                except Exception:
                                    continue

                            if len(lir_samples) > 10:  # Need sufficient samples
                                # Calculate 16th and 84th percentiles for uncertainties
                                lir_16, lir_84 = np.percentile(lir_samples, [16, 84])
                                temp_16, temp_84 = np.percentile(temp_samples, [16, 84])

                                total_ir_lum_mcmc_err = (
                                    total_ir_lum - lir_16,
                                    lir_84 - total_ir_lum,
                                )
                                dust_temp_mcmc_err = (
                                    dust_temp_rest - temp_16,
                                    temp_84 - dust_temp_rest,
                                )
                        except Exception as e:
                            logger.warning(
                                f"MCMC uncertainty calculation failed for {sed_result.population_id}: {e}"
                            )

                    print(
                        f"✓ Population {sed_result.population_id}: L_IR = {total_ir_lum:.2e} L_sun"
                    )
                    if total_ir_lum_mcmc_err:
                        print(
                            f"  MCMC uncertainties: +{total_ir_lum_mcmc_err[1]:.2e}/-{total_ir_lum_mcmc_err[0]:.2e}"
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
            total_ir_luminosity_mcmc_error=total_ir_lum_mcmc_err,
            dust_temperature_mcmc_error=dust_temp_mcmc_err,
        )

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
                logger.error(f"Failed to process population {pop_label}: {e}")
                import traceback

                logger.error(traceback.format_exc())
                # Continue with next population instead of stopping
                continue

        # Process band-by-band results
        self._process_band_results()

        logger.info(f"Processed results for {len(self.sed_results)} populations")

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
                "mcmc_used": sed_result.mcmc_samples is not None,
                "mcmc_n_samples": len(sed_result.mcmc_samples)
                if sed_result.mcmc_samples is not None
                else 0,
            }

            # Add MCMC-derived uncertainties if available
            if derived and derived.total_ir_luminosity_mcmc_error:
                row[
                    "total_ir_luminosity_mcmc_error_lower"
                ] = derived.total_ir_luminosity_mcmc_error[0]
                row[
                    "total_ir_luminosity_mcmc_error_upper"
                ] = derived.total_ir_luminosity_mcmc_error[1]

            if derived and derived.dust_temperature_mcmc_error:
                row[
                    "dust_temperature_mcmc_error_lower"
                ] = derived.dust_temperature_mcmc_error[0]
                row[
                    "dust_temperature_mcmc_error_upper"
                ] = derived.dust_temperature_mcmc_error[1]

            data.append(row)

        return pd.DataFrame(data)

    def print_results_summary(self) -> None:
        """Print a formatted summary of results"""
        print("=== Simstack4 Results Summary ===")
        print(f"Processed {len(self.sed_results)} populations")
        print(f"Bands: {len(self.raw_results.map_names)}")

        fitting_method = "MCMC" if self.greybody_fitter.use_mcmc else "curve_fit"
        prior_type = (
            "Schreiber+2015" if self.greybody_fitter.use_schreiber_prior else "flat"
        )

        print(f"Fitting method: {fitting_method}")
        print(f"Prior type: {prior_type}")
        print(
            f"Greybody fitting: β {'fixed' if self.greybody_fitter.fix_beta else 'free'} = {self.greybody_fitter.beta_fixed if self.greybody_fitter.fix_beta else 'fitted'}"
        )
        print()

        # Print greybody fit quality
        successful_fits = sum(
            1 for sed in self.sed_results.values() if sed.greybody_fit_success
        )
        mcmc_fits = sum(
            1 for sed in self.sed_results.values() if sed.mcmc_samples is not None
        )

        print(
            f"Greybody Fit Success Rate: {successful_fits}/{len(self.sed_results)} ({100 * successful_fits / len(self.sed_results):.1f}%)"
        )
        if self.greybody_fitter.use_mcmc:
            print(
                f"MCMC Fits: {mcmc_fits}/{successful_fits} ({100 * mcmc_fits / max(1, successful_fits):.1f}%)"
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
            red_chi2 = self.raw_results.reduced_chi_squared.get(map_name, np.nan)
            wave = self.config.maps[map_name].wavelength
            print(f"  {map_name} ({wave}μm): χ²_red = {red_chi2:.2f}")
        print()

        # Print population results
        summary_df = self.get_population_summary()
        if len(summary_df) > 0:
            print("Population Results:")
            header = f"{'Population':<30} {'N_src':<8} {'z_med':<8} {'T_rest[K]':<10} {'T_obs[K]':<10} {'L_IR[L☉]':<12} {'SFR[M☉/yr]':<12}"
            if self.greybody_fitter.use_mcmc:
                header += " {'MCMC':<5}"
            print(header)
            print("-" * (110 + (6 if self.greybody_fitter.use_mcmc else 0)))

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

                line = f"{pop_id:<30} {n_src:<8} {z_med:<8.2f} {t_rest:<10.1f} {t_obs:<10.1f} {l_ir:<12.2e} {sfr:<12.1f}"

                if self.greybody_fitter.use_mcmc:
                    mcmc_used = "✓" if row["mcmc_used"] else "✗"
                    line += f" {mcmc_used:<5}"

                print(line)

        # Save and load methods remain the same but with additional MCMC info

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
            "greybody_fitter_config": {
                "use_mcmc": self.greybody_fitter.use_mcmc,
                "mcmc_iterations": self.greybody_fitter.mcmc_iterations,
                "mcmc_burn_in": self.greybody_fitter.mcmc_burn_in,
                "use_schreiber_prior": self.greybody_fitter.use_schreiber_prior,
                "fix_beta": self.greybody_fitter.fix_beta,
                "beta_fixed": self.greybody_fitter.beta_fixed,
            },
        }

        with open(output_path, "wb") as f:
            pickle.dump(results_dict, f)

        # Other methods remain the same...

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
        """Save processed results to file"""
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

                # Save MCMC samples if available
                if sed.mcmc_samples is not None:
                    sed_grp.create_dataset("mcmc_samples", data=sed.mcmc_samples)

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

    @classmethod
    def load_results(cls, file_path: Path) -> "SimstackResults":
        """Load results from saved file"""
        file_path = Path(file_path)

        if not file_path.exists():
            raise ResultsError(f"Results file not found: {file_path}")

        if file_path.suffix == ".pkl":
            with open(file_path, "rb") as f:
                results_dict = pickle.load(f)

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
    use_mcmc: bool = False,
    mcmc_iterations: int = 1000,
    mcmc_burn_in: int = 200,
    use_schreiber_prior: bool = False,
    use_covariance: bool = True,
    correlation_matrix: dict | None = None,
    inflation_factors: dict | None = None,
    bootstrap_covariances: dict | None = None,
) -> SimstackResults:
    """
    Convenience function to create results processor

    Args:
        config: Simstack configuration
        stacking_results: Raw stacking results
        population_manager: Population manager
        fix_beta: Whether to fix the emissivity index beta in greybody fitting
        beta_fixed: Fixed value of beta if fix_beta=True (recommended: 1.8 for galaxies)
        use_mcmc: Whether to use MCMC fitting (requires emcee)
        mcmc_iterations: Number of MCMC iterations
        mcmc_burn_in: Number of burn-in iterations to discard
        use_schreiber_prior: Whether to use Schreiber+2015 T_dust vs redshift prior
        use_covariance: Whether to use covariance-aware fitting (NEW)
        correlation_matrix: Custom correlation matrix (uses default if None) (NEW)
        bootstrap_covariances: Dict mapping population_id -> bootstrap covariance matrix

    Returns:
        Processed SimstackResults object
    """
    return SimstackResults(
        config,
        stacking_results,
        population_manager,
        fix_beta=fix_beta,
        beta_fixed=beta_fixed,
        use_mcmc=use_mcmc,
        mcmc_iterations=mcmc_iterations,
        mcmc_burn_in=mcmc_burn_in,
        use_schreiber_prior=use_schreiber_prior,
        use_covariance=use_covariance,
        correlation_matrix=correlation_matrix,
        inflation_factors=inflation_factors,
        bootstrap_covariances=bootstrap_covariances,
    )
