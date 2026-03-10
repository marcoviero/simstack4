"""
Greybody (modified blackbody) dust emission model for Simstack4.

Contains the core SED model, Planck function, temperature priors,
L_IR calculation, and base fitting (curve_fit + MCMC).

Classes
-------
Greybody : Core greybody model and fitter.
SEDResults : Dataclass for per-population SED fit results.
DerivedQuantities : Dataclass for derived physical quantities.
"""
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import curve_fit

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

from .cosmology import CosmologyCalculator
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

    # Fit quality metadata
    fit_quality_tier: str | None = None  # "A" (data), "B" (assisted), "C" (prior-dominated)
    sed_snr: float | None = None  # Median SNR across positive bands
    prior_center: float | None = None  # T_rest prior center used (K)
    prior_sigma: float | None = None  # T_rest prior sigma used (K)

    # Per-bin median catalog properties (from bin_property_columns config)
    bin_properties: dict[str, float] | None = None

    # Regression greybody fit results (from global polynomial fit)
    regression_T_rest: float | None = None  # K
    regression_log10_A: float | None = None
    regression_L_IR: float | None = None  # L_sun
    regression_chi2: float | None = None  # per-population χ²


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


class Greybody:
    """Improved Greybody fitter with robust MCMC handling"""

    def __init__(
        self,
        fix_beta: bool = True,
        beta_fixed: float = 1.8,
        use_mcmc: bool = False,
        mcmc_iterations: int = 1000,
        mcmc_burn_in: int = 200,
        temperature_prior: str = "flat",  # "flat", "schreiber", "viero"
        cosmology_calc: "CosmologyCalculator | None" = None,
        # --- Temperature bounds (rest frame, K) ---
        T_rest_min: float = 15.0,
        T_rest_max: float = 90.0,
        # --- Amplitude bounds (log10) ---
        amplitude_min: float = -41.0,
        amplitude_max: float = -29.0,
        # --- Beta bounds (free-beta mode) ---
        beta_min: float = 0.5,
        beta_max: float = 2.5,
        # --- SNR thresholds for fit quality tiers ---
        snr_high: float = 5.0,   # Tier A: data-driven
        snr_low: float = 1.0,    # Tier C: prior-dominated
        # --- Prior sigma scaling ---
        snr_sigma_clip_min: float = 0.3,  # Floor for σ_eff / σ_base
        snr_sigma_clip_max: float = 2.0,  # Ceiling for σ_eff / σ_base
        # --- Error inflation ---
        inflation_factors: dict | None = None,
    ):
        """
        Initialize fitter.

        Temperature bounds
        ------------------
        T_rest_min, T_rest_max : Rest-frame dust temperature bounds in K.
            Default [15, 80] K. Raise T_rest_max for high-z populations
            where T_dust can exceed 60 K.

        SNR thresholds
        --------------
        snr_high : Median SED SNR above which fits are tier A (data-driven).
        snr_low  : Below this, fits are tier C (prior-dominated).
            Between snr_low and snr_high → tier B (prior-assisted).

        Prior sigma scaling
        -------------------
        When temperature_prior != "flat", the effective prior σ is:
            σ_eff = σ_prior × clip(SNR / snr_high, snr_sigma_clip_min, snr_sigma_clip_max)
        So at SNR = snr_high: σ_eff = σ_prior (prior barely matters).
        At SNR = 1: σ_eff ≈ 0.3 × σ_prior (prior dominates).
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
        self.temperature_prior = temperature_prior.lower() if isinstance(temperature_prior, str) else "flat"

        # Configurable bounds
        self.T_rest_min = T_rest_min
        self.T_rest_max = T_rest_max
        self.amplitude_min = amplitude_min
        self.amplitude_max = amplitude_max
        self.beta_min = beta_min
        self.beta_max = beta_max

        # SNR thresholds
        self.SNR_HIGH = snr_high
        self.SNR_LOW = snr_low
        self.snr_sigma_clip_min = snr_sigma_clip_min
        self.snr_sigma_clip_max = snr_sigma_clip_max

        # Error inflation
        self.inflation_factors = inflation_factors

        # Cosmology — use proper calculator if provided, else fall back to Planck18
        if cosmology_calc is not None:
            self._cosmology_calc = cosmology_calc
        else:
            try:
                self._cosmology_calc = CosmologyCalculator()
            except Exception:
                self._cosmology_calc = None
                logger.warning(
                    "CosmologyCalculator unavailable, "
                    "luminosity_distance will use Hubble-law fallback"
                )

        if use_mcmc and not HAS_EMCEE:
            logger.warning(
                "MCMC requested but emcee not available. Falling back to curve_fit."
            )

        self._prior_override = None  # Set by fit_sed during two-pass fitting

    @staticmethod
    def compute_sed_snr(fluxes, flux_errors):
        """
        Median SNR across positive-flux bands.

        Returns 0 if no positive detections.
        """
        positive = fluxes > 0
        if not np.any(positive):
            return 0.0
        snr_per_band = fluxes[positive] / flux_errors[positive]
        return float(np.median(snr_per_band))

    def _validate_data(self, wavelengths, fluxes, flux_errors):
        """
        Validate and filter data for SED fitting.

        Handles the large dynamic range typical of multi-band IR photometry
        (e.g., MIPS 24μm through SCUBA 850μm).

        Returns
        -------
        valid_mask : array
            Boolean mask for valid data points.
        use_for_fit : array
            Boolean mask for data to use in fitting (more restrictive).
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

        # Strong positive detections (>2σ)
        strong_detections = (fluxes > 0) & (snr > 2.0)

        # Reasonable measurements (>1σ, cap very high SNR outliers)
        reasonable_measurements = (snr > 1.0) & (snr < 1000)

        # Exclude bands with extremely large errors (>100× median error),
        # which typically contribute no constraining power
        log_errors = np.log10(flux_errors)
        median_log_error = np.median(log_errors)
        reasonable_errors = (log_errors - median_log_error) < 2.0

        # Final: reasonable errors AND (strong detection OR reasonable measurement)
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
        Get initial guess for rest-frame fitting.

        wavelengths are assumed to already be in rest frame.
        Temperature guess and bounds are in rest frame.
        """
        try:
            if self.temperature_prior != "flat" and redshift > 0:
                T_guess, _ = self.temperature_prior_relation(redshift)
                T_guess = np.clip(T_guess, self.T_rest_min + 1, self.T_rest_max - 2)
            else:
                T_guess = 30.0  # typical rest-frame T
            amplitude_guess = -35.0

            # Quick curve_fit for better initial parameters (rest-frame bounds)
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
                        bounds=(
                            [self.amplitude_min, self.T_rest_min],
                            [self.amplitude_max, self.T_rest_max],
                        ),
                        maxfev=1000,
                    )
                    amplitude_guess, T_guess = popt
                    logger.debug(
                        f"Curve_fit initial guess: A={amplitude_guess:.2f}, T_rest={T_guess:.1f}K"
                    )
                except Exception as e:
                    logger.debug(
                        f"Curve_fit for initial guess failed: {e}, using defaults"
                    )

            return amplitude_guess, T_guess

        except Exception as e:
            logger.warning(f"Initial guess estimation failed: {e}")
            return -35.0, 30.0

    def temperature_prior_relation(self, redshift: float) -> tuple[float, float]:
        """
        Calculate expected REST-FRAME dust temperature from chosen relation.

        Parameters
        ----------
        redshift : float

        Returns
        -------
        T_rest, sigma_T : tuple[float, float]
            Expected temperature and intrinsic scatter (both in K).
        """
        if redshift <= 0:
            return 25.0, 5.0

        prior = self.temperature_prior

        if prior == "viero":
            # Viero et al. 2022 (MNRAS Letters 516, L30):
            # Quadratic T-z from stacking COSMOS2020 + Herschel/SCUBA-2.
            # Predicts ~105K at z=8.4; "near exponential" evolution.
            T_rest = 23.8 + 2.7 * redshift + 0.9 * redshift**2
            # Scatter: ~3K at low-z, growing with redshift
            T_sigma = 3.0 + 1.0 * min(redshift, 4.0)

        elif prior == "schreiber":
            # Schreiber et al. 2018 (A&A 609, A30):
            # Linear T-z from stacking CANDELS + Herschel + ALMA.
            # T_d = 32.9 + 4.60 * (z - 2), valid z ~ 0-4.
            T_rest = 32.9 + 4.60 * (redshift - 2.0)
            # Intrinsic scatter ~12% (Schreiber+2018 Section 4.3)
            T_sigma = max(0.12 * T_rest, 3.0)

        elif prior == "flat":
            # No preferred temperature — return midpoint with wide sigma
            T_rest = (self.T_rest_min + self.T_rest_max) / 2.0
            T_sigma = (self.T_rest_max - self.T_rest_min) / 4.0
            return T_rest, T_sigma

        else:
            raise ValueError(
                f"Unknown temperature_prior '{prior}'. "
                f"Choose from: 'flat', 'schreiber', 'viero'"
            )

        T_rest = np.clip(T_rest, self.T_rest_min, self.T_rest_max)
        return T_rest, T_sigma

    def log_prior(self, theta: list, redshift: float = 0.0) -> float:
        """
        Calculate log prior with rest-frame temperature bounds.

        If self._prior_override is set (by fit_sed during two-pass fitting),
        uses that (center, sigma) instead of Schreiber.
        """
        amplitude, temperature = theta  # temperature is T_rest

        # Amplitude bounds (slightly inset from curve_fit bounds for MCMC stability)
        if not (self.amplitude_min + 3 < amplitude < self.amplitude_max + 1):
            return -np.inf

        # Rest-frame temperature bounds — stable across all redshifts
        if not (self.T_rest_min < temperature < self.T_rest_max):
            return -np.inf

        # Temperature priors
        if self._prior_override is not None:
            T_center, T_sigma = self._prior_override
            log_p_temp = -0.5 * ((temperature - T_center) / T_sigma) ** 2
        elif self.temperature_prior != "flat" and redshift > 0:
            T_expected, T_sigma = self.temperature_prior_relation(redshift)
            log_p_temp = -0.5 * ((temperature - T_expected) / T_sigma) ** 2
        else:
            # Mild preference for typical dust temperatures
            T_mid = (self.T_rest_min + self.T_rest_max) / 2
            if 20 <= temperature <= 45:
                log_p_temp = 0.0
            elif self.T_rest_min <= temperature < 20 or 45 < temperature <= self.T_rest_max:
                log_p_temp = -0.5 * ((temperature - T_mid) / 15) ** 2
            else:
                log_p_temp = -np.inf

        return log_p_temp

    def log_likelihood(self, theta, wavelengths, fluxes, flux_errors):
        """
        Gaussian log-likelihood for SED fitting.

        Handles negative observed fluxes (common in stacking).
        """
        amplitude, temperature = theta

        try:
            model_fluxes = self.greybody_model(
                wavelengths, amplitude, temperature, self.beta_fixed
            )

            if not np.all(np.isfinite(model_fluxes)) or np.any(model_fluxes <= 0):
                return -np.inf

            residuals = (fluxes - model_fluxes) / flux_errors
            chi2 = np.sum(residuals**2)

            if not np.isfinite(chi2) or chi2 > 10000:
                return -np.inf

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
        Initialize MCMC walkers near the curve_fit solution (rest-frame T).
        """
        amplitude_guess, temperature_guess = initial_guess
        pos = []

        for _i in range(n_walkers):
            amp_trial = amplitude_guess + np.random.normal(0, 0.1)
            temp_trial = temperature_guess + np.random.normal(0, 1.0)

            # Rest-frame bounds (inset from hard bounds for MCMC stability)
            amp_trial = np.clip(
                amp_trial, self.amplitude_min + 3.5, self.amplitude_max + 1.5
            )
            temp_trial = np.clip(
                temp_trial, self.T_rest_min + 1, self.T_rest_max - 2
            )

            test_prob = self.log_posterior(
                [amp_trial, temp_trial], wavelengths, fluxes, flux_errors, redshift
            )

            if np.isfinite(test_prob):
                pos.append([amp_trial, temp_trial])
            else:
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

        # Run MCMC
        try:
            effective_iterations = min(self.mcmc_iterations, 500)
            effective_burn_in = min(self.mcmc_burn_in, effective_iterations // 4)

            logger.info(
                f"Running MCMC: {effective_iterations} iterations, "
                f"{effective_burn_in} burn-in, starting near T_rest={T_guess:.1f}K"
            )

            try:
                sampler.run_mcmc(pos, effective_iterations, progress=True)
                total_steps = effective_iterations
            except Exception as progress_error:
                logger.info(
                    f"Progress bar failed ({progress_error}), running without"
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
            "temperature_rest_frame": temperature_best,
            "temperature_error": temperature_err,
            "temperature_percentiles": temperature_percentiles,
            "n_samples": len(samples),
            "acceptance_fraction": final_acceptance,
            "total_steps": total_steps,
            "effective_iterations": effective_iterations,
            "effective_burn_in": effective_burn_in,
        }

    def _inflate_band_errors(self, wavelengths, flux_errors):
        """
        Inflate uncertainties for specific bands.

        Parameters
        ----------
        wavelengths : array
            Wavelengths in microns.
        flux_errors : array
            Original flux errors.

        Returns
        -------
        Inflated copy (or original if no inflation configured).
        """
        if self.inflation_factors is None:
            return flux_errors

        inflated_errors = flux_errors.copy()

        for key, factor in self.inflation_factors.items():
            if isinstance(key, tuple):
                wave_min, wave_max = key
                mask = (wavelengths >= wave_min) & (wavelengths <= wave_max)
            else:
                wave_target = float(key)
                mask = np.abs(wavelengths - wave_target) < (wave_target * 0.1)

            inflated_errors[mask] *= factor

            if np.any(mask):
                affected_waves = wavelengths[mask]
                logger.info(
                    f"Inflated errors by {factor}x for {key} "
                    f"(affected: {affected_waves})"
                )

        return inflated_errors

    def fit_sed(self, wavelengths, fluxes, flux_errors, redshift,
                prior_override=None):
        """
        Fit greybody model in the rest frame.

        Wavelengths are transformed to rest frame before fitting so that
        the temperature parameter is T_rest directly, with bounds
        [T_rest_min, T_rest_max] K at all redshifts.

        Parameters
        ----------
        prior_override : tuple(float, float) or None
            (T_center, T_sigma) to override Schreiber prior.
            Used by two-pass empirical Bayes fitting.
        """
        # Apply error inflation before anything else
        flux_errors = self._inflate_band_errors(wavelengths, flux_errors)

        # Validate and filter data (in observed frame)
        valid_mask, fit_mask = self._validate_data(wavelengths, fluxes, flux_errors)

        if np.sum(fit_mask) < 3:
            logger.warning(f"Insufficient valid data points: {np.sum(fit_mask)}")
            return {"fit_success": False, "reason": "insufficient_data"}

        wave_obs_fit = wavelengths[fit_mask]
        flux_fit = fluxes[fit_mask]
        error_fit = flux_errors[fit_mask]

        # Compute SED signal-to-noise
        sed_snr = self.compute_sed_snr(flux_fit, error_fit)

        # Determine prior and T bounds based on SNR
        # High SNR → wide bounds, weak/no prior (data-driven)
        # Low SNR  → narrow bounds around prior center, strong prior
        if prior_override is not None:
            T_center, T_sigma = prior_override
            self._prior_override = prior_override
        elif self.temperature_prior != "flat" and redshift > 0:
            T_center, T_sigma_base = self.temperature_prior_relation(redshift)
            # Scale sigma with SNR: σ_eff = σ_base × (SNR_HIGH / max(SNR, 1))
            # At SNR=5: σ=σ_base (prior barely matters)
            # At SNR=2: σ=σ_base×2.5 ... wait, we want TIGHTER at low SNR
            # Invert: σ_eff = σ_base × max(SNR, 1) / SNR_HIGH
            # At SNR=5: σ=σ_base (normal)
            # At SNR=2: σ=0.4×σ_base (tight)
            # At SNR=1: σ=0.2×σ_base (very tight)
            snr_ratio = max(sed_snr, 0.5) / self.SNR_HIGH
            T_sigma = T_sigma_base * np.clip(
                snr_ratio, self.snr_sigma_clip_min, self.snr_sigma_clip_max
            )
            self._prior_override = (T_center, T_sigma)
        else:
            T_center = None
            T_sigma = None
            self._prior_override = None

        # Set T bounds: narrow around prior for low SNR
        T_lo, T_hi = self.T_rest_min, self.T_rest_max
        if T_center is not None and sed_snr < self.SNR_HIGH:
            # Narrow bounds: ±3σ around prior center, but at least ±5K
            half_width = max(3 * T_sigma, 5.0)
            T_lo = max(self.T_rest_min, T_center - half_width)
            T_hi = min(self.T_rest_max, T_center + half_width)

        # Assign quality tier
        if sed_snr >= self.SNR_HIGH:
            fit_quality_tier = "A"  # data-driven
        elif sed_snr >= self.SNR_LOW:
            fit_quality_tier = "B"  # prior-assisted
        else:
            fit_quality_tier = "C"  # prior-dominated

        # Transform to rest frame
        z1 = max(1 + redshift, 1.001)  # guard against z=0
        wave_rest_fit = wave_obs_fit / z1

        logger.info(
            f"Fitting SED (rest frame): {len(wave_rest_fit)} points, "
            f"{np.sum(flux_fit > 0)} positive detections, z={redshift:.2f}, "
            f"SNR={sed_snr:.1f}, tier={fit_quality_tier}"
        )

        try:
            # ============================================================
            # Helper: solve amplitude analytically at fixed T, beta
            # Returns (amplitude, amplitude_err)
            # ============================================================
            def _solve_amplitude_at_T(T_rest, beta_val):
                """
                Linear least-squares for 10^A at fixed T, beta.
                flux_i = 10^A × template_i  →  x = Σ(f·t/σ²) / Σ(t²/σ²)

                Uses only positive-flux bands for the solve.  Negative
                fluxes in stacked data indicate noise-dominated bands
                where the template shape has no constraining power; they
                can make x_opt negative, producing a fallback amplitude
                that is *not* physically negligible.
                """
                # Filter to positive detections — matching the approach
                # that works in the Schreiber overlay plot.
                pos = flux_fit > 0
                if np.sum(pos) < 1:
                    return self.amplitude_min, 1.0

                template = self.greybody_model(
                    wave_rest_fit[pos], 0.0, T_rest, beta_val
                )
                w = 1.0 / error_fit[pos]**2
                denom = np.sum(template**2 * w)
                if denom <= 0:
                    return self.amplitude_min, 1.0
                x_opt = np.sum(flux_fit[pos] * template * w) / denom

                if x_opt > 0:
                    amp = np.clip(
                        np.log10(x_opt), self.amplitude_min, self.amplitude_max
                    )
                    amp_err = np.sqrt(1.0 / denom) / (x_opt * np.log(10))
                else:
                    # All positive-flux bands still give negative x_opt:
                    # truly no detection.
                    amp = self.amplitude_min
                    amp_err = 1.0
                return amp, amp_err

            # ============================================================
            # TIER C: Prior-dominated (SNR < snr_low)
            # Fix T = T_center, solve amplitude analytically.
            # ============================================================
            if fit_quality_tier == "C" and T_center is not None and self.fix_beta:
                temperature_rest = T_center
                beta = self.beta_fixed
                amplitude, amplitude_err = _solve_amplitude_at_T(
                    temperature_rest, beta
                )
                temperature_err = T_sigma  # Error is the prior width
                beta_err = 0.0

                logger.info(
                    f"Tier-C analytical fit: T_rest={temperature_rest:.1f}K (fixed), "
                    f"A={amplitude:.2f}±{amplitude_err:.2f}"
                )

            elif fit_quality_tier == "B" and T_center is not None and self.fix_beta:
                # ============================================================
                # TIER B: Prior-assisted (snr_low ≤ SNR < snr_high)
                # Two-step: regularized curve_fit for T, then analytical A.
                # This breaks the T–A degeneracy that causes amplitude
                # inflation on the Wien side at high z.
                # ============================================================
                amplitude_guess, T_rest_guess = self._get_initial_guess(
                    wave_rest_fit, flux_fit, error_fit, redshift
                )
                if T_center is not None:
                    T_rest_guess = T_center

                logger.debug(
                    f"Tier-B two-step: T_guess={T_rest_guess:.1f}K, "
                    f"bounds=[{T_lo:.0f}, {T_hi:.0f}]K"
                )

                # Step 1: fit T only (amplitude pinned to a reasonable value
                # from the initial guess, regularized toward T_center)
                pos_mask = flux_fit > 0
                flux_pos = flux_fit[pos_mask] if np.any(pos_mask) else flux_fit
                error_pos = error_fit[pos_mask] if np.any(pos_mask) else error_fit

                def model_func_T_only(wave, temp):
                    """1-parameter model: solve A analytically at each T."""
                    template_full = self.greybody_model(
                        wave, 0.0, temp, self.beta_fixed
                    )
                    template_pos = template_full[pos_mask] if np.any(pos_mask) else template_full
                    w = 1.0 / error_pos**2
                    x = np.sum(flux_pos * template_pos * w) / np.sum(template_pos**2 * w)
                    x = max(x, 1e-40)
                    return template_full * x

                try:
                    popt_T, pcov_T = curve_fit(
                        model_func_T_only,
                        wave_rest_fit,
                        flux_fit,
                        sigma=error_fit,
                        p0=[T_rest_guess],
                        bounds=([T_lo], [T_hi]),
                        maxfev=5000,
                    )
                    temperature_rest = popt_T[0]
                    temperature_err = np.sqrt(pcov_T[0, 0])
                except Exception:
                    # Fallback: use prior center
                    temperature_rest = T_center
                    temperature_err = T_sigma

                # Step 2: analytical amplitude at fitted T
                beta = self.beta_fixed
                amplitude, amplitude_err = _solve_amplitude_at_T(
                    temperature_rest, beta
                )
                beta_err = 0.0

                logger.info(
                    f"Tier-B two-step: T_rest={temperature_rest:.1f}±{temperature_err:.1f}K, "
                    f"A={amplitude:.2f}±{amplitude_err:.2f}"
                )

            else:
                # ============================================================
                # TIER A (or no prior): Standard curve_fit
                # ============================================================
                amplitude_guess, T_rest_guess = self._get_initial_guess(
                    wave_rest_fit, flux_fit, error_fit, redshift
                )
                if T_center is not None:
                    T_rest_guess = T_center

                logger.debug(
                    f"Initial guess: A={amplitude_guess:.2f}, T_rest={T_rest_guess:.1f}K, "
                    f"bounds=[{T_lo:.0f}, {T_hi:.0f}]K"
                )

                if self.fix_beta:

                    def model_func(wave, amp, temp):
                        return self.greybody_model(wave, amp, temp, self.beta_fixed)

                    popt, pcov = curve_fit(
                        model_func,
                        wave_rest_fit,
                        flux_fit,
                        sigma=error_fit,
                        p0=[amplitude_guess, T_rest_guess],
                        bounds=(
                            [self.amplitude_min, T_lo],
                            [self.amplitude_max, T_hi],
                        ),
                        maxfev=5000,
                    )
                    amplitude, temperature_rest = popt
                    beta = self.beta_fixed
                    param_errors = np.sqrt(np.diag(pcov))
                    amplitude_err, temperature_err = param_errors
                    beta_err = 0.0
                else:
                    popt, pcov = curve_fit(
                        self.greybody_model,
                        wave_rest_fit,
                        flux_fit,
                        sigma=error_fit,
                        p0=[amplitude_guess, T_rest_guess, 1.8],
                        bounds=(
                            [self.amplitude_min, T_lo, self.beta_min],
                            [self.amplitude_max, T_hi, self.beta_max],
                        ),
                        maxfev=5000,
                    )
                    amplitude, temperature_rest, beta = popt
                    param_errors = np.sqrt(np.diag(pcov))
                    amplitude_err, temperature_err, beta_err = param_errors

            logger.info(
                f"Curve fit: T_rest={temperature_rest:.1f}±{temperature_err:.1f}K"
            )

            # Run MCMC if requested (also in rest frame)
            mcmc_results = None
            if self.use_mcmc and self.fix_beta:
                try:
                    mcmc_results = self.run_mcmc(
                        wave_rest_fit, flux_fit, error_fit, redshift
                    )
                    amplitude = mcmc_results["amplitude"]
                    temperature_rest = mcmc_results["temperature_rest_frame"]
                    amplitude_err = mcmc_results["amplitude_error"]
                    temperature_err = mcmc_results["temperature_error"]

                    logger.info(
                        f"MCMC fit: T_rest={temperature_rest:.1f}±{temperature_err:.1f}K"
                    )

                except Exception as e:
                    logger.warning(f"MCMC failed, using curve_fit results: {e}")
                    mcmc_results = None

            # Derive observed-frame temperature
            temperature_observed = temperature_rest / z1

            # Goodness of fit (evaluate at rest-frame wavelengths)
            model_fluxes = self.greybody_model(
                wave_rest_fit, amplitude, temperature_rest, beta
            )
            chi2 = np.sum(((flux_fit - model_fluxes) / error_fit) ** 2)
            dof = len(wave_rest_fit) - (2 if self.fix_beta else 3)
            chi2_reduced = chi2 / max(1, dof)

            # Calculate L_IR (pass rest-frame parameters)
            L_IR, L_IR_error = self.calculate_LIR(
                amplitude, temperature_rest, beta, redshift
            )

            # Model curve for plotting (rest-frame wavelengths)
            wave_model_rest = np.logspace(
                np.log10(np.min(wave_rest_fit) * 0.5),
                np.log10(np.max(wave_rest_fit) * 2),
                200,
            )
            flux_model = self.greybody_model(
                wave_model_rest, amplitude, temperature_rest, beta
            )

            results = {
                "fit_success": True,
                "amplitude": amplitude,
                "amplitude_error": amplitude_err,
                "temperature_rest_frame": temperature_rest,
                "temperature_observed_frame": temperature_observed,
                "temperature_error": temperature_err,
                "beta": beta,
                "beta_error": beta_err,
                "chi2_reduced": chi2_reduced,
                "L_IR": L_IR,
                "L_IR_error": L_IR_error,
                "n_points": len(wave_rest_fit),
                "n_positive": np.sum(flux_fit > 0),
                "wavelengths_fit": wave_rest_fit,
                "fluxes_fit": flux_fit,
                "flux_errors_fit": error_fit,
                "model_wavelengths": wave_model_rest,
                "model_fluxes": flux_model,
                "redshift_used": redshift,
                "mcmc_used": mcmc_results is not None,
                "temperature_prior": self.temperature_prior,
                "sed_snr": sed_snr,
                "fit_quality_tier": fit_quality_tier,
                "prior_center": T_center,
                "prior_sigma": T_sigma if T_center is not None else None,
            }

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
            return {
                "fit_success": False, "reason": str(e),
                "sed_snr": sed_snr, "fit_quality_tier": fit_quality_tier,
            }
        finally:
            self._prior_override = None

    def greybody_model(
        self, wavelength_um, amplitude, temperature, beta=1.8, alpha=2.0
    ):
        """
        Modified blackbody (greybody) model with Wien-side power-law extension.

        At long wavelengths: ν^β × B_ν(T) (modified blackbody).
        At short wavelengths (ν > ν_cut): ν^(-α) power law, matched
        continuously at ν_cut = (3 + β + α) × k_B T / h.

        Parameters
        ----------
        wavelength_um : array
            Wavelengths in microns (rest-frame if fitting in rest frame).
        amplitude : float
            log10 amplitude.
        temperature : float
            Dust temperature (K).
        beta : float
            Emissivity index.
        alpha : float
            Wien-side power-law slope.
        """
        nu_in = self.c * 1.0e6 / wavelength_um  # Hz
        A = 10**amplitude

        # Transition frequency: Wien peak of modified blackbody
        nu_cut = (3.0 + beta + alpha) * self.k_B / self.h * temperature

        # Power-law normalization: matched to greybody at ν_cut
        # Constants use h in units of 1e-34 J·s, k_B in 1e-23 J/K, c in 1e8 m/s
        # to avoid overflow in intermediate products.
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

        # Modified blackbody (Rayleigh-Jeans side)
        graybody = A * nu_in**beta * self.black(nu_in, temperature)[0] / 1000.0

        # Power law (Wien side)
        powerlaw = w_div * nu_in ** (-alpha)

        # Apply transition
        flux_density = graybody.copy()
        ind_cut = nu_in >= nu_cut
        flux_density[ind_cut] = powerlaw[ind_cut]

        return flux_density

    def black(self, nu_in, T):
        """
        Planck function B_ν(T) in units that give mJy when multiplied by ν^β.

        Constants:
            a0 = 2h/c² × 1e29 = 1.4718e-21  (converts to ~mJy scale)
            a1 = h/k_B = 4.7993e-11 s·K  (exponent scale)
        """
        a0 = 1.4718e-21  # 2h × 10^29 / c²
        a1 = 4.7993e-11  # h / k_B

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
        Calculate total infrared luminosity from rest-frame greybody parameters.

        L_IR = 4π D_L² / (1+z) × ∫(8–1000 μm rest) S_ν(ν_rest) dν_rest

        The 1/(1+z) factor accounts for the bandwidth compression when
        converting from observed-frame flux density to rest-frame luminosity.

        Parameters
        ----------
        amplitude, temperature, beta : float
            Rest-frame greybody parameters.
        redshift : float
            Source redshift.

        Returns
        -------
        L_IR : float
            Total IR luminosity in L_sun.
        L_IR_error : float
            Propagated uncertainty in L_sun.
        """
        if np.isnan(redshift) or redshift <= 0:
            redshift = 0.01

        D_L = self.luminosity_distance(redshift)  # Mpc
        L_IR = self._integrate_LIR(amplitude, temperature, beta, redshift, D_L)

        # Error propagation via finite differences
        if amplitude_err is not None and temperature_err is not None:
            delta_amp = max(0.001, abs(amplitude_err) * 0.1)
            delta_temp = max(0.1, abs(temperature_err) * 0.1)

            try:
                L_plus_a = self._integrate_LIR(
                    amplitude + delta_amp, temperature, beta, redshift, D_L
                )
                L_minus_a = self._integrate_LIR(
                    amplitude - delta_amp, temperature, beta, redshift, D_L
                )
                dL_damp = (L_plus_a - L_minus_a) / (2 * delta_amp)

                L_plus_t = self._integrate_LIR(
                    amplitude, temperature + delta_temp, beta, redshift, D_L
                )
                L_minus_t = self._integrate_LIR(
                    amplitude, temperature - delta_temp, beta, redshift, D_L
                )
                dL_dtemp = (L_plus_t - L_minus_t) / (2 * delta_temp)

                L_IR_error = np.sqrt(
                    (dL_damp * amplitude_err) ** 2
                    + (dL_dtemp * temperature_err) ** 2
                )
            except Exception as e:
                logger.warning(f"L_IR error propagation failed: {e}, using 15%")
                L_IR_error = L_IR * 0.15
        else:
            L_IR_error = L_IR * 0.15

        return L_IR, L_IR_error

    def _integrate_LIR(self, amplitude, temperature, beta, redshift, D_L_mpc):
        """
        Core L_IR integration (rest-frame 8–1000 μm).

        Parameters
        ----------
        D_L_mpc : float
            Luminosity distance in Mpc.

        Returns
        -------
        L_IR in solar luminosities.
        """
        wavelength_rest = np.logspace(np.log10(8.0), np.log10(1000.0), 1000)
        model_sed_jy = self.greybody_model(
            wavelength_rest, amplitude, temperature, beta
        )

        # Convert rest-frame wavelengths to frequencies (decreasing order)
        nu_rest = self.c * 1.0e6 / wavelength_rest  # Hz

        # Integrate S_ν dν using trapezoidal rule (negate because ν decreasing)
        integral_jy_hz = -np.trapezoid(model_sed_jy, nu_rest)

        # L_IR = 4π D_L² / (1+z) × ∫ S_ν dν
        # Jy → W/m²/Hz: × 1e-26
        # D_L: Mpc → m: × 3.08568025e22
        D_L_m = D_L_mpc * 3.08568025e22
        L_IR_watts = 4.0 * np.pi * D_L_m**2 * 1e-26 * integral_jy_hz / (1 + redshift)

        return L_IR_watts / self.L_sun

    def luminosity_distance(self, z):
        """Calculate luminosity distance in Mpc using proper cosmology."""
        if z <= 0:
            z = 0.01
        if self._cosmology_calc is not None:
            return self._cosmology_calc.luminosity_distance(z)
        else:
            # Fallback: Hubble-law approximation (only used if astropy unavailable)
            logger.warning("Using Hubble-law D_L fallback — inaccurate at z > 0.3")
            c_km_s = 299792.458  # km/s
            H0 = 67.7  # Planck18
            D_L = c_km_s * z / H0  # Mpc
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




# Backwards-compatible alias
GreybodyFitter = Greybody
