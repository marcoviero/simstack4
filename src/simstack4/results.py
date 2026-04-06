"""
Results processing and analysis orchestration for Simstack4.

Coordinates SED fitting across populations, manages I/O,
and calculates derived physical quantities.

Classes
-------
SimstackResults : Main results processor and orchestrator.
"""
import pdb
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

from .algorithm import StackingResults
from .config import SimstackConfig
from .cosmology import CosmologyCalculator
from .exceptions.simstack_exceptions import ResultsError
from .greybody import Greybody, GreybodyFitter, SEDResults, DerivedQuantities
from .sed_fitting import CovarianceGreybodyFitter, RegressionGreybodyFitter
from .populations import PopulationManager
from .utils import setup_logging

logger = setup_logging()

# Re-export for backwards compatibility — these classes moved to
# greybody.py and sed_fitting.py but are still importable from here.
__all__ = [
    "SimstackResults",
    "create_results_processor",
    "SEDResults",
    "DerivedQuantities",
    "Greybody",
    "GreybodyFitter",
    "CovarianceGreybodyFitter",
    "RegressionGreybodyFitter",
]


def _ensure_dict(obj):
    """Convert bin_properties value to dict if it was stringified during serialization."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        import ast
        try:
            parsed = ast.literal_eval(obj)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return None



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
        temperature_prior: str = "flat",  # "flat", "schreiber", "viero"
        use_covariance: bool = True,  # NEW
        correlation_matrix: dict | None = None,  # NEW
        inflation_factors: dict | None = None,  # NEW
        bootstrap_covariances: dict | None = None,
        exclude_wavelengths: list[float] | None = None,  # NEW
        # --- Regression greybody configuration ---
        use_regression: bool = False,
        use_regression_prior: bool = False,
        regression_degree: str = 'linear',
        regression_property_names: list | None = None,
        regression_min_sources: int = 5,
        # --- Greybody fitter configuration (forwarded to GreybodyFitter) ---
        T_rest_min: float = 15.0,
        T_rest_max: float = 140.0,
        amplitude_min: float = -41.0,
        amplitude_max: float = -29.0,
        beta_min: float = 0.5,
        beta_max: float = 2.5,
        snr_high: float = 5.0,
        snr_low: float = 2.0,
        use_pah: bool = True,
        wien_mode: str = "physical",
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
            temperature_prior: Temperature prior ("flat", "schreiber", "viero")
            use_covariance: Whether to use covariance-aware fitting
            correlation_matrix: Custom correlation matrix (uses default if None)
            exclude_wavelengths: Wavelengths (µm) to exclude from SED fitting.
                e.g. [24.0] to skip MIPS 24µm (PAH emission, not dust continuum).
                Excluded bands still appear in SED data for plotting.
            T_rest_min: Minimum rest-frame T_dust bound (K)
            T_rest_max: Maximum rest-frame T_dust bound (K). Raise for high-z.
            snr_high: SED SNR threshold for tier-A (data-driven) fits
            snr_low: SED SNR threshold below which fits are tier-C (prior-dominated)
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

        # Greybody fitter kwargs shared between standard and covariance fitters
        fitter_kwargs = dict(
            fix_beta=fix_beta,
            beta_fixed=beta_fixed,
            use_mcmc=use_mcmc,
            mcmc_iterations=mcmc_iterations,
            mcmc_burn_in=mcmc_burn_in,
            temperature_prior=temperature_prior,
            cosmology_calc=self.cosmology_calc,
            T_rest_min=T_rest_min,
            T_rest_max=T_rest_max,
            amplitude_min=amplitude_min,
            amplitude_max=amplitude_max,
            beta_min=beta_min,
            beta_max=beta_max,
            snr_high=snr_high,
            snr_low=snr_low,
            inflation_factors=inflation_factors,
        )

        # Initialize greybody fitter with MCMC and covariance options
        if use_covariance:
            self.greybody_fitter = CovarianceGreybodyFitter(
                correlation_matrix=correlation_matrix,
                **fitter_kwargs,
            )
            logger.info("Using covariance-aware greybody fitter")
        else:
            self.greybody_fitter = GreybodyFitter(**fitter_kwargs)
            logger.info("Using standard greybody fitter")

        # Store settings for logging
        self.inflation_factors = inflation_factors
        self.use_bootstrap_covariance = bool(bootstrap_covariances)
        self.exclude_wavelengths = exclude_wavelengths or []

        # Initialize processed results containers
        self.sed_results: dict[str, SEDResults] = {}
        self.derived_quantities: dict[str, DerivedQuantities] = {}
        self.band_results: dict[str, dict[str, Any]] = {}
        self.regression_result: dict[str, dict] | None = None  # type → result

        # Store regression configuration
        self.use_regression = use_regression or use_regression_prior
        self.use_regression_prior = use_regression_prior
        self.regression_degree = regression_degree
        self.regression_property_names = regression_property_names  # None = auto-detect
        self.regression_min_sources = regression_min_sources

        self.use_pah = use_pah
        #self.wien_mode = wien_mode

        # Process results
        self._process_results()

    def _create_sed_for_population(
        self, pop_label: str, pop_index: int, prior_override=None
    ) -> SEDResults:
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
            err_bootstrap = self.raw_results.flux_errors[map_name][pop_index]

            # Combine bootstrap and formal (systematic) errors in quadrature.
            # Per-bin bootstrap only captures source-assignment variance;
            # formal errors capture map noise.  Both are needed.
            # When bootstrap is NOT run, both arrays are identical —
            # skip combination to avoid double-counting.
            err_formal = 0.0
            if hasattr(self.raw_results, 'flux_errors_systematic'):
                sys_errors = self.raw_results.flux_errors_systematic.get(
                    map_name, None
                )
                if sys_errors is not None:
                    err_formal = sys_errors[pop_index]

            if err_formal > 0 and abs(err_bootstrap - err_formal) > 1e-30:
                # Bootstrap was run — errors are independent components
                error = np.sqrt(err_bootstrap**2 + err_formal**2)
            else:
                # No bootstrap, or both are identical — use as-is
                error = err_bootstrap

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

        # Exclude specified wavelengths from fitting (but keep in SED data)
        fit_mask = np.ones(len(wavelengths), dtype=bool)
        if self.exclude_wavelengths:
            for exc_wave in self.exclude_wavelengths:
                # Match within 10% tolerance
                band_match = np.abs(wavelengths - exc_wave) < (exc_wave * 0.1)
                fit_mask &= ~band_match
                if np.any(band_match):
                    logger.info(
                        f"Excluding {wavelengths[band_match]} µm from fit "
                        f"for {pop_label}"
                    )

        wave_fit = wavelengths[fit_mask]
        flux_fit = flux_densities[fit_mask]
        error_fit = flux_errors[fit_mask]
        bootstrap_cov_fit = None
        if bootstrap_cov is not None:
            bootstrap_cov_fit = bootstrap_cov[np.ix_(fit_mask, fit_mask)]

        # Set PAH state for this population (z and M* from catalog)
        self.greybody_fitter._pah_z = z_median
        self.greybody_fitter._pah_log_stellar_mass = pop_bin.median_stellar_mass
        self.greybody_fitter.use_pah = self.use_pah  # ← uncomment when ready
        #self.greybody_fitter.wien_mode = self.wien_mode

        # Fit greybody model
        if isinstance(self.greybody_fitter, CovarianceGreybodyFitter):
            greybody_results = self.greybody_fitter.fit_sed_with_covariance(
                wave_fit, flux_fit, error_fit, z_median, bootstrap_cov_fit,
                prior_override=prior_override,
            )
        else:
            greybody_results = self.greybody_fitter.fit_sed(
                wave_fit, flux_fit, error_fit, z_median,
                prior_override=prior_override,
            )

        # Create SED result
        # Use n_sources from StackingResults (definitely populated by algorithm)
        # with fallback to population manager
        n_sources = self.raw_results.n_sources.get(
            pop_label, pop_bin.n_sources
        )

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
            n_sources=n_sources,
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

        # Quality metadata (stored even for failed fits)
        sed_result.fit_quality_tier = greybody_results.get("fit_quality_tier")
        sed_result.sed_snr = greybody_results.get("sed_snr")
        sed_result.prior_center = greybody_results.get("prior_center")
        sed_result.prior_sigma = greybody_results.get("prior_sigma")

        # Per-bin median catalog properties
        if isinstance(self.raw_results.bin_properties, dict):
            props = self.raw_results.bin_properties.get(pop_label)
            props = _ensure_dict(props)
            if props:
                sed_result.bin_properties = props

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

        # MCMC-derived uncertainties
        total_ir_lum_mcmc_err = None
        dust_temp_mcmc_err = None

        if sed_result.greybody_fit_success:
            # Use the stored fit parameters to calculate L_IR
            amplitude = sed_result.amplitude
            temperature_rest = sed_result.dust_temperature_rest_frame
            beta = sed_result.emissivity_index or self.greybody_fitter.beta_fixed
            redshift = sed_result.median_redshift

            if amplitude is not None and temperature_rest is not None:
                try:
                    # calculate_LIR now expects T_rest
                    total_ir_lum, total_ir_lum_err = self.greybody_fitter.calculate_LIR(
                        amplitude, temperature_rest, beta, redshift
                    )

                    # calculate_dust_mass also expects T_rest
                    try:
                        dust_mass, _ = self.greybody_fitter.calculate_dust_mass(
                            amplitude, temperature_rest, beta, redshift
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
                                sample_amp, sample_temp_rest = sed_result.mcmc_samples[i]

                                try:
                                    sample_lir, _ = self.greybody_fitter.calculate_LIR(
                                        sample_amp, sample_temp_rest, beta, redshift
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

                    logger.info(
                        f"Population {sed_result.population_id}: L_IR = {total_ir_lum:.2e} L_sun"
                    )
                    if total_ir_lum_mcmc_err:
                        logger.info(
                            f"  MCMC uncertainties: +{total_ir_lum_mcmc_err[1]:.2e}/-{total_ir_lum_mcmc_err[0]:.2e}"
                        )

                except Exception as e:
                    logger.warning(
                        f"L_IR calculation failed for {sed_result.population_id}: {e}"
                    )
                    total_ir_lum = 0.0
                    total_ir_lum_err = 0.0
            else:
                logger.warning(f"Missing fit parameters for {sed_result.population_id}")
        else:
            logger.warning(f"No successful greybody fit for {sed_result.population_id}")

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
        """
        Process raw results into SEDs and derived quantities.

        Multi-pass approach:
          Pass 1: Fit all SEDs with SNR-scaled Schreiber prior.
          Pass 2: Compute empirical T(z) from high-confidence fits,
                  re-fit low-SNR SEDs with empirical prior center.
          Pass 3 (optional): Regression greybody fit — polynomial T(M*,z,β)
                  and A(M*,z,β) across all populations simultaneously.
          Pass 4 (optional): Refit per-population SEDs using regression
                  surface as prior. High-SNR populations free to depart;
                  low-SNR pulled toward "main sequence".
        """
        logger.info("Processing stacking results...")

        pop_labels = [
            lbl for lbl in self.raw_results.population_labels
            if lbl != "foreground"
        ]
        pop_indices = {
            lbl: i for i, lbl in enumerate(self.raw_results.population_labels)
            if lbl != "foreground"
        }

        # === Pass 1: Fit all with SNR-scaled Schreiber prior ===
        logger.info("Pass 1: fitting all SEDs with Schreiber prior...")
        pass1_results = {}
        for pop_label in pop_labels:
            try:
                sed_result = self._create_sed_for_population(
                    pop_label, pop_indices[pop_label]
                )
                pass1_results[pop_label] = sed_result
            except Exception as e:
                logger.error(f"Pass 1 failed for {pop_label}: {e}")
                continue

        # === Compute empirical T(z) from high-confidence fits ===
        empirical_prior = self._compute_empirical_t_of_z(pass1_results)

        # === Pass 2: Re-fit low-SNR SEDs with empirical prior ===
        n_refit = 0
        empirical_z_values = np.array(list(empirical_prior.keys())) if empirical_prior else np.array([])

        for pop_label, sed_result in pass1_results.items():
            tier = sed_result.fit_quality_tier
            if tier not in ("B", "C"):
                # Tier A: data-driven, keep pass 1 result
                self.sed_results[pop_label] = sed_result
                continue

            z = sed_result.median_redshift

            # Find nearest empirical anchor (within Δz < 0.5)
            emp = None
            if len(empirical_z_values) > 0:
                nearest_idx = np.argmin(np.abs(empirical_z_values - z))
                nearest_z = empirical_z_values[nearest_idx]
                if abs(nearest_z - z) < 0.5:
                    emp = empirical_prior[nearest_z]

            if emp is None:
                # No empirical anchor for this z — keep pass 1
                self.sed_results[pop_label] = sed_result
                continue

            T_emp, T_emp_sigma = emp
            n_refit += 1
            logger.info(
                f"Pass 2 refit: {pop_label} (tier {tier}, SNR={sed_result.sed_snr:.1f}) "
                f"with empirical T={T_emp:.1f}±{T_emp_sigma:.1f}K"
            )

            try:
                sed_result_v2 = self._create_sed_for_population(
                    pop_label, pop_indices[pop_label],
                    prior_override=(T_emp, T_emp_sigma),
                )
                self.sed_results[pop_label] = sed_result_v2
            except Exception as e:
                logger.error(f"Pass 2 failed for {pop_label}: {e}, keeping pass 1")
                self.sed_results[pop_label] = sed_result

        logger.info(
            f"Pass 1: {len(pass1_results)} fits, "
            f"Pass 2: {n_refit} refits with empirical prior"
        )

        # Calculate derived quantities for all
        for pop_label, sed_result in self.sed_results.items():
            try:
                derived = self._calculate_derived_quantities(sed_result)
                self.derived_quantities[pop_label] = derived
            except Exception as e:
                logger.error(f"Failed to derive quantities for {pop_label}: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Process band-by-band results
        self._process_band_results()

        # === Pass 3 (optional): Regression greybody fit ===
        if self.use_regression:
            self._run_regression_fit()

        # === Pass 4 (optional): Refit with regression as prior ===
        if self.use_regression_prior and self.regression_result is not None:
            self._refit_with_regression_prior(pop_indices)

        logger.info(f"Processed results for {len(self.sed_results)} populations")

    def _compute_empirical_t_of_z(
        self, pass1_results: dict[str, SEDResults]
    ) -> dict[float, tuple[float, float]]:
        """
        Compute empirical T_rest(z) from high-confidence (tier A) fits.

        Groups by median_redshift, computes median T_rest and robust σ
        per redshift bin. Returns {z_median: (T_center, T_sigma)}.

        Only bins with ≥3 tier-A fits contribute an anchor.
        """
        from collections import defaultdict

        z_groups = defaultdict(list)
        for pop_label, sed_result in pass1_results.items():
            if (
                sed_result.fit_quality_tier == "A"
                and sed_result.greybody_fit_success
                and sed_result.dust_temperature_rest_frame is not None
            ):
                z = sed_result.median_redshift
                z_groups[z].append(sed_result.dust_temperature_rest_frame)

        empirical = {}
        for z, temps in z_groups.items():
            if len(temps) >= 3:
                T_median = float(np.median(temps))
                # Robust sigma: 1.4826 × MAD
                T_sigma = max(
                    1.4826 * float(np.median(np.abs(np.array(temps) - T_median))),
                    2.0,  # Floor: at least 2K
                )
                empirical[z] = (T_median, T_sigma)
                logger.info(
                    f"Empirical T(z={z:.2f}): {T_median:.1f}±{T_sigma:.1f}K "
                    f"from {len(temps)} tier-A fits"
                )

        if not empirical:
            logger.info("No empirical T(z) anchors (too few tier-A fits)")

        return empirical

    @staticmethod
    def _extract_pop_type(pop_id: str) -> str:
        """
        Extract population type from ID.  Handles both prefix and suffix
        styles that PopulationManager can produce:

        Prefix:  'sfg__stellar_mass_9.5_10.0__redshift_0.5_1.0'      → 'sfg'
        Suffix:  'redshift_0.01_0.5__stellar_mass_8.5_10.0__split_0'  → 'split_0'
        Neither: 'stellar_mass_9.5_10.0__redshift_0.5_1.0'            → '_all_'
        """
        segments = pop_id.split('__')

        # Check first segment (prefix style: sfg__...)
        first = segments[0]
        if not any(c.isdigit() for c in first):
            return first

        # Check last segment (suffix style: ...__split_0)
        if len(segments) > 1:
            last = segments[-1]
            parts = last.split('_')
            # A bin-range segment has ≥3 parts ending in two floats
            # e.g. "stellar_mass_8.5_10.0"  →  skip
            # A type-label segment does not, e.g. "split_0"
            if len(parts) >= 3:
                try:
                    float(parts[-2])
                    float(parts[-1])
                    return '_all_'
                except ValueError:
                    return last
            else:
                return last

        return '_all_'

    def _run_regression_fit(self) -> None:
        """
        Pass 3: Fit regression greybody model across all populations.

        Parameterizes T_rest and log10(A) as polynomial functions of
        population properties (M*, z, β_UV), fitting ~8 coefficients
        instead of ~900 independent parameters.

        Fits SEPARATE regression surfaces per population type (sfg,
        quiescent, agn, etc.) since different types have fundamentally
        different FIR properties.

        Results are stored alongside (not replacing) existing per-population fits.
        """
        logger.info("Pass 3: Regression greybody fit...")

        # Build ordered lists aligned with sed_results
        pop_ids = list(self.sed_results.keys())
        N_pop = len(pop_ids)
        if N_pop == 0:
            logger.warning("No populations to fit regression model")
            return

        # Auto-detect binning dimensions from PopulationManager
        available_dims = set()
        for pop_id in pop_ids:
            if pop_id in self.population_manager.populations:
                br = self.population_manager.populations[pop_id].bin_ranges
                available_dims.update(br.keys())
                break

        # Determine which property names to use
        if self.regression_property_names is not None:
            # User specified — validate against available dimensions
            prop_names = []
            for pname in self.regression_property_names:
                if pname == 'redshift' or pname in available_dims:
                    prop_names.append(pname)
                else:
                    logger.warning(
                        f"Regression property '{pname}' not in binning "
                        f"dimensions {available_dims}, skipping"
                    )
            if not prop_names:
                prop_names = ['redshift']  # always available
        else:
            # Auto-detect: use binning dimensions + redshift
            prop_names = []
            # Always include these if they're binning axes
            for candidate in ['stellar_mass', 'redshift', 'beta_uv']:
                if candidate in available_dims or candidate == 'redshift':
                    prop_names.append(candidate)

        logger.info(f"Regression properties: {prop_names} "
                    f"(binning dims: {available_dims})")

        # Gather wavelengths from first SED
        first_sed = self.sed_results[pop_ids[0]]
        wavelengths_obs = first_sed.wavelengths.copy()
        N_bands = len(wavelengths_obs)

        # Collect data and group by population type
        pop_types = {}  # type → list of (pop_idx, pop_id)
        fluxes = np.zeros((N_pop, N_bands))
        errors = np.zeros((N_pop, N_bands))
        redshifts = np.zeros(N_pop)
        n_sources = np.zeros(N_pop, dtype=int)
        prop_arrays = {pname: np.zeros(N_pop) for pname in prop_names}

        for i, pop_id in enumerate(pop_ids):
            sed = self.sed_results[pop_id]
            fluxes[i] = sed.flux_densities
            errors[i] = sed.flux_errors
            redshifts[i] = sed.median_redshift
            n_sources[i] = sed.n_sources

            # Group by type
            ptype = self._extract_pop_type(pop_id)
            pop_types.setdefault(ptype, []).append(i)

            # Get bin centers from PopulationManager
            if pop_id in self.population_manager.populations:
                pop_bin = self.population_manager.populations[pop_id]
                br = pop_bin.bin_ranges
                for pname in prop_names:
                    if pname == 'redshift':
                        if 'redshift' in br:
                            prop_arrays['redshift'][i] = (
                                br['redshift'][0] + br['redshift'][1]) / 2
                        else:
                            prop_arrays['redshift'][i] = sed.median_redshift
                    elif pname == 'stellar_mass':
                        if 'stellar_mass' in br:
                            prop_arrays['stellar_mass'][i] = (
                                br['stellar_mass'][0] + br['stellar_mass'][1]) / 2
                        else:
                            prop_arrays['stellar_mass'][i] = sed.median_mass
                    elif pname in br:
                        prop_arrays[pname][i] = (
                            br[pname][0] + br[pname][1]) / 2
            else:
                if 'redshift' in prop_arrays:
                    prop_arrays['redshift'][i] = sed.median_redshift
                if 'stellar_mass' in prop_arrays:
                    prop_arrays['stellar_mass'][i] = sed.median_mass

        logger.info(f"Population types found: {list(pop_types.keys())} "
                    f"({', '.join(f'{k}: {len(v)}' for k, v in pop_types.items())})")

        # Fit separate regression surface per type
        self.regression_result = {}  # type → result dict
        n_min_params = 2 * (len(prop_names) + 1)

        for ptype, type_indices in pop_types.items():
            type_indices = np.array(type_indices)

            # Filter to populations with enough sources
            source_mask = n_sources[type_indices] >= self.regression_min_sources
            type_indices_f = type_indices[source_mask]
            n_excluded = np.sum(~source_mask)

            if n_excluded > 0:
                logger.info(f"  [{ptype}] excluding {n_excluded} populations "
                            f"with < {self.regression_min_sources} sources")

            if len(type_indices_f) < n_min_params:
                logger.warning(
                    f"  [{ptype}] too few populations ({len(type_indices_f)}) "
                    f"for {n_min_params}-param regression, skipping")
                continue

            # Extract data for this type
            fluxes_t = fluxes[type_indices_f]
            errors_t = errors[type_indices_f]
            redshifts_t = redshifts[type_indices_f]
            props_t = {pname: arr[type_indices_f]
                       for pname, arr in prop_arrays.items()}

            # Build sed_results for initialization
            type_pop_ids = [pop_ids[j] for j in type_indices_f]
            type_seds = {pid: self.sed_results[pid] for pid in type_pop_ids}

            # Create and run the regression fitter
            reg_fitter = RegressionGreybodyFitter(
                greybody_fitter=self.greybody_fitter,
                property_names=prop_names,
                degree=self.regression_degree,
                min_sources=self.regression_min_sources,
            )

            try:
                reg_result = reg_fitter.fit(
                    wavelengths_obs, fluxes_t, errors_t, redshifts_t,
                    props_t, sed_results=type_seds,
                )
            except Exception as e:
                logger.error(f"  [{ptype}] regression fit failed: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue

            # Store per-type result
            self.regression_result[ptype] = reg_result

            # Write regression values back into SEDResults
            for fi, orig_i in enumerate(type_indices_f):
                pid = pop_ids[orig_i]
                sed = self.sed_results[pid]
                sed.regression_T_rest = float(reg_result['T_rest'][fi])
                sed.regression_log10_A = float(reg_result['log10_A'][fi])
                sed.regression_L_IR = float(reg_result['L_IR'][fi])
                sed.regression_chi2 = float(reg_result['chi2_per_pop'][fi])

            logger.info(
                f"  [{ptype}] χ²/dof={reg_result['chi2_reduced']:.2f}, "
                f"T={reg_result['T_rest'].min():.1f}–"
                f"{reg_result['T_rest'].max():.1f}K, "
                f"{len(type_indices_f)} populations"
            )

        if not self.regression_result:
            logger.warning("No regression fits succeeded")
            self.regression_result = None

    def _refit_with_regression_prior(self, pop_indices: dict) -> None:
        """
        Pass 4: Refit per-population SEDs using the regression surface as prior.

        The regression provides a smooth "main sequence" prediction for T_rest
        as a function of (M★, z, β). This is used as prior_center for refitting,
        with sigma derived from the scatter of high-confidence (tier A) individual
        fits around the regression surface, COMPUTED PER TYPE.

        The existing SNR-scaled prior mechanism then handles the strength:
            High-SNR (tier A): σ_eff ≈ σ_regression (prior barely matters,
                               population free to depart from MS)
            Low-SNR (tier B/C): σ_eff ≈ 0.3 × σ_regression (pulled toward MS)
        """
        reg = self.regression_result
        if reg is None:
            return

        # Compute regression prior sigma PER TYPE from tier-A scatter
        type_sigmas = {}
        for pop_id, sed in self.sed_results.items():
            if (sed.fit_quality_tier == "A"
                    and sed.regression_T_rest is not None
                    and sed.dust_temperature_rest_frame is not None
                    and np.isfinite(sed.dust_temperature_rest_frame)):
                ptype = self._extract_pop_type(pop_id)
                type_sigmas.setdefault(ptype, []).append(
                    sed.dust_temperature_rest_frame - sed.regression_T_rest
                )

        regression_sigmas = {}
        for ptype, residuals in type_sigmas.items():
            if len(residuals) < 3:
                logger.warning(f"  [{ptype}] too few tier-A fits "
                               f"({len(residuals)}) for regression sigma")
                continue
            residuals = np.array(residuals)
            sigma = 1.4826 * np.median(np.abs(residuals - np.median(residuals)))
            regression_sigmas[ptype] = max(sigma, 3.0)  # floor at 3K

        if not regression_sigmas:
            logger.warning("No types have enough tier-A fits for regression sigma")
            return

        for ptype, sigma in regression_sigmas.items():
            n_tier_a = len(type_sigmas[ptype])
            logger.info(
                f"Pass 4 [{ptype}]: σ_regression = {sigma:.1f}K "
                f"(from {n_tier_a} tier-A residuals)"
            )

        # Refit all populations that have a regression prediction
        n_refit = 0
        for pop_id, sed in list(self.sed_results.items()):
            if sed.regression_T_rest is None:
                continue

            ptype = self._extract_pop_type(pop_id)
            if ptype not in regression_sigmas:
                continue

            T_prior = sed.regression_T_rest
            sigma = regression_sigmas[ptype]
            pop_index = pop_indices.get(pop_id)
            if pop_index is None:
                continue

            try:
                sed_refit = self._create_sed_for_population(
                    pop_id, pop_index,
                    prior_override=(T_prior, sigma),
                )
                # Preserve regression fields from Pass 3
                sed_refit.regression_T_rest = sed.regression_T_rest
                sed_refit.regression_log10_A = sed.regression_log10_A
                sed_refit.regression_L_IR = sed.regression_L_IR
                sed_refit.regression_chi2 = sed.regression_chi2

                self.sed_results[pop_id] = sed_refit
                n_refit += 1

                # Recompute derived quantities
                derived = self._calculate_derived_quantities(sed_refit)
                self.derived_quantities[pop_id] = derived

            except Exception as e:
                logger.error(f"Pass 4 refit failed for {pop_id}: {e}")

        logger.info(
            f"Pass 4 complete: {n_refit} populations refitted with regression prior"
        )

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
                "fit_quality_tier": sed_result.fit_quality_tier,
                "sed_snr": sed_result.sed_snr,
                "prior_center_K": sed_result.prior_center,
                "prior_sigma_K": sed_result.prior_sigma,
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

            # Add per-bin median catalog properties
            props = _ensure_dict(sed_result.bin_properties)
            if props:
                for col_name, val in props.items():
                    row[f"median_{col_name}"] = val
            elif sed_result.bin_properties is not None:
                logger.debug(
                    f"Unexpected bin_properties type for {pop_id}: "
                    f"{type(sed_result.bin_properties).__name__} = "
                    f"{sed_result.bin_properties!r}"
                )

            # Add regression greybody results if available
            if sed_result.regression_T_rest is not None:
                row["regression_T_rest_K"] = sed_result.regression_T_rest
                row["regression_log10_A"] = sed_result.regression_log10_A
                row["regression_L_IR_lsun"] = sed_result.regression_L_IR
                row["regression_chi2"] = sed_result.regression_chi2

            data.append(row)

        return pd.DataFrame(data)

    def print_results_summary(self) -> None:
        """Print a formatted summary of results"""
        print("=== Simstack4 Results Summary ===")
        print(f"Processed {len(self.sed_results)} populations")
        print(f"Bands: {len(self.raw_results.map_names)}")

        fitting_method = "MCMC" if self.greybody_fitter.use_mcmc else "curve_fit"
        prior_type = (
            self.greybody_fitter.temperature_prior
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

        # Fit quality tier summary
        from collections import Counter
        tiers = Counter(
            sed.fit_quality_tier for sed in self.sed_results.values()
            if sed.fit_quality_tier is not None
        )
        if tiers:
            tier_parts = [
                f"A(data)={tiers.get('A', 0)}",
                f"B(assisted)={tiers.get('B', 0)}",
                f"C(prior)={tiers.get('C', 0)}",
            ]
            print(f"Fit quality tiers: {', '.join(tier_parts)}")

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

            # Detect bin_property columns (prefixed with "median_" from bin_properties)
            bp_cols = [c for c in summary_df.columns if c.startswith("median_")
                       and c not in ("median_redshift", "median_log_mass")]

            # Build header
            header = f"{'Population':<30} {'N_src':<8} {'z_med':<8} "
            for col in bp_cols:
                short = col.replace("median_", "")[:10]
                header += f"{short:<12}"
            header += f"{'T_rest[K]':<10} {'T_obs[K]':<10} {'L_IR[L☉]':<12} {'SFR[M☉/yr]':<12}"
            if self.greybody_fitter.use_mcmc:
                header += f" {'MCMC':<5}"
            print(header)
            print("-" * len(header))

            for _, row in summary_df.iterrows():
                pop_id = row["population_id"][:29]
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

                line = f"{pop_id:<30} {n_src:<8} {z_med:<8.2f} "
                for col in bp_cols:
                    val = row.get(col, np.nan)
                    if pd.notna(val):
                        line += f"{val:<12.2f}"
                    else:
                        line += f"{'--':<12}"
                line += f"{t_rest:<10.1f} {t_obs:<10.1f} {l_ir:<12.2e} {sfr:<12.1f}"

                if self.greybody_fitter.use_mcmc:
                    mcmc_used = "✓" if row["mcmc_used"] else "✗"
                    line += f" {mcmc_used:<5}"

                print(line)

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
                "temperature_prior": self.greybody_fitter.temperature_prior,
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

                # Save per-bin catalog properties
                props = _ensure_dict(sed.bin_properties)
                if props:
                    for col_name, val in props.items():
                        sed_grp.attrs[f"median_{col_name}"] = val

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
    temperature_prior: str = "flat",  # "flat", "schreiber", "viero"
    use_covariance: bool = True,
    correlation_matrix: dict | None = None,
    inflation_factors: dict | None = None,
    bootstrap_covariances: dict | None = None,
    exclude_wavelengths: list[float] | None = None,
    use_regression: bool = False,
    use_regression_prior: bool = False,
    regression_degree: str = 'linear',
    regression_property_names: list | None = None,
    regression_min_sources: int = 5,
    **greybody_kwargs,
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
        temperature_prior: Temperature prior ("flat", "schreiber", "viero")
        use_covariance: Whether to use covariance-aware fitting
        correlation_matrix: Custom correlation matrix (uses default if None)
        bootstrap_covariances: Dict mapping population_id -> bootstrap covariance matrix
        use_regression: Whether to also fit a regression greybody model
        use_regression_prior: Whether to use regression as prior for refitting
            (implies use_regression=True). High-SNR populations can depart from
            the regression surface; low-SNR are pulled toward it.
        regression_degree: Polynomial degree ('linear', 'interactions', 'quadratic')
        regression_property_names: Properties for regression (default: M*, z, β)
        regression_min_sources: Min sources per population for regression
        **greybody_kwargs: Additional kwargs forwarded to GreybodyFitter
            (T_rest_min, T_rest_max, snr_high, snr_low, etc.)

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
        temperature_prior=temperature_prior,
        use_covariance=use_covariance,
        correlation_matrix=correlation_matrix,
        inflation_factors=inflation_factors,
        bootstrap_covariances=bootstrap_covariances,
        exclude_wavelengths=exclude_wavelengths,
        use_regression=use_regression,
        use_regression_prior=use_regression_prior,
        regression_degree=regression_degree,
        regression_property_names=regression_property_names,
        regression_min_sources=regression_min_sources,
        **greybody_kwargs,
    )
