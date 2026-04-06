"""
Advanced SED fitting for Simstack4.

Extends the base Greybody model with covariance-aware fitting,
bootstrap error estimation, and regression-based fitting.

Classes
-------
CovarianceGreybodyFitter : Greybody fitting with full covariance matrix.
RegressionGreybodyFitter : Multi-population regression fitting.
"""
import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import curve_fit

try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False

from .greybody import Greybody, GreybodyFitter, SEDResults
from .cosmology import CosmologyCalculator
from .utils import setup_logging

logger = setup_logging()

class CovarianceGreybodyFitter(Greybody):
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
        super().__init__(inflation_factors=inflation_factors, **kwargs)
        self.correlation_matrix = correlation_matrix
        self.covariance_matrix = None
        self.inv_cov_chol = None  # Cholesky decomposition for efficiency

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

        # Positive definite check + Cholesky decomposition
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
        """Log-likelihood using full covariance matrix (Cholesky decomposition)."""
        amplitude, temperature = theta

        try:
            # Calculate model fluxes
            #print('using PAH')
            model_fluxes = self.greybody_model(
                wavelengths, amplitude, temperature, self.beta_fixed,
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
        Run MCMC with proper covariance treatment.

        wavelengths should be rest-frame. Temperature parameter is T_rest.
        """
        if not HAS_EMCEE:
            raise ImportError("emcee is required for MCMC fitting")

        # Setup covariance matrix
        fit_mask = np.ones(len(wavelengths), dtype=bool)
        self._setup_covariance_matrix(flux_errors, fit_mask)

        # Get initial guess (rest-frame)
        amplitude_guess, T_guess = self._get_initial_guess(
            wavelengths, fluxes, flux_errors, redshift
        )

        logger.info(f"MCMC with covariance: initial T_rest={T_guess:.1f}K")

        # Schreiber prior (now returns T_rest)
        if self.temperature_prior != "flat" and redshift > 0:
            T_prior, T_sigma = self.temperature_prior_relation(redshift)
            logger.info(
                f"T_dust prior ({self.temperature_prior}): T_rest={T_prior:.1f}±{T_sigma:.1f}K (z={redshift:.2f})"
            )
            alpha = 0.7
            T_start = alpha * T_prior + (1 - alpha) * T_guess
            T_spread = max(T_sigma, 2.0)
        else:
            T_start = T_guess
            T_spread = 3.0

        logger.info(
            f"Starting MCMC around T_rest={T_start:.1f}K with spread {T_spread:.1f}K"
        )

        # Setup walkers with rest-frame T bounds
        n_walkers = 32
        n_dim = 2
        pos = []

        for _i in range(n_walkers):
            amp_trial = amplitude_guess + np.random.normal(0, 0.2)
            temp_trial = T_start + np.random.normal(0, T_spread)

            amp_trial = np.clip(
                amp_trial, self.amplitude_min + 3.5, self.amplitude_max + 1.5
            )
            temp_trial = np.clip(
                temp_trial, self.T_rest_min + 1, self.T_rest_max - 2
            )

            test_prob = self.log_posterior_with_covariance(
                [amp_trial, temp_trial], wavelengths, fluxes, flux_errors, redshift
            )

            if np.isfinite(test_prob):
                pos.append([amp_trial, temp_trial])
            else:
                pos.append(
                    [
                        amplitude_guess + np.random.normal(0, 0.1),
                        T_start + np.random.normal(0, 1.0),
                    ]
                )

        pos = np.array(pos)

        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_dim,
            self.log_posterior_with_covariance,
            args=(wavelengths, fluxes, flux_errors, redshift),
        )

        logger.info(f"Running MCMC with covariance: {self.mcmc_iterations} iterations")

        try:
            sampler.run_mcmc(pos, self.mcmc_iterations, progress=True)
        except Exception:
            logger.info("Progress bar failed, running without progress")
            sampler.run_mcmc(pos, self.mcmc_iterations, progress=False)

        acceptance = np.mean(sampler.acceptance_fraction)
        logger.info(f"MCMC acceptance: {acceptance:.3f}")

        samples = sampler.get_chain(discard=self.mcmc_burn_in, flat=True)

        if len(samples) < 50:
            raise ValueError(f"Too few MCMC samples: {len(samples)}")

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
            "temperature_rest_frame": temperature_best,
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
        self, wavelengths, fluxes, flux_errors, redshift, bootstrap_cov=None,
        prior_override=None,
    ):
        """
        Enhanced SED fitting with covariance matrix (rest-frame).
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
        error_fit = flux_errors[fit_mask]

        # Transform to rest frame for covariance setup and MCMC
        z1 = max(1 + redshift, 1.001)
        wave_rest_fit = wave_fit / z1

        # Set up correlation matrix for these specific wavelengths
        if isinstance(self.correlation_matrix, dict) and self.correlation_matrix:
            logger.info("Converting correlation matrix from dict to numpy array")
            self.set_correlation_matrix_from_dict(self.correlation_matrix, wave_fit)

        # Setup covariance matrix
        self._setup_covariance_matrix(error_fit, fit_mask, bootstrap_cov)

        logger.info(f"Fitting SED with covariance: {len(wave_fit)} points")

        try:
            # super().fit_sed handles its own rest-frame transform
            initial_result = super().fit_sed(
                wave_fit, flux_fit, error_fit, redshift,
                prior_override=prior_override,
            )

            if not initial_result["fit_success"]:
                return initial_result

            # Run MCMC with covariance (pass rest-frame wavelengths)
            if self.use_mcmc:
                try:
                    mcmc_results = self.run_mcmc_with_covariance(
                        wave_rest_fit, flux_fit, error_fit, redshift
                    )

                    # Temperature from MCMC is T_rest
                    T_rest = mcmc_results["temperature_rest_frame"]
                    T_obs = T_rest / z1

                    initial_result.update(
                        {
                            "amplitude": mcmc_results["amplitude"],
                            "amplitude_error": mcmc_results["amplitude_error"],
                            "temperature_rest_frame": T_rest,
                            "temperature_observed_frame": T_obs,
                            "temperature_error": mcmc_results["temperature_error"],
                            "mcmc_used": True,
                            "covariance_used": mcmc_results["covariance_used"],
                            "bootstrap_covariance_used": bootstrap_cov is not None,
                            "mcmc_samples": mcmc_results["samples"],
                            "mcmc_acceptance": mcmc_results["acceptance_fraction"],
                        }
                    )

                    logger.info(
                        f"Final result with covariance: T_rest={T_rest:.1f}±{mcmc_results['temperature_error']:.1f}K"
                    )

                except Exception as e:
                    logger.warning(f"MCMC with covariance failed: {e}")

            return initial_result

        except Exception as e:
            logger.error(f"Covariance SED fitting failed: {e}")
            return {"fit_success": False, "reason": str(e)}

    def fit_sed(self, wavelengths, fluxes, flux_errors, redshift,
                prior_override=None):
        """
        Override parent fit_sed to automatically use covariance version
        """
        return self.fit_sed_with_covariance(
            wavelengths, fluxes, flux_errors, redshift,
            prior_override=prior_override,
        )


# ---------------------------------------------------------------------------
# Regression Greybody Fitter
# ---------------------------------------------------------------------------

class _DesignMatrix:
    """Polynomial design matrix with standardized inputs."""

    def __init__(self, property_names, degree='linear'):
        self.property_names = list(property_names)
        self.degree = degree
        self._fitted = False

    def fit_transform(self, properties):
        n = len(next(iter(properties.values())))
        self.means_ = {k: np.nanmean(properties[k]) for k in self.property_names}
        self.stds_ = {k: max(np.nanstd(properties[k]), 1e-10)
                      for k in self.property_names}
        self._fitted = True
        return self._build(properties, n)

    def transform(self, properties):
        assert self._fitted
        n = len(next(iter(properties.values())))
        return self._build(properties, n)

    def _build(self, properties, n):
        std = {}
        for name in self.property_names:
            std[name] = (np.asarray(properties[name]) - self.means_[name]) / self.stds_[name]

        cols = [np.ones(n)]
        names = ['1']
        for name in self.property_names:
            cols.append(std[name])
            names.append(name)

        if self.degree in ('interactions', 'quadratic'):
            for i, n1 in enumerate(self.property_names):
                for n2 in self.property_names[i + 1:]:
                    cols.append(std[n1] * std[n2])
                    names.append(f'{n1}×{n2}')

        if self.degree == 'quadratic':
            for name in self.property_names:
                cols.append(std[name]**2)
                names.append(f'{name}²')

        self.col_names_ = names
        return np.column_stack(cols)

    @property
    def n_features(self):
        return len(self.col_names_)


class RegressionGreybodyFitter:
    """
    Fit greybody SEDs across all populations simultaneously, with T_rest
    and log10(A) parameterized as polynomial functions of (M★, z, β_UV).

    Model per population i:
        ln(T_rest_i) = X_i @ θ_T
        log10(A_i)   = X_i @ θ_A
        S_ν(λ)       = greybody(λ/(1+z_i), log10(A_i), T_rest_i)

    Minimizes χ² = Σ_i Σ_j [(S_obs_ij − S_model_ij) / σ_ij]²
    """

    def __init__(self, greybody_fitter, property_names=None,
                 degree='linear', T_bounds=(12.0, 140.0),
                 min_sources=5):
        """
        Parameters
        ----------
        greybody_fitter : GreybodyFitter
            Existing fitter instance (provides greybody_model, calculate_LIR).
        property_names : list of str
            Population properties for the polynomial. Default: M*, z, β.
        degree : str
            'linear', 'interactions', or 'quadratic'.
        T_bounds : tuple
            (T_min, T_max) clipping bounds for predicted T_rest.
        min_sources : int
            Minimum sources per population to include in fit.
        """
        self.gb = greybody_fitter
        self.property_names = property_names or ['stellar_mass', 'redshift', 'beta_uv']
        self.degree = degree
        self.T_bounds = T_bounds
        self.min_sources = min_sources
        self.dm = _DesignMatrix(self.property_names, degree)

    def _predict_params(self, theta, X):
        nf = X.shape[1]
        ln_T = X @ theta[:nf]
        T_rest = np.clip(np.exp(ln_T), self.T_bounds[0], self.T_bounds[1])
        log10_A = X @ theta[nf:2 * nf]
        return T_rest, log10_A

    def _predict_fluxes(self, theta, X, wavelengths_obs, redshifts):
        T_rest, log10_A = self._predict_params(theta, X)
        N_pop = len(redshifts)
        N_bands = len(wavelengths_obs)
        S = np.zeros((N_pop, N_bands))
        for i in range(N_pop):
            wav_rest = wavelengths_obs / (1 + redshifts[i])
            S[i] = self.gb.greybody_model(wav_rest, log10_A[i], T_rest[i],
                                          self.gb.beta_fixed)
        return S

    def _cost(self, theta, X, wavelengths_obs, fluxes, errors, redshifts):
        S_model = self._predict_fluxes(theta, X, wavelengths_obs, redshifts)
        valid = errors > 0
        residuals = np.where(valid, (fluxes - S_model) / errors, 0.0)
        return np.sum(residuals**2)

    def _init_from_individual_fits(self, X, sed_results):
        """Initialize θ via OLS on existing per-population fits."""
        T_vals, A_vals, X_valid = [], [], []
        for i, (pop_id, sed) in enumerate(sed_results.items()):
            if (sed.dust_temperature_rest_frame is not None
                    and np.isfinite(sed.dust_temperature_rest_frame)
                    and sed.dust_temperature_rest_frame > 0
                    and sed.amplitude is not None
                    and np.isfinite(sed.amplitude)):
                T_vals.append(sed.dust_temperature_rest_frame)
                A_vals.append(sed.amplitude)
                X_valid.append(X[i])

        if len(T_vals) < X.shape[1]:
            return None

        T_vals = np.array(T_vals)
        A_vals = np.array(A_vals)
        X_valid = np.array(X_valid)

        theta_T, _, _, _ = np.linalg.lstsq(X_valid, np.log(T_vals), rcond=None)
        theta_A, _, _, _ = np.linalg.lstsq(X_valid, A_vals, rcond=None)
        logger.info(f"Regression init from {len(T_vals)} individual fits "
                    f"(T={T_vals.min():.1f}–{T_vals.max():.1f}K)")
        return np.concatenate([theta_T, theta_A])

    def _init_from_flux_ratios(self, X, wavelengths_obs, fluxes, errors,
                               redshifts):
        """Fallback init: grid-search T per population, regress."""
        N_pop = len(redshifts)
        T_est = np.full(N_pop, 30.0)
        A_est = np.full(N_pop, -35.0)

        for i in range(N_pop):
            wav_rest = wavelengths_obs / (1 + redshifts[i])
            f, e = fluxes[i], errors[i]
            snr = np.where(e > 0, np.abs(f / e), 0)
            good = snr > 1
            if np.sum(good) < 2:
                good = np.isin(np.arange(len(f)), np.argsort(-snr)[:2])

            best_chi2 = np.inf
            for T_try in np.arange(15, 75, 2):
                model = self.gb.greybody_model(wav_rest, 0.0, T_try,
                                               self.gb.beta_fixed)
                valid = (e > 0) & good
                if not np.any(valid):
                    continue
                m, fi, ei = model[valid], f[valid], e[valid]
                denom = np.sum(m**2 / ei**2)
                if denom == 0:
                    continue
                A_lin = np.sum(fi * m / ei**2) / denom
                if A_lin <= 0:
                    A_lin = 1e-10
                chi2 = np.sum(((fi - A_lin * m) / ei)**2)
                if chi2 < best_chi2:
                    best_chi2 = chi2
                    T_est[i] = T_try
                    A_est[i] = np.log10(A_lin)

        theta_T, _, _, _ = np.linalg.lstsq(X, np.log(T_est), rcond=None)
        theta_A, _, _, _ = np.linalg.lstsq(X, A_est, rcond=None)
        logger.info(f"Regression init from flux ratios "
                    f"(T={T_est.min():.0f}–{T_est.max():.0f}K)")
        return np.concatenate([theta_T, theta_A])

    def fit(self, wavelengths_obs, fluxes, errors, redshifts, properties,
            sed_results=None):
        """
        Fit the regression greybody model to all populations.

        Parameters
        ----------
        wavelengths_obs : (N_bands,) array, microns
        fluxes : (N_pop, N_bands) array, Jy
        errors : (N_pop, N_bands) array, Jy
        redshifts : (N_pop,) array
        properties : dict of {name: (N_pop,) array}
        sed_results : dict of {pop_id: SEDResults} or None
            Existing per-population fits for initialization.

        Returns
        -------
        result : dict
        """
        from scipy.optimize import minimize as sp_minimize

        X = self.dm.fit_transform(properties)
        nf = self.dm.n_features
        N_pop, N_bands = fluxes.shape
        n_valid = np.sum(errors > 0)
        n_params = 2 * nf

        logger.info(f"Regression greybody: {N_pop} pops × {N_bands} bands "
                    f"= {N_pop * N_bands} points, {n_params} params "
                    f"({self.dm.col_names_})")

        # Initialize
        theta0 = None
        if sed_results is not None:
            theta0 = self._init_from_individual_fits(X, sed_results)
        if theta0 is None:
            theta0 = self._init_from_flux_ratios(
                X, wavelengths_obs, fluxes, errors, redshifts
            )

        chi2_init = self._cost(theta0, X, wavelengths_obs, fluxes, errors,
                               redshifts)
        n_dof = n_valid - n_params
        logger.info(f"Regression init χ²/dof = {chi2_init / max(n_dof, 1):.2f}")

        # Optimize: Nelder-Mead then L-BFGS-B refinement
        result = sp_minimize(
            self._cost, theta0,
            args=(X, wavelengths_obs, fluxes, errors, redshifts),
            method='Nelder-Mead',
            options={'maxiter': 100_000, 'xatol': 1e-10, 'fatol': 1e-10,
                     'adaptive': True}
        )

        try:
            result2 = sp_minimize(
                self._cost, result.x,
                args=(X, wavelengths_obs, fluxes, errors, redshifts),
                method='L-BFGS-B',
                options={'maxiter': 10_000}
            )
            if result2.fun < result.fun:
                result = result2
        except Exception:
            pass

        chi2_final = result.fun
        T_rest, log10_A = self._predict_params(result.x, X)
        S_model = self._predict_fluxes(result.x, X, wavelengths_obs, redshifts)

        # Per-population χ²
        valid = errors > 0
        residuals = np.where(valid, (fluxes - S_model) / errors, np.nan)
        chi2_per_pop = np.nansum(residuals**2, axis=1)

        # L_IR per population
        L_IR = np.zeros(N_pop)
        for i in range(N_pop):
            D_L = self.gb.luminosity_distance(redshifts[i])
            L_IR[i] = self.gb._integrate_LIR(
                log10_A[i], T_rest[i], self.gb.beta_fixed,
                redshifts[i], D_L
            )

        # Polynomial coefficients
        theta_T = result.x[:nf]
        theta_A = result.x[nf:2 * nf]

        logger.info(f"Regression final χ²/dof = {chi2_final / max(n_dof, 1):.2f} "
                    f"(T={T_rest.min():.1f}–{T_rest.max():.1f}K, "
                    f"converged={result.success})")
        logger.info("Regression coefficients (standardized):")
        for j, col in enumerate(self.dm.col_names_):
            logger.info(f"  {col:<20} θ_T={theta_T[j]:+.4f}  "
                        f"θ_A={theta_A[j]:+.4f}")

        return {
            'theta_T': theta_T,
            'theta_A': theta_A,
            'T_rest': T_rest,
            'log10_A': log10_A,
            'L_IR': L_IR,
            'chi2': chi2_final,
            'chi2_per_pop': chi2_per_pop,
            'n_dof': n_dof,
            'chi2_reduced': chi2_final / max(n_dof, 1),
            'S_model': S_model,
            'converged': result.success,
            'design_columns': self.dm.col_names_,
        }


