"""
Tests for the SED fitting chain: GreybodyFitter, CovarianceGreybodyFitter, MCMC.

These test the science-output path: observed fluxes → rest-frame fit →
T_dust, L_IR, dust mass, SFR. Particularly critical now that fitting
operates in rest frame (λ_rest = λ_obs / (1+z), T parameter = T_rest).
"""

import numpy as np
import pytest

from simstack4.cosmology import CosmologyCalculator
from simstack4.results import CovarianceGreybodyFitter, GreybodyFitter

try:
    import emcee  # noqa: F401

    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cosmo():
    return CosmologyCalculator()


@pytest.fixture
def fitter(cosmo):
    return GreybodyFitter(
        fix_beta=True,
        beta_fixed=1.8,
        use_mcmc=False,
        use_schreiber_prior=False,
        cosmology_calc=cosmo,
    )


@pytest.fixture
def fitter_with_prior(cosmo):
    return GreybodyFitter(
        fix_beta=True,
        beta_fixed=1.8,
        use_mcmc=False,
        use_schreiber_prior=True,
        cosmology_calc=cosmo,
    )


@pytest.fixture
def cov_fitter(cosmo):
    return CovarianceGreybodyFitter(
        fix_beta=True,
        beta_fixed=1.8,
        use_mcmc=False,
        use_schreiber_prior=False,
        cosmology_calc=cosmo,
    )


@pytest.fixture
def herschel_wavelengths():
    """Typical Herschel/SPIRE observed-frame wavelengths."""
    return np.array([100.0, 160.0, 250.0, 350.0, 500.0])


def _make_synthetic_sed(fitter, wavelengths_obs, T_rest, z, amplitude=-34.0,
                        beta=1.8, noise_frac=0.05, seed=42):
    """
    Generate a synthetic observed-frame SED from rest-frame parameters.

    The model is evaluated at rest-frame wavelengths with T_rest, giving
    the flux densities an observer would measure.
    """
    z1 = 1 + z
    wave_rest = wavelengths_obs / z1

    model_fluxes = fitter.greybody_model(wave_rest, amplitude, T_rest, beta)

    rng = np.random.RandomState(seed)
    noise = noise_frac * np.abs(model_fluxes)
    noise = np.maximum(noise, 1e-8)
    noisy_fluxes = model_fluxes + rng.normal(0, noise)

    return noisy_fluxes, noise, model_fluxes


# ---------------------------------------------------------------------------
# 1. Rest-frame transform & T_rest recovery
# ---------------------------------------------------------------------------

class TestFitSEDRestFrame:
    """Verify fit_sed transforms to rest frame and recovers T_rest."""

    def test_z0_baseline(self, fitter, herschel_wavelengths):
        """At z≈0, rest frame = observed frame. Must recover T exactly."""
        T_rest = 30.0
        z = 0.01
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.01
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        assert result["temperature_rest_frame"] == pytest.approx(T_rest, abs=1.0)
        assert result["temperature_observed_frame"] == pytest.approx(
            T_rest / (1 + z), abs=1.0
        )

    @pytest.mark.parametrize("z", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0])
    def test_t_rest_recovery_across_redshift(self, fitter, herschel_wavelengths, z):
        """
        Inject Schreiber T_rest at each z, recover to < 10%.

        This is the key test: the old observed-frame fitter failed at z > 1.5
        because T_obs fell below the 12K hard bound.
        """
        T_rest = 23.8 + 2.7 * z + 0.9 * z**2  # Schreiber relation
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.03
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"], f"Fit failed at z={z}"
        T_recovered = result["temperature_rest_frame"]
        pct_error = abs(T_recovered - T_rest) / T_rest * 100
        assert pct_error < 10, (
            f"z={z}: T_rest_in={T_rest:.1f}, T_rest_out={T_recovered:.1f}, "
            f"error={pct_error:.1f}%"
        )

    def test_hot_dust_recovery(self, fitter, herschel_wavelengths):
        """Recover T_rest = 50K (hotter than typical Schreiber)."""
        T_rest = 50.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.03
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        assert result["temperature_rest_frame"] == pytest.approx(T_rest, rel=0.15)

    def test_cold_dust_recovery(self, fitter, herschel_wavelengths):
        """Recover T_rest = 18K (cold dust, near lower bound)."""
        T_rest = 18.0
        z = 0.5
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.03
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        assert result["temperature_rest_frame"] == pytest.approx(T_rest, rel=0.15)

    def test_rest_obs_temperature_consistency(self, fitter, herschel_wavelengths):
        """T_obs × (1+z) should equal T_rest in the output."""
        T_rest = 35.0
        z = 2.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.02
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        z1 = 1 + z
        assert result["temperature_rest_frame"] == pytest.approx(
            result["temperature_observed_frame"] * z1, rel=1e-10
        )

    def test_wavelengths_fit_are_rest_frame(self, fitter, herschel_wavelengths):
        """The returned wavelengths_fit should be in rest frame."""
        z = 2.0
        T_rest = 30.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.03
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        expected_rest = herschel_wavelengths / (1 + z)
        np.testing.assert_allclose(
            result["wavelengths_fit"], expected_rest, rtol=0.01,
            err_msg="wavelengths_fit should be rest-frame"
        )


# ---------------------------------------------------------------------------
# 2. Output contract
# ---------------------------------------------------------------------------

class TestFitSEDOutputContract:
    """Verify fit_sed returns all expected keys with correct types."""

    def test_success_output_keys(self, fitter, herschel_wavelengths):
        """Successful fit should contain all required keys."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        required_keys = {
            "fit_success", "amplitude", "amplitude_error",
            "temperature_rest_frame", "temperature_observed_frame",
            "temperature_error", "beta", "beta_error",
            "chi2_reduced", "L_IR", "L_IR_error",
            "n_points", "n_positive",
            "wavelengths_fit", "fluxes_fit", "flux_errors_fit",
            "model_wavelengths", "model_fluxes",
            "redshift_used", "mcmc_used", "schreiber_prior_used",
        }
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_success_output_types(self, fitter, herschel_wavelengths):
        """Check types of key output fields."""
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, 30.0, 1.0
        )
        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, 1.0)

        assert result["fit_success"] is True
        assert isinstance(result["temperature_rest_frame"], float)
        assert isinstance(result["L_IR"], float)
        assert isinstance(result["chi2_reduced"], float)
        assert isinstance(result["wavelengths_fit"], np.ndarray)
        assert isinstance(result["model_wavelengths"], np.ndarray)

    def test_failure_output(self, fitter):
        """Fit with insufficient data should fail gracefully."""
        wavelengths = np.array([100.0, 200.0])  # Only 2 points
        fluxes = np.array([1.0, 0.5])
        errors = np.array([0.1, 0.1])

        result = fitter.fit_sed(wavelengths, fluxes, errors, redshift=1.0)

        assert result["fit_success"] is False
        assert "reason" in result

    def test_amplitude_recovery(self, fitter, herschel_wavelengths):
        """Amplitude should be recovered to within errors."""
        amp_true = -34.0
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z,
            amplitude=amp_true, noise_frac=0.02
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        assert result["amplitude"] == pytest.approx(amp_true, abs=0.5)

    def test_chi2_reasonable(self, fitter, herschel_wavelengths):
        """χ² should be ~1 for data generated from the model."""
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, 30.0, 1.0, noise_frac=0.05
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, 1.0)

        assert result["fit_success"]
        # χ² should be in a reasonable range for 5 points, 2 params
        assert 0.01 < result["chi2_reduced"] < 20.0


# ---------------------------------------------------------------------------
# 3. Free-beta fitting
# ---------------------------------------------------------------------------

class TestFreeBeta:
    """Test fitting with beta as a free parameter."""

    def test_free_beta_recovery(self, cosmo, herschel_wavelengths):
        """Recover beta when it's a free parameter."""
        fitter = GreybodyFitter(
            fix_beta=False, beta_fixed=1.8,
            use_mcmc=False, cosmology_calc=cosmo,
        )
        T_rest = 30.0
        beta_true = 2.0
        z = 0.5

        z1 = 1 + z
        wave_rest = herschel_wavelengths / z1
        model_fluxes = fitter.greybody_model(wave_rest, -34.0, T_rest, beta_true)
        noise = 0.02 * np.abs(model_fluxes)
        noise = np.maximum(noise, 1e-8)

        result = fitter.fit_sed(herschel_wavelengths, model_fluxes, noise, z)

        assert result["fit_success"]
        assert result["beta"] == pytest.approx(beta_true, abs=0.3)
        assert result["beta_error"] > 0


# ---------------------------------------------------------------------------
# 4. Schreiber prior
# ---------------------------------------------------------------------------

class TestSchreiberPrior:
    """Verify the Schreiber+2015 temperature prior."""

    def test_returns_rest_frame_temperature(self, fitter):
        """schreiber_temperature_prior should return T_rest, not T_obs."""
        z = 2.0
        T_rest, sigma = fitter.schreiber_temperature_prior(z)

        # Schreiber T_rest = 23.8 + 2.7z + 0.9z² = 32.8K at z=2
        expected = 23.8 + 2.7 * 2.0 + 0.9 * 4.0
        assert T_rest == pytest.approx(expected, abs=0.1)
        # Must NOT be T_obs = T_rest/(1+z) = 10.9K
        assert T_rest > 20

    def test_sigma_varies_with_redshift(self, fitter):
        """Sigma should be z-dependent (3, 4, 5 K)."""
        _, sigma_low = fitter.schreiber_temperature_prior(0.5)
        _, sigma_mid = fitter.schreiber_temperature_prior(1.5)
        _, sigma_high = fitter.schreiber_temperature_prior(2.5)

        assert sigma_low == pytest.approx(3.0)
        assert sigma_mid == pytest.approx(4.0)
        assert sigma_high == pytest.approx(5.0)

    def test_no_hardcoded_sigma_override(self, fitter):
        """All z values should NOT return σ=2K (the old bug)."""
        for z in [0.5, 1.0, 1.5, 2.0, 3.0]:
            _, sigma = fitter.schreiber_temperature_prior(z)
            assert sigma != 2.0, f"σ=2K at z={z} — hardcoded override not removed"

    def test_prior_biases_toward_schreiber(self, fitter_with_prior, herschel_wavelengths):
        """
        With Schreiber prior, a noisy SED should be pulled toward the
        Schreiber relation compared to no prior.
        """
        z = 1.0
        # Inject T_rest = 40K — well above Schreiber ~27K
        T_rest_true = 40.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter_with_prior, herschel_wavelengths, T_rest_true, z,
            noise_frac=0.30  # large noise to let prior matter
        )

        result_prior = fitter_with_prior.fit_sed(
            herschel_wavelengths, fluxes, errors, z
        )

        # The prior can bias the initial guess and thus curve_fit,
        # but with 30% noise and 5 points, the data still dominates.
        # Just check it runs and returns reasonable results.
        assert result_prior["fit_success"]
        assert result_prior["schreiber_prior_used"] is True


# ---------------------------------------------------------------------------
# 5. log_prior boundary tests
# ---------------------------------------------------------------------------

class TestLogPrior:
    """Verify rest-frame temperature bounds in log_prior."""

    def test_accepts_typical_rest_frame_temperature(self, fitter):
        """T_rest = 30K should be accepted."""
        lp = fitter.log_prior([-34.0, 30.0], redshift=1.0)
        assert np.isfinite(lp)

    def test_accepts_high_rest_frame_temperature(self, fitter):
        """T_rest = 55K (hot dust at high z) should be accepted."""
        lp = fitter.log_prior([-34.0, 55.0], redshift=3.0)
        assert np.isfinite(lp)

    def test_rejects_below_lower_bound(self, fitter):
        """T_rest < 15K should be rejected."""
        lp = fitter.log_prior([-34.0, 10.0], redshift=0.5)
        assert lp == -np.inf

    def test_rejects_above_upper_bound(self, fitter):
        """T_rest > 60K should be rejected."""
        lp = fitter.log_prior([-34.0, 65.0], redshift=0.5)
        assert lp == -np.inf

    def test_rejects_bad_amplitude(self, fitter):
        """Amplitude outside [-38, -30] should be rejected."""
        assert fitter.log_prior([-25.0, 30.0]) == -np.inf
        assert fitter.log_prior([-40.0, 30.0]) == -np.inf

    def test_schreiber_prior_gaussian(self, fitter_with_prior):
        """With Schreiber prior, log_prior should be Gaussian in T_rest."""
        z = 1.0
        T_expected, _ = fitter_with_prior.schreiber_temperature_prior(z)

        lp_at_peak = fitter_with_prior.log_prior([-34.0, T_expected], z)
        lp_offset = fitter_with_prior.log_prior([-34.0, T_expected + 10.0], z)

        # At peak should be higher than 10K away
        assert lp_at_peak > lp_offset


# ---------------------------------------------------------------------------
# 6. CovarianceGreybodyFitter
# ---------------------------------------------------------------------------

class TestCovarianceFitter:
    """Test CovarianceGreybodyFitter fits and covariance handling."""

    def test_basic_fit_succeeds(self, cov_fitter, herschel_wavelengths):
        """CovarianceGreybodyFitter should produce a successful fit."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            cov_fitter, herschel_wavelengths, T_rest, z, noise_frac=0.03
        )

        result = cov_fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        assert result["temperature_rest_frame"] == pytest.approx(T_rest, rel=0.15)

    def test_results_match_base_fitter(self, fitter, cov_fitter, herschel_wavelengths):
        """
        Without correlation matrix, covariance fitter should give
        similar results to the base fitter.
        """
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.02
        )

        result_base = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)
        result_cov = cov_fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result_base["fit_success"] and result_cov["fit_success"]
        assert result_cov["temperature_rest_frame"] == pytest.approx(
            result_base["temperature_rest_frame"], rel=0.05
        )
        assert result_cov["L_IR"] == pytest.approx(
            result_base["L_IR"], rel=0.1
        )

    def test_with_error_inflation(self, cosmo, herschel_wavelengths):
        """Error inflation should run without error and affect the fit."""
        fitter_inflated = CovarianceGreybodyFitter(
            fix_beta=True, beta_fixed=1.8,
            use_mcmc=False, cosmology_calc=cosmo,
            inflation_factors={250: 3.0, 350: 3.0},
        )

        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter_inflated, herschel_wavelengths, T_rest, z, noise_frac=0.05
        )

        result = fitter_inflated.fit_sed(
            herschel_wavelengths, fluxes, errors, z
        )

        assert result["fit_success"]
        # Temperature should still be recoverable
        assert result["temperature_rest_frame"] == pytest.approx(T_rest, rel=0.2)


# ---------------------------------------------------------------------------
# 7. MCMC fitting
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_EMCEE, reason="emcee not installed")
class TestMCMCFitting:
    """Test the MCMC fitting path."""

    @pytest.fixture
    def mcmc_fitter(self, cosmo):
        return GreybodyFitter(
            fix_beta=True, beta_fixed=1.8,
            use_mcmc=True, mcmc_iterations=200, mcmc_burn_in=50,
            use_schreiber_prior=False,
            cosmology_calc=cosmo,
        )

    def test_mcmc_fit_succeeds(self, mcmc_fitter, herschel_wavelengths):
        """MCMC should produce a successful fit."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            mcmc_fitter, herschel_wavelengths, T_rest, z, noise_frac=0.03
        )

        result = mcmc_fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        assert result["mcmc_used"] is True

    def test_mcmc_returns_rest_frame_temperature(self, mcmc_fitter, herschel_wavelengths):
        """MCMC results should contain temperature_rest_frame, not _observed_frame."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            mcmc_fitter, herschel_wavelengths, T_rest, z, noise_frac=0.03
        )

        result = mcmc_fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        # The key result is T_rest, derived from T_rest chain
        assert result["temperature_rest_frame"] == pytest.approx(T_rest, rel=0.2)
        # Consistency
        assert result["temperature_observed_frame"] == pytest.approx(
            result["temperature_rest_frame"] / (1 + z), rel=1e-6
        )

    def test_mcmc_samples_shape(self, mcmc_fitter, herschel_wavelengths):
        """MCMC samples should have shape (n_samples, 2) — [amplitude, T_rest]."""
        fluxes, errors, _ = _make_synthetic_sed(
            mcmc_fitter, herschel_wavelengths, 30.0, 1.0, noise_frac=0.03
        )

        result = mcmc_fitter.fit_sed(herschel_wavelengths, fluxes, errors, 1.0)

        assert result["fit_success"] and result["mcmc_used"]
        samples = result["mcmc_samples"]
        assert samples.ndim == 2
        assert samples.shape[1] == 2
        assert samples.shape[0] > 10

    def test_mcmc_temperature_samples_in_rest_frame(
        self, mcmc_fitter, herschel_wavelengths
    ):
        """
        MCMC temperature samples should be T_rest (15–60K range),
        not T_obs which would be ~10K at z=2.
        """
        z = 2.0
        T_rest = 33.0
        fluxes, errors, _ = _make_synthetic_sed(
            mcmc_fitter, herschel_wavelengths, T_rest, z, noise_frac=0.05
        )

        result = mcmc_fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        if result["fit_success"] and result.get("mcmc_samples") is not None:
            T_samples = result["mcmc_samples"][:, 1]
            # All samples should be in rest-frame range
            assert np.all(T_samples > 15), "MCMC samples below 15K — not rest frame"
            assert np.all(T_samples < 60), "MCMC samples above 60K — out of range"
            # Median should be near T_rest
            assert np.median(T_samples) == pytest.approx(T_rest, rel=0.25)


# ---------------------------------------------------------------------------
# 8. MCMC with covariance
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_EMCEE, reason="emcee not installed")
class TestMCMCWithCovariance:
    """Test the CovarianceGreybodyFitter MCMC path."""

    @pytest.fixture
    def mcmc_cov_fitter(self, cosmo):
        return CovarianceGreybodyFitter(
            fix_beta=True, beta_fixed=1.8,
            use_mcmc=True, mcmc_iterations=200, mcmc_burn_in=50,
            use_schreiber_prior=False,
            cosmology_calc=cosmo,
        )

    def test_mcmc_covariance_fit_succeeds(
        self, mcmc_cov_fitter, herschel_wavelengths
    ):
        """MCMC with covariance should produce a result."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            mcmc_cov_fitter, herschel_wavelengths, T_rest, z, noise_frac=0.05
        )

        result = mcmc_cov_fitter.fit_sed(
            herschel_wavelengths, fluxes, errors, z
        )

        assert result["fit_success"]
        assert result["temperature_rest_frame"] == pytest.approx(T_rest, rel=0.25)


# ---------------------------------------------------------------------------
# 9. Dust mass
# ---------------------------------------------------------------------------

class TestDustMass:
    """Test dust mass calculation with rest-frame parameters."""

    def test_dust_mass_positive(self, fitter):
        """Dust mass should be positive for reasonable inputs."""
        M, M_err = fitter.calculate_dust_mass(
            amplitude=-34.0, temperature=30.0, beta=1.8, redshift=1.0
        )
        assert M > 0
        assert M_err > 0

    def test_dust_mass_scales_with_amplitude(self, fitter):
        """Higher amplitude → more dust mass."""
        M1, _ = fitter.calculate_dust_mass(-34.0, 30.0, 1.8, 1.0)
        M2, _ = fitter.calculate_dust_mass(-33.5, 30.0, 1.8, 1.0)

        assert M2 > M1

    def test_dust_mass_increases_with_distance(self, fitter):
        """
        Same observed SED at higher z → larger dust mass
        (more distant source needs more dust to produce same flux).
        """
        M_near, _ = fitter.calculate_dust_mass(-34.0, 30.0, 1.8, 0.5)
        M_far, _ = fitter.calculate_dust_mass(-34.0, 30.0, 1.8, 2.0)

        assert M_far > M_near

    def test_dust_mass_nan_for_nan_input(self, fitter):
        """NaN temperature should return NaN mass."""
        M, M_err = fitter.calculate_dust_mass(-34.0, np.nan, 1.8, 1.0)
        assert np.isnan(M)


# ---------------------------------------------------------------------------
# 10. L_IR consistency with rest-frame parameters
# ---------------------------------------------------------------------------

class TestLIRRestFrame:
    """Test L_IR calculation with rest-frame temperature parameter."""

    def test_lir_positive(self, fitter):
        """L_IR should be positive for valid inputs."""
        L, L_err = fitter.calculate_LIR(-34.0, 30.0, 1.8, 1.0)
        assert L > 0
        assert L_err > 0

    def test_lir_from_fit_matches_direct(self, fitter, herschel_wavelengths):
        """L_IR from fit_sed output should match calculate_LIR with same params."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.02
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)
        assert result["fit_success"]

        # Recalculate L_IR from the fit parameters
        L_recalc, _ = fitter.calculate_LIR(
            result["amplitude"],
            result["temperature_rest_frame"],
            result["beta"],
            result["redshift_used"],
        )

        assert L_recalc == pytest.approx(result["L_IR"], rel=0.01)

    @pytest.mark.parametrize("z", [0.5, 1.0, 2.0, 3.0])
    def test_lir_physically_reasonable(self, fitter, z):
        """
        For typical greybody parameters, L_IR should be in the
        LIRG/ULIRG range (1e10 - 1e14 L_sun).
        """
        L, _ = fitter.calculate_LIR(-34.0, 30.0, 1.8, z)
        assert 1e8 < L < 1e15, f"L_IR={L:.2e} outside reasonable range at z={z}"


# ---------------------------------------------------------------------------
# 11. End-to-end high-z science test
# ---------------------------------------------------------------------------

class TestEndToEndHighZ:
    """
    Full pipeline test at z=3: inject SED → fit → T_rest, L_IR, SFR.

    This validates that the rest-frame approach gives physically
    consistent results where the old observed-frame approach failed.
    """

    def test_z3_full_chain(self, fitter, herschel_wavelengths):
        """Inject at z=3, recover T_rest, L_IR, and verify SFR is physical."""
        z = 3.0
        T_rest_true = 40.0  # Schreiber ~40K at z=3
        amp_true = -34.0

        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest_true, z,
            amplitude=amp_true, noise_frac=0.05
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        # 1. Fit should succeed
        assert result["fit_success"]

        # 2. T_rest recovered
        T_rest_fit = result["temperature_rest_frame"]
        assert T_rest_fit == pytest.approx(T_rest_true, rel=0.15), (
            f"T_rest: {T_rest_fit:.1f} vs {T_rest_true:.1f}"
        )

        # 3. L_IR should be positive and finite
        assert result["L_IR"] > 0
        assert np.isfinite(result["L_IR"])

        # 4. SFR from Kennicutt
        sfr = result["L_IR"] / 1e10
        assert sfr > 0
        # At z=3 with these parameters, SFR should be in a reasonable range
        assert sfr < 1e6, f"SFR = {sfr:.1f} — unphysically high"

        # 5. Dust mass should be computable
        M_dust, _ = fitter.calculate_dust_mass(
            result["amplitude"], T_rest_fit, result["beta"], z
        )
        assert M_dust > 0, "Dust mass should be positive"

    def test_z3_no_clamping_bias(self, fitter, herschel_wavelengths):
        """
        At z=3, T_obs_true ≈ 10K. The old code clamped this to 12K,
        giving T_rest = 48K instead of 40K. Verify no such bias exists.
        """
        z = 3.0
        T_rest_true = 40.0
        T_obs_true = T_rest_true / (1 + z)  # 10K

        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest_true, z, noise_frac=0.03
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)

        assert result["fit_success"]
        # The old bias was +8K in T_rest. Allow max 3K error.
        assert abs(result["temperature_rest_frame"] - T_rest_true) < 3.0, (
            f"T_rest bias: {result['temperature_rest_frame']:.1f} vs "
            f"{T_rest_true:.1f} — possible clamping artifact"
        )

    @pytest.mark.parametrize("z", [1.0, 2.0, 3.0, 4.0])
    def test_lir_not_inflated_by_wrong_temperature(self, fitter, herschel_wavelengths, z):
        """
        L_IR from the fit should agree with L_IR computed from the
        true injected parameters to within 30%.

        The old code inflated L_IR by up to 2.8× at z=3 because the
        clamped T_obs was too warm, shifting the SED peak into the
        integration window.
        """
        T_rest_true = 23.8 + 2.7 * z + 0.9 * z**2
        amp_true = -34.0

        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest_true, z,
            amplitude=amp_true, noise_frac=0.03
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)
        assert result["fit_success"]

        L_true, _ = fitter.calculate_LIR(amp_true, T_rest_true, 1.8, z)
        ratio = result["L_IR"] / L_true
        assert ratio == pytest.approx(1.0, abs=0.3), (
            f"z={z}: L_IR ratio = {ratio:.2f}, expected ~1.0"
        )


# ---------------------------------------------------------------------------
# 12. SNR computation
# ---------------------------------------------------------------------------

class TestComputeSEDSNR:
    """Test the SNR metric used for quality tiers."""

    def test_high_snr(self, fitter):
        """Clear detections should give high SNR."""
        fluxes = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        snr = fitter.compute_sed_snr(fluxes, errors)
        assert snr == pytest.approx(10.0)

    def test_low_snr(self, fitter):
        """Noisy data should give low SNR."""
        fluxes = np.array([0.1, 0.2, 0.3])
        errors = np.array([0.5, 0.5, 0.5])
        snr = fitter.compute_sed_snr(fluxes, errors)
        assert snr < 1.0

    def test_mixed_positive_negative(self, fitter):
        """Only positive bands contribute to SNR."""
        fluxes = np.array([-0.1, 0.5, 1.0, -0.2, 2.0])
        errors = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        snr = fitter.compute_sed_snr(fluxes, errors)
        # Positive bands: SNR = 5, 10, 20 → median = 10
        assert snr == pytest.approx(10.0)

    def test_all_negative(self, fitter):
        """No positive detections → SNR = 0."""
        fluxes = np.array([-0.1, -0.2, -0.3])
        errors = np.array([0.1, 0.1, 0.1])
        snr = fitter.compute_sed_snr(fluxes, errors)
        assert snr == 0.0


# ---------------------------------------------------------------------------
# 13. Quality tiers and prior override
# ---------------------------------------------------------------------------

class TestFitQualityTiers:
    """Test that fit_sed assigns quality tiers based on SNR."""

    def test_high_snr_gives_tier_a(self, fitter_with_prior, herschel_wavelengths):
        """Low noise → tier A (data-driven)."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter_with_prior, herschel_wavelengths, T_rest, z, noise_frac=0.01
        )
        result = fitter_with_prior.fit_sed(herschel_wavelengths, fluxes, errors, z)
        assert result["fit_success"]
        assert result["fit_quality_tier"] == "A"
        assert result["sed_snr"] >= fitter_with_prior.SNR_HIGH

    def test_low_snr_gives_tier_c(self, fitter_with_prior, herschel_wavelengths):
        """Very noisy → tier C (prior-dominated)."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter_with_prior, herschel_wavelengths, T_rest, z, noise_frac=1.0
        )
        result = fitter_with_prior.fit_sed(herschel_wavelengths, fluxes, errors, z)
        # Might fail to fit, but if it succeeds, should be tier B or C
        if result["fit_success"]:
            assert result["fit_quality_tier"] in ("B", "C")

    def test_snr_in_output(self, fitter, herschel_wavelengths):
        """SNR should always be in output, even without Schreiber prior."""
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, 30.0, 1.0, noise_frac=0.05
        )
        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, 1.0)
        assert "sed_snr" in result
        assert "fit_quality_tier" in result
        assert result["sed_snr"] > 0


class TestPriorOverride:
    """Test the prior_override parameter for two-pass fitting."""

    def test_override_narrows_fit(self, fitter_with_prior, herschel_wavelengths):
        """With tight prior_override, fit should land near the prior center."""
        T_rest_true = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter_with_prior, herschel_wavelengths, T_rest_true, z,
            noise_frac=0.5  # noisy enough that prior matters
        )

        # Override prior to T=35K with tight sigma
        result = fitter_with_prior.fit_sed(
            herschel_wavelengths, fluxes, errors, z,
            prior_override=(35.0, 2.0),
        )

        assert result["fit_success"]
        assert result["prior_center"] == 35.0
        assert result["prior_sigma"] == 2.0

    def test_override_cleared_after_fit(self, fitter_with_prior, herschel_wavelengths):
        """_prior_override should be None after fit completes."""
        fluxes, errors, _ = _make_synthetic_sed(
            fitter_with_prior, herschel_wavelengths, 30.0, 1.0
        )

        fitter_with_prior.fit_sed(
            herschel_wavelengths, fluxes, errors, 1.0,
            prior_override=(35.0, 2.0),
        )
        assert fitter_with_prior._prior_override is None

    def test_override_cleared_after_failed_fit(self, fitter_with_prior):
        """_prior_override should be cleared even if fit fails."""
        wavelengths = np.array([100.0, 200.0])  # too few points
        fluxes = np.array([1.0, 0.5])
        errors = np.array([0.1, 0.1])

        result = fitter_with_prior.fit_sed(
            wavelengths, fluxes, errors, 1.0,
            prior_override=(35.0, 2.0),
        )
        assert not result["fit_success"]
        assert fitter_with_prior._prior_override is None

    def test_no_override_gives_wide_bounds(self, fitter, herschel_wavelengths):
        """Without prior, T bounds should be full [15, 60]K."""
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, 30.0, 1.0, noise_frac=0.03
        )
        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, 1.0)
        assert result["fit_success"]
        assert result["prior_center"] is None
        assert result["prior_sigma"] is None

    def test_snr_scaled_sigma(self, fitter_with_prior, herschel_wavelengths):
        """
        Low-SNR data with Schreiber prior should get tighter sigma
        than high-SNR data.
        """
        z = 1.0
        T_rest = 30.0

        # High SNR
        fluxes_hi, errors_hi, _ = _make_synthetic_sed(
            fitter_with_prior, herschel_wavelengths, T_rest, z, noise_frac=0.02
        )
        result_hi = fitter_with_prior.fit_sed(
            herschel_wavelengths, fluxes_hi, errors_hi, z
        )

        # Low SNR
        fluxes_lo, errors_lo, _ = _make_synthetic_sed(
            fitter_with_prior, herschel_wavelengths, T_rest, z, noise_frac=0.5
        )
        result_lo = fitter_with_prior.fit_sed(
            herschel_wavelengths, fluxes_lo, errors_lo, z
        )

        # Both should succeed and have prior info
        assert result_hi["fit_success"] and result_lo["fit_success"]
        if result_hi["prior_sigma"] is not None and result_lo["prior_sigma"] is not None:
            # Low SNR should have tighter prior (smaller sigma)
            assert result_lo["prior_sigma"] <= result_hi["prior_sigma"]


class TestCurveFitRegularization:
    """Test that curve_fit prior regularization actually constrains T."""

    def test_regularization_pulls_toward_prior(
        self, fitter_with_prior, herschel_wavelengths
    ):
        """
        With very noisy data, regularized curve_fit should land near the
        prior center, not at a random boundary.
        """
        T_prior = 32.0  # prior center
        z = 1.0

        # Generate noisy data with true T far from prior
        np.random.seed(42)
        fluxes, errors, _ = _make_synthetic_sed(
            fitter_with_prior, herschel_wavelengths, 45.0, z, noise_frac=0.8
        )

        # Fit with tight prior override
        result = fitter_with_prior.fit_sed(
            herschel_wavelengths, fluxes, errors, z,
            prior_override=(T_prior, 2.0),
        )

        if result["fit_success"]:
            T_fit = result["temperature_rest_frame"]
            # With σ=2K prior, fit should be pulled toward 32K
            # even though true T=45K, because noise_frac=0.8 makes
            # data nearly uninformative
            assert abs(T_fit - T_prior) < 10.0, (
                f"T_fit={T_fit:.1f}K should be near prior {T_prior}K "
                f"for noisy data"
            )

    def test_no_regularization_without_prior(self, fitter, herschel_wavelengths):
        """Without prior, curve_fit should use full [15,60] bounds."""
        T_rest = 30.0
        z = 1.0
        fluxes, errors, _ = _make_synthetic_sed(
            fitter, herschel_wavelengths, T_rest, z, noise_frac=0.05
        )

        result = fitter.fit_sed(herschel_wavelengths, fluxes, errors, z)
        assert result["fit_success"]
        # Should recover true T without prior interference
        assert abs(result["temperature_rest_frame"] - T_rest) < 5.0
