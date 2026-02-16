"""
Luminosity estimator validation tests for simstack4.

Tests validate the full chain:
  greybody model → L_IR integration → D_L → SFR

These tests document and quantify two known bugs:
  BUG #1: GreybodyFitter.luminosity_distance() uses Hubble-law (c*z/H0)
  BUG #2: calculate_LIR() evaluates observed-frame model at rest-frame wavelengths

Tests are structured to:
  - Verify the greybody model against known physics
  - Quantify how wrong L_IR is due to each bug
  - Provide regression targets for when the bugs are fixed
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the classes under test
# ---------------------------------------------------------------------------

from simstack4.results import GreybodyFitter
from simstack4.cosmology import CosmologyCalculator
from simstack4.config import Cosmology


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fitter():
    """Standard GreybodyFitter with default settings."""
    return GreybodyFitter(fix_beta=True, beta_fixed=1.8, use_mcmc=False)


@pytest.fixture
def cosmo():
    """Proper cosmology calculator (Planck18)."""
    return CosmologyCalculator(Cosmology.PLANCK18)


# ---------------------------------------------------------------------------
# Test 7: Greybody model sanity checks
# ---------------------------------------------------------------------------


class TestGreybodyModel:
    """Verify the modified blackbody model against known physics."""

    def test_peak_wavelength_wien(self, fitter):
        """
        The SED peak should roughly follow Wien's displacement law.

        For a modified blackbody (ν^β × B_ν), the peak shifts to shorter
        wavelengths compared to a pure Planck function. For β=1.8,
        the peak is at roughly λ_peak ≈ 2900/T * correction_factor μm
        (where correction depends on β).

        We just verify the peak moves monotonically with temperature
        and is in a reasonable range.
        """
        wavelengths = np.logspace(np.log10(30), np.log10(1500), 500)

        peak_wavelengths = []
        for temp in [20.0, 30.0, 40.0, 50.0]:
            sed = fitter.greybody_model(wavelengths, amplitude=-34.0, temperature=temp)
            peak_idx = np.argmax(sed)
            peak_wavelengths.append(wavelengths[peak_idx])

        # Peak should shift to shorter wavelengths at higher temperatures
        for i in range(len(peak_wavelengths) - 1):
            assert peak_wavelengths[i + 1] < peak_wavelengths[i], (
                f"Peak not shifting blueward: T={[20,30,40,50][i]}K → "
                f"λ_peak={peak_wavelengths[i]:.0f}μm, "
                f"T={[20,30,40,50][i+1]}K → λ_peak={peak_wavelengths[i+1]:.0f}μm"
            )

        # At T=35K, peak should be roughly 80-200μm for β=1.8 greybody
        sed_35k = fitter.greybody_model(wavelengths, amplitude=-34.0, temperature=35.0)
        peak_35k = wavelengths[np.argmax(sed_35k)]
        assert 60 < peak_35k < 250, f"T=35K peak at {peak_35k:.0f}μm — out of range"

    def test_rayleigh_jeans_slope(self, fitter):
        """
        On the Rayleigh-Jeans (long wavelength) side, the modified blackbody
        goes as S_ν ∝ ν^(2+β). In wavelength: S_ν ∝ λ^(-2-β).

        Check that the log-log slope at long wavelengths approaches 2+β.
        """
        beta = 1.8
        # Need very long wavelengths (>> hν/kT ~ 400μm at T=35K) for RJ limit
        wavelengths = np.array([3000.0, 5000.0, 8000.0, 12000.0, 20000.0])
        sed = fitter.greybody_model(
            wavelengths, amplitude=-34.0, temperature=35.0, beta=beta
        )

        # Convert to frequency space for slope check
        c_light = 299792458.0  # m/s
        nu = c_light * 1e6 / wavelengths  # Hz

        # Log-log slope: d(log S) / d(log ν)
        log_nu = np.log(nu)
        log_s = np.log(sed)

        slopes = np.diff(log_s) / np.diff(log_nu)
        expected_slope = 2.0 + beta  # Modified BB: S_ν ∝ ν^(2+β) in RJ limit

        # Longest wavelength pair should be closest to asymptotic slope
        assert slopes[-1] == pytest.approx(expected_slope, abs=0.05), (
            f"Deep RJ slope: got {slopes[-1]:.3f}, expected {expected_slope:.1f}"
        )

    def test_power_law_transition(self, fitter):
        """
        At short wavelengths (Wien side), the model transitions to a
        power law S_ν ∝ ν^(-α) with α=2.0. Verify the transition
        occurs and the slope matches.
        """
        # Short wavelengths well into the Wien/power-law regime
        wavelengths = np.array([5.0, 8.0, 12.0, 18.0])
        sed = fitter.greybody_model(
            wavelengths, amplitude=-34.0, temperature=35.0, beta=1.8, alpha=2.0
        )

        c_light = 299792458.0
        nu = c_light * 1e6 / wavelengths
        log_nu = np.log(nu)
        log_s = np.log(np.maximum(sed, 1e-100))

        slopes = np.diff(log_s) / np.diff(log_nu)
        expected_slope = -2.0  # power law index α

        # The shortest wavelengths should be firmly in the power-law regime
        assert slopes[0] == pytest.approx(expected_slope, abs=0.3), (
            f"Wien-side slope: got {slopes[0]:.2f}, expected {expected_slope:.1f}"
        )

    def test_amplitude_scaling(self, fitter):
        """
        Amplitude is log10 of a scaling factor. Increasing amplitude by 1
        should multiply the SED by 10.
        """
        wavelengths = np.logspace(1, 3, 100)

        sed_low = fitter.greybody_model(wavelengths, amplitude=-35.0, temperature=35.0)
        sed_high = fitter.greybody_model(wavelengths, amplitude=-34.0, temperature=35.0)

        ratio = sed_high / sed_low
        np.testing.assert_allclose(ratio, 10.0, rtol=1e-10,
            err_msg="Amplitude +1 should multiply SED by 10")

    def test_sed_positivity(self, fitter):
        """SED should be positive everywhere."""
        wavelengths = np.logspace(0, 4, 1000)  # 1 to 10000 μm
        sed = fitter.greybody_model(wavelengths, amplitude=-34.0, temperature=35.0)

        assert np.all(sed > 0), "SED has non-positive values"


# ---------------------------------------------------------------------------
# Test 8: L_IR integration
# ---------------------------------------------------------------------------


class TestLIRIntegration:
    """
    Verify the L_IR integration logic, independent of the D_L bug.

    We test the *integration* itself by checking that the integrated
    Jy·Hz scales correctly with amplitude and temperature.
    """

    def test_lir_scales_with_amplitude(self, fitter):
        """
        L_IR ∝ 10^amplitude, so doubling the log-amplitude by +0.3
        should double L_IR.
        """
        # Use z=0.01 where D_L bug is negligible
        L1, _ = fitter.calculate_LIR(
            amplitude=-34.0, temperature=35.0, beta=1.8, redshift=0.01
        )
        L2, _ = fitter.calculate_LIR(
            amplitude=-33.7, temperature=35.0, beta=1.8, redshift=0.01
        )

        ratio = L2 / L1
        assert ratio == pytest.approx(2.0, rel=0.01), (
            f"Amplitude +0.3 should double L_IR, got ratio={ratio:.3f}"
        )

    def test_lir_increases_with_temperature(self, fitter):
        """Hotter dust → more luminosity (monotonic)."""
        lir_values = []
        for temp in [20.0, 30.0, 40.0, 50.0]:
            L, _ = fitter.calculate_LIR(
                amplitude=-34.0, temperature=temp, beta=1.8, redshift=0.01
            )
            lir_values.append(L)

        for i in range(len(lir_values) - 1):
            assert lir_values[i + 1] > lir_values[i], (
                f"L_IR not increasing: T={[20,30,40,50][i]}K → L={lir_values[i]:.2e}, "
                f"T={[20,30,40,50][i+1]}K → L={lir_values[i+1]:.2e}"
            )

    def test_lir_at_z0_vs_direct_integration(self, fitter):
        """
        At z≈0, we can cross-check L_IR by direct integration.

        L_IR = 4π D_L² × ∫ S_ν dν  (in appropriate units)

        Compute the integral independently and compare.
        """
        amplitude = -34.0
        temperature = 35.0
        beta = 1.8
        redshift = 0.01

        # Direct integration (same as calculate_LIR internals)
        wavelength_range = np.logspace(np.log10(8), np.log10(1000), 2000)
        model_sed = fitter.greybody_model(wavelength_range, amplitude, temperature, beta)
        c_light = 299792458.0
        nu = c_light * 1e6 / wavelength_range
        dnu = np.abs(np.diff(nu))
        dnu = np.append(dnu[0], dnu)
        integral_jy_hz = np.sum(model_sed * dnu)

        # Convert to L_sun using same D_L
        D_L_mpc = fitter.luminosity_distance(redshift)
        D_L_m = D_L_mpc * 3.08568025e22
        L_direct = 4.0 * np.pi * D_L_m**2 * integral_jy_hz * 1e-26 / fitter.L_sun

        # Compare with calculate_LIR
        L_method, _ = fitter.calculate_LIR(amplitude, temperature, beta, redshift)

        assert L_direct == pytest.approx(L_method, rel=0.01), (
            f"Direct integration {L_direct:.3e} != calculate_LIR {L_method:.3e}"
        )


# ---------------------------------------------------------------------------
# Test 9: Luminosity distance — BUG #1
# ---------------------------------------------------------------------------


class TestLuminosityDistance:
    """
    Verify that GreybodyFitter now uses proper cosmological D_L.

    Previously used Hubble-law D_L = c*z/H0 (BUG #1, fixed).
    Now delegates to CosmologyCalculator (Planck18 via astropy).
    """

    @pytest.mark.parametrize("z", [0.1, 0.5, 1.0, 2.0, 4.0])
    def test_fitter_matches_cosmology_calculator(self, fitter, cosmo, z):
        """Fitter D_L should now match CosmologyCalculator exactly."""
        dl_fitter = fitter.luminosity_distance(z)
        dl_cosmo = cosmo.luminosity_distance(z)

        assert dl_fitter == pytest.approx(dl_cosmo, rel=1e-10), (
            f"z={z}: fitter D_L={dl_fitter:.2f} != cosmo D_L={dl_cosmo:.2f}"
        )

    def test_matches_astropy_directly(self, fitter):
        """Fitter D_L should match astropy Planck18 exactly."""
        from astropy.cosmology import Planck18

        for z in [0.1, 0.5, 1.0, 2.0, 4.0]:
            dl_fitter = fitter.luminosity_distance(z)
            dl_astropy = Planck18.luminosity_distance(z).value

            assert dl_fitter == pytest.approx(dl_astropy, rel=1e-10), (
                f"z={z}: fitter D_L={dl_fitter:.2f} != astropy {dl_astropy:.2f}"
            )

    def test_proper_cosmology_matches_astropy(self, cosmo):
        """CosmologyCalculator should match astropy exactly."""
        from astropy.cosmology import Planck18

        for z in [0.1, 0.5, 1.0, 2.0, 4.0]:
            dl_calc = cosmo.luminosity_distance(z)
            dl_astropy = Planck18.luminosity_distance(z).value  # Mpc

            assert dl_calc == pytest.approx(dl_astropy, rel=1e-10), (
                f"z={z}: CosmologyCalculator {dl_calc:.2f} != astropy {dl_astropy:.2f}"
            )

    def test_no_lir_impact_from_dl(self, fitter, cosmo):
        """
        Since fitter now uses proper D_L, there should be no D_L-induced
        L_IR error at any redshift.
        """
        for z in [0.5, 1.0, 2.0]:
            dl_fitter = fitter.luminosity_distance(z)
            dl_cosmo = cosmo.luminosity_distance(z)

            lir_ratio = (dl_fitter / dl_cosmo) ** 2
            pct_error = abs(1.0 - lir_ratio) * 100

            assert pct_error < 0.001, (
                f"z={z}: Residual D_L-induced L_IR error = {pct_error:.4f}%"
            )


# ---------------------------------------------------------------------------
# Test 9b: Frame mixing in calculate_LIR — BUG #2
# ---------------------------------------------------------------------------


class TestFrameMixing:
    """
    Verify that calculate_LIR uses correct observed-frame integration range.

    Previously integrated over rest-frame wavelengths (8-1000um) with T_obs (BUG #2, fixed).
    Now integrates over observed-frame wavelengths [8*(1+z), 1000*(1+z)]um with T_obs.
    """

    def test_frame_consistency_at_z0(self, fitter):
        """At z~0, observed and rest frames are the same."""
        amp, temp, beta = -34.0, 35.0, 1.8

        L_method, _ = fitter.calculate_LIR(amp, temp, beta, redshift=0.01)
        L_direct = self._integrate_lir_observed_frame(fitter, amp, temp, beta, z=0.01)

        assert L_method == pytest.approx(L_direct, rel=0.01)

    @pytest.mark.parametrize("z", [0.5, 1.0, 2.0, 3.0])
    def test_frame_consistency_at_high_z(self, fitter, z):
        """
        calculate_LIR should agree with independent observed-frame
        integration at all redshifts (< 0.5% from numerical differences).
        """
        amp, temp_obs, beta = -34.0, 35.0, 1.8

        L_method, _ = fitter.calculate_LIR(amp, temp_obs, beta, redshift=z)
        L_direct = self._integrate_lir_observed_frame(fitter, amp, temp_obs, beta, z)

        ratio = L_method / L_direct
        assert ratio == pytest.approx(1.0, abs=0.005), (
            f"z={z}: calculate_LIR / direct integration = {ratio:.4f}, expected ~1.0"
        )

    def test_lir_increases_with_redshift_for_fixed_model(self, fitter):
        """
        For fixed (A, T_obs, beta), L_IR should increase with z because
        D_L grows and the integration range shifts to capture more flux.
        """
        amp, temp_obs, beta = -34.0, 35.0, 1.8

        lir_values = []
        for z in [0.1, 0.5, 1.0, 2.0]:
            L, _ = fitter.calculate_LIR(amp, temp_obs, beta, z)
            lir_values.append(L)

        for i in range(len(lir_values) - 1):
            assert lir_values[i + 1] > lir_values[i]

    @staticmethod
    def _integrate_lir_observed_frame(fitter, amplitude, temperature, beta, z):
        """
        Correct L_IR: integrate observed-frame model over the observed-frame
        wavelength range corresponding to rest-frame 8–1000μm.
        """
        wav_obs_min = 8.0 * (1 + z)
        wav_obs_max = 1000.0 * (1 + z)

        wavelengths = np.logspace(np.log10(wav_obs_min), np.log10(wav_obs_max), 2000)
        sed = fitter.greybody_model(wavelengths, amplitude, temperature, beta)

        c_light = 299792458.0
        nu = c_light * 1e6 / wavelengths
        dnu = np.abs(np.diff(nu))
        dnu = np.append(dnu[0], dnu)
        integral_jy_hz = np.sum(sed * dnu)

        D_L_mpc = fitter.luminosity_distance(z)
        D_L_m = D_L_mpc * 3.08568025e22
        L_solar = 4.0 * np.pi * D_L_m**2 * integral_jy_hz * 1e-26 / fitter.L_sun

        return L_solar


# ---------------------------------------------------------------------------
# Test 10: Flux ↔ luminosity round-trip (CosmologyCalculator)
# ---------------------------------------------------------------------------


class TestFluxLuminosityRoundTrip:
    """
    Verify that flux_to_luminosity and luminosity_to_flux are inverses.
    Uses CosmologyCalculator (proper cosmology).
    """

    @pytest.mark.parametrize("z", [0.1, 0.5, 1.0, 2.0])
    def test_round_trip(self, cosmo, z):
        """Convert flux → luminosity → flux and check recovery."""
        flux_jy = 0.01  # 10 mJy
        rest_wavelength = 250.0  # μm

        luminosity = cosmo.flux_to_luminosity(flux_jy, z, rest_wavelength)
        recovered_flux = cosmo.luminosity_to_flux(luminosity, z, rest_wavelength)

        assert recovered_flux == pytest.approx(flux_jy, rel=1e-6), (
            f"z={z}: Round-trip failed: {flux_jy} → {luminosity:.2e} L_sun → {recovered_flux:.6e} Jy"
        )

    def test_luminosity_increases_with_distance(self, cosmo):
        """Same flux at higher z → higher luminosity (more distant)."""
        flux_jy = 0.01
        rest_wavelength = 250.0

        lum_values = []
        for z in [0.1, 0.5, 1.0, 2.0, 4.0]:
            lum = cosmo.flux_to_luminosity(flux_jy, z, rest_wavelength)
            lum_values.append(lum)

        for i in range(len(lum_values) - 1):
            assert lum_values[i + 1] > lum_values[i], (
                f"Luminosity not increasing with z"
            )

    def test_flux_decreases_with_distance(self, cosmo):
        """Same luminosity at higher z → fainter flux."""
        luminosity = 1e11  # L_sun (LIRG)
        rest_wavelength = 250.0

        flux_values = []
        for z in [0.1, 0.5, 1.0, 2.0]:
            flux = cosmo.luminosity_to_flux(luminosity, z, rest_wavelength)
            flux_values.append(flux)

        for i in range(len(flux_values) - 1):
            assert flux_values[i + 1] < flux_values[i], (
                f"Flux not decreasing with z"
            )


# ---------------------------------------------------------------------------
# Test 11: SFR from L_IR
# ---------------------------------------------------------------------------


class TestSFRConversion:
    """
    Verify the Kennicutt (1998) SFR conversion:
    SFR [M_sun/yr] = L_IR [L_sun] / 1e10
    """

    def test_kennicutt_relation(self):
        """Check the conversion factor."""
        # LIRG: L_IR = 10^11 L_sun → SFR = 10 M_sun/yr
        assert 1e11 / 1e10 == pytest.approx(10.0)

        # ULIRG: L_IR = 10^12 L_sun → SFR = 100 M_sun/yr
        assert 1e12 / 1e10 == pytest.approx(100.0)

        # Milky Way-like: L_IR = 10^10 L_sun → SFR = 1 M_sun/yr
        assert 1e10 / 1e10 == pytest.approx(1.0)

    def test_sfr_from_greybody_fit(self, fitter):
        """
        Compute L_IR from a greybody fit, convert to SFR,
        and verify it's in a physically reasonable range.

        Note: At z=1, both the D_L and frame-mixing bugs affect L_IR,
        so we test at z≈0 where the bugs are negligible, then separately
        check that at z=1 the result is inflated (documenting the bug).
        """
        amplitude = -34.0
        temperature = 30.0  # K
        beta = 1.8

        # At z≈0, bugs are negligible — SFR should be physically plausible
        L_IR_local, _ = fitter.calculate_LIR(amplitude, temperature, beta, redshift=0.01)
        sfr_local = L_IR_local / 1e10
        assert 0.001 < sfr_local < 10, (
            f"Local SFR = {sfr_local:.3f} M_sun/yr from L_IR = {L_IR_local:.2e} — "
            f"outside plausible range for these parameters"
        )

        # At z=1, L_IR is inflated by bugs. The D_L bug underestimates D_L
        # (lowering L_IR) while frame mixing overestimates L_IR. Document
        # that the result is at least positive and finite.
        L_IR_z1, _ = fitter.calculate_LIR(amplitude, temperature, beta, redshift=1.0)
        assert L_IR_z1 > 0 and np.isfinite(L_IR_z1)


# ---------------------------------------------------------------------------
# Test 12: Combined D_L + frame bugs — total L_IR error budget
# ---------------------------------------------------------------------------


class TestCombinedLIRErrors:
    """
    Verify that both bugs are fixed: L_IR computed by calculate_LIR
    should now match an independent correct calculation.
    """

    def test_lir_accuracy_vs_independent_calculation(self, fitter, cosmo):
        """
        Compare calculate_LIR against an independent implementation
        using proper D_L and observed-frame integration.
        Should agree to < 0.5% (numerical integration differences).
        """
        amp, temp_obs, beta = -34.0, 35.0, 1.8

        for z in [0.1, 0.5, 1.0, 2.0, 3.0]:
            # calculate_LIR (now fixed)
            L_method, _ = fitter.calculate_LIR(amp, temp_obs, beta, z)

            # Independent calculation with proper D_L and observed-frame range
            wav_min = 8.0 * (1 + z)
            wav_max = 1000.0 * (1 + z)
            wavelengths = np.logspace(np.log10(wav_min), np.log10(wav_max), 2000)
            sed = fitter.greybody_model(wavelengths, amp, temp_obs, beta)
            c_light = 299792458.0
            nu = c_light * 1e6 / wavelengths
            dnu = np.abs(np.diff(nu))
            dnu = np.append(dnu[0], dnu)
            integral = np.sum(sed * dnu)
            D_L_m = cosmo.luminosity_distance(z) * 3.08568025e22
            L_independent = 4.0 * np.pi * D_L_m**2 * integral * 1e-26 / fitter.L_sun

            ratio = L_method / L_independent
            assert ratio == pytest.approx(1.0, abs=0.005), (
                f"z={z}: calculate_LIR / independent = {ratio:.4f}"
            )

    def test_fitter_dl_equals_cosmo_dl(self, fitter, cosmo):
        """Fitter and CosmologyCalculator should return identical D_L."""
        for z in [0.5, 1.0, 2.0, 3.0]:
            dl_fitter = fitter.luminosity_distance(z)
            dl_cosmo = cosmo.luminosity_distance(z)

            assert dl_fitter == pytest.approx(dl_cosmo, rel=1e-10), (
                f"z={z}: D_L mismatch: fitter={dl_fitter:.2f}, cosmo={dl_cosmo:.2f}"
            )
