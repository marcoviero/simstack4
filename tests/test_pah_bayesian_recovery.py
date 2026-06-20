"""
Injection-recovery tests for fit_bayesian_forward_model.

Strategy: PAHModel.simulate_pah_data() generates synthetic flux ratios from a
known forward model (known α_true, known baseline).  We then run the Bayesian
fitter on that synthetic data and verify that the posterior median is close to
the ground truth.

Three escalating tiers:

  1. Simulator consistency  — no MCMC; check simulate_pah_data output shape,
     positivity, and that the fitter initialises without error.

  2. MAP recovery — near-low noise (SNR ~100), chain long enough for convergence.
     Confirms index arithmetic and parameter layout for all three modes.
     Tolerance: |recovered − truth| / truth < 15%.

  3. Group differentiation — data where groups have very different amplitudes;
     verify modes 2/3 preserve the amplitude ranking across bins.

Bin design — property bins with wide z coverage
------------------------------------------------
The Bayesian fitter bins sources by a PROPERTY (σ_SFR, stellar mass) — not by
redshift.  Within each property bin, sources span a wide z range that covers
both PAH peaks visible through the MIPS 24 μm bandpass:

    Feature      λ_rest   z at band centre (24 μm)   T_g peak
    7.7 μm       7.7      z ≈ 2.12                   T_g[0] ≈ 0.43
    8.6 μm       8.6      z ≈ 1.79                   T_g[0] ≈ 0.43 (blended)
    12.7 μm     12.7      z ≈ 0.89                   T_g[1] ≈ 0.14

With sources spanning z = 0.5–2.5 within each property bin, T_g(z) rises to a
peak at z ≈ 0.89 (12.7 μm) and again at z ≈ 1.9–2.1 (7.7+8.6 μm complex).
This peaked T_g variation cannot be absorbed by the degree-2 polynomial
baseline, so the fitter can separate the PAH excess from the SED continuum.

Role of dithering: three stacking runs with slightly offset narrow z bins
(e.g. [0.50–0.75], [0.625–0.875], [0.75–1.0], …) generate many flux
measurements at closely spaced z within each σ_SFR group.  When those
measurements are combined into one DataFrame, each property bin effectively
has sources uniformly distributed across z = 0.5–2.5 — this is what
simulate_pah_data replicates with a wide BIN_Z_RANGES entry per property bin.

Modes 2/3 need longer chains than mode 1 (more parameters, harder geometry).
The MCMC_* dicts are tuned per mode; total suite runs in ≲ 3 min.
"""

import numpy as np
import pytest

from simstack4.pah_model import PAHModel


# ---------------------------------------------------------------------------
# Shared fixtures and constants
# ---------------------------------------------------------------------------

# Two PAH feature groups visible across z ∈ [0.5, 2.5]:
#   Group 0: 7.7 + 8.6 μm  → T_g ≈ 0.43 (peaks at z ≈ 1.9–2.1)
#   Group 1: 12.7 μm       → T_g ≈ 0.14 (peaks at z ≈ 0.89)
FEATURE_GROUPS = [[1, 2], [4]]

# Three property bins (low / mid / high σ_SFR), each with sources spanning
# z = 0.5–2.5.  The wide within-bin z coverage creates large T_g variation,
# making the PAH signal separable from the polynomial baseline.
# This is the dithered-stacking scenario: combining many narrow stacking
# z-bins into one property-grouped dataset.
BIN_Z_RANGES = [(0.5, 2.5)] * 3
M = len(BIN_Z_RANGES)

# True PAH amplitudes — constant across property bins so β_z ≈ 0
ALPHA_MODE1  = np.array([1.5, 2.5, 3.5])               # (M,)  — increasing with bin
ALPHA_MODE23 = np.array([[2.0] * M, [0.5] * M])        # (G, M): group 0 > group 1

# MCMC settings tuned per mode.  The wide-z bins give informative likelihoods
# per MCMC step so τ stays manageable (τ ≈ 30–60 for all modes with M=3).
MCMC_MODE1  = dict(n_steps=600,  n_burn=200, progress=False, verbose=False)
MCMC_MODE2  = dict(n_steps=1200, n_burn=400, progress=False, verbose=False)
MCMC_MODE3  = dict(n_steps=2000, n_burn=600, progress=False, verbose=False)

ALPHA_TOL = 0.15   # |recovered − truth| / truth tolerance

# Flat power-law baseline (n=0, flux=0.5) — scale the Bayesian fitter expects.
# The frozen fitter's internal polynomial can trivially absorb a constant baseline.
BASELINE_COEFFS = np.tile([0.0, np.log10(0.5)], (M, 1))


@pytest.fixture(scope="module")
def pah():
    return PAHModel()


def _fit_and_check(pah, sim, group_amplitudes, independent_betas):
    """Fit sim data and assert alpha_per_bin recovery within ALPHA_TOL."""
    if not group_amplitudes:
        mcmc = MCMC_MODE1
    elif independent_betas:
        mcmc = MCMC_MODE3
    else:
        mcmc = MCMC_MODE2
    result = pah.fit_bayesian_forward_model(
        sim["df"],
        group_col="log_sigma_sfr",
        flux_col="f24_to_fpeak",
        feature_groups=FEATURE_GROUPS,
        bin_edges=sim["bin_edges"],
        group_amplitudes=group_amplitudes,
        independent_betas=independent_betas,
        **mcmc,
    )
    assert result is not None, "fitter returned None"

    alpha_true = sim["true_params"]["alpha"]
    alpha_rec  = result["alpha_per_bin"]
    assert alpha_rec.shape == alpha_true.shape, (
        f"alpha shape mismatch: got {alpha_rec.shape}, expected {alpha_true.shape}"
    )

    frac_err = np.abs(alpha_rec - alpha_true) / alpha_true
    bad = frac_err > ALPHA_TOL
    assert not bad.any(), (
        f"alpha recovery outside {ALPHA_TOL*100:.0f}% tolerance:\n"
        f"  truth:     {alpha_true}\n"
        f"  recovered: {alpha_rec}\n"
        f"  frac_err:  {np.round(frac_err, 3)}"
    )
    return result


# ---------------------------------------------------------------------------
# Tier 1 — simulator consistency (no MCMC, runs in seconds)
# ---------------------------------------------------------------------------

class TestSimulatorConsistency:
    """Verify simulate_pah_data without running the sampler."""

    def test_single_alpha_positive_flux(self, pah):
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, ALPHA_MODE1,
            feature_groups=FEATURE_GROUPS, noise_level=0.0,
        )
        assert (sim["df"]["f24_to_fpeak"] > 0).all()

    def test_group_mode_shape(self, pah):
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, ALPHA_MODE23, feature_groups=FEATURE_GROUPS,
        )
        assert sim["true_params"]["group_mode"] is True
        assert sim["true_params"]["alpha"].shape == (2, M)

    def test_bin_edges_count(self, pah):
        sim = pah.simulate_pah_data(BIN_Z_RANGES, ALPHA_MODE1,
                                    feature_groups=FEATURE_GROUPS)
        assert len(sim["bin_edges"]) == M

    def test_required_columns_present(self, pah):
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, ALPHA_MODE1,
            feature_groups=FEATURE_GROUPS,
            property_col_values={"log_l_ir": [11.5, 12.0, 12.3]},
        )
        for col in ["z", "f24_to_fpeak", "f24_to_fpeak_err",
                    "log_sigma_sfr", "log_l_ir"]:
            assert col in sim["df"].columns

    def test_fitter_initialises_without_error(self, pah):
        """Run a minimal chain (50 steps) to confirm the sampler starts cleanly."""
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, ALPHA_MODE1,
            feature_groups=FEATURE_GROUPS, noise_level=0.02, n_obs_per_bin=20,
        )
        result = pah.fit_bayesian_forward_model(
            sim["df"],
            group_col="log_sigma_sfr", flux_col="f24_to_fpeak",
            feature_groups=FEATURE_GROUPS, bin_edges=sim["bin_edges"],
            n_steps=60, n_burn=20, progress=False, verbose=False,
        )
        assert result is not None
        assert np.isfinite(result["acceptance_fraction"])


# ---------------------------------------------------------------------------
# Tier 2 — MAP recovery (MCMC, all three modes)
# ---------------------------------------------------------------------------

class TestMode1Recovery:
    """Single PAH amplitude α_m per bin, global group ratios r_g."""

    def test_varying_amplitude(self, pah):
        """α increases across property bins — exercises β evolution recovery."""
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, ALPHA_MODE1,
            feature_groups=FEATURE_GROUPS,
            r_true=np.array([1.0, 0.6]),
            baseline_coeffs=BASELINE_COEFFS,
            noise_level=0.01, n_obs_per_bin=50,
        )
        _fit_and_check(pah, sim, group_amplitudes=False, independent_betas=False)

    def test_equal_amplitudes(self, pah):
        """All bins share the same α — β ≈ 0, δ_m ≈ 0."""
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, np.full(M, 2.0),
            feature_groups=FEATURE_GROUPS,
            baseline_coeffs=BASELINE_COEFFS,
            noise_level=0.01, n_obs_per_bin=50,
        )
        _fit_and_check(pah, sim, group_amplitudes=False, independent_betas=False)


class TestMode2Recovery:
    """Per-group amplitude α_gm, shared evolution slope β."""

    def test_group_amplitudes_recovered(self, pah):
        """Groups differ by 4×; wide z coverage separates the two PAH peaks."""
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, ALPHA_MODE23,
            feature_groups=FEATURE_GROUPS,
            baseline_coeffs=BASELINE_COEFFS,
            noise_level=0.01, n_obs_per_bin=50,
        )
        _fit_and_check(pah, sim, group_amplitudes=True, independent_betas=False)


class TestMode3Recovery:
    """Per-group amplitude α_gm AND per-group evolution slope β_gk."""

    def test_group_amplitudes_independent_betas(self, pah):
        """Mode 3: each PAH feature group gets its own β slope."""
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, ALPHA_MODE23,
            feature_groups=FEATURE_GROUPS,
            baseline_coeffs=BASELINE_COEFFS,
            noise_level=0.01, n_obs_per_bin=50,
        )
        _fit_and_check(pah, sim, group_amplitudes=True, independent_betas=True)


# ---------------------------------------------------------------------------
# Tier 3 — group differentiation
# ---------------------------------------------------------------------------

class TestGroupDifferentiation:
    """
    Modes 2/3 should correctly separate groups with different amplitudes.
    This is the unique claim of per-group fitting that mode 1 cannot make.
    We use a 10× amplitude ratio between groups so the ordering is clear
    even with imperfect chain convergence.
    """

    @pytest.mark.parametrize("independent_betas", [False, True])
    def test_strong_weak_ordering_preserved(self, pah, independent_betas):
        alpha_true = np.array([
            [3.0] * M,   # 7.7+8.6 μm — strong in all property bins
            [0.3] * M,   # 12.7 μm    — 10× weaker in all property bins
        ])
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, alpha_true,
            feature_groups=FEATURE_GROUPS,
            baseline_coeffs=BASELINE_COEFFS,
            noise_level=0.01, n_obs_per_bin=50,
        )
        mcmc = MCMC_MODE3 if independent_betas else MCMC_MODE2
        result = pah.fit_bayesian_forward_model(
            sim["df"],
            group_col="log_sigma_sfr", flux_col="f24_to_fpeak",
            feature_groups=FEATURE_GROUPS, bin_edges=sim["bin_edges"],
            group_amplitudes=True, independent_betas=independent_betas,
            **mcmc,
        )
        assert result is not None
        alpha_rec = result["alpha_per_bin"]   # (G, M)
        # Group 0 (7.7+8.6 μm) must exceed group 1 (12.7 μm) in every property bin
        for m in range(M):
            assert alpha_rec[0, m] > alpha_rec[1, m], (
                f"bin {m}: group 0 (7.7+8.6) should exceed group 1 (12.7); "
                f"got {alpha_rec[0, m]:.3f} vs {alpha_rec[1, m]:.3f}"
            )

    def test_mode1_runs_on_group_data(self, pah):
        """
        Mode 1 should run without error on data generated from per-group α.
        It won't separate groups, but it should converge to some positive α_m.
        """
        sim = pah.simulate_pah_data(
            BIN_Z_RANGES, np.array([[3.0] * M, [0.3] * M]),
            feature_groups=FEATURE_GROUPS,
            baseline_coeffs=BASELINE_COEFFS,
            noise_level=0.02, n_obs_per_bin=50,
        )
        result = pah.fit_bayesian_forward_model(
            sim["df"],
            group_col="log_sigma_sfr", flux_col="f24_to_fpeak",
            feature_groups=FEATURE_GROUPS, bin_edges=sim["bin_edges"],
            group_amplitudes=False,
            **MCMC_MODE1,
        )
        assert result is not None
        assert result["alpha_per_bin"].shape == (M,)
        assert np.all(result["alpha_per_bin"] > 0)
