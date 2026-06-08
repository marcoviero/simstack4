"""
Per-bin error estimation tests for simstack4.

Tests validate the per_bin bootstrap method at the linear algebra level,
using the same synthetic layer infrastructure as test_stacking_recovery.py.

Per-bin method: split one population at a time, hold others fixed.
All-bins method: split every population simultaneously.
Both give flux estimates from the full (unsplit) solve.
"""

import numpy as np
import pytest
from scipy import linalg
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Helpers (same as test_stacking_recovery.py)
# ---------------------------------------------------------------------------


def gaussian_psf_layer(
    map_shape, source_positions, fwhm_pix, mean_subtract=True
):
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    delta_map = np.zeros(map_shape, dtype=np.float64)
    for y, x in source_positions:
        if 0 <= y < map_shape[0] and 0 <= x < map_shape[1]:
            delta_map[y, x] += 1.0
    convolved = gaussian_filter(delta_map, sigma=sigma, mode="constant")
    if mean_subtract:
        convolved -= np.mean(convolved)
    return convolved.ravel()


def solve_for_fluxes(layer_matrix, observed):
    result = linalg.lstsq(layer_matrix.T, observed)
    coeffs = result[0]
    residual = observed - layer_matrix.T @ coeffs
    n_pix = len(observed)
    n_layers = layer_matrix.shape[0]
    dof = n_pix - n_layers
    mse = np.sum(residual**2) / dof if dof > 0 else np.inf
    try:
        cov = np.linalg.inv(layer_matrix @ layer_matrix.T) * mse
        formal_errors = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        formal_errors = np.full(n_layers, np.inf)
    return coeffs, formal_errors


# ---------------------------------------------------------------------------
# Per-bin error estimation (standalone, mirrors algorithm.py logic)
# ---------------------------------------------------------------------------


def run_per_bin_bootstrap(
    layer_matrix, observed, source_positions_per_pop, map_shape, fwhm_pix,
    n_iterations=200, split_fraction=0.5, seed=42
):
    """
    Per-bin bootstrap at linear algebra level.

    Parameters
    ----------
    layer_matrix : (N_pop, N_pix) — full layers (all sources)
    observed : (N_pix,) — observed map
    source_positions_per_pop : list of list of (y,x) — positions per population
    map_shape : (ny, nx)
    fwhm_pix : PSF FWHM in pixels
    n_iterations : number of bootstrap iterations per population
    split_fraction : fraction of sources going to "A"
    seed : random seed base

    Returns
    -------
    fluxes : (N_pop,) — flux estimates from full solve
    per_bin_errors : (N_pop,) — std of (flux_A_k + flux_B_k) per pop
    formal_errors : (N_pop,) — lstsq formal errors from full solve
    """
    n_pop = layer_matrix.shape[0]

    # Full solve
    fluxes, formal_errors = solve_for_fluxes(layer_matrix, observed)

    # Per-bin error estimation
    per_bin_errors = np.zeros(n_pop)

    for k in range(n_pop):
        positions_k = source_positions_per_pop[k]
        n_sources = len(positions_k)

        if n_sources < 2:
            per_bin_errors[k] = np.inf
            continue

        n_A = int(n_sources * split_fraction)
        if n_A == 0 or n_A == n_sources:
            per_bin_errors[k] = np.inf
            continue

        flux_k_samples = []

        for iteration in range(n_iterations):
            rng = np.random.RandomState(seed + k * n_iterations + iteration)

            # Shuffle and split population k's sources
            shuffled_idx = rng.permutation(n_sources)
            positions_A = [positions_k[i] for i in shuffled_idx[:n_A]]
            positions_B = [positions_k[i] for i in shuffled_idx[n_A:]]

            # Build layer matrix: unsplit layers for all pops except k,
            # A and B layers for pop k
            layers = []
            for j in range(n_pop):
                if j == k:
                    layer_A = gaussian_psf_layer(
                        map_shape, positions_A, fwhm_pix, mean_subtract=True
                    )
                    layer_B = gaussian_psf_layer(
                        map_shape, positions_B, fwhm_pix, mean_subtract=True
                    )
                    layers.append(layer_A)
                    layers.append(layer_B)
                else:
                    layers.append(layer_matrix[j])

            split_matrix = np.array(layers)

            # Solve
            split_fluxes, _ = solve_for_fluxes(split_matrix, observed)

            # Half-difference for pop k: captures noise asymmetry, avoids A+B cancellation
            flux_A_k = split_fluxes[k]
            flux_B_k = split_fluxes[k + 1]
            flux_k_samples.append((flux_A_k - flux_B_k) / 2.0)

        per_bin_errors[k] = np.std(flux_k_samples, ddof=1)

    return fluxes, per_bin_errors, formal_errors


def run_all_bins_bootstrap(
    layer_matrix, observed, source_positions_per_pop, map_shape, fwhm_pix,
    n_iterations=200, split_fraction=0.5, seed=42
):
    """
    All-bins bootstrap at linear algebra level (for comparison).

    Returns
    -------
    fluxes : (N_pop,) — from full solve
    all_bins_errors : (N_pop,) — std of (flux_A_k + flux_B_k) across iterations
    """
    n_pop = layer_matrix.shape[0]
    fluxes, formal_errors = solve_for_fluxes(layer_matrix, observed)

    bootstrap_samples = []

    for iteration in range(n_iterations):
        rng = np.random.RandomState(seed + iteration)

        # Split ALL populations
        layers = []
        for k in range(n_pop):
            positions_k = source_positions_per_pop[k]
            n_sources = len(positions_k)
            n_A = int(n_sources * split_fraction)

            shuffled_idx = rng.permutation(n_sources)
            positions_A = [positions_k[i] for i in shuffled_idx[:n_A]]
            positions_B = [positions_k[i] for i in shuffled_idx[n_A:]]

            layer_A = gaussian_psf_layer(
                map_shape, positions_A, fwhm_pix, mean_subtract=True
            )
            layer_B = gaussian_psf_layer(
                map_shape, positions_B, fwhm_pix, mean_subtract=True
            )
            layers.append(layer_A)
            layers.append(layer_B)

        split_matrix = np.array(layers)
        split_fluxes, _ = solve_for_fluxes(split_matrix, observed)

        # Half-difference per population: (x_A - x_B) / 2.
        # Layers are interleaved [A0, B0, A1, B1, ...], so even-indexed
        # elements are the A fluxes and odd-indexed are the B fluxes.
        total_fluxes = (split_fluxes[0::2] - split_fluxes[1::2]) / 2.0
        bootstrap_samples.append(total_fluxes)

    all_bins_errors = np.std(bootstrap_samples, axis=0, ddof=1)
    return fluxes, all_bins_errors


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPerBinBasics:
    """Basic per-bin error estimation sanity checks."""

    def _make_two_pop_scenario(self, noise_std=0.001):
        """Two well-separated populations on a 64x64 map."""
        map_shape = (64, 64)
        fwhm = 3.0
        rng = np.random.default_rng(12345)

        # Pop A: 20 sources in top-left quadrant
        positions_a = [(rng.integers(5, 28), rng.integers(5, 28)) for _ in range(20)]
        # Pop B: 20 sources in bottom-right quadrant
        positions_b = [(rng.integers(36, 59), rng.integers(36, 59)) for _ in range(20)]

        layer_a = gaussian_psf_layer(map_shape, positions_a, fwhm)
        layer_b = gaussian_psf_layer(map_shape, positions_b, fwhm)
        layer_matrix = np.array([layer_a, layer_b])

        true_fluxes = np.array([0.005, 0.010])
        observed = layer_matrix.T @ true_fluxes
        observed += rng.normal(0, noise_std, len(observed))

        return (layer_matrix, observed, [positions_a, positions_b],
                map_shape, fwhm, true_fluxes)

    def test_flux_recovery(self):
        """Per-bin should recover injected fluxes (from full solve)."""
        (layer_matrix, observed, positions, map_shape,
         fwhm, true_fluxes) = self._make_two_pop_scenario()

        fluxes, per_bin_errs, formal_errs = run_per_bin_bootstrap(
            layer_matrix, observed, positions, map_shape, fwhm,
            n_iterations=100
        )

        for k in range(len(true_fluxes)):
            assert fluxes[k] == pytest.approx(true_fluxes[k], abs=3 * formal_errs[k]), (
                f"Pop {k}: recovered {fluxes[k]:.6f} != true {true_fluxes[k]:.6f}"
            )

    def test_per_bin_errors_positive(self):
        """Per-bin errors should be positive and finite."""
        (layer_matrix, observed, positions, map_shape,
         fwhm, true_fluxes) = self._make_two_pop_scenario()

        _, per_bin_errs, _ = run_per_bin_bootstrap(
            layer_matrix, observed, positions, map_shape, fwhm,
            n_iterations=100
        )

        for k, err in enumerate(per_bin_errs):
            assert err > 0, f"Pop {k}: error not positive ({err})"
            assert np.isfinite(err), f"Pop {k}: error not finite ({err})"

    def test_per_bin_same_flux_as_all_bins(self):
        """Both methods use the full solve for fluxes — they should agree exactly."""
        (layer_matrix, observed, positions, map_shape,
         fwhm, true_fluxes) = self._make_two_pop_scenario()

        fluxes_pb, _, _ = run_per_bin_bootstrap(
            layer_matrix, observed, positions, map_shape, fwhm,
            n_iterations=50
        )
        fluxes_ab, _ = run_all_bins_bootstrap(
            layer_matrix, observed, positions, map_shape, fwhm,
            n_iterations=50
        )

        np.testing.assert_array_equal(fluxes_pb, fluxes_ab,
            err_msg="Both methods should return identical flux estimates")


class TestPerBinVsAllBins:
    """Compare per-bin and all-bins error estimates."""

    def test_errors_comparable_for_separated_pops(self):
        """
        For well-separated populations, both per_bin and all_bins errors
        should be positive and in the same order of magnitude.

        With the diff estimator, both methods give errors comparable to the
        formal OLS error.  Per-bin holds other populations fixed so it
        captures isolated variance; all-bins splits everything simultaneously
        so it also captures cross-population coupling.  For separated pops
        the coupling is near-zero, so the two should be within 5× of each other.
        """
        map_shape = (64, 64)
        fwhm = 3.0
        rng = np.random.default_rng(999)

        positions_a = [(rng.integers(5, 25), rng.integers(5, 25)) for _ in range(30)]
        positions_b = [(rng.integers(40, 60), rng.integers(40, 60)) for _ in range(30)]

        layer_a = gaussian_psf_layer(map_shape, positions_a, fwhm)
        layer_b = gaussian_psf_layer(map_shape, positions_b, fwhm)
        layer_matrix = np.array([layer_a, layer_b])

        true_fluxes = np.array([0.01, 0.02])
        observed = layer_matrix.T @ true_fluxes
        observed += rng.normal(0, 0.001, len(observed))

        _, per_bin_errs, _ = run_per_bin_bootstrap(
            layer_matrix, observed, [positions_a, positions_b],
            map_shape, fwhm, n_iterations=200
        )
        _, all_bins_errs = run_all_bins_bootstrap(
            layer_matrix, observed, [positions_a, positions_b],
            map_shape, fwhm, n_iterations=200
        )

        for k in range(2):
            assert per_bin_errs[k] > 0
            assert all_bins_errs[k] > 0
            ratio = per_bin_errs[k] / all_bins_errs[k]
            assert 0.2 < ratio < 5.0, (
                f"Pop {k}: per_bin ({per_bin_errs[k]:.2e}) and "
                f"all_bins ({all_bins_errs[k]:.2e}) differ by more than 5× "
                f"for separated populations (ratio={ratio:.2f})"
            )

    def test_per_bin_errors_scale_with_noise(self):
        """Doubling noise should roughly double per-bin errors."""
        map_shape = (64, 64)
        fwhm = 3.0
        rng = np.random.default_rng(42)

        positions = [(rng.integers(5, 59), rng.integers(5, 59)) for _ in range(30)]
        layer = gaussian_psf_layer(map_shape, positions, fwhm)
        layer_matrix = np.array([layer])

        true_flux = np.array([0.01])

        errs = []
        for noise_std in [0.001, 0.002]:
            rng2 = np.random.default_rng(42)
            observed = layer_matrix.T @ true_flux
            observed += rng2.normal(0, noise_std, len(observed))

            _, per_bin_errs, _ = run_per_bin_bootstrap(
                layer_matrix, observed, [positions], map_shape, fwhm,
                n_iterations=200
            )
            errs.append(per_bin_errs[0])

        ratio = errs[1] / errs[0]
        assert 1.5 < ratio < 3.0, (
            f"Error ratio for 2x noise: {ratio:.2f}, expected ~2"
        )

    def test_per_bin_errors_decrease_with_more_sources(self):
        """
        More sources per population → smaller per-bin error.

        Use 2 populations so the A/B split for one pop has the other
        pop providing constraint (avoids single-pop degeneracy).
        """
        map_shape = (64, 64)
        fwhm = 3.0

        # Fixed second population for context
        rng_b = np.random.default_rng(999)
        positions_b = [
            (rng_b.integers(40, 60), rng_b.integers(40, 60))
            for _ in range(30)
        ]
        layer_b = gaussian_psf_layer(map_shape, positions_b, fwhm)

        errs_pop_a = []
        for n_sources in [10, 50]:
            rng = np.random.default_rng(123)
            positions_a = [
                (rng.integers(5, 30), rng.integers(5, 30))
                for _ in range(n_sources)
            ]
            layer_a = gaussian_psf_layer(map_shape, positions_a, fwhm)
            layer_matrix = np.array([layer_a, layer_b])

            true_fluxes = np.array([0.01, 0.02])
            rng_noise = np.random.default_rng(42)
            observed = layer_matrix.T @ true_fluxes
            observed += rng_noise.normal(0, 0.001, len(observed))

            _, per_bin_errs, _ = run_per_bin_bootstrap(
                layer_matrix, observed, [positions_a, positions_b],
                map_shape, fwhm, n_iterations=200
            )
            errs_pop_a.append(per_bin_errs[0])

        assert errs_pop_a[1] < errs_pop_a[0], (
            f"Error with 50 sources ({errs_pop_a[1]:.4e}) should be < "
            f"error with 10 sources ({errs_pop_a[0]:.4e})"
        )


class TestPerBinMultiPopulation:
    """Per-bin with multiple overlapping populations."""

    def test_three_populations_overlapping(self):
        """
        Three populations with partial spatial overlap.
        Per-bin should still give finite, positive errors.
        """
        map_shape = (64, 64)
        fwhm = 5.0  # Larger beam → more overlap
        rng = np.random.default_rng(777)

        positions = [
            [(rng.integers(10, 50), rng.integers(10, 50)) for _ in range(20)],
            [(rng.integers(15, 55), rng.integers(15, 55)) for _ in range(20)],
            [(rng.integers(20, 60), rng.integers(20, 60)) for _ in range(20)],
        ]

        layers = [
            gaussian_psf_layer(map_shape, pos, fwhm) for pos in positions
        ]
        layer_matrix = np.array(layers)

        true_fluxes = np.array([0.005, 0.010, 0.015])
        observed = layer_matrix.T @ true_fluxes
        observed += rng.normal(0, 0.001, len(observed))

        fluxes, per_bin_errs, formal_errs = run_per_bin_bootstrap(
            layer_matrix, observed, positions, map_shape, fwhm,
            n_iterations=100
        )

        for k in range(3):
            assert per_bin_errs[k] > 0, f"Pop {k}: error not positive"
            assert np.isfinite(per_bin_errs[k]), f"Pop {k}: error not finite"
            # Flux should be recovered within 5σ
            assert abs(fluxes[k] - true_fluxes[k]) < 5 * formal_errs[k], (
                f"Pop {k}: recovered {fluxes[k]:.6f} far from true {true_fluxes[k]:.6f}"
            )

    def test_per_bin_isolates_variance(self):
        """
        Per-bin for population k should NOT be affected by the source
        count of a different population j.

        Set up: pop A (20 sources, fixed), pop B (varies: 10 vs 40 sources).
        Per-bin error for pop A should be similar regardless of pop B's size.
        """
        map_shape = (64, 64)
        fwhm = 3.0

        rng = np.random.default_rng(555)
        positions_a = [(rng.integers(5, 25), rng.integers(5, 25)) for _ in range(20)]

        errors_a = []
        for n_b in [10, 40]:
            rng_b = np.random.default_rng(666)
            positions_b = [
                (rng_b.integers(40, 60), rng_b.integers(40, 60))
                for _ in range(n_b)
            ]

            layer_a = gaussian_psf_layer(map_shape, positions_a, fwhm)
            layer_b = gaussian_psf_layer(map_shape, positions_b, fwhm)
            layer_matrix = np.array([layer_a, layer_b])

            true_fluxes = np.array([0.01, 0.02])
            rng_noise = np.random.default_rng(42)
            observed = layer_matrix.T @ true_fluxes
            observed += rng_noise.normal(0, 0.001, len(observed))

            _, per_bin_errs, _ = run_per_bin_bootstrap(
                layer_matrix, observed, [positions_a, positions_b],
                map_shape, fwhm, n_iterations=200
            )
            errors_a.append(per_bin_errs[0])

        # Pop A's error should be similar regardless of pop B's size
        ratio = errors_a[1] / errors_a[0]
        assert 0.5 < ratio < 2.0, (
            f"Pop A error changed by {ratio:.2f}x when pop B size changed — "
            f"per_bin should isolate variance"
        )


class TestPerBinConfigIntegration:
    """Verify config plumbing for per_bin method."""

    def test_config_default_method(self):
        """Default method should be all_bins."""
        from simstack4.config import BootstrapConfig
        config = BootstrapConfig()
        assert config.method == "all_bins"

    def test_config_per_bin_method(self):
        """Can set method to per_bin."""
        from simstack4.config import BootstrapConfig
        config = BootstrapConfig(method="per_bin")


# ---------------------------------------------------------------------------
# Analytic Gram-matrix bounds for OLS (uniform noise)
# ---------------------------------------------------------------------------


def _analytic_ols_bounds(
    layer_matrix: np.ndarray, noise_std: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the two analytic error bounds for OLS under uniform i.i.d. noise.

    Parameters
    ----------
    layer_matrix : (n_pop, n_pix)
    noise_std : per-pixel noise standard deviation

    Returns
    -------
    marginal : (n_pop,) — noise_std * sqrt(diag(inv(G)))
        The exact 1-sigma error from the lstsq covariance matrix.
        Captures cross-population confusion via the off-diagonal terms.

    conditional : (n_pop,) — noise_std / sqrt(diag(G))
        Theoretical lower bound when all other populations are held fixed.
        Per_bin bootstrap does NOT reliably track this — for non-overlapping
        PSFs the source-sampling variance collapses toward zero (A+B identity).
    """
    G = layer_matrix @ layer_matrix.T
    try:
        G_inv = np.linalg.inv(G)
        marginal = noise_std * np.sqrt(np.clip(np.diag(G_inv), 0, None))
    except np.linalg.LinAlgError:
        marginal = np.full(layer_matrix.shape[0], np.inf)
    conditional = noise_std / np.sqrt(np.diag(G))
    return marginal, conditional


def run_noise_mc(
    layer_matrix: np.ndarray,
    noise_std: float,
    true_fluxes: np.ndarray,
    n_realizations: int = 500,
    seed: int = 0,
) -> np.ndarray:
    """
    Monte Carlo estimate of the formal OLS error: std(x_hat) across noise draws.

    Produces the same result as noise_std * sqrt(diag(inv(G))) analytically.
    This is the correct ground truth for comparing bootstrap error estimates.

    Parameters
    ----------
    layer_matrix : (n_pop, n_pix)
    noise_std : per-pixel noise standard deviation (i.i.d. uniform)
    true_fluxes : (n_pop,) — signal amplitudes (variance is independent of this)
    n_realizations : number of independent noise draws

    Returns
    -------
    errors : (n_pop,) — std of fitted fluxes across noise realizations
    """
    rng = np.random.default_rng(seed)
    signal = layer_matrix.T @ true_fluxes
    n_pix = signal.shape[0]
    fluxes_list = []
    for _ in range(n_realizations):
        observed = signal + rng.normal(0, noise_std, n_pix)
        f, _ = solve_for_fluxes(layer_matrix, observed)
        fluxes_list.append(f)
    return np.std(fluxes_list, axis=0, ddof=1)


# ---------------------------------------------------------------------------
# Test 11: Per-bin errors vs noise-MC formal error
# ---------------------------------------------------------------------------


class TestPerBinAnalyticBound:
    """
    Characterises per_bin bootstrap error estimates against a noise Monte Carlo
    ground truth: std(x_hat) over independent noise draws = noise_std * sqrt(diag(inv(G))).

    Key finding documented here:
    Per_bin bootstrap SYSTEMATICALLY UNDERESTIMATES the formal OLS error.

    Mechanism (separated regime): When source PSFs do not overlap and the A/B
    split is 50:50, an algebraic identity forces x_A + x_B to be nearly constant
    regardless of which sources land in A vs B — so source-sampling variance
    collapses far below the noise-induced formal error.

    Confused regime: PSF overlap breaks the identity. Source-sampling variance
    is nonzero but still smaller than the formal error (per_bin holds other
    populations fixed, missing the cross-population coupling).

    Implication: per_bin error bars are a lower bound, most severe for
    well-separated populations.
    """

    TRUE_FLUXES = np.array([0.01, 0.02])

    # ------------------------------------------------------------------
    # Scenario builders
    # ------------------------------------------------------------------

    @staticmethod
    def _separated_scenario(map_shape=(128, 128), n_per_pop=60, noise_std=0.001):
        """Two populations in opposite halves of the map, small PSF."""
        ny, nx = map_shape
        fwhm = 4.0  # PSF sigma ≈ 1.7 px → negligible spillover across the midline
        rng = np.random.default_rng(2001)

        positions = [
            [(int(rng.integers(10, ny - 10)), int(rng.integers(5, nx // 2 - 5)))
             for _ in range(n_per_pop)],
            [(int(rng.integers(10, ny - 10)), int(rng.integers(nx // 2 + 5, nx - 5)))
             for _ in range(n_per_pop)],
        ]
        layers = [gaussian_psf_layer(map_shape, pos, fwhm) for pos in positions]
        layer_matrix = np.array(layers)

        true_fluxes = np.array([0.01, 0.02])
        rng2 = np.random.default_rng(2002)
        observed = layer_matrix.T @ true_fluxes
        observed += rng2.normal(0, noise_std, observed.shape)

        return layer_matrix, observed, positions, map_shape, fwhm, noise_std

    @staticmethod
    def _confused_scenario(map_shape=(128, 128), n_per_pop=60, noise_std=0.001):
        """Two populations spread over the full map, large PSF — heavy confusion."""
        ny, nx = map_shape
        fwhm = 22.0  # PSF sigma ≈ 9.4 px → broad, deeply overlapping layers
        rng = np.random.default_rng(3001)

        positions = [
            [(int(rng.integers(25, ny - 25)), int(rng.integers(25, nx - 25)))
             for _ in range(n_per_pop)],
            [(int(rng.integers(25, ny - 25)), int(rng.integers(25, nx - 25)))
             for _ in range(n_per_pop)],
        ]
        layers = [gaussian_psf_layer(map_shape, pos, fwhm) for pos in positions]
        layer_matrix = np.array(layers)

        true_fluxes = np.array([0.01, 0.02])
        rng2 = np.random.default_rng(3002)
        observed = layer_matrix.T @ true_fluxes
        observed += rng2.normal(0, noise_std, observed.shape)

        return layer_matrix, observed, positions, map_shape, fwhm, noise_std

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_sum_estimator_collapses_for_separated_populations(self):
        """
        Documents the A+B cancellation identity that motivated the switch to
        the diff estimator.

        With the OLD sum estimator (x_A + x_B), non-overlapping PSFs and a
        50:50 split cause x_A + x_B to be constant across partitions — the
        noise terms rearrange but cancel, collapsing variance to near-zero.

        The NEW diff estimator (x_A − x_B)/2 avoids this by keeping exactly
        the term the sum discards.  This test is a regression guard: if the
        sum is accidentally reinstated, it will collapse here.

        Asserts:
        - noise_MC ≈ analytic marginal (validates ground truth)
        - direct sum of half-layers collapses << noise_MC  (the old bug)
        - per_bin (diff estimator) does NOT collapse (the fix works)
        """
        lm, obs, pos, ms, fwhm, sigma = self._separated_scenario()
        marginal, _ = _analytic_ols_bounds(lm, sigma)
        noise_mc_errs = run_noise_mc(lm, sigma, self.TRUE_FLUXES, n_realizations=500, seed=42)

        for k in range(2):
            mc_ratio = noise_mc_errs[k] / marginal[k]
            assert 0.7 < mc_ratio < 1.5, (
                f"Pop {k}: noise_MC ({noise_mc_errs[k]:.3e}) / marginal ({marginal[k]:.3e}) = "
                f"{mc_ratio:.2f} — noise_MC should equal marginal for uniform noise"
            )

        # Compute the old sum estimator directly for comparison
        n_pop = lm.shape[0]
        sum_samples = [[] for _ in range(n_pop)]
        for k in range(n_pop):
            pos_k = pos[k]
            n_src = len(pos_k)
            n_A = n_src // 2
            for it in range(400):
                rng = np.random.RandomState(10 + k * 400 + it)
                idx = rng.permutation(n_src)
                posA = [pos_k[i] for i in idx[:n_A]]
                posB = [pos_k[i] for i in idx[n_A:]]
                layers = []
                for j in range(n_pop):
                    if j == k:
                        layers.append(gaussian_psf_layer(ms, posA, fwhm))
                        layers.append(gaussian_psf_layer(ms, posB, fwhm))
                    else:
                        layers.append(lm[j])
                x = linalg.lstsq(np.array(layers).T, obs)[0]
                sum_samples[k].append(x[k] + x[k + 1])

        _, pb_errs, _ = run_per_bin_bootstrap(lm, obs, pos, ms, fwhm, n_iterations=400, seed=10)

        for k in range(2):
            sum_err = np.std(sum_samples[k], ddof=1)
            # Old sum estimator must collapse
            assert sum_err / noise_mc_errs[k] < 0.3, (
                f"Pop {k}: sum estimator ({sum_err:.3e}) did not collapse vs "
                f"formal ({noise_mc_errs[k]:.3e}) — regression in sum/diff switch?"
            )
            # New diff estimator must not collapse
            assert pb_errs[k] / noise_mc_errs[k] > 0.3, (
                f"Pop {k}: diff estimator ({pb_errs[k]:.3e}) collapsed vs "
                f"formal ({noise_mc_errs[k]:.3e}) — diff estimator reverted to sum?"
            )

    def test_confused_per_bin_within_factor_two_of_formal(self):
        """
        Confused populations: per_bin (diff estimator) should be within
        ~0.5–2× the formal error.

        In the confused regime the diff estimator slightly overshoots (~1.1–1.2×)
        because the A/B layers are highly correlated, inflating the individual
        x_A and x_B variances beyond the conditional bound toward the marginal.
        This is a conservative bias (error bars are slightly too large, not too
        small), and within a factor of 2.

        Asserts:
        - 0.5 × formal < per_bin < 2 × formal
        - Gram off-diagonal confirms confusion is present
        """
        lm, obs, pos, ms, fwhm, sigma = self._confused_scenario()

        G = lm @ lm.T
        off_diag_fraction = abs(G[0, 1]) / np.sqrt(G[0, 0] * G[1, 1])
        assert off_diag_fraction > 0.3, (
            f"Scenario not confused enough: off-diagonal fraction {off_diag_fraction:.2f} < 0.3"
        )

        noise_mc_errs = run_noise_mc(lm, sigma, self.TRUE_FLUXES, n_realizations=500, seed=42)
        _, pb_errs, _ = run_per_bin_bootstrap(lm, obs, pos, ms, fwhm, n_iterations=400, seed=10)

        for k in range(2):
            ratio = pb_errs[k] / noise_mc_errs[k]
            assert 0.5 < ratio < 2.0, (
                f"Pop {k} (confused): per_bin ({pb_errs[k]:.3e}) / "
                f"formal ({noise_mc_errs[k]:.3e}) = {ratio:.2f}, expected 0.5–2.0"
            )

    def test_diff_estimator_tracks_formal_error(self):
        """
        With the half-difference estimator (x_A − x_B)/2, per_bin errors
        should be within 0.5–2× the noise-MC formal error in both regimes.

        This validates the production change from sum to diff.
        """
        for label, scenario_fn in [
            ("separated", self._separated_scenario),
            ("confused", self._confused_scenario),
        ]:
            lm, obs, pos, ms, fwhm, sigma = scenario_fn()
            noise_mc_errs = run_noise_mc(lm, sigma, self.TRUE_FLUXES, n_realizations=500, seed=42)
            _, pb_errs, _ = run_per_bin_bootstrap(lm, obs, pos, ms, fwhm, n_iterations=400, seed=10)

            for k in range(2):
                ratio = pb_errs[k] / noise_mc_errs[k]
                assert 0.5 < ratio < 2.0, (
                    f"{label} pop {k}: per_bin ({pb_errs[k]:.3e}) / "
                    f"formal ({noise_mc_errs[k]:.3e}) = {ratio:.2f}, "
                    f"expected 0.5–2.0"
                )

    def test_diff_estimator_consistent_across_psf_sizes(self):
        """
        With the diff estimator, per_bin / formal_error ratio should be
        consistent (within 0.5–2.5×) across both small and large PSF sizes.

        This is the key property the old sum estimator lacked: the sum collapsed
        by 10× for small PSF (separated) while giving ~0.5× for large PSF
        (confused). The diff estimator should give a stable ratio in both cases.
        """
        map_shape = (128, 128)
        ny, nx = map_shape
        n_per_pop = 60
        noise_std = 0.001
        true_fluxes = np.array([0.01, 0.02])

        rng_pos = np.random.default_rng(4001)
        positions = [
            [(int(rng_pos.integers(20, ny - 20)), int(rng_pos.integers(20, nx - 20)))
             for _ in range(n_per_pop)],
            [(int(rng_pos.integers(20, ny - 20)), int(rng_pos.integers(20, nx - 20)))
             for _ in range(n_per_pop)],
        ]

        for label, fwhm in [("small_beam", 4.0), ("large_beam", 22.0)]:
            layers = [gaussian_psf_layer(map_shape, pos, fwhm) for pos in positions]
            lm = np.array(layers)
            rng_n = np.random.default_rng(4002)
            obs = lm.T @ true_fluxes + rng_n.normal(0, noise_std, lm.shape[1])

            noise_mc_errs = run_noise_mc(lm, noise_std, true_fluxes, n_realizations=300, seed=42)
            _, pb_errs, _ = run_per_bin_bootstrap(
                lm, obs, positions, map_shape, fwhm, n_iterations=300, seed=20
            )

            for k in range(2):
                ratio = pb_errs[k] / noise_mc_errs[k]
                assert 0.5 < ratio < 2.5, (
                    f"{label} pop {k}: per_bin ({pb_errs[k]:.3e}) / "
                    f"formal ({noise_mc_errs[k]:.3e}) = {ratio:.2f}, expected 0.5–2.5"
                )
