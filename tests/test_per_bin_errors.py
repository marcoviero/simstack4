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

            # Sum A and B for pop k
            flux_A_k = split_fluxes[k]
            flux_B_k = split_fluxes[k + 1]
            flux_k_samples.append(flux_A_k + flux_B_k)

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

        # Sum A+B for each population
        total_fluxes = split_fluxes[:n_pop] + split_fluxes[n_pop:]
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
        For well-separated populations with no overlap, per-bin errors
        should be smaller than or comparable to all-bins errors.

        Per-bin avoids artificial cross-population covariance that all-bins
        introduces by splitting everything simultaneously, so per-bin
        errors can be significantly smaller for well-separated populations.
        """
        map_shape = (64, 64)
        fwhm = 3.0
        rng = np.random.default_rng(999)

        # Two populations far apart — no layer overlap
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
            # Per-bin should not be much LARGER than all-bins
            assert per_bin_errs[k] < all_bins_errs[k] * 3.0, (
                f"Pop {k}: per_bin error ({per_bin_errs[k]:.2e}) >> "
                f"all_bins error ({all_bins_errs[k]:.2e})"
            )
            # Both should be positive
            assert per_bin_errs[k] > 0
            assert all_bins_errs[k] > 0

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
        assert config.method == "per_bin"
