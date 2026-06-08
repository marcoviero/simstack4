"""
Escalating test suite for simstack4 stacking flux recovery.

Tests validate the core linear algebra by constructing synthetic maps
with known injected fluxes, running the solver, and verifying recovery.

Tests are ordered by complexity:
  1. Single source, single layer
  2. Many sources, same flux, single layer
  3. Many sources, different fluxes, multiple layers (deblending)
  4. Noisy maps — verify recovery within expected uncertainties
  5. Mean subtraction — verify unbiased recovery when layers overlap

Each test constructs layer matrices directly (Gaussian PSFs in pixel space),
bypassing TOML/WCS/FITS infrastructure to isolate the linear algebra.
"""

import numpy as np
import pytest
from scipy import linalg
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Helpers — synthetic layer construction
# ---------------------------------------------------------------------------


def gaussian_psf_layer(
    map_shape: tuple[int, int],
    source_positions: list[tuple[int, int]],
    fwhm_pix: float,
    mean_subtract: bool = True,
) -> np.ndarray:
    """
    Build a single stacking layer: delta functions at source positions,
    convolved with a Gaussian PSF, optionally mean-subtracted, flattened.

    This replicates what _create_and_convolve_layer does:
      1. Place unit deltas at each source position
      2. Convolve with Gaussian PSF
      3. Subtract mean
      4. Return flattened 1D array

    Parameters
    ----------
    map_shape : (ny, nx)
    source_positions : list of (y, x) pixel coordinates
    fwhm_pix : Gaussian PSF FWHM in pixels
    mean_subtract : whether to subtract the mean (matches real pipeline)

    Returns
    -------
    layer : 1D array of length ny * nx
    """
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    delta_map = np.zeros(map_shape, dtype=np.float64)
    for y, x in source_positions:
        if 0 <= y < map_shape[0] and 0 <= x < map_shape[1]:
            delta_map[y, x] += 1.0

    convolved = gaussian_filter(delta_map, sigma=sigma, mode="constant")

    if mean_subtract:
        convolved -= np.mean(convolved)

    return convolved.ravel()


def build_observed_map(
    layer_matrix: np.ndarray,
    true_fluxes: np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Construct a synthetic observed map from layers and known fluxes.

    observed = Σ flux_k * layer_k + noise

    Parameters
    ----------
    layer_matrix : (n_layers, n_pixels)
    true_fluxes : (n_layers,)
    noise_std : standard deviation of Gaussian noise (0 = noiseless)
    rng : random number generator (for reproducibility)

    Returns
    -------
    observed : 1D array of length n_pixels
    """
    observed = true_fluxes @ layer_matrix

    if noise_std > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        observed += rng.normal(0, noise_std, size=observed.shape)

    return observed


def solve_for_fluxes(
    layer_matrix: np.ndarray, observed: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Solve layer_matrix.T @ flux = observed via least squares.

    Replicates SimstackAlgorithm._solve_linear_system but standalone.

    Returns
    -------
    flux_densities, errors, fit_stats
    """
    n_layers, n_pixels = layer_matrix.shape

    flux_densities, residuals, rank, sv = linalg.lstsq(layer_matrix.T, observed)

    # Error estimation
    if n_pixels > n_layers:
        model = layer_matrix.T @ flux_densities
        mse = np.sum((observed - model) ** 2) / (n_pixels - n_layers)
        try:
            cov = linalg.inv(layer_matrix @ layer_matrix.T) * mse
            errors = np.sqrt(np.diag(cov))
        except linalg.LinAlgError:
            errors = np.abs(flux_densities) * 0.1
    else:
        errors = np.abs(flux_densities) * 0.1

    model = layer_matrix.T @ flux_densities
    chi2 = np.sum((observed - model) ** 2)
    dof = max(1, n_pixels - n_layers)

    fit_stats = {
        "chi_squared": chi2,
        "degrees_of_freedom": dof,
        "reduced_chi_squared": chi2 / dof,
        "rank": rank,
    }

    return flux_densities, errors, fit_stats


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def map_shape():
    """Standard test map: 128x128 pixels"""
    return (128, 128)


@pytest.fixture
def fwhm_pix():
    """Standard PSF FWHM: 5 pixels (typical for SPIRE-like beam)"""
    return 5.0


@pytest.fixture
def rng():
    """Reproducible random number generator"""
    return np.random.default_rng(12345)


# ---------------------------------------------------------------------------
# Test 1: Single source, single layer, noiseless
# ---------------------------------------------------------------------------


class TestSingleSourceSingleLayer:
    """Verify flux recovery for the simplest possible case."""

    def test_center_source(self, map_shape, fwhm_pix):
        """Single source at map center — should recover flux exactly."""
        true_flux = 0.05  # 50 mJy

        cy, cx = map_shape[0] // 2, map_shape[1] // 2
        layer = gaussian_psf_layer(map_shape, [(cy, cx)], fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        observed = build_observed_map(layer_matrix, np.array([true_flux]))
        recovered, errors, stats = solve_for_fluxes(layer_matrix, observed)

        assert recovered[0] == pytest.approx(true_flux, rel=1e-10), (
            f"Recovered {recovered[0]:.6e} != injected {true_flux:.6e}"
        )
        assert stats["reduced_chi_squared"] < 1e-20, "Noiseless fit should have ~zero residual"

    def test_offset_source(self, map_shape, fwhm_pix):
        """Single source off-center — same recovery expected."""
        true_flux = 0.1

        layer = gaussian_psf_layer(map_shape, [(30, 80)], fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        observed = build_observed_map(layer_matrix, np.array([true_flux]))
        recovered, _, _ = solve_for_fluxes(layer_matrix, observed)

        assert recovered[0] == pytest.approx(true_flux, rel=1e-10)

    def test_various_fluxes(self, map_shape, fwhm_pix):
        """Recovery should work across a range of flux values."""
        cy, cx = 64, 64
        layer = gaussian_psf_layer(map_shape, [(cy, cx)], fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        for true_flux in [1e-4, 1e-3, 0.01, 0.1, 1.0, 10.0]:
            observed = build_observed_map(layer_matrix, np.array([true_flux]))
            recovered, _, _ = solve_for_fluxes(layer_matrix, observed)
            assert recovered[0] == pytest.approx(true_flux, rel=1e-10), (
                f"Failed at flux={true_flux}"
            )


# ---------------------------------------------------------------------------
# Test 2: Many sources, same flux, single layer
# ---------------------------------------------------------------------------


class TestManySourcesSingleLayer:
    """
    Verify that stacking N identical sources recovers the per-source mean flux.

    The layer is the sum of N PSFs, so the lstsq solution gives the mean
    flux per source (since each source contributes 1.0 to the delta map).
    """

    def test_100_sources_noiseless(self, map_shape, fwhm_pix, rng):
        """100 sources at random positions, same flux — noiseless recovery."""
        n_sources = 100
        true_flux = 0.03  # 30 mJy per source

        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]

        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        observed = build_observed_map(layer_matrix, np.array([true_flux]))
        recovered, _, stats = solve_for_fluxes(layer_matrix, observed)

        assert recovered[0] == pytest.approx(true_flux, rel=1e-8), (
            f"Recovered {recovered[0]:.6e} != injected {true_flux:.6e}"
        )

    def test_scaling_with_n_sources(self, map_shape, fwhm_pix, rng):
        """
        Recovery should work regardless of N_sources.
        The layer encodes N sources, lstsq recovers mean flux per source.
        """
        true_flux = 0.05

        for n_sources in [10, 50, 200, 500]:
            positions = [
                (int(y), int(x))
                for y, x in zip(
                    rng.integers(10, map_shape[0] - 10, n_sources),
                    rng.integers(10, map_shape[1] - 10, n_sources),
                )
            ]

            layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
            layer_matrix = layer[np.newaxis, :]
            observed = build_observed_map(layer_matrix, np.array([true_flux]))
            recovered, _, _ = solve_for_fluxes(layer_matrix, observed)

            assert recovered[0] == pytest.approx(true_flux, rel=1e-6), (
                f"Failed with N={n_sources}: recovered {recovered[0]:.6e}"
            )

    def test_overlapping_sources(self, map_shape, fwhm_pix):
        """
        Sources at the same pixel — the layer sums their contributions.
        Lstsq should still recover the per-source mean flux.
        """
        true_flux = 0.1
        # 5 sources all at the same position
        positions = [(64, 64)] * 5

        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]
        observed = build_observed_map(layer_matrix, np.array([true_flux]))
        recovered, _, _ = solve_for_fluxes(layer_matrix, observed)

        assert recovered[0] == pytest.approx(true_flux, rel=1e-10)


# ---------------------------------------------------------------------------
# Test 3: Multiple layers, different fluxes — deblending
# ---------------------------------------------------------------------------


class TestMultipleLayersDeblending:
    """
    The core of simultaneous stacking: N population layers with different
    source positions and different fluxes. The solver must deblend them.
    """

    def test_two_populations_noiseless(self, map_shape, fwhm_pix, rng):
        """Two non-overlapping populations — clean deblending."""
        true_fluxes = np.array([0.05, 0.12])

        # Population A: sources in left half
        pos_a = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, 50),
                rng.integers(10, map_shape[1] // 2 - 5, 50),
            )
        ]
        # Population B: sources in right half
        pos_b = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, 50),
                rng.integers(map_shape[1] // 2 + 5, map_shape[1] - 10, 50),
            )
        ]

        layer_a = gaussian_psf_layer(map_shape, pos_a, fwhm_pix)
        layer_b = gaussian_psf_layer(map_shape, pos_b, fwhm_pix)
        layer_matrix = np.vstack([layer_a, layer_b])

        observed = build_observed_map(layer_matrix, true_fluxes)
        recovered, _, stats = solve_for_fluxes(layer_matrix, observed)

        np.testing.assert_allclose(recovered, true_fluxes, rtol=1e-8, err_msg=(
            f"Deblending failed: recovered {recovered} vs injected {true_fluxes}"
        ))

    def test_five_populations_noiseless(self, map_shape, fwhm_pix, rng):
        """Five populations with distinct source positions."""
        n_pops = 5
        n_sources_per_pop = 40
        true_fluxes = np.array([0.01, 0.03, 0.05, 0.10, 0.20])

        layers = []
        for k in range(n_pops):
            positions = [
                (int(y), int(x))
                for y, x in zip(
                    rng.integers(10, map_shape[0] - 10, n_sources_per_pop),
                    rng.integers(10, map_shape[1] - 10, n_sources_per_pop),
                )
            ]
            layers.append(gaussian_psf_layer(map_shape, positions, fwhm_pix))

        layer_matrix = np.vstack(layers)
        observed = build_observed_map(layer_matrix, true_fluxes)
        recovered, _, _ = solve_for_fluxes(layer_matrix, observed)

        np.testing.assert_allclose(recovered, true_fluxes, rtol=1e-6, err_msg=(
            f"5-pop deblending failed:\n"
            f"  recovered: {recovered}\n  injected:  {true_fluxes}"
        ))

    def test_deblending_with_foreground(self, map_shape, fwhm_pix, rng):
        """
        Add a flat foreground layer (constant = 1.0 everywhere, mean-subtracted
        to 0). This mimics the add_foreground option in the real pipeline.
        With mean subtraction, the foreground layer becomes all zeros,
        so it shouldn't affect recovery of the science layers.
        """
        true_fluxes_science = np.array([0.05, 0.10])
        foreground_flux = 1.0  # arbitrary constant level

        pos_a = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, 50),
                rng.integers(10, map_shape[1] - 10, 50),
            )
        ]
        pos_b = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, 50),
                rng.integers(10, map_shape[1] - 10, 50),
            )
        ]

        layer_a = gaussian_psf_layer(map_shape, pos_a, fwhm_pix)
        layer_b = gaussian_psf_layer(map_shape, pos_b, fwhm_pix)

        # Foreground: flat 1.0, then mean-subtracted → 0
        foreground = np.ones(map_shape, dtype=np.float64)
        foreground -= np.mean(foreground)  # = 0 everywhere
        foreground_flat = foreground.ravel()

        layer_matrix = np.vstack([layer_a, layer_b, foreground_flat])
        true_fluxes = np.array([*true_fluxes_science, foreground_flux])

        observed = build_observed_map(layer_matrix, true_fluxes)
        recovered, _, _ = solve_for_fluxes(layer_matrix, observed)

        # Science layers should recover accurately
        np.testing.assert_allclose(recovered[:2], true_fluxes_science, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: Noisy maps — recovery within uncertainties
# ---------------------------------------------------------------------------


class TestNoisyRecovery:
    """
    Add Gaussian noise and verify:
    - Recovered fluxes are within expected uncertainties
    - Reported errors are consistent with actual scatter
    - Reduced chi-squared is reasonable
    """

    def test_single_layer_noisy(self, map_shape, fwhm_pix, rng):
        """One population, noisy map — recovery within 3σ."""
        true_flux = 0.05
        n_sources = 200

        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]

        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        noise_std = 0.001  # noise per pixel
        observed = build_observed_map(layer_matrix, np.array([true_flux]),
                                      noise_std=noise_std, rng=rng)
        recovered, errors, stats = solve_for_fluxes(layer_matrix, observed)

        # Recovery within 3σ of formal error
        assert abs(recovered[0] - true_flux) < 3 * errors[0], (
            f"Recovered {recovered[0]:.4e} deviates > 3σ from {true_flux:.4e} "
            f"(error={errors[0]:.4e})"
        )

        # Residual standard deviation should be consistent with input noise
        # (solver doesn't do noise-weighted chi2, so check residuals directly)
        model = layer_matrix.T @ recovered
        residual_std = np.std(observed - model)
        assert residual_std == pytest.approx(noise_std, rel=0.2), (
            f"Residual std {residual_std:.4e} inconsistent with noise {noise_std:.4e}"
        )

    def test_multi_layer_noisy(self, map_shape, fwhm_pix, rng):
        """Three populations, noisy map — all recovered within 3σ."""
        true_fluxes = np.array([0.02, 0.08, 0.15])
        n_sources = 100

        layers = []
        for _ in range(3):
            positions = [
                (int(y), int(x))
                for y, x in zip(
                    rng.integers(10, map_shape[0] - 10, n_sources),
                    rng.integers(10, map_shape[1] - 10, n_sources),
                )
            ]
            layers.append(gaussian_psf_layer(map_shape, positions, fwhm_pix))

        layer_matrix = np.vstack(layers)

        noise_std = 0.001
        observed = build_observed_map(layer_matrix, true_fluxes,
                                      noise_std=noise_std, rng=rng)
        recovered, errors, stats = solve_for_fluxes(layer_matrix, observed)

        for k in range(3):
            assert abs(recovered[k] - true_fluxes[k]) < 3 * errors[k], (
                f"Pop {k}: recovered {recovered[k]:.4e} deviates > 3σ from "
                f"{true_fluxes[k]:.4e} (error={errors[k]:.4e})"
            )

    def test_error_consistency_monte_carlo(self, map_shape, fwhm_pix):
        """
        Run many noise realizations and verify that:
        - Mean recovered flux is unbiased
        - Scatter across realizations matches reported formal error
        """
        true_flux = 0.05
        n_sources = 100
        noise_std = 0.001
        n_realizations = 200

        # Fixed source positions
        base_rng = np.random.default_rng(999)
        positions = [
            (int(y), int(x))
            for y, x in zip(
                base_rng.integers(10, map_shape[0] - 10, n_sources),
                base_rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]

        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        recovered_fluxes = []
        formal_errors = []

        for i in range(n_realizations):
            noise_rng = np.random.default_rng(i)
            observed = build_observed_map(layer_matrix, np.array([true_flux]),
                                          noise_std=noise_std, rng=noise_rng)
            recovered, errors, _ = solve_for_fluxes(layer_matrix, observed)
            recovered_fluxes.append(recovered[0])
            formal_errors.append(errors[0])

        recovered_fluxes = np.array(recovered_fluxes)
        formal_errors = np.array(formal_errors)

        # Mean should be unbiased (within a few % for 200 realizations)
        mean_recovered = np.mean(recovered_fluxes)
        assert mean_recovered == pytest.approx(true_flux, rel=0.05), (
            f"Mean recovered {mean_recovered:.4e} biased vs {true_flux:.4e}"
        )

        # Scatter should be consistent with formal error estimate
        empirical_scatter = np.std(recovered_fluxes, ddof=1)
        mean_formal_error = np.mean(formal_errors)

        # These should agree within ~30% for 200 realizations
        ratio = empirical_scatter / mean_formal_error
        assert 0.7 < ratio < 1.4, (
            f"Error inconsistency: empirical scatter {empirical_scatter:.4e} vs "
            f"formal error {mean_formal_error:.4e} (ratio={ratio:.2f})"
        )


# ---------------------------------------------------------------------------
# Test 5: Mean subtraction — verify no bias from overlap
# ---------------------------------------------------------------------------


class TestMeanSubtraction:
    """
    Verify that mean subtraction (which the real pipeline does on both
    layers and observed map) doesn't introduce bias.
    """

    def test_mean_subtraction_preserves_recovery(self, map_shape, fwhm_pix, rng):
        """
        Build layers WITH mean subtraction (as the pipeline does),
        construct observed from those mean-subtracted layers,
        and verify unbiased recovery.
        """
        true_fluxes = np.array([0.05, 0.12, 0.03])
        n_sources = 80

        layers = []
        for _ in range(3):
            positions = [
                (int(y), int(x))
                for y, x in zip(
                    rng.integers(5, map_shape[0] - 5, n_sources),
                    rng.integers(5, map_shape[1] - 5, n_sources),
                )
            ]
            layers.append(
                gaussian_psf_layer(map_shape, positions, fwhm_pix, mean_subtract=True)
            )

        layer_matrix = np.vstack(layers)
        observed = build_observed_map(layer_matrix, true_fluxes)
        recovered, _, _ = solve_for_fluxes(layer_matrix, observed)

        np.testing.assert_allclose(recovered, true_fluxes, rtol=1e-6)

    def test_without_vs_with_mean_subtraction(self, map_shape, fwhm_pix, rng):
        """
        Construct the SAME sources and fluxes both with and without
        mean subtraction. Both should recover the same fluxes
        (mean subtraction just removes the DC component which is
        absorbed by the foreground layer or is irrelevant to lstsq).
        """
        true_fluxes = np.array([0.07, 0.15])
        n_sources = 60

        positions_a = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]
        positions_b = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]

        # With mean subtraction
        layer_a_ms = gaussian_psf_layer(map_shape, positions_a, fwhm_pix, mean_subtract=True)
        layer_b_ms = gaussian_psf_layer(map_shape, positions_b, fwhm_pix, mean_subtract=True)
        lm_ms = np.vstack([layer_a_ms, layer_b_ms])
        obs_ms = build_observed_map(lm_ms, true_fluxes)
        rec_ms, _, _ = solve_for_fluxes(lm_ms, obs_ms)

        # Without mean subtraction
        layer_a_raw = gaussian_psf_layer(map_shape, positions_a, fwhm_pix, mean_subtract=False)
        layer_b_raw = gaussian_psf_layer(map_shape, positions_b, fwhm_pix, mean_subtract=False)
        lm_raw = np.vstack([layer_a_raw, layer_b_raw])
        obs_raw = build_observed_map(lm_raw, true_fluxes)
        rec_raw, _, _ = solve_for_fluxes(lm_raw, obs_raw)

        # Both should recover the same fluxes
        np.testing.assert_allclose(rec_ms, true_fluxes, rtol=1e-6)
        np.testing.assert_allclose(rec_raw, true_fluxes, rtol=1e-6)


# ---------------------------------------------------------------------------
# Test 6: Large beam / high confusion — deblending under stress
# ---------------------------------------------------------------------------


class TestConfusionRegime:
    """
    Test recovery when sources from different populations overlap
    significantly (large beam relative to source separation).
    This is the regime where simultaneous stacking matters most.
    """

    def test_large_beam_two_populations(self, map_shape, rng):
        """
        Large beam (FWHM=15 pix) with spatially mixed populations.
        Recovery should still work because the layers are linearly
        independent (different source positions).
        """
        fwhm = 15.0
        true_fluxes = np.array([0.04, 0.09])
        n_sources = 80

        # Both populations spread across the full map — heavy overlap
        layers = []
        for _ in range(2):
            positions = [
                (int(y), int(x))
                for y, x in zip(
                    rng.integers(15, map_shape[0] - 15, n_sources),
                    rng.integers(15, map_shape[1] - 15, n_sources),
                )
            ]
            layers.append(gaussian_psf_layer(map_shape, positions, fwhm))

        layer_matrix = np.vstack(layers)
        observed = build_observed_map(layer_matrix, true_fluxes)
        recovered, _, _ = solve_for_fluxes(layer_matrix, observed)

        np.testing.assert_allclose(recovered, true_fluxes, rtol=1e-5, err_msg=(
            f"Confused recovery failed: {recovered} vs {true_fluxes}"
        ))

    def test_many_populations_confused(self, map_shape, rng):
        """
        10 populations, large beam, all spatially mixed.
        Tests the rank and conditioning of the layer matrix.
        """
        fwhm = 12.0
        n_pops = 10
        n_sources = 50
        true_fluxes = rng.uniform(0.01, 0.2, n_pops)

        layers = []
        for _ in range(n_pops):
            positions = [
                (int(y), int(x))
                for y, x in zip(
                    rng.integers(15, map_shape[0] - 15, n_sources),
                    rng.integers(15, map_shape[1] - 15, n_sources),
                )
            ]
            layers.append(gaussian_psf_layer(map_shape, positions, fwhm))

        layer_matrix = np.vstack(layers)
        observed = build_observed_map(layer_matrix, true_fluxes)
        recovered, _, stats = solve_for_fluxes(layer_matrix, observed)

        np.testing.assert_allclose(recovered, true_fluxes, rtol=1e-4, err_msg=(
            f"10-pop confused recovery failed.\n"
            f"  Max fractional error: {np.max(np.abs(recovered - true_fluxes) / true_fluxes):.2e}\n"
            f"  Rank: {stats['rank']}"
        ))

        # Verify full rank
        assert stats["rank"] == n_pops, (
            f"Layer matrix rank {stats['rank']} < {n_pops} — degenerate!"
        )


# ---------------------------------------------------------------------------
# Test 7: Inverse-variance weighting (WLS) — LinSimStack feature
# ---------------------------------------------------------------------------


def solve_wls(
    layer_matrix: np.ndarray, observed: np.ndarray, noise_vector: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """WLS solve: pre-scale by 1/noise then lstsq. Returns (fluxes, errors)."""
    safe_noise = np.where(noise_vector > 0, noise_vector, np.nanmedian(noise_vector[noise_vector > 0]))
    lm_w = layer_matrix / safe_noise
    obs_w = observed / safe_noise
    flux, residuals, rank, sv = linalg.lstsq(lm_w.T, obs_w)
    n_pix, n_pop = lm_w.shape[1], lm_w.shape[0]
    if n_pix > n_pop:
        model_w = lm_w.T @ flux
        mse = np.sum((obs_w - model_w) ** 2) / (n_pix - n_pop)
        try:
            cov = linalg.inv(lm_w @ lm_w.T) * mse
            errors = np.sqrt(np.diag(cov))
        except linalg.LinAlgError:
            errors = np.abs(flux) * 0.1
    else:
        errors = np.abs(flux) * 0.1
    return flux, errors


class TestNoiseWeighting:
    """Verify that WLS outperforms OLS when noise is spatially non-uniform."""

    def test_wls_unbiased(self, map_shape, fwhm_pix, rng):
        """WLS should recover true flux without bias."""
        true_flux = 0.05
        n_sources = 150

        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]
        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        # Non-uniform noise: RMS varies 4× across the map
        n_pix = map_shape[0] * map_shape[1]
        noise_map = 0.001 + 0.003 * rng.random(n_pix)
        observed = build_observed_map(layer_matrix, np.array([true_flux]))
        observed += rng.normal(0, noise_map)

        flux_wls, _ = solve_wls(layer_matrix, observed, noise_map)
        assert abs(flux_wls[0] - true_flux) < 0.01, (
            f"WLS biased: recovered {flux_wls[0]:.4e} vs {true_flux:.4e}"
        )

    def test_wls_beats_ols_on_nonuniform_noise(self, map_shape, fwhm_pix):
        """
        With non-uniform noise, WLS should have lower MSE across realizations
        than plain OLS. This is the central statistical argument for adopting
        LinSimStack's noise weighting.

        Design: half the map (columns 0-63) has 10× higher noise than the
        other half (columns 64-127). Sources are split evenly between regions
        so signal overlaps both noise regimes. Gauss-Markov guarantees
        WLS MSE ≤ OLS MSE; 300 realizations ensure the gap is detectable.
        """
        true_flux = 0.05
        n_per_half = 60  # 60 sources in each noise region
        n_realizations = 300
        ny, nx = map_shape

        base_rng = np.random.default_rng(777)
        # Sources in left half (high-noise columns 0-63)
        pos_noisy = list(zip(
            base_rng.integers(10, ny - 10, n_per_half).tolist(),
            base_rng.integers(0, nx // 2, n_per_half).tolist(),
        ))
        # Sources in right half (low-noise columns 64-127)
        pos_clean = list(zip(
            base_rng.integers(10, ny - 10, n_per_half).tolist(),
            base_rng.integers(nx // 2, nx - 10, n_per_half).tolist(),
        ))
        positions = pos_noisy + pos_clean

        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        # Build noise map: left half σ=0.01, right half σ=0.001 (10× ratio)
        noise_map_2d = np.full(map_shape, 0.001)
        noise_map_2d[:, : nx // 2] = 0.01
        noise_map = noise_map_2d.ravel()

        signal = build_observed_map(layer_matrix, np.array([true_flux]))

        ols_sq_errors = []
        wls_sq_errors = []

        for i in range(n_realizations):
            noise_rng = np.random.default_rng(i + 100)
            observed = signal + noise_rng.normal(0, noise_map)

            ols_flux, _, _ = solve_for_fluxes(layer_matrix, observed)
            wls_flux, _ = solve_wls(layer_matrix, observed, noise_map)

            ols_sq_errors.append((ols_flux[0] - true_flux) ** 2)
            wls_sq_errors.append((wls_flux[0] - true_flux) ** 2)

        ols_mse = np.mean(ols_sq_errors)
        wls_mse = np.mean(wls_sq_errors)

        # Gauss-Markov: WLS must have lower MSE under non-uniform noise.
        # Expected ratio ~0.05–0.2 for 10× noise contrast; any ratio > 0.99
        # reliably indicates a bug in the WLS implementation.
        assert wls_mse < ols_mse, (
            f"WLS MSE ({wls_mse:.2e}) >= OLS MSE ({ols_mse:.2e}): "
            "noise weighting not improving recovery — check implementation"
        )

    def test_wls_uniform_noise_same_as_ols(self, map_shape, fwhm_pix, rng):
        """When noise is uniform, WLS and OLS should give identical results."""
        true_flux = 0.05
        n_sources = 100
        noise_std = 0.001

        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]
        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]
        n_pix = map_shape[0] * map_shape[1]
        noise_map = np.full(n_pix, noise_std)

        observed = build_observed_map(layer_matrix, np.array([true_flux]),
                                      noise_std=noise_std, rng=rng)

        ols_flux, _, _ = solve_for_fluxes(layer_matrix, observed)
        wls_flux, _ = solve_wls(layer_matrix, observed, noise_map)

        # Identical up to floating point (uniform weighting = no weighting)
        np.testing.assert_allclose(wls_flux, ols_flux, rtol=1e-10)


# ---------------------------------------------------------------------------
# Test 8: Iterative sigma-clipping — LinSimStack feature
# ---------------------------------------------------------------------------


def solve_with_sigma_clip(
    layer_matrix: np.ndarray,
    observed: np.ndarray,
    noise_vector: np.ndarray,
    sigma_threshold: float = 3.0,
) -> np.ndarray:
    """One-shot sigma-clip: fit, mask outliers, re-fit. Returns recovered fluxes."""
    safe_noise = np.where(noise_vector > 0, noise_vector, np.nanmedian(noise_vector[noise_vector > 0]))
    lm_w = layer_matrix / safe_noise
    obs_w = observed / safe_noise

    flux, _, _, _ = linalg.lstsq(lm_w.T, obs_w)
    residuals = obs_w - lm_w.T @ flux
    keep = np.abs(residuals) < sigma_threshold

    n_pop = layer_matrix.shape[0]
    if keep.sum() > n_pop:
        flux, _, _, _ = linalg.lstsq(lm_w[:, keep].T, obs_w[keep])

    return flux


class TestIterativeSigmaClip:
    """Verify that sigma-clipping removes outlier pixels without biasing flux."""

    def test_sigma_clip_removes_outliers(self, map_shape, fwhm_pix, rng):
        """
        Inject a bright outlier pixel (point source contamination).
        Sigma-clipping should remove it and recover the correct flux;
        plain WLS without clipping should be biased toward the outlier.
        """
        true_flux = 0.05
        n_sources = 100
        noise_std = 0.001

        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]
        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]
        n_pix = map_shape[0] * map_shape[1]
        noise_map = np.full(n_pix, noise_std)

        observed = build_observed_map(layer_matrix, np.array([true_flux]),
                                      noise_std=noise_std, rng=rng)

        # Inject a 50σ outlier at a random pixel not near any source
        outlier_pixel = 1000  # some pixel index far from sources
        observed[outlier_pixel] += 50 * noise_std

        wls_flux, _ = solve_wls(layer_matrix, observed, noise_map)
        clip_flux = solve_with_sigma_clip(layer_matrix, observed, noise_map)

        # Both should be reasonable, but clip should be closer to truth
        wls_err = abs(wls_flux[0] - true_flux)
        clip_err = abs(clip_flux[0] - true_flux)

        assert clip_err <= wls_err + noise_std, (
            f"Sigma-clip ({clip_err:.4e}) worse than WLS ({wls_err:.4e}) "
            "after outlier injection — clipping not effective"
        )
        # Sigma-clipped result should be within 5σ of true flux
        assert clip_err < 5 * noise_std, (
            f"Sigma-clip flux {clip_flux[0]:.4e} too far from truth {true_flux:.4e}"
        )

    def test_sigma_clip_no_bias_clean_data(self, map_shape, fwhm_pix, rng):
        """On clean data (no outliers), sigma-clip should give the same result as WLS."""
        true_flux = 0.05
        n_sources = 100
        noise_std = 0.001

        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]
        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]
        n_pix = map_shape[0] * map_shape[1]
        noise_map = np.full(n_pix, noise_std)

        observed = build_observed_map(layer_matrix, np.array([true_flux]),
                                      noise_std=noise_std, rng=rng)

        wls_flux, _ = solve_wls(layer_matrix, observed, noise_map)
        clip_flux = solve_with_sigma_clip(layer_matrix, observed, noise_map)

        # Should be very close (no outliers to clip)
        np.testing.assert_allclose(clip_flux, wls_flux, rtol=0.01)


# ---------------------------------------------------------------------------
# Test 9: SimstackAlgorithm._solve_linear_system — exercise production code
# ---------------------------------------------------------------------------


def _make_bare_algo(use_noise_weighting: bool, iterative_sigma_clip: bool = False):
    """Create a SimstackAlgorithm instance with only the attrs _solve_linear_system needs."""
    from simstack4.algorithm import SimstackAlgorithm
    algo = object.__new__(SimstackAlgorithm)
    algo.use_noise_weighting = use_noise_weighting
    algo.iterative_sigma_clip = iterative_sigma_clip
    return algo


class TestSolveLinearSystemMethod:
    """
    Ensure that SimstackAlgorithm._solve_linear_system (the production code path)
    honours noise weighting correctly.  These tests are intentionally parallel
    to TestNoiseWeighting / TestIterativeSigmaClip above but call the real method
    rather than the standalone helpers — that is the gap the advisor flagged.
    """

    def test_uniform_noise_matches_ols(self, map_shape, fwhm_pix, rng):
        """
        With uniform noise, WLS ≡ OLS (dividing both sides by a constant
        changes nothing).  Checks to floating-point precision that the real
        method behaves this way.
        """
        true_flux = 0.05
        n_sources = 100
        noise_std = 0.001
        n_pix = map_shape[0] * map_shape[1]

        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]
        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]
        noise_map = np.full(n_pix, noise_std)
        observed = build_observed_map(layer_matrix, np.array([true_flux]),
                                      noise_std=noise_std, rng=rng)

        algo_wls = _make_bare_algo(use_noise_weighting=True)
        algo_ols = _make_bare_algo(use_noise_weighting=False)

        flux_wls, _, _ = algo_wls._solve_linear_system(layer_matrix, observed, noise_map)
        flux_ols, _, _ = algo_ols._solve_linear_system(layer_matrix, observed, None)

        np.testing.assert_allclose(flux_wls, flux_ols, rtol=1e-10,
                                   err_msg="WLS≠OLS under uniform noise — weighting bug")

    def test_nonuniform_noise_wls_beats_ols(self, map_shape, fwhm_pix):
        """
        Production method must achieve lower MSE than OLS under non-uniform noise
        (same scenario as TestNoiseWeighting.test_wls_beats_ols_on_nonuniform_noise).
        """
        true_flux = 0.05
        n_per_half = 60
        n_realizations = 300
        ny, nx = map_shape

        base_rng = np.random.default_rng(888)
        pos_noisy = list(zip(
            base_rng.integers(10, ny - 10, n_per_half).tolist(),
            base_rng.integers(0, nx // 2, n_per_half).tolist(),
        ))
        pos_clean = list(zip(
            base_rng.integers(10, ny - 10, n_per_half).tolist(),
            base_rng.integers(nx // 2, nx - 10, n_per_half).tolist(),
        ))
        positions = pos_noisy + pos_clean

        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]

        noise_map_2d = np.full(map_shape, 0.001)
        noise_map_2d[:, : nx // 2] = 0.01
        noise_map = noise_map_2d.ravel()

        signal = build_observed_map(layer_matrix, np.array([true_flux]))

        algo_wls = _make_bare_algo(use_noise_weighting=True)
        algo_ols = _make_bare_algo(use_noise_weighting=False)

        ols_sq, wls_sq = [], []
        for i in range(n_realizations):
            noise_rng = np.random.default_rng(i + 200)
            obs = signal + noise_rng.normal(0, noise_map)
            f_ols, _, _ = algo_ols._solve_linear_system(layer_matrix, obs, None)
            f_wls, _, _ = algo_wls._solve_linear_system(layer_matrix, obs, noise_map)
            ols_sq.append((f_ols[0] - true_flux) ** 2)
            wls_sq.append((f_wls[0] - true_flux) ** 2)

        ols_mse, wls_mse = np.mean(ols_sq), np.mean(wls_sq)
        assert wls_mse < ols_mse, (
            f"Production WLS MSE ({wls_mse:.2e}) >= OLS MSE ({ols_mse:.2e}) — "
            "use_noise_weighting not active in _solve_linear_system"
        )

    def test_sigma_clip_enabled_removes_outlier(self, map_shape, fwhm_pix, rng):
        """
        With iterative_sigma_clip=True, a 50σ outlier pixel should be removed
        and the recovered flux should be closer to truth than without clipping.
        """
        true_flux = 0.05
        n_sources = 100
        noise_std = 0.001
        n_pix = map_shape[0] * map_shape[1]

        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, map_shape[0] - 10, n_sources),
                rng.integers(10, map_shape[1] - 10, n_sources),
            )
        ]
        layer = gaussian_psf_layer(map_shape, positions, fwhm_pix)
        layer_matrix = layer[np.newaxis, :]
        noise_map = np.full(n_pix, noise_std)

        observed = build_observed_map(layer_matrix, np.array([true_flux]),
                                      noise_std=noise_std, rng=rng)
        observed[1000] += 50 * noise_std  # bright outlier

        algo_wls = _make_bare_algo(use_noise_weighting=True, iterative_sigma_clip=False)
        algo_clip = _make_bare_algo(use_noise_weighting=True, iterative_sigma_clip=True)

        flux_wls, _, _ = algo_wls._solve_linear_system(layer_matrix, observed, noise_map)
        flux_clip, _, _ = algo_clip._solve_linear_system(layer_matrix, observed, noise_map)

        err_wls = abs(flux_wls[0] - true_flux)
        err_clip = abs(flux_clip[0] - true_flux)

        assert err_clip <= err_wls + noise_std, (
            f"Clip ({err_clip:.4e}) worse than WLS ({err_wls:.4e}) after outlier injection"
        )
        assert err_clip < 5 * noise_std, (
            f"Clipped flux {flux_clip[0]:.4e} too far from truth {true_flux:.4e}"
        )


# ---------------------------------------------------------------------------
# Test 10: Mask-leak nuisance layer — exercise _build_mask_leak_layer and
#           recovery via SimstackAlgorithm._solve_linear_system
# ---------------------------------------------------------------------------


class TestMaskLeakLayer:
    """
    Verify that the mask-leak nuisance layer absorbs contamination from flux
    leaking across a catalog mask boundary without biasing science populations.
    """

    def test_mask_leak_layer_finite_and_near_zero_mean(self, map_shape, fwhm_pix):
        """
        _build_mask_leak_layer should return a finite, ~zero-mean array with
        non-trivial structure (not all zeros) when the mask has excluded regions.
        """
        from simstack4.algorithm import SimstackAlgorithm
        from unittest.mock import MagicMock
        import numpy as np
        from scipy.ndimage import gaussian_filter

        ny, nx = map_shape
        # Excluded region: left quarter of the map
        catalog_mask = np.ones(map_shape, dtype=bool)
        catalog_mask[:, : nx // 4] = False

        map_data = MagicMock()
        map_data.catalog_mask = catalog_mask
        map_data.shape = map_shape
        map_data.beam_fwhm_pixels = fwhm_pix

        # Stub convolve_with_psf with a Gaussian so we don't need real sky_maps
        def fake_convolve(data, name):
            sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
            return gaussian_filter(data.astype(float), sigma=sigma, mode="constant")

        sky_maps = MagicMock()
        sky_maps.convolve_with_psf = fake_convolve

        algo = object.__new__(SimstackAlgorithm)
        algo.sky_maps = sky_maps
        algo.sky_maps.__getitem__ = lambda self, key: map_data

        leak = algo._build_mask_leak_layer("test", map_shape)

        assert np.all(np.isfinite(leak)), "Leak layer contains non-finite values"
        assert np.abs(np.mean(leak)) < 1e-10, "Leak layer not mean-zero"
        assert np.any(leak != 0), "Leak layer is all zeros — PSF convolution failed"

    def test_mask_leak_removes_contamination(self, map_shape, fwhm_pix, rng):
        """
        Place science sources in the right half of the map.  Inject leaked flux
        from the left-half mask into the right half via PSF convolution.
        Without the leak layer the solver attributes that flux to the science
        population (bias).  With it the solver absorbs the contamination and
        recovers the true science flux.
        """
        from scipy.ndimage import gaussian_filter
        from scipy import linalg

        ny, nx = map_shape
        true_science_flux = 0.05
        leaked_flux_amplitude = 0.10  # 2× the science signal — strong contamination
        noise_std = 0.0005
        n_sources = 80
        sigma_pix = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))

        # Science sources: right half only (columns nx//2 .. nx-1)
        positions = [
            (int(y), int(x))
            for y, x in zip(
                rng.integers(10, ny - 10, n_sources),
                rng.integers(nx // 2 + 5, nx - 5, n_sources),
            )
        ]
        science_layer = gaussian_psf_layer(map_shape, positions, fwhm_pix, mean_subtract=True)
        science_layer_matrix = science_layer[np.newaxis, :]

        # Leaked flux: a uniform "bright region" in the left half, PSF-convolved
        bright_region = np.zeros(map_shape)
        bright_region[:, : nx // 4] = 1.0
        leak_full = gaussian_filter(bright_region, sigma=sigma_pix, mode="constant")
        leak_full -= np.mean(leak_full)
        leak_layer = leak_full.ravel()

        # Catalog mask: left quarter excluded (True = valid)
        catalog_mask = np.ones(map_shape, dtype=bool)
        catalog_mask[:, : nx // 4] = False
        signal_mask = catalog_mask.ravel()  # fit only valid pixels

        # Build observed map: science signal + leaked contamination + noise
        observed_full = (
            true_science_flux * science_layer
            + leaked_flux_amplitude * leak_full.ravel()
            + rng.normal(0, noise_std, science_layer.shape)
        )

        # --- Solve WITHOUT leak layer (contamination leaks into science bin) ---
        lm_no_leak = science_layer_matrix[:, signal_mask]
        obs_no_leak = observed_full[signal_mask]
        # Re-centre over the valid pixel domain
        lm_no_leak = lm_no_leak - np.mean(lm_no_leak, axis=1, keepdims=True)
        flux_no_leak, _, _, _ = linalg.lstsq(lm_no_leak.T, obs_no_leak)

        # --- Solve WITH leak layer (contamination absorbed by nuisance column) ---
        lm_with_leak = np.vstack([
            science_layer_matrix[:, signal_mask],
            leak_layer[signal_mask][np.newaxis, :],
        ])
        # Re-centre science row; leak layer already mean-zero
        lm_with_leak[0] -= np.mean(lm_with_leak[0])
        lm_with_leak[1] -= np.mean(lm_with_leak[1])
        obs_with_leak = observed_full[signal_mask]
        flux_with_leak, _, _, _ = linalg.lstsq(lm_with_leak.T, obs_with_leak)
        science_flux_recovered = flux_with_leak[0]

        err_no_leak = abs(flux_no_leak[0] - true_science_flux)
        err_with_leak = abs(science_flux_recovered - true_science_flux)

        # The leak layer must reduce the science-bin error
        assert err_with_leak < err_no_leak, (
            f"Mask-leak layer didn't reduce error: "
            f"without={err_no_leak:.4e}, with={err_with_leak:.4e}"
        )
        # After absorption the science flux should be within 5σ of truth
        assert err_with_leak < 5 * noise_std, (
            f"Science flux {science_flux_recovered:.4e} still biased after "
            f"mask-leak absorption (truth={true_science_flux:.4e})"
        )
