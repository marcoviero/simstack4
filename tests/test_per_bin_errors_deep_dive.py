"""
Stress tests for per-bin bootstrap error estimation.

Key finding: per-bin bootstrap only captures source-assignment variance
(shuffling which sources go to A vs B). It does NOT capture map noise.
The correct total error is:
    σ_total = √(σ_per_bin² + σ_formal²)

These tests validate that the combined error gives correct statistical
coverage across many independent noise realizations.
"""

import numpy as np
import pytest
from scipy import linalg
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_psf_layer(map_shape, source_positions, fwhm_pix, mean_subtract=True):
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


def run_per_bin_bootstrap(layer_matrix, observed, source_positions_per_pop,
                          map_shape, fwhm_pix, n_iterations=200,
                          split_fraction=0.5, seed=42):
    n_pop = layer_matrix.shape[0]
    fluxes, formal_errors = solve_for_fluxes(layer_matrix, observed)
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
            shuffled_idx = rng.permutation(n_sources)
            positions_A = [positions_k[i] for i in shuffled_idx[:n_A]]
            positions_B = [positions_k[i] for i in shuffled_idx[n_A:]]

            layers = []
            for j in range(n_pop):
                if j == k:
                    layers.append(gaussian_psf_layer(map_shape, positions_A, fwhm_pix))
                    layers.append(gaussian_psf_layer(map_shape, positions_B, fwhm_pix))
                else:
                    layers.append(layer_matrix[j])
            split_matrix = np.array(layers)
            split_fluxes, _ = solve_for_fluxes(split_matrix, observed)
            flux_k_samples.append(split_fluxes[k] + split_fluxes[k + 1])

        per_bin_errors[k] = np.std(flux_k_samples, ddof=1)

    return fluxes, per_bin_errors, formal_errors


# ---------------------------------------------------------------------------
# Coverage calibration engine
# ---------------------------------------------------------------------------

def run_coverage_experiment(map_shape, fwhm, positions_list, true_fluxes,
                            noise_std, n_realizations=200,
                            n_bootstrap=100, seed_base=0):
    n_pop = len(positions_list)
    layers = [gaussian_psf_layer(map_shape, pos, fwhm) for pos in positions_list]
    layer_matrix = np.array(layers)
    noiseless = layer_matrix.T @ true_fluxes

    pulls_perbin = np.zeros((n_realizations, n_pop))
    pulls_formal = np.zeros((n_realizations, n_pop))
    pulls_combined = np.zeros((n_realizations, n_pop))

    for r in range(n_realizations):
        rng = np.random.default_rng(seed_base + r)
        observed = noiseless + rng.normal(0, noise_std, len(noiseless))

        fluxes, pb_err, formal_err = run_per_bin_bootstrap(
            layer_matrix, observed, positions_list, map_shape, fwhm,
            n_iterations=n_bootstrap, seed=seed_base + r * 1000)

        combined_err = np.sqrt(pb_err**2 + formal_err**2)

        for k in range(n_pop):
            residual = fluxes[k] - true_fluxes[k]
            if pb_err[k] > 0:
                pulls_perbin[r, k] = residual / pb_err[k]
            if formal_err[k] > 0:
                pulls_formal[r, k] = residual / formal_err[k]
            if combined_err[k] > 0:
                pulls_combined[r, k] = residual / combined_err[k]

    return pulls_perbin, pulls_formal, pulls_combined


# ---------------------------------------------------------------------------
# FAST TESTS (< 5 seconds each)
# ---------------------------------------------------------------------------

class TestFastSanity:

    def test_combined_error_larger_than_either(self):
        map_shape = (64, 64)
        fwhm = 3.0
        rng = np.random.default_rng(42)
        pos_a = [(rng.integers(5, 28), rng.integers(5, 28)) for _ in range(20)]
        pos_b = [(rng.integers(36, 59), rng.integers(36, 59)) for _ in range(20)]
        layers = [gaussian_psf_layer(map_shape, p, fwhm) for p in [pos_a, pos_b]]
        lm = np.array(layers)
        obs = lm.T @ np.array([0.005, 0.010]) + rng.normal(0, 0.001, 64*64)

        _, pb, formal = run_per_bin_bootstrap(lm, obs, [pos_a, pos_b],
                                              map_shape, fwhm, n_iterations=50)
        combined = np.sqrt(pb**2 + formal**2)
        for k in range(2):
            assert combined[k] >= pb[k]
            assert combined[k] >= formal[k]

    def test_errors_positive_finite(self):
        map_shape = (64, 64)
        fwhm = 4.0
        rng = np.random.default_rng(77)
        positions = [
            [(rng.integers(5, 59), rng.integers(5, 59)) for _ in range(25)]
            for _ in range(3)
        ]
        layers = [gaussian_psf_layer(map_shape, p, fwhm) for p in positions]
        lm = np.array(layers)
        obs = lm.T @ np.array([0.005, 0.010, 0.015]) + rng.normal(0, 0.001, 64*64)

        _, pb, _ = run_per_bin_bootstrap(lm, obs, positions, map_shape, fwhm,
                                         n_iterations=50)
        for k in range(3):
            assert pb[k] > 0 and np.isfinite(pb[k])

    def test_zero_noise_small_errors(self):
        map_shape = (64, 64)
        fwhm = 3.0
        rng = np.random.default_rng(99)
        pos = [(rng.integers(5, 59), rng.integers(5, 59)) for _ in range(30)]
        layer = gaussian_psf_layer(map_shape, pos, fwhm)
        lm = np.array([layer])
        obs = lm.T @ np.array([0.01])

        fluxes, pb, _ = run_per_bin_bootstrap(lm, obs, [pos], map_shape, fwhm,
                                               n_iterations=100)
        assert abs(fluxes[0] - 0.01) < 1e-10
        assert 0 < pb[0] < 0.001

    def test_error_scales_with_noise(self):
        map_shape = (64, 64)
        fwhm = 3.0
        rng = np.random.default_rng(42)
        pos = [(rng.integers(5, 59), rng.integers(5, 59)) for _ in range(30)]
        layer = gaussian_psf_layer(map_shape, pos, fwhm)
        lm = np.array([layer])

        errs = []
        for noise in [0.001, 0.002]:
            obs = lm.T @ np.array([0.01]) + np.random.default_rng(42).normal(0, noise, 64*64)
            _, pb, formal = run_per_bin_bootstrap(lm, obs, [pos], map_shape, fwhm,
                                                   n_iterations=100)
            errs.append(np.sqrt(pb[0]**2 + formal[0]**2))

        ratio = errs[1] / errs[0]
        assert 1.3 < ratio < 3.0, f"Ratio {ratio:.2f}"


# ---------------------------------------------------------------------------
# COVERAGE TESTS (~30s each)
# ---------------------------------------------------------------------------

class TestCoverageCalibration:

    def _check(self, pulls, k, label, min_cov=0.45, max_cov=0.90):
        cov = np.mean(np.abs(pulls[:, k]) < 1.0)
        std = np.std(pulls[:, k])
        print(f"  {label} pop {k}: coverage={cov:.1%}, pull_std={std:.2f}")
        assert min_cov < cov < max_cov, (
            f"{label} pop {k}: coverage {cov:.1%} outside [{min_cov:.0%},{max_cov:.0%}]")

    def test_separated_combined_coverage(self):
        rng = np.random.default_rng(100)
        pos_a = [(rng.integers(5, 25), rng.integers(5, 25)) for _ in range(25)]
        pos_b = [(rng.integers(40, 60), rng.integers(40, 60)) for _ in range(25)]
        _, _, pulls_c = run_coverage_experiment(
            (64, 64), 3.0, [pos_a, pos_b], np.array([0.008, 0.012]),
            0.001, n_realizations=150, n_bootstrap=80, seed_base=0)
        for k in range(2):
            self._check(pulls_c, k, "separated-combined")

    def test_overlapping_combined_coverage(self):
        rng = np.random.default_rng(200)
        positions = [
            [(rng.integers(10, 50), rng.integers(10, 50)) for _ in range(20)]
            for _ in range(3)
        ]
        _, _, pulls_c = run_coverage_experiment(
            (64, 64), 5.0, positions, np.array([0.005, 0.010, 0.015]),
            0.001, n_realizations=150, n_bootstrap=80, seed_base=5000)
        for k in range(3):
            self._check(pulls_c, k, "overlapping-combined")

    def test_high_noise_coverage(self):
        rng = np.random.default_rng(300)
        pos_a = [(rng.integers(5, 30), rng.integers(5, 30)) for _ in range(30)]
        pos_b = [(rng.integers(35, 60), rng.integers(35, 60)) for _ in range(30)]
        _, _, pulls_c = run_coverage_experiment(
            (64, 64), 3.0, [pos_a, pos_b], np.array([0.005, 0.010]),
            0.005, n_realizations=150, n_bootstrap=80, seed_base=10000)
        for k in range(2):
            self._check(pulls_c, k, "highnoise-combined")

    def test_perbin_alone_undercounts(self):
        rng = np.random.default_rng(400)
        pos = [(rng.integers(5, 59), rng.integers(5, 59)) for _ in range(30)]
        pulls_pb, _, _ = run_coverage_experiment(
            (64, 64), 3.0, [pos], np.array([0.01]),
            0.002, n_realizations=150, n_bootstrap=80, seed_base=20000)
        std = np.std(pulls_pb[:, 0])
        cov = np.mean(np.abs(pulls_pb[:, 0]) < 1.0)
        print(f"  perbin-alone: pull_std={std:.2f}, coverage={cov:.1%}")
        assert std > 1.2, f"Pull std {std:.2f} — expected >1.2"


# ---------------------------------------------------------------------------
# Summary table (run standalone)
# ---------------------------------------------------------------------------

def print_summary_table():
    print("\n" + "=" * 95)
    print("PER-BIN BOOTSTRAP ERROR CALIBRATION SUMMARY")
    print("=" * 95)

    scenarios = [
        ("Separated, low noise",  (64,64), 3.0, 0.001, True),
        ("Separated, high noise", (64,64), 3.0, 0.005, True),
        ("Overlapping, low noise", (64,64), 5.0, 0.001, False),
        ("Overlapping, high noise",(64,64), 5.0, 0.005, False),
    ]

    print(f"\n{'Scenario':<28} {'Pop':>3} {'σ_pb':>9} {'σ_formal':>9} "
          f"{'σ_comb':>9} {'Cov_pb':>7} {'Cov_comb':>8} {'Pull_std':>9}")
    print("-" * 95)

    for name, map_shape, fwhm, noise, separated in scenarios:
        rng = np.random.default_rng(abs(hash(name)) % 2**31)
        if separated:
            pos_a = [(rng.integers(5, 25), rng.integers(5, 25)) for _ in range(25)]
            pos_b = [(rng.integers(40, 60), rng.integers(40, 60)) for _ in range(25)]
        else:
            pos_a = [(rng.integers(10, 50), rng.integers(10, 50)) for _ in range(25)]
            pos_b = [(rng.integers(15, 55), rng.integers(15, 55)) for _ in range(25)]

        true = np.array([0.008, 0.012])
        positions = [pos_a, pos_b]

        pulls_pb, _, pulls_c = run_coverage_experiment(
            map_shape, fwhm, positions, true, noise,
            n_realizations=100, n_bootstrap=60, seed_base=abs(hash(name)) % 10000)

        # Get one example for error magnitudes
        layers = [gaussian_psf_layer(map_shape, p, fwhm) for p in positions]
        lm = np.array(layers)
        obs = lm.T @ true + rng.normal(0, noise, map_shape[0]*map_shape[1])
        _, pb_err, f_err = run_per_bin_bootstrap(
            lm, obs, positions, map_shape, fwhm, n_iterations=60)

        for k in range(2):
            cov_pb = np.mean(np.abs(pulls_pb[:, k]) < 1.0)
            cov_c = np.mean(np.abs(pulls_c[:, k]) < 1.0)
            c_err = np.sqrt(pb_err[k]**2 + f_err[k]**2)

            print(f"{name:<28} {k:>3} {pb_err[k]:>9.2e} {f_err[k]:>9.2e} "
                  f"{c_err:>9.2e} {cov_pb:>6.1%} {cov_c:>7.1%} "
                  f"{np.std(pulls_c[:, k]):>9.2f}")

    print("\n" + "=" * 95)
    print("Target: Cov_comb ~ 68%, Pull_std ~ 1.0")
    print("Per-bin alone undercounts because it misses map noise.")
    print("Combined = sqrt(sigma_pb^2 + sigma_formal^2) corrects this.")
    print("=" * 95)


if __name__ == "__main__":
    print_summary_table()
