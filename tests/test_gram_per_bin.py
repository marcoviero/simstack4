"""
Test that the Gram-matrix + PSF-stamping per_bin approach produces
the same results as the naive per_bin (full layer rebuild + lstsq).
"""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy import linalg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def gaussian_psf_layer(map_shape, source_positions, fwhm_pix, mean_subtract=True):
    """Build a PSF-convolved layer the naive way (FFT via gaussian_filter)."""
    sigma = fwhm_pix / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    delta_map = np.zeros(map_shape, dtype=np.float64)
    for y, x in source_positions:
        if 0 <= y < map_shape[0] and 0 <= x < map_shape[1]:
            delta_map[y, x] += 1.0
    convolved = gaussian_filter(delta_map, sigma=sigma, mode="constant")
    if mean_subtract:
        convolved -= np.mean(convolved)
    return convolved.ravel()


def make_psf_kernel(fwhm_pix):
    """Build a discrete Gaussian PSF kernel matching _build_psf_kernel."""
    sigma = fwhm_pix / 2.3548200
    half = int(np.ceil(3.0 * sigma))
    y, x = np.mgrid[-half : half + 1, -half : half + 1]
    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel /= kernel.sum()
    return kernel, half


def stamp_psf_cropped(src_rows, src_cols, psf_kernel, psf_half,
                      flat_to_crop, map_shape, n_crop):
    """Replicate _stamp_psf_cropped logic standalone."""
    n_rows, n_cols = map_shape
    layer = np.zeros(n_crop, dtype=np.float64)

    threshold = psf_kernel.max() * 1e-4
    ky, kx = np.where(psf_kernel > threshold)
    kvals = psf_kernel[ky, kx]
    k_dr = ky - psf_half
    k_dc = kx - psf_half
    n_kpix = len(kvals)

    all_rows = src_rows[:, None] + k_dr[None, :]
    all_cols = src_cols[:, None] + k_dc[None, :]

    valid = (
        (all_rows >= 0) & (all_rows < n_rows)
        & (all_cols >= 0) & (all_cols < n_cols)
    )

    all_flat = np.where(valid, all_rows * n_cols + all_cols, 0)
    all_crop = flat_to_crop[all_flat.ravel()].reshape(all_flat.shape)
    valid &= all_crop >= 0

    v_mask = valid.ravel()
    v_crop_idx = all_crop.ravel()[v_mask]
    v_vals = np.broadcast_to(
        kvals[None, :], (len(src_rows), n_kpix)
    ).ravel()[v_mask]

    np.add.at(layer, v_crop_idx, v_vals)
    return layer


def solve_for_fluxes(layer_matrix, observed):
    result = linalg.lstsq(layer_matrix.T, observed)
    return result[0]


def solve_gram(G, h):
    try:
        return np.linalg.solve(G, h)
    except np.linalg.LinAlgError:
        x, _, _, _ = np.linalg.lstsq(G, h, rcond=None)
        return x


# ---------------------------------------------------------------------------
# Test fixture: synthetic populations on a small map
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_setup():
    """Create a synthetic map with 4 populations for testing."""
    rng = np.random.RandomState(12345)
    map_shape = (128, 128)
    fwhm_pix = 5.0
    n_full = map_shape[0] * map_shape[1]
    n_pop = 4
    true_fluxes = np.array([1.0, 0.5, 0.3, 0.8])

    # Place sources randomly
    source_positions_per_pop = []
    for k in range(n_pop):
        n_src = rng.randint(40, 80)
        rows = rng.randint(10, map_shape[0] - 10, size=n_src)
        cols = rng.randint(10, map_shape[1] - 10, size=n_src)
        source_positions_per_pop.append(list(zip(rows, cols)))

    # Build base layers (via gaussian_filter — the "ground truth" method)
    base_layers = np.zeros((n_pop, n_full))
    for k in range(n_pop):
        base_layers[k] = gaussian_psf_layer(
            map_shape, source_positions_per_pop[k], fwhm_pix, mean_subtract=True
        )

    # Observed map = true signal + noise
    noise = rng.normal(0, 0.1, n_full)
    observed = base_layers.T @ true_fluxes + noise

    # Build crop mask: circles around all sources (simplified)
    sigma = fwhm_pix / 2.3548200
    radius = max(3.0, fwhm_pix)
    mask_2d = np.zeros(map_shape, dtype=bool)
    for positions in source_positions_per_pop:
        for y, x in positions:
            y_lo = max(0, int(y - radius))
            y_hi = min(map_shape[0], int(y + radius) + 1)
            x_lo = max(0, int(x - radius))
            x_hi = min(map_shape[1], int(x + radius) + 1)
            yy, xx = np.ogrid[y_lo:y_hi, x_lo:x_hi]
            dist = np.sqrt((xx - x)**2 + (yy - y)**2)
            mask_2d[y_lo:y_hi, x_lo:x_hi][dist <= radius] = True

    flat_indices = np.where(mask_2d.ravel())[0]
    n_crop = len(flat_indices)

    flat_to_crop = np.full(n_full, -1, dtype=np.int32)
    flat_to_crop[flat_indices] = np.arange(n_crop, dtype=np.int32)

    # Crop
    cropped_layers = base_layers[:, flat_indices]
    cropped_obs = observed[flat_indices]
    cropped_obs -= np.mean(cropped_obs)

    # PSF kernel
    psf_kernel, psf_half = make_psf_kernel(fwhm_pix)

    return {
        "map_shape": map_shape,
        "fwhm_pix": fwhm_pix,
        "n_pop": n_pop,
        "source_positions_per_pop": source_positions_per_pop,
        "base_layers": base_layers,         # (n_pop, n_full) full map
        "cropped_layers": cropped_layers,   # (n_pop, n_crop)
        "cropped_obs": cropped_obs,
        "true_fluxes": true_fluxes,
        "flat_to_crop": flat_to_crop,
        "flat_indices": flat_indices,
        "n_crop": n_crop,
        "psf_kernel": psf_kernel,
        "psf_half": psf_half,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGramMatrixSolveEquivalence:
    """Gram matrix solve G x = h should equal lstsq on the full matrix."""

    def test_flux_recovery(self, synthetic_setup):
        s = synthetic_setup
        # lstsq
        x_lstsq = solve_for_fluxes(s["cropped_layers"], s["cropped_obs"])
        # Gram
        G = s["cropped_layers"] @ s["cropped_layers"].T
        h = s["cropped_layers"] @ s["cropped_obs"]
        x_gram = solve_gram(G, h)

        np.testing.assert_allclose(x_gram, x_lstsq, atol=1e-10)


class TestPSFStampingAccuracy:
    """PSF stamping should closely match gaussian_filter convolution."""

    def test_stamp_matches_fft(self, synthetic_setup):
        """Stamped layer on cropped pixels ≈ FFT-convolved layer on cropped pixels."""
        s = synthetic_setup
        k = 0
        positions = s["source_positions_per_pop"][k]
        rows = np.array([p[0] for p in positions])
        cols = np.array([p[1] for p in positions])

        # FFT approach (ground truth)
        layer_fft = s["base_layers"][k]  # full map, mean-subtracted
        cropped_fft = layer_fft[s["flat_indices"]]

        # PSF stamping approach
        layer_stamp = stamp_psf_cropped(
            rows, cols, s["psf_kernel"], s["psf_half"],
            s["flat_to_crop"], s["map_shape"], s["n_crop"],
        )
        # Mean-subtract to match FFT convention
        n_full = s["map_shape"][0] * s["map_shape"][1]
        layer_stamp -= len(rows) / n_full

        # The gaussian_filter and discrete kernel won't match exactly
        # (edge handling, discretisation), but should be close.
        corr = np.corrcoef(cropped_fft, layer_stamp)[0, 1]
        assert corr > 0.99, f"Correlation = {corr:.4f}, expected > 0.99"

        # Check magnitude is similar
        ratio = np.std(layer_stamp) / np.std(cropped_fft)
        assert 0.9 < ratio < 1.1, f"Amplitude ratio = {ratio:.3f}"


class TestPerBinGramVsNaive:
    """
    The Gram-matrix per_bin should produce the same flux_k_total as
    the naive rebuild-everything approach.
    """

    def test_identical_flux_samples(self, synthetic_setup):
        """For each iteration, flux_A + flux_B should match between methods."""
        s = synthetic_setup
        n_pop = s["n_pop"]
        cache = s["cropped_layers"]  # (n_pop, n_crop)
        obs = s["cropped_obs"]
        split_fraction = 0.5
        n_iter = 10
        seed = 42

        G_base = cache @ cache.T
        h_base = cache @ obs

        for k in range(n_pop):
            positions_k = s["source_positions_per_pop"][k]
            n_src = len(positions_k)
            n_A = int(n_src * split_fraction)

            for iteration in range(n_iter):
                rng = np.random.RandomState(seed + k * n_iter + iteration)
                perm = rng.permutation(n_src)

                positions_A = [positions_k[i] for i in perm[:n_A]]
                positions_B = [positions_k[i] for i in perm[n_A:]]

                # ---- NAIVE: rebuild + lstsq ----
                layers_naive = []
                for j in range(n_pop):
                    if j == k:
                        layer_A_naive = gaussian_psf_layer(
                            s["map_shape"], positions_A, s["fwhm_pix"]
                        )
                        layer_B_naive = gaussian_psf_layer(
                            s["map_shape"], positions_B, s["fwhm_pix"]
                        )
                        layers_naive.append(layer_A_naive[s["flat_indices"]])
                        layers_naive.append(layer_B_naive[s["flat_indices"]])
                    else:
                        layers_naive.append(cache[j])

                matrix_naive = np.array(layers_naive)
                x_naive = solve_for_fluxes(matrix_naive, obs)
                flux_naive = x_naive[k] + x_naive[k + 1]

                # ---- GRAM: stamp + rank-2 update ----
                rows_A = np.array([p[0] for p in positions_A])
                cols_A = np.array([p[1] for p in positions_A])

                layer_A_stamp = stamp_psf_cropped(
                    rows_A, cols_A,
                    s["psf_kernel"], s["psf_half"],
                    s["flat_to_crop"], s["map_shape"], s["n_crop"],
                )
                n_full = s["map_shape"][0] * s["map_shape"][1]
                layer_A_stamp -= len(rows_A) / n_full
                layer_B_stamp = cache[k] - layer_A_stamp

                # Build modified Gram system
                n_layers = n_pop
                n_new = n_layers + 1
                A_dot_base = cache @ layer_A_stamp
                B_dot_base = cache @ layer_B_stamp
                A_dot_A = np.dot(layer_A_stamp, layer_A_stamp)
                B_dot_B = np.dot(layer_B_stamp, layer_B_stamp)
                A_dot_B = np.dot(layer_A_stamp, layer_B_stamp)
                h_A = np.dot(layer_A_stamp, obs)
                h_B = np.dot(layer_B_stamp, obs)

                G_new = np.empty((n_new, n_new))
                h_new = np.empty(n_new)

                old_keep = np.concatenate(
                    [np.arange(k), np.arange(k + 1, n_layers)]
                )
                new_keep = np.concatenate(
                    [np.arange(k), np.arange(k + 2, n_new)]
                )

                G_new[np.ix_(new_keep, new_keep)] = G_base[
                    np.ix_(old_keep, old_keep)
                ]
                h_new[new_keep] = h_base[old_keep]

                G_new[k, new_keep] = A_dot_base[old_keep]
                G_new[new_keep, k] = A_dot_base[old_keep]
                G_new[k, k] = A_dot_A
                G_new[k, k + 1] = A_dot_B
                G_new[k + 1, k] = A_dot_B
                G_new[k + 1, new_keep] = B_dot_base[old_keep]
                G_new[new_keep, k + 1] = B_dot_base[old_keep]
                G_new[k + 1, k + 1] = B_dot_B
                h_new[k] = h_A
                h_new[k + 1] = h_B

                x_gram = solve_gram(G_new, h_new)
                flux_gram = x_gram[k] + x_gram[k + 1]

                # The two methods won't be EXACTLY equal because:
                # - PSF stamping uses a discrete kernel vs gaussian_filter FFT
                # - Mean subtraction is approximate
                # But they should be very close.
                np.testing.assert_allclose(
                    flux_gram, flux_naive, atol=0.02, rtol=0.05,
                    err_msg=f"pop={k}, iter={iteration}: "
                    f"naive={flux_naive:.4f} vs gram={flux_gram:.4f}",
                )


class TestPerBinErrorConsistency:
    """Per-bin errors from the Gram approach should be consistent."""

    def test_errors_are_reasonable(self, synthetic_setup):
        """Per-bin bootstrap errors should be positive and finite."""
        s = synthetic_setup
        n_pop = s["n_pop"]
        cache = s["cropped_layers"]
        obs = s["cropped_obs"]
        G_base = cache @ cache.T
        h_base = cache @ obs
        split_fraction = 0.5
        n_iter = 100
        seed = 42

        # Full solve
        x_full = solve_gram(G_base, h_base)

        for k in range(n_pop):
            positions_k = s["source_positions_per_pop"][k]
            n_src = len(positions_k)
            n_A = int(n_src * split_fraction)

            flux_k_samples = []

            for iteration in range(n_iter):
                rng = np.random.RandomState(seed + k * n_iter + iteration)
                perm = rng.permutation(n_src)
                positions_A = [positions_k[i] for i in perm[:n_A]]
                rows_A = np.array([p[0] for p in positions_A])
                cols_A = np.array([p[1] for p in positions_A])

                layer_A = stamp_psf_cropped(
                    rows_A, cols_A,
                    s["psf_kernel"], s["psf_half"],
                    s["flat_to_crop"], s["map_shape"], s["n_crop"],
                )
                n_full = s["map_shape"][0] * s["map_shape"][1]
                layer_A -= len(rows_A) / n_full
                layer_B = cache[k] - layer_A

                n_layers = n_pop
                n_new = n_layers + 1
                A_dot_base = cache @ layer_A
                B_dot_base = cache @ layer_B

                G_new = np.empty((n_new, n_new))
                h_new = np.empty(n_new)

                old_keep = np.concatenate(
                    [np.arange(k), np.arange(k + 1, n_layers)]
                )
                new_keep = np.concatenate(
                    [np.arange(k), np.arange(k + 2, n_new)]
                )
                G_new[np.ix_(new_keep, new_keep)] = G_base[
                    np.ix_(old_keep, old_keep)
                ]
                h_new[new_keep] = h_base[old_keep]
                G_new[k, new_keep] = A_dot_base[old_keep]
                G_new[new_keep, k] = A_dot_base[old_keep]
                G_new[k, k] = np.dot(layer_A, layer_A)
                G_new[k, k + 1] = np.dot(layer_A, layer_B)
                G_new[k + 1, k] = G_new[k, k + 1]
                G_new[k + 1, new_keep] = B_dot_base[old_keep]
                G_new[new_keep, k + 1] = B_dot_base[old_keep]
                G_new[k + 1, k + 1] = np.dot(layer_B, layer_B)
                h_new[k] = np.dot(layer_A, obs)
                h_new[k + 1] = np.dot(layer_B, obs)

                x_new = solve_gram(G_new, h_new)
                flux_k_samples.append(x_new[k] + x_new[k + 1])

            err = np.std(flux_k_samples, ddof=1)

            # Per-bin error should be positive
            assert err > 0, f"Pop {k}: error is zero"
            # Per-bin error should be finite and much less than the flux
            assert np.isfinite(err), f"Pop {k}: error is not finite"
            assert err < abs(x_full[k]) * 5, (
                f"Pop {k}: error ({err:.4f}) implausibly large vs "
                f"flux ({x_full[k]:.4f})"
            )
            # Variance across iterations should be small (stable method)
            cv = err / abs(np.mean(flux_k_samples))
            assert cv < 0.5, (
                f"Pop {k}: coefficient of variation ({cv:.3f}) too high"
            )
