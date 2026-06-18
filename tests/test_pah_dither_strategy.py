"""
Tests for the PAH dithered-stacking simulator and Fisher strategy evaluator.

Tier 1 — simulator/kernel consistency (no fitting):
    bandpass registry integrity, kernel vs direct-integration cross-check,
    photo-z matrix properties, covariance limiting cases, CRLB monotonicity
    and saturation, stagger sub-additivity with shared-source covariance.

Tier 2 — GLS injection recovery (no MCMC):
    amplitude and continuum recovery at high SNR, the photo-z-matched
    kernel's unbiasedness, and the negative control showing that ignoring
    photo-z smearing in the kernel biases the blended 7.7+8.6 amplitude.
"""

import numpy as np
import pytest

from simstack4.pah_bandpass import MIPS_24, MIPS_70, get_bandpass
from simstack4.pah_dither import (
    DitherScheme,
    NoiseModel,
    TruthSpectrum,
    compute_pz_matrix,
    fisher_for_scheme,
    injection_recovery_sweep,
    make_dndz,
    shared_fraction_matrix,
    simulate_dithered_fluxes,
)
from simstack4.pah_spectrum import (
    build_design_matrix,
    solve_linear_amplitudes,
    warm_continuum_kernel,
)

PARAM_TOL = 0.20  # fractional tolerance on recovered amplitudes


@pytest.fixture(scope="module")
def truth():
    return TruthSpectrum()


@pytest.fixture(scope="module")
def scheme():
    return DitherScheme.uniform(z_min=0.5, z_max=3.5, dz=0.15, n_stagger=3)


# ---------------------------------------------------------------------------
# Tier 1 — bandpasses
# ---------------------------------------------------------------------------


class TestBandpasses:
    def test_mips24_matches_pah_model_arrays(self):
        """Registry copy must not drift from the frozen pah_model arrays."""
        from simstack4 import pah_model

        assert np.array_equal(MIPS_24.lam_um, pah_model._MIPS24_LAM)
        assert np.array_equal(MIPS_24.resp, pah_model._MIPS24_RESP)

    def test_mips70_bandpass_sane(self):
        assert MIPS_70.lam_um[0] > 45.0 and MIPS_70.lam_um[-1] < 120.0
        assert np.all(np.diff(MIPS_70.lam_um) > 0)
        assert MIPS_70.resp.max() == pytest.approx(1.0)
        assert MIPS_70.resp.min() >= 0.0
        assert MIPS_70.norm > 0
        assert 68.0 < MIPS_70.lam_eff < 76.0

    def test_get_bandpass_unknown_raises(self):
        with pytest.raises(KeyError):
            get_bandpass("PACS_100")

    def test_rest_coverage(self):
        lo, hi = MIPS_24.rest_coverage(2.0)
        assert lo == pytest.approx(18.005 / 3.0)
        assert hi == pytest.approx(32.207 / 3.0)


# ---------------------------------------------------------------------------
# Tier 1 — kernel vs direct integration (the two independent code paths)
# ---------------------------------------------------------------------------


class TestKernelConsistency:
    def test_kernel_matches_direct_integration(self, truth):
        """pz @ per-group curves == direct full-spectrum bandpass integral."""
        z_grid = np.linspace(0.5, 3.5, 150)
        pz = np.eye(len(z_grid))  # delta bins → sharp-z comparison
        K = build_design_matrix(pz, z_grid)
        W = warm_continuum_kernel(pz, z_grid, T_w=truth.T_warm, beta_w=truth.beta_warm)
        A = truth.amplitudes()
        for b, band in enumerate(("MIPS_24", "MIPS_70")):
            model = truth.continuum_amp * (W[:, b] + K[:, b, :] @ A)
            direct = truth.band_flux_curve(z_grid, band)
            assert np.allclose(model, direct, rtol=1e-2), band

    def test_feature_sweep_redshifts(self):
        """Group kernels peak where (1+z)·λ_feature crosses the band center."""
        z_grid = np.linspace(0.3, 3.8, 500)
        pz = np.eye(len(z_grid))
        K = build_design_matrix(pz, z_grid)
        # 7.7+8.6 complex through MIPS 24: z_peak ≈ 24/7.9 − 1 ≈ 2.0
        z_pk = z_grid[np.argmax(K[:, 0, 1])]
        assert 1.6 < z_pk < 2.3
        # 12.7 through MIPS 24: z_peak ≈ 24/12.7 − 1 ≈ 0.9
        z_pk = z_grid[np.argmax(K[:, 0, 3])]
        assert 0.7 < z_pk < 1.2


# ---------------------------------------------------------------------------
# Tier 1 — photo-z matrix
# ---------------------------------------------------------------------------


class TestPzMatrix:
    def test_rows_normalized(self, scheme):
        pz, _ = compute_pz_matrix(scheme, make_dndz(), sigma_z0=0.02)
        assert np.allclose(pz.sum(axis=1), 1.0)

    def test_smearing_broadens_rows(self, scheme):
        """Effective row width grows with sigma_z0."""
        widths = []
        for sz0 in (0.001, 0.05):
            pz, zg = compute_pz_matrix(scheme, make_dndz(), sigma_z0=sz0)
            i = len(pz) // 4
            mu = pz[i] @ zg
            widths.append(np.sqrt(pz[i] @ (zg - mu) ** 2))
        assert widths[1] > 1.5 * widths[0]

    def test_outlier_pedestal(self, scheme):
        """f_cat > 0 puts probability mass far from the bin."""
        pz0, zg = compute_pz_matrix(scheme, make_dndz(), sigma_z0=0.01, f_cat=0.0)
        pz1, _ = compute_pz_matrix(scheme, make_dndz(), sigma_z0=0.01, f_cat=0.2)
        i = 0  # lowest-z bin; far mass = high-z half of the grid
        far = zg > 2.0
        assert pz1[i, far].sum() > pz0[i, far].sum() + 0.01


# ---------------------------------------------------------------------------
# Tier 1 — simulator
# ---------------------------------------------------------------------------


class TestSimulator:
    def test_zero_amplitude_zero_noise_is_continuum(self, scheme):
        """With no features and no noise the stack is the pure continuum."""
        flat = TruthSpectrum(amp0=np.full(5, 1e-30))
        sim = simulate_dithered_fluxes(
            scheme, flat, n_total=150_000, sigma_z0=0.005, noise_scale=0.0, seed=2
        )
        df = sim["df"]
        pz, zg = compute_pz_matrix(scheme, make_dndz(), sigma_z0=0.005)
        W = warm_continuum_kernel(pz, zg)
        good = df["n_sources"].to_numpy() > 500
        f24 = df["MIPS_24"].to_numpy()[good]
        w24 = W[good, 0] * flat.continuum_amp
        assert np.allclose(f24, w24, rtol=0.05)

    def test_dataframe_schema(self, scheme, truth):
        sim = simulate_dithered_fluxes(scheme, truth, n_total=20_000, seed=0)
        df = sim["df"]
        for col in (
            "run_id",
            "zbin_id",
            "prop_bin_id",
            "z_lo",
            "z_hi",
            "z_mid",
            "n_sources",
            "MIPS_24",
            "MIPS_24_err",
            "MIPS_70",
            "MIPS_70_err",
        ):
            assert col in df.columns, col
        assert len(df) == scheme.n_zbins
        assert 0 in sim["cov"]
        n_flat = scheme.n_zbins * len(scheme.bands)
        assert sim["cov"][0].shape == (n_flat, n_flat)

    def test_property_bins_scale_amplitudes(self):
        """beta_mass shifts the injected amplitudes between property bins."""
        props = [
            {"log_M_star": 9.5, "log_sigma_sfr": 0.0},
            {"log_M_star": 11.0, "log_sigma_sfr": 0.0},
        ]
        truth = TruthSpectrum(beta_mass=0.3)
        scheme = DitherScheme.uniform(property_bins=props)
        sim = simulate_dithered_fluxes(scheme, truth, n_total=40_000, seed=0)
        a = sim["true_params"]["A_per_prop"]
        assert np.all(a[1] > a[0])
        assert a[1, 0] / a[0, 0] == pytest.approx(10 ** (0.3 * 1.5))


# ---------------------------------------------------------------------------
# Tier 1 — shared-source covariance
# ---------------------------------------------------------------------------


class TestSharedSourceCovariance:
    def test_identical_runs_fully_correlated(self):
        edges = np.arange(0.5, 3.51, 0.25)
        scheme = DitherScheme(runs=[edges, edges.copy()])
        frac = shared_fraction_matrix(scheme, make_dndz())
        n = len(edges) - 1
        # bin i of run 0 and bin i of run 1 select identical sources
        for i in range(n):
            assert frac[i, n + i] == pytest.approx(frac[i, i], rel=1e-6)
        cov = NoiseModel().covariance(np.diag(frac) * 1e5, frac * 1e5)
        sig = np.sqrt(np.diag(cov))
        corr_pair = cov[0, 2 * n] / (sig[0] * sig[2 * n])  # row-major (bin, band)
        assert corr_pair == pytest.approx(1.0, abs=1e-6)

    def test_same_run_bins_uncorrelated(self, scheme):
        frac = shared_fraction_matrix(scheme, make_dndz())
        nb0 = len(scheme.runs[0]) - 1
        block = frac[:nb0, :nb0]
        assert np.allclose(block - np.diag(np.diag(block)), 0.0)

    def test_covariance_positive_semidefinite(self, scheme):
        frac = shared_fraction_matrix(scheme, make_dndz())
        cov = NoiseModel().covariance(np.diag(frac) * 1e5, frac * 1e5)
        eig = np.linalg.eigvalsh(cov)
        assert eig.min() > -1e-10 * eig.max()

    def test_simulated_shared_counts_match_expectation(self, scheme, truth):
        """Exact per-source sharing tracks the analytic overlap fractions."""
        n_total = 200_000
        sim = simulate_dithered_fluxes(
            scheme, truth, n_total=n_total, sigma_z0=0.001, seed=5
        )
        frac = shared_fraction_matrix(scheme, make_dndz())
        n_src = sim["df"][sim["df"]["prop_bin_id"] == 0]["n_sources"].to_numpy()
        # analytic occupancy is normalized inside [z_min, z_max]; the
        # simulator pool extends z_margin beyond it, so allow that dilution
        in_range = n_src.sum() / (np.diag(frac).sum() * n_total)
        assert np.allclose(
            n_src, np.diag(frac) * n_total * in_range, rtol=0.10, atol=50
        )


# ---------------------------------------------------------------------------
# Tier 1 — Fisher / CRLB behavior
# ---------------------------------------------------------------------------


class TestFisher:
    def test_crlb_improves_with_finer_bins(self, truth):
        """Coarse single-run binning is strictly worse for detection."""
        fr_coarse = fisher_for_scheme(DitherScheme.uniform(dz=0.40, n_stagger=1), truth)
        fr_fine = fisher_for_scheme(DitherScheme.uniform(dz=0.10, n_stagger=1), truth)
        assert np.all(fr_fine.crlb_flux < fr_coarse.crlb_flux)

    def test_crlb_saturates_below_kernel_width(self, truth):
        """The broad bandpass sets the resolution floor: dz « kernel width
        buys almost nothing."""
        fr_a = fisher_for_scheme(DitherScheme.uniform(dz=0.05, n_stagger=1), truth)
        fr_b = fisher_for_scheme(DitherScheme.uniform(dz=0.02, n_stagger=1), truth)
        assert np.all(fr_b.crlb_flux > 0.95 * fr_a.crlb_flux)

    def test_photoz_degrades_crlb(self, truth):
        fr_sharp = fisher_for_scheme(
            DitherScheme.uniform(dz=0.10, n_stagger=1), truth, sigma_z0=0.003
        )
        fr_smear = fisher_for_scheme(
            DitherScheme.uniform(dz=0.10, n_stagger=1), truth, sigma_z0=0.10
        )
        assert np.all(fr_smear.crlb_flux >= fr_sharp.crlb_flux)
        assert fr_smear.crlb_flux[1] > 1.05 * fr_sharp.crlb_flux[1]

    def test_stagger_info_gain_subadditive(self, truth):
        """Staggered runs reuse sources: the SNR gain from n_stagger=1→3 at
        fixed dz must fall far short of the independent-sample sqrt(3)."""
        f1 = fisher_for_scheme(DitherScheme.uniform(dz=0.15, n_stagger=1), truth)
        f3 = fisher_for_scheme(DitherScheme.uniform(dz=0.15, n_stagger=3), truth)
        gain = f3.snr / f1.snr
        assert np.all(gain < 1.35)
        assert np.all(gain > 0.95)  # but never hurts

    def test_stagger_rescues_coarse_bins(self, truth):
        """Staggering matters when the base bins are coarser than the kernel."""
        f1 = fisher_for_scheme(DitherScheme.uniform(dz=0.40, n_stagger=1), truth)
        f4 = fisher_for_scheme(DitherScheme.uniform(dz=0.40, n_stagger=4), truth)
        assert f4.snr.min() > 1.2 * f1.snr.min()

    def test_property_split_costs_sqrt(self, truth):
        """Splitting into M property bins inflates per-bin CRLB ~ sqrt(M)."""
        props = [{"log_M_star": m, "log_sigma_sfr": 0.0} for m in (9.5, 10.5, 11.2)]
        f1 = fisher_for_scheme(DitherScheme.uniform(), truth, n_total=120_000)
        f3 = fisher_for_scheme(
            DitherScheme.uniform(property_bins=props), truth, n_total=120_000
        )
        ratio = f3.crlb_flux / f1.crlb_flux
        assert np.allclose(ratio, np.sqrt(3.0), rtol=0.15)


# ---------------------------------------------------------------------------
# Tier 2 — GLS injection recovery
# ---------------------------------------------------------------------------


def _fit_simulation(scheme, truth, sim, kernel_sigma_z0, ridge=0.0):
    pz, zg = compute_pz_matrix(scheme, make_dndz(), sigma_z0=kernel_sigma_z0)
    K = build_design_matrix(pz, zg, scheme.bands, truth.features, truth.feature_groups)
    W = warm_continuum_kernel(
        pz, zg, scheme.bands, T_w=truth.T_warm, beta_w=truth.beta_warm
    )
    sub = sim["df"][sim["df"]["prop_bin_id"] == 0]
    F = sub[list(scheme.bands)].to_numpy()
    return solve_linear_amplitudes(F, K, W, cov=sim["cov"][0], ridge=ridge)


class TestInjectionRecovery:
    def test_injection_recovery_lstsq(self, scheme, truth):
        """High-SNR recovery of all feature groups within PARAM_TOL."""
        sim = simulate_dithered_fluxes(
            scheme, truth, n_total=400_000, sigma_z0=0.01, seed=11
        )
        res = _fit_simulation(scheme, truth, sim, kernel_sigma_z0=0.01)
        A_true = truth.amplitudes()
        assert np.all(
            np.abs(res.A / A_true - 1.0) < PARAM_TOL
        ), f"A={res.A}, true={A_true}"

    def test_continuum_amplitude_recovery(self, scheme, truth):
        sim = simulate_dithered_fluxes(
            scheme, truth, n_total=400_000, sigma_z0=0.01, seed=13
        )
        res = _fit_simulation(scheme, truth, sim, kernel_sigma_z0=0.01)
        assert abs(res.C / truth.continuum_amp - 1.0) < 3.0 * res.C_err

    def test_77_86_grouped_recovery(self, scheme, truth):
        """The blended complex is the best-measured group."""
        sim = simulate_dithered_fluxes(
            scheme, truth, n_total=400_000, sigma_z0=0.01, seed=17
        )
        res = _fit_simulation(scheme, truth, sim, kernel_sigma_z0=0.01)
        frac_err = res.A_err / res.A
        assert np.argmin(frac_err) == 1
        assert abs(res.A[1] / truth.amplitudes()[1] - 1.0) < 0.10

    def test_errors_calibrated(self, truth):
        """Reported GLS errors track the Monte-Carlo scatter and the CRLB."""
        df = injection_recovery_sweep(
            [DitherScheme.uniform(dz=0.15, n_stagger=2)],
            truth,
            n_realizations=10,
            n_total=150_000,
            seed=23,
        )
        # scatter within a factor ~2 of the CRLB, biases subdominant
        assert np.all(df["scatter"] < 2.0 * df["crlb"])
        assert np.all(df["scatter"] > 0.3 * df["crlb"])
        assert np.all(np.abs(df["bias"]) < 2.0 * df["scatter"])

    def test_photoz_matched_kernel_unbiased(self, truth):
        """Strong smearing handled in the kernel keeps recovery unbiased."""
        scheme = DitherScheme.uniform(dz=0.10, n_stagger=2)
        biases = []
        for seed in (31, 32, 33):
            sim = simulate_dithered_fluxes(
                scheme, truth, n_total=300_000, sigma_z0=0.05, seed=seed
            )
            res = _fit_simulation(scheme, truth, sim, kernel_sigma_z0=0.05)
            biases.append(res.A[1] / truth.amplitudes()[1] - 1.0)
        assert abs(np.mean(biases)) < 0.10

    def test_photoz_ignored_kernel_biased(self, truth):
        """NEGATIVE CONTROL: fitting smeared data with a sharp kernel
        badly misattributes the diluted modulation between groups. Run
        nearly noise-free so the kernel mismatch is the only error."""
        scheme = DitherScheme.uniform(dz=0.10, n_stagger=2)
        sim = simulate_dithered_fluxes(
            scheme, truth, n_total=500_000, sigma_z0=0.06, noise_scale=0.01, seed=41
        )
        res_m = _fit_simulation(scheme, truth, sim, kernel_sigma_z0=0.06)
        res_i = _fit_simulation(scheme, truth, sim, kernel_sigma_z0=0.001)
        rel_m = np.abs(res_m.A / truth.amplitudes() - 1.0)
        rel_i = np.abs(res_i.A / truth.amplitudes() - 1.0)
        assert rel_m.max() < 0.06  # matched kernel: percent-level
        assert rel_i.max() > 0.30  # sharp kernel: catastrophic
        assert res_i.chi2 / res_i.dof > 10.0 * res_m.chi2 / res_m.dof
        # the blended 7.7+8.6 complex is biased LOW (diluted contrast)
        assert res_i.A[1] < res_m.A[1]
