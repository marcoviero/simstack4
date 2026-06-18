"""
Tests for the PAH forward-model deconvolution (pah_spectrum).

Tier 1 — core math (no noise, no fitting loops):
    group-weight rules, design-matrix shapes, exact noiseless GLS solution,
    NaN masking.

Tier 2 — GLS behavior:
    ridge regularization of deliberately collinear groups, and the
    MIPS 24-only vs joint 24+70 degeneracy (the 70 µm band pins the warm
    continuum, shrinking the continuum amplitude error and the
    feature-ratio bounds).

Tier 3 (MCMC, PAHSpectrumModel) lives further down and uses the
MCMC_FAST/STD presets from the dust-evolution test conventions.
"""

import numpy as np
import pytest

from simstack4.pah_dither import (
    DitherScheme,
    TruthSpectrum,
    compute_pz_matrix,
    fisher_for_scheme,
    make_dndz,
    simulate_dithered_fluxes,
)
from simstack4.pah_spectrum import (
    DEFAULT_FEATURES,
    DEFAULT_GROUPS,
    build_design_matrix,
    feature_band_curves,
    group_weights,
    solve_linear_amplitudes,
    warm_band_curve,
    warm_continuum_kernel,
)

PARAM_TOL = 0.20


@pytest.fixture(scope="module")
def grids():
    z_grid = np.linspace(0.4, 3.6, 200)
    n_bins = 40
    edges = np.linspace(0.5, 3.5, n_bins + 1)
    pz = np.zeros((n_bins, len(z_grid)))
    for i in range(n_bins):
        sel = (z_grid >= edges[i]) & (z_grid < edges[i + 1])
        pz[i, sel] = 1.0
        pz[i] /= pz[i].sum()
    return z_grid, pz


# ---------------------------------------------------------------------------
# Tier 1 — core math
# ---------------------------------------------------------------------------


class TestGroupWeights:
    def test_single_feature_group_unit_weight(self):
        w = group_weights(DEFAULT_FEATURES, [[3]])  # 11.3, catalog strength 0.30
        assert np.allclose(w[0], [1.0])

    def test_multi_feature_group_normalized_to_strongest(self):
        w = group_weights(DEFAULT_FEATURES, [[1, 2]])  # 7.7 (0.4577), 8.6 (0.6089)
        assert w[0].max() == pytest.approx(1.0)
        assert w[0][0] == pytest.approx(0.4577 / 0.6089, rel=1e-6)

    def test_zero_strength_group_falls_back_to_ones(self):
        feats = [(6.2, 0.0, 0.19), (7.7, 0.0, 0.70)]
        w = group_weights(feats, [[0, 1]])
        assert np.allclose(w[0], 1.0)


class TestDesignMatrix:
    def test_shapes(self, grids):
        z_grid, pz = grids
        K = build_design_matrix(pz, z_grid)
        W = warm_continuum_kernel(pz, z_grid)
        assert K.shape == (len(pz), 2, len(DEFAULT_GROUPS))
        assert W.shape == (len(pz), 2)

    def test_kernel_values_bounded(self, grids):
        """Band-averaged response to a unit-peak feature is in [0, 1]."""
        z_grid, pz = grids
        K = build_design_matrix(pz, z_grid)
        assert np.all(K >= 0.0)
        assert np.all(K <= 1.0)

    def test_custom_grouping_changes_columns(self, grids):
        z_grid, pz = grids
        K_split = build_design_matrix(
            pz, z_grid, feature_groups=[[0], [1], [2], [3], [4], [5], [6]]
        )
        assert K_split.shape[-1] == 7

    def test_curves_zero_outside_band(self):
        """A feature that never enters the band gives a null column."""
        z_grid = np.linspace(0.5, 1.0, 20)  # 6.2 µm needs z ≈ 2.2–4.2 for MIPS 24
        T = feature_band_curves(z_grid, "MIPS_24", feature_groups=[[0]])
        assert np.all(T < 1e-6)

    def test_warm_curve_fades_with_z_at_24(self):
        """At 24 µm, higher z probes bluer rest wavelengths — deeper into
        the Wien tail — so the warm continuum fades steeply with z."""
        z_grid = np.linspace(0.5, 3.5, 50)
        w = warm_band_curve(z_grid, "MIPS_24", T_w=60.0, beta_w=1.5)
        assert np.all(np.diff(w) < 0)
        assert w[0] > 100.0 * w[-1]


class TestLinearSolve:
    def test_noiseless_solution_exact(self, grids):
        z_grid, pz = grids
        K = build_design_matrix(pz, z_grid)
        W = warm_continuum_kernel(pz, z_grid)
        A_true = np.array([0.5, 2.0, 0.3, 0.8, 0.4])
        C_true = 3.0
        F = C_true * W + np.einsum("ibg,g->ib", K, C_true * A_true)
        res = solve_linear_amplitudes(F, K, W, sigma=np.full_like(F, 1e-3))
        assert res.C == pytest.approx(C_true, rel=1e-8)
        assert np.allclose(res.A, A_true, rtol=1e-8)
        assert res.chi2 == pytest.approx(0.0, abs=1e-12)

    def test_nan_masking(self, grids):
        z_grid, pz = grids
        K = build_design_matrix(pz, z_grid)
        W = warm_continuum_kernel(pz, z_grid)
        A_true = np.array([0.5, 2.0, 0.3, 0.8, 0.4])
        F = 1.0 * W + np.einsum("ibg,g->ib", K, A_true)
        F[5:15, 1] = np.nan
        res = solve_linear_amplitudes(F, K, W, sigma=np.full_like(F, 1e-3))
        assert res.mask.sum() == F.size - 10
        assert res.dof == F.size - 10 - 6
        assert np.allclose(res.A, A_true, rtol=1e-8)

    def test_sigma_and_cov_paths_agree(self, grids):
        z_grid, pz = grids
        K = build_design_matrix(pz, z_grid)
        W = warm_continuum_kernel(pz, z_grid)
        A_true = np.array([0.5, 2.0, 0.3, 0.8, 0.4])
        rng = np.random.default_rng(0)
        F = W + np.einsum("ibg,g->ib", K, A_true) + rng.normal(0, 1e-3, W.shape)
        sigma = np.full_like(F, 1e-3)
        res_s = solve_linear_amplitudes(F, K, W, sigma=sigma)
        res_c = solve_linear_amplitudes(F, K, W, cov=np.diag(np.full(F.size, 1e-6)))
        assert np.allclose(res_s.A, res_c.A)
        assert np.allclose(res_s.A_err, res_c.A_err)

    def test_both_sigma_and_cov_raises(self, grids):
        z_grid, pz = grids
        K = build_design_matrix(pz, z_grid)
        W = warm_continuum_kernel(pz, z_grid)
        F = W.copy()
        with pytest.raises(ValueError):
            solve_linear_amplitudes(F, K, W, sigma=np.ones_like(F), cov=np.eye(F.size))


# ---------------------------------------------------------------------------
# Tier 2 — conditioning and band leverage
# ---------------------------------------------------------------------------


class TestConditioning:
    def test_ridge_stabilizes_collinear_split(self, grids):
        """Splitting 7.7 and 8.6 into separate groups makes their columns
        near-collinear; ridge shrinks the variance explosion."""
        z_grid, pz = grids
        split = [[0], [1], [2], [3], [4], [5, 6]]
        K = build_design_matrix(pz, z_grid, feature_groups=split)
        W = warm_continuum_kernel(pz, z_grid)
        A_true = np.array([0.5, 2.0, 1.5, 0.3, 0.8, 0.4])
        rng = np.random.default_rng(1)
        F = W + np.einsum("ibg,g->ib", K, A_true) + rng.normal(0, 5e-4, W.shape)
        sigma = np.full_like(F, 5e-4)
        res0 = solve_linear_amplitudes(F, K, W, sigma=sigma, ridge=0.0)
        res1 = solve_linear_amplitudes(F, K, W, sigma=sigma, ridge=1e-2)
        # the split pair is far noisier than in the grouped fit, and
        # ridge damps it
        assert res1.A_err[1] < res0.A_err[1]
        assert res1.A_err[2] < res0.A_err[2]

    def test_split_77_86_strongly_correlated(self, grids):
        z_grid, pz = grids
        split = [[1], [2]]
        K = build_design_matrix(pz, z_grid, feature_groups=split)
        W = warm_continuum_kernel(pz, z_grid)
        F = W + np.einsum("ibg,g->ib", K, np.array([2.0, 1.5]))
        res = solve_linear_amplitudes(F, K, W, sigma=np.full_like(F, 1e-3))
        corr = res.A_cov[0, 1] / (res.A_err[0] * res.A_err[1])
        assert corr < -0.3  # blended pair anti-correlates

    def test_joint_bands_beat_24_only(self):
        """MIPS 70 pins the warm continuum: C_err and the feature-ratio
        CRLBs shrink when the bands are fit jointly."""
        truth = TruthSpectrum()
        sch24 = DitherScheme.uniform(bands=("MIPS_24",))
        sch_joint = DitherScheme.uniform(bands=("MIPS_24", "MIPS_70"))
        fr24 = fisher_for_scheme(sch24, truth)
        frj = fisher_for_scheme(sch_joint, truth)
        assert frj.C_err < 0.7 * fr24.C_err
        assert np.all(frj.crlb <= fr24.crlb * 1.001)

    def test_70_only_loses_77(self):
        """MIPS 70 alone never sees the 7.7+8.6 complex below z≈4.6."""
        truth = TruthSpectrum()
        fr70 = fisher_for_scheme(DitherScheme.uniform(bands=("MIPS_70",)), truth)
        frj = fisher_for_scheme(
            DitherScheme.uniform(bands=("MIPS_24", "MIPS_70")), truth
        )
        i77 = 1  # A(7.7+8.6)
        assert fr70.crlb_flux[i77] > 10.0 * frj.crlb_flux[i77]


class TestKernelRoundTripWithSimulator:
    def test_full_pipeline_recovery(self):
        """Simulator → matched kernel → GLS recovers the truth (smoke-level
        duplicate of the Tier-2 dither tests, exercised from this module's
        entry points)."""
        truth = TruthSpectrum()
        scheme = DitherScheme.uniform(dz=0.15, n_stagger=2)
        sim = simulate_dithered_fluxes(
            scheme, truth, n_total=300_000, sigma_z0=0.01, seed=3
        )
        pz, zg = compute_pz_matrix(scheme, make_dndz(), sigma_z0=0.01)
        K = build_design_matrix(pz, zg, scheme.bands)
        W = warm_continuum_kernel(pz, zg, scheme.bands)
        sub = sim["df"][sim["df"]["prop_bin_id"] == 0]
        res = solve_linear_amplitudes(
            sub[list(scheme.bands)].to_numpy(), K, W, cov=sim["cov"][0]
        )
        assert np.all(np.abs(res.A / truth.amplitudes() - 1.0) < PARAM_TOL)


# ---------------------------------------------------------------------------
# Tier 3 — PAHSpectrumModel (MAP + MCMC)
# ---------------------------------------------------------------------------

from simstack4.pah_spectrum import PAHSpectrumModel  # noqa: E402

MCMC_FAST = dict(n_steps=400, n_burn=150, n_walkers=32, progress=False, verbose=False)

MASS_PROPS = [{"log_M_star": m, "log_sigma_sfr": 0.0} for m in (9.5, 10.5, 11.2)]


@pytest.fixture(scope="module")
def mass_sim():
    """Three mass bins, beta_mass=0.35, moderate noise — shared by Tier 3."""
    truth = TruthSpectrum(beta_mass=0.35)
    scheme = DitherScheme.uniform(dz=0.15, n_stagger=2, property_bins=MASS_PROPS)
    sim = simulate_dithered_fluxes(
        scheme, truth, n_total=450_000, sigma_z0=0.01, seed=5
    )
    return {"truth": truth, "scheme": scheme, "sim": sim}


class TestFitLstsq:
    def test_recovers_amplitudes_and_Tw(self, mass_sim):
        model = PAHSpectrumModel(sigma_z0=0.01)
        res = model.fit_lstsq(
            mass_sim["sim"]["df"],
            cov=mass_sim["sim"]["cov"],
            scheme=mass_sim["scheme"],
        )
        truth = mass_sim["truth"]
        assert abs(res.theta_global[0] - truth.T_warm) < 3.0 * max(
            res.theta_err[0], 1.0
        )
        for i, prop in enumerate(MASS_PROPS):
            A_true = truth.amplitudes(prop)
            # 16.4+17 is the weakest constraint; PARAM_TOL on the rest
            rel = np.abs(res.A[i][:4] / A_true[:4] - 1.0)
            assert np.all(rel < PARAM_TOL), (i, rel)

    def test_fix_T_w_skips_optimization(self, mass_sim):
        model = PAHSpectrumModel(sigma_z0=0.01)
        res = model.fit_lstsq(
            mass_sim["sim"]["df"],
            cov=mass_sim["sim"]["cov"],
            scheme=mass_sim["scheme"],
            fix_T_w=60.0,
        )
        assert res.theta_global[0] == 60.0
        assert res.theta_err[0] == 0.0

    def test_per_bin_results_attached(self, mass_sim):
        model = PAHSpectrumModel(sigma_z0=0.01)
        res = model.fit_lstsq(
            mass_sim["sim"]["df"],
            cov=mass_sim["sim"]["cov"],
            scheme=mass_sim["scheme"],
            fix_T_w=60.0,
        )
        assert res.per_bin is not None and len(res.per_bin) == len(MASS_PROPS)
        assert res.chi2_red < 3.0


class TestFitMcmc:
    def test_beta_mass_recovery(self, mass_sim):
        """The pooled evolution slope is recovered within 3σ (and grossly,
        within 0.15 absolute)."""
        model = PAHSpectrumModel(sigma_z0=0.01)
        res = model.fit_mcmc(
            mass_sim["sim"]["df"],
            cov=mass_sim["sim"]["cov"],
            scheme=mass_sim["scheme"],
            fix_beta_sigma=True,
            seed=1,
            **MCMC_FAST,
        )
        i = res.param_names.index("beta_mass")
        assert abs(res.theta_global[i] - 0.35) < max(3.0 * res.theta_err[i], 0.15)
        assert res.acceptance_fraction > 0.15

    def test_log_a0_recovery(self, mass_sim):
        model = PAHSpectrumModel(sigma_z0=0.01)
        res = model.fit_mcmc(
            mass_sim["sim"]["df"],
            cov=mass_sim["sim"]["cov"],
            scheme=mass_sim["scheme"],
            fix_beta_sigma=True,
            seed=2,
            **MCMC_FAST,
        )
        A0_true = mass_sim["truth"].amp0
        for g in range(4):  # skip weakest group
            fit = 10.0 ** res.theta_global[1 + g]
            assert abs(fit / A0_true[g] - 1.0) < PARAM_TOL, res.labels[g]

    def test_outlier_fraction_robustness(self):
        """f_cat=5% in the data, matched in the kernel: recovery stays
        consistent with the reported errors (a continuum-amplitude draw
        moves all ratios coherently, so test in σ units, not fractions)."""
        truth = TruthSpectrum()
        scheme = DitherScheme.uniform(dz=0.15, n_stagger=2)
        sim = simulate_dithered_fluxes(
            scheme, truth, n_total=300_000, sigma_z0=0.02, f_cat=0.05, seed=9
        )
        model = PAHSpectrumModel(sigma_z0=0.02, f_cat=0.05)
        res = model.fit_lstsq(sim["df"], cov=sim["cov"], scheme=scheme)
        pull = (res.A[0][:4] - truth.amplitudes()[:4]) / res.A_err[0][:4]
        assert np.all(np.abs(pull) < 3.5), pull

    def test_full_cov_vs_diagonal_errors(self, mass_sim):
        """Dropping the shared-source correlations changes the GLS weights;
        the full-covariance chi² stays near 1 while errors remain finite."""
        model = PAHSpectrumModel(sigma_z0=0.01)
        res_full = model.fit_lstsq(
            mass_sim["sim"]["df"],
            cov=mass_sim["sim"]["cov"],
            scheme=mass_sim["scheme"],
            fix_T_w=60.0,
        )
        res_diag = model.fit_lstsq(
            mass_sim["sim"]["df"],
            cov=None,
            scheme=mass_sim["scheme"],
            fix_T_w=60.0,
        )
        assert res_full.chi2_red < 3.0
        assert np.all(np.isfinite(res_diag.A_err))
        # diagonal weighting reports overconfident (smaller) errors
        assert np.median(res_diag.A_err / res_full.A_err) < 1.0


class TestPseudoSpectrum:
    def test_pseudo_spectrum_schema_and_peak(self, mass_sim):
        """The continuum-normalized excess peaks near the 7.7 µm complex."""
        model = PAHSpectrumModel(sigma_z0=0.01)
        res = model.fit_lstsq(
            mass_sim["sim"]["df"],
            cov=mass_sim["sim"]["cov"],
            scheme=mass_sim["scheme"],
            fix_T_w=60.0,
        )
        spec = model.pseudo_spectrum(
            mass_sim["sim"]["df"], res, scheme=mass_sim["scheme"]
        )
        for col in (
            "prop_bin_id",
            "band",
            "z_mid",
            "lam_rest",
            "ratio",
            "ratio_err",
            "excess_snr",
        ):
            assert col in spec.columns
        # the raw ratio diverges where the continuum → 0 (high z at 24 µm);
        # the excess SIGNIFICANCE peaks at the 7.7 µm complex
        sub = spec[(spec.prop_bin_id == 2) & (spec.band == "MIPS_24")]
        lam_pk = sub.loc[sub.excess_snr.idxmax(), "lam_rest"]
        assert 6.5 < lam_pk < 9.5
