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
    FEATURES_86_CALIBRATED,
    FEATURES_CALIBRATED,
    build_design_matrix,
    feature_band_curves,
    feature_profile_area,
    get_bandpass,
    group_weights,
    rescale_feature_strength,
    solve_linear_amplitudes,
    warm_band_curve,
    warm_continuum_kernel,
)

PARAM_TOL = 0.20

# TruthSpectrum injects with DEFAULT_FEATURES/DEFAULT_GROUPS, while
# PAHSpectrumModel now defaults to the welded-at-physical configuration
# (FEATURES_86_CALIBRATED/PHYSICAL_GROUPS). These sim→fit round-trips test the
# GLS recovery, not the grouping convention, so they pin the fitter to the
# simulator's template explicitly.
LEGACY_TEMPLATE = {
    "features": DEFAULT_FEATURES,
    "feature_groups": DEFAULT_GROUPS,
}


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


class TestCalibratedFeatureStrengths:
    """The 2026-07-23 within-group calibration of the 7.7+8.6 blend."""

    def test_rescale_touches_only_the_named_strength(self):
        out = rescale_feature_strength(DEFAULT_FEATURES, 2, 0.25)
        assert out[2] == (DEFAULT_FEATURES[2][0], 0.25, DEFAULT_FEATURES[2][2])
        for j, feat in enumerate(out):
            if j != 2:
                assert feat == DEFAULT_FEATURES[j]
        # the input list must not be mutated
        assert DEFAULT_FEATURES[2][1] == pytest.approx(0.6089)

    def test_default_blend_is_8p6_dominated(self):
        """Guards the motivation: the frozen catalog makes 8.6 the group peak."""
        w = group_weights(DEFAULT_FEATURES, [[1, 2]])[0]
        assert w[1] > w[0]  # 8.6 outweighs 7.7 — backwards vs observed spectra

    def test_calibrated_blend_is_7p7_dominated_at_the_fixed_ratio(self):
        w = group_weights(FEATURES_86_CALIBRATED, [[1, 2]])[0]
        assert w[0] == pytest.approx(1.0)  # 7.7 is now the group peak
        assert w[1] == pytest.approx(
            0.5, abs=1e-3
        )  # fixed drop-in 8.6/7.7 (not a measurement)

    def test_calibrated_list_differs_from_default_only_at_8p6(self):
        assert len(FEATURES_86_CALIBRATED) == len(DEFAULT_FEATURES)
        for j, (cal, dflt) in enumerate(
            zip(FEATURES_86_CALIBRATED, DEFAULT_FEATURES, strict=True)
        ):
            if j == 2:
                assert cal[0] == dflt[0] and cal[2] == dflt[2]
                assert cal[1] < dflt[1]
            else:
                assert cal == dflt

    def test_calibrated_blend_reweights_the_band_integrated_template(self):
        """The calibration must re-shape the blend's kernel, not just relabel it.

        group_weights renormalizes to the group's strongest member, so the
        calibration moves weight from the 8.6 side to the 7.7 side: the
        band-integrated template must rise where MIPS 24 samples rest 7.7 µm
        (z≈2.1) relative to where it samples 8.6 µm (z≈1.8).
        """
        lam_eff = get_bandpass("MIPS_24").lam_eff
        z_77, z_86 = lam_eff / 7.7 - 1.0, lam_eff / 8.6 - 1.0
        z = np.array([z_86, z_77])
        k_def = feature_band_curves(z, "MIPS_24", DEFAULT_FEATURES, [[1, 2]])[:, 0]
        k_cal = feature_band_curves(z, "MIPS_24", FEATURES_86_CALIBRATED, [[1, 2]])[
            :, 0
        ]
        assert (k_cal[1] / k_def[1]) > (k_cal[0] / k_def[0])


class TestCalibratedIntegratedRatios:
    """The 2026-07-25 calibration of the 11.3+12.7 blend.

    Literature band ratios are INTEGRATED; catalog strengths are unit-PEAK.
    Conflating the two is what left 12.7 asserted 8.6x too strong.
    """

    def _integ(self, feats, i, j, profile="drude"):
        return (feats[j][1] * feature_profile_area(feats[j], profile)) / (
            feats[i][1] * feature_profile_area(feats[i], profile)
        )

    def test_default_catalog_has_12p7_inverted_and_inflated(self):
        """Guards the motivation: the frozen catalog asserts 12.7 >> 11.3."""
        assert self._integ(DEFAULT_FEATURES, 3, 4) > 3.0

    def test_calibrated_matches_the_measured_integrated_ratio(self):
        """Hernan-Caballero+2020: R_int(12.7/11.2) = 0.377 +- 0.020.

        Checked against the NUMERICALLY EXACT area, so this is the round trip
        on _strength_for_integrated_ratio's FWHM-ratio shortcut.
        """
        assert self._integ(FEATURES_CALIBRATED, 3, 4) == pytest.approx(0.377, rel=1e-3)

    def test_integrated_ratio_is_profile_independent(self):
        """The area/FWHM constant cancels, so gaussian must agree with drude."""
        d = self._integ(FEATURES_CALIBRATED, 3, 4, "drude")
        g = self._integ(FEATURES_CALIBRATED, 3, 4, "gaussian")
        assert d == pytest.approx(g, rel=2e-3)

    def test_peak_ratio_is_NOT_the_integrated_ratio(self):
        """The trap itself: 12.7 is 1.9x wider, so peak != integrated."""
        peak = FEATURES_CALIBRATED[4][1] / FEATURES_CALIBRATED[3][1]
        assert peak == pytest.approx(0.377 * 0.24 / 0.45, rel=1e-3)
        assert abs(peak - 0.377) > 0.15

    def test_calibrated_keeps_the_8p6_fix_and_changes_only_12p7(self):
        for j, (cal, prev) in enumerate(
            zip(FEATURES_CALIBRATED, FEATURES_86_CALIBRATED, strict=True)
        ):
            if j == 4:
                assert cal[0] == prev[0] and cal[2] == prev[2]
                assert cal[1] < prev[1]
            else:
                assert cal == prev

    def test_model_defaults_to_the_fully_calibrated_list(self):
        assert PAHSpectrumModel().features == FEATURES_CALIBRATED


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
        """A feature that never enters the band gives a null column.

        Only exactly true for the Gaussian: the Drude default has power-law
        wings, so its column is small-but-finite out of band (which is the
        point of the profile — see TestDrudeProfile). Both are asserted.
        """
        z_grid = np.linspace(0.5, 1.0, 20)  # 6.2 µm needs z ≈ 2.2–4.2 for MIPS 24
        T_g = feature_band_curves(
            z_grid, "MIPS_24", feature_groups=[[0]], profile="gaussian"
        )
        assert np.all(T_g < 1e-6)
        T_d = feature_band_curves(z_grid, "MIPS_24", feature_groups=[[0]])
        in_band = feature_band_curves(
            np.array([2.85]), "MIPS_24", feature_groups=[[0]]
        ).max()
        assert np.all(T_d < 0.05 * in_band)  # wings present, but well sub-dominant

    def test_warm_curve_fades_with_z_at_24(self):
        """At 24 µm, higher z probes bluer rest wavelengths — deeper into
        the Wien tail — so the warm continuum fades steeply with z."""
        z_grid = np.linspace(0.5, 3.5, 50)
        w = warm_band_curve(z_grid, "MIPS_24", T_w=60.0, beta_w=1.5)
        assert np.all(np.diff(w) < 0)
        assert w[0] > 100.0 * w[-1]


class TestDrudeProfile:
    """profile="drude" is the branch-12 default; gaussian is the systematic."""

    def test_default_profile_is_drude(self):
        """Guards the 2026-07-25 flip.

        Gaussian used to be the default "for backward compatibility", which
        meant every caller that forgot the argument silently got a different
        line shape from the fit. The notebooks' L_PAH conversion did exactly
        that for weeks. The default must now match what the fitter uses.
        """
        z_grid = np.linspace(0.5, 3.5, 80)
        a = feature_band_curves(z_grid, "MIPS_24")
        d = feature_band_curves(z_grid, "MIPS_24", profile="drude")
        np.testing.assert_array_equal(a, d)
        g = feature_band_curves(z_grid, "MIPS_24", profile="gaussian")
        assert not np.allclose(a, g)

    def test_every_public_entry_point_defaults_to_drude(self):
        """One forgotten default is all it takes; check them together."""
        import inspect

        from simstack4 import pah_spectrum as ps

        for fn in (
            ps.feature_profile_area,
            ps.feature_band_curves,
            ps.build_design_matrix,
            ps.plateau_band_curves,
            ps.feature_template_luminosity,
        ):
            sig = inspect.signature(fn)
            assert sig.parameters["profile"].default == "drude", fn.__name__
        assert (
            inspect.signature(ps.PAHSpectrumModel).parameters["profile"].default
            == "drude"
        )
        from simstack4.pah_dither import TruthSpectrum

        assert TruthSpectrum().profile == "drude"

    def test_unknown_profile_raises(self):
        with pytest.raises(ValueError, match="profile"):
            feature_band_curves(np.array([1.0]), "MIPS_24", profile="lorentz")

    def test_l_pah_integral_is_converged_and_untruncated(self):
        """L_PAH must not lose Drude wings to the integration range.

        Two failure modes, historically confused: grid *resolution* (never the
        problem) and range *truncation* (the real -1.5% loss at the old
        (3, 20) um default). Guard both against the defaults.
        """
        from simstack4 import pah_spectrum as ps

        r = [1.0, 0.6, 0.4]
        # A far-wider range is the reference. It is not "the truth" -- the Drude
        # blue wing contributes a constant per unit lambda in nu-space, so the
        # integral creeps as lam_min -> 0 -- but every defensible bracket of the
        # 3.3-17 um features agrees to <=0.3%, so that is the tolerance.
        ref = ps.feature_template_luminosity(
            1.5, 1e4, r, n_lam=200_000, lam_range=(0.5, 200.0)
        )
        default = ps.feature_template_luminosity(1.5, 1e4, r)
        assert default == pytest.approx(ref, rel=5e-3), "L_PAH integral truncated"

        # Resolution alone is converged at the default n_lam.
        fine = ps.feature_template_luminosity(1.5, 1e4, r, n_lam=50_000)
        assert default == pytest.approx(fine, rel=1e-5)

        # The old default is the -1.5% low value; keep it reproducible.
        old = ps.feature_template_luminosity(
            1.5, 1e4, r, lam_range=(3.0, 20.0), n_lam=400
        )
        assert old / ref == pytest.approx(0.985, abs=0.003)

        # The loss is common-mode: it must not tilt the mass slope. Across the
        # plausible spread of neutral-group ratios the bias varies <0.2%.
        biases = [
            ps.feature_template_luminosity(
                1.5, 1e4, [1.0, 0.6, rn], lam_range=(3.0, 20.0), n_lam=400
            )
            / ps.feature_template_luminosity(1.5, 1e4, [1.0, 0.6, rn])
            for rn in (0.15, 0.4, 0.9)
        ]
        assert max(biases) - min(biases) < 2e-3

    def test_drude_area_ratio(self):
        # Drude/Gaussian area at fixed peak+FWHM ≈ (π/2)/1.0645 ≈ 1.46
        for j in (0, 1, 4):  # 6.2, 7.7, 12.7
            a_g = feature_profile_area(DEFAULT_FEATURES[j], "gaussian")
            a_d = feature_profile_area(DEFAULT_FEATURES[j], "drude")
            fwhm = DEFAULT_FEATURES[j][2]
            assert a_g == pytest.approx(1.0645 * fwhm, rel=0.01)
            assert a_d / a_g == pytest.approx(1.46, rel=0.03)

    def test_drude_inband_peak_and_wing_floor(self):
        """Band integration does not wash the wings out: the in-band peak
        rises ×1.1–1.5 and a wing floor persists where the Gaussian is dead
        (the 2026-07-19 quantification these numbers come from)."""
        z_grid = np.linspace(0.2, 6.0, 400)
        g = feature_band_curves(
            z_grid, "MIPS_24", feature_groups=[[1, 2]], profile="gaussian"
        )[:, 0]
        d = feature_band_curves(
            z_grid, "MIPS_24", feature_groups=[[1, 2]], profile="drude"
        )[:, 0]
        assert 1.1 < d.max() / g.max() < 1.5
        wing = g < 0.01 * g.max()
        assert wing.any()
        assert d[wing].max() > 0.05 * d.max()

    def test_model_threads_profile_into_kernel(self):
        """PAHSpectrumModel(profile=...) must reach the design matrix."""
        import pandas as pd

        from simstack4.pah_spectrum import PAHSpectrumModel

        rows = []
        edges = np.round(np.arange(1.4, 2.6 + 1e-9, 0.15), 4)
        for zlo, zhi in zip(edges[:-1], edges[1:], strict=False):
            rows.append(
                {
                    "run_id": 0,
                    "z_lo": float(zlo),
                    "z_hi": float(zhi),
                    "z_mid": 0.5 * (zlo + zhi),
                    "prop_bin_id": 0,
                    "log_M_star": 10.5,
                    "MIPS_24": 1.0,
                    "MIPS_24_err": 0.1,
                }
            )
        df = pd.DataFrame(rows)
        kw = {"feature_groups": [[1, 2]], "bands": ("MIPS_24",), "sigma_z0": 0.01}
        K_g = PAHSpectrumModel(**kw, profile="gaussian")._prepare(
            df, None, None, None, None, None
        )["bins"][0]["K"]
        K_d = PAHSpectrumModel(**kw)._prepare(df, None, None, None, None, None)["bins"][
            0
        ][
            "K"
        ]  # drude by default
        # 7.7+8.6 is in-band over this z range: Drude kernel strictly larger.
        assert np.all(K_d >= K_g)
        assert K_d.max() / K_g.max() > 1.1


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
        model = PAHSpectrumModel(**LEGACY_TEMPLATE, sigma_z0=0.01)
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
        model = PAHSpectrumModel(**LEGACY_TEMPLATE, sigma_z0=0.01)
        res = model.fit_lstsq(
            mass_sim["sim"]["df"],
            cov=mass_sim["sim"]["cov"],
            scheme=mass_sim["scheme"],
            fix_T_w=60.0,
        )
        assert res.theta_global[0] == 60.0
        assert res.theta_err[0] == 0.0

    def test_per_bin_results_attached(self, mass_sim):
        model = PAHSpectrumModel(**LEGACY_TEMPLATE, sigma_z0=0.01)
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
        model = PAHSpectrumModel(**LEGACY_TEMPLATE, sigma_z0=0.01)
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
        model = PAHSpectrumModel(**LEGACY_TEMPLATE, sigma_z0=0.01)
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
        model = PAHSpectrumModel(**LEGACY_TEMPLATE, sigma_z0=0.02, f_cat=0.05)
        res = model.fit_lstsq(sim["df"], cov=sim["cov"], scheme=scheme)
        pull = (res.A[0][:4] - truth.amplitudes()[:4]) / res.A_err[0][:4]
        assert np.all(np.abs(pull) < 3.5), pull

    def test_full_cov_vs_diagonal_errors(self, mass_sim):
        """Dropping the shared-source correlations changes the GLS weights;
        the full-covariance chi² stays near 1 while errors remain finite."""
        model = PAHSpectrumModel(**LEGACY_TEMPLATE, sigma_z0=0.01)
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
        model = PAHSpectrumModel(**LEGACY_TEMPLATE, sigma_z0=0.01)
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
