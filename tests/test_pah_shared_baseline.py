"""Recovery tests for the shared-ratio PAH fit + smoothed main-sequence baseline.

These exercise the physical-baseline path added to ``PAHSpectrumModel``:
- ``smoothed_ms_baseline`` (Wien-tail baseline stabilization), and
- ``PAHSpectrumModel.fit_shared`` (shared feature ratios + per-bin amplitude
  against a cold-greybody baseline).

Synthetic data only — no FITS/catalogs.
"""

import numpy as np
import pandas as pd
import pytest

from simstack4.greybody import Greybody
from simstack4.pah_spectrum import PAHSpectrumModel, smoothed_ms_baseline

FEATURE_GROUPS = [[0], [1, 2], [4]]  # 6.2 | 7.7+8.6 | 12.7


def _tomographic_skeleton(mass_centers, z_edges):
    """Build a one-run tomographic df skeleton (one row per mass×z bin)."""
    rows = []
    for m, logM in enumerate(mass_centers):
        for zlo, zhi in zip(z_edges[:-1], z_edges[1:], strict=False):
            zmid = 0.5 * (zlo + zhi)
            rows.append(
                {
                    "run_id": 0,
                    "z_lo": round(float(zlo), 6),
                    "z_hi": round(float(zhi), 6),
                    "z_mid": zmid,
                    "prop_bin_id": m,
                    "log_M_star": float(logM),
                    "log_sigma_sfr": 0.0,
                    "MIPS_24": 1.0,
                    "MIPS_24_err": 1.0,  # placeholders
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["prop_bin_id", "run_id", "z_lo"])
        .reset_index(drop=True)
    )


@pytest.fixture
def model():
    return PAHSpectrumModel(
        feature_groups=FEATURE_GROUPS, bands=("MIPS_24",), sigma_z0=0.01, f_cat=0.0
    )


def test_fit_shared_recovers_injected_amplitudes(model):
    rng = np.random.default_rng(0)
    mass_centers = [10.25, 10.65, 11.0]
    z_edges = np.round(np.arange(0.5, 3.5 + 1e-9, 0.15), 4)

    df = _tomographic_skeleton(mass_centers, z_edges)
    # A decreasing cold baseline in absolute flux (mJy), same scale for all bins.
    df["f24_cold"] = 0.03 / (1.0 + df["z_mid"]) ** 2

    # Truth: rising PAH/continuum amplitude with mass, fixed group ratios.
    A_pah_true = np.array([0.8, 1.6, 2.6])  # alpha/C per bin
    r_true = np.array([1.0, 1.7, 4.0])  # r_0 ≡ 1
    C_true = np.array([0.02, 0.03, 0.05])  # continuum level per bin (mJy)

    # Use the model's own prep so the injected kernel matches the fit exactly.
    prep = model._prepare(df, None, None, None, None, None, baseline_col="f24_cold")
    for i, b in enumerate(prep["bins"]):
        sub_idx = df.index[df["prop_bin_id"] == b["m"]]
        # b["K"][:,0,:] rows align with df rows for this bin (no rows dropped here)
        K_rows = b["K"][:, 0, :]
        fc = b["f_cold"]
        med = float(np.median(fc))
        continuum = C_true[i] * fc / med
        alpha_i = A_pah_true[i] * C_true[i]
        excess = alpha_i * (K_rows @ r_true)
        flux = continuum + excess
        err = 0.03 * flux + 1e-5
        noisy = flux + rng.normal(0.0, err)
        df.loc[sub_idx, "MIPS_24"] = noisy
        df.loc[sub_idx, "MIPS_24_err"] = err

    res = model.fit_shared(df, baseline_col="f24_cold")
    assert res is not None
    assert res["valid"] == [0, 1, 2]

    A = np.array(res["A_pah"])
    r = np.array(res["r"])
    # Amplitudes recovered to ~20% and monotonic ordering preserved.
    assert np.all(np.isfinite(A))
    np.testing.assert_allclose(A, A_pah_true, rtol=0.25)
    assert A[0] < A[1] < A[2]
    # Shared ratios recovered (r_0 fixed at 1).
    assert r[0] == pytest.approx(1.0)
    np.testing.assert_allclose(r[1:], r_true[1:], rtol=0.3)
    # A[m,g] = A_pah[m]*r[g] consistency.
    np.testing.assert_allclose(res["A"], A[:, None] * r[None, :], rtol=1e-6)


def test_fit_shared_missing_baseline_raises(model):
    df = _tomographic_skeleton([10.5], np.round(np.arange(0.5, 3.5, 0.15), 4))
    with pytest.raises(ValueError):
        model.fit_shared(df, baseline_col="f24_cold")  # column absent


def test_smoothed_ms_baseline_reduces_scatter():
    rng = np.random.default_rng(1)
    z_edges = np.round(np.arange(0.5, 3.5 + 1e-9, 0.15), 4)
    mass_centers = [10.25, 10.65, 11.0]
    df = _tomographic_skeleton(mass_centers, z_edges)

    # True main-sequence relations; per-bin fit values scattered around them.
    T_true = (
        24.0 + 5.0 * df["z_mid"].to_numpy() - 2.5 * (df["log_M_star"].to_numpy() - 10.0)
    )
    logA_true = -36.0 + 1.0 * df["z_mid"].to_numpy()
    df["T_dust"] = T_true + rng.normal(0.0, 3.0, len(df))  # noisy individual fits
    df["log_amp"] = logA_true + rng.normal(0.0, 0.15, len(df))
    df["beta"] = 1.8
    df["tier"] = "A"

    gb = Greybody()
    df["f24_cold"] = [
        float(gb.greybody_model(np.array([24.0 / (1.0 + z)]), a, t, 1.8)[0])
        for z, a, t in zip(df["z_mid"], df["log_amp"], df["T_dust"], strict=False)
    ]

    out = smoothed_ms_baseline(df, baseline_col="f24_cold")

    # Original preserved; smoothed columns added.
    assert "f24_cold_raw" in out.columns
    np.testing.assert_array_equal(
        out["f24_cold_raw"].to_numpy(), df["f24_cold"].to_numpy()
    )
    assert "T_dust_smooth" in out.columns

    # Smoothing pulls T back toward the truth (less scatter than the noisy input).
    raw_scatter = float(np.std(df["T_dust"].to_numpy() - T_true))
    sm_scatter = float(np.std(out["T_dust_smooth"].to_numpy() - T_true))
    assert sm_scatter < raw_scatter
    # Predicted T stays in the physical clip range.
    assert out["T_dust_smooth"].between(15.0, 60.0).all()


def test_smoothed_ms_baseline_too_few_training_rows_noop():
    df = pd.DataFrame(
        {
            "z_mid": [1.0, 2.0],
            "log_M_star": [10.5, 11.0],
            "T_dust": [30.0, 35.0],
            "log_amp": [-35.0, -34.0],
            "beta": [1.8, 1.8],
            "tier": ["A", "B"],
            "f24_cold": [1e-3, 2e-3],
        }
    )
    out = smoothed_ms_baseline(df, baseline_col="f24_cold")
    # <4 training rows → baseline unchanged but raw column still recorded.
    np.testing.assert_array_equal(out["f24_cold"].to_numpy(), df["f24_cold"].to_numpy())
    assert "f24_cold_raw" in out.columns


# ---------------------------------------------------------------------------
# PAHFIT-style hot-dust ladder (fixed-T MBB rungs, non-negative amplitudes)
# ---------------------------------------------------------------------------


def _hot_synthetic(model_hot, hot_frac=0.0, seed=3):
    """Synthetic df through the model's own kernels, optionally + hot MBB.

    ``hot_frac``: peak in-band hot flux of the injected 200 K rung as a
    multiple of the bin's continuum level C_true.
    """
    rng = np.random.default_rng(seed)
    mass_centers = [10.25, 10.65, 11.0]
    z_edges = np.round(np.arange(0.5, 3.5 + 1e-9, 0.15), 4)
    df = _tomographic_skeleton(mass_centers, z_edges)
    df["f24_cold"] = 0.03 / (1.0 + df["z_mid"]) ** 2

    A_pah_true = np.array([0.8, 1.6, 2.6])
    r_true = np.array([1.0, 1.7, 4.0])
    C_true = np.array([0.02, 0.03, 0.05])

    prep = model_hot._prepare(df, None, None, None, None, None, baseline_col="f24_cold")
    from simstack4.pah_spectrum import _hot_columns

    hot_true = np.zeros((3, 2))
    for i, b in enumerate(prep["bins"]):
        sub_idx = df.index[df["prop_bin_id"] == b["m"]]
        K_rows = b["K"][:, 0, :]
        fc = b["f_cold"]
        med = float(np.median(fc))
        flux = C_true[i] * fc / med + A_pah_true[i] * C_true[i] * (K_rows @ r_true)
        if hot_frac:
            Hn = _hot_columns(b["H"][:, 0, :])
            hot_true[i, 1] = hot_frac * C_true[i]  # inject on the 200 K rung
            flux = flux + Hn @ hot_true[i]
        err = 0.03 * flux + 1e-5
        df.loc[sub_idx, "MIPS_24"] = flux + rng.normal(0.0, err)
        df.loc[sub_idx, "MIPS_24_err"] = err
    return df, A_pah_true, r_true, hot_true


def test_hot_ladder_null_leaves_pah_recovery_intact():
    """No injected hot flux → rung amplitudes ~0 and PAH recovery unchanged."""
    kw = {"feature_groups": FEATURE_GROUPS, "bands": ("MIPS_24",), "sigma_z0": 0.01}
    model_hot = PAHSpectrumModel(**kw, hot_ladder=(120.0, 200.0))
    df, A_true, r_true, _ = _hot_synthetic(model_hot, hot_frac=0.0)

    res = model_hot.fit_shared(df, baseline_col="f24_cold")
    assert res is not None
    assert res["hot_T"] == (120.0, 200.0)
    hot = np.asarray(res["hot_amp"])
    assert hot.shape == (3, 2)
    # Non-negativity honored; null injection → small vs the continuum level.
    assert np.all(hot[np.isfinite(hot)] >= -1e-12)
    assert np.nanmax(hot) < 0.3 * 0.02
    np.testing.assert_allclose(res["A_pah"], A_true, rtol=0.3)
    np.testing.assert_allclose(res["r"][1:], r_true[1:], rtol=0.35)


def test_hot_ladder_absorbs_injected_hot_component():
    """Injected 200 K flux lands in the ladder, not in the PAH amplitudes."""
    kw = {"feature_groups": FEATURE_GROUPS, "bands": ("MIPS_24",), "sigma_z0": 0.01}
    model_hot = PAHSpectrumModel(**kw, hot_ladder=(120.0, 200.0))
    df, A_true, r_true, hot_true = _hot_synthetic(model_hot, hot_frac=1.5)

    res_hot = model_hot.fit_shared(df, baseline_col="f24_cold")
    res_no = PAHSpectrumModel(**kw).fit_shared(df, baseline_col="f24_cold")
    assert res_hot is not None and res_no is not None

    # With the ladder: PAH amplitudes recovered, total hot flux recovered
    # (rung split is degenerate between neighboring temperatures — compare
    # the summed ladder flux, not per-rung amplitudes).
    np.testing.assert_allclose(res_hot["A_pah"], A_true, rtol=0.35)
    tot_rec = np.asarray(res_hot["hot_amp"]).sum(axis=1)
    tot_true = hot_true.sum(axis=1)
    np.testing.assert_allclose(tot_rec, tot_true, rtol=0.5)

    # Without it the injected hot flux biases the fit high somewhere: the
    # ladder fit must be a strictly better description of the data.
    assert res_hot["chi2"] < res_no["chi2"]
    bias_no = np.max(np.abs(np.asarray(res_no["A_pah"]) / A_true - 1.0))
    bias_hot = np.max(np.abs(np.asarray(res_hot["A_pah"]) / A_true - 1.0))
    assert bias_hot < bias_no


def test_hot_ladder_guard_on_unwired_paths():
    model_hot = PAHSpectrumModel(
        feature_groups=FEATURE_GROUPS, bands=("MIPS_24",), hot_ladder=(200.0,)
    )
    df = _tomographic_skeleton([10.5], np.round(np.arange(0.5, 2.0, 0.15), 4))
    df["f24_cold"] = 1e-3
    with pytest.raises(NotImplementedError):
        model_hot.fit_lstsq(df)
    with pytest.raises(NotImplementedError):
        model_hot.fit_evolving_mcmc(df, baseline_col="f24_cold")


def test_hot_ladder_evolving_path_consistent():
    """fit_evolving (slopes off) with the ladder matches the fit_shared story."""
    kw = {"feature_groups": FEATURE_GROUPS, "bands": ("MIPS_24",), "sigma_z0": 0.01}
    model_hot = PAHSpectrumModel(**kw, hot_ladder=(120.0, 200.0))
    df, A_true, r_true, hot_true = _hot_synthetic(model_hot, hot_frac=1.5)

    res = model_hot.fit_evolving(
        df, baseline_col="f24_cold", evolve_amp=False, evolve_ratios=False
    )
    assert res is not None
    np.testing.assert_allclose(res["A_pah"], A_true, rtol=0.35)
    tot_rec = np.asarray(res["hot_amp"]).sum(axis=1)
    np.testing.assert_allclose(tot_rec, hot_true.sum(axis=1), rtol=0.5)


def test_hot_ladder_under_baseline_envelope():
    """Envelope-consistent injection: features AND rungs dim with the source
    (feature_envelope="baseline"); ladder amplitudes still land on truth."""
    rng = np.random.default_rng(7)
    kw = {"feature_groups": FEATURE_GROUPS, "bands": ("MIPS_24",), "sigma_z0": 0.01}
    model_hot = PAHSpectrumModel(**kw, hot_ladder=(120.0, 200.0))

    mass_centers = [10.25, 10.65, 11.0]
    z_edges = np.round(np.arange(0.5, 3.5 + 1e-9, 0.15), 4)
    df = _tomographic_skeleton(mass_centers, z_edges)
    df["f24_cold"] = 0.03 / (1.0 + df["z_mid"]) ** 2

    A_pah_true = np.array([0.8, 1.6, 2.6])
    r_true = np.array([1.0, 1.7, 4.0])
    C_true = np.array([0.02, 0.03, 0.05])
    from simstack4.pah_spectrum import _hot_columns

    prep = model_hot._prepare(df, None, None, None, None, None, baseline_col="f24_cold")
    hot_true = np.zeros((3, 2))
    hot_peak_true = np.zeros(3)
    for i, b in enumerate(prep["bins"]):
        sub_idx = df.index[df["prop_bin_id"] == b["m"]]
        fc = b["f_cold"]
        med = float(np.median(fc))
        env = fc / med
        K_rows = b["K"][:, 0, :] * env[:, None]  # features dim with source
        Hn = _hot_columns(b["H"][:, 0, :]) * env[:, None]  # so do the rungs
        hot_true[i, 1] = 1.5 * C_true[i]
        hot_flux = Hn @ hot_true[i]
        hot_peak_true[i] = float(hot_flux.max())
        flux = (
            C_true[i] * env + A_pah_true[i] * C_true[i] * (K_rows @ r_true) + hot_flux
        )
        err = 0.03 * flux + 1e-5
        df.loc[sub_idx, "MIPS_24"] = flux + rng.normal(0.0, err)
        df.loc[sub_idx, "MIPS_24_err"] = err

    res = model_hot.fit_evolving(
        df,
        baseline_col="f24_cold",
        evolve_amp=False,
        evolve_ratios=False,
        feature_envelope="baseline",
    )
    assert res is not None
    np.testing.assert_allclose(res["A_pah"], A_pah_true, rtol=0.35)
    # The fit normalizes rung columns AFTER the envelope, so its summed
    # amplitude is the bin's PEAK fitted hot flux — compare against the
    # peak injected hot flux (convention-free), not the raw rung values.
    tot_rec = np.asarray(res["hot_amp"]).sum(axis=1)
    np.testing.assert_allclose(tot_rec, hot_peak_true, rtol=0.5)


# ---------------------------------------------------------------------------
# Branch-12: PAH plateau (continuum-under-features) + 9.7 µm silicate
# ---------------------------------------------------------------------------

PLATEAUS = ((8.2, 4.0),)  # the 5-10 µm plateau under the 7.7 complex


def _inject(
    *,
    A_pah_true,
    r_true,
    C_true,
    tau_true=0.0,
    q_true=None,
    noise=0.03,
    seed=0,
):
    """Build a tomographic df with an optional plateau and silicate screen.

    Injects through the model's OWN kernels (including the tau-tabulated
    ones), so these are identifiability tests, not kernel-accuracy tests.
    Returns (df, injector_model).
    """
    rng = np.random.default_rng(seed)
    mass_centers = [10.25, 10.65, 11.0]
    z_edges = np.round(np.arange(0.5, 3.5 + 1e-9, 0.15), 4)
    df = _tomographic_skeleton(mass_centers, z_edges)
    df["f24_cold"] = 0.03 / (1.0 + df["z_mid"]) ** 2

    inj = PAHSpectrumModel(
        feature_groups=FEATURE_GROUPS,
        bands=("MIPS_24",),
        sigma_z0=0.01,
        f_cat=0.0,
        plateaus=PLATEAUS if q_true is not None else None,
        include_silicate=tau_true > 0,
    )
    prep = inj._prepare(df, None, None, None, None, None, baseline_col="f24_cold")
    tg = prep["tau_grid"]
    for i, b in enumerate(prep["bins"]):
        sub_idx = df.index[df["prop_bin_id"] == b["m"]]
        fc = b["f_cold"]
        med = float(np.median(fc))
        if tg is None:
            K, S = b["K"][:, 0, :], 1.0
            P = b["P"][:, 0, :] if b["P"] is not None else None
        else:
            K = inj._interp_tau(b["K_tau"][:, :, 0, :], tg, tau_true)
            S = inj._interp_tau(b["S_tau"][:, :, 0], tg, tau_true)
            P = (
                inj._interp_tau(b["P_tau"][:, :, 0, :], tg, tau_true)
                if b["P_tau"] is not None
                else None
            )
        flux = C_true[i] * (fc / med) * S + A_pah_true[i] * C_true[i] * (K @ r_true)
        if q_true is not None:
            # match the fitter's peak normalization (taken at tau=0)
            scale = b["P"][:, 0, :].max(axis=0)
            flux = flux + (P / scale) @ np.full(len(PLATEAUS), q_true[i])
        err = noise * flux + 1e-5
        df.loc[sub_idx, "MIPS_24"] = flux + rng.normal(0.0, err)
        df.loc[sub_idx, "MIPS_24_err"] = err
    return df, inj


_TRUTH = {
    "A_pah_true": np.array([0.8, 1.6, 2.6]),
    "r_true": np.array([1.0, 1.7, 4.0]),
    "C_true": np.array([0.02, 0.03, 0.05]),
}


def _model(**extra):
    return PAHSpectrumModel(
        feature_groups=FEATURE_GROUPS,
        bands=("MIPS_24",),
        sigma_z0=0.01,
        f_cat=0.0,
        **extra,
    )


class TestPlateau:
    """The broad Drude plateau: identifiable, but strongly correlated with 7.7."""

    def test_noiseless_null_gives_exactly_zero_plateau(self):
        """With no plateau in the data the fit must not invent one.

        This is the structural test: at negligible noise the plateau column
        is NOT degenerate with the feature+baseline block, so q solves to 0
        and the PAH amplitudes are untouched. Any offset seen at realistic
        noise is therefore boundary rectification (q >= 0), not a
        mis-specified model.
        """
        df, _ = _inject(**_TRUTH, noise=1e-6, seed=3)
        res = _model(plateaus=PLATEAUS).fit_shared(df, baseline_col="f24_cold")
        assert np.allclose(res["plateau_amp"], 0.0, atol=1e-9)
        np.testing.assert_allclose(res["A_pah"], _TRUTH["A_pah_true"], rtol=0.02)

    def test_plateau_recovered_and_chi2_improves(self):
        """Injected plateau: modelling it restores the fit; ignoring it does not."""
        q_true = np.full(3, 0.004)
        df, _ = _inject(**_TRUTH, q_true=q_true, seed=0)
        off = _model().fit_shared(df, baseline_col="f24_cold")
        on = _model(plateaus=PLATEAUS).fit_shared(df, baseline_col="f24_cold")
        # Ignoring a real plateau degrades the fit and inflates the PAH term
        # (the features are the only thing left to absorb it).
        assert on["chi2_red"] < 0.6 * off["chi2_red"]
        assert np.all(off["A_pah"] > on["A_pah"])
        assert np.all(on["plateau_amp"] > 0)

    def test_differential_plateau_amplitude_is_unbiased(self):
        """The plateau's ABSOLUTE amplitude carries a noise-rectified offset,
        but its differential response is faithful.

        The plateau column is ~0.85-collinear with the 7.7+8.6 kernel, so at
        realistic noise q picks up a positive bias (bounded at zero) that
        comes out of A_pah. Differences between two datasets that share the
        realization cancel it — so quote plateau CHANGES, not absolute
        plateau amplitudes.
        """
        dq = 0.004
        for seed in (0, 1):
            df_a, _ = _inject(**_TRUTH, q_true=np.full(3, 0.002), seed=seed)
            df_b, _ = _inject(**_TRUTH, q_true=np.full(3, 0.002 + dq), seed=seed)
            fit = lambda d: _model(plateaus=PLATEAUS).fit_shared(  # noqa: E731
                d, baseline_col="f24_cold"
            )
            rec = (fit(df_b)["plateau_amp"] - fit(df_a)["plateau_amp"]).ravel()
            np.testing.assert_allclose(rec, dq, rtol=0.25)

    def test_plateau_column_is_collinear_with_the_77_complex(self):
        """Documents WHY the absolute amplitude is fragile (and 12.7 is not)."""
        df, _ = _inject(**_TRUTH)
        model = _model(plateaus=PLATEAUS)
        b = model._prepare(df, None, None, None, None, None, baseline_col="f24_cold")[
            "bins"
        ][0]
        K, P = b["K"][:, 0, :], b["P"][:, 0, :]
        unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
        cos_77 = float(unit(P[:, 0]) @ unit(K[:, 1]))  # 7.7+8.6
        cos_127 = float(unit(P[:, 0]) @ unit(K[:, 2]))  # 12.7
        assert 0.7 < cos_77 < 0.95
        assert cos_127 < 0.4

    def test_plateau_amplitudes_are_non_negative(self):
        """Emission components cannot be negative (PAHFIT convention)."""
        df, _ = _inject(**_TRUTH, noise=0.15, seed=7)
        res = _model(plateaus=PLATEAUS).fit_shared(df, baseline_col="f24_cold")
        assert np.all(np.asarray(res["plateau_amp"]) >= -1e-12)


class TestSilicate:
    def test_tau_recovered(self):
        df, _ = _inject(**_TRUTH, tau_true=0.8, seed=0)
        res = _model(include_silicate=True).fit_shared(df, baseline_col="f24_cold")
        assert res["tau_sil"] == pytest.approx(0.8, abs=0.1)
        np.testing.assert_allclose(res["A_pah"], _TRUTH["A_pah_true"], rtol=0.2)

    def test_ignoring_real_absorption_wrecks_the_fit(self):
        """Motivates fitting tau: the 9.7 µm trough cannot be absorbed elsewhere."""
        df, _ = _inject(**_TRUTH, tau_true=0.8, seed=0)
        off = _model().fit_shared(df, baseline_col="f24_cold")
        on = _model(include_silicate=True).fit_shared(df, baseline_col="f24_cold")
        assert off["chi2_red"] > 10 * on["chi2_red"]
        # the un-screened fit compensates by inflating the PAH amplitude
        assert np.all(off["A_pah"] > 2 * _TRUTH["A_pah_true"])

    def test_null_absorption_recovers_tau_zero(self):
        df, _ = _inject(**_TRUTH, seed=0)
        res = _model(include_silicate=True).fit_shared(df, baseline_col="f24_cold")
        assert res["tau_sil"] == pytest.approx(0.0, abs=0.05)
        # and leaves the no-silicate answer alone (the bounded optimizer stops
        # a hair above zero, so this is "unchanged", not bitwise identical)
        base = _model().fit_shared(df, baseline_col="f24_cold")
        np.testing.assert_allclose(res["A_pah"], base["A_pah"], rtol=1e-3)

    def test_screen_is_applied_inside_the_band_integral(self):
        """The band resolves the trough, so a monochromatic screen is wrong.

        Guards the choice to attenuate inside the integral rather than copy
        pah_model's band-averaged-tau screen: at tau=1, z=1.46 (MIPS 24 on
        rest 9.7 µm) the monochromatic approximation is ~20% off while the
        band-averaged-tau one is ~1% — small here but growing with tau.
        """
        from simstack4.pah_bandpass import get_bandpass
        from simstack4.pah_spectrum import (
            silicate_optical_depth,
            silicate_transmission,
        )

        bp = get_bandpass("MIPS_24")
        z = np.array([bp.lam_eff / 9.7 - 1.0])
        lam_rest = bp.lam_fine[None, :] / (1.0 + z[:, None])
        tau_bar = (
            np.trapezoid(
                silicate_optical_depth(lam_rest) * bp.resp_fine, bp.lam_fine, axis=1
            )
            / bp.norm
        )
        proper = float(silicate_transmission(z, "MIPS_24", 1.0)[0])
        band_avg_tau = float(np.exp(-tau_bar)[0])
        mono = float(np.exp(-silicate_optical_depth(bp.lam_eff / (1.0 + z)))[0])
        assert abs(proper / band_avg_tau - 1.0) < 0.03
        assert abs(proper / mono - 1.0) > 0.10

    def test_tau_zero_is_byte_identical_to_the_unattenuated_kernel(self):
        from simstack4.pah_spectrum import feature_band_curves

        z = np.linspace(0.5, 3.5, 50)
        a = feature_band_curves(z, "MIPS_24", profile="drude")
        b = feature_band_curves(z, "MIPS_24", profile="drude", tau_sil=0.0)
        assert np.array_equal(a, b)


class TestPlateauSilicateWiring:
    def test_joint_fit_evolving_recovers_both(self):
        df, _ = _inject(**_TRUTH, tau_true=0.6, q_true=np.full(3, 0.003), seed=1)
        res = _model(
            plateaus=PLATEAUS, include_silicate=True, tau_sil_prior=1.0
        ).fit_evolving(df, baseline_col="f24_cold", eta_prior_sigma=1.0)
        assert res is not None
        assert res["tau_sil"] == pytest.approx(0.6, abs=0.1)
        np.testing.assert_allclose(
            np.asarray(res["plateau_amp"]).ravel(), 0.003, rtol=0.6
        )

    def test_result_keys_absent_when_components_are_off(self):
        df, _ = _inject(**_TRUTH)
        res = _model().fit_shared(df, baseline_col="f24_cold")
        for key in ("plateau_amp", "tau_sil", "hot_amp"):
            assert key not in res

    @pytest.mark.parametrize(
        "kwargs", [{"plateaus": PLATEAUS}, {"include_silicate": True}]
    )
    def test_unwired_fitters_raise(self, kwargs):
        df, _ = _inject(**_TRUTH)
        with pytest.raises(NotImplementedError, match="fit_shared/fit_evolving"):
            _model(**kwargs).fit_lstsq(df)

    def test_bad_silicate_scope_rejected(self):
        with pytest.raises(ValueError, match="silicate_scope"):
            _model(silicate_scope="nonsense")


class TestSilicateMCMC:
    """tau_sil is sampled by fit_evolving_mcmc (the plateau/hot blocks are not)."""

    def test_mcmc_samples_tau_and_recovers_it(self):
        df, _ = _inject(**_TRUTH, tau_true=0.7, seed=0)
        res = _model(include_silicate=True, tau_sil_prior=1.0).fit_evolving_mcmc(
            df,
            baseline_col="f24_cold",
            n_walkers=24,
            n_steps=400,
            n_burn=150,
            seed=1,
        )
        assert res["names"][-1] == "tau_sil"  # sampled LAST, so other slices hold
        assert res["chain"].shape[1] == len(res["names"])
        assert res["tau_sil"] == pytest.approx(0.7, abs=0.15)
        assert res["tau_sil_err"] > 0

    def test_decomposition_applies_the_screen(self):
        """evolving_flux_decomposition must reconstruct with tau, not without it.

        If it ignored the sampled tau the modelled total would sit above the
        data everywhere the trough bites.
        """
        from simstack4.pah_spectrum import evolving_flux_decomposition

        df, _ = _inject(**_TRUTH, tau_true=0.7, seed=0)
        res = _model(include_silicate=True, tau_sil_prior=1.0).fit_evolving_mcmc(
            df, baseline_col="f24_cold", n_walkers=24, n_steps=400, n_burn=150, seed=1
        )
        dec = evolving_flux_decomposition(res, n_draws=40, seed=2)
        assert np.isfinite(dec["total"]).all()
        # totals track the data (no systematic offset from a dropped screen)
        pull = (dec["total"] - dec["f_obs"]) / dec["f_err"]
        assert abs(float(np.median(pull))) < 1.0

    def test_mcmc_still_rejects_the_non_negative_blocks(self):
        df, _ = _inject(**_TRUTH)
        with pytest.raises(NotImplementedError, match="plateaus"):
            _model(plateaus=PLATEAUS).fit_evolving_mcmc(df, baseline_col="f24_cold")

    def test_no_tau_key_when_silicate_off(self):
        df, _ = _inject(**_TRUTH)
        res = _model().fit_evolving_mcmc(
            df, baseline_col="f24_cold", n_walkers=24, n_steps=200, n_burn=80, seed=1
        )
        assert "tau_sil" not in res
        assert "tau_sil" not in res["names"]
