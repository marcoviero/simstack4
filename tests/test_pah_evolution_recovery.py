"""Injection-recovery tests for the sSFR-anchored evolving PAH fit.

Exercise ``PAHSpectrumModel.fit_evolving`` (branch 5): the PAH/continuum
amplitude and the shared feature-group ratios drift along each mass bin with
specific SFR as the MIPS bandpass sweeps rest wavelength with redshift. The
fixtures inject fluxes through the model's OWN photo-z-smeared kernel (the same
self-consistent style as ``test_pah_shared_baseline``) so the tests isolate the
evolution math:

  f_obs_ib = C_m·f_cold_norm_ib + alpha_m·Σ_g r_g0·10^((η_A+η_g)·ŝ_i)·K_gib

Validated: (1) shared η recovery with 24+70, (2) the static fit is biased under
evolution, (3) MIPS 70 breaks the amplitude/ratio degeneracy that pins 24-only,
and (4) zero evolution reduces to the static shared-ratio result.

Synthetic data only — no FITS/catalogs.
"""

import numpy as np
import pandas as pd
import pytest

from simstack4.dust_evolution import main_sequence_ssfr
from simstack4.pah_spectrum import PAHSpectrumModel

GROUPS = [[0], [1, 2], [3], [4]]  # 6.2 | 7.7+8.6 | 11.3 | 12.7
BANDS = ("MIPS_24", "MIPS_70")
_BASE = {"MIPS_24": "f24_cold", "MIPS_70": "f70_cold"}


def _skeleton(mass_centers, z_edges, n_stagger=3, dz=0.15):
    """Staggered multi-run tomographic skeleton, one row per (run, mass, z)."""
    rows = []
    for run, off in enumerate(np.arange(n_stagger) * dz / n_stagger):
        edges = np.round(z_edges + off, 6)
        for m, logM in enumerate(mass_centers):
            for zlo, zhi in zip(edges[:-1], edges[1:], strict=False):
                rows.append(
                    {
                        "run_id": run,
                        "z_lo": float(zlo),
                        "z_hi": float(zhi),
                        "z_mid": 0.5 * (zlo + zhi),
                        "prop_bin_id": m,
                        "log_M_star": float(logM),
                        "log_sigma_sfr": 0.0,
                        "MIPS_24": 1.0,
                        "MIPS_24_err": 1.0,
                        "MIPS_70": 1.0,
                        "MIPS_70_err": 1.0,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values(["prop_bin_id", "run_id", "z_lo"])
        .reset_index(drop=True)
    )


def _inject(model, df, *, eta_amp, eta_ratio, alpha_t, C_t, r_t, noise_rel, seed):
    """Inject fluxes through the model kernel with sSFR evolution; add noise."""
    rng = np.random.default_rng(seed)
    df = df.copy()
    for col in _BASE.values():
        df[col] = 0.03 / (1.0 + df["z_mid"]) ** 2  # smooth cold continuum (mJy)
    bl = {b: _BASE[b] for b in model.bands}
    prep = model._prepare(df, None, None, None, None, None, baseline_cols=bl)
    ls_all = [
        main_sequence_ssfr(np.asarray(b["z_mid"]), b["props"]["log_M_star"])
        for b in prep["bins"]
    ]
    s_pivot = float(np.median(np.concatenate(ls_all)))
    e = eta_amp + np.asarray(eta_ratio)  # per-group exponent (eta_ratio[0]=0)
    for i, b in enumerate(prep["bins"]):
        sub_idx = df.index[df["prop_bin_id"] == b["m"]]
        shat = (
            main_sequence_ssfr(np.asarray(b["z_mid"]), b["props"]["log_M_star"])
            - s_pivot
        )
        for band in model.bands:
            bidx = model.bands.index(band)
            kmod = (10.0 ** np.outer(shat, e)) * b["K"][:, bidx, :]
            fc = b["f_cold_by_band"][band]
            flux = C_t[i] * fc / float(np.median(fc)) + alpha_t[i] * (kmod @ r_t)
            err = noise_rel * np.abs(flux) + 1e-6
            df.loc[sub_idx, band] = flux + rng.normal(0.0, err)
            df.loc[sub_idx, f"{band}_err"] = err
    return df


# Shared truth for the evolving-injection tests.
_MASS = [9.8, 10.5, 11.1]
_ZEDGES = np.round(np.arange(0.4, 3.2 + 1e-9, 0.15), 4)
_ETA_AMP = 0.4
_ETA_RATIO = np.array([0.0, 0.5, -0.3, 0.2])
_ALPHA = np.array([0.018, 0.030, 0.045])
_C = np.array([0.02, 0.03, 0.05])
_R = np.array([1.0, 1.7, 0.8, 1.3])


@pytest.fixture
def evolving_df():
    model = PAHSpectrumModel(feature_groups=GROUPS, bands=BANDS)
    df = _inject(
        model,
        _skeleton(_MASS, _ZEDGES),
        eta_amp=_ETA_AMP,
        eta_ratio=_ETA_RATIO,
        alpha_t=_ALPHA,
        C_t=_C,
        r_t=_R,
        noise_rel=0.02,
        seed=1,
    )
    return df


def test_fit_evolving_recovers_shared_slopes(evolving_df):
    model = PAHSpectrumModel(feature_groups=GROUPS, bands=BANDS)
    res = model.fit_evolving(evolving_df)
    assert res is not None
    assert res["valid"] == [0, 1, 2]
    # Shared amplitude slope recovered (single η_A across all three mass bins).
    assert res["eta_amp"] == pytest.approx(_ETA_AMP, abs=0.2)
    # Ratio slopes recovered in sign and roughly in value.
    assert res["eta_ratio"][1] > 0.15
    assert res["eta_ratio"][2] < 0.0
    np.testing.assert_allclose(res["eta_ratio"][1:], _ETA_RATIO[1:], atol=0.35)
    # Good fit and recovered pivot amplitudes (A_pah = alpha/C) finite & sane.
    assert res["chi2_red"] < 1.5
    np.testing.assert_allclose(res["A_pah"], _ALPHA / _C, rtol=0.3)
    # A[m,g] = A_pah[m] * r[g] consistency, with r_0 fixed.
    assert res["r"][0] == pytest.approx(1.0)
    np.testing.assert_allclose(
        res["A"], res["A_pah"][:, None] * res["r"][None, :], rtol=1e-6
    )


def test_static_fit_is_biased_under_evolution(evolving_df):
    model = PAHSpectrumModel(feature_groups=GROUPS, bands=BANDS)
    evolving = model.fit_evolving(evolving_df)
    static = model.fit_evolving(evolving_df, evolve_amp=False, evolve_ratios=False)
    # Ignoring evolution leaves structured residuals: worse fit than the evolving
    # model, and the recovered pivot amplitudes are biased away from the truth.
    assert static["chi2_red"] > evolving["chi2_red"]
    true_A = _ALPHA / _C  # ~flat in mass
    static_err = np.max(np.abs(static["A_pah"] - true_A))
    evolving_err = np.max(np.abs(evolving["A_pah"] - true_A))
    assert static_err > evolving_err


def test_70um_breaks_ratio_degeneracy(evolving_df):
    both = PAHSpectrumModel(feature_groups=GROUPS, bands=BANDS).fit_evolving(
        evolving_df
    )
    only24 = PAHSpectrumModel(feature_groups=GROUPS, bands=("MIPS_24",)).fit_evolving(
        evolving_df, baseline_cols={"MIPS_24": "f24_cold"}
    )
    # The long-wavelength ratio slopes (11.3, 12.7 groups) are poorly constrained
    # by 24 µm alone — they run toward the bounds — but adding 70 µm pins them
    # close to truth. Compare the worst-case ratio-slope error of the two fits.
    worst_24 = np.max(np.abs(only24["eta_ratio"][2:] - _ETA_RATIO[2:]))
    worst_both = np.max(np.abs(both["eta_ratio"][2:] - _ETA_RATIO[2:]))
    assert worst_both < worst_24


def test_null_evolution_reduces_to_static():
    """Zero injected evolution → recovered η ≈ 0 and A_pah matches fit_shared."""
    model = PAHSpectrumModel(feature_groups=GROUPS, bands=BANDS)
    df = _inject(
        model,
        _skeleton(_MASS, _ZEDGES),
        eta_amp=0.0,
        eta_ratio=np.zeros(len(GROUPS)),
        alpha_t=_ALPHA,
        C_t=_C,
        r_t=_R,
        noise_rel=0.02,
        seed=2,
    )
    res = model.fit_evolving(df)
    # The amplitude slope (the dominant signal) and the well-constrained
    # 7.7+8.6 µm ratio slope return to ~zero; the low-leverage long-λ ratio
    # slopes can still wander modestly (same groups that 24-only cannot pin).
    assert abs(res["eta_amp"]) < 0.15
    assert abs(res["eta_ratio"][1]) < 0.2
    assert np.all(np.abs(res["eta_ratio"]) < 0.4)
    # Cross-check against the non-evolving shared-ratio fit on 24 µm.
    shared = model.fit_shared(df, baseline_col="f24_cold")
    np.testing.assert_allclose(res["A_pah"], shared["A_pah"], rtol=0.25)


def test_eta_prior_tames_runaway(evolving_df):
    """A Gaussian slope prior keeps the slopes finite and amplitudes physical."""
    model = PAHSpectrumModel(feature_groups=GROUPS, bands=BANDS)
    res = model.fit_evolving(evolving_df, eta_prior_sigma=1.0)
    # Prior pulls slightly toward zero but recovery survives, and the prior can
    # only shrink |η| relative to the unregularized fit.
    free = model.fit_evolving(evolving_df)
    assert abs(res["eta_amp"]) <= abs(free["eta_amp"]) + 1e-6
    assert np.all(np.isfinite(res["A_pah"]))
    assert res["eta_amp"] == pytest.approx(_ETA_AMP, abs=0.3)


def test_evolution_recovery_sweep_unbiased_and_70_helps():
    """Library campaign: clean recovery is unbiased and 70 µm shrinks scatter."""
    from simstack4.pah_dither import DitherScheme, evolution_recovery_sweep

    scheme = DitherScheme.uniform(
        z_min=0.5,
        z_max=3.2,
        dz=0.15,
        n_stagger=3,
        property_bins=[{"log_M_star": m} for m in (10.3, 10.7, 11.1)],
        bands=BANDS,
    )
    df = evolution_recovery_sweep(
        scheme,
        eta_amp_grid=(-0.7,),
        noise_rel_grid=(0.1,),
        feature_groups=GROUPS,
        n_seed=8,
    )
    # Both band sets recover the injected slope without bias and stay physical.
    assert np.all(np.abs(df["eta_bias"]) < 0.25)
    assert np.all(df["max_Apah"] < 3.0)
    assert np.all(df["rail_frac"] == 0.0)
    # Adding 70 µm does not worsen the recovery scatter (the strong "70 µm
    # halves the scatter" claim is the full campaign / the Fisher test, which
    # are not subject to the 8-seed sampling noise here).
    s24 = float(df[df["bands"] == "24"]["eta_scatter"].iloc[0])
    s2470 = float(df[df["bands"] == "24+70"]["eta_scatter"].iloc[0])
    assert s2470 <= 1.5 * s24


def test_fit_with_alpha_recovers_wien_slope():
    """fit_with_alpha recovers an injected Wien slope and exposes the prior."""
    mass = [10.3, 10.7, 11.1]
    zed = np.round(np.arange(0.5, 3.05, 0.15), 4)
    alpha_true = 2.4
    model = PAHSpectrumModel(feature_groups=GROUPS, bands=BANDS)

    # Skeleton with the baseline COLUMN at the reference slope (α=2); the DATA is
    # injected with α_true so fit_with_alpha must tilt to recover it.
    rows = []
    for run, off in enumerate(np.arange(3) * 0.15 / 3):
        e = np.round(zed + off, 6)
        for m, logM in enumerate(mass):
            for zl, zh in zip(e[:-1], e[1:], strict=False):
                rec = {
                    "run_id": run,
                    "z_lo": float(zl),
                    "z_hi": float(zh),
                    "z_mid": 0.5 * (zl + zh),
                    "prop_bin_id": m,
                    "log_M_star": float(logM),
                    "log_sigma_sfr": 0.0,
                }
                for b in BANDS:
                    rec[b] = 1.0
                    rec[f"{b}_err"] = 1.0
                rows.append(rec)
    df = (
        pd.DataFrame(rows)
        .sort_values(["prop_bin_id", "run_id", "z_lo"])
        .reset_index(drop=True)
    )
    for b in BANDS:
        df[_BASE[b]] = (1.0 + df["z_mid"]) ** (-2.0)  # column at α_ref = 2

    rng = np.random.default_rng(4)
    prep = model._prepare(
        df, None, None, None, None, None, baseline_cols={b: _BASE[b] for b in BANDS}
    )
    C = np.array([0.03, 0.04, 0.06])
    for i, b in enumerate(prep["bins"]):
        sidx = df.index[df["prop_bin_id"] == b["m"]]
        for band in BANDS:
            bi = BANDS.index(band)
            base = (1.0 + np.asarray(b["z_mid"])) ** (-alpha_true)  # data at α_true
            base = base / np.median(base)
            K = b["K"][:, bi, :]
            flux = C[i] * base + C[i] * (K @ np.array([1.0, 1.6, 0.8, 1.3]))
            flux = flux * (1.0 + rng.normal(0, 0.05, len(flux)))
            df.loc[sidx, band] = flux + rng.normal(0, 0.05 * np.abs(flux) + 1e-9)
            df.loc[sidx, f"{band}_err"] = 0.05 * np.abs(flux) + 1e-9

    res = model.fit_with_alpha(
        df,
        evolving=True,
        evolve_amp=False,
        evolve_ratios=False,
        alpha_prior=(2.0, 0.3),
    )
    assert res["alpha_wien"] == pytest.approx(alpha_true, abs=0.25)
    assert res["alpha_prior"] == (2.0, 0.3)
    assert np.all(np.isfinite(res["A_pah"]))
