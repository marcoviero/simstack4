"""Generate 2026-07-03-pah-money-plots.ipynb.

Branch-7 money-plot reproduction (user-directed 2026-07-03), Objective-4
figure set. Two figures, regenerated standalone from the 3 K-fold stacks:

  1. Neutral/ionized PAH band ratio (12.7/6.2 um) vs stellar mass with
     fold-ensemble errors -- letter-candidates notebook Sec 4a. Single-band
     fit_shared path: untouched by the 2026-07-03 library fixes, so this
     must REPRODUCE the documented values (72.4+/-23.1 -> 12.8+/-4.0 ->
     3.7+/-0.5 -> 0.8+/-0.6).
  2. The Narayanan+26 confrontation figure (L_PAH/L_IR vs M* against the
     density/shattering and enrichment/PZR channel bands) -- confrontation
     notebook Sec 6. CAUTION: the measured slope runs through fit_with_alpha
     with BOTH bands, i.e. through the 2026-07-03 multi-band normalization
     fix -- this is a RE-DERIVATION and the number may legitimately shift
     from the documented +0.236+/-0.076 (documented value was computed with
     the buggy per-band normalization).

Data cells are verbatim from the two source notebooks. Requires PICKLESPATH.

Run:  uv run python notebooks/build_pah_money_plots_notebook.py
Then: uv run jupyter nbconvert --to notebook --execute --inplace \
          notebooks/2026-07-03-pah-money-plots.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(
    r"""# Branch-7 money plots, reproduced standalone

**2026-07-03.** The two headline figures of the branch, regenerated in one
notebook from the 3 disjoint K-fold COSMOS2020 stacks
(`cosmos2020_PAH_split{0,1,2}of3`, 20260630_{193627, 211122, 222635}):

1. **§2 — PAH band-ratio shift vs stellar mass** (12.7 µm / 6.2 µm,
   neutral-PAH over ionized-PAH, fold-ensemble errors) — the §1a lead
   result, from `2026-07-01-pah-forward-model-letter-candidates.ipynb` §4a.
   This path (24 µm-only `fit_shared`) is untouched by the 2026-07-03
   library fixes, so the numbers must reproduce exactly.
2. **§3 — the Narayanan+26 confrontation figure** (L_PAH/L_IR vs M\*
   against the two mass-axis channels of the shattering model) — from
   `2026-07-02-pah-narayanan-confrontation.ipynb` §6. **This is a
   re-derivation, not a pure reproduction**: the measured slope runs
   through `fit_with_alpha` with both bands, i.e. through the 2026-07-03
   multi-band normalization fix (the documented +0.236 ± 0.076 was computed
   with the old per-band normalization that forced equal 24/70 continuum
   levels through the shared C_m). Agreement says the fix didn't matter for
   this configuration; a shift means the corrected number supersedes the
   documented one.

Figures are saved as `pah_money_bandratio_vs_mass.png` and
`pah_money_narayanan_confrontation.png`. All data cells are verbatim from
the source notebooks."""
)

code(
    r'''import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simstack4.wrapper import SimstackWrapper
from simstack4.plots import _extract_pop_type
from simstack4.pah_spectrum import (
    PAHSpectrumModel, feature_band_curves, group_weights, DEFAULT_FEATURES,
)
from simstack4.greybody import Greybody as _Greybody
from simstack4.dust_evolution import main_sequence_ssfr

# Each simstack4 submodule calls setup_logging() (-> INFO + its own stdout
# handler) at IMPORT time, so this must run AFTER the imports above or it
# gets immediately overridden (same note as the letter notebook).
logging.getLogger("simstack4").setLevel(logging.WARNING)

path_json = os.path.join(os.environ["PICKLESPATH"], "simstack", "stacked_flux_densities")

ANALYSIS_KWARGS = dict(
    use_mcmc=False,
    temperature_prior="schreiber",
    snr_high=5.0,
    snr_low=2.0,
    inflation_factors={
        24: 10000,
        70: {(0.0, 0.8): 1.0, (0.8, 99.0): 10000},
    },
    use_covariance=True,
    use_pah=False,
)

# The three runs are DISJOINT catalog folds (cosmos2020_PAH_split{0,1,2}of3)
# that ALSO carry different redshift-bin offsets (0.0000 / 0.0375 / 0.0750).
RUN_DATES = {
    0: "20260630_193627",   # cosmos2020_PAH_split0of3, offset 0.0000
    1: "20260630_211122",   # cosmos2020_PAH_split1of3, offset 0.0375
    2: "20260630_222635",   # cosmos2020_PAH_split2of3, offset 0.0750
}

# 4 science mass bins (8.5-10.0 stays an unanalysed nuisance layer in the config)
MASS_BINS = [
    (10.0, 10.5, "C0", r"$10.0 < \log M_* < 10.5$"),
    (10.5, 10.8, "C1", r"$10.5 < \log M_* < 10.8$"),
    (10.8, 11.1, "C2", r"$10.8 < \log M_* < 11.1$"),
    (11.1, 12.0, "C3", r"$\log M_* > 11.1$"),
]
# Feature groups: 0=6.2, 1=7.7, 2=8.6, 3=11.3(blind), 4=12.7, 5=16.4, 6=17.0
FEATURE_GROUPS = [[0], [1, 2], [4]]   # 6.2 | 7.7+8.6 | 12.7  (11.3 blind)
SIGMA_Z0 = 0.01   # sigma_z(1+z) for COSMOS2020 photo-z
'''
)

code(
    r'''wrapper_0 = SimstackWrapper()
wrapper_0.load_stacking_results(
    os.path.join(path_json, f"cosmos20_stacking_{RUN_DATES[0]}.json"))
wrapper_0.run_analysis_only(**ANALYSIS_KWARGS)

wrapper_1 = SimstackWrapper()
wrapper_1.load_stacking_results(
    os.path.join(path_json, f"cosmos20_stacking_{RUN_DATES[1]}.json"))
wrapper_1.run_analysis_only(**ANALYSIS_KWARGS)

wrapper_2 = SimstackWrapper()
wrapper_2.load_stacking_results(
    os.path.join(path_json, f"cosmos20_stacking_{RUN_DATES[2]}.json"))
wrapper_2.run_analysis_only(**ANALYSIS_KWARGS)

WRAPPERS = [wrapper_0, wrapper_1, wrapper_2]
print("Loaded 3 disjoint-fold stacking runs:", RUN_DATES)
'''
)

code(
    r'''def build_pah_spectrum_df(wrappers, mass_bins, split_filter=None, min_tier="B"):
    """Extract raw stacked fluxes (mJy) and greybody Wien-side extrapolations.

    Returns one row per (run, mass bin, redshift bin): MIPS_24/70 with errors,
    z range, source count, tier, and the FIR-fit T_dust/amplitude/beta needed
    to extrapolate the cold continuum to 24/70 um.
    """
    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank[min_tier.upper()]
    rows = []
    _gb_row = _Greybody()   # reuse one instance -- constructing per-row spams the log
    for run_id, wrapper in enumerate(wrappers):
        if wrapper is None:
            continue
        pr = getattr(wrapper, "processed_results", None)
        if pr is None or not pr.sed_results:
            continue
        pops = wrapper.population_manager.populations
        for pop_id, sed in pr.sed_results.items():
            if not sed.greybody_fit_success:
                continue
            tier = getattr(sed, "fit_quality_tier", None) or "C"
            if tier_rank.get(tier, 2) > min_rank:
                continue
            pop_type = _extract_pop_type(pop_id)
            if split_filter is not None:
                allowed = {f"split_{i}" for i in split_filter}
                if pop_type not in allowed and pop_type != "_all_":
                    continue
            elif pop_type == "split_2":
                continue
            pop = pops.get(pop_id)
            if pop is None:
                continue
            z_range = pop.bin_ranges.get("redshift")
            if z_range is None:
                continue
            z_lo, z_hi = float(z_range[0]), float(z_range[1])
            props = sed.bin_properties or {}
            if isinstance(props, str):
                import ast
                try:
                    props = ast.literal_eval(props)
                except Exception:
                    props = {}
            stellar_mass = None
            log_ssfr_measured = np.nan
            for key, val in props.items():
                if "mass" in key.lower() and "delta" not in key.lower():
                    stellar_mass = float(val)
                if "ssfr" in key.lower():
                    log_ssfr_measured = float(val)
            if stellar_mass is None:
                continue
            prop_bin_id = next(
                (i for i, (lo, hi, *_) in enumerate(mass_bins)
                 if lo <= stellar_mass < hi),
                None,
            )
            if prop_bin_id is None:
                continue
            f24 = f24_err = f70 = f70_err = np.nan
            for j, wl in enumerate(sed.wavelengths):
                if abs(wl - 24.0) / 24.0 < 0.15:
                    f24 = float(sed.flux_densities[j])
                    f24_err = float(sed.flux_errors[j])
                elif abs(wl - 70.0) / 70.0 < 0.15:
                    f70 = float(sed.flux_densities[j])
                    f70_err = float(sed.flux_errors[j])
            if not (np.isfinite(f24) and f24 > 0
                    and np.isfinite(f24_err) and f24_err > 0):
                continue
            f24_cold = f70_cold = np.nan
            if (sed.greybody_fit_success and sed.amplitude is not None
                    and sed.dust_temperature_rest_frame is not None
                    and sed.emissivity_index is not None):
                _z = 0.5 * (z_lo + z_hi)
                f24_cold = float(_gb_row.greybody_model(
                    np.array([24.0 / (1.0 + _z)]), sed.amplitude,
                    sed.dust_temperature_rest_frame, sed.emissivity_index)[0])
                f70_cold = float(_gb_row.greybody_model(
                    np.array([70.0 / (1.0 + _z)]), sed.amplitude,
                    sed.dust_temperature_rest_frame, sed.emissivity_index)[0])
            rows.append({
                "run_id": run_id, "z_lo": z_lo, "z_hi": z_hi,
                "z_mid": 0.5 * (z_lo + z_hi), "prop_bin_id": int(prop_bin_id),
                "log_M_star": stellar_mass,
                "log_ssfr_measured": log_ssfr_measured,
                "n_sources": int(getattr(sed, "n_sources", 0)),
                "MIPS_24": f24, "MIPS_24_err": f24_err,
                "MIPS_70": f70, "MIPS_70_err": f70_err,
                "f24_cold": f24_cold, "f70_cold": f70_cold,
                "tier": tier,
                "T_dust": (float(sed.dust_temperature_rest_frame)
                             if sed.dust_temperature_rest_frame is not None else np.nan),
                "log_amp": float(sed.amplitude) if sed.amplitude is not None else np.nan,
                "beta": (float(sed.emissivity_index)
                          if sed.emissivity_index is not None else np.nan),
            })
    df = (pd.DataFrame(rows)
            .sort_values(["prop_bin_id", "run_id", "z_mid"])
            .reset_index(drop=True))
    return df


# Keep down to Tier C (as the letter notebook does): the smoothed baseline is
# what makes Tier C usable.
df_pool_raw = build_pah_spectrum_df(WRAPPERS, MASS_BINS, split_filter=[0], min_tier="C")
print(f"Pooled DataFrame: {len(df_pool_raw)} points, "
      f"{df_pool_raw['run_id'].nunique()} runs, "
      f"{df_pool_raw['prop_bin_id'].nunique()} mass bins")
'''
)

code(
    r'''def _design(z, dM, quad):
    cols = [np.ones_like(z), z]
    if quad:
        cols.append(z ** 2)
    cols.append(dM)
    return np.column_stack(cols)

def _fit_bic(y, z, dM, w=None, allow_quad=True):
    """OLS/WLS fit of y ~ (z[,z^2], dM); pick linear vs quadratic-in-z by BIC."""
    n = len(y)
    cands = [False, True] if (allow_quad and n >= 6) else [False]
    best = None
    for quad in cands:
        X = _design(z, dM, quad)
        if X.shape[1] >= n:
            continue
        if w is None:
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            rss = float(np.sum((y - X @ coef) ** 2))
        else:
            sw = np.sqrt(w)
            coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
            rss = float(np.sum(w * (y - X @ coef) ** 2))
        k = X.shape[1]
        bic = n * np.log(max(rss, 1e-300) / n) + k * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, "quadratic" if quad else "linear", coef, quad)
    _, lbl, coef, quad = best
    return lbl, coef, (lambda zz, dd: _design(zz, dd, quad) @ coef), best[0]


def smooth_baseline(dff, tier_col="tier"):
    """Recompute f24_cold/f70_cold from a smooth T(z,M*), logA(z,M*) relation
    trained on Tier A/B only; returns a copy of dff with f24_cold/f70_cold
    replaced (raw values kept in *_raw columns)."""
    if tier_col in dff.columns:
        train_mask = dff[tier_col].isin(["A", "B"])
    else:
        train_mask = np.ones(len(dff), dtype=bool)
    train = dff[train_mask & np.isfinite(dff["T_dust"]) & np.isfinite(dff["log_amp"])]
    out = dff.copy()
    if len(train) < 6:
        return out
    zt, dMt = train["z_mid"].values, train["log_M_star"].values - 10.0
    beta0 = float(np.nanmedian(train["beta"].values))
    lblT, coefT, predT, _ = _fit_bic(train["T_dust"].values, zt, dMt)
    lblA, coefA, predA, _ = _fit_bic(train["log_amp"].values, zt, dMt)
    z_lo_tr, z_hi_tr = float(zt.min()), float(zt.max())
    dM_lo_tr, dM_hi_tr = float(dMt.min()), float(dMt.max())
    zc, dMc = dff["z_mid"].values, dff["log_M_star"].values - 10.0
    zc_p = np.clip(zc, z_lo_tr, z_hi_tr)
    dMc_p = np.clip(dMc, dM_lo_tr, dM_hi_tr)
    T_sm = np.clip(predT(zc_p, dMc_p), 15.0, 60.0)
    A_sm = predA(zc_p, dMc_p)
    gb = _Greybody()
    for band_um, bcol in ((24.0, "f24_cold"), (70.0, "f70_cold")):
        out[f"{bcol}_raw"] = out[bcol]
        out[f"{bcol}_smooth"] = np.array([
            float(gb.greybody_model(np.array([band_um / (1.0 + z)]), a, t, beta0)[0])
            for z, a, t in zip(zc, A_sm, T_sm)])
        out[bcol] = out[f"{bcol}_smooth"]
    out["T_dust_smooth"], out["log_amp_smooth"] = T_sm, A_sm
    out["beta0_smooth"] = beta0
    print(f"  T_dust(z,M*): {lblT:<9} coef={np.round(coefT,3)}   "
          f"logA(z,M*): {lblA:<9} coef={np.round(coefA,3)}   beta0={beta0:.2f}")
    return out


print("Pooled, smoothed:")
df_pool_sm = smooth_baseline(df_pool_raw)

fold_dfs_raw, fold_dfs_sm = [], []
for k, w in enumerate(WRAPPERS):
    d = build_pah_spectrum_df([w], MASS_BINS, split_filter=[0], min_tier="C")
    fold_dfs_raw.append(d)
    print(f"Fold {k} ({len(d)} points), smoothed:")
    fold_dfs_sm.append(smooth_baseline(d))
'''
)

md(
    r"""## 2 · Money plot 1 — neutral/ionized band ratio vs stellar mass (§4a)

Per mass bin, per fold: `fit_shared` (24 µm only, smoothed baseline) with
the feature-group ratios floating — 4 independent single-bin fits per fold.
Group ratios are referenced to 6.2 µm (r₀ ≡ 1), so `r[2]` is the
12.7 µm/6.2 µm neutral-over-ionized ratio. Fold-ensemble error = scatter of
the 3 disjoint folds / √3 (the project's standard convention).

Documented values to reproduce (branch-7 brief §1a): 72.4±23.1 → 12.8±4.0 →
3.7±0.5 → 0.8±0.6, every adjacent bin separated by 2.2–4.0σ."""
)

code(
    r'''_kmodel = PAHSpectrumModel(feature_groups=FEATURE_GROUPS, bands=("MIPS_24",),
                           sigma_z0=SIGMA_Z0, f_cat=0.03)
group_labels = ["+".join(str(_kmodel.features[j][0]) for j in g) + " um"
                for g in FEATURE_GROUPS]

print("=== Per-mass-bin r_g, EACH FOLD SEPARATELY ===")
fold_results = {i: [] for i in range(len(MASS_BINS))}
for fi, dff in enumerate(fold_dfs_sm):
    print(f"--- fold {fi} ({len(dff)} points) ---")
    for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
        sub = dff[dff["prop_bin_id"] == i].copy()
        sub["prop_bin_id"] = 0
        if len(sub) < 8:
            print(f"  {lbl}: too few points ({len(sub)}), skipping")
            continue
        res_i = _kmodel.fit_shared(sub, baseline_col="f24_cold")
        if res_i is None:
            print(f"  {lbl}: fit failed")
            continue
        fold_results[i].append(res_i["r"])
        r_str = "  ".join(f"{glbl}={res_i['r'][g]:.2f}"
                          for g, glbl in enumerate(group_labels))
        print(f"  {lbl:<28} n={len(sub):>3}  chi2_red={res_i['chi2_red']:.2f}  {r_str}")

# POOLED per-mass-bin fits: all 3 disjoint folds jointly -- the dense
# interleaved wavelength sampling the dither design was built for. The pooled
# fit is the MEASUREMENT; the folds above only supply the subsample error
# (they share the field, so fold scatter is a jackknife-style error, not
# three independent observations).
pooled_flux = []
for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
    sub = df_pool_sm[df_pool_sm["prop_bin_id"] == i].copy()
    sub["prop_bin_id"] = 0
    res_i = _kmodel.fit_shared(sub, baseline_col="f24_cold")
    pooled_flux.append(res_i["r"][2])
'''
)

code(
    r'''labels = [lbl for _, _, _, lbl in MASS_BINS]
means, errs = [], []
print(f"{'mass bin':<28} {'fold values (12.7/6.2)':<24} {'fold mean':>10} {'pooled fit':>11} {'+/- fold/sqrt3':>15}")
for i in range(len(MASS_BINS)):
    v = np.array([r[2] for r in fold_results[i]])
    m, e = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    means.append(m); errs.append(e)
    print(f"{labels[i]:<28} {str(np.round(v, 2)):<24} {m:>10.2f} {pooled_flux[i]:>11.2f} {e:>15.2f}")

print("\nAdjacent-mass-bin separation (pooled centrals, fold-ensemble errors):")
for i in range(len(MASS_BINS) - 1):
    sep = (pooled_flux[i] - pooled_flux[i + 1]) / np.sqrt(errs[i] ** 2 + errs[i + 1] ** 2)
    print(f"  {labels[i]} ({pooled_flux[i]:.2f}+/-{errs[i]:.2f})  vs  "
          f"{labels[i+1]} ({pooled_flux[i+1]:.2f}+/-{errs[i+1]:.2f})  ->  {sep:.2f} sigma")


# Cross-check against the documented Sec 4a values (branch-7 brief Sec 1a).
# The documented numbers are FOLD-ENSEMBLE MEANS; the check verifies those
# reproduce. The pooled fit is the canonical central value going forward
# (2026-07-05 convention fix: pooled = measurement, folds = error bar only).
DOCUMENTED = [(72.4, 23.1), (12.8, 4.0), (3.7, 0.5), (0.8, 0.6)]
print("\nreproduction check vs documented fold means (brief Sec 1a):")
for i, (dm, de) in enumerate(DOCUMENTED):
    print(f"  bin {i}: fold mean {means[i]:6.1f} +/- {errs[i]:4.1f}   "
          f"documented {dm:6.1f} +/- {de:4.1f}   "
          f"{'MATCH' if abs(means[i]-dm) < 0.15*max(abs(dm),1) + 1e-9 else 'DIFFERS'}"
          f"   -> pooled central {pooled_flux[i]:6.1f}")
'''
)

code(
    r'''bin_ctrs = np.array([0.5 * (lo + hi) for lo, hi, *_ in MASS_BINS])
fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(bin_ctrs, pooled_flux, yerr=errs, fmt="o-", ms=9, color="C3", capsize=5, elinewidth=1.5,
            label="pooled 3-fold fit +/- fold scatter/sqrt(3)")
for fi in range(3):
    ys = [fold_results[i][fi][2] if fi < len(fold_results[i]) else np.nan
          for i in range(len(MASS_BINS))]
    ax.plot(bin_ctrs, ys, "o", ms=4, alpha=0.4, color="0.5")
ax.set_yscale("log")
ax.set_xlabel(r"$\log M_*/M_\odot$")
ax.set_ylabel(r"$r_{12.7\mu m} / r_{6.2\mu m}$  (neutral-PAH / ionized-PAH band ratio)")
ax.set_title("PAH band-ratio shift vs mass, with fold-ensemble errors\n"
              "(small grey points: the 3 individual folds behind the mean)")
ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.4)
ax.legend(fontsize=9); ax.grid(alpha=0.15, which="both")
plt.tight_layout()
fig.savefig("pah_money_bandratio_vs_mass.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

add2b_md = r"""### 2b · Envelope-aware re-derivation (2026-07-03 fix applied)

`fit_shared` has a constant-FLUX feature term, and each feature group's
amplitude is measured in its own redshift window (12.7 µm at z≈0.7–1.1,
7.7+8.6 at z≈1.4–2.5, 6.2 µm at z≈2.6–3.4) — so each amplitude absorbs the
mean source-dimming envelope of *its* window. The fitted ratio
r₁₂.₇/r₆.₂ is then (intrinsic band ratio) × (envelope ratio between the two
windows) ≈ ×4 — the same factor in every mass bin (the envelope shape is
mass-independent), so the **mass trend and σ-separations are unaffected**,
but the **absolute calibration** ("parity at high mass") is
envelope-contaminated. The envelope-aware static fit below
(`fit_evolving`, evolution off, `feature_envelope="baseline"`, 24 µm only)
measures the intrinsic template ratios — these are the values to compare
against literature band ratios (Xie & Ho 2022, Whitcomb+24)."""
md(add2b_md)

code(
    r'''# Envelope-aware per-bin per-fold fits (same structure, features dim with source)
fold_results_env = {i: [] for i in range(len(MASS_BINS))}
for fi, dff in enumerate(fold_dfs_sm):
    for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
        sub = dff[dff["prop_bin_id"] == i].copy()
        sub["prop_bin_id"] = 0
        if len(sub) < 8:
            continue
        res_i = _kmodel.fit_evolving(
            sub, evolve_amp=False, evolve_ratios=False,
            baseline_cols={"MIPS_24": "f24_cold"}, feature_envelope="baseline")
        if res_i is None:
            continue
        fold_results_env[i].append(res_i["r"])

# POOLED envelope-aware per-bin fits (canonical central values)
pooled_env = []
for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
    sub = df_pool_sm[df_pool_sm["prop_bin_id"] == i].copy()
    sub["prop_bin_id"] = 0
    res_i = _kmodel.fit_evolving(sub, evolve_amp=False, evolve_ratios=False,
                                 baseline_cols={"MIPS_24": "f24_cold"},
                                 feature_envelope="baseline")
    pooled_env.append(res_i["r"][2])

means_env, errs_env = [], []
print(f"{'mass bin':<28} {'env-aware pooled':>17} {'fold mean':>10} {'+/- fold/sqrt3':>15} {'flux-amp pooled':>16}")
for i in range(len(MASS_BINS)):
    v = np.array([r[2] for r in fold_results_env[i]])
    m, e = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    means_env.append(m); errs_env.append(e)
    print(f"{labels[i]:<28} {pooled_env[i]:>17.2f} {m:>10.2f} {e:>15.2f} {pooled_flux[i]:>16.2f}")
print("\nwindow-envelope bias (flux-amp/env-aware, pooled):",
      np.round(np.array(pooled_flux) / np.array(pooled_env), 2))
print("\nAdjacent-mass-bin separation, envelope-aware pooled (sigma):")
for i in range(len(MASS_BINS) - 1):
    sep = (pooled_env[i] - pooled_env[i + 1]) / np.sqrt(errs_env[i]**2 + errs_env[i+1]**2)
    print(f"  {labels[i]}  vs  {labels[i+1]}  ->  {sep:.2f} sigma")
'''
)

code(
    r'''fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(bin_ctrs, pooled_flux, yerr=errs, fmt="o--", ms=7, color="0.6", capsize=4,
            label="flux-amplitude ratios (Sec 2, window-envelope contaminated)")
ax.errorbar(bin_ctrs, pooled_env, yerr=errs_env, fmt="o-", ms=9, color="C3",
            capsize=5, elinewidth=1.5,
            label="envelope-aware pooled fit (intrinsic template ratios)")
for fi in range(3):
    ys = [fold_results_env[i][fi][2] if fi < len(fold_results_env[i]) else np.nan
          for i in range(len(MASS_BINS))]
    ax.plot(bin_ctrs, ys, "o", ms=4, alpha=0.4, color="C3", mfc="none")
ax.set_yscale("log")
ax.set_xlabel(r"$\log M_*/M_\odot$")
ax.set_ylabel(r"$r_{12.7\mu m} / r_{6.2\mu m}$  (neutral-PAH / ionized-PAH)")
ax.set_title("PAH band-ratio shift vs mass: envelope-aware calibration\n"
             "(trend preserved; absolute values shift by the common window-envelope factor)")
ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.4)
ax.legend(fontsize=8.5)
ax.grid(alpha=0.15, which="both")
plt.tight_layout()
fig.savefig("pah_money_bandratio_vs_mass_envaware.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""### 2c · The model against the data, coloured by measured sSFR

Rework of the letter-notebook §4e figure (a first attempt here failed for two
reasons, both fixed): it needed the per-bin measured sSFR, which this
notebook's DataFrame builder now extracts (`lp_sSFR_med` →
`log_ssfr_measured`; deliberately NOT named `log_ssfr`, so the §3 fits keep
their main-sequence proxy and remain strict re-derivations), and it overlaid
the old per-bin `fit_shared` model — the overlay below is the current one:
the pooled, envelope-aware `fit_evolving_mcmc` posterior, decomposed into
baseline + feature-group contributions. Points are the pooled stacked fluxes
coloured by each bin's median measured sSFR; the smooth baseline (§2c first
panel shows raw-vs-smoothed) is what makes the model bands smooth. Two views
of the same posterior: the standard decomposition figure (credible band,
plain points) and the sSFR-coloured variant."""
)

code(
    r'''# Baseline sanity: raw per-bin Wien extrapolation vs smoothed MS relation
fig, axes = plt.subplots(1, len(MASS_BINS), figsize=(5 * len(MASS_BINS), 4), sharey=True)
for ax, (i, (m_lo, m_hi, col, lbl)) in zip(axes, enumerate(MASS_BINS)):
    sub = df_pool_sm[df_pool_sm["prop_bin_id"] == i].sort_values("z_mid")
    ax.scatter(sub["z_mid"], sub["f24_cold_raw"] * 1e3, s=18, color="0.6",
               alpha=0.6, label="raw per-bin")
    o = np.argsort(sub["z_mid"].values)
    ax.plot(sub["z_mid"].values[o], sub["f24_cold_smooth"].values[o] * 1e3,
            "-", color=col, lw=2.2, label="smoothed MS")
    ax.set_title(lbl, fontsize=9)
    ax.set_xlabel("z")
    if i == 0:
        ax.set_ylabel(r"$f_{24,\rm cold}$ [mJy]")
    ax.set_yscale("log")
    ax.grid(alpha=0.15)
    ax.legend(fontsize=8)
fig.suptitle("Baseline: raw per-bin Wien extrapolation vs smoothed main-sequence relation")
plt.tight_layout()
plt.show()
'''
)

code(
    r'''# Pooled envelope-aware evolving MCMC on the smoothed df (mJy working copy).
# Reference group FIRST (7.7+8.6 um) per the branch-7 identifiability lesson.
from simstack4.pah_spectrum import evolving_flux_decomposition

GROUPS_DEC = [[1, 2], [0], [4]]
model_dec = PAHSpectrumModel(feature_groups=GROUPS_DEC, bands=("MIPS_24", "MIPS_70"),
                             sigma_z0=SIGMA_Z0, f_cat=0.03)
df_mjy = df_pool_sm.copy()
for c in ["MIPS_24", "MIPS_24_err", "MIPS_70", "MIPS_70_err", "f24_cold", "f70_cold"]:
    df_mjy[c] = 1e3 * df_mjy[c]

evolving = model_dec.fit_evolving_mcmc(df_mjy, feature_envelope="baseline",
                                       eta_prior_sigma=1.0,
                                       n_walkers=32, n_steps=800, n_burn=300, seed=2)
print(f"pooled evolving MCMC: chi2_red = {evolving['chi2_red']:.2f}, "
      f"eta_A = {evolving['eta_amp']:+.3f} +/- {evolving['eta_amp_err']:.3f}, "
      f"r = {np.round(evolving['r'], 3)}")
dec = evolving_flux_decomposition(evolving, n_draws=100, seed=3)
'''
)

code(
    r'''# The standard decomposition view first (same machinery as the evolving-MCMC
# notebook's Sec 7 figure): posterior-median baseline + stacked feature-group
# wedges + 68% band, data as plain points.
from simstack4.plots import plot_pah_flux_decomposition

fig = plot_pah_flux_decomposition(
    dec, band="MIPS_24",
    mass_labels=[lbl for *_, lbl in MASS_BINS],
    save_path="pah_money_f24_decomposition.png",
)
plt.show()
'''
)

code(
    r'''# f24 vs z per mass bin: data coloured by measured sSFR, model decomposition under it
from simstack4.plots import _PAH_GROUP_COLORS

dec_labels = ["7.7+8.6 um", "6.2 um", "12.7 um"]   # GROUPS_DEC order
_ssfr = df_mjy["log_ssfr_measured"].replace([np.inf, -np.inf], np.nan)
vmin, vmax = np.nanpercentile(_ssfr.dropna(), [5, 95])
norm = plt.Normalize(vmin=vmin, vmax=vmax)

fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), sharex=True)
contrib_cols = [c for c in dec.columns if c.startswith("contrib_")]
for k, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
    ax = axes[k // 2, k % 2]
    dm = dec[(dec["prop_bin_id"] == k) & (dec["band"] == "MIPS_24")].sort_values("z_mid")
    if len(dm):
        z = dm["z_mid"].to_numpy()
        cum = dm["baseline"].to_numpy().copy()
        ax.plot(z, cum, "--", color="0.4", lw=1.5, label="cold baseline", zorder=2)
        for g, cc in enumerate(contrib_cols):
            top = cum + dm[cc].to_numpy()
            ax.fill_between(z, cum, top, color=_PAH_GROUP_COLORS[g % 5], alpha=0.55,
                            lw=0, label=dec_labels[g], zorder=1)
            cum = top
        ax.plot(z, dm["total"].to_numpy(), "k-", lw=1.2, zorder=3, label="model total")
    sub = df_mjy[df_mjy["prop_bin_id"] == k]
    ax.errorbar(sub["z_mid"], sub["MIPS_24"], yerr=sub["MIPS_24_err"], fmt="none",
                ecolor="0.65", elinewidth=0.8, capsize=2, zorder=4)
    sc = ax.scatter(sub["z_mid"], sub["MIPS_24"], c=sub["log_ssfr_measured"],
                    cmap="Blues", norm=norm, s=40, edgecolor="0.25", linewidth=0.4,
                    zorder=5, label="stacked flux (colour: sSFR)")
    ax.set_title(f"MIPS 24 um -- {lbl}", fontsize=10)
    ax.grid(alpha=0.15)
    if k % 2 == 0:
        ax.set_ylabel("stacked flux [mJy]")
    if k // 2 == 1:
        ax.set_xlabel("redshift")
    if k == 0:
        ax.legend(fontsize=8, ncol=2, loc="upper right")
cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.01)
cb.set_label(r"median measured $\log\,{\rm sSFR}$ [yr$^{-1}$]")
fig.suptitle("Pooled stacks vs the envelope-aware evolving model, coloured by measured sSFR",
             fontsize=12)
fig.savefig("pah_money_f24_vs_z_ssfr.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""## 3 · Money plot 2 — the Narayanan+26 confrontation (§5–6)

Measured fold-ensemble L_PAH/L_IR mass slope (free Wien-side α, bolometric
L_PAH convention) against the two mass-axis channels derived from the
Narayanan et al. (2026) shattering model: **branch A** (density chain —
f_mol is saturated at cosmic noon so the Σ_SFR anti-correlation dominates,
negative slopes) and **branch B** (enrichment/PZR chain, positive slopes).
Channel-band derivation is verbatim from the confrontation notebook §5 (our
construction from their mechanism + standard scaling relations — the paper
publishes no q_PAH(M\*) at fixed z).

The slope fit is `fit_with_alpha` (static, 24+70 µm) → runs through the
2026-07-03 multi-band normalization fix; the cell prints the re-derived
fold-ensemble slope next to the documented +0.236 ± 0.076."""
)

code(
    r'''def lshape_at_z(gb, z, r_ratios, feature_groups, features=None):
    """Bolometric luminosity (L_sun) of a unit-height, r_g-weighted PAH
    line template at redshift z -- mirrors _pah_flux_lir's internal
    normalization exactly, so alpha_m * lshape_at_z(z) = L_PAH(z)."""
    features = DEFAULT_FEATURES if features is None else features
    lam_fine = np.logspace(np.log10(4.0), np.log10(20.0), 400)
    weights = group_weights(features, feature_groups)
    shape = np.zeros_like(lam_fine)
    for g, (grp, w) in enumerate(zip(feature_groups, weights)):
        r_g = r_ratios[g] if g < len(r_ratios) else 0.0
        for j, wj in zip(grp, w):
            center, _, fwhm = features[j]
            sigma = fwhm / 2.355
            shape += r_g * wj * np.exp(-0.5 * ((lam_fine - center) / sigma) ** 2)
    nu_fine = gb.c * 1.0e6 / lam_fine
    D_L_m = gb.luminosity_distance(z) * 3.08568025e22
    L_watts = 4.0 * np.pi * D_L_m**2 * 1e-26 * (-np.trapezoid(shape, nu_fine)) / (1.0 + z)
    return L_watts / gb.L_sun


def inband_unit_lum(gb, z, r_ratios, feature_groups):
    """nu*L_nu luminosity (L_sun) captured in the observed MIPS-24 band by a
    unit-height, r_g-weighted PAH template at z -- the June-28 'in-band'
    convention. Sharp-z (no photo-z smear; smearing redistributes in-band
    flux across neighbouring z bins but barely changes the per-bin mean)."""
    K = feature_band_curves(np.array([z]), "MIPS_24",
                            feature_groups=feature_groups)[0]
    f_pah = float(K @ np.asarray(r_ratios, float))     # Jy per unit alpha_m
    nu24 = gb.c * 1.0e6 / 24.0                          # Hz, observed frame
    D_L_m = gb.luminosity_distance(z) * 3.08568025e22
    L_watts = 4.0 * np.pi * D_L_m**2 * 1e-26 * f_pah * nu24 / (1.0 + z)
    return L_watts / gb.L_sun


def lpah_lir_by_bin(dff, alpha_m, r_ratios, alpha_wien_value,
                    lpah_def="bolo", label="", verbose=True):
    """L_PAH/L_IR per mass bin (n_sources-weighted mean over points) and the
    unweighted log-linear mass slope -- letter 9a per-point loop, generalized
    to (frame, L_PAH definition)."""
    _gb = _Greybody()
    _gb.alpha_wien = alpha_wien_value
    lum_fn = {"bolo": lshape_at_z, "inband": inband_unit_lum}[lpah_def]
    rows = []
    for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
        sub = dff[(dff["prop_bin_id"] == i) & np.isfinite(dff["T_dust"])
                  & np.isfinite(dff["log_amp"]) & np.isfinite(dff["beta"])]
        ratios, weights = [], []
        for _, row in sub.iterrows():
            z_i = row["z_mid"]
            L_IR_i, _ = _gb.calculate_LIR(row["log_amp"], row["T_dust"], row["beta"], z_i)
            if not (np.isfinite(L_IR_i) and L_IR_i > 0):
                continue
            L_unit_i = lum_fn(_gb, z_i, r_ratios, FEATURE_GROUPS)
            L_PAH_i = alpha_m[i] * L_unit_i
            ratios.append(L_PAH_i / L_IR_i)
            weights.append(float(row["n_sources"]) if np.isfinite(row["n_sources"]) else 1.0)
        ratios, weights = np.array(ratios), np.array(weights)
        if len(ratios) == 0:
            rows.append({"logM": 0.5 * (m_lo + m_hi), "ratio": np.nan,
                         "scatter": np.nan, "n": 0})
            continue
        w = weights / weights.sum()
        rows.append({"logM": 0.5 * (m_lo + m_hi),
                     "ratio": float(np.sum(w * ratios)),
                     "scatter": float(np.std(ratios)) if len(ratios) > 2 else np.nan,
                     "n": len(ratios)})
    out = pd.DataFrame(rows)
    ok = np.isfinite(out["ratio"]) & (out["ratio"] > 0)
    if ok.sum() >= 2:
        a, d = np.polyfit(out.loc[ok, "logM"], np.log10(out.loc[ok, "ratio"]), 1)
    else:
        a, d = np.nan, np.nan
    if verbose:
        print(f"[{label}] alpha_wien={alpha_wien_value:.2f} lpah={lpah_def}: "
              + "  ".join(f"{r*100:.2f}%" for r in out["ratio"])
              + f"   ->  slope = {a:+.4f} dex/dex")
    return out, a, d


bin_ctrs = np.array([0.5 * (lo + hi) for lo, hi, *_ in MASS_BINS])
'''
)

code(
    r'''_emodel = PAHSpectrumModel(feature_groups=FEATURE_GROUPS, bands=("MIPS_24", "MIPS_70"),
                           sigma_z0=SIGMA_Z0, f_cat=0.03)
_acols = {"MIPS_24": "f24_cold", "MIPS_70": "f70_cold"}

fold_results = {"alpha=2": [], "alpha free": []}
for k, dff in enumerate(fold_dfs_sm):
    res2 = _emodel.fit_with_alpha(
        dff, evolving=True, evolve_amp=False, evolve_ratios=False,
        baseline_cols=_acols, alpha_prior=(2.0, 1e-3), alpha_bounds=(1.9, 2.1))
    resf = _emodel.fit_with_alpha(
        dff, evolving=True, evolve_amp=False, evolve_ratios=False,
        baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0))
    for tag, res, aw in (("alpha=2", res2, 2.0),
                         ("alpha free", resf, float(resf["alpha_wien"]))):
        out, a, d = lpah_lir_by_bin(dff, np.asarray(res["alpha"]),
                                    np.asarray(res["r"]), aw,
                                    lpah_def="bolo", verbose=False)
        fold_results[tag].append({"fold": k, "slope": a, "intercept": d,
                                  "alpha_wien": aw,
                                  "ratios": out["ratio"].to_numpy()})
        print(f"  fold {k} [{tag:>10}]: alpha_wien={aw:.2f}  "
              + " ".join(f"{r*100:5.2f}%" for r in out["ratio"])
              + f"  slope={a:+.3f}")

print()
FOLD_SUMMARY = {}
for tag, folds in fold_results.items():
    slopes = np.array([f["slope"] for f in folds])
    R = np.vstack([f["ratios"] for f in folds])
    n = len(folds)
    FOLD_SUMMARY[tag] = {
        "slope_mean": float(np.nanmean(slopes)),
        "slope_err": float(np.nanstd(slopes, ddof=1) / np.sqrt(n)),
        "ratio_mean": np.nanmean(R, axis=0),
        "ratio_std": np.nanstd(R, axis=0, ddof=1),
    }
    s = FOLD_SUMMARY[tag]
    print(f"[{tag}] fold-ensemble L_PAH/L_IR mass slope = "
          f"{s['slope_mean']:+.3f} +/- {s['slope_err']:.3f} dex/dex "
          f"({s['slope_mean']/s['slope_err']:.1f} sigma)"
          if s["slope_err"] > 0 else f"[{tag}] slope err degenerate")

# Pooled reference at free alpha (the headline configuration), for the figure.
res_pool_free = _emodel.fit_with_alpha(
    df_pool_sm, evolving=True, evolve_amp=False, evolve_ratios=False,
    baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0))
ALPHA_WIEN_POOL = float(res_pool_free["alpha_wien"])
lir_pool, slope_pool, intercept_pool = lpah_lir_by_bin(
    df_pool_sm, np.asarray(res_pool_free["alpha"]), np.asarray(res_pool_free["r"]),
    ALPHA_WIEN_POOL, lpah_def="bolo",
    label=f"pooled, alpha_wien={ALPHA_WIEN_POOL:.2f}")
print(f"\nPooled reference: alpha_wien = {ALPHA_WIEN_POOL:.3f}, "
      f"slope = {slope_pool:+.3f} dex/dex "
      f"(pivot logM*=10.75: {10**(slope_pool*10.75+intercept_pool)*100:.2f}%)")


print("\nreproduction check: documented fold-ensemble (pre-normalization-fix)")
print("  alpha free: +0.236 +/- 0.076 (3.1 sigma), per-fold alpha_wien 3.17/3.27/2.40")
s = FOLD_SUMMARY["alpha free"]
print(f"  re-derived: {s['slope_mean']:+.3f} +/- {s['slope_err']:.3f} "
      f"({s['slope_mean']/s['slope_err']:.1f} sigma), per-fold alpha_wien "
      + "/".join(f"{f['alpha_wien']:.2f}" for f in fold_results["alpha free"]))
'''
)

code(
    r'''LOGM_GRID = np.linspace(10.0, 11.3, 27)
ZS = [1.0, 1.5, 2.0, 2.5]

# -- scaling relations ---------------------------------------------------
def mu_gas_tacconi18(z, logM, dms=0.0):
    """log10(M_H2 / M*), Tacconi+18 global fit (beta=2 form)."""
    return (0.12 - 3.62 * (np.log10(1.0 + z) - 0.66) ** 2
            + 0.53 * dms - 0.35 * (logM - 10.7))

def re_kpc_vdw14(z, logM):
    """Effective radius (kpc), van der Wel+14 late-type relation."""
    return 8.9 * (10.0 ** logM / 5e10) ** 0.22 * (1.0 + z) ** (-0.75)

def surface_density(mass_msun, re_kpc):
    """Half of the total within R_e, per pc^2."""
    return 0.5 * mass_msun / (np.pi * (re_kpc * 1e3) ** 2)

def galaxy_ism(z, logM):
    """Sigma_H2 [Msun/pc^2], Sigma_SFR [Msun/yr/kpc^2] on the MS."""
    m_h2 = 10.0 ** (mu_gas_tacconi18(z, logM) + logM)
    re = re_kpc_vdw14(z, logM)
    log_ssfr = np.array([main_sequence_ssfr(z, m) for m in np.atleast_1d(logM)])
    sfr = 10.0 ** (log_ssfr + np.atleast_1d(logM))
    sigma_h2 = surface_density(m_h2, re)
    sigma_sfr = 0.5 * sfr / (np.pi * re ** 2)
    return sigma_h2, sigma_sfr

def f_mol(sigma_h2, sigma_hi_sat):
    return sigma_h2 / (sigma_h2 + sigma_hi_sat)

# -- Branch A: density chain ---------------------------------------------
# S_q calibration from their z-evolution (see markdown above).
DLOG_QPAH_Z = np.log10(1e-2 / 5e-4)          # +1.30 dex, z=4 -> 0 (their Fig 5)
DFMOL_Z = (-0.3, -0.8)                        # bracketed f_mol drop, z=4 -> 0
SQ_BAND = tuple(DLOG_QPAH_Z / d for d in DFMOL_Z)      # dex per unit f_mol (<0)
SIGMA_HI_BAND = (5.0, 10.0, 20.0)

# Sigma_SFR channel: same calibration against Sigma_SFR(z) at fixed M*.
_s_lo = galaxy_ism(0.05, np.array([10.7]))[1][0]
_s_hi = galaxy_ism(4.0, np.array([10.7]))[1][0]
DLOG_SSFR_Z = np.log10(_s_hi / _s_lo)         # rise in Sigma_SFR, z=0 -> 4
SQS = -DLOG_QPAH_Z / DLOG_SSFR_Z              # dex q_PAH per dex Sigma_SFR (<0)
SQS_BAND = (0.7 * SQS, 1.4 * SQS)

G0_SYS = 0.10   # +/- dex/dex observable-mapping systematic (their decoupling)

print(f"S_q (f_mol channel)      : {SQ_BAND[0]:+.2f} .. {SQ_BAND[1]:+.2f} dex per unit f_mol")
print(f"Sigma_SFR(z=0->4, logM*=10.7) rises {DLOG_SSFR_Z:+.2f} dex "
      f"-> S_q,Sigma = {SQS:+.2f} (band {SQS_BAND[0]:+.2f} .. {SQS_BAND[1]:+.2f}) dex/dex")

rows = []
for z in ZS:
    sigma_h2, sigma_sfr = galaxy_ism(z, LOGM_GRID)
    dlog_ssfr_dlogm = np.polyfit(LOGM_GRID, np.log10(sigma_sfr), 1)[0]
    fm = {s: f_mol(sigma_h2, s) for s in SIGMA_HI_BAND}
    dfm_dlogm = {s: np.polyfit(LOGM_GRID, fm[s], 1)[0] for s in SIGMA_HI_BAND}
    slopes_A_fmol = [sq * dfm for sq in SQ_BAND for dfm in dfm_dlogm.values()]
    slopes_A_sfr = [sqs * dlog_ssfr_dlogm for sqs in SQS_BAND]
    slopes_A = slopes_A_fmol + slopes_A_sfr
    rows.append({
        "z": z,
        "f_mol(10.0)": fm[10.0][0], "f_mol(11.3)": fm[10.0][-1],
        "d f_mol/dlogM*": dfm_dlogm[10.0],
        "d logSigSFR/dlogM*": dlog_ssfr_dlogm,
        "A_lo": min(slopes_A) - G0_SYS, "A_hi": max(slopes_A) + G0_SYS,
    })
branchA = pd.DataFrame(rows)
print("\nBranch A (density chain) -- predicted d log(L_PAH/L_IR)/d log M*:")
print(branchA.round(3).to_string(index=False))

# -- Branch B: enrichment chain -------------------------------------------
GAMMA_MZR = (0.15, 0.30)     # dex Z per dex M* over our mass range at z~2
S_PZR = (0.0, 1.5)           # dex q_PAH per dex Z (0 = saturated PZR)
B_lo = GAMMA_MZR[0] * S_PZR[0] - G0_SYS
B_hi = GAMMA_MZR[1] * S_PZR[1] + G0_SYS
print(f"\nBranch B (enrichment chain): slope in [{B_lo:+.2f}, {B_hi:+.2f}] dex/dex "
      f"(gamma_MZR x s_PZR, +/- {G0_SYS} observable systematic)")

# z-averaged Branch A band over the redshifts that dominate our leverage
A_LO = float(branchA["A_lo"].min())
A_HI = float(branchA["A_hi"].max())
B_LO, B_HI = float(B_lo), float(B_hi)
print(f"\nBands used in the figure:  Branch A [{A_LO:+.2f}, {A_HI:+.2f}]   "
      f"Branch B [{B_LO:+.2f}, {B_HI:+.2f}]")

s_meas = FOLD_SUMMARY["alpha free"]
print(f"\nMeasured (fold ensemble, alpha free): "
      f"{s_meas['slope_mean']:+.3f} +/- {s_meas['slope_err']:.3f} dex/dex")
for tag in ("alpha=2", "alpha free"):
    s = FOLD_SUMMARY[tag]
    inA = A_LO <= s["slope_mean"] <= A_HI
    inB = B_LO <= s["slope_mean"] <= B_HI
    tA = (s["slope_mean"] - A_HI) / s["slope_err"] if not inA else 0.0
    print(f"  [{tag}] inside Branch A band: {inA}"
          + (f" (above its upper edge by {tA:+.1f} sigma)" if not inA and tA > 0 else "")
          + f"; inside Branch B band: {inB}")
'''
)

code(
    r'''PIVOT = 10.75
mgrid = np.linspace(bin_ctrs.min() - 0.1, bin_ctrs.max() + 0.1, 60)
ratio_piv = 10.0 ** (slope_pool * PIVOT + intercept_pool)

fig, ax = plt.subplots(figsize=(7.5, 5.5))

def band(lo, hi, color, label):
    lo_line = ratio_piv * 10.0 ** (lo * (mgrid - PIVOT)) * 100
    hi_line = ratio_piv * 10.0 ** (hi * (mgrid - PIVOT)) * 100
    ax.fill_between(mgrid, lo_line, hi_line, color=color, alpha=0.18, lw=0)
    ax.plot(mgrid, lo_line, color=color, lw=1.0, alpha=0.55)
    ax.plot(mgrid, hi_line, color=color, lw=1.0, alpha=0.55, label=label)

band(A_LO, A_HI, "C0",
     f"Narayanan+26 branch A: density/shattering chain [{A_LO:+.2f}, {A_HI:+.2f}]")
band(B_LO, B_HI, "C1",
     f"Narayanan+26 branch B: enrichment/PZR chain [{B_LO:+.2f}, {B_HI:+.2f}]")

s = FOLD_SUMMARY["alpha free"]
ax.errorbar(bin_ctrs, lir_pool["ratio"] * 100,
            yerr=np.maximum(s["ratio_std"], 1e-9) * 100,
            fmt="o", ms=9, capsize=4, color="C3", zorder=5,
            label=(f"measured (pooled, $\\alpha_w$={ALPHA_WIEN_POOL:.2f}; "
                   "errors = fold scatter)"))
ax.plot(mgrid, 10 ** (slope_pool * mgrid + intercept_pool) * 100, "-", color="C3",
        lw=1.8, alpha=0.9)
ax.annotate(f"measured slope = {s['slope_mean']:+.3f} $\\pm$ {s['slope_err']:.3f} dex/dex",
            xy=(0.03, 0.95), xycoords="axes fraction", fontsize=10,
            color="C3", va="top")

# z~0 anchor -- directional context only (all-band PAH vs our partial template)
ax.axhspan(10.0, 13.0, color="0.55", alpha=0.13, lw=0)
ax.annotate("z$\\approx$0 SINGS total-PAH/$L_{\\rm TIR}$ (Smith+07; all PAH bands --\n"
            "not directly comparable to our partial template)",
            xy=(0.03, 0.80), xycoords="axes fraction", fontsize=8,
            color="0.35", va="top")

ax.set_xlabel(r"$\log\, M_*/M_\odot$")
ax.set_ylabel(r"$L_{\rm PAH}/L_{\rm IR}$  [%]")
ax.set_yscale("log")
ax.set_title("PAH-to-IR ratio vs stellar mass at cosmic noon:\n"
             "measurement vs the two mass-axis channels of the shattering model")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.15, which="both")
plt.tight_layout()
fig.savefig("pah_money_narayanan_confrontation.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""### 3b · Envelope-aware re-derivation of the L_PAH/L_IR slope

The §3 estimator takes the fitted constant-flux amplitude α_m and converts
it to L_PAH at every point's z with the full 4πD_L² factor — but under real
source dimming the per-point feature FLUX is α_m·env_i, not α_m. Two
envelope-aware variants bracket the estimator systematic: **static +
envelope** (direct analog of the §3 configuration) and **envelope + η_A
amplitude evolution** (the model the §7 real-data comparison prefers; the
per-point amplitude then also carries 10^(η_A·ŝ_i)). Fits are
`fit_with_alpha` (24+70, free α) with `feature_envelope="baseline"`."""
)

code(
    r'''def lpah_lir_by_bin_env(dff, res, alpha_wien_value, evolving_amp=False, label=""):
    """Envelope-aware per-point L_PAH: alpha_m * env_i [* 10^(eta_A shat_i)],
    env_i = alpha-tilted f24_cold_i / per-bin median (mirrors _evolving_data)."""
    _gb = _Greybody()
    _gb.alpha_wien = alpha_wien_value
    alpha_m = np.asarray(res["alpha"])
    r_ratios = np.asarray(res["r"])
    dt = dff.copy()
    dt["f24_cold_t"] = dt["f24_cold"] * (1 + dt["z_mid"]) ** (2.0 - alpha_wien_value)
    rows = []
    for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
        sub = dt[(dt["prop_bin_id"] == i) & np.isfinite(dt["T_dust"])
                 & np.isfinite(dt["log_amp"]) & np.isfinite(dt["beta"])]
        okm = (np.isfinite(sub["MIPS_24"]) & (sub["MIPS_24"] > 0)
               & np.isfinite(sub["MIPS_24_err"]) & (sub["MIPS_24_err"] > 0)
               & np.isfinite(sub["f24_cold_t"]) & (sub["f24_cold_t"] > 0))
        if okm.sum() < 3:
            rows.append({"logM": 0.5 * (m_lo + m_hi), "ratio": np.nan,
                         "scatter": np.nan, "n": 0})
            continue
        med = float(np.median(sub.loc[okm, "f24_cold_t"]))
        logM_bin = float(sub["log_M_star"].iloc[0])
        ratios, weights = [], []
        for _, row in sub.iterrows():
            z_i = row["z_mid"]
            L_IR_i, _ = _gb.calculate_LIR(row["log_amp"], row["T_dust"], row["beta"], z_i)
            if not (np.isfinite(L_IR_i) and L_IR_i > 0 and np.isfinite(row["f24_cold_t"])
                    and row["f24_cold_t"] > 0):
                continue
            env_i = row["f24_cold_t"] / med
            evol_i = (10.0 ** (res["eta_amp"]
                               * (main_sequence_ssfr(z_i, logM_bin) - res["s_pivot"]))
                      if evolving_amp else 1.0)
            L_PAH_i = alpha_m[i] * env_i * evol_i * lshape_at_z(_gb, z_i, r_ratios,
                                                                FEATURE_GROUPS)
            ratios.append(L_PAH_i / L_IR_i)
            weights.append(float(row["n_sources"]) if np.isfinite(row["n_sources"]) else 1.0)
        ratios, weights = np.array(ratios), np.array(weights)
        if len(ratios) == 0:
            rows.append({"logM": 0.5 * (m_lo + m_hi), "ratio": np.nan,
                         "scatter": np.nan, "n": 0})
            continue
        w = weights / weights.sum()
        rows.append({"logM": 0.5 * (m_lo + m_hi), "ratio": float(np.sum(w * ratios)),
                     "scatter": float(np.std(ratios)) if len(ratios) > 2 else np.nan,
                     "n": len(ratios)})
    out = pd.DataFrame(rows)
    ok = np.isfinite(out["ratio"]) & (out["ratio"] > 0)
    if ok.sum() >= 2:
        a, d = np.polyfit(out.loc[ok, "logM"], np.log10(out.loc[ok, "ratio"]), 1)
    else:
        a, d = np.nan, np.nan
    print(f"[{label}] alpha_wien={alpha_wien_value:.2f}: "
          + "  ".join(f"{r*100:.2f}%" for r in out["ratio"])
          + f"   ->  slope = {a:+.4f} dex/dex")
    return out, a, d


fold_env = {"envelope static": [], "envelope + eta_A": []}
for k, dff in enumerate(fold_dfs_sm):
    res_s = _emodel.fit_with_alpha(
        dff, evolving=True, evolve_amp=False, evolve_ratios=False,
        baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0),
        feature_envelope="baseline")
    res_e = _emodel.fit_with_alpha(
        dff, evolving=True, evolve_amp=True, evolve_ratios=False,
        baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0),
        feature_envelope="baseline", eta_prior_sigma=1.0)
    for tag, res, ev in (("envelope static", res_s, False),
                         ("envelope + eta_A", res_e, True)):
        extra = f" eta_A={res['eta_amp']:+.2f}" if ev else ""
        out, a, d = lpah_lir_by_bin_env(dff, res, float(res["alpha_wien"]),
                                        evolving_amp=ev,
                                        label=f"fold {k} [{tag}]{extra}")
        fold_env[tag].append(a)

print()
print("L_PAH/L_IR mass slope, fold ensembles:")
print(f"  Sec 3 original estimator (constant-flux features): +0.234 +/- 0.077")
for tag, slopes in fold_env.items():
    v = np.array(slopes)
    print(f"  {tag:<18}: {v.mean():+.3f} +/- {v.std(ddof=1)/np.sqrt(len(v)):.3f} "
          f"(folds: " + " ".join(f"{x:+.3f}" for x in v) + ")")
print("\nBranch bands for reference: A (density) [-0.35, +0.09], B (enrichment) [-0.10, +0.55]")
'''
)

md(
    r"""### 3c · Redshift-resolved L_PAH/L_IR vs stellar mass

Can the money plots be split into z~1 / z~2 / z~3 lines? **Money plot 1
cannot** — at 24 µm the 12.7 µm feature is only in-band at z ≈ 0.7–1.1 and
6.2 µm only at z ≈ 2.6–3.4, so their ratio is intrinsically a cross-redshift
construct. **Money plot 2 can**: per (mass bin × z slice) the feature
amplitude is refit (2-parameter WLS, envelope-aware), and converted to total
L_PAH with a fixed template.

**Template choice matters — a first version of this section was wrong.** A
single *global* template forced onto all mass bins produced spuriously
flat/declining slices, because §2b shows the true band ratios vary ~100×
across mass and each slice's amplitude is anchored by one feature window
(12.7 µm at z~1, 7.7+8.6 at z~2, 6.2 at z~3): under-weighting a bin's
dominant feature misconverts that bin's slice level. The fit below uses
**per-mass-bin envelope-aware templates** (the §2b pooled per-bin fits).
Validation: (i) with an all-z window this estimator reproduces the §3 money
plot means; (ii) the resulting pattern — L_PAH/L_IR *rising* with z at low
mass, *falling* at high mass — independently matches the letter-notebook
§5b non-parametric sSFR median-split. Absolute slice levels still inherit
the template conversion; the crossing pattern and the per-line trends are
the robust content."""
)

code(
    r'''Z_SLICES = [(0.5, 1.4, "z~1"), (1.4, 2.4, "z~2"), (2.4, 3.5, "z~3")]

# Anchors: global envelope-aware alpha_wien + PER-MASS-BIN envelope-aware
# templates (the Sec 2b pooled per-bin fits, full r vectors).
res_env_pool = _emodel.fit_with_alpha(
    df_pool_sm, evolving=True, evolve_amp=False, evolve_ratios=False,
    baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0),
    feature_envelope="baseline")
AW_POOL = float(res_env_pool["alpha_wien"])
R_BINS = []
for i in range(len(MASS_BINS)):
    sub = df_pool_sm[df_pool_sm["prop_bin_id"] == i].copy()
    sub["prop_bin_id"] = 0
    ri = _kmodel.fit_evolving(sub, evolve_amp=False, evolve_ratios=False,
                              baseline_cols={"MIPS_24": "f24_cold"},
                              feature_envelope="baseline")
    R_BINS.append(np.asarray(ri["r"]))
print(f"alpha_wien (global, envelope-aware) = {AW_POOL:.2f}")
for i, r in enumerate(R_BINS):
    print(f"  per-bin template r (bin {i}):", np.round(r, 3))


def zslice_ratios(dff, r_by_bin, aw, z_windows):
    """L_PAH/L_IR per (mass bin, z window): envelope-aware amplitude refit per
    window with (per-bin r, alpha_wien) fixed; returns (n_bins, n_win)."""
    m24 = PAHSpectrumModel(feature_groups=FEATURE_GROUPS, bands=("MIPS_24",),
                           sigma_z0=SIGMA_Z0, f_cat=0.03)
    dt = dff.copy()
    dt["f24_cold"] = dt["f24_cold"] * (1 + dt["z_mid"]) ** (2.0 - aw)
    prep = m24._prepare(dt, None, None, None, None, None,
                        baseline_cols={"MIPS_24": "f24_cold"})
    _gb = _Greybody()
    _gb.alpha_wien = aw
    out = np.full((len(MASS_BINS), len(z_windows)), np.nan)
    for b in prep["bins"]:
        i = b["m"]
        r_ratios = np.asarray(r_by_bin[i])
        K = b["K"][:, 0, :]
        fcold = b["f_cold_by_band"]["MIPS_24"]
        f, e = b["F"][:, 0], b["Ferr"][:, 0]
        z = np.asarray(b["z_mid"], dtype=float)
        sub = dt[dt["prop_bin_id"] == i].sort_values(["run_id", "z_lo"]).reset_index(drop=True)
        ok = (np.isfinite(f) & np.isfinite(e) & (e > 0)
              & np.isfinite(fcold) & (fcold > 0) & (f > 0)
              & np.isfinite(sub["T_dust"].to_numpy())
              & np.isfinite(sub["log_amp"].to_numpy()))
        if ok.sum() < 6:
            continue
        med = float(np.median(fcold[ok]))
        env = fcold / med
        t_full = env * (K @ r_ratios)
        base = fcold / med
        for k, (zlo, zhi) in enumerate(z_windows):
            msk = ok & (z >= zlo) & (z < zhi)
            if msk.sum() < 4:
                continue
            D = np.column_stack([base[msk], t_full[msk]])
            w = 1.0 / e[msk] ** 2
            try:
                C_s, a_s = np.linalg.solve(D.T @ (w[:, None] * D), D.T @ (w * f[msk]))
            except np.linalg.LinAlgError:
                continue
            ratios, wts = [], []
            for j in np.where(msk)[0]:
                row = sub.iloc[j]
                L_IR, _ = _gb.calculate_LIR(row["log_amp"], row["T_dust"],
                                            row["beta"], row["z_mid"])
                if not (np.isfinite(L_IR) and L_IR > 0):
                    continue
                L_PAH = a_s * env[j] * lshape_at_z(_gb, row["z_mid"], r_ratios,
                                                   FEATURE_GROUPS)
                ratios.append(L_PAH / L_IR)
                wts.append(float(row["n_sources"]) if np.isfinite(row["n_sources"]) else 1.0)
            if ratios:
                wts = np.array(wts) / np.sum(wts)
                out[i, k] = float(np.sum(wts * np.array(ratios)))
    return out


wins = [(zlo, zhi) for zlo, zhi, _ in Z_SLICES]
zr_pool = zslice_ratios(df_pool_sm, R_BINS, AW_POOL, wins)
zr_folds = np.stack([zslice_ratios(dff, R_BINS, AW_POOL, wins) for dff in fold_dfs_sm])
zr_err = np.nanstd(zr_folds, axis=0, ddof=1) / np.sqrt(3)

# Validation: the same estimator with an all-z window should track the Sec 3
# money-plot means (it does; residual offsets are the per-bin-template vs
# global-template conversion).
allz = zslice_ratios(df_pool_sm, R_BINS, AW_POOL, [(0.5, 3.5)])[:, 0]
print(f"\n{'mass bin':<28}" + "".join(f"{lab:>16}" for *_, lab in Z_SLICES) + f"{'all-z':>10}{'Sec 3':>8}")
sec3_ref = 100 * lir_pool["ratio"].to_numpy()
for i in range(len(MASS_BINS)):
    cells = []
    for k in range(len(Z_SLICES)):
        v, ev = zr_pool[i, k], zr_err[i, k]
        cells.append(f"{100*v:6.2f}+/-{100*ev:4.2f}%" if np.isfinite(v) else "      --   ")
    print(f"{labels[i]:<28}" + "".join(f"{c:>16}" for c in cells)
          + f"{100*allz[i]:>9.2f}%{sec3_ref[i]:>7.2f}%")

'''
)

code(
    r'''zcols = plt.cm.Blues(np.linspace(0.45, 0.95, len(Z_SLICES)))
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    x = bin_ctrs + (k - 1) * 0.02
    okk = np.isfinite(zr_pool[:, k])
    ax.errorbar(x[okk], 100 * zr_pool[okk, k], yerr=100 * zr_err[okk, k],
                fmt="o-", ms=7, capsize=4, color=zcols[k],
                label=f"{lab}  (z = {zlo}-{zhi})")
ax.set_yscale("log")
ax.set_xlabel(r"$\log\, M_*/M_\odot$")
ax.set_ylabel(r"$L_{\rm PAH}/L_{\rm IR}$  [%]")
ax.set_title("PAH-to-IR ratio vs stellar mass, split by redshift slice\n"
             "(amplitude refit per slice; per-mass-bin envelope-aware templates)")
ax.grid(alpha=0.15, which="both")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig("pah_money_lpah_lir_vs_mass_zslices.png", dpi=150, bbox_inches="tight")
plt.show()

'''
)

md(
    r"""### 3d · The z-resolved confrontation: slice mass slopes vs the branch bands

The §5 branch derivation is already per-redshift (branch A's ingredients —
gas fractions, sizes, Σ_SFR — were evaluated at z = 1.0/1.5/2.0/2.5), so the
§3c slices can be confronted at their own epochs: measure each slice's mass
slope (pooled centrals, fold-ensemble errors) and compare with branch A
evaluated over that slice's z range (the z~3 slice uses the z=2.5 row — a
mild extrapolation) and the z-independent branch B band. Expectation from
the crossing pattern: the slope should run from enrichment-like (positive,
branch-B territory) at z~1 to destruction-like (negative, branch-A
territory) at z~3, with the all-z slope (+0.234) as the blend.

Caveats carried from §3c and §5: the bands are OUR construction from the
Narayanan+26 mechanism (the paper publishes no mass-slope prediction); the
slice slopes inherit the measured per-mass-bin templates (template
uncertainty not propagated); slices with a non-positive bin drop that bin
from the log-slope fit."""
)

code(
    r'''def _mass_slope(vals):
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(bin_ctrs[ok], np.log10(v[ok]), 1)[0])


slice_slopes = np.array([_mass_slope(zr_pool[:, k]) for k in range(len(Z_SLICES))])
fold_slopes = np.array([[_mass_slope(zr_folds[f][:, k]) for k in range(len(Z_SLICES))]
                        for f in range(len(fold_dfs_sm))])
n_ok = np.sum(np.isfinite(fold_slopes), axis=0)
slice_serrs = np.nanstd(fold_slopes, axis=0, ddof=1) / np.sqrt(np.maximum(n_ok, 1))

# Branch A band per slice from the Sec 5 per-z rows; branch B is z-independent
bandA_slice = []
for zlo, zhi, _ in Z_SLICES:
    rows = branchA[(branchA["z"] >= zlo) & (branchA["z"] <= zhi)]
    if len(rows) == 0:
        rows = branchA.iloc[[int(np.argmin(np.abs(branchA["z"] - 0.5 * (zlo + zhi))))]]
    bandA_slice.append((float(rows["A_lo"].min()), float(rows["A_hi"].max())))

print(f"{'slice':<8}{'mass slope':>18}{'branch A band':>19}{'in A?':>7}{'in B?':>7}")
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    alo, ahi = bandA_slice[k]
    sl, se = slice_slopes[k], slice_serrs[k]
    inA = alo <= sl <= ahi
    inB = B_LO <= sl <= B_HI
    print(f"{lab:<8}{sl:+11.3f}+/-{se:5.3f}   [{alo:+.2f}, {ahi:+.2f}]"
          f"{str(inA):>8}{str(inB):>7}")
print(f"\nall-z measurement (money plot 2): +0.234 +/- 0.077 "
      f"(branch B [{B_LO:+.2f}, {B_HI:+.2f}])")

zmids = np.array([0.5 * (zlo + zhi) for zlo, zhi, _ in Z_SLICES])
fig, ax = plt.subplots(figsize=(7.5, 5.2))
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    alo, ahi = bandA_slice[k]
    ax.fill_between([zlo, zhi], [alo, alo], [ahi, ahi], color="C0", alpha=0.20,
                    lw=0, label="Narayanan+26 branch A (density/shattering)" if k == 0 else None)
ax.axhspan(B_LO, B_HI, color="C1", alpha=0.13, lw=0,
           label="Narayanan+26 branch B (enrichment/PZR)")
ax.errorbar(zmids, slice_slopes, yerr=slice_serrs, fmt="o", ms=9, capsize=4,
            color="C3", zorder=5, label="slice mass slope (pooled, fold errors)")
ax.axhline(0.234, color="0.35", ls="--", lw=1.4,
           label="all-z slope +0.234 (money plot 2)")
ax.axhspan(0.234 - 0.077, 0.234 + 0.077, color="0.5", alpha=0.10, lw=0)
ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
ax.set_xlabel("redshift")
ax.set_ylabel(r"$d\,\log(L_{\rm PAH}/L_{\rm IR})\, /\, d\,\log M_*$  [dex/dex]")
ax.set_title("Mass slope of the PAH-to-IR ratio vs redshift:\n"
             "enrichment-like at z~1, destruction-like at z~3")
ax.grid(alpha=0.15)
ax.legend(fontsize=8.5, loc="lower left")
plt.tight_layout()
fig.savefig("pah_money_slice_slopes_vs_branches.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""## 4 · Verdict

- **Central values are the POOLED 3-fold fits** (2026-07-05 convention fix):
  the folds are disjoint catalog subsets sharing the field, so the pooled fit
  carries the dense interleaved wavelength sampling the dither design bought,
  and fold scatter serves only as a jackknife-style subsample error (it
  undercounts field-level cosmic variance and shared-map systematics — one
  field). Fold-mean centrals (the previously documented values) are biased
  10–40% high relative to the pooled fits.
- **Money plot 1**: use the §2b envelope-aware POOLED ratios for any
  literature comparison; the §2 flux-amplitude version is the internal-trend
  calibration only. The ratio itself cannot be redshift-resolved — its two
  features never share the bandpass at one z.
- **Money plot 2**: the documented slope survives both the normalization fix
  and the pooled-vs-fold convention (its central curve was always the pooled
  fit). §3b brackets the envelope-estimator systematic; §3c adds the
  z-resolved view — with PER-MASS-BIN templates (a global template
  spuriously flattens the slices) it shows a crossing pattern: L_PAH/L_IR
  rises with z at low mass and falls at high mass, matching the
  letter-notebook §5b non-parametric split; the rising mass trend lives in
  the all-z aggregate and steepens at z~1. §3d confronts each slice's mass
  slope with the branch bands at its own redshift: the slope runs from
  branch-B-like (positive) at z~1 to branch-A-like (negative) at z~3 — the
  two chains trading dominance with epoch, with the all-z +0.234 as the
  blend.
- **§2c** replaces the failed sSFR-coloured figure: measured `lp_sSFR_med`
  now flows through the DataFrame builder as `log_ssfr_measured` (kept
  distinct from `log_ssfr` so the §3 fits stay strict re-derivations), and
  the model overlay is the pooled envelope-aware evolving-MCMC decomposition,
  not the superseded per-bin `fit_shared`.
"""
)


nb["cells"] = cells
out = "notebooks/2026-07-03-pah-money-plots.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out} ({len(cells)} cells)")
