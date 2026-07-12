"""Generate 2026-07-10-pah-money-plots.ipynb.

Clean rebuild of 2026-07-07-pah-money-plots.ipynb (which accumulated a lot of
now-superseded scaffolding from live editing: the raw flux-amplitude band
ratio, the documented-value reproduction checks, duplicate cells). This
version keeps only the current-best-method path for each result and adds the
combined (non-split, non-K-fold) stack as a second, independent central-value
estimate everywhere a K-fold pooled number is quoted:

  1. Money plot 1 -- envelope-aware 12.7/6.2 band ratio vs stellar mass.
  2. Money plot 2a -- all-z L_PAH/L_IR vs stellar mass (Narayanan+26
     confrontation).
  3. Money plot 2b -- z-resolved crossing pattern, three independent
     estimators (pooled-template, self-consistent-per-fold, combined stack).
  4. Appendix -- bin0 template-SNR fix and the z>2 decomposition overlay,
     K-fold pooled vs combined.

K-fold stacks: cosmos2020_PAH_split{0,1,2}of3, widened z>2.6 bins + widened
mass bin0 (2026-07-07, RUN_DATES below). Combined stack: same widened bins,
full (non-split) catalog, 3 dither offsets, all complete as of 2026-07-08.

Requires PICKLESPATH.

Run:  uv run python notebooks/build_pah_money_plots_2026-07-10_notebook.py
Then: uv run jupyter nbconvert --to notebook --execute --inplace \
          notebooks/2026-07-10-pah-money-plots.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(
    r"""# PAH money plots: do the headline results survive the combined stack?

**2026-07-10.** Clean rebuild of `2026-07-07-pah-money-plots.ipynb`. Two
headline results, each checked against three estimators where possible:

- **pooled** (3 K-fold catalogs, jointly fit -- the dither design's dense
  wavelength sampling; fold scatter is the error bar, not 3 independent
  observations of the same field)
- **self-consistent per-fold** (branch-9 Objective 1 stress test -- each fold
  supplies its own template instead of reusing the pooled one; exposes
  template systematics the pooled-fold error misses)
- **combined** (the same catalog, NOT K-fold split, 3 dither-offset runs
  pooled -- 3x the sources per (z, mass) cell of any one K-fold, at the cost
  of no independent-subsample error bar of its own)

Both K-fold and combined stacks use the 2026-07-07 widened binning (z>2.6
locally coarsened; mass bin0 widened to 9.9-10.6) that fixed the bin0
reference-amplitude instability found in the narrower-bin version (§3f
degeneracy diagnostic, this notebook's appendix)."""
)

code(
    r'''import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from simstack4.wrapper import SimstackWrapper
from simstack4.plots import _extract_pop_type, plot_pah_flux_decomposition, _PAH_GROUP_COLORS
from simstack4.pah_spectrum import (
    PAHSpectrumModel, feature_band_curves, group_weights, DEFAULT_FEATURES,
    evolving_flux_decomposition,
)
from simstack4.greybody import Greybody as _Greybody
from simstack4.dust_evolution import main_sequence_ssfr

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

# K-fold: disjoint catalog folds, widened z>2.6 bins, 2026-07-07.
RUN_DATES = {
    0: "20260707_171122",   # cosmos2020_PAH_split0of3, offset 0.0000
    1: "20260707_155843",   # cosmos2020_PAH_split1of3, offset 0.0375
    2: "20260707_144533",   # cosmos2020_PAH_split2of3, offset 0.0750
}

# Combined: full (non-split) catalog, 3 dither-offset runs, all complete.
RUN_DATES_COMBINED = {
    0: "20260707_204926",   # offset 0.0000
    1: "20260707_225921",   # offset 0.0375
    2: "20260708_091724",   # offset 0.0750
}

# 4 science mass bins (9.0-9.9 stays an unanalysed nuisance layer in the config)
MASS_BINS = [
    (9.90, 10.6, "C0", r"$9.9 < \log M_* < 10.6$"),
    (10.6, 10.8, "C1", r"$10.6 < \log M_* < 10.8$"),
    (10.8, 11.0, "C2", r"$10.8 < \log M_* < 11.0$"),
    (11.0, 12.0, "C3", r"$\log M_* > 11.0$"),
]
# Feature groups: 0=6.2, 1=7.7, 2=8.6, 3=11.3(blind), 4=12.7, 5=16.4, 6=17.0
FEATURE_GROUPS = [[0], [1, 2], [4]]   # 6.2 | 7.7+8.6 | 12.7  (11.3 blind)
SIGMA_Z0 = 0.01   # sigma_z(1+z) for COSMOS2020 photo-z
bin_ctrs = np.array([0.5 * (lo + hi) for lo, hi, *_ in MASS_BINS])
labels = [lbl for _, _, _, lbl in MASS_BINS]
'''
)

md(r"""## 0 · Load both stacks, build the working DataFrames""")

code(
    r'''WRAPPERS = []
for k, date in RUN_DATES.items():
    w = SimstackWrapper()
    w.load_stacking_results(os.path.join(path_json, f"cosmos20_stacking_{date}.json"))
    w.run_analysis_only(**ANALYSIS_KWARGS)
    WRAPPERS.append(w)
print("Loaded 3 disjoint-fold stacking runs:", RUN_DATES)

WRAPPERS_COMBINED = []
for k, date in RUN_DATES_COMBINED.items():
    if not date:
        print(f"combined offset slot {k}: not run yet, skipping")
        continue
    w = SimstackWrapper()
    w.load_stacking_results(os.path.join(path_json, f"cosmos20_stacking_{date}.json"))
    w.run_analysis_only(**ANALYSIS_KWARGS)
    WRAPPERS_COMBINED.append(w)
print(f"Loaded {len(WRAPPERS_COMBINED)}/3 combined-stack offset runs:", RUN_DATES_COMBINED)

# The combined catalog isn't K-fold-partitioned -- confirm split_filter=[0]
# is still the right population-type filter rather than assuming.
pop_types_seen = set()
for w in WRAPPERS_COMBINED:
    pr = getattr(w, "processed_results", None)
    if pr is None:
        continue
    for pop_id in pr.sed_results:
        pop_types_seen.add(_extract_pop_type(pop_id))
print("population types found in the combined stack:", sorted(pop_types_seen))
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


def _design(z, dM, quad):
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


df_pool_raw = build_pah_spectrum_df(WRAPPERS, MASS_BINS, split_filter=[0], min_tier="C")
print("Pooled, smoothed:")
df_pool_sm = smooth_baseline(df_pool_raw)

fold_dfs_sm = []
for k, w in enumerate(WRAPPERS):
    d = build_pah_spectrum_df([w], MASS_BINS, split_filter=[0], min_tier="C")
    print(f"Fold {k} ({len(d)} points), smoothed:")
    fold_dfs_sm.append(smooth_baseline(d))

df_combined_raw = build_pah_spectrum_df(WRAPPERS_COMBINED, MASS_BINS, split_filter=[0], min_tier="C")
print("\nCombined, smoothed:")
df_combined_sm = smooth_baseline(df_combined_raw)

print(f"\nN per mass bin -- K-fold pooled (3 folds) vs combined ({len(WRAPPERS_COMBINED)}/3 offsets):")
for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
    n_pool = (df_pool_sm["prop_bin_id"] == i).sum()
    n_comb = (df_combined_sm["prop_bin_id"] == i).sum()
    print(f"  {lbl:<28} pooled={n_pool:>4}   combined={n_comb:>4}")
'''
)

md(
    r"""## 1 · Money plot 1 -- neutral/ionized band ratio vs stellar mass

Envelope-aware `fit_evolving` (evolution off, `feature_envelope="baseline"`,
24 µm only): features dim with the source, so the fitted ratio is the
intrinsic template ratio, not window-envelope-contaminated. `r[2]` is the
12.7 µm/6.2 µm neutral-over-ionized ratio (`r_0 ≡ 1` at 6.2 µm)."""
)

code(
    r'''_kmodel = PAHSpectrumModel(feature_groups=FEATURE_GROUPS, bands=("MIPS_24",),
                           sigma_z0=SIGMA_Z0, f_cat=0.03)
group_labels = ["+".join(str(_kmodel.features[j][0]) for j in g) + " um"
                for g in FEATURE_GROUPS]


def bandratio_env(dff):
    sub_full = dff
    out = []
    for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
        sub = sub_full[sub_full["prop_bin_id"] == i].copy()
        sub["prop_bin_id"] = 0
        if len(sub) < 8:
            out.append(np.nan)
            continue
        res_i = _kmodel.fit_evolving(sub, evolve_amp=False, evolve_ratios=False,
                                     baseline_cols={"MIPS_24": "f24_cold"},
                                     feature_envelope="baseline")
        out.append(np.nan if res_i is None else res_i["r"][2])
    return np.array(out)


fold_ratios = np.array([bandratio_env(dff) for dff in fold_dfs_sm])
pooled_ratio = bandratio_env(df_pool_sm)
combined_ratio = bandratio_env(df_combined_sm)
fold_err = np.nanstd(fold_ratios, axis=0, ddof=1) / np.sqrt(len(fold_dfs_sm))

print(f"{'mass bin':<28}{'pooled (K-fold)':>18}{'+/- fold/sqrt3':>15}{'combined':>12}")
for i in range(len(MASS_BINS)):
    print(f"{labels[i]:<28}{pooled_ratio[i]:>18.2f}{fold_err[i]:>15.2f}{combined_ratio[i]:>12.2f}")
print("\nAdjacent-bin separation, pooled K-fold (sigma):")
for i in range(len(MASS_BINS) - 1):
    sep = (pooled_ratio[i] - pooled_ratio[i+1]) / np.sqrt(fold_err[i]**2 + fold_err[i+1]**2)
    print(f"  {labels[i]} vs {labels[i+1]} -> {sep:.2f} sigma")
'''
)

code(
    r'''fig, ax = plt.subplots(figsize=(7, 5))
ax.errorbar(bin_ctrs, pooled_ratio, yerr=fold_err, fmt="o-", ms=9, color="C3",
            capsize=5, elinewidth=1.5, label="pooled K-fold (+/- fold scatter/sqrt3)")
ax.plot(bin_ctrs, combined_ratio, "^-", ms=10, color="C2",
        label=f"combined stack ({len(WRAPPERS_COMBINED)}/3 offsets)")
for fi in range(len(fold_dfs_sm)):
    ax.plot(bin_ctrs, fold_ratios[fi], "o", ms=4, alpha=0.4, color="0.5")
ax.set_yscale("log")
ax.set_xlabel(r"$\log M_*/M_\odot$")
ax.set_ylabel(r"$r_{12.7\mu m} / r_{6.2\mu m}$  (neutral-PAH / ionized-PAH)")
ax.set_title("PAH band-ratio vs mass: K-fold pooled vs combined stack\n"
              "(small grey points: the 3 individual K-folds)")
ax.axhline(1.0, color="k", lw=0.7, ls="--", alpha=0.4)
ax.legend(fontsize=9); ax.grid(alpha=0.15, which="both")
plt.tight_layout()
fig.savefig("pah_money_bandratio_vs_mass.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""## 2 · Money plot 2a -- L_PAH/L_IR vs stellar mass, all-z (Narayanan+26 confrontation)

Free Wien-side α (`fit_with_alpha`, 24+70 µm), bolometric L_PAH convention.
Channel bands are OUR construction from the Narayanan+26 shattering
mechanism (density/shattering = branch A, enrichment/PZR = branch B; the
paper publishes no q_PAH(M*) at fixed z)."""
)

code(
    r'''_emodel = PAHSpectrumModel(feature_groups=FEATURE_GROUPS, bands=("MIPS_24", "MIPS_70"),
                           sigma_z0=SIGMA_Z0, f_cat=0.03)
_acols = {"MIPS_24": "f24_cold", "MIPS_70": "f70_cold"}


def lshape_at_z(gb, z, r_ratios, feature_groups, features=None):
    """Bolometric luminosity (L_sun) of a unit-height, r_g-weighted PAH
    line template at redshift z -- alpha_m * lshape_at_z(z) = L_PAH(z)."""
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


def lpah_lir_by_bin(dff, alpha_m, r_ratios, alpha_wien_value, label="", verbose=True):
    """L_PAH/L_IR per mass bin (n_sources-weighted mean) and the unweighted
    log-linear mass slope."""
    _gb = _Greybody()
    _gb.alpha_wien = alpha_wien_value
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
            L_PAH_i = alpha_m[i] * lshape_at_z(_gb, z_i, r_ratios, FEATURE_GROUPS)
            ratios.append(L_PAH_i / L_IR_i)
            weights.append(float(row["n_sources"]) if np.isfinite(row["n_sources"]) else 1.0)
        ratios, weights = np.array(ratios), np.array(weights)
        if len(ratios) == 0:
            rows.append({"logM": 0.5 * (m_lo + m_hi), "ratio": np.nan, "n": 0})
            continue
        w = weights / weights.sum()
        rows.append({"logM": 0.5 * (m_lo + m_hi), "ratio": float(np.sum(w * ratios)), "n": len(ratios)})
    out = pd.DataFrame(rows)
    ok = np.isfinite(out["ratio"]) & (out["ratio"] > 0)
    if ok.sum() >= 2:
        a, d = np.polyfit(out.loc[ok, "logM"], np.log10(out.loc[ok, "ratio"]), 1)
    else:
        a, d = np.nan, np.nan
    if verbose:
        print(f"[{label}] alpha_wien={alpha_wien_value:.2f}: "
              + "  ".join(f"{r*100:.2f}%" for r in out["ratio"])
              + f"   ->  slope = {a:+.4f} dex/dex")
    return out, a, d


fold_slopes = []
for k, dff in enumerate(fold_dfs_sm):
    res_f = _emodel.fit_with_alpha(
        dff, evolving=True, evolve_amp=False, evolve_ratios=False,
        baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0))
    aw = float(res_f["alpha_wien"])
    _, a, _ = lpah_lir_by_bin(dff, np.asarray(res_f["alpha"]), np.asarray(res_f["r"]), aw,
                              label=f"fold {k}")
    fold_slopes.append(a)
fold_slopes = np.array(fold_slopes)
SLOPE_FOLD_MEAN = float(np.nanmean(fold_slopes))
SLOPE_FOLD_ERR = float(np.nanstd(fold_slopes, ddof=1) / np.sqrt(len(fold_slopes)))
print(f"\n[K-fold fold-ensemble] slope = {SLOPE_FOLD_MEAN:+.3f} +/- {SLOPE_FOLD_ERR:.3f} dex/dex")

res_pool_free = _emodel.fit_with_alpha(
    df_pool_sm, evolving=True, evolve_amp=False, evolve_ratios=False,
    baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0))
ALPHA_WIEN_POOL = float(res_pool_free["alpha_wien"])
lir_pool, SLOPE_POOL, INTERCEPT_POOL = lpah_lir_by_bin(
    df_pool_sm, np.asarray(res_pool_free["alpha"]), np.asarray(res_pool_free["r"]),
    ALPHA_WIEN_POOL, label=f"pooled K-fold, alpha_wien={ALPHA_WIEN_POOL:.2f}")

res_combined_free = _emodel.fit_with_alpha(
    df_combined_sm, evolving=True, evolve_amp=False, evolve_ratios=False,
    baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0))
ALPHA_WIEN_COMBINED = float(res_combined_free["alpha_wien"])
lir_combined, SLOPE_COMBINED, INTERCEPT_COMBINED = lpah_lir_by_bin(
    df_combined_sm, np.asarray(res_combined_free["alpha"]), np.asarray(res_combined_free["r"]),
    ALPHA_WIEN_COMBINED, label=f"combined, alpha_wien={ALPHA_WIEN_COMBINED:.2f}")

print(f"\nAll-z L_PAH/L_IR mass slope:")
print(f"  pooled K-fold (fold-ensemble error): {SLOPE_FOLD_MEAN:+.3f} +/- {SLOPE_FOLD_ERR:.3f}")
print(f"  pooled K-fold (pointwise fit)       : {SLOPE_POOL:+.3f}")
print(f"  combined stack (pointwise fit)      : {SLOPE_COMBINED:+.3f}")
'''
)

code(
    r'''LOGM_GRID = np.linspace(10.0, 11.3, 27)
ZS = [1.0, 1.5, 2.0, 2.5]

def mu_gas_tacconi18(z, logM, dms=0.0):
    return (0.12 - 3.62 * (np.log10(1.0 + z) - 0.66) ** 2
            + 0.53 * dms - 0.35 * (logM - 10.7))

def re_kpc_vdw14(z, logM):
    return 8.9 * (10.0 ** logM / 5e10) ** 0.22 * (1.0 + z) ** (-0.75)

def surface_density(mass_msun, re_kpc):
    return 0.5 * mass_msun / (np.pi * (re_kpc * 1e3) ** 2)

def galaxy_ism(z, logM):
    m_h2 = 10.0 ** (mu_gas_tacconi18(z, logM) + logM)
    re = re_kpc_vdw14(z, logM)
    log_ssfr = np.array([main_sequence_ssfr(z, m) for m in np.atleast_1d(logM)])
    sfr = 10.0 ** (log_ssfr + np.atleast_1d(logM))
    sigma_h2 = surface_density(m_h2, re)
    sigma_sfr = 0.5 * sfr / (np.pi * re ** 2)
    return sigma_h2, sigma_sfr

def f_mol(sigma_h2, sigma_hi_sat):
    return sigma_h2 / (sigma_h2 + sigma_hi_sat)

DLOG_QPAH_Z = np.log10(1e-2 / 5e-4)
DFMOL_Z = (-0.3, -0.8)
SQ_BAND = tuple(DLOG_QPAH_Z / d for d in DFMOL_Z)
SIGMA_HI_BAND = (5.0, 10.0, 20.0)
_s_lo = galaxy_ism(0.05, np.array([10.7]))[1][0]
_s_hi = galaxy_ism(4.0, np.array([10.7]))[1][0]
DLOG_SSFR_Z = np.log10(_s_hi / _s_lo)
SQS = -DLOG_QPAH_Z / DLOG_SSFR_Z
SQS_BAND = (0.7 * SQS, 1.4 * SQS)
G0_SYS = 0.10

rows = []
for z in ZS:
    sigma_h2, sigma_sfr = galaxy_ism(z, LOGM_GRID)
    dlog_ssfr_dlogm = np.polyfit(LOGM_GRID, np.log10(sigma_sfr), 1)[0]
    fm = {s: f_mol(sigma_h2, s) for s in SIGMA_HI_BAND}
    dfm_dlogm = {s: np.polyfit(LOGM_GRID, fm[s], 1)[0] for s in SIGMA_HI_BAND}
    slopes_A_fmol = [sq * dfm for sq in SQ_BAND for dfm in dfm_dlogm.values()]
    slopes_A_sfr = [sqs * dlog_ssfr_dlogm for sqs in SQS_BAND]
    slopes_A = slopes_A_fmol + slopes_A_sfr
    rows.append({"z": z, "A_lo": min(slopes_A) - G0_SYS, "A_hi": max(slopes_A) + G0_SYS})
branchA = pd.DataFrame(rows)

GAMMA_MZR = (0.15, 0.30)
S_PZR = (0.0, 1.5)
B_LO = GAMMA_MZR[0] * S_PZR[0] - G0_SYS
B_HI = GAMMA_MZR[1] * S_PZR[1] + G0_SYS
A_LO, A_HI = float(branchA["A_lo"].min()), float(branchA["A_hi"].max())
print(f"Branch A (density/shattering): [{A_LO:+.2f}, {A_HI:+.2f}] dex/dex")
print(f"Branch B (enrichment/PZR)    : [{B_LO:+.2f}, {B_HI:+.2f}] dex/dex")

PIVOT = 10.75
mgrid = np.linspace(bin_ctrs.min() - 0.1, bin_ctrs.max() + 0.1, 60)
ratio_piv = 10.0 ** (SLOPE_POOL * PIVOT + INTERCEPT_POOL)

fig, ax = plt.subplots(figsize=(7.5, 5.5))

def band(lo, hi, color, label):
    lo_line = ratio_piv * 10.0 ** (lo * (mgrid - PIVOT)) * 100
    hi_line = ratio_piv * 10.0 ** (hi * (mgrid - PIVOT)) * 100
    ax.fill_between(mgrid, lo_line, hi_line, color=color, alpha=0.18, lw=0)
    ax.plot(mgrid, lo_line, color=color, lw=1.0, alpha=0.55)
    ax.plot(mgrid, hi_line, color=color, lw=1.0, alpha=0.55, label=label)

band(A_LO, A_HI, "C0", f"branch A: density/shattering [{A_LO:+.2f}, {A_HI:+.2f}]")
band(B_LO, B_HI, "C1", f"branch B: enrichment/PZR [{B_LO:+.2f}, {B_HI:+.2f}]")

ax.errorbar(bin_ctrs, lir_pool["ratio"] * 100, fmt="o", ms=9, capsize=4, color="C3", zorder=5,
            label=f"pooled K-fold (slope {SLOPE_FOLD_MEAN:+.3f}+/-{SLOPE_FOLD_ERR:.3f})")
ax.plot(mgrid, 10 ** (SLOPE_POOL * mgrid + INTERCEPT_POOL) * 100, "-", color="C3", lw=1.8, alpha=0.9)
ax.plot(bin_ctrs, lir_combined["ratio"] * 100, "^", ms=10, color="C2", zorder=5,
        label=f"combined stack (slope {SLOPE_COMBINED:+.3f})")
ax.plot(mgrid, 10 ** (SLOPE_COMBINED * mgrid + INTERCEPT_COMBINED) * 100, "--", color="C2", lw=1.8, alpha=0.9)

ax.set_xlabel(r"$\log\, M_*/M_\odot$")
ax.set_ylabel(r"$L_{\rm PAH}/L_{\rm IR}$  [%]")
ax.set_yscale("log")
ax.set_title("PAH-to-IR ratio vs stellar mass at cosmic noon:\n"
             "pooled K-fold vs combined stack, against the shattering-model channels")
ax.legend(fontsize=8.5, loc="lower right")
ax.grid(alpha=0.15, which="both")
plt.tight_layout()
fig.savefig("pah_money_narayanan_confrontation.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""## 3 · Money plot 2b -- the z-resolved crossing pattern

Per (mass bin x z slice) the feature amplitude is refit (envelope-aware) and
converted to total L_PAH with a **per-mass-bin template** (a global template
spuriously flattens the slices -- see the appendix history). Three
estimators: pooled-template (one shared template reused for every fold),
self-consistent per-fold (branch-9 stress test -- each fold uses its own
template), and the combined stack (its own single template, no fold
structure to check internally, but 3x the N of any one fold)."""
)

code(
    r'''Z_SLICES = [(0.5, 1.4, "z~1"), (1.4, 2.4, "z~2"), (2.4, 3.5, "z~3")]
wins = [(zlo, zhi) for zlo, zhi, _ in Z_SLICES]
zmids = np.array([0.5 * (zlo + zhi) for zlo, zhi, _ in Z_SLICES])


def per_bin_template(dff):
    """(alpha_wien, [r_bin0, r_bin1, ...]) from single-bin envelope-aware fits."""
    res_env = _emodel.fit_with_alpha(
        dff, evolving=True, evolve_amp=False, evolve_ratios=False,
        baseline_cols=_acols, alpha_prior=(2.0, 0.3), alpha_bounds=(1.0, 5.0),
        feature_envelope="baseline")
    aw = float(res_env["alpha_wien"])
    r_bins = []
    for i in range(len(MASS_BINS)):
        sub = dff[dff["prop_bin_id"] == i].copy()
        sub["prop_bin_id"] = 0
        ri = _kmodel.fit_evolving(sub, evolve_amp=False, evolve_ratios=False,
                                  baseline_cols={"MIPS_24": "f24_cold"},
                                  feature_envelope="baseline")
        r_bins.append(np.asarray(ri["r"]))
    return aw, r_bins


def zslice_ratios(dff, r_by_bin, aw, z_windows):
    """L_PAH/L_IR per (mass bin, z window): envelope-aware amplitude refit per
    window with (per-bin r, alpha_wien) fixed."""
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
                L_PAH = a_s * env[j] * lshape_at_z(_gb, row["z_mid"], r_ratios, FEATURE_GROUPS)
                ratios.append(L_PAH / L_IR)
                wts.append(float(row["n_sources"]) if np.isfinite(row["n_sources"]) else 1.0)
            if ratios:
                wts = np.array(wts) / np.sum(wts)
                out[i, k] = float(np.sum(wts * np.array(ratios)))
    return out


def _mass_slope(vals):
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(bin_ctrs[ok], np.log10(v[ok]), 1)[0])


# Pooled-template: one shared template (from df_pool_sm) applied to every fold.
AW_POOL, R_BINS = per_bin_template(df_pool_sm)
zr_pool = zslice_ratios(df_pool_sm, R_BINS, AW_POOL, wins)
zr_folds = np.stack([zslice_ratios(dff, R_BINS, AW_POOL, wins) for dff in fold_dfs_sm])
slice_slopes = np.array([_mass_slope(zr_pool[:, k]) for k in range(len(Z_SLICES))])
fold_slope_mat = np.array([[_mass_slope(zr_folds[f][:, k]) for k in range(len(Z_SLICES))]
                           for f in range(len(fold_dfs_sm))])
slice_serrs = np.nanstd(fold_slope_mat, axis=0, ddof=1) / np.sqrt(len(fold_dfs_sm))

# Self-consistent per-fold: each fold gets its OWN template (branch-9 stress test).
sc_templates = [per_bin_template(dff) for dff in fold_dfs_sm]
zr_folds_sc = np.stack([zslice_ratios(dff, r_bins, aw, wins)
                        for dff, (aw, r_bins) in zip(fold_dfs_sm, sc_templates)])
slice_slopes_sc = np.array([[_mass_slope(zr_folds_sc[f][:, k]) for k in range(len(Z_SLICES))]
                            for f in range(len(fold_dfs_sm))])
sc_mean = np.nanmean(slice_slopes_sc, axis=0)
sc_serr = np.nanstd(slice_slopes_sc, axis=0, ddof=1) / np.sqrt(len(fold_dfs_sm))

# Combined stack: its own single template.
AW_COMBINED, R_BINS_COMBINED = per_bin_template(df_combined_sm)
zr_combined = zslice_ratios(df_combined_sm, R_BINS_COMBINED, AW_COMBINED, wins)
slice_slopes_combined = np.array([_mass_slope(zr_combined[:, k]) for k in range(len(Z_SLICES))])

print(f"{'slice':<8}{'pooled-template':>18}{'self-consistent':>18}{'combined stack':>18}")
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    print(f"{lab:<8}{slice_slopes[k]:+10.3f}+/-{slice_serrs[k]:<7.3f}"
          f"{sc_mean[k]:+10.3f}+/-{sc_serr[k]:<7.3f}{slice_slopes_combined[k]:+16.3f}")

print(f"\ncombined stack, L_PAH/L_IR [%] per (mass bin, slice):")
print(f"{'mass bin':<28}" + "".join(f"{lab:>12}" for *_, lab in Z_SLICES))
for i in range(len(MASS_BINS)):
    cells = [f"{100*zr_combined[i,k]:6.2f}%" if np.isfinite(zr_combined[i, k]) else "    --   "
             for k in range(len(Z_SLICES))]
    print(f"{labels[i]:<28}" + "".join(f"{c:>12}" for c in cells))
'''
)

code(
    r'''bandA_slice = []
for zlo, zhi, _ in Z_SLICES:
    rows_a = branchA[(branchA["z"] >= zlo) & (branchA["z"] <= zhi)]
    if len(rows_a) == 0:
        rows_a = branchA.iloc[[int(np.argmin(np.abs(branchA["z"] - 0.5 * (zlo + zhi))))]]
    bandA_slice.append((float(rows_a["A_lo"].min()), float(rows_a["A_hi"].max())))

fig, ax = plt.subplots(figsize=(7.5, 5.2))
for k, (zlo, zhi, _) in enumerate(Z_SLICES):
    alo, ahi = bandA_slice[k]
    ax.fill_between([zlo, zhi], [alo, alo], [ahi, ahi], color="C0", alpha=0.20,
                    lw=0, label="branch A: density/shattering" if k == 0 else None)
ax.axhspan(B_LO, B_HI, color="C1", alpha=0.13, lw=0, label="branch B: enrichment/PZR")
ax.errorbar(zmids - 0.05, slice_slopes, yerr=slice_serrs, fmt="s", ms=8, capsize=4,
            color="0.4", zorder=4, label="pooled template, fold errors")
ax.errorbar(zmids, sc_mean, yerr=sc_serr, fmt="o", ms=9, capsize=4,
            color="C3", zorder=5, label="self-consistent per-fold templates")
ax.plot(zmids + 0.05, slice_slopes_combined, "^", ms=10, color="C2", zorder=6,
        label=f"combined stack ({len(WRAPPERS_COMBINED)}/3 offsets)")
ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
ax.set_xlabel("redshift")
ax.set_ylabel(r"$d\,\log(L_{\rm PAH}/L_{\rm IR})\, /\, d\,\log M_*$  [dex/dex]")
ax.set_title("Mass slope of the PAH-to-IR ratio vs redshift:\n"
             "three independent estimators, does the crossing pattern survive?")
ax.grid(alpha=0.15)
ax.legend(fontsize=8.5, loc="lower left")
plt.tight_layout()
fig.savefig("pah_money_slice_slopes_vs_branches.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""### 3c · L_PAH/L_IR vs stellar mass, one line per redshift slice

The original §3c money plot (mass on the x-axis, three lines coloured by z
slice) -- the direct-data view the slope-vs-z figure above is a summary of.
K-fold pooled (fold-scatter error bars) and combined stack side by side."""
)

code(
    r'''zr_err = np.nanstd(zr_folds, axis=0, ddof=1) / np.sqrt(len(fold_dfs_sm))
zcols = plt.cm.Blues(np.linspace(0.45, 0.95, len(Z_SLICES)))

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    x = bin_ctrs + (k - 1) * 0.02
    okk = np.isfinite(zr_pool[:, k])
    axes[0].errorbar(x[okk], 100 * zr_pool[okk, k], yerr=100 * zr_err[okk, k],
                fmt="o-", ms=7, capsize=4, color=zcols[k],
                label=f"{lab}  (z = {zlo}-{zhi})")
    okc = np.isfinite(zr_combined[:, k])
    axes[1].plot(x[okc], 100 * zr_combined[okc, k], "o-", ms=7, color=zcols[k],
                 label=f"{lab}  (z = {zlo}-{zhi})")
for ax, tag in zip(axes, ("K-fold pooled (fold-scatter errors)", "combined stack")):
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log\, M_*/M_\odot$")
    ax.set_title(tag, fontsize=10)
    ax.grid(alpha=0.15, which="both")
axes[0].set_ylabel(r"$L_{\rm PAH}/L_{\rm IR}$  [%]")
axes[0].legend(fontsize=9)
fig.suptitle("PAH-to-IR ratio vs stellar mass, split by redshift slice", fontsize=12)
plt.tight_layout()
fig.savefig("pah_money_lpah_lir_vs_mass_zslices.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""### 3d · The crossing pattern, redrawn -- L_PAH/L_IR vs redshift, one line per mass bin

Same data, axes swapped: each mass bin's own line across the three z slices,
so the low-mass and high-mass lines visibly cross rather than just their
fitted slopes changing sign."""
)

code(
    r'''fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for ax, zr, tag in ((axes[0], zr_pool, "K-fold pooled"), (axes[1], zr_combined, "combined")):
    for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
        okk = np.isfinite(zr[i, :])
        ax.plot(zmids[okk], 100 * zr[i, okk], "o-", ms=7, color=col, label=lbl)
    ax.set_xlabel("redshift")
    ax.set_yscale("log")
    ax.set_title(tag, fontsize=10)
    ax.grid(alpha=0.15, which="both")
axes[0].set_ylabel(r"$L_{\rm PAH}/L_{\rm IR}$  [%]")
axes[0].legend(fontsize=8, loc="upper left")
fig.suptitle("Do the mass-bin lines actually cross?", fontsize=12)
plt.tight_layout()
fig.savefig("pah_money_lpah_lir_vs_z_by_mass.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""### 3e · Are the mass-bin edges distorting the slope fit?

`MASS_BINS` is deliberately uneven in log-mass (widths 0.7/0.2/0.2/1.0 dex) --
SNR-equalized, not evenly spaced, and the N-per-bin table above confirms it
worked (71-72 sources in every bin, both stacks). The mass-slope fits above
use the *geometric* bin center as each point's x-value; for the two wide
outer bins, the N-weighted mean mass of the actual sources can sit
meaningfully off-center if the mass function is steep within the bin. Check
by refitting with the true weighted mean mass instead of the naive center."""
)

code(
    r'''def weighted_mean_mass(dff):
    out = []
    for i in range(len(MASS_BINS)):
        sub = dff[dff["prop_bin_id"] == i]
        w = sub["n_sources"].fillna(1).to_numpy()
        out.append(float(np.average(sub["log_M_star"].to_numpy(), weights=w)))
    return np.array(out)

mean_mass_pool = weighted_mean_mass(df_pool_sm)
mean_mass_combined = weighted_mean_mass(df_combined_sm)
print(f"{'mass bin':<28}{'naive center':>14}{'N-wtd mean (pooled)':>22}{'N-wtd mean (combined)':>24}")
for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
    print(f"{lbl:<28}{bin_ctrs[i]:>14.3f}{mean_mass_pool[i]:>22.3f}{mean_mass_combined[i]:>24.3f}")


def _mass_slope_x(vals, x):
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], np.log10(v[ok]), 1)[0])


print(f"\n{'slice':<8}{'naive-center slope':>20}{'wtd-mean-mass slope':>22}   (pooled)")
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    s_naive = _mass_slope(zr_pool[:, k])
    s_wtd = _mass_slope_x(zr_pool[:, k], mean_mass_pool)
    print(f"{lab:<8}{s_naive:>+20.3f}{s_wtd:>+22.3f}")
s_naive_allz = _mass_slope(lir_pool["ratio"].to_numpy())
s_wtd_allz = _mass_slope_x(lir_pool["ratio"].to_numpy(), mean_mass_pool)
print(f"{'all-z':<8}{s_naive_allz:>+20.3f}{s_wtd_allz:>+22.3f}")

print(f"\n{'slice':<8}{'naive-center slope':>20}{'wtd-mean-mass slope':>22}   (combined)")
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    s_naive = _mass_slope(zr_combined[:, k])
    s_wtd = _mass_slope_x(zr_combined[:, k], mean_mass_combined)
    print(f"{lab:<8}{s_naive:>+20.3f}{s_wtd:>+22.3f}")
s_naive_allz_c = _mass_slope(lir_combined["ratio"].to_numpy())
s_wtd_allz_c = _mass_slope_x(lir_combined["ratio"].to_numpy(), mean_mass_combined)
print(f"{'all-z':<8}{s_naive_allz_c:>+20.3f}{s_wtd_allz_c:>+22.3f}")
'''
)

md(
    r"""### 3f · Is letting the slope vary by z-slice earning its keep, or just overfitting?

Nested model comparison on the K-fold-pooled z-sliced data (the only version
with genuine per-point errors, from fold scatter): **Model A** -- one shared
mass-slope + intercept across all 12 (mass bin x z slice) points (2 params).
**Model B** -- an independent slope + intercept per z slice (6 params, what
§3/§3d already report). B nests A (set all three slices' slopes/intercepts
equal to recover A), so chi2_A >= chi2_B always; the question is whether the
chi2 improvement from the extra 4 parameters is bigger than the AIC/BIC
penalty for using them, and whether an F-test would call it significant if
this were a real nested-regression problem."""
)

code(
    r'''def _wls_fit(x, y, w):
    sw, swx, swy = np.sum(w), np.sum(w * x), np.sum(w * y)
    swxy, swxx = np.sum(w * x * y), np.sum(w * x * x)
    denom = sw * swxx - swx ** 2
    slope = (sw * swxy - swx * swy) / denom
    intercept = (swy - slope * swx) / sw
    chi2 = float(np.sum(w * (y - (slope * x + intercept)) ** 2))
    return float(slope), float(intercept), chi2

ok = np.isfinite(zr_pool) & np.isfinite(zr_err) & (zr_pool > 0) & (zr_err > 0)
x_all, y_all, w_all = [], [], []
chi2_B, dof_B = 0.0, 0
for k in range(len(Z_SLICES)):
    okk = ok[:, k]
    xk = mean_mass_pool[okk]
    yk = np.log10(zr_pool[okk, k])
    sig_log = zr_err[okk, k] / (zr_pool[okk, k] * np.log(10))
    wk = 1.0 / sig_log ** 2
    x_all.append(xk); y_all.append(yk); w_all.append(wk)
    _, _, chi2_k = _wls_fit(xk, yk, wk)
    chi2_B += chi2_k
    dof_B += len(xk) - 2

x_all, y_all, w_all = np.concatenate(x_all), np.concatenate(y_all), np.concatenate(w_all)
_, _, chi2_A = _wls_fit(x_all, y_all, w_all)
dof_A = len(x_all) - 2
N = len(x_all)

d_chi2 = chi2_A - chi2_B
d_k = 6 - 2
d_aic = d_chi2 - 2 * d_k
d_bic = d_chi2 - d_k * np.log(N)

from scipy import stats
f_stat = (d_chi2 / d_k) / (chi2_B / dof_B)
p_value = 1.0 - stats.f.cdf(f_stat, d_k, dof_B)

print(f"Model A (1 global slope, 2 params):  chi2 = {chi2_A:.2f}, dof = {dof_A}")
print(f"Model B (3 per-slice slopes, 6 params): chi2 = {chi2_B:.2f}, dof = {dof_B}")
print(f"\nDelta chi2 (A - B) = {d_chi2:.2f} for {d_k} extra parameters")
print(f"Delta AIC = {d_aic:+.2f}  ({'favors B (crossing)' if d_aic > 0 else 'favors A (flat)'})")
print(f"Delta BIC = {d_bic:+.2f}  ({'favors B (crossing)' if d_bic > 0 else 'favors A (flat)'})")
print(f"F({d_k},{dof_B}) = {f_stat:.2f}, p = {p_value:.4f} "
      f"({'reject A in favor of B' if p_value < 0.05 else 'cannot reject A'} at 0.05)")
'''
)

md(
    r"""## Appendix A -- bin0 template SNR: did the widened bins fix it?

§3f of the 2026-07-07 notebook found bin0's reference (6.2 µm) amplitude was
statistically consistent with zero (SNR=-3.6) in the *narrow*-bin
configuration -- a near-zero-denominator instability that explained the
wild fold-to-fold swings in the band ratio and the z~1 slice sign flip. Same
diagnostic, current (widened) bins, K-fold pooled vs combined."""
)

code(
    r'''param_labels = ["C (cont.)"] + [f"A({lbl})" for lbl in group_labels]

def template_snr(dff):
    out = []
    for i in range(len(MASS_BINS)):
        sub = dff[dff["prop_bin_id"] == i].copy()
        sub["prop_bin_id"] = 0
        prep_i = _kmodel._prepare(sub, None, None, None, None, None,
                                  baseline_cols={"MIPS_24": "f24_cold"})
        data_i, valid_i, _ = _kmodel._evolving_data(
            prep_i, {"MIPS_24": "f24_cold"}, "speagle2014", feature_envelope="baseline")
        if not valid_i:
            out.append(np.full(len(param_labels), np.nan))
            continue
        d = data_i[0]
        D = np.column_stack([d["f_cold_norm"]] + [d["K"][:, g] for g in range(len(FEATURE_GROUPS))])
        w = d["w"]
        H = D.T @ (w[:, None] * D)
        cov = np.linalg.pinv(H)
        theta = np.linalg.solve(H, D.T @ (w * d["f_obs"]))
        sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
        out.append(theta / np.where(sd > 0, sd, np.nan))
    return out

snr_pool = template_snr(df_pool_sm)
snr_combined = template_snr(df_combined_sm)
for i, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
    print(f"\n{lbl}")
    print("  SNR (K-fold pooled):", np.round(snr_pool[i], 2))
    print("  SNR (combined)     :", np.round(snr_combined[i], 2))
'''
)

md(
    r"""## Appendix B -- decomposition overlay: does the z>2 systematic shrink?

Pooled envelope-aware evolving-MCMC fit, K-fold pooled vs combined,
overlaid per mass bin. The 2026-07-07 discussion flagged low-mass data
systematically below the model and high-mass data systematically above it
at z>2 -- a plausible baseline-training-set artifact (Tier A/B counts thin
fastest at the mass extremes at high z). More N per (z, mass) cell should
shrink this if that diagnosis is right."""
)

code(
    r'''GROUPS_DEC = [[1, 2], [0], [4]]   # reference group (7.7+8.6) first
model_dec = PAHSpectrumModel(feature_groups=GROUPS_DEC, bands=("MIPS_24", "MIPS_70"),
                             sigma_z0=SIGMA_Z0, f_cat=0.03)

def to_mjy(dff):
    out = dff.copy()
    for c in ["MIPS_24", "MIPS_24_err", "MIPS_70", "MIPS_70_err", "f24_cold", "f70_cold"]:
        out[c] = 1e3 * out[c]
    return out

df_mjy = to_mjy(df_pool_sm)
evolving = model_dec.fit_evolving_mcmc(df_mjy, feature_envelope="baseline",
                                       eta_prior_sigma=1.0,
                                       n_walkers=32, n_steps=800, n_burn=300, seed=2)
dec = evolving_flux_decomposition(evolving, n_draws=100, seed=3)
print(f"K-fold pooled: chi2_red={evolving['chi2_red']:.2f}  eta_A={evolving['eta_amp']:+.3f}+/-{evolving['eta_amp_err']:.3f}")

df_mjy_combined = to_mjy(df_combined_sm)
evolving_combined = model_dec.fit_evolving_mcmc(
    df_mjy_combined, feature_envelope="baseline", eta_prior_sigma=1.0,
    n_walkers=32, n_steps=800, n_burn=300, seed=2)
dec_combined = evolving_flux_decomposition(evolving_combined, n_draws=100, seed=3)
print(f"combined:      chi2_red={evolving_combined['chi2_red']:.2f}  eta_A={evolving_combined['eta_amp']:+.3f}+/-{evolving_combined['eta_amp_err']:.3f}")
'''
)

code(
    r'''fig, axes = plt.subplots(2, 2, figsize=(14.5, 9), sharex=True)
for k, (m_lo, m_hi, col, lbl) in enumerate(MASS_BINS):
    ax = axes[k // 2, k % 2]
    for tag, dsrc, dfm, mcolor, mstyle in (
        ("K-fold pooled", dec, df_mjy, "k", "-"),
        ("combined", dec_combined, df_mjy_combined, "C3", "--"),
    ):
        dm = dsrc[(dsrc["prop_bin_id"] == k) & (dsrc["band"] == "MIPS_24")].sort_values("z_mid")
        if len(dm):
            ax.plot(dm["z_mid"], dm["total"], mstyle, color=mcolor, lw=1.3,
                    label=f"{tag} model", zorder=3)
        sub = dfm[dfm["prop_bin_id"] == k]
        ax.errorbar(sub["z_mid"], sub["MIPS_24"], yerr=sub["MIPS_24_err"], fmt="o",
                    ms=4, color=mcolor, alpha=0.6, elinewidth=0.8, capsize=2,
                    label=f"{tag} data", zorder=4)
    ax.set_yscale("log")
    ax.set_title(f"MIPS 24 um -- {lbl}", fontsize=10)
    ax.grid(alpha=0.15)
    if k % 2 == 0:
        ax.set_ylabel("stacked flux [mJy]")
    if k // 2 == 1:
        ax.set_xlabel("redshift")
    if k == 0:
        ax.legend(fontsize=8, ncol=2, loc="upper right")
fig.suptitle("K-fold pooled vs combined stack: does more N per bin\n"
             "shrink the z>2 systematic (low mass below, high mass above the model)?",
             fontsize=12)
plt.tight_layout()
fig.savefig("pah_money_combined_vs_kfold_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""## 4 · Verdict -- do the headline results hold?

**Money plot 1 (band ratio vs mass): qualitative trend holds, bin0 still not trustworthy in the K-fold-pooled fit.**
Monotonic decline survives in both pooled K-fold and combined (6.4/0.8/0.7/0.4 percent-scale
vs 1.6/0.5/0.4/0.15 -- roughly a uniform ~2-4x offset, not a sign or ordering change), and the
non-bin0 adjacent-bin separations stay >5 sigma. But bin0's reference amplitude A(6.2) is
**still SNR=-4.17 (consistent with zero, wrong sign) in the K-fold-pooled fit even after the
widened bins** -- only the combined stack actually fixes it (SNR=+8.19). Quote the
**combined-stack bin0 ratio (1.6), not the K-fold-pooled one (6.4)**, until this is resolved.

**Money plot 2a (all-z L_PAH/L_IR slope): the headline number collapsed to zero, and that's not
a problem -- it's a prediction of money plot 2b.** Previously documented +0.234 +/- 0.077.
Re-derived on the widened bins: K-fold fold-ensemble -0.004 +/- 0.038, pooled -0.006, combined
+0.001 -- all three agree with each other and with flat. **This is not evidence against the
crossing pattern.** A slope that runs +0.3 (z~1) through ~0 (z~2) to -0.6/-0.7 (z~3) integrates
to ~0 when you force one global amplitude across 0.5<z<3.5 -- the flat number is the expected
consequence of the crossing existing, not a competing measurement. The all-z framing should be
retired as the headline in favor of the z-resolved one.

**Money plot 2b (z-resolved crossing pattern): holds, strengthens under the weighted-mass
correction, and beats the flat model in a formal nested comparison.** All three estimators agree
in sign at each slice: z~1 positive (+0.30/+0.35/+0.33), z~2 near-zero (+0.07/+0.07/+0.01), z~3
solidly negative (-0.69/-0.72/-0.57). Correcting the wide bins' geometric center to their true
N-weighted mean mass (bin3: 11.5 -> 11.19) *widens* the swing further (z~1 -> +0.36/+0.39, z~3 ->
-0.87/-0.70) -- the naive center was diluting the pattern, not exaggerating it. A nested model
comparison on the pooled data (one global slope, 2 params, vs. an independent slope per z slice,
6 params) gives chi2 368.8 (dof 10) vs 40.9 (dof 6): Delta-AIC=+320, Delta-BIC=+318, F(4,6)=12.0,
p=0.005 -- the extra flexibility earns its keep by roughly a factor of 40 in chi2 against an
~8-10 parameter penalty.

**Caveat on that p-value, stated plainly so it isn't overclaimed:** the errors feeding this test
are K-fold fold-scatter, which is a jackknife-style estimator that undercounts field-level cosmic
variance and shared-map systematics (one field, shared maps/photo-z/PSF across folds) --
correlated, not independent, and likely undersized. The F-statistic is a ratio and is protected
against a *uniform* rescaling of all the errors (it would take a uniform ~6.4x underestimate to
flip the AIC/BIC verdict, well beyond typical single-field cosmic-variance inflation), but it is
NOT protected against a *non-uniform* correction (if cosmic variance affects some z-slices more
than others). Net: treat "the crossing pattern fits far better than the flat model" as a robust
qualitative conclusion; do not quote p=0.005 externally as a calibrated significance, since
neither the independence nor the absolute error size behind it is verified.

**Side finding, not yet reconciled: eta_A (sSFR-amplitude evolution) differs by ~2x between
pooled K-fold (+0.421 +/- 0.069) and combined (+0.860 +/- 0.041).** The combined value matches
the previously-documented branch-7 headline (+0.844 +/- 0.026); the K-fold-pooled value does
not. Chi2_red is 2.79 (pooled) vs 3.45 (combined) -- combined fits somewhat worse, so this isn't
simply "combined is better." Open question.

**Net read:** the crossing pattern is the most defensible result in this notebook -- three
independent central-value estimators agreeing in sign, a mass-center correction that makes it
*stronger*, and a formal model comparison that rejects the single-slope alternative by a wide
margin (caveated on error correlation as above). The band-ratio mass trend survives but needs
the combined-stack correction for its steepest point. The all-z Narayanan-confrontation slope, as
previously headlined, should be retired in favor of the z-resolved framing -- its collapse to
zero is now understood as a consequence of the crossing, not a contradiction of it.
"""
)


nb["cells"] = cells
out = "notebooks/2026-07-10-pah-money-plots.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out} ({len(cells)} cells)")
