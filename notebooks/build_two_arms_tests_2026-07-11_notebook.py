"""Generate 2026-07-11-two-arms-tests.ipynb.

The crossing pattern (mass slope of L_PAH/L_IR running +0.3 at z~1 -> 0 at
z~2 -> -0.7 at z~3) structurally requires TWO opposing terms whose balance
shifts with z. Two candidate identifications (docs/pah-interpretation-
candidates.md):

  C1 -- metallicity-regulated supply (positive arm) vs radiation-field
        destruction in compact intense star formation (negative arm)
  C2 -- single shattering mechanism (Narayanan+26): diffuse-gas production
        (positive arm) vs dense-ISM suppression (negative arm)

This notebook works through the discriminating tests in order:

  D1  plateau test          -- can the q_PAH(Z) step carry the z~1 slope?
  D2  gas-tracer floor      -- can "PAH traces molecular gas" alone do it?
  D3  windowed band ratios  -- does the population mix change WITH the
                               amplitude inversion (destruction fingerprint)
                               or stay put (production fingerprint)?
  D4  mediator separation   -- at fixed mass, does the deficit follow the
                               radiation proxy (T_dust) or the density proxy
                               (sigma_SFR)?
  D5  metallicity-track violation at z~3 (falls out of D1's machinery)
  D6  scaling-relation arms -- which (positive, negative) arm pair actually
                               fits the three slice slopes?

Data: the 2026-07-07 widened-bin stacks (K-fold pooled + combined), same as
2026-07-10-pah-money-plots.ipynb, plus the 2026-06-14 sigma_SFR cross-cut
(COSMOS25) for D4. Requires PICKLESPATH.

Run:  uv run python notebooks/build_two_arms_tests_2026-07-11_notebook.py
Then: uv run jupyter nbconvert --to notebook --execute --inplace \
          --ExecutePreprocessor.timeout=3600 \
          notebooks/2026-07-11-two-arms-tests.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(
    r"""# Two-arms tests: which physics carries each arm of the crossing pattern?

**2026-07-11.** The z-resolved crossing of the L_PAH/L_IR mass slope needs two
opposing terms; no single controlling variable (metallicity, depletion time,
Sigma_SFR) flips its own mass slope with z under standard scaling relations.
This notebook runs the discriminating tests D1-D6 from
`docs/pah-interpretation-candidates.md` in order, ending in a verdict table:
for each arm, does the evidence favor

- **C1** (supply-by-enrichment + destruction-by-radiation), or
- **C2** (production-by-shattering, modulated by ISM density)?

Measured inputs come from the same widened-bin stacks as
`2026-07-10-pah-money-plots.ipynb` (K-fold pooled with fold-scatter errors,
plus the combined non-split stack as the bias check). Mass slopes here use the
**N-weighted mean bin masses** throughout (the naive-center variant dilutes
the pattern; branch-9 finding)."""
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
    PAHSpectrumModel, group_weights, DEFAULT_FEATURES,
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
# Combined: full (non-split) catalog, 3 dither-offset runs.
RUN_DATES_COMBINED = {
    0: "20260707_204926",   # offset 0.0000
    1: "20260707_225921",   # offset 0.0375
    2: "20260708_091724",   # offset 0.0750
}
# sigma_SFR cross-cut (D4): COSMOS25, 2 mass x 3 sigma_SFR bins, dz=0.15,
# 4 dither offsets (2026-06-14/15).
RUN_DATES_SIGMA = {
    0: "20260614_203609",
    1: "20260614_220935",
    2: "20260614_230839",
    3: "20260615_000803",
}

MASS_BINS = [
    (9.90, 10.6, "C0", r"$9.9 < \log M_* < 10.6$"),
    (10.6, 10.8, "C1", r"$10.6 < \log M_* < 10.8$"),
    (10.8, 11.0, "C2", r"$10.8 < \log M_* < 11.0$"),
    (11.0, 12.0, "C3", r"$\log M_* > 11.0$"),
]
# Feature groups: 0=6.2, 1=7.7, 2=8.6, 3=11.3(blind), 4=12.7, 5=16.4, 6=17.0
FEATURE_GROUPS = [[0], [1, 2], [4]]   # 6.2 | 7.7+8.6 | 12.7  (11.3 blind)
SIGMA_Z0 = 0.01
bin_ctrs = np.array([0.5 * (lo + hi) for lo, hi, *_ in MASS_BINS])
labels = [lbl for _, _, _, lbl in MASS_BINS]

Z_SLICES = [(0.5, 1.4, "z~1"), (1.4, 2.4, "z~2"), (2.4, 3.5, "z~3")]
wins = [(zlo, zhi) for zlo, zhi, _ in Z_SLICES]
zmids = np.array([0.5 * (zlo + zhi) for zlo, zhi, _ in Z_SLICES])
'''
)

md(r"""## 0 · Load the stacks, rebuild the measured slice slopes""")

code(
    r'''def build_pah_spectrum_df(wrappers, mass_bins, split_filter=None, min_tier="B"):
    """One row per (run, mass bin, z bin): MIPS 24/70 fluxes, greybody
    Wien-side extrapolations, T_dust/amplitude/beta. Same as the money-plots
    notebook."""
    tier_rank = {"A": 0, "B": 1, "C": 2}
    min_rank = tier_rank[min_tier.upper()]
    rows = []
    _gb_row = _Greybody()
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
    """f24_cold/f70_cold from a smooth T(z,M*), logA(z,M*) trained on Tier A/B."""
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


WRAPPERS = []
for k, date in RUN_DATES.items():
    w = SimstackWrapper()
    w.load_stacking_results(os.path.join(path_json, f"cosmos20_stacking_{date}.json"))
    w.run_analysis_only(**ANALYSIS_KWARGS)
    WRAPPERS.append(w)
print("Loaded 3 disjoint-fold stacking runs:", RUN_DATES)

WRAPPERS_COMBINED = []
for k, date in RUN_DATES_COMBINED.items():
    w = SimstackWrapper()
    w.load_stacking_results(os.path.join(path_json, f"cosmos20_stacking_{date}.json"))
    w.run_analysis_only(**ANALYSIS_KWARGS)
    WRAPPERS_COMBINED.append(w)
print(f"Loaded {len(WRAPPERS_COMBINED)}/3 combined-stack offset runs:", RUN_DATES_COMBINED)

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
'''
)

code(
    r'''_kmodel = PAHSpectrumModel(feature_groups=FEATURE_GROUPS, bands=("MIPS_24",),
                           sigma_z0=SIGMA_Z0, f_cat=0.03)
_emodel = PAHSpectrumModel(feature_groups=FEATURE_GROUPS, bands=("MIPS_24", "MIPS_70"),
                           sigma_z0=SIGMA_Z0, f_cat=0.03)
_acols = {"MIPS_24": "f24_cold", "MIPS_70": "f70_cold"}


def lshape_at_z(gb, z, r_ratios, feature_groups, features=None):
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
    """L_PAH/L_IR per (mass bin, z window), envelope-aware amplitude refit."""
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


def weighted_mean_mass(dff):
    out = []
    for i in range(len(MASS_BINS)):
        sub = dff[dff["prop_bin_id"] == i]
        w = sub["n_sources"].fillna(1).to_numpy()
        out.append(float(np.average(sub["log_M_star"].to_numpy(), weights=w)))
    return np.array(out)


def _mass_slope_x(vals, x):
    v = np.asarray(vals, dtype=float)
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], np.log10(v[ok]), 1)[0])


mean_mass_pool = weighted_mean_mass(df_pool_sm)
mean_mass_combined = weighted_mean_mass(df_combined_sm)

AW_POOL, R_BINS = per_bin_template(df_pool_sm)
zr_pool = zslice_ratios(df_pool_sm, R_BINS, AW_POOL, wins)
zr_folds = np.stack([zslice_ratios(dff, R_BINS, AW_POOL, wins) for dff in fold_dfs_sm])
slice_slopes = np.array([_mass_slope_x(zr_pool[:, k], mean_mass_pool)
                         for k in range(len(Z_SLICES))])
fold_slope_mat = np.array([[_mass_slope_x(zr_folds[f][:, k], mean_mass_pool)
                            for k in range(len(Z_SLICES))]
                           for f in range(len(fold_dfs_sm))])
slice_serrs = np.nanstd(fold_slope_mat, axis=0, ddof=1) / np.sqrt(len(fold_dfs_sm))

AW_COMBINED, R_BINS_COMBINED = per_bin_template(df_combined_sm)
zr_combined = zslice_ratios(df_combined_sm, R_BINS_COMBINED, AW_COMBINED, wins)
slice_slopes_combined = np.array([_mass_slope_x(zr_combined[:, k], mean_mass_combined)
                                  for k in range(len(Z_SLICES))])

print("Measured mass slopes of L_PAH/L_IR (N-weighted mean-mass x-values):")
print(f"{'slice':<8}{'pooled +/- fold err':>22}{'combined':>12}")
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    print(f"{lab:<8}{slice_slopes[k]:+14.3f} +/- {slice_serrs[k]:<5.3f}"
          f"{slice_slopes_combined[k]:>+12.3f}")
MEASURED = slice_slopes.copy()
MEASURED_ERR = slice_serrs.copy()
'''
)

md(
    r"""## 1 · D1 + D5 -- the metallicity arm, quantified

The C1 supply arm is a **step function**: JWST (Shivaei+24 SMILES) and local
spectroscopy (Whitcomb+24) agree q_PAH is ~flat above a metallicity threshold
(0.5-0.67 Z_sun) and collapses below it. Chain it through the mass-metallicity
relation (Sanders+21 anchors, bracketed slope) and it makes a definite
prediction for the mass slope of L_PAH/L_IR at each z slice -- with **zero
reference to our data**.

- **D1 (z~1)**: if our bins all sit on the plateau at z~1, the predicted slope
  is ~0 -- the step function cannot carry a measured +0.3.
- **D5 (z~3)**: the lowest bin falls to/below threshold by z~3, so the
  prediction turns *positive* there -- if we measure strongly negative, the
  metallicity track is violated and Z is not the controlling variable at z~3.

Bracketed systematics: MZR slope gamma in [0.15, 0.30], threshold in
{0.5, 2/3} Z_sun, below-threshold decline slope s_dec in [1.5, 3.5] dex/dex
(Shivaei+24: 3.4% -> <1% over ~0.22 dex of O/H gives ~2.4)."""
)

code(
    r'''OH_SOLAR = 8.69
# 12+log(O/H) at log M*=10: local saturation ~8.75 (Curti+20-like),
# Sanders+21 z=2.3 -> 8.51, z=3.3 -> 8.41; linear interp between anchors.
_MZR_Z = np.array([0.0, 2.3, 3.3])
_MZR_OH10 = np.array([8.75, 8.51, 8.41])

def oh_mzr(logM, z, gamma):
    oh10 = np.interp(z, _MZR_Z, _MZR_OH10)
    return oh10 + gamma * (np.asarray(logM) - 10.0)

def log_qpah(oh, oh_thr, s_dec, log_q0=np.log10(0.034)):
    oh = np.asarray(oh, dtype=float)
    return np.where(oh >= oh_thr, log_q0, log_q0 - s_dec * (oh_thr - oh))

GAMMAS = (0.15, 0.30)
OH_THRS = (OH_SOLAR + np.log10(0.5), OH_SOLAR + np.log10(2.0 / 3.0))   # 8.39, 8.51
S_DECS = (1.5, 3.5)

print("12+log(O/H) at the N-weighted bin masses (gamma bracket midpoint 0.225):")
print(f"{'slice':<8}" + "".join(f"{m:>9.2f}" for m in mean_mass_pool)
      + f"{'thr 0.5Zsun':>14}{'thr 2/3Zsun':>13}")
for zlo, zhi, lab in Z_SLICES:
    zc = 0.5 * (zlo + zhi)
    ohs = oh_mzr(mean_mass_pool, zc, 0.225)
    print(f"{lab:<8}" + "".join(f"{o:>9.2f}" for o in ohs)
          + f"{OH_THRS[0]:>14.2f}{OH_THRS[1]:>13.2f}")

pred_band, pred_mid = [], []
for zlo, zhi, lab in Z_SLICES:
    zc = 0.5 * (zlo + zhi)
    slopes = []
    for gamma in GAMMAS:
        for oh_thr in OH_THRS:
            for s_dec in S_DECS:
                lq = log_qpah(oh_mzr(mean_mass_pool, zc, gamma), oh_thr, s_dec)
                slopes.append(float(np.polyfit(mean_mass_pool, lq, 1)[0]))
    pred_band.append((min(slopes), max(slopes)))
    lq_mid = log_qpah(oh_mzr(mean_mass_pool, zc, 0.225), np.mean(OH_THRS), 2.4)
    pred_mid.append(float(np.polyfit(mean_mass_pool, lq_mid, 1)[0]))
pred_band = np.array(pred_band)
pred_mid = np.array(pred_mid)

print(f"\n{'slice':<8}{'metallicity-step prediction':>30}{'measured (pooled)':>22}{'combined':>12}")
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    print(f"{lab:<8}   [{pred_band[k,0]:+.3f}, {pred_band[k,1]:+.3f}] (mid {pred_mid[k]:+.3f})"
          f"{MEASURED[k]:>+15.3f} +/- {MEASURED_ERR[k]:.3f}{slice_slopes_combined[k]:>+12.3f}")

# D1 verdict: how many sigma above the step-function ceiling is the z~1 slope?
d1_excess = (MEASURED[0] - pred_band[0, 1]) / MEASURED_ERR[0]
print(f"\nD1: z~1 measured slope sits {d1_excess:+.1f} sigma above the metallicity-step CEILING")
print("    -> the q_PAH(Z) step cannot carry the z~1 positive arm" if d1_excess > 2
      else "    -> consistent with the metallicity step; C1 supply arm suffices at z~1")
print("    NOTE (manual check, Shivaei+24 Fig): confirm their q_PAH(M*) is flat")
print("    within 9.9 < log M* < 11.2 specifically -- if their correlation is")
print("    carried by log M* < 10, this test is decisive as computed.")

# D5 verdict: sign violation at z~3.
d5_gap = (MEASURED[2] - pred_band[2, 0]) / MEASURED_ERR[2]
print(f"\nD5: z~3 measured {MEASURED[2]:+.3f} vs metallicity-track floor {pred_band[2,0]:+.3f}"
      f" -> {d5_gap:+.1f} sigma below")
print("    -> Z is not the controlling variable at z~3; the negative arm must")
print("       overwhelm an OPPOSING metallicity gradient" if d5_gap < -2 else "")
'''
)

code(
    r'''zgrid = np.linspace(0.6, 3.4, 60)
band_lo, band_hi = [], []
for zc in zgrid:
    slopes = []
    for gamma in GAMMAS:
        for oh_thr in OH_THRS:
            for s_dec in S_DECS:
                lq = log_qpah(oh_mzr(mean_mass_pool, zc, gamma), oh_thr, s_dec)
                slopes.append(float(np.polyfit(mean_mass_pool, lq, 1)[0]))
    band_lo.append(min(slopes)); band_hi.append(max(slopes))

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.fill_between(zgrid, band_lo, band_hi, color="C1", alpha=0.20, lw=0,
                label="metallicity-step arm (q_PAH(Z) x MZR, full bracket)")
ax.errorbar(zmids, MEASURED, yerr=MEASURED_ERR, fmt="o", ms=9, capsize=4,
            color="C3", zorder=5, label="measured (pooled K-fold, fold errors)")
ax.plot(zmids + 0.05, slice_slopes_combined, "^", ms=10, color="C2", zorder=5,
        label="measured (combined stack)")
ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
ax.set_xlabel("redshift")
ax.set_ylabel(r"$d\,\log(L_{\rm PAH}/L_{\rm IR})\,/\,d\,\log M_*$  [dex/dex]")
ax.set_title("D1 + D5: the metallicity step predicts the WRONG shape\n"
             "(flat at z~1 where we measure +, positive at z~3 where we measure $-$)")
ax.grid(alpha=0.15)
ax.legend(fontsize=9, loc="lower left")
plt.tight_layout()
fig.savefig("pah_twoarms_d1_metallicity_arm.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""## 2 · D2 -- the gas-tracer floor

The purest C2 reading -- "PAH light simply traces molecular gas" (PAH-CO,
arXiv:2409.05710) -- makes L_PAH/L_IR proportional to the depletion time
t_dep = M_gas/SFR. Two routes to its mass slope, bracketing the Tacconi+18
internal tension: the composite route (mu_gas scaling minus the main-sequence
sSFR scaling) and their direct global t_dep fit (+0.09, z-independent). If
both are far shallower than the measured slopes and neither flips sign, pure
gas tracing is ruled out as the whole story and the *residual* is the
shattering-efficiency / destruction term the data demand."""
)

code(
    r'''def mu_gas_tacconi18(z, logM, dms=0.0):
    return (0.12 - 3.62 * (np.log10(1.0 + z) - 0.66) ** 2
            + 0.53 * dms - 0.35 * (logM - 10.7))

LOGM_GRID = np.linspace(mean_mass_pool.min(), mean_mass_pool.max(), 25)

print(f"{'slice':<8}{'t_dep slope (composite)':>25}{'(direct fit)':>14}"
      f"{'measured':>12}{'residual demanded':>20}")
tdep_slopes = []
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    zc = 0.5 * (zlo + zhi)
    log_ssfr = np.array([main_sequence_ssfr(zc, m) for m in LOGM_GRID])
    log_tdep = mu_gas_tacconi18(zc, LOGM_GRID) - log_ssfr
    s_comp = float(np.polyfit(LOGM_GRID, log_tdep, 1)[0])
    s_direct = 0.09
    tdep_slopes.append((s_comp, s_direct))
    resid_lo = MEASURED[k] - max(s_comp, s_direct)
    resid_hi = MEASURED[k] - min(s_comp, s_direct)
    print(f"{lab:<8}{s_comp:>+25.3f}{s_direct:>+14.3f}{MEASURED[k]:>+12.3f}"
          f"   [{resid_lo:+.3f}, {resid_hi:+.3f}]")
tdep_slopes = np.array(tdep_slopes)

swing_meas = MEASURED[2] - MEASURED[0]
swing_tdep = (tdep_slopes[2] - tdep_slopes[0])
print(f"\nD2: measured z~1 -> z~3 swing = {swing_meas:+.3f} dex/dex;")
print(f"    t_dep-arm swing = [{swing_tdep.min():+.3f}, {swing_tdep.max():+.3f}]"
      f" -- {'cannot produce the crossing alone' if abs(swing_meas) > 3*max(abs(swing_tdep)) else 'check'}")
print("    Pure gas tracing is a near-z-independent, shallow term: the crossing")
print("    requires the density/shattering (or destruction) term on top of it.")
'''
)

md(
    r"""## 3 · D3 -- windowed band ratios: destruction fingerprint or production fingerprint?

**Design constraint stated honestly**: one broad band sees each rest feature
at its own redshift (12.7 um crosses MIPS 24 at z~0.9; 7.7+8.6 at z~1.6-2.6;
6.2 only at z>2.4). A per-z-slice 12.7/6.2 ratio is therefore impossible --
the all-z "intrinsic" ratio is already a cross-z composite. What IS possible:
split at z=2.1 (the 7.7 um crossing, so the reference group anchors both
sides) and fit each window's co-constrained pair:

- low window (z<2.1): r(12.7 / 7.7+8.6) -- neutral-band weight, anchored z~1.3
- high window (z>=2.1): r(6.2 / 7.7+8.6) -- ionized-band weight, anchored z~2.7

The discriminant: if the z~3 amplitude inversion is **destruction** (C1), the
same photons reprocess the survivors, so the high-window mix should trend with
mass at least as strongly as the amplitude does. If it is **suppressed
production** (C2), the mix is set by the birth population and the high-window
ratio trend should be comparatively flat while the amplitude inverts.
Cross-check: the product r(12.7/ref)_low x [1/r(6.2/ref)_high] should
reproduce the headline 12.7/6.2 ratio."""
)

code(
    r'''Z_CUT = 2.1
GROUPS_LOW = [[1, 2], [4]]    # ref 7.7+8.6 ; r[1] = 12.7/(7.7+8.6),  z < 2.1
GROUPS_HIGH = [[1, 2], [0]]   # ref 7.7+8.6 ; r[1] = 6.2/(7.7+8.6),   z >= 2.1
_model_low = PAHSpectrumModel(feature_groups=GROUPS_LOW, bands=("MIPS_24",),
                              sigma_z0=SIGMA_Z0, f_cat=0.03)
_model_high = PAHSpectrumModel(feature_groups=GROUPS_HIGH, bands=("MIPS_24",),
                               sigma_z0=SIGMA_Z0, f_cat=0.03)


def windowed_ratio(dff, model, zlo, zhi):
    """Per mass bin: (r[1], r_err[1]) from an envelope-aware static fit
    restricted to [zlo, zhi)."""
    r_out = np.full(len(MASS_BINS), np.nan)
    e_out = np.full(len(MASS_BINS), np.nan)
    for i in range(len(MASS_BINS)):
        sub = dff[(dff["prop_bin_id"] == i)
                  & (dff["z_mid"] >= zlo) & (dff["z_mid"] < zhi)].copy()
        sub["prop_bin_id"] = 0
        if len(sub) < 8:
            continue
        try:
            res = model.fit_evolving(sub, evolve_amp=False, evolve_ratios=False,
                                     baseline_cols={"MIPS_24": "f24_cold"},
                                     feature_envelope="baseline")
        except Exception as exc:
            print(f"    bin {i} [{zlo},{zhi}): fit failed ({exc})")
            continue
        if res is None:
            continue
        r_out[i] = float(res["r"][1])
        e_out[i] = float(res["r_err"][1])
    return r_out, e_out


print("Combined stack (primary -- bin0 template SNR is only trustworthy here):")
rlow_c, elow_c = windowed_ratio(df_combined_sm, _model_low, 0.5, Z_CUT)
rhigh_c, ehigh_c = windowed_ratio(df_combined_sm, _model_high, Z_CUT, 3.5)
print("K-fold pooled:")
rlow_p, elow_p = windowed_ratio(df_pool_sm, _model_low, 0.5, Z_CUT)
rhigh_p, ehigh_p = windowed_ratio(df_pool_sm, _model_high, Z_CUT, 3.5)
rlow_f = np.array([windowed_ratio(dff, _model_low, 0.5, Z_CUT)[0] for dff in fold_dfs_sm])
rhigh_f = np.array([windowed_ratio(dff, _model_high, Z_CUT, 3.5)[0] for dff in fold_dfs_sm])
elow_fold = np.nanstd(rlow_f, axis=0, ddof=1) / np.sqrt(len(fold_dfs_sm))
ehigh_fold = np.nanstd(rhigh_f, axis=0, ddof=1) / np.sqrt(len(fold_dfs_sm))

print(f"\n{'mass bin':<28}{'r(12.7/ref) z<2.1':>20}{'r(6.2/ref) z>=2.1':>20}{'12.7/6.2 (product)':>20}")
for i in range(len(MASS_BINS)):
    prod = rlow_c[i] / rhigh_c[i] if (np.isfinite(rlow_c[i]) and rhigh_c[i] > 0) else np.nan
    print(f"{labels[i]:<28}{rlow_c[i]:>14.3f}+/-{elow_c[i]:<5.3f}"
          f"{rhigh_c[i]:>14.3f}+/-{ehigh_c[i]:<5.3f}{prod:>20.2f}")

def _log_slope(r, x):
    ok = np.isfinite(r) & (r > 0)
    if ok.sum() < 3:
        return np.nan
    return float(np.polyfit(x[ok], np.log10(r[ok]), 1)[0])

s_low_c = _log_slope(rlow_c, mean_mass_combined)
s_high_c = _log_slope(rhigh_c, mean_mass_combined)
s_low_p = _log_slope(rlow_p, mean_mass_pool)
s_high_p = _log_slope(rhigh_p, mean_mass_pool)
s_low_f = np.array([_log_slope(rlow_f[f], mean_mass_pool) for f in range(len(fold_dfs_sm))])
s_high_f = np.array([_log_slope(rhigh_f[f], mean_mass_pool) for f in range(len(fold_dfs_sm))])
s_low_ferr = np.nanstd(s_low_f, ddof=1) / np.sqrt(len(fold_dfs_sm))
s_high_ferr = np.nanstd(s_high_f, ddof=1) / np.sqrt(len(fold_dfs_sm))

print(f"\nMass slopes of the window ratios [dex/dex], vs the amplitude slopes:")
print(f"{'window':<26}{'ratio slope (comb)':>20}{'(pooled +/- fold)':>22}{'amplitude slope':>18}")
print(f"{'z<2.1  r(12.7/ref)':<26}{s_low_c:>+20.3f}{s_low_p:>+14.3f} +/- {s_low_ferr:<5.3f}"
      f"{MEASURED[0]:>+16.3f}")
print(f"{'z>=2.1 r(6.2/ref)':<26}{s_high_c:>+20.3f}{s_high_p:>+14.3f} +/- {s_high_ferr:<5.3f}"
      f"{MEASURED[2]:>+16.3f}")
print("\nReading: |ratio slope| >~ |amplitude slope| in the high window, with the")
print("ionized band gaining weight at high mass -> processing/destruction (C1).")
print("Ratio slopes ~flat while the amplitude inverts -> production-side (C2).")
'''
)

code(
    r'''fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
for ax, (r_c, e_c, r_p, e_f, ttl, ylab) in zip(axes, [
    (rlow_c, elow_c, rlow_p, elow_fold, f"low window  z < {Z_CUT}",
     r"$r_{12.7}/r_{7.7+8.6}$  (neutral / reference)"),
    (rhigh_c, ehigh_c, rhigh_p, ehigh_fold, rf"high window  z $\geq$ {Z_CUT}",
     r"$r_{6.2}/r_{7.7+8.6}$  (ionized / reference)"),
]):
    ax.errorbar(bin_ctrs, r_p, yerr=e_f, fmt="o", ms=8, capsize=4, color="C3",
                label="pooled K-fold (fold err)")
    ax.errorbar(bin_ctrs + 0.03, r_c, yerr=e_c, fmt="^", ms=9, capsize=3, color="C2",
                label="combined (fit err)")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\log\, M_*/M_\odot$")
    ax.set_ylabel(ylab)
    ax.set_title(ttl, fontsize=10)
    ax.grid(alpha=0.15, which="both")
axes[0].legend(fontsize=9)
fig.suptitle("D3: does the PAH population mix trend with mass in each z window?", fontsize=12)
plt.tight_layout()
fig.savefig("pah_twoarms_d3_windowed_ratios.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(
    r"""## 4 · D4 -- mediator separation: radiation proxy vs density proxy

At fixed mass, both candidates predict "more concentrated star formation ->
PAH deficit", but through different mediators: C1 through the radiation field
(proxy: T_dust, free from every greybody fit) and C2 through gas density
(proxy: sigma_SFR). The 2026-06-14 cross-cut (COSMOS25, 2 mass x 3 sigma_SFR
bins, 4 dither offsets) lets us hold mass fixed and vary sigma_SFR directly.

Per (mass, sigma_SFR) cell we fit the envelope-aware PAH-to-continuum
amplitude A_pah and take the Tier A/B median T_dust; then ask which proxy
orders A_pah at fixed mass. **Directional first pass only**: 3 sigma_SFR
points per mass row, a different catalog (COSMOS25 vs COSMOS2020), and A_pah
rather than L_PAH/L_IR."""
)

code(
    r'''D4_OK = False
try:
    WRAPPERS_SIGMA = []
    for k, date in RUN_DATES_SIGMA.items():
        fp = os.path.join(path_json, f"cosmos25_stacking_{date}.json")
        if not os.path.exists(fp):
            print(f"missing: {fp} -- skipping")
            continue
        w = SimstackWrapper()
        w.load_stacking_results(fp)
        w.run_analysis_only(**ANALYSIS_KWARGS)
        WRAPPERS_SIGMA.append(w)
    print(f"Loaded {len(WRAPPERS_SIGMA)}/4 sigma_SFR cross-cut runs")

    import re

    _SIG_RE = re.compile(r"sigma_sfr_(-?[0-9.]+)_(-?[0-9.]+)")
    _MASS_RE = re.compile(r"stellar_mass_(-?[0-9.]+)_(-?[0-9.]+)")

    def build_sigma_df(wrappers):
        """The sigma_SFR runs store only the redshift range in bin_ranges;
        the mass and sigma_SFR bin edges live in the population ID string
        (redshift_a_b__sigma_sfr_c_d__stellar_mass_e_f__split_g)."""
        rows = []
        _gb_row = _Greybody()
        for run_id, wrapper in enumerate(wrappers):
            pr = getattr(wrapper, "processed_results", None)
            if pr is None or not pr.sed_results:
                continue
            pops = wrapper.population_manager.populations
            for pop_id, sed in pr.sed_results.items():
                if not sed.greybody_fit_success:
                    continue
                if _extract_pop_type(pop_id) != "split_0":
                    continue
                pop = pops.get(pop_id)
                if pop is None:
                    continue
                z_range = pop.bin_ranges.get("redshift")
                if z_range is None:
                    continue
                m_sig = _SIG_RE.search(pop_id)
                m_mass = _MASS_RE.search(pop_id)
                if m_sig is None or m_mass is None:
                    continue
                sigma_range = (float(m_sig.group(1).rstrip(".")),
                               float(m_sig.group(2).rstrip(".")))
                mass_range = (float(m_mass.group(1).rstrip(".")),
                              float(m_mass.group(2).rstrip(".")))
                z_lo, z_hi = float(z_range[0]), float(z_range[1])
                _z = 0.5 * (z_lo + z_hi)
                f24 = f24_err = np.nan
                for j, wl in enumerate(sed.wavelengths):
                    if abs(wl - 24.0) / 24.0 < 0.15:
                        f24 = float(sed.flux_densities[j])
                        f24_err = float(sed.flux_errors[j])
                if not (np.isfinite(f24) and f24 > 0 and np.isfinite(f24_err) and f24_err > 0):
                    continue
                if (sed.amplitude is None or sed.dust_temperature_rest_frame is None
                        or sed.emissivity_index is None):
                    continue
                f24_cold = float(_gb_row.greybody_model(
                    np.array([24.0 / (1.0 + _z)]), sed.amplitude,
                    sed.dust_temperature_rest_frame, sed.emissivity_index)[0])
                rows.append({
                    "run_id": run_id, "z_lo": z_lo, "z_hi": z_hi, "z_mid": _z,
                    "mass_lo": float(mass_range[0]), "mass_hi": float(mass_range[1]),
                    "sigma_lo": float(sigma_range[0]), "sigma_hi": float(sigma_range[1]),
                    "log_M_star": 0.5 * (float(mass_range[0]) + float(mass_range[1])),
                    "MIPS_24": f24, "MIPS_24_err": f24_err, "f24_cold": f24_cold,
                    "n_sources": int(getattr(sed, "n_sources", 0)),
                    "tier": getattr(sed, "fit_quality_tier", None) or "C",
                    "T_dust": float(sed.dust_temperature_rest_frame),
                    "log_amp": float(sed.amplitude),
                    "beta": float(sed.emissivity_index),
                })
        return pd.DataFrame(rows)

    df_sig = build_sigma_df(WRAPPERS_SIGMA)
    mass_cells = sorted(df_sig.groupby(["mass_lo", "mass_hi"]).groups.keys())
    sigma_cells = sorted(df_sig.groupby(["sigma_lo", "sigma_hi"]).groups.keys())
    print(f"{len(df_sig)} rows; mass cells: {mass_cells}; sigma cells: {sigma_cells}")
    D4_OK = len(mass_cells) >= 2 and len(sigma_cells) >= 2
except Exception as exc:
    print(f"D4 data loading failed: {exc}")
'''
)

code(
    r'''D4_ROWS = []
if D4_OK:
    _m1 = PAHSpectrumModel(feature_groups=FEATURE_GROUPS, bands=("MIPS_24",),
                           sigma_z0=SIGMA_Z0, f_cat=0.03)
    for (mlo, mhi) in mass_cells:
        for (slo, shi) in sigma_cells:
            sub = df_sig[(df_sig["mass_lo"] == mlo) & (df_sig["sigma_lo"] == slo)].copy()
            sub["prop_bin_id"] = 0
            if len(sub) < 8:
                continue
            try:
                res = _m1.fit_evolving(sub, evolve_amp=False, evolve_ratios=False,
                                       baseline_cols={"MIPS_24": "f24_cold"},
                                       feature_envelope="baseline")
            except Exception as exc:
                print(f"  cell M[{mlo},{mhi}) x S[{slo},{shi}): fit failed ({exc})")
                continue
            ab = sub[sub["tier"].isin(["A", "B"])]
            D4_ROWS.append({
                "mass_lo": mlo, "mass_hi": mhi, "sigma_lo": slo, "sigma_hi": shi,
                "sigma_mid": 0.5 * (slo + shi),
                "A_pah": float(res["A_pah"][0]), "A_pah_err": float(res["A_pah_err"][0]),
                "T_dust_med": float(np.nanmedian(ab["T_dust"])) if len(ab) else np.nan,
                "n_pts": len(sub),
            })
    d4 = pd.DataFrame(D4_ROWS)
    print(d4.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    mcolors = {mc: f"C{i}" for i, mc in enumerate(mass_cells)}
    for mc in mass_cells:
        sub = d4[d4["mass_lo"] == mc[0]].sort_values("sigma_mid")
        lab = rf"$ {mc[0]:.1f} < \log M_* < {mc[1]:.1f}$"
        axes[0].errorbar(sub["sigma_mid"], sub["A_pah"], yerr=sub["A_pah_err"],
                         fmt="o-", ms=8, capsize=4, color=mcolors[mc], label=lab)
        axes[1].errorbar(sub["T_dust_med"], sub["A_pah"], yerr=sub["A_pah_err"],
                         fmt="o-", ms=8, capsize=4, color=mcolors[mc], label=lab)
        s_sig = np.polyfit(sub["sigma_mid"], np.log10(np.maximum(sub["A_pah"], 1e-4)), 1)[0] \
            if len(sub) >= 3 else np.nan
        ok_t = np.isfinite(sub["T_dust_med"])
        s_td = np.polyfit(sub.loc[ok_t, "T_dust_med"],
                          np.log10(np.maximum(sub.loc[ok_t, "A_pah"], 1e-4)), 1)[0] \
            if ok_t.sum() >= 3 else np.nan
        print(f"mass {mc}: d logA/d log_sigma_SFR = {s_sig:+.3f}   "
              f"d logA/d T_dust = {s_td:+.4f} per K")
    axes[0].set_xlabel(r"$\log\,\Sigma_{\rm SFR}$ (bin mid)  [density proxy, C2]")
    axes[1].set_xlabel(r"median $T_{\rm dust}$ [K]  [radiation proxy, C1]")
    for ax in axes:
        ax.set_ylabel(r"$A_{\rm PAH}$ (PAH / cold continuum)")
        ax.grid(alpha=0.15)
    axes[0].legend(fontsize=9)
    fig.suptitle("D4: at fixed mass, which mediator orders the PAH amplitude?", fontsize=12)
    plt.tight_layout()
    fig.savefig("pah_twoarms_d4_mediators.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("D4 skipped -- sigma_SFR cross-cut not loadable in the expected shape.")
'''
)

md(
    r"""## 5 · D6 -- which (positive, negative) arm pair fits the three slopes?

The branch machinery from the Narayanan confrontation gives each mechanism
arm's predicted mass slope **as a function of z** through the same scaling
relations (Tacconi+18, van der Wel+14, Speagle+14):

- **P_Z(z)** -- metallicity step (from D1; effectively 0 at z~1, positive at z~3)
- **P_B** -- enrichment/PZR power law (branch B: q proportional to Z^s with NO
  saturation -- the version the JWST plateau result contradicts)
- **N_F(z)** -- f_mol/shattering arm (Narayanan channel through gas structure)
- **N_S(z)** -- Sigma_SFR arm (dense/intense star formation; C1 destruction and
  C2 dense-ISM suppression share this proxy -- D3/D4 separate them, not D6)

Fit measured(z_k) = c_P * P(z_k) + c_N * N(z_k) with non-negative weights for
each (P, N) pair; a pair "works" if it fits with O(1) coefficients (the
mechanism at face value), not just with arbitrary rescaling."""
)

code(
    r'''from scipy.optimize import nnls

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

def arm_slopes(zc):
    """Per-arm mass slope of log(L_PAH/L_IR) at redshift zc: (fmol band, sfr band)."""
    sigma_h2, sigma_sfr = galaxy_ism(zc, LOGM_GRID)
    dlog_ssfr_dlogm = np.polyfit(LOGM_GRID, np.log10(sigma_sfr), 1)[0]
    fmol_slopes = [sq * np.polyfit(LOGM_GRID, f_mol(sigma_h2, s), 1)[0]
                   for sq in SQ_BAND for s in SIGMA_HI_BAND]
    sfr_slopes = [sqs * dlog_ssfr_dlogm for sqs in SQS_BAND]
    return (min(fmol_slopes), max(fmol_slopes)), (min(sfr_slopes), max(sfr_slopes))

GAMMA_MZR = (0.15, 0.30)
S_PZR = (0.0, 1.5)
P_B_BAND = (GAMMA_MZR[0] * S_PZR[0], GAMMA_MZR[1] * S_PZR[1])
P_B_MID = 0.5 * sum(P_B_BAND)

print(f"{'slice':<8}{'P_Z metallicity step':>22}{'N_F fmol/shatter':>20}{'N_S Sigma_SFR':>18}"
      f"{'measured':>12}")
NF_mid, NS_mid = [], []
for k, (zlo, zhi, lab) in enumerate(Z_SLICES):
    zc = 0.5 * (zlo + zhi)
    (flo, fhi), (slo_, shi_) = arm_slopes(zc)
    NF_mid.append(0.5 * (flo + fhi)); NS_mid.append(0.5 * (slo_ + shi_))
    print(f"{lab:<8}{pred_mid[k]:>+22.3f}   [{flo:+.3f},{fhi:+.3f}]   [{slo_:+.3f},{shi_:+.3f}]"
          f"{MEASURED[k]:>+12.3f}")
NF_mid, NS_mid = np.array(NF_mid), np.array(NS_mid)
print(f"\nP_B enrichment/PZR power law (z-independent band): [{P_B_BAND[0]:+.3f}, {P_B_BAND[1]:+.3f}]")

pairs = {
    "P_Z step  + N_S Sigma_SFR": (pred_mid, NS_mid),
    "P_Z step  + N_F fmol":      (pred_mid, NF_mid),
    "P_B PZR   + N_S Sigma_SFR": (np.full(3, P_B_MID), NS_mid),
    "P_B PZR   + N_F fmol":      (np.full(3, P_B_MID), NF_mid),
}
NNLS_ROWS = []
print(f"\n{'pair':<28}{'c_P':>8}{'c_N':>8}{'chi2 (3 pts, 2 params)':>25}")
for name, (P, N) in pairs.items():
    X = np.column_stack([P, N]) / MEASURED_ERR[:, None]
    y = MEASURED / MEASURED_ERR
    coef, rnorm = nnls(X, y)
    chi2 = rnorm ** 2
    o1 = (0.3 < coef[0] < 3) and (0.3 < coef[1] < 3)
    flag = "  <- O(1) coefficients" if o1 else ""
    NNLS_ROWS.append((name, coef[0], coef[1], chi2, o1))
    print(f"{name:<28}{coef[0]:>8.2f}{coef[1]:>8.2f}{chi2:>18.2f}{flag}")
print("\nc ~ 1 means the mechanism works at face value through the scaling")
print("relations; c >> 1 means it must be inflated well beyond its published")
print("sensitivity; chi2 >> 1 (for 1 dof) means the pair cannot fit the shape.")
'''
)

md(
    r"""### D6 interpretation -- when NO arm pair fits

If every pair fails (chi2 >> 1 or absurd coefficients), that is itself the
finding: **the crossing is sharper than any equilibrium, linear-response
mechanism built on mean scaling relations.** All the candidate arms are
near-z-independent in their *mass slopes* because the scaling relations'
exponents barely evolve -- only their *normalizations* do (Sigma_SFR of a
massive main-sequence galaxy rises ~30x from z~1 to z~3). A linear-in-log
response to a variable whose mass slope is constant cannot flip the observed
slope; a **threshold response** can: if the PAH deficit switches on above a
critical Sigma_SFR (as it does locally -- normal disks show no deficit,
compact (U)LIRG cores do), then at z~1 even massive bins sit below threshold
(no suppression, slope set by the positive arm) while by z~3 the massive bins
cross it (strong suppression, slope inverts). Both C1-destruction and
C2-dense-ISM suppression admit threshold versions; D3/D4 remain the
separators. The actionable consequence: confront the Narayanan+26 simulation
*outputs* directly (they contain the full nonlinear response and the scatter
about the mean relations), rather than our linearized reconstruction of their
mechanism."""
)

code(
    r'''zgrid5 = np.linspace(0.6, 3.4, 40)
NF_lo, NF_hi, NS_lo, NS_hi = [], [], [], []
for zc in zgrid5:
    (flo, fhi), (slo_, shi_) = arm_slopes(zc)
    NF_lo.append(flo); NF_hi.append(fhi); NS_lo.append(slo_); NS_hi.append(shi_)

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.fill_between(zgrid, band_lo, band_hi, color="C1", alpha=0.20, lw=0,
                label="P_Z: metallicity step (D1 bracket)")
ax.fill_between(zgrid5, NF_lo, NF_hi, color="C0", alpha=0.20, lw=0,
                label="N_F: f_mol / shattering arm")
ax.fill_between(zgrid5, NS_lo, NS_hi, color="C4", alpha=0.20, lw=0,
                label=r"N_S: $\Sigma_{\rm SFR}$ arm")
ax.axhspan(P_B_BAND[0], P_B_BAND[1], color="C8", alpha=0.12, lw=0,
           label="P_B: enrichment/PZR power law")
ax.errorbar(zmids, MEASURED, yerr=MEASURED_ERR, fmt="o", ms=9, capsize=4,
            color="C3", zorder=5, label="measured (pooled, fold errors)")
ax.plot(zmids + 0.05, slice_slopes_combined, "^", ms=10, color="C2", zorder=5,
        label="measured (combined)")
ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
ax.set_xlabel("redshift")
ax.set_ylabel(r"$d\,\log(L_{\rm PAH}/L_{\rm IR})\,/\,d\,\log M_*$  [dex/dex]")
ax.set_title("D6: mechanism arms vs the measured slice slopes")
ax.grid(alpha=0.15)
ax.legend(fontsize=8.5, loc="lower left", ncol=2)
plt.tight_layout()
fig.savefig("pah_twoarms_d6_arm_decomposition.png", dpi=150, bbox_inches="tight")
plt.show()
'''
)

md(r"""## 6 · Verdict table""")

code(
    r'''print("=" * 96)
print("TWO-ARMS VERDICT -- which physics carries each arm of the crossing?")
print("=" * 96)
print(f"measured slopes (pooled, N-wtd mass): "
      + "  ".join(f"{lab} {MEASURED[k]:+.3f}+/-{MEASURED_ERR[k]:.3f}"
                  for k, (_, _, lab) in enumerate(Z_SLICES)))
print(f"combined stack check:                 "
      + "  ".join(f"{lab} {slice_slopes_combined[k]:+.3f}"
                  for k, (_, _, lab) in enumerate(Z_SLICES)))
print("-" * 96)

rows = []
rows.append(("D1 plateau (z~1 arm)",
             f"step ceiling {pred_band[0,1]:+.3f} vs measured {MEASURED[0]:+.3f}"
             f" ({d1_excess:+.1f} sigma above)",
             "favors C2 production arm" if d1_excess > 2 else
             ("inconclusive" if d1_excess > 1 else "C1 supply arm suffices")))
rows.append(("D2 gas-tracer floor",
             f"t_dep swing [{swing_tdep.min():+.2f},{swing_tdep.max():+.2f}]"
             f" vs measured swing {swing_meas:+.2f}",
             "pure gas tracing ruled out; density/destruction term required"))
d3_verdict = "not computed"
if np.isfinite(s_high_c) and np.isfinite(s_high_p):
    estimators_agree = (np.sign(s_high_c) == np.sign(s_high_p)
                        and abs(s_high_p) > 2 * s_high_ferr)
    if not estimators_agree:
        d3_verdict = ("INCONCLUSIVE -- combined and pooled disagree "
                      f"({s_high_c:+.2f} vs {s_high_p:+.2f}+/-{s_high_ferr:.2f}); "
                      "SNR-limited (6.2 um leverage only at z>2.4)")
    elif abs(s_high_c) > 0.5 * abs(MEASURED[2]):
        d3_verdict = ("mix trends WITH amplitude: "
                      + ("6.2 GAINS weight at high mass -> charging (B1), not "
                         "small-grain destruction" if s_high_c > 0 else
                         "6.2 LOSES weight at high mass -> destruction fingerprint (C1 arm)"))
    else:
        d3_verdict = "mix ~flat while amplitude inverts -> production fingerprint (C2 arm)"
rows.append(("D3 windowed ratios (z~3 arm)",
             f"high-window ratio slope: combined {s_high_c:+.3f}, "
             f"pooled {s_high_p:+.3f}+/-{s_high_ferr:.3f}; amplitude {MEASURED[2]:+.3f}",
             d3_verdict))
if D4_OK and len(D4_ROWS):
    rows.append(("D4 mediators (fixed mass)",
                 "see per-mass-row slopes above",
                 "directional only -- 3 sigma_SFR points/row; COSMOS25 catalog"))
else:
    rows.append(("D4 mediators", "not run", "sigma_SFR cross-cut unavailable"))
rows.append(("D5 metallicity track (z~3)",
             f"floor {pred_band[2,0]:+.3f} vs measured {MEASURED[2]:+.3f}"
             f" ({d5_gap:+.1f} sigma below)",
             "Z is not the controller at z~3" if d5_gap < -2 else "inconclusive"))
any_o1 = any(o1 and chi2 < 4 for *_, chi2, o1 in NNLS_ROWS)
best_pair = min(NNLS_ROWS, key=lambda r: r[3])
rows.append(("D6 arm pairing",
             f"best pair: {best_pair[0].strip()} (c_P={best_pair[1]:.1f}, "
             f"c_N={best_pair[2]:.1f}, chi2={best_pair[3]:.1f})",
             ("viable identification found" if any_o1 else
              "NO pair fits at face value -> crossing sharper than any "
              "equilibrium scaling-relation response; threshold/nonlinear "
              "physics or direct sim confrontation needed")))

for name, evidence, verdict in rows:
    print(f"{name:<30} {evidence}")
    print(f"{'':<30} -> {verdict}")
    print("-" * 96)
'''
)

md(
    r"""### How to read this, and what it does NOT settle

- **D1/D5 are literature-vs-us tests**: they use no free knobs of ours, only the
  published q_PAH(Z) saturation and the MZR. If the z~1 slope clears the step
  ceiling, the positive arm needs a mass-correlated driver that keeps rising
  above half-solar metallicity -- gas structure (C2) is the natural candidate,
  but any such driver would do. The manual check on Shivaei+24's q_PAH(M*)
  *within our mass window* remains open.
- **D3 is the internal fingerprint test** but it is SNR-limited: the high
  window leans on the 6.2 um amplitude, which is only trustworthy in the
  combined stack (bin0 A(6.2) SNR defect, branch-9). Treat a null here as
  weak evidence, a strong coupled trend as meaningful.
- **D4 is directional only** in this first pass (3 sigma_SFR points per mass
  row, different catalog). If it looks promising the right follow-up is a
  dedicated cross-cut on COSMOS2020 with the widened bins.
- **D6 uses OUR construction of the mechanism arms** through scaling relations
  (same machinery as the branch-A/B bands); the sims themselves publish no
  q_PAH(M*) at fixed z. The definitive external arbiter is still extracting
  that slope from the Narayanan+26 outputs directly.
- The error bars everywhere are fold-scatter (single field): calibrated for
  ordering/sign statements, not for quotable p-values (branch-9 caveat)."""
)


nb["cells"] = cells
out = "notebooks/2026-07-11-two-arms-tests.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out} ({len(cells)} cells)")
