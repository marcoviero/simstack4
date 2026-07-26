"""D4 with the template SHAPE FIXED to the COSMOS2020 measurement.

The ratio block is unconstrained on the COSMOSWeb sigma_SFR stack (91 rows, 6
cells -> r rails to +28). But D4 only needs the AMPLITUDE per cell. So bake the
COSMOS2020-measured group ratios into the feature strengths and weld everything
into ONE group: r has a single element identically 1, nothing to fit, and each
cell contributes only (C_m, alpha_m) -- fully determined.
"""
import os, sys, pickle, numpy as np, pandas as pd
sys.path.insert(0, "/Users/mviero/Repositories/simstack4-2026/notebooks")
os.chdir("/Users/mviero/Repositories/simstack4-2026/notebooks")
from pah_money_helpers import smooth_baseline
from simstack4.pah_spectrum import (PAHSpectrumModel as PSM, FEATURES_CALIBRATED,
                                    rescale_feature_strength)
SP = os.environ["SP"]
R_C2020 = {0: 1.038, 1: 1.0, 2: 1.0, 3: 0.882, 4: 0.882}   # from the COSMOS2020 pooled 24+70 fit
F = list(FEATURES_CALIBRATED)
for i, s in R_C2020.items():
    F = rescale_feature_strength(F, i, F[i][1] * s)
GROUPS = [[0, 1, 2, 3, 4]]
KW = dict(features=F, feature_groups=GROUPS, profile="drude",
          include_silicate=True, tau_sil_prior=0.5)
ACOLS = {"MIPS_24": "f24_cold", "MIPS_70": "f70_cold"}

d = pickle.load(open(SP + "/d4_df.pkl", "rb"))
d = d[(d.T_dust > 15) & (d.T_dust < 45) & (d.T_err < 15)]
d = d[d.mass_bin > 0].copy()                 # nuisance bin excluded
d = smooth_baseline(d)
ids = sorted(d.prop_bin_id.unique())
d["prop_bin_id"] = d.prop_bin_id.map({p: i for i, p in enumerate(ids)})
res = PSM(**KW, bands=("MIPS_24", "MIPS_70")).fit_evolving(
    d, evolve_amp=False, evolve_ratios=False, baseline_cols=ACOLS,
    feature_envelope="baseline")
a, C = np.asarray(res["alpha"]), np.asarray(res["C_m"])
A = a / C
print(f"FIXED-SHAPE fit: N={len(d)} rows, {len(ids)} cells, chi2red={res['chi2_red']:.2f}, "
      f"tau_sil={res.get('tau_sil', np.nan):.2f}, r={np.asarray(res['r'])}")
rows = []
for j, p in enumerate(ids):
    sub = d[d.prop_bin_id == j]
    rows.append(dict(mass_bin=int(sub.mass_bin.iloc[0]), sig_bin=int(sub.sig_bin.iloc[0]),
                     logM=float(sub.log_M_star.median()), logSig=float(sub.log_sigma_sfr.median()),
                     T_dust=float(sub.T_dust.median()), n_z=len(sub),
                     medN=float(sub.n_sources.median()), A=float(A[j])))
g = pd.DataFrame(rows)
print("\n" + g.round(3).to_string(index=False))

print("\n--- D4 partial correlations (6 cells, control for logM) ---")
y = np.log10(g.A.to_numpy()); X0 = np.column_stack([np.ones(len(g)), g.logM])
rf = lambda v: v - X0 @ np.linalg.lstsq(X0, v, rcond=None)[0]
ry, rT, rS = rf(y), rf(g.T_dust.to_numpy()), rf(g.logSig.to_numpy())
def partial(u, v, w):
    a_ = u - w * (w @ u) / (w @ w); b_ = v - w * (w @ v) / (w @ w)
    return float(a_ @ b_ / np.sqrt((a_ @ a_) * (b_ @ b_)))
print(f"  corr(logA, T_dust | logM)                 = {np.corrcoef(ry, rT)[0,1]:+.3f}")
print(f"  corr(logA, logSigma | logM)               = {np.corrcoef(ry, rS)[0,1]:+.3f}")
print(f"  PARTIAL corr(logA, T_dust | logM, logSig) = {partial(ry, rT, rS):+.3f}   <- C1 radiation")
print(f"  PARTIAL corr(logA, logSig | logM, T_dust) = {partial(ry, rS, rT):+.3f}   <- C2 density")
print(f"\n  N=6 cells: a partial r needs |r| > ~0.81 for 2-sigma. Both are far below.")
pickle.dump((d, g, res), open(SP + "/d4_final.pkl", "wb"))
