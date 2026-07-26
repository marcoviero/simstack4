"""D4 lever-arm check: is T_dust separable from sigma_SFR at fixed (z, M*)?

D4 (docs/pah-interpretation-candidates.md) asks whether the PAH deficit is mediated
by RADIATION (T_dust -> destruction/charging) or by GAS DENSITY (sigma_SFR ->
suppressed shattering). That test is only possible if T_dust carries information
sigma_SFR does not, at fixed (z, M*). This measures that lever before any D4
machinery gets built.

Method: project (z, log M*) out of both T_dust and log sigma_SFR, then
  partial r          -- how collinear the two mediators are (near +-1 => D4 impossible)
  independent lever  -- resid_T * sqrt(1 - r^2), the part sigma_SFR cannot explain
  S/N                -- that lever over the median statistical error on T_dust

CRITICAL: filter on T_dust ERROR and physicality, not tier. The tier grades band
SNR, not whether the SED fit converged -- one cell here has T=140 K with a 1514 K
error bar and is labelled Tier A. Unfiltered it doubles the apparent lever
(S/N 5.3 vs the true 2.5).

Run:  uv run python notebooks/build_d4_lever_check.py
"""
import ast, os, sys, numpy as np, pandas as pd
sys.path.insert(0, "/Users/mviero/Repositories/simstack4-2026/notebooks")
os.chdir("/Users/mviero/Repositories/simstack4-2026/notebooks")
from simstack4.wrapper import SimstackWrapper
from simstack4.plots import _extract_pop_type

RUN = "cosmos25_stacking_20260317_201727"      # 5 z x 4 mass x 5 sigma_SFR
path = os.path.join(os.environ["PICKLESPATH"], "simstack", "stacked_flux_densities")
w = SimstackWrapper(); w.load_stacking_results(os.path.join(path, RUN + ".json"))
w.run_analysis_only(use_mcmc=False, temperature_prior="viero", snr_high=5.0, snr_low=2.0,
                    use_covariance=True, use_pah=False)

rows = []
for pid, sed in w.processed_results.sed_results.items():
    if not getattr(sed, "greybody_fit_success", False):
        continue
    bp = sed.bin_properties
    if isinstance(bp, str):
        try: bp = ast.literal_eval(bp)
        except Exception: continue
    if not isinstance(bp, dict):
        continue
    rows.append(dict(
        pop=_extract_pop_type(pid),
        tier=getattr(sed, "fit_quality_tier", "C") or "C",
        n=getattr(sed, "n_sources", np.nan),
        T=getattr(sed, "dust_temperature_rest_frame", np.nan),
        Terr=getattr(sed, "dust_temperature_error", np.nan),
        z=bp.get("zpdf_med", np.nan),
        logM=bp.get("mass_med", np.nan),
        logSig=bp.get("log_sigma_sfr", np.nan)))
d = pd.DataFrame(rows)
d = d[(d["pop"] == "split_0")]
d = d[np.isfinite(d[["T", "z", "logM", "logSig"]]).all(axis=1)]
d = d[(d["T"] > 0) & np.isfinite(d["Terr"]) & (d["Terr"] > 0)]
print(f"SF (split_0) cells usable: {len(d)}   tiers {d.tier.value_counts().to_dict()}")
print(f"  z      {d.z.min():.2f}-{d.z.max():.2f}   logM {d.logM.min():.2f}-{d.logM.max():.2f}"
      f"   logSig {d.logSig.min():.2f}-{d.logSig.max():.2f}")
print(f"  T_dust {d['T'].min():.1f}-{d['T'].max():.1f} K, median stat err {d['Terr'].median():.2f} K")

def resid(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b

T_MAX = 45.0        # greybody bounds are [15, 60] K; MS galaxies live well below 45
TERR_MAX = 15.0     # reject unconstrained fits (one cell here has err = 1514 K)
cuts = [("ALL (no cuts -- inflated)", d),
        (f"T < {T_MAX:g} K", d[d["T"] < T_MAX]),
        (f"T < {T_MAX:g} K + err < {TERR_MAX:g} K", d[(d["T"] < T_MAX) & (d["Terr"] < TERR_MAX)]),
        (f"T < {T_MAX:g} K + err cut + Tier A/B",
         d[(d["T"] < T_MAX) & (d["Terr"] < TERR_MAX) & d.tier.isin(["A", "B"])])]
print(f"\n  cells failing the physicality/error cut: "
      f"{int(((d['T'] >= T_MAX) | (d['Terr'] >= TERR_MAX)).sum())} of {len(d)}"
      f"  (worst error bar {d['Terr'].max():.0f} K, tier "
      f"{d.loc[d['Terr'].idxmax(), 'tier']})")
for tag, sub in cuts:
    if len(sub) < 10:
        print(f"\n[{tag}] only {len(sub)} cells — skipped"); continue
    X = np.column_stack([np.ones(len(sub)), sub.z, sub.logM])
    rT, rS = resid(sub["T"].to_numpy(), X), resid(sub.logSig.to_numpy(), X)
    r = float(np.corrcoef(rT, rS)[0, 1])
    sT, sS = rT.std(ddof=1), rS.std(ddof=1)
    err = float(sub["Terr"].median())
    ind = sT * np.sqrt(max(0.0, 1 - r**2))
    print(f"\n[{tag}]  N={len(sub)}   after projecting out (z, logM):")
    print(f"  resid T_dust scatter    {sT:6.2f} K      resid logSigma scatter {sS:.2f} dex")
    print(f"  partial corr(T, logSig | z, logM) = {r:+.3f}   -> {100*r**2:.0f}% shared, "
          f"{100*(1-r**2):.0f}% independent")
    print(f"  INDEPENDENT T_dust lever = {ind:.2f} K   vs median stat err {err:.2f} K"
          f"   -> S/N = {ind/err:.2f}")
    print(f"  verdict: {'LEVER EXISTS' if ind / err > 1.5 else 'NO USABLE LEVER'}"
          f"   (partial r^2 = {100 * r**2:.0f}% shared -> mediators are"
          f" {'COLLINEAR' if abs(r) > 0.7 else 'separable'})")
out = os.path.join(os.environ.get("SP", "."), "d4_lever_cells.csv")
d.to_csv(out, index=False)
print(f"\nper-cell table -> {out}")
