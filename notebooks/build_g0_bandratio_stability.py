"""Task 1 of pah-interpretation-3: can the ionised/neutral decomposition be stabilised?

The brief calls this "the gate": pooled and combined gave ionised/neutral mass slopes
that inverted once the (leverage-free) 6.2 group was dropped, so nothing downstream was
safe.  Answer, 2026-07-27: **the disagreement was never about the measurable quantity,
and the measurable quantity is nonetheless not usable.**

  1. The two estimators agree on the RATIO (neutral/ionised) and disagree only on the
     amplitude normalisation A = alpha/C_m -- the known-degenerate quantity.
  2. z-sampling is not the cause: downsampling pooled onto combined's z grid does not
     move it toward combined.
  3. But MIPS 24 samples the two groups at DISJOINT redshifts (neutral leverage peaks
     at z~0.95, ionised at z~1.95), so the ratio is structurally a comparison of the
     z~0.95 flux against the z~1.85 flux -- not a measurement at a redshift.
  4. Consequence: the ratio inherits the full mass-dependent baseline systematic across
     that Dz~0.9, and switching the cold baseline from smoothed to raw FLIPS ITS SIGN
     (-0.227 -> +0.153).  And its z-evolution -- task 2's target -- cannot be measured
     at all, because obtaining one ratio value already requires integrating the window.

Run:  uv run python notebooks/build_g0_bandratio_stability.py   (~6 min; the jackknives
      dominate).  Needs the branch-12 money-plot state pickle; set QPAH_SP.
"""
import os, sys, time, pickle, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simstack4.pah_spectrum import PAHSpectrumModel as PSM, feature_band_curves

SP = os.environ.get(
    "QPAH_SP",
    "/private/tmp/claude-501/-Users-mviero-Repositories-simstack4-2026"
    "/d8fb5aec-f3fd-4822-b15e-a2b304344e5f/scratchpad",
)
S = pickle.load(open(SP + "/state2.pkl", "rb"))
KW, MB, bc = S["PAH_MODEL_KW"], S["MASS_BINS"], np.asarray(S["bin_ctrs"])
G3 = [[1, 2], [0], [3, 4]]          # as published: ionised ref, 6.2, neutral
G2 = [[1, 2], [3, 4]]               # 6.2 dropped
POOL, COMB = S["df_pool_sm"], S["df_combined_sm"]


def fit_bins(dff, groups=G2, zlo=0.0, zhi=2.0, zkeep=None,
             baseline="f24_cold", silicate=None, nmin=5):
    """Per-mass-bin A = alpha/C_m (PAH per continuum) and r = neutral/ionised."""
    kw = {**KW, "bands": ("MIPS_24",), "feature_groups": groups}
    if silicate is not None:
        kw["include_silicate"] = silicate
    m = PSM(**kw)
    A, R = [], []
    for i in range(len(MB)):
        sub = dff[(dff.prop_bin_id == i) & (dff.z_mid >= zlo) & (dff.z_mid < zhi)].copy()
        if zkeep is not None:
            sub = sub[sub.z_mid.isin(zkeep)]
        if len(sub) < nmin:
            A.append(np.nan); R.append(np.nan); continue
        sub["prop_bin_id"] = 0
        r = m.fit_evolving(sub, evolve_amp=False, evolve_ratios=False,
                           baseline_cols={"MIPS_24": baseline},
                           feature_envelope="baseline")
        A.append(float(np.asarray(r["alpha"])[0]) / float(np.asarray(r["C_m"])[0]))
        R.append(float(np.asarray(r["r"])[-1]))
    return np.array(A), np.array(R)


def sl(v):
    ok = np.isfinite(v) & (v > 0)
    return float(np.polyfit(bc[ok], np.log10(v[ok]), 1)[0]) if ok.sum() >= 3 else np.nan


def three(A, R):
    return sl(A), sl(A * R), sl(R)


def jackknife(dff, zgrid, **kw):
    v = np.array([three(*fit_bins(dff, zkeep=set(zgrid) - {z}, **kw)) for z in zgrid])
    n = len(v)
    mu = np.nanmean(v, 0)
    return mu, np.sqrt((n - 1) / n * np.nansum((v - mu) ** 2, axis=0))


# ============================================================ 1. the ratio is stable
print("=" * 78)
print("1.  WHAT ACTUALLY DISAGREES.  'RATIO' = neutral/ionised, the G0 observable;")
print("    it is the difference of the first two columns and needs no alpha/C_m split.")
print(f"\n{'':<12}{'variant':<26}{'ion':>9}{'neu':>9}{'RATIO':>9}")
res = {}
for tag, dff in (("pooled", POOL), ("combined", COMB)):
    for lbl, g, zh in (("3 groups (6.2 railed)", G3, 2.0), ("6.2 dropped, z<2", G2, 2.0),
                       ("6.2 dropped, z<1.6", G2, 1.6), ("6.2 dropped, z<2.4", G2, 2.4)):
        res[(tag, lbl)] = three(*fit_bins(dff, groups=g, zhi=zh))
        print(f"  {tag:<10}{lbl:<26}" + "".join(f"{x:+9.3f}" for x in res[(tag, lbl)]))
print("\n  spread across all 8 (estimator x window) variants:")
for k, nm in enumerate(("ionised", "neutral", "RATIO")):
    v = np.array([res[key][k] for key in res])
    print(f"    {nm:<10} mean {v.mean():+.3f}  sd {v.std(ddof=1):.3f}  "
          f"range {v.min():+.3f} .. {v.max():+.3f}")
print("""
  The amplitude columns scatter by 0.26 and 0.22; the RATIO by 0.058, with six of the
  eight inside -0.227..-0.264.  The two estimators never disagreed about the ratio --
  they disagree about A = alpha/C_m, which branch-5/6 already established is strongly
  alpha-degenerate.  Same lesson as the EW slope: quote the differential.""")

# =========================================================== 2. not a z-sampling issue
print("\n" + "=" * 78)
print("2.  IS IT Z-SAMPLING?  Downsample pooled onto combined's z grid.")
zc = np.sort(COMB.query("z_mid < 2.0").z_mid.unique())
zp = np.sort(POOL.query("z_mid < 2.0").z_mid.unique())
near = {zp[np.argmin(np.abs(zp - z))] for z in zc}
print(f"  combined z<2: {len(zc)} pts;  pooled z<2: {len(zp)} pts;  matched subset: {len(near)}")
for lbl, v in (("pooled, full grid", res[("pooled", "6.2 dropped, z<2")]),
               ("pooled on combined grid", three(*fit_bins(POOL, zkeep=near))),
               ("combined", res[("combined", "6.2 dropped, z<2")])):
    print(f"  {lbl:<26}" + "".join(f"{x:+9.3f}" for x in v))
print("  -> matching the grid moves the ionised slope AWAY from combined's.  Not sampling.")

# ==================================================== 3. the structural leverage problem
print("\n" + "=" * 78)
print("3.  WHY THE RATIO IS NOT A MEASUREMENT AT A REDSHIFT")
C = feature_band_curves(zp, "MIPS_24", features=KW["features"], feature_groups=G2,
                        profile=KW.get("profile", "drude"), tau_sil=0.55)
print("  MIPS 24 samples a feature centre at z = 24/lambda - 1:")
print("    12.7um -> z=0.89   11.3um -> z=1.12   8.6um -> z=1.79   7.7um -> z=2.12 (OUT)")
print(f"\n  bandpass-integrated group leverage T_g(z):")
print(f"  {'z':>6}{'rest lam':>10}{'ionised':>11}{'neutral':>11}")
for k, zz in enumerate(zp):
    if k % 2 == 0:
        print(f"  {zz:>6.2f}{24/(1+zz):>10.2f}{C[k,0]:>11.4f}{C[k,1]:>11.4f}")
for nm, lo, hi in (("z 0.2-1.1", 0.2, 1.1), ("z 1.1-2.0", 1.1, 2.0), ("z 0.2-2.0", 0.2, 2.0)):
    s = C[(zp >= lo) & (zp < hi)].sum(0)
    print(f"  {nm}: summed  ion {s[0]:.3f}  neu {s[1]:.3f}   ion/neu {s[0]/s[1]:.2f}")
print("""
  Neutral leverage peaks at z~0.95, ionised at z~1.95.  In z<1.1 the ionised group holds
  3% of its z<2 leverage; in z>1.1 the neutral group holds 42% of its.  NEITHER HALF
  CONSTRAINS BOTH.  The fitted ratio is therefore a comparison of the z~0.95 flux against
  the z~1.85 flux -- anything that varies with z between those epochs in a mass-dependent
  way maps straight onto it.""")

# ================================================ 4. disjoint halves: structurally dead
print("\n" + "=" * 78)
print("4.  TASK 2's TARGET (does the ratio's mass slope EVOLVE?) IS NOT MEASURABLE")
t0 = time.time()
for lo, hi in ((0.2, 1.1), (1.1, 2.0), (0.2, 2.0)):
    zw = zp[(zp >= lo) & (zp < hi)]
    A, R = fit_bins(POOL, zlo=lo, zhi=hi)
    mu, se = jackknife(POOL, zw, zlo=lo, zhi=hi)
    print(f"  z {lo}-{hi}  ({len(zw)} pts, A>0 in {int(np.sum(np.isfinite(A)&(A>0)))}/4 bins)  "
          + "  ".join(f"{n} {mu[k]:+.3f}+-{se[k]:.3f}"
                      for k, n in enumerate(("ion", "neu", "RATIO"))))
    print(f"           r per mass bin: " + " ".join(f"{x:7.3f}" for x in R))
print(f"  ({time.time()-t0:.0f}s)")
print("""
  Lower half: r pinned at ~1.00 in all four mass bins with a jackknife error of 0.019 --
  that is an unconstrained parameter sitting at its degenerate value, not a tight
  measurement.  Upper half: A rails to -106 and -268 in two of four bins and the ratio
  error blows up to 0.45.  A slope that appears in the union but in neither disjoint
  half is not a mass trend -- it is the z-baseline being read as one.""")

# ======================================================= 5. the baseline flips the sign
print("\n" + "=" * 78)
print("5.  SYSTEMATIC BUDGET ON THE RATIO SLOPE -- the baseline choice flips its SIGN")
rows = []
for lbl, kw in (("baseline = f24_cold (smoothed)", {}),
                ("baseline = f24_cold_smooth", {"baseline": "f24_cold_smooth"}),
                ("baseline = f24_cold_raw", {"baseline": "f24_cold_raw"}),
                ("silicate OFF", {"silicate": False})):
    v = three(*fit_bins(POOL, **kw))
    rows.append((lbl, v))
    print(f"  {lbl:<32}" + "".join(f"{x:+9.3f}" for x in v))
mu, se = jackknife(POOL, zp)
print(f"\n  statistical (delete-one-z jackknife, pooled z<2): RATIO {mu[2]:+.3f} +- {se[2]:.3f}")
print(f"  estimator x window systematic                  : +-0.058")
print(f"  silicate on/off                                : {rows[3][1][2]-rows[0][1][2]:+.3f}")
print(f"  cold baseline raw -> smoothed                  : {rows[0][1][2]-rows[2][1][2]:+.3f}"
      f"   <-- SIGN FLIP ({rows[2][1][2]:+.3f} -> {rows[0][1][2]:+.3f})")
print("""
  How the cold baseline is interpolated across the Dz~0.9 that separates the two groups'
  leverage peaks decides whether PAHs look MORE ionised with stellar mass (-0.227) or
  LESS (+0.153).  `smoothed_ms_baseline` exists precisely to stabilise that Wien-side
  tail, and was chosen for reasons that have nothing to do with this measurement.
  Section 6 checks whether that is coincidence.""")

# ============================================ 6. the baseline tilt ACCOUNTS for the signal
print("\n" + "=" * 78)
print("6.  IT IS NOT JUST A SYSTEMATIC -- IT ACCOUNTS FOR THE SIGNAL")
print("  Smoothing changes the baseline BETWEEN the two leverage epochs, mass-dependently:")
print(f"  {'':>16}{'z~0.95':>10}{'z~1.85':>10}{'difference':>13}")
d = POOL[POOL.z_mid < 2.0]
tilt = []
for i in range(len(MB)):
    s = d[d.prop_bin_id == i]
    lo = np.log10(s[(s.z_mid > 0.8) & (s.z_mid < 1.1)].eval("f24_cold_smooth/f24_cold_raw")).mean()
    hi = np.log10(s[(s.z_mid > 1.7) & (s.z_mid < 2.0)].eval("f24_cold_smooth/f24_cold_raw")).mean()
    tilt.append(hi - lo)
    print(f"  logM={bc[i]:<11.2f}{lo:>+10.4f}{hi:>+10.4f}{hi-lo:>+13.4f}")
pred = float(np.polyfit(bc, tilt, 1)[0])
meas = rows[0][1][2] - rows[2][1][2]
print(f"\n  d(that tilt)/d log M*                       = {pred:+.3f} dex/dex")
print(f"  measured ratio-slope shift (smoothed - raw) = {meas:+.3f} dex/dex")
print(f"  -> the baseline tilt accounts for {abs(pred/meas)*100:.0f}% of it.")
print("""
  Sign check: smoothing lowers the z~1.85 baseline relative to z~0.95, and does so more
  for massive bins.  A lower ionised-epoch baseline forces a LARGER ionised amplitude,
  hence a SMALLER neutral/ionised ratio -- increasingly so with mass.  That is precisely
  the "PAHs become more ionised with stellar mass" signal, manufactured by the smoothing.

  VERDICT: the gate does not open.  The band ratio is the only G0 estimator that is not
  a scaling relation in disguise, and with a single broad band it is degenerate with the
  baseline's mass-dependent z-tilt -- which reproduces 84% of the claimed signal.
  Separating them needs the two groups sampled at the SAME redshift -- MIRI, not more
  stacking.  MIPS 70 does not help: it reaches 12.7um at z~4.5 and 11.3um at z~5.2, a
  different epoch again.""")
