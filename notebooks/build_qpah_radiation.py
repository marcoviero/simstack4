"""Step 0-bis, done properly: <U> from T_dust, and the search for a G0 proxy.

Supersedes `build_qpah_backout.py` (which is kept as the first pass).  Four things
that pass did not do:

  1. propagate T_dust through <U> ~ T^(4+beta) with the disjoint-fold ensemble, and
     expose the estimator systematic (median vs source-weighted mean) that turns out
     to be as large as the effect being claimed;
  2. cross-check the measured T(M*,z) mass gradient against Viero+2013 (ApJ 779, 32)
     Eq 19-21 -- an EXTERNAL relation, not built from the same greybody fit that made
     L_IR, so it breaks the T_dust <-> L_IR covariance the brief warns about;
  3. replace the ad-hoc `G0 ~ sSFR^a` proxy with the physical decoupling index
     `G0 ~ <U>^g`, which makes explicit that "T_dust correction only" is not a
     null assumption -- it is g=0, the most extreme decoupling possible;
  4. INVERT the problem: ask what d log G0 / d log M* would have to be, and screen
     every proxy in hand against that requirement.

Run:  uv run python notebooks/build_qpah_radiation.py
Needs the branch-12 money-plot state pickles (`state2.pkl`, `zr3.pkl`); set QPAH_SP.
"""
import os, pickle, numpy as np

SP = os.environ.get(
    "QPAH_SP",
    "/private/tmp/claude-501/-Users-mviero-Repositories-simstack4-2026"
    "/d8fb5aec-f3fd-4822-b15e-a2b304344e5f/scratchpad",
)
S = pickle.load(open(SP + "/state2.pkl", "rb"))
Z = pickle.load(open(SP + "/zr3.pkl", "rb"))

MC = np.array(S["bin_ctrs"])                       # 10.25 10.70 10.90 11.25
WINS = [(0.5, 1.4, "z~1"), (1.4, 2.4, "z~2"), (2.4, 3.5, "z~3")]
ZC = np.array([0.95, 1.90, 2.95])                  # slice centres used throughout
NB, NZ = len(MC), len(WINS)
BETA, P = 1.8, 5.8                                 # beta FIXED in the fits; <U> ~ T^P

# d log Sigma_SFR / d log M* per slice, from the threshold fit (branch-2,
# COSMOSWeb sizes) -- docs/pah-interpretation-candidates.md
DSIG = np.array([0.242, 0.310, 0.343])


def grid(d, col, how="wmean"):
    """(mass, z-window) summary of `col`, with the T_dust physicality cut applied."""
    d = d[(d.T_dust > 15) & (d.T_dust < 45) & np.isfinite(d.T_dust)]
    out = np.full((NB, NZ), np.nan)
    for i in range(NB):
        for j, (zlo, zhi, _) in enumerate(WINS):
            sub = d[(d.prop_bin_id == i) & (d.z_mid >= zlo) & (d.z_mid < zhi)]
            v = sub[col].to_numpy() if len(sub) else np.array([])
            if not np.isfinite(v).any():
                continue
            w = sub.n_sources.to_numpy().astype(float)
            ok = np.isfinite(v)
            out[i, j] = (np.median(v[ok]) if how == "median"
                         else np.sum(v[ok] * w[ok]) / np.sum(w[ok]))
    return out


def slopes(M):
    """d(quantity)/d log M* in each z window."""
    out = np.full(NZ, np.nan)
    for j in range(NZ):
        ok = np.isfinite(M[:, j])
        if ok.sum() >= 3:
            out[j] = np.polyfit(MC[ok], M[ok, j], 1)[0]
    return out


def viero13_T(logM, z):
    """T(M,z) = A_z * (log10 M*)^alpha_T   -- Viero+2013 ApJ 779, 32, Eq 19-21:
        A_z     = A0 + A1 (1+z)^A2,        A = [-439.83, 578.93, 0.11]
        alpha_T = a0 + a1 (1+z)^a2,        a = [-0.81, 2.84e-5, 3.55]
    Note their beta = 2 (ours is 1.8) and the mass enters as log10 M*, not M*."""
    A = -439.83 + 578.93 * (1.0 + z) ** 0.11
    a = -0.81 + 2.84e-5 * (1.0 + z) ** 3.55
    return A * np.asarray(logM, float) ** a


def row(label, v, extra=""):
    print(f"  {label:<34s}" + "".join(f"{x:+9.3f}" for x in v) + extra)


logLR = np.log10(Z["zr_pool"])
s_obs = slopes(logLR)
Tw = grid(S["df_pool_sm"], "T_dust", "wmean")
Tm = grid(S["df_pool_sm"], "T_dust", "median")
Tv = np.array([[viero13_T(MC[i], ZC[j]) for j in range(NZ)] for i in range(NB)])

# =====================================================================  1. T grids
print("=" * 78)
print("1.  T_dust(M*, z):  measured (two estimators) vs Viero+2013 Eq 19")
print(f"{'':>12}{'wmean':>24}{'median':>24}{'Viero+13':>21}")
print(f"{'':>12}" + ("    z~1    z~2    z~3" + " " * 3) * 3)
for i in range(NB):
    print(f"  logM={MC[i]:5.2f}" + "   " + "".join(f"{Tw[i,j]:7.2f}" for j in range(NZ))
          + "   " + "".join(f"{Tm[i,j]:7.2f}" for j in range(NZ))
          + "   " + "".join(f"{Tv[i,j]:7.2f}" for j in range(NZ)))

gw, gm, gv = (slopes(np.log10(x)) for x in (Tw, Tm, Tv))
print("\n  d log10 T / d log M*                z~1      z~2      z~3     (z3 - z1)")
for nm, g in (("measured, source-weighted mean", gw), ("measured, median", gm),
              ("Viero+2013 Eq 19-21", gv)):
    row(nm, g, f"   {g[2]-g[0]:+.4f}")
print("\n  => the <U> term it contributes, p * dlogT/dlogM  (p = 4 + beta = 5.8)")
for nm, g in (("measured, source-weighted mean", gw), ("measured, median", gm),
              ("Viero+2013 Eq 19-21", gv)):
    row(nm, P * g, f"   {P*(g[2]-g[0]):+.4f}")
print("""
  Viero+2013's mass gradient is z-INDEPENDENT to four decimals: its Eq 21 fitted
  alpha_T,1 = 2.8e-5, i.e. the data preferred no evolution of the T-M* slope.  An
  external T(M*,z) therefore shifts all three slopes by the same -0.19 dex/dex and
  changes the swing by EXACTLY ZERO.  Every bit of swing the T correction buys comes
  from our own fits' z-evolving gradient (-0.057 -> -0.007), which Viero+2013 does
  not see -- on a mass-selected Herschel stack of the same kind.""")

# ==============================================================  2. decoupling index
print("\n" + "=" * 78)
print("2.  q_PAH = (L_PAH/L_IR) * <U>/G0,  with  G0 ~ <U>^g")
print("    g=1: fixed geometry, the field terms cancel, NO correction")
print("    g=0: the PAH-heating field is constant across 1 dex of M* and z=0.5-3.5")
print("         (this is what 'T_dust correction only' silently assumed)")
row("OBSERVED", s_obs, f"   swing {s_obs[2]-s_obs[0]:+.3f}")
print(f"\n{'g':>6}{'z~1':>10}{'z~2':>10}{'z~3':>10}{'swing':>10}   crossing")
for g in (1.0, 0.75, 0.5, 0.25, 0.0):
    s = slopes(logLR + (1 - g) * P * np.log10(Tw))
    print(f"{g:>6.2f}" + "".join(f"{x:+10.3f}" for x in s)
          + f"{s[2]-s[0]:+10.3f}   {'YES' if (s[0] > 0 and s[2] < 0) else 'gone'}")
gc = 1.0 + s_obs[0] / (P * gw[0])
print(f"\n  z~1 arm reaches zero only at g = {gc:+.2f} (weighted mean): G0 would have to"
      f"\n  ANTI-correlate with <U>.  With the median estimator the arm is already at zero"
      f"  at g = {1.0 + s_obs[0]/(P*gm[0]):+.2f}.  That gap IS the estimator systematic.")

# ==================================================================  3. error budget
print("\n" + "=" * 78)
print("3.  DISJOINT-FOLD ENSEMBLE (T and L_PAH/L_IR both re-measured in each fold)")
folds, zrf = S["fold_dfs_sm"], Z["zr_folds"]
for how in ("wmean", "median"):
    print(f"\n  estimator = {how}")
    for g in (1.0, 0.5, 0.0):
        ens = np.array([slopes(np.log10(zrf[k]) + (1 - g) * P * np.log10(grid(fd, "T_dust", how)))
                        for k, fd in enumerate(folds)])
        mu, sd = np.nanmean(ens, 0), np.nanstd(ens, 0, ddof=1)
        sw = ens[:, 2] - ens[:, 0]
        print(f"    g={g:.1f}  " + "  ".join(f"{mu[j]:+.3f}+-{sd[j]:.3f}" for j in range(NZ))
              + f"   swing {np.nanmean(sw):+.3f}+-{np.nanstd(sw, ddof=1):.3f}"
              + f"  ({abs(np.nanmean(sw))/np.nanstd(sw, ddof=1):.1f} sigma)")

Tk = np.array([grid(fd, "T_dust", "wmean") for fd in folds])
print(f"\n  fold scatter on T is {np.nanmax(np.nanstd(Tk, 0, ddof=1)):.2f} K at worst;"
      f" 1 K at 30 K = {P*np.log10(31/30):.3f} dex in <U>.")
print("  beta sensitivity (it is FIXED at 1.8, and 4+beta IS the lever):")
for b in (1.5, 1.8, 2.0):
    s = slopes(logLR + (4 + b) * np.log10(Tw))
    print(f"    beta={b:.1f}  p={4+b:.1f}   g=0 slopes " + "".join(f"{x:+8.3f}" for x in s)
          + f"   swing {s[2]-s[0]:+.3f}")

# =============================================================  4. required G0, and
#                                                                  the proxy screen
print("\n" + "=" * 78)
print("4.  INVERSION: what would G0 have to do?")
print("    If q_PAH's mass gradient does NOT evolve (the D1/D6 result -- metallicity")
print("    supply gives a near-constant gradient), then all of the swing must come")
print("    from the radiation term:  d log(G0/<U>)/d log M*  =  observed slope.")
need_G0 = s_obs + P * gw                       # d log G0 / d log M* required
row("required d log(G0/<U>)/dlogM*", s_obs, f"   change {s_obs[2]-s_obs[0]:+.3f}")
row("required d log G0 /dlogM*", need_G0, f"   change {need_G0[2]-need_G0[0]:+.3f}")

print("\n  Screen: a proxy X can only move the swing if its OWN mass gradient evolves.")
print(f"  {'proxy':<24}{'z~1':>9}{'z~2':>9}{'z~3':>9}   d(slope)   b needed")
from simstack4.dust_evolution import main_sequence_ssfr
SSp = grid(S["df_pool_sm"], "log_ssfr_measured", "wmean")
MS = np.array([[main_sequence_ssfr(ZC[j], MC[i]) for j in range(NZ)] for i in range(NB)])
need_change = need_G0[2] - need_G0[0]
for nm, M in (("log Sigma_SFR (branch-2)", None), ("log sSFR (measured)", SSp),
              ("log sSFR_MS(z,M*)", MS), ("log Delta_MS", SSp - MS),
              ("log T_dust", np.log10(Tw))):
    s = DSIG if M is None else slopes(M)
    ch = s[2] - s[0]
    b = need_change / ch if abs(ch) > 1e-6 else np.inf
    print(f"  {nm:<24}" + "".join(f"{x:+9.3f}" for x in s)
          + f"{ch:+11.3f}{b:+11.1f}")
print(f"""
  G0 ~ X^b needs b = {need_change:+.3f} / d(slope).  Sigma_SFR wants b = {need_change/(DSIG[2]-DSIG[0]):+.1f}
  (NEGATIVE -- G0 would have to fall as star formation concentrates); sSFR wants
  b = {need_change/(slopes(SSp)[2]-slopes(SSp)[0]):+.1f}.  Both are unphysical in sign or size.  No power-law
  G0 proxy built from the main sequence plus sizes can supply the requirement --
  the same D6 wall that killed every abundance arm, now reached from the radiation
  side.  G0 has to be MEASURED (the ionised/neutral band ratio), not modelled.""")
