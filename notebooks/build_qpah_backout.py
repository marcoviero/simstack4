"""Back out q_PAH using T_dust for <U> and sSFR for G0, and ask if the crossing survives."""
import os, pickle, numpy as np, pandas as pd
SP = os.environ["SP"]
S = pickle.load(open(SP + "/state2.pkl", "rb"))
Z = pickle.load(open(SP + "/zr3.pkl", "rb"))
zr = Z["zr_pool"]                       # (4 mass, 3 z) L_PAH/L_IR
MC = np.array([10.25, 10.70, 10.90, 11.25])
WINS = [(0.5,1.4,"z~1"), (1.4,2.4,"z~2"), (2.4,3.5,"z~3")]
d = S["df_pool_sm"]
print("columns available for the proxies:",
      [c for c in d.columns if "ssfr" in c.lower() or "T_dust" in c or "beta" in c])
# per (mass, z-window) medians of T_dust and measured log sSFR
T = np.full((4,3), np.nan); SS = np.full((4,3), np.nan); BE = np.full((4,3), np.nan)
for i in range(4):
    for j,(zlo,zhi,_) in enumerate(WINS):
        sub = d[(d.prop_bin_id==i) & (d.z_mid>=zlo) & (d.z_mid<zhi)]
        if not len(sub): continue
        T[i,j] = sub.T_dust.median(); BE[i,j] = sub.beta.median()
        SS[i,j] = sub.log_ssfr_measured.median()
print(f"\nT_dust [K] per (mass, z):")
for i in range(4): print(f"  logM={MC[i]:.2f}: " + "  ".join(f"{T[i,j]:6.2f}" for j in range(3)))
print(f"log sSFR per (mass, z):")
for i in range(4): print(f"  logM={MC[i]:.2f}: " + "  ".join(f"{SS[i,j]:6.2f}" for j in range(3)))
beta = np.nanmedian(BE); p = 4 + beta
print(f"\nbeta = {beta:.2f}  ->  <U> ~ T_dust^{p:.2f}")

def slopes(M):
    out = []
    for j in range(3):
        v = M[:,j]; ok = np.isfinite(v)
        out.append(np.polyfit(MC[ok], v[ok], 1)[0] if ok.sum()>=3 else np.nan)
    return np.array(out)

logLR = np.log10(zr)
s_obs = slopes(logLR)
print(f"\nOBSERVED  d log(L_PAH/L_IR)/d logM* : " + "  ".join(f"{s:+.3f}" for s in s_obs)
      + f"   swing {s_obs[2]-s_obs[0]:+.3f}")
print("\nq_PAH = (L_PAH/L_IR) * <U>/G0,  <U> ~ T^(4+beta),  G0 ~ sSFR^a")
print(f"{'a':>6}{'z~1':>10}{'z~2':>10}{'z~3':>10}{'swing':>10}   {'crossing?':>10}")
for a in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
    logq = logLR + p*np.log10(T) - a*SS
    s = slopes(logq)
    cross = "YES" if (s[0] > 0 and s[2] < 0) else "gone"
    print(f"{a:>6.2f}" + "".join(f"{x:+10.3f}" for x in s) + f"{s[2]-s[0]:+10.3f}   {cross:>10}")
print("\n(a=0 is the T_dust correction alone; the sSFR term is what varies)")
# how much does each correction move things on its own?
s_T = slopes(logLR + p*np.log10(T))
print(f"\nT_dust correction alone : " + "  ".join(f"{x:+.3f}" for x in s_T)
      + f"   swing {s_T[2]-s_T[0]:+.3f}")
s_s = slopes(logLR - 1.0*SS)
print(f"sSFR (a=1) alone        : " + "  ".join(f"{x:+.3f}" for x in s_s)
      + f"   swing {s_s[2]-s_s[0]:+.3f}")
