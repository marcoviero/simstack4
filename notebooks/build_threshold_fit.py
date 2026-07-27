"""Two-term threshold model for the crossing: production + Sigma_SFR-triggered destruction."""
import os, sys, pickle, numpy as np
from scipy.optimize import least_squares
sys.path.insert(0, "/Users/mviero/Repositories/simstack4-2026/notebooks")
from simstack4.dust_evolution import main_sequence_ssfr
SP = os.environ["SP"]
d = pickle.load(open(SP + "/zr3.pkl", "rb"))
zr, zf = d["zr_pool"], d["zr_folds"]            # (4 mass, 3 z) and (3 folds, 4, 3)
err = np.nanstd(zf, axis=0, ddof=1) / np.sqrt(zf.shape[0])
MC = np.array([10.25, 10.70, 10.90, 11.25])     # mass-bin centres
ZC = np.array([0.95, 1.90, 2.95])               # z-slice centres
MPIV = 10.5

# --- Sigma_SFR(M*, z) on the main sequence (Speagle+14 x van der Wel+14) -----
def logSigSFR(logM, z):
    re = 8.9 * (10.0**logM / 5e10)**0.22 * (1.0 + z)**(-0.75)      # kpc
    sfr = 10.0**(main_sequence_ssfr(z, logM) + logM)               # Msun/yr
    return np.log10(0.5 * sfr / (np.pi * re**2))                   # Msun/yr/kpc^2

SIG = np.array([[logSigSFR(m, z) for z in ZC] for m in MC])
print("log Sigma_SFR [Msun/yr/kpc^2] on the grid:")
print(f"{'logM':>7}" + "".join(f"{'z='+str(z):>10}" for z in ZC))
for i, m in enumerate(MC):
    print(f"{m:>7.2f}" + "".join(f"{SIG[i,j]:>10.2f}" for j in range(3)))

y = np.log10(zr); sy = err / (zr * np.log(10))
ok = np.isfinite(y) & np.isfinite(sy) & (sy > 0)
print(f"\nusable cells: {ok.sum()} of {y.size}")

def model_thresh(p):
    a0, a1, dstr, sc = p
    return a0 + a1*(MC[:, None] - MPIV) - dstr*np.clip(SIG - sc, 0, None)
def model_prod(p):          # production only, no threshold
    a0, a1 = p
    return a0 + a1*(MC[:, None] - MPIV) + 0*SIG
def model_prod_z(p):        # production slope + free normalisation per z (equilibrium-like)
    a1, n1, n2, n3 = p
    return np.array([n1, n2, n3])[None, :] + a1*(MC[:, None] - MPIV)

def fit(mfun, p0, bounds, name, npar):
    r = least_squares(lambda p: ((mfun(p) - y)[ok] / sy[ok]), p0, bounds=bounds)
    chi2 = float(np.sum(r.fun**2)); dof = int(ok.sum()) - npar
    aic = chi2 + 2*npar
    print(f"  {name:<34} chi2={chi2:7.2f}  dof={dof:>2}  chi2red={chi2/max(dof,1):6.2f}  AIC={aic:7.2f}")
    return r, chi2, aic

print("\n--- model comparison (12 points) ---")
r_p, c_p, a_p = fit(model_prod, [-1.0, 0.0], ([-3, -2], [1, 2]), "production only (2 par)", 2)
r_pz, c_pz, a_pz = fit(model_prod_z, [0.0, -1, -1, -1],
                       ([-2, -3, -3, -3], [2, 1, 1, 1]), "production + free norm(z) (4 par)", 4)
r_t, c_t, a_t = fit(model_thresh, [-0.8, 0.4, 1.0, -0.5],
                    ([-3, -1, 0, -3], [1, 3, 10, 2]), "THRESHOLD: prod + destruction (4 par)", 4)
a0, a1, dstr, sc = r_t.x
print(f"\nTHRESHOLD best fit:")
print(f"  production slope a1   = {a1:+.3f} dex/dex   (mass slope with NO destruction)")
print(f"  destruction strength  = {dstr:.3f} dex per dex of Sigma_SFR above threshold")
print(f"  log Sigma_crit        = {sc:+.3f}  -> Sigma_crit = {10**sc:.2f} Msun/yr/kpc^2")
print(f"  normalisation a0      = {a0:+.3f}")
print(f"\n  Delta AIC vs production-only        = {a_t - a_p:+.1f}")
print(f"  Delta AIC vs free-norm(z) (same npar) = {a_t - a_pz:+.1f}")
print("\nrecovered vs measured mass slopes per z slice:")
mt = model_thresh(r_t.x)
for j, z in enumerate(ZC):
    sm = np.polyfit(MC[ok[:, j]], y[ok[:, j], j], 1)[0]
    st = np.polyfit(MC, mt[:, j], 1)[0]
    print(f"  z={z:.2f}:  measured {sm:+.3f}   threshold model {st:+.3f}")
print("\nwhich cells are ABOVE threshold (destruction active)?")
print(f"{'logM':>7}" + "".join(f"{'z='+str(z):>10}" for z in ZC))
for i, m in enumerate(MC):
    print(f"{m:>7.2f}" + "".join(f"{('YES' if SIG[i,j] > sc else '  -'):>10}" for j in range(3)))
pickle.dump(dict(p=r_t.x, SIG=SIG, MC=MC, ZC=ZC, y=y, sy=sy, ok=ok), open(SP+"/thresh_fit.pkl","wb"))
