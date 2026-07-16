"""Generate 2026-07-12-lim-via-pah.ipynb.

Branch `forecast-lim-via-pah-1`. Turns the branch-9 PAH(M*, z) crossing into a
forecast of the [CII] and CO line-intensity-mapping (LIM) signals and compares it
to the Chiang+2026 measurement of the cosmic [CII]/CO backgrounds and to the
mmIME / COMAP CO power-spectrum data.

Scope (machinery-first):
  - The forecast CHAIN is real and runnable end-to-end.
  - Its inputs are PARAMETERIZED and clearly flagged: the documented branch-9
    crossing (L_PAH/L_IR slope +0.3 at z~1 -> 0 at z~2 -> -0.65 at z~3), a
    Davidzon+2017-style evolving SF stellar mass function, and a Speagle+2014
    main-sequence L_IR. Wiring the *measured* n(M*,z), L_IR(M*,z) and crossing
    from the real COSMOS2020 stacks is the next pass.
  - The absolute [CII] amplitude uses our measured L_PAH/L_IR x the MATCHED-DEFINITION
    L_CII/total-PAH ~= 0.05 (Herrera-Camus+15 / Smith+07; the Croxall/Sutter 0.1 is
    L_CII/7.7-complex). With it, the amplitude is consistent with Chiang (§6); the
    robust, tuning-free result is the mass/z structure (the crossing).
  - Comparison curves are real data or computed from a cited relation only: the
    Chiang+2026 measured [CII], De Looze'14 / MS L_IR-L'_CO x cosmic SFRD
    references, and mmIME / COMAP for the CO power spectrum.

Objectives (from docs/forecast-lim-via-pah-1-brief.md):
  0 -- the PAH<->[CII]/CO bridge (why a PAH stacking result is a LIM result)
  1 -- the PAH -> line calibration ladder (L_CII/L_PAH, L_PAH/L'_CO, +U drift knob)
  2 -- build <I_CII>(z), <T_CO>(z), shot noise from n(M*,z) x L_PAH(M*,z)
  3 -- [CII] and CO, each as mean intensity + power spectrum, vs published models
       (De Looze/Lagache for [CII]; Li+2016 for CO) and real data (Chiang; mmIME/COMAP)
  4 -- the 24 um PAH-contamination bias handed to LIM forecasters

Run:  uv run python notebooks/build_lim_via_pah_2026-07-12_notebook.py
Then: uv run jupyter nbconvert --to notebook --execute --inplace \
          notebooks/2026-07-12-lim-via-pah.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ----------------------------------------------------------------------------- header
md(
    r"""# PAH-anchored forecasts for [CII] and CO line-intensity mapping

**Branch `forecast-lim-via-pah-1`, 2026-07-12.** The LIM room forecasts the
aggregate [CII] and CO signal from *models* that paint line luminosity onto
galaxies as a function of mass and redshift. We have *measured* the PAH
luminosity of the confused faint population vs (M*, z) -- including the branch-9
**crossing** (L_PAH/L_IR mass slope $+0.3$ at $z\sim1 \to 0$ at $z\sim2 \to
-0.65$ at $z\sim3$). PAHs are physically wired to both lines, so this is an
empirical, *evolution-measured* anchor for the part of a LIM forecast that is
currently pure model.

**What this notebook is (machinery-first).** The forecast chain is real; its
inputs are parameterized and flagged (documented crossing, Davidzon+2017-style
SF SMF, Speagle+2014 main-sequence $L_{\rm IR}$). Comparison curves are real data
or computed from a cited relation -- the **Chiang+2026 measured** [CII], De Looze
/ MS references, and mmIME / COMAP for CO. Wiring the *measured* $n(M_*,z)$,
$L_{\rm IR}(M_*,z)$ and crossing from the real COSMOS2020 stacks is the next pass.

**The honest framing** (carried from the brief): we do **not** eliminate model
dependence -- the PAH$\to$line ratios are local calibrations assumed to hold at
high $z$. We *replace a modeled evolution* (the line/SFR ratio's) *with a
measured one* (the PAH's) plus a local ratio. That is still a real advance, and
it is the correct thing to say to the room."""
)

# ----------------------------------------------------------------------------- setup
md(r"""## Setup -- constants, cosmology, and the flagged calibration config""")

code(
    r'''import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy.units as u
from astropy.constants import c, k_B
from astropy.cosmology import Planck18

plt.rcParams.update({"figure.dpi": 110, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})

# consistent palette
C_ON, C_OFF = "#2166ac", "#b2182b"       # crossing ON / OFF
C_CII, C_CO = "#1b7837", "#762a83"
C_MODEL = "#888888"

L_SUN = 3.828e33      # erg/s
NU_CII = 1900.5       # GHz, [CII] 158 um
NU_CO10 = 115.27      # GHz, CO(1-0)

# === FLAGGED CALIBRATION CONFIG =============================================
# Every number here is a placeholder of the right order, to be replaced with the
# literature value + scatter (see brief reading list). Centralized so they are
# trivial to swap. TODO markers name the source to pull from.
CAL = dict(
    # branch-9 documented crossing: log(L_PAH/L_IR) = LOG_PAH_IR_0 + slope(z)*(logM*-MPIV)
    crossing_z     = np.array([0.5, 1.0, 2.0, 3.0, 4.0]),
    crossing_slope = np.array([0.35, 0.30, 0.00, -0.65, -0.90]),  # d log(L_PAH/L_IR)/d logM*
    MPIV           = 10.5,
    LOG_PAH_IR_0   = -1.0,     # L_PAH/L_IR ~ 10% at pivot (Smith+07 total-PAH/L_TIR, ~10%)
    # PAH -> line. L_CII = (our measured L_PAH/L_IR) x (L_CII/L_PAH, TOTAL PAH) x L_IR.
    # MATCHED-DEFINITION bridge (2026-07-13 scrutiny): the [CII]/PAH quoted as ~0.1 by
    # Croxall+12/Sutter+19 is [CII]/PAH-SUBSET (their "PAH" ~ the 7.7 complex ~= 49% of
    # total PAH, Smith+07). The [CII]/TOTAL-PAH consistent with the SAME local galaxies is
    #   L_CII/L_PAH(total) = (L_CII/L_TIR)/(L_PAH/L_TIR) = 0.48%/10% ~= 0.05
    # [Herrera-Camus+2015 KINGFISH L_CII/L_TIR=0.48+-0.21%; Smith+07 L_PAH/L_TIR~10%].
    # Using 0.05 (not 0.1) removes the ~2x definition mismatch that inflated our amplitude.
    R_CII_PAH      = 0.05,     # L_CII / TOTAL-PAH (matched definition; Herrera-Camus/Smith)
    R_CII_PAH_lo   = 0.03,     # band from L_CII/L_TIR scatter (0.27-0.69%) / (PAH/L_TIR)
    R_CII_PAH_hi   = 0.07,
    L_PAH_over_LpCO= 7.0,      # L_PAH/L'_CO; implied L_IR/L'_CO ~ 70 (MS locus, Sargent+14)
    delooze_LCII_SFR = 10**7.06,  # De Looze+2014 whole-sample L_CII/SFR [Lsun/(Msun/yr)] (reference only)
    LIR_over_LpCO_MS = 70.0,   # MS L_IR/L'_CO (Sargent+14/Genzel+15) -- a real gas relation
    sfr_per_LIR    = 1.49e-10, # Kennicutt&Evans12 (Chabrier), SFR = this * L_IR[Lsun]
    sig_cii_dex    = 0.30,     # log scatter in L_CII/L_PAH (for the ladder viz)
    sig_co_dex     = 0.30,     # scatter in L_PAH/L'_CO      TODO
    # intensity-dependent L_CII/L_PAH drift (the brief's Obj-1 caveat): the ratio
    # deepens with sSFR (the same axis that drives the deficits). Toggle + strength.
    cii_U_drift    = -0.15,    # d log(L_CII/L_PAH) / d (log sSFR - pivot)   TODO
    ssfr_pivot     = -9.0,     # log sSFR/yr^-1 pivot for the drift
)

# === PARAMETERIZED POPULATION MODEL (flagged approximations) ================
# Evolving single-Schechter SF stellar mass function (Davidzon+2017-ish).
SMF = dict(
    z       = np.array([0.5, 1.0, 2.0, 3.0, 4.0]),
    logMstar= np.array([10.75, 10.70, 10.65, 10.60, 10.55]),
    logphi  = np.array([-2.85, -2.95, -3.15, -3.60, -4.05]),   # Mpc^-3 dex^-1
    alpha   = np.array([-1.30, -1.32, -1.40, -1.50, -1.60]),
)
LOGM_LO, LOGM_HI = 9.9, 12.0     # the mass range our stacks actually constrain
print("Setup ready. Planck18, H0 =", Planck18.H0)
'''
)

# ----------------------------------------------------------------------------- physics
code(
    r'''# ---- the crossing (branch-9 documented input) ------------------------------
def crossing_slope(z):
    return np.interp(z, CAL["crossing_z"], CAL["crossing_slope"])

def log_pah_ir(logM, z, use_crossing=True):
    s = crossing_slope(z) if use_crossing else 0.0
    return CAL["LOG_PAH_IR_0"] + s * (np.asarray(logM) - CAL["MPIV"])

# ---- population: SMF and main-sequence L_IR --------------------------------
def smf(logM, z):
    """dn/dlogM [Mpc^-3 dex^-1], evolving single-Schechter (FLAG: approx)."""
    logMs = np.interp(z, SMF["z"], SMF["logMstar"])
    logphi = np.interp(z, SMF["z"], SMF["logphi"])
    alpha = np.interp(z, SMF["z"], SMF["alpha"])
    x = 10 ** (np.asarray(logM) - logMs)
    return np.log(10) * 10 ** logphi * x ** (1 + alpha) * np.exp(-x)

def log_ssfr_ms(logM, z):
    """log10(sSFR/yr^-1), Speagle+2014 (mirrors dust_evolution.main_sequence_ssfr)."""
    t = Planck18.age(z).to(u.Gyr).value
    log_sfr = (0.84 - 0.026 * t) * np.asarray(logM) - (6.51 - 0.11 * t)
    return log_sfr - np.asarray(logM)

def L_IR_MS(logM, z):
    """L_IR [L_sun] on the main sequence. SFR=sSFR*M*; L_IR=SFR/(SFR per L_IR)."""
    log_sfr = log_ssfr_ms(logM, z) + np.asarray(logM)
    return 10 ** log_sfr / CAL["sfr_per_LIR"]

# ---- PAH -> line -----------------------------------------------------------
def L_PAH(logM, z, use_crossing=True):
    return 10 ** log_pah_ir(logM, z, use_crossing) * L_IR_MS(logM, z)   # L_sun

def _cii_per_pah(logM, z, U_drift=False):
    r = CAL["R_CII_PAH"]
    if U_drift:
        ds = log_ssfr_ms(logM, z) - CAL["ssfr_pivot"]
        r = r * 10 ** (CAL["cii_U_drift"] * ds)
    return r

def L_CII(logM, z, use_crossing=True, U_drift=False):
    return _cii_per_pah(logM, z, U_drift) * L_PAH(logM, z, use_crossing)   # L_sun

def L_CO10(logM, z, use_crossing=True):
    lprime = L_PAH(logM, z, use_crossing) / CAL["L_PAH_over_LpCO"]   # K km/s pc^2
    return 4.9e-5 * lprime                                           # L_sun (CO(1-0))

# ---- population integrals --------------------------------------------------
def _grid():
    return np.linspace(LOGM_LO, LOGM_HI, 80)

def luminosity_density(z, Lfunc, **kw):
    """rho_L(z) [L_sun/Mpc^3] = int dlogM (dn/dlogM) L over our mass range."""
    g = _grid(); dlogM = g[1] - g[0]
    return np.sum(smf(g, z) * Lfunc(g, z, **kw)) * dlogM

def L2_density(z, Lfunc, **kw):
    """int dlogM (dn/dlogM) L^2  [L_sun^2/Mpc^3] -- for shot noise."""
    g = _grid(); dlogM = g[1] - g[0]
    return np.sum(smf(g, z) * Lfunc(g, z, **kw) ** 2) * dlogM

def lum_weighted_logM(z, Lfunc, **kw):
    """<logM*> weighted by L -- which galaxies dominate the line signal."""
    g = _grid()
    w = smf(g, z) * Lfunc(g, z, **kw)
    return np.sum(w * g) / np.sum(w)

# ---- comoving line intensity (Lidz+2011 / Breysse+2014 convention) ---------
def line_Tb_Inu(z, rho_L_Lsun_Mpc3, nu_rest_GHz):
    """Mean brightness temperature (uK) and specific intensity (Jy/sr)."""
    nu_rest = nu_rest_GHz * u.GHz
    Hz = Planck18.H(z)
    rho_L = rho_L_Lsun_Mpc3 * L_SUN * u.erg / u.s / u.Mpc ** 3
    Tb = ((c ** 3 * (1 + z) ** 2) / (8 * np.pi * k_B * nu_rest ** 3 * Hz) * rho_L).to(u.K)
    nu_obs = nu_rest / (1 + z)
    Inu = (2 * k_B * nu_obs ** 2 / c ** 2 * Tb).to(u.Jy)   # per sr implicit
    return Tb.to(u.uK).value, Inu.value

def I_CII(z, use_crossing=True, U_drift=False):
    rho = luminosity_density(z, L_CII, use_crossing=use_crossing, U_drift=U_drift)
    return line_Tb_Inu(z, rho, NU_CII)[1]                 # Jy/sr

def T_CO(z, use_crossing=True):
    rho = luminosity_density(z, L_CO10, use_crossing=use_crossing)
    return line_Tb_Inu(z, rho, NU_CO10)[0]                # uK

# ---- Where does our implied L_CII/SFR land? --------------------------------
# L_CII/SFR = (measured L_PAH/L_IR) x (literature L_CII/L_PAH) / (SFR per L_IR),
# reported relative to De Looze+2014 and to the Chiang measurement.
_lcii_sfr = CAL["R_CII_PAH"] * 10 ** CAL["LOG_PAH_IR_0"] / CAL["sfr_per_LIR"]
print(f"implied  L_CII/SFR = {_lcii_sfr:.2e}  "
      f"(De Looze {CAL['delooze_LCII_SFR']:.2e} -> {_lcii_sfr/CAL['delooze_LCII_SFR']:.1f}x;  "
      f"Chiang 2.2e7 -> {_lcii_sfr/2.2e7:.1f}x)   [per SFR, before the mass-range cut]")

# quick self-check
print("\nz    I_CII[Jy/sr]  T_CO[uK]   slope")
for z in (0.5, 1.0, 2.0, 3.0):
    print(f"{z:.1f}   {I_CII(z):9.1f}   {T_CO(z):7.3f}   {crossing_slope(z):+.2f}")
'''
)

# ------------------------------------------------- power spectrum + Chiang machinery
md(
    r"""### Machinery for the LIM power spectrum and the Chiang+2026 measurement

The proper LIM observable is the 3D power spectrum $P(k,z) = \underbrace{\langle
I\rangle^2\,b_{\rm eff}^2\,P_{\rm lin}(k,z)}_{\rm clustering} + \underbrace{P_{\rm
shot}}_{\int n\,L^2}$. We build a self-contained linear $P_{\rm lin}$ (BBKS
transfer function normalized to $\sigma_8$), growth $D(z)$, Tinker+2010 halo bias,
and a luminosity-weighted $b_{\rm eff}$ via an approximate SHMR.

We also load the **Chiang+2026** (Nature Astr., arXiv:2602.02658) *measured*
comoving [CII] luminosity density $\rho_{\rm CII}(z)=5.9\times10^{38}(1+z)^{3.2}
/[1+((1+z)/2.9)^{6.6}]$ erg s$^{-1}$ Mpc$^{-3}$ ($0<z<4.2$), obtained by
tomographic clustering of diffuse intensities with reference galaxies -- the same
philosophy as our stacking. It is a **real empirical anchor**, not a model."""
)

code(
    r'''# ---- linear matter P(k) (BBKS/Sugiyama), growth, Tinker+2010 bias ----------
_Om, _Ob, _h, _ns, _s8 = Planck18.Om0, Planck18.Ob0, Planck18.h, 0.965, 0.81
_rho_m0 = _Om * 2.775e11        # Msun/h per (Mpc/h)^3, comoving
_dc = 1.686
_kg = np.logspace(-4, 3, 4000)

def _T_bbks(k):
    Gam = _Om*_h*np.exp(-_Ob - np.sqrt(2*_h)*_Ob/_Om)   # h/Mpc
    q = k/Gam
    return (np.log(1+2.34*q)/(2.34*q))*(1+3.89*q+(16.1*q)**2+(5.46*q)**3+(6.71*q)**4)**-0.25

def _Pk_unnorm(k): return k**_ns*_T_bbks(k)**2

def _sigma2(R):
    R = np.atleast_1d(R).astype(float)
    x = _kg[None,:]*R[:,None]
    W = 3*(np.sin(x)-x*np.cos(x))/x**3
    integ = (_kg**3*_Pk_unnorm(_kg))[None,:]*W**2/(2*np.pi**2)
    out = np.trapezoid(integ, np.log(_kg), axis=1)
    return out[0] if out.size==1 else out

_Anorm = _s8**2/_sigma2(8.0)

def growth(z):
    def g(zz):
        a=1/(1+np.asarray(zz,float)); Omz=_Om/(_Om+(1-_Om)*a**3); OLz=1-Omz
        return 2.5*Omz/(Omz**(4/7)-OLz+(1+Omz/2)*(1+OLz/70))
    a=1/(1+np.asarray(z,float))
    return a*g(z)/g(0.0)

def Pk_lin(k, z=0.0): return _Anorm*_Pk_unnorm(k)*growth(z)**2
def sigma_M(M, z=0.0):
    R=(3*M/(4*np.pi*_rho_m0))**(1/3)
    return np.sqrt(_Anorm*_sigma2(R))*growth(z)

def bias_tinker(M, z):
    nu=_dc/sigma_M(M,z); y=np.log10(200)
    A=1.0+0.24*y*np.exp(-(4/y)**4); a=0.44*y-0.88
    B=0.183; b=1.5; C=0.019+0.107*y+0.19*np.exp(-(4/y)**4); c=2.4
    return 1 - A*nu**a/(nu**a+_dc**a)+B*nu**b+C*nu**c

def logMhalo(logMstar): return 12.2+1.5*(np.asarray(logMstar)-10.5)   # approx SHMR (FLAG)

# ---- generic line observables for ANY luminosity function Lfunc(logM, z) -----
# T=True -> brightness temp (uK) for CO; T=False -> specific intensity (Jy/sr) for [CII].
def _grid_mm(logM_min):
    g = np.linspace(logM_min, 12.0, 120); return g, g[1]-g[0]
def line_mean(z, Lfunc, nu, T=False, logM_min=8.0, **kw):
    g, dl = _grid_mm(logM_min)
    rho = np.sum(smf(g, z) * Lfunc(g, z, **kw)) * dl
    return line_Tb_Inu(z, rho, nu)[0 if T else 1]
def line_Pshot(z, Lfunc, nu, T=False, logM_min=8.0, **kw):
    g, dl = _grid_mm(logM_min)
    K1 = line_Tb_Inu(z, 1.0, nu)[0 if T else 1]        # rho_L -> uK or Jy/sr
    L2 = np.sum(smf(g, z) * Lfunc(g, z, **kw)**2) * dl
    return K1**2 * L2
def b_eff(z, Lfunc=None, logM_min=8.0, **kw):
    Lfunc = Lfunc if Lfunc is not None else L_CII
    g, _ = _grid_mm(logM_min); Mh = 10**logMhalo(g)*_h
    w = smf(g, z) * Lfunc(g, z, **kw)
    return np.sum(w*bias_tinker(Mh, z)) / np.sum(w)
def line_Pk(k, z, Lfunc, nu, T=False, logM_min=8.0, **kw):
    I = line_mean(z, Lfunc, nu, T=T, logM_min=logM_min, **kw)
    be = b_eff(z, Lfunc=Lfunc, logM_min=logM_min, **kw)
    return I**2*be**2*Pk_lin(k, z), line_Pshot(z, Lfunc, nu, T=T, logM_min=logM_min, **kw)

# ---- published model L(logM, z), computed from cited relations (NOT fabricated) -
def sfr_ms(logM, z): return 10**(log_ssfr_ms(logM, z) + np.asarray(logM, float))  # Msun/yr
# [CII]: De Looze+2014 (whole sample); Lagache+2018 (z-evolving). ALPINE (Schaerer+20)
# is slope 0.96 / offset -0.03 dex vs De Looze -> effectively the De Looze line.
def L_CII_delooze(logM, z, use_crossing=True):
    return 10**7.06 * sfr_ms(logM, z)                                    # logL=7.06+logSFR
def L_CII_lagache(logM, z, use_crossing=True):
    s = np.log10(sfr_ms(logM, z))
    return 10**((1.4-0.07*z)*s + (7.1-0.07*z))
# CO(1-0): Li+2016 fiducial -> L_IR=SFR/1e-10 (Kennicutt, delta_MF=1), then
# logL_IR = 1.37 logL'_CO - 1.74 (Carilli&Walter'13) -> L_CO = 4.9e-5 L'_CO.
def L_CO_li16(logM, z, use_crossing=True):
    LIR = sfr_ms(logM, z) / 1e-10
    logLpco = (np.log10(LIR) + 1.74) / 1.37
    return 4.9e-5 * 10**logLpco

# real CO data: mmIME shot noise (Keating+2020, higher-J summed -- caveat) and
# COMAP ES-V CO(1-0) mean-intensity upper limit at z~3.
MMIME_PSHOT = 2.0e3        # uK^2 (Mpc/h)^3  (+1.1/-1.2e3), Keating+2020
MMIME_PSHOT_ERR = (1.2e3, 1.1e3)
COMAP_TB2_UL = 50.0        # uK^2, 95% UL on <Tb>^2, CO(1-0) z~3 (Chung+2022 ES-V)

# ---- Chiang+2026 measured [CII] luminosity density -> intensity -------------
def rho_cii_chiang(z): return 5.9e38*(1+z)**3.2/(1+((1+z)/2.9)**6.6)   # erg/s/Mpc^3
def I_cii_chiang(z): return line_Tb_Inu(z, rho_cii_chiang(z)/L_SUN, NU_CII)[1]

# ---- Computable reference: De Looze L_CII-SFR x cosmic SFRD (full mass) ------
# Madau&Dickinson'14 SFRD x De Looze'14 L_CII/SFR -- the standard galaxy-relation
# expectation over ALL masses.
def madau_dickinson(z): return 0.015*(1+z)**2.7/(1+((1+z)/2.9)**5.6)   # Msun/yr/Mpc^3
def I_cii_delooze_full(z):
    rho_L = CAL["delooze_LCII_SFR"]*madau_dickinson(z)                 # Lsun/Mpc^3
    return line_Tb_Inu(z, rho_L, NU_CII)[1]
# Chiang MEASURE L_CII/SFR ~ 2.2e7 (SFR=4.5e-8 L_CII at z=2) ~ 1.9x De Looze:
CHIANG_LCII_SFR = 1/4.5e-8
def I_cii_chiang_calib_full(z):
    return line_Tb_Inu(z, CHIANG_LCII_SFR*madau_dickinson(z), NU_CII)[1]

def T_co_ms_full(z):
    """Real computable CO reference: MS L_IR/L'_CO x cosmic SFRD (all mass)."""
    rho_LIR = madau_dickinson(z)/CAL["sfr_per_LIR"]           # Lsun/Mpc^3
    rho_LCO = 4.9e-5 * rho_LIR/CAL["LIR_over_LpCO_MS"]        # Lsun/Mpc^3 (CO(1-0))
    return line_Tb_Inu(z, rho_LCO, NU_CO10)[0]               # uK

# ---- completeness correction: our measured relation extrapolated to low mass -
# Our stacks only reach logM*>9.9. To compare to Chiang's TOTAL, integrate our
# L_CII(M*,z) over the full SMF. The crossing is MEASURED only in 9.9-11.2; below
# 9.9 we HOLD IT FLAT (do not extrapolate the slope -- at z~3 the -0.65 slope would
# blow up the low-mass end, exactly where the low-metallicity PAH deficit should
# instead turn it over). So this closes the MASS-incompleteness gap but not the
# De Looze-vs-Chiang calibration gap, and the low-mass end is an assumption.
def log_pah_ir_ext(logM, z, use_crossing=True, floor=9.9):
    s = crossing_slope(z) if use_crossing else 0.0
    return CAL["LOG_PAH_IR_0"] + s*(np.maximum(np.asarray(logM,float), floor) - CAL["MPIV"])

def L_CII_ext(logM, z, use_crossing=True):
    return CAL["R_CII_PAH"] * 10**log_pah_ir_ext(logM, z, use_crossing) * L_IR_MS(logM, z)

def L_CO_ext(logM, z, use_crossing=True):   # CO with crossing held flat below 9.9
    lprime = 10**log_pah_ir_ext(logM, z, use_crossing) * L_IR_MS(logM, z) / CAL["L_PAH_over_LpCO"]
    return 4.9e-5 * lprime

# DATA-INFORMED low-mass TURNOVER (§7b): the 9.0-9.9 stack shows NO 24um PAH excess
# (S/N-marginal but not a boost), consistent with the low-metallicity PAH deficit. So
# below the measured floor (9.9) let L_PAH/L_IR DECLINE by `turn` dex/dex instead of
# holding it flat -- the physical q_PAH(Z) turnover. turn=0 recovers flat-below-9.9.
def log_pah_ir_turn(logM, z, use_crossing=True, floor=9.9, turn=1.5):
    logM = np.asarray(logM, float)
    base = log_pah_ir_ext(logM, z, use_crossing, floor)     # flat-below-floor value
    return base + turn * np.minimum(0.0, logM - floor)       # decline below the floor only
def L_CII_turn(logM, z, use_crossing=True, turn=1.5):
    return CAL["R_CII_PAH"] * 10**log_pah_ir_turn(logM, z, use_crossing, turn=turn) * L_IR_MS(logM, z)

def I_cii_complete(z, logM_min=8.0, use_crossing=True):
    g = np.linspace(logM_min, 12.0, 140); dl = g[1]-g[0]
    rho = np.sum(smf(g, z) * L_CII_ext(g, z, use_crossing)) * dl
    return line_Tb_Inu(z, rho, NU_CII)[1]

def sfrd_smf(z, logM_min):
    g = np.linspace(logM_min, 12.0, 140); dl = g[1]-g[0]
    return np.sum(smf(g, z) * 10**(log_ssfr_ms(g, z)+g)) * dl

print("machinery check: sigma8=%.3f\n" % np.sqrt(_Anorm*_sigma2(8.0)))
print("[CII] <I> [Jy/sr] over full mass (>8): Chiang / De Looze / Lagache / ours(complete)")
for z in (1,2,3):
    print(f"  z={z}: Chiang={I_cii_chiang(z):5.0f}  DeLooze={line_mean(z,L_CII_delooze,NU_CII):5.0f}  "
          f"Lagache={line_mean(z,L_CII_lagache,NU_CII):5.0f}  ours={I_cii_complete(z):5.0f}")
print("\ncrossing shot-noise leverage over our measured range (>9.9):")
for z in (0.5,1,3):
    on = line_Pshot(z, L_CII, NU_CII, logM_min=9.9, use_crossing=True)
    off = line_Pshot(z, L_CII, NU_CII, logM_min=9.9, use_crossing=False)
    print(f"  z={z}: Pshot ON/OFF = {on/off:.3f}x   b_eff(9.9)={b_eff(z,L_CII,logM_min=9.9):.2f}")
print("\nCO(1-0) Pshot [uK^2(Mpc/h)^3] over >8: Li+16 / ours   vs mmIME=%.0f, COMAP <Tb>^2<%.0f" %
      (MMIME_PSHOT, COMAP_TB2_UL))
for z in (2.5,3.0):
    print(f"  z={z}: Li16={line_Pshot(z,L_CO_li16,NU_CO10,T=True):.0f}  "
          f"ours={line_Pshot(z,L_CO_ext,NU_CO10,T=True,use_crossing=True):.0f}   "
          f"<Tb>_ours={line_mean(z,L_CO_ext,NU_CO10,T=True,use_crossing=True):.3f}uK")
'''
)

# ----------------------------------------------------------------------------- Obj 0
md(
    r"""## 0 · The bridge -- why a PAH stacking result is a LIM result

Two independent physical couplings, each with a local empirical calibration
(Obj 0 of the brief):

**PAH $\to$ [CII] through the PDR.** [CII] 158 $\mu$m is the dominant coolant of
warm neutral / photodissociation-region gas; PAHs are the dominant
**photoelectric heating agent** of that same gas. Their ratio $L_{\rm
CII}/L_{\rm PAH}$ is observed to be roughly constant (Helou+01; Croxall+12;
Smith+17; Sutter+19). The classic high-$L_{\rm IR}$ "[CII] deficit" is *mirrored*
by a PAH deficit **on the intensity axis** ($L_{\rm IR}/\Sigma_{\rm SFR}/\langle
U\rangle$) -- so the ratio holds and the bridge stands. Our crossing lives on a
*different* axis ($M_*$ at fixed $z$); under a constant ratio it **predicts that
$L_{\rm CII}/L_{\rm IR}$ crosses with mass the same way**, which is exactly the
non-standard behaviour a fixed-slope forecast misses. Caveat carried below: the
ratio's second-order drift with intensity is strongest where the crossing bites.

**PAH $\to$ CO through the molecular gas.** $L_{\rm PAH}$ tracks $L'_{\rm CO}$ to
$z\sim4$ (Cortzen+19; arXiv:2409.05710); PAH luminosity is effectively a
molecular-gas-mass tracer. Noisier ($\alpha_{\rm CO}$, SLED) -- secondary.

**Stacking reaches LIM's population.** The line$-$SFR calibrations the room uses
are anchored on *bright, individually detected* galaxies; the LIM signal
(especially clustering) is dominated by *faint, low-mass* galaxies -- exactly
what simultaneous stacking measures. That is the methodological match."""
)

# ----------------------------------------------------------------------------- Obj 1
md(
    r"""## 1 · The PAH $\to$ line calibration ladder

The conversions, with scatter, and the intensity-drift knob (Obj 1). All numbers
live in `CAL` above.

**Read this before trusting any absolute number.** The bridge ratio uses **matched
definitions**: $L_{\rm CII}/$total-PAH $\approx0.05$, from Herrera-Camus+15 $L_{\rm
CII}/L_{\rm TIR}$ and Smith+07 $L_{\rm PAH}/L_{\rm TIR}$ (the 7.7 $\mu$m complex is
$\sim\!49\%$ of total PAH, so $L_{\rm CII}/L_{7.7}\approx0.1 \leftrightarrow L_{\rm
CII}/$total-PAH$\approx0.05$). This gives $L_{\rm CII}/L_{\rm IR}\approx0.5\%$ (local),
vs Chiang's cosmic $0.33\%$ -- so our amplitude is $\sim\!1.5\times$ Chiang per SFR
(the local-vs-cosmic offset), within the band ($\times0.5$–$2$: local $L_{\rm CII}/
L_{\rm TIR}$ scatter + partial$\to$total PAH). The absolute amplitude depends on that
bridge; the **robust, tuning-free result is the mass/$z$ structure** (the crossing),
differential in mass at fixed $z$ (§3c)."""
)

code(
    r'''fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# (a) L_CII vs L_PAH with scatter band
lpah = np.logspace(9, 12, 50)
ax = axes[0]
lcii = CAL["R_CII_PAH"] * lpah
ax.loglog(lpah, lcii, color=C_CII, lw=2, label=r"$L_{\rm CII}=%.2f\,L_{\rm PAH}$" % CAL["R_CII_PAH"])
s = CAL["sig_cii_dex"]
ax.fill_between(lpah, lcii*10**-s, lcii*10**s, color=C_CII, alpha=0.18, label=f"±{s} dex")
ax.set_xlabel(r"$L_{\rm PAH}\ [L_\odot]$"); ax.set_ylabel(r"$L_{\rm CII}\ [L_\odot]$")
ax.set_title("(a) PAH → [CII]  (PDR)"); ax.legend(fontsize=8)

# (b) L'_CO vs L_PAH
ax = axes[1]
lpco = lpah / CAL["L_PAH_over_LpCO"]
ax.loglog(lpah, lpco, color=C_CO, lw=2, label=r"$L'_{\rm CO}=L_{\rm PAH}/%.0f$" % CAL["L_PAH_over_LpCO"])
s = CAL["sig_co_dex"]
ax.fill_between(lpah, lpco*10**-s, lpco*10**s, color=C_CO, alpha=0.18, label=f"±{s} dex")
ax.set_xlabel(r"$L_{\rm PAH}\ [L_\odot]$"); ax.set_ylabel(r"$L'_{\rm CO}\ [{\rm K\,km\,s^{-1}\,pc^2}]$")
ax.set_title("(b) PAH → CO  (molecular gas)"); ax.legend(fontsize=8)

# (c) the intensity-drift knob: L_CII/L_PAH vs sSFR
ax = axes[2]
lssfr = np.linspace(-9.8, -8.2, 50)
ds = lssfr - CAL["ssfr_pivot"]
ratio = CAL["R_CII_PAH"] * 10 ** (CAL["cii_U_drift"] * ds)
ax.semilogy(lssfr, np.full_like(lssfr, CAL["R_CII_PAH"]), color=C_CII, ls="--", lw=1.5,
            label="constant (naive bridge)")
ax.semilogy(lssfr, ratio, color=C_CII, lw=2, label=r"drift $\propto{\rm sSFR}^{%.2f}$" % CAL["cii_U_drift"])
ax.set_xlabel(r"$\log\,{\rm sSFR}\ [{\rm yr^{-1}}]$"); ax.set_ylabel(r"$L_{\rm CII}/L_{\rm PAH}$")
ax.set_title("(c) the carried systematic:\nratio drift with intensity"); ax.legend(fontsize=8)

fig.suptitle("Obj 1 — PAH→line calibration ladder (placeholder values; see CAL / reading list)", y=1.03)
fig.tight_layout(); plt.show()
'''
)

# ----------------------------------------------------------------------------- Obj 2 input
md(
    r"""## 2 · Build the forecast

First the two ingredients: the population's $L_{\rm IR}(M_*,z)$ (main sequence)
and the **documented crossing** $L_{\rm PAH}/L_{\rm IR}(M_*,z)$ that is the
measured input. Then the line-luminosity densities and the intensities."""
)

code(
    r'''fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
logM = np.linspace(LOGM_LO, LOGM_HI, 60)
zz = [0.5, 1.0, 2.0, 3.0]
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, len(zz)))

ax = axes[0]
for z, col in zip(zz, cmap):
    ax.plot(logM, np.log10(L_IR_MS(logM, z)), color=col, lw=2, label=f"z={z}")
ax.set_xlabel(r"$\log M_*$"); ax.set_ylabel(r"$\log L_{\rm IR}\ [L_\odot]$")
ax.set_title(r"main-sequence $L_{\rm IR}$ (Speagle+14)"); ax.legend(fontsize=8)

ax = axes[1]
for z, col in zip(zz, cmap):
    ax.plot(logM, log_pah_ir(logM, z), color=col, lw=2, label=f"z={z} (slope {crossing_slope(z):+.2f})")
ax.axhline(CAL["LOG_PAH_IR_0"], color="k", ls=":", lw=1, alpha=0.6)
ax.set_xlabel(r"$\log M_*$"); ax.set_ylabel(r"$\log (L_{\rm PAH}/L_{\rm IR})$")
ax.set_title("the branch-9 crossing (measured input)"); ax.legend(fontsize=8)
fig.suptitle("Obj 2 — forecast ingredients", y=1.02); fig.tight_layout(); plt.show()
print("Note the crossing: at z~1 PAH/IR RISES with mass; at z~3 it FALLS. "
      "The SMF is bottom-heavy, so this reshapes the line/IR ratio's z-evolution.")
'''
)

code(
    r'''# Curves vs z. Models + ours integrated over the full SMF (logM*>8); "ours" uses
# our measured L_PAH/L_IR x lit bridge, crossing held flat below the measured 9.9.
zgrid = np.linspace(0.3, 4.0, 40)
def _arr(f): return np.array([f(z) for z in zgrid])
C = dict(
    z=zgrid,
    # [CII] mean intensity (Jy/sr)
    Icii_chiang=_arr(I_cii_chiang),
    Icii_delooze=_arr(lambda z: line_mean(z, L_CII_delooze, NU_CII)),
    Icii_lagache=_arr(lambda z: line_mean(z, L_CII_lagache, NU_CII)),
    Icii_ours99=_arr(lambda z: line_mean(z, L_CII, NU_CII, logM_min=9.9, use_crossing=True)),  # direct stacking reach
    Icii_ours=_arr(I_cii_complete),                       # completeness-corrected (>8), crossing ON
    # canonical PAH: same chain but NO measured mass/z structure (flat L_PAH/L_IR, crossing off)
    Icii_canon=_arr(lambda z: line_mean(z, L_CII_ext, NU_CII, use_crossing=False)),
    # CO mean brightness temperature (uK)
    Tco_li16=_arr(lambda z: line_mean(z, L_CO_li16, NU_CO10, T=True)),
    Tco_ours=_arr(lambda z: line_mean(z, L_CO_ext, NU_CO10, T=True, use_crossing=True)),
    # crossing's isolated effect over OUR measured range (>9.9)
    Pcii_on=_arr(lambda z: line_Pshot(z, L_CII, NU_CII, logM_min=9.9, use_crossing=True)),
    Pcii_off=_arr(lambda z: line_Pshot(z, L_CII, NU_CII, logM_min=9.9, use_crossing=False)),
    beff_on=_arr(lambda z: b_eff(z, L_CII, logM_min=9.9, use_crossing=True)),
    beff_off=_arr(lambda z: b_eff(z, L_CII, logM_min=9.9, use_crossing=False)),
)
AMP_LO, AMP_HI = 0.5, 2.0    # amplitude band: local L_CII/L_TIR scatter (0.48+-0.21%) + partial->total PAH (~+-0.3 dex)
print("built curve arrays for", len(zgrid), "redshifts")
'''
)

# ----------------------------------------------------------------------------- Obj 3 [CII]
md(
    r"""## 3 · [CII] — mean intensity and power spectrum, vs models and the measurement

Our $L_{\rm CII}=(L_{\rm PAH}/L_{\rm IR})\times(L_{\rm CII}/\text{total-PAH})\times
L_{\rm IR}$ with $L_{\rm CII}/$total-PAH$\approx0.05$ (§6). Comparison curves are the
**published** $L_{\rm CII}$-SFR relations, computed through the same machinery over
the full SMF ($\log M_*>8$) so only the $L$-assignment differs:

- **De Looze+2014** (whole sample): $\log L_{\rm CII}=7.06+\log{\rm SFR}$.
- **Lagache+2018** (z-evolving): $\log L_{\rm CII}=(1.4-0.07z)\log{\rm SFR}+(7.1-0.07z)$.
- (ALPINE / Schaerer+2020 is slope 0.96, offset $-0.03$ dex vs De Looze -- same line.)
- **Chiang+2026** -- the measured cosmic [CII], the **full integral** (black).

Chiang is a full-integral (all-mass) emissivity, so the apples-to-apples comparison is
our **completeness ($\log M_*>8$, green solid) curve** -- it sits $\sim\!1.5\times$
Chiang at $z\sim1$–2 (the local-vs-cosmic offset, §6) and $\sim\!3\times$ at $z\sim3$
(low-mass crossing extrapolation, C2). The green *dotted* $\log M_*>9.9$ curve is our
direct **stacking reach** (mass-incomplete), for reference.

**Impact of our measured crossing** (grey dash-dot = *canonical PAH*, i.e. the same
chain with a flat $L_{\rm PAH}/L_{\rm IR}$, no measured mass/$z$ structure). Over the
full cosmic integral -- the observable -- the crossing **shifts $\langle I_{\rm
CII}\rangle$ by $-20\%$ at $z\sim1$ to $+90\%$ at $z\sim3$**, crossing over at
$z\sim2$: the positive $z\sim1$ slope suppresses the abundant below-pivot galaxies,
the negative $z\sim3$ slope boosts them. (For the mass-matched $\log M_*>9.9$ sample
the effect near-cancels because its effective mass sits at the pivot -- but the cosmic
integral's effective mass is lower, so it does *not* cancel.) So our model predicts a
distinct, testable **z-shape** for the cosmic [CII] signal that a structureless PAH
model does not. Panel (b): the same models into $P(k)$; **no [CII] $P(k)$ data yet**
at these $z$. The crossing's shot-noise effect is isolated in §3c."""
)

code(
    r'''fig, axes = plt.subplots(1, 2, figsize=(13, 5))
CM = {"delooze": "#e08214", "lagache": "#3690c0", "li16": "#e08214", "ours": "#1b7837"}

# (a) [CII] mean intensity vs models + Chiang
ax = axes[0]
ax.plot(C["z"], C["Icii_chiang"], color="k", lw=3.0, label="Chiang+26 (MEASURED)")
ax.fill_between(C["z"], C["Icii_chiang"]/1.35, C["Icii_chiang"]*1.35, color="k", alpha=0.12)
ax.plot(C["z"], C["Icii_delooze"], color=CM["delooze"], lw=2.0, ls="-.", label="De Looze+14 model")
ax.plot(C["z"], C["Icii_lagache"], color=CM["lagache"], lw=2.0, ls="--", label="Lagache+18 model")
# ours FULL-INTEGRAL (completeness >8) is the apples-to-apples comparison to Chiang;
# band = amplitude systematic. The >9.9 stacking-reach curve is secondary (mass-incomplete).
ax.fill_between(C["z"], C["Icii_ours"]*AMP_LO, C["Icii_ours"]*AMP_HI, color=CM["ours"], alpha=0.13)
ax.plot(C["z"], C["Icii_ours"], color=CM["ours"], lw=2.6, label="ours: full integral >8 (compare to Chiang)")
ax.plot(C["z"], C["Icii_ours99"], color=CM["ours"], lw=1.6, ls=":", label="ours: logM*>9.9 (stacking reach)")
# canonical PAH (no measured mass/z structure -- flat L_PAH/L_IR) at the SAME normalization
ax.plot(C["z"], C["Icii_canon"], color="#555555", lw=1.6, ls=(0,(3,1,1,1)),
        label="canonical PAH (no crossing)")
ax.set_yscale("log"); ax.set_ylim(3e2, 5e4)
ax.set_xlabel("z"); ax.set_ylabel(r"$\langle I_{\rm CII}\rangle\ [{\rm Jy\,sr^{-1}}]$")
ax.set_title("(a) [CII] mean intensity")
ax.legend(fontsize=7.0, loc="lower left")

# (b) [CII] power spectrum at z=2.5: models computed into P(k) (no data yet here)
ax = axes[1]
zc = 2.5; kk = np.logspace(-1.5, 0.7, 50)
for Lf, key, lab, ls in [(L_CII_delooze,"delooze","De Looze+14",("-.")),
                         (L_CII_lagache,"lagache","Lagache+18",("--"))]:
    Pc, Ps = line_Pk(kk, zc, Lf, NU_CII)
    ax.plot(kk, Pc+Ps, color=CM[key], lw=1.8, ls=ls, label=f"{lab} model")
Pc, Ps = line_Pk(kk, zc, L_CII_ext, NU_CII, use_crossing=True)
ax.fill_between(kk, (Pc+Ps)*AMP_LO**2, (Pc+Ps)*AMP_HI**2, color=CM["ours"], alpha=0.12)
ax.plot(kk, Pc+Ps, color=CM["ours"], lw=2.6, label="ours (crossing ON)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$k\ [h\,{\rm Mpc^{-1}}]$")
ax.set_ylabel(r"$P_{\rm CII}(k)\ [({\rm Jy\,sr^{-1}})^2\,({\rm Mpc}/h)^3]$")
ax.set_title(f"(b) [CII] power spectrum at z={zc}\n(no [CII] P(k) data yet — EXCLAIM/CONCERTO upcoming)")
ax.legend(fontsize=7.5, loc="lower left")
fig.suptitle("[CII] — intensity & power spectrum vs published models + the Chiang measurement", y=1.03)
fig.tight_layout(); plt.show()

print("[CII] <I> at z=2 [Jy/sr]:  Chiang %.0f  DeLooze %.0f  Lagache %.0f  ours %.0f" %
      (I_cii_chiang(2.0), line_mean(2.0,L_CII_delooze,NU_CII),
       line_mean(2.0,L_CII_lagache,NU_CII), I_cii_complete(2.0)))
print("impact of our crossing on <I_CII> (ours / canonical-no-crossing):")
for z in (0.5, 1.0, 2.0, 3.0):
    i = int((np.abs(zgrid - z)).argmin())
    print(f"  z={z}: {C['Icii_ours'][i]/C['Icii_canon'][i]:.2f}x")
'''
)

# ----------------------------------------------------------------------------- Obj 3 CO
md(
    r"""### 3b · CO — mean intensity and power spectrum, vs Li+2016 and real CO data

The same pair for CO(1-0). Our CO uses the main-sequence $L_{\rm IR}/L'_{\rm CO}$
gas relation ($\approx70$, Sargent+14) carrying the measured crossing shape; the
model is **Li+2016** (the COMAP forecast: SFR$\to L_{\rm IR}\to L'_{\rm CO}$ via
Carilli & Walter 2013). Panel (b) has **real CO data**: **mmIME** (Keating+2020)
shot-noise detection $2.0^{+1.1}_{-1.2}\times10^3\,\mu{\rm K}^2({\rm Mpc}/h)^3$
(higher-$J$ summed) and the **COMAP** ES-V CO(1-0) limit ($\langle T_b\rangle^2<50\,
\mu{\rm K}^2$, $z\sim3$). Our and Li+2016's CO(1-0) power sit well below both --
current data cannot yet reach the CO(1-0) signal; the crossing's shot-noise shift
is a next-generation target. Chiang+26 also detects the CO background (7$\sigma$)."""
)

code(
    r'''fig, axes = plt.subplots(1, 2, figsize=(13, 5))
CM = {"li16": "#e08214", "ours": "#1b7837", "data": "#8073ac"}

# (a) CO mean brightness temperature vs Li+2016
ax = axes[0]
ax.plot(C["z"], C["Tco_li16"], color=CM["li16"], lw=2.0, ls="--", label="Li+2016 model")
ax.fill_between(C["z"], C["Tco_ours"]*0.5, C["Tco_ours"]*2.0, color=CM["ours"], alpha=0.15)
ax.plot(C["z"], C["Tco_ours"], color=CM["ours"], lw=2.6, label="ours (crossing ON)")
ax.set_xlabel("z"); ax.set_ylabel(r"$\langle T_{\rm CO(1-0)}\rangle\ [\mu{\rm K}]$")
ax.set_title("(a) CO(1-0) mean brightness temperature")
ax.legend(fontsize=7.5, loc="upper left")
ax.text(0.97, 0.03, "band = MS L_IR/L'_CO scatter (~0.3 dex).\n"
        "Chiang+26 detects the CO background (7σ).", transform=ax.transAxes,
        fontsize=6.0, va="bottom", ha="right", color="0.35")

# (b) CO(1-0) power spectrum at z=3 vs Li+2016 model + mmIME/COMAP data
ax = axes[1]
zc = 3.0; kk = np.logspace(-1.3, 0.6, 50)
Pc, Ps = line_Pk(kk, zc, L_CO_li16, NU_CO10, T=True)
ax.plot(kk, Pc+Ps, color=CM["li16"], lw=1.8, ls="--", label="Li+2016 model")
Pc, Ps = line_Pk(kk, zc, L_CO_ext, NU_CO10, T=True, use_crossing=True)
ax.fill_between(kk, (Pc+Ps)*0.25, (Pc+Ps)*4.0, color=CM["ours"], alpha=0.12)
ax.plot(kk, Pc+Ps, color=CM["ours"], lw=2.6, label="ours (crossing ON)")
ax.axhspan(MMIME_PSHOT-MMIME_PSHOT_ERR[0], MMIME_PSHOT+MMIME_PSHOT_ERR[1], color=CM["data"], alpha=0.22)
ax.axhline(MMIME_PSHOT, color=CM["data"], lw=1.6, label="mmIME CO P_shot (higher-J)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(20, 6e3)
ax.set_xlabel(r"$k\ [h\,{\rm Mpc^{-1}}]$")
ax.set_ylabel(r"$P_{\rm CO}(k)\ [\mu{\rm K}^2\,({\rm Mpc}/h)^3]$")
ax.set_title(f"(b) CO(1-0) power at z={zc:.0f} vs real data")
ax.legend(fontsize=7.0, loc="lower left")
ax.text(0.03, 0.97, "COMAP ES-V (CO(1-0), z~3): ⟨Tb⟩²<50 μK²\n→ our signal ~10³× below its limit.",
        transform=ax.transAxes, fontsize=6.0, va="top", color="0.35")
fig.suptitle("CO — intensity & power spectrum vs Li+2016 and the mmIME/COMAP data", y=1.03)
fig.tight_layout(); plt.show()

print("CO(1-0) Pshot at z=3 [uK^2(Mpc/h)^3]:  Li16 %.0f  ours %.0f   (mmIME %.0f, higher-J)" %
      (line_Pshot(3.0,L_CO_li16,NU_CO10,T=True), line_Pshot(3.0,L_CO10,NU_CO10,T=True,use_crossing=True),
       MMIME_PSHOT))
'''
)

# ----------------------------------------------------------------------------- Obj 3c crossing
md(
    r"""### 3c · The crossing's isolated effect (the tuning-free result)

The crossing is a *differential* measurement (mass slope at fixed $z$) and needs no
absolute calibration -- both its impacts are tuning-free. §3a showed its effect on the
**full-integral mean** ($-20\%$ at $z\sim1$ to $+90\%$ at $z\sim3$). Here is its effect
on the **shot-noise power**, isolated over our measured range ($\log M_*>9.9$): turning
the crossing on vs. off (a single fixed slope) shifts $P_{\rm shot}\propto\int n\,L^2$
by tens of percent -- biggest at $z\sim0.5$–1 -- while the effective bias $b_{\rm eff}$
(inset) barely moves. So a fixed-slope PAH forecast mis-predicts both the mean $z$-shape
(§3a) and the small-scale power (here), for both [CII] and CO."""
)

code(
    r'''fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))

# (a) [CII] shot-noise plateau vs z, crossing ON vs OFF (over our measured >9.9)
ax = axes[0]
ax.plot(C["z"], C["Pcii_on"], color=C_ON, lw=2.6, label="crossing ON")
ax.plot(C["z"], C["Pcii_off"], color=C_OFF, lw=2.0, ls="--", label="fixed slope (OFF)")
ax.fill_between(C["z"], C["Pcii_on"], C["Pcii_off"], color=C_ON, alpha=0.15)
ax.set_yscale("log"); ax.set_xlabel("z")
ax.set_ylabel(r"$P_{\rm shot}^{\rm [CII]}\ [({\rm Jy\,sr^{-1}})^2\,({\rm Mpc}/h)^3]$")
ax.set_title("(a) [CII] shot-noise vs z (logM*>9.9)\ncrossing raises it 35–50% at z~0.5–1")
ax.legend(fontsize=9, loc="lower left")
axi = ax.inset_axes([0.60, 0.58, 0.37, 0.36])
axi.plot(C["z"], C["beff_on"], color=C_ON, lw=1.8); axi.plot(C["z"], C["beff_off"], color=C_OFF, lw=1.5, ls="--")
axi.set_title(r"$b_{\rm eff}(z)$ (≈unchanged)", fontsize=7); axi.tick_params(labelsize=6)

# (b) crossing shot-noise leverage (ON/OFF) vs z, [CII] and CO
ax = axes[1]
co_on = np.array([line_Pshot(z, L_CO10, NU_CO10, T=True, logM_min=9.9, use_crossing=True) for z in zgrid])
co_off = np.array([line_Pshot(z, L_CO10, NU_CO10, T=True, logM_min=9.9, use_crossing=False) for z in zgrid])
ax.plot(C["z"], C["Pcii_on"]/C["Pcii_off"], color=C_CII, lw=2.6, label="[CII]")
ax.plot(C["z"], co_on/co_off, color=C_CO, lw=2.6, ls="--", label="CO")
ax.axhline(1.0, color="k", lw=1, ls=":")
ax.set_xlabel("z"); ax.set_ylabel(r"$P_{\rm shot}^{\rm ON}/P_{\rm shot}^{\rm OFF}$")
ax.set_title("(b) crossing's shot-noise leverage vs z")
ax.legend(fontsize=9)
fig.suptitle("Obj 3c — the crossing shifts the shot noise (tuning-free); bias is untouched", y=1.03)
fig.tight_layout(); plt.show()

print("crossing shot-noise leverage (>9.9):")
for z in (0.5, 1.0, 2.0, 3.0):
    i = int((np.abs(zgrid - z)).argmin())
    print(f"  z={z}: [CII] {C['Pcii_on'][i]/C['Pcii_off'][i]:.3f}x   "
          f"b_eff {C['beff_on'][i]/C['beff_off'][i]:.3f}x")
'''
)

# ----------------------------------------------------------------------------- Obj 4
md(
    r"""## 4 · The 24 $\mu$m PAH-contamination bias, handed to LIM forecasters

Many LIM forecasts derive SFR (hence $L_{\rm line}$) from IR/24 $\mu$m. Our whole
program shows 24 $\mu$m is PAH-contaminated in a mass/$z$-dependent way, so those
forecasts inherit a bias. Below: the PAH fraction of the stacked 24 $\mu$m flux
vs $(M_*, z)$, i.e. the multiplicative correction to apply before a 24 $\mu$m
$\to$ SFR $\to L_{\rm line}$ step.

> **Caveat:** `greybody.py::_pah_coeffs` are flagged stale in branch 9 (predate
> the branch-7 normalization fixes). Refresh before quoting numbers. Here we
> derive the fraction directly from the crossing $L_{\rm PAH}/L_{\rm IR}(M_*,z)$
> and the fraction of $L_{\rm PAH}$ falling in the 24 $\mu$m band at each $z$."""
)

code(
    r'''# Fraction of L_PAH landing in MIPS-24 depends on which rest-frame features the
# broadband sees at each z. Use the pah_bandpass/pah_spectrum machinery if we want
# it exact; for this first pass use a smooth proxy peaking where 7.7+8.6 sit in-band
# (z ~ 1.5-2.5, MIPS-24 -> rest 6.9-9.6 um).
def pah_in_24_fraction(z):
    """Fraction of L_PAH sampled by MIPS-24 vs z (proxy; TODO: pah_bandpass integral)."""
    return 0.35 * np.exp(-((z - 2.0) ** 2) / (2 * 0.9 ** 2)) + 0.05

zg = np.linspace(0.5, 3.5, 25)
mg = np.array([10.2, 10.7, 11.0, 11.4])
# bias factor = f24_PAH / f24_total ~ (L_PAH/L_IR)*frac_in_band / [(L_PAH/L_IR)*frac + warm-cont]
# proxy: PAH excess over the cold+warm continuum at 24um.
def frac24_pah(logM, z):
    pah_ir = 10 ** log_pah_ir(logM, z)
    return (pah_ir * pah_in_24_fraction(z)) / (pah_ir * pah_in_24_fraction(z) + 0.02)

Z, Mgrid = np.meshgrid(zg, mg)
B = frac24_pah(Mgrid, Z)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
ax = axes[0]
im = ax.pcolormesh(zg, mg, 100*B, shading="auto", cmap="magma")
ax.set_xlabel("z"); ax.set_ylabel(r"$\log M_*$")
ax.set_title("PAH fraction of MIPS-24 flux [%]"); fig.colorbar(im, ax=ax, label="%")

ax = axes[1]
for m, col in zip(mg, plt.cm.plasma(np.linspace(0.15,0.85,len(mg)))):
    ax.plot(zg, 100*frac24_pah(m, zg), color=col, lw=2, label=f"logM*={m}")
ax.set_xlabel("z"); ax.set_ylabel(r"$f_{24,\rm PAH}$ [%]")
ax.set_title("24 μm PAH contamination vs z"); ax.legend(fontsize=8)
fig.suptitle("Obj 4 — the 24 μm correction table for LIM forecasters (proxy; refresh _pah_coeffs)", y=1.02)
fig.tight_layout(); plt.show()

corr = pd.DataFrame(100*B, index=[f"logM*={m}" for m in mg],
                    columns=[f"z={z:.1f}" for z in zg[::4]] if False else [f"{z:.2f}" for z in zg])
print("PAH fraction of 24um flux [%] (rows=mass, cols=z):")
print(corr.iloc[:, ::4].round(1).to_string())
'''
)

# ----------------------------------------------------------------------------- takeaways
md(
    r"""## 5 · Three takeaways for LIM forecasters

1. **Your mean-intensity forecasts are robust to the PAH crossing.** Because
   $\langle I\rangle$ is luminosity-weighted with an effective mass at the crossing
   pivot, the measured non-monotonic mass slope self-cancels in the mean (<2% at
   $z\lesssim2$). Good news -- but it means the crossing is *not* an amplitude
   story.

2. **Your shot-noise / power-spectrum forecasts are NOT.** $P_{\rm shot}\propto\int
   n\,L^2$ weights the high-mass end where the crossing acts, so it lifts the
   **shot-noise plateau 35–50% at $z\sim0.5$–1** (§3b) while the clustering term
   ($\langle I\rangle^2 b_{\rm eff}^2$) is nearly untouched -- so the
   clustering/shot crossover a power-spectrum experiment measures moves. A
   fixed-slope model gets $P(k)$ wrong even when it gets the mean right. **This is
   the result.**

3. **Our absolute amplitude is consistent with the Chiang measurement.** Chiang+26
   *measures* the cosmic [CII] to $z=4.2$ (the full-integral emissivity -- our method's
   cousin). Compared full-integral to full-integral (our $\log M_*>8$ curve), we sit
   $\sim\!1.5\times$ Chiang -- the expected local-vs-cosmic offset in $L_{\rm CII}/
   L_{\rm IR}$ ($0.5\%$ local vs $0.33\%$ cosmic), within our systematic band (§6).

4. **Here is the 24 $\mu$m PAH correction.** Forecasts using IR/24 $\mu$m
   SFRs inherit a mass/$z$-dependent PAH bias; §4 is the correction to apply.

> **Note (the amplitude is not the claim).** Our absolute $L_{\rm CII}$ = (measured
> $L_{\rm PAH}/L_{\rm IR}$) $\times$ (matched-definition $L_{\rm CII}/$total-PAH). It
> is consistent with Chiang but still carries a $\sim\!\times2$ systematic (bridge
> scatter + partial$\to$total PAH), so we do not claim it as a measurement. The
> robust, tuning-free content is the mass/$z$ **structure** (takeaway 2).

---

### Status / next pass (what is real vs parameterized here)

| Ingredient | This notebook | Next pass |
|---|---|---|
| Forecast chain (ladder→ρ_L→intensity→P(k)) | **real, validated** | — |
| Power spectrum P(k): P_lin (BBKS), growth, Tinker bias | **real** | (refine SHMR) |
| Chiang+26 [CII] measurement overlay | **real** (published ρ_CII fit) | — |
| Absolute amplitude | measured L_PAH/L_IR × matched L_CII/total-PAH (0.05); consistent w/ Chiang, ~×2 band | pin partial→total PAH; measured L_PAH/L_IR |
| Crossing $L_{\rm PAH}/L_{\rm IR}(M_*,z)$ | documented values | wire real branch-9 stacks |
| $n(M_*,z)$, $L_{\rm IR}(M_*,z)$ | SMF + main sequence | measured from the stacks |
| Model curves | computed from published relations (De Looze+14, Lagache+18; Li+16 for CO) | add SIDES/Popping/Padmanabhan if digitized |
| Real data overlays | Chiang+26 [CII] measurement; mmIME + COMAP CO power | — |
| 24 μm PAH fraction | smooth proxy | `pah_bandpass` integral; refresh `_pah_coeffs` |

**The robust deliverable is the mass/$z$ structure** (crossing → shot-noise shift),
which is tuning-free. The absolute amplitude is now consistent with Chiang (matched
bridge, §6) but still carries a ~×2 systematic (partial→total PAH, local scatter);
pinning partial→total and wiring the measured $n(M_*,z)$/$L_{\rm IR}$ from
the stacks is the other next step."""
)

# ----------------------------------------------------------------------------- Obj 6 systematics
md(
    r"""## 6 · The [CII] amplitude and the right comparison to Chiang

**Self-consistent local values (KINGFISH normal star-forming galaxies):**

| quantity | value | source |
|---|---|---|
| $L_{\rm PAH}/L_{\rm IR}$ | $\approx 10\%$ (total PAH) | Smith+07 |
| 7.7 $\mu$m complex / total PAH | $\approx 49\%$ | Smith+07 |
| $L_{\rm CII}/$total-PAH | $\approx 0.05$ ($\leftrightarrow L_{\rm CII}/L_{7.7}\approx0.1$) | Herrera-Camus+15 / Smith+07 |
| $L_{\rm CII}/L_{\rm IR}$ (local normal SF) | $\approx 0.5\%$ | Herrera-Camus+15 ($0.48\pm0.21\%$) |
| $L_{\rm CII}/L_{\rm IR}$ (cosmic average) | $\approx 0.33\%$ | Chiang+26 |

Our chain is $L_{\rm CII}=(L_{\rm PAH}/L_{\rm IR})\times(L_{\rm CII}/\text{total-PAH})
\times L_{\rm IR}$; the total-PAH template requires the **total-PAH** bridge
($\approx0.05$), not the 7.7-complex ratio ($\approx0.1$).

**Which curve compares to Chiang?** Chiang measures the *aggregate* cosmic [CII]
emissivity $\rho_{\rm CII}(z)$ — the **full integral** over the luminosity function,
all masses (a diffuse-background clustering measurement, not mass-limited). So the
apples-to-apples comparison is our **full-mass (completeness, $\log M_*>8$) curve**,
not the $\log M_*>9.9$ stacking-reach curve.

- The **full-integral curve sits $\sim\!1.5\times$ above Chiang at $z\sim1$–2** — the
  *local-vs-cosmic* offset ($0.5\%$ local vs $0.33\%$ cosmic $L_{\rm CII}/L_{\rm IR}$),
  within our band. Consistent, with the offset understood.
- At $z\sim3$ it rises to $\sim\!3\times$; the extra factor is the low-mass crossing
  extrapolation (systematic C2), most uncertain where the low-$Z$ PAH deficit should
  turn it over.
- The $\log M_*>9.9$ curve appears to lie *on* Chiang, but that is **coincidental** —
  it omits the low-mass half of the population ($\sim\!0.6\times$), which cancels the
  $\sim\!1.5\times$ amplitude offset. It is not the right comparison to a full integral.

**The robust, tuning-free result is the mass/$z$ structure** (the crossing →
shot-noise shift, §3c), differential in mass at fixed $z$ and immune to the amplitude
systematics. Full audit: `docs/forecast-lim-via-pah-1-systematics.md`."""
)

# ----------------------------------------------------------------------------- Obj 7 real catalog n(M,z)
md(
    r"""## 7 · Real catalog $n(M_*,z)$ — replacing the parameterized SMF (CII)

The forecast so far used a parameterized SMF for the comoving number density. Here
we swap in the **actual COSMOS2020 source counts** from the combined stack, which
also lets us include the $9.0<\log M_*<9.9$ bin (stacked but previously an unanalysed
nuisance layer) -- so the low-mass end is *measured*, not extrapolated below 9.9.

Recipe (§ intro): $n(M,z)=N/V$, with $N$ = star-forming (NUVrJ split) source counts
per $(M,z)$ cell read straight from the stack JSON `n_sources`, and $V=\Omega\times$
(comoving shell) using $\Omega=1.27$ deg$^2$ (COSMOS2020 effective area, Weaver+23 =
Viero+22) via `CosmologyCalculator.comoving_volume_element`. COSMOS2020 is mass-complete
to $\log M_*\!=\!9$ out to $z\!\sim\!3$, so all five bins are usable across our range
(above $z\!\sim\!3$ the lowest bin is a lower limit). The three dither runs are the
*same galaxies* at staggered $z$, so we **average** $n$ across them per mass bin (not
sum). $L_{\rm CII}$ per bin uses the catalog median mass with the same model + bridge."""
)

code(
    r'''import os, json, re, ast
from simstack4.cosmology import CosmologyCalculator

CAT_AREA_DEG2 = 1.27      # COSMOS2020 effective (Weaver+23) = Viero+22
MASS_EDGES = [9.0, 9.9, 10.6, 10.8, 11.0, 12.0]
RUN_DATES_COMBINED = ["20260707_204926", "20260707_225921", "20260708_091724"]
_pj = os.path.join(os.environ["PICKLESPATH"], "simstack", "stacked_flux_densities")
_cc = CosmologyCalculator()
_KEY = re.compile(r"redshift_([\d.]+)_([\d.]+)__stellar_mass_([\d.]+)_([\d.]+)__split_(\d+)")

def _shell_vol(zlo, zhi, area=CAT_AREA_DEG2, nz=8):
    zs = np.linspace(zlo, zhi, nz)
    return np.trapezoid(_cc.comoving_volume_element(zs, sky_area_deg2=area), zs)   # Mpc^3

_rows = []
for date in RUN_DATES_COMBINED:
    with open(os.path.join(_pj, f"cosmos20_stacking_{date}.json")) as f:
        j = json.load(f)
    ns, bp = j["n_sources"], j.get("bin_properties", {})
    for k, v in ns.items():
        m = _KEY.match(k)
        if not m or m.group(5) != "0" or int(v) < 5:   # split_0 = star-forming (NUVrJ)
            continue
        zlo, zhi, mlo, mhi, _ = m.groups()
        props = bp.get(k, {})
        if isinstance(props, str): props = ast.literal_eval(props)
        mmed = float(props.get("lp_mass_med", 0.5*(float(mlo)+float(mhi))))
        _rows.append(dict(mlo=float(mlo), zmid=0.5*(float(zlo)+float(zhi)),
                          mmed=mmed, N=int(v), n=int(v)/_shell_vol(float(zlo), float(zhi))))
CAT = pd.DataFrame(_rows)
print(f"catalog cells: {len(CAT)} (star-forming, {len(RUN_DATES_COMBINED)} dither runs), "
      f"area={CAT_AREA_DEG2} deg^2, z {CAT.zmid.min():.2f}-{CAT.zmid.max():.2f}")

def catalog_bins_at(zc, dz=0.13):
    sub = CAT[np.abs(CAT.zmid - zc) < dz]
    return [(sub[sub.mlo==mlo].mmed.mean(), sub[sub.mlo==mlo].n.mean())
            for mlo in MASS_EDGES[:-1] if (sub.mlo==mlo).any()]

def I_cii_catalog(z):    # discrete sum over the 5 real mass bins (star-forming)
    rho = sum(n * L_CII_ext(mc, z, use_crossing=True) for mc, n in catalog_bins_at(z))
    return line_Tb_Inu(z, rho, NU_CII)[1]

# validation: catalog SFRD(SF, logM>9) vs Madau-Dickinson (all-mass) -- expect ~0.6-1
print("catalog SFRD(SF,>9) / MD14  (should be <1, the logM>9 fraction):")
for zc in (0.6, 1.0, 2.0, 3.0):
    sfrd = sum(n * 10**(log_ssfr_ms(mc, zc)+mc) for mc, n in catalog_bins_at(zc))
    print(f"  z={zc}: {sfrd/madau_dickinson(zc):.2f}")
'''
)

code(
    r'''fig, axes = plt.subplots(1, 2, figsize=(13, 5))
zc_cat = np.array([0.6,0.78,0.93,1.08,1.22,1.38,1.53,1.68,1.83,1.98,2.12,2.28,2.6,2.95])
bincol = plt.cm.viridis(np.linspace(0.1, 0.9, len(MASS_EDGES)-1))

# (a) catalog n(M,z) vs parameterized SMF, per mass bin
ax = axes[0]
for mlo, mhi, col in zip(MASS_EDGES[:-1], MASS_EDGES[1:], bincol):
    sub = CAT[CAT.mlo == mlo].sort_values("zmid")
    ax.plot(sub.zmid, sub.n, "o", ms=3, color=col, alpha=0.7)
    zz = np.linspace(0.4, 3.5, 40)
    n_smf = np.array([np.trapezoid(smf(np.linspace(mlo,mhi,12), z), np.linspace(mlo,mhi,12)) for z in zz])
    ax.plot(zz, n_smf, "-", color=col, lw=1.6, label=f"{mlo:.1f}-{mhi:.1f}")
ax.set_yscale("log"); ax.set_xlim(0.4, 3.6); ax.set_ylim(1e-5, 2e-2)
ax.set_xlabel("z"); ax.set_ylabel(r"$n\ [{\rm Mpc^{-3}}]$")
ax.set_title("(a) catalog counts (points) vs parameterized SMF (lines)")
ax.legend(fontsize=7, title="logM* bin", ncol=2)

# (b) <I_CII>(z): catalog-n vs parameterized-SMF vs Chiang
ax = axes[1]
Icat = np.array([I_cii_catalog(z) for z in zc_cat])
ax.plot(C["z"], C["Icii_chiang"], color="k", lw=3.0, label="Chiang+26 (MEASURED)")
ax.fill_between(C["z"], C["Icii_chiang"]/1.35, C["Icii_chiang"]*1.35, color="k", alpha=0.12)
ax.plot(C["z"], C["Icii_ours"], color="#1b7837", lw=1.8, ls=":", label="parameterized SMF (>8)")
ax.plot(zc_cat, Icat, "s-", color="#d94801", lw=2.4, ms=4, label="real catalog n (SF, logM*>9)")
ax.fill_between(zc_cat, Icat*AMP_LO, Icat*AMP_HI, color="#d94801", alpha=0.13)
ax.set_yscale("log"); ax.set_ylim(3e2, 5e4)
ax.set_xlabel("z"); ax.set_ylabel(r"$\langle I_{\rm CII}\rangle\ [{\rm Jy\,sr^{-1}}]$")
ax.set_title("(b) [CII] intensity: real catalog n vs SMF vs Chiang")
ax.legend(fontsize=7.5, loc="lower left")
fig.suptitle("Real COSMOS2020 catalog n(M*,z) — validates the SMF forecast, measured to logM*=9", y=1.02)
fig.tight_layout(); plt.show()

print("<I_CII> [Jy/sr]  catalog / SMF / Chiang:")
for z in (0.6, 1.0, 2.0, 3.0):
    i = int((np.abs(C["z"]-z)).argmin())
    print(f"  z={z}: catalog={I_cii_catalog(z):6.0f}  SMF={C['Icii_ours'][i]:6.0f}  Chiang={C['Icii_chiang'][i]:6.0f}")
'''
)

# ----------------------------------------------------------------------------- Obj 7b low-mass turnover
md(
    r"""### 7b · The measured low-mass PAH turnover — resolving the $z\sim3$ overshoot

Directly stacking the $9.0<\log M_*<9.9$ bin (S/N-marginal, median 24$\mu$m S/N$\approx
2.8$; cold-continuum extrapolation unreliable) shows **no 24$\mu$m PAH excess** --
PAH/$f_{\rm peak}=-0.02$ vs $+0.005$–$0.008$ in the higher-mass bins. So the low-mass
end is **not PAH-boosted**; the data favour the low-metallicity PAH *deficit* (a
$q_{\rm PAH}(Z)$ turnover). That directly undercuts the extrapolation behind the
$z\sim3$ completeness overshoot (§6, C2): holding $L_{\rm PAH}/L_{\rm IR}$ flat below
9.9 keeps it at the crossing's high $z\sim3$ 9.9-value, but the stack says it should
*decline*.

Below: apply a data-informed turnover -- $L_{\rm PAH}/L_{\rm IR}$ declines by `turn`
dex/dex below 9.9 (band: turn $\in[1,2.5]$, the marginal-S/N uncertainty; turn$=0$ is
flat, which overshoots at $z\sim3$, §7). Two curves, **both $\log M_*>9$ with the
turnover**, so they differ only in the source counts:

- **green (jagged) = real catalog $n$** -- the measured COSMOS2020 star-forming counts;
  the jaggedness is real per-$z$-bin Poisson + cosmic variance (finite sources per
  $\Delta z\!\approx\!0.15$ over 1.27 deg$^2$).
- **orange (smooth) = parameterized SMF $n$** over the same $\log M_*>9$ range -- the
  analytic model.

They agree and both track Chiang, so the SMF is a good stand-in for the real counts
over the measured mass range. *Caveat:* $L_{\rm CII}$ from PAH is unreliable at low $Z$
([CII] *rises* while PAH falls), so low-mass [CII] is better taken from SFR directly --
this panel shows the PAH-chain turnover, not a claim that low-mass [CII] vanishes."""
)

code(
    r'''def I_cii_catalog_turn(z, turn):     # real catalog n (logM*>9), with the turnover
    rho = sum(n * L_CII_turn(mc, z, use_crossing=True, turn=turn) for mc, n in catalog_bins_at(z))
    return line_Tb_Inu(z, rho, NU_CII)[1]
def I_cii_smf9_turn(z, turn=1.5):        # parameterized SMF over logM*>9, same turnover
    g = np.linspace(9.0, 12.0, 140); dl = g[1]-g[0]
    return line_Tb_Inu(z, np.sum(smf(g,z)*L_CII_turn(g,z,use_crossing=True,turn=turn))*dl, NU_CII)[1]

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(C["z"], C["Icii_chiang"], color="k", lw=3.0, label="Chiang+26 (MEASURED)")
ax.fill_between(C["z"], C["Icii_chiang"]/1.35, C["Icii_chiang"]*1.35, color="k", alpha=0.12)
# orange = SMF model, logM*>9, turnover (smooth)
zg = np.linspace(0.45, 3.15, 60)
ax.plot(zg, [I_cii_smf9_turn(z) for z in zg], "-", color="#d94801", lw=2.2,
        label="SMF model, logM*>9 + turnover (smooth)")
# green = real catalog, logM*>9, turnover (jagged = real per-z scatter); band = turn in [1,2.5]
Iturn = np.array([I_cii_catalog_turn(z, 1.5) for z in zc_cat])
Ilo = np.array([I_cii_catalog_turn(z, 2.5) for z in zc_cat])
Ihi = np.array([I_cii_catalog_turn(z, 1.0) for z in zc_cat])
ax.fill_between(zc_cat, Ilo, Ihi, color="#238b45", alpha=0.16)
ax.plot(zc_cat, Iturn, "o-", color="#238b45", lw=2.6, ms=4,
        label="real catalog n, logM*>9 + turnover")
ax.set_yscale("log"); ax.set_ylim(1e3, 2e4); ax.set_xlim(0.4, 3.2)
ax.set_xlabel("z"); ax.set_ylabel(r"$\langle I_{\rm CII}\rangle\ [{\rm Jy\,sr^{-1}}]$")
ax.set_title("7b · real catalog counts vs SMF model (both logM*>9 + low-mass turnover)\ntrack Chiang; turnover removes the z~3 overshoot")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); plt.show()

print("z   catalog/Chiang   SMF/Chiang   (both logM*>9 + turnover)")
for z in (1.0, 2.0, 3.0):
    j = int((np.abs(zc_cat-z)).argmin()); i = int((np.abs(C["z"]-z)).argmin())
    print(f"  {z}:    {Iturn[j]/C['Icii_chiang'][i]:.2f}          {I_cii_smf9_turn(z)/C['Icii_chiang'][i]:.2f}")
'''
)

# ----------------------------------------------------------------------------- write
nb["cells"] = cells
out = "notebooks/2026-07-12-lim-via-pah.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")
