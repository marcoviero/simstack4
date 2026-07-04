"""Generate 2026-07-02-pah-evolving-template-mcmc-simulation.ipynb.

Branch-7 notebook (user-directed 2026-07-02, extended 2026-07-03): inject a
realistically *evolving* set of PAH SEDs (amplitude AND line ratios drifting
with sSFR(z, M*) at fixed stellar mass, on top of a mass trend), "stack" them
through the MIPS 24+70 µm bandpasses with the dithered scheme actually used
on COSMOS2020 (dz=0.15 x 3 staggered runs, 4 mass bins, ~140 sources/bin),
and run MCMC on the forward-model parameters (PAHSpectrumModel.fit_evolving_mcmc,
per-bin linear pair profiled analytically) at increasing flexibility:

  L1  shared ratios + amplitude evolution only
  L2  shared ratios + amplitude + ratio evolution   (the fit_evolving model)
  L2n L2 with the constant-flux feature term        (dimming-absorption demo)
  L2b L2 with 24 um only                            (70 um leverage test)
  L2d L2 at 33x depth                               (systematics floor)
  L3  per-mass-bin ratios, no evolution             (Sec 1a-style flexibility)
  L4  per-mass-bin ratios + evolution               (everything at once)

2026-07-03 additions (user review of the first executed version):
  - The truth now carries a flux envelope E(z, M*) calibrated to the real
    3-fold smoothed f24_cold(z, M*), so simulated f24 matches the measured
    scale and z-shape (the first version had no distance dimming at all).
  - feature_envelope="baseline" model option: features dim with the source.
  - Sec 4c: posterior-predictive check explaining why the truth sits
    off-center in the MCMC corners.
  - Sec 7: static vs evolving model comparison ON THE REAL DATA (3 K-fold
    COSMOS2020 stacks), with a scatter-null calibration and the real-data
    f24(z) decomposition figure.

Requires PICKLESPATH for Sec 7 (real stacking JSONs).

Run:  uv run python notebooks/build_pah_evolving_mcmc_notebook.py
Then: uv run jupyter nbconvert --to notebook --execute --inplace \
          notebooks/2026-07-02-pah-evolving-template-mcmc-simulation.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(src):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


# ----------------------------------------------------------------------------
md(
    r"""# Evolving PAH templates through the MIPS bandpasses: an MCMC flexibility study

**Branch 7, 2026-07-02 (simulation) + 2026-07-03 (realism calibration + real-data comparison).**

Different redshifts probe different parts of the PAH SED through the broad
MIPS 24 µm bandpass, and at fixed stellar mass the contributing galaxies'
sSFR rises with z — so both the overall PAH amplitude *and the inter-band
line ratios* can evolve along a single mass bin's tomographic curve. MIPS
70 µm probes a different rest wavelength at the same z, giving leverage
against the amplitude-vs-ratio degeneracies.

**Questions**: how much model flexibility can the dithered-stacking data
actually support (§3–§6, simulation)? And is a non-evolving template
dramatically worse than an evolving one on the real COSMOS2020 stacks (§7)?

Provenance notes:
- Fitting machinery is `PAHSpectrumModel.fit_evolving_mcmc`: emcee over the
  evolution slopes (η_A, η_g) and log feature-group ratios, with each mass
  bin's linear pair (C_m, α_m) profiled analytically at every step — the
  `DustEvolutionModel` pattern. Injection-recovery guards live in
  `tests/test_pah_evolution_recovery.py`.
- The truth is injected through `TruthSpectrum.band_flux_curve` — direct
  integration of the evolving rest spectrum, *independent* of the fitter's
  kernel decomposition — and per-source photo-z scattering
  (`simulate_dithered_fluxes`), so kernel-approximation systematics are in
  play, unlike the self-consistent test fixtures.
- **2026-07-03 realism fix (user review)**: the first executed version had
  no flux envelope — band fluxes were comoving-luminosity-like (constant
  amplitude with z) while the real stacked f24 falls ~7–10× over
  z = 0.5–3.5 from distance dimming + luminosity evolution. The truth now
  carries `flux_envelope` calibrated to the real 3-fold smoothed
  f24_cold(z, M*) (coefficients derived in §7's cross-check), and the fits
  use the matching `feature_envelope="baseline"` model option so the
  *features* dim with the source too. Without that option the constant-flux
  feature term lets the dimming leak into a spuriously negative η_A — rung
  L2n demonstrates it, and §7 shows what it did to real-data fits.
- Developing this notebook exposed (and fixed) a multi-band normalization
  bug in the evolving fits' data prep: each band's cold baseline was
  median-normalized separately while sharing one C_m, silently forcing
  equal continuum levels in 24 and 70 µm (the injection tests never caught
  it because they use identical baseline columns for both bands). All bands
  are now normalized by one per-bin scalar, preserving the cross-band
  amplitude ratio. Single-band fits — everything the branch-6/7 headline
  results are built on — are numerically unchanged.
"""
)

# ----------------------------------------------------------------------------
code(
    r"""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from simstack4.pah_dither import (
    DitherScheme, TruthSpectrum, simulate_dithered_fluxes,
)
from simstack4.pah_spectrum import (
    PAHSpectrumModel, evolving_flux_decomposition,
)
from simstack4.plots import plot_pah_flux_decomposition, _PAH_GROUP_COLORS
from simstack4.dust_evolution import _greybody_nu

rng_seed = 11
pd.set_option("display.width", 150)

# Feature groups with the REFERENCE GROUP FIRST: the fitted ratios r_g and
# ratio slopes eta_g are defined relative to group 0 (r_0 == 1, eta_0 == 0),
# and the amplitude slope eta_A *is* the reference group's slope. Anchoring
# the reference on the strongest complex (7.7+8.6 um) keeps eta_A pinned by
# the best-measured feature; referencing the weak 6.2 um instead lets eta_A
# float (verified in a development scratch comparison).
GROUPS = [[1, 2], [0], [3], [4], [5, 6]]   # 7.7+8.6 | 6.2 | 11.3 | 12.7 | 16.4+17.0
GROUP_NAMES = ["7.7+8.6", "6.2", "11.3", "12.7", "16.4+17.0"]

# Truth: peak feature-to-MIR-continuum ratios per group (at the mass pivot),
# a rising-with-mass EW trend (branch-6/7 measured direction, +0.35 dex/dex),
# and per-group sSFR slopes e_g. The (eta_A, eta_g) reparameterization
# subtracts the reference group's slope.
RATIO_TARGETS = np.array([2.0, 0.6, 0.3, 0.8, 0.5])  # feature peak / continuum
E_G_TRUE = np.array([0.9, 0.4, 0.1, 0.6, 0.4])       # d log A_g / d log sSFR
ETA_A_TRUE = float(E_G_TRUE[0])
ETA_G_TRUE = E_G_TRUE - ETA_A_TRUE                   # eta_g (ref group -> 0)
BETA_MASS = 0.35
SIGMA_Z0, F_CAT = 0.02, 0.02

# Real 4-bin science mass scheme (bin centers)
MASS_CENTERS = [10.25, 10.65, 10.95, 11.30]
MASS_LABELS = ["10.0-10.5", "10.5-10.8", "10.8-11.1", ">11.1"]

scheme = DitherScheme.uniform(
    z_min=0.5, z_max=3.5, dz=0.15, n_stagger=3,
    property_bins=[{"log_M_star": m} for m in MASS_CENTERS],
    bands=("MIPS_24", "MIPS_70"),
)

# ── Realism calibration (2026-07-03) ────────────────────────────────────────
# (1) Flat (rest-lambda-independent) hot/VSG MIR continuum under the PAH
#     features, scaled so the CONTINUUM f70/f24 ratio is ~8.5 at z~0.6 — the
#     real 3-fold value (f70_cold/f24_cold at z<0.8, Sec 7 data). The 60 K
#     warm MBB supplies the 70 um side; the flat plaw supplies 5-16 um.
#     "Flat" makes feature/continuum ratios track the real behavior (the
#     observed f24/f24_cold stays ~1-2 across all z) and makes the
#     feature_envelope="baseline" model exact for the sim.
MIR_PLAW_AMP = float(_greybody_nu(2.998e14 / 43.75, 60.0, 1.5)) / 7.5

# (2) Observed-flux envelope E(z, M*): distance dimming + luminosity
#     evolution + bandwidth factors, shared by continuum and features.
#     Coefficients = quadratic-in-z + linear-in-mass fit to the real 3-fold
#     smoothed f24_cold [mJy] (scatter 0.016 dex; cross-checked against the
#     live data in Sec 7):  log10 f24_cold = c0 + c1 z + c2 z^2 + c3 (logM-10.75)
ENV_COEF = (-0.3213, -0.6455, 0.0899, 0.6135)


def flux_envelope(z, prop_bin=None):
    dM = (prop_bin or {}).get("log_M_star", 10.75) - 10.75
    c0, c1, c2, c3 = ENV_COEF
    # divide by the flat plaw amp so sim f24_cold(z, M*) == the real relation
    return 10.0 ** (c0 + c1 * z + c2 * z**2 + c3 * dM) / MIR_PLAW_AMP


truth = TruthSpectrum(
    feature_groups=GROUPS, amp0=RATIO_TARGETS * MIR_PLAW_AMP,
    beta_mass=BETA_MASS,
    eta_ssfr_amp=ETA_A_TRUE, eta_ssfr_ratio=ETA_G_TRUE,
    mir_plaw_amp=MIR_PLAW_AMP, mir_plaw_slope=0.0,
    flux_envelope=flux_envelope,
)
# Continuum-only twin: the (perfect) cold/warm baseline the fits are given.
cont_truth = TruthSpectrum(
    feature_groups=GROUPS, amp0=np.zeros(len(GROUPS)), beta_mass=BETA_MASS,
    mir_plaw_amp=MIR_PLAW_AMP, mir_plaw_slope=0.0,
    flux_envelope=flux_envelope,
)


def add_cold_baselines(df, scheme, cont):
    "Attach f24_cold / f70_cold columns: the continuum-only in-band flux."
    zg = np.linspace(0.2, 4.0, 400)
    df = df.copy()
    for band, col in [("MIPS_24", "f24_cold"), ("MIPS_70", "f70_cold")]:
        vals = np.full(len(df), np.nan)
        for m, prop in enumerate(scheme.property_bins):
            curve = cont.band_flux_curve(zg, band, prop)
            sel = (df["prop_bin_id"] == m).to_numpy()
            vals[sel] = np.interp(df.loc[sel, "z_mid"], zg, curve)
        df[col] = vals
    return df


def truth_at_pivot(res):
    "Expected (r_g, e_g) at the FIT's sSFR pivot (truth pivot is -9.0)."
    ds = res["s_pivot"] - truth.s_pivot
    r_exp = (RATIO_TARGETS / RATIO_TARGETS[0]) * 10.0 ** (ETA_G_TRUE * ds)
    return r_exp, E_G_TRUE


def rung_summary(name, res):
    "One comparison row per feature group for a fit result."
    r_exp, e_true = truth_at_pivot(res)
    e_rec = res["eta_amp"] + res["eta_ratio"]
    e_err = np.hypot(
        res["eta_amp_err"] if np.isfinite(res["eta_amp_err"]) else 0.0,
        res["eta_ratio_err"],
    )
    return pd.DataFrame({
        "rung": name, "group": GROUP_NAMES,
        "r_rec": np.round(res["r"], 3), "r_err": np.round(res["r_err"], 3),
        "r_true": np.round(r_exp, 3),
        "e_rec": np.round(e_rec, 3), "e_err": np.round(e_err, 3),
        "e_true": e_true,
        "chi2_red": np.round(res["chi2_red"], 2),
    })


print(f"MIR_PLAW_AMP = {MIR_PLAW_AMP:.4f} (continuum f70c/f24c ~ 8.5 at z~0.6)")
"""
)

# ----------------------------------------------------------------------------
md(
    r"""## 1. The evolving truth, calibrated to the measured stacks

Rest-frame spectrum: warm modified blackbody (T=60 K, β=1.5) + a flat
hot/VSG MIR continuum + the 7 PAH features in 5 groups, all multiplied by
an observed-flux envelope E(z, M\*) fit to the real 3-fold smoothed
f24_cold — so the simulated f₂₄ has the measured mJy scale, the measured
~7–10× decline from z=0.5 to 3.5, and the measured mass ordering (the
first executed version had none of these — see the intro provenance note).
Group amplitudes scale as

$$\log_{10} A_g(z, M_*) = \log_{10} A_{g,0} + 0.35\,(\log M_* - 10.5) + e_g\,[\hat s(z, M_*) + 9.0]$$

with ŝ the Speagle+2014 main-sequence log sSFR — at fixed M\*, higher z
means higher sSFR, so the spectrum seen by a mass bin's high-z (short
rest-λ) points is genuinely different from its low-z one. The injected
per-group slopes $e_g$ make the 7.7+8.6 µm complex strengthen fastest
(+0.9 dex/dex) and 11.3 µm the slowest (+0.1) — i.e. both amplitude *and*
ratio evolution, in the ionized-over-neutral direction."""
)

code(
    r"""# Truth tables: group amplitudes at the mass pivot and the injected slopes
print("Injected per-group truth (at log M*=10.5, sSFR pivot -9.0):")
print(pd.DataFrame({
    "group": GROUP_NAMES, "peak/continuum": RATIO_TARGETS,
    "r_g (vs 7.7+8.6)": np.round(RATIO_TARGETS / RATIO_TARGETS[0], 3),
    "e_g (dex/dex sSFR)": E_G_TRUE,
}).to_string(index=False))
print(f"\nReparameterized: eta_A = {ETA_A_TRUE}, eta_g =", ETA_G_TRUE)

# Rest-frame spectra at three redshifts, lowest vs highest mass bin
lam = np.linspace(4, 25, 800)
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, (mc, lab) in zip(axes, [(MASS_CENTERS[0], MASS_LABELS[0]),
                                (MASS_CENTERS[-1], MASS_LABELS[-1])]):
    for z, colr in [(0.7, "#86b6ef"), (1.8, "#3987e5"), (3.0, "#1c5cab")]:
        spec = truth.rest_spectrum(lam[None, :], {"log_M_star": mc},
                                   z=np.array([z]))[0]
        ax.plot(lam, spec, color=colr, lw=1.6, label=f"z={z} population")
    ax.set_xlabel("rest wavelength [um]")
    ax.set_title(f"log M* = {lab}")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("rest f_nu [pre-envelope, arb]")
axes[0].legend(fontsize=8)
fig.suptitle("Evolving truth: same mass bin, different epochs -> different PAH spectrum", y=1.02)
fig.tight_layout()

# What the bandpasses see: in-band OBSERVED flux vs z (envelope included)
zg = np.linspace(0.4, 3.6, 500)
mass_colors = plt.cm.Blues(np.linspace(0.45, 0.95, len(MASS_CENTERS)))
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
for ax, band in zip(axes, ("MIPS_24", "MIPS_70")):
    for m, (mc, lab) in enumerate(zip(MASS_CENTERS, MASS_LABELS)):
        ax.plot(zg, truth.band_flux_curve(zg, band, {"log_M_star": mc}),
                color=mass_colors[m], lw=1.6, label=f"log M* {lab}")
        ax.plot(zg, cont_truth.band_flux_curve(zg, band, {"log_M_star": mc}),
                color=mass_colors[m], lw=0.9, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("redshift")
    ax.set_title(band.replace("_", " ") + " um (dotted: continuum only)")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("in-band flux [mJy]")
axes[0].legend(fontsize=8)
fig.tight_layout()
plt.show()
"""
)

# ----------------------------------------------------------------------------
md(
    r"""## 2. Stack it: dithered simulation at the real depth

`simulate_dithered_fluxes` draws 12,000 sources (≈140 per z-bin per mass
bin — the real Tier-B occupancy), scatters photo-z's (σ_z0=0.02, 2%
catastrophic outliers), assigns them to the 3 staggered runs, and adds
noise from the bootstrap-calibrated shared-source covariance. Because the
noise is flux-independent while the envelope dims the signal, the
per-point SNR now falls with z exactly like the real data (≈15–19 at z<1
down to ~1–4 at z>3). The fits get the *perfect* continuum baseline
(f24_cold/f70_cold from the continuum-only twin) — baseline error is
studied separately in §6.

Note the fitter weights points diagonally (per-point σ) while the injected
noise is run-to-run correlated (staggered runs share sources), the same
situation as the real fits — a mild χ²_red elevation is expected."""
)

code(
    r"""sim = simulate_dithered_fluxes(
    scheme, truth, n_total=12_000, sigma_z0=SIGMA_Z0, f_cat=F_CAT, seed=rng_seed,
)
df = add_cold_baselines(sim["df"], scheme, cont_truth)

print(df.groupby("prop_bin_id")["n_sources"].median().rename("median n/bin"))
for zlo, zhi in [(0.5, 1.0), (1.5, 2.2), (3.0, 3.5)]:
    s = df[(df["z_mid"] >= zlo) & (df["z_mid"] < zhi)]
    print(f"z {zlo}-{zhi}: median SNR24 = {np.nanmedian(s['MIPS_24']/s['MIPS_24_err']):5.1f}   "
          f"median f24 = {np.nanmedian(s['MIPS_24']):.3f} mJy   "
          f"(real 3-fold: ~15/0.2, ~11/0.13, ~3/0.05)")

# Raw pseudo-tomography: the simulated stacked f24 vs z per mass bin
fig, ax = plt.subplots(figsize=(8.5, 4.5))
for m, (mc, lab) in enumerate(zip(MASS_CENTERS, MASS_LABELS)):
    sub = df[df["prop_bin_id"] == m].sort_values("z_mid")
    ax.errorbar(sub["z_mid"], sub["MIPS_24"], yerr=sub["MIPS_24_err"],
                fmt="o", ms=3, color=mass_colors[m], lw=0.8, alpha=0.8,
                label=f"log M* {lab}")
    ax.plot(zg, truth.band_flux_curve(zg, "MIPS_24", {"log_M_star": mc}),
            color=mass_colors[m], lw=1.0, alpha=0.6)
ax.set_yscale("log")
ax.set_xlabel("redshift")
ax.set_ylabel("stacked MIPS 24 um flux [mJy]")
ax.set_title("Simulated dithered stacks (points) vs noiseless truth (lines)")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)
fig.tight_layout()
plt.show()
"""
)

# ----------------------------------------------------------------------------
md(
    r"""## 3. Point estimates first: a static template is biased under evolution

Before the MCMC: the alternating-WLS point fits, both with the
envelope-aware feature term (`feature_envelope="baseline"`: features dim
with the source like the continuum; α is then an EW-like
feature-to-continuum ratio). The static shared-ratio model (no evolution)
against the full evolving one."""
)

code(
    r"""model = PAHSpectrumModel(feature_groups=GROUPS, sigma_z0=SIGMA_Z0, f_cat=F_CAT)

pt_static = model.fit_evolving(df, scheme=scheme, evolve_amp=False,
                               evolve_ratios=False, feature_envelope="baseline")
pt_evolve = model.fit_evolving(df, scheme=scheme, feature_envelope="baseline")

print(f"static  : chi2_red = {pt_static['chi2_red']:.2f}")
print(f"evolving: chi2_red = {pt_evolve['chi2_red']:.2f}   "
      f"eta_A = {pt_evolve['eta_amp']:.2f} (truth {ETA_A_TRUE})")
print("\nPer-bin PAH/continuum amplitude at the pivot (A_pah = alpha/C):")
print(pd.DataFrame({
    "log M*": MASS_LABELS,
    "A_pah static": np.round(pt_static["A_pah"], 3),
    "A_pah evolving": np.round(pt_evolve["A_pah"], 3),
}).to_string(index=False))
"""
)

# ----------------------------------------------------------------------------
md(
    r"""## 4. The MCMC flexibility ladder

`fit_evolving_mcmc` samples θ = [η_A, η_g, log₁₀ r-block] with (C_m, α_m)
profiled analytically per step; Gaussian prior (σ=1 dex) on every slope,
flat prior on log r within ±2. Rungs:

| rung | ratios | evolution | dims | question |
|---|---|---|---|---|
| L1 | shared | η_A only | 5 | is amplitude evolution alone enough? |
| L2 | shared | η_A + η_g | 9 | the `fit_evolving` model |
| L2n | shared | η_A + η_g, **constant-flux features** | 9 | what does ignoring the envelope do? |
| L2b | shared | η_A + η_g, **24 µm only** | 9 | what does 70 µm buy? |
| L2d | shared | η_A + η_g, **33× depth** | 9 | the systematics floor |
| L3 | **per-bin** | none | 16 | §1a-style static ratio freedom |
| L4 | **per-bin** | η_A + η_g | 21 | everything at once |

All rungs except L2n use `feature_envelope="baseline"`. Recovery is judged
on the per-group *total* slopes $e_g = η_A + η_g$ and ratios $r_g$
(evaluated at the fit's sSFR pivot), which are what the data constrain —
the η_A/η_g split is only defined relative to the reference group."""
)

code(
    r"""mcmc_common = dict(scheme=scheme, eta_prior_sigma=1.0, n_burn=300, seed=1)
ENV = dict(feature_envelope="baseline")
results = {}

results["L1 amp-only"] = model.fit_evolving_mcmc(
    df, evolve_ratios=False, n_walkers=32, n_steps=800, **ENV, **mcmc_common)
results["L2 amp+ratios"] = model.fit_evolving_mcmc(
    df, n_walkers=32, n_steps=800, **ENV, **mcmc_common)
results["L2n no-envelope"] = model.fit_evolving_mcmc(
    df, n_walkers=32, n_steps=800, **mcmc_common)

model24 = PAHSpectrumModel(feature_groups=GROUPS, bands=("MIPS_24",),
                           sigma_z0=SIGMA_Z0, f_cat=F_CAT)
results["L2b 24-only"] = model24.fit_evolving_mcmc(
    df, baseline_cols={"MIPS_24": "f24_cold"},
    n_walkers=32, n_steps=800, **ENV, **mcmc_common)

results["L3 per-bin r"] = model.fit_evolving_mcmc(
    df, evolve_amp=False, evolve_ratios=False, per_bin_ratios=True,
    n_walkers=48, n_steps=800, **ENV, **mcmc_common)
results["L4 per-bin r + evol"] = model.fit_evolving_mcmc(
    df, per_bin_ratios=True, n_walkers=48, n_steps=1000, **ENV, **mcmc_common)

for name, res in results.items():
    print(f"{name:22s} ndim={res['chain'].shape[1]:2d}  acc={res['acceptance_fraction']:.2f}  "
          f"chi2_red={res['chi2_red']:.2f}  eta_A={res['eta_amp']:+.2f}+/-{res['eta_amp_err']:.2f}")

# Continuum-anchor sanity for the workhorse rung: the profiled C_m should sit
# at each bin's median 24 um cold-baseline flux (the normalization scalar),
# and A_pah = alpha/C_m should rise with mass (beta_mass = +0.35 injected;
# expected Delta log A_pah ~ 0.37 dex over the 1.05 dex mass range).
res2 = results["L2 amp+ratios"]
print("\nL2 profiled continuum & amplitude per mass bin:")
print(pd.DataFrame({
    "log M*": MASS_LABELS,
    "C_m": np.round(res2["C_m"], 4),
    "median f24_cold": np.round(
        df.groupby("prop_bin_id")["f24_cold"].median().to_numpy(), 4),
    "A_pah": np.round(res2["A_pah"], 2),
    "A_pah_err": np.round(res2["A_pah_err"], 2),
}).to_string(index=False))
"""
)

code(
    r"""# The deep run: same model, 33x more sources -> per-point SNR ~10x higher.
# Formal errors collapse; what remains is the kernel-approximation floor
# (the fit evaluates the evolving amplitude and the envelope at each bin's
# z_mid, while the truth evolves continuously across each bin's
# photo-z-smeared membership).
sim_deep = simulate_dithered_fluxes(
    scheme, truth, n_total=400_000, sigma_z0=SIGMA_Z0, f_cat=F_CAT, seed=rng_seed,
)
df_deep = add_cold_baselines(sim_deep["df"], scheme, cont_truth)
results["L2d deep"] = model.fit_evolving_mcmc(
    df_deep, n_walkers=32, n_steps=800, **ENV, **mcmc_common)

summary = pd.concat(
    [rung_summary(name, res) for name, res in results.items()],
    ignore_index=True,
)
print(summary.to_string(index=False))
"""
)

code(
    r"""# Corner plot of the workhorse rung (L2): slopes + shared log-ratios,
# with the truth (at the fit pivot) overlaid.
import corner

res2 = results["L2 amp+ratios"]
r_exp, e_true = truth_at_pivot(res2)
truth_vec = ([ETA_A_TRUE] + list(ETA_G_TRUE[1:]) + list(np.log10(r_exp[1:])))
fig = corner.corner(
    res2["chain"], labels=res2["names"], truths=truth_vec,
    truth_color="#e34948", show_titles=True, title_fmt=".2f",
    quantiles=[0.16, 0.5, 0.84], label_kwargs={"fontsize": 9},
)
fig.suptitle("L2 posterior vs truth (red): shared ratios + amplitude/ratio evolution",
             y=1.02, fontsize=12)
plt.show()
"""
)

# ----------------------------------------------------------------------------
md(
    r"""### 4c. Why does the truth sit off-center in the corners?

The red truth lines above sit systematically off the posterior medians
(more so in the first, pre-envelope version of this notebook, where the
per-point SNR was ~5× higher). Three contributors, in order:

1. **Injection-vs-kernel mismatch** (dominant): the truth evolves
   *continuously* — every source carries its own $10^{e_g \hat s(z)}$ and
   envelope at its true z, and photo-z scattering mixes them into each bin
   — while the fit modulates the photo-z-smeared kernel by a single factor
   evaluated at the bin's z_mid. That approximation error is a fixed
   percent-level model bias, so posteriors tighten onto the wrong point as
   SNR grows (the L2d rung).
2. **Prior shrinkage**: the σ=1 dex Gaussian slope prior pulls η toward 0.
3. **Leakage from the railed 16.4+17.0 group** into correlated parameters.

The check below separates (1) from a sampler/likelihood bug: draw a
synthetic replica FROM the fitted model itself (posterior-median parameters
+ per-point noise) and refit it. If the machinery is self-consistent, the
generating parameters should sit centered (|pull| ~ 1); the injected-truth
offsets then measure the kernel floor, not a code problem. The fix, if the
floor ever matters at real-data SNR (it does not — statistical errors are
~10× larger), is to integrate the sSFR modulation and envelope over p(z)
inside the kernel instead of evaluating at z_mid."""
)

code(
    r"""# Posterior-predictive replica: simulate from the fitted L2 model, refit,
# and compare pulls (recovered - generating) / sigma against the real
# injection run's pulls (recovered - truth) / sigma.
dec_med = evolving_flux_decomposition(results["L2 amp+ratios"], n_draws=10, seed=3)
piv = dec_med.pivot_table(index=["prop_bin_id", "z_mid"], columns="band",
                          values="total").reset_index()
piv.columns = ["prop_bin_id", "z_mid"] + [f"model_{c}" for c in piv.columns[2:]]
df_ppc = df.merge(piv, on=["prop_bin_id", "z_mid"], how="left")
rng_ppc = np.random.default_rng(5)
for band in ("MIPS_24", "MIPS_70"):
    mcol = f"model_{band}"
    if mcol not in df_ppc:
        continue
    tot = df_ppc[mcol].to_numpy()
    err = df_ppc[f"{band}_err"].to_numpy()
    okp = np.isfinite(tot) & np.isfinite(err)
    newf = df_ppc[band].to_numpy().copy()
    newf[okp] = tot[okp] + rng_ppc.normal(0.0, err[okp])
    df_ppc[band] = newf

res_ppc = model.fit_evolving_mcmc(df_ppc, n_walkers=32, n_steps=800, **ENV, **mcmc_common)

gen = res2  # the generating (posterior-median) parameters
rows = []
for g in range(len(GROUP_NAMES)):
    e_gen = gen["eta_amp"] + gen["eta_ratio"][g]
    e_ppc = res_ppc["eta_amp"] + res_ppc["eta_ratio"][g]
    e_err = np.hypot(res_ppc["eta_amp_err"], res_ppc["eta_ratio_err"][g]) or res_ppc["eta_amp_err"]
    e_tru, r_tru = E_G_TRUE[g], truth_at_pivot(res2)[0][g]
    e_inj = res2["eta_amp"] + res2["eta_ratio"][g]
    e_ierr = np.hypot(res2["eta_amp_err"], res2["eta_ratio_err"][g]) or res2["eta_amp_err"]
    rows.append({
        "group": GROUP_NAMES[g],
        "pull vs gen (PPC)": round((e_ppc - e_gen) / max(e_err, 1e-9), 1),
        "pull vs truth (injection)": round((e_inj - e_tru) / max(e_ierr, 1e-9), 1),
    })
print(pd.DataFrame(rows).to_string(index=False))
print("\nPPC pulls ~ O(1) -> sampler + profiled likelihood are self-consistent;")
print("larger injection pulls = the z_mid-vs-p(z) kernel approximation floor.")
"""
)

# ----------------------------------------------------------------------------
md(
    r"""## 5. Headline: the stacked f₂₄(z) reconstructed as shaded line contributions

Per mass bin, the simulated stacked fluxes vs z, the posterior-median cold
baseline (gray), and each PAH feature group's shaded contribution stacked
on top — the posterior decomposition of the tomographic curve into the
lines that produced it, now at the measured flux scale and z-shape.
Black band: 68% credible interval on the total. Red dashed: the noiseless
truth. Top axis: the rest wavelength the bandpass center probes. (§7 ends
with the same figure made from the real data.)"""
)

code(
    r"""dec = evolving_flux_decomposition(results["L2 amp+ratios"], n_draws=120, seed=2)

truth_curves = {
    m: (zg, truth.band_flux_curve(zg, "MIPS_24", {"log_M_star": mc}))
    for m, mc in enumerate(MASS_CENTERS)
}
fig = plot_pah_flux_decomposition(
    dec, band="MIPS_24",
    mass_labels=[f"log M* {mass_lab}" for mass_lab in MASS_LABELS],
    truth_curves=truth_curves,
    save_path="pah_evolving_mcmc_f24_decomposition.png",
)
plt.show()
"""
)

code(
    r"""truth_curves_70 = {
    m: (zg, truth.band_flux_curve(zg, "MIPS_70", {"log_M_star": mc}))
    for m, mc in enumerate(MASS_CENTERS)
}
fig = plot_pah_flux_decomposition(
    dec, band="MIPS_70",
    mass_labels=[f"log M* {mass_lab}" for mass_lab in MASS_LABELS],
    truth_curves=truth_curves_70,
    logy=False,  # low per-point 70 um SNR: negative fluxes, linear axis
    save_path="pah_evolving_mcmc_f70_decomposition.png",
)
plt.show()
"""
)

# ----------------------------------------------------------------------------
md(
    r"""## 6. Baseline stress: a tilted continuum masquerades as evolution

The branch-5/6 lesson (A_pah is strongly α-sensitive; a wrong Wien slope
re-tilts everything) applies here too: re-run L2 with the 24 µm baseline
tilted by $(1+z)^{\pm 0.3}$. With `feature_envelope="baseline"` the tilt
now perturbs both the continuum subtraction *and* the feature dimming —
exactly the coupling a wrong α would produce on real data."""
)

code(
    r"""stress = {}
for tilt in (+0.3, -0.3):
    df_t = df.copy()
    df_t["f24_cold"] = df_t["f24_cold"] * (1 + df_t["z_mid"]) ** tilt
    stress[tilt] = model.fit_evolving_mcmc(
        df_t, n_walkers=32, n_steps=600, **ENV, **mcmc_common)

rows = []
for tag, res in [("no tilt", results["L2 amp+ratios"]),
                 ("+0.3 tilt", stress[+0.3]), ("-0.3 tilt", stress[-0.3])]:
    rows.append({
        "baseline": tag,
        "eta_A": f"{res['eta_amp']:+.2f} +/- {res['eta_amp_err']:.2f}",
        "e(6.2)": f"{res['eta_amp']+res['eta_ratio'][1]:+.2f}",
        "e(11.3)": f"{res['eta_amp']+res['eta_ratio'][2]:+.2f}",
        "r(6.2)": f"{res['r'][1]:.2f}",
        "chi2_red": round(res["chi2_red"], 2),
    })
print(pd.DataFrame(rows).to_string(index=False))
print(f"\n(truth: eta_A = {ETA_A_TRUE:+.2f}, e(6.2) = {E_G_TRUE[1]:+.2f}, "
      f"e(11.3) = {E_G_TRUE[2]:+.2f})")
"""
)

# ----------------------------------------------------------------------------
md(
    r"""## 7. Real data: is a non-evolving template dramatically worse?

The same machinery on the 3 disjoint K-fold COSMOS2020 stacks
(`cosmos2020_PAH_split{0,1,2}of3`, run dates 20260630_{193627, 211122,
222635}). Data-handling cells are copied verbatim-or-minimally-adapted from
`2026-07-01-pah-forward-model-letter.ipynb` so results are comparable;
fluxes converted to mJy; feature groups follow the letter convention
(6.2 | 7.7+8.6 | 12.7, 11.3 blind) but reordered so the reference group is
the 7.7+8.6 µm complex (the §4 identifiability lesson).

Design of the comparison:
- **Static vs evolving**, each with and without the envelope-aware feature
  term — the no-envelope evolving fit is (up to this branch's fixes) the
  branch-5 configuration whose η_A railed to −2.37 and was declared an
  artifact.
- Real-data χ²_red is scatter-inflated (branch-5: galaxy-to-galaxy PAH
  scatter, NOT baseline error), so raw Δχ² overstates significance. We
  report Δχ² both raw and rescaled by the static fit's χ²_red, plus BIC,
  **and** calibrate against a scatter-null: the §2 simulation with ZERO
  injected evolution and extra per-point scatter tuned to the real static
  χ²_red — how big a Δχ² and |η_A| does pure scatter produce?"""
)

code(
    r"""# ── Load the 3 K-fold stacking runs (verbatim-adapted from the letter nb) ──
import os
import logging

from simstack4.wrapper import SimstackWrapper
from simstack4.plots import _extract_pop_type
from simstack4.greybody import Greybody as _Greybody

logging.getLogger("simstack4").setLevel(logging.WARNING)

path_json = os.path.join(os.environ["PICKLESPATH"], "simstack", "stacked_flux_densities")
RUN_DATES = {
    0: "20260630_193627",   # cosmos2020_PAH_split0of3, offset 0.0000
    1: "20260630_211122",   # cosmos2020_PAH_split1of3, offset 0.0375
    2: "20260630_222635",   # cosmos2020_PAH_split2of3, offset 0.0750
}
MASS_BINS = [
    (10.0, 10.5, "C0", "10.0 < logM* < 10.5"),
    (10.5, 10.8, "C1", "10.5 < logM* < 10.8"),
    (10.8, 11.1, "C2", "10.8 < logM* < 11.1"),
    (11.1, 12.0, "C3", "logM* > 11.1"),
]
ANALYSIS_KWARGS = dict(
    use_mcmc=False, temperature_prior="schreiber", snr_high=5.0, snr_low=2.0,
    inflation_factors={24: 10000, 70: {(0.0, 0.8): 1.0, (0.8, 99.0): 10000}},
    use_covariance=True, use_pah=False,
)

WRAPPERS = []
for k in range(3):
    w = SimstackWrapper()
    w.load_stacking_results(
        os.path.join(path_json, f"cosmos20_stacking_{RUN_DATES[k]}.json"))
    w.run_analysis_only(**ANALYSIS_KWARGS)
    WRAPPERS.append(w)
print("Loaded 3 disjoint-fold stacking runs:", RUN_DATES)
"""
)

code(
    r'''# ── Tomographic DataFrame + smoothed baseline (verbatim from the letter nb) ──
def build_pah_spectrum_df(wrappers, mass_bins, split_filter=None, min_tier="B"):
    """Extract raw stacked fluxes and greybody Wien-side extrapolations.

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
            for key, val in props.items():
                if "mass" in key.lower() and "delta" not in key.lower():
                    stellar_mass = float(val)
                    break
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
    dfx = (pd.DataFrame(rows)
             .sort_values(["prop_bin_id", "run_id", "z_mid"])
             .reset_index(drop=True))
    return dfx


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


df_real = build_pah_spectrum_df(WRAPPERS, MASS_BINS, split_filter=[0], min_tier="C")
print(f"Built DataFrame: {len(df_real)} points, {df_real['run_id'].nunique()} runs, "
      f"{df_real['prop_bin_id'].nunique()} mass bins")
df_real = smooth_baseline(df_real)
# wrapper fluxes are in Jy; work in mJy like the simulation
for c in ["MIPS_24", "MIPS_24_err", "MIPS_70", "MIPS_70_err",
          "f24_cold", "f70_cold"]:
    df_real[c] = 1e3 * df_real[c]

# Cross-check the hardcoded Sec 1 envelope against the live data
okc = np.isfinite(df_real["f24_cold"]) & (df_real["f24_cold"] > 0) & (df_real["z_mid"] < 3.6)
dd = df_real[okc]
Xc = np.column_stack([np.ones(len(dd)), dd["z_mid"], dd["z_mid"]**2,
                      dd["log_M_star"] - 10.75])
cf, *_ = np.linalg.lstsq(Xc, np.log10(dd["f24_cold"]), rcond=None)
print("\nenvelope refit :", np.round(cf, 4))
print("Sec 1 hardcoded:", np.round(ENV_COEF, 4))
'''
)

code(
    r"""# ── Static vs evolving on the pooled real data (point fits) ────────────────
GROUPS_REAL = [[1, 2], [0], [4]]   # 7.7+8.6 (reference) | 6.2 | 12.7; 11.3 blind
GROUP_NAMES_REAL = ["7.7+8.6", "6.2", "12.7"]
model_real = PAHSpectrumModel(feature_groups=GROUPS_REAL, sigma_z0=0.01, f_cat=0.03)

VARIANTS = [
    ("static, envelope",       dict(evolve_amp=False, evolve_ratios=False,
                                    feature_envelope="baseline")),
    ("evolve amp, envelope",   dict(evolve_ratios=False, feature_envelope="baseline",
                                    eta_prior_sigma=1.0)),
    ("evolve amp+rat, envelope", dict(feature_envelope="baseline",
                                      eta_prior_sigma=1.0)),
    ("static, no envelope",    dict(evolve_amp=False, evolve_ratios=False)),
    ("evolve amp, no envelope", dict(evolve_ratios=False, eta_prior_sigma=1.0)),
]
real_fits = {}
rows = []
G_R = len(GROUPS_REAL)
for name, kwargs in VARIANTS:
    r = model_real.fit_evolving(df_real, **kwargs)
    real_fits[name] = r
    n_eta = (0 if "static" in name else 1) + ((G_R - 1) if "amp+rat" in name else 0)
    k = 2 * len(r["valid"]) + (G_R - 1) + n_eta
    n = r["dof"] + k
    rows.append({
        "model": name, "k": k, "chi2": round(r["chi2"], 1),
        "chi2_red": round(r["chi2_red"], 3),
        "eta_A": (f"{r['eta_amp']:+.3f}+/-{r['eta_amp_err']:.3f}"
                  if "static" not in name else "--"),
        "BIC": round(r["chi2"] + k * np.log(n), 1),
    })
tab = pd.DataFrame(rows)
print(tab.to_string(index=False))

chi2_static = real_fits["static, envelope"]["chi2"]
chi2red_static = real_fits["static, envelope"]["chi2_red"]
for nm in ("evolve amp, envelope", "evolve amp+rat, envelope"):
    dchi2 = chi2_static - real_fits[nm]["chi2"]
    print(f"\n{nm}: raw Delta chi2 = {dchi2:.1f}"
          f"   scatter-rescaled (/{chi2red_static:.2f}) = {dchi2/chi2red_static:.1f}")
print("\nNo-envelope pair: Delta chi2 =",
      round(real_fits['static, no envelope']['chi2']
            - real_fits['evolve amp, no envelope']['chi2'], 1),
      "-> without the envelope the dimming saturates the feature term and")
print("evolution has little left to explain (the branch-5 configuration).")
"""
)

md(
    r"""**Reading the table**: with the envelope-aware feature term, one
evolution parameter is worth a large Δχ² — but real-data χ²_red is
scatter-inflated, and branch-5 showed a controlled sim with χ²_red≈6
scatter can produce a *spurious* η_A ≈ +1.5 from zero truth. So before
calling non-evolving "dramatically worse", calibrate the null: the same
§2 simulation with ZERO injected evolution, extra lognormal per-point
scatter tuned to reproduce the real static χ²_red, fit with the same two
models. The spread of null Δχ² and |η_A| is what scatter alone buys."""
)

code(
    r"""# ── Scatter-null calibration ────────────────────────────────────────────────
truth_null = TruthSpectrum(
    feature_groups=GROUPS, amp0=RATIO_TARGETS * MIR_PLAW_AMP,
    beta_mass=BETA_MASS, mir_plaw_amp=MIR_PLAW_AMP, mir_plaw_slope=0.0,
    flux_envelope=flux_envelope,
)  # eta == 0: static truth


def null_trial(seed, sigma_scat):
    simn = simulate_dithered_fluxes(
        scheme, truth_null, n_total=12_000, sigma_z0=SIGMA_Z0, f_cat=F_CAT,
        seed=seed)
    dfn = add_cold_baselines(simn["df"], scheme, cont_truth)
    rngn = np.random.default_rng(seed + 1000)
    for band in ("MIPS_24", "MIPS_70"):
        dfn[band] = dfn[band] * 10.0 ** rngn.normal(0.0, sigma_scat, len(dfn))
    st = model.fit_evolving(dfn, scheme=scheme, evolve_amp=False,
                            evolve_ratios=False, feature_envelope="baseline")
    ev = model.fit_evolving(dfn, scheme=scheme, evolve_ratios=False,
                            feature_envelope="baseline", eta_prior_sigma=1.0)
    return st, ev


# Tune sigma_scat so the null static fit's chi2_red matches the real one.
# Average a few seeds per grid value: the seed-to-seed chi2_red spread is
# large enough (~+/-0.5) to fool a single-trial match.
target = chi2red_static
best = None
for s in (0.05, 0.06, 0.07, 0.08, 0.10, 0.13):
    m = float(np.mean([null_trial(200 + 7 * j, s)[0]["chi2_red"]
                       for j in range(3)]))
    if best is None or abs(m - target) < abs(best[0] - target):
        best = (m, s)
mean_null, sigma_scat = best
print(f"real static chi2_red = {target:.2f} -> sigma_scat = {sigma_scat} dex "
      f"(null 3-seed mean chi2_red = {mean_null:.2f})")

null_rows = []
for sd in range(20, 28):
    st, ev = null_trial(sd, sigma_scat)
    null_rows.append({"seed": sd, "chi2_red_static": round(st["chi2_red"], 2),
                      "eta_A_spurious": round(ev["eta_amp"], 3),
                      "dchi2_spurious": round(st["chi2"] - ev["chi2"], 1)})
null_tab = pd.DataFrame(null_rows)
print(null_tab.to_string(index=False))

dchi2_real = chi2_static - real_fits["evolve amp, envelope"]["chi2"]
print(f"\nnull:  |eta_A| up to {null_tab['eta_A_spurious'].abs().max():.2f}, "
      f"Delta chi2 up to {null_tab['dchi2_spurious'].max():.1f}")
print(f"real:  eta_A = {real_fits['evolve amp, envelope']['eta_amp']:+.3f}, "
      f"Delta chi2 = {dchi2_real:.1f}")
"""
)

code(
    r"""# ── Per-fold consistency (disjoint galaxies -> independent checks) ─────────
fold_rows = []
for fi, w in enumerate(WRAPPERS):
    dff = build_pah_spectrum_df([w], MASS_BINS, split_filter=[0], min_tier="C")
    if len(dff) < 8:
        continue
    dff = smooth_baseline(dff)
    for c in ["MIPS_24", "MIPS_24_err", "MIPS_70", "MIPS_70_err",
              "f24_cold", "f70_cold"]:
        dff[c] = 1e3 * dff[c]
    st = model_real.fit_evolving(dff, evolve_amp=False, evolve_ratios=False,
                                 feature_envelope="baseline")
    ev = model_real.fit_evolving(dff, evolve_ratios=False,
                                 feature_envelope="baseline", eta_prior_sigma=1.0)
    fold_rows.append({
        "fold": fi, "n_pts": st["dof"] + 2 * len(st["valid"]) + (G_R - 1),
        "chi2_red static": round(st["chi2_red"], 2),
        "chi2_red evolving": round(ev["chi2_red"], 2),
        "dchi2": round(st["chi2"] - ev["chi2"], 1),
        "eta_A": round(ev["eta_amp"], 3),
    })
fold_tab = pd.DataFrame(fold_rows)
print(fold_tab.to_string(index=False))
etas = fold_tab["eta_A"].to_numpy()
print(f"\nfold-ensemble eta_A = {etas.mean():+.3f} +/- "
      f"{etas.std(ddof=1)/np.sqrt(len(etas)):.3f} (mean +/- scatter/sqrt(3))")
"""
)

code(
    r"""# ── Real-data decomposition figure (MCMC on the winning model) ─────────────
res_real = model_real.fit_evolving_mcmc(
    df_real, feature_envelope="baseline", eta_prior_sigma=1.0,
    n_walkers=32, n_steps=800, n_burn=300, seed=2)
print(f"real-data MCMC: chi2_red = {res_real['chi2_red']:.2f}, "
      f"eta_A = {res_real['eta_amp']:+.3f} +/- {res_real['eta_amp_err']:.3f}")
print("r =", np.round(res_real["r"], 3), " A_pah =", np.round(res_real["A_pah"], 2))

dec_real = evolving_flux_decomposition(res_real, n_draws=120, seed=4)
fig = plot_pah_flux_decomposition(
    dec_real, band="MIPS_24",
    mass_labels=[lbl for *_, lbl in MASS_BINS],
    save_path="pah_evolving_mcmc_f24_decomposition_real.png",
)
plt.show()
"""
)

# ----------------------------------------------------------------------------
md(
    r"""## 8. Takeaways

Qualitative statements verified on the executed run; the recap cell below
reprints the machine-readable summary.

**Simulation (flexibility ladder, §3–§6):**

1. **The reference group defines what "amplitude evolution" means.** With
   the reference anchored on the 7.7+8.6 µm complex, its slope
   ($e_{7.7+8.6}$, reported as η_A) is recovered at the real depth and
   noise; anchored on the weak 6.2 µm instead, η_A floats (development
   check). Real-data η_A claims should be phrased as "the 7.7+8.6 µm
   amplitude slope".
2. **Per-group total slopes $e_g$ are the identifiable quantities**, and
   only for groups the bandpasses sweep with leverage; 16.4+17.0 always
   rails. Per-bin ratio freedom (L3/L4) converges but inflates weak-group
   posteriors.
3. **Ignoring the flux envelope (L2n) corrupts the slopes**: the
   constant-flux feature term forces the ~7–10× source dimming into the
   evolution parameters. The same mechanism operated on the branch-5
   real-data fits (§7).
4. **The truth sits off-center at high SNR because of the z_mid-vs-p(z)
   kernel approximation, not a code bug** (§4c: a posterior-predictive
   replica refits centered). The upgrade path — integrate the sSFR
   modulation and envelope over p(z) inside the kernel — only matters well
   above the real data's SNR.

**Real data (§7):**

5. **Yes, the non-evolving template is dramatically worse in fit terms** —
   with the envelope-aware feature term, one shared amplitude slope buys a
   Δχ² far beyond both the parameter cost and the scatter-null calibration,
   consistently in all three independent folds, and η_A comes out
   *positive* (feature-to-continuum ratio rising with sSFR along each mass
   bin) — opposite in sign to the railed branch-5 value, which the L2n rung
   identifies as envelope absorption.
6. **Interpretation stays guarded**: the scatter-null bounds how much of
   the slope pure per-point scatter can fake; anything systematic that
   tilts feature-vs-continuum with z (α_wien, T(z) relation residuals — §6)
   can still contribute to η_A. "Evolution required to fit the stacks" is
   the defensible claim; its physical decomposition needs the α systematic
   quoted alongside, exactly as branch-5/6 concluded for the amplitudes.
"""
)

code(
    r"""# Machine-readable recap: every rung, every group.
recap = summary.copy()
recap["e_pull"] = np.where(
    recap["e_err"] > 0, (recap["e_rec"] - recap["e_true"]) / recap["e_err"], np.nan
).round(1)
print(recap.to_string(index=False))
"""
)

nb["cells"] = cells
out = "notebooks/2026-07-02-pah-evolving-template-mcmc-simulation.ipynb"
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out} ({len(cells)} cells)")
