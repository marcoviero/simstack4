#!/usr/bin/env python3
"""
Figure 3: Forward model credibility — injection → recovery, then real data.

Two-row × three-column layout:

  Row A (simulation, top):
    A1 — synthetic f₂₄/f_peak pseudo-spectra per mass bin with injected α
    A2 — detrended residuals with forward model overlay (the bumps)
    A3 — injected vs recovered α scatter with 1σ bars (proof of recovery)

  Row B (real data, bottom):
    B1, B2, B3 — same panels applied to the four measured stacking runs

Credibility argument: the model recovers known amplitudes in simulation (Row A);
the same fitter applied to data (Row B) is the measurement.

Animation: Row A only — A2 and A3 reveal a convergence story: points appear
 (A2) and the α estimates update (A3) as the optimizer iterates from zero.
Row B is always static. This is deliberately NOT "fit converging through data
points" — it is injection → recovery in simulation.

Deliverables written to ../figures/:
  pah_fig3.mp4          H.264, 30 fps
  pah_fig3_paper.png    static two-row layout for paper
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from simstack4.pah_model import PAHModel, PAH_FEATURES
from simstack4.pah_bandpass import get_bandpass

# ── Shared visual language ────────────────────────────────────────────────────
FEAT_COLORS  = ['C0', 'C1', 'C2', 'C3', 'C4']
BP_COLOR     = '#3b6ea5'
FIG_BG       = 'white'
LABEL_FS     = 10
TICK_FS      = 8

blues          = matplotlib.colormaps['Blues']
FEATURE_GROUPS = [[0], [1, 2], [4]]   # 6.2 | 7.7+8.6 | 12.7

FPS     = 24
DPI     = 120
N_ANIM  = 60    # convergence frames for Row A animation

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
PICKLES_DIR = os.path.expandvars('$PICKLESPATH/simstack/stacked_flux_densities')

RUN_JSON = [
    os.path.join(PICKLES_DIR, 'cosmos25_stacking_20260612_190116.json'),  # offset 0.000
    os.path.join(PICKLES_DIR, 'cosmos25_stacking_20260612_180838.json'),  # offset 0.0375
    os.path.join(PICKLES_DIR, 'cosmos25_stacking_20260612_164943.json'),  # offset 0.0750
    os.path.join(PICKLES_DIR, 'cosmos25_stacking_20260612_160940.json'),  # offset 0.1125
]

ANALYSIS_KWARGS = dict(
    use_mcmc=False,
    temperature_prior='schreiber',
    snr_high=5.0,
    snr_low=2.0,
    inflation_factors={
        24: 10000,
        70: {(0.0, 0.8): 1.0, (0.8, 99.0): 10000},
    },
    use_covariance=True,
    use_pah=False,
)

# ── Physics setup ─────────────────────────────────────────────────────────────
pah = PAHModel()
bp  = get_bandpass('MIPS_24')

LAM_RANGE = (5.5, 14.5)
Z_RANGE   = (0.5, 3.5)

z_fine    = np.linspace(*Z_RANGE, 500)
feat_at_z = pah.feature_spectrum(bp.lam_fine[:, None] / (1 + z_fine))
T_fine    = np.trapezoid(feat_at_z * bp.resp_fine[:, None],
                         bp.lam_fine, axis=0) / bp.norm

# ── Load real data FIRST ──────────────────────────────────────────────────────
# Real data must be loaded before simulation so the simulation can use the
# same baseline amplitude and noise level, making injection→recovery a fair
# analog of the actual measurement.
real_df        = None
fit_real       = None
alpha_real     = None
alpha_err_real = None
real_labels    = []
N_MASS_REAL    = 0

try:
    from simstack4.wrapper import SimstackWrapper
    from simstack4.analyze_pah import combine_pah_spectra

    wrappers = []
    for i, path in enumerate(RUN_JSON):
        print(f'  Loading run {i}: {os.path.basename(path)}')
        w = SimstackWrapper()
        w.load_stacking_results(path)
        w.run_analysis_only(**ANALYSIS_KWARGS)
        wrappers.append(w)
    real_df = combine_pah_spectra(wrappers, split_filter=[0])

    if real_df is not None and len(real_df) > 10:
        print('Fitting forward model on real data…')
        fit_real = pah.fit_forward_model_multibin(
            real_df,
            group_col      = 'stellar_mass',
            flux_col       = 'f24_to_fpeak',
            feature_groups = FEATURE_GROUPS,
            verbose        = True,
        )
        alpha_real     = np.array(fit_real['alpha_per_bin'])
        alpha_err_real = np.array(fit_real['alpha_err_per_bin'])
        real_labels    = fit_real['bin_labels']
        N_MASS_REAL    = len(real_labels)
        print(f'  Real α: {alpha_real.round(3)}')
except Exception as exc:
    print(f'Real data not available ({exc}); Row B will be empty.')

# ── Derive simulation parameters from real data ───────────────────────────────
# If real data is available, use its fitted baseline and measured noise so
# the simulation runs at the same flux scale and SNR as the real measurement.
# Otherwise fall back to generic defaults.
if fit_real is not None:
    N_MASS_SIM      = N_MASS_REAL
    sim_baseline    = fit_real['baseline_coeffs']   # shape (N_bins, 3)
    sim_bin_centers = [fit_real['model_per_bin'][l]['bin_center'] for l in real_labels]
    # Fractional noise: median(flux_err / flux) across all bins
    frac = [np.nanmedian(fit_real['model_per_bin'][l]['flux_err'] /
                         np.abs(fit_real['model_per_bin'][l]['flux']))
            for l in real_labels]
    sim_noise_level = float(np.nanmedian(frac))
    # Inject α values slightly different from measured so recovery is non-trivial
    ALPHA_INJECTED  = np.array([1.10, 0.90, 0.80, 0.70])[:N_MASS_SIM]
    print(f'  Sim uses real baseline; fractional noise estimate: {sim_noise_level:.2f}')
else:
    N_MASS_SIM      = 3
    sim_baseline    = None          # simulate_pah_data default
    sim_bin_centers = [9.4, 10.5, 11.2]
    sim_noise_level = 0.04
    ALPHA_INJECTED  = np.array([1.07, 0.87, 0.69])

MASS_COLORS_SIM  = [blues(0.30 + 0.18 * i) for i in range(N_MASS_SIM)]
MASS_COLORS_REAL = [blues(0.30 + 0.18 * i) for i in range(N_MASS_REAL)]

# ── Generate synthetic data for Row A ─────────────────────────────────────────
# Simulation uses the real data's baseline coefficients and noise level so that
# A-row and B-row share the same flux scale — the injection→recovery test is
# a direct analog of the real measurement, not a toy problem at 10× higher SNR.
print('Generating simulation data…')
sim_result = pah.simulate_pah_data(
    bin_z_ranges    = [(0.5, 3.5)] * N_MASS_SIM,
    alpha_true      = ALPHA_INJECTED,
    feature_groups  = FEATURE_GROUPS,
    group_col       = 'stellar_mass',
    bin_centers     = sim_bin_centers,
    baseline_coeffs = sim_baseline,
    n_obs_per_bin   = 65,
    noise_level     = sim_noise_level,
    flux_col        = 'f24_to_fpeak',
    seed            = 42,
)
sim_df = sim_result['df']

print('Fitting forward model on simulated data…')
fit_sim = pah.fit_forward_model_multibin(
    sim_df,
    group_col      = 'stellar_mass',
    flux_col       = 'f24_to_fpeak',
    feature_groups = FEATURE_GROUPS,
    verbose        = True,
)
alpha_rec     = np.array(fit_sim['alpha_per_bin'])
alpha_err_rec = np.array(fit_sim['alpha_err_per_bin'])
sim_labels    = fit_sim['bin_labels']
print(f'  Injected α: {ALPHA_INJECTED}')
print(f'  Recovered α: {alpha_rec.round(3)}')
print(f'  Errors:      {alpha_err_rec.round(3)}')

# ── Helper: compute detrended residuals ───────────────────────────────────────
def detrend_df(fit_result):
    result = {}
    for label, d in fit_result['model_per_bin'].items():
        result[label] = {
            'lam_rest': 24.0 / (1 + d['z']),
            'z':        d['z'],
            'resid':    d['detrended_data'] - 1.0,
            'model':    d['detrended_model'] - 1.0,
            'err':      d['detrended_err'],
        }
    return result

residuals_sim  = detrend_df(fit_sim)
residuals_real = detrend_df(fit_real) if fit_real is not None else None

# ── Figure setup: 2 rows × 3 columns ─────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=DPI,
                         gridspec_kw={'hspace': 0.45, 'wspace': 0.35})
fig.patch.set_facecolor(FIG_BG)
for ax in axes.flat:
    ax.set_facecolor(FIG_BG)

# ── Panel labels ──────────────────────────────────────────────────────────────
for col, letter in enumerate(['A', 'B', 'C']):
    axes[0, col].set_title(f'({letter.lower()}{1})  Simulation', fontsize=LABEL_FS,
                            fontweight='bold', loc='left', pad=4)
    axes[1, col].set_title(f'({letter.lower()}{2})  Real data', fontsize=LABEL_FS,
                            fontweight='bold', loc='left', pad=4)

# Helper to set up pseudo-spectrum axis
def setup_spec_ax(ax, title=''):
    ax.set_xlim(*LAM_RANGE)
    ax.set_xlabel('Rest-frame wavelength (µm)', fontsize=LABEL_FS)
    ax.set_ylabel('$f_{24} / f_\\mathrm{peak}$', fontsize=LABEL_FS)
    ax.axhline(1.0, color='gray', lw=0.7, ls='--', alpha=0.4)
    ax.tick_params(labelsize=TICK_FS)
    for j in range(len(PAH_FEATURES)):
        if PAH_FEATURES[j][1] > 0.001:
            lc = PAH_FEATURES[j][0]
            if LAM_RANGE[0] < lc < LAM_RANGE[1]:
                ax.axvline(lc, color=FEAT_COLORS[j], lw=0.9, ls=':', alpha=0.4)
                ax.text(lc, ax.get_ylim()[1],
                        f'{lc:.1f}', fontsize=7, ha='center', va='top',
                        color=FEAT_COLORS[j], alpha=0.7,
                        clip_on=True)

def setup_resid_ax(ax):
    ax.set_xlim(*LAM_RANGE)
    ax.set_xlabel('Rest-frame wavelength (µm)', fontsize=LABEL_FS)
    ax.set_ylabel('$(f_{24} - \\mathrm{baseline}) / \\mathrm{baseline}$', fontsize=LABEL_FS)
    ax.axhline(0.0, color='gray', lw=0.7, ls='--', alpha=0.4)
    ax.tick_params(labelsize=TICK_FS)
    for j in range(len(PAH_FEATURES)):
        if PAH_FEATURES[j][1] > 0.001:
            lc = PAH_FEATURES[j][0]
            if LAM_RANGE[0] < lc < LAM_RANGE[1]:
                ax.axvline(lc, color=FEAT_COLORS[j], lw=0.9, ls=':', alpha=0.35)

def setup_alpha_ax(ax, injected=None):
    ax.set_xlabel('Injected  α', fontsize=LABEL_FS)
    ax.set_ylabel('Recovered  α', fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    lo, hi = 0.4, 1.4
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.plot([lo, hi], [lo, hi], '-', color='gray', lw=1, alpha=0.5)  # diagonal
    ax.fill_between([lo, hi], [lo * 0.85, hi * 0.85], [lo * 1.15, hi * 1.15],
                    alpha=0.07, color='gray')  # ±15% envelope
    if injected is not None:
        ax.set_xticks(injected.round(2))

# ── Panel A1: simulation pseudo-spectra ──────────────────────────────────────
ax_A1 = axes[0, 0]
# Compute ylim from data — simulate_pah_data baseline ~10^(-0.3z), not ~1.0
_a1_flux = np.concatenate([fit_sim['model_per_bin'][l]['flux'] for l in sim_labels])
_a1_mod  = np.concatenate([fit_sim['model_per_bin'][l]['model'] for l in sim_labels])
_a1_lo   = min(np.nanpercentile(_a1_flux, 2), np.nanmin(_a1_mod))
_a1_hi   = max(np.nanpercentile(_a1_flux, 98), np.nanmax(_a1_mod))
_a1_mg   = 0.12 * (_a1_hi - _a1_lo)
ax_A1.set_ylim(_a1_lo - _a1_mg, _a1_hi + _a1_mg)
setup_spec_ax(ax_A1)

for i, (label, mcolor) in enumerate(zip(sim_labels, MASS_COLORS_SIM)):
    d = fit_sim['model_per_bin'][label]
    lam = 24.0 / (1 + d['z'])
    ax_A1.scatter(lam, d['flux'],
                  c=np.array([mcolor]), s=12, alpha=0.55, edgecolors='none', zorder=4)
    sort_idx = np.argsort(lam)
    ax_A1.plot(lam[sort_idx], d['model'][sort_idx],
               '-', color=mcolor, lw=2, alpha=0.7, zorder=5)

legend_els_sim = [Line2D([0], [0], marker='o', color='w', markerfacecolor=MASS_COLORS_SIM[i],
                          markersize=7, label=f'log M*  {sim_labels[i]}') for i in range(N_MASS_SIM)]
ax_A1.legend(handles=legend_els_sim, fontsize=7, loc='upper right', framealpha=0.8)

# ── Panel A2: simulation residuals ────────────────────────────────────────────
ax_A2 = axes[0, 1]
setup_resid_ax(ax_A2)

# These will be revealed in the animation; pre-plot final state for static PNG
resid_scats_A2 = []
resid_lines_A2 = []
for i, (label, mcolor) in enumerate(zip(sim_labels, MASS_COLORS_SIM)):
    rd = residuals_sim[label]
    sort_idx = np.argsort(rd['lam_rest'])
    sc = ax_A2.scatter(rd['lam_rest'], rd['resid'],
                       c=np.array([mcolor]), s=12, alpha=0.55, edgecolors='none', zorder=4)
    ln, = ax_A2.plot(rd['lam_rest'][sort_idx], rd['model'][sort_idx],
                     '-', color=mcolor, lw=2.5, alpha=0.8, zorder=5)
    resid_scats_A2.append(sc)
    resid_lines_A2.append(ln)

# Set y-limits to contain full PAH bumps
_a2_r = np.concatenate([residuals_sim[l]['resid'] for l in sim_labels])
_a2_m = np.concatenate([residuals_sim[l]['model'] for l in sim_labels])
_a2_lo, _a2_hi = min(_a2_r.min(), _a2_m.min()), max(_a2_r.max(), _a2_m.max())
_a2_margin = 0.15 * (_a2_hi - _a2_lo)
ax_A2.set_ylim(_a2_lo - _a2_margin, _a2_hi + _a2_margin)

# ── Panel A3: injected vs recovered α ────────────────────────────────────────
ax_A3 = axes[0, 2]
setup_alpha_ax(ax_A3, injected=ALPHA_INJECTED)
ax_A3.set_title('(a3)  Simulation', fontsize=LABEL_FS, fontweight='bold',
                 loc='left', pad=4)

# Final state (also used for static PNG)
for i, (alpha_inj, alpha_r, alpha_e, mcolor) in enumerate(
        zip(ALPHA_INJECTED, alpha_rec, alpha_err_rec, MASS_COLORS_SIM)):
    ax_A3.errorbar(alpha_inj, alpha_r, yerr=alpha_e,
                   fmt='o', color=mcolor, ms=9, elinewidth=0.8,
                   capsize=3, zorder=5)
    ax_A3.text(alpha_inj + 0.01, alpha_r + 0.02, f'log M* {sim_labels[i]}',
               fontsize=6.5, color=mcolor, va='bottom')

chi2_text = ax_A3.text(0.05, 0.95, f'χ²_red = {fit_sim["chi2_red"]:.2f}',
                        transform=ax_A3.transAxes, fontsize=9, va='top',
                        bbox=dict(fc='white', ec='gray', alpha=0.85))

# ── Row B: real data ──────────────────────────────────────────────────────────
ax_B1, ax_B2, ax_B3 = axes[1, 0], axes[1, 1], axes[1, 2]

if real_df is not None and fit_real is not None:
    # B1: real pseudo-spectra — set data-driven ylim BEFORE setup_spec_ax
    # (setup_spec_ax reads get_ylim() for feature label placement)
    _b1_flux = np.concatenate([fit_real['model_per_bin'][l]['flux'] for l in real_labels])
    _b1_mod  = np.concatenate([fit_real['model_per_bin'][l]['model'] for l in real_labels])
    _b1_lo = min(np.nanpercentile(_b1_flux, 2), np.nanmin(_b1_mod))
    _b1_hi = max(np.nanpercentile(_b1_flux, 98), np.nanmax(_b1_mod))
    _b1_margin = 0.18 * (_b1_hi - _b1_lo)
    ax_B1.set_ylim(_b1_lo - _b1_margin, _b1_hi + _b1_margin)
    setup_spec_ax(ax_B1)
    ax_B1.set_ylabel('$f_{24} / f_\\mathrm{peak}$', fontsize=LABEL_FS)
    for i, (label, mcolor) in enumerate(zip(real_labels, MASS_COLORS_REAL)):
        d = fit_real['model_per_bin'][label]
        lam = 24.0 / (1 + d['z'])
        sort_idx = np.argsort(lam)
        ax_B1.errorbar(lam, d['flux'], yerr=d['flux_err'],
                       fmt='o', color=mcolor, ms=4, elinewidth=0.8, alpha=0.65,
                       capsize=2, zorder=4)
        ax_B1.plot(lam[sort_idx], d['model'][sort_idx],
                   '-', color=mcolor, lw=2, alpha=0.7, zorder=5)
    legend_els_real = [Line2D([0], [0], marker='o', color='w',
                               markerfacecolor=MASS_COLORS_REAL[i],
                               markersize=7, label=f'log M*  {real_labels[i]}')
                        for i in range(N_MASS_REAL)]
    ax_B1.legend(handles=legend_els_real, fontsize=7, loc='upper right', framealpha=0.8)

    # B2: real residuals
    setup_resid_ax(ax_B2)
    for i, (label, mcolor) in enumerate(zip(real_labels, MASS_COLORS_REAL)):
        rd = residuals_real[label]
        sort_idx = np.argsort(rd['lam_rest'])
        ax_B2.errorbar(rd['lam_rest'], rd['resid'], yerr=rd['err'],
                       fmt='o', color=mcolor, ms=4, elinewidth=0.8,
                       alpha=0.65, capsize=2, zorder=4)
        ax_B2.plot(rd['lam_rest'][sort_idx], rd['model'][sort_idx],
                   '-', color=mcolor, lw=2.5, alpha=0.8, zorder=5)

    # Set symmetric y-limits from data (clip extreme noise outliers at 2%)
    _b2_r = np.concatenate([residuals_real[l]['resid'] for l in real_labels])
    _b2_m = np.concatenate([residuals_real[l]['model'] for l in real_labels])
    _b2_abs = max(np.nanpercentile(np.abs(_b2_r), 98), np.nanmax(np.abs(_b2_m)))
    ax_B2.set_ylim(-_b2_abs * 1.25, _b2_abs * 1.25)

    # B3: real measured α
    ax_B3.set_xlim(0.4, 1.4); ax_B3.set_ylim(0.4, 1.4)
    setup_alpha_ax(ax_B3, injected=alpha_real)
    for i, (alpha_r, alpha_e, mcolor) in enumerate(
            zip(alpha_real, alpha_err_real, MASS_COLORS_REAL)):
        ax_B3.errorbar(alpha_r, alpha_r, yerr=alpha_e,
                       fmt='o', color=mcolor, ms=9, elinewidth=0.8, capsize=3, zorder=5)
    ax_B3.set_xlabel('Measured  α  (same axis)', fontsize=LABEL_FS)
    ax_B3.set_ylabel('Measured  α', fontsize=LABEL_FS)
    # Bar chart alternative for real data: show α vs log M*
    ax_B3.set_title('(b3)  Measured α(M*)', fontsize=LABEL_FS, fontweight='bold',
                     loc='left', pad=4)

    # Use a simpler bar display for B3 (x=mass bin index, y=α)
    ax_B3.cla()
    ax_B3.set_facecolor(FIG_BG)
    ax_B3.tick_params(labelsize=TICK_FS)
    ax_B3.set_xlabel('Stellar mass bin', fontsize=LABEL_FS)
    ax_B3.set_ylabel('PAH amplitude  α', fontsize=LABEL_FS)
    ax_B3.set_title('(b3)  Measured α vs M*', fontsize=LABEL_FS, fontweight='bold',
                     loc='left', pad=4)
    bin_x_real = np.arange(N_MASS_REAL)
    ax_B3.bar(bin_x_real, alpha_real, yerr=alpha_err_real, color=MASS_COLORS_REAL,
              edgecolor='white', linewidth=0.5, capsize=4,
              error_kw=dict(elinewidth=0.8, ecolor='gray'))
    ax_B3.set_xticks(bin_x_real)
    ax_B3.set_xticklabels([str(lb) for lb in real_labels],
                           fontsize=7, rotation=15)
    ax_B3.axhline(0, color='gray', lw=0.5, alpha=0.4)
    ax_B3.set_ylim(0, max(alpha_real) * 1.35)

    # Also add A3 bar chart for visual parallel
    ax_A3.cla()
    ax_A3.set_facecolor(FIG_BG)
    ax_A3.tick_params(labelsize=TICK_FS)
    ax_A3.set_xlabel('Stellar mass bin', fontsize=LABEL_FS)
    ax_A3.set_ylabel('PAH amplitude  α', fontsize=LABEL_FS)
    ax_A3.set_title('(a3)  Injected vs recovered α', fontsize=LABEL_FS,
                     fontweight='bold', loc='left', pad=4)
    bin_x_sim = np.arange(N_MASS_SIM)
    ax_A3.bar(bin_x_sim - 0.2, ALPHA_INJECTED, width=0.35, color=MASS_COLORS_SIM,
              edgecolor='white', linewidth=0.5, label='Injected', alpha=0.6)
    ax_A3.bar(bin_x_sim + 0.2, alpha_rec, yerr=alpha_err_rec, width=0.35,
              color=MASS_COLORS_SIM, edgecolor='white', linewidth=0.5,
              capsize=4, error_kw=dict(elinewidth=0.8, ecolor='gray'), label='Recovered')
    ax_A3.set_xticks(bin_x_sim)
    ax_A3.set_xticklabels([str(lb) for lb in sim_labels],
                           fontsize=7, rotation=15)
    ax_A3.axhline(0, color='gray', lw=0.5, alpha=0.4)
    ax_A3.set_ylim(0, max(ALPHA_INJECTED) * 1.40)
    ax_A3.legend(fontsize=8, loc='upper right')

else:
    for ax in [ax_B1, ax_B2, ax_B3]:
        ax.text(0.5, 0.5, 'Real data not loaded\n(JSON files not found)',
                transform=ax.transAxes, ha='center', va='center',
                fontsize=10, color='gray')

# ── Row labels ────────────────────────────────────────────────────────────────
fig.text(0.01, 0.74, 'Simulation', va='center', ha='left',
         rotation=90, fontsize=12, fontweight='bold', color='#2c3e50')
fig.text(0.01, 0.27, 'Real data', va='center', ha='left',
         rotation=90, fontsize=12, fontweight='bold', color='#2c3e50')

fig.subplots_adjust(left=0.07, right=0.97, top=0.93, bottom=0.07,
                    hspace=0.55, wspace=0.38)

# ── Static paper PNG ──────────────────────────────────────────────────────────
png_path = os.path.join(FIGURES_DIR, 'pah_fig3_paper.png')
fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
print(f'Paper PNG saved → {png_path}')

# ── Animation: Row A reveal (A2 and A3 convergence) ─────────────────────────
# We animate a fake convergence in A2 and A3 only.
# The convergence path: from zero to the final fit values.
t_path   = np.linspace(0, 1, N_ANIM)
progress = 1.0 / (1 + np.exp(-9 * (t_path - 0.45)))   # S-curve

# Hide A2 and A3 scatters/lines initially (re-draw each frame)
for sc in resid_scats_A2:
    sc.set_alpha(0)
for ln in resid_lines_A2:
    ln.set_alpha(0)

def update_anim(frame):
    p = float(progress[min(frame, N_ANIM - 1)])

    for sc in resid_scats_A2:
        sc.set_alpha(p * 0.55)
    for ln in resid_lines_A2:
        ln.set_alpha(p * 0.8)
        # Scale the model line amplitude by p (converges from 0)
    for i, (b, mcolor) in enumerate(zip(sim_labels, MASS_COLORS_SIM)):
        rd = residuals_sim[b]
        sort_idx = np.argsort(rd['lam_rest'])
        resid_lines_A2[i].set_ydata(rd['model'][sort_idx] * p)

    return resid_lines_A2 + resid_scats_A2

anim = FuncAnimation(fig, update_anim, frames=N_ANIM + 20,
                     interval=1000 // FPS, blit=False)

mp4_path = os.path.join(FIGURES_DIR, 'pah_fig3.mp4')
print(f'Saving MP4 → {mp4_path}')
anim.save(mp4_path, writer='ffmpeg', dpi=DPI, fps=FPS,
          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '18', '-preset', 'fast'])
print('MP4 saved.')
plt.close(fig)
print('Figure 3 complete.')
