#!/usr/bin/env python3
"""
Figure 2: Dithered stacking — from sparse comb to per-population pseudo-spectrum.

Three-panel animated figure:
  Left   — four staggered z-bin combs as horizontal bars at their z positions
  Centre — f₂₄/f_peak vs λ_rest for the intermediate mass bin,
            built up run by run in each run's colour
  Right  — same pseudo-spectra for all four mass bins simultaneously
            (Blues colormap, light→dark = low→high M*)

Animation sequence:
  1. Centre only: Run 1 points appear (sparse)
  2. Runs 2, 3, 4 added to Centre (structure becomes visible)
  3. Freeze → Left panel revealed (comb layout geometry)
  4. Right panel revealed: all mass bins, same build-up, amplitude decreasing

Real-data overlay (if JSON files available):
  combine_pah_spectra points fade in as grey markers after the synthetic build-up.

Deliverables written to ../figures/:
  pah_fig2.mp4          H.264, 30 fps
  pah_fig2_final.png    static three-panel layout (final frame)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from simstack4.pah_model import PAHModel, PAH_FEATURES
from simstack4.pah_bandpass import get_bandpass
from simstack4.pah_dither import DitherScheme

# ── Shared visual language ────────────────────────────────────────────────────
FEAT_COLORS  = ['C0', 'C1', 'C2', 'C3', 'C4']
BP_COLOR     = '#3b6ea5'
BP_ALPHA     = 0.25
FIG_BG       = 'white'
LABEL_FS     = 11
TICK_FS      = 9

# Four dithered runs: distinct colours
RUN_COLORS   = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd']

# Mass bins: Blues sequential, light→dark = low→high M* (4 bins in actual runs)
MASS_LABELS  = ['8.5–10.2', '10.2–10.6', '10.6–11.0', '11.0–12.0']
N_MASS       = 4
blues        = matplotlib.colormaps['Blues']
MASS_COLORS  = [blues(0.30 + 0.18 * i) for i in range(N_MASS)]

FPS      = 24
DPI      = 120

FIGURES_DIR  = os.path.join(os.path.dirname(__file__), '..', 'figures')
PICKLES_DIR  = os.path.expandvars('$PICKLESPATH/simstack/stacked_flux_densities')

# JSON run IDs — order matches dither offsets 0, Δz/4, Δz/2, 3Δz/4
# (per notebooks/2026-06-12-load-json-fit-seds-redshift-stellar-mass-PAH-dithered-dz015.ipynb)
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
pah    = PAHModel()
bp     = get_bandpass('MIPS_24')
scheme = DitherScheme.uniform(dz=0.15, n_stagger=4, z_min=0.5, z_max=3.5)

# Precompute T_total(z): bandpass-integrated PAH excess above continuum
z_fine    = np.linspace(0.5, 3.5, 400)
feat_at_z = pah.feature_spectrum(bp.lam_fine[:, None] / (1 + z_fine))
T_fine    = np.trapezoid(feat_at_z * bp.resp_fine[:, None],
                         bp.lam_fine, axis=0) / bp.norm

# Feature peak redshifts for x-axis ticks
FEAT_PEAK_Z = {j: 24.0 / PAH_FEATURES[j][0] - 1
               for j in range(len(PAH_FEATURES)) if PAH_FEATURES[j][1] > 0.001}

# ── Synthetic data for each run and each mass bin ─────────────────────────────
# PAH amplitudes — 4 mass bins interpolated from measured 3-bin values [1.07, 0.87, 0.69]
ALPHA_TRUTH = [1.07, 0.93, 0.80, 0.69]
RNG = np.random.default_rng(42)
NOISE_SIGMA = 0.025   # fractional flux noise

def make_synthetic_run(run_id):
    """Generate synthetic f24/f_peak vs lambda_rest for one dither run, all mass bins."""
    edges = scheme.runs[run_id]
    z_centers = 0.5 * (edges[:-1] + edges[1:])
    lam_rest  = 24.0 / (1 + z_centers)  # rest-frame wavelengths probed by MIPS24

    per_bin = {}
    for m, alpha_m in enumerate(ALPHA_TRUTH):
        T_at_z = np.interp(z_centers, z_fine, T_fine)
        # Model: f24/f_peak = 1 + α_m × T(z)  (baseline normalised to 1)
        f_model = 1.0 + alpha_m * T_at_z
        noise   = RNG.normal(0, NOISE_SIGMA, len(f_model))
        per_bin[m] = {'lam_rest': lam_rest, 'f24': f_model + noise, 'z': z_centers}
    return per_bin

# Pre-generate all runs
RUNS_DATA = [make_synthetic_run(r) for r in range(4)]

# ── Load real data (optional) ─────────────────────────────────────────────────
real_data = None
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
    print('Loaded and analysed 4 stacking runs — adding real data overlay.')
    real_df = combine_pah_spectra(wrappers, split_filter=[0])
    if real_df is not None:
        real_data = real_df
        print(f'  {len(real_df)} real spectral points.')
except Exception as exc:
    print(f'Real data not available ({exc}); synthetic only.')

# ── Build animation schedule ──────────────────────────────────────────────────
# Phase 1 (Centre only): add runs 0→3 point by point
# Phase 2: hold 25 frames, reveal Left panel
# Phase 3: reveal Right panel (all mass bins, same run-by-run build-up)
# Phase 4: hold; if real_data available, fade in grey markers

PAUSE_BETWEEN_RUNS = 12
PAUSE_BEFORE_LEFT  = 30
PAUSE_BEFORE_RIGHT = 20
HOLD_END           = 40
REAL_DATA_FRAMES   = 20 if real_data is not None else 0

# Build schedule entries: (phase, run_id, point_idx)
# Phases: 'centre_run', 'pause_left', 'show_left', 'pause_right', 'show_right', 'hold', 'real'
schedule = []

for run_id in range(4):
    for _ in range(PAUSE_BETWEEN_RUNS):
        schedule.append(('centre_run', run_id, -1))
    n_pts = len(scheme.runs[run_id]) - 1
    for pt in range(n_pts):
        schedule.append(('centre_run', run_id, pt))

for _ in range(PAUSE_BEFORE_LEFT):
    schedule.append(('pause_left', 3, -1))
schedule.append(('show_left', 3, -1))

for _ in range(PAUSE_BEFORE_RIGHT):
    schedule.append(('pause_right', 3, -1))

for run_id in range(4):
    for _ in range(PAUSE_BETWEEN_RUNS // 2):
        schedule.append(('show_right', run_id, -1))
    n_pts = len(scheme.runs[run_id]) - 1
    for pt in range(n_pts):
        schedule.append(('show_right', run_id, pt))

for _ in range(HOLD_END):
    schedule.append(('hold', 3, -1))

for frame_r in range(REAL_DATA_FRAMES):
    schedule.append(('real', frame_r, -1))

N_FRAMES = len(schedule)

# ── Figure setup ──────────────────────────────────────────────────────────────
fig, (ax_left, ax_cen, ax_right) = plt.subplots(
    1, 3, figsize=(17, 5.5), dpi=DPI,
    gridspec_kw={'width_ratios': [0.9, 1.3, 1.3]})
fig.patch.set_facecolor(FIG_BG)

lam_range = (5.5, 14.5)
f24_range = (0.85, 1.35)

# ── Left panel: z-bin comb layout ─────────────────────────────────────────────
ax_left.set_facecolor(FIG_BG)
ax_left.set_xlim(-0.05, 1.05)
ax_left.set_ylim(0.5, 3.5)
ax_left.set_ylabel('Redshift  z', fontsize=LABEL_FS)
ax_left.set_xlabel('Dither run', fontsize=LABEL_FS)
ax_left.set_title('Staggered z-bin combs\n(4 runs, Δz=0.15)', fontsize=LABEL_FS,
                   fontweight='bold')
ax_left.tick_params(labelsize=TICK_FS)
ax_left.set_xticks([0.125, 0.375, 0.625, 0.875])
ax_left.set_xticklabels(['Run 1', 'Run 2', 'Run 3', 'Run 4'], fontsize=8)
ax_left.set_visible(False)  # hidden until revealed

# Draw the comb bars (will be made visible when left panel is revealed)
comb_artists = []
for run_id, (edges, color) in enumerate(zip(scheme.runs, RUN_COLORS)):
    x_center = 0.125 + run_id * 0.25
    for lo, hi in zip(edges[:-1], edges[1:]):
        z_mid = 0.5 * (lo + hi)
        bar = ax_left.barh(z_mid, 0.18, left=x_center - 0.09, height=hi - lo - 0.01,
                           color=color, alpha=0.75, edgecolor='white', linewidth=0.4)
        comb_artists.append(bar)

left_visible = [False]

# ── Centre panel: f24/f_peak vs lambda_rest (one mass bin) ───────────────────
CENTRE_MASS = 1  # intermediate mass bin

ax_cen.set_facecolor(FIG_BG)
ax_cen.set_xlim(*lam_range)
ax_cen.set_ylim(*f24_range)
ax_cen.set_xlabel('Rest-frame wavelength (µm)', fontsize=LABEL_FS)
ax_cen.set_ylabel('$f_{24} / f_\\mathrm{peak}$', fontsize=LABEL_FS)
ax_cen.set_title(f'M* = {MASS_LABELS[CENTRE_MASS]} M☉\n(one mass bin, run-by-run build-up)',
                  fontsize=LABEL_FS, fontweight='bold')
ax_cen.tick_params(labelsize=TICK_FS)
ax_cen.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.4)

# Feature peak λ_rest annotations
for j, z_pk in FEAT_PEAK_Z.items():
    lc = PAH_FEATURES[j][0]
    if lam_range[0] < lc < lam_range[1]:
        ax_cen.axvline(lc, color=FEAT_COLORS[j], lw=1, ls=':', alpha=0.45)
        ax_cen.text(lc, f24_range[1] * 0.99, f'{lc:.1f}',
                    fontsize=8, ha='center', va='top', color=FEAT_COLORS[j])

# Model curve for intermediate mass bin (ghost)
lam_model = np.linspace(*lam_range, 400)
z_model   = 24.0 / lam_model - 1
T_model   = np.interp(z_model, z_fine, T_fine)
f_model   = 1.0 + ALPHA_TRUTH[CENTRE_MASS] * T_model
ax_cen.plot(lam_model, f_model, '-', color=RUN_COLORS[0], lw=1.5, alpha=0.12, zorder=1)

run_label_cen = ax_cen.text(0.03, 0.95, '', transform=ax_cen.transAxes,
                             fontsize=11, va='top', fontweight='bold')

cen_scatter = {r: None for r in range(4)}
cen_plotted = {r: [] for r in range(4)}

# ── Right panel: all mass bins ────────────────────────────────────────────────
ax_right.set_facecolor(FIG_BG)
ax_right.set_xlim(*lam_range)
ax_right.set_ylim(0.75, 1.45)  # wider to accommodate amplitude differences
ax_right.set_xlabel('Rest-frame wavelength (µm)', fontsize=LABEL_FS)
ax_right.set_ylabel('$f_{24} / f_\\mathrm{peak}$  (offset by M*)', fontsize=LABEL_FS)
ax_right.set_title('All mass bins — amplitude\ndecreases with M*',
                    fontsize=LABEL_FS, fontweight='bold')
ax_right.tick_params(labelsize=TICK_FS)
ax_right.set_visible(False)

for j, z_pk in FEAT_PEAK_Z.items():
    lc = PAH_FEATURES[j][0]
    if lam_range[0] < lc < lam_range[1]:
        ax_right.axvline(lc, color=FEAT_COLORS[j], lw=1, ls=':', alpha=0.45)

# Offsets so mass bins don't overlap in Right panel
OFFSETS = [-0.075, -0.025, 0.025, 0.075]

for m, (alpha_m, mcolor, offset, mlabel) in enumerate(
        zip(ALPHA_TRUTH, MASS_COLORS, OFFSETS, MASS_LABELS)):
    f_m = 1.0 + alpha_m * T_model + offset
    ax_right.plot(lam_model, f_m, '-', color=mcolor, lw=1.5, alpha=0.15, zorder=1)
    ax_right.text(lam_range[1] - 0.1, 1.0 + alpha_m * max(T_model) + offset,
                  f'log M* {mlabel}', fontsize=7, ha='right', va='bottom',
                  color=mcolor, alpha=0.6)

right_scatter = {(r, m): None for r in range(4) for m in range(N_MASS)}
right_plotted = {r: {m: [] for m in range(N_MASS)} for r in range(4)}
right_visible = [False]

real_scat = [None]

# ── Animation update ──────────────────────────────────────────────────────────
def update(frame):
    phase, run_id, pt_idx = schedule[frame]

    # --- Centre panel update (runs 0→3) ---
    if phase in ('centre_run',):
        if pt_idx >= 0:
            cen_plotted[run_id].append(pt_idx)
        for r in range(run_id + 1):
            pts = cen_plotted[r]
            if cen_scatter[r] is not None:
                cen_scatter[r].remove()
                cen_scatter[r] = None
            if pts:
                d = RUNS_DATA[r][CENTRE_MASS]
                cen_scatter[r] = ax_cen.scatter(
                    d['lam_rest'][pts], d['f24'][pts],
                    c=RUN_COLORS[r], s=22, zorder=5, alpha=0.85,
                    edgecolors='white', linewidths=0.4)
        run_label_cen.set_text(f'Run {run_id + 1} / 4')
        run_label_cen.set_color(RUN_COLORS[run_id])

    # --- Left panel reveal ---
    if phase == 'show_left' and not left_visible[0]:
        ax_left.set_visible(True)
        left_visible[0] = True
        fig.canvas.draw_idle()

    # --- Right panel: all mass bins ---
    if phase == 'show_right':
        if not right_visible[0]:
            ax_right.set_visible(True)
            right_visible[0] = True
        if pt_idx >= 0:
            for m in range(N_MASS):
                right_plotted[run_id][m].append(pt_idx)
        for r in range(run_id + 1):
            for m in range(N_MASS):
                pts = right_plotted[r][m]
                key = (r, m)
                if right_scatter[key] is not None:
                    right_scatter[key].remove()
                    right_scatter[key] = None
                if pts:
                    d = RUNS_DATA[r][m]
                    right_scatter[key] = ax_right.scatter(
                        d['lam_rest'][pts],
                        d['f24'][pts] + OFFSETS[m],
                        c=np.array([MASS_COLORS[m]]),
                        s=16, zorder=5, alpha=0.80,
                        edgecolors='none')

    # --- Real data overlay ---
    if phase == 'real' and real_data is not None:
        alpha_fade = min(1.0, (pt_idx + 1) / REAL_DATA_FRAMES)
        if real_scat[0] is not None:
            real_scat[0].remove()
        # Map stellar_mass bins to the three mass categories
        df_m = real_data.copy()
        real_scat[0] = ax_cen.scatter(
            df_m['rest_lam_24'], df_m['f24_to_fpeak'],
            c='gray', s=15, alpha=alpha_fade * 0.55, zorder=3,
            edgecolors='none', marker='D')

    return []

# ── Render ────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=1.2)

anim = FuncAnimation(fig, update, frames=N_FRAMES,
                     interval=1000 // FPS, blit=False)

mp4_path = os.path.join(FIGURES_DIR, 'pah_fig2.mp4')
print(f'Saving MP4 → {mp4_path}  ({N_FRAMES} frames)')
anim.save(mp4_path, writer='ffmpeg', dpi=DPI, fps=FPS,
          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '18', '-preset', 'fast'])
print('MP4 saved.')

# ── Static final-frame PNG ────────────────────────────────────────────────────
# Run all frames to reach final state
for f in range(N_FRAMES):
    update(f)

# Make sure both side panels are visible
ax_left.set_visible(True)
ax_right.set_visible(True)

png_path = os.path.join(FIGURES_DIR, 'pah_fig2_final.png')
fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
print(f'Final PNG saved → {png_path}')
plt.close(fig)
print('Figure 2 complete.')
