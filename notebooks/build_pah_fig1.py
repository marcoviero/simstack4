#!/usr/bin/env python3
"""
Figure 1: PAH features transiting the MIPS 24 µm bandpass.

Three-panel animated figure:
  Left   — static rest-frame PAH template, features colored C0–C4
  Centre — MIPS 24 µm bandpass window sliding LEFT over the template
           (rest-frame convention: spectrum fixed, bandpass moves)
  Right  — T(z) kernel tracing in real time

Deliverables written to ../figures/:
  pah_fig1.mp4          H.264, 1280×480, 30 fps
  pah_fig1_keyframe.png static PNG at z=1.7 (7.7+8.6 µm in band)
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

# ── Shared visual language ────────────────────────────────────────────────────
FEAT_COLORS = ['C0', 'C1', 'C2', 'C3', 'C4']   # fixed color per PAH feature
BP_COLOR    = '#3b6ea5'                           # MIPS bandpass: steel blue
BP_ALPHA    = 0.25
FIG_BG      = 'white'
LABEL_FS    = 11
TICK_FS     = 9

# ── Animation parameters ──────────────────────────────────────────────────────
FPS      = 30
DPI      = 120
N_FRAMES = 160
Z_MIN, Z_MAX = 0.5, 3.8

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Physics setup ─────────────────────────────────────────────────────────────
pah = PAHModel()
bp  = get_bandpass('MIPS_24')

# Rest-frame wavelength grid for left/centre panels
lam_plot = np.linspace(5, 16, 600)

# Per-feature spectra (for individual coloring in Left panel)
feat_specs = []
for lc, strength, fwhm in PAH_FEATURES:
    sigma = fwhm / 2.355
    feat_specs.append(strength * np.exp(-0.5 * ((lam_plot - lc) / sigma) ** 2))

feat_total = np.sum(feat_specs, axis=0)
spec_ymax  = feat_total.max() * 1.25

# T(z) precomputed on fine z grid (integrate PAH features over bandpass)
# T(z) = ∫ feat(λ_obs/(1+z)) × R(λ_obs) dλ_obs / ∫ R(λ_obs) dλ_obs
z_fine    = np.linspace(Z_MIN, Z_MAX, 600)
lam_obs_2d = bp.lam_fine[:, None]                         # (N_bp, 1)
feat_at_z  = pah.feature_spectrum(lam_obs_2d / (1 + z_fine))  # (N_bp, N_z)
T_fine     = np.trapezoid(feat_at_z * bp.resp_fine[:, None],
                          bp.lam_fine, axis=0) / bp.norm   # (N_z,)

T_ymin, T_ymax = T_fine.min() * 0.95, T_fine.max() * 1.12

# Redshifts at which each feature centres in the MIPS band: z_peak = 24/λ - 1
FEAT_PEAK_Z = {j: (24.0 / PAH_FEATURES[j][0] - 1) for j in range(len(PAH_FEATURES))
               if PAH_FEATURES[j][1] > 0.001}

# Key redshifts annotated on Centre panel (bandpass centre in rest frame = 24/(1+z))
KEY_EVENTS = [(0.7, 12.7, '12.7 µm\nz=0.7'),
              (1.7,  8.15, '7.7+8.6 µm\nz=1.7'),
              (2.9,  6.2,  '6.2 µm\nz=2.9')]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_left, ax_cen, ax_right) = plt.subplots(
    1, 3, figsize=(16, 5.2), dpi=DPI,
    gridspec_kw={'width_ratios': [1, 1.4, 1]})
fig.patch.set_facecolor(FIG_BG)

# ── Left panel (static PAH template) ─────────────────────────────────────────
ax_left.set_facecolor(FIG_BG)
ax_left.set_xlim(5, 16)
ax_left.set_ylim(-0.01, spec_ymax)
ax_left.set_xlabel('Rest-frame wavelength (µm)', fontsize=LABEL_FS)
ax_left.set_ylabel('PAH feature flux (relative)', fontsize=LABEL_FS)
ax_left.set_title('PAH template spectrum', fontsize=LABEL_FS, fontweight='bold')
ax_left.tick_params(labelsize=TICK_FS)
ax_left.axhline(0, color='gray', lw=0.5, alpha=0.4)

for j, (fs, color) in enumerate(zip(feat_specs, FEAT_COLORS)):
    if PAH_FEATURES[j][1] > 0.001:
        ax_left.fill_between(lam_plot, fs, alpha=0.30, color=color)
        ax_left.plot(lam_plot, fs, color=color, lw=2)
        lc = PAH_FEATURES[j][0]
        ax_left.text(lc, PAH_FEATURES[j][1] + 0.025, f'{lc:.1f}',
                     fontsize=8, ha='center', color=color, fontweight='bold')

# Full envelope (faint outline)
ax_left.plot(lam_plot, feat_total, '-', color='gray', lw=1, alpha=0.35)

# ── Centre panel setup ────────────────────────────────────────────────────────
ax_cen.set_facecolor(FIG_BG)
ax_cen.set_xlim(5, 16)
ax_cen.set_ylim(-0.01, spec_ymax)
ax_cen.set_xlabel('Rest-frame wavelength (µm)', fontsize=LABEL_FS)
ax_cen.set_title('Bandpass window slides left\nas z increases →', fontsize=LABEL_FS,
                  fontweight='bold')
ax_cen.tick_params(labelsize=TICK_FS)
ax_cen.axhline(0, color='gray', lw=0.5, alpha=0.4)

# Background: per-feature fills (light, for context)
for j, (fs, color) in enumerate(zip(feat_specs, FEAT_COLORS)):
    if PAH_FEATURES[j][1] > 0.001:
        ax_cen.fill_between(lam_plot, fs, alpha=0.12, color=color)
        ax_cen.plot(lam_plot, fs, '-', color=color, lw=1.5, alpha=0.35)

# Key redshift annotations (static vertical guidelines)
for z_key, lam_key, label in KEY_EVENTS:
    ax_cen.axvline(lam_key, color='gray', lw=0.8, ls='--', alpha=0.40)
    ax_cen.text(lam_key, spec_ymax * 0.98, label,
                fontsize=7, ha='center', va='top', color='gray', alpha=0.75,
                bbox=dict(fc='white', ec='none', alpha=0.6, pad=1))

# Animated elements in Centre
bp_fill_cen = [None]
bp_line_cen, = ax_cen.plot([], [], '-', color=BP_COLOR, lw=2.5, alpha=0.8)
z_text_cen   = ax_cen.text(0.97, 0.06, '', transform=ax_cen.transAxes,
                            fontsize=13, fontweight='bold', ha='right', va='bottom',
                            color=BP_COLOR,
                            bbox=dict(fc='white', ec=BP_COLOR, alpha=0.9, boxstyle='round'))

# ── Right panel setup ─────────────────────────────────────────────────────────
ax_right.set_facecolor(FIG_BG)
ax_right.set_xlim(Z_MIN, Z_MAX)
ax_right.set_ylim(T_ymin, T_ymax)
ax_right.set_xlabel('Redshift  z', fontsize=LABEL_FS)
ax_right.set_ylabel('Bandpass-integrated PAH flux  T(z)', fontsize=LABEL_FS)
ax_right.set_title('T(z) kernel', fontsize=LABEL_FS, fontweight='bold')
ax_right.tick_params(labelsize=TICK_FS)
ax_right.axhline(0, color='gray', lw=0.5, alpha=0.4)

# Ghost full T(z) curve (very faint guide)
ax_right.plot(z_fine, T_fine, '-', color='lightgray', lw=1.5, alpha=0.5, zorder=1)

# Feature peak redshift markers
for j, z_pk in FEAT_PEAK_Z.items():
    if Z_MIN < z_pk < Z_MAX:
        lc = PAH_FEATURES[j][0]
        ax_right.axvline(z_pk, color=FEAT_COLORS[j], lw=1.2, ls=':', alpha=0.55)
        ax_right.text(z_pk, T_ymax * 0.97, f'{lc:.1f}',
                      fontsize=8, ha='center', va='top', color=FEAT_COLORS[j])

# Animated trace elements
T_trail, = ax_right.plot([], [], '-', color='#2c3e50', lw=2.5, zorder=4)
T_dot,   = ax_right.plot([], [], 'o', color='#e74c3c', ms=8, zorder=5)

z_hist: list[float] = []
T_hist: list[float] = []

z_sweep = np.linspace(Z_MIN, Z_MAX, N_FRAMES)

# ── Animation functions ───────────────────────────────────────────────────────
def init():
    T_trail.set_data([], [])
    T_dot.set_data([], [])
    bp_line_cen.set_data([], [])
    z_text_cen.set_text('')
    return T_trail, T_dot, bp_line_cen, z_text_cen

def update(frame):
    z = z_sweep[frame]

    # Centre: bandpass response in rest-frame coordinates
    # resp_rest[i] = R(lam_plot[i] * (1+z))  — bandpass weight at each rest wavelength
    resp_rest   = bp.response_at(lam_plot * (1 + z))
    scaled_resp = resp_rest * spec_ymax * 0.85  # scale to be visible on PAH y-axis

    if bp_fill_cen[0] is not None:
        bp_fill_cen[0].remove()
    bp_fill_cen[0] = ax_cen.fill_between(
        lam_plot, 0, scaled_resp,
        alpha=BP_ALPHA, color=BP_COLOR, zorder=5)
    bp_line_cen.set_data(lam_plot, scaled_resp)
    z_text_cen.set_text(f'z = {z:.2f}')

    # Right: T(z) trace (interpolate precomputed curve)
    T_val = float(np.interp(z, z_fine, T_fine))
    z_hist.append(z)
    T_hist.append(T_val)
    T_trail.set_data(z_hist, T_hist)
    T_dot.set_data([z], [T_val])

    return bp_line_cen, T_trail, T_dot, z_text_cen

# ── Render MP4 ────────────────────────────────────────────────────────────────
plt.tight_layout(pad=1.2)

anim = FuncAnimation(fig, update, init_func=init,
                     frames=N_FRAMES, interval=1000 // FPS, blit=False)

mp4_path = os.path.join(FIGURES_DIR, 'pah_fig1.mp4')
print(f'Saving MP4 → {mp4_path}')
anim.save(mp4_path, writer='ffmpeg', dpi=DPI, fps=FPS,
          extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
                      '-crf', '18', '-preset', 'fast'])
print('MP4 saved.')

# ── Static keyframe at z=1.7 (7.7+8.6 µm in band) ────────────────────────────
z_hist.clear(); T_hist.clear()
z_key = 1.7
iframe_key = int(round((z_key - Z_MIN) / (Z_MAX - Z_MIN) * (N_FRAMES - 1)))
for f in range(iframe_key + 1):
    update(f)

png_path = os.path.join(FIGURES_DIR, 'pah_fig1_keyframe.png')
fig.savefig(png_path, dpi=150, bbox_inches='tight', facecolor=FIG_BG)
print(f'Keyframe PNG saved → {png_path}')
plt.close(fig)
print('Figure 1 complete.')
