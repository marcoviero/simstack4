# PAH Forward Model — Branch 3 Brief

**Goal**: Build three publication- and talk-quality figures that make the PAH tomographic
stacking method self-explanatory to an audience — from mechanism through measurement through
credibility. Statistics come after the pictures land.

---

## Scope: three figures, nothing else first

### Figure 1 — The measurement mechanism: PAH features transiting the MIPS bandpass

**One-sentence story**: as redshift increases, the MIPS 24 µm bandpass slides blueward in
rest-frame wavelength, and each PAH feature in turn modulates the stacked flux.

**Convention** (hold this across all three figures): the *rest-frame* PAH spectrum is fixed;
the MIPS bandpass window moves left as z increases. This matches how an astronomer thinks
("the spectrum is the object; the instrument is the instrument").

**Panels**:
- Left: intrinsic PAH template spectrum (5–16 µm). One fixed color per feature; use the
  exact `PAH_FEATURES` list from `pah_model.py` — not a cartoon — so the figure is honest.
- Centre: the bandpass window at three representative redshifts (z=0.7 illuminates 12.7 µm;
  z=1.7 illuminates 7.7+8.6 µm; z=2.9 illuminates 6.2 µm), drawn as a shaded rectangle
  sliding left over the spectrum.
- Right: the resulting T(z) kernel — bandpass-integrated PAH flux vs z — showing the
  multi-bump structure that dithered stacking is designed to recover.

**Animation**: sweep z from 0.5 → 3.8 (160 frames). The bandpass window slides; the T(z)
panel traces out in real time. The "aha" moment is watching bumps build up in T(z) exactly
as the corresponding features pass through the window.

**Source**: use `pah_bandpass.get_bandpass(24.0)` for the real response curve and
`PAHModel().feature_spectrum(lam)` for the template. No synthetic noise — this is a mechanism
figure, not a data figure.

---

### Figure 2 — Dithered stacking: from a coarse measurement to a per-population pseudo-spectrum

**One-sentence story**: four offset redshift-bin combs interleave to tile the T(z) curve at
4× sampling, and splitting sources by stellar mass reveals that α decreases with M*.

**Panels** (static layout; animate the accumulation):
- Left: z-bin layout — four staggered combs (Δz=0.15, offsets 0, Δz/4, Δz/2, 3Δz/4) shown
  as horizontal bars at the correct z positions. Colour each run differently.
- Centre: f₂₄/f_peak vs z (or λ_rest) for a single mass bin, built up run by run in the
  animation — each comb adds its points in its run colour until the combined pseudo-spectrum
  is visible.
- Right: final combined pseudo-spectra for all four mass bins stacked vertically (or
  colour-coded), showing the amplitude decreasing from low M* to high M*.

**Real data overlay**: after the animation completes the synthetic build-up, reveal the actual
`combine_pah_spectra` points from the stacking runs (grey markers with error bars) behind the
model traces. This is the transition from "here's how the method works" to "here's what we
measured."

**Populations**: primary axis is stellar mass (4 bins, runs complete). Include σ_SFR as a
simulated panel or annotation — once run 4 of the σ_SFR stacking lands it can be swapped for
real data without changing the figure structure.

**Source**: `DitherScheme.uniform(dz=0.15, n_stagger=4)` from `pah_dither.py` for the
synthetic layout; `combine_pah_spectra(wrappers, split_filter=[0])` for real points.

---

### Figure 3 — Forward model: injection → recovery proves the method; real data applies it

**One-sentence story**: we can inject known PAH amplitudes into a simulation, recover them
blind, and then apply the same fitter to real data — the agreement between injected and
recovered α is what justifies the measurement.

**This figure must be injection → recovery, not "watch the fit converge through the data."**
Converging through the data is circular — the model is forced to fit by construction.
Injection → recovery is the credibility argument.

**Layout** (two rows):
- Top row (simulation): inject α = [1.07, 0.87, 0.69] (the measured values) into
  `simulate_dithered_fluxes`; run `fit_forward_model_multibin` blind; show recovered α vs
  injected α with 1σ bars. Panels: (a) injected f₂₄/f_peak pseudo-spectra per mass bin,
  (b) recovered detrended residuals with model overlay, (c) injected vs recovered α scatter.
- Bottom row (real data): the same three panels applied to the actual stacked fluxes.
  The visual parallel says: "the method works on simulations at the top; here it is on data
  at the bottom."

**Animation (optional)**: if helpful, animate the fitter converging in the simulation row only
(not the data row), so the convergence is clearly framed as "this is how we know the model
identifies the right amplitudes."

**Source**: `simulate_dithered_fluxes(scheme, TruthSpectrum(...))` from `pah_dither.py`;
`PAHModel(include_silicate=True).fit_forward_model_multibin(...)` for both sim and data.

---

## Shared visual language (enforce across all three figures)

| Element | Convention |
|---------|-----------|
| PAH features | One fixed color per feature: 6.2 µm `C0`, 7.7 µm `C1`, 8.6 µm `C2`, 11.3 µm `C3`, 12.7 µm `C4` |
| MIPS 24 µm bandpass | Steel blue (#3b6ea5), alpha=0.25 fill |
| Mass bins | Sequential colormap (Blues), light→dark = low→high M* |
| σ_SFR bins | Sequential colormap (Oranges), light→dark = low→high σ_SFR |
| Error bars | Thin, same color as marker, elinewidth=0.8 |
| Background | White (`figure.facecolor="white"`) |
| Font | Same family throughout; axis labels 11pt, tick labels 9pt |

---

## Deliverables per figure

Each figure produces three files:

1. **`notebooks/build_pah_fig{N}.py`** — committed builder script (generates the notebook
   from code; matches repo convention; notebooks themselves are gitignored).
2. **`notebooks/2026-06-18-pah-fig{N}-*.ipynb`** — the generated notebook with outputs
   (static + animation in jshtml for local inspection).
3. **`figures/pah_fig{N}.mp4`** (H.264, 1920×1080 or 1280×720) — for Keynote drop-in.
   GIF fallback at 720p for contexts where MP4 doesn't autoplay.
   Static PNG key-frame at 150 dpi — for paper draft.

Builder scripts use `FuncAnimation.save(..., writer="ffmpeg", dpi=120)` for MP4.
In-notebook preview uses `HTML(anim.to_jshtml())`.

---

## Prior art to audit, not copy-paste

Read these notebooks before starting — understand what they got right and what needs updating:

- `notebooks/2026-04-28-pah-resolution-animation.ipynb` — Animations 1–4 including the
  bandpass sweep, dithered build-up, and forward model convergence. Self-contained synthetic.
  Issues: 3-run scheme, no real data, single-bin fitter.
- `notebooks/2026-04-29-animated-figures-for-lim-talk.ipynb` — Talk-focused version;
  imports real `PAHModel`, `PAH_FEATURES`, `_pah_template_in_band`. Closer to what we need.
  Issues: uses frozen `fit_forward_model` (single bin); `pah_optimized_zbins` rather than
  `DitherScheme`; no per-mass-bin split; HTML export only.

Upgrade plan:
- 4 runs, Δz=0.15, real bin edges (use `DitherScheme.uniform(dz=0.15, n_stagger=4)`)
- `fit_forward_model_multibin` instead of `fit_forward_model` (multiple mass bins)
- MP4 export via ffmpeg
- Real data overlay from `combine_pah_spectra` in Figures 2 and 3

---

## What is NOT in this branch

The following are real and important but deferred until the three figures are done:

- Error rescaling by √(χ²_red) = 1.83
- Robustness suite (baseline degree, bin edge shifts, jackknife)
- Combined slope significance reporting
- Two-pass PAH correction to T_dust
- σ_SFR analysis (pending stacking run 4)
- Accordion vs uniform z-bin verdict

These belong in pah-forward-model-4 or a dedicated analysis pass once the figures are
accepted as the narrative vehicle.
