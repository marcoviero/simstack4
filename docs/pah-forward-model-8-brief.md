# PAH Forward Model — Branch 8 Brief: talk figure set

**Goal.** Turn the branch-7 results into a coherent, presentation-ready figure set for the
talk (branch-7 brief "Objective 4", now unblocked: the mechanism question has landed on
ionization-state + enrichment, no-AGN framing). Polish > new science: every figure should be
readable from the back of a room, self-explanatory in its caption, and consistent in style
across the set.

Model note: this branch is figure iteration — run it on Sonnet (`claude-sonnet-5`), not
Fable/Opus.

---

## Candidate figure inventory (from branch 7; regenerate, don't screenshot)

| # | Figure | Source | Status |
|---|---|---|---|
| 1 | Band ratio (12.7/6.2) vs M\*, envelope-aware + fold ensemble | `2026-07-03-pah-money-plots.ipynb` §2b (`pah_money_bandratio_vs_mass_envaware.png`) | money plot; needs style pass |
| 2 | Narayanan+26 confrontation (L_PAH/L_IR vs M\* + channel bands) | same notebook §3 (`pah_money_narayanan_confrontation.png`) | money plot; add estimator-systematic band? |
| 3 | Real-data f₂₄(z) decomposition (shaded feature contributions per mass bin) | `2026-07-02-pah-evolving-template-mcmc-simulation.ipynb` §7 (`pah_evolving_mcmc_f24_decomposition_real.png`) | the "how it works" figure — strong talk opener |
| 4 | Sim decomposition + truth overlay (method validation twin of #3) | same notebook §5 | pair with #3 or backup slide |
| 5 | Evolving-truth rest spectra / bandpass sweep (method cartoon) | same notebook §1 | candidate intro animation base (see `2026-04-28-pah-resolution-animation.ipynb`) |
| 6 | Static-vs-evolving Δχ² + fold table (evolution is required) | same notebook §7 | maybe a table-slide, not a figure |
| 7 | Existing talk assets to revisit | `notebooks/build_pah_fig1-5.py`, branch-3 brief figure list | audit which survive the branch-6/7 pivots |

## Tasks

1. **Style pass on the money plots** (#1, #2): consistent fonts/sizes for projection,
   colorblind-safe series colors (dataviz-skill palette; the decomposition figures already
   use the validated categorical slots), direct labels over legends where possible, units
   and pivot conventions stated on-axis. Keep one visual language across the set.
2. **Figure 2 upgrade**: show the estimator systematic honestly — e.g. the envelope-aware
   slope range (+0.13…+0.23) as a secondary measurement band or whisker, per branch-7
   summary. Decide one canonical slope to headline (free-α original) with the spread quoted.
3. **Decomposition figure (#3) talk variant**: possibly single-panel (one mass bin) for the
   main deck with the 4-panel as backup; larger annotations for the feature-group wedges.
4. **Audit `build_pah_fig1-5.py`** against the branch-6/7 supersessions (headline pivot,
   envelope-aware calibration): regenerate what survives, retire what doesn't.
5. **Uncommitted `plots.py` styling upgrades** (plot_pah_fit / plot_pah_forward_fit /
   plot_pah_vs_property\* / plot_pah_multibin_forward_fit — committed with branch 7):
   finish/verify against the current result set.
6. Keep numbers frozen: this branch changes presentation only. If a figure exposes a
   science problem, log it in this brief and defer to branch 9.

## Conventions

- One `build_pah_talk_figs.py` (or extend `build_pah_fig*.py`) per figure family; PNGs +
  PDFs into `notebooks/` (gitignored) with build scripts force-added, per repo convention.
- Figure captions drafted alongside each figure in the build script docstring, so the talk
  and a future paper pull from the same source.
