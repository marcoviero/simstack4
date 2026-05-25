# Context for three-component-graybody-1

Read `CLAUDE.md` first — this document only covers what's new on this branch.

## Science motivation

`DustEvolutionModel` currently fits a two-component SED:

```
F_ν = A_c · [GB(T_c, β=1.8) + f_w · GB(T_w, β=1.5)]
```

The "warm" component is a single modified blackbody that must simultaneously explain
two physically distinct emission mechanisms:

1. **Hot dust continuum** — small grains stochastically heated to T ~ 100–300 K
   inside PDRs; SED peaks at 20–40 µm.
2. **PAH features** — large aromatic molecules emitting at 6.2, 7.7, 8.6, 12.7 µm
   via single-photon heating; not a thermal continuum at all.

We already handle (2) with `PAHModel` (zero free parameters, calibrated from COSMOS
stacking). But the `GB(T_w)` component still conflates the two, which is why the
MAP gets stuck at `b_z = 0`: `f_w` inflates to match the PACS 100 µm slope whether
the cause is PAH, hot continuum, or a real T_cold rise.

**The fix**: replace `GB(T_w)` with a physically separated two-sub-component warm SED:

```
F_warm(λ) = f_hd · GB(T_hd, β=1.5)   [hot dust continuum, peaks ~25 µm]
           + f_pah · PAH(λ)            [PAH features, peaks ~8 µm]
```

This gives the optimizer distinct spectral levers for 8–15 µm (PAH) vs. 20–60 µm
(hot dust) vs. 80–500 µm (cold dust), removing the degeneracy that forces `b_z = 0`.

Full three-component SED (9 or 10 global parameters):

```
F_ν = A_c · GB(T_c, β=1.8)            [cold dust, ~24–35 K]
    + A_hd · GB(T_hd, β=1.5)          [hot dust continuum, ~100–300 K]
    + A_pah · PAH_template(λ)         [PAH features]

T_c(z)     = T_c0 + b_z · z
T_hd(σ)   = T_hd0 + c_σ · log_σ_SFR
log_f_hd  = a0 + a_z·z + a_M·log_M*
A_pah     = f(log_L_IR, z)            ← from PAHModel (zero extra free params)
```

The key gain: **`b_z` becomes identifiable** because no single component is flexible
enough to absorb both the short-wavelength (PAH) and mid-wavelength (hot continuum)
excess simultaneously.

## What CIGALE / LEPHARE give us

Our current PAH Gaussians and the hot-dust greybody are phenomenological. LEPHARE
carries physics-grounded empirical template libraries that directly span the
wavelength range of interest:

| Library | Origin | Useful content |
|---------|--------|----------------|
| **Dale & Helou 2002 / Dale+14** (LEPHARE) | Empirical IR template sequence | SED shape as a function of α (ISRF gradient), hot-to-cold ratio |
| **Chary & Elbaz 2001** (LEPHARE) | Observed galaxy SEDs | Empirical L_IR–SED shape relation |
| **Lagache+04** (LEPHARE) | Two-component: starburst + cirrus | Explicit separation of warm/cold components |

CIGALE (Draine+Li 2007 radiative-transfer grid) is held in reserve: use it if the
LEPHARE templates give poor fits or if we need physically motivated `T_hd` priors
from the `Umin`/`Umax` grid.

**Immediate scientific value from LEPHARE**:

1. **Validate PAH amplitudes**: do our Gaussian feature strengths match the
   template library at the COSMOS L_IR range?
2. **Constrain hot-dust SED shape**: the mid-IR continuum under the PAH features
   in Dale+14 / CE01 templates defines the shape we want for `GB(T_hd)`.
3. **Constrain T_hd prior**: effective dust temperature derived from template
   peak position sets physically motivated bounds for `T_hd0`.
4. **Cross-check f_hd vs L_IR**: Dale+14 hot-to-cold fraction as a function of
   α (ISRF slope, correlates with sSFR) — compare against our `a_z`, `a_M` recovery.

## Templates — data on hand

### COSMOSWeb LEPHARE SED catalog (primary)

```
COSMOSWeb-LePhare-SEDs_v1.1.h5
```

This is the COSMOSWeb photometric redshift run: per-source best-fit LEPHARE SEDs
stored in HDF5. It gives us the actual fitted SED shapes at known redshifts for
the same COSMOS field we stack — a direct apples-to-apples comparison.

Workflow:
- Open with `h5py`; explore keys to find the SED arrays (λ, F_ν), redshifts,
  stellar masses, and L_IR estimates.
- Select sources that overlap with our stacking bins (same z, M* ranges).
- Stack or median the per-source SEDs within each bin → "template SED per bin".
- Compare median template to our two-component model prediction for that bin.

This sidesteps downloading a generic template library entirely — we have the
real fitted SEDs for our own sources.

### CIGALE (Draine+Li 2007 — held in reserve)
```
# Only if COSMOSWeb LEPHARE SEDs give poor coverage (e.g. missing mid-IR)
https://cigale.lam.fr/
# Module: dust_emission (draine2007)
# Grid files available via: pcigale --help → data download
```

### Spitzer/IRS spectral decomposition (observational anchor)
Smith+07 (ApJ 656) provides measured PAH equivalent widths and feature ratios from
59 SINGS galaxies. This is the empirical ground truth our Gaussians in `pah_model.py`
should reproduce.

## What to build on this branch

### Step 0: Template survey (before any code)

Open `COSMOSWeb-LePhare-SEDs_v1.1.h5` with `h5py`; map the file structure (keys, array shapes, units).
- Select sources matching our stacking bin boundaries (z, M*).
- Median-stack per-source SEDs within each bin → one "template SED per bin".
- Separate each stacked SED into PAH region (6–15 µm) and hot continuum (15–60 µm)
  by subtracting a fitted modified blackbody from the long-wavelength anchor.
- Measure: effective T_hd per bin, ratio `L_PAH/L_IR` vs. L_IR and M*, hot-to-cold
  ratio as a function of redshift.

Deliverable: a notebook (`notebooks/2026-05-25-lephare-template-survey.ipynb`) that
loads the templates and plots:
1. Full SED library colored by qPAH.
2. PAH region overlay vs. our `PAHModel` prediction.
3. Hot-continuum residual after PAH subtraction, colored by Umin.
4. T_hd(Umin) relation — this becomes the prior for `T_hd0`.

### Step 1: Update `dust_evolution.py`

Split the warm component into two:

```python
@dataclass
class DustEvolutionModel:
    use_pah: bool = True         # carry PAH features (zero extra params)
    use_hot_dust: bool = True    # separate hot-dust continuum
    T_hd_prior: tuple = (120.0, 50.0)  # (center, sigma) from template survey
```

New global params added to `theta`:
- `T_hd0` — hot dust anchor temperature (~100–200 K)
- `c_hd_sigma` — T_hd slope vs. log_σ_SFR (optional; can fix=0 initially)
- `log_f_hd0`, `f_hd_z` — hot dust fraction evolution (parallels `a0, a_z`)

The analytic `A_c` solve is preserved; `A_hd` is solved analytically at fixed
`T_hd` and `T_c` in the same least-squares step.

Likelihood structure is unchanged: loop over bins, compute model SED, compare to
stacked fluxes with noise floor.

### Step 2: Synthetic recovery test

Before fitting real data, verify identifiability:
- Generate synthetic data with known `b_z=3`, `T_hd=150 K`, `f_hd=0.15`, PAH at
  COSMOS-typical amplitude.
- Fit with three-component model; confirm `b_z` recovers within 1σ.
- Fit with two-component model on the same data; confirm `b_z ≈ 0` (reproduces the
  current failure mode, proving the three-component model fixes it).

This test becomes `test_b_z_recovery_three_component` in
`test_dust_evolution_recovery.py`.

### Step 3: COSMOS fit

Re-run the end-to-end fit on `cosmos25_stacking_20260317_201727.json` (same data
as the dust-temp-evol-2 notebook) with the three-component model.
Compare `b_z` posteriors: two-component vs. three-component.

New notebook: `notebooks/2026-05-25-three-component-graybody.ipynb`

### Step 4: Template validation overlay

On the best-fit COSMOS posterior:
- Overplot the Draine+Li template at matching qPAH, Umin against the decomposed
  per-bin SEDs.
- Report residuals in each band — do the CIGALE templates predict the right hot-dust
  to PAH ratio for our population?

This is the "legitimize with physics" step.

## Constraints

- Do not change `pah_model.py` — read-only.
- Do not break existing 14 tests in `test_dust_evolution_recovery.py`.
- Keep `A_c` analytic; extend the analytic solve to `A_hd` where possible.
- CIGALE templates go in a new top-level `templates/` directory (not tracked by git
  unless small; add `templates/*.fits` to `.gitignore`).
- If `lephare` or `cigale` Python packages are added as optional dependencies, gate
  them behind `try/except ImportError` so the package installs without them.

## Data

```
# Stacking results (unchanged from dust-temp-evol-2)
/Users/mviero/data/Astronomy/pickles/simstack/stacked_flux_densities/cosmos25_stacking_20260317_201727.json

# COSMOSWeb LEPHARE SED catalog (transfer from other Mac via Thunderbolt)
COSMOSWeb-LePhare-SEDs_v1.1.h5
# Expected destination: /Users/mviero/data/Astronomy/COSMOSWeb/COSMOSWeb-LePhare-SEDs_v1.1.h5

# CIGALE templates (held in reserve — only if LEPHARE SEDs are insufficient)
# /Users/mviero/data/Astronomy/templates/cigale/draine2007/
```

## Run tests

```bash
uv run pytest tests/test_dust_evolution_recovery.py -v
```
