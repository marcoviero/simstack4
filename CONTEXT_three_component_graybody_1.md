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

Our current PAH Gaussians and the hot-dust greybody are phenomenological. CIGALE
and LEPHARE carry physics-grounded template libraries:

| Library | Origin | Useful content |
|---------|--------|----------------|
| **Draine & Li 2007** (CIGALE) | Full radiative-transfer grain models | qPAH, Umin/Umax, hot continuum vs. cold continuum split as a function of ISRF intensity |
| **Dale & Helou 2002 / Dale+14** (CIGALE, LEPHARE) | Empirical IR template sequence | SED shape as a function of α (ISRF gradient), hot-to-cold ratio |
| **Chary & Elbaz 2001** (LEPHARE) | Observed galaxy SEDs | Empirical L_IR–SED shape relation |
| **Lagache+04** (LEPHARE) | Two-component: starburst + cirrus | Explicit separation of warm/cold components |

**Immediate scientific value**:

1. **Validate PAH amplitudes**: do our Gaussian feature strengths match the
   Draine+Li qPAH grid at the COSMOS L_IR range?
2. **Constrain hot-dust SED shape**: Draine+Li gives the mid-IR continuum under
   the PAH features (the template after subtracting PAH Gaussians is the continuum
   we want for `GB(T_hd)`).
3. **Constrain T_hd prior**: the Draine+Li grid spans `Umin ∈ [0.1, 25]` and
   `Umax ∈ [10^3, 10^6]` — maps to effective dust temperature via
   `T_eff ∝ U^(1/(4+β))`. This sets physically motivated bounds for `T_hd`.
4. **Cross-check f_hd vs L_IR**: Dale+14 gives hot-to-cold fraction as a function of
   α (the ISRF slope parameter), which correlates with sSFR. We can compare against
   our `a_z` and `a_M` recovery.

## Templates to download

### CIGALE (Draine+Li 2007)
```
https://cigale.lam.fr/
# Module: dustatt_modified_starburst, dust_emission (draine2007)
# Grid files: dl2007_templates.fits or from pcigale data download
```

Key grid axes: `qPAH ∈ {0.47%, 1.12%, 1.77%, 2.50%, 3.19%, 3.90%, 4.58%}`,
`Umin`, `Umax`, `gamma`.

The templates are stored as `L_ν / L_IR` vs. λ in rest frame — directly comparable
to our normalized SED components.

### LEPHARE (Dale+14 or Chary-Elbaz)
```
https://www.cfht.hawaii.edu/~arnouts/LEPHARE/lephare.html
# Template dir: $LEPHAREDIR/sed/GAL/CHARY_ELBAZ/ or DALE/
# Or via Conda: conda install -c conda-forge lephare
```

### Spitzer/IRS spectral decomposition (optional, observational anchor)
Smith+07 (ApJ 656) provides measured PAH equivalent widths and feature ratios from
59 SINGS galaxies. This is the empirical ground truth our Gaussians in `pah_model.py`
should reproduce.

## What to build on this branch

### Step 0: Template survey (before any code)

Download the Draine+Li 2007 CIGALE templates.  For each model in the grid:
- Extract `L_ν / L_IR` at rest 6–100 µm.
- Separate the SED into "PAH region" (6–15 µm) and "hot continuum" (15–60 µm) by
  subtracting a fitted modified blackbody from the long-wavelength side.
- Measure: effective T_hd(Umin, Umax), ratio `L_PAH/L_IR` vs. `qPAH`, hot-to-cold
  ratio as a function of ISRF.

Deliverable: a notebook (`notebooks/2026-05-xx-cigale-template-survey.ipynb`) that
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

New notebook: `notebooks/2026-05-xx-three-component-graybody.ipynb`

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

# Templates (to be downloaded)
/Users/mviero/data/Astronomy/templates/cigale/draine2007/
/Users/mviero/data/Astronomy/templates/lephare/dale2014/
```

## Run tests

```bash
uv run pytest tests/test_dust_evolution_recovery.py -v
```
