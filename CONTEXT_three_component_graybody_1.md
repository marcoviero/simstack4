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

## What CIGALE gives us

`CIGAL_SEDs_v1.tar.gz` (187.83 GB, on the remote machine) contains per-source
CIGALE SED fits for the COSMOSWeb field. CIGALE fits the full UV-to-submm SED using
stellar + dust emission modules (Draine+Li 2007 or Dale+14), so these SEDs cover the
FIR range we need — unlike the COSMOSWeb LEPHARE catalog which only reaches ~12 µm.

**What we extract from CIGALE SEDs**:

1. **PAH amplitude vs. L_IR, z, M***: does our `PAHModel` calibration hold across
   the parameter space?
2. **Hot-dust continuum shape** (15–60 µm residual after PAH subtraction): validates
   or replaces the `GB(T_hd, β=1.5)` ansatz.
3. **T_hd prior**: effective dust temperature of the hot component per bin, from
   the Draine+Li `Umin`/`Umax` grid → sets `T_hd0` and its scatter.
4. **f_hd vs. z, M***: hot-to-cold fraction as a function of galaxy properties →
   physical prior on `a0`, `a_z`, `a_M`.

## Template survey — remote workflow

The 187.83 GB file lives on the other machine. The remote survey produces small
summary files (< 10 MB total) that are transferred back here for use in
`dust_evolution.py` and the fitting notebook.

See `REMOTE_CONTEXT_cigale_survey.md` for the self-contained prompt to run on
the remote machine. Expected outputs transferred back:

```
templates/cigale/
    cigale_sed_grid.npz          # median L_ν/L_IR per (z, M*) bin, rest-frame λ grid
    cigale_pah_amplitudes.csv    # L_PAH/L_IR vs log_L_IR, z, log_M* per source/bin
    cigale_hot_dust_shapes.npz   # continuum residual 15–60 µm per bin
    cigale_t_hd_grid.csv         # effective T_hd(z, M*, log_SFR) from Draine+Li Umin
```

## What to build on this branch

### Step 0: Template survey (remote machine)

Run `REMOTE_CONTEXT_cigale_survey.md` on the machine with `CIGAL_SEDs_v1.tar.gz`.
Transfer the four summary files above into `templates/cigale/` here.

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

On the best-fit COSMOS posterior, overplot the CIGALE median template per bin
against the decomposed model SED. Report residuals in each band.

New notebook: `notebooks/2026-05-25-cigale-template-survey.ipynb`
(populated after Step 0 outputs arrive from the remote machine)

## Constraints

- Do not change `pah_model.py` — read-only.
- Do not break existing 14 tests in `test_dust_evolution_recovery.py`.
- Keep `A_c` analytic; extend the analytic solve to `A_hd` where possible.
- CIGALE summary files go in `templates/cigale/` (add `templates/` to `.gitignore`
  for the large raw files; the small summary .npz/.csv files ARE committed).

## Data

```
# Stacking results (unchanged from dust-temp-evol-2)
/Users/mviero/data/Astronomy/pickles/simstack/stacked_flux_densities/cosmos25_stacking_20260317_201727.json

# CIGALE SED catalog — lives on remote machine only (187.83 GB)
# See REMOTE_CONTEXT_cigale_survey.md
```

## Run tests

```bash
uv run pytest tests/test_dust_evolution_recovery.py -v
```
