# PAH Forward Model — Branch 10 Brief: starburst-labeled 3pop catalogs + z=8 dithered binning

**Goal.** Revisit the K-fold COSMOS2020 catalog + dithered z-binning scheme built in
branch 4 (`docs/pah-forward-model-4-brief.md`, "K-fold Source Partitioning") along three
axes: (1) let `prepare_cosmos2020_catalog.py` optionally label starbursts, (2) regenerate
the K-fold catalogs under a `cosmos2020_PAH_3pop` stem, and (3) revive the dz-adaptive/
staggered binning approach — narrow bins where sources are dense, wide where sparse,
governed by how well a given dz samples the PAH ladder as convolved through the MIPS
bandpass — and push it out to z=8 so MIPS 24 µm reaches rest-frame 3.3 µm (the aromatic
C–H stretch, `pah_model.py::PAH_FEATURES_MINOR`), which no prior config has reached.

---

## Objective 1 — starburst labeling: binning dimension, not a new population_class code

### Problem

"Add starbursts" has two structurally different implementations: give starbursts their
own `population_class` code (so they get their own K-fold signal/nuisance split, valid
χ² the same way SFG/QT do), or flag them as a column and let the existing generalized
`[catalog.classification.binning.<dim>]` machinery cross them with mass/z.

The first attempt in this branch picked the former — reasoned it gave a real measured
`α_SB(M*)` at the cost of generalizing `_assign_kfold_splits`/`_make_split_population_class`
beyond class 0/1/2. The user corrected this: **`prepare_cosmos2025_catalog.py` already
solved this problem for COSMOSWeb**, and picked the second pattern deliberately — "mimic
that unless it's suboptimal." It isn't: K-folding an already-rare subsample (starbursts
are ~a few–20% of SFGs depending on threshold) three ways on top of a mass/z grid starves
it further, and `PopulationManager._create_populations` already crosses every binning
dimension with every `population_class` value via `itertools.product`, so starburst-as-a-
binning-dimension gets the "3 populations" (SFG-MS, SFG-SB, QT) result for free, no changes
to the K-fold code at all.

### Calibration problem found along the way

Mimicking `prepare_cosmos_catalog.flag_starbursts` literally (Elbaz+2018: SFR/SFR_MS > 3,
Schreiber+2015 Eq. 9 main-sequence SFR) on COSMOS2020 flagged **30% of SFGs** as
"starburst" — nowhere near the few-percent rarity the term implies. Diagnosis: this
catalog's LePhare `lp_SFR_best` sits ~0.21 dex above the Schreiber+2015 ridge (verified
`lp_SFR_best` is self-consistent with `lp_sSFR_med`, corr=0.87, median diff≈0 — a genuine
zero-point mismatch between SFR estimators/IMFs, not a units bug). Recentering by the
catalog's own median SF-population offset (same self-calibration idea as `pah_spectrum.py`'s
`s_pivot`) before applying the threshold gives **17.4%** — still broader than the canonical
literature rarity (because this catalog's residual scatter about the MS is wider than the
relation assumes), but the user chose to keep the literal 3× cut post-recentering rather
than switch to a percentile-targeted definition, since 17% still gives good per-bin
statistics and stays literature-traceable.

### Implementation

`prepare_cosmos2020_catalog.py`:
- `_flag_starbursts()` — recentered Elbaz+2018 cut, writes `starburst` (0/1) and
  `log_delta_ms` (recentered log10(SFR/SFR_MS)) columns. Rides through the K-fold `df.copy()`
  loop unchanged — no touches to `_assign_kfold_splits`/`_make_split_population_class`.
- `--no-starburst` to skip; `--sfr-col`/`--sfr-linear`/`--starburst-threshold` to override.
- Fixed a latent bug found while touching this file: default `star_col="type"` never
  matched (the real COSMOS2020 FARMER column is `lp_type`) — every catalog produced by this
  script before this branch, including the currently-used `cosmos2020_PAH_split{0,1,2}of3.parquet`,
  silently never excluded stars. Default is now `"lp_type"`.

### Star-cut bug: where else it could be a problem — audited, not a repeat elsewhere

Checked both other catalog-prep scripts for the same class of bug (a hardcoded
star/galaxy column name that silently never matches the real column, since
`_apply_quality_cuts`/`apply_quality_cuts` both fail *open* — `if star_col and star_col in
df.columns` — so a wrong name doesn't error, it just quietly skips the cut):

- **`prepare_cosmos_catalog.py`** (COSMOSWeb): `star_col = cols.get("star_flag")` is
  TOML-configured, not hardcoded — `config/catalog/defaults.toml` sets
  `star_flag = "flag_star"`. Verified directly against the real FITS files this script
  loads (`load_fits_catalogs`, merging `phot_file` + `sed_file` + optional `cigale_file`):
  `flag_star` does NOT exist in `COSMOSWeb_mastercatalog_v1_lephare.fits` (the `sed_file`)
  or the CIGALE table, but DOES exist in `COSMOSWeb_mastercatalog_v1.fits` — which is
  exactly `defaults.toml`'s configured `phot_file`, the first table merged in. **Correctly
  wired**, not a repeat of the bug.
- **`prepare_cosmos2025_catalog.py`**: imports `apply_quality_cuts`/`flag_starbursts`
  etc. from `prepare_cosmos_catalog.py` directly (no separate star-cut logic of its own)
  — covered by the check above.
- **`sky_catalogs.py`** (the live TOML pipeline `SkyCatalogs`/`PopulationManager` code
  path): no star/galaxy filtering at all — that responsibility lives entirely in the
  catalog-prep scripts, pre-computed before the parquet is fed into a TOML. Nothing to
  audit there.

**Conclusion: this was specific to `prepare_cosmos2020_catalog.py`'s Python-level
hardcoded default, now fixed. Not a systemic pattern.** The general lesson — worth
carrying forward — is that both `_apply_quality_cuts` implementations fail silently
(print a "SKIPPED" line, no exception) when the configured/default column name doesn't
match, so a future rename in an upstream catalog would reproduce this class of bug without
any test catching it. No test currently asserts the star cut actually removes anything on
real data for either script.

---

## Objective 2 — regenerate the K-fold catalogs

Reproduction (from repo root, with `$CATSPATH` set):

```bash
uv run prepare-cosmos2020-catalog \
    --catalog "$CATSPATH/cosmos/cosmos2020_FARMER.csv" \
    --stem cosmos2020_PAH_3pop --z-max 8.0 --splits 1   # pooled

uv run prepare-cosmos2020-catalog \
    --catalog "$CATSPATH/cosmos/cosmos2020_FARMER.csv" \
    --stem cosmos2020_PAH_3pop --z-max 8.0 --splits 3   # K-fold
```

→ `cosmos2020_PAH_3pop_catalog.parquet` (pooled) and
`cosmos2020_PAH_3pop_split{0,1,2}of3.parquet` (K-fold), written to
`$CATSPATH/cosmos/` (same directory as `--catalog` by default; override with
`--output-dir`). z≤8.0, mass>8.0, `lp_type==0` galaxies only (the bug fix above).
480,414 sources; 455,207 SFG / 25,207 QT; 79,398 SFG (17.4%) flagged starburst
(identical starburst fraction/counts in both invocations — classification runs before
the K-fold split). Default starburst settings (`--sfr-col lp_SFR_best
--starburst-threshold 3.0`, recentered) match Objective 1; pass `--no-starburst` to
reproduce the pre-branch-10 2-class-only catalogs, or `--seed` to change the K-fold
random partition (default 42, reproducible).

---

## Objective 3 — revive the adaptive dz binning, extended to z=8

### The tool, not hand-rolled math

`analyze_pah.py::staggered_pah_zbins` (wrapping `adaptive_pah_zbins`, exposed as
`pah_dither.py::DitherScheme.adaptive`) already implements exactly the "narrow where
dense, wide where sparse, respect a max dz for PAH-ladder sampling" logic — the existing
`cosmos20_PAH_dithered.toml`'s active z-bins (0.5→5.0, irregular widening past z~2.6) are
already this machinery's output, just capped at z=5.0. This branch re-runs it with
`z_max=8.0` and a `secondary={"stellar_mass": ...}` guarantee, rather than hand-tuning
new numbers.

### The real constraint: total population count, not just per-bin source count

The user's framing ("widen the zbins and add starbursts with the goal of having the
total number of bins manageable, under ~280") turned out to have a precise empirical
anchor: the *current production* `cosmos20_PAH_dithered_3cats.toml` (pre-branch-10, z≤5.0,
4 mass bins, no starburst dimension) already sits at **exactly 280 of 288 possible**
nonempty populations, counted against the real `cosmos2020_PAH_split2of3.parquet`. That's
where the ~280 ceiling comes from — it isn't a soft guess, it's the actual current budget.

`PopulationManager._create_populations` crosses **every** bin-dimension combination with
**every** `population_class` value (signal/nuisance/qt) via `itertools.product` — so
adding starburst as a binning dimension roughly doubles the total at fixed z/mass binning,
and restoring the 5th mass bin (9.0–9.9, added back by the user after the first version of
this branch dropped it) roughly triples it. Verified directly against the real 5-bin-mass
scheme: with the branch's first z-bin attempt (core dz≤0.35/tail dz≤0.70, tuned for 4 mass
bins), the K-fold config landed at 280–327 populations across the 4 staggered runs — over
budget the moment the 5th mass bin came back.

### n_stagger=3, not 4 — matching the K=3 catalogs

The branch's first pass used `n_stagger=4`, carried over from the pah-forward-model skill
doc's "actual stacking scheme used: dz=0.15, 4 dither runs." That number predates K-fold
source partitioning (branch 4) — it describes the *pooled*, unsplit-catalog scheme.
`pah-forward-model-4-brief.md` explicitly built a **3-run equivalent** of that 4-run
scheme specifically to pair with K=3 catalogs ("3 runs × K=3 catalogs = 9 total stacking
jobs"), and the user caught the branch-10 configs repeating the older, pre-K-fold number:
"4 set of zbins and 3-fold catalogs is not compatible." Fixed by re-deriving with
`n_stagger=3` throughout. Cost is small — staggering is sub-additive at the kernel floor
(branch-2 finding), so the dropped 4th offset buys little resolution anyway.

### Final scheme

Two-region `staggered_pah_zbins` call spliced at z=3.0 (core: z=0.5–3.0, tail: z=3.0–8.0),
run separately per region so each keeps its own edges pinned at the splice point across
all 3 staggered offsets (n_stagger=3), then concatenated:

| Region | min_sources | max_dz | Why |
|---|---|---|---|
| core (0.5–3.0) | 500 | 0.50 | R~5-7; more resolution cost than the historical dz=0.15/R~10-20, unavoidable once starburst + the 5th mass bin roughly triple the population count at fixed z binning |
| tail (3.0–8.0) | 400 | 1.00 | sparse regime; the z~5.2–8 window where 3.3 µm actually falls is upper-limit/marginal-SNR territory, not detection-grade |

Verified against all 3 real K-fold split parquets and the pooled parquet, all 3 staggered
runs: **202–247** populations (K-fold) / **122–148** (pooled) — comfortably under 280.
Cross-checked independently via `SimstackWrapper(read_maps=False, read_catalog=True)`
catalog-only loads against the actual TOML files (122 pooled / 204 K-fold split0, run 0),
which matched the hand-count exactly.

The previous 4-mass-bin z-bin scheme (dz≤0.35 core / 0.70 tail, also re-derived at
n_stagger=3, 215–252 populations / 133–153 pooled) is kept as a **fully staggered**
commented alternative in both TOML files (all 3 runs, not just run 0), paired explicitly
with the 4-bin mass edges and labeled so the two schemes' z-bin/mass-bin blocks can't be
mixed — a 5-bin-derived z-grid against 4-bin masses wastes the freed budget headroom.
Spot-verified by temporarily activating the 4-bin pair end-to-end (220 populations for
K-fold split0, run 0 — matches the hand-count exactly).

---

## Outcome (2026-07-17) — committed, not yet merged, stacking not yet run

Commit `ab51335` + follow-up on `pah-forward-model-10` (n_stagger 4→3 fix, full 4-bin
alternate staggering, this doc). Delivered: `prepare_cosmos2020_catalog.py` starburst
flagging + `lp_type` fix; `cosmos2020_PAH_3pop_catalog.parquet` +
`_split{0,1,2}of3.parquet`; `config/cosmos20_PAH_dithered_3pop.toml` +
`_3pop_3cats.toml` with the starburst binning dimension and z=8 staggered bins at
n_stagger=3 (one active run + 2 commented per mass scheme, both mass schemes fully
staggered). Full test suite (270 passed, 1 slow-marked deselected) and catalog-load
smoke tests both pass.

**Not done / next branch:**
- No actual stacking (real maps) run yet on any of the new configs — population *counts*
  are verified, stacked *fluxes* are not. The z~5.2–8 tail's usability (Tier A/B/C
  breakdown, whether 3.3 µm is even marginally detectable) is unknown until a real run.
- Only run-0 (offset 0) of the 3 staggered z-bin sets is active in each TOML; runs 1–2
  are commented but not yet exercised.
- `PAHModel(include_minor_features=True)` (which actually turns on the 3.3 µm feature in
  the fitting machinery) hasn't been wired into an analysis notebook against this new
  data — that's a notebook/analysis-time argument change, out of scope for this branch's
  catalog/TOML work.
- σ_SFR-style cross-cut extensions (branch-2's Run 2c precedent) weren't revisited here;
  `starburst`/`log_delta_ms` are new columns available for that if wanted later.
