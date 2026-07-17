# PAH Forward Model — Branch 10 Summary

**Goal.** Extend the branch-4 K-fold COSMOS2020 catalogs with an optional starburst
label, regenerate them, and revive the dz-adaptive dithered z-binning approach out to
z=8 — far enough that MIPS 24 µm reaches rest-frame 3.3 µm
(`pah_model.py::PAH_FEATURES_MINOR`), never reachable in any prior config.

See `pah-forward-model-10-brief.md` for the full derivation and process notes (including
a course-correction on how starburst should be represented — see below).

---

## Headline results

**1. Starburst is a binning dimension, not a new `population_class` code.**
`prepare_cosmos2020_catalog.py` now writes `starburst` (0/1) + `log_delta_ms` columns
(Elbaz+2018 SFR/SFR_MS>3 against Schreiber+2015, **recentered** to this catalog's own SF
ridge — see finding 2). `population_class` stays the existing 0=sfg_signal/1=sfg_nuisance/
2=qt K-fold scheme untouched. "3 populations" (SFG-MS, SFG-SB, QT) come from crossing
`population_class` × `stellar_mass` × `starburst` in the TOML's generic
`[catalog.classification.binning.*]` machinery (`PopulationManager`'s
`itertools.product`), mirroring the pattern `prepare_cosmos2025_catalog.py` already
established for COSMOSWeb. An initial attempt in this branch gave starbursts their own
K-fold split before being corrected — see the brief's Objective 1 for why that's the
wrong shape here (starves an already-rare subsample further, no benefit over the
binning-dimension approach since `PopulationManager` already crosses every dimension
with every class).

**2. COSMOS2020's LePhare SFRs run ~0.21 dex above the Schreiber+2015 main sequence —
a real zero-point mismatch, not a bug.** The literal Elbaz+2018 3× cut flagged 30% of
SFGs as "starburst" before recentering (verified `lp_SFR_best` is self-consistent with
`lp_sSFR_med`, corr=0.87 — ruling out a units error). Recentering by the catalog's own
median SF-population offset (the same `s_pivot` self-calibration idea already used in
`pah_spectrum.py`) gives 17.4% — kept as the working definition rather than switching
to a percentile target, since it's still literature-traceable and gives good per-bin
statistics.

**3. Found and fixed a real bug: the star/galaxy cut has silently never fired.**
`prepare_cosmos2020_catalog.py`'s default `star_col="type"` never matched (the real
COSMOS2020 FARMER column is `lp_type`) — every catalog this script has ever produced,
including the currently-used `cosmos2020_PAH_split{0,1,2}of3.parquet`, never actually
excluded stars. Default fixed to `"lp_type"`. **Audited the other two catalog-prep
scripts for the same class of bug — not a repeat**: `prepare_cosmos_catalog.py`
(COSMOSWeb) uses a TOML-configured `star_flag="flag_star"`, verified against the real
FITS files it loads (`flag_star` exists in `COSMOSWeb_mastercatalog_v1.fits`, the
configured `phot_file`); `prepare_cosmos2025_catalog.py` imports that same function, no
separate logic. Both `_apply_quality_cuts` implementations fail *silently* (print
"SKIPPED", no exception) on a column-name mismatch — worth a test asserting the star cut
actually removes sources on real data, since nothing currently catches a future rename.

**4. The "~280 bin" ceiling has a precise origin, and adding starburst + a 5th mass bin
costs roughly ×3 at fixed z-binning.** The *current production*
`cosmos20_PAH_dithered_3cats.toml` (pre-branch-10, z≤5.0, 4 mass bins, no starburst)
already sits at exactly 280/288 possible nonempty populations — `PopulationManager`
crosses every bin-dimension combination with every `population_class` value via
`itertools.product`. Re-deriving z-bins to z=8 with starburst included, keeping the
historical dz≤0.15 core resolution, and the 5-mass-bin scheme the user restored, all
simultaneously is not possible under that budget — the core region had to widen to
dz≤0.50 (tail dz≤1.00 past z=3.0), verified against the real K-fold catalogs at
202–247 populations across all 3 staggered dither runs (vs. 280–327 with the tighter
dz≤0.35 scheme once the 5th mass bin was restored).

**5. n_stagger=3, not 4 — the dither count must match K.** The first pass used
`n_stagger=4`, carried over from the pah-forward-model skill doc's pre-K-fold
"4 dither runs" convention. `pah-forward-model-4-brief.md` established `n_stagger=K`
("3 runs × K=3 catalogs = 9 stacking jobs") once K-fold splitting exists — the user
caught the mismatch ("4 set of zbins and 3-fold catalogs is not compatible") and it's
fixed throughout: both TOML files, both mass-bin schemes (the 4-mass-bin alternate is
now fully staggered too — all 3 runs, not just 1 — and explicitly paired/labeled so its
z-bins can't be mixed with the active 5-bin scheme's).

---

## New/changed columns and files

| Where | What |
|---|---|
| `prepare_cosmos2020_catalog.py` | `_flag_starbursts()`; `starburst`/`log_delta_ms` output columns; `--no-starburst`/`--sfr-col`/`--sfr-linear`/`--starburst-threshold` CLI; `star_col` default `"type"`→`"lp_type"` |
| `$CATSPATH/cosmos/cosmos2020_PAH_3pop_catalog.parquet` | pooled, z≤8.0, 480,414 sources |
| `$CATSPATH/cosmos/cosmos2020_PAH_3pop_split{0,1,2}of3.parquet` | K-fold, z≤8.0 |
| `config/cosmos20_PAH_dithered_3pop.toml` | pooled TOML; starburst binning dim; z=8 staggered bins, n_stagger=3 (122–148 populations, 5-bin mass) |
| `config/cosmos20_PAH_dithered_3pop_3cats.toml` | K-fold TOML; same, pointed at `split0of3.parquet` (202–247 populations, 5-bin mass) |

Reproduction:
```bash
uv run prepare-cosmos2020-catalog --catalog "$CATSPATH/cosmos/cosmos2020_FARMER.csv" \
    --stem cosmos2020_PAH_3pop --z-max 8.0 --splits 1
uv run prepare-cosmos2020-catalog --catalog "$CATSPATH/cosmos/cosmos2020_FARMER.csv" \
    --stem cosmos2020_PAH_3pop --z-max 8.0 --splits 3
```

---

## Verification

- Full test suite: 270 passed, 1 `@pytest.mark.slow` deselected — no regressions.
- Population counts verified two independent ways: (1) hand-counted directly against
  the real parquet files with the exact z/mass/starburst/split cross-product logic
  `PopulationManager._create_populations` uses; (2) `SimstackWrapper(read_maps=False,
  read_catalog=True)` catalog-only load against the actual committed TOML files. Both
  agree exactly (122/204 for pooled/K-fold run-0).

---

## Recommendations / next steps

- **No real stacking (maps) has been run yet on the new configs** — only population
  *counts* are verified, not stacked *fluxes*. The z~5.2–8 tail (where 3.3 µm actually
  falls) is source-sparse; whether it's even marginally detectable (Tier A/B/C mix) is
  unknown until a real run.
- Only the offset-0 z-bin run is active per TOML; the 3 staggered alternates are
  commented but unexercised.
- `PAHModel(include_minor_features=True)` needs to be wired into an analysis notebook to
  actually turn on 3.3 µm fitting against this data — separate, analysis-time work.
- Consider adding a regression test asserting the star/galaxy cut removes a nonzero
  fraction on real data, for both `prepare_cosmos2020_catalog.py` and
  `prepare_cosmos_catalog.py` — the bug found here would not have been caught otherwise.
- `starburst`/`log_delta_ms` are now available for a σ_SFR-style cross-cut (branch-2's
  Run 2c precedent) if wanted later — not exercised in this branch.

---

## Files changed (tracked)

- `src/simstack4/scripts/prepare_cosmos2020_catalog.py`
- `config/cosmos20_PAH_dithered_3pop.toml`, `config/cosmos20_PAH_dithered_3pop_3cats.toml`
- `docs/pah-forward-model-10-brief.md`, `docs/pah-forward-model-10-summary.md`

(New parquet catalogs live under `$CATSPATH`, outside the repo, per project convention.)
