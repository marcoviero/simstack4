# pah-forward-model-12 — summary

**Branch goal** (see `pah-forward-model-12-brief.md`): rebuild the tomographic PAH fit as a
confusion-aware "simstack for the lines" — tie the unresolvable features at physical ratios
instead of fitting a free amplitude per group, then add the two missing physical components
(a **plateau** and the **9.7 µm silicate dip**) and see whether they absorb the coherent
residual inflating χ².

**Headline: the residual is absorption, not a missing emission plateau.** The plateau
hypothesis is rejected by the data; silicate is detected at τ ≈ 0.55 and it, not the 8.6 µm
split, is what had been inflating the within-bin evolution slope η_A.

## What was built

All in `src/simstack4/pah_spectrum.py`:

1. **`PHYSICAL_GROUPS = [[1,2],[0],[3,4]]` + `FEATURES_86_CALIBRATED` are now the
   `PAHSpectrumModel` defaults.** 7.7+8.6 welded at peak ratio 0.5 and used as the reference
   group; 6.2 and 11.3+12.7 free. 16.4+17.0 and 3.3 µm are deliberately absent (they rail /
   are unsampled on the real z-range). `DEFAULT_FEATURES`/`DEFAULT_GROUPS` stay importable —
   pass them explicitly to reproduce any pre-branch-12 fit.
2. **Plateau component** — `plateaus=DEFAULT_PLATEAUS` (broad Drude, λ₀=8.2, FWHM=4.0 µm) adds
   non-negative per-mass-bin columns to `fit_shared`/`fit_evolving`, contiguous with the hot
   ladder. Off by default.
3. **9.7 µm silicate** — `include_silicate=True` fits one global τ_sil ≥ 0 through the outer
   optimizer. The screen is applied **inside** the feature/plateau band integrals (kernels
   tabulated over a τ grid and interpolated) and as a response-weighted band transmission on
   the smooth components. `silicate_scope="features"` leaves the baseline unscreened;
   `tau_sil_prior` regularizes toward zero.
4. **`fit_evolving_mcmc` samples τ_sil** (appended last to θ so the η/log-r slices keep their
   indices; `evolving_flux_decomposition` applies the sampled screen). It still rejects the
   non-negative plateau/hot blocks, which would need an NNLS inside the profiled likelihood.
5. Guards: `fit_lstsq`/`fit_mcmc` raise if any extra component is configured
   (`_reject_unwired`). `tau_sil=0` short-circuits to byte-identical pre-branch code.

Tests: 24 new in `tests/test_pah_shared_baseline.py` (`TestPlateau`, `TestSilicate`,
`TestPlateauSilicateWiring`, `TestSilicateMCMC`). Full suite **305 passed**.

## Results on the real data

2026-07-23 K-fold COSMOS2020 stacks (`20260723_181128/_183744/_190437`), star-forming
`split_0`, Tier C, `starburst_filter=0`, mass bins [9.9, 10.6, 10.8, 11.0, 11.5], pooled
176 points; `fit_evolving`, drude, MIPS 24+70, `feature_envelope="baseline"`,
`eta_prior_sigma=1.0`.

### Relaxation ladder (pooled)

| config | χ²_red | η_A | τ_sil | plateau |
|---|---|---|---|---|
| L0 welded-physical (new default) | 5.654 | +0.621 | — | — |
| L1 + plateau | 5.728 | +0.621 | — | **0, 0, 0, 0** |
| L2 + silicate | **4.646** | **+0.107** | **0.489** | — |
| L3 + plateau + silicate | 4.701 | +0.120 | 0.494 | 0, 0.009, 0, 0 |
| S0 free-8.6 split (old default) | 5.397 | +0.154 | — | — |
| S3 free split + plateau + silicate | 4.518 | +0.136 | 0.817 | 0, 0, 0, 0 |

### Fold ensemble (the branch's error convention — scatter over the 3 disjoint folds)

| grouping | silicate | τ_sil | η_A | d log A_pah / d log M* |
|---|---|---|---|---|
| welded | off | — | +0.664 ± 0.059 | +0.574 ± 0.092 |
| welded | **on** | **0.551 ± 0.010** | **+0.031 ± 0.041** | +0.356 ± 0.066 |
| free split | off | — | +0.153 ± 0.066 | +0.451 ± 0.081 |
| free split | **on** | **0.897 ± 0.094** | +0.109 ± 0.090 | +0.395 ± 0.054 |

## Conclusions

**1. The plateau is rejected — and the way it fails is the finding.** Constrained (q ≥ 0) it
pins at exactly zero in every configuration; χ²_red goes slightly *up* (5.654 → 5.728) purely
from the dof penalty. Released from non-negativity it goes strongly **negative**
(−0.027…−0.079 mJy against C_m ≈ 0.013–0.037 mJy) and χ²_red falls to 3.95 — and τ_sil then
collapses to 0. The negative plateau and the silicate screen are fitting **the same feature**.
The data want *absorption* near rest 8–10 µm; silicate is the physical parametrization of it
and the negative plateau is an unphysical proxy. This is consistent with the brief's residual
anatomy (model over-predicts by −4.7σ at 9.0–9.8 µm).

The machinery is kept (a plateau is a real PAHFIT component and may matter for other data or
z-ranges) but it is **off by default and should stay off here**.

**2. Silicate is detected.** τ_sil = 0.551 ± 0.010 (welded) / 0.897 ± 0.094 (free split) from
the fold ensemble — both far from zero, with a prior that pulls *toward* zero. The brief was
right that the earlier per-fold τ ≈ 0 was confounded by free per-group amplitudes and no
plateau. Δχ²_red ≈ 1.0 is the single largest improvement anywhere in the ladder.

**3. η_A was mostly unmodelled silicate, not the 8.6 µm split.** This supersedes the brief's
framing. The brief expected welded-at-physical to give an honest η_A ≈ +0.6 upper limit versus
the free split's ≈ +0.15, and read the difference as a template-rigidity degeneracy. With
silicate in the model the two groupings **agree**: η_A = +0.031 ± 0.041 (welded) vs
+0.109 ± 0.090 (free split), both consistent with zero. The +0.62 was the 9.7 µm trough being
absorbed by the evolution term. Within-bin sSFR evolution remains an upper limit, now a
tighter one — but see §3a: it is *less* hostage to the 8.6 prior, not free of it.

**3a. The 8.6 µm strength and τ_sil are the same lever — quote them as a pair.**
8.6 µm sits on the **blue shoulder of the 9.7 µm trough** (Drude FWHM 3.3 µm → half-depth
spans rest ≈ 8.05–11.35 µm), so in a broadband tomographic fit "more 8.6 emission" and "more
9.7 absorption" are nearly the same degree of freedom. Two consequences:

*The free 8.6 knob measures nothing.* It reverses sign depending on whether silicate is in the
model, and is wildly fold-unstable either way:

| free-split fit | peak 8.6/7.7 | integrated | per-fold |
|---|---|---|---|
| no silicate | +0.20 ± 0.14 | 0.10 | −0.00 / 0.39 / 0.21 |
| with silicate | +1.84 ± 0.56 | 0.89 | 0.93 / 2.23 / 2.35 |

Both bracket the physical value (integrated 0.2–0.3) from opposite sides. Welding is therefore
not just defensible, it is **required** — it is the only thing preventing a runaway.

*The χ² preference for the ratio flips when silicate is added*, and τ_sil rises monotonically
with the asserted 8.6 (the anti-degeneracy, directly visible):

| assumed peak 8.6/7.7 | 0.00 | 0.25 | **0.50** | 0.75 | 1.00 | 1.33 |
|---|---|---|---|---|---|---|
| integrated | 0.00 | 0.12 | **0.243** | 0.36 | 0.49 | 0.65 |
| χ²_red, silicate ON | 5.21 | 4.84 | **4.65** | 4.54 | 4.49 | 4.48 |
| χ²_red, silicate OFF | — | 5.44 | **5.65** | — | 6.17 | 6.51 |
| fitted τ_sil | 0.18 | 0.35 | **0.49** | 0.62 | 0.72 | 0.84 |
| η_A | −0.086 | +0.013 | **+0.107** | +0.180 | +0.246 | +0.318 |

So the adopted `_R86_FIXED = 0.5` (integrated 0.243, mid-range of the Smith+2007 SINGS /
Draine & Li 0.2–0.3) is a **prior, not a measurement** — χ² alone would now pull it higher.
That is the correct design given the degeneracy, but it must be stated as a prior. The
residual η_A sensitivity is ~0.3 per unit peak ratio (halved from ~0.7 without silicate), so
over the physical range 0.25–0.75:

> **η_A = +0.11 ± 0.05 (fold) ± 0.08 (8.6 prior)** — consistent with zero, prior load-bearing.

The science is safe: across the same physical range the A_pah mass slope moves only
+0.339 → +0.362 (±3%), and the band-ratio *normalization* moves ±13% roughly common-mode
across mass bins, so the declining trend with M\* is unaffected.

*Why 8.6 belongs in the ionized denominator anyway:* 8.6 µm is the C–H **in-plane** bend,
cation-enhanced like the 6.2/7.7 C–C stretches; the neutral-side bands are the C–H
**out-of-plane** bends at 11.3/12.7 (Draine & Li 2001; Hudgins & Allamandola). So
(11.3+12.7)/(7.7+8.6) is a *cleaner* ionization contrast than a 7.7-alone denominator — the
denominator is the full ionized complex. The cost is that the ratio inherits a τ_sil
systematic, which the 7.7-alone version did not have.

See `figures/pah12_r86_tau_degeneracy.png` for the joint surface (untracked — `figures/`
and `notebooks/` are gitignored by repo convention; only `notebooks/build_*.py` scripts
are committed).

**4. α is only mildly affected.** Free Wien slope (weak prior N(2, 1.0), bounds [1,4]) gives
α = 2.43 with no silicate and 2.245 with plateau+silicate — a move toward 2 but not a
resolution. A_pah's α-sensitivity is unchanged (ratio A_pah(α=2.5)/A_pah(α=1.5) ≈ 2.2–2.5
with and without the plateau, since the plateau amplitude is zero). **The branch-11 decision
to fix α = 2.0 stands**, and the α systematic on absolute A_pah stands with it.

**5. Science invariance — mostly, with a caveat.** The A_pah(M\*) slope is positive and
significant in every configuration, and adding silicate makes the grouping choice stop
mattering (+0.356 ± 0.066 welded vs +0.395 ± 0.054 free split, agreeing within errors, where
before they differed +0.574 vs +0.451). But the magnitude is **not** invariant: welded moves
+0.574 → +0.356 when silicate is added. Sign, significance and the grouping-independence are
robust; the numeric slope should be quoted from the silicate-on fit. (Note this is the A_pah
= PAH/continuum normalization, not L_PAH/L_IR — see `pah-signflip-diagnosis`.)

**6. χ² is improved but not solved.** 5.65 → 4.65. The brief hoped plateau + silicate would
pull χ²_red toward 1; they do not. The remaining excess is consistent with the branch-5
finding that it is real galaxy-to-galaxy PAH scatter, not baseline error. The negative-plateau
fit reaching 3.95 says the trough is somewhat deeper or broader than a pure 9.7 µm Drude
screen — a shape refinement worth a future look, not a fit failure.

## Open / next

- Quantify the trough shape: does a second absorption component (18 µm silicate) or a
  wider/deeper 9.7 profile close the 4.65 → 3.95 gap the negative plateau reaches?
- τ_sil differs between groupings (0.55 vs 0.90) — it partly absorbs whatever the 8.6 knob
  would have. Worth a joint (τ, r_8.6) contour.
- ~~Re-run the L_PAH/L_IR crossing and band-ratio figures with silicate on.~~ **Done** — see
  the Notebook section: the crossing survives and strengthens (p 0.0039 → 0.0006).

## Notebook

`notebooks/2026-07-24-pah-money-plots-wider-z-bins.ipynb` — the 2026-07-23 money-plots
notebook rebuilt on this plumbing (welded template + silicate on, plateau off). All model
construction now goes through one `pah_model()` factory driven by cell 1, and the
neutral-band group index is **derived** from `FEATURE_GROUPS` (`IDX_NEUTRAL`) rather than
hard-coded `r[3]` — it moves between the welded and free-split templates, which was a live
foot-gun. Executes clean end to end.

Outcomes vs the 2026-07-23 free-split run:

| quantity | 2026-07-23 (free split) | 2026-07-24 (welded + silicate) |
|---|---|---|
| η_A (pooled) | +0.135 ± 0.058 | +0.108 ± 0.046 |
| χ²_red (pooled) | 5.40 | 4.65 |
| crossing (z~1 / z~2 / z~3) | +0.42 / +0.11 / −0.63 | +0.442 ± 0.035 / +0.027 ± 0.061 / −0.680 ± 0.117 |
| crossing F-test | F(4,6)=13.23, p=0.0039 | **F(4,6)=26.27, p=0.0006** |
| all-z L_PAH/L_IR slope | ≈ +0.09 | +0.115 ± 0.014 |
| free α_wien | ~2.4 | 2.18–2.35 |

The crossing is preserved and **more** significant. **Caveat when comparing section 1:** the
band-ratio denominator is the welded 7.7+8.6 complex again, so those ratios are not on the
same scale as the 2026-07-23 7.7-alone values (0.73/0.54/0.45/0.46, declining with mass,
5.0σ between the two lowest bins). Everything normalized per-bin is comparable.

## LIM notebook (2026-07-25)

`notebooks/2026-07-25-lim-via-pah.ipynb` — the LIM forecast notebook rebuilt on this
branch's model. Executes clean. Three substantive fixes, two of which were bugs:

**1. A normalization bug in the Chiang comparison.** `log_pah_ir_ext` — behind *every*
curve compared against Chiang+2026 — used the z-AVERAGED pivot while the surrounding
text described the z-resolved measured one. It suppressed the curve ~2× at z~1 and
boosted it ~1.6× at z~3. With the correct pivot:

| z | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|---|---|
| ours / Chiang | 2.13× | 1.70× | 1.22× | **1.16×** | 1.29× | 1.59× |

We sit 1.2–2.1× above Chiang, closest at cosmic noon — consistent within the bridge
(×0.56–1.44) and Chiang's own ±35%, but NOT the "1.1× at z~1, remarkably close" the
2026-07-23 version reported. The qualitative conclusion survives; the specific
agreement claim does not.

**2. The CO section was comparing different transitions.** It set
`MMIME_PSHOT = 2.0e3 μK²(Mpc/h)³` labelled "mmIME higher-J summed" — a value that
appears nowhere in Keating+2020 — and plotted it against a **CO(1-0)** model curve,
concluding "our signal is ~10³× below". mmIME does not measure CO(1-0) at all. Now
rebuilt with every point at its own transition and redshift: COPSS II (CO(1-0),
z=2.8, 3.0±1.3e3), COMAP Season 2 (CO(1-0) UL, kP(k)<2.4–4.9e3 μK² Mpc²), mmIME
per transition (CO(2-1)/(3-2)/(4-3) at z=1.3/2.5/3.6), ASPECS LP (5 ρ(H₂) bins),
COLDz, COMAP S2 ρ(H₂). Our model is carried **up** the excitation ladder
(r₃₁=0.84±0.26, VLASPECS) to meet mmIME on mmIME's transitions.

The corrected comparison is informative in a way the old one could not be: our CO
**mean intensity agrees** with ASPECS/COLDz/COMAP within ~2×, but our **shot power
under-predicts** (~5–9× vs mmIME, ~90× vs COPSS II). Since ⟨T⟩∝∫nL and
P_shot∝∫nL², matching one and missing the other means the CO luminosity function is
too flat at the bright end — the MS-only, logM*<11.5 construction excludes the rare
bright emitters that dominate ∫nL². That is a population-model property, not a PAH
result. (COPSS II is itself a 2σ detection above essentially every model and in
tension with COMAP's limits.)

**3. Bridge ratios are now derived in a dedicated §1b** with propagated uncertainty:
L_CII/total-PAH = 0.0048/0.10 = **0.048** (range 0.027–0.069) from Smith+2017
(KINGFISH, L_CII/L_TIR = 0.48±0.21%) and Smith+2007 (SINGS, L_PAH/L_TIR ≈ 10%). Both
check out independently — this is a derivation fixed before any comparison, not a
value tuned onto Chiang. (The earlier attribution of the 0.48% to "Herrera-Camus+2015"
is corrected to Smith+2017.) The CO bridge is carried **two ways** — "gas"
(L_IR/L'_CO, no PAH input, no crossing) and "PAH" (constant L_PAH/L'_CO) — because
the difference IS the assumption; they differ by only 0.63–0.84×.

Also of note: the amplitude-estimator systematic has largely **closed** on the
branch-12 model. The all-z and z-resolved pivots now differ by 0.87×, versus 2.4× on
the 2026-07-23 free-split template, so `PAH_NORM_MODE` is no longer the dominant
systematic — the bridge is.

Measured inputs are **recomputed, not transcribed**: slope and pivot now come out of
the same `zr_pool` matrix in one script, which is what let them drift apart before.
Branch-12 values: slope +0.442/+0.027/−0.680, pivot log(L_PAH/L_IR)@10.5 =
−0.918/−1.161/−1.321 at z = 0.95/1.90/2.95.
