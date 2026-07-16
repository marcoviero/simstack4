# Forecast LIM via PAH — Branch 1 Brief: what our PAH measurements say about [CII] and CO

**Audience.** A line-intensity-mapping (LIM) conference. The room measures the aggregate,
redshift-binned surface brightness of [CII] 158 µm (CONCERTO, TIME, EXCLAIM, CCAT-prime/FYST)
and the CO ladder (COMAP, mmIME) from galaxies that are individually undetected. Their
forecasts of the mean intensity ⟨I_line(z)⟩ and its power spectrum rest on **models** that
paint line luminosity onto halos/galaxies as a function of mass and redshift — L_CII–SFR
relations plus an SFR function or SFR–M_halo abundance matching for [CII]; L′_CO–L_IR (or
L_CO–SFR) plus a CO SLED for CO. Those recipes are uncertain by factors of a few, worst
exactly where LIM lives: the faint/low-mass end that dominates the clustering signal, and the
z-evolution of the line-to-SFR ratio.

**The pitch.** We have just measured, by dithered simultaneous stacking, the PAH luminosity of
the confused faint population as a function of (M*, z) — including a non-trivial evolving mass
dependence (the L_PAH/L_IR "crossing," branch 9). PAHs are physically wired to *both* target
lines. So our result is an **empirical, evolution-measured anchor** the LIM community can drop
into the part of their forecast that is currently pure model. This brief is the plan to turn
the branch-7/8/9 PAH results into a PAH-anchored ⟨I_CII(z)⟩ / ⟨I_CO(z)⟩ forecast and confront
it with the recipes the room already uses.

**Framing constraints carried forward** (branch 7–9): NO AGN-based interpretation; quote
free-α results as primary; pooled fits are centrals, fold scatter is the error; combined-stack
bin0 numbers over K-fold-pooled where the pooled fit has the known A(6.2) SNR defect; the
nested-model p=0.005 is directional, not calibrated. This is a **talk**, not a second letter —
the deliverable is figures + a defensible narrative, not a precision forecast.

---

## Why a PAH stacking result is a LIM result (the bridge)

Two independent physical couplings, each with a local empirical calibration:

1. **PAH → [CII] through the PDR.** [CII] is the dominant coolant of warm neutral / PDR gas;
   PAHs are the dominant **photoelectric heating agent** of that same gas (grain
   photoelectrons do the heating that [CII] radiates away). Because both scale with the PDR,
   L_CII/L_PAH is observed to be *roughly* constant across resolved regions and galaxies
   (Helou+01; Croxall+12; Smith+17 KINGFISH; Sutter+19) — that near-constant ratio is what
   lets L_PAH(M*, z) stand in for L_CII(M*, z). Mind the direction of the argument, because
   it interacts with our crossing:
   - The classic high-L_IR "[CII] deficit" and the PAH deficit are *mirror* trends **on the
     L_IR / Σ_SFR / ⟨U⟩ (radiation-intensity) axis** — both fall in compact, intense systems,
     so their *ratio* stays put. That is bridge **support**, and it is a *different axis* from
     our L_PAH/L_IR-vs-M* crossing. No direct contradiction.
   - The crossing does not conflict with the mirrored deficits — under constant L_CII/L_PAH it
     *predicts* that **L_CII/L_IR crosses with mass the same way**. That non-monotonic,
     evolving line/IR ratio is precisely the behaviour standard (monotonic-with-luminosity)
     deficit forecasts miss, and it is what we hand to LIM. Where the two *appear* to fight —
     the z~1 positive slope (PAH/L_IR rising toward massive, higher-L_IR galaxies, opposite the
     canonical deficit's decline with L_IR) — that is the deficit picture being incomplete, i.e.
     the result; the z~3 slope goes negative and recovers the canonical direction.
   - **The one real caveat** (was hiding under "mirrored"): L_CII/L_PAH is not exactly constant
     — it deepens further in the most extreme systems, driven by the *same* intensity variable
     as both deficits. So the constant-ratio assumption is weakest exactly where the crossing
     bites; carry the ratio's second-order drift as a systematic (Obj 1), do not assume it away.

2. **PAH → CO through the molecular gas.** L_PAH correlates tightly with L′_CO out to z~4
   (Cortzen+19; arXiv:2409.05710) — PAH luminosity is effectively a molecular-gas-mass tracer.
   So L_PAH(M*, z) → M_H2 → L_CO given a CO conversion + SLED.

3. **Stacking reaches exactly LIM's population.** The line–SFR calibrations the community uses
   are anchored on *bright, individually detected* galaxies; the LIM signal (especially the
   clustering term) is dominated by *faint/low-mass* galaxies. Simultaneous stacking measures
   the mean over the unresolved population — the same population, the same confusion regime.
   This methodological match is a large part of the pitch.

4. **We are replacing a *modeled* evolution with a *measured* one.** We do not remove model
   dependence — the PAH→line ratios are still local calibrations assumed to hold at high z.
   But the LIM forecast's most uncertain ingredient is the *z-evolution of the line/SFR ratio*;
   we substitute a measured PAH evolution (branch 9's crossing + η_A amplitude evolution) plus
   a local ratio for a fully modeled one. State this honestly — it is the correct, defensible
   framing and it is still a real advance.

**What the branch-9 results specifically buy:**
- The **crossing pattern** (L_PAH/L_IR mass slope +0.3 at z~1 → 0 at z~2 → −0.6/−0.7 at z~3)
  says the PAH-anchored line predictions have a **non-monotonic, evolving mass dependence** that
  fixed-slope L_line–SFR models cannot produce. That reshapes the effective **line bias**
  (halo-mass weighting) and the clustering-vs-shot-noise split — the headline "this is what a
  standard forecast misses."
- The **24 µm SFR bias** (the entire program): LIM forecasts that derive SFR from IR/24 µm
  inherit our mass/z-dependent PAH contamination. We can hand the room a correction.

---

## Objective 0 — audience framing (one figure, do first)

- [ ] The bridge figure: a schematic PDR / molecular-cloud cartoon showing PAH photoelectric
      heating → [CII] cooling and PAH ↔ CO co-spatiality, annotated with the local L_CII/L_PAH
      and L_PAH/L′_CO calibration scatter. This is the "why should I care" slide; everything
      else hangs off it.
- [ ] One-sentence thesis to test against every result below: *the redshift evolution of the
      PAH-to-IR ratio that we measure directly is the same evolution LIM forecasts currently
      have to assume for the line-to-SFR ratio.*

## Objective 1 — the PAH → line calibration ladder

Assemble the conversions with their scatter and, critically, decide the anchor observable.

- [ ] **Anchor choice.** Total L_PAH, or a single feature (7.7 or 11.3 µm) that best correlates
      with [CII]/CO? The literature correlations are often feature-specific; our tomography
      constrains specific features at specific z. Pick the anchor that (a) has the tightest
      local line calibration and (b) is best constrained by our stacking at the relevant z.
- [ ] **Partial → total PAH conversion.** Our 3-group template is a *partial* PAH measurement;
      the line calibrations are usually against total PAH. Build the partial→total conversion
      (this is the same open item as branch-9 Objective 2 / Smith+07 absolute-scale check —
      share the code, don't duplicate).
- [ ] **L_PAH → L_CII.** Adopt L_CII/L_PAH from Croxall+12 / Smith+17 / Sutter+19 with its
      scatter. Critically, model the ratio's **drift with intensity** (L_IR / Σ_SFR / ⟨U⟩):
      L_CII/L_PAH deepens in the most extreme systems, and because that intensity axis is
      correlated with our M*/z crossing, treating the ratio as strictly constant is the bridge's
      weakest assumption exactly where the crossing lives. Fold it in as an intensity-dependent
      L_CII/L_PAH(⟨U⟩) (T_dust is free from every greybody fit → a usable ⟨U⟩ proxy), and report
      the forecast's sensitivity to switching it on/off.
- [ ] **L_PAH → L′_CO.** Adopt the Cortzen+19 / 2409.05710 relation; propagate to L_CO with a
      stated α_CO and a J-ladder assumption (COMAP is CO(1–0)-ish at its z; be explicit about
      which transition each experiment sees at which z).
- [ ] **Ratio evolution as an explicit knob.** Default = local ratio, z-independent. Provide a
      parameterized ratio-evolution term so the forecast's sensitivity to *that* assumption is
      visible (it is the residual model dependence we are honest about in the bridge section).

## Objective 2 — build the PAH-anchored forecast

- [ ] **Use the catalog's own n(M*, z), not an assumed LF.** ⟨I_line(z)⟩ = (c / 4π ν H(z)) ×
      Σ_i n_i(M*, z) L_line,i — we have the per-(M*, z)-cell source counts directly from the
      stacking catalog, so the abundance is measured, not modeled. Reuse `CosmologyCalculator`
      (Planck18) for the volume/luminosity-distance factors.
- [ ] **Mean intensity curves** ⟨I_CII(z)⟩ and ⟨I_CO(z)⟩ from L_PAH(M*, z) × the ladder above.
- [ ] **Shot-noise power** P_shot(z) = (c/4π ν H)² Σ_i n_i L_i² — dominated by the bright end,
      a clean second observable, and a good cross-check that our mass binning captures the L²
      weighting sensibly.
- [ ] Propagate the fold-scatter errors + the calibration scatter into a forecast error band.
      Label it illustrative (single field, cosmic variance, uncalibrated fold errors — same
      caveat the branch-9 letter carries).

## Objective 3 — confront real measurements (the payoff)

- [ ] **[CII] vs the Chiang+2026 measurement.** Chiang measures the cosmic [CII] luminosity
      density to z=4.2 by tomographic clustering with reference galaxies (our method's cousin) —
      a real anchor, not a model. Compare our ⟨I_CII(z)⟩ to it, plus the De Looze × SFRD
      reference. Note: the absolute amplitude is systematics-limited (bridge ratio + partial→total
      PAH, factor ~3), so it is not the result — see the next bullet.
- [ ] **Isolate the crossing pattern's effect — on the power spectrum, not the mean.** ⟨I(z)⟩ is
      luminosity-weighted with an effective mass at the crossing pivot, so the crossing barely
      moves it; the leverage is on the **shot-noise power** P_shot ∝ Σ n L² (high-mass weighted),
      which the crossing lifts tens of percent while the clustering term is ~fixed. That is the
      headline: *what a fixed-slope P(k) forecast gets wrong*. Compare the CO(1-0) P(k) to the
      mmIME detection and COMAP upper limit to place it against current sensitivity.
- [ ] **Effective line bias.** The non-monotonic mass weighting changes which halos dominate;
      quote the shift in b_eff / the luminosity-weighted mean mass vs a fixed-slope recipe.

## Objective 4 — the 24 µm bias deliverable for LIM forecasters

- [ ] Quantify the bias in a 24 µm→SFR→L_line forecast that ignores PAH contamination, as a
      function of (M*, z), using `greybody.py`'s PAH-correction path (note: `_pah_coeffs` are
      flagged stale in branch 9 — refresh before quoting numbers).
- [ ] Provide the correction as a simple, citable multiplicative factor table the room can
      apply to their own 24 µm-anchored forecasts. This is the most directly *useful* takeaway
      even for people who don't buy the crossing interpretation.

## Deliverable

One forecast notebook (tracked build script per repo convention) producing: the bridge figure
(Obj 0), the PAH-anchored ⟨I_CII(z)⟩ / ⟨I_CO(z)⟩ + power-spectrum curves (Obj 2), the
comparison to the Chiang measurement and the CO P(k) vs mmIME/COMAP with the crossing-pattern
isolation (Obj 3), and the 24 µm-bias correction table (Obj 4). Plus a slide-ready **"three
takeaways for LIM forecasters"**: (1) the mean-intensity forecast is robust to the crossing,
but the **shot-noise / power-spectrum** is not — it shifts tens of percent (below current
sensitivity, a next-gen target); (2) our absolute amplitude is systematics-limited by the
PAH→line bridge, so the robust deliverable is the mass/z *structure*, not the amplitude;
(3) here is the 24 µm PAH correction.

---

## Risks & scope control (state these; do not let them balloon the talk)

- **Local calibrations at high z** are the central assumption. Mitigation: make ratio evolution
  an explicit knob (Obj 1) and lead with "we replace modeled evolution with measured PAH
  evolution + a local ratio," not "we eliminate model dependence."
- **Partial → total PAH** and the **absolute L_PAH scale** are unfinished from branch 9. The
  forecast's *shape* in (M*, z) is more robust than its absolute normalization — lean on the
  shape (crossing, bias) for the headline; caveat the absolute ⟨I⟩ normalization.
- **Single field / cosmic variance / uncalibrated fold errors** — inherited from the letter.
  Present the forecast as illustrative and methodological, not a precision prediction.
- **CO transition/SLED and α_CO** add their own factor-of-few uncertainty on the CO side —
  [CII] is the cleaner PAH bridge (shared PDR physics); consider making [CII] the primary
  result and CO the secondary, gas-tracer-framed one.
- **Do not re-derive the PAH measurements here.** This branch consumes branch-9 outputs; if a
  branch-9 number is still open (η_A pooled-vs-combined, `_pah_coeffs` staleness), forecast
  with the combined-stack value and flag the dependency, don't re-litigate it.

## Targeted reading list (pull the numbers; do not quote from memory)

| Paper | Why |
|---|---|
| Croxall+12; Smith+17 (KINGFISH); Sutter+19 | L_CII/L_PAH ratio + scatter — the [CII] anchor calibration. |
| Helou+01 | Foundational PAH–[CII] photoelectric-heating link — the bridge citation. |
| Cortzen+19; arXiv:2409.05710 | L_PAH–L′_CO to z~4 — the CO anchor; already in the branch-9 gas-tracer framing. |
| De Looze+14; Lagache+18; Schaerer+20 | Standard L_CII–SFR forecasts to overlay in Obj 3. |
| arXiv:2506.13863 (PAH intensity mapping) | Methodological kin — PAH tomographic stacking as a LIM-adjacent technique; already noted in interpretation-candidates. |
| Experiment refs: CONCERTO, TIME, EXCLAIM, CCAT-prime/FYST ([CII]); COMAP, mmIME (CO) | Match each forecast to the transition/z each instrument actually probes. |

New references belong in `docs/pah-refs.md` under a new "LIM / line-forecasting" section
(currently absent — this branch adds it).

---

## Progress log (through 2026-07-15)

Notebook `notebooks/build_lim_via_pah_2026-07-12_notebook.py` → `2026-07-12-lim-via-pah.ipynb`
(27 cells, runs clean). Sections §0 bridge, §1 ladder, §2 build, §3 [CII] I+P(k), §3b CO I+P(k),
§3c crossing shot-noise, §4 24µm bias, §5 takeaways, §6 amplitude & the right Chiang comparison,
§7 real catalog n(M*,z), §7b low-mass PAH turnover. CII-focused; CO present but secondary.

**Method settled.**
- Absolute [CII]: L_CII = (L_PAH/L_IR ≈ 10%, Smith+07 total PAH) × (L_CII/**total**-PAH ≈ 0.05,
  Herrera-Camus+15 / Smith+07 — the 7.7 complex is ~49% of total, so the Croxall/Sutter
  L_CII/L(7.7) ≈ 0.1 ↔ 0.05 on total) × L_IR. → L_CII/L_IR ≈ 0.5% local vs Chiang cosmic 0.33%.
- Comparison curves are real data or computed from cited relations only (NO fabrications):
  De Looze+14 & Lagache+18 ([CII] models), Li+2016 (CO); **Chiang+2026** (measured cosmic [CII]/CO,
  arXiv:2602.02658, the key anchor); mmIME + COMAP ES-V (CO P(k) data).
- Intensity is a surface brightness — no survey area needed for the parameterized SMF; area
  enters only when using real catalog counts (§7).

**Key results.**
- Robust, tuning-free result = the mass/z **structure** (the crossing): +34–48% shot-noise
  shift at z~0.5–1 (§3c), and a −20%→+90% (z~1→3) shift of the full-integral mean ⟨I_CII⟩ vs a
  canonical no-structure PAH model (§3a). Below current LIM sensitivity → next-gen target.
- Absolute amplitude is consistent with Chiang: full-integral vs full-integral ~1.5× (the
  local-vs-cosmic offset), systematics-limited (factor ~2). The apparent 2–3× "excess" was a
  bridge-ratio definition mismatch (0.1 = L_CII/7.7-subset; matched total-PAH = 0.05). Full
  systematics audit in `docs/forecast-lim-via-pah-1-systematics.md`.
- Real COSMOS2020 catalog n(M*,z) (§7): star-forming counts from the stack JSON `n_sources`
  (split_0), Ω = 1.27 deg², volume via `CosmologyCalculator`; validates the SMF forecast (SFRD
  and ⟨I_CII⟩ agree). Measured to logM=9.
- Low-mass PAH turnover (§7b): the 9.0–9.9 stack shows no 24µm PAH excess (S/N-marginal),
  ruling out the low-mass boost; applying the turnover removes the z~3 overshoot (2.74→1.55×
  Chiang). Caveat: L_CII from PAH breaks at low Z ([CII] rises while PAH falls).

**Open / next.** (a) SFR-based low-mass [CII] cross-check (PAH→[CII] breaks at low Z);
(b) digitize Chung+2020 [CII] & COMAP-fiducial CO model curves (halo-LF-based; do not fabricate);
(c) wire measured L_IR (vs MS) from the stacks; (d) CO version of the catalog-n / turnover work;
(e) tighten partial→total PAH + bridge scatter. Full running detail in memory
`forecast-lim-via-pah-1-status`.
