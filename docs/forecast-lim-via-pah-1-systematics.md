# Forecast-LIM-via-PAH — systematics drill-down

**Correct amplitude values (self-consistent, KINGFISH normal star-forming galaxies):**

| quantity | value | source |
|---|---|---|
| L_PAH/L_IR | ≈ 10% (total PAH) | Smith+07 |
| 7.7 µm complex / total PAH | ≈ 49% | Smith+07 |
| L_CII/total-PAH | ≈ 0.05 (↔ L_CII/L(7.7) ≈ 0.1) | Herrera-Camus+15 / Smith+07 |
| L_CII/L_IR (local normal SF) | ≈ 0.5% | Herrera-Camus+15 (0.48±0.21%) |
| L_CII/L_IR (cosmic average) | ≈ 0.33% | Chiang+26 |

**The right comparison to Chiang.** Chiang measures the aggregate cosmic [CII] emissivity —
the full integral over the luminosity function (all masses), not mass-limited. So the
apples-to-apples curve is our **full-mass (completeness, logM*>8)** curve: it sits ~1.5× Chiang
at z~1–2 (the local-vs-cosmic offset, 0.5% vs 0.33%; within the band) and ~3× at z~3 (the extra
factor is the low-mass crossing extrapolation, C2). The logM*>9.9 stacking-reach curve is
mass-incomplete and only coincidentally lands on Chiang. The absolute amplitude carries a ~×2
systematic (partial→total PAH + local scatter); the robust result is the mass/z structure.

This document lists every systematic in the chain, why it matters, and how to pin it. B1/B2
(bridge value/definition) are settled by the matched values above; A1 (10% total PAH/TIR) is
confirmed (Smith+07); the live amplitude item is C2 (z~3 low-mass extrapolation).

**Scope note.** The **CO** prediction is *not* anomalous — it tracks Li+2016 (both are
L_IR/L'_CO–based), sits ~25× below mmIME (higher-J) and ~10³× below the COMAP limit. So CO is a
consistency check, not the claim. **The [CII] excess is the thing to defend.** Separately, the
*mass/z structure* (the crossing → shot-noise shift) is **differential** and survives every
amplitude systematic below — it needs no absolute calibration. Keep the two claims apart:
structure (robust) vs. absolute amplitude (the gamechanger, systematics-limited).

**The chain.** Each factor is a multiplicative systematic:

```
<I_CII>(z) = (c / 4π ν_rest H(z)) · Σ_i  n_i(M*,z) · [ L_PAH/L_IR ]_i · [ L_CII/L_PAH ] · [ L_IR ]_i
                                    └ abundance ┘   └ our measurement ┘ └ lit. bridge ┘ └ SFR→L_IR ┘
```

The [CII] excess = our `L_CII/L_IR ≈ 1%` vs the canonical `~0.3%` (De Looze) / Chiang's ~0.32%.
That 1% = `L_PAH/L_IR (10%)` × `L_CII/L_PAH (0.1)`. **Both factors are ±0.3 dex uncertain, so the
two together can produce the whole ~0.5 dex (3×) excess with no new physics.** Those two are the
prime suspects; everything else is second order.

---

## A · Our measured PAH input — `L_PAH/L_IR(M*,z)`

| # | Systematic | Why it matters / direction | Size | How to scrutinize |
|---|---|---|---|---|
| A1 | **partial→total PAH conversion** | We measure a 3-group *partial* template (6.2, 7.7+8.6, 12.7). `LOG_PAH_IR_0=-1.0` (10%) assumes total. If our "total" is really partial-scaled, we *over*-assign L_PAH. **Prime suspect.** | ±0.2–0.4 dex | Build the partial→total template (Smith+07 SINGS total-PAH/TIR fractions per band); cross-check the absolute L_PAH/L_IR against Smith+07 (~10–13%) and against Shivaei+17 L_7.7/L_IR. Branch-9 open item. |
| A2 | **absolute PAH amplitude (α) vs shape** | The mass *slope* is robust; the *normalization* (α at the pivot) is softer. Sets the amplitude, not the crossing. | ±0.2 dex | Tier A/B-only refits (Eddington bias), bootstrap-over-sources error, refresh stale `_pah_coeffs` (flagged in branch 9). |
| A3 | **L_IR: main-sequence vs measured** | Notebook uses Speagle+14 MS L_IR; real stacks *measure* L_IR. If the stacked population is off-MS (quenched tail, starbursts) the MS over/under-states L_IR → scales L_CII. | ±0.1–0.3 dex | Wire the **measured** L_IR(M*,z) from the stacks (next-pass); compare MS vs stacked L_IR bin by bin. |
| A4 | **branch-9 measurement systematics** | Tier-C Eddington bias, single-field fold-scatter errors (uncalibrated), `_pah_coeffs` staleness feed the L_PAH/L_IR central + errors. | ±0.1–0.2 dex | Carry the branch-9 robustness suite forward (Tier cuts, bootstrap, coeff refresh). |

## B · The PAH→line bridge ratios

| # | Systematic | Why it matters / direction | Size | How to scrutinize |
|---|---|---|---|---|
| B1 | **L_CII/L_PAH central value** ✅ RESOLVED | Was the single biggest lever. **Matched value = 0.05** (L_CII/total-PAH), not 0.1. Adopted in `CAL`. | ±0.2 dex | Done: L_CII/L_TIR (Herrera-Camus 0.48±0.21%) ÷ L_PAH/L_TIR (Smith+07 ~10%) = 0.048. Band from the 0.27–0.69% CII/TIR scatter. |
| B2 | **definition mismatch (which PAH)** ✅ RESOLVED | The 0.1 is L_CII/PAH-**subset** (their "PAH" ≈ 7.7 complex ≈ 49% of total PAH, Smith+07). Our total-PAH template needs the total-PAH bridge → 0.1×0.49 ≈ 0.05, consistent with B1. This was the ~2× that inflated the amplitude. | factor ~2 (fixed) | Done. Cross-checks: 0.1×(7.7 fraction) and CII/TIR÷PAH/TIR both give ~0.05. |
| B3 | **z-evolution of L_CII/L_PAH** | We assume the *local* ratio holds at z~1–4. If [CII]/PAH rises or falls with z (metallicity, ⟨U⟩, ISM density) the amplitude and its z-shape move. | unknown, ±0.2 dex? | High-z samples with both [CII] and PAH (ALPINE + JWST/MIRI overlap); the intensity-drift knob (sSFR/⟨U⟩ dependence, already in `CAL`); Shivaei+24 q_PAH(z). |
| B4 | **CO bridge: MS L_IR/L'_CO (=70), α_CO, SLED** | Sets L_CO. Our stacked galaxies may be off the Sargent+14 MS locus; α_CO and the CO SLED add factors. (CO currently *matches* Li+16, so this is lower priority unless CO becomes a claim.) | ±0.3 dex | Check L_IR/L'_CO for the stacked population; state α_CO; CO(1-0) vs higher-J SLED for the mmIME comparison. |

## C · Population model (parameterized → replace with the real stacks)

| # | Systematic | Why it matters / direction | Size | How to scrutinize |
|---|---|---|---|---|
| C1 | **SMF n(M*,z) normalization** | ⟨I⟩ ∝ ∫ n L. Our Davidzon+17-ish SMF recovers ~90% of the MD14 SFRD (reasonable) but is approximate; a normalization error shifts all curves equally (models + ours). | ±0.1 dex | Use the **catalog's own n(M*,z)** (measured abundance, not a modelled LF) — a next-pass item; cross-check against Davidzon+17/Weaver+23. |
| C2 | **low-mass extrapolation (completeness)** | The completeness (>8) curve holds the crossing flat below 9.9; at z~3 it is extrapolation-sensitive. Affects only the completeness curve, not the >9.9 direct-reach curve. | factor ~1.5 at z~3 | Impose the physical low-Z PAH turnover (Shivaei q_PAH(Z) step) rather than flat; low-mass-limit sensitivity band (already shown). |
| C3 | **MS SFR (Speagle) + SFR–L_IR (Kennicutt/Chabrier)** | Sets L_IR and the SFRD normalization of *all* curves (ours and the models), so it partly cancels in ratios but not in absolute ⟨I⟩. | ±0.1–0.2 dex | MS scatter + starburst fraction; IMF choice (affects models and ours equally). |

## D · Intensity / power-spectrum machinery

| # | Systematic | Why it matters / direction | Size | How to scrutinize |
|---|---|---|---|---|
| D1 | **comoving intensity formula** | Low risk — validated. Cross-checked: our ρ_L→⟨I⟩ conversion reproduces Chiang's ρ_CII→⟨I⟩ consistently. | <0.05 dex | Already cross-checked vs Chiang; keep the unit test. |
| D2 | **P_lin (BBKS), growth, Tinker bias, SHMR** | Affects the *clustering* term (∝ b_eff²), NOT the shot noise. The crossing result is a shot-noise statement, so it is nearly immune. | b_eff ±10–20% | Swap BBKS→CAMB/colossus P(k); Behroozi/Moster SHMR for logMhalo; but low priority for the shot-noise headline. |

## E · Are the comparison targets themselves right?

| # | Systematic | Why it matters / direction | Size | How to scrutinize |
|---|---|---|---|---|
| E1 | **Chiang+2026 measurement** | Our benchmark. 3σ [CII] detection via tomographic clustering of broadband intensities × reference galaxies. If Chiang is *low* (residual CIB/continuum, calibration), our excess shrinks; if robust, the excess is real. | benchmark ±0.2 dex | Read Chiang's systematics section: their own L_CII/SFR (2.2e7), their statement that earlier "detections" are CIB-contaminated, and that they are "broadly consistent with recent models." |
| E2 | **the model relations (De Looze, Lagache, Li16)** | The models we overshoot have their own scatter/validity range at our M*, z. De Looze is whole-sample; Lagache is a SAM; both ~0.3 dex scatter. | ±0.3 dex | Add more models (below); use the intrinsic scatter as the model band, not a single line. |
| E3 | **models to ADD (per user)** | Broaden the comparison so the overshoot is judged against the full model spread, not two curves. | — | **Chung, Viero, Church & Wechsler 2020** (ApJ 892, 51; arXiv:1812.08135) — [CII] LIM forecast, primarily z~4–8 (UniverseMachine + SFR–[CII]); add its L_CII prescription + z~4 point. **COMAP ES-V fiducial CO model** (Chung+2022, ApJ 933, 186; UM+COLDz+COPSS) — add the model curve, not just the upper limit. Both need digitizing from the papers (do not fabricate). |

## F · Sample / field

| # | Systematic | Why it matters / direction | Size | How to scrutinize |
|---|---|---|---|---|
| F1 | **single field (COSMOS), cosmic variance** | Normalization + error bars; a ~2 deg² field has non-negligible cosmic variance in the bright end (shot noise especially). | ±0.1–0.3 dex | Second field; bootstrap-over-sources; the branch-9 fold-scatter caveat carries here. |

---

## Verdict priorities (what to do first)

1. ✅ **B1/B2 (bridge ratio value & definition) — DONE.** The matched L_CII/total-PAH ≈ 0.05
   (not 0.1) removes the ~2× and lands our curve on Chiang. A1 (10% total PAH/TIR) confirmed from
   Smith+07. **The excess is resolved as a definition artifact.** Remaining amplitude uncertainty
   ~×2 (partial→total PAH exact fraction + local CII/TIR scatter).
2. **C2 (z~3 low-mass crossing extrapolation)** — now the live item: the completeness (>8) curve
   still overshoots at z~3. Impose the physical low-Z PAH turnover (Shivaei q_PAH(Z)) instead of
   holding the crossing flat below 9.9.
3. **B3 (bridge z-evolution)** and **A3 (measured vs MS L_IR)** — needed to trust
   the z-*shape* even now that the normalization is fixed.
4. **E1 (Chiang systematics) + E3 (add Chung+2020, COMAP fiducial)** — confront the benchmark and
   broaden the model spread.
5. **C1 (measured n(M*,z))** and **F1 (cosmic variance)** — normalization + honest errors.
6. **D2 (P(k) machinery)** — lowest priority; matters only for the clustering term, and the
   headline (crossing) is a shot-noise result.

**Bottom line for the talk.** Present the crossing / shot-noise *structure* as the robust result
(immune to A–F). Present the absolute [CII] amplitude as **consistent with Chiang** once the
bridge is matched (B1/B2) — no longer an excess — with a residual ~×2 systematic (partial→total
PAH, local scatter) and the z~3 completeness overshoot flagged as C2. That is the honest,
defensible framing: scrutiny turned an apparent gamechanger into a *resolved definition
artifact* — the extraordinary claim evaporated, and the real result (the structure) survives.
