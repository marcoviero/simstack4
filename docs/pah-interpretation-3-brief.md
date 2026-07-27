# pah-interpretation-3 — brief

**Opened 2026-07-27, from the end of pah-interpretation-2.**

## The question

Is the crossing an **abundance** effect (q_PAH) or a **radiation-geometry** effect (G₀)?

    L_PAH/L_IR  ∝  q_PAH × (G₀ / ⟨U⟩)

Every mechanism we have tested — metallicity supply (D1, D5), gas tracing (D2),
shattering suppression, Σ_SFR-triggered destruction — is a statement about **q_PAH**.
All of them fail. But we measure **L_PAH/L_IR**, and Narayanan+26 state explicitly that
the two *"do not evolve in lockstep"*: L_PAH/M_PAH ∝ G₀ while q_PAH anti-correlates with
Σ_SFR. If the G₀/⟨U⟩ term carries the mass dependence, we have been fitting the wrong
quantity for three branches.

## Why it could work where everything else failed

D6's argument is that mean scaling relations have near-constant *exponents*, so nothing
built from the main sequence and sizes can flip a mass slope with redshift. That is what
kills every equilibrium arm and, as of 2026-07-26, the Σ_SFR threshold model too
(it recovers z~3 slope −0.096 against a measured −0.699; see the threshold-fit section).

**G₀/⟨U⟩ is not a scaling relation.** It is a geometry factor — the concentration of
young stars relative to the bulk dust. Compact high-z starbursts vs extended low-z disks
is exactly the kind of structural change that can alter its *mass dependence* with
redshift. It is the only candidate identified so far that is not excluded by D6.

## Evidence in hand

1. **§3h** (branch-12 notebook): at z<2 the neutral-band luminosity per L_IR is
   mass-invariant (α·r_neutral slope = −0.005 ± 0.058) while the whole positive arm is
   carried by the ionised 7.7+8.6 complex. Ionisation state is set by the radiation
   field — a G₀ fingerprint, on the crossing's own catalog.
2. **Direct decomposition** (2026-07-27): pooled gives ionised +0.17…+0.30 vs neutral
   −0.06…+0.04 across all windows/groupings. **But the combined stack inverts** once the
   (leverage-free) 6.2 group is dropped. Not yet a result.
3. **Direct Σ_SFR test**: L_PAH/L_IR *rises* with Σ_SFR — consistent with q_PAH falling
   (Narayanan) *only if* G₀ dominates. Independent support for the decoupling being real
   and large.

## Step 0-bis — the cheap route that BYPASSES the gate (added 2026-07-27)

Task 1 below needs the ionised/neutral decomposition, which is exactly what is unstable.
There is a cheaper route to the same question that does not touch it:

    q_PAH  ∝  (L_PAH/L_IR) × ⟨U⟩ / G₀ ,   ⟨U⟩ ∝ T_dust^(4+β) ,   G₀ ∝ sSFR^a

`T_dust` comes free from every greybody fit and `lp_sSFR_med` is in the COSMOS2020
catalog, so this runs on data already in hand. (Σ_SFR is *not* available for the
crossing data — COSMOS2020 has no sizes — so sSFR is the only usable G₀ proxy there.)

**The two halves are not equally sound.** ⟨U⟩ ∝ T_dust^(4+β) is radiative equilibrium,
a physical relation. G₀ ∝ sSFR^a is a proxy whose exponent is unknown and which bakes
in fixed geometry — the very thing under test. So this can show the radiation term is
*sufficient*; a null would not exclude a geometry term sSFR fails to capture.

### First-pass result (`notebooks/build_qpah_backout.py`)

| a | z~1 | z~2 | z~3 | swing |
|---|---|---|---|---|
| **observed** | **+0.380** | −0.009 | **−0.699** | **−1.079** |
| 0.00 (T_dust only) | **−0.010** | −0.241 | −0.654 | **−0.644** |
| 0.50 | +0.496 | +0.359 | −0.132 | −0.628 |
| 1.00 | +1.003 | +0.959 | +0.390 | −0.613 |

1. **The T_dust correction alone removes the z~1 positive arm** (+0.380 → −0.010) and
   **40% of the swing**. Massive galaxies at z~1 are colder (30.3 → 26.0 K across the
   mass range), so at fixed L_IR they carry more dust mass. The "unexplained piece" that
   D1 found 5σ above the metallicity ceiling may substantially be a **temperature
   gradient**.
2. **sSFR cannot change the swing — the D6 failure again.** Its mass gradient is nearly
   z-independent (−1.00 at z~1, −1.05 at z~3), so dividing by sSFR^a shifts all three
   slopes almost equally: the swing moves only −0.644 → −0.582 as a goes 0 → 2. A proxy
   whose mass gradient does not evolve cannot create a z-dependent slope change. **Drop
   the sSFR term.**
3. **A residual crossing survives** the full ⟨U⟩ correction: swing −0.64 vs −1.08.

### Before any of this is quoted

- **Propagate T_dust errors through T^(4+β) = T^5.8.** A 1 K error at 30 K is 3.3% in T
  but **19% in ⟨U⟩**. Use the fold ensemble, not the formal errors — a correlated T bias
  across a mass bin would masquerade as a slope.
- **T_dust is not independent of L_IR.** L_IR is computed from (amplitude, T, β), so
  multiplying by T^5.8 partly undoes that calculation. What is actually recovered is
  **L_PAH/M_dust**, which is the cleaner quantity to quote — and it needs no G₀ proxy at
  all, given the sSFR term does nothing.
- **β is fixed at 1.8** and 4+β *is* the lever. If β varies with mass, so does the
  exponent.

**If the z~1 positive arm does not survive that error propagation, D1's headline and the
"mass-correlated driver above half-solar Z" that has been the unexplained piece for three
branches were partly a dust-temperature gradient — and the letter's framing changes
materially.** Do this before task 1.

## Tasks, in order

1. **Stabilise the ionised/neutral decomposition.** Drop 6.2 (no leverage at z<2, it
   rails to −11…−60 and contaminates its neighbours), match the combined stack's
   z-sampling to pooled's, and establish whether the two estimators can be made to
   agree. Until they do, nothing else here is safe. *This is the gate.*
2. **If it holds: measure d log(G₀/⟨U⟩)/d log M\* and its z-evolution.** The ionised/
   neutral ratio is the G₀ proxy; the question is whether its mass slope changes sign or
   magnitude between the z-slices the way the crossing does.
3. **Revisit the §3g bands.** They carry only **±0.10 dex/dex** for the G₀ correction,
   on the untested assumption that radiation-field factors "cancel to first order in the
   ratio". If G₀/⟨U⟩ varies by several tenths of a dex per dex of mass, the whole
   Narayanan confrontation must be rebuilt on q_PAH rather than L_PAH/L_IR — and the
   8.6σ swing result would need restating.
4. **Convert L_PAH/L_IR → q_PAH** where possible, so the comparison is like-for-like
   with simulations. Needs a G₀ estimator per bin (ionised/neutral ratio, or T_dust).

## What NOT to redo

- Σ_SFR or sSFR triggers — both excluded (mass gradients grow 1.11× and 0.87× against a
  required 2.8×).
- D4 mediator separation — inconclusive and structurally so; z-slicing the amplitude is
  self-defeating for a tomographic measurement, and dithering does not fix it.
- Finer binning or more dither offsets on COSMOSWeb — both tried, neither helps.

## Health warnings

- The band ratio is a **z<2 measurement only** (§1b): above z=2 the neutral group holds
  2% of its peak leverage. So the G₀ test can be run on the *positive* arm and not the
  negative one — the same limitation that made D3 inconclusive.
- Tier grades band SNR, not fit convergence. Always apply explicit physicality + error
  cuts on T_dust (15 < T < 45 K, err < 15 K).
- α stays FIXED at 2.0. Free-α fits move the full-range mass slope by up to 0.9 dex/dex
  under components fitted to zero.
