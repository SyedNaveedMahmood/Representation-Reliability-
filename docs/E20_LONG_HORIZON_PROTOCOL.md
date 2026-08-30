# E20 Long-Horizon Extension of the Temporal Causal-Organization Profile

Status: **frozen before any E20 measurement code exists and before any E20
quantity is computed.** Committed and pushed before execution.

Open discovery. No confirmation split is created or accessed; no consumed holdout
(E01, E13, E14) is touched. No confirmatory claim is made.

## 1. Why this extension exists

E19 established the target signature on a sufficient carrier: decodability stayed
at ceiling across the horizon while `Q`, `A` and `G` fell to 0.71, 0.73 and 0.55
of baseline by `k=8`. But **every half-life was right-censored** — no component
halved inside the grid, so no persistence timescale could be quoted.

E20 changes exactly one thing: **how far the horizon reaches**. The carrier, the
two-locus design, the components, the estimands, the controls, the inference and
the claim boundary are all inherited unchanged from
`docs/E19_TEMPORAL_CAUSAL_ORGANIZATION_PROTOCOL.md`. This is a range extension,
not a redesign, and it introduces no new hypothesis.

It also promotes one control. E19 discovered *after* its run that one of its four
curves had a shrinking intervention magnitude, which made its apparent decay a
magnitude artifact. That control was post-hoc there. **Here it is a preregistered
validity gate that excludes a failing curve from every hypothesis test**, not
merely from the outcome label.

## 2. What is inherited unchanged

```text
loci        S_source    state_word_last @ resid_post L8   (age AND distance grow)
            D_decision  decision        @ resid_post L24  (age grows, distance ~0)
components  Q = Y10-Y00,  A = Y01-Y00,  G = (Y11-Y10)-(Y01-Y00)
estimands   native (horizon-local axis and targets) and ref (frozen k0 axis)
arms        no_op, full_state_patch, Y10/Y01/Y11 x {native, ref},
            random_norm_matched x3, orthogonal_random x3
inference   episode-cluster bootstrap resampling whole horizon curves together,
            2000 draws, 95 percent, Holm across the Q/A/G family
half-life   metrics.temporal_half_life.half_life only, which refuses a summary
            for a non-monotone curve and returns right-censored rather than
            extrapolating
model       Qwen/Qwen3-1.7B, bf16
```

The loci come from E18's map and are **not** searched over again. No layer, site,
component or estimand is added, removed or re-selected.

## 3. The horizon problem and how it is resolved

E15 measured forced-choice behaviour collapsing to 0.560 at `k=16` and 0.500 at
`k=32` with short distractor steps. A decision the model cannot make is not a
decision whose causal organization can be read, so E19 stopped at `k=8`.

E15's Stage 3b hinted that **step count, not token distance, drives that
collapse**: 6 long steps kept `B` at 0.837 while 16 short steps dropped it to
0.560 at a comparable token distance. That contrast was itself confounded (its
token distances were 162 versus 250 and it is recorded as a failed control), so
it is treated here as a hypothesis to test, not a fact to rely on.

E20 therefore selects its horizon range in a **non-causal Phase 1**, exactly as
E17 screened candidate models on engineering, behaviour and decodability only.

### Phase 1 — non-causal range selection

Candidate horizon grid, frozen:

```text
k in {1, 2, 4, 8, 16, 24, 32}
```

Candidate distractor pools, frozen: `SHORT_DISTRACTORS` (as used by E15, E18 and
E19) and `LONG_DISTRACTORS` (as used by E15 Stage 3b). Both already exist in the
frozen generator; neither is new content.

For each pool, on `discovery_test` only, compute **clean forwards only**:

```text
B(k)   forced-choice accuracy
D(k)   decodability at each locus
mean token distance from the state write to the decision
```

**Forbidden during Phase 1:** any intervention, any `Q`, `A`, `G`, any
full-state patch, any edited forward. This is what prevents the range being
chosen because it happened to produce a nicer decay curve.

Frozen selection rule, applied to non-causal quantities only:

1. a horizon is *admissible* for a pool if `B(k) >= 0.70`;
2. a pool's *reach* is the longest prefix of the candidate grid that is
   admissible;
3. the pool with the greater reach is selected; ties go to `SHORT_DISTRACTORS`,
   which preserves continuity with E15, E18 and E19;
4. the selected grid is that pool's reach.

Both pools' Phase 1 tables are reported whether selected or not. If neither pool
reaches beyond `k=8`, that is a real result: E20 reports that the environment
cannot support a longer horizon at interpretable behaviour, runs nothing further,
and the half-life stays censored.

### Phase 2 — the E19 sweep on the selected grid

The full E19 arm set, unchanged, on the selected pool and grid.

## 4. Gates

Inherited from E19:

* **G0 corpus** — label oracle exact; twins differ in exactly one clearance word;
  token parity; spans unique; pairs never split; no duplicate prompts.
* **G1 numerics** — `no_op` deviation `<= 1e-6`; zero residual hooks after every
  batch; norm-matched controls within `1e-6` relative; setpoint fidelity inside
  the frozen bf16 tolerances; exact row identity.
* **G2 behaviour** — `B(k) >= 0.70`, applied in Phase 1 before any intervention.
* **G3 carrier sufficiency, per locus and horizon** — the full-state patch must
  reach flip rate `>= 0.10` with an effect CI excluding zero and exceeding the
  same-norm random patch. A cell failing G3 is recorded but its `Q/A/G` is **not
  interpreted**.

New and preregistered:

* **G4 intervention-magnitude stability.** For each (locus, estimand) curve,
  compare the mean residual fraction `||dh||/||h||` of the `Y10_scalar` arm at
  `k*` against its value at `k0`:

  ```text
  magnitude_stable  iff  r(k*) / r(k0) >= 0.80
  ```

  A curve failing G4 is **excluded from H20.1-H20.4, from every half-life
  estimate and from the outcome label**, and is reported as
  `excluded_magnitude_unstable`. Its numbers are still persisted.

  Rationale: the `native` estimand recomputes its validation setpoint targets at
  each horizon, so converging class medians shrink the edit and a falling `Q`
  would be a magnitude artifact rather than pathway loss. E19 found exactly this
  in `D_decision/native` (ratio 0.452). The `0.80` threshold is E19's value,
  promoted here from a post-hoc check to a gate.

## 5. Hypotheses

Identical in form to E19's, restated for the extended range. `k*` is the largest
horizon surviving G2.

**H20.1 — representational persistence.** `D(k*) - D(k0) > -0.05` at both loci,
tested as non-inferiority.

**H20.2 — causal-organization change under persistent representation.** At `k*`,
at least one of `Q`, `A`, `G` differs from its `k0` value by at least the SESOI
with a paired CI excluding zero, Holm-corrected across the three components, on a
G4-passing curve.

**H20.3 — differential pathway persistence.** At least two normalized component
curves differ from each other by at least the SESOI at `k*`, on a G4-passing
curve.

**H20.4 — age versus remaining distance.** The locus-D curve isolates state age;
the locus-S curve adds propagation distance. Their difference at `k*` is the
frozen descriptive estimate of the distance contribution.

**Frozen SESOI**, unchanged from E19:

```text
SESOI = 0.25 of the component's own |value at k0|, and the paired CI must
        exclude zero.
```

A component whose `k0` value is indistinguishable from zero is reported as **not
assessable** rather than given a ratio with a near-zero denominator.

## 6. The half-life question

This is the reason E20 exists, so its rule is stated explicitly and is not
relaxed afterwards.

A half-life is reported **only** when all of the following hold, as already
enforced by `metrics.temporal_half_life.half_life`:

```text
the curve's k0 baseline is positive with a CI excluding zero
Spearman rho between k and the relative curve <= -0.70
no upward step between adjacent grid points greater than 0.15
the relative curve actually reaches 0.5 inside the measured grid
```

plus, added here:

```text
the curve passes G4
```

If a curve never reaches 0.5 inside the grid, its half-life is **right-censored
at k_max** and no number is quoted. If the curve is not monotone enough, the
half-life is **not estimable** and the raw curve stands. Both are reported as
results, not as failures.

## 7. Corpus

Fresh namespace `e20-{split}-v1`, so E20 is not conditioned on rows E15, E18 or
E19 inspected:

```text
train:            600 directed /  300 pairs   seed 20262001
validation:       300 directed /  150 pairs   seed 20262002
discovery_test:   300 directed /  150 pairs   seed 20262003
```

Rendered at the candidate grid by prefix extension, so a base episode's identity,
carrier position and nuisance content stay byte-identical across horizons and
only the state-to-decision distance varies.

Seeds: probe `20262010`, directions `20262020`, bootstrap `20262030`.

## 8. Stop rules

* If Phase 1 selects a grid no longer than E19's, stop after Phase 1 and report
  that the environment cannot support a longer interpretable horizon.
* If G3 fails at a horizon, that cell's decomposition is not interpreted, and the
  horizon at which the carrier stops being sufficient is itself reported.
* If G4 fails for a curve, that curve is excluded from all inference.
* A failed gate is never repaired by adding arms, loci, layers, pools or models.

## 9. Claim boundary

E20 supports only a task-, model-, site- and discovery-specific statement about
how the causal organization of one semantic variable evolves over a longer
horizon in Qwen3-1.7B on this environment. It is not a decay law, not a memory
theory, and licenses no cross-model claim. Any half-life it reports is a
descriptive timescale for this task and carrier, nothing more.
