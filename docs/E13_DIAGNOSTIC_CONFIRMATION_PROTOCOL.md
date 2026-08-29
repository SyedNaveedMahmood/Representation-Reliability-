# E13 Diagnostic Confirmation Protocol

Status: **frozen before any access to `e13_confirmation_v1`**. This document and
the runner that implements it are committed and pushed before the holdout is
materialized. Every quantity below is fixed by discovery evidence that already
exists; nothing here may be revised after access.

## 1. What is being confirmed

The confirmed claim is the **diagnostic** one, not the failed conversion-response
method:

> A student can achieve teacher-like task behavior after distillation while
> retaining a systematically different causal organization of an already-decodable
> semantic variable.

The method branch (R4-R16) is closed. This protocol does not test, rescue, or
reference any conversion-response objective as a confirmatory endpoint.

The causal-organization profile is `C = (D, Q, A, G, B)`. These are components of
a profile, **not** a strict causal hierarchy, and are reported as such.

```text
D = semantic probe AUROC
B = native target-label Yes-minus-No margin AUROC
Q = Y10 - Y00
A = Y01,matched - Y00
G = (Y11 - Y10) - (Y01 - Y00), matched context
```

## 2. Why COD is not a confirmatory endpoint

Causal Organization Distance may appear only as a **descriptive secondary
metric**. It is disqualified as a primary endpoint because the completed
method-revision discovery established that:

* COD is a mean per-example Euclidean norm over `(Q_z, A_z, G_z)` and is
  therefore dominated by the largest component, A (teacher train SD `0.679`
  versus Q `0.072` and G `0.098`);
* generic response regularization reduces COD without any sample-specific causal
  transfer — random-response R6 captured most of R5's improvement; and
* R8, whose teacher targets were permuted within relation family so that no
  sample received its own causal response, obtained the **lowest COD of the whole
  campaign** (`0.454964`).

A statistic that improves under destroyed semantic correspondence cannot carry a
confirmatory claim. Primary inference therefore uses componentwise gaps.

## 3. Frozen model and checkpoint identities

Teacher: `Qwen/Qwen3-1.7B`, model and tokenizer revision
`70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
Student base (R0): `Qwen/Qwen3-0.6B`, model and tokenizer revision
`c1899de289a04d12100db370d81485cdf75e47ca`.

Baseline discovery campaign: `runs/E13_MULTI_SEED/E13MS_04daa7fcc66c`, protocol
SHA-256 `04daa7fcc66cc1c93f8077de23962dfec9861c9412c44367d83603ed0ccb7cac`,
corpus digest `853660ad567a44fba01762cef8f03f4150c552ddf99c502f58bc5f88202e536b`.

**No model is retrained.** The six confirmed students are exactly the existing
discovery-trained checkpoints selected by the already-frozen validation-only rule

```text
t* = argmin_t |B_validation_student(t) - B_validation_teacher|,  earliest-step tie-break
```

over steps `0/10/25/50/100`. Discovery causal quantities never entered that
selection, and no confirmation outcome may alter it.

| key | regime | seed | step | validation B | abs. validation B gap | run identity (sha256) | weight sha256 |
|---|---|---|---|---|---|---|---|
| R2_seed_20261305 | R2 | 20261305 | 10 | 0.947408 | 0.026200 | `6c3082fa17564aad…` | `c6276176b2dadbb0…` |
| R2_seed_20261315 | R2 | 20261315 | 10 | 0.959080 | 0.014528 | `d6c714d2f2a5e15c…` | `37820917b3fbdca3…` |
| R2_seed_20261325 | R2 | 20261325 | 10 | 0.990304 | 0.016696 | `0c80e8504997dfa4…` | `5d86f7badce0f264…` |
| R3_seed_20261305 | R3 | 20261305 | 25 | 1.000000 | 0.026392 | `61cec33ea4d5a3e1…` | `418ce8d4adfe9b2a…` |
| R3_seed_20261315 | R3 | 20261315 | 10 | 0.959552 | 0.014056 | `10a626202961670f…` | `28969080420d7cfa…` |
| R3_seed_20261325 | R3 | 20261325 | 10 | 0.991216 | 0.017608 | `03ebb10bf53579c1…` | `ef7adf7137ef7567…` |

Teacher validation B is `0.973608`. R3 entries additionally lock their trained
projector by SHA-256. Full-length digests live in `CHECKPOINT_REGISTRY` in
`src/representation_reliability/runners/e13_diagnostic_confirmation_support.py`;
the registry digest is recorded in the run manifest. `resolve_checkpoint` refuses
any checkpoint whose atomic marker identity, step, weight digest, or projector
digest differs from the registry.

## 4. Frozen mechanism

Unchanged from E13 discovery and E01: layer `17` (zero-indexed), canonical
`resid_post`, `last_prompt` selector, native module name persisted, candidate
token IDs `[7414, 2308]`.

* **Probe / scaler.** Each model uses its own model-local reference. The student
  reference (probe coefficients, scaler, semantic direction, q-targets) is loaded
  verbatim from the frozen discovery artifact
  `runs/E13_MULTI_SEED/E13MS_04daa7fcc66c/reference/initial_student_reference.npz`
  and `initial_student_targets.json`; all six trained checkpoints and R0 are
  probed along that same fixed initial-student axis. The teacher reference is
  rebuilt deterministically by `_reference_from_model` from **train and
  validation rows only**. Probe fitting touches `split in {train, validation}`
  exclusively and can never see confirmation rows.
* **q targets.** Source-free `q0*`/`q1*` setpoints from the frozen targets file;
  a row's target is `q1*` when its gold label is `0`, else `q0*`.
* **Semantic direction.** The frozen unit direction from the same reference.
* **Context construction.** Matched orthogonal context is the orthogonal
  component of the counterfactual partner's activation with respect to the base
  and the semantic direction, standardized to the base row's reference norm.
* **Random controls.** Frozen direction seeds `2130/2131/2132`, each norm-matched
  per row to that row's matched-context reference norm.
* **Validation scales.** `sigma_margin_validation` is the population SD of clean
  native margins on the **validation** split, recomputed per model with that
  model's own scale. It never uses confirmation rows. `Q_z`, `A_z`, `G_z` are the
  raw factorial effects divided by that scale.
* **Cross-space rule.** Teacher and student always operate in their own hidden
  spaces (2048 vs 1024). No direction, probe, or target is reused across models.

## 5. Confirmation split specification

```text
namespace:        e13_confirmation_v1
generator:        generate_synthetic_relations
generator_seed:   20261304
n_directed:       200
n_pairs:          100
n_entities:       42
families:         north_south, east_west, above_below, before_after, larger_smaller
sample_id_prefix: e13-confirmation-v1-sample-
pair_id_prefix:   e13-confirmation-v1-pair-
```

Specification digest (canonical JSON, SHA-256):
`9b93c5b0d42eb42e451fd265863b7e604f3b8da928f5c8f09b62a1c9cb0d4f57`.

The holdout continues the frozen open corpus's **global prompt-pair
deduplication chain**: the open corpus (train 4,000 / validation 500 /
discovery 300) is rebuilt first, its prompt-pair signatures are collected, and
confirmation pairs colliding with them are skipped. Counterfactual pairs are
never split. The runner asserts that the combined open + confirmation frame has
no duplicate sample identities and no duplicate prompts.

**Access ledger.** `E13_CONFIRMATION_ACCESS.json` records the first-access UTC
timestamp, `access_count: 1`, and the protocol identity. Resume does not
increment the count; a second campaign is refused.

## 6. Primary confirmatory structure

Hierarchical gatekeeping. Stage B is interpretable for a regime **only if** that
regime's Stage A test passes.

### Stage A — behavioral non-inferiority (gatekeeper)

For regime `R` and seed `s`, `Delta_B = B_R,s - B_T` on the confirmation rows.
The frozen discovery margin is reused unchanged:

```text
delta_B = 0.03
```

* **H13-C1** — R2 behavior remains teacher-like: `Delta_B > -0.03`.
* **H13-C2** — R3 behavior remains teacher-like: same test, same margin.

`B` is an AUROC over the whole confirmation set, so it is not a per-example
quantity. The bootstrap therefore resamples **counterfactual pairs** with
replacement and recomputes teacher and student AUROC inside each draw, keeping
both members of a pair together. The seed aggregate is hierarchical: one draw
resamples pairs once, every seed is re-evaluated on that same resampled pair set,
and the draw statistic is the mean over seeds.

PASS requires **both**:

```text
aggregate lower 95% bound > -0.03
AND at least 2/3 individual seeds have point gap > -0.03
```

### Stage B — causal-organization mismatch

For each component `X` in `{Q, A, G}` the per-example gap is
`Delta X_z = X_z,student - X_z,teacher`, using the validation-standardized
definitions of section 4. The smallest effect size of interest reuses the
`0.10` standardized floor already used in the discovery method-trigger logic:

```text
delta_C = 0.10
H0: |mean Delta X_z| <= 0.10     vs     H1: |mean Delta X_z| > 0.10
```

A component is **confirmatorily mismatched** only when all three hold:

1. its 95% pair-cluster CI lies completely outside `[-0.10, +0.10]`;
2. its Holm-adjusted p-value is `< 0.05` within that regime's `{Q, A, G}` family;
3. the same direction is reproduced beyond the SESOI in at least 2/3 seeds.

The p-value is a one-sided pair-cluster bootstrap probability against the nearer
SESOI boundary, with the standard `(exceed + 1) / (draws + 1)` correction; when
the point estimate lies inside the null region the test cannot reject and
`p = 1`.

* **H13-C3** — R2 causal mismatch: Stage-A R2 passes **and** at least one of
  Q/A/G is confirmatorily mismatched.
* **H13-C4** — R3 causal mismatch: same rule. This asks whether hidden-state KD
  can achieve teacher-like behavior while causal organization still differs.

**All three components are reported whether or not they are significant.** It is
a protocol violation to headline only the favorable component.

Bootstrap: 10,000 draws, seed `20261304`, percentile intervals, sampling unit =
counterfactual pair.

## 7. Multiplicity

Family-wise alpha `0.05`, Holm step-down over exactly `{Q, A, G}` **within each
regime**. Stage-A behavioral tests are gatekeepers and are deliberately not
members of the component family; they are not Holm-adjusted against it. The
hierarchy is fixed here, before access.

## 8. Secondary analyses (pre-registered, non-confirmatory)

`D_native`; `D_frozen_initial_axis`; `Q_prob`/`A_prob`/`G_prob`; strict target
flip rates (q-only, context-only, joint, matched and per random seed, plus
matched-minus-mean-random contrasts); COD and controlled COD; per-example Q/A/G
profile correlations; teacher/student causal sign agreement; linear CKA; and
projected hidden cosine/MSE for R3 where already well defined.

Representation-similarity analyses **must not be used to rescue a failed primary
causal claim**. There is no formal primary test that "R3 improves representation
similarity". R2 and R3 similarity views are reported descriptively and answer:
did the explicit hidden-state objective raise teacher/student representation
similarity while causal mismatch persisted? If the representation metric does not
clearly improve on confirmation, the claim is narrowed accordingly.

## 9. Classification

* **Strong diagnostic confirmation** — H13-C1 and H13-C3 pass, and preferably
  H13-C2 and H13-C4 as well. If both regimes pass both stages, the supported
  statement is that *both output distillation and representation distillation can
  produce teacher-like behavior without reproducing teacher causal organization*.
* **Partial confirmation** — e.g. R2 confirms and R3 does not, or behavior
  confirms while causal mismatch is inconclusive in one regime. The narrower
  supported statement is used.
* **Failure** — if R2 behavioral non-inferiority or R2 causal mismatch materially
  fails, stop before any cross-family replication, unless R3 independently
  produces a clear behavioral + mismatch confirmation. At least one standard
  distillation regime must confirm the central phenomenon.

The claim `causal organization can be transferred` is **not** available from this
confirmation under any outcome; the method branch did not establish it.

## 10. Integrity, stop conditions, and a disclosed deviation

Stop conditions: checkpoint identity/step/weight/projector drift; probe, target,
direction, or scale refit; any attempt to evaluate confirmation rows without the
confirmation flag or open rows with it; nonfinite factorial evidence; no-op hook
deviation above `1e-6`; context/direction dot product above `1e-10`; duplicate
identities or prompts across splits; a second access attempt.

`validate_evaluation_routing` makes split routing and the persisted
`confirmation_accessed` flag mutually entailing, so open evidence can never be
mislabelled as confirmation evidence or vice versa.

**Disclosed deviation.** During implementation, an earlier revision of
`tests/test_e13_diagnostic_confirmation.py` called `materialize_e13_holdout` with
the frozen specification and asserted its shape, namespace, and determinism.
That generated the 200 holdout **prompt strings** in memory before this protocol
was committed. No model was ever run on them, no label or causal quantity was
inspected, nothing was written to disk, and the access ledger was not opened; the
assertions were authored before execution and all passed, so no design choice
could have been influenced. The tests were rewritten to exercise the same
contract against a decoy namespace (`generator_seed 999001`), and
`materialize_e13_holdout` now accepts an explicit specification so the frozen
namespace is reachable only from the runner, after the ledger opens. This is
recorded here rather than silently omitted.

## 11. Artifacts

`runs/E13_DIAGNOSTIC_CONFIRMATION/` holds the access ledger, run manifest, frozen
checkpoint selection evidence, holdout identity digests, per-model metrics, raw
per-example factorial and validation rows written **before** any aggregate, the
primary and component result tables, and the verdict JSON. The report is
`E13_DIAGNOSTIC_CONFIRMATION_SUMMARY.md`.
