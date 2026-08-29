# E13 Diagnostic Confirmation

Status date: 2026-08-29. **One-shot confirmation complete. Classification: strong.**

Protocol: `docs/E13_DIAGNOSTIC_CONFIRMATION_PROTOCOL.md`
Protocol commit `6daa148`, SHA-256 `c0c96f9c79586daf14692a62a9499233fab58017132a3251d1ce02f19e2aa16a`
Implementation commit at access: `95c62be8f5b9682ec123e8d47b7be5265303a82f`
Campaign: `runs/E13_DIAGNOSTIC_CONFIRMATION/E13DC_e13-diagnostic-confirmation-v1`

## Access ledger

```text
namespace:        e13_confirmation_v1
first access:     2026-08-29T18:36:21.640521+00:00
access count:     1
rows:             200 directed
pairs:            100 counterfactual
split digest:     8b7e21c787abb064eb91c021613cd93b033674947161a1c91efb8b7e14ea102d
spec digest:      9b93c5b0d42eb42e451fd265863b7e604f3b8da928f5c8f09b62a1c9cb0d4f57
registry digest:  c2da5b704ed99887852e989013fb8f86d4c054dcf3589786adc8abe71d28be1a
dedup collisions: 12 pairs skipped against the frozen open corpus
```

The holdout continued the open corpus's global prompt-pair deduplication chain;
12 candidate pairs collided with open prompts and were skipped. No sample
identity or prompt is shared between the open and confirmation corpora.

## Integrity

| check | result |
|---|---|
| no-op hook max abs logit deviation | `0.0` for all 8 models |
| max context/direction dot product | `<= 1.8e-15` |
| max setpoint deviation | `0.1135` (teacher), `<= 0.0298` (students) |
| finite factorial evidence | all models |
| evaluated rows | 200 per model |
| eval split routing | `confirmation` with `confirmation_accessed: true` on every row |
| checkpoint identity / weights / projector | all 6 verified against the frozen registry |
| models retrained | none |

No probe, target, direction, scale, checkpoint, or threshold was refit or revised
after access.

## Primary hierarchical results

| Hypothesis | Aggregate estimate | 95% CI | Adjusted result | Verdict |
|---|---:|---|---|---|
| H13-C1 — R2 behavior non-inferior | `+0.043917` | `[+0.005415, +0.088137]` | gatekeeper, 3/3 seeds | **PASS** |
| H13-C2 — R3 behavior non-inferior | `+0.055467` | `[+0.018649, +0.098835]` | gatekeeper, 3/3 seeds | **PASS** |
| H13-C3 — R2 causal mismatch | `+0.421201` (A) | `[+0.254503, +0.586878]` | Holm `0.0004` | **PASS** |
| H13-C4 — R3 causal mismatch | `+0.507840` (A) | `[+0.371629, +0.646350]` | Holm `0.0003` | **PASS** |

Stage A margin `delta_B = 0.03`; Stage B SESOI `delta_C = 0.10`; family-wise alpha
`0.05`; 10,000 pair-cluster bootstrap draws, seed `20261304`.

Both students were not merely non-inferior but **behaviorally superior** to the
teacher on the holdout, which makes the Stage-A gate pass comfortably and removes
any "the student is just worse" reading of the causal mismatch.

### Per-seed behavioral non-inferiority

| Regime | Seed | Student B | Teacher B | Delta B | 95% CI | Non-inferior |
|---|---:|---:|---:|---:|---|---|
| R2 | 20261305 | 0.964500 | 0.936900 | +0.027600 | `[-0.017102, +0.077153]` | yes |
| R2 | 20261315 | 0.980250 | 0.936900 | +0.043350 | `[+0.002998, +0.088502]` | yes |
| R2 | 20261325 | 0.997700 | 0.936900 | +0.060800 | `[+0.024650, +0.103550]` | yes |
| R3 | 20261305 | 1.000000 | 0.936900 | +0.063100 | `[+0.027050, +0.105501]` | yes |
| R3 | 20261315 | 0.979650 | 0.936900 | +0.042750 | `[+0.002200, +0.087951]` | yes |
| R3 | 20261325 | 0.997450 | 0.936900 | +0.060550 | `[+0.024649, +0.103052]` | yes |

### Component table — all components reported

| Regime | Component | Teacher | Student (mean) | Gap | 95% CI | Holm p | Mismatch? |
|---|---|---:|---:|---:|---|---:|---|
| R2 | Q | 0.163849 | -0.003652 | **-0.167500** | `[-0.177598, -0.157585]` | 0.0003 | **yes** |
| R2 | A | 1.194830 | 1.616032 | **+0.421201** | `[+0.254503, +0.586878]` | 0.0004 | **yes** |
| R2 | G | 0.048878 | 0.061532 | +0.012654 | `[+0.001182, +0.024307]` | 1.0000 | no |
| R3 | Q | 0.163849 | -0.009262 | **-0.173110** | `[-0.182916, -0.163196]` | 0.0003 | **yes** |
| R3 | A | 1.194830 | 1.702670 | **+0.507840** | `[+0.371629, +0.646350]` | 0.0003 | **yes** |
| R3 | G | 0.048878 | 0.061361 | +0.012483 | `[+0.001872, +0.023126]` | 1.0000 | no |

Teacher and student values are the validation-standardized means implied by the
per-example gaps. Q and A are mismatched in both regimes, in the same direction,
in 3/3 seeds. **G is not mismatched in either regime**: its signed mean gap is
about `+0.012`, an order of magnitude inside the `0.10` SESOI, and its Holm p is
`1.0`. This is reported as a genuine null, not omitted.

The G null is informative rather than merely absent. Per-example G gaps are large
in magnitude (discovery mean absolute gap around `0.083`) but not systematically
directional, so the students differ from the teacher on G example-by-example
without a consistent bias. Q and A differ *systematically*.

## Secondary: decodability

| Model | B | D native | D frozen initial axis |
|---|---:|---:|---:|
| Teacher (Qwen3-1.7B) | 0.936900 | 1.000000 | 1.000000 |
| R0 (untrained Qwen3-0.6B) | 0.800500 | 1.000000 | 1.000000 |
| R2 seed 20261305 | 0.964500 | 1.000000 | 0.982700 |
| R2 seed 20261315 | 0.980250 | 1.000000 | 0.995200 |
| R2 seed 20261325 | 0.997700 | 1.000000 | 0.995000 |
| R3 seed 20261305 | 1.000000 | 1.000000 | 1.000000 |
| R3 seed 20261315 | 0.979650 | 1.000000 | 0.994800 |
| R3 seed 20261325 | 0.997450 | 1.000000 | 0.995800 |

`D_native = 1.000000` for **every** model, including the untrained student and the
teacher. The semantic variable is perfectly and equally decodable across all
eight models on the holdout. This is the premise the confirmed claim needs, and
it holds without qualification: the mismatch is not a decodability deficit.

## Secondary: representation similarity, and a narrowed claim

| Model | linear CKA | mean cosine after projector | projected hidden MSE |
|---|---:|---:|---:|
| R0 | 0.744678 | — | — |
| R2 seed 20261305 | 0.748461 | — | — |
| R2 seed 20261315 | 0.777240 | — | — |
| R2 seed 20261325 | 0.783324 | — | — |
| R3 seed 20261305 | 0.555858 | 0.090449 | 1.213591 |
| R3 seed 20261315 | 0.775059 | 0.015310 | 1.311508 |
| R3 seed 20261325 | 0.782145 | 0.017920 | 1.302104 |

**The pre-registered descriptive question does not resolve in R3's favour, and the
claim is narrowed accordingly.** At the validation-selected B-matched checkpoints,
explicit hidden-state KD did **not** clearly raise teacher/student representation
similarity: R3's mean CKA is `0.704` against R2's `0.770` and the untrained R0's
`0.745`, and R3 is below R2 in two of three seeds.

The honest reading is bounded by the operating point. R3's B-matched checkpoints
are step 25, 10 and 10 — the hidden-state objective has had only 10 to 25
optimizer updates there, and the projector is correspondingly weak (mean cosine
`0.015`-`0.090`, MSE about `1.2`-`1.3`). Discovery separately showed R3 projected
cosine and MSE improving by step 100 while COD did not follow. So the supported
statement is:

> At behavior-matched checkpoints, hidden-state KD neither reproduced the
> teacher's causal organization nor clearly improved representation similarity.

It is **not** supported to say "R3 improved representation similarity but not
causal organization" on this evidence. Representation-similarity views were not
used to support any primary claim, and none was needed: the primary passed on its
own terms.

## COD, reported as secondary only

COD is deliberately not a confirmatory endpoint (protocol section 2). The
method-revision discovery showed it is a magnitude-dominated norm that improves
under generic response regularization, and that family-shuffled targets carrying
no sample-level semantic correspondence achieved the campaign's lowest COD. It is
retained in the campaign artifacts as a descriptive quantity and carries no part
of the confirmed claim.

## Confirmed claim

Both regimes passed both stages, so the strong form is supported:

> **A student can achieve teacher-like — here, teacher-exceeding — task behavior
> after distillation while retaining a systematically different causal
> organization of a semantic variable that is perfectly decodable in both models.**

And, because R3 passed as well:

> **Explicit hidden-state matching does not guarantee recovery of teacher causal
> organization.**

The mismatch is specific in form, and its shape is the scientifically interesting
part. The teacher converts the scalar semantic coordinate into behavior
(`Q_z = 0.1638`); both distilled students essentially do not (`Q_z` about `-0.0037`
to `-0.0093`, a gap of `-0.17` standardized units in 3/3 seeds for both regimes).
Meanwhile both students over-rely on the matched-context additive pathway
(`A` gaps of `+0.42` and `+0.51`). Distillation transferred the behavior and left
the decodable variable intact, but rebuilt how that variable is used: less scalar
conversion, more contextual addition.

What is **not** claimed: that causal organization can be transferred. The
conversion-response method branch failed its frozen success criterion and is
closed; nothing in this confirmation revives it.

## Claim boundary

One teacher/student family (Qwen3-1.7B to Qwen3-0.6B), one synthetic relation
task with five families, one layer (17), one site (`resid_post`), one token
selector (`last_prompt`), 100 optimizer steps, three seeds, 200 held-out directed
examples. Probe results establish decodability, not endogenous causal use. The
components `(D, Q, A, G, B)` are a causal-organization *profile*, not a strict
hierarchy. Cross-family generality is tested separately in E17 and is not assumed
here.

## Cross-family entry gate

`entry_gate_for_cross_family: true` — both R2 and R3 passed behavioral
non-inferiority and causal mismatch, so the E17 cross-family replication is
authorized under the frozen rule.
