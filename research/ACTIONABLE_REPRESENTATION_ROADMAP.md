# Actionable Representation Research Roadmap

Status date: 2026-08-28

Master question:

> **What makes an internal representation actionable?**

This roadmap converts the master question into a gated experiment program.

## Phase A — establish the causal object

### A1. E01B-1 source-free setpoints — full discovery complete

Goal: establish whether donor-free scalar targets reproduce the E01A causal response.

Success signature:

```text
source-free semantic setpoints -> monotonic native-readout response
semantic treatment > random/orthogonal controls
1.7B conversion > 0.6B conversion
```

Outcome: donor-free semantic setpoints reproduce E01A, beat norm-matched
random/orthogonal controls, and preserve the large 1.7B-over-0.6B conversion
gap. The 1.7B grid is strongly monotonic; 0.6B has a positive population slope
but weak per-example monotonicity. Confirmation remains untouched.

### A2. E01B-2 orthogonal-context modulation — full discovery complete

Goal: determine whether the same scalar semantic displacement has stable causal meaning across orthogonal representational contexts.

Success outcomes are informative in either direction:

- context-insensitive -> approximately one-dimensional causal control;
- structured-context-sensitive -> context-gated utilization;
- random-context-sensitive -> semantic specificity concern.

Outcome: structured opposite-label orthogonal contexts produce much larger
effects than norm-matched random context at the same scalar target in both
models. Matched and same-family effects exceed different-family effects in the
aggregate, although the ordering is heterogeneous by relation family. The
effect enters native readout immediately at L17 while q is fixed and then
changes downstream q propagation. This supports operational context-gated
utilization, while leaving additive orthogonal signal versus multiplicative
gating unresolved. Confirmation remains untouched.

The A2 mechanism must now be frozen before any separately authorized
confirmation run.

## Phase B — test fragility under compression

### B1. E14 quantization reliability — default first extension

Goal: test whether representational availability survives compression better than causal utilization.

Start with Qwen3-1.7B:

```text
BF16 -> INT8 -> INT4
```

Primary question:

```text
Can D remain near ceiling while C/readout conversion drops?
```

Why first:

- no retraining required;
- low compute;
- same checkpoint controls many confounds;
- direct deployment relevance;
- high novelty-to-effort ratio.

Do not add INT3/INT2 until BF16/INT8/INT4 are stable.

## Phase C — test transfer and learnability

### C1. E13 distillation reliability — gated on E01B

Goal: determine whether distillation transfers representation and utilization together or separately.

Primary regimes:

```text
frozen student
hard-label SFT
logit KD
hidden-state/trajectory KD
optional combined KD
```

Track:

```text
D(t), readout(t), C(t), B(t)
```

High-value result:

```text
D already high
C changes substantially under KD
```

or hidden-state similarity improves without corresponding C transfer.

### C2. conversion-response distillation — method trigger only

Do not implement by default.

Trigger only if standard KD leaves a robust teacher-student conversion gap after controlling for task learning.

Then test whether the student's response to source-free latent setpoints can be trained to match the teacher's response curve.

## Phase D — test emergence during training

### D1. E16 utilization emergence — checkpoint family required

Goal: determine whether representational accessibility emerges before causal utilization during training.

Primary signature:

```text
D(t) plateaus
while
C(t) continues to rise
```

This would turn a static checkpoint comparison into a developmental mechanism.

Only start after identifying a public checkpoint family with stable architecture and sufficient intermediate checkpoints.

## Phase E — test temporal persistence

### E1. E15 temporal causal half-life — later conceptual extension

Goal: determine whether a represented state remains decodable after its causal influence on future actions has decayed.

Primary signature:

```text
H_C < H_D
```

where `H_C` is causal half-life and `H_D` is representation half-life.

Do not begin with open-ended agents. First freeze one structured stateful task with explicit latent state and auditable future decisions.

## Confirmation policy

The existing E01 confirmation split remains locked until the E01B mechanism is frozen.

Application branches must not consume that holdout. Each new branch should have its own discovery/confirmation partition where appropriate.

## Branch priority

Current recommended ordering:

| Rank | Experiment | Why |
|---:|---|---|
| 1 | E01B-1 | complete: establishes donor-free causal object |
| 2 | E01B-2 | complete: establishes contextual sensitivity in discovery |
| 3 | E14 | cheapest strong test of utilization fragility; not authorized |
| 4 | E13 | highest transfer/method upside; not authorized |
| 5 | E16 | deepest developmental claim if checkpoints permit; not authorized |
| 6 | E15 | highest long-horizon conceptual upside; not authorized |

## Paper decision points

### Paper path 1 — core mechanism paper

If E01B succeeds strongly and confirmation replicates:

```text
D ≠ C across checkpoints
causal setpoint sufficiency/context
readout-conversion localization
```

This can stand without every extension.

### Paper path 2 — compression/application paper

If E14 shows D robust but C fragile:

> Representational integrity is not functional integrity under quantization.

### Paper path 3 — transfer/method paper

If E13 shows a representation/utilization transfer gap:

> Distillation can transfer what is represented and how it is used at different rates.

If conversion-response distillation repairs this gap, that becomes a separate method contribution.

### Paper path 4 — developmental paper

If E16 shows `t_D < t_C`:

> Models learn a semantic variable before learning to deploy it functionally.

### Paper path 5 — long-horizon paper

If E15 shows `H_C < H_D`:

> Information can remain internally represented after its ability to govern future action has decayed.

## Stop rules

Do not keep expanding breadth if a branch fails its mechanism gate.

Do not collect many weak applications. Prefer one clean transformation that shows a large D/C dissociation.

Do not claim a universal theory from one variable, model family, or task.

Do not open confirmation merely because discovery results are exciting. Freeze the hypothesis, intervention, metrics, and mechanism first.
