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

### A3. E01B-3 additive-vs-gating factorial decomposition — strongly confirmed

Goal: separate the E01B-2 context increment into an independent context-only
effect and a true interaction with the frozen scalar intervention.

The pre-registered four-arm design is:

```text
Y00 clean
Y10 semantic setpoint only
Y01 orthogonal context only
Y11 semantic setpoint plus context
```

It estimates additive context signal `A = Y01-Y00` and factorial interaction
`G = (Y11-Y10)-(Y01-Y00)`. Site, targets, source plans, context norms, lambdas,
controls, traces, and pair-cluster inference are frozen from E01B-1/E01B-2.
Full discovery is complete. In 0.6B, structured context increments are almost
entirely additive: structured A is positive, while G and all
structured-minus-random G contrasts include zero. In 1.7B, structured A is much
larger and a smaller positive structured G also exceeds random, supporting a
mixed additive-plus-gating result. L17 q remains fixed; additive readout effects
appear immediately and 1.7B develops downstream q/readout interaction. The
additive matched/same-family advantage does not carry over to G; different-
family interaction is largest in 1.7B. The single preregistered confirmation
passed H1-H4 after Holm correction. The core mechanism is frozen: scalar q is
causally effective; structured orthogonal state carries independent causal
information in both checkpoints; and 1.7B additionally shows structured
q-by-context interaction. No such interaction was detected in 0.6B, without an
equivalence claim.

## Phase B — test fragility under compression

### B1. E14 quantization reliability — default first extension

Status: **strong E14 confirmation complete; mechanism frozen** under
`docs/E14_FULL_DISCOVERY_AND_CONFIRMATION_PROTOCOL.md`.

Goal: test whether representational availability survives compression better than causal utilization.

Start with Qwen3-1.7B:

```text
BF16 -> INT8 -> INT4
```

Primary question:

```text
Can D remain near ceiling while scalar Q, multidimensional additive A, or
q-by-context interaction G drops?
```

The bounded Qwen3-1.7B ladder uses one weight-only backend (Optimum-Quanto
0.2.7) across BF16/INT8/INT4. It measures precision-native and frozen-BF16-axis
D, behavior B, source-free Q0, matched/random A and G, and depth tracing.

Why first:

- no retraining required;
- low compute;
- same checkpoint controls many confounds;
- direct deployment relevance;
- high novelty-to-effort ratio.

Do not add INT3/INT2 until BF16/INT8/INT4 are stable.

Bounded result: INT8 preserved both decodability views and Q/A/G. INT4 kept
precision-native D at ceiling but reduced frozen-BF16-axis D, reduced
structured-minus-random A by 14.9%, and reduced structured-minus-random G by
49.3%; Q increased. Native relation-margin discrimination also declined, while
prompt perplexity did not catastrophically worsen. The pilot therefore supports
higher-order compression fragility but does not yet isolate semantic-specific
damage from all task-level degradation. Full discovery is now authorized with
300 directed examples, ten random seeds, both frozen lambdas, and preregistered
WikiText/HellaSwag controls. A new E14-specific holdout may be accessed once only
if native INT4 D remains at least 0.99 and the paired G-reduction CI excludes
zero. See `E14_BOUNDED_PILOT_SUMMARY.md`.

Full discovery passed that gate. Across 300 directed examples, INT4 retained
precision-native D at 0.99991 AUROC but reduced structured-minus-random A by
26.0% and G by 53.9%; both paired CIs excluded zero, while Q increased by 17.7%.
WikiText-2 perplexity increased 65.2%, crossing the frozen generic-damage flag,
while HellaSwag accuracy fell by 0.07. Consequently, confirmation is authorized
but any positive result must be described as mixed actionability and general
degradation rather than selective semantic damage. See
`E14_FULL_DISCOVERY_SUMMARY.md`.

The single E14 confirmation then passed H14.1-H14.3 after Holm correction:
native INT4 D remained 1.0, while structured-minus-random A and G were both
lower than BF16 with directional paired CIs excluding zero. Because WikiText
PPL again crossed the generic-damage flag, freeze the claim as mixed
actionability plus general degradation. The E14 holdout is consumed. This strong
confirmation unlocks the separately preregistered bounded E13 distillation
branch; it does not authorize reopening E14 or testing a new quantizer.

## Phase C — test transfer and learnability

### C1. E13 distillation reliability — gated on E01B

Status: **discovery complete; method branch closed as diagnostic**. Three-seed
R0/R1/R2/R3 and R4/R5/R6/R2-C full discovery passed all frozen baseline Gates
A-E and all 12 authorized method jobs completed. The subsequent method-revision
campaign (R7-R16, one bounded seed, `runs/E13_METHOD_REVISION/`) then failed its
frozen bounded gate with `selected_regime: null`, so no three-seed method wave
was authorized or run. Classification `METHOD REVISION INCONCLUSIVE`;
recommendation is to retain E13 as diagnostic evidence and not to preregister a
confirmation. E13 confirmation remains locked, unmaterialized, and unaccessed.
See `E13_METHOD_REVISION_DISCOVERY_SUMMARY.md`.

Pilot result: D was saturated throughout and both SFT/KD reached B=1.0, but Q
remained weak—especially under KD—while A/G changed strongly and differently by
objective. SFT substantially overshot teacher A/G; KD ended near teacher G but
far above teacher A and near baseline Q. This justifies multi-seed replication
and triggers a proposed conversion-response objective, but neither is authorized
by the one-seed pilot. The newly authorized discovery adds validation-scaled,
probability, flip, causal-organization-distance, behavior-matched-checkpoint,
representation-similarity, and general-quality views. See
`E13_BOUNDED_PILOT_SUMMARY.md` and the multi-seed protocol.

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

### C2. conversion-response distillation — discovery complete

The frozen multi-seed gate passed and the preregistered R4/R5/R6/R2-C campaign
is complete. R5 had the lowest mean COD (`0.543`) but failed the frozen primary
criterion because one seed missed teacher-like validation B; R5 also failed to
improve all Q/A/G gaps and did not consistently beat R6. R2-C exactly matched
R2, ruling out extra detached-forward compute as the explanation. The current
classification is heterogeneous/unresolved method evidence, and the next action
is to revise the conversion-response method in a newly frozen discovery design,
not to access confirmation.

Trigger only if R2 or R3 achieves teacher-like validation behavior in at least
two seeds while a preregistered standardized causal-organization gap remains and
general quality is interpretable. If the gate passes, the executed method
protocol must be committed and pushed before any R4/R5/R6/R2-C training.

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

The E01 confirmation split was consumed exactly once after remote
preregistration; the result was strong confirmation. It must not be reopened or
used for core-mechanism tuning.

Application branches must not consume that holdout. Each new branch should have its own discovery/confirmation partition where appropriate.

## Branch priority

Current recommended ordering:

| Rank | Experiment | Why |
|---:|---|---|
| 1 | E01B-1 | complete: establishes donor-free causal object |
| 2 | E01B-2 | complete: establishes contextual sensitivity in discovery |
| 3 | E01B-3 | strong confirmation complete; mechanism frozen |
| 4 | E14 | strong confirmation complete; mixed actionability/general degradation |
| 5 | E13 | diagnostic claim CONFIRMED (strong); holdout consumed; method branch stays closed |
| 5b | E17 | cross-family replication COMPLETE; phenomenon replicated in OLMo-2 |
| 6 | E16 | deepest developmental claim if checkpoints permit; not authorized |
| 7 | E15 | highest long-horizon conceptual upside; not authorized |

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

E13 supports this diagnostic claim and it is the part worth writing. The method
half did not survive. Conversion-response distillation does **not** repair the
gap: its frozen bounded revision gate selected no candidate from six objective
variants, and mechanism controls showed the apparent COD benefit is generic
local-sensitivity regularization — shuffling teacher targets within relation
family, destroying all sample-level semantic correspondence, gave the lowest COD
in the campaign. A semantic-specific signal exists only in the smallest profile
component (Q), which the frozen selection statistic cannot resolve. There is no
separate method contribution to claim.

If the method branch is ever resumed it needs two changes, both requiring a new
frozen discovery design: a response coefficient calibrated per objective against
the KD gradient scale rather than fixed at 1.0 (the response gradient currently
runs 3-5x the KD gradient), and a selection statistic sensitive to per-example
profile correspondence rather than a magnitude-dominated norm.

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
