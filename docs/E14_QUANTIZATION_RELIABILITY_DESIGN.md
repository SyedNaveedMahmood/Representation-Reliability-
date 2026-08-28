# E14 — Quantization Reliability of Distributed Actionability

Status: bounded Stage 0/1, frozen full discovery, and the single locked
E14-specific confirmation are complete. The confirmation protocol is frozen under
`docs/E14_FULL_DISCOVERY_AND_CONFIRMATION_PROTOCOL.md`. The consumed E01
confirmation remains prohibited.

The E14 confirmation strongly passed H14.1-H14.3. The final claim is mixed
actionability plus general degradation because the frozen WikiText catastrophe
flag fired. Both E01 and E14 holdouts are consumed and inaccessible.

## 1. Question and frozen mechanism

> Does post-training weight quantization preserve semantic decodability while
> disproportionately degrading one or more components of causal actionability?

The locked E01 confirmation established four quantities that E14 keeps distinct:

```text
D  semantic decodability
Q  source-free scalar-setpoint effect, Q0 = Y10 - Y00
A  context-only additive effect, A_c = Y01,c - Y00
G  q-by-context interaction,
   G_c = (Y11,c - Y10) - (Y01,c - Y00)
```

E14 asks which component fails first under compression. It does not retune the
frozen E01 mechanism.

## 2. Scope, model, and data

Primary and only model in Stage 0/1:

```text
Qwen/Qwen3-1.7B
model revision: the resolved revision in the frozen E01 artifacts
tokenizer revision: the resolved revision in the frozen E01 artifacts
```

Use the existing synthetic-relation train/validation/discovery split and prompt
construction. Stage 0 uses at most 50 directed discovery examples; Stage 1 uses
at most 150. Pair subsets are deterministic and whole-pair preserving. The E01
confirmation split and its raw artifacts are inaccessible to E14.

Probe fitting and target construction use train/validation only. Evaluation and
all E14 comparisons use the bounded discovery subset.

## 3. Frozen intervention identity

```text
site: resid_post
layer: 17 (0-indexed)
token selector: last_prompt, resolved per sample
candidate strings: " Yes", " No"
trace layers: 17,20,23,27
q targets: validation-only opposite-class medians
matched context: deterministic matched counterfactual orthogonal component
random context: deterministic norm-matched random orthogonal component
context lambda: 1.0
```

The probe direction, validation target construction, matched context construction,
random seed identity, orientation, context norm, and factorial algebra are the
same scientific definitions used by the frozen E01 pipeline. They are recomputed
within a precision only where explicitly required below; discovery outcomes never
select a direction, target, context, or threshold.

## 4. Quantization ladder and backend

Use a single backend and algorithm family:

```text
backend: optimum-quanto 0.2.7
BF16: unquantized Hugging Face reference
INT8: qint8 weight-only
INT4: qint4 weight-only
activation/compute dtype: bfloat16
activation quantization: none
calibration data: none (weight-only PTQ)
quantized modules: every backend-supported Linear module, including lm_head
excluded modules: embeddings and normalization layers (unsupported weight types)
freeze: optimum.quanto.freeze after in-place quantize
```

INT8 uses per-output-channel symmetric absmax scaling. INT4 uses the backend's
per-output-channel groupwise affine min/max optimizer with floating shift (no
integer zero point); the group size is 128 when the input dimension permits it,
otherwise the backend deterministically tries 96, 64, and 32, and falls back to
ungrouped affine quantization when none divides the input dimension. The manifest
must record the effective qtype, group size, kernel/tensor implementation, and
all module exclusions for every quantized module.

No INT3, INT2, alternative backend, activation quantization, mixed algorithm, or
module-specific accuracy rescue is allowed in Stage 0/1.

## 5. Measurements

### D — representation

Report two held-out discovery AUROCs (and AUPRC):

1. **precision-native probe**: train-standardized logistic probe fitted separately
   for each precision, C selected on validation only;
2. **frozen BF16 axis**: the exact BF16 train-fitted scaler and linear decision
   function applied unchanged to each precision's runtime BF16 residuals.

The second view measures preservation of the original coordinate, not the best
recoverable coordinate after quantization.

### B — native behavior and general quality

Report native Yes/No balanced accuracy, raw candidate margin discrimination, and
mean causal-language-model prompt NLL/perplexity on the same bounded discovery
examples. Token IDs and prompt token counts must be identical across precisions.

### Q, A, and G — actionability

For target-oriented Yes-minus-No margins, run:

```text
Y00 clean
Y10 source-free opposite-class setpoint
Y01,matched matched orthogonal context only
Y11,matched setpoint plus matched context
Y01,random norm-matched random orthogonal context only
Y11,random setpoint plus random context
```

Report raw and validation-standardized:

```text
Q0
A_matched
A_random
A_matched - A_random
G_matched
G_random
G_matched - G_random
```

Use pair-cluster bootstrap intervals. Stage 0 and Stage 1 use preregistered random
seeds 1729, 1730, and 1731 when their profile permits; Stage 0 uses only the first
seed, and Stage 1 uses all three. Average random-seed rows per base before
structured-minus-random contrasts.

### Layerwise localization

At L17/L20/L23/L27 save raw clean/intervened q and native margin for every arm.
Report factorial q and margin A/G decompositions. Standardization uses each
precision's validation-only coordinate and native-margin scales, while all raw
effects remain available.

## 6. Numerical and engineering gates

Each precision must satisfy:

```text
model and tokenizer revision identity
candidate token ID identity
runtime residual captured in floating BF16
last_prompt/right-padding and batch-row identity
Y00 hooked no-op equals unhooked clean within dtype-aware tolerance
Y10/Y11 target projection fidelity
Y01 q preservation
matched/random context orthogonality
per-example context norm matching
hook removal and no cross-batch leakage
complete raw rows and traces before aggregates
finite activations, logits, losses, and metrics
```

Stage 0 stops a precision condition as catastrophically inoperable if any core
numerical gate fails, prompt perplexity is nonfinite, or prompt perplexity exceeds
10 times BF16 on the identical examples. This is an engineering exclusion, not a
claim that smaller quality changes are immaterial.

## 7. Stage 0 and Stage 1

### Stage 0 — engineering smoke

```text
precisions: BF16, INT8, INT4
directed discovery examples: <=50, complete pairs
random directions: 1
bootstrap draws: 200
trace layers: 17,20,23,27
```

Proceed in ladder order. A failed INT4 condition is reported and not forced.

### Stage 1 — bounded pilot

Only after Stage 0 passes for a precision:

```text
directed discovery examples: <=150, complete pairs
random directions: 3
bootstrap draws: 500
trace layers: 17,20,23,27
```

This is exploratory discovery evidence. It cannot authorize thresholds or tune
the measurement definitions.

## 8. Interpretation and thresholds

No binary scientific preservation threshold is defined for Stage 1. Report raw,
standardized, and percentage changes from BF16 with pair-cluster intervals.
BF16 duplicate-forward numerical reproducibility is an engineering reference,
not an effect-size threshold inferred from INT8/INT4.

High-value signatures include:

```text
D preserved while Q, A, or G degrades
D and Q preserved while A and/or G degrades
D, Q, and A preserved while G degrades
```

Possible outcomes also include joint degradation, generic damage, or complete
preservation. Semantic-specific fragility requires an actionability change beyond
random-context changes and without catastrophic general-quality collapse.

## 9. Required artifacts

Each run writes raw evidence before aggregates under `runs/E14/`:

```text
config.resolved.yaml
manifest.json
status.json
probe_metrics.parquet
behavior_rows.parquet
factorial_rows.parquet
trace_rows.parquet
precision_metrics.json
E14_SUMMARY.md
```

The backend manifest records versions, compute dtype, quantization type,
per-module qtype/group size/kernel, exclusions, model/tokenizer revisions, prompt
and split hashes, seeds, wall time, peak VRAM, and project Git SHA.

The bounded cross-precision report is `E14_BOUNDED_PILOT_SUMMARY.md` and includes
D, B, Q, structured-minus-random A/G, general quality, fidelity, traces, runtime,
and VRAM.

## 10. Claim boundary and authorization

A positive bounded pilot would support only an exploratory statement under the
tested Qwen3-1.7B checkpoint, relation task, layer/site, and Quanto weight-only
scheme. It would not establish a universal quantization threshold or authorize
full E14 discovery.

Stage 2 full discovery is separately authorized only under the frozen full
protocol. Qwen3-0.6B, other models/quantizers, activation quantization, and any
lower-bit stress ladder remain unauthorized. E14 confirmation may be accessed
once only after the preregistered full-discovery gate passes.
