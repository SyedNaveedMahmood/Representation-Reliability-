# E14 — Quantization Reliability: Does Compression Preserve Representation but Break Utilization?

Status: proposed, not authorized.

## 1. Question

> **Can post-training quantization preserve a highly decodable semantic representation while disproportionately degrading its downstream causal conversion into native behavior?**

This experiment tests whether representational integrity and functional integrity have different compression thresholds.

## 2. Scientific motivation

Current discovery shows near-matched semantic D across Qwen3-0.6B and Qwen3-1.7B but sharply different C/readout conversion. Quantization provides a qualitatively different perturbation: the same checkpoint is numerically compressed without changing its training objective or dataset.

If D survives while C degrades, this would support the broader thesis that utilization is more fragile than representation.

## 3. Model choice

Primary checkpoint:

```text
Qwen/Qwen3-1.7B
```

Reason: its E01A/E01B causal conversion is strong enough to reveal degradation before hitting a floor.

Secondary checkpoint, only if primary gates pass:

```text
Qwen/Qwen3-0.6B
```

Do not begin with a large model family sweep.

## 4. Quantization ladder

Use one reproducible PTQ implementation per ladder. Avoid mixing algorithms when interpreting bit-width effects.

Primary ladder:

```text
BF16 reference
INT8 / 8-bit weight-only or closest stable equivalent
INT4
```

Optional stress ladder after primary analysis:

```text
INT3
INT2
```

Only include a low-bit condition if the implementation is numerically stable and the model remains operational.

Record exact backend, grouping, zero-point/symmetry, calibration data, compute dtype, kernel, library versions, and whether embeddings/output head/norms remain unquantized.

## 5. Frozen semantic measurement

Reuse the frozen E01B source-free protocol after E01B is finalized:

```text
site = resid_post
layer = 17
selector = last_prompt
validation-defined setpoints
same semantic dataset/splits or a separately frozen E14 corpus
```

The key requirement is comparability across precision conditions.

Two analysis views are required.

### View A — precision-native probe

Fit the standard probe independently inside each precision condition using train/validation only.

Question: is the semantic variable still linearly decodable in the compressed model?

### View B — frozen BF16 semantic coordinate

Where dimensions/modules remain directly compatible, apply the BF16-fitted raw-space direction to each quantized checkpoint's dequantized/captured residual activations.

Question: does the original semantic axis survive compression geometrically?

Do not silently conflate the two views.

## 6. Primary quantities

For each precision condition measure:

- held-out D / AUROC;
- native behavioral accuracy / balanced accuracy;
- native Yes-minus-No margin discrimination;
- E01B source-free causal effect at the opposite-class median target;
- standardized continuous setpoint-response slope;
- downstream truth-signal retention at L20/L23/L27;
- downstream standardized native-margin response;
- random/orthogonal control effects;
- full inference quality sanity metrics such as perplexity or task loss if available.

## 7. Compression thresholds

Predeclare descriptive thresholds only after choosing tolerances from BF16 validation variability, not from quantized outcomes.

Define:

```text
b_D* = lowest precision that preserves D within the predeclared tolerance
b_C* = lowest precision that preserves causal conversion within tolerance
b_B* = lowest precision that preserves behavior within tolerance
```

The high-value signature is:

```text
b_C* > b_D*
```

in the sense that causal utilization fails at a less aggressive compression level than representation availability.

Because bit width is ordered inversely with information retained, report the threshold convention carefully rather than using ambiguous greater/less-than language in the paper.

## 8. Main hypotheses

### H14.1 — representational survival

Moderate quantization leaves held-out D nearly unchanged.

### H14.2 — utilization fragility

At least one quantization level preserves D but materially reduces source-free C / standardized native-margin response.

### H14.3 — localization

The C degradation can be localized to one or both of:

```text
reduced downstream propagation
reduced conversion/readout coupling conditional on surviving Δq
```

### H14.4 — semantic specificity

The degradation is larger for the semantic treatment than can be explained by generic numerical instability, as judged against no-op, random-direction, orthogonal-random, and general task-quality controls.

## 9. Falsification

The hypothesis weakens if:

- D and C degrade in lockstep at every precision;
- C appears preserved whenever D is preserved;
- semantic effects disappear only when general model quality catastrophically fails;
- random/orthogonal controls grow comparably to semantic effects under quantization;
- backend-specific artifacts dominate the result.

A null result is still useful: it would show that the causal semantic pathway is robust to the tested compression.

## 10. Important confounds

Quantization can change activation scales and native logit scales. Therefore:

- never compare raw coordinate magnitudes across precisions without standardization;
- preserve raw metrics but add validation-standardized metrics;
- report clean native-margin SD per precision;
- report activation-norm and intervention-norm ratios;
- keep candidate tokenization identical;
- verify the intervention hook acts on the intended dequantized runtime residual state;
- compare exact model revisions.

## 11. Minimum experiment

Stage 0: engineering feasibility on 1.7B BF16/INT8/INT4, <=50 examples.

Stage 1: bounded pilot on <=150 directed discovery examples.

Stage 2: full discovery only if:

```text
D remains measurable
intervention fidelity passes
no-op is stable
semantic controls remain valid
INT4 model quality is not catastrophically broken
```

Stage 3: optional INT3/INT2 stress study.

## 12. Artifacts

Per precision save:

```text
model/quantization manifest
probe metrics
behavior metrics
source-free intervention rows
trace rows
aggregate metrics
control contrasts
precision comparison table
```

Global analysis should produce:

```text
D vs bit width
C vs bit width
B vs bit width
propagation vs bit width
conversion conditional on Δq vs bit width
```

## 13. Claim boundary

A positive E14 would support:

> Under the tested model/task/PTQ scheme, semantic representation availability survives a compression regime in which its functional utilization is measurably degraded.

It would not establish that all quantization methods or all knowledge behave this way.
