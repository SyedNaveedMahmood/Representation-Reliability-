# E01B-2 Full Discovery Summary

Status date: 2026-08-28. Full discovery is complete for Qwen3-0.6B and
Qwen3-1.7B. Confirmation remains locked and was not accessed.

## Question and claim boundary

E01B-2 asks whether orthogonal representational context changes native behavior
when the decoded truth coordinate is held at the exact validation-defined
opposite-class setpoint from E01B-1. The frozen site is
`resid_post / layer 17 / last_prompt`.

The primary estimand is the target-oriented margin effect for a context
condition minus the effect of `coordinate_only` for the same base. Structured
contexts are per-example norm matched to the matched-twin orthogonal norm and
are compared with random orthogonal context at the same scalar target and norm.

This design establishes context sensitivity under the intervention. It does
not by itself distinguish a multiplicative utilization interaction from an
additive causal signal carried by the structured orthogonal component. It also
does not establish endogenous natural use, universality across tasks/sites/model
families, or confirmation-level evidence.

## Completed runs

| Model | Run | Model revision | Directed examples | Pairs | Raw rows | Trace rows |
|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | `E01B2_f2d75dab1eba` | `c1899de289a04d12100db370d81485cdf75e47ca` | 300 | 150 | 8,700 | 34,800 |
| Qwen3-1.7B | `E01B2_e2b4b02cb3a4` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 300 | 150 | 8,700 | 34,800 |

Both runs used project commit
`0753cfb2dd6fbe001374ba5e66ca89a67ff089de`, context strengths 0.5 and 1.0,
ten deterministic random orthogonal directions, 2,000 pair-cluster bootstrap
draws, and trace layers 17/20/23/27. Yes and No were distinct single tokenizer
tokens (`7414` and `2308`). The scalar targets and standardization scales came
from the same 300 validation examples used by E01B-1.

## Contract and integrity checks

| Check | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| coordinate-only target deviation from E01B-1 | 0 | 0 |
| coordinate-only output-margin deviation from E01B-1 | 0 | 0 |
| maximum setpoint projection deviation | 0.026315 | 0.121055 |
| maximum projection deviation / validation sigma q | 0.030240 | 0.012468 |
| maximum context dot truth direction | 4.44e-16 | 1.55e-15 |
| maximum context-norm relative mismatch | 3.45e-16 | 2.91e-16 |
| maximum total-decomposition error | 2.22e-16 | 1.78e-15 |
| maximum target-state relative L2 deviation | 0.002176 | 0.002274 |
| post-forward hook-leakage deviation | 0 | 0 |
| degenerate-reference fallback rows | 0 | 0 |
| required-field NaN/Inf check | pass | pass |
| source-selection and trace-completeness gates | pass | pass |
| confirmation accessed | no | no |

Every non-coordinate context used the same per-example reference norm. At
lambda 1, mean total-edit/base norms were 0.1234 for 0.6B and 0.1829 for 1.7B;
random and structured contexts were norm matched within each example.

## Primary lambda-1 result

Effects are target-oriented changes in native Yes-minus-No margin. Context
increments subtract the paired `coordinate_only` effect.

### Qwen3-0.6B

| Condition | Mean effect | Median effect | Context increment | Pair-cluster 95% CI |
|---|---:|---:|---:|---:|
| coordinate only | 0.026250 | 0 | 0 | [0, 0] |
| matched orthogonal | 1.153333 | 0.875000 | 1.127083 | [0.954990, 1.292510] |
| same-family shuffled orthogonal | 0.890417 | 0.687500 | 0.864167 | [0.722917, 1.003333] |
| different-family shuffled orthogonal | 0.646250 | 0.500000 | 0.620000 | [0.519146, 0.728333] |
| same-label orthogonal | 0.131667 | 0 | 0.105417 | [0.009563, 0.206687] |
| random orthogonal | 0.006229 | 0 | -0.020021 | [-0.027896, -0.012167] |

### Qwen3-1.7B

| Condition | Mean effect | Median effect | Context increment | Pair-cluster 95% CI |
|---|---:|---:|---:|---:|
| coordinate only | 0.666042 | 0.750000 | 0 | [0, 0] |
| matched orthogonal | 4.278958 | 4.375000 | 3.612917 | [3.275620, 3.946745] |
| same-family shuffled orthogonal | 3.746458 | 3.750000 | 3.080417 | [2.776625, 3.364437] |
| different-family shuffled orthogonal | 3.149167 | 3.000000 | 2.483125 | [2.296443, 2.680630] |
| same-label orthogonal | 0.712917 | 0.750000 | 0.046875 | [-0.107724, 0.203000] |
| random orthogonal | 0.729896 | 0.750000 | 0.063854 | [0.044643, 0.082797] |

The random increment is statistically nonzero and scale dependent, but it is
small relative to the opposite-label structured increments. Random context is
not approximately equivalent to matched, same-family, or different-family
context in either model.

Target-flip rates at lambda 1 were 0.170/0.160/0.117 for matched/same-family/
different-family in 0.6B and 0.050/0.040/0.040 in 1.7B. The margin effect,
rather than the thresholded flip rate, remains the predeclared primary outcome.

## Lambda-0.5 sensitivity

| Model | Condition | Mean effect | Context increment | Pair-cluster 95% CI |
|---|---|---:|---:|---:|
| 0.6B | matched | 0.587500 | 0.561250 | [0.479573, 0.652104] |
| 0.6B | same-family | 0.446667 | 0.420417 | [0.354583, 0.487917] |
| 0.6B | different-family | 0.330000 | 0.303750 | [0.254583, 0.355021] |
| 0.6B | same-label | 0.066667 | 0.040417 | [-0.008344, 0.090010] |
| 0.6B | random | 0.019167 | -0.007083 | [-0.014377, 0.000210] |
| 1.7B | matched | 2.409167 | 1.743125 | [1.568490, 1.916458] |
| 1.7B | same-family | 2.130208 | 1.464167 | [1.321641, 1.594792] |
| 1.7B | different-family | 1.839792 | 1.173750 | [1.093536, 1.256250] |
| 1.7B | same-label | 0.678542 | 0.012500 | [-0.066047, 0.090651] |
| 1.7B | random | 0.704229 | 0.038188 | [0.025186, 0.051146] |

The structured increments approximately scale with context strength without a
sign reversal. The frozen sensitivity therefore supports the primary result.

## Structured-control contrasts at lambda 1

| Contrast | Qwen3-0.6B difference (95% CI) | Qwen3-1.7B difference (95% CI) |
|---|---:|---:|
| matched - random | 1.147104 `[0.973644, 1.324955]` | 3.549063 `[3.213165, 3.877772]` |
| same-family - random | 0.884187 `[0.744785, 1.027449]` | 3.016562 `[2.727657, 3.303480]` |
| different-family - random | 0.640021 `[0.542930, 0.744376]` | 2.419271 `[2.236742, 2.597339]` |
| same-label - random | 0.125438 `[0.025403, 0.221196]` | -0.016979 `[-0.166863, 0.138157]` |
| matched - different-family | 0.507083 `[0.391615, 0.622510]` | 1.129792 `[0.873224, 1.392172]` |
| same-family - different-family | 0.244167 `[0.141229, 0.349594]` | 0.597292 `[0.366443, 0.813974]` |

Matched and same-family contexts exceed different-family context in the pooled
analysis for both checkpoints. This supports aggregate relation-family gating,
but the family-stratified analysis below prevents a universal ordering claim.

## Downstream context trajectory

Values are validation-standardized mean context increments relative to
coordinate-only, formatted as `delta q z / delta native-margin z` at lambda 1.

### Qwen3-0.6B

| Layer | Matched | Same-family | Different-family | Same-label | Random |
|---:|---:|---:|---:|---:|---:|
| 17 | -0.000 / 0.639 | 0.000 / 0.509 | 0.000 / 0.314 | 0.001 / 0.048 | 0.001 / 0.004 |
| 20 | 0.522 / 0.893 | 0.395 / 0.678 | 0.283 / 0.445 | 0.075 / 0.060 | -0.001 / -0.005 |
| 23 | 0.719 / 0.928 | 0.552 / 0.704 | 0.395 / 0.460 | 0.089 / 0.060 | -0.003 / -0.014 |
| 27 | 0.899 / 0.815 | 0.697 / 0.625 | 0.490 / 0.448 | 0.111 / 0.076 | 0.009 / -0.014 |

### Qwen3-1.7B

| Layer | Matched | Same-family | Different-family | Same-label | Random |
|---:|---:|---:|---:|---:|---:|
| 17 | 0.000 / 1.042 | -0.000 / 0.878 | -0.000 / 0.748 | 0.000 / 0.010 | 0.000 / 0.009 |
| 20 | 0.467 / 1.108 | 0.404 / 0.942 | 0.337 / 0.768 | 0.019 / 0.005 | 0.011 / 0.046 |
| 23 | 0.630 / 1.158 | 0.547 / 0.992 | 0.455 / 0.795 | 0.016 / 0.017 | 0.012 / 0.032 |
| 27 | 0.875 / 1.156 | 0.752 / 0.987 | 0.631 / 0.795 | 0.032 / 0.016 | 0.035 / 0.020 |

At the intervention layer, structured contexts change the native fixed readout
while the decoded scalar displacement remains unchanged. Structured contexts
then change downstream decoded-coordinate propagation. The strongest
localization is therefore immediate context-sensitive readout, followed by a
secondary propagation difference.

## Relation-family sensitivity

Lambda-1 context increments by family:

| Model | Family | Matched | Same-family | Different-family | Random |
|---|---|---:|---:|---:|---:|
| 0.6B | above_below | 1.528 | 1.241 | 0.853 | -0.028 |
| 0.6B | before_after | 0.821 | 0.634 | 0.460 | -0.022 |
| 0.6B | east_west | 0.619 | 0.433 | 0.456 | 0.001 |
| 0.6B | larger_smaller | 1.913 | 1.540 | 0.915 | -0.028 |
| 0.6B | north_south | 0.787 | 0.508 | 0.432 | -0.024 |
| 1.7B | above_below | 4.708 | 4.064 | 2.923 | 0.193 |
| 1.7B | before_after | 4.084 | 3.427 | 2.171 | 0.096 |
| 1.7B | east_west | 1.786 | 1.343 | 1.970 | -0.016 |
| 1.7B | larger_smaller | 5.977 | 5.319 | 3.593 | 0.080 |
| 1.7B | north_south | 1.762 | 1.471 | 1.814 | -0.019 |

All opposite-label structured effects remain positive, but the pooled
matched/same-family ordering reverses for east-west and north-south in 1.7B,
and same-family is slightly below different-family for east-west in 0.6B.
Relation-family specificity is therefore supported in aggregate but
heterogeneous in detail; no family was selected or removed.

## Cross-scale and mechanistic interpretation

At lambda 1, the 1.7B context increment is 3.21 times the 0.6B increment for
matched context, 3.56 times for same-family context, and 4.01 times for
different-family context. The scale difference therefore persists after the
scalar semantic displacement and orthogonal-context norm are controlled.

The supported discovery classification is:

```text
structured context gated
aggregate relation-family gated, with family-level heterogeneity
not approximately one-dimensional
not explained by nonspecific random perturbation sensitivity
```

The strongest supported statement is that native behavior under the frozen
source-free truth setpoint depends strongly on structured orthogonal state, and
that this dependence is larger in Qwen3-1.7B. At L17 the decoded q target is
unchanged while native readout changes, so a scalar coordinate alone is not a
complete causal description of the manipulated state.

Because there is no context-only factorial arm in E01B-2, “context-gated” is an
operational description: the experiment does not yet identify whether the
orthogonal component modulates conversion of q or contributes an additional
causal truth signal independently. Confirmation and a future factorial test
must preserve this distinction.

## Next boundary

Freeze the E01B-2 site, targets, context construction, lambdas, controls,
metrics, and primary structured-vs-random contrasts before deciding whether to
authorize confirmation. Do not access confirmation or begin an application
branch merely because full discovery succeeded.
