# E01B — Source-Free Semantic Setpoints and Orthogonal Context

Status: E01B-1 and E01B-2 full discovery complete.
Confirmation remains locked and was not accessed.

Depends on:

- E01A full discovery;
- E01A trace-mechanism analysis;
- frozen primary site `resid_post / layer 17 / last_prompt`.

## 1. Motivation

E01A showed that matched and shuffled opposite-label donors produce equivalent
coordinate-only effects. Trace analysis then showed that, after controlling
actual scalar coordinate displacement, matched-source indicators are near zero.
That result is expected because the E01A rank-one intervention discards every
source property except the scalar target coordinate.

E01B therefore separates two questions that E01A could not answer cleanly:

1. **Is a source example needed at all, or is the scalar semantic coordinate
   itself sufficient to modulate native behavior?**
2. **If the scalar coordinate is held fixed, can orthogonal representational
   context alter how strongly that coordinate is converted into behavior?**

E01B has two subexperiments. Run E01B-1 first. E01B-2 is conditional on a clean
E01B-1 result.

---

# E01B-1 — Source-Free Setpoint Causality

## 2. Primary question

At the frozen E01A site and probe direction, does directly setting the decoded
truth coordinate to validation-defined target values reproduce a signed,
monotonic native-readout response without using any donor activation?

## 3. Frozen representation

Use the same scientific identity as E01A:

```text
site = resid_post
layer = 17
token_selector = last_prompt
```

Use the same probe-training contract:

```text
fit scaler/probe on train
select C on validation
never fit on discovery_test
never touch confirmation
```

If the exact E01A probe artifact is reproducibly recoverable, reuse it. If the
project does not persist the fitted object, reconstruct it deterministically
from the same train/validation activations and verify direction/metrics against
E01A before proceeding.

Let the resulting unit truth direction be `u` and the scalar coordinate be:

```text
q = u^T h
```

## 4. Validation-only target construction

Target values must be estimated **only from validation**.

### Primary class setpoints

Define:

```text
q0* = median(q | y=0, validation)
q1* = median(q | y=1, validation)
```

For every discovery-test base sample, define the opposite-class target:

```text
if base gold label = 0: target = q1*
if base gold label = 1: target = q0*
```

The primary intervention is:

```text
delta = (q_target - q_base) * u
h' = h + delta
```

No donor/source example exists.

### Secondary continuous setpoint grid

Construct a label-agnostic dose-response grid from pooled validation
coordinates using fixed validation quantiles:

```text
Q05, Q25, Q50, Q75, Q95
```

Every discovery base receives every target value.

This tests whether output margin follows the scalar setpoint continuously rather
than only when jumping to an opposite-class median.

Do not select quantiles after looking at discovery results.

## 5. Conditions

Required conditions:

1. `source_free_opposite_class_median` — primary treatment;
2. `source_free_grid` — five validation-quantile targets;
3. `same_class_median` — set to the validation median of the base's own class;
4. `random_direction` — per-example norm matched to the semantic edit;
5. `orthogonal_random` — per-example norm matched and orthogonal to `u`;
6. `no_op` — target equals `q_base` exactly.

Use at least ten deterministic random and ten deterministic orthogonal-random
directions in full discovery. Smoke/pilot may use fewer.

## 6. Primary outcomes

### Opposite-class median treatment

Raw native outcome:

```text
m = logit(Yes) - logit(No)
```

Orient toward the target class so positive change means movement toward the
source-free opposite-class target.

Primary effect:

```text
Delta m_target = oriented(m_after) - oriented(m_before)
```

Report:

- mean and median `Delta m_target`;
- pair-cluster 95% CI;
- actual base-to-target flip rate;
- expected-target rate after intervention;
- standardized `Delta m_z` using a predeclared non-confirmation reference;
- `Delta q` and `Delta q_z`;
- exploratory `kappa = Delta m_z / |Delta q_z|` with zero guards.

### Continuous grid

For each base, fit/descriptively summarize the relationship between target
setpoint and native Yes-minus-No margin.

Required metrics:

- within-example monotonicity rate;
- pooled Spearman correlation between target `q` and output margin;
- mixed/cluster-aware slope of output margin on target `q`;
- saturation/nonlinearity diagnostics;
- target-level aggregate means and pair-cluster CIs.

Do not select a favorable subrange post hoc.

## 7. Falsification controls

The source-free scalar claim passes only if:

1. semantic setpoint effects exceed norm-matched random/orthogonal controls;
2. opposite-class medians reproduce the E01A effect direction;
3. the continuous grid shows coherent monotonic behavior;
4. alpha/setpoint fidelity is numerically exact within dtype-aware tolerance;
5. the result is not driven by one relation family;
6. confirmation remains untouched.

If source-free setpoints fail while donor-based E01A succeeds, the donor-derived
scalar was not sufficient and E01B-2 should be reconsidered before running.

## 8. Cross-check against E01A

For the same discovery examples, compare source-free opposite-class-median
responses to E01A matched/shuffled coordinate responses after matching actual
`Delta q` as closely as possible.

Question:

> Does output depend primarily on the requested coordinate displacement rather
> than on whether that displacement came from a donor or a fixed validation
> setpoint?

This is a sensitivity analysis, not a reason to retune targets.

---

# E01B-2 — Orthogonal-Context Modulation at Fixed Setpoint

## 9. Authorization gate

Do not run E01B-2 until E01B-1 passes engineering and semantic controls.

## 10. Primary question

At a fixed source-free semantic target `Delta q`, does adding structured
orthogonal representational context change the native behavioral effect?

This is the experiment that actually gives source/nuisance information a route
to matter.

## 11. Fixed scalar intervention

Use the same source-free opposite-class median target from E01B-1:

```text
delta_q = (q_target - q_base) * u
```

This scalar component is identical across all context conditions for a given
base example.

## 12. Orthogonal decomposition

For source activation `h_s` and base activation `h_b`:

```text
d = h_s - h_b
v_perp = d - (u^T d)u
```

Normalize every structured/random orthogonal context to a common per-example
reference norm before adding it. Primary reference:

```text
r_base = ||v_perp(matched twin)||
```

If the matched orthogonal norm is degenerate, use a predeclared fallback such
as the validation median nondegenerate matched norm and record the fallback.

The full edit is:

```text
delta = delta_q + lambda * v_perp_normalized
```

Because `v_perp` is orthogonal to `u`, the scalar truth setpoint remains fixed.
Assert this numerically after BF16/FP16 execution.

## 13. Context conditions

Required:

1. `coordinate_only` — no orthogonal context;
2. `matched_orthogonal` — nuisance-matched counterfactual twin;
3. `same_family_shuffled_orthogonal` — different pair, same relation family;
4. `different_family_shuffled_orthogonal` — different relation family;
5. `random_orthogonal` — deterministic random orthogonal vector;
6. `same_label_orthogonal` — different-pair same-label source, secondary control.

Primary context strength:

```text
lambda = 1
```

Secondary predeclared sensitivity:

```text
lambda = 0.5
```

Do not add a larger grid unless pilot shows numerical failure rather than a weak
effect.

## 14. Primary contrast

For each context condition:

```text
context_increment = Delta m(condition) - Delta m(coordinate_only)
```

Use pair-cluster bootstrap CIs.

The central question is whether structured orthogonal context changes the causal
conversion of an identical semantic setpoint.

## 15. Interpretations

### Outcome A — approximately one-dimensional

If matched/same-family/different-family orthogonal contexts add little beyond
coordinate-only and random orthogonal controls are also near zero:

> The measured causal effect is largely determined by a scalar semantic
> setpoint under this task/site.

### Outcome B — context gated

If structured orthogonal context changes the effect while random orthogonal
context does not:

> The scalar coordinate is actionable but its conversion is gated by broader
> representational context.

### Outcome C — nonspecific perturbation sensitivity

If random orthogonal context changes behavior comparably to structured context:

> The apparent context modulation is not semantically specific.

### Outcome D — source-family specificity

If matched or same-family context systematically differs from different-family
context at matched scalar target and norm:

> The same decoded semantic coordinate has context-dependent causal meaning.

This would be particularly relevant to Representation Reliability.

## 16. Downstream tracing

Trace the same layers used by E01A:

```text
17, 20, 23, 27
```

For both subexperiments save:

- truth-coordinate trajectory;
- native fixed-readout trajectory;
- standardized conversion trajectory;
- treatment-control contrasts by layer.

The goal is to determine whether source-free/context effects enter immediately
or arise downstream.

## 17. Models

Run the same two checkpoints first:

```text
Qwen3-0.6B
Qwen3-1.7B
```

Do not add another family until the mechanism is frozen.

## 18. Confirmation policy

All E01B work remains discovery-only.

Do not access the existing confirmation split during:

- target estimation;
- target-grid selection;
- context-source selection;
- lambda selection;
- metric selection;
- debugging;
- pilot analysis.

After E01B-1 and E01B-2 are frozen, the project may decide whether to spend the
untouched confirmation split on the final combined mechanism.

## 19. Required artifacts

At minimum:

```text
intervention_rows.parquet
trace_rows.parquet
aggregate_metrics.parquet
control_contrasts.parquet
setpoint_targets.json
source_context_plan.parquet   # E01B-2
manifest.json
status.json
E01B_SUMMARY.md
```

Raw evidence must be saved before summaries.

## 20. Execution status and boundary

E01B-1 full discovery completed on 2026-08-28 for Qwen3-0.6B and
Qwen3-1.7B. The source-free treatment reproduced E01A effects and beat both
norm-matched direction controls. See `E01B1_FULL_DISCOVERY_SUMMARY.md`.

E01B-2 full discovery completed on 2026-08-28 for Qwen3-0.6B and
Qwen3-1.7B. Opposite-label structured orthogonal contexts exceeded
coordinate-only and norm-matched random context at the fixed scalar target.
Matched and same-family effects exceeded different-family effects in the pooled
analysis, with relation-family heterogeneity. The readout effect was already
present at L17 while decoded q remained fixed, followed by downstream q
propagation differences. See `E01B2_FULL_DISCOVERY_SUMMARY.md`.

This supports operational context-gated utilization under the frozen
intervention, but does not distinguish multiplicative gating from an additive
causal signal in the orthogonal component. Do not access confirmation or begin
an application branch without separate authorization.
