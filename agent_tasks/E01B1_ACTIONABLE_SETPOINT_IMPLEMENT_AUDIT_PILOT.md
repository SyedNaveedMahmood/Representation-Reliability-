# Coding-Agent Task — E01B-1 Source-Free Actionable Setpoints

Required repository state:

`main` must contain commit `a62992ae2dad9494bfab119e87737c9aa7125816` or a later descendant. Do not require an exact HEAD because documentation/prompt commits may advance `main` without changing the frozen E01B-1 design.

## Mission

Implement and validate the immediate next experiment under the master question:

> **What makes an internal representation actionable?**

The experiment is **E01B-1: Source-Free Setpoint Causality**.

The scientific purpose is to determine whether the E01A causal effect can be reproduced by directly setting the decoded semantic coordinate to validation-defined scalar targets, with **no donor/source activation at all**.

Do not redesign the experiment.

Do not run full discovery.

Do not run E01B-2, E13, E14, E15, E16, or confirmation.

## 1. Pull and read

```powershell
git pull origin main
git status
git rev-parse HEAD
git rev-parse origin/main
git merge-base --is-ancestor a62992ae2dad9494bfab119e87737c9aa7125816 HEAD
```

The last command must succeed.

Read completely:

```text
research/MASTER_QUESTION_ACTIONABLE_REPRESENTATIONS.md
research/ACTIONABLE_REPRESENTATION_ROADMAP.md
research/REPRESENTATION_UTILIZATION_NOVELTY_PROGRAM.md
research/EXPERIMENT_REGISTRY.yaml

docs/E01B_EXPERIMENT_DESIGN.md
docs/E01A_CAUSAL_SPEC.md
E01A_FULL_DISCOVERY_SUMMARY.md
E01A_TRACE_MECHANISM_ANALYSIS.md

src/representation_reliability/runners/e01a.py
src/representation_reliability/runners/e01a_support.py
src/representation_reliability/interventions/truth_coordinate.py
src/representation_reliability/adapters/intervention.py
src/representation_reliability/metrics/causal.py
src/representation_reliability/probes/linear.py
src/representation_reliability/data/splits.py
src/representation_reliability/data/synthetic.py
src/representation_reliability/extraction/activations.py
src/representation_reliability/cli.py
src/representation_reliability/config/schema.py

tests/
```

## 2. Frozen scientific contract

Primary site:

```text
resid_post
layer 17
last_prompt
```

Models:

```text
Qwen/Qwen3-0.6B
Qwen/Qwen3-1.7B
```

Probe protocol:

```text
train: fit scaler/probe
validation: select C and define setpoints
discovery_test: causal evaluation
confirmation: inaccessible
```

The intervention is source-free:

```text
u = unit-normalized frozen raw truth-probe direction
q_base = u^T h_base
q_target = validation-defined scalar target

delta = (q_target - q_base) u
h_after = h_base + delta
```

Required identities, within dtype-aware tolerance:

```text
u^T h_after == q_target
P_perp_u(h_after) == P_perp_u(h_base)
```

Do not represent a target using a fake donor activation.

## 3. Validation-only target construction

From validation residuals only:

```text
q_i = u^T h_i
```

Primary class medians:

```text
q0_star = median(q | y=0, validation)
q1_star = median(q | y=1, validation)
```

For each discovery base:

```text
if gold == 0:
    opposite_target = q1_star
    same_target = q0_star
else:
    opposite_target = q0_star
    same_target = q1_star
```

Continuous pooled validation grid:

```text
Q05
Q25
Q50
Q75
Q95
```

Persist exact targets and validation reference statistics in a run artifact.

No discovery-test label may influence target construction.

No confirmation row/label may be loaded.

## 4. Required conditions

Implement:

```text
source_free_opposite_class_median
same_class_median
source_free_grid
random_direction
orthogonal_random
no_op
```

For `source_free_grid`, every base receives every target:

```text
Q05,Q25,Q50,Q75,Q95
```

Do not invent class-oriented accuracy for intermediate quantiles. Grid analysis is based on the signed native Yes-minus-No margin as a function of scalar target.

## 5. Controls

For each opposite-class semantic edit, random and orthogonal-random controls must be norm matched per example:

```text
||delta_control|| == ||delta_semantic||
```

The orthogonal direction must satisfy:

```text
abs(dot(v_orthogonal, u)) < tolerance
```

Randomness must be deterministic by explicit seed.

`no_op` must use zero delta and must reproduce an unhooked clean forward.

## 6. Primary outcome

Raw native answer margin:

```text
m = logit(Yes) - logit(No)
```

For the opposite-class target, orient margin toward that target.

Primary effect:

```text
delta_margin_toward_target = oriented(after) - oriented(before)
```

Report:

```text
mean
median
pair-cluster 95% CI
actual base-to-target flip rate
expected-target rate after
```

Bootstrap by `pair_id`, not directed example.

## 7. Validation-standardized quantities

From clean validation data only compute:

```text
sigma_q_validation
sigma_margin_validation
```

Then:

```text
Delta_q_z = (q_after - q_before) / sigma_q_validation
Delta_m_z = oriented_delta_margin / sigma_margin_validation
kappa_z = Delta_m_z / abs(Delta_q_z)
```

`kappa_z` is exploratory and undefined for zero edit.

Do not use discovery SD for the primary standardized metric.

## 8. Continuous setpoint-response

For each base, sort Q05..Q95 by exact `q_target` and compute:

```text
Spearman(q_target, margin_after)
monotonic nondecreasing indicator
```

Aggregate:

```text
median per-example Spearman
fraction positive
fraction >= 0.8
fraction monotonic nondecreasing
```

Also estimate a within-base centered target-to-margin slope with pair-cluster bootstrap CI.

Preserve raw target-level margins.

## 9. Downstream trace

Trace:

```text
17,20,23,27
```

Save for every intervention:

```text
clean truth coordinate
intervened truth coordinate
delta truth coordinate
clean native margin
intervened native margin
delta native margin
```

The purpose is to see whether source-free setpoints reproduce E01A's readout-dominated mixed bottleneck.

## 10. Required numerical gates

For semantic setpoint rows save/check:

```text
q_base
q_target
q_after
projection deviation
orthogonal-space deviation
activation norm
delta norm
delta/activation ratio
```

Gate failures:

```text
no-op changes selected logits
hook leakage
setpoint projection exceeds dtype-aware tolerance
orthogonal preservation exceeds tolerance
random/orthogonal norm mismatch
NaN/Inf in required fields
wrong token site
wrong candidate token IDs
```

Use the audited E01A BF16 path as the tolerance reference.

## 11. Implementation structure

Prefer reusable project-native modules, for example:

```text
src/representation_reliability/interventions/setpoint.py
src/representation_reliability/metrics/setpoint.py
src/representation_reliability/runners/e01b.py
src/representation_reliability/runners/e01b_support.py
configs/experiments/E01B_source_free_setpoints.yaml
```

Add a CLI command such as:

```text
e01b
```

Do not broadly refactor E01A unless required for safe reuse.

## 12. Required artifacts

Per run save at least:

```text
setpoint_targets.json
intervention_rows.parquet
trace_rows.parquet
aggregate_metrics.parquet
grid_metrics.parquet
control_contrasts.parquet
probe metrics
manifest.json
status.json
E01B_SUMMARY.md
```

Raw evidence before aggregates.

Complete runs must never be overwritten.

If safe resume is implemented, identity-validate every reusable shard/target definition.

## 13. Tests

Add deterministic tests for at least:

```text
source-free setpoint math
projection identity
orthogonal preservation
validation class medians
validation quantiles
confirmation exclusion
probe orientation
validation standardization
zero-variance guard
Yes-target orientation
No-target orientation
continuous grid ordering
within-base slope
pair-cluster bootstrap
random norm match
orthogonal direction check
artifact row counts/schema
```

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

and Ruff on modified files.

## 14. GPU contract validation

Before pilot verify both models for:

```text
last_prompt indexing
right padding
batch row identity
hook removal
no-op logits
setpoint projection
orthogonal preservation
finite outputs
candidate token IDs
downstream capture
```

## 15. Smoke

Run 0.6B first, then 1.7B only if clean.

Target smoke budget:

```text
<=25 matched pairs
<=50 directed examples
all semantic target types
1 random direction
1 orthogonal direction
trace 17/20/23/27
~200 bootstrap draws
```

If 0.6B smoke fails, fix and rerun before touching 1.7B.

## 16. Pilot

After both smokes pass:

```text
<=75 matched pairs
<=150 directed examples
3 random directions
3 orthogonal directions
~500 bootstrap draws
trace 17/20/23/27
```

Run:

```text
0.6B pilot
1.7B pilot
```

Do not tune class medians, quantile grid, layer, controls, or metrics based on pilot outcomes.

## 17. Pilot scientific sanity

A positive pilot should approximately show:

1. opposite-class source-free setpoints move output in the E01A-expected direction;
2. semantic treatment is not explained by random/orthogonal controls;
3. Q05..Q95 produce a coherent target-to-margin relationship;
4. 1.7B remains more conversion-sensitive than 0.6B;
5. donor activation is unnecessary for the effect.

A negative result is allowed. Do not redesign simply to force these outcomes.

## 18. Allowed bug fixes

Fix only demonstrated issues such as:

```text
API/import mismatch
tensor/device/dtype bug
token-index bug
probe reconstruction mismatch
validation target leakage
sign/orientation bug
standardization bug
control norm mismatch
orthogonalization bug
hook leakage
trace corruption
resume/artifact bug
CLI wiring bug
```

For every fix: reproduce, explain scientific impact, add regression test, patch minimally, rerun affected validation.

## 19. Forbidden scope

Do not:

```text
run E01B-1 full discovery
run E01B-2
open confirmation
implement/run E13 distillation
implement/run E14 quantization
implement/run E15 long horizon
implement/run E16 training checkpoints
start E02
```

The new master roadmap is design context, not authorization.

## 20. Final response format

Return:

### Audit verdict

One of:

```text
implementation valid without fixes
implementation valid after bounded fixes
engineering blocker
methodological blocker
source-free hypothesis falsified in pilot
```

### Tests

```text
pytest:
ruff:
GPU contracts:
```

### Validation targets

For each model:

```text
q0_star
q1_star
Q05
Q25
Q50
Q75
Q95
sigma_q_validation
sigma_margin_validation
```

### 0.6B smoke/pilot

Include:

```text
run_dir
status
n pairs/examples
no-op deviation
projection/orthogonal fidelity
control norm matching
opposite-class effect + CI
same-class effect
random/orthogonal effects
grid margins
median within-base Spearman
fraction monotonic
within-base slope + CI
Delta_q_z
Delta_m_z
kappa_z
L17/L20/L23/L27 Delta_m_z
runtime
peak VRAM
```

### 1.7B smoke/pilot

Same fields.

### Cross-scale pilot-only interpretation

Answer:

1. Does source-free targeting reproduce E01A directionality?
2. Does it beat random/orthogonal controls?
3. Is target-q to output-margin response monotonic?
4. Is 1.7B still more conversion-sensitive?
5. Is donor identity unnecessary under this intervention?
6. Any falsification concern?

### Repository

```text
HEAD
origin/main
working tree
```

### Exact full-discovery commands

Return exact valid PowerShell commands for 0.6B and 1.7B E01B-1 full discovery using all discovery pairs, 10 random directions, 10 orthogonal directions, 2,000 pair-cluster bootstrap draws, and trace layers 17/20/23/27.

Do not execute them.

## Stop condition

Stop after implementation, tests, smoke, pilot, push, and full-command handoff.
