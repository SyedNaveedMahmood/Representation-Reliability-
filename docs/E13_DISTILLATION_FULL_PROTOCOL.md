# E13 Bounded Distillation Reliability Protocol

Status: **authorized implementation and one-seed bounded diagnostic only**.

Frozen before student optimization on 2026-08-28. E01 and E14 confirmation
holdouts are consumed and prohibited. The E13 confirmation namespace is defined
but must not be materialized in this bounded task.

## Scientific question and quantities

Does ordinary fine-tuning or logit distillation transfer causal organization
when the student already linearly represents the relation truth variable?

```text
B = native Yes-minus-No margin AUROC
D = layer-17 resid_post last_prompt decodability AUROC
Q = Y10 - Y00
A = Y01,matched - Y00
G = (Y11-Y10) - (Y01-Y00)
```

Primary comparisons are R0 frozen Qwen3-0.6B, R1 hard-label SFT, and R2 logit
KD from frozen Qwen3-1.7B. No hidden-state KD or conversion-response loss is
authorized. The teacher has no gradients.

## Fresh corpus

Use `generate_synthetic_relations` with all five relation families and 42
entities. Generate candidates independently, preserve whole counterfactual
pairs, deduplicate by the ordered pair of prompt strings across all open splits,
and select deterministically:

```text
train:            4,000 directed / 2,000 pairs, seed 20261301
validation:         500 directed /   250 pairs, seed 20261302
discovery eval:     300 directed /   150 pairs, seed 20261303
```

Candidate generation may oversample deterministically. Record candidate and
collision counts. Stop if the nonduplicative quota cannot be reached. Prefix
all IDs with their E13 namespace.

Locked, unmaterialized E13 confirmation specification:

```text
namespace: e13_confirmation_v1
seed: 20261304
200 directed / 100 matched pairs
n_entities: 42
same five families
new e13-confirmation-v1- sample/pair prefixes
```

Neither confirmation rows nor labels may be generated, loaded, or used in this
pilot.

## Training

Student update seed is `20261305`. Use full-parameter BF16 AdamW for R1/R2:

```text
optimizer: AdamW, betas (0.9,0.95), eps 1e-8, weight decay 0.01
peak learning rate: 2e-5
schedule: 10-step linear warmup then cosine decay
microbatch: 2
gradient accumulation: 4
effective batch: 8
optimizer steps: 100
gradient clipping: 1.0
checkpoint steps: 0,10,25,50,100
```

R1 minimizes full-vocabulary cross-entropy for the correct single-token ` Yes`
or ` No` completion at `last_prompt`. R2 minimizes

```text
0.5 * CE + 0.5 * T^2 * KL(teacher_T || student_T), T=2.0
```

over the full next-token vocabulary. The teacher forward is deterministic,
inference-only, and uses the same prompt batch. No outcome-based tuning or early
stopping is allowed. Basic nonfinite-loss/gradient instability stops a regime.

## Frozen evaluation

The initial student train/validation probe, validation class-median q targets,
and validation standardization scales are frozen for all student Q/A/G
measurements. Later weights cannot redefine the scientific intervention.
Checkpoint-native D may fit a new train-only scaler/probe and select C on
validation only. Also report D using the frozen initial-student probe.

Teacher reference uses its own train/validation-only probe and targets on the
same fresh open corpus. Evaluate all metrics on the 300 directed discovery rows.
Use the same layer-17 `resid_post` / `last_prompt` site, matched orthogonal
context, lambda 1, and deterministic random orthogonal seeds `2130,2131,2132`.
Report matched-minus-random A/G, raw Q, and pair-cluster 95% CIs with 500 draws.

## Interpretation and trigger

This is a one-seed pilot. It cannot establish a training law. Report which of
B/D/Q/A/G changes and whether R2 differs from R1. Teacher-gap closure ratios are
descriptive and omitted when denominators are near zero.

Conversion-response distillation is scientifically triggered only if standard
KD materially improves B, initial/frozen D is already high, and a substantial
teacher-student A or G gap remains. If triggered, design but do not run it.
