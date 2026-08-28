# E16 — Utilization Emergence During Training

Status: proposed, not authorized.

## 1. Question

> **Does a semantic representation become linearly available before the model learns to use it causally and behaviorally?**

This is the developmental version of the master question.

Instead of comparing two final checkpoints, E16 tracks the same model family through training checkpoints and asks when different links in the chain emerge:

```text
encoding -> propagation -> readout conversion -> behavior
```

## 2. High-value hypothesis

The key hypothesis is a temporal ordering:

```text
t_D < t_C <= t_B
```

where:

- `t_D` = training point where held-out decodability becomes reliably high;
- `t_C` = training point where source-free causal conversion becomes reliably high;
- `t_B` = training point where native behavior reaches the corresponding threshold.

Thresholds must be predeclared from validation/reference variability rather than selected after seeing trajectories.

## 3. Model/checkpoint requirements

Use a model family with accessible intermediate checkpoints and stable architecture/tokenizer across training.

Preferred properties:

- many checkpoints spanning early to late training;
- same hidden dimensionality and layer structure across checkpoints;
- public training-token metadata or reliable step ordering;
- sufficient model quality at intermediate stages to run the synthetic task;
- manageable size for repeated inference/intervention.

A Pythia-like checkpoint family is an example of the required structure, but the final model choice should be made only after checking checkpoint availability and architectural compatibility.

## 4. Primary experiment

For each frozen checkpoint `t`:

1. run the same grouped synthetic relation dataset and split protocol;
2. fit the standard train/validation truth probe;
3. measure held-out D;
4. measure native behavior/readout;
5. run source-free E01B setpoint interventions at the frozen relative or absolute layer/site protocol;
6. measure C and downstream propagation/readout conversion.

Do not reuse a probe trained on a future checkpoint as the primary D estimate for an earlier checkpoint.

A secondary cross-check may project a late-checkpoint direction into earlier checkpoints only if architecture/dimensions are identical and the analysis is explicitly labeled cross-check.

## 5. Layer/site alignment across training

Because architecture is fixed within a training series, use the same absolute layer where scientifically justified.

If the selected family has a different layer count from Qwen3, predeclare the site using a relative-depth mapping anchored by an initial discovery pilot, then freeze it before the main trajectory.

Do not choose the best layer independently at every training checkpoint; that would confound emergence with site selection.

## 6. Required trajectories

For every checkpoint record:

```text
training step / tokens seen
D AUROC
LOFO D if feasible on selected checkpoints
native behavior
native readout AUROC / margin discrimination
source-free causal effect C
validation-standardized Δm_z
validation-standardized Δq_z
late propagation retention
conversion slope diagnostic
```

Primary plots:

```text
D vs training progress
C vs training progress
B vs training progress
D/C/B overlaid after normalization
propagation/readout conversion over training
```

## 7. Emergence ordering

Define predeclared thresholds:

```text
D_threshold
C_threshold
B_threshold
```

Estimate the first checkpoint at which each threshold is crossed and remains crossed for at least the next `r` checkpoints, where `r` is predeclared to avoid transient spikes.

The high-value result is a stable lag:

```text
representation becomes available first
functional conversion emerges later
```

Report uncertainty due to finite checkpoint spacing; do not pretend the exact emergence step is known between saved checkpoints.

## 8. Main hypotheses

### H16.1 — representation-first emergence

D reaches its high-performance regime earlier than C.

### H16.2 — utilization learning

C can increase substantially across checkpoints after D has already plateaued.

### H16.3 — mechanism transition

The increase in C is accompanied primarily by stronger native-readout conversion and/or downstream propagation, allowing the developmental change to be localized.

### H16.4 — behavior linkage

Growth in C explains behavioral improvement beyond changes in D alone.

This can be assessed descriptively/regressively across checkpoints but must not be overclaimed causally from a small number of checkpoints.

## 9. Controls

Required:

- random-label probes at representative checkpoints;
- same surface/task distribution at every checkpoint;
- fixed tokenizer/prompt interface;
- source-free no-op and random/orthogonal intervention controls;
- exact checkpoint/revision logging;
- identical bootstrap unit and metric definitions;
- general language-model quality sanity metrics where available.

## 10. Falsification

The representation-first story weakens if:

- D and C emerge together within checkpoint resolution;
- D continues changing strongly throughout the same interval as C;
- the apparent lag is caused by a shifting optimal layer/site;
- early-checkpoint behavior is too degenerate for the intervention metric to be interpretable;
- random-feature controls reproduce the same D trajectory.

A null result is valuable because it would show the current Qwen checkpoint dissociation is not a generic training-stage phenomenon.

## 11. Staged execution

Stage 0: identify one checkpoint family and inspect 6–10 logarithmically spaced checkpoints.

Stage 1: CPU/config audit and tiny forward smoke at three checkpoints: early/mid/late.

Stage 2: bounded D/B/C pilot at 5 checkpoints.

Stage 3: only if a plausible lag exists, fill in intermediate checkpoints around the transition.

Stage 4: full trajectory with frozen metrics.

Do not start with dozens of checkpoints.

## 12. Claim boundary

A positive E16 would support that, in the tested training series and semantic task, representational accessibility precedes or saturates before causal utilization.

It would not establish a universal developmental law without replication across training runs/families.
