# Representation Utilization Research Program

Status date: 2026-08-28

This document freezes the next research program after E01A full discovery and
its trace-mechanism analysis. Confirmation remains locked.

## 1. Central claim to protect

The project should no longer be framed around the generic statement that
"decodability does not imply behavior." That space is crowded.

The current discovery-supported claim is narrower and more mechanistic:

> **Semantic representation formation and representation-to-behavior
> conversion are separable. Across two Qwen3 checkpoints with nearly identical
> near-ceiling decodability, direct manipulation of the decoded coordinate
> produces sharply different native-readout effects. Trace analysis localizes
> the difference primarily to readout conversion from the intervention layer
> onward, with an additional late propagation disadvantage in the smaller
> checkpoint.**

This remains checkpoint-, task-, site-, and discovery-specific. It is not yet a
scaling law, and it does not establish that the unperturbed model naturally
uses exactly the probe-defined one-dimensional coordinate.

## 2. Current evidence chain

The paper spine should preserve this sequence:

1. **Encoded:** truth is near-perfectly linearly decodable in both checkpoints.
2. **Learned/general:** random-init controls are near chance and LOFO remains high.
3. **Poorly expressed:** behavior/native readout can lag far behind D.
4. **Causally actionable:** E01A coordinate interventions beat random,
   orthogonal, and same-label controls with signed dose response.
5. **Checkpoint-dependent conversion:** 1.7B converts a comparable early
   standardized coordinate perturbation into much larger native-readout change.
6. **Mixed bottleneck:** early readout conversion dominates the difference;
   later layers add a propagation disadvantage in 0.6B.

The next experiments should strengthen this exact chain rather than expand into
unrelated reliability dimensions.

## 3. Novelty boundary from the targeted literature check

Targeted searches on 2026-08-28 found close work in three neighboring areas:

- decodable information can be weakly actionable or steerable;
- representation geometry/readout alignment can change with checkpoint/scale;
- knowledge distillation can align logits, hidden states, or layerwise
  representation trajectories;
- RAG work already studies an "integration bottleneck," internal
  representations of retrieved evidence, knowledge conflicts, and causal
  routing/tracing.

Representative close work includes:

- Dasgupta & Cohn, *Improving Language Model Distillation through Hidden State
  Matching*, ICLR 2025;
- Koo et al., *SWITCH: Studying with Teacher for Knowledge Distillation of
  Large Language Models*, Findings of NAACL 2025;
- Tiapkin et al., *On Teacher Hacking in Language Model Distillation*, ICML
  2025;
- Chi et al., *MTA: Multi-Granular Trajectory Alignment for Large Language
  Model Distillation*, ACL 2026;
- Yeh & Li, *How Retrieved Context Shapes Internal Representations in RAG*,
  Findings of ACL 2026;
- Zhao et al., *Guaranteeing Knowledge Integration with Joint Decoding for
  Retrieval-Augmented Generation*, ACL 2026;
- Guo et al., *Why Retrieval-Augmented Generation Fails: A Graph Perspective*,
  arXiv:2605.14192;
- Ma et al., *CoRect: Context-Aware Logit Contrast for Hidden State
  Rectification to Resolve Knowledge Conflicts*, arXiv:2602.08221.

The targeted search did **not** identify a direct collision that combines all
of the following:

1. near-matched semantic decodability across checkpoints;
2. direct causal manipulation of the same decoded semantic variable;
3. layerwise localization of a representation-to-readout conversion gap; and
4. knowledge distillation analyzed in terms of whether representation and
   causal utilization transfer together or separately.

This is not proof that no such paper exists. Continue a narrow collision search
in parallel, but do not delay the experimental program waiting for exhaustive
absence proof.

## 4. Program architecture

The program has one core mechanism track and one high-upside novelty track.
A RAG application is retained only as a later backup/application test because
that literature is substantially more crowded.

### Track A — finish the mechanism

**E01B: Source-free setpoints and orthogonal-context modulation**

Goal: determine whether the causal object is approximately a scalar semantic
set-point and whether orthogonal representational context gates its effect.

This is the immediate next experiment.

### Track B — test transfer under distillation

**E13: Distillation Reliability Transfer**

Core question:

> When a smaller student already contains the semantic representation, does
> knowledge distillation transfer the teacher's ability to *use* that
> representation, or can representation and causal conversion remain
> dissociated after distillation?

This converts the current mechanistic finding into a training/compression
question with direct practical relevance.

The high-value phenomenon is a **distillation utilization gap**:

```text
teacher:  representation strong -> conversion strong
student:  representation strong -> conversion weak
              |
              | distillation
              v
student': does D change, C change, both, or neither?
```

The strongest possible discovery would be that standard representation-matching
or hidden-state KD makes the student look more teacher-like internally without
proportionally transferring causal conversion. That would show that
"representation transfer" and "functional deployment transfer" are distinct
objectives.

### Track C — later application branch: RAG

RAG is scientifically relevant because it contains a natural version of the
same problem: retrieved evidence can be available internally yet fail to govern
output. However, 2026 literature already directly studies retrieved-context
representations, integration bottlenecks, knowledge conflicts, and causal
routing.

Therefore RAG is **not** the immediate novelty bet. If used later, the question
should be specifically whether the project's D/C conversion metrics predict or
explain evidence-integration failures beyond existing RAG diagnostics.

Do not add a RAG experiment merely to claim application breadth.

## 5. Track A decision tree

### E01B-1 source-free setpoints

If source-free targets reproduce a monotonic causal response, the donor sample
is unnecessary and the semantic coordinate itself becomes the cleaner causal
object.

### E01B-2 orthogonal-context modulation

At a fixed scalar coordinate target, add controlled orthogonal context.

Possible outcomes:

- **context-insensitive:** coordinate-only and context-added effects are close;
  the measured causal effect is approximately one-dimensional;
- **context-gated:** matched or structured orthogonal context changes the effect;
  the scalar coordinate is actionable but not sufficient;
- **random-context-sensitive:** arbitrary orthogonal additions have comparable
  effects; the intervention is not semantically specific enough.

Only after E01B-1 and E01B-2 are frozen should the project spend the untouched
confirmation split.

## 6. Track B decision tree: distillation

E13 begins as a diagnostic study, **not** as a new distillation method.

Compare a fixed 1.7B teacher and 0.6B student under:

1. no additional training;
2. hard-label supervised fine-tuning;
3. standard logit KD;
4. hidden-state/trajectory KD;
5. optional combined logit + representation KD.

Measure throughout training:

- behavioral performance B;
- decodability D;
- native readout alignment/readout AUROC;
- source-free causal conversion C using the frozen E01B protocol;
- layerwise propagation and standardized native-margin response.

Primary diagnostic questions:

1. Does D remain near ceiling while C improves under KD?
2. Does hidden-state KD improve representational similarity more than causal
   conversion?
3. Does logit KD improve behavior/readout without materially changing D?
4. Do different KD objectives move different links in the
   `encoding -> propagation -> readout conversion -> behavior` chain?

### Method trigger, not precommitted method

Only if standard KD leaves a reproducible teacher-student conversion gap should
we implement a **causal-response / conversion-response distillation objective**.

The provisional idea is to cache the teacher's normalized native-margin
response to source-free coordinate perturbations and train the student to match
that response function, rather than only matching logits or hidden states.

For standardized setpoint perturbation `delta z`:

```text
L_conversion = mean_delta_z (
    Delta_m_student_z(delta z) - Delta_m_teacher_z(delta z)
)^2
```

This deliberately avoids cross-model hidden-dimension alignment. It distills a
causal response curve.

Do **not** implement this objective unless E13 diagnostic gates show that
standard objectives leave a meaningful conversion gap.

## 7. Why distillation is the preferred novelty extension

The current student-like 0.6B checkpoint already provides the unusual setup we
need: semantic D is near ceiling while causal conversion is weak. Therefore a
teacher can potentially transfer *use* of an already-present representation.

That lets the project ask a question standard KD evaluation usually cannot:

> Is the student missing knowledge, or is it missing the machinery that turns
> already-present knowledge into behavior?

This is tightly connected to the central paper claim rather than being a bolt-on
application.

## 8. Falsification gates

### E01B falsification

The scalar-coordinate story weakens if:

- source-free setpoints do not reproduce E01A directional effects;
- random/orthogonal controls are comparable to semantic setpoints;
- effects are unstable to modest target grids;
- orthogonal random context dominates structured context.

### E13 falsification

The distillation-utilization story weakens if:

- student D and C always move together across all KD regimes;
- standard hidden-state KD transfers C as fully as teacher behavior with no
  dissociation;
- the apparent KD effect is explained solely by general task fine-tuning;
- conversion metrics do not predict any teacher-student difference beyond
  ordinary accuracy/logit loss.

A negative E13 is still informative but should not become the paper headline.

## 9. Compute strategy

Immediate E01B should remain cheap and use the existing intervention harness.

E13 should cache frozen-teacher quantities whenever possible. Teacher gradients
are never needed. A 0.6B student can be fully fine-tuned with BF16,
gradient-checkpointing, and small microbatches on a high-memory single GPU; the
teacher can be evaluated offline or in inference-only passes. Do not begin with
multi-model/multi-task scaling.

Use staged gates:

1. smoke on a tiny subset;
2. one-seed pilot;
3. multi-seed synthetic discovery;
4. only then one second task/domain;
5. only then consider a conversion-aware distillation method.

## 10. Publication narrative if the program succeeds

The strongest final narrative would be:

> LLMs can learn a semantic representation before they learn to deploy it
> reliably. We show near-identical semantic decodability across checkpoints but
> sharply different causal conversion into native readout. The difference is
> localized primarily to readout coupling, with additional late propagation
> loss. Source-free setpoints establish that the decoded semantic coordinate can
> act as a causal control variable, while orthogonal-context tests determine
> whether that control is approximately one-dimensional. Finally, knowledge
> distillation reveals whether functional deployment transfers together with
> representation, exposing a distinction between distilling *what a model
> represents* and distilling *how it uses that representation*.

This is the research program to optimize for. Do not dilute it with unrelated
experiments unless the core gates fail.
