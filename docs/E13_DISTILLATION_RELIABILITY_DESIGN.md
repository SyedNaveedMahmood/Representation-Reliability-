# E13 — Distillation Reliability Transfer

Status: proposed novelty extension, not authorized for execution.

This design is intentionally downstream of E01B. The project should first
finish the causal object it is trying to measure before asking whether
knowledge distillation transfers it.

## 1. Core question

> When a student already contains a highly decodable semantic representation,
> does knowledge distillation transfer the teacher's ability to *use* that
> representation, or can representation and representation-to-behavior
> conversion remain dissociated after distillation?

This is different from ordinary KD evaluation. Standard KD typically asks
whether the student matches teacher outputs, hidden states, task accuracy, or
representation geometry. E13 asks whether the teacher's **causal utilization of
an internal semantic variable** is transferred.

## 2. Why the current system is unusually suitable

Current discovery evidence already provides a controlled teacher/student-like
contrast:

```text
Qwen3-0.6B: D approximately saturated, causal conversion weak
Qwen3-1.7B: D approximately saturated, causal conversion much stronger
```

Thus the smaller model is not obviously missing the representation. It may be
missing the machinery that turns that representation into behavior.

This creates a clean training question:

> Can distillation improve functional deployment without needing to create the
> representation from scratch?

## 3. Novelty hypothesis

Primary hypothesis:

> **Distillation can transfer representation and utilization at different
> rates, and standard representation-matching objectives may overstate
> successful knowledge transfer when causal conversion remains student-like.**

A stronger possible finding would be:

> Hidden-state/trajectory KD makes student representations more teacher-like
> while leaving a measurable teacher-student causal-conversion gap.

A complementary finding would be:

> Logit KD improves native readout/behavior while D is already near ceiling,
> demonstrating that KD can repair utilization rather than add decodable
> information.

## 4. Literature boundary

Nearby work includes:

- ICLR 2025 hidden-state matching via representation similarity;
- 2025–2026 logit and sequence-level LLM KD methods;
- ACL 2026 trajectory-alignment KD;
- work on teacher hacking and distribution-transfer limitations;
- 2026 preprints arguing that distillation can preserve representational
  geometry.

These works motivate strong controls but do not, from the targeted 2026-08-28
search, directly measure a teacher-student gap in **causal conversion of an
already-decodable semantic representation**.

Continue a narrow literature collision search before paper submission.

## 5. Experimental stages

E13 has four gated stages.

### E13-A — baseline distillation diagnostic

Compare training regimes without introducing a new method.

### E13-B — training-trajectory analysis

Measure how D, readout, causal conversion, and behavior evolve during training.

### E13-C — second-task replication

Only after a robust synthetic result, repeat on one more natural or controlled
semantic task.

### E13-D — optional conversion-response distillation

Only if standard KD leaves a reproducible causal-conversion gap.

Do not jump directly to E13-D.

---

# E13-A — Baseline Distillation Diagnostic

## 6. Teacher and student

Primary pair:

```text
teacher: Qwen3-1.7B
student: Qwen3-0.6B
```

Teacher is always frozen and inference-only.

The student should use full-parameter BF16 fine-tuning for the primary
scientific comparison if hardware permits. LoRA may be retained as a compute
sensitivity analysis but should not be the only training mechanism because an
adapter-only result is weaker evidence about distillation of internal
utilization.

## 7. Data

Do not train on E01 confirmation.

Preferred design: generate a **fresh distillation discovery corpus** using the
same relational semantics but new generator seeds and disjoint surface forms.
Maintain grouped train/validation/discovery-test splits and a new untouched
E13-specific confirmation split.

The existing E01 confirmation split remains untouched.

The fresh corpus should preserve counterfactual-pair structure so the full
representation-reliability measurement stack can be reused.

Recommended initial scale:

```text
train: 8k–20k directed examples
validation: 1k–2k
discovery_test: 1k–2k
E13 confirmation: reserved
```

A smaller pilot corpus is allowed for engineering validation.

## 8. Training regimes

Required baselines:

### R0 — frozen student

No training.

### R1 — hard-label SFT

Standard supervised next-token/answer loss on ground-truth targets.

Purpose: distinguish generic task fine-tuning from teacher-specific transfer.

### R2 — logit KD

Teacher-student KL divergence with temperature plus a standard supervised term.

Primary form:

```text
L = lambda_ce * L_CE + lambda_kd * T^2 * KL(p_teacher^T || p_student^T)
```

Use the full relevant next-token distribution when feasible. If engineering
constraints require top-k approximation, predeclare the approximation and
validate that it preserves teacher Yes/No margins.

### R3 — hidden-state / representation KD

Align teacher/student hidden-state structure at predeclared mapped layers.
Because hidden dimensions differ, use a standard, separately justified
alignment mechanism rather than inventing a bespoke method solely for this
study.

Acceptable candidates include:

- learned linear projection plus normalized hidden-state loss;
- CKA/Gram-structure matching;
- a small trajectory-alignment objective inspired by existing KD literature.

Pick **one** primary representation-KD baseline before full discovery.

### R4 — combined logit + representation KD

Optional but recommended if compute allows.

Purpose: determine whether matching outputs and internal geometry jointly closes
the utilization gap.

## 9. Hyperparameter discipline

Tune training hyperparameters on validation only.

Use the same:

- optimizer family;
- total training-token budget;
- batch-equivalent budget;
- checkpoint schedule;
- evaluation cadence;

across comparable regimes unless the objective mathematically requires a
change.

Report total student update FLOPs/time and teacher inference overhead.

## 10. Training checkpoints

Save enough checkpoints to resolve trajectory ordering.

Recommended normalized progress:

```text
0%, 5%, 10%, 25%, 50%, 75%, 100%
```

If the run is very short, use a fixed step schedule that approximates these
fractions.

Do not select checkpoints post hoc based on interesting behavior.

---

# E13-B — Reliability Transfer Trajectory

## 11. Measurement stack at every checkpoint

For every student checkpoint measure:

### B — behavior

- balanced accuracy;
- answer-margin AUROC;
- validation-calibrated balanced accuracy if calibration is retained.

### D — decodability

- same probe protocol as the core project;
- intervention-layer D;
- LOFO if affordable at selected checkpoints.

### L — native readout

- fixed native-readout AUROC;
- normalized-space probe/native alignment;
- relevant margin correlations.

### C — causal conversion

Use the **frozen E01B source-free setpoint protocol**, not donor-derived E01A
coordinates.

At minimum report:

- source-free opposite-class median effect;
- standardized `Delta m_z`;
- standardized `Delta q_z`;
- conversion ratio/curve;
- random and orthogonal controls;
- layerwise propagation/readout traces.

### Teacher gap

Express each student metric relative to the frozen teacher measured on the same
corpus and protocol.

## 12. Probe-direction sensitivity across training

Because student weights change, report two views:

### checkpoint-specific probe

Refit the same train/validation probe protocol independently at each saved
checkpoint. This measures the best linearly accessible coordinate at that
checkpoint.

### baseline-frozen probe sensitivity

For the 0.6B student only, also project later checkpoints onto the initial
student's frozen probe direction where dimensionality is unchanged.

This distinguishes genuine emergence/rotation from a purely refitted readout.

Do not compare raw probe vectors between teacher and student because hidden
sizes differ.

## 13. Primary E13 statistics

The central plot is a trajectory in representation-utilization space:

```text
x-axis = D
 y-axis = C or standardized native-margin response
```

for each training regime over time.

Also plot:

```text
training progress -> B
training progress -> D
training progress -> native readout
training progress -> C
```

The high-value patterns are:

### Pattern A — utilization repair

```text
D stays near ceiling
C/readout/behavior improve substantially
```

Interpretation: distillation improves use of an existing representation.

### Pattern B — representation without utilization

```text
representation KD increases representation similarity/D
C remains weak
```

Interpretation: hidden-state transfer is not equivalent to functional transfer.

### Pattern C — joint transfer

```text
D and C improve together
```

Interpretation: no dissociation under this objective/task.

### Pattern D — output imitation without internal transfer

```text
behavior improves
D/C internal metrics remain student-like
```

Interpretation: the student can imitate output without reproducing the
teacher's causal internal organization.

All four are scientifically interpretable.

## 14. Distillation-utilization lag

Define exploratory lag metrics only after the trajectory is plotted.

One predeclared option:

For metric `M`, define `t_M(90%)` as the earliest training fraction where the
student closes 90% of its baseline-to-final improvement on validation.

Compare:

```text
t_D(90%)
t_C(90%)
t_B(90%)
```

Do not call this a universal law from one training run.

Use multi-seed uncertainty before emphasizing timing differences.

## 15. Multi-seed gate

A publishable distillation phenomenon requires at least three student training
seeds for the primary regimes that survive pilot.

Do not run every expensive regime at three seeds before the one-seed pilot shows
it is informative.

---

# E13-C — Second-Task Replication

## 16. Purpose

The current relational task is intentionally controlled. A second task should
show that the utilization-transfer phenomenon is not an artifact of binary
spatial/ordering relations.

Select **one** second task, not a benchmark suite.

Desirable properties:

- clean binary or small-choice semantic variable;
- counterfactual construction or matched pairs;
- tractable teacher/student evaluation;
- no massive retrieval/index infrastructure;
- clear internal representation target.

Candidate classes:

- simple natural-language entailment polarity;
- factual attribute verification with counterfactual twins;
- controlled logical implication;
- another semantic relation family not used in E01.

Freeze the task before running E13-C.

---

# E13-D — Optional Conversion-Response Distillation

## 17. Trigger

Only implement a new objective if E13-A/B show that standard KD leaves a
meaningful, reproducible teacher-student conversion gap.

Do not invent a method merely because the project wants another contribution.

## 18. Provisional objective

The teacher and student receive their own source-free semantic setpoint
perturbations expressed in standardized coordinate units.

For a predeclared perturbation grid, e.g.:

```text
delta_z in {-1, +1}
```

cache the teacher's standardized native-margin response:

```text
r_T(delta_z) = Delta m_T / sigma_m_T
```

During student training, add:

```text
L_conversion = mean_delta_z (
    r_S(delta_z) - stopgrad(r_T(delta_z))
)^2
```

Total student objective:

```text
L = L_standard_KD + lambda_conversion * L_conversion
```

This transfers a **causal response function** rather than a hidden vector.
It avoids direct teacher/student hidden-dimension matching.

## 19. Method controls

If E13-D is reached, compare against equal-compute baselines:

- extra logit-KD updates;
- extra SFT updates;
- random-direction response matching;
- hidden-state KD;
- no conversion term.

A valid method claim requires improved held-out behavior/reliability or reduced
teacher-student C gap without simply spending more compute.

## 20. Main falsification gates

E13 should not become a paper centerpiece if:

1. D and C move together under all regimes;
2. standard KD fully closes the teacher-student conversion gap;
3. representation KD and logit KD are indistinguishable on all internal metrics;
4. the phenomenon disappears across training seeds;
5. the only signal is ordinary task accuracy improvement;
6. a second task fails qualitatively.

## 21. Compute plan

Primary teacher quantities should be cached offline when possible.
Teacher gradients are never required.

Recommended execution order:

```text
engineering smoke
-> one-seed R0/R1/R2 pilot
-> add one representation-KD baseline
-> measure trajectory
-> select informative regimes
-> three-seed discovery
-> one second task
-> only then consider E13-D
```

Do not run all branches simultaneously.

## 22. Relation to the paper narrative

If successful, E13 extends the core mechanism from inference-time causality to
training-time transfer:

```text
representation formation
    !=
representation utilization
    !=
representation-utilization transfer under distillation
```

The paper can then distinguish:

- what the student represents;
- whether that representation is causally usable;
- whether distillation transfers the teacher's utilization pathway;
- whether standard KD objectives optimize the right internal property.

This is the high-upside novelty extension. It should remain gated on E01B so the
causal quantity being transferred is defined cleanly first.
