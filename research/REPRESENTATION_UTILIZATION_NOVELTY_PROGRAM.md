# Representation Utilization Research Program

Status date: 2026-08-28

Master question:

> **What determines whether an internal representation is merely present or functionally utilized?**

See `MASTER_QUESTION_ACTIONABLE_REPRESENTATIONS.md` for the full thesis and `ACTIONABLE_REPRESENTATION_ROADMAP.md` for experiment gates.

Confirmation remains locked.

## 1. Central claim to protect

The project should not be framed around the generic statement that "decodability does not imply behavior." That space is crowded.

The current discovery-supported claim is narrower and mechanistic:

> **Semantic representation formation and representation-to-behavior conversion are separable. Across two Qwen3 checkpoints with nearly identical near-ceiling decodability, direct manipulation of the decoded coordinate produces sharply different native-readout effects. Trace analysis localizes the difference primarily to readout conversion from the intervention layer onward, with an additional late propagation disadvantage in the smaller checkpoint.**

This remains checkpoint-, task-, site-, and discovery-specific. It is not yet a scaling law, and it does not establish that the unperturbed model naturally uses exactly the probe-defined one-dimensional coordinate.

The broader thesis to test, not assume, is:

> **Functional utilization is a distinct and potentially more fragile property than representational availability.**

## 2. Current evidence chain

The paper spine should preserve this sequence:

1. **Encoded:** truth is near-perfectly linearly decodable in both checkpoints.
2. **Learned/general:** random-init controls are near chance and LOFO remains high.
3. **Poorly expressed:** behavior/native readout can lag far behind D.
4. **Causally actionable:** E01A coordinate interventions beat random, orthogonal, and same-label controls with signed dose response.
5. **Checkpoint-dependent conversion:** 1.7B converts a comparable early standardized coordinate perturbation into much larger native-readout change.
6. **Mixed bottleneck:** early readout conversion dominates the difference; later layers add a propagation disadvantage in 0.6B.

The next experiments should strengthen this exact chain rather than expand into unrelated reliability dimensions.

## 3. Novelty boundary

Nearby work already establishes parts of the landscape:

- decodable information can be weakly actionable or steerable;
- model representations can encode variables that behavior does not faithfully use;
- representation geometry/readout alignment can differ across scale/checkpoint;
- tool-use and agent work already studies static cognition/action or belief/action gaps;
- RAG work already studies integration bottlenecks, retrieved-context representations, knowledge conflicts, and causal routing;
- distillation work already studies logit, hidden-state, trajectory, and even steering-direction transfer;
- quantization work studies accuracy, geometry, information loss, and some post-quantization decodability.

The targeted 2026-08-28 search did not identify a direct collision that unifies:

1. matched/high semantic decodability;
2. direct causal manipulation of that same semantic variable;
3. layerwise localization of representation-to-readout conversion;
4. testing whether utilization can be independently learned, transferred, compressed, or temporally lost.

This is not proof of absence. Continue narrow collision checks around each branch before execution.

## 4. Program architecture

The program now has a common mechanistic foundation and four transformation branches.

### Track A — establish the causal object

**E01B: Source-free setpoints and orthogonal-context modulation**

Questions:

- Is a donor-free scalar semantic set-point sufficient to reproduce E01A causal conversion?
- At fixed scalar displacement, does structured orthogonal context change the effect?

This is the immediate next experiment.

### Track B — compression fragility

**E14: Quantization Reliability**

Question:

> Can quantization preserve semantic D while degrading C/readout conversion?

This is the default first extension after E01B because it changes the same checkpoint without retraining, is inexpensive, has direct deployment relevance, and cleanly tests whether utilization is more fragile than representation.

### Track C — transfer and learnability

**E13: Distillation Reliability Transfer**

Question:

> When a student already contains the representation, does distillation transfer the teacher's ability to use it?

This has the highest method-development upside. If standard KD leaves a reproducible conversion gap, it may justify a conversion-response distillation objective that matches the teacher's causal response to source-free semantic setpoints.

### Track D — developmental emergence

**E16: Utilization Emergence During Training**

Question:

> Does D become high before C during training?

A positive result would convert the current static checkpoint dissociation into a developmental mechanism: a model may learn a semantic variable before learning to deploy it.

### Track E — temporal persistence

**E15: Temporal Causal Half-Life**

Question:

> Can a state remain decodable over a long trajectory after its causal influence on future decisions has decayed?

This is the highest-complexity branch. It should use one structured stateful task, not a generic open-ended agent benchmark.

### RAG as a later application, not the novelty spine

RAG remains relevant because retrieved information can be internally available yet fail to govern output, but the 2026 literature already directly studies this integration problem. Use RAG later only if the project's D/C machinery adds a new predictive or causal diagnostic beyond existing RAG analyses.

## 5. Track A decision tree

### E01B-1 source-free setpoints

If source-free targets reproduce a monotonic causal response, donor identity is unnecessary for the measured effect and the semantic coordinate becomes the cleaner causal object.

### E01B-2 orthogonal-context modulation

At fixed scalar coordinate target, vary orthogonal context.

Possible outcomes:

- **context-insensitive:** coordinate-only and context-added effects are close;
- **context-gated:** structured orthogonal context changes conversion;
- **random-context-sensitive:** arbitrary orthogonal additions have comparable effects, weakening semantic specificity.

Only after E01B is frozen should the existing E01 confirmation split be spent.

## 6. Track B decision tree: quantization

Primary ladder:

```text
Qwen3-1.7B BF16 -> INT8 -> INT4
```

Measure the frozen source-free semantic protocol at every precision.

High-value outcome:

```text
D remains high
while
C / standardized native-readout response drops
```

This would directly support:

> **Representational integrity is not functional integrity under compression.**

If D and C degrade together or both remain stable, do not force a fragility story.

## 7. Track C decision tree: distillation

E13 begins as a diagnostic study, not as a method.

Compare:

1. frozen student;
2. hard-label SFT;
3. standard logit KD;
4. one standard hidden-state/trajectory KD baseline;
5. optional combined KD.

Track throughout training:

- B;
- D;
- native-readout alignment;
- source-free C;
- propagation/readout conversion.

High-value outcomes include:

- D already saturated while C increases;
- hidden-state similarity improves without proportional C transfer;
- behavior improves while causal organization remains student-like.

### Method trigger

Only if standard KD leaves a robust conversion gap should the project test **conversion-response distillation**:

```text
L_conversion = mean_delta_z (
    Delta_m_student_z(delta z) - Delta_m_teacher_z(delta z)
)^2
```

The purpose is to distill how a model uses a representation, not merely its logits or hidden vector.

## 8. Track D: training emergence

For a public checkpoint family with fixed architecture, track:

```text
D(t), C(t), B(t), propagation(t), readout conversion(t)
```

The key signature is:

```text
D(t) plateaus before C(t)
```

Do not densify checkpoint sampling until a coarse early/mid/late pilot shows a plausible transition.

## 9. Track E: long-horizon causal half-life

Choose one explicit latent/state variable in a structured sequential task.

For future horizon `k`, measure:

```text
D(k): decodability of the original relevant state
C(k): causal influence of that represented state on the future decision
```

The high-value signature is:

```text
H_C < H_D
```

Do not reduce this to a generic agent success or knowing-doing study.

## 10. Falsification philosophy

The overall thesis weakens if every controlled transformation shows D and C moving together.

Each branch has its own null that must be accepted if observed:

- E01B: source-free setpoints may fail or context may dominate;
- E14: quantization may preserve both D and C or degrade both together;
- E13: standard KD may transfer D and C together;
- E16: D and C may emerge together;
- E15: C may decay only when D decays.

Negative branches should not be turned into headlines by adding more knobs.

## 11. Compute and execution priority

Use this order:

```text
1. E01B-1
2. E01B-2
3. E14 quantization
4. E13 distillation
5. E16 training emergence
6. E15 temporal causal half-life
```

This is an ordering of expected information gain per engineering cost, not a requirement that every experiment belong in one paper.

## 12. Publication paths

The project can split cleanly if one branch becomes strong:

### Core mechanism paper

```text
near-matched D
checkpoint-dependent C
source-free causal setpoints
context sufficiency/gating
trace localization
```

### Compression paper

```text
D robust under PTQ
C fragile under PTQ
```

### Distillation/method paper

```text
representation-transfer != utilization-transfer
possibly conversion-response distillation
```

### Training dynamics paper

```text
representation formation precedes utilization
```

### Long-horizon paper

```text
causal half-life shorter than representation half-life
```

## 13. Narrative to optimize for

The deepest coherent narrative is:

> **Representation availability is not functional availability. Internal variables can be learned, preserved, or transferred without an equally reliable pathway that converts them into behavior. We measure this missing pathway causally and ask when it emerges, how it depends on context, whether compression destroys it, whether distillation transfers it, and how long it remains functionally alive.**

Do not dilute this with unrelated experiments unless the central gates fail.
