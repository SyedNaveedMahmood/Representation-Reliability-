# Master Research Question: What Makes Knowledge Actionable?

Status date: 2026-08-28

## 1. The question

> **What determines whether an internal representation is merely present or functionally utilized by a neural network?**

Operationally, the project studies the missing transformation:

```text
representation availability
        -> propagation
        -> integration / readout coupling
        -> behavioral expression
```

The core object is **representation-to-behavior conversion**. A representation can be highly decodable without being equally propagated, integrated, read out, or behaviorally expressed.

This project therefore distinguishes at least four properties:

- **D — Decodability:** is the variable linearly accessible?
- **P — Propagation:** does a perturbation of that variable survive downstream computation?
- **C — Causal conversion:** does changing the variable systematically change native predictive behavior?
- **B — Behavior:** does the unperturbed model produce the correct/desired output?

`P` is a mechanism diagnostic rather than a new canonical reliability dimension unless later work justifies promoting it.

## 2. Working thesis

The working thesis is:

> **Functional utilization is a distinct and potentially more fragile property than representational availability.**

A stronger version, to be earned rather than assumed, is:

> **Neural networks can preserve what they represent while changing, losing, failing to transfer, or failing to sustain how that representation controls behavior.**

The project should try to falsify this across transformations that change models in qualitatively different ways.

## 3. Current evidence

The current Qwen3 discovery already provides one controlled instance:

1. Qwen3-0.6B and Qwen3-1.7B have near-ceiling truth decodability.
2. The decoded variable is learned and relation-general under the current task.
3. Native behavior/readout differs substantially despite similar D.
4. Direct coordinate interventions are causally actionable relative to random, orthogonal, and same-label controls.
5. At L17/L20, the standardized injected truth perturbation is comparable across checkpoints, while native-readout response is much stronger in 1.7B.
6. At L23/L27, 1.7B additionally retains more of the standardized perturbation.
7. The current discovery classification is therefore a mixed bottleneck dominated by readout conversion.
8. Source-free validation-defined setpoints reproduce donor-derived coordinate effects.
9. At an identical scalar setpoint and orthogonal norm, opposite-label structured
   contexts change behavior far more than random context in both checkpoints.
10. The context effect enters native readout at L17 while decoded q remains fixed,
    then changes downstream q propagation; the effect is larger in 1.7B.

This is evidence for checkpoint-dependent utilization, not yet a general law.

## 4. The four high-value unanswered subquestions

### Q1 — Sufficiency and context

> Is a scalar semantic set-point itself sufficient to control behavior, or is its causal meaning gated by orthogonal representational context?

Experiment family: **E01B**.

Full discovery supports structured context sensitivity. The remaining boundary
is whether the orthogonal component gates conversion multiplicatively or carries
an additional causal signal additively; confirmation remains untouched.

### Q2 — Learning and transfer

> Does a model learn or inherit the representation before it learns or inherits the machinery that uses it?

Experiment families:

- **E13 — Distillation Reliability Transfer**
- **E16 — Utilization Emergence During Training**

The high-value signature is `D` saturating before `C`, or KD changing `C` while `D` is already near ceiling.

### Q3 — Compression fragility

> Can numerical/model compression preserve a representation while selectively degrading its causal utilization?

Experiment family: **E14 — Quantization Reliability**.

The high-value signature is a bit-width regime where `D` remains high but `C` or downstream readout conversion drops sharply.

### Q4 — Temporal persistence

> How long does represented information remain functionally alive over a long trajectory?

Experiment family: **E15 — Temporal Causal Half-Life**.

The high-value signature is stable decodability of a state across steps while its ability to causally influence future action decays with horizon.

## 5. Why these branches belong together

These transformations probe different failure modes of the same hidden arrow:

```text
                   What makes a representation actionable?
                                  |
          -------------------------------------------------
          |                 |              |              |
      mechanism          transfer       compression     horizon
        E01B               E13              E14            E15
          |                 |              |              |
   sufficient/context   inherited?      preserved?      persists?
                                  |
                           training emergence
                                  E16
```

A coherent paper or paper series should not claim all branches at once. The value of this structure is that each experiment asks the same mechanistic question under a different perturbation of the model/system.

## 6. Primary falsifiable hypotheses

### H1 — Representation/utilization separability

There exist controlled conditions where D is matched or nearly matched while C differs materially.

Falsifier: across properly controlled transformations, D and C always move together within uncertainty.

### H2 — Utilization fragility

There exist transformations that leave D largely intact while degrading C, P, or native-readout coupling.

Candidate transformations: quantization, long-horizon propagation, some student-training regimes.

Falsifier: every degradation in C is fully accounted for by loss of D.

### H3 — Utilization learnability

C can improve substantially while D is already saturated.

Candidate transformations: distillation, continued training, post-training.

Falsifier: C improves only when D materially improves.

### H4 — Contextual causal meaning

The same semantic coordinate displacement can have different behavioral consequences under different orthogonal contexts.

Candidate experiment: E01B-2.

Falsifier: structured orthogonal context has no reproducible effect beyond random orthogonal context.

Discovery status: the falsifier was not observed. Structured opposite-label
contexts exceeded random context in both models, with aggregate relation-family
specificity and family-level heterogeneity. This remains task/site/model-family
specific and unconfirmed.

### H5 — Temporal causal decay

A representation may remain decodable over a trajectory after its causal influence on future decisions has substantially decayed.

Candidate experiment: E15.

Falsifier: causal influence tracks decodability over horizon without meaningful divergence.

## 7. Required measurement discipline

All branches should reuse the same conceptual measurement stack where possible:

1. measure D on held-out data;
2. use explicit causal interventions rather than probes alone;
3. preserve source-free setpoint controls where the variable permits them;
4. measure downstream P / readout conversion;
5. compare against norm-matched random and orthogonal controls;
6. keep per-example raw evidence;
7. cluster uncertainty at the correct semantic unit;
8. preserve untouched confirmation until the mechanism is frozen;
9. avoid interpreting a probe direction as endogenous natural use without additional evidence.

## 8. Program order

The intended order is:

```text
E01B-1 source-free setpoints
    -> E01B-2 orthogonal context
    -> freeze core mechanism
    -> choose ONE high-upside extension first
         default: E14 quantization reliability
         second: E13 distillation reliability
    -> E16 training emergence if suitable checkpoints are available
    -> E15 long-horizon causal half-life only after a clean sequential task is selected
    -> confirmation after the mechanism/claim intended for the paper is frozen
```

E14 is the default first application/robustness extension because it is cheap, deployment-relevant, and directly asks whether functional utilization is more fragile than representation under compression.

E13 has the highest method-development upside because a reproducible transfer gap could justify conversion-response distillation.

E15 has high conceptual upside but much higher task/evaluation complexity and should not be used merely as an agent benchmark.

## 9. Publication-level narrative to optimize for

A strong final narrative would be:

> **Representation availability is not functional availability. We identify a semantic variable that is comparably decodable across checkpoints yet converted into behavior very differently. Causal interventions and layerwise tracing localize the gap primarily to readout conversion, with additional late propagation loss. Source-free and context-controlled interventions determine what part of the representation is sufficient. We then test whether functional utilization is learned, transferred, compressed, or temporally sustained independently of representation itself.**

The project should only broaden to the latter branches if they sharpen this claim. It should not become a collection of unrelated benchmarks.

## 10. Claim boundary

Until further experiments succeed, the strongest supported statement remains checkpoint-, task-, and site-specific discovery evidence. Terms such as "knowledge," "functional utilization," and "actionability" must always be tied to an operationally measured variable and intervention.

The master question is intentionally broader than the current evidence; the experiment registry controls when broader claims become justified.
