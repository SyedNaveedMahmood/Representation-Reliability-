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
11. Factorial decomposition shows that structured orthogonal state independently
    carries causal signal in both checkpoints; this additive component explains
    nearly all of the 0.6B E01B-2 increment.
12. Qwen3-1.7B additionally has a smaller structured q-by-context interaction
    beyond random context, whereas Qwen3-0.6B does not resolve such gating.
13. One preregistered untouched confirmation strongly replicated scalar
    actionability, additive structured information in both checkpoints, the
    1.7B structured interaction, and its cross-checkpoint difference after
    Holm correction across exactly four primary hypotheses.

This is confirmed checkpoint-dependent utilization under the frozen task/site,
not a general law across tasks, sites, architectures, or model families.

## 4. The four high-value unanswered subquestions

### Q1 — Sufficiency and context

> Is a scalar semantic set-point itself sufficient to control behavior, or is its causal meaning gated by orthogonal representational context?

Experiment family: **E01B**.

Full discovery resolves the decomposition under the frozen task/site. Orthogonal
structured state carries a large additive causal signal in both checkpoints.
Only 1.7B additionally shows a resolved structured q-by-context interaction,
and that interaction is much smaller than the additive component. The frozen
mechanism received strong confirmation; no structured interaction was detected
in 0.6B, but equivalence to zero was not tested.

### Q2 — Learning and transfer

> Does a model learn or inherit the representation before it learns or inherits the machinery that uses it?

Experiment families:

- **E13 — Distillation Reliability Transfer**
- **E16 — Utilization Emergence During Training**

The high-value signature is `D` saturating before `C`, or KD changing `C` while `D` is already near ceiling.

### Q3 — Compression fragility

> Can numerical/model compression preserve a representation while selectively degrading its causal utilization?

Experiment family: **E14 — Quantization Reliability**.

The preregistered E14 discovery and single confirmation now establish that
signature for Optimum-Quanto INT4 on Qwen3-1.7B: precision-native D remains at
ceiling while structured additive A and interaction G decline. The transformation
also causes substantial general language-model damage, so this is evidence for
mixed actionability/general fragility rather than selective semantic damage.

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
    -> E01B-3 additive-vs-gating decomposition
    -> freeze full-discovery claim
    -> one preregistered untouched confirmation (complete: strong)
    -> choose ONE high-upside extension first
         E14 quantization reliability (strong confirmation complete)
         E13 distillation reliability (bounded pilot next)
    -> E16 training emergence if suitable checkpoints are available
    -> E15 long-horizon causal half-life only after a clean sequential task is selected
    -> application-specific discovery/confirmation without retuning core E01
```

E14 is complete for the frozen primary ladder. It confirms that native-axis
decodability can outlive higher-order actionability under INT4, with the explicit
boundary that generic quality also degrades. E13 is now the active bounded branch.

E13 has the highest method-development upside because a reproducible transfer gap could justify conversion-response distillation.

E15 has high conceptual upside but much higher task/evaluation complexity and should not be used merely as an agent benchmark.

## 9. Publication-level narrative to optimize for

A strong final narrative would be:

> **Representation availability is not functional availability. We identify a semantic variable that is comparably decodable across checkpoints yet converted into behavior very differently. Causal interventions and layerwise tracing localize the gap primarily to readout conversion, with additional late propagation loss. Source-free and context-controlled interventions determine what part of the representation is sufficient. We then test whether functional utilization is learned, transferred, compressed, or temporally sustained independently of representation itself.**

The project should only broaden to the latter branches if they sharpen this claim. It should not become a collection of unrelated benchmarks.

## 10. Claim boundary

The strongest supported statement is checkpoint-, task-, and site-specific
confirmed evidence. Terms such as "knowledge," "functional utilization," and
"actionability" must remain tied to the operational variable and intervention.

The master question is intentionally broader than the current evidence; the experiment registry controls when broader claims become justified.
