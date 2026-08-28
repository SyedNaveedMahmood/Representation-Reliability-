# E15 — Temporal Causal Half-Life: How Long Does Represented Information Remain Functionally Alive?

Status: proposed, not authorized.

## 1. Question

> **In a long-horizon task, can a state variable remain internally decodable while its causal influence on future actions decays with horizon?**

This is deliberately narrower than a generic belief-action or agent-success study.

The scientific object is the **temporal causal half-life** of an internal representation.

## 2. Core idea

At step `t`, identify a state variable `z_t` that should remain relevant to future decisions.

For future step offsets `k` measure:

```text
D(k): how decodable is z_t from the agent/model state at t+k?
C(k): how much does a controlled intervention on z_t or its carried representation change the future action/output at t+k?
```

The high-value signature is:

```text
D(k) remains high
while
C(k) declines materially with k
```

This would show that represented information can remain present after becoming functionally disconnected from action.

## 3. Task requirements

Do not use a task merely because it is long.

The chosen task must provide:

1. a clearly defined latent/state variable;
2. deterministic or auditable state transitions;
3. known future decisions that depend on the variable;
4. controllable nuisance factors;
5. trajectories long enough to vary horizon;
6. a natural intervention semantics;
7. enough repeated episodes for cluster-aware inference.

Preferred initial domains:

```text
structured navigation with persistent goal/state
procedural planning with delayed constraints
multi-step synthetic tool workflows
stateful text environments with explicit hidden state
```

Avoid open-web browsing or unconstrained coding agents for the first version because evaluation ambiguity would dominate the mechanistic question.

## 4. Candidate variable examples

Examples of suitable variables:

- a required destination / goal identity;
- a delayed constraint that becomes relevant several steps later;
- a permission/forbidden-action flag;
- an inventory/resource state;
- a task dependency that must be remembered across distractor steps.

The variable should have matched counterfactual episodes differing only in the target state where feasible.

## 5. Representation sites

The specific architecture depends on the selected agent setup.

Possible carriers:

```text
current transformer residual state
explicit memory token representation
compressed trajectory summary
recurrent scratchpad state
KV-cache-derived representation
```

Do not mix carriers in the first experiment.

Predeclare one primary carrier and one primary token/site before discovery.

## 6. Primary measurements

For each horizon `k`:

### Decodability

```text
D(k) = held-out probe performance for z_t from state at t+k
```

### Causal utilization

Apply a semantically controlled setpoint/counterfactual intervention at the chosen carrier and measure the change in the future decision that depends on `z_t`.

```text
C(k) = counterfactual-oriented change in future action/output margin
```

Where possible, use source-free setpoints rather than donor-state patching.

### Propagation / persistence

Measure how much of the intervention-defined coordinate/state perturbation remains detectable across subsequent steps.

## 7. Temporal causal half-life

Define a descriptive normalized utilization curve:

```text
C_rel(k) = C(k) / C(k0)
```

where `k0` is the earliest valid future decision horizon.

Define the **causal half-life**:

```text
H_C = smallest k such that C_rel(k) <= 0.5
```

Only report `H_C` if the curve is sufficiently smooth/monotonic for the summary to be meaningful.

Similarly define a representation half-life `H_D` from a predeclared normalized D measure.

The high-value inequality is:

```text
H_C < H_D
```

meaning functional influence dies before representational accessibility.

## 8. Controls

Required where applicable:

- no intervention;
- norm-matched random direction/state perturbation;
- target-variable-orthogonal perturbation;
- irrelevant-state variable intervention;
- matched horizon/distractor count;
- action-frequency and position controls;
- episode-level cluster bootstrap;
- shuffled future-decision mapping as a diagnostic null.

## 9. Main hypotheses

### H15.1 — temporal decodability persistence

The target state remains decodable across a meaningful horizon.

### H15.2 — utilization decay

Causal influence on future decisions decays faster than D.

### H15.3 — distractor sensitivity

At fixed horizon, additional irrelevant computation/distractors reduce C more strongly than D.

### H15.4 — checkpoint dependence

If tested across checkpoints, models may have similar D(k) but different H_C.

Do not begin with cross-model scaling; establish the phenomenon in one model first.

## 10. Falsification

The main claim weakens if:

- D and C decay together without meaningful separation;
- apparent C decay is fully explained by future decision uncertainty;
- interventions cease to be numerically faithful at later steps;
- long-horizon errors reflect task-state corruption rather than utilization loss;
- the representation itself is no longer present.

## 11. Minimum staged design

Stage 0: choose one synthetic/stateful task and validate exact state labels.

Stage 1: establish D(k) without intervention.

Stage 2: intervention smoke at 2–3 horizons.

Stage 3: full horizon curve for one model.

Stage 4: only if the dissociation exists, replicate on one additional model/checkpoint or task.

Do not combine tool-use, RAG, memory, and planning in the first study.

## 12. Why this is not just a knowing-doing gap

A static knowing-doing gap asks whether a represented belief/state influences an action.

E15 asks a temporal mechanistic question:

> **How does the causal efficacy of the same represented variable evolve as computation proceeds and the relevant action moves farther away?**

The contribution is a measured decay curve and a causal persistence timescale, not merely an agent success gap.

## 13. Claim boundary

A positive E15 would support a task/model-specific statement that functional influence can decay faster than representational accessibility over a trajectory.

It would not imply a universal exponential decay law or a general memory theory without replication.
