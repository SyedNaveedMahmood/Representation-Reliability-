# E01A — Causal Conversion of a Decoded Truth Coordinate

Status: full discovery completed for Qwen3-0.6B and Qwen3-1.7B on
2026-08-27; confirmation remains locked. See
`../E01A_FULL_DISCOVERY_SUMMARY.md`.

## Question

At a predeclared residual-stream site where truth is almost perfectly linearly
decodable, does changing only the frozen probe-defined truth coordinate causally
move the model's native Yes/No output toward a nuisance-matched counterfactual?

The scale comparison asks whether the same type of decoded semantic coordinate
has different **causal conversion efficiency** in Qwen3-0.6B and Qwen3-1.7B.

## Intervention

Let `u` be the unit-normalized raw hidden-space direction recovered from a
logistic truth probe trained on train and selected on validation only.

For base residual `h_b` and matched counterfactual source `h_s`:

```text
q_b = u^T h_b
q_s = u^T h_s
delta_truth(alpha) = alpha * (q_s - q_b) * u
h'_b = h_b + delta_truth(alpha)
```

At `alpha=1`, only the coordinate along `u` is copied from the source. Every
component orthogonal to `u` is preserved.

## Primary outcome

The native first-token answer margin is:

```text
m = logit(Yes) - logit(No)
```

For expected counterfactual label `y_s`, the margin is oriented so larger means
more support for `y_s`. The primary effect is:

```text
Delta_m_toward_source = oriented(m_after, y_s) - oriented(m_before, y_s)
```

The provisional normalized causal-conversion diagnostic is:

```text
kappa = Delta_m_toward_source / (alpha * |q_s - q_b|)
```

for non-zero truth-coordinate interventions.

This is a diagnostic ratio, not a new reliability dimension.

## Discovery design

- dataset: existing matched-twin synthetic relational task;
- probe training: train only;
- probe C selection: validation only;
- causal evaluation: discovery-test only;
- confirmation: inaccessible;
- default site: `resid_post / layer 17 / last_prompt`;
- both directions of every selected matched pair are evaluated;
- uncertainty: cluster bootstrap by `pair_id`, not by directed example.

## Alpha profiles

Smoke: `[0, 1]`

Pilot: `[-0.5, 0, 0.5, 1, 1.5]`

Full: `[-1, -0.5, 0, 0.25, 0.5, 1, 1.5, 2]`

`alpha=0` is the explicit no-intervention baseline.

## Controls

1. `random_direction`: deterministic Gaussian direction, norm matched to the truth edit.
2. `orthogonal_random`: random direction projected orthogonal to truth, norm matched.
3. `same_label_coordinate`: different-pair same-label source using the same probe direction.
4. `shuffled_coordinate`: opposite-label source from a different matched pair.
5. `full_residual_patch`: `alpha * (h_source - h_base)` upper-bound intervention.

## Downstream trace

At predeclared layers, independently train frozen truth probes on train/validation
activations. After intervention, record truth coordinate and native fixed-readout
Yes-No margin relative to the clean base state.

Default current-Qwen trace: `17, 20, 23, 26, 27`, subject to model layer-count validation.

## Falsification gates

1. Probe validity: discovery-test D remains strong at the intervention layer.
2. Intervention fidelity: target-layer captured residual matches `h_base + delta`.
3. Magnitude control: random/orthogonal edits match truth-edit norms.
4. Dose response: counterfactual-oriented output moves coherently with alpha.
5. Specificity: truth-coordinate effect exceeds random/orthogonal controls with pair-cluster uncertainty.
6. No confirmation leakage.

If random/orthogonal controls match the truth-coordinate effect, the decoded axis
is not causally specific.

## Discovery outcome

The truth-coordinate treatment beats random, orthogonal-random, and same-label
controls in both checkpoints. It does not beat the shuffled opposite-label
coordinate control at any non-zero alpha. Therefore causal actionability under
the tested coordinate intervention is supported, while matched-source
specificity and the stricter "beats every control" gate are not supported.

Conversion is substantially stronger in Qwen3-1.7B than Qwen3-0.6B in both raw
margin effect and exploratory kappa. These findings remain discovery-only.

## Scale interpretation

The high-value comparison is `D_0.6B ≈ D_1.7B` but `C_0.6B != C_1.7B`, where C
is operationalized by controlled change in native answer margin per unit decoded-coordinate
intervention. Do not attribute any difference solely to parameter count because the
checkpoints can also differ in learned readout/interface behavior.

## Claim boundary

E01A can establish causal sensitivity to a **probe-defined coordinate under an
intervention**. It does not automatically establish that the unperturbed model
naturally uses exactly that coordinate as a causal variable.
