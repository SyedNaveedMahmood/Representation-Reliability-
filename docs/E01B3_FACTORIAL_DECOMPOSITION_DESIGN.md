# E01B-3 — Additive-vs-Gating Factorial Decomposition

Status: `authorized_implementation_smoke_pilot_only`

Pre-registered: 2026-08-28, before any E01B-3 GPU execution.

## Question and scope

E01B-2 established that structured orthogonal context changes native behavior
while the probe-defined truth coordinate is fixed. E01B-3 asks whether that
increment is an independent additive signal, a change in the efficacy of the
scalar intervention, or both.

This is a discovery-only mechanistic decomposition. It authorizes implementation,
deterministic tests, bounded GPU contracts, and smoke/pilot runs. It does not
authorize full discovery, confirmation access, E02, E13, E14, E15, or E16.

## Frozen scientific identity

- models: `Qwen/Qwen3-0.6B` and `Qwen/Qwen3-1.7B`;
- site: `resid_post`, zero-indexed layer 17, `last_prompt`;
- direction: the frozen unit truth-probe direction used by E01B-1/E01B-2;
- target: the validation-only opposite-class median from E01B-1;
- standardization: the existing validation-only E01B-1 coordinate and margin scales;
- context plans: the immutable E01B-2 `source_context_plan.parquet` artifacts;
- contexts: matched, same-family shuffled, different-family shuffled,
  same-label, and deterministic random orthogonal;
- context norm: each base example's E01B-2 matched-orthogonal reference norm,
  including its frozen deterministic fallback;
- context strengths: lambda 1.0 primary and lambda 0.5 sensitivity;
- trace layers: 17, 20, 23, and 27;
- target orientation: always toward the base example's frozen opposite-class
  target, never toward a context-source label.

No target, direction, source plan, norm, lambda, site, trace layer, outcome, or
interpretation gate may be changed in response to pilot results.

## Factorial arms

For base activation `h_b`, semantic edit `Delta q`, and frozen orthogonal context
`v_c`, the four arms are:

```text
Y00 = m(h_b)                         # clean
Y10 = m(h_b + Delta q)               # semantic setpoint only
Y01 = m(h_b + lambda * v_c)          # context only
Y11 = m(h_b + Delta q + lambda*v_c)  # setpoint plus context
```

`m` is the Yes-minus-No margin oriented toward the frozen opposite-class target.
Only Y01 is a new full-data forward arm. Y00, Y10, and Y11 must be reused from
compatible E01B-2 evidence after an independent bounded GPU reproduction gate.

## Primary estimands

```text
Q0 = Y10 - Y00
A_c = Y01 - Y00
Q_c = Y11 - Y01
G_c = (Y11 - Y10) - (Y01 - Y00) = Q_c - Q0
```

`A_c` is the additive context-only effect. `G_c` is the primary factorial
interaction: positive values mean that context increases the efficacy of the
same scalar intervention, negative values mean suppression, and values near
zero mean approximate additivity.

At lambda 1, report A and G for all five contexts. Primary interaction contrasts
are matched, same-family, and different-family minus random. Secondary contrasts
are matched minus different-family and same-family minus different-family.
Corresponding A contrasts are also required. Confidence intervals use a
pair-cluster bootstrap over `pair_id`; directed examples are never independently
resampled.

## Evidence reuse and compatibility gate

Each factorial row records immutable run provenance for all four arms. Before
reuse, require exact agreement on:

```text
base sample and pair
model ID and resolved revision
tokenizer identity and Yes/No token IDs
probe/scaler scientific identity
validation target and target label
context condition and source ID
context selection/random seed
lambda and applied context norm
relation metadata
site, layer, and token selector/index
```

The reconstructed probe/scaler is deterministically digested and numerically
checked against E01B-2 base coordinates, targets, and probe metrics. Source IDs,
seeds, lambdas, and norms must match the persisted E01B-2 plan. Compatibility
mismatch count must be zero. A bounded GPU contract batch independently
reproduces Y00, Y10, and Y11 before prior evidence is accepted.

## Numerical contracts

For context-only Y01 at layer 17:

```text
dot(u, context) approximately 0
dot(u, h_01) approximately dot(u, h_b)
context norm equals the frozen E01B-2 applied norm
source ID, seed, and lambda equal the E01B-2 plan
```

For Y11, `dot(u, h_11)` must equal the frozen validation target. All arms must be
finite, use the exact sample-specific last-prompt token, remove hooks after each
forward, and preserve row identity under right padding.

## Trace decomposition

At each frozen trace layer `l`:

```text
A_q(l) = q01(l) - q00(l)
G_q(l) = (q11(l)-q10(l)) - (q01(l)-q00(l))
A_m(l) = m01(l) - m00(l)
G_m(l) = (m11(l)-m10(l)) - (m01(l)-m00(l))
```

Validation-only scales produce compatible z versions. At L17, A_q and G_q must
be approximately zero by construction. Nonzero A_m or G_m at L17 therefore
localizes the corresponding effect to immediate readout geometry rather than a
change in the probe coordinate.

## Bounded execution

Smoke runs use at most 25 pairs, one frozen random seed, both lambdas, all
structured contexts, 200 bootstrap draws, and all trace layers. Pilots use at
most 75 pairs, three frozen random seeds, both lambdas, all structured contexts,
500 bootstrap draws, and all trace layers. The order is 0.6B smoke, 1.7B smoke,
0.6B pilot, 1.7B pilot. Scaling stops on any failed identity or numerical gate.

Future full-discovery commands may be handed off but not executed here. They use
all 150 discovery pairs, ten existing E01B-2 random seeds, both lambdas, all
structured contexts, 2,000 pair-cluster draws, and trace layers 17/20/23/27.

## Interpretation gates

- mostly additive multidimensional signal: structured A is positive while G is
  approximately zero;
- true structured gating: structured G is positive and exceeds random G;
- mixed additive plus gating: both A and G are positive;
- suppressive interaction: G is negative;
- nonspecific interaction: random G is comparable to structured G;
- heterogeneous/unresolved: signs or family effects are unstable.

Relation-family results for all five frozen families are reported without
selection. A null or negative interaction is a valid result. E01B-2's increment
must not be called gating unless the factorial G term supports that claim.

## Confirmation prohibition

Confirmation rows, labels, caches, metrics, and activations remain inaccessible.
They may not be used for debugging, target or source selection, standardization,
metric choice, or interpretation. Confirmation remains locked after this bounded
task regardless of pilot outcome.
