# E01B-3 Additive-vs-Gating Factorial Decomposition — Bounded Pilot

Date: 2026-08-28

Status: bounded smoke/pilot complete; full discovery not run or authorized;
confirmation locked and not accessed.

## Question

E01B-2 showed that structured orthogonal context changes behavior at a fixed
probe-defined semantic coordinate. E01B-3 decomposes that increment into:

```text
A = Y01 - Y00
G = (Y11 - Y10) - (Y01 - Y00)
```

where A is independent context-only causal signal and G is the factorial
q-by-context interaction.

## Evidence and integrity

| Model | Smoke | Pilot | Prior Y00/Y10/Y11 evidence |
|---|---|---|---|
| Qwen3-0.6B | `E01B3_079ba20e610a` | `E01B3_735b49dda22e` | `E01B2_d013bfeeabfa` for pilot |
| Qwen3-1.7B | `E01B3_908b50f0b82b` | `E01B3_097e8fb1390c` | `E01B2_909b590300eb` for pilot |

Each pilot used 75 discovery pairs / 150 directed examples, lambdas 0.5 and
1.0, all frozen structured contexts, three frozen random contexts, 500
pair-cluster bootstrap draws, and traces at layers 17/20/23/27. Only Y01 was
newly run for the full selected set. A leading contract batch independently
reproduced Y00, Y10, and Y11 with zero output and coordinate deviation in both
models. Source-plan, arm-merge, target, token, lambda, seed, and norm mismatch
counts were zero. Hook leakage was zero, all values were finite, and confirmation
access was false.

## Lambda-1 pilot decomposition

### Qwen3-0.6B

| Context | A context-only (95% CI) | G interaction (95% CI) |
|---|---:|---:|
| matched | 1.1008 [0.8557, 1.3659] | 0.0075 [-0.0088, 0.0242] |
| same-family | 0.9050 [0.7103, 1.1011] | 0.0042 [-0.0154, 0.0242] |
| different-family | 0.5867 [0.4454, 0.7250] | 0.0033 [-0.0150, 0.0200] |
| same-label | 0.0583 [-0.0830, 0.2025] | 0.0017 [-0.0154, 0.0183] |
| random | -0.0026 [-0.0129, 0.0083] | -0.0003 [-0.0121, 0.0132] |

All structured-minus-random A contrasts exclude zero. No G contrast excludes
zero. The E01B-2 increments are therefore almost entirely additive in this
pilot.

### Qwen3-1.7B

| Context | A context-only (95% CI) | G interaction (95% CI) |
|---|---:|---:|
| matched | 3.5479 [3.1446, 3.9544] | 0.1121 [0.0744, 0.1546] |
| same-family | 2.9958 [2.5925, 3.3971] | 0.1608 [0.1193, 0.2025] |
| different-family | 2.2754 [2.0728, 2.4871] | 0.1750 [0.1314, 0.2177] |
| same-label | 0.0238 [-0.1459, 0.1924] | -0.0225 [-0.0577, 0.0125] |
| random | -0.0018 [-0.0202, 0.0163] | -0.0203 [-0.0402, 0.0000] |

Opposite-label structured A and G effects are positive. Structured-minus-random
G contrasts exclude zero: matched 0.1324 [0.0983, 0.1707], same-family 0.1811
[0.1412, 0.2196], and different-family 0.1953 [0.1505, 0.2418]. Thus the 1.7B
pilot is mixed additive plus gating, although A remains the larger component.

## Trace localization

At L17, context-only A_q_z and interaction G_q_z remain approximately zero in
both models, confirming that the context arm preserves the frozen scalar
coordinate at the intervention site. Structured A_margin_z is already large at
L17. Downstream, structured context-only states generate positive A_q_z, showing
that orthogonal causal signal is transformed into the decoded truth coordinate.

In 0.6B, G_q_z and G_margin_z remain near zero through L27. In 1.7B, structured
G_q_z and G_margin_z emerge downstream (for example different-family L27:
G_q_z=0.1114 and G_margin_z=0.0531), while random interactions remain small or
negative. The scale difference therefore reflects a much larger additive
structured signal in 1.7B plus a smaller but resolved structured interaction.

## Pilot-only mechanistic verdict

- Qwen3-0.6B: mostly additive multidimensional signal.
- Qwen3-1.7B: mixed additive plus structured gating.
- pooled comparison: multidimensional-additive structure is common; resolved
  q-by-context gating is scale dependent and appears in 1.7B, not 0.6B.

The supported claim boundary is: the probe-defined scalar semantic coordinate
is actionable but incomplete. Structured orthogonal state independently carries
causal signal in both models; in the 1.7B pilot it also changes the efficacy of
the same scalar intervention. This is bounded discovery evidence, not a frozen
full-discovery or confirmation result.
