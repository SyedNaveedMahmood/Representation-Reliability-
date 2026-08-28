# E01 Actionable-Representation Confirmation

Status: **strong confirmation; core E01 mechanism frozen**

## Provenance and access

- protocol: `docs/E01_CONFIRMATION_PROTOCOL.md`
- final preregistration commit: `e0ddfae54b350c0545c71a8237645375bdf84929`
- protocol SHA-256: `46312baf59923a2a0e5d1b755d313cd83b42883760baf7b5ba1209410fba81a3`
- runner commit at first access: `080e3bee9a8797eb3a2fc0adfe38c3d26eaa319d`
- first access: `2026-08-28T13:27:38.622637+00:00`
- confirmation campaigns/access count: `1`
- run: `runs/CONFIRMATION/CONFIRMATION_46312baf5992`
- sample: 200 directed examples / 100 matched pairs per checkpoint
- site: layer-17 `resid_post`, `last_prompt`
- traces: L17/L20/L23/L27
- primary lambda: 1.0; secondary lambda: 0.5
- candidate tokens: Yes `7414`, No `2308`

No confirmation artifact was accessed before the final protocol was committed and present on `origin/main`. Both model configurations were frozen before the joint campaign opened the holdout. The runner used the same semantic sample/source identities in both checkpoints and did not expose 0.6B results before beginning 1.7B.

## Primary H1-H4 family

Family-wise alpha was 0.05 with Holm correction across exactly H1-H4. Directional p-values used 100,000 pair-cluster sign-flip draws; intervals used 10,000 pair-cluster bootstrap draws. H1/H2 used their preregistered intersection-union conjunctions.

| Hypothesis | Estimate | 95% CI | Raw p | Holm p | Verdict |
|---|---:|---:|---:|---:|---|
| H1 scalar actionability/control separation in both checkpoints | 0.014375 | [0.000625, 0.019062] | 0.026080 | 0.026080 | PASS |
| H2 matched additive signal above random in both checkpoints | 1.273375 | [1.069399, 1.484001] | 0.000010 | 0.000040 | PASS |
| H3 structured interaction above random in Qwen3-1.7B | 0.132844 | [0.110312, 0.156095] | 0.000010 | 0.000040 | PASS |
| H4 structured-interaction checkpoint difference | 0.137906 | [0.113219, 0.163469] | 0.000010 | 0.000040 | PASS |

H1's checkpoint components were:

| Checkpoint | Q0 | 95% CI | Q0-random | 95% CI | Q0-orthogonal | 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.014375 | [0.001250, 0.028125] | 0.017188 | [0.007375, 0.026625] | 0.016250 | [0.006686, 0.025688] |
| Qwen3-1.7B | 0.701250 | [0.665000, 0.737188] | 0.684063 | [0.649405, 0.718720] | 0.680281 | [0.644594, 0.715313] |

H2 components were `1.273375` (`[1.068779, 1.483006]`) in 0.6B and `3.620344` (`[3.201155, 4.041351]`) in 1.7B. The confirmed 1.7B H3 contrast was `0.132844`; the corresponding 0.6B estimate was `-0.005062` (`[-0.016375, 0.006688]`). This is “no structured interaction detected in 0.6B,” not evidence of equivalence to zero.

Classification: **strong confirmation**.

## Discovery-to-confirmation comparison

| Checkpoint | Quantity | Discovery | Confirmation | Same sign | Confirmation/discovery |
|---|---|---:|---:|---|---:|
| 0.6B | Q0 | 0.026250 | 0.014375 | yes | 0.548 |
| 0.6B | A matched-random | 1.140396 | 1.273375 | yes | 1.117 |
| 0.6B | G matched-random | 0.006708 | -0.005062 | no | -0.755 |
| 1.7B | Q0 | 0.666042 | 0.701250 | yes | 1.053 |
| 1.7B | A matched-random | 3.418250 | 3.620344 | yes | 1.059 |
| 1.7B | G matched-random | 0.130812 | 0.132844 | yes | 1.016 |
| 1.7B-minus-0.6B | G matched-random | 0.124104 | 0.137906 | yes | 1.111 |

All corresponding discovery and confirmation intervals overlap descriptively. The 0.6B structured-minus-random interaction changed point sign but remained unresolved around zero in both datasets; this does not imply a confirmed negative interaction.

## Secondary confirmation

These analyses were preregistered as secondary and are not part of H1-H4.

### Continuous source-free setpoints

| Checkpoint | centered slope | 95% CI | median Spearman | fraction Spearman >= 0.8 | monotonic fraction |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.012733 | [0.007785, 0.017592] | 0.000 | 0.120 | 0.370 |
| Qwen3-1.7B | 0.039164 | [0.036832, 0.041582] | 1.000 | 0.980 | 0.905 |

This reproduces a positive population response in both checkpoints, with weak per-example monotonicity in 0.6B and near-uniform monotonicity in 1.7B.

### Layerwise factorial trace

For matched context at lambda 1:

| Model | Layer | A_q_z | G_q_z | A_margin_z | G_margin_z |
|---|---:|---:|---:|---:|---:|
| 0.6B | 17 | 0.000653 | 0.000192 | 0.792267 | -0.002301 |
| 0.6B | 20 | 0.613090 | 0.000136 | 1.111598 | 0.000927 |
| 0.6B | 23 | 0.855686 | -0.001906 | 1.082445 | 0.001418 |
| 0.6B | 27 | 1.050477 | 0.008857 | 0.900402 | 0.006328 |
| 1.7B | 17 | 0.000086 | 0.000273 | 1.052972 | -0.004644 |
| 1.7B | 20 | 0.463924 | 0.030505 | 1.122506 | 0.031407 |
| 1.7B | 23 | 0.625836 | 0.035113 | 1.157113 | 0.045067 |
| 1.7B | 27 | 0.837960 | 0.048312 | 1.180604 | 0.042740 |

Additive context signal again enters native readout immediately at L17 while decoded q is held fixed. Structured interaction develops downstream only in 1.7B. Relation-family estimates are heterogeneous: 1.7B matched G resolves for above/below, before/after, and larger/smaller, but not east/west or north/south. No family was selected or removed.

## Integrity

- status complete for both models;
- probe digests, frozen targets, revisions, and Yes/No token IDs exactly match;
- semantic source-plan digest is identical across models;
- no-op and post-hook leakage maximum selected-logit deviations are `0`;
- maximum context-dot-u is `4.16e-16` (0.6B) / `2.89e-15` (1.7B);
- maximum context norm relative mismatch is below `3.1e-16`;
- all values are finite and traces complete;
- raw scalar, factorial, and trace evidence preceded primary aggregates.

## Frozen final mechanism

Under this relation task, L17 `resid_post` / `last_prompt` site, and the two tested Qwen3 checkpoints, semantic actionability is distributed rather than reducible to one linearly decoded feature. A validation-defined probe-coordinate setpoint is causally effective in both checkpoints. Structured orthogonal state carries substantial independent causal information beyond a same-norm random orthogonal perturbation in both checkpoints. Qwen3-1.7B additionally exhibits a reproducible structured q-by-context interaction and a larger structured interaction than Qwen3-0.6B. In Qwen3-0.6B, no structured interaction was detected; zero interaction was not established.

The core E01 mechanism is now frozen. No further discovery tuning of its task, site, targets, contexts, lambdas, or estimands is permitted.

## Next gate and boundaries

H1-H4 passed without a material sign reversal, so the preregistered gate to design and execute E14 stage 0/stage 1 passes. This confirmation is checkpoint-, model-family-, task-, intervention-, and site-specific. It does not prove that the probe axis is the endogenous natural code or that distributed actionability generalizes to unrelated concepts or architectures.
