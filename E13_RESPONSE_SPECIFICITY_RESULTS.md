# E13 Response-Specificity Results

Status: frozen one-seed discovery controls complete; E13 confirmation was not accessed or materialized.

Protocol: `docs/E13_RESPONSE_SPECIFICITY_PROTOCOL.md`, SHA-256 `08b45d7912c2110b87fb2915b983dcc9e4d2b66a0c961efa2bb8c46438d078ba`
Campaign: `runs/E13_METHOD_REVISION/E13MR_08b45d7912c2_79d9141a25b4`
Seed: `20261305`; permutation seed `20261331`; 100 optimizer steps; effective batch 8; response coefficient 1.0.

Frozen references: teacher validation B `0.973608`, teacher `Qz 0.163981`, `Az 1.168644`, `Gz 0.046239`, `COD 0`; untrained student R0 `B 0.747000`, `Qz -0.000147`, `Az 0.787599`, `Gz 0.006628`, `COD 0.799059`.

## Behavior-matched results

| regime | B_student | Q_z | A_z | G_z | COD | Q_gap | A_gap | G_gap |
|---|---|---|---|---|---|---|---|---|
| R5 | 0.945133 | 0.103726 | 1.167765 | 0.030552 | 0.507220 | 0.135970 | 0.436942 | 0.112094 |
| R6 | 1.000000 | 0.006688 | 1.410263 | 0.014650 | 0.552956 | 0.161925 | 0.480930 | 0.110635 |
| R7 | 0.992911 | 0.019089 | 0.940671 | 0.009014 | 0.545301 | 0.153818 | 0.479802 | 0.093381 |
| R8 | 0.999200 | 0.027335 | 1.255196 | 0.008542 | 0.454964 | 0.146949 | 0.386120 | 0.091891 |
| R9 | 0.810711 | -0.005440 | 0.161380 | 0.038078 | 1.246526 | 0.300941 | 1.056315 | 0.343442 |
| R10 | 0.960222 | 0.013558 | 1.286000 | 0.015441 | 0.827450 | 0.151823 | 0.786247 | 0.077413 |

Selected steps and absolute validation B gaps: R5 step 10 (`0.016528`), R6 step 25 (`0.026376`), R7 step 25 (`0.017360`), R8 step 25 (`0.025840`), R9 step 100 (`0.164744`), R10 step 10 (`0.020976`). Quality stayed in band for every regime (`PPL 31.04-32.17`, HellaSwag `0.470-0.480`; R0 `30.574`/`0.470`).

## Sample-level alignment, which COD alone does not expose

COD is a mean per-example Euclidean norm of the `(Q_z, A_z, G_z)` gap, so it is dominated by the large-magnitude A component. The per-example profile correlation and per-component sign agreement separate *magnitude* matching from *sample-level correspondence*.

| regime | COD | profile pearson | Q sign agreement | A sign agreement | G sign agreement |
|---|---|---|---|---|---|
| R2 | 0.733286 | 0.7852 | 0.600 | 0.950 | 0.677 |
| R5 | 0.507220 | 0.8570 | 0.903 | 0.950 | 0.640 |
| R6 | 0.552956 | 0.8427 | 0.823 | 0.953 | 0.610 |
| R7 | 0.545301 | 0.8509 | 0.850 | 0.953 | 0.657 |
| R8 | 0.454964 | 0.8858 | 0.867 | 0.953 | 0.637 |
| R9 | 1.246526 | 0.2123 | 0.800 | 0.823 | 0.670 |
| R10 | 0.827450 | 0.7308 | 0.793 | 0.953 | 0.633 |

## Permutation integrity

R7 and R8 mappings are fixed-point free at the pair level (`fixed_point` false for all 4,000 rows), preserve gold-label orientation on every row (`gold_label == target_gold_label`, fraction `1.0000`) and, for R8, relation family exactly (fraction `1.0000`, versus `0.1900` for the deliberately global R7 permutation). The exact marginal-preservation assertion on sorted `R5_Q`/`R5_A`/`R5_G` passed in-run for both regimes. R9 and R10 use the identity sample mapping by design and vary only the target family or the intervention geometry.

## Interpretation

**Sample-specific semantic teacher-response matching is not what produces the COD improvement.** R8, whose teacher targets are permuted within relation family so that no sample receives its own causal response, reached the *lowest* COD of the whole campaign (`0.454964` versus R5 `0.507220` and R6 `0.552956`) while remaining teacher-like in behavior. R7, permuted globally, was also close to R5 (`0.545301`). Hypothesis **H-A** is supported: most of the effect is generic structured response regularization.

The mechanism is visible in the `A_z` trajectories. Plain logit KD *overshoots* the teacher's action effect — R2 drives `A_z` from `0.788` to `1.757` by step 25 against a teacher value of `1.169`. Every response penalty, semantic or not, acts as a brake on that overshoot: at their selected steps R5 sits at `1.168`, R8 at `1.255`, R6 at `1.410`. Because A dominates the COD norm, any objective that restrains the response magnitude buys most of the COD reduction. This also explains the diagnostic's finding that R6 lowers the matched-context sensitivity proxy from `2.872` to `1.201`.

**A smaller, genuinely semantic-specific signal does exist, and it lives in Q, not in COD.** Q sign agreement rises from `0.600` under plain KD to `0.823` under random-response matching — the generic-regularization effect — and then to `0.903` only when the teacher's own sample-matched semantic Q/A/G targets are used. The shuffled controls land in between (R7 `0.850`, R8 `0.867`), which is what a partly-destroyed correspondence signal should look like. R5 is also the only specificity regime that moves `Q_z` materially toward the teacher's `0.163981` (`0.103726` versus `0.006688`-`0.027335` for the controls). So sample-specific semantic targets do transfer something, but it is a second-order effect on the smallest component, and the frozen COD summary is not sensitive enough to reward it.

**Geometry and target must be drawn from the same family.** The two intentionally mismatched controls both failed, in different ways. R9 (semantic student geometry, random teacher targets) collapsed outright: it never recovered teacher-like behavior (best absolute validation B gap `0.164744`), drove `A_z` down to `0.161`, and produced the worst COD in the campaign (`1.246526`) with profile correlation falling to `0.2123`. R10 (random student geometry, semantic teacher targets) kept behavior but was worse than plain KD on COD (`0.827450` versus R2 `0.733286`) and had the highest response-to-KD loss ratio of the four controls (`6.04`). Asking a model to reproduce response magnitudes measured along directions it is not being probed along is actively harmful, which rules out the strongest reading of a pure scale-regularization account: it is not the case that *any* response penalty helps.

Per the frozen protocol, R7-R10 are mechanism controls and are never eligible to become `R_best`; no control was promoted to additional seeds.
