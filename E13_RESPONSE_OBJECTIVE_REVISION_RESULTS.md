# E13 Response-Objective Revision Results

Status: frozen one-seed objective comparison complete; E13 confirmation was not accessed or materialized.

Protocol: `docs/E13_RESPONSE_OBJECTIVE_REVISION_PROTOCOL.md`, SHA-256 `79d9141a25b420ed6b8fcd290297384cb9d31e920d53dec634bb0462b6a180b0`
Campaign: `runs/E13_METHOD_REVISION/E13MR_08b45d7912c2_79d9141a25b4`
Seed: `20261305`; 100 optimizer steps; effective batch 8; response coefficient 1.0.

## Behavior-matched results

| regime | B_student | Q_z | A_z | G_z | COD | Q_gap | A_gap | G_gap |
|---|---|---|---|---|---|---|---|---|
| R11 | 0.924889 | 0.069604 | 1.375384 | 0.016357 | 0.635353 | 0.124031 | 0.576841 | 0.111692 |
| R12 | 0.880311 | 0.030847 | 1.191939 | 0.003856 | 0.768399 | 0.161391 | 0.697597 | 0.133858 |
| R13 | 0.912311 | 0.100713 | 1.094943 | 0.047692 | 0.594429 | 0.107507 | 0.549557 | 0.100125 |
| R14 | 0.959133 | 0.009564 | 1.329864 | 0.031284 | 0.697306 | 0.157385 | 0.644674 | 0.080236 |
| R15 | 1.000000 | 0.052866 | 0.643558 | 0.008404 | 0.667481 | 0.118813 | 0.627551 | 0.084223 |
| R16 | 1.000000 | 0.066886 | 0.820997 | 0.032750 | 0.595298 | 0.115408 | 0.556096 | 0.084456 |

Comparators at the same seed: R2 `COD 0.733286`, R3 `0.707304`, R5 `0.507220`, R6 `0.552956`.

## Frozen candidate gate

| regime | criterion_1_B_gap | criterion_2_COD_below_R5 | criterion_3_COD_below_R6 | criterion_4_G_gap | criterion_5_quality | eligible | COD | Q_gap | A_gap | G_gap |
|---|---|---|---|---|---|---|---|---|---|---|
| R11 | False | False | False | True | True | False | 0.635353 | 0.124031 | 0.576841 | 0.111692 |
| R12 | False | False | False | False | True | False | 0.768399 | 0.161391 | 0.697597 | 0.133858 |
| R13 | False | False | False | True | True | False | 0.594429 | 0.107507 | 0.549557 | 0.100125 |
| R14 | True | False | False | True | True | False | 0.697306 | 0.157385 | 0.644674 | 0.080236 |
| R15 | True | False | False | True | True | False | 0.667481 | 0.118813 | 0.627551 | 0.084223 |
| R16 | True | False | False | True | True | False | 0.595298 | 0.115408 | 0.556096 | 0.084456 |

**No candidate passed the frozen gate.** `selected_regime: null`; every regime failed criteria 2 and 3, and R11/R12/R13 additionally failed the behavior criterion. Under the frozen rule no three-seed method wave was run.

## Objective scale is the controlling variable

Mean training-loss magnitudes over all 100 optimizer steps:

| regime | mean KD | mean response | response/KD |
|---|---|---|---|
| R14 (G only) | 0.6259 | 1.3712 | 2.19 |
| R15 (projected) | 0.6644 | 2.0636 | 3.11 |
| R16 (norm-balanced) | 0.5549 | 2.2076 | 3.98 |
| R13 (Q/A only) | 0.7964 | 5.6819 | 7.13 |
| R11 (full arm) | 0.8476 | 7.2458 | 8.55 |
| R12 (whitened) | 0.9847 | 21.7047 | 22.04 |

The rank order of this ratio predicts the behavior failures almost exactly. The three regimes that lost teacher-like behavior are the three with the largest response terms: R12 (ratio `22.0`, absolute validation B gap `0.103744`), R11 (`8.6`, `0.054824`) and R13 (`7.1`, `0.059552`). The three that stayed teacher-like are the three smallest. At coefficient 1.0 these are not auxiliary losses — they are the dominant objective, and the frozen coefficient was inherited from R5 rather than re-derived per objective. This is the single clearest engineering lesson of the wave.

## Does full-arm matching beat Q/A/G matching?

No. R11 raised the response loss to `7.25`, cost teacher-like behavior (B gap `0.054824` against the `0.03` bound) and produced a worse COD than R5 (`0.635353` versus `0.507220`). Matching `r10`/`r01`/`r11` as a summed triple rather than an averaged derived triple multiplies the effective response weight without adding usable structure. Hypothesis **H-E** is not supported in this design.

## Does whitening help?

No — it was the worst variant tested. R12's inverse shrinkage covariance inflates exactly the low-variance directions the diagnostic flagged: teacher G has train SD `0.098` against A's `0.679`, so whitening amplifies the noisiest component by roughly the inverse variance ratio. The result was a response term of `21.70` against a KD term of `0.98`, monotone behavioral collapse (validation B `0.8699`, `0.7965`, `0.6495`, `0.6356` across steps 10/25/50/100), the only criterion-4 failure in the wave, and the worst COD among the objective variants (`0.768399`). **H-B** is not supported: the problem with the R5 objective is not that a high-variance component dominates, and correcting for that made matters strictly worse.

## Is direct G optimization harmful?

Not harmful, but nearly inert — and the more interesting half of the answer is that G does not come along for free. R14 (G only) is the best-behaved regime in the wave (B gap `0.022536`, response ratio `2.19`) and delivers the smallest G gap of any objective variant (`0.080236`, better than R5's `0.112094`), so G *is* individually learnable and direct G matching does not destabilize training. But it barely moves the overall organization: COD `0.697306`, only `0.036` below plain KD.

Conversely R13, which excludes G from the loss entirely and leaves it as an untouched outcome, has a G gap of `0.100125` — worse than every regime that does train on G directly (R14 `0.080236`, R15 `0.084223`, R16 `0.084456`) — with a `G_z` that drifts upward across training (`0.0161`, `0.0124`, `0.0477`, `0.0785` at steps 10/25/50/100), overshooting the teacher's `0.046239` by the final step. Improving Q and A does not automatically improve G. G is a distinct transfer problem, consistent with the premise behind hypothesis **H-C** about G's weak signal (16.9% of teacher G targets within `0.001` of zero), even though the predicted harm from direct G matching did not appear.

## Task 4: gradient conflict

The frozen conflict gate fired in the Task 1 diagnostic, so R15 and R16 were authorized and run. R15 additionally records the in-training conflict directly, from full accumulated `g_KD` and `g_response` at each of the 100 optimizer steps — a much stronger measurement than the two-batch read-only audit that opened the gate.

| statistic | value |
|---|---|
| mean cos(g_KD, g_response) | -0.0090 |
| median | -0.0092 |
| SD | 0.1686 |
| p05 / p95 | -0.2601 / 0.2674 |
| min / max | -0.4366 / 0.3751 |
| fraction negative | 0.550 |
| fraction below -0.2 | 0.150 |
| steps where projection applied | 55 / 100 |
| mean ratio of response to KD gradient norm | 3.373 |
| median ratio of response to KD gradient norm | 3.087 |

Conflict is real but mild and roughly symmetric about zero: the two objectives are close to orthogonal on average, with a negative tail on 15% of steps. R16's independent measurement agrees on magnitude (median ratio `4.635`). **The dominant pathology is not direction, it is scale** — the response gradient is roughly three to five times larger than the KD gradient throughout training. Hypothesis **H-D** is only weakly supported.

Neither remedy rescued the method under the frozen gate, but they behaved differently and informatively. R16 (norm-balancing to `rho = 0.5`) is the better of the two on every summary: it holds validation B at `1.000000` with a `0.026392` gap, and gets COD to `0.595298` — the best of the six objective variants — versus R15's `0.667481`. R15's projection, which only acts on the 55% of steps with a negative dot product, left the scale problem untouched and drove `A_z` down to `0.6436` at its selected step, well below the teacher's `1.168644`.

## Sample-level alignment

As in the specificity wave, COD understates what the gradient-balanced variants achieve at the per-example level:

| regime | COD | profile pearson | Q sign agreement |
|---|---|---|---|
| R2 | 0.733286 | 0.7852 | 0.600 |
| R5 | 0.507220 | 0.8570 | 0.903 |
| R6 | 0.552956 | 0.8427 | 0.823 |
| R13 | 0.594429 | 0.8066 | 0.970 |
| R15 | 0.667481 | 0.8917 | 0.970 |
| R16 | 0.595298 | 0.8639 | 0.980 |

R15 has the highest per-example profile correlation in the entire campaign (`0.8917`) and R16 the highest Q sign agreement (`0.980`), yet both are ineligible because their `A_z` undershoots the teacher and COD is a magnitude-sensitive norm. Every regime whose loss contains a sample-matched semantic Q target (R5, R13, R15, R16) reaches Q sign agreement at or above `0.903`; every regime without one stays at or below `0.867`. The frozen selection statistic and the semantic-specific signal are measuring different things.

## Provenance note

R15's artifacts were first produced while two identical `e13-revision-job --regime R15 --seed 20261305` processes were briefly alive at once, because a resume scheduler was started while the original wave was still running. The losing process raised `FileExistsError` from `begin_atomic_checkpoint` and the survivor rewrote `status.json`, so the run could not be attributed to a single trajectory. Those artifacts were quarantined under `runs/E13_METHOD_REVISION/quarantine/` with a README, and R15 was re-run as a single process into a fresh directory. The re-run reproduced the quarantined run **exactly** — maximum absolute difference `0.000e+00` across all 25 recorded checkpoint values and an identical `run_identity_sha256` (`dba8db71d6302133`) — which both confirms the training path is deterministic and establishes that the race did not affect any reported number. R7-R14 and R16 each have a single writer, verified from their status histories. Only the clean R15 appears in the tables above. The duplicate scheduler record is retained at `runs/E13_METHOD_REVISION/20260829T_revision_wave_v2/`.
