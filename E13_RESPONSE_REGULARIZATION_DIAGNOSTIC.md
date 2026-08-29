# E13 Response-Regularization Diagnostic

Status: complete read-only discovery diagnostic. No optimizer update was performed; E13 confirmation was not accessed.

## Existing training-loss scales

| regime | seed | loss_component | mean | sd | median | p05 | p95 |
|---|---|---|---|---|---|---|---|
| R5 | 20261305 | L_KD_unweighted | 0.655251 | 0.303664 | 0.554498 | 0.481538 | 1.076125 |
| R5 | 20261305 | L_response | 2.489243 | 0.759326 | 2.472882 | 1.385370 | 3.791692 |
| R5 | 20261315 | L_KD_unweighted | 0.667428 | 0.300742 | 0.575385 | 0.468236 | 1.341592 |
| R5 | 20261315 | L_response | 2.769192 | 0.878535 | 2.585240 | 1.502398 | 4.059713 |
| R5 | 20261325 | L_KD_unweighted | 0.716112 | 0.286973 | 0.622974 | 0.538629 | 1.599515 |
| R5 | 20261325 | L_response | 3.717216 | 0.964021 | 3.564554 | 2.207655 | 5.278125 |
| R6 | 20261305 | L_KD_unweighted | 0.659054 | 0.313542 | 0.556070 | 0.475540 | 1.197689 |
| R6 | 20261305 | L_response | 2.177436 | 0.699350 | 2.144857 | 1.203942 | 3.351069 |
| R6 | 20261315 | L_KD_unweighted | 0.665561 | 0.316348 | 0.556571 | 0.473729 | 1.474809 |
| R6 | 20261315 | L_response | 2.145048 | 0.725058 | 2.007204 | 1.059437 | 3.223919 |
| R6 | 20261325 | L_KD_unweighted | 0.665302 | 0.330902 | 0.555516 | 0.479807 | 1.740932 |
| R6 | 20261325 | L_response | 2.131056 | 0.680693 | 2.017222 | 1.259564 | 3.291923 |

`L_KD_unweighted` is the logged KL term before its frozen `0.5*T^2` multiplier; `L_response` is the actual added response objective.

## Teacher target distributions (training split)

| family | component | mean | sd | median | iqr | p05 | p95 | fraction_near_zero |
|---|---|---|---|---|---|---|---|---|
| semantic | Q | -0.172190 | 0.072368 | -0.177918 | 0.118612 | -0.276762 | -0.039537 | 0.011000 |
| semantic | A | -1.171435 | 0.678660 | -1.225660 | 0.948898 | -2.174559 | 0.039537 | 0.007000 |
| semantic | G | -0.044153 | 0.098275 | -0.039537 | 0.158150 | -0.237225 | 0.079075 | 0.169250 |
| random | Q | 0.000939 | 0.084458 | 0.000000 | 0.079075 | -0.118612 | 0.158150 | 0.189750 |
| random | A | -0.005115 | 0.234085 | 0.000000 | 0.316299 | -0.395374 | 0.395374 | 0.071500 |
| random | G | 0.001245 | 0.057864 | 0.000000 | 0.079075 | -0.079075 | 0.079075 | 0.255750 |

Near zero means absolute standardized response <= 0.001. Validation distributions and all raw rows are retained in the diagnostic directory.

## Semantic/random correlations

Train corr(A_T, random-A_T): 0.057724.
Train corr(G_T, random-G_T): 0.104578.
Train semantic/random response-norm correlation: 0.256373.

## B-matched sensitivity proxies

| regime | family | mean_squared_response | response_variance | mean_abs_lipschitz_proxy | jacobian_norm_proxy |
|---|---|---|---|---|---|
| R2 | matched_context | 2.872480 | 0.863651 | 0.097178 | 0.103081 |
| R2 | semantic_q | 0.005690 | 0.005638 | 0.020847 | 0.027211 |
| R2 | unseen_random_pooled | 0.016863 | 0.016813 | 0.007811 | 0.009473 |
| R5 | matched_context | 1.538629 | 0.385443 | 0.047786 | 0.051802 |
| R5 | semantic_q | 0.016520 | 0.011404 | 0.097283 | 0.518541 |
| R5 | unseen_random_pooled | 0.264950 | 0.262981 | 0.014146 | 0.017781 |
| R6 | matched_context | 1.200614 | 0.080582 | 0.031998 | 0.032366 |
| R6 | semantic_q | 0.002771 | 0.002756 | 0.057601 | 0.308415 |
| R6 | unseen_random_pooled | 0.013624 | 0.011994 | 0.002760 | 0.003584 |

The unseen-random family pools frozen evaluation directions 2130/2131/2132, which were not the R6 training direction identity. Ratios are directional finite-difference response divided by intervention norm, not a full Jacobian.

## Full-model gradient audit

| loss_component | loss_mean | loss_min | loss_max | gradient_norm_mean | gradient_norm_min | gradient_norm_max |
|---|---|---|---|---|---|---|
| A | 1.566489 | 0.028533 | 2.629180 | 361.996232 | 31.749586 | 750.595074 |
| G | 1.283540 | 0.287875 | 2.422092 | 266.274924 | 138.319566 | 374.806866 |
| KD | 1.687768 | 1.289729 | 2.124240 | 105.604832 | 26.534428 | 161.609220 |
| Q | 8.163270 | 4.117502 | 13.151805 | 750.366122 | 352.882000 | 1249.807283 |
| R6 | 2.448895 | 1.496651 | 3.346802 | 469.259178 | 359.852935 | 711.300078 |

| checkpoint_step | batch_index | left | right | cosine |
|---|---|---|---|---|
| 10 | 0 | KD | Q | 0.099134 |
| 10 | 0 | KD | A | -0.383577 |
| 10 | 0 | KD | G | 0.084124 |
| 10 | 0 | KD | R6 | -0.241562 |
| 10 | 1 | KD | Q | -0.022400 |
| 10 | 1 | KD | A | -0.110206 |
| 10 | 1 | KD | G | -0.115915 |
| 10 | 1 | KD | R6 | -0.175782 |
| 100 | 0 | KD | Q | 0.004701 |
| 100 | 0 | KD | A | 0.009453 |
| 100 | 0 | KD | G | -0.078784 |
| 100 | 0 | KD | R6 | 0.083446 |
| 100 | 1 | KD | Q | 0.465811 |
| 100 | 1 | KD | A | -0.343798 |
| 100 | 1 | KD | G | -0.529496 |
| 100 | 1 | KD | R6 | 0.148710 |

Frozen conflict gate fired: **True**.

Gradient norms, response-component losses, and every pairwise Q/A/G/R6 cosine are retained as portable tables.

## Interpretation

The response term is not a small auxiliary: its training mean is 2.13-3.72, while the logged unweighted KL mean is 0.66-0.72. On the frozen gradient batches, Q is the largest semantic component (mean loss 8.16; mean full-model gradient norm 750), followed by A (1.57; 362) and G (1.28; 266), versus KD's 1.69 and 106. Q mismatch therefore dominates the audited response geometry even after validation-only scaling.

Teacher semantic A has the largest spread (train SD 0.679), while G is much smaller (SD 0.098, mean -0.044) and 16.9% of G targets are within 0.001 of zero. Random G is smaller still (SD 0.058; 25.6% near zero). Direct G matching consequently has a materially weaker and more quantized target than A, consistent with the G-noise hypothesis.

Semantic/random target correlations are weak: Q 0.103, A 0.058, G 0.105, and response-norm 0.256 on training data. R6 is therefore not succeeding because its cached targets are close sample-wise to the semantic targets.

R6 reduces the B-matched matched-context standardized mean-square response from R2's 2.872 to 1.201 and has a low unseen-random Jacobian proxy (0.0036 versus R2 0.0095). R5 also reduces matched-context sensitivity (1.539) but is anisotropic on unseen directions (pooled proxy 0.0178, driven by direction 2131). The best current explanation is that R6 supplies strong generic local-sensitivity/Jacobian regularization, which moves A/G magnitudes toward the teacher profile without semantic target correspondence.

Gradient conflict is heterogeneous but real. Mean cosine(KD,A) is -0.207, with values below -0.2 at both audited checkpoints; cosine(KD,G) reaches -0.529, while Q is usually aligned or near orthogonal. The frozen repeated-conflict gate therefore fires and authorizes the preregistered R15/R16 bounded controls.
