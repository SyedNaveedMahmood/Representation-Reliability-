# E13 Conversion-Response Distillation Discovery

Status date: 2026-08-29. The frozen open-discovery campaign is complete. E13 confirmation was not accessed or materialized.

Protocol SHA-256: `3f3dd9a65347fc9ba6a20c29686aba11bd578f52b28d87818c399b422325846b`  
Baseline campaign: `runs/E13_MULTI_SEED/E13MS_04daa7fcc66c`  
Method campaign: `runs/E13_CONVERSION_RESPONSE/E13CR_3f3dd9a65347`  
Scheduler record: `runs/E13_OVERNIGHT/20260829T_method_wave_fixed`

## Engineering and cache provenance

The first scheduler stopped before training because teacher-cache construction passed the 1024-dimensional student semantic direction to 2048-dimensional teacher activations. This was an implementation violation of the already-frozen model-local-space protocol, not a scientific failure. Commit `39d4f378a019ab152fce40d2a1584b6208378177` introduced explicit model-local references and shape/revision/role contracts. Follow-up commits `3c74ee6ca70d483762a75f42c6311737a74e8307` and `b114fe267a6da26e41201a33dee5634b3f71bfd6` made the deterministic live check preserve original batch identity and handle the final partial cache batch. Full details are in `docs/E13_TEACHER_CACHE_HIDDEN_SPACE_BUGFIX.md`.

The completed cache has 4,500 rows, a 2048-dimensional teacher direction, teacher probe digest `3d64282f54e36e3ed41443e4e0fc91fadc06982f73c529c30d747552733ce7c2`, teacher target digest `09ea85c5bc82f4863e527d6bed9bc5e987477307b4a73c85d380493d5278e5f1`, and response tensor digest `190b298d5ab094ca246f9cfce166ba3d0635a2fa88d5f337d563cd9a77f97b1c`. Every R4/R5/R6 arm in the frozen 16-row live check had max absolute and mean absolute discrepancy `0.0`, below the frozen `0.02` and `0.002` tolerances.

## B-matched primary results

| Regime | Seed | Step | B | Qz | Az | Gz | COD | PPL | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R2 | 20261305 | 10 | 0.957511 | 0.011149 | 1.289121 | 0.037009 | 0.733286 | 31.310094 | 0.472 |
| R2 | 20261315 | 10 | 0.962356 | -0.003058 | 1.395682 | 0.063449 | 0.876203 | 31.172057 | 0.470 |
| R2 | 20261325 | 10 | 0.996111 | -0.004694 | 1.554582 | 0.056832 | 0.742157 | 31.268386 | 0.470 |
| R3 | 20261305 | 25 | 1.000000 | -0.007894 | 1.738960 | 0.051873 | 0.707304 | 32.272230 | 0.466 |
| R3 | 20261315 | 10 | 0.960200 | 0.001315 | 1.402560 | 0.056914 | 0.893400 | 31.176124 | 0.476 |
| R3 | 20261325 | 10 | 0.997200 | -0.005299 | 1.553740 | 0.059490 | 0.743829 | 31.278389 | 0.474 |
| R2-C | 20261305 | 10 | 0.957511 | 0.011149 | 1.289121 | 0.037009 | 0.733286 | 31.310094 | 0.472 |
| R2-C | 20261315 | 10 | 0.962356 | -0.003058 | 1.395682 | 0.063449 | 0.876203 | 31.172057 | 0.470 |
| R2-C | 20261325 | 10 | 0.996111 | -0.004694 | 1.554582 | 0.056832 | 0.742157 | 31.268386 | 0.470 |
| R4 | 20261305 | 25 | 1.000000 | -0.002458 | 1.747378 | 0.048557 | 0.705450 | 32.169976 | 0.466 |
| R4 | 20261315 | 10 | 0.964444 | 0.005985 | 1.407305 | 0.050314 | 0.878875 | 31.153464 | 0.474 |
| R4 | 20261325 | 10 | 0.997511 | -0.005557 | 1.558761 | 0.058262 | 0.740436 | 31.300372 | 0.474 |
| R5 | 20261305 | 10 | 0.945133 | 0.103726 | 1.167765 | 0.030552 | 0.507220 | 31.040166 | 0.480 |
| R5 | 20261315 | 100 | 0.999911 | 0.043331 | 1.055913 | 0.023363 | 0.450057 | 32.527512 | 0.474 |
| R5 | 20261325 | 10 | 0.856911 | 0.052066 | 0.990420 | 0.021984 | 0.670267 | 31.114978 | 0.480 |
| R6 | 20261305 | 25 | 1.000000 | 0.006688 | 1.410263 | 0.014650 | 0.552956 | 31.884759 | 0.472 |
| R6 | 20261315 | 25 | 0.984733 | -0.000502 | 0.743863 | 0.015319 | 0.646412 | 32.213229 | 0.472 |
| R6 | 20261325 | 25 | 1.000000 | -0.000547 | 0.904389 | 0.018049 | 0.598398 | 32.745825 | 0.466 |

Raw Q/A/G, bounded-probability Q/A/G, flip metrics, native/frozen D, validation selection values, and per-example evidence are retained in `method_b_matched_results.parquet`, the individual job directories, and their checkpoint artifacts.

## Three-seed aggregates

Values are mean ± sample SD, followed by `[min, max]`. With only three seeds, these are descriptive summaries, not population estimates.

| Regime | B | Qz | Az | Gz | COD |
|---|---|---|---|---|---|
| R2 | 0.971993 ± 0.021027 [0.957511, 0.996111] | 0.001132 ± 0.008713 [-0.004694, 0.011149] | 1.413128 ± 0.133587 [1.289121, 1.554582] | 0.052430 ± 0.013759 [0.037009, 0.063449] | 0.783882 ± 0.080076 [0.733286, 0.876203] |
| R3 | 0.985800 ± 0.022214 [0.960200, 1.000000] | -0.003959 ± 0.004748 [-0.007894, 0.001315] | 1.565087 ± 0.168487 [1.402560, 1.738960] | 0.056092 ± 0.003874 [0.051873, 0.059490] | 0.781511 ± 0.098605 [0.707304, 0.893400] |
| R2-C | 0.971993 ± 0.021027 [0.957511, 0.996111] | 0.001132 ± 0.008713 [-0.004694, 0.011149] | 1.413128 ± 0.133587 [1.289121, 1.554582] | 0.052430 ± 0.013759 [0.037009, 0.063449] | 0.783882 ± 0.080076 [0.733286, 0.876203] |
| R4 | 0.987319 ± 0.019849 [0.964444, 1.000000] | -0.000676 ± 0.005974 [-0.005557, 0.005985] | 1.571148 ± 0.170374 [1.407305, 1.747378] | 0.052378 ± 0.005171 [0.048557, 0.058262] | 0.774921 ± 0.091711 [0.705450, 0.878875] |
| R5 | 0.933985 ± 0.072149 [0.856911, 0.999911] | 0.066374 ± 0.032641 [0.043331, 0.103726] | 1.071366 ± 0.089677 [0.990420, 1.167765] | 0.025299 ± 0.004601 [0.021984, 0.030552] | 0.542515 ± 0.114269 [0.450057, 0.670267] |
| R6 | 0.994911 ± 0.008814 [0.984733, 1.000000] | 0.001880 ± 0.004164 [-0.000547, 0.006688] | 1.019505 ± 0.347795 [0.743863, 1.410263] | 0.016006 ± 0.001801 [0.014650, 0.018049] | 0.599255 ± 0.046734 [0.552956, 0.646412] |

## Method and control comparisons

R5 minus comparator mean gaps are shown below; negative is better for COD or a component gap.

| Comparator | ΔCOD | ΔQ gap | ΔA gap | ΔG gap | Per-seed COD result |
|---|---:|---:|---:|---:|---|
| R2 | -0.241367 | -0.022262 | -0.257307 | +0.018030 | R5 lower in 3/3 seeds |
| R3 | -0.238996 | -0.026734 | -0.256040 | +0.020700 | R5 lower in 3/3 seeds |
| R2-C | -0.241367 | -0.022262 | -0.257307 | +0.018030 | R5 lower in 3/3 seeds |
| R4 | -0.232406 | -0.024553 | -0.249906 | +0.023330 | R5 lower in 3/3 seeds |
| R6 | -0.056740 | -0.020395 | -0.057422 | +0.003745 | R5 lower in 2/3 seeds |

The exact equality of R2-C and R2 at every selected result verifies that its additional detached, zero-weighted intervention forwards did not alter gradients or outcomes. R4 did not preferentially reduce the Q gap. R5 substantially reduced mean Q and A gaps, but increased the mean G gap against every comparator. R6 captured much of the apparent COD gain and beat R5 in seed 20261325, so the semantic specificity of the R5 effect is unresolved.

## Representation comparison

R3 B-matched CKA was `0.525173`, `0.771606`, and `0.767625`; its B-matched COD was `0.707304`, `0.893400`, and `0.743829`. At the final checkpoint, projected cosine and MSE improved in all three seeds, while CKA was mixed and COD remained `0.678913`, `0.666911`, and `0.714180`. R3 mean B-matched COD (`0.781511`) was essentially unchanged from R2 (`0.783882`). Representation alignment therefore did not imply causal-organizational alignment.

## Frozen test and required questions

The frozen primary R5 criterion **failed**: R5 had the lowest mean COD and passed quality controls, but seed 20261325 was not teacher-like in validation B (absolute gap `0.128624`), and R5 did not beat all comparators in every seed.

1. **Does ordinary logit KD achieve teacher-like B while retaining mismatch?** Yes. R2 validation-B gaps were `0.026200`, `0.014528`, and `0.016696`, while COD remained `0.733286`, `0.876203`, and `0.742157`.
2. **Does hidden-state KD reduce the mismatch?** Not materially at the frozen B-matched checkpoints: R3 mean COD was `0.781511` versus R2 `0.783882`.
3. **Does representation similarity improve without equivalent COD improvement?** Yes. R3 representation metrics improved in important views, but its B-matched COD did not improve equivalently and remained far from the teacher.
4. **Does R4 preferentially reduce the Q gap?** No. Mean Q gap increased from R2 `0.167155` to R4 `0.169446`.
5. **Does R5 reduce Q/A/G jointly?** No. Relative to R2, Q and A gaps fell by `0.022262` and `0.257307`, but G gap increased by `0.018030`.
6. **Does R5 reduce COD relative to R2?** Descriptively yes in 3/3 seeds; means `0.542515` versus `0.783882`.
7. **Does R5 reduce COD relative to R3?** Descriptively yes in 3/3 seeds; means `0.542515` versus `0.781511`.
8. **Does R5 beat compute-matched R2-C?** Descriptively yes in COD in 3/3 seeds; means `0.542515` versus `0.783882`. Extra compute alone does not explain that contrast.
9. **Does R5 beat random-response R6?** No robustly. Mean COD favored R5 by `0.056740`, but only 2/3 paired seeds favored R5, and R5's mean G gap was worse.
10. **Is any R5 advantage present in all three seeds?** Its COD advantage over R2, R3, R2-C, and R4 is; its advantage over R6 and its teacher-like behavior are not.
11. **Does R5 preserve general quality?** Yes under the frozen controls: PPL was `31.040166`–`32.527512`, and HellaSwag was `0.474`–`0.480`.
12. **Does behavioral equivalence not imply causal-organizational equivalence survive?** Yes, model/task/site-specifically. R2/R3/R2-C/R4 can be teacher-like in B while retaining large COD.
13. **Is there evidence that causal organization can be explicitly transferred?** There is a descriptive R5 signal in COD, Q, and A, but not clean evidence of semantic-specific transfer: the behavior condition failed in one seed, G did not improve, and R6 explained much of the effect.

## Scientific classification and recommendation

Supported classifications are **behavioral equivalence with causal-organizational mismatch**, **representation similarity without causal-organizational equivalence**, and **results heterogeneous / unresolved** for conversion-response transfer. The data do not support claiming that factorial CRD has transferred causal organization, nor that the random control fully explains the effect.

Recommendation: **revise conversion-response method before confirmation**. Keep E13 confirmation locked. A revision requires a new frozen discovery protocol or other explicitly authorized next step; these discovery results must not be used to tune against the untouched confirmation set.
