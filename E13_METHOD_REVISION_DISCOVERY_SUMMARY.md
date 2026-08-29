# E13 Method-Revision Discovery Summary

Status date: 2026-08-29. Open discovery only. E13 confirmation (`e13_confirmation_v1`) was **not** accessed, inspected, or materialized at any point in this campaign.

Campaign: `runs/E13_METHOD_REVISION/E13MR_08b45d7912c2_79d9141a25b4`
Specificity protocol SHA-256: `08b45d7912c2110b87fb2915b983dcc9e4d2b66a0c961efa2bb8c46438d078ba`
Objective protocol SHA-256: `79d9141a25b420ed6b8fcd290297384cb9d31e920d53dec634bb0462b6a180b0`
Schedulers: `20260829T_revision_wave_v1` (complete, `selected_regime: null`), `20260829T_revision_wave_v2` (duplicate resume scheduler, retained as provenance)

**Frozen bounded gate result: no candidate passed. `METHOD REVISION INCONCLUSIVE`. No three-seed method wave was run.**

## What the campaign asked

R5 (factorial conversion-response distillation) had the lowest mean COD of the earlier method wave but did not clearly beat R6, its own random-response control. This campaign asked why, and whether a semantic-specific causal-transfer signal can be isolated. Fourteen regimes were compared at validation-selected behavior-matched checkpoints on seed `20261305`: four mechanism controls (R7-R10), six objective revisions (R11-R16), and the four existing comparators (R2, R3, R5, R6).

## Bounded comparison, seed 20261305

| regime | step | B | B gap | Qz | Az | Gz | COD | profile r | Q sign | PPL | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher | — | 0.964933 | — | 0.163981 | 1.168644 | 0.046239 | 0.000000 | 1.0000 | 1.000 | 23.371 | 0.578 |
| R0 | 0 | 0.747000 | — | -0.000147 | 0.787599 | 0.006628 | 0.799059 | — | — | 30.574 | 0.470 |
| R2 | 10 | 0.957511 | 0.026200 | 0.011149 | 1.289121 | 0.037009 | 0.733286 | 0.7852 | 0.600 | 31.310 | 0.472 |
| R3 | 25 | 1.000000 | 0.026392 | -0.007894 | 1.738960 | 0.051873 | 0.707304 | 0.8147 | 0.637 | 32.272 | 0.466 |
| R5 | 10 | 0.945133 | 0.016528 | 0.103726 | 1.167765 | 0.030552 | 0.507220 | 0.8570 | 0.903 | 31.040 | 0.480 |
| R6 | 25 | 1.000000 | 0.026376 | 0.006688 | 1.410263 | 0.014650 | 0.552956 | 0.8427 | 0.823 | 31.885 | 0.472 |
| R7 | 25 | 0.992911 | 0.017360 | 0.019089 | 0.940671 | 0.009014 | 0.545301 | 0.8509 | 0.850 | 32.172 | 0.472 |
| R8 | 25 | 0.999200 | 0.025840 | 0.027335 | 1.255196 | 0.008542 | **0.454964** | 0.8858 | 0.867 | 31.991 | 0.472 |
| R9 | 100 | 0.810711 | 0.164744 | -0.005440 | 0.161380 | 0.038078 | 1.246526 | 0.2123 | 0.800 | 31.989 | 0.470 |
| R10 | 10 | 0.960222 | 0.020976 | 0.013558 | 1.286000 | 0.015441 | 0.827450 | 0.7308 | 0.793 | 31.245 | 0.474 |
| R11 | 10 | 0.924889 | 0.054824 | 0.069604 | 1.375384 | 0.016357 | 0.635353 | 0.8304 | 0.930 | 30.865 | 0.480 |
| R12 | 10 | 0.880311 | 0.103744 | 0.030847 | 1.191939 | 0.003856 | 0.768399 | 0.7474 | 0.830 | 30.756 | 0.480 |
| R13 | 50 | 0.912311 | 0.059552 | 0.100713 | 1.094943 | 0.047692 | 0.594429 | 0.8066 | 0.970 | 31.923 | 0.480 |
| R14 | 10 | 0.959133 | 0.022536 | 0.009564 | 1.329864 | 0.031284 | 0.697306 | 0.7902 | 0.760 | 31.136 | 0.474 |
| R15 | 50 | 1.000000 | 0.026392 | 0.052866 | 0.643558 | 0.008404 | 0.667481 | **0.8917** | 0.970 | 32.734 | 0.476 |
| R16 | 25 | 1.000000 | 0.026392 | 0.066886 | 0.820997 | 0.032750 | 0.595298 | 0.8639 | **0.980** | 32.005 | 0.474 |

`B gap` is the absolute validation B gap against the teacher's `0.973608`. `profile r` is the per-example Pearson correlation of the full `(Q_z, A_z, G_z)` profile; `Q sign` is per-example Q sign agreement. Detail is in `E13_RESPONSE_SPECIFICITY_RESULTS.md` and `E13_RESPONSE_OBJECTIVE_REVISION_RESULTS.md`.

## Existing three-seed comparators, unchanged

| regime | mean COD | mean Qz | mean Az | mean Gz | teacher-like B seeds |
|---|---:|---:|---:|---:|---:|
| R2 | 0.783882 | 0.001132 | 1.413128 | 0.052430 | 3/3 |
| R3 | 0.781511 | -0.003959 | 1.565087 | 0.056092 | 3/3 |
| R5 | 0.542515 | 0.066374 | 1.071366 | 0.025300 | 2/3 |
| R6 | 0.599255 | 0.001880 | 1.019505 | 0.016006 | 3/3 |

## Frozen gate outcome

No regime among R11-R16 satisfied the frozen eligibility rule. All six failed criterion 2 (COD below R5's `0.507220`) and criterion 3 (COD below R6's `0.552956`); R11, R12 and R13 also failed criterion 1 (absolute validation B gap at most `0.03`); R12 also failed criterion 4. Every regime passed criterion 5 — general quality never deteriorated catastrophically anywhere in the wave (PPL `30.76`-`32.73` against R0's `30.574`, HellaSwag `0.466`-`0.480` against R0's `0.470`).

Per the frozen protocol, this triggers the stop condition: **do not run a three-seed method wave; conclude that the current semantic-specific CRD method is not ready for confirmation.**

## Why R6 improves COD

Plain logit KD does not fail to move the student's causal organization — it moves it too far in the A direction. R2 drives `A_z` from the untrained `0.788` up to `1.757` by step 25, against a teacher value of `1.169`. Because COD is a mean per-example Euclidean norm of the `(Q_z, A_z, G_z)` gap and A is by far the largest component (teacher train SD `0.679` versus Q `0.072` and G `0.098`), COD is essentially an A-overshoot meter at these checkpoints.

Any structured penalty on the student's intervention response acts as a brake on that overshoot. The Task 1 diagnostic showed R6 cutting the B-matched matched-context mean-square response from R2's `2.872` to `1.201` and the unseen-random Jacobian proxy from `0.0095` to `0.0036`. R6's cached targets are not sample-wise close to the semantic ones (train correlations: Q `0.103`, A `0.058`, G `0.105`; response-norm `0.256`), so it cannot be transferring the teacher's causal profile. It is supplying generic local-sensitivity regularization that happens to counteract the specific failure mode COD measures. This is hypothesis **H-A**, and it is the campaign's best-supported explanation.

The specificity controls confirm it directly. R8, which permutes teacher targets within relation family so that no sample ever receives its own causal response, achieved the **lowest COD in the entire campaign** (`0.454964`) while staying teacher-like in behavior. R7, permuted globally, reached `0.545301`. Destroying sample-level semantic correspondence did not destroy — and here slightly improved — the COD benefit.

## Where the semantic signal actually is

The effect is not zero; it is in Q, and COD is not sensitive enough to reward it.

Per-example Q sign agreement rises in two distinct steps. From plain KD to any response regularization: R2 `0.600` to R6 `0.823`. Then a further, smaller step that only appears when the teacher's own sample-matched semantic targets are used: R5 `0.903`, R13 `0.970`, R15 `0.970`, R16 `0.980`. The shuffled controls sit in between (R7 `0.850`, R8 `0.867`), exactly as a partially destroyed correspondence signal should. Every regime whose loss contains a sample-matched semantic Q target reaches at least `0.903`; every regime without one stays at or below `0.867`. R5 is also the only specificity regime that moves `Q_z` materially toward the teacher's `0.163981`.

So sample-specific teacher causal response does transfer something real and measurable — but it lands on the smallest component of the profile, while the frozen selection statistic is dominated by the largest. R15 has the best per-example profile correlation in the campaign (`0.8917`) and is still ineligible, because balancing the gradient left `A_z` undershooting at `0.643558`.

## Why the objective revisions failed

Scale, not form. Mean response-to-KD loss ratios across the wave were R14 `2.19`, R15 `3.11`, R16 `3.98`, R13 `7.13`, R11 `8.55`, R12 `22.04`, and this ordering predicts the behavior failures: the three regimes that lost teacher-like behavior are the three with the largest response terms. At the frozen coefficient of 1.0 — inherited from R5 rather than re-derived per objective — these are not auxiliary losses but the dominant one. Full-arm matching (R11) and whitening (R12) both made this worse by construction, whitening catastrophically so, since inverting the teacher covariance amplifies precisely the low-variance, 25%-near-zero G direction.

Gradient conflict turned out to be a minor factor. R15's in-training measurement over all 100 optimizer steps gives mean `cos(g_KD, g_response) = -0.0090`, median `-0.0092`, negative on 55% of steps but below `-0.2` on only 15%. The same trace shows the response gradient running `3.1`-`3.4` times the KD gradient norm. The frozen two-batch gate that authorized R15/R16 fired on a genuine but unrepresentative tail. Of the two remedies, norm-balancing (R16) clearly beat projection (R15): best COD of the six objective variants at `0.595298` with validation B held at `1.000000`.

## Required questions

1. **Why did R6 improve COD?** Generic local-sensitivity regularization that brakes logit KD's A-overshoot, not causal transfer. Its targets are near-uncorrelated with the semantic ones (A `0.058`), and it cuts the matched-context sensitivity proxy from `2.872` to `1.201`.
2. **Is R5's effect semantic-specific?** Only partly, and not in the COD summary. Shuffled-target controls match or beat R5 on COD (R8 `0.454964` versus R5 `0.507220`), so the COD gain is not semantic. A semantic-specific effect is real but confined to Q sign agreement and `Q_z`.
3. **Does sample-specific teacher causal response matter?** Yes for Q alignment (`0.903`-`0.980` with sample-matched targets versus at most `0.867` without), no for COD. It is also not sufficient: the matched target must be paired with matching intervention geometry, or the run degrades (R9 collapsed, R10 fell below plain KD).
4. **Is the current G target learnable?** Yes, individually. R14 (G only) is the best-behaved regime in the wave and gives the smallest G gap of any objective variant (`0.080236`, better than R5's `0.112094`) without destabilizing behavior. But it moves overall organization very little (COD `0.697306`).
5. **Does a revised objective improve G?** Marginally, and only by targeting G directly. Regimes training on G reach G gaps of `0.080`-`0.084`; R13, which omits G, reaches `0.100125` with `G_z` drifting past the teacher to `0.0785`. Improving Q and A does not carry G along — G is a separate transfer problem.
6. **Does a revised method beat ordinary KD?** On COD, yes for several (R16 `0.595298`, R13 `0.594429` versus R2 `0.733286`), and R16 does so while holding validation B at `1.000000`. But none beats R5 or R6, so none is eligible.
7. **Does it beat hidden-state KD?** Same picture. R3's `0.707304` is barely below R2's `0.733286`; representation alignment still does not deliver causal-organizational alignment.
8. **Does it beat random-response regularization?** No. Not one of R11-R16 got below R6's `0.552956`. This is the campaign's central negative result.
9. **Does behavioral equivalence still coexist with causal-organizational mismatch?** Yes, robustly. R6, R15 and R16 all reach validation B of `1.000000` — behaviorally indistinguishable from the teacher on the selection metric — while retaining COD of `0.553`, `0.667` and `0.595` against a teacher value of `0`. The core E13 dissociation is unchanged and, if anything, better demonstrated.
10. **Is there sufficient evidence that causal organization itself can be transferred?** No. The COD improvements attributed to conversion-response distillation are reproduced, and exceeded, by targets with no sample-level semantic content. The one component with a genuine semantic-specific signature (Q) is too small to move the frozen summary statistic. Causal-organization transfer is not demonstrated by this method family under this design.

## Verdict and recommendation

```text
METHOD REVISION INCONCLUSIVE
```

Recommendation, one of the three permitted options:

```text
stop method branch and write E13 as diagnostic evidence
```

The diagnostic result is strong and replicated: behavioral equivalence coexists with large causal-organizational mismatch, and hidden-state KD does not repair it. The method branch is not close to a confirmation-ready claim. It failed its own preregistered gate on all six revisions, and the mechanism controls showed that its apparent success statistic is not measuring semantic transfer at all. Continuing to iterate objectives against COD would be optimizing a statistic now known to be dominated by response-magnitude regularization.

If the method branch is ever resumed, this campaign identifies the two changes it would need, neither of which is a small tweak: a response coefficient calibrated per objective against the KD gradient scale rather than fixed at 1.0, and a selection statistic sensitive to per-example profile correspondence (profile correlation, per-component sign agreement) rather than a magnitude-dominated norm. Both would require a new frozen discovery design. Neither is authorized here.

E13 confirmation remains locked, unmaterialized, and unaccessed.
