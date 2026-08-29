# E13 Response-Objective Revision Protocol

Status: **frozen on 2026-08-29 before R11-R16 implementation or training**. Open discovery only. E13 confirmation remains locked, unmaterialized, and inaccessible.

## Shared identity

R11-R16 retain the exact E13 R2 training identity: Qwen3-1.7B teacher, Qwen3-0.6B student, open corpus, layer/site/token selector, model-local teacher/student geometry, optimizer, LR schedule, 100 updates, effective batch 8, checkpoints 0/10/25/50/100, response coefficient 1.0, seeds, B-matched validation selector, discovery metrics, COD, and general-quality controls. Each run starts from the original student. Cross-model transfer occurs only in scalar response space.

The bounded seed is `20261305`. Validation-derived matrices/scales are computed once from the existing 500-row teacher cache and persisted. Floors and epsilon below are numerical conditioning constants, not tuned outcomes.

## R11 — full arm-response matching

For each model, target-oriented margins are divided by that model's frozen validation margin SD. Let standardized arms be Y00/Y10/Y01/Y11 and define:

```text
r10 = Y10 - Y00
r01 = Y01 - Y00
r11 = Y11 - Y00
```

For each component, compute the teacher population SD on the validation split and floor at `1e-6`. Divide both student and teacher responses by that teacher-derived component scale. The per-row arm loss is the sum, not mean, of the three squared differences:

```text
L_arm = (r10*_S-r10*_T)^2 + (r01*_S-r01*_T)^2 + (r11*_S-r11*_T)^2
L_R11 = L_R2 + mean_rows(L_arm)
```

This preserves the full joint response increment rather than replacing it with the derived factorial interaction.

## R12 — whitened Q/A/G matching

Using teacher validation rows, form standardized `x=[Q_z,A_z,G_z]` and the population covariance `Sigma_T`. Define:

```text
epsilon = 1e-4 * trace(Sigma_T) / 3
Sigma_epsilon = Sigma_T + epsilon I
```

The matrix must be finite, symmetric positive definite, and have its eigenvalues/condition number persisted. For `d=[Qz_S-Qz_T, Az_S-Az_T, Gz_S-Gz_T]`:

```text
L_W = d^T inverse(Sigma_epsilon) d
L_R12 = L_R2 + mean_rows(L_W)
```

No further division by R5 component SDs is applied.

## R13 — Q/A only

Use the existing R5 semantic geometry, targets, and teacher validation component scales for Q and A. G receives no training loss and remains a frozen evaluation outcome.

```text
L_QA = (Q*_S-Q*_T)^2 + (A*_S-A*_T)^2
L_R13 = L_R2 + mean_rows(L_QA)
```

## R14 — G-only auxiliary

Use existing R5 semantic geometry, G target, and teacher validation G scale:

```text
L_G = (G*_S-G*_T)^2
L_R14 = L_R2 + mean_rows(L_G)
```

## Frozen gradient-conflict gate

The read-only Task 1 diagnostic audits full-model gradients on two validation batches at R5 steps 10 and 100. The gate fires if the same response component has `cos(g_KD,g_response) < -0.2` in at least two audits spanning the frozen batches/checkpoints. The completed diagnostic fired this gate for A, so R15 and R16 are authorized for the bounded seed. This decision precedes all revised-method outcomes.

## R15 — projected response gradient

R15 uses the unchanged original R5 mean Q/A/G response loss. Compute full accumulated `g_KD` and `g_response` over the same effective batch before clipping. If their dot product is negative:

```text
g_response' = g_response - (dot(g_response,g_KD)/||g_KD||^2) g_KD
g_update = g_KD + g_response'
```

Otherwise use `g_update=g_KD+g_response`. Apply the existing global gradient clip once to the combined update. Persist pre/post norms, dot products, cosines, and projection decisions per optimizer step.

## R16 — norm-balanced response gradient

R16 uses the unchanged original R5 response target. After accumulation, scale the response gradient to frozen ratio `rho=0.5`:

```text
g_response' = g_response * (0.5*||g_KD|| / max(||g_response||,1e-12))
g_update = g_KD + g_response'
```

Then apply the existing global clip. Persist all scales and norms. A zero/nonfinite norm is a stop condition.

## One-seed evaluation

Run R11-R16 on seed 20261305 for 100 steps and evaluate every checkpoint. R15/R16 are included because the frozen conflict gate fired. Compare B-matched results against R2, R3, R5, and R6. Final-step metrics cannot select a method when the B-matched step differs.

## Frozen candidate selection

Only revised objectives R11-R16 are eligible; specificity controls R7-R10 are not methods. A candidate is eligible only if its seed-20261305 B-matched checkpoint satisfies all:

1. absolute validation B gap <= 0.03;
2. COD < seed-20261305 R5 COD;
3. COD < seed-20261305 R6 COD;
4. G gap <= seed-20261305 R5 G gap + 0.01;
5. finite PPL < 10x R0 PPL and HellaSwag >= R0 accuracy - 0.20.

Select the eligible candidate with lowest COD; ties use smaller G gap, then Q gap, then A gap, then the simpler objective in order R14, R13, R11, R12, R16, R15. The completed bounded seed may be reused as the seed-20261305 member of the full wave only if its entire immutable identity matches exactly.

If no candidate passes, do not run a three-seed wave and conclude that semantic-specific CRD is not ready for confirmation. If a candidate passes, name it `R_best` and complete seeds 20261305/20261315/20261325 without changing its objective.

## Three-seed success condition

`METHOD REVISION SUCCESS` requires all:

- teacher-like validation B in at least 2/3 seeds;
- COD below paired R2 in 3/3;
- COD below paired R6 in at least 2/3;
- mean COD below mean R6;
- at least two mean component gaps among Q/A/G below R6;
- no catastrophic general-quality deterioration.

Otherwise classify `METHOD REVISION INCONCLUSIVE`. The only allowed final recommendations are those specified by the campaign request. No discovery result authorizes confirmation access.

## Artifacts, scheduling, and resume

Use immutable jobs beneath `runs/E13_METHOD_REVISION/`, with campaign manifest, job status, logs, scientific identity, protocol digests, cache digest, model revisions, student reference digests, optimizer/RNG/data cursor state, checkpoints, raw evidence, and `confirmation_accessed: false`. Use one process per detected GPU. OOM changes may only preserve effective batch 8 exactly. Stop on nonfinite values, arm/target mismatch, invalid covariance, gradient-reconstruction mismatch, hook leakage, cache/reference mismatch, or confirmation access.
