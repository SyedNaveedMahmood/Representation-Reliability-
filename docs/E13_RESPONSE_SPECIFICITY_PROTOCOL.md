# E13 Response-Specificity Ablation Protocol

Status: **frozen on 2026-08-29 before R7-R10 implementation or training**. Open discovery only. E13 confirmation remains locked, unmaterialized, and inaccessible.

## Question and shared identity

This protocol asks whether R5's improvement requires sample-specific semantic teacher responses or follows from generic structured response regularization. R7-R10 use the exact E13 teacher/student revisions, 4,000/500/300 open splits, layer 17 `resid_post` at `last_prompt`, student seeds, R2 objective, optimizer, 100 updates, effective batch 8, checkpoint schedule, validation-only B selection, causal evaluation, COD, and quality controls already frozen for R5/R6. Every model uses its own intervention geometry; only scalar response targets cross models.

The bounded specificity seed is exactly `20261305`. Each regime starts independently from the original Qwen3-0.6B student. The response coefficient is 1.0. No result-dependent coefficient, permutation, scale, checkpoint, or direction choice is permitted.

## Deterministic pair-preserving permutations

The common permutation seed is `20261331`. A permutation operates on counterfactual pair IDs, never individual directed rows. The pair order is the frozen training-table first-occurrence order. NumPy `PCG64(seed)` permutes that order; if any pair maps to itself, the permuted list is deterministically cyclically shifted until no fixed points remain. A target row receives the mapped pair's row with the same gold-label orientation. This preserves pair grouping, both directed labels, and every marginal component distribution exactly while breaking sample correspondence.

R8 applies the same algorithm independently within each relation family, using a generator seeded by the first eight bytes of SHA-256 over `20261331|R8|relation_family`. Each family must contain at least two pairs and must yield a fixed-point-free permutation. The mapping and its SHA-256 digest are persisted and audited.

## R7 — globally shuffled semantic targets

Student interventions are exactly R5 semantic and matched-context interventions. Teacher R5 Q/A/G targets are taken from the globally permuted same-label rows. Teacher validation component scales remain the frozen R5 validation scales.

```text
L_R7 = L_R2 + mean[(Q*_S-Q*_{T,perm})^2
                   +(A*_S-A*_{T,perm})^2
                   +(G*_S-G*_{T,perm})^2] / 3
```

## R8 — relation-family shuffled semantic targets

R8 is identical to R7 except that the fixed-point-free pair permutation is performed within relation family. It preserves family-level target distributions and scale while destroying sample-level correspondence.

## R9 — semantic student geometry with random teacher targets

Student Q/A/G are produced with R5 semantic and matched-context geometry. Targets are the same row's cached R6 random Q/A/G scalars. Both student values and teacher targets are divided by the frozen R6 validation component scales. This isolates semantic student geometry from semantic teacher targets.

```text
L_R9 = L_R2 + mean[(Q_S-Q_{T,random})^2/s_Q,R6^2
                   +(A_S-A_{T,random})^2/s_A,R6^2
                   +(G_S-G_{T,random})^2/s_G,R6^2] / 3
```

## R10 — random student geometry with semantic teacher targets

Student responses use the exact R6 random-q/random-context geometry. Targets are the same row's cached R5 semantic Q/A/G scalars. Both are divided by the frozen R5 validation component scales. This is an intentionally mismatched negative control.

## Bounded evaluation and interpretation

Run R7-R10 for seed 20261305 through all 100 steps and evaluate the full frozen stack. Compare B-matched R5/R6/R7/R8/R9/R10 on B, Qz, Az, Gz, component gaps, COD, PPL, and HellaSwag.

- If R7 and R8 approximate R5, sample-specific semantic response correspondence is not necessary.
- If R9 approximates R6, target identity dominates student intervention geometry.
- If R10 performs well, scale/local-sensitivity regularization can explain improvement even under geometry/target mismatch.

R7-R10 are mechanism controls, not candidate revised methods, and are never eligible to become `R_best`. They are not automatically promoted to additional seeds. A later multi-seed control requires an explicit frozen gate based on a material one-seed distinction; this campaign authorizes no such automatic promotion.

## Integrity and stopping

Persist permutation maps/digests, target-source IDs, pair IDs, family labels, cache/protocol digests, model-local references, raw per-example evidence, and `confirmation_accessed: false`. Stop on a fixed point, pair/label/family mismatch, marginal-distribution mismatch, cache mismatch, nonfinite response, hook leakage, cross-space geometry, or confirmation access. Resume only from an exact atomic checkpoint identity.
