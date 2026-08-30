# E01 Cross-Checkpoint Calibration Audit — Results

Status: **complete.** Required by `docs/Reproduction_Reliability_Next_Direction_Review.md`
section 8, before any scale or checkpoint claim is worded.

Campaign: `runs/E01_CALIBRATION_AUDIT/`. Models: `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, bf16.
Site: `resid_post` L17 / `last_prompt` — the frozen E01 site, no layer search.
Rows: 48,000 across 300 discovery-test examples (150 matched pairs) per model.

Nothing was retuned. The consumed E01 confirmation holdout was **not** touched:
`discovery_view` removes confirmation rows entirely and the runner asserts their
absence. Only the intervention's **magnitude parameterization** changed, which is
the audit's whole purpose.

## Headline

The confound the review warned about is **real and measurable**, and the E01
conclusion **survives it**.

```text
E01's own operating point, expressed as residual fraction r = ||dh|| / ||h||:

    Qwen3-0.6B    r = 0.0196
    Qwen3-1.7B    r = 0.0547        <-- 2.79x larger perturbation
```

The frozen opposite-class-median setpoint is a **2.79x larger residual
perturbation in the 1.7B checkpoint than in the 0.6B checkpoint**. The headline
single-point contrast (`Q0` 0.0144 versus 0.7013) was therefore never
norm-matched, and its ~49x ratio must not be quoted as a calibrated effect.

But when both checkpoints are driven at the *same* residual fraction, and again
at the same achieved standardized semantic displacement, the ordering and the
qualitative conclusion hold at every point tested.

## Q(r), A(r), G(r) at matched residual fraction

Mean oriented margin change, pair-cluster bootstrap, 2000 draws, 95%.
`*` marks a CI excluding zero. Shaded region `r <= 0.10` is on-manifold (below).

| estimand | model | r=0.02 | r=0.05 | r=0.10 | r=0.20 | r=0.40 |
|---|---|---:|---:|---:|---:|---:|
| **Q** | Qwen3-0.6B | +0.0262* | +0.0642* | +0.1408* | +0.2765* | +0.5900* |
| **Q** | Qwen3-1.7B | +0.2433* | +0.6256* | +1.3175* | +2.7856* | +5.3850* |
| **A** | Qwen3-0.6B | +0.1604* | +0.3917* | +0.7733* | +1.5402* | +2.8196* |
| **A** | Qwen3-1.7B | +0.3113* | +0.8237* | +1.7721* | +3.8546* | +6.7073* |
| **G** | Qwen3-0.6B | -0.0021 | -0.0046 | -0.0098 | -0.0581* | -0.3202* |
| **G** | Qwen3-1.7B | +0.0123 | +0.0848* | +0.2502* | -0.3523* | -3.4319* |

The 1.7B advantage in `Q` is roughly **9.5x at every matched residual fraction**
(9.3x, 9.7x, 9.4x at r = 0.02, 0.05, 0.10) — not the ~49x the uncalibrated
single point suggested, but stable, large, and always in the same direction.

## Second ruler: matched achieved semantic displacement

At matched `r` the *smaller* model actually receives the **larger** standardized
coordinate push, because its validation coordinate SD is much smaller
(`sigma_q` = 0.870 versus 9.709):

| r | 0.6B achieved `dq_z` | 1.7B achieved `dq_z` |
|---:|---:|---:|
| 0.02 | 1.90 | 0.69 |
| 0.05 | 4.74 | 1.72 |
| 0.10 | 9.48 | 3.43 |
| 0.20 | 18.96 | 6.87 |
| 0.40 | 37.92 | 13.73 |

Matching on `dq_z` instead (linear interpolation, no extrapolation) makes the
contrast **larger**, not smaller: the `Q` ratio is ~27x across the whole
overlapping range, and the ordering is preserved at every interpolated point.
`A` shows a stable ~6x ratio. So the checkpoint difference is not an artifact of
pushing the larger model harder — under the semantic ruler the larger model is
pushed *less* and still converts far more.

## On-manifold validity — which part of the grid is trustworthy

k-NN distance to the validation activation cloud, edited versus clean, and
diagonal-covariance Mahalanobis. Frozen acceptance band: k-NN ratio <= 1.35.

| r | 0.6B kNN ratio | 1.7B kNN ratio | on-manifold | `||dh||` / between-class distance |
|---:|---:|---:|---|---:|
| 0.02 | 1.01 | 1.01 | yes | 0.08-0.10 |
| 0.05 | 1.08 | 1.07 | yes | 0.20-0.24 |
| 0.10 | 1.32 | 1.25 | yes | 0.40-0.49 |
| 0.20 | 2.00 | 1.78 | **no** | 0.79-0.97 |
| 0.40 | 3.63 | 3.06 | **no** | 1.58-1.94 |

The trustworthy region is `r <= 0.10`, which brackets both checkpoints' original
operating points (0.0196 and 0.0547). Restricted to it, the checkpoint ordering
is preserved for **all three** estimands:

| estimand | r=0.02 | r=0.05 | r=0.10 |
|---|---:|---:|---:|
| Q (1.7B - 0.6B) | +0.2171 | +0.5615 | +1.1767 |
| A (1.7B - 0.6B) | +0.1508 | +0.4321 | +0.9987 |
| G (1.7B - 0.6B) | +0.0144 | +0.0894 | +0.2600 |

**The `G` sign reversal at r >= 0.20 is an off-manifold artifact, not a model
property.** It appears only where edited states sit 1.8-3.6x further from the
activation cloud than clean states and where the edit exceeds the natural
between-class distance. E01B-3's confirmed positive `G` for 1.7B (estimate
0.1328) is reproduced at the comparable on-manifold operating point
(audit `G` = +0.0848 at r = 0.05, CI excludes zero).

## Integrity

* no-op maximum margin deviation: **exactly 0.0** in both models;
* achieved residual fraction error: `1.1e-16` (0.6B), `1.7e-16` (1.7B) — every
  single-component arm is norm-matched by construction;
* frozen probe recipe reproduces E01's decodability: AUROC 1.00000 (0.6B),
  0.99951 (1.7B);
* controls present at every `r`: random, random-orthogonal, same-label context,
  shuffled context, full-patch direction;
* pair-cluster bootstrap throughout; discovery rows only; `confirmation_accessed: false`.

## Verdict and required wording changes

1. **The single-point ratio must be retired.** "0.0144 versus 0.7013" is measured
   at two different residual fractions (2.79x apart) and its ~49x ratio is not a
   calibrated quantity. Report the curve.
2. **The checkpoint difference is real and survives calibration.** Under matched
   residual fraction it is ~9.5x; under matched standardized semantic
   displacement ~27x. It is ruler-dependent in *magnitude* but not in *direction*.
3. **Keep "scale" as an interpretation, not a conclusion.** Two checkpoints are
   not a randomized manipulation of parameter count. The defensible sentence is:
   *at matched intervention magnitude the tested 1.7B checkpoint converts a
   semantic coordinate perturbation into native readout change roughly an order
   of magnitude more effectively than the tested 0.6B checkpoint.*
4. **Restrict all `Q/A/G` claims to `r <= 0.10`.** Beyond that the edits leave the
   activation manifold and effects become magnitude artifacts.
5. **A secondary result worth keeping:** within-model, the 0.6B checkpoint relies
   far more on structured orthogonal state than on the scalar coordinate
   (`A/Q` = 6.1 at r = 0.05) than the 1.7B checkpoint does (`A/Q` = 1.32). This is
   robust to normalization and sharpens the distributed-actionability claim.

The audit is descriptive discovery. It makes no confirmatory claim and does not
re-open E01.
