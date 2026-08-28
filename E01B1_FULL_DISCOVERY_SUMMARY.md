# E01B-1 Full Discovery Summary

Status date: 2026-08-28. Full discovery is complete for Qwen3-0.6B and
Qwen3-1.7B. Confirmation remains locked and was not accessed.

## Question and claim boundary

E01B-1 tests whether directly setting the frozen decoded truth coordinate to
validation-defined scalar targets at `resid_post / layer 17 / last_prompt`
causally changes the native Yes-minus-No answer margin without any donor hidden
state. It can establish source-free sufficiency under this intervention. It
does not establish endogenous natural use of exactly this axis, universality
across tasks/models/sites, or confirmation-level evidence.

## Completed runs

| Model | Run | Model revision | Directed examples | Pairs | Raw rows | Trace rows |
|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | `E01B1_e1169f3ffe11` | `c1899de289a04d12100db370d81485cdf75e47ca` | 300 | 150 | 8,400 | 33,600 |
| Qwen3-1.7B | `E01B1_5b9d70c8cffe` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 300 | 150 | 8,400 | 33,600 |

Both runs used project commit
`4ecac5a94daa5cd41f2c9a19f8cbcd5def0215f2`, ten random directions, ten
orthogonal-random directions, 2,000 pair-cluster bootstrap draws, and trace
layers 17/20/23/27. Yes and No were distinct single tokenizer tokens (`7414`
and `2308`). Every base received all five validation-pooled grid targets.

## Validation-only targets

| Model | q0* | q1* | Q05 | Q25 | Q50 | Q75 | Q95 | sigma q | sigma margin |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | -3.145819 | -1.540436 | -3.516775 | -3.145578 | -2.213101 | -1.540737 | -0.983606 | 0.870186 | 1.386427 |
| Qwen3-1.7B | -3.195017 | 14.815494 | -8.341544 | -3.194676 | 5.353255 | 14.813350 | 18.201304 | 9.709164 | 3.119307 |

Each target and standardization scale used exactly 300 validation examples
(150 per label). Discovery-test labels did not construct targets or scales.

## Contract and integrity checks

| Check | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| L17 probe AUROC | 1.000000 | 0.999511 |
| no-op maximum selected-logit deviation | 0 | 0 |
| post-forward hook-leakage deviation | 0 | 0 |
| maximum target-state relative L2 deviation | 0.002099 | 0.002158 |
| maximum projection deviation / validation sigma q | 0.030660 | 0.012754 |
| maximum orthogonal-space relative deviation | 0.001463 | 0.001549 |
| maximum control-norm relative mismatch | 3.00e-16 | 3.99e-16 |
| required-field and grid-metric NaN/Inf check | pass | pass |
| confirmation accessed | no | no |

Raw evidence was written before aggregates. Control norms were matched to the
opposite-class semantic edit per example, not by a global mean. Uncertainty
resampled `pair_id`, preserving the matched semantic cluster.

## Opposite-class setpoint result

Effects are target-oriented changes in native Yes-minus-No margin. Positive
means movement toward the opposite-class validation median.

| Metric | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| mean effect | 0.026250 | 0.666042 |
| median effect | 0 | 0.750000 |
| pair-cluster 95% CI | [0.016667, 0.035833] | [0.641042, 0.692302] |
| actual base-to-target flip rate | 0.023333 | 0.043333 |
| expected-target rate after | 0.416667 | 0.503333 |
| mean target-oriented delta q z | 1.858219 | 1.876993 |
| mean delta margin z | 0.018934 | 0.213522 |
| mean exploratory kappa z | 0.010773 | 0.117800 |
| mean edit norm | 1.617496 | 18.224541 |
| mean edit/base norm | 0.019601 | 0.054662 |

The source-free effect is 25.37 times larger in raw margin and 11.28 times
larger after validation-only output standardization in 1.7B. Exploratory
`kappa_z` preserves the ordering with a 10.93-fold difference.

## Controls

Random and orthogonal effects below average the ten predeclared directions.

| Condition | Qwen3-0.6B effect | Qwen3-1.7B effect |
|---|---:|---:|
| source-free opposite-class median | 0.026250 | 0.666042 |
| same-class median | -0.008750 | 0.024792 |
| random direction | 0.005042 | 0.005896 |
| orthogonal random | 0.004958 | 0.007083 |
| no-op | 0 | 0 |

Paired source-free-minus-control contrasts:

| Control | Qwen3-0.6B difference (95% CI) | Qwen3-1.7B difference (95% CI) |
|---|---:|---:|
| random direction | 0.021208 `[0.013416, 0.028834]` | 0.660146 `[0.636872, 0.685292]` |
| orthogonal random | 0.021292 `[0.013666, 0.028875]` | 0.658958 `[0.635956, 0.682607]` |

The source-free semantic treatment beats both norm-matched direction controls
in both checkpoints.

## Continuous validation-grid response

Mean native margin by frozen target:

| Model | Q05 | Q25 | Q50 | Q75 | Q95 |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.865417 | 0.878333 | 0.893333 | 0.897083 | 0.903333 |
| Qwen3-1.7B | 3.596250 | 3.801458 | 4.123750 | 4.487917 | 4.619167 |

| Metric | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| median within-example Spearman | 0.223607 | 1.000000 |
| fraction Spearman positive | 0.513333 | 1.000000 |
| fraction Spearman >= 0.8 | 0.116667 | 0.990000 |
| fraction monotonically nondecreasing | 0.420000 | 0.936667 |
| within-base centered slope | 0.014022 | 0.038412 |
| pair-cluster 95% slope CI | [0.010617, 0.017646] | [0.036730, 0.040176] |

Both population slopes are positive with confidence intervals above zero.
Per-example monotonicity is strong in 1.7B but weak/sparse in 0.6B; E01B-1
therefore does not support a uniformly monotonic 0.6B response.

## Downstream source-free trajectory

Validation-standardized, target-oriented trace changes for the opposite-class
median treatment:

| Model | Layer | mean delta q z | mean delta native-margin z |
|---|---:|---:|---:|
| Qwen3-0.6B | 17 | 1.858219 | 0.073742 |
| Qwen3-0.6B | 20 | 1.160191 | 0.001853 |
| Qwen3-0.6B | 23 | 0.785002 | 0.011344 |
| Qwen3-0.6B | 27 | 0.497280 | 0.018984 |
| Qwen3-1.7B | 17 | 1.876993 | 0.396126 |
| Qwen3-1.7B | 20 | 1.131529 | 0.193353 |
| Qwen3-1.7B | 23 | 0.944992 | 0.310404 |
| Qwen3-1.7B | 27 | 0.687993 | 0.213666 |

The injected standardized coordinate is nearly identical at L17 and L20.
The 1.7B checkpoint retains more at L23/L27, but its much larger native-margin
change is already present at L17 and persists throughout depth. This reproduces
E01A's mixed bottleneck dominated by readout conversion.

## E01A comparison and supported interpretation

E01A alpha-1 donor-derived effects were 0.028333 (0.6B) and 0.662083 (1.7B).
E01B-1 source-free effects are 0.026250 and 0.666042. The close agreement,
together with successful semantic-vs-control contrasts, supports these bounded
discovery conclusions:

1. Donor hidden states are unnecessary for the measured coordinate-only effect.
2. Validation-defined scalar setpoints are causally sufficient under this
   intervention in both checkpoints.
3. Conversion is weak/sparse in 0.6B and substantially stronger in 1.7B.
4. A strong continuous per-example setpoint response is supported in 1.7B,
   while 0.6B shows only a small positive population-level slope.
5. Source-free tracing reproduces the mixed/readout-dominated E01A mechanism.

These claims remain discovery-only and model/task/site specific. They do not
show that source identity or orthogonal context is irrelevant when orthogonal
information is explicitly allowed to enter.

## Next registered question

E01B-2 should hold the source-free scalar displacement fixed while varying
orthogonal context to test context-sensitive utilization. It remains proposed
and requires separate authorization. Confirmation remains untouched.
