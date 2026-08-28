# E01A Full Discovery Summary

Status date: 2026-08-27. Full discovery is complete for Qwen3-0.6B and
Qwen3-1.7B. Confirmation remains locked and was not accessed.

## Question and claim boundary

E01A tests whether changing only a frozen probe-decoded truth coordinate at
`resid_post / layer 17 / last_prompt` changes the model's native Yes-minus-No
answer margin toward an opposite-label source. This can establish causal
sensitivity under the tested intervention. It does not by itself establish
that the unperturbed model endogenously uses exactly the same one-dimensional
coordinate, and this discovery result is not confirmation evidence.

## Completed runs

| Model | Run | Model revision | Directed examples | Matched pairs | Raw rows | Trace rows |
|---|---|---|---:|---:|---:|---:|
| Qwen3-0.6B | `E01A_c6cd215d7bf8` | `c1899de289a04d12100db370d81485cdf75e47ca` | 300 | 150 | 57,600 | 230,400 |
| Qwen3-1.7B | `E01A_821138e998c7` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | 300 | 150 | 57,600 | 230,400 |

Both runs used project commit
`94682d3af7c86753f174dc83b21e1b32c3db0dbc`, the predeclared alpha profile
`[-1, -0.5, 0, 0.25, 0.5, 1, 1.5, 2]`, ten random and ten orthogonal-random
directions, layer-specific downstream probes at layers 17/20/23/27, and 2,000
pair-cluster bootstrap draws. Yes and No were each one tokenizer token (`7414`
and `2308`), so the behavioral result is not a multi-token/first-token proxy
mismatch.

## Contract and integrity checks

| Check | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| intervention-layer probe AUROC | 1.000000 | 0.999511 |
| alpha-zero maximum logit deviation | 0 | 0 |
| post-forward hook-leakage deviation | 0 | 0 |
| maximum target-state relative L2 deviation | 0.002339 | 0.002369 |
| maximum truth-projection relative deviation | 0.007767 | 0.004904 |
| maximum orthogonal-space relative deviation | 0.001493 | 0.001616 |
| maximum random-control norm relative mismatch | 2.58e-15 | 1.05e-15 |
| required-field NaN/Inf check | pass | pass |
| confirmation accessed | no | no |

All raw evidence was written before aggregates. Random/orthogonal edits were
norm-matched per sample and alpha. Uncertainty resampled `pair_id`, preserving
the matched-twin cluster.

## Alpha-1 primary results

Effects are mean changes in counterfactual-oriented Yes-minus-No margin;
positive means movement toward the source label. Random and orthogonal values
average the ten predeclared directions.

| Condition | Qwen3-0.6B effect | Qwen3-1.7B effect |
|---|---:|---:|
| truth coordinate | 0.028333 | 0.662083 |
| random direction | 0.003125 | 0.006292 |
| orthogonal random | 0.005125 | 0.010917 |
| same-label coordinate | 0.005833 | -0.019792 |
| shuffled opposite-label coordinate | 0.022917 | 0.661667 |
| full residual patch (upper bound) | 1.150417 | 4.262708 |

The pair-cluster 95% CIs for the truth-coordinate mean are `[0.019167,
0.037917]` for 0.6B and `[0.632500, 0.695417]` for 1.7B. Actual
base-to-counterfactual flip rates are 0.023333 and 0.043333 respectively.

Truth-minus-control paired contrasts at alpha 1:

| Control | Qwen3-0.6B difference (95% CI) | Qwen3-1.7B difference (95% CI) |
|---|---:|---:|
| random direction | 0.025208 `[0.017624, 0.033709]` | 0.655792 `[0.627820, 0.683630]` |
| orthogonal random | 0.023208 `[0.015582, 0.030917]` | 0.651167 `[0.622372, 0.679858]` |
| same-label coordinate | 0.022500 `[0.012083, 0.032500]` | 0.681875 `[0.656656, 0.708552]` |
| shuffled opposite-label coordinate | 0.005417 `[-0.004583, 0.015833]` | 0.000417 `[-0.024375, 0.025000]` |

The truth treatment therefore beats magnitude-matched direction controls and
the same-label coordinate control in both models. It does **not** beat the
shuffled opposite-label coordinate control in either model. This equivalence
holds across every non-zero alpha: every truth-minus-shuffled confidence
interval includes zero.

## Dose response

| Alpha | Qwen3-0.6B truth effect | Qwen3-1.7B truth effect |
|---:|---:|---:|
| -1 | -0.021667 | -0.589375 |
| -0.5 | -0.004583 | -0.313958 |
| 0 | 0 | 0 |
| 0.25 | 0.008750 | 0.152708 |
| 0.5 | 0.010833 | 0.317083 |
| 1 | 0.028333 | 0.662083 |
| 1.5 | 0.039583 | 1.028333 |
| 2 | 0.058333 | 1.412500 |

Both models show the expected negative-alpha sign reversal and positive-alpha
dose response without intervention explosion. The 0.6B response is small and
sparse: its alpha-1 median is zero. The 1.7B alpha-1 median is 0.625.

## Cross-scale result

At alpha 1, the 1.7B raw truth-coordinate effect is 23.37 times the 0.6B
effect. The exploratory causal-conversion efficiency
`kappa = delta_margin / |delta_truth_coordinate|` is 0.017443 for 0.6B and
0.037977 for 1.7B, a 2.18-fold difference. Mean edit norm is also larger in
1.7B (18.4486 versus 1.6310), so the raw-margin ratio alone is not a
norm-controlled scale comparison; kappa preserves the same ordering.

The truth edit reaches 2.46% of the full-patch upper-bound effect in 0.6B and
15.53% in 1.7B. Together with Phase 0A.2, this yields the discovery pattern:
decodability is nearly saturated at both scales, while conversion from the
decoded coordinate into native answer margin is substantially stronger in
1.7B.

## Supported interpretation

1. The frozen decoded coordinate is causally actionable under intervention in
   both checkpoints relative to random, orthogonal, and same-label controls.
2. The conversion is weak/sparse in 0.6B and substantially stronger in 1.7B.
3. Matched-twin source identity is not supported as the reason for the effect:
   an unrelated opposite-label source produces the same coordinate treatment
   response.
4. The strict specificity statement "matched truth-coordinate treatment beats
   every control" fails because the shuffled-coordinate control is equivalent.
5. These are discovery-only, model/task/site-specific results. Confirmation
   remains untouched.

## Next registered hypothesis

E01B should distinguish coordinate-target causality from matched-source
specificity by predeclaring multiple opposite-label sources per base and
testing whether effect size is explained by the requested coordinate target,
source identity, relation-family matching, or another source attribute. E01B
is proposed only; it is not authorized by completion of E01A.

