# E01B-3 Additive-vs-Gating Factorial Decomposition — Full Discovery

Date: 2026-08-28

Status: full discovery complete; confirmation locked and not accessed.

## Question

E01B-2 showed that structured orthogonal context changes behavior while the
probe-defined semantic coordinate is fixed. E01B-3 separates that context
increment into:

```text
A_c = Y01(c) - Y00
G_c = (Y11(c) - Y10) - (Y01(c) - Y00)
```

`A_c` is an independent context-only causal effect. `G_c` is the factorial
q-by-context interaction: the change in efficacy of the identical source-free
semantic setpoint when context is present.

## Runs and frozen design

| Model | E01B-3 run | Reused E01B-2 Y00/Y10/Y11 |
|---|---|---|
| Qwen3-0.6B | `E01B3_84ee4bae8564` | `E01B2_f2d75dab1eba` |
| Qwen3-1.7B | `E01B3_e7462847a558` | `E01B2_e2b4b02cb3a4` |

Each run used all 150 discovery pairs / 300 directed examples, lambdas 0.5 and
1.0, the four frozen structured context plans, ten frozen random orthogonal
seeds, 2,000 pair-cluster bootstrap draws, and traces at L17/L20/L23/L27. Only
the Y01 context-only arm required new full-set forwards.

## Integrity

- both statuses are `complete`;
- all required raw, trace, factorial, aggregate, contrast, relation, manifest,
  and summary artifacts are present;
- each run has 8,400 factorial rows and 33,600 factorial trace rows;
- compatibility, source-plan, and confirmation-access mismatch counts are zero;
- Y00, Y10, and Y11 contract-batch reproduction deviations are exactly zero;
- context norm relative mismatch and hook leakage are exactly zero;
- maximum `|context dot u|` is `4.44e-16` for 0.6B and `1.55e-15` for 1.7B;
- maximum context-only q drift is `0.0253` and `0.00919` validation standard
  deviations, below the frozen `0.05` gate;
- all raw/factorial values are finite;
- confirmation access is false in statuses, manifests, raw rows, and traces.

## Lambda-1 decomposition

### Qwen3-0.6B

| Context | A context-only (95% CI) | G interaction (95% CI) | E01B-2 increment A+G |
|---|---:|---:|---:|
| matched | 1.1208 [0.9450, 1.2946] | 0.0063 [-0.0054, 0.0179] | 1.1271 |
| same-family | 0.8571 [0.7233, 0.9980] | 0.0071 [-0.0067, 0.0217] | 0.8642 |
| different-family | 0.6179 [0.5196, 0.7150] | 0.0021 [-0.0104, 0.0146] | 0.6200 |
| same-label | 0.1038 [0.0137, 0.1988] | 0.0017 [-0.0129, 0.0154] | 0.1054 |
| random | -0.0196 [-0.0281, -0.0109] | -0.0005 [-0.0104, 0.0089] | -0.0200 |

The opposite-label structured A effects and their structured-minus-random
contrasts are positive. Every G interval and every structured-minus-random G
interval includes zero. The E01B-2 context effect is therefore overwhelmingly
an additive multidimensional signal in 0.6B, not evidence of q gating.

### Qwen3-1.7B

| Context | A context-only (95% CI) | G interaction (95% CI) | E01B-2 increment A+G |
|---|---:|---:|---:|
| matched | 3.4673 [3.1285, 3.7963] | 0.1456 [0.1215, 0.1698] | 3.6129 |
| same-family | 2.9179 [2.6354, 3.2096] | 0.1625 [0.1340, 0.1902] | 3.0804 |
| different-family | 2.2871 [2.1125, 2.4592] | 0.1960 [0.1594, 0.2317] | 2.4831 |
| same-label | 0.0200 [-0.1369, 0.1655] | 0.0269 [0.0023, 0.0510] | 0.0469 |
| random | 0.0490 [0.0309, 0.0664] | 0.0148 [-0.0003, 0.0291] | 0.0639 |

Opposite-label structured A and G effects are positive. Structured-minus-random
G contrasts exclude zero: matched `0.1308 [0.1113, 0.1510]`, same-family
`0.1477 [0.1232, 0.1728]`, and different-family `0.1812 [0.1511, 0.2114]`.
The 1.7B result is mixed additive plus structured gating, with A still accounting
for most of the total context increment.

## Lambda-0.5 sensitivity

The qualitative result is unchanged. In 0.6B, matched/same-family/different-
family A is `0.5583`/`0.4296`/`0.3038`, while all three G intervals include
zero. In 1.7B, A is `1.5631`/`1.3050`/`1.0333`, and G is
`0.1800`/`0.1592`/`0.1404`, with every interval excluding zero. The finding
does not depend on lambda 1 alone.

## Relation structure

At lambda 1, additive matched-minus-different-family and same-family-minus-
different-family contrasts are positive in both models:

| Model | A matched - different | A same-family - different |
|---|---:|---:|
| 0.6B | 0.5029 [0.3829, 0.6208] | 0.2392 [0.1333, 0.3483] |
| 1.7B | 1.1802 [0.9054, 1.4340] | 0.6308 [0.4106, 0.8577] |

This ordering does not extend to G. In 1.7B, different-family G is larger than
matched and same-family G: matched-minus-different is `-0.0504 [-0.0775,
-0.0221]`, and same-family-minus-different is `-0.0335 [-0.0625, -0.0075]`.
Accordingly, nuisance/relation compatibility structures the additive signal,
not a monotonic positive gating hierarchy. Family-stratified estimates remain
heterogeneous, especially for `north_south`.

## Trace decomposition

At L17, all context-only and interaction q changes remain approximately zero by
construction. Structured A nevertheless enters the native readout immediately:

| Model | Context | L17 A_q_z | L17 A_margin_z | L17 G_q_z | L17 G_margin_z |
|---|---|---:|---:|---:|---:|
| 0.6B | matched | 0.00057 | 0.6420 | -0.00099 | -0.0030 |
| 0.6B | same-family | 0.00018 | 0.5085 | 0.00021 | 0.0006 |
| 0.6B | different-family | 0.00032 | 0.3135 | 0.00000 | 0.0001 |
| 1.7B | matched | 0.00002 | 1.0388 | 0.00021 | 0.0033 |
| 1.7B | same-family | 0.00004 | 0.8740 | -0.00005 | 0.0037 |
| 1.7B | different-family | -0.00001 | 0.7402 | -0.00004 | 0.0077 |

Structured context-only state then produces large downstream A_q_z in both
models. At L27, matched A_q_z/A_margin_z is `0.895/0.811` for 0.6B and
`0.826/1.111` for 1.7B. The interaction path differs by scale: 0.6B G_q_z and
G_margin_z remain near zero, while 1.7B develops positive structured downstream
interaction. At L27, G_q_z/G_margin_z is `0.0490/0.0456` for matched,
`0.0789/0.0522` for same-family, and `0.1011/0.0621` for different-family.

Thus orthogonal structured state has an immediate additive readout route and is
subsequently transformed into the decoded coordinate. In 1.7B, but not 0.6B,
q-setting also interacts with this downstream propagation/readout pathway.

## Full-discovery mechanistic verdict

- Qwen3-0.6B: **mostly additive multidimensional signal**.
- Qwen3-1.7B: **mixed additive plus structured q-by-context gating**.
- Cross-scale: both checkpoints use causal information outside the scalar probe
  coordinate; the resolved interaction is scale dependent and much smaller than
  the additive component.

The E01B-2 phrase “context gated” must therefore be narrowed. Its total context
increment is not itself a gating estimate. In 0.6B it is additive under the
factorial test; in 1.7B it combines a dominant additive signal with a smaller
structured interaction.

## Strongest supported claim

> Under the frozen synthetic-relation task, layer-17 site, and Qwen3 checkpoints,
> the probe-defined scalar semantic coordinate is causal but incomplete.
> Opposite-label orthogonal state independently carries substantial causal
> information in both models. In Qwen3-1.7B, but not Qwen3-0.6B, structured
> orthogonal state also measurably changes the causal efficacy of the identical
> scalar setpoint.

This is full discovery evidence, not confirmation. It does not establish that
the probe axis is the endogenous natural code, generalize across tasks/model
families/sites, or authorize any application experiment.

## Next boundary

The E01B-1/E01B-2/E01B-3 discovery mechanism is now decomposed. Freeze the exact
claim, models, site, target, context plans, estimands, contrasts, and decision
rule before separately deciding whether to authorize the untouched confirmation
split. Confirmation, E02, E13, E14, E15, and E16 remain unauthorized.
