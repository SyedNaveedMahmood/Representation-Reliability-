# E14 Quantization Reliability Confirmation

Status date: 2026-08-28. The single preregistered E14-specific confirmation
campaign is complete. The holdout was accessed once and is now consumed.

Protocol commit: `3893814c11ba3f7bc4bc39ccae83191431c35443`  
Protocol SHA-256: `54e5d6865ea91f41936948271b3bfcf357240017e82ba853733bef67bd5dbbef`  
Campaign: `runs/E14_CONFIRMATION/E14_CONFIRMATION_54e5d6865ea9`  
First access: `2026-08-28T15:13:34.318147+00:00`  
Access count: `1`

## Primary confirmation

| Hypothesis | Estimate | 95% pair-cluster CI | Raw p | Holm p | Verdict |
|---|---:|---:|---:|---:|---|
| H14.1: INT4 native D − 0.99 | 0.010000 | [0.010000, 0.010000] | 0.000100 | 0.000100 | PASS |
| H14.2: BF16 − INT4 G matched-random | 0.111094 | [0.079688, 0.142220] | 0.000010 | 0.000030 | PASS |
| H14.3: BF16 − INT4 A matched-random | 0.897062 | [0.522125, 1.282543] | 0.000010 | 0.000030 | PASS |

Classification: **strong E14 confirmation**.

## Confirmed precision results

| Precision | D native | D frozen | B | Q | A matched-random | G matched-random | WikiText PPL | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 1.000000 | 1.000000 | 0.970800 | 0.712500 | 3.419656 | 0.178750 | 23.379758 | 0.580000 |
| INT8 | 1.000000 | 1.000000 | 0.977600 | 0.749687 | 3.680844 | 0.193594 | 23.188792 | 0.576000 |
| INT4 | 1.000000 | 0.958700 | 0.848950 | 0.766875 | 2.522594 | 0.067656 | 38.612419 | 0.510000 |

## Frozen claim

Under Qwen3-1.7B, the synthetic relation task, layer-17 `resid_post`
`last_prompt` site, and Optimum-Quanto 0.2.7 weight-only ladder,
precision-native semantic decodability survives INT4 while both structured
additive actionability A and structured interaction actionability G degrade.
Scalar Q does not fail first and instead increases descriptively.

This is not evidence of selective semantic damage: INT4 raises WikiText-2
perplexity by 65.2%, crossing the preregistered generic-damage boundary, while
HellaSwag accuracy falls by 0.07. The supported classification is therefore
**mixed actionability plus general degradation**. Frozen-axis D also declines,
so native-axis recoverability is not geometric identity.

No result generalizes to another backend, checkpoint, task, site, or bit width.
The E14 holdout must not be reopened.
