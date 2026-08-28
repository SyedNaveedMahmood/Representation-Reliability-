# E14 Bounded Pilot Summary

Status: Stage 0 and Stage 1 complete; full discovery is not authorized.

E14 used only discovery data. The E01 confirmation split was not accessed.

## Backend

Optimum-Quanto 0.2.7 weight-only BF16/INT8/INT4 on Qwen3-1.7B; runtime activations and compute remained BF16. INT8 was per-channel symmetric; INT4 was groupwise affine. No calibration data were used.

## Bounded pilot

| Precision | D native AUROC | D frozen-BF16 AUROC | B AUROC | Prompt PPL | Q | A matched-random | G matched-random |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 0.999111 | 0.999111 | 0.953511 | 244.151 | 0.682500 | 3.284028 | 0.209167 |
| INT8 | 0.999822 | 0.999822 | 0.964978 | 248.105 | 0.730833 | 3.494306 | 0.225972 |
| INT4 | 1.000000 | 0.947378 | 0.860622 | 227.624 | 0.828333 | 2.793889 | 0.106111 |

## Paired changes from BF16

### INT8

- Q: +0.048333 (95% pair-cluster CI [+0.021250, +0.078333]); +7.1% from BF16.
- A: +0.210278 (95% pair-cluster CI [+0.137906, +0.286264]); +6.4% from BF16.
- G: +0.016806 (95% pair-cluster CI [-0.010278, +0.047226]); +8.0% from BF16.

### INT4

- Q: +0.145833 (95% pair-cluster CI [+0.056656, +0.230437]); +21.4% from BF16.
- A: -0.490139 (95% pair-cluster CI [-0.861833, -0.115219]); -14.9% from BF16.
- G: -0.103056 (95% pair-cluster CI [-0.141250, -0.064024]); -49.3% from BF16.

## Trace localization

Values are precision-validation-standardized. A/G columns are matched-structured minus seed-averaged random orthogonal context.

| Precision | Layer | Q q_z | Q margin_z | A q_z | A margin_z | G q_z | G margin_z |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 17 | 1.874590 | 0.398726 | 0.000053 | 1.077635 | -0.000666 | 0.009659 |
| BF16 | 20 | 1.125069 | 0.188355 | 0.363718 | 1.024667 | 0.075585 | 0.036706 |
| BF16 | 23 | 0.938248 | 0.306699 | 0.492362 | 1.028487 | 0.090735 | 0.068136 |
| BF16 | 27 | 0.679480 | 0.220206 | 0.725679 | 1.051739 | 0.089139 | 0.065617 |
| INT8 | 17 | 1.809421 | 0.389490 | 0.000020 | 1.104360 | 0.000600 | 0.005550 |
| INT8 | 20 | 1.101663 | 0.189009 | 0.382165 | 1.070150 | 0.078667 | 0.039311 |
| INT8 | 23 | 0.905922 | 0.303393 | 0.523108 | 1.081968 | 0.097896 | 0.069879 |
| INT8 | 27 | 0.655388 | 0.226764 | 0.762595 | 1.087363 | 0.102644 | 0.068673 |
| INT4 | 17 | 1.842411 | 0.242704 | 0.000445 | 0.932501 | -0.000411 | 0.010534 |
| INT4 | 20 | 0.963590 | 0.200940 | 0.431483 | 0.949229 | 0.068530 | 0.016110 |
| INT4 | 23 | 0.760294 | 0.309510 | 0.565731 | 0.855478 | 0.060718 | 0.059441 |
| INT4 | 27 | 0.348615 | 0.247131 | 0.413608 | 0.828061 | 0.021791 | 0.033883 |
## Interpretation

- INT8 preserved both representation views and all three actionability components; its small increases are compatible with numerical perturbation rather than damage.
- INT4 preserved precision-native decodability but reduced alignment with the frozen BF16 semantic axis and reduced structured-minus-random A and G. Q did not degrade.
- The INT4 G reduction is the clearest higher-order fragility. A also declines, while prompt perplexity does not show catastrophic generic failure. Native task-margin AUROC does decline, so the pilot does not establish purely semantic-specific damage.
- Trace decomposition localizes the A reduction immediately at L17 and downstream. The G reduction is most visible during downstream conversion, especially L20/L27, rather than as failure to apply the L17 scalar setpoint.

## Decision

A full E14 discovery study is justified scientifically, but it remains unauthorized. It should retain continuous effects and add a stronger frozen general-quality corpus before making semantic-specific compression claims.

## Integrity

- BF16: exact no-op; finite=True; max |context·u|=1.665e-15; peak VRAM=3.68 GiB; runtime=25.1s.
- INT8: exact no-op; finite=True; max |context·u|=1.388e-15; peak VRAM=4.51 GiB; runtime=27.4s.
- INT4: exact no-op; finite=True; max |context·u|=1.998e-15; peak VRAM=3.91 GiB; runtime=31.4s.
