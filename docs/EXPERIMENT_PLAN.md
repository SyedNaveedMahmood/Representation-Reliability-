# Staged Experiment Plan

## E00 — Representation Cartography
Find D across layers/sites/tokens.

Start Qwen3-0.6B, then Qwen3-1.7B.
Output D profiles + random-label/text controls.

## E01 — Decode -> Causal Gap
At identical sites compare D to matched replacement-patch C.
First every 4th layer, then refine.

## E02 — Decode -> Steer Gap
Diff-mean + probe-normal steering.
Alpha:
`[-4,-2,-1,-0.5,0,0.5,1,2,4]`.
Include norm-matched random directions.

## E03 — Causal -> Steer Gap
Select high-C/weak-S sites.
Test replacement vs additive steering.
Only later: rotation/ReFT.

## E04 — Internal Alarm Paradox
Compare corruption-state monitor to final-output-error monitor on identical states.

## E05 — Temporal Commitment
Qwen3-0.6B first.
Compute trajectory geometry, rank change points, patch top/bottom/random steps.

## E06 — Observer vs Causal Components
Compare probe attribution with isolated PGB-CT output.

## E07 — Causal Robustness
Paraphrase, reframe, benign distractor, clause reorder.
Measure activation similarity + D/C/S stability.

## E08 — SAE Causal Consistency
Only after E07 infrastructure.

## E09 — KV Cache Scars
Correction/revocation setup; K-only, V-only, replace/zero/interpolate.

## E10 — Monitor Spoofing
Move monitor score while constraining output KL/accuracy.

## E11 — Behavioral Rank
Rank 1/2/4/8/16 control.

## E12 — Latent Transfer
Speculative cross-model communication.

## First end-to-end target
For one controlled relational task:
```text
D_l
C_l
S_l
K_l
```
Freeze candidate on discovery split, then test 3 seeds + paraphrase + confirmation split.
