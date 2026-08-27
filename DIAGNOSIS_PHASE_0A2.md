# Phase 0A.2 Diagnosis: Representation Origin and Readout Bottleneck

Status: complete on 2026-08-27. Primary run:
`runs/E00C/E00C_5754f1ceaaba-r2` (Qwen3-0.6B, n=2,000). Gated scale run:
`runs/E00C/E00C_f7fa7e06c7f2` (Qwen3-1.7B, the same semantic examples).
All reported evaluation metrics use the untouched discovery-test split (n=300).
Confirmation labels remained inaccessible.

## 1. Current phenomenon

Qwen3-0.6B has a near-saturated, held-out linear truth decoder in the
predeclared mid/late band (layers 14-27), but its native Yes/No behavior is
much weaker. Mean last-prompt D is 0.9988 AUROC, whereas raw forced-choice
accuracy is 0.600 and the fixed native readout averages 0.7166 AUROC. This is
a decodability/readout association, not evidence that the decoded direction is
causally used.

Validity verdict: **learned abstract readout bottleneck supported**, for the
0.6B condition. The 1.7B replication materially limits generality: its
representation remains nearly unchanged, while calibration and the official
chat interface close most of the behavioral gap.

## 2. Literature overlap

Decodable information failing to appear in generation, confidence, or
steering is already established and is not the contribution here. The useful
increment is the combined matched-random-initialization, token-embedding,
leave-one-family-out (LOFO), fixed-readout geometry, and paired-interface audit
on one counterfactual-twin semantic task. Full positioning and claim limits are
in `docs/LITERATURE_POSITIONING_PHASE0A2.md`.

## 3. Random-init origin test

All random models use the exact resolved architecture and tokenizer but no
checkpoint weights. Seeds are deterministic and recorded per row.

| Model state | Peak D | Mid/late D | LOFO D | Notes |
|---|---:|---:|---:|---|
| 0.6B pretrained | 1.0000 | 0.9988 | 0.9680 | Post-trained checkpoint |
| 0.6B random seed 0 | 0.5080 | 0.4846 | 0.4926 | Exact architecture, no weights |
| 0.6B random seed 1 | 0.5055 | 0.4897 | 0.4959 | Exact architecture, no weights |
| 0.6B random seed 2 | 0.5190 | 0.5053 | 0.5099 | Exact architecture, no weights |
| 1.7B pretrained | 0.9998 | 0.9994 | 0.9917 | Gated scale replication |
| 1.7B random seed 0 | 0.4991 | 0.4816 | 0.4913 | Scale branch uses one random seed |

For 0.6B, mean mid/late pretrained-minus-random D is 0.5056. At layer 17
the paired delta is 0.5039 (95% bootstrap CI 0.4631-0.5464). The corrected
paired bootstrap resamples labels and all score arrays with the same semantic
indices. Across the three random-init seeds, mid/late D is 0.4932 mean with
0.0108 SD; LOFO D is 0.4995 mean with 0.0092 SD. TF-IDF AUROC is 0.4540, and
all three random-label controls remain near chance. Random features therefore
do not explain the learned-model D.

## 4. LOFO abstraction test

Each LOFO probe excludes one relation family from both training and validation
and evaluates only that family in discovery-test. Qwen3-0.6B mean mid/late
last-prompt AUROCs are:

| Held-out family | Pretrained | Random-init mean | Difference |
|---|---:|---:|---:|
| above/below | 0.9501 | 0.4796 | 0.4704 |
| before/after | 0.9284 | 0.5112 | 0.4172 |
| east/west | 0.9725 | 0.5089 | 0.4636 |
| larger/smaller | 0.9982 | 0.4982 | 0.4999 |
| north/south | 0.9909 | 0.4995 | 0.4914 |

Every family clears the predeclared D > 0.65 and learned advantage > 0.05
gate. This supports family-general linear accessibility; it does not by itself
establish an abstract causal mechanism.

## 5. Calibration test

The decision threshold was selected on validation only. For 0.6B, moving the
raw threshold from 0 to tau=1.4375 raises discovery-test balanced accuracy
from 0.600 to 0.730 (paired gain 0.130, 95% CI 0.060-0.203). A validation-fit
one-dimensional logistic calibration reaches 0.693. Threshold calibration
closes 32.5% of the D-minus-B reference gap, so it helps but does not explain
most of the 0.6B dissociation.

At 1.7B, the raw margin is strongly ranked but badly offset: margin AUROC is
0.9509 while threshold-zero accuracy is 0.533. Validation-only calibration at
tau=4.0 reaches 0.890 balanced accuracy, closing 76.5% of the reference gap.

## 6. Interface test

Semantic sample IDs are paired across raw completion and Qwen's official
non-thinking chat template. Cache identities differ, and each output row
records the interface, template flag, thinking state, verbalizers, and token
IDs.

| Condition | Accuracy | Balanced acc | Margin AUROC |
|---|---:|---:|---:|
| 0.6B raw threshold 0 | 0.600 | 0.600 | 0.7632 |
| 0.6B raw calibrated | 0.730 | 0.730 | 0.7632 |
| 0.6B Qwen chat non-thinking | 0.500 | 0.500 | 0.8142 |
| 1.7B raw threshold 0 | 0.533 | 0.533 | 0.9509 |
| 1.7B raw calibrated | 0.890 | 0.890 | 0.9509 |
| 1.7B Qwen chat non-thinking | 0.927 | 0.927 | 0.9758 |

Chat worsens 0.6B accuracy by 0.100 (paired 95% CI -0.157 to -0.043), but
improves 1.7B by 0.393. Interface choice therefore does not explain the 0.6B
gap, but it explains much of the scale model's raw-interface failure.

## 7. Behavior-error decodability

The already-fitted frozen layer-17 probe, without subset retraining, perfectly
ranks truth on the 0.6B native-error subset: n=120, AUROC/AUPRC 1.000 and
balanced accuracy 0.9947. Thus the 0.6B errors do not lack linearly accessible
truth at this measured site.

The analogous 1.7B raw-threshold error subset contains only gold-negative
examples (n=140), so its AUROC/AUPRC are correctly reported as undefined. That
single-class subset is evidence of threshold bias, not evidence for or against
error-subset decodability.

## 8. Fixed-readout analysis

The fixed readout applies the model's final normalization and LM head to each
layer. Its final-layer logits are GPU-tested against native logits.

| Measure (last prompt) | 0.6B layer 17 | 0.6B mid/late mean | 1.7B layer 17 | 1.7B mid/late mean |
|---|---:|---:|---:|---:|
| external probe D | 1.0000 | 0.9988 | 0.9998 | 0.9994 |
| fixed-readout L | 0.6890 | 0.7166 | 0.9104 | 0.8889 |
| G_DL = D-L | 0.3110 | 0.2822 | 0.0894 | 0.1105 |
| absolute probe/native cosine | 0.0381 | 0.0380 | 0.0720 | 0.0334 |
| probe score vs behavior margin | 0.6263 | 0.6285 | 0.8798 | 0.8530 |
| lens margin vs behavior margin | 0.4386 | 0.5694 | 0.8282 | 0.8261 |

The 0.6B native readout is substantially weaker than the held-out decoder and
nearly orthogonal to its raw probe direction. Scaling raises native-readout
ranking and score/margin association without materially changing D.

## 9. Scale branch

| Model | D | LOFO D | L | B |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 0.9988 | 0.9680 | 0.7166 | 0.600 raw / 0.500 chat |
| Qwen3-1.7B | 0.9994 | 0.9917 | 0.8889 | 0.533 raw / 0.927 chat |

Values D, LOFO D, and L are predeclared mid/late last-prompt means; B is
discovery-test balanced accuracy. Scaling does not strengthen an already
saturated D appreciably. It narrows G_DL from 0.2822 to 0.1105 and, under the
matched chat interface, raises behavior to 0.927. The observed bottleneck is
therefore model-scale/interface dependent rather than a universal property of
this family.

## 10. Falsified explanations

- Random-network separability does not explain pretrained D.
- Relation-family-specific memorization does not explain D under LOFO.
- Text/surface leakage does not explain D under the TF-IDF and twin controls.
- Validation-only threshold calibration does not explain most of the 0.6B gap.
- The official non-thinking chat interface does not rescue 0.6B behavior.
- Absence of linearly accessible truth on native errors does not explain 0.6B
  failures at layer 17.
- A scale-invariant readout bottleneck is falsified: 1.7B narrows the fixed
  readout gap, and calibration/chat closes most of its behavior gap.

## 11. Surviving hypothesis

Pretraining yields a family-general linear truth signal in this task, while
the mapping from that signal to the fixed Yes/No readout depends strongly on
model scale and prompt interface. In 0.6B, the fitted truth axis and native
readout are geometrically poorly aligned and native behavior remains weak; in
1.7B, D stays saturated but native-readout ranking, calibration, and chat
behavior improve. These observations motivate a causal test of whether the
0.6B decoded axis is upstream of behavior or merely correlated with another
state the readout uses.

## 12. One exact next causal question

At Qwen3-0.6B `resid_post`, last-prompt token, layer 17, does a
nuisance-matched source-to-base replacement along the discovery-selected truth
contrast change the Yes-minus-No logit margin toward the predeclared
counterfactual more than magnitude-matched random-direction and shuffled-source
controls across multiple intervention magnitudes, on untouched confirmation
data?

Do not execute this E01 question without explicit authorization.

## Reproducibility and validation

- Full test suite: 94 passed; 20 tests are in the Phase 0A.2 diagnostics file.
- GPU contracts cover final fixed-readout/native-logit equality as well as the
  existing extraction/cache checks.
- 0.6B wall time: 2,467.5 s; peak allocated/reserved VRAM: 3.16/4.53 GiB.
- 1.7B wall time: 1,466.4 s; peak allocated/reserved VRAM: 6.54/6.70 GiB.
- Per-example predictions/scores precede aggregates; caches are sharded,
  integrity-marked, and resumable. The PC shutdown left one stale `running`
  directory, while the resumed `-r2` run is explicitly `complete`.
