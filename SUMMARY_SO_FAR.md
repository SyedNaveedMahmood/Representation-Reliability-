# Summary So Far - Representation Reliability Harness

Status date: 2026-08-27. Phase 0A.2 and E01A full discovery are complete.
Confirmation remains locked.

## Mission and claim boundary

The repository is a modular harness for measuring gaps among decodability (D),
causal use (C), steerability (S), monitoring (M), robustness (R), and collateral
safety (K). Current results establish held-out linear decodability,
readout/behavior associations, and causal sensitivity to a frozen decoded
coordinate under the E01A intervention. They do not establish that the
unperturbed model endogenously uses exactly that one-dimensional coordinate.

## What is implemented

Phase 0A now includes strict layered configuration, full run manifests, a
stable Hugging Face adapter, explicit canonical activation sites and token
selectors, sharded atomic safetensors/parquet caching, group-isolated synthetic
data splits, linear probes, mandatory baselines, behavior evaluation, and CLI
runners.

Phase 0A.2 adds:

- exact-architecture deterministic random-initialization controls;
- leave-one-relation-family-out probing;
- validation-only threshold and one-dimensional calibration;
- paired raw-completion and Qwen chat-nonthinking interfaces;
- token-embedding and fixed native-readout rungs;
- raw probe/native-readout geometry;
- frozen-probe analysis on native behavior errors;
- paired bootstrap comparisons, five required figures, and per-example raw
  evidence for every arm.

E01A adds a discovery-only truth-coordinate intervention at a frozen canonical
site, magnitude-matched random and orthogonal controls, same-label and shuffled
source controls, full-residual upper bounds, exact-batch BF16 fidelity checks,
downstream tracing, pair-cluster inference, and shard-safe resume.

The full suite passes: **116 tests**, including GPU checks that the final fixed
readout matches native logits and that extraction/cache/intervention contracts
remain exact.

## Integrity history

The original Phase 0A runs `E00_bf9efb94222b`, `E00_18a016f37eb9`, and
`E00_50519d0b487d` reported D near 0.53. Those probe results are invalidated by
a multi-shard activation/metadata row-ordering bug and must not be cited.
`INTEGRITY_AUDIT_PHASE_0A1.md` records the failure and repair. Corrected
schema-v2 runs `E00_b137cfabe7d3` and `E00_5162f18f1901` show a stable
near-saturated mid/late D band under chance random-label and TF-IDF controls.

During the full E00-C run, a PC shutdown killed the first process at the
random-init stage. Shard-safe resume produced the completed 0.6B run
`E00C_5754f1ceaaba-r2`; the stale predecessor is not treated as complete. A
paired-bootstrap indexing error affected only initially derived origin CIs,
not point estimates or raw rows. It was repaired, regression-tested, and the
CIs were deterministically rebuilt from aligned per-example evidence.

## Current evidence

Primary condition: Qwen3-0.6B, n=2,000 matched counterfactual-twin semantic
examples, discovery-test n=300.

| Diagnostic | Result |
|---|---:|
| peak held-out D | 1.0000 AUROC |
| mid/late D, layers 14-27 | 0.9988 |
| mid/late random-init D, seeds 0/1/2 | 0.4846 / 0.4897 / 0.5053 |
| mid/late learned origin gain | 0.5056 |
| mid/late LOFO D | 0.9680 |
| mid/late random-init LOFO D | 0.4995 |
| raw behavior accuracy / margin AUROC | 0.600 / 0.7632 |
| calibrated behavior accuracy | 0.730 |
| chat-nonthinking behavior accuracy | 0.500 |
| mid/late fixed-readout L | 0.7166 |
| mid/late G_DL | 0.2822 |
| mid/late absolute probe/native cosine | 0.0380 |
| D on raw native-error subset, layer 17 | 1.0000 AUROC (n=120) |

All five LOFO families exceed 0.928 mean mid/late AUROC and beat matched
random initialization by at least 0.417. Validation-only calibration improves
0.6B accuracy by 0.130 but closes only 32.5% of the reference D-minus-B gap;
chat makes accuracy worse. All predeclared 0.6B gates pass.

The supported 0.6B diagnosis is: **learned abstract readout bottleneck
supported**. "Abstract" here means cross-family linear generalization, not a
causally established feature.

## Gated 1.7B scale comparison

Branch C justified an exact semantic-dataset replication on Qwen3-1.7B:
`E00C_f7fa7e06c7f2`.

| Model | mid/late D | LOFO D | fixed L | raw B | calibrated B | chat B |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.9988 | 0.9680 | 0.7166 | 0.600 | 0.730 | 0.500 |
| Qwen3-1.7B | 0.9994 | 0.9917 | 0.8889 | 0.533 | 0.890 | 0.927 |

Scaling barely changes the already saturated D, but it narrows G_DL from
0.2822 to 0.1105. The 1.7B raw margin has AUROC 0.9509 despite threshold-zero
accuracy 0.533; validation-only calibration reaches 0.890 and matched chat
reaches 0.927. The readout bottleneck is therefore not scale/interface
invariant. Scale changes native-readout alignment and behavioral expression
far more than representation strength in these conditions.

## E01A full discovery

The completed full runs are `E01A_c6cd215d7bf8` (Qwen3-0.6B) and
`E01A_821138e998c7` (Qwen3-1.7B), with 300 directed discovery examples / 150
matched pairs per model. Both used layer 17 `resid_post`, `last_prompt`, the
full eight-alpha profile, ten random directions, and 2,000 pair-cluster
bootstrap draws. Confirmation was not accessed.

At alpha 1:

| Model | truth effect | random | orthogonal | same-label | shuffled opposite | full patch | kappa |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.0283 | 0.0031 | 0.0051 | 0.0058 | 0.0229 | 1.1504 | 0.0174 |
| Qwen3-1.7B | 0.6621 | 0.0063 | 0.0109 | -0.0198 | 0.6617 | 4.2627 | 0.0380 |

The truth treatment beats random, orthogonal, and same-label controls in both
models with pair-cluster intervals excluding zero. It does not beat the
shuffled opposite-label coordinate control at any non-zero alpha. Thus the
decoded coordinate is causally actionable under intervention, but matched-twin
source specificity is not supported.

The 1.7B raw effect is 23.37 times larger and exploratory conversion efficiency
is 2.18 times larger. Both models have the correct negative-alpha reversal and
positive dose response. The 0.6B effect remains small/sparse (alpha-1 median
zero), while the 1.7B alpha-1 median is 0.625. See
`E01A_FULL_DISCOVERY_SUMMARY.md` for exact confidence intervals, integrity
checks, dose response, and claim boundaries.

## What is and is not resolved

Supported for Qwen3-0.6B:

- D is learned rather than explained by random-network separability.
- D generalizes across held-out relation families.
- simple calibration and the official chat interface do not explain most of
  the 0.6B gap.
- the frozen truth decoder remains accurate on native errors.
- the fixed native readout is weaker and geometrically poorly aligned with the
  external decoder.

Not established:

- that the unperturbed model endogenously uses exactly the intervened axis;
- that nuisance-matched source identity matters beyond the requested
  opposite-label coordinate target;
- that this is universal across model families, tasks, or scales;
- that visible output behavior faithfully reports internal reasoning;
- that one selected layer would confirm without untouched confirmation data.

## Exact next question

E01B should test why matched and shuffled opposite-label coordinate sources
are equivalent across the full dose response: is causal conversion determined
only by the target coordinate, or do source identity, relation-family matching,
and nuisance attributes matter under a more discriminating predeclared design?
This is a registered proposal, not authorization to run E01B or touch
confirmation.

See `DIAGNOSIS_PHASE_0A2.md` for full measured results and
`E01A_FULL_DISCOVERY_SUMMARY.md` for the causal-discovery results.
