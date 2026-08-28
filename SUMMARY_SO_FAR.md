# Summary So Far - Representation Reliability Harness

Status date: 2026-08-28. Phase 0A.2, E01A/E01B discovery, the exploratory
E01A trace-mechanism analysis, and the single preregistered E01 confirmation
are complete. All four primary confirmation hypotheses passed Holm correction;
the core E01 mechanism is frozen. The separately preregistered E14 bounded
BF16/INT8/INT4 pilot on Qwen3-1.7B is also complete.

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

E01B-1 adds validation-defined source-free class medians and continuous
quantile setpoints, per-example norm-matched random/orthogonal controls,
validation-only standardization, finite flat-grid handling, dtype-aware
setpoint gates, intervened-forward tracing, and treatment-shard resume. No
donor hidden state is used.

E01B-2 adds fixed-setpoint orthogonal-context decomposition, deterministic
matched/same-family/different-family/same-label source plans, validation-only
fallback norms, per-example norm standardization, ten-seed random orthogonal
controls, exact coordinate-only reproduction gates, context-increment
contrasts, relation-family summaries, downstream tracing, and shard-safe
resume.

The CPU-only E01A trace analysis adds clean-baseline deduplication,
expected-label-oriented layer trajectories, discovery-standardized propagation
and native-readout metrics, pair-cluster conversion regressions, cross-scale
uncertainty, source-equivalence regressions, relation-family/error strata, and
six exploratory figures. It reads completed evidence only and performs no new
model forward.

The full suite passes: **173 tests**, including GPU checks that the final fixed
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

## E01A trace-mechanism result

The alpha-1 standardized injected coordinate is comparable across scales at
L17 (0.6B `1.826`, 1.7B `1.867`) and L20 (`1.152`, `1.132`). At L23/L27,
1.7B retains more signal (`0.950`/`0.679` versus `0.776`/`0.490`). The larger
difference is conversion into native readout: standardized L27 margin change
is `0.212` versus `0.020`, and the no-intercept standardized conversion slope
is `0.308` versus `0.039`. The discovery mechanism classification is
**mixed bottleneck, dominated by readout conversion**.

Matched and shuffled alpha-1 coordinate targets are strongly correlated, and
their non-zero-alpha regression has no residual matched-source coefficient
after actual coordinate displacement is included: `0.00054` (95% CI
`[-0.00289, 0.00401]`) for 0.6B and `0.00048` (`[-0.00591, 0.00666]`) for
1.7B. This supports the bounded explanation that E01A's coordinate-only
conditions carry only a scalar target; it does not prove source context would
be irrelevant if orthogonal information could enter. See
`E01A_TRACE_MECHANISM_ANALYSIS.md`.

## E01B-1 source-free full-discovery result

The completed runs `E01B1_e1169f3ffe11` (Qwen3-0.6B) and
`E01B1_5b9d70c8cffe` (Qwen3-1.7B) used all 150 discovery pairs, ten random and
ten orthogonal-random directions, and validation-only targets/scales.
Confirmation was not accessed.

Opposite-class source-free median effects are `0.026250` (95% CI `[0.016667,
0.035833]`) for 0.6B and `0.666042` (`[0.641042, 0.692302]`) for 1.7B. Both
beat norm-matched random and orthogonal controls. These closely reproduce E01A
alpha-1 donor-derived effects (`0.028333` and `0.662083`), supporting the
bounded conclusion that donor hidden states are unnecessary for the measured
coordinate-only effect.

The validation-grid population slope is positive in both models: `0.014022`
(`[0.010617, 0.017646]`) for 0.6B and `0.038412` (`[0.036730, 0.040176]`) for
1.7B. Per-example monotonicity is nevertheless weak in 0.6B (median Spearman
`0.224`; exact monotonic fraction `0.420`) and strong in 1.7B (`1.000` and
`0.937`). Final-layer standardized native-margin changes are `0.018984` and
`0.213666`, while standardized coordinate retention is `0.497` and `0.688`.
This reproduces a mixed bottleneck dominated by readout conversion. See
`E01B1_FULL_DISCOVERY_SUMMARY.md` for complete targets, gates, controls, traces,
and claim boundaries.

## E01B-2 orthogonal-context full-discovery result

The completed runs `E01B2_f2d75dab1eba` (Qwen3-0.6B) and
`E01B2_e2b4b02cb3a4` (Qwen3-1.7B) used all 150 discovery pairs, the frozen
source-free opposite-class setpoint, context strengths 0.5/1.0, ten random
orthogonal directions, and 2,000 pair-cluster bootstrap draws. Confirmation
was not accessed.

At lambda 1, matched/same-family/different-family context increments over the
paired coordinate-only effect are `1.127`/`0.864`/`0.620` for 0.6B and
`3.613`/`3.080`/`2.483` for 1.7B. Every opposite-label
structured-minus-random CI excludes zero. Random increments are small but
nonzero (`-0.020` and `0.064`), and the
0.6B same-label increment is also small but positive (`0.105`); neither is
comparable to the opposite-label structured effects.

At L17, structured context changes validation-standardized native readout by
`0.314` to `0.639` in 0.6B and `0.748` to `1.042` in 1.7B while changing the
decoded q displacement by less than `0.001`. Structured contexts subsequently
alter downstream q propagation. The discovery classification is **structured
context gated, with aggregate relation-family gating and family-level
heterogeneity**. This operational result does not distinguish a multiplicative
q-by-context interaction from an additive causal signal in the orthogonal
structured component. See `E01B2_FULL_DISCOVERY_SUMMARY.md`.

## E01B-3 additive-vs-gating full-discovery result

The completed runs `E01B3_84ee4bae8564` (Qwen3-0.6B) and
`E01B3_e7462847a558` (Qwen3-1.7B) used all 150 discovery pairs, the frozen
E01B-2 context plans, lambdas 0.5/1.0, ten random contexts, and 2,000
pair-cluster bootstrap draws. Y00/Y10/Y11 were identity-audited reuse from the
matching E01B-2 runs; only the Y01 context-only arm required new full-set
forwards. All compatibility and numerical gates passed, and confirmation was
not accessed.

At lambda 1, the additive context-only effects A for matched/same-family/
different-family contexts are `1.121`/`0.857`/`0.618` in 0.6B and
`3.467`/`2.918`/`2.287` in 1.7B. The corresponding factorial interactions G
are `0.006`/`0.007`/`0.002` in 0.6B, with all structured-minus-random G
intervals including zero. In 1.7B, G is `0.146`/`0.163`/`0.196`, and all three
structured-minus-random intervals exclude zero.

The corrected full-discovery classification is therefore **mostly additive
multidimensional signal** in 0.6B and **mixed additive plus structured gating**
in 1.7B. The additive component dominates the E01B-2 context increment in both
models. At L17, q remains fixed while additive context signal reaches native
readout immediately; downstream q changes emerge in both models, but resolved
downstream interaction emerges only in 1.7B.

Matched and same-family contexts have larger additive effects than different-
family context in both models. The interaction does not follow that hierarchy:
in 1.7B, different-family G is larger than matched and same-family G. Relation
compatibility therefore structures the additive signal, not a monotonic positive
gating hierarchy. See `E01B3_FULL_DISCOVERY_SUMMARY.md`.

## E01 strong confirmation and final mechanism freeze

The single joint campaign `CONFIRMATION_46312baf5992` evaluated 200 directed
examples / 100 untouched pairs per checkpoint under the remotely pushed
protocol `e0ddfae54b350c0545c71a8237645375bdf84929`. The split was first
accessed at `2026-08-28T13:27:38.622637+00:00`; the campaign/access count is
one.

All four primary hypotheses passed Holm correction: H1 scalar actionability in
both checkpoints (`0.014375`, Holm `0.026080`); H2 matched additive signal over
random in both checkpoints (`1.273375`, Holm `0.000040`); H3 structured
interaction over random in 1.7B (`0.132844`, Holm `0.000040`); and H4 the
1.7B-over-0.6B structured-interaction difference (`0.137906`, Holm `0.000040`).
The classification is **strong confirmation**.

The final frozen result is distributed semantic actionability: scalar q is
causally effective, structured orthogonal state carries substantial independent
causal information in both checkpoints, and 1.7B also has a reproducible
structured q-by-context interaction. No structured interaction was detected in
0.6B; equivalence to zero was not tested. See `E01_CONFIRMATION_SUMMARY.md`.

## What is and is not resolved

Supported for Qwen3-0.6B:

- D is learned rather than explained by random-network separability.
- D generalizes across held-out relation families.
- simple calibration and the official chat interface do not explain most of
  the 0.6B gap.
- the frozen truth decoder remains accurate on native errors.
- the fixed native readout is weaker and geometrically poorly aligned with the
  external decoder.
- the same frozen scalar setpoint has strongly context-dependent causal effects
  under structured orthogonal state edits;
- structured context sensitivity and its downstream readout/propagation effects
  are substantially larger in 1.7B;
- structured orthogonal state carries independent additive causal signal in both
  checkpoints;
- a smaller structured q-by-context interaction is resolved in 1.7B but not
  0.6B under the frozen factorial design.

Not established:

- that the unperturbed model endogenously uses exactly the intervened axis;
- that nuisance-matched source identity matters beyond the requested
  opposite-label coordinate target;
- that this is universal across model families, tasks, or scales;
- that visible output behavior faithfully reports internal reasoning;
- that the confirmed selected-layer mechanism generalizes to another site.
- whether the E01B-3 decomposition generalizes beyond this task, site, model
  family, and pair construction;
- whether the probe-defined axis and orthogonal components correspond to the
  model's endogenous natural coding scheme.

## Exact next boundary

E01 discovery and its single untouched confirmation are complete; the core
mechanism is frozen against further tuning. E14 Stage 0/1 is complete on 150
directed discovery examples. INT8 preserved D/Q/A/G. INT4 preserved
precision-native D but reduced frozen-axis D and especially the structured
interaction G; it also reduced additive A while scalar Q increased. The paired
INT4-versus-BF16 changes were `-14.9%` for A and `-49.3%` for G, with both
pair-cluster CIs excluding zero. Native task-margin discrimination declined but
prompt perplexity did not catastrophically worsen, so semantic-specific damage
is not yet isolated from all task-level degradation.

Full E14 discovery is scientifically justified but remains unauthorized. Any
full study should retain the frozen Q/A/G measurements and add a stronger,
predeclared general-quality corpus. The consumed E01 confirmation split must
not be reused as an E14 holdout.

See `DIAGNOSIS_PHASE_0A2.md` for full measured results and
`E01A_FULL_DISCOVERY_SUMMARY.md` for the causal-discovery results, and
`E01A_TRACE_MECHANISM_ANALYSIS.md` for the exploratory pathway localization.
See `E01B1_FULL_DISCOVERY_SUMMARY.md` for donor-free setpoint causality.
See `E01B2_FULL_DISCOVERY_SUMMARY.md` for fixed-setpoint orthogonal-context
causality and its claim boundaries. See `E01B3_FULL_DISCOVERY_SUMMARY.md` for
the additive-versus-gating factorial decomposition. See
`E14_BOUNDED_PILOT_SUMMARY.md` for the bounded quantization result.
