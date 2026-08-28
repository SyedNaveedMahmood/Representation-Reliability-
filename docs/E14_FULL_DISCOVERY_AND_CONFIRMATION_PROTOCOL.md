# E14 Full Discovery and Confirmation Protocol

Status: **full discovery authorized; one E14-specific confirmation campaign
preregistered and locked**.

This protocol was frozen before E14 full discovery and before materializing the
E14 confirmation namespace. The consumed E01 confirmation split is prohibited.

## Frozen identity

```text
model: Qwen/Qwen3-1.7B
model revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
tokenizer revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
backend: optimum-quanto 0.2.7
ladder: BF16, qint8 weight-only, qint4 weight-only
compute/runtime activation dtype: bfloat16
activation quantization/calibration: none
site: resid_post, layer 17, last_prompt
candidate texts: " Yes", " No"
candidate token IDs: 7414, 2308
trace layers: 17,20,23,27
```

Quanto quantizes all 197 supported linear modules, including `lm_head`. INT8 is
per-output-channel symmetric absmax. INT4 is groupwise affine with floating
shift and group size 128 for every module in this checkpoint. Embeddings and
RMSNorms remain BF16. No alternate backend, INT3, or INT2 is permitted.

## Full-discovery sample and estimands

Use all 150 matched discovery pairs / 300 directed examples from the established
synthetic relation split. Probe training is train only, C selection validation
only, and evaluation discovery-test only. Save a precision-native probe for each
precision. Apply the exact BF16 scaler/classifier unchanged for frozen-axis D.

The BF16 validation-only class medians, BF16 unit probe directions, and target
orientation define the causal intervention in every precision:

```text
Q = Y10 - Y00
A_c = Y01,c - Y00
G_c = (Y11,c - Y10) - (Y01,c - Y00)
```

Primary A/G are matched structured context minus the mean of ten deterministic
random orthogonal contexts. Random seeds are `1729..1738`. Context norms are
matched per example to that precision's matched-orthogonal norm; the deterministic
validation median is the degenerate fallback. Lambda 1 is primary and lambda 0.5
is secondary. Bootstrap the `pair_id` cluster with 2,000 draws. Preserve raw and
precision-validation-standardized values.

Primary discovery quantities are D-native AUROC/AUPRC, frozen-BF16-axis
AUROC/AUPRC, relation-margin B AUROC, Q, matched/random/contrast A and G,
WikiText NLL/PPL, HellaSwag accuracy and mean selected normalized log-likelihood,
and Q/A/G traces at L17/L20/L23/L27.

## Frozen general-quality controls

### WikiText-2 raw test

```text
dataset: Salesforce/wikitext
configuration: wikitext-2-raw-v1
split: test
revision: b08601e04326c79dfdd32d625aee71d232d685c3
```

Concatenate nonempty test text in dataset order with newline separators, tokenize
once with the frozen Qwen tokenizer, and score exactly the first 10,000 tokens.
Use disjoint blocks of at most 512 tokens; each token after the first token of a
block contributes one next-token loss. Report token-weighted NLL and perplexity.

### HellaSwag validation

```text
dataset: Rowan/hellaswag
configuration: default
split: validation
revision: 218ec52e09a7e7462a5400043bb9a69a41d06b76
subset size: 500
subset seed: 20261402
```

Order rows by SHA-256 of `20261402|ind`; take the first 500. Prompt is the
dataset's `ctx` field; score all four `endings` as continuations with the local
adapter's whitespace convention. Primary choice score is mean continuation
log-likelihood; choose argmax with lowest-index tie breaking. Persist selected
`ind` values/digest, accuracy, and mean gold normalized log-likelihood.

Catastrophic generic damage is flagged if, relative to BF16, WikiText PPL rises
more than 25% or HellaSwag accuracy falls more than 0.10 absolute. This changes
claim scope; it does not erase measured D/Q/A/G.

## Full-discovery gate

Proceed to the locked holdout only if:

```text
INT4 D_native >= 0.99 AUROC
paired (INT4 - BF16) G matched-random CI is wholly below zero
all integrity gates pass
```

No alternative precision/backend can rescue a failed gate.

## E14-specific untouched confirmation namespace

Only the generation specification is frozen here; rows must not be materialized
before the discovery gate passes.

```text
namespace: e14_confirmation_v1
generator: generate_synthetic_relations
generator seed: 20261401
n_samples: 200 (100 matched pairs)
n_entities: 42
families: north_south,east_west,above_below,before_after,larger_smaller
sample ID prefix: e14-confirmation-v1-
pair ID prefix: e14-confirmation-v1-
surface generation: generator-native deterministic variants
access campaigns authorized: 1
```

Specification digest is SHA-256 over the canonical JSON above. This holdout may
not fit probes, select C, construct targets, choose quantization, debug execution,
select metrics, or alter thresholds. BF16 full-discovery artifacts supply probe
directions/targets; each precision's full-discovery artifact supplies its frozen
precision-native D probe. All precision configurations are fixed before the
first holdout result is viewed.

Confirmation uses all 200 directed rows, lambda 1 primary, lambda 0.5 secondary,
ten random seeds, traces L17/L20/L23/L27, 10,000 pair-cluster bootstrap draws,
and 100,000 pair-cluster sign-flip draws. WikiText/HellaSwag are rerun unchanged
as general controls but are not confirmation labels.

## Primary H14 family

Family-wise alpha is 0.05 with Holm correction across exactly H14.1-H14.3.

### H14.1 — precision-native representation survival

`D_native_INT4 >= 0.99 AUROC`. Report clustered AUROC CI. The directional
bootstrap p-value is the plus-one-corrected fraction of pair-bootstrap AUROCs
below 0.99. PASS requires point AUROC at least 0.99 and Holm p below 0.05.

### H14.2 — interaction actionability degradation

For paired holdout rows, `G_INT4(matched-random) < G_BF16(matched-random)`.
Use a one-sided pair-cluster sign-flip p-value and paired cluster-bootstrap CI.

### H14.3 — additive actionability degradation

For paired holdout rows, `A_INT4(matched-random) < A_BF16(matched-random)` with
the same directional paired procedure.

Q is a secondary diagnostic: report paired INT4-minus-BF16 estimate, CI, sign,
and relative change. No noninferiority margin is defined. Secondary outcomes are
INT8, lambda 0.5, frozen-axis D, B, relation families, and all traces.

Strong confirmation requires H14.1-H14.3 PASS. Partial confirmation requires
H14.1/H14.2 PASS with H14.3 inconclusive. D collapse or failure to reproduce G
degradation is failure. Claim scope is additionally classified as selective,
mixed semantic plus general degradation, or catastrophic generic degradation
using the frozen general-control flags.

## Claim boundary

The intended claim, only if confirmed, is that under this checkpoint, task,
site, and Quanto scheme, precision-native semantic availability survives INT4
more strongly than higher-order actionability. Frozen-axis loss must be reported.
No conclusion generalizes to other quantizers, tasks, models, or bits.
