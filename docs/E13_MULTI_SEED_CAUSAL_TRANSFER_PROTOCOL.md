# E13 Multi-Seed Causal-Organization Transfer Protocol

Status: **full discovery authorized; frozen before new student optimization on
2026-08-28**. E13 confirmation is locked and must remain unmaterialized.
Conversion-response distillation is conditional on the gate in this document.

## Question and claim boundary

When teacher and student have similar semantic decodability and behavior, do
they implement the same causal organization, and can intervention-response
matching transfer that organization more directly than output or hidden-state
matching? This is discovery in one Qwen model family, relation task, layer, and
token site. Probe results establish decodability, not endogenous causal use.

The frozen profile is

```text
C = (D, Q, A, G, B)
D = semantic probe AUROC
B = native target-label Yes-minus-No margin AUROC
Q = Y10 - Y00
A = Y01,matched - Y00
G = (Y11 - Y10) - (Y01 - Y00), matched context
```

These quantities are a profile, not a strict hierarchy. Raw matched-minus-mean-
random A/G contrasts remain required intervention controls and retain continuity
with the bounded pilot. No confirmation result is authorized here.

## Locked holdout and frozen mechanism

The namespace `e13_confirmation_v1`, seed `20261304`, 200 directed examples / 100
pairs is forbidden during this campaign. Code must not generate, materialize,
load, inspect, count, debug, evaluate, or fit on it. Every artifact records
`confirmation_accessed: false`.

The E01 mechanism is unchanged: Qwen3 relation prompts from all five frozen
relation families, layer 17 (zero-indexed), canonical `resid_post`, native module
name persisted, `last_prompt`, validation-defined source-free semantic targets,
matched orthogonal context, semantic q direction, and four-arm Y00/Y10/Y01/Y11
algebra. Random contexts use seeds 2130/2131/2132 and are norm matched per row.

## Models, corpus, seeds, and training

Teacher is frozen `Qwen/Qwen3-1.7B`; student is `Qwen/Qwen3-0.6B`. Resolved model
and tokenizer revisions are manifest fields. The open corpus is fixed across all
runs: train 4,000 directed rows (seed 20261301), validation 500 (20261302), and
discovery 300 (20261303), preserving counterfactual pairs and global prompt
deduplication. Corpus and input-ID digests must match across regimes and seeds.

Student seeds are exactly `20261305`, `20261315`, and `20261325`. R0 is evaluated
once. R1/R2/R3 use 100 optimizer updates, checkpoints 0/10/25/50/100, microbatch
2, accumulation 4, effective batch 8, AdamW betas (0.9,0.95), epsilon 1e-8,
weight decay 0.01, gradient clipping 1.0, peak LR 2e-5, ten-step linear warmup,
then cosine decay. Any OOM correction may only lower microbatch while increasing
accumulation to preserve effective batch and must be recorded.

```text
R1: full-vocabulary hard-label cross entropy
R2: 0.5 CE + 0.5 T^2 KL(teacher_T || student_T), T=2
R3: R2 + L_H, lambda_H=1
```

For R3, at layer 17 `resid_post` / `last_prompt`, independently normalize each
hidden vector as `h / sqrt(mean(h^2)+1e-8)`. A jointly trained linear projector
maps student to teacher width. `L_H` is mean squared projected-to-stop-gradient
teacher error divided by teacher width (equivalently elementwise mean MSE).
Projector and optimizer state are saved portably with every checkpoint. Student
updates, examples, ordering rule, and base LR are identical to R2.

Teacher logits and hidden states may be cached. Cache identity includes resolved
model/tokenizer revisions, ordered sample IDs, input-ID digest, logits digest,
layer-17 activation digest, dtype, site, selector, and prompt/corpus digests. A
fixed subset must agree with direct inference within dtype tolerance before use.

## Validation-only quantities and checkpoint selection

For every model/checkpoint, calculate from clean **validation** margins only:

```text
sigma_margin_validation = population SD of native Yes-minus-No margins
validation CE, binary conditional entropy, mean absolute margin, margin SD
B_validation = target-label margin AUROC
```

The scale must be finite and greater than 1e-8 and is persisted. It never uses
discovery rows. Teacher uses its own scale; every student checkpoint uses its
own scale. The frozen initial-student axis is retained for `D_frozen`; native D
continues to use train fitting and validation-only C selection.

For each trained seed/regime, select

```text
argmin_step abs(B_validation_student(step) - B_validation_teacher)
```

over 0/10/25/50/100, with earliest-step tie breaking. Discovery quantities are
not accepted by the selector. The selected step is the `B-matched checkpoint`.

## Causal effect views

All arms persist oriented Yes/No logits and target-oriented margins. Raw effects
are the frozen Q/A/G algebra. Cross-regime standardized effects are
`Q_z=Q_raw/sigma`, `A_z=A_raw/sigma`, and `G_z=G_raw/sigma` using that
checkpoint's validation scale.

For each arm, compute conditional binary `p_yes=exp(l_yes)/(exp(l_yes)+exp(l_no))`
stably as `sigmoid(l_yes-l_no)`, orient it to the frozen counterfactual target,
and apply the identical factorial algebra to obtain Q/A/G probability effects.

A strict target flip for an arm is `Y00 <= 0 and Yarm > 0` on the target-oriented
margin. Report q-only (Y10), context-only (Y01), and joint (Y11) flip rates for
matched context and each random seed, plus matched-minus-mean-random contrasts.
Also retain target-prediction rates so strict flips are auditable.

## Causal Organization Distance

Rows align exactly by discovery sample ID and pair ID; duplicates, omissions,
target mismatches, or teacher/student order mismatches are stop conditions. For
each row, the primary profile uses the matched-context effects:

```text
c_i = [Q_z, A_z, G_z]
COD = mean_i ||c_i_student - c_i_teacher||_2
```

Report raw-profile distance descriptively; mean absolute Q_z/A_z/G_z gaps;
Pearson and Spearman correlation over the flattened three-component profiles;
and per-component sign agreement. Undefined correlations from degenerate
variance are `NA` with a reason. Matched-minus-random profiles and distances are
secondary controlled views, not replacements for primary COD.

## Representation and quality diagnostics

R3 additionally reports linear CKA, mean cosine after projection, and projected
hidden MSE on validation and discovery. These diagnose similarity and are not
causal-equivalence evidence.

At step 0, B-matched, and final, use the exact frozen E14 data revisions/subsets:
WikiText-2 token-weighted NLL/perplexity over at least 10,000 deterministic
scored tokens, and 500 fixed HellaSwag validation examples scored by length-
normalized conditional log likelihood. Save HellaSwag per-example rows. Also
report validation CE, clean binary output entropy, mean absolute margin, and SD.

## Statistical unit and reporting

All per-example evidence precedes aggregates. Within-run uncertainty uses pair-
cluster bootstrap. Report every seed, seed mean, sample SD, and min/max; n=3 is
not precise population variance. A hierarchical bootstrap may resample seeds
then pairs only if validated against deterministic synthetic cases. Discovery
layer/site scans are prohibited.

## Frozen method gate

After all R0/R1/R2/R3 runs finish, conversion-response work is authorized only
if all gates pass:

- A: R0 native D >=0.98 and frozen-axis D is also high (>=0.98).
- B: R2 or R3 has absolute validation B gap <=0.03 at its B-matched checkpoint.
- C: primary COD is above numerical zero (>1e-8) and at least one absolute mean
  component gap (Q_z, A_z, G_z) is >=0.20 there.
- D: B and C jointly hold in at least two of three seeds for one regime.
- E: quality evidence is scientifically interpretable: finite metrics, WikiText
  PPL less than 10x R0, and HellaSwag accuracy no more than 0.20 below R0.

The numeric quality bounds are catastrophic-damage stop flags, not equivalence
claims. If any gate fails, stop after baseline discovery. If all pass, record
`CONVERSION-RESPONSE METHOD AUTHORIZED`, create and remotely preregister
`docs/E13_CONVERSION_RESPONSE_FULL_PROTOCOL.md`, then and only then run the
already specified R4/R5/R6/R2-C three-seed method campaign. No coefficient may
be chosen from discovery Q/A/G outcomes.

## Integrity, resume, and execution

Each regime/seed has immutable identity including scientific config, corpus,
seed, model revision, cache/probe/target digests, and checkpoint schedule.
Checkpoint writes are atomic and include model, projector if any, optimizer,
scheduler position, RNG state, data-order/cursor state, and a completion marker.
Resume accepts only an exact identity and resumes from the latest complete
checkpoint; partial files are never marked complete. Independent processes may
not share a run directory.

Before full training: all unit tests, Ruff on modified Python, CPU/tiny contracts,
and <=50-example GPU smoke must pass. Stop an affected regime on nonfinite loss
or gradient, corrupt checkpoint, no-op failure, hook leakage, nonfinite factorial
arms, row mismatch, digest mismatch, confirmation access, factorial identity
failure, or irrecoverable OOM. A bug requires a reproducer, impact statement,
regression test, minimal fix, and rerun of all affected results.

The scheduler uses detected GPU indices, at most one training process per GPU,
captures commands/logs/PIDs/timestamps/exit codes/peak VRAM, retries only proven
transient or OOM failures without changing scientific identity, and never
silently skips an incomplete job.
