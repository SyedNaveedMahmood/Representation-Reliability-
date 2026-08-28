# E13 Conversion-Response Distillation Full Discovery Protocol

Status: **authorized by the frozen E13 multi-seed discovery gate; frozen before
method implementation or training on 2026-08-28**. This is open discovery only.
The E13 confirmation namespace remains locked and unmaterialized.

## Authorization and question

All Gates A-E in `docs/E13_MULTI_SEED_CAUSAL_TRANSFER_PROTOCOL.md` passed. R2
and R3 each achieved the joint behavior/mismatch/quality trigger in all three
training seeds. The method question is whether matching controlled intervention
responses reduces causal organization distance beyond logit KD, hidden-state
KD, random-response regularization, or extra response-forward computation.

No method coefficient, intervention, checkpoint, or statistic may be selected
from discovery Q/A/G outcomes. The frozen E01 layer/site/task mechanism and all
baseline evaluation definitions remain unchanged.

## Shared training identity

Teacher: frozen `Qwen/Qwen3-1.7B`. Student: `Qwen/Qwen3-0.6B`. Use the exact E13
open corpus, resolved model/tokenizer revisions, student ordering, optimizer,
100 updates, effective batch 8, checkpoint schedule 0/10/25/50/100, and seeds
`20261305`, `20261315`, `20261325` from the multi-seed protocol. R2 remains

```text
L_R2 = 0.5 CE + 0.5 T^2 KL(teacher_T || student_T), T=2
```

All methods start independently from the same base student. There is no warm
start from a baseline checkpoint and no extra student optimizer update.

## Frozen probes, scales, and training pairs

Teacher and student each use their own frozen train/validation-only layer-17
`resid_post` / `last_prompt` semantic probe, unit direction, class-median q
targets, `sigma_q_validation`, and clean native `sigma_margin_validation` from
the immutable E13 reference. These never update during method training.

Every directed training row uses its existing counterfactual twin as the
matched structured-context source. Base/source IDs, pair IDs, labels, prompts,
token sites, probe/target digests, and context-source identity are cached and
audited. Source-free semantic deltas target the opposite-class validation median
for factorial methods. Structured context is the source-minus-base residual
component orthogonal to the model's own semantic direction, standardized to its
raw orthogonal norm exactly as in E01/E13 evaluation. Teacher and student use
their own hidden spaces; teacher hidden vectors are never mapped into student
space.

Training intervention responses use target-oriented Yes-minus-No margins and
are divided by the corresponding model's frozen validation margin SD. A clean
Y00 forward from the ordinary KD computation may be reused only when prompt,
weights, dtype, token indices, and gradient semantics are identical. Added
intervention forwards use the stable local adapter API and all hooks are removed
in `finally`.

## R4: scalar conversion-response distillation

For each training row and each `delta_q_z` in `{-1,+1}`, apply

```text
delta_h = delta_q_z * sigma_q_validation * semantic_unit_direction
r(delta) = oriented_margin(h + delta_h) - oriented_margin(h)
r_z(delta) = r(delta) / sigma_margin_validation
```

Teacher responses are stop-gradient cache targets. Student responses remain
differentiable. The scalar response loss is

```text
L_CRQ = mean_delta (r_z_student - r_z_teacher)^2
L_R4 = L_R2 + 1.0 * L_CRQ
```

The mean over the two already standardized responses is the sole numerical
normalization; no outcome-dependent rescaling or lambda tuning is permitted.

## R5: factorial conversion-response distillation

For every training row compute the matched-context four arms:

```text
Y00 clean
Y10 opposite-class source-free semantic target only
Y01 matched structured orthogonal context only
Y11 semantic target plus matched context
```

Form validation-margin-standardized Q, A, and G with the frozen factorial
algebra. To keep the three squared losses numerically comparable without using
discovery outcomes, compute `s_Q`, `s_A`, and `s_G` once as the population SDs
of the teacher's corresponding responses over the 500-row **validation split**.
Each scale is floored at `1e-6`, persisted, and reused by teacher and student.

```text
Q* = Q_z / s_Q
A* = A_z / s_A
G* = G_z / s_G
L_F = (Q*_S-Q*_T)^2 + (A*_S-A*_T)^2 + (G*_S-G*_T)^2
L_R5 = L_R2 + 1.0 * mean(L_F)
```

Weights are exactly one after this validation-only component standardization.

## R6: random-response matching control

R6 uses the same three additional differentiable student intervention forwards
and the same response-loss algebra/coefficient as R5. Replace the structured
semantic/context edits for each sample and model with two deterministic random
directions generated from SHA-256 of the sample ID, arm name, model revision,
and fixed seed `20261307`:

- random-q is orthogonal to the model's semantic direction and norm matched to
  that row's semantic setpoint delta;
- random-context is orthogonal to both the semantic and random-q directions and
  norm matched to that row's matched structured context;
- Y10 uses random-q, Y01 random-context, and Y11 their sum.

Teacher random-response targets and validation-only component scales are cached
separately from R5. Degenerate orthogonalization is a STOP condition; directions
are never redrawn after outcomes are seen.

## R2-C: compute-matched logit-KD control

R2-C has exactly the ordinary R2 loss. Per microbatch it performs the same three
additional student intervention forwards as R5 using the R5 semantic/matched
interventions, with outputs detached and multiplied by literal zero. These
forwards occur before the optimizer update and do not contribute gradients.
The same examples, ordering, optimizer, updates, and checkpoints are used. This
controls additional student response-forward computation and deterministic
numerical side effects; it does not pretend to match cached teacher compute.

## Teacher response cache

Before method training, cache R4 scalar, R5 factorial, and R6 random teacher
targets for all 4,000 training rows plus the validation responses needed for
component scales. Portable metadata/tensors record ordered sample/pair/source
IDs, base margin, Q/A/G or +/- scalar responses, random targets, model/tokenizer
revisions, input-ID digest, probe and target digests, token site, dtype, and all
cache tensor digests. A deterministic 16-row subset must match live teacher
inference within BF16 tolerance (`max_abs <= 2e-2`, `mean_abs <= 2e-3`) before
any cache use. Any ID/digest/tolerance mismatch is a STOP condition.

## Checkpoint evaluation and primary method test

For every R4/R5/R6/R2-C seed, evaluate the full frozen baseline stack at
0/10/25/50/100: B, native/frozen D and controls, raw/z/probability/flip Q/A/G,
primary and controlled COD, per-example evidence, validation calibration
diagnostics, WikiText/HellaSwag at step 0/B-matched/final, and integrity gates.
Select B-matched checkpoints from validation B only with earliest tie breaking.

At B-matched checkpoints compare R2, R3, R2-C, R4, R5, and R6. The primary
descriptive method success requires R5 to retain teacher-like behavior and have
lower mean COD than every listed comparator. Report each seed and mean/sample
SD/min/max. Also report whether R4 preferentially reduces Q gap, whether R5
jointly reduces Q/A/G gaps, and whether any benefit survives all three seeds and
general-quality controls. With n=3, do not make precise population-variance or
universal superiority claims.

## Identity, resume, and stop rules

Method regime and loss/cache identity are added to immutable per-run digests.
Atomic checkpoints include model, optimizer, RNG/data cursor, and complete
markers; R4/R5/R6/R2-C never share a run directory. The one-process-per-GPU
overnight scheduler records commands, PIDs, logs, timestamps, exit status, and
peak VRAM. It may resume only an exact identity.

All prior engineering STOP rules apply, plus nonfinite response/component loss,
failed response-cache live check, hook leakage, response arm misalignment,
component-scale provenance other than validation, random-direction norm or
orthogonality mismatch, and compute-control gradient contribution. A bug needs a
reproducer, impact statement, regression test, minimal fix, and rerun of every
affected result. E13 confirmation, E15, E16, new model families, and new
quantization backends remain forbidden.
