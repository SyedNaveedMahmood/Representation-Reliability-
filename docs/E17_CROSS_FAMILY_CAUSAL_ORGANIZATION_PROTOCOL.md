# E17 Cross-Family Causal-Organization Replication Protocol

Status: **frozen before any candidate model is loaded and before any causal
quantity is computed for any non-Qwen model.** Committed and pushed before
execution. E17 is open discovery, not confirmation; no p-value here is to be read
as a preregistered confirmatory test.

Experiment code is `E17`. `E16` remains reserved for training-emergence work and
is untouched.

## 1. Question

E13's confirmed diagnostic result holds inside one Qwen teacher/student system.
E17 asks whether it generalises:

> Can a non-Qwen student already encode the semantic variable, attain
> teacher-like behavior after training, yet retain a different Q/A/G causal
> organization?

The replication target is the **broader principle** — behavioral and
representational similarity do not guarantee causal-organizational similarity —
not the specific Qwen Q/A/G pattern. Different families are permitted to differ
in which components mismatch.

## 2. Candidate pairs, frozen priority order

Attempted strictly in this order. The **first** pair satisfying the non-causal
screen is selected; later candidates are then not inspected at all.

1. **SmolLM2** — teacher `HuggingFaceTB/SmolLM2-1.7B-Instruct`, student
   `HuggingFaceTB/SmolLM2-360M-Instruct`.
2. **Llama 3.2** — teacher `meta-llama/Llama-3.2-3B-Instruct`, student
   `meta-llama/Llama-3.2-1B-Instruct`. Eligible **only** if already accessible
   without changing account permissions during the experiment.
3. **OLMo-2** — teacher `allenai/OLMo-2-1124-7B-Instruct`, student
   `allenai/OLMo-2-0425-1B-Instruct`. Checkpoint names are recorded here, before
   any causal evaluation, so no substitution can occur later.

No unrelated architecture may be silently substituted.

## 3. Screening rule — no causal peeking

A pair is eligible on **engineering, behavior and decodability only**.

**Forbidden during screening:** `Q`, `A`, `G`, `COD`, steering, factorial
interventions, or any intervened forward pass. Only `D`, `B` and engineering
checks may be computed. This is what prevents choosing a replication model
because it happened to reproduce the desired mechanism.

Engineering criteria:

* both models load;
* `resid_post` intervention site resolves;
* `last_prompt` selector supported;
* candidate Yes/No scoring supported, and teacher and student resolve the **same
  candidate token ids** (logit KD requires a shared vocabulary);
* the student trains within available GPU memory.

Quantitative criteria, on the open E17 **validation** split:

```text
teacher B  >= 0.85
teacher D  >= 0.95
student D  >= 0.95
```

Preferred but not disqualifying: behavioral headroom `B_T - B_S0 >= 0.05`. If the
first three criteria pass while student behavior is already teacher-like, the
pair may still be used and the limited headroom is documented.

Every attempted candidate's screen record is persisted, including failures.

## 4. Corpus

Fresh namespace `e17-{split}-v1`, same five relation families, fresh deterministic
seeds:

```text
train:           4000 directed / 2000 pairs   seed 20261701
validation:       500 directed /  250 pairs   seed 20261702
discovery_test:   300 directed /  150 pairs   seed 20261703
```

Counterfactual pairs are never split; prompt pairs are globally deduplicated
across E17 splits and collision counts are persisted. Sample and pair identities
are disjoint from E13 by construction.

Because E17 reuses the same generator and entity pool, a fraction of E17 prompt
*strings* also occur in E13. This is measured and reported, and is harmless: E17
evaluates entirely different models, which have never been trained or selected on
E13 rows. No E13 model, probe, direction, target, or scale is reused in E17.

## 5. Site — frozen relative depth, no causal search

**No causal layer search is permitted.** Every model uses the same relative-depth
rule:

```text
r      = 0.60
layer* = round(0.60 * (num_layers - 1))
```

Teacher and student therefore use different absolute indices when their depths
differ. Site is `resid_post`, selector `last_prompt`, native module name
persisted. For reference, the rule reproduces approximately the Qwen primary
site: Qwen3 has 28 blocks, giving `layer* = 16` against E13's frozen 17.

`D` at nearby layers may be reported descriptively, but the primary causal
analysis stays at `layer*`. The site is never selected using Q/A/G.

## 6. Training regimes

```text
E17-R0  frozen student, evaluated once
E17-R1  hard-label SFT (full-vocabulary cross entropy)
E17-R2  logit KD:  0.5 CE + 0.5 T^2 KL(teacher_T || student_T), T = 2
E17-R3  R2 + hidden-state KD, lambda_H = 1
```

**No conversion-response method.** The CRD branch (E13 R4-R16) is closed and is
not reopened under any outcome. The teacher is frozen throughout.

For R3, hidden states are taken at each model's own `layer*`, each RMS-normalized
as `h / sqrt(mean(h^2) + 1e-8)`, and a jointly trained linear projector maps
student width to teacher width. `L_H` is elementwise mean squared error against
the stop-gradient teacher. Teacher and student always operate in their own hidden
spaces; no direction, probe, or target crosses models.

## 7. Seeds and budget

Training seeds exactly `20261705`, `20261715`, `20261725`.

Budget inherited from E13: 100 optimizer steps, checkpoints `0/10/25/50/100`,
microbatch 2, accumulation 4, effective batch 8, AdamW betas `(0.9, 0.95)`,
epsilon `1e-8`, weight decay `0.01`, gradient clip `1.0`, peak LR `2e-5`,
ten-step linear warmup then cosine decay.

An OOM correction may only lower microbatch while raising accumulation so the
effective batch stays exactly 8, and must be recorded. LR may be changed **only**
if the smoke run demonstrates numerical or training instability, and only before
full discovery, with the replacement frozen in advance. No outcome-based LR
tuning.

## 8. Metrics

```text
B
D_native, and D_initial_frozen where dimension-compatible
Q_raw, Q_z, Q_prob
A_raw, A_z, A_prob
G_raw, G_z, G_prob
strict flip metrics (q-only, context-only, joint; matched and per random seed)
representation similarity: linear CKA, projected cosine, projected MSE
```

Standardization uses each model's own validation margin SD, exactly as in E13.

**COD is not primary in E17 either.** Teacher/student comparison uses
componentwise gaps. COD may appear as a secondary descriptive number only, for
the reasons frozen in the E13 confirmation protocol.

## 9. Behavior-matched checkpoint

Validation-only, identical to E13:

```text
t* = argmin_t |B_val_student(t) - B_val_teacher|,  earliest-checkpoint tie-break
```

Discovery Q/A/G must not affect selection.

## 10. General quality

At step 0, the B-matched step, and the final step: WikiText-2 perplexity and a
fixed 500-example HellaSwag subset, where tokenizer/model compatibility permits.
If a benchmark implementation is incompatible it must be fixed **before** any E17
causal outcome is seen, or marked unavailable. No benchmark may be substituted
after results are viewed.

## 11. Replication criterion (discovery, not confirmation)

The phenomenon is called **replicated** if, for either R2 or R3, all of:

* **A. Representation availability** — `D_student >= 0.95`;
* **B. Behavioral similarity** — `B_S - B_T > -0.03` at the validation-selected
  B-matched checkpoint in at least 2/3 seeds;
* **C. Causal mismatch** — at least one of `Q_z`, `A_z`, `G_z` shows
  `|Delta| >= 0.10` in the **same direction** in at least 2/3 seeds;
* **D. Not catastrophic specialization** — no general-quality collapse that makes
  the behavioral comparison uninterpretable.

These are descriptive discovery thresholds reusing E13's frozen `0.03` and `0.10`
values. E17 is not a preregistered confirmation and no Holm-adjusted claim is
made from it.

## 12. Negative replication is a real result

If the selected pair does **not** reproduce the phenomenon, candidates 2 and 3
must **not** be tried afterwards. Candidate selection ended before any causal
quantity was computed; trying another family after seeing a negative causal
outcome would convert the frozen screen into an outcome-driven search. A failed
replication stands as a genuine negative result and is reported as one.

## 13. Integrity and stop conditions

Per model and regime: finite losses and gradients; no-op intervention exact
within tolerance; hook cleanup; probe identity; target identity; context
orthogonality; context norm matching; factorial algebra; sample/source alignment;
no cross-space direction reuse; teacher and student always in their own hidden
spaces. Immutable run identities; one process per detected GPU; resume only from
an exact atomic checkpoint identity.

E13 confirmation is complete and consumed; E17 must not touch, re-open, or
re-evaluate `e13_confirmation_v1`.
