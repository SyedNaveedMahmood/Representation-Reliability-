# E15 Temporal Causal Half-Life — Execution Protocol

Status: **frozen before any model is loaded and before any E15 causal quantity is
computed.** Committed and pushed before execution.

This protocol *implements* `docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_DESIGN.md`. It does
not revise it. Every choice below is either quoted from the design or is an
engineering resolution taken from an analogous frozen experiment (E01A, E01B-1,
E14, E17). Where the design leaves an option open, the option is chosen here,
before execution, and never re-opened.

E15 is **open discovery**. No preregistered confirmation is claimed, no
confirmation holdout is created, consumed or touched, and no p-value produced
here is confirmatory. The E01, E13 and E14 holdouts are consumed/locked and are
not read by any E15 code path.

## 0. Scope discipline

The design's staged plan (design section 11) is the authorization structure:

```text
Stage 0  choose one synthetic/stateful task and validate exact state labels
Stage 1  establish D(k) without intervention
Stage 2  intervention smoke at 2-3 horizons
Stage 3  full horizon curve for one model
Stage 4  only if the dissociation exists, replicate on one model/checkpoint/task
```

Primary hypotheses executed here are **H15.1** (temporal decodability
persistence) and **H15.2** (utilization decay).

**H15.3** (distractor sensitivity) is executed only as a bounded *secondary*
contrast (Stage 3b) after Stage 3 completes, and never contributes to a primary
claim. **H15.4** (checkpoint dependence) belongs to Stage 4 and is executed only
if the Stage 4 gate in section 9 passes.

No additional hypothesis is introduced. No knob is added after any outcome is
seen.

## 1. Stage 0 — the frozen task

Domain choice, from the design's preferred list: **stateful text environment with
explicit hidden state**. Not an open-web or coding agent.

### 1.1 Environment

A deterministic single-episode console log. One episode renders to one
completion-style prompt (no chat template), matching E00/E01 convention.

```text
Console log. A terminal may run a transfer only if its clearance is GRANTED.
Step 01: Operator Karsi logs the intake pressure.
Step 02: Operator Sidra sets the clearance for terminal Reko to DENIED.
Step 03: Operator Ovell sets the clearance for terminal Luma to GRANTED.
Step 04: Operator Vanta checks the coolant level.
...
Step NN: Terminal Luma requests to run a transfer.
Question: Should the request be approved?
Answer:
```

Fixed episode skeleton:

```text
header
1 prefix distractor step
2 clearance-setting steps       (order randomised per episode, balanced)
k gap distractor steps          (k = the horizon)
1 request step
question line
"Answer:"
```

### 1.2 Latent state variable

`z` = the clearance flag of the **queried** terminal, in `{GRANTED, DENIED}`.
Decision label `y = 1` (Yes / approve) iff `z = GRANTED`.

Both terminals carry an independent clearance flag. The **non-queried** terminal's
flag is the frozen **irrelevant state variable** used by the control in section 6.
The four (target, irrelevant) flag combinations are balanced by construction, so
the irrelevant flag is uncorrelated with the label.

Design requirement check (design section 3):

1. clearly defined latent variable — the binary clearance flag;
2. deterministic auditable transitions — one write per terminal, no later writes;
3. known future decisions depending on the variable — the approval decision;
4. controllable nuisance factors — operators, terminals, distractor content,
   clearance-step order, horizon, all independently controlled;
5. trajectories long enough to vary horizon — `k` up to 32 steps;
6. natural intervention semantics — the state-write step has a well-defined
   carrier token;
7. enough repeated episodes for cluster-aware inference — 150 discovery-test
   matched pairs.

### 1.3 Matched counterfactual twins

Each episode has exactly one twin: **only the queried terminal's clearance word is
flipped**. Everything else — header, operators, terminals, distractor text, step
numbering, clearance-step order, horizon, the irrelevant flag — is identical, and
the decision label flips. Twins share `pair_id` and are never split across splits.
All inference clusters on `pair_id`.

`GRANTED` and `DENIED` are used because both encode to exactly two Qwen3 tokens,
so twins have identical token length and every downstream absolute token position
is identical. Token parity is re-verified at runtime as a Stage 0 gate.

### 1.4 Horizon rendering — prefix extension

The frozen horizon grid is

```text
k in {1, 2, 4, 8, 16, 32}      k0 = 1
```

For one base episode, the horizon-`k` prompt is produced by inserting the first
`k` of that episode's deterministic gap distractors between the clearance block
and the request step. Therefore **every prompt prefix up to and including the
clearance block is byte-identical across all horizons of the same episode**, and
because the model is causal the carrier activation is horizon-invariant. This is
verified numerically as a Stage 0/1 gate. Horizon varies the distance from the
state write to the decision and nothing else before the carrier.

### 1.5 Corpus

Fresh namespace `e15-{split}-v1`. No E15 confirmation split exists.

```text
train:           1200 directed /  600 pairs   seed 20261501
validation:       400 directed /  200 pairs   seed 20261502
discovery_test:   300 directed /  150 pairs   seed 20261503
```

Prompts are globally deduplicated across splits at the base-episode level; pairs
are never split. All causal evaluation happens on `discovery_test` only.

### 1.6 Stage 0 gate G0

All of the following must hold, else E15 stops as an engineering failure:

* the label oracle reproduces every stored label exactly;
* each twin pair differs in exactly one clearance word and in nothing else;
* `GRANTED`/`DENIED` have equal token length under the model tokenizer;
* the target clearance line occurs exactly once in its prompt;
* the carrier token index is identical across all six horizons of every episode;
* no duplicate prompts or sample ids; every pair is complete;
* the four flag combinations are balanced within each split.

## 2. Model, site and carrier

Primary model: **`Qwen/Qwen3-1.7B`**, bf16, the checkpoint whose causal mechanism
is already frozen by E01A/E01B/E14.

**No causal layer search is permitted.** E15 reuses the repository's already
frozen Qwen3 site rather than selecting a new one:

```text
site     resid_post
layer    17            (0-indexed; frozen by E01A/E01B-1/E01B-2/E01B-3/E13/E14)
```

Design section 5 requires one predeclared carrier and one predeclared token site.

```text
primary carrier   transformer residual stream (resid_post, L17)
carrier token     last token of the TARGET clearance-setting step line
                  (selector `target_span_last` over that exact line)
decision token    last prompt token ("Answer:" position, selector `last_prompt`)
```

No other carrier (memory token, KV-derived state, scratchpad) is used, per the
design's instruction not to mix carriers in the first experiment.

Decision readout: first-token logits of `" Yes"` / `" No"`; margin
`m = logit(Yes) - logit(No)`; prediction `1` iff `m >= 0`. Identical to E01A.

## 3. Stage 1 — D(k) without intervention

Probes are logistic regression, standardized, `C` grid `[0.01, 0.1, 1.0, 10.0]`,
fitted on `train` and selected on `validation` only, evaluated on
`discovery_test`. Probe seed `20261510`.

Predeclared decodability quantities:

```text
D_carrier      AUROC for z at the carrier (resid_post L17, target clearance line)
               horizon-invariant; this is the k = 0 anchor
D_primary(k)   AUROC for z at the DECISION token (resid_post L17) of the
               horizon-k prompt                                <-- primary D
D_layer(k, l)  same, at l in {18, 21, 24, 27}, descriptive only
D_carry(k)     AUROC for z at the last token of the k-th gap distractor step,
               descriptive persistence view
```

`D_primary` is the primary D because it is the decodability that exists at the
moment and place where the decision is produced — the quantity `C(k)` is paired
with.

Also recorded per horizon, without intervention:

```text
B(k)            forced-choice accuracy on discovery_test
sigma_m_val(k)  validation clean-margin standard deviation (standardisation scale)
mean |m|(k)     clean margin magnitude
```

Random-label controls (three seeds) at every probe site, per repository Gate 1.

### 3.1 Stage 1 gate G1

* **G1a** `D_carrier >= 0.90` on discovery_test. Failure means there is no
  decoded coordinate to set and E15 stops.
* **G1b** `D_primary(k0) >= 0.90`. Failure means the state is not present at the
  decision point even at the shortest horizon and E15 stops.
* **G1c** `B(k) >= 0.70` at every horizon. If some horizons fail, the frozen grid
  is truncated to its **longest prefix** of horizons that pass, and every later
  horizon is reported as uninterpretable rather than silently dropped. If `k0`
  fails, E15 stops. `B(k)` is a *non-causal* quantity computed before any
  intervention runs, so this truncation cannot be outcome-driven.
* **G1d** random-label AUROC within `[0.40, 0.60]` at the primary sites.

## 4. Intervention

Source-free setpoint, per design section 6 ("where possible, use source-free
setpoints rather than donor-state patching"). Math is the frozen E01B-1
implementation, unchanged.

```text
u          unit probe direction at the carrier (resid_post L17), from `train`,
           C selected on `validation`
q0*, q1*   class-conditional MEDIAN carrier coordinates on `validation`
q_target   opposite-class median for the base's own label y:
              y = 1 -> q0* ;  y = 0 -> q1*
delta      (q_target - u.h_base) * u        (rank-one; orthogonal complement fixed)
```

Expected counterfactual label is `1 - y`. The oriented effect is E01A's

```text
delta_margin_toward_expected = m_toward(m_after, 1-y) - m_toward(m_before, 1-y)
```

with `m_toward(m, l) = m if l == 1 else -m`.

## 5. Primary measurements

```text
C_raw(k)   mean over discovery_test episodes of delta_margin_toward_expected
           for the setpoint arm at horizon k              <-- primary C
C_z(k)     C_raw(k) / sigma_m_val(k)                      <-- co-primary
```

`C_z` is co-primary specifically to address design falsification bullet 2
("apparent C decay is fully explained by future decision uncertainty"): if the
clean margin scale changes with horizon, `C_raw` and `C_z` can disagree. **If the
two curves disagree in verdict, E15 is reported as unresolved, not positive.**

Cluster bootstrap on `pair_id`, 2000 draws, 95 percent, seed `20261530`.

### 5.1 Propagation / persistence (design section 6)

In the treatment forward at horizon `k`, `resid_post` at layers
`{18, 21, 24, 27}` is captured at the **decision token** (layers strictly after
the edit layer, because an edit at L17 cannot change L17 at any other position):

```text
P_norm(k, l)  ||h_dec_intervened - h_dec_clean|| / ||h_dec_clean||
P_q(k, l)     change in the decision-site probe coordinate at layer l
```

## 6. Controls (design section 8)

At every horizon, on the same discovery-test episodes:

```text
clean                    no intervention (and an unhooked forward for no-op check)
setpoint                 TREATMENT, section 4
random_normmatched       norm-matched random direction at the carrier, 5 seeds
orthogonal_normmatched   norm-matched random direction orthogonal to u, 5 seeds
irrelevant_state         same u and same q_target applied at the NON-queried
                         terminal's clearance-line carrier
late_position            same u and same q_target applied at the final gap
                         distractor's carrier (always distance 1 from the
                         decision): a content-free, position-late control
```

Five random and five orthogonal directions per horizon (E01B-1 used ten; E15
multiplies every arm by six horizons, so five is frozen here in advance).
Direction seed base `20261520`.

Further required controls, satisfied by construction or by analysis:

* matched horizon/distractor count — the prefix-extension rendering of 1.4;
* position control — the carrier's absolute token index is identical across all
  horizons, plus the `late_position` arm;
* action-frequency control — twins are label-balanced and the decision template
  is identical at every horizon;
* episode-level cluster bootstrap — all CIs cluster on `pair_id`;
* shuffled future-decision mapping — a diagnostic null in which each episode's
  observed margin change is scored against a **permuted** expected
  counterfactual label (1000 permutations, seed `20261540`); the treatment
  effect must exceed this null.

## 7. Stage 2 — intervention smoke

Horizons `{1, 4, 16}`, first 30 discovery-test pairs by sorted id, every arm.

### 7.1 Stage 2 gate G2 — engineering only

Deliberately contains **no effect-size or effect-sign condition**, so the smoke
cannot select for a positive result.

* no-op (zero delta) maximum selected-logit deviation `<= 1e-6` versus the
  unhooked forward;
* zero residual forward hooks left registered after every batch;
* setpoint fidelity within the frozen bf16 tolerances of
  `interventions.setpoint.setpoint_fidelity_tolerances("bfloat16")`;
* exact row/sample identity through every batch;
* norm-matched controls match the treatment delta norm to `<= 1e-6` relative.

## 8. Stage 3 — full horizon curve

All 150 discovery-test pairs (300 directed episodes), all horizons surviving
G1c, all arms.

### 8.1 Half-life definitions (design section 7)

```text
C_rel(k) = C(k) / C(k0)                                   k0 = 1
D_rel(k) = (D_primary(k) - 0.5) / (D_primary(k0) - 0.5)
```

`H_C` is the smallest `k` with `C_rel(k) <= 0.5`, obtained by linear
interpolation between the bracketing grid points. `H_D` is defined identically
from `D_rel`.

The design permits reporting a half-life **only if the curve is sufficiently
smooth/monotonic**. Frozen criterion, applied identically to both curves:

* Spearman rho between `k` and the relative curve `<= -0.70`; and
* no upward step between adjacent grid points greater than `0.15`; and
* `C(k0)` (respectively `D_primary(k0) - 0.5`) is positive with a
  cluster-bootstrap CI excluding zero.

If the criterion fails, the half-life is reported as **not estimable** and the
raw curve is reported instead. If the curve never reaches `0.5` inside the grid,
the half-life is reported as **right-censored at `k_max`**, never as a number.

### 8.2 Stage 3b — secondary, H15.3 only

Run only after Stage 3 completes. At an approximately matched *token* distance
between the state write and the decision, compare a **sparse** rendering (6 long
gap steps) with a **dense** rendering (16 short gap steps). Arms: `clean`,
`setpoint`, `random_normmatched` (5 seeds). Reports the change in `C` and in
`D_primary` between the two densities. Explicitly secondary; contributes to no
primary claim and to no gate.

## 9. Stage 4 gate — escalation rule

Stage 4 runs **only** if all of the following hold on Stage 3 output:

* **G3.1** `C_raw(k0)` mean positive with cluster-bootstrap CI excluding zero,
  **and** the treatment exceeds every control arm at `k0` in a paired contrast
  whose CI excludes zero (`random`, `orthogonal`, `irrelevant_state`,
  `late_position`);
* **G3.2** `H_C` is estimable under 8.1 and finite inside the grid, for **both**
  `C_raw` and `C_z`;
* **G3.3** `D_rel(H_C) >= 0.90` — the representation is still essentially present
  at the causal half-life, i.e. `H_C < H_D`;
* **G3.4** no fidelity or identity failure at any horizon;
* **G3.5** the treatment exceeds the shuffled-decision permutation null at `k0`.

If the gate passes, Stage 4 replicates the frozen pipeline unchanged on the
second frozen checkpoint **`Qwen/Qwen3-0.6B`** (chosen and recorded here, before
any causal outcome is seen, because it is the repository's other frozen
checkpoint and the natural test of H15.4). No third model, no task substitution,
no re-tuning.

If the gate fails, E15 stops and the null or unresolved result stands as the
result. A failed gate must not be repaired by adding horizons, arms, layers,
carriers, tasks or models.

## 10. Integrity and stop conditions

* Immutable deterministic run identities; atomic status; per-example raw
  evidence written before any aggregate.
* Every arm records `||delta h||`, `||h||` and their ratio.
* No probe, direction, target or scale crosses models or splits.
* Targets and standardisation scales come from `validation` only.
* Discovery-test labels are never used to fit anything.
* A failed control, a fidelity breach or a numerical non-finite value is
  recorded and stops the affected stage; it is never worked around.
* Null and negative outcomes are preserved verbatim in the summary and in the
  registry.

## 11. Claim boundary

A positive E15 supports only a task-, model-, site- and discovery-specific
statement that functional influence can decay faster than representational
accessibility over a trajectory. It is not an exponential decay law, not a memory
theory, and not a cross-model claim without Stage 4.
