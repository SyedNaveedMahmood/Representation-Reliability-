# E15 Temporal Causal Half-Life — Discovery Summary

Status: **staged execution complete through Stage 3b. Stage 4 not authorized —
the frozen Stage 4 gate did not pass.**

Protocol: `docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_PROTOCOL.md`, frozen, committed and
pushed at `8b022c8` before any model was loaded and before any causal quantity
was computed.
Design: `docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_DESIGN.md`, unchanged.
Campaign: `runs/E15_TEMPORAL/`. Model: `Qwen/Qwen3-1.7B`, bf16.
Site: `resid_post` L17 (the repository-frozen Qwen3 site; no causal layer search).

E15 is open discovery. No confirmation split was created, materialised or
accessed, and no consumed holdout (E01, E13, E14) was touched.

## Headline

At a token position where the latent state is **perfectly linearly decodable**
(`D_carrier = 1.000`), a full source-free counterfactual setpoint on that decoded
coordinate — moving it across the entire class boundary at
`||delta h|| / ||h|| = 0.187` with essentially perfect numerical fidelity —
produced **no detectable change in the delayed decision the state governs, at any
horizon, including the shortest one**.

Because there is no baseline causal effect at `k0`, a causal decay curve cannot
be measured. The proposed temporal dissociation `H_C < H_D` is therefore
**unresolved at the predeclared carrier, not refuted**.

Verdict: `unresolved_no_causal_handle_at_predeclared_carrier`.

## The frozen task

A deterministic stateful console log. Two terminals each receive one clearance
write (`GRANTED` / `DENIED`); `k` distractor steps follow; then the queried
terminal requests a transfer and the model must approve or refuse. The queried
terminal's flag is the latent state `z`; the other terminal's flag is the frozen
irrelevant-state control and is uncorrelated with the label.

Horizon renderings are **pure prefix extensions** of the same base episode, so an
episode's identity, carrier position, operators, terminals, distractor content
and clearance order are byte-identical across the horizon grid; only the distance
from the state write to the decision varies. Twins flip exactly one clearance
word, and `GRANTED`/`DENIED` are both two Qwen3 tokens, so twin prompts have
identical token length and identical downstream positions.

Corpus: 600 / 200 / 150 pairs (train / validation / discovery_test), rendered at
`k in {1, 2, 4, 8, 16, 32}`; 11,400 samples, 5,700 pairs, no duplicate prompt or
identity, no pair straddling a split.

## Stage outcomes

| Stage | Gate | Outcome |
|---|---|---|
| 0 — freeze task, validate state labels | G0 | **passed** |
| 1 — `D(k)`, `B(k)`, no intervention | G1a–G1d | **passed**; grid truncated to `k in {1,2,4,8}` by G1c |
| 2 — intervention smoke, horizons {1,4,16} | G2 (engineering only) | **passed** |
| 3 — full horizon curve, all arms | G3.1–G3.5 (Stage 4 gate) | **not passed** |
| 3b — secondary distractor density (H15.3) | none | ran; uninformative for `C` |
| 4 — replication on Qwen3-0.6B | gated on G3 | **not run** (runner refused) |

### Stage 0

Label oracle exact on all 11,400 samples; every twin pair differs on exactly one
line and only in the clearance word; `GRANTED` = `[14773, 49468]` and `DENIED` =
`[66186, 9326]` (token parity holds); zero twin token-length mismatches; the
carrier token index is identical across all six horizons of every episode; the
four flag combinations are exactly balanced.

Resolved sites are exactly as specified — carrier `'.\n'` at index 49 ending the
target clearance line, decision `':'` of `Answer:` at the final prompt index.

### Stage 1 — decodability persists, behaviour collapses

| `k` | `D_carrier` | `D_primary(k)` | `D_carry(k)` | `B(k)` | `sigma_m_val(k)` |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.000 | 0.99987 | 0.882 | 0.800 | 0.840 |
| 2 | 1.000 | 0.99964 | 0.866 | 0.833 | 0.837 |
| 4 | 1.000 | 0.99724 | 0.860 | 0.817 | 0.709 |
| 8 | 1.000 | 0.99524 | 0.811 | 0.800 | 0.619 |
| 16 | 1.000 | 0.98547 | 0.764 | **0.560** | 0.548 |
| 32 | 1.000 | 0.97280 | 0.733 | **0.500** | 0.459 |

* **H15.1 is supported.** The state remains near-perfectly decodable at the
  decision token across the whole horizon grid (`0.9999 -> 0.9728`).
* Random-label controls sit at chance (primary-site range `0.454-0.576`, mean
  `0.516`).
* **`B(k)` collapses to exactly chance by `k = 32` while `D_primary(32)` is still
  `0.973`.** This is a large decodability-versus-behaviour dissociation over
  horizon. It is a `D`/`B` result, not the `D`/`C` result E15 targets, and it is
  reported as an observation, not as the E15 claim.
* Because `B(16) = 0.56` and `B(32) = 0.50` fall below the frozen `0.70` floor,
  gate G1c truncated the interpretable grid to `k in {1, 2, 4, 8}`. This decision
  used only non-causal Stage 1 quantities, before any intervention ran, exactly
  as the protocol requires; `k = 16, 32` are reported as uninterpretable rather
  than silently dropped.

Carrier horizon-invariance measured `0.0177` median relative L2, against an
empirical **bf16 shape-noise floor of `0.0175` median** (the same prompt read in
a batch versus alone). The carrier is horizon-invariant to within the numerical
noise floor, as the prefix-extension construction requires.

### Stage 2 — engineering contract

No-op maximum margin deviation **exactly `0.0`**; zero residual hooks left
registered; norm-matched controls match the treatment delta norm to `2.7e-16`;
setpoint projection deviation `0.0029` validation sigma (tolerance `0.05`);
orthogonal-subspace deviation `0.0015` (tolerance `0.02`). G2 contains no
effect-size or effect-sign condition by construction, so it cannot select for a
positive result.

### Stage 3 — the causal null

150 pairs / 300 directed episodes per horizon, 16,800 raw rows.

Intervention magnitude and fidelity (setpoint arm, all horizons):

```text
|q1* - q0*|                       64.5      (validation class medians)
||delta h||                       64.0
||h||                            341.2
||delta h|| / ||h||                0.187
projection relative deviation      0.00024
orthogonal relative deviation      0.00136
```

The decoded coordinate was driven fully across the class boundary, essentially
exactly onto its target. Nothing about the intervention was weak or unfaithful.

`C(k)` — mean oriented margin change, pair-cluster bootstrap, 2000 draws:

| `k` | `C_raw(k)` | 95% CI | `C_z(k)` | `D_primary(k)` |
|---:|---:|---|---:|---:|
| 1 | 0.00333 | [-0.00709, 0.01376] | 0.00397 | 0.99987 |
| 2 | -0.00375 | [-0.01333, 0.00583] | -0.00448 | 0.99964 |
| 4 | 0.00208 | [-0.00708, 0.01083] | 0.00294 | 0.99724 |
| 8 | 0.00583 | [-0.00333, 0.01583] | 0.00943 | 0.99524 |

Every CI includes zero. For scale, the clean decision margin averages `0.55-0.79`
in absolute value, and the validation margin SD is `0.84` at `k0`; the treatment
moves the margin by roughly `0.4%` of one validation SD. Counterfactual flip
rates are `0.3%-2.7%`.

Controls at `k0`, paired per episode:

| control | treatment - control | 95% CI | excludes 0 |
|---|---:|---|---|
| `random_normmatched` | 0.00283 | [-0.00392, 0.00975] | no |
| `orthogonal_normmatched` | 0.00325 | [-0.00309, 0.01059] | no |
| `irrelevant_state` | -0.00417 | [-0.01417, 0.00542] | no |
| `late_position` | 0.00250 | [-0.00708, 0.01167] | no |

The semantic treatment does not separate from any control.

Shuffled future-decision permutation null at `k0` (1000 permutations):
observed `0.00333`, null mean `0.00006`, null SD `0.00522`, `p = 0.564`. The
treatment does not exceed its own diagnostic null.

**Propagation is the mechanism.** Relative L2 change at the decision token,
pooled over horizons:

| arm | L18 | L21 | L24 | L27 |
|---|---:|---:|---:|---:|
| `no_op` | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| `irrelevant_state` | 0.00226 | 0.00713 | 0.00979 | 0.01259 |
| `late_position` | 0.00226 | 0.00706 | 0.00974 | 0.01252 |
| `random_normmatched` | 0.00257 | 0.00736 | 0.00999 | 0.01281 |
| `orthogonal_normmatched` | 0.00256 | 0.00737 | 0.01001 | 0.01280 |
| `setpoint` | 0.00265 | 0.00738 | 0.01001 | 0.01288 |

A 19%-of-norm edit at the carrier reaches the decision token as a `0.3%-1.3%`
perturbation that is **numerically indistinguishable from what a random direction
of the same norm produces**. There is no semantically specific propagation from
this carrier to the decision. `no_op` is exactly zero at every layer, which also
proves the clean and edited forwards are bit-identical when the delta is zero, so
every measured difference above is real rather than batching noise.

### Stage 3b — secondary, H15.3

| condition | gap steps | mean token distance | `C_raw` | 95% CI | `B` |
|---|---:|---:|---:|---|---:|
| `sparse_long_steps` | 6 | 162.0 | 0.00083 | [-0.00917, 0.01083] | 0.837 |
| `dense_short_steps` | 16 | 250.2 | 0.00417 | [-0.00458, 0.01292] | 0.560 |

Both `C` values are indistinguishable from zero, so **H15.3's causal half is
uninformative** given the Stage 3 null. The descriptive `B` contrast is
suggestive that *step count*, not raw token distance, drives the behavioural
collapse — but see the failed control below.

## Falsification, controls and integrity

Passed:

* no-op equality exact (`0.0`), hook cleanup clean, sample identity exact
  through every batch;
* setpoint fidelity far inside the frozen bf16 tolerances at every horizon;
* norm-matched controls matched to `1e-16` relative;
* random-label probe controls at chance;
* pair-cluster bootstrap throughout; no discovery-test label used to fit anything;
* targets and standardisation scales taken from validation only;
* Stage 4 correctly refused to run when the frozen gate failed.

Against the design's own falsification list (design section 10):

* "the representation itself is no longer present" — **not the failure mode**;
  `D_carrier = 1.000` and `D_primary >= 0.995` across the interpretable grid;
* "interventions cease to be numerically faithful at later steps" — **no**;
  fidelity is flat and excellent at every horizon;
* "apparent `C` decay is fully explained by future decision uncertainty" —
  untestable, since there is no `C` decay to explain. `C_raw` and `C_z` agree
  that there is no effect, so the co-primary check is at least consistent;
* "long-horizon errors reflect task-state corruption rather than utilization
  loss" — **this is real** at `k >= 16`, which is exactly why G1c excluded those
  horizons before any causal measurement.

### Limitations and one failed control

1. **The Stage 3b token-distance match failed.** The protocol intended
   approximately matched token distance between the sparse and dense conditions;
   the realised distances were 162 versus 250 tokens, a 54% difference. That
   contrast therefore confounds step count with token distance and cannot support
   any claim. Recorded, not worked around.
2. **Single carrier, single layer, single model, single task.** The null is a
   statement about `resid_post` L17 at the clearance-line-final token in
   Qwen3-1.7B on this environment. It is not a claim that the state is causally
   inert everywhere.
3. **Bypass routes are not excluded.** Editing one position at one layer leaves
   the state-word tokens themselves (two tokens earlier) and every layer below 17
   untouched, so the decision can read the state through routes this intervention
   never intercepts. This is the most likely explanation of the null and is a
   property of a single-site intervention, not of the temporal question.
4. **The interpretable grid is short.** After the G1c truncation only four
   horizons remain, which limits how much curve shape any half-life estimate
   could ever have resolved.
5. `B(k0) = 0.80` rather than near-ceiling, so roughly one episode in five is
   already answered incorrectly before any intervention.

## What E15 does and does not establish

Established:

* **H15.1 — supported.** The state stays near-perfectly decodable at the decision
  token out to 32 intervening steps.
* A large **decodability-versus-behaviour** dissociation over horizon:
  `D_primary(32) = 0.973` while `B(32) = 0.500`.
* A **positional** decodability-versus-causality dissociation: a position with
  `D = 1.000` is not a causal route to the decision it describes.

Not established:

* **H15.2 — unresolved.** With `C(k0)` indistinguishable from zero there is no
  baseline utilization whose decay could be measured. `H_C` is **not estimable**
  (baseline CI includes zero; the curve is not monotone).
* `H_D` is **right-censored at `k = 8`** — `D_rel` never fell to 0.5 inside the
  interpretable grid (`D_rel(8) = 0.991`), so no number is reported for it.
* `H_C < H_D` is therefore **neither supported nor refuted**.
* **H15.3 — uninformative** given the null, and its own control failed.
* **H15.4 — not run.** Stage 4 was refused by the frozen gate.

## Scientifically justified next step

Do **not** repair this result by adding carriers, layers, horizons, arms, tasks
or models to E15 — the protocol forbids it and doing so would convert a frozen
design into an outcome-driven search.

The single defensible next step is a **new, separately frozen carrier-localisation
design** that answers the prerequisite question E15 exposed:

> Where, if anywhere, is the delayed decision's causal read of a remembered state
> variable located?

That design should predeclare a small carrier set (the state-word tokens
themselves, the clearance-line boundary token, and the decision token), sweep
layer depth rather than fixing one, and require a demonstrated causal handle at
`k0` as its entry gate. Only if such a handle exists does the temporal question
E15 asks become measurable, at which point the E15 machinery — corpus, arms,
controls, curve and half-life estimation, all built and tested here — can be
reused unchanged.

Until then E15 stands as a preserved null: **the temporal causal half-life was
not measurable at the predeclared carrier, because there was nothing causal there
to decay.**
