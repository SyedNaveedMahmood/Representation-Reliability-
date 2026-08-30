# E20 Long-Horizon Extension — Results

Status: **stopped after Phase 1 by the frozen stop rule.** Protocol
`docs/E20_LONG_HORIZON_PROTOCOL.md`, frozen, committed and pushed at `6ed6e31`
**before any E20 measurement code existed**. Campaign `runs/E20_LONG_HORIZON/`.
Model `Qwen/Qwen3-1.7B`, bf16. Fresh `e20` namespace (seeds 20262001/2/3),
150 discovery-test pairs per horizon per pool.

Open discovery. No confirmation split; no consumed holdout touched.

## Headline

**The horizon could not be extended, so no causal half-life became estimable.**

Both distractor pools reach exactly `k in {1, 2, 4, 8}` at `B >= 0.70` — the same
grid E19 already used. Under the frozen stop rule ("if Phase 1 selects a grid no
longer than E19's, stop after Phase 1"), E20 ran no interventions and every
half-life remains **right-censored at k=8**, exactly as in E19.

This is the preregistered null outcome, not a failure of the run.

Two things were nevertheless learned, both from non-causal measurement:

* **Step count, not token distance, drives the behavioural collapse** — settling
  the question E15's failed Stage 3b control left open;
* **`D` stays at 1.0000 out to 32 steps / 719 tokens while `B` falls to exactly
  chance**, a far wider decodability-versus-behaviour dissociation than E15
  measured.

## Phase 1 — non-causal only

Every intervention was forbidden here by construction, exactly as E17 screened
candidate models on engineering, behaviour and decodability only. The run records
`causal_quantities_inspected: false`.

| pool | k | B | margin AUROC | mean token distance | `D` source | `D` decision |
|---|---:|---:|---:|---:|---:|---:|
| short | 1 | 0.807 | 0.923 | 48.6 | 1.0000 | 0.9998 |
| short | 2 | 0.823 | 0.928 | 62.2 | 1.0000 | 0.9991 |
| short | 4 | 0.807 | 0.904 | 89.2 | 1.0000 | 0.9939 |
| short | 8 | **0.793** | 0.894 | 143.0 | 1.0000 | 0.9852 |
| short | 16 | 0.540 | 0.833 | 250.8 | 1.0000 | 0.9808 |
| short | 24 | 0.537 | 0.789 | 358.9 | 1.0000 | 0.9697 |
| short | 32 | **0.500** | 0.769 | 466.8 | 1.0000 | 0.9598 |
| long | 1 | 0.837 | 0.956 | 56.5 | 1.0000 | 0.9993 |
| long | 2 | 0.857 | 0.947 | 78.0 | 1.0000 | 0.9978 |
| long | 4 | 0.837 | 0.940 | 120.8 | 1.0000 | 0.9970 |
| long | 8 | **0.737** | 0.922 | 206.1 | 1.0000 | 0.9851 |
| long | 16 | 0.550 | 0.863 | 377.0 | 1.0000 | 0.9677 |
| long | 24 | 0.523 | 0.841 | 548.0 | 1.0000 | 0.9577 |
| long | 32 | **0.500** | 0.792 | 718.8 | 1.0000 | 0.9512 |

```text
reach(short) = [1, 2, 4, 8]      reach(long) = [1, 2, 4, 8]
selection     -> tie -> short, preserving continuity with E15/E18/E19
```

Behaviour falls off a cliff between `k=8` and `k=16` in both pools and reaches
**exactly chance (0.500) at k=32** in both.

### Step count, not token distance

E15's Stage 3b suggested step count drove the collapse, but its own comparison
was confounded — its token distances were 162 versus 250 and it is recorded as a
failed control. E20's Phase 1 is the properly matched test, because the same step
counts are measured at very different token distances:

```text
same step count, different tokens:   k=16 short (251 tok) B=0.540
                                     k=16 long  (377 tok) B=0.550    -> same B

similar tokens, different steps:     k=8  long  (206 tok) B=0.737
                                     k=16 short (251 tok) B=0.540    -> B differs
```

Fifty percent more tokens at the same step count changes `B` by 0.010. Half the
step count at a similar token distance changes it by 0.197. **The collapse tracks
discrete state-tracking steps, not context length.** E15's hint is confirmed, and
the confounded Stage 3b contrast is superseded.

This also explains why the long pool bought nothing: it lengthens each step
without reducing how many there are.

### The dissociation widens

`D` at the source carrier is **1.0000 at every horizon in both pools**, including
`k=32`, where the model answers at exactly chance. At the decision token `D` only
falls from 0.9998 to 0.9598 across the same range.

So over 32 steps and ~719 tokens the state remains essentially perfectly
decodable while the model's forced-choice decision degrades to a coin flip. E15
measured this out to `k=32` with one pool; E20 confirms it across both pools with
a wider token range and shows the graded margin retains partial signal
(margin AUROC 0.923 → 0.769) even where the thresholded decision does not.

## Answering the two questions directly

**Does `D` remain high while `Q/A/G` continue to decay?**

`D` remains high — 1.0000 at the source carrier out to `k=32`. But `Q/A/G` were
**not measured beyond `k=8`**, because the frozen behaviour gate stopped the
experiment there. A decision the model cannot make is not a decision whose causal
organization can be read, so extending the components into the collapsed region
would have produced numbers, not evidence. The `Q/A/G` decay therefore stands
exactly where E19 left it: 0.713, 0.729 and 0.551 of baseline at `k=8`.

**Is any causal half-life legitimately estimable?**

**No.** Nothing changed about the measurable range, so nothing halves inside it.
Every half-life remains **right-censored at k=8**. No persistence timescale is
quoted, and the frozen half-life rule — which requires the relative curve to
actually reach 0.5 inside the measured grid — was not relaxed to manufacture one.

## Gates and integrity

| gate | status |
|---|---|
| G0 corpus | pass — fresh `e20` namespace, pairs complete, no duplicate prompts |
| G2 behaviour | applied in Phase 1 on non-causal data; reach `[1,2,4,8]` for both pools |
| G1 numerics | **not exercised** — no intervention ran |
| G3 carrier sufficiency | **not exercised** — no intervention ran |
| G4 magnitude stability | **not exercised on E20 data** |

G4 is the control E19 discovered post-hoc and this protocol promoted to a
preregistered gate. It is implemented, unit-tested (including that it excludes a
failing curve from H19.2, H19.3, H19.4 *and* from any half-life, not merely from
the outcome label), and wired into E20's Phase 2 — but Phase 2 never ran, so **G4
has not yet fired on real E20 data**. It stands ready for the next experiment of
this shape rather than being validated here.

E19's recorded results are unchanged: `analyze_rows` gained the gate as an
opt-in flag defaulting to off, and re-running E19's analysis reproduces its
committed outcome exactly.

## Limitations

1. **The primary objective was not achieved.** E20 exists to make a half-life
   estimable and it did not.
2. **G4 is preregistered but unexercised.** Its first real test is still ahead.
3. Phase 1 measures `B` on 150 discovery-test pairs per cell, so each `B` carries
   roughly `±0.06` at 95%. The `k=8` versus `k=16` gap (0.793 → 0.540 short) is
   far larger than that, but `long` at `k=8` (0.737) sits closer to the 0.70 bar
   than is comfortable.
4. Single model, single task. The step-count constraint is a property of
   Qwen3-1.7B on this console environment, not a general claim.

## Scientifically justified next step

The binding constraint is now identified precisely: **this environment tops out
at about eight discrete state-tracking steps**, and lengthening the steps does
not help because the cost is per step. Three routes follow, and the honest
ordering is:

1. **Reduce the per-step tracking load rather than the step count.** The current
   distractors are all "Operator X does Y" lines that are structurally identical
   to the state-write line, so every step is a potential interference source. A
   task whose distractors are typographically distinct from state writes might
   push the collapse out several steps. This is a task redesign and needs its own
   frozen design and a fresh E18-style carrier check, because changing the
   environment invalidates the carrier map.
2. **Use a stronger model.** If the collapse is a capability limit, a larger
   checkpoint may track more steps, which would also give the cross-model
   replication E19 lacks. This costs a new carrier map.
3. **Accept the censoring and report the curve, not the timescale.** E19's result
   stands on its own: `D` at ceiling while `Q`, `A` and `G` decline materially,
   with `Q` decaying with propagation distance and `A` with state age. A
   half-life was always a nice-to-have summary, not the claim.

Route 3 is what the current evidence supports. Routes 1 and 2 should not be
started as repairs of E20 — the stop rule forbids that — but as separately frozen
designs if a persistence timescale is judged worth the cost.
