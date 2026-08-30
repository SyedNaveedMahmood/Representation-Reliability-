# E19 Temporal Persistence and Reorganization of Causal Organization — Results

Status: **complete.** Protocol `docs/E19_TEMPORAL_CAUSAL_ORGANIZATION_PROTOCOL.md`,
frozen, committed and pushed at `1e69de1` **before any E19 measurement code
existed**. Campaign `runs/E19_TEMPORAL_ORG/`. Model `Qwen/Qwen3-1.7B`, bf16.
Fresh `e19` namespace (seeds 20261901/2/3), 150 discovery-test pairs per horizon.

Open discovery. No confirmation split; no consumed holdout touched.

## Headline

On a carrier that passes a sufficiency gate at **every** horizon, decodability
stays at ceiling while causal organization decays materially:

```text
D  = 1.000  1.000  1.000  1.000          (k = 1, 2, 4, 8, locus S)

Q  = 1.000  0.994  0.909  0.713          relative to k0
A  = 1.000  0.991  0.852  0.729
G  = 1.000  0.957  0.645  0.551
```

This is the signature E15 set out to measure and could not, because its carrier
carried nothing. Outcome: `componentwise_reorganization`.

The two-locus design then decomposes *why* each pathway decays:

```text
Q decays with propagation DISTANCE, not with state age
A decays with state AGE
```

## Gates

| gate | result |
|---|---|
| G1 numerics | **pass** — no-op deviation exactly `0.0`, norm match `3.6e-16`, projection `0.040` (tol `0.05`), orthogonality `0.0021` (tol `0.02`), no hook leak |
| G2 behaviour | **pass at every horizon** — `B(k)` = 0.843, 0.840, 0.793, 0.820; full grid interpretable |
| G3 carrier sufficiency | **pass at all 8 cells** — flip 0.60-0.69, effects 0.84-1.30, all CIs excluding zero |

G3 is the gate E15 lacked. E15's carrier scored a flip rate of 0.010; both E19
loci score 0.60-0.69 at every horizon, so every `Q/A/G` decomposition below
decomposes an effect that is actually there.

## The required magnitude control

The `native` estimand recomputes its validation setpoint targets at each
horizon. If the class medians converge with horizon, the native edit shrinks and
a falling `Q` would be a **magnitude artifact**, not pathway loss. Measured:

| curve | `‖Δh‖/‖h‖` at k=1 | at k=8 | ratio | magnitude-stable |
|---|---:|---:|---:|---|
| `S_source/native` | 0.4590 | 0.4594 | 1.001 | yes |
| `S_source/ref` | 0.4590 | 0.4594 | 1.001 | yes |
| `D_decision/ref` | 0.0198 | 0.0169 | 0.853 | yes |
| `D_decision/native` | 0.0198 | 0.0089 | **0.452** | **no** |

`D_decision/native`'s edit shrinks by more than half across the grid, so its
apparent `Q` collapse to 0.312 is **discarded as a magnitude artifact** and
excluded from the outcome label. Every number quoted as a result below comes
from a magnitude-stable curve. This control was not in the protocol; it was added
after the run when the raw magnitudes were inspected, and it changes the reading
of one of the four curves, so it is reported prominently rather than buried.

## H19.1 — representational persistence: SUPPORTED

| locus | `D(k0)` | `D(k*)` | change | non-inferior at −0.05 |
|---|---:|---:|---:|---|
| `S_source` | 1.0000 | 1.0000 | 0.0000 | yes |
| `D_decision` | 0.9992 | 0.9942 | −0.0051 | yes |

The state is essentially perfectly decodable at both loci at every horizon.

## H19.2 — causal organization changes while representation persists: SUPPORTED

Change from `k0` to `k*=8`, episode-cluster bootstrap resampling whole horizon
curves, Holm-corrected across `Q/A/G`:

| curve | component | rel at k=8 | change | 95% CI | SESOI | Holm p | supported |
|---|---|---:|---:|---|---:|---:|---|
| `S_source/native` | Q | 0.713 | −0.1025 | [−0.130, −0.073] | 0.089 | 0.000 | **yes** |
| `S_source/native` | A | 0.729 | −0.1613 | [−0.214, −0.105] | 0.149 | 0.000 | **yes** |
| `S_source/native` | G | 0.551 | −0.1229 | [−0.160, −0.090] | 0.068 | 0.000 | **yes** |
| `S_source/ref` | Q | 0.705 | −0.1054 | [−0.133, −0.075] | 0.089 | 0.000 | **yes** |
| `S_source/ref` | A | 0.744 | −0.1525 | [−0.208, −0.095] | 0.149 | 0.000 | **yes** |
| `S_source/ref` | G | 0.519 | −0.1317 | [−0.165, −0.100] | 0.068 | 0.000 | **yes** |
| `D_decision/ref` | Q | 1.000 | +0.0000 | [−0.012, +0.012] | 0.023 | 1.000 | no |
| `D_decision/ref` | A | 0.658 | −0.4108 | [−0.491, −0.332] | 0.300 | 0.000 | **yes** |

`G` at `D_decision` is **not assessable**: its `k0` value is indistinguishable
from zero, so no relative curve is reported for it rather than manufacturing a
ratio from a near-zero denominator.

All half-lives are **right-censored at k=8** — no component fell to half its
baseline inside the interpretable grid. No half-life number is reported, per the
frozen smoothness/censoring rule.

## H19.3 — differential pathway persistence: mixed

Supported only at `D_decision/ref`, where `Q` is flat at 1.000 while `A` falls to
0.658 (relative difference +0.342, CI [+0.209, +0.491]).

**Not supported at `S_source`.** There the three components decline together:
`A_vs_G` differs by 0.178 and `Q_vs_G` by 0.162, both below the frozen 0.25
SESOI. `G` declines fastest in point estimate (0.551 versus 0.713 and 0.729) but
not by enough to clear the threshold. Recorded as a near-miss, not a result.

## H19.4 — state age versus remaining distance

This is what the two-locus design was built for. Locus D grows state age with the
distance to the decision held at ~0; locus S grows both.

| component | `S_rel` at k=8 (age + distance) | `D_rel` at k=8 (age only) | reading |
|---|---:|---:|---|
| Q | 0.705 | **1.000** | decays with **distance**, not age |
| A | 0.744 | **0.658** | decays with **age**; distance adds nothing |
| G | 0.519 | not assessable | no decomposition available |

So the scalar and additive pathways decay for **different reasons**. Holding the
propagation path fixed, the scalar coordinate keeps its full causal efficacy no
matter how old the state is; the additive-context pathway loses a third of its
efficacy to state age alone.

Propagation corroborates the `Q` reading — the perturbation from locus S reaches
the decision token less as distance grows:

| k | `P_norm` L12 | L17 | L21 | L27 |
|---:|---:|---:|---:|---:|
| 1 | 0.0153 | 0.0552 | 0.0731 | 0.0741 |
| 2 | 0.0138 | 0.0532 | 0.0701 | 0.0732 |
| 4 | 0.0133 | 0.0453 | 0.0580 | 0.0613 |
| 8 | 0.0126 | 0.0384 | 0.0497 | 0.0615 |

## Code rotation versus pathway loss

The protocol's two estimands answer the review's rotation question directly.

```text
cos(u_0, u_k)      k=1     k=2     k=4     k=8
S_source           1.000   1.000   1.000   1.000
D_decision         1.000   0.784   0.660   0.524
```

At the source token the decoded axis does not rotate at all. At the decision
token it rotates substantially — and yet `Q` measured on the **frozen k0 axis**
stays flat at 1.000 there.

The decodable direction at the decision token rotates away from the causally
effective one, which does not move. A probe tracks the rotation; the causal
pathway does not. This is exactly the distinction the review asked for, and it
cuts against reading a probe axis as the model's endogenous causal code.

## Limitations

1. **The magnitude control was added after the run**, not preregistered. It
   invalidates one of four curves. The three surviving curves were unaffected by
   it, but the control belongs in any future protocol of this shape.
2. **`D_decision/ref` drifts mildly** in magnitude (ratio 0.853). Its `A` decline
   of 34% is much larger than that drift, but its `Q` flatness should be read as
   "no decline detected at roughly constant magnitude", not as an exact null.
3. **Four horizons, all right-censored.** No half-life is estimable; the grid
   ends before any component halves.
4. **`G` is small and at `D_decision` not assessable at all**, so the interaction
   pathway is only characterised at locus S.
5. Single model, single task, single environment. Discovery only.

## What E19 establishes

* The E15 target signature is real in this task, once measured on a sufficient
  carrier: **`D` at ceiling across the horizon while `Q`, `A` and `G` fall to
  0.52-0.74 of baseline**, every decline exceeding its SESOI with Holm-corrected
  `p` below 0.05.
* The components do **not** decay for the same reason: `Q` decays with
  propagation distance, `A` with state age.
* At the decision token the decodable axis rotates while the causal direction
  does not — decodability and causal efficacy come apart *geometrically*, not
  only in magnitude.
* They do **not** decay at materially different *rates* at the source locus, so
  the strong "reorganization rather than decay" claim is not supported there.

## Scientifically justified next step

The obvious extension is a second checkpoint or family, since every number here
is one model on one task. But the more informative next move is **extending the
horizon grid** so the curves are not right-censored: no component halves by
`k=8`, so no persistence timescale can be quoted. E15 showed behaviour collapses
by `k=16` with short distractor steps, while its Stage 3b hinted that step
*count*, not token distance, drives that collapse. A newly frozen design using
fewer, longer distractor steps could plausibly reach much longer token horizons
with `B` intact, which is what a half-life estimate would require.

That design should preregister the magnitude control from this run as a gate
rather than a post-hoc check.
