# Actionable Representation Research Roadmap

Status date: 2026-08-28

Master question:

> **What makes an internal representation actionable?**

This roadmap converts the master question into a gated experiment program.

## Phase A — establish the causal object

### A1. E01B-1 source-free setpoints — full discovery complete

Goal: establish whether donor-free scalar targets reproduce the E01A causal response.

Success signature:

```text
source-free semantic setpoints -> monotonic native-readout response
semantic treatment > random/orthogonal controls
1.7B conversion > 0.6B conversion
```

Outcome: donor-free semantic setpoints reproduce E01A, beat norm-matched
random/orthogonal controls, and preserve the large 1.7B-over-0.6B conversion
gap. The 1.7B grid is strongly monotonic; 0.6B has a positive population slope
but weak per-example monotonicity. Confirmation remains untouched.

### A2. E01B-2 orthogonal-context modulation — full discovery complete

Goal: determine whether the same scalar semantic displacement has stable causal meaning across orthogonal representational contexts.

Success outcomes are informative in either direction:

- context-insensitive -> approximately one-dimensional causal control;
- structured-context-sensitive -> context-gated utilization;
- random-context-sensitive -> semantic specificity concern.

Outcome: structured opposite-label orthogonal contexts produce much larger
effects than norm-matched random context at the same scalar target in both
models. Matched and same-family effects exceed different-family effects in the
aggregate, although the ordering is heterogeneous by relation family. The
effect enters native readout immediately at L17 while q is fixed and then
changes downstream q propagation. This supports operational context-gated
utilization, while leaving additive orthogonal signal versus multiplicative
gating unresolved. Confirmation remains untouched.

### A3. E01B-3 additive-vs-gating factorial decomposition — strongly confirmed

Goal: separate the E01B-2 context increment into an independent context-only
effect and a true interaction with the frozen scalar intervention.

The pre-registered four-arm design is:

```text
Y00 clean
Y10 semantic setpoint only
Y01 orthogonal context only
Y11 semantic setpoint plus context
```

It estimates additive context signal `A = Y01-Y00` and factorial interaction
`G = (Y11-Y10)-(Y01-Y00)`. Site, targets, source plans, context norms, lambdas,
controls, traces, and pair-cluster inference are frozen from E01B-1/E01B-2.
Full discovery is complete. In 0.6B, structured context increments are almost
entirely additive: structured A is positive, while G and all
structured-minus-random G contrasts include zero. In 1.7B, structured A is much
larger and a smaller positive structured G also exceeds random, supporting a
mixed additive-plus-gating result. L17 q remains fixed; additive readout effects
appear immediately and 1.7B develops downstream q/readout interaction. The
additive matched/same-family advantage does not carry over to G; different-
family interaction is largest in 1.7B. The single preregistered confirmation
passed H1-H4 after Holm correction. The core mechanism is frozen: scalar q is
causally effective; structured orthogonal state carries independent causal
information in both checkpoints; and 1.7B additionally shows structured
q-by-context interaction. No such interaction was detected in 0.6B, without an
equivalence claim.

## Phase B — test fragility under compression

### B1. E14 quantization reliability — default first extension

Status: **strong E14 confirmation complete; mechanism frozen** under
`docs/E14_FULL_DISCOVERY_AND_CONFIRMATION_PROTOCOL.md`.

Goal: test whether representational availability survives compression better than causal utilization.

Start with Qwen3-1.7B:

```text
BF16 -> INT8 -> INT4
```

Primary question:

```text
Can D remain near ceiling while scalar Q, multidimensional additive A, or
q-by-context interaction G drops?
```

The bounded Qwen3-1.7B ladder uses one weight-only backend (Optimum-Quanto
0.2.7) across BF16/INT8/INT4. It measures precision-native and frozen-BF16-axis
D, behavior B, source-free Q0, matched/random A and G, and depth tracing.

Why first:

- no retraining required;
- low compute;
- same checkpoint controls many confounds;
- direct deployment relevance;
- high novelty-to-effort ratio.

Do not add INT3/INT2 until BF16/INT8/INT4 are stable.

Bounded result: INT8 preserved both decodability views and Q/A/G. INT4 kept
precision-native D at ceiling but reduced frozen-BF16-axis D, reduced
structured-minus-random A by 14.9%, and reduced structured-minus-random G by
49.3%; Q increased. Native relation-margin discrimination also declined, while
prompt perplexity did not catastrophically worsen. The pilot therefore supports
higher-order compression fragility but does not yet isolate semantic-specific
damage from all task-level degradation. Full discovery is now authorized with
300 directed examples, ten random seeds, both frozen lambdas, and preregistered
WikiText/HellaSwag controls. A new E14-specific holdout may be accessed once only
if native INT4 D remains at least 0.99 and the paired G-reduction CI excludes
zero. See `E14_BOUNDED_PILOT_SUMMARY.md`.

Full discovery passed that gate. Across 300 directed examples, INT4 retained
precision-native D at 0.99991 AUROC but reduced structured-minus-random A by
26.0% and G by 53.9%; both paired CIs excluded zero, while Q increased by 17.7%.
WikiText-2 perplexity increased 65.2%, crossing the frozen generic-damage flag,
while HellaSwag accuracy fell by 0.07. Consequently, confirmation is authorized
but any positive result must be described as mixed actionability and general
degradation rather than selective semantic damage. See
`E14_FULL_DISCOVERY_SUMMARY.md`.

The single E14 confirmation then passed H14.1-H14.3 after Holm correction:
native INT4 D remained 1.0, while structured-minus-random A and G were both
lower than BF16 with directional paired CIs excluding zero. Because WikiText
PPL again crossed the generic-damage flag, freeze the claim as mixed
actionability plus general degradation. The E14 holdout is consumed. This strong
confirmation unlocks the separately preregistered bounded E13 distillation
branch; it does not authorize reopening E14 or testing a new quantizer.

## Phase C — test transfer and learnability

### C1. E13 distillation reliability — gated on E01B

Status: **discovery complete; method branch closed as diagnostic**. Three-seed
R0/R1/R2/R3 and R4/R5/R6/R2-C full discovery passed all frozen baseline Gates
A-E and all 12 authorized method jobs completed. The subsequent method-revision
campaign (R7-R16, one bounded seed, `runs/E13_METHOD_REVISION/`) then failed its
frozen bounded gate with `selected_regime: null`, so no three-seed method wave
was authorized or run. Classification `METHOD REVISION INCONCLUSIVE`;
recommendation is to retain E13 as diagnostic evidence and not to preregister a
confirmation. E13 confirmation remains locked, unmaterialized, and unaccessed.
See `E13_METHOD_REVISION_DISCOVERY_SUMMARY.md`.

Pilot result: D was saturated throughout and both SFT/KD reached B=1.0, but Q
remained weak—especially under KD—while A/G changed strongly and differently by
objective. SFT substantially overshot teacher A/G; KD ended near teacher G but
far above teacher A and near baseline Q. This justifies multi-seed replication
and triggers a proposed conversion-response objective, but neither is authorized
by the one-seed pilot. The newly authorized discovery adds validation-scaled,
probability, flip, causal-organization-distance, behavior-matched-checkpoint,
representation-similarity, and general-quality views. See
`E13_BOUNDED_PILOT_SUMMARY.md` and the multi-seed protocol.

Goal: determine whether distillation transfers representation and utilization together or separately.

Primary regimes:

```text
frozen student
hard-label SFT
logit KD
hidden-state/trajectory KD
optional combined KD
```

Track:

```text
D(t), readout(t), C(t), B(t)
```

High-value result:

```text
D already high
C changes substantially under KD
```

or hidden-state similarity improves without corresponding C transfer.

### C2. conversion-response distillation — discovery complete

The frozen multi-seed gate passed and the preregistered R4/R5/R6/R2-C campaign
is complete. R5 had the lowest mean COD (`0.543`) but failed the frozen primary
criterion because one seed missed teacher-like validation B; R5 also failed to
improve all Q/A/G gaps and did not consistently beat R6. R2-C exactly matched
R2, ruling out extra detached-forward compute as the explanation. The current
classification is heterogeneous/unresolved method evidence, and the next action
is to revise the conversion-response method in a newly frozen discovery design,
not to access confirmation.

Trigger only if R2 or R3 achieves teacher-like validation behavior in at least
two seeds while a preregistered standardized causal-organization gap remains and
general quality is interpretable. If the gate passes, the executed method
protocol must be committed and pushed before any R4/R5/R6/R2-C training.

Then test whether the student's response to source-free latent setpoints can be trained to match the teacher's response curve.

## Phase D — test emergence during training

### D1. E16 utilization emergence — checkpoint family required

Goal: determine whether representational accessibility emerges before causal utilization during training.

Primary signature:

```text
D(t) plateaus
while
C(t) continues to rise
```

This would turn a static checkpoint comparison into a developmental mechanism.

Only start after identifying a public checkpoint family with stable architecture and sufficient intermediate checkpoints.

## Phase E — test temporal persistence

### E1. E15 temporal causal half-life — discovery complete, unresolved null

Goal: determine whether a represented state remains decodable after its causal influence on future actions has decayed.

Primary signature:

```text
H_C < H_D
```

where `H_C` is causal half-life and `H_D` is representation half-life.

Status: **executed 2026-08-30 through Stage 3b under
`docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_PROTOCOL.md` (frozen at `8b022c8`); the
Stage 4 gate did not pass and Stage 4 was not run.**

The frozen task is a deterministic stateful console log with one binary clearance
flag, a delayed approval decision, matched counterfactual twins, and a second
uncorrelated flag as the irrelevant-state control. Horizon renderings are pure
prefix extensions, so only the state-write-to-decision distance varies.

Outcome: `H15.1` is supported — the state stays near-perfectly decodable at the
decision token across the whole grid (`D` falls only `0.9999 -> 0.9728` from
`k=1` to `k=32`). But a full source-free counterfactual setpoint at the
predeclared carrier (`resid_post` L17, the clearance-line-final token), driving
the decoded coordinate across the entire class boundary at
`||delta h||/||h|| = 0.187` with near-exact fidelity, produced **no detectable
change in the delayed decision at any horizon including `k0`**
(`C(1) = 0.0033`, CI `[-0.0071, 0.0138]`), did not separate from any control, and
did not exceed its own shuffled-decision null (`p = 0.564`). Propagation to the
decision token was numerically indistinguishable from a random direction of the
same norm.

Consequently `H_C` is not estimable, `H_D` is right-censored at `k=8`, and
`H_C < H_D` is **neither supported nor refuted**. `H15.2` is unresolved, not
falsified: a decay curve cannot be measured from a null baseline.

Two preserved side results. First, behaviour collapses to exactly chance by
`k=32` (`B = 0.500`) while `D` is still `0.973` — a decodability-versus-behaviour
dissociation over horizon; the frozen G1c rule truncated the interpretable grid
to `k in {1,2,4,8}` on that non-causal basis before any intervention ran. Second,
a position with `D = 1.000` turned out not to be a causal route to the decision
it describes — a positional, rather than temporal, D/C dissociation.

A Gate 1 carrier-sufficiency addendum (2026-08-30, required by the direction
review section 10.2) then settled *why*. Replacing the entire carrier state with
its exact counterfactual twin - a 40%-of-norm edit, twice the frozen setpoint -
flips the delayed decision in only 1% of episodes, with a mean-effect CI that
includes zero, although it does beat a same-norm random patch. The carrier holds
a faint real semantic signal but is roughly two orders of magnitude too weak to
support a decay-curve study.

So the Stage 3 null is a statement about the **carrier**, not about nonlinearity:
the causal code is not hiding outside the `Q/A/G` decomposition at this site, the
site is simply not where the delayed decision reads the state. The flagship
temporal experiment is not buildable on this carrier as frozen.

Do not repair E15 by adding carriers, layers, horizons, arms, tasks or models.
See `E15_TEMPORAL_CAUSAL_HALF_LIFE_SUMMARY.md`.

E15 stays closed. The prerequisite it exposed was answered separately by E18
below, which found the causal read one token earlier and about ten layers
earlier than E15's frozen site.

## Phase F - validity audits

### F1. E01 cross-checkpoint calibration audit - complete

Triggered by `docs/Reproduction_Reliability_Next_Direction_Review.md` section 8:
raw-unit interventions at a fixed layer and coefficient can manufacture apparent
cross-model scaling trends, so E01's single-point `Q0` contrast needed
calibrating before any scale wording could stand.

The audit re-parameterized the frozen E01 intervention by residual fraction
`r = ||dh||/||h||` on the frozen discovery examples, frozen site, frozen probe
recipe and frozen context construction. Nothing was retuned and the consumed
holdout was never loaded.

**The confound is real.** E01's own operating point is `r = 0.0196` in
Qwen3-0.6B and `r = 0.0547` in Qwen3-1.7B - the larger checkpoint was being
pushed 2.79x harder in residual-norm terms. The headline `0.0144 versus 0.7013`
ratio is therefore not a calibrated quantity and must be retired.

**The conclusion survives it.** At matched residual fraction the 1.7B `Q`
advantage is a stable ~9.5x at every grid point; matching instead on achieved
standardized semantic displacement makes it ~27x, because at matched `r` the
smaller model actually receives the *larger* coordinate push. Ordering is
preserved under both rulers.

On-manifold validity restricts the trustworthy grid to `r <= 0.10` (k-NN
distance to the validation cloud grows at most 1.35x). Inside that region all
three estimands - `Q`, `A` and `G` - preserve the checkpoint ordering, and
E01B-3's confirmed positive `G` for 1.7B is reproduced. The `G` sign reversal
seen at `r >= 0.20` is an off-manifold artifact, not a model property.

Required wording changes, now in force across this program:

```text
retire the single-point ratio; report the calibrated curve
the magnitude of the contrast is ruler-dependent; the direction is not
"scale" stays an interpretation - two checkpoints are not a randomized manipulation
restrict every Q/A/G claim to residual fraction <= 0.10
```

A secondary result worth keeping: within-model reliance on structured orthogonal
state versus the scalar coordinate differs sharply and robustly - `A/Q` is 6.10
in 0.6B against 1.32 in 1.7B.

See `E01_CALIBRATION_AUDIT_SUMMARY.md`.

### F2. E18 causal-read localisation - complete

E15's Gate 1 showed its carrier was not causally sufficient, and the direction
review's section 10.2 requires that a weak full patch trigger redesign rather
than post-hoc analysis. E18 answered the prerequisite first: for a delayed
decision that provably depends on a remembered binary state, **where** does a
full-state counterfactual replacement actually change the decision?

A declared 6-site by 8-layer grid, every cell reported, on a fresh corpus
namespace. The map is sharp:

```text
state_word_last   STRONG at L0, L4, L8   ->  PARTIAL at L12  ->  WEAK from L17
decision          WEAK to L12            ->  PARTIAL at L17  ->  STRONG at L21-27
carrier (E15)     WEAK at all eight layers
```

Three results follow.

**The causal content of the whole prefix sits in one token.** At L0-L8 patching
the single state word (flip 0.603) equals patching the entire 20-token clearance
line (0.577) and the entire 69-token prefix (0.580). Sixty-eight extra tokens add
nothing.

**There is a hand-off between L12 and L17.** The source token stops being causal
exactly where the decision token starts being causal - the state is copied
forward out of its source position mid-network. This explains E15's null
completely: E15 intervened at a prefix position at L17, the one region where
nothing upstream is causal any more. It was one token late (index 49 versus 48)
and about ten layers late.

**The D/C dissociation is now a map.** 33 of 48 cells decode the state at
`D >= 0.95` and **21 of those are causally WEAK**. The sharpest case is E15's own
carrier: `D = 1.000` at all eight layers with a flip rate never above 0.097.
Decodability is near-universal across positions and depths; causal efficacy
occupies a narrow band. A probe finds the variable almost anywhere it has been
copied to; the model only uses it in one place at one depth.

The carrier survives horizon: at `k = 8` the source token is still STRONG at
L0/L4/L8 (0.567/0.563/0.533).

See `E18_CAUSAL_READ_LOCALISATION_SUMMARY.md`.

### F3. E19 temporal causal organization - complete

The flagship, run on the carrier E18 validated. Every cell passes the
carrier-sufficiency gate E15 lacked (flip 0.60-0.69 against E15's 0.010), so
every decomposition below decomposes an effect that is actually present.

**The E15 signature is real in this task.** Decodability stays at ceiling across
the horizon while causal organization decays materially:

```text
D  = 1.000  1.000  1.000  1.000     (k = 1, 2, 4, 8)
Q  = 1.000  0.994  0.909  0.713     relative to k0
A  = 1.000  0.991  0.852  0.729
G  = 1.000  0.957  0.645  0.551
```

Every decline exceeds its SESOI with Holm-corrected `p` below 0.05.

**The two-locus design separates why each pathway decays.** Intervening at the
decision token holds the propagation path fixed while state age grows; at the
source token both grow. The components come apart:

```text
Q decays with propagation DISTANCE, not with state age  (S_rel 0.705, D_rel 1.000)
A decays with state AGE                                 (S_rel 0.744, D_rel 0.658)
```

**Code rotation is not pathway loss.** At the decision token the decoded axis
rotates hard (`cos(u_0, u_k)` 1.000 -> 0.524) while `Q` measured on the frozen
`k0` axis stays flat at 1.000. The decodable direction rotates away from the
causally effective one, which does not move - a direct caution against reading a
probe axis as the model's endogenous causal code.

Not everything held. The components do not decay at materially different *rates*
at the source locus (differences 0.16-0.22, below the 0.25 SESOI), so the strong
"reorganization rather than decay" claim is unsupported there. All half-lives are
right-censored: nothing halves by `k=8`. And a magnitude control added after the
run invalidated one of four curves, whose native setpoint had shrunk to 0.452 of
its baseline residual fraction.

See `E19_TEMPORAL_CAUSAL_ORGANIZATION_SUMMARY.md`.

### F4. E20 long-horizon extension - complete, null on its primary objective

E19 left every half-life right-censored, so E20 tried to extend the horizon far
enough for a persistence timescale to become estimable. It changed exactly one
thing - how far the horizon reaches - and inherited the carrier, loci, components,
estimands, arms and inference unchanged.

**It did not work, and the stop rule was obeyed.** A non-causal Phase 1 measured
both distractor pools across `k in {1,2,4,8,16,24,32}` on behaviour and
decodability only, with every intervention forbidden. Both pools reach exactly
`[1,2,4,8]` at `B >= 0.70` - the grid E19 already had - so E20 ran no
interventions and **every half-life remains right-censored at k=8**. No
persistence timescale is quoted and the frozen half-life rule was not relaxed to
manufacture one.

Two things were learned anyway, both non-causal.

**Step count, not token distance, drives the behavioural collapse.** At matched
`k=16` the two pools differ by 50% in tokens (251 versus 377) and by 0.010 in
`B`; at similar token distance, 8 long steps (206 tokens, `B`=0.737) beats 16
short steps (251 tokens, `B`=0.540) by 0.197. This settles what E15's Stage 3b
only hinted at with a confounded contrast, and explains why the long pool bought
nothing: it lengthens each step without reducing how many there are.

**The decodability-versus-behaviour dissociation widens.** `D` at the source
carrier is `1.0000` at every horizon in both pools out to `k=32` and 719 tokens,
where forced-choice behaviour sits at exactly chance (0.500).

E20 also promotes E19's post-hoc magnitude control to the preregistered gate G4,
which excludes a magnitude-unstable curve from every hypothesis and from any
half-life. It is implemented and unit-tested but, since Phase 2 never ran, has
not yet fired on real data.

See `E20_LONG_HORIZON_SUMMARY.md`.

## Confirmation policy

The E01 confirmation split was consumed exactly once after remote
preregistration; the result was strong confirmation. It must not be reopened or
used for core-mechanism tuning.

Application branches must not consume that holdout. Each new branch should have its own discovery/confirmation partition where appropriate.

## Branch priority

Current recommended ordering:

| Rank | Experiment | Why |
|---:|---|---|
| 1 | E01B-1 | complete: establishes donor-free causal object |
| 2 | E01B-2 | complete: establishes contextual sensitivity in discovery |
| 3 | E01B-3 | strong confirmation complete; mechanism frozen |
| 4 | E14 | strong confirmation complete; mixed actionability/general degradation |
| 5 | E13 | diagnostic claim CONFIRMED (strong); holdout consumed; method branch stays closed |
| 5b | E17 | cross-family replication COMPLETE; phenomenon replicated in OLMo-2 |
| 6 | E16 | deepest developmental claim if checkpoints permit; not authorized |
| 7 | E15 | discovery complete; unresolved null — no causal handle at the predeclared carrier |
| 8 | E01AUDIT | complete: the checkpoint contrast survives residual-fraction and semantic-shift calibration |
| 9 | E18 | complete: causal read localised to one token at L0-L8; a usable carrier exists |
| 10 | E19 | complete: D at ceiling while Q/A/G decay; Q decays with distance, A with age |
| 11 | E20 | complete: null - the horizon is not extendable past 8 steps, so no half-life is estimable |

## Paper decision points

### Paper path 1 — core mechanism paper

If E01B succeeds strongly and confirmation replicates:

```text
D ≠ C across checkpoints
causal setpoint sufficiency/context
readout-conversion localization
```

This can stand without every extension.

### Paper path 2 — compression/application paper

If E14 shows D robust but C fragile:

> Representational integrity is not functional integrity under quantization.

### Paper path 3 — transfer/method paper

If E13 shows a representation/utilization transfer gap:

> Distillation can transfer what is represented and how it is used at different rates.

E13 supports this diagnostic claim and it is the part worth writing. The method
half did not survive. Conversion-response distillation does **not** repair the
gap: its frozen bounded revision gate selected no candidate from six objective
variants, and mechanism controls showed the apparent COD benefit is generic
local-sensitivity regularization — shuffling teacher targets within relation
family, destroying all sample-level semantic correspondence, gave the lowest COD
in the campaign. A semantic-specific signal exists only in the smallest profile
component (Q), which the frozen selection statistic cannot resolve. There is no
separate method contribution to claim.

If the method branch is ever resumed it needs two changes, both requiring a new
frozen discovery design: a response coefficient calibrated per objective against
the KD gradient scale rather than fixed at 1.0 (the response gradient currently
runs 3-5x the KD gradient), and a selection statistic sensitive to per-example
profile correspondence rather than a magnitude-dominated norm.

### Paper path 4 — developmental paper

If E16 shows `t_D < t_C`:

> Models learn a semantic variable before learning to deploy it functionally.

### Paper path 5 — long-horizon paper

If E15 shows `H_C < H_D`:

> Information can remain internally represented after its ability to govern future action has decayed.

E15 did not reach this path. The temporal question was not measurable at the
predeclared carrier because there was no baseline causal effect to decay. The
prerequisite was a separately frozen carrier-localisation design establishing
*where* a delayed decision causally reads a remembered state.

**E18 has now met that prerequisite.** A usable single-token carrier exists -
`state_word_last` at L4-L8, STRONG at both `k=1` and `k=8` - so the flagship is
buildable without a transplant bottleneck. Two constraints from the E18 map must
be built into it:

```text
the causal locus MOVES between L12 and L17, so a single fixed layer would
conflate "the state stopped mattering" with "the state moved"

site and depth must be declared jointly; the carrier is a narrow
(position, depth) region, not a layer
```

The temporal design must therefore separate native-local organization at each
horizon from transported-reference organization off a frozen early axis, and gate
on full-state-patch sufficiency at every horizon before interpreting any `Q/A/G`
decomposition.

## Stop rules

Do not keep expanding breadth if a branch fails its mechanism gate.

Do not collect many weak applications. Prefer one clean transformation that shows a large D/C dissociation.

Do not claim a universal theory from one variable, model family, or task.

Do not open confirmation merely because discovery results are exciting. Freeze the hypothesis, intervention, metrics, and mechanism first.
