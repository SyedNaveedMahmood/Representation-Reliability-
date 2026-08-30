# E19 Temporal Persistence and Reorganization of Causal Organization

Status: **frozen before any E19 measurement code exists and before any E19
quantity is computed.** Committed and pushed before execution.

Open discovery. No confirmation split is created or accessed; no consumed holdout
(E01, E13, E14) is touched. No confirmatory claim is made.

## 1. Question

From `docs/Reproduction_Reliability_Next_Direction_Review.md` section 9.1:

> When a semantic state must remain relevant across delayed computation, does its
> linearly available representation persist longer than the causal organization
> that connects it to future action?

and its stronger form:

> Do scalar, additive-context and interaction pathways have distinct temporal
> trajectories, including substitution or reorganization rather than uniform
> decay?

E19 measures, at each horizon `k`, the componentwise profile

```text
O(k) = ( Q(k), A(k), G(k) )     plus  D(k), P(k), B(k)
```

The target result is a **trajectory**, not a single half-life. A half-life is
reported only where the frozen smoothness rule of `metrics/temporal_half_life.py`
permits it; otherwise the curve stands on its own.

## 2. What licenses this design

E15 failed at Gate 1: its predeclared carrier was not causally sufficient. E18
then mapped where the causal read actually is, and found:

```text
state_word_last   STRONG at L0, L4, L8  ->  PARTIAL at L12  ->  WEAK from L17
decision          WEAK to L12           ->  PARTIAL at L17  ->  STRONG at L21-27
```

Two constraints follow, and both are built in below:

* the causal locus **moves** between L12 and L17, so a single fixed layer would
  conflate "the state stopped mattering" with "the state moved";
* site and depth must be declared **jointly** — the carrier is a narrow
  (position, depth) region, not a layer.

E15 stays closed. E19 does not re-open it, does not re-test H15.1-H15.4, and
inherits none of its claims.

## 3. Separating state age from remaining distance

Review section 10.1 requires separating

```text
a = age of the state          (distractor steps since it was written)
r = remaining distance        (from the intervention site to the decision)
```

E15 confounded them: it intervened at the source and grew the gap, so `a` and `r`
rose together. E18 shows the two causal loci in this task sit at opposite ends of
the trajectory, which makes a clean two-locus decomposition available:

```text
locus S   state_word_last @ L8    intervening at the SOURCE
                                  k grows  ->  a and r BOTH grow
locus D   decision @ L24          intervening at the DECISION token
                                  k grows  ->  a grows, r stays ~0
```

Any decay at **locus D** is therefore state-age / interference, because nothing
about the post-intervention path changes with `k`. Decay at **locus S** on top of
that is the additional cost of propagation distance. The contrast between the two
curves is the frozen decomposition.

`L8` is chosen for locus S as the deepest STRONG layer in E18's early band, so
the result cannot be dismissed as editing the input embedding; `L0` is excluded
for exactly that reason. `L24` is chosen for locus D as its highest-flip STRONG
layer. Both are taken from E18's map and neither is searched over here.

## 4. Task, corpus, model

The frozen stateful console environment, generator unchanged, fresh namespace
`e19-{split}-v1` so E19 is not conditioned on rows E15 or E18 inspected:

```text
train:            800 directed /  400 pairs   seed 20261901
validation:       400 directed /  200 pairs   seed 20261902
discovery_test:   300 directed /  150 pairs   seed 20261903
```

Rendered at horizons `k in {1, 2, 4, 8}` by prefix extension, so a base episode's
identity, carrier position and nuisance content are byte-identical across the
grid and only the state-to-decision distance varies.

Model: `Qwen/Qwen3-1.7B`, bf16 — the checkpoint E18 mapped.

The horizon grid stops at 8 because E15 measured forced-choice behaviour
collapsing to 0.560 at `k=16` and 0.500 at `k=32` in this environment; a decision
the model cannot make is not a decision whose causal organization can be read. A
behaviour gate re-checks this on E19's own corpus (section 8).

## 5. The two estimands, both predeclared

Review section 10.4 requires these to be declared separately and never mixed:

```text
O_native(k)   horizon-specific probe axis u_k, validation-only setpoints at k.
              Asks whether the semantic code available AT THAT HORIZON is
              actionable.                                          <- PRIMARY

O_ref(k)      the k0 axis u_0 held frozen and applied at later horizons.
              Asks whether the ORIGINAL coordinate stays functionally
              connected.                                           <- SECONDARY
```

A decline in `O_ref` with stable `O_native` indicates code rotation or pathway
migration rather than loss. A decline in both is stronger evidence of functional
decay. Geometric rotation is measured directly as `cos(u_0, u_k)` and subspace
alignment at each locus.

## 6. Factorial arms

At each (locus, horizon), on the same discovery-test episodes:

```text
no_op                  zero delta                          numerical contract
full_state_patch       h_twin(site) - h_base(site)         SUFFICIENCY GATE
Y10_scalar             source-free setpoint on u           }
Y01_context            matched probe-orthogonal context    } x {native, ref}
Y11_both               setpoint + context                  }
random_norm_matched    3 seeds, matched to the setpoint norm
orthogonal_random      3 seeds, orthogonal to u, matched to the context norm
```

Estimands, exactly the frozen E01B-3 algebra:

```text
Q(k) = Y10 - Y00
A(k) = Y01 - Y00
G(k) = (Y11 - Y10) - (Y01 - Y00)
```

`Y00` is the clean forward. Setpoint targets are opposite-class medians computed
on `validation` **at that horizon and locus** for the native estimand, and at
`k0` for the reference estimand. Context is the twin displacement projected
orthogonal to `u` and per-example norm-matched, as in E01B-2/E01B-3.

## 7. Propagation

For locus S, the intervened forward captures `resid_post` at layers
`{12, 17, 21, 27}` at the **decision token** — strictly after the edit layer,
since an edit at L8 cannot change L8 at another position:

```text
P_norm(k, l)  ||h_dec_intervened - h_dec_clean|| / ||h_dec_clean||
```

For locus D the edit is at L24 and propagation is captured at L27 only.

## 8. Gates

* **G0 corpus** — label oracle exact; twins differ in exactly one clearance word;
  token parity; every span unique; pairs never split; no duplicate prompts.
* **G1 numerics** — `no_op` maximum selected-logit deviation `<= 1e-6` at every
  layer; zero residual hooks after every batch; norm-matched controls within
  `1e-6` relative; setpoint fidelity inside the frozen bf16 tolerances; exact row
  identity.
* **G2 behaviour** — clean forced-choice accuracy `B(k) >= 0.70` at every horizon.
  If some horizons fail, the grid is truncated to its longest passing prefix and
  later horizons are reported as uninterpretable, never silently dropped. This is
  decided on non-causal quantities before any intervention runs.
* **G3 carrier sufficiency, per horizon and locus** — the full-state patch must
  reach E18's frozen PARTIAL bar (flip rate `>= 0.10`, effect CI excluding zero,
  and exceeding the same-norm random patch). **`Q/A/G` at a (locus, horizon) cell
  that fails G3 is recorded but not interpreted**, because a decomposition of an
  absent causal effect is meaningless. This is the gate E15 lacked.

## 9. Hypotheses

Declared before execution. `k*` is the largest horizon surviving G2.

**H19.1 — representational persistence.** `D_native(k*) - D_native(k0) > -0.05`
at both loci. Tested as non-inferiority; failure to reject a difference is not
evidence of persistence.

**H19.2 — causal-organization change under persistent representation.** At `k*`,
at least one of `Q`, `A`, `G` differs from its `k0` value by at least the SESOI,
with a paired episode-cluster CI excluding zero, Holm-corrected across the three
components.

**H19.3 — differential pathway persistence.** At least two of the normalized
component curves `Q_rel`, `A_rel`, `G_rel` differ from each other by at least the
SESOI at `k*`, tested as a curve contrast rather than by separate pointwise
significance.

**H19.4 — age versus remaining distance.** The locus-D curve isolates state age;
the locus-S curve adds propagation distance. Their difference at `k*` is the
frozen estimate of the propagation-distance contribution. Reported descriptively:
E19 does not claim a mechanism for any difference it finds.

### Frozen SESOI

```text
SESOI = 0.25 of the component's own |value at k0|,
        and the paired CI must exclude zero.
```

A component whose `k0` value is itself indistinguishable from zero has no
meaningful relative change and is reported as **not assessable** rather than
being given a ratio with a near-zero denominator.

## 10. Inference

* Episode-level cluster bootstrap, 2000 draws, 95%, resampling **whole horizon
  curves together** — a base episode is rendered at every horizon, so an episode
  is the cluster and its entire trajectory is resampled as a unit. Seed 20261930.
* Holm correction across the `Q`/`A`/`G` family for H19.2.
* Half-lives only via `metrics/temporal_half_life.half_life`, which refuses a
  summary for a non-monotone curve and returns right-censored rather than
  extrapolating.
* Every cell is reported, including those failing G3.

## 11. Outcomes and what each means

| Outcome | Reading |
|---|---|
| `D` stable, `Q/A/G` decay at different rates | componentwise causal persistence — the target result |
| `D` stable, `Q` falls while `A` or `G` rises | reorganization rather than decay — the stronger result |
| `D` and all components decay together | informative null against the dissociation in this task |
| `O_ref` falls while `O_native` holds | code rotation / pathway migration, not loss |
| Locus D flat, locus S decays | the cost is propagation distance, not state age |
| Locus D decays too | genuine state-age interference |
| G3 fails at longer horizons | the carrier stops being sufficient; report the horizon at which it does and stop there |

Every one of these is a real result.

## 12. Claim boundary

E19 supports only a task-, model-, site- and discovery-specific statement about
how the causal organization of one semantic variable evolves over horizon in
Qwen3-1.7B on this environment. It is not a decay law, not a memory theory, and
licenses no cross-model claim.
