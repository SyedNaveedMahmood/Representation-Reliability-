# E18 Causal-Read Localisation — Execution Protocol

Status: **frozen before any model is loaded and before any E18 causal quantity is
computed.** Committed and pushed before execution.

E18 is **open discovery** and a **prerequisite study**, not a rescue of E15. It
creates no confirmation split, touches no consumed holdout (E01, E13, E14), and
makes no confirmatory claim.

## 1. Why this experiment exists

E15's Gate 1 addendum established that the predeclared E15 carrier — the final
token of the state-writing step, at `resid_post` L17 — is **not causally
sufficient**. A full counterfactual replacement of that state (0.402 of the
residual norm) flipped the delayed decision in 1% of episodes with a mean-effect
CI including zero.

`docs/Reproduction_Reliability_Next_Direction_Review.md` section 10.2 requires a
carrier to pass a full-state-patch sufficiency gate before any `Q/A/G`
interpretation, and section 10.10 says a weak full patch means the carrier and
task should be **redesigned rather than analyzed post hoc**. Before redesigning,
one prerequisite question must be answered:

> For a delayed decision that provably depends on a remembered binary state,
> **at which token sites and layers does a full-state counterfactual replacement
> actually change that decision?**

E18 answers exactly that and produces a **map**, not a hypothesis test about
temporal decay. Its output determines whether the flagship temporal experiment is
buildable on a single-token carrier at all, or whether a transplant bottleneck is
required.

## 2. What E18 is not

* It does not re-open, re-run, or re-interpret E15. E15's verdict
  (`unresolved_no_causal_handle_at_predeclared_carrier`) stands unchanged.
* It does not test H15.1-H15.4.
* It does not search for a site that makes any previous result come out better.
  Every site and layer below is declared here, in advance, and **all cells are
  reported** whether they pass or fail.
* It uses a **fresh corpus namespace** with fresh seeds so it is not conditioned
  on rows already inspected by E15 Stage 3 or the Gate 1 addendum.

## 3. Task and corpus

The frozen E15 stateful console environment, generator unchanged, fresh namespace
`e18-{split}-v1`:

```text
train:            800 directed /  400 pairs   seed 20261801
validation:       300 directed /  150 pairs   seed 20261802
discovery_test:   300 directed /  150 pairs   seed 20261803
```

Rendered at horizons `k in {1, 8}`. Matched counterfactual twins differ in
exactly one clearance word; `GRANTED`/`DENIED` are both two Qwen3 tokens, so twin
prompts have identical token length and identical positions throughout.

All causal evaluation is on `discovery_test` only. Inference clusters on
`pair_id`.

## 4. Model

`Qwen/Qwen3-1.7B`, bf16 — the same checkpoint as E15, so the map is directly
comparable to the Gate 1 result.

## 5. Predeclared sites

Six sites, all resolved from character spans computed deterministically from the
prompt's fixed line structure and validated against stored sample metadata:

```text
state_word_last        last token of the clearance VALUE word in the target line
                       (the "GRANTED"/"DENIED" token itself)
carrier                last token of the target clearance line
                       (the E15 carrier; included as the known-failing reference)
clearance_line_span    every token of the target clearance line (multi-token)
request_step_last      last token of the request step line
decision               the final prompt token (the "Answer:" colon)
prefix_span            every token from position 0 through the end of the
                       clearance block (multi-token upper-bound anchor)
```

`prefix_span` is the sanity anchor: if a full counterfactual replacement of the
entire pre-distractor prefix does not move the decision, the measurement itself
is suspect and the map must not be interpreted.

The header contains the literal word `GRANTED`, so the state-word span is located
**within the target line's character range**, never by a bare prompt search.

## 6. Predeclared layers

```text
0, 4, 8, 12, 17, 21, 24, 27
```

Eight `resid_post` layers spanning the depth of the 28-block model. L17 is
included because it is the repository-frozen Qwen3 site. This is a **declared
sweep whose every cell is reported**, not a search for a best layer.

## 7. Arms

At each (site, layer) cell, on the same discovery-test episodes:

```text
no_op                zero delta                          numerical contract
full_state_patch     h_twin(site, layer) - h_base(...)   the measurement
random_norm_matched  same-norm random direction, 3 seeds specificity control
```

For multi-token sites the patch and its controls are applied at **every token of
the span**, with the control's per-token norm matched to the patch's per-token
norm. `no_op` is run once per layer; it does not depend on the site.

Direction seed base `20261820`, disjoint from every E15 and Gate 1 seed block.

## 8. Primary measurement and the frozen strength scale

For each cell, the counterfactual-oriented decision change, exactly as in E15:

```text
effect = m_toward(m_after, 1-y) - m_toward(m_before, 1-y)
flip   = prediction moved to the counterfactual label
```

Pair-cluster bootstrap on `pair_id`, 2000 draws, 95%.

A cell is graded on the **flip rate** of the full-state patch, because the review
requires the patch to change the decision *strongly*, and a margin shift that
never crosses the decision boundary is not that:

```text
STRONG    flip rate >= 0.50   and effect CI excludes zero
                              and paired contrast vs same-norm random excludes zero
PARTIAL   flip rate in [0.10, 0.50)  with the same two CI conditions
WEAK      anything else
```

These thresholds are frozen here, before execution. E15's failing carrier scored
a flip rate of 0.010 under the identical measurement, which this scale grades
WEAK.

## 9. Secondary: decodability at the same cells

For every (site, layer) cell, a logistic probe for `z` is fitted on `train`, its
`C` selected on `validation` only, and evaluated on `discovery_test` — the frozen
repository recipe. Multi-token spans are represented by their mean-pooled state.

This is descriptive and secondary. Its purpose is the positional
decodability-versus-causality contrast: E15 already found one site with
`D = 1.000` and no causal effect, and E18 measures how general that is.

## 10. Horizon

The map is computed at `k0 = 1`. A cell graded STRONG or PARTIAL at `k = 1` is
re-measured at `k = 8`; cells graded WEAK are not, because a site that cannot
carry the state one step will not carry it eight. This conditional second pass is
declared here in advance.

## 11. Integrity gates

* **G0** — corpus: label oracle exact, twins differ in exactly one clearance
  word, token-length parity, every declared span unique and correctly located,
  no duplicate prompts or identities, pairs never split.
* **G1** — numerics: `no_op` maximum selected-logit deviation `<= 1e-6` at every
  layer; zero residual hooks registered after every batch; norm-matched controls
  match the patch norm to `<= 1e-6` relative; exact row identity through every
  batch.
* **G2** — measurement validity: `prefix_span` must be graded STRONG at **at
  least one** layer. If it is not, the whole map is reported as uninterpretable
  and no site conclusion is drawn.

A failed gate is recorded and stops the affected stage. It is never worked
around.

## 12. Outcomes and what each means

| Outcome | Reading | Consequence |
|---|---|---|
| A single-token site is STRONG at some layer | a usable single-token carrier exists | the flagship temporal design can use it, after its own frozen Gate 1 |
| Only `clearance_line_span` is STRONG | the read is distributed across the state clause | the flagship needs a multi-token carrier, not a single token |
| Only `prefix_span` is STRONG | no localised read; the state is re-read from raw context | a transplant bottleneck is required; a plain long prompt cannot work |
| Nothing is STRONG including `prefix_span` | the measurement is invalid | do not interpret; fix the measurement first |
| Many sites STRONG with high `D` everywhere | causal read is redundant/distributed | report as a redundancy finding; single-site interventions understate causal use |

Every one of these is a real result. None of them is a failure of the experiment.

## 13. Claim boundary

E18 supports only a task-, model- and site-specific statement about where a
delayed decision causally reads a remembered binary state in Qwen3-1.7B on this
environment. It is not a general claim about transformer memory, and it licenses
no temporal claim by itself.
