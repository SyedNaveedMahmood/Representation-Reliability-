# E18 Causal-Read Localisation — Results

Status: **complete.** Protocol `docs/E18_CAUSAL_READ_LOCALISATION_PROTOCOL.md`,
frozen on disk before any measurement code was written, committed and pushed at
`75e2de3`. Campaign `runs/E18_LOCALISATION/`. Model `Qwen/Qwen3-1.7B`, bf16.
Fresh corpus namespace (`e18`, seeds 20261801/2/3), 2,800 samples, 150
discovery-test pairs per horizon.

Open discovery. No confirmation split; no consumed holdout touched.

**Provenance note.** The protocol was frozen before the runner, tests or pilot
existed, and nothing in it changed after any result was seen. It was committed
and pushed while the full sweep was already running rather than strictly before
it — the ordering that matters scientifically holds, the git ordering is
imperfect, and it is recorded here rather than glossed.

## Headline

The state is causally read at **one token, in an early band of layers**, and then
handed off to the decision position:

```text
state_word_last   STRONG at L0, L4, L8   ->  PARTIAL at L12  ->  WEAK from L17
decision          WEAK to L12            ->  PARTIAL at L17  ->  STRONG at L21-27
```

E15's carrier is **WEAK at every one of the eight layers**. It sits at token index
49; the causal site sits at index 48. One token earlier, at the same layer, the
flip rate goes from 0.010 to 0.603.

Outcome: `single_token_carrier_exists`.

## The map — flip rate under full-state counterfactual patch, k=1

150 pairs, pair-cluster bootstrap. Frozen grade scale: STRONG at flip >= 0.50,
PARTIAL at >= 0.10, both also requiring the effect CI to exclude zero and the
paired contrast against a same-norm random patch to exclude zero.

| site | L0 | L4 | L8 | L12 | L17 | L21 | L24 | L27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_word_last` | **0.603** | **0.600** | **0.603** | *0.267* | 0.010 | 0.010 | 0.017 | 0.000 |
| `carrier` (E15) | 0.010 | 0.013 | 0.010 | 0.097 | 0.013 | 0.000 | 0.010 | 0.000 |
| `clearance_line_span` | **0.577** | **0.573** | **0.577** | **0.547** | 0.017 | 0.010 | 0.007 | 0.000 |
| `prefix_span` (anchor) | **0.580** | **0.567** | **0.603** | **0.580** | 0.023 | 0.020 | 0.010 | 0.000 |
| `request_step_last` | 0.007 | 0.013 | 0.007 | 0.007 | 0.010 | 0.003 | 0.003 | 0.000 |
| `decision` | 0.010 | 0.017 | 0.007 | 0.033 | *0.497* | **0.590** | **0.600** | **0.587** |

Bold = STRONG, italic = PARTIAL.

### Three things this settles

**1. The causal content of the whole prefix is carried by one token.** At L0-L8,
patching the single state word (0.603) is indistinguishable from patching the
entire 20-token clearance line (0.577) or the entire 69-token prefix (0.580).
Sixty-eight extra tokens add nothing. The read is localised, not distributed.

**2. There is a hand-off between L12 and L17.** The source token stops being
causal exactly where the decision token starts being causal. The state is copied
forward out of the source position during the middle of the network.

**3. E15's null is fully explained.** E15 intervened at a prefix position at L17
— the one region of this map where nothing upstream is causal any more. It was
one token late and about ten layers late.

## Effect sizes and controls at the key cells

| site | layer | effect | 95% CI | patch − random | beats random | ‖Δh‖/‖h‖ | grade |
|---|---:|---:|---|---:|---|---:|---|
| `state_word_last` | 0 | 1.241 | [1.143, 1.336] | 1.209 | yes | 1.051 | STRONG |
| `state_word_last` | 4 | 1.214 | [1.110, 1.314] | 1.264 | yes | 0.792 | STRONG |
| `state_word_last` | 8 | 1.228 | [1.138, 1.325] | 1.259 | yes | 0.791 | STRONG |
| `state_word_last` | 12 | 0.666 | [0.620, 0.712] | 0.627 | yes | 0.743 | PARTIAL |
| `state_word_last` | 17 | 0.019 | [0.009, 0.030] | 0.007 | no | 0.664 | WEAK |
| `carrier` | 12 | 0.338 | [0.300, 0.377] | 0.336 | yes | 0.427 | WEAK |
| `carrier` | 17 | 0.008 | [-0.002, 0.019] | 0.008 | no | 0.402 | WEAK |
| `decision` | 17 | 0.968 | [0.865, 1.065] | 0.973 | yes | 0.072 | PARTIAL |
| `decision` | 24 | 1.304 | [1.201, 1.411] | 1.309 | yes | 0.095 | STRONG |

The magnitudes are informative in their own right. At `state_word_last` the
counterfactual displacement is large (0.79-1.05 of the residual norm) because the
two state words genuinely differ there. At `carrier` L0 it is only 0.084 — the
line-final `.` token barely differs between twins — yet a probe still reads the
state from it at AUROC 1.000. At `carrier` L17 the displacement is a large 0.402
and the flip rate is still 0.013, so the weakness is not an under-powered edit.

`carrier` at L12 is the one cell where E15's site shows a real but sub-threshold
effect (0.338, CI excludes zero, beats random, flip 0.097). The carrier is not
causally inert; it is roughly six times too weak to qualify, which is consistent
with Gate 1's G1b passing while G1a failed.

## The dissociation, now as a map

Decodability at the same 48 cells (probe on `train`, `C` on `validation`,
evaluated on `discovery_test`):

| site | L0 | L4 | L8 | L12 | L17 | L21 | L24 | L27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_word_last` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `carrier` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `clearance_line_span` | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| `request_step_last` | 0.868 | 0.857 | 0.884 | 0.979 | 1.000 | 1.000 | 1.000 | 1.000 |
| `decision` | 0.857 | 0.866 | 0.854 | 0.888 | 1.000 | 0.999 | 0.999 | 0.998 |
| `prefix_span` | 0.863 | 0.853 | 0.853 | 0.854 | 0.852 | 0.849 | 0.847 | 0.846 |

**33 of 48 cells have `D >= 0.95`. Twenty-one of those are causally WEAK.**

The sharpest single case: `carrier` decodes the state at **AUROC 1.000 at all
eight layers** while its flip rate never exceeds 0.097 at any of them. So does
`state_word_last` at L17-L27 — perfectly decodable, causally spent. And
`request_step_last` reaches `D = 1.000` from L17 while being causally WEAK
everywhere.

This is the repository's core thesis rendered as a two-dimensional map:
**decodability is near-universal across positions and depths; causal efficacy
occupies a narrow band.** A probe finds the variable almost anywhere it has been
copied to; the model only *uses* it in one place at one depth.

`prefix_span`'s lower `D` (~0.85) is an artifact of mean-pooling 69 tokens for the
probe representation and should not be read as low decodability.

## Secondary horizon, k = 8

The frozen conditional pass re-measured every cell graded STRONG or PARTIAL at
k=1:

| site | L0 | L4 | L8 | L12 | L17 | L21 | L24 | L27 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `state_word_last` | **0.567** | **0.563** | **0.533** | *0.260* | — | — | — | — |
| `clearance_line_span` | **0.553** | **0.527** | *0.457* | *0.437* | — | — | — | — |
| `prefix_span` | **0.547** | **0.513** | **0.510** | *0.473* | — | — | — | — |
| `decision` | — | — | — | — | *0.393* | **0.550** | **0.557** | **0.527** |

The structure survives eight intervening distractor steps with only mild
attenuation (0.603 -> 0.567 at the source token). The carrier is a usable one at
horizon as well as at k=1.

## Integrity

* `no_op` maximum margin deviation: **exactly 0.0** at every layer;
* norm-matched controls match the patch norm to `5.6e-16` relative;
* zero residual hooks left registered;
* **anchor gate G2 passed**: `prefix_span` is STRONG at four layers, so the
  measurement is valid and the map is interpretable;
* every one of the 48 cells is reported, pass or fail;
* fresh corpus namespace, so the map is not conditioned on rows E15 already
  inspected; `confirmation_accessed: false`.

## Limitations

1. **L0 is close to trivial.** `resid_post` at layer 0 is near the token
   embedding, so patching the state word there is nearly editing the input text.
   The non-trivial content is the *depth profile*: STRONG at L4 and L8 is several
   blocks of computation past the embedding, and the fall-off at L12-L17 is what
   localises the hand-off.
2. **The flip ceiling is ~0.60, not 1.0**, partly because roughly a fifth of
   episodes are answered incorrectly before any intervention.
3. Single model, single task, single environment. This is a map of Qwen3-1.7B on
   this console task, not a general claim about transformer memory.
4. The `decision` site being STRONG late is close to a readout result rather than
   a memory result, and should not be treated as a carrier for a temporal study.

## What this licenses next

E18 answers the prerequisite that E15's Gate 1 raised. Against the protocol's
declared outcome table this is the first row: **a usable single-token carrier
exists** — `state_word_last` at layers 0-8, STRONG at both k=1 and k=8.

The flagship temporal study is therefore buildable *without* a transplant
bottleneck, on that carrier. Two constraints follow directly from this map and
must be built into its design:

* **The hand-off is the confound.** Because the causal locus moves from the source
  token to the decision token between L12 and L17, a temporal design must
  separate the review's two estimands explicitly: native-local organization at
  each horizon versus transported-reference organization from a frozen early
  axis. Measuring at one fixed layer would conflate "the state stopped mattering"
  with "the state moved".
* **Site and depth must be jointly declared.** A single frozen layer is not
  sufficient in this task; the carrier is a (position, depth) region, and the
  region is narrow.

The immediate next step is a newly frozen temporal design on the
`state_word_last` carrier at L4-L8, gated on its own full-state-patch
sufficiency check at every horizon before any `Q/A/G` decomposition is
interpreted. E15 itself stays closed and its verdict unchanged.
