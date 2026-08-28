# Literature Positioning — Phase 0A.2

Update (2026-08-27): E01A full discovery is now complete. Statements below
describing E01 as pending are retained as the historical Phase 0A.2 boundary;
see `../E01A_FULL_DISCOVERY_SUMMARY.md` for the causal-intervention result.

Purpose: state what the cited recent work already establishes, where our
current E00/E00-B result overlaps, what would be genuinely additional, and
which claims we must not make. Sources were retrieved from arXiv on
2026-08-27 (abstracts verified).

## Papers reviewed

### 1. arXiv:2608.17843 — *Encoded but Not Actionable: Auditing the
Decode–Generate–Steer Gap in Frozen LLMs for Geometric Constraints*
(Liang, Cheng & Wajid, submitted 2026-08-18)

Establishes:
- A controlled geometric-constraint testbed separating local pairwise
  relations from higher-level constraint status.
- **Random-initialization finding (directly relevant)**: "sketch-level DOF
  status is already highly decodable from randomly initialized representations
  and improves only modestly with pretraining" — i.e. much of that probe
  signal is available without learned weights.
- Pretraining substantially improves decoding of local pairwise relations even
  after shuffled-order controls.
- Decode ≠ generate ≠ activation-influence ≠ steerability: generation often
  fails to express decodable info; activation-restoration effects vanish at
  patched positions while decodability persists across depth; mean-difference
  steering does not reliably control outputs.

### 2. arXiv:2608.07528 — *The Knowing–Saying Gap: When Probes See Errors that
Confidence Misses* (Goel, Bandyopadhyay & Shenk, submitted 2026-07-21)

Establishes:
- Probes detecting corrupted context with near-perfect accuracy can be
  uninformative about final-answer correctness.
- Probe persistence across hops does not separate correct from incorrect
  outcomes; structured-confidence formats collapse confidence granularity.
- Probe-based interventions are sharply model/error-type dependent; no single
  intervention dominates.
- Generalises across model families including reasoning models.

### 3. arXiv:2608.21766 — *Evaluation Awareness in Language Models:
Representation, Verbalization, and Control*
(Heidari, Memarian & Rabusseau, submitted 2026-08-22)

Establishes:
- "Being under evaluation" is linearly decodable (AUROC ≥ 0.7) from residual
  streams of every tested model.
- Representation↔verbalization alignment is partial and varies by model,
  layer, and readout choice; probe-derived steering shifts verbalization.
- Olmo checkpoint trajectory: evaluation awareness present in base models,
  amplified by SFT, stable afterwards; steering effects grow at every stage.

## Where our current E00/E00-B result overlaps

1. **D > behavior**: our corrected D≈1 vs forced-choice B≈0.598 is another
   instance of the decode-vs-generation/behavior divergence these papers and
   the older literature document. By itself this contributes nothing novel
   (#19 rule of the task).
2. **Probe-detected ≠ reliable final outcome**: our planned frozen-probe-on-
   native-errors analysis overlaps conceptually with the knowing-saying gap of
   paper #2.
3. **Probe→readout alignment variation**: our D-vs-fixed-readout geometry
   relates to paper #3's representation↔verbalization misalignment, but their
   target variable ("under evaluation") differs from truth-of-composition.

## What would be genuinely additional here

None of the three papers performs the combination we set up below:

a. **Matched random-initialization control under an exact same-tokenizer,
   same-prompt-interface, same-split identity-validated pipeline** — paper #1
   includes a random-init comparison for its constraint-status variable; our
   addition must therefore be framed as replicating+extending that *control
   methodology* onto a semantic-composition latent with **matched
   counterfactual twins** (which also neutralize surface leakage), plus a
   token-embedding rung between text and contextual features.
b. **Leave-one-relation-family-out probing** as an abstraction measure of the
   SAME latent used for both D and B, tying family-generalization to the
   behavioral readout measured on identical examples.
c. **Layer-wise fixed native readout diagnostic** (`final_norm`+`lm_head`
   applied per layer, validated against native logits) compared against tuned
   external decoding on one axis, with explicit probe-direction/native-readout
   cosine geometry. The cited papers audit generate/steer/influence, not the
   geometry of the native Yes-No head versus a held-out decoder for the same
   latent.
d. **Interface sensitivity under Qwen's official non-thinking chat template**
   as a matched-pair counterfactual interface (same semantic examples), which
   none of the three covers.

## Claims we must not make

- That "decodable-but-not-expressed/used" is itself a discovery (known).
- That our D implies causal accessibility (E01 pending; see audit + gates).
- That B≈0.598 indicates "the model doesn't know" (threshold/interface
  calibration first; e.g., Family heterogeneity could be dominated by
  verbalizer/token effects).
- Any claim of novelty resting solely on the existence of a D–B gap.
- Any statement about post-training effects without the matched interface arms
  required by §16 of the task (never compare base vs instruct under
  mismatched interfaces and call it scale or training effect).

If Gate A fails (random init explains D), we position the outcome as a
negative-control replication of paper #1's random-feature warning within a
composition-latent design — still publishable methodology, zero mechanism
claims.
