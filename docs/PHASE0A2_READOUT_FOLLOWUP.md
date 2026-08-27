# Phase 0A.2 Readout Follow-up

This follow-up closes two narrow loose ends from `DIAGNOSIS_PHASE_0A2.md`
before E01 is authorized.

## 1. Calibrate the 0.6B chat arm

The existing 0.6B non-thinking chat result has threshold-zero balanced
accuracy `0.500` but margin AUROC `0.8142`. That means the ranking signal is
substantially better than the default decision threshold suggests.

The follow-up therefore selects a single Yes/No margin threshold on the
**validation split only**, freezes it, and reports discovery-test accuracy and
balanced accuracy. The same calculation is also repeated for the raw-completion
arm for a matched table.

No discovery-test label is used for threshold selection.

## 2. Replace the raw-residual cosine with an exact normalized-space comparison

The previous report included a cosine between:

- the probe direction expressed in raw residual coordinates; and
- `gamma * (w_yes - w_no)` from an RMSNorm + LM-head readout.

That cosine is descriptive but is **not an exact characterization of
cross-example native-readout ranking**. RMSNorm divides every sample by its own
positive RMS value. A positive sample-specific denominator preserves each
sample's sign, but it can change the ordering of scores across different
samples.

The new diagnostic therefore:

1. applies the model's **actual final normalization** to the cached residual
   vectors;
2. trains the same held-out linear truth probe in that final-normalized
   coordinate space;
3. compares its signed direction with the exact LM-head
   `w_yes - w_no` direction in that same coordinate space;
4. reconstructs selected-token logits from normalized states and verifies them
   against the existing exact `final_norm + LM head` path.

The new signed normalized-space cosine supersedes the old raw-residual
`abs(cosine)` for geometry claims. The old fixed-readout AUROC values remain
valid because those were already computed using the exact final norm and LM
head.

## Scope

This is a **non-causal diagnostic follow-up**. It does not implement E01,
activation patching, steering, SAEs, ReFT, KV-cache interventions, or PGB-CT.

If the 0.6B gap remains material after chat calibration and the normalized-space
geometry still shows a large external-probe/native-readout discrepancy, the
next scientifically useful step is a targeted E01 causal-use experiment.
