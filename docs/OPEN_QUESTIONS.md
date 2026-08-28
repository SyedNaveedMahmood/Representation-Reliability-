# Open Questions and Discovery Targets

Scores are qualitative: 1 low, 5 high.

## E01A discovery update (2026-08-27)

E00-C supports a learned, cross-family decodable truth signal and a fixed
readout mismatch for Qwen3-0.6B. E01A now shows that the frozen decoded
coordinate is causally actionable relative to random, orthogonal, and
same-label controls at both scales, with much stronger conversion in 1.7B.
The matched and shuffled opposite-label coordinate treatments are equivalent
across the full dose response. Post hoc discovery tracing localizes a mixed
bottleneck dominated by native-readout conversion: injected standardized
signal is similar at L17/L20, while 1.7B both converts it much more strongly at
every traced layer and retains more at L23/L27. Scalar target regressions find
no residual matched-source indicator in either model. This promotes two bounded
questions:

- Can source-free coordinate setpoints reproduce E01A conversion?
- Does matched or relation-family context matter when its orthogonal component
  is allowed to enter at a fixed scalar coordinate displacement?
- Which scale/post-training/interface change improves native readout alignment
  without materially changing representation strength?

The first is registered as proposed E01B and is not authorized. Confirmation
for E01A remains locked.

## Tier A — immediate

### Q1. Why are matched and shuffled coordinate sources equivalent?

At the frozen layer 17 `resid_post`/last-prompt site, first use source-free
class-median or standardized coordinate setpoints estimated from a
non-confirmation development split. Then hold scalar displacement fixed and
vary the orthogonal component among none, matched twin, same-family shuffled,
different-family shuffled, and norm-matched random context. Do not touch
confirmation data.

Interesting:
- source-free setpoints reproduce the coordinate-only response;
- matched identity matters only when orthogonal context can enter;
- context modulation or conversion differs across model scale.

**Novelty 5 | Importance 5 | Compute 5 | Cleanliness 5**

### Q2. Can a representation be causal but resistant to steering?
Compare replacement patches with additive diff-mean/probe-normal steering.

Interesting:
- high causal recovery but no safe alpha works;
- steering works only at norms causing collateral drift;
- low-rank/rotation later succeeds where rank-1 fails.

**5 | 5 | 5 | 5**

### Q3. Does the model internally detect an error that it later ignores?
Train identical-state probes for:
1. corruption-state label;
2. final-output-error label.

Interesting:
- corruption detection far stronger than output-error prediction;
- a later layer where gap closes;
- moving monitor score does not fix behavior.

**5 | 5 | 5 | 4**

### Q4. Are there causal commitment points in generated reasoning?
Use hidden-state velocity/curvature/change points, then patch high-score vs low/random steps.

Interesting:
- abrupt state transitions predict intervention sensitivity;
- pre/post commitment interventions have asymmetric recoverability.

**5 | 5 | 5 | 4**

### Q5. Are observer-visible components different from model-used components?
Compare probe-attributed heads/MLPs with PGB-CT or causal component sets.

Interesting:
- low overlap despite stable attribution within each method;
- decodable signal distributed while causal mechanism sparse.

**5 | 5 | 4 | 4**

## Tier B — robustness and control structure

### Q6. What shapes do steering dose-response curves have?
Classify monotonic, threshold, saturation, inverted-U, sign reversal, and context-dependent curves.

### Q7. Is representational invariance also causal invariance?
A probe may remain stable across paraphrases while patching/steering changes.

### Q8. Does a concept migrate across token positions?
Track D/C/S site under question reframing, distractors, templates, paraphrase.

### Q9. Are behavioral directions model-specific or transferable?
Test direct/cross-model aligned subspaces with Procrustes/CCA.

### Q10. Is causal meaning stable for SAE features?
Compare activation consistency to intervention-effect consistency.

## Tier C — emerging

### Q11. Are steering failures caused by off-manifold movement?
Relate side effects to activation norm, nearest-natural-state distance, local covariance/Mahalanobis distance, and later SAE reconstruction distance.

### Q12. Do corrected instructions leave causally active "cache scars"?
Give a claim/instruction, revoke it, then selectively intervene on K/V states.

### Q13. Are there narrow cache write windows for decisions?
Map layer × token cache interventions and look for sparse causal windows.

### Q14. Can internal monitors be spoofed while output behavior is preserved?
Move monitor score under an output-KL/accuracy constraint.

### Q15. What is the intrinsic rank of reliable behavioral control?
Test rank 1/2/4/8/16.

### Q16. Does representation reliability predict latent communication utility?
Later: test whether reliable states make better cross-model communication channels.

## Avoid as first projects

- generic better-probe papers;
- "best steering layer" only;
- SAE visualization without causal tests;
- pure KV compression;
- large RL/RLVR;
- 32B sweeps;
- one-model/one-dataset general claims;
- effects that vanish under norm-matched controls.
