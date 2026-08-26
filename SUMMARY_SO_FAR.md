# Summary So Far — Representation Reliability Harness

*Status date: 2026-08-26 · Phase 0A complete · E00 piloted through discovery scale*

---

## 1. Mission

Build a **modular causal-representation experimentation harness** to discover
open phenomena in representation reliability — the empirical gaps between:

- **D** — decodability (a held-out decoder separates a latent variable from hidden states);
- **C** — causal use (intervening on those states changes behavior toward a pre-declared counterfactual);
- **S** — steerability, **M** — monitorability, **R** — robustness, **K** — collateral safety.

The charter (`docs/RESEARCH_CHARTER.md`) defines these as logically distinct;
every claim in this document respects that ladder. Probe results establish
**decodability only**, never "the model uses X".

## 2. What exists now (Phase 0A delivered)

```
dataset generation → tokenization → model loading → activation extraction
→ activation caching (sharded/atomic/resumable) → linear probing → controls
→ layerwise metrics (+ bootstrap CIs) → reproducible run artifacts + figures + summary
```

Implemented under `src/representation_reliability/`:

| Area | Key guarantees |
|---|---|
| `config/` | Strict Pydantic v2 schema (`extra="forbid"` everywhere); merge base→model→experiment→CLI dot-path overrides; deterministic SHA-256 config hash; invalid sites/layers/dtypes/missing IDs rejected |
| `runtime/` | Deterministic run ID = hash(exp ID, config hash, seed, model revision, split hash); atomic `status.json` lifecycle; full manifest incl. git state, external repo SHAs, package versions, GPU info, per-stage seeds, wall time, peak VRAM |
| `adapters/hf.py` | Stable local API over HF; canonical sites `resid_pre/attn_out/mlp_out/resid_post` mapped to verified native module paths; load-time convention calibration |
| `data/` | Deterministic 5-family synthetic relational generator with **matched counterfactual twins** (twins share premise, question word, entities; only queried-entity order flips the label); group splits 60/15/15/10 with hard confirmation-isolation guards; invariant paraphrase and controlled-change query-swap transforms |
| `extraction/` | Explicit token selectors (`last_prompt`, `target_span_last`, `explicit`) storing strategy/resolved index/token id/text/char spans/seq length/chat-template flag; safetensors+parquet shard cache with sha256 integrity markers, atomic writes, work-unit-space boundaries, contiguous-prefix resume |
| `probes/` / `metrics/` | Logistic regression with train-only standardization and validation-only C selection; AUROC/AUPRC/balanced accuracy/majority/class balance; percentile bootstrap CIs |
| `runners/`, `reporting/`, CLI | Sharded resumable extraction; E00 orchestration incl. mandatory controls (majority, random-label ×3 seeds, TF-IDF surface baseline, optional random features); layer figures; descriptive-only `DISCOVERY_SUMMARY.md`; CLI `validate-config / e00 / summarize` |

Test suite: **53 tests passing**, including on-GPU contract tests: hook vs
`output_hidden_states` agreement (interior layers), cache round-trip exactness
(max abs dev 0.0), sample-ID alignment, corrupt-shard detection, and an NNsight
cross-check agreeing to **rel-dev 0.000000** at matched dtype/tokenization.

## 3. Runs executed

All runs use Qwen/Qwen3-0.6B, site resid_post. Every run carries
`config.resolved.yaml`, `manifest.json`, `status.json`, `samples.parquet`,
`activation_index.parquet`, `probe_metrics.{parquet,json}`,
`controls/random_label_metrics.parquet`, `controls/text_baseline_metrics.json`,
per-probe `.npz` coefficients, three D-vs-layer figures, and a descriptive-only
`DISCOVERY_SUMMARY.md`. Confirmation splits were never extracted, probed, or read.

| Run dir | Scale | Config | Wall | Peak VRAM alloc | Status |
|---|---|---|---|---|---|
| `runs/E00/E00_c554233556d2*` | smoke n=50 | layers {0,7,14,21,27}, last_prompt | 7.6 s | 1.31 GB | complete (+ resume re-test) |
| `runs/E00/E00_bf9efb94222b(-r2)` | pilot n=500 | all 28 layers, both selectors | 83 s | 1.32 GB | complete |
| `runs/E00/E00_18a016f37eb9` | discovery n=2000 (seed 20260827) | all layers, both selectors | 266 s | 1.32 GB | complete |
| `runs/E00/E00_50519d0b487d` | discovery n=2000 (data_seed 88112255) | same | 264 s | ~1.3 GB | complete |

### Headline numbers (diagnostic only)

| Run | Best probe AUROC (selector/layer) | Random-label control | Text baseline |
|---|---|---|---|
| Pilot (n=500) | 0.546 (target_span_last L16) | 0.52 ± 0.07 | 0.609 |
| Discovery #1 (n=2000) | 0.534 (last_prompt L1) | 0.503 ± 0.029 | **0.454** |
| Discovery #2 (n=2000, new seed) | 0.539 (target_span_last L13) | 0.498 ± 0.032 | 0.516 |

Best-layer CIs include 0.5 in both discovery runs; the argmax layer moves
across data seeds (L1/L22/L24 vs L13/L19) — exploratory peak instability,
exactly as the multiple-comparisons rule predicts.

## 4. Scientific state of E00

- **Anti-leakage construction works.** The counterfactual-twin design makes the
  label's unigram distribution identical across classes by construction.
  Empirically the TF-IDF surface baseline collapses from the small-n artifact
  (0.61 at n=500, ≈1.6σ noise) to chance at n=2000 (0.454 / 0.516). No lexical
  shortcut generalizes at scale.
- **Gate 1 not yet passed.** Linear decodability of the truth label from
  `resid_post` at either token site is at most marginal (AUROC ≈ 0.53–0.54)
  under this maximally-pure task variant; the layer profile is near-flat.
- This is recorded in `research/EXPERIMENT_REGISTRY.yaml`
  (`E00.status: piloted_phase0a`) with an explicit do-not-escalate note for this
  dataset variant: E01-style causal scans should wait for either a stronger
  latent (tasks where the model demonstrably succeeds behaviorally) or matched
  relaxed-purity comparison arms where D is expected to exist.

**No causal claims are made anywhere from probe results.**

## 5. Engineering problems found & fixed during bring-up

1. **Transformers ≥4.5x changed `output_hidden_states` semantics** — entries now
   hold *entering*-layer residual states and the final element is
   post-final-norm; nothing equals any raw decoder-layer output. Fix:
   hook-based extraction as production path + load-time calibration;
   hidden-states path restricted to interior-layer cross-validation only.
2. **BPE merges `?` into trailing whitespace** (`?\n`): end-in-span selection
   silently skipped the question-final token. Fix: start-in-span rule with the
   resolved char span kept in provenance.
3. **Resume bug**: shard accounting was row-based while work units were
   sample-based, causing silent skips of partial work. Fix: deterministic
   work-unit-space shard boundaries and longest-complete-prefix resume with
   stale-shard pruning (verified: recomputed exactly the missing units).
4. **`tensor_file` mis-stamping** when batches crossed flush boundaries; fixed
   by stamping at flush time.
5. Small-n metric instability quantified via random-label spreads, which
   correctly flagged the pilot-scale "text baseline 0.61" as noise.

## 6. Repository state & reproduction

- Core external repos cloned to `external/repos/` (SHAs logged per run):
  nnsight, pyvene, axbench, internal_probing, Reasoning-Flow,
  multi-component-causal-tracing, activation-steering. Only nnsight is imported
  in the main environment; it stays dependency-clean.
- Activation cache: `data/cache/activations/<exp>_<config-hash8>/shard_*/`.
- Reproduce a discovery-scale run:
  ```
  .venv\Scripts\python.exe -m representation_reliability.cli e00 ^
      --set dataset.n_samples=2000 --set statistics.bootstrap_samples=1000
  ```

## 7. Next steps (in order)

1. **Strengthen or compare latents before more scanning**: choose one task where
   the model demonstrably answers correctly at high rate (behavioral floor) and
   measure D there — the E00 null may reflect dataset purity rather than absent
   representation. Candidate variants: fact-recall probes at the answer
   position; a relaxed twin structure as an ablation arm.
2. If D clears Gate 1 there, proceed to **E01 Decode→Causal Gap**
   (pyvene/NNsight replacements, norm-matched random controls, α-sweeps).
3. Formal multi-seed comparisons (≥3 seeds, paired bootstrap) before any
   escalated finding, per docs/METRICS.md.

