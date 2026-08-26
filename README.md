# Representation Reliability

A discovery-oriented research codebase for studying **when internal signals in language models can be decoded, trusted, causally used, robustly monitored, and safely manipulated**.

The project asks a stronger question than "can we probe a concept?" or "can we steer a behavior?":

> **When an internal representation exposes information to an external observer, when is that information actually used by the model, controllable by intervention, stable under perturbation, and safe to rely on?**

## Reliability dimensions

For target variable/behavior \(z\), representation \(h_{\ell,t}\), and output \(y\):

- **D — Decodability:** can an external readout recover \(z\) from \(h_{\ell,t}\)?
- **C — Causal use:** does a counterfactual intervention on \(h_{\ell,t}\) predictably change \(y\)?
- **S — Steerability:** can \(h_{\ell,t}\) control behavior with a usable dose-response relation?
- **M — Monitorability:** can internal state forecast correctness, hallucination, uncertainty, or failure?
- **R — Robustness:** do D/C/S/M persist under paraphrase, distribution shift, token/layer changes, and model changes?
- **K — Collateral safety:** does an intervention achieve the target without unrelated capability/behavior drift?

\[
\mathcal{R}(z,\ell,t)=[D,C,S,M,R,K]
\]

Primary discovery targets are **dissociations**:

```text
D >> C              decodable but not causally used
D ~ C >> S          causal but control-resistant
M high, C low       monitor sees a problem the model ignores
S high, R low       steerable but brittle
S high, K low       target changes with large collateral drift
```

## Research philosophy

1. Discover phenomena before scaling.
2. A high score alone is not a mechanism; a reproducible mismatch often is.
3. Test the same examples/layers/token sites across D, C, S, and M whenever possible.
4. Never infer causal use from probe accuracy.
5. Every intervention gets magnitude-matched random controls and norm diagnostics.
6. Small-model falsification precedes 7B/8B validation.
7. Keep negative results if they eliminate a plausible mechanism.
8. External repos are reference/baseline implementations, not this project's architecture.

## Model ladder

1. `Qwen/Qwen3-0.6B` — smoke tests, trajectory scans, dense instrumentation.
2. `Qwen/Qwen3-1.7B-Base` — primary causal discovery model.
3. `Qwen/Qwen3-1.7B` or another 1B–3B instruct model — monitoring/natural QA.
4. `Qwen/Qwen2.5-3B-Instruct` — cross-model robustness.
5. 7B/8B only after a phenomenon passes the early gates.

Normal target: **each experiment <= 12 GPU-hours on a 16GB-class GPU**.

## Core external repositories

- `https://github.com/ndif-team/nnsight.git`
- `https://github.com/frankaging/pyvene.git`
- `https://github.com/stanfordnlp/axbench.git`
- `https://github.com/zazamrykh/internal_probing.git`
- `https://github.com/MasterZhou1/Reasoning-Flow.git`
- `https://github.com/ZiruiYan/multi-component-causal-tracing.git`
- `https://github.com/IBM/activation-steering.git`

Optional references and isolation rules are in `docs/EXTERNAL_MODULES.md`.

## Coding-agent reading order

1. `AGENTS.md`
2. `docs/RESEARCH_CHARTER.md`
3. `docs/IMPLEMENTATION_SPEC.md`
4. `docs/METRICS.md`
5. `docs/FALSIFICATION_GATES.md`
6. `docs/EXPERIMENT_PLAN.md`
7. `external/manifest.yaml`
8. the experiment config being implemented

## First complete scientific deliverable

For the same task/examples, produce layerwise:

\[
D_\ell,\quad C_\ell,\quad S_\ell,\quad K_\ell
\]

with confidence intervals and matched controls.

A pathway is escalated only if a dissociation is:
- larger than control variation;
- stable across seeds;
- not explained by tokenization/lexical leakage;
- preserved or systematically transformed under at least one innocuous transformation;
- confirmed on held-out examples.
