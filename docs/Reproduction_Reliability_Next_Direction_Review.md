# Representation Reliability
## Evidence synthesis, novelty audit, and recommended next research direction

**Literature cutoff:** 30 August 2026  
**Repository snapshot checked:** `02d48ebd27cdeb088dc90d8f65604abfcb97c78d`  
**Inputs reviewed:** current proposal, current-results record, the four-sheet literature workbook, the repository, all 23 papers in `2_Novelty_Threats`, and additional high-relevance papers found during targeted searches.

---

## Executive decision

The strongest next scientific direction is **not** another static demonstration that information can be decoded but is not used. That result is now heavily occupied by several 2025-2026 papers, including direct demonstrations in language, vision, multimodal, tool-use, dynamics, and clinical settings.

The project should instead become a study of **causal organization as a dynamic object**:

> **How does the causal organization of a fixed semantic variable persist, decay, or reorganize as computation proceeds, even when the variable remains decodable?**

The flagship next experiment should be an upgraded E15:

# **Temporal Persistence and Reorganization of Causal Organization**

At each controlled horizon `k`, estimate

\[
D(k),\quad P(k),\quad Q(k),\quad A(k),\quad G(k),\quad B(k),
\]

where:

- `D(k)` is semantic decodability;
- `P(k)` is propagation/retention of an intervention;
- `Q(k)` is scalar semantic-coordinate actionability;
- `A(k)` is the independent additive contribution of structured probe-orthogonal state;
- `G(k)` is the interaction between scalar semantic state and structured context;
- `B(k)` is native behavioral performance.

The high-value result is not merely `D(k) > C(k)`. It is a **componentwise causal-organization trajectory**, for example:

\[
D(k)\approx 1,\qquad Q(k)\downarrow,\qquad A(k)\text{ persists},\qquad G(k)\text{ appears, disappears, or changes sign}.
\]

This would extend the present project from static transformations of causal organization - checkpoint, quantization, and distillation - to **within-computation reorganization**. In the reviewed corpus and targeted searches through the cutoff date, I did not find a paper that tracks a fixed semantic variable through time while separately estimating scalar, additive orthogonal, and interaction pathways.

Two prerequisites should come first:

1. **Finish E17 exactly as preregistered.** It determines whether the confirmed Qwen distillation dissociation has cross-family support.
2. **Run a frozen cross-model calibration audit of E01.** A new 17-model study shows that raw-unit interventions can manufacture apparent scaling trends. The within-checkpoint Q/A/G evidence remains important, but the wording “scale changes utilization” is currently stronger than the design establishes.

A generic E16 asking whether steerability emerges during pretraining should **not** be the next flagship: a 2025 longitudinal paper already makes that contribution. E16 remains valuable only after being redesigned to ask when **Q, A, and G emerge relative to D**, rather than when linear steerability first appears.

No novelty recommendation can be literally future-proof because new papers can appear. This is the most defensible direction under the literature available through 30 August 2026.

---

# 1. What the project is actually trying to establish

The original six-dimensional “representation reliability” framework is broader than the evidence now supports. The completed project has converged on a narrower, stronger scientific object:

\[
\text{representation availability}
\rightarrow
\text{propagation/context integration}
\rightarrow
\text{native readout conversion}
\rightarrow
\text{behavior}.
\]

The central question is no longer simply whether a semantic variable is encoded. It is:

> **What organization of hidden state makes an encoded variable functionally effective?**

The current factorial language gives this organization a concrete form:

\[
\mathcal O(z;\ell,t)=\big(Q,A,G,P\big),
\]

with `D` and `B` measured separately. This distinction matters:

- `D` says that an external observer can recover `z`;
- `Q` says that changing a validation-defined scalar coordinate changes native behavior;
- `A` says that structured state orthogonal to that coordinate independently matters;
- `G` says that context changes the efficacy of the scalar intervention;
- `P` localizes whether an effect survives downstream computation;
- `B` records ordinary task performance.

This is a better scientific identity than a broad checklist of decodability, steerability, monitorability, robustness, and safety. The project now has enough evidence to study **causal organization**, not merely “reliable representations.”

---

# 2. What has been done and what the results show

## 2.1 Representation cartography and the readout gap

On the synthetic relational truth task, both Qwen3 checkpoints have nearly saturated semantic decodability:

| Model | Mid/late `D` | LOFO `D` | Fixed native-readout score | Raw behavioral accuracy | Calibrated accuracy |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 0.9988 | 0.9680 | 0.7166 | 0.600 | 0.730 |
| Qwen3-1.7B | 0.9994 | 0.9917 | 0.8889 | 0.533 | 0.890 |

The 0.6B model retains `D = 1.0` on its native-error subset. Random-initialized controls remain near chance. Thus the principal failure is not absence of a linearly available semantic variable.

**Supported implication:** representation availability and native expression are separable in this task.

**Unsupported stronger implication:** a probe direction is necessarily the untouched model's endogenous semantic code.

## 2.2 Scalar intervention and source-free setpoints

A rank-one intervention on the probe coordinate is causal in both checkpoints, but the confirmed magnitude differs sharply:

| Model | Confirmed `Q0` | 95% CI |
|---|---:|---|
| Qwen3-0.6B | 0.014375 | [0.001250, 0.028125] |
| Qwen3-1.7B | 0.701250 | [0.665000, 0.737188] |

Validation-only source-free setpoints reproduce the donor-derived effects. The 1.7B model also exhibits near-uniform per-example monotonic control, whereas the 0.6B effect is mainly a small positive population response.

**Supported implication:** the decoded scalar coordinate has causal efficacy, and donor identity is unnecessary for the measured scalar effect.

## 2.3 Distributed actionability: `A` and `G`

Holding the decoded scalar coordinate fixed does not hold behavior fixed. Probe-orthogonal structured state contributes substantial independent causal signal in both checkpoints. The joint confirmation found:

| Component | Qwen3-0.6B | Qwen3-1.7B |
|---|---:|---:|
| Matched additive signal over random `A` | 1.273375 | 3.620344 |
| Structured interaction over random `G` | -0.005062, CI includes 0 | 0.132844, CI excludes 0 |

The interaction difference between checkpoints was confirmed. The layerwise trace indicates that additive context affects native readout immediately while decoded `q` is held fixed; a resolved interaction develops downstream in the 1.7B checkpoint.

**Supported implication:** in this task, actionability is not reducible to one scalar line. Structured orthogonal state carries independent signal, and the larger checkpoint displays context-dependent scalar efficacy.

**Important boundary:** “no interaction detected” in 0.6B is not evidence that its true interaction is exactly zero.

## 2.4 Propagation versus readout conversion

At the intervention layer, standardized semantic displacements are similar (`1.826` versus `1.867`). The larger difference is conversion into native readout. The project therefore classifies the checkpoint difference as a mixed bottleneck dominated by readout conversion, with a later propagation advantage for 1.7B.

This is more informative than a generic decodability/behavior gap because it locates the failure after representation formation.

## 2.5 Quantization

For Qwen3-1.7B:

| Precision | Native `D` | Frozen-BF16-axis `D` | `Q` | `A` | `G` | WikiText PPL | HellaSwag |
|---|---:|---:|---:|---:|---:|---:|---:|
| BF16 | 1.0000 | 1.0000 | 0.7125 | 3.4197 | 0.1788 | 23.38 | 0.580 |
| INT8 | 1.0000 | 1.0000 | 0.7497 | 3.6808 | 0.1936 | 23.19 | 0.576 |
| INT4 | 1.0000 | 0.9587 | 0.7669 | 2.5226 | 0.0677 | 38.61 | 0.510 |

**Supported implication:** precision-native decodability can remain perfect while higher-order actionability and geometric identity degrade.

**Necessary caveat:** general quality also degrades substantially. The result is mixed actionability fragility plus general degradation, not selective semantic damage.

## 2.6 Distillation

Both logit KD and hidden-state KD students achieve behavior at least as good as the teacher while retaining native `D = 1.0`. Yet their causal organization differs systematically:

| Regime | Behavioral gap vs teacher | `Q` gap | `A` gap | Systematic `G` mismatch? |
|---|---:|---:|---:|---|
| R2 logit KD | +0.043917 | -0.167500 | +0.421201 | No at the 0.10 SESOI |
| R3 hidden-state KD | +0.055467 | -0.173110 | +0.507840 | No at the 0.10 SESOI |

The students almost eliminate the teacher's scalar conversion pathway and over-rely on the additive contextual pathway. Hidden-state KD does not restore the teacher organization and does not clearly improve CKA at the behavior-matched checkpoint.

**Strong supported implication:** behavioral equivalence and perfect decodability do not imply causal-organizational equivalence.

The conversion-response method branch correctly remains closed. Its aggregate COD improvement was invalidated by shuffled-target controls and scale domination by `A`.

## 2.7 E17

E17 has selected OLMo-2 without inspecting causal outcomes and has constructed the reference state. The initial student already has nearly perfect `D`, leaving behavioral headroom. The repository snapshot contains the analyzer but no completed training/causal verdict.

E17 is therefore an immediate unfinished obligation, not a result that can yet be used in a paper claim.

---

# 3. What the current results imply - and what they do not

## 3.1 Strongest current scientific interpretation

The results support this picture:

\[
\boxed{\text{representation availability} \neq \text{functional availability}}
\]

and, more specifically:

\[
\boxed{\text{causal organization is distributed and transformation-sensitive}.}
\]

A semantic variable may remain perfectly readable while:

- its scalar causal efficacy changes;
- the model shifts reliance toward or away from structured orthogonal state;
- context gating appears or disappears;
- geometric identity rotates;
- behavior remains equal or improves.

That is the project's best unifying contribution.

## 3.2 Claims that should be avoided

The present evidence does not establish that:

1. the linear probe axis is the unique or natural endogenous code;
2. high `D` is necessary for all forms of control;
3. model size by itself caused the checkpoint difference;
4. the phenomenon is universal across tasks, concepts, sites, or architectures;
5. INT4 selectively damages semantic utilization independently of general degradation;
6. hidden-state similarity improved under R3;
7. conversion-response KD transferred causal organization;
8. E17 has replicated the Qwen result;
9. `Q/A/G` exhaust all causal organization, especially under nonlinear or distributed codes.

---

# 4. Literature synthesis: what is already occupied

## 4.1 “Decodable but not used” is no longer a novelty claim

The basic dissociation has historical roots in task-relevance probing and amnesic probing, and the 2026 literature now contains many direct demonstrations. The closest papers separate decoding from generation, restoration, steering, policy expression, or task use in CAD constraints, behavioral economics, hallucination/refusal, bracket models, dynamics, causal reasoning, procedural hallucination, multimodal perception, tool use, VLA progress, and clinical triage.

A paper built mainly around “the model knows internally but does not act” would now be viewed as an application or replication, not a foundational conceptual contribution.

## 4.2 The converse dissociation also exists

Function-vector work reports effective steering even when the logit lens cannot decode the answer. Counting-ViT work likewise finds causal tokens with weak probe accuracy. Therefore the framework should not imply a one-way hierarchy in which decodability is a prerequisite for causal control.

The safe statement is:

> Decodability and causal efficacy are distinct measurements. Either can be high while the other is low under a particular readout and intervention family.

## 4.3 A scalar direction is often not the causal object

Several papers find that:

- detection and control directions are geometrically misaligned;
- scalar swaps fail while full-state patches work;
- sparse distributed interventions outperform a single direction;
- nonlinear multi-layer synergy matters;
- context-dependent or curved steering can outperform fixed linear steering.

This literature overlaps the project's motivation for `A` and `G`. The remaining distinction is the project's explicit, preregistered factorial decomposition of a validation-defined scalar coordinate and structured orthogonal context, followed by held-out confirmation.

## 4.4 Interaction effects are established as a methodological issue

The multiple-mediators paper shows that activation-patching indirect effects include interactions with other components. This means “interaction matters” by itself is not novel. The project's stronger contribution is to define and experimentally manipulate a semantically interpretable interaction: decoded scalar state by structured probe-orthogonal context.

## 4.5 Causal/interventional distillation already exists

Causal Distillation for Language Models introduced DIITO to make a student imitate teacher causal dynamics through interchange interventions. The project cannot claim first causal distillation. Its defensible E13 contribution is diagnostic:

> ordinary logit and hidden-state distillation can produce teacher-equal behavior while leaving a componentwise causal-organization mismatch.

The failed conversion-response branch should not be revived merely to obtain a positive method result.

## 4.6 Generic steerability emergence during training is already claimed

A 2025 longitudinal study tracks linear steerability over pretraining and reports concept-specific onset times. Thus a generic E16 “does control emerge later than representation?” is too close to existing work.

E16 remains differentiated only if it measures the onset and reorganization of `Q`, `A`, and `G`, controls cross-checkpoint operating points, and distinguishes representation rotation from causal-pathway formation.

## 4.7 Temporal causal organization remains comparatively open

Nearby work studies:

- prediction of future tokens from a single hidden state;
- reusable hidden interfaces defined by future operations;
- sequential activation patching over chain-of-thought tokens;
- persistent latent policy states;
- early trajectory commitment in hallucination.

These papers establish that internal computation is temporally structured. They do not, however, jointly measure whether a fixed semantic variable remains decodable while its scalar, additive-context, and interaction pathways follow different persistence curves. This is the clearest unoccupied extension of the current project.

---

# 5. Paper-by-paper audit of the 23 novelty-threat entries

The entries below reflect full-text review of the title, abstract, introduction, methods, reported main results/tables, discussion, conclusions, and stated limitations where available. The “remaining space” column is an assessment, not a quotation from the paper.

| # | Paper and source | What it actually did | Principal finding | Threat to this project | Remaining defensible space |
|---:|---|---|---|---|---|
| 1 | **Encoded but Not Actionable**. [DOI](https://doi.org/10.48550/arXiv.2608.17843) | Six frozen decoder LMs; CAD SketchGraphs; local relation, global degrees-of-freedom, and forced-choice tasks; random-init/shuffled-order controls; activation restoration and mean-difference steering. | Learned local relations can be highly decodable while generation, restoration influence, and target-specific steering fail; some global probe success is already present in random networks. | Very high. It directly occupies decode/generate/influence/steer dissociation and uses “actionable.” | Factorial `Q/A/G`, source-free setpoints, transformation of causal organization, held-out confirmation, and temporal component trajectories. |
| 2 | **Representation Without Control**. [DOI](https://doi.org/10.48550/arXiv.2605.25151) | Gemma 3 4B; 54,450 prompt-level behavioral rows, 756 training pairs, held-out prompts, and 648 steering prompts; realization-status mean-difference direction. | Prompt sensitivity and held-out linear readout do not yield reliable or sign-symmetric causal control of risk choice. | Very high. Direct readout-versus-control precedent. | The project identifies *where* remaining causal signal lies (`A/G`) rather than reporting only a failed direction. |
| 3 | **Perfect Detection, Failed Control**. [DOI](https://doi.org/10.48550/arXiv.2606.24952) | Gemma 2-2B-it primary; hallucination detection, refusal steering, SAE/neuron tests, cross-model geometry, and controlled rotation toward the refusal axis. | Fake entities are perfectly separable from layer 5, but the detection direction has cosine about 0.12 with the refusal-control direction; a 15-degree rotation partly restores control. | Very high. Strong geometric version of decoding/control mismatch. | `Q/A/G` explains componentwise causal organization around a decoded semantic axis; it should not claim the probe axis is the control axis. |
| 4 | **Steerable but Not Decodable**. [DOI](https://doi.org/10.48550/arXiv.2604.02608) | 4,032 task-template pairs; 12 tasks; six base/instruction-tuned models from three families; layer sweeps, logit/tuned lenses, stronger probes, and patching. | Steering exceeds logit-lens decoding across every task/model; ten instances are steerable without decodability and only three show the predicted converse. | Very high. Refutes a one-way `D -> C` narrative. | Treat `D` and causal efficacy as non-ordered measurements; include full-state/nonlinear upper bounds when `D` is weak. |
| 5 | **Dissociating Decodability and Causal Use in Bracket-Sequence Transformers**. [DOI](https://doi.org/10.48550/arXiv.2604.22128) | Tiny two-layer, one-head Dyck transformers; probes for depth/distance/top-of-stack; direction ablation and attention-edge knockout. | Depth/distance directions are decodable but largely dispensable; the actual top-of-stack attention edge is strongly causal. | Very high conceptual precedent, but small-model/domain specific. | LLM-scale semantic decomposition, context interaction, transformations, and confirmation. |
| 6 | **The Objective Decides**. [DOI](https://doi.org/10.48550/arXiv.2607.03728) | Mechanical, circuit, PDE systems, and a 158M PDE foundation model; linear invariant probes and donor-value activation interchange; objective manipulation. | Conserved quantities can have `R2≈1` and transfer-correlation near zero, then become causally load-bearing when the objective rewards them; deployment gap predicts OOD accuracy. | Very high. A clean representation-versus-deployment result with causal intervention. | Semantic LLM causal organization, orthogonal/additive/context interaction, distillation/quantization, and temporal pathway reorganization. |
| 7 | **Patches of Nonlinearity**. [DOI](https://doi.org/10.48550/arXiv.2602.07930) | OLMo-2 1B/7B base, SFT, and DPO; eight instruction tasks; instruction vectors, layer patching, and multi-layer composition. | Instructions are highly linearly separable, but multi-layer patches show super-additive nonlinear synergy and early representations select later circuits. | High overlap with nonlinear/distributed actionability. | A semantically explicit factorial estimand (`Q/A/G`) and componentwise transformation/persistence. |
| 8 | **The Curse of Multiple Mediators**. [DOI](https://doi.org/10.48550/arXiv.2606.27510) | Causal derivation of activation patching; `NIE = PIE + INT`; GPT-2 IOI experiments; pairwise/higher-order interaction analysis. | Patching effects include state-dependent interactions; component rankings can change substantially, and interaction variance explains faithfulness instability. | Very high methodological threat to simplistic patch interpretations. | The project deliberately estimates an interpretable scalar-by-context `G`, rather than treating all interaction as hidden contamination. |
| 9 | **Distributed Sparse Interventions**. [DOI](https://doi.org/10.48550/arXiv.2607.07128) | Qwen3-8B, Gemma3-4B, Llama3.2-3B; 12 tasks; optimized interventions on 8-64 neurons across layers. | Sparse distributed, nonlinear interventions can match or exceed few-shot ICL and reveal shared plus task-specific neurons. | High. Shows one-direction steering is incomplete. | Probe-relative factorial organization, causal comparison under transformation, and temporal profiles. |
| 10 | **Causal Activation Steering via Sparse Mediation**. [DOI](https://doi.org/10.18653/v1/2026.findings-eacl.57) | Front-door-style mediation objective with learned direction and sparse binary mask; Llama2-7B/Mistral-7B; power, wealth, hallucination, and jailbreak tasks. | 70-90% sparsity retains roughly 97-100% of dense steering performance in the reported settings. | High method overlap with causal steering and mediation. | `Q/A/G` is a diagnostic decomposition, not a sparse steering optimizer; E15 would add temporal causal organization. |
| 11 | **Hidden APIs in Language Models**. [DOI](https://doi.org/10.48550/arXiv.2607.27617) | Forked futures sampled after prefix-state formation; prequential causal MDL; Shared/Local/Mixture/Distributed interface competition; transplantation and mediation. | Identical current outputs can hide states that support different future computations; Shared interfaces win in the detailed evaluations and mediate much more target effect than null paths. | High, especially for future-computation framing. | Track a labeled semantic variable's `D/Q/A/G` over horizon rather than infer an unlabeled interface architecture. |
| 12 | **Causality != Decodability, and Vice Versa: Counting ViTs**. [DOI](https://doi.org/10.48550/arXiv.2510.09794) | Small counting ViT; probes and patching across image-token and CLS positions using controlled image pairs. | Middle object tokens can be causal despite weak exact-count decoding; late CLS states can decode count but be causally inert. | High because it demonstrates both directions of the dissociation. | LLM-scale semantic decomposition, context interaction, transformations, and longitudinal organization. |
| 13 | **Model-Adaptive Tool Necessity**. [DOI](https://doi.org/10.48550/arXiv.2605.14038) | Four LLMs; 4,000 arithmetic and 817 TruthfulQA instances; stochastic no-tool runs define model-specific necessity; cognition/action probes. | Models frequently know whether a tool is needed but fail at the execution transition; mismatch rates are substantial across models/tasks. | High applied knowing-doing precedent. | Mechanistic causal decomposition and direct interventions, rather than probe/output mismatch alone. |
| 14 | **They Infer What You Meant**. [DOI](https://doi.org/10.48550/arXiv.2607.03598) | Surface-matched recognition/evaluation prompts, 60 objects, eight phrasings, leave-one-phrase-out tests; Qwen2.5-3B primary plus cross-model study; steering. | Communicative intent is represented more reliably than acted upon; steering can substantially improve some models, but effects are model-specific. | High direct representation/readout precedent. | Distinguish scalar, additive context, and gating pathways; examine persistence or transformation. |
| 15 | **Senses Wide Shut**. [DOI](https://doi.org/10.48550/arXiv.2605.13737) | 500 audiovisual clips; 2x2 standard/misleading conditions; eight open omni models plus Gemini; ablations, probes, residualization, and probability-guided logit adjustment. | Perceptual evidence can be internally recoverable while action under misleading prompts collapses; adjustment improves accuracy but does not establish hidden-state causal organization. | High multimodal representation-action precedent. | Componentwise causal organization and interventions; multimodal replication could be a later task extension. |
| 16 | **Causal Tongue-Tie**. [DOI](https://doi.org/10.48550/arXiv.2605.25891) | Eight instruction-tuned models; anti-commonsense causal benchmarks; fixed probes, SVD subspaces, scalar swaps, projection, and full-state patches. | Causal direction can be decoded at about 0.97 while Yes/No output is about chance; scalar swaps do not reliably flip answers, but fuller state manipulation improves behavior. | Very high. Extremely close to scalar insufficiency and readout bottleneck. | Explicit orthogonal additive/context interaction decomposition, held-out confirmation, and transformation/persistence. |
| 17 | **Attention Deficits**. [DOI](https://doi.org/10.48550/arXiv.2602.19239) | Long-context procedural binding; stage-specific probes and late-MLP patch/checkpoint interventions. | Correct binding remains decodable on error trials, and late restoration can recover behavior, localizing failures to attention/binding and readout stages. | Medium-high. Strong stage-localized encoded-but-unexpressed result. | General factorial causal organization and temporal pathway persistence. |
| 18 | **Causal Probing for Internal Visual Representations in MLLMs**. [DOI](https://doi.org/10.48550/arXiv.2605.05593) | Six MLLMs; four visual concept categories; mean-difference directions, steering/suppression, semantic/logit metrics, and architecture/scale comparisons. | Entity concepts are more localized, abstract concepts more distributed; reverse steering can trigger compensatory activation; geometric recognition may not activate reasoning. | Medium-high, especially for scale and distributed representation claims. | The project's paired factorial decomposition and teacher/student transformation story. |
| 19 | **Scale Determines Whether Language Models Organize Representation Geometry for Prediction**. [DOI](https://doi.org/10.48550/arXiv.2605.17084) | Seven Pythia sizes plus three cross-family models; Subspace PGA aligns representation-distance geometry with the unembedding readout; training checkpoint analysis. | Small models lose late predictive organization during training while retaining underlying structure; large models preserve it. | Very high for a scale/readout-organization narrative. | Direct causal `Q/A/G` intervention and behavior; avoid claiming first size-dependent representation-to-readout organization. |
| 20 | **Actionable Neural Representations**. [DOI](https://doi.org/10.48550/arXiv.2209.15563) | Group/representation-theoretic account of representations that transform consistently under actions; derives grid-cell structure under minimal constraints. | “Actionable” already has an established representation-learning meaning tied to predicting consequences of actions. | Naming/terminology threat, not direct empirical overlap. | Define “actionability” operationally as intervention-to-native-behavior efficacy and acknowledge the earlier usage. |
| 21 | **Manifestation Unit Protocol**. [DOI](https://doi.org/10.48550/arXiv.2607.00089) | A typed protocol `M=(E,S,R,D,G,T)` for storing/retrieving mechanistic findings; beta-VAE, CNN, and GPT-2 demonstrations. | Typed representation records improve retrieval and reuse; some retrieved units satisfy causal sufficiency/necessity tests. | Medium-high infrastructure/terminology overlap. | Scientific estimands and transformation findings, not a protocol for recording mechanisms. |
| 22 | **Cell-Based Representation of Relational Binding**. [DOI](https://doi.org/10.18653/v1/2026.acl-long.2194) | Five controlled discourse domains; Llama3-8B-Instruct and Qwen3-8B; PLS latent components; dense 2-D intervention grids and index steering. | Entity/relation indices form causal cell-like structures; perturbing the learned relational representation changes bound answers across domains/templates/models. | High positive counterexample and strong adjacent task. | An excellent external testbed for whether `Q/A/G` generalizes beyond binary truth; it does not itself provide the proposed factorial decomposition. |
| 23 | **Decoding Task Progress from VLA Representations**. [DOI](https://doi.org/10.48550/arXiv.2608.13474) | Pi0.5/PaliGemma VLA; ten VLABench tasks with 500 trajectories each; naive and contrastive probes, random-label controls, OOD detection, and activation injection. | Task progress is highly decodable, but naive probes exploit proxies and the tested hidden-state injection does not propagate or meaningfully steer policy. | High direct decodability/steerability precedent in embodied models. | Componentwise causal organization and robust temporal persistence rather than a single progress direction. |

---

# 6. Additional papers that materially change the recommendation

## 6.1 Interpretability without actionability in clinical triage

This paper was not in the novelty-threat sheet but is a direct threat. It compares four mechanistic intervention families on 400 physician-adjudicated vignettes. A Qwen2.5-7B linear probe reaches 98.2% AUROC while output sensitivity is 45.1%. Concept steering disrupts correct detections more often than it repairs errors, SAE steering has zero effect despite thousands of significant features, and high-strength truth-vector steering leaves most errors uncorrected.

**Implication:** “near-perfect internal representation but weak correction” is already demonstrated in a safety-critical application. The project must emphasize causal organization, not a generic knowledge-action gap.

Source: [Basu et al., 2026](https://doi.org/10.48550/arXiv.2603.18353).

## 6.2 CASE: decodability useful for selection but not steering

A paper posted on 17 August 2026 trains a leakage-free correctness gate on answer-token hidden states. Within-question decodability predicts held-out gains of hidden-state candidate selection over majority voting (`r=0.75`) and identifies a useful threshold near AUC 0.60. Yet four intervention variants using the correctness direction do not improve generation and can reduce accuracy.

**Implication:** external readout utility is itself a distinct functional property. A representation can be useful for monitoring/selection while not being a causal generative lever.

Source: [Wang, Hong, & Bagci, 2026](https://doi.org/10.48550/arXiv.2608.17124).

## 6.3 Measurement confounds in cross-model steering

Wu, Zhao, and Chen audit 17 models from five families. With raw activation units and a fixed layer/coefficient, concept steerability appears to scale. With residual-norm-comparable interventions and held-out operating-point selection, the Qwen3 scale trend is no longer significant, although a moderate positive slope is not ruled out.

This is the most important methodological challenge to the project's current “scale changes utilization” language.

The project already has meaningful protections:

- validation-defined semantic setpoints;
- matched examples and controls;
- similar standardized `q` displacement at the intervention layer;
- an untouched confirmation;
- layerwise propagation/readout tracing.

But it does not yet establish that cross-checkpoint perturbations are comparable in residual-norm or on-manifold units. In the discovery trace, mean raw oriented coordinate displacement is about `1.63` for 0.6B and `18.45` for 1.7B, while standardized displacement is matched. Because the probe direction is unit normalized, raw scalar displacement is also the rank-one intervention norm. Whether residual-stream norms scale enough to make these interventions comparable has not been reported.

**Required action:** run a frozen sensitivity audit before interpreting the magnitude difference as a scale effect.

Source: [Wu, Zhao, & Chen, 2026](https://doi.org/10.48550/arXiv.2608.08159).

## 6.4 Controllability emergence during pretraining

She et al. already track linear steerability through pretraining and report concept-specific onset stages. This directly occupies the generic E16 idea.

**Implication:** E16 should be reframed as **emergence of distributed causal organization**:

\[
t_D,\quad t_Q,\quad t_A,\quad t_G,\quad t_B,
\]

with residual-norm-comparable intervention curves and frozen sites. The novel result would be an ordering or reorganization of pathways, not first emergence of linear steerability.

Source: [She et al., 2025](https://doi.org/10.48550/arXiv.2508.01892).

## 6.5 Temporal neighboring work

Four lines of work bound the temporal novelty claim:

- **Future Lens** shows that single hidden states predict future tokens and can be transplanted causally. [DOI](https://doi.org/10.48550/arXiv.2311.04897)
- **Hidden APIs** defines reusable interfaces from future-operation signatures. [DOI](https://doi.org/10.48550/arXiv.2607.27617)
- **Hallucination as Trajectory Commitment** studies early attractor commitment and asymmetric correction/corruption across generation steps. [DOI](https://doi.org/10.48550/arXiv.2604.15400)
- **Sequential Activation Patching** traces distributed causal effects over chain-of-thought token positions. [DOI](https://doi.org/10.48550/arXiv.2608.22332)
- **Persistent Latent Policy States** models reasoning trajectories as switching dynamical systems and tests causal state swaps. [DOI](https://doi.org/10.48550/arXiv.2607.18532)

These papers mean the project cannot claim first temporal causal analysis of LLM hidden states. The remaining gap is narrower and stronger: **componentwise persistence and reorganization of a fixed semantic variable's causal organization while decodability is measured in parallel.**

---

# 7. Novelty verdict: what to claim, modify, or drop

| Candidate claim | Verdict | Recommended wording |
|---|---|---|
| “Models can encode information without using it.” | Not novel | Use only as motivation and cite prior work. |
| “Decodability does not imply steerability.” | Not novel | Treat as established background. |
| “Steerability and decodability are separable.” | Not novel, and bidirectional | State that neither orders the other under fixed measurement families. |
| “A probe direction can be causal.” | Established in many settings | Your contribution is the controlled setpoint and decomposition, not causality alone. |
| “Causal information is distributed/nonlinear.” | Crowded | Emphasize the explicit `Q/A/G` factorial estimands and confirmation. |
| “Interaction matters in activation patching.” | Not novel | Emphasize semantic scalar-by-context interaction rather than generic mediator interaction. |
| “Scale improves actionability.” | Currently vulnerable | Say the tested 1.7B checkpoint has stronger conversion; treat scale as a hypothesis until calibrated across more checkpoints. |
| “Quantization preserves representation but harms use.” | Supported with caveat | State mixed actionability fragility plus general degradation. |
| “Distillation can preserve behavior but change mechanism.” | Strong and defensible | Specify the measured `Q/A` mismatch, `D=1`, and teacher-equal/exceeding behavior. |
| “Hidden-state KD guarantees teacher mechanism.” | Refuted in your setting | Phrase as a negative diagnostic, not a universal impossibility theorem. |
| “First causal distillation method.” | False | DIITO is direct prior art; the conversion-response branch also failed. |
| “Linear controllability emerges during training.” | Prior art | Upgrade to onset/reorganization of `Q/A/G`. |
| “Causal organization can persist or reorganize over horizon while `D` stays high.” | Best open direction | Make this the next flagship, with strong temporal controls and confirmation. |

## Strongest present novelty statement

A defensible current statement is:

> In a controlled semantic task, a linearly decoded variable has a causal scalar component, but its behavioral efficacy is distributed across scalar state, structured probe-orthogonal state, and a model-dependent interaction between them. This causal organization can change under compression or distillation without destroying native decodability, and behaviorally equivalent students can retain a systematically different organization.

This is narrower than the original project ambition, but materially stronger and more publishable.

---

# 8. Mandatory validity audit before a scale/checkpoint claim

## 8.1 Why the audit is needed

The current project compares two trained checkpoints, not a randomized manipulation of parameter count. They differ in size, training trajectory, internal scale, and possibly many unobserved properties. “Scale causes stronger utilization” is therefore not identified even if the effect is real.

Furthermore, the new cross-model audit literature shows that intervention units and operating points can create apparent scaling trends.

## 8.2 Frozen audit design

Do not retune E01 or reopen the consumed holdout. Use the already frozen directions, sites, semantic targets, and examples.

For each model and every existing intervention row, compute or recover:

1. `||Delta h|| / ||h||` at the intervention site;
2. `||Delta h||` relative to the natural within-class and between-class distance distributions;
3. achieved `Delta q` in native units and validation SD units;
4. local Mahalanobis distance, k-nearest-neighbor distance, or a density-ratio diagnostic for the edited state;
5. native-margin change in raw and validation-standardized units;
6. no-op, random, random-orthogonal, same-label, shuffled, and full-patch effects at matched residual fractions.

Then report effect curves rather than one operating point:

\[
Q(r),\quad A(r),\quad G(r),\qquad r=\frac{\|\Delta h\|}{\|h\|}.
\]

Predeclare a small residual-fraction grid. Match either:

- residual fraction `r` across checkpoints;
- achieved standardized semantic shift `Delta q_z` across checkpoints;
- or both through two-dimensional interpolation.

No single normalization is guaranteed to be uniquely correct, so the conclusion should be robust across reasonable rulers.

## 8.3 Interpretation outcomes

- **Difference persists:** the readout-conversion contrast is much stronger and can be described as a robust checkpoint difference. Scale remains an interpretation, not a causal conclusion.
- **Difference shrinks but remains:** report the calibrated effect, not the raw ratio.
- **Difference disappears:** retain the within-checkpoint causal mechanism and distributed `Q/A/G` result, but withdraw the scale story.
- **Edited states are strongly divergent:** add on-manifold regularization or natural-donor sensitivity analyses and narrow causal claims.

---

# 9. Recommended flagship: Temporal Persistence and Reorganization of Causal Organization

## 9.1 Primary research question

> When a semantic state must remain relevant across delayed computation, does its linearly available representation persist longer than the causal organization that connects it to future action?

The stronger form asks:

> Do scalar, additive-context, and interaction pathways have distinct temporal trajectories, including substitution or reorganization rather than uniform decay?

## 9.2 Why this is the right next step

It satisfies five criteria simultaneously:

1. **Directly builds on confirmed machinery.** The project already has `D`, source-free `Q`, structured `A`, factorial `G`, tracing, pair-cluster inference, and integrity checks.
2. **Adds a genuinely new axis.** Current results compare static systems; temporal persistence studies the same causal organization as computation evolves.
3. **Survives the novelty threats better than E16.** Temporal LLM studies exist, but componentwise `D/Q/A/G` persistence for a fixed semantic variable was not found.
4. **Can produce informative nulls.** Co-decay, reorganization, or stable organization each answer a real mechanistic question.
5. **Unifies the publication story.** Compression, distillation, training, and temporal computation become transformations of causal organization.

## 9.3 The key conceptual upgrade: reorganization, not only half-life

The existing E15 design emphasizes `H_C < H_D`. Keep that as a possible summary, but do not make exponential decay or monotonicity an assumption.

At each horizon define:

\[
\mathcal O(k)=\big(Q(k),A(k),G(k)\big).
\]

Then measure:

- componentwise persistence curves;
- area under each normalized curve;
- first stable departure from baseline;
- changepoints;
- sign changes;
- substitution, such as `Q` decreasing while `A` increases;
- geometric rotation of the probe axis;
- causal coverage relative to a full-state patch.

Report a “half-life” only when a component is sign-stable and sufficiently monotone. Otherwise use nonparametric curve summaries.

---

# 10. Preregistration-ready experimental blueprint

## 10.1 Task design

Start with one controlled stateful task, not an unconstrained agent benchmark.

Recommended first task: **delayed relational-state binding with a causal bottleneck**.

Each counterfactual pair should differ only in a latent state `z`, such as:

- a goal identity;
- a permission/forbidden-action flag;
- a relation that determines a delayed choice;
- an inventory/resource state;
- a delayed procedural constraint.

After `z` is established, insert controlled distractor operations and make the final decision depend on `z`.

### Essential design requirement: separate two kinds of distance

Let:

- `a` = age of the semantic state, or number of controlled distractors since encoding;
- `r` = remaining computational distance from the intervention site to the decision.

Vary `a` while holding `r` fixed in the primary experiment. Otherwise an apparent decline in `Q/A/G` may simply reflect a longer propagation path after intervention.

Use matched variants to separately estimate:

1. **state-age/interference effects:** vary `a`, fix `r`;
2. **post-intervention propagation effects:** vary `r`, fix `a`.

## 10.2 Preventing bypass

A standard long prompt leaves the original evidence in attention memory, so later computation may bypass the chosen carrier. The primary carrier must therefore pass a causal sufficiency gate.

Use one predeclared bottleneck implementation:

- a designated state token whose hidden/KV state is transplanted into a common future suffix;
- a controlled recurrent memory token;
- or a forked-future prefix state with a shared post-bottleneck operation bank.

Before any `Q/A/G` interpretation, require:

- full-state counterfactual patching at the bottleneck to change the delayed decision strongly;
- a no-op patch to be numerically null;
- a random same-norm patch to be weak;
- the shared suffix to contain no direct textual copy of `z`;
- source/base pairs to remain matched outside `z`.

If full-state patching is weak, the carrier is not causally sufficient and the task should be redesigned rather than analyzed post hoc.

## 10.3 Models and sites

Primary discovery should use one manageable model with reliable task behavior and one frozen site. Do not begin with several families.

A reasonable order is:

1. Qwen3-1.7B, because its `Q/A/G` organization is confirmed and controllable;
2. one second model only after the task and carrier pass all integrity gates;
3. if E17 is positive, OLMo-2 becomes a strong replication candidate;
4. if E17 is negative, do not switch families post hoc to chase a positive temporal result.

Use a fixed absolute or preregistered relative-depth site. Do not select a different “best” layer at every horizon.

## 10.4 Representation measurements

For every horizon `k`, estimate:

1. `D_native(k)`: a horizon-specific probe trained without leakage;
2. `D_frozen(k)`: performance of the origin or reference probe axis at horizon `k`;
3. `cos(u_0,u_k)` and subspace alignment: distinguishes persistence from geometric rotation;
4. LOFO or held-out relation-family generalization;
5. random-label and exact-architecture random-init controls;
6. text/surface baselines and position/distractor-count baselines, including recency and rehearsal controls;
7. calibration metrics if the probe is interpreted probabilistically.

Predeclare two distinct causal estimands and do not mix them:

- **Native-local organization** `O_native(k)`: use the horizon-specific axis `u_k` and validation-only setpoints at `k`. This asks whether the semantic code available *at that horizon* is actionable.
- **Transported-reference organization** `O_ref(k)`: keep the early/reference axis `u_0` frozen and test it at later horizons. This asks whether the *original coordinate* remains functionally connected after computation proceeds.

A decline in `O_ref(k)` with stable `O_native(k)` indicates code rotation or pathway migration, not simple loss. A decline in both is stronger evidence of functional decay.

Use pair-, episode-, and template-grouped splits. The CASE paper shows how ordinary random splits can create question-identity leakage.

## 10.5 Causal measurements

At each horizon run the same four-arm factorial design for the predeclared native-local estimand, and repeat the transported-reference estimand at a smaller prespecified set of horizons:

\[
Y_{00},\quad Y_{10},\quad Y_{01},\quad Y_{11}.
\]

Estimate:

\[
Q(k)=Y_{10}-Y_{00},
\]

\[
A_c(k)=Y_{01}-Y_{00},
\]

\[
G_c(k)=\big(Y_{11}-Y_{10}\big)-\big(Y_{01}-Y_{00}\big).
\]

Use validation-only source-free setpoints for `Q`. Construct matched probe-orthogonal context at the same horizon for `A/G`.

Required causal arms:

- scalar setpoint;
- matched structured orthogonal context;
- same-family shuffled context;
- different-family shuffled context;
- random orthogonal context;
- irrelevant-state context;
- same-label context;
- full-state patch upper bound;
- no-op.

Add a nonlinear or sparse upper-bound arm at a limited set of horizons if resources allow. This prevents the project from interpreting weak linear `Q/A/G` as absence of causal organization when control may live in a nonlinear subspace.

## 10.6 Intervention comparability and on-manifold diagnostics

For every arm/horizon record:

- `||Delta h||`;
- `||Delta h||/||h||`;
- achieved target-coordinate displacement;
- orthogonality error `|u^T v_perp|`;
- activation-density or kNN distance;
- full-state/natural-donor distance;
- selected-logit leakage before the intended hook;
- downstream retention of the perturbation.

Compare effects at matched semantic displacement **and** matched residual fraction. Include a natural-donor or distribution-regularized sensitivity analysis because causal interventions can create divergent states.

## 10.7 Primary hypotheses

Do not preregister a vague “gap.” Use effect-size thresholds.

### H15.1 - representational persistence

At a predeclared delayed horizon `k*`, native decodability is non-inferior to the early reference:

\[
D_{native}(k^*)-D_{native}(k_0)>-\delta_D.
\]

A reasonable `delta_D` must be selected from pilot variability, not chosen after seeing results.

### H15.2 - causal-organization change under persistent representation

At the same `k*`, at least one of `Q`, `A`, or `G` differs from its early value by more than a preregistered SESOI, with family-wise correction.

This conjunction is the core dissociation:

\[
D\text{ persists} \quad\land\quad \mathcal O\text{ changes materially}.
\]

### H15.3 - differential pathway persistence

At least two normalized component curves differ materially over the preregistered horizon set. Test curve contrasts, not only separate pointwise significance.

### H15.4 - semantic interference specificity

At matched token count and position, semantically competing distractors alter `Q/A/G` more than neutral filler or random-state distractors.

### H15.5 - causal organization predicts delayed failure beyond `D`

On untouched examples, componentwise causal measurements improve prediction of delayed behavioral errors beyond `D`, surface controls, and native confidence. This should be secondary unless sample size supports a strictly held-out test.

## 10.8 Statistical plan

- Cluster all uncertainty by counterfactual `pair_id` or episode, resampling the entire horizon curve together.
- Use untouched confirmation data after all horizons, sites, tasks, and estimands are frozen.
- Correct the primary family across `Q/A/G` and the prespecified curve contrasts.
- Use equivalence/non-inferiority for “`D` remains high”; failure to reject a difference is not evidence of persistence.
- Use SESOIs for causal changes; tiny nonzero effects are not scientifically sufficient.
- Predeclare missing/invalid trajectory handling.
- Power the experiment by simulation from a bounded pilot, targeting at least 90% power for the smallest primary SESOI. Do not inherit `n=100` pairs merely because it worked in E01.
- Report discovery and confirmation separately.

## 10.9 Integrity and falsification gates

### Gate 0 - task validity

- `z` is exactly labeled;
- matched pairs differ only in `z`;
- delayed decisions genuinely depend on `z`;
- no surface shortcut achieves strong performance.

### Gate 1 - carrier sufficiency

- full-state patch has a strong target-oriented effect;
- no-op is zero;
- random controls are weak;
- the shared future suffix cannot directly recover the original text evidence.

### Gate 2 - intervention validity

- target `q` is reached;
- context remains orthogonal;
- residual fractions are comparable;
- edited states are not pathologically off-manifold;
- no pre-hook or token-index leakage occurs.

### Gate 3 - discovery

Proceed to confirmation only if:

- `D` is interpretable over the selected horizon;
- at least one causal component has a stable, non-artifactual trajectory;
- controls do not reproduce the effect;
- the result is not entirely explained by intervention-to-output distance.

### Gate 4 - confirmation

One-shot untouched holdout, frozen hypotheses, pair-cluster inference, and explicit access ledger.

## 10.10 Interpreting possible outcomes

| Outcome | Interpretation |
|---|---|
| `D` stable; `Q/A/G` decay at different rates | Flagship support for componentwise causal persistence. |
| `D` stable; `Q` falls while `A` or `G` rises | Stronger result: causal organization reorganizes rather than simply decays. |
| `D` and all causal components decay together | Informative null against the proposed dissociation in that task. |
| `D` stable; `Q/A/G` weak; full patch remains strong | The relevant causal code is outside the current linear/factorial decomposition; investigate nonlinear/distributed organization. |
| Full patch is weak | Carrier/task invalid for this question; redesign rather than interpret. |
| Effects disappear under residual matching/on-manifold control | Original temporal effect was an intervention artifact. |
| Neutral and semantic distractors are equivalent | Temporal distance, not semantic interference, is the likely driver. |

---

# 11. How E16 should be redesigned

E16 should be retained as the second study in the same program, not the immediate flagship.

Replace the current hypothesis

\[
t_D<t_C
\]

with a richer set:

\[
t_D,\quad t_Q,\quad t_A,\quad t_G,\quad t_B.
\]

Questions that remain novel enough:

1. Does `D` plateau before `Q`?
2. Does additive contextual signal `A` emerge before reliable scalar control?
3. Does interaction/gating `G` appear only after a model learns a mature readout architecture?
4. Do causal pathways substitute during training, even when `D` and behavior are smooth?
5. Does distillation reproduce the mature pathway order or construct an alternative route?

The design must use:

- a fixed architecture/checkpoint series;
- frozen site and token rules;
- residual-norm-comparable intervention curves;
- horizon/checkpoint-specific and frozen-axis probes;
- no best-layer search per checkpoint;
- exact training-token metadata;
- general-quality controls;
- replication across at least two training runs or families before making a developmental-law claim.

This would be a study of **causal-organization ontogeny**, not a duplicate of existing linear-steerability emergence work.

---

# 12. Ranked alternatives

## Rank 1 - Temporal causal-organization persistence/reorganization

**Novelty:** highest.  
**Fit to existing evidence:** highest.  
**Risk:** medium-high because the carrier and temporal controls must be designed carefully.  
**Recommendation:** primary next flagship after E17 and the calibration audit.

## Rank 2 - Emergence of `Q/A/G` during training

**Novelty:** medium-high after reformulation; low if framed as generic steerability emergence.  
**Fit:** high.  
**Risk:** medium; many checkpoints and cross-checkpoint calibration.  
**Recommendation:** follow-up study or parallel bounded pilot, not the main novelty claim.

## Rank 3 - External task/concept replication of the core `Q/A/G` mechanism

Promising testbeds include relational binding, contextual truth, communicative intent, and delayed binding. This is necessary for generality but less novel as a standalone paper.

**Recommendation:** include one external task inside E15 or as confirmation, rather than running an isolated replication campaign first.

## Rank 4 - A new causal-organization distillation objective

Direct prior art exists, and the current conversion-response branch failed its own falsification gates. A method project would require a new objective with component-balanced targets, sample-specific controls, and independent validation.

**Recommendation:** defer until E17 establishes the diagnostic phenomenon outside Qwen and the target metric is demonstrably well identified.

## Rank 5 - More quantization variants

Additional bit widths/backends would broaden E14 but would not resolve the key selectivity confound unless general quality is matched or causally controlled.

**Recommendation:** secondary robustness work, not the next flagship.

---

# 13. Decision tree for the immediate program

## Step A - Complete E17

### If E17 is positive

- State that behavioral/representational equivalence can coexist with causal-organizational mismatch in two model families.
- Use OLMo as a replication candidate for E15 only after the temporal task is frozen in Qwen.

### If E17 is negative

- Do not screen another family after seeing the causal outcome.
- Treat the negative replication as a boundary result.
- Continue E15 as a within-family mechanistic question; do not claim universality.

## Step B - E01 calibration audit

### If calibrated checkpoint differences persist

Retain a robust checkpoint/readout-conversion contrast, but avoid saying parameter count alone caused it.

### If calibrated differences weaken or vanish

Reframe the core paper around distributed actionability and transformation-sensitive causal organization. Those findings do not depend on a scale law.

## Step C - Bounded E15 pilot

Test five horizons, one carrier, one model, and all integrity gates. Do not begin with a large model sweep.

## Step D - Freeze discovery and confirmation

Only after the carrier, horizon set, estimands, residual-fraction grid, and full-patch sufficiency criterion are fixed should the full discovery and one-shot confirmation proceed.

---

# 14. Recommended publication framing

## Proposed umbrella thesis

> **Representations do not have a single functional status. They have a causal organization: a structured mapping from encoded variables through scalar, contextual, and interaction pathways into behavior. That organization can be preserved, weakened, substituted, or rebuilt independently of decodability and behavioral equivalence.**

## Present paper contribution

1. Near-matched semantic decodability with sharply different scalar conversion in two tested checkpoints.
2. Source-free scalar setpoint causality.
3. Confirmed additive probe-orthogonal causal signal.
4. Confirmed structured scalar-by-context interaction in Qwen3-1.7B.
5. Quantization that preserves native `D` more strongly than higher-order actionability, with general degradation caveat.
6. Distilled students that equal/exceed teacher behavior while retaining systematic `Q/A` mismatch at `D=1`.
7. E17 as cross-family boundary/replication, once completed.

## Next paper contribution

1. A temporal causal-organization profile `O(k)`.
2. A design that separates state age from post-intervention propagation distance.
3. Componentwise persistence/reorganization under persistent decodability.
4. A preregistered, full-patch-gated causal bottleneck.
5. Cross-task or cross-model confirmation only after the primary mechanism is frozen.

Possible title:

> **Decodable After It Stops Mattering: Temporal Reorganization of Causal Semantic Pathways in Language Models**

A more conservative title:

> **Temporal Persistence of Representation and Causal Organization in Language Models**

---

# 15. Short exact quotations from decisive papers

The quotations below are intentionally short; the analytical comparisons elsewhere are independent synthesis.

- Liang, Cheng, and Wajid: “decodability, generation, activation-level influence, and steerability can diverge in the tested setting.”
- Walsh and Barkett: “Behavioral sensitivity, latent readout, and causal control are three distinct properties that do not automatically co-occur.”
- Nadaf: “FV steering succeeds even when the logit lens cannot decode the correct answer at any layer.”
- Liao and Cao: “causal deployment, not decodability, is what interpretability should measure”.
- Vaidyanathan et al.: “INT is not a nuisance to be eliminated, but rather a diagnostic”.
- Ma et al.: “Identical language-model answers can arise from hidden states that support different future computations”.
- Basu et al.: “Current mechanistic interpretability methods cannot reliably translate internal knowledge into corrected outputs”.
- She et al.: “intervention efficacy, measured by linear steerability, emerges during intermediate stages of training.”
- Wang, Hong, and Bagci: “usable for selection, not steering”.

---

# 16. Bottom line

The project should stop competing for the broad claim that decoded information may be unused. That claim is established and crowded.

Its strongest existing contribution is the **confirmed, componentwise causal organization** of a semantic variable and the finding that this organization can differ despite preserved decodability or behavior.

The highest-value next move is therefore:

1. finish the frozen E17 cross-family test;
2. calibrate the cross-checkpoint intervention comparison;
3. launch a tightly controlled temporal study of `D/Q/A/G/P/B`;
4. treat E16 as a later study of causal-organization emergence, not generic steerability onset.

The project's durable research program should be framed as:

\[
\boxed{\text{When is causal organization preserved, when is it rebuilt, and when does it cease to control behavior?}}
\]

---

# References

## Novelty-threat papers

Bhardwaj, A., Duan, E. W., Dan, P., Ma, W.-C., & Culbertson, P. (2026). *Decoding task progress from VLA representations*. arXiv. https://doi.org/10.48550/arXiv.2608.13474

Bigoulaeva, I., Rohweder, J., Dutta, S., & Gurevych, I. (2026). *Patches of nonlinearity: Instruction vectors in large language models*. arXiv. https://doi.org/10.48550/arXiv.2602.07930

Cheng, Y., Fan, C., JafariRaviz, M., Rezaei, K., & Feizi, S. (2026). *Model-adaptive tool necessity reveals the knowing-doing gap in LLM tool use*. arXiv. https://doi.org/10.48550/arXiv.2605.14038

Chouman, H., Sasaki, W., Matsui, T., Suwa, H., & Yasumoto, K. (2026). *Representation as a bottleneck for mechanistic interpretability: The manifestation unit protocol*. arXiv. https://doi.org/10.48550/arXiv.2607.00089

Dai, Q., Heinzerling, B., & Inui, K. (2026). Cell-based representation of relational binding in language models. *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics*. https://doi.org/10.18653/v1/2026.acl-long.2194

Deng, Z., Ju, T., Wu, Z., He, L., Lan, J., Zhu, H., Wang, W., & Zhang, Z. (2026). *Causal probing for internal visual representations in multimodal large language models*. arXiv. https://doi.org/10.48550/arXiv.2605.05593

Ding, Z., & Zhang, X.-P. (2026). *Causal tongue-tie: LLMs can encode causal direction, but their Yes/No outputs fail to express it*. arXiv. https://doi.org/10.48550/arXiv.2605.25891

Doan, T., Le, U., & Nguyen, T. (2026). Causal activation steering via sparse mediation. *Findings of the Association for Computational Linguistics: EACL 2026*. https://doi.org/10.18653/v1/2026.findings-eacl.57

Dorrell, W., Latham, P. E., Behrens, T. E. J., & Whittington, J. C. R. (2023). *Actionable neural representations: Grid cells from minimal constraints*. International Conference on Learning Representations. https://doi.org/10.48550/arXiv.2209.15563

Ernst, M. S., Linhardt, L., Peikert, A., & Eberle, O. (2026). *Distributed sparse interventions in language models*. arXiv. https://doi.org/10.48550/arXiv.2607.07128

Galeone, C., Ettorre, A., Park, M., Ettorre, G., & Ligorio, D. (2026). *Perfect detection, failed control: The geometry of knowing vs. steering in language models*. arXiv. https://doi.org/10.48550/arXiv.2606.24952

Huang, L., & Chang, Y. (2025). *Causality != decodability, and vice versa: Lessons from interpreting counting ViTs*. arXiv. https://doi.org/10.48550/arXiv.2510.09794

Karim, A., Sheaib, F., Khamis, Z., Chlon, M., Awada, J., & Chlon, L. (2026). *Attention deficits in language models: Causal explanations for procedural hallucinations*. arXiv. https://doi.org/10.48550/arXiv.2602.19239

Kwon, A. (2026). *They infer what you meant: Models represent communicative intent more reliably than they act on it*. arXiv. https://doi.org/10.48550/arXiv.2607.03598

Liang, M., Cheng, X., & Wajid, F. (2026). *Encoded but not actionable: Auditing the decode-generate-steer gap in frozen LLMs for geometric constraints*. arXiv. https://doi.org/10.48550/arXiv.2608.17843

Liao, C.-T., & Cao, X. (2026). *The objective decides: When a learned dynamics model uses a conserved quantity*. arXiv. https://doi.org/10.48550/arXiv.2607.03728

Ma, S., Luo, Y., Zhangji, Xiao, C., Gao, A., Huang, W.-H., Wang, W., Wu, Q., Li, X., Wei, J., & Zhang, Q. (2026). *Hidden APIs in language models: Discovering reusable causal interfaces from forked futures*. arXiv. https://doi.org/10.48550/arXiv.2607.27617

Nadaf, M. S. B. (2026). *Steerable but not decodable: Function vectors operate beyond the logit lens*. arXiv. https://doi.org/10.48550/arXiv.2604.02608

Nguyen Quang, T., Gao, Y., Pu, F., Zhang, K., Sun, S., & Liu, Z. (2026). *Senses wide shut: A representation-action gap in omnimodal LLMs*. arXiv. https://doi.org/10.48550/arXiv.2605.13737

Sharma, A., Dawes, C., & Raval, S. (2026). *Dissociating decodability and causal use in bracket-sequence transformers*. arXiv. https://doi.org/10.48550/arXiv.2604.22128

Vaidyanathan, S., Arbour, D., Mueller, A., Niekum, S., & Jensen, D. (2026). *The curse of multiple mediators: Hidden interaction effects in activation patching*. arXiv. https://doi.org/10.48550/arXiv.2606.27510

Walsh, C., & Barkett, E. (2026). *Representation without control: Testing the realization effect in language models*. arXiv. https://doi.org/10.48550/arXiv.2605.25151

Xu, W. (2026). *Scale determines whether language models organize representation geometry for prediction*. arXiv. https://doi.org/10.48550/arXiv.2605.17084

## Additional decisive references

Akarlar, G. A. (2026). *Hallucination as trajectory commitment: Causal evidence for asymmetric attractor dynamics in transformer generation*. arXiv. https://doi.org/10.48550/arXiv.2604.15400

Basu, S., Patel, S. Y., Sheth, P., Muralidharan, B., Elamaran, N., Kinra, A., Morgan, J., & Batniji, R. (2026). *Interpretability without actionability: Mechanistic methods cannot correct language model errors despite near-perfect internal representations*. arXiv. https://doi.org/10.48550/arXiv.2603.18353

Canby, M. E., Davies, A., Rastogi, C., & Hockenmaier, J. (2024). *How reliable are causal probing interventions?* arXiv. https://doi.org/10.48550/arXiv.2408.15510

Dura, M., Ozturk, S., & Tekir, S. (2026). *Mechanistic interpretability of chain-of-thought reasoning via sequential activation patching*. arXiv. https://doi.org/10.48550/arXiv.2608.22332

Grant, S., Han, S. J., Tartaglini, A., & Potts, C. (2025). *Addressing divergent representations from causal interventions on neural networks*. arXiv. https://doi.org/10.48550/arXiv.2511.04638

Harrasse, A., Lan, M., Batra, H., Hashemi Chaleshtori, F., & Bandi, C. (2026). *Reasoning fine-tuning induces persistent latent policy states*. arXiv. https://doi.org/10.48550/arXiv.2607.18532

Pal, K., Sun, J., Yuan, A., Wallace, B. C., & Bau, D. (2023). *Future lens: Anticipating subsequent tokens from a single hidden state*. arXiv. https://doi.org/10.48550/arXiv.2311.04897

She, J., Li, X., Xing, E., Liu, Z., & Ho, Q. (2025). *How does controllability emerge in language models during pretraining?* arXiv. https://doi.org/10.48550/arXiv.2508.01892

Wang, Z., Hong, Z., & Bagci, U. (2026). *A decodability criterion predicts when hidden-state selection beats majority voting in large language models*. arXiv. https://doi.org/10.48550/arXiv.2608.17124

Wu, Y., Zhao, S., & Chen, J. (2026). *When is a steerable concept representation real? Measurement confounds in a cross-family audit of neuroscience parallels in LLMs*. arXiv. https://doi.org/10.48550/arXiv.2608.08159

Wu, Z., Geiger, A., Rozner, J., Kreiss, E., Lu, H., Icard, T., Potts, C., & Goodman, N. D. (2022). Causal distillation for language models. *Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 4288-4295. https://doi.org/10.18653/v1/2022.naacl-main.318
