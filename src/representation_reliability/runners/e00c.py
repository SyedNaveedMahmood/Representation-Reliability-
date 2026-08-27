"""E00-C - Representation Origin & Readout Bottleneck (non-causal).

Gates A-F evaluated against matched random-initialization controls,
LOFO family ablation, embedding-rung features, raw-vs-chat interface arms,
validation-only calibration, frozen-probe decodability on native behavior
errors, and an untuned fixed native readout diagnostic.
No causal intervention is performed in this phase.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..adapters.hf import HFAdapter
from ..config import config_hash, resolve_config, save_resolved_config
from ..data.base import samples_to_dataframe
from ..data.splits import (
    ConfirmationSplitAccessError,
    apply_splits,
    assign_group_splits,
    build_discovery_label_map,
    discovery_view,
    validate_splits,
)
from ..data.synthetic import generate_synthetic_relations
from ..extraction.activations import (
    build_cache_identity,
    extract_dataset_activations,
)
from ..extraction.cache import ActivationCacheReader, completed_shard_ids
from ..metrics.bootstrap import bootstrap_ci
from ..metrics.decoding import (
    class_balance,
    classification_metrics,
    majority_baseline_accuracy,
)
from ..probes.linear import (
    fit_probe,
    randomized_control_labels,
    raw_probe_direction,
    transform_features,
)
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash
from ..runtime.run_id import allocate_run_dir, make_run_id
from ..runtime.status import StatusFile
from .e00 import _auroc_fn, _confirmation_ids_digest, _strip_confirmation_labels
from .extract import load_adapter
from .probe import ShardedMatrixLoader, design_matrices_for, text_baseline_metrics

logger = logging.getLogger(__name__)

LEARNING_SPLITS = ("train", "validation", "discovery_test")


def add_result_metadata(
    frame: pd.DataFrame,
    *,
    model_id: str,
    resolved_revision: str | None,
    training_stage: str,
    prompt_interface: str,
    chat_template_used: bool,
    thinking_enabled: str,
    candidates,
    candidate_token_ids,
    provenance_note: str | None = None,
) -> pd.DataFrame:
    """Attach mandatory interface/model provenance to every result row."""
    out = frame.copy()
    out["model_id"] = model_id
    out["resolved_revision"] = resolved_revision
    out["training_stage"] = training_stage
    out["prompt_interface"] = prompt_interface
    out["chat_template_used"] = bool(chat_template_used)
    out["thinking_enabled"] = thinking_enabled
    out["candidate_verbalizers"] = json.dumps(list(candidates))
    out["candidate_token_ids"] = json.dumps(candidate_token_ids)
    out["model_provenance_note"] = provenance_note
    return out


def _generate_dataset(cfg):
    samples = generate_synthetic_relations(
        n_samples=int(cfg.dataset.n_samples),
        seed=int(cfg.reproducibility.data_seed),
        n_entities=int(cfg.dataset.n_entities),
    )
    df = samples_to_dataframe(samples)
    assignment = assign_group_splits(df["pair_id"].tolist(),
                                     seed=int(cfg.reproducibility.split_seed))
    df = apply_splits(df, assignment)
    validate_splits(df)
    return samples, df


def lofo_family_split(discovery_df: pd.DataFrame,
                      held_out_family: str) -> dict:
    rel = discovery_df["relation"].astype(str)
    fams = set(rel.unique())
    if held_out_family not in fams:
        raise ValueError(f"unknown family {held_out_family!r}")
    other = discovery_df[rel.ne(held_out_family)]
    held = discovery_df[rel.eq(held_out_family)]
    tr_val = other[other["split"].isin(("train", "validation"))]
    val_only = tr_val[tr_val["split"] == "validation"]
    eval_rows = held[held["split"] == "discovery_test"]
    if len(val_only) == 0 or len(eval_rows) == 0:
        raise RuntimeError("LOFO degenerate split for " + held_out_family)
    return {
        "train_ids": tr_val.loc[tr_val["split"] == "train",
                                "sample_id"].tolist(),
        "validation_ids": val_only["sample_id"].tolist(),
        "evaluation_ids": eval_rows["sample_id"].tolist(),
    }


def select_threshold(y_val: np.ndarray, margins_val: np.ndarray) -> float:
    """Validation-only threshold maximizing balanced accuracy."""
    y = np.asarray(y_val).astype(int)
    uniq = np.unique(margins_val)
    mids = (uniq[:-1] + uniq[1:]) / 2 if len(uniq) > 1 else uniq
    grid = np.concatenate(([uniq.min() - 1.0], mids, [uniq.max() + 1.0]))
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    best_tau, best = 0.0, -np.inf
    for tau in grid:
        pred = (margins_val >= tau).astype(int)
        tpr = ((pred == 1) & (y == 1)).sum() / max(pos, 1)
        tnr = ((pred == 0) & (y == 0)).sum() / max(neg, 1)
        s = 0.5 * (tpr + tnr)
        if s > best:
            best, best_tau = s, float(tau)
    return best_tau


def fit_1d_calibration(y_fit, margins_fit):
    X = np.asarray(margins_fit, dtype=np.float64).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(C=1e6, max_iter=1000)
    clf.fit(X, np.asarray(y_fit).astype(int))
    a = float(clf.coef_[0][0])
    b = float(clf.intercept_[0])

    def predict(m):
        z = a * np.asarray(m, dtype=np.float64) + b
        return (1.0 / (1.0 + np.exp(-z)) >= 0.5).astype(int)

    return predict, {"slope_a": a, "intercept_b": b}


def build_chat_prompts(adapter: HFAdapter, sample_ids, raw_prompts,
                       instruction: str = "Answer only Yes or No."):
    """Qwen official non-thinking chat interface via its own template."""
    if len(sample_ids) != len(raw_prompts):
        raise ValueError("chat prompt sample-id/prompt length mismatch")
    tok = adapter.tokenizer
    if not getattr(tok, "chat_template", None):
        raise ValueError("chat arm requires a tokenizer chat template")
    out = []
    for sid, raw in zip(sample_ids, raw_prompts):
        premise_l = next(l for l in raw.splitlines()
                         if l.startswith("Premise:"))
        question_l = next(l for l in raw.splitlines()
                          if l.startswith("Question:"))
        messages = [{"role": "user",
                     "content": f"{premise_l}\n{question_l}\n{instruction}"}]
        try:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        out.append(str(text))
    return out


def build_chat_samples(adapter: HFAdapter, samples):
    """Create a semantically paired chat arm with explicit interface metadata."""
    import dataclasses

    sample_ids = [s.sample_id for s in samples]
    prompts = build_chat_prompts(adapter, sample_ids, [s.prompt for s in samples])
    return [
        dataclasses.replace(
            sample,
            prompt=prompt,
            metadata={**sample.metadata, "chat_template_used": True,
                      "thinking_enabled": False,
                      "prompt_interface": "qwen_chat_nonthinking"},
        )
        for sample, prompt in zip(samples, prompts)
    ]


def arm_content_hash(sample_ids, prompts) -> str:
    if len(sample_ids) != len(prompts):
        raise ValueError("content hash sample-id/prompt length mismatch")
    h = hashlib.sha256()
    for sid, pr in zip(sample_ids, prompts):
        h.update(f"{sid}\x00{pr}\x1e".encode())
    return h.hexdigest()


def candidate_token_id_lists(adapter: HFAdapter, candidates) -> list[list[int]]:
    ids: list[list[int]] = []
    for cand in candidates:
        cont = cand if cand.startswith(" ") else " " + cand
        enc = adapter.tokenizer(cont, add_special_tokens=False)["input_ids"]
        ids.append([int(token_id) for token_id in enc])
    return ids


def candidate_first_token_ids(adapter: HFAdapter, candidates) -> list[int]:
    return [ids[0] for ids in candidate_token_id_lists(adapter, candidates)]


def mask_cell(index_df, site, selector, layer):
    return ((index_df["site"] == site)
            & (index_df["token_selector"] == selector)
            & (index_df["layer"] == layer))


def logit_lens_layer_margins(adapter: HFAdapter, cache_dir, index_df,
                             yes_token_id: int, no_token_id: int, label_of):
    """Untuned fixed readout over cached resid_post (final_norm+lm_head)."""
    rows = []
    evidence: dict[tuple[str, str, int], dict[str, Any]] = {}
    loader = ShardedMatrixLoader(cache_dir, index_df)
    for key in sorted(set(zip(index_df["site"], index_df["token_selector"],
                              index_df["layer"]))):
        site, selector, layer = key
        eval_mask = (mask_cell(index_df, site, selector, layer)
                     & index_df["split"].eq("discovery_test"))
        X, meta = loader.rows_for(eval_mask)
        token_logits = adapter.final_readout_token_logits(
            X, [yes_token_id, no_token_id])
        margins = (token_logits[:, 0] - token_logits[:, 1]).astype(np.float64)
        y = np.asarray([int(label_of(sid)) for sid in meta["sample_id"]],
                       dtype=int)
        m = classification_metrics(y, margins)
        rows.append({"site": site, "token_selector": selector,
                     "layer": int(layer),
                     "native_margin_auroc": m.get("auroc"),
                     "native_margin_auprc": m.get("auprc"),
                     "n_eval": len(y)})
        evidence[(str(site), str(selector), int(layer))] = {
            "sample_ids": meta["sample_id"].astype(str).tolist(),
            "y": y,
            "scores": margins,
        }
    return pd.DataFrame(rows), evidence


def native_readout_direction(adapter: HFAdapter, yes_id: int, no_id: int):
    """Effective Yes-No direction in raw residual coordinates.

    lm_head(RMSNorm(h)) divides by the positive scalar rms(h), so the native
    Yes-No logit-difference ranking equals <gamma * (W_yes - W_no), h>.
    Assumptions: single linear LM head without bias (Qwen3), first-token
    comparison. See DIAGNOSIS_PHASE_0A2.md section 8.
    """
    return adapter.native_first_token_direction(yes_id, no_id)


def probe_all_cells(loader, index_df, label_of, expected_ids_by_split, *,
                    sites, selectors, layers, c_grid, seed,
                    n_bootstraps=500, confidence_level=0.95,
                    val_randomized=False):
    """Standard truth-probe sweep; returns (metrics_df, per-cell score data)."""
    rows: list[dict] = []
    scores: dict[tuple[str, str, int], dict[str, Any]] = {}
    for site in sites:
        for selector in selectors:
            for layer in layers:
                mats = design_matrices_for(
                    loader, index_df, selector, layer, site,
                    label_of=label_of,
                    expected_ids_by_split=expected_ids_by_split,
                )
                if any(s not in mats for s in LEARNING_SPLITS):
                    continue
                y_tr = mats["train"]["y"]
                y_val = mats["validation"]["y"]
                extra: dict[str, Any] = {}
                if val_randomized:
                    y_tr, y_val, perm_seeds = randomized_control_labels(
                        y_tr, y_val, seed=seed)
                    extra.update(train_perm_seed=perm_seeds[0],
                                 validation_perm_seed=perm_seeds[1])
                fit = fit_probe(mats["train"]["X"], y_tr,
                                mats["validation"]["X"], y_val,
                                c_grid=c_grid, seed=seed,
                                standardize=True, class_weight="balanced")
                yte = mats["discovery_test"]["y"]
                sc = fit["classifier"].decision_function(
                    transform_features(fit, mats["discovery_test"]["X"]))
                m = classification_metrics(yte, sc)
                ci = (bootstrap_ci(
                    yte, sc, metric_fn=_auroc_fn,
                    n_bootstraps=n_bootstraps,
                    confidence_level=confidence_level, seed=seed)
                    if n_bootstraps > 0 else {})
                key = (site, selector, int(layer))
                rows.append({"site": site, "token_selector": selector,
                             "layer": int(layer), **m, **ci, **extra,
                             "chosen_C": fit["chosen_C"],
                             "majority_accuracy_discovery_test":
                                 majority_baseline_accuracy(yte),
                             "class_balance": class_balance(yte)})
                scores[key] = {"scores": np.asarray(sc, dtype=np.float64),
                               "y": np.asarray(yte),
                               "sample_ids": list(
                                   mats["discovery_test"]["sample_ids"]),
                               "fit": fit}
    return pd.DataFrame(rows), scores


def _align_scores(reference_ids, block):
    ref_pos = {sid: i for i, sid in enumerate(reference_ids)}
    ids = block["sample_ids"]
    if ids == reference_ids:
        return block["scores"], block["y"]
    perm = [ref_pos[s] for s in ids]
    inv = np.argsort(perm)
    return (np.asarray(block["scores"])[inv],
            np.asarray(block["y"])[inv])


def paired_bootstrap_delta(scores_pre, scores_rand_list, y,
                           n_bootstraps=500, seed=0):
    """Paired bootstrap of D_pre - mean_r(D_rand) on identical examples."""
    rng = np.random.default_rng(seed)
    n = len(y)
    point = _auroc_fn(y, scores_pre) - float(np.mean(
        [_auroc_fn(y, s) for s in scores_rand_list]))
    stats = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y[idx])) < 2:
            continue
        d = (_auroc_fn(y[idx], scores_pre[idx]) - float(np.mean(
            [_auroc_fn(y[idx], s[idx]) for s in scores_rand_list])))
        stats.append(d)
    lo, hi = (np.percentile(stats, [2.5, 97.5]) if stats
              else (float("nan"), float("nan")))
    return {"delta_D": float(point), "ci_low": float(lo), "ci_high": float(hi)}


def paired_bootstrap_accuracy_delta(
    y,
    pred_reference,
    pred_comparison,
    *,
    n_bootstraps: int = 1000,
    seed: int = 0,
):
    """Paired semantic-example bootstrap for an accuracy difference."""
    y = np.asarray(y, dtype=int)
    a = np.asarray(pred_reference, dtype=int)
    b = np.asarray(pred_comparison, dtype=int)
    if not (len(y) == len(a) == len(b)):
        raise ValueError("paired bootstrap arrays must have equal length")
    point = float((b == y).mean() - (a == y).mean())
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y), size=len(y))
        stats.append(float((b[idx] == y[idx]).mean()
                           - (a[idx] == y[idx]).mean()))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"delta_accuracy": point,
            "ci_low": float(lo), "ci_high": float(hi)}


def bootstrap_accuracy_interval(
    y,
    pred,
    *,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 0,
):
    y = np.asarray(y, dtype=int)
    pred = np.asarray(pred, dtype=int)
    if len(y) != len(pred):
        raise ValueError("accuracy bootstrap arrays must have equal length")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_bootstraps):
        idx = rng.integers(0, len(y), size=len(y))
        stats.append(float((pred[idx] == y[idx]).mean()))
    alpha = (1.0 - confidence_level) / 2.0
    lo, hi = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return {"point_estimate": float((pred == y).mean()),
            "ci_low": float(lo), "ci_high": float(hi)}


def _lofo_cell(loader_root, idx_full, selector, layer, site, plan,
               fam_label_map, *, c_grid, seed):
    """LOFO probe; held-out family appears ONLY at evaluation."""
    from sklearn.metrics import balanced_accuracy_score

    allowed = set(plan["train_ids"]) | set(plan["validation_ids"])
    eval_ids = set(plan["evaluation_ids"])
    idx_tv = idx_full[idx_full["sample_id"].isin(allowed)].reset_index(drop=True)
    loader_tv = ShardedMatrixLoader(loader_root, idx_tv)
    mats = design_matrices_for(
        loader_tv, idx_tv,
        selector, layer, site, label_of=lambda s: fam_label_map[s],
        expected_ids_by_split={
            "train": plan["train_ids"],
            "validation": plan["validation_ids"],
        })
    fit = fit_probe(mats["train"]["X"], mats["train"]["y"],
                    mats["validation"]["X"], mats["validation"]["y"],
                    c_grid=c_grid, seed=seed, standardize=True,
                    class_weight="balanced")
    sub = idx_full[(idx_full["sample_id"].isin(eval_ids))
                   & (idx_full["site"] == site)
                   & (idx_full["token_selector"] == selector)
                   & (idx_full["layer"] == layer)].reset_index(drop=True)
    Xe, meta_e = ShardedMatrixLoader(loader_root, sub).rows_for(
        mask_cell(sub, site, selector, layer))
    if len(meta_e) == 0:
        raise RuntimeError("LOFO evaluation subset empty")
    y_e = np.asarray([int(fam_label_map[s]) for s in meta_e["sample_id"]],
                     dtype=int)
    sc = fit["classifier"].decision_function(transform_features(fit, Xe))
    m = classification_metrics(y_e, sc)
    pred = (sc >= 0).astype(int)
    metrics = {"selector": selector, "layer": int(layer),
               "auroc": m.get("auroc"), "auprc": m.get("auprc"),
               "balanced_accuracy": float(balanced_accuracy_score(y_e, pred)),
               "n_eval": len(y_e), "chosen_C": fit["chosen_C"],
               "n_train": len(mats["train"]["y"])}
    evidence = {"sample_ids": meta_e["sample_id"].astype(str).tolist(),
                "y": y_e, "scores": np.asarray(sc, dtype=np.float64)}
    return metrics, evidence


def balanced_accuracy_score_global(y_true, y_pred) -> float:
    from sklearn.metrics import balanced_accuracy_score

    return float(balanced_accuracy_score(np.asarray(y_true).astype(int),
                                         np.asarray(y_pred)))


def probe_score_evidence_frame(arm: str, blocks: dict) -> pd.DataFrame:
    """Flatten per-example frozen probe scores before aggregate reporting."""
    rows: list[dict[str, Any]] = []
    for (site, selector, layer), block in blocks.items():
        for sample_id, gold, score in zip(
                block["sample_ids"], block["y"], block["scores"]):
            rows.append({"arm": arm, "site": site, "selector": selector,
                         "layer": int(layer), "sample_id": str(sample_id),
                         "gold": int(gold), "probe_score": float(score)})
    return pd.DataFrame(rows)


def frozen_probe_subset_metrics(
    probe_blocks: dict,
    behavior_by_id: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Partition already-frozen discovery scores; never trains a new probe."""
    rows: list[dict[str, Any]] = []
    for (site, selector, layer), block in probe_blocks.items():
        ids = [str(s) for s in block["sample_ids"]]
        correct = np.asarray([bool(behavior_by_id[s]["correct"]) for s in ids])
        y = np.asarray(block["y"], dtype=int)
        scores = np.asarray(block["scores"], dtype=np.float64)
        for subset_name, subset_mask in (("errors", ~correct),
                                         ("correct", correct)):
            n = int(subset_mask.sum())
            single_class = n == 0 or len(np.unique(y[subset_mask])) < 2
            row: dict[str, Any] = {
                "site": site, "selector": selector, "layer": int(layer),
                "subset": subset_name, "n": n,
                "label_balance": float(y[subset_mask].mean()) if n else None,
                "auroc": None, "auprc": None, "balanced_accuracy": None,
                "score_mean": float(scores[subset_mask].mean()) if n else None,
                "score_std": float(scores[subset_mask].std()) if n else None,
                "score_q05": (float(np.quantile(scores[subset_mask], 0.05))
                              if n else None),
                "score_q50": (float(np.quantile(scores[subset_mask], 0.50))
                              if n else None),
                "score_q95": (float(np.quantile(scores[subset_mask], 0.95))
                              if n else None),
            }
            if single_class:
                row["note"] = "empty" if n == 0 else "single-class; AUROC invalid"
            else:
                metrics = classification_metrics(y[subset_mask],
                                                 scores[subset_mask])
                row.update({
                    "auroc": metrics.get("auroc"),
                    "auprc": metrics.get("auprc"),
                    "balanced_accuracy": balanced_accuracy_score_global(
                        y[subset_mask], (scores[subset_mask] >= 0).astype(int)),
                })
            rows.append(row)
    return pd.DataFrame(rows)


def _make_random_adapter(cfg, seed: int) -> HFAdapter:
    torch.manual_seed(int(seed))
    adapter = HFAdapter(cfg.model, cfg.runtime)
    adapter.configure_random_init(seed)
    return adapter.load()


def _run_arm_extraction(adapter, arm_samples, cache_dir, identity, cfg,
                        split_of):
    return extract_dataset_activations(
        adapter, arm_samples, cache_dir,
        sites=list(cfg.representation.sites),
        layers=list(range(adapter.num_layers)),
        token_selectors=list(cfg.representation.token_selectors),
        shard_size=int(cfg.storage.shard_size),
        batch_size=int(cfg.runtime.batch_size),
        split_of=split_of, identity=identity,
        resume=bool(cfg.runtime.resume))


def run_e00c(base_path=None, model_path=None, experiment_path=None,
             overrides: tuple[str, ...] = ()) -> Path:
    """Execute the full E00-C diagnostic suite; returns the run directory."""
    repo_root = Path(__file__).resolve().parents[3]
    if experiment_path is None:
        experiment_path = (
            repo_root / "configs" / "experiments"
            / "E00C_readout_diagnosis.yaml"
        )
    cfg, provenance = resolve_config(base_path=base_path,
                                     model_path=model_path,
                                     experiment_path=experiment_path,
                                     overrides=overrides)
    c_hash = config_hash(cfg)
    samples, df = _generate_dataset(cfg)
    split_hash = dataset_split_hash(
        dict(zip(df["sample_id"], df["split"])))
    run_id = make_run_id(cfg.experiment.id, c_hash,
                         int(cfg.reproducibility.seed),
                         cfg.model.revision or "unpinned", split_hash)
    run_dir = allocate_run_dir(repo_root / cfg.project.output_root,
                               cfg.experiment.id, run_id)
    for subdir in ("figures", "controls"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)
    save_resolved_config(cfg, run_dir / "config.resolved.yaml", provenance)
    status = StatusFile.create(run_dir, run_id, cfg.experiment.id)
    manifest = RunManifest(run_dir)
    manifest.set_start(c_hash, provenance, cfg.effective_seeds())

    results: dict[str, Any] = {"interface_metadata": {
        "model_id_raw": cfg.model.id,
        "candidates_primary": list(cfg.behavior.candidates_primary),
        "raw_interface": "completion Premise/Question/Answer:",
        "chat_template": bool(cfg.diagnostics.chat_arm),
        "thinking_enabled": False,
        "chat_mode": "tokenizer.apply_chat_template(enable_thinking=False)",
    }}
    try:
        discovery_df = discovery_view(df).reset_index(drop=True)
        label_of = build_discovery_label_map(discovery_df)
        expected_ids_by_split = {
            s: g["sample_id"].tolist()
            for s, g in discovery_df.groupby("split")}
        split_map = dict(zip(discovery_df["sample_id"],
                             discovery_df["split"]))
        ids_all = discovery_df["sample_id"].tolist()
        gold_all = np.asarray([int(label_of[s]) for s in ids_all], dtype=int)
        manifest.update_dataset_info(
            n_samples=len(df), split_hash=split_hash,
            split_summary={
                split: int((df["split"] == split).sum())
                for split in df["split"].unique()
            },
            dataset_type=cfg.dataset.type,
            confirmation_labels_observed=False,
            confirmation_sample_ids_sha256=_confirmation_ids_digest(df),
        )
        _strip_confirmation_labels(df).to_parquet(
            run_dir / "samples.parquet", index=False)

        def _split_of(sid):
            if sid not in split_map:
                raise ConfirmationSplitAccessError(f"{sid} outside discovery")
            return str(split_map[sid])

        discovery_ids = set(discovery_df["sample_id"])
        arm_samples = [s for s in samples if s.sample_id in discovery_ids]
        if [s.sample_id for s in arm_samples] != ids_all:
            raise RuntimeError("semantic sample order changed before arm construction")
        families = sorted(set(discovery_df["relation"].astype(str)))
        yes_text, no_text = cfg.behavior.candidates_primary
        selectors = list(cfg.representation.token_selectors)
        sites = list(cfg.representation.sites)
        c_grid = [float(c) for c in cfg.probe.C_grid]
        seed_p = int(cfg.reproducibility.probe_seed)
        n_bootstraps = int(cfg.statistics.bootstrap_samples)
        confidence_level = float(cfg.statistics.confidence_level)
        cache_root = repo_root / cfg.project.cache_root / "activations"
        raw_prompts = discovery_df["prompt"].tolist()

        def _identity_for(adapter, chat, layers_list):
            revs = adapter.resolved_revisions()
            chosen = build_chat_samples(adapter, arm_samples) if chat else arm_samples
            tok_sets = {"padding_side": "right",
                        "add_special_tokens": True,
                        "chat_template": bool(chat),
                        "thinking_enabled": False}
            return build_cache_identity(
                experiment_id=(f"{cfg.experiment.id}_chat" if chat
                               else cfg.experiment.id),
                samples=chosen,
                model_id=adapter.display_model_id,
                model_resolved_revision=revs.get("model_sha"),
                tokenizer_id=cfg.model.id,
                tokenizer_resolved_revision=revs.get("tokenizer_sha"),
                sites=sites, layers=layers_list,
                token_selectors=selectors, model_dtype=cfg.model.dtype,
                tokenization=tok_sets), chosen

        # ---- pretrained: load, extract both interfaces --------------------
        status.update(progress={"stage": "pretrained"})
        pre = load_adapter(cfg)
        pre_revs = pre.resolved_revisions()
        layers = list(range(pre.num_layers))
        manifest.update_model_info(
            id=cfg.model.id, revision=cfg.model.revision,
            dtype=cfg.model.dtype, tokenizer_id=cfg.model.id,
            resolved_revision=pre_revs.get("model_sha"),
            tokenizer_revision=pre_revs.get("tokenizer_sha"),
            num_layers=pre.num_layers, hidden_size=pre.hidden_size,
            notes={"revision_resolution": pre_revs,
                   "training_stage": "post-trained/chat"},
        )
        results["interface_metadata"]["model_resolved_revision"] = \
            pre_revs.get("model_sha")
        candidate_ids = candidate_token_id_lists(pre, [yes_text, no_text])
        yes_id, no_id = candidate_ids[0][0], candidate_ids[1][0]
        results["interface_metadata"].update({
            "training_stage": "post-trained/chat",
            "candidate_token_ids": {
                "yes": candidate_ids[0], "no": candidate_ids[1]},
            "candidate_first_token_ids": {"yes": yes_id, "no": no_id},
        })

        def _pretrained_metadata(frame: pd.DataFrame, *, chat: bool = False):
            return add_result_metadata(
                frame, model_id=pre.display_model_id,
                resolved_revision=pre_revs.get("model_sha"),
                training_stage="post-trained/chat",
                prompt_interface=("qwen_chat_nonthinking" if chat
                                  else "raw_completion"),
                chat_template_used=chat,
                thinking_enabled="false" if chat else "not-applicable",
                candidates=[yes_text, no_text],
                candidate_token_ids=candidate_ids,
                provenance_note=pre_revs.get("resolution_note"),
            )

        chat_prompts = None
        if cfg.diagnostics.chat_arm:
            chat_prompts = [s.prompt for s in build_chat_samples(pre, arm_samples)]
            results["interface_metadata"].update({
                "chat_fingerprint_sha256":
                    arm_content_hash(ids_all, chat_prompts),
                "chat_prompt_example": chat_prompts[0][:500],
            })

        status.update(progress={"stage": "extract_raw"})
        raw_dir = cache_root / f"{cfg.experiment.id}_v2_{c_hash[:8]}_raw"
        idx_raw, _info = _run_arm_extraction(
            pre, arm_samples, raw_dir,
            _identity_for(pre, False, layers)[0], cfg, _split_of)
        save_table(idx_raw, run_dir / "activation_index_raw.parquet")

        chat_dir = None
        if cfg.diagnostics.chat_arm:
            status.update(progress={"stage": "extract_chat"})
            ident_chat, chat_samples = _identity_for(pre, True, layers)
            chat_dir = cache_root / f"{cfg.experiment.id}_v2_{c_hash[:8]}_chat"
            idx_chat_extracted, _ = _run_arm_extraction(
                pre, chat_samples, chat_dir, ident_chat, cfg, _split_of)
            save_table(idx_chat_extracted,
                       run_dir / "activation_index_chat.parquet")


        # ---- behavior margins (raw + chat), pretrained only ---------------
        status.update(progress={"stage": "behavior"})
        def _score_behavior(adapter, prompts_desc, *, prompt_interface,
                            chat_template_used):
            out: dict[str, dict] = {}
            bs_eff = max(8, int(cfg.runtime.batch_size) * 4)
            for st in range(0, len(ids_all), bs_eff):
                chunk = prompts_desc[st : st + bs_eff]
                pairs = adapter.score_continuations(chunk,
                                                    [yes_text, no_text])
                for j, sid in enumerate(ids_all[st : st + bs_eff]):
                    py_, pn_ = pairs[j][0], pairs[j][1]
                    margin_t = float(py_["logp_total"] - pn_["logp_total"])
                    gold = int(label_of[sid])
                    out[str(sid)] = {
                        "margin_total": margin_t,
                        "margin_mean": float(py_["logp_mean"]
                                             - pn_["logp_mean"]),
                        "pred": int(margin_t >= 0), "gold": gold,
                        "correct": int((margin_t >= 0) == bool(gold)),
                        "yes_token_ids": list(py_["token_ids"]),
                        "no_token_ids": list(pn_["token_ids"]),
                        "model_id": adapter.display_model_id,
                        "resolved_revision": pre_revs.get("model_sha"),
                        "training_stage": "post-trained/chat",
                        "prompt_interface": prompt_interface,
                        "chat_template_used": bool(chat_template_used),
                        "thinking_enabled": "false",
                        "candidate_verbalizers": json.dumps(
                            [yes_text, no_text]),
                    }
            return out

        def _behavior_summary(margins_by_id):
            disc_ids = [s for s in ids_all if split_map[s] == "discovery_test"]
            y = np.asarray([int(margins_by_id[s]["gold"]) for s in disc_ids])
            mvec = np.asarray([margins_by_id[s]["margin_total"]
                               for s in disc_ids])
            pred = np.asarray([margins_by_id[s]["pred"] for s in disc_ids])

            rng = np.random.default_rng(0)
            accs = []
            for _ in range(n_bootstraps):
                i_ = rng.integers(0, len(y), size=len(y))
                accs.append(float(((mvec[i_] >= 0).astype(int)
                                   == y[i_]).mean()))
            alpha = (1.0 - confidence_level) / 2.0
            lo, hi = np.percentile(accs, [100 * alpha, 100 * (1 - alpha)])
            return {
                "n": len(y),
                "threshold0_accuracy": float(((pred == y)).mean()),
                "accuracy_ci_95pct": [float(lo), float(hi)],
                "threshold0_balanced_accuracy":
                    float(balanced_accuracy_score_global(y, pred)),
                **classification_metrics(y, mvec),
            }

        marg_raw = _score_behavior(
            pre, raw_prompts, prompt_interface="raw_completion",
            chat_template_used=False)
        results["behavior_raw"] = _behavior_summary(marg_raw)
        results["behavior_raw"]["interface"] = "raw_completion"
        pd.DataFrame([{"sample_id": s, **v} for s, v in marg_raw.items()]
                     ).to_parquet(run_dir / "behavior_predictions.parquet",
                                  index=False)

        marg_chat = None
        if cfg.diagnostics.chat_arm and chat_prompts is not None:
            marg_chat = _score_behavior(
                pre, chat_prompts, prompt_interface="qwen_chat_nonthinking",
                chat_template_used=True)
            results["behavior_chat_nonthinking"] = \
                _behavior_summary(marg_chat)
            results["behavior_chat_nonthinking"]["interface"] = \
                "qwen_chat_nonthinking"
            pd.DataFrame([{"sample_id": s, **v}
                          for s, v in marg_chat.items()]).to_parquet(
                run_dir / "behavior_predictions_chat.parquet", index=False)
            disc_pair_ids = [s for s in ids_all
                             if split_map[s] == "discovery_test"]
            y_pair = np.asarray([marg_raw[s]["gold"] for s in disc_pair_ids])
            pred_raw_pair = np.asarray([marg_raw[s]["pred"]
                                        for s in disc_pair_ids])
            pred_chat_pair = np.asarray([marg_chat[s]["pred"]
                                         for s in disc_pair_ids])
            results["behavior_chat_vs_raw_paired"] = \
                paired_bootstrap_accuracy_delta(
                    y_pair, pred_raw_pair, pred_chat_pair,
                    n_bootstraps=int(cfg.statistics.bootstrap_samples),
                    seed=seed_p)
        save_json({"behavior_raw": results["behavior_raw"],
                   "behavior_chat_nonthinking":
                       results.get("behavior_chat_nonthinking")},
                  run_dir / "behavior_metrics.json")

        # ---- pretrained truth probes (raw + chat interfaces) --------------
        status.update(progress={"stage": "probe_pretrained"})
        loader_raw = ShardedMatrixLoader(raw_dir, idx_raw)
        mdf_pre, sc_pre = probe_all_cells(
            loader_raw, idx_raw, lambda s: int(label_of[s]),
            expected_ids_by_split, sites=sites, selectors=selectors,
            layers=layers, c_grid=c_grid, seed=seed_p,
            n_bootstraps=n_bootstraps, confidence_level=confidence_level)
        save_table(_pretrained_metadata(
            probe_score_evidence_frame("pretrained_raw", sc_pre)),
                   run_dir / "origin_probe_scores_pretrained.parquet")
        save_table(_pretrained_metadata(mdf_pre),
                   run_dir / "origin_probe_metrics_pretrained.parquet")
        results["D_raw_best"] = {
            f"{site}/{selector}": float(group["auroc"].max())
            for (site, selector), group in
            mdf_pre.groupby(["site", "token_selector"])
        }

        mdf_chat = None
        if chat_dir is not None:
            idx_chat = ActivationCacheReader(chat_dir).index()
            loader_chat = ShardedMatrixLoader(chat_dir, idx_chat)
            mdf_chat, sc_chat = probe_all_cells(
                loader_chat, idx_chat, lambda s: int(label_of[s]),
                expected_ids_by_split, sites=sites, selectors=selectors,
                layers=layers, c_grid=c_grid, seed=seed_p,
                n_bootstraps=n_bootstraps,
                confidence_level=confidence_level)
            save_table(_pretrained_metadata(
                probe_score_evidence_frame("pretrained_chat", sc_chat),
                chat=True),
                       run_dir / "interface_probe_scores_chat.parquet")
            save_table(_pretrained_metadata(mdf_chat, chat=True),
                       run_dir / "interface_probe_metrics_chat.parquet")
            results["D_chat_best"] = {
                f"{s_}/{sel_}": float(sub["auroc"].max())
                for (s_, sel_), sub in
                mdf_chat.groupby(["site", "token_selector"])}
            interface_probe_rows = []
            for key, raw_block in sc_pre.items():
                if key not in sc_chat:
                    continue
                chat_scores, chat_y = _align_scores(
                    raw_block["sample_ids"], sc_chat[key])
                raw_y = np.asarray(raw_block["y"])
                if not np.array_equal(chat_y, raw_y):
                    raise RuntimeError("raw/chat probe label pairing mismatch")
                d_raw = float(_auroc_fn(raw_y, raw_block["scores"]))
                d_chat = float(_auroc_fn(raw_y, chat_scores))
                interface_probe_rows.append({
                    "site": key[0], "selector": key[1], "layer": key[2],
                    "D_raw": d_raw, "D_chat_nonthinking": d_chat,
                    "delta_D": d_chat - d_raw,
                })
            save_table(_pretrained_metadata(pd.DataFrame(interface_probe_rows),
                                            chat=True),
                       run_dir / "interface_probe_delta.parquet")

        # ---- mandatory probe controls -------------------------------------
        status.update(progress={"stage": "probe_controls"})
        if cfg.controls.random_labels:
            control_metrics = []
            control_scores = []
            for control_index, control_seed in enumerate(
                    cfg.controls.random_label_seeds):
                null_seed = (int(cfg.reproducibility.control_seed)
                             + 1000 * int(control_seed))
                null_metrics, null_blocks = probe_all_cells(
                    loader_raw, idx_raw, lambda s: int(label_of[s]),
                    expected_ids_by_split, sites=sites, selectors=selectors,
                    layers=layers, c_grid=c_grid, seed=null_seed,
                    n_bootstraps=0, confidence_level=confidence_level,
                    val_randomized=True)
                null_metrics["control"] = "random_labels"
                null_metrics["control_index"] = control_index
                null_metrics["control_seed"] = int(control_seed)
                null_scores = probe_score_evidence_frame(
                    f"random_labels_seed{control_seed}", null_blocks)
                null_scores["control_index"] = control_index
                null_scores["control_seed"] = int(control_seed)
                control_metrics.append(null_metrics)
                control_scores.append(null_scores)
            save_table(_pretrained_metadata(pd.concat(control_scores,
                                                       ignore_index=True)),
                       run_dir / "controls" / "random_label_scores.parquet")
            save_table(_pretrained_metadata(pd.concat(control_metrics,
                                                        ignore_index=True)),
                       run_dir / "controls" / "random_label_metrics.parquet")

        if cfg.controls.text_baseline:
            def _text_baseline(prompts_for_ids, interface_name, chat_used):
                prompt_of = dict(zip(ids_all, prompts_for_ids))
                texts_by_split: dict[str, list[str]] = {}
                y_by_split: dict[str, list[int]] = {}
                for split_name in LEARNING_SPLITS:
                    split_ids = expected_ids_by_split[split_name]
                    texts_by_split[split_name] = [prompt_of[s] for s in split_ids]
                    y_by_split[split_name] = [int(label_of[s]) for s in split_ids]
                metrics = text_baseline_metrics(
                    texts_by_split, y_by_split, c_grid=c_grid,
                    seed=int(cfg.reproducibility.control_seed),
                    class_weight=cfg.probe.class_weight)
                return {
                    **metrics, "model_id": "TF-IDF logistic surface baseline",
                    "resolved_revision": None, "training_stage": "not-applicable",
                    "prompt_interface": interface_name,
                    "chat_template_used": chat_used,
                    "thinking_enabled": ("false" if chat_used
                                         else "not-applicable"),
                    "candidate_verbalizers": [yes_text, no_text],
                    "candidate_token_ids": candidate_ids,
                }

            surface_results = {
                "raw_completion": _text_baseline(
                    raw_prompts, "raw_completion", False)}
            if chat_prompts is not None:
                surface_results["qwen_chat_nonthinking"] = _text_baseline(
                    chat_prompts, "qwen_chat_nonthinking", True)
            save_json(surface_results,
                      run_dir / "controls" / "text_baseline_metrics.json")


        # ---- random-init arms + paired origin deltas -----------------------
        status.update(progress={"stage": "random_init"})
        rand_seeds = [int(x) for x in cfg.diagnostics.random_init_seeds]
        if not rand_seeds:
            raise ValueError("E00-C requires at least one random-init seed")
        results["random_init_seeds"] = rand_seeds
        results["random_init_provenance"] = []
        sc_rand_all: list[tuple[int, dict]] = []
        for seed_k in rand_seeds:
            ad_r = _make_random_adapter(cfg, seed_k)
            revs_r = ad_r.resolved_revisions()
            results["random_init_provenance"].append({
                "seed": seed_k, "display_model_id": ad_r.display_model_id,
                **revs_r,
            })
            rdir = cache_root / (f"{cfg.experiment.id}_v2_{c_hash[:8]}"
                                 f"_rand{seed_k}")
            idx_r, _info = _run_arm_extraction(
                ad_r, arm_samples, rdir,
                build_cache_identity(
                    experiment_id=f"{cfg.experiment.id}_rand{seed_k}",
                    samples=arm_samples,
                    model_id=ad_r.display_model_id,
                    model_resolved_revision=revs_r.get("model_sha"),
                    tokenizer_id=cfg.model.id,
                    tokenizer_resolved_revision=revs_r.get("tokenizer_sha"),
                    sites=sites, layers=layers, token_selectors=selectors,
                    model_dtype=cfg.model.dtype,
                    tokenization={"padding_side": "right",
                                  "add_special_tokens": True}),
                cfg, _split_of)
            mdf_r, sc_r = probe_all_cells(
                ShardedMatrixLoader(rdir, idx_r), idx_r,
                lambda s: int(label_of[s]), expected_ids_by_split,
                sites=sites, selectors=selectors, layers=layers,
                c_grid=c_grid, seed=seed_p, n_bootstraps=n_bootstraps,
                confidence_level=confidence_level)
            random_model_id = ad_r.display_model_id
            random_revision = revs_r.get("model_sha")
            random_note = revs_r.get("resolution_note")
            save_table(
                add_result_metadata(
                    probe_score_evidence_frame(f"random_seed{seed_k}", sc_r),
                    model_id=random_model_id,
                    resolved_revision=random_revision,
                    training_stage="random_initialization",
                    prompt_interface="raw_completion",
                    chat_template_used=False,
                    thinking_enabled="not-applicable",
                    candidates=[yes_text, no_text],
                    candidate_token_ids=candidate_ids,
                    provenance_note=random_note),
                run_dir / f"origin_probe_scores_random_seed{seed_k}.parquet")
            save_table(
                add_result_metadata(
                    mdf_r, model_id=random_model_id,
                    resolved_revision=random_revision,
                    training_stage="random_initialization",
                    prompt_interface="raw_completion",
                    chat_template_used=False,
                    thinking_enabled="not-applicable",
                    candidates=[yes_text, no_text],
                    candidate_token_ids=candidate_ids,
                    provenance_note=random_note),
                run_dir / f"origin_probe_metrics_random_seed{seed_k}.parquet")
            sc_rand_all.append((seed_k, sc_r))
            del ad_r
            torch.cuda.empty_cache()

        delta_rows = []
        for key in sorted(sc_pre.keys()):
            site_, sel_, layer_ = key
            blk_p = sc_pre[key]
            ids_ref = blk_p["sample_ids"]
            y_ref = np.asarray(blk_p["y"])
            rscores = []
            for _sk, sc_r in sc_rand_all:
                if key not in sc_r:
                    break
                sr, yr = _align_scores(ids_ref, sc_r[key])
                assert (yr == y_ref).all()
                rscores.append(sr)
            if len(rscores) != len(rand_seeds):
                continue
            pb = paired_bootstrap_delta(
                blk_p["scores"], rscores, y_ref,
                n_bootstraps=n_bootstraps, seed=seed_p)
            pre_auroc = float(_auroc_fn(y_ref, blk_p["scores"]))
            rnd_aurocs = [float(_auroc_fn(y_ref, s)) for s in rscores]
            delta_rows.append({
                "site": site_, "token_selector": sel_, "layer": layer_,
                "D_pretrained": pre_auroc,
                **{f"D_random_seed{sk}": a
                   for (sk, _s), a in zip(sc_rand_all, rnd_aurocs)},
                "D_random_mean": float(np.mean(rnd_aurocs)),
                "D_random_sd": (float(np.std(rnd_aurocs, ddof=1))
                                if len(rnd_aurocs) > 1 else 0.0),
                **pb, "G_learned": pre_auroc - float(np.mean(rnd_aurocs)),
            })
        delta_df = pd.DataFrame(delta_rows)
        save_table(add_result_metadata(
            delta_df, model_id=f"{pre.display_model_id} vs matched random-init",
            resolved_revision=pre_revs.get("model_sha"),
            training_stage="pretrained_vs_random_comparison",
            prompt_interface="raw_completion", chat_template_used=False,
            thinking_enabled="not-applicable",
            candidates=[yes_text, no_text], candidate_token_ids=candidate_ids,
            provenance_note=f"random seeds {rand_seeds}"),
            run_dir / "origin_delta_vs_random.parquet")

        # ---- LOFO cross-family abstraction --------------------------------
        status.update(progress={"stage": "lofo"})
        main_site = sites[0]
        arm_list = ([("pretrained", raw_dir)] + [
            (f"random_seed{sk}",
             cache_root / f"{cfg.experiment.id}_v2_{c_hash[:8]}_rand{sk}")
            for sk in rand_seeds])
        lofo_rows: list[dict] = []
        lofo_evidence_rows: list[dict] = []
        for arm_name, cdir_arm in arm_list:
            if not completed_shard_ids(cdir_arm):
                logger.warning("LOFO skip %s: no shards", arm_name)
                continue
            idx_arm = ActivationCacheReader(cdir_arm).index()
            selectors_lofo = (list(selectors) if arm_name == "pretrained"
                              else ["last_prompt"])
            for fam in families:
                plan = lofo_family_split(discovery_df, fam)
                fam_map = {s: int(label_of[s]) for s in
                           plan["train_ids"] + plan["validation_ids"]
                           + plan["evaluation_ids"]}
                for sel_l in selectors_lofo:
                    for layer_l in layers:
                        res, evidence = _lofo_cell(
                            cdir_arm, idx_arm, sel_l, layer_l,
                            main_site, plan, fam_map,
                            c_grid=c_grid, seed=seed_p)
                        lofo_rows.append({"arm": arm_name,
                                          "held_out_family": fam, **res})
                        for sid, gold, score in zip(
                                evidence["sample_ids"], evidence["y"],
                                evidence["scores"]):
                            lofo_evidence_rows.append({
                                "arm": arm_name, "held_out_family": fam,
                                "site": main_site, "selector": sel_l,
                                "layer": int(layer_l), "sample_id": sid,
                                "gold": int(gold), "probe_score": float(score),
                            })
            del idx_arm
        lofo_df = pd.DataFrame(lofo_rows)
        lofo_evidence_df = pd.DataFrame(lofo_evidence_rows)

        def _lofo_metadata(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            model_id_map = {"pretrained": pre.display_model_id,
                            **{f"random_seed{s}":
                               f"{cfg.model.id}-random-init-seed{s}"
                               for s in rand_seeds}}
            out["model_id"] = out["arm"].map(model_id_map)
            out["resolved_revision"] = np.where(
                out["arm"].eq("pretrained"), pre_revs.get("model_sha"), None)
            out["training_stage"] = np.where(
                out["arm"].eq("pretrained"), "post-trained/chat",
                "random_initialization")
            out["prompt_interface"] = "raw_completion"
            out["chat_template_used"] = False
            out["thinking_enabled"] = "not-applicable"
            out["candidate_verbalizers"] = json.dumps([yes_text, no_text])
            out["candidate_token_ids"] = json.dumps(candidate_ids)
            return out

        save_table(_lofo_metadata(lofo_evidence_df),
                   run_dir / "lofo_predictions.parquet")
        save_table(_lofo_metadata(lofo_df), run_dir / "lofo_metrics.parquet")
        lofo_delta_rows: list[dict[str, Any]] = []
        pre_lofo = lofo_df[(lofo_df["arm"] == "pretrained")
                           & (lofo_df["selector"] == "last_prompt")]
        for _, row_pre in pre_lofo.iterrows():
            matching = lofo_df[
                lofo_df["arm"].isin([f"random_seed{s}" for s in rand_seeds])
                & (lofo_df["held_out_family"] == row_pre["held_out_family"])
                & (lofo_df["selector"] == "last_prompt")
                & (lofo_df["layer"] == row_pre["layer"])
            ]
            random_values = {
                str(row["arm"]): float(row["auroc"])
                for _, row in matching.iterrows()
            }
            if len(random_values) != len(rand_seeds):
                continue
            values = list(random_values.values())
            random_mean = float(np.mean(values))
            lofo_delta_rows.append({
                "held_out_family": str(row_pre["held_out_family"]),
                "site": main_site, "selector": "last_prompt",
                "layer": int(row_pre["layer"]),
                "D_LOFO_pretrained": float(row_pre["auroc"]),
                **{f"D_LOFO_{arm}": value
                   for arm, value in random_values.items()},
                "D_LOFO_random_mean": random_mean,
                "D_LOFO_random_sd": (float(np.std(values, ddof=1))
                                     if len(values) > 1 else 0.0),
                "delta_D_LOFO": float(row_pre["auroc"]) - random_mean,
            })
        lofo_delta_df = pd.DataFrame(lofo_delta_rows)
        save_table(add_result_metadata(
            lofo_delta_df,
            model_id=f"{pre.display_model_id} vs matched random-init",
            resolved_revision=pre_revs.get("model_sha"),
            training_stage="pretrained_vs_random_comparison",
            prompt_interface="raw_completion", chat_template_used=False,
            thinking_enabled="not-applicable",
            candidates=[yes_text, no_text], candidate_token_ids=candidate_ids,
            provenance_note=f"random seeds {rand_seeds}"),
            run_dir / "lofo_delta_vs_random.parquet")

        # ---- token-embedding rung -----------------------------------------
        status.update(progress={"stage": "embed_rung"})
        embed_rows: list[dict] = []
        embed_evidence_rows: list[dict] = []
        if cfg.diagnostics.embed_control:
            from ..extraction.activations import build_token_selections
            for sel_r in selectors:
                token_ids = []
                for smp in arm_samples:
                    tok_sel = build_token_selections(pre, smp, [sel_r])[0]
                    token_ids.append(int(tok_sel.token_id))
                X_emb_all = pre.token_embeddings(token_ids)
                tr_idx = [i for i, s in enumerate(ids_all)
                          if split_map[s] == "train"]
                va_idx = [i for i, s in enumerate(ids_all)
                          if split_map[s] == "validation"]
                te_idx = [i for i, s in enumerate(ids_all)
                          if split_map[s] == "discovery_test"]
                g_tr = gold_all[tr_idx]; g_va = gold_all[va_idx]
                g_te = gold_all[te_idx]
                fit_e = fit_probe(X_emb_all[tr_idx], g_tr,
                                  X_emb_all[va_idx], g_va,
                                  c_grid=c_grid, seed=seed_p,
                                  standardize=True,
                                  class_weight="balanced")
                sc_te = fit_e["classifier"].decision_function(
                    transform_features(fit_e, X_emb_all[te_idx]))
                for row_idx, score in zip(te_idx, sc_te):
                    embed_evidence_rows.append({
                        "rung": "token_embedding_pretrained",
                        "selector": sel_r,
                        "sample_id": ids_all[row_idx],
                        "token_id": token_ids[row_idx],
                        "gold": int(gold_all[row_idx]),
                        "probe_score": float(score),
                    })
                m_e = classification_metrics(g_te, sc_te)
                row = {"rung": "token_embedding_pretrained",
                       "selector": sel_r, **m_e}
                rel_arr = discovery_df["relation"].astype(str).to_numpy()
                fam_lofo = {}
                for fam in families:
                    fmask = rel_arr[te_idx] == fam
                    if len(np.unique(g_te[fmask])) > 1:
                        mf = classification_metrics(g_te[fmask], sc_te[fmask])
                        fam_lofo[fam] = mf.get("auroc")
                row["family_aurocs"] = fam_lofo
                embed_rows.append(row)
        if embed_rows:
            save_table(_pretrained_metadata(pd.DataFrame(embed_evidence_rows)),
                       run_dir / "embedding_rung_predictions.parquet")
            embed_metrics_df = _pretrained_metadata(pd.DataFrame(embed_rows))
            save_json(embed_metrics_df.to_dict(orient="records"),
                      run_dir / "embedding_rung_metrics.json")

        # ---- fixed native readout + geometry -------------------------------
        status.update(progress={"stage": "fixed_readout"})
        results["candidate_first_token_ids"] = {"yes": yes_id, "no": no_id}
        lens_df, lens_scores = logit_lens_layer_margins(
            pre, raw_dir, idx_raw, yes_id, no_id,
            label_of=lambda sid: label_of[sid])
        lens_evidence = probe_score_evidence_frame("fixed_native_readout",
                                                   lens_scores).rename(
            columns={"probe_score": "fixed_readout_margin"})
        save_table(_pretrained_metadata(lens_evidence),
                   run_dir / "fixed_readout_predictions.parquet")
        save_table(_pretrained_metadata(lens_df),
                   run_dir / "fixed_readout_metrics.parquet")

        sel_best = "last_prompt"
        dir_native = native_readout_direction(pre, yes_id, no_id)

        def nrm(v):
            v = np.asarray(v, dtype=np.float64)
            return v / (np.linalg.norm(v) + 1e-12)

        geom_rows: list[dict] = []
        for key, blk in sc_pre.items():
            s_, sel_, L_ = key
            if sel_ != sel_best:
                continue
            fit = blk["fit"]
            w_raw = raw_probe_direction(fit)
            cos_signed = float(np.dot(nrm(w_raw), nrm(dir_native)))
            ids_t = blk["sample_ids"]
            sc_t = blk["scores"]
            beh = np.asarray([marg_raw[s]["margin_total"] for s in ids_t])
            lens_m, lens_y = _align_scores(ids_t, lens_scores[key])
            if not np.array_equal(lens_y, np.asarray(blk["y"])):
                raise RuntimeError("fixed-readout/probe label alignment mismatch")
            lrow = lens_df[(lens_df.site == s_)
                           & (lens_df.token_selector == sel_)
                           & (lens_df.layer == L_)]
            geom_rows.append({
                "site": s_, "selector": sel_, "layer": int(L_),
                "D_layer": float(_auroc_fn(blk["y"], sc_t)),
                "L_layer": (float(lrow["native_margin_auroc"].iloc[0])
                            if len(lrow) else None),
                "cosine_abs_probe_vs_native": abs(cos_signed),
                "cosine_signed_probe_vs_native": cos_signed,
                "corr_probe_vs_behavior_margin":
                    float(np.corrcoef(sc_t, beh)[0, 1]),
                "corr_lens_vs_behavior_margin":
                    float(np.corrcoef(lens_m, beh)[0, 1]),
            })
        geom_df = pd.DataFrame(geom_rows)
        geom_df["G_DL"] = geom_df["D_layer"] - geom_df["L_layer"]
        save_table(_pretrained_metadata(geom_df),
                   run_dir / "readout_geometry.parquet")

        # ---- calibration (validation-only) ---------------------------------
        status.update(progress={"stage": "calibration"})
        val_ids = [s for s in ids_all if split_map[s] == "validation"]
        disc_ids = [s for s in ids_all if split_map[s] == "discovery_test"]
        train_ids = [s for s in ids_all if split_map[s] == "train"]
        y_val = np.asarray([marg_raw[s]["gold"] for s in val_ids])
        m_val = np.asarray([marg_raw[s]["margin_total"] for s in val_ids])
        tau_star = select_threshold(y_val, m_val)
        y_disc = np.asarray([marg_raw[s]["gold"] for s in disc_ids])
        m_disc = np.asarray([marg_raw[s]["margin_total"] for s in disc_ids])
        pred_disc = (m_disc >= 0).astype(int)
        pred_cal = (m_disc >= tau_star).astype(int)
        calib_1d, coef_1d = fit_1d_calibration(
            np.asarray([marg_raw[s]["gold"] for s in train_ids + val_ids]),
            np.asarray([marg_raw[s]["margin_total"]
                        for s in train_ids + val_ids]))
        pred_1d = calib_1d(m_disc)
        test_rel = discovery_df[
            discovery_df["split"] == "discovery_test"]["relation"].astype(str)
        per_family: dict[str, dict] = {}
        for fam in families:
            fm = np.asarray(test_rel == fam)
            if fm.sum() and len(np.unique(y_disc[fm])) > 1:
                per_family[fam] = {
                    "raw_bal_acc": float(
                        balanced_accuracy_score_global(y_disc[fm],
                                                       pred_disc[fm])),
                    "calibrated_bal_acc": float(
                        balanced_accuracy_score_global(y_disc[fm],
                                                       pred_cal[fm])),
                }
        results["calibration"] = {
            "selected_threshold_tau_star": tau_star,
            "raw_threshold0_accuracy": float((pred_disc == y_disc).mean()),
            "calibrated_accuracy": float((pred_cal == y_disc).mean()),
            "raw_threshold0_balanced_accuracy": float(
                balanced_accuracy_score_global(y_disc, pred_disc)),
            "calibrated_balanced_accuracy": float(
                balanced_accuracy_score_global(y_disc, pred_cal)),
            "one_d_logistic_calibrated_accuracy":
                float((pred_1d == y_disc).mean()),
            "one_d_logistic_coefficients": coef_1d,
            "margin_auroc_vs_gold":
                classification_metrics(y_disc, m_disc)["auroc"],
            "per_family_diagnostic_bal_acc": per_family,
            "raw_accuracy_bootstrap": bootstrap_accuracy_interval(
                y_disc, pred_disc, n_bootstraps=n_bootstraps,
                confidence_level=confidence_level, seed=seed_p),
            "calibrated_accuracy_bootstrap": bootstrap_accuracy_interval(
                y_disc, pred_cal, n_bootstraps=n_bootstraps,
                confidence_level=confidence_level, seed=seed_p),
            "calibrated_minus_raw_paired": paired_bootstrap_accuracy_delta(
                y_disc, pred_disc, pred_cal, n_bootstraps=n_bootstraps,
                seed=seed_p),
        }
        calibration_predictions = pd.DataFrame({
            "sample_id": disc_ids,
            "relation": test_rel.to_numpy(),
            "gold": y_disc,
            "margin_total": m_disc,
            "raw_prediction": pred_disc,
            "calibrated_prediction": pred_cal,
            "one_d_logistic_prediction": pred_1d,
        })
        save_table(_pretrained_metadata(calibration_predictions),
                   run_dir / "calibration_predictions.parquet")
        save_json(results["calibration"], run_dir / "calibration.json")

        # ---- frozen-probe decodability on native behavior errors ----------
        status.update(progress={"stage": "error_subset"})
        err_df = frozen_probe_subset_metrics(sc_pre, marg_raw)
        save_table(_pretrained_metadata(err_df),
                   run_dir / "error_subset_decodability.parquet")

        # ---- gates A-F -----------------------------------------------------
        band_start = 14 if max(layers) >= 14 else layers[len(layers) // 2]
        band = delta_df[(delta_df.token_selector == "last_prompt")
                        & (delta_df.layer >= band_start)]
        mid_gain = float(band["G_learned"].mean()) if len(band) else float("nan")
        fam_table: dict[str, Any] = {}
        lp = lofo_df[lofo_df.arm == "pretrained"]
        lr_ = lofo_df[lofo_df.arm.str.startswith("random_seed")]
        for fam in families:
            fam_delta = lofo_delta_df[
                (lofo_delta_df.held_out_family == fam)
                & (lofo_delta_df.layer >= band_start)]
            fam_table[fam] = {
                "pretrained_LOFO_midlate_mean":
                    float(fam_delta["D_LOFO_pretrained"].mean()),
                "random_LOFO_midlate_mean":
                    float(fam_delta["D_LOFO_random_mean"].mean()),
                "pretrained_minus_random_midlate_mean":
                    float(fam_delta["delta_D_LOFO"].mean()),
                "pretrained_LOFO_exploratory_max": float(
                    lp[(lp.held_out_family == fam)
                       & (lp.selector == sel_best)]["auroc"].max()),
            }
        gate_b_pass = all(
            v["pretrained_LOFO_midlate_mean"] > 0.65
            and v["pretrained_minus_random_midlate_mean"] > 0.05
            for v in fam_table.values())
        cal_gain = float(balanced_accuracy_score_global(
            y_disc, pred_cal)
            - balanced_accuracy_score_global(y_disc, pred_disc))
        interface_delta = None
        if marg_chat is not None:
            disc_chat_pred = np.asarray([marg_chat[s]["pred"]
                                         for s in disc_ids])
            interface_delta = float(
                balanced_accuracy_score_global(y_disc, disc_chat_pred)
                - balanced_accuracy_score_global(y_disc, pred_disc))
        reference_layer = 17 if 17 in layers else layers[len(layers) // 2]
        er_ref = err_df[(err_df.subset == "errors")
                        & (err_df.site == main_site)
                        & (err_df.selector == sel_best)
                        & (err_df.layer == reference_layer)]
        if len(er_ref) != 1:
            raise RuntimeError("missing predeclared error-subset gate row")
        e_row = er_ref.iloc[0]
        e_auroc = e_row["auroc"]
        gb = geom_df[geom_df.site == main_site]
        gb_lp = gb[(gb.selector == sel_best) & (gb.layer >= band_start)]
        d_reference_row = mdf_pre[
            (mdf_pre.site == main_site)
            & (mdf_pre.token_selector == sel_best)
            & (mdf_pre.layer == reference_layer)]
        d_reference = float(d_reference_row["auroc"].iloc[0])
        raw_bal = balanced_accuracy_score_global(y_disc, pred_disc)
        behavior_gap = max(0.0, d_reference - raw_bal)
        cal_closure = cal_gain / behavior_gap if behavior_gap > 0 else 0.0
        interface_closure = (interface_delta / behavior_gap
                             if interface_delta is not None and behavior_gap > 0
                             else None)
        gates = {
            "A_learned_origin": {
                "predeclared_band_start": band_start,
                "midlate_mean_G_learned_last_prompt": mid_gain,
                "pass": bool(mid_gain > 0.05)},
            "B_abstraction": {
                "families": fam_table,
                "criterion": "each family midlate LOFO >0.65 and gain >0.05",
                "pass": bool(gate_b_pass)},
            "C_calibration": {
                "balanced_accuracy_gain": cal_gain,
                "fraction_of_reference_D_minus_B_gap_closed": cal_closure,
                "pass_not_explained_by_calibration": bool(cal_closure < 0.5)},
            "D_interface": {
                "chat_minus_raw_balanced_accuracy": interface_delta,
                "fraction_of_reference_D_minus_B_gap_closed": interface_closure,
                "pass_not_explained_by_interface":
                    bool(interface_closure is not None and interface_closure < 0.5),
            },
            "E_error_accessibility": {
                "reference_layer": reference_layer,
                "errors_subset_AUROC": (float(e_auroc)
                                         if pd.notna(e_auroc) else None),
                "n_errors": int(e_row["n"]),
                "pass": bool(pd.notna(e_auroc)
                             and int(e_row["n"]) >= 20
                             and float(e_auroc) > 0.7)},
            "F_fixed_readout_gap": {
                "predeclared_band_start": band_start,
                "mean_G_DL_midlate_last_prompt":
                    float(gb_lp["G_DL"].mean()),
                "mean_cosine_abs_probe_vs_native":
                    float(gb_lp["cosine_abs_probe_vs_native"].abs().mean()),
                "pass": bool(float(gb_lp["G_DL"].mean()) > 0.10)},
        }
        results["gates"] = gates
        gate_passes = {
            "A": gates["A_learned_origin"]["pass"],
            "B": gates["B_abstraction"]["pass"],
            "C": gates["C_calibration"]["pass_not_explained_by_calibration"],
            "D": gates["D_interface"]["pass_not_explained_by_interface"],
            "E": gates["E_error_accessibility"]["pass"],
            "F": gates["F_fixed_readout_gap"]["pass"],
        }
        if not gate_passes["A"]:
            decision = {
                "branch": "A",
                "validity_verdict": "D mainly random-feature separability",
                "exact_next_recommendation": "redesign E00 latent/task",
                "run_1_7B": False,
            }
        elif not gate_passes["B"]:
            decision = {
                "branch": "A",
                "validity_verdict": "D mainly family-specific",
                "exact_next_recommendation": "redesign E00 latent/task",
                "run_1_7B": False,
            }
        elif not gate_passes["C"] or not gate_passes["D"]:
            decision = {
                "branch": "B",
                "validity_verdict": "B-D gap mainly calibration/interface",
                "exact_next_recommendation":
                    "investigate interface/readout calibration further",
                "run_1_7B": False,
            }
        elif gate_passes["E"] and gate_passes["F"]:
            decision = {
                "branch": "C",
                "validity_verdict": "learned abstract readout bottleneck supported",
                "exact_next_recommendation": "replicate diagnosis on another model family",
                "run_1_7B": True,
            }
        else:
            decision = {
                "branch": "no-scale-inconclusive",
                "validity_verdict": "inconclusive",
                "exact_next_recommendation": "additional integrity repair required",
                "run_1_7B": False,
            }
        results["branch_decision"] = decision
        save_json(results, run_dir / "diagnostics.json")

        # ---- figures -------------------------------------------------------
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            band_df = delta_df[(delta_df.site == main_site)
                               & (delta_df.token_selector == sel_best)]
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            ax1.plot(band_df.layer, band_df.D_pretrained, marker="o",
                     label="pretrained D")
            ax1.plot(band_df.layer, band_df.D_random_mean, marker="s",
                     label="random-init mean D")
            ax1.fill_between(band_df.layer,
                             band_df.D_random_mean - band_df.D_random_sd,
                             band_df.D_random_mean + band_df.D_random_sd,
                             alpha=0.25, label="random-init +/-1 sd")
            ax1.set_xlabel("layer"); ax1.set_ylabel("AUROC")
            ax1.set_title("Figure 1 - representation origin")
            ax1.legend()
            fig1.tight_layout()
            fig1.savefig(run_dir / "figures" / "fig1_origin.png", dpi=150)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(8.5, 5))
            for fam in families:
                sub_p = lp[(lp.held_out_family == fam)
                           & (lp.selector == sel_best)]
                ax2.plot(sub_p.layer, sub_p.auroc,
                         label=f"{fam} pretrained", alpha=0.6)
                sub_r = lr_[(lr_.held_out_family == fam)]
                if len(sub_r):
                    prof = (sub_r[sub_r.selector == "last_prompt"]
                            .groupby("layer")["auroc"].mean())
                    ax2.plot(prof.index, prof.values, linestyle="--",
                             alpha=0.7, label=f"{fam} random")
            ax2.set_xlabel("layer"); ax2.set_ylabel("LOFO AUROC")
            ax2.set_title("Figure 2 - cross-family abstraction "
                          "(solid=pretrained, dashed=random)")
            ax2.legend(fontsize=6, ncol=2)
            fig2.tight_layout()
            fig2.savefig(run_dir / "figures" / "fig2_lofo.png", dpi=150)
            plt.close(fig2)
        except Exception as exc:  # noqa: BLE001 - figures are best-effort artifacts
            logger.warning("figs 1-2 failed: %s", exc)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            gsub = geom_df[(geom_df.site == main_site)
                           & (geom_df.selector == sel_best)]
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.plot(gsub.layer, gsub.D_layer, marker="o",
                     label=f"tuned probe D_l ({sel_best})")
            ax3.plot(gsub.layer, gsub.L_layer, marker="^",
                     label="fixed native readout L_l")
            ax3.axhline(0.5, color="gray", linewidth=0.8, linestyle="--")
            ax3.set_xlabel("layer"); ax3.set_ylabel("AUROC")
            ax3.set_title("Figure 3 - external decoder vs native readout")
            ax3.legend()
            fig3.tight_layout()
            fig3.savefig(run_dir / "figures" / "fig3_readout.png", dpi=150)
            plt.close(fig3)

            band_layer = reference_layer
            blk17 = sc_pre[(main_site, sel_best, band_layer)]
            ids17 = blk17["sample_ids"]
            sc17 = np.asarray(blk17["scores"])
            y17 = np.asarray(blk17["y"])
            cor17 = np.asarray([marg_raw[s]["correct"] for s in ids17])
            m17 = cor17.astype(bool)
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            ax4.scatter(np.where(m17 & (y17 == 1))[0], sc17[m17 & (y17 == 1)],
                        marker="o", alpha=0.8,
                        label="model-correct, gold Yes")
            ax4.scatter(np.where(m17 & (y17 == 0))[0], sc17[m17 & (y17 == 0)],
                        marker="o", facecolors="none", alpha=0.8,
                        label="model-correct, gold No")
            ax4.scatter(np.where((~m17) & (y17 == 1))[0],
                        sc17[(~m17) & (y17 == 1)], marker="x",
                        label="model-error, gold Yes")
            ax4.scatter(np.where((~m17) & (y17 == 0))[0],
                        sc17[(~m17) & (y17 == 0)], marker="+",
                        label="model-error, gold No")
            ax4.axhline(0, color="gray", linewidth=0.8, linestyle="--")
            ax4.set_xlabel("discovery-test example index")
            ax4.set_ylabel(f"frozen truth-probe score (L{band_layer})")
            ax4.set_title("Figure 4 - decodability on native behavior errors")
            ax4.legend(fontsize=7)
            fig4.tight_layout()
            fig4.savefig(run_dir / "figures" / "fig4_errors.png", dpi=150)
            plt.close(fig4)
        except Exception as exc:  # noqa: BLE001 - figures are best-effort artifacts
            logger.warning("figs 3-4 failed: %s", exc)

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            labels5, vals5 = [], []
            labels5.append("raw B (tau=0)")
            vals5.append(results["behavior_raw"]["threshold0_accuracy"])
            labels5.append("raw B (calibrated)")
            vals5.append(results["calibration"]["calibrated_accuracy"])
            if marg_chat is not None:
                chat_sum = _behavior_summary(marg_chat)
                labels5.append("chat non-thinking B")
                vals5.append(chat_sum["threshold0_accuracy"])
            labels5.append("D (best layer)")
            vals5.append(float(max(mdf_pre["auroc"])))
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            ax5.bar(labels5, vals5)
            ax5.axhline(0.5, linestyle="--", color="gray", label="chance")
            ax5.set_ylim(0, 1.05); ax5.set_ylabel("accuracy / AUROC")
            ax5.set_title("Figure 5 - behavior vs decodability by interface")
            ax5.legend()
            fig5.tight_layout()
            fig5.savefig(run_dir / "figures" / "fig5_interface.png", dpi=150)
            plt.close(fig5)
        except Exception as exc:  # noqa: BLE001 - figures are best-effort artifacts
            logger.warning("fig5 failed: %s", exc)

        manifest.finish(runs_summary=[{"gates": gates}])
        status.complete(message="E00-C diagnostics complete (schema-v2 caches)")
        return run_dir
    except Exception as exc:
        logger.exception("E00-C failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish()
        raise
