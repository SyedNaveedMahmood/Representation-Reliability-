"""Phase 0A.2 follow-up: chat calibration and normalized-space readout geometry.

This is a narrow non-causal audit that closes two loose ends from E00-C:

1. calibrate the Qwen non-thinking chat Yes/No margin using validation only;
2. compare a truth probe with the native Yes/No LM-head direction in the
   *actual final-normalized hidden space*, where the native direction is exact.

It intentionally does not implement E01 or any intervention.
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
from sklearn.metrics import balanced_accuracy_score

from ..config import config_hash, resolve_config, save_resolved_config
from ..data.splits import build_discovery_label_map, discovery_view
from ..extraction.activations import build_cache_identity, extract_dataset_activations
from ..metrics.decoding import classification_metrics
from ..probes.linear import fit_probe, raw_probe_direction, transform_features
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash
from ..runtime.run_id import allocate_run_dir, make_run_id
from ..runtime.status import StatusFile
from .e00c import (
    _generate_dataset,
    build_chat_samples,
    candidate_first_token_ids,
    paired_bootstrap_accuracy_delta,
    select_threshold,
)
from .extract import load_adapter
from .probe import ShardedMatrixLoader, design_matrices_for

logger = logging.getLogger(__name__)

FOLLOWUP_VERSION = "phase0a2-readout-followup-v1"
LEARNING_SPLITS = ("train", "validation", "discovery_test")


def signed_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Signed cosine similarity; direction orientation is scientifically meaningful."""
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0:
        raise ValueError("cosine requires two non-zero vectors")
    return float(np.dot(x, y) / denom)


def _final_normalize_hidden(adapter, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    """Apply the model's exact final normalization to residual rows.

    This deliberately works in the same coordinate space consumed by the
    LM head. For RMSNorm this avoids treating the sample-dependent RMS
    denominator as if it were a single global linear scaling.
    """
    decoder = getattr(adapter, "_decoder_module", None)
    if decoder is None:
        raise RuntimeError("adapter has no resolved decoder stack")
    norm = getattr(decoder, "norm", None)
    if norm is None:
        raise RuntimeError("decoder stack has no final normalization module")
    param = next(norm.parameters())
    X = np.asarray(X)
    pieces: list[np.ndarray] = []
    for start in range(0, len(X), max(1, int(batch_size))):
        h = torch.as_tensor(
            X[start : start + batch_size],
            device=adapter.device,
            dtype=param.dtype,
        )
        with torch.inference_mode():
            z = norm(h)
        pieces.append(z.detach().float().cpu().numpy())
    return np.concatenate(pieces, axis=0) if pieces else np.zeros_like(X, dtype=np.float32)


def _native_direction_in_normalized_space(adapter, yes_id: int, no_id: int) -> np.ndarray:
    """Exact first-token Yes-minus-No direction after final normalization."""
    head = adapter.model.get_output_embeddings()
    if not isinstance(head, torch.nn.Linear):
        raise TypeError("normalized-space direction requires a linear output head")
    delta = (
        head.weight[int(yes_id)].detach().float()
        - head.weight[int(no_id)].detach().float()
    )
    return delta.cpu().numpy()


def _native_logits_from_normalized(
    adapter,
    Z: np.ndarray,
    token_ids: list[int],
) -> np.ndarray:
    """Apply only the output head to already-final-normalized rows."""
    head = adapter.model.get_output_embeddings()
    ids = torch.as_tensor(token_ids, dtype=torch.long, device=adapter.device)
    h = torch.as_tensor(Z, device=adapter.device, dtype=head.weight.dtype)
    with torch.inference_mode():
        if isinstance(head, torch.nn.Linear):
            weight = head.weight.index_select(0, ids)
            out = h @ weight.transpose(0, 1)
            if head.bias is not None:
                out = out + head.bias.index_select(0, ids)
        else:
            out = head(h).index_select(-1, ids)
    return out.detach().float().cpu().numpy()


def _score_behavior_arm(
    adapter,
    arm_samples,
    split_of: dict[str, str],
    label_of: dict[str, int],
    candidates: list[str],
    *,
    batch_size: int,
    arm_name: str,
) -> pd.DataFrame:
    prompts = [s.prompt for s in arm_samples]
    scores = adapter.score_continuations(
        prompts,
        candidates,
        batch_size=max(1, int(batch_size)),
    )
    rows: list[dict[str, Any]] = []
    for sample, pair in zip(arm_samples, scores):
        yes, no = pair[0], pair[1]
        margin = float(yes["logp_total"] - no["logp_total"])
        pred = int(margin >= 0.0)
        gold = int(label_of[sample.sample_id])
        rows.append(
            {
                "sample_id": sample.sample_id,
                "split": split_of[sample.sample_id],
                "gold_label": gold,
                "arm": arm_name,
                "margin_total": margin,
                "threshold0_prediction": pred,
                "threshold0_correct": int(pred == gold),
                "yes_token_ids": list(yes["token_ids"]),
                "no_token_ids": list(no["token_ids"]),
            }
        )
    return pd.DataFrame(rows)


def _calibrated_behavior_metrics(
    pred_df: pd.DataFrame,
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    val = pred_df[pred_df["split"] == "validation"].copy()
    test = pred_df[pred_df["split"] == "discovery_test"].copy()
    if len(val) == 0 or len(test) == 0:
        raise RuntimeError("behavior calibration requires validation and discovery_test rows")

    y_val = val["gold_label"].to_numpy(dtype=int)
    m_val = val["margin_total"].to_numpy(dtype=np.float64)
    tau = select_threshold(y_val, m_val)

    y = test["gold_label"].to_numpy(dtype=int)
    margins = test["margin_total"].to_numpy(dtype=np.float64)
    pred0 = (margins >= 0.0).astype(int)
    pred_cal = (margins >= tau).astype(int)

    rank = classification_metrics(y, margins)
    delta = paired_bootstrap_accuracy_delta(
        y,
        pred0,
        pred_cal,
        n_bootstraps=max(200, int(bootstrap_samples)),
        seed=int(seed),
    )
    return {
        "n_validation": int(len(val)),
        "n_discovery_test": int(len(test)),
        "selected_threshold_validation_only": float(tau),
        "threshold0_accuracy": float((pred0 == y).mean()),
        "threshold0_balanced_accuracy": float(balanced_accuracy_score(y, pred0)),
        "calibrated_accuracy": float((pred_cal == y).mean()),
        "calibrated_balanced_accuracy": float(
            balanced_accuracy_score(y, pred_cal)
        ),
        "calibrated_minus_threshold0_accuracy": delta,
        "margin_auroc": rank.get("auroc"),
        "margin_auprc": rank.get("auprc"),
    }


def _expected_ids_by_split(discovery_df: pd.DataFrame) -> dict[str, list[str]]:
    return {
        split: discovery_df.loc[
            discovery_df["split"] == split, "sample_id"
        ].astype(str).tolist()
        for split in LEARNING_SPLITS
    }


def _write_markdown_summary(run_dir: Path, summary: dict[str, Any]) -> None:
    raw = summary["behavior"]["raw_completion"]
    chat = summary["behavior"]["qwen_chat_nonthinking"]
    geom = summary["normalized_space_alignment"]
    text = f"""# Phase 0A.2 Readout Follow-up

Model: `{summary['model_id']}`  
Resolved revision: `{summary['resolved_revision']}`  
Layer/site/token: `resid_post / L{summary['layer']} / last_prompt`

## Chat calibration

| Arm | threshold-0 BA | calibrated BA | margin AUROC | validation-selected threshold |
|---|---:|---:|---:|---:|
| raw completion | {raw['threshold0_balanced_accuracy']:.4f} | {raw['calibrated_balanced_accuracy']:.4f} | {raw['margin_auroc']:.4f} | {raw['selected_threshold_validation_only']:.6g} |
| Qwen chat non-thinking | {chat['threshold0_balanced_accuracy']:.4f} | {chat['calibrated_balanced_accuracy']:.4f} | {chat['margin_auroc']:.4f} | {chat['selected_threshold_validation_only']:.6g} |

## Normalized-space readout geometry

- truth-probe AUROC after the **actual final norm**: `{geom['normalized_probe_auroc']:.4f}`
- native fixed-readout AUROC: `{geom['native_readout_auroc']:.4f}`
- signed cosine, normalized-space probe vs exact LM-head Yes-No direction: `{geom['signed_probe_native_cosine']:.6f}`
- absolute cosine: `{geom['absolute_probe_native_cosine']:.6f}`
- probe-score / native-margin correlation: `{geom['probe_score_native_margin_correlation']:.6f}`
- probe-score / raw-behavior-margin correlation: `{geom['probe_score_raw_behavior_margin_correlation']:.6f}`
- native-margin / raw-behavior-margin correlation: `{geom['native_margin_raw_behavior_margin_correlation']:.6f}`
- final-norm+head reconstruction max absolute logit deviation: `{geom['readout_reconstruction_max_abs_dev']:.6g}`

## Interpretation rule

The earlier raw-residual `abs(cosine)` is **not** an exact geometric
characterization of an RMSNorm readout because RMSNorm introduces a
sample-dependent positive denominator. It is superseded for geometry claims
by the signed cosine reported above in the actual final-normalized coordinate
space.

No causal conclusion follows from this follow-up. E01 remains unstarted.
"""
    (run_dir / "READOUT_FOLLOWUP_SUMMARY.md").write_text(text, encoding="utf-8")


def run_e00c_followup(
    base_path=None,
    model_path=None,
    experiment_path=None,
    overrides: tuple[str, ...] = (),
    *,
    layer: int = 17,
) -> Path:
    cfg, provenance = resolve_config(
        base_path=base_path,
        model_path=model_path,
        experiment_path=experiment_path,
        overrides=overrides,
    )
    samples, df = _generate_dataset(cfg)
    discovery_df = discovery_view(df).reset_index(drop=True)
    label_of = build_discovery_label_map(discovery_df)
    split_of = dict(
        zip(
            discovery_df["sample_id"].astype(str),
            discovery_df["split"].astype(str),
        )
    )
    discovery_ids = set(split_of)
    discovery_samples = [s for s in samples if s.sample_id in discovery_ids]
    if len(discovery_samples) != len(discovery_df):
        raise RuntimeError("discovery sample identity mismatch")

    adapter = load_adapter(cfg)
    if not (0 <= int(layer) < adapter.num_layers):
        raise ValueError(
            f"layer {layer} out of range for {adapter.num_layers}-layer model"
        )
    revisions = adapter.resolved_revisions()
    resolved_revision = revisions.get("model_sha")

    split_hash = dataset_split_hash(split_of)
    base_hash = config_hash(cfg)
    followup_hash = hashlib.sha256(
        f"{FOLLOWUP_VERSION}|{base_hash}|layer={int(layer)}".encode()
    ).hexdigest()
    run_id = make_run_id(
        experiment_id="E00CF",
        config_hash=followup_hash,
        seed=int(cfg.reproducibility.seed),
        model_revision=resolved_revision or cfg.model.revision or "unpinned",
        dataset_split_hash=split_hash,
    )

    repo_root = Path(__file__).resolve().parents[3]
    run_dir = allocate_run_dir(
        repo_root / cfg.project.output_root,
        "E00CF",
        run_id,
    )
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    save_resolved_config(
        cfg,
        run_dir / "config.resolved.yaml",
        {
            **provenance,
            "followup_version": FOLLOWUP_VERSION,
            "followup_layer": int(layer),
        },
    )
    status = StatusFile.create(run_dir, run_id, "E00CF")
    manifest = RunManifest(run_dir)
    manifest.set_start(
        followup_hash,
        {
            **provenance,
            "followup_version": FOLLOWUP_VERSION,
            "followup_layer": int(layer),
        },
        cfg.effective_seeds(),
    )
    manifest.update_model_info(
        id=cfg.model.id,
        revision=cfg.model.revision,
        dtype=cfg.model.dtype,
        resolved_revision=resolved_revision,
        tokenizer_revision=revisions.get("tokenizer_sha"),
        notes={"revision_resolution": revisions},
    )

    try:
        candidates = list(cfg.behavior.candidates_primary)
        if len(candidates) != 2:
            raise ValueError("follow-up requires exactly two behavior candidates")
        batch_size = max(8, int(cfg.runtime.batch_size) * 4)

        raw_pred = _score_behavior_arm(
            adapter,
            discovery_samples,
            split_of,
            label_of,
            candidates,
            batch_size=batch_size,
            arm_name="raw_completion",
        )
        chat_samples = build_chat_samples(adapter, discovery_samples)
        chat_pred = _score_behavior_arm(
            adapter,
            chat_samples,
            split_of,
            label_of,
            candidates,
            batch_size=batch_size,
            arm_name="qwen_chat_nonthinking",
        )
        behavior_predictions = pd.concat(
            [raw_pred, chat_pred], ignore_index=True
        )
        save_table(
            behavior_predictions,
            run_dir / "behavior_followup_predictions.parquet",
        )

        behavior = {
            "raw_completion": _calibrated_behavior_metrics(
                raw_pred,
                bootstrap_samples=int(cfg.statistics.bootstrap_samples),
                seed=int(cfg.reproducibility.seed),
            ),
            "qwen_chat_nonthinking": _calibrated_behavior_metrics(
                chat_pred,
                bootstrap_samples=int(cfg.statistics.bootstrap_samples),
                seed=int(cfg.reproducibility.seed) + 1,
            ),
        }

        sites = ["resid_post"]
        selectors = ["last_prompt"]
        layers = [int(layer)]
        identity = build_cache_identity(
            experiment_id="E00CF",
            samples=discovery_samples,
            model_id=cfg.model.id,
            model_resolved_revision=resolved_revision,
            tokenizer_id=str(adapter.tokenizer.name_or_path),
            tokenizer_resolved_revision=revisions.get("tokenizer_sha"),
            sites=sites,
            layers=layers,
            token_selectors=selectors,
            model_dtype=cfg.model.dtype,
            tokenization={
                "prompt_interface": "raw_completion",
                "followup_version": FOLLOWUP_VERSION,
            },
        )
        identity_digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:12]
        cache_dir = (
            repo_root
            / cfg.project.cache_root
            / "activations"
            / f"E00CF_v2_{identity_digest}"
        )
        index_df, extraction_info = extract_dataset_activations(
            adapter,
            discovery_samples,
            cache_dir,
            sites=sites,
            layers=layers,
            token_selectors=selectors,
            shard_size=int(cfg.storage.shard_size),
            batch_size=int(cfg.runtime.batch_size),
            split_of=lambda sid: split_of[str(sid)],
            identity=identity,
            resume=bool(cfg.runtime.resume),
        )
        loader = ShardedMatrixLoader(cache_dir, index_df)
        mats = design_matrices_for(
            loader,
            index_df,
            "last_prompt",
            int(layer),
            "resid_post",
            label_of=label_of,
            expected_ids_by_split=_expected_ids_by_split(discovery_df),
        )

        Z: dict[str, np.ndarray] = {
            split: _final_normalize_hidden(adapter, mats[split]["X"])
            for split in LEARNING_SPLITS
        }
        fit_norm = fit_probe(
            Z["train"],
            mats["train"]["y"],
            Z["validation"],
            mats["validation"]["y"],
            c_grid=cfg.probe.C_grid,
            seed=int(cfg.reproducibility.seed),
            standardize=True,
            class_weight=cfg.probe.class_weight,
        )
        z_test = Z["discovery_test"]
        y_test = np.asarray(mats["discovery_test"]["y"], dtype=int)
        probe_scores = fit_norm["classifier"].decision_function(
            transform_features(fit_norm, z_test)
        )
        probe_metrics = classification_metrics(y_test, probe_scores)
        probe_direction = raw_probe_direction(fit_norm)

        first_ids = candidate_first_token_ids(adapter, candidates)
        yes_id, no_id = int(first_ids[0]), int(first_ids[1])
        native_direction = _native_direction_in_normalized_space(
            adapter, yes_id, no_id
        )
        cos = signed_cosine(probe_direction, native_direction)

        exact_logits = adapter.final_readout_token_logits(
            mats["discovery_test"]["X"],
            [yes_id, no_id],
        )
        reconstructed_logits = _native_logits_from_normalized(
            adapter,
            z_test,
            [yes_id, no_id],
        )
        max_dev = float(
            np.max(np.abs(exact_logits.astype(np.float64) - reconstructed_logits))
        )
        native_margin = (
            exact_logits[:, 0] - exact_logits[:, 1]
        ).astype(np.float64)
        native_metrics = classification_metrics(y_test, native_margin)

        test_ids = list(mats["discovery_test"]["sample_ids"])
        raw_test = raw_pred[raw_pred["split"] == "discovery_test"].set_index(
            "sample_id"
        )
        missing = [sid for sid in test_ids if sid not in raw_test.index]
        if missing:
            raise RuntimeError(
                f"raw behavior rows missing {len(missing)} probe test samples"
            )
        raw_margin = raw_test.loc[test_ids, "margin_total"].to_numpy(
            dtype=np.float64
        )

        def corr(a, b) -> float:
            a = np.asarray(a, dtype=np.float64)
            b = np.asarray(b, dtype=np.float64)
            if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
                return float("nan")
            return float(np.corrcoef(a, b)[0, 1])

        raw_correct = (
            raw_test.loc[test_ids, "threshold0_correct"]
            .to_numpy(dtype=int)
            .astype(bool)
        )
        err = ~raw_correct
        error_subset: dict[str, Any] = {
            "n": int(err.sum()),
            "probe_auroc": None,
            "probe_auprc": None,
        }
        if err.any() and len(np.unique(y_test[err])) >= 2:
            em = classification_metrics(y_test[err], probe_scores[err])
            error_subset.update(
                probe_auroc=em.get("auroc"),
                probe_auprc=em.get("auprc"),
            )

        normalized_alignment = {
            "site": "resid_post",
            "layer": int(layer),
            "token_selector": "last_prompt",
            "normalized_probe_auroc": probe_metrics.get("auroc"),
            "normalized_probe_auprc": probe_metrics.get("auprc"),
            "native_readout_auroc": native_metrics.get("auroc"),
            "native_readout_auprc": native_metrics.get("auprc"),
            "signed_probe_native_cosine": float(cos),
            "absolute_probe_native_cosine": float(abs(cos)),
            "probe_score_native_margin_correlation": corr(
                probe_scores, native_margin
            ),
            "probe_score_raw_behavior_margin_correlation": corr(
                probe_scores, raw_margin
            ),
            "native_margin_raw_behavior_margin_correlation": corr(
                native_margin, raw_margin
            ),
            "readout_reconstruction_max_abs_dev": max_dev,
            "yes_first_token_id": yes_id,
            "no_first_token_id": no_id,
            "behavior_error_subset": error_subset,
            "legacy_raw_residual_cosine_status": (
                "superseded_for_exact_geometry_claims: RMSNorm has a "
                "sample-dependent positive denominator, so raw-space cosine "
                "does not exactly characterize cross-example readout ranking"
            ),
        }

        summary = {
            "followup_version": FOLLOWUP_VERSION,
            "model_id": cfg.model.id,
            "resolved_revision": resolved_revision,
            "layer": int(layer),
            "behavior": behavior,
            "normalized_space_alignment": normalized_alignment,
            "cache_dir": str(cache_dir),
            "extraction": extraction_info,
            "scientific_scope": (
                "non-causal diagnostic only; E01 interventions remain unstarted"
            ),
        }
        save_json(summary, run_dir / "readout_followup_metrics.json")
        _write_markdown_summary(run_dir, summary)

        manifest.finish(
            runs_summary=[
                {
                    "raw_calibrated_balanced_accuracy": behavior[
                        "raw_completion"
                    ]["calibrated_balanced_accuracy"],
                    "chat_calibrated_balanced_accuracy": behavior[
                        "qwen_chat_nonthinking"
                    ]["calibrated_balanced_accuracy"],
                    "normalized_probe_auroc": normalized_alignment[
                        "normalized_probe_auroc"
                    ],
                    "native_readout_auroc": normalized_alignment[
                        "native_readout_auroc"
                    ],
                    "signed_probe_native_cosine": normalized_alignment[
                        "signed_probe_native_cosine"
                    ],
                }
            ]
        )
        status.complete(message="Phase 0A.2 readout follow-up complete")
        return run_dir
    except Exception as exc:
        logger.exception("Phase 0A.2 readout follow-up failed")
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.finish()
        raise
