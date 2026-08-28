"""E14 bounded quantization reliability runner (Qwen3-1.7B only)."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from ..adapters.quantization import normalize_precision, quantize_weight_only
from ..config import config_hash, resolve_config, save_resolved_config
from ..data.splits import build_discovery_label_map, discovery_view
from ..interventions.orthogonal_context import (
    orthogonal_component,
    resolve_context_reference_norm,
    standardize_orthogonal_context,
)
from ..interventions.setpoint import source_free_setpoint_delta
from ..interventions.truth_coordinate import coordinate_value, random_unit_direction
from ..metrics.decoding import classification_metrics
from ..metrics.quantization import evidence_is_finite, factorial_components, summarize_factorial
from ..metrics.setpoint import validation_setpoint_targets
from ..probes.linear import fit_probe, raw_probe_direction, transform_features
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash, prompt_hash
from ..runtime.run_id import allocate_run_dir, make_run_id
from ..runtime.status import StatusFile
from .e00c import _generate_dataset, candidate_token_id_lists
from .e01a import _prediction, _selected_margin
from .e01a_support import (
    deterministic_subset_pair_ids,
    extract_resid_post_layers,
    intervention_base_activations,
    parse_trace_layers,
    run_intervention_batches,
    run_unintervened_batches,
    save_activation_snapshot,
)
from .e01b2_support import (
    CONTEXT_EPSILON,
    build_context_source_plans,
    validation_matched_norm_fallback,
)
from .extract import load_adapter

logger = logging.getLogger(__name__)

E14_VERSION = "e14-bounded-dqag-v1"
SITE = "resid_post"
SELECTOR = "last_prompt"
RANDOM_SEEDS = (1729, 1730, 1731)


def e14_profile(profile: str, max_pairs: int | None) -> tuple[int, int, int]:
    name = str(profile).strip().lower()
    if name not in {"smoke", "pilot"}:
        raise ValueError("E14 authorizes only smoke and pilot profiles")
    default_pairs, random_count, bootstraps = {
        "smoke": (25, 1, 200),
        "pilot": (75, 3, 500),
    }[name]
    pairs = default_pairs if max_pairs is None else int(max_pairs)
    if pairs <= 0 or pairs > default_pairs:
        raise ValueError(f"{name} max_pairs must be in [1, {default_pairs}]")
    return pairs, random_count, bootstraps


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= 1e-12 or not np.isfinite(norm):
        raise ValueError("probe direction is degenerate")
    return value / norm


def _fit_layer_probe(
    activations: dict[int, dict[str, np.ndarray]],
    frame: pd.DataFrame,
    labels: dict[str, int],
    layer: int,
    *,
    c_grid,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    def split(name: str) -> tuple[np.ndarray, np.ndarray]:
        ids = frame.loc[frame["split"].astype(str) == name, "sample_id"].astype(str).tolist()
        return (
            np.stack([activations[int(layer)][sid] for sid in ids]),
            np.asarray([labels[sid] for sid in ids], dtype=int),
        )

    x_train, y_train = split("train")
    x_val, y_val = split("validation")
    fit = fit_probe(x_train, y_train, x_val, y_val, c_grid=c_grid, seed=seed)
    return fit, {
        "layer": int(layer),
        "chosen_C": float(fit["chosen_C"]),
        "validation_auroc": float(fit["validation_auroc_best"]),
    }


def _save_bf16_reference(
    run_dir: Path,
    fits: dict[int, dict[str, Any]],
    directions: dict[int, np.ndarray],
    targets: dict[str, Any],
    identity: dict[str, Any],
) -> None:
    payload: dict[str, Any] = {}
    for layer, fit in fits.items():
        payload[f"coef_{layer}"] = np.asarray(fit["classifier"].coef_[0], dtype=np.float64)
        payload[f"intercept_{layer}"] = np.asarray(fit["classifier"].intercept_, dtype=np.float64)
        payload[f"mean_{layer}"] = np.asarray(fit["scaler_mean"], dtype=np.float64)
        payload[f"scale_{layer}"] = np.asarray(fit["scaler_scale"], dtype=np.float64)
        payload[f"direction_{layer}"] = np.asarray(directions[layer], dtype=np.float64)
    np.savez(run_dir / "bf16_probe_reference.npz", **payload)
    save_json({"targets": targets, "identity": identity}, run_dir / "bf16_reference.json")


def _find_bf16_reference(repo_root: Path, profile: str, split_hash: str) -> Path:
    root = repo_root / "runs" / "E14"
    if not root.exists():
        raise RuntimeError("run BF16 E14 reference before a quantized precision")
    for candidate in sorted(root.iterdir(), reverse=True):
        status = StatusFile.load(candidate)
        metrics_path = candidate / "precision_metrics.json"
        manifest_path = candidate / "manifest.json"
        if status is None or not status.is_complete() or not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            metrics.get("precision") == "bf16"
            and metrics.get("profile") == profile
            and manifest.get("dataset", {}).get("split_hash") == split_hash
            and (candidate / "bf16_probe_reference.npz").exists()
            and (candidate / "bf16_reference.json").exists()
        ):
            return candidate
    raise RuntimeError(f"completed BF16 E14 {profile} reference not found")


def _load_bf16_reference(
    source: Path, layers: list[int]
) -> tuple[dict[int, dict[str, np.ndarray]], dict[int, np.ndarray], dict[str, Any], dict[str, Any]]:
    fits: dict[int, dict[str, np.ndarray]] = {}
    directions: dict[int, np.ndarray] = {}
    with np.load(source / "bf16_probe_reference.npz") as archive:
        for layer in layers:
            fits[layer] = {
                "coef": archive[f"coef_{layer}"].copy(),
                "intercept": archive[f"intercept_{layer}"].copy(),
                "mean": archive[f"mean_{layer}"].copy(),
                "scale": archive[f"scale_{layer}"].copy(),
            }
            directions[layer] = archive[f"direction_{layer}"].copy()
    metadata = json.loads((source / "bf16_reference.json").read_text(encoding="utf-8"))
    return fits, directions, metadata["targets"], metadata["identity"]


def _frozen_scores(reference: dict[str, np.ndarray], vectors: np.ndarray) -> np.ndarray:
    z = (np.asarray(vectors, dtype=np.float64) - reference["mean"]) / reference["scale"]
    return z @ reference["coef"] + float(reference["intercept"].reshape(-1)[0])


def _prompt_nll(adapter, samples_by_id, ids: list[str], batch_size: int) -> dict[str, float]:
    result: dict[str, float] = {}
    for start in range(0, len(ids), max(1, int(batch_size))):
        chunk = ids[start : start + max(1, int(batch_size))]
        encoded = adapter.tokenize([samples_by_id[sid].prompt for sid in chunk])
        input_ids = encoded["input_ids"].to(adapter.device)
        attention = encoded["attention_mask"].to(adapter.device)
        with torch.inference_mode():
            logits = adapter.model(input_ids=input_ids, attention_mask=attention).logits
            losses = F.cross_entropy(
                logits[:, :-1].float().transpose(1, 2),
                input_ids[:, 1:],
                reduction="none",
            )
        mask = attention[:, 1:].to(losses.dtype)
        per_row = (losses * mask).sum(1) / mask.sum(1).clamp_min(1)
        for row, sid in enumerate(chunk):
            result[sid] = float(per_row[row].detach().cpu())
    return result


def _trace_values(adapter, results, ids, layers, directions, token_ids):
    q: dict[int, dict[str, float]] = {layer: {} for layer in layers}
    margin: dict[int, dict[str, float]] = {layer: {} for layer in layers}
    for layer in layers:
        vectors = np.stack([results[sid]["captured"][layer] for sid in ids])
        logits = adapter.final_readout_token_logits(vectors, token_ids)
        for row, sid in enumerate(ids):
            q[layer][sid] = coordinate_value(vectors[row], directions[layer])
            margin[layer][sid] = _selected_margin(logits[row])
    return q, margin


def _write_summary(run_dir: Path, metrics: dict[str, Any]) -> None:
    contrasts = metrics["factorial_contrasts"]
    text = f"""# E14 Bounded Precision Result

Status: `{metrics['status']}`  
Precision: `{metrics['precision']}`  
Profile: `{metrics['profile']}`  
Examples/pairs: `{metrics['n_examples']}` / `{metrics['n_pairs']}`

## D / B / quality

- precision-native D AUROC: `{metrics['D_native']['auroc']}`
- frozen-BF16-axis D AUROC: `{metrics['D_frozen_bf16_axis']['auroc']}`
- behavior balanced accuracy: `{metrics['B']['balanced_accuracy']}`
- prompt perplexity: `{metrics['general_quality']['prompt_perplexity']}`

## Frozen actionability

- Q0: `{metrics['Q0']}`
- A matched-random: `{contrasts['A_matched_minus_random']['mean']}`
- G matched-random: `{contrasts['G_matched_minus_random']['mean']}`

This is a bounded exploratory discovery result. Confirmation was not accessed.
"""
    (run_dir / "E14_SUMMARY.md").write_text(text, encoding="utf-8")


def run_e14(
    base_path=None,
    model_path=None,
    experiment_path=None,
    overrides: tuple[str, ...] = (),
    *,
    precision: str,
    profile: str,
    max_pairs: int | None = None,
    layer: int = 17,
    trace_layers: str = "17,20,23,27",
) -> Path:
    start_time = time.monotonic()
    precision = normalize_precision(precision)
    profile = str(profile).lower()
    pair_cap, random_count, n_bootstraps = e14_profile(profile, max_pairs)
    cfg, provenance = resolve_config(
        base_path=base_path,
        model_path=model_path,
        experiment_path=experiment_path,
        overrides=overrides,
    )
    if str(cfg.experiment.id) != "E14" or str(cfg.experiment.mode) != "discovery":
        raise RuntimeError("E14 runner requires the frozen discovery configuration")
    if str(cfg.model.id) != "Qwen/Qwen3-1.7B":
        raise RuntimeError("E14 Stage 0/1 is authorized only for Qwen3-1.7B")
    if int(layer) != 17:
        raise RuntimeError("E14 intervention layer is frozen at 17")

    samples, full_df = _generate_dataset(cfg)
    frame = discovery_view(full_df).reset_index(drop=True)
    if (frame["split"].astype(str) == "confirmation").any():
        raise RuntimeError("discovery view exposed confirmation")
    labels = build_discovery_label_map(frame)
    ids = frame["sample_id"].astype(str).tolist()
    id_set = set(ids)
    samples_by_id = {str(sample.sample_id): sample for sample in samples if str(sample.sample_id) in id_set}
    split_of = dict(zip(frame["sample_id"].astype(str), frame["split"].astype(str)))
    split_hash = dataset_split_hash(split_of)

    adapter = load_adapter(cfg)
    revisions = adapter.resolved_revisions()
    quantization = quantize_weight_only(adapter.model, precision)
    adapter.model.eval()
    traces = parse_trace_layers(trace_layers, intervention_layer=layer, num_layers=adapter.num_layers)
    if traces != [17, 20, 23, 27]:
        raise RuntimeError("E14 trace layers are frozen at 17,20,23,27")
    needed_layers = sorted({layer, *traces})
    candidates = list(cfg.behavior.candidates_primary)
    candidate_lists = candidate_token_id_lists(adapter, candidates)
    if len(candidates) != 2 or any(len(value) != 1 for value in candidate_lists):
        raise RuntimeError("E14 requires single-token Yes/No candidates")
    token_ids = [int(value[0]) for value in candidate_lists]

    shape = {
        "version": E14_VERSION,
        "precision": precision,
        "profile": profile,
        "max_pairs": pair_cap,
        "random_seeds": list(RANDOM_SEEDS[:random_count]),
        "trace_layers": traces,
    }
    resolved_hash = hashlib.sha256(
        (config_hash(cfg) + "|" + json.dumps(shape, sort_keys=True)).encode()
    ).hexdigest()
    run_id = make_run_id(
        "E14",
        resolved_hash,
        int(cfg.reproducibility.seed),
        revisions.get("model_sha") or "unpinned",
        split_hash,
    )
    repo_root = Path(__file__).resolve().parents[3]
    canonical = repo_root / cfg.project.output_root / "E14" / run_id
    existing = StatusFile.load(canonical)
    if existing is not None and existing.is_complete():
        return canonical
    run_dir = allocate_run_dir(repo_root / cfg.project.output_root, "E14", run_id)
    save_resolved_config(cfg, run_dir / "config.resolved.yaml", {**provenance, "shape": shape})
    status = StatusFile.create(run_dir, run_id, "E14")
    manifest = RunManifest(run_dir)
    manifest.set_start(resolved_hash, {**provenance, "e14_shape": shape}, cfg.effective_seeds())
    manifest.update_model_info(
        id=cfg.model.id,
        revision=cfg.model.revision,
        resolved_revision=revisions.get("model_sha"),
        tokenizer_revision=revisions.get("tokenizer_sha"),
        dtype=cfg.model.dtype,
        precision=precision,
        quantization=quantization,
        num_layers=adapter.num_layers,
        hidden_size=adapter.hidden_size,
        candidate_texts=candidates,
        candidate_token_ids=token_ids,
        resolved_native_modules={
            f"{SITE}:{trace}": adapter.resolve_site(SITE, trace).native_module_name
            for trace in needed_layers
        },
    )
    manifest.update_dataset_info(
        split_hash=split_hash,
        prompt_hash_sample=prompt_hash(str(frame.iloc[0]["prompt"])),
        confirmation_accessed=False,
    )

    try:
        activations, token_indices, token_sites = extract_resid_post_layers(
            adapter,
            [samples_by_id[sid] for sid in ids],
            layers=needed_layers,
            token_selector=SELECTOR,
            batch_size=int(cfg.runtime.batch_size),
        )
        save_activation_snapshot(
            run_dir,
            activations,
            sample_ids=ids,
            token_indices=token_indices,
            token_sites=token_sites,
        )
        fits: dict[int, dict[str, Any]] = {}
        native_directions: dict[int, np.ndarray] = {}
        probe_rows: list[dict[str, Any]] = []
        for trace in needed_layers:
            fit, row = _fit_layer_probe(
                activations,
                frame,
                labels,
                trace,
                c_grid=cfg.probe.C_grid,
                seed=int(cfg.reproducibility.probe_seed),
            )
            fits[trace] = fit
            native_directions[trace] = _unit(raw_probe_direction(fit))
            probe_rows.append(row)

        validation_ids = frame.loc[frame["split"] == "validation", "sample_id"].astype(str).tolist()
        validation_labels = np.asarray([labels[sid] for sid in validation_ids], dtype=int)
        if precision == "bf16":
            reference_fits = {
                trace: {
                    "coef": np.asarray(fits[trace]["classifier"].coef_[0]),
                    "intercept": np.asarray(fits[trace]["classifier"].intercept_),
                    "mean": np.asarray(fits[trace]["scaler_mean"]),
                    "scale": np.asarray(fits[trace]["scaler_scale"]),
                }
                for trace in needed_layers
            }
            reference_directions = native_directions
            validation_clean = run_unintervened_batches(
                adapter,
                samples_by_id,
                validation_ids,
                token_indices=token_indices,
                output_token_ids=token_ids,
                batch_size=int(cfg.runtime.batch_size),
            )
            validation_margins = np.asarray(
                [_selected_margin(validation_clean[sid]) for sid in validation_ids]
            )
            targets = validation_setpoint_targets(
                np.stack([activations[layer][sid] for sid in validation_ids])
                @ reference_directions[layer],
                validation_labels,
                validation_margins,
            )
            reference_identity = {
                "model_id": str(cfg.model.id),
                "resolved_revision": revisions.get("model_sha"),
                "tokenizer_revision": revisions.get("tokenizer_sha"),
                "split_hash": split_hash,
                "candidate_token_ids": token_ids,
                "layers": needed_layers,
                "confirmation_accessed": False,
            }
            _save_bf16_reference(run_dir, fits, reference_directions, targets, reference_identity)
            reference_source = run_dir
        else:
            reference_source = _find_bf16_reference(repo_root, profile, split_hash)
            reference_fits, reference_directions, targets, reference_identity = (
                _load_bf16_reference(reference_source, needed_layers)
            )
            expected_identity = {
                "model_id": str(cfg.model.id),
                "resolved_revision": revisions.get("model_sha"),
                "tokenizer_revision": revisions.get("tokenizer_sha"),
                "split_hash": split_hash,
                "candidate_token_ids": token_ids,
                "layers": needed_layers,
                "confirmation_accessed": False,
            }
            if reference_identity != expected_identity:
                raise RuntimeError("BF16 reference identity mismatch")

        selected_pairs = deterministic_subset_pair_ids(
            frame, max_pairs=pair_cap, seed=int(cfg.reproducibility.control_seed)
        )
        base_df = frame[
            (frame["split"] == "discovery_test")
            & frame["pair_id"].astype(str).isin(selected_pairs)
        ].copy()
        base_ids = base_df["sample_id"].astype(str).tolist()
        if len(base_ids) > (50 if profile == "smoke" else 150):
            raise RuntimeError("E14 bounded profile exceeded its directed-example limit")
        if any(
            token_indices[sid] != int(token_sites[sid]["sequence_length"]) - 1
            for sid in base_ids
        ):
            raise RuntimeError("last_prompt is not the final non-padding token")

        x_eval = np.stack([activations[layer][sid] for sid in base_ids])
        y_eval = np.asarray([labels[sid] for sid in base_ids], dtype=int)
        native_scores = fits[layer]["classifier"].decision_function(
            transform_features(fits[layer], x_eval)
        )
        frozen_scores = _frozen_scores(reference_fits[layer], x_eval)
        d_native = classification_metrics(y_eval, native_scores)
        d_frozen = classification_metrics(y_eval, frozen_scores)
        for row in probe_rows:
            trace = int(row["layer"])
            eval_vectors = np.stack([activations[trace][sid] for sid in base_ids])
            row.update(
                {
                    "precision": precision,
                    "native_auroc_bounded": classification_metrics(
                        y_eval,
                        fits[trace]["classifier"].decision_function(
                            transform_features(fits[trace], eval_vectors)
                        ),
                    )["auroc"],
                    "frozen_bf16_axis_auroc_bounded": classification_metrics(
                        y_eval, _frozen_scores(reference_fits[trace], eval_vectors)
                    )["auroc"],
                }
            )
        save_table(pd.DataFrame(probe_rows), run_dir / "probe_metrics.parquet")

        validation_references: dict[int, dict[str, float]] = {}
        for trace in needed_layers:
            vectors = np.stack([activations[trace][sid] for sid in validation_ids])
            margins = adapter.final_readout_token_logits(vectors, token_ids)
            sigma_q = float(np.std(vectors @ reference_directions[trace], ddof=1))
            sigma_margin = float(np.std(margins[:, 0] - margins[:, 1], ddof=1))
            if min(sigma_q, sigma_margin) <= 1e-12:
                raise RuntimeError("quantized validation standardization is degenerate")
            validation_references[trace] = {
                "sigma_q": sigma_q,
                "sigma_margin": sigma_margin,
            }

        hidden_dim = int(adapter.hidden_size)
        zeros = {sid: np.zeros(hidden_dim, dtype=np.float64) for sid in base_ids}
        clean = run_intervention_batches(
            adapter,
            samples_by_id,
            base_ids,
            layer=layer,
            token_indices=token_indices,
            deltas_by_id=zeros,
            output_token_ids=token_ids,
            capture_layers=traces,
            batch_size=int(cfg.runtime.batch_size),
        )
        unhooked = run_unintervened_batches(
            adapter,
            samples_by_id,
            base_ids,
            token_indices=token_indices,
            output_token_ids=token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        no_op_max = max(
            float(np.max(np.abs(clean[sid]["selected_logits"] - unhooked[sid])))
            for sid in base_ids
        )
        bases = intervention_base_activations(clean, base_ids, layer=layer)
        u = reference_directions[layer]
        q_targets = {
            sid: float(targets["q1_star"] if labels[sid] == 0 else targets["q0_star"])
            for sid in base_ids
        }
        semantic = {
            sid: source_free_setpoint_delta(bases[sid], u, q_targets[sid]) for sid in base_ids
        }
        y10 = run_intervention_batches(
            adapter,
            samples_by_id,
            base_ids,
            layer=layer,
            token_indices=token_indices,
            deltas_by_id=semantic,
            output_token_ids=token_ids,
            capture_layers=traces,
            batch_size=int(cfg.runtime.batch_size),
        )
        plans = build_context_source_plans(
            frame,
            samples_by_id,
            base_sample_ids=base_ids,
            seed=int(cfg.reproducibility.control_seed),
        )
        fallback = validation_matched_norm_fallback(
            frame,
            samples_by_id,
            activations[layer],
            u,
            epsilon=CONTEXT_EPSILON,
        )
        matched_context: dict[str, np.ndarray] = {}
        reference_norms: dict[str, float] = {}
        context_diagnostics: dict[tuple[str, int | None], dict[str, dict[str, Any]]] = {}
        for sid in base_ids:
            raw = orthogonal_component(
                activations[layer][plans[sid].matched_source_id], bases[sid], u
            )
            raw_norm = float(np.linalg.norm(raw))
            reference_norm, _, _ = resolve_context_reference_norm(
                raw_norm,
                float(fallback["median_nondegenerate_matched_orthogonal_norm"]),
                epsilon=CONTEXT_EPSILON,
            )
            context, diagnostics = standardize_orthogonal_context(
                raw,
                u,
                reference_norm,
                epsilon=CONTEXT_EPSILON,
                fallback_direction=random_unit_direction(hidden_dim, 99173, orthogonal_to=u),
            )
            matched_context[sid] = context
            reference_norms[sid] = reference_norm
            context_diagnostics.setdefault(("matched", None), {})[sid] = diagnostics

        contexts: list[tuple[str, int | None, dict[str, np.ndarray]]] = [
            ("matched", None, matched_context)
        ]
        for direction_seed in RANDOM_SEEDS[:random_count]:
            direction = random_unit_direction(hidden_dim, direction_seed, orthogonal_to=u)
            by_id: dict[str, np.ndarray] = {}
            diagnostics_by_id: dict[str, dict[str, Any]] = {}
            for sid in base_ids:
                context, diagnostics = standardize_orthogonal_context(
                    direction,
                    u,
                    reference_norms[sid],
                    epsilon=CONTEXT_EPSILON,
                )
                by_id[sid] = context
                diagnostics_by_id[sid] = diagnostics
            contexts.append(("random", direction_seed, by_id))
            context_diagnostics[("random", direction_seed)] = diagnostics_by_id

        clean_q, clean_trace_margin = _trace_values(
            adapter, clean, base_ids, traces, reference_directions, token_ids
        )
        q10_trace, m10_trace = _trace_values(
            adapter, y10, base_ids, traces, reference_directions, token_ids
        )
        clean_margins = {sid: _selected_margin(clean[sid]["selected_logits"]) for sid in base_ids}
        q10_margins = {sid: _selected_margin(y10[sid]["selected_logits"]) for sid in base_ids}
        factorial_rows: list[dict[str, Any]] = []
        trace_rows: list[dict[str, Any]] = []
        max_q_preservation = 0.0
        max_target_error = 0.0
        max_context_dot = 0.0
        max_norm_error = 0.0
        for context_name, direction_seed, vectors in contexts:
            y01 = run_intervention_batches(
                adapter,
                samples_by_id,
                base_ids,
                layer=layer,
                token_indices=token_indices,
                deltas_by_id=vectors,
                output_token_ids=token_ids,
                capture_layers=traces,
                batch_size=int(cfg.runtime.batch_size),
            )
            combined = {sid: semantic[sid] + vectors[sid] for sid in base_ids}
            y11 = run_intervention_batches(
                adapter,
                samples_by_id,
                base_ids,
                layer=layer,
                token_indices=token_indices,
                deltas_by_id=combined,
                output_token_ids=token_ids,
                capture_layers=traces,
                batch_size=int(cfg.runtime.batch_size),
            )
            q01_trace, m01_trace = _trace_values(
                adapter, y01, base_ids, traces, reference_directions, token_ids
            )
            q11_trace, m11_trace = _trace_values(
                adapter, y11, base_ids, traces, reference_directions, token_ids
            )
            for sid in base_ids:
                target_label = 1 - int(labels[sid])
                orientation = 1.0 if target_label == 1 else -1.0
                y00_raw = clean_margins[sid]
                y10_raw = q10_margins[sid]
                y01_raw = _selected_margin(y01[sid]["selected_logits"])
                y11_raw = _selected_margin(y11[sid]["selected_logits"])
                estimates = factorial_components(
                    orientation * y00_raw,
                    orientation * y10_raw,
                    orientation * y01_raw,
                    orientation * y11_raw,
                )
                q_base = coordinate_value(bases[sid], u)
                q01_after = coordinate_value(y01[sid]["captured"][layer], u)
                q11_after = coordinate_value(y11[sid]["captured"][layer], u)
                max_q_preservation = max(max_q_preservation, abs(q01_after - q_base))
                max_target_error = max(max_target_error, abs(q11_after - q_targets[sid]))
                diagnostics = context_diagnostics[(context_name, direction_seed)][sid]
                max_context_dot = max(
                    max_context_dot, abs(float(diagnostics["context_dot_truth_direction"]))
                )
                max_norm_error = max(
                    max_norm_error, float(diagnostics["context_norm_relative_error"])
                )
                factorial_rows.append(
                    {
                        "model_id": str(cfg.model.id),
                        "precision": precision,
                        "base_sample_id": sid,
                        "pair_id": str(samples_by_id[sid].pair_id),
                        "relation_family": str(samples_by_id[sid].metadata["relation"]),
                        "gold_label": int(labels[sid]),
                        "target_label": target_label,
                        "context": context_name,
                        "direction_seed": direction_seed,
                        "context_source_id": (
                            plans[sid].matched_source_id if context_name == "matched" else None
                        ),
                        "q_base": q_base,
                        "q_target": q_targets[sid],
                        "q_after_y01": q01_after,
                        "q_after_y11": q11_after,
                        "context_norm": float(np.linalg.norm(vectors[sid])),
                        "context_dot_u": float(np.dot(vectors[sid], u)),
                        "Y00": orientation * y00_raw,
                        "Y10": orientation * y10_raw,
                        "Y01": orientation * y01_raw,
                        "Y11": orientation * y11_raw,
                        "Q0": float(estimates["Q0"]),
                        "A": float(estimates["A"]),
                        "Q_context": float(estimates["Q_context"]),
                        "G": float(estimates["G"]),
                        "confirmation_accessed": False,
                    }
                )
                for trace in traces:
                    trace_est_q = factorial_components(
                        orientation * clean_q[trace][sid],
                        orientation * q10_trace[trace][sid],
                        orientation * q01_trace[trace][sid],
                        orientation * q11_trace[trace][sid],
                    )
                    trace_est_m = factorial_components(
                        orientation * clean_trace_margin[trace][sid],
                        orientation * m10_trace[trace][sid],
                        orientation * m01_trace[trace][sid],
                        orientation * m11_trace[trace][sid],
                    )
                    refs = validation_references[trace]
                    trace_rows.append(
                        {
                            "precision": precision,
                            "base_sample_id": sid,
                            "pair_id": str(samples_by_id[sid].pair_id),
                            "context": context_name,
                            "direction_seed": direction_seed,
                            "trace_layer": trace,
                            "q00": clean_q[trace][sid],
                            "q10": q10_trace[trace][sid],
                            "q01": q01_trace[trace][sid],
                            "q11": q11_trace[trace][sid],
                            "m00": clean_trace_margin[trace][sid],
                            "m10": m10_trace[trace][sid],
                            "m01": m01_trace[trace][sid],
                            "m11": m11_trace[trace][sid],
                            "A_q_z": float(trace_est_q["A"]) / refs["sigma_q"],
                            "G_q_z": float(trace_est_q["G"]) / refs["sigma_q"],
                            "A_margin_z": float(trace_est_m["A"]) / refs["sigma_margin"],
                            "G_margin_z": float(trace_est_m["G"]) / refs["sigma_margin"],
                            "confirmation_accessed": False,
                        }
                    )

        factorial_df = pd.DataFrame(factorial_rows)
        trace_df = pd.DataFrame(trace_rows)
        save_table(factorial_df, run_dir / "factorial_rows.parquet")
        save_table(trace_df, run_dir / "trace_rows.parquet")
        aggregate, contrasts = summarize_factorial(
            factorial_df,
            n_bootstraps=n_bootstraps,
            confidence_level=float(cfg.statistics.confidence_level),
            seed=int(cfg.reproducibility.bootstrap_seed),
        )
        save_table(aggregate, run_dir / "factorial_metrics.parquet")

        nll = _prompt_nll(adapter, samples_by_id, base_ids, int(cfg.runtime.batch_size))
        behavior_rows = pd.DataFrame(
            [
                {
                    "precision": precision,
                    "base_sample_id": sid,
                    "pair_id": str(samples_by_id[sid].pair_id),
                    "gold_label": int(labels[sid]),
                    "yes_no_margin": clean_margins[sid],
                    "prediction": _prediction(clean_margins[sid]),
                    "prompt_nll": nll[sid],
                    "prompt_token_count": int(token_sites[sid]["sequence_length"]),
                    "token_index": int(token_indices[sid]),
                    "candidate_yes_id": token_ids[0],
                    "candidate_no_id": token_ids[1],
                    "confirmation_accessed": False,
                }
                for sid in base_ids
            ]
        )
        save_table(behavior_rows, run_dir / "behavior_rows.parquet")
        b_metrics = classification_metrics(y_eval, behavior_rows["yes_no_margin"].to_numpy(float))
        prompt_nll = float(behavior_rows["prompt_nll"].mean())
        q0 = float(factorial_df.groupby("base_sample_id")["Q0"].first().mean())
        metrics = {
            "version": E14_VERSION,
            "status": "complete",
            "precision": precision,
            "profile": profile,
            "run_id": run_id,
            "n_examples": len(base_ids),
            "n_pairs": int(base_df["pair_id"].nunique()),
            "D_native": d_native,
            "D_frozen_bf16_axis": d_frozen,
            "B": b_metrics,
            "general_quality": {
                "mean_prompt_nll": prompt_nll,
                "prompt_perplexity": float(math.exp(min(prompt_nll, 50.0))),
            },
            "Q0": q0,
            "factorial_metrics": aggregate.to_dict(orient="records"),
            "factorial_contrasts": contrasts,
            "validation_references": {str(key): value for key, value in validation_references.items()},
            "integrity": {
                "no_op_max_abs_logit_deviation": no_op_max,
                "context_only_q_max_abs_deviation": max_q_preservation,
                "combined_target_q_max_abs_deviation": max_target_error,
                "max_context_dot_u": max_context_dot,
                "max_context_norm_relative_error": max_norm_error,
                "finite": bool(
                    evidence_is_finite(factorial_df, trace_df)
                ),
                "trace_rows": len(trace_df),
                "runtime_residual_dtype": "bfloat16",
                "confirmation_accessed": False,
            },
            "quantization": quantization,
            "bf16_reference_source": str(reference_source.relative_to(repo_root)),
            "wall_time_s": float(time.monotonic() - start_time),
            "confirmation_accessed": False,
        }
        if not metrics["integrity"]["finite"]:
            raise RuntimeError("E14 produced nonfinite evidence")
        primary_sigma_q = validation_references[layer]["sigma_q"]
        if no_op_max > 1e-6:
            raise RuntimeError(f"E14 hooked no-op mismatch: {no_op_max}")
        if max_q_preservation / primary_sigma_q > 0.05:
            raise RuntimeError("E14 context-only edit drifted from native q")
        if max_target_error / primary_sigma_q > 0.05:
            raise RuntimeError("E14 q+context edit missed the frozen setpoint")
        if max_context_dot > 1e-10 or max_norm_error > 1e-10:
            raise RuntimeError("E14 context orthogonality or norm matching failed")
        if profile == "smoke" and precision != "bf16":
            bf16_metrics = json.loads(
                (reference_source / "precision_metrics.json").read_text(encoding="utf-8")
            )
            if metrics["general_quality"]["prompt_perplexity"] > 10.0 * float(
                bf16_metrics["general_quality"]["prompt_perplexity"]
            ):
                raise RuntimeError("quantized prompt perplexity exceeds the frozen 10x smoke gate")
        save_json(metrics, run_dir / "precision_metrics.json")
        _write_summary(run_dir, metrics)
        manifest.finish([{"precision": precision, "profile": profile, "status": "complete"}])
        status.complete("E14 bounded precision run complete")
        return run_dir
    except Exception as exc:
        status.fail(f"{type(exc).__name__}: {exc}")
        manifest.manifest.setdefault("errors", []).append(
            {"type": type(exc).__name__, "message": str(exc)}
        )
        manifest.finish()
        raise
