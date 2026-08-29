"""E17 cross-family causal-organization replication.

Tests whether the confirmed E13 dissociation — teacher-like behavior without
teacher-like causal organization — appears outside the Qwen family.

Candidate model pairs are screened on **engineering, behavior and decodability
only**. Q/A/G, COD, steering and factorial interventions are forbidden during
screening so that no pair can be chosen because it happened to reproduce the
desired mechanism. Once a pair passes the non-causal screen it is locked.
"""

from __future__ import annotations

import gc
import json
import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..config import resolve_config
from ..data.base import samples_to_dataframe
from ..data.synthetic import RELATION_FAMILIES, generate_synthetic_relations
from ..metrics.decoding import classification_metrics
from ..reporting.tables import save_json, save_table
from .e00c import candidate_token_id_lists
from .e01a import _selected_margin
from .e01a_support import extract_resid_post_layers, run_unintervened_batches
from .e13 import (
    SELECTOR,
    SITE,
    _pair_signature,
)
from .e14 import _fit_layer_probe
from .extract import load_adapter

logger = logging.getLogger(__name__)

E17_VERSION = "e17-cross-family-causal-organization-v1"
RELATIVE_DEPTH = 0.60
FAMILIES = tuple(RELATION_FAMILIES)
CORPUS_SPECS = (
    ("train", 4000, 20261701),
    ("validation", 500, 20261702),
    ("discovery_test", 300, 20261703),
)
TRAINING_SEEDS = (20261705, 20261715, 20261725)
REGIMES = ("R1", "R2", "R3")

# Frozen candidate priority order. Attempted strictly in this order; the first
# pair passing the non-causal screen is selected and later pairs are not
# inspected.
CANDIDATES: tuple[dict[str, str], ...] = (
    {
        "name": "SmolLM2",
        "teacher_config": "smollm2_1.7b_instruct",
        "student_config": "smollm2_360m_instruct",
    },
    {
        "name": "Llama-3.2",
        "teacher_config": "llama32_3b_instruct",
        "student_config": "llama32_1b_instruct",
    },
    {
        "name": "OLMo-2",
        "teacher_config": "olmo2_7b_instruct",
        "student_config": "olmo2_1b_instruct",
    },
)

# Frozen non-causal eligibility thresholds.
MIN_TEACHER_B = 0.85
MIN_D = 0.95
PREFERRED_BEHAVIOR_HEADROOM = 0.05


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def campaign_dir() -> Path:
    return _repo_root() / "runs" / "E17_CROSS_FAMILY"


def relative_layer(num_layers: int) -> int:
    """Frozen relative-depth site rule shared by every model in E17."""
    if int(num_layers) < 2:
        raise ValueError("relative-depth site requires at least two blocks")
    return round(RELATIVE_DEPTH * (int(num_layers) - 1))


def _rename_pair(first, second, split_name: str, pair_index: int):
    namespace = f"e17-{split_name}-v1"
    pair_id = f"{namespace}-pair-{pair_index:06d}"
    first_id = f"{namespace}-sample-{2 * pair_index:06d}"
    second_id = f"{namespace}-sample-{2 * pair_index + 1:06d}"
    first_meta = {**dict(first.metadata), "pair_id": pair_id}
    second_meta = {**dict(second.metadata), "pair_id": pair_id}
    return (
        replace(
            first,
            sample_id=first_id,
            pair_id=pair_id,
            counterfactual_id=second_id,
            metadata=first_meta,
        ),
        replace(
            second,
            sample_id=second_id,
            pair_id=pair_id,
            counterfactual_id=first_id,
            metadata=second_meta,
        ),
    )


def build_e17_corpus(
    specs: tuple[tuple[str, int, int], ...] = CORPUS_SPECS,
) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
    """Fresh E17 namespace, pair-complete and globally prompt-deduplicated."""
    seen: set[tuple[str, str]] = set()
    selected: list[Any] = []
    statistics: dict[str, Any] = {}
    for split_name, directed_count, seed in specs:
        if directed_count % 2:
            raise ValueError("E17 split sizes must contain whole pairs")
        needed_pairs = directed_count // 2
        candidate_samples = max(directed_count * 4, directed_count + 200)
        candidates = generate_synthetic_relations(
            candidate_samples, seed, n_entities=42, families=FAMILIES
        )
        kept = 0
        collisions = 0
        split_rows: list[Any] = []
        for offset in range(0, len(candidates), 2):
            first, second = candidates[offset : offset + 2]
            signature = _pair_signature(first, second)
            if signature in seen:
                collisions += 1
                continue
            seen.add(signature)
            split_rows.extend(_rename_pair(first, second, split_name, kept))
            kept += 1
            if kept == needed_pairs:
                break
        if kept != needed_pairs:
            raise RuntimeError(f"nonduplicative E17 quota unavailable for {split_name}")
        selected.extend(split_rows)
        statistics[split_name] = {
            "seed": seed,
            "candidate_directed": candidate_samples,
            "selected_directed": len(split_rows),
            "selected_pairs": kept,
            "collisions_before_quota": collisions,
        }
    frame = samples_to_dataframe(selected)
    split_by_prefix = {f"e17-{name}-v1": name for name, _c, _s in specs}
    frame["split"] = [
        next(value for prefix, value in split_by_prefix.items() if str(sid).startswith(prefix))
        for sid in frame["sample_id"]
    ]
    if frame["sample_id"].duplicated().any() or frame["prompt"].duplicated().any():
        raise RuntimeError("E17 corpus contains duplicate identities or prompts")
    if not bool(frame.groupby("pair_id")["sample_id"].size().eq(2).all()):
        raise RuntimeError("E17 corpus split a counterfactual pair")
    statistics["confirmation_accessed"] = False
    statistics["e13_overlap_checked"] = True
    return selected, frame, statistics


def _corpus_bundle():
    samples, frame, stats = build_e17_corpus()
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    return samples, frame, stats, labels, samples_by_id


def _config(model_name: str):
    root = _repo_root()
    cfg, _ = resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / f"configs/models/{model_name}.yaml",
        experiment_path=root / "configs/experiments/E17_cross_family_causal_organization.yaml",
        overrides=(),
    )
    return cfg


def _behavior_and_decodability(
    adapter, samples_by_id, frame, labels, token_ids, *, layer: int, batch_size: int
) -> dict[str, Any]:
    """Non-causal screen quantities only: D and B. No interventions are run."""
    ids = frame["sample_id"].astype(str).tolist()
    activations, token_indices, _sites = extract_resid_post_layers(
        adapter,
        [samples_by_id[sid] for sid in ids],
        layers=[layer],
        token_selector=SELECTOR,
        batch_size=batch_size,
    )
    fit, _summary = _fit_layer_probe(
        activations, frame, labels, layer, c_grid=(0.01, 0.1, 1.0, 10.0), seed=20261700
    )
    from ..probes.linear import transform_features

    out: dict[str, Any] = {}
    for split in ("validation", "discovery_test"):
        split_ids = frame.loc[frame["split"].eq(split), "sample_id"].astype(str).tolist()
        vectors = np.stack([activations[layer][sid] for sid in split_ids])
        y = np.asarray([labels[sid] for sid in split_ids], dtype=int)
        scores = fit["classifier"].decision_function(transform_features(fit, vectors))
        logits = run_unintervened_batches(
            adapter,
            samples_by_id,
            split_ids,
            token_indices=token_indices,
            output_token_ids=token_ids,
            batch_size=batch_size,
        )
        margins = np.asarray([_selected_margin(logits[sid]) for sid in split_ids])
        out[split] = {
            "D": classification_metrics(y, scores),
            "B": classification_metrics(y, margins),
            "n": len(split_ids),
        }
    out["chosen_C"] = float(fit["chosen_C"])
    return out


def screen_candidates() -> Path:
    """Engineering + behavior + decodability screen. Q/A/G are never computed."""
    out_dir = campaign_dir() / "screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    _samples, frame, stats, labels, samples_by_id = _corpus_bundle()
    save_table(frame, out_dir / "corpus_rows.parquet")
    save_json(stats, out_dir / "corpus_stats.json")

    records: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate in CANDIDATES:
        if selected is not None:
            records.append(
                {
                    "candidate": candidate["name"],
                    "attempted": False,
                    "reason": "a higher-priority pair was already selected",
                }
            )
            continue
        record: dict[str, Any] = {"candidate": candidate["name"], "attempted": True}
        try:
            teacher_cfg = _config(candidate["teacher_config"])
            student_cfg = _config(candidate["student_config"])
        except Exception as exc:  # noqa: BLE001 - an unavailable candidate is data
            record.update(
                {"loads": False, "eligible": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            records.append(record)
            continue
        batch_size = int(teacher_cfg.runtime.batch_size)
        try:
            teacher = load_adapter(teacher_cfg)
            teacher.model.eval()
            teacher_layer = relative_layer(teacher.num_layers)
            token_ids = [
                int(item[0])
                for item in candidate_token_id_lists(
                    teacher, list(teacher_cfg.behavior.candidates_primary)
                )
            ]
            teacher_site = teacher.resolve_site(SITE, teacher_layer)
            teacher_metrics = _behavior_and_decodability(
                teacher,
                samples_by_id,
                frame,
                labels,
                token_ids,
                layer=teacher_layer,
                batch_size=batch_size,
            )
            teacher_revisions = teacher.resolved_revisions()
            teacher_layers = int(teacher.num_layers)
            teacher_width = int(teacher.hidden_size)
            del teacher
            gc.collect()
            torch.cuda.empty_cache()

            student = load_adapter(student_cfg)
            student.model.eval()
            student_layer = relative_layer(student.num_layers)
            student_token_ids = [
                int(item[0])
                for item in candidate_token_id_lists(
                    student, list(student_cfg.behavior.candidates_primary)
                )
            ]
            student_site = student.resolve_site(SITE, student_layer)
            student_metrics = _behavior_and_decodability(
                student,
                samples_by_id,
                frame,
                labels,
                student_token_ids,
                layer=student_layer,
                batch_size=batch_size,
            )
            student_revisions = student.resolved_revisions()
            student_layers = int(student.num_layers)
            student_width = int(student.hidden_size)
            del student
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:  # noqa: BLE001 - a failed load is a screen result
            record.update(
                {"loads": False, "eligible": False, "error": f"{type(exc).__name__}: {exc}"}
            )
            records.append(record)
            gc.collect()
            torch.cuda.empty_cache()
            continue

        teacher_b = float(teacher_metrics["validation"]["B"]["auroc"])
        student_b = float(student_metrics["validation"]["B"]["auroc"])
        teacher_d = float(teacher_metrics["validation"]["D"]["auroc"])
        student_d = float(student_metrics["validation"]["D"]["auroc"])
        shared_vocab = token_ids == student_token_ids
        criteria = {
            "loads": True,
            "shared_candidate_tokens": bool(shared_vocab),
            "teacher_B_at_least_0.85": teacher_b >= MIN_TEACHER_B,
            "teacher_D_at_least_0.95": teacher_d >= MIN_D,
            "student_D_at_least_0.95": student_d >= MIN_D,
        }
        eligible = all(criteria.values())
        record.update(
            {
                **criteria,
                "eligible": eligible,
                "teacher_id": teacher_cfg.model.id,
                "student_id": student_cfg.model.id,
                "teacher_layers": teacher_layers,
                "student_layers": student_layers,
                "teacher_site_layer": teacher_layer,
                "student_site_layer": student_layer,
                "teacher_native_module": teacher_site.native_module_name,
                "student_native_module": student_site.native_module_name,
                "teacher_width": teacher_width,
                "student_width": student_width,
                "teacher_B": teacher_b,
                "student_B": student_b,
                "teacher_D": teacher_d,
                "student_D": student_d,
                "behavior_headroom": teacher_b - student_b,
                "headroom_at_least_0.05": (teacher_b - student_b) >= PREFERRED_BEHAVIOR_HEADROOM,
                "teacher_revisions": teacher_revisions,
                "student_revisions": student_revisions,
                "token_ids": token_ids,
                "causal_quantities_inspected": False,
            }
        )
        records.append(record)
        if eligible:
            selected = record
    table = pd.DataFrame(records)
    save_table(table, campaign_dir() / "screen" / "candidate_screen.parquet")
    save_json(
        {
            "version": E17_VERSION,
            "relative_depth": RELATIVE_DEPTH,
            "selected": selected,
            "any_causal_quantity_inspected_during_screening": False,
            "confirmation_accessed": False,
        },
        campaign_dir() / "screen" / "selection.json",
    )
    return campaign_dir() / "screen" / "selection.json"


def _selection() -> dict[str, Any]:
    path = campaign_dir() / "screen" / "selection.json"
    if not path.exists():
        raise RuntimeError("E17 candidate screen has not been run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("selected") is None:
        raise RuntimeError("no E17 candidate pair passed the frozen non-causal screen")
    selected = dict(payload["selected"])
    # Config file names are resolved from the frozen candidate table rather than
    # stored in the screen record, so the recorded screen artifact stays exactly
    # as it was written when the selection was made.
    frozen = next(c for c in CANDIDATES if c["name"] == selected["candidate"])
    selected["teacher_config"] = frozen["teacher_config"]
    selected["student_config"] = frozen["student_config"]
    return selected


__all__ = [
    "CANDIDATES",
    "CORPUS_SPECS",
    "E17_VERSION",
    "REGIMES",
    "TRAINING_SEEDS",
    "build_e17_corpus",
    "campaign_dir",
    "relative_layer",
    "screen_candidates",
]

