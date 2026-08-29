"""Pre-registered secondary representation-similarity analysis on the E13 holdout.

Section 8 of ``docs/E13_DIAGNOSTIC_CONFIRMATION_PROTOCOL.md`` pre-registers linear
CKA and, for R3, projected hidden cosine/MSE as *secondary* views. This script
computes them on the already-materialized confirmation rows inside the same
campaign and the same single access; it introduces no threshold, changes no
primary verdict, and is explicitly not permitted to rescue a primary claim.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from representation_reliability.metrics.causal_organization import (
    representation_similarity,
)
from representation_reliability.reporting.tables import save_json, save_table
from representation_reliability.runners.e01a_support import (
    extract_resid_post_layers,
)
from representation_reliability.runners.e13 import LAYER, SELECTOR
from representation_reliability.runners.e13_diagnostic_confirmation import (
    _open_corpus_signatures,
    _student_config,
    _teacher_config,
)
from representation_reliability.runners.e13_diagnostic_confirmation_support import (
    CHECKPOINT_REGISTRY,
    materialize_e13_holdout,
    resolve_checkpoint,
)
from representation_reliability.runners.e13_multiseed import (
    _checkpoint_adapter,
    _load_projector,
)
from representation_reliability.runners.extract import load_adapter

CAMPAIGN = ROOT / "runs" / "E13_DIAGNOSTIC_CONFIRMATION" / "E13DC_e13-diagnostic-confirmation-v1"


def _require_single_prior_access() -> dict:
    ledger = ROOT / "runs" / "E13_DIAGNOSTIC_CONFIRMATION" / "E13_CONFIRMATION_ACCESS.json"
    record = json.loads(ledger.read_text(encoding="utf-8"))
    if int(record["access_count"]) != 1:
        raise RuntimeError("secondary analysis requires exactly one prior access")
    if not (CAMPAIGN / "confirmation_verdict.json").exists():
        raise RuntimeError("primary confirmation must be complete before secondary analysis")
    return record


def _activations(adapter, samples_by_id, ids, batch_size):
    activations, _indices, _sites = extract_resid_post_layers(
        adapter,
        [samples_by_id[sid] for sid in ids],
        layers=[LAYER],
        token_selector=SELECTOR,
        batch_size=batch_size,
    )
    return np.stack([activations[LAYER][sid] for sid in ids])


def main() -> int:
    record = _require_single_prior_access()
    signatures, _open_frame, _stats = _open_corpus_signatures()
    samples, frame = materialize_e13_holdout(signatures)
    samples_by_id = {str(sample.sample_id): sample for sample in samples}
    ids = frame["sample_id"].astype(str).tolist()

    stored = json.loads((CAMPAIGN / "holdout_identity.json").read_text(encoding="utf-8"))
    if len(ids) != int(stored["holdout"]["n_rows"]):
        raise RuntimeError("secondary analysis holdout size differs from the primary run")

    cfg = _student_config()
    batch_size = int(cfg.runtime.batch_size)

    teacher = load_adapter(_teacher_config())
    teacher.model.eval()
    teacher_matrix = _activations(teacher, samples_by_id, ids, batch_size)
    teacher_width = int(teacher.hidden_size)
    del teacher
    gc.collect()
    torch.cuda.empty_cache()

    teacher_normalized = teacher_matrix / np.sqrt(
        np.mean(teacher_matrix.astype(np.float64) ** 2, axis=1, keepdims=True) + 1e-8
    )

    rows = []
    student_cfg = cfg
    r0 = load_adapter(student_cfg)
    r0.model.eval()
    student_width = int(r0.hidden_size)
    r0_matrix = _activations(r0, samples_by_id, ids, batch_size)
    del r0
    gc.collect()
    torch.cuda.empty_cache()
    rows.append(
        {
            "model": "R0",
            "regime": "R0",
            "seed": None,
            "linear_CKA": representation_similarity(
                r0_matrix, teacher_normalized, teacher_normalized, cka_teacher=teacher_matrix
            )["linear_CKA"],
            "mean_cosine_after_projector": None,
            "projected_hidden_MSE": None,
        }
    )

    for key in sorted(CHECKPOINT_REGISTRY):
        entry = CHECKPOINT_REGISTRY[key]
        checkpoint = resolve_checkpoint(ROOT, key)
        adapter = _checkpoint_adapter(student_cfg, checkpoint)
        adapter.model.eval()
        matrix = _activations(adapter, samples_by_id, ids, batch_size)
        if "projector_sha256" in entry:
            projector = _load_projector(
                checkpoint / "projector.safetensors",
                student_width,
                teacher_width,
                adapter.device,
                torch.float32,
            )
            with torch.no_grad():
                tensor = torch.from_numpy(matrix).to(adapter.device, torch.float32)
                normalized = tensor / torch.sqrt(
                    torch.mean(tensor**2, dim=1, keepdim=True) + 1e-8
                )
                projected = projector(normalized).float().cpu().numpy()
            similarity = representation_similarity(
                matrix, teacher_normalized, projected, cka_teacher=teacher_matrix
            )
        else:
            similarity = {
                "linear_CKA": representation_similarity(
                    matrix, teacher_normalized, teacher_normalized, cka_teacher=teacher_matrix
                )["linear_CKA"],
                "mean_cosine_after_projector": None,
                "projected_hidden_MSE": None,
            }
        rows.append(
            {
                "model": key,
                "regime": str(entry["regime"]),
                "seed": int(entry["seed"]),
                **similarity,
            }
        )
        del adapter
        gc.collect()
        torch.cuda.empty_cache()

    table = pd.DataFrame(rows)
    save_table(table, CAMPAIGN / "representation_similarity_secondary.parquet")
    save_json(
        {
            "analysis": "pre-registered secondary representation similarity",
            "primary_verdict_unchanged": True,
            "access_record": record,
            "n_confirmation_rows": len(ids),
        },
        CAMPAIGN / "representation_similarity_secondary.json",
    )
    pd.set_option("display.width", 250)
    print(table.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
