"""Deterministic bookkeeping for E15 (temporal causal half-life).

Corpus construction, frozen carrier/decision site resolution, batched clean and
edited forwards, and probe plumbing. Scientific intervention math stays in
``interventions.setpoint``; model hooks stay in ``adapters.intervention``.

Everything here is fixed by ``docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_PROTOCOL.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..adapters.intervention import forward_multi_capture, forward_resid_post_edit
from ..data.base import samples_to_dataframe
from ..data.stateful_console import (
    generate_console_samples,
    validate_console_samples,
)
from ..extraction.token_selection import resolve_token_selection
from ..metrics.decoding import classification_metrics
from ..probes.linear import (
    evaluate_probe,
    fit_probe,
    randomized_control_labels,
    raw_probe_direction,
)

# ------------------------------------------------------------------ constants
E15_VERSION = "e15-temporal-causal-half-life-v1"

HORIZONS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
K0 = 1

SITE = "resid_post"
CARRIER_LAYER = 17                     # repository-frozen Qwen3 site; no search
PROPAGATION_LAYERS: tuple[int, ...] = (18, 21, 24, 27)

CORPUS_SPECS: tuple[tuple[str, int, int], ...] = (
    ("train", 600, 20261501),
    ("validation", 200, 20261502),
    ("discovery_test", 150, 20261503),
)

PROBE_SEED = 20261510
DIRECTION_SEED_BASE = 20261520
BOOTSTRAP_SEED = 20261530
PERMUTATION_SEED = 20261540
RANDOM_LABEL_SEEDS: tuple[int, ...] = (0, 1, 2)

N_RANDOM_DIRECTIONS = 5
N_ORTHOGONAL_DIRECTIONS = 5

SMOKE_HORIZONS: tuple[int, ...] = (1, 4, 16)
SMOKE_PAIRS = 30

MIN_DECODABILITY = 0.90
MIN_BEHAVIOR = 0.70
RANDOM_LABEL_BAND = (0.40, 0.60)
NO_OP_TOLERANCE = 1e-6

# Frozen carrier/decision sites (protocol section 2). ``target_span_last`` is
# resolved against the exact line text stored on each sample.
SITE_SPANS: tuple[tuple[str, str], ...] = (
    ("carrier", "target_clearance_line"),
    ("irrelevant_carrier", "irrelevant_clearance_line"),
    ("late_carrier", "late_gap_line"),
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def campaign_dir() -> Path:
    return repo_root() / "runs" / "E15_TEMPORAL"


# ------------------------------------------------------------------- corpus
def build_e15_corpus(
    specs: Sequence[tuple[str, int, int]] = CORPUS_SPECS,
    horizons: Sequence[int] = HORIZONS,
    *,
    distractor_pool: Sequence[str] | None = None,
    namespace_suffix: str = "",
) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
    """Fresh ``e15-{split}-v1`` namespace, pair-complete and deduplicated.

    Every split renders the *same* base episodes at every horizon, so an
    episode's identity, carrier position and nuisance content are constant across
    the horizon grid by construction.
    """
    grid = [int(k) for k in horizons]
    all_samples: list[Any] = []
    statistics: dict[str, Any] = {"horizons": grid, "confirmation_split_exists": False}
    for split_name, n_pairs, seed in specs:
        namespace = f"e15-{split_name}-v1{namespace_suffix}"
        samples = generate_console_samples(
            int(n_pairs),
            int(seed),
            grid,
            namespace=namespace,
            distractor_pool=distractor_pool,
        )
        statistics[split_name] = {
            "seed": int(seed),
            "n_pairs_per_horizon": int(n_pairs),
            "n_samples": len(samples),
            **validate_console_samples(samples),
        }
        all_samples.extend(samples)

    frame = samples_to_dataframe(all_samples)
    prefixes = {f"e15-{name}-v1{namespace_suffix}": name for name, _n, _s in specs}
    frame["split"] = [
        next(value for prefix, value in prefixes.items() if str(sid).startswith(prefix))
        for sid in frame["sample_id"]
    ]
    if frame["sample_id"].duplicated().any():
        raise RuntimeError("E15 corpus contains duplicate sample identities")
    if frame["prompt"].duplicated().any():
        raise RuntimeError("E15 corpus contains duplicate prompts across splits")
    if not bool(frame.groupby("pair_id")["sample_id"].size().eq(2).all()):
        raise RuntimeError("E15 corpus split a counterfactual pair")
    cross = frame.groupby("pair_id")["split"].nunique()
    if int((cross > 1).sum()):
        raise RuntimeError("an E15 pair straddles two splits")
    statistics["n_samples_total"] = len(frame)
    return all_samples, frame, statistics


def corpus_bundle(**kwargs: Any):
    samples, frame, stats = build_e15_corpus(**kwargs)
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    by_id = {str(s.sample_id): s for s in samples}
    return samples, frame, stats, labels, by_id


def horizon_view(frame: pd.DataFrame, split: str, horizon: int) -> pd.DataFrame:
    view = frame[
        (frame["split"].astype(str) == str(split))
        & (frame["horizon"].astype(int) == int(horizon))
    ]
    if view.empty:
        raise RuntimeError(f"no rows for split={split} horizon={horizon}")
    return view.sort_values("sample_id").reset_index(drop=True)


# -------------------------------------------------------------- site resolve
def resolve_sites(adapter, sample: Any) -> dict[str, dict[str, Any]]:
    """Resolve every frozen E15 token site for one sample."""
    resolved: dict[str, dict[str, Any]] = {}
    for name, meta_key in SITE_SPANS:
        token = resolve_token_selection(
            strategy="target_span_last",
            tokenizer=adapter.tokenizer,
            prompt_text=sample.prompt,
            chat_template_used=False,
            target_text=str(sample.metadata[meta_key]),
        )
        resolved[name] = token.as_dict()
    decision = resolve_token_selection(
        strategy="last_prompt",
        tokenizer=adapter.tokenizer,
        prompt_text=sample.prompt,
        chat_template_used=False,
    )
    resolved["decision"] = decision.as_dict()
    order = [resolved[name]["token_index"] for name, _k in SITE_SPANS]
    if not (order[0] != order[1] and max(order) < resolved["decision"]["token_index"]):
        raise RuntimeError(
            f"E15 site ordering violated for {sample.sample_id}: {resolved}"
        )
    return resolved


def resolve_site_table(adapter, samples: Sequence[Any]) -> dict[str, dict[str, dict[str, Any]]]:
    return {str(s.sample_id): resolve_sites(adapter, s) for s in samples}


def capture_specs_for(
    sites: Mapping[str, Mapping[str, Mapping[str, Any]]],
    sample_ids: Sequence[str],
    *,
    decision_layers: Sequence[int],
) -> list[tuple[str, int, list[int]]]:
    """Build the frozen clean-forward capture plan for one batch."""
    ids = list(map(str, sample_ids))

    def idx(name: str) -> list[int]:
        return [int(sites[sid][name]["token_index"]) for sid in ids]

    specs: list[tuple[str, int, list[int]]] = [
        ("carrier", CARRIER_LAYER, idx("carrier")),
        ("irrelevant_carrier", CARRIER_LAYER, idx("irrelevant_carrier")),
        ("late_carrier", CARRIER_LAYER, idx("late_carrier")),
    ]
    for layer in sorted({CARRIER_LAYER, *map(int, decision_layers)}):
        specs.append((f"decision_l{layer}", layer, idx("decision")))
    return specs


# ------------------------------------------------------------------ forwards
def run_clean_batches(
    adapter,
    samples_by_id: Mapping[str, Any],
    sample_ids: Sequence[str],
    sites: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    output_token_ids: Sequence[int],
    decision_layers: Sequence[int],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    """Clean forwards returning decision logits and every frozen site state."""
    ids = list(map(str, sample_ids))
    result: dict[str, dict[str, Any]] = {}
    stride = max(1, int(batch_size))
    for start in range(0, len(ids), stride):
        chunk = ids[start : start + stride]
        out = forward_multi_capture(
            adapter,
            [samples_by_id[sid].prompt for sid in chunk],
            readout_token_indices=[int(sites[sid]["decision"]["token_index"]) for sid in chunk],
            output_token_ids=list(map(int, output_token_ids)),
            capture_specs=capture_specs_for(sites, chunk, decision_layers=decision_layers),
        )
        logits = np.asarray(out["selected_logits"], dtype=np.float64)
        for row, sid in enumerate(chunk):
            result[sid] = {
                "selected_logits": logits[row].copy(),
                "sites": {
                    name: np.asarray(values[row], dtype=np.float64).copy()
                    for name, values in out["captured"].items()
                },
            }
    if set(result) != set(ids):
        raise RuntimeError("clean forward sample identity mismatch")
    return result


def run_edit_batches(
    adapter,
    samples_by_id: Mapping[str, Any],
    sample_ids: Sequence[str],
    sites: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    edit_site: str,
    deltas_by_id: Mapping[str, np.ndarray],
    output_token_ids: Sequence[int],
    propagation_layers: Sequence[int],
    batch_size: int,
) -> dict[str, dict[str, Any]]:
    """One edited condition over the given episodes, with exact row identity."""
    ids = list(map(str, sample_ids))
    result: dict[str, dict[str, Any]] = {}
    stride = max(1, int(batch_size))
    for start in range(0, len(ids), stride):
        chunk = ids[start : start + stride]
        out = forward_resid_post_edit(
            adapter,
            [samples_by_id[sid].prompt for sid in chunk],
            edit_layer=CARRIER_LAYER,
            edit_token_indices=[int(sites[sid][edit_site]["token_index"]) for sid in chunk],
            deltas=np.stack(
                [np.asarray(deltas_by_id[sid], dtype=np.float64) for sid in chunk]
            ),
            readout_token_indices=[
                int(sites[sid]["decision"]["token_index"]) for sid in chunk
            ],
            output_token_ids=list(map(int, output_token_ids)),
            capture_layers=list(map(int, propagation_layers)),
        )
        logits = np.asarray(out["selected_logits"], dtype=np.float64)
        edited = np.asarray(out["edited_carrier_state"], dtype=np.float64)
        for row, sid in enumerate(chunk):
            result[sid] = {
                "selected_logits": logits[row].copy(),
                "edited_carrier_state": edited[row].copy(),
                "propagation": {
                    int(layer): np.asarray(values[row], dtype=np.float64).copy()
                    for layer, values in out["captured"].items()
                },
            }
    if set(result) != set(ids):
        raise RuntimeError("edited forward sample identity mismatch")
    return result


def selected_margin(logits: np.ndarray) -> float:
    """Yes-minus-No decision margin, identical to the E01A convention."""
    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(arr) != 2:
        raise ValueError("E15 expects exactly two selected answer logits")
    return float(arr[0] - arr[1])


# -------------------------------------------------------------------- probes
def fit_site_probe(
    states: Mapping[str, Mapping[str, np.ndarray]],
    frame: pd.DataFrame,
    labels: Mapping[str, int],
    *,
    site_key: str,
    horizon: int,
    c_grid: Sequence[float],
    seed: int,
    label_column: str = "target_label",
) -> dict[str, Any]:
    """Train on ``train``, select C on ``validation``, evaluate on the test split."""

    def block(split: str) -> tuple[list[str], np.ndarray, np.ndarray]:
        view = horizon_view(frame, split, horizon)
        ids = view["sample_id"].astype(str).tolist()
        x = np.stack([np.asarray(states[sid][site_key], dtype=np.float64) for sid in ids])
        if label_column == "target_label":
            y = np.asarray([int(labels[sid]) for sid in ids], dtype=int)
        else:
            y = view[label_column].astype(int).to_numpy()
        return ids, x, y

    _tr_ids, x_train, y_train = block("train")
    _va_ids, x_val, y_val = block("validation")
    test_ids, x_test, y_test = block("discovery_test")
    fit = fit_probe(x_train, y_train, x_val, y_val, c_grid=list(c_grid), seed=int(seed))
    metrics = evaluate_probe(fit, x_test, y_test)

    control_metrics: list[dict[str, Any]] = []
    for control_seed in RANDOM_LABEL_SEEDS:
        y_tr_shuffled, y_va_shuffled, used = randomized_control_labels(
            y_train, y_val, seed=int(seed) + int(control_seed)
        )
        control_fit = fit_probe(
            x_train, y_tr_shuffled, x_val, y_va_shuffled,
            c_grid=list(c_grid), seed=int(seed),
        )
        control_metrics.append(
            {
                "control_seed": int(control_seed),
                "permutation_seeds": list(map(int, used)),
                **evaluate_probe(control_fit, x_test, y_test),
            }
        )

    return {
        "site_key": str(site_key),
        "horizon": int(horizon),
        "label_column": str(label_column),
        "chosen_C": float(fit["chosen_C"]),
        "validation_auroc": float(fit["validation_auroc_best"]),
        "test_metrics": metrics,
        "random_label_controls": control_metrics,
        "direction": raw_probe_direction(fit),
        "test_scores": fit["classifier"].decision_function(
            (x_test - fit["scaler_mean"]) / fit["scaler_scale"]
            if fit.get("standardized") else x_test
        ),
        "test_ids": test_ids,
        "test_labels": y_test,
    }


def probe_metrics_record(result: Mapping[str, Any]) -> dict[str, Any]:
    """JSON-safe view of a probe fit (no arrays, no sklearn objects)."""
    controls = [dict(c) for c in result["random_label_controls"]]
    return {
        "site_key": result["site_key"],
        "horizon": int(result["horizon"]),
        "label_column": result["label_column"],
        "chosen_C": float(result["chosen_C"]),
        "validation_auroc": float(result["validation_auroc"]),
        "test_auroc": result["test_metrics"].get("auroc"),
        "test_auprc": result["test_metrics"].get("auprc"),
        "test_balanced_accuracy": result["test_metrics"].get("balanced_accuracy"),
        "random_label_auroc": [c.get("auroc") for c in controls],
        "n_eval": result["test_metrics"].get("n_eval"),
    }


def behavior_metrics(
    clean: Mapping[str, Mapping[str, Any]],
    labels: Mapping[str, int],
    sample_ids: Sequence[str],
) -> dict[str, Any]:
    """Clean forced-choice accuracy, margin scale and margin-AUROC."""
    ids = list(map(str, sample_ids))
    margins = np.asarray([selected_margin(clean[sid]["selected_logits"]) for sid in ids])
    y = np.asarray([int(labels[sid]) for sid in ids], dtype=int)
    predictions = (margins >= 0.0).astype(int)
    return {
        "n": len(ids),
        "accuracy": float((predictions == y).mean()),
        "margin_mean": float(margins.mean()),
        "margin_abs_mean": float(np.abs(margins).mean()),
        "margin_sd": float(margins.std(ddof=1)),
        "margin_metrics": classification_metrics(y, margins),
    }
