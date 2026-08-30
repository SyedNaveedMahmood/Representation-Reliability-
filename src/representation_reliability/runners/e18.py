"""E18 causal-read localisation for a delayed state decision.

Frozen by `docs/E18_CAUSAL_READ_LOCALISATION_PROTOCOL.md`.

E15's Gate 1 showed its predeclared carrier is not causally sufficient. Before any
redesign, one prerequisite question must be answered:

    for a delayed decision that provably depends on a remembered binary state,
    at which token sites and layers does a full-state counterfactual replacement
    actually change that decision?

E18 measures that on a declared 6-site by 8-layer grid and reports **every** cell.
It is a map, not a hypothesis test, and it is not a rescue of E15.
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from ..adapters.intervention import (
    forward_multi_capture,
    forward_resid_post_span_edit,
    resid_post_hook_count,
)
from ..config import config_hash, resolve_config, save_resolved_config
from ..data.stateful_console import DENIED, GRANTED
from ..interventions.truth_coordinate import random_unit_direction
from ..metrics.causal import (
    cluster_bootstrap_mean_ci,
    counterfactual_outcome,
    margin_toward_label,
)
from ..probes.linear import evaluate_probe, fit_probe
from ..reporting.tables import save_json, save_table
from ..runtime.manifest import RunManifest, dataset_split_hash, prompt_hash
from ..runtime.status import StatusFile, atomic_write_json
from .e00c import candidate_first_token_ids
from .e01a import _prediction
from .e15_support import build_e15_corpus, horizon_view, selected_margin
from .extract import load_adapter

logger = logging.getLogger(__name__)

E18_VERSION = "e18-causal-read-localisation-v1"

MODEL = "qwen3_1.7b"
CORPUS_SPECS: tuple[tuple[str, int, int], ...] = (
    ("train", 400, 20261801),
    ("validation", 150, 20261802),
    ("discovery_test", 150, 20261803),
)
HORIZONS: tuple[int, ...] = (1, 8)
K0 = 1
SECONDARY_HORIZON = 8

SITES: tuple[str, ...] = (
    "state_word_last",
    "carrier",
    "clearance_line_span",
    "request_step_last",
    "decision",
    "prefix_span",
)
ANCHOR_SITE = "prefix_span"
LAYERS: tuple[int, ...] = (0, 4, 8, 12, 17, 21, 24, 27)

N_RANDOM_DIRECTIONS = 3
DIRECTION_SEED_BASE = 20261820
PROBE_SEED = 20261810
BOOTSTRAP_SEED = 20261830

NO_OP_TOLERANCE = 1e-6
STRONG_FLIP_RATE = 0.50
PARTIAL_FLIP_RATE = 0.10


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def campaign_dir() -> Path:
    path = repo_root() / "runs" / "E18_LOCALISATION"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config():
    root = repo_root()
    return resolve_config(
        base_path=root / "configs/base.yaml",
        model_path=root / f"configs/models/{MODEL}.yaml",
        experiment_path=root / "configs/experiments/E18_causal_read_localisation.yaml",
        overrides=(),
    )


# ----------------------------------------------------------------- site spans
def prompt_char_spans(sample: Any) -> dict[str, tuple[int, int]]:
    """Character spans of every frozen E18 site, from the fixed line structure.

    The prompt is::

        [0] header
        [1] prefix distractor step
        [2] clearance step
        [3] clearance step
        [4..] gap distractor steps
        [-3] request step
        [-2] question line
        [-1] "Answer:"

    The header contains the literal word GRANTED, so the state word is located
    inside the target clearance line's own character range and never by a bare
    prompt search.
    """
    prompt = str(sample.prompt)
    lines = prompt.split("\n")
    if len(lines) < 7:
        raise RuntimeError(f"unexpected prompt shape for {sample.sample_id}")
    target_line = str(sample.metadata["target_clearance_line"])
    if target_line not in (lines[2], lines[3]):
        raise RuntimeError(
            f"target clearance line is not in the clearance block for {sample.sample_id}"
        )

    starts: list[int] = []
    cursor = 0
    for line in lines:
        starts.append(cursor)
        cursor += len(line) + 1

    target_index = 2 if target_line == lines[2] else 3
    target_start = starts[target_index]
    target_end = target_start + len(target_line)

    state_word = str(sample.metadata["target_state"])
    if state_word not in (GRANTED, DENIED):
        raise RuntimeError(f"unknown clearance state for {sample.sample_id}")
    offset = target_line.find(state_word)
    if offset < 0 or target_line.count(state_word) != 1:
        raise RuntimeError(f"state word is not uniquely inside its line for {sample.sample_id}")

    request_start = starts[len(lines) - 3]
    request_end = request_start + len(lines[len(lines) - 3])
    clearance_block_end = starts[3] + len(lines[3])

    return {
        "state_word_last": (target_start + offset, target_start + offset + len(state_word)),
        "carrier": (target_start, target_end),
        "clearance_line_span": (target_start, target_end),
        "request_step_last": (request_start, request_end),
        "decision": (len(prompt) - 1, len(prompt)),
        "prefix_span": (0, clearance_block_end),
    }


SPAN_SITES = frozenset({"clearance_line_span", "prefix_span"})


def resolve_site_tokens(tokenizer, sample: Any) -> dict[str, list[int]]:
    """Token positions for every frozen site.

    Single-token sites take the LAST token whose start lies inside the span;
    span sites take every such token.
    """
    prompt = str(sample.prompt)
    enc = tokenizer(prompt, add_special_tokens=True, return_offsets_mapping=True)
    offsets = [tuple(map(int, pair)) for pair in enc["offset_mapping"]]
    final_index = len(enc["input_ids"]) - 1
    spans = prompt_char_spans(sample)
    resolved: dict[str, list[int]] = {}
    for name, (start, end) in spans.items():
        if name == "decision":
            # The decision site is DEFINED positionally as the final prompt
            # token, so it is resolved directly rather than through a character
            # span. Deriving it from characters would make a positional
            # definition depend on how the tokenizer happens to split
            # "Answer:".
            resolved[name] = [final_index]
            continue
        inside = [
            index
            for index, (a, b) in enumerate(offsets)
            if b > a and start <= a < end
        ]
        if not inside:
            raise RuntimeError(f"site {name} resolved to no tokens for {sample.sample_id}")
        resolved[name] = inside if name in SPAN_SITES else [inside[-1]]
    resolved["_readout"] = [final_index]
    for name in SITES:
        if name == "decision":
            continue
        if max(resolved[name]) >= final_index:
            raise RuntimeError(
                f"site {name} reaches the decision token for {sample.sample_id}"
            )
    return resolved


def _site_table(tokenizer, samples) -> dict[str, dict[str, list[int]]]:
    return {str(s.sample_id): resolve_site_tokens(tokenizer, s) for s in samples}


# -------------------------------------------------------------------- forwards
def _clean_states(
    adapter, samples_by_id, ids, sites, *, layer: int, output_token_ids, batch_size: int
) -> dict[str, dict[str, Any]]:
    """One clean forward per batch capturing every site at one layer.

    Span sites are captured position by position, so a span's full state is
    available for a counterfactual replacement.
    """
    result: dict[str, dict[str, Any]] = {}
    stride = max(1, int(batch_size))
    max_span = {
        name: max(len(sites[sid][name]) for sid in ids)
        for name in SITES
    }
    for start in range(0, len(ids), stride):
        chunk = ids[start : start + stride]
        specs: list[tuple[str, int, list[int]]] = []
        for name in SITES:
            for slot in range(max_span[name]):
                # Rows whose span is shorter reuse their last position; those
                # captures are discarded below, never used as evidence.
                specs.append(
                    (
                        f"{name}#{slot}",
                        int(layer),
                        [
                            sites[sid][name][min(slot, len(sites[sid][name]) - 1)]
                            for sid in chunk
                        ],
                    )
                )
        out = forward_multi_capture(
            adapter,
            [samples_by_id[sid].prompt for sid in chunk],
            readout_token_indices=[sites[sid]["_readout"][0] for sid in chunk],
            output_token_ids=list(map(int, output_token_ids)),
            capture_specs=specs,
        )
        logits = np.asarray(out["selected_logits"], dtype=np.float64)
        for row, sid in enumerate(chunk):
            per_site: dict[str, np.ndarray] = {}
            for name in SITES:
                span = len(sites[sid][name])
                per_site[name] = np.stack(
                    [
                        np.asarray(out["captured"][f"{name}#{slot}"][row], dtype=np.float64)
                        for slot in range(span)
                    ]
                )
            result[sid] = {"selected_logits": logits[row].copy(), "sites": per_site}
    if set(result) != set(ids):
        raise RuntimeError("clean forward sample identity mismatch")
    return result


def _twin_map(view: pd.DataFrame) -> dict[str, str]:
    by_pair: dict[str, list[str]] = {}
    for row in view.to_dict("records"):
        by_pair.setdefault(str(row["pair_id"]), []).append(str(row["sample_id"]))
    twins: dict[str, str] = {}
    for pair_id, ids in by_pair.items():
        if len(ids) != 2:
            raise RuntimeError(f"pair {pair_id} is incomplete")
        first, second = sorted(ids)
        twins[first] = second
        twins[second] = first
    return twins


def _run_span_arm(
    adapter, samples_by_id, ids, sites, *, site: str, layer: int,
    deltas_by_id: dict[str, np.ndarray], output_token_ids, batch_size: int,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    stride = max(1, int(batch_size))
    for start in range(0, len(ids), stride):
        chunk = ids[start : start + stride]
        out = forward_resid_post_span_edit(
            adapter,
            [samples_by_id[sid].prompt for sid in chunk],
            edit_layer=int(layer),
            edit_token_indices=[sites[sid][site] for sid in chunk],
            deltas=[deltas_by_id[sid] for sid in chunk],
            readout_token_indices=[sites[sid]["_readout"][0] for sid in chunk],
            output_token_ids=list(map(int, output_token_ids)),
        )
        logits = np.asarray(out["selected_logits"], dtype=np.float64)
        for row, sid in enumerate(chunk):
            result[sid] = {
                "selected_logits": logits[row].copy(),
                "edited_states": np.asarray(out["edited_states"][row], dtype=np.float64),
            }
    if set(result) != set(ids):
        raise RuntimeError("span arm sample identity mismatch")
    return result


# --------------------------------------------------------------------- grading
def grade_cell(flip_rate: float, effect_ci_excludes_zero: bool, beats_random: bool) -> str:
    """Frozen strength scale from protocol section 8."""
    if not (effect_ci_excludes_zero and beats_random):
        return "WEAK"
    if float(flip_rate) >= STRONG_FLIP_RATE:
        return "STRONG"
    if float(flip_rate) >= PARTIAL_FLIP_RATE:
        return "PARTIAL"
    return "WEAK"


def _corpus():
    samples, frame, stats = build_e15_corpus(
        specs=CORPUS_SPECS, horizons=HORIZONS, namespace_suffix="-e18"
    )
    labels = dict(zip(frame["sample_id"].astype(str), frame["target_label"].astype(int)))
    by_id = {str(s.sample_id): s for s in samples}
    return samples, frame, stats, labels, by_id


def _sweep_horizon(
    adapter, cfg, *, frame, by_id, labels, sites, output_token_ids,
    horizon: int, cells: list[tuple[str, int]] | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run every requested (site, layer) cell at one horizon."""
    view = horizon_view(frame, "discovery_test", int(horizon))
    ids = view["sample_id"].astype(str).tolist()
    meta = view.set_index("sample_id")
    twins = _twin_map(view)
    batch_size = int(cfg.runtime.batch_size)
    hidden = int(adapter.hidden_size)

    requested = cells if cells is not None else [(s, l) for l in LAYERS for s in SITES]
    layers_needed = sorted({int(l) for _s, l in requested})
    rows: list[dict[str, Any]] = []
    diagnostics = {
        "no_op_max_margin_deviation": 0.0,
        "max_norm_match_relative_deviation": 0.0,
        "hook_leak_detected": False,
    }

    for layer in layers_needed:
        clean = _clean_states(
            adapter, by_id, ids, sites,
            layer=layer, output_token_ids=output_token_ids, batch_size=batch_size,
        )
        clean_margin = {sid: selected_margin(clean[sid]["selected_logits"]) for sid in ids}

        # no_op once per layer; it does not depend on the site.
        zero = {sid: np.zeros((len(sites[sid]["carrier"]), hidden)) for sid in ids}
        no_op = _run_span_arm(
            adapter, by_id, ids, sites, site="carrier", layer=layer,
            deltas_by_id=zero, output_token_ids=output_token_ids, batch_size=batch_size,
        )
        for sid in ids:
            diagnostics["no_op_max_margin_deviation"] = max(
                diagnostics["no_op_max_margin_deviation"],
                abs(selected_margin(no_op[sid]["selected_logits"]) - clean_margin[sid]),
            )
        del no_op

        for site in [s for s, l in requested if int(l) == layer]:
            patch = {
                sid: clean[twins[sid]]["sites"][site] - clean[sid]["sites"][site]
                for sid in ids
            }
            arms: list[dict[str, Any]] = [
                {"condition": "full_state_patch", "index": -1, "deltas": patch}
            ]
            for i in range(N_RANDOM_DIRECTIONS):
                control: dict[str, np.ndarray] = {}
                for sid in ids:
                    block = patch[sid]
                    scaled = np.empty_like(block)
                    for position in range(block.shape[0]):
                        direction = random_unit_direction(
                            hidden, DIRECTION_SEED_BASE + 1000 * i + position
                        )
                        scaled[position] = float(np.linalg.norm(block[position])) * direction
                    control[sid] = scaled
                arms.append(
                    {"condition": "random_norm_matched", "index": i, "deltas": control}
                )

            for arm in arms:
                edited = _run_span_arm(
                    adapter, by_id, ids, sites, site=site, layer=layer,
                    deltas_by_id=arm["deltas"], output_token_ids=output_token_ids,
                    batch_size=batch_size,
                )
                for sid in ids:
                    block = np.asarray(arm["deltas"][sid], dtype=np.float64)
                    base = clean[sid]["sites"][site]
                    margin_before = clean_margin[sid]
                    margin_after = selected_margin(edited[sid]["selected_logits"])
                    label = int(labels[sid])
                    expected = 1 - label
                    outcome = counterfactual_outcome(
                        _prediction(margin_before), _prediction(margin_after), expected
                    )
                    delta_norm = float(np.linalg.norm(block))
                    activation_norm = float(np.linalg.norm(base))
                    if arm["condition"] == "random_norm_matched":
                        reference = float(np.linalg.norm(patch[sid]))
                        diagnostics["max_norm_match_relative_deviation"] = max(
                            diagnostics["max_norm_match_relative_deviation"],
                            abs(delta_norm - reference) / max(reference, 1e-12),
                        )
                    rows.append({
                        "horizon": int(horizon),
                        "site": site,
                        "layer": int(layer),
                        "condition": str(arm["condition"]),
                        "direction_index": int(arm["index"]),
                        "n_positions": int(block.shape[0]),
                        "base_sample_id": sid,
                        "pair_id": str(meta.loc[sid, "pair_id"]),
                        "target_label": label,
                        "expected_label": expected,
                        "margin_before": margin_before,
                        "margin_after": margin_after,
                        "delta_margin_toward_expected": float(
                            margin_toward_label(margin_after, expected)
                            - margin_toward_label(margin_before, expected)
                        ),
                        "prediction_before": _prediction(margin_before),
                        "prediction_after": _prediction(margin_after),
                        "counterfactual_flip": outcome["counterfactual_flip"],
                        "expected_label_after": outcome["expected_label_after"],
                        "delta_norm": delta_norm,
                        "activation_norm": activation_norm,
                        "delta_over_activation_norm": delta_norm / max(activation_norm, 1e-12),
                    })
                del edited
        del clean
        gc.collect()

    diagnostics["hook_leak_detected"] = any(
        resid_post_hook_count(adapter, layer=layer) for layer in layers_needed
    )
    return pd.DataFrame(rows), diagnostics


def _grade_table(rows: pd.DataFrame, *, bootstraps: int, confidence: float) -> pd.DataFrame:
    """Cell-level effect, flip rate, control contrast and frozen grade."""
    records: list[dict[str, Any]] = []
    for (horizon, site, layer), block in rows.groupby(["horizon", "site", "layer"], sort=True):
        patch = block[block["condition"] == "full_state_patch"]
        control = block[block["condition"] == "random_norm_matched"]
        if patch.empty or control.empty:
            raise RuntimeError(f"incomplete cell {site}/L{layer}")
        seed = BOOTSTRAP_SEED + 7 * int(layer) + len(str(site))
        effect = cluster_bootstrap_mean_ci(
            patch["delta_margin_toward_expected"].to_numpy(float),
            patch["pair_id"].astype(str).tolist(),
            n_bootstraps=bootstraps, confidence_level=confidence, seed=seed,
        )
        flip = cluster_bootstrap_mean_ci(
            patch["counterfactual_flip"].to_numpy(float),
            patch["pair_id"].astype(str).tolist(),
            n_bootstraps=bootstraps, confidence_level=confidence, seed=seed + 1,
        )
        keys = ["base_sample_id", "pair_id"]
        merged = (
            patch.groupby(keys, as_index=False)["delta_margin_toward_expected"].mean()
            .merge(
                control.groupby(keys, as_index=False)["delta_margin_toward_expected"].mean(),
                on=keys, suffixes=("_p", "_c"), how="inner",
            )
        )
        contrast = cluster_bootstrap_mean_ci(
            (
                merged["delta_margin_toward_expected_p"].to_numpy(float)
                - merged["delta_margin_toward_expected_c"].to_numpy(float)
            ),
            merged["pair_id"].astype(str).tolist(),
            n_bootstraps=bootstraps, confidence_level=confidence, seed=seed + 2,
        )
        effect_excludes = bool(effect["ci_low"] > 0.0 or effect["ci_high"] < 0.0)
        beats_random = bool(contrast["ci_low"] > 0.0)
        records.append({
            "horizon": int(horizon),
            "site": str(site),
            "layer": int(layer),
            "n_positions": int(patch["n_positions"].mean()),
            "effect": effect["mean"],
            "effect_ci_low": effect["ci_low"],
            "effect_ci_high": effect["ci_high"],
            "effect_ci_excludes_zero": effect_excludes,
            "flip_rate": flip["mean"],
            "flip_ci_low": flip["ci_low"],
            "flip_ci_high": flip["ci_high"],
            "random_effect": float(control["delta_margin_toward_expected"].mean()),
            "patch_minus_random": contrast["mean"],
            "patch_minus_random_ci_low": contrast["ci_low"],
            "patch_minus_random_ci_high": contrast["ci_high"],
            "beats_random": beats_random,
            "mean_delta_over_activation_norm": float(
                patch["delta_over_activation_norm"].mean()
            ),
            "grade": grade_cell(flip["mean"], effect_excludes, beats_random),
        })
    return pd.DataFrame(records).sort_values(["horizon", "site", "layer"]).reset_index(drop=True)


def _decodability(
    adapter, cfg, *, frame, by_id, labels, sites, output_token_ids, horizon: int
) -> list[dict[str, Any]]:
    """Secondary: probe for z at every (site, layer) cell. Span sites mean-pool."""
    records: list[dict[str, Any]] = []
    split_ids = {
        split: horizon_view(frame, split, int(horizon))["sample_id"].astype(str).tolist()
        for split in ("train", "validation", "discovery_test")
    }
    every_id = [sid for split in ("train", "validation", "discovery_test") for sid in split_ids[split]]
    for layer in LAYERS:
        clean = _clean_states(
            adapter, by_id, every_id, sites,
            layer=layer, output_token_ids=output_token_ids,
            batch_size=int(cfg.runtime.batch_size),
        )
        def block(split: str, states, site_name: str):
            ids = split_ids[split]
            x = np.stack([states[sid]["sites"][site_name].mean(axis=0) for sid in ids])
            y = np.asarray([int(labels[sid]) for sid in ids], dtype=int)
            return x, y

        for site in SITES:
            x_tr, y_tr = block("train", clean, site)
            x_va, y_va = block("validation", clean, site)
            x_te, y_te = block("discovery_test", clean, site)
            fit = fit_probe(x_tr, y_tr, x_va, y_va, c_grid=cfg.probe.C_grid, seed=PROBE_SEED)
            metrics = evaluate_probe(fit, x_te, y_te)
            records.append({
                "horizon": int(horizon),
                "site": site,
                "layer": int(layer),
                "chosen_C": float(fit["chosen_C"]),
                "validation_auroc": float(fit["validation_auroc_best"]),
                "test_auroc": metrics.get("auroc"),
                "test_balanced_accuracy": metrics.get("balanced_accuracy"),
            })
        del clean
        gc.collect()
    return records


def run_e18() -> Path:
    """Execute the frozen E18 localisation map."""
    cfg, provenance = _config()
    run_dir = campaign_dir() / MODEL
    run_dir.mkdir(parents=True, exist_ok=True)
    status = StatusFile.create(run_dir, run_id=f"E18-{MODEL}", experiment_id="E18")
    manifest = RunManifest(run_dir)
    adapter = None
    try:
        samples, frame, stats, labels, by_id = _corpus()
        manifest.set_start(
            config_hash(cfg),
            {**provenance, "e18_version": E18_VERSION, "sites": list(SITES),
             "layers": list(LAYERS), "horizons": list(HORIZONS)},
            {"probe": PROBE_SEED, "direction_base": DIRECTION_SEED_BASE,
             "bootstrap": BOOTSTRAP_SEED,
             "corpus": {n: s for n, _c, s in CORPUS_SPECS}},
        )
        save_resolved_config(cfg, run_dir / "resolved_config.yaml", provenance)

        adapter = load_adapter(cfg)
        if max(LAYERS) >= adapter.num_layers:
            raise RuntimeError(f"frozen layers exceed {adapter.num_layers} blocks")
        output_token_ids = candidate_first_token_ids(adapter, cfg.behavior.candidates_primary)
        sites = _site_table(adapter.tokenizer, samples)
        manifest.update_model_info(
            id=adapter.display_model_id, dtype=str(cfg.model.dtype),
            num_layers=adapter.num_layers, hidden_size=adapter.hidden_size,
            candidate_token_ids=output_token_ids,
        )
        manifest.update_dataset_info(
            split_hash=dataset_split_hash(
                dict(zip(frame["sample_id"].astype(str), frame["split"].astype(str)))
            ),
            prompt_hash_sample=prompt_hash(str(frame["prompt"].iloc[0])),
            confirmation_accessed=False,
        )
        save_json(stats, run_dir / "corpus_statistics.json")
        example = str(frame["sample_id"].iloc[0])
        save_json(
            {"sample_id": example, "spans": {k: list(v) for k, v in sites[example].items()}},
            run_dir / "example_sites.json",
        )

        bootstraps = int(cfg.statistics.bootstrap_samples)
        confidence = float(cfg.statistics.confidence_level)

        rows_k0, diag_k0 = _sweep_horizon(
            adapter, cfg, frame=frame, by_id=by_id, labels=labels, sites=sites,
            output_token_ids=output_token_ids, horizon=K0, cells=None,
        )
        grades_k0 = _grade_table(rows_k0, bootstraps=bootstraps, confidence=confidence)

        # G2: the anchor must be STRONG somewhere or the map is uninterpretable.
        anchor = grades_k0[grades_k0["site"] == ANCHOR_SITE]
        anchor_strong = bool((anchor["grade"] == "STRONG").any())

        # Frozen conditional second pass at the secondary horizon.
        promoted = [
            (str(r["site"]), int(r["layer"]))
            for r in grades_k0.to_dict("records")
            if r["grade"] in ("STRONG", "PARTIAL")
        ]
        rows_k8 = pd.DataFrame()
        grades_k8 = pd.DataFrame()
        diag_k8: dict[str, Any] = {}
        if promoted:
            rows_k8, diag_k8 = _sweep_horizon(
                adapter, cfg, frame=frame, by_id=by_id, labels=labels, sites=sites,
                output_token_ids=output_token_ids, horizon=SECONDARY_HORIZON, cells=promoted,
            )
            grades_k8 = _grade_table(rows_k8, bootstraps=bootstraps, confidence=confidence)

        decodability = _decodability(
            adapter, cfg, frame=frame, by_id=by_id, labels=labels, sites=sites,
            output_token_ids=output_token_ids, horizon=K0,
        )

        all_rows = pd.concat([r for r in (rows_k0, rows_k8) if not r.empty], ignore_index=True)
        all_grades = pd.concat(
            [g for g in (grades_k0, grades_k8) if not g.empty], ignore_index=True
        )
        save_table(all_rows, run_dir / "localisation_rows.parquet")
        save_table(all_grades, run_dir / "localisation_grades.parquet")
        save_table(pd.DataFrame(decodability), run_dir / "decodability.parquet")

        numerics_ok = bool(
            diag_k0["no_op_max_margin_deviation"] <= NO_OP_TOLERANCE
            and diag_k0["max_norm_match_relative_deviation"] <= 1e-6
            and not diag_k0["hook_leak_detected"]
        )
        strong = [
            f"{r['site']}@L{r['layer']}"
            for r in grades_k0.to_dict("records") if r["grade"] == "STRONG"
        ]
        partial = [
            f"{r['site']}@L{r['layer']}"
            for r in grades_k0.to_dict("records") if r["grade"] == "PARTIAL"
        ]
        single_token_strong = sorted(
            {
                r["site"] for r in grades_k0.to_dict("records")
                if r["grade"] == "STRONG" and r["site"] not in SPAN_SITES
            }
        )
        if not anchor_strong:
            outcome = "measurement_invalid_anchor_not_strong"
        elif single_token_strong:
            outcome = "single_token_carrier_exists"
        elif "clearance_line_span" in {
            r["site"] for r in grades_k0.to_dict("records") if r["grade"] == "STRONG"
        }:
            outcome = "read_is_distributed_across_the_state_clause"
        else:
            outcome = "no_localised_read_transplant_bottleneck_required"

        summary = {
            "e18_version": E18_VERSION,
            "model": adapter.display_model_id,
            "sites": list(SITES),
            "layers": list(LAYERS),
            "horizon_primary": K0,
            "horizon_secondary": SECONDARY_HORIZON,
            "gates": {
                "G1_numerics_passed": numerics_ok,
                "no_op_max_margin_deviation": diag_k0["no_op_max_margin_deviation"],
                "max_norm_match_relative_deviation": diag_k0["max_norm_match_relative_deviation"],
                "hook_leak_detected": diag_k0["hook_leak_detected"],
                "G2_anchor_strong_somewhere": anchor_strong,
            },
            "strong_cells_k0": strong,
            "partial_cells_k0": partial,
            "single_token_sites_strong_k0": single_token_strong,
            "promoted_to_secondary_horizon": [f"{s}@L{l}" for s, l in promoted],
            "outcome": outcome,
            "grades_k0": grades_k0.to_dict("records"),
            "grades_secondary": grades_k8.to_dict("records") if not grades_k8.empty else [],
            "decodability": decodability,
            "diagnostics": {"k0": diag_k0, "secondary": diag_k8},
            "confirmation_accessed": False,
        }
        save_json(summary, run_dir / "e18_summary.json")
        atomic_write_json(campaign_dir() / "E18_LOCALISATION.json", summary)
        manifest.finish([{"stage": "e18", "outcome": outcome}])
        if not numerics_ok:
            status.fail("E18 numerics gate G1 failed")
            raise RuntimeError(f"E18 G1 failed: {summary['gates']}")
        status.complete(f"E18 localisation complete: {outcome}")
        return run_dir
    except Exception as exc:
        if status.state_name == "running":
            status.fail(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if adapter is not None:
            del adapter
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def read_e18() -> dict[str, Any] | None:
    path = campaign_dir() / "E18_LOCALISATION.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
