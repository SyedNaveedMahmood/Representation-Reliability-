"""Discovery analysis for the E17 cross-family replication.

E17 is open discovery. The replication criterion reuses E13's frozen ``0.03``
behavioral margin and ``0.10`` component SESOI as *descriptive* thresholds; no
Holm-adjusted or confirmatory claim is made from E17.
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..metrics.causal_organization import matched_profiles
from ..reporting.tables import markdown_table, save_json, save_table
from .e17 import REGIMES, TRAINING_SEEDS, campaign_dir
from .e17_support import reference_paths

logger = logging.getLogger(__name__)

DELTA_B = 0.03
DELTA_C = 0.10


def _job_dir(regime: str, seed: int) -> Path:
    matches = sorted(glob.glob(str(campaign_dir() / "jobs" / f"{regime}_seed_{seed}_*")))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one E17 job for {regime} seed {seed}")
    return Path(matches[0])


def _profile_at(run_dir: Path, step: int) -> pd.DataFrame:
    rows = pd.read_parquet(run_dir / "checkpoints" / f"step_{step:03d}" / "factorial_rows.parquet")
    return matched_profiles(rows)


def _teacher_profile() -> pd.DataFrame:
    rows = pd.read_parquet(reference_paths()["root"] / "teacher" / "factorial_rows.parquet")
    return matched_profiles(rows)


def _r0_profile() -> pd.DataFrame:
    rows = pd.read_parquet(reference_paths()["root"] / "R0" / "factorial_rows.parquet")
    return matched_profiles(rows)


def _component_means(profile: pd.DataFrame) -> dict[str, float]:
    return {name: float(profile[f"{name}_z"].mean()) for name in ("Q", "A", "G")}


def analyze_e17() -> Path:
    """Build the frozen E17 discovery tables and evaluate the replication rule."""
    reference = json.loads(reference_paths()["summary"].read_text(encoding="utf-8"))
    teacher_profile = _teacher_profile()
    teacher_means = _component_means(teacher_profile)
    teacher_b = float(reference["teacher"]["B"]["auroc"])
    teacher_val_b = float(reference["teacher"]["validation_B"])
    r0_means = _component_means(_r0_profile())

    rows: list[dict[str, Any]] = []
    for regime in REGIMES:
        for seed in TRAINING_SEEDS:
            run_dir = _job_dir(regime, int(seed))
            job = json.loads((run_dir / "job_summary.json").read_text(encoding="utf-8"))
            step = int(job["b_matched"]["selected_step"])
            checkpoint = next(c for c in job["checkpoints"] if int(c["step"]) == step)
            quality = job["general_quality"].get(f"step_{step:03d}", {})
            if step == 0:
                student_profile = _r0_profile()
            else:
                student_profile = _profile_at(run_dir, step)
            if student_profile["base_sample_id"].tolist() != (
                teacher_profile["base_sample_id"].tolist()
            ):
                raise RuntimeError("E17 teacher/student rows are not aligned")
            student_means = _component_means(student_profile)
            record: dict[str, Any] = {
                "regime": regime,
                "seed": int(seed),
                "selected_step": step,
                "B": float(checkpoint["B"]["auroc"]),
                "validation_B": float(checkpoint["validation_B"]),
                "delta_B_vs_teacher": float(checkpoint["B"]["auroc"]) - teacher_b,
                "absolute_validation_B_gap": float(job["b_matched"]["absolute_B_gap"]),
                "D_native": float(checkpoint["D_native"]["auroc"]),
                "D_frozen": float(checkpoint["D_frozen_initial_axis"]["auroc"]),
            }
            for name in ("Q", "A", "G"):
                record[f"{name}z"] = student_means[name]
                record[f"delta_{name}z"] = student_means[name] - teacher_means[name]
            similarity = checkpoint.get("representation_similarity", {}).get("discovery_test", {})
            record["linear_CKA"] = similarity.get("linear_CKA")
            record["projected_cosine"] = similarity.get("mean_cosine_after_projector")
            record["projected_MSE"] = similarity.get("projected_hidden_MSE")
            record["wikitext_perplexity"] = (
                float(quality["wikitext"]["perplexity"]) if quality else float("nan")
            )
            record["hellaswag_accuracy"] = (
                float(quality["hellaswag"]["accuracy"]) if quality else float("nan")
            )
            rows.append(record)
    results = pd.DataFrame(rows)
    save_table(results, campaign_dir() / "e17_b_matched_results.parquet")

    r0_quality = reference["R0_quality"]
    replication: dict[str, Any] = {}
    for regime in REGIMES:
        block = results.loc[results["regime"].eq(regime)]
        seeds_behavior = int((block["delta_B_vs_teacher"] > -DELTA_B).sum())
        component_hits = {}
        for name in ("Q", "A", "G"):
            gaps = block[f"delta_{name}z"].to_numpy(float)
            beyond = np.abs(gaps) >= DELTA_C
            positive = int(np.sum(beyond & (gaps > 0)))
            negative = int(np.sum(beyond & (gaps < 0)))
            component_hits[name] = {
                "seeds_beyond_sesoi": int(beyond.sum()),
                "max_same_direction": max(positive, negative),
                "direction": 1 if positive >= negative else -1,
                "mean_gap": float(np.mean(gaps)),
            }
        quality_ok = bool(
            block["wikitext_perplexity"].lt(10.0 * float(r0_quality["wikitext"]["perplexity"])).all()
            and block["hellaswag_accuracy"].ge(
                float(r0_quality["hellaswag"]["accuracy"]) - 0.20
            ).all()
        )
        criteria = {
            "A_representation_availability": bool(block["D_native"].ge(0.95).all()),
            "B_behavioral_similarity_2of3": seeds_behavior >= 2,
            "C_causal_mismatch_same_direction_2of3": any(
                item["max_same_direction"] >= 2 for item in component_hits.values()
            ),
            "D_no_catastrophic_specialization": quality_ok,
        }
        replication[regime] = {
            **criteria,
            "replicated": all(criteria.values()),
            "seeds_behaviorally_similar": seeds_behavior,
            "components": component_hits,
            "mismatched_components": [
                name
                for name, item in component_hits.items()
                if item["max_same_direction"] >= 2
            ],
        }
    any_replicated = any(item["replicated"] for item in replication.values())
    verdict = {
        "version": reference["version"],
        "pair": reference["selected_pair"],
        "teacher_layer": reference["teacher_layer"],
        "student_layer": reference["student_layer"],
        "teacher_B": teacher_b,
        "teacher_validation_B": teacher_val_b,
        "teacher_component_means": teacher_means,
        "R0_component_means": r0_means,
        "delta_B": DELTA_B,
        "delta_C": DELTA_C,
        "replication": replication,
        "phenomenon_replicated": any_replicated,
        "discovery_not_confirmation": True,
        "confirmation_accessed": False,
    }
    save_json(verdict, campaign_dir() / "e17_replication_verdict.json")

    display = [
        "regime", "seed", "selected_step", "B", "delta_B_vs_teacher", "D_native",
        "Qz", "delta_Qz", "Az", "delta_Az", "Gz", "delta_Gz",
        "linear_CKA", "wikitext_perplexity", "hellaswag_accuracy",
    ]
    report = Path(campaign_dir().parents[1]) / "E17_CROSS_FAMILY_DISCOVERY_SUMMARY.md"
    lines = [
        "# E17 Cross-Family Causal-Organization Discovery",
        "",
        "Status: open discovery. E13 confirmation was not accessed or re-evaluated.",
        "",
        (
            f"Pair: `{reference['selected_pair']['teacher_id']}` -> "
            f"`{reference['selected_pair']['student_id']}`."
        ),
        (
            f"Frozen relative depth `r = 0.60`; teacher layer "
            f"`{reference['teacher_layer']}`, student layer `{reference['student_layer']}`."
        ),
        f"Teacher B `{teacher_b:.6f}`, validation B `{teacher_val_b:.6f}`.",
        (
            f"Teacher component means: Qz `{teacher_means['Q']:.6f}`, "
            f"Az `{teacher_means['A']:.6f}`, Gz `{teacher_means['G']:.6f}`."
        ),
        (
            f"R0 component means: Qz `{r0_means['Q']:.6f}`, "
            f"Az `{r0_means['A']:.6f}`, Gz `{r0_means['G']:.6f}`."
        ),
        "",
        "## Behavior-matched results",
        "",
        markdown_table(results[display], float_fmt="{:.6f}"),
        "",
        "## Replication criterion",
        "",
        "```json",
        json.dumps(replication, indent=2, sort_keys=True),
        "```",
        "",
        f"Phenomenon replicated: **{any_replicated}**.",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


__all__ = ["analyze_e17"]
