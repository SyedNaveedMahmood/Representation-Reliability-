"""Scale-robust causal-organization metrics for E13 discovery."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

PROFILE_COMPONENTS = ("Q_z", "A_z", "G_z")


def validation_margin_statistics(
    margins: Iterable[float], labels: Iterable[int]
) -> dict[str, float]:
    """Compute checkpoint scale/calibration diagnostics from validation only."""
    margin = np.asarray(list(margins), dtype=np.float64)
    target = np.asarray(list(labels), dtype=np.int64)
    if margin.ndim != 1 or target.shape != margin.shape or not len(margin):
        raise ValueError("validation margins and labels must be aligned nonempty vectors")
    if not np.isfinite(margin).all() or not np.isin(target, (0, 1)).all():
        raise ValueError("validation margins/labels contain invalid values")
    sigma = float(np.std(margin, ddof=0))
    if not math.isfinite(sigma) or sigma <= 1e-8:
        raise ValueError("validation-only clean-margin scale is degenerate")
    oriented = np.where(target == 1, margin, -margin)
    probability = 1.0 / (1.0 + np.exp(-np.clip(margin, -80.0, 80.0)))
    entropy = -(probability * np.log(np.clip(probability, 1e-15, 1.0)))
    entropy -= (1.0 - probability) * np.log(
        np.clip(1.0 - probability, 1e-15, 1.0)
    )
    ce = np.logaddexp(0.0, -oriented)
    return {
        "source_split": "validation",
        "n": len(margin),
        "sigma_margin_validation": sigma,
        "validation_ce": float(np.mean(ce)),
        "clean_output_entropy": float(np.mean(entropy)),
        "mean_abs_yes_no_margin": float(np.mean(np.abs(margin))),
        "margin_sd": sigma,
    }


def factorial_effect_views(
    y00: Any, y10: Any, y01: Any, y11: Any, *, sigma_margin_validation: float
) -> dict[str, np.ndarray]:
    """Return raw, validation-z, probability, and strict-flip factorial views."""
    arrays = [np.asarray(value, dtype=np.float64) for value in (y00, y10, y01, y11)]
    if not all(array.shape == arrays[0].shape for array in arrays):
        raise ValueError("factorial arms must have identical shapes")
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("factorial arms must be finite")
    sigma = float(sigma_margin_validation)
    if not math.isfinite(sigma) or sigma <= 1e-8:
        raise ValueError("validation margin scale must be finite and positive")
    y0, yq, yc, yqc = arrays
    q_raw = yq - y0
    a_raw = yc - y0
    g_raw = (yqc - yq) - (yc - y0)
    probability = [1.0 / (1.0 + np.exp(-np.clip(v, -80.0, 80.0))) for v in arrays]
    p0, pq, pc, pqc = probability
    q_prob = pq - p0
    a_prob = pc - p0
    g_prob = (pqc - pq) - (pc - p0)
    clean_not_target = y0 <= 0.0
    q_flip = clean_not_target & (yq > 0.0)
    context_flip = clean_not_target & (yc > 0.0)
    joint_flip = clean_not_target & (yqc > 0.0)
    return {
        "Q_raw": q_raw,
        "A_raw": a_raw,
        "G_raw": g_raw,
        "Q_z": q_raw / sigma,
        "A_z": a_raw / sigma,
        "G_z": g_raw / sigma,
        "Q_prob": q_prob,
        "A_prob": a_prob,
        "G_prob": g_prob,
        "p00_target": p0,
        "p10_target": pq,
        "p01_target": pc,
        "p11_target": pqc,
        "q_target_flip": q_flip.astype(np.int8),
        "context_target_flip": context_flip.astype(np.int8),
        "joint_target_flip": joint_flip.astype(np.int8),
        "clean_target_prediction": (y0 > 0.0).astype(np.int8),
        "q_target_prediction": (yq > 0.0).astype(np.int8),
        "context_target_prediction": (yc > 0.0).astype(np.int8),
        "joint_target_prediction": (yqc > 0.0).astype(np.int8),
    }


def add_factorial_effect_views(
    rows: pd.DataFrame, *, sigma_margin_validation: float
) -> pd.DataFrame:
    required = {"Y00", "Y10", "Y01", "Y11"}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"factorial rows missing columns: {missing}")
    result = rows.copy()
    views = factorial_effect_views(
        result["Y00"].to_numpy(),
        result["Y10"].to_numpy(),
        result["Y01"].to_numpy(),
        result["Y11"].to_numpy(),
        sigma_margin_validation=sigma_margin_validation,
    )
    for name, values in views.items():
        result[name] = values
    result["sigma_margin_validation"] = float(sigma_margin_validation)
    result["scale_source_split"] = "validation"
    return result


def matched_profiles(rows: pd.DataFrame) -> pd.DataFrame:
    """Return one primary matched-context profile per discovery example."""
    required = {"base_sample_id", "pair_id", "context", *PROFILE_COMPONENTS}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError(f"profile rows missing columns: {missing}")
    matched = rows.loc[rows["context"].eq("matched")].copy()
    if matched["base_sample_id"].duplicated().any():
        raise ValueError("matched causal profile has duplicate sample IDs")
    if matched.empty:
        raise ValueError("matched causal profile is empty")
    return matched[["base_sample_id", "pair_id", *PROFILE_COMPONENTS]].sort_values(
        "base_sample_id"
    ).reset_index(drop=True)


def controlled_profiles(rows: pd.DataFrame) -> pd.DataFrame:
    """Secondary matched-minus-mean-random per-example profile."""
    matched = matched_profiles(rows).set_index("base_sample_id")
    random_rows = rows.loc[rows["context"].eq("random")]
    if random_rows.empty:
        raise ValueError("controlled causal profile requires random contexts")
    random_mean = random_rows.groupby("base_sample_id", sort=True)[
        ["A_z", "G_z"]
    ].mean()
    if set(matched.index) != set(random_mean.index):
        raise ValueError("matched/random profile sample IDs do not align")
    output = matched.copy()
    output[["A_z", "G_z"]] = output[["A_z", "G_z"]] - random_mean
    return output.reset_index()


def _correlation(first: np.ndarray, second: np.ndarray) -> tuple[float | None, str | None]:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return None, "degenerate variance"
    return float(np.corrcoef(first, second)[0, 1]), None


def causal_organization_distance(
    teacher: pd.DataFrame, student: pd.DataFrame
) -> dict[str, Any]:
    """Strictly align teacher/student rows and calculate primary COD diagnostics."""
    required = {"base_sample_id", "pair_id", *PROFILE_COMPONENTS}
    for name, frame in (("teacher", teacher), ("student", student)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} profile missing columns: {missing}")
        if frame["base_sample_id"].duplicated().any():
            raise ValueError(f"{name} profile has duplicate sample IDs")
    left = teacher.sort_values("base_sample_id").reset_index(drop=True)
    right = student.sort_values("base_sample_id").reset_index(drop=True)
    if left["base_sample_id"].tolist() != right["base_sample_id"].tolist():
        raise ValueError("teacher/student sample IDs do not align")
    if left["pair_id"].astype(str).tolist() != right["pair_id"].astype(str).tolist():
        raise ValueError("teacher/student pair IDs do not align")
    teacher_values = left[list(PROFILE_COMPONENTS)].to_numpy(np.float64)
    student_values = right[list(PROFILE_COMPONENTS)].to_numpy(np.float64)
    if not np.isfinite(teacher_values).all() or not np.isfinite(student_values).all():
        raise ValueError("teacher/student profiles contain nonfinite values")
    gap = student_values - teacher_values
    flat_teacher = teacher_values.reshape(-1)
    flat_student = student_values.reshape(-1)
    pearson, pearson_reason = _correlation(flat_teacher, flat_student)
    teacher_ranks = pd.Series(flat_teacher).rank(method="average").to_numpy()
    student_ranks = pd.Series(flat_student).rank(method="average").to_numpy()
    spearman, spearman_reason = _correlation(teacher_ranks, student_ranks)
    output: dict[str, Any] = {
        "n": len(left),
        "COD": float(np.mean(np.linalg.norm(gap, axis=1))),
        "pearson_profile_correlation": pearson,
        "pearson_na_reason": pearson_reason,
        "spearman_profile_correlation": spearman,
        "spearman_na_reason": spearman_reason,
    }
    for index, component in enumerate(PROFILE_COMPONENTS):
        short = component[0]
        output[f"mean_abs_{component}_gap"] = float(np.mean(np.abs(gap[:, index])))
        output[f"mean_signed_{component}_gap"] = float(np.mean(gap[:, index]))
        output[f"{short}_sign_agreement"] = float(
            np.mean(np.signbit(teacher_values[:, index]) == np.signbit(student_values[:, index]))
        )
    return output


def select_b_matched_checkpoint(
    checkpoint_rows: pd.DataFrame, teacher_validation_b: float
) -> dict[str, Any]:
    """Select only from validation B, breaking exact ties by earliest checkpoint."""
    required = {"step", "validation_B", "selection_split"}
    missing = sorted(required - set(checkpoint_rows.columns))
    if missing:
        raise ValueError(f"checkpoint rows missing columns: {missing}")
    if not checkpoint_rows["selection_split"].eq("validation").all():
        raise ValueError("B-matched checkpoint selection must use validation rows only")
    candidates = checkpoint_rows[["step", "validation_B"]].copy()
    if candidates["step"].duplicated().any() or candidates.empty:
        raise ValueError("checkpoint selection requires unique nonempty steps")
    candidates["absolute_B_gap"] = np.abs(
        candidates["validation_B"].astype(float) - float(teacher_validation_b)
    )
    minimum = float(candidates["absolute_B_gap"].min())
    tied = candidates.loc[
        np.isclose(candidates["absolute_B_gap"], minimum, rtol=0.0, atol=1e-12)
    ]
    selected = tied.sort_values("step", kind="mergesort").iloc[0]
    return {
        "selected_step": int(selected["step"]),
        "validation_B": float(selected["validation_B"]),
        "teacher_validation_B": float(teacher_validation_b),
        "absolute_B_gap": float(selected["absolute_B_gap"]),
        "selection_split": "validation",
        "tie_break": "earliest_checkpoint",
    }


def linear_cka(first: Any, second: Any) -> float | None:
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0] or x.shape[0] < 2:
        raise ValueError("linear CKA requires aligned 2D matrices with at least two rows")
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    numerator = float(np.linalg.norm(x.T @ y, ord="fro") ** 2)
    denominator = float(
        np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    )
    if denominator <= 1e-24:
        return None
    return numerator / denominator


def representation_similarity(
    student: Any,
    teacher: Any,
    projected_student: Any,
    *,
    cka_teacher: Any | None = None,
) -> dict[str, float | None]:
    source = np.asarray(student, dtype=np.float64)
    target = np.asarray(teacher, dtype=np.float64)
    projected = np.asarray(projected_student, dtype=np.float64)
    if source.ndim != 2 or target.ndim != 2 or projected.shape != target.shape:
        raise ValueError("representation diagnostics require aligned matrices")
    dot = np.sum(projected * target, axis=1)
    norms = np.linalg.norm(projected, axis=1) * np.linalg.norm(target, axis=1)
    cosine = np.divide(dot, norms, out=np.full_like(dot, np.nan), where=norms > 1e-12)
    return {
        "linear_CKA": linear_cka(
            source,
            target if cka_teacher is None else np.asarray(cka_teacher, dtype=np.float64),
        ),
        "mean_cosine_after_projector": (
            float(np.nanmean(cosine)) if np.isfinite(cosine).any() else None
        ),
        "projected_hidden_MSE": float(np.mean((projected - target) ** 2)),
    }


def immutable_run_identity(payload: dict[str, Any]) -> str:
    """Canonical digest used to reject cross-seed or cross-regime resume."""
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
