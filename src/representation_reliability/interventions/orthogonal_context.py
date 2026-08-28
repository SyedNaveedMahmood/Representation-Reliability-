"""Pure intervention math for fixed-setpoint orthogonal context."""

from __future__ import annotations

import numpy as np

from .setpoint import source_free_setpoint_delta
from .truth_coordinate import normalized_direction


def orthogonal_component(
    source_hidden: np.ndarray,
    base_hidden: np.ndarray,
    unit_truth_direction: np.ndarray,
) -> np.ndarray:
    """Project a source-minus-base displacement orthogonal to truth."""
    source = np.asarray(source_hidden, dtype=np.float64).reshape(-1)
    base = np.asarray(base_hidden, dtype=np.float64).reshape(-1)
    u = normalized_direction(unit_truth_direction)
    if source.shape != base.shape or source.shape != u.shape:
        raise ValueError("source/base/direction shape mismatch")
    if not np.isfinite(source).all() or not np.isfinite(base).all():
        raise ValueError("source and base activations must be finite")
    displacement = source - base
    context = displacement - float(np.dot(u, displacement)) * u
    # Remove the final floating-point residue explicitly.
    context = context - float(np.dot(u, context)) * u
    return context


def resolve_context_reference_norm(
    matched_orthogonal_norm: float,
    validation_fallback_norm: float,
    *,
    epsilon: float,
) -> tuple[float, str, bool]:
    """Resolve the predeclared per-example norm and its audited provenance."""
    matched = float(matched_orthogonal_norm)
    fallback = float(validation_fallback_norm)
    eps = float(epsilon)
    if eps <= 0.0 or fallback < eps or not np.isfinite(fallback):
        raise ValueError("fallback norm and epsilon must be positive and finite")
    if np.isfinite(matched) and matched >= eps:
        return matched, "matched_twin", False
    return fallback, "validation_median_nondegenerate_matched", True


def standardize_orthogonal_context(
    raw_context: np.ndarray,
    unit_truth_direction: np.ndarray,
    reference_norm: float,
    *,
    epsilon: float,
    fallback_direction: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Orthogonalize and per-example norm-match one context vector.

    A deterministic orthogonal fallback direction may be supplied for a
    degenerate raw vector. This prevents silent example deletion while making
    the exceptional substitution explicit in row-level evidence.
    """
    raw = np.asarray(raw_context, dtype=np.float64).reshape(-1)
    u = normalized_direction(unit_truth_direction)
    if raw.shape != u.shape:
        raise ValueError("context/direction shape mismatch")
    ref = float(reference_norm)
    eps = float(epsilon)
    if ref < eps or eps <= 0.0 or not np.isfinite(ref):
        raise ValueError("reference norm and epsilon must be positive and finite")
    projected = raw - float(np.dot(raw, u)) * u
    raw_norm = float(np.linalg.norm(projected))
    vector_fallback = raw_norm < eps
    if vector_fallback:
        if fallback_direction is None:
            raise ValueError("degenerate context requires a fallback direction")
        projected = np.asarray(fallback_direction, dtype=np.float64).reshape(-1)
        if projected.shape != u.shape:
            raise ValueError("fallback direction shape mismatch")
        projected = projected - float(np.dot(projected, u)) * u
        raw_norm = float(np.linalg.norm(projected))
        if raw_norm < eps:
            raise ValueError("fallback direction is degenerate")
    applied = ref * projected / raw_norm
    applied = applied - float(np.dot(applied, u)) * u
    applied *= ref / float(np.linalg.norm(applied))
    applied_norm = float(np.linalg.norm(applied))
    return applied, {
        "context_raw_norm": float(np.linalg.norm(raw)),
        "context_projected_raw_norm": raw_norm,
        "context_applied_norm": applied_norm,
        "context_dot_truth_direction": float(np.dot(applied, u)),
        "context_norm_relative_error": abs(applied_norm - ref) / ref,
        "context_vector_fallback_used": bool(vector_fallback),
    }


def fixed_setpoint_context_edit(
    base_hidden: np.ndarray,
    unit_truth_direction: np.ndarray,
    q_target: float,
    standardized_context: np.ndarray,
    context_strength: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return semantic, scaled-context, and total edit components."""
    base = np.asarray(base_hidden, dtype=np.float64).reshape(-1)
    u = normalized_direction(unit_truth_direction)
    context = np.asarray(standardized_context, dtype=np.float64).reshape(-1)
    if base.shape != u.shape or context.shape != u.shape:
        raise ValueError("base/context/direction shape mismatch")
    if abs(float(np.dot(context, u))) > 1e-10:
        raise ValueError("context is not orthogonal to truth direction")
    strength = float(context_strength)
    if not np.isfinite(strength):
        raise ValueError("context strength must be finite")
    semantic = source_free_setpoint_delta(base, u, q_target)
    scaled_context = strength * context
    total = semantic + scaled_context
    return semantic, scaled_context, total
