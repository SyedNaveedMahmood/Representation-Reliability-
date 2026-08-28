"""Pure source-free coordinate-setpoint intervention math."""

from __future__ import annotations

import numpy as np

from .truth_coordinate import coordinate_value, normalized_direction


def source_free_setpoint_delta(
    base_hidden: np.ndarray,
    unit_direction: np.ndarray,
    q_target: float,
) -> np.ndarray:
    """Return the rank-one edit that sets ``u @ h`` to ``q_target``."""
    base = np.asarray(base_hidden, dtype=np.float64).reshape(-1)
    u = normalized_direction(unit_direction)
    if base.shape != u.shape:
        raise ValueError("base/direction shape mismatch")
    target = float(q_target)
    if not np.isfinite(target) or not np.isfinite(base).all():
        raise ValueError("base and q_target must be finite")
    return (target - coordinate_value(base, u)) * u


def norm_matched_direction_delta(
    reference_delta: np.ndarray,
    unit_control_direction: np.ndarray,
) -> np.ndarray:
    """Place the reference edit norm on a supplied unit control direction."""
    reference = np.asarray(reference_delta, dtype=np.float64).reshape(-1)
    direction = normalized_direction(unit_control_direction)
    if reference.shape != direction.shape:
        raise ValueError("reference/control direction shape mismatch")
    norm = float(np.linalg.norm(reference))
    if not np.isfinite(norm):
        raise ValueError("reference delta must be finite")
    return norm * direction


def setpoint_identity_diagnostics(
    base_hidden: np.ndarray,
    intervened_hidden: np.ndarray,
    unit_direction: np.ndarray,
    q_target: float,
) -> dict[str, float]:
    """Measure target projection and orthogonal-subspace preservation."""
    base = np.asarray(base_hidden, dtype=np.float64).reshape(-1)
    after = np.asarray(intervened_hidden, dtype=np.float64).reshape(-1)
    u = normalized_direction(unit_direction)
    if base.shape != after.shape or base.shape != u.shape:
        raise ValueError("base/intervened/direction shape mismatch")
    q_base = coordinate_value(base, u)
    q_after = coordinate_value(after, u)
    target = float(q_target)
    projection_abs = abs(q_after - target)
    projection_relative = projection_abs / max(abs(q_base), abs(target), 1.0)
    base_orthogonal = base - q_base * u
    after_orthogonal = after - q_after * u
    orthogonal_abs = float(np.linalg.norm(after_orthogonal - base_orthogonal))
    orthogonal_relative = orthogonal_abs / max(float(np.linalg.norm(base_orthogonal)), 1.0)
    return {
        "q_base": q_base,
        "q_target": target,
        "q_after": q_after,
        "projection_abs_deviation": float(projection_abs),
        "projection_relative_deviation": float(projection_relative),
        "orthogonal_abs_deviation": orthogonal_abs,
        "orthogonal_relative_deviation": float(orthogonal_relative),
    }
