"""Pure intervention math for one-dimensional decoded representation coordinates."""

from __future__ import annotations

import numpy as np


def normalized_direction(direction: np.ndarray) -> np.ndarray:
    """Return a float64 unit vector; reject degenerate probe directions."""
    u = np.asarray(direction, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(u))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("direction must have finite non-zero norm")
    return u / norm


def coordinate_value(hidden: np.ndarray, unit_direction: np.ndarray) -> float:
    """Scalar coordinate of one residual vector along a unit direction."""
    h = np.asarray(hidden, dtype=np.float64).reshape(-1)
    u = normalized_direction(unit_direction)
    if h.shape != u.shape:
        raise ValueError(f"hidden/direction shape mismatch: {h.shape} vs {u.shape}")
    return float(np.dot(h, u))


def coordinate_transfer_delta(
    base_hidden: np.ndarray,
    source_hidden: np.ndarray,
    unit_direction: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Change only the decoded coordinate toward a source coordinate.

    For unit ``u`` this returns

        alpha * ((u @ source) - (u @ base)) * u

    so alpha=1 copies only the source coordinate while preserving every
    component orthogonal to ``u``.
    """
    base = np.asarray(base_hidden, dtype=np.float64).reshape(-1)
    source = np.asarray(source_hidden, dtype=np.float64).reshape(-1)
    u = normalized_direction(unit_direction)
    if base.shape != source.shape or base.shape != u.shape:
        raise ValueError("base/source/direction shape mismatch")
    scalar = float(alpha) * float(np.dot(source - base, u))
    return scalar * u


def full_residual_patch_delta(
    base_hidden: np.ndarray,
    source_hidden: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Whole-residual interpolation/extrapolation upper-bound control."""
    base = np.asarray(base_hidden, dtype=np.float64).reshape(-1)
    source = np.asarray(source_hidden, dtype=np.float64).reshape(-1)
    if base.shape != source.shape:
        raise ValueError("base/source shape mismatch")
    return float(alpha) * (source - base)


def random_unit_direction(
    dim: int,
    seed: int,
    *,
    orthogonal_to: np.ndarray | None = None,
) -> np.ndarray:
    """Deterministic Gaussian unit direction, optionally orthogonal to a vector."""
    if int(dim) <= 0:
        raise ValueError("dim must be positive")
    rng = np.random.default_rng(int(seed))
    v = rng.standard_normal(int(dim)).astype(np.float64)
    if orthogonal_to is not None:
        u = normalized_direction(orthogonal_to)
        if len(u) != int(dim):
            raise ValueError("orthogonal reference dimensionality mismatch")
        v = v - np.dot(v, u) * u
    norm = float(np.linalg.norm(v))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise RuntimeError("random direction became degenerate")
    return v / norm
