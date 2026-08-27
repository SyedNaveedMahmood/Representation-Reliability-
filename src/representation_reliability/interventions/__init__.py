"""Local intervention primitives for representation-reliability experiments."""

from .truth_coordinate import (
    coordinate_value,
    coordinate_transfer_delta,
    full_residual_patch_delta,
    normalized_direction,
    random_unit_direction,
)

__all__ = [
    "coordinate_value",
    "coordinate_transfer_delta",
    "full_residual_patch_delta",
    "normalized_direction",
    "random_unit_direction",
]
