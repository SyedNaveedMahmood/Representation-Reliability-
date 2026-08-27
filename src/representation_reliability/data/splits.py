"""Deterministic group-based dataset splits with hard confirmation isolation.

Splits: train 60% | validation 15% | discovery_test 15% | confirmation 10%.

Splitting is performed over *groups* (``pair_id``), so matched
counterfactual twins always land in the same split.  The ``confirmation``
split is a protected holdout: accessing its rows requires explicitly
overriding the guard, and E00 discovery code never does.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pandas as pd

SPLIT_NAMES = ("train", "validation", "discovery_test", "confirmation")
DEFAULT_Split_FRACTIONS = {
    "train": 0.60,
    "validation": 0.15,
    "discovery_test": 0.15,
    "confirmation": 0.10,
}


class ConfirmationSplitAccessError(RuntimeError):
    """Raised when code touches the confirmation split without explicit consent."""


def assign_group_splits(
    groups: Sequence[str],
    fractions: dict[str, float] | None = None,
    seed: int = 0,
) -> dict[str, str]:
    """Deterministically assign whole groups to splits by target proportions."""
    fr = dict(fractions or DEFAULT_Split_FRACTIONS)
    total = sum(fr.values())
    if not abs(total - 1.0) < 1e-9:
        raise ValueError(f"split fractions must sum to 1.0 (got {total})")
    for name in SPLIT_NAMES:
        if name not in fr:
            raise ValueError(f"missing split fraction for {name!r}")
        if fr[name] < 0:
            raise ValueError(f"negative fraction for {name!r}")

    unique_groups = sorted(set(groups))
    if len(unique_groups) < len(SPLIT_NAMES):
        raise ValueError(
            f"need at least {len(SPLIT_NAMES)} groups to form all splits; got "
            f"{len(unique_groups)}"
        )
    rng = random.Random(seed)
    shuffled = list(unique_groups)
    rng.shuffle(shuffled)

    n = len(shuffled)
    counts = {name: int(round(fr[name] * n)) for name in SPLIT_NAMES}
    # Fix rounding drift so all groups are assigned exactly once.
    while sum(counts.values()) > n:
        counts["train"] -= 1
    while sum(counts.values()) < n:
        counts["train"] += 1

    assignment: dict[str, str] = {}
    cursor = 0
    for name in SPLIT_NAMES:
        take = counts[name]
        for g in shuffled[cursor : cursor + take]:
            assignment[g] = name
        cursor += take
    return assignment


def apply_splits(df: pd.DataFrame, assignment: dict[str, str],
                 group_col: str = "pair_id") -> pd.DataFrame:
    out = df.copy()
    out["split"] = out[group_col].map(assignment)
    unmapped = out[out["split"].isna()][group_col].unique().tolist()
    if unmapped:
        raise ValueError(f"groups missing from split assignment: {unmapped[:5]}")
    return out


def discovery_view(
    df: pd.DataFrame,
    split_col: str = "split",
    allow_confirmation: bool = False,
) -> pd.DataFrame:
    """Return rows usable for exploration.

    By default the confirmation split is *removed entirely* — not merely
    hidden — so that neither training nor evaluation can read it.
    """
    if allow_confirmation:
        return df.copy()
    return df[df[split_col] != "confirmation"].copy()


def require_confirmation_access(requested_by: str) -> None:
    """Explicit opt-in gate for confirmatory evaluation."""
    answer = requested_by.strip().lower()
    if answer != "confirmatory_evaluation":
        raise ConfirmationSplitAccessError(
            "the confirmation split is reserved for explicit confirmatory "
            "evaluation only; pass requested_by='confirmatory_evaluation' and "
            "record the access in the run manifest"
        )


def build_discovery_label_map(df: pd.DataFrame, split_col: str = "split") -> dict[str, int]:
    """Label map over NON-confirmation rows only.

    Requesting a confirmation-split sample id raises
    :class:`ConfirmationSplitAccessError` — during discovery, confirmation
    labels must never be constructed, even incidentally. Deterministic
    regeneration recovers them later for explicit confirmatory evaluation.
    """
    mapping: dict[str, int] = {}
    for sid, split, label in zip(
        df["sample_id"], df[split_col], df["target_label"].astype(int)
    ):
        if split == "confirmation":
            raise ConfirmationSplitAccessError(
                f"attempted to read a confirmation label for {sid!r} during "
                "discovery; confirmation labels stay unobservable until an "
                "explicit confirmatory evaluation"
            )
        mapping[str(sid)] = int(label)
    return mapping


def validate_splits(df: pd.DataFrame, split_col: str = "split") -> dict[str, dict]:
    """Sanity-check splits WITHOUT inspecting confirmation labels."""
    problems: list[str] = []
    summary: dict[str, dict] = {}
    for name in SPLIT_NAMES:
        sub = df[df[split_col] == name]
        if name == "confirmation":
            # Holdout: verify existence/size only — labels stay unobserved.
            summary[name] = {
                "n_rows": len(sub),
                "n_groups": int(sub["pair_id"].nunique()),
                "labels_present": None,
                "labels_observed": False,
            }
            continue
        labels = sorted(int(x) for x in sub["target_label"].unique().tolist())
        if len(sub) == 0:
            problems.append(f"split '{name}' is empty")
        elif len(labels) < 2:
            problems.append(f"split '{name}' has degenerate classes: {labels}")
        summary[name] = {
            "n_rows": len(sub),
            "n_groups": int(sub["pair_id"].nunique()),
            "labels_present": labels,
        }
    cross = df.groupby("pair_id")[split_col].nunique()
    n_crossed = int((cross > 1).sum())
    if n_crossed:
        problems.append(f"{n_crossed} pair group(s) straddle multiple splits")
    if problems:
        raise ValueError("split validation failed: " + "; ".join(problems))
    return summary
