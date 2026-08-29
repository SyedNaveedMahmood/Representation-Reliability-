"""Locked primary inference for the E13 diagnostic confirmation.

Two frozen test forms are implemented:

* Stage A -- behavioral non-inferiority of a distilled student against the
  teacher, ``Delta_B = B_student - B_teacher > -delta_B``.  ``B`` is an AUROC over
  the whole confirmation set, so it is not a per-example quantity; the sampling
  unit is the counterfactual pair and the statistic is recomputed inside each
  bootstrap draw.
* Stage B -- component mismatch against a symmetric smallest-effect-size-of-
  interest region, ``H0: |Delta X_z| <= delta_C`` versus ``H1: |Delta X_z| >
  delta_C``.  These *are* per-example quantities, so the pair-cluster bootstrap
  resamples pair means directly.

Both stages share one hierarchical resampling scheme: a draw resamples pairs
once and every seed is re-evaluated on that same resampled pair set, so seeds
stay coupled through the examples they have in common.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

COMPONENTS = ("Q", "A", "G")


def holm_adjust_components(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjustment over exactly the Q/A/G component family."""
    if set(p_values) != set(COMPONENTS):
        raise ValueError("Holm family must contain exactly Q, A and G")
    ordered = sorted(((str(k), float(v)) for k, v in p_values.items()), key=lambda x: x[1])
    if any(not 0.0 <= value <= 1.0 for _key, value in ordered):
        raise ValueError("p-values must be finite values in [0, 1]")
    adjusted: dict[str, float] = {}
    running = 0.0
    size = len(ordered)
    for rank, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (size - rank) * value))
        adjusted[key] = running
    return adjusted


def _auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based AUROC that tolerates ties and degenerate label vectors."""
    positive = labels == 1
    n_pos = int(positive.sum())
    n_neg = int(labels.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _pair_index(pair_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Map rows to a dense pair index and return (row_pair_index, unique_pairs)."""
    series = pd.Series(list(map(str, pair_ids)))
    codes, uniques = pd.factorize(series, sort=True)
    return np.asarray(codes, dtype=np.int64), np.asarray(uniques)


def _pair_row_blocks(row_pair: np.ndarray, n_pairs: int) -> list[np.ndarray]:
    """Precompute, once, the row indices belonging to each pair."""
    order = np.argsort(row_pair, kind="stable")
    sorted_pairs = row_pair[order]
    starts = np.searchsorted(sorted_pairs, np.arange(n_pairs), side="left")
    ends = np.searchsorted(sorted_pairs, np.arange(n_pairs), side="right")
    blocks = [order[starts[index] : ends[index]] for index in range(n_pairs)]
    if any(block.size == 0 for block in blocks):
        raise RuntimeError("a confirmation pair contributed no rows")
    return blocks


def _draw_row_selection(
    blocks: list[np.ndarray], n_pairs: int, rng: np.random.Generator
) -> np.ndarray:
    """Resample pairs with replacement and expand to the rows they contain."""
    picked = rng.integers(0, n_pairs, size=n_pairs)
    return np.concatenate([blocks[index] for index in picked])


def behavior_noninferiority(
    per_seed: Mapping[int, pd.DataFrame],
    *,
    teacher: pd.DataFrame,
    delta_b: float,
    n_draws: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Pair-cluster bootstrap of ``B_student - B_teacher`` with a seed aggregate.

    Every frame must carry ``base_sample_id``, ``pair_id``, ``gold_label`` and
    ``clean_margin`` for exactly the same confirmation rows in the same order.
    """
    if not per_seed:
        raise ValueError("behavioral non-inferiority requires at least one seed")
    reference = teacher.sort_values("base_sample_id").reset_index(drop=True)
    row_pair, uniques = _pair_index(reference["pair_id"])
    n_pairs = int(uniques.size)
    labels = reference["gold_label"].to_numpy(dtype=int)
    teacher_scores = reference["clean_margin"].to_numpy(dtype=np.float64)

    student_scores: dict[int, np.ndarray] = {}
    for training_seed, frame in sorted(per_seed.items()):
        aligned = frame.sort_values("base_sample_id").reset_index(drop=True)
        if aligned["base_sample_id"].tolist() != reference["base_sample_id"].tolist():
            raise RuntimeError("confirmation rows are not aligned across models")
        if aligned["pair_id"].astype(str).tolist() != reference["pair_id"].astype(str).tolist():
            raise RuntimeError("confirmation pair identities are not aligned across models")
        if aligned["gold_label"].to_numpy(dtype=int).tolist() != labels.tolist():
            raise RuntimeError("confirmation labels are not aligned across models")
        student_scores[int(training_seed)] = aligned["clean_margin"].to_numpy(dtype=np.float64)

    teacher_b = _auroc(labels, teacher_scores)
    point = {
        training_seed: _auroc(labels, values) - teacher_b
        for training_seed, values in student_scores.items()
    }
    aggregate_point = float(np.mean(list(point.values())))

    rng = np.random.default_rng(int(seed))
    blocks = _pair_row_blocks(row_pair, n_pairs)
    draws = np.empty(int(n_draws), dtype=np.float64)
    per_seed_draws = {key: np.empty(int(n_draws), dtype=np.float64) for key in student_scores}
    for index in range(int(n_draws)):
        rows = _draw_row_selection(blocks, n_pairs, rng)
        drawn_labels = labels[rows]
        base = _auroc(drawn_labels, teacher_scores[rows])
        values = []
        for key, scores in student_scores.items():
            gap = _auroc(drawn_labels, scores[rows]) - base
            per_seed_draws[key][index] = gap
            values.append(gap)
        draws[index] = float(np.mean(values))

    tail = (1.0 - float(confidence_level)) / 2.0

    def interval(sample: np.ndarray) -> tuple[float, float]:
        finite = sample[np.isfinite(sample)]
        if finite.size == 0:
            return float("nan"), float("nan")
        low, high = np.quantile(finite, [tail, 1.0 - tail])
        return float(low), float(high)

    aggregate_low, aggregate_high = interval(draws)
    seeds_above = sum(1 for value in point.values() if value > -float(delta_b))
    return {
        "teacher_B": float(teacher_b),
        "per_seed": {
            int(key): {
                "student_B": float(_auroc(labels, student_scores[key])),
                "delta_B": float(point[key]),
                "ci_low": interval(per_seed_draws[key])[0],
                "ci_high": interval(per_seed_draws[key])[1],
                "noninferior": bool(point[key] > -float(delta_b)),
            }
            for key in sorted(point)
        },
        "aggregate_delta_B": aggregate_point,
        "aggregate_ci_low": aggregate_low,
        "aggregate_ci_high": aggregate_high,
        "delta_b": float(delta_b),
        "seeds_noninferior": int(seeds_above),
        "n_seeds": len(point),
        "n_pairs": n_pairs,
        "verdict": "PASS"
        if (aggregate_low > -float(delta_b) and seeds_above >= 2)
        else "FAIL",
    }


def _pair_means(values: np.ndarray, row_pair: np.ndarray, n_pairs: int) -> np.ndarray:
    sums = np.bincount(row_pair, weights=values, minlength=n_pairs)
    counts = np.bincount(row_pair, minlength=n_pairs)
    if np.any(counts == 0):
        raise RuntimeError("a confirmation pair contributed no rows")
    return sums / counts


def component_mismatch(
    per_seed: Mapping[int, pd.DataFrame],
    *,
    component: str,
    delta_c: float,
    n_draws: int,
    seed: int,
    confidence_level: float = 0.95,
) -> dict[str, object]:
    """Test ``H0: |mean gap| <= delta_c`` for one causal-organization component.

    Each frame must carry ``base_sample_id``, ``pair_id`` and a ``gap`` column
    holding the per-example student-minus-teacher standardized component gap.
    """
    if not per_seed:
        raise ValueError("component mismatch requires at least one seed")
    frames = {
        int(key): value.sort_values("base_sample_id").reset_index(drop=True)
        for key, value in per_seed.items()
    }
    reference = frames[min(frames)]
    row_pair, uniques = _pair_index(reference["pair_id"])
    n_pairs = int(uniques.size)
    gaps: dict[int, np.ndarray] = {}
    for key, frame in frames.items():
        if frame["base_sample_id"].tolist() != reference["base_sample_id"].tolist():
            raise RuntimeError("component gaps are not aligned across seeds")
        values = frame["gap"].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite {component} gap on confirmation rows")
        gaps[key] = values

    pair_means = {key: _pair_means(values, row_pair, n_pairs) for key, values in gaps.items()}
    point = {key: float(np.mean(values)) for key, values in pair_means.items()}
    aggregate_point = float(np.mean(list(point.values())))

    rng = np.random.default_rng(int(seed))
    stacked = np.stack([pair_means[key] for key in sorted(pair_means)], axis=0)
    draws = np.empty(int(n_draws), dtype=np.float64)
    for index in range(int(n_draws)):
        picked = rng.integers(0, n_pairs, size=n_pairs)
        draws[index] = float(np.mean(stacked[:, picked]))

    tail = (1.0 - float(confidence_level)) / 2.0
    low, high = (float(v) for v in np.quantile(draws, [tail, 1.0 - tail]))

    # One-sided bootstrap p-value against the nearer SESOI boundary. When the
    # point estimate is inside the null region the test cannot reject, so p = 1.
    bound = float(delta_c)
    if aggregate_point > bound:
        exceed = int(np.sum(draws <= bound))
    elif aggregate_point < -bound:
        exceed = int(np.sum(draws >= -bound))
    else:
        exceed = int(n_draws)
    raw_p = float((exceed + 1) / (int(n_draws) + 1))

    direction = int(np.sign(aggregate_point))
    seeds_same_direction = sum(
        1
        for value in point.values()
        if abs(value) >= bound and int(np.sign(value)) == direction and direction != 0
    )
    return {
        "component": str(component),
        "aggregate_gap": aggregate_point,
        "ci_low": low,
        "ci_high": high,
        "raw_p": raw_p,
        "delta_c": bound,
        "ci_outside_sesoi": bool(low > bound or high < -bound),
        "direction": direction,
        "seeds_beyond_sesoi_same_direction": int(seeds_same_direction),
        "n_seeds": len(point),
        "n_pairs": n_pairs,
        "per_seed": {int(key): float(value) for key, value in sorted(point.items())},
    }


def evaluate_regime_components(
    gaps_by_component: Mapping[str, Mapping[int, pd.DataFrame]],
    *,
    delta_c: float,
    n_draws: int,
    seed: int,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Run all three component tests for one regime and Holm-adjust within it."""
    if set(gaps_by_component) != set(COMPONENTS):
        raise ValueError("component family must be exactly Q, A and G")
    results = {}
    for offset, component in enumerate(COMPONENTS):
        results[component] = component_mismatch(
            gaps_by_component[component],
            component=component,
            delta_c=delta_c,
            n_draws=n_draws,
            seed=int(seed) + 97 * offset,
        )
    adjusted = holm_adjust_components(
        {key: float(value["raw_p"]) for key, value in results.items()}
    )
    records = []
    for component in COMPONENTS:
        item = dict(results[component])
        item["holm_p"] = float(adjusted[component])
        item["mismatch"] = bool(
            item["ci_outside_sesoi"]
            and item["holm_p"] < float(alpha)
            and int(item["seeds_beyond_sesoi_same_direction"]) >= 2
        )
        records.append(item)
    return pd.DataFrame(records)


def classify_confirmation(
    behavior: Mapping[str, Mapping[str, object]],
    components: Mapping[str, pd.DataFrame],
) -> dict[str, object]:
    """Apply the frozen hierarchical gatekeeping classification."""
    verdicts: dict[str, dict[str, object]] = {}
    for regime in sorted(behavior):
        stage_a = str(behavior[regime]["verdict"]) == "PASS"
        frame = components[regime]
        stage_b = bool(stage_a and frame["mismatch"].any())
        verdicts[regime] = {
            "behavior_noninferiority": "PASS" if stage_a else "FAIL",
            "causal_mismatch": "PASS" if stage_b else "FAIL",
            "mismatched_components": (
                frame.loc[frame["mismatch"], "component"].astype(str).tolist()
            ),
        }
    r2 = verdicts.get("R2", {})
    r3 = verdicts.get("R3", {})
    r2_full = r2.get("behavior_noninferiority") == "PASS" and r2.get("causal_mismatch") == "PASS"
    r3_full = r3.get("behavior_noninferiority") == "PASS" and r3.get("causal_mismatch") == "PASS"
    if r2_full and r3_full:
        classification = "strong"
    elif r2_full or r3_full:
        classification = "partial"
    else:
        classification = "failed"
    return {
        "regimes": verdicts,
        "classification": classification,
        "entry_gate_for_cross_family": bool(r2_full or r3_full),
    }
