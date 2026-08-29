"""Contract tests for the frozen E15 stateful console environment."""

from __future__ import annotations

import pytest

from representation_reliability.data.stateful_console import (
    DENIED,
    GRANTED,
    LONG_DISTRACTORS,
    SHORT_DISTRACTORS,
    decision_label,
    generate_console_episodes,
    generate_console_samples,
    render_episode,
    validate_console_samples,
)
from representation_reliability.runners.e15_support import build_e15_corpus

HORIZONS = (1, 2, 4, 8)


def _samples(n_pairs: int = 8, seed: int = 20261501):
    return generate_console_samples(
        n_pairs, seed, HORIZONS, namespace="e15-test-v1"
    )


def test_label_oracle_is_deterministic_and_exact():
    assert decision_label(GRANTED) == 1
    assert decision_label(DENIED) == 0
    with pytest.raises(ValueError):
        decision_label("MAYBE")
    for sample in _samples():
        assert decision_label(sample.metadata["target_state"]) == sample.target_label


def test_twins_differ_in_exactly_one_clearance_word():
    samples = _samples()
    report = validate_console_samples(samples)
    assert report["n_pairs"] == len(samples) // 2
    pairs: dict[str, list] = {}
    for sample in samples:
        pairs.setdefault(str(sample.pair_id), []).append(sample)
    for rows in pairs.values():
        first, second = sorted(rows, key=lambda s: str(s.sample_id))
        differing = [
            (a, b)
            for a, b in zip(first.prompt.split("\n"), second.prompt.split("\n"))
            if a != b
        ]
        assert len(differing) == 1
        assert GRANTED in differing[0][0] or DENIED in differing[0][0]
        assert first.target_label != second.target_label


def test_horizon_rendering_is_a_pure_prefix_extension():
    episodes = generate_console_episodes(4, 20261501)
    for episode in episodes:
        short = render_episode(episode, 1)
        for horizon in (2, 4, 8, 16, 32):
            longer = render_episode(episode, horizon)
            prefix = "\n".join(short["prompt"].split("\n")[:5])
            assert longer["prompt"].startswith(prefix)
            assert longer["target_clearance_line"] == short["target_clearance_line"]
            assert longer["irrelevant_clearance_line"] == short["irrelevant_clearance_line"]
            # the late carrier tracks the horizon, the target carrier does not
            assert longer["late_gap_line"] != short["late_gap_line"] or horizon == 1


def test_carrier_spans_are_unique_and_state_free_where_required():
    for sample in _samples():
        for key in (
            "target_clearance_line",
            "irrelevant_clearance_line",
            "late_gap_line",
        ):
            assert sample.prompt.count(sample.metadata[key]) == 1
        late = sample.metadata["late_gap_line"]
        assert GRANTED not in late and DENIED not in late


def test_irrelevant_state_is_balanced_and_uncorrelated_with_the_label():
    samples = generate_console_samples(
        400, 20261501, (1,), namespace="e15-balance-v1"
    )
    positives = [s for s in samples if s.target_label == 1]
    negatives = [s for s in samples if s.target_label == 0]
    assert len(positives) == len(negatives) == 400

    def granted_fraction(rows):
        return sum(r.metadata["other_state"] == GRANTED for r in rows) / len(rows)

    assert granted_fraction(positives) == pytest.approx(0.5, abs=0.02)
    assert granted_fraction(negatives) == pytest.approx(0.5, abs=0.02)


def test_horizon_beyond_the_generated_gap_is_rejected():
    episode = generate_console_episodes(1, 1, max_gap_steps=4)[0]
    render_episode(episode, 4)
    with pytest.raises(ValueError):
        render_episode(episode, 5)
    with pytest.raises(ValueError):
        render_episode(episode, 0)


def test_long_distractor_pool_lengthens_steps_without_changing_the_state():
    short = generate_console_samples(4, 7, (4,), namespace="e15-short-v1",
                                     distractor_pool=SHORT_DISTRACTORS)
    long = generate_console_samples(4, 7, (4,), namespace="e15-long-v1",
                                    distractor_pool=LONG_DISTRACTORS)
    assert [s.target_label for s in short] == [s.target_label for s in long]
    assert sum(len(s.prompt) for s in long) > sum(len(s.prompt) for s in short)


def test_full_corpus_is_pair_complete_deduplicated_and_split_clean():
    samples, frame, stats = build_e15_corpus(
        specs=(("train", 20, 20261501), ("validation", 10, 20261502),
               ("discovery_test", 10, 20261503)),
        horizons=(1, 2),
    )
    assert not frame["sample_id"].duplicated().any()
    assert not frame["prompt"].duplicated().any()
    assert frame.groupby("pair_id")["split"].nunique().max() == 1
    assert frame.groupby("pair_id")["sample_id"].size().eq(2).all()
    assert stats["confirmation_split_exists"] is False
    assert set(frame["split"]) == {"train", "validation", "discovery_test"}
    assert len(samples) == (20 + 10 + 10) * 2 * 2
