"""Deterministic stateful console-log episodes for E15 (temporal causal half-life).

The environment is frozen by ``docs/E15_TEMPORAL_CAUSAL_HALF_LIFE_PROTOCOL.md``
section 1 and is deliberately narrow: one explicit binary latent state, one
deterministic write per terminal, one delayed decision that depends on it.

Episode skeleton::

    header
    1 prefix distractor step
    2 clearance-setting steps      (order varies per episode, balanced)
    k gap distractor steps         (k = the horizon)
    1 request step
    question line
    "Answer:"

Two properties make the horizon the only thing that varies:

* **prefix extension** — the horizon-``k`` prompt inserts the first ``k`` of an
  episode's deterministic gap distractors. Every byte up to and including the
  clearance block is therefore identical across horizons, so in a causal LM the
  carrier activation is horizon-invariant by construction;
* **matched counterfactual twins** — a twin flips *only* the queried terminal's
  clearance word, so the decision label flips while operators, terminals,
  distractor text, step numbering, clearance order, horizon and the irrelevant
  terminal's flag are all identical.

The non-queried terminal carries an independent flag; it is the frozen
irrelevant-state control variable and is uncorrelated with the label.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..contracts import Sample

TASK_NAME = "stateful_console"

GRANTED = "GRANTED"
DENIED = "DENIED"
STATE_WORDS = (GRANTED, DENIED)

HEADER = (
    "Console log. A terminal may run a transfer only if its clearance is GRANTED."
)
QUESTION_LINE = "Question: Should the request be approved?"
ANSWER_CUE = "Answer:"

# Invented proper nouns, disjoint roles so an operator is never a terminal.
OPERATORS = [
    "Karsi", "Sidra", "Ovell", "Vanta", "Dennit", "Bravu", "Torren", "Halvit",
    "Quoran", "Esmik", "Yalta", "Gomar", "Neshi", "Aldive", "Wexley", "Farro",
    "Thale", "Rhevik", "Odari", "Sammet", "Pallor", "Marnet", "Jesco", "Uliah",
]
TERMINALS = [
    "Luma", "Reko", "Milqueth", "Prue", "Zubon", "Iona", "Bretta", "Onnavi",
    "Clavian", "Druse", "Vellum", "Gritta", "Hobban", "Junis", "Yune", "Ashen",
]

# Distractor clauses. They never mention clearance, terminals, transfers or the
# state words, so no distractor can leak or restate the latent variable.
SHORT_DISTRACTORS = [
    "logs the intake pressure",
    "checks the coolant level",
    "records the ambient humidity",
    "reads the spool counter",
    "notes the cabinet temperature",
    "files the shift report",
    "inspects the filter housing",
    "resets the panel timer",
    "measures the line voltage",
    "labels the sample tray",
    "updates the maintenance sheet",
    "clears the console buffer",
    "verifies the lamp indicator",
    "stamps the delivery docket",
    "counts the spare fuses",
    "wipes the sensor window",
]
LONG_DISTRACTORS = [
    "logs the intake pressure and files the reading with the shift supervisor",
    "checks the coolant level and copies the value onto the maintenance sheet",
    "records the ambient humidity and compares it against the morning baseline",
    "reads the spool counter and annotates the discrepancy in the day book",
    "notes the cabinet temperature and schedules a follow-up inspection window",
    "files the shift report and forwards a duplicate to the records office",
    "inspects the filter housing and marks the replacement date on the tag",
    "resets the panel timer and confirms the countdown against the wall clock",
]

MAX_GAP_STEPS = 32


@dataclass(frozen=True)
class ConsoleEpisode:
    """One base episode; horizons are renderings of the same episode."""

    episode_index: int
    operator_prefix: str
    operator_first: str
    operator_second: str
    terminal_target: str
    terminal_other: str
    target_state: str
    other_state: str
    target_slot: int                 # 1 or 2: which clearance step writes z
    prefix_distractor: str
    gap_distractors: tuple[str, ...]
    gap_operators: tuple[str, ...]

    def flipped(self) -> ConsoleEpisode:
        """The matched twin: only the queried terminal's clearance word flips."""
        return replace(
            self,
            target_state=DENIED if self.target_state == GRANTED else GRANTED,
        )


def decision_label(target_state: str) -> int:
    """Deterministic oracle: approve iff the queried terminal is GRANTED."""
    if target_state == GRANTED:
        return 1
    if target_state == DENIED:
        return 0
    raise ValueError(f"unknown clearance state {target_state!r}")


def _clearance_line(step_number: int, operator: str, terminal: str, state: str) -> str:
    return (
        f"Step {step_number:02d}: Operator {operator} sets the clearance for "
        f"terminal {terminal} to {state}."
    )


def _distractor_line(step_number: int, operator: str, clause: str) -> str:
    return f"Step {step_number:02d}: Operator {operator} {clause}."


def _request_line(step_number: int, terminal: str) -> str:
    return f"Step {step_number:02d}: Terminal {terminal} requests to run a transfer."


def render_episode(episode: ConsoleEpisode, horizon: int) -> dict[str, Any]:
    """Render one episode at one horizon and report every frozen span.

    Returns the prompt plus the exact line strings of the target clearance step,
    the irrelevant clearance step and the final gap distractor step. Those three
    lines are the frozen carrier spans of protocol sections 2 and 6.
    """
    k = int(horizon)
    if k < 1:
        raise ValueError("horizon must be at least one gap step")
    if k > len(episode.gap_distractors):
        raise ValueError(
            f"episode {episode.episode_index} carries "
            f"{len(episode.gap_distractors)} gap distractors, horizon {k} requested"
        )

    lines: list[str] = [HEADER]
    step = 1
    lines.append(_distractor_line(step, episode.operator_prefix, episode.prefix_distractor))
    step += 1

    target_line = _clearance_line(
        step if episode.target_slot == 1 else step + 1,
        episode.operator_first if episode.target_slot == 1 else episode.operator_second,
        episode.terminal_target,
        episode.target_state,
    )
    other_line = _clearance_line(
        step + 1 if episode.target_slot == 1 else step,
        episode.operator_second if episode.target_slot == 1 else episode.operator_first,
        episode.terminal_other,
        episode.other_state,
    )
    if episode.target_slot == 1:
        lines.extend([target_line, other_line])
    else:
        lines.extend([other_line, target_line])
    step += 2

    gap_lines: list[str] = []
    for offset in range(k):
        gap_lines.append(
            _distractor_line(
                step + offset,
                episode.gap_operators[offset],
                episode.gap_distractors[offset],
            )
        )
    lines.extend(gap_lines)
    step += k

    lines.append(_request_line(step, episode.terminal_target))
    lines.append(QUESTION_LINE)
    lines.append(ANSWER_CUE)
    prompt = "\n".join(lines)

    for line in (target_line, other_line, gap_lines[-1]):
        if prompt.count(line) != 1:
            raise RuntimeError(
                f"carrier span is not unique in episode {episode.episode_index}"
            )
    return {
        "prompt": prompt,
        "target_clearance_line": target_line,
        "irrelevant_clearance_line": other_line,
        "late_gap_line": gap_lines[-1],
        "n_steps": step,
        "horizon": k,
    }


def _make_sample(
    episode: ConsoleEpisode,
    horizon: int,
    *,
    namespace: str,
    pair_index: int,
    twin_index: int,
) -> Sample:
    rendered = render_episode(episode, horizon)
    label = decision_label(episode.target_state)
    pair_id = f"{namespace}-h{int(horizon):02d}-pair-{pair_index:06d}"
    sample_id = f"{namespace}-h{int(horizon):02d}-sample-{2 * pair_index + twin_index:06d}"
    counterfactual_id = (
        f"{namespace}-h{int(horizon):02d}-sample-"
        f"{2 * pair_index + (1 - twin_index):06d}"
    )
    metadata: dict[str, Any] = {
        "episode_index": int(episode.episode_index),
        "episode_id": f"{namespace}-episode-{episode.episode_index:06d}",
        "horizon": int(horizon),
        "pair_id": pair_id,
        "target_label": label,
        "target_state": episode.target_state,
        "other_state": episode.other_state,
        "terminal_target": episode.terminal_target,
        "terminal_other": episode.terminal_other,
        "target_slot": int(episode.target_slot),
        "irrelevant_label": decision_label(episode.other_state),
        "n_steps": int(rendered["n_steps"]),
        # Frozen carrier spans. `target_text` drives the `target_span_last`
        # selector, which resolves the primary carrier token.
        "target_text": rendered["target_clearance_line"],
        "target_clearance_line": rendered["target_clearance_line"],
        "irrelevant_clearance_line": rendered["irrelevant_clearance_line"],
        "late_gap_line": rendered["late_gap_line"],
        "chat_template_used": False,
    }
    return Sample(
        sample_id=sample_id,
        prompt=rendered["prompt"],
        target_label=label,
        task_name=TASK_NAME,
        pair_id=pair_id,
        counterfactual_id=counterfactual_id,
        expected_counterfactual_label=1 - label,
        metadata=metadata,
    )


def generate_console_episodes(
    n_episodes: int,
    seed: int,
    *,
    max_gap_steps: int = MAX_GAP_STEPS,
    distractor_pool: Sequence[str] | None = None,
) -> list[ConsoleEpisode]:
    """Deterministically build base episodes with balanced flag combinations.

    The four (target flag, irrelevant flag) combinations and the two clearance
    orders are cycled rather than sampled, so every split is exactly balanced.
    """
    if int(n_episodes) <= 0:
        raise ValueError("n_episodes must be positive")
    pool = list(distractor_pool if distractor_pool is not None else SHORT_DISTRACTORS)
    if len(pool) < 2:
        raise ValueError("distractor pool must contain at least two clauses")
    rng = random.Random(int(seed))
    combos = [
        (GRANTED, GRANTED),
        (GRANTED, DENIED),
        (DENIED, GRANTED),
        (DENIED, DENIED),
    ]
    episodes: list[ConsoleEpisode] = []
    for index in range(int(n_episodes)):
        target_state, other_state = combos[index % len(combos)]
        target_slot = 1 + ((index // len(combos)) % 2)
        terminal_target, terminal_other = rng.sample(TERMINALS, 2)
        ops = rng.sample(OPERATORS, 3)
        gap_clauses = tuple(rng.choice(pool) for _ in range(int(max_gap_steps)))
        gap_operators = tuple(rng.choice(OPERATORS) for _ in range(int(max_gap_steps)))
        episodes.append(
            ConsoleEpisode(
                episode_index=index,
                operator_prefix=ops[0],
                operator_first=ops[1],
                operator_second=ops[2],
                terminal_target=terminal_target,
                terminal_other=terminal_other,
                target_state=target_state,
                other_state=other_state,
                target_slot=target_slot,
                prefix_distractor=rng.choice(pool),
                gap_distractors=gap_clauses,
                gap_operators=gap_operators,
            )
        )
    return episodes


def generate_console_samples(
    n_pairs: int,
    seed: int,
    horizons: Sequence[int],
    *,
    namespace: str,
    max_gap_steps: int = MAX_GAP_STEPS,
    distractor_pool: Sequence[str] | None = None,
) -> list[Sample]:
    """Render ``n_pairs`` twin pairs at every requested horizon.

    Every horizon reuses the *same* base episodes, so an episode's identity,
    carrier position and nuisance content are constant across the horizon grid.
    """
    grid = [int(k) for k in horizons]
    if not grid or len(set(grid)) != len(grid) or any(k < 1 for k in grid):
        raise ValueError("horizons must be a non-empty set of positive integers")
    episodes = generate_console_episodes(
        int(n_pairs),
        int(seed),
        max_gap_steps=max(max_gap_steps, max(grid)),
        distractor_pool=distractor_pool,
    )
    samples: list[Sample] = []
    for horizon in grid:
        for pair_index, episode in enumerate(episodes):
            twin = episode.flipped()
            positive, negative = (
                (episode, twin)
                if decision_label(episode.target_state) == 1
                else (twin, episode)
            )
            samples.append(
                _make_sample(
                    positive, horizon, namespace=namespace,
                    pair_index=pair_index, twin_index=0,
                )
            )
            samples.append(
                _make_sample(
                    negative, horizon, namespace=namespace,
                    pair_index=pair_index, twin_index=1,
                )
            )
    return samples


def validate_console_samples(samples: Sequence[Sample]) -> dict[str, Any]:
    """Stage 0 label/nuisance audit (protocol section 1.6), tokenizer-free parts.

    Raises on any violation; returns descriptive counts on success.
    """
    if not samples:
        raise ValueError("no samples to validate")
    by_id = {str(s.sample_id): s for s in samples}
    if len(by_id) != len(samples):
        raise RuntimeError("duplicate sample ids in console corpus")
    prompts = [s.prompt for s in samples]
    if len(set(prompts)) != len(prompts):
        raise RuntimeError("duplicate prompts in console corpus")

    pairs: dict[str, list[Sample]] = {}
    for sample in samples:
        pairs.setdefault(str(sample.pair_id), []).append(sample)
    bad_pairs = [pid for pid, rows in pairs.items() if len(rows) != 2]
    if bad_pairs:
        raise RuntimeError(f"incomplete twin pairs: {bad_pairs[:5]}")

    other_state_counts: dict[str, int] = {}
    slot_counts: dict[int, int] = {}
    for pid, rows in pairs.items():
        first, second = sorted(rows, key=lambda s: str(s.sample_id))
        if int(first.target_label) == int(second.target_label):
            raise RuntimeError(f"twin pair {pid} does not flip the decision label")
        if str(first.counterfactual_id) != str(second.sample_id):
            raise RuntimeError(f"twin pair {pid} counterfactual link is not reciprocal")
        if str(second.counterfactual_id) != str(first.sample_id):
            raise RuntimeError(f"twin pair {pid} counterfactual link is not reciprocal")
        if int(first.expected_counterfactual_label) != int(second.target_label):
            raise RuntimeError(f"twin pair {pid} expected counterfactual label mismatch")
        for field in (
            "terminal_target", "terminal_other", "other_state", "target_slot",
            "horizon", "episode_index", "n_steps",
        ):
            if first.metadata[field] != second.metadata[field]:
                raise RuntimeError(f"twin pair {pid} nuisance mismatch on {field}")
        if first.metadata["target_state"] == second.metadata["target_state"]:
            raise RuntimeError(f"twin pair {pid} did not flip the clearance word")
        # The two prompts must differ in exactly one clearance word.
        diff = [
            (a, b)
            for a, b in zip(first.prompt.split("\n"), second.prompt.split("\n"))
            if a != b
        ]
        if len(diff) != 1:
            raise RuntimeError(
                f"twin pair {pid} differs on {len(diff)} lines; expected exactly one"
            )
        line_a, line_b = diff[0]
        if line_a.replace(GRANTED, DENIED) != line_b.replace(GRANTED, DENIED):
            raise RuntimeError(
                f"twin pair {pid} differing line varies beyond the clearance word"
            )
        other = str(first.metadata["other_state"])
        slot = int(first.metadata["target_slot"])
        other_state_counts[other] = other_state_counts.get(other, 0) + 1
        slot_counts[slot] = slot_counts.get(slot, 0) + 1

    for sample in samples:
        if decision_label(str(sample.metadata["target_state"])) != int(sample.target_label):
            raise RuntimeError(f"label oracle mismatch for {sample.sample_id}")
        for field in (
            "target_clearance_line", "irrelevant_clearance_line", "late_gap_line",
        ):
            line = str(sample.metadata[field])
            if sample.prompt.count(line) != 1:
                raise RuntimeError(
                    f"carrier span {field} is not unique in {sample.sample_id}"
                )
        for word in STATE_WORDS:
            if str(sample.metadata["late_gap_line"]).count(word):
                raise RuntimeError("a distractor line leaked a clearance word")

    return {
        "n_samples": len(samples),
        "n_pairs": len(pairs),
        "irrelevant_state_counts": dict(sorted(other_state_counts.items())),
        "target_slot_counts": {str(k): v for k, v in sorted(slot_counts.items())},
        "horizons": sorted({int(s.metadata["horizon"]) for s in samples}),
    }
