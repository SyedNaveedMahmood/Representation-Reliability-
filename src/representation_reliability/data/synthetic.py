"""Deterministic synthetic relational dataset with matched counterfactual twins.

Design intent (anti-leakage):

Every generated *fact* (``entity_a`` holds ``pos_word`` toward ``entity_b``,
e.g. "Luma is north of Reko") produces exactly two examples — a positive one
and its **counterfactual twin**::

    fact: (family=north_south, a=Luma, b=Reko), queried word = north

    twin A:  Question: Is Luma north of Reko?   -> label 1 (yes)
    twin B:  Question: Is Reko north of Luma?   -> label 0 (no)

The twins share the premise sentence, the question word, the question
template, and the entity vocabulary; ONLY the queried entity order differs,
and flipping it flips the ground-truth label.  Consequently each unigram
token distribution is identical across classes by construction, so a probe
must compose premise and question structure rather than exploit a lexical
cue.  The ``target_span_last`` selector then probes the final token of the
question, where that binding must be resolved.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ..contracts import Sample
from .base import render_completion_prompt

TASK_NAME = "synthetic_relations"

# Invented proper nouns (unambiguous entities, no real-world knowledge needed).
ENTITIES = [
    "Luma", "Reko", "Vanta", "Sidra", "Ovell", "Karsi", "Dennit", "Bravu",
    "Milqueth", "Torren", "Halvit", "Quoran", "Esmik", "Yalta", "Prue",
    "Gomar", "Neshi", "Aldive", "Cortes", "Wexley", "Iona", "Farro",
    "Zubon", "Thale", "Rhevik", "Odari", "Sammet", "Yune", "Pallor",
    "Marnet", "Jesco", "Uliah", "Bretta", "Onnavi", "Clavian", "Druse",
    "Fyodor", "Vellum", "Ashen", "Gritta", "Hobban", "Junis",
]


@dataclass(frozen=True)
class RelationFamily:
    key: str
    pos_word: str          # word that appears verbatim inside premises
    neg_word: str          # its antonym (the inverse relation)
    premise_variants: tuple[str, ...]
    question_variants: tuple[str, ...]

    def inverse(self, word: str) -> str:
        if word == self.pos_word:
            return self.neg_word
        if word == self.neg_word:
            return self.pos_word
        raise ValueError(f"word {word!r} not in family {self.key}")


RELATION_FAMILIES: dict[str, RelationFamily] = {
    fam.key: fam
    for fam in [
        RelationFamily(
            key="north_south",
            pos_word="north",
            neg_word="south",
            premise_variants=("PREM_A is PREM_W of PREM_B.",),
            question_variants=("Is SUBJ Q_W OBJ?",),
        ),
        RelationFamily(
            key="east_west",
            pos_word="east",
            neg_word="west",
            premise_variants=("PREM_A is PREM_W of PREM_B.",),
            question_variants=("Is SUBJ Q_W OBJ?",),
        ),
        RelationFamily(
            key="above_below",
            pos_word="above",
            neg_word="below",
            premise_variants=(
                "PREM_A sits PREM_W PREM_B.",
                "PREM_A is located PREM_W PREM_B.",
            ),
            question_variants=("Is SUBJ Q_W OBJ?",),
        ),
        RelationFamily(
            key="before_after",
            pos_word="before",
            neg_word="after",
            premise_variants=("Event PREM_A happens PREM_W event PREM_B.",),
            question_variants=("Does event SUBJ happen Q_W event OBJ?",),
        ),
        RelationFamily(
            key="larger_smaller",
            pos_word="larger",
            neg_word="smaller",
            premise_variants=("PREM_A is PREM_W than PREM_B.",),
            question_variants=("Is SUBJ Q_W than OBJ?",),
        ),
    ]
}


# Semantic oracle -----------------------------------------------------------
#
# A fact (family, a, b) asserts ``a --pos_word--> b``.
# Query word w with subject/order side s in {"ab","ba"} is TRUE iff:
#   s == "ab" and w == pos_word
#   s == "ba" and w == neg_word


def query_label(family: RelationFamily, side: str, word: str) -> int:
    """Ground-truth binary label for a query against a held fact."""
    if side == "ab":
        return int(word == family.pos_word)
    elif side == "ba":
        return int(word == family.neg_word)
    raise ValueError(f"side must be 'ab' or 'ba', got {side!r}")


# Mirrored-side mapping: same queried word, other entity order -> label flips.
_MIRROR_SIDE = {"ab": "ba", "ba": "ab"}


def _fill(template: str, **subs: str) -> str:
    out = template
    for key, value in subs.items():
        out = out.replace(key, value)
    return out


def make_fact_sample(
    *,
    family: RelationFamily,
    entity_a: str,
    entity_b: str,
    side: str,
    queried_word: str,
    premise_idx: int,
    question_idx: int,
    pair_id: str,
    sample_index: int,
) -> Sample:
    """Build one labeled example for one held fact."""
    if entity_a == entity_b:
        raise ValueError("entities must be distinct")
    if side not in ("ab", "ba"):
        raise ValueError("side must be 'ab' or 'ba'")
    if queried_word not in (family.pos_word, family.neg_word):
        raise ValueError(
            f"queried_word {queried_word!r} invalid for family {family.key}"
        )

    premise_t = family.premise_variants[premise_idx % len(family.premise_variants)]
    question_t = family.question_variants[question_idx % len(family.question_variants)]

    premise = _fill(
        premise_t, PREM_A=entity_a, PREM_B=entity_b, PREM_W=family.pos_word
    )
    subj, obj = (entity_a, entity_b) if side == "ab" else (entity_b, entity_a)
    question = _fill(question_t, SUBJ=subj, OBJ=obj, Q_W=queried_word)

    prompt, target_text = render_completion_prompt(premise, question)
    label = query_label(family, side, queried_word)

    meta: dict[str, Any] = {
        "entity_a": entity_a,
        "entity_b": entity_b,
        "relation": family.key,
        "inverse_relation": family.inverse(queried_word),
        "truth_label": label,
        "template_id": f"{family.key}.p{premise_idx}",
        "question_variant": f"{family.key}.q{question_idx}",
        "pair_id": pair_id,
        "queried_word": queried_word,
        "queried_side": side,
        "pos_word": family.pos_word,
        "neg_word": family.neg_word,
        "premise": premise,
        "target_text": target_text,
        "chat_template_used": False,
    }
    return Sample(
        sample_id=f"{TASK_NAME}-{sample_index:06d}",
        prompt=prompt,
        target_label=label,
        task_name=TASK_NAME,
        pair_id=pair_id,
        metadata=meta,
    )


def generate_synthetic_relations(
    n_samples: int,
    seed: int,
    n_entities: int = 32,
    families: Sequence[str] | None = None,
) -> list[Sample]:
    """Deterministically generate ``n_samples`` balanced labeled examples.

    Every second sample is the matched counterfactual twin of the previous
    one (same fact, same queried word, mirrored entity order => flipped
    label).  Twins share ``pair_id``.
    """
    if n_samples < 2 or n_samples % 2 != 0:
        raise ValueError("n_samples must be a positive even number")

    fam_keys = list(families) if families else list(RELATION_FAMILIES.keys())
    unknown = [k for k in fam_keys if k not in RELATION_FAMILIES]
    if unknown:
        raise ValueError(f"unknown relation families: {unknown}")

    rng = random.Random(seed)
    entity_pool = ENTITIES[: max(3, min(n_entities, len(ENTITIES)))]

    n_facts = n_samples // 2
    samples: list[Sample] = []
    idx = 0
    for fact_i in range(n_facts):
        fam = RELATION_FAMILIES[fam_keys[fact_i % len(fam_keys)]]
        a, b = rng.sample(entity_pool, 2)
        queried_word = fam.pos_word if rng.random() < 0.5 else fam.neg_word
        side_pos = "ab" if rng.random() < 0.5 else "ba"
        side_neg = _MIRROR_SIDE[side_pos]
        premise_idx = rng.randrange(len(fam.premise_variants))
        question_idx = rng.randrange(len(fam.question_variants))
        pair_id = f"{fam.key}-fact{fact_i:05d}"

        # Positive-label twin first, negative-label twin second.
        first_side = side_pos if query_label(fam, side_pos, queried_word) else side_neg
        second_side = _MIRROR_SIDE[first_side]

        pos_sample = make_fact_sample(
            family=fam, entity_a=a, entity_b=b, side=first_side,
            queried_word=queried_word, premise_idx=premise_idx,
            question_idx=question_idx, pair_id=pair_id, sample_index=idx,
        )
        neg_sample = make_fact_sample(
            family=fam, entity_a=a, entity_b=b, side=second_side,
            queried_word=queried_word, premise_idx=premise_idx,
            question_idx=question_idx, pair_id=pair_id, sample_index=idx + 1,
        )
        assert pos_sample.target_label == 1 and neg_sample.target_label == 0

        from dataclasses import replace as _dc_replace

        pos_sample = _dc_replace(
            pos_sample,
            counterfactual_id=neg_sample.sample_id,
            expected_counterfactual_label=0,
        )
        neg_sample = _dc_replace(
            neg_sample,
            counterfactual_id=pos_sample.sample_id,
            expected_counterfactual_label=1,
        )

        samples.append(pos_sample)
        samples.append(neg_sample)
        idx += 2

    return samples

