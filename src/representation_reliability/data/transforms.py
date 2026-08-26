"""Dataset transformations.

Two families per docs/DATASETS.md:

- **invariant** transformations keep the target label fixed (paraphrase);
- **controlled-change** transformations change the label predictably
  (query entity swap flips the truth value of the relational query).

Every transformed sample records parent sample id, transform class, and the
expected relation between old and new targets.  Note: negation is treated as
controlled-change wherever it changes the queried truth value.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any

from ..contracts import Sample
from .base import find_subspan
from .synthetic import RELATION_FAMILIES, RelationFamily, make_fact_sample

# Invariant paraphrase variants (premise re-wording only; label unchanged).
_PARAPHRASE_PREMISES: dict[str, tuple[str, ...]] = {
    "north_south": ("PREM_A lies PREM_W of PREM_B.",),
    "east_west": ("PREM_A lies PREM_W of PREM_B.",),
    "above_below": ("PREM_A is placed PREM_W PREM_B.",),
    "before_after": ("Event PREM_A occurred PREM_W event PREM_B.",),
    "larger_smaller": ("PREM_A measures PREM_W than PREM_B.",),
}


def _sample_family(sample: Sample) -> RelationFamily:
    key = sample.metadata.get("relation")
    if key not in RELATION_FAMILIES:
        raise ValueError(f"sample has unknown relation family {key!r}")
    return RELATION_FAMILIES[key]


def transform_paraphrase(sample: Sample, seed: int) -> Sample:
    """Invariant paraphrase: same fact, same label, different premise wording.

    Returns a new sample with a derived id, parent linkage recorded.
    """
    fam = _sample_family(sample)
    rng = random.Random(seed)
    variants = _PARAPHRASE_PREMISES[fam.key]
    template_idx = rng.randrange(len(variants))

    entity_a = sample.metadata["entity_a"]
    entity_b = sample.metadata["entity_b"]
    premise_t = variants[template_idx]
    question_t = sample.prompt.split("\nQuestion: ")[1].split("\nAnswer:")[0]

    def fill(template: str) -> str:
        return (
            template.replace("PREM_A", entity_a)
            .replace("PREM_B", entity_b)
            .replace("PREM_W", fam.pos_word)
        )

    premise = fill(premise_t)
    prompt = (
        "Premise: " + premise + "\nQuestion: " + question_t + "\nAnswer:"
    )
    meta: dict[str, Any] = dict(sample.metadata)
    meta.update({
        "premise": premise,
        "parent_sample_id": sample.sample_id,
        "transform_name": "paraphrase",
        "transform_class": "invariant",
        "expected_label_relation": "same",
        "target_text": question_t,
    })

    span = find_subspan(prompt, question_t)
    assert span is not None
    return replace(
        sample,
        sample_id=f"{sample.sample_id}-para{template_idx}",
        prompt=prompt,
        metadata=meta,
    )


def transform_query_entity_swap(sample: Sample) -> Sample:
    """Controlled-change swap: same fact & word, mirrored query order.

    The label flips predictably (truth value of the mirrored query).
    """
    fam = _sample_family(sample)
    side = sample.metadata.get("queried_side")
    mirror = {"ab": "ba", "ba": "ab"}[side]
    new_sample = make_fact_sample(
        family=fam,
        entity_a=sample.metadata["entity_a"],
        entity_b=sample.metadata["entity_b"],
        side=mirror,
        queried_word=sample.metadata["queried_word"],
        premise_idx=0,
        question_idx=int(
            sample.metadata.get("question_variant", ".q0").split("q")[-1]
        ),
        pair_id=sample.pair_id or sample.sample_id,
        sample_index=-1,
    )
    expected = int(sample.expected_counterfactual_label)
    if new_sample.target_label != expected:
        raise AssertionError(
            "query_entity_swap produced an unexpected label: "
            f"{new_sample.target_label} != {expected}"
        )
    meta = dict(new_sample.metadata)
    meta.update({
        "parent_sample_id": sample.sample_id,
        "transform_name": "query_entity_swap",
        "transform_class": "controlled_change",
        "expected_label_relation": "flipped",
    })
    from dataclasses import replace

    return replace(new_sample, metadata=meta)


TRANSFORM_REGISTRY = {
    "paraphrase": (transform_paraphrase, "invariant"),
    "query_entity_swap": (transform_query_entity_swap, "controlled_change"),
}
