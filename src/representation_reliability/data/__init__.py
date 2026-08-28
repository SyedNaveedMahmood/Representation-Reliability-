"""Data subsystem: datasets, splits, transformations."""

from .base import (
    answer_to_label,
    check_label_balance,
    find_subspan,
    render_completion_prompt,
    samples_to_dataframe,
    sha256_text,
)
from .splits import (
    SPLIT_NAMES,
    ConfirmationSplitAccessError,
    DEFAULT_Split_FRACTIONS,
    apply_splits,
    assign_group_splits,
    confirmation_view,
    discovery_view,
    require_confirmation_access,
    validate_splits,
)
from .synthetic import (
    ENTITIES,
    RELATION_FAMILIES,
    TASK_NAME,
    RelationFamily,
    generate_synthetic_relations,
    make_fact_sample,
    query_label,
)
from .transforms import (
    TRANSFORM_REGISTRY,
    transform_paraphrase,
    transform_query_entity_swap,
)

__all__ = [
    "ENTITIES",
    "RELATION_FAMILIES",
    "SPLIT_NAMES",
    "TASK_NAME",
    "TRANSFORM_REGISTRY",
    "ConfirmationSplitAccessError",
    "DEFAULT_Split_FRACTIONS",
    "RelationFamily",
    "answer_to_label",
    "apply_splits",
    "assign_group_splits",
    "check_label_balance",
    "confirmation_view",
    "discovery_view",
    "find_subspan",
    "generate_synthetic_relations",
    "make_fact_sample",
    "query_label",
    "render_completion_prompt",
    "require_confirmation_access",
    "samples_to_dataframe",
    "sha256_text",
    "transform_paraphrase",
    "transform_query_entity_swap",
    "validate_splits",
]
