"""Activation extraction subsystem: token selection + portable sharded cache."""

from .activations import (
    CACHE_SCHEMA_VERSION,
    build_cache_identity,
    build_token_selections,
    dataset_content_hash,
    extract_dataset_activations,
)
from .cache import (
    ActivationCacheReader,
    ActivationShardWriter,
    CacheIdentityMismatchError,
    LegacyCacheError,
    completed_shard_ids,
    ensure_cache_identity,
    read_marker,
    shard_dir,
    shard_is_complete,
)
from .token_selection import SELECTOR_NAMES, ResolvedToken, resolve_token_selection

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "SELECTOR_NAMES",
    "ActivationCacheReader",
    "ActivationShardWriter",
    "CacheIdentityMismatchError",
    "LegacyCacheError",
    "ResolvedToken",
    "build_cache_identity",
    "build_token_selections",
    "completed_shard_ids",
    "dataset_content_hash",
    "ensure_cache_identity",
    "extract_dataset_activations",
    "read_marker",
    "resolve_token_selection",
    "shard_dir",
    "shard_is_complete",
]

