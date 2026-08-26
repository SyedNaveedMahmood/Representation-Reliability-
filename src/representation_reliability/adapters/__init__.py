"""Model adapters. External model internals stay behind these APIs."""

from .hf import HFAdapter, SiteResolution

__all__ = ["HFAdapter", "SiteResolution"]
