"""Strict Pydantic schemas for the resolved experiment configuration.

Every nested model forbids unknown keys so that scientific configuration
parameters cannot silently disappear.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SITE_NAMES = ("resid_pre", "attn_out", "mlp_out", "resid_post")
TOKEN_SELECTOR_NAMES = ("last_prompt", "target_span_last", "explicit")
SPLIT_NAMES = ("train", "validation", "discovery_test", "confirmation")
SUPPORTED_DTYPES = ("bfloat16", "float16", "float32")


class StrictModel(BaseModel):
    """Base model that rejects unknown fields."""

    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    name: str
    output_root: Path = Path("runs")
    cache_root: Path = Path("data/cache")


class ReproducibilityConfig(StrictModel):
    seed: int
    deterministic: bool = False
    save_environment: bool = True
    save_git_state: bool = True
    # Separate recorded seeds per stage; None -> derived from `seed`.
    data_seed: int | None = None
    split_seed: int | None = None
    probe_seed: int | None = None
    control_seed: int | None = None
    bootstrap_seed: int | None = None


class RuntimeConfig(StrictModel):
    device: str = "cuda"
    dtype: str = "bfloat16"
    batch_size: int = Field(gt=0)
    resume: bool = True
    num_workers: int = 0

    @field_validator("dtype")
    @classmethod
    def _check_dtype(cls, v: str) -> str:
        if v not in SUPPORTED_DTYPES:
            raise ValueError(
                f"unsupported dtype {v!r}; supported: {list(SUPPORTED_DTYPES)}"
            )
        return v


class BudgetConfig(StrictModel):
    max_gpu_hours: float = Field(gt=0)
    max_examples: int = Field(gt=0)


class StorageConfig(StrictModel):
    activation_format: Literal["safetensors"] = "safetensors"
    metadata_format: Literal["parquet"] = "parquet"
    shard_size: int = Field(gt=0)
    save_raw_predictions: bool = True
    save_raw_interventions: bool = True


class StatisticsConfig(StrictModel):
    bootstrap_samples: int = Field(gt=0)
    confidence_level: float = Field(gt=0, lt=1)


class GenerationConfig(StrictModel):
    do_sample: bool = False
    max_new_tokens: int = Field(default=256, gt=0)


class ModelConfig(StrictModel):
    id: str
    revision: str | None = None
    family: str
    backend: Literal["hf"] = "hf"
    dtype: str = "bfloat16"
    device_map: str = "cuda"
    trust_remote_code: bool = False
    generation: GenerationConfig = GenerationConfig()

    @field_validator("dtype")
    @classmethod
    def _check_dtype(cls, v: str) -> str:
        if v not in SUPPORTED_DTYPES:
            raise ValueError(
                f"unsupported dtype {v!r}; supported: {list(SUPPORTED_DTYPES)}"
            )
        return v


class RepresentationConfig(StrictModel):
    sites: list[str]
    layers: Literal["all"] | list[int] = "all"
    token_selectors: list[str]

    @field_validator("sites")
    @classmethod
    def _check_sites(cls, v: list[str]) -> list[str]:
        bad = [s for s in v if s not in SITE_NAMES]
        if bad:
            raise ValueError(
                f"unknown representation site(s) {bad}; allowed: {list(SITE_NAMES)}"
            )
        if len(set(v)) != len(v):
            raise ValueError("duplicate sites in configuration")
        return v

    @field_validator("layers")
    @classmethod
    def _check_layers(cls, v):
        if isinstance(v, list):
            if not v:
                raise ValueError("layer list may not be empty")
            if any(int(x) < 0 for x in v):
                raise ValueError("layers are 0-indexed; negative layers are invalid")
            if len(set(v)) != len(v):
                raise ValueError("duplicate layers in configuration")
        return v

    @field_validator("token_selectors")
    @classmethod
    def _check_selectors(cls, v: list[str]) -> list[str]:
        bad = [s for s in v if s not in TOKEN_SELECTOR_NAMES]
        if bad:
            raise ValueError(
                f"unknown token selector(s) {bad}; allowed: {list(TOKEN_SELECTOR_NAMES)}"
            )
        if len(set(v)) != len(v):
            raise ValueError("duplicate token selectors in configuration")
        return v


class DatasetConfig(StrictModel):
    type: Literal["synthetic_relations"]
    n_samples: int = Field(gt=0)
    split: str = "discovery"
    invariant_transforms: list[str] = []
    controlled_change_transforms: list[str] = []
    n_entities: int = Field(default=32, gt=2)
    apply_transforms: bool = False


class ExperimentConfig(StrictModel):
    id: str
    name: str
    mode: Literal["discovery", "confirmation"] = "discovery"

    @model_validator(mode="after")
    def _require_id(self) -> ExperimentConfig:
        if not self.id or not self.id.strip():
            raise ValueError("missing experiment ID")
        return self


class ProbeConfig(StrictModel):
    type: Literal["logistic_regression"]
    standardize: bool = True
    C_grid: list[float]
    class_weight: Literal["balanced"] | None = None

    @field_validator("C_grid")
    @classmethod
    def _check_C(cls, v: list[float]) -> list[float]:
        if not v:
            raise ValueError("C_grid may not be empty")
        if any(c <= 0 for c in v):
            raise ValueError("all C values must be positive")
        return sorted(set(float(c) for c in v))


class ControlsConfig(StrictModel):
    random_labels: bool = True
    random_label_seeds: list[int] = []
    text_baseline: Literal["tfidf_logreg"] | None = None
    random_features: bool = False


class OutputsConfig(StrictModel):
    save_coefficients: bool = True
    layerwise_metrics: bool = True


class AppConfig(StrictModel):
    """Fully-resolved experiment configuration."""

    project: ProjectConfig
    reproducibility: ReproducibilityConfig
    runtime: RuntimeConfig
    budget: BudgetConfig
    storage: StorageConfig
    statistics: StatisticsConfig
    model: ModelConfig
    representation: RepresentationConfig
    dataset: DatasetConfig
    experiment: ExperimentConfig
    probe: ProbeConfig
    controls: ControlsConfig
    outputs: OutputsConfig

    @model_validator(mode="after")
    def _resolve_derived_seeds(self) -> AppConfig:
        r = self.reproducibility
        if r.data_seed is None:
            r.data_seed = r.seed
        if r.split_seed is None:
            r.split_seed = r.seed + 1
        if r.probe_seed is None:
            r.probe_seed = r.seed + 2
        if r.control_seed is None:
            r.control_seed = r.seed + 3
        if r.bootstrap_seed is None:
            r.bootstrap_seed = r.seed + 4
        return self

    def to_dict(self) -> dict:
        return self.model_dump(mode="json", exclude_none=False)

    def effective_seeds(self) -> dict[str, int]:
        return {
            "seed": self.reproducibility.seed,
            "data": self.reproducibility.data_seed,
            "split": self.reproducibility.split_seed,
            "probe": self.reproducibility.probe_seed,
            "control": self.reproducibility.control_seed,
            "bootstrap": self.reproducibility.bootstrap_seed,
        }

