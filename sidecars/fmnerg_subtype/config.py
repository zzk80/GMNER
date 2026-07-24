"""Configuration schema for the independent subtype sidecar."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SubtypeDataConfig:
    train_source: str
    dev_source: str
    train_gold_features: str
    dev_gold_features: str
    dev_formal_predictions: str
    dev_formal_features: str


@dataclass
class SubtypeFrozenConfig:
    stage1_config: str
    stage1_checkpoint: str
    evidence_config: str
    evidence_checkpoint: str
    formal_dev_cache: str
    expanded_dev_cache: str


@dataclass
class SubtypeModelConfig:
    input_size: int = 2304
    hidden_size: int = 768
    dropout: float = 0.1
    head_architecture: str = "shared_hard"
    parent_hidden_size: int | None = 192


@dataclass
class SubtypeOptimConfig:
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    num_epochs: int = 30
    early_stop_patience: int = 5
    gradient_clip_norm: float = 1.0
    loss_mode: str = "ce"
    effective_number_beta: float = 0.999
    parent_normalize_class_weights: bool = True


@dataclass
class SubtypeRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16_features: bool = True
    output_dir: str = "outputs/fmnerg_roberta128_subtype_sidecar"
    save_best_metric: str = "fmnerg_f1"
    expected_dev_gmner_f1: float | None = 0.621316
    expected_dev_gmner_tolerance: float = 5e-7


@dataclass
class SubtypeSidecarConfig:
    taxonomy: str
    data: SubtypeDataConfig
    frozen: SubtypeFrozenConfig
    model: SubtypeModelConfig = field(default_factory=SubtypeModelConfig)
    optim: SubtypeOptimConfig = field(default_factory=SubtypeOptimConfig)
    runtime: SubtypeRuntimeConfig = field(default_factory=SubtypeRuntimeConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_sidecar_config(path: str | Path) -> SubtypeSidecarConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("taxonomy", "data", "frozen"):
        if section not in raw:
            raise ValueError(f"Subtype sidecar config requires {section!r}.")
    config = SubtypeSidecarConfig(
        taxonomy=str(raw["taxonomy"]),
        data=SubtypeDataConfig(**raw["data"]),
        frozen=SubtypeFrozenConfig(**raw["frozen"]),
        model=SubtypeModelConfig(**raw.get("model", {})),
        optim=SubtypeOptimConfig(**raw.get("optim", {})),
        runtime=SubtypeRuntimeConfig(**raw.get("runtime", {})),
    )
    if config.model.input_size <= 0 or config.model.hidden_size <= 0:
        raise ValueError("Subtype sidecar dimensions must be positive.")
    if not 0 <= config.model.dropout < 1:
        raise ValueError("Subtype sidecar dropout must be in [0, 1).")
    if config.model.head_architecture not in {
        "shared_hard",
        "parent_specific_hard",
    }:
        raise ValueError(
            "head_architecture must be shared_hard or parent_specific_hard."
        )
    if (
        config.model.parent_hidden_size is not None
        and config.model.parent_hidden_size <= 0
    ):
        raise ValueError("parent_hidden_size must be positive when configured.")
    if config.optim.num_epochs <= 0 or config.optim.batch_size <= 0:
        raise ValueError("Subtype sidecar epochs and batch size must be positive.")
    if config.optim.loss_mode not in {
        "ce",
        "class_weighted",
        "effective_number",
    }:
        raise ValueError(
            "loss_mode must be ce, class_weighted, or effective_number."
        )
    if not 0 < config.optim.effective_number_beta < 1:
        raise ValueError("effective_number_beta must be in (0, 1).")
    if config.runtime.save_best_metric not in {
        "fine_mner_f1",
        "fmnerg_f1",
        "subtype_macro_f1_on_gold_spans",
    }:
        raise ValueError(
            "save_best_metric must be fine_mner_f1, fmnerg_f1, or "
            "subtype_macro_f1_on_gold_spans."
        )
    if config.runtime.expected_dev_gmner_tolerance < 0:
        raise ValueError("expected_dev_gmner_tolerance must be non-negative.")
    return config
