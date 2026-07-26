"""Configuration schema for the trainable FMNERG text-encoder sidecar."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .model import HEAD_ARCHITECTURES


ENCODER_SCOPES = ("frozen", "last_n", "all")


@dataclass
class SubtypeEncoderDataConfig:
    train_source: str
    dev_source: str
    dev_formal_predictions: str


@dataclass
class SubtypeEncoderInitializationConfig:
    stage1_config: str
    stage1_checkpoint: str


@dataclass
class SubtypeEncoderModelConfig:
    input_size: int = 2304
    hidden_size: int = 768
    dropout: float = 0.1
    head_architecture: str = "shared_hard"
    parent_hidden_size: int | None = 192
    encoder_scope: str = "last_n"
    unfreeze_last_n_layers: int = 4
    upper_layer_count: int = 4
    gradient_checkpointing: bool = False


@dataclass
class SubtypeEncoderOptimConfig:
    batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    head_learning_rate: float = 1e-4
    backbone_upper_learning_rate: float = 5e-6
    backbone_lower_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    num_epochs: int = 15
    warmup_ratio: float = 0.1
    early_stop_patience: int = 3
    gradient_clip_norm: float = 1.0


@dataclass
class SubtypeEncoderRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    output_dir: str = "outputs/fmnerg_roberta128_subtype_encoder"
    expected_dev_gmner_f1: float | None = 0.621316
    expected_dev_gmner_tolerance: float = 5e-7


@dataclass
class SubtypeEncoderConfig:
    taxonomy: str
    data: SubtypeEncoderDataConfig
    initialization: SubtypeEncoderInitializationConfig
    model: SubtypeEncoderModelConfig = field(
        default_factory=SubtypeEncoderModelConfig
    )
    optim: SubtypeEncoderOptimConfig = field(
        default_factory=SubtypeEncoderOptimConfig
    )
    runtime: SubtypeEncoderRuntimeConfig = field(
        default_factory=SubtypeEncoderRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_subtype_encoder_config(
    path: str | Path,
) -> SubtypeEncoderConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("taxonomy", "data", "initialization"):
        if section not in raw:
            raise ValueError(
                f"Subtype encoder config requires {section!r}."
            )
    config = SubtypeEncoderConfig(
        taxonomy=str(raw["taxonomy"]),
        data=SubtypeEncoderDataConfig(**raw["data"]),
        initialization=SubtypeEncoderInitializationConfig(
            **raw["initialization"]
        ),
        model=SubtypeEncoderModelConfig(**raw.get("model", {})),
        optim=SubtypeEncoderOptimConfig(**raw.get("optim", {})),
        runtime=SubtypeEncoderRuntimeConfig(**raw.get("runtime", {})),
    )
    if config.model.input_size <= 0 or config.model.hidden_size <= 0:
        raise ValueError("Subtype encoder dimensions must be positive.")
    if not 0 <= config.model.dropout < 1:
        raise ValueError("Subtype encoder dropout must be in [0, 1).")
    if config.model.head_architecture not in HEAD_ARCHITECTURES:
        raise ValueError(
            f"Unknown subtype head: {config.model.head_architecture!r}."
        )
    if (
        config.model.parent_hidden_size is not None
        and config.model.parent_hidden_size <= 0
    ):
        raise ValueError("parent_hidden_size must be positive when configured.")
    if config.model.encoder_scope not in ENCODER_SCOPES:
        raise ValueError(
            f"encoder_scope must be one of {ENCODER_SCOPES}."
        )
    if config.model.unfreeze_last_n_layers <= 0:
        raise ValueError("unfreeze_last_n_layers must be positive.")
    if config.model.upper_layer_count <= 0:
        raise ValueError("upper_layer_count must be positive.")
    if (
        config.optim.batch_size <= 0
        or config.optim.eval_batch_size <= 0
        or config.optim.gradient_accumulation_steps <= 0
        or config.optim.num_epochs <= 0
    ):
        raise ValueError("Batch sizes, accumulation, and epochs must be positive.")
    for value in (
        config.optim.head_learning_rate,
        config.optim.backbone_upper_learning_rate,
        config.optim.backbone_lower_learning_rate,
    ):
        if value <= 0:
            raise ValueError("All configured learning rates must be positive.")
    if config.optim.weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if not 0 <= config.optim.warmup_ratio < 1:
        raise ValueError("warmup_ratio must be in [0, 1).")
    if config.optim.early_stop_patience <= 0:
        raise ValueError("early_stop_patience must be positive.")
    if config.runtime.expected_dev_gmner_tolerance < 0:
        raise ValueError("expected_dev_gmner_tolerance must be non-negative.")
    return config
