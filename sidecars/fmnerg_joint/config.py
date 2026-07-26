"""Configuration for the isolated J0 subtype-region fusion experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


EXPERIMENT_MODES = ("visual_fusion", "text_continuation")


@dataclass
class JointSubtypeDataConfig:
    train_source: str
    dev_source: str
    train_expanded_cache: str
    dev_expanded_cache: str
    dev_formal_predictions: str


@dataclass
class JointSubtypeInitializationConfig:
    subtype_encoder_config: str
    subtype_checkpoint_pattern: str


@dataclass
class JointSubtypeModelConfig:
    stage: str = "j0"
    experiment_mode: str = "visual_fusion"
    text_feature_size: int = 2304
    region_feature_size: int = 768
    geometry_size: int = 4
    hidden_size: int = 768
    dropout: float = 0.1
    residual_scale: float = 2.0


@dataclass
class JointSubtypeLossConfig:
    lambda_fused: float = 1.0
    lambda_text: float = 0.25
    lambda_residual: float = 0.01


@dataclass
class JointSubtypeOptimConfig:
    batch_size: int = 8
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 1
    fusion_learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 12
    warmup_ratio: float = 0.1
    early_stop_patience: int = 3
    gradient_clip_norm: float = 1.0


@dataclass
class JointSubtypeRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    mixed_precision: bool = True
    output_dir: str = "outputs/fmnerg_joint_j0_seed42"
    expected_dev_gmner_f1: float = 0.621316108
    expected_dev_gmner_tolerance: float = 5e-7
    expected_initial_fmnerg_f1: float | None = None
    expected_initial_fmnerg_tolerance: float = 5e-7


@dataclass
class JointSubtypeConfig:
    taxonomy: str
    data: JointSubtypeDataConfig
    initialization: JointSubtypeInitializationConfig
    model: JointSubtypeModelConfig = field(
        default_factory=JointSubtypeModelConfig
    )
    loss: JointSubtypeLossConfig = field(
        default_factory=JointSubtypeLossConfig
    )
    optim: JointSubtypeOptimConfig = field(
        default_factory=JointSubtypeOptimConfig
    )
    runtime: JointSubtypeRuntimeConfig = field(
        default_factory=JointSubtypeRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def subtype_checkpoint(self, seed: int) -> str:
        try:
            return self.initialization.subtype_checkpoint_pattern.format(
                seed=int(seed)
            )
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "subtype_checkpoint_pattern must support the {seed} field."
            ) from exc


def load_joint_subtype_config(path: str | Path) -> JointSubtypeConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("taxonomy", "data", "initialization"):
        if section not in raw:
            raise ValueError(
                f"Joint subtype config requires {section!r}."
            )
    config = JointSubtypeConfig(
        taxonomy=str(raw["taxonomy"]),
        data=JointSubtypeDataConfig(**raw["data"]),
        initialization=JointSubtypeInitializationConfig(
            **raw["initialization"]
        ),
        model=JointSubtypeModelConfig(**raw.get("model", {})),
        loss=JointSubtypeLossConfig(**raw.get("loss", {})),
        optim=JointSubtypeOptimConfig(**raw.get("optim", {})),
        runtime=JointSubtypeRuntimeConfig(**raw.get("runtime", {})),
    )
    if config.model.stage != "j0":
        raise ValueError(
            "Only the frozen-region J0 stage is implemented. J1/J2 must use "
            "separate, explicitly preregistered configs."
        )
    if config.model.experiment_mode not in EXPERIMENT_MODES:
        raise ValueError(
            f"model.experiment_mode must be one of {EXPERIMENT_MODES}."
        )
    for name, value in (
        ("text_feature_size", config.model.text_feature_size),
        ("region_feature_size", config.model.region_feature_size),
        ("geometry_size", config.model.geometry_size),
        ("hidden_size", config.model.hidden_size),
    ):
        if int(value) <= 0:
            raise ValueError(f"model.{name} must be positive.")
    if not 0 <= config.model.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1).")
    if config.model.residual_scale <= 0:
        raise ValueError("model.residual_scale must be positive.")
    for name, value in asdict(config.loss).items():
        if float(value) < 0:
            raise ValueError(f"loss.{name} must be non-negative.")
    if config.loss.lambda_fused <= 0:
        raise ValueError("loss.lambda_fused must be positive.")
    if (
        config.optim.batch_size <= 0
        or config.optim.eval_batch_size <= 0
        or config.optim.gradient_accumulation_steps <= 0
        or config.optim.num_epochs <= 0
    ):
        raise ValueError(
            "Batch sizes, accumulation, and epochs must be positive."
        )
    if config.optim.fusion_learning_rate <= 0:
        raise ValueError("optim.fusion_learning_rate must be positive.")
    if config.optim.weight_decay < 0:
        raise ValueError("optim.weight_decay must be non-negative.")
    if not 0 <= config.optim.warmup_ratio < 1:
        raise ValueError("optim.warmup_ratio must be in [0, 1).")
    if config.optim.early_stop_patience <= 0:
        raise ValueError("optim.early_stop_patience must be positive.")
    if config.runtime.expected_dev_gmner_tolerance < 0:
        raise ValueError(
            "runtime.expected_dev_gmner_tolerance must be non-negative."
        )
    if config.runtime.expected_initial_fmnerg_tolerance < 0:
        raise ValueError(
            "runtime.expected_initial_fmnerg_tolerance must be non-negative."
        )
    config.subtype_checkpoint(config.runtime.seed)
    return config
