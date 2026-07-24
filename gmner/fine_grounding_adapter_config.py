"""Configuration schema for the M3.2 fine grounding adapter."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gmner.models.fine_grounding_adapter import FineGroundingAdapterConfig


@dataclass
class FineGroundingDataConfig:
    formal_train_cache: str
    expanded_train_cache: str
    formal_dev_cache: str
    expanded_dev_cache: str
    num_workers: int = 0
    require_oof_train_cache: bool = False


@dataclass
class FineGroundingFrozenConfig:
    hierarchical_config: str
    hierarchical_checkpoint: str
    coarse_checkpoint: str


@dataclass
class FineGroundingLossConfig:
    lambda_multi_positive: float = 1.0
    lambda_iou: float = 0.2
    lambda_correction_margin: float = 1.0
    lambda_preservation_margin: float = 0.5
    lambda_residual: float = 0.05
    correction_margin: float = 0.5
    preservation_margin: float = 0.2
    iou_temperature: float = 0.1
    correction_group_weight: float = 0.4
    preservation_group_weight: float = 0.4
    other_group_weight: float = 0.2
    promoted_correction_fraction: float = 0.5


@dataclass
class FineGroundingOptimConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class FineGroundingRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = "outputs/fmnerg_roberta128_fine_grounding_adapter"
    save_best_metric: str = "gmner_score"
    save_best_tie_breakers: list[str] = field(
        default_factory=lambda: [
            "visible_net_correction",
            "base_correct_preservation_rate",
            "promoted_gold_recovery_rate",
        ]
    )
    early_stop_patience: int = 3
    log_every_steps: int = 50


@dataclass
class FineGroundingTrainingConfig:
    data: FineGroundingDataConfig
    frozen: FineGroundingFrozenConfig
    model: FineGroundingAdapterConfig = field(
        default_factory=FineGroundingAdapterConfig
    )
    loss: FineGroundingLossConfig = field(
        default_factory=FineGroundingLossConfig
    )
    optim: FineGroundingOptimConfig = field(
        default_factory=FineGroundingOptimConfig
    )
    runtime: FineGroundingRuntimeConfig = field(
        default_factory=FineGroundingRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_fine_grounding_adapter_config(
    path: str | Path,
) -> FineGroundingTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("data", "frozen"):
        if section not in raw:
            raise ValueError(f"Fine grounding config requires a {section} section.")
    config = FineGroundingTrainingConfig(
        data=FineGroundingDataConfig(**raw["data"]),
        frozen=FineGroundingFrozenConfig(**raw["frozen"]),
        model=FineGroundingAdapterConfig(**raw.get("model", {})),
        loss=FineGroundingLossConfig(**raw.get("loss", {})),
        optim=FineGroundingOptimConfig(**raw.get("optim", {})),
        runtime=FineGroundingRuntimeConfig(**raw.get("runtime", {})),
    )
    if config.model.base_keep < 0 or config.model.base_keep > config.model.final_budget:
        raise ValueError("model.base_keep must be within the final candidate budget.")
    if config.model.final_budget <= 0:
        raise ValueError("model.final_budget must be positive.")
    return config
