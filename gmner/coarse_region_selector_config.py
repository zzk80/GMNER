"""Configuration schema for the recall-preserving coarse region selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gmner.models.coarse_region_selector import CoarseRegionSelectorConfig


@dataclass
class CoarseSelectorDataConfig:
    train_cache: str
    dev_cache: str
    num_workers: int = 0


@dataclass
class CoarseSelectorPolicyConfig:
    expanded_budget: int = 36
    final_budget: int = 16
    base_keep_values: list[int] = field(default_factory=lambda: [8, 10])


@dataclass
class CoarseSelectorLossConfig:
    lambda_multi_positive: float = 1.0
    lambda_iou: float = 0.2
    lambda_correction_margin: float = 0.5
    lambda_preservation_margin: float = 0.5
    correction_margin: float = 0.2
    preservation_margin: float = 0.2
    correction_group_weight: float = 0.5
    preservation_group_weight: float = 0.5
    iou_temperature: float = 0.1


@dataclass
class CoarseSelectorOptimConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class CoarseSelectorRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = "outputs/fmnerg_roberta128_coarse_selector"
    save_best_metric: str = "union_base8_learned8_recall_eligible"
    save_best_tie_breaker: str = "union_base8_learned8_top16_preservation"
    early_stop_patience: int = 3
    log_every_steps: int = 50


@dataclass
class CoarseRegionSelectorTrainingConfig:
    data: CoarseSelectorDataConfig
    model: CoarseRegionSelectorConfig = field(
        default_factory=CoarseRegionSelectorConfig
    )
    policy: CoarseSelectorPolicyConfig = field(
        default_factory=CoarseSelectorPolicyConfig
    )
    loss: CoarseSelectorLossConfig = field(
        default_factory=CoarseSelectorLossConfig
    )
    optim: CoarseSelectorOptimConfig = field(
        default_factory=CoarseSelectorOptimConfig
    )
    runtime: CoarseSelectorRuntimeConfig = field(
        default_factory=CoarseSelectorRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_coarse_region_selector_config(
    path: str | Path,
) -> CoarseRegionSelectorTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "data" not in raw:
        raise ValueError("Coarse selector config requires a data section.")
    return CoarseRegionSelectorTrainingConfig(
        data=CoarseSelectorDataConfig(**raw["data"]),
        model=CoarseRegionSelectorConfig(**raw.get("model", {})),
        policy=CoarseSelectorPolicyConfig(**raw.get("policy", {})),
        loss=CoarseSelectorLossConfig(**raw.get("loss", {})),
        optim=CoarseSelectorOptimConfig(**raw.get("optim", {})),
        runtime=CoarseSelectorRuntimeConfig(**raw.get("runtime", {})),
    )
