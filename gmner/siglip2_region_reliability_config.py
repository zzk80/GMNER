"""Configuration for the dev-only M3.4A SigLIP 2 reliability study."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gmner.models.siglip2_region_reliability import (
    FEATURE_MODES,
    Siglip2RegionReliabilityHeadConfig,
)


@dataclass
class Siglip2ReliabilityDataConfig:
    formal_train_cache: str
    expanded_train_cache: str
    formal_dev_cache: str
    expanded_dev_cache: str
    siglip2_train_cache: str | None = None
    siglip2_dev_cache: str | None = None
    num_workers: int = 0
    require_oof_train_cache: bool = False
    verify_siglip2_cache_hashes: bool = True


@dataclass
class Siglip2ReliabilityFrozenConfig:
    fine_config: str
    fine_checkpoint: str
    evidence_visibility_config: str
    evidence_visibility_checkpoint: str


@dataclass
class Siglip2ReliabilityLossConfig:
    lambda_quality_focal: float = 1.0
    lambda_positive_max: float = 0.5
    lambda_null_suppress: float = 0.5
    lambda_rank: float = 0.5
    lambda_brier: float = 0.2
    lambda_hard_ab_bce: float = 1.0
    lambda_hard_ab_rank: float = 0.5
    quality_focal_gamma: float = 2.0
    rank_margin: float = 0.5
    hard_ab_rank_margin: float = 0.5
    low_iou: float = 0.1
    positive_iou: float = 0.5
    hard_negative_count: int = 4
    other_entity_negative_count: int = 2
    compatibility_negative_count: int = 2
    group_a_weight: float = 0.30
    group_b_weight: float = 0.30
    group_null_weight: float = 0.20
    group_ordinary_weight: float = 0.20
    positive_pair_weight: float = 2.0
    high_score_negative_weight: float = 2.0
    promoted_negative_weight: float = 1.5
    other_entity_negative_weight: float = 1.5
    compatibility_negative_weight: float = 1.5


@dataclass
class Siglip2ReliabilityEvaluationConfig:
    reliability_threshold: float = 0.5
    null_preservation_floor: float = 0.98
    calibration_bins: int = 10
    detector_reference_budget: int = 16
    minimum_hard_ab_auc: float = 0.70
    minimum_balanced_accuracy: float = 0.62
    minimum_risk_net_correction: int = 15
    minimum_promoted_fix_count: int = 1


@dataclass
class Siglip2ReliabilityOptimConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class Siglip2ReliabilityRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = "outputs/fmnerg_roberta128_siglip2_reliability"
    early_stop_patience: int = 3
    log_every_steps: int = 50


@dataclass
class Siglip2ReliabilityTrainingConfig:
    data: Siglip2ReliabilityDataConfig
    frozen: Siglip2ReliabilityFrozenConfig
    model: Siglip2RegionReliabilityHeadConfig = field(
        default_factory=Siglip2RegionReliabilityHeadConfig
    )
    loss: Siglip2ReliabilityLossConfig = field(
        default_factory=Siglip2ReliabilityLossConfig
    )
    evaluation: Siglip2ReliabilityEvaluationConfig = field(
        default_factory=Siglip2ReliabilityEvaluationConfig
    )
    optim: Siglip2ReliabilityOptimConfig = field(
        default_factory=Siglip2ReliabilityOptimConfig
    )
    runtime: Siglip2ReliabilityRuntimeConfig = field(
        default_factory=Siglip2ReliabilityRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_siglip2_region_reliability_config(
    path: str | Path,
) -> Siglip2ReliabilityTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("data", "frozen"):
        if section not in raw:
            raise ValueError(f"M3.4A config requires a {section} section.")
    config = Siglip2ReliabilityTrainingConfig(
        data=Siglip2ReliabilityDataConfig(**raw["data"]),
        frozen=Siglip2ReliabilityFrozenConfig(**raw["frozen"]),
        model=Siglip2RegionReliabilityHeadConfig(**raw.get("model", {})),
        loss=Siglip2ReliabilityLossConfig(**raw.get("loss", {})),
        evaluation=Siglip2ReliabilityEvaluationConfig(
            **raw.get("evaluation", {})
        ),
        optim=Siglip2ReliabilityOptimConfig(**raw.get("optim", {})),
        runtime=Siglip2ReliabilityRuntimeConfig(**raw.get("runtime", {})),
    )
    if config.model.feature_mode not in FEATURE_MODES:
        raise ValueError(
            f"model.feature_mode must be one of {sorted(FEATURE_MODES)}."
        )
    needs_siglip2 = config.model.feature_mode != "vinvl_only"
    if needs_siglip2 and (
        not config.data.siglip2_train_cache
        or not config.data.siglip2_dev_cache
    ):
        raise ValueError(
            f"{config.model.feature_mode} requires train/dev SigLIP 2 caches."
        )
    weights = (
        config.loss.group_a_weight,
        config.loss.group_b_weight,
        config.loss.group_null_weight,
        config.loss.group_ordinary_weight,
    )
    if any(value < 0.0 for value in weights) or sum(weights) <= 0.0:
        raise ValueError("Reliability group weights must be nonnegative.")
    if not 0.0 < config.loss.positive_iou <= 1.0:
        raise ValueError("loss.positive_iou must be in (0, 1].")
    if config.loss.low_iou >= config.loss.positive_iou:
        raise ValueError("loss.low_iou must be below loss.positive_iou.")
    if not 0.0 < config.evaluation.reliability_threshold < 1.0:
        raise ValueError("evaluation.reliability_threshold must be in (0, 1).")
    if not 0.0 <= config.evaluation.null_preservation_floor <= 1.0:
        raise ValueError("evaluation.null_preservation_floor must be in [0, 1].")
    if config.model.siglip2_candidate_temperature <= 0.0:
        raise ValueError("model.siglip2_candidate_temperature must be positive.")
    return config
