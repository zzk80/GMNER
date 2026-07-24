"""Configuration schema for the M3.3A region-evidence visibility head."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gmner.models.evidence_visibility import EvidenceVisibilityHeadConfig


@dataclass
class EvidenceVisibilityDataConfig:
    formal_train_cache: str
    expanded_train_cache: str
    formal_dev_cache: str
    expanded_dev_cache: str
    num_workers: int = 0
    require_oof_train_cache: bool = False


@dataclass
class EvidenceVisibilityFrozenConfig:
    fine_config: str
    fine_checkpoint: str


@dataclass
class EvidenceVisibilityLossConfig:
    lambda_bce: float = 1.0
    lambda_visible_correction: float = 1.0
    lambda_null_preservation: float = 1.0
    lambda_keep: float = 0.5
    lambda_residual: float = 0.05
    visible_correction_group_weight: float = 0.35
    visible_preservation_group_weight: float = 0.15
    null_correction_group_weight: float = 0.20
    null_preservation_group_weight: float = 0.30
    visible_margin_gamma: float = 1.0
    uncertainty_entropy_threshold: float = 0.65
    uncertainty_margin_threshold: float = 0.08


@dataclass
class EvidenceVisibilityOptimConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class EvidenceVisibilityRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = "outputs/fmnerg_roberta128_evidence_visibility"
    save_best_metric: str = "gmner_score"
    save_best_tie_breakers: list[str] = field(
        default_factory=lambda: [
            "null_correct_preservation_rate",
            "visibility_net_correction",
        ]
    )
    early_stop_patience: int = 3
    log_every_steps: int = 50


@dataclass
class EvidenceVisibilityTrainingConfig:
    data: EvidenceVisibilityDataConfig
    frozen: EvidenceVisibilityFrozenConfig
    model: EvidenceVisibilityHeadConfig = field(
        default_factory=EvidenceVisibilityHeadConfig
    )
    loss: EvidenceVisibilityLossConfig = field(
        default_factory=EvidenceVisibilityLossConfig
    )
    optim: EvidenceVisibilityOptimConfig = field(
        default_factory=EvidenceVisibilityOptimConfig
    )
    runtime: EvidenceVisibilityRuntimeConfig = field(
        default_factory=EvidenceVisibilityRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_evidence_visibility_config(
    path: str | Path,
) -> EvidenceVisibilityTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("data", "frozen"):
        if section not in raw:
            raise ValueError(
                f"Evidence visibility config requires a {section} section."
            )
    config = EvidenceVisibilityTrainingConfig(
        data=EvidenceVisibilityDataConfig(**raw["data"]),
        frozen=EvidenceVisibilityFrozenConfig(**raw["frozen"]),
        model=EvidenceVisibilityHeadConfig(**raw.get("model", {})),
        loss=EvidenceVisibilityLossConfig(**raw.get("loss", {})),
        optim=EvidenceVisibilityOptimConfig(**raw.get("optim", {})),
        runtime=EvidenceVisibilityRuntimeConfig(**raw.get("runtime", {})),
    )
    if config.model.residual_scale <= 0:
        raise ValueError("model.residual_scale must be positive.")
    group_weights = (
        config.loss.visible_correction_group_weight,
        config.loss.visible_preservation_group_weight,
        config.loss.null_correction_group_weight,
        config.loss.null_preservation_group_weight,
    )
    if any(value < 0 for value in group_weights) or sum(group_weights) <= 0:
        raise ValueError("Visibility supervision group weights must be nonnegative.")
    return config
