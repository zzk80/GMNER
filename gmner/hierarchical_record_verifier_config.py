"""Configuration for the hierarchical record verifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gmner.models.hierarchical_record_verifier import (
    HierarchicalRecordVerifierConfig,
)


@dataclass
class HierarchicalRecordDataConfig:
    train_cache: str
    dev_cache: str
    test_cache: str | None = None
    num_workers: int = 0
    require_oof_train_cache: bool = False


@dataclass
class HierarchicalRecordOptimConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 12
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class HierarchicalRecordLossConfig:
    lambda_entity: float = 1.0
    lambda_visibility: float = 1.0
    lambda_region_multi_positive: float = 1.0
    lambda_region_iou: float = 0.2
    lambda_region_hard: float = 0.5
    lambda_region_preserve: float = 0.5
    lambda_override_utility: float = 0.0
    lambda_action_listwise: float = 0.0
    lambda_action_expected_regret: float = 0.0
    lambda_action_fix_margin: float = 0.0
    lambda_action_damage_margin: float = 0.0
    lambda_action_neutral_cost: float = 0.0
    entity_positive_weight: float = 2.0
    visibility_positive_weight: float = 2.0
    visibility_error_weight: float = 3.0
    visibility_preserve_weight: float = 1.0
    region_hard_margin: float = 0.2
    region_preserve_margin: float = 0.2
    iou_temperature: float = 0.1
    override_utility_neutral_weight: float = 0.5
    override_utility_fix_weight: float = 2.0
    override_utility_damage_weight: float = 3.0
    override_utility_require_correct_type: bool = True
    override_utility_stage1_only: bool = True
    action_fix_margin: float = 0.5
    action_damage_margin: float = 0.5
    action_neutral_margin: float = 0.05
    action_risk_damage_cost: float = 3.0
    action_risk_neutral_cost: float = 0.05
    action_hard_damage_k: int = 3
    action_hard_neutral_k: int = 2
    action_fixable_group_weight: float = 0.5
    action_preserve_group_weight: float = 0.25
    action_ordinary_group_weight: float = 0.25
    action_require_correct_type: bool = True
    action_stage1_only: bool = True
    source_weights: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.25]
    )
    grounding_stage1_only: bool = True


@dataclass
class HierarchicalRecordDecodeConfig:
    entity_threshold: float = 0.0
    strategy: str = "interval"
    stage1_spans_only: bool = True
    enable_visibility_correction: bool = True
    enable_region_override: bool = True
    visible_from_null_threshold: float = 0.8
    null_from_visible_threshold: float = 0.2
    region_override_mode: str = "margin"
    region_override_logit_margin: float = 0.2
    region_override_probability_margin: float = 0.05
    override_damage_cost: float = 3.0
    override_utility_threshold: float = 0.0
    include_override_risk_curve: bool = False
    enable_action_controller: bool = False
    action_top_k: int = 4
    action_execution_margin: float = 0.0
    include_action_risk_curve: bool = False


@dataclass
class HierarchicalRecordRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = "outputs/fmnerg_hierarchical_record_verifier"
    save_best_metric: str = "gmner_score"
    save_best_tie_breakers: list[str] = field(default_factory=list)
    early_stop_patience: int = 3
    log_every_steps: int = 50
    init_checkpoint: str | None = None
    train_override_utility_only: bool = False
    train_action_controller_only: bool = False
    evaluate_test_after_training: bool = True


@dataclass
class HierarchicalRecordTrainingConfig:
    data: HierarchicalRecordDataConfig
    model: HierarchicalRecordVerifierConfig = field(
        default_factory=HierarchicalRecordVerifierConfig
    )
    optim: HierarchicalRecordOptimConfig = field(
        default_factory=HierarchicalRecordOptimConfig
    )
    loss: HierarchicalRecordLossConfig = field(
        default_factory=HierarchicalRecordLossConfig
    )
    decode: HierarchicalRecordDecodeConfig = field(
        default_factory=HierarchicalRecordDecodeConfig
    )
    runtime: HierarchicalRecordRuntimeConfig = field(
        default_factory=HierarchicalRecordRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_hierarchical_record_verifier_config(
    path: str | Path,
) -> HierarchicalRecordTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "data" not in raw:
        raise ValueError("Hierarchical verifier config requires a data section.")
    return HierarchicalRecordTrainingConfig(
        data=HierarchicalRecordDataConfig(**raw["data"]),
        model=HierarchicalRecordVerifierConfig(**raw.get("model", {})),
        optim=HierarchicalRecordOptimConfig(**raw.get("optim", {})),
        loss=HierarchicalRecordLossConfig(**raw.get("loss", {})),
        decode=HierarchicalRecordDecodeConfig(**raw.get("decode", {})),
        runtime=HierarchicalRecordRuntimeConfig(**raw.get("runtime", {})),
    )
