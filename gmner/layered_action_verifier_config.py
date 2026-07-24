"""Configuration schema for the M3.6A layered action verifier."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gmner.models.layered_action_verifier import (
    ACTION_MODES,
    LayeredActionVerifierConfig,
)


@dataclass
class LayeredActionFrozenConfig:
    reliability_config: str
    reliability_checkpoint: str


@dataclass
class LayeredActionOOFConfig:
    require_full_chain_oof: bool = False
    train_feature_cache: str | None = None
    expected_num_folds: int = 10
    expected_records: int | None = None


@dataclass
class LayeredActionLossConfig:
    lambda_layer1: float = 1.0
    lambda_layer2: float = 1.0
    lambda_keep_margin: float = 0.5
    lambda_correction_margin: float = 0.5
    lambda_preservation: float = 0.2
    lambda_residual: float = 0.02
    keep_margin: float = 0.5
    correction_margin: float = 0.5
    keep_group_weight: float = 0.20
    to_null_group_weight: float = 0.40
    to_visible_group_weight: float = 0.40
    false_release_weight: float = 3.0
    missed_release_weight: float = 1.0
    stage1_spans_only: bool = True
    require_correct_type: bool = True


@dataclass
class LayeredActionEvaluationConfig:
    execution_margin: float = 0.0
    include_risk_curve: bool = True
    identity_tolerance: float = 1e-12
    expected_baseline_gmner: float | None = 0.621316
    expected_baseline_tolerance: float = 5e-6
    minimum_keep_preservation_rate: float = 0.97
    minimum_net_correction: int = 10


@dataclass
class LayeredActionOptimConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class LayeredActionRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = "outputs/fmnerg_roberta128_layered_action_verifier"
    save_best_metric: str = "gmner_score"
    save_best_tie_breakers: list[str] = field(
        default_factory=lambda: [
            "to_visible_net_correction",
            "to_null_net_correction",
            "keep_correct_preservation_rate",
        ]
    )
    early_stop_patience: int = 3
    log_every_steps: int = 50


@dataclass
class LayeredActionTrainingConfig:
    frozen: LayeredActionFrozenConfig
    oof: LayeredActionOOFConfig = field(default_factory=LayeredActionOOFConfig)
    model: LayeredActionVerifierConfig = field(
        default_factory=LayeredActionVerifierConfig
    )
    loss: LayeredActionLossConfig = field(default_factory=LayeredActionLossConfig)
    evaluation: LayeredActionEvaluationConfig = field(
        default_factory=LayeredActionEvaluationConfig
    )
    optim: LayeredActionOptimConfig = field(default_factory=LayeredActionOptimConfig)
    runtime: LayeredActionRuntimeConfig = field(
        default_factory=LayeredActionRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_layered_action_verifier_config(
    path: str | Path,
) -> LayeredActionTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "frozen" not in raw:
        raise ValueError("M3.6A config requires a frozen section.")
    config = LayeredActionTrainingConfig(
        frozen=LayeredActionFrozenConfig(**raw["frozen"]),
        oof=LayeredActionOOFConfig(**raw.get("oof", {})),
        model=LayeredActionVerifierConfig(**raw.get("model", {})),
        loss=LayeredActionLossConfig(**raw.get("loss", {})),
        evaluation=LayeredActionEvaluationConfig(**raw.get("evaluation", {})),
        optim=LayeredActionOptimConfig(**raw.get("optim", {})),
        runtime=LayeredActionRuntimeConfig(**raw.get("runtime", {})),
    )
    if int(config.model.top_k) != 4:
        raise ValueError("M3.6A fixes model.top_k to Fine Top-4.")
    if config.model.action_mode not in ACTION_MODES:
        raise ValueError(
            f"model.action_mode must be one of {ACTION_MODES}, got "
            f"{config.model.action_mode!r}."
        )
    if config.model.keep_initial_bias <= config.model.action_initial_bias:
        raise ValueError("KEEP initial bias must exceed both action initial biases.")
    group_weights = (
        config.loss.keep_group_weight,
        config.loss.to_null_group_weight,
        config.loss.to_visible_group_weight,
    )
    if any(value < 0.0 for value in group_weights) or sum(group_weights) <= 0:
        raise ValueError("Layer-1 group weights must be nonnegative.")
    if (
        config.model.action_mode == "to_real_only"
        and config.loss.to_null_group_weight != 0.0
    ):
        raise ValueError("to_real_only requires loss.to_null_group_weight=0.")
    if (
        config.model.action_mode == "to_null_only"
        and config.loss.to_visible_group_weight != 0.0
    ):
        raise ValueError("to_null_only requires loss.to_visible_group_weight=0.")
    if config.model.action_mode == "null_release_only":
        if config.loss.to_null_group_weight != 0.0:
            raise ValueError(
                "null_release_only requires loss.to_null_group_weight=0."
            )
        if config.loss.false_release_weight <= config.loss.missed_release_weight:
            raise ValueError(
                "null_release_only requires false_release_weight greater than "
                "missed_release_weight."
            )
        if not config.oof.require_full_chain_oof:
            raise ValueError(
                "null_release_only requires oof.require_full_chain_oof=true. "
                "Use the older to_real_only config for in-sample engineering "
                "reproduction."
            )
        if not config.oof.train_feature_cache:
            raise ValueError(
                "null_release_only requires oof.train_feature_cache."
            )
        if config.oof.expected_num_folds != 10:
            raise ValueError("Formal NULL Release training requires exactly 10 folds.")
    if config.loss.keep_margin < 0.0 or config.loss.correction_margin < 0.0:
        raise ValueError("Action margins must be nonnegative.")
    if config.evaluation.execution_margin < 0.0:
        raise ValueError("evaluation.execution_margin must be nonnegative.")
    if not 0.0 <= config.evaluation.minimum_keep_preservation_rate <= 1.0:
        raise ValueError("minimum_keep_preservation_rate must lie in [0, 1].")
    return config
