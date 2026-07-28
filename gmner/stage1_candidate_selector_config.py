"""Configuration for the strict-OOF D1 Stage1 candidate selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gmner.models.stage1_candidate_selector import (
    Stage1CandidateSelectorConfig,
)


@dataclass
class Stage1SelectorDataConfig:
    train_cache: str
    dev_cache: str
    phase1_audit: str
    num_workers: int = 0


@dataclass
class Stage1SelectorOptimConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 12
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class Stage1SelectorLossConfig:
    lambda_entity: float = 1.0
    lambda_overlap_margin: float = 0.5
    lambda_residual: float = 0.05
    overlap_margin: float = 0.2


@dataclass
class Stage1SelectorDecodeConfig:
    threshold: float = 0.0
    strategy: str = "weighted_interval"


@dataclass
class Stage1SelectorRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = False
    output_dir: str = "outputs/stage1_candidate_selector_seed42"
    save_best_metric: str = "span_f1"
    save_best_tie_breakers: list[str] = field(
        default_factory=lambda: ["mner_f1", "gmner_score"]
    )
    early_stop_patience: int = 3
    log_every_steps: int = 25


@dataclass
class Stage1SelectorTrainingConfig:
    data: Stage1SelectorDataConfig
    model: Stage1CandidateSelectorConfig = field(
        default_factory=Stage1CandidateSelectorConfig
    )
    optim: Stage1SelectorOptimConfig = field(
        default_factory=Stage1SelectorOptimConfig
    )
    loss: Stage1SelectorLossConfig = field(
        default_factory=Stage1SelectorLossConfig
    )
    decode: Stage1SelectorDecodeConfig = field(
        default_factory=Stage1SelectorDecodeConfig
    )
    runtime: Stage1SelectorRuntimeConfig = field(
        default_factory=Stage1SelectorRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_stage1_candidate_selector_config(
    path: str | Path,
) -> Stage1SelectorTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if "data" not in raw:
        raise ValueError("Stage1 selector config requires a data section.")
    config = Stage1SelectorTrainingConfig(
        data=Stage1SelectorDataConfig(**raw["data"]),
        model=Stage1CandidateSelectorConfig(**raw.get("model", {})),
        optim=Stage1SelectorOptimConfig(**raw.get("optim", {})),
        loss=Stage1SelectorLossConfig(**raw.get("loss", {})),
        decode=Stage1SelectorDecodeConfig(**raw.get("decode", {})),
        runtime=Stage1SelectorRuntimeConfig(**raw.get("runtime", {})),
    )
    if config.decode.strategy != "weighted_interval":
        raise ValueError("D1 uses only the preregistered weighted_interval decode.")
    if float(config.decode.threshold) != 0.0:
        raise ValueError("D1 decode threshold is frozen at 0.0.")
    if float(config.model.formal_prior) != 0.5:
        raise ValueError("D1 formal source prior is frozen at +0.5.")
    if float(config.model.nonformal_prior) != -0.5:
        raise ValueError("D1 non-formal source prior is frozen at -0.5.")
    if float(config.model.residual_scale) != 1.0:
        raise ValueError("D1 residual scale is frozen at 1.0.")
    return config
