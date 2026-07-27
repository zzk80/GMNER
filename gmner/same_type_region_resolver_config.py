"""Configuration for the M3.3A-P3 conditional same-type resolver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from gmner.constants import ENTITY_TYPE2ID
from gmner.models.same_type_region_resolver import (
    COMPETITION_SCALAR_COUNT,
    SameTypeRegionResolverConfig,
)


@dataclass
class SameTypeResolverDataConfig:
    formal_train_cache: str
    expanded_train_cache: str
    formal_dev_cache: str
    expanded_dev_cache: str
    num_workers: int = 0
    require_oof_train_cache: bool = False


@dataclass
class SameTypeResolverFrozenConfig:
    evidence_config: str
    evidence_checkpoint: str


@dataclass
class SameTypeResolverLossConfig:
    lambda_correction: float = 1.0
    lambda_preserve_kl: float = 1.0
    lambda_preserve_margin: float = 0.5
    lambda_residual: float = 0.05
    preserve_margin: float = 0.2
    kl_temperature: float = 1.0


@dataclass
class SameTypeResolverCandidateConfig:
    use_full_fine_candidate_mask: bool = True
    top_k_union: int = 0
    include_null: bool = False


@dataclass
class SameTypeResolverOptimConfig:
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class SameTypeResolverRuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = (
        "outputs/fmnerg_roberta128_same_type_region_resolver_c1"
    )
    save_best_metric: str = "gmner_score"
    save_best_tie_breakers: list[str] = field(
        default_factory=lambda: [
            "base_correct_trigger_preservation_rate",
            "gmner_net_correction",
        ]
    )
    early_stop_patience: int = 3
    log_every_steps: int = 50
    expected_dev_baseline_gmner: float = 0.6213161081953977
    baseline_tolerance: float = 5e-6


@dataclass
class SameTypeResolverTrainingConfig:
    data: SameTypeResolverDataConfig
    frozen: SameTypeResolverFrozenConfig
    model: SameTypeRegionResolverConfig = field(
        default_factory=SameTypeRegionResolverConfig
    )
    loss: SameTypeResolverLossConfig = field(
        default_factory=SameTypeResolverLossConfig
    )
    candidate: SameTypeResolverCandidateConfig = field(
        default_factory=SameTypeResolverCandidateConfig
    )
    optim: SameTypeResolverOptimConfig = field(
        default_factory=SameTypeResolverOptimConfig
    )
    runtime: SameTypeResolverRuntimeConfig = field(
        default_factory=SameTypeResolverRuntimeConfig
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_same_type_region_resolver_config(
    path: str | Path,
) -> SameTypeResolverTrainingConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    for section in ("data", "frozen"):
        if section not in raw:
            raise ValueError(
                f"Same-type resolver config requires a {section} section."
            )
    config = SameTypeResolverTrainingConfig(
        data=SameTypeResolverDataConfig(**raw["data"]),
        frozen=SameTypeResolverFrozenConfig(**raw["frozen"]),
        model=SameTypeRegionResolverConfig(**raw.get("model", {})),
        loss=SameTypeResolverLossConfig(**raw.get("loss", {})),
        candidate=SameTypeResolverCandidateConfig(
            **raw.get("candidate", {})
        ),
        optim=SameTypeResolverOptimConfig(**raw.get("optim", {})),
        runtime=SameTypeResolverRuntimeConfig(**raw.get("runtime", {})),
    )
    if int(config.model.hidden_size) != 256:
        raise ValueError("C1 must reuse the 256-dimensional Fine states.")
    if int(config.model.scalar_count) != COMPETITION_SCALAR_COUNT:
        raise ValueError(
            f"C1 requires {COMPETITION_SCALAR_COUNT} scalar features."
        )
    if int(config.model.per_type_id) != ENTITY_TYPE2ID["PER"]:
        raise ValueError("model.per_type_id must use constants PER=1.")
    if int(config.model.min_visible_same_type_count) != 2:
        raise ValueError("C1 requires at least two visible PER entities.")
    if float(config.model.residual_scale) != 1.0:
        raise ValueError("C1 fixes residual_scale/alpha to 1.0.")
    if float(config.model.override_margin) not in {0.0, 0.2}:
        raise ValueError("Only preregistered C1=0.0 or C2=0.2 is valid.")
    if not config.candidate.use_full_fine_candidate_mask:
        raise ValueError("C1 must use each entity's full Fine mask.")
    if int(config.candidate.top_k_union) != 0:
        raise ValueError("C1 forbids Top-K candidate union.")
    if bool(config.candidate.include_null):
        raise ValueError("C1 candidate actions exclude NULL.")
    if float(config.loss.kl_temperature) <= 0:
        raise ValueError("loss.kl_temperature must be positive.")
    if float(config.runtime.baseline_tolerance) < 0:
        raise ValueError("runtime.baseline_tolerance must be nonnegative.")
    return config
