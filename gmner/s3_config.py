"""Configuration contract for the isolated S3.1 Stage1 experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class S3BaseConfig:
    formal_config: str = "configs/fmnerg_twitter10000_stage1.yaml"
    initialization_checkpoint: str = (
        "outputs/fmnerg_stage1_roberta128/best_model.pt"
    )
    baseline_lock: str = (
        "docs/experiments/s3_stage1_baseline_lock.json"
    )


@dataclass
class S3ModelConfig:
    hidden_size: int = 768
    boundary_dropout: float = 0.1
    type_dropout: float = 0.1
    num_boundary_tags: int = 3
    num_types: int = 4


@dataclass
class S3OptimConfig:
    batch_size: int = 4
    learning_rate: float = 2e-5
    new_module_learning_rate: float = 1e-4
    high_level_learning_rate: float = 1e-5
    backbone_learning_rate: float = 3e-6
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1


@dataclass
class S3LossConfig:
    label_smoothing: float = 0.0
    lambda_boundary: float = 1.0
    lambda_type: float = 1.0
    lambda_grounding: float = 1.0
    lambda_alignment: float = 1.0
    scaling_report: str = ""
    require_scaling_report: bool = True


@dataclass
class S3ProbeConfig:
    steps: int = 100
    audit_interval: int = 10
    lambda_min: float = 0.05
    lambda_max: float = 20.0
    epsilon: float = 1e-12


@dataclass
class S3RuntimeConfig:
    seed: int = 42
    device: str = "cuda"
    fp16: bool = True
    output_dir: str = "outputs/s3_stage1/seed42"
    probe_output: str = "outputs/s3_stage1/scaling_probe_seed42.json"
    early_stopping_patience: int = 3
    log_every_steps: int = 20
    num_workers: int = 0


@dataclass
class S3Stage1Config:
    base: S3BaseConfig = field(default_factory=S3BaseConfig)
    model: S3ModelConfig = field(default_factory=S3ModelConfig)
    optim: S3OptimConfig = field(default_factory=S3OptimConfig)
    loss: S3LossConfig = field(default_factory=S3LossConfig)
    probe: S3ProbeConfig = field(default_factory=S3ProbeConfig)
    runtime: S3RuntimeConfig = field(default_factory=S3RuntimeConfig)


def _update(instance: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise ValueError(
                f"Unknown S3 configuration key "
                f"{type(instance).__name__}.{key}."
            )
        current = getattr(instance, key)
        if isinstance(current, bool) and not isinstance(value, bool):
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"true", "1", "yes", "on"}:
                    value = True
                elif normalized in {"false", "0", "no", "off"}:
                    value = False
                else:
                    raise ValueError(f"Invalid boolean value: {value!r}.")
            else:
                value = bool(value)
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, str):
            value = str(value)
        setattr(instance, key, value)


def validate_s3_config(config: S3Stage1Config) -> S3Stage1Config:
    if config.model.num_boundary_tags != 3:
        raise ValueError("S3.1 Boundary CRF must use exactly O/B/I.")
    if config.model.num_types != 4:
        raise ValueError("S3.1 Span Type Head must use four coarse types.")
    if config.optim.batch_size < 1:
        raise ValueError("S3.1 batch_size must be positive.")
    if config.optim.gradient_accumulation_steps < 1:
        raise ValueError(
            "S3.1 gradient_accumulation_steps must be positive."
        )
    if config.probe.steps != 100:
        raise ValueError(
            "The preregistered S3.1 scaling probe is fixed at 100 steps."
        )
    if config.probe.audit_interval < 1:
        raise ValueError("probe.audit_interval must be positive.")
    if not 0 < config.probe.lambda_min <= config.probe.lambda_max:
        raise ValueError("Invalid S3.1 static-scaling clip range.")
    for name in (
        "lambda_boundary",
        "lambda_type",
        "lambda_grounding",
        "lambda_alignment",
    ):
        if float(getattr(config.loss, name)) <= 0:
            raise ValueError(f"loss.{name} must be positive.")
    return config


def load_s3_config(path: str | Path) -> S3Stage1Config:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    allowed_sections = {
        "base",
        "model",
        "optim",
        "loss",
        "probe",
        "runtime",
    }
    unknown = sorted(set(payload) - allowed_sections)
    if unknown:
        raise ValueError(f"Unknown S3 configuration sections: {unknown}.")
    config = S3Stage1Config()
    for section in allowed_sections:
        values = payload.get(section)
        if values:
            if not isinstance(values, dict):
                raise ValueError(f"S3 section {section} must be a mapping.")
            _update(getattr(config, section), values)
    return validate_s3_config(config)


def dump_s3_config(
    config: S3Stage1Config,
    path: str | Path,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            asdict(config),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
