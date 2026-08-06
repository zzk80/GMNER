"""Strict configuration loader for the authorized TP M0/M0.5/M1 scope."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class TPM1Config:
    variant: str
    base_config: str
    base_checkpoint: str
    train_clip_cache: str
    dev_clip_cache: str
    m0_5_report: str
    output_dir: str
    hidden_size: int = 768
    attention_heads: int = 8
    ffn_intermediate_size: int = 1536
    dropout: float = 0.1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 8
    epochs: int = 15
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    fp16: bool = True
    seed: int = 42
    lambda_preserve: float = 1.0
    lambda_residual: float = 0.01
    distillation_temperature: float = 1.0
    eeg_preservation_tolerance: float = 0.001


@dataclass(frozen=True)
class TPJointM1Config(TPM1Config):
    backbone_learning_rate: float = 3e-6
    fusion_learning_rate: float = 1e-5
    unfreeze_last_n_layers: int = 4
    amp_dtype: str = "bfloat16"
    initialization_checkpoint: str | None = None
    grounding_objective: bool = False
    lambda_grounding_supervision: float = 0.0
    lambda_grounding_preserve: float = 0.0
    grounding_temperature: float = 1.0
    train_residual: bool = True


def load_tp_m1_config(path: str | Path) -> TPM1Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if payload.get("stage") != "M1":
        raise ValueError("TP training config must explicitly declare stage: M1.")
    if payload.get("test_accessed") is not False:
        raise ValueError("TP M1 config must lock test_accessed: false.")
    model = payload.get("model") or {}
    data = payload.get("data") or {}
    runtime = payload.get("runtime") or {}
    optim = payload.get("optim") or {}
    loss = payload.get("loss") or {}
    config = TPM1Config(
        variant=str(model["variant"]),
        base_config=str(data["base_config"]),
        base_checkpoint=str(data["base_checkpoint"]),
        train_clip_cache=str(data["train_clip_cache"]),
        dev_clip_cache=str(data["dev_clip_cache"]),
        m0_5_report=str(data["m0_5_report"]),
        output_dir=str(runtime["output_dir"]),
        hidden_size=int(model.get("hidden_size", 768)),
        attention_heads=int(model.get("attention_heads", 8)),
        ffn_intermediate_size=int(model.get("ffn_intermediate_size", 1536)),
        dropout=float(model.get("dropout", 0.1)),
        learning_rate=float(optim.get("learning_rate", 1e-4)),
        weight_decay=float(optim.get("weight_decay", 0.01)),
        batch_size=int(optim.get("batch_size", 8)),
        epochs=int(optim.get("epochs", 15)),
        warmup_ratio=float(optim.get("warmup_ratio", 0.1)),
        gradient_clip_norm=float(optim.get("gradient_clip_norm", 1.0)),
        fp16=bool(optim.get("fp16", True)),
        seed=int(runtime.get("seed", 42)),
        lambda_preserve=float(loss.get("lambda_preserve", 1.0)),
        lambda_residual=float(loss.get("lambda_residual", 0.01)),
        distillation_temperature=float(loss.get("distillation_temperature", 1.0)),
        eeg_preservation_tolerance=float(runtime.get("eeg_preservation_tolerance", 0.001)),
    )
    frozen = {
        "hidden_size": 768,
        "attention_heads": 8,
        "ffn_intermediate_size": 1536,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "batch_size": 8,
        "epochs": 15,
        "warmup_ratio": 0.1,
        "gradient_clip_norm": 1.0,
        "lambda_preserve": 1.0,
        "lambda_residual": 0.01,
        "distillation_temperature": 1.0,
    }
    for name, expected in frozen.items():
        if getattr(config, name) != expected:
            raise ValueError(f"TP M1 preregistered {name}={expected}, found {getattr(config, name)}.")
    if config.variant not in {"a_text", "a1_global", "a2_r16"}:
        raise ValueError(f"Unsupported TP M1 variant: {config.variant}.")
    return config


def load_tp_joint_m1_config(path: str | Path) -> TPJointM1Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    stage = payload.get("stage")
    if stage not in {
        "M1_JOINT_EXPLORATORY",
        "M1_GROUNDING_PROTECTED",
        "M1_GROUNDING_RESIDUAL_FROZEN",
    }:
        raise ValueError(
            "Joint TP config must declare stage M1_JOINT_EXPLORATORY or "
            "M1_GROUNDING_PROTECTED, or M1_GROUNDING_RESIDUAL_FROZEN."
        )
    if payload.get("test_accessed") is not False:
        raise ValueError("Joint TP training must lock test_accessed: false.")
    model = payload.get("model") or {}
    data = payload.get("data") or {}
    runtime = payload.get("runtime") or {}
    optim = payload.get("optim") or {}
    loss = payload.get("loss") or {}
    config = TPJointM1Config(
        variant=str(model["variant"]),
        base_config=str(data["base_config"]),
        base_checkpoint=str(data["base_checkpoint"]),
        train_clip_cache=str(data["train_clip_cache"]),
        dev_clip_cache=str(data["dev_clip_cache"]),
        m0_5_report=str(data["m0_5_report"]),
        output_dir=str(runtime["output_dir"]),
        hidden_size=int(model.get("hidden_size", 768)),
        attention_heads=int(model.get("attention_heads", 8)),
        ffn_intermediate_size=int(model.get("ffn_intermediate_size", 1536)),
        dropout=float(model.get("dropout", 0.1)),
        learning_rate=float(optim.get("residual_learning_rate", 1e-4)),
        backbone_learning_rate=float(optim.get("backbone_learning_rate", 3e-6)),
        fusion_learning_rate=float(optim.get("fusion_learning_rate", 1e-5)),
        unfreeze_last_n_layers=int(optim.get("unfreeze_last_n_layers", 4)),
        amp_dtype=str(optim.get("amp_dtype", "bfloat16")),
        weight_decay=float(optim.get("weight_decay", 0.01)),
        batch_size=int(optim.get("batch_size", 8)),
        epochs=int(optim.get("epochs", 15)),
        warmup_ratio=float(optim.get("warmup_ratio", 0.1)),
        gradient_clip_norm=float(optim.get("gradient_clip_norm", 1.0)),
        fp16=bool(optim.get("fp16", True)),
        seed=int(runtime.get("seed", 42)),
        lambda_preserve=float(loss.get("lambda_preserve", 1.0)),
        lambda_residual=float(loss.get("lambda_residual", 0.01)),
        distillation_temperature=float(loss.get("distillation_temperature", 1.0)),
        eeg_preservation_tolerance=float(runtime.get("eeg_preservation_tolerance", 0.001)),
        initialization_checkpoint=(
            str(data["initialization_checkpoint"])
            if data.get("initialization_checkpoint")
            else None
        ),
        grounding_objective=bool(loss.get("grounding_objective", False)),
        lambda_grounding_supervision=float(
            loss.get("lambda_grounding_supervision", 0.0)
        ),
        lambda_grounding_preserve=float(
            loss.get("lambda_grounding_preserve", 0.0)
        ),
        grounding_temperature=float(loss.get("grounding_temperature", 1.0)),
        train_residual=bool(optim.get("train_residual", True)),
    )
    if config.variant not in {"a_text", "a2_r16"}:
        raise ValueError(
            "Joint TP exploratory matrix is restricted to a_text and a2_r16."
        )
    if config.unfreeze_last_n_layers not in {4, 12}:
        raise ValueError(
            "Joint TP unfreeze_last_n_layers must be 4 (J1) or 12 (J2)."
        )
    expected = {
        "hidden_size": 768,
        "attention_heads": 8,
        "ffn_intermediate_size": 1536,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "backbone_learning_rate": 3e-6,
        "fusion_learning_rate": 1e-5,
        "amp_dtype": "bfloat16",
        "weight_decay": 0.01,
        "batch_size": 8,
        "warmup_ratio": 0.1,
        "gradient_clip_norm": 1.0,
        "lambda_preserve": 1.0,
        "lambda_residual": 0.01,
        "distillation_temperature": 1.0,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"Joint TP preregistered {name}={value}, found {getattr(config, name)}."
            )
    if stage == "M1_GROUNDING_PROTECTED":
        if config.variant != "a_text" or config.unfreeze_last_n_layers != 4:
            raise ValueError(
                "Grounding-protected J3 is restricted to A-text with the J1 "
                "four-layer trainable scope."
            )
        if not config.initialization_checkpoint:
            raise ValueError("Grounding-protected J3 requires a J1 initialization checkpoint.")
        if not config.grounding_objective:
            raise ValueError("Grounding-protected J3 must enable grounding_objective.")
        if config.epochs != 15 or not config.train_residual:
            raise ValueError("Original J3 requires 15 epochs and a trainable residual.")
        if (
            config.lambda_grounding_supervision != 1.0
            or config.lambda_grounding_preserve != 1.0
            or config.grounding_temperature != 8.0
        ):
            raise ValueError(
                "Grounding-protected J3 freezes supervision/preservation weights at 1.0 "
                "and grounding temperature at 8.0."
            )
    elif stage == "M1_GROUNDING_RESIDUAL_FROZEN":
        if config.variant != "a_text" or config.unfreeze_last_n_layers != 4:
            raise ValueError(
                "Grounding-protected J3-r1 is restricted to A-text with the J1 "
                "four-layer trainable scope."
            )
        if not config.initialization_checkpoint or not config.grounding_objective:
            raise ValueError(
                "J3-r1 requires J1 initialization and the grounding objective."
            )
        if config.epochs != 5 or config.train_residual:
            raise ValueError("J3-r1 requires five epochs and a frozen residual.")
        if (
            config.lambda_grounding_supervision != 1.0
            or config.lambda_grounding_preserve != 1.0
            or config.grounding_temperature != 8.0
        ):
            raise ValueError("J3-r1 must retain the registered J3 grounding losses.")
    elif config.initialization_checkpoint or config.grounding_objective:
        raise ValueError(
            "J1/J2 configs cannot silently enable the J3 initialization or grounding objective."
        )
    elif config.epochs != 15 or not config.train_residual:
        raise ValueError("J1/J2 require 15 epochs and a trainable residual.")
    return config


def load_tp_training_config(path: str | Path) -> TPM1Config | TPJointM1Config:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if payload.get("stage") in {
        "M1_JOINT_EXPLORATORY",
        "M1_GROUNDING_PROTECTED",
        "M1_GROUNDING_RESIDUAL_FROZEN",
    }:
        return load_tp_joint_m1_config(path)
    return load_tp_m1_config(path)
