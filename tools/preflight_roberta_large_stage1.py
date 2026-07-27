"""Validate the controlled RoBERTa-large Stage1 Dev experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from gmner.config import GMNERConfig, load_config


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="docs/experiments/roberta_large_stage1_phase1_protocol.yaml",
    )
    parser.add_argument(
        "--output",
        default="outputs/fmnerg_stage1_roberta_large_seed42/preflight.json",
    )
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    source = Path(path)
    return source if source.is_absolute() else ROOT / source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def changed_fields(left: Any, right: Any) -> set[str]:
    left_values = asdict(left)
    right_values = asdict(right)
    return {
        key
        for key in left_values
        if left_values[key] != right_values[key]
    }


def validate_controlled_configs(
    baseline: GMNERConfig,
    candidate: GMNERConfig,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    expected = protocol["controlled_changes"]
    if asdict(baseline.data) != asdict(candidate.data):
        raise ValueError("Data configuration changed outside the experiment.")
    if asdict(baseline.loss) != asdict(candidate.loss):
        raise ValueError("Loss configuration changed outside the experiment.")

    actual_changes = {
        "model": changed_fields(baseline.model, candidate.model),
        "optim": changed_fields(baseline.optim, candidate.optim),
        "runtime": changed_fields(baseline.runtime, candidate.runtime),
    }
    for section in ("model", "optim", "runtime"):
        expected_fields = set(expected[section])
        if actual_changes[section] != expected_fields:
            raise ValueError(
                f"Unexpected {section} changes: "
                f"expected={sorted(expected_fields)} "
                f"actual={sorted(actual_changes[section])}"
            )

    baseline_effective_batch = (
        baseline.optim.batch_size
        * baseline.optim.gradient_accumulation_steps
    )
    candidate_effective_batch = (
        candidate.optim.batch_size
        * candidate.optim.gradient_accumulation_steps
    )
    if baseline_effective_batch != candidate_effective_batch:
        raise ValueError("Effective batch size differs from RoBERTa-base.")
    if candidate.model.hidden_size != int(expected["shared_hidden_size"]):
        raise ValueError("Shared graph/grounding hidden size must remain fixed.")
    if candidate.runtime.save_best_metric != "gmner_score":
        raise ValueError("Stage1 checkpoint selection must use gmner_score.")
    if candidate.runtime.seed != int(protocol["scope"]["seed"]):
        raise ValueError("Candidate seed differs from the protocol.")
    if not candidate.runtime.fp16:
        raise ValueError("RoBERTa-large Phase 1 requires FP16.")
    if candidate.runtime.init_checkpoint:
        raise ValueError("RoBERTa-large must start from its pretrained backbone.")
    if candidate.optim.num_epochs != baseline.optim.num_epochs:
        raise ValueError("Epoch budget must match RoBERTa-base.")
    if candidate.runtime.early_stopping_patience != (
        baseline.runtime.early_stopping_patience
    ):
        raise ValueError("Early-stopping behavior must match RoBERTa-base.")

    return {
        "actual_changes": {
            key: sorted(value)
            for key, value in actual_changes.items()
        },
        "baseline_effective_batch_size": baseline_effective_batch,
        "candidate_effective_batch_size": candidate_effective_batch,
    }


def verify_baseline_checkpoint(
    checkpoint_path: Path,
    expected_metrics: dict[str, float],
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    metrics = dict(checkpoint.get("metrics") or {})
    for key in ("token_f1", "entity_f1", "eeg_f1", "gmner_score"):
        expected = float(expected_metrics[key])
        actual = float(metrics[key])
        if abs(actual - expected) > 1e-10:
            raise ValueError(
                f"Baseline checkpoint metric drift for {key}: "
                f"expected={expected} actual={actual}"
            )
    return {
        "epoch": int(checkpoint["epoch"]),
        "metrics": {
            key: float(metrics[key])
            for key in (
                "token_f1",
                "entity_f1",
                "grounding_accuracy",
                "eeg_f1",
                "gmner_score",
            )
        },
    }


def inspect_local_roberta_large(model_path: Path) -> dict[str, Any]:
    from transformers import AutoConfig, AutoTokenizer

    backbone = AutoConfig.from_pretrained(
        model_path,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        local_files_only=True,
    )
    if backbone.model_type != "roberta":
        raise ValueError(f"Expected RoBERTa, found {backbone.model_type}.")
    if int(backbone.hidden_size) != 1024:
        raise ValueError("RoBERTa-large must expose hidden_size=1024.")
    if int(backbone.num_hidden_layers) != 24:
        raise ValueError("RoBERTa-large must expose 24 transformer layers.")
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("RoBERTa-large requires a fast tokenizer.")

    weight_files = sorted(
        path
        for pattern in ("*.safetensors", "pytorch_model*.bin")
        for path in model_path.glob(pattern)
    )
    if not weight_files:
        raise FileNotFoundError(
            f"No model weights found under {model_path}."
        )
    return {
        "model_type": backbone.model_type,
        "hidden_size": int(backbone.hidden_size),
        "num_hidden_layers": int(backbone.num_hidden_layers),
        "max_position_embeddings": int(backbone.max_position_embeddings),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "weight_files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in weight_files
        ],
    }


def main() -> None:
    args = parse_args()
    protocol_path = resolve(args.protocol)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("kind") != "roberta_large_stage1_phase1_protocol":
        raise ValueError("Unexpected protocol kind.")
    if protocol["scope"].get("test_accessed") is not False:
        raise ValueError("Phase 1 protocol must prohibit Test access.")
    if protocol["scope"].get("phase2_enabled") is not False:
        raise ValueError("Phase 2 must remain disabled before the gate.")

    baseline_config_path = resolve(protocol["baseline"]["config"])
    candidate_config_path = resolve(protocol["candidate"]["config"])
    baseline = load_config(baseline_config_path)
    candidate = load_config(candidate_config_path)
    controlled = validate_controlled_configs(
        baseline,
        candidate,
        protocol,
    )

    required_paths = {
        "train_file": resolve(candidate.data.train_file),
        "dev_file": resolve(candidate.data.dev_file),
        "image_dir": resolve(candidate.data.image_dir),
        "image_feature_dir": resolve(candidate.data.image_feature_dir),
        "image_annotation_dir": resolve(candidate.data.image_annotation_dir),
        "groundability_type_priors": resolve(
            candidate.data.groundability_type_priors
        ),
        "groundability_mention_priors": resolve(
            candidate.data.groundability_mention_priors
        ),
        "baseline_checkpoint": resolve(
            protocol["baseline"]["checkpoint"]
        ),
        "candidate_model": resolve(protocol["candidate"]["model_path"]),
    }
    for name, path in required_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    baseline_checkpoint = verify_baseline_checkpoint(
        required_paths["baseline_checkpoint"],
        protocol["baseline"]["metrics"],
    )
    backbone = inspect_local_roberta_large(
        required_paths["candidate_model"]
    )
    result = {
        "metadata": {
            "kind": "roberta_large_stage1_phase1_preflight",
            "format_version": 1,
            "split": "train+dev",
            "test_accessed": False,
            "test_path_resolved": False,
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": sha256_file(protocol_path),
            "baseline_config_sha256": sha256_file(
                baseline_config_path
            ),
            "candidate_config_sha256": sha256_file(
                candidate_config_path
            ),
            "baseline_checkpoint_sha256": sha256_file(
                required_paths["baseline_checkpoint"]
            ),
        },
        "controlled_experiment": controlled,
        "baseline_checkpoint": baseline_checkpoint,
        "candidate_backbone": backbone,
        "checked_paths": {
            key: str(path.resolve())
            for key, path in required_paths.items()
        },
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
