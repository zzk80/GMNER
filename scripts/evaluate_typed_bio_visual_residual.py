#!/usr/bin/env python3
"""Evaluate one frozen TP M1 checkpoint on Dev, paired or preregistered shuffled."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from gmner.config import load_config
from gmner.data.artifact_utils import sha256_file
from gmner.data.clip_r16_cache import ClipR16Cache
from gmner.engine.tp_visual_residual_evaluator import (
    deranged_image_id_map,
    evaluate_tp_visual_stage1,
)
from gmner.models.typed_bio_visual_residual import (
    ProtectedTypedBIOVisualStage1,
    TypedBIOVisualResidual,
    TypedBIOVisualResidualConfig,
    restore_joint_student_state,
)
from gmner.tp.config import TPJointM1Config, load_tp_training_config
from gmner.tp.grounding_replay import GroundabilityPriorLookup
from gmner.tp.runtime import build_tp_runtime, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shuffle-seed", type=int, choices=(101, 102, 103, 104, 105))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment_path = resolve_path(args.config, root)
    experiment = load_tp_training_config(experiment_path)
    checkpoint_path = resolve_path(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("kind") != "tp_typed_bio_visual_residual":
        raise ValueError("Not a TP M1 residual checkpoint.")
    if checkpoint.get("test_accessed") is not False:
        raise ValueError("Checkpoint does not prove test_accessed=false.")
    if checkpoint.get("experiment_config_sha256") != sha256_file(experiment_path):
        raise ValueError("TP experiment config fingerprint mismatch.")
    base_config_path = resolve_path(experiment.base_config, root)
    base_checkpoint_path = resolve_path(experiment.base_checkpoint, root)
    dev_cache_path = resolve_path(experiment.dev_clip_cache, root)
    if checkpoint.get("base_checkpoint_sha256") != sha256_file(base_checkpoint_path):
        raise ValueError("Frozen Stage1 checkpoint fingerprint mismatch.")
    if checkpoint.get("dev_clip_manifest_sha256") != sha256_file(dev_cache_path / "manifest.json"):
        raise ValueError("Dev CLIP cache fingerprint mismatch.")
    base_config = load_config(base_config_path)
    base_config.data.expand_entities_for_grounding = False
    runtime = build_tp_runtime(
        config=base_config,
        checkpoint_path=base_checkpoint_path,
        project_root=root,
        cache_dir=resolve_path(experiment.output_dir, root) / "eval_dataset_cache",
        batch_size=experiment.batch_size,
        include_train=False,
    )
    joint_training = isinstance(experiment, TPJointM1Config)
    checkpoint_mode = checkpoint.get("training_mode", "protected")
    if joint_training != (checkpoint_mode == "joint"):
        raise ValueError("TP config/checkpoint training mode mismatch.")
    if joint_training:
        restore_joint_student_state(
            runtime["model"], checkpoint.get("student_trainable_state_dict") or {}
        )
    cache = ClipR16Cache(dev_cache_path, expected_split="dev")
    cache.preload_all()
    residual_config = TypedBIOVisualResidualConfig(**checkpoint["residual_config"])
    residual = TypedBIOVisualResidual(residual_config)
    residual.load_state_dict(checkpoint["residual_state_dict"])
    device = torch.device(args.device)
    model = ProtectedTypedBIOVisualStage1(runtime["model"], residual).to(device)
    model.offload_unused_image_encoder()
    prior_lookup = GroundabilityPriorLookup(
        resolve_path(base_config.data.groundability_type_priors, root),
        resolve_path(base_config.data.groundability_mention_priors, root),
    )
    image_id_map = None
    mode = "paired"
    if args.shuffle_seed is not None:
        mode = f"shuffled_seed_{args.shuffle_seed}"
        image_ids = [str(sample["image_id"]) for sample in runtime["datasets"]["dev"].samples]
        image_id_map = deranged_image_id_map(image_ids, args.shuffle_seed)
        if any(source == target for source, target in image_id_map.items()):
            raise RuntimeError("Shuffled diagnostic contains a fixed point.")
    metrics = evaluate_tp_visual_stage1(
        model=model,
        dataloader=runtime["loaders"]["dev"],
        clip_cache=cache,
        device=device,
        prior_lookup=prior_lookup,
        image_id_map=image_id_map,
    )
    metrics.pop("prediction_records", None)
    metrics.pop("record_metrics", None)
    result = {
        "kind": "tp_m1_dev_evaluation",
        "variant": experiment.variant,
        "training_mode": checkpoint_mode,
        "mode": mode,
        "shuffle_seed": args.shuffle_seed,
        "metrics": metrics,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "test_accessed": False,
    }
    output = resolve_path(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
