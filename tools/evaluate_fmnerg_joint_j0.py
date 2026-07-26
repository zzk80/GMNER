"""Evaluate one J0 checkpoint on frozen M3.3A Dev predictions only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_joint.config import load_joint_subtype_config
from sidecars.fmnerg_joint.data import (
    JointOnlineSubtypeCollator,
    load_joint_subtype_data,
)
from sidecars.fmnerg_joint.evaluator import (
    evaluate_joint_formal_predictions,
    evaluate_joint_gold_spans,
)
from sidecars.fmnerg_joint.model import (
    build_j0_visual_subtype_model,
    load_j0_checkpoint_state,
)
from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, root)
    config = load_joint_subtype_config(config_path)
    seed = int(args.seed if args.seed is not None else config.runtime.seed)
    requested_device = args.device or config.runtime.device
    device = torch.device(
        requested_device
        if str(requested_device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    taxonomy = SubtypeTaxonomy.from_file(resolve_path(config.taxonomy, root))
    (
        _,
        dev_gold_dataset,
        dev_formal_dataset,
        formal_payload,
        data_artifacts,
    ) = load_joint_subtype_data(
        config=config,
        taxonomy=taxonomy,
        root=root,
    )
    model, tokenizer, _, initialization = build_j0_visual_subtype_model(
        config=config,
        taxonomy=taxonomy,
        root=root,
        device=device,
        seed=seed,
    )
    checkpoint_path = resolve_path(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("kind") != "fmnerg_joint_j0_visual_fusion":
        raise ValueError("Not a J0 visual-subtype checkpoint.")
    if checkpoint.get("test_accessed") is not False:
        raise ValueError("J0 checkpoint accessed Test data.")
    if checkpoint.get("taxonomy_sha256") != taxonomy.source_sha256:
        raise ValueError("J0 checkpoint taxonomy fingerprint changed.")
    if checkpoint.get("config_sha256") != sha256_file(config_path):
        raise ValueError("J0 checkpoint config fingerprint changed.")
    stored_initialization = dict(checkpoint.get("initialization") or {})
    if (
        stored_initialization.get("subtype_checkpoint_sha256")
        != initialization.get("subtype_checkpoint_sha256")
    ):
        raise ValueError("J0 paired F2 checkpoint changed.")
    if dict(checkpoint.get("data_artifacts") or {}) != data_artifacts:
        raise ValueError("J0 frozen data artifacts changed.")
    load_j0_checkpoint_state(model, checkpoint["model"])
    model.to(device).eval()
    max_length = int(
        initialization["subtype_encoder_initialization"]["max_length"]
    )
    collator = JointOnlineSubtypeCollator(
        tokenizer,
        max_length=max_length,
        region_feature_size=config.model.region_feature_size,
        geometry_size=config.model.geometry_size,
    )
    gold = evaluate_joint_gold_spans(
        model,
        dev_gold_dataset,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
        include_detailed=True,
    )
    formal = evaluate_joint_formal_predictions(
        model,
        dev_formal_dataset,
        formal_payload,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
    )
    result = {
        "metadata": {
            **formal["metadata"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "paired_f2_seed": seed,
            "formal_stage1_mutated": False,
            "formal_region_mutated": False,
            "test_accessed": False,
        },
        "metrics": {**gold, **formal["metrics"]},
    }
    output = resolve_path(
        args.output
        or checkpoint_path.parent / "dev_metrics_recomputed.json",
        root,
    )
    save_json_atomic(result, output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
