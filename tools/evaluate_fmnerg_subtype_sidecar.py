"""Evaluate GMNER and FMNERG together without changing the frozen GMNER chain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.config import load_sidecar_config
from sidecars.fmnerg_subtype.data import SubtypeFeatureDataset
from sidecars.fmnerg_subtype.evaluator import (
    evaluate_formal_predictions,
    evaluate_gold_spans,
    load_formal_predictions,
    save_json_atomic,
    validate_feature_contract,
    validate_expected_frozen_gmner,
)
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.model import HierarchicalSubtypeSidecar
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--include-records", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_sidecar_config(args.config)
    taxonomy = SubtypeTaxonomy.from_file(resolve_path(config.taxonomy, root))
    checkpoint_path = resolve_path(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("kind") != "fmnerg_hierarchical_subtype_sidecar":
        raise ValueError("Not an FMNERG hierarchical subtype checkpoint.")
    if int(checkpoint.get("format_version", -1)) != 1:
        raise ValueError("Unsupported FMNERG subtype checkpoint format.")
    if checkpoint.get("test_accessed") is not False:
        raise ValueError("Subtype checkpoint accessed test data.")
    if checkpoint.get("taxonomy_sha256") != taxonomy.source_sha256:
        raise ValueError("Subtype checkpoint taxonomy fingerprint changed.")
    checkpoint_model = dict(
        dict(checkpoint.get("config") or {}).get("model") or {}
    )
    checkpoint_input_size = int(
        checkpoint_model.get("input_size", config.model.input_size)
    )
    checkpoint_hidden_size = int(
        checkpoint_model.get("hidden_size", config.model.hidden_size)
    )
    checkpoint_dropout = float(
        checkpoint_model.get("dropout", config.model.dropout)
    )
    checkpoint_head_architecture = str(
        checkpoint_model.get("head_architecture", "shared_hard")
    )
    checkpoint_parent_hidden_size = checkpoint_model.get(
        "parent_hidden_size",
        config.model.parent_hidden_size,
    )
    if checkpoint_input_size != config.model.input_size:
        raise ValueError(
            "Subtype checkpoint input size differs from the feature config."
        )

    requested_device = args.device or config.runtime.device
    device = torch.device(
        requested_device
        if str(requested_device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    model = HierarchicalSubtypeSidecar(
        input_size=checkpoint_input_size,
        hidden_size=checkpoint_hidden_size,
        dropout=checkpoint_dropout,
        taxonomy=taxonomy,
        head_architecture=checkpoint_head_architecture,
        parent_hidden_size=checkpoint_parent_hidden_size,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    dev_gold_path = resolve_path(config.data.dev_gold_features, root)
    dev_formal_path = resolve_path(config.data.dev_formal_features, root)
    formal_predictions_path = resolve_path(
        config.data.dev_formal_predictions,
        root,
    )
    dev_gold_dataset = SubtypeFeatureDataset.from_file(dev_gold_path)
    dev_formal_dataset = SubtypeFeatureDataset.from_file(dev_formal_path)
    formal_payload = load_formal_predictions(
        formal_predictions_path,
        taxonomy=taxonomy,
    )
    validate_expected_frozen_gmner(
        formal_payload,
        expected=config.runtime.expected_dev_gmner_f1,
        tolerance=config.runtime.expected_dev_gmner_tolerance,
    )
    stage1_sha256 = str(checkpoint.get("stage1_checkpoint_sha256", ""))
    validate_feature_contract(
        dev_gold_dataset,
        taxonomy=taxonomy,
        split="dev",
        mode="gold",
        input_size=config.model.input_size,
        expected_stage1_sha256=stage1_sha256,
    )
    validate_feature_contract(
        dev_formal_dataset,
        taxonomy=taxonomy,
        split="dev",
        mode="formal",
        input_size=config.model.input_size,
        expected_stage1_sha256=stage1_sha256,
    )
    if (
        dev_formal_dataset.metadata.get("formal_predictions_sha256")
        != sha256_file(formal_predictions_path)
    ):
        raise ValueError(
            "Formal prediction artifact differs from the formal feature cache."
        )

    gold_metrics = evaluate_gold_spans(
        model,
        dev_gold_dataset,
        taxonomy=taxonomy,
        batch_size=config.optim.batch_size,
        device=device,
        include_detailed=True,
    )
    formal_result = evaluate_formal_predictions(
        model,
        dev_formal_dataset,
        formal_payload,
        taxonomy=taxonomy,
        batch_size=config.optim.batch_size,
        device=device,
        include_records=args.include_records,
    )
    result = {
        "metadata": {
            **formal_result["metadata"],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "head_architecture": checkpoint_head_architecture,
            "parent_hidden_size": model.parent_hidden_size,
            "model_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "base_model_loaded": False,
            "test_accessed": False,
        },
        "metrics": {
            **gold_metrics,
            **formal_result["metrics"],
        },
    }
    if args.include_records:
        result["records"] = formal_result["records"]
    output = resolve_path(
        args.output
        or (
            Path(config.runtime.output_dir)
            / "dev_metrics.json"
        ),
        root,
    )
    save_json_atomic(result, output)
    print(
        json.dumps(
            {
                "output": str(output),
                "checkpoint_epoch": result["metadata"]["checkpoint_epoch"],
                "gmner_identity_exact": result["metadata"][
                    "gmner_identity_exact"
                ],
                "test_accessed": False,
                **result["metrics"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
