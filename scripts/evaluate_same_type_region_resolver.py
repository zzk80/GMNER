"""Evaluate M3.3A-P3 C1 on Dev without exposing a Test interface."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import (
    PairedRecordCandidateCollator,
    PairedRecordCandidateDataset,
    RecordCandidateDataset,
)
from gmner.engine.same_type_region_resolver_evaluator import (
    evaluate_same_type_region_resolver,
)
from gmner.same_type_region_resolver_config import (
    load_same_type_region_resolver_config,
)
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)
from scripts.train_same_type_region_resolver import load_frozen_chain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Required unless --disable-resolver is used.",
    )
    parser.add_argument("--disable-resolver", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.disable_resolver and not args.checkpoint:
        raise ValueError(
            "--checkpoint is required when the resolver is enabled."
        )
    root = Path(__file__).resolve().parents[1]
    config = load_same_type_region_resolver_config(args.config)
    if args.device:
        config.runtime.device = args.device
    formal = RecordCandidateDataset(
        resolve(config.data.formal_dev_cache, root)
    )
    expanded = RecordCandidateDataset(
        resolve(config.data.expanded_dev_cache, root)
    )
    paired = PairedRecordCandidateDataset(formal, expanded)
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
        _,
        _,
    ) = load_frozen_chain(config, root, device)
    validate_fingerprints(
        paired,
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    if args.checkpoint:
        checkpoint = torch.load(
            resolve(args.checkpoint, root), map_location="cpu"
        )
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loader = DataLoader(
        paired,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=PairedRecordCandidateCollator(),
    )
    metrics = evaluate_same_type_region_resolver(
        model,
        evidence_model,
        fine_model,
        hierarchy,
        loader,
        device,
        decode_options=decode_options(hierarchy_config),
        loss_options=vars(config.loss).copy(),
        enabled=not args.disable_resolver,
    )
    baseline_difference = abs(
        float(metrics["baseline_gmner_score"])
        - float(config.runtime.expected_dev_baseline_gmner)
    )
    if baseline_difference > float(
        config.runtime.baseline_tolerance
    ):
        raise RuntimeError(
            "Frozen M3.3A baseline mismatch: "
            f"{metrics['baseline_gmner_score']} versus "
            f"{config.runtime.expected_dev_baseline_gmner}."
        )
    if args.disable_resolver and (
        abs(float(metrics["gmner_delta"])) > 1e-12
        or int(metrics["override_count"]) != 0
    ):
        raise RuntimeError(
            "Disabled resolver did not exactly reproduce M3.3A."
        )
    payload = {
        "split": "dev",
        "resolver_enabled": not args.disable_resolver,
        "metrics": metrics,
        "test_accessed": False,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve(args.output, root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
