"""Evaluate a frozen M3.4A reliability checkpoint on dev only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.engine.siglip2_region_reliability_evaluator import (
    evaluate_siglip2_region_reliability,
)
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)
from scripts.train_siglip2_region_reliability import (
    _base_paired,
    _paired_dataset,
    load_frozen_reliability_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_siglip2_region_reliability_config(args.config)
    if args.device:
        config.runtime.device = args.device
    dataset, collator = _paired_dataset(config, root, "dev")
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
        evidence_checkpoint,
    ) = load_frozen_reliability_chain(config, root, device)
    validate_fingerprints(
        _base_paired(dataset),
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    checkpoint = torch.load(resolve(args.checkpoint, root), map_location="cpu")
    expected_evidence_epoch = checkpoint.get(
        "evidence_visibility_checkpoint_epoch"
    )
    if (
        expected_evidence_epoch is not None
        and expected_evidence_epoch != evidence_checkpoint.get("epoch")
    ):
        raise ValueError(
            "Reliability and Evidence Visibility checkpoints do not align."
        )
    checkpoint_mode = str(
        ((checkpoint.get("config") or {}).get("model") or {}).get(
            "feature_mode", ""
        )
    )
    if checkpoint_mode and checkpoint_mode != config.model.feature_mode:
        raise ValueError(
            f"Checkpoint mode {checkpoint_mode} != config mode "
            f"{config.model.feature_mode}."
        )
    if hasattr(dataset, "siglip2"):
        expected = checkpoint.get("siglip2_dev_build_signature")
        actual = dataset.siglip2.manifest.get("build_signature")
        if expected and expected != actual:
            raise ValueError("Checkpoint and dev SigLIP 2 cache signatures differ.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )
    metrics = evaluate_siglip2_region_reliability(
        model,
        evidence_model,
        fine_model,
        hierarchy,
        loader,
        device,
        decode_options=decode_options(hierarchy_config),
        loss_options=vars(config.loss).copy(),
        **vars(config.evaluation),
    )
    payload = {
        "split": "dev",
        "feature_mode": config.model.feature_mode,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metrics": metrics,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve(args.output, root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
