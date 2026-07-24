"""Evaluate an M3.2 fine grounding adapter on dev only."""

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
from gmner.engine.fine_grounding_adapter_evaluator import (
    evaluate_fine_grounding_adapter,
)
from gmner.fine_grounding_adapter_config import (
    load_fine_grounding_adapter_config,
)
from scripts.train_fine_grounding_adapter import (
    decode_options,
    load_frozen_models,
    resolve,
    validate_fingerprints,
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
    config = load_fine_grounding_adapter_config(args.config)
    if args.device:
        config.runtime.device = args.device
    formal = RecordCandidateDataset(resolve(config.data.formal_dev_cache, root))
    expanded = RecordCandidateDataset(
        resolve(config.data.expanded_dev_cache, root)
    )
    paired = PairedRecordCandidateDataset(formal, expanded)
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
    ) = load_frozen_models(config, root, device)
    validate_fingerprints(
        paired,
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    checkpoint = torch.load(resolve(args.checkpoint, root), map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loader = DataLoader(
        paired,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=PairedRecordCandidateCollator(),
    )
    loss_options = vars(config.loss).copy()
    loss_options["detector_reference_budget"] = (
        config.model.detector_reference_budget
    )
    metrics = evaluate_fine_grounding_adapter(
        model,
        hierarchy,
        loader,
        device,
        decode_options=decode_options(hierarchy_config),
        loss_options=loss_options,
    )
    payload = {"split": "dev", "metrics": metrics}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve(args.output, root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
