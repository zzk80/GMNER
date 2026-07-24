"""Evaluate a recall-preserving coarse selector checkpoint on dev."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.coarse_region_selector_config import load_coarse_region_selector_config
from gmner.data import HierarchicalRecordCandidateCollator, RecordCandidateDataset
from gmner.engine.coarse_region_selector_evaluator import (
    evaluate_coarse_region_selector,
)
from gmner.models.coarse_region_selector import RecallPreservingCoarseSelector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_coarse_region_selector_config(args.config)
    checkpoint_path = resolve(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    dataset = RecordCandidateDataset(
        resolve(config.data.dev_cache, root),
        expected_stage1_sha256=checkpoint.get("stage1_checkpoint_sha256"),
        expected_candidate_sha256=checkpoint.get("candidate_config_sha256"),
    )
    loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=HierarchicalRecordCandidateCollator(),
    )
    device_name = args.device or config.runtime.device
    device = torch.device(
        device_name
        if str(device_name).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    model = RecallPreservingCoarseSelector(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loss_options = vars(config.loss).copy()
    loss_options["reference_budget"] = config.policy.final_budget
    metrics = evaluate_coarse_region_selector(
        model,
        loader,
        device,
        final_budget=config.policy.final_budget,
        base_keep_values=config.policy.base_keep_values,
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
