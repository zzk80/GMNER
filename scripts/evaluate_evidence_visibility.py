"""Evaluate a fixed M3.3A checkpoint on dev or an explicit one-time test set."""

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
from gmner.engine.evidence_visibility_evaluator import (
    evaluate_evidence_visibility,
)
from gmner.evidence_visibility_config import load_evidence_visibility_config
from scripts.train_evidence_visibility import load_frozen_chain
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--formal-cache",
        default=None,
        help="Explicit R16 cache. Required for test; optional dev override.",
    )
    parser.add_argument(
        "--expanded-cache",
        default=None,
        help="Explicit R36 cache. Required for test; optional dev override.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def evaluation_cache_paths(
    config,
    *,
    split: str,
    formal_cache: str | None,
    expanded_cache: str | None,
) -> tuple[str, str]:
    """Keep test access explicit while preserving the dev-only train config."""
    if split == "dev":
        return (
            formal_cache or config.data.formal_dev_cache,
            expanded_cache or config.data.expanded_dev_cache,
        )
    if not formal_cache or not expanded_cache:
        raise ValueError(
            "One-time test evaluation requires both --formal-cache and "
            "--expanded-cache. Test paths are intentionally absent from the "
            "training config."
        )
    return formal_cache, expanded_cache


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_evidence_visibility_config(args.config)
    if args.device:
        config.runtime.device = args.device
    formal_cache, expanded_cache = evaluation_cache_paths(
        config,
        split=args.split,
        formal_cache=args.formal_cache,
        expanded_cache=args.expanded_cache,
    )
    formal = RecordCandidateDataset(resolve(formal_cache, root))
    expanded = RecordCandidateDataset(resolve(expanded_cache, root))
    paired = PairedRecordCandidateDataset(formal, expanded)
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
    ) = load_frozen_chain(config, root, device)
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
    metrics = evaluate_evidence_visibility(
        model,
        fine_model,
        hierarchy,
        loader,
        device,
        decode_options=decode_options(hierarchy_config),
        loss_options=vars(config.loss).copy(),
    )
    payload = {"split": args.split, "metrics": metrics}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve(args.output, root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
