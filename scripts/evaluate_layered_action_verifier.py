"""Evaluate M3.6A on dev without opening the frozen formal test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.engine.layered_action_verifier_evaluator import (
    evaluate_layered_action_verifier,
)
from gmner.layered_action_verifier_config import (
    load_layered_action_verifier_config,
)
from gmner.models.layered_action_verifier import (
    ACTION_MODE_NULL_RELEASE_ONLY,
    LayeredActionVerifier,
)
from gmner.models.null_release_verifier import NullReleaseVerifier
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
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--output", default=None)
    parser.add_argument("--execution-margin", type=float, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--no-risk-curve", action="store_true", help="Omit the long curve."
    )
    return parser.parse_args()


def _build_model(config):
    if config.action_mode == ACTION_MODE_NULL_RELEASE_ONLY:
        return NullReleaseVerifier(config)
    return LayeredActionVerifier(config)


def main(*, required_action_mode: str | None = None) -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_layered_action_verifier_config(args.config)
    if (
        required_action_mode is not None
        and config.model.action_mode != required_action_mode
    ):
        raise ValueError(
            f"This entry point requires model.action_mode={required_action_mode!r}, "
            f"found {config.model.action_mode!r}."
        )
    reliability_config = load_siglip2_region_reliability_config(
        resolve(config.frozen.reliability_config, root)
    )
    dataset, collator = _paired_dataset(reliability_config, root, "dev")
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    (
        reliability_model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
        _,
    ) = load_frozen_reliability_chain(reliability_config, root, device)
    reliability_checkpoint = torch.load(
        resolve(config.frozen.reliability_checkpoint, root),
        map_location="cpu",
    )
    reliability_model.load_state_dict(reliability_checkpoint["model_state_dict"])
    reliability_model.to(device).eval()
    validate_fingerprints(
        _base_paired(dataset),
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    if args.max_records is not None:
        dataset = Subset(dataset, range(min(max(1, args.max_records), len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=reliability_config.data.num_workers,
        collate_fn=collator,
    )
    checkpoint = torch.load(resolve(args.checkpoint, root), map_location="cpu")
    checkpoint_mode = (
        checkpoint.get("config", {}).get("model", {}).get("action_mode", "full")
    )
    if checkpoint_mode != config.model.action_mode:
        raise ValueError(
            "Checkpoint/config action_mode mismatch: "
            f"{checkpoint_mode!r} != {config.model.action_mode!r}."
        )
    if (
        config.model.action_mode == ACTION_MODE_NULL_RELEASE_ONLY
        and not bool(checkpoint.get("full_chain_oof"))
    ):
        raise ValueError(
            "NULL Release evaluation requires a checkpoint trained from the "
            "validated 10-fold full-chain OOF cache."
        )
    model = _build_model(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    execution_margin = (
        config.evaluation.execution_margin
        if args.execution_margin is None
        else float(args.execution_margin)
    )
    metrics = evaluate_layered_action_verifier(
        model,
        reliability_model,
        evidence_model,
        fine_model,
        hierarchy,
        loader,
        device,
        decode_options=decode_options(hierarchy_config),
        loss_options=vars(config.loss).copy(),
        execution_margin=execution_margin,
        include_risk_curve=(
            config.evaluation.include_risk_curve and not args.no_risk_curve
        ),
        identity_tolerance=config.evaluation.identity_tolerance,
        minimum_keep_preservation_rate=(
            config.evaluation.minimum_keep_preservation_rate
        ),
        minimum_net_correction=config.evaluation.minimum_net_correction,
    )
    output = (
        resolve(args.output, root)
        if args.output
        else resolve(config.runtime.output_dir, root) / "dev_metrics.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "split": "dev",
                "checkpoint": str(resolve(args.checkpoint, root).resolve()),
                "action_mode": config.model.action_mode,
                "execution_margin": execution_margin,
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    summary_keys = (
        "baseline_gmner_score",
        "gmner_score",
        "eeg_f1",
        "entity_f1",
        "gmner_net_correction",
        "to_null_net_correction",
        "to_real_net_correction",
        "null_release_net_correction",
        "region_switch_net_correction",
        "keep_correct_preservation_rate",
        "layer1_accuracy",
        "layer2_top4_accuracy",
        "action_cumulative_max_net_correction",
        "epoch0_identity_pass",
        "go_no_go",
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "action_mode": config.model.action_mode,
                **{key: metrics.get(key) for key in summary_keys},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
