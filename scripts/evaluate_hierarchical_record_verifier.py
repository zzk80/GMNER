"""Evaluate a hierarchical record-verifier checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import HierarchicalRecordCandidateCollator, RecordCandidateDataset
from gmner.data.hierarchical_record_candidate_collator import (
    missing_hierarchical_cache_fields,
)
from gmner.engine.hierarchical_record_verifier_evaluator import (
    evaluate_hierarchical_record_verifier,
)
from gmner.hierarchical_record_verifier_config import (
    load_hierarchical_record_verifier_config,
)
from gmner.models.hierarchical_record_verifier import HierarchicalRecordVerifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument("--entity-threshold", type=float, default=None)
    parser.add_argument("--visible-from-null-threshold", type=float, default=None)
    parser.add_argument("--null-from-visible-threshold", type=float, default=None)
    parser.add_argument(
        "--region-override-mode",
        choices=["margin", "always", "utility"],
        default=None,
    )
    parser.add_argument("--region-override-logit-margin", type=float, default=None)
    parser.add_argument("--region-override-probability-margin", type=float, default=None)
    parser.add_argument("--override-damage-cost", type=float, default=None)
    parser.add_argument("--override-utility-threshold", type=float, default=None)
    parser.add_argument("--action-top-k", type=int, default=None)
    parser.add_argument("--action-execution-margin", type=float, default=None)
    parser.add_argument(
        "--include-override-risk-curve",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-visibility-correction",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-region-override",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--enable-action-controller",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--include-action-risk-curve",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args()


def resolve(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


STDOUT_METRIC_KEYS = (
    "loss",
    "span_f1",
    "entity_f1",
    "eeg_f1",
    "gmner_score",
    "pre_override_triple_f1",
    "action_label_span_count",
    "action_fixable_span_count",
    "action_preserve_span_count",
    "action_policy_fixable_top1_recall",
    "action_fix_score_mean",
    "action_damage_score_mean",
    "action_neutral_score_mean",
    "action_safe_neutral_score_mean",
    "action_useless_neutral_score_mean",
    "action_controller_executed_count",
    "action_controller_execution_margin",
    "action_controller_fix_count",
    "action_controller_damage_count",
    "action_controller_neutral_count",
    "action_controller_net_correction",
    "action_controller_action_precision",
    "action_controller_fix_rate_over_executed",
    "action_controller_neutral_rate_over_executed",
    "action_controller_to_null_count",
    "action_controller_to_real_count",
    "action_keep_correct_preservation_rate",
    "action_keep_correct_damaged_count",
    "action_keep_wrong_corrected_count",
    "action_controller_cumulative_max_net_correction",
    "action_controller_cumulative_max_count",
    "action_controller_cumulative_max_threshold",
    "region_override_count",
    "override_fix_count",
    "override_damage_count",
    "override_neutral_count",
    "override_net_correction",
    "override_cumulative_max_net_correction",
    "override_cumulative_max_count",
    "visible_corrected",
    "visible_damaged",
    "null_corrected",
    "null_damaged",
    "net_corrections",
    "base_correct_region_preservation_rate",
    "base_wrong_region_correction_rate",
    "gold_in_candidate_visible_count",
)


def build_stdout_payload(
    payload: dict,
    *,
    output_path: Path | None = None,
) -> dict:
    """Keep large risk curves in the JSON artifact, not in terminal output."""
    metrics = payload.get("metrics", {})
    omitted_fields = {
        key: len(value)
        for key, value in metrics.items()
        if isinstance(value, list)
    }
    if not omitted_fields:
        return payload

    compact_metrics = {
        key: metrics[key]
        for key in STDOUT_METRIC_KEYS
        if key in metrics
    }
    result = {
        "split": payload.get("split"),
        "metrics": compact_metrics,
        "omitted_list_fields": omitted_fields,
        "full_output": str(output_path) if output_path is not None else None,
    }
    if output_path is None:
        result["note"] = (
            "List-valued diagnostics were omitted from stdout. "
            "Pass --output to save the complete risk curves."
        )
    return result


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_hierarchical_record_verifier_config(args.config)
    cache_path = resolve(
        {
            "train": config.data.train_cache,
            "dev": config.data.dev_cache,
            "test": config.data.test_cache,
        }[args.split],
        root,
    )
    checkpoint = torch.load(resolve(args.checkpoint, root), map_location="cpu")
    dataset = RecordCandidateDataset(
        cache_path,
        expected_stage1_sha256=checkpoint.get("stage1_checkpoint_sha256"),
        expected_candidate_sha256=checkpoint.get("candidate_config_sha256"),
    )
    first_record = dataset.records[0] if dataset.records else {}
    missing = missing_hierarchical_cache_fields(first_record)
    if missing:
        raise ValueError(
            f"Cache {dataset.path} lacks hierarchical fields {missing}; verify the "
            "config path and rebuild it with the updated cache builder."
        )
    loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=HierarchicalRecordCandidateCollator(),
    )
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    model = HierarchicalRecordVerifier(config.model)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    loss_options = vars(config.loss).copy()
    loss_options["source_weights"] = torch.tensor(
        loss_options["source_weights"], device=device
    )
    decode = config.decode
    loss_options.update(
        {
            "action_top_k": (
                decode.action_top_k if args.action_top_k is None else args.action_top_k
            ),
            "action_enable_visibility_correction": (
                decode.enable_visibility_correction
                if args.enable_visibility_correction is None
                else args.enable_visibility_correction
            ),
            "action_visible_from_null_threshold": (
                decode.visible_from_null_threshold
                if args.visible_from_null_threshold is None
                else args.visible_from_null_threshold
            ),
            "action_null_from_visible_threshold": (
                decode.null_from_visible_threshold
                if args.null_from_visible_threshold is None
                else args.null_from_visible_threshold
            ),
        }
    )
    metrics = evaluate_hierarchical_record_verifier(
        model,
        loader,
        device,
        entity_threshold=(
            decode.entity_threshold
            if args.entity_threshold is None
            else args.entity_threshold
        ),
        decode_strategy=decode.strategy,
        stage1_spans_only=decode.stage1_spans_only,
        enable_visibility_correction=(
            decode.enable_visibility_correction
            if args.enable_visibility_correction is None
            else args.enable_visibility_correction
        ),
        enable_region_override=(
            decode.enable_region_override
            if args.enable_region_override is None
            else args.enable_region_override
        ),
        visible_from_null_threshold=(
            decode.visible_from_null_threshold
            if args.visible_from_null_threshold is None
            else args.visible_from_null_threshold
        ),
        null_from_visible_threshold=(
            decode.null_from_visible_threshold
            if args.null_from_visible_threshold is None
            else args.null_from_visible_threshold
        ),
        region_override_mode=(
            decode.region_override_mode
            if args.region_override_mode is None
            else args.region_override_mode
        ),
        region_override_logit_margin=(
            decode.region_override_logit_margin
            if args.region_override_logit_margin is None
            else args.region_override_logit_margin
        ),
        region_override_probability_margin=(
            decode.region_override_probability_margin
            if args.region_override_probability_margin is None
            else args.region_override_probability_margin
        ),
        override_damage_cost=(
            decode.override_damage_cost
            if args.override_damage_cost is None
            else args.override_damage_cost
        ),
        override_utility_threshold=(
            decode.override_utility_threshold
            if args.override_utility_threshold is None
            else args.override_utility_threshold
        ),
        include_override_risk_curve=(
            decode.include_override_risk_curve
            if args.include_override_risk_curve is None
            else args.include_override_risk_curve
        ),
        enable_action_controller=(
            decode.enable_action_controller
            if args.enable_action_controller is None
            else args.enable_action_controller
        ),
        action_top_k=(
            decode.action_top_k if args.action_top_k is None else args.action_top_k
        ),
        action_execution_margin=(
            decode.action_execution_margin
            if args.action_execution_margin is None
            else args.action_execution_margin
        ),
        include_action_risk_curve=(
            decode.include_action_risk_curve
            if args.include_action_risk_curve is None
            else args.include_action_risk_curve
        ),
        loss_options=loss_options,
    )
    payload = {"split": args.split, "metrics": metrics}
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    output = None
    if args.output:
        output = resolve(args.output, root)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    stdout_payload = build_stdout_payload(payload, output_path=output)
    print(json.dumps(stdout_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
