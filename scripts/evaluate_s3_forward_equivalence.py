"""Evaluate S3.0 equivalence on Dev only; this tool has no Test entry."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/fmnerg_twitter10000_stage1.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/fmnerg_stage1_roberta128/best_model.pt",
    )
    parser.add_argument(
        "--baseline-lock",
        default="docs/experiments/s3_stage1_baseline_lock.json",
    )
    parser.add_argument("--text-model-name", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--emission-tolerance",
        type=float,
        default=None,
        help=(
            "Diagnostic override. Formal runs must equal the baseline lock."
        ),
    )
    parser.add_argument(
        "--grounding-tolerance",
        type=float,
        default=None,
        help=(
            "Diagnostic override. Formal runs must equal the baseline lock."
        ),
    )
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help=(
            "Allow unlocked tolerances, but make the result ineligible for "
            "the formal S3.0 Gate."
        ),
    )
    parser.add_argument(
        "--output",
        default="outputs/s3_stage1/s3_0_dev_equivalence.json",
    )
    return parser.parse_args()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = _resolve(args.config, root)
    checkpoint_path = _resolve(args.checkpoint, root)
    lock_path = _resolve(args.baseline_lock, root)
    output_path = _resolve(args.output, root)
    baseline_lock = json.loads(lock_path.read_text(encoding="utf-8"))
    gate_lock = dict(baseline_lock.get("s3_0_gate") or {})
    if not gate_lock:
        raise ValueError("Baseline lock is missing the S3.0 Gate contract.")
    locked_emission_tolerance = float(
        gate_lock["emission_tolerance"]
    )
    locked_grounding_tolerance = float(
        gate_lock["grounding_tolerance"]
    )
    original_grounding_tolerance = float(
        gate_lock["original_grounding_tolerance"]
    )
    emission_tolerance = (
        locked_emission_tolerance
        if args.emission_tolerance is None
        else float(args.emission_tolerance)
    )
    grounding_tolerance = (
        locked_grounding_tolerance
        if args.grounding_tolerance is None
        else float(args.grounding_tolerance)
    )
    tolerance_matches_lock = (
        math.isclose(
            emission_tolerance,
            locked_emission_tolerance,
            rel_tol=0.0,
            abs_tol=0.0,
        )
        and math.isclose(
            grounding_tolerance,
            locked_grounding_tolerance,
            rel_tol=0.0,
            abs_tol=0.0,
        )
    )
    if not tolerance_matches_lock and not args.diagnostic_only:
        raise ValueError(
            "Formal S3.0 tolerances are locked. Use --diagnostic-only "
            "for a non-formal tolerance experiment."
        )
    if _sha256(config_path) != baseline_lock["config"]["sha256"]:
        raise ValueError("Formal Stage1 config differs from the baseline lock.")
    if _sha256(checkpoint_path) != baseline_lock["checkpoint"]["sha256"]:
        raise ValueError(
            "Formal Stage1 checkpoint differs from the baseline lock."
        )

    from gmner.config import load_config
    from gmner.constants import DEFAULT_LABEL2ID
    from gmner.data import (
        MMNERJsonDataset,
        RecordLevelStage1Collator,
        RecordLevelStage1Dataset,
        TextGraphBuilder,
        load_word_aligned_tokenizer,
    )
    from gmner.data.graph_builders import GraphBuilderConfig
    from gmner.engine.s3_forward_equivalence import (
        evaluate_s3_forward_equivalence,
    )
    from gmner.models import GMNERModel, LegacyStage1RecordWrapper
    from gmner.utils.io import maybe_convert_conll

    config = load_config(config_path)
    if args.text_model_name:
        config.model.text_model_name = args.text_model_name
    tokenizer = load_word_aligned_tokenizer(
        config.model.text_model_name,
        local_files_only=True,
    )
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=config.data.use_dependency_graph,
            dependency_backend=config.data.dependency_backend,
            dependency_model=config.data.dependency_model,
            window_size=config.data.graph_window_size,
        )
    )
    dev_source = maybe_convert_conll(
        _resolve(config.data.dev_file, root),
        output_path.parent,
    )
    expanded = MMNERJsonDataset(
        jsonl_path=str(dev_source),
        image_dir=str(_resolve(config.data.image_dir, root)),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=True,
        expand_entities_for_grounding=True,
        image_feature_dir=str(
            _resolve(config.data.image_feature_dir, root)
        ),
        image_annotation_dir=str(
            _resolve(config.data.image_annotation_dir, root)
        ),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=str(
            _resolve(config.data.groundability_type_priors, root)
        ),
        groundability_mention_priors=str(
            _resolve(config.data.groundability_mention_priors, root)
        ),
        region_min_score=config.data.region_min_score,
    )
    records = RecordLevelStage1Dataset(expanded, split="dev")
    if args.max_records is not None:
        records = Subset(
            records,
            range(min(int(args.max_records), len(records))),
        )
    dataloader = DataLoader(
        records,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=RecordLevelStage1Collator(tokenizer),
    )
    teacher = GMNERModel(
        config=config,
        num_labels=len(DEFAULT_LABEL2ID),
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    teacher.to(device).eval()
    wrapper = LegacyStage1RecordWrapper(teacher).to(device).eval()
    formal_gate_eligible = (
        args.max_records is None
        and not args.diagnostic_only
        and tolerance_matches_lock
    )
    report = evaluate_s3_forward_equivalence(
        teacher=teacher,
        wrapper=wrapper,
        dataloader=dataloader,
        device=device,
        emission_tolerance=emission_tolerance,
        grounding_tolerance=grounding_tolerance,
        original_grounding_tolerance=(
            original_grounding_tolerance
        ),
        expected_baseline=(
            baseline_lock if formal_gate_eligible else None
        ),
    )
    report["config_sha256"] = _sha256(config_path)
    report["checkpoint_sha256"] = _sha256(checkpoint_path)
    report["locked_tolerances"] = {
        "emission": locked_emission_tolerance,
        "grounding": locked_grounding_tolerance,
        "original_grounding": original_grounding_tolerance,
    }
    report["tolerance_matches_lock"] = tolerance_matches_lock
    report["diagnostic_only"] = bool(args.diagnostic_only)
    report["formal_gate_eligible"] = formal_gate_eligible
    report["formal_gate_passed"] = bool(
        formal_gate_eligible and report["gate_passed"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if formal_gate_eligible and not report["formal_gate_passed"]:
        raise SystemExit(2)
    if not formal_gate_eligible and not report["gate_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
