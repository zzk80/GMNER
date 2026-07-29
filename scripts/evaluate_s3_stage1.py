"""Evaluate one S3.1 checkpoint on Dev only; no Test entry exists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.s3_stage1_builder import (
    build_s3_dataloader,
    build_s3_record_dataset,
    load_formal_stage1_teacher,
    load_locked_s3_initialization,
    load_s3_tokenizer,
    resolve_project_path,
)
from gmner.engine.s3_stage1_evaluator import evaluate_s3_stage1
from gmner.models.stage1 import (
    HierarchicalJointStage1,
    LegacyStage1RecordWrapper,
)
from gmner.s3_config import load_s3_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/fmnerg_twitter10000_stage1_s3_1.yaml",
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output",
        default="outputs/s3_stage1/seed42/dev_manual.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_s3_config(resolve_project_path(args.config, root))
    initialization = load_locked_s3_initialization(
        config,
        project_root=root,
    )
    tokenizer = load_s3_tokenizer(initialization)
    output_path = resolve_project_path(args.output, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dev_dataset = build_s3_record_dataset(
        formal_config=initialization.formal_config,
        tokenizer=tokenizer,
        project_root=root,
        working_dir=output_path.parent,
        split="dev",
    )
    dev_loader = build_s3_dataloader(
        dev_dataset,
        tokenizer=tokenizer,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.runtime.num_workers,
        seed=config.runtime.seed,
    )
    device = _device(config.runtime.device)
    teacher = load_formal_stage1_teacher(initialization)
    student = HierarchicalJointStage1(
        teacher,
        boundary_dropout=config.model.boundary_dropout,
        type_dropout=config.model.type_dropout,
    )
    checkpoint_path = resolve_project_path(args.checkpoint, root)
    payload = torch.load(checkpoint_path, map_location="cpu")
    if bool(payload.get("test_accessed", True)):
        raise ValueError("S3.1 checkpoint provenance indicates Test access.")
    if int(payload.get("seed", -1)) != 42:
        raise ValueError("Only the authorized S3.1 Seed42 checkpoint is valid.")
    student.load_state_dict(payload["model_state_dict"])
    student.to(device).eval()
    wrapper = LegacyStage1RecordWrapper(teacher).to(device).eval()
    report = evaluate_s3_stage1(
        model=student,
        dataloader=dev_loader,
        device=device,
        baseline_wrapper=wrapper,
        baseline_lock=initialization.baseline_lock,
    )
    report["checkpoint"] = str(checkpoint_path)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _device(value: str) -> torch.device:
    if str(value).startswith("cuda") and torch.cuda.is_available():
        return torch.device(value)
    return torch.device("cpu")


if __name__ == "__main__":
    main()
