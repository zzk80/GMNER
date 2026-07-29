"""Run the preregistered S3.1 100-step Train-only scaling probe."""

from __future__ import annotations

import argparse
import gc
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
from gmner.engine.s3_stage1_training import (
    run_s3_scaling_probe,
    verify_student_backbone_initialization,
)
from gmner.models.stage1 import (
    HierarchicalJointStage1,
    LegacyStage1RecordWrapper,
)
from gmner.s3_config import load_s3_config
from gmner.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/fmnerg_twitter10000_stage1_s3_1.yaml",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Validate locked Train initialization, then exit before updates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve_project_path(args.config, root)
    config = load_s3_config(config_path)
    if int(config.runtime.seed) != 42:
        raise ValueError(
            "S3.1 first-stage authorization is restricted to Seed42."
        )
    set_seed(config.runtime.seed)
    initialization = load_locked_s3_initialization(
        config,
        project_root=root,
    )
    output_path = resolve_project_path(
        config.runtime.probe_output,
        root,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = load_s3_tokenizer(initialization)
    train_dataset = build_s3_record_dataset(
        formal_config=initialization.formal_config,
        tokenizer=tokenizer,
        project_root=root,
        working_dir=output_path.parent,
        split="train",
    )
    train_loader = build_s3_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=config.optim.batch_size,
        shuffle=True,
        num_workers=config.runtime.num_workers,
        seed=config.runtime.seed,
    )
    check_loader = build_s3_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=0,
        seed=config.runtime.seed,
    )
    device = _device(config.runtime.device)
    teacher = load_formal_stage1_teacher(initialization)
    student = HierarchicalJointStage1(
        teacher,
        boundary_dropout=config.model.boundary_dropout,
        type_dropout=config.model.type_dropout,
    ).to(device)
    wrapper = LegacyStage1RecordWrapper(teacher).to(device).eval()
    initialization_check = verify_student_backbone_initialization(
        student=student,
        wrapper=wrapper,
        batch=next(iter(check_loader)),
        device=device,
    )
    if not initialization_check["passed"]:
        raise RuntimeError(
            "S3.1 Student initialization differs from S3.0: "
            f"{initialization_check}"
        )
    if args.preflight:
        preflight = {
            "kind": "s3_1_scaling_probe_preflight",
            "scope": "train_only",
            "initialization_check": initialization_check,
            "records": len(train_dataset),
            "probe_updates": 0,
            "checkpoint_saved": False,
            "dev_accessed": False,
            "test_accessed": False,
        }
        preflight_path = output_path.with_name(
            "s3_1_preflight_seed42.json"
        )
        preflight_path.write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return
    del wrapper, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    report = run_s3_scaling_probe(
        model=student,
        train_loader=train_loader,
        config=config,
        device=device,
        initialization=initialization,
        s3_config_path=config_path,
        initialization_check=initialization_check,
    )
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
