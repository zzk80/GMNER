"""Train S3.1 Seed42 on Train and select one checkpoint by Dev GMNER."""

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
from gmner.engine.s3_stage1_evaluator import evaluate_s3_stage1
from gmner.engine.s3_stage1_training import (
    S3Stage1Trainer,
    load_and_apply_scaling_report,
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
    scaling_path = resolve_project_path(
        config.loss.scaling_report,
        root,
    )
    scaling_report = load_and_apply_scaling_report(
        config=config,
        report_path=scaling_path,
        initialization=initialization,
        s3_config_path=config_path,
    )
    config.loss.scaling_report = str(scaling_path)
    output_dir = resolve_project_path(config.runtime.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_s3_tokenizer(initialization)
    train_dataset = build_s3_record_dataset(
        formal_config=initialization.formal_config,
        tokenizer=tokenizer,
        project_root=root,
        working_dir=output_dir,
        split="train",
    )
    dev_dataset = build_s3_record_dataset(
        formal_config=initialization.formal_config,
        tokenizer=tokenizer,
        project_root=root,
        working_dir=output_dir,
        split="dev",
    )
    train_loader = build_s3_dataloader(
        train_dataset,
        tokenizer=tokenizer,
        batch_size=config.optim.batch_size,
        shuffle=True,
        num_workers=config.runtime.num_workers,
        seed=config.runtime.seed,
    )
    dev_loader = build_s3_dataloader(
        dev_dataset,
        tokenizer=tokenizer,
        batch_size=config.optim.batch_size,
        shuffle=False,
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
    fixed_batch = next(iter(check_loader))
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
        batch=fixed_batch,
        device=device,
    )
    if not initialization_check["passed"]:
        raise RuntimeError(
            "S3.1 Student initialization differs from S3.0: "
            f"{initialization_check}"
        )
    del wrapper, teacher
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    trainer = S3Stage1Trainer(
        model=student,
        config=config,
        train_loader=train_loader,
        dev_loader=dev_loader,
        device=device,
        output_dir=output_dir,
        initialization=initialization,
        scaling_report=scaling_report,
        fixed_audit_batch=fixed_batch,
    )
    best_path, gradient_report = trainer.train()
    trainer.optimizer = None
    trainer.scheduler = None
    trainer.scaler = None
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    teacher = load_formal_stage1_teacher(initialization)
    wrapper = LegacyStage1RecordWrapper(teacher).to(device).eval()
    final_report = evaluate_s3_stage1(
        model=student,
        dataloader=dev_loader,
        device=device,
        baseline_wrapper=wrapper,
        baseline_lock=initialization.baseline_lock,
    )
    final_report["best_checkpoint"] = str(best_path)
    final_report["selection_metric"] = "stage1_dev_gmner"
    final_report["initialization_check"] = initialization_check
    final_report["gradient_audit_path"] = str(
        output_dir / "gradient_audit.json"
    )
    final_report["static_lambdas"] = {
        key: value
        for key, value in scaling_report["derived_lambdas"].items()
    }
    audit_regions = {
        region
        for audit in gradient_report["audits"]
        for region in audit[
            "weighted_max_min_ratio_by_region"
        ]
    }
    final_report["gradient_audit_static_scaling_unresolved"] = any(
        all(
            audit["weighted_max_min_ratio_by_region"].get(
                region, 0.0
            )
            >= 100.0
            for audit in gradient_report["audits"]
        )
        for region in audit_regions
    )
    final_path = output_dir / "dev_seed42_gate.json"
    final_path.write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


def _device(value: str) -> torch.device:
    if str(value).startswith("cuda") and torch.cuda.is_available():
        return torch.device(value)
    return torch.device("cpu")


if __name__ == "__main__":
    main()
