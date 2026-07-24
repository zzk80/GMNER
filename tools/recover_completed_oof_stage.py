"""Recover a completed OOF Stage1 whose Python process segfaulted at shutdown."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gmner.data.full_chain_oof_contract import (  # noqa: E402
    FULL_CHAIN_PIPELINE_KIND,
    FULL_CHAIN_PIPELINE_VERSION,
    fold_from_manifest,
    source_tree_sha256,
    validate_fold_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file  # noqa: E402


RECOVERY_KIND = "null_release_oof_completed_stage_recovery"
RECOVERY_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-work", required=True)
    parser.add_argument("--output-work-root", required=True)
    parser.add_argument("--fold-id", required=True, type=int)
    parser.add_argument("--failure-log", required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write the recovery receipt and promote Stage1 to complete.",
    )
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _descriptor(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Expected recovery artifact is missing: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def _command_sha256(commands: list[list[str]]) -> str:
    value = json.dumps(commands, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_descendant(path: Path, parent: Path, label: str) -> None:
    if path == parent or parent not in path.parents:
        raise ValueError(f"{label} must be a strict descendant of {parent}: {path}")


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _validate_checkpoint(
    checkpoint_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if Path(str(summary.get("best_checkpoint", ""))).resolve() != checkpoint_path:
        raise ValueError("Training summary references another best checkpoint.")
    metric_name = str(summary.get("best_metric_name", ""))
    if not metric_name:
        raise ValueError("Training summary has no best metric name.")
    summary_value = float(summary["best_metric_value"])

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError("Stage1 checkpoint payload is not a mapping.")
    model_state = dict(checkpoint.get("model_state_dict") or {})
    if not model_state:
        raise ValueError("Stage1 checkpoint has no model state.")
    metrics = dict(checkpoint.get("metrics") or {})
    if metric_name not in metrics:
        raise ValueError(f"Stage1 checkpoint lacks best metric {metric_name!r}.")
    checkpoint_value = float(metrics[metric_name])
    if not math.isclose(
        summary_value,
        checkpoint_value,
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        raise ValueError("Stage1 checkpoint and summary best metrics differ.")
    epoch = int(checkpoint.get("epoch", -1))
    if epoch < 0:
        raise ValueError("Stage1 checkpoint has an invalid epoch.")
    return {
        "best_metric_name": metric_name,
        "best_metric_value": summary_value,
        "best_epoch": epoch,
        "model_state_tensors": len(model_state),
    }


def _validate_tokenizer(output_root: Path) -> list[str]:
    tokenizer_root = output_root / "stage1" / "tokenizer"
    if not (tokenizer_root / "tokenizer_config.json").is_file():
        raise FileNotFoundError("Saved Stage1 tokenizer_config.json is missing.")
    vocabulary_names = {
        "tokenizer.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "bpe.codes",
    }
    observed = sorted(
        path.name
        for path in tokenizer_root.iterdir()
        if path.is_file() and path.name in vocabulary_names
    )
    if not observed:
        raise FileNotFoundError("Saved Stage1 tokenizer vocabulary is missing.")
    return observed


def recover_stage1(
    *,
    project_root: Path,
    fold_work: Path,
    output_work_root: Path,
    fold_id: int,
    failure_log: Path,
    execute: bool,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    fold_work = fold_work.resolve()
    output_work_root = output_work_root.resolve()
    failure_log = failure_log.resolve()
    if fold_id not in range(10):
        raise ValueError("Formal NULL Release OOF fold id must be in 0..9.")
    _require_descendant(fold_work, project_root / "knowledge", "fold work")
    _require_descendant(
        output_work_root, project_root / "outputs", "fold output work root"
    )
    if fold_work.name != f"fold{fold_id}" or output_work_root.name != f"fold{fold_id}":
        raise ValueError("Fold work and output roots do not match the fold id.")

    pipeline_path = fold_work / "pipeline_manifest.json"
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if pipeline.get("kind") != FULL_CHAIN_PIPELINE_KIND:
        raise ValueError("Not a NULL Release full-chain pipeline manifest.")
    if int(pipeline.get("format_version", -1)) != FULL_CHAIN_PIPELINE_VERSION:
        raise ValueError("Unsupported full-chain pipeline version.")
    if int(pipeline.get("fold_id", -1)) != fold_id:
        raise ValueError("Pipeline fold id does not match the recovery request.")
    if pipeline.get("sealed") is not False:
        raise ValueError("Only an unsealed pipeline can be recovered.")
    if pipeline.get("test_accessed") is not False:
        raise ValueError("Cannot recover a pipeline that accessed test data.")

    fold_manifest_path = Path(str(pipeline.get("fold_manifest", ""))).resolve()
    fold_manifest = validate_fold_manifest(
        fold_manifest_path, expected_num_folds=10
    )
    if pipeline.get("fold_manifest_sha256") != sha256_file(fold_manifest_path):
        raise ValueError("Pipeline fold-manifest hash changed.")
    current_source_hash = source_tree_sha256(project_root)
    if current_source_hash != fold_manifest.get("source_tree_sha256"):
        raise ValueError(
            "Current experiment source tree differs from the fold manifest."
        )
    if pipeline.get("source_tree_sha256") != current_source_hash:
        raise ValueError("Pipeline source-tree hash differs from the fold manifest.")
    fold = fold_from_manifest(fold_manifest, fold_id)
    if pipeline.get("train_record_ids_sha256") != fold["train_record_ids_sha256"]:
        raise ValueError("Pipeline training-id digest differs from the fold.")
    if pipeline.get("heldout_record_ids_sha256") != fold[
        "heldout_record_ids_sha256"
    ]:
        raise ValueError("Pipeline heldout-id digest differs from the fold.")

    stage = dict(dict(pipeline.get("stages") or {}).get("stage1") or {})
    receipt_path = fold_work / "stage1_completion_recovery.json"
    if stage.get("status") == "complete" and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(receipt["checkpoint"]["path"])
        if sha256_file(checkpoint_path) != receipt["checkpoint"]["sha256"]:
            raise ValueError("Recovered Stage1 checkpoint changed.")
        return receipt
    if stage.get("status") != "failed":
        raise ValueError(
            f"Stage1 must be failed before recovery, found {stage.get('status')!r}."
        )

    config_path = fold_work / "configs" / "stage1.yaml"
    config = _load_yaml(config_path)
    data = dict(config.get("data") or {})
    runtime = dict(config.get("runtime") or {})
    train_path = _resolve(str(data.get("train_file", "")), project_root)
    dev_path = _resolve(str(data.get("dev_file", "")), project_root)
    expected_train_path = Path(str(fold["train_file"])).resolve()
    if train_path != expected_train_path:
        raise ValueError("Stage1 config does not use this fold's train file.")
    if not dev_path.is_file():
        raise FileNotFoundError(f"Stage1 dev file is missing: {dev_path}")
    if data.get("test_file") != "__OOF_TEST_DISABLED__":
        raise ValueError("Stage1 OOF config does not disable the test split.")
    if runtime.get("evaluate_test_after_training") not in (False, None):
        raise ValueError("Stage1 OOF config enables automatic test evaluation.")

    checkpoint_path = output_work_root / "stage1" / "best_model.pt"
    summary_path = output_work_root / "stage1" / "train_summary.json"
    train_log_path = output_work_root / "stage1" / "train.log"
    resolved_config_path = output_work_root / "stage1" / "resolved_config.yaml"
    for path in (
        checkpoint_path,
        summary_path,
        train_log_path,
        resolved_config_path,
        failure_log,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required Stage1 completion artifact missing: {path}")
    if (output_work_root / "stage1" / "test_metrics.json").exists():
        raise ValueError("Stage1 recovery found a forbidden test_metrics.json.")

    train_log = train_log_path.read_text(encoding="utf-8", errors="replace")
    completion_markers = (
        f"Best checkpoint: {checkpoint_path}",
        "Skipping final test evaluation by request.",
    )
    for marker in completion_markers:
        if marker not in train_log:
            raise ValueError(f"Stage1 training log lacks completion marker: {marker}")
    failure_text = failure_log.read_text(encoding="utf-8", errors="replace")
    if "Signals.SIGSEGV: 11" not in failure_text:
        raise ValueError("Failure log does not record SIGSEGV signal 11.")
    if str(config_path) not in failure_text or str(project_root / "scripts" / "train.py") not in failure_text:
        raise ValueError("Failure log does not identify this Stage1 command.")

    checkpoint_validation = _validate_checkpoint(checkpoint_path, summary_path)
    tokenizer_files = _validate_tokenizer(output_work_root)
    expected_commands = [
        [
            sys.executable,
            "-u",
            str(project_root / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--skip-test-evaluation",
        ]
    ]
    expected_command_hash = _command_sha256(expected_commands)
    if stage.get("command_sha256") != expected_command_hash:
        raise ValueError("Failed Stage1 command hash differs from the formal command.")

    recovered_stage = {
        "status": "complete",
        "command_sha256": expected_command_hash,
        "test_accessed": False,
        "outputs": [_descriptor(checkpoint_path)],
        "inputs": [_descriptor(train_path), _descriptor(dev_path)],
        "heldout_excluded": True,
        "train_record_ids_sha256": fold["train_record_ids_sha256"],
        "config": _descriptor(config_path),
        "checkpoint": _descriptor(checkpoint_path),
    }
    receipt = {
        "format_version": RECOVERY_VERSION,
        "kind": RECOVERY_KIND,
        "status": "dry_run" if not execute else "recovered",
        "fold_id": fold_id,
        "stage": "stage1",
        "reason": "Training completed and SIGSEGV occurred during process shutdown.",
        "failure_signal": "SIGSEGV:11",
        "test_accessed": False,
        "source_tree_sha256": current_source_hash,
        "pipeline_manifest": str(pipeline_path),
        "pipeline_manifest_sha256_before": sha256_file(pipeline_path),
        "failure_log": _descriptor(failure_log),
        "summary": _descriptor(summary_path),
        "resolved_config": _descriptor(resolved_config_path),
        "checkpoint": _descriptor(checkpoint_path),
        "checkpoint_validation": checkpoint_validation,
        "tokenizer_files": tokenizer_files,
        "completion_markers": list(completion_markers),
        "recovered_stage": recovered_stage,
    }
    if not execute:
        return receipt

    backup_path = fold_work / "pipeline_manifest.pre_stage1_recovery.json"
    if not backup_path.exists():
        shutil.copy2(pipeline_path, backup_path)
    pipeline.setdefault("stages", {})["stage1"] = recovered_stage
    pipeline.setdefault("completion_recovery_history", []).append(
        {
            "recovered_at_utc": _utc_now(),
            "stage": "stage1",
            "reason": receipt["reason"],
            "failure_signal": receipt["failure_signal"],
            "failure_log_sha256": receipt["failure_log"]["sha256"],
            "checkpoint_sha256": receipt["checkpoint"]["sha256"],
            "summary_sha256": receipt["summary"]["sha256"],
            "source_tree_sha256": current_source_hash,
            "test_accessed": False,
        }
    )
    _atomic_write_json(pipeline_path, pipeline)
    receipt["recovered_at_utc"] = _utc_now()
    receipt["pipeline_manifest_sha256_after"] = sha256_file(pipeline_path)
    _atomic_write_json(receipt_path, receipt)
    return receipt


def main() -> None:
    args = parse_args()
    project_root = _resolve(args.project_root, ROOT)
    fold_work = _resolve(args.fold_work, project_root)
    output_work_root = _resolve(args.output_work_root, project_root)
    failure_log = _resolve(args.failure_log, project_root)
    result = recover_stage1(
        project_root=project_root,
        fold_work=fold_work,
        output_work_root=output_work_root,
        fold_id=args.fold_id,
        failure_log=failure_log,
        execute=args.execute,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "fold_id": result["fold_id"],
                "stage": result["stage"],
                "best_epoch": result["checkpoint_validation"]["best_epoch"],
                "best_metric_name": result["checkpoint_validation"][
                    "best_metric_name"
                ],
                "best_metric_value": result["checkpoint_validation"][
                    "best_metric_value"
                ],
                "checkpoint_sha256": result["checkpoint"]["sha256"],
                "test_accessed": result["test_accessed"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
