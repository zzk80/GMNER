from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from gmner.data.full_chain_oof_contract import (
    FULL_CHAIN_FOLD_MANIFEST_KIND,
    FULL_CHAIN_FOLD_MANIFEST_VERSION,
    FULL_CHAIN_PIPELINE_KIND,
    FULL_CHAIN_PIPELINE_VERSION,
    source_tree_sha256,
)
from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from tools.recover_completed_oof_stage import (
    _command_sha256,
    recover_stage1,
)


def _write_records(path: Path, record_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record_id in record_ids:
            stream.write(json.dumps({"id": record_id}) + "\n")


def _fixture(tmp_path: Path, *, sigsegv: bool = True) -> dict[str, Path]:
    project_root = tmp_path
    (project_root / "gmner").mkdir()
    (project_root / "gmner" / "placeholder.py").write_text(
        "VALUE = 1\n", encoding="utf-8"
    )
    (project_root / "scripts").mkdir()
    (project_root / "scripts" / "train.py").write_text(
        "print('placeholder')\n", encoding="utf-8"
    )
    (project_root / "configs").mkdir()
    (project_root / "configs" / "base.yaml").write_text(
        "name: base\n", encoding="utf-8"
    )
    source_hash = source_tree_sha256(project_root)

    fold_work = (
        project_root / "knowledge" / "null_release_oof" / "roberta128" / "fold1"
    )
    output_root = (
        project_root / "outputs" / "null_release_oof" / "roberta128" / "fold1"
    )
    folds_root = fold_work.parent / "folds"
    fold_work.mkdir(parents=True)
    output_root.mkdir(parents=True)
    dev_file = project_root / "data" / "dev.jsonl"
    _write_records(dev_file, ["dev"])

    all_ids = [str(index) for index in range(10)]
    folds = []
    for fold_id, heldout_id in enumerate(all_ids):
        train_ids = [value for value in all_ids if value != heldout_id]
        heldout_ids = [heldout_id]
        train_file = folds_root / f"train_fold{fold_id}.jsonl"
        heldout_file = folds_root / f"heldout_fold{fold_id}.jsonl"
        _write_records(train_file, train_ids)
        _write_records(heldout_file, heldout_ids)
        folds.append(
            {
                "fold": fold_id,
                "train_file": str(train_file.resolve()),
                "heldout_file": str(heldout_file.resolve()),
                "train_file_sha256": sha256_file(train_file),
                "heldout_file_sha256": sha256_file(heldout_file),
                "train_record_ids": train_ids,
                "heldout_record_ids": heldout_ids,
                "train_record_ids_sha256": stable_id_digest(train_ids),
                "heldout_record_ids_sha256": stable_id_digest(heldout_ids),
            }
        )
    manifest = {
        "format_version": FULL_CHAIN_FOLD_MANIFEST_VERSION,
        "kind": FULL_CHAIN_FOLD_MANIFEST_KIND,
        "num_folds": 10,
        "source_split": "train",
        "test_accessed": False,
        "source_tree_sha256": source_hash,
        "records": 10,
        "record_ids": all_ids,
        "record_ids_sha256": stable_id_digest(all_ids),
        "folds": folds,
    }
    manifest_path = folds_root / "fold_summary.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    config_path = fold_work / "configs" / "stage1.yaml"
    config_path.parent.mkdir()
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "train_file": folds[1]["train_file"],
                    "dev_file": str(dev_file.resolve()),
                    "test_file": "__OOF_TEST_DISABLED__",
                },
                "runtime": {"evaluate_test_after_training": False},
            }
        ),
        encoding="utf-8",
    )
    stage_output = output_root / "stage1"
    stage_output.mkdir(parents=True)
    checkpoint_path = stage_output / "best_model.pt"
    torch.save(
        {
            "epoch": 7,
            "metrics": {"gmner_score": 0.61},
            "model_state_dict": {"weight": torch.ones(1)},
        },
        checkpoint_path,
    )
    (stage_output / "train_summary.json").write_text(
        json.dumps(
            {
                "best_metric_name": "gmner_score",
                "best_metric_value": 0.61,
                "best_checkpoint": str(checkpoint_path.resolve()),
            }
        ),
        encoding="utf-8",
    )
    (stage_output / "resolved_config.yaml").write_text(
        config_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (stage_output / "train.log").write_text(
        f"Best checkpoint: {checkpoint_path.resolve()}\n"
        "Skipping final test evaluation by request.\n",
        encoding="utf-8",
    )
    tokenizer = stage_output / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")

    commands = [
        [
            sys.executable,
            "-u",
            str(project_root / "scripts" / "train.py"),
            "--config",
            str(config_path.resolve()),
            "--skip-test-evaluation",
        ]
    ]
    pipeline = {
        "format_version": FULL_CHAIN_PIPELINE_VERSION,
        "kind": FULL_CHAIN_PIPELINE_KIND,
        "fold_id": 1,
        "num_folds": 10,
        "fold_manifest": str(manifest_path.resolve()),
        "fold_manifest_sha256": sha256_file(manifest_path),
        "source_tree_sha256": source_hash,
        "train_record_ids_sha256": folds[1]["train_record_ids_sha256"],
        "heldout_record_ids_sha256": folds[1]["heldout_record_ids_sha256"],
        "test_accessed": False,
        "sealed": False,
        "stages": {
            "stage1": {
                "status": "failed",
                "command_sha256": _command_sha256(commands),
                "test_accessed": False,
            }
        },
    }
    pipeline_path = fold_work / "pipeline_manifest.json"
    pipeline_path.write_text(json.dumps(pipeline), encoding="utf-8")
    failure_log = project_root / "null_release_oof_fold1.log"
    signal_text = "Signals.SIGSEGV: 11" if sigsegv else "exit status 1"
    failure_log.write_text(
        f"Command ['{project_root / 'scripts' / 'train.py'}', "
        f"'{config_path.resolve()}'] died with <{signal_text}>.\n",
        encoding="utf-8",
    )
    return {
        "project_root": project_root,
        "fold_work": fold_work,
        "output_root": output_root,
        "failure_log": failure_log,
        "pipeline": pipeline_path,
        "checkpoint": checkpoint_path,
    }


def test_recover_completed_stage1_requires_execute_and_is_idempotent(
    tmp_path: Path,
) -> None:
    paths = _fixture(tmp_path)

    dry_run = recover_stage1(
        project_root=paths["project_root"],
        fold_work=paths["fold_work"],
        output_work_root=paths["output_root"],
        fold_id=1,
        failure_log=paths["failure_log"],
        execute=False,
    )

    assert dry_run["status"] == "dry_run"
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    assert pipeline["stages"]["stage1"]["status"] == "failed"

    recovered = recover_stage1(
        project_root=paths["project_root"],
        fold_work=paths["fold_work"],
        output_work_root=paths["output_root"],
        fold_id=1,
        failure_log=paths["failure_log"],
        execute=True,
    )

    assert recovered["status"] == "recovered"
    assert recovered["checkpoint_validation"]["best_epoch"] == 7
    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    stage = pipeline["stages"]["stage1"]
    assert stage["status"] == "complete"
    assert stage["test_accessed"] is False
    assert stage["heldout_excluded"] is True
    assert stage["checkpoint"]["sha256"] == sha256_file(paths["checkpoint"])
    assert len(pipeline["completion_recovery_history"]) == 1
    assert (paths["fold_work"] / "pipeline_manifest.pre_stage1_recovery.json").is_file()

    repeated = recover_stage1(
        project_root=paths["project_root"],
        fold_work=paths["fold_work"],
        output_work_root=paths["output_root"],
        fold_id=1,
        failure_log=paths["failure_log"],
        execute=True,
    )
    assert repeated["status"] == "recovered"


def test_recovery_rejects_a_non_sigsegv_failure(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, sigsegv=False)

    with pytest.raises(ValueError, match="does not record SIGSEGV"):
        recover_stage1(
            project_root=paths["project_root"],
            fold_work=paths["fold_work"],
            output_work_root=paths["output_root"],
            fold_id=1,
            failure_log=paths["failure_log"],
            execute=True,
        )

    pipeline = json.loads(paths["pipeline"].read_text(encoding="utf-8"))
    assert pipeline["stages"]["stage1"]["status"] == "failed"
