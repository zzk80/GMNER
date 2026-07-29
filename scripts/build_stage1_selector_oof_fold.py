"""Build, validate, seal, and clean one strict D1 Stage1 OOF fold."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    atomic_write_json,
    fold_from_manifest,
    source_tree_sha256,
    validate_fold_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file
from gmner.data.stage1_selector_oof_cache import validate_selector_oof_payload


PIPELINE_KIND = "stage1_candidate_selector_oof_fold_pipeline"
PIPELINE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage1-config",
        default="configs/fmnerg_twitter10000_stage1.yaml",
    )
    parser.add_argument(
        "--fold-summary",
        default=(
            "knowledge/stage1_candidate_selector_oof/roberta128/"
            "folds/fold_summary.json"
        ),
    )
    parser.add_argument(
        "--reference-proof-root",
        default="knowledge/null_release_oof/roberta128",
    )
    parser.add_argument(
        "--work-root",
        default="knowledge/stage1_candidate_selector_oof/roberta128",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/stage1_candidate_selector_oof/roberta128",
    )
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--reuse-stage1-checkpoint", default=None)
    parser.add_argument(
        "--require-reference-file-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--rebuild-fold-manifest", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=8.0)
    return parser.parse_args()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _write_yaml(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _command_digest(command: list[str]) -> str:
    import hashlib

    encoded = json.dumps(command, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _run_logged(
    command: list[str],
    *,
    root: Path,
    log_path: Path,
    allow_completed_stage1_crash: bool = False,
) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    recovered = False
    if result.returncode != 0 and allow_completed_stage1_crash:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        completed = (
            "Best checkpoint:" in text
            and "Skipping final test evaluation by request." in text
        )
        if completed:
            recovered = True
        else:
            raise subprocess.CalledProcessError(result.returncode, command)
    elif result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return {
        "command": command,
        "command_sha256": _command_digest(command),
        "returncode": int(result.returncode),
        "post_completion_process_failure_recovered": recovered,
        "log": str(log_path),
        "log_sha256": sha256_file(log_path),
    }


def _reference_proof(
    proof_root: Path,
    *,
    fold: dict,
    fold_id: int,
    require_file_hashes: bool,
) -> tuple[Path, dict, str]:
    proof_path = proof_root / f"fold{fold_id}" / "fold_proof.json"
    pipeline_path = proof_root / f"fold{fold_id}" / "pipeline_manifest.json"
    if not proof_path.is_file() or not pipeline_path.is_file():
        raise FileNotFoundError(
            f"Missing archived fold proof or pipeline for fold {fold_id}."
        )
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if int(proof.get("fold_id", -1)) != fold_id:
        raise ValueError("Archived proof has the wrong fold id.")
    if proof.get("excluded_heldout") is not True:
        raise ValueError("Archived proof does not assert heldout exclusion.")
    if [str(value) for value in proof.get("training_record_ids") or []] != [
        str(value) for value in fold["train_record_ids"]
    ]:
        raise ValueError("Fold train IDs differ from the archived OOF proof.")
    if [str(value) for value in proof.get("heldout_record_ids") or []] != [
        str(value) for value in fold["heldout_record_ids"]
    ]:
        raise ValueError("Fold heldout IDs differ from the archived OOF proof.")
    if require_file_hashes:
        if proof.get("train_file_sha256") != fold["train_file_sha256"]:
            raise ValueError("Fold train file hash differs from the archived proof.")
        if proof.get("heldout_file_sha256") != fold["heldout_file_sha256"]:
            raise ValueError("Fold heldout file hash differs from the archived proof.")
    stage1 = dict((pipeline.get("stages") or {}).get("stage1") or {})
    checkpoint_sha256 = str(
        dict(stage1.get("checkpoint") or {}).get("sha256", "")
    )
    if not checkpoint_sha256:
        raise ValueError("Archived pipeline has no Stage1 checkpoint hash.")
    return proof_path, proof, checkpoint_sha256


def _training_config_view(config: dict) -> dict:
    payload = deepcopy(config)
    data = payload.setdefault("data", {})
    runtime = payload.setdefault("runtime", {})
    for key in ("train_file", "dev_file", "test_file"):
        data.pop(key, None)
    runtime.pop("output_dir", None)
    return payload


def _validate_reused_checkpoint(
    checkpoint: Path,
    *,
    expected_sha256: str,
    effective_config: dict,
    archived_config_path: Path,
) -> None:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = sha256_file(checkpoint)
    if actual != expected_sha256:
        raise ValueError(
            "Reused Stage1 checkpoint hash differs from the archived fold proof: "
            f"expected {expected_sha256}, found {actual}."
        )
    if not archived_config_path.is_file():
        raise FileNotFoundError(archived_config_path)
    archived = _load_yaml(archived_config_path)
    if _training_config_view(archived) != _training_config_view(effective_config):
        raise ValueError(
            "Current Stage1 training settings differ from the reused checkpoint."
        )
    try:
        torch.load(checkpoint, map_location="cpu")
    except Exception as error:
        raise ValueError(f"Reused Stage1 checkpoint cannot be loaded: {error}") from error


def _safe_remove_tree(path: Path, *, allowed_root: Path) -> int:
    resolved = path.resolve()
    root = allowed_root.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to remove path outside fold output root: {resolved}")
    if not resolved.exists():
        return 0
    size = sum(
        item.stat().st_size
        for item in resolved.rglob("*")
        if item.is_file()
    )
    shutil.rmtree(resolved)
    return size


def _prepare_manifest(
    *,
    root: Path,
    stage1_template: Path,
    manifest_path: Path,
    seed: int,
    rebuild: bool,
) -> None:
    if manifest_path.is_file() and not rebuild:
        return
    command = [
        sys.executable,
        str(root / "scripts" / "build_evidence_folds.py"),
        "--config",
        str(stage1_template),
        "--output-dir",
        str(manifest_path.parent),
        "--num-folds",
        "10",
        "--seed",
        str(seed),
    ]
    if rebuild:
        command.append("--force")
    log_path = manifest_path.parent / "fold_build.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=root,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def build_fold(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    if args.fold_id not in range(10):
        raise ValueError("--fold-id must be in 0..9.")
    stage1_template_path = _resolve(args.stage1_config, root)
    manifest_path = _resolve(args.fold_summary, root)
    proof_root = _resolve(args.reference_proof_root, root)
    work_root = _resolve(args.work_root, root)
    output_root = _resolve(args.output_root, root)
    if not stage1_template_path.is_file():
        raise FileNotFoundError(stage1_template_path)

    _prepare_manifest(
        root=root,
        stage1_template=stage1_template_path,
        manifest_path=manifest_path,
        seed=args.seed,
        rebuild=args.rebuild_fold_manifest,
    )
    manifest = validate_fold_manifest(manifest_path, expected_num_folds=10)
    current_source_tree = source_tree_sha256(root)
    if manifest.get("source_tree_sha256") != current_source_tree:
        raise ValueError(
            "Source/config tree changed after the D1 fold manifest was created. "
            "Restart Phase 1 with REBUILD_FOLDS=1 before training any fold."
        )
    fold = fold_from_manifest(manifest, args.fold_id)
    proof_path, _, reference_checkpoint_sha256 = _reference_proof(
        proof_root,
        fold=fold,
        fold_id=args.fold_id,
        require_file_hashes=args.require_reference_file_hashes,
    )

    fold_work = work_root / f"fold{args.fold_id}"
    fold_output = output_root / f"fold{args.fold_id}"
    config_path = fold_work / "configs" / "stage1.yaml"
    compact_path = fold_work / "heldout_candidates.pt"
    full_cache_path = fold_work / "intermediate" / "heldout_r16_full.pt"
    pipeline_path = fold_work / "pipeline_manifest.json"
    stage1_log = fold_work / "logs" / "stage1_train.log"
    candidate_log = fold_work / "logs" / "candidate_build.log"
    compact_log = fold_work / "logs" / "compact.log"
    checkpoint_path = fold_output / "stage1" / "best_model.pt"
    fold_work.mkdir(parents=True, exist_ok=True)

    existing: dict = {}
    if pipeline_path.is_file():
        existing = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if existing and compact_path.is_file():
        if existing.get("sealed") is True:
            payload = torch.load(compact_path, map_location="cpu")
            validate_selector_oof_payload(
                payload,
                expected_fold_id=args.fold_id,
                expected_num_folds=10,
                expected_record_ids=fold["heldout_record_ids"],
            )
            if sha256_file(compact_path) != existing.get("compact_cache_sha256"):
                raise ValueError("Sealed compact selector cache hash changed.")
            return {
                "status": "already_sealed",
                "fold_id": args.fold_id,
                "records": len(fold["heldout_record_ids"]),
                "compact_cache": str(compact_path),
                "compact_cache_sha256": sha256_file(compact_path),
                "test_accessed": False,
            }

    template = _load_yaml(stage1_template_path)
    stage1_config = deepcopy(template)
    stage1_config["data"]["train_file"] = str(Path(fold["train_file"]).resolve())
    stage1_config["data"]["dev_file"] = str(
        _resolve(template["data"]["dev_file"], root)
    )
    stage1_config["data"]["test_file"] = "__OOF_TEST_DISABLED__"
    stage1_config["runtime"]["output_dir"] = str(fold_output / "stage1")
    stage1_config["runtime"]["seed"] = int(args.seed)
    stage1_config["runtime"]["save_latest_checkpoint"] = False
    _write_yaml(config_path, stage1_config)

    preflight = {
        "format_version": PIPELINE_VERSION,
        "kind": PIPELINE_KIND,
        "sealed": False,
        "status": "prepared",
        "fold_id": args.fold_id,
        "records_train": len(fold["train_record_ids"]),
        "records_heldout": len(fold["heldout_record_ids"]),
        "fold_summary": str(manifest_path),
        "fold_summary_sha256": sha256_file(manifest_path),
        "reference_fold_proof": str(proof_path),
        "reference_fold_proof_sha256": sha256_file(proof_path),
        "stage1_config": str(config_path),
        "stage1_config_sha256": sha256_file(config_path),
        "source_tree_sha256": current_source_tree,
        "test_accessed": False,
    }
    existing_stage1 = dict(existing.get("stage1") or {})
    same_preflight = (
        existing.get("fold_summary_sha256") == preflight["fold_summary_sha256"]
        and existing.get("reference_fold_proof_sha256")
        == preflight["reference_fold_proof_sha256"]
        and existing.get("stage1_config_sha256")
        == preflight["stage1_config_sha256"]
        and existing.get("source_tree_sha256")
        == preflight["source_tree_sha256"]
        and existing.get("test_accessed") is False
    )
    if existing_stage1 and same_preflight:
        preflight["stage1"] = existing_stage1
        preflight["status"] = str(existing.get("status", "stage1_complete"))
    if args.prepare_only and args.reuse_stage1_checkpoint:
        reused = _resolve(args.reuse_stage1_checkpoint, root)
        archived_config = (
            proof_root / f"fold{args.fold_id}" / "configs" / "stage1.yaml"
        )
        _validate_reused_checkpoint(
            reused,
            expected_sha256=reference_checkpoint_sha256,
            effective_config=stage1_config,
            archived_config_path=archived_config,
        )
        preflight["reuse_checkpoint_preflight"] = {
            "checkpoint": str(reused),
            "checkpoint_sha256": sha256_file(reused),
            "matches_reference_checkpoint": True,
            "training_config_compatible": True,
        }
    atomic_write_json(pipeline_path, preflight)
    if args.prepare_only:
        return preflight

    free_bytes = shutil.disk_usage(root).free
    required_bytes = int(float(args.min_free_gb) * (1024**3))
    if free_bytes < required_bytes:
        raise OSError(
            f"D1 fold requires {args.min_free_gb:.1f} GiB free by policy; "
            f"found {free_bytes / (1024**3):.2f} GiB."
        )

    stage1_run: dict
    if args.reuse_stage1_checkpoint:
        reused = _resolve(args.reuse_stage1_checkpoint, root)
        archived_config = proof_root / f"fold{args.fold_id}" / "configs" / "stage1.yaml"
        _validate_reused_checkpoint(
            reused,
            expected_sha256=reference_checkpoint_sha256,
            effective_config=stage1_config,
            archived_config_path=archived_config,
        )
        checkpoint_path = reused
        stage1_run = {
            "mode": "reused_archived_checkpoint",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "reference_checkpoint_sha256": reference_checkpoint_sha256,
            "post_completion_process_failure_recovered": False,
        }
    elif preflight.get("stage1"):
        previous = dict(preflight["stage1"])
        previous_checkpoint = Path(str(previous.get("checkpoint", "")))
        if (
            previous.get("mode") in {
                "trained_for_d1",
                "trained_for_d1_resume",
            }
            and previous_checkpoint.is_file()
            and previous.get("checkpoint_sha256")
            == sha256_file(previous_checkpoint)
        ):
            checkpoint_path = previous_checkpoint
            stage1_run = {
                **previous,
                "mode": "trained_for_d1_resume",
            }
        else:
            raise ValueError(
                "Existing unsealed Stage1 provenance is invalid. Remove only the "
                "fold pipeline manifest after auditing it before retraining."
            )
    else:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-u",
            str(root / "scripts" / "train.py"),
            "--config",
            str(config_path),
            "--skip-test-evaluation",
        ]
        run = _run_logged(
            command,
            root=root,
            log_path=stage1_log,
            allow_completed_stage1_crash=True,
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "Stage1 training did not produce best_model.pt."
            )
        try:
            torch.load(checkpoint_path, map_location="cpu")
        except Exception as error:
            raise ValueError(
                f"Stage1 checkpoint cannot be loaded: {error}"
            ) from error
        stage1_run = {
            "mode": "trained_for_d1",
            **run,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "reference_checkpoint_sha256": reference_checkpoint_sha256,
        }
    preflight["status"] = "stage1_complete"
    preflight["stage1"] = stage1_run
    atomic_write_json(pipeline_path, preflight)

    full_cache_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_command = [
        sys.executable,
        "-u",
        str(root / "scripts" / "build_record_candidate_cache.py"),
        "--config",
        str(config_path),
        "--checkpoint",
        str(checkpoint_path),
        "--split",
        "train",
        "--input-file",
        str(Path(fold["heldout_file"]).resolve()),
        "--oof-fold-id",
        str(args.fold_id),
        "--output",
        str(full_cache_path),
        "--k-best",
        "6",
        "--max-span-candidates",
        "12",
        "--top-m-types",
        "3",
        "--boundary-shift",
        "1",
        "--boundary-penalty",
        "0.25",
        "--max-regions",
        "16",
        "--batch-size",
        str(args.inference_batch_size),
        "--device",
        args.device,
    ]
    candidate_run = _run_logged(
        candidate_command,
        root=root,
        log_path=candidate_log,
    )
    if not full_cache_path.is_file():
        raise FileNotFoundError("Candidate builder did not produce the R16 cache.")

    compact_command = [
        sys.executable,
        "-u",
        str(root / "scripts" / "compact_stage1_selector_oof_cache.py"),
        "--input-cache",
        str(full_cache_path),
        "--output",
        str(compact_path),
        "--fold-summary",
        str(manifest_path),
        "--fold-id",
        str(args.fold_id),
        "--stage1-config",
        str(config_path),
        "--reference-fold-proof",
        str(proof_path),
    ]
    if not args.require_reference_file_hashes:
        compact_command.append("--no-require-reference-file-hashes")
    compact_run = _run_logged(
        compact_command,
        root=root,
        log_path=compact_log,
    )
    compact_payload = torch.load(compact_path, map_location="cpu")
    validate_selector_oof_payload(
        compact_payload,
        expected_fold_id=args.fold_id,
        expected_num_folds=10,
        expected_record_ids=fold["heldout_record_ids"],
    )
    compact_sha256 = sha256_file(compact_path)
    full_cache_sha256 = sha256_file(full_cache_path)

    deleted_bytes = 0
    if args.cleanup:
        full_summary = full_cache_path.with_suffix(".summary.json")
        if full_summary.is_file():
            deleted_bytes += full_summary.stat().st_size
            full_summary.unlink()
        deleted_bytes += full_cache_path.stat().st_size
        full_cache_path.unlink()
        if str(stage1_run["mode"]).startswith("trained_for_d1"):
            deleted_bytes += _safe_remove_tree(
                fold_output,
                allowed_root=output_root,
            )
        reloaded = torch.load(compact_path, map_location="cpu")
        validate_selector_oof_payload(
            reloaded,
            expected_fold_id=args.fold_id,
            expected_num_folds=10,
            expected_record_ids=fold["heldout_record_ids"],
        )
        if sha256_file(compact_path) != compact_sha256:
            raise ValueError("Compact selector cache changed during cleanup.")

    pipeline = {
        "format_version": PIPELINE_VERSION,
        "kind": PIPELINE_KIND,
        "sealed": True,
        "status": "cleaned" if args.cleanup else "complete",
        "fold_id": args.fold_id,
        "num_folds": 10,
        "records": len(fold["heldout_record_ids"]),
        "train_record_ids_sha256": fold["train_record_ids_sha256"],
        "heldout_record_ids_sha256": fold["heldout_record_ids_sha256"],
        "fold_summary": str(manifest_path),
        "fold_summary_sha256": sha256_file(manifest_path),
        "reference_fold_proof": str(proof_path),
        "reference_fold_proof_sha256": sha256_file(proof_path),
        "stage1_config": str(config_path),
        "stage1_config_sha256": sha256_file(config_path),
        "stage1": stage1_run,
        "candidate_build": {
            **candidate_run,
            "source_cache": str(full_cache_path),
            "source_cache_sha256": full_cache_sha256,
        },
        "compact": {
            **compact_run,
            "cache": str(compact_path),
            "cache_sha256": compact_sha256,
        },
        "compact_cache": str(compact_path),
        "compact_cache_sha256": compact_sha256,
        "cleanup_enabled": bool(args.cleanup),
        "bytes_deleted": int(deleted_bytes),
        "source_tree_sha256": current_source_tree,
        "git_commit": _git_commit(root),
        "test_accessed": False,
    }
    atomic_write_json(pipeline_path, pipeline)
    return {
        "status": pipeline["status"],
        "fold_id": args.fold_id,
        "records": pipeline["records"],
        "compact_cache": str(compact_path),
        "compact_cache_sha256": compact_sha256,
        "bytes_deleted": int(deleted_bytes),
        "test_accessed": False,
    }


def main() -> None:
    result = build_fold(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
