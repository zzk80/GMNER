"""Build the paired full-fit Dev cache for the D1 candidate selector."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import source_tree_sha256
from gmner.data.null_release_oof_cache import sha256_file
from gmner.data.stage1_selector_oof_cache import (
    atomic_save_selector_payload,
    build_dev_selector_payload,
    validate_selector_dev_payload,
    write_json,
)
from scripts.merge_stage1_selector_oof import audit_selector_records


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
        "--output",
        default=(
            "knowledge/stage1_candidate_selector_oof/roberta128/"
            "dev_candidates.pt"
        ),
    )
    parser.add_argument("--input-cache", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


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


def _run_candidate_builder(
    *,
    root: Path,
    config: Path,
    checkpoint: Path,
    output: Path,
    device: str,
    batch_size: int,
    log_path: Path,
) -> None:
    command = [
        sys.executable,
        "-u",
        str(root / "scripts" / "build_record_candidate_cache.py"),
        "--config",
        str(config),
        "--checkpoint",
        str(checkpoint),
        "--split",
        "dev",
        "--output",
        str(output),
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
        str(batch_size),
        "--device",
        device,
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("+ " + subprocess.list2cmdline(command) + "\n")
        log.flush()
        subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )


def build_dev_cache(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    current_source_tree = source_tree_sha256(root)
    config_path = _resolve(args.config, root)
    checkpoint_path = _resolve(args.checkpoint, root)
    output_path = _resolve(args.output, root)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if output_path.is_file():
        payload = torch.load(output_path, map_location="cpu")
        validated = validate_selector_dev_payload(payload)
        metadata = validated["metadata"]
        if metadata["source_tree_sha256"] != current_source_tree:
            raise ValueError("Existing Dev cache was built from another source tree.")
        if metadata["stage1_checkpoint_sha256"] != sha256_file(checkpoint_path):
            raise ValueError("Existing Dev cache uses another Stage1 checkpoint.")
        if metadata["stage1_config_sha256"] != sha256_file(config_path):
            raise ValueError("Existing Dev cache uses another Stage1 config.")
        return {
            "status": "already_complete",
            "records": len(validated["records"]),
            "output": str(output_path),
            "output_sha256": sha256_file(output_path),
            "test_accessed": False,
        }

    if args.input_cache:
        full_cache_path = _resolve(args.input_cache, root)
        if not full_cache_path.is_file():
            raise FileNotFoundError(full_cache_path)
        generated = False
    else:
        full_cache_path = (
            output_path.parent / "intermediate" / "dev_r16_full.pt"
        )
        _run_candidate_builder(
            root=root,
            config=config_path,
            checkpoint=checkpoint_path,
            output=full_cache_path,
            device=args.device,
            batch_size=args.batch_size,
            log_path=output_path.parent / "logs" / "dev_candidate_build.log",
        )
        generated = True

    full_payload = torch.load(full_cache_path, map_location="cpu")
    metadata = dict(full_payload.get("metadata") or {})
    if metadata.get("stage1_checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("Dev source cache uses another Stage1 checkpoint.")
    payload = build_dev_selector_payload(
        full_payload,
        source_candidate_cache=str(full_cache_path),
        source_candidate_cache_sha256=sha256_file(full_cache_path),
        stage1_config=str(config_path),
        stage1_config_sha256=sha256_file(config_path),
        git_commit=_git_commit(root),
        source_tree_sha256=current_source_tree,
    )
    atomic_save_selector_payload(output_path, payload)
    reloaded = torch.load(output_path, map_location="cpu")
    validated = validate_selector_dev_payload(reloaded)
    audit = audit_selector_records(
        validated["records"],
        dict(validated["metadata"]["source2id"]),
    )
    summary = {
        "kind": "stage1_candidate_selector_dev_summary",
        "status": "complete",
        "records": len(validated["records"]),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "stage1_checkpoint_sha256": sha256_file(checkpoint_path),
        "test_accessed": False,
        "audit": audit,
    }
    write_json(output_path.with_suffix(".summary.json"), summary)

    if generated and args.cleanup:
        source_summary = full_cache_path.with_suffix(".summary.json")
        if source_summary.is_file():
            source_summary.unlink()
        full_cache_path.unlink()
        validate_selector_dev_payload(torch.load(output_path, map_location="cpu"))
    return summary


def main() -> None:
    result = build_dev_cache(parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
