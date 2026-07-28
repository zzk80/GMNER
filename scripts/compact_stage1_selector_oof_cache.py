"""Compact one held-out Stage1 candidate cache for the D1 selector."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    fold_from_manifest,
    source_tree_sha256,
    validate_fold_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file
from gmner.data.stage1_selector_oof_cache import (
    atomic_save_selector_payload,
    build_fold_selector_payload,
    validate_selector_oof_payload,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fold-summary", required=True)
    parser.add_argument("--fold-id", required=True, type=int)
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--reference-fold-proof", required=True)
    parser.add_argument(
        "--require-reference-file-hashes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require regenerated fold JSONL bytes to match the archived Linux proof.",
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


def _validate_reference_proof(
    proof_path: Path,
    *,
    fold: dict,
    fold_id: int,
    require_file_hashes: bool,
) -> dict:
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    if int(proof.get("fold_id", -1)) != int(fold_id):
        raise ValueError("Archived fold proof has the wrong fold id.")
    if int(proof.get("num_folds", -1)) != 10:
        raise ValueError("Archived fold proof is not a ten-fold proof.")
    if proof.get("excluded_heldout") is not True:
        raise ValueError("Archived fold proof does not assert heldout exclusion.")
    if [str(value) for value in proof.get("training_record_ids") or []] != [
        str(value) for value in fold["train_record_ids"]
    ]:
        raise ValueError("Regenerated training IDs differ from the archived fold proof.")
    if [str(value) for value in proof.get("heldout_record_ids") or []] != [
        str(value) for value in fold["heldout_record_ids"]
    ]:
        raise ValueError("Regenerated heldout IDs differ from the archived fold proof.")
    if require_file_hashes:
        if proof.get("train_file_sha256") != fold.get("train_file_sha256"):
            raise ValueError("Regenerated train file differs from the archived proof.")
        if proof.get("heldout_file_sha256") != fold.get("heldout_file_sha256"):
            raise ValueError("Regenerated heldout file differs from the archived proof.")
    return proof


def compact(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    input_path = _resolve(args.input_cache, root)
    output_path = _resolve(args.output, root)
    fold_summary_path = _resolve(args.fold_summary, root)
    stage1_config_path = _resolve(args.stage1_config, root)
    proof_path = _resolve(args.reference_fold_proof, root)
    for path in (
        input_path,
        fold_summary_path,
        stage1_config_path,
        proof_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = validate_fold_manifest(
        fold_summary_path,
        expected_num_folds=10,
    )
    current_source_tree = source_tree_sha256(root)
    if manifest.get("source_tree_sha256") != current_source_tree:
        raise ValueError(
            "Source/config tree differs from the frozen D1 fold manifest."
        )
    fold = fold_from_manifest(manifest, args.fold_id)
    _validate_reference_proof(
        proof_path,
        fold=fold,
        fold_id=args.fold_id,
        require_file_hashes=args.require_reference_file_hashes,
    )

    candidate_payload = torch.load(input_path, map_location="cpu")
    candidate_metadata = dict(candidate_payload.get("metadata") or {})
    if candidate_metadata.get("data_source_sha256") != fold["heldout_file_sha256"]:
        raise ValueError("Candidate cache was not generated from this heldout fold.")
    candidate_ids = [
        str(dict(record.get("metadata") or {}).get("record_id", ""))
        for record in candidate_payload.get("records") or []
    ]
    if candidate_ids != [str(value) for value in fold["heldout_record_ids"]]:
        raise ValueError("Candidate cache record order differs from the heldout fold.")

    payload = build_fold_selector_payload(
        candidate_payload,
        fold_id=args.fold_id,
        num_folds=10,
        source_candidate_cache=str(input_path),
        source_candidate_cache_sha256=sha256_file(input_path),
        stage1_config=str(stage1_config_path),
        stage1_config_sha256=sha256_file(stage1_config_path),
        fold_manifest=str(fold_summary_path),
        fold_manifest_sha256=sha256_file(fold_summary_path),
        reference_fold_proof=str(proof_path),
        reference_fold_proof_sha256=sha256_file(proof_path),
        git_commit=_git_commit(root),
        source_tree_sha256=current_source_tree,
    )
    atomic_save_selector_payload(output_path, payload)
    reloaded = torch.load(output_path, map_location="cpu")
    validated = validate_selector_oof_payload(
        reloaded,
        expected_fold_id=args.fold_id,
        expected_num_folds=10,
        expected_record_ids=fold["heldout_record_ids"],
    )
    summary = {
        "kind": "stage1_candidate_selector_oof_fold_summary",
        "fold_id": int(args.fold_id),
        "records": len(validated["records"]),
        "candidates": sum(
            int(record["span_mask"].sum().item())
            for record in validated["records"]
        ),
        "formal_candidates": sum(
            int((record["span_mask"] & record["formal_candidate_mask"]).sum().item())
            for record in validated["records"]
        ),
        "nonformal_candidates": sum(
            int(
                (
                    record["span_mask"]
                    & ~record["formal_candidate_mask"]
                ).sum().item()
            )
            for record in validated["records"]
        ),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "test_accessed": False,
    }
    write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def main() -> None:
    summary = compact(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
