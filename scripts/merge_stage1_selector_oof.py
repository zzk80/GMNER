"""Merge ten compact D1 fold caches and audit selector supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    fold_from_manifest,
    source_tree_sha256,
    validate_fold_manifest,
)
from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from gmner.data.stage1_selector_oof_cache import (
    STAGE1_SELECTOR_CACHE_KIND,
    STAGE1_SELECTOR_CACHE_VERSION,
    STAGE1_SELECTOR_SCOPE_TRAIN,
    atomic_save_selector_payload,
    selector_record_id,
    validate_selector_oof_payload,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--fold-summary", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _composite_sha256(values: object) -> str:
    encoded = json.dumps(
        values,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _gold_by_span(record: dict) -> dict[tuple[int, int], dict]:
    metadata = dict(record.get("metadata") or {})
    return {
        tuple(int(value) for value in entity["span"]): entity
        for entity in metadata.get("gold_entities") or []
    }


def audit_selector_records(records: list[dict], source2id: dict[str, int]) -> dict:
    source_names = {int(value): str(key) for key, value in source2id.items()}
    counts: Counter[str] = Counter()
    by_source: dict[str, Counter[str]] = {
        name: Counter() for name in source_names.values()
    }

    for record in records:
        span_mask = record["span_mask"].bool()
        formal = record["formal_candidate_mask"].bool() & span_mask
        gold_span = record["gold_span_mask"].bool() & span_mask
        gold_type_available = record["gold_type_mask"].bool().any(dim=-1) & gold_span
        fixed_type = record["fixed_type_ids"].long()
        base_regions = record["base_region_indices"].long()
        spans = record["span_candidates"].long()
        source_ids = record["span_source_ids"].long()
        gold_entities = _gold_by_span(record)

        counts["records"] += 1
        counts["gold_entities"] += len(gold_entities)
        counts["candidates"] += int(span_mask.sum().item())
        counts["formal_candidates"] += int(formal.sum().item())
        counts["nonformal_candidates"] += int((span_mask & ~formal).sum().item())
        counts["gold_span_candidates"] += int(gold_span.sum().item())
        counts["gold_span_formal"] += int((gold_span & formal).sum().item())
        counts["gold_span_nonformal"] += int((gold_span & ~formal).sum().item())
        counts["non_gold_formal"] += int((formal & ~gold_span).sum().item())
        counts["non_gold_nonformal"] += int(
            (span_mask & ~formal & ~gold_span).sum().item()
        )
        counts["typed_span_candidates"] += int(gold_type_available.sum().item())
        counts["typed_span_nonformal"] += int(
            (gold_type_available & ~formal).sum().item()
        )

        for row in torch.nonzero(span_mask, as_tuple=False).squeeze(-1).tolist():
            source_name = source_names.get(int(source_ids[row].item()), "unknown")
            source_counter = by_source.setdefault(source_name, Counter())
            source_counter["candidates"] += 1
            is_gold_span = bool(gold_span[row].item())
            source_counter["gold_span"] += int(is_gold_span)
            if not is_gold_span:
                source_counter["negative"] += 1
                continue
            span = tuple(int(value) for value in spans[row].tolist())
            gold = gold_entities.get(span)
            if gold is None:
                raise ValueError("gold_span_mask has no matching metadata gold entity.")
            top1_type_correct = int(fixed_type[row].item()) == int(gold["type_id"])
            region_correct = int(base_regions[row].item()) in {
                int(value) for value in gold.get("region_positive_indices") or []
            }
            source_counter["top1_type_correct"] += int(top1_type_correct)
            source_counter["base_region_correct"] += int(region_correct)
            source_counter["base_triple_correct"] += int(
                top1_type_correct and region_correct
            )
            counts["top1_type_correct"] += int(top1_type_correct)
            counts["base_region_correct"] += int(region_correct)
            counts["base_triple_correct"] += int(
                top1_type_correct and region_correct
            )
            if not bool(formal[row].item()):
                counts["promote_top1_type_correct"] += int(top1_type_correct)
                counts["promote_base_region_correct"] += int(region_correct)
                counts["promote_base_triple_correct"] += int(
                    top1_type_correct and region_correct
                )

    gold_count = max(counts["gold_entities"], 1)
    promoted = max(counts["gold_span_nonformal"], 1)
    summary = {
        key: int(value)
        for key, value in sorted(counts.items())
    }
    summary.update(
        {
            "span_candidate_coverage": (
                counts["gold_span_candidates"] / gold_count
            ),
            "typed_span_candidate_coverage": (
                counts["typed_span_candidates"] / gold_count
            ),
            "base_triple_candidate_coverage": (
                counts["base_triple_correct"] / gold_count
            ),
            "promoted_top1_type_accuracy": (
                counts["promote_top1_type_correct"] / promoted
            ),
            "promoted_base_region_accuracy": (
                counts["promote_base_region_correct"] / promoted
            ),
            "promoted_base_triple_accuracy": (
                counts["promote_base_triple_correct"] / promoted
            ),
            "candidate_sources": {
                source: {
                    key: int(value)
                    for key, value in sorted(counter.items())
                }
                for source, counter in sorted(by_source.items())
            },
        }
    )
    return summary


def merge_caches(
    inputs: list[Path],
    *,
    fold_summary: Path,
    output: Path,
) -> dict:
    manifest = validate_fold_manifest(fold_summary, expected_num_folds=10)
    root = Path(__file__).resolve().parents[1]
    current_source_tree = source_tree_sha256(root)
    if manifest.get("source_tree_sha256") != current_source_tree:
        raise ValueError(
            "Source/config tree differs from the frozen D1 fold manifest."
        )
    manifest_sha256 = sha256_file(fold_summary)
    if len(inputs) != 10:
        raise ValueError(f"Expected 10 selector fold caches, found {len(inputs)}.")

    fold_payloads: dict[int, dict] = {}
    fold_metadata: dict[int, dict] = {}
    input_by_fold: dict[int, Path] = {}
    for path in inputs:
        payload = torch.load(path, map_location="cpu")
        metadata = dict(payload.get("metadata") or {})
        fold_id = int(metadata.get("fold_id", -1))
        if fold_id in fold_payloads:
            raise ValueError(f"Duplicate selector OOF fold {fold_id}.")
        fold = fold_from_manifest(manifest, fold_id)
        validated = validate_selector_oof_payload(
            payload,
            expected_fold_id=fold_id,
            expected_num_folds=10,
            expected_record_ids=fold["heldout_record_ids"],
        )
        if metadata.get("fold_manifest_sha256") != manifest_sha256:
            raise ValueError(f"Fold {fold_id} references another fold manifest.")
        fold_payloads[fold_id] = payload
        fold_metadata[fold_id] = validated["metadata"]
        input_by_fold[fold_id] = path
    if sorted(fold_payloads) != list(range(10)):
        raise ValueError(f"Selector fold ids are incomplete: {sorted(fold_payloads)}.")

    common_keys = (
        "candidate_config_sha256",
        "formal_source_id",
        "source2id",
        "hidden_size",
        "source_tree_sha256",
    )
    for key in common_keys:
        values = {
            json.dumps(
                fold_metadata[fold_id].get(key),
                ensure_ascii=False,
                sort_keys=True,
            )
            for fold_id in range(10)
        }
        if len(values) != 1:
            raise ValueError(f"Selector OOF folds disagree on {key}.")

    records_by_id: dict[str, dict] = {}
    for fold_id in range(10):
        for record in fold_payloads[fold_id]["records"]:
            record_id = selector_record_id(record)
            if record_id in records_by_id:
                raise ValueError(f"Duplicate merged Train record id: {record_id}.")
            records_by_id[record_id] = record
    expected_ids = [str(value) for value in manifest["record_ids"]]
    if set(records_by_id) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(records_by_id))
        unknown = sorted(set(records_by_id) - set(expected_ids))
        raise ValueError(
            "Merged selector OOF coverage mismatch: "
            f"missing={missing[:5]}, unknown={unknown[:5]}."
        )
    records = [records_by_id[record_id] for record_id in expected_ids]
    stage1_hashes = {
        str(fold_id): fold_metadata[fold_id]["stage1_checkpoint_sha256"]
        for fold_id in range(10)
    }
    data_hashes = {
        str(fold_id): fold_metadata[fold_id]["data_source_sha256"]
        for fold_id in range(10)
    }
    config_hashes = {
        str(fold_id): fold_metadata[fold_id]["stage1_config_sha256"]
        for fold_id in range(10)
    }
    cache_hashes = {
        str(fold_id): sha256_file(input_by_fold[fold_id])
        for fold_id in range(10)
    }
    first = fold_metadata[0]
    if first["source_tree_sha256"] != current_source_tree:
        raise ValueError("Fold caches were built from another source/config tree.")
    audit = audit_selector_records(records, dict(first["source2id"]))
    payload = {
        "metadata": {
            "format_version": STAGE1_SELECTOR_CACHE_VERSION,
            "kind": STAGE1_SELECTOR_CACHE_KIND,
            "scope": STAGE1_SELECTOR_SCOPE_TRAIN,
            "oof": True,
            "test_accessed": False,
            "num_folds": 10,
            "fold_ids": list(range(10)),
            "records": len(records),
            "record_ids": expected_ids,
            "record_ids_sha256": stable_id_digest(expected_ids),
            "hidden_size": int(first["hidden_size"]),
            "formal_source_id": int(first["formal_source_id"]),
            "source2id": dict(first["source2id"]),
            "candidate_config": dict(first["candidate_config"]),
            "candidate_config_sha256": first["candidate_config_sha256"],
            "stage1_checkpoint_sha256": _composite_sha256(stage1_hashes),
            "stage1_checkpoint_sha256s": stage1_hashes,
            "data_source": "strict_10fold_train_oof",
            "data_source_sha256": _composite_sha256(data_hashes),
            "data_source_sha256s": data_hashes,
            "source_candidate_cache_sha256s": cache_hashes,
            "stage1_config_sha256s": config_hashes,
            "fold_manifest": str(fold_summary),
            "fold_manifest_sha256": manifest_sha256,
            "source_tree_sha256": first["source_tree_sha256"],
            "fold_source_tree_sha256s": {
                str(fold_id): fold_metadata[fold_id]["source_tree_sha256"]
                for fold_id in range(10)
            },
            "git_commits": {
                str(fold_id): fold_metadata[fold_id].get("git_commit")
                for fold_id in range(10)
            },
            "reference_fold_proof_sha256s": {
                str(fold_id): fold_metadata[fold_id][
                    "reference_fold_proof_sha256"
                ]
                for fold_id in range(10)
            },
            "audit": audit,
        },
        "records": records,
    }
    validate_selector_oof_payload(
        payload,
        expected_num_folds=10,
        expected_record_ids=expected_ids,
    )
    atomic_save_selector_payload(output, payload)
    reloaded = torch.load(output, map_location="cpu")
    validate_selector_oof_payload(
        reloaded,
        expected_num_folds=10,
        expected_record_ids=expected_ids,
    )
    summary = {
        "kind": "stage1_candidate_selector_oof_train_summary",
        "records": len(records),
        "folds": 10,
        "output": str(output),
        "output_sha256": sha256_file(output),
        "test_accessed": False,
        "audit": audit,
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    input_paths = [_resolve(value, root) for value in args.inputs]
    fold_summary = _resolve(args.fold_summary, root)
    output = _resolve(args.output, root)
    for path in [*input_paths, fold_summary]:
        if not path.is_file():
            raise FileNotFoundError(path)
    summary = merge_caches(
        input_paths,
        fold_summary=fold_summary,
        output=output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
