"""Audit strict Train-OOF and paired Dev caches before D1 model training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import source_tree_sha256
from gmner.data.stage1_selector_oof_cache import (
    validate_selector_dev_payload,
    validate_selector_oof_payload,
    write_json,
)
from scripts.merge_stage1_selector_oof import audit_selector_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-train-records", type=int, default=7000)
    parser.add_argument("--expected-dev-records", type=int, default=1500)
    return parser.parse_args()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _derived(audit: dict) -> dict:
    records = max(int(audit["records"]), 1)
    candidates = max(int(audit["candidates"]), 1)
    formal = max(int(audit["formal_candidates"]), 1)
    nonformal = max(int(audit["nonformal_candidates"]), 1)
    promotable = int(audit["gold_span_nonformal"])
    return {
        "candidates_per_record": audit["candidates"] / records,
        "formal_candidates_per_record": audit["formal_candidates"] / records,
        "nonformal_candidates_per_record": audit["nonformal_candidates"] / records,
        "candidate_positive_rate": audit["gold_span_candidates"] / candidates,
        "formal_gold_rate": audit["gold_span_formal"] / formal,
        "nonformal_gold_rate": promotable / nonformal,
        "promotable_gold_spans": promotable,
        "nonformal_negative_per_promotable": (
            audit["non_gold_nonformal"] / max(promotable, 1)
        ),
        "span_candidate_coverage": audit["span_candidate_coverage"],
        "typed_span_candidate_coverage": audit["typed_span_candidate_coverage"],
        "promoted_top1_type_accuracy": audit["promoted_top1_type_accuracy"],
        "promoted_base_triple_accuracy": audit[
            "promoted_base_triple_accuracy"
        ],
    }


def audit_phase1(
    *,
    train_cache: Path,
    dev_cache: Path,
    output: Path,
    expected_train_records: int,
    expected_dev_records: int,
) -> dict:
    root = Path(__file__).resolve().parents[1]
    train_payload = torch.load(train_cache, map_location="cpu")
    dev_payload = torch.load(dev_cache, map_location="cpu")
    train = validate_selector_oof_payload(
        train_payload,
        expected_num_folds=10,
    )
    dev = validate_selector_dev_payload(dev_payload)
    if train["metadata"]["scope"] != "oof_train":
        raise ValueError("D1 Train cache is not the merged strict OOF cache.")
    if len(train["records"]) != int(expected_train_records):
        raise ValueError("D1 Train cache has the wrong record count.")
    if len(dev["records"]) != int(expected_dev_records):
        raise ValueError("D1 Dev cache has the wrong record count.")
    current_source_tree = source_tree_sha256(root)
    if train["metadata"]["source_tree_sha256"] != current_source_tree:
        raise ValueError("D1 Train cache uses another source/config tree.")
    if dev["metadata"]["source_tree_sha256"] != current_source_tree:
        raise ValueError("D1 Dev cache uses another source/config tree.")
    common_keys = (
        "candidate_config_sha256",
        "formal_source_id",
        "source2id",
        "hidden_size",
    )
    mismatched = [
        key
        for key in common_keys
        if train["metadata"].get(key) != dev["metadata"].get(key)
    ]
    if mismatched:
        raise ValueError(
            f"Train OOF and Dev candidate contracts differ: {mismatched}."
        )
    train_audit = audit_selector_records(
        train["records"],
        dict(train["metadata"]["source2id"]),
    )
    dev_audit = audit_selector_records(
        dev["records"],
        dict(dev["metadata"]["source2id"]),
    )
    train_derived = _derived(train_audit)
    dev_derived = _derived(dev_audit)
    comparisons = {
        key: {
            "train": train_derived[key],
            "dev": dev_derived[key],
            "dev_minus_train": dev_derived[key] - train_derived[key],
        }
        for key in train_derived
        if isinstance(train_derived[key], (int, float))
    }
    supervision_present = {
        "train_promotable_positive": train_audit["gold_span_nonformal"] > 0,
        "train_nonformal_negative": train_audit["non_gold_nonformal"] > 0,
        "dev_promotable_positive": dev_audit["gold_span_nonformal"] > 0,
        "dev_nonformal_negative": dev_audit["non_gold_nonformal"] > 0,
    }
    report = {
        "kind": "stage1_candidate_selector_phase1_audit",
        "status": "VALID_AUDIT",
        "contract_passed": True,
        "selector_training_supervision_present": all(
            supervision_present.values()
        ),
        "supervision_checks": supervision_present,
        "train_protocol": "strict_10fold_stage1_oof",
        "dev_protocol": "paired_full_fit_stage1",
        "candidate_config_sha256": train["metadata"][
            "candidate_config_sha256"
        ],
        "train": {
            "audit": train_audit,
            "derived": train_derived,
        },
        "dev": {
            "audit": dev_audit,
            "derived": dev_derived,
        },
        "train_dev_comparison": comparisons,
        "test_accessed": False,
    }
    write_json(output, report)
    return report


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    train_cache = _resolve(args.train_cache, root)
    dev_cache = _resolve(args.dev_cache, root)
    output = _resolve(args.output, root)
    for path in (train_cache, dev_cache):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = audit_phase1(
        train_cache=train_cache,
        dev_cache=dev_cache,
        output=output,
        expected_train_records=args.expected_train_records,
        expected_dev_records=args.expected_dev_records,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
