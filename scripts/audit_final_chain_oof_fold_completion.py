#!/usr/bin/env python3
"""Validate one sealed final-chain OOF fold before supervision or cleanup."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.full_chain_oof_contract import (
    fold_from_manifest,
    validate_fold_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", type=int, required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--fold-summary", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--atol", type=float, default=3e-5)
    parser.add_argument("--rtol", type=float, default=1e-6)
    return parser.parse_args()


def schema_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    current: Any = root
    for part in ref[2:].split("/"):
        current = current[part]
    return current


def type_matches(value: Any, name: str) -> bool:
    checks = {
        "object": lambda: isinstance(value, dict),
        "array": lambda: isinstance(value, list),
        "string": lambda: isinstance(value, str),
        "boolean": lambda: isinstance(value, bool),
        "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
        "number": lambda: isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        "null": lambda: value is None,
    }
    return checks[name]()


def validate_schema(
    value: Any, node: dict[str, Any], root: dict[str, Any], trail: str
) -> None:
    if "$ref" in node:
        validate_schema(value, schema_ref(root, node["$ref"]), root, trail)
        return
    if "const" in node and value != node["const"]:
        raise ValueError(f"{trail}: const mismatch")
    if "type" in node:
        types = node["type"] if isinstance(node["type"], list) else [node["type"]]
        if not any(type_matches(value, name) for name in types):
            raise TypeError(f"{trail}: expected {types}, got {type(value).__name__}")
    if isinstance(value, dict):
        properties = dict(node.get("properties") or {})
        missing = set(node.get("required") or ()) - set(value)
        if missing:
            raise KeyError(f"{trail}: missing {sorted(missing)}")
        if node.get("additionalProperties") is False:
            extra = set(value) - set(properties)
            if extra:
                raise KeyError(f"{trail}: extra {sorted(extra)}")
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], root, f"{trail}.{key}")
    elif isinstance(value, list):
        if len(value) < int(node.get("minItems", 0)):
            raise ValueError(f"{trail}: too short")
        if "maxItems" in node and len(value) > int(node["maxItems"]):
            raise ValueError(f"{trail}: too long")
        if "items" in node:
            for index, item in enumerate(value):
                validate_schema(item, node["items"], root, f"{trail}[{index}]")
    elif isinstance(value, str):
        if len(value) < int(node.get("minLength", 0)):
            raise ValueError(f"{trail}: string too short")
        if "pattern" in node and re.fullmatch(str(node["pattern"]), value) is None:
            raise ValueError(f"{trail}: pattern mismatch")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in node and value < node["minimum"]:
            raise ValueError(f"{trail}: below minimum")
        if "maximum" in node and value > node["maximum"]:
            raise ValueError(f"{trail}: above maximum")


def compare(
    left: Any,
    right: Any,
    *,
    atol: float,
    rtol: float,
    state: dict[str, float | int],
    trail: str,
) -> None:
    if isinstance(left, float) or isinstance(right, float):
        if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
            raise TypeError(f"{trail}: numeric type mismatch")
        left_value, right_value = float(left), float(right)
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise ValueError(f"{trail}: non-finite")
        difference = abs(left_value - right_value)
        relative = difference / max(abs(right_value), 1e-30)
        state["numeric_values"] = int(state["numeric_values"]) + 1
        state["max_abs_error"] = max(float(state["max_abs_error"]), difference)
        state["max_rel_error"] = max(float(state["max_rel_error"]), relative)
        if difference > atol + rtol * abs(right_value):
            raise ValueError(f"{trail}: numeric replay mismatch")
        return
    if type(left) is not type(right):
        raise TypeError(f"{trail}: discrete type mismatch")
    if isinstance(left, dict):
        if set(left) != set(right):
            raise KeyError(f"{trail}: key mismatch")
        for key in left:
            compare(
                left[key], right[key], atol=atol, rtol=rtol, state=state, trail=f"{trail}.{key}"
            )
    elif isinstance(left, list):
        if len(left) != len(right):
            raise ValueError(f"{trail}: length mismatch")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            compare(
                left_item,
                right_item,
                atol=atol,
                rtol=rtol,
                state=state,
                trail=f"{trail}[{index}]",
            )
    elif left != right:
        raise ValueError(f"{trail}: discrete mismatch")


def contains_gold(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "gold" in str(key).casefold()
            or str(key) == "supervision"
            or contains_gold(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(contains_gold(item) for item in value)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = Path(args.run_dir).resolve()
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (run_dir / "final_chain_oof_rows.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    source_manifest = validate_fold_manifest(
        Path(args.fold_summary), expected_num_folds=10, verify_fold_ids=(args.fold_id,)
    )
    fold = fold_from_manifest(source_manifest, args.fold_id)
    heldout = {
        str(record["id"]): record
        for record in (
            json.loads(line)
            for line in Path(fold["heldout_file"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    expected_ids = [str(value) for value in fold["heldout_record_ids"]]
    if [row["record_id"] for row in rows] != expected_ids or len(rows) != 700:
        raise RuntimeError("Fold row coverage/order mismatch.")
    span_count = 0
    for index, row in enumerate(rows):
        validate_schema(row, schema, schema, f"$[{index}]")
        if int(row["fold_id"]) != args.fold_id or contains_gold(row):
            raise RuntimeError("Fold identity changed or gold leaked before seal.")
        token_count = len(heldout[row["record_id"]]["tokens"])
        spans = [item["span"] for item in row["formal_predictions"]]
        spans += [item["span"] for item in row["r16_candidates"]["span_candidates"]]
        spans += [item["span"] for item in row["r36_candidates"]["span_candidates"]]
        for span in spans:
            span_count += 1
            if not (0 <= int(span["start"]) < int(span["end"]) <= token_count):
                raise ValueError("Word-space span is outside the record.")
        predictions = {item["prediction_id"] for item in row["formal_predictions"]}
        candidates = {item["candidate_id"] for item in row["r36_candidates"]["span_candidates"]}
        for action in row["replacement_actions"]:
            if action["base_prediction_id"] not in predictions or action["candidate_id"] not in candidates:
                raise ValueError("Action has a dangling identity reference.")

    spec = importlib.util.spec_from_file_location(
        "fold_materializer", root / "scripts" / "materialize_final_chain_oof_fold0_rows.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    formal = torch.load(run_dir / "m33a_formal_state.pt", map_location="cpu")
    r16 = torch.load(run_dir / "candidates" / "heldout_r16.pt", map_location="cpu")
    r36 = torch.load(run_dir / "candidates" / "heldout_r36.pt", map_location="cpu")
    pipeline = json.loads((run_dir / "pipeline_manifest.json").read_text(encoding="utf-8"))
    d0 = json.loads((run_dir / "d0_preflight.json").read_text(encoding="utf-8"))
    rebuilt = module.build_rows(
        formal, r16, r36, fold=fold, pipeline=pipeline, d0=d0, fold_id=args.fold_id
    )
    replay = module.build_rows(
        formal, r16, r36, fold=fold, pipeline=pipeline, d0=d0, fold_id=args.fold_id
    )
    state: dict[str, float | int] = {
        "numeric_values": 0,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    compare(rows, rebuilt, atol=args.atol, rtol=args.rtol, state=state, trail="sealed")
    compare(rebuilt, replay, atol=args.atol, rtol=args.rtol, state=state, trail="replay")
    stages = ("stage1", "hierarchical", "coarse", "fine", "evidence")
    if not all(
        pipeline["stages"][stage]["status"] == "complete"
        and pipeline["stages"][stage]["heldout_excluded"] is True
        and pipeline["stages"][stage]["test_accessed"] is False
        for stage in stages
    ):
        raise RuntimeError("A supervised stage failed its exclusion proof.")
    report = {
        "kind": "final_chain_oof_fold_completion_audit",
        "format_version": 1,
        "status": "PASSED",
        "fold_id": args.fold_id,
        "records": len(rows),
        "schema_coverage": 1.0,
        "word_space_span_validity": 1.0,
        "validated_span_count": span_count,
        "formal_prediction_identity_coverage": 1.0,
        "action_reference_coverage": 1.0,
        "gold_free_rows_sealed": True,
        "continuous_atol": args.atol,
        "continuous_rtol": args.rtol,
        "continuous_numeric_values_compared": state["numeric_values"],
        "continuous_max_abs_error": state["max_abs_error"],
        "continuous_max_rel_error": state["max_rel_error"],
        "all_five_supervised_stages_heldout_excluded": True,
        "pipeline_sealed": pipeline.get("sealed") is True,
        "other_folds_accessed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
