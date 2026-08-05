#!/usr/bin/env python3
"""Audit historical OOF source capabilities without loading model/data payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


STATUSES = {
    "VALID",
    "INCOMPLETE",
    "SEMANTICALLY_INVALID",
    "PROVENANCE_INVALID",
    "MISSING",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="docs/experiments/final_chain_oof_source_registry.json",
    )
    parser.add_argument(
        "--schema",
        default="docs/experiments/final_chain_oof_minimum_row_schema.json",
    )
    parser.add_argument(
        "--output-json",
        default="docs/experiments/final_chain_oof_source_inventory.json",
    )
    parser.add_argument(
        "--output-md",
        default="docs/experiments/final_chain_oof_source_inventory.md",
    )
    return parser.parse_args()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def derive_status(source: dict[str, Any], required: list[str]) -> tuple[str, list[str]]:
    capabilities = dict(source.get("capabilities") or {})
    missing = [name for name in required if capabilities.get(name) is not True]
    artifact_state = str(source.get("artifact_state", ""))
    if artifact_state == "NOT_FOUND":
        return "MISSING", missing
    if source.get("provenance_valid") is not True:
        return "PROVENANCE_INVALID", missing
    if source.get("semantic_valid") is not True:
        return "SEMANTICALLY_INVALID", missing
    if missing:
        return "INCOMPLETE", missing
    return "VALID", missing


def build_inventory(registry: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    required = list(schema.get("x_final_chain_oof_required_capabilities") or [])
    if not required or len(required) != len(set(required)):
        raise ValueError("Schema required capabilities are missing or duplicated.")
    access = dict(registry.get("access") or {})
    if any(access.values()):
        raise ValueError("Source inventory must remain payload-free and split-locked.")

    rows: list[dict[str, Any]] = []
    for source in registry.get("sources") or []:
        status, missing = derive_status(source, required)
        if status not in STATUSES:
            raise ValueError(f"Unsupported source status: {status}")
        if status != source.get("declared_status"):
            raise ValueError(
                f"Source {source.get('source_id')} declared {source.get('declared_status')} "
                f"but derives as {status}."
            )
        rows.append(
            {
                "source_id": source["source_id"],
                "description": source["description"],
                "status": status,
                "artifact_state": source["artifact_state"],
                "provenance_valid": bool(source.get("provenance_valid")),
                "semantic_valid": bool(source.get("semantic_valid")),
                "available_capabilities": sorted(
                    name
                    for name in required
                    if (source.get("capabilities") or {}).get(name) is True
                ),
                "missing_capabilities": missing,
                "failed_folds": list(source.get("failed_folds") or []),
                "evidence": list(source.get("evidence") or []),
            }
        )

    counts = Counter(row["status"] for row in rows)
    valid = [row["source_id"] for row in rows if row["status"] == "VALID"]
    return {
        "kind": "final_chain_oof_source_inventory",
        "format_version": 1,
        "scope": "historical_train_oof_sources_only",
        "historical_evidence_commit": registry.get("historical_evidence_commit"),
        "status": "READY_FOR_SINGLE_FOLD_DRY_RUN" if valid else "BLOCKED_NO_VALID_SOURCE",
        "required_capabilities": required,
        "sources": rows,
        "status_counts": {name: counts.get(name, 0) for name in sorted(STATUSES)},
        "valid_sources": valid,
        "source_gate_passed": bool(valid),
        "next_authorized_step": "single_fold_dry_run" if valid else None,
        "next_required_decision": (
            None if valid else "explicitly_authorize_new_single_fold_dry_run"
        ),
        "registry_sha256": canonical_json_sha256(registry),
        "schema_sha256": canonical_json_sha256(schema),
        "access": {
            "payloads_deserialized": False,
            "training_data_loaded": False,
            "oracle_computed": False,
            "folds_8_9_accessed": False,
            "dev_accessed": False,
            "test_accessed": False,
            "b1_a1_training_run": False,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Final-chain OOF Source Inventory",
        "",
        f"**Status:** `{report['status']}`",
        "",
        "This inventory is metadata-only. It did not deserialize model, cache,",
        "dataset, Dev, or Test payloads and did not compute Oracle labels.",
        "",
        "| Source | Status | Key blocker |",
        "| --- | --- | --- |",
    ]
    for source in report["sources"]:
        if source["status"] == "SEMANTICALLY_INVALID":
            blocker = "semantic mismatch in folds " + ", ".join(
                str(value) for value in source["failed_folds"]
            )
        elif source["status"] == "MISSING":
            blocker = "artifact set not found"
        else:
            blocker = ", ".join(source["missing_capabilities"][:4])
            if len(source["missing_capabilities"]) > 4:
                blocker += ", ..."
        lines.append(f"| `{source['source_id']}` | `{source['status']}` | {blocker} |")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "No historical source satisfies the frozen minimum row contract.",
            "B1/A1 population materialization and training remain unauthorized.",
            "The only admissible next step is a newly generated single-fold",
            "full-chain OOF dry run under the frozen protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    report = build_inventory(registry, schema)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_bytes(
        (json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
            "utf-8"
        )
    )
    output_md.write_bytes(render_markdown(report).encode("utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
