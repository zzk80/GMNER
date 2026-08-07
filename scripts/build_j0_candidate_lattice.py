#!/usr/bin/env python3
"""Build and seal the J0-A gold-free typed-span lattice."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.j0_candidate_lattice import (
    build_lattice_record,
    canonical_bytes,
    contains_gold_or_supervision,
    finite,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--merge-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    return parser.parse_args()


def reject_dev_test_path(path: Path) -> None:
    lowered = str(path).replace("\\", "/").casefold()
    if "/dev" in lowered or "_dev" in lowered or "/test" in lowered or "_test" in lowered:
        raise PermissionError(f"Dev/Test path is forbidden for J0-A: {path}")


def validate_authorization(payload: dict[str, Any]) -> None:
    if payload.get("kind") != "j0_a_candidate_lattice_oracle_preregistration":
        raise ValueError("Unexpected J0 authorization kind.")
    if payload.get("status") != "AUTHORIZED_J0_A_ONLY":
        raise PermissionError("J0-A is not authorized.")
    access = payload.get("authorization", {})
    if access.get("j0_a_gold_free_build") is not True:
        raise PermissionError("Gold-free lattice build is not authorized.")
    for key in (
        "j0_b_latent_rematerialization",
        "j1_training",
        "j2_visual",
        "j3_structured_decoder",
        "dev_access",
        "test_access",
    ):
        if access.get(key) is not False:
            raise PermissionError(f"J0 lock is not active: {key}")


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    args = parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "authorization": args.authorization,
            "rows": args.rows,
            "merge_manifest": args.merge_manifest,
            "output": args.output,
            "manifest_output": args.manifest_output,
        }.items()
    }
    for path in paths.values():
        reject_dev_test_path(path)
    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    validate_authorization(authorization)
    inputs = authorization["inputs"]
    if sha256_file(paths["rows"]) != inputs["gold_free_rows_sha256"]:
        raise ValueError("Sealed OOF row SHA256 differs from preregistration.")
    if sha256_file(paths["merge_manifest"]) != inputs["merge_manifest_sha256"]:
        raise ValueError("OOF merge manifest SHA256 differs from preregistration.")
    merge = json.loads(paths["merge_manifest"].read_text(encoding="utf-8"))
    if (
        merge.get("status") != "PASSED"
        or int(merge.get("records", -1)) != 7000
        or merge.get("dev_accessed") is not False
        or merge.get("test_accessed") is not False
        or not all(bool(value) for value in merge.get("checks", {}).values())
    ):
        raise RuntimeError("The final-chain OOF mother-set Gate is not clean.")

    source_sha_before = sha256_file(paths["rows"])
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["output"].with_suffix(paths["output"].suffix + ".tmp")
    record_ids: list[str] = []
    folds: dict[int, int] = {}
    counts = {
        "formal_predictions": 0,
        "replacement_groups": 0,
        "addition_groups": 0,
        "raw_alternatives": 0,
        "deduplicated_alternatives": 0,
    }
    source_audit = {
        "r16_r36_mismatch_records": 0,
        "r16_only_candidates": 0,
        "r36_only_candidates": 0,
    }
    finite_values = 0
    discrete_digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        for source in iter_jsonl(paths["rows"]):
            lattice = build_lattice_record(source)
            if contains_gold_or_supervision(lattice):
                raise RuntimeError("Gold/supervision leaked into the sealed lattice.")
            line = canonical_bytes(lattice) + b"\n"
            stream.write(line)
            record_ids.append(str(lattice["record_id"]))
            fold_id = int(lattice["fold_id"])
            folds[fold_id] = folds.get(fold_id, 0) + 1
            for key in counts:
                counts[key] += int(lattice["counts"][key])
            audit = lattice["candidate_source_audit"]
            source_audit["r16_r36_mismatch_records"] += int(
                not audit["r16_r36_semantic_match"]
            )
            source_audit["r16_only_candidates"] += int(
                audit["r16_only_candidates"]
            )
            source_audit["r36_only_candidates"] += int(
                audit["r36_only_candidates"]
            )
            finite_values += finite(lattice)
            discrete_digest.update(
                canonical_bytes(
                    {
                        "record_id": lattice["record_id"],
                        "groups": [
                            {
                                "group_id": group["group_id"],
                                "control": group["control"]["hypothesis_id"],
                                "alternatives": [
                                    item["hypothesis_id"]
                                    for item in group["alternatives"]
                                ],
                            }
                            for group in lattice["groups"]
                        ],
                    }
                )
            )
    temporary.replace(paths["output"])

    replay_file_digest = hashlib.sha256()
    replay_discrete_digest = hashlib.sha256()
    replay_records = 0
    for source in iter_jsonl(paths["rows"]):
        lattice = build_lattice_record(source)
        replay_file_digest.update(canonical_bytes(lattice) + b"\n")
        replay_discrete_digest.update(
            canonical_bytes(
                {
                    "record_id": lattice["record_id"],
                    "groups": [
                        {
                            "group_id": group["group_id"],
                            "control": group["control"]["hypothesis_id"],
                            "alternatives": [
                                item["hypothesis_id"]
                                for item in group["alternatives"]
                            ],
                        }
                        for group in lattice["groups"]
                    ],
                }
            )
        )
        replay_records += 1

    source_sha_after = sha256_file(paths["rows"])
    lattice_sha = sha256_file(paths["output"])
    checks = {
        "records_exact_7000": len(record_ids) == replay_records == 7000,
        "record_ids_unique_7000": len(set(record_ids)) == 7000,
        "folds_0_9_exact_700_each": folds == {fold: 700 for fold in range(10)},
        "source_rows_unchanged": source_sha_before == source_sha_after,
        "double_build_file_digest_exact": lattice_sha == replay_file_digest.hexdigest(),
        "double_build_discrete_digest_exact": (
            discrete_digest.hexdigest() == replay_discrete_digest.hexdigest()
        ),
        "finite_values": finite_values > 0,
        "dev_test_accessed_false": True,
        "gold_supervision_absent": True,
    }
    if not all(checks.values()):
        raise RuntimeError(f"J0 lattice seal failed: {checks}")
    report = {
        "kind": "j0_gold_free_candidate_lattice_manifest",
        "format_version": 1,
        "status": "SEALED",
        "authorization_sha256": sha256_file(paths["authorization"]),
        "implementation": {
            "contract_sha256": sha256_file(
                Path(__file__).resolve().parents[1]
                / "gmner"
                / "data"
                / "j0_candidate_lattice.py"
            ),
            "builder_sha256": sha256_file(Path(__file__).resolve()),
        },
        "source": {
            "rows_sha256_before": source_sha_before,
            "rows_sha256_after": source_sha_after,
            "merge_manifest_sha256": sha256_file(paths["merge_manifest"]),
        },
        "lattice": {
            "path": str(paths["output"]),
            "sha256": lattice_sha,
            "bytes": paths["output"].stat().st_size,
            "records": len(record_ids),
            "record_ids_sha256": stable_id_digest(record_ids),
            "discrete_digest_sha256": discrete_digest.hexdigest(),
            **counts,
        },
        "fold_record_counts": {str(key): value for key, value in sorted(folds.items())},
        "candidate_source_audit": {
            **source_audit,
            "selected_namespace": "sealed_r36_span_candidates",
            "union_used": False,
        },
        "finite_values_checked": finite_values,
        "checks": checks,
        "supervision_attached": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    paths["manifest_output"].parent.mkdir(parents=True, exist_ok=True)
    paths["manifest_output"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
