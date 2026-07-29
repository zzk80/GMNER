#!/usr/bin/env python3
"""Run the read-only P4-R0-A checkpoint replay feasibility audit."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gmner.data.full_chain_oof_contract import atomic_write_json
from gmner.data.p4_r0_replay_contract import (
    P4_DEVELOPMENT_FOLDS,
    audit_fold_json_provenance,
    build_r0_a_report,
    discover_named_artifacts,
    external_available_artifacts,
    find_source_tree_commits,
    hash_named_artifacts,
    match_expectations_by_sha256,
    sha256_file,
    validate_external_inventory,
    validate_r0_a_report,
    validate_r0_preregistration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREREGISTRATION = (
    PROJECT_ROOT
    / "docs"
    / "experiments"
    / "p4_r0_full_chain_oof_r16_regeneration_preregistration.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=DEFAULT_PREREGISTRATION,
    )
    parser.add_argument(
        "--full-chain-root",
        type=Path,
        default=PROJECT_ROOT / "knowledge/null_release_oof/roberta128",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        action="append",
        default=[],
        help="Read-only local root searched for exact checkpoint basenames.",
    )
    parser.add_argument("--external-inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def main() -> None:
    args = parse_args()
    preregistration = _read_json(args.preregistration.resolve())
    external_inventory = _read_json(args.external_inventory.resolve())
    validate_r0_preregistration(preregistration)
    validate_external_inventory(external_inventory)

    fold_reports = [
        audit_fold_json_provenance(
            args.full_chain_root.resolve() / f"fold{fold_id}",
            fold_id=fold_id,
        )
        for fold_id in P4_DEVELOPMENT_FOLDS
    ]
    checkpoint_expectations = [
        item
        for report in fold_reports
        for item in report["checkpoint_expectations"]
    ]
    source_expectations = [
        item
        for report in fold_reports
        for item in report["fold_source_expectations"]
    ]

    checkpoint_names = {item["basename"] for item in checkpoint_expectations}
    discovery = discover_named_artifacts(
        args.artifact_root,
        basenames=checkpoint_names,
    )
    local_available = hash_named_artifacts(discovery["paths"])
    external_available = external_available_artifacts(external_inventory)
    available = local_available + external_available

    source_tree_hashes = {
        report["source_tree_sha256"] for report in fold_reports
    }
    if len(source_tree_hashes) != 1:
        raise ValueError("Folds 0-7 reference different archived source trees.")
    source_tree = find_source_tree_commits(
        PROJECT_ROOT,
        expected_sha256=next(iter(source_tree_hashes)),
    )

    implementation = {
        "git_head": _git_head(),
        "contract_path": (
            PROJECT_ROOT / "gmner/data/p4_r0_replay_contract.py"
        ).relative_to(PROJECT_ROOT).as_posix(),
        "contract_sha256": sha256_file(
            PROJECT_ROOT / "gmner/data/p4_r0_replay_contract.py"
        ),
        "driver_path": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "preregistration_path": args.preregistration.resolve()
        .relative_to(PROJECT_ROOT)
        .as_posix(),
        "preregistration_sha256": sha256_file(args.preregistration.resolve()),
        "external_inventory_path": args.external_inventory.resolve()
        .relative_to(PROJECT_ROOT)
        .as_posix(),
        "external_inventory_sha256": sha256_file(
            args.external_inventory.resolve()
        ),
    }
    report = build_r0_a_report(
        preregistration=preregistration,
        fold_reports=fold_reports,
        checkpoint_matches=match_expectations_by_sha256(
            checkpoint_expectations,
            available,
        ),
        source_matches=match_expectations_by_sha256(
            source_expectations,
            available,
        ),
        source_tree=source_tree,
        external_inventory={
            **external_inventory,
            "local_discovery": {
                "roots": [
                    path.resolve().as_posix() for path in args.artifact_root
                ],
                "candidate_files_hashed": len(local_available),
                "locked_directories_skipped": [
                    path.as_posix()
                    for path in discovery["locked_directories_skipped"]
                ],
            },
        },
        implementation=implementation,
    )
    validate_r0_a_report(report)
    atomic_write_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
