#!/usr/bin/env python3
"""Audit exact P4.0 formal R16 recovery without labels or Oracle computation."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from gmner.data.null_release_oof_cache import sha256_file
from gmner.data.p4_formal_r16_recovery import (
    DEFAULT_FORMAL_CACHE_PATTERNS,
    P4_FORMAL_RECOVERY_BLOCKED,
    build_blocked_recovery_report,
    discover_formal_cache_candidates,
    formal_cache_expectations,
    hash_recovery_candidates,
    match_exact_formal_caches,
    validate_recovery_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PHASE_A_REPORT = (
    PROJECT_ROOT / "docs" / "experiments" / "p4_0_source_preparation_report.json"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs" / "experiments" / "p4_0_formal_r16_recovery_report.json"
)


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _load_optional_inventory(path: Path | None) -> tuple[dict | None, str | None]:
    if path is None:
        return None, None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("External search inventory must be a JSON object.")
    access = dict(payload.get("access_contract") or {})
    if access.get("calibration_folds_opened") is not False:
        raise PermissionError("External search inventory opened calibration folds.")
    if access.get("dev_accessed") is not False:
        raise PermissionError("External search inventory accessed Dev.")
    if access.get("test_accessed") is not False:
        raise PermissionError("External search inventory accessed Test.")
    return payload, sha256_file(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase-a-report",
        type=Path,
        default=DEFAULT_PHASE_A_REPORT,
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        action="append",
        default=[],
        help="Repeatable direct-file search root. Calibration/Dev/Test paths are skipped.",
    )
    parser.add_argument(
        "--candidate",
        type=Path,
        action="append",
        default=[],
        help="Repeatable explicit candidate file; acceptance still requires exact SHA256.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help="Optional direct-cache filename pattern.",
    )
    parser.add_argument("--external-search-inventory", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    phase_a_path = args.phase_a_report.resolve()
    phase_a = json.loads(phase_a_path.read_text(encoding="utf-8"))
    expectations = formal_cache_expectations(phase_a)
    patterns = tuple(args.pattern) if args.pattern else DEFAULT_FORMAL_CACHE_PATTERNS
    discovery = discover_formal_cache_candidates(
        search_roots=args.search_root,
        explicit_candidates=args.candidate,
        patterns=patterns,
    )
    descriptors = hash_recovery_candidates(discovery["candidate_paths"])
    matches = match_exact_formal_caches(expectations, descriptors)
    missing = [
        item for item in matches if item["status"] != "RECOVERED_EXACT_HASH"
    ]
    if not missing:
        raise SystemExit(
            "All exact formal R16 artifacts were found. Sidecar materialization "
            "requires the separately reviewed completion path; this blocked-only "
            "audit intentionally made no writes."
        )

    inventory, inventory_sha = _load_optional_inventory(
        args.external_search_inventory.resolve()
        if args.external_search_inventory is not None
        else None
    )
    if inventory is not None:
        inventory = {
            "path": _logical_path(args.external_search_inventory),
            "sha256": inventory_sha,
            "content": inventory,
        }
    implementation = {
        "git_head": _git_commit(),
        "contract_path": _logical_path(
            PROJECT_ROOT / "gmner" / "data" / "p4_formal_r16_recovery.py"
        ),
        "contract_sha256": sha256_file(
            PROJECT_ROOT / "gmner" / "data" / "p4_formal_r16_recovery.py"
        ),
        "driver_path": _logical_path(Path(__file__)),
        "driver_sha256": sha256_file(Path(__file__)),
    }
    report = build_blocked_recovery_report(
        expectations=expectations,
        discovery=discovery,
        candidate_descriptors=descriptors,
        matches=matches,
        implementation=implementation,
        phase_a_report_path=_logical_path(phase_a_path),
        phase_a_report_sha256=sha256_file(phase_a_path),
        external_search_inventory=inventory,
    )
    validate_recovery_report(report)
    _atomic_write_json(args.output.resolve(), report)
    print(
        json.dumps(
            {
                "status": P4_FORMAL_RECOVERY_BLOCKED,
                "missing_folds": report["missing_exact_artifact_folds"],
                "candidate_files_hashed": len(descriptors),
                "payloads_loaded": False,
                "sidecars_generated": False,
                "source_manifest_sealed": False,
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
