"""Contracts for the one-time final FMNERG subtype Test evaluation."""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml

from .io import resolve_path, sha256_file


FINAL_TEST_PROTOCOL_KIND = "fmnerg_subtype_encoder_final_test_protocol"
FINAL_TEST_PROTOCOL_VERSION = 1
FINAL_TEST_SEEDS = (41, 42, 43)
FINAL_TEST_SCOPE = "all"
FINAL_TEST_REPORT = "mean_std"


def load_final_test_protocol(
    path: str | Path,
    root: Path,
) -> dict[str, Any]:
    protocol_path = resolve_path(path, root)
    raw = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    if raw.get("kind") != FINAL_TEST_PROTOCOL_KIND:
        raise ValueError("Not an FMNERG subtype final-Test protocol.")
    if int(raw.get("format_version", -1)) != FINAL_TEST_PROTOCOL_VERSION:
        raise ValueError("Unsupported FMNERG subtype final-Test protocol.")
    method = dict(raw.get("method") or {})
    if method.get("encoder_scope") != FINAL_TEST_SCOPE:
        raise ValueError("Final Test is locked to the full encoder scope.")
    if tuple(map(int, method.get("seeds") or [])) != FINAL_TEST_SEEDS:
        raise ValueError("Final Test seeds must be exactly 41, 42, and 43.")
    if method.get("selection_source") != "dev":
        raise ValueError("Final Test method must be selected on Dev.")
    if method.get("report") != FINAL_TEST_REPORT:
        raise ValueError("Final Test must report mean and standard deviation.")
    if method.get("select_best_seed_on_test") is not False:
        raise ValueError("Selecting a seed on Test is forbidden.")
    checkpoints = list(raw.get("checkpoints") or [])
    if tuple(int(item["seed"]) for item in checkpoints) != FINAL_TEST_SEEDS:
        raise ValueError("Final Test checkpoints are not in fixed seed order.")
    for item in checkpoints:
        digest = str(item.get("sha256", ""))
        if len(digest) != 64:
            raise ValueError("Every final checkpoint requires a SHA-256.")
    for section in ("artifacts", "test_data"):
        values = dict(raw.get(section) or {})
        if not values:
            raise ValueError(f"Final Test protocol requires {section}.")
        for name, item in values.items():
            if not item.get("path") or len(str(item.get("sha256", ""))) != 64:
                raise ValueError(
                    f"Final Test artifact {section}.{name} is incomplete."
                )
    expected = dict(raw.get("expected_test_main_chain") or {})
    for key in ("coarse_mner_f1", "eeg_f1", "gmner_f1", "tolerance"):
        if key not in expected:
            raise ValueError(f"Final Test expected metric {key} is missing.")
    if float(expected["tolerance"]) < 0:
        raise ValueError("Final Test metric tolerance must be non-negative.")
    output_dir = raw.get("output_dir")
    if not output_dir:
        raise ValueError("Final Test output_dir is required.")
    return {
        **raw,
        "_protocol_path": str(protocol_path),
        "_protocol_sha256": sha256_file(protocol_path),
    }


def validate_artifact(
    specification: dict[str, Any],
    root: Path,
) -> Path:
    path = resolve_path(specification["path"], root)
    if not path.is_file():
        raise FileNotFoundError(f"Final Test artifact is missing: {path}")
    actual = sha256_file(path)
    expected = str(specification["sha256"])
    if actual != expected:
        raise ValueError(
            f"Final Test artifact hash changed: {path}; "
            f"expected={expected}, actual={actual}."
        )
    return path


def validate_dev_acceptance(
    summary_path: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = dict(summary.get("metadata") or {})
    if metadata.get("test_accessed") is not False:
        raise ValueError("Dev ablation summary already accessed Test.")
    if summary.get("best_scope_by_mean_dev_fmnerg") != FINAL_TEST_SCOPE:
        raise ValueError("Full encoder was not the Dev-selected scope.")
    comparison = dict(summary.get("comparison", {}).get(FINAL_TEST_SCOPE) or {})
    if comparison.get("accepted") is not True:
        raise ValueError("Full encoder did not pass the Dev acceptance gate.")
    if tuple(map(int, metadata.get("seeds") or [])) != FINAL_TEST_SEEDS:
        raise ValueError("Dev ablation seeds differ from final Test seeds.")
    return summary


def aggregate_seed_metrics(
    rows: list[dict[str, float]],
    metric_names: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    if len(rows) != len(FINAL_TEST_SEEDS):
        raise ValueError("Final Test aggregation requires exactly three seeds.")
    return {
        metric: {
            "mean": statistics.mean(float(row[metric]) for row in rows),
            "std": statistics.pstdev(float(row[metric]) for row in rows),
        }
        for metric in metric_names
    }
