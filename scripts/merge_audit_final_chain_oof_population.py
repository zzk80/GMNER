#!/usr/bin/env python3
"""Merge and audit the sealed ten-fold final-chain OOF population."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from statistics import mean, median
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.full_chain_oof_contract import validate_fold_manifest


STAGES = ("stage1", "hierarchical", "coarse", "fine", "evidence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        default="docs/experiments/final_chain_oof_ten_fold_merge_authorization.json",
    )
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def finite(value: Any, trail: str) -> int:
    count = 0
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {trail}.")
        return 1
    if isinstance(value, dict):
        for key, item in value.items():
            count += finite(item, f"{trail}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            count += finite(item, f"{trail}[{index}]")
    return count


def contains_gold(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            "gold" in str(key).casefold()
            or str(key) == "supervision"
            or contains_gold(item)
            for key, item in value.items()
        )
    return isinstance(value, list) and any(contains_gold(item) for item in value)


def distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": median(ordered),
        "mean": mean(ordered),
        "max": ordered[-1],
    }


def read_raw_jsonl(path: Path) -> list[tuple[bytes, dict[str, Any]]]:
    output: list[tuple[bytes, dict[str, Any]]] = []
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            stripped = raw.rstrip(b"\r\n")
            if not stripped:
                continue
            try:
                value = json.loads(stripped.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from error
            output.append((stripped, value))
    return output


def descriptor(path: Path) -> dict[str, Any]:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_concat(output: Path, sources: list[list[tuple[bytes, dict[str, Any]]]]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        for source in sources:
            for raw, _ in source:
                stream.write(raw)
                stream.write(b"\n")
    temporary.replace(output)


def fold_directory(fold0_root: Path, population_root: Path, fold_id: int) -> Path:
    return fold0_root if fold_id == 0 else population_root / f"fold{fold_id}"


def validate_authorization(payload: dict[str, Any]) -> None:
    if (
        payload.get("kind") != "final_chain_oof_ten_fold_merge_authorization"
        or payload.get("status") != "AUTHORIZED"
        or tuple(payload.get("input_folds") or ()) != tuple(range(10))
    ):
        raise PermissionError("Ten-fold merge authorization is invalid.")
    forbidden = dict(payload.get("forbidden") or {})
    required = (
        "b1_a1_training",
        "feature_selection",
        "auroc_computation",
        "threshold_selection",
        "calibration",
        "dev_file_access",
        "test_access",
    )
    if any(forbidden.get(key) is not True for key in required):
        raise PermissionError("A merge-stage method/access lock is disabled.")


def completion_heldout_excluded(payload: dict[str, Any], fold_id: int) -> bool:
    if fold_id == 0:
        return (
            payload.get(
                "all_five_supervised_stages_complete_and_heldout_excluded"
            )
            is True
        )
    return payload.get("all_five_supervised_stages_heldout_excluded") is True


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_authorization(authorization)
    inputs = dict(authorization["input_contract"])
    outputs = dict(authorization["output_contract"])
    gate_contract = dict(authorization["distribution_gate"])
    fold0_root = resolve(root, inputs["fold0_root"])
    population_root = resolve(root, inputs["population_root"])
    manifest_path = resolve(root, inputs["fold_manifest"])
    schema_path = resolve(root, inputs["schema"])
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    completion_spec = importlib.util.spec_from_file_location(
        "final_chain_completion_contract",
        root / "scripts" / "audit_final_chain_oof_fold_completion.py",
    )
    completion_module = importlib.util.module_from_spec(completion_spec)
    assert completion_spec.loader is not None
    completion_spec.loader.exec_module(completion_module)
    output_root = resolve(root, outputs["root"])
    merged_rows_path = output_root / outputs["gold_free_rows"]
    merged_sidecar_path = output_root / outputs["supervision_sidecar"]
    merge_manifest_path = output_root / outputs["merge_manifest"]
    distribution_path = output_root / outputs["distribution_gate"]
    markdown_path = output_root / outputs["distribution_report"]

    manifest = validate_fold_manifest(manifest_path, expected_num_folds=10)
    expected_all_ids = {str(value) for value in manifest["record_ids"]}
    all_ids: list[str] = []
    row_sources: list[list[tuple[bytes, dict[str, Any]]]] = []
    sidecar_sources: list[list[tuple[bytes, dict[str, Any]]]] = []
    fold_reports: list[dict[str, Any]] = []
    b1_confusion: Counter[str] = Counter()
    a1_protected: Counter[str] = Counter()
    a1_metric: Counter[str] = Counter()
    source_labels: dict[str, Counter[str]] = defaultdict(Counter)
    source_scores: dict[str, list[float]] = defaultdict(list)
    label_scores: dict[str, list[float]] = defaultdict(list)
    label_margins: dict[str, list[float]] = defaultdict(list)
    actions_per_prediction: list[int] = []
    finite_values = 0
    prediction_ids: set[str] = set()
    action_ids: set[str] = set()
    resource_folds: list[dict[str, Any]] = []

    for fold_id in range(10):
        directory = fold_directory(fold0_root, population_root, fold_id)
        rows_path = directory / "final_chain_oof_rows.jsonl"
        sidecar_path = directory / f"fold{fold_id}_supervision.jsonl"
        materialization_path = directory / f"fold{fold_id}_materialization_report.json"
        completion_path = directory / f"fold{fold_id}_completion_audit.json"
        supervision_path = directory / f"fold{fold_id}_supervision_audit.json"
        pipeline_path = directory / "pipeline_manifest.json"
        required_paths = (
            rows_path,
            sidecar_path,
            materialization_path,
            completion_path,
            supervision_path,
            pipeline_path,
        )
        missing = [str(path) for path in required_paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Fold {fold_id} is incomplete: {missing}")
        materialization = json.loads(materialization_path.read_text(encoding="utf-8"))
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        supervision = json.loads(supervision_path.read_text(encoding="utf-8"))
        pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
        if (
            materialization.get("status") != "PASSED"
            or int(materialization.get("records", -1)) != 700
            or materialization.get("double_run_formal_digest_exact") is not True
            or materialization.get("double_run_action_digest_exact") is not True
            or materialization.get("rows_sha256") != sha256_file(rows_path)
        ):
            raise RuntimeError(f"Fold {fold_id} materialization Gate failed.")
        if (
            completion.get("status") != "PASSED"
            or completion.get("schema_coverage") != 1.0
            or completion.get("word_space_span_validity") != 1.0
            or completion.get("formal_prediction_identity_coverage") != 1.0
            or completion.get("action_reference_coverage") != 1.0
            or not completion_heldout_excluded(completion, fold_id)
            or completion.get("pipeline_sealed") is not True
            or completion.get("dev_accessed") is not False
            or completion.get("test_accessed") is not False
        ):
            raise RuntimeError(f"Fold {fold_id} completion Gate failed.")
        if (
            supervision.get("sealed_rows_unchanged") is not True
            or supervision.get("sealed_rows_sha256_before") != sha256_file(rows_path)
            or supervision.get("sealed_rows_sha256_after") != sha256_file(rows_path)
            or supervision.get("sidecar_sha256") != sha256_file(sidecar_path)
            or supervision.get("sealed_discrete_replay_sha256")
            != materialization.get("discrete_replay_sha256")
            or supervision.get("training_run") is not False
            or supervision.get("auroc_computed") is not False
            or supervision.get("threshold_selected") is not False
            or supervision.get("dev_accessed") is not False
            or supervision.get("test_accessed") is not False
        ):
            raise RuntimeError(f"Fold {fold_id} supervision Gate failed.")
        if pipeline.get("sealed") is not True or pipeline.get("test_accessed") is not False:
            raise RuntimeError(f"Fold {fold_id} pipeline is not sealed.")
        for stage in STAGES:
            state = dict(pipeline.get("stages", {}).get(stage) or {})
            if (
                state.get("status") != "complete"
                or state.get("heldout_excluded") is not True
                or state.get("test_accessed") is not False
            ):
                raise RuntimeError(f"Fold {fold_id} stage {stage} failed exclusion.")

        rows = read_raw_jsonl(rows_path)
        sidecars = read_raw_jsonl(sidecar_path)
        if len(rows) != 700 or len(sidecars) != 700:
            raise RuntimeError(f"Fold {fold_id} row count is not 700.")
        expected_ids = [
            str(value)
            for value in next(
                item for item in manifest["folds"] if int(item["fold"]) == fold_id
            )["heldout_record_ids"]
        ]
        row_ids = [str(value["record_id"]) for _, value in rows]
        sidecar_ids = [str(value["record_id"]) for _, value in sidecars]
        if row_ids != expected_ids or sidecar_ids != expected_ids:
            raise RuntimeError(f"Fold {fold_id} held-out row order changed.")
        if set(all_ids) & set(row_ids):
            raise RuntimeError("A Train record occurs in multiple folds.")
        all_ids.extend(row_ids)
        row_by_id = {value["record_id"]: value for _, value in rows}
        sidecar_by_id = {value["record_id"]: value for _, value in sidecars}
        fold_action_counts: Counter[str] = Counter()
        for record_id in row_ids:
            row = row_by_id[record_id]
            sidecar = sidecar_by_id[record_id]
            completion_module.validate_schema(row, schema, schema, f"fold{fold_id}.{record_id}")
            if (
                row.get("heldout") is not True
                or int(row.get("fold_id", -1)) != fold_id
                or row.get("test_accessed") is not False
                or contains_gold(row)
            ):
                raise RuntimeError(f"Fold {fold_id} gold-free row contract failed.")
            if canonical_sha256(row) != sidecar.get("source_row_sha256"):
                raise RuntimeError(f"Fold {fold_id} sidecar does not reference its row.")
            if sidecar.get("dev_accessed") is not False or sidecar.get("test_accessed") is not False:
                raise RuntimeError(f"Fold {fold_id} sidecar accessed Dev/Test.")
            finite_values += finite(row, f"fold{fold_id}.{record_id}.row")
            finite_values += finite(sidecar, f"fold{fold_id}.{record_id}.sidecar")
            predictions = {item["prediction_id"]: item for item in row["formal_predictions"]}
            action_by_id = {item["action_id"]: item for item in row["replacement_actions"]}
            if len(predictions) != len(row["formal_predictions"]):
                raise RuntimeError("Duplicate prediction identity within a record.")
            if len(action_by_id) != len(row["replacement_actions"]):
                raise RuntimeError("Duplicate action identity within a record.")
            if prediction_ids & set(predictions) or action_ids & set(action_by_id):
                raise RuntimeError("Prediction/action identity repeats across records.")
            prediction_ids.update(predictions)
            action_ids.update(action_by_id)
            b1_ids = [item["prediction_id"] for item in sidecar["b1_predictions"]]
            a1_ids = [item["action_id"] for item in sidecar["a1_actions"]]
            if set(b1_ids) != set(predictions) or len(b1_ids) != len(predictions):
                raise RuntimeError("B1 sidecar identity coverage is incomplete.")
            if set(a1_ids) != set(action_by_id) or len(a1_ids) != len(action_by_id):
                raise RuntimeError("A1 sidecar identity coverage is incomplete.")
            action_counts = Counter(
                item["base_prediction_id"] for item in row["replacement_actions"]
            )
            actions_per_prediction.extend(
                action_counts.get(prediction_id, 0) for prediction_id in predictions
            )
            for action_label in sidecar["a1_actions"]:
                action = action_by_id[action_label["action_id"]]
                label = str(action_label["protected_label"])
                outcome = str(action_label["metric_outcome"])
                source = str(action_label["candidate_source"])
                if source != str(action["candidate_source"]):
                    raise RuntimeError("A1 source differs between row and sidecar.")
                a1_protected[label] += 1
                a1_metric[outcome] += 1
                source_labels[source][label] += 1
                source_scores[source].append(float(action["candidate_score"]))
                label_scores[label].append(float(action["candidate_score"]))
                label_margins[label].append(float(action["base_candidate_margin"]))
                fold_action_counts[label] += 1
        b1_confusion.update(supervision["b1"]["wrong_type_confusion"])
        b1_wrong = int(supervision["b1"]["base_wrong"])
        b1_exact = int(supervision["b1"]["exact_span_population"])
        fold_report = {
            "fold_id": fold_id,
            "records": len(rows),
            "b1_exact_span": b1_exact,
            "b1_base_correct": int(supervision["b1"]["base_correct"]),
            "b1_base_wrong": b1_wrong,
            "b1_wrong_rate": b1_wrong / b1_exact,
            "b1_wrong_type_confusion": supervision["b1"]["wrong_type_confusion"],
            "a1_actions": sum(fold_action_counts.values()),
            "a1_positive": fold_action_counts["positive"],
            "a1_positive_rate": fold_action_counts["positive"]
            / sum(fold_action_counts.values()),
            "a1_neutral": fold_action_counts["neutral"],
            "a1_damaging": fold_action_counts["damaging"],
            "materialization_discrete_replay_sha256": materialization[
                "discrete_replay_sha256"
            ],
            "rows_sha256": sha256_file(rows_path),
            "sidecar_sha256": sha256_file(sidecar_path),
        }
        if fold_id == 0:
            resource_path = directory / "fold0_scale_resource_report.json"
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            resource_folds.append(
                {
                    "fold_id": 0,
                    "status": "COMPLETE_WITH_APPROVED_STAGE1_TELEMETRY_WAIVER",
                    "stage1_peak_available": False,
                    "resource_report": descriptor(resource_path),
                }
            )
        else:
            archive_path = directory / "fold_archive_manifest.json"
            resource_path = directory / f"fold{fold_id}_resource_summary.json"
            archive = json.loads(archive_path.read_text(encoding="utf-8"))
            resource = json.loads(resource_path.read_text(encoding="utf-8"))
            if (
                archive.get("status") != "CLEANED"
                or resource.get("status") != "COMPLETE"
                or resource.get("hard_stop_reason") is not None
                or resource.get("failure_reason") is not None
                or resource.get("test_accessed") is not False
                or not all(stage in resource.get("stages", {}) for stage in STAGES)
            ):
                raise RuntimeError(f"Fold {fold_id} resource/archive Gate failed.")
            for stage in STAGES:
                stage_resource = dict(resource["stages"][stage])
                required_resource_fields = (
                    "wall_time_seconds",
                    "peak_gpu_mib",
                    "start_disk_free_bytes",
                    "end_disk_free_bytes",
                    "start_fold_bytes",
                    "end_fold_bytes",
                )
                if any(field not in stage_resource for field in required_resource_fields):
                    raise RuntimeError(
                        f"Fold {fold_id} stage {stage} resource telemetry is incomplete."
                    )
            resource_folds.append(
                {
                    "fold_id": fold_id,
                    "status": "COMPLETE",
                    "wall_time_seconds": resource["wall_time_seconds"],
                    "peak_gpu_mib": resource["peak_gpu_mib"],
                    "maximum_transient_bytes": resource["maximum_transient_bytes"],
                    "retained_bytes": archive["retained_bytes"],
                    "stages": resource["stages"],
                    "resource_report": descriptor(resource_path),
                    "archive_manifest": descriptor(archive_path),
                }
            )
        fold_reports.append(fold_report)
        row_sources.append(rows)
        sidecar_sources.append(sidecars)

    if len(all_ids) != 7000 or len(set(all_ids)) != 7000 or set(all_ids) != expected_all_ids:
        raise RuntimeError("Ten-fold union does not equal the 7000 Train records.")
    atomic_concat(merged_rows_path, row_sources)
    atomic_concat(merged_sidecar_path, sidecar_sources)
    merged_rows = read_raw_jsonl(merged_rows_path)
    merged_sidecars = read_raw_jsonl(merged_sidecar_path)
    if len(merged_rows) != 7000 or len(merged_sidecars) != 7000:
        raise RuntimeError("Merged population reload failed.")

    total_wrong = sum(item["b1_base_wrong"] for item in fold_reports)
    total_exact = sum(item["b1_exact_span"] for item in fold_reports)
    total_positive = a1_protected["positive"]
    wrong_rate = total_wrong / total_exact
    dev_wrong_rate = float(gate_contract["formal_dev_wrong_rate"])
    wrong_rate_ratio = wrong_rate / dev_wrong_rate
    max_wrong_share = max(item["b1_base_wrong"] for item in fold_reports) / total_wrong
    max_positive_share = max(item["a1_positive"] for item in fold_reports) / total_positive
    development = fold_reports[:8]
    calibration = fold_reports[8:]
    checks = {
        "records_union_7000": len(all_ids) == 7000,
        "record_ids_unique_7000": len(set(all_ids)) == 7000,
        "manifest_coverage_exact": set(all_ids) == expected_all_ids,
        "all_folds_contract_passed": True,
        "all_folds_sealed": True,
        "all_five_stages_heldout_excluded": True,
        "gold_free_rows_unchanged": True,
        "all_replay_digests_exact": True,
            "schema_and_id_coverage_complete": True,
        "nan_inf_count_zero": True,
        "dev_test_accessed_false": True,
        "b1_minimum_population": total_wrong >= int(gate_contract["minimum_b1_base_wrong"]),
        "a1_minimum_population": total_positive >= int(gate_contract["minimum_a1_protected_positive"]),
        "b1_wrong_in_every_fold": all(item["b1_base_wrong"] > 0 for item in fold_reports),
        "a1_positive_in_every_fold": all(item["a1_positive"] > 0 for item in fold_reports),
        "b1_not_single_fold_dominated": max_wrong_share <= float(gate_contract["maximum_single_fold_share"]),
        "a1_not_single_fold_dominated": max_positive_share <= float(gate_contract["maximum_single_fold_share"]),
        "oof_wrong_rate_same_order_as_formal_dev": (
            float(gate_contract["acceptable_oof_to_dev_wrong_rate_ratio"][0])
            <= wrong_rate_ratio
            <= float(gate_contract["acceptable_oof_to_dev_wrong_rate_ratio"][1])
        ),
        "development_partition_has_b1_and_a1": (
            sum(item["b1_base_wrong"] for item in development) > 0
            and sum(item["a1_positive"] for item in development) > 0
        ),
        "calibration_partition_has_b1_and_a1": (
            sum(item["b1_base_wrong"] for item in calibration) > 0
            and sum(item["a1_positive"] for item in calibration) > 0
        ),
    }
    data_gate_passed = all(checks.values())
    merge_manifest = {
        "kind": "final_chain_oof_ten_fold_merge_manifest",
        "format_version": 1,
        "status": "PASSED" if data_gate_passed else "FAILED",
        "authorization": descriptor(authorization_path),
        "fold_manifest": descriptor(manifest_path),
        "schema": descriptor(schema_path),
        "records": len(all_ids),
        "record_ids_sha256_in_merged_order": stable_id_digest(all_ids),
        "record_ids_sha256_sorted": stable_id_digest(sorted(all_ids)),
        "gold_free_rows": descriptor(merged_rows_path),
        "supervision_sidecar": descriptor(merged_sidecar_path),
        "folds": fold_reports,
        "finite_values_checked": finite_values,
        "nan_inf_count": 0,
        "checks": checks,
        "dev_accessed": False,
        "test_accessed": False,
    }
    merge_manifest_path.write_text(
        json.dumps(merge_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    distribution_gate = {
        "kind": "final_chain_oof_ten_fold_data_distribution_gate",
        "format_version": 1,
        "status": "PASSED_B1_T_MAY_BE_SEPARATELY_AUTHORIZED"
        if data_gate_passed
        else "FAILED_METHOD_TRAINING_REMAINS_LOCKED",
        "data_gate_passed": data_gate_passed,
        "checks": checks,
        "b1": {
            "exact_span": total_exact,
            "base_correct": sum(item["b1_base_correct"] for item in fold_reports),
            "base_wrong": total_wrong,
            "wrong_rate": wrong_rate,
            "formal_dev_wrong_rate": dev_wrong_rate,
            "absolute_wrong_rate_delta_vs_formal_dev": wrong_rate - dev_wrong_rate,
            "oof_to_formal_dev_wrong_rate_ratio": wrong_rate_ratio,
            "max_single_fold_wrong_share": max_wrong_share,
            "wrong_type_confusion": dict(sorted(b1_confusion.items())),
            "folds": fold_reports,
        },
        "a1": {
            "actions": sum(a1_protected.values()),
            "protected_label": dict(sorted(a1_protected.items())),
            "metric_outcome": dict(sorted(a1_metric.items())),
            "max_single_fold_positive_share": max_positive_share,
            "by_candidate_source": {
                source: {
                    "protected_label": dict(sorted(labels.items())),
                    "candidate_score": distribution(source_scores[source]),
                }
                for source, labels in sorted(source_labels.items())
            },
            "candidate_score_by_protected_label": {
                label: distribution(values) for label, values in sorted(label_scores.items())
            },
            "base_candidate_margin_by_protected_label": {
                label: distribution(values) for label, values in sorted(label_margins.items())
            },
            "actions_per_prediction": distribution(actions_per_prediction),
            "hard_negative_count": a1_protected["damaging"],
            "neutral_count": a1_protected["neutral"],
        },
        "partition_support": {
            "folds_0_7": {
                "b1_base_wrong": sum(item["b1_base_wrong"] for item in development),
                "a1_positive": sum(item["a1_positive"] for item in development),
            },
            "folds_8_9": {
                "b1_base_wrong": sum(item["b1_base_wrong"] for item in calibration),
                "a1_positive": sum(item["a1_positive"] for item in calibration),
            },
        },
        "resources": {
            "folds": resource_folds,
            "fold0_stage1_telemetry_waiver_applied": True,
            "folds_1_9_all_complete": all(item["status"] == "COMPLETE" for item in resource_folds[1:]),
        },
        "b1_a1_training_authorized": False,
        "auroc_computed": False,
        "feature_selected": False,
        "threshold_selected": False,
        "calibration_run": False,
        "dev_file_accessed": False,
        "test_accessed": False,
    }
    distribution_path.write_text(
        json.dumps(distribution_gate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Final-chain OOF Ten-fold Merge and Data Gate",
        "",
        f"- Status: `{distribution_gate['status']}`",
        f"- Records: `{len(all_ids)}` unique Train records",
        f"- B1 exact/base-correct/base-wrong: `{total_exact}` / "
        f"`{sum(item['b1_base_correct'] for item in fold_reports)}` / `{total_wrong}`",
        f"- B1 wrong rate: `{wrong_rate:.6f}`; frozen formal Dev: `{dev_wrong_rate:.6f}`; ratio: `{wrong_rate_ratio:.3f}`",
        f"- A1 positive/neutral/damaging: `{a1_protected['positive']}` / "
        f"`{a1_protected['neutral']}` / `{a1_protected['damaging']}`",
        f"- Maximum fold share, B1 wrong/A1 positive: `{max_wrong_share:.3f}` / `{max_positive_share:.3f}`",
        "",
        "| Fold | B1 exact | B1 wrong | Wrong rate | A1 positive | A1 neutral | A1 damaging |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in fold_reports:
        lines.append(
            f"| {item['fold_id']} | {item['b1_exact_span']} | {item['b1_base_wrong']} | "
            f"{item['b1_wrong_rate']:.6f} | {item['a1_positive']} | "
            f"{item['a1_neutral']} | {item['a1_damaging']} |"
        )
    lines.extend(
        [
            "",
            "The data Gate does not authorize B1/A1 training, feature selection, calibration, Dev-file access, or Test access.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(distribution_gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
