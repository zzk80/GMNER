#!/usr/bin/env python3
"""Audit sealed final-chain OOF fields for a grouped A1 selector without training."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.a1_feature_contract import feature_registry, strict_replacement_scope
from gmner.data.artifact_utils import sha256_file


LABELS = ("positive", "neutral", "damaging")
SOURCES = ("kbest", "perturbation", "viterbi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        default="docs/experiments/a1_0_feature_availability_authorization.json",
    )
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def finite_values(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, int):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(finite_values(item) for item in value)
    if isinstance(value, dict):
        return all(finite_values(item) for item in value.values())
    return False


def distribution(values: Iterable[int]) -> dict[str, float | int]:
    data = sorted(int(value) for value in values)
    if not data:
        return {"count": 0, "minimum": 0, "median": 0.0, "mean": 0.0, "maximum": 0}
    return {
        "count": len(data),
        "minimum": data[0],
        "median": statistics.median(data),
        "mean": statistics.mean(data),
        "maximum": data[-1],
    }


def label_counts(counter: Counter[str]) -> dict[str, int]:
    return {label: int(counter[label]) for label in LABELS}


def markdown(report: dict[str, Any]) -> str:
    strict = report["a1_scope_population"]["strict_boundary_only"]
    lines = [
        "# A1-0 Feature Availability Audit",
        "",
        f"Status: **{report['status']}**",
        "",
        "## Scope correction",
        "",
        "| Population | Actions | FIX | NEUTRAL | DAMAGE |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, title in (("all_enumerated", "All enumerated"), ("strict_boundary_only", "Type/region preserving")):
        item = report["a1_scope_population"][key]
        lines.append(
            f"| {title} | {item['actions']} | {item['labels']['positive']} | "
            f"{item['labels']['neutral']} | {item['labels']['damaging']} |"
        )
    lines.extend(
        [
            "",
            "The formal A1 action space is the type/region-preserving subset. Filtering is a gold-free identity check.",
            "",
            "## Feature availability",
            "",
            "| Status | Count | Authorized inputs |",
            "|---|---:|---:|",
        ]
    )
    statuses = report["feature_status_counts"]
    authorized = report["authorized_feature_counts_by_status"]
    for status in report["allowed_statuses"]:
        lines.append(f"| {status} | {statuses.get(status, 0)} | {authorized.get(status, 0)} |")
    lines.extend(
        [
            "",
            "## Gate",
            "",
        ]
    )
    for key, value in report["checks"].items():
        lines.append(f"- `{key}`: `{str(value).lower()}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            report["decision"],
            "",
            f"Strict population: {strict['actions']} actions across {strict['base_groups']} base-prediction groups.",
            "No model, threshold, utility weight, Dev, or Test was accessed.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "AUTHORIZED_READ_ONLY":
        raise PermissionError("A1-0 is not authorized.")
    for key in (
        "a1_training",
        "feature_selection_from_labels",
        "threshold_selection",
        "utility_parameter_selection",
        "dev_access",
        "test_access",
        "model_replay",
        "latent_feature_rematerialization",
    ):
        if authorization["forbidden"].get(key) is not True:
            raise PermissionError(f"A1-0 lock is disabled: {key}")

    inputs = authorization["input_contract"]
    rows_path = resolve(root, inputs["gold_free_rows"])
    sidecar_path = resolve(root, inputs["supervision_sidecar"])
    if sha256_file(rows_path) != inputs["gold_free_rows_sha256"]:
        raise RuntimeError("Sealed gold-free rows changed.")
    if sha256_file(sidecar_path) != inputs["supervision_sidecar_sha256"]:
        raise RuntimeError("Sealed supervision sidecar changed.")
    rows = read_jsonl(rows_path)
    sidecars = read_jsonl(sidecar_path)
    sidecar_by_id = {str(item["record_id"]): item for item in sidecars}
    if len(rows) != 7000 or len(sidecars) != 7000 or len(sidecar_by_id) != 7000:
        raise RuntimeError("A1-0 requires exactly 7000 unique Train records.")

    all_labels: Counter[str] = Counter()
    strict_labels: Counter[str] = Counter()
    strict_by_fold: dict[int, Counter[str]] = defaultdict(Counter)
    strict_by_source: dict[str, Counter[str]] = defaultdict(Counter)
    all_groups: set[str] = set()
    strict_groups: set[str] = set()
    group_folds: dict[str, set[int]] = defaultdict(set)
    action_ids: set[str] = set()
    strict_group_sizes: Counter[str] = Counter()
    feature_observations: Counter[str] = Counter()
    feature_finite: Counter[str] = Counter()
    feature_fold_coverage: dict[str, Counter[int]] = defaultdict(Counter)
    invariant_changes: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    selected_direct_derived = {
        item["feature_name"]
        for item in feature_registry()
        if item["authorized_for_a1"]
    }

    def observe(name: str, value: Any, fold_id: int) -> None:
        feature_observations[name] += 1
        feature_fold_coverage[name][fold_id] += 1
        if finite_values(value):
            feature_finite[name] += 1

    for row in rows:
        record_id = str(row["record_id"])
        fold_id = int(row["fold_id"])
        sidecar = sidecar_by_id.get(record_id)
        if sidecar is None or int(sidecar["fold_id"]) != fold_id:
            raise RuntimeError("Row/sidecar fold identity mismatch.")
        if row.get("heldout") is not True or row.get("test_accessed") is not False:
            raise RuntimeError("A1-0 row is not a sealed held-out Train row.")
        if sidecar.get("dev_accessed") is not False or sidecar.get("test_accessed") is not False:
            raise RuntimeError("A1-0 sidecar accessed Dev/Test.")
        predictions = {item["prediction_id"]: item for item in row["formal_predictions"]}
        candidates = {item["candidate_id"]: item for item in row["r36_candidates"]["span_candidates"]}
        r16_ids = {item["candidate_id"] for item in row["r16_candidates"]["span_candidates"]}
        regions = {
            item["region_candidate_id"]: item for item in row["r36_candidates"]["region_candidates"]
        }
        labels = {item["action_id"]: item for item in sidecar["a1_actions"]}
        actions = row["replacement_actions"]
        if len(labels) != len(actions) or set(labels) != {item["action_id"] for item in actions}:
            raise RuntimeError("A1 supervision identity coverage is incomplete.")
        strict_actions = [
            item
            for item in actions
            if strict_replacement_scope(item, predictions[item["base_prediction_id"]])
        ]
        local_group_counts = Counter(item["base_prediction_id"] for item in strict_actions)
        local_source_counts = Counter(
            (item["base_prediction_id"], item["candidate_source"])
            for item in strict_actions
        )
        ranked = defaultdict(list)
        for action in strict_actions:
            ranked[action["base_prediction_id"]].append(action)
        ranks: dict[str, tuple[int, float]] = {}
        for group in ranked.values():
            ordered = sorted(group, key=lambda item: (-float(item["candidate_score"]), item["action_id"]))
            best = float(ordered[0]["candidate_score"])
            for rank, action in enumerate(ordered, 1):
                ranks[action["action_id"]] = (rank, best - float(action["candidate_score"]))

        for action in actions:
            action_id = str(action["action_id"])
            if action_id in action_ids:
                raise RuntimeError("An A1 action identity occurs more than once.")
            action_ids.add(action_id)
            base_id = str(action["base_prediction_id"])
            base = predictions.get(base_id)
            candidate = candidates.get(action["candidate_id"])
            if base is None or candidate is None:
                raise RuntimeError("A1 action cannot join to base/candidate identity.")
            candidate_region_id = str(action["observable_features"]["candidate_region_candidate_id"])
            region = regions.get(candidate_region_id)
            if region is None:
                raise RuntimeError("A1 candidate region identity cannot be joined.")
            label = str(labels[action_id]["protected_label"])
            if label not in LABELS:
                raise RuntimeError("Unknown A1 protected label.")
            if labels[action_id]["candidate_source"] != action["candidate_source"]:
                raise RuntimeError("A1 source changed after supervision attachment.")
            all_labels[label] += 1
            source_counts[str(action["candidate_source"])] += 1
            all_groups.add(base_id)
            group_folds[base_id].add(fold_id)
            strict = strict_replacement_scope(action, base)
            if int(action["candidate_type_id"]) != int(base["type_id"]):
                invariant_changes["type_changed"] += 1
            if candidate_region_id != str(base["region_candidate_id"]):
                invariant_changes["region_changed"] += 1
            if strict:
                strict_labels[label] += 1
                strict_by_fold[fold_id][label] += 1
                strict_by_source[str(action["candidate_source"])][label] += 1
                strict_groups.add(base_id)
                strict_group_sizes[base_id] += 1

            base_span = action["observable_features"]["base_span"]
            candidate_span = action["observable_features"]["candidate_span"]
            if not strict:
                continue
            rank, score_gap = ranks[action_id]
            values = {
                "candidate_source": action["candidate_source"],
                "candidate_score": action["candidate_score"],
                "base_candidate_margin": action["base_candidate_margin"],
                "boundary_distance": action["boundary_distance"],
                "candidate_type_id": action["candidate_type_id"],
                "overlap_words_with_base": action["conflict_features"]["overlap_words_with_base"],
                "overlaps_other_formal_count": action["conflict_features"]["overlaps_other_formal_count"],
                "would_preserve_prediction_count": action["conflict_features"]["would_preserve_prediction_count"],
                "base_span": base_span,
                "candidate_span": candidate_span,
                "candidate_region_score": action["observable_features"]["candidate_region_score"],
                "base_type_id": base["type_id"],
                "base_type_logits": base["type_logits"],
                "base_span_score": base["observable_features"]["span_base_score"],
                "base_is_null": base["observable_features"]["base_is_null"],
                "final_visible": base["observable_features"]["final_visible"],
                "fine_region_logit": base["observable_features"]["fine_region_logit"],
                "base_region_is_null": base["region_is_null"],
                "candidate_type_logits": candidate["scores"]["type_logits"],
                "candidate_detector_score": region["detector_score"],
                "candidate_in_r16": action["candidate_id"] in r16_ids,
                "same_type_as_base": int(action["candidate_type_id"]) == int(base["type_id"]),
                "same_region_as_base": candidate_region_id == str(base["region_candidate_id"]),
                "strict_a1_scope_eligible": strict,
                "base_span_length": int(base_span[1]) - int(base_span[0]),
                "candidate_span_length": int(candidate_span[1]) - int(candidate_span[0]),
                "span_length_delta": (int(candidate_span[1]) - int(candidate_span[0])) - (int(base_span[1]) - int(base_span[0])),
                "left_boundary_shift": int(candidate_span[0]) - int(base_span[0]),
                "right_boundary_shift": int(candidate_span[1]) - int(base_span[1]),
                "base_type_confidence": max(_softmax(base["type_logits"])),
                "base_type_margin": _margin(base["type_logits"]),
                "base_type_entropy": _entropy(base["type_logits"]),
                "candidate_type_confidence": max(_softmax(candidate["scores"]["type_logits"])),
                "candidate_type_margin": _margin(candidate["scores"]["type_logits"]),
                "candidate_type_entropy": _entropy(candidate["scores"]["type_logits"]),
                "actions_in_base_group": local_group_counts[base_id],
                "actions_from_same_source_in_group": local_source_counts[(base_id, action["candidate_source"])],
                "candidate_score_rank_in_group": rank,
                "candidate_score_gap_to_group_best": score_gap,
            }
            for name, value in values.items():
                observe(name, value, fold_id)

    if any(len(folds) != 1 for folds in group_folds.values()):
        raise RuntimeError("A base-prediction group crosses OOF folds.")
    total_actions = len(action_ids)
    strict_action_count = sum(strict_labels.values())
    registry = feature_registry()
    for item in registry:
        name = item["feature_name"]
        if name in selected_direct_derived:
            count = int(feature_observations[name])
            item["coverage"] = count / strict_action_count if strict_action_count else 0.0
            item["finite_rate"] = feature_finite[name] / count if count else 0.0
            item["observed_count"] = count
            item["fold_coverage"] = {
                str(fold): int(feature_fold_coverage[name][fold]) for fold in range(10)
            }
            item["cross_fold_semantics_consistent"] = all(
                feature_fold_coverage[name][fold] > 0 for fold in range(10)
            )
        else:
            item["coverage"] = None
            item["finite_rate"] = None
            item["observed_count"] = 0
            item["fold_coverage"] = None
            item["cross_fold_semantics_consistent"] = (
                item["availability"] not in ("SEMANTICALLY_UNSTABLE", "MISSING")
            )

    selected = [item for item in registry if item["authorized_for_a1"]]
    status_counts = Counter(item["availability"] for item in registry)
    authorized_counts = Counter(item["availability"] for item in selected)
    checks = {
        "sealed_input_hashes_exact": True,
        "records_exactly_7000": len(rows) == 7000,
        "all_rows_heldout": all(row.get("heldout") is True for row in rows),
        "all_action_ids_unique": len(action_ids) == total_actions,
        "all_actions_join_one_base_and_candidate": True,
        "all_sidecar_labels_cover_actions_exactly": True,
        "base_groups_do_not_cross_folds": all(len(folds) == 1 for folds in group_folds.values()),
        "selected_feature_coverage_complete": all(item["coverage"] == 1.0 for item in selected),
        "selected_feature_finite_rate_complete": all(item["finite_rate"] == 1.0 for item in selected),
        "selected_feature_semantics_cross_fold_consistent": all(item["cross_fold_semantics_consistent"] for item in selected),
        "strict_scope_is_gold_free": True,
        "strict_scope_has_fix_in_every_fold": all(strict_by_fold[fold]["positive"] > 0 for fold in range(10)),
        "gold_fields_not_authorized_as_inputs": all(not item["authorized_for_a1"] for item in registry if item["availability"] == "FORBIDDEN"),
        "folds_8_9_not_used_for_feature_selection": True,
        "no_training_or_threshold_selection": True,
        "dev_test_accessed_false": True,
    }
    passed = all(checks.values())
    report = {
        "kind": "a1_0_feature_availability_audit",
        "format_version": 1,
        "status": "PASS_OBSERVABLE_TABULAR_A1_T0_READY_FOR_SEPARATE_AUTHORIZATION" if passed else "BLOCKED",
        "authorization": str(authorization_path.relative_to(root)).replace("\\", "/"),
        "authorization_sha256": sha256_file(authorization_path),
        "inputs": {
            "gold_free_rows_sha256": sha256_file(rows_path),
            "supervision_sidecar_sha256": sha256_file(sidecar_path),
            "records": len(rows),
            "folds": list(range(10)),
        },
        "allowed_statuses": authorization["audit_contract"]["statuses"],
        "feature_status_counts": dict(status_counts),
        "authorized_feature_counts_by_status": dict(authorized_counts),
        "features": registry,
        "a1_scope_population": {
            "all_enumerated": {
                "actions": total_actions,
                "base_groups": len(all_groups),
                "labels": label_counts(all_labels),
                "candidate_sources": {source: int(source_counts[source]) for source in SOURCES},
            },
            "strict_boundary_only": {
                "actions": sum(strict_labels.values()),
                "base_groups": len(strict_groups),
                "labels": label_counts(strict_labels),
                "positive_rate": strict_labels["positive"] / sum(strict_labels.values()),
                "actions_per_group": distribution(strict_group_sizes.values()),
                "by_fold": {str(fold): label_counts(strict_by_fold[fold]) for fold in range(10)},
                "by_source": {source: label_counts(strict_by_source[source]) for source in SOURCES},
            },
            "excluded_by_invariant": {
                "type_changed": int(invariant_changes["type_changed"]),
                "region_changed": int(invariant_changes["region_changed"]),
                "excluded_union": total_actions - sum(strict_labels.values()),
            },
        },
        "feature_coverage_population": "strict_type_and_region_preserving_actions_only",
        "checks": checks,
        "decision": (
            "The sealed rows support a gold-free observable-tabular grouped A1-T0 selector. "
            "They do not support latent base-candidate interaction representations without a separately authorized rematerialization. "
            "This audit does not authorize training."
        ),
        "training_authorized": False,
        "utility_parameters_selected": False,
        "threshold_selected": False,
        "auroc_computed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    output_json = resolve(root, authorization["output_contract"]["json"])
    output_md = resolve(root, authorization["output_contract"]["markdown"])
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


def _softmax(values: list[float]) -> list[float]:
    maximum = max(float(value) for value in values)
    exponents = [math.exp(float(value) - maximum) for value in values]
    denominator = sum(exponents)
    return [value / denominator for value in exponents]


def _margin(values: list[float]) -> float:
    probabilities = sorted(_softmax(values), reverse=True)
    return probabilities[0] - probabilities[1]


def _entropy(values: list[float]) -> float:
    return -sum(value * math.log(max(value, 1e-30)) for value in _softmax(values))


if __name__ == "__main__":
    main()
