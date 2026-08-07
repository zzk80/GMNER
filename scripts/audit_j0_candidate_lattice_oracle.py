#!/usr/bin/env python3
"""Attach Train-only supervision after sealing and compute constrained J0-A Oracle."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID
from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.j0_candidate_lattice import (
    baseline_result,
    canonical_sha256,
    contains_gold_or_supervision,
    evaluate_oracle_stage,
    finite,
    oracle_action_breakdown,
)
from gmner.utils.metrics import extract_entities_from_word_labels
from scripts.build_j0_candidate_lattice import reject_dev_test_path, validate_authorization
from scripts.convert_gmner_conll_to_jsonl import parse_conll


ID2LABEL = {value: key for key, value in DEFAULT_LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization", required=True)
    parser.add_argument("--lattice", required=True)
    parser.add_argument("--lattice-manifest", required=True)
    parser.add_argument("--train-source", required=True)
    parser.add_argument("--output-sidecar", required=True)
    parser.add_argument("--output-report", required=True)
    return parser.parse_args()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def gold_entities(record: dict[str, Any]) -> list[tuple[int, int, int]]:
    entities = extract_entities_from_word_labels(
        [int(value) for value in record["ner_tags"]],
        [str(value) for value in record["tokens"]],
        ID2LABEL,
    )
    return [
        (
            int(entity["start"]),
            int(entity["end"]),
            int(ENTITY_TYPE2ID[str(entity["type"])]),
        )
        for entity in entities
    ]


def mner_f1(correct: int, predicted: int, gold: int) -> float:
    denominator = int(predicted) + int(gold)
    return 0.0 if denominator == 0 else 2.0 * int(correct) / denominator


def stage_specifications(authorization: dict[str, Any]) -> list[dict[str, Any]]:
    budgets = authorization["budgets"]
    stages = [
        {
            "name": "raw_semantic_oracle",
            "top_k": None,
            "enforce_nonoverlap": False,
            "max_record_alternatives": None,
            "max_additions": None,
        },
        {
            "name": "deduplicated_oracle",
            "top_k": None,
            "enforce_nonoverlap": False,
            "max_record_alternatives": None,
            "max_additions": None,
        },
    ]
    for top_k in budgets["reported_per_group_top_k"]:
        stages.append(
            {
                "name": f"top_k_{int(top_k)}_oracle",
                "top_k": int(top_k),
                "enforce_nonoverlap": False,
                "max_record_alternatives": None,
                "max_additions": None,
            }
        )
    stages.extend(
        [
            {
                "name": "record_constrained_oracle",
                "top_k": int(budgets["final_per_group_top_k"]),
                "enforce_nonoverlap": True,
                "max_record_alternatives": None,
                "max_additions": None,
            },
            {
                "name": "final_budget_constrained_oracle",
                "top_k": int(budgets["final_per_group_top_k"]),
                "enforce_nonoverlap": True,
                "max_record_alternatives": int(
                    budgets["max_noncontrol_hypotheses_per_record"]
                ),
                "max_additions": int(budgets["max_additions_per_record"]),
            },
        ]
    )
    return stages


def aggregate_result(values: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(values)
    correct = sum(int(row["correct"]) for row in rows)
    predicted = sum(int(row["predicted"]) for row in rows)
    gold = sum(int(row["gold"]) for row in rows)
    additions = sum(int(row.get("additions", 0)) for row in rows)
    result = {
        "correct": correct,
        "predicted": predicted,
        "gold": gold,
        "additions": additions,
        "mner_f1": mner_f1(correct, predicted, gold),
    }
    if rows and all("span_correct" in row for row in rows):
        result["span_correct"] = sum(int(row["span_correct"]) for row in rows)
    return result


def main() -> None:
    args = parse_args()
    paths = {
        name: Path(value).resolve()
        for name, value in {
            "authorization": args.authorization,
            "lattice": args.lattice,
            "lattice_manifest": args.lattice_manifest,
            "train_source": args.train_source,
            "output_sidecar": args.output_sidecar,
            "output_report": args.output_report,
        }.items()
    }
    for path in paths.values():
        reject_dev_test_path(path)
    authorization = json.loads(paths["authorization"].read_text(encoding="utf-8"))
    validate_authorization(authorization)
    if authorization["authorization"].get("j0_a_postseal_oracle") is not True:
        raise PermissionError("J0-A post-seal Oracle is not authorized.")
    if sha256_file(paths["train_source"]) != authorization["inputs"]["train_source_sha256"]:
        raise ValueError("Train source SHA256 differs from preregistration.")
    manifest = json.loads(paths["lattice_manifest"].read_text(encoding="utf-8"))
    lattice_sha_before = sha256_file(paths["lattice"])
    root = Path(__file__).resolve().parents[1]
    if (
        manifest.get("status") != "SEALED"
        or manifest.get("lattice", {}).get("sha256") != lattice_sha_before
        or manifest.get("supervision_attached") is not False
        or not all(bool(value) for value in manifest.get("checks", {}).values())
        or manifest.get("implementation", {}).get("contract_sha256")
        != sha256_file(root / "gmner" / "data" / "j0_candidate_lattice.py")
        or manifest.get("implementation", {}).get("builder_sha256")
        != sha256_file(root / "scripts" / "build_j0_candidate_lattice.py")
    ):
        raise RuntimeError("Gold-free lattice seal is not valid.")

    source_records = parse_conll(paths["train_source"], image_ext=".jpg")
    source_by_id = {str(record["id"]): record for record in source_records}
    if len(source_by_id) != 7000:
        raise RuntimeError("J0-A requires exactly 7000 Train records.")

    specs = stage_specifications(authorization)
    stage_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fold_stage_rows: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    baseline_rows: list[dict[str, Any]] = []
    fold_baseline_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
    record_ids: list[str] = []
    positive_hypotheses = Counter()
    final_action_breakdown = Counter()
    paths["output_sidecar"].parent.mkdir(parents=True, exist_ok=True)
    temporary = paths["output_sidecar"].with_suffix(
        paths["output_sidecar"].suffix + ".tmp"
    )
    with temporary.open("w", encoding="utf-8", newline="\n") as sidecar:
        for lattice in iter_jsonl(paths["lattice"]):
            if contains_gold_or_supervision(lattice):
                raise RuntimeError("Sealed lattice contains gold/supervision.")
            record_id = str(lattice["record_id"])
            if record_id not in source_by_id:
                raise RuntimeError(f"Missing Train supervision for record {record_id}.")
            gold = gold_entities(source_by_id[record_id])
            gold_set = set(gold)
            baseline = baseline_result(lattice, gold)
            baseline_rows.append(baseline)
            fold_id = int(lattice["fold_id"])
            fold_baseline_rows[fold_id].append(baseline)
            outcomes = {}
            for spec in specs:
                result = evaluate_oracle_stage(
                    lattice,
                    gold,
                    top_k=spec["top_k"],
                    enforce_nonoverlap=spec["enforce_nonoverlap"],
                    max_record_alternatives=spec["max_record_alternatives"],
                    max_additions=spec["max_additions"],
                )
                outcomes[spec["name"]] = result
                stage_rows[spec["name"]].append(result)
                fold_stage_rows[fold_id][spec["name"]].append(result)
            record_breakdown = oracle_action_breakdown(
                lattice,
                gold,
                outcomes["final_budget_constrained_oracle"][
                    "selected_hypothesis_ids"
                ],
            )
            final_action_breakdown.update(record_breakdown)
            labels = []
            for group in lattice["groups"]:
                for hypothesis in [group["control"], *group["alternatives"]]:
                    if hypothesis["span"] is None:
                        exact = False
                    else:
                        span = hypothesis["span"]
                        key = (
                            int(span["start"]),
                            int(span["end"]),
                            int(hypothesis["type_id"]),
                        )
                        exact = key in gold_set
                    if exact:
                        positive_hypotheses[str(hypothesis["primary_source"])] += 1
                    labels.append(
                        {
                            "hypothesis_id": hypothesis["hypothesis_id"],
                            "exact_mner": exact,
                        }
                    )
            sidecar_row = {
                "kind": "j0_candidate_lattice_record_supervision",
                "format_version": 1,
                "record_id": record_id,
                "fold_id": fold_id,
                "source_lattice_row_sha256": canonical_sha256(lattice),
                "gold_entities": [list(item) for item in gold],
                "hypothesis_labels": labels,
                "baseline": baseline,
                "oracle": outcomes,
                "final_budget_action_breakdown": record_breakdown,
                "dev_accessed": False,
                "test_accessed": False,
            }
            finite(sidecar_row)
            sidecar.write(
                json.dumps(sidecar_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            record_ids.append(record_id)
    temporary.replace(paths["output_sidecar"])

    lattice_sha_after = sha256_file(paths["lattice"])
    baseline = aggregate_result(baseline_rows)
    expected = authorization["baseline_contract"]
    baseline_checks = {
        "records": len(record_ids) == int(expected["records"]),
        "record_ids_unique": len(set(record_ids)) == int(expected["records"]),
        "predicted": baseline["predicted"] == int(expected["predicted"]),
        "gold": baseline["gold"] == int(expected["gold"]),
        "span_correct": baseline.get("span_correct") == int(expected["span_correct"]),
        "mner_correct": baseline["correct"] == int(expected["mner_correct"]),
    }
    stages: dict[str, Any] = {}
    for spec in specs:
        name = spec["name"]
        aggregate = aggregate_result(stage_rows[name])
        aggregate["net_correct_gain"] = aggregate["correct"] - baseline["correct"]
        aggregate["mner_f1_delta"] = aggregate["mner_f1"] - baseline["mner_f1"]
        aggregate["dev_1500_equivalent_net_gain"] = (
            aggregate["net_correct_gain"] * 1500.0 / len(record_ids)
        )
        aggregate["configuration"] = spec
        stages[name] = aggregate

    per_fold = []
    final_name = "final_budget_constrained_oracle"
    for fold_id in range(10):
        fold_baseline = aggregate_result(fold_baseline_rows[fold_id])
        fold_final = aggregate_result(fold_stage_rows[fold_id][final_name])
        per_fold.append(
            {
                "fold_id": fold_id,
                "records": len(fold_baseline_rows[fold_id]),
                "baseline": fold_baseline,
                "final_budget_oracle": {
                    **fold_final,
                    "net_correct_gain": fold_final["correct"]
                    - fold_baseline["correct"],
                    "mner_f1_delta": fold_final["mner_f1"]
                    - fold_baseline["mner_f1"],
                },
            }
        )

    final = stages[final_name]
    raw = stages["raw_semantic_oracle"]
    deduplicated = stages["deduplicated_oracle"]
    gate = authorization["continuation_gate"]
    checks = {
        "baseline_contract_exact": all(baseline_checks.values()),
        "lattice_unchanged": lattice_sha_before == lattice_sha_after,
        "record_ids_exact": set(record_ids) == set(source_by_id),
        "raw_dedup_oracle_exact": (
            raw["correct"] == deduplicated["correct"]
            and raw["predicted"] == deduplicated["predicted"]
        ),
        "final_action_breakdown_reconciles": (
            final_action_breakdown["net_correct_contribution"]
            == final["net_correct_gain"]
        ),
        "final_budget_net_gain_at_least_308": final["net_correct_gain"]
        >= int(gate["minimum_oof_net_correct_gain"]),
        "every_fold_positive_net_gain": all(
            row["final_budget_oracle"]["net_correct_gain"] > 0 for row in per_fold
        ),
        "gold_free_lattice_remained_separate": True,
        "dev_test_accessed_false": True,
    }
    gate_passed = all(checks.values())
    report = {
        "kind": "j0_a_candidate_lattice_oracle_audit",
        "format_version": 1,
        "status": "PASSED_J0_B_REMAINS_LOCKED" if gate_passed else "NO_GO_STOP",
        "authorization_sha256": sha256_file(paths["authorization"]),
        "lattice_manifest_sha256": sha256_file(paths["lattice_manifest"]),
        "lattice_sha256_before": lattice_sha_before,
        "lattice_sha256_after": lattice_sha_after,
        "train_source_sha256": sha256_file(paths["train_source"]),
        "sidecar": {
            "path": str(paths["output_sidecar"]),
            "sha256": sha256_file(paths["output_sidecar"]),
            "bytes": paths["output_sidecar"].stat().st_size,
        },
        "records": len(record_ids),
        "record_ids_sha256": stable_id_digest(record_ids),
        "baseline": baseline,
        "baseline_checks": baseline_checks,
        "candidate_counts": manifest["lattice"],
        "positive_hypotheses_by_primary_source": dict(sorted(positive_hypotheses.items())),
        "oracle_stages": stages,
        "final_budget_action_breakdown": dict(
            sorted(final_action_breakdown.items())
        ),
        "per_fold": per_fold,
        "continuation_gate": gate,
        "checks": checks,
        "gate_passed": gate_passed,
        "j0_b_authorized": False,
        "j1_authorized": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    paths["output_report"].parent.mkdir(parents=True, exist_ok=True)
    paths["output_report"].write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
