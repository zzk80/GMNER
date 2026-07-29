"""Run all three frozen M3.3A Dev-only Oracle diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import (
    PairedRecordCandidateCollator,
    PairedRecordCandidateDataset,
    RecordCandidateDataset,
)
from gmner.engine.evidence_visibility_diagnostics import (
    collect_m33a_error_records,
    summarize_error_taxonomy,
)
from gmner.engine.evidence_visibility_evaluator import (
    evaluate_evidence_visibility,
)
from gmner.engine.m33a_oracles import (
    assignment_bootstrap,
    evaluate_visibility_policy_curves,
    span_recovery_oracle,
    visibility_gold_oracle,
)
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.fine_grounding_adapter_config import (
    load_fine_grounding_adapter_config,
)
from scripts.analyze_m33a_error_taxonomy import (
    git_commit,
    git_is_dirty,
    require_dev_cache,
    sha256_file,
    write_json,
)
from scripts.train_evidence_visibility import load_frozen_chain
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--formal-cache", required=True)
    parser.add_argument("--expanded-cache", required=True)
    parser.add_argument("--split", choices=("dev",), default="dev")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--bootstrap-iterations", type=int, default=10000)
    return parser.parse_args()


def _visibility_payload(
    records: list[dict],
    *,
    predicted: int,
    gold_count: int,
    formal_budget: int,
    expanded_budget: int,
) -> tuple[dict, list[dict]]:
    gold_oracle = visibility_gold_oracle(
        records,
        predicted=predicted,
        gold_count=gold_count,
        formal_budget=formal_budget,
        expanded_budget=expanded_budget,
    )
    curves = evaluate_visibility_policy_curves(
        records,
        predicted=predicted,
        gold_count=gold_count,
    )
    rows = [
        row
        for direction in ("null_to_visible", "visible_to_null")
        for row in curves[direction]["rules"]
    ]
    qualifying = [
        row
        for direction in ("null_to_visible", "visible_to_null")
        for row in curves[direction]["qualifying_rules"]
    ]
    combined_corrected = int(
        gold_oracle["combined_gold_oracle"]["oracle_corrected"]
    )
    gate_checks = {
        "gold_oracle_recoverable_at_least_15": combined_corrected >= 15,
        "observable_rule_net_at_least_15": any(
            int(row["net"]) >= 15 for row in rows
        ),
        "observable_rule_action_precision_at_least_0_95": any(
            int(row["net"]) >= 15
            and float(row["action_precision"]) >= 0.95
            for row in rows
        ),
        "observable_rule_gmner_delta_at_least_0_005": any(
            int(row["net"]) >= 15
            and float(row["action_precision"]) >= 0.95
            and float(row["gmner_delta"]) >= 0.005
            for row in rows
        ),
        "observable_rules_use_no_gold_conditions": all(
            row["uses_gold_in_condition"] is False for row in rows
        ),
    }
    payload = {
        "gold_oracle": gold_oracle,
        "observable_policy_curves": {
            direction: {
                "candidate_predictions": curves[direction][
                    "candidate_predictions"
                ],
                "rule_count": len(curves[direction]["rules"]),
                "has_nonempty_rule": curves[direction][
                    "has_nonempty_rule"
                ],
                "best_rule_by_net": curves[direction][
                    "best_rule_by_net"
                ],
                "best_rule_by_net_including_noop": curves[direction][
                    "best_rule_by_net_including_noop"
                ],
                "qualifying_rule_count": len(
                    curves[direction]["qualifying_rules"]
                ),
                "qualifying_rule_ids": [
                    row["rule_id"]
                    for row in curves[direction]["qualifying_rules"]
                ],
            }
            for direction in ("null_to_visible", "visible_to_null")
        },
        "gate": {
            **gate_checks,
            "passed": bool(qualifying)
            and all(
                (
                    gate_checks["gold_oracle_recoverable_at_least_15"],
                    gate_checks[
                        "observable_rules_use_no_gold_conditions"
                    ],
                )
            ),
        },
        "dev_rule_selection_is_optimistic": True,
        "formal_method_requires_train_oof_fitting": True,
    }
    return payload, rows


def _write_curves(path: Path, rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = (
        "direction",
        "rule_id",
        "conditions",
        "triggered",
        "corrected",
        "damaged",
        "neutral",
        "net",
        "action_precision",
        "gmner_delta",
        "uses_gold_in_condition",
        "optimistic_dev_diagnostic",
    )
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **row,
                    "conditions": json.dumps(
                        row["conditions"],
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                }
            )
    temporary.replace(path)


def _comparison(
    visibility: dict, span: dict, assignment: dict
) -> list[dict]:
    best_visibility = max(
        (
            visibility["observable_policy_curves"][direction][
                "best_rule_by_net"
            ]
            for direction in ("null_to_visible", "visible_to_null")
        ),
        key=lambda row: (
            int(row["net"]),
            float(row["action_precision"]),
        ),
    )
    return [
        {
            "oracle": "P1_visibility",
            "observable_net_recovery": int(best_visibility["net"]),
            "theoretical_gmner_delta": float(
                visibility["gold_oracle"]["combined_gold_oracle"][
                    "oracle_gmner_delta"
                ]
            ),
            "formal_oof_cost": "medium_high",
            "method_innovation": "medium",
            "damage_risk": "measured_by_policy_curves",
            "gate_passed": bool(visibility["gate"]["passed"]),
        },
        {
            "oracle": "P2_span_recovery",
            "observable_net_recovery": None,
            "theoretical_gmner_delta": float(
                span["theoretical_deltas"]["gmner_r36_delta"]
            ),
            "formal_oof_cost": "very_high_full_chain_rebuild",
            "method_innovation": "medium_high",
            "damage_risk": "unknown_until_gold_free_selector",
            "gate_passed": bool(span["gate"]["passed"]),
        },
        {
            "oracle": "P3_same_type_assignment",
            "observable_net_recovery": None,
            "theoretical_gmner_delta": float(
                assignment["assignment"]["theoretical_gmner_delta"]
            ),
            "formal_oof_cost": "medium_high",
            "method_innovation": "high",
            "damage_risk": "conditional_module_required",
            "gate_passed": bool(assignment["gate"]["passed"]),
        },
    ]


def _report(
    *,
    formal_metrics: dict,
    visibility: dict,
    span: dict,
    assignment: dict,
    comparison: list[dict],
) -> str:
    def rate_with_ci(payload: dict) -> str:
        rate = payload.get("rate")
        if rate is None:
            return "N/A"
        lower, upper = payload["ci95"]
        return f"{float(rate):.6f} [{lower:.6f}, {upper:.6f}]"

    def difference_with_ci(payload: dict) -> str:
        difference = payload.get("difference")
        if difference is None:
            return "N/A"
        lower, upper = payload["ci95"]
        return (
            f"{float(difference):.6f} "
            f"[{lower:.6f}, {upper:.6f}]"
        )

    directions = ("null_to_visible", "visible_to_null")
    gold_difference = assignment["gold_same_type_multiplicity"][
        "eligible_rate_difference_2plus_minus_1"
    ]
    predicted_difference = assignment[
        "predicted_same_type_multiplicity"
    ]["eligible_rate_difference_2plus_minus_1"]
    lines = [
        "# M3.3A Oracle Diagnostic Report",
        "",
        "## Protocol",
        "",
        "- Split: Dev only",
        "- Test accessed: false",
        "- Model modified: false",
        "- All three Oracles were run independently.",
        "",
        "## Formal Metrics",
        "",
        "| Metric | F1 |",
        "|---|---:|",
        f"| Span | {float(formal_metrics['span_f1']):.6f} |",
        f"| MNER | {float(formal_metrics['entity_f1']):.6f} |",
        f"| EEG | {float(formal_metrics['eeg_f1']):.6f} |",
        f"| GMNER | {float(formal_metrics['gmner_score']):.6f} |",
        "",
        "## P1 Visibility Recoverability",
        "",
        (
            "- Combined gold-oracle corrections: "
            f"{visibility['gold_oracle']['combined_gold_oracle']['oracle_corrected']}"
        ),
        (
            "- Combined theoretical GMNER delta: "
            f"{visibility['gold_oracle']['combined_gold_oracle']['oracle_gmner_delta']:.6f}"
        ),
        "",
        "### Gold Oracle And Force-All Risk",
        "",
        "| Direction | Candidates | Oracle corrected | Force-all corrected | Force-all damaged | Force-all net |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for direction in directions:
        gold_direction = visibility["gold_oracle"][direction]
        force_all = gold_direction["force_all_risk"]
        lines.append(
            f"| {direction} | {gold_direction['candidate_count']} | "
            f"{gold_direction['gold_oracle']['oracle_corrected']} | "
            f"{force_all['corrected']} | {force_all['damaged']} | "
            f"{force_all['net']} |"
        )
    lines.extend(
        [
            "",
            "### Best Non-Empty Preregistered Observable Rules",
            "",
            "| Direction | Policy candidates | Rule | Triggered | Corrected | Damaged | Neutral | Net | Precision | GMNER delta |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for direction in directions:
        policy = visibility["observable_policy_curves"][direction]
        best = policy["best_rule_by_net"]
        lines.append(
            f"| {direction} | {policy['candidate_predictions']} | "
            f"`{best['rule_id']}` | {best['triggered']} | "
            f"{best['corrected']} | {best['damaged']} | "
            f"{best['neutral']} | {best['net']} | "
            f"{best['action_precision']:.6f} | "
            f"{best['gmner_delta']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- P1 Gate passed: {str(visibility['gate']['passed']).lower()}",
            "- Decision: no-go. The gold ceiling is large, but no "
            "registered observable rule meets the net, precision, and "
            "GMNER-delta requirements.",
            "",
        "## P2 Span Recovery",
        "",
            "| S1 category | Count |",
            "|---|---:|",
        ]
    )
    for category, count in span["decomposition"].items():
        lines.append(f"| {category} | {count} |")
    lines.extend(
        [
            "",
            "| Ceiling | Recoverable | Theoretical F1 delta |",
            "|---|---:|---:|",
            (
                "| Span-compatible | "
                f"{span['ceilings']['span_compatible']} | "
                f"{span['theoretical_deltas']['span_f1_delta']:.6f} |"
            ),
            (
                "| MNER-compatible | "
                f"{span['ceilings']['mner_compatible']} | "
                f"{span['theoretical_deltas']['mner_f1_delta']:.6f} |"
            ),
            (
                "| GMNER-compatible R16 | "
                f"{span['ceilings']['gmner_compatible_r16']} | "
                f"{span['theoretical_deltas']['gmner_r16_delta']:.6f} |"
            ),
            (
                "| GMNER-compatible R36 | "
                f"{span['ceilings']['gmner_compatible_r36']} | "
                f"{span['theoretical_deltas']['gmner_r36_delta']:.6f} |"
            ),
            "",
            (
                "- R36-only GMNER-compatible cases: "
                f"{span['ceilings']['gmner_compatible_r36_only']}"
            ),
            (
                "- Oracle-selected candidate source distribution: "
                f"{json.dumps(span['observable_candidate_inventory']['best_oracle_candidate_source_distribution'], ensure_ascii=True)}"
            ),
        (
            "- Gold-free recovery rule validated: "
            f"{str(span['observable_candidate_inventory']['gold_free_recovery_rule_validated']).lower()}"
        ),
        f"- P2 Gate passed: {str(span['gate']['passed']).lower()}",
            "- Decision: no-go for implementation until a gold-free "
            "candidate selector is fitted on Train/OOF and validated on "
            "Dev. Promoting non-Stage1 spans requires rebuilding the "
            "formal downstream chain.",
        "",
        "## P3 Same-Type Assignment",
        "",
            "| R3 rate | Estimate and record-bootstrap 95% CI |",
            "|---|---:|",
            (
                "| Unconditional | "
                f"{rate_with_ci(assignment['overall_rates']['unconditional_r3'])} |"
            ),
            (
                "| Eligible conditional | "
                f"{rate_with_ci(assignment['overall_rates']['eligible_conditional_r3'])} |"
            ),
        (
            "- Gold multiplicity conditional difference and CI: "
                f"{difference_with_ci(gold_difference)}"
        ),
        (
            "- Predicted multiplicity conditional difference and CI: "
                f"{difference_with_ci(predicted_difference)}"
        ),
        (
                "- Unique A1/A2 recoverable entities: "
                f"{assignment['assignment']['A1_unique_recoverable_count']}"
                "/"
                f"{assignment['assignment']['A2_unique_recoverable_count']}"
                " (union "
                f"{assignment['assignment']['unique_recoverable_entities']})"
        ),
        (
            "- Theoretical GMNER delta: "
            f"{assignment['assignment']['theoretical_gmner_delta']:.6f}"
        ),
            (
                "- R3 versus R1+R2: "
                f"{assignment['region_error_balance']['R3_count']}/"
                f"{assignment['region_error_balance']['R1_plus_R2_count']}"
            ),
        f"- P3 Gate passed: {str(assignment['gate']['passed']).lower()}",
        "",
            "### Per-Type R3",
            "",
            "| Type | Records | Gold entities | Eligible | Unconditional R3 | Conditional R3 | A1 | A2 | Unique |",
            "|---|---:|---:|---:|---|---|---:|---:|---:|",
        ]
    )
    for type_name, payload in assignment["by_type"].items():
        lines.append(
            f"| {type_name} | {payload['record_count']} | "
            f"{payload['gold_entity_count']} | "
            f"{payload['eligible_entity_count']} | "
            f"{rate_with_ci(payload['unconditional_r3'])} | "
            f"{rate_with_ci(payload['eligible_conditional_r3'])} | "
            f"{payload['A1_unique_recoverable_count']} | "
            f"{payload['A2_unique_recoverable_count']} | "
            f"{payload['A1_A2_unique_recoverable_count']} |"
        )
    lines.extend(
        [
            "",
            "- Decision: gate passed. Only a predicted-multiplicity "
            "conditioned region-assignment module is supported by this "
            "diagnostic; gold multiplicity cannot be a deployment trigger.",
            "",
        "## Horizontal Comparison",
        "",
            "| Oracle | Observable Net | Theoretical GMNER Delta | OOF Cost | Innovation | Damage Risk | Gate |",
            "|---|---:|---:|---|---|---|---|",
        ]
    )
    for item in comparison:
        observable = (
            "N/A"
            if item["observable_net_recovery"] is None
            else str(item["observable_net_recovery"])
        )
        lines.append(
            f"| {item['oracle']} | {observable} | "
            f"{item['theoretical_gmner_delta']:.6f} | "
            f"{item['formal_oof_cost']} | {item['method_innovation']} | "
            f"{item['damage_risk']} | "
            f"{str(item['gate_passed']).lower()} |"
        )
    lines.extend(
        [
            "",
            "No new model is selected or implemented by this diagnostic. "
            "Dev policy results are optimistic and require Train/OOF fitting.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.split != "dev":
        raise ValueError("M3.3A Oracles are Dev-only.")
    if int(args.bootstrap_iterations) != 10000:
        raise ValueError(
            "The registered protocol requires exactly 10000 iterations."
        )
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    evidence_checkpoint_path = resolve(args.checkpoint, root)
    formal_cache_path = resolve(args.formal_cache, root)
    expanded_cache_path = resolve(args.expanded_cache, root)
    output_dir = resolve(args.output_dir, root)

    config = load_evidence_visibility_config(config_path)
    if args.device:
        config.runtime.device = args.device
    formal = RecordCandidateDataset(formal_cache_path)
    expanded = RecordCandidateDataset(expanded_cache_path)
    require_dev_cache(formal, "Formal")
    require_dev_cache(expanded, "Expanded")
    paired = PairedRecordCandidateDataset(formal, expanded)
    if paired.formal_budget != 16 or paired.expanded_budget != 36:
        raise ValueError(
            "Registered M3.3A Oracles require R16/R36 caches."
        )
    device = torch.device(
        config.runtime.device
        if str(config.runtime.device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    (
        model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
    ) = load_frozen_chain(config, root, device)
    validate_fingerprints(
        paired,
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    checkpoint = torch.load(evidence_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    loader = DataLoader(
        paired,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=PairedRecordCandidateCollator(),
    )
    registered_decode_options = decode_options(hierarchy_config)
    formal_metrics = evaluate_evidence_visibility(
        model,
        fine_model,
        hierarchy,
        loader,
        device,
        decode_options=registered_decode_options,
        loss_options=vars(config.loss).copy(),
    )
    records = collect_m33a_error_records(
        model,
        fine_model,
        hierarchy,
        loader,
        device,
        decode_options=registered_decode_options,
        formal_budget=paired.formal_budget,
        expanded_budget=paired.expanded_budget,
    )
    taxonomy_summary = summarize_error_taxonomy(
        records,
        formal_metrics=formal_metrics,
        tolerance=5e-6,
    )
    predicted = int(
        taxonomy_summary["overall_metrics"]["gmner"]["predicted"]
    )
    gold_count = int(
        taxonomy_summary["overall_metrics"]["gmner"]["gold"]
    )

    visibility, policy_rows = _visibility_payload(
        records,
        predicted=predicted,
        gold_count=gold_count,
        formal_budget=paired.formal_budget,
        expanded_budget=paired.expanded_budget,
    )
    formal_records_by_id = {
        str(record["metadata"]["record_id"]): record
        for record in formal.records
    }
    span = span_recovery_oracle(
        records,
        formal_records_by_id,
        source2id={
            str(key): int(value)
            for key, value in formal.metadata["source2id"].items()
        },
        predicted=predicted,
        gold_count=gold_count,
        formal_budget=paired.formal_budget,
        expanded_budget=paired.expanded_budget,
    )
    assignment = assignment_bootstrap(
        records,
        predicted=predicted,
        gold_count=gold_count,
        formal_budget=paired.formal_budget,
        iterations=args.bootstrap_iterations,
        seed=args.bootstrap_seed,
    )
    comparison = _comparison(visibility, span, assignment)

    fine_config_path = resolve(config.frozen.fine_config, root)
    fine_config = load_fine_grounding_adapter_config(fine_config_path)
    hierarchy_checkpoint_path = resolve(
        fine_config.frozen.hierarchical_checkpoint, root
    )
    coarse_checkpoint_path = resolve(
        fine_config.frozen.coarse_checkpoint, root
    )
    fine_checkpoint_path = resolve(config.frozen.fine_checkpoint, root)
    oracle_module_path = root / "gmner" / "engine" / "m33a_oracles.py"
    diagnostic_module_path = (
        root / "gmner" / "engine" / "evidence_visibility_diagnostics.py"
    )
    test_path = root / "tests" / "test_m33a_oracles.py"
    protocol = {
        "git_commit": git_commit(root),
        "git_dirty": git_is_dirty(root),
        "date": datetime.now(timezone.utc).isoformat(),
        "split": "dev",
        "test_accessed": False,
        "model_modified": False,
        "bootstrap_seed": int(args.bootstrap_seed),
        "bootstrap_iterations": int(args.bootstrap_iterations),
        "bootstrap_unit": "record",
        "confidence_interval": "95% percentile",
        "formal_budget": int(paired.formal_budget),
        "expanded_budget": int(paired.expanded_budget),
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "evidence_checkpoint_path": str(
            evidence_checkpoint_path.resolve()
        ),
        "evidence_checkpoint_sha256": sha256_file(
            evidence_checkpoint_path
        ),
        "hierarchy_checkpoint_path": str(
            hierarchy_checkpoint_path.resolve()
        ),
        "hierarchy_checkpoint_sha256": sha256_file(
            hierarchy_checkpoint_path
        ),
        "coarse_checkpoint_path": str(coarse_checkpoint_path.resolve()),
        "coarse_checkpoint_sha256": sha256_file(
            coarse_checkpoint_path
        ),
        "fine_checkpoint_path": str(fine_checkpoint_path.resolve()),
        "fine_checkpoint_sha256": sha256_file(fine_checkpoint_path),
        "stage1_checkpoint_sha256": str(
            formal.metadata.get("stage1_checkpoint_sha256") or ""
        ),
        "formal_cache_path": str(formal_cache_path.resolve()),
        "formal_cache_sha256": sha256_file(formal_cache_path),
        "expanded_cache_path": str(expanded_cache_path.resolve()),
        "expanded_cache_sha256": sha256_file(expanded_cache_path),
        "decode_options": registered_decode_options,
        "oracle_module_sha256": sha256_file(oracle_module_path),
        "diagnostic_module_sha256": sha256_file(
            diagnostic_module_path
        ),
        "oracle_script_sha256": sha256_file(Path(__file__).resolve()),
        "oracle_test_sha256": sha256_file(test_path),
        "visibility_rule_grid_is_preregistered": True,
        "bootstrap_used": True,
    }
    if not all(
        (
            taxonomy_summary["verification"][
                "formal_metrics_reproduced"
            ],
            taxonomy_summary["verification"]["gold_accounting_passed"],
            taxonomy_summary["verification"][
                "prediction_accounting_passed"
            ],
            protocol["test_accessed"] is False,
            protocol["model_modified"] is False,
        )
    ):
        raise RuntimeError("Oracle input Gate failed before output.")

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "visibility_oracle.json", visibility)
    _write_curves(
        output_dir / "visibility_policy_curves.csv", policy_rows
    )
    write_json(output_dir / "span_recovery_oracle.json", span)
    write_json(output_dir / "assignment_bootstrap.json", assignment)
    write_json(output_dir / "protocol.json", protocol)
    report = _report(
        formal_metrics=formal_metrics,
        visibility=visibility,
        span=span,
        assignment=assignment,
        comparison=comparison,
    )
    temporary_report = output_dir / "report.md.tmp"
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(output_dir / "report.md")
    print(
        json.dumps(
            {
                "records": len(records),
                "formal_gmner": formal_metrics["gmner_score"],
                "P1": {
                    "gold_oracle": visibility["gold_oracle"][
                        "combined_gold_oracle"
                    ],
                    "gate": visibility["gate"],
                },
                "P2": {
                    "decomposition": span["decomposition"],
                    "ceilings": span["ceilings"],
                    "gate": span["gate"],
                },
                "P3": {
                    "assignment": assignment["assignment"],
                    "gate": assignment["gate"],
                },
                "comparison": comparison,
                "output_dir": str(output_dir.resolve()),
                "test_accessed": False,
                "model_modified": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
