"""Compare matched B0 and Stage1-F seed 42 using Dev artifacts only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-b0", required=True)
    parser.add_argument("--baseline-r16-oracle", required=True)
    parser.add_argument("--stage1-dev", required=True)
    parser.add_argument("--stage1-r16-oracle", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_dev_artifact(path: str | Path, *, expected_kind: str) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("kind") != expected_kind:
        raise ValueError(
            f"Expected {expected_kind!r}, found {metadata.get('kind')!r}."
        )
    if metadata.get("split") != "dev":
        raise ValueError(f"{source} is not a Dev artifact.")
    if metadata.get("test_accessed") is not False:
        raise ValueError(f"{source} accessed Test.")
    return payload


def nested_metric(mapping: dict, *keys: str) -> float:
    value: object = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return float(value)


def summarize(
    *,
    baseline_b0: dict,
    baseline_oracle: dict,
    stage1_dev: dict,
    stage1_oracle: dict,
) -> dict:
    baseline_metrics = dict(baseline_b0["metrics"])
    stage1_metrics = dict(stage1_dev["metrics"])
    baseline_oracle_metrics = dict(baseline_oracle["metrics"])
    stage1_oracle_metrics = dict(stage1_oracle["metrics"])

    baseline_span = nested_metric(
        baseline_oracle_metrics,
        "stage1_bypass",
        "span",
        "f1",
    )
    stage1_span = nested_metric(
        stage1_oracle_metrics,
        "stage1_bypass",
        "span",
        "f1",
    )
    baseline = {
        "fine_mner_f1": float(baseline_metrics["fine_mner_f1"]),
        "fmnerg_f1": float(baseline_metrics["fmnerg_f1"]),
        "span_f1": baseline_span,
        "visible_region_oracle_recall": float(
            baseline_oracle_metrics["visible_region_oracle_recall"]
        ),
    }
    stage1 = {
        "fine_mner_f1": float(stage1_metrics["fine_mner_f1"]),
        "fmnerg_f1": float(stage1_metrics["fmnerg_f1"]),
        "span_f1": stage1_span,
        "visible_region_oracle_recall": float(
            stage1_oracle_metrics["visible_region_oracle_recall"]
        ),
        "hierarchy_consistency": float(
            stage1_metrics["hierarchy_consistency"]
        ),
    }
    cached_fine = nested_metric(
        stage1_oracle_metrics,
        "stage1_bypass",
        "fine_mner",
        "f1",
    )
    cached_fmnerg = nested_metric(
        stage1_oracle_metrics,
        "stage1_bypass",
        "fmnerg",
        "f1",
    )
    if abs(cached_fine - stage1["fine_mner_f1"]) > 1e-10:
        raise ValueError("Stage1-F Dev and R16 Fine MNER differ.")
    if abs(cached_fmnerg - stage1["fmnerg_f1"]) > 1e-10:
        raise ValueError("Stage1-F Dev and R16 FMNERG differ.")

    deltas = {
        key: stage1[key] - baseline[key]
        for key in (
            "fine_mner_f1",
            "fmnerg_f1",
            "span_f1",
            "visible_region_oracle_recall",
        )
    }
    checks = {
        "fine_mner_delta_at_least_0.003": (
            deltas["fine_mner_f1"] >= 0.003
        ),
        "fmnerg_delta_at_least_0.005": (
            deltas["fmnerg_f1"] >= 0.005
        ),
        "span_delta_at_least_minus_0.003": (
            deltas["span_f1"] >= -0.003
        ),
        "visible_oracle_delta_at_least_minus_0.002": (
            deltas["visible_region_oracle_recall"] >= -0.002
        ),
        "hierarchy_consistency_exact": (
            stage1["hierarchy_consistency"] == 1.0
        ),
    }
    return {
        "baseline_b0": baseline,
        "stage1_f_seed42": stage1,
        "deltas": deltas,
        "single_seed_signal_checks": checks,
        "single_seed_signal_passed": all(checks.values()),
        "decision_scope": (
            "F1 seed42 signal only; F2 still requires at least 2/3 seeds."
        ),
    }


def main() -> None:
    args = parse_args()
    baseline_b0 = load_dev_artifact(
        args.baseline_b0,
        expected_kind="fmnerg_stage1_matched_b0",
    )
    baseline_oracle = load_dev_artifact(
        args.baseline_r16_oracle,
        expected_kind="fmnerg_r16_visible_region_oracle",
    )
    stage1_dev = load_dev_artifact(
        args.stage1_dev,
        expected_kind="fmnerg_stage1_f_dev_evaluation",
    )
    stage1_oracle = load_dev_artifact(
        args.stage1_r16_oracle,
        expected_kind="fmnerg_r16_visible_region_oracle",
    )
    summary = summarize(
        baseline_b0=baseline_b0,
        baseline_oracle=baseline_oracle,
        stage1_dev=stage1_dev,
        stage1_oracle=stage1_oracle,
    )
    result = {
        "metadata": {
            "kind": "fmnerg_stage1_f1_summary",
            "format_version": 1,
            "split": "dev",
            "test_accessed": False,
        },
        **summary,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
