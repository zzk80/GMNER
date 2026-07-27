"""Apply the frozen RoBERTa-large Stage1 Phase 1 Dev gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
METRIC_KEYS = (
    "span_f1",
    "token_f1",
    "entity_f1",
    "grounding_accuracy",
    "eeg_f1",
    "gmner_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        default="docs/experiments/roberta_large_stage1_phase1_protocol.yaml",
    )
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--candidate-metrics", required=True)
    parser.add_argument("--preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown-output", required=True)
    return parser.parse_args()


def resolve(path: str | Path) -> Path:
    source = Path(path)
    return source if source.is_absolute() else ROOT / source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summarize(
    *,
    protocol: dict[str, Any],
    baseline_metrics: dict[str, Any] | None = None,
    candidate_metrics: dict[str, Any],
) -> dict[str, Any]:
    expected_baseline = {
        key: float(value)
        for key, value in protocol["baseline"]["metrics"].items()
    }
    if baseline_metrics is None:
        baseline = expected_baseline
    else:
        missing_baseline = [
            key for key in METRIC_KEYS if key not in baseline_metrics
        ]
        if missing_baseline:
            raise KeyError(
                "Recomputed baseline Dev metrics are incomplete: "
                + ", ".join(missing_baseline)
            )
        baseline = {
            key: float(baseline_metrics[key])
            for key in METRIC_KEYS
        }
        for key, expected in expected_baseline.items():
            if abs(baseline[key] - expected) > 1e-9:
                raise ValueError(
                    f"Recomputed baseline drift for {key}: "
                    f"expected={expected} actual={baseline[key]}"
                )
    missing = [
        key for key in METRIC_KEYS if key not in candidate_metrics
    ]
    if missing:
        raise KeyError(
            "Candidate Dev metrics are incomplete: "
            + ", ".join(missing)
        )
    candidate = {
        key: float(candidate_metrics[key])
        for key in METRIC_KEYS
    }
    deltas = {
        key: candidate[key] - baseline[key]
        for key in baseline
        if key in candidate
    }
    gate = protocol["gate"]
    checks = {
        "span_capacity_signal": (
            deltas["span_f1"]
            >= float(gate["span_f1_minimum_delta"])
        ),
        "mner_capacity_signal": (
            deltas["entity_f1"]
            >= float(gate["entity_f1_minimum_delta"])
        ),
        "eeg_safety": (
            deltas["eeg_f1"]
            >= float(gate["eeg_f1_minimum_delta"])
        ),
        "gmner_safety": (
            deltas["gmner_score"]
            >= float(gate["gmner_score_minimum_delta"])
        ),
    }
    capacity_signal = (
        checks["span_capacity_signal"]
        or checks["mner_capacity_signal"]
    )
    safety_passed = checks["eeg_safety"] and checks["gmner_safety"]
    passed = capacity_signal and (
        safety_passed
        if bool(gate["require_safety_checks"])
        else True
    )
    return {
        "baseline": baseline,
        "candidate": candidate,
        "deltas": deltas,
        "checks": checks,
        "capacity_signal_passed": capacity_signal,
        "safety_passed": safety_passed,
        "phase1_passed": passed,
        "decision": (
            protocol["decision"]["on_pass"]
            if passed
            else protocol["decision"]["on_no_go"]
        ),
    }


def markdown_report(result: dict[str, Any]) -> str:
    baseline = result["baseline"]
    candidate = result["candidate"]
    deltas = result["deltas"]
    labels = {
        "span_f1": "Span F1",
        "token_f1": "Token F1",
        "entity_f1": "MNER F1",
        "grounding_accuracy": "Grounding accuracy",
        "eeg_f1": "EEG F1",
        "gmner_score": "GMNER F1",
    }
    rows = [
        "| Metric | RoBERTa-base | RoBERTa-large | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in METRIC_KEYS:
        rows.append(
            f"| {labels[key]} | {baseline.get(key, float('nan')):.6f} "
            f"| {candidate[key]:.6f} "
            f"| {deltas.get(key, float('nan')):+.6f} |"
        )
    status = "PASS" if result["phase1_passed"] else "NO-GO"
    return "\n".join(
        [
            "# RoBERTa-large Stage1 Phase 1",
            "",
            "**Scope:** Seed 42, Dev only, Test not accessed.",
            "",
            *rows,
            "",
            f"## Decision: {status}",
            "",
            f"- Capacity signal: `{result['capacity_signal_passed']}`",
            f"- EEG/GMNER safety: `{result['safety_passed']}`",
            f"- Next action: `{result['decision']}`",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    protocol_path = resolve(args.protocol)
    baseline_path = resolve(args.baseline_metrics)
    candidate_path = resolve(args.candidate_metrics)
    preflight_path = resolve(args.preflight)
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    baseline_metrics = json.loads(
        baseline_path.read_text(encoding="utf-8")
    )
    candidate_metrics = json.loads(
        candidate_path.read_text(encoding="utf-8")
    )
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if (
        preflight.get("metadata", {}).get("test_accessed")
        is not False
    ):
        raise ValueError("Preflight does not prove Test isolation.")
    if (
        preflight.get("metadata", {}).get("test_path_resolved")
        is not False
    ):
        raise ValueError("Preflight resolved the Test path.")

    result = summarize(
        protocol=protocol,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
    )
    payload = {
        "metadata": {
            "kind": "roberta_large_stage1_phase1_summary",
            "format_version": 1,
            "split": "dev",
            "seed": int(protocol["scope"]["seed"]),
            "test_accessed": False,
            "protocol": str(protocol_path.resolve()),
            "protocol_sha256": sha256_file(protocol_path),
            "baseline_metrics": str(baseline_path.resolve()),
            "baseline_metrics_sha256": sha256_file(baseline_path),
            "candidate_metrics": str(candidate_path.resolve()),
            "candidate_metrics_sha256": sha256_file(candidate_path),
            "preflight": str(preflight_path.resolve()),
            "preflight_sha256": sha256_file(preflight_path),
        },
        **result,
    }
    output_path = resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)

    markdown_path = resolve(args.markdown_output)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        markdown_report(result),
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
