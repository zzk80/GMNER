from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gmner.config import load_config
from tools.preflight_roberta_large_stage1 import (
    validate_controlled_configs,
)
from tools.summarize_roberta_large_stage1 import summarize


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT
    / "docs"
    / "experiments"
    / "roberta_large_stage1_phase1_protocol.yaml"
)


def load_protocol() -> dict:
    return yaml.safe_load(PROTOCOL_PATH.read_text(encoding="utf-8"))


def test_roberta_large_config_is_a_controlled_backbone_change():
    protocol = load_protocol()
    baseline = load_config(ROOT / protocol["baseline"]["config"])
    candidate = load_config(ROOT / protocol["candidate"]["config"])

    result = validate_controlled_configs(
        baseline,
        candidate,
        protocol,
    )

    assert result["actual_changes"] == {
        "model": ["text_model_name"],
        "optim": ["batch_size", "gradient_accumulation_steps"],
        "runtime": ["output_dir"],
    }
    assert result["baseline_effective_batch_size"] == 8
    assert result["candidate_effective_batch_size"] == 8
    assert candidate.model.hidden_size == 768
    assert candidate.runtime.save_best_metric == "gmner_score"


def candidate_metrics(protocol: dict, **deltas: float) -> dict:
    baseline = protocol["baseline"]["metrics"]
    metrics = {
        "grounding_accuracy": 0.76,
        **baseline,
    }
    for key, delta in deltas.items():
        metrics[key] = float(baseline[key]) + delta
    return metrics


def test_phase1_gate_requires_capacity_signal_and_safety():
    protocol = load_protocol()
    passed = summarize(
        protocol=protocol,
        candidate_metrics=candidate_metrics(
            protocol,
            span_f1=0.011,
            eeg_f1=0.0,
            gmner_score=0.0,
        ),
    )
    unsafe = summarize(
        protocol=protocol,
        candidate_metrics=candidate_metrics(
            protocol,
            entity_f1=0.011,
            eeg_f1=-0.003,
            gmner_score=0.0,
        ),
    )

    assert passed["phase1_passed"]
    assert passed["decision"] == "build_dev_and_train_r16_then_run_phase2"
    assert not unsafe["phase1_passed"]
    assert unsafe["capacity_signal_passed"]
    assert not unsafe["safety_passed"]


def test_phase1_gate_rejects_small_capacity_delta():
    protocol = load_protocol()
    result = summarize(
        protocol=protocol,
        candidate_metrics=candidate_metrics(
            protocol,
            span_f1=0.009,
            entity_f1=0.009,
            eeg_f1=0.001,
            gmner_score=0.001,
        ),
    )

    assert not result["phase1_passed"]
    assert not result["capacity_signal_passed"]


def test_phase1_baseline_reproduction_tolerates_metric_roundoff():
    protocol = load_protocol()
    baseline = dict(protocol["baseline"]["metrics"])
    baseline["span_f1"] -= 5.1e-9

    result = summarize(
        protocol=protocol,
        baseline_metrics=baseline,
        candidate_metrics=candidate_metrics(protocol),
    )

    assert result["baseline"]["span_f1"] == baseline["span_f1"]


def test_phase1_baseline_reproduction_rejects_real_drift():
    protocol = load_protocol()
    baseline = dict(protocol["baseline"]["metrics"])
    baseline["span_f1"] -= 2e-6

    with pytest.raises(ValueError, match="Recomputed baseline drift"):
        summarize(
            protocol=protocol,
            baseline_metrics=baseline,
            candidate_metrics=candidate_metrics(protocol),
        )
