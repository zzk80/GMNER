from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gmner.engine.s3_stage1_evaluator import (
    _update_change_diagnostics,
)
from gmner.engine.s3_stage1_training import derive_static_lambdas
from gmner.s3_config import load_s3_config


def _observation(step: int) -> dict:
    regions = {
        "roberta_layer_0": 2.0,
        "roberta_layer_5": 2.0,
        "roberta_layer_11": 2.0,
        "cross_modal_aligner": 2.0,
    }
    return {
        "step": step,
        "raw_gradient_norms": {
            "boundary": regions,
            "type": {key: 1.0 for key in regions},
            "grounding": {key: 4.0 for key in regions},
            "alignment": {key: 0.01 for key in regions},
        },
    }


def test_static_scaling_uses_boundary_log_ratio_and_clip() -> None:
    values = derive_static_lambdas(
        [_observation(0), _observation(100)],
        lambda_min=0.05,
        lambda_max=20.0,
        epsilon=1e-12,
    )
    assert values == {
        "boundary": 1.0,
        "type": 2.0,
        "grounding": 0.5,
        "alignment": 20.0,
    }


def test_s3_config_rejects_non_preregistered_probe_length(
    tmp_path: Path,
) -> None:
    config = tmp_path / "bad.yaml"
    config.write_text("probe:\n  steps: 99\n", encoding="utf-8")
    with pytest.raises(ValueError, match="100 steps"):
        load_s3_config(config)


def test_s3_entrypoints_have_no_test_argument() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/probe_s3_gradient_scaling.py",
        "scripts/train_s3_stage1.py",
        "scripts/evaluate_s3_stage1.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        assert 'add_argument("--test' not in source
        assert "split=\"test\"" not in source


def test_boundary_shift_diagnostic_handles_new_non_gold_span() -> None:
    from collections import defaultdict

    diagnostics = defaultdict(float)
    student = [{"span": [1, 3], "type_id": 1, "region_index": 0}]
    gold = [{"span": [1, 2], "type_id": 1, "text": "x"}]

    _update_change_diagnostics(
        diagnostics,
        student_predictions=student,
        baseline_predictions=[],
        gold=gold,
        student_matches={"gmner": set()},
        baseline_matches={"gmner": set()},
        metadata={"gt_boxes_by_name": {}},
        region_boxes=torch.zeros(1, 4),
        null_region_index=0,
    )

    assert diagnostics["boundary_shift_span_count"] == 1.0
    assert diagnostics["boundary_shift_type_correct"] == 1.0
    assert diagnostics["boundary_shift_grounding_correct"] == 1.0
