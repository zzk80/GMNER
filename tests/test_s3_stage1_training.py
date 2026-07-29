from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from gmner.engine.s3_stage1_evaluator import (
    _update_change_diagnostics,
)
from gmner.engine.s3_stage1_training import (
    build_s3_optimizer,
    derive_static_lambdas,
    late_training_static_scaling_unresolved,
)
from gmner.s3_config import S3Stage1Config, load_s3_config


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


class _FakeBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embeddings = nn.Linear(2, 2)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            nn.Linear(2, 2) for _ in range(12)
        )


class _FakeS3Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = nn.Module()
        self.text_encoder.backbone = _FakeBackbone()
        self.text_projector = nn.Linear(2, 2)
        self.text_graph_encoder = nn.Module()
        self.text_graph_encoder.layers = nn.ModuleList(
            [nn.Linear(2, 2), nn.Linear(2, 2)]
        )
        self.aligner = nn.Linear(2, 2)
        self.boundary_head = nn.Linear(2, 3)
        self.span_type_head = nn.Linear(2, 4)
        self.grounding_head = nn.Linear(2, 2)
        self.region_projector = nn.Linear(2, 2)
        self.image_graph_encoder = nn.Linear(2, 2)


def test_s3_optimizer_assigns_all_roberta_layers_to_backbone_lr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    model = _FakeS3Model()
    config = S3Stage1Config()
    optimizer = build_s3_optimizer(model, config)

    owner_by_id = {}
    lr_by_group = {}
    for group in optimizer.param_groups:
        group_name = str(group["group_name"])
        lr_by_group[group_name] = float(group["lr"])
        for parameter in group["params"]:
            identity = id(parameter)
            assert identity not in owner_by_id
            owner_by_id[identity] = group_name

    trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    assert set(owner_by_id) == set(trainable)
    backbone_names = {
        identity: name
        for identity, name in trainable.items()
        if name.startswith("text_encoder.backbone.")
    }
    assert backbone_names
    assert all(
        owner_by_id[identity] == "backbone"
        for identity in backbone_names
    )
    assert lr_by_group["backbone"] == pytest.approx(
        config.optim.backbone_learning_rate
    )
    assert owner_by_id[
        id(model.text_encoder.backbone.encoder.layer[0].weight)
    ] == "backbone"
    assert owner_by_id[
        id(model.text_encoder.backbone.encoder.layer[11].weight)
    ] == "backbone"

    audit = optimizer.s3_group_audit
    assert all(audit["checks"].values())
    assert audit["wrong_backbone_group"] == []
    output = capsys.readouterr().out
    assert "S3.1 optimizer groups:" in output
    assert "backbone: lr=3.00e-06" in output
    assert "first_parameters=[" in output


def test_late_training_scaling_ignores_step100_but_requires_late_persistence(
) -> None:
    audits = [
        {
            "label": "formal_step_100",
            "weighted_max_min_ratio_by_region": {"layer": 20.0},
        },
        {
            "label": "epoch_1_end",
            "weighted_max_min_ratio_by_region": {"layer": 150.0},
        },
        {
            "label": "best_checkpoint",
            "weighted_max_min_ratio_by_region": {"layer": 200.0},
        },
    ]
    assert late_training_static_scaling_unresolved(audits)
    audits[-1]["weighted_max_min_ratio_by_region"]["layer"] = 50.0
    assert not late_training_static_scaling_unresolved(audits)
