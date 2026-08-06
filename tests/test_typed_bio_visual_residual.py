from __future__ import annotations

import pytest
import torch

from gmner.models.typed_bio_visual_residual import (
    JointTypedBIOVisualStage1,
    ProtectedTypedBIOVisualStage1,
    TypedBIOVisualResidual,
    TypedBIOVisualResidualConfig,
    restore_joint_student_state,
)
from gmner.engine.tp_visual_residual_evaluator import _f1, deranged_image_id_map
from gmner.tp.config import load_tp_joint_m1_config, load_tp_m1_config
from gmner.tp.interfaces import extract_tp_stage1_interfaces


def _inputs(batch: int = 2):
    torch.manual_seed(7)
    return {
        "base_tokens": torch.randn(batch, 5, 12),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 0]] * batch),
        "global_features": torch.randn(batch, 6),
        "region_features": torch.randn(batch, 4, 6),
        "region_boxes": torch.tensor(
            [[[0, 0, 10, 10], [10, 10, 20, 20], [0, 0, 0, 0], [0, 0, 0, 0]]] * batch,
            dtype=torch.float32,
        ),
        "region_scores": torch.tensor([[0.9, 0.7, 0.0, 0.0]] * batch),
        "region_mask": torch.tensor([[1, 1, 0, 0]] * batch),
        "image_sizes": torch.tensor([[20, 20]] * batch),
    }


def _config(variant: str) -> TypedBIOVisualResidualConfig:
    return TypedBIOVisualResidualConfig(
        variant=variant,
        clip_feature_dim=6,
        hidden_size=12,
        attention_heads=3,
        ffn_intermediate_size=24,
        dropout=0.0,
        region_budget=4,
        rho=0.75,
    )


@pytest.mark.parametrize("variant", ["a_text", "a1_global", "a2_r16"])
def test_epoch_zero_residual_is_exactly_zero(variant: str) -> None:
    model = TypedBIOVisualResidual(_config(variant)).eval()
    output = model(**_inputs())
    assert torch.count_nonzero(output["delta_emissions"]) == 0
    assert output["delta_emissions"].shape == (2, 5, 9)


def test_a_text_masks_every_visual_slot() -> None:
    model = TypedBIOVisualResidual(_config("a_text")).eval()
    first = model(**_inputs())
    changed = _inputs()
    changed["global_features"].mul_(1000)
    changed["region_features"].mul_(-1000)
    second = model(**changed)
    assert not first["visual_valid_mask"].any()
    assert torch.equal(first["interaction_states"], second["interaction_states"])


def test_a_text_dummy_replay_matches_masked_visual_forward_exactly() -> None:
    model = TypedBIOVisualResidual(_config("a_text")).eval()
    inputs = _inputs()
    formal = model(**inputs)
    replay = model.forward_text_only(
        base_tokens=inputs["base_tokens"],
        attention_mask=inputs["attention_mask"],
    )

    assert torch.equal(formal["interaction_states"], replay["interaction_states"])
    assert torch.equal(formal["delta_emissions"], replay["delta_emissions"])


def test_a2_uses_only_valid_regions() -> None:
    model = TypedBIOVisualResidual(_config("a2_r16")).eval()
    inputs = _inputs()
    first = model(**inputs)["interaction_states"]
    inputs["region_features"][:, 2:].fill_(1e4)
    second = model(**inputs)["interaction_states"]
    assert torch.allclose(first, second, atol=1e-6)


def test_a2_detector_score_is_not_normalized_away() -> None:
    model = TypedBIOVisualResidual(_config("a2_r16")).eval()
    inputs = _inputs()
    first = model(**inputs)["interaction_states"]
    inputs["region_scores"][:, 0] = 0.1
    second = model(**inputs)["interaction_states"]
    assert not torch.equal(first, second)


def test_residual_bound_and_gradient_path() -> None:
    model = TypedBIOVisualResidual(_config("a2_r16"))
    with torch.no_grad():
        model.residual_head[-1].weight.normal_()
    output = model(**_inputs())
    assert output["delta_emissions"].abs().max() <= 0.75 + 1e-7
    output["delta_emissions"].sum().backward()
    assert model.cross_attention.in_proj_weight.grad is not None
    assert torch.any(model.cross_attention.in_proj_weight.grad != 0)


def test_text_and_a2_have_identical_parameter_counts() -> None:
    text = TypedBIOVisualResidual(_config("a_text"))
    visual = TypedBIOVisualResidual(_config("a2_r16"))
    assert sum(p.numel() for p in text.parameters()) == sum(p.numel() for p in visual.parameters())


def test_preregistered_image_derangement_has_no_fixed_points() -> None:
    mapping = deranged_image_id_map(["a", "b", "c", "d"], seed=101)
    assert set(mapping) == set(mapping.values())
    assert all(source != target for source, target in mapping.items())


def test_tp_f1_uses_the_formal_evaluator_operation_order() -> None:
    correct = 1508
    predicted = 2516
    gold = 2450
    precision = correct / predicted
    recall = correct / gold
    expected = 2.0 * precision * recall / max(precision + recall, 1e-8)
    assert _f1(correct, predicted, gold) == expected


def test_protected_wrapper_keeps_the_formal_stage1_in_eval_mode() -> None:
    base = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Dropout(0.5),
    )
    residual = torch.nn.Linear(4, 4)
    protected = ProtectedTypedBIOVisualStage1(base, residual)

    protected.train()

    assert protected.training
    assert protected.residual.training
    assert not protected.base_model.training
    assert all(not parameter.requires_grad for parameter in protected.base_model.parameters())


def test_protected_wrapper_requires_formal_r16_region_features() -> None:
    protected = ProtectedTypedBIOVisualStage1(
        torch.nn.Linear(4, 4), torch.nn.Linear(4, 4)
    )

    with pytest.raises(ValueError, match="formal R16 region_features"):
        protected({}, {})


def test_inactive_pixel_encoder_can_be_kept_on_cpu() -> None:
    class Base(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.image_encoder = torch.nn.Linear(4, 4)
            self.active = torch.nn.Linear(4, 4)

    protected = ProtectedTypedBIOVisualStage1(Base(), torch.nn.Linear(4, 4))
    protected.offload_unused_image_encoder()

    assert protected._image_encoder_offloaded is True
    assert all(
        parameter.device.type == "cpu"
        for parameter in protected.base_model.image_encoder.parameters()
    )


def test_joint_wrapper_has_complete_non_overlapping_optimizer_scope() -> None:
    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(12)]
            )

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()

    class TextEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.backbone = Backbone()

    class Graph(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(3)]
            )

    class Head(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.classifier = torch.nn.Linear(4, 4)
            self.crf = torch.nn.Linear(4, 4)

    class Base(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = TextEncoder()
            self.text_projector = torch.nn.Linear(4, 4)
            self.text_graph_encoder = Graph()
            self.aligner = torch.nn.Linear(4, 4)
            self.ner_head = Head()
            self.image_encoder = torch.nn.Linear(4, 4)
            self.grounding_head = torch.nn.Linear(4, 4)

    model = JointTypedBIOVisualStage1(
        Base(), torch.nn.Linear(4, 4), unfreeze_last_n_layers=4
    )
    groups = model.parameter_groups()
    grouped = [parameter for values in groups.values() for parameter in values]

    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    assert all(
        not parameter.requires_grad
        for layer in model.base_model.text_encoder.backbone.encoder.layer[:8]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad
        for layer in model.base_model.text_encoder.backbone.encoder.layer[8:]
        for parameter in layer.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.base_model.ner_head.crf.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.base_model.grounding_head.parameters()
    )
    assert all(not parameter.requires_grad for parameter in model.teacher_model.parameters())


def test_joint_config_rejects_scope_drift(tmp_path) -> None:
    source = """
stage: M1_JOINT_EXPLORATORY
test_accessed: false
data:
  base_config: base.yaml
  base_checkpoint: base.pt
  train_clip_cache: train
  dev_clip_cache: dev
  m0_5_report: report.json
model:
  variant: a2_r16
optim:
  unfreeze_last_n_layers: 5
runtime:
  output_dir: out
"""
    path = tmp_path / "joint_drift.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="unfreeze_last_n_layers"):
        load_tp_joint_m1_config(path)


def test_j2_config_expands_only_the_roberta_unfreeze_scope() -> None:
    config = load_tp_joint_m1_config(
        "configs/tp_clip_mner/m1_j2_a_text_full_roberta.yaml"
    )

    assert config.variant == "a_text"
    assert config.unfreeze_last_n_layers == 12
    assert config.backbone_learning_rate == 3e-6
    assert config.fusion_learning_rate == 1e-5
    assert config.learning_rate == 1e-4
    assert config.amp_dtype == "bfloat16"


def test_j3_config_freezes_grounding_protected_scope() -> None:
    config = load_tp_joint_m1_config(
        "configs/tp_clip_mner/m1_j3_a_text_grounding_protected.yaml"
    )

    assert config.variant == "a_text"
    assert config.unfreeze_last_n_layers == 4
    assert config.initialization_checkpoint.endswith(
        "m1_j1_a_text_joint_seed42/best_model.pt"
    )
    assert config.grounding_objective is True
    assert config.lambda_grounding_supervision == 1.0
    assert config.lambda_grounding_preserve == 1.0
    assert config.grounding_temperature == 8.0
    assert config.train_residual is True


def test_j3_r1_config_freezes_residual_and_shortens_training() -> None:
    config = load_tp_joint_m1_config(
        "configs/tp_clip_mner/m1_j3_r1_a_text_grounding_frozen_residual.yaml"
    )

    assert config.variant == "a_text"
    assert config.epochs == 5
    assert config.train_residual is False
    assert config.grounding_objective is True
    assert config.grounding_temperature == 8.0


def test_j3_grounding_loss_supervises_all_rows_and_preserves_teacher_correct_rows() -> None:
    class NerHead(torch.nn.Module):
        def compute_loss(self, *, logits, **kwargs):
            return logits.sum() * 0.0

    class Base(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.ner_head = NerHead()

    model = JointTypedBIOVisualStage1.__new__(JointTypedBIOVisualStage1)
    torch.nn.Module.__init__(model)
    model.base_model = Base()
    model.grounding_objective = True

    student_grounding = torch.tensor(
        [[2.0, 0.0, -1.0], [1.5, 0.0, -1.0]], requires_grad=True
    )
    teacher_grounding = torch.tensor(
        [[2.0, 0.0, -1.0], [1.5, 0.0, -1.0]]
    )
    emissions = torch.zeros((2, 2, 9), requires_grad=True)
    outputs = {
        "teacher_emissions": emissions.detach(),
        "corrected_emissions": emissions,
        "normalized_delta": emissions,
        "base_outputs": {"grounding_logits": student_grounding},
        "teacher_grounding_logits": teacher_grounding,
    }
    batch = {
        "ner_labels": torch.zeros((2, 2), dtype=torch.long),
        "attention_mask": torch.ones((2, 2), dtype=torch.long),
        "ner_loss_weight": torch.ones(2),
        "region_mask": torch.ones((2, 3), dtype=torch.bool),
        "region_positive_mask": torch.tensor(
            [[0, 1, 0], [1, 0, 0]], dtype=torch.bool
        ),
    }

    losses = model.compute_loss(
        outputs,
        batch,
        lambda_grounding_supervision=1.0,
        lambda_grounding_preserve=1.0,
        grounding_temperature=8.0,
    )
    losses["loss"].backward()

    assert losses["grounding_supervision_count"].item() == 2
    assert losses["grounding_teacher_error_count"].item() == 1
    assert losses["grounding_preservation_count"].item() == 1
    assert losses["loss_grounding_supervision"].item() > 0
    assert abs(losses["loss_grounding_preserve"].item()) < 1e-7
    assert student_grounding.grad is not None
    assert student_grounding.grad[0, 1].item() < 0


def test_joint_interface_is_differentiable_only_when_explicitly_requested() -> None:
    tokens = torch.randn(2, 3, 4, requires_grad=True)
    emissions = torch.randn(2, 3, 9, requires_grad=True)
    outputs = {
        "fused_tokens": tokens,
        "ner_logits": emissions,
        "pre_prototype_fused_tokens": tokens,
        "image_nodes": torch.randn(2, 2, 4, requires_grad=True),
        "image_mask": torch.ones(2, 2),
    }

    protected = extract_tp_stage1_interfaces(outputs)
    joint = extract_tp_stage1_interfaces(outputs, detach=False)

    assert not protected.base_emissions.requires_grad
    assert not protected.mner_base_tokens.requires_grad
    assert joint.base_emissions.requires_grad
    assert joint.mner_base_tokens.requires_grad


def test_joint_student_checkpoint_restore_is_exact() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
    state = {
        "0.weight": torch.full_like(model[0].weight, 2.5),
        "0.bias": torch.full_like(model[0].bias, -1.25),
    }

    restore_joint_student_state(model, state)

    assert torch.equal(model[0].weight, state["0.weight"])
    assert torch.equal(model[0].bias, state["0.bias"])
    with pytest.raises(ValueError, match="unknown Student parameters"):
        restore_joint_student_state(model, {"missing.weight": torch.ones(1)})


def test_j3_r1_residual_is_frozen_and_kept_in_eval_mode() -> None:
    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(12)]
            )

    class Backbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = Encoder()

    class Base(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = torch.nn.Module()
            self.text_encoder.backbone = Backbone()
            self.text_graph_encoder = torch.nn.Module()
            self.text_graph_encoder.layers = torch.nn.ModuleList(
                [torch.nn.Linear(4, 4) for _ in range(3)]
            )
            self.aligner = torch.nn.Linear(4, 4)
            self.text_projector = torch.nn.Linear(4, 4)
            self.ner_head = torch.nn.Module()
            self.ner_head.classifier = torch.nn.Linear(4, 4)

    residual = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout(0.5))
    model = JointTypedBIOVisualStage1(
        Base(), residual, unfreeze_last_n_layers=4, train_residual=False
    )
    model.train()

    assert all(not parameter.requires_grad for parameter in model.residual.parameters())
    assert model.residual.training is False
    assert model.parameter_groups()["residual"] == []


def test_m1_config_rejects_protocol_drift(tmp_path) -> None:
    source = """
stage: M1
test_accessed: false
data:
  base_config: base.yaml
  base_checkpoint: base.pt
  train_clip_cache: train
  dev_clip_cache: dev
  m0_5_report: report.json
model:
  variant: a2_r16
optim:
  learning_rate: 0.0002
runtime:
  output_dir: out
"""
    path = tmp_path / "drift.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate"):
        load_tp_m1_config(path)
