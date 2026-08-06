from __future__ import annotations

import torch
import torch.nn as nn

from gmner.config import GMNERConfig
from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID
from gmner.data.type_query_collator import _span_valid_mask
from gmner.losses.tq_dv_mner_loss import compute_tq_dv_mner_losses
from gmner.models import tq_dv_mner as tq_module
from gmner.models.tq_dv_mner import _weighted_interval_decode
from scripts.evaluate_tq_fixed_span_type_replay import _decoded_record_entities


class _DummyTextEncoder(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_size = 16
        self.backbone = nn.Embedding(64, self.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        states = self.backbone(input_ids)
        return states, states[:, 0]


def _config() -> GMNERConfig:
    config = GMNERConfig()
    config.model.tq_enabled = True
    config.model.tq_visual_dim = 8
    config.model.tq_clip_feature_dim = 5
    config.model.region_feature_dim = 6
    config.model.cross_attention_heads = 2
    config.model.tq_max_span_length = 3
    config.model.tq_decode_top_k_per_type = 4
    config.model.tq_existence_threshold = 0.5
    config.model.tq_span_score_threshold = 0.0
    return config


def test_fixed_span_replay_decode_is_self_contained() -> None:
    decoded = torch.tensor(
        [[
            DEFAULT_LABEL2ID["O"],
            DEFAULT_LABEL2ID["B-PER"],
            DEFAULT_LABEL2ID["I-PER"],
            DEFAULT_LABEL2ID["O"],
            DEFAULT_LABEL2ID["O"],
        ]],
        dtype=torch.long,
    )
    batch = {
        "metadata": [{"tokens": ["New", "York", "wins"]}],
        "word_count": torch.tensor([3]),
        "first_subword_indices": torch.tensor([[1, 2, 3]]),
        "subword_to_word": torch.tensor([[-1, 0, 1, 2, -1]]),
    }

    masks, type_ids, valid, spans = _decoded_record_entities(decoded, batch)

    assert spans == [[[0, 2]]]
    assert type_ids.tolist() == [[ENTITY_TYPE2ID["PER"]]]
    assert valid.tolist() == [[True]]
    assert masks.tolist() == [[[False, True, True, False, False]]]


def _batch() -> dict[str, torch.Tensor]:
    word_mask = torch.ones(1, 4, 3, dtype=torch.bool)
    query_token_mask = torch.zeros(1, 4, 6, dtype=torch.bool)
    query_token_mask[:, :, 1] = True
    return {
        "query_input_ids": torch.randint(0, 64, (1, 4, 6)),
        "query_attention_mask": torch.ones(1, 4, 6, dtype=torch.bool),
        "query_token_mask": query_token_mask,
        "query_first_subword_indices": torch.tensor([[[2, 3, 4]] * 4]),
        "query_word_mask": word_mask,
        "query_span_valid_mask": _span_valid_mask(
            word_mask, max_span_length=3
        ),
        "clip_global_features": torch.randn(1, 5),
        "clip_patch_features": torch.randn(1, 2, 5),
        "clip_patch_mask": torch.ones(1, 2, dtype=torch.bool),
        "region_features": torch.randn(1, 3, 6),
        "region_mask": torch.ones(1, 3, dtype=torch.bool),
        "region_is_null": torch.tensor([[False, False, True]]),
        "query_existence_targets": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "query_start_targets": torch.tensor(
            [[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        "query_end_targets": torch.tensor(
            [[[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]
        ),
        "query_span_positive_mask": torch.nn.functional.one_hot(
            torch.tensor([1]), num_classes=36
        ).bool().reshape(1, 4, 3, 3),
        "query_region_positive_mask": torch.tensor(
            [[[True, False, False], [False, False, False], [False, False, False], [False, False, False]]]
        ),
        "query_region_supervision_mask": torch.tensor(
            [[True, False, False, False]]
        ),
    }


def test_tq_model_zero_initializes_visual_residual(monkeypatch) -> None:
    monkeypatch.setattr(tq_module, "TextEncoder", _DummyTextEncoder)
    model = tq_module.TQDualVisualMNER(_config())
    outputs = model(_batch())
    assert torch.equal(outputs["visual_delta"], torch.zeros_like(outputs["visual_delta"]))
    assert not any(
        name.startswith(("clip_encoder", "clip_model"))
        for name, _ in model.named_parameters()
    )


def test_tq_losses_are_finite_and_train_residual(monkeypatch) -> None:
    monkeypatch.setattr(tq_module, "TextEncoder", _DummyTextEncoder)
    model = tq_module.TQDualVisualMNER(_config())
    batch = _batch()
    outputs = model(batch)
    losses = compute_tq_dv_mner_losses(
        model=model, outputs=outputs, batch=batch
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.visual_residual.delta.weight.grad is not None
    assert model.start_head.weight.grad is not None
    assert model.qg_query_projection.weight.grad is not None


def test_joint_decode_prefers_best_non_overlapping_set() -> None:
    decoded = _weighted_interval_decode(
        [
            {"span": [0, 2], "type_id": 1, "score": 5.0},
            {"span": [0, 1], "type_id": 0, "score": 3.0},
            {"span": [1, 2], "type_id": 2, "score": 3.0},
            {"span": [3, 4], "type_id": 3, "score": 1.0},
        ]
    )
    assert [(item["span"], item["type_id"]) for item in decoded] == [
        ([0, 1], 0),
        ([1, 2], 2),
        ([3, 4], 3),
    ]


def test_span_mask_uses_end_inclusive_matrix_and_length_limit() -> None:
    mask = _span_valid_mask(
        torch.tensor([[[True, True, True, True]]]), max_span_length=2
    )
    assert bool(mask[0, 0, 0, 0])
    assert bool(mask[0, 0, 0, 1])
    assert not bool(mask[0, 0, 0, 2])
    assert not bool(mask[0, 0, 2, 1])
