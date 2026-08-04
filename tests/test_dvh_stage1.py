from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from gmner.config import GMNERConfig
from gmner.data.frozen_clip_cache import (
    DVH_CLIP_CACHE_KIND,
    FrozenClipFeatureStore,
    sha256_file,
)
from gmner.losses.dvh_stage1_loss import compute_dvh_stage1_losses
from gmner.models import dvh_stage1 as dvh_module
from gmner.models.dvh_stage1 import pool_clip_patches_in_boxes


class _DummyTextEncoder(nn.Module):
    def __init__(self, model_name: str, dropout: float = 0.1) -> None:
        super().__init__()
        del model_name, dropout
        self.hidden_size = 8
        self.backbone = nn.Embedding(32, self.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        del attention_mask, token_type_ids
        states = self.backbone(input_ids)
        return states, states[:, 0]


def _config() -> GMNERConfig:
    config = GMNERConfig()
    config.model.dvh_enabled = True
    config.model.dvh_use_clip = True
    config.model.dvh_use_vinvl = True
    config.model.hidden_size = 8
    config.model.region_feature_dim = 6
    config.model.dvh_clip_feature_dim = 5
    config.model.dvh_clip_patch_grid_size = 2
    config.model.cross_attention_heads = 2
    config.model.graph_layers = 1
    config.model.graph_dropout = 0.0
    config.model.dropout = 0.0
    config.loss.lambda_boundary = 1.0
    config.loss.lambda_type = 1.0
    config.loss.lambda_grounding = 1.0
    config.loss.lambda_alignment = 0.1
    config.loss.lambda_gate_regularization = 0.01
    return config


def _batch() -> dict[str, torch.Tensor]:
    batch_size = 2
    sequence_length = 4
    word_count = 2
    entity_count = 1
    region_count = 3
    adjacency = torch.eye(sequence_length).unsqueeze(0).repeat(batch_size, 1, 1)
    entity_masks = torch.zeros(batch_size, entity_count, sequence_length, dtype=torch.bool)
    entity_masks[:, :, 1] = True
    positive = torch.zeros(batch_size, entity_count, region_count, dtype=torch.bool)
    positive[0, 0, 0] = True
    positive[1, 0, 1] = True
    return {
        "input_ids": torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]]),
        "attention_mask": torch.ones(batch_size, sequence_length, dtype=torch.bool),
        "adjacency": adjacency,
        "first_subword_indices": torch.tensor([[1, 2], [1, 2]]),
        "word_mask": torch.ones(batch_size, word_count, dtype=torch.bool),
        "typed_bio_labels": torch.tensor([[1, 0], [1, 0]]),
        "gold_subword_masks": entity_masks,
        "gold_type_ids": torch.tensor([[1], [2]]),
        "type_entity_mask": torch.ones(batch_size, entity_count, dtype=torch.bool),
        "grounding_entity_mask": torch.ones(batch_size, entity_count, dtype=torch.bool),
        "gold_region_positive_mask": positive,
        "region_features": torch.randn(batch_size, region_count, 6),
        "region_boxes": torch.tensor(
            [
                [[0, 0, 50, 50], [50, 50, 100, 100], [0, 0, 0, 0]],
                [[0, 0, 50, 50], [50, 50, 100, 100], [0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        "region_mask": torch.ones(batch_size, region_count, dtype=torch.bool),
        "region_is_null": torch.tensor(
            [[False, False, True], [False, False, True]]
        ),
        "region_scores": torch.tensor([[0.9, 0.8, 1.0], [0.9, 0.8, 1.0]]),
        "image_sizes": torch.tensor([[100.0, 100.0], [100.0, 100.0]]),
        "clip_global_features": torch.randn(batch_size, 5),
        "clip_patch_features": torch.randn(batch_size, 4, 5),
        "clip_patch_mask": torch.ones(batch_size, 4, dtype=torch.bool),
    }


def test_frozen_clip_store_validates_and_loads(tmp_path: Path):
    entry = {
        "image_id": "img0",
        "global_feature": torch.ones(5, dtype=torch.float16),
        "patch_features": torch.ones(4, 5, dtype=torch.float16),
        "patch_mask": torch.ones(4, dtype=torch.bool),
    }
    shard = tmp_path / "shard_00000.pt"
    torch.save({"entries": [entry]}, shard)
    manifest = {
        "kind": DVH_CLIP_CACHE_KIND,
        "format_version": 1,
        "split": "train",
        "records": 1,
        "feature_dim": 5,
        "patch_count": 4,
        "model": {"fully_frozen": True},
        "index": {"img0": {"shard": shard.name, "offset": 0}},
        "shards": {
            shard.name: {"records": 1, "sha256": sha256_file(shard)}
        },
        "test_accessed": False,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    store = FrozenClipFeatureStore(tmp_path, expected_split="train")
    loaded = store.get("img0")
    assert loaded["global_feature"].shape == (5,)
    assert loaded["patch_features"].shape == (4, 5)
    assert loaded["global_feature"].dtype == torch.float32


def test_dvh_has_no_clip_encoder_and_initial_visual_delta_is_zero(monkeypatch):
    monkeypatch.setattr(dvh_module, "TextEncoder", _DummyTextEncoder)
    model = dvh_module.DVHStage1(_config())
    assert not any(
        name.startswith(("clip_encoder", "clip_model"))
        for name, _ in model.named_parameters()
    )
    batch = _batch()
    model.train()
    outputs = model(batch)
    assert torch.count_nonzero(outputs["boundary_visual_delta"]) == 0
    assert torch.count_nonzero(outputs["type_visual_delta"]) == 0
    assert torch.count_nonzero(outputs["grounding_visual_delta"]) == 0
    losses = compute_dvh_stage1_losses(model=model, outputs=outputs, batch=batch)
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert model.boundary_visual_residual.delta.weight.grad is not None
    assert model.type_visual_residual.delta.weight.grad is not None
    assert model.grounding_visual_residual.delta.weight.grad is not None


def test_clip_box_pool_excludes_explicit_null():
    patches = torch.tensor(
        [[[[1.0], [2.0], [3.0], [4.0]]]]
    ).reshape(1, 4, 1)
    pooled = pool_clip_patches_in_boxes(
        patches,
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[[0, 0, 50, 50], [50, 50, 100, 100], [0, 0, 0, 0]]]),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[False, False, True]]),
        torch.tensor([[100.0, 100.0]]),
        grid_size=2,
    )
    assert pooled.shape == (1, 3, 1)
    assert pooled[0, 0, 0].item() == 1.0
    assert pooled[0, 1, 0].item() == 4.0
    assert pooled[0, 2, 0].item() == 0.0
