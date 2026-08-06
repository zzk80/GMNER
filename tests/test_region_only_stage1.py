from __future__ import annotations

import torch
import torch.nn as nn

from gmner.data.collator import GMNERCollator
from gmner.models.gmner_model import GMNERModel


class _Tokenizer:
    def pad(self, entries, padding, return_tensors):
        del padding, return_tensors
        max_length = max(len(entry["input_ids"]) for entry in entries)
        input_ids = []
        attention_mask = []
        for entry in entries:
            missing = max_length - len(entry["input_ids"])
            input_ids.append(entry["input_ids"] + [0] * missing)
            attention_mask.append(entry["attention_mask"] + [0] * missing)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def _feature(*, with_regions: bool) -> dict:
    feature = {
        "input_ids": [1, 2],
        "attention_mask": [1, 1],
        "target_mask": [1, 0],
        "adjacency": torch.eye(2),
        "image_path": "must_not_be_read.jpg",
    }
    if with_regions:
        feature.update(
            {
                "region_features": torch.ones(2, 4).numpy(),
                "region_mask": [1.0, 1.0],
            }
        )
    return feature


def test_collator_uses_regions_without_materializing_raw_images() -> None:
    batch = GMNERCollator(_Tokenizer())([_feature(with_regions=True)])

    assert "images" not in batch
    assert batch["region_features"].shape == (1, 2, 4)


def test_collator_rejects_samples_without_region_features() -> None:
    collator = GMNERCollator(_Tokenizer())

    try:
        collator([_feature(with_regions=False)])
    except ValueError as error:
        assert "requires region_features" in str(error)
    else:
        raise AssertionError("Expected raw-image-only input to be rejected.")


def test_legacy_resnet_checkpoint_keys_are_ignored_explicitly() -> None:
    model = GMNERModel.__new__(GMNERModel)
    nn.Module.__init__(model)
    model.probe = nn.Linear(2, 2)
    legacy_state = model.state_dict()
    legacy_state["image_encoder.backbone.conv1.weight"] = torch.ones(1)

    incompatible = model.load_state_dict(legacy_state, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
