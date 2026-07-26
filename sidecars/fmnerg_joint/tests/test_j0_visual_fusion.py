from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from sidecars.fmnerg_joint.config import load_joint_subtype_config
from sidecars.fmnerg_joint.formal_chain import FrozenM33AFeatureProvider
from sidecars.fmnerg_joint.model import J0VisualSubtypeFusion
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


ROOT = Path(__file__).resolve().parents[3]
TAXONOMY = SubtypeTaxonomy.from_file(
    ROOT / "sidecars" / "fmnerg_subtype" / "taxonomy_twitter10000.json"
)


class FakeTextEncoder(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.feature_size = feature_size

    def forward(self, *, coarse_type_ids, **kwargs):
        count = int(coarse_type_ids.numel())
        features = torch.arange(
            count * self.feature_size,
            dtype=torch.float32,
            device=coarse_type_ids.device,
        ).reshape(count, self.feature_size)
        raw = torch.zeros(
            count,
            TAXONOMY.num_subtypes,
            device=coarse_type_ids.device,
        )
        raw[:, 0] = 1.0
        logits = TAXONOMY.mask_logits(raw, coarse_type_ids)
        return {
            "features": features,
            "raw_logits": raw,
            "logits": logits,
            "predicted_subtype_ids": logits.argmax(dim=-1),
        }


class JointJ0Test(unittest.TestCase):
    def test_j0_config_is_isolated_and_dev_only(self):
        config = load_joint_subtype_config(
            ROOT
            / "sidecars"
            / "fmnerg_joint"
            / "configs"
            / "j0_visual_fusion.yaml"
        )
        self.assertEqual(config.model.stage, "j0")
        self.assertEqual(
            config.subtype_checkpoint(41),
            "outputs/fmnerg_roberta128_subtype_encoder_ablation/"
            "all_seed41/best_model.pt",
        )
        self.assertFalse(hasattr(config.data, "test_source"))
        self.assertFalse(hasattr(config.data, "test_formal_predictions"))

    def test_feature_provider_preserves_formal_region_and_gold_iou(self):
        record = {
            "span_candidates": torch.tensor([[0, 1], [2, 3]]),
            "region_features": torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [9.0, 9.0, 9.0, 9.0],
                ]
            ),
            "region_geometry": torch.tensor(
                [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.5, 0.6, 0.7, 0.8],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ),
            "region_detector_scores": torch.tensor([0.4, 0.9, 1.0]),
            "region_mask": torch.tensor([True, True, True]),
            "region_is_null": torch.tensor([False, False, True]),
            "gold_region_positive_mask": torch.tensor(
                [
                    [True, True, False],
                    [False, False, True],
                ]
            ),
            "region_iou_targets": torch.tensor(
                [
                    [0.3, 0.8, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            ),
            "metadata": {"record_id": "r0", "null_region_index": 2},
        }
        provider = FrozenM33AFeatureProvider(
            SimpleNamespace(records=[record], path=Path("synthetic.pt")),
            expanded_cache_sha256="abc",
        )
        gold = provider.gold_evidence("r0", (0, 1))
        self.assertEqual(gold.region_index, 1)
        self.assertEqual(gold.selection_source, "gold_best_iou")
        self.assertTrue(
            torch.equal(
                gold.region_feature,
                torch.tensor([0.0, 2.0, 0.0, 0.0]),
            )
        )
        null = provider.gold_evidence("r0", (2, 3))
        self.assertTrue(null.is_null)
        self.assertTrue(torch.equal(null.region_feature, torch.zeros(4)))
        formal = provider.formal_evidence("r0", (0, 1), 0)
        self.assertEqual(formal.region_index, 0)
        self.assertEqual(formal.selection_source, "formal_m33a")
        self.assertTrue(
            torch.equal(
                formal.region_feature,
                torch.tensor([1.0, 0.0, 0.0, 0.0]),
            )
        )

    def test_zero_initialized_j0_exactly_reproduces_f2_logits(self):
        model = J0VisualSubtypeFusion(
            text_encoder=FakeTextEncoder(feature_size=6),
            taxonomy=TAXONOMY,
            text_feature_size=6,
            region_feature_size=4,
            geometry_size=4,
            hidden_size=8,
            dropout=0.0,
            residual_scale=2.0,
        )
        coarse = torch.tensor([0, 1])
        outputs = model(
            input_ids=torch.ones(2, 3, dtype=torch.long),
            attention_mask=torch.ones(2, 3, dtype=torch.long),
            span_record_indices=torch.tensor([0, 1]),
            span_start_indices=torch.tensor([1, 1]),
            span_end_indices=torch.tensor([1, 1]),
            span_token_mask=torch.tensor(
                [[False, True, False], [False, True, False]]
            ),
            coarse_type_ids=coarse,
            joint_region_features=torch.randn(2, 4),
            joint_region_geometry=torch.randn(2, 4),
            joint_detector_scores=torch.tensor([0.8, 0.4]),
            joint_region_is_null=torch.tensor([False, True]),
            joint_visual_available=torch.tensor([True, True]),
        )
        self.assertTrue(
            torch.equal(outputs["logits"], outputs["base_logits"])
        )
        self.assertTrue(
            torch.equal(
                outputs["predicted_subtype_ids"],
                outputs["base_predicted_subtype_ids"],
            )
        )
        self.assertFalse(bool(outputs["formal_region_mutated"].item()))
        predicted = outputs["predicted_subtype_ids"].tolist()
        self.assertEqual(
            [TAXONOMY.parent_id(value) for value in predicted],
            coarse.tolist(),
        )


if __name__ == "__main__":
    unittest.main()
