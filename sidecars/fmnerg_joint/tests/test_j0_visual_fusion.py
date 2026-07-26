from __future__ import annotations

import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from sidecars.fmnerg_joint.config import load_joint_subtype_config
from sidecars.fmnerg_joint.formal_chain import FrozenM33AFeatureProvider
from sidecars.fmnerg_joint.model import J0VisualSubtypeFusion
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy
from tools.summarize_fmnerg_joint_matched_control import (
    build_matched_summary,
)


ROOT = Path(__file__).resolve().parents[3]
TAXONOMY = SubtypeTaxonomy.from_file(
    ROOT / "sidecars" / "fmnerg_subtype" / "taxonomy_twitter10000.json"
)


class FakeTextEncoder(nn.Module):
    def __init__(self, feature_size: int):
        super().__init__()
        self.feature_size = feature_size
        self.raw_bias = nn.Parameter(torch.zeros(TAXONOMY.num_subtypes))

    def forward(self, *, coarse_type_ids, **kwargs):
        count = int(coarse_type_ids.numel())
        features = torch.arange(
            count * self.feature_size,
            dtype=torch.float32,
            device=coarse_type_ids.device,
        ).reshape(count, self.feature_size)
        raw_template = torch.zeros(
            count,
            TAXONOMY.num_subtypes,
            device=coarse_type_ids.device,
        )
        raw_template[:, 0] = 1.0
        raw = raw_template + self.raw_bias.unsqueeze(0)
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

    def test_c1_matches_j0_except_mode_and_output(self):
        visual = load_joint_subtype_config(
            ROOT
            / "sidecars"
            / "fmnerg_joint"
            / "configs"
            / "j0_visual_fusion.yaml"
        )
        control = load_joint_subtype_config(
            ROOT
            / "sidecars"
            / "fmnerg_joint"
            / "configs"
            / "c1_text_continuation.yaml"
        )
        self.assertEqual(visual.model.experiment_mode, "visual_fusion")
        self.assertEqual(control.model.experiment_mode, "text_continuation")
        visual_model = asdict(visual.model)
        control_model = asdict(control.model)
        visual_model.pop("experiment_mode")
        control_model.pop("experiment_mode")
        self.assertEqual(visual_model, control_model)
        self.assertEqual(visual.data, control.data)
        self.assertEqual(visual.initialization, control.initialization)
        self.assertEqual(visual.loss, control.loss)
        self.assertEqual(visual.optim, control.optim)
        visual_runtime = asdict(visual.runtime)
        control_runtime = asdict(control.runtime)
        visual_runtime.pop("output_dir")
        control_runtime.pop("output_dir")
        self.assertEqual(visual_runtime, control_runtime)

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
            experiment_mode="visual_fusion",
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

    def test_c1_ignores_visual_branch_even_if_its_weights_change(self):
        model = J0VisualSubtypeFusion(
            text_encoder=FakeTextEncoder(feature_size=6),
            taxonomy=TAXONOMY,
            text_feature_size=6,
            region_feature_size=4,
            geometry_size=4,
            hidden_size=8,
            dropout=0.0,
            residual_scale=2.0,
            experiment_mode="text_continuation",
        )
        with torch.no_grad():
            model.fusion_head[-1].bias.fill_(5.0)
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
        self.assertEqual(
            float(outputs["bounded_visual_residual_logits"].abs().sum()),
            0.0,
        )

    def test_c1_matches_visual_rng_path_and_blocks_visual_gradients(self):
        visual = J0VisualSubtypeFusion(
            text_encoder=FakeTextEncoder(feature_size=6),
            taxonomy=TAXONOMY,
            text_feature_size=6,
            region_feature_size=4,
            geometry_size=4,
            hidden_size=8,
            dropout=0.5,
            residual_scale=2.0,
            experiment_mode="visual_fusion",
        )
        control = J0VisualSubtypeFusion(
            text_encoder=FakeTextEncoder(feature_size=6),
            taxonomy=TAXONOMY,
            text_feature_size=6,
            region_feature_size=4,
            geometry_size=4,
            hidden_size=8,
            dropout=0.5,
            residual_scale=2.0,
            experiment_mode="text_continuation",
        )
        control.load_state_dict(visual.state_dict())
        visual.train()
        control.train()
        inputs = {
            "input_ids": torch.ones(2, 3, dtype=torch.long),
            "attention_mask": torch.ones(2, 3, dtype=torch.long),
            "span_record_indices": torch.tensor([0, 1]),
            "span_start_indices": torch.tensor([1, 1]),
            "span_end_indices": torch.tensor([1, 1]),
            "span_token_mask": torch.tensor(
                [[False, True, False], [False, True, False]]
            ),
            "coarse_type_ids": torch.tensor([0, 1]),
            "joint_region_features": torch.randn(2, 4),
            "joint_region_geometry": torch.randn(2, 4),
            "joint_detector_scores": torch.tensor([0.8, 0.4]),
            "joint_region_is_null": torch.tensor([False, True]),
            "joint_visual_available": torch.tensor([True, True]),
        }

        torch.manual_seed(7126)
        visual(**inputs)
        visual_rng_state = torch.random.get_rng_state()
        torch.manual_seed(7126)
        control_outputs = control(**inputs)
        control_rng_state = torch.random.get_rng_state()
        self.assertTrue(torch.equal(visual_rng_state, control_rng_state))

        control_outputs["raw_logits"].sum().backward()
        self.assertIsNotNone(control.text_encoder.raw_bias.grad)
        for name, parameter in control.named_parameters():
            if name.startswith("text_encoder."):
                continue
            self.assertIsNone(
                parameter.grad,
                msg=f"C1 visual parameter unexpectedly received grad: {name}",
            )

    def test_matched_summary_separates_overall_and_visual_gains(self):
        j0 = []
        c1 = []
        for seed, base, control, visual in (
            (41, 0.510, 0.514, 0.519),
            (42, 0.512, 0.515, 0.520),
            (43, 0.511, 0.516, 0.518),
        ):
            common = {
                "initial_f2_fine_mner_f1": 0.67,
                "initial_f2_fmnerg_f1": base,
                "coarse_mner_f1": 0.81,
                "eeg_f1": 0.65,
                "gmner_f1": 0.621316108,
            }
            c1.append(
                {
                    "seed": seed,
                    "best_epoch": 3,
                    "metrics": {
                        **common,
                        "fine_mner_f1": 0.675,
                        "fmnerg_f1": control,
                    },
                }
            )
            j0.append(
                {
                    "seed": seed,
                    "best_epoch": 4,
                    "metrics": {
                        **common,
                        "fine_mner_f1": 0.678,
                        "fmnerg_f1": visual,
                    },
                }
            )
        result = build_matched_summary(j0, c1)
        self.assertTrue(result["acceptance"]["overall_scheme"]["accepted"])
        self.assertTrue(result["acceptance"]["visual_module"]["accepted"])
        self.assertGreaterEqual(
            result["aggregate"]["visual_fmnerg_delta_vs_c1"]["mean"],
            0.003,
        )
        self.assertFalse(result["metadata"]["select_best_seed"])
        self.assertNotIn("best_seed", result)


if __name__ == "__main__":
    unittest.main()
