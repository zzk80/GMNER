import json
from pathlib import Path
import unittest

import torch

from gmner.engine.b1_t0 import action_metrics, freeze_threshold
from gmner.models.b1_t0 import B1T0TextCorrectionModel


ROOT = Path(__file__).resolve().parents[1]


class B1T0Test(unittest.TestCase):
    def test_authorization_keeps_visual_a1_dev_and_test_locked(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "docs"
                / "experiments"
                / "b1_t0_oof_separability_authorization.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["partition_contract"]["development_folds"], list(range(8)))
        self.assertEqual(payload["partition_contract"]["locked_evaluation_folds"], [8, 9])
        for key in ("a1_training", "b1_tv", "visual_features", "clip", "dev_access", "test_access"):
            self.assertTrue(payload["forbidden"][key])

    def test_model_has_separate_gate_and_target_heads(self) -> None:
        model = B1T0TextCorrectionModel(text_size=12, scalar_size=10)
        gate, target = model(torch.randn(5, 12), torch.randn(5, 10))
        self.assertEqual(tuple(gate.shape), (5,))
        self.assertEqual(tuple(target.shape), (5, 4))

    def test_threshold_freeze_respects_precision_and_preservation(self) -> None:
        predictions = {
            "gate_score": torch.tensor([0.95, 0.90, 0.85, 0.10]),
            "target_prediction": torch.tensor([1, 2, 3, 0]),
            "gate_label": torch.tensor([1.0, 1.0, 0.0, 0.0]),
            "gold_type": torch.tensor([1, 2, 0, 1]),
            "base_type": torch.tensor([0, 0, 0, 1]),
            "fold": torch.tensor([0, 1, 2, 3]),
        }
        threshold, metrics = freeze_threshold(
            predictions,
            {
                "minimum_action_precision": 0.75,
                "minimum_base_correct_preservation": 0.5,
            },
        )
        self.assertAlmostEqual(threshold, 0.90, places=6)
        self.assertEqual(metrics["corrected"], 2)
        self.assertEqual(metrics["damaged"], 0)

    def test_type_correction_metrics_do_not_change_prediction_count(self) -> None:
        predictions = {
            "gate_score": torch.tensor([0.9, 0.8, 0.1]),
            "target_prediction": torch.tensor([1, 2, 3]),
            "gate_label": torch.tensor([1.0, 0.0, 1.0]),
            "gold_type": torch.tensor([1, 0, 2]),
            "base_type": torch.tensor([0, 0, 0]),
            "fold": torch.tensor([8, 8, 9]),
        }
        metrics = action_metrics(predictions, 0.5)
        self.assertEqual(metrics["examples"], 3)
        self.assertEqual(metrics["actions"], 2)
        self.assertEqual(metrics["corrected"], 1)
        self.assertEqual(metrics["damaged"], 1)


if __name__ == "__main__":
    unittest.main()
