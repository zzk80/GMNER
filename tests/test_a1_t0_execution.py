import json
from pathlib import Path
import unittest

import torch

from gmner.engine.a1_t0 import (
    group_winners,
    load_frozen_protocol,
    quantile_deltas,
    selected_indices,
)
from gmner.models.a1_t0 import A1T0ActionModel, CLASS_ORDER, SOURCE_ORDER
from scripts.build_a1_t0_dataset import CONCEPTUAL_FEATURE_NAMES, NUMERIC_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]


class A1T0ExecutionTest(unittest.TestCase):
    def test_execution_authorization_preserves_dev_test_locks(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "docs"
                / "experiments"
                / "a1_t0_execution_authorization.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["frozen_protocol_commit"][:7], "f31dcd2")
        self.assertTrue(payload["allowed"]["train_on_folds_0_7"])
        self.assertTrue(payload["allowed"]["one_locked_folds_8_9_evaluation"])
        self.assertTrue(payload["forbidden"]["dev_access"])
        self.assertTrue(payload["forbidden"]["test_access"])
        self.assertTrue(payload["forbidden"]["latent_feature_rematerialization"])
        protocol = load_frozen_protocol(ROOT, payload)
        self.assertEqual(protocol["model_contract"]["epochs"], 30)

    def test_class_source_and_model_output_contract(self) -> None:
        self.assertEqual(CLASS_ORDER, ("FIX", "NEUTRAL", "DAMAGE"))
        self.assertEqual(SOURCE_ORDER, ("kbest", "perturbation", "viterbi"))
        model = A1T0ActionModel(numeric_size=5, source_aware=True)
        output = model(torch.randn(4, 5), torch.eye(3)[torch.tensor([0, 1, 2, 0])])
        self.assertEqual(tuple(output.shape), (4, 3))
        self.assertEqual(len(CONCEPTUAL_FEATURE_NAMES), 35)
        self.assertEqual(len(NUMERIC_FEATURE_NAMES), 42)

    def test_group_tie_break_uses_lexicographically_smaller_action_id(self) -> None:
        probabilities = torch.tensor(
            [[0.8, 0.1, 0.1], [0.8, 0.1, 0.1]], dtype=torch.float32
        )
        metadata = [
            {
                "base_prediction_id": "base",
                "candidate_score": 1.0,
                "action_id": "action:b",
            },
            {
                "base_prediction_id": "base",
                "candidate_score": 1.0,
                "action_id": "action:a",
            },
        ]
        winners = group_winners(probabilities, metadata, 1.0, 0.0)
        self.assertEqual(winners[0][0], 1)

    def test_execute_rule_is_strictly_greater_than_delta(self) -> None:
        winners = [(0, 0.5), (1, 0.500001)]
        self.assertEqual(selected_indices(winners, 0.5), [1])

    def test_quantile_is_linear_and_deduplicated(self) -> None:
        values = quantile_deltas([0.0, 1.0], [0.5, 0.5])
        self.assertEqual(values, [0.0, 0.5])

    def test_protocol_fixes_epoch_30_and_three_of_three(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "docs"
                / "experiments"
                / "a1_t0_observable_tabular_preregistration.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["model_contract"]["epochs"], 30)
        self.assertEqual(
            payload["model_contract"]["checkpoint_selection"],
            "fixed_epoch_no_locked_fold_selection",
        )
        self.assertEqual(payload["final_gate"]["passing_seed_requirement"], "3_of_3")


if __name__ == "__main__":
    unittest.main()
