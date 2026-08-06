import json
from pathlib import Path
import unittest

from gmner.data.a1_feature_contract import feature_registry, strict_replacement_scope


ROOT = Path(__file__).resolve().parents[1]


class A1FeatureContractTest(unittest.TestCase):
    def test_authorization_is_read_only(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "docs"
                / "experiments"
                / "a1_0_feature_availability_authorization.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["status"], "AUTHORIZED_READ_ONLY")
        for key in (
            "a1_training",
            "threshold_selection",
            "utility_parameter_selection",
            "dev_access",
            "test_access",
            "model_replay",
        ):
            self.assertTrue(payload["forbidden"][key])

    def test_strict_scope_preserves_type_region_and_count(self) -> None:
        base = {"type_id": 2, "region_candidate_id": "region:x"}
        action = {
            "candidate_type_id": 2,
            "observable_features": {"candidate_region_candidate_id": "region:x"},
            "conflict_features": {"would_preserve_prediction_count": True},
        }
        self.assertTrue(strict_replacement_scope(action, base))
        action["candidate_type_id"] = 1
        self.assertFalse(strict_replacement_scope(action, base))

    def test_gold_fields_are_never_authorized_inputs(self) -> None:
        registry = feature_registry()
        forbidden = [item for item in registry if item["availability"] == "FORBIDDEN"]
        self.assertTrue(forbidden)
        self.assertTrue(all(not item["authorized_for_a1"] for item in forbidden))
        latent = [
            item
            for item in registry
            if item["availability"] == "REQUIRES_REMATERIALIZATION"
        ]
        self.assertTrue(all(not item["authorized_for_a1"] for item in latent))

    def test_a1_t0_preregistration_does_not_authorize_training(self) -> None:
        payload = json.loads(
            (
                ROOT
                / "docs"
                / "experiments"
                / "a1_t0_observable_tabular_preregistration.json"
            ).read_text(encoding="utf-8")
        )
        self.assertEqual(payload["evidence_contract"]["actions"], 31138)
        self.assertEqual(payload["evidence_contract"]["labels"]["FIX"], 286)
        self.assertEqual(payload["final_gate"]["passing_seed_requirement"], "3_of_3")
        self.assertFalse(
            payload["access_and_authorization"]["a1_t0_training_authorized"]
        )
        self.assertFalse(
            payload["access_and_authorization"][
                "a1_t0_locked_evaluation_authorized"
            ]
        )


if __name__ == "__main__":
    unittest.main()
