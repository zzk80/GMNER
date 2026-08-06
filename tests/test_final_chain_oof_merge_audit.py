import importlib.util
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "merge_audit_final_chain_oof_population.py"
SPEC = importlib.util.spec_from_file_location("merge_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FinalChainOOFMergeAuditTest(unittest.TestCase):
    def test_authorization_keeps_training_and_access_locked(self) -> None:
        path = (
            ROOT
            / "docs"
            / "experiments"
            / "final_chain_oof_ten_fold_merge_authorization.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        MODULE.validate_authorization(payload)
        for key in (
            "b1_a1_training",
            "feature_selection",
            "auroc_computation",
            "threshold_selection",
            "calibration",
            "dev_file_access",
            "test_access",
        ):
            changed = json.loads(json.dumps(payload))
            changed["forbidden"][key] = False
            with self.assertRaises(PermissionError):
                MODULE.validate_authorization(changed)

    def test_gold_detection_and_finite_gate(self) -> None:
        self.assertTrue(MODULE.contains_gold({"gold_type": 1}))
        self.assertFalse(MODULE.contains_gold({"type_logits": [1.0, 2.0]}))
        self.assertEqual(MODULE.finite({"x": [1.0, 2]}, "row"), 1)
        with self.assertRaises(ValueError):
            MODULE.finite({"x": float("nan")}, "row")

    def test_distribution_is_descriptive(self) -> None:
        report = MODULE.distribution([1.0, 2.0, 3.0])
        self.assertEqual(report["count"], 3)
        self.assertEqual(report["median"], 2.0)

    def test_fold0_historical_completion_key_is_explicitly_supported(self) -> None:
        self.assertTrue(
            MODULE.completion_heldout_excluded(
                {
                    "all_five_supervised_stages_complete_and_heldout_excluded": True
                },
                0,
            )
        )
        self.assertFalse(
            MODULE.completion_heldout_excluded(
                {"all_five_supervised_stages_heldout_excluded": True}, 0
            )
        )
        self.assertTrue(
            MODULE.completion_heldout_excluded(
                {"all_five_supervised_stages_heldout_excluded": True}, 1
            )
        )


if __name__ == "__main__":
    unittest.main()
