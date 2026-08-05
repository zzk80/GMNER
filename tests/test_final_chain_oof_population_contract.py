import json
from pathlib import Path
import unittest

from gmner.data.final_chain_oof_population_contract import (
    regeneration_metadata_for_contract,
    validate_dynamic_regeneration_metadata,
    validate_final_chain_authorization,
)


ROOT = Path(__file__).resolve().parents[1]


class FinalChainOOFPopulationContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.authorization = json.loads(
            (
                ROOT
                / "docs"
                / "experiments"
                / "final_chain_oof_folds1_9_authorization.json"
            ).read_text(encoding="utf-8")
        )

    def test_authorization_is_limited_to_folds_1_9(self) -> None:
        for fold_id in range(1, 10):
            contract = validate_final_chain_authorization(
                self.authorization, fold_id=fold_id
            )
            self.assertEqual(contract.execution_folds, tuple(range(1, 10)))
        with self.assertRaises(PermissionError):
            validate_final_chain_authorization(self.authorization, fold_id=0)

    def test_forbidden_training_and_access_locks_are_required(self) -> None:
        for key in (
            "b1_a1_training",
            "auroc_feature_selection",
            "threshold_or_calibration",
            "dev_access",
            "test_access",
        ):
            changed = json.loads(json.dumps(self.authorization))
            changed["forbidden"][key] = False
            with self.assertRaises(PermissionError):
                validate_final_chain_authorization(changed, fold_id=1)

    def test_nonformal_modules_and_missing_outputs_are_rejected(self) -> None:
        changed = json.loads(json.dumps(self.authorization))
        changed["chain_contract"]["clip"] = True
        with self.assertRaises(PermissionError):
            validate_final_chain_authorization(changed, fold_id=1)
        changed = json.loads(json.dumps(self.authorization))
        changed["allowed"]["deterministic_replay"] = False
        with self.assertRaises(PermissionError):
            validate_final_chain_authorization(changed, fold_id=1)

    def test_regeneration_identity_round_trip(self) -> None:
        contract = validate_final_chain_authorization(self.authorization, fold_id=3)
        authorization_sha256 = "a" * 64
        metadata = regeneration_metadata_for_contract(
            contract, authorization_sha256=authorization_sha256, fold_id=3
        )
        validate_dynamic_regeneration_metadata(
            metadata,
            artifact_identity=contract.artifact_identity,
            authorization_sha256=authorization_sha256,
            fold_id=3,
            experiment_id=contract.experiment_id,
        )
        self.assertEqual(metadata["execution_folds"], list(range(1, 10)))

    def test_launcher_freezes_parent_pid_before_background_monitor(self) -> None:
        launcher = (ROOT / "tools" / "run_final_chain_oof_folds1_9.sh").read_text(
            encoding="utf-8"
        )
        assignment = 'fold_launcher_pid="$BASHPID"'
        invocation = '--root-pid "$fold_launcher_pid"'
        self.assertIn(assignment, launcher)
        self.assertIn(invocation, launcher)
        self.assertLess(launcher.index(assignment), launcher.index(invocation))


if __name__ == "__main__":
    unittest.main()
