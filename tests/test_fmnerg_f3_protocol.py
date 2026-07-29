from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sidecars.fmnerg_subtype.encoder_config import (
    load_subtype_encoder_config,
)
from sidecars.fmnerg_subtype.f3_protocol import (
    F3ProtocolError,
    evaluate_three_seed_gate,
    load_f3_p1_protocol,
    load_training_summary,
    select_seed42_winner,
    sha256_file,
)
from tools.summarize_fmnerg_subtype_f3_p1 import (
    summarize_final,
    summarize_screen,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    ROOT / "sidecars" / "fmnerg_subtype" / "f3_p1_protocol.yaml"
)
BASE_CONFIG_PATH = (
    ROOT / "sidecars" / "fmnerg_subtype" / "roberta128_encoder_all.yaml"
)


class F3ProtocolTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = load_f3_p1_protocol(PROTOCOL_PATH)

    def test_six_configs_change_exactly_one_learning_rate(self) -> None:
        baseline = load_subtype_encoder_config(BASE_CONFIG_PATH)
        base_rates = {
            "subtype_head": baseline.optim.head_learning_rate,
            "backbone_upper": baseline.optim.backbone_upper_learning_rate,
            "backbone_lower": baseline.optim.backbone_lower_learning_rate,
        }
        for candidate in self.protocol["candidates"]:
            config = load_subtype_encoder_config(ROOT / candidate["config"])
            rates = {
                "subtype_head": config.optim.head_learning_rate,
                "backbone_upper": config.optim.backbone_upper_learning_rate,
                "backbone_lower": config.optim.backbone_lower_learning_rate,
            }
            changed = [
                name
                for name in rates
                if rates[name] != base_rates[name]
            ]
            self.assertEqual(changed, [candidate["changed_group"]])
            changed_group = candidate["changed_group"]
            self.assertAlmostEqual(
                rates[changed_group],
                base_rates[changed_group] * float(candidate["multiplier"]),
            )
            self.assertEqual(config.model, baseline.model)
            self.assertEqual(config.data, baseline.data)
            self.assertEqual(config.initialization, baseline.initialization)
            self.assertEqual(config.optim.num_epochs, baseline.optim.num_epochs)
            self.assertEqual(
                config.runtime.expected_dev_gmner_f1,
                0.6213161081953977,
            )

    def scores(self, default_delta: float) -> dict[str, float]:
        baseline = self.protocol["_baseline_by_seed"][42]
        return {
            candidate["id"]: baseline + default_delta
            for candidate in self.protocol["candidates"]
        }

    def test_seed42_screen_can_end_without_winner(self) -> None:
        result = select_seed42_winner(
            self.scores(default_delta=0.0019),
            self.protocol,
        )
        self.assertIsNone(result["winner_id"])
        self.assertFalse(result["advance_to_confirmation"])

    def test_seed42_screen_selects_unique_best(self) -> None:
        scores = self.scores(default_delta=0.001)
        scores["lr4_upper_double"] = (
            self.protocol["_baseline_by_seed"][42] + 0.003
        )
        result = select_seed42_winner(scores, self.protocol)
        self.assertEqual(result["winner_id"], "lr4_upper_double")
        self.assertEqual(result["selection_reason"], "largest_paired_delta")

    def test_seed42_near_tie_uses_conservative_order(self) -> None:
        scores = self.scores(default_delta=0.001)
        baseline = self.protocol["_baseline_by_seed"][42]
        scores["lr4_upper_double"] = baseline + 0.003
        scores["lr1_head_half"] = baseline + 0.002596
        result = select_seed42_winner(scores, self.protocol)
        self.assertEqual(result["winner_id"], "lr1_head_half")
        self.assertEqual(result["selection_reason"], "conservative_tie_break")

    def valid_contracts(self) -> dict[int, dict[str, object]]:
        return {
            seed: {
                "gmner_f1": 0.6213161081953977,
                "gmner_identity_exact": True,
                "formal_stage1_mutated": False,
                "test_accessed": False,
            }
            for seed in (41, 42, 43)
        }

    def write_training_summary(
        self,
        *,
        output_root: Path,
        candidate_id: str,
        seed: int,
        fmnerg_f1: float,
    ) -> None:
        candidate = self.protocol["_candidate_by_id"][candidate_id]
        config_path = ROOT / candidate["config"]
        summary_path = (
            output_root
            / candidate_id
            / f"seed{seed}"
            / "train_summary.json"
        )
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "metadata": {
                        "kind": "fmnerg_subtype_encoder_training_summary",
                        "encoder_scope": "all",
                        "seed": seed,
                        "config_sha256": sha256_file(config_path),
                        "gmner_identity_exact": True,
                        "formal_stage1_mutated": False,
                        "test_accessed": False,
                        "best_epoch": 8,
                    },
                    "metrics": {
                        "fmnerg_f1": fmnerg_f1,
                        "fine_mner_f1": 0.68,
                        "gmner_f1": 0.6213161081953977,
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_three_seed_gate_passes_only_full_contract(self) -> None:
        baseline = self.protocol["_baseline_by_seed"]
        result = evaluate_three_seed_gate(
            winner_id="lr1_head_half",
            fmnerg_by_seed={
                41: baseline[41] + 0.0031,
                42: baseline[42] + 0.0032,
                43: baseline[43] + 0.0030,
            },
            run_contract_by_seed=self.valid_contracts(),
            protocol=self.protocol,
        )
        self.assertTrue(result["passed"])
        self.assertEqual(
            result["decision"],
            "freeze_f3_and_stop_model_selection",
        )

    def test_three_seed_gate_rejects_nonpositive_pair(self) -> None:
        baseline = self.protocol["_baseline_by_seed"]
        result = evaluate_three_seed_gate(
            winner_id="lr1_head_half",
            fmnerg_by_seed={
                41: baseline[41] + 0.006,
                42: baseline[42] + 0.004,
                43: baseline[43],
            },
            run_contract_by_seed=self.valid_contracts(),
            protocol=self.protocol,
        )
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["every_paired_delta_positive"])

    def test_training_summary_requires_seed_and_config_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / "candidate.yaml"
            config.write_text("candidate: 1\n", encoding="utf-8")
            summary = root / "train_summary.json"
            payload = {
                "metadata": {
                    "kind": "fmnerg_subtype_encoder_training_summary",
                    "encoder_scope": "all",
                    "seed": 42,
                    "config_sha256": sha256_file(config),
                    "gmner_identity_exact": True,
                    "formal_stage1_mutated": False,
                    "test_accessed": False,
                    "best_epoch": 7,
                },
                "metrics": {
                    "fmnerg_f1": 0.53,
                    "fine_mner_f1": 0.68,
                    "gmner_f1": 0.6213161081953977,
                },
            }
            summary.write_text(json.dumps(payload), encoding="utf-8")
            loaded = load_training_summary(
                summary,
                expected_seed=42,
                expected_config_sha256=sha256_file(config),
            )
            self.assertEqual(loaded["best_epoch"], 7)
            with self.assertRaises(F3ProtocolError):
                load_training_summary(
                    summary,
                    expected_seed=41,
                    expected_config_sha256=sha256_file(config),
                )

    def test_screen_and_final_summarizers_form_a_closed_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary)
            baseline = self.protocol["_baseline_by_seed"]
            for candidate in self.protocol["candidates"]:
                candidate_id = candidate["id"]
                delta = (
                    0.0032
                    if candidate_id == "lr1_head_half"
                    else 0.001
                )
                self.write_training_summary(
                    output_root=output_root,
                    candidate_id=candidate_id,
                    seed=42,
                    fmnerg_f1=baseline[42] + delta,
                )

            screen = summarize_screen(
                protocol=self.protocol,
                repo_root=ROOT,
                output_root=output_root,
            )
            self.assertEqual(screen["winner_id"], "lr1_head_half")
            screen_path = output_root / "screen_seed42.json"
            screen_path.write_text(json.dumps(screen), encoding="utf-8")

            for seed, delta in ((41, 0.0031), (43, 0.0030)):
                self.write_training_summary(
                    output_root=output_root,
                    candidate_id="lr1_head_half",
                    seed=seed,
                    fmnerg_f1=baseline[seed] + delta,
                )

            final = summarize_final(
                protocol=self.protocol,
                repo_root=ROOT,
                output_root=output_root,
                screen_summary_path=screen_path,
            )
            self.assertTrue(final["passed"])
            self.assertEqual(
                final["decision"],
                "freeze_f3_and_stop_model_selection",
            )


if __name__ == "__main__":
    unittest.main()
