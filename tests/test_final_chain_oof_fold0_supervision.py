from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "fold0_supervision",
    ROOT / "scripts" / "audit_final_chain_oof_fold0_supervision.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def prediction(prediction_id: str, start: int, end: int, type_id: int) -> dict:
    return {
        "prediction_id": prediction_id,
        "span": {"start": start, "end": end, "space": "word_half_open"},
        "type_id": type_id,
    }


def candidate(candidate_id: str, start: int, end: int, type_id: int) -> dict:
    return {
        "candidate_id": candidate_id,
        "span": {"start": start, "end": end, "space": "word_half_open"},
        "type_id": type_id,
    }


class Fold0SupervisionTest(unittest.TestCase):
    def test_labels_b1_and_positive_boundary_replacement(self) -> None:
        row = {
            "record_id": "1",
            "formal_predictions": [prediction("prediction:base", 0, 1, 1)],
            "r36_candidates": {
                "span_candidates": [candidate("candidate:gold", 0, 2, 0)]
            },
            "replacement_actions": [
                {
                    "action_id": "action:fix",
                    "base_prediction_id": "prediction:base",
                    "candidate_id": "candidate:gold",
                    "candidate_source": "kbest",
                    "conflict_features": {"overlaps_other_formal_count": 0},
                }
            ],
        }
        source = {
            "id": 1,
            "tokens": ["New", "York"],
            "ner_tags": [3, 4],
        }
        _, b1, a1 = MODULE.supervise_record(row, source)
        self.assertEqual(b1[0]["population_label"], "not_exact_span")
        self.assertEqual(a1[0]["metric_outcome"], "positive")
        self.assertEqual(a1[0]["protected_label"], "positive")
        self.assertEqual(a1[0]["span_correct_delta"], 1)
        self.assertEqual(a1[0]["mner_correct_delta"], 1)

    def test_overlap_conflict_is_protected_damage(self) -> None:
        row = {
            "record_id": "1",
            "formal_predictions": [prediction("prediction:base", 0, 1, 1)],
            "r36_candidates": {
                "span_candidates": [candidate("candidate:gold", 0, 2, 0)]
            },
            "replacement_actions": [
                {
                    "action_id": "action:risk",
                    "base_prediction_id": "prediction:base",
                    "candidate_id": "candidate:gold",
                    "candidate_source": "perturbation",
                    "conflict_features": {"overlaps_other_formal_count": 1},
                }
            ],
        }
        source = {"id": 1, "tokens": ["New", "York"], "ner_tags": [3, 4]}
        _, _, a1 = MODULE.supervise_record(row, source)
        self.assertEqual(a1[0]["metric_outcome"], "positive")
        self.assertEqual(a1[0]["protected_label"], "damaging")

    def test_exact_span_type_population_labels(self) -> None:
        row = {
            "record_id": "1",
            "formal_predictions": [
                prediction("prediction:correct", 0, 1, 1),
                prediction("prediction:wrong", 2, 3, 2),
            ],
            "r36_candidates": {"span_candidates": []},
            "replacement_actions": [],
        }
        source = {
            "id": 1,
            "tokens": ["Alice", "at", "Paris"],
            "ner_tags": [1, 0, 3],
        }
        _, b1, _ = MODULE.supervise_record(row, source)
        self.assertEqual(
            [item["population_label"] for item in b1],
            ["base_correct", "base_wrong"],
        )
        self.assertEqual(b1[1]["gold_type_id"], 0)


if __name__ == "__main__":
    unittest.main()
