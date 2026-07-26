"""Regression tests for the subtype-conditioned R36 evidence probe."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from sidecars.fmnerg_joint.subtype_region_oracle import (
    analyze_visible_error,
    build_visual_prototype_bank,
    summarize_seed_rows,
)
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


ROOT = Path(__file__).resolve().parents[3]


class SubtypeRegionOracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.taxonomy = SubtypeTaxonomy.from_file(
            ROOT
            / "sidecars"
            / "fmnerg_subtype"
            / "taxonomy_twitter10000.json"
        )
        members_by_parent: dict[int, list[int]] = {}
        for subtype_id in range(cls.taxonomy.num_subtypes):
            parent = cls.taxonomy.parent_id(subtype_id)
            members_by_parent.setdefault(parent, []).append(subtype_id)
        cls.parent_id, members = next(
            (parent, values)
            for parent, values in members_by_parent.items()
            if len(values) >= 2
        )
        cls.gold_id, cls.predicted_id = members[:2]

    def build_bank(self):
        return build_visual_prototype_bank(
            [
                (self.gold_id, torch.tensor([1.0, 0.0])),
                (self.gold_id, torch.tensor([0.9, 0.1])),
                (self.predicted_id, torch.tensor([0.0, 1.0])),
            ],
            num_subtypes=self.taxonomy.num_subtypes,
            feature_size=2,
        )

    def test_top2_can_recover_visual_evidence_missing_from_formal_region(self):
        evidence = analyze_visible_error(
            formal_region_index=0,
            fine_ranked_region_indices=[0, 1, 2],
            positive_region_indices={0, 1},
            all_real_region_indices=[0, 1, 2],
            region_features=torch.tensor(
                [
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [0.5, 0.5],
                ]
            ),
            bank=self.build_bank(),
            taxonomy=self.taxonomy,
            parent_id=self.parent_id,
            gold_subtype_id=self.gold_id,
            predicted_subtype_id=self.predicted_id,
            top_ks=[1, 2],
        )
        self.assertLess(
            evidence["formal_region"]["pairwise_margin"],
            0.0,
        )
        self.assertFalse(
            evidence["top_k"]["1"]["gold_beats_predicted"]
        )
        self.assertTrue(
            evidence["top_k"]["2"]["gold_beats_predicted"]
        )
        self.assertTrue(
            evidence["top_k"]["2"]["gold_beats_all_siblings"]
        )

        rows = [
            {
                "visibility": "visible",
                "coarse_type": "PER",
                "gold_subtype": self.taxonomy.labels[self.gold_id],
                "predicted_subtype": self.taxonomy.labels[
                    self.predicted_id
                ],
                "visual_evidence": evidence,
            },
            {
                "visibility": "null",
                "coarse_type": "PER",
                "gold_subtype": self.taxonomy.labels[self.gold_id],
                "predicted_subtype": self.taxonomy.labels[
                    self.predicted_id
                ],
            },
        ]
        summary = summarize_seed_rows(
            rows,
            top_ks=[1, 2],
            formal_prediction_count=10,
            gmner_correct_count=8,
        )
        self.assertEqual(summary["visible_subtype_errors"], 1.0)
        self.assertEqual(summary["null_subtype_errors"], 1.0)
        self.assertEqual(
            summary["formal_region_probe"]["pairwise_support_count"],
            0.0,
        )
        self.assertEqual(
            summary["fine_top_k_positive_oracle"]["2"][
                "pairwise_support_count"
            ],
            1.0,
        )
        self.assertEqual(
            summary["fine_top_k_positive_oracle"]["2"][
                "incremental_pairwise_recovery_over_formal"
            ],
            1.0,
        )

    def test_invalid_features_do_not_create_visual_prototypes(self):
        bank = build_visual_prototype_bank(
            [
                (self.gold_id, torch.zeros(2)),
                (
                    self.predicted_id,
                    torch.tensor([float("nan"), 0.0]),
                ),
            ],
            num_subtypes=self.taxonomy.num_subtypes,
            feature_size=2,
        )
        self.assertEqual(int(bank.available.sum().item()), 0)

    def test_visible_analysis_rejects_a_nonpositive_formal_region(self):
        with self.assertRaises(ValueError):
            analyze_visible_error(
                formal_region_index=0,
                fine_ranked_region_indices=[0, 1],
                positive_region_indices={1},
                all_real_region_indices=[0, 1],
                region_features=torch.eye(2),
                bank=self.build_bank(),
                taxonomy=self.taxonomy,
                parent_id=self.parent_id,
                gold_subtype_id=self.gold_id,
                predicted_subtype_id=self.predicted_id,
                top_ks=[1, 2],
            )


if __name__ == "__main__":
    unittest.main()
