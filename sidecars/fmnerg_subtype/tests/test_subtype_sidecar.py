from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from sidecars.fmnerg_subtype.data import (
    FEATURE_CACHE_KIND,
    FEATURE_CACHE_VERSION,
    SubtypeFeatureDataset,
)
from sidecars.fmnerg_subtype.evaluator import (
    evaluate_formal_predictions,
    load_formal_predictions,
    validate_expected_frozen_gmner,
)
from sidecars.fmnerg_subtype.losses import build_subtype_class_weights
from sidecars.fmnerg_subtype.metrics import (
    canonical_coarse_prediction_sha256,
    coarse_end_to_end_metrics,
)
from sidecars.fmnerg_subtype.model import HierarchicalSubtypeSidecar
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy
from tools.analyze_fmnerg_subtype_errors import group_metrics


ROOT = Path(__file__).resolve().parents[3]
TAXONOMY_PATH = (
    ROOT / "sidecars" / "fmnerg_subtype" / "taxonomy_twitter10000.json"
)


class FixedSubtypeModel(nn.Module):
    def forward(self, features, coarse_type_ids=None):
        predicted = features[:, 0].long()
        return {"predicted_subtype_ids": predicted}


class SubtypeSidecarTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = SubtypeTaxonomy.from_file(TAXONOMY_PATH)

    def test_taxonomy_has_expected_hierarchy(self):
        self.assertEqual(self.taxonomy.num_subtypes, 51)
        coarse = torch.arange(4)
        counts = self.taxonomy.allowed_mask(coarse).sum(dim=-1).tolist()
        self.assertEqual(counts, [11, 13, 10, 17])

    def test_error_report_includes_zero_prediction_subtype_f1(self):
        metrics = group_metrics(Counter({"gold": 3}))

        self.assertEqual(metrics["predicted"], 0.0)
        self.assertEqual(metrics["subtype_precision"], 0.0)
        self.assertEqual(metrics["subtype_recall"], 0.0)
        self.assertEqual(metrics["subtype_f1"], 0.0)
        self.assertEqual(metrics["prediction_to_gold_ratio"], 0.0)

    def test_model_never_predicts_outside_parent(self):
        model = HierarchicalSubtypeSidecar(
            input_size=4,
            hidden_size=8,
            dropout=0.0,
            taxonomy=self.taxonomy,
        )
        output = model(torch.randn(4, 4), torch.arange(4))
        predicted = output["predicted_subtype_ids"].tolist()
        self.assertEqual(
            [self.taxonomy.parent_id(value) for value in predicted],
            [0, 1, 2, 3],
        )

    def test_parent_specific_heads_are_isolated_and_parameter_matched(self):
        shared = HierarchicalSubtypeSidecar(
            input_size=2304,
            hidden_size=768,
            dropout=0.0,
            taxonomy=self.taxonomy,
            head_architecture="shared_hard",
            parent_hidden_size=192,
        )
        parent_specific = HierarchicalSubtypeSidecar(
            input_size=2304,
            hidden_size=768,
            dropout=0.0,
            taxonomy=self.taxonomy,
            head_architecture="parent_specific_hard",
            parent_hidden_size=192,
        )
        shared_count = sum(parameter.numel() for parameter in shared.parameters())
        parent_count = sum(
            parameter.numel() for parameter in parent_specific.parameters()
        )
        self.assertLess(abs(parent_count - shared_count) / shared_count, 0.02)

        small = HierarchicalSubtypeSidecar(
            input_size=4,
            hidden_size=8,
            dropout=0.0,
            taxonomy=self.taxonomy,
            head_architecture="parent_specific_hard",
            parent_hidden_size=2,
        )
        parent_id = 1
        target = self.taxonomy.subtype_id("athlete")
        outputs = small(torch.randn(3, 4), torch.full((3,), parent_id))
        F.cross_entropy(
            outputs["logits"],
            torch.full((3,), target),
        ).backward()
        for index, classifier in enumerate(small.parent_classifiers):
            gradient_norm = sum(
                float(parameter.grad.abs().sum().item())
                for parameter in classifier.parameters()
                if parameter.grad is not None
            )
            if index == parent_id:
                self.assertGreater(gradient_norm, 0.0)
            else:
                self.assertEqual(gradient_norm, 0.0)

    def test_soft_parent_prior_is_constant_under_hard_parent_mask(self):
        logits = torch.randn(4, self.taxonomy.num_subtypes)
        coarse_ids = torch.arange(4)
        coarse_log_probs = torch.log_softmax(torch.randn(4, 4), dim=-1)
        parent_ids = torch.tensor(self.taxonomy.parent_ids)
        prior = coarse_log_probs[:, parent_ids]

        original = self.taxonomy.mask_logits(logits, coarse_ids).argmax(dim=-1)
        adjusted = self.taxonomy.mask_logits(
            logits + prior,
            coarse_ids,
        ).argmax(dim=-1)

        self.assertTrue(torch.equal(original, adjusted))

    def test_preregistered_frozen_gmner_gate(self):
        payload = {
            "metadata": {
                "coarse_metrics": {"gmner_f1": 0.6213161082},
            }
        }
        validate_expected_frozen_gmner(
            payload,
            expected=0.621316,
            tolerance=5e-7,
        )
        with self.assertRaises(ValueError):
            validate_expected_frozen_gmner(
                payload,
                expected=0.62,
                tolerance=5e-7,
            )

    def test_weighted_losses_are_parent_normalized(self):
        labels = torch.cat(
            [
                torch.full((index + 1,), index, dtype=torch.long)
                for index in range(self.taxonomy.num_subtypes)
            ]
        )
        for mode in ("class_weighted", "effective_number"):
            weights, report = build_subtype_class_weights(
                labels,
                taxonomy=self.taxonomy,
                mode=mode,
                effective_number_beta=0.999,
                parent_normalize=True,
            )
            self.assertIsNotNone(weights)
            for mean in report["parent_weight_means"].values():
                self.assertAlmostEqual(mean, 1.0, places=6)
            for parent_id in range(4):
                members = [
                    index
                    for index, value in enumerate(self.taxonomy.parent_ids)
                    if value == parent_id
                ]
                self.assertGreater(weights[min(members)], weights[max(members)])

    def test_formal_sidecar_preserves_every_coarse_prediction(self):
        athlete = self.taxonomy.subtype_id("athlete")
        company = self.taxonomy.subtype_id("company")
        records = [
            {
                "record_id": "0",
                "predictions": [
                    {"span": [0, 1], "type_id": 1, "region_index": 2}
                ],
                "gold_entities": [
                    {
                        "span": [0, 1],
                        "type_id": 1,
                        "subtype_id": athlete,
                        "region_positive_indices": [2],
                    }
                ],
            },
            {
                "record_id": "1",
                "predictions": [
                    {"span": [2, 4], "type_id": 2, "region_index": 36}
                ],
                "gold_entities": [
                    {
                        "span": [2, 4],
                        "type_id": 2,
                        "subtype_id": company,
                        "region_positive_indices": [36],
                    }
                ],
            },
        ]
        digest = canonical_coarse_prediction_sha256(records)
        coarse_metrics = coarse_end_to_end_metrics(records)
        formal_payload = {
            "metadata": {
                "kind": "fmnerg_frozen_formal_predictions",
                "format_version": 1,
                "split": "dev",
                "taxonomy_sha256": self.taxonomy.source_sha256,
                "coarse_prediction_sha256": digest,
                "coarse_metrics": coarse_metrics,
                "test_accessed": False,
            },
            "records": records,
        }
        feature_payload = {
            "metadata": {
                "kind": FEATURE_CACHE_KIND,
                "format_version": FEATURE_CACHE_VERSION,
                "split": "dev",
                "mode": "formal",
                "taxonomy_sha256": self.taxonomy.source_sha256,
                "stage1_checkpoint_sha256": "synthetic",
                "coarse_prediction_sha256": digest,
                "test_accessed": False,
            },
            "features": torch.tensor(
                [[float(athlete)], [float(company)]],
                dtype=torch.float32,
            ),
            "coarse_type_ids": torch.tensor([1, 2]),
            "subtype_ids": torch.tensor([athlete, company]),
            "examples": [
                {
                    "record_id": "0",
                    "record_index": 0,
                    "prediction_index": 0,
                    "span": [0, 1],
                    "coarse_type_id": 1,
                },
                {
                    "record_id": "1",
                    "record_index": 1,
                    "prediction_index": 0,
                    "span": [2, 4],
                    "coarse_type_id": 2,
                },
            ],
        }
        result = evaluate_formal_predictions(
            FixedSubtypeModel(),
            SubtypeFeatureDataset(feature_payload),
            formal_payload,
            taxonomy=self.taxonomy,
            batch_size=2,
            device=torch.device("cpu"),
            include_records=True,
        )
        self.assertTrue(result["metadata"]["gmner_identity_exact"])
        self.assertEqual(
            result["metadata"]["coarse_prediction_sha256_before"],
            result["metadata"]["coarse_prediction_sha256_after"],
        )
        self.assertEqual(result["metrics"]["gmner_f1"], 1.0)
        self.assertEqual(result["metrics"]["fine_mner_f1"], 1.0)
        self.assertEqual(result["metrics"]["fmnerg_f1"], 1.0)
        self.assertEqual(result["metrics"]["hierarchy_consistency_rate"], 1.0)
        self.assertEqual(result["metadata"]["split"], "dev")
        self.assertFalse(result["metadata"]["test_accessed"])

    def test_test_formal_predictions_require_explicit_release(self):
        records = []
        payload = {
            "metadata": {
                "kind": "fmnerg_frozen_formal_predictions",
                "format_version": 1,
                "split": "test",
                "taxonomy_sha256": self.taxonomy.source_sha256,
                "coarse_prediction_sha256": (
                    canonical_coarse_prediction_sha256(records)
                ),
                "coarse_metrics": coarse_end_to_end_metrics(records),
                "test_accessed": True,
            },
            "records": records,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_formal_predictions(path, taxonomy=self.taxonomy)
            loaded = load_formal_predictions(
                path,
                taxonomy=self.taxonomy,
                expected_split="test",
            )
        self.assertTrue(loaded["metadata"]["test_accessed"])


if __name__ == "__main__":
    unittest.main()
