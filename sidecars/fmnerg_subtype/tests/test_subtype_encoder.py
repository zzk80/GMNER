from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from sidecars.fmnerg_subtype.encoder_config import (
    load_subtype_encoder_config,
)
from sidecars.fmnerg_subtype.encoder_model import (
    TrainableSubtypeEncoder,
    build_optimizer_groups,
    configure_backbone_trainability,
    load_trainable_checkpoint_state,
    pool_online_span_features,
    trainable_checkpoint_state,
)
from sidecars.fmnerg_subtype.final_test import (
    FINAL_TEST_SEEDS,
    aggregate_seed_metrics,
    load_final_test_protocol,
)
from sidecars.fmnerg_subtype.online_data import (
    OnlineSubtypeCollator,
    OnlineSubtypeRecordDataset,
)
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy
from tools.summarize_fmnerg_subtype_encoder_ablation import (
    load_frozen_f0_runs,
)


ROOT = Path(__file__).resolve().parents[3]
TAXONOMY = SubtypeTaxonomy.from_file(
    ROOT / "sidecars" / "fmnerg_subtype" / "taxonomy_twitter10000.json"
)


class FakeEncoding(dict):
    def __init__(self, values, word_ids):
        super().__init__(values)
        self._word_ids = word_ids

    def word_ids(self, batch_index):
        return self._word_ids[batch_index]


class FakeFastTokenizer:
    is_fast = True

    def __call__(self, token_lists, **kwargs):
        max_tokens = max(len(tokens) for tokens in token_lists)
        sequence_length = max_tokens + 2
        input_ids = []
        attention_mask = []
        word_ids = []
        for tokens in token_lists:
            ids = [0] + list(range(1, len(tokens) + 1)) + [99]
            words = [None] + list(range(len(tokens))) + [None]
            padding = sequence_length - len(ids)
            input_ids.append(ids + [0] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
            word_ids.append(words + [None] * padding)
        return FakeEncoding(
            {
                "input_ids": torch.tensor(input_ids),
                "attention_mask": torch.tensor(attention_mask),
            },
            word_ids,
        )


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int, layers: int):
        super().__init__()
        self.layer = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(layers)]
        )


class FakeBackbone(nn.Module):
    def __init__(self, hidden_size: int = 4, layers: int = 6):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embeddings = nn.Embedding(128, hidden_size)
        self.encoder = FakeEncoder(hidden_size, layers)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        hidden = self.embeddings(input_ids)
        for layer in self.encoder.layer:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class SubtypeEncoderTest(unittest.TestCase):
    def test_online_configs_isolate_formal_chain_and_set_scopes(self):
        last4 = load_subtype_encoder_config(
            ROOT
            / "sidecars"
            / "fmnerg_subtype"
            / "roberta128_encoder_last4.yaml"
        )
        full = load_subtype_encoder_config(
            ROOT
            / "sidecars"
            / "fmnerg_subtype"
            / "roberta128_encoder_all.yaml"
        )
        self.assertEqual(last4.model.encoder_scope, "last_n")
        self.assertEqual(last4.model.unfreeze_last_n_layers, 4)
        self.assertEqual(full.model.encoder_scope, "all")
        self.assertEqual(full.optim.backbone_lower_learning_rate, 1e-6)
        self.assertEqual(full.optim.backbone_upper_learning_rate, 5e-6)
        self.assertEqual(full.optim.head_learning_rate, 1e-4)
        self.assertFalse(hasattr(last4.data, "test_source"))
        self.assertFalse(hasattr(full.data, "test_source"))

    def test_online_collator_keeps_global_span_alignment(self):
        dataset = OnlineSubtypeRecordDataset(
            [
                {
                    "record_id": "0",
                    "tokens": ["A", "B", "C"],
                    "spans": [
                        {
                            "start": 0,
                            "end": 2,
                            "coarse_type_id": 1,
                            "subtype_id": TAXONOMY.subtype_id("athlete"),
                        }
                    ],
                },
                {
                    "record_id": "1",
                    "tokens": ["D", "E"],
                    "spans": [
                        {
                            "start": 1,
                            "end": 2,
                            "coarse_type_id": 2,
                            "subtype_id": TAXONOMY.subtype_id("company"),
                        }
                    ],
                },
            ]
        )
        batch = OnlineSubtypeCollator(
            FakeFastTokenizer(),
            max_length=16,
        )([dataset[0], dataset[1]])
        self.assertEqual(batch["example_indices"].tolist(), [0, 1])
        self.assertEqual(batch["span_record_indices"].tolist(), [0, 1])
        self.assertEqual(batch["span_start_indices"].tolist(), [1, 2])
        self.assertEqual(batch["span_end_indices"].tolist(), [2, 2])
        self.assertEqual(
            batch["span_token_mask"].sum(dim=-1).tolist(),
            [2, 1],
        )

    def test_span_pooling_matches_start_end_mean_contract(self):
        hidden = torch.tensor(
            [
                [
                    [0.0, 0.0],
                    [1.0, 2.0],
                    [3.0, 4.0],
                    [5.0, 6.0],
                ]
            ]
        )
        pooled = pool_online_span_features(
            hidden,
            span_record_indices=torch.tensor([0]),
            span_start_indices=torch.tensor([1]),
            span_end_indices=torch.tensor([2]),
            span_token_mask=torch.tensor(
                [[False, True, True, False]]
            ),
        )
        expected = torch.tensor([[1.0, 2.0, 3.0, 4.0, 2.0, 3.0]])
        self.assertTrue(torch.equal(pooled, expected))

    def test_last_four_layers_and_full_scope_are_distinct(self):
        last4 = FakeBackbone()
        report = configure_backbone_trainability(
            last4,
            scope="last_n",
            last_n_layers=4,
            gradient_checkpointing=False,
        )
        self.assertEqual(report["trainable_layer_indices"], [2, 3, 4, 5])
        self.assertFalse(last4.embeddings.weight.requires_grad)
        self.assertFalse(last4.encoder.layer[1].weight.requires_grad)
        self.assertTrue(last4.encoder.layer[2].weight.requires_grad)

        full = FakeBackbone()
        report = configure_backbone_trainability(
            full,
            scope="all",
            last_n_layers=4,
            gradient_checkpointing=False,
        )
        self.assertEqual(report["trainable_layer_indices"], list(range(6)))
        self.assertTrue(
            all(parameter.requires_grad for parameter in full.parameters())
        )

    def test_optimizer_groups_and_delta_checkpoint_do_not_overlap(self):
        config = load_subtype_encoder_config(
            ROOT
            / "sidecars"
            / "fmnerg_subtype"
            / "roberta128_encoder_last4.yaml"
        )
        backbone = FakeBackbone()
        configure_backbone_trainability(
            backbone,
            scope="last_n",
            last_n_layers=4,
            gradient_checkpointing=False,
        )
        model = TrainableSubtypeEncoder(
            backbone=backbone,
            taxonomy=TAXONOMY,
            input_size=12,
            hidden_size=8,
            dropout=0.0,
            head_architecture="shared_hard",
            parent_hidden_size=2,
        )
        groups, report = build_optimizer_groups(model, config)
        self.assertNotIn("backbone_lower", report)
        self.assertIn("backbone_upper", report)
        self.assertIn("subtype_head", report)
        identifiers = [
            id(parameter)
            for group in groups
            for parameter in group["params"]
        ]
        self.assertEqual(len(identifiers), len(set(identifiers)))

        state = trainable_checkpoint_state(model)
        clone_backbone = FakeBackbone()
        configure_backbone_trainability(
            clone_backbone,
            scope="last_n",
            last_n_layers=4,
            gradient_checkpointing=False,
        )
        clone = TrainableSubtypeEncoder(
            backbone=clone_backbone,
            taxonomy=TAXONOMY,
            input_size=12,
            hidden_size=8,
            dropout=0.0,
            head_architecture="shared_hard",
            parent_hidden_size=2,
        )
        load_trainable_checkpoint_state(clone, state)
        for name in state["backbone_trainable_names"]:
            self.assertTrue(
                torch.equal(
                    model.backbone.state_dict()[name],
                    clone.backbone.state_dict()[name],
                )
            )

    def test_frozen_f0_runs_are_loaded_per_seed(self):
        metrics = {
            "fine_mner_f1": 0.62,
            "fmnerg_f1": 0.48,
            "subtype_accuracy_on_gold_spans": 0.7,
            "subtype_macro_f1_on_gold_spans": 0.53,
            "parent_conditioned_subtype_accuracy": 0.75,
            "gmner_f1": 0.621316,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in (41, 42, 43):
                run = root / f"frozen_seed{seed}"
                run.mkdir()
                (run / "dev_metrics.json").write_text(
                    json.dumps(
                        {
                            "metadata": {
                                "checkpoint_epoch": seed,
                                "gmner_identity_exact": True,
                                "test_accessed": False,
                            },
                            "metrics": metrics,
                        }
                    ),
                    encoding="utf-8",
                )
            runs = load_frozen_f0_runs(root, [41, 42, 43])
        self.assertEqual([run["seed"] for run in runs], [41, 42, 43])
        self.assertTrue(
            all(
                run["metrics"]["gmner_f1"] == 0.621316
                for run in runs
            )
        )

    def test_final_test_protocol_is_frozen_before_access(self):
        protocol = load_final_test_protocol(
            ROOT
            / "sidecars"
            / "fmnerg_subtype"
            / "roberta128_encoder_final_test.yaml",
            ROOT,
        )
        self.assertEqual(
            tuple(protocol["method"]["seeds"]),
            FINAL_TEST_SEEDS,
        )
        self.assertEqual(protocol["method"]["encoder_scope"], "all")
        self.assertEqual(protocol["method"]["selection_source"], "dev")
        self.assertEqual(protocol["method"]["report"], "mean_std")
        self.assertFalse(protocol["method"]["select_best_seed_on_test"])
        self.assertEqual(
            [item["seed"] for item in protocol["checkpoints"]],
            list(FINAL_TEST_SEEDS),
        )

    def test_final_test_aggregation_has_no_seed_selection(self):
        rows = [{"fmnerg_f1": value} for value in (0.5, 0.52, 0.51)]
        result = aggregate_seed_metrics(rows, ("fmnerg_f1",))
        self.assertAlmostEqual(result["fmnerg_f1"]["mean"], 0.51)
        self.assertIn("std", result["fmnerg_f1"])
        self.assertNotIn("best", result["fmnerg_f1"])

    def test_frozen_final_test_result_preserves_main_chain(self):
        path = (
            ROOT
            / "sidecars"
            / "fmnerg_subtype"
            / "roberta128_encoder_final_test_result.json"
        )
        result = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(result["test_access_count"], 1)
        self.assertFalse(result["select_best_seed_on_test"])
        self.assertEqual(
            [row["seed"] for row in result["per_seed"]],
            list(FINAL_TEST_SEEDS),
        )
        self.assertAlmostEqual(
            result["fixed_main_chain_metrics"]["gmner_f1"],
            0.6152941176470589,
        )
        self.assertNotIn("best_seed", result)


if __name__ == "__main__":
    unittest.main()
