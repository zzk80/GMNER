"""Tests for the Train-only Stage1 D0 gradient audit."""

from __future__ import annotations

import unittest

import torch
from torch import nn

from gmner.diagnostics.stage1_gradient_conflicts import (
    aggregate_gradient_observations,
    compute_gradient_observation,
    encoder_layer_parameter_groups,
    stable_probe_record_ids,
)


class _ToyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList(
            [nn.Linear(2, 2, bias=False) for _ in range(3)]
        )
        for layer in self.layer:
            nn.init.eye_(layer.weight)


class _ToyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _ToyEncoder()


class _ToyTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = _ToyBackbone()


class _ToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_encoder = _ToyTextEncoder()

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = value
        for layer in self.text_encoder.backbone.encoder.layer:
            output = layer(output)
        return output


class Stage1GradientConflictTest(unittest.TestCase):
    def test_probe_selection_is_deterministic_and_order_independent(self) -> None:
        first = stable_probe_record_ids(range(20), count=7, seed=42)
        second = stable_probe_record_ids(reversed(range(20)), count=7, seed=42)
        other_seed = stable_probe_record_ids(range(20), count=7, seed=43)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)
        self.assertNotEqual(first, other_seed)

    def test_opposite_losses_produce_negative_cosine(self) -> None:
        model = _ToyModel()
        groups = encoder_layer_parameter_groups(model, [0, 2])
        output = model(torch.tensor([[1.0, -2.0]]))
        ner = output.sum()
        grounding = -output.sum()
        alignment = output.square().sum()

        observation = compute_gradient_observation(
            {
                "ner": ner,
                "grounding": grounding,
                "alignment": alignment,
            },
            groups,
        )

        for layer_name in ("layer_0", "layer_2"):
            pair = observation["layers"][layer_name]["pairs"][
                "ner_vs_grounding"
            ]
            self.assertAlmostEqual(pair["cosine"], -1.0, places=6)
            self.assertAlmostEqual(pair["norm_ratio"], 1.0, places=6)

    def test_missing_task_is_reported_without_inventing_zero_gradient(self) -> None:
        model = _ToyModel()
        groups = encoder_layer_parameter_groups(model, [1])
        output = model(torch.tensor([[1.0, 2.0]]))
        observation = compute_gradient_observation(
            {"ner": output.sum(), "grounding": -output.sum()},
            groups,
        )

        self.assertEqual(observation["skipped_tasks"]["alignment"], "missing")
        self.assertNotIn(
            "ner_vs_alignment",
            observation["layers"]["layer_1"]["pairs"],
        )

    def test_registered_gate_detects_repeated_strong_conflict(self) -> None:
        model = _ToyModel()
        groups = encoder_layer_parameter_groups(model, [0])
        observations = []
        for _ in range(3):
            output = model(torch.tensor([[1.0, -2.0]]))
            observations.append(
                compute_gradient_observation(
                    {
                        "ner": output.sum(),
                        "grounding": -output.sum(),
                        "alignment": output.square().sum(),
                    },
                    groups,
                )
            )

        summary = aggregate_gradient_observations(
            observations,
            min_valid_batches=3,
        )
        pair = summary["layers"]["layer_0"]["pairs"][
            "ner_vs_grounding"
        ]
        self.assertTrue(pair["significant_conflict"])
        self.assertEqual(
            summary["recommendation"]["status"],
            "significant_conflict",
        )
        self.assertTrue(summary["recommendation"]["recommend_d2"])

    def test_insufficient_batches_do_not_enable_d2(self) -> None:
        model = _ToyModel()
        groups = encoder_layer_parameter_groups(model, [0])
        output = model(torch.tensor([[1.0, -2.0]]))
        observation = compute_gradient_observation(
            {
                "ner": output.sum(),
                "grounding": -output.sum(),
                "alignment": output.square().sum(),
            },
            groups,
        )
        summary = aggregate_gradient_observations(
            [observation],
            min_valid_batches=2,
        )

        self.assertEqual(
            summary["recommendation"]["status"],
            "insufficient_data",
        )
        self.assertFalse(summary["recommendation"]["recommend_d2"])


if __name__ == "__main__":
    unittest.main()
