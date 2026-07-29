from __future__ import annotations

import inspect

import torch
from torch.utils.data import DataLoader

from gmner.data.stage1_candidate_selector import (
    Stage1CandidateSelectorCollator,
)
from gmner.engine.stage1_candidate_selector_evaluator import (
    decode_stage1_candidate_record,
    evaluate_stage1_candidate_selector,
)
from gmner.losses.stage1_candidate_selector_loss import (
    stage1_candidate_selector_loss,
    stage1_candidate_selector_supervision,
)
from gmner.models.stage1_candidate_selector import (
    Stage1CandidateSelector,
    Stage1CandidateSelectorConfig,
)
from scripts.train_stage1_candidate_selector import parse_args


def _record() -> dict:
    return {
        "span_candidates": torch.tensor([[0, 1], [2, 3], [4, 5]]),
        "span_mask": torch.tensor([True, True, True]),
        "span_features": torch.zeros(3, 4, dtype=torch.float16),
        "span_base_scores": torch.tensor([2.0, 1.0, 0.5]),
        "span_source_ids": torch.tensor([0, 0, 2]),
        "span_lengths": torch.tensor([1, 1, 1]),
        "type_candidates": torch.tensor([[1, 0], [2, 0], [3, 1]]),
        "type_base_scores": torch.zeros(3, 2),
        "fixed_type_ids": torch.tensor([1, 2, 3]),
        "base_region_indices": torch.tensor([4, 5, 6]),
        "gold_span_mask": torch.tensor([True, False, True]),
        "gold_type_mask": torch.tensor(
            [[True, False], [False, False], [True, False]]
        ),
        "formal_candidate_mask": torch.tensor([True, True, False]),
        "metadata": {
            "record_id": "one",
            "candidate_sources": ["stage1", "stage1", "kbest"],
            "stage1_predictions": [
                {"span": [0, 1], "type_id": 1, "region_index": 4},
                {"span": [2, 3], "type_id": 2, "region_index": 5},
            ],
            "gold_entities": [
                {
                    "span": [0, 1],
                    "type_id": 1,
                    "region_positive_indices": [4],
                },
                {
                    "span": [4, 5],
                    "type_id": 3,
                    "region_positive_indices": [6],
                },
            ],
            "null_region_index": 6,
        },
    }


def _model() -> Stage1CandidateSelector:
    return Stage1CandidateSelector(
        Stage1CandidateSelectorConfig(
            input_size=4,
            hidden_size=8,
            num_sources=4,
            dropout=0.0,
        )
    )


def test_epoch0_exactly_reproduces_formal_stage1_predictions() -> None:
    loader = DataLoader(
        [_record()],
        batch_size=1,
        collate_fn=Stage1CandidateSelectorCollator(),
    )
    metrics = evaluate_stage1_candidate_selector(
        _model(),
        loader,
        torch.device("cpu"),
    )
    assert metrics["prediction_set_equal_to_stage1"] is True
    assert metrics["formal_selected_count"] == 2.0
    assert metrics["nonformal_selected_count"] == 0.0
    assert metrics["span_f1_delta"] == 0.0
    assert metrics["gmner_f1_delta"] == 0.0


def test_disabled_mode_is_an_exact_noop() -> None:
    loader = DataLoader(
        [_record()],
        batch_size=1,
        collate_fn=Stage1CandidateSelectorCollator(),
    )
    metrics = evaluate_stage1_candidate_selector(
        _model(),
        loader,
        torch.device("cpu"),
        disabled=True,
    )
    assert metrics["prediction_sha256"] == metrics["stage1_prediction_sha256"]
    assert metrics["prediction_set_equal_to_stage1"] is True


def test_nonformal_promotion_and_formal_rejection_are_reachable() -> None:
    record = _record()
    selected, predictions = decode_stage1_candidate_record(
        spans=record["span_candidates"],
        span_mask=record["span_mask"],
        utility=torch.tensor([0.5, -0.1, 0.6]),
        formal_mask=record["formal_candidate_mask"],
        fixed_type_ids=record["fixed_type_ids"],
        type_candidates=record["type_candidates"],
        base_region_indices=record["base_region_indices"],
    )
    assert selected == [0, 2]
    assert [prediction["type_id"] for prediction in predictions] == [1, 3]
    assert predictions[1]["region_index"] == 6


def test_decode_has_no_gold_input_and_uses_stored_promoted_type() -> None:
    parameters = inspect.signature(decode_stage1_candidate_record).parameters
    assert all("gold" not in name for name in parameters)
    record = _record()
    _, predictions = decode_stage1_candidate_record(
        spans=record["span_candidates"][2:],
        span_mask=torch.tensor([True]),
        utility=torch.tensor([0.6]),
        formal_mask=torch.tensor([False]),
        fixed_type_ids=torch.tensor([0]),
        type_candidates=torch.tensor([[3, 1]]),
        base_region_indices=torch.tensor([6]),
    )
    assert predictions == [
        {"span": [4, 5], "type_id": 3, "region_index": 6}
    ]


def test_supervision_includes_every_non_gold_candidate() -> None:
    batch = Stage1CandidateSelectorCollator()([_record()])
    supervision = stage1_candidate_selector_supervision(batch)
    assert supervision["targets"].tolist() == [[1.0, 0.0, 1.0]]
    assert supervision["candidate_weights"].tolist() == [[3.0, 1.5, 2.0]]
    assert supervision["valid_mask"].sum().item() == 3


def test_overlap_margin_has_nonzero_utility_gradient() -> None:
    batch = {
        "span_candidates": torch.tensor([[[0, 2], [1, 3]]]),
        "span_mask": torch.tensor([[True, True]]),
        "gold_span_mask": torch.tensor([[True, False]]),
        "formal_candidate_mask": torch.tensor([[True, False]]),
        "span_base_scores": torch.zeros(1, 2),
    }
    utility = torch.tensor([[0.0, 1.0]], requires_grad=True)
    outputs = {
        "utility": utility,
        "residual": torch.zeros(1, 2),
    }
    losses = stage1_candidate_selector_loss(
        outputs,
        batch,
        lambda_entity=0.0,
        lambda_overlap_margin=1.0,
        lambda_residual=0.0,
        overlap_margin=0.2,
    )
    losses["loss"].backward()
    assert losses["overlap_positive_pairs"].item() == 1
    assert utility.grad is not None
    assert utility.grad.abs().sum().item() > 0


def test_entity_loss_is_normalized_per_record() -> None:
    batch = {
        "span_candidates": torch.tensor(
            [[[0, 1], [0, 0]], [[0, 1], [2, 3]]]
        ),
        "span_mask": torch.tensor([[True, False], [True, True]]),
        "gold_span_mask": torch.tensor([[True, False], [True, True]]),
        "formal_candidate_mask": torch.tensor([[True, False], [True, True]]),
        "span_base_scores": torch.zeros(2, 2),
    }
    outputs = {
        "utility": torch.zeros(2, 2),
        "residual": torch.zeros(2, 2),
    }
    losses = stage1_candidate_selector_loss(
        outputs,
        batch,
        lambda_overlap_margin=0.0,
        lambda_residual=0.0,
    )
    assert torch.allclose(
        losses["loss_entity"],
        torch.log(torch.tensor(2.0)),
    )


def test_training_cli_has_no_test_option(monkeypatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["train_stage1_candidate_selector.py", "--config", "selector.yaml"],
    )
    args = parse_args()
    assert not hasattr(args, "test")
    assert not hasattr(args, "test_cache")
