from __future__ import annotations

import copy

import pytest
import torch

from scripts.aggregate_m33a_oof_metrics import aggregate_m33a_oof_batches


def _batch(fold_id: int, record_id: str, *, type_correct: bool) -> dict:
    # One selected gold span and one selected false-positive span.
    span_mask = torch.tensor([[True, True]])
    selected = span_mask.clone()
    region_mask = torch.tensor([[True, True, True]])
    region_is_null = torch.tensor([[False, False, True]])
    candidate_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    final_logits = torch.tensor([[[4.0, 1.0, -2.0], [3.0, 1.0, -2.0]]])
    fixed_type = 0 if type_correct else 1
    type_candidates = torch.tensor([[[0, 1], [0, 1]]])
    gold_type_mask = torch.tensor([[[True, False], [False, False]]])
    gold_regions = torch.tensor(
        [[[True, False, False], [False, False, False]]]
    )
    return {
        "fold_id": fold_id,
        "record_ids": [record_id],
        "fine_outputs": {
            "candidate_mask": candidate_mask,
            "final_region_logits": final_logits,
            "fixed_type_ids": torch.tensor([[fixed_type, 0]]),
        },
        "hierarchy_outputs": {
            "fixed_type_ids": torch.tensor([[fixed_type, 0]]),
        },
        "expanded": {
            "span_mask": span_mask,
            "span_source_ids": torch.zeros(1, 2, dtype=torch.long),
            "gold_span_mask": torch.tensor([[True, False]]),
            "type_candidates": type_candidates,
            "gold_type_mask": gold_type_mask,
            "gold_region_positive_mask": gold_regions,
            "region_mask": region_mask,
            "region_is_null": region_is_null,
        },
        "current_visible": torch.tensor([[True, True]]),
        "deployment_span_mask": selected,
    }


def test_aggregate_m33a_oof_metrics_uses_cached_formal_predictions() -> None:
    result = aggregate_m33a_oof_batches(
        [
            _batch(0, "0", type_correct=True),
            _batch(1, "1", type_correct=False),
        ],
        {"0": 1, "1": 1},
        expected_folds=2,
        expected_records=2,
    )

    assert result["counts"]["predicted"] == 4
    assert result["counts"]["gold"] == 2
    assert result["counts"]["span_correct"] == 2
    assert result["counts"]["entity_correct"] == 1
    assert result["counts"]["eeg_correct"] == 2
    assert result["counts"]["triple_correct"] == 1
    assert result["metrics"]["span_f1"] == pytest.approx(2.0 / 3.0)
    assert result["metrics"]["mner_f1"] == pytest.approx(1.0 / 3.0)
    assert result["metrics"]["eeg_f1"] == pytest.approx(2.0 / 3.0)
    assert result["metrics"]["gmner_score"] == pytest.approx(1.0 / 3.0)


def test_aggregate_m33a_oof_metrics_rejects_duplicate_records() -> None:
    first = _batch(0, "0", type_correct=True)
    duplicate = copy.deepcopy(_batch(1, "0", type_correct=True))
    with pytest.raises(ValueError, match="appear more than once"):
        aggregate_m33a_oof_batches(
            [first, duplicate],
            {"0": 1},
            expected_folds=2,
            expected_records=1,
        )


def test_aggregate_m33a_oof_metrics_requires_exact_train_id_coverage() -> None:
    with pytest.raises(ValueError, match="exactly equal source train ids"):
        aggregate_m33a_oof_batches(
            [
                _batch(0, "0", type_correct=True),
                _batch(1, "1", type_correct=True),
            ],
            {"0": 1, "1": 1, "2": 1},
            expected_folds=2,
            expected_records=2,
        )
