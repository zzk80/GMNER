from __future__ import annotations

import torch
import pytest

from gmner.data.record_candidate_collator import RecordCandidateCollator
from gmner.data.record_candidate_dataset import (
    CACHE_FORMAT_VERSION,
    RecordCandidateDataset,
)
from gmner.models.structured_interval_decoder import (
    greedy_interval_decode,
    weighted_interval_decode,
)


def _record(spans: int, regions: int, hidden: int = 8) -> dict:
    types = 2
    positive = torch.zeros(spans, types, regions, dtype=torch.bool)
    positive[0, 0, 0] = True
    return {
        "span_candidates": torch.tensor([[index, index + 1] for index in range(spans)]),
        "span_mask": torch.ones(spans, dtype=torch.bool),
        "span_features": torch.randn(spans, hidden),
        "span_base_scores": torch.linspace(-0.1, -1.0, spans),
        "span_source_ids": torch.arange(spans) % 4,
        "span_lengths": torch.ones(spans),
        "type_candidates": torch.tensor([[0, 1]] * spans),
        "type_base_scores": torch.log_softmax(torch.randn(spans, types), dim=-1),
        "type_mask": torch.ones(spans, types, dtype=torch.bool),
        "region_features": torch.randn(regions, hidden),
        "region_boxes": torch.zeros(regions, 4),
        "region_geometry": torch.zeros(regions, 4),
        "region_detector_scores": torch.ones(regions),
        "region_base_scores": torch.log_softmax(torch.randn(spans, regions), dim=-1),
        "type_region_compatibility": torch.zeros(spans, types, regions),
        "region_mask": torch.ones(regions, dtype=torch.bool),
        "region_is_null": torch.tensor([False] * (regions - 1) + [True]),
        "image_global": torch.randn(hidden),
        "gold_span_mask": torch.tensor([True] + [False] * (spans - 1)),
        "gold_type_mask": torch.tensor([[True, False]] + [[False, False]] * (spans - 1)),
        "gold_region_positive_mask": positive.any(dim=1),
        "positive_triple_mask": positive,
        "visibility_targets": torch.tensor([1.0] + [-1.0] * (spans - 1)),
        "metadata": {"record_id": str(spans)},
    }


def test_record_collator_pads_all_candidate_axes() -> None:
    batch = RecordCandidateCollator()([_record(2, 3), _record(4, 5)])
    assert batch["span_features"].shape == (2, 4, 8)
    assert batch["type_candidates"].shape == (2, 4, 2)
    assert batch["positive_triple_mask"].shape == (2, 4, 2, 5)
    assert batch["region_features"].shape == (2, 5, 8)
    assert not batch["span_mask"][0, 2:].any()
    assert batch["region_is_null"][0, 2]
    assert not batch["region_is_null"][0, 3:].any()
    assert torch.equal(batch["visibility_targets"][0, 2:], torch.tensor([-1.0, -1.0]))


def test_interval_decoder_beats_greedy_on_overlapping_candidates() -> None:
    spans = [(0, 3), (0, 1), (1, 3)]
    scores = [5.0, 3.0, 3.0]
    assert greedy_interval_decode(spans, scores) == [0]
    assert weighted_interval_decode(spans, scores) == [1, 2]


def test_interval_decoder_filters_rejected_spans() -> None:
    assert weighted_interval_decode([(0, 1), (1, 2)], [0.0, -1.0]) == []


def test_candidate_cache_validates_fingerprints(tmp_path) -> None:
    path = tmp_path / "cache.pt"
    torch.save(
        {
            "metadata": {
                "format_version": CACHE_FORMAT_VERSION,
                "stage1_checkpoint_sha256": "stage1",
                "candidate_config_sha256": "candidate",
            },
            "records": [_record(1, 2)],
        },
        path,
    )
    dataset = RecordCandidateDataset(
        path,
        expected_stage1_sha256="stage1",
        expected_candidate_sha256="candidate",
    )
    assert len(dataset) == 1
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        RecordCandidateDataset(path, expected_stage1_sha256="other")


def test_candidate_dataset_keeps_v1_cache_compatibility(tmp_path) -> None:
    path = tmp_path / "legacy_cache.pt"
    torch.save(
        {
            "metadata": {"format_version": 1},
            "records": [_record(1, 2)],
        },
        path,
    )
    assert len(RecordCandidateDataset(path)) == 1
