from __future__ import annotations

import torch
from torch import nn

from gmner.engine.fine_grounding_adapter_evaluator import (
    map_formal_regions_to_expanded,
)
from gmner.data.paired_record_candidate_dataset import (
    PairedRecordCandidateDataset,
)
from gmner.data.record_candidate_dataset import RecordCandidateDataset
from gmner.losses.fine_grounding_adapter_loss import (
    fine_grounding_adapter_loss,
    fine_grounding_supervision,
)
from gmner.models.fine_grounding_adapter import (
    SOURCE_BASE_ONLY,
    SOURCE_LEARNED_ONLY,
    CorrectionPreservationGroundingAdapter,
    FineGroundingAdapterConfig,
    normalized_masked_rank,
)


class _FixedCoarse(nn.Module):
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        real = (
            batch["region_mask"].bool()
            & ~batch["region_is_null"].bool()
        )[:, None, :].expand(-1, batch["span_mask"].size(1), -1)
        logits = torch.tensor(
            [[[0.0, 1.0, 8.0, 2.0, -1e4], [2.0, 1.0, 8.0, 0.0, -1e4]]],
            device=batch["span_mask"].device,
        )
        return {
            "coarse_logits": logits,
            "real_region_mask": real,
            "base_region_scores": batch["base_region_scores"].float(),
        }


def _batch() -> dict[str, torch.Tensor]:
    return {
        "span_mask": torch.tensor([[True, True]]),
        "span_features": torch.randn(1, 2, 4),
        "span_source_ids": torch.tensor([[0, 0]]),
        "gold_span_mask": torch.tensor([[True, True]]),
        "fixed_type_ids": torch.tensor([[0, 1]]),
        "type_candidates": torch.tensor([[[0, 2], [1, 3]]]),
        "region_mask": torch.tensor([[True, True, True, True, True]]),
        "region_is_null": torch.tensor([[False, False, False, False, True]]),
        "region_features": torch.randn(1, 5, 4),
        "base_region_scores": torch.tensor(
            [[[4.0, 3.0, 1.0, 0.0, -2.0], [0.0, 4.0, 3.0, 1.0, -2.0]]]
        ),
        "region_detector_scores": torch.tensor([[0.9, 0.8, 0.7, 0.6, 1.0]]),
        "region_geometry": torch.zeros(1, 5, 4),
        "type_region_compatibility": torch.zeros(1, 2, 2, 5),
        "visibility_targets": torch.tensor([[1.0, 1.0]]),
        "gold_region_positive_mask": torch.tensor(
            [[[False, False, True, False, False], [False, True, False, False, False]]]
        ),
        "region_iou_targets": torch.tensor(
            [[[0.0, 0.1, 0.9, 0.0, 0.0], [0.0, 0.9, 0.1, 0.0, 0.0]]]
        ),
    }


def test_normalized_masked_rank_is_descending() -> None:
    scores = torch.tensor([[[4.0, 3.0, 1.0, 0.0, 100.0]]])
    valid = torch.tensor([[[True, True, True, True, False]]])
    rank = normalized_masked_rank(scores, valid)
    assert torch.allclose(
        rank,
        torch.tensor([[[0.0, 1 / 3, 2 / 3, 1.0, 1.0]]]),
    )


def test_fine_adapter_builds_sources_and_balanced_supervision() -> None:
    batch = _batch()
    model = CorrectionPreservationGroundingAdapter(
        FineGroundingAdapterConfig(
            input_size=4,
            hidden_size=8,
            final_budget=3,
            base_keep=2,
            detector_reference_budget=2,
        ),
        _FixedCoarse(),
    )
    outputs = model(batch)
    assert outputs["candidate_mask"].shape == (1, 2, 5)
    assert outputs["candidate_mask"][0, 0].tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert outputs["candidate_source_ids"][0, 0, 0].item() == SOURCE_BASE_ONLY
    assert (
        outputs["candidate_source_ids"][0, 0, 2].item()
        == SOURCE_LEARNED_ONLY
    )
    assert outputs["final_region_logits"][0, 0, -1].item() < -1000

    baseline_indices = torch.tensor([[0, 1]])
    baseline_visible = torch.tensor([[True, True]])
    supervision = fine_grounding_supervision(
        outputs,
        batch,
        baseline_region_indices=baseline_indices,
        baseline_visible_mask=baseline_visible,
        detector_reference_budget=2,
    )
    assert supervision["correction_mask"].tolist() == [[True, False]]
    assert supervision["preservation_mask"].tolist() == [[False, True]]
    assert supervision["promoted_gold_mask"].tolist() == [[True, False]]

    losses = fine_grounding_adapter_loss(
        outputs,
        batch,
        baseline_region_indices=baseline_indices,
        baseline_visible_mask=baseline_visible,
        detector_reference_budget=2,
    )
    assert torch.isfinite(losses["loss"])
    assert int(losses["correction_count"].item()) == 1
    assert int(losses["preservation_count"].item()) == 1


def test_formal_null_maps_to_expanded_null_without_changing_real_indices() -> None:
    mapped = map_formal_regions_to_expanded(
        torch.tensor([[3, 16]]),
        [{"null_region_index": 16}],
        [{"null_region_index": 36}],
    )
    assert mapped.tolist() == [[3, 36]]


def _memory_dataset(
    records: list[dict], *, max_regions: int
) -> RecordCandidateDataset:
    dataset = object.__new__(RecordCandidateDataset)
    dataset.records = records
    dataset.metadata = {
        "stage1_checkpoint_sha256": "same",
        "candidate_config": {"max_regions": max_regions, "top_m_types": 2},
    }
    return dataset


def _paired_record(spans: list[list[int]], sources: list[int], budget: int) -> dict:
    span_count = len(spans)
    region_count = budget + 1
    return {
        "span_candidates": torch.tensor(spans),
        "span_mask": torch.ones(span_count, dtype=torch.bool),
        "span_features": torch.arange(span_count * 4).reshape(span_count, 4),
        "span_source_ids": torch.tensor(sources),
        "fixed_type_ids": torch.tensor([0] * span_count),
        "region_boxes": torch.zeros(region_count, 4),
        "region_features": torch.zeros(region_count, 4),
        "metadata": {"record_id": "r0", "null_region_index": budget},
    }


def test_paired_dataset_aligns_r36_rows_by_span_and_masks_non_stage1_misses() -> None:
    formal_record = _paired_record([[0, 1], [2, 3], [4, 5]], [0, 1, 2], 2)
    expanded_record = _paired_record([[2, 3], [0, 1], [8, 9]], [1, 0, 2], 4)
    formal = _memory_dataset([formal_record], max_regions=2)
    expanded = _memory_dataset([expanded_record], max_regions=4)

    paired = PairedRecordCandidateDataset(formal, expanded)
    item = paired[0]["expanded"]

    assert item["span_candidates"].tolist() == [[0, 1], [2, 3], [4, 5]]
    assert item["span_mask"].tolist() == [True, True, False]
    assert item["span_features"][0].tolist() == expanded_record[
        "span_features"
    ][1].tolist()
    assert paired.alignment_summary["matched_stage1_spans"] == 1


def test_paired_dataset_rejects_a_missing_formal_stage1_span() -> None:
    formal_record = _paired_record([[0, 1]], [0], 2)
    expanded_record = _paired_record([[2, 3]], [0], 4)
    formal = _memory_dataset([formal_record], max_regions=2)
    expanded = _memory_dataset([expanded_record], max_regions=4)

    try:
        PairedRecordCandidateDataset(formal, expanded)
    except ValueError as error:
        assert "missing formal Stage1 span" in str(error)
    else:
        raise AssertionError("Missing Stage1 spans must fail cache pairing.")


def test_paired_dataset_keeps_formal_type_when_region_budget_changes_it() -> None:
    formal_record = _paired_record([[0, 1]], [0], 2)
    expanded_record = _paired_record([[0, 1]], [0], 4)
    formal_record["fixed_type_ids"][0] = 2
    expanded_record["fixed_type_ids"][0] = 1
    formal = _memory_dataset([formal_record], max_regions=2)
    expanded = _memory_dataset([expanded_record], max_regions=4)

    paired = PairedRecordCandidateDataset(formal, expanded)
    item = paired[0]["expanded"]

    assert item["fixed_type_ids"].tolist() == [2]
    assert paired.alignment_summary["stage1_type_mismatches"] == 1
