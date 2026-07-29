from __future__ import annotations

import torch

from gmner.data.formal_candidate_anchor import (
    load_formal_anchor_cache,
    stage1_entities_from_anchor,
)


def _anchor_record() -> dict:
    return {
        "metadata": {
            "record_id": "r0",
            "stage1_predictions": [
                {"span": [1, 3], "type_id": 0, "region_index": 4},
            ],
        }
    }


def test_stage1_entities_are_restored_from_formal_anchor() -> None:
    entities = stage1_entities_from_anchor(
        _anchor_record(),
        ["before", "New", "York", "after"],
    )

    assert entities == [
        {
            "start": 1,
            "end": 3,
            "type": "LOC",
            "text": "New York",
        }
    ]


def test_stage1_entities_preserve_formal_subtype() -> None:
    record = _anchor_record()
    record["metadata"]["stage1_predictions"][0]["subtype_id"] = 17

    entities = stage1_entities_from_anchor(
        record,
        ["before", "New", "York", "after"],
    )

    assert entities[0]["subtype_id"] == 17


def test_formal_anchor_requires_same_stage1_and_candidate_contract(
    tmp_path,
) -> None:
    path = tmp_path / "formal.pt"
    formal_spec = {
        "max_regions": 16,
        "max_span_candidates": 12,
        "top_m_types": 3,
    }
    torch.save(
        {
            "metadata": {
                "stage1_checkpoint_sha256": "stage1",
                "data_source_sha256": "source",
                "candidate_config": formal_spec,
                "candidate_config_sha256": "formal-spec",
            },
            "records": [_anchor_record()],
        },
        path,
    )

    records, provenance = load_formal_anchor_cache(
        path,
        stage1_checkpoint_sha256="stage1",
        data_source_sha256="source",
        expanded_candidate_spec={**formal_spec, "max_regions": 36},
    )

    assert records[0]["metadata"]["record_id"] == "r0"
    assert provenance["max_regions"] == 16
    assert provenance["sha256"]


def test_formal_anchor_rejects_non_region_candidate_differences(
    tmp_path,
) -> None:
    path = tmp_path / "formal.pt"
    torch.save(
        {
            "metadata": {
                "stage1_checkpoint_sha256": "stage1",
                "data_source_sha256": "source",
                "candidate_config": {
                    "max_regions": 16,
                    "max_span_candidates": 12,
                },
            },
            "records": [_anchor_record()],
        },
        path,
    )

    try:
        load_formal_anchor_cache(
            path,
            stage1_checkpoint_sha256="stage1",
            data_source_sha256="source",
            expanded_candidate_spec={
                "max_regions": 36,
                "max_span_candidates": 10,
            },
        )
    except ValueError as error:
        assert "differ only in max_regions" in str(error)
    else:
        raise AssertionError("Candidate-contract drift must reject the anchor.")


def test_formal_anchor_rejects_taxonomy_drift(tmp_path) -> None:
    path = tmp_path / "formal.pt"
    formal_spec = {
        "max_regions": 16,
        "label_schema": "fine_hierarchical",
        "taxonomy_sha256": "taxonomy-a",
    }
    torch.save(
        {
            "metadata": {
                "stage1_checkpoint_sha256": "stage1",
                "data_source_sha256": "source",
                "label_schema": "fine_hierarchical",
                "taxonomy_sha256": "taxonomy-a",
                "candidate_config": formal_spec,
            },
            "records": [_anchor_record()],
        },
        path,
    )

    try:
        load_formal_anchor_cache(
            path,
            stage1_checkpoint_sha256="stage1",
            data_source_sha256="source",
            expanded_candidate_spec={
                **formal_spec,
                "max_regions": 36,
            },
            expected_label_schema="fine_hierarchical",
            expected_taxonomy_sha256="taxonomy-b",
        )
    except ValueError as error:
        assert "another subtype taxonomy" in str(error)
    else:
        raise AssertionError("Taxonomy drift must reject the formal anchor.")
