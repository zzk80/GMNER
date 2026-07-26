from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gmner.constants import IGNORE_INDEX
from gmner.fmnerg.baseline import build_matched_b0_payload
from gmner.fmnerg.candidate_contract import (
    FINE_CANDIDATE_SCHEMA,
    FINE_CANDIDATE_SCHEMA_VERSION,
)
from gmner.fmnerg.data_utils import first_record_indices
from gmner.fmnerg.taxonomy import SubtypeTaxonomy
from scripts.analyze_fmnerg_r16_oracle import analyze_payload
from scripts.summarize_fmnerg_stage1_f1 import summarize
from sidecars.fmnerg_subtype.metrics import (
    coarse_end_to_end_metrics as sidecar_coarse_metrics,
)


TAXONOMY_PATH = (
    Path(__file__).resolve().parents[2]
    / "sidecars"
    / "fmnerg_subtype"
    / "taxonomy_twitter10000.json"
)


def _taxonomy() -> SubtypeTaxonomy:
    return SubtypeTaxonomy.from_file(TAXONOMY_PATH)


def _stage1_bypass(
    *,
    span_f1: float,
    fine_mner_f1: float | None = None,
    fmnerg_f1: float | None = None,
) -> dict:
    result = {"span": {"precision": span_f1, "recall": span_f1, "f1": span_f1}}
    if fine_mner_f1 is not None:
        result["fine_mner"] = {
            "precision": fine_mner_f1,
            "recall": fine_mner_f1,
            "f1": fine_mner_f1,
        }
    if fmnerg_f1 is not None:
        result["fmnerg"] = {
            "precision": fmnerg_f1,
            "recall": fmnerg_f1,
            "f1": fmnerg_f1,
        }
    return result


def _coarse_oracle_payload() -> dict:
    return {
        "metadata": {
            "split": "dev",
            "summary": {
                "stage1_bypass": _stage1_bypass(span_f1=0.8),
            },
        },
        "records": [
            {
                "span_candidates": torch.tensor([[0, 1]]),
                "region_mask": torch.tensor([True, True, True]),
                "metadata": {
                    "record_id": "0",
                    "null_region_index": 2,
                    "gold_entities": [
                        {
                            "span": [0, 1],
                            "visible": True,
                            "region_positive_indices": [1],
                        },
                        {
                            "span": [2, 3],
                            "visible": True,
                            "region_positive_indices": [],
                        },
                        {
                            "span": [4, 5],
                            "visible": False,
                            "region_positive_indices": [2],
                        },
                    ],
                },
            }
        ],
    }


def test_r16_oracle_is_region_proposal_recall_not_span_coverage() -> None:
    metrics = analyze_payload(_coarse_oracle_payload())

    assert metrics["visible_region_oracle_recall"] == 0.5
    assert metrics["visible_joint_span_region_coverage"] == 0.5
    assert metrics["span_candidate_coverage"] == pytest.approx(1 / 3)


def test_r16_oracle_rejects_test_cache() -> None:
    payload = _coarse_oracle_payload()
    payload["metadata"]["split"] = "test"

    with pytest.raises(ValueError, match="Dev cache"):
        analyze_payload(payload)


def test_fine_r16_oracle_validates_taxonomy_and_all_candidate_logits() -> None:
    taxonomy = _taxonomy()
    actor_id = taxonomy.subtype_id("actor")
    raw_logits = torch.zeros(1, taxonomy.num_subtypes)
    raw_logits[0, actor_id] = 4.0
    parent_ids = torch.tensor([1])
    probabilities = torch.softmax(
        taxonomy.mask_logits(raw_logits, parent_ids),
        dim=-1,
    )
    top_values = probabilities.topk(2, dim=-1).values
    payload = {
        "metadata": {
            "split": "dev",
            "label_schema": FINE_CANDIDATE_SCHEMA,
            "fine_schema_version": FINE_CANDIDATE_SCHEMA_VERSION,
            **taxonomy.fingerprint_metadata(),
            "summary": {
                "visible_region_oracle_recall": 1.0,
                "stage1_bypass": _stage1_bypass(
                    span_f1=0.8,
                    fine_mner_f1=0.7,
                    fmnerg_f1=0.6,
                ),
            },
        },
        "records": [
            {
                "span_candidates": torch.tensor([[0, 1]]),
                "fixed_type_ids": parent_ids,
                "fixed_parent_ids": parent_ids,
                "subtype_raw_logits": raw_logits,
                "fixed_subtype_ids": torch.tensor([actor_id]),
                "subtype_confidence": top_values[:, 0],
                "subtype_margin": top_values[:, 0] - top_values[:, 1],
                "subtype_entropy": -(
                    probabilities
                    * probabilities.clamp_min(1e-8).log()
                ).sum(dim=-1),
                "gold_subtype_ids": torch.tensor([actor_id]),
                "region_mask": torch.tensor([True, True]),
                "metadata": {
                    "record_id": "0",
                    "null_region_index": 1,
                    "gold_entities": [
                        {
                            "span": [0, 1],
                            "visible": True,
                            "region_positive_indices": [0],
                        }
                    ],
                },
            }
        ],
    }

    metrics = analyze_payload(payload, taxonomy=taxonomy)

    assert metrics["visible_region_oracle_recall"] == 1.0


def test_b0_payload_preserves_coarse_prediction_and_adds_fine_gold() -> None:
    cache = {
        "metadata": {
            "split": "dev",
            "stage1_checkpoint_sha256": "stage1",
        },
        "records": [
            {
                "metadata": {
                    "record_id": "0",
                    "stage1_predictions": [
                        {
                            "span": [0, 1],
                            "type_id": 1,
                            "region_index": 2,
                            "subtype_id": 9,
                        }
                    ],
                    "gold_entities": [
                        {
                            "span": [0, 1],
                            "type_id": 1,
                            "region_positive_indices": [2],
                        }
                    ],
                }
            }
        ],
    }
    payload = build_matched_b0_payload(
        cache,
        fine_gold={
            "0": {
                (0, 1): {
                    "subtype": "actor",
                    "subtype_id": 3,
                }
            }
        },
    )

    prediction = payload["records"][0]["predictions"][0]
    target = payload["records"][0]["gold_entities"][0]
    assert prediction == {
        "span": [0, 1],
        "type_id": 1,
        "region_index": 2,
    }
    assert target["subtype"] == "actor"
    assert target["subtype_id"] == 3
    assert payload["metadata"]["coarse_metrics"]["gmner_f1"] == 1.0
    assert payload["metadata"]["coarse_metrics"] == sidecar_coarse_metrics(
        payload["records"]
    )


def test_first_record_indices_keep_all_entity_expansions() -> None:
    class Dataset:
        samples = [
            {"record_id": "0"},
            {"record_id": "0"},
            {"record_id": "1"},
            {"record_id": "2"},
        ]

    assert first_record_indices(Dataset(), 2) == [0, 1, 2]


def _artifact(kind: str, metrics: dict) -> dict:
    return {
        "metadata": {
            "kind": kind,
            "split": "dev",
            "test_accessed": False,
        },
        "metrics": metrics,
    }


def test_f1_summary_applies_preregistered_single_seed_checks() -> None:
    baseline_b0 = _artifact(
        "fmnerg_stage1_matched_b0",
        {"fine_mner_f1": 0.70, "fmnerg_f1": 0.50},
    )
    baseline_oracle = _artifact(
        "fmnerg_r16_visible_region_oracle",
        {
            "visible_region_oracle_recall": 0.84,
            "stage1_bypass": _stage1_bypass(span_f1=0.85),
        },
    )
    stage1_dev = _artifact(
        "fmnerg_stage1_f_dev_evaluation",
        {
            "fine_mner_f1": 0.704,
            "fmnerg_f1": 0.506,
            "hierarchy_consistency": 1.0,
        },
    )
    stage1_oracle = _artifact(
        "fmnerg_r16_visible_region_oracle",
        {
            "visible_region_oracle_recall": 0.839,
            "stage1_bypass": _stage1_bypass(
                span_f1=0.848,
                fine_mner_f1=0.704,
                fmnerg_f1=0.506,
            ),
        },
    )

    result = summarize(
        baseline_b0=baseline_b0,
        baseline_oracle=baseline_oracle,
        stage1_dev=stage1_dev,
        stage1_oracle=stage1_oracle,
    )

    assert result["single_seed_signal_passed"] is True
    assert result["deltas"]["fmnerg_f1"] == pytest.approx(0.006)


def test_fine_gold_ignore_index_constant_is_stable() -> None:
    assert IGNORE_INDEX == -100
