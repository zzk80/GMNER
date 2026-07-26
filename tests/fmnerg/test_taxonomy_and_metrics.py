from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from gmner.constants import DEFAULT_LABEL2ID
from gmner.fmnerg.metrics import (
    end_to_end_fine_metrics,
    fine_entities_from_bio_tags,
    subtype_classification_metrics,
)
from gmner.fmnerg.taxonomy import (
    EXPECTED_SUBTYPE_COUNT,
    SubtypeTaxonomy,
    bind_config_taxonomy_fingerprint,
    validate_taxonomy_fingerprint,
)


TAXONOMY_PATH = (
    Path(__file__).resolve().parents[2]
    / "sidecars"
    / "fmnerg_subtype"
    / "taxonomy_twitter10000.json"
)


def test_twitter10000_taxonomy_has_fixed_hierarchy() -> None:
    taxonomy = SubtypeTaxonomy.from_file(TAXONOMY_PATH)

    assert taxonomy.num_subtypes == EXPECTED_SUBTYPE_COUNT
    assert taxonomy.num_parents == 4
    assert set(taxonomy.parent_ids) == {0, 1, 2, 3}
    assert taxonomy.parent_name(taxonomy.subtype_id("actor")) == "PER"
    assert taxonomy.parent_name(taxonomy.subtype_id("company")) == "ORG"


def test_taxonomy_masks_illegal_sibling_predictions() -> None:
    taxonomy = SubtypeTaxonomy.from_file(TAXONOMY_PATH)
    logits = torch.zeros(2, taxonomy.num_subtypes)
    logits[0, taxonomy.subtype_id("company")] = 10.0
    logits[0, taxonomy.subtype_id("actor")] = 1.0
    logits[1, taxonomy.subtype_id("actor")] = 10.0
    logits[1, taxonomy.subtype_id("company")] = 1.0

    masked = taxonomy.mask_logits(logits, torch.tensor([1, 2]))

    assert masked[0].argmax().item() == taxonomy.subtype_id("actor")
    assert masked[1].argmax().item() == taxonomy.subtype_id("company")


def test_taxonomy_fingerprint_is_a_hard_contract() -> None:
    taxonomy = SubtypeTaxonomy.from_file(TAXONOMY_PATH)
    validate_taxonomy_fingerprint(
        taxonomy.fingerprint_metadata(),
        taxonomy,
        artifact_name="synthetic cache",
    )
    with pytest.raises(ValueError, match="taxonomy SHA mismatch"):
        validate_taxonomy_fingerprint(
            {
                **taxonomy.fingerprint_metadata(),
                "taxonomy_sha256": "other",
            },
            taxonomy,
            artifact_name="synthetic cache",
        )


def test_resolved_config_binds_and_validates_taxonomy_sha() -> None:
    taxonomy = SubtypeTaxonomy.from_file(TAXONOMY_PATH)

    class DataConfig:
        subtype_taxonomy_sha256 = ""

    config = DataConfig()
    bind_config_taxonomy_fingerprint(config, taxonomy)
    assert config.subtype_taxonomy_sha256 == taxonomy.source_sha256

    config.subtype_taxonomy_sha256 = "other"
    with pytest.raises(ValueError, match="Configured subtype taxonomy SHA"):
        bind_config_taxonomy_fingerprint(config, taxonomy)


def test_taxonomy_rejects_wrong_class_count(tmp_path: Path) -> None:
    payload = json.loads(TAXONOMY_PATH.read_text(encoding="utf-8"))
    payload["subtype_parents"].pop("actor")
    path = tmp_path / "taxonomy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Expected 51"):
        SubtypeTaxonomy.from_file(path)


def test_fine_bio_extraction_validates_parent() -> None:
    taxonomy = SubtypeTaxonomy.from_file(TAXONOMY_PATH)
    entities = fine_entities_from_bio_tags(
        tokens=["Taylor", "Swift", "performed"],
        coarse_tags=[
            DEFAULT_LABEL2ID["B-PER"],
            DEFAULT_LABEL2ID["I-PER"],
            DEFAULT_LABEL2ID["O"],
        ],
        fine_tags=["B-musician", "I-musician", "O"],
        taxonomy=taxonomy,
        coarse_id2label={
            value: label for label, value in DEFAULT_LABEL2ID.items()
        },
    )

    assert entities == [
        {
            "span": [0, 2],
            "start": 0,
            "end": 2,
            "text": "Taylor Swift",
            "type": "PER",
            "type_id": 1,
            "subtype": "musician",
            "subtype_id": taxonomy.subtype_id("musician"),
        }
    ]


def test_fmnerg_metrics_require_subtype_and_region() -> None:
    records = [
        {
            "predictions": [
                {"span": [0, 1], "subtype_id": 3, "region_index": 2},
                {"span": [2, 3], "subtype_id": 4, "region_index": 0},
            ],
            "gold_entities": [
                {
                    "span": [0, 1],
                    "subtype_id": 3,
                    "region_positive_indices": [2],
                },
                {
                    "span": [2, 3],
                    "subtype_id": 4,
                    "region_positive_indices": [1],
                },
            ],
        }
    ]

    metrics = end_to_end_fine_metrics(records)

    assert metrics["fine_mner_f1"] == 1.0
    assert metrics["fmnerg_f1"] == 0.5
    assert metrics["fmnerg_f1"] <= metrics["fine_mner_f1"]


def test_subtype_macro_f1_counts_observed_classes() -> None:
    metrics = subtype_classification_metrics(
        [0, 1, 1],
        [0, 1, 2],
        num_classes=3,
    )

    assert metrics["subtype_accuracy"] == pytest.approx(2 / 3)
    assert 0.0 < metrics["subtype_macro_f1"] < 1.0
