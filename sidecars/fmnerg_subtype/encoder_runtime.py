"""Shared loading contract for trainable subtype-encoder experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .data import read_fine_conll
from .encoder_config import SubtypeEncoderConfig
from .evaluator import (
    load_formal_predictions,
    validate_expected_frozen_gmner,
)
from .io import resolve_path, sha256_file
from .online_data import (
    OnlineSubtypeRecordDataset,
    formal_online_records,
    gold_online_records,
)
from .taxonomy import SubtypeTaxonomy


def validate_online_gold_hierarchy(
    dataset: OnlineSubtypeRecordDataset,
    taxonomy: SubtypeTaxonomy,
) -> None:
    expected = torch.tensor(
        [
            taxonomy.parent_id(int(subtype_id))
            for subtype_id in dataset.subtype_ids.tolist()
        ],
        dtype=torch.long,
    )
    if not torch.equal(expected, dataset.coarse_type_ids):
        mismatch = torch.nonzero(
            expected.ne(dataset.coarse_type_ids),
            as_tuple=False,
        ).reshape(-1)
        raise ValueError(
            "Online subtype gold hierarchy mismatch at examples "
            f"{mismatch[:20].tolist()}."
        )


def load_online_subtype_data(
    *,
    config: SubtypeEncoderConfig,
    taxonomy: SubtypeTaxonomy,
    root: Path,
) -> tuple[
    OnlineSubtypeRecordDataset,
    OnlineSubtypeRecordDataset,
    OnlineSubtypeRecordDataset,
    dict[str, Any],
    dict[str, Any],
]:
    train_source = resolve_path(config.data.train_source, root)
    dev_source = resolve_path(config.data.dev_source, root)
    formal_path = resolve_path(config.data.dev_formal_predictions, root)
    train_records = read_fine_conll(
        train_source,
        taxonomy,
        require_all_subtypes=True,
    )
    dev_records = read_fine_conll(
        dev_source,
        taxonomy,
        require_all_subtypes=True,
    )
    formal_payload = load_formal_predictions(
        formal_path,
        taxonomy=taxonomy,
    )
    validate_expected_frozen_gmner(
        formal_payload,
        expected=config.runtime.expected_dev_gmner_f1,
        tolerance=config.runtime.expected_dev_gmner_tolerance,
    )
    train_dataset = OnlineSubtypeRecordDataset(
        gold_online_records(train_records, taxonomy)
    )
    dev_gold_dataset = OnlineSubtypeRecordDataset(
        gold_online_records(dev_records, taxonomy)
    )
    dev_formal_dataset = OnlineSubtypeRecordDataset(
        formal_online_records(formal_payload, dev_records, taxonomy)
    )
    validate_online_gold_hierarchy(train_dataset, taxonomy)
    validate_online_gold_hierarchy(dev_gold_dataset, taxonomy)
    expected_formal = sum(
        len(record.get("predictions") or [])
        for record in formal_payload["records"]
    )
    if len(dev_formal_dataset.examples) != expected_formal:
        raise ValueError(
            "Online formal dataset does not cover every frozen prediction."
        )
    artifacts = {
        "train_source": {
            "path": str(train_source),
            "sha256": sha256_file(train_source),
        },
        "dev_source": {
            "path": str(dev_source),
            "sha256": sha256_file(dev_source),
        },
        "dev_formal_predictions": {
            "path": str(formal_path),
            "sha256": sha256_file(formal_path),
            "coarse_prediction_sha256": formal_payload["metadata"][
                "coarse_prediction_sha256"
            ],
        },
    }
    return (
        train_dataset,
        dev_gold_dataset,
        dev_formal_dataset,
        formal_payload,
        artifacts,
    )
