"""Span-aligned text and frozen-region data for J0 visual subtype fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from sidecars.fmnerg_subtype.data import FineRecord, read_fine_conll
from sidecars.fmnerg_subtype.encoder_runtime import (
    validate_online_gold_hierarchy,
)
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.online_data import (
    OnlineSubtypeCollator,
    OnlineSubtypeRecordDataset,
    formal_online_records,
    gold_online_records,
)
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy

from .config import JointSubtypeConfig
from .formal_chain import (
    FrozenM33AFeatureProvider,
    load_frozen_dev_contract,
)


def _augment_gold_records(
    records: list[FineRecord],
    taxonomy: SubtypeTaxonomy,
    provider: FrozenM33AFeatureProvider,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = gold_online_records(records, taxonomy)
    report = {
        "examples": 0,
        "visual_available": 0,
        "selected_real": 0,
        "selected_null": 0,
        "span_missing": 0,
        "region_missing": 0,
    }
    for record in output:
        record_id = str(record["record_id"])
        for span in record["spans"]:
            evidence = provider.gold_evidence(
                record_id,
                (int(span["start"]), int(span["end"])),
            )
            span.update(evidence.as_span_fields())
            report["examples"] += 1
            report["visual_available"] += int(evidence.available)
            report["selected_real"] += int(
                evidence.available and not evidence.is_null
            )
            report["selected_null"] += int(
                evidence.available and evidence.is_null
            )
            report["span_missing"] += int(
                evidence.selection_source == "gold_span_missing"
            )
            report["region_missing"] += int(
                evidence.selection_source == "gold_region_missing"
            )
    return output, report


def _augment_formal_records(
    formal_payload: dict[str, Any],
    records: list[FineRecord],
    taxonomy: SubtypeTaxonomy,
    provider: FrozenM33AFeatureProvider,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = formal_online_records(formal_payload, records, taxonomy)
    report = {
        "examples": 0,
        "visual_available": 0,
        "selected_real": 0,
        "selected_null": 0,
    }
    for record in output:
        record_index = None
        for span in record["spans"]:
            current_record_index = int(span["record_index"])
            if record_index is None:
                record_index = current_record_index
            elif record_index != current_record_index:
                raise ValueError(
                    "Formal span record indices changed within one record."
                )
            prediction = formal_payload["records"][current_record_index][
                "predictions"
            ][int(span["prediction_index"])]
            evidence = provider.formal_evidence(
                str(record["record_id"]),
                (int(span["start"]), int(span["end"])),
                int(prediction["region_index"]),
            )
            span.update(evidence.as_span_fields())
            report["examples"] += 1
            report["visual_available"] += int(evidence.available)
            report["selected_real"] += int(not evidence.is_null)
            report["selected_null"] += int(evidence.is_null)
    return output, report


class JointOnlineSubtypeCollator:
    """Add frozen visual evidence to the existing online text collator."""

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int,
        region_feature_size: int,
        geometry_size: int,
    ) -> None:
        self.text_collator = OnlineSubtypeCollator(
            tokenizer,
            max_length=max_length,
        )
        self.region_feature_size = int(region_feature_size)
        self.geometry_size = int(geometry_size)

    def __call__(
        self,
        records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        batch = self.text_collator(records)
        examples = list(batch["examples"])
        region_features = torch.stack(
            [
                torch.as_tensor(
                    example["joint_region_feature"],
                    dtype=torch.float32,
                )
                for example in examples
            ]
        )
        geometry = torch.stack(
            [
                torch.as_tensor(
                    example["joint_region_geometry"],
                    dtype=torch.float32,
                )
                for example in examples
            ]
        )
        if region_features.shape != (
            len(examples),
            self.region_feature_size,
        ):
            raise ValueError(
                "Joint region features do not match the configured dimension."
            )
        if geometry.shape != (len(examples), self.geometry_size):
            raise ValueError(
                "Joint region geometry does not match the configured dimension."
            )
        batch.update(
            {
                "joint_region_features": region_features,
                "joint_region_geometry": geometry,
                "joint_detector_scores": torch.tensor(
                    [
                        float(example["joint_detector_score"])
                        for example in examples
                    ],
                    dtype=torch.float32,
                ),
                "joint_region_is_null": torch.tensor(
                    [
                        bool(example["joint_region_is_null"])
                        for example in examples
                    ],
                    dtype=torch.bool,
                ),
                "joint_visual_available": torch.tensor(
                    [
                        bool(example["joint_visual_available"])
                        for example in examples
                    ],
                    dtype=torch.bool,
                ),
                "joint_region_indices": torch.tensor(
                    [
                        int(example["joint_region_index"])
                        for example in examples
                    ],
                    dtype=torch.long,
                ),
            }
        )
        return batch


def load_joint_subtype_data(
    *,
    config: JointSubtypeConfig,
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
    train_cache = resolve_path(config.data.train_expanded_cache, root)
    dev_cache = resolve_path(config.data.dev_expanded_cache, root)
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
    train_provider = FrozenM33AFeatureProvider.from_path(train_cache)
    formal_payload, dev_provider = load_frozen_dev_contract(
        formal_predictions_path=formal_path,
        expanded_cache_path=dev_cache,
        taxonomy=taxonomy,
    )
    for name, provider in (
        ("Train", train_provider),
        ("Dev", dev_provider),
    ):
        if provider.region_feature_size != config.model.region_feature_size:
            raise ValueError(
                f"{name} R36 region feature size changed: "
                f"{provider.region_feature_size} != "
                f"{config.model.region_feature_size}."
            )
        if provider.geometry_size != config.model.geometry_size:
            raise ValueError(
                f"{name} R36 geometry size changed: "
                f"{provider.geometry_size} != {config.model.geometry_size}."
            )

    train_joint, train_report = _augment_gold_records(
        train_records,
        taxonomy,
        train_provider,
    )
    dev_gold_joint, dev_gold_report = _augment_gold_records(
        dev_records,
        taxonomy,
        dev_provider,
    )
    dev_formal_joint, dev_formal_report = _augment_formal_records(
        formal_payload,
        dev_records,
        taxonomy,
        dev_provider,
    )
    train_dataset = OnlineSubtypeRecordDataset(train_joint)
    dev_gold_dataset = OnlineSubtypeRecordDataset(dev_gold_joint)
    dev_formal_dataset = OnlineSubtypeRecordDataset(dev_formal_joint)
    validate_online_gold_hierarchy(train_dataset, taxonomy)
    validate_online_gold_hierarchy(dev_gold_dataset, taxonomy)
    expected_formal = sum(
        len(record.get("predictions") or [])
        for record in formal_payload["records"]
    )
    if len(dev_formal_dataset.examples) != expected_formal:
        raise ValueError(
            "Joint formal dataset does not cover every frozen prediction."
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
        "train_provider": train_provider.artifact_report(),
        "dev_provider": dev_provider.artifact_report(),
        "dev_formal_predictions": {
            "path": str(formal_path),
            "sha256": sha256_file(formal_path),
            "coarse_prediction_sha256": formal_payload["metadata"][
                "coarse_prediction_sha256"
            ],
        },
        "coverage": {
            "train_gold": train_report,
            "dev_gold": dev_gold_report,
            "dev_formal": dev_formal_report,
        },
        "formal_chain_mutated": False,
        "test_accessed": False,
    }
    return (
        train_dataset,
        dev_gold_dataset,
        dev_formal_dataset,
        formal_payload,
        artifacts,
    )
