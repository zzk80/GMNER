"""Evaluation contract for the independent hierarchical subtype sidecar."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from .data import SubtypeFeatureDataset
from .io import sha256_file
from .metrics import (
    canonical_coarse_prediction_sha256,
    coarse_end_to_end_metrics,
    end_to_end_metrics,
    subtype_classification_metrics,
    subtype_classification_report,
)
from .model import HierarchicalSubtypeSidecar
from .taxonomy import SubtypeTaxonomy


def validate_feature_contract(
    dataset: SubtypeFeatureDataset,
    *,
    taxonomy: SubtypeTaxonomy,
    split: str,
    mode: str,
    input_size: int,
    expected_stage1_sha256: str | None = None,
) -> str:
    metadata = dataset.metadata
    if metadata.get("split") != split or metadata.get("mode") != mode:
        raise ValueError(
            f"Expected {split}/{mode} subtype features, found "
            f"{metadata.get('split')}/{metadata.get('mode')}."
        )
    if metadata.get("test_accessed") is not False:
        raise ValueError("Subtype feature cache accessed test data.")
    if metadata.get("taxonomy_sha256") != taxonomy.source_sha256:
        raise ValueError("Subtype feature cache taxonomy fingerprint changed.")
    if dataset.features.ndim != 2 or dataset.features.size(1) != input_size:
        raise ValueError(
            f"Expected subtype features [N, {input_size}], found "
            f"{tuple(dataset.features.shape)}."
        )
    stage1_sha256 = str(metadata.get("stage1_checkpoint_sha256", ""))
    if not stage1_sha256:
        raise ValueError("Subtype feature cache has no Stage1 fingerprint.")
    if (
        expected_stage1_sha256 is not None
        and stage1_sha256 != expected_stage1_sha256
    ):
        raise ValueError("Subtype feature caches use different Stage1 checkpoints.")
    if mode == "gold" and torch.any(dataset.subtype_ids < 0):
        raise ValueError("Gold-span subtype cache contains unlabeled examples.")
    return stage1_sha256


def load_formal_predictions(
    path: str | Path,
    *,
    taxonomy: SubtypeTaxonomy,
) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("kind") != "fmnerg_frozen_formal_predictions":
        raise ValueError("Not a frozen FMNERG formal-prediction artifact.")
    if int(metadata.get("format_version", -1)) != 1:
        raise ValueError("Unsupported frozen formal-prediction format.")
    if metadata.get("split") != "dev":
        raise ValueError("Only dev formal predictions are allowed before test release.")
    if metadata.get("test_accessed") is not False:
        raise ValueError("Frozen formal predictions accessed test data.")
    if metadata.get("taxonomy_sha256") != taxonomy.source_sha256:
        raise ValueError("Frozen formal-prediction taxonomy fingerprint changed.")
    records = list(payload.get("records") or [])
    digest = canonical_coarse_prediction_sha256(records)
    if digest != metadata.get("coarse_prediction_sha256"):
        raise ValueError("Frozen coarse predictions fail their SHA-256 audit.")
    recomputed = coarse_end_to_end_metrics(records)
    if recomputed != metadata.get("coarse_metrics"):
        raise ValueError(
            "Frozen formal coarse metrics do not exactly match their records."
        )
    return payload


def validate_expected_frozen_gmner(
    formal_payload: dict[str, Any],
    *,
    expected: float | None,
    tolerance: float,
) -> None:
    if expected is None:
        return
    actual = float(formal_payload["metadata"]["coarse_metrics"]["gmner_f1"])
    if abs(actual - float(expected)) > float(tolerance):
        raise ValueError(
            "Frozen formal chain does not reproduce the preregistered dev GMNER: "
            f"actual={actual:.12f}, expected={float(expected):.12f}, "
            f"tolerance={float(tolerance):.2e}."
        )


@torch.inference_mode()
def predict_subtypes(
    model: HierarchicalSubtypeSidecar,
    dataset: SubtypeFeatureDataset,
    *,
    batch_size: int,
    device: torch.device,
    coarse_type_ids: torch.Tensor | None = None,
) -> list[int]:
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
    )
    predictions: list[int] = []
    override = None
    if coarse_type_ids is not None:
        override = torch.as_tensor(coarse_type_ids, dtype=torch.long)
        if override.ndim != 1 or override.numel() != len(dataset):
            raise ValueError(
                "Subtype coarse override must contain one parent id per example."
            )
    for batch in loader:
        parent_ids = batch["coarse_type_ids"]
        if override is not None:
            parent_ids = override[batch["example_index"].long()]
        outputs = model(
            batch["features"].to(device),
            parent_ids.to(device),
        )
        predictions.extend(
            outputs["predicted_subtype_ids"].detach().cpu().tolist()
        )
    if len(predictions) != len(dataset):
        raise RuntimeError("Subtype prediction count differs from feature count.")
    return [int(value) for value in predictions]


@torch.inference_mode()
def evaluate_gold_spans(
    model: HierarchicalSubtypeSidecar,
    dataset: SubtypeFeatureDataset,
    *,
    taxonomy: SubtypeTaxonomy,
    batch_size: int,
    device: torch.device,
    include_detailed: bool = False,
) -> dict[str, Any]:
    predicted = predict_subtypes(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    gold = [int(value) for value in dataset.subtype_ids.tolist()]
    raw = subtype_classification_metrics(
        predicted,
        gold,
        num_classes=taxonomy.num_subtypes,
    )
    report = subtype_classification_report(
        predicted,
        gold,
        taxonomy=taxonomy,
    )
    metrics: dict[str, Any] = {
        "subtype_accuracy_on_gold_spans": raw["subtype_accuracy"],
        "subtype_micro_f1_on_gold_spans": raw["subtype_micro_f1"],
        "subtype_macro_f1_on_gold_spans": raw["subtype_macro_f1"],
        "gold_span_examples": float(len(gold)),
    }
    for parent, score in report["parent_macro_f1"].items():
        metrics[f"subtype_{parent.lower()}_macro_f1_on_gold_spans"] = score
    if include_detailed:
        metrics["subtype_classification_report_on_gold_spans"] = report
    return metrics


def _assert_exact_coarse_identity(
    *,
    original_records: list[dict[str, Any]],
    augmented_records: list[dict[str, Any]],
    expected_digest: str,
    expected_metrics: dict[str, float],
) -> dict[str, Any]:
    before_digest = canonical_coarse_prediction_sha256(original_records)
    after_digest = canonical_coarse_prediction_sha256(augmented_records)
    before_metrics = coarse_end_to_end_metrics(original_records)
    after_metrics = coarse_end_to_end_metrics(augmented_records)
    if before_digest != expected_digest:
        raise ValueError("Original formal prediction digest changed before inference.")
    if after_digest != before_digest:
        raise AssertionError(
            "Subtype sidecar changed span/type/region/order in formal predictions."
        )
    if before_metrics != expected_metrics:
        raise ValueError("Original formal coarse metrics changed before inference.")
    if after_metrics != before_metrics:
        raise AssertionError(
            "Subtype sidecar changed exact coarse GMNER metric values."
        )
    return {
        "gmner_identity_exact": True,
        "coarse_prediction_sha256_before": before_digest,
        "coarse_prediction_sha256_after": after_digest,
        "coarse_metrics_before": before_metrics,
        "coarse_metrics_after": after_metrics,
    }


def evaluate_formal_subtype_ids(
    *,
    subtype_predictions: list[int],
    gold_parent_predictions: list[int],
    target_subtype_ids: torch.Tensor,
    predicted_parent_ids: torch.Tensor,
    examples: list[dict[str, Any]],
    formal_payload: dict[str, Any],
    taxonomy: SubtypeTaxonomy,
    include_records: bool = False,
) -> dict[str, Any]:
    if not (
        len(subtype_predictions)
        == len(gold_parent_predictions)
        == target_subtype_ids.numel()
        == predicted_parent_ids.numel()
        == len(examples)
    ):
        raise ValueError("Formal subtype predictions and examples are misaligned.")
    metadata = dict(formal_payload["metadata"])
    target_subtype_ids = target_subtype_ids.long()
    predicted_parent_ids = predicted_parent_ids.long()
    target_available = target_subtype_ids.ge(0)
    oracle_parent_ids = predicted_parent_ids.clone()
    if target_available.any():
        oracle_parent_ids[target_available] = torch.tensor(
            [
                taxonomy.parent_id(int(subtype_id))
                for subtype_id in target_subtype_ids[target_available].tolist()
            ],
            dtype=torch.long,
        )

    original_records = list(formal_payload["records"])
    augmented_records = copy.deepcopy(original_records)
    assigned: set[tuple[int, int]] = set()
    hierarchy_consistent = 0

    for subtype_id, example in zip(subtype_predictions, examples):
        record_index = int(example["record_index"])
        prediction_index = int(example["prediction_index"])
        key = (record_index, prediction_index)
        if key in assigned:
            raise ValueError(f"Duplicate formal subtype assignment: {key}.")
        assigned.add(key)
        try:
            original = original_records[record_index]["predictions"][
                prediction_index
            ]
            prediction = augmented_records[record_index]["predictions"][
                prediction_index
            ]
        except IndexError as exc:
            raise ValueError(
                f"Formal subtype feature points outside prediction records: {key}."
            ) from exc
        if str(original_records[record_index]["record_id"]) != str(
            example["record_id"]
        ):
            raise ValueError(f"Formal record id mismatch at {key}.")
        example_span = example.get("span")
        if example_span is None:
            example_span = [example["start"], example["end"]]
        if list(map(int, original["span"])) != list(map(int, example_span)):
            raise ValueError(f"Formal span mismatch at {key}.")
        if int(original["type_id"]) != int(example["coarse_type_id"]):
            raise ValueError(f"Formal coarse type mismatch at {key}.")
        parent_id = taxonomy.parent_id(subtype_id)
        if parent_id != int(original["type_id"]):
            raise AssertionError(
                f"Hierarchy mask failed for formal prediction {key}."
            )
        prediction["subtype_id"] = subtype_id
        prediction["subtype"] = taxonomy.labels[subtype_id]
        hierarchy_consistent += 1

    expected_assignments = sum(
        len(record.get("predictions") or []) for record in original_records
    )
    if len(assigned) != expected_assignments:
        raise ValueError(
            f"Assigned {len(assigned)} subtype predictions for "
            f"{expected_assignments} formal entities."
        )

    identity = _assert_exact_coarse_identity(
        original_records=original_records,
        augmented_records=augmented_records,
        expected_digest=str(metadata["coarse_prediction_sha256"]),
        expected_metrics=dict(metadata["coarse_metrics"]),
    )
    metrics = end_to_end_metrics(augmented_records)
    exact_span_total = 0
    exact_span_subtype_correct = 0
    parent_conditioned_total = 0
    parent_conditioned_subtype_correct = 0
    for record in augmented_records:
        gold_by_span = {
            tuple(map(int, target["span"])): target
            for target in record.get("gold_entities") or []
        }
        for prediction in record.get("predictions") or []:
            target = gold_by_span.get(tuple(map(int, prediction["span"])))
            if target is None:
                continue
            exact_span_total += 1
            subtype_correct = int(prediction["subtype_id"]) == int(
                target["subtype_id"]
            )
            exact_span_subtype_correct += int(subtype_correct)
            if int(prediction["type_id"]) == int(target["type_id"]):
                parent_conditioned_total += 1
                parent_conditioned_subtype_correct += int(subtype_correct)

    metrics.update(
        {
            "subtype_accuracy_on_correct_predicted_spans": (
                exact_span_subtype_correct / max(exact_span_total, 1)
            ),
            "parent_conditioned_subtype_accuracy": (
                parent_conditioned_subtype_correct
                / max(parent_conditioned_total, 1)
            ),
            "exact_predicted_span_count": float(exact_span_total),
            "parent_conditioned_span_count": float(parent_conditioned_total),
            "hierarchy_consistency_rate": (
                hierarchy_consistent / max(expected_assignments, 1)
            ),
        }
    )
    predicted_subtypes_tensor = torch.tensor(
        subtype_predictions,
        dtype=torch.long,
    )
    gold_parent_predictions_tensor = torch.tensor(
        gold_parent_predictions,
        dtype=torch.long,
    )
    coarse_wrong = target_available & predicted_parent_ids.ne(oracle_parent_ids)
    gold_parent_correct = (
        target_available
        & gold_parent_predictions_tensor.eq(target_subtype_ids)
    )
    predicted_parent_correct = (
        target_available
        & predicted_subtypes_tensor.eq(target_subtype_ids)
    )
    metrics.update(
        {
            "predicted_parent_subtype_accuracy_on_exact_predicted_spans": (
                int(predicted_parent_correct.sum().item())
                / max(int(target_available.sum().item()), 1)
            ),
            "gold_parent_subtype_accuracy_on_exact_predicted_spans": (
                int(gold_parent_correct.sum().item())
                / max(int(target_available.sum().item()), 1)
            ),
            "coarse_wrong_exact_predicted_span_count": float(
                coarse_wrong.sum().item()
            ),
            "coarse_wrong_gold_parent_subtype_correct": float(
                (coarse_wrong & gold_parent_correct).sum().item()
            ),
            "coarse_wrong_gold_parent_subtype_recovery_rate": (
                int((coarse_wrong & gold_parent_correct).sum().item())
                / max(int(coarse_wrong.sum().item()), 1)
            ),
        }
    )
    if metrics["fine_mner_f1"] > metrics["coarse_mner_f1"] + 1e-12:
        raise AssertionError("Fine MNER F1 exceeds coarse MNER F1.")
    if metrics["fmnerg_f1"] > metrics["gmner_f1"] + 1e-12:
        raise AssertionError("FMNERG F1 exceeds GMNER F1.")

    result: dict[str, Any] = {
        "metadata": {
            "kind": "fmnerg_subtype_sidecar_evaluation",
            "format_version": 1,
            "split": "dev",
            "test_accessed": False,
            **identity,
        },
        "metrics": metrics,
    }
    if include_records:
        result["records"] = augmented_records
    return result


@torch.inference_mode()
def evaluate_formal_predictions(
    model: HierarchicalSubtypeSidecar,
    dataset: SubtypeFeatureDataset,
    formal_payload: dict[str, Any],
    *,
    taxonomy: SubtypeTaxonomy,
    batch_size: int,
    device: torch.device,
    include_records: bool = False,
) -> dict[str, Any]:
    metadata = dict(formal_payload["metadata"])
    if dataset.metadata.get("coarse_prediction_sha256") != metadata.get(
        "coarse_prediction_sha256"
    ):
        raise ValueError(
            "Formal feature cache and formal prediction digest are inconsistent."
        )
    formal_path = dataset.metadata.get("formal_predictions")
    if formal_path and Path(formal_path).exists():
        if sha256_file(formal_path) != dataset.metadata.get(
            "formal_predictions_sha256"
        ):
            raise ValueError("Formal prediction file changed after feature extraction.")

    subtype_predictions = predict_subtypes(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
    )
    target_subtype_ids = dataset.subtype_ids.long()
    target_available = target_subtype_ids.ge(0)
    oracle_parent_ids = dataset.coarse_type_ids.clone().long()
    if target_available.any():
        oracle_parent_ids[target_available] = torch.tensor(
            [
                taxonomy.parent_id(int(subtype_id))
                for subtype_id in target_subtype_ids[target_available].tolist()
            ],
            dtype=torch.long,
        )
    gold_parent_predictions = predict_subtypes(
        model,
        dataset,
        batch_size=batch_size,
        device=device,
        coarse_type_ids=oracle_parent_ids,
    )
    return evaluate_formal_subtype_ids(
        subtype_predictions=subtype_predictions,
        gold_parent_predictions=gold_parent_predictions,
        target_subtype_ids=target_subtype_ids,
        predicted_parent_ids=dataset.coarse_type_ids,
        examples=dataset.examples,
        formal_payload=formal_payload,
        taxonomy=taxonomy,
        include_records=include_records,
    )


def save_json_atomic(payload: dict[str, Any], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
