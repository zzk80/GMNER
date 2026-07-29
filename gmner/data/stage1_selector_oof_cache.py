"""Compact, auditable Stage1 candidate caches for the D1 selector."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from gmner.data.null_release_oof_cache import stable_id_digest


STAGE1_SELECTOR_CACHE_KIND = "stage1_candidate_selector_oof"
STAGE1_SELECTOR_CACHE_VERSION = 1
STAGE1_SELECTOR_SCOPE_FOLD = "oof_fold"
STAGE1_SELECTOR_SCOPE_TRAIN = "oof_train"
STAGE1_SELECTOR_SCOPE_DEV = "dev"

SELECTOR_TENSOR_FIELDS = (
    "span_candidates",
    "span_mask",
    "span_features",
    "span_base_scores",
    "span_source_ids",
    "span_lengths",
    "type_candidates",
    "type_base_scores",
    "fixed_type_ids",
    "base_region_indices",
    "gold_span_mask",
    "gold_type_mask",
    "formal_candidate_mask",
)

_EXPECTED_DTYPES = {
    "span_candidates": torch.int64,
    "span_mask": torch.bool,
    "span_features": torch.float16,
    "span_base_scores": torch.float32,
    "span_source_ids": torch.int64,
    "span_lengths": torch.int64,
    "type_candidates": torch.int64,
    "type_base_scores": torch.float32,
    "fixed_type_ids": torch.int64,
    "base_region_indices": torch.int64,
    "gold_span_mask": torch.bool,
    "gold_type_mask": torch.bool,
    "formal_candidate_mask": torch.bool,
}


def _as_cpu_tensor(
    value: Any,
    *,
    dtype: torch.dtype,
    field: str,
) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Candidate field {field!r} must be a tensor.")
    return value.detach().to(device="cpu", dtype=dtype).contiguous()


def selector_record_id(record: dict) -> str:
    metadata = dict(record.get("metadata") or {})
    return str(metadata.get("record_id", ""))


def compact_candidate_record(record: dict, *, formal_source_id: int) -> dict:
    """Drop region tensors while preserving fields needed by D1 and its gates."""

    missing = [
        field
        for field in SELECTOR_TENSOR_FIELDS
        if field != "formal_candidate_mask" and field not in record
    ]
    if missing:
        raise ValueError(f"Candidate record is missing selector fields: {missing}.")

    source_ids = _as_cpu_tensor(
        record["span_source_ids"],
        dtype=torch.int64,
        field="span_source_ids",
    )
    metadata = dict(record.get("metadata") or {})
    record_id = str(metadata.get("record_id", ""))
    if not record_id:
        raise ValueError("Candidate record has no metadata.record_id.")

    compact = {
        "span_candidates": _as_cpu_tensor(
            record["span_candidates"],
            dtype=torch.int64,
            field="span_candidates",
        ),
        "span_mask": _as_cpu_tensor(
            record["span_mask"],
            dtype=torch.bool,
            field="span_mask",
        ),
        "span_features": _as_cpu_tensor(
            record["span_features"],
            dtype=torch.float16,
            field="span_features",
        ),
        "span_base_scores": _as_cpu_tensor(
            record["span_base_scores"],
            dtype=torch.float32,
            field="span_base_scores",
        ),
        "span_source_ids": source_ids,
        "span_lengths": _as_cpu_tensor(
            record["span_lengths"],
            dtype=torch.int64,
            field="span_lengths",
        ),
        "type_candidates": _as_cpu_tensor(
            record["type_candidates"],
            dtype=torch.int64,
            field="type_candidates",
        ),
        "type_base_scores": _as_cpu_tensor(
            record["type_base_scores"],
            dtype=torch.float32,
            field="type_base_scores",
        ),
        "fixed_type_ids": _as_cpu_tensor(
            record["fixed_type_ids"],
            dtype=torch.int64,
            field="fixed_type_ids",
        ),
        "base_region_indices": _as_cpu_tensor(
            record["base_region_indices"],
            dtype=torch.int64,
            field="base_region_indices",
        ),
        "gold_span_mask": _as_cpu_tensor(
            record["gold_span_mask"],
            dtype=torch.bool,
            field="gold_span_mask",
        ),
        "gold_type_mask": _as_cpu_tensor(
            record["gold_type_mask"],
            dtype=torch.bool,
            field="gold_type_mask",
        ),
        "formal_candidate_mask": source_ids.eq(int(formal_source_id)),
        "metadata": {
            "record_id": record_id,
            "text": str(metadata.get("text", "")),
            "tokens": list(metadata.get("tokens") or []),
            "candidate_sources": list(metadata.get("candidate_sources") or []),
            "stage1_predictions": list(metadata.get("stage1_predictions") or []),
            "gold_entities": list(metadata.get("gold_entities") or []),
            "null_region_index": int(metadata.get("null_region_index", -1)),
        },
    }
    validate_selector_record(compact, formal_source_id=formal_source_id)
    return compact


def validate_selector_record(record: dict, *, formal_source_id: int) -> None:
    missing = [field for field in SELECTOR_TENSOR_FIELDS if field not in record]
    if missing:
        raise ValueError(f"Selector record is missing fields: {missing}.")
    record_id = selector_record_id(record)
    if not record_id:
        raise ValueError("Selector record is missing metadata.record_id.")

    for field, dtype in _EXPECTED_DTYPES.items():
        value = record[field]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Selector field {field!r} must be a tensor.")
        if value.device.type != "cpu":
            raise ValueError(f"Selector field {field!r} must be stored on CPU.")
        if value.dtype != dtype:
            raise ValueError(
                f"Selector field {field!r} has dtype {value.dtype}; expected {dtype}."
            )

    spans = record["span_candidates"]
    span_mask = record["span_mask"]
    span_count = int(spans.size(0))
    if spans.ndim != 2 or spans.size(1) != 2:
        raise ValueError(f"Record {record_id} has invalid span-candidate shape.")
    one_dimensional = (
        "span_mask",
        "span_base_scores",
        "span_source_ids",
        "span_lengths",
        "fixed_type_ids",
        "base_region_indices",
        "gold_span_mask",
        "formal_candidate_mask",
    )
    for field in one_dimensional:
        value = record[field]
        if value.ndim != 1 or value.size(0) != span_count:
            raise ValueError(f"Record {record_id} has invalid {field} shape.")
    if record["span_features"].ndim != 2 or record["span_features"].size(0) != span_count:
        raise ValueError(f"Record {record_id} has invalid span_features shape.")
    type_candidates = record["type_candidates"]
    type_scores = record["type_base_scores"]
    type_gold = record["gold_type_mask"]
    if (
        type_candidates.ndim != 2
        or type_candidates.size(0) != span_count
        or type_scores.shape != type_candidates.shape
        or type_gold.shape != type_candidates.shape
    ):
        raise ValueError(f"Record {record_id} has inconsistent type-candidate fields.")
    if not torch.equal(
        record["formal_candidate_mask"],
        record["span_source_ids"].eq(int(formal_source_id)),
    ):
        raise ValueError(f"Record {record_id} has an invalid formal-candidate mask.")
    valid_spans = spans[span_mask]
    if valid_spans.numel() and (
        valid_spans[:, 0].lt(0).any()
        or valid_spans[:, 1].le(valid_spans[:, 0]).any()
    ):
        raise ValueError(f"Record {record_id} contains an invalid half-open span.")
    span_tuples = [tuple(value) for value in valid_spans.tolist()]
    if len(span_tuples) != len(set(span_tuples)):
        raise ValueError(f"Record {record_id} contains duplicate span candidates.")
    expected_lengths = spans[:, 1] - spans[:, 0]
    if not torch.equal(record["span_lengths"], expected_lengths):
        raise ValueError(f"Record {record_id} has inconsistent span lengths.")

    metadata = dict(record.get("metadata") or {})
    candidate_sources = list(metadata.get("candidate_sources") or [])
    if candidate_sources and len(candidate_sources) != span_count:
        raise ValueError(f"Record {record_id} has inconsistent candidate sources.")
    formal_rows = torch.nonzero(
        record["formal_candidate_mask"] & span_mask,
        as_tuple=False,
    ).squeeze(-1)
    formal_predictions = list(metadata.get("stage1_predictions") or [])
    if len(formal_predictions) != int(formal_rows.numel()):
        raise ValueError(
            f"Record {record_id} formal candidates do not reproduce Stage1 count."
        )
    expected_predictions = {
        (
            tuple(record["span_candidates"][row].tolist()),
            int(record["fixed_type_ids"][row].item()),
            int(record["base_region_indices"][row].item()),
        )
        for row in formal_rows.tolist()
    }
    observed_predictions = {
        (
            tuple(int(value) for value in prediction["span"]),
            int(prediction["type_id"]),
            int(prediction["region_index"]),
        )
        for prediction in formal_predictions
    }
    if expected_predictions != observed_predictions:
        raise ValueError(
            f"Record {record_id} compact cache does not reproduce Stage1 predictions."
        )


def build_fold_selector_payload(
    candidate_payload: dict,
    *,
    fold_id: int,
    num_folds: int,
    source_candidate_cache: str,
    source_candidate_cache_sha256: str,
    stage1_config: str,
    stage1_config_sha256: str,
    fold_manifest: str,
    fold_manifest_sha256: str,
    reference_fold_proof: str,
    reference_fold_proof_sha256: str,
    git_commit: str | None,
    source_tree_sha256: str,
) -> dict:
    candidate_metadata = dict(candidate_payload.get("metadata") or {})
    records = list(candidate_payload.get("records") or [])
    if not bool(candidate_metadata.get("oof_heldout")):
        raise ValueError("Source candidate cache is not marked oof_heldout=true.")
    if int(candidate_metadata.get("oof_fold_id", -1)) != int(fold_id):
        raise ValueError("Source candidate cache has the wrong OOF fold id.")
    source2id = dict(candidate_metadata.get("source2id") or {})
    if "stage1" not in source2id:
        raise ValueError("Source candidate cache does not declare the Stage1 source id.")
    formal_source_id = int(source2id["stage1"])
    compact_records = [
        compact_candidate_record(record, formal_source_id=formal_source_id)
        for record in records
    ]
    record_ids = [selector_record_id(record) for record in compact_records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Source candidate cache contains duplicate record ids.")

    payload = {
        "metadata": {
            "format_version": STAGE1_SELECTOR_CACHE_VERSION,
            "kind": STAGE1_SELECTOR_CACHE_KIND,
            "scope": STAGE1_SELECTOR_SCOPE_FOLD,
            "oof": True,
            "test_accessed": False,
            "fold_id": int(fold_id),
            "num_folds": int(num_folds),
            "records": len(compact_records),
            "record_ids": record_ids,
            "record_ids_sha256": stable_id_digest(record_ids),
            "hidden_size": int(candidate_metadata.get("hidden_size", -1)),
            "formal_source_id": formal_source_id,
            "source2id": source2id,
            "candidate_config": dict(candidate_metadata.get("candidate_config") or {}),
            "candidate_config_sha256": str(
                candidate_metadata.get("candidate_config_sha256", "")
            ),
            "stage1_checkpoint_sha256": str(
                candidate_metadata.get("stage1_checkpoint_sha256", "")
            ),
            "data_source": str(candidate_metadata.get("data_source", "")),
            "data_source_sha256": str(
                candidate_metadata.get("data_source_sha256", "")
            ),
            "source_candidate_cache": str(source_candidate_cache),
            "source_candidate_cache_sha256": str(source_candidate_cache_sha256),
            "stage1_config": str(stage1_config),
            "stage1_config_sha256": str(stage1_config_sha256),
            "fold_manifest": str(fold_manifest),
            "fold_manifest_sha256": str(fold_manifest_sha256),
            "reference_fold_proof": str(reference_fold_proof),
            "reference_fold_proof_sha256": str(reference_fold_proof_sha256),
            "git_commit": git_commit,
            "source_tree_sha256": str(source_tree_sha256),
        },
        "records": compact_records,
    }
    validate_selector_oof_payload(
        payload,
        expected_fold_id=fold_id,
        expected_num_folds=num_folds,
        expected_record_ids=record_ids,
    )
    return payload


def build_dev_selector_payload(
    candidate_payload: dict,
    *,
    source_candidate_cache: str,
    source_candidate_cache_sha256: str,
    stage1_config: str,
    stage1_config_sha256: str,
    git_commit: str | None,
    source_tree_sha256: str,
) -> dict:
    """Compact a full-fit, label-free Dev candidate cache."""

    candidate_metadata = dict(candidate_payload.get("metadata") or {})
    records = list(candidate_payload.get("records") or [])
    if str(candidate_metadata.get("split", "")) != "dev":
        raise ValueError("Dev selector source cache must declare split=dev.")
    if candidate_metadata.get("oof_heldout"):
        raise ValueError("Dev selector cache cannot be marked as an OOF fold.")
    source2id = dict(candidate_metadata.get("source2id") or {})
    if "stage1" not in source2id:
        raise ValueError("Source candidate cache does not declare the Stage1 source id.")
    formal_source_id = int(source2id["stage1"])
    compact_records = [
        compact_candidate_record(record, formal_source_id=formal_source_id)
        for record in records
    ]
    record_ids = [selector_record_id(record) for record in compact_records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Dev candidate cache contains duplicate record ids.")
    payload = {
        "metadata": {
            "format_version": STAGE1_SELECTOR_CACHE_VERSION,
            "kind": STAGE1_SELECTOR_CACHE_KIND,
            "scope": STAGE1_SELECTOR_SCOPE_DEV,
            "split": "dev",
            "oof": False,
            "test_accessed": False,
            "records": len(compact_records),
            "record_ids": record_ids,
            "record_ids_sha256": stable_id_digest(record_ids),
            "hidden_size": int(candidate_metadata.get("hidden_size", -1)),
            "formal_source_id": formal_source_id,
            "source2id": source2id,
            "candidate_config": dict(candidate_metadata.get("candidate_config") or {}),
            "candidate_config_sha256": str(
                candidate_metadata.get("candidate_config_sha256", "")
            ),
            "stage1_checkpoint_sha256": str(
                candidate_metadata.get("stage1_checkpoint_sha256", "")
            ),
            "data_source": str(candidate_metadata.get("data_source", "")),
            "data_source_sha256": str(
                candidate_metadata.get("data_source_sha256", "")
            ),
            "source_candidate_cache": str(source_candidate_cache),
            "source_candidate_cache_sha256": str(source_candidate_cache_sha256),
            "stage1_config": str(stage1_config),
            "stage1_config_sha256": str(stage1_config_sha256),
            "git_commit": git_commit,
            "source_tree_sha256": str(source_tree_sha256),
        },
        "records": compact_records,
    }
    validate_selector_dev_payload(payload, expected_record_ids=record_ids)
    return payload


def validate_selector_dev_payload(
    payload: dict,
    *,
    expected_record_ids: list[str] | None = None,
) -> dict:
    if not isinstance(payload, dict) or "metadata" not in payload or "records" not in payload:
        raise ValueError("Invalid Stage1 selector Dev cache payload.")
    metadata = dict(payload["metadata"])
    if metadata.get("kind") != STAGE1_SELECTOR_CACHE_KIND:
        raise ValueError("Not a Stage1 selector candidate cache.")
    if int(metadata.get("format_version", -1)) != STAGE1_SELECTOR_CACHE_VERSION:
        raise ValueError("Unsupported Stage1 selector cache version.")
    if metadata.get("scope") != STAGE1_SELECTOR_SCOPE_DEV:
        raise ValueError("Stage1 selector cache is not a Dev cache.")
    if metadata.get("split") != "dev" or metadata.get("oof") is not False:
        raise ValueError("Stage1 selector Dev cache has an invalid split contract.")
    if metadata.get("test_accessed") is not False:
        raise ValueError("Stage1 selector Dev cache must assert test_accessed=false.")
    required_metadata = (
        "candidate_config_sha256",
        "stage1_checkpoint_sha256",
        "data_source_sha256",
        "source_tree_sha256",
    )
    missing_metadata = [
        key for key in required_metadata if not str(metadata.get(key, ""))
    ]
    if missing_metadata:
        raise ValueError(
            f"Stage1 selector Dev cache lacks provenance: {missing_metadata}."
        )
    records = list(payload["records"])
    formal_source_id = int(metadata.get("formal_source_id", -1))
    for record in records:
        validate_selector_record(record, formal_source_id=formal_source_id)
    record_ids = [selector_record_id(record) for record in records]
    if any(not record_id for record_id in record_ids):
        raise ValueError("Stage1 selector Dev cache contains an empty record id.")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Stage1 selector Dev cache contains duplicate record ids.")
    if int(metadata.get("records", -1)) != len(records):
        raise ValueError("Stage1 selector Dev cache record count is inconsistent.")
    if metadata.get("record_ids") != record_ids:
        raise ValueError("Stage1 selector Dev cache record order metadata changed.")
    if stable_id_digest(record_ids) != metadata.get("record_ids_sha256"):
        raise ValueError("Stage1 selector Dev cache record-id digest is inconsistent.")
    if expected_record_ids is not None and record_ids != [
        str(value) for value in expected_record_ids
    ]:
        raise ValueError("Stage1 selector Dev cache record order differs from source.")
    return {"metadata": metadata, "records": records, "record_ids": record_ids}


def validate_selector_oof_payload(
    payload: dict,
    *,
    expected_fold_id: int | None = None,
    expected_num_folds: int = 10,
    expected_record_ids: list[str] | None = None,
) -> dict:
    if not isinstance(payload, dict) or "metadata" not in payload or "records" not in payload:
        raise ValueError("Invalid Stage1 selector OOF cache payload.")
    metadata = dict(payload["metadata"])
    if metadata.get("kind") != STAGE1_SELECTOR_CACHE_KIND:
        raise ValueError("Not a Stage1 selector OOF cache.")
    if int(metadata.get("format_version", -1)) != STAGE1_SELECTOR_CACHE_VERSION:
        raise ValueError("Unsupported Stage1 selector OOF cache version.")
    if metadata.get("test_accessed") is not False:
        raise ValueError("Stage1 selector OOF cache must assert test_accessed=false.")
    if not bool(metadata.get("oof")):
        raise ValueError("Stage1 selector cache is not marked as OOF.")
    if int(metadata.get("num_folds", -1)) != int(expected_num_folds):
        raise ValueError("Stage1 selector OOF cache has the wrong fold count.")
    scope = str(metadata.get("scope", ""))
    if scope not in {STAGE1_SELECTOR_SCOPE_FOLD, STAGE1_SELECTOR_SCOPE_TRAIN}:
        raise ValueError(f"Unsupported Stage1 selector cache scope: {scope!r}.")
    if expected_fold_id is not None:
        if scope != STAGE1_SELECTOR_SCOPE_FOLD:
            raise ValueError("Expected a fold cache, found a merged Train cache.")
        if int(metadata.get("fold_id", -1)) != int(expected_fold_id):
            raise ValueError("Stage1 selector cache has the wrong fold id.")
    required_metadata = (
        "candidate_config_sha256",
        "stage1_checkpoint_sha256",
        "data_source_sha256",
        "fold_manifest_sha256",
        "source_tree_sha256",
    )
    missing_metadata = [
        key for key in required_metadata if not str(metadata.get(key, ""))
    ]
    if missing_metadata:
        raise ValueError(
            f"Stage1 selector cache lacks provenance: {missing_metadata}."
        )

    records = list(payload["records"])
    formal_source_id = int(metadata.get("formal_source_id", -1))
    for record in records:
        validate_selector_record(record, formal_source_id=formal_source_id)
    record_ids = [selector_record_id(record) for record in records]
    if any(not record_id for record_id in record_ids):
        raise ValueError("Stage1 selector cache contains an empty record id.")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Stage1 selector cache contains duplicate record ids.")
    if int(metadata.get("records", -1)) != len(records):
        raise ValueError("Stage1 selector cache record count is inconsistent.")
    if metadata.get("record_ids") != record_ids:
        raise ValueError("Stage1 selector cache record order metadata changed.")
    if stable_id_digest(record_ids) != metadata.get("record_ids_sha256"):
        raise ValueError("Stage1 selector cache record-id digest is inconsistent.")
    if expected_record_ids is not None and record_ids != [
        str(value) for value in expected_record_ids
    ]:
        raise ValueError("Stage1 selector cache record order differs from the fold.")
    return {"metadata": metadata, "records": records, "record_ids": record_ids}


def atomic_save_selector_payload(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def load_selector_oof_payload(path: str | Path) -> dict:
    payload = torch.load(Path(path), map_location="cpu")
    validate_selector_oof_payload(payload)
    return payload


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output)
