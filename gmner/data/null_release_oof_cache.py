"""Materialized full-chain OOF features for the NULL Release verifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


NULL_RELEASE_OOF_FORMAT_VERSION = 1
NULL_RELEASE_OOF_KIND = "null_release_full_chain_oof"

FINE_OUTPUT_KEYS = (
    "candidate_mask",
    "final_region_logits",
    "fine_top4_indices",
    "fine_top4_valid_mask",
    "span_grounding_state",
    "region_grounding_state",
    "type_grounding_state",
    "candidate_source_ids",
    "base_log_prior",
    "coarse_log_prior",
    "base_rank",
    "coarse_rank",
    "detector_rank",
    "fixed_type_region_compatibility",
    "promoted_candidate_mask",
    "fixed_type_ids",
)
EXPANDED_BATCH_KEYS = (
    "span_mask",
    "span_source_ids",
    "gold_span_mask",
    "visibility_targets",
    "type_candidates",
    "gold_type_mask",
    "gold_region_positive_mask",
    "region_mask",
    "region_is_null",
    "region_detector_scores",
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_digest(record_ids: list[str]) -> str:
    value = json.dumps(sorted(record_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _cpu_feature(value: torch.Tensor, *, preserve_float32: bool = False) -> torch.Tensor:
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point() and not preserve_float32:
        tensor = tensor.to(torch.float16)
    return tensor


def _required(mapping: dict[str, Any], keys: tuple[str, ...], name: str) -> dict:
    missing = [key for key in keys if key not in mapping]
    if missing:
        raise ValueError(f"Cannot cache {name}; missing fields: {missing}.")
    return {key: mapping[key] for key in keys}


def pack_null_release_context_batch(context: dict[str, object], fold_id: int) -> dict:
    """Keep only tensors consumed by the release model and its loss."""

    fine = _required(dict(context["fine_outputs"]), FINE_OUTPUT_KEYS, "Fine output")
    expanded = _required(
        dict(context["expanded"]), EXPANDED_BATCH_KEYS, "expanded batch"
    )
    hierarchy = dict(context["hierarchy_outputs"])
    evidence = dict(context["evidence_outputs"])
    reliability = dict(context["reliability_outputs"])
    if "fixed_type_ids" not in hierarchy:
        raise ValueError("Hierarchy output is missing fixed_type_ids.")
    if "evidence_scalar_features" not in evidence:
        raise ValueError("Evidence output is missing evidence_scalar_features.")
    if "reliability_probability" not in reliability:
        raise ValueError("Reliability output is missing reliability_probability.")

    metadata = list(dict(context["expanded"])["metadata"])
    record_ids = [str(item.get("record_id", "")) for item in metadata]
    if any(not record_id for record_id in record_ids):
        raise ValueError("Every cached OOF record must have a record_id.")

    preserve = {
        "final_region_logits",
        "base_log_prior",
        "coarse_log_prior",
    }
    return {
        "fold_id": int(fold_id),
        "record_ids": record_ids,
        "fine_outputs": {
            key: _cpu_feature(value, preserve_float32=key in preserve)
            for key, value in fine.items()
        },
        "hierarchy_outputs": {
            "fixed_type_ids": _cpu_feature(hierarchy["fixed_type_ids"])
        },
        "evidence_outputs": {
            "evidence_scalar_features": _cpu_feature(
                evidence["evidence_scalar_features"]
            )
        },
        "expanded": {
            key: _cpu_feature(value)
            for key, value in expanded.items()
        },
        "reliability_outputs": {
            "reliability_probability": _cpu_feature(
                reliability["reliability_probability"]
            )
        },
        "current_visible": _cpu_feature(context["current_visible"]),
        "base_is_null": _cpu_feature(context["base_is_null"]),
        "deployment_span_mask": _cpu_feature(context["deployment_span_mask"]),
    }


def move_null_release_context_batch(batch: dict, device: torch.device) -> dict:
    def move(value):
        if isinstance(value, torch.Tensor):
            return value.to(device, non_blocking=True)
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value

    return move(batch)


def _validate_batches(payload: dict) -> tuple[list[dict], list[str], set[int]]:
    if not isinstance(payload, dict) or "batches" not in payload:
        raise ValueError("Invalid NULL Release OOF feature cache.")
    batches = list(payload["batches"])
    record_ids: list[str] = []
    observed_folds: set[int] = set()
    for index, batch in enumerate(batches):
        for key in (
            "fold_id",
            "record_ids",
            "fine_outputs",
            "hierarchy_outputs",
            "evidence_outputs",
            "expanded",
            "reliability_outputs",
            "current_visible",
            "base_is_null",
            "deployment_span_mask",
        ):
            if key not in batch:
                raise ValueError(f"OOF batch {index} is missing {key!r}.")
        batch_ids = [str(value) for value in batch["record_ids"]]
        batch_size = int(batch["expanded"]["span_mask"].size(0))
        if len(batch_ids) != batch_size:
            raise ValueError(f"OOF batch {index} record-id count is inconsistent.")
        record_ids.extend(batch_ids)
        observed_folds.add(int(batch["fold_id"]))
        fine = dict(batch["fine_outputs"])
        missing_fine = [key for key in FINE_OUTPUT_KEYS if key not in fine]
        if missing_fine:
            raise ValueError(
                f"OOF batch {index} is missing Fine fields: {missing_fine}."
            )
        top4_indices = fine["fine_top4_indices"].long()
        top4_valid = fine["fine_top4_valid_mask"].bool()
        candidate_mask = fine["candidate_mask"].bool()
        if top4_indices.shape != top4_valid.shape or top4_indices.size(-1) != 4:
            raise ValueError(f"OOF batch {index} has an invalid fixed Top-4 shape.")
        if top4_indices.shape[:-1] != candidate_mask.shape[:-1]:
            raise ValueError(
                f"OOF batch {index} fixed Top-4 does not align with spans."
            )
        if top4_valid.any():
            values = top4_indices[top4_valid]
            if int(values.min()) < 0 or int(values.max()) >= candidate_mask.size(-1):
                raise ValueError(
                    f"OOF batch {index} fixed Top-4 contains an invalid index."
                )
        safe = top4_indices.clamp(0, max(candidate_mask.size(-1) - 1, 0))
        if not torch.all(candidate_mask.gather(-1, safe) | ~top4_valid):
            raise ValueError(
                f"OOF batch {index} fixed Top-4 is outside the Fine candidates."
            )
        one_hot = torch.nn.functional.one_hot(
            safe, num_classes=candidate_mask.size(-1)
        ).bool()
        if (one_hot & top4_valid.unsqueeze(-1)).sum(dim=-2).gt(1).any():
            raise ValueError(
                f"OOF batch {index} fixed Top-4 contains duplicate actions."
            )
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("Full-chain OOF feature cache contains duplicate record ids.")
    return batches, record_ids, observed_folds


def _validate_common_metadata(metadata: dict, *, require_reliability: bool) -> None:
    if metadata.get("kind") != NULL_RELEASE_OOF_KIND:
        raise ValueError("Feature cache is not a full-chain NULL Release OOF cache.")
    if int(metadata.get("format_version", -1)) != NULL_RELEASE_OOF_FORMAT_VERSION:
        raise ValueError("Unsupported NULL Release OOF feature-cache version.")
    if not bool(metadata.get("full_chain_oof")):
        raise ValueError("NULL Release cache is not marked full_chain_oof=true.")
    if require_reliability and not bool(metadata.get("includes_reliability")):
        raise ValueError("Model requires OOF Reliability outputs, but cache omits them.")


def validate_fold_oof_payload(
    payload: dict,
    *,
    expected_fold_id: int,
    expected_record_ids: list[str],
    require_reliability: bool = True,
) -> dict:
    metadata = dict(payload.get("metadata") or {})
    _validate_common_metadata(metadata, require_reliability=require_reliability)
    if int(metadata.get("num_folds", -1)) != 10:
        raise ValueError("Formal NULL Release fold cache must declare 10 folds.")
    if int(metadata.get("fold_id", -1)) != int(expected_fold_id):
        raise ValueError("NULL Release fold cache has the wrong fold id.")
    batches, record_ids, observed_folds = _validate_batches(payload)
    if observed_folds != {int(expected_fold_id)}:
        raise ValueError("NULL Release fold batches contain another fold id.")
    if record_ids != [str(value) for value in expected_record_ids]:
        raise ValueError("NULL Release fold record order differs from the manifest.")
    if int(metadata.get("records", -1)) != len(record_ids):
        raise ValueError("OOF fold metadata record count is inconsistent.")
    if stable_id_digest(record_ids) != str(metadata.get("record_ids_sha256", "")):
        raise ValueError("OOF fold record-id digest mismatch.")
    return {"metadata": metadata, "batches": batches, "records": len(record_ids)}


def validate_full_chain_oof_payload(
    payload: dict,
    *,
    expected_num_folds: int,
    expected_records: int | None,
    require_reliability: bool,
) -> dict:
    metadata = dict(payload.get("metadata") or {})
    _validate_common_metadata(metadata, require_reliability=require_reliability)
    if int(metadata.get("num_folds", -1)) != int(expected_num_folds):
        raise ValueError(
            "NULL Release OOF fold-count mismatch: "
            f"expected {expected_num_folds}, found {metadata.get('num_folds')}."
        )
    fold_ids = sorted(int(value) for value in metadata.get("fold_ids") or [])
    if fold_ids != list(range(int(expected_num_folds))):
        raise ValueError(f"Expected contiguous OOF fold ids, found {fold_ids}.")
    batches, record_ids, observed_folds = _validate_batches(payload)
    if expected_records is not None and len(record_ids) != int(expected_records):
        raise ValueError(
            f"Expected {expected_records} OOF records, found {len(record_ids)}."
        )
    if int(metadata.get("records", -1)) != len(record_ids):
        raise ValueError("OOF metadata record count is inconsistent.")
    if observed_folds != set(fold_ids):
        raise ValueError(
            f"OOF batch folds {sorted(observed_folds)} disagree with metadata {fold_ids}."
        )
    actual_digest = stable_id_digest(record_ids)
    if actual_digest != str(metadata.get("record_ids_sha256", "")):
        raise ValueError("OOF record-id digest mismatch.")
    return {
        "metadata": metadata,
        "batches": batches,
        "records": len(record_ids),
    }


def load_full_chain_oof_cache(
    path: str | Path,
    *,
    expected_num_folds: int = 10,
    expected_records: int | None = None,
    require_reliability: bool = True,
) -> dict:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(
            "Full-chain OOF feature cache is required before NULL Release training: "
            f"{cache_path}"
        )
    payload = torch.load(cache_path, map_location="cpu")
    result = validate_full_chain_oof_payload(
        payload,
        expected_num_folds=expected_num_folds,
        expected_records=expected_records,
        require_reliability=require_reliability,
    )
    result["path"] = cache_path
    result["sha256"] = sha256_file(cache_path)
    return result
