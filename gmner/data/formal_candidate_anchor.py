"""Formal-cache anchoring for expanded record candidates."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from gmner.constants import ID2ENTITY_TYPE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_record_id(record: dict) -> str:
    return str((record.get("metadata") or {}).get("record_id", ""))


def candidate_config_mismatches(
    formal: dict,
    expanded: dict,
) -> dict[str, tuple[object, object]]:
    ignored = {"max_regions"}
    return {
        key: (formal.get(key), expanded.get(key))
        for key in sorted((set(formal) | set(expanded)) - ignored)
        if formal.get(key) != expanded.get(key)
    }


def load_formal_anchor_cache(
    path: Path,
    *,
    stage1_checkpoint_sha256: str,
    data_source_sha256: str,
    expanded_candidate_spec: dict,
) -> tuple[list[dict], dict]:
    """Load and validate the authoritative formal Stage1 decode."""

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
        raise ValueError(f"Invalid formal anchor cache: {path}")
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("stage1_checkpoint_sha256") != stage1_checkpoint_sha256:
        raise ValueError("Formal anchor cache uses another Stage1 checkpoint.")
    if metadata.get("data_source_sha256") != data_source_sha256:
        raise ValueError("Formal anchor cache uses another source file.")
    formal_spec = dict(metadata.get("candidate_config") or {})
    mismatches = candidate_config_mismatches(
        formal_spec,
        expanded_candidate_spec,
    )
    if mismatches:
        raise ValueError(
            "Formal anchor and expanded cache must differ only in max_regions; "
            f"mismatches={mismatches}."
        )
    formal_budget = int(formal_spec.get("max_regions", 0))
    expanded_budget = int(expanded_candidate_spec.get("max_regions", 0))
    if formal_budget <= 0 or formal_budget >= expanded_budget:
        raise ValueError(
            "Formal anchor max_regions must be smaller than the expanded "
            f"budget; formal={formal_budget}, expanded={expanded_budget}."
        )
    records = list(payload["records"])
    ids = [cache_record_id(record) for record in records]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("Formal anchor cache has missing or duplicate record ids.")
    provenance = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "max_regions": formal_budget,
        "candidate_config_sha256": str(
            metadata.get("candidate_config_sha256") or ""
        ),
    }
    return records, provenance


def stage1_entities_from_anchor(
    record: dict,
    tokens: list[str],
) -> list[dict]:
    """Restore the exact formal span/type decode stored in an R16 cache."""

    predictions = list((record.get("metadata") or {}).get("stage1_predictions") or [])
    entities: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for prediction in predictions:
        span = list(prediction.get("span") or [])
        if len(span) != 2:
            raise ValueError(
                f"Formal anchor record {cache_record_id(record)} has an invalid span."
            )
        start, end = map(int, span)
        type_id = int(prediction.get("type_id", -1))
        if (
            not 0 <= start < end <= len(tokens)
            or type_id not in ID2ENTITY_TYPE
            or (start, end) in seen
        ):
            raise ValueError(
                f"Formal anchor record {cache_record_id(record)} has an invalid "
                f"Stage1 prediction: span={(start, end)}, type_id={type_id}."
            )
        seen.add((start, end))
        entities.append(
            {
                "start": start,
                "end": end,
                "type": ID2ENTITY_TYPE[type_id],
                "text": " ".join(tokens[start:end]),
            }
        )
    return entities
