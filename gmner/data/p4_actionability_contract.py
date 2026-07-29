"""Contracts for the P4.0 protected-promotion source audit.

P4.0 is intentionally read-only with respect to Model-G.  The helpers in this
module validate archived full-chain OOF provenance, expose label-free candidate
views, and prevent candidate rows from being joined across incompatible caches.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch

from gmner.data.full_chain_oof_contract import (
    REQUIRED_PIPELINE_STAGES,
    SUPERVISED_PIPELINE_STAGES,
)
from gmner.data.null_release_oof_cache import (
    sha256_file,
    stable_id_digest,
    validate_fold_oof_payload,
)
from gmner.data.stage1_selector_oof_cache import (
    selector_record_id,
)


P4_DEVELOPMENT_FOLDS = tuple(range(8))
P4_CALIBRATION_FOLDS = (8, 9)

P4_PROVENANCE_REPORT_KIND = "p4_full_chain_oof_provenance_report"
P4_SOURCE_MANIFEST_KIND = "p4_protected_promotion_source_manifest"
P4_GOLD_FREE_CACHE_KIND = "p4_gold_free_candidate_source"
P4_FORMAT_VERSION = 1

FULL_CHAIN_ARCHIVE_KIND = "null_release_oof_fold_archive"
FULL_CHAIN_PROOF_KIND = "null_release_full_chain_fold_proof"
FULL_CHAIN_PIPELINE_KIND = "null_release_full_chain_fold_pipeline"

P4_SELECTOR_RECORD_FIELDS = (
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
)
P4_SELECTOR_METADATA_FIELDS = (
    "record_id",
    "text",
    "tokens",
    "candidate_sources",
    "null_region_index",
)
P4_SELECTOR_METADATA_PROVENANCE_FIELDS = (
    "candidate_config",
    "candidate_config_sha256",
    "stage1_checkpoint_sha256",
    "data_source_sha256",
    "source_candidate_cache_sha256",
    "stage1_config_sha256",
    "fold_manifest_sha256",
    "reference_fold_proof_sha256",
    "git_commit",
    "source_tree_sha256",
    "source2id",
    "formal_source_id",
)

_FORBIDDEN_SCOPE_TOKENS = {"dev", "test"}
_GOLD_DERIVED_FIELD_NAMES = {
    "gold_entities",
    "gold_region_positive_mask",
    "gold_span_mask",
    "gold_type_mask",
    "visibility_targets",
}


def canonical_json_sha256(payload: dict) -> str:
    """Hash a JSON payload independently of indentation and key order."""

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def attach_manifest_sha256(payload: dict) -> dict:
    """Return a copy with a digest over every field except the digest itself."""

    output = copy.deepcopy(payload)
    output.pop("manifest_sha256", None)
    output["manifest_sha256"] = canonical_json_sha256(output)
    return output


def validate_manifest_sha256(payload: dict) -> None:
    expected = str(payload.get("manifest_sha256", ""))
    if not expected:
        raise ValueError("P4 manifest is missing manifest_sha256.")
    unsigned = copy.deepcopy(payload)
    unsigned.pop("manifest_sha256", None)
    if canonical_json_sha256(unsigned) != expected:
        raise ValueError("P4 manifest digest is inconsistent.")


def _forbidden_scope_tokens(value: str | Path) -> set[str]:
    path = Path(value)
    exact_components = {component.lower() for component in path.parts[:-1] if component}
    final_tokens = {
        token for token in re.split(r"[^a-z0-9]+", path.stem.lower()) if token
    }
    return (exact_components | final_tokens) & _FORBIDDEN_SCOPE_TOKENS


def enforce_p4_development_access(
    fold_ids: Iterable[int],
    *,
    scope_labels: Iterable[str | Path] = (),
) -> tuple[int, ...]:
    """Reject calibration folds and Dev/Test paths before any payload is read."""

    folds = tuple(int(value) for value in fold_ids)
    if not folds:
        raise ValueError("P4.0 source preparation requires at least one fold.")
    if len(folds) != len(set(folds)):
        raise ValueError(f"P4.0 fold list contains duplicates: {folds}.")
    forbidden_folds = sorted(set(folds) - set(P4_DEVELOPMENT_FOLDS))
    if forbidden_folds:
        raise PermissionError(
            "P4.0 source preparation may only read folds 0-7; "
            f"blocked folds: {forbidden_folds}."
        )
    for label in scope_labels:
        forbidden = _forbidden_scope_tokens(label)
        if forbidden:
            raise PermissionError(
                f"P4.0 source preparation cannot access Dev/Test scope: {label!s}."
            )
    return folds


def parse_p4_development_folds(value: str) -> tuple[int, ...]:
    """Parse comma-separated fold ids or inclusive ranges such as ``0-7``."""

    values: list[int] = []
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"Invalid descending fold range: {token}.")
            values.extend(range(start, end + 1))
        else:
            values.append(int(token))
    return enforce_p4_development_access(values)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required P4 audit artifact is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"{label} does not contain a valid SHA-256 digest.")
    return digest


def _validate_local_descriptor(
    path: Path,
    descriptor: dict,
    *,
    label: str,
) -> str:
    expected = _require_sha256(descriptor.get("sha256"), label=label)
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} hash mismatch: expected {expected}, found {actual}.")
    expected_bytes = descriptor.get("bytes")
    if expected_bytes is not None and int(expected_bytes) != path.stat().st_size:
        raise ValueError(f"{label} byte count differs from its archive descriptor.")
    return actual


def _validate_sha256_sidecar(path: Path, expected: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Missing feature-cache SHA sidecar: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").strip().split()
    if not fields or fields[0] != expected:
        raise ValueError("Feature-cache SHA sidecar differs from the archive manifest.")


def _artifact_descriptors(stage: dict, group: str, stage_name: str) -> list[dict]:
    values = list(stage.get(group) or [])
    if not values:
        raise ValueError(f"Archived stage {stage_name!r} has no {group} proof.")
    for index, descriptor in enumerate(values):
        _require_sha256(
            dict(descriptor).get("sha256"),
            label=f"{stage_name} {group}[{index}]",
        )
        if not str(dict(descriptor).get("path", "")):
            raise ValueError(
                f"Archived stage {stage_name!r} has an empty artifact path."
            )
    return values


def _validate_archived_pipeline(
    pipeline: dict,
    *,
    fold_id: int,
    train_digest: str,
    heldout_digest: str,
) -> dict:
    if pipeline.get("kind") != FULL_CHAIN_PIPELINE_KIND:
        raise ValueError("Not a full-chain OOF pipeline manifest.")
    if int(pipeline.get("format_version", -1)) != 1:
        raise ValueError("Unsupported archived pipeline-manifest version.")
    if int(pipeline.get("fold_id", -1)) != int(fold_id):
        raise ValueError("Pipeline manifest has the wrong fold id.")
    if pipeline.get("sealed") is not True:
        raise ValueError("Full-chain OOF pipeline is not sealed.")
    if pipeline.get("test_accessed") is not False:
        raise ValueError("Full-chain OOF pipeline accessed Test.")
    if pipeline.get("train_record_ids_sha256") != train_digest:
        raise ValueError("Pipeline training-id digest differs from the fold proof.")
    if pipeline.get("heldout_record_ids_sha256") != heldout_digest:
        raise ValueError("Pipeline heldout-id digest differs from the fold proof.")

    stages = dict(pipeline.get("stages") or {})
    stage_report = {}
    for name in REQUIRED_PIPELINE_STAGES:
        stage = dict(stages.get(name) or {})
        if stage.get("status") != "complete":
            raise ValueError(f"Required archived OOF stage {name!r} is incomplete.")
        if stage.get("test_accessed") is not False:
            raise ValueError(f"Archived OOF stage {name!r} accessed Test.")
        inputs = _artifact_descriptors(stage, "inputs", name)
        outputs = _artifact_descriptors(stage, "outputs", name)
        item = {
            "status": "complete",
            "test_accessed": False,
            "input_artifacts": len(inputs),
            "output_artifacts": len(outputs),
        }
        if name in SUPERVISED_PIPELINE_STAGES:
            if stage.get("heldout_excluded") is not True:
                raise ValueError(
                    f"Supervised archived stage {name!r} lacks heldout exclusion."
                )
            if stage.get("train_record_ids_sha256") != train_digest:
                raise ValueError(
                    f"Supervised archived stage {name!r} used another train split."
                )
            config = dict(stage.get("config") or {})
            checkpoint = dict(stage.get("checkpoint") or {})
            item.update(
                {
                    "heldout_excluded": True,
                    "config_sha256": _require_sha256(
                        config.get("sha256"),
                        label=f"{name} config",
                    ),
                    "checkpoint_sha256": _require_sha256(
                        checkpoint.get("sha256"),
                        label=f"{name} checkpoint",
                    ),
                }
            )
        stage_report[name] = item

    for revision in list(pipeline.get("source_revision_history") or []):
        if dict(revision).get("test_accessed") is not False:
            raise ValueError("Pipeline source-revision history accessed Test.")
    return stage_report


def _validate_proof_ids(proof: dict, fold_id: int) -> tuple[list[str], list[str]]:
    if proof.get("kind") != FULL_CHAIN_PROOF_KIND:
        raise ValueError("Not a full-chain OOF fold proof.")
    if int(proof.get("format_version", -1)) != 1:
        raise ValueError("Unsupported full-chain fold-proof version.")
    if int(proof.get("fold_id", -1)) != int(fold_id):
        raise ValueError("Full-chain fold proof has the wrong fold id.")
    if int(proof.get("num_folds", -1)) != 10:
        raise ValueError("Full-chain fold proof must declare ten folds.")
    if proof.get("excluded_heldout") is not True:
        raise ValueError("Full-chain fold proof lacks heldout exclusion.")

    train_ids = [str(value) for value in proof.get("training_record_ids") or []]
    heldout_ids = [str(value) for value in proof.get("heldout_record_ids") or []]
    if not train_ids or not heldout_ids:
        raise ValueError("Full-chain fold proof has an empty Train or heldout split.")
    if len(train_ids) != len(set(train_ids)):
        raise ValueError("Full-chain fold proof repeats a training record id.")
    if len(heldout_ids) != len(set(heldout_ids)):
        raise ValueError("Full-chain fold proof repeats a heldout record id.")
    if set(train_ids) & set(heldout_ids):
        raise ValueError("Full-chain fold proof has Train/heldout overlap.")
    return train_ids, heldout_ids


def validate_archived_full_chain_fold(
    fold_dir: str | Path,
    *,
    expected_fold_id: int,
) -> dict:
    """Validate a cleaned fold without requiring deleted live checkpoints."""

    fold_id = enforce_p4_development_access([expected_fold_id])[0]
    directory = Path(fold_dir)
    enforce_p4_development_access([fold_id], scope_labels=[directory])

    archive_path = directory / "fold_archive_manifest.json"
    proof_path = directory / "fold_proof.json"
    pipeline_path = directory / "pipeline_manifest.json"
    features_path = directory / "heldout_features.pt"

    archive = _read_json(archive_path)
    proof = _read_json(proof_path)
    pipeline = _read_json(pipeline_path)
    if archive.get("kind") != FULL_CHAIN_ARCHIVE_KIND:
        raise ValueError("Not a full-chain OOF archive manifest.")
    if int(archive.get("format_version", -1)) != 1:
        raise ValueError("Unsupported fold-archive version.")
    if archive.get("status") != "cleaned":
        raise ValueError("P4 requires a sealed and cleaned full-chain OOF fold.")
    if int(archive.get("fold_id", -1)) != fold_id:
        raise ValueError("Fold archive has the wrong fold id.")
    if archive.get("pipeline_sealed") is not True:
        raise ValueError("Fold archive does not assert pipeline_sealed=true.")
    if archive.get("test_accessed") is not False:
        raise ValueError("Fold archive accessed Test.")

    feature_sha = _validate_local_descriptor(
        features_path,
        dict(archive.get("heldout_features") or {}),
        label="heldout feature cache",
    )
    proof_sha = _validate_local_descriptor(
        proof_path,
        dict(archive.get("fold_proof") or {}),
        label="fold proof",
    )
    pipeline_sha = _validate_local_descriptor(
        pipeline_path,
        dict(archive.get("pipeline_manifest") or {}),
        label="pipeline manifest",
    )
    _validate_sha256_sidecar(features_path, feature_sha)

    if (
        _require_sha256(
            proof.get("pipeline_manifest_sha256"),
            label="proof pipeline manifest",
        )
        != pipeline_sha
    ):
        raise ValueError("Fold proof references another pipeline manifest.")

    train_ids, heldout_ids = _validate_proof_ids(proof, fold_id)
    train_digest = stable_id_digest(train_ids)
    heldout_digest = stable_id_digest(heldout_ids)
    stages = _validate_archived_pipeline(
        pipeline,
        fold_id=fold_id,
        train_digest=train_digest,
        heldout_digest=heldout_digest,
    )

    proof_artifacts = {
        str(key): _require_sha256(value, label=f"proof artifact {key}")
        for key, value in dict(proof.get("artifact_sha256") or {}).items()
    }
    archive_artifacts = {
        str(key): _require_sha256(value, label=f"archive artifact {key}")
        for key, value in dict(archive.get("proof_artifact_sha256") or {}).items()
    }
    if proof_artifacts != archive_artifacts:
        raise ValueError("Archive artifact hashes differ from the fold proof.")
    if any(
        not list(paths)
        for paths in dict(archive.get("proof_artifact_matches") or {}).values()
    ):
        raise ValueError("Archive could not map every proof artifact to its pipeline.")

    pre = dict(archive.get("pre_cleanup_validation") or {})
    post = dict(archive.get("post_cleanup_validation") or {})
    required_pre = (
        "all_required_stages_complete",
        "pipeline_sealed",
        "fixed_top4_valid",
        "self_contained_payload",
        "artifact_hashes_verified",
    )
    if not all(pre.get(key) is True for key in required_pre):
        raise ValueError("Fold archive failed its pre-cleanup validation.")
    if pre.get("test_accessed") is not False:
        raise ValueError("Pre-cleanup validation accessed Test.")
    if post.get("self_contained_reload") is not True:
        raise ValueError("Fold archive failed its post-cleanup self-contained reload.")
    if post.get("test_accessed") is not False:
        raise ValueError("Post-cleanup validation accessed Test.")

    payload = torch.load(features_path, map_location="cpu")
    validated = validate_fold_oof_payload(
        payload,
        expected_fold_id=fold_id,
        expected_record_ids=heldout_ids,
        require_reliability=True,
    )
    metadata = dict(validated["metadata"])
    if metadata.get("excluded_heldout") is not True:
        raise ValueError("Compact full-chain cache lacks heldout exclusion.")
    if metadata.get("fold_proof_sha256") != proof_sha:
        raise ValueError("Compact full-chain cache references another fold proof.")
    if dict(metadata.get("artifact_sha256") or {}) != proof_artifacts:
        raise ValueError("Compact full-chain cache artifact hashes changed.")
    if [str(value) for value in metadata.get("training_record_ids") or []] != train_ids:
        raise ValueError("Compact cache training ids differ from the fold proof.")
    if [
        str(value) for value in metadata.get("heldout_record_ids") or []
    ] != heldout_ids:
        raise ValueError("Compact cache heldout ids differ from the fold proof.")

    contains_span_coordinates = all(
        "span_candidates" in dict(batch.get("expanded") or {})
        for batch in validated["batches"]
    )
    restored_formal_cache = directory / "candidates" / "heldout_r16.pt"
    restored_formal_cache_valid = False
    if restored_formal_cache.is_file():
        restored_formal_cache_valid = sha256_file(
            restored_formal_cache
        ) == proof_artifacts.get("formal_cache")

    if int(archive.get("records", -1)) != len(heldout_ids):
        raise ValueError("Fold archive record count differs from the fold proof.")
    if int(post.get("records", -1)) != len(heldout_ids):
        raise ValueError("Post-cleanup record count differs from the fold proof.")

    return {
        "fold_id": fold_id,
        "status": "PASSED",
        "records": len(heldout_ids),
        "training_records": len(train_ids),
        "train_record_ids_sha256": train_digest,
        "heldout_record_ids_sha256": heldout_digest,
        "heldout_feature_sha256": feature_sha,
        "fold_proof_sha256": proof_sha,
        "pipeline_manifest_sha256": pipeline_sha,
        "pipeline_source_tree_sha256": str(pipeline.get("source_tree_sha256", "")),
        "stage_provenance": stages,
        "artifact_sha256": proof_artifacts,
        "formal_span_coordinates_in_compact_cache": contains_span_coordinates,
        "restored_formal_cache_present": restored_formal_cache.is_file(),
        "restored_formal_cache_hash_valid": restored_formal_cache_valid,
        "formal_span_identity_available": bool(
            contains_span_coordinates or restored_formal_cache_valid
        ),
        "test_accessed": False,
        "_payload": payload,
        "_heldout_record_ids": heldout_ids,
    }


def _fixed_type_scores(record: dict) -> torch.Tensor:
    candidates = record["type_candidates"].long()
    scores = record["type_base_scores"].float()
    fixed = record["fixed_type_ids"].long().unsqueeze(-1)
    matches = candidates.eq(fixed)
    if not torch.all(matches.any(dim=-1) | ~record["span_mask"].bool()):
        raise ValueError(
            f"Record {selector_record_id(record)} has a fixed type outside its candidates."
        )
    return scores.masked_fill(~matches, -torch.inf).max(dim=-1).values


def gold_free_selector_record(record: dict) -> dict:
    """Copy only observable candidate fields; labels are never returned."""

    metadata = dict(record.get("metadata") or {})
    output = {
        key: record[key].detach().cpu().contiguous().clone()
        for key in P4_SELECTOR_RECORD_FIELDS
    }
    output["fixed_type_scores"] = _fixed_type_scores(record).cpu()
    output["source_formal_candidate_mask"] = (
        record["formal_candidate_mask"].detach().cpu().bool().contiguous().clone()
    )
    output["metadata"] = {
        key: copy.deepcopy(metadata.get(key)) for key in P4_SELECTOR_METADATA_FIELDS
    }
    assert_gold_free_payload(output)
    return output


def validate_label_free_selector_source(
    payload: dict,
    *,
    expected_fold_id: int,
    expected_record_ids: list[str],
) -> dict:
    """Validate only observable D1 fields without consulting stored labels."""

    fold_id = enforce_p4_development_access([expected_fold_id])[0]
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("kind") != "stage1_candidate_selector_oof":
        raise ValueError("Not a Stage1 selector OOF candidate source.")
    if int(metadata.get("format_version", -1)) != 1:
        raise ValueError("Unsupported Stage1 selector candidate-source version.")
    if metadata.get("scope") != "oof_fold" or metadata.get("oof") is not True:
        raise ValueError("P4 candidate source must be an individual OOF fold.")
    if metadata.get("test_accessed") is not False:
        raise ValueError("P4 candidate source accessed Test.")
    if int(metadata.get("fold_id", -1)) != fold_id:
        raise ValueError("P4 candidate source has the wrong fold id.")
    if int(metadata.get("num_folds", -1)) != 10:
        raise ValueError("P4 candidate source must declare ten OOF folds.")
    for key in (
        "candidate_config_sha256",
        "stage1_checkpoint_sha256",
        "data_source_sha256",
        "fold_manifest_sha256",
        "source_tree_sha256",
        "reference_fold_proof_sha256",
    ):
        if not str(metadata.get(key, "")):
            raise ValueError(f"P4 candidate source lacks provenance field {key!r}.")

    formal_source_id = int(metadata.get("formal_source_id", -1))
    records = list(payload.get("records") or [])
    record_ids: list[str] = []
    for record in records:
        record_id = selector_record_id(record)
        if not record_id:
            raise ValueError("P4 candidate source contains an empty record id.")
        record_ids.append(record_id)
        missing = [
            key
            for key in (*P4_SELECTOR_RECORD_FIELDS, "formal_candidate_mask")
            if key not in record
        ]
        if missing:
            raise ValueError(
                f"P4 candidate record {record_id} lacks observable fields: {missing}."
            )
        spans = record["span_candidates"]
        span_mask = record["span_mask"]
        if spans.dtype != torch.int64 or spans.ndim != 2 or spans.size(-1) != 2:
            raise ValueError(f"P4 candidate record {record_id} has invalid spans.")
        if span_mask.dtype != torch.bool or span_mask.shape != spans.shape[:-1]:
            raise ValueError(f"P4 candidate record {record_id} has invalid span mask.")
        count = int(spans.size(0))
        for key in (
            "span_base_scores",
            "span_source_ids",
            "span_lengths",
            "fixed_type_ids",
            "base_region_indices",
            "formal_candidate_mask",
        ):
            if record[key].ndim != 1 or record[key].size(0) != count:
                raise ValueError(
                    f"P4 candidate record {record_id} has invalid {key} shape."
                )
        if (
            record["span_features"].ndim != 2
            or record["span_features"].size(0) != count
        ):
            raise ValueError(
                f"P4 candidate record {record_id} has invalid span features."
            )
        if (
            record["type_candidates"].ndim != 2
            or record["type_candidates"].size(0) != count
            or record["type_base_scores"].shape != record["type_candidates"].shape
        ):
            raise ValueError(
                f"P4 candidate record {record_id} has invalid type candidates."
            )
        if not torch.equal(
            record["formal_candidate_mask"].bool(),
            record["span_source_ids"].eq(formal_source_id),
        ):
            raise ValueError(
                f"P4 candidate record {record_id} has an invalid source formal mask."
            )
        valid_spans = spans[span_mask]
        if valid_spans.numel() and (
            valid_spans[:, 0].lt(0).any()
            or valid_spans[:, 1].le(valid_spans[:, 0]).any()
        ):
            raise ValueError(
                f"P4 candidate record {record_id} has an invalid half-open span."
            )
        span_tuples = [tuple(value) for value in valid_spans.tolist()]
        if len(span_tuples) != len(set(span_tuples)):
            raise ValueError(f"P4 candidate record {record_id} repeats a span.")
        if not torch.equal(
            record["span_lengths"].long(),
            spans[:, 1] - spans[:, 0],
        ):
            raise ValueError(
                f"P4 candidate record {record_id} has inconsistent span lengths."
            )
        sources = list(
            dict(record.get("metadata") or {}).get("candidate_sources") or []
        )
        if sources and len(sources) != count:
            raise ValueError(
                f"P4 candidate record {record_id} has inconsistent source names."
            )

    if len(record_ids) != len(set(record_ids)):
        raise ValueError("P4 candidate source repeats record ids.")
    if int(metadata.get("records", -1)) != len(records):
        raise ValueError("P4 candidate-source record count is inconsistent.")
    if metadata.get("record_ids") != record_ids:
        raise ValueError("P4 candidate-source record order metadata changed.")
    if stable_id_digest(record_ids) != metadata.get("record_ids_sha256"):
        raise ValueError("P4 candidate-source record digest changed.")
    if record_ids != [str(value) for value in expected_record_ids]:
        raise ValueError("P4 candidate-source records differ from full-chain OOF.")
    return {"metadata": metadata, "records": records, "record_ids": record_ids}


def _walk_field_names(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _walk_field_names(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_field_names(item)


def assert_gold_free_payload(payload: dict) -> None:
    forbidden = sorted(set(_walk_field_names(payload)) & _GOLD_DERIVED_FIELD_NAMES)
    if forbidden:
        raise ValueError(f"P4 gold-free payload contains label fields: {forbidden}.")


def build_gold_free_candidate_payload(
    selector_payload: dict,
    *,
    fold_id: int,
    source_cache_sha256: str,
    full_chain_provenance: dict,
) -> dict:
    """Materialize an auditable candidate source without copying labels."""

    fold_id = enforce_p4_development_access([fold_id])[0]
    validated = validate_label_free_selector_source(
        selector_payload,
        expected_fold_id=fold_id,
        expected_record_ids=full_chain_provenance["_heldout_record_ids"],
    )
    metadata = dict(validated["metadata"])
    if (
        metadata.get("reference_fold_proof_sha256")
        != full_chain_provenance["fold_proof_sha256"]
    ):
        raise ValueError("Candidate cache references another full-chain fold proof.")
    candidate_config = dict(metadata.get("candidate_config") or {})
    if candidate_config.get("inject_gold_types") not in {None, False}:
        raise ValueError("P4 candidate generation cannot inject gold types.")

    records = [gold_free_selector_record(record) for record in validated["records"]]
    record_ids = [str(record["metadata"]["record_id"]) for record in records]
    output_metadata = {
        "kind": P4_GOLD_FREE_CACHE_KIND,
        "format_version": P4_FORMAT_VERSION,
        "scope": "oof_source_development",
        "fold_id": fold_id,
        "allowed_folds": list(P4_DEVELOPMENT_FOLDS),
        "records": len(records),
        "record_ids": record_ids,
        "record_ids_sha256": stable_id_digest(record_ids),
        "gold_free": True,
        "gold_values_used_for_candidate_generation": False,
        "test_accessed": False,
        "source_candidate_cache_sha256": _require_sha256(
            source_cache_sha256,
            label="candidate source cache",
        ),
        "full_chain_feature_sha256": full_chain_provenance["heldout_feature_sha256"],
        "full_chain_fold_proof_sha256": full_chain_provenance["fold_proof_sha256"],
        "full_chain_stage1_checkpoint_sha256": full_chain_provenance[
            "stage_provenance"
        ]["stage1"]["checkpoint_sha256"],
        "full_chain_formal_span_identity_available": full_chain_provenance[
            "formal_span_identity_available"
        ],
    }
    for key in P4_SELECTOR_METADATA_PROVENANCE_FIELDS:
        if key in metadata:
            output_metadata[key] = copy.deepcopy(metadata[key])
    output = {"metadata": output_metadata, "records": records}
    assert_gold_free_payload(output)
    return output


def validate_gold_free_candidate_payload(
    payload: dict,
    *,
    expected_fold_id: int,
) -> dict:
    fold_id = enforce_p4_development_access([expected_fold_id])[0]
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("kind") != P4_GOLD_FREE_CACHE_KIND:
        raise ValueError("Not a P4 gold-free candidate cache.")
    if int(metadata.get("format_version", -1)) != P4_FORMAT_VERSION:
        raise ValueError("Unsupported P4 gold-free candidate-cache version.")
    if int(metadata.get("fold_id", -1)) != fold_id:
        raise ValueError("P4 candidate cache has the wrong fold id.")
    if metadata.get("gold_free") is not True:
        raise ValueError("P4 candidate cache is not marked gold_free=true.")
    if metadata.get("gold_values_used_for_candidate_generation") is not False:
        raise ValueError("P4 candidate cache does not assert label-free generation.")
    if metadata.get("test_accessed") is not False:
        raise ValueError("P4 candidate cache accessed Test.")
    records = list(payload.get("records") or [])
    if int(metadata.get("records", -1)) != len(records):
        raise ValueError("P4 candidate-cache record count is inconsistent.")
    record_ids = [
        str(dict(record.get("metadata") or {}).get("record_id", ""))
        for record in records
    ]
    if metadata.get("record_ids") != record_ids:
        raise ValueError("P4 candidate-cache record order changed.")
    if stable_id_digest(record_ids) != metadata.get("record_ids_sha256"):
        raise ValueError("P4 candidate-cache record-id digest changed.")
    assert_gold_free_payload(payload)
    return {"metadata": metadata, "records": records, "record_ids": record_ids}


def _full_chain_observable_rows(payload: dict) -> dict[str, dict[str, torch.Tensor]]:
    rows: dict[str, dict[str, torch.Tensor]] = {}
    for batch in payload["batches"]:
        expanded = dict(batch["expanded"])
        fine = dict(batch["fine_outputs"])
        for index, record_id in enumerate(batch["record_ids"]):
            rows[str(record_id)] = {
                "span_mask": expanded["span_mask"][index].bool().cpu(),
                "span_source_ids": expanded["span_source_ids"][index].long().cpu(),
                "fixed_type_ids": fine["fixed_type_ids"][index].long().cpu(),
                "deployment_span_mask": batch["deployment_span_mask"][index]
                .bool()
                .cpu(),
            }
    return rows


def audit_cross_cache_candidate_identity(
    full_chain_payload: dict,
    gold_free_payload: dict,
) -> dict:
    """Audit observable row alignment without inferring missing span coordinates."""

    validated = validate_gold_free_candidate_payload(
        gold_free_payload,
        expected_fold_id=int(gold_free_payload["metadata"]["fold_id"]),
    )
    full_rows = _full_chain_observable_rows(full_chain_payload)
    records = {
        str(record["metadata"]["record_id"]): record for record in validated["records"]
    }
    if set(full_rows) != set(records):
        raise ValueError("Full-chain and candidate-source record sets differ.")

    exact_rows = 0
    count_matches = 0
    source_matches = 0
    type_matches = 0
    deployment_count_matches = 0
    mismatched_record_ids: list[str] = []
    for record_id in validated["record_ids"]:
        full = full_rows[record_id]
        source = records[record_id]
        full_mask = full["span_mask"]
        source_mask = source["span_mask"].bool()
        count_match = int(full_mask.sum()) == int(source_mask.sum())
        count_matches += int(count_match)
        source_match = False
        type_match = False
        if count_match:
            source_match = torch.equal(
                full["span_source_ids"][full_mask],
                source["span_source_ids"][source_mask],
            )
            type_match = torch.equal(
                full["fixed_type_ids"][full_mask],
                source["fixed_type_ids"][source_mask],
            )
        source_matches += int(source_match)
        type_matches += int(type_match)
        source_formal = source["source_formal_candidate_mask"].bool() & source_mask
        deployment_match = int(full["deployment_span_mask"].sum()) == int(
            source_formal.sum()
        )
        deployment_count_matches += int(deployment_match)
        exact = count_match and source_match and type_match
        exact_rows += int(exact)
        if not exact and len(mismatched_record_ids) < 25:
            mismatched_record_ids.append(record_id)

    full_checkpoint = str(
        gold_free_payload["metadata"]["full_chain_stage1_checkpoint_sha256"]
    )
    candidate_checkpoint = str(
        gold_free_payload["metadata"].get("stage1_checkpoint_sha256", "")
    )
    compact_coordinates = bool(
        gold_free_payload["metadata"].get(
            "full_chain_formal_span_identity_available",
            False,
        )
    )
    return {
        "records": len(records),
        "candidate_count_matches": count_matches,
        "source_sequence_matches": source_matches,
        "type_sequence_matches": type_matches,
        "observable_row_identity_matches": exact_rows,
        "deployment_count_matches": deployment_count_matches,
        "observable_row_identity_ratio": exact_rows / max(len(records), 1),
        "mismatched_record_id_examples": mismatched_record_ids,
        "candidate_stage1_checkpoint_sha256": candidate_checkpoint,
        "full_chain_stage1_checkpoint_sha256": full_checkpoint,
        "stage1_checkpoint_identity": candidate_checkpoint == full_checkpoint,
        "formal_span_identity_available": compact_coordinates,
        "index_attachment_permitted": False,
        "index_attachment_blockers": [
            reason
            for reason, blocked in (
                (
                    "full_chain_formal_span_coordinates_unavailable",
                    not compact_coordinates,
                ),
                (
                    "candidate_and_full_chain_stage1_checkpoints_differ",
                    candidate_checkpoint != full_checkpoint,
                ),
                (
                    "candidate_row_sequences_are_not_exact",
                    exact_rows != len(records),
                ),
            )
            if blocked
        ],
    }


def candidate_source_statistics(payload: dict) -> dict:
    metadata = dict(payload["metadata"])
    validated = validate_gold_free_candidate_payload(
        payload,
        expected_fold_id=int(metadata["fold_id"]),
    )
    source2id = {
        str(key): int(value)
        for key, value in dict(metadata.get("source2id") or {}).items()
    }
    id2source = {value: key for key, value in source2id.items()}
    source_counts: Counter[str] = Counter()
    records_with_nonformal = 0
    total = 0
    source_formal = 0
    finite_span_scores = True
    finite_type_scores = True
    region_null = 0
    for record in validated["records"]:
        mask = record["span_mask"].bool()
        total += int(mask.sum())
        formal = record["source_formal_candidate_mask"].bool() & mask
        source_formal += int(formal.sum())
        records_with_nonformal += int(bool((mask & ~formal).any()))
        for source_id in record["span_source_ids"][mask].tolist():
            source_counts[id2source.get(int(source_id), f"unknown:{source_id}")] += 1
        finite_span_scores &= bool(
            torch.isfinite(record["span_base_scores"][mask]).all()
        )
        finite_type_scores &= bool(
            torch.isfinite(record["fixed_type_scores"][mask]).all()
        )
        null_index = int(record["metadata"].get("null_region_index", -1))
        region_null += int(record["base_region_indices"][mask].eq(null_index).sum())
    return {
        "records": len(validated["records"]),
        "candidate_rows": total,
        "source_formal_rows": source_formal,
        "nonformal_rows": total - source_formal,
        "records_with_nonformal_rows": records_with_nonformal,
        "candidate_rows_by_source": dict(sorted(source_counts.items())),
        "null_region_rows": region_null,
        "real_region_rows": total - region_null,
        "finite_span_scores": finite_span_scores,
        "finite_fixed_type_scores": finite_type_scores,
        "gold_free": True,
    }


def source_seal_blockers(
    provenance_reports: list[dict],
    alignment_reports: list[dict],
    *,
    score_composition_frozen: bool = False,
    require_cross_cache_index_join: bool = False,
) -> list[str]:
    blockers: list[str] = []
    if not provenance_reports:
        blockers.append("no_full_chain_oof_folds_validated")
    expected = set(P4_DEVELOPMENT_FOLDS)
    observed = {int(report["fold_id"]) for report in provenance_reports}
    if observed != expected:
        blockers.append("source_development_folds_0_7_incomplete")
    if any(
        not report["formal_span_identity_available"] for report in provenance_reports
    ):
        blockers.append("frozen_model_g_formal_span_coordinates_unavailable")
    if require_cross_cache_index_join and any(
        not report["index_attachment_permitted"] for report in alignment_reports
    ):
        blockers.append("cross_cache_candidate_identity_not_proven")
    if not score_composition_frozen:
        # Current compact D1 sources provide a region action but not a calibrated
        # region confidence.  The score definition must remain open until the
        # actionability audit determines whether scalar score components suffice.
        blockers.append("joint_candidate_score_composition_not_frozen")
    return sorted(set(blockers))


def build_source_manifest(
    *,
    provenance_reports: list[dict],
    candidate_descriptors: list[dict],
    alignment_reports: list[dict],
    blockers: list[str],
    implementation: dict | None = None,
) -> dict:
    """Build a deterministic sealed manifest or an explicit blocked draft."""

    folds = sorted(int(item["fold_id"]) for item in provenance_reports)
    status = "SEALED" if not blockers else "BLOCKED_UNSEALED"
    manifest = {
        "kind": P4_SOURCE_MANIFEST_KIND,
        "format_version": P4_FORMAT_VERSION,
        "phase": "P4.0_source_preparation",
        "status": status,
        "sealed": not blockers,
        "allowed_folds_read": folds,
        "calibration_folds_opened": False,
        "dev_accessed": False,
        "test_accessed": False,
        "p4_1_authorized": False,
        "implementation": copy.deepcopy(implementation or {}),
        "candidate_source_definition": {
            "name": "independent_stage1_oof_candidate_replay",
            "origin": "D1 compact OOF candidate caches",
            "candidate_generation_uses_gold": False,
            "candidate_rows": "all valid rows retained until frozen Model-G filtering",
            "candidate_sources": ["stage1", "viterbi", "kbest", "perturbation"],
            "source_formal_flag_is_not_model_g_formal": True,
            "cross_cache_index_join_forbidden": True,
        },
        "feature_schema": {
            "observable_fields": [
                "span[start,end)",
                "span_source",
                "span_base_score",
                "span_length",
                "fixed_type_id",
                "fixed_type_score",
                "base_region_index",
                "source_formal_flag",
            ],
            "dense_feature": "source Stage1 span_features fp16",
            "grounding_confidence_available": False,
            "gold_fields": [],
        },
        "score_composition": {
            "status": "UNFROZEN" if blockers else "FROZEN",
            "allowed_frozen_components": [
                "span_base_score",
                "fixed_type_score",
                "candidate_source_id",
                "span_length",
                "base_region_is_null",
            ],
            "gold_components_forbidden": True,
            "dev_selection_forbidden": True,
        },
        "deduplication": {
            "key": [
                "record_id",
                "span_start",
                "span_end",
                "fixed_type_id",
                "base_region_index",
            ],
            "applied_before_max_one": True,
        },
        "non_overlap": {
            "interval_convention": "word-space half-open [start,end)",
            "rule": "candidate span must not overlap any frozen Model-G formal span",
            "requires_frozen_formal_span_coordinates": True,
            "currently_evaluable": all(
                item["formal_span_identity_available"] for item in provenance_reports
            ),
        },
        "deterministic_tie_break": {
            "status": "PROVISIONAL" if blockers else "FROZEN",
            "order": [
                "composed_score_desc",
                "source_priority_asc",
                "span_start_asc",
                "span_end_asc",
                "fixed_type_id_asc",
                "base_region_index_asc",
            ],
            "source_priority": [
                "stage1",
                "viterbi",
                "kbest",
                "perturbation",
            ],
        },
        "full_chain_provenance": [
            {key: value for key, value in item.items() if not key.startswith("_")}
            for item in sorted(provenance_reports, key=lambda value: value["fold_id"])
        ],
        "candidate_artifacts": sorted(
            candidate_descriptors,
            key=lambda value: value["fold_id"],
        ),
        "cross_cache_alignment": sorted(
            alignment_reports,
            key=lambda value: value["fold_id"],
        ),
        "seal_blockers": sorted(set(blockers)),
    }
    return attach_manifest_sha256(manifest)


def json_safe_provenance(report: dict) -> dict:
    return {key: value for key, value in report.items() if not key.startswith("_")}


def tensor_payload_sha256(path: str | Path) -> str:
    return sha256_file(path)


def validate_finite_number(value: float, *, label: str) -> None:
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} is not finite.")
