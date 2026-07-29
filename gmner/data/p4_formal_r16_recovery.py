"""Exact-artifact recovery contract for P4.0 Phase B.

The recovery gate is deliberately all-or-nothing. Candidate files are hashed
before any payload is loaded. If one development fold lacks an exact archived
formal-cache hash match, no cache payload is opened and no formal-span sidecar
is produced.
"""

from __future__ import annotations

import copy
import fnmatch
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable

import torch

from gmner.data.null_release_oof_cache import (
    sha256_file,
    stable_id_digest,
    validate_fold_oof_payload,
)
from gmner.data.p4_actionability_contract import (
    P4_DEVELOPMENT_FOLDS,
    P4_PROVENANCE_REPORT_KIND,
    canonical_json_sha256,
    enforce_p4_development_access,
)
from gmner.data.record_candidate_dataset import (
    SUPPORTED_CACHE_FORMAT_VERSIONS,
)


P4_FORMAL_RECOVERY_REPORT_KIND = "p4_formal_r16_recovery_report"
P4_FORMAL_SPAN_SIDECAR_KIND = "p4_formal_model_g_span_sidecar"
P4_FORMAL_RECOVERY_FORMAT_VERSION = 1

P4_FORMAL_RECOVERY_BLOCKED = "P4_0_FORMAL_ARTIFACT_RECOVERY_BLOCKED"
P4_FORMAL_RECOVERY_COMPLETE = "P4_0_FORMAL_ARTIFACT_RECOVERY_COMPLETE"

DEFAULT_FORMAL_CACHE_PATTERNS = (
    "heldout_r16.pt",
    "*heldout*r16*.pt",
    "*formal*r16*.pt",
    "*r16*formal*.pt",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CALIBRATION_DIRECTORY_RE = re.compile(r"^fold(?:8|9)$", re.IGNORECASE)
_FORBIDDEN_SCOPE_NAMES = {"dev", "test"}


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} is not a valid SHA-256 digest.")
    return digest


def _json_object(path: str | Path) -> dict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source}.")
    return payload


def _contains_locked_scope(path: Path) -> bool:
    for component in path.parts[:-1]:
        if _CALIBRATION_DIRECTORY_RE.fullmatch(component):
            return True
        if component.lower() in _FORBIDDEN_SCOPE_NAMES:
            return True
    final_tokens = {
        token
        for token in re.split(r"[^a-z0-9]+", path.stem.lower())
        if token
    }
    return bool(final_tokens & _FORBIDDEN_SCOPE_NAMES)


def formal_cache_expectations(
    phase_a_report: dict,
    *,
    folds: Iterable[int] = P4_DEVELOPMENT_FOLDS,
) -> list[dict]:
    """Extract exact formal-cache descriptors from the sealed Phase A audit."""

    fold_ids = enforce_p4_development_access(folds)
    if phase_a_report.get("kind") != P4_PROVENANCE_REPORT_KIND:
        raise ValueError("Input is not a P4.0 Phase A provenance report.")
    access = dict(phase_a_report.get("access_contract") or {})
    if access.get("calibration_folds_opened") is not False:
        raise PermissionError("Phase A opened calibration folds.")
    if access.get("dev_accessed") is not False:
        raise PermissionError("Phase A accessed Dev.")
    if access.get("test_accessed") is not False:
        raise PermissionError("Phase A accessed Test.")
    if access.get("oracle_labels_computed") is not False:
        raise PermissionError("Phase A computed Oracle labels.")

    by_fold = {
        int(item["fold_id"]): dict(item)
        for item in list(phase_a_report.get("provenance") or [])
    }
    if set(by_fold) != set(fold_ids):
        raise ValueError(
            "Phase A provenance folds differ from the P4 development partition."
        )

    expectations: list[dict] = []
    for fold_id in fold_ids:
        item = by_fold[fold_id]
        if item.get("status") != "PASSED":
            raise ValueError(f"Phase A fold {fold_id} did not pass provenance.")
        if item.get("test_accessed") is not False:
            raise PermissionError(f"Phase A fold {fold_id} accessed Test.")
        artifacts = dict(item.get("artifact_sha256") or {})
        expectations.append(
            {
                "fold_id": fold_id,
                "expected_sha256": _require_sha256(
                    artifacts.get("formal_cache"),
                    label=f"fold {fold_id} formal cache",
                ),
                "expected_records": int(item.get("records", -1)),
                "expected_record_ids_sha256": _require_sha256(
                    item.get("heldout_record_ids_sha256"),
                    label=f"fold {fold_id} heldout record ids",
                ),
                "fold_proof_sha256": _require_sha256(
                    item.get("fold_proof_sha256"),
                    label=f"fold {fold_id} proof",
                ),
                "heldout_feature_sha256": _require_sha256(
                    item.get("heldout_feature_sha256"),
                    label=f"fold {fold_id} full-chain features",
                ),
            }
        )
    if any(item["expected_records"] <= 0 for item in expectations):
        raise ValueError("Phase A contains an invalid heldout record count.")
    return expectations


def discover_formal_cache_candidates(
    *,
    search_roots: Iterable[str | Path],
    explicit_candidates: Iterable[str | Path] = (),
    patterns: Iterable[str] = DEFAULT_FORMAL_CACHE_PATTERNS,
) -> dict:
    """Find direct cache files without opening calibration, Dev, or Test data."""

    pattern_values = tuple(str(value).lower() for value in patterns)
    if not pattern_values:
        raise ValueError("At least one formal-cache filename pattern is required.")

    roots = [Path(value).resolve() for value in search_roots]
    explicit = [Path(value).resolve() for value in explicit_candidates]
    if not roots and not explicit:
        raise ValueError("Formal-cache recovery requires a search root or candidate.")

    discovered: set[Path] = set()
    skipped_locked: set[Path] = set()
    missing_roots: list[Path] = []
    for path in explicit:
        if _contains_locked_scope(path):
            skipped_locked.add(path)
        elif path.is_file():
            discovered.add(path)

    for root in roots:
        if _contains_locked_scope(root):
            raise PermissionError(f"Locked P4 scope cannot be searched: {root}")
        if not root.exists():
            missing_roots.append(root)
            continue
        if root.is_file():
            if any(fnmatch.fnmatch(root.name.lower(), value) for value in pattern_values):
                discovered.add(root)
            continue
        for current, directory_names, file_names in os.walk(root, topdown=True):
            current_path = Path(current)
            allowed_directories = []
            for name in directory_names:
                directory = (current_path / name).resolve()
                if _contains_locked_scope(directory):
                    skipped_locked.add(directory)
                else:
                    allowed_directories.append(name)
            directory_names[:] = allowed_directories
            for name in file_names:
                if not any(
                    fnmatch.fnmatch(name.lower(), value)
                    for value in pattern_values
                ):
                    continue
                resolved = (current_path / name).resolve()
                if _contains_locked_scope(resolved):
                    skipped_locked.add(resolved)
                else:
                    discovered.add(resolved)

    return {
        "patterns": list(pattern_values),
        "search_roots": [path.as_posix() for path in roots],
        "missing_search_roots": [path.as_posix() for path in missing_roots],
        "candidate_paths": [
            path.as_posix() for path in sorted(discovered, key=lambda value: value.as_posix())
        ],
        "locked_paths_skipped": [
            path.as_posix()
            for path in sorted(skipped_locked, key=lambda value: value.as_posix())
        ],
        "calibration_folds_opened": False,
        "dev_accessed": False,
        "test_accessed": False,
    }


def hash_recovery_candidates(candidate_paths: Iterable[str | Path]) -> list[dict]:
    """Hash direct candidates; this function never deserializes them."""

    descriptors: list[dict] = []
    for value in candidate_paths:
        path = Path(value).resolve()
        if _contains_locked_scope(path):
            raise PermissionError(f"Locked P4 artifact cannot be hashed: {path}")
        if not path.is_file():
            raise FileNotFoundError(f"Recovery candidate is missing: {path}")
        descriptors.append(
            {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return sorted(descriptors, key=lambda item: item["path"])


def match_exact_formal_caches(
    expectations: list[dict],
    candidate_descriptors: list[dict],
) -> list[dict]:
    """Match folds by archived hash only; paths and names are non-authoritative."""

    candidates_by_sha: dict[str, list[dict]] = {}
    for descriptor in candidate_descriptors:
        digest = _require_sha256(
            descriptor.get("sha256"),
            label="formal-cache recovery candidate",
        )
        candidates_by_sha.setdefault(digest, []).append(dict(descriptor))

    matches: list[dict] = []
    for expectation in expectations:
        expected = expectation["expected_sha256"]
        copies = sorted(
            candidates_by_sha.get(expected, []),
            key=lambda item: item["path"],
        )
        matches.append(
            {
                **copy.deepcopy(expectation),
                "status": "RECOVERED_EXACT_HASH" if copies else "MISSING_EXACT_HASH",
                "exact_match_count": len(copies),
                "restored_path": copies[0]["path"] if copies else None,
                "actual_sha256": copies[0]["sha256"] if copies else None,
                "file_size": copies[0]["bytes"] if copies else None,
                "duplicate_exact_paths": [
                    item["path"] for item in copies[1:]
                ],
            }
        )
    return matches


def _cache_record_id(record: dict) -> str:
    return str((record.get("metadata") or {}).get("record_id", ""))


def _validate_formal_prediction_rows(record: dict, *, record_id: str) -> None:
    spans = torch.as_tensor(record["span_candidates"]).long()
    span_mask = torch.as_tensor(record["span_mask"]).bool()
    source_ids = torch.as_tensor(record["span_source_ids"]).long()
    fixed_types = torch.as_tensor(record["fixed_type_ids"]).long()
    base_regions = torch.as_tensor(record["base_region_indices"]).long()
    if spans.ndim != 2 or spans.size(-1) != 2:
        raise ValueError(f"Formal record {record_id} has invalid span coordinates.")
    count = spans.size(0)
    for value, label in (
        (span_mask, "span_mask"),
        (source_ids, "span_source_ids"),
        (fixed_types, "fixed_type_ids"),
        (base_regions, "base_region_indices"),
    ):
        if value.ndim != 1 or value.size(0) != count:
            raise ValueError(f"Formal record {record_id} has invalid {label}.")
    tokens = list((record.get("metadata") or {}).get("tokens") or [])
    for start, end in spans.tolist():
        if int(start) < 0 or int(end) <= int(start) or int(end) > len(tokens):
            raise ValueError(
                f"Formal record {record_id} has a non-word-space half-open span."
            )
    if len({tuple(map(int, row)) for row in spans.tolist()}) != count:
        raise ValueError(f"Formal record {record_id} contains duplicate span rows.")

    row_by_span = {
        tuple(map(int, span)): index for index, span in enumerate(spans.tolist())
    }
    predictions = list(
        (record.get("metadata") or {}).get("stage1_predictions") or []
    )
    for prediction in predictions:
        span = tuple(map(int, prediction.get("span") or []))
        if span not in row_by_span:
            raise ValueError(
                f"Formal Stage1 prediction {span} is absent from record {record_id}."
            )
        row = row_by_span[span]
        if not bool(span_mask[row]) or int(source_ids[row]) != 0:
            raise ValueError(
                f"Formal Stage1 prediction {span} lacks an active Stage1 row."
            )
        if int(fixed_types[row]) != int(prediction["type_id"]):
            raise ValueError(
                f"Formal Stage1 type differs from its row in record {record_id}."
            )
        if int(base_regions[row]) != int(prediction["region_index"]):
            raise ValueError(
                f"Formal Stage1 region differs from its row in record {record_id}."
            )


def load_and_validate_exact_formal_cache(
    path: str | Path,
    *,
    expectation: dict,
    expected_record_ids: list[str],
) -> dict:
    """Load only after the exact archive hash has passed."""

    source = Path(path).resolve()
    expected_sha = _require_sha256(
        expectation.get("expected_sha256"),
        label="expected formal cache",
    )
    actual_sha = sha256_file(source)
    if actual_sha != expected_sha:
        raise ValueError(
            "Formal cache hash mismatch; payload was not loaded: "
            f"expected {expected_sha}, found {actual_sha}."
        )

    payload = torch.load(source, map_location="cpu")
    if not isinstance(payload, dict) or "records" not in payload:
        raise ValueError("Recovered formal cache has an invalid payload.")
    metadata = dict(payload.get("metadata") or {})
    if int(metadata.get("format_version", -1)) not in SUPPORTED_CACHE_FORMAT_VERSIONS:
        raise ValueError("Recovered formal cache has an unsupported format.")
    fold_id = int(expectation["fold_id"])
    if metadata.get("oof_heldout") is not True:
        raise ValueError("Recovered formal cache is not marked oof_heldout=true.")
    if int(metadata.get("oof_fold_id", -1)) != fold_id:
        raise ValueError("Recovered formal cache has the wrong OOF fold id.")

    records = list(payload["records"])
    record_ids = [_cache_record_id(record) for record in records]
    if len(record_ids) != len(set(record_ids)) or any(not value for value in record_ids):
        raise ValueError("Recovered formal cache has missing or duplicate record ids.")
    if record_ids != [str(value) for value in expected_record_ids]:
        raise ValueError("Recovered formal-cache record order differs from heldout OOF.")
    if len(records) != int(expectation["expected_records"]):
        raise ValueError("Recovered formal cache has the wrong record count.")
    if stable_id_digest(record_ids) != expectation["expected_record_ids_sha256"]:
        raise ValueError("Recovered formal-cache record-id digest differs from Phase A.")
    for record_id, record in zip(record_ids, records):
        _validate_formal_prediction_rows(record, record_id=record_id)
    return {
        "path": source,
        "sha256": actual_sha,
        "metadata": metadata,
        "records": records,
        "record_ids": record_ids,
        "record_ids_sha256": stable_id_digest(record_ids),
    }


def _prediction_digest(records: list[dict]) -> str:
    canonical = []
    for record in records:
        predictions = sorted(
            [
                {
                    "span_start": int(item["span_start"]),
                    "span_end": int(item["span_end"]),
                    "type_id": int(item["type_id"]),
                    "region_index": int(item["region_index"]),
                    "region_is_null": bool(item["region_is_null"]),
                }
                for item in list(record.get("formal_predictions") or [])
            ],
            key=lambda item: (
                item["span_start"],
                item["span_end"],
                item["type_id"],
                item["region_index"],
                item["region_is_null"],
            ),
        )
        canonical.append(
            {
                "record_id": str(record["record_id"]),
                "formal_predictions": predictions,
            }
        )
    canonical.sort(key=lambda item: item["record_id"])
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def build_formal_span_sidecar(
    *,
    formal_cache: dict,
    full_chain_payload: dict,
    expectation: dict,
    full_chain_feature_sha256: str,
    generator_git_head: str,
    generator_path: str,
    generator_sha256: str,
) -> dict:
    """Join exact R16 coordinates to frozen Model-G final deployment outputs."""

    fold_id = int(expectation["fold_id"])
    expected_ids = [str(value) for value in formal_cache["record_ids"]]
    validated = validate_fold_oof_payload(
        full_chain_payload,
        expected_fold_id=fold_id,
        expected_record_ids=expected_ids,
        require_reliability=True,
    )
    records_by_id = {
        _cache_record_id(record): record for record in formal_cache["records"]
    }
    sidecar_records: list[dict] = []
    for batch in validated["batches"]:
        expanded = dict(batch["expanded"])
        fine = dict(batch["fine_outputs"])
        hierarchy = dict(batch["hierarchy_outputs"])
        for row, record_id in enumerate(batch["record_ids"]):
            record_id = str(record_id)
            formal = records_by_id[record_id]
            spans = torch.as_tensor(formal["span_candidates"]).long()
            formal_sources = torch.as_tensor(formal["span_source_ids"]).long()
            span_count = spans.size(0)
            selected = batch["deployment_span_mask"][row].bool()
            valid_spans = expanded["span_mask"][row].bool()
            if selected.size(0) < span_count or valid_spans.size(0) < span_count:
                raise ValueError(
                    f"Compact full-chain rows are shorter than R16 for {record_id}."
                )
            if selected[span_count:].any():
                raise ValueError(
                    f"Deployment selected a row outside exact R16 for {record_id}."
                )
            selected = selected[:span_count]
            if (selected & ~valid_spans[:span_count]).any():
                raise ValueError(
                    f"Deployment selected an invalid aligned row for {record_id}."
                )
            if (selected & formal_sources.ne(0)).any():
                raise ValueError(
                    f"Deployment selected a non-Stage1 formal row for {record_id}."
                )

            fixed_types = hierarchy["fixed_type_ids"][row, :span_count].long()
            fine_types = fine["fixed_type_ids"][row, :span_count].long()
            if (selected & fixed_types.ne(fine_types)).any():
                raise ValueError(
                    f"Hierarchy and Fine type disagree for {record_id}."
                )
            region_mask = expanded["region_mask"][row].bool()
            null_mask = expanded["region_is_null"][row].bool()
            if int(null_mask.sum().item()) != 1:
                raise ValueError(f"Record {record_id} lacks exactly one NULL region.")
            candidate_mask = fine["candidate_mask"][row, :span_count].bool()
            real_mask = (
                candidate_mask
                & region_mask.unsqueeze(0)
                & ~null_mask.unsqueeze(0)
            )
            current_visible = batch["current_visible"][row, :span_count].bool()
            if (selected & current_visible & ~real_mask.any(dim=-1)).any():
                raise ValueError(
                    f"Visible deployment has no real region for {record_id}."
                )
            fine_top1 = (
                fine["final_region_logits"][row, :span_count]
                .float()
                .masked_fill(~real_mask, -1e4)
                .argmax(dim=-1)
            )
            null_index = int(null_mask.float().argmax().item())
            final_regions = torch.where(
                current_visible,
                fine_top1,
                torch.full_like(fine_top1, null_index),
            )

            predictions = []
            for index in torch.nonzero(selected, as_tuple=False).flatten().tolist():
                start, end = spans[index].tolist()
                region_index = int(final_regions[index].item())
                predictions.append(
                    {
                        "span_start": int(start),
                        "span_end": int(end),
                        "type_id": int(fixed_types[index].item()),
                        "region_index": region_index,
                        "region_is_null": bool(null_mask[region_index].item()),
                    }
                )
            sidecar_records.append(
                {
                    "record_id": record_id,
                    "formal_predictions": predictions,
                }
            )

    if [item["record_id"] for item in sidecar_records] != expected_ids:
        raise ValueError("Formal sidecar record order differs from exact R16.")
    prediction_count = sum(
        len(item["formal_predictions"]) for item in sidecar_records
    )
    prediction_sha = _prediction_digest(sidecar_records)
    sidecar = {
        "kind": P4_FORMAL_SPAN_SIDECAR_KIND,
        "format_version": P4_FORMAL_RECOVERY_FORMAT_VERSION,
        "phase": "P4.0_formal_r16_recovery",
        "fold_id": fold_id,
        "records": len(sidecar_records),
        "formal_prediction_count": prediction_count,
        "record_ids_sha256": stable_id_digest(expected_ids),
        "formal_prediction_sha256": prediction_sha,
        "source_formal_cache_sha256": formal_cache["sha256"],
        "source_full_chain_feature_sha256": _require_sha256(
            full_chain_feature_sha256,
            label="full-chain feature cache",
        ),
        "generator": {
            "git_head": str(generator_git_head),
            "path": str(generator_path),
            "sha256": _require_sha256(
                generator_sha256,
                label="formal-sidecar generator",
            ),
        },
        "coordinate_contract": "word-space half-open [start,end)",
        "test_accessed": False,
        "records_payload": sidecar_records,
    }
    sidecar["sidecar_sha256"] = canonical_json_sha256(sidecar)
    return sidecar


def validate_formal_span_sidecar(sidecar: dict) -> dict:
    """Validate canonical digest and final frozen prediction preservation."""

    if sidecar.get("kind") != P4_FORMAL_SPAN_SIDECAR_KIND:
        raise ValueError("Not a P4 formal-span sidecar.")
    if int(sidecar.get("format_version", -1)) != P4_FORMAL_RECOVERY_FORMAT_VERSION:
        raise ValueError("Unsupported P4 formal-span sidecar version.")
    if sidecar.get("test_accessed") is not False:
        raise PermissionError("Formal-span sidecar accessed Test.")
    expected_sidecar_sha = _require_sha256(
        sidecar.get("sidecar_sha256"),
        label="formal-span sidecar",
    )
    unsigned = copy.deepcopy(sidecar)
    unsigned.pop("sidecar_sha256", None)
    if canonical_json_sha256(unsigned) != expected_sidecar_sha:
        raise ValueError("Formal-span sidecar digest is inconsistent.")

    records = list(sidecar.get("records_payload") or [])
    record_ids = [str(item.get("record_id", "")) for item in records]
    if any(not value for value in record_ids) or len(record_ids) != len(set(record_ids)):
        raise ValueError("Formal-span sidecar has missing or duplicate record ids.")
    if len(records) != int(sidecar.get("records", -1)):
        raise ValueError("Formal-span sidecar record count is inconsistent.")
    if stable_id_digest(record_ids) != sidecar.get("record_ids_sha256"):
        raise ValueError("Formal-span sidecar record-id digest is inconsistent.")
    prediction_count = sum(
        len(item.get("formal_predictions") or []) for item in records
    )
    if prediction_count != int(sidecar.get("formal_prediction_count", -1)):
        raise ValueError("Formal-span sidecar prediction count is inconsistent.")
    prediction_sha = _prediction_digest(records)
    if prediction_sha != sidecar.get("formal_prediction_sha256"):
        raise ValueError("Formal-span sidecar prediction digest is inconsistent.")
    return {
        "records": len(records),
        "formal_prediction_count": prediction_count,
        "record_ids_sha256": stable_id_digest(record_ids),
        "formal_prediction_sha256": prediction_sha,
        "formal_predictions_preserved": True,
        "test_accessed": False,
    }


def build_blocked_recovery_report(
    *,
    expectations: list[dict],
    discovery: dict,
    candidate_descriptors: list[dict],
    matches: list[dict],
    implementation: dict,
    phase_a_report_path: str,
    phase_a_report_sha256: str,
    external_search_inventory: dict | None = None,
) -> dict:
    """Build the only valid report when at least one exact artifact is absent."""

    missing = [
        int(item["fold_id"])
        for item in matches
        if item["status"] != "RECOVERED_EXACT_HASH"
    ]
    if not missing:
        raise ValueError("Blocked recovery report requires at least one missing fold.")
    report = {
        "kind": P4_FORMAL_RECOVERY_REPORT_KIND,
        "format_version": P4_FORMAL_RECOVERY_FORMAT_VERSION,
        "phase": "P4.0_formal_r16_recovery",
        "status": P4_FORMAL_RECOVERY_BLOCKED,
        "implementation": copy.deepcopy(implementation),
        "phase_a": {
            "path": str(phase_a_report_path),
            "sha256": _require_sha256(
                phase_a_report_sha256,
                label="Phase A report",
            ),
            "provenance_status": "PASSED",
            "source_manifest_status": "BLOCKED_UNSEALED",
        },
        "access_contract": {
            "folds_considered": list(P4_DEVELOPMENT_FOLDS),
            "calibration_folds_opened": False,
            "dev_accessed": False,
            "test_accessed": False,
            "gold_values_used": False,
            "oracle_labels_computed": False,
            "p4_1_code_executed": False,
        },
        "search": {
            **copy.deepcopy(discovery),
            "candidate_files_hashed": len(candidate_descriptors),
            "candidate_descriptors": copy.deepcopy(candidate_descriptors),
            "external_inventory": copy.deepcopy(external_search_inventory),
        },
        "expected_formal_caches": copy.deepcopy(expectations),
        "fold_recovery": copy.deepcopy(matches),
        "missing_exact_artifact_folds": missing,
        "payload_deserialization": {
            "attempted": False,
            "reason": "all_folds_must_match_archived_sha256_before_any_payload_load",
        },
        "formal_span_sidecars": {
            "generated": False,
            "count": 0,
        },
        "preservation_validation": {
            "run": False,
            "reason": "exact_formal_r16_set_incomplete",
        },
        "score_composition": {
            "status": "UNFROZEN",
            "reason": "formal_coordinate_recovery_precedes_score_freeze",
        },
        "source_manifest": {
            "status": "BLOCKED_UNSEALED",
            "sealed": False,
            "non_overlap_currently_evaluable": False,
        },
        "next_authorized_state": P4_FORMAL_RECOVERY_BLOCKED,
        "prohibited_substitutions": [
            "D1 coordinates",
            "candidate row indices",
            "approximate decode",
            "regenerated full-chain artifacts without separate authorization",
        ],
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def validate_recovery_report(report: dict) -> None:
    if report.get("kind") != P4_FORMAL_RECOVERY_REPORT_KIND:
        raise ValueError("Not a P4 formal R16 recovery report.")
    if report.get("status") != P4_FORMAL_RECOVERY_BLOCKED:
        raise ValueError("Only blocked Phase B reports are currently supported.")
    expected = _require_sha256(
        report.get("report_sha256"),
        label="formal R16 recovery report",
    )
    unsigned = copy.deepcopy(report)
    unsigned.pop("report_sha256", None)
    if canonical_json_sha256(unsigned) != expected:
        raise ValueError("Formal R16 recovery report digest is inconsistent.")
    access = dict(report.get("access_contract") or {})
    if access.get("calibration_folds_opened") is not False:
        raise PermissionError("Recovery report opened calibration folds.")
    if access.get("dev_accessed") is not False:
        raise PermissionError("Recovery report accessed Dev.")
    if access.get("test_accessed") is not False:
        raise PermissionError("Recovery report accessed Test.")
    if access.get("oracle_labels_computed") is not False:
        raise PermissionError("Recovery report computed Oracle labels.")
    if dict(report.get("source_manifest") or {}).get("sealed") is not False:
        raise ValueError("Blocked recovery report cannot seal the source manifest.")
    if dict(report.get("formal_span_sidecars") or {}).get("generated") is not False:
        raise ValueError("Blocked recovery report cannot claim sidecar generation.")
