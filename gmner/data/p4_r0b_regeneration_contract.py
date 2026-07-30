"""Contracts for authorized P4-R0-B full-chain OOF regeneration."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch

from gmner.data.full_chain_oof_contract import source_tree_sha256
from gmner.data.null_release_oof_cache import (
    sha256_file,
    stable_id_digest,
    validate_fold_oof_payload,
)


P4_R0B_PREREGISTRATION_KIND = (
    "p4_r0_b_full_chain_oof_regeneration_preregistration"
)
P4_R0B_FORMAT_VERSION = 1
P4_R0B_ARTIFACT_IDENTITY = "REGENERATED_FULL_CHAIN_OOF_R16"
P4_R0B_EXECUTION_FOLDS = tuple(range(8))
P4_R0B_FOLD_REPORT_KIND = "p4_r0_b_fold_semantic_consistency_report"
P4_R0B_AGGREGATE_REPORT_KIND = "p4_r0_b_regeneration_aggregate_report"
P4_R0B_M33A_CACHE_KIND = "p4_r0_b_m33a_formal_oof"
P4_R0B_M33A_CACHE_VERSION = 1
P4_R0B_M33A_REQUIRED_STAGES = (
    "stage1",
    "candidate_caches",
    "hierarchical",
    "coarse",
    "fine",
    "evidence",
    "formal_materialize",
)
P4_R0B_M33A_SUPERVISED_STAGES = (
    "stage1",
    "hierarchical",
    "coarse",
    "fine",
    "evidence",
)

SEMANTIC_TENSOR_PATHS = (
    "base_is_null",
    "current_visible",
    "deployment_span_mask",
    "expanded.span_mask",
    "expanded.span_source_ids",
    "expanded.type_candidates",
    "expanded.region_mask",
    "expanded.region_is_null",
    "expanded.region_detector_scores",
    "fine_outputs.candidate_mask",
    "fine_outputs.fine_top4_indices",
    "fine_outputs.fine_top4_valid_mask",
    "fine_outputs.promoted_candidate_mask",
    "fine_outputs.fixed_type_ids",
    "hierarchy_outputs.fixed_type_ids",
)

CONTINUOUS_DIAGNOSTIC_PATHS = (
    "fine_outputs.final_region_logits",
)

M33A_FINE_KEYS = (
    "candidate_mask",
    "final_region_logits",
    "fine_top4_indices",
    "fine_top4_valid_mask",
    "promoted_candidate_mask",
    "fixed_type_ids",
)
M33A_EXPANDED_KEYS = (
    "span_mask",
    "span_source_ids",
    "type_candidates",
    "region_mask",
    "region_is_null",
    "region_detector_scores",
)


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git_head(root: str | Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(root).resolve(),
        text=True,
    ).strip()


def validate_r0b_preregistration(payload: dict) -> None:
    if payload.get("kind") != P4_R0B_PREREGISTRATION_KIND:
        raise ValueError("Not a P4-R0-B preregistration.")
    if int(payload.get("format_version", -1)) != P4_R0B_FORMAT_VERSION:
        raise ValueError("Unsupported P4-R0-B preregistration version.")
    if payload.get("artifact_identity") != P4_R0B_ARTIFACT_IDENTITY:
        raise ValueError("P4-R0-B artifact identity changed.")
    supersedes = dict(payload.get("supersedes_prior_lock") or {})
    if (
        supersedes.get("scope")
        != "authorization.r0_b_full_oof_retraining"
        or supersedes.get("new_value") is not True
        or len(str(supersedes.get("sha256", ""))) != 64
    ):
        raise PermissionError("P4-R0-B prior-lock supersession is invalid.")
    authorization = dict(payload.get("authorization") or {})
    if authorization.get("full_chain_retraining") is not True:
        raise PermissionError("P4-R0-B full-chain retraining is not authorized.")
    folds = tuple(int(value) for value in authorization.get("execution_folds") or [])
    if folds != P4_R0B_EXECUTION_FOLDS:
        raise PermissionError("P4-R0-B execution folds must be exactly 0-7.")
    if authorization.get("upstream_official_dev_checkpoint_validation") is not True:
        raise PermissionError("The fixed upstream Dev validation contract changed.")
    locked = (
        "p4_dev_candidate_generation",
        "p4_dev_oracle_or_threshold_selection",
        "p4_dev_evaluation",
        "folds_8_9_execution",
        "test_access",
        "p4_oracle",
        "p4_1",
        "downstream_rebuild",
        "formal_sidecar_generation",
        "p4_source_attachment",
    )
    enabled = [name for name in locked if authorization.get(name) is not False]
    if enabled:
        raise PermissionError(f"P4-R0-B locked authorization changed: {enabled}.")
    source = dict(payload.get("source_contract") or {})
    if int(source.get("seed", -1)) != 42:
        raise ValueError("P4-R0-B seed must remain 42.")
    if source.get("checkpoint_reuse") is not False:
        raise PermissionError("P4-R0-B cannot reuse an old checkpoint.")
    chain = dict(payload.get("chain_contract") or {})
    if (
        chain.get("identity") != "M3.3A_FORMAL_BEST_CHAIN"
        or chain.get("siglip2") is not False
        or chain.get("fusion_reliability") is not False
        or chain.get("null_release") is not False
    ):
        raise PermissionError("P4-R0-B must use only the formal M3.3A chain.")
    if tuple(payload.get("required_stages") or ()) != (
        P4_R0B_M33A_REQUIRED_STAGES
    ):
        raise PermissionError("P4-R0-B required stages changed.")
    storage = dict(payload.get("storage_contract") or {})
    paths = {
        str(storage.get("work_root", "")),
        str(storage.get("output_root", "")),
        str(storage.get("legacy_evidence_root", "")),
    }
    if "" in paths or len(paths) != 3:
        raise ValueError("P4-R0-B storage roots must be distinct.")


def regeneration_metadata(
    *,
    authorization_sha256: str,
    fold_id: int,
    experiment_id: str,
) -> dict:
    if int(fold_id) not in P4_R0B_EXECUTION_FOLDS:
        raise PermissionError("P4-R0-B cannot generate folds 8-9.")
    if len(str(authorization_sha256)) != 64:
        raise ValueError("Invalid P4-R0-B authorization SHA256.")
    return {
        "artifact_identity": P4_R0B_ARTIFACT_IDENTITY,
        "regeneration_authorization_sha256": str(authorization_sha256),
        "regeneration_fold_id": int(fold_id),
        "regeneration_experiment_id": str(experiment_id),
    }


def validate_regeneration_metadata(
    metadata: dict,
    *,
    authorization_sha256: str,
    fold_id: int,
    experiment_id: str,
) -> None:
    expected = regeneration_metadata(
        authorization_sha256=authorization_sha256,
        fold_id=fold_id,
        experiment_id=experiment_id,
    )
    observed = {key: metadata.get(key) for key in expected}
    if observed != expected:
        raise ValueError(
            f"Regenerated artifact identity differs: {observed} != {expected}."
        )


def tree_sha256(path: str | Path) -> dict:
    root = Path(path).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    files = [item for item in root.rglob("*") if item.is_file()]
    digest = hashlib.sha256()
    total_bytes = 0
    for item in sorted(files):
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
        total_bytes += item.stat().st_size
    return {
        "path": str(root),
        "files": len(files),
        "bytes": total_bytes,
        "tree_sha256": digest.hexdigest(),
    }


def file_bundle_sha256(paths: Iterable[str | Path]) -> dict:
    resolved = sorted(Path(value).resolve() for value in paths)
    digest = hashlib.sha256()
    descriptors = []
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(path)
        file_digest = sha256_file(path)
        encoded = path.name.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
        digest.update(bytes.fromhex(file_digest))
        descriptors.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": file_digest,
            }
        )
    return {"sha256": digest.hexdigest(), "files": descriptors}


def validate_fold_cleanup_path(
    target: str | Path,
    *,
    allowed_root: str | Path,
    fold_id: int,
    allow_fold_root: bool = False,
) -> Path:
    resolved_target = Path(target).resolve()
    resolved_root = Path(allowed_root).resolve()
    if int(fold_id) not in P4_R0B_EXECUTION_FOLDS:
        raise PermissionError("Cleanup is limited to authorized folds 0-7.")
    try:
        relative = resolved_target.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError("Cleanup target is outside its authorized root.") from error
    if not relative.parts or relative.parts[0] != f"fold{int(fold_id)}":
        raise ValueError("Cleanup target does not belong to the requested fold.")
    if (
        resolved_target == resolved_root / f"fold{int(fold_id)}"
        and not allow_fold_root
    ):
        raise ValueError("Cleanup cannot remove the retained fold root itself.")
    return resolved_target


def build_regeneration_fold_manifest(
    archived: dict,
    *,
    root: str | Path,
    stage1_config: str | Path,
    authorization_path: str | Path,
    authorization: dict,
) -> dict:
    validate_r0b_preregistration(authorization)
    project_root = Path(root).resolve()
    config_path = Path(stage1_config).resolve()
    authorization_file = Path(authorization_path).resolve()
    output = copy.deepcopy(archived)
    output["source_tree_sha256"] = source_tree_sha256(project_root)
    output["git_commit"] = git_head(project_root)
    output["config"] = str(config_path)
    output["config_sha256"] = sha256_file(config_path)
    output["test_accessed"] = False
    output["regeneration"] = {
        **regeneration_metadata(
            authorization_sha256=sha256_file(authorization_file),
            fold_id=0,
            experiment_id=str(authorization["experiment_id"]),
        ),
        "regeneration_fold_id": None,
        "execution_folds": list(P4_R0B_EXECUTION_FOLDS),
        "chain_contract": copy.deepcopy(authorization["chain_contract"]),
        "authorization_path": str(authorization_file),
        "upstream_validation_dev_access": True,
        "p4_dev_access": False,
        "test_accessed": False,
    }
    output.setdefault("source_revision_history", []).append(
        {
            "kind": "P4_R0_B_FULL_CHAIN_REGENERATION",
            "previous_source_tree_sha256": archived.get("source_tree_sha256"),
            "source_tree_sha256": output["source_tree_sha256"],
            "artifact_identity": P4_R0B_ARTIFACT_IDENTITY,
            "authorization_sha256": sha256_file(authorization_file),
            "execution_folds": list(P4_R0B_EXECUTION_FOLDS),
            "test_accessed": False,
        }
    )
    return output


def _compact_tensor(
    value: torch.Tensor,
    *,
    preserve_float32: bool = False,
) -> torch.Tensor:
    tensor = value.detach().cpu().contiguous()
    if tensor.is_floating_point() and not preserve_float32:
        tensor = tensor.to(torch.float16)
    return tensor


def pack_m33a_formal_batch(
    context: dict,
    *,
    fold_id: int,
) -> dict:
    expanded = dict(context["expanded"])
    fine = dict(context["fine_outputs"])
    hierarchy = dict(context["hierarchy_outputs"])
    missing_fine = [key for key in M33A_FINE_KEYS if key not in fine]
    missing_expanded = [key for key in M33A_EXPANDED_KEYS if key not in expanded]
    if missing_fine or missing_expanded:
        raise ValueError(
            "Cannot pack M3.3A formal state: "
            f"fine={missing_fine}, expanded={missing_expanded}."
        )
    if "fixed_type_ids" not in hierarchy:
        raise ValueError("M3.3A hierarchy output lacks fixed_type_ids.")
    record_ids = [
        str(item.get("record_id", ""))
        for item in expanded["metadata"]
    ]
    if any(not value for value in record_ids):
        raise ValueError("M3.3A formal state contains an empty record id.")
    return {
        "fold_id": int(fold_id),
        "record_ids": record_ids,
        "fine_outputs": {
            key: _compact_tensor(
                fine[key],
                preserve_float32=key == "final_region_logits",
            )
            for key in M33A_FINE_KEYS
        },
        "hierarchy_outputs": {
            "fixed_type_ids": _compact_tensor(hierarchy["fixed_type_ids"])
        },
        "expanded": {
            key: _compact_tensor(expanded[key])
            for key in M33A_EXPANDED_KEYS
        },
        "current_visible": _compact_tensor(context["current_visible"]),
        "base_is_null": _compact_tensor(context["base_is_null"]),
        "deployment_span_mask": _compact_tensor(
            context["deployment_span_mask"]
        ),
    }


def validate_m33a_formal_oof_payload(
    payload: dict,
    *,
    expected_fold_id: int,
    expected_record_ids: list[str],
) -> dict:
    metadata = dict(payload.get("metadata") or {})
    if metadata.get("kind") != P4_R0B_M33A_CACHE_KIND:
        raise ValueError("Not a P4-R0-B M3.3A formal-state cache.")
    if int(metadata.get("format_version", -1)) != P4_R0B_M33A_CACHE_VERSION:
        raise ValueError("Unsupported P4-R0-B M3.3A cache version.")
    if int(metadata.get("fold_id", -1)) != int(expected_fold_id):
        raise ValueError("M3.3A formal-state cache has another fold id.")
    if int(metadata.get("num_folds", -1)) != 10:
        raise ValueError("M3.3A formal-state cache must retain the 10-fold split.")
    for excluded in (
        "siglip2_included",
        "reliability_included",
        "null_release_included",
    ):
        if metadata.get(excluded) is not False:
            raise ValueError(f"M3.3A formal-state cache must set {excluded}=false.")
    batches = list(payload.get("batches") or [])
    record_ids = []
    for batch_index, batch in enumerate(batches):
        required = (
            "fold_id",
            "record_ids",
            "fine_outputs",
            "hierarchy_outputs",
            "expanded",
            "current_visible",
            "base_is_null",
            "deployment_span_mask",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise ValueError(
                f"M3.3A batch {batch_index} is missing fields: {missing}."
            )
        if int(batch["fold_id"]) != int(expected_fold_id):
            raise ValueError("M3.3A batch carries another fold id.")
        batch_ids = [str(value) for value in batch["record_ids"]]
        batch_size = int(batch["expanded"]["span_mask"].size(0))
        if len(batch_ids) != batch_size:
            raise ValueError("M3.3A batch record-id count is inconsistent.")
        record_ids.extend(batch_ids)
        fine = dict(batch["fine_outputs"])
        expanded = dict(batch["expanded"])
        if any(key not in fine for key in M33A_FINE_KEYS):
            raise ValueError("M3.3A batch lacks a required Fine field.")
        if any(key not in expanded for key in M33A_EXPANDED_KEYS):
            raise ValueError("M3.3A batch lacks a required expanded field.")
        top4_indices = fine["fine_top4_indices"].long()
        top4_valid = fine["fine_top4_valid_mask"].bool()
        candidate_mask = fine["candidate_mask"].bool()
        if top4_indices.shape != top4_valid.shape or top4_indices.size(-1) != 4:
            raise ValueError("M3.3A fixed Top-4 shape is invalid.")
        if top4_indices.shape[:-1] != candidate_mask.shape[:-1]:
            raise ValueError("M3.3A fixed Top-4 does not align with spans.")
        safe = top4_indices.clamp(0, max(candidate_mask.size(-1) - 1, 0))
        if not torch.all(candidate_mask.gather(-1, safe) | ~top4_valid):
            raise ValueError("M3.3A fixed Top-4 leaves its Fine candidate mask.")
        one_hot = torch.nn.functional.one_hot(
            safe,
            num_classes=candidate_mask.size(-1),
        ).bool()
        if (one_hot & top4_valid.unsqueeze(-1)).sum(dim=-2).gt(1).any():
            raise ValueError("M3.3A fixed Top-4 contains duplicate actions.")
    expected_ids = [str(value) for value in expected_record_ids]
    if record_ids != expected_ids:
        raise ValueError("M3.3A formal-state record order differs.")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("M3.3A formal-state cache has duplicate record ids.")
    if int(metadata.get("records", -1)) != len(record_ids):
        raise ValueError("M3.3A formal-state record count differs.")
    if metadata.get("record_ids_sha256") != stable_id_digest(record_ids):
        raise ValueError("M3.3A formal-state record digest differs.")
    return {
        "metadata": metadata,
        "batches": batches,
        "records": len(record_ids),
    }


def _nested(mapping: dict, path: str) -> torch.Tensor:
    value: Any = mapping
    for component in path.split("."):
        value = value[component]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Semantic path is not a tensor: {path}.")
    return value


def _rows(payload: dict) -> dict[str, tuple[dict, int]]:
    rows: dict[str, tuple[dict, int]] = {}
    for batch in payload["batches"]:
        for index, record_id in enumerate(batch["record_ids"]):
            key = str(record_id)
            if key in rows:
                raise ValueError(f"Repeated compact record id: {key}.")
            rows[key] = (batch, index)
    return rows


def _fine_top1(batch: dict) -> torch.Tensor:
    fine = dict(batch["fine_outputs"])
    logits = fine["final_region_logits"].float()
    candidate_mask = fine["candidate_mask"].bool()
    return logits.masked_fill(~candidate_mask, -torch.inf).argmax(dim=-1)


def compare_compact_semantics(
    reference: dict,
    regenerated: dict,
    *,
    fold_id: int,
    authorization_sha256: str,
    experiment_id: str,
) -> dict:
    expected_ids = [
        str(value)
        for value in dict(reference.get("metadata") or {}).get(
            "heldout_record_ids", []
        )
    ]
    validate_fold_oof_payload(
        reference,
        expected_fold_id=fold_id,
        expected_record_ids=expected_ids,
        require_reliability=True,
    )
    validate_m33a_formal_oof_payload(
        regenerated,
        expected_fold_id=fold_id,
        expected_record_ids=expected_ids,
    )
    validate_regeneration_metadata(
        dict(regenerated["metadata"]),
        authorization_sha256=authorization_sha256,
        fold_id=fold_id,
        experiment_id=experiment_id,
    )
    reference_rows = _rows(reference)
    regenerated_rows = _rows(regenerated)
    id_order_reference = [
        str(value)
        for batch in reference["batches"]
        for value in batch["record_ids"]
    ]
    id_order_regenerated = [
        str(value)
        for batch in regenerated["batches"]
        for value in batch["record_ids"]
    ]
    field_reports = {}
    for path in SEMANTIC_TENSOR_PATHS:
        exact_records = 0
        mismatched_examples = []
        for record_id in id_order_reference:
            old_batch, old_index = reference_rows[record_id]
            new_batch, new_index = regenerated_rows[record_id]
            old_value = _nested(old_batch, path)[old_index]
            new_value = _nested(new_batch, path)[new_index]
            exact = old_value.shape == new_value.shape and torch.equal(
                old_value, new_value
            )
            exact_records += int(exact)
            if not exact and len(mismatched_examples) < 20:
                mismatched_examples.append(record_id)
        field_reports[path] = {
            "exact_records": exact_records,
            "records": len(id_order_reference),
            "exact_ratio": exact_records / max(len(id_order_reference), 1),
            "mismatched_record_id_examples": mismatched_examples,
        }

    top1_exact = 0
    top1_mismatches = []
    for record_id in id_order_reference:
        old_batch, old_index = reference_rows[record_id]
        new_batch, new_index = regenerated_rows[record_id]
        exact = torch.equal(
            _fine_top1(old_batch)[old_index],
            _fine_top1(new_batch)[new_index],
        )
        top1_exact += int(exact)
        if not exact and len(top1_mismatches) < 20:
            top1_mismatches.append(record_id)

    continuous = {}
    for path in CONTINUOUS_DIAGNOSTIC_PATHS:
        comparable = 0
        maximum_error = 0.0
        for record_id in id_order_reference:
            old_batch, old_index = reference_rows[record_id]
            new_batch, new_index = regenerated_rows[record_id]
            old_value = _nested(old_batch, path)[old_index].float()
            new_value = _nested(new_batch, path)[new_index].float()
            if old_value.shape != new_value.shape:
                continue
            comparable += 1
            if old_value.numel():
                maximum_error = max(
                    maximum_error,
                    float((old_value - new_value).abs().max().item()),
                )
        continuous[path] = {
            "shape_comparable_records": comparable,
            "records": len(id_order_reference),
            "max_abs_error": maximum_error,
            "gate_field": False,
        }

    all_fields_exact = all(
        item["exact_records"] == len(id_order_reference)
        for item in field_reports.values()
    )
    return {
        "record_ids_and_order_exact": (
            id_order_reference == id_order_regenerated
        ),
        "records": len(id_order_reference),
        "semantic_fields": field_reports,
        "all_semantic_fields_exact": all_fields_exact,
        "fine_top1_region": {
            "exact_records": top1_exact,
            "records": len(id_order_reference),
            "exact_ratio": top1_exact / max(len(id_order_reference), 1),
            "mismatched_record_id_examples": top1_mismatches,
        },
        "continuous_diagnostics": continuous,
        "gate_passed": (
            id_order_reference == id_order_regenerated
            and all_fields_exact
            and top1_exact == len(id_order_reference)
        ),
    }


def canonical_formal_triple_digest(
    r16_payload: dict,
    compact_payload: dict,
    *,
    fold_id: int,
    authorization_sha256: str,
    experiment_id: str,
) -> dict:
    metadata = dict(r16_payload.get("metadata") or {})
    validate_regeneration_metadata(
        metadata,
        authorization_sha256=authorization_sha256,
        fold_id=fold_id,
        experiment_id=experiment_id,
    )
    records = list(r16_payload.get("records") or [])
    by_id = {
        str(dict(record.get("metadata") or {}).get("record_id", "")): record
        for record in records
    }
    compact_rows = _rows(compact_payload)
    ordered_ids = [
        str(value)
        for batch in compact_payload["batches"]
        for value in batch["record_ids"]
    ]
    if list(by_id) != ordered_ids:
        raise ValueError("Regenerated R16 and compact record order differ.")
    canonical = []
    prediction_count = 0
    for record_id in ordered_ids:
        record = by_id[record_id]
        spans = record["span_candidates"].long()
        formal_sources = record["span_source_ids"].long()
        formal_types = record["fixed_type_ids"].long()
        batch, row = compact_rows[record_id]
        span_mask = batch["expanded"]["span_mask"][row].bool()
        span_count = int(spans.size(0))
        compact_sources = batch["expanded"]["span_source_ids"][row].long()
        if (
            span_mask.ndim != 1
            or span_mask.numel() < span_count
            or compact_sources.numel() < span_count
        ):
            raise ValueError(
                f"Record {record_id} compact span table is shorter than R16."
            )
        if span_mask[span_count:].any():
            raise ValueError(
                f"Record {record_id} compact span table has active padded rows."
            )
        if not torch.equal(compact_sources[:span_count], formal_sources):
            raise ValueError(
                f"Record {record_id} compact span sources do not align with R16."
            )
        formal_stage1 = formal_sources.eq(0)
        if not span_mask[:span_count][formal_stage1].all():
            raise ValueError(
                f"Record {record_id} masks a formal Stage1 span."
            )
        selected_full = (
            batch["deployment_span_mask"][row].bool() & span_mask
        )
        if selected_full[span_count:].any():
            raise ValueError(
                f"Record {record_id} selected a padded compact span."
            )
        selected = selected_full[:span_count]
        if (selected & formal_sources.ne(0)).any():
            raise ValueError(
                f"Record {record_id} selected a non-formal R16 span."
            )
        fine_fixed_type = batch["fine_outputs"]["fixed_type_ids"][
            row, :span_count
        ].long()
        hierarchy_fixed_type = batch["hierarchy_outputs"]["fixed_type_ids"][
            row, :span_count
        ].long()
        if (
            selected
            & fine_fixed_type.ne(hierarchy_fixed_type)
        ).any():
            raise ValueError(
                f"Record {record_id} has inconsistent fixed coarse types."
            )
        if (selected & fine_fixed_type.ne(formal_types)).any():
            raise ValueError(
                f"Record {record_id} changed a formal R16 coarse type."
            )
        visible = batch["current_visible"][row, :span_count].bool()
        region_mask = batch["expanded"]["region_mask"][row].bool()
        null_mask = batch["expanded"]["region_is_null"][row].bool()
        null_indices = null_mask.nonzero(as_tuple=False).flatten()
        if len(null_indices) != 1:
            raise ValueError(f"Record {record_id} has no unique NULL region.")
        null_index = int(null_indices.item())
        candidate_mask = batch["fine_outputs"]["candidate_mask"][
            row, :span_count
        ].bool()
        real_mask = (
            candidate_mask
            & region_mask.unsqueeze(0)
            & ~null_mask.unsqueeze(0)
        )
        if (selected & visible & ~real_mask.any(dim=-1)).any():
            raise ValueError(
                f"Record {record_id} has a visible span without a real candidate."
            )
        fine_top1 = (
            batch["fine_outputs"]["final_region_logits"][row, :span_count]
            .float()
            .masked_fill(~real_mask, -torch.inf)
            .argmax(dim=-1)
            .long()
        )
        triples = []
        for span_index in selected.nonzero(as_tuple=False).flatten().tolist():
            start, end = [int(value) for value in spans[span_index].tolist()]
            if start < 0 or end <= start:
                raise ValueError(f"Record {record_id} has an invalid formal span.")
            region_index = (
                int(fine_top1[span_index].item())
                if bool(visible[span_index].item())
                else null_index
            )
            triples.append(
                {
                    "span_start": start,
                    "span_end": end,
                    "type_id": int(fine_fixed_type[span_index].item()),
                    "region_index": region_index,
                    "region_is_null": region_index == null_index,
                }
            )
        prediction_count += len(triples)
        canonical.append({"record_id": record_id, "formal_predictions": triples})
    return {
        "records": len(canonical),
        "predictions": prediction_count,
        "canonical_formal_triple_sha256": canonical_json_sha256(canonical),
    }
