"""Read-only provenance contract for P4-R0-A checkpoint replay auditing."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


P4_DEVELOPMENT_FOLDS = tuple(range(8))
P4_R0_PREREGISTRATION_KIND = (
    "p4_r0_full_chain_oof_r16_regeneration_preregistration"
)
P4_R0_A_REPORT_KIND = "p4_r0_a_checkpoint_replay_feasibility_report"
P4_R0_FORMAT_VERSION = 1
P4_R0_A_FEASIBLE = "P4_R0_A_CHECKPOINT_REPLAY_FEASIBLE"
P4_R0_A_BLOCKED = "P4_R0_A_CHECKPOINT_REPLAY_BLOCKED"

SUPERVISED_STAGES = (
    "stage1",
    "hierarchical",
    "coarse",
    "fine",
    "evidence",
    "reliability",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LOCKED_DIRECTORY_NAMES = {"dev", "test", "fold8", "fold9"}
P4_R0_EXTERNAL_INVENTORY_KIND = "p4_r0_a_external_storage_inventory"


def canonical_json_sha256(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def enforce_r0_development_access(fold_ids: Iterable[int]) -> tuple[int, ...]:
    folds = tuple(int(value) for value in fold_ids)
    if not folds:
        raise ValueError("P4-R0-A requires at least one development fold.")
    if len(folds) != len(set(folds)):
        raise ValueError("P4-R0-A fold list contains duplicates.")
    forbidden = sorted(set(folds) - set(P4_DEVELOPMENT_FOLDS))
    if forbidden:
        raise PermissionError(f"P4-R0-A cannot open locked folds: {forbidden}.")
    return folds


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_digest(values: Iterable[str]) -> str:
    encoded = json.dumps(
        sorted(str(value) for value in values),
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    digest = str(value or "")
    if not _SHA256_RE.fullmatch(digest):
        raise ValueError(f"{label} is not a valid SHA256 digest.")
    return digest


def _read_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def validate_r0_preregistration(payload: dict) -> None:
    if payload.get("kind") != P4_R0_PREREGISTRATION_KIND:
        raise ValueError("Not a P4-R0 preregistration.")
    if int(payload.get("format_version", -1)) != P4_R0_FORMAT_VERSION:
        raise ValueError("Unsupported P4-R0 preregistration version.")
    authorization = dict(payload.get("authorization") or {})
    if authorization.get("r0_a_read_only_audit") is not True:
        raise PermissionError("R0-A read-only audit is not authorized.")
    forbidden = (
        "r0_a_checkpoint_replay_execution",
        "r0_b_full_oof_retraining",
        "checkpoint_or_cache_deserialization",
        "model_execution",
        "candidate_generation",
        "oracle",
        "p4_1",
        "downstream_rebuild",
        "calibration_folds_8_9",
        "dev_access",
        "test_access",
    )
    enabled = [name for name in forbidden if authorization.get(name) is not False]
    if enabled:
        raise PermissionError(f"P4-R0 locked authorizations changed: {enabled}.")
    folds = enforce_r0_development_access(payload.get("development_folds") or [])
    if folds != P4_DEVELOPMENT_FOLDS:
        raise ValueError("P4-R0 development folds must be exactly 0-7.")
    if tuple(payload.get("required_supervised_stages") or []) != SUPERVISED_STAGES:
        raise ValueError("P4-R0 supervised stage contract changed.")


def _validate_descriptor(path: Path, descriptor: dict, *, label: str) -> None:
    expected = _require_sha256(descriptor.get("sha256"), label=label)
    if not path.is_file():
        raise FileNotFoundError(f"Missing retained provenance file: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch: expected {expected}, found {actual}."
        )


def audit_fold_json_provenance(fold_dir: str | Path, *, fold_id: int) -> dict:
    """Validate retained JSON/config provenance without opening model/data payloads."""

    enforce_r0_development_access([fold_id])
    directory = Path(fold_dir).resolve()
    archive_path = directory / "fold_archive_manifest.json"
    proof_path = directory / "fold_proof.json"
    pipeline_path = directory / "pipeline_manifest.json"
    archive = _read_json(archive_path)
    proof = _read_json(proof_path)
    pipeline = _read_json(pipeline_path)

    if archive.get("kind") != "null_release_oof_fold_archive":
        raise ValueError("Not a retained full-chain fold archive.")
    if archive.get("status") != "cleaned" or archive.get("pipeline_sealed") is not True:
        raise ValueError("Fold archive is not sealed and cleaned.")
    if int(archive.get("fold_id", -1)) != fold_id:
        raise ValueError("Fold archive has the wrong fold id.")
    if archive.get("test_accessed") is not False:
        raise PermissionError("Fold archive accessed Test.")
    _validate_descriptor(
        proof_path,
        dict(archive.get("fold_proof") or {}),
        label=f"fold {fold_id} proof",
    )
    _validate_descriptor(
        pipeline_path,
        dict(archive.get("pipeline_manifest") or {}),
        label=f"fold {fold_id} pipeline",
    )

    if proof.get("kind") != "null_release_full_chain_fold_proof":
        raise ValueError("Not a full-chain fold proof.")
    if pipeline.get("kind") != "null_release_full_chain_fold_pipeline":
        raise ValueError("Not a full-chain pipeline manifest.")
    if int(proof.get("fold_id", -1)) != fold_id:
        raise ValueError("Fold proof has the wrong fold id.")
    if int(pipeline.get("fold_id", -1)) != fold_id:
        raise ValueError("Pipeline has the wrong fold id.")
    if proof.get("excluded_heldout") is not True:
        raise ValueError("Fold proof does not exclude heldout records.")
    if pipeline.get("sealed") is not True or pipeline.get("test_accessed") is not False:
        raise ValueError("Pipeline is unsealed or accessed Test.")

    training_ids = [str(value) for value in proof.get("training_record_ids") or []]
    heldout_ids = [str(value) for value in proof.get("heldout_record_ids") or []]
    if len(training_ids) != 6300 or len(heldout_ids) != 700:
        raise ValueError("Fold proof does not contain the expected 6300/700 split.")
    if len(set(training_ids)) != 6300 or len(set(heldout_ids)) != 700:
        raise ValueError("Fold proof contains duplicate record ids.")
    if set(training_ids) & set(heldout_ids):
        raise ValueError("Training and heldout record ids overlap.")
    train_digest = stable_id_digest(training_ids)
    heldout_digest = stable_id_digest(heldout_ids)
    if pipeline.get("train_record_ids_sha256") != train_digest:
        raise ValueError("Pipeline training-id digest differs from the fold proof.")
    if pipeline.get("heldout_record_ids_sha256") != heldout_digest:
        raise ValueError("Pipeline heldout-id digest differs from the fold proof.")

    stages = dict(pipeline.get("stages") or {})
    checkpoint_expectations = []
    config_expectations = []
    config_seeds: dict[str, int | None] = {}
    for stage_name in SUPERVISED_STAGES:
        stage = dict(stages.get(stage_name) or {})
        if stage.get("status") != "complete":
            raise ValueError(f"Fold {fold_id} stage {stage_name} is incomplete.")
        if stage.get("heldout_excluded") is not True:
            raise ValueError(f"Fold {fold_id} stage {stage_name} lacks exclusion.")
        if stage.get("train_record_ids_sha256") != train_digest:
            raise ValueError(f"Fold {fold_id} stage {stage_name} used another split.")
        if stage.get("test_accessed") is not False:
            raise PermissionError(f"Fold {fold_id} stage {stage_name} accessed Test.")
        checkpoint = dict(stage.get("checkpoint") or {})
        config = dict(stage.get("config") or {})
        checkpoint_expectations.append(
            {
                "fold_id": fold_id,
                "stage": stage_name,
                "artifact_kind": "checkpoint",
                "original_path": str(checkpoint.get("path", "")),
                "basename": Path(str(checkpoint.get("path", ""))).name,
                "expected_sha256": _require_sha256(
                    checkpoint.get("sha256"),
                    label=f"fold {fold_id} {stage_name} checkpoint",
                ),
            }
        )
        config_path = directory / "configs" / Path(str(config.get("path", ""))).name
        _validate_descriptor(
            config_path,
            config,
            label=f"fold {fold_id} {stage_name} config",
        )
        seed_match = re.search(
            r"(?m)^\s*seed:\s*(\d+)\s*$",
            config_path.read_text(encoding="utf-8"),
        )
        config_seeds[stage_name] = int(seed_match.group(1)) if seed_match else None
        config_expectations.append(
            {
                "fold_id": fold_id,
                "stage": stage_name,
                "artifact_kind": "config",
                "path": config_path.as_posix(),
                "sha256": sha256_file(config_path),
                "seed": config_seeds[stage_name],
            }
        )

    archive_checkpoints = {
        str(item["stage"]): str(item["sha256"])
        for item in list(archive.get("checkpoint_artifacts") or [])
    }
    expected_checkpoints = {
        item["stage"]: item["expected_sha256"] for item in checkpoint_expectations
    }
    if archive_checkpoints != expected_checkpoints:
        raise ValueError("Archive and pipeline checkpoint hashes differ.")

    source_expectations = [
        {
            "fold_id": fold_id,
            "artifact_kind": "fold_summary",
            "original_path": str(proof.get("fold_summary", "")),
            "basename": Path(str(proof.get("fold_summary", ""))).name,
            "expected_sha256": _require_sha256(
                proof.get("fold_summary_sha256"),
                label=f"fold {fold_id} summary",
            ),
        },
        {
            "fold_id": fold_id,
            "artifact_kind": "train_source",
            "original_path": str(proof.get("train_file", "")),
            "basename": Path(str(proof.get("train_file", ""))).name,
            "expected_sha256": _require_sha256(
                proof.get("train_file_sha256"),
                label=f"fold {fold_id} train source",
            ),
        },
        {
            "fold_id": fold_id,
            "artifact_kind": "heldout_source",
            "original_path": str(proof.get("heldout_file", "")),
            "basename": Path(str(proof.get("heldout_file", ""))).name,
            "expected_sha256": _require_sha256(
                proof.get("heldout_file_sha256"),
                label=f"fold {fold_id} heldout source",
            ),
        },
    ]
    proof_artifacts = dict(proof.get("artifact_sha256") or {})
    return {
        "fold_id": fold_id,
        "status": "PROVENANCE_VALID",
        "records": {"training": 6300, "heldout": 700},
        "train_record_ids_sha256": train_digest,
        "heldout_record_ids_sha256": heldout_digest,
        "heldout_excluded": True,
        "source_tree_sha256": _require_sha256(
            pipeline.get("source_tree_sha256"),
            label=f"fold {fold_id} source tree",
        ),
        "checkpoint_expectations": checkpoint_expectations,
        "config_artifacts": config_expectations,
        "config_seeds": config_seeds,
        "fold_source_expectations": source_expectations,
        "recorded_input_fingerprints": {
            "formal_cache_sha256": _require_sha256(
                proof_artifacts.get("formal_cache"),
                label=f"fold {fold_id} formal cache",
            ),
            "expanded_cache_sha256": _require_sha256(
                proof_artifacts.get("expanded_cache"),
                label=f"fold {fold_id} expanded cache",
            ),
            "siglip2_manifest_sha256": _require_sha256(
                proof_artifacts.get("siglip2_manifest"),
                label=f"fold {fold_id} SigLIP manifest",
            ),
            "vinvl_feature_tree_sha256": None,
            "text_tokenizer_tree_sha256": None,
            "grounding_prior_bundle_sha256": None,
        },
        "candidate_command_sha256": _require_sha256(
            dict(stages.get("candidate_caches") or {}).get("command_sha256"),
            label=f"fold {fold_id} candidate command",
        ),
        "checkpoint_payloads_loaded": 0,
        "candidate_payloads_loaded": 0,
        "training_records_parsed": 0,
        "dev_accessed": False,
        "test_accessed": False,
    }


def _locked_path(path: Path) -> bool:
    return any(component.lower() in _LOCKED_DIRECTORY_NAMES for component in path.parts)


def discover_named_artifacts(
    roots: Iterable[str | Path],
    *,
    basenames: Iterable[str],
) -> dict:
    """Search only preregistered basenames while pruning locked scopes."""

    names = {str(value).lower() for value in basenames if str(value)}
    paths: set[Path] = set()
    skipped: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).resolve()
        if _locked_path(root):
            raise PermissionError(f"Cannot search a locked R0-A scope: {root}")
        if not root.exists():
            continue
        for current, directory_names, file_names in os.walk(root, topdown=True):
            current_path = Path(current)
            allowed = []
            for name in directory_names:
                candidate = (current_path / name).resolve()
                if _locked_path(candidate):
                    skipped.add(candidate)
                else:
                    allowed.append(name)
            directory_names[:] = allowed
            for name in file_names:
                if name.lower() in names:
                    paths.add((current_path / name).resolve())
    return {
        "paths": sorted(paths, key=lambda value: value.as_posix()),
        "locked_directories_skipped": sorted(
            skipped, key=lambda value: value.as_posix()
        ),
    }


def hash_named_artifacts(paths: Iterable[str | Path]) -> list[dict]:
    return [
        {
            "path": Path(value).resolve().as_posix(),
            "bytes": Path(value).resolve().stat().st_size,
            "sha256": sha256_file(Path(value).resolve()),
        }
        for value in paths
    ]


def match_expectations_by_sha256(
    expectations: list[dict],
    available: list[dict],
) -> list[dict]:
    by_hash: dict[str, list[dict]] = {}
    for item in available:
        digest = _require_sha256(item.get("sha256"), label="available artifact")
        by_hash.setdefault(digest, []).append(item)
    results = []
    for expectation in expectations:
        expected = expectation["expected_sha256"]
        matches = sorted(
            by_hash.get(expected, []),
            key=lambda item: str(item.get("path", "")),
        )
        results.append(
            {
                **copy.deepcopy(expectation),
                "status": "EXACT_SHA256_AVAILABLE" if matches else "MISSING",
                "match_count": len(matches),
                "matched_paths": [str(item["path"]) for item in matches],
            }
        )
    return results


def _git_source_entries(
    repository: Path,
    commit: str,
) -> list[tuple[str, str]]:
    raw = subprocess.check_output(
        [
            "git",
            "ls-tree",
            "-r",
            "-z",
            commit,
            "--",
            "gmner",
            "scripts",
            "configs",
        ],
        cwd=repository,
    )
    entries: list[tuple[str, str]] = []
    for item in raw.split(b"\0"):
        if not item:
            continue
        metadata, path_bytes = item.split(b"\t", 1)
        _, kind, object_id = metadata.split()
        path = path_bytes.decode("utf-8")
        allowed = (
            (
                path.startswith(("gmner/", "scripts/"))
                and path.endswith(".py")
            )
            or (
                path.startswith("configs/")
                and path.endswith((".yaml", ".yml"))
            )
        )
        if allowed and kind == b"blob":
            entries.append((path, object_id.decode("ascii")))
    return sorted(entries)


def _git_blob_sha256s(
    repository: Path,
    object_ids: Iterable[str],
) -> dict[str, bytes]:
    """Read unique Git blobs through one batch process and hash their contents."""

    unique = sorted(set(object_ids))
    if not unique:
        return {}
    request = "".join(f"{object_id}\n" for object_id in unique).encode("ascii")
    output = subprocess.check_output(
        ["git", "cat-file", "--batch"],
        cwd=repository,
        input=request,
    )
    cursor = 0
    digests: dict[str, bytes] = {}
    for requested_id in unique:
        line_end = output.index(b"\n", cursor)
        header = output[cursor:line_end].decode("ascii")
        cursor = line_end + 1
        fields = header.split()
        if len(fields) != 3 or fields[1] != "blob":
            raise ValueError(f"Git object is not a blob: {header}")
        object_id, _, size_text = fields
        size = int(size_text)
        content = output[cursor : cursor + size]
        cursor += size
        if output[cursor : cursor + 1] != b"\n":
            raise ValueError("Malformed git cat-file --batch output.")
        cursor += 1
        if object_id != requested_id:
            raise ValueError("Git batch output order differs from its request.")
        digests[object_id] = hashlib.sha256(content).digest()
    if cursor != len(output):
        raise ValueError("Unexpected trailing data in git cat-file output.")
    return digests


def _source_tree_digest(
    entries: Iterable[tuple[str, str]],
    blob_sha256s: dict[str, bytes],
) -> str:
    digest = hashlib.sha256()
    for path, object_id in sorted(entries):
        relative = path.encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(blob_sha256s[object_id])
    return digest.hexdigest()


def git_source_tree_sha256(repo: str | Path, commit: str) -> str:
    """Compute the archived source-tree contract directly from Git blobs."""

    repository = Path(repo).resolve()
    entries = _git_source_entries(repository, commit)
    blob_sha256s = _git_blob_sha256s(
        repository,
        (object_id for _, object_id in entries),
    )
    return _source_tree_digest(entries, blob_sha256s)


def find_source_tree_commits(
    repo: str | Path,
    *,
    expected_sha256: str,
) -> dict:
    expected = _require_sha256(expected_sha256, label="archived source tree")
    repository = Path(repo).resolve()
    commits = subprocess.check_output(
        ["git", "rev-list", "--all"],
        cwd=repository,
        text=True,
    ).splitlines()
    entries_by_commit = {
        commit: _git_source_entries(repository, commit) for commit in commits
    }
    blob_sha256s = _git_blob_sha256s(
        repository,
        (
            object_id
            for entries in entries_by_commit.values()
            for _, object_id in entries
        ),
    )
    matches = []
    for commit, entries in entries_by_commit.items():
        if _source_tree_digest(entries, blob_sha256s) == expected:
            matches.append(commit)
    return {
        "expected_sha256": expected,
        "commits_checked": len(commits),
        "unique_blobs_checked": len(blob_sha256s),
        "matching_commits": matches,
        "exact_source_tree_available": bool(matches),
    }


def validate_external_inventory(payload: dict) -> None:
    if payload.get("kind") != P4_R0_EXTERNAL_INVENTORY_KIND:
        raise ValueError("Not a P4-R0-A external storage inventory.")
    if int(payload.get("format_version", -1)) != P4_R0_FORMAT_VERSION:
        raise ValueError("Unsupported P4-R0-A external inventory version.")
    access = dict(payload.get("access_contract") or {})
    if int(access.get("payloads_deserialized", -1)) != 0:
        raise PermissionError("External inventory deserialized a payload.")
    if int(access.get("training_records_parsed", -1)) != 0:
        raise PermissionError("External inventory parsed training records.")
    for field in (
        "calibration_folds_opened",
        "dev_accessed",
        "test_accessed",
        "oracle_labels_computed",
        "model_executed",
    ):
        if access.get(field) is not False:
            raise PermissionError(f"External inventory changed locked field: {field}.")
    if sorted(int(value) for value in payload.get("folds_checked") or []) != list(
        P4_DEVELOPMENT_FOLDS
    ):
        raise ValueError("External inventory must cover exactly folds 0-7.")


def external_available_artifacts(payload: dict) -> list[dict]:
    validate_external_inventory(payload)
    artifacts = []
    for location in list(payload.get("locations") or []):
        for item in list(dict(location).get("available_artifacts") or []):
            descriptor = dict(item)
            artifacts.append(
                {
                    "path": str(descriptor.get("path", "")),
                    "bytes": int(descriptor.get("bytes", 0)),
                    "sha256": _require_sha256(
                        descriptor.get("sha256"),
                        label="external available artifact",
                    ),
                    "location": str(dict(location).get("name", "")),
                }
            )
    return artifacts


def build_r0_a_report(
    *,
    preregistration: dict,
    fold_reports: list[dict],
    checkpoint_matches: list[dict],
    source_matches: list[dict],
    source_tree: dict,
    external_inventory: dict,
    implementation: dict,
) -> dict:
    folds = sorted(int(item["fold_id"]) for item in fold_reports)
    if folds != list(P4_DEVELOPMENT_FOLDS):
        raise ValueError("R0-A report requires exactly folds 0-7.")
    checkpoints_complete = all(
        item["status"] == "EXACT_SHA256_AVAILABLE" for item in checkpoint_matches
    )
    sources_complete = all(
        item["status"] == "EXACT_SHA256_AVAILABLE" for item in source_matches
    )
    configs_complete = all(
        len(item["config_artifacts"]) == len(SUPERVISED_STAGES)
        and all(value is not None for value in item["config_seeds"].values())
        for item in fold_reports
    )
    fingerprint_fields = (
        "vinvl_feature_tree_sha256",
        "text_tokenizer_tree_sha256",
        "grounding_prior_bundle_sha256",
    )
    fingerprints_complete = all(
        all(
            item["recorded_input_fingerprints"].get(field)
            for field in fingerprint_fields
        )
        for item in fold_reports
    )
    gate = {
        "fold_provenance_complete": all(
            item["status"] == "PROVENANCE_VALID" for item in fold_reports
        ),
        "all_supervised_checkpoints_exact": checkpoints_complete,
        "all_stage_configs_exact_and_seeded": configs_complete,
        "all_fold_sources_exact": sources_complete,
        "source_tree_exactly_recoverable": bool(
            source_tree["exact_source_tree_available"]
        ),
        "required_input_fingerprints_recorded": fingerprints_complete,
        "checkpoint_payloads_loaded_zero": True,
        "candidate_payloads_loaded_zero": True,
        "training_records_parsed_zero": True,
        "calibration_folds_opened_false": True,
        "dev_accessed_false": True,
        "test_accessed_false": True,
        "oracle_labels_computed_false": True,
    }
    passed = all(gate.values())
    blockers = sorted(name for name, value in gate.items() if not value)
    report = {
        "kind": P4_R0_A_REPORT_KIND,
        "format_version": P4_R0_FORMAT_VERSION,
        "phase": "P4-R0-A_checkpoint_replay_feasibility",
        "status": P4_R0_A_FEASIBLE if passed else P4_R0_A_BLOCKED,
        "implementation": copy.deepcopy(implementation),
        "authorization": copy.deepcopy(preregistration["authorization"]),
        "access_contract": {
            "folds_read": folds,
            "checkpoint_payloads_loaded": 0,
            "candidate_payloads_loaded": 0,
            "training_records_parsed": 0,
            "calibration_folds_opened": False,
            "dev_accessed": False,
            "test_accessed": False,
            "oracle_labels_computed": False,
            "model_executed": False,
        },
        "fold_provenance": copy.deepcopy(fold_reports),
        "checkpoint_inventory": {
            "required": len(checkpoint_matches),
            "exact_available": sum(
                item["status"] == "EXACT_SHA256_AVAILABLE"
                for item in checkpoint_matches
            ),
            "matches": copy.deepcopy(checkpoint_matches),
        },
        "fold_source_inventory": {
            "required": len(source_matches),
            "exact_available": sum(
                item["status"] == "EXACT_SHA256_AVAILABLE"
                for item in source_matches
            ),
            "matches": copy.deepcopy(source_matches),
        },
        "source_tree": copy.deepcopy(source_tree),
        "input_fingerprint_audit": {
            "required_fields": list(fingerprint_fields),
            "all_recorded": fingerprints_complete,
            "candidate_cache_output_hashes_recorded": True,
            "siglip_manifest_hashes_recorded": True,
        },
        "external_inventory": copy.deepcopy(external_inventory),
        "gate": gate,
        "gate_blockers": blockers,
        "gate_passed": passed,
        "checkpoint_replay_execution_authorized": False,
        "r0_b_authorized": False,
        "next_state": (
            "REQUEST_SEPARATE_HELDOUT_R16_REPLAY_APPROVAL"
            if passed
            else "STOP_WITHOUT_AUTHORIZING_R0_B"
        ),
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def validate_r0_a_report(report: dict) -> None:
    if report.get("kind") != P4_R0_A_REPORT_KIND:
        raise ValueError("Not a P4-R0-A report.")
    expected = _require_sha256(report.get("report_sha256"), label="R0-A report")
    unsigned = copy.deepcopy(report)
    unsigned.pop("report_sha256", None)
    if canonical_json_sha256(unsigned) != expected:
        raise ValueError("P4-R0-A report digest is inconsistent.")
    access = dict(report.get("access_contract") or {})
    zero_fields = (
        "checkpoint_payloads_loaded",
        "candidate_payloads_loaded",
        "training_records_parsed",
    )
    if any(int(access.get(field, -1)) != 0 for field in zero_fields):
        raise ValueError("R0-A loaded a forbidden payload.")
    if any(
        access.get(field) is not False
        for field in (
            "calibration_folds_opened",
            "dev_accessed",
            "test_accessed",
            "oracle_labels_computed",
            "model_executed",
        )
    ):
        raise PermissionError("R0-A accessed a locked scope.")
    if report.get("checkpoint_replay_execution_authorized") is not False:
        raise PermissionError("R0-A report cannot authorize replay execution.")
    if report.get("r0_b_authorized") is not False:
        raise PermissionError("R0-A report cannot authorize R0-B.")
    gate = dict(report.get("gate") or {})
    expected_blockers = sorted(name for name, value in gate.items() if not value)
    if report.get("gate_blockers") != expected_blockers:
        raise ValueError("P4-R0-A gate blocker list is inconsistent.")
    if bool(report.get("gate_passed")) != all(gate.values()):
        raise ValueError("P4-R0-A gate status is inconsistent.")
