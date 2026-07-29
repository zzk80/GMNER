"""Auditable contracts for the full-chain NULL Release OOF pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from gmner.utils.io import read_jsonl


FULL_CHAIN_FOLD_MANIFEST_KIND = "null_release_full_chain_fold_manifest"
FULL_CHAIN_FOLD_MANIFEST_VERSION = 1
FULL_CHAIN_PIPELINE_KIND = "null_release_full_chain_fold_pipeline"
FULL_CHAIN_PIPELINE_VERSION = 1
SUPERVISED_PIPELINE_STAGES = (
    "stage1",
    "hierarchical",
    "coarse",
    "fine",
    "evidence",
    "reliability",
)
REQUIRED_PIPELINE_STAGES = (
    "stage1",
    "candidate_caches",
    "hierarchical",
    "coarse",
    "fine",
    "evidence",
    "siglip2_caches",
    "reliability",
)


def record_id(record: dict) -> str:
    metadata = dict(record.get("metadata") or {})
    return str(record.get("id", metadata.get("record_id", "")))


def source_tree_sha256(root: str | Path) -> str:
    """Fingerprint source/config files without depending on Git availability."""

    root = Path(root).resolve()
    files: list[Path] = []
    for relative, patterns in (
        ("gmner", ("*.py",)),
        ("scripts", ("*.py",)),
        ("configs", ("*.yaml", "*.yml")),
    ):
        directory = root / relative
        for pattern in patterns:
            files.extend(directory.rglob(pattern))
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        relative = path.resolve().relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(output)


def _ids_from_file(path: Path) -> list[str]:
    values = [record_id(item) for item in read_jsonl(path)]
    if any(not value for value in values):
        raise ValueError(f"Fold file contains a record without an id: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"Fold file contains duplicate record ids: {path}")
    return values


def validate_fold_manifest(
    path: str | Path,
    *,
    expected_num_folds: int = 10,
    verify_files: bool = True,
    verify_fold_ids: Iterable[int] | None = None,
) -> dict:
    manifest_path = Path(path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("kind") != FULL_CHAIN_FOLD_MANIFEST_KIND:
        raise ValueError("Not a NULL Release full-chain fold manifest.")
    if int(manifest.get("format_version", -1)) != FULL_CHAIN_FOLD_MANIFEST_VERSION:
        raise ValueError("Unsupported full-chain fold manifest version.")
    if int(manifest.get("num_folds", -1)) != int(expected_num_folds):
        raise ValueError(
            f"Expected {expected_num_folds} folds, found {manifest.get('num_folds')}."
        )
    if manifest.get("source_split") != "train" or manifest.get("test_accessed") is not False:
        raise ValueError("OOF manifest must be train-only with test_accessed=false.")
    all_ids = [str(value) for value in manifest.get("record_ids") or []]
    if any(not value for value in all_ids) or len(all_ids) != len(set(all_ids)):
        raise ValueError("OOF manifest has missing or duplicate source record ids.")
    if int(manifest.get("records", -1)) != len(all_ids):
        raise ValueError("OOF manifest source record count is inconsistent.")
    if stable_id_digest(all_ids) != manifest.get("record_ids_sha256"):
        raise ValueError("OOF manifest source record-id digest is inconsistent.")

    folds = list(manifest.get("folds") or [])
    fold_ids = sorted(int(item.get("fold", -1)) for item in folds)
    if fold_ids != list(range(int(expected_num_folds))):
        raise ValueError(f"OOF manifest fold ids are invalid: {fold_ids}.")
    verified_folds = (
        set(fold_ids)
        if verify_fold_ids is None
        else {int(value) for value in verify_fold_ids}
    )
    unknown_verified_folds = sorted(verified_folds - set(fold_ids))
    if unknown_verified_folds:
        raise ValueError(
            f"Cannot verify unknown OOF folds: {unknown_verified_folds}."
        )
    source_set = set(all_ids)
    observed_heldout: set[str] = set()
    for fold in folds:
        train_ids = [str(value) for value in fold.get("train_record_ids") or []]
        heldout_ids = [str(value) for value in fold.get("heldout_record_ids") or []]
        if len(train_ids) != len(set(train_ids)) or len(heldout_ids) != len(
            set(heldout_ids)
        ):
            raise ValueError(f"Fold {fold.get('fold')} contains duplicate ids.")
        if set(train_ids) & set(heldout_ids):
            raise ValueError(f"Fold {fold.get('fold')} train/heldout ids overlap.")
        if set(train_ids) | set(heldout_ids) != source_set:
            raise ValueError(f"Fold {fold.get('fold')} is not a source complement.")
        if observed_heldout & set(heldout_ids):
            raise ValueError("Held-out ids occur in more than one fold.")
        observed_heldout.update(heldout_ids)
        if stable_id_digest(train_ids) != fold.get("train_record_ids_sha256"):
            raise ValueError(f"Fold {fold.get('fold')} train-id digest mismatch.")
        if stable_id_digest(heldout_ids) != fold.get("heldout_record_ids_sha256"):
            raise ValueError(f"Fold {fold.get('fold')} heldout-id digest mismatch.")
        if not verify_files or int(fold.get("fold", -1)) not in verified_folds:
            continue
        for role, expected_ids in (
            ("train", train_ids),
            ("heldout", heldout_ids),
        ):
            file_path = Path(fold[f"{role}_file"]).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Missing fold {role} file: {file_path}")
            if sha256_file(file_path) != fold.get(f"{role}_file_sha256"):
                raise ValueError(f"Fold {fold.get('fold')} {role} file hash mismatch.")
            if _ids_from_file(file_path) != expected_ids:
                raise ValueError(f"Fold {fold.get('fold')} {role} file order changed.")
    if observed_heldout != source_set:
        raise ValueError("Held-out folds do not cover the source split exactly once.")
    return manifest


def fold_from_manifest(manifest: dict, fold_id: int) -> dict:
    matches = [
        item for item in manifest.get("folds") or []
        if int(item.get("fold", -1)) == int(fold_id)
    ]
    if len(matches) != 1:
        raise ValueError(f"Fold manifest has no unique fold {fold_id}.")
    return matches[0]


def validate_pipeline_manifest(
    path: str | Path,
    *,
    fold_manifest: dict,
    fold_id: int,
    required_stages: Iterable[str] | None = None,
    supervised_stages: Iterable[str] | None = None,
) -> dict:
    pipeline_path = Path(path).resolve()
    pipeline = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if pipeline.get("kind") != FULL_CHAIN_PIPELINE_KIND:
        raise ValueError("Not a NULL Release full-chain pipeline manifest.")
    if int(pipeline.get("format_version", -1)) != FULL_CHAIN_PIPELINE_VERSION:
        raise ValueError("Unsupported full-chain pipeline manifest version.")
    if int(pipeline.get("fold_id", -1)) != int(fold_id):
        raise ValueError("Pipeline manifest fold id does not match the proof.")
    if pipeline.get("test_accessed") is not False:
        raise ValueError("Formal OOF pipeline must assert test_accessed=false.")
    if pipeline.get("source_tree_sha256") != fold_manifest.get(
        "source_tree_sha256"
    ):
        raise ValueError("Pipeline source-tree fingerprint differs from the fold manifest.")
    fold = fold_from_manifest(fold_manifest, fold_id)
    expected_train_digest = fold["train_record_ids_sha256"]
    expected_heldout_digest = fold["heldout_record_ids_sha256"]
    if pipeline.get("train_record_ids_sha256") != expected_train_digest:
        raise ValueError("Pipeline training-id digest does not match the fold.")
    if pipeline.get("heldout_record_ids_sha256") != expected_heldout_digest:
        raise ValueError("Pipeline heldout-id digest does not match the fold.")
    stages = dict(pipeline.get("stages") or {})
    required = tuple(required_stages or REQUIRED_PIPELINE_STAGES)
    supervised = tuple(supervised_stages or SUPERVISED_PIPELINE_STAGES)
    if not set(supervised).issubset(required):
        raise ValueError("Supervised pipeline stages must be required stages.")
    for name in required:
        stage = dict(stages.get(name) or {})
        if stage.get("status") != "complete":
            raise ValueError(f"Required OOF stage {name!r} is not complete.")
        if stage.get("test_accessed") is not False:
            raise ValueError(f"Required OOF stage {name!r} accessed test data.")
        for group in ("inputs", "outputs"):
            artifacts = list(stage.get(group) or [])
            if not artifacts:
                raise ValueError(f"Required OOF stage {name!r} has no {group} proof.")
            for artifact in artifacts:
                artifact_path = Path(str(artifact.get("path", "")))
                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"Missing {name} {group} artifact: {artifact_path}"
                    )
                if sha256_file(artifact_path) != artifact.get("sha256"):
                    raise ValueError(f"{name} {group} artifact hash changed.")
    for name in supervised:
        stage = dict(stages.get(name) or {})
        if stage.get("status") != "complete":
            raise ValueError(f"Supervised OOF stage {name!r} is not complete.")
        if stage.get("test_accessed") is not False:
            raise ValueError(f"Supervised OOF stage {name!r} accessed test data.")
        if stage.get("heldout_excluded") is not True:
            raise ValueError(f"Supervised OOF stage {name!r} lacks heldout exclusion.")
        if stage.get("train_record_ids_sha256") != expected_train_digest:
            raise ValueError(f"Supervised OOF stage {name!r} used another train split.")
        for role in ("config", "checkpoint"):
            artifact = dict(stage.get(role) or {})
            artifact_path = Path(str(artifact.get("path", "")))
            if not artifact_path.exists():
                raise FileNotFoundError(
                    f"Missing {name} {role} artifact: {artifact_path}"
                )
            if sha256_file(artifact_path) != artifact.get("sha256"):
                raise ValueError(f"{name} {role} artifact hash changed.")
    return pipeline
