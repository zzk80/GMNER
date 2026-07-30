from __future__ import annotations

import json

from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.full_chain_oof_contract import (
    FULL_CHAIN_FOLD_MANIFEST_KIND,
    FULL_CHAIN_FOLD_MANIFEST_VERSION,
    LEGACY_FULL_CHAIN_FOLD_MANIFEST_KIND,
    fold_from_manifest,
    validate_fold_manifest,
)


def _write_records(path, record_ids: list[str]) -> None:
    path.write_text(
        "".join(
            json.dumps({"id": record_id}) + "\n"
            for record_id in record_ids
        ),
        encoding="utf-8",
    )


def _manifest(tmp_path, kind: str) -> dict:
    train_ids = ["1"]
    heldout_ids = ["2"]
    train_path = tmp_path / "train.jsonl"
    heldout_path = tmp_path / "heldout.jsonl"
    _write_records(train_path, train_ids)
    _write_records(heldout_path, heldout_ids)
    return {
        "kind": kind,
        "format_version": FULL_CHAIN_FOLD_MANIFEST_VERSION,
        "num_folds": 2,
        "source_split": "train",
        "test_accessed": False,
        "records": 2,
        "record_ids": ["1", "2"],
        "record_ids_sha256": stable_id_digest(["1", "2"]),
        "folds": [
            {
                "fold": 0,
                "train_record_ids": train_ids,
                "heldout_record_ids": heldout_ids,
                "train_record_ids_sha256": stable_id_digest(train_ids),
                "heldout_record_ids_sha256": stable_id_digest(heldout_ids),
                "train_file": str(train_path),
                "heldout_file": str(heldout_path),
                "train_file_sha256": sha256_file(train_path),
                "heldout_file_sha256": sha256_file(heldout_path),
            },
            {
                "fold": 1,
                "train_record_ids": heldout_ids,
                "heldout_record_ids": train_ids,
                "train_record_ids_sha256": stable_id_digest(heldout_ids),
                "heldout_record_ids_sha256": stable_id_digest(train_ids),
                "train_file": str(heldout_path),
                "heldout_file": str(train_path),
                "train_file_sha256": sha256_file(heldout_path),
                "heldout_file_sha256": sha256_file(train_path),
            },
        ],
    }


def test_artifact_fingerprints_are_order_stable(tmp_path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"formal-model-g")

    assert len(sha256_file(artifact)) == 64
    assert stable_id_digest(["2", "1"]) == stable_id_digest(["1", "2"])


def test_oof_contract_accepts_current_and_archived_manifest_kinds(
    tmp_path,
) -> None:
    for kind in (
        FULL_CHAIN_FOLD_MANIFEST_KIND,
        LEGACY_FULL_CHAIN_FOLD_MANIFEST_KIND,
    ):
        payload = _manifest(tmp_path, kind)
        path = tmp_path / f"{kind}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")

        validated = validate_fold_manifest(path, expected_num_folds=2)
        assert fold_from_manifest(validated, 1)["heldout_record_ids"] == ["1"]
