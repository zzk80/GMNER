"""Stable artifact fingerprints shared by training and OOF pipelines."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id_digest(record_ids: Iterable[str]) -> str:
    value = json.dumps(
        sorted(str(record_id) for record_id in record_ids),
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(value).hexdigest()
