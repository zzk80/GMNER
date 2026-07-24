"""Offline record-level candidate data for the structured verifier."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


CACHE_FORMAT_VERSION = 2
SUPPORTED_CACHE_FORMAT_VERSIONS = frozenset({1, CACHE_FORMAT_VERSION})


class RecordCandidateDataset(Dataset):
    """Load a Stage-1 candidate cache without rerunning the frozen backbone."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_stage1_sha256: str | None = None,
        expected_candidate_sha256: str | None = None,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Record candidate cache not found: {self.path}")
        payload = torch.load(self.path, map_location="cpu")
        if not isinstance(payload, dict) or "records" not in payload:
            raise ValueError(f"Invalid record candidate cache: {self.path}")
        self.metadata = dict(payload.get("metadata") or {})
        version = int(self.metadata.get("format_version", -1))
        if version not in SUPPORTED_CACHE_FORMAT_VERSIONS:
            raise ValueError(
                f"Unsupported candidate cache version {version}; "
                f"supported versions are {sorted(SUPPORTED_CACHE_FORMAT_VERSIONS)}."
            )
        self.records = list(payload["records"])
        self._validate_fingerprint("stage1_checkpoint_sha256", expected_stage1_sha256)
        self._validate_fingerprint("candidate_config_sha256", expected_candidate_sha256)

    def _validate_fingerprint(self, key: str, expected: str | None) -> None:
        if not expected:
            return
        actual = str(self.metadata.get(key, ""))
        if actual != str(expected):
            raise ValueError(
                f"Candidate cache fingerprint mismatch for {key}: "
                f"expected {expected}, found {actual or '<missing>'}."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]
