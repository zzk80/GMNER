"""Run the read-only S3 P0-A boundary/type audit on an existing cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.stage1_candidate_selector import (
    Stage1CandidateSelectorDataset,
)
from gmner.diagnostics import audit_boundary_type_errors


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", required=True)
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Stage1CandidateSelectorDataset(
        args.candidate_cache,
        split=args.split,
    )
    report = audit_boundary_type_errors(
        dataset.records,
        split=args.split,
    )
    cache_path = Path(args.candidate_cache).resolve()
    report["source_candidate_cache"] = str(cache_path)
    report["source_candidate_cache_sha256"] = _sha256(cache_path)
    report["candidate_contract"] = {
        key: dataset.metadata.get(key)
        for key in (
            "kind",
            "format_version",
            "scope",
            "candidate_config_sha256",
            "stage1_checkpoint_sha256",
            "stage1_config_sha256",
        )
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
