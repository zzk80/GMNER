"""File-system and serialization utilities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from gmner.constants import DEFAULT_LABEL2ID, normalize_bio_label


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def maybe_convert_conll(input_path: str | Path, output_dir: str | Path) -> Path:
    input_path = Path(input_path)
    if input_path.suffix.lower() != ".txt":
        return input_path

    output_path = Path(output_dir) / "cache" / f"{input_path.stem}.jsonl"
    signature_path = output_path.with_suffix(".source.sha1")
    source_signature = hashlib.sha1(input_path.read_bytes() + b"\nparser:v2").hexdigest()
    if output_path.exists() and signature_path.exists():
        if signature_path.read_text(encoding="ascii").strip() == source_signature:
            return output_path

    records = []
    tokens: list[str] = []
    tags: list[str] = []
    fine_tags: list[str] = []
    image_id: str | None = None

    def flush_record() -> None:
        if image_id and tokens:
            record = {
                "id": len(records),
                "tokens": tokens.copy(),
                "ner_tags": [
                    DEFAULT_LABEL2ID.get(normalize_bio_label(tag), DEFAULT_LABEL2ID["O"])
                    for tag in tags
                ],
                "image": f"{image_id}.jpg",
            }
            if fine_tags and len(fine_tags) == len(tokens):
                record["fine_ner_tags"] = fine_tags.copy()
            records.append(record)

    with input_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                flush_record()
                tokens.clear()
                tags.clear()
                fine_tags.clear()
                image_id = None
            elif line.startswith("IMGID:"):
                if image_id and tokens:
                    flush_record()
                    tokens.clear()
                    tags.clear()
                    fine_tags.clear()
                image_id = line.split("IMGID:", 1)[1].strip()
            else:
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    if len(parts) >= 3:
                        tags.append(parts[1])
                        fine_tags.append(parts[2])
                    else:
                        tags.append(parts[-1])

    flush_record()
    write_jsonl(output_path, records)
    signature_path.write_text(source_signature, encoding="ascii")
    return output_path
