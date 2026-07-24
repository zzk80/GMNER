"""Convert GMNER CoNLL Twitter10000 data to jsonl for MMNERJsonDataset."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from gmner.constants import DEFAULT_LABEL2ID, normalize_bio_label
from gmner.utils.io import write_jsonl


LABEL2ID: Dict[str, int] = DEFAULT_LABEL2ID


def parse_conll(path: Path, image_ext: str) -> List[dict]:
    records: List[dict] = []
    tokens: List[str] = []
    tags: List[str] = []
    fine_tags: List[str] = []
    image_id: str | None = None
    unknown_tags: Dict[str, int] = {}

    def flush_record() -> None:
        if not image_id or not tokens:
            return
        mapped = []
        for tag in tags:
            if tag not in LABEL2ID:
                unknown_tags[tag] = unknown_tags.get(tag, 0) + 1
                mapped.append(LABEL2ID["O"])
            else:
                mapped.append(LABEL2ID[tag])
        record = {
            "id": len(records),
            "tokens": tokens.copy(),
            "ner_tags": mapped,
            "image": f"{image_id}{image_ext}",
        }
        if fine_tags and len(fine_tags) == len(tokens):
            record["fine_ner_tags"] = fine_tags.copy()
        records.append(record)

    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.rstrip("\n")
            if not line:
                flush_record()
                tokens.clear()
                tags.clear()
                fine_tags.clear()
                image_id = None
                continue

            if line.startswith("IMGID:"):
                if image_id and tokens:
                    flush_record()
                    tokens.clear()
                    tags.clear()
                    fine_tags.clear()
                image_id = line.split("IMGID:", 1)[1].strip()
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            tokens.append(parts[0])
            if len(parts) >= 3:
                tags.append(normalize_bio_label(parts[1]))
                fine_tags.append(parts[2])
            else:
                tags.append(normalize_bio_label(parts[-1]))

    flush_record()

    if unknown_tags:
        summary = ", ".join([f"{key}:{value}" for key, value in sorted(unknown_tags.items())])
        print(f"[WARN] Unknown tags mapped to O: {summary}")

    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GMNER CoNLL to jsonl")
    parser.add_argument("--input", type=str, required=True, help="Path to train/dev/test txt file")
    parser.add_argument("--output", type=str, required=True, help="Output jsonl path")
    parser.add_argument("--image-ext", type=str, default=".jpg", help="Image file extension")
    args = parser.parse_args()

    records = parse_conll(Path(args.input), image_ext=args.image_ext)
    write_jsonl(Path(args.output), records)
    print(f"Wrote {len(records)} records to {args.output}")


if __name__ == "__main__":
    main()
