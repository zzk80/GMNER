"""Create a leakage-safe external-knowledge schema from FMNERG labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.constants import DEFAULT_LABEL2ID, normalize_entity_type, strip_bio_prefix
from gmner.knowledge.prototype_descriptions import (
    SUBTYPE_DESCRIPTIONS,
    TYPE_DESCRIPTIONS,
)
from gmner.models.external_knowledge import normalize_subtype_name
from gmner.utils.io import read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize the external knowledge JSONL schema from train labels. "
            "The output is a taxonomy seed, not a complete external knowledge base."
        )
    )
    parser.add_argument("--input", required=True, help="Fine-grained train .txt or .jsonl")
    parser.add_argument(
        "--output",
        default="knowledge/external/subtype_knowledge.seed.jsonl",
    )
    return parser.parse_args()


def _pair_from_labels(coarse_label: object, fine_label: object) -> tuple[str, str] | None:
    coarse_text = str(coarse_label or "O").strip()
    fine_text = str(fine_label or "O").strip()
    if fine_text == "O":
        return None
    coarse_type = normalize_entity_type(strip_bio_prefix(coarse_text))
    fine_type = normalize_subtype_name(strip_bio_prefix(fine_text))
    if coarse_type == "O" or not fine_type or fine_type == "o":
        return None
    return coarse_type, fine_type


def discover_subtypes(path: str | Path) -> list[tuple[str, str]]:
    """Read only label names; entity mentions and contexts are never copied."""

    source = Path(path)
    pairs: set[tuple[str, str]] = set()
    if source.suffix.lower() == ".txt":
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text or text.startswith("IMGID:"):
                    continue
                parts = text.split()
                if len(parts) < 3:
                    continue
                pair = _pair_from_labels(parts[1], parts[2])
                if pair is not None:
                    pairs.add(pair)
    else:
        id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
        for record in read_jsonl(source):
            coarse_tags = record.get("ner_tags", [])
            fine_tags = record.get("fine_ner_tags", [])
            for index, fine_tag in enumerate(fine_tags):
                if index >= len(coarse_tags):
                    continue
                coarse_tag = coarse_tags[index]
                if isinstance(coarse_tag, int):
                    coarse_tag = id2label.get(coarse_tag, "O")
                pair = _pair_from_labels(coarse_tag, fine_tag)
                if pair is not None:
                    pairs.add(pair)
    if not pairs:
        raise ValueError(f"No fine-grained subtype labels found in {source}.")

    subtype_to_type: dict[str, str] = {}
    for coarse_type, fine_type in pairs:
        previous = subtype_to_type.setdefault(fine_type, coarse_type)
        if previous != coarse_type:
            raise ValueError(
                f"Subtype {fine_type!r} maps to both {previous} and {coarse_type}."
            )
    return sorted(pairs)


def seed_description(coarse_type: str, fine_type: str) -> str:
    curated = SUBTYPE_DESCRIPTIONS.get(coarse_type, {}).get(fine_type)
    if curated:
        return f"{TYPE_DESCRIPTIONS[coarse_type]} {curated}"
    readable = fine_type.replace("_", " ")
    return (
        f"{TYPE_DESCRIPTIONS[coarse_type]} "
        f"The fine-grained subtype is {readable}."
    )


def build_seed_records(pairs: list[tuple[str, str]]) -> list[dict]:
    return [
        {
            "id": f"taxonomy:{coarse_type.lower()}:{fine_type}",
            "coarse_type": coarse_type,
            "fine_type": fine_type,
            "text": seed_description(coarse_type, fine_type),
            "source": "fmnerg_taxonomy_seed",
            "confidence": 0.5,
            "is_seed": True,
        }
        for coarse_type, fine_type in pairs
    ]


def main() -> None:
    args = parse_args()
    pairs = discover_subtypes(args.input)
    records = build_seed_records(pairs)
    output = Path(args.output)
    write_jsonl(output, records)
    print(
        json.dumps(
            {
                "subtypes": len(records),
                "output": str(output.resolve()),
                "warning": (
                    "This is a taxonomy seed. Add independent Wikidata, "
                    "Wikipedia, or curated records before the formal run."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
