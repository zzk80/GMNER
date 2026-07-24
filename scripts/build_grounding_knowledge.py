"""Build train-only type and mention groundability priors for GMNER."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.utils.io import ensure_dir, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GMNER groundability priors")
    parser.add_argument("--occurrences", type=str, default="knowledge/offline/entity_occurrences.jsonl")
    parser.add_argument("--output-dir", type=str, default="knowledge/grounding")
    return parser.parse_args()


def rate(numerator: int, denominator: int) -> float:
    return numerator / max(denominator, 1)


def build_groundability_stats(
    occurrences: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_mention_type: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for item in occurrences:
        entity_type = str(item["entity_type"])
        by_type[entity_type].append(item)
        by_mention_type[(str(item["normalized_mention"]), entity_type)].append(item)

    type_rows = []
    for entity_type, items in sorted(by_type.items()):
        groundable = sum(1 for item in items if item.get("groundable"))
        matched = sum(1 for item in items if item.get("best_region_index") is not None)
        type_rows.append(
            {
                "id": f"groundability:type:{entity_type.lower()}",
                "level": "groundability_type",
                "entity_type": entity_type,
                "count": len(items),
                "groundable_count": groundable,
                "matched_region_count": matched,
                "groundability_rate": rate(groundable, len(items)),
                "region_match_rate": rate(matched, len(items)),
                "null_prior": 1.0 - rate(groundable, len(items)),
            }
        )

    mention_rows = []
    for (mention, entity_type), items in sorted(
        by_mention_type.items(),
        key=lambda pair: (-len(pair[1]), pair[0][1], pair[0][0]),
    ):
        groundable = sum(1 for item in items if item.get("groundable"))
        matched = sum(1 for item in items if item.get("best_region_index") is not None)
        mention_rows.append(
            {
                "id": f"groundability:mention:{entity_type.lower()}:{mention}",
                "level": "groundability_mention_type",
                "mention": mention,
                "entity_type": entity_type,
                "count": len(items),
                "groundable_count": groundable,
                "matched_region_count": matched,
                "groundability_rate": rate(groundable, len(items)),
                "region_match_rate": rate(matched, len(items)),
                "null_prior": 1.0 - rate(groundable, len(items)),
            }
        )

    return type_rows, mention_rows


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    occurrences = read_jsonl(args.occurrences)
    if not occurrences:
        raise ValueError(f"No occurrences found: {args.occurrences}")

    type_rows, mention_rows = build_groundability_stats(occurrences)
    write_jsonl(output_dir / "groundability_by_type.jsonl", type_rows)
    write_jsonl(output_dir / "groundability_by_mention_type.jsonl", mention_rows)

    summary = {
        "occurrences": len(occurrences),
        "groundable_entities": sum(1 for item in occurrences if item.get("groundable")),
        "matched_region_entities": sum(
            1 for item in occurrences if item.get("best_region_index") is not None
        ),
        "type_groundability_rows": len(type_rows),
        "mention_groundability_rows": len(mention_rows),
    }
    with (output_dir / "grounding_knowledge_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"occurrences={summary['occurrences']}")
    print(f"groundable_entities={summary['groundable_entities']}")
    print(f"matched_region_entities={summary['matched_region_entities']}")
    print(f"saved_to={output_dir}")


if __name__ == "__main__":
    main()

