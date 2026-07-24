"""Build a training-set entity inventory for offline GMNER knowledge construction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.knowledge import build_entity_inventory
from gmner.utils.io import ensure_dir, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GMNER entity inventory")
    parser.add_argument("--input", type=str, required=True, help="Train txt/jsonl file.")
    parser.add_argument("--output-dir", type=str, default="knowledge/offline")
    parser.add_argument("--image-annotation-dir", type=str, default=None)
    parser.add_argument("--image-feature-dir", type=str, default=None)
    parser.add_argument("--image-ext", type=str, default=".jpg")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--min-ambiguous-count", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)

    occurrences, inventory, summary = build_entity_inventory(
        input_path=args.input,
        image_annotation_dir=args.image_annotation_dir,
        image_feature_dir=args.image_feature_dir,
        image_ext=args.image_ext,
        iou_threshold=args.iou_threshold,
    )

    ambiguous = [
        item
        for item in inventory
        if item["ambiguous"] and item["count"] >= args.min_ambiguous_count
    ]

    write_jsonl(output_dir / "entity_occurrences.jsonl", occurrences)
    write_jsonl(output_dir / "mention_inventory.jsonl", inventory)
    write_jsonl(output_dir / "ambiguous_mentions.jsonl", ambiguous)

    with (output_dir / "inventory_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)

    print(f"records={summary['records']}")
    print(f"entities={summary['entities']}")
    print(f"unique_mentions={summary['unique_mentions']}")
    print(f"ambiguous_mentions={summary['ambiguous_mentions']}")
    print(f"review_queue={len(ambiguous)}")
    print(f"saved_to={output_dir}")


if __name__ == "__main__":
    main()
