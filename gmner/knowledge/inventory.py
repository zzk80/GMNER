"""Entity inventory construction for offline GMNER knowledge bases."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import math
import re
import xml.etree.ElementTree as ET

import numpy as np

from gmner.constants import DEFAULT_LABEL2ID
from gmner.utils.io import read_jsonl


ID2LABEL = {value: key for key, value in DEFAULT_LABEL2ID.items()}


def normalize_mention(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"^[@#]+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def read_conll_records(path: str | Path, image_ext: str = ".jpg") -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    tokens: list[str] = []
    tags: list[str] = []
    image_id: str | None = None

    def flush_record() -> None:
        if image_id and tokens:
            records.append(
                {
                    "id": len(records),
                    "image_id": image_id,
                    "image": f"{image_id}{image_ext}",
                    "tokens": tokens.copy(),
                    "ner_tags": tags.copy(),
                }
            )

    with path.open("r", encoding="utf-8") as fp:
        for raw_line in fp:
            line = raw_line.rstrip("\n")
            if not line:
                flush_record()
                tokens.clear()
                tags.clear()
                image_id = None
                continue

            if line.startswith("IMGID:"):
                if image_id and tokens:
                    flush_record()
                    tokens.clear()
                    tags.clear()
                image_id = line.split("IMGID:", 1)[1].strip()
                continue

            parts = line.split()
            if len(parts) >= 2:
                tokens.append(parts[0])
                tags.append(parts[-1])

    flush_record()
    return records


def read_records(path: str | Path, image_ext: str = ".jpg") -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".txt":
        return read_conll_records(path, image_ext=image_ext)

    records = read_jsonl(path)
    for idx, record in enumerate(records):
        record.setdefault("id", idx)
        image = str(record.get("image", ""))
        record.setdefault("image_id", Path(image).stem)
        tags = []
        for tag in record.get("ner_tags", []):
            if isinstance(tag, int):
                tags.append(ID2LABEL.get(tag, "O"))
            else:
                tags.append(str(tag))
        record["ner_tags"] = tags
    return records


def extract_entities_from_record(record: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = list(record.get("tokens") or [])
    tags = list(record.get("ner_tags") or [])
    entities: list[dict[str, Any]] = []
    start = None
    current_type = None

    for idx, tag in enumerate(tags + ["O"]):
        starts_entity = tag.startswith("B-")
        continues_wrong_type = tag.startswith("I-") and (
            start is None or current_type != tag[2:]
        )
        closes_entity = tag == "O" or starts_entity or continues_wrong_type

        if closes_entity and start is not None:
            mention = " ".join(tokens[start:idx])
            entities.append(
                {
                    "mention": mention,
                    "normalized_mention": normalize_mention(mention),
                    "entity_type": current_type,
                    "start": start,
                    "end": idx,
                    "context": " ".join(tokens),
                }
            )
            start = None
            current_type = None

        if starts_entity or continues_wrong_type:
            start = idx
            current_type = tag[2:]

    return entities


def read_xml_boxes(xml_dir: Path | None, image_id: str) -> dict[str, list[list[float]]]:
    if xml_dir is None:
        return {}

    xml_path = xml_dir / f"{image_id}.xml"
    if not xml_path.exists():
        return {}

    boxes: dict[str, list[list[float]]] = defaultdict(list)
    try:
        root = ET.parse(str(xml_path)).getroot()
    except Exception:
        return {}

    for obj in root.findall("object"):
        name_node = obj.find("name")
        box_node = obj.find("bndbox")
        if name_node is None or box_node is None:
            continue
        try:
            box = [
                float(box_node.findtext("xmin", default="0")),
                float(box_node.findtext("ymin", default="0")),
                float(box_node.findtext("xmax", default="0")),
                float(box_node.findtext("ymax", default="0")),
            ]
        except Exception:
            continue
        boxes[normalize_mention(name_node.text or "")].append(box)
    return dict(boxes)


def load_region_boxes(feature_dir: Path | None, image_id: str) -> np.ndarray | None:
    if feature_dir is None:
        return None

    npz_path = feature_dir / f"{image_id}.jpg.npz"
    if not npz_path.exists():
        return None

    try:
        data = np.load(str(npz_path))
        if "bounding_boxes" not in data:
            return None
        num_boxes = int(data["num_boxes"]) if "num_boxes" in data else data["bounding_boxes"].shape[0]
        return data["bounding_boxes"][:num_boxes].astype("float32")
    except Exception:
        return None


def box_iou(box: list[float], candidates: np.ndarray) -> np.ndarray:
    if candidates.size == 0:
        return np.zeros((0,), dtype="float32")

    query = np.asarray(box, dtype="float32")
    x1 = np.maximum(query[0], candidates[:, 0])
    y1 = np.maximum(query[1], candidates[:, 1])
    x2 = np.minimum(query[2], candidates[:, 2])
    y2 = np.minimum(query[3], candidates[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    query_area = max(0.0, query[2] - query[0]) * max(0.0, query[3] - query[1])
    cand_area = np.maximum(0.0, candidates[:, 2] - candidates[:, 0]) * np.maximum(
        0.0,
        candidates[:, 3] - candidates[:, 1],
    )
    union = np.maximum(query_area + cand_area - inter, 1e-6)
    return inter / union


def attach_grounding_metadata(
    occurrence: dict[str, Any],
    xml_boxes: dict[str, list[list[float]]],
    region_boxes: np.ndarray | None,
    iou_threshold: float,
) -> None:
    gt_boxes = xml_boxes.get(occurrence["normalized_mention"], [])
    occurrence["groundable"] = bool(gt_boxes)
    occurrence["gt_boxes"] = gt_boxes
    occurrence["has_region_features"] = region_boxes is not None
    occurrence["best_region_index"] = None
    occurrence["best_region_iou"] = 0.0

    if not gt_boxes or region_boxes is None or region_boxes.size == 0:
        return

    best_index = None
    best_iou = 0.0
    for gt_box in gt_boxes:
        ious = box_iou(gt_box, region_boxes)
        if ious.size == 0:
            continue
        idx = int(ious.argmax())
        value = float(ious[idx])
        if value > best_iou:
            best_iou = value
            best_index = idx

    occurrence["best_region_index"] = best_index if best_iou >= iou_threshold else None
    occurrence["best_region_iou"] = best_iou


def entropy(counter: Counter[str]) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in counter.values())


def build_entity_inventory(
    input_path: str | Path,
    image_annotation_dir: str | Path | None = None,
    image_feature_dir: str | Path | None = None,
    image_ext: str = ".jpg",
    iou_threshold: float = 0.5,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    records = read_records(input_path, image_ext=image_ext)
    xml_dir = Path(image_annotation_dir) if image_annotation_dir else None
    feature_dir = Path(image_feature_dir) if image_feature_dir else None

    occurrences: list[dict[str, Any]] = []
    xml_cache: dict[str, dict[str, list[list[float]]]] = {}
    region_cache: dict[str, np.ndarray | None] = {}

    for record in records:
        image_id = str(record.get("image_id") or Path(str(record.get("image", ""))).stem)
        if image_id not in xml_cache:
            xml_cache[image_id] = read_xml_boxes(xml_dir, image_id)
        if image_id not in region_cache:
            region_cache[image_id] = load_region_boxes(feature_dir, image_id)

        for entity_idx, entity in enumerate(extract_entities_from_record(record)):
            occurrence = {
                "occurrence_id": f"{record.get('id')}:{entity_idx}",
                "record_id": str(record.get("id")),
                "image_id": image_id,
                **entity,
            }
            attach_grounding_metadata(
                occurrence=occurrence,
                xml_boxes=xml_cache[image_id],
                region_boxes=region_cache[image_id],
                iou_threshold=iou_threshold,
            )
            occurrences.append(occurrence)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in occurrences:
        grouped[occurrence["normalized_mention"]].append(occurrence)

    inventory: list[dict[str, Any]] = []
    for mention, items in grouped.items():
        type_counts = Counter(str(item["entity_type"]) for item in items)
        groundable_count = sum(1 for item in items if item.get("groundable"))
        matched_region_count = sum(1 for item in items if item.get("best_region_index") is not None)
        examples = []
        for item in items[:5]:
            examples.append(
                {
                    "occurrence_id": item["occurrence_id"],
                    "type": item["entity_type"],
                    "context": item["context"],
                    "groundable": item.get("groundable", False),
                }
            )

        type_entropy = entropy(type_counts)
        inventory.append(
            {
                "mention": mention,
                "display_mention": items[0]["mention"],
                "count": len(items),
                "type_counts": dict(type_counts),
                "ambiguous": len(type_counts) > 1,
                "type_entropy": type_entropy,
                "priority_score": type_entropy * math.log1p(len(items)),
                "groundable_count": groundable_count,
                "matched_region_count": matched_region_count,
                "groundability_rate": groundable_count / max(1, len(items)),
                "region_match_rate": matched_region_count / max(1, len(items)),
                "examples": examples,
            }
        )

    inventory.sort(key=lambda item: (-item["priority_score"], -item["count"], item["mention"]))

    summary = {
        "records": len(records),
        "entities": len(occurrences),
        "unique_mentions": len(inventory),
        "ambiguous_mentions": sum(1 for item in inventory if item["ambiguous"]),
        "type_counts": dict(Counter(str(item["entity_type"]) for item in occurrences)),
        "groundable_entities": sum(1 for item in occurrences if item.get("groundable")),
        "matched_region_entities": sum(1 for item in occurrences if item.get("best_region_index") is not None),
    }
    return occurrences, inventory, summary
