"""Encode external subtype knowledge and build fixed multi-center prototypes."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.constants import ENTITY_TYPE2ID, normalize_entity_type
from gmner.models.external_knowledge import normalize_subtype_name
from gmner.utils.io import read_jsonl


TYPE_NAMES = ["LOC", "PER", "ORG", "OTHER"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a fixed external subtype prototype bank without an LLM."
    )
    parser.add_argument("--knowledge-file", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", default="knowledge/external")
    parser.add_argument("--max-centers-per-subtype", type=int, default=3)
    parser.add_argument("--kmeans-iterations", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_knowledge_records(path: str | Path) -> list[dict]:
    """Validate and deduplicate the source JSONL while preserving provenance."""

    raw_records = read_jsonl(path)
    records: list[dict] = []
    seen: set[tuple[str, str, str]] = set()
    for index, raw in enumerate(raw_records):
        coarse_type = normalize_entity_type(
            raw.get("coarse_type", raw.get("entity_type", ""))
        )
        fine_type = normalize_subtype_name(
            raw.get("fine_type", raw.get("subtype", ""))
        )
        text = str(raw.get("text", raw.get("description", ""))).strip()
        if coarse_type not in ENTITY_TYPE2ID or coarse_type == "O":
            raise ValueError(f"Record {index} has invalid coarse_type={coarse_type!r}.")
        if not fine_type:
            raise ValueError(f"Record {index} has no fine_type/subtype.")
        if not text:
            raise ValueError(f"Record {index} has no knowledge text.")
        key = (coarse_type, fine_type, " ".join(text.lower().split()))
        if key in seen:
            continue
        seen.add(key)
        confidence = float(raw.get("confidence", 1.0))
        records.append(
            {
                **raw,
                "id": str(raw.get("id", f"knowledge:{index}")),
                "coarse_type": coarse_type,
                "fine_type": fine_type,
                "text": text,
                "confidence": min(max(confidence, 0.0), 1.0),
                "source": str(raw.get("source", "unspecified")),
            }
        )
    if not records:
        raise ValueError("No valid external knowledge records were found.")
    return records


def _weighted_center(features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    center = torch.sum(features * weights.unsqueeze(-1), dim=0)
    return F.normalize(center, dim=0, eps=1e-6)


def spherical_kmeans(
    features: torch.Tensor,
    weights: torch.Tensor,
    num_centers: int,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic weighted spherical k-means for a single subtype."""

    features = F.normalize(features.float(), dim=-1, eps=1e-6)
    weights = weights.float().clamp_min(1e-6)
    num_centers = max(1, min(int(num_centers), features.size(0)))
    first = int(
        torch.matmul(features, _weighted_center(features, weights)).argmax().item()
    )
    selected = [first]
    while len(selected) < num_centers:
        similarities = torch.matmul(features, features[selected].transpose(0, 1))
        nearest = similarities.max(dim=-1).values
        nearest[selected] = 1.0
        selected.append(int(nearest.argmin().item()))
    centers = features[selected].clone()

    assignments = torch.zeros(features.size(0), dtype=torch.long)
    for _ in range(max(1, int(iterations))):
        assignments = torch.matmul(features, centers.transpose(0, 1)).argmax(dim=-1)
        updated = []
        for center_id in range(num_centers):
            mask = assignments == center_id
            if torch.any(mask):
                updated.append(_weighted_center(features[mask], weights[mask]))
            else:
                updated.append(centers[center_id])
        new_centers = torch.stack(updated)
        if torch.allclose(new_centers, centers, atol=1e-6, rtol=0.0):
            centers = new_centers
            break
        centers = new_centers
    assignments = torch.matmul(features, centers.transpose(0, 1)).argmax(dim=-1)
    return F.normalize(centers, dim=-1), assignments


def build_prototype_payload(
    records: list[dict],
    embeddings: torch.Tensor,
    max_centers_per_subtype: int = 3,
    kmeans_iterations: int = 20,
) -> dict:
    if embeddings.ndim != 2 or embeddings.size(0) != len(records):
        raise ValueError("Embeddings must have shape [len(records), hidden_size].")
    if max_centers_per_subtype < 1:
        raise ValueError("max_centers_per_subtype must be positive.")

    grouped: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        type_id = ENTITY_TYPE2ID[record["coarse_type"]]
        grouped[(type_id, record["fine_type"])].append(index)

    subtype_keys = sorted(grouped)
    subtype_names = [fine_type for _, fine_type in subtype_keys]
    subtype_type_ids = torch.tensor(
        [type_id for type_id, _ in subtype_keys],
        dtype=torch.long,
    )
    prototype_vectors = []
    prototype_type_ids = []
    prototype_subtype_ids = []
    prototype_metadata = []
    normalized = F.normalize(embeddings.float(), dim=-1, eps=1e-6)

    for subtype_id, (type_id, fine_type) in enumerate(subtype_keys):
        indices = grouped[(type_id, fine_type)]
        group_features = normalized[indices]
        group_weights = torch.tensor(
            [records[index]["confidence"] for index in indices],
            dtype=torch.float32,
        )
        centers, assignments = spherical_kmeans(
            features=group_features,
            weights=group_weights,
            num_centers=min(max_centers_per_subtype, len(indices)),
            iterations=kmeans_iterations,
        )
        for center_id, center in enumerate(centers):
            member_positions = torch.nonzero(
                assignments == center_id,
                as_tuple=False,
            ).squeeze(-1).tolist()
            member_indices = [indices[position] for position in member_positions]
            prototype_vectors.append(center)
            prototype_type_ids.append(type_id)
            prototype_subtype_ids.append(subtype_id)
            prototype_metadata.append(
                {
                    "prototype_id": f"{TYPE_NAMES[type_id]}:{fine_type}:{center_id}",
                    "coarse_type": TYPE_NAMES[type_id],
                    "fine_type": fine_type,
                    "support": len(member_indices),
                    "knowledge_ids": [records[index]["id"] for index in member_indices],
                    "sources": dict(
                        Counter(records[index]["source"] for index in member_indices)
                    ),
                }
            )

    return {
        "prototypes": torch.stack(prototype_vectors),
        "prototype_type_ids": torch.tensor(prototype_type_ids, dtype=torch.long),
        "prototype_subtype_ids": torch.tensor(
            prototype_subtype_ids,
            dtype=torch.long,
        ),
        "subtype_type_ids": subtype_type_ids,
        "type_names": TYPE_NAMES,
        "subtype_names": subtype_names,
        "prototype_metadata": prototype_metadata,
    }


def encode_texts(
    texts: Iterable[str],
    model_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    text_list = list(texts)
    outputs = []
    with torch.no_grad():
        for start in range(0, len(text_list), batch_size):
            encoded = tokenizer(
                text_list[start : start + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            special_mask = encoded.pop("special_tokens_mask").to(device)
            encoded = {key: value.to(device) for key, value in encoded.items()}
            states = model(**encoded).last_hidden_state
            content_mask = encoded["attention_mask"].bool() & ~special_mask.bool()
            empty = ~content_mask.any(dim=-1)
            if torch.any(empty):
                content_mask[empty] = encoded["attention_mask"][empty].bool()
            denominator = content_mask.sum(dim=-1, keepdim=True).clamp_min(1)
            pooled = (
                states * content_mask.unsqueeze(-1).to(dtype=states.dtype)
            ).sum(dim=1) / denominator.to(dtype=states.dtype)
            outputs.append(pooled.cpu())
    return torch.cat(outputs, dim=0)


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("batch-size and max-length must be positive.")
    records = load_knowledge_records(args.knowledge_file)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    embeddings = encode_texts(
        texts=[record["text"] for record in records],
        model_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device,
    )
    payload = build_prototype_payload(
        records=records,
        embeddings=embeddings,
        max_centers_per_subtype=args.max_centers_per_subtype,
        kmeans_iterations=args.kmeans_iterations,
    )
    source_path = Path(args.knowledge_file).resolve()
    payload.update(
        {
            "encoder_name": args.model_name,
            "source_file": str(source_path),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "knowledge_record_count": len(records),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "subtype_prototypes.pt"
    torch.save(payload, output_path)
    summary = {
        "knowledge_records": len(records),
        "types": len(payload["type_names"]),
        "subtypes": len(payload["subtype_names"]),
        "prototype_centers": int(payload["prototypes"].size(0)),
        "hidden_size": int(payload["prototypes"].size(1)),
        "encoder_name": args.model_name,
        "source_file": str(source_path),
        "source_sha256": payload["source_sha256"],
        "output": str(output_path.resolve()),
        "prototypes": payload["prototype_metadata"],
    }
    summary_path = output_dir / "subtype_prototypes.summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "prototypes"}, indent=2))


if __name__ == "__main__":
    main()
