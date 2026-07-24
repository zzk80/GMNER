"""Fine-label parsing and tensor-cache datasets for the subtype sidecar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset

from gmner.constants import ENTITY_TYPE2ID, normalize_entity_type, strip_bio_prefix

from .taxonomy import SubtypeTaxonomy


FEATURE_CACHE_KIND = "fmnerg_hierarchical_subtype_features"
FEATURE_CACHE_VERSION = 1


@dataclass(frozen=True)
class FineEntity:
    start: int
    end: int
    text: str
    coarse_type: str
    subtype: str


@dataclass(frozen=True)
class FineRecord:
    record_id: str
    image_id: str
    tokens: tuple[str, ...]
    entities: tuple[FineEntity, ...]


def _entities_from_tags(
    tokens: list[str],
    coarse_tags: list[str],
    fine_tags: list[str],
) -> tuple[FineEntity, ...]:
    if not (len(tokens) == len(coarse_tags) == len(fine_tags)):
        raise ValueError("Token, coarse-tag, and fine-tag lengths differ.")
    entities: list[FineEntity] = []
    start: int | None = None
    subtype: str | None = None

    def flush(end: int) -> None:
        nonlocal start, subtype
        if start is None or subtype is None:
            return
        coarse_values = {
            normalize_entity_type(strip_bio_prefix(tag))
            for tag in coarse_tags[start:end]
            if str(tag) != "O"
        }
        if len(coarse_values) != 1:
            raise ValueError(
                f"Fine entity [{start}, {end}) has inconsistent coarse tags: "
                f"{sorted(coarse_values)}"
            )
        coarse_type = next(iter(coarse_values))
        entities.append(
            FineEntity(
                start=start,
                end=end,
                text=" ".join(tokens[start:end]),
                coarse_type=coarse_type,
                subtype=subtype,
            )
        )
        start = None
        subtype = None

    for index, raw_tag in enumerate(fine_tags):
        tag = str(raw_tag)
        if tag == "O":
            flush(index)
            continue
        if "-" not in tag:
            raise ValueError(f"Invalid fine BIO label: {tag!r}")
        prefix, current_subtype = tag.split("-", 1)
        if prefix == "B":
            flush(index)
            start = index
            subtype = current_subtype
        elif prefix == "I":
            if start is None or subtype != current_subtype:
                flush(index)
                start = index
                subtype = current_subtype
        else:
            raise ValueError(f"Invalid fine BIO prefix: {tag!r}")
    flush(len(tokens))
    return tuple(entities)


def read_fine_conll(
    path: str | Path,
    taxonomy: SubtypeTaxonomy,
    *,
    require_all_subtypes: bool,
) -> list[FineRecord]:
    source = Path(path)
    records: list[FineRecord] = []
    tokens: list[str] = []
    coarse_tags: list[str] = []
    fine_tags: list[str] = []
    image_id = ""

    def flush() -> None:
        nonlocal tokens, coarse_tags, fine_tags, image_id
        if not tokens:
            return
        if not image_id:
            raise ValueError("Fine-label record has no IMGID.")
        entities = _entities_from_tags(tokens, coarse_tags, fine_tags)
        records.append(
            FineRecord(
                record_id=str(len(records)),
                image_id=image_id,
                tokens=tuple(tokens),
                entities=entities,
            )
        )
        tokens = []
        coarse_tags = []
        fine_tags = []
        image_id = ""

    with source.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\n")
            if not line:
                flush()
                continue
            if line.startswith("IMGID:"):
                if tokens:
                    flush()
                image_id = line.split("IMGID:", 1)[1].strip()
                continue
            fields = line.split()
            if len(fields) < 3:
                raise ValueError(
                    f"FMNERG source requires token/coarse/fine columns: {line!r}"
                )
            tokens.append(fields[0])
            coarse_tags.append(fields[1])
            fine_tags.append(fields[2])
    flush()

    observed = [entity.subtype for record in records for entity in record.entities]
    taxonomy.validate_labels(observed, require_all=require_all_subtypes)
    for record in records:
        for entity in record.entities:
            subtype_id = taxonomy.subtype_id(entity.subtype)
            if taxonomy.parent_id(subtype_id) != ENTITY_TYPE2ID[entity.coarse_type]:
                raise ValueError(
                    f"Subtype parent mismatch for {entity.subtype!r}: "
                    f"taxonomy={taxonomy.parent_id(subtype_id)} "
                    f"data={ENTITY_TYPE2ID[entity.coarse_type]}"
                )
    return records


def fine_gold_by_record(
    records: Iterable[FineRecord],
    taxonomy: SubtypeTaxonomy,
) -> dict[str, dict[tuple[int, int], dict[str, Any]]]:
    output: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for record in records:
        spans: dict[tuple[int, int], dict[str, Any]] = {}
        for entity in record.entities:
            span = (entity.start, entity.end)
            if span in spans:
                raise ValueError(
                    f"Duplicate gold span in record {record.record_id}: {span}"
                )
            spans[span] = {
                "subtype": entity.subtype,
                "subtype_id": taxonomy.subtype_id(entity.subtype),
                "coarse_type": entity.coarse_type,
                "coarse_type_id": ENTITY_TYPE2ID[entity.coarse_type],
                "text": entity.text,
            }
        output[record.record_id] = spans
    return output


class SubtypeFeatureDataset(Dataset):
    def __init__(self, payload: dict[str, Any]) -> None:
        metadata = dict(payload.get("metadata") or {})
        if metadata.get("kind") != FEATURE_CACHE_KIND:
            raise ValueError("Not an FMNERG subtype feature cache.")
        if int(metadata.get("format_version", -1)) != FEATURE_CACHE_VERSION:
            raise ValueError("Unsupported FMNERG subtype feature cache version.")
        self.metadata = metadata
        self.features = torch.as_tensor(payload["features"])
        self.coarse_type_ids = torch.as_tensor(
            payload["coarse_type_ids"], dtype=torch.long
        )
        self.subtype_ids = torch.as_tensor(
            payload["subtype_ids"], dtype=torch.long
        )
        self.examples = list(payload.get("examples") or [])
        count = self.features.size(0)
        if (
            self.coarse_type_ids.numel() != count
            or self.subtype_ids.numel() != count
            or len(self.examples) != count
        ):
            raise ValueError("Subtype feature cache arrays are misaligned.")

    @classmethod
    def from_file(cls, path: str | Path) -> "SubtypeFeatureDataset":
        return cls(torch.load(Path(path), map_location="cpu"))

    def __len__(self) -> int:
        return self.features.size(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features": self.features[index].float(),
            "coarse_type_ids": self.coarse_type_ids[index],
            "subtype_ids": self.subtype_ids[index],
            "example_index": index,
        }
