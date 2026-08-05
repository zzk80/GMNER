#!/usr/bin/env python3
"""Materialize frozen Train/Dev features for the sparse visual type probe."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from gmner.config import load_config
from gmner.constants import DEFAULT_LABEL2ID
from gmner.data.artifact_utils import sha256_file
from gmner.data.graph_builders import GraphBuilderConfig, TextGraphBuilder
from gmner.data.record_level_stage1_collator import RecordLevelStage1Collator
from gmner.data.record_level_stage1_dataset import RecordLevelStage1Dataset
from gmner.data.tokenization import load_word_aligned_tokenizer
from gmner.engine.utils import move_batch_to_device
from gmner.models import GMNERModel
from gmner.models.stage1.formal_record_encoder import (
    FrozenFormalRecordEncoder,
    decoded_record_entities,
)
from scripts.train import build_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fmnerg_twitter10000_stage1.yaml")
    parser.add_argument(
        "--checkpoint", default="outputs/fmnerg_stage1_roberta128/best_model.pt"
    )
    parser.add_argument(
        "--output-dir", default="knowledge/span_sparse_visual_type/seed42"
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--splits", nargs="+", choices=("train", "dev"), default=("train", "dev"))
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _geometry(
    boxes: torch.Tensor,
    image_size: torch.Tensor,
) -> torch.Tensor:
    height = image_size[0].float().clamp_min(1.0)
    width = image_size[1].float().clamp_min(1.0)
    x1 = boxes[:, 0].float() / width
    y1 = boxes[:, 1].float() / height
    x2 = boxes[:, 2].float() / width
    y2 = boxes[:, 3].float() / height
    box_width = (x2 - x1).clamp_min(0.0)
    box_height = (y2 - y1).clamp_min(0.0)
    area = box_width * box_height
    aspect = (box_width / box_height.clamp_min(1e-4)).clamp_max(20.0) / 20.0
    return torch.stack([x1, y1, x2, y2, area, aspect], dim=-1)


def _slice_entities(value: torch.Tensor, row: int, count: int, regions: int | None = None) -> torch.Tensor:
    result = value[row, :count]
    if regions is not None:
        result = result[..., :regions]
    return result.detach().cpu()


def _canonical_digest(record_id: str, spans: list[list[int]], type_ids: torch.Tensor) -> bytes:
    values = sorted(
        (int(span[0]), int(span[1]), int(type_ids[index].item()))
        for index, span in enumerate(spans)
    )
    return json.dumps([record_id, values], separators=(",", ":")).encode("utf-8")


def _anchor_type_logits(
    logits: torch.Tensor,
    type_ids: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Preserve formal CRF types while retaining pooled emission margins."""

    if logits.size(1) == 0:
        return logits
    anchored = logits.clone()
    safe_type_ids = type_ids.clamp(0, logits.size(-1) - 1)
    required = logits.max(dim=-1).values + 1e-4
    current = anchored.gather(-1, safe_type_ids.unsqueeze(-1)).squeeze(-1)
    replacement = torch.maximum(current, required)
    replacement = torch.where(valid, replacement, current)
    return anchored.scatter(
        -1, safe_type_ids.unsqueeze(-1), replacement.unsqueeze(-1)
    )


@torch.no_grad()
def materialize_split(
    *,
    split: str,
    dataset: Any,
    tokenizer: Any,
    encoder: FrozenFormalRecordEncoder,
    device: torch.device,
    batch_size: int,
    max_records: int | None,
) -> dict[str, Any]:
    records = RecordLevelStage1Dataset(dataset, split=split)
    if max_records is not None:
        records = torch.utils.data.Subset(records, range(min(max_records, len(records))))
    loader = DataLoader(
        records,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=RecordLevelStage1Collator(tokenizer),
    )
    payload_records: list[dict[str, Any]] = []
    prediction_digest = hashlib.sha256()
    formal_prediction_count = 0
    matched_prediction_count = 0
    base_mner_correct = 0
    gold_count = 0

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        encoded = encoder.encode_records(batch)
        formal = decoded_record_entities(encoded["decoded_tags"], batch)
        formal_states = encoder.span_states(encoded["fused_tokens"], formal["masks"])
        formal_type_logits = encoder.type_logits(encoded["ner_logits"], formal["masks"])
        formal_type_logits = _anchor_type_logits(
            formal_type_logits,
            formal["type_ids"],
            formal["valid"],
        )
        formal_grounding = encoder.score_entities(
            fused_tokens=encoded["fused_tokens"],
            image_nodes=encoded["image_nodes"],
            entity_masks=formal["masks"],
            entity_type_ids=formal["type_ids"],
            batch=batch,
            null_prior=0.5,
        )

        gold_masks = batch["gold_subword_masks"]
        gold_type_logits = encoder.type_logits(encoded["ner_logits"], gold_masks)
        gold_base_types = gold_type_logits.argmax(dim=-1)
        gold_states = encoder.span_states(encoded["fused_tokens"], gold_masks)
        gold_grounding = encoder.score_entities(
            fused_tokens=encoded["fused_tokens"],
            image_nodes=encoded["image_nodes"],
            entity_masks=gold_masks,
            entity_type_ids=gold_base_types,
            batch=batch,
            null_prior=0.5,
        )

        for row, metadata in enumerate(batch["metadata"]):
            record_id = str(metadata["record_id"])
            spans = formal["spans"][row]
            entity_count = len(spans)
            region_count = int(batch["null_region_index"][row].item()) + 1
            gold_valid = batch["gold_entity_mask"][row].bool()
            gold_valid_indices = gold_valid.nonzero(as_tuple=False).squeeze(-1)
            gold_indices = {
                tuple(int(v) for v in batch["gold_spans"][row, index].tolist()): index
                for index in range(gold_valid.size(0))
                if bool(gold_valid[index].item())
            }
            matched_types = torch.full((entity_count,), -1, dtype=torch.long)
            positive = torch.zeros((entity_count, region_count), dtype=torch.bool)
            type_valid = torch.zeros(entity_count, dtype=torch.bool)
            for entity_index, span in enumerate(spans):
                gold_index = gold_indices.get((int(span[0]), int(span[1])))
                if gold_index is None:
                    continue
                type_valid[entity_index] = True
                matched_types[entity_index] = int(
                    batch["gold_type_ids"][row, gold_index].item()
                )
                positive[entity_index] = batch["gold_region_positive_mask"][
                    row, gold_index, :region_count
                ].detach().cpu().bool()
            null_index = int(batch["null_region_index"][row].item())
            real_positive = positive.clone()
            real_positive[:, null_index] = False
            gold_visible = real_positive.any(dim=-1)
            base_types = formal["type_ids"][row, :entity_count].detach().cpu()
            prediction_digest.update(_canonical_digest(record_id, spans, base_types))
            prediction_digest.update(b"\n")
            formal_prediction_count += entity_count
            matched_prediction_count += int(type_valid.sum().item())
            base_mner_correct += int(
                (base_types.eq(matched_types) & type_valid).sum().item()
            )
            current_gold_count = int(gold_valid.sum().item())
            gold_count += current_gold_count
            region_mask = batch["region_mask"][row, :region_count].detach().cpu().bool()
            region_is_null = batch["region_is_null"][row, :region_count].detach().cpu().bool()
            payload_records.append(
                {
                    "record_id": record_id,
                    "formal_spans": torch.tensor(spans, dtype=torch.long).reshape(-1, 2),
                    "span_states": _slice_entities(formal_states, row, entity_count).half(),
                    "base_type_logits": _slice_entities(formal_type_logits, row, entity_count).float(),
                    "formal_type_ids": base_types,
                    "formal_grounding_logits": _slice_entities(
                        formal_grounding["formal_logits"], row, entity_count, region_count
                    ).float(),
                    "compatibility": _slice_entities(
                        formal_grounding["compatibility"], row, entity_count, region_count
                    ).float(),
                    "gold_type_ids": matched_types,
                    "type_valid": type_valid,
                    "gold_region_positive_mask": positive,
                    "gold_visible": gold_visible,
                    "region_states": encoded["image_nodes"][row, :region_count].detach().cpu().half(),
                    "region_mask": region_mask,
                    "region_is_null": region_is_null,
                    "region_scores": batch["region_scores"][row, :region_count].detach().cpu().float(),
                    "region_geometry": _geometry(
                        batch["region_boxes"][row, :region_count].detach().cpu(),
                        batch["image_sizes"][row].detach().cpu(),
                    ),
                    "gold_count": current_gold_count,
                    "gold_probe": {
                        "span_states": gold_states[row].index_select(
                            0, gold_valid_indices
                        ).detach().cpu().half(),
                        "base_type_logits": gold_type_logits[row].index_select(
                            0, gold_valid_indices
                        ).detach().cpu().float(),
                        "formal_grounding_logits": gold_grounding[
                            "formal_logits"
                        ][row].index_select(0, gold_valid_indices)[
                            :, :region_count
                        ].detach().cpu().float(),
                        "compatibility": gold_grounding["compatibility"][
                            row
                        ].index_select(0, gold_valid_indices)[
                            :, :region_count
                        ].detach().cpu().float(),
                        "gold_type_ids": batch["gold_type_ids"][row].index_select(
                            0, gold_valid_indices
                        ).detach().cpu().long(),
                        "gold_region_positive_mask": batch[
                            "gold_region_positive_mask"
                        ][row].index_select(0, gold_valid_indices)[
                            :, :region_count
                        ].detach().cpu().bool(),
                    },
                }
            )
    return {
        "kind": "span_sparse_visual_type_cache",
        "format_version": 1,
        "split": split,
        "records": payload_records,
        "summary": {
            "record_count": len(payload_records),
            "formal_prediction_count": formal_prediction_count,
            "matched_prediction_count": matched_prediction_count,
            "base_mner_correct": base_mner_correct,
            "gold_count": gold_count,
            "formal_prediction_sha256": prediction_digest.hexdigest(),
        },
        "test_accessed": False,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    checkpoint_path = resolve(args.checkpoint, root)
    output_dir = resolve(args.output_dir, root)
    config = load_config(config_path)
    tokenizer = load_word_aligned_tokenizer(
        config.model.text_model_name, local_files_only=True
    )
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=config.data.use_dependency_graph,
            dependency_backend=config.data.dependency_backend,
            dependency_model=config.data.dependency_model,
            window_size=config.data.graph_window_size,
        )
    )
    train_data, dev_data, _, num_labels = build_datasets(
        config=config,
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        project_root=root,
        output_dir=output_dir / "converted",
        build_test=False,
    )
    datasets = {"train": train_data, "dev": dev_data}
    teacher = GMNERModel(config=config, num_labels=num_labels)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    teacher.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    encoder = FrozenFormalRecordEncoder(teacher.to(device)).to(device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        payload = materialize_split(
            split=split,
            dataset=datasets[split],
            tokenizer=tokenizer,
            encoder=encoder,
            device=device,
            batch_size=args.batch_size,
            max_records=args.max_records,
        )
        payload["provenance"] = {
            "config_sha256": sha256_file(config_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "diagnostic_in_sample_train": split == "train",
        }
        path = output_dir / f"{split}.pt"
        torch.save(payload, path)
        print(json.dumps({"split": split, "path": str(path), **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
