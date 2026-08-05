#!/usr/bin/env python3
"""Train the fixed-span sparse visual coarse-type diagnostic on Train/Dev."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from gmner.models.span_sparse_visual_type import (
    SparseVisualTypeConfig,
    SpanConditionedSparseVisualTypeRefiner,
    sparse_visual_type_loss,
)


FROZEN_DEV = {
    "prediction_count": 2516,
    "gold_count": 2450,
    "span_correct": 2162,
    "mner_correct": 2023,
    "span_f1": 0.8707209021345148,
    "mner_f1": 0.8147402335884012,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/span_sparse_visual_type/seed42.yaml"
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


class RecordCacheDataset(Dataset):
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.records[index]


def collate_records(
    records: list[dict[str, Any]],
    *,
    gold_probe: bool = False,
) -> dict[str, Any]:
    batch_size = len(records)
    entities = [
        int(
            (record["gold_probe"] if gold_probe else record)["span_states"].size(0)
        )
        for record in records
    ]
    regions = [int(record["region_states"].size(0)) for record in records]
    max_entities = max(entities, default=0)
    max_regions = max(regions, default=0)
    batch: dict[str, Any] = {
        "span_states": torch.zeros(batch_size, max_entities, 2304, dtype=torch.float16),
        "region_states": torch.zeros(batch_size, max_regions, 768, dtype=torch.float16),
        "base_type_logits": torch.zeros(batch_size, max_entities, 4),
        "formal_grounding_logits": torch.full(
            (batch_size, max_entities, max_regions), -1e4
        ),
        "compatibility": torch.zeros(batch_size, max_entities, max_regions),
        "gold_type_ids": torch.full((batch_size, max_entities), -1, dtype=torch.long),
        "type_valid": torch.zeros(batch_size, max_entities, dtype=torch.bool),
        "gold_region_positive_mask": torch.zeros(
            batch_size, max_entities, max_regions, dtype=torch.bool
        ),
        "gold_visible": torch.zeros(batch_size, max_entities, dtype=torch.bool),
        "entity_mask": torch.zeros(batch_size, max_entities, dtype=torch.bool),
        "region_mask": torch.zeros(batch_size, max_regions, dtype=torch.bool),
        "region_is_null": torch.zeros(batch_size, max_regions, dtype=torch.bool),
        "region_scores": torch.zeros(batch_size, max_regions),
        "region_geometry": torch.zeros(batch_size, max_regions, 6),
        "record_ids": [],
        "gold_counts": torch.tensor([int(item["gold_count"]) for item in records]),
    }
    for row, record in enumerate(records):
        source = record["gold_probe"] if gold_probe else record
        entity_count = entities[row]
        region_count = regions[row]
        for key in ("span_states", "base_type_logits"):
            batch[key][row, :entity_count] = source[key]
        for key in ("formal_grounding_logits", "compatibility"):
            batch[key][row, :entity_count, :region_count] = source[key]
        batch["gold_type_ids"][row, :entity_count] = source["gold_type_ids"]
        batch["type_valid"][row, :entity_count] = True if gold_probe else source["type_valid"]
        batch["entity_mask"][row, :entity_count] = True
        batch["gold_region_positive_mask"][
            row, :entity_count, :region_count
        ] = source["gold_region_positive_mask"]
        null_index = int(record["region_is_null"].long().argmax().item())
        positive = source["gold_region_positive_mask"]
        real_positive = positive.clone()
        real_positive[:, null_index] = False
        visible = real_positive.any(dim=-1)
        batch["gold_visible"][row, :entity_count] = visible
        for key in (
            "region_states",
            "region_mask",
            "region_is_null",
            "region_scores",
            "region_geometry",
        ):
            batch[key][row, :region_count] = record[key]
        batch["record_ids"].append(record["record_id"])
    return batch


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def forward_model(
    model: SpanConditionedSparseVisualTypeRefiner,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return model(
        span_states=batch["span_states"],
        region_states=batch["region_states"],
        base_type_logits=batch["base_type_logits"],
        formal_grounding_logits=batch["formal_grounding_logits"],
        compatibility=batch["compatibility"],
        region_scores=batch["region_scores"],
        region_geometry=batch["region_geometry"],
        entity_mask=batch["entity_mask"],
        region_mask=batch["region_mask"],
        region_is_null=batch["region_is_null"],
    )


def f1(correct: int, predicted: int, gold: int) -> float:
    return 2.0 * correct / max(predicted + gold, 1)


@torch.no_grad()
def evaluate(
    model: SpanConditionedSparseVisualTypeRefiner,
    records: list[dict[str, Any]],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    loader = DataLoader(
        RecordCacheDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )
    predicted = gold = span_correct = base_correct_count = final_correct = 0
    changed = corrected = damaged = 0
    base_correct_total = base_correct_preserved = 0
    region_count = region_top1 = region_top3 = 0
    entropy_sum = gate_sum = gate_count = 0.0
    per_type = {
        name: {"corrected": 0, "damaged": 0, "net": 0}
        for name in ("LOC", "PER", "ORG", "OTHER")
    }
    digest = hashlib.sha256()
    for batch in loader:
        batch = move(batch, device)
        outputs = forward_model(model, batch)
        base = batch["base_type_logits"].argmax(dim=-1)
        final = outputs["adjusted_type_logits"].argmax(dim=-1)
        valid = batch["type_valid"] & batch["entity_mask"]
        base_correct = base.eq(batch["gold_type_ids"]) & valid
        final_is_correct = final.eq(batch["gold_type_ids"]) & valid
        changed_mask = final.ne(base) & batch["entity_mask"]
        corrected_mask = ~base_correct & final_is_correct
        damaged_mask = base_correct & ~final_is_correct
        predicted += int(batch["entity_mask"].sum().item())
        gold += int(batch["gold_counts"].sum().item())
        span_correct += int(valid.sum().item())
        base_correct_count += int(base_correct.sum().item())
        final_correct += int(final_is_correct.sum().item())
        changed += int(changed_mask.sum().item())
        corrected += int(corrected_mask.sum().item())
        damaged += int(damaged_mask.sum().item())
        base_correct_total += int(base_correct.sum().item())
        base_correct_preserved += int((base_correct & final_is_correct).sum().item())
        for type_id, name in enumerate(("LOC", "PER", "ORG", "OTHER")):
            gold_type = batch["gold_type_ids"].eq(type_id)
            per_type[name]["corrected"] += int((corrected_mask & gold_type).sum().item())
            per_type[name]["damaged"] += int((damaged_mask & gold_type).sum().item())
        positive = batch["gold_region_positive_mask"] & outputs["region_candidate_mask"]
        region_valid = valid & batch["gold_visible"] & positive.any(dim=-1)
        top1 = outputs["region_scores"].argmax(dim=-1)
        top1_hit = positive.gather(-1, top1.unsqueeze(-1)).squeeze(-1)
        top3_hit = (positive & outputs["region_topk_mask"]).any(dim=-1)
        region_count += int(region_valid.sum().item())
        region_top1 += int((top1_hit & region_valid).sum().item())
        region_top3 += int((top3_hit & region_valid).sum().item())
        entropy_sum += float(
            (outputs["attention_entropy_r16"] * region_valid).sum().item()
        )
        gate_sum += float((outputs["type_gate"] * batch["entity_mask"]).sum().item())
        gate_count += float(batch["entity_mask"].sum().item())
        for row, record_id in enumerate(batch["record_ids"]):
            values = [
                int(value)
                for value in final[row, batch["entity_mask"][row]].tolist()
            ]
            digest.update(json.dumps([record_id, values], separators=(",", ":")).encode())
            digest.update(b"\n")
    for values in per_type.values():
        values["net"] = values["corrected"] - values["damaged"]

    gold_loader = DataLoader(
        RecordCacheDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda rows: collate_records(rows, gold_probe=True),
    )
    gold_base_correct = gold_final_correct = gold_probe_count = 0
    for batch in gold_loader:
        batch = move(batch, device)
        outputs = forward_model(model, batch)
        valid = batch["entity_mask"]
        gold_probe_count += int(valid.sum().item())
        gold_base_correct += int(
            (
                batch["base_type_logits"].argmax(dim=-1).eq(batch["gold_type_ids"])
                & valid
            ).sum().item()
        )
        gold_final_correct += int(
            (
                outputs["adjusted_type_logits"].argmax(dim=-1).eq(
                    batch["gold_type_ids"]
                )
                & valid
            ).sum().item()
        )
    baseline_mner = f1(base_correct_count, predicted, gold)
    final_mner = f1(final_correct, predicted, gold)
    baseline_span = f1(span_correct, predicted, gold)
    return {
        "prediction_count": predicted,
        "gold_count": gold,
        "span_correct": span_correct,
        "span_f1": baseline_span,
        "base_mner_correct": base_correct_count,
        "base_mner_f1": baseline_mner,
        "mner_correct": final_correct,
        "mner_f1": final_mner,
        "mner_delta": final_mner - baseline_mner,
        "mner_correct_delta": final_correct - base_correct_count,
        "type_changed": changed,
        "corrected": corrected,
        "damaged": damaged,
        "net_correction": corrected - damaged,
        "base_correct_preservation": base_correct_preserved / max(base_correct_total, 1),
        "predicted_correct_span_type_accuracy": final_correct / max(span_correct, 1),
        "gold_span_base_type_accuracy": gold_base_correct / max(gold_probe_count, 1),
        "gold_span_final_type_accuracy": gold_final_correct / max(gold_probe_count, 1),
        "region_covered_visible_count": region_count,
        "gold_region_recall_at_1": region_top1 / max(region_count, 1),
        "gold_region_recall_at_3": region_top3 / max(region_count, 1),
        "attention_entropy_r16": entropy_sum / max(region_count, 1),
        "type_gate_mean": gate_sum / max(gate_count, 1.0),
        "per_gold_type_actions": per_type,
        "type_prediction_sha256": digest.hexdigest(),
    }


def assert_frozen_baseline(metrics: dict[str, Any]) -> None:
    checks = {
        key: (
            math.isclose(float(metrics[key]), float(expected), abs_tol=1e-9)
            if isinstance(expected, float)
            else int(metrics[key]) == int(expected)
        )
        for key, expected in FROZEN_DEV.items()
    }
    if not all(checks.values()):
        raise RuntimeError(f"Frozen fixed-span baseline mismatch: {checks}")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed = int(config["runtime"]["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    train_payload = torch.load(resolve(config["data"]["train_cache"], root), map_location="cpu")
    dev_payload = torch.load(resolve(config["data"]["dev_cache"], root), map_location="cpu")
    if train_payload["split"] != "train" or dev_payload["split"] != "dev":
        raise ValueError("Train/Dev cache split contract is invalid.")
    if train_payload.get("test_accessed") or dev_payload.get("test_accessed"):
        raise ValueError("The sparse visual type probe cannot consume Test artifacts.")
    device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    model = SpanConditionedSparseVisualTypeRefiner(
        SparseVisualTypeConfig(**config["model"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["learning_rate"]),
        weight_decay=float(config["optim"]["weight_decay"]),
    )
    train_loader = DataLoader(
        RecordCacheDataset(train_payload["records"]),
        batch_size=int(config["optim"]["batch_size"]),
        shuffle=True,
        collate_fn=collate_records,
    )
    output_dir = resolve(config["runtime"]["output_dir"], root)
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_metrics = evaluate(
        model,
        dev_payload["records"],
        batch_size=int(config["optim"]["eval_batch_size"]),
        device=device,
    )
    assert_frozen_baseline(initial_metrics)
    if initial_metrics["mner_correct"] != initial_metrics["base_mner_correct"]:
        raise RuntimeError("Zero-initialized epoch 0 changed formal coarse types.")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": 0,
            "metrics": initial_metrics,
            "config": config,
        },
        output_dir / "best_model.pt",
    )
    best_metrics = initial_metrics
    best_epoch = 0
    history = [{"epoch": 0, "metrics": initial_metrics}]
    for epoch in range(1, int(config["optim"]["num_epochs"]) + 1):
        model.train()
        running = 0.0
        steps = 0
        for batch in train_loader:
            batch = move(batch, device)
            outputs = forward_model(model, batch)
            loss, _ = sparse_visual_type_loss(
                outputs,
                batch,
                **config["loss"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["optim"]["gradient_clip_norm"])
            )
            optimizer.step()
            running += float(loss.detach().item())
            steps += 1
        metrics = evaluate(
            model,
            dev_payload["records"],
            batch_size=int(config["optim"]["eval_batch_size"]),
            device=device,
        )
        history.append(
            {"epoch": epoch, "train_loss": running / max(steps, 1), "metrics": metrics}
        )
        if metrics["mner_f1"] > best_metrics["mner_f1"]:
            best_metrics = metrics
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                    "config": config,
                },
                output_dir / "best_model.pt",
            )
        print(json.dumps(history[-1], ensure_ascii=False))
    gate = {
        "net_correction_at_least_15": best_metrics["net_correction"] >= 15,
        "corrected_exceeds_damaged": best_metrics["corrected"] > best_metrics["damaged"],
        "base_correct_preservation_at_least_0_99": best_metrics[
            "base_correct_preservation"
        ] >= 0.99,
        "attention_entropy_below_0_88": best_metrics["attention_entropy_r16"] < 0.88,
        "mner_delta_at_least_0_004": best_metrics["mner_delta"] >= 0.004,
        "span_exact": math.isclose(
            best_metrics["span_f1"], FROZEN_DEV["span_f1"], abs_tol=1e-9
        ),
    }
    summary = {
        "kind": "span_sparse_visual_type_probe",
        "format_version": 1,
        "status": "GO" if all(gate.values()) else "NO_GO",
        "checkpoint_selected_only_by_dev_mner": True,
        "diagnostic_in_sample_train": True,
        "formal_span_source_sha256": dev_payload["summary"][
            "formal_prediction_sha256"
        ],
        "epoch0": initial_metrics,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "gate": gate,
        "gate_passed": all(gate.values()),
        "history": history,
        "test_accessed": False,
    }
    (output_dir / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
