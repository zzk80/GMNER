"""Evaluate the PA1 Phase 1 Dev Gate against the frozen Stage1 model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_protected_region_mner_equivalence import build_dev_loader, resolve
from gmner.config import load_config
from gmner.constants import DEFAULT_LABEL2ID, IGNORE_INDEX
from gmner.engine import evaluate_model
from gmner.engine.utils import move_batch_to_device
from gmner.models import GMNERModel
from gmner.utils.metrics import (
    extract_entities_from_word_labels,
    word_labels_from_subwords,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/fmnerg_twitter10000_stage1.yaml",
    )
    parser.add_argument(
        "--protected-config",
        default="configs/protected_region_mner/pa1_phase1_seed42.yaml",
    )
    parser.add_argument(
        "--base-checkpoint",
        default="outputs/fmnerg_stage1_roberta128/best_model.pt",
    )
    parser.add_argument(
        "--protected-checkpoint",
        default="outputs/protected_region_mner/pa1_phase1_seed42/best_model.pt",
    )
    parser.add_argument(
        "--output",
        default="outputs/protected_region_mner/pa1_phase1_seed42/dev_gate.json",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_model(config, checkpoint_path: Path, device: torch.device) -> GMNERModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = GMNERModel(config, num_labels=9)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def entity_key(entity: dict) -> tuple[int, int, str]:
    return int(entity["start"]), int(entity["end"]), str(entity["type"])


@torch.no_grad()
def compare_mner_predictions(
    *,
    base_model: GMNERModel,
    protected_model: GMNERModel,
    loader,
    device: torch.device,
) -> dict[str, object]:
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    seen: set[str] = set()
    corrected = 0
    damaged = 0
    type_corrected = 0
    type_damaged = 0
    baseline_correct = 0
    baseline_correct_preserved = 0
    per_type = defaultdict(lambda: {"corrected": 0, "damaged": 0})
    projection_gate_sum = 0.0
    projection_gate_count = 0
    entity_gate_sum = 0.0
    entity_gate_count = 0
    non_entity_gate_sum = 0.0
    non_entity_gate_count = 0
    entropy_sum = 0.0
    entropy_count = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        base_outputs = base_model(batch)
        protected_outputs = protected_model(batch)
        labels = batch["ner_labels"]
        valid = labels.ne(IGNORE_INDEX)
        base_tokens = base_model.ner_head.decode(
            base_outputs["ner_logits"],
            batch["attention_mask"],
            valid_mask=valid,
        )
        protected_tokens = protected_model.ner_head.decode(
            protected_outputs["ner_logits"],
            batch["attention_mask"],
            valid_mask=valid,
        )

        token_gate = protected_outputs["protected_token_gate"]
        entity_mask = valid & labels.ne(DEFAULT_LABEL2ID["O"])
        non_entity_mask = valid & labels.eq(DEFAULT_LABEL2ID["O"])
        entity_gate_sum += float(token_gate[entity_mask].sum().item())
        entity_gate_count += int(entity_mask.sum().item())
        non_entity_gate_sum += float(token_gate[non_entity_mask].sum().item())
        non_entity_gate_count += int(non_entity_mask.sum().item())
        real_mask = protected_outputs["protected_real_region_mask"]
        projection_gate = protected_outputs["protected_region_gate"]
        projection_gate_sum += float(projection_gate[real_mask].sum().item())
        projection_gate_count += int(real_mask.sum().item())
        attention_entropy = protected_outputs["protected_attention_entropy"]
        entropy_sum += float(attention_entropy[valid].sum().item())
        entropy_count += int(valid.sum().item())

        metadata = batch.get("metadata", [])
        for index, meta in enumerate(metadata):
            record_id = str(meta.get("sample_id"))
            if record_id in seen:
                continue
            seen.add(record_id)
            tokens = meta.get("tokens") or []
            word_ids = meta.get("word_ids") or []
            base_words = word_labels_from_subwords(base_tokens[index].tolist(), word_ids)
            protected_words = word_labels_from_subwords(
                protected_tokens[index].tolist(), word_ids
            )
            gold_words = word_labels_from_subwords(labels[index].tolist(), word_ids)
            base_entities = {
                entity_key(item)
                for item in extract_entities_from_word_labels(base_words, tokens, id2label)
            }
            protected_entities = {
                entity_key(item)
                for item in extract_entities_from_word_labels(
                    protected_words, tokens, id2label
                )
            }
            gold_entities = {
                entity_key(item)
                for item in extract_entities_from_word_labels(gold_words, tokens, id2label)
            }
            base_correct = base_entities & gold_entities
            protected_correct = protected_entities & gold_entities
            recovered = protected_correct - base_correct
            lost = base_correct - protected_correct
            corrected += len(recovered)
            damaged += len(lost)
            baseline_correct += len(base_correct)
            baseline_correct_preserved += len(base_correct & protected_entities)
            for item in recovered:
                per_type[item[2]]["corrected"] += 1
            for item in lost:
                per_type[item[2]]["damaged"] += 1
            base_by_span = {(item[0], item[1]): item[2] for item in base_entities}
            protected_by_span = {
                (item[0], item[1]): item[2] for item in protected_entities
            }
            for start, end, gold_type in gold_entities:
                span = (start, end)
                base_type = base_by_span.get(span)
                protected_type = protected_by_span.get(span)
                if base_type != gold_type and protected_type == gold_type:
                    type_corrected += 1
                elif base_type == gold_type and protected_type not in {None, gold_type}:
                    type_damaged += 1

    return {
        "records": len(seen),
        "mner_corrected": corrected,
        "mner_damaged": damaged,
        "mner_net": corrected - damaged,
        "type_corrected": type_corrected,
        "type_damaged": type_damaged,
        "type_net": type_corrected - type_damaged,
        "formal_correct_count": baseline_correct,
        "formal_correct_preserved": baseline_correct_preserved,
        "formal_correct_preservation_rate": (
            baseline_correct_preserved / max(baseline_correct, 1)
        ),
        "per_type_mner": dict(sorted(per_type.items())),
        "mean_projection_gate": projection_gate_sum / max(projection_gate_count, 1),
        "mean_entity_token_gate": entity_gate_sum / max(entity_gate_count, 1),
        "mean_non_entity_token_gate": non_entity_gate_sum / max(
            non_entity_gate_count, 1
        ),
        "mean_attention_entropy": entropy_sum / max(entropy_count, 1),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    base_config = load_config(args.base_config)
    protected_config = load_config(args.protected_config)
    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    base_model = load_model(base_config, resolve(args.base_checkpoint, root), device)
    protected_model = load_model(
        protected_config,
        resolve(args.protected_checkpoint, root),
        device,
    )
    loader = build_dev_loader(protected_config, root)
    action_metrics = compare_mner_predictions(
        base_model=base_model,
        protected_model=protected_model,
        loader=loader,
        device=device,
    )
    base_metrics = evaluate_model(base_model, loader, device)
    protected_metrics = evaluate_model(protected_model, loader, device)

    deltas = {
        "mner_f1": protected_metrics["entity_f1"] - base_metrics["entity_f1"],
        "span_f1": protected_metrics["span_f1"] - base_metrics["span_f1"],
        "eeg_f1": protected_metrics["eeg_f1"] - base_metrics["eeg_f1"],
        "gmner_f1": protected_metrics["gmner_score"] - base_metrics["gmner_score"],
    }
    checks = {
        "mner_delta_at_least_0.004": deltas["mner_f1"] >= 0.004,
        "span_delta_at_least_minus_0.001": deltas["span_f1"] >= -0.001,
        "eeg_delta_at_least_minus_0.002": deltas["eeg_f1"] >= -0.002,
        "gmner_delta_at_least_minus_0.002": deltas["gmner_f1"] >= -0.002,
        "formal_correct_preservation_at_least_0.99": action_metrics[
            "formal_correct_preservation_rate"
        ]
        >= 0.99,
        "corrected_exceeds_damaged": action_metrics["type_corrected"]
        > action_metrics["type_damaged"],
        "test_accessed_false": True,
    }
    report = {
        "kind": "protected_region_mner_pa1_phase1_dev_gate",
        "baseline": {
            key: base_metrics[key]
            for key in ("span_f1", "entity_f1", "eeg_f1", "gmner_score")
        },
        "protected": {
            key: protected_metrics[key]
            for key in ("span_f1", "entity_f1", "eeg_f1", "gmner_score")
        },
        "deltas": deltas,
        "diagnostics": action_metrics,
        "checks": checks,
        "gate_passed": all(checks.values()),
        "test_accessed": False,
    }
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
