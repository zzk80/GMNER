#!/usr/bin/env python3
"""Replay TQ type scores on frozen legacy Stage1 spans using Dev only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from gmner.config import load_config
from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID
from gmner.data.frozen_clip_cache import (
    DVHRecordDataset,
    FrozenClipFeatureStore,
    sha256_file,
)
from gmner.data.graph_builders import GraphBuilderConfig, TextGraphBuilder
from gmner.data.record_level_stage1_dataset import RecordLevelStage1Dataset
from gmner.data.tokenization import load_word_aligned_tokenizer
from gmner.data.type_query_collator import TypeQueryRecordCollator
from gmner.engine.utils import f1_counts, move_batch_to_device
from gmner.models import GMNERModel
from gmner.models.tq_dv_mner import TQDualVisualMNER
from gmner.utils.metrics import extract_entities_from_word_labels
from scripts.train import build_datasets


FROZEN_BASELINE = {
    "prediction_count": 2516,
    "gold_count": 2450,
    "span_correct": 2162,
    "mner_correct": 2023,
    "span_f1": 0.8707209021345148,
    "mner_f1": 0.8147402335884012,
}
_ID2LABEL = {value: key for key, value in DEFAULT_LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--formal-config",
        default="configs/fmnerg_twitter10000_stage1.yaml",
    )
    parser.add_argument(
        "--formal-checkpoint",
        default="outputs/fmnerg_stage1_roberta128/best_model.pt",
    )
    parser.add_argument(
        "--tq-config",
        default="configs/tq_dv_mner/type_query_dual_visual_seed42.yaml",
    )
    parser.add_argument(
        "--tq-checkpoint",
        default="outputs/tq_dv_mner/type_query_dual_visual_seed42/best_model.pt",
    )
    parser.add_argument(
        "--output",
        default=(
            "outputs/tq_dv_mner/type_query_dual_visual_seed42/"
            "dev_fixed_span_type_replay.json"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def resolve(path: str, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _decoded_record_entities(
    decoded: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[int]]]]:
    """Convert a first-subword typed-BIO decode to padded record entities."""

    parsed: list[list[dict[str, Any]]] = []
    spans_by_record: list[list[list[int]]] = []
    for row, metadata in enumerate(batch["metadata"]):
        word_count = int(batch["word_count"][row].item())
        first_indices = batch["first_subword_indices"][row, :word_count]
        labels = [
            (
                int(decoded[row, position].item())
                if position >= 0
                else DEFAULT_LABEL2ID["O"]
            )
            for position in first_indices.tolist()
        ]
        tokens = list(metadata.get("tokens") or [])[:word_count]
        entities = extract_entities_from_word_labels(labels, tokens, _ID2LABEL)
        parsed.append(entities)
        spans_by_record.append(
            [
                [int(entity["start"]), int(entity["end"])]
                for entity in entities
            ]
        )

    max_entities = max((len(value) for value in parsed), default=0)
    masks = torch.zeros(
        decoded.size(0),
        max_entities,
        decoded.size(1),
        dtype=torch.bool,
        device=decoded.device,
    )
    type_ids = torch.full(
        (decoded.size(0), max_entities),
        ENTITY_TYPE2ID["O"],
        dtype=torch.long,
        device=decoded.device,
    )
    valid = torch.zeros(
        (decoded.size(0), max_entities),
        dtype=torch.bool,
        device=decoded.device,
    )
    subword_to_word = batch["subword_to_word"]
    for row, entities in enumerate(parsed):
        for entity_index, entity in enumerate(entities):
            start, end = int(entity["start"]), int(entity["end"])
            masks[row, entity_index] = (
                subword_to_word[row].ge(start)
                & subword_to_word[row].lt(end)
            )
            type_ids[row, entity_index] = ENTITY_TYPE2ID[str(entity["type"])]
            valid[row, entity_index] = bool(
                masks[row, entity_index].any().item()
            )
    return masks, type_ids, valid, spans_by_record


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    root = Path(__file__).resolve().parents[1]
    formal_config_path = resolve(args.formal_config, root)
    formal_checkpoint_path = resolve(args.formal_checkpoint, root)
    tq_config_path = resolve(args.tq_config, root)
    tq_checkpoint_path = resolve(args.tq_checkpoint, root)
    output_path = resolve(args.output, root)
    formal_config = load_config(formal_config_path)
    tq_config = load_config(tq_config_path)
    if str(tq_config.data.test_file).strip():
        raise ValueError("TQ replay requires the Test-locked TQ configuration.")
    if formal_config.model.text_model_name != tq_config.model.text_model_name:
        raise ValueError("Formal and TQ tokenizers must be identical for span replay.")

    device = torch.device(args.device)
    tokenizer = load_word_aligned_tokenizer(formal_config.model.text_model_name)
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=formal_config.data.use_dependency_graph,
            dependency_backend=formal_config.data.dependency_backend,
            dependency_model=formal_config.data.dependency_model,
            window_size=formal_config.data.graph_window_size,
        )
    )
    _, expanded_dev, _, num_labels = build_datasets(
        config=formal_config,
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        project_root=root,
        output_dir=output_path.parent / "replay_converted",
        build_test=False,
    )
    records = RecordLevelStage1Dataset(expanded_dev, split="dev")
    clip_store = FrozenClipFeatureStore(
        resolve(tq_config.data.frozen_clip_feature_dir, root) / "dev",
        expected_split="dev",
        expected_kind=tq_config.data.frozen_clip_cache_kind,
    )
    dataset = DVHRecordDataset(records, clip_store, shuffle_clip=False)
    collator = TypeQueryRecordCollator(
        tokenizer,
        max_length=int(tq_config.data.max_length),
        max_span_length=int(tq_config.model.tq_max_span_length),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )

    formal_model = GMNERModel(config=formal_config, num_labels=num_labels)
    formal_checkpoint = torch.load(formal_checkpoint_path, map_location="cpu")
    formal_incompatible = formal_model.load_state_dict(
        formal_checkpoint["model_state_dict"], strict=False
    )
    tq_model = TQDualVisualMNER(tq_config)
    tq_checkpoint = torch.load(tq_checkpoint_path, map_location="cpu")
    tq_model.load_state_dict(tq_checkpoint["model_state_dict"], strict=True)
    formal_model.to(device).eval()
    tq_model.to(device).eval()
    tq_model.set_visual_enabled(True)

    predicted_count = 0
    gold_count = 0
    span_correct = 0
    baseline_mner_correct = 0
    replay_mner_correct = 0
    type_changed = 0
    corrected = 0
    damaged = 0
    unavailable = 0
    baseline_digest = hashlib.sha256()
    replay_digest = hashlib.sha256()

    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        legacy_batch = {
            key: batch[key]
            for key in (
                "input_ids",
                "attention_mask",
                "adjacency",
                "region_features",
                "region_boxes",
                "region_mask",
                "region_scores",
                "metadata",
            )
        }
        if "token_type_ids" in batch:
            legacy_batch["token_type_ids"] = batch["token_type_ids"]
        formal_outputs = formal_model(legacy_batch)
        formal_decoded = formal_model.ner_head.decode(
            formal_outputs["ner_logits"],
            batch["attention_mask"],
            valid_mask=batch["legacy_ner_labels"].ne(-100),
        )
        _, formal_types, formal_valid, formal_spans = _decoded_record_entities(
            formal_decoded, batch
        )
        tq_outputs = tq_model(batch)

        for row, metadata in enumerate(batch["metadata"]):
            gold = _gold_by_span(batch, row)
            baseline_predictions: list[tuple[int, int, int]] = []
            replay_predictions: list[tuple[int, int, int]] = []
            for entity_index, span in enumerate(formal_spans[row]):
                if not bool(formal_valid[row, entity_index].item()):
                    continue
                start, end = int(span[0]), int(span[1])
                baseline_type = int(formal_types[row, entity_index].item())
                baseline_predictions.append((start, end, baseline_type))
                replay_type = _replay_type(
                    outputs=tq_outputs,
                    batch=batch,
                    row=row,
                    start=start,
                    end=end,
                    fallback_type=baseline_type,
                    existence_weight=float(
                        tq_config.model.tq_existence_score_weight
                    ),
                )
                unavailable += int(replay_type is None)
                if replay_type is None:
                    replay_type = baseline_type
                replay_predictions.append((start, end, replay_type))
                type_changed += int(replay_type != baseline_type)
                gold_type = gold.get((start, end))
                baseline_correct = gold_type == baseline_type
                replay_correct = gold_type == replay_type
                corrected += int(not baseline_correct and replay_correct)
                damaged += int(baseline_correct and not replay_correct)

            baseline_set = set(baseline_predictions)
            replay_set = set(replay_predictions)
            gold_set = {
                (start, end, type_id)
                for (start, end), type_id in gold.items()
            }
            predicted_spans = {(start, end) for start, end, _ in baseline_set}
            span_correct += len(predicted_spans & set(gold))
            baseline_mner_correct += len(baseline_set & gold_set)
            replay_mner_correct += len(replay_set & gold_set)
            predicted_count += len(baseline_set)
            gold_count += len(gold_set)
            record_id = str(metadata.get("record_id", ""))
            _update_digest(baseline_digest, record_id, baseline_set)
            _update_digest(replay_digest, record_id, replay_set)

    baseline = _metrics(
        span_correct=span_correct,
        mner_correct=baseline_mner_correct,
        predicted=predicted_count,
        gold=gold_count,
    )
    replay = _metrics(
        span_correct=span_correct,
        mner_correct=replay_mner_correct,
        predicted=predicted_count,
        gold=gold_count,
    )
    baseline_checks = {
        key: (
            abs(float(baseline[key]) - float(value)) < 1e-9
            if isinstance(value, float)
            else int(baseline[key]) == int(value)
        )
        for key, value in FROZEN_BASELINE.items()
    }
    if not all(baseline_checks.values()):
        raise RuntimeError(
            f"Frozen Stage1 baseline replay mismatch: {baseline_checks}"
        )
    report = {
        "kind": "tq_fixed_span_type_replay",
        "format_version": 1,
        "scope": "dev",
        "score_formula": (
            "start_logit + end_logit + span_match + "
            "0.5 * log_sigmoid(type_existence_logit)"
        ),
        "span_source": "frozen_formal_stage1_typed_bio_decode",
        "boundary_changed": False,
        "threshold_scanned": False,
        "baseline": baseline,
        "replay": replay,
        "delta": {
            "mner_f1": replay["mner_f1"] - baseline["mner_f1"],
            "mner_correct": replay_mner_correct - baseline_mner_correct,
        },
        "actions": {
            "type_changed": type_changed,
            "corrected": corrected,
            "damaged": damaged,
            "net_correction": corrected - damaged,
            "unavailable_fallback": unavailable,
        },
        "checks": {
            "frozen_baseline_reproduced": all(baseline_checks.values()),
            "prediction_count_preserved": (
                int(replay["prediction_count"])
                == int(baseline["prediction_count"])
            ),
            "span_metrics_preserved": replay["span_f1"] == baseline["span_f1"],
            "test_accessed_false": True,
        },
        "baseline_checks": baseline_checks,
        "baseline_prediction_sha256": baseline_digest.hexdigest(),
        "replay_prediction_sha256": replay_digest.hexdigest(),
        "formal_load": {
            "missing_keys": list(formal_incompatible.missing_keys),
            "unexpected_keys": list(formal_incompatible.unexpected_keys),
        },
        "artifacts": {
            "formal_config_sha256": sha256_file(formal_config_path),
            "formal_checkpoint_sha256": sha256_file(formal_checkpoint_path),
            "tq_config_sha256": sha256_file(tq_config_path),
            "tq_checkpoint_sha256": sha256_file(tq_checkpoint_path),
            "clip_dev_manifest_sha256": sha256_file(clip_store.manifest_path),
        },
        "test_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _replay_type(
    *,
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    row: int,
    start: int,
    end: int,
    fallback_type: int,
    existence_weight: float,
) -> int | None:
    end_index = end - 1
    if start < 0 or end_index < start:
        return None
    if end_index >= int(outputs["start_logits"].size(-1)):
        return None
    valid = batch["query_span_valid_mask"][row, :, start, end_index].bool()
    if not valid.any():
        return None
    scores = (
        outputs["start_logits"][row, :, start]
        + outputs["end_logits"][row, :, end_index]
        + outputs["span_logits"][row, :, start, end_index]
        + existence_weight * F.logsigmoid(outputs["existence_logits"][row])
    ).masked_fill(~valid, -1e4)
    selected = int(scores.argmax().item())
    return selected if bool(valid[selected].item()) else fallback_type


def _gold_by_span(batch: dict[str, Any], row: int) -> dict[tuple[int, int], int]:
    indices = torch.nonzero(
        batch["gold_entity_mask"][row].bool(), as_tuple=False
    ).squeeze(-1)
    return {
        (
            int(batch["gold_spans"][row, index, 0].item()),
            int(batch["gold_spans"][row, index, 1].item()),
        ): int(batch["gold_type_ids"][row, index].item())
        for index in indices.tolist()
    }


def _metrics(
    *, span_correct: int, mner_correct: int, predicted: int, gold: int
) -> dict[str, float]:
    span_p, span_r, span_f1 = f1_counts(span_correct, predicted, gold)
    mner_p, mner_r, mner_f1 = f1_counts(mner_correct, predicted, gold)
    return {
        "prediction_count": float(predicted),
        "gold_count": float(gold),
        "span_precision": span_p,
        "span_recall": span_r,
        "span_f1": span_f1,
        "span_correct": float(span_correct),
        "mner_precision": mner_p,
        "mner_recall": mner_r,
        "mner_f1": mner_f1,
        "mner_correct": float(mner_correct),
    }


def _update_digest(
    digest: Any, record_id: str, predictions: set[tuple[int, int, int]]
) -> None:
    digest.update(
        json.dumps(
            {"record_id": record_id, "predictions": sorted(predictions)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


if __name__ == "__main__":
    main()
