#!/usr/bin/env python3
"""Materialize the frozen strict observable-tabular A1-T0 population."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.a1_feature_contract import strict_replacement_scope
from gmner.data.artifact_utils import sha256_file
from gmner.engine.a1_t0 import load_frozen_protocol
from gmner.models.a1_t0 import CLASS_ORDER, SOURCE_ORDER


LABEL_TO_ID = {"positive": 0, "neutral": 1, "damaging": 2}
SOURCE_TO_ID = {name: index for index, name in enumerate(SOURCE_ORDER)}
CONCEPTUAL_FEATURE_NAMES = (
    "candidate_source",
    "candidate_score",
    "base_candidate_margin",
    "boundary_distance",
    "candidate_type_id",
    "overlap_words_with_base",
    "overlaps_other_formal_count",
    "base_span",
    "candidate_span",
    "candidate_region_score",
    "base_type_id",
    "base_type_logits",
    "base_span_score",
    "base_is_null",
    "final_visible",
    "fine_region_logit",
    "base_region_is_null",
    "candidate_type_logits",
    "candidate_detector_score",
    "candidate_in_r16",
    "base_span_length",
    "candidate_span_length",
    "span_length_delta",
    "left_boundary_shift",
    "right_boundary_shift",
    "base_type_confidence",
    "base_type_margin",
    "base_type_entropy",
    "candidate_type_confidence",
    "candidate_type_margin",
    "candidate_type_entropy",
    "actions_in_base_group",
    "actions_from_same_source_in_group",
    "candidate_score_rank_in_group",
    "candidate_score_gap_to_group_best",
)
NUMERIC_FEATURE_NAMES = (
    "candidate_score",
    "base_candidate_margin",
    "boundary_distance",
    "candidate_type_id",
    "overlap_words_with_base",
    "overlaps_other_formal_count",
    "base_span_start",
    "base_span_end",
    "candidate_span_start",
    "candidate_span_end",
    "candidate_region_score",
    "base_type_id",
    "base_type_logit_LOC",
    "base_type_logit_PER",
    "base_type_logit_ORG",
    "base_type_logit_OTHER",
    "base_span_score",
    "base_is_null",
    "final_visible",
    "fine_region_logit",
    "base_region_is_null",
    "candidate_type_logit_LOC",
    "candidate_type_logit_PER",
    "candidate_type_logit_ORG",
    "candidate_type_logit_OTHER",
    "candidate_detector_score",
    "candidate_in_r16",
    "base_span_length",
    "candidate_span_length",
    "span_length_delta",
    "left_boundary_shift",
    "right_boundary_shift",
    "base_type_confidence",
    "base_type_margin",
    "base_type_entropy",
    "candidate_type_confidence",
    "candidate_type_margin",
    "candidate_type_entropy",
    "actions_in_base_group",
    "actions_from_same_source_in_group",
    "candidate_score_rank_in_group",
    "candidate_score_gap_to_group_best",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        default="docs/experiments/a1_t0_execution_authorization.json",
    )
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def softmax_summary(logits: list[float]) -> tuple[float, float, float]:
    maximum = max(logits)
    values = [math.exp(float(value) - maximum) for value in logits]
    denominator = sum(values)
    probabilities = [value / denominator for value in values]
    ranked = sorted(probabilities, reverse=True)
    entropy = -sum(value * math.log(max(value, 1e-30)) for value in probabilities)
    return ranked[0], ranked[0] - ranked[1], entropy


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    if authorization.get("status") != "AUTHORIZED":
        raise PermissionError("A1-T0 execution is not authorized.")
    for key in (
        "protocol_modification",
        "legacy_action_population",
        "latent_feature_rematerialization",
        "raw_text_or_token_access",
        "dev_access",
        "test_access",
    ):
        if authorization["forbidden"].get(key) is not True:
            raise PermissionError(f"A1-T0 execution lock is disabled: {key}")
    protocol = load_frozen_protocol(root, authorization)
    if protocol["status"] != "PREREGISTERED_TRAINING_NOT_AUTHORIZED":
        raise RuntimeError("A1-T0 preregistration status changed.")

    audit_path = root / "knowledge/final_chain_oof/a1_0/a1_feature_availability_audit.json"
    if sha256_file(audit_path) != authorization["feature_audit_sha256"]:
        raise RuntimeError("A1-0 feature audit changed.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["status"] != "PASS_OBSERVABLE_TABULAR_A1_T0_READY_FOR_SEPARATE_AUTHORIZATION":
        raise RuntimeError("A1-0 feature audit did not pass.")
    audited_features = {
        str(item["feature_name"])
        for item in audit["features"]
        if item.get("authorized_for_a1") is True
    }
    if audited_features != set(CONCEPTUAL_FEATURE_NAMES):
        raise RuntimeError("A1-T0 implementation does not match the 35-feature audit registry.")
    if len(CONCEPTUAL_FEATURE_NAMES) != int(protocol["feature_contract"]["total_model_inputs"]):
        raise RuntimeError("A1-T0 conceptual feature count changed.")
    rows_path = root / "knowledge/final_chain_oof/ten_fold_population/gold_free_rows.jsonl"
    sidecar_path = root / "knowledge/final_chain_oof/ten_fold_population/supervision_sidecar.jsonl"
    if sha256_file(rows_path) != protocol["evidence_contract"]["gold_free_rows_sha256"]:
        raise RuntimeError("Gold-free rows changed.")
    if sha256_file(sidecar_path) != protocol["evidence_contract"]["supervision_sidecar_sha256"]:
        raise RuntimeError("Supervision sidecar changed.")

    rows = read_jsonl(rows_path)
    sidecars = {str(item["record_id"]): item for item in read_jsonl(sidecar_path)}
    numeric_features: list[list[float]] = []
    source_ids: list[int] = []
    labels_out: list[int] = []
    metadata: list[dict[str, Any]] = []
    record_contract: dict[str, dict[str, Any]] = {}
    counts_by_fold = defaultdict(Counter)

    for row in rows:
        record_id = str(row["record_id"])
        fold_id = int(row["fold_id"])
        sidecar = sidecars[record_id]
        predictions = {item["prediction_id"]: item for item in row["formal_predictions"]}
        candidates = {item["candidate_id"]: item for item in row["r36_candidates"]["span_candidates"]}
        r16_ids = {item["candidate_id"] for item in row["r16_candidates"]["span_candidates"]}
        regions = {item["region_candidate_id"]: item for item in row["r36_candidates"]["region_candidates"]}
        supervision = {item["action_id"]: item for item in sidecar["a1_actions"]}
        strict_actions = [
            action
            for action in row["replacement_actions"]
            if strict_replacement_scope(action, predictions[action["base_prediction_id"]])
        ]
        group_counts = Counter(action["base_prediction_id"] for action in strict_actions)
        source_group_counts = Counter(
            (action["base_prediction_id"], action["candidate_source"])
            for action in strict_actions
        )
        groups = defaultdict(list)
        for action in strict_actions:
            groups[action["base_prediction_id"]].append(action)
        ranks: dict[str, tuple[int, float]] = {}
        for group in groups.values():
            ordered = sorted(
                group,
                key=lambda item: (-float(item["candidate_score"]), item["action_id"]),
            )
            best = float(ordered[0]["candidate_score"])
            for rank, action in enumerate(ordered, 1):
                ranks[action["action_id"]] = (
                    rank,
                    best - float(action["candidate_score"]),
                )

        record_contract[record_id] = {
            "fold_id": fold_id,
            "prediction_count": len(row["formal_predictions"]),
            "gold_count": int(sidecar["gold_entity_count"]),
            "base_mner_correct": int(sidecar["base_mner_correct_count"]),
            "formal_predictions": [
                {
                    "prediction_id": prediction["prediction_id"],
                    "span": [int(prediction["span"]["start"]), int(prediction["span"]["end"])],
                    "type_id": int(prediction["type_id"]),
                    "region_candidate_id": prediction["region_candidate_id"],
                }
                for prediction in row["formal_predictions"]
            ],
        }
        for action in strict_actions:
            action_id = str(action["action_id"])
            base = predictions[action["base_prediction_id"]]
            candidate = candidates[action["candidate_id"]]
            candidate_region_id = action["observable_features"]["candidate_region_candidate_id"]
            region = regions[candidate_region_id]
            label = supervision[action_id]
            if label["protected_label"] not in LABEL_TO_ID:
                raise RuntimeError("Unknown A1-T0 protected label.")
            base_span = [int(value) for value in action["observable_features"]["base_span"]]
            candidate_span = [int(value) for value in action["observable_features"]["candidate_span"]]
            base_logits = [float(value) for value in base["type_logits"]]
            candidate_logits = [float(value) for value in candidate["scores"]["type_logits"]]
            base_conf, base_margin, base_entropy = softmax_summary(base_logits)
            candidate_conf, candidate_margin, candidate_entropy = softmax_summary(candidate_logits)
            rank, score_gap = ranks[action_id]
            values = [
                float(action["candidate_score"]),
                float(action["base_candidate_margin"]),
                float(action["boundary_distance"]),
                float(action["candidate_type_id"]),
                float(action["conflict_features"]["overlap_words_with_base"]),
                float(action["conflict_features"]["overlaps_other_formal_count"]),
                float(base_span[0]),
                float(base_span[1]),
                float(candidate_span[0]),
                float(candidate_span[1]),
                float(action["observable_features"]["candidate_region_score"]),
                float(base["type_id"]),
                *base_logits,
                float(base["observable_features"]["span_base_score"]),
                float(base["observable_features"]["base_is_null"]),
                float(base["observable_features"]["final_visible"]),
                float(base["observable_features"]["fine_region_logit"]),
                float(base["region_is_null"]),
                *candidate_logits,
                float(region["detector_score"]),
                float(action["candidate_id"] in r16_ids),
                float(base_span[1] - base_span[0]),
                float(candidate_span[1] - candidate_span[0]),
                float((candidate_span[1] - candidate_span[0]) - (base_span[1] - base_span[0])),
                float(candidate_span[0] - base_span[0]),
                float(candidate_span[1] - base_span[1]),
                base_conf,
                base_margin,
                base_entropy,
                candidate_conf,
                candidate_margin,
                candidate_entropy,
                float(group_counts[action["base_prediction_id"]]),
                float(source_group_counts[(action["base_prediction_id"], action["candidate_source"])]),
                float(rank),
                float(score_gap),
            ]
            if len(values) != len(NUMERIC_FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
                raise RuntimeError("A1-T0 numeric feature contract failed.")
            numeric_features.append(values)
            source_ids.append(SOURCE_TO_ID[action["candidate_source"]])
            labels_out.append(LABEL_TO_ID[label["protected_label"]])
            metadata.append(
                {
                    "record_id": record_id,
                    "fold_id": fold_id,
                    "base_prediction_id": action["base_prediction_id"],
                    "action_id": action_id,
                    "candidate_id": action["candidate_id"],
                    "candidate_source": action["candidate_source"],
                    "candidate_score": float(action["candidate_score"]),
                    "candidate_span": candidate_span,
                    "candidate_type_id": int(action["candidate_type_id"]),
                    "candidate_region_candidate_id": candidate_region_id,
                    "protected_label": label["protected_label"],
                    "metric_outcome": label["metric_outcome"],
                    "mner_correct_delta": int(label["mner_correct_delta"]),
                    "lost_correct_mner_entities": label["lost_correct_mner_entities"],
                }
            )
            counts_by_fold[fold_id][label["protected_label"]] += 1

    numeric_tensor = torch.tensor(numeric_features, dtype=torch.float32)
    if not torch.isfinite(numeric_tensor).all():
        raise RuntimeError("A1-T0 dataset contains non-finite values.")
    if tuple(numeric_tensor.shape) != (31138, len(NUMERIC_FEATURE_NAMES)):
        raise RuntimeError(f"A1-T0 strict population changed: {tuple(numeric_tensor.shape)}")
    label_counts = Counter(labels_out)
    if [label_counts[index] for index in range(3)] != [286, 4128, 26724]:
        raise RuntimeError("A1-T0 label counts changed.")

    output_root = resolve(root, authorization["outputs"]["root"])
    manifest_path = output_root / "dataset/dataset_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fold_artifacts = []
    fold_ids_tensor = torch.tensor([item["fold_id"] for item in metadata], dtype=torch.long)
    source_tensor = torch.tensor(source_ids, dtype=torch.long)
    label_tensor = torch.tensor(labels_out, dtype=torch.long)
    for fold_id in range(10):
        indices = torch.nonzero(fold_ids_tensor.eq(fold_id), as_tuple=False).flatten()
        fold_metadata = [metadata[int(index)] for index in indices.tolist()]
        fold_record_ids = {item["record_id"] for item in fold_metadata}
        fold_path = output_root / f"dataset/fold{fold_id}.pt"
        torch.save({
            "kind": "a1_t0_strict_observable_tabular_dataset",
            "format_version": 1,
            "fold_id": fold_id,
            "class_order": CLASS_ORDER,
            "source_order": SOURCE_ORDER,
            "numeric_feature_names": NUMERIC_FEATURE_NAMES,
            "numeric_features": numeric_tensor[indices],
            "source_ids": source_tensor[indices],
            "labels": label_tensor[indices],
            "metadata": fold_metadata,
            "record_contract": {
                record_id: record_contract[record_id] for record_id in sorted(fold_record_ids)
            },
            "dev_accessed": False,
            "test_accessed": False,
        }, fold_path)
        fold_artifacts.append({
            "fold_id": fold_id,
            "path": str(fold_path.relative_to(root)).replace("\\", "/"),
            "sha256": sha256_file(fold_path),
            "actions": int(indices.numel()),
        })
    manifest = {
        "kind": "a1_t0_dataset_manifest",
        "format_version": 1,
        "status": "PASSED",
        "fold_artifacts": fold_artifacts,
        "actions": len(metadata),
        "base_prediction_groups": len({item["base_prediction_id"] for item in metadata}),
        "class_order": list(CLASS_ORDER),
        "source_order": list(SOURCE_ORDER),
        "conceptual_feature_names": list(CONCEPTUAL_FEATURE_NAMES),
        "conceptual_feature_count": len(CONCEPTUAL_FEATURE_NAMES),
        "numeric_feature_names": list(NUMERIC_FEATURE_NAMES),
        "numeric_feature_count": len(NUMERIC_FEATURE_NAMES),
        "expanded_model_input_count": len(NUMERIC_FEATURE_NAMES) + len(SOURCE_ORDER),
        "counts_by_fold": {str(fold): dict(counts_by_fold[fold]) for fold in range(10)},
        "gold_free_action_filter": True,
        "legacy_population_rejected": True,
        "dev_accessed": False,
        "test_accessed": False,
    }
    manifest_path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
