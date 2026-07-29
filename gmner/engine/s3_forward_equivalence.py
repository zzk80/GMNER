"""S3.0 eval-only equivalence checks against the frozen legacy Stage1."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import torch
from torchvision.ops import box_iou

from gmner.constants import DEFAULT_LABEL2ID, ENTITY_TYPE2ID
from gmner.engine.utils import f1_counts
from gmner.knowledge.region_compatibility import compatibility_score
from gmner.models.stage1 import LegacyStage1RecordWrapper
from gmner.utils.metrics import extract_entities_from_word_labels


_ID2LABEL = {value: key for key, value in DEFAULT_LABEL2ID.items()}
_METRICS = ("span", "mner", "eeg", "gmner")


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.shape != right.shape:
        return float("inf")
    if left.numel() == 0:
        return 0.0
    return float((left.float() - right.float()).abs().max().item())


def _prediction_key(item: dict[str, Any]) -> tuple:
    return (
        tuple(int(value) for value in item["span"]),
        int(item["type_id"]),
        int(item["region_index"]),
    )


def _update_digest(
    digest: Any,
    record_id: str,
    predictions: list[dict[str, Any]],
) -> None:
    payload = {
        "record_id": str(record_id),
        "predictions": [
            {
                "span": list(key[0]),
                "type_id": key[1],
                "region_index": key[2],
            }
            for key in sorted(_prediction_key(item) for item in predictions)
        ],
    }
    digest.update(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(b"\n")


def _decoded_record_entities(
    decoded: torch.Tensor,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[list[int]]]]:
    """Convert legacy first-subword BIO decode to padded record entities."""

    parsed: list[list[dict[str, Any]]] = []
    spans_by_record: list[list[list[int]]] = []
    for row, metadata in enumerate(batch["metadata"]):
        word_count = int(batch["word_count"][row].item())
        first_indices = batch["first_subword_indices"][row, :word_count]
        labels = []
        for position in first_indices.tolist():
            labels.append(
                int(decoded[row, position].item())
                if position >= 0
                else DEFAULT_LABEL2ID["O"]
            )
        tokens = list(metadata.get("tokens") or [])[:word_count]
        entities = extract_entities_from_word_labels(
            labels,
            tokens,
            _ID2LABEL,
        )
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
            type_ids[row, entity_index] = ENTITY_TYPE2ID[
                str(entity["type"])
            ]
            valid[row, entity_index] = bool(
                masks[row, entity_index].any().item()
            )
    return masks, type_ids, valid, spans_by_record


def _legacy_scalar_scores(
    teacher: torch.nn.Module,
    *,
    fused_tokens: torch.Tensor,
    image_nodes: torch.Tensor,
    entity_masks: torch.Tensor,
    entity_type_ids: torch.Tensor,
    entity_valid: torch.Tensor,
    null_priors: torch.Tensor,
    batch: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Expose each original single-entity grounding stage."""

    batch_size, entity_count, _ = entity_masks.shape
    region_count = image_nodes.size(1)
    stage_names = (
        "raw_logits",
        "after_entity_null_prior",
        "after_global_null_bias",
        "after_detector_prior",
        "after_compatibility_prior",
        "formal_logits",
    )
    stages = {
        name: torch.full(
            (batch_size, entity_count, region_count),
            -1e4,
            dtype=image_nodes.dtype,
            device=image_nodes.device,
        )
        for name in stage_names
    }
    config = teacher.config
    null_weight = float(
        getattr(config.model, "grounding_null_prior_weight", 0.0)
    )
    null_bias = float(
        getattr(config.model, "grounding_null_logit_bias", 0.0)
    )
    detector_weight = float(
        getattr(config.model, "region_score_prior_weight", 0.0)
    )
    compatibility_weight = float(
        getattr(
            config.model,
            "region_object_compatibility_weight",
            0.0,
        )
    )
    for row in range(batch_size):
        for entity_index in range(entity_count):
            if not bool(entity_valid[row, entity_index].item()):
                continue
            mask = entity_masks[
                row : row + 1, entity_index
            ].to(fused_tokens.dtype)
            query = (
                fused_tokens[row : row + 1] * mask.unsqueeze(-1)
            ).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            current_raw = teacher.grounding_head(
                query=query,
                image_nodes=image_nodes[row : row + 1],
                image_mask=batch["region_mask"][row : row + 1],
            )
            current = current_raw.clone()
            stages["raw_logits"][row, entity_index] = current_raw[0]
            if null_weight:
                prior = null_priors[row, entity_index].to(
                    current
                ).clamp(1e-4, 1.0 - 1e-4)
                current[:, -1] += (
                    torch.log(prior / (1.0 - prior)) * null_weight
                )
            stages["after_entity_null_prior"][
                row, entity_index
            ] = current[0]
            if null_bias:
                current[:, -1] += null_bias
            stages["after_global_null_bias"][
                row, entity_index
            ] = current[0]
            if detector_weight:
                scores = batch["region_scores"][
                    row : row + 1
                ].to(current).clamp(1e-4, 1.0)
                detector = scores.log() * detector_weight
                valid_detector = batch["region_mask"][
                    row : row + 1
                ].bool().clone()
                valid_detector[:, -1] = False
                current = current + detector.masked_fill(
                    ~valid_detector,
                    0.0,
                )
            stages["after_detector_prior"][
                row, entity_index
            ] = current[0]
            if compatibility_weight:
                metadata = batch["metadata"][row]
                labels = list(
                    metadata.get("region_object_labels") or []
                )
                attributes = list(
                    metadata.get("region_object_attributes") or []
                )
                compatibility = torch.zeros_like(current)
                region_limit = min(len(labels), region_count)
                if region_limit == region_count:
                    region_limit -= 1
                for region_index in range(max(region_limit, 0)):
                    attribute = (
                        attributes[region_index]
                        if region_index < len(attributes)
                        else ""
                    )
                    compatibility[0, region_index] = compatibility_score(
                        int(
                            entity_type_ids[
                                row, entity_index
                            ].item()
                        ),
                        labels[region_index],
                        attribute,
                    )
                current = current + (
                    compatibility * compatibility_weight
                )
            stages["after_compatibility_prior"][
                row, entity_index
            ] = current[0]
            current = current.masked_fill(
                ~batch["region_mask"][row : row + 1].bool(),
                -1e4,
            )
            mini_batch = {
                "grounding_null_prior": null_priors[
                    row : row + 1, entity_index
                ],
                "region_scores": batch["region_scores"][row : row + 1],
                "metadata": [batch["metadata"][row]],
            }
            current_formal = teacher._apply_grounding_knowledge(
                logits=current_raw,
                image_nodes=image_nodes[row : row + 1],
                image_mask=batch["region_mask"][row : row + 1],
                batch=mini_batch,
                target_type_ids=entity_type_ids[
                    row : row + 1, entity_index
                ],
            )
            if not torch.allclose(
                current,
                current_formal,
                atol=1e-6,
                rtol=0.0,
            ):
                raise AssertionError(
                    "Scalar stage exposure differs from the legacy "
                    "_apply_grounding_knowledge result."
                )
            stages["formal_logits"][row, entity_index] = current_formal[0]
    return stages


def _gold_for_record(
    batch: dict[str, Any],
    row: int,
) -> list[dict[str, Any]]:
    gold = []
    tokens = list(batch["metadata"][row].get("tokens") or [])
    entity_indices = torch.nonzero(
        batch["gold_entity_mask"][row],
        as_tuple=False,
    ).squeeze(-1)
    for entity_index in entity_indices.tolist():
        span = batch["gold_spans"][row, entity_index].tolist()
        positive = torch.nonzero(
            batch["gold_region_positive_mask"][row, entity_index],
            as_tuple=False,
        ).squeeze(-1)
        gold.append(
            {
                "span": span,
                "type_id": int(
                    batch["gold_type_ids"][row, entity_index].item()
                ),
                "text": " ".join(tokens[int(span[0]) : int(span[1])]),
                "region_positive_indices": positive.tolist(),
            }
        )
    return gold


def _formal_region_matches(
    *,
    region_index: int,
    gold_entity: dict[str, Any],
    metadata: dict[str, Any],
    region_boxes: torch.Tensor,
    null_region_index: int,
) -> bool:
    """Apply the frozen Stage1 paper-metric region rule exactly."""

    gold_name = str(gold_entity.get("text", "")).strip().lower()
    gt_boxes = (metadata.get("gt_boxes_by_name") or {}).get(
        gold_name,
        [],
    )
    if not gt_boxes:
        return int(region_index) == int(null_region_index)
    if int(region_index) == int(null_region_index):
        return False
    predicted_box = region_boxes[int(region_index)].unsqueeze(0)
    targets = torch.tensor(
        gt_boxes,
        dtype=predicted_box.dtype,
        device=predicted_box.device,
    )
    ious = box_iou(targets, predicted_box).squeeze(1)
    return bool((ious > 0.5).any().item())


def _match_formal_record_predictions(
    *,
    predictions: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    metadata: dict[str, Any],
    region_boxes: torch.Tensor,
    null_region_index: int,
) -> dict[str, set[int]]:
    """Match predictions with the same stopping rules as evaluator.py."""

    matched = {name: set() for name in _METRICS}
    for prediction in predictions:
        pred_span = tuple(prediction["span"])
        pred_type = int(prediction["type_id"])
        pred_region = int(prediction["region_index"])

        for gold_index, target in enumerate(gold):
            if gold_index in matched["span"]:
                continue
            if tuple(target["span"]) == pred_span:
                matched["span"].add(gold_index)
                break

        for gold_index, target in enumerate(gold):
            if gold_index in matched["mner"]:
                continue
            if (
                tuple(target["span"]) == pred_span
                and int(target["type_id"]) == pred_type
            ):
                matched["mner"].add(gold_index)
                break

        for gold_index, target in enumerate(gold):
            if gold_index in matched["eeg"]:
                continue
            if tuple(target["span"]) != pred_span:
                continue
            if _formal_region_matches(
                region_index=pred_region,
                gold_entity=target,
                metadata=metadata,
                region_boxes=region_boxes,
                null_region_index=null_region_index,
            ):
                matched["eeg"].add(gold_index)
            break

        for gold_index, target in enumerate(gold):
            if gold_index in matched["gmner"]:
                continue
            if (
                tuple(target["span"]) != pred_span
                or int(target["type_id"]) != pred_type
            ):
                continue
            if _formal_region_matches(
                region_index=pred_region,
                gold_entity=target,
                metadata=metadata,
                region_boxes=region_boxes,
                null_region_index=null_region_index,
            ):
                matched["gmner"].add(gold_index)
            break
    return matched


def _metric_report(
    correct: dict[str, int],
    predicted: int,
    gold: int,
) -> dict[str, float]:
    output: dict[str, float] = {
        "prediction_count": float(predicted),
        "gold_count": float(gold),
    }
    for name in _METRICS:
        precision, recall, score = f1_counts(
            correct[name],
            predicted,
            gold,
        )
        output[f"{name}_precision"] = precision
        output[f"{name}_recall"] = recall
        output[f"{name}_f1"] = score
        output[f"{name}_correct"] = float(correct[name])
    return output


@torch.no_grad()
def evaluate_s3_forward_equivalence(
    *,
    teacher: torch.nn.Module,
    wrapper: LegacyStage1RecordWrapper,
    dataloader: Any,
    device: torch.device,
    emission_tolerance: float = 1e-6,
    grounding_tolerance: float = 1e-5,
    expected_baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare old scalar and new record-level paths on fixed Dev records."""

    teacher.eval()
    wrapper.eval()
    state_errors = {
        "base_text_nodes": 0.0,
        "text_graph_nodes": 0.0,
        "pre_prototype_fused_tokens": 0.0,
        "image_nodes": 0.0,
        "fused_global": 0.0,
        "alignment_score": 0.0,
    }
    grounding_errors = {
        "raw_logits": 0.0,
        "after_entity_null_prior": 0.0,
        "after_global_null_bias": 0.0,
        "after_detector_prior": 0.0,
        "after_compatibility_prior": 0.0,
        "formal_logits": 0.0,
    }
    emission_error = 0.0
    decoded_equal = True
    grounding_argmax_equal = True
    null_visible_equal = True
    positive_set_correctness_equal = True
    prediction_set_equal = True
    null_index_aligned = True
    encoded_record_count = 0
    record_count = 0
    predicted_count = 0
    gold_count = 0
    correct = {name: 0 for name in _METRICS}
    old_digest = hashlib.sha256()
    new_digest = hashlib.sha256()

    for raw_batch in dataloader:
        batch = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in raw_batch.items()
        }
        new_outputs = wrapper(batch)
        encoded_record_count += int(batch["input_ids"].size(0))
        null_index_aligned &= bool(
            batch["null_region_index"].eq(
                batch["region_mask"].size(1) - 1
            ).all().item()
        )
        legacy_batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "adjacency": batch["adjacency"],
            "region_features": batch["region_features"],
            "region_boxes": batch["region_boxes"],
            "region_mask": batch["region_mask"],
            "region_scores": batch["region_scores"],
            "metadata": batch["metadata"],
        }
        if "token_type_ids" in batch:
            legacy_batch["token_type_ids"] = batch["token_type_ids"]
        legacy_outputs = teacher(legacy_batch)
        legacy_decoded = teacher.ner_head.decode(
            legacy_outputs["ner_logits"],
            batch["attention_mask"],
            valid_mask=batch["legacy_ner_labels"].ne(-100),
        )
        emission_error = max(
            emission_error,
            _max_abs(
                legacy_outputs["ner_logits"],
                new_outputs["ner_logits"],
            ),
        )
        decoded_equal &= bool(
            torch.equal(legacy_decoded, new_outputs["decoded_tags"])
        )
        for state_name in state_errors:
            state_errors[state_name] = max(
                state_errors[state_name],
                _max_abs(
                    legacy_outputs[state_name],
                    new_outputs[state_name],
                ),
            )

        gold_valid = batch["grounding_entity_mask"].bool()
        old_gold_stages = _legacy_scalar_scores(
            teacher,
            fused_tokens=legacy_outputs[
                "pre_prototype_fused_tokens"
            ],
            image_nodes=legacy_outputs["image_nodes"],
            entity_masks=batch["gold_subword_masks"],
            entity_type_ids=batch["gold_type_ids"],
            entity_valid=gold_valid,
            null_priors=batch["grounding_null_prior"],
            batch=batch,
        )
        if gold_valid.any():
            for stage_name in grounding_errors:
                grounding_errors[stage_name] = max(
                    grounding_errors[stage_name],
                    _max_abs(
                        old_gold_stages[stage_name][gold_valid],
                        new_outputs[f"grounding_{stage_name}"][
                            gold_valid
                        ],
                    ),
                )
            old_gold_formal = old_gold_stages["formal_logits"]
            new_gold_formal = new_outputs["grounding_formal_logits"]
            grounding_argmax_equal &= bool(
                torch.equal(
                    old_gold_formal[gold_valid].argmax(dim=-1),
                    new_gold_formal[gold_valid].argmax(dim=-1),
                )
            )
            rows, entities = torch.nonzero(
                gold_valid,
                as_tuple=True,
            )
            old_indices = old_gold_formal[gold_valid].argmax(dim=-1)
            new_indices = new_gold_formal[gold_valid].argmax(dim=-1)
            old_null = old_indices.eq(
                batch["null_region_index"][rows]
            )
            new_null = new_indices.eq(
                batch["null_region_index"][rows]
            )
            null_visible_equal &= bool(torch.equal(old_null, new_null))
            positives = batch["gold_region_positive_mask"][
                rows, entities
            ]
            old_positive = positives.gather(
                1, old_indices.unsqueeze(1)
            ).squeeze(1)
            new_positive = positives.gather(
                1, new_indices.unsqueeze(1)
            ).squeeze(1)
            positive_set_correctness_equal &= bool(
                torch.equal(old_positive, new_positive)
            )

        (
            predicted_masks,
            predicted_type_ids,
            predicted_valid,
            predicted_spans,
        ) = _decoded_record_entities(new_outputs["decoded_tags"], batch)
        neutral_priors = torch.full(
            predicted_type_ids.shape,
            0.5,
            dtype=new_outputs["image_nodes"].dtype,
            device=device,
        )
        new_predicted = wrapper.score_entities(
            fused_tokens=new_outputs["pre_prototype_fused_tokens"],
            image_nodes=new_outputs["image_nodes"],
            entity_subword_masks=predicted_masks,
            entity_type_ids=predicted_type_ids,
            grounding_null_prior=neutral_priors,
            batch=batch,
        )
        old_predicted_stages = _legacy_scalar_scores(
            teacher,
            fused_tokens=legacy_outputs[
                "pre_prototype_fused_tokens"
            ],
            image_nodes=legacy_outputs["image_nodes"],
            entity_masks=predicted_masks,
            entity_type_ids=predicted_type_ids,
            entity_valid=predicted_valid,
            null_priors=neutral_priors,
            batch=batch,
        )
        if predicted_valid.any():
            for stage_name in grounding_errors:
                grounding_errors[stage_name] = max(
                    grounding_errors[stage_name],
                    _max_abs(
                        old_predicted_stages[stage_name][
                            predicted_valid
                        ],
                        new_predicted[stage_name][predicted_valid],
                    ),
                )
        old_predicted_formal = old_predicted_stages["formal_logits"]

        for row, metadata in enumerate(batch["metadata"]):
            old_predictions = []
            new_predictions = []
            for entity_index, span in enumerate(predicted_spans[row]):
                if not bool(predicted_valid[row, entity_index].item()):
                    continue
                common = {
                    "span": span,
                    "type_id": int(
                        predicted_type_ids[row, entity_index].item()
                    ),
                }
                old_predictions.append(
                    {
                        **common,
                        "region_index": int(
                            old_predicted_formal[
                                row, entity_index
                            ].argmax().item()
                        ),
                    }
                )
                new_predictions.append(
                    {
                        **common,
                        "region_index": int(
                            new_predicted["formal_logits"][
                                row, entity_index
                            ].argmax().item()
                        ),
                    }
                )
            old_keys = {_prediction_key(item) for item in old_predictions}
            new_keys = {_prediction_key(item) for item in new_predictions}
            prediction_set_equal &= old_keys == new_keys
            record_id = str(metadata.get("record_id", ""))
            _update_digest(old_digest, record_id, old_predictions)
            _update_digest(new_digest, record_id, new_predictions)
            gold = _gold_for_record(batch, row)
            matches = _match_formal_record_predictions(
                predictions=new_predictions,
                gold=gold,
                metadata=metadata,
                region_boxes=batch["region_boxes"][row],
                null_region_index=int(
                    batch["null_region_index"][row].item()
                ),
            )
            for name in _METRICS:
                correct[name] += len(matches[name])
            record_count += 1
            predicted_count += len(new_predictions)
            gold_count += len(gold)

    metrics = _metric_report(correct, predicted_count, gold_count)
    new_sha = new_digest.hexdigest()
    old_sha = old_digest.hexdigest()
    prediction_set_equal &= old_sha == new_sha
    checks: dict[str, bool] = {
        "backbone_forward_states": all(
            value < emission_tolerance
            for value in state_errors.values()
        ),
        "typed_bio_emissions": emission_error < emission_tolerance,
        "typed_bio_decode": decoded_equal,
        "raw_grounding_logits": (
            grounding_errors["raw_logits"] < grounding_tolerance
        ),
        "after_entity_null_prior": (
            grounding_errors["after_entity_null_prior"]
            < grounding_tolerance
        ),
        "after_global_null_bias": (
            grounding_errors["after_global_null_bias"]
            < grounding_tolerance
        ),
        "after_detector_prior": (
            grounding_errors["after_detector_prior"]
            < grounding_tolerance
        ),
        "after_compatibility_prior": (
            grounding_errors["after_compatibility_prior"]
            < grounding_tolerance
        ),
        "formal_grounding_logits": (
            grounding_errors["formal_logits"] < grounding_tolerance
        ),
        "region_null_argmax": grounding_argmax_equal,
        "null_visible_decision": null_visible_equal,
        "positive_set_correctness": positive_set_correctness_equal,
        "prediction_set": prediction_set_equal,
        "prediction_digest": old_sha == new_sha,
        "record_encoded_once": encoded_record_count == record_count,
        "legacy_null_index_alignment": null_index_aligned,
    }
    baseline_checks: dict[str, bool] = {}
    if expected_baseline is not None:
        expected_dev = dict(expected_baseline.get("dev") or {})
        baseline_checks = {
            "record_count": record_count
            == int(expected_dev.get("records", -1)),
            "prediction_count": predicted_count
            == int(expected_dev.get("prediction_count", -1)),
            "gold_count": gold_count
            == int(expected_dev.get("gold_count", -1)),
            "prediction_sha256": new_sha
            == str(expected_dev.get("prediction_sha256", "")),
        }
        for name in _METRICS:
            expected_key = "mner_f1" if name == "mner" else f"{name}_f1"
            baseline_checks[expected_key] = (
                abs(
                    float(metrics[f"{name}_f1"])
                    - float(expected_dev.get(expected_key, -1.0))
                )
                < emission_tolerance
            )
            correct_key = (
                "mner_correct"
                if name == "mner"
                else f"{name}_correct"
            )
            baseline_checks[correct_key] = int(
                metrics[f"{name}_correct"]
            ) == int(expected_dev.get(correct_key, -1))
        checks["frozen_baseline"] = all(baseline_checks.values())

    return {
        "kind": "s3_forward_equivalence",
        "format_version": 1,
        "scope": "dev",
        "records": record_count,
        "tolerance": {
            "emissions_and_states": float(emission_tolerance),
            "grounding": float(grounding_tolerance),
        },
        "max_abs_error": {
            "backbone_states": state_errors,
            "typed_bio_emissions": emission_error,
            "grounding": grounding_errors,
        },
        "checks": checks,
        "baseline_checks": baseline_checks,
        "metrics": metrics,
        "legacy_prediction_sha256": old_sha,
        "record_prediction_sha256": new_sha,
        "formal_decode_null_prior_mode": "legacy_neutral_0.5",
        "gate_passed": all(checks.values()),
        "test_accessed": False,
    }
