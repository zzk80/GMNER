#!/usr/bin/env python3
"""Materialize deterministic final-chain OOF rows and action population."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.full_chain_oof_contract import fold_from_manifest, validate_fold_manifest
from gmner.data.p4_r0b_regeneration_contract import validate_m33a_formal_oof_payload


SOURCE_NAMES = {0: "stage1", 1: "viterbi", 2: "kbest", 3: "perturbation"}
TYPE_ORDER = ("LOC", "PER", "ORG", "OTHER")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-state", required=True)
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--r16-cache", required=True)
    parser.add_argument("--r36-cache", required=True)
    parser.add_argument("--fold-summary", required=True)
    parser.add_argument("--pipeline-manifest", required=True)
    parser.add_argument("--d0-preflight", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def identity(prefix: str, payload: dict[str, Any]) -> str:
    def reject_float(value: Any) -> None:
        if isinstance(value, float):
            raise TypeError(f"Floating identity input is forbidden: {payload}")
        if isinstance(value, dict):
            for item in value.values():
                reject_float(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                reject_float(item)

    reject_float(payload)
    return f"{prefix}:{hashlib.sha256(canonical_bytes(payload)).hexdigest()}"


def finite(value: Any, trail: str = "row") -> None:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite value at {trail}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            finite(item, f"{trail}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            finite(item, f"{trail}[{index}]")


def span_object(span: list[int] | tuple[int, int]) -> dict[str, Any]:
    start, end = (int(span[0]), int(span[1]))
    if start < 0 or end <= start:
        raise ValueError(f"Invalid word-space span: {span}")
    return {"start": start, "end": end, "space": "word_half_open"}


def overlap(left: tuple[int, int], right: tuple[int, int]) -> int:
    return max(0, min(left[1], right[1]) - max(left[0], right[0]))


def tensor_row(batch: dict, key_path: tuple[str, ...], row: int) -> torch.Tensor:
    value: Any = batch
    for key in key_path:
        value = value[key]
    return value[row]


def flatten_states(payload: dict) -> dict[str, tuple[dict, int]]:
    result: dict[str, tuple[dict, int]] = {}
    for batch in payload["batches"]:
        for row, record_id in enumerate(batch["record_ids"]):
            record_id = str(record_id)
            if record_id in result:
                raise ValueError(f"Duplicate formal-state record: {record_id}")
            result[record_id] = (batch, row)
    return result


def region_id(record_id: str, image_id: str, source_index: int | None) -> str:
    payload = (
        {"kind": "region", "record_id": record_id, "null": "NULL"}
        if source_index is None
        else {
            "kind": "region",
            "record_id": record_id,
            "image_id": image_id,
            "vinvl_source_index": int(source_index),
        }
    )
    return identity("region", payload)


def region_rows(record_id: str, image_id: str, record: dict) -> tuple[list[dict], dict[int, str]]:
    mask = record["region_mask"].bool()
    null_mask = record["region_is_null"].bool()
    source = record["region_source_indices"].long()
    rows = []
    mapping: dict[int, str] = {}
    for index in torch.nonzero(mask, as_tuple=False).squeeze(-1).tolist():
        is_null = bool(null_mask[index].item())
        source_index = None if is_null else int(source[index].item())
        if not is_null and source_index < 0:
            raise ValueError(f"Missing VinVL source index for {record_id} region {index}.")
        stable = region_id(record_id, image_id, source_index)
        mapping[int(index)] = stable
        rows.append(
            {
                "region_candidate_id": stable,
                "local_index": int(index),
                "vinvl_source_index": source_index,
                "bbox_xyxy": [float(value) for value in record["region_boxes"][index].tolist()],
                "detector_score": float(record["region_detector_scores"][index].item()),
                "is_null": is_null,
            }
        )
    return rows, mapping


def best_region(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
    visible: bool,
    null_index: int,
) -> int:
    if not visible:
        return int(null_index)
    real = candidate_mask.bool().clone()
    if 0 <= null_index < real.numel():
        real[null_index] = False
    if not real.any():
        return int(null_index)
    return int(logits.float().masked_fill(~real, -1e4).argmax().item())


def candidate_set(
    record_id: str,
    image_id: str,
    record: dict,
    *,
    final_logits: torch.Tensor | None = None,
    final_candidate_mask: torch.Tensor | None = None,
    final_visible: torch.Tensor | None = None,
) -> tuple[dict, list[dict]]:
    regions, region_mapping = region_rows(record_id, image_id, record)
    null_index = int(record["metadata"]["null_region_index"])
    candidates = []
    for index in range(int(record["span_mask"].sum().item())):
        span = [int(value) for value in record["span_candidates"][index].tolist()]
        type_id = int(record["fixed_type_ids"][index].item())
        source = SOURCE_NAMES.get(int(record["span_source_ids"][index].item()), "unknown")
        if final_logits is None:
            region_index = int(record["base_region_indices"][index].item())
        else:
            region_index = best_region(
                final_logits[index],
                final_candidate_mask[index],
                bool(final_visible[index].item()),
                null_index,
            )
        stable_region = region_mapping[region_index]
        candidate_identity = identity(
            "candidate",
            {
                "kind": "candidate",
                "record_id": record_id,
                "span": span,
                "type_id": type_id,
                "candidate_source": source,
                "region_candidate_id": stable_region,
            },
        )
        type_logits = [float(value) for value in record["stage1_type_logits"][index].tolist()]
        candidates.append(
            {
                "candidate_id": candidate_identity,
                "span": span_object(span),
                "type_id": type_id,
                "source": source,
                "region_candidate_id": stable_region,
                "region_index": region_index,
                "scores": {
                    "span_base_score": float(record["span_base_scores"][index].item()),
                    "type_logits": type_logits,
                    "region_score": float(
                        (final_logits[index, region_index] if final_logits is not None else record["base_region_scores"][index, region_index]).item()
                    ),
                },
                "_span_tuple": tuple(span),
                "_row_index": index,
            }
        )
    serialized = []
    for candidate in candidates:
        serialized.append({key: value for key, value in candidate.items() if not key.startswith("_") and key != "region_index"})
    return {
        "null_region_index": null_index,
        "span_candidates": serialized,
        "region_candidates": regions,
    }, candidates


def stage_provenance(pipeline: dict) -> dict:
    result = {}
    for name in ("stage1", "hierarchical", "coarse", "fine", "evidence"):
        stage = pipeline["stages"][name]
        result[name] = {
            "checkpoint_sha256": stage["checkpoint"]["sha256"],
            "config_sha256": stage["config"]["sha256"],
            "heldout_excluded": bool(stage["heldout_excluded"]),
        }
    return result


def discrete_projection(rows: list[dict]) -> dict:
    return {
        "record_ids": [row["record_id"] for row in rows],
        "predictions": [
            [item["prediction_id"] for item in row["formal_predictions"]]
            for row in rows
        ],
        "actions": [
            [item["action_id"] for item in row["replacement_actions"]]
            for row in rows
        ],
        "r16_candidates": [
            [item["candidate_id"] for item in row["r16_candidates"]["span_candidates"]]
            for row in rows
        ],
        "r36_regions": [
            [item["region_candidate_id"] for item in row["r36_candidates"]["region_candidates"]]
            for row in rows
        ],
    }


def build_rows(
    formal_payload: dict,
    r16_payload: dict,
    r36_payload: dict,
    *,
    fold: dict,
    pipeline: dict,
    d0: dict,
    fold_id: int = 0,
) -> list[dict]:
    states = flatten_states(formal_payload)
    r16_by_id = {str(row["metadata"]["record_id"]): row for row in r16_payload["records"]}
    r36_by_id = {str(row["metadata"]["record_id"]): row for row in r36_payload["records"]}
    expected = [str(value) for value in fold["heldout_record_ids"]]
    if list(states) != expected or list(r16_by_id) != expected or list(r36_by_id) != expected:
        raise ValueError("Formal state and candidate cache orders differ from the fold.")
    provenance = {
        "source_dataset_sha256": str(d0["heldout_file_sha256"]),
        "heldout_exclusion_proof_sha256": sha256_file(Path(d0["fold_manifest"])),
        "code_sha256": str(d0["source_tree_sha256"]),
        "stages": stage_provenance(pipeline),
    }
    rows = []
    for record_id in expected:
        batch, batch_row = states[record_id]
        r16 = r16_by_id[record_id]
        r36 = r36_by_id[record_id]
        image_id = str(r36["metadata"].get("image_id") or "")
        if not image_id:
            raise ValueError(f"Missing image_id for {record_id}.")
        final_logits = tensor_row(batch, ("fine_outputs", "final_region_logits"), batch_row).float()
        final_candidate_mask = tensor_row(batch, ("fine_outputs", "candidate_mask"), batch_row).bool()
        final_visible = tensor_row(batch, ("current_visible",), batch_row).bool()
        deployment = tensor_row(batch, ("deployment_span_mask",), batch_row).bool()
        fixed_types = tensor_row(batch, ("hierarchy_outputs", "fixed_type_ids"), batch_row).long()
        base_is_null = tensor_row(batch, ("base_is_null",), batch_row).bool()
        r16_set, _ = candidate_set(record_id, image_id, r16)
        r36_set, r36_candidates = candidate_set(
            record_id,
            image_id,
            r36,
            final_logits=final_logits,
            final_candidate_mask=final_candidate_mask,
            final_visible=final_visible,
        )
        region_map = {
            item["local_index"]: item["region_candidate_id"]
            for item in r36_set["region_candidates"]
        }
        null_index = int(r36_set["null_region_index"])
        formal_predictions = []
        formal_by_row: dict[int, dict] = {}
        for span_index in torch.nonzero(deployment, as_tuple=False).squeeze(-1).tolist():
            span = [int(value) for value in r36["span_candidates"][span_index].tolist()]
            type_id = int(fixed_types[span_index].item())
            region_index = best_region(
                final_logits[span_index],
                final_candidate_mask[span_index],
                bool(final_visible[span_index].item()),
                null_index,
            )
            stable_region = region_map[region_index]
            logits = r36["stage1_type_logits"][span_index].float().clone()
            logits[type_id] = max(float(logits[type_id].item()), float(logits.max().item()) + 1e-4)
            identity_inputs = {
                "kind": "prediction",
                "record_id": record_id,
                "span": span,
                "type_id": type_id,
                "region_candidate_id": stable_region,
            }
            prediction_id = identity("prediction", identity_inputs)
            stage1_id = identity("stage1", {**identity_inputs, "kind": "stage1"})
            prediction = {
                "prediction_id": prediction_id,
                "span": span_object(span),
                "type_id": type_id,
                "type_logits": [float(value) for value in logits.tolist()],
                "region_index": region_index,
                "region_candidate_id": stable_region,
                "region_is_null": region_index == null_index,
                "stage1_identity": stage1_id,
                "observable_features": {
                    "span_base_score": float(r36["span_base_scores"][span_index].item()),
                    "type_order": list(TYPE_ORDER),
                    "base_is_null": bool(base_is_null[span_index].item()),
                    "final_visible": bool(final_visible[span_index].item()),
                    "fine_region_logit": float(final_logits[span_index, region_index].item()),
                },
            }
            formal_predictions.append(prediction)
            formal_by_row[span_index] = prediction

        actions = []
        formal_spans = {
            index: tuple(int(value) for value in r36["span_candidates"][index].tolist())
            for index in formal_by_row
        }
        for base_index, base_prediction in formal_by_row.items():
            base_span = formal_spans[base_index]
            base_score = float(r36["span_base_scores"][base_index].item())
            for candidate in r36_candidates:
                candidate_index = int(candidate["_row_index"])
                candidate_span = candidate["_span_tuple"]
                if candidate_index == base_index or overlap(base_span, candidate_span) <= 0:
                    continue
                other_conflicts = sum(
                    overlap(candidate_span, other_span) > 0
                    for other_index, other_span in formal_spans.items()
                    if other_index != base_index
                )
                action_id = identity(
                    "action",
                    {
                        "kind": "action",
                        "record_id": record_id,
                        "base_prediction_id": base_prediction["prediction_id"],
                        "candidate_id": candidate["candidate_id"],
                    },
                )
                candidate_score = float(candidate["scores"]["span_base_score"])
                actions.append(
                    {
                        "action_id": action_id,
                        "base_prediction_id": base_prediction["prediction_id"],
                        "candidate_id": candidate["candidate_id"],
                        "candidate_source": candidate["source"],
                        "candidate_score": candidate_score,
                        "base_candidate_margin": base_score - candidate_score,
                        "boundary_distance": abs(base_span[0] - candidate_span[0]) + abs(base_span[1] - candidate_span[1]),
                        "candidate_type_id": int(candidate["type_id"]),
                        "conflict_features": {
                            "overlap_words_with_base": overlap(base_span, candidate_span),
                            "overlaps_other_formal_count": int(other_conflicts),
                            "would_preserve_prediction_count": True,
                        },
                        "observable_features": {
                            "base_span": list(base_span),
                            "candidate_span": list(candidate_span),
                            "candidate_region_candidate_id": candidate["region_candidate_id"],
                            "candidate_region_score": float(candidate["scores"]["region_score"]),
                        },
                    }
                )
        actions.sort(key=lambda item: item["action_id"])
        row = {
            "kind": "final_chain_oof_record",
            "format_version": 1,
            "record_id": record_id,
            "image_id": image_id,
            "fold_id": int(fold_id),
            "heldout": True,
            "test_accessed": False,
            "provenance": provenance,
            "formal_predictions": formal_predictions,
            "r16_candidates": r16_set,
            "r36_candidates": r36_set,
            "chain_state": {
                "hierarchical": {"fixed_type_ids": [int(value) for value in fixed_types[: int(r36["span_mask"].sum())].tolist()]},
                "coarse": {"promoted_candidate_count": int(tensor_row(batch, ("fine_outputs", "promoted_candidate_mask"), batch_row).sum().item())},
                "fine": {"candidate_counts": [int(value) for value in final_candidate_mask.sum(dim=-1).tolist()]},
                "evidence": {"final_visible": [bool(value) for value in final_visible.tolist()]},
            },
            "replacement_actions": actions,
        }
        finite(row)
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.fold_summary).resolve()
    manifest = validate_fold_manifest(
        manifest_path, expected_num_folds=10, verify_fold_ids=(args.fold_id,)
    )
    fold = fold_from_manifest(manifest, args.fold_id)
    formal_payload = torch.load(Path(args.formal_state), map_location="cpu")
    validate_m33a_formal_oof_payload(
        formal_payload,
        expected_fold_id=args.fold_id,
        expected_record_ids=[str(value) for value in fold["heldout_record_ids"]],
    )
    r16_payload = torch.load(Path(args.r16_cache), map_location="cpu")
    r36_payload = torch.load(Path(args.r36_cache), map_location="cpu")
    pipeline = json.loads(Path(args.pipeline_manifest).read_text(encoding="utf-8"))
    d0 = json.loads(Path(args.d0_preflight).read_text(encoding="utf-8"))
    rows = build_rows(
        formal_payload,
        r16_payload,
        r36_payload,
        fold=fold,
        pipeline=pipeline,
        d0=d0,
        fold_id=args.fold_id,
    )
    replay = build_rows(
        formal_payload,
        r16_payload,
        r36_payload,
        fold=fold,
        pipeline=pipeline,
        d0=d0,
        fold_id=args.fold_id,
    )
    first_discrete = discrete_projection(rows)
    replay_discrete = discrete_projection(replay)
    if first_discrete != replay_discrete:
        raise RuntimeError("Fold deterministic replay changed discrete outputs.")
    discrete_sha = hashlib.sha256(canonical_bytes(first_discrete)).hexdigest()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(output)
    report = {
        "kind": "final_chain_oof_fold_materialization",
        "format_version": 1,
        "status": "PASSED",
        "fold_id": int(args.fold_id),
        "records": len(rows),
        "record_ids_sha256": stable_id_digest([row["record_id"] for row in rows]),
        "formal_prediction_count": sum(len(row["formal_predictions"]) for row in rows),
        "replacement_action_count": sum(len(row["replacement_actions"]) for row in rows),
        "discrete_replay_sha256": discrete_sha,
        "double_run_formal_digest_exact": True,
        "double_run_action_digest_exact": True,
        "rows": str(output),
        "rows_sha256": sha256_file(output),
        "other_folds_accessed": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    manifest_output = Path(args.manifest_output).resolve()
    manifest_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
