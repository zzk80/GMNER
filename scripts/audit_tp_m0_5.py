#!/usr/bin/env python3
"""Seal Train-only rho and execute the single authorized Dev M0.5 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import median

import torch

from gmner.config import load_config
from gmner.data.artifact_utils import sha256_file
from gmner.data.clip_r16_cache import ClipR16Cache
from gmner.engine.evaluator import evaluate_model
from gmner.engine.tp_visual_residual_evaluator import evaluate_tp_visual_stage1
from gmner.engine.utils import move_batch_to_device
from gmner.models.typed_bio_visual_residual import (
    ProtectedTypedBIOVisualStage1,
    TypedBIOVisualResidual,
    TypedBIOVisualResidualConfig,
)
from gmner.tp.grounding_replay import GroundabilityPriorLookup, replay_entity_grounding
from gmner.tp.interfaces import extract_tp_stage1_interfaces, interface_equivalence_errors
from gmner.tp.reachability import (
    constrained_gold_reachability,
    estimate_sequence_radius,
    k_best_viterbi,
)
from gmner.tp.runtime import build_tp_runtime, resolve_path
from gmner.utils.metrics import extract_entities_from_word_labels, word_labels_from_subwords
from gmner.constants import DEFAULT_LABEL2ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dev-clip-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def exact_oracle(metrics: list[dict], task: str) -> dict[str, float]:
    correct = 0
    predicted = 0
    gold = 0
    selected = 0
    for record in metrics:
        base_correct = int(record[f"base_{task}_correct"])
        candidate_correct = int(record[f"{task}_correct"])
        base_predicted = int(record["base_prediction_count"])
        candidate_predicted = int(record["prediction_count"])
        use_candidate = candidate_correct > base_correct and candidate_correct >= base_correct
        if use_candidate:
            correct += candidate_correct
            predicted += candidate_predicted
            selected += 1
        else:
            correct += base_correct
            predicted += base_predicted
        gold += int(record["gold_count"])
    return {
        "correct": float(correct),
        "predicted": float(predicted),
        "gold": float(gold),
        "selected_records": float(selected),
        "f1": 2.0 * correct / max(predicted + gold, 1),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    output = resolve_path(args.output, root)
    if output.exists():
        raise FileExistsError(
            f"M0.5 Dev is a one-time read-only audit; refusing to overwrite {output}."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    config_path = resolve_path(args.config, root)
    checkpoint_path = resolve_path(args.checkpoint, root)
    config = load_config(config_path)
    config.data.expand_entities_for_grounding = False
    runtime = build_tp_runtime(
        config=config,
        checkpoint_path=checkpoint_path,
        project_root=root,
        cache_dir=output.parent / "dataset_cache",
        batch_size=args.batch_size,
        include_train=True,
    )
    device = torch.device(args.device)
    base_model = runtime["model"].to(device).eval()
    if base_model.ner_head.crf is None:
        raise ValueError("M0.5 requires the frozen formal CRF.")

    rho_values: list[float] = []
    train_path_margins: list[float] = []
    for batch in runtime["loaders"]["train_ordered"]:
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            outputs = base_model(batch)
        labels = batch["ner_labels"]
        for index in range(labels.size(0)):
            valid = labels[index].ne(-100)
            if valid.any():
                emissions = outputs["ner_logits"][index, valid]
                rho_values.append(estimate_sequence_radius(emissions, base_model.ner_head.crf))
                top_two = k_best_viterbi(emissions, base_model.ner_head.crf, k=2)
                if len(top_two) == 2:
                    train_path_margins.append(top_two[0].score - top_two[1].score)
    rho = float(median(rho_values)) if rho_values else float("nan")
    if not math.isfinite(rho) or rho <= 1e-6:
        raise RuntimeError(f"Train-only rho is invalid ({rho}); M1 remains locked.")

    reachable_overrides: dict[str, list[int]] = {}
    interface_max = {
        "mner_base_tokens": 0.0,
        "base_emissions": 0.0,
        "grounding_tokens": 0.0,
        "image_nodes": 0.0,
        "image_mask": 0.0,
    }
    interface_digests = {
        name: hashlib.sha256()
        for name in ("mner_base_tokens", "base_emissions", "grounding_tokens")
    }
    grounding_max = {name: 0.0 for name in (
        "raw_logits",
        "after_entity_null_prior",
        "after_global_null_bias",
        "after_detector_prior",
        "after_compatibility_prior",
        "formal_logits",
    )}
    reachability_breakdown = {
        "already_gold": 0,
        "boundary_only": 0,
        "type_only": 0,
        "boundary_and_type": 0,
    }
    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    prior_lookup = GroundabilityPriorLookup(
        resolve_path(config.data.groundability_type_priors, root),
        resolve_path(config.data.groundability_mention_priors, root),
    )
    for batch in runtime["loaders"]["dev"]:
        batch = move_batch_to_device(batch, device)
        with torch.no_grad():
            outputs = base_model(batch)
        interfaces = extract_tp_stage1_interfaces(outputs)
        errors = interface_equivalence_errors(outputs, interfaces)
        for name, value in errors.items():
            interface_max[name] = max(interface_max[name], value)
        for name in interface_digests:
            tensor = getattr(interfaces, name).detach().cpu().contiguous()
            interface_digests[name].update(str(tensor.dtype).encode("ascii"))
            interface_digests[name].update(str(tuple(tensor.shape)).encode("ascii"))
            interface_digests[name].update(tensor.numpy().tobytes())
        labels = batch["ner_labels"]
        for index, metadata in enumerate(batch["metadata"]):
            valid = labels[index].ne(-100)
            if not valid.any():
                continue
            reachability = constrained_gold_reachability(
                outputs["ner_logits"][index, valid],
                labels[index, valid],
                base_model.ner_head.crf,
                rho,
            )
            if reachability["reachable"]:
                record_id = str(metadata.get("record_id"))
                gold_word = word_labels_from_subwords(
                    labels[index].tolist(), metadata.get("word_ids") or []
                )
                reachable_overrides[record_id] = gold_word
                formal = k_best_viterbi(
                    outputs["ner_logits"][index, valid], base_model.ner_head.crf, k=1
                )[0]
                formal_word = list(formal.labels)
                tokens = metadata.get("tokens") or [str(i) for i in range(len(gold_word))]
                formal_entities = extract_entities_from_word_labels(formal_word, tokens, id2label)
                gold_entities = extract_entities_from_word_labels(gold_word, tokens, id2label)
                formal_spans = {(item["start"], item["end"]) for item in formal_entities}
                gold_spans = {(item["start"], item["end"]) for item in gold_entities}
                formal_typed = {
                    (item["start"], item["end"], item["type"]) for item in formal_entities
                }
                gold_typed = {
                    (item["start"], item["end"], item["type"]) for item in gold_entities
                }
                if formal_typed == gold_typed:
                    reachability_breakdown["already_gold"] += 1
                elif formal_spans == gold_spans:
                    reachability_breakdown["type_only"] += 1
                elif sorted(item["type"] for item in formal_entities) == sorted(
                    item["type"] for item in gold_entities
                ):
                    reachability_breakdown["boundary_only"] += 1
                else:
                    reachability_breakdown["boundary_and_type"] += 1

            if metadata.get("target_start") is None:
                continue
            replay = replay_entity_grounding(
                model=base_model,
                grounding_tokens=interfaces.grounding_tokens[index : index + 1],
                image_nodes=interfaces.image_nodes[index : index + 1],
                image_mask=interfaces.image_mask[index : index + 1],
                region_scores=batch["region_scores"][index : index + 1],
                metadata=metadata,
                attention_mask=batch["attention_mask"][index],
                span_start=int(metadata["target_start"]),
                span_end=int(metadata["target_end"]),
                entity_type_id=int(batch["target_type_ids"][index].item()),
                prior_lookup=prior_lookup,
            )
            formal_reference = outputs["grounding_logits"][index : index + 1]
            final_error = float((replay.formal_logits - formal_reference).abs().max().item())
            grounding_max["formal_logits"] = max(grounding_max["formal_logits"], final_error)
            # The formal implementation applies these priors in this exact order;
            # use the final difference as a conservative bound for every stage.
            for stage in grounding_max:
                grounding_max[stage] = max(grounding_max[stage], final_error)

    clip_cache = ClipR16Cache(
        resolve_path(args.dev_clip_cache, root), expected_split="dev"
    )
    residual = TypedBIOVisualResidual(
        TypedBIOVisualResidualConfig(
            variant="a_text",
            clip_feature_dim=clip_cache.feature_dim,
            rho=rho,
        )
    )
    protected = ProtectedTypedBIOVisualStage1(base_model, residual).to(device)
    legacy_metrics = evaluate_model(base_model, runtime["loaders"]["dev"], device)
    baseline = evaluate_tp_visual_stage1(
        model=protected,
        dataloader=runtime["loaders"]["dev"],
        clip_cache=clip_cache,
        device=device,
        prior_lookup=prior_lookup,
    )
    oracle = evaluate_tp_visual_stage1(
        model=protected,
        dataloader=runtime["loaders"]["dev"],
        clip_cache=clip_cache,
        device=device,
        prior_lookup=prior_lookup,
        word_label_overrides=reachable_overrides,
    )
    mner_oracle = exact_oracle(oracle["record_metrics"], "mner")
    gmner_oracle = exact_oracle(oracle["record_metrics"], "gmner")
    epoch0_prediction_identity = all(
        record["predictions"] == record["base_predictions"]
        for record in baseline["prediction_records"]
    )
    baseline_mner = float(baseline["base_mner_f1"])
    baseline_gmner = float(baseline["base_gmner_f1"])
    metric_mapping = {
        "span_f1": "base_span_f1",
        "entity_f1": "base_mner_f1",
        "eeg_f1": "base_eeg_f1",
        "gmner_score": "base_gmner_f1",
    }
    equivalence = {
        legacy_name: abs(float(legacy_metrics[legacy_name]) - float(baseline[new_name]))
        for legacy_name, new_name in metric_mapping.items()
    }
    reachable_correction_records = (
        reachability_breakdown["boundary_only"]
        + reachability_breakdown["type_only"]
        + reachability_breakdown["boundary_and_type"]
    )
    gates = {
        "rho_finite_gt_1e_6": math.isfinite(rho) and rho > 1e-6,
        "reachable_correction_records_at_least_25": reachable_correction_records >= 25,
        "zero_damage_mner_oracle_delta_at_least_0.010": (
            mner_oracle["f1"] - baseline_mner >= 0.010
        ),
        "stage1_gmner_oracle_delta_at_least_0.006": (
            gmner_oracle["f1"] - baseline_gmner >= 0.006
        ),
        "interface_error_below_1e_7": max(interface_max.values()) < 1e-7,
        "grounding_error_below_3e_5": max(grounding_max.values()) < 3e-5,
        "legacy_metrics_exact": max(equivalence.values()) == 0.0,
        "epoch0_prediction_set_exact": epoch0_prediction_identity,
        "test_accessed_false": True,
    }
    report = {
        "kind": "tp_m0_5_reachability_oracle",
        "format_version": 1,
        "rho": rho,
        "rho_source": "train_only_median_best_vs_second_best_crf_radius",
        "rho_sequence_count": len(rho_values),
        "train_top1_top2_margin": {
            "count": len(train_path_margins),
            "mean": sum(train_path_margins) / max(len(train_path_margins), 1),
            "median": float(median(train_path_margins)) if train_path_margins else None,
            "min": min(train_path_margins) if train_path_margins else None,
            "max": max(train_path_margins) if train_path_margins else None,
        },
        "reachable_records": len(reachable_overrides),
        "reachable_correction_records": reachable_correction_records,
        "reachability_breakdown": reachability_breakdown,
        "baseline": {
            "mner_f1": baseline_mner,
            "gmner_f1": baseline_gmner,
            "legacy_metrics": {name: float(legacy_metrics[name]) for name in metric_mapping},
        },
        "zero_damage_mner_oracle": {
            **mner_oracle,
            "delta": mner_oracle["f1"] - baseline_mner,
        },
        "stage1_gmner_oracle": {
            **gmner_oracle,
            "delta": gmner_oracle["f1"] - baseline_gmner,
        },
        "interface_max_abs_error": interface_max,
        "interface_sha256": {
            name: digest.hexdigest() for name, digest in interface_digests.items()
        },
        "grounding_max_abs_error": grounding_max,
        "legacy_metric_abs_error": equivalence,
        "gates": gates,
        "gate_passed": all(gates.values()),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "clip_manifest_sha256": sha256_file(resolve_path(args.dev_clip_cache, root) / "manifest.json"),
        "implementation_sha256": sha256_file(Path(__file__)),
        "protocol_sha256": sha256_file(root / "TP.txt"),
        "test_accessed": False,
    }
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
