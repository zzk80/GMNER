"""Probe whether frozen R36 regions contain subtype-discriminative evidence.

This tool is deliberately Dev-only and read-only. It builds subtype visual
prototypes from Train gold-visible regions, then probes the frozen F2 errors
whose span, coarse type, and grounding are already correct.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.constants import ID2ENTITY_TYPE
from gmner.data import (
    PairedRecordCandidateCollator,
    PairedRecordCandidateDataset,
    RecordCandidateDataset,
)
from gmner.engine.fine_grounding_adapter_evaluator import (
    move_paired_record_batch,
)
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.fine_grounding_adapter_config import (
    load_fine_grounding_adapter_config,
)
from sidecars.fmnerg_joint.config import load_joint_subtype_config
from sidecars.fmnerg_joint.formal_chain import (
    FrozenM33AFeatureProvider,
    load_frozen_dev_contract,
)
from sidecars.fmnerg_joint.subtype_region_oracle import (
    analyze_visible_error,
    build_visual_prototype_bank,
    summarize_seed_rows,
)
from sidecars.fmnerg_subtype.data import (
    fine_gold_by_record,
    read_fine_conll,
)
from sidecars.fmnerg_subtype.encoder_config import (
    load_subtype_encoder_config,
)
from sidecars.fmnerg_subtype.encoder_evaluator import predict_online_subtypes
from sidecars.fmnerg_subtype.encoder_model import (
    build_trainable_subtype_encoder,
    load_trainable_checkpoint_state,
)
from sidecars.fmnerg_subtype.encoder_runtime import load_online_subtype_data
from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.online_data import OnlineSubtypeCollator
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy
from scripts.train_fine_grounding_adapter import (
    load_frozen_models,
    validate_fingerprints,
)


PREREGISTERED_SEEDS = (41, 42, 43)
DEFAULT_TOP_K = (1, 2, 4, 8, 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--joint-config",
        default="sidecars/fmnerg_joint/configs/j0_visual_fusion.yaml",
    )
    parser.add_argument(
        "--evidence-config",
        default="configs/fmnerg_twitter10000_evidence_visibility.yaml",
    )
    parser.add_argument("--seeds", default="41,42,43")
    parser.add_argument("--top-k", default="1,2,4,8,16")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        default=(
            "outputs/fmnerg_joint_subtype_region_oracle/"
            "dev_oracle.json"
        ),
    )
    return parser.parse_args()


def parse_integer_list(raw: str, *, name: str) -> list[int]:
    try:
        values = sorted({int(value.strip()) for value in raw.split(",")})
    except ValueError as exc:
        raise ValueError(f"{name} must be a comma-separated integer list.") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive.")
    return values


def select_device(raw: str) -> torch.device:
    if str(raw).startswith("cuda") and torch.cuda.is_available():
        return torch.device(raw)
    return torch.device("cpu")


def build_train_visual_prototypes(
    *,
    source_path: Path,
    provider: FrozenM33AFeatureProvider,
    taxonomy: SubtypeTaxonomy,
) -> tuple[Any, dict[str, Any]]:
    records = read_fine_conll(
        source_path,
        taxonomy,
        require_all_subtypes=True,
    )
    entity_features: list[tuple[int, torch.Tensor]] = []
    counts = Counter()
    subtype_visible = Counter()
    for record in tqdm(records, desc="Train visual prototypes"):
        if record.record_id not in provider.records:
            raise ValueError(
                f"Train R36 cache is missing record {record.record_id!r}."
            )
        raw = provider.records[record.record_id]
        span_rows = provider.span_rows[record.record_id]
        region_mask = torch.as_tensor(raw["region_mask"]).bool()
        null_mask = torch.as_tensor(raw["region_is_null"]).bool()
        region_features = torch.as_tensor(raw["region_features"]).float()
        for entity in record.entities:
            counts["gold_entities"] += 1
            span = (int(entity.start), int(entity.end))
            row = span_rows.get(span)
            if row is None:
                counts["span_missing"] += 1
                continue
            positive = torch.as_tensor(
                raw["gold_region_positive_mask"]
            )[row].bool()
            real_positive = positive & region_mask & ~null_mask
            if not real_positive.any():
                counts["no_real_positive"] += 1
                continue
            indices = torch.nonzero(
                real_positive, as_tuple=False
            ).reshape(-1)
            weights = torch.as_tensor(
                raw["region_iou_targets"]
            )[row, indices].float().clamp_min(1e-6)
            features = F.normalize(
                region_features.index_select(0, indices),
                dim=-1,
            )
            entity_feature = (
                features * (weights / weights.sum()).unsqueeze(-1)
            ).sum(dim=0)
            subtype_id = taxonomy.subtype_id(entity.subtype)
            entity_features.append((subtype_id, entity_feature))
            subtype_visible[entity.subtype] += 1
            counts["visible_entities"] += 1
            counts["positive_region_instances"] += int(indices.numel())
    bank = build_visual_prototype_bank(
        entity_features,
        num_subtypes=taxonomy.num_subtypes,
        feature_size=provider.region_feature_size,
    )
    if counts["span_missing"]:
        raise ValueError(
            "Train R36 cache does not cover every gold subtype span: "
            f"{counts['span_missing']} missing."
        )
    report = {
        "method": (
            "IoU-weighted mean of normalized Train gold-positive R36 "
            "features, followed by one normalized centroid per subtype"
        ),
        "source": str(source_path),
        "source_sha256": sha256_file(source_path),
        "expanded_cache": provider.artifact_report(),
        **{key: float(value) for key, value in counts.items()},
        "prototype_subtypes_available": float(bank.available.sum().item()),
        "prototype_subtypes_missing": [
            taxonomy.labels[index]
            for index in range(taxonomy.num_subtypes)
            if not bool(bank.available[index].item())
        ],
        "visible_entities_per_subtype": {
            label: float(subtype_visible.get(label, 0))
            for label in taxonomy.labels
        },
    }
    return bank, report


@torch.inference_mode()
def extract_fine_region_rankings(
    *,
    evidence_config_path: Path,
    expected_expanded_path: Path,
    root: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[dict[tuple[str, int, int], list[int]], dict[str, Any]]:
    evidence_config = load_evidence_visibility_config(evidence_config_path)
    formal_path = resolve_path(
        evidence_config.data.formal_dev_cache,
        root,
    )
    expanded_path = resolve_path(
        evidence_config.data.expanded_dev_cache,
        root,
    )
    if sha256_file(expanded_path) != sha256_file(expected_expanded_path):
        raise ValueError(
            "Fine Adapter Dev R36 cache differs from the joint Oracle cache."
        )
    formal = RecordCandidateDataset(formal_path)
    expanded = RecordCandidateDataset(expanded_path)
    paired = PairedRecordCandidateDataset(formal, expanded)

    fine_config_path = resolve_path(
        evidence_config.frozen.fine_config,
        root,
    )
    fine_config = load_fine_grounding_adapter_config(fine_config_path)
    (
        fine_model,
        hierarchy,
        _,
        hierarchy_checkpoint,
        coarse_checkpoint,
    ) = load_frozen_models(fine_config, root, device)
    validate_fingerprints(
        paired,
        hierarchy_checkpoint=hierarchy_checkpoint,
        coarse_checkpoint=coarse_checkpoint,
        require_oof=False,
    )
    fine_checkpoint_path = resolve_path(
        evidence_config.frozen.fine_checkpoint,
        root,
    )
    fine_checkpoint = torch.load(fine_checkpoint_path, map_location="cpu")
    fine_model.load_state_dict(fine_checkpoint["model_state_dict"])
    fine_model.to(device).eval()
    del hierarchy, hierarchy_checkpoint, coarse_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    loader = DataLoader(
        paired,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=0,
        collate_fn=PairedRecordCandidateCollator(),
    )
    rankings: dict[tuple[str, int, int], list[int]] = {}
    for raw_batch in tqdm(loader, desc="Frozen Fine Top-K"):
        batch = move_paired_record_batch(raw_batch, device)
        outputs = fine_model(batch["expanded"])
        logits = outputs["final_region_logits"].detach().cpu()
        candidate_mask = outputs["candidate_mask"].detach().cpu().bool()
        spans = raw_batch["expanded"]["span_candidates"]
        span_mask = raw_batch["expanded"]["span_mask"].bool()
        for row, metadata in enumerate(
            raw_batch["expanded"]["metadata"]
        ):
            record_id = str(metadata["record_id"])
            for span_row in torch.nonzero(
                span_mask[row], as_tuple=False
            ).reshape(-1).tolist():
                start, end = map(
                    int,
                    spans[row, span_row].tolist(),
                )
                valid = torch.nonzero(
                    candidate_mask[row, span_row],
                    as_tuple=False,
                ).reshape(-1)
                order = valid[
                    torch.argsort(
                        logits[row, span_row, valid],
                        descending=True,
                    )
                ].tolist()
                key = (record_id, start, end)
                if key in rankings:
                    raise ValueError(f"Duplicate Fine span ranking: {key}.")
                rankings[key] = [int(value) for value in order]
    report = {
        "formal_cache": str(formal_path),
        "formal_cache_sha256": sha256_file(formal_path),
        "expanded_cache": str(expanded_path),
        "expanded_cache_sha256": sha256_file(expanded_path),
        "fine_config": str(fine_config_path),
        "fine_config_sha256": sha256_file(fine_config_path),
        "fine_checkpoint": str(fine_checkpoint_path),
        "fine_checkpoint_sha256": sha256_file(fine_checkpoint_path),
        "fine_checkpoint_epoch": int(fine_checkpoint["epoch"]),
        "fine_checkpoint_metrics": dict(fine_checkpoint["metrics"]),
        "ranked_spans": float(len(rankings)),
        "candidate_budget": float(fine_config.model.final_budget),
    }
    del fine_model, fine_checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rankings, report


def load_f2_predictions(
    *,
    joint_config: Any,
    taxonomy: SubtypeTaxonomy,
    root: Path,
    device: torch.device,
    seeds: list[int],
) -> tuple[
    dict[int, dict[tuple[str, int, int], int]],
    dict[str, Any],
]:
    encoder_config_path = resolve_path(
        joint_config.initialization.subtype_encoder_config,
        root,
    )
    encoder_config = load_subtype_encoder_config(encoder_config_path)
    (
        _,
        _,
        dev_formal_dataset,
        formal_payload,
        data_artifacts,
    ) = load_online_subtype_data(
        config=encoder_config,
        taxonomy=taxonomy,
        root=root,
    )
    model, tokenizer, initialization, trainability = (
        build_trainable_subtype_encoder(
            config=encoder_config,
            taxonomy=taxonomy,
            root=root,
            device=device,
        )
    )
    collator = OnlineSubtypeCollator(
        tokenizer,
        max_length=int(initialization["max_length"]),
    )
    predictions_by_seed: dict[
        int, dict[tuple[str, int, int], int]
    ] = {}
    checkpoint_reports: dict[str, Any] = {}
    expected_keys = {
        (
            str(example["record_id"]),
            int(example["start"]),
            int(example["end"]),
        )
        for example in dev_formal_dataset.examples
    }
    for seed in seeds:
        checkpoint_path = resolve_path(
            joint_config.subtype_checkpoint(seed),
            root,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if checkpoint.get("kind") != "fmnerg_trainable_subtype_encoder":
            raise ValueError("Oracle input is not an F2 subtype checkpoint.")
        if checkpoint.get("test_accessed") is not False:
            raise ValueError("F2 subtype checkpoint accessed Test data.")
        if checkpoint.get("taxonomy_sha256") != taxonomy.source_sha256:
            raise ValueError("F2 subtype taxonomy fingerprint changed.")
        if checkpoint.get("config_sha256") != sha256_file(
            encoder_config_path
        ):
            raise ValueError("F2 subtype encoder config fingerprint changed.")
        if dict(checkpoint.get("trainability") or {}) != trainability:
            raise ValueError("F2 subtype trainability contract changed.")
        load_trainable_checkpoint_state(model, checkpoint["model"])
        predicted = predict_online_subtypes(
            model,
            dev_formal_dataset,
            collator=collator,
            batch_size=encoder_config.optim.eval_batch_size,
            device=device,
        )
        mapping = {
            (
                str(example["record_id"]),
                int(example["start"]),
                int(example["end"]),
            ): int(subtype_id)
            for example, subtype_id in zip(
                dev_formal_dataset.examples,
                predicted,
            )
        }
        if set(mapping) != expected_keys:
            raise ValueError("F2 subtype prediction coverage is incomplete.")
        predictions_by_seed[int(seed)] = mapping
        checkpoint_reports[str(seed)] = {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": int(checkpoint["epoch"]),
            "metrics": dict(checkpoint["metrics"]),
            "test_accessed": False,
        }
        del checkpoint
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return predictions_by_seed, {
        "encoder_config": str(encoder_config_path),
        "encoder_config_sha256": sha256_file(encoder_config_path),
        "initialization": initialization,
        "trainability": trainability,
        "data": data_artifacts,
        "formal_coarse_prediction_sha256": formal_payload["metadata"][
            "coarse_prediction_sha256"
        ],
        "checkpoints": checkpoint_reports,
    }


def analyze_seed(
    *,
    seed: int,
    formal_payload: dict[str, Any],
    provider: FrozenM33AFeatureProvider,
    fine_gold: dict[str, dict[tuple[int, int], dict[str, Any]]],
    fine_rankings: dict[tuple[str, int, int], list[int]],
    subtype_predictions: dict[tuple[str, int, int], int],
    bank: Any,
    taxonomy: SubtypeTaxonomy,
    top_ks: list[int],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    formal_prediction_count = 0
    gmner_correct_count = 0
    for formal_record in formal_payload["records"]:
        record_id = str(formal_record["record_id"])
        gold_spans = fine_gold[record_id]
        raw = provider.records[record_id]
        span_rows = provider.span_rows[record_id]
        region_mask = torch.as_tensor(raw["region_mask"]).bool()
        null_mask = torch.as_tensor(raw["region_is_null"]).bool()
        region_features = torch.as_tensor(raw["region_features"]).float()
        for prediction in formal_record.get("predictions") or []:
            formal_prediction_count += 1
            start, end = map(int, prediction["span"])
            span = (start, end)
            gold = gold_spans.get(span)
            if gold is None:
                continue
            if int(prediction["type_id"]) != int(gold["coarse_type_id"]):
                continue
            row = span_rows.get(span)
            if row is None:
                raise ValueError(
                    f"Formal span {record_id}/{span} is absent from R36."
                )
            positive = torch.as_tensor(
                raw["gold_region_positive_mask"]
            )[row].bool()
            formal_region_index = int(prediction["region_index"])
            if (
                formal_region_index < 0
                or formal_region_index >= positive.numel()
                or not bool(positive[formal_region_index].item())
            ):
                continue
            gmner_correct_count += 1
            key = (record_id, start, end)
            predicted_subtype_id = int(subtype_predictions[key])
            gold_subtype_id = int(gold["subtype_id"])
            if predicted_subtype_id == gold_subtype_id:
                continue
            parent_id = int(gold["coarse_type_id"])
            if taxonomy.parent_id(predicted_subtype_id) != parent_id:
                raise ValueError(
                    "F2 prediction escaped its fixed coarse parent."
                )
            visible = not bool(null_mask[formal_region_index].item())
            item: dict[str, Any] = {
                "seed": int(seed),
                "record_id": record_id,
                "span": [start, end],
                "text": str(gold["text"]),
                "coarse_type": ID2ENTITY_TYPE[parent_id],
                "gold_subtype": taxonomy.labels[gold_subtype_id],
                "predicted_subtype": taxonomy.labels[predicted_subtype_id],
                "formal_region_index": formal_region_index,
                "visibility": "visible" if visible else "null",
            }
            if visible:
                positive_real = set(
                    torch.nonzero(
                        positive & region_mask & ~null_mask,
                        as_tuple=False,
                    ).reshape(-1).tolist()
                )
                all_real = torch.nonzero(
                    region_mask & ~null_mask,
                    as_tuple=False,
                ).reshape(-1).tolist()
                ranking = fine_rankings.get(key)
                if ranking is None:
                    raise ValueError(f"Fine Top-K is missing span {key}.")
                item["visual_evidence"] = analyze_visible_error(
                    formal_region_index=formal_region_index,
                    fine_ranked_region_indices=ranking,
                    positive_region_indices=positive_real,
                    all_real_region_indices=[
                        int(value) for value in all_real
                    ],
                    region_features=region_features,
                    bank=bank,
                    taxonomy=taxonomy,
                    parent_id=parent_id,
                    gold_subtype_id=gold_subtype_id,
                    predicted_subtype_id=predicted_subtype_id,
                    top_ks=top_ks,
                )
            rows.append(item)
    expected_correct = int(
        formal_payload["metadata"]["coarse_metrics"]["gmner_correct"]
    )
    if gmner_correct_count != expected_correct:
        raise ValueError(
            "Recomputed Dev GMNER-correct count changed: "
            f"{gmner_correct_count} != {expected_correct}."
        )
    return {
        "seed": int(seed),
        "summary": summarize_seed_rows(
            rows,
            top_ks=top_ks,
            formal_prediction_count=formal_prediction_count,
            gmner_correct_count=gmner_correct_count,
        ),
        "errors": rows,
    }


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate_seed_results(
    results: list[dict[str, Any]],
    *,
    top_ks: list[int],
) -> dict[str, Any]:
    summaries = [dict(result["summary"]) for result in results]

    def values(path: tuple[str, ...]) -> list[float]:
        output = []
        for summary in summaries:
            current: Any = summary
            for key in path:
                current = current[key]
            output.append(float(current))
        return output

    error_sets = [
        {
            (
                str(row["record_id"]),
                tuple(map(int, row["span"])),
            )
            for row in result["errors"]
        }
        for result in results
    ]
    visible_error_sets = [
        {
            (
                str(row["record_id"]),
                tuple(map(int, row["span"])),
            )
            for row in result["errors"]
            if row["visibility"] == "visible"
        }
        for result in results
    ]
    return {
        "seed_count": float(len(results)),
        "subtype_wrong_given_gmner_correct": mean_std(
            values(("subtype_wrong_given_gmner_correct",))
        ),
        "visible_subtype_errors": mean_std(
            values(("visible_subtype_errors",))
        ),
        "null_subtype_errors": mean_std(
            values(("null_subtype_errors",))
        ),
        "formal_pairwise_support_rate": mean_std(
            values(
                (
                    "formal_region_probe",
                    "pairwise_support_rate_among_visible_errors",
                )
            )
        ),
        "formal_sibling_top1_rate": mean_std(
            values(
                (
                    "formal_region_probe",
                    "sibling_top1_rate_among_visible_errors",
                )
            )
        ),
        "fine_top_k": {
            str(value): {
                "pairwise_support_rate": mean_std(
                    values(
                        (
                            "fine_top_k_positive_oracle",
                            str(value),
                            "pairwise_support_rate_among_visible_errors",
                        )
                    )
                ),
                "sibling_top1_rate": mean_std(
                    values(
                        (
                            "fine_top_k_positive_oracle",
                            str(value),
                            "sibling_top1_rate_among_visible_errors",
                        )
                    )
                ),
                "incremental_pairwise_recovery_over_formal": mean_std(
                    values(
                        (
                            "fine_top_k_positive_oracle",
                            str(value),
                            "incremental_pairwise_recovery_over_formal",
                        )
                    )
                ),
                "incremental_sibling_recovery_over_formal": mean_std(
                    values(
                        (
                            "fine_top_k_positive_oracle",
                            str(value),
                            "incremental_sibling_recovery_over_formal",
                        )
                    )
                ),
            }
            for value in top_ks
        },
        "full_r36_positive_oracle": {
            "pairwise_support_rate": mean_std(
                values(
                    (
                        "full_r36_positive_oracle",
                        "pairwise_support_rate_among_visible_errors",
                    )
                )
            ),
            "sibling_top1_rate": mean_std(
                values(
                    (
                        "full_r36_positive_oracle",
                        "sibling_top1_rate_among_visible_errors",
                    )
                )
            ),
            "incremental_pairwise_recovery_over_formal": mean_std(
                values(
                    (
                        "full_r36_positive_oracle",
                        "incremental_pairwise_recovery_over_formal",
                    )
                )
            ),
            "incremental_sibling_recovery_over_formal": mean_std(
                values(
                    (
                        "full_r36_positive_oracle",
                        "incremental_sibling_recovery_over_formal",
                    )
                )
            ),
        },
        "error_entity_union": float(len(set().union(*error_sets))),
        "error_entity_intersection": float(
            len(set.intersection(*error_sets))
        ),
        "visible_error_entity_union": float(
            len(set().union(*visible_error_sets))
        ),
        "visible_error_entity_intersection": float(
            len(set.intersection(*visible_error_sets))
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    seeds = parse_integer_list(args.seeds, name="seeds")
    if tuple(seeds) != PREREGISTERED_SEEDS:
        raise ValueError(
            "This frozen Oracle requires the preregistered seeds "
            f"{PREREGISTERED_SEEDS}."
        )
    top_ks = parse_integer_list(args.top_k, name="top-k")
    if top_ks != sorted(set(top_ks)) or top_ks[-1] > 16:
        raise ValueError("Fine Top-K must be unique, sorted, and at most 16.")
    if int(args.batch_size) <= 0:
        raise ValueError("batch-size must be positive.")
    device = select_device(args.device)
    joint_config_path = resolve_path(args.joint_config, root)
    evidence_config_path = resolve_path(args.evidence_config, root)
    joint_config = load_joint_subtype_config(joint_config_path)
    taxonomy_path = resolve_path(joint_config.taxonomy, root)
    taxonomy = SubtypeTaxonomy.from_file(taxonomy_path)

    train_expanded_path = resolve_path(
        joint_config.data.train_expanded_cache,
        root,
    )
    dev_expanded_path = resolve_path(
        joint_config.data.dev_expanded_cache,
        root,
    )
    formal_path = resolve_path(
        joint_config.data.dev_formal_predictions,
        root,
    )
    train_provider = FrozenM33AFeatureProvider.from_path(
        train_expanded_path
    )
    formal_payload, dev_provider = load_frozen_dev_contract(
        formal_predictions_path=formal_path,
        expanded_cache_path=dev_expanded_path,
        taxonomy=taxonomy,
    )
    train_source = resolve_path(joint_config.data.train_source, root)
    dev_source = resolve_path(joint_config.data.dev_source, root)
    bank, prototype_report = build_train_visual_prototypes(
        source_path=train_source,
        provider=train_provider,
        taxonomy=taxonomy,
    )
    del train_provider
    gc.collect()
    fine_rankings, fine_report = extract_fine_region_rankings(
        evidence_config_path=evidence_config_path,
        expected_expanded_path=dev_expanded_path,
        root=root,
        device=device,
        batch_size=int(args.batch_size),
    )
    predictions_by_seed, f2_report = load_f2_predictions(
        joint_config=joint_config,
        taxonomy=taxonomy,
        root=root,
        device=device,
        seeds=seeds,
    )
    if (
        f2_report["formal_coarse_prediction_sha256"]
        != formal_payload["metadata"]["coarse_prediction_sha256"]
    ):
        raise ValueError("F2 and Oracle formal predictions differ.")
    dev_records = read_fine_conll(
        dev_source,
        taxonomy,
        require_all_subtypes=True,
    )
    dev_gold = fine_gold_by_record(dev_records, taxonomy)
    per_seed = [
        analyze_seed(
            seed=seed,
            formal_payload=formal_payload,
            provider=dev_provider,
            fine_gold=dev_gold,
            fine_rankings=fine_rankings,
            subtype_predictions=predictions_by_seed[seed],
            bank=bank,
            taxonomy=taxonomy,
            top_ks=top_ks,
        )
        for seed in seeds
    ]
    payload = {
        "metadata": {
            "kind": "fmnerg_subtype_region_evidence_oracle",
            "format_version": 1,
            "split": "dev",
            "scope": (
                "formal GMNER-correct but F2 subtype-wrong predictions"
            ),
            "interpretation": (
                "A deterministic Train-prototype probe of visual subtype "
                "separability; not deployable accuracy and not a trained model"
            ),
            "candidate_semantics": (
                "Only official gold-positive real R36 regions count as clean "
                "visual evidence; Fine Top-K controls whether such evidence "
                "is reachable without using a non-gold region"
            ),
            "seeds": seeds,
            "top_k": top_ks,
            "training_performed": False,
            "gradient_updates": 0,
            "formal_stage1_mutated": False,
            "formal_region_mutated": False,
            "test_accessed": False,
        },
        "artifacts": {
            "joint_config": {
                "path": str(joint_config_path),
                "sha256": sha256_file(joint_config_path),
            },
            "evidence_config": {
                "path": str(evidence_config_path),
                "sha256": sha256_file(evidence_config_path),
            },
            "taxonomy": {
                "path": str(taxonomy_path),
                "sha256": taxonomy.source_sha256,
            },
            "dev_source": {
                "path": str(dev_source),
                "sha256": sha256_file(dev_source),
            },
            "formal_predictions": {
                "path": str(formal_path),
                "sha256": sha256_file(formal_path),
                "coarse_prediction_sha256": formal_payload["metadata"][
                    "coarse_prediction_sha256"
                ],
                "coarse_metrics": formal_payload["metadata"][
                    "coarse_metrics"
                ],
            },
            "prototype_bank": prototype_report,
            "fine_adapter": fine_report,
            "f2": f2_report,
        },
        "aggregate": aggregate_seed_results(
            per_seed,
            top_ks=top_ks,
        ),
        "per_seed": per_seed,
    }
    output = resolve_path(args.output, root)
    save_json_atomic(payload, output)
    compact = {
        "output": str(output),
        "test_accessed": False,
        "prototype_subtypes_available": prototype_report[
            "prototype_subtypes_available"
        ],
        "aggregate": payload["aggregate"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
