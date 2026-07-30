"""Build offline record-level candidates from a frozen Stage-1 checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.ops import box_iou
from tqdm.auto import tqdm
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.constants import (
    DEFAULT_LABEL2ID,
    ENTITY_TYPE2ID,
    ID2ENTITY_TYPE,
    IGNORE_INDEX,
)
from gmner.data import (
    GMNERCollator,
    MMNERJsonDataset,
    TextGraphBuilder,
    load_word_aligned_tokenizer,
    validate_model_input_length,
)
from gmner.data.graph_builders import GraphBuilderConfig
from gmner.data.formal_candidate_anchor import (
    cache_record_id,
    load_formal_anchor_cache,
    stage1_entities_from_anchor,
)
from gmner.data.record_candidate_dataset import CACHE_FORMAT_VERSION
from gmner.engine.utils import move_batch_to_device
from gmner.fmnerg.candidate_contract import (
    FINE_CANDIDATE_SCHEMA,
    FINE_CANDIDATE_SCHEMA_VERSION,
    validate_fine_candidate_record,
)
from gmner.fmnerg.metrics import (
    end_to_end_fine_metrics,
    fine_entities_from_bio_tags,
)
from gmner.fmnerg.taxonomy import (
    SubtypeTaxonomy,
    bind_config_taxonomy_fingerprint,
    validate_taxonomy_fingerprint,
)
from gmner.knowledge.region_compatibility import compatibility_score
from gmner.models import GMNERModel
from gmner.utils.candidate_decoding import (
    bio_constraint_masks,
    build_span_candidates,
    extract_crf_parameters,
    k_best_viterbi_decode,
)
from gmner.utils.io import maybe_convert_conll
from gmner.utils.metrics import (
    extract_entities_from_word_labels,
    word_labels_from_subwords,
)


SOURCE2ID = {"stage1": 0, "viterbi": 1, "kbest": 2, "perturbation": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=["train", "dev", "test"], required=True)
    parser.add_argument(
        "--input-file",
        default=None,
        help="Optional split-file override, used for held-out OOF records.",
    )
    parser.add_argument(
        "--oof-fold-id",
        type=int,
        default=None,
        help="Mark this cache as predictions from one held-out OOF fold.",
    )
    parser.add_argument(
        "--artifact-identity",
        default=None,
        help="Optional immutable experiment identity for regenerated caches.",
    )
    parser.add_argument(
        "--regeneration-authorization-sha256",
        default=None,
        help="Authorization fingerprint paired with --artifact-identity.",
    )
    parser.add_argument(
        "--regeneration-fold-id",
        type=int,
        default=None,
        help="Regeneration fold identity for every cache split in one fold chain.",
    )
    parser.add_argument(
        "--regeneration-experiment-id",
        default=None,
        help="Preregistered regeneration experiment identifier.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--k-best", type=int, default=6)
    parser.add_argument("--max-span-candidates", type=int, default=12)
    parser.add_argument("--top-m-types", type=int, default=3)
    parser.add_argument("--boundary-shift", type=int, default=1)
    parser.add_argument("--boundary-penalty", type=float, default=0.25)
    parser.add_argument("--max-span-length", type=int, default=10)
    parser.add_argument(
        "--enforce-bio-constraints",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--inject-gold-types",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Training-only diagnostic; formal evaluation must leave this disabled.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument(
        "--max-regions",
        type=int,
        default=None,
        help=(
            "Diagnostic override for the number of VinVL proposals. The primary "
            "config is left unchanged; use a separate output path."
        ),
    )
    parser.add_argument(
        "--formal-anchor-cache",
        default=None,
        help=(
            "Optional smaller-region candidate cache whose frozen Stage1 "
            "span/type predictions are authoritative. Use this when building "
            "an expanded cache such as R36 from a formal R16 cache."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help=(
            "Required to build a fine_hierarchical Test cache after the "
            "architecture and protocol are frozen."
        ),
    )
    return parser.parse_args()


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def span_mask_from_words(
    start: int,
    end: int,
    word_ids: list[int | None],
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    mask = torch.zeros_like(attention_mask, dtype=torch.float32)
    for token_index, word_id in enumerate(word_ids[: attention_mask.numel()]):
        if word_id is not None and start <= int(word_id) < end:
            mask[token_index] = 1.0
    return mask


def masked_log_softmax(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = logits.float().masked_fill(~mask.bool(), -1e4)
    return F.log_softmax(masked, dim=-1).masked_fill(~mask.bool(), -20.0)


def grounding_scores(
    model: GMNERModel,
    outputs: dict[str, torch.Tensor],
    batch: dict,
    index: int,
    target_mask: torch.Tensor,
    config,
) -> torch.Tensor:
    token_states = outputs["pre_prototype_fused_tokens"][index : index + 1]
    target_mask = target_mask.to(device=token_states.device, dtype=token_states.dtype)
    query = (token_states * target_mask.unsqueeze(-1)).sum(dim=1) / target_mask.sum(
        dim=1, keepdim=True
    ).clamp_min(1.0)
    logits = model.grounding_head(
        query=query,
        image_nodes=outputs["image_nodes"][index : index + 1],
        image_mask=batch["region_mask"][index : index + 1],
    )[0]
    detector_weight = float(getattr(config.model, "region_score_prior_weight", 0.0))
    if detector_weight:
        detector = batch["region_scores"][index].to(logits).clamp(1e-4, 1.0)
        bias = detector.log() * detector_weight
        if config.data.add_null_region:
            bias = bias.clone()
            bias[-1] = 0.0
        logits = logits + bias
    null_bias = float(getattr(config.model, "grounding_null_logit_bias", 0.0))
    if null_bias and config.data.add_null_region:
        logits = logits.clone()
        logits[-1] += null_bias
    return logits


def compatibility_tensor(
    type_ids: torch.Tensor,
    labels: list[str],
    attributes: list[str],
    num_regions: int,
    *,
    null_index: int,
) -> torch.Tensor:
    result = torch.zeros(type_ids.numel(), num_regions, dtype=torch.float32)
    for type_index, type_id in enumerate(type_ids.tolist()):
        type_name = ID2ENTITY_TYPE.get(int(type_id), "OTHER")
        for region_index in range(min(num_regions, len(labels))):
            if region_index == null_index:
                continue
            attribute = attributes[region_index] if region_index < len(attributes) else ""
            result[type_index, region_index] = compatibility_score(
                type_name, labels[region_index], attribute
            )
    return result


def positive_regions(
    entity: dict,
    gt_boxes_by_name: dict,
    region_boxes: torch.Tensor,
    region_mask: torch.Tensor,
    *,
    null_index: int,
    iou_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    result = torch.zeros_like(region_mask, dtype=torch.bool)
    iou_targets = torch.zeros_like(region_mask, dtype=torch.float32)
    gt_boxes = gt_boxes_by_name.get(str(entity["text"]).strip().lower(), [])
    visible = bool(gt_boxes)
    if not visible:
        result[null_index] = True
        iou_targets[null_index] = 1.0
        return result, iou_targets, False
    visible_mask = region_mask.bool().clone()
    visible_mask[null_index] = False
    indices = torch.nonzero(visible_mask, as_tuple=False).squeeze(-1)
    if indices.numel() == 0:
        return result, iou_targets, True
    gold = torch.tensor(gt_boxes, dtype=region_boxes.dtype, device=region_boxes.device)
    ious = box_iou(gold, region_boxes[indices]).max(dim=0).values
    iou_targets[indices] = ious.clamp(0.0, 1.0)
    result[indices[ious >= float(iou_threshold)]] = True
    return result, iou_targets, True


def normalized_geometry(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    height = float(image_size[0].item()) if image_size.numel() > 0 else 0.0
    width = float(image_size[1].item()) if image_size.numel() > 1 else 0.0
    scale = boxes.new_tensor([max(width, 1.0), max(height, 1.0), max(width, 1.0), max(height, 1.0)])
    return (boxes / scale).clamp(0.0, 1.5)


def f1(correct: int, predicted: int, gold: int) -> dict[str, float]:
    precision = correct / max(predicted, 1)
    recall = correct / max(gold, 1)
    score = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"precision": precision, "recall": recall, "f1": score}


def count_matches(predictions: list[dict], gold: list[dict]) -> dict[str, int]:
    matched = {"span": set(), "mner": set(), "eeg": set(), "gmner": set()}
    for prediction in predictions:
        pred_span = tuple(prediction["span"])
        for name in matched:
            for index, target in enumerate(gold):
                if index in matched[name] or tuple(target["span"]) != pred_span:
                    continue
                type_ok = int(prediction["type_id"]) == int(target["type_id"])
                region_ok = int(prediction["region_index"]) in set(
                    target["region_positive_indices"]
                )
                if (
                    name == "span"
                    or (name == "mner" and type_ok)
                    or (name == "eeg" and region_ok)
                    or (name == "gmner" and type_ok and region_ok)
                ):
                    matched[name].add(index)
                    break
    return {name: len(indices) for name, indices in matched.items()}


@torch.no_grad()
def build(args: argparse.Namespace) -> dict:
    root = Path(__file__).resolve().parents[1]
    regeneration_values = (
        args.artifact_identity,
        args.regeneration_authorization_sha256,
        args.regeneration_fold_id,
        args.regeneration_experiment_id,
    )
    if any(value is not None for value in regeneration_values) and not all(
        value is not None for value in regeneration_values
    ):
        raise ValueError(
            "Regenerated cache identity requires artifact identity, authorization "
            "SHA256, fold id, and experiment id together."
        )
    if args.regeneration_authorization_sha256 is not None and len(
        str(args.regeneration_authorization_sha256)
    ) != 64:
        raise ValueError("Invalid regeneration authorization SHA256.")
    if args.oof_fold_id is not None and (
        args.split != "train" or args.input_file is None
    ):
        raise ValueError(
            "--oof-fold-id requires --split train and an explicit --input-file."
        )
    config = load_config(args.config)
    label_schema = str(
        getattr(config.data, "label_schema", "coarse")
    )
    fine_schema = label_schema == FINE_CANDIDATE_SCHEMA
    if fine_schema and args.split == "test" and not args.allow_test:
        raise ValueError(
            "Fine-hierarchical Test cache access requires --allow-test."
        )
    taxonomy = None
    if fine_schema:
        if not bool(
            getattr(config.model, "use_fine_subtype_head", False)
        ):
            raise ValueError(
                "fine_hierarchical cache requires the formal subtype head."
            )
        taxonomy_path = resolve_path(
            config.data.subtype_taxonomy,
            root,
        )
        taxonomy = SubtypeTaxonomy.from_file(taxonomy_path)
        config.data.subtype_taxonomy = str(taxonomy_path)
        bind_config_taxonomy_fingerprint(config.data, taxonomy)
        config.model.num_subtypes = taxonomy.num_subtypes
    elif bool(getattr(config.model, "use_fine_subtype_head", False)):
        raise ValueError(
            "The formal subtype head requires "
            "data.label_schema=fine_hierarchical."
        )
    if args.max_regions is not None:
        if int(args.max_regions) <= 0:
            raise ValueError("--max-regions must be a positive integer.")
        config.data.max_regions = int(args.max_regions)
    checkpoint_path = resolve_path(args.checkpoint, root)
    output_path = resolve_path(args.output, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source_path = resolve_path(
        args.input_file
        or {
            "train": config.data.train_file,
            "dev": config.data.dev_file,
            "test": config.data.test_file,
        }[args.split],
        root,
    )
    data_path = maybe_convert_conll(source_path, resolve_path(config.runtime.output_dir, root))
    device_name = args.device or config.runtime.device
    device = torch.device(
        device_name if str(device_name).startswith("cuda") and torch.cuda.is_available() else "cpu"
    )

    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    backbone_config = AutoConfig.from_pretrained(config.model.text_model_name)
    validate_model_input_length(
        tokenizer,
        backbone_config,
        config.data.max_length,
    )
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=config.data.use_dependency_graph,
            dependency_backend=config.data.dependency_backend,
            dependency_model=config.data.dependency_model,
            window_size=config.data.graph_window_size,
        )
    )
    dataset = MMNERJsonDataset(
        jsonl_path=str(data_path),
        image_dir=str(resolve_path(config.data.image_dir, root)),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=True,
        expand_entities_for_grounding=False,
        image_feature_dir=str(resolve_path(config.data.image_feature_dir, root)),
        image_annotation_dir=str(resolve_path(config.data.image_annotation_dir, root)),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        region_min_score=config.data.region_min_score,
        subtype_taxonomy=taxonomy,
    )
    selected_dataset = dataset
    if args.max_records is not None:
        selected_dataset = Subset(dataset, range(min(len(dataset), max(0, args.max_records))))
    loader = DataLoader(
        selected_dataset,
        batch_size=args.batch_size or config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=GMNERCollator(tokenizer=tokenizer),
    )

    model = GMNERModel(config=config, num_labels=len(DEFAULT_LABEL2ID))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if taxonomy is not None:
        validate_taxonomy_fingerprint(
            dict(checkpoint.get("model_metadata") or {}),
            taxonomy,
            artifact_name="Stage1-F checkpoint",
        )
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=taxonomy is not None,
    )
    model.to(device).eval()

    id2label = {value: key for key, value in DEFAULT_LABEL2ID.items()}
    transitions, start_transitions, end_transitions, transition_source = extract_crf_parameters(
        model.ner_head.crf,
        len(id2label),
        device=device,
        dtype=torch.float32,
    )
    allowed_start = allowed_transitions = None
    if args.enforce_bio_constraints:
        allowed_start, allowed_transitions = bio_constraint_masks(
            id2label, len(id2label), device=device
        )

    candidate_spec = {
        "cache_format_version": CACHE_FORMAT_VERSION,
        "text_model_name": str(config.model.text_model_name),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(getattr(tokenizer, "is_fast", False)),
        "max_length": int(config.data.max_length),
        "k_best": int(args.k_best),
        "max_span_candidates": int(args.max_span_candidates),
        "top_m_types": min(4, max(1, int(args.top_m_types))),
        "boundary_shift": int(args.boundary_shift),
        "boundary_penalty": float(args.boundary_penalty),
        "max_span_length": int(args.max_span_length),
        "enforce_bio_constraints": bool(args.enforce_bio_constraints),
        "inject_gold_types": bool(args.inject_gold_types),
        "preserve_stage1_type": True,
        "grounding_iou_threshold": float(config.data.grounding_iou_threshold),
        "max_regions": int(config.data.max_regions),
        **(
            {
                "label_schema": label_schema,
                "fine_schema_version": FINE_CANDIDATE_SCHEMA_VERSION,
                "taxonomy_sha256": taxonomy.source_sha256,
            }
            if taxonomy is not None
            else {}
        ),
    }
    stage1_checkpoint_sha256 = sha256_file(checkpoint_path)
    data_source_sha256 = sha256_file(source_path)
    formal_anchor_records: list[dict] | None = None
    formal_anchor_provenance: dict | None = None
    if args.formal_anchor_cache:
        anchor_path = resolve_path(args.formal_anchor_cache, root)
        formal_anchor_records, formal_anchor_provenance = load_formal_anchor_cache(
            anchor_path,
            stage1_checkpoint_sha256=stage1_checkpoint_sha256,
            data_source_sha256=data_source_sha256,
            expanded_candidate_spec=candidate_spec,
            expected_label_schema=(
                FINE_CANDIDATE_SCHEMA if taxonomy is not None else None
            ),
            expected_taxonomy_sha256=(
                taxonomy.source_sha256 if taxonomy is not None else None
            ),
        )
    records: list[dict] = []
    source_counts = Counter()
    gold_count = span_covered = type_covered = region_covered = triple_covered = 0
    visible_gold_count = visible_region_covered = 0
    visible_joint_span_region_covered = 0
    stage1_gold_lost = 0
    bypass_predicted = bypass_gold = 0
    bypass_correct = Counter()
    fine_bypass_correct = Counter()

    for batch in tqdm(loader, desc=f"Caching {args.split} records"):
        batch = move_batch_to_device(batch, device)
        outputs = model(batch)
        labels = batch["ner_labels"]
        decoded = model.ner_head.decode(
            outputs["ner_logits"],
            batch["attention_mask"],
            valid_mask=labels != IGNORE_INDEX,
        )
        for index, metadata in enumerate(batch.get("metadata", [])):
            tokens = list(metadata.get("tokens") or [])
            word_ids = list(metadata.get("word_ids") or [])
            word_gold = word_labels_from_subwords(labels[index].tolist(), word_ids)
            if formal_anchor_records is None:
                word_pred = word_labels_from_subwords(
                    decoded[index].tolist(), word_ids
                )
                pred_entities = extract_entities_from_word_labels(
                    word_pred, tokens, id2label
                )
            else:
                if len(records) >= len(formal_anchor_records):
                    raise ValueError(
                        "Expanded source contains more records than its formal anchor."
                    )
                anchor_record = formal_anchor_records[len(records)]
                current_id = str(
                    metadata.get("record_id", metadata.get("sample_id", ""))
                )
                if cache_record_id(anchor_record) != current_id:
                    raise ValueError(
                        "Formal anchor record order differs from the expanded "
                        f"source at position {len(records)}: "
                        f"formal={cache_record_id(anchor_record)!r}, "
                        f"expanded={current_id!r}."
                    )
                pred_entities = stage1_entities_from_anchor(
                    anchor_record,
                    tokens,
                )
            gold_entities_raw = extract_entities_from_word_labels(word_gold, tokens, id2label)
            gold_fine_by_span: dict[tuple[int, int], dict] = {}
            if taxonomy is not None:
                fine_tags = metadata.get("fine_ner_tags")
                if not isinstance(fine_tags, list):
                    raise ValueError(
                        "fine_hierarchical cache requires fine_ner_tags in "
                        f"record {metadata.get('record_id')}."
                    )
                gold_fine_entities = fine_entities_from_bio_tags(
                    tokens=tokens,
                    coarse_tags=word_gold,
                    fine_tags=fine_tags,
                    taxonomy=taxonomy,
                    coarse_id2label=id2label,
                )
                gold_fine_by_span = {
                    tuple(map(int, item["span"])): item
                    for item in gold_fine_entities
                }
            stage1_spans = {(int(item["start"]), int(item["end"])) for item in pred_entities}
            stage1_types_by_span = {
                (int(item["start"]), int(item["end"])): ENTITY_TYPE2ID[str(item["type"])]
                for item in pred_entities
            }
            stage1_subtypes_by_span = {
                (int(item["start"]), int(item["end"])): int(
                    item["subtype_id"]
                )
                for item in pred_entities
                if item.get("subtype_id") is not None
            }

            valid_subwords = batch["attention_mask"][index].bool() & labels[index].ne(IGNORE_INDEX)
            compact_positions: list[int] = []
            compact_word_indices: list[int] = []
            seen_words: set[int] = set()
            for position in torch.nonzero(valid_subwords, as_tuple=False).squeeze(-1).tolist():
                word_id = word_ids[position] if position < len(word_ids) else None
                if word_id is None or int(word_id) in seen_words:
                    continue
                seen_words.add(int(word_id))
                compact_positions.append(int(position))
                compact_word_indices.append(int(word_id))
            sequences = []
            if compact_positions:
                emissions = outputs["base_ner_logits"][
                    index, torch.tensor(compact_positions, device=device)
                ].float()
                sequences = k_best_viterbi_decode(
                    emissions,
                    k=max(1, int(args.k_best)),
                    transitions=transitions,
                    start_transitions=start_transitions,
                    end_transitions=end_transitions,
                    allowed_start=allowed_start,
                    allowed_transitions=allowed_transitions,
                )
            candidates = build_span_candidates(
                sequences,
                word_indices=compact_word_indices,
                id2label=id2label,
                num_words=max(compact_word_indices, default=-1) + 1,
                max_candidates=max(1, int(args.max_span_candidates)),
                boundary_shift=max(0, int(args.boundary_shift)),
                boundary_penalty=float(args.boundary_penalty),
                max_span_length=max(1, int(args.max_span_length)),
                required_spans=sorted(stage1_spans),
            )
            candidate_boundaries = [candidate.boundary for candidate in candidates]
            candidate_index = {boundary: offset for offset, boundary in enumerate(candidate_boundaries)}
            stage1_gold_lost += sum(
                int((int(entity["start"]), int(entity["end"])) in stage1_spans and (int(entity["start"]), int(entity["end"])) not in candidate_index)
                for entity in gold_entities_raw
            )

            region_mask = batch["region_mask"][index].bool()
            region_boxes = batch["region_boxes"][index].float()
            region_nodes = outputs["image_nodes"][index].float()
            num_regions = region_nodes.size(0)
            null_index = config.data.max_regions if config.data.add_null_region else num_regions - 1
            null_mask = torch.zeros(num_regions, dtype=torch.bool, device=device)
            null_mask[null_index] = bool(config.data.add_null_region)
            visible_mask = region_mask & ~null_mask
            image_global = (
                region_nodes[visible_mask].mean(dim=0)
                if visible_mask.any()
                else torch.zeros(region_nodes.size(-1), device=device)
            )
            labels_for_regions = list(metadata.get("region_object_labels") or [])
            attrs_for_regions = list(metadata.get("region_object_attributes") or [])

            span_features = []
            type_candidates = []
            type_scores = []
            region_scores = []
            compatibilities = []
            source_ids = []
            candidate_masks: list[torch.Tensor] = []
            for candidate in candidates:
                mask = span_mask_from_words(
                    candidate.start,
                    candidate.end,
                    word_ids,
                    batch["attention_mask"][index],
                ).unsqueeze(0)
                candidate_masks.append(mask[0])
                token_states = outputs["pre_prototype_fused_tokens"][index]
                feature = (token_states * mask[0].unsqueeze(-1)).sum(dim=0) / mask.sum().clamp_min(1.0)
                span_features.append(feature)
                logits = model._span_type_logits_from_ner(
                    outputs["base_ner_logits"][index : index + 1], mask.to(device)
                )[0]
                top_count = candidate_spec["top_m_types"]
                selected_types = logits.topk(top_count).indices
                stage1_type_id = stage1_types_by_span.get(candidate.boundary)
                if (
                    stage1_type_id is not None
                    and stage1_type_id not in selected_types.tolist()
                ):
                    selected_types[-1] = int(stage1_type_id)
                gold_match = next(
                    (
                        item for item in gold_entities_raw
                        if (int(item["start"]), int(item["end"])) == candidate.boundary
                    ),
                    None,
                )
                if args.inject_gold_types and args.split == "train" and gold_match is not None:
                    gold_type_id = ENTITY_TYPE2ID[str(gold_match["type"])]
                    if gold_type_id not in selected_types.tolist():
                        replacement = next(
                            (
                                slot
                                for slot in range(selected_types.numel() - 1, -1, -1)
                                if int(selected_types[slot].item()) != stage1_type_id
                            ),
                            selected_types.numel() - 1,
                        )
                        selected_types[replacement] = gold_type_id
                type_candidates.append(selected_types)
                type_scores.append(F.log_softmax(logits.float(), dim=-1)[selected_types])
                raw_region = grounding_scores(model, outputs, batch, index, mask, config)
                region_scores.append(masked_log_softmax(raw_region, region_mask))
                compatibilities.append(
                    compatibility_tensor(
                        selected_types.detach().cpu(),
                        labels_for_regions,
                        attrs_for_regions,
                        num_regions,
                        null_index=null_index,
                    ).to(device)
                )
                effective_source = "stage1" if candidate.preserve_stage1 else candidate.source
                source_ids.append(SOURCE2ID.get(effective_source, SOURCE2ID["kbest"]))
                source_counts[effective_source] += 1

            span_count = len(candidates)
            top_m = candidate_spec["top_m_types"]
            hidden_size = region_nodes.size(-1)
            if span_count:
                span_features_tensor = torch.stack(span_features)
                type_candidates_tensor = torch.stack(type_candidates)
                type_scores_tensor = torch.stack(type_scores)
                region_scores_tensor = torch.stack(region_scores)
                compatibility_values = torch.stack(compatibilities)
                proposal_scores = F.log_softmax(
                    torch.tensor([item.score for item in candidates], device=device), dim=0
                )
            else:
                span_features_tensor = torch.empty(0, hidden_size, device=device)
                type_candidates_tensor = torch.empty(0, top_m, dtype=torch.long, device=device)
                type_scores_tensor = torch.empty(0, top_m, device=device)
                region_scores_tensor = torch.empty(0, num_regions, device=device)
                compatibility_values = torch.empty(0, top_m, num_regions, device=device)
                proposal_scores = torch.empty(0, device=device)

            compatibility_weight = float(
                getattr(config.model, "region_object_compatibility_weight", 0.0)
            )
            fixed_type_ids = torch.full(
                (span_count,), -1, dtype=torch.long, device=device
            )
            base_region_scores = torch.full(
                (span_count, num_regions), -20.0, device=device
            )
            base_region_indices = torch.full(
                (span_count,), -1, dtype=torch.long, device=device
            )
            for row, candidate in enumerate(candidates):
                fixed_type_id = stage1_types_by_span.get(
                    candidate.boundary,
                    int(type_candidates_tensor[row, 0].item()),
                )
                fixed_type_ids[row] = int(fixed_type_id)
                type_hits = torch.nonzero(
                    type_candidates_tensor[row].eq(int(fixed_type_id)),
                    as_tuple=False,
                ).squeeze(-1)
                type_slot = int(type_hits[0].item()) if type_hits.numel() else 0
                adjusted = (
                    region_scores_tensor[row]
                    + compatibility_weight * compatibility_values[row, type_slot]
                )
                fixed_scores = masked_log_softmax(adjusted, region_mask)
                base_region_scores[row] = fixed_scores
                base_region_indices[row] = int(fixed_scores.argmax().item())

            subtype_raw_logits = torch.empty(
                span_count,
                taxonomy.num_subtypes if taxonomy is not None else 0,
                dtype=torch.float32,
                device=device,
            )
            fixed_subtype_ids = torch.full(
                (span_count,),
                IGNORE_INDEX,
                dtype=torch.long,
                device=device,
            )
            subtype_confidence = torch.zeros(
                span_count,
                dtype=torch.float32,
                device=device,
            )
            subtype_margin = torch.zeros_like(subtype_confidence)
            subtype_entropy = torch.zeros_like(subtype_confidence)
            if taxonomy is not None and span_count:
                subtype_text_states = outputs["base_text_nodes"][
                    index : index + 1
                ].expand(span_count, -1, -1)
                subtype_outputs = model.score_fine_subtypes(
                    token_states=subtype_text_states,
                    target_mask=torch.stack(candidate_masks).to(device),
                    parent_ids=fixed_type_ids,
                )
                subtype_raw_logits = subtype_outputs[
                    "raw_logits"
                ].float()
                fixed_subtype_ids = subtype_outputs[
                    "predicted_subtype_ids"
                ]
                subtype_confidence = subtype_outputs["confidence"].float()
                subtype_margin = subtype_outputs["margin"].float()
                subtype_entropy = subtype_outputs["entropy"].float()
                for row, candidate in enumerate(candidates):
                    anchored_subtype = stage1_subtypes_by_span.get(
                        candidate.boundary
                    )
                    if (
                        anchored_subtype is not None
                        and anchored_subtype
                        != int(fixed_subtype_ids[row].item())
                    ):
                        raise ValueError(
                            "Expanded cache changed the formal R16 subtype "
                            f"for record {metadata.get('record_id')} span "
                            f"{candidate.boundary}."
                        )

            gold_span_mask = torch.zeros(span_count, dtype=torch.bool, device=device)
            gold_type_mask = torch.zeros(span_count, top_m, dtype=torch.bool, device=device)
            gold_region_mask = torch.zeros(span_count, num_regions, dtype=torch.bool, device=device)
            positive_triples = torch.zeros(
                span_count, top_m, num_regions, dtype=torch.bool, device=device
            )
            region_iou_targets = torch.zeros(
                span_count, num_regions, dtype=torch.float32, device=device
            )
            visibility_targets = torch.full((span_count,), -1.0, device=device)
            gold_subtype_ids = torch.full(
                (span_count,),
                IGNORE_INDEX,
                dtype=torch.long,
                device=device,
            )
            gold_entities: list[dict] = []
            gt_boxes_by_name = metadata.get("gt_boxes_by_name") or {}
            for entity in gold_entities_raw:
                boundary = (int(entity["start"]), int(entity["end"]))
                fine_entity = gold_fine_by_span.get(boundary)
                if taxonomy is not None and fine_entity is None:
                    raise ValueError(
                        "Coarse and fine gold spans differ in record "
                        f"{metadata.get('record_id')}: {boundary}."
                    )
                region_positive, region_iou, visible = positive_regions(
                    entity,
                    gt_boxes_by_name,
                    region_boxes,
                    region_mask,
                    null_index=null_index,
                    iou_threshold=config.data.grounding_iou_threshold,
                )
                type_id = ENTITY_TYPE2ID[str(entity["type"])]
                gold_entities.append(
                    {
                        "span": list(boundary),
                        "type_id": int(type_id),
                        **(
                            {
                                "subtype": str(fine_entity["subtype"]),
                                "subtype_id": int(
                                    fine_entity["subtype_id"]
                                ),
                            }
                            if fine_entity is not None
                            else {}
                        ),
                        "text": str(entity["text"]),
                        "visible": bool(visible),
                        "region_positive_indices": torch.nonzero(region_positive, as_tuple=False).squeeze(-1).tolist(),
                    }
                )
                gold_count += 1
                visible_gold_count += int(visible)
                visible_region_covered += int(
                    visible and region_positive.any().item()
                )
                if boundary not in candidate_index:
                    continue
                span_covered += 1
                row = candidate_index[boundary]
                gold_span_mask[row] = True
                visibility_targets[row] = float(visible)
                type_hits = type_candidates_tensor[row].eq(type_id)
                gold_type_mask[row] = type_hits
                gold_region_mask[row] = region_positive
                region_iou_targets[row] = region_iou
                if fine_entity is not None:
                    gold_subtype_ids[row] = int(
                        fine_entity["subtype_id"]
                    )
                type_covered += int(type_hits.any().item())
                region_covered += int(region_positive.any().item())
                visible_joint_span_region_covered += int(
                    visible and region_positive.any().item()
                )
                if type_hits.any() and region_positive.any():
                    positive_triples[row] = type_hits[:, None] & region_positive[None, :]
                    triple_covered += 1

            stage1_predictions: list[dict] = []
            for entity in pred_entities:
                boundary = (int(entity["start"]), int(entity["end"]))
                row = candidate_index[boundary]
                stage1_predictions.append(
                    {
                        "span": list(boundary),
                        "type_id": int(fixed_type_ids[row].item()),
                        **(
                            {
                                "subtype_id": int(
                                    fixed_subtype_ids[row].item()
                                )
                            }
                            if taxonomy is not None
                            else {}
                        ),
                        "region_index": int(base_region_indices[row].item()),
                    }
                )

            matches = count_matches(stage1_predictions, gold_entities)
            bypass_predicted += len(stage1_predictions)
            bypass_gold += len(gold_entities)
            bypass_correct.update(matches)
            if taxonomy is not None:
                fine_metrics = end_to_end_fine_metrics(
                    [
                        {
                            "predictions": stage1_predictions,
                            "gold_entities": gold_entities,
                        }
                    ]
                )
                fine_bypass_correct["fine_mner"] += int(
                    fine_metrics["fine_mner_correct"]
                )
                fine_bypass_correct["fmnerg"] += int(
                    fine_metrics["fmnerg_correct"]
                )

            geometry = normalized_geometry(region_boxes, batch["image_sizes"][index])
            record = {
                "span_candidates": torch.tensor(
                    candidate_boundaries, dtype=torch.long
                ).reshape(-1, 2),
                "span_mask": torch.ones(span_count, dtype=torch.bool),
                "span_features": span_features_tensor.detach().cpu().to(torch.float16),
                "span_base_scores": proposal_scores.detach().cpu().to(torch.float32),
                "span_source_ids": torch.tensor(source_ids, dtype=torch.long),
                "span_lengths": torch.tensor(
                    [end - start for start, end in candidate_boundaries], dtype=torch.float32
                ),
                "type_candidates": type_candidates_tensor.detach().cpu().long(),
                "type_base_scores": type_scores_tensor.detach().cpu().float(),
                "type_mask": torch.ones(span_count, top_m, dtype=torch.bool),
                "region_features": region_nodes.detach().cpu().to(torch.float16),
                "region_boxes": region_boxes.detach().cpu().float(),
                "region_geometry": geometry.detach().cpu().float(),
                "region_detector_scores": batch["region_scores"][index].detach().cpu().float(),
                "region_base_scores": region_scores_tensor.detach().cpu().float(),
                "type_region_compatibility": compatibility_values.detach().cpu().float(),
                "fixed_type_ids": fixed_type_ids.detach().cpu(),
                **(
                    {
                        "fixed_parent_ids": fixed_type_ids.detach().cpu(),
                        "subtype_raw_logits": (
                            subtype_raw_logits.detach().cpu().float()
                        ),
                        "fixed_subtype_ids": (
                            fixed_subtype_ids.detach().cpu()
                        ),
                        "subtype_confidence": (
                            subtype_confidence.detach().cpu().float()
                        ),
                        "subtype_margin": (
                            subtype_margin.detach().cpu().float()
                        ),
                        "subtype_entropy": (
                            subtype_entropy.detach().cpu().float()
                        ),
                        "gold_subtype_ids": (
                            gold_subtype_ids.detach().cpu()
                        ),
                    }
                    if taxonomy is not None
                    else {}
                ),
                "base_region_indices": base_region_indices.detach().cpu(),
                "base_region_scores": base_region_scores.detach().cpu().float(),
                "region_mask": region_mask.detach().cpu(),
                "region_is_null": null_mask.detach().cpu(),
                "image_global": image_global.detach().cpu().to(torch.float16),
                "gold_span_mask": gold_span_mask.detach().cpu(),
                "gold_type_mask": gold_type_mask.detach().cpu(),
                "gold_region_positive_mask": gold_region_mask.detach().cpu(),
                "positive_triple_mask": positive_triples.detach().cpu(),
                "region_iou_targets": region_iou_targets.detach().cpu(),
                "visibility_targets": visibility_targets.detach().cpu(),
                "metadata": {
                    "record_id": str(metadata.get("record_id", metadata.get("sample_id"))),
                    "text": str(metadata.get("text") or " ".join(tokens)),
                    "tokens": tokens,
                    "candidate_sources": [
                        "stage1" if item.preserve_stage1 else item.source for item in candidates
                    ],
                    "stage1_predictions": stage1_predictions,
                    "gold_entities": gold_entities,
                    "null_region_index": int(null_index),
                },
            }
            if taxonomy is not None:
                validate_fine_candidate_record(record, taxonomy)
            records.append(record)

    if (
        formal_anchor_records is not None
        and args.max_records is None
        and len(records) != len(formal_anchor_records)
    ):
        raise ValueError(
            "Expanded source and formal anchor contain different record counts: "
            f"expanded={len(records)}, formal={len(formal_anchor_records)}."
        )

    bypass_metrics = {
        name: f1(int(bypass_correct[name]), bypass_predicted, bypass_gold)
        for name in ("span", "mner", "eeg", "gmner")
    }
    if taxonomy is not None:
        bypass_metrics.update(
            {
                name: f1(
                    int(fine_bypass_correct[name]),
                    bypass_predicted,
                    bypass_gold,
                )
                for name in ("fine_mner", "fmnerg")
            }
        )
    summary = {
        "split": args.split,
        "records": len(records),
        "span_candidates": sum(int(item["span_mask"].sum()) for item in records),
        "candidate_sources": dict(source_counts),
        "gold_entities": gold_count,
        "span_coverage": span_covered / max(gold_count, 1),
        "typed_span_coverage": type_covered / max(gold_count, 1),
        "region_coverage": region_covered / max(gold_count, 1),
        "visible_region_oracle_recall": (
            visible_region_covered / max(visible_gold_count, 1)
        ),
        "visible_gold_entities": visible_gold_count,
        "visible_region_covered": visible_region_covered,
        "visible_joint_span_region_coverage": (
            visible_joint_span_region_covered
            / max(visible_gold_count, 1)
        ),
        "visible_joint_span_region_covered": (
            visible_joint_span_region_covered
        ),
        "triple_coverage": triple_covered / max(gold_count, 1),
        "stage1_gold_lost_by_final_candidates": stage1_gold_lost,
        "stage1_bypass": bypass_metrics,
    }
    cache_metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "split": args.split,
        "stage1_checkpoint": str(checkpoint_path.resolve()),
        "stage1_checkpoint_sha256": stage1_checkpoint_sha256,
        "data_source": str(source_path.resolve()),
        "data_source_sha256": data_source_sha256,
        "candidate_config": candidate_spec,
        "candidate_config_sha256": sha256_json(candidate_spec),
        "transition_source": transition_source,
        "source2id": SOURCE2ID,
        "hidden_size": int(config.model.hidden_size),
        "num_types": 4,
        "summary": summary,
        **(
            {
                "label_schema": FINE_CANDIDATE_SCHEMA,
                "fine_schema_version": FINE_CANDIDATE_SCHEMA_VERSION,
                **taxonomy.fingerprint_metadata(),
            }
            if taxonomy is not None
            else {}
        ),
    }
    if args.artifact_identity is not None:
        cache_metadata.update(
            {
                "artifact_identity": str(args.artifact_identity),
                "regeneration_authorization_sha256": str(
                    args.regeneration_authorization_sha256
                ),
                "regeneration_fold_id": int(args.regeneration_fold_id),
                "regeneration_experiment_id": str(
                    args.regeneration_experiment_id
                ),
            }
        )
    if formal_anchor_provenance is not None:
        cache_metadata["formal_anchor_cache"] = formal_anchor_provenance
    if args.oof_fold_id is not None:
        cache_metadata["oof_fold_id"] = int(args.oof_fold_id)
        cache_metadata["oof_heldout"] = True
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save({"metadata": cache_metadata, "records": records}, temporary)
    temporary.replace(output_path)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_to={output_path}")
    return summary


if __name__ == "__main__":
    build(parse_args())
