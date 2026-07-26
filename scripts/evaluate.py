"""Evaluate a trained GMNER checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.data import (
    GMNERCollator,
    MMNERJsonDataset,
    TextGraphBuilder,
    load_word_aligned_tokenizer,
    validate_model_input_length,
)
from gmner.data.graph_builders import GraphBuilderConfig
from gmner.engine import evaluate_model
from gmner.fmnerg.taxonomy import (
    SubtypeTaxonomy,
    bind_config_taxonomy_fingerprint,
    validate_taxonomy_fingerprint,
)
from gmner.models import GMNERModel
from gmner.utils.logging import create_logger
from gmner.utils.io import maybe_convert_conll



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GMNER checkpoint")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test"])
    parser.add_argument(
        "--reranker-null-bias",
        type=float,
        default=None,
        help="Optional evaluation-time override for grounding_reranker_null_logit_bias. Negative values reduce NULL predictions.",
    )
    parser.add_argument(
        "--reranker-alpha",
        type=float,
        default=None,
        help="Evaluation-time fixed fusion alpha. When set, uses final_logits = base_logits + alpha * reranker_logits.",
    )
    parser.add_argument("--num-labels", type=int, default=None)
    parser.add_argument("--text-model-name", type=str, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()



def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    config = load_config(args.config)
    if args.text_model_name:
        config.model.text_model_name = args.text_model_name
    if args.max_length is not None:
        if args.max_length < 3:
            raise ValueError("--max-length must leave room for content and special tokens.")
        config.data.max_length = args.max_length
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.reranker_null_bias is not None:
        config.model.grounding_reranker_null_logit_bias = args.reranker_null_bias
    if args.reranker_alpha is not None:
        config.model.grounding_reranker_fusion_mode = "fixed"
        config.model.grounding_reranker_weight = args.reranker_alpha
    if bool(getattr(config.model, "use_fine_subtype_head", False)):
        if str(getattr(config.data, "label_schema", "")) != "fine_hierarchical":
            raise ValueError(
                "Stage1-F requires data.label_schema=fine_hierarchical."
            )
        config.data.subtype_taxonomy = str(
            resolve_path(config.data.subtype_taxonomy, project_root)
        )
        taxonomy = SubtypeTaxonomy.from_file(
            config.data.subtype_taxonomy
        )
        bind_config_taxonomy_fingerprint(config.data, taxonomy)
        config.model.num_subtypes = taxonomy.num_subtypes
    else:
        taxonomy = None
    if config.data.semantic_prototype_path:
        config.data.semantic_prototype_path = str(
            resolve_path(config.data.semantic_prototype_path, project_root)
        )
    if config.data.external_knowledge_prototype_path:
        config.data.external_knowledge_prototype_path = str(
            resolve_path(
                config.data.external_knowledge_prototype_path,
                project_root,
            )
        )

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = create_logger("gmner.eval", output_dir / "eval.log")
    if config.model.use_semantic_prototypes:
        logger.warning(
            "Semantic prototypes enabled with prototype_span_source=%s. "
            "Use predicted-span evaluation for stricter Stage 2 reporting.",
            config.model.prototype_span_source,
        )
    if config.model.use_external_knowledge:
        logger.info(
            "External knowledge enabled: %s",
            config.data.external_knowledge_prototype_path,
        )

    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    backbone_config = AutoConfig.from_pretrained(config.model.text_model_name)
    model_input_limit = validate_model_input_length(
        tokenizer,
        backbone_config,
        config.data.max_length,
    )
    logger.info(
        "Tokenizer: class=%s fast=%s max_length=%d model_limit=%s",
        type(tokenizer).__name__,
        bool(getattr(tokenizer, "is_fast", False)),
        config.data.max_length,
        model_input_limit,
    )
    graph_builder_cfg = GraphBuilderConfig(
        use_dependency_graph=config.data.use_dependency_graph,
        dependency_backend=config.data.dependency_backend,
        dependency_model=config.data.dependency_model,
        window_size=config.data.graph_window_size,
    )
    graph_builder = TextGraphBuilder(graph_builder_cfg)

    data_file = config.data.dev_file if args.split == "dev" else config.data.test_file
    data_path = maybe_convert_conll(resolve_path(data_file, project_root), output_dir)
    image_dir = resolve_path(config.data.image_dir, project_root)

    groundability_type_priors = None
    groundability_mention_priors = None
    if config.data.groundability_type_priors:
        groundability_type_priors = resolve_path(config.data.groundability_type_priors, project_root)
    if config.data.groundability_mention_priors:
        groundability_mention_priors = resolve_path(config.data.groundability_mention_priors, project_root)

    dataset = MMNERJsonDataset(
        jsonl_path=str(data_path),
        image_dir=str(image_dir),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=config.data.grounding_enabled,
        expand_entities_for_grounding=config.data.expand_entities_for_grounding,
        image_feature_dir=str(resolve_path(config.data.image_feature_dir, project_root)),
        image_annotation_dir=str(resolve_path(config.data.image_annotation_dir, project_root)),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=str(groundability_type_priors) if groundability_type_priors else None,
        groundability_mention_priors=str(groundability_mention_priors) if groundability_mention_priors else None,
        region_min_score=config.data.region_min_score,
        subtype_taxonomy=taxonomy,
    )
    num_labels = args.num_labels if args.num_labels is not None else 9
    if config.model.use_subtype_auxiliary and config.model.num_subtypes <= 0:
        config.model.num_subtypes = len(getattr(dataset, "subtype_label2id", {}))

    dataloader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=GMNERCollator(tokenizer=tokenizer),
    )

    model = GMNERModel(config=config, num_labels=num_labels)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if taxonomy is not None:
        validate_taxonomy_fingerprint(
            dict(checkpoint.get("model_metadata") or {}),
            taxonomy,
            artifact_name="Stage1-F checkpoint",
        )
    model.load_state_dict(checkpoint["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() and config.runtime.device.startswith("cuda") else "cpu")
    model.to(device)

    metrics = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
    )

    logger.info("%s metrics: %s", args.split, metrics)

    with (output_dir / f"{args.split}_metrics_from_checkpoint.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)
