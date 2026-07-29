"""Train GMNER model from config."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Tuple

import torch
import yaml
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import GMNERConfig, load_config
from gmner.data import (
    GMNERCollator,
    MMNERJsonDataset,
    TextGraphBuilder,
    load_word_aligned_tokenizer,
    validate_model_input_length,
)
from gmner.data.graph_builders import GraphBuilderConfig
from gmner.engine import GMNERTrainer, evaluate_model
from gmner.fmnerg.taxonomy import (
    SubtypeTaxonomy,
    bind_config_taxonomy_fingerprint,
)
from gmner.models import GMNERModel
from gmner.utils.io import ensure_dir, maybe_convert_conll
from gmner.utils.logging import create_logger
from gmner.utils.seed import set_seed


TRAINABLE_MODULE_ALIASES = {
    "reranker": ("grounding_reranker",),
    "grounding_reranker": ("grounding_reranker",),
    "grounding_head": ("grounding_head",),
    "grounding_adapter": ("grounding_residual_adapter",),
    "grounding_residual_adapter": ("grounding_residual_adapter",),
    "multiscale": ("multiscale_grounding_aligner",),
    "multiscale_grounding": ("multiscale_grounding_aligner",),
    "multiscale_grounding_aligner": ("multiscale_grounding_aligner",),
    "evidence_decoder": ("entity_evidence_decoder",),
    "entity_evidence_decoder": ("entity_evidence_decoder",),
    "prototype": ("prototype_bank",),
    "prototypes": ("prototype_bank",),
    "prototype_bank": ("prototype_bank",),
    "type_reranker": ("prototype_bank",),
    "prototype_type_reranker": ("prototype_bank",),
    "external_knowledge": ("external_knowledge_bank",),
    "external_knowledge_bank": ("external_knowledge_bank",),
    "external_arbiter": ("external_knowledge_bank.type_arbiter",),
    "joint": ("joint_entity_adapter", "joint_type_region_verifier"),
    "jtrv": ("joint_entity_adapter", "joint_type_region_verifier"),
    "joint_verifier": ("joint_entity_adapter", "joint_type_region_verifier"),
    "joint_type_region_verifier": ("joint_type_region_verifier",),
    "joint_entity_adapter": ("joint_entity_adapter",),
    "ner": ("ner_head",),
    "ner_head": ("ner_head",),
    "text": ("text_encoder", "text_projector", "text_graph_encoder"),
    "image": ("image_encoder", "region_projector", "region_norm", "image_graph_encoder"),
    "aligner": ("aligner",),
    "all": ("",),
}


def apply_trainable_modules(model: GMNERModel, trainable_modules: str, logger) -> None:
    """Freeze parameters outside the configured trainable module set."""

    requested = [
        item.strip().lower()
        for item in str(trainable_modules or "all").split(",")
        if item.strip()
    ]
    if not requested:
        requested = ["all"]
    if "all" in requested:
        for parameter in model.parameters():
            parameter.requires_grad = True
        trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in model.parameters())
        logger.info("Trainable modules: all (%d/%d parameters)", trainable, total)
        return

    prefixes: list[str] = []
    unknown: list[str] = []
    for name in requested:
        if name not in TRAINABLE_MODULE_ALIASES:
            unknown.append(name)
            continue
        prefixes.extend(TRAINABLE_MODULE_ALIASES[name])
    if unknown:
        raise ValueError(f"Unknown trainable_modules entries: {', '.join(unknown)}")

    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = any(
            parameter_name == prefix or parameter_name.startswith(prefix + ".")
            for prefix in prefixes
        )

    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable_names:
        raise ValueError(f"trainable_modules={trainable_modules!r} did not match any model parameters.")

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    preview = ", ".join(trainable_names[:8])
    if len(trainable_names) > 8:
        preview += ", ..."
    logger.info(
        "Trainable modules: %s (%d/%d parameters, %.2f%%). First params: %s",
        ", ".join(requested),
        trainable,
        total,
        100.0 * trainable / max(1, total),
        preview,
    )



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GMNER")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--train-file", type=str, default=None, help="Optional override for train file.")
    parser.add_argument("--dev-file", type=str, default=None, help="Optional override for dev file.")
    parser.add_argument("--test-file", type=str, default=None, help="Optional override for test file.")
    parser.add_argument("--image-dir", type=str, default=None, help="Optional override for image folder.")
    parser.add_argument("--task-type", type=str, default=None, help="Override task type.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional override for runtime output dir.")
    parser.add_argument(
        "--text-model-name",
        type=str,
        default=None,
        help="Optional local path or model id for the text backbone.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional tokenizer sequence-length override.",
    )
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Optional override for init checkpoint.")
    parser.add_argument("--trainable-modules", type=str, default=None, help="Optional override for trainable modules.")
    parser.add_argument("--num-epochs", type=int, default=None, help="Optional override for number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional override for training/evaluation batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Optional override for gradient accumulation steps.",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="Optional override for optimizer learning rate.")
    parser.add_argument(
        "--reranker-null-bias",
        type=float,
        default=None,
        help="Optional override for grounding_reranker_null_logit_bias. Negative values reduce NULL predictions.",
    )
    parser.add_argument(
        "--reranker-alpha",
        type=float,
        default=None,
        help="Optional fixed fusion alpha. When set, uses final_logits = base_logits + alpha * reranker_logits.",
    )
    parser.add_argument("--max-train-samples", type=int, default=None, help="Use only the first N train samples for debugging.")
    parser.add_argument("--num-labels", type=int, default=None, help="Label number for NER task.")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional runtime seed override.",
    )
    parser.add_argument(
        "--lambda-fine-subtype",
        type=float,
        default=None,
        help="Optional Stage1-F subtype-loss weight override.",
    )
    parser.add_argument(
        "--skip-test-evaluation",
        action="store_true",
        help="Skip final test evaluation. Useful for out-of-fold checkpoint generation.",
    )
    return parser.parse_args()



def resolve_path(path_str: str, project_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return project_root / path



def apply_overrides(config: GMNERConfig, args: argparse.Namespace) -> GMNERConfig:
    if args.train_file:
        config.data.train_file = args.train_file
    if args.dev_file:
        config.data.dev_file = args.dev_file
    if args.test_file:
        config.data.test_file = args.test_file
    if args.image_dir:
        config.data.image_dir = args.image_dir
    if args.task_type:
        config.data.task_type = args.task_type
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.text_model_name:
        config.model.text_model_name = args.text_model_name
    if args.max_length is not None:
        if args.max_length < 3:
            raise ValueError("--max-length must leave room for content and special tokens.")
        config.data.max_length = args.max_length
    if args.init_checkpoint is not None:
        config.runtime.init_checkpoint = args.init_checkpoint
    if args.trainable_modules:
        config.runtime.trainable_modules = args.trainable_modules
    if args.num_epochs is not None:
        config.optim.num_epochs = args.num_epochs
    if args.batch_size is not None:
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive.")
        config.optim.batch_size = args.batch_size
    if args.gradient_accumulation_steps is not None:
        if args.gradient_accumulation_steps < 1:
            raise ValueError("--gradient-accumulation-steps must be positive.")
        config.optim.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.learning_rate is not None:
        config.optim.learning_rate = args.learning_rate
    if args.reranker_null_bias is not None:
        config.model.grounding_reranker_null_logit_bias = args.reranker_null_bias
    if args.reranker_alpha is not None:
        config.model.grounding_reranker_fusion_mode = "fixed"
        config.model.grounding_reranker_weight = args.reranker_alpha
    if args.seed is not None:
        config.runtime.seed = int(args.seed)
    if args.lambda_fine_subtype is not None:
        if args.lambda_fine_subtype < 0:
            raise ValueError("--lambda-fine-subtype must be non-negative.")
        config.loss.lambda_fine_subtype = float(
            args.lambda_fine_subtype
        )
    return config



def build_datasets(
    config: GMNERConfig,
    tokenizer,
    graph_builder: TextGraphBuilder,
    project_root: Path,
    output_dir: Path,
    num_labels_override: int | None = None,
    build_test: bool = True,
) -> Tuple:
    subtype_taxonomy = None
    if bool(getattr(config.model, "use_fine_subtype_head", False)):
        subtype_taxonomy = SubtypeTaxonomy.from_file(
            resolve_path(config.data.subtype_taxonomy, project_root)
        )
        bind_config_taxonomy_fingerprint(
            config.data,
            subtype_taxonomy,
        )
    train_path = maybe_convert_conll(resolve_path(config.data.train_file, project_root), output_dir)
    dev_path = maybe_convert_conll(resolve_path(config.data.dev_file, project_root), output_dir)
    test_path = None
    if build_test:
        test_path = maybe_convert_conll(
            resolve_path(config.data.test_file, project_root), output_dir
        )
    image_dir = resolve_path(config.data.image_dir, project_root)
    image_feature_dir = None
    image_annotation_dir = None
    if config.data.image_feature_dir:
        image_feature_dir = resolve_path(config.data.image_feature_dir, project_root)
    if config.data.image_annotation_dir:
        image_annotation_dir = resolve_path(config.data.image_annotation_dir, project_root)

    groundability_type_priors = None
    groundability_mention_priors = None
    if config.data.groundability_type_priors:
        groundability_type_priors = resolve_path(config.data.groundability_type_priors, project_root)
    if config.data.groundability_mention_priors:
        groundability_mention_priors = resolve_path(config.data.groundability_mention_priors, project_root)

    if config.data.grounding_enabled:
        if not image_feature_dir or not image_feature_dir.exists():
            raise ValueError("Grounding enabled but image_feature_dir is missing or not found.")
        if not image_annotation_dir or not image_annotation_dir.exists():
            raise ValueError("Grounding enabled but image_annotation_dir is missing or not found.")

    train_dataset = MMNERJsonDataset(
        jsonl_path=str(train_path),
        image_dir=str(image_dir),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=config.data.grounding_enabled,
        expand_entities_for_grounding=config.data.expand_entities_for_grounding,
        image_feature_dir=str(image_feature_dir) if image_feature_dir else None,
        image_annotation_dir=str(image_annotation_dir) if image_annotation_dir else None,
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=str(groundability_type_priors) if groundability_type_priors else None,
        groundability_mention_priors=str(groundability_mention_priors) if groundability_mention_priors else None,
        region_min_score=config.data.region_min_score,
        subtype_taxonomy=subtype_taxonomy,
        require_all_subtypes=subtype_taxonomy is not None,
    )
    dev_dataset = MMNERJsonDataset(
        jsonl_path=str(dev_path),
        image_dir=str(image_dir),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=config.data.grounding_enabled,
        expand_entities_for_grounding=config.data.expand_entities_for_grounding,
        image_feature_dir=str(image_feature_dir) if image_feature_dir else None,
        image_annotation_dir=str(image_annotation_dir) if image_annotation_dir else None,
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=str(groundability_type_priors) if groundability_type_priors else None,
        groundability_mention_priors=str(groundability_mention_priors) if groundability_mention_priors else None,
        region_min_score=config.data.region_min_score,
        subtype_taxonomy=subtype_taxonomy,
    )
    test_dataset = None
    if test_path is not None:
        test_dataset = MMNERJsonDataset(
            jsonl_path=str(test_path),
            image_dir=str(image_dir),
            tokenizer=tokenizer,
            graph_builder=graph_builder,
            max_length=config.data.max_length,
            grounding_enabled=config.data.grounding_enabled,
            expand_entities_for_grounding=config.data.expand_entities_for_grounding,
            image_feature_dir=str(image_feature_dir) if image_feature_dir else None,
            image_annotation_dir=str(image_annotation_dir) if image_annotation_dir else None,
            max_regions=config.data.max_regions,
            region_feature_dim=config.model.region_feature_dim,
            grounding_iou_threshold=config.data.grounding_iou_threshold,
            add_null_region=config.data.add_null_region,
            groundability_type_priors=str(groundability_type_priors) if groundability_type_priors else None,
            groundability_mention_priors=str(groundability_mention_priors) if groundability_mention_priors else None,
            region_min_score=config.data.region_min_score,
            subtype_taxonomy=subtype_taxonomy,
        )

    num_labels = num_labels_override if num_labels_override is not None else 9
    return train_dataset, dev_dataset, test_dataset, num_labels


if __name__ == "__main__":
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]

    config = load_config(args.config)
    config = apply_overrides(config, args)
    if bool(getattr(config.model, "use_fine_subtype_head", False)):
        if str(getattr(config.data, "label_schema", "")) != "fine_hierarchical":
            raise ValueError(
                "Stage1-F requires data.label_schema=fine_hierarchical."
            )
        if not str(getattr(config.data, "subtype_taxonomy", "")).strip():
            raise ValueError(
                "Stage1-F requires data.subtype_taxonomy."
            )
        config.data.subtype_taxonomy = str(
            resolve_path(config.data.subtype_taxonomy, project_root)
        )
        taxonomy = SubtypeTaxonomy.from_file(config.data.subtype_taxonomy)
        bind_config_taxonomy_fingerprint(config.data, taxonomy)
        config.model.num_subtypes = taxonomy.num_subtypes
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

    output_dir = ensure_dir(config.runtime.output_dir)
    logger = create_logger("gmner.train", output_dir / "train.log")

    set_seed(config.runtime.seed)

    logger.info("Loading tokenizer: %s", config.model.text_model_name)
    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    backbone_config = AutoConfig.from_pretrained(config.model.text_model_name)
    model_input_limit = validate_model_input_length(
        tokenizer,
        backbone_config,
        config.data.max_length,
    )
    logger.info(
        "Tokenizer loaded: class=%s fast=%s max_length=%d model_limit=%s",
        type(tokenizer).__name__,
        bool(getattr(tokenizer, "is_fast", False)),
        config.data.max_length,
        model_input_limit,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        logger.warning(
            "Slow tokenizer detected; using explicit per-word subword alignment."
        )

    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as fp:
        yaml.safe_dump(asdict(config), fp, sort_keys=False, allow_unicode=True)

    graph_builder_cfg = GraphBuilderConfig(
        use_dependency_graph=config.data.use_dependency_graph,
        dependency_backend=config.data.dependency_backend,
        dependency_model=config.data.dependency_model,
        window_size=config.data.graph_window_size,
    )
    graph_builder = TextGraphBuilder(graph_builder_cfg)

    logger.info("Building datasets (dependency_graph=%s)", config.data.use_dependency_graph)
    if config.model.use_semantic_prototypes:
        logger.warning(
            "Semantic prototypes enabled with prototype_span_source=%s. "
            "gold_train_pred_eval trains with gold spans and evaluates with predicted spans.",
            config.model.prototype_span_source,
        )
    if config.model.use_external_knowledge:
        logger.info(
            "External knowledge enabled: %s",
            config.data.external_knowledge_prototype_path,
        )

    train_dataset, dev_dataset, test_dataset, num_labels = build_datasets(
        config=config,
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        project_root=project_root,
        output_dir=output_dir,
        num_labels_override=args.num_labels,
        build_test=not args.skip_test_evaluation,
    )
    train_subtype_count = len(getattr(train_dataset, "subtype_label2id", {}))
    if args.max_train_samples is not None:
        max_train_samples = max(1, int(args.max_train_samples))
        if max_train_samples < len(train_dataset):
            train_dataset = Subset(train_dataset, list(range(max_train_samples)))
            logger.info("Using first %d train samples for debug overfit.", max_train_samples)
    logger.info("Datasets built")

    logger.info("Train samples: %d", len(train_dataset))
    logger.info("Dev samples: %d", len(dev_dataset))
    if test_dataset is None:
        logger.info("Test dataset not built (--skip-test-evaluation).")
    else:
        logger.info("Test samples: %d", len(test_dataset))
    if config.model.use_subtype_auxiliary and config.model.num_subtypes <= 0:
        config.model.num_subtypes = train_subtype_count
    if config.model.use_subtype_auxiliary:
        logger.info("Subtype auxiliary labels: %d", config.model.num_subtypes)
    if config.model.use_fine_subtype_head:
        taxonomy = SubtypeTaxonomy.from_file(config.data.subtype_taxonomy)
        config.model.num_subtypes = taxonomy.num_subtypes
        logger.info(
            "Formal subtype taxonomy: labels=%d sha256=%s",
            taxonomy.num_subtypes,
            taxonomy.source_sha256,
        )

    collator = GMNERCollator(tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.optim.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collator,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.optim.batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
            collate_fn=collator,
        )

    model = GMNERModel(config=config, num_labels=num_labels)
    if config.runtime.init_checkpoint:
        init_checkpoint = resolve_path(config.runtime.init_checkpoint, project_root)
        if not init_checkpoint.exists():
            raise FileNotFoundError(f"Initial checkpoint not found: {init_checkpoint}")
        checkpoint = torch.load(init_checkpoint, map_location="cpu")
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(
            "Initialized from %s (missing=%d, unexpected=%d)",
            init_checkpoint,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )

    apply_trainable_modules(model, config.runtime.trainable_modules, logger)

    trainer = GMNERTrainer(
        model=model,
        config=config,
        train_dataloader=train_loader,
        dev_dataloader=dev_loader,
        num_labels=num_labels,
        logger=logger,
    )
    best_checkpoint_path = trainer.train()
    logger.info("Best checkpoint: %s", best_checkpoint_path)

    if args.skip_test_evaluation:
        logger.info("Skipping final test evaluation by request.")
    else:
        if test_loader is None:
            raise RuntimeError("Test evaluation requested without a test dataloader.")
        # Training has already finished, so release optimizer states before
        # restoring the best checkpoint. Loading directly onto CUDA creates a
        # second full model state dict and can OOM on a shared GPU even though
        # forward evaluation itself fits.
        model.zero_grad(set_to_none=True)
        trainer.optimizer = None
        trainer.scheduler = None
        gc.collect()
        if trainer.device.type == "cuda":
            torch.cuda.empty_cache()

        checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint
        gc.collect()
        if trainer.device.type == "cuda":
            torch.cuda.empty_cache()

        test_metrics = evaluate_model(
            model=model,
            dataloader=test_loader,
            device=trainer.device,
        )
        logger.info("Test metrics: %s", test_metrics)

        with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as fp:
            json.dump(test_metrics, fp, ensure_ascii=False, indent=2)

    tokenizer.save_pretrained(output_dir / "tokenizer")
