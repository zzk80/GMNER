"""Shared Train/Dev-only runtime construction for TP M0.5 and M1."""

from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig

from gmner.data import (
    GMNERCollator,
    MMNERJsonDataset,
    TextGraphBuilder,
    load_word_aligned_tokenizer,
    validate_model_input_length,
)
from gmner.data.graph_builders import GraphBuilderConfig
from gmner.models import GMNERModel
from gmner.utils.io import maybe_convert_conll


def resolve_path(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def build_record_level_dataset(
    *,
    config,
    split: str,
    tokenizer,
    graph_builder: TextGraphBuilder,
    project_root: Path,
    cache_dir: Path,
    expand_entities_for_grounding: bool = False,
) -> MMNERJsonDataset:
    if split not in {"train", "dev"}:
        raise ValueError("TP runtime is restricted to train/dev.")
    data_path = maybe_convert_conll(
        resolve_path(getattr(config.data, f"{split}_file"), project_root),
        cache_dir,
    )
    dataset = MMNERJsonDataset(
        jsonl_path=str(data_path),
        image_dir=str(resolve_path(config.data.image_dir, project_root)),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=True,
        expand_entities_for_grounding=expand_entities_for_grounding,
        image_feature_dir=str(resolve_path(config.data.image_feature_dir, project_root)),
        image_annotation_dir=str(resolve_path(config.data.image_annotation_dir, project_root)),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=str(
            resolve_path(config.data.groundability_type_priors, project_root)
        ),
        groundability_mention_priors=str(
            resolve_path(config.data.groundability_mention_priors, project_root)
        ),
        region_min_score=config.data.region_min_score,
    )
    if not expand_entities_for_grounding:
        record_ids = [str(sample["record_id"]) for sample in dataset.samples]
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("TP M1 record-level contract found duplicate record IDs.")
        if len(record_ids) != len(dataset.records):
            raise ValueError(
                "TP M1 requires exactly one training sample per source record: "
                f"records={len(dataset.records)} samples={len(record_ids)}."
            )
    return dataset


def build_tp_runtime(
    *,
    config,
    checkpoint_path: str | Path,
    project_root: Path,
    cache_dir: Path,
    batch_size: int,
    include_train: bool,
    train_expand_entities_for_grounding: bool = False,
) -> dict:
    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    backbone_config = AutoConfig.from_pretrained(config.model.text_model_name)
    validate_model_input_length(tokenizer, backbone_config, config.data.max_length)
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=config.data.use_dependency_graph,
            dependency_backend=config.data.dependency_backend,
            dependency_model=config.data.dependency_model,
            window_size=config.data.graph_window_size,
        )
    )
    dev = build_record_level_dataset(
        config=config,
        split="dev",
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        project_root=project_root,
        cache_dir=cache_dir,
    )
    train = None
    if include_train:
        train = build_record_level_dataset(
            config=config,
            split="train",
            tokenizer=tokenizer,
            graph_builder=graph_builder,
            project_root=project_root,
            cache_dir=cache_dir,
            expand_entities_for_grounding=train_expand_entities_for_grounding,
        )
    collator = GMNERCollator(tokenizer=tokenizer)
    loaders = {
        "dev": DataLoader(dev, batch_size=batch_size, shuffle=False, collate_fn=collator),
    }
    if train is not None:
        loaders["train"] = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        loaders["train_ordered"] = DataLoader(
            train,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )
    model = GMNERModel(config=copy.deepcopy(config), num_labels=9)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return {
        "tokenizer": tokenizer,
        "datasets": {"train": train, "dev": dev},
        "loaders": loaders,
        "model": model,
        "checkpoint": checkpoint,
        "test_accessed": False,
    }
