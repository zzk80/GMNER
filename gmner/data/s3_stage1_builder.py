"""Locked Train/Dev data and initialization helpers for S3.1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from gmner.config import GMNERConfig, load_config
from gmner.constants import DEFAULT_LABEL2ID
from gmner.data.graph_builders import (
    GraphBuilderConfig,
    TextGraphBuilder,
)
from gmner.data.mmner_dataset import MMNERJsonDataset
from gmner.data.record_level_stage1_collator import (
    RecordLevelStage1Collator,
)
from gmner.data.record_level_stage1_dataset import (
    RecordLevelStage1Dataset,
)
from gmner.data.tokenization import load_word_aligned_tokenizer
from gmner.models.gmner_model import GMNERModel
from gmner.s3_config import S3Stage1Config
from gmner.utils.io import maybe_convert_conll


@dataclass(frozen=True)
class S3Initialization:
    formal_config_path: Path
    checkpoint_path: Path
    baseline_lock_path: Path
    formal_config_sha256: str
    checkpoint_sha256: str
    baseline_lock: dict[str, Any]
    formal_config: GMNERConfig


def resolve_project_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_s3_initialization(
    config: S3Stage1Config,
    *,
    project_root: Path,
) -> S3Initialization:
    formal_config_path = resolve_project_path(
        config.base.formal_config,
        project_root,
    )
    checkpoint_path = resolve_project_path(
        config.base.initialization_checkpoint,
        project_root,
    )
    lock_path = resolve_project_path(
        config.base.baseline_lock,
        project_root,
    )
    for path in (formal_config_path, checkpoint_path, lock_path):
        if not path.exists():
            raise FileNotFoundError(f"Required S3.1 artifact missing: {path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if bool(lock.get("test_accessed", True)):
        raise ValueError("The S3 baseline lock records Test access.")
    config_hash = file_sha256(formal_config_path)
    checkpoint_hash = file_sha256(checkpoint_path)
    if config_hash != str(lock["config"]["sha256"]):
        raise ValueError("Formal Stage1 config differs from the S3 lock.")
    if checkpoint_hash != str(lock["checkpoint"]["sha256"]):
        raise ValueError(
            "Formal Stage1 checkpoint differs from the S3 lock."
        )
    formal_config = load_config(formal_config_path)
    if int(formal_config.model.hidden_size) != int(
        config.model.hidden_size
    ):
        raise ValueError(
            "S3 Student hidden size differs from the formal Stage1."
        )
    return S3Initialization(
        formal_config_path=formal_config_path,
        checkpoint_path=checkpoint_path,
        baseline_lock_path=lock_path,
        formal_config_sha256=config_hash,
        checkpoint_sha256=checkpoint_hash,
        baseline_lock=lock,
        formal_config=formal_config,
    )


def load_formal_stage1_teacher(
    initialization: S3Initialization,
) -> GMNERModel:
    teacher = GMNERModel(
        config=initialization.formal_config,
        num_labels=len(DEFAULT_LABEL2ID),
    )
    payload = torch.load(
        initialization.checkpoint_path,
        map_location="cpu",
    )
    teacher.load_state_dict(payload["model_state_dict"])
    return teacher


def build_s3_record_dataset(
    *,
    formal_config: GMNERConfig,
    tokenizer: Any,
    project_root: Path,
    working_dir: Path,
    split: str,
) -> RecordLevelStage1Dataset:
    normalized = str(split).lower()
    if normalized not in {"train", "dev"}:
        raise ValueError("S3.1 data access is restricted to Train/Dev.")
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=(
                formal_config.data.use_dependency_graph
            ),
            dependency_backend=(
                formal_config.data.dependency_backend
            ),
            dependency_model=formal_config.data.dependency_model,
            window_size=formal_config.data.graph_window_size,
        )
    )
    source_value = (
        formal_config.data.train_file
        if normalized == "train"
        else formal_config.data.dev_file
    )
    source = maybe_convert_conll(
        resolve_project_path(source_value, project_root),
        working_dir,
    )
    expanded = MMNERJsonDataset(
        jsonl_path=str(source),
        image_dir=str(
            resolve_project_path(
                formal_config.data.image_dir,
                project_root,
            )
        ),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=formal_config.data.max_length,
        grounding_enabled=True,
        expand_entities_for_grounding=True,
        image_feature_dir=str(
            resolve_project_path(
                formal_config.data.image_feature_dir,
                project_root,
            )
        ),
        image_annotation_dir=str(
            resolve_project_path(
                formal_config.data.image_annotation_dir,
                project_root,
            )
        ),
        max_regions=formal_config.data.max_regions,
        region_feature_dim=formal_config.model.region_feature_dim,
        grounding_iou_threshold=(
            formal_config.data.grounding_iou_threshold
        ),
        add_null_region=formal_config.data.add_null_region,
        groundability_type_priors=str(
            resolve_project_path(
                formal_config.data.groundability_type_priors,
                project_root,
            )
        ),
        groundability_mention_priors=str(
            resolve_project_path(
                formal_config.data.groundability_mention_priors,
                project_root,
            )
        ),
        region_min_score=formal_config.data.region_min_score,
    )
    return RecordLevelStage1Dataset(expanded, split=normalized)


def build_s3_dataloader(
    dataset: Dataset,
    *,
    tokenizer: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        collate_fn=RecordLevelStage1Collator(tokenizer),
        generator=generator if shuffle else None,
    )


def load_s3_tokenizer(
    initialization: S3Initialization,
) -> Any:
    return load_word_aligned_tokenizer(
        initialization.formal_config.model.text_model_name,
        local_files_only=True,
    )
