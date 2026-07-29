"""Evaluate Stage1-F on Dev only with strict predicted-span subtype decode."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.constants import DEFAULT_LABEL2ID
from gmner.data import (
    GMNERCollator,
    MMNERJsonDataset,
    TextGraphBuilder,
    load_word_aligned_tokenizer,
    validate_model_input_length,
)
from gmner.data.graph_builders import GraphBuilderConfig
from gmner.engine import evaluate_model
from gmner.fmnerg.data_utils import first_record_indices
from gmner.fmnerg.taxonomy import (
    SubtypeTaxonomy,
    bind_config_taxonomy_fingerprint,
    validate_taxonomy_fingerprint,
)
from gmner.models import GMNERModel
from gmner.utils.io import maybe_convert_conll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def resolve(path: str | Path, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(resolve(args.config, root))
    if (
        config.data.label_schema != "fine_hierarchical"
        or not config.model.use_fine_subtype_head
    ):
        raise ValueError("This evaluator only accepts Stage1-F configs.")

    taxonomy_path = resolve(config.data.subtype_taxonomy, root)
    taxonomy = SubtypeTaxonomy.from_file(taxonomy_path)
    config.data.subtype_taxonomy = str(taxonomy_path)
    bind_config_taxonomy_fingerprint(config.data, taxonomy)
    config.model.num_subtypes = taxonomy.num_subtypes
    checkpoint_path = resolve(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_taxonomy_fingerprint(
        dict(checkpoint.get("model_metadata") or {}),
        taxonomy,
        artifact_name="Stage1-F checkpoint",
    )

    requested_device = args.device or config.runtime.device
    device = torch.device(
        requested_device
        if str(requested_device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    validate_model_input_length(
        tokenizer,
        AutoConfig.from_pretrained(config.model.text_model_name),
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
    output_path = resolve(args.output, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dev_path = maybe_convert_conll(
        resolve(config.data.dev_file, root),
        output_path.parent,
    )
    dataset = MMNERJsonDataset(
        jsonl_path=str(dev_path),
        image_dir=str(resolve(config.data.image_dir, root)),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=True,
        expand_entities_for_grounding=True,
        image_feature_dir=str(resolve(config.data.image_feature_dir, root)),
        image_annotation_dir=str(
            resolve(config.data.image_annotation_dir, root)
        ),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=(
            str(resolve(config.data.groundability_type_priors, root))
            if config.data.groundability_type_priors
            else None
        ),
        groundability_mention_priors=(
            str(resolve(config.data.groundability_mention_priors, root))
            if config.data.groundability_mention_priors
            else None
        ),
        region_min_score=config.data.region_min_score,
        subtype_taxonomy=taxonomy,
    )
    selected = dataset
    if args.max_records is not None:
        selected = Subset(dataset, first_record_indices(dataset, args.max_records))
    dataloader = DataLoader(
        selected,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=GMNERCollator(tokenizer=tokenizer),
    )
    model = GMNERModel(
        config=config,
        num_labels=len(DEFAULT_LABEL2ID),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device)
    metrics = evaluate_model(model, dataloader, device)
    result = {
        "metadata": {
            "kind": "fmnerg_stage1_f_dev_evaluation",
            "format_version": 1,
            "split": "dev",
            "test_accessed": False,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "evaluated_records": (
                min(args.max_records, len(dataset.records))
                if args.max_records is not None
                else len(dataset.records)
            ),
            **taxonomy.fingerprint_metadata(),
        },
        "metrics": metrics,
    }
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
