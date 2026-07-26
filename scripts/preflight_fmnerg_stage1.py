"""Validate M3.3F F0/F1 inputs without loading Test or training a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.fmnerg.taxonomy import (
    SubtypeTaxonomy,
    bind_config_taxonomy_fingerprint,
)
from sidecars.fmnerg_subtype.data import read_fine_conll


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_path(value: str | Path, root: Path, *, name: str) -> Path:
    path = resolve(value, root)
    if not path.exists():
        raise FileNotFoundError(f"Missing {name}: {path}")
    return path


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = required_path(args.config, root, name="Stage1-F config")
    config = load_config(config_path)
    if config.data.label_schema != "fine_hierarchical":
        raise ValueError("Stage1-F requires label_schema=fine_hierarchical.")
    if not config.model.use_fine_subtype_head:
        raise ValueError("Stage1-F requires use_fine_subtype_head=true.")
    if config.model.fine_subtype_input_source != "text_only":
        raise ValueError("F1 subtype input must remain text_only.")
    if config.runtime.save_best_metric != "fmnerg_score":
        raise ValueError("Stage1-F checkpoint selection must use fmnerg_score.")
    if config.data.max_regions != 16:
        raise ValueError("F1 Stage1-F must use the formal R16 budget.")
    if config.loss.lambda_fine_subtype not in {0.5, 1.0}:
        raise ValueError("F1 lambda_fine_subtype must be 0.5 or 1.0.")

    taxonomy_path = required_path(
        config.data.subtype_taxonomy,
        root,
        name="subtype taxonomy",
    )
    taxonomy = SubtypeTaxonomy.from_file(taxonomy_path)
    bind_config_taxonomy_fingerprint(config.data, taxonomy)
    train_path = required_path(
        config.data.train_file,
        root,
        name="fine Train source",
    )
    dev_path = required_path(
        config.data.dev_file,
        root,
        name="fine Dev source",
    )
    checked_paths = {
        "image_dir": required_path(
            config.data.image_dir,
            root,
            name="image directory",
        ),
        "image_feature_dir": required_path(
            config.data.image_feature_dir,
            root,
            name="VinVL feature directory",
        ),
        "image_annotation_dir": required_path(
            config.data.image_annotation_dir,
            root,
            name="image annotation directory",
        ),
        "text_model_name": required_path(
            config.model.text_model_name,
            root,
            name="RoBERTa model",
        ),
        "init_checkpoint": required_path(
            config.runtime.init_checkpoint,
            root,
            name="M3.3A Stage1 initialization",
        ),
    }
    for field_name, display_name in (
        ("groundability_type_priors", "groundability type priors"),
        ("groundability_mention_priors", "groundability mention priors"),
    ):
        configured = str(getattr(config.data, field_name, "") or "")
        if configured:
            checked_paths[field_name] = required_path(
                configured,
                root,
                name=display_name,
            )
    train_records = read_fine_conll(
        train_path,
        taxonomy,
        require_all_subtypes=True,
    )
    dev_records = read_fine_conll(
        dev_path,
        taxonomy,
        require_all_subtypes=False,
    )
    train_entities = sum(len(record.entities) for record in train_records)
    dev_entities = sum(len(record.entities) for record in dev_records)
    result = {
        "metadata": {
            "kind": "fmnerg_stage1_f_preflight",
            "format_version": 1,
            "split": "train+dev",
            "test_accessed": False,
            "test_path_resolved": False,
            "config": str(config_path.resolve()),
            "config_sha256": sha256_file(config_path),
            **taxonomy.fingerprint_metadata(),
        },
        "checks": {
            "label_schema": config.data.label_schema,
            "subtype_input_source": (
                config.model.fine_subtype_input_source
            ),
            "save_best_metric": config.runtime.save_best_metric,
            "lambda_fine_subtype": config.loss.lambda_fine_subtype,
            "parent_types": taxonomy.num_parents,
            "subtypes": taxonomy.num_subtypes,
            "train_records": len(train_records),
            "train_entities": train_entities,
            "dev_records": len(dev_records),
            "dev_entities": dev_entities,
            "train_source_sha256": sha256_file(train_path),
            "dev_source_sha256": sha256_file(dev_path),
            "init_checkpoint_sha256": sha256_file(
                checked_paths["init_checkpoint"]
            ),
            "checked_paths": {
                key: str(path.resolve())
                for key, path in checked_paths.items()
            },
        },
    }
    output_path = resolve(args.output, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
