"""Build frozen gold-span or formal-predicted-span subtype features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.constants import ENTITY_TYPE2ID
from sidecars.fmnerg_subtype.config import load_sidecar_config
from sidecars.fmnerg_subtype.data import (
    FEATURE_CACHE_KIND,
    FEATURE_CACHE_VERSION,
    read_fine_conll,
)
from sidecars.fmnerg_subtype.frozen_encoder import (
    encode_record_spans,
    load_frozen_stage1_backbone,
)
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from sidecars.fmnerg_subtype.metrics import (
    canonical_coarse_prediction_sha256,
)
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--mode", choices=("gold", "formal"), required=True)
    parser.add_argument("--formal-predictions", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--no-fp16-features",
        action="store_true",
    )
    return parser.parse_args()


def gold_encoding_records(records, taxonomy):
    return [
        {
            "record_id": record.record_id,
            "tokens": list(record.tokens),
            "spans": [
                {
                    "start": entity.start,
                    "end": entity.end,
                    "coarse_type_id": ENTITY_TYPE2ID[entity.coarse_type],
                    "subtype_id": taxonomy.subtype_id(entity.subtype),
                    "subtype": entity.subtype,
                    "target_available": True,
                }
                for entity in record.entities
            ],
        }
        for record in records
    ]


def formal_encoding_records(
    formal_payload: dict,
    fine_records,
    taxonomy,
):
    tokens_by_id = {
        record.record_id: list(record.tokens) for record in fine_records
    }
    gold_by_id = {
        record.record_id: {
            (entity.start, entity.end): entity for entity in record.entities
        }
        for record in fine_records
    }
    output = []
    for record_index, record in enumerate(formal_payload["records"]):
        record_id = str(record["record_id"])
        if record_id not in tokens_by_id:
            raise ValueError(
                f"Formal prediction record {record_id!r} is absent from source."
            )
        spans = []
        for prediction_index, prediction in enumerate(
            record.get("predictions") or []
        ):
            start, end = map(int, prediction["span"])
            gold_entity = gold_by_id[record_id].get((start, end))
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "coarse_type_id": int(prediction["type_id"]),
                    "subtype_id": (
                        taxonomy.subtype_id(gold_entity.subtype)
                        if gold_entity is not None
                        else -100
                    ),
                    "subtype": (
                        gold_entity.subtype if gold_entity is not None else None
                    ),
                    "target_available": gold_entity is not None,
                    "record_index": record_index,
                    "prediction_index": prediction_index,
                }
            )
        output.append(
            {
                "record_id": record_id,
                "tokens": tokens_by_id[record_id],
                "spans": spans,
            }
        )
    return output


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_sidecar_config(args.config)
    taxonomy_path = resolve_path(config.taxonomy, root)
    taxonomy = SubtypeTaxonomy.from_file(taxonomy_path)
    source_value = (
        config.data.train_source if args.split == "train" else config.data.dev_source
    )
    source_path = resolve_path(source_value, root)
    fine_records = read_fine_conll(
        source_path,
        taxonomy,
        require_all_subtypes=True,
    )
    formal_path = None
    formal_payload = None
    if args.mode == "formal":
        if args.split != "dev":
            raise ValueError("Formal subtype features are dev-only before test release.")
        formal_path = resolve_path(
            args.formal_predictions or config.data.dev_formal_predictions,
            root,
        )
        formal_payload = json.loads(formal_path.read_text(encoding="utf-8"))
        formal_metadata = dict(formal_payload.get("metadata") or {})
        if formal_metadata.get("kind") != "fmnerg_frozen_formal_predictions":
            raise ValueError("Invalid frozen formal prediction file.")
        if formal_metadata.get("split") != "dev":
            raise ValueError("Only dev formal predictions are allowed.")
        if formal_metadata.get("test_accessed") is not False:
            raise ValueError("Formal feature input accessed test data.")
        if formal_metadata.get("taxonomy_sha256") != taxonomy.source_sha256:
            raise ValueError("Formal prediction taxonomy changed.")
        if (
            canonical_coarse_prediction_sha256(formal_payload["records"])
            != formal_metadata.get("coarse_prediction_sha256")
        ):
            raise ValueError("Formal coarse prediction digest is invalid.")
        encoding_records = formal_encoding_records(
            formal_payload,
            fine_records,
            taxonomy,
        )
        default_output = config.data.dev_formal_features
    else:
        encoding_records = gold_encoding_records(fine_records, taxonomy)
        default_output = (
            config.data.train_gold_features
            if args.split == "train"
            else config.data.dev_gold_features
        )

    requested_device = args.device or config.runtime.device
    device = torch.device(
        requested_device
        if str(requested_device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    fp16 = bool(config.runtime.fp16_features and not args.no_fp16_features)
    backbone, tokenizer, encoder_metadata = load_frozen_stage1_backbone(
        stage1_config_path=config.frozen.stage1_config,
        stage1_checkpoint_path=config.frozen.stage1_checkpoint,
        root=root,
        device=device,
    )
    features, examples = encode_record_spans(
        backbone=backbone,
        tokenizer=tokenizer,
        records=encoding_records,
        max_length=int(encoder_metadata["max_length"]),
        batch_size=args.batch_size,
        device=device,
        fp16=fp16,
    )
    coarse_type_ids = torch.tensor(
        [int(example["coarse_type_id"]) for example in examples],
        dtype=torch.long,
    )
    subtype_ids = torch.tensor(
        [int(example["subtype_id"]) for example in examples],
        dtype=torch.long,
    )
    if features.size(1) != int(config.model.input_size):
        raise ValueError(
            f"Configured input_size={config.model.input_size}, "
            f"extracted feature_size={features.size(1)}."
        )
    parent_check = subtype_ids >= 0
    if parent_check.any():
        expected_parent = torch.tensor(
            [
                taxonomy.parent_id(int(value))
                for value in subtype_ids[parent_check].tolist()
            ]
        )
        actual_parent = coarse_type_ids[parent_check]
        if args.mode == "gold" and not torch.equal(
            expected_parent,
            actual_parent,
        ):
            raise ValueError("Gold subtype feature parents are inconsistent.")

    metadata = {
        "kind": FEATURE_CACHE_KIND,
        "format_version": FEATURE_CACHE_VERSION,
        "split": args.split,
        "mode": args.mode,
        "records": len(encoding_records),
        "examples": len(examples),
        "labeled_examples": int((subtype_ids >= 0).sum().item()),
        "source_file": str(source_path),
        "source_sha256": sha256_file(source_path),
        "taxonomy": str(taxonomy_path),
        "taxonomy_sha256": taxonomy.source_sha256,
        "feature_dtype": str(features.dtype),
        "test_accessed": False,
        **encoder_metadata,
    }
    if formal_path is not None and formal_payload is not None:
        metadata.update(
            {
                "formal_predictions": str(formal_path),
                "formal_predictions_sha256": sha256_file(formal_path),
                "coarse_prediction_sha256": formal_payload["metadata"][
                    "coarse_prediction_sha256"
                ],
                "coarse_metrics": formal_payload["metadata"]["coarse_metrics"],
            }
        )
    payload = {
        "metadata": metadata,
        "features": features,
        "coarse_type_ids": coarse_type_ids,
        "subtype_ids": subtype_ids,
        "examples": examples,
    }
    output = resolve_path(args.output or default_output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    print(
        json.dumps(
            {
                "output": str(output),
                "split": args.split,
                "mode": args.mode,
                "records": len(encoding_records),
                "examples": len(examples),
                "labeled_examples": metadata["labeled_examples"],
                "feature_size": features.size(1),
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
