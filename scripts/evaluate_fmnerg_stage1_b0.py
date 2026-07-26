"""Build the matched Stage1 bypass + F2 subtype Dev baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.data import (
    fine_gold_by_record,
    read_fine_conll,
)
from sidecars.fmnerg_subtype.encoder_config import (
    load_subtype_encoder_config,
)
from sidecars.fmnerg_subtype.encoder_evaluator import (
    evaluate_online_formal_predictions,
    evaluate_online_gold_spans,
)
from sidecars.fmnerg_subtype.encoder_model import (
    build_trainable_subtype_encoder,
    load_trainable_checkpoint_state,
)
from sidecars.fmnerg_subtype.online_data import (
    OnlineSubtypeCollator,
    OnlineSubtypeRecordDataset,
    formal_online_records,
    gold_online_records,
)
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy
from sidecars.fmnerg_subtype.io import resolve_path, sha256_file
from gmner.fmnerg.baseline import build_matched_b0_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subtype-config", required=True)
    parser.add_argument("--subtype-checkpoint", required=True)
    parser.add_argument("--stage1-dev-cache", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_subtype_encoder_config(
        resolve_path(args.subtype_config, root)
    )
    taxonomy = SubtypeTaxonomy.from_file(
        resolve_path(config.taxonomy, root)
    )
    fine_records = read_fine_conll(
        resolve_path(config.data.dev_source, root),
        taxonomy,
        require_all_subtypes=False,
    )
    cache_path = resolve_path(args.stage1_dev_cache, root)
    cache = torch.load(cache_path, map_location="cpu")
    fine_gold = fine_gold_by_record(fine_records, taxonomy)
    formal_payload = build_matched_b0_payload(
        cache,
        fine_gold=fine_gold,
    )

    requested_device = args.device or config.runtime.device
    device = torch.device(
        requested_device
        if str(requested_device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    model, tokenizer, initialization, trainability = (
        build_trainable_subtype_encoder(
            config=config,
            taxonomy=taxonomy,
            root=root,
            device=device,
        )
    )
    if (
        formal_payload["metadata"]["stage1_checkpoint_sha256"]
        != initialization["stage1_checkpoint_sha256"]
    ):
        raise ValueError(
            "B0 Stage1 cache and F2 initialization checkpoint differ."
        )
    checkpoint_path = resolve_path(args.subtype_checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("kind") != "fmnerg_trainable_subtype_encoder":
        raise ValueError("B0 requires an F2 subtype-encoder checkpoint.")
    if checkpoint.get("test_accessed") is not False:
        raise ValueError("F2 checkpoint accessed Test data.")
    if checkpoint.get("taxonomy_sha256") != taxonomy.source_sha256:
        raise ValueError("F2 taxonomy fingerprint changed.")
    if dict(checkpoint.get("trainability") or {}) != trainability:
        raise ValueError("F2 trainability contract changed.")
    load_trainable_checkpoint_state(model, checkpoint["model"])
    model.to(device).eval()

    gold_dataset = OnlineSubtypeRecordDataset(
        gold_online_records(fine_records, taxonomy)
    )
    formal_dataset = OnlineSubtypeRecordDataset(
        formal_online_records(
            formal_payload,
            fine_records,
            taxonomy,
        )
    )
    collator = OnlineSubtypeCollator(
        tokenizer,
        max_length=int(initialization["max_length"]),
    )
    gold_metrics = evaluate_online_gold_spans(
        model,
        gold_dataset,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
        include_detailed=False,
    )
    formal = evaluate_online_formal_predictions(
        model,
        formal_dataset,
        formal_payload,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
    )
    result = {
        "metadata": {
            "kind": "fmnerg_stage1_matched_b0",
            "format_version": 1,
            "split": "dev",
            "test_accessed": False,
            "stage1_dev_cache": str(cache_path.resolve()),
            "stage1_dev_cache_sha256": sha256_file(cache_path),
            "subtype_checkpoint": str(checkpoint_path.resolve()),
            "subtype_checkpoint_sha256": sha256_file(checkpoint_path),
            "taxonomy_sha256": taxonomy.source_sha256,
        },
        "metrics": {
            **gold_metrics,
            **formal["metrics"],
        },
    }
    output_path = resolve_path(args.output, root)
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
