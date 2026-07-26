"""Re-evaluate one trainable FMNERG subtype encoder on frozen Dev output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from sidecars.fmnerg_subtype.encoder_runtime import (
    load_online_subtype_data,
)
from sidecars.fmnerg_subtype.evaluator import save_json_atomic
from sidecars.fmnerg_subtype.io import resolve_path
from sidecars.fmnerg_subtype.online_data import OnlineSubtypeCollator
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_subtype_encoder_config(resolve_path(args.config, root))
    requested_device = args.device or config.runtime.device
    device = torch.device(
        requested_device
        if str(requested_device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    taxonomy = SubtypeTaxonomy.from_file(resolve_path(config.taxonomy, root))
    (
        _,
        dev_gold_dataset,
        dev_formal_dataset,
        formal_payload,
        _,
    ) = load_online_subtype_data(
        config=config,
        taxonomy=taxonomy,
        root=root,
    )
    model, tokenizer, initialization, trainability = (
        build_trainable_subtype_encoder(
            config=config,
            taxonomy=taxonomy,
            root=root,
            device=device,
        )
    )
    checkpoint_path = resolve_path(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if checkpoint.get("kind") != "fmnerg_trainable_subtype_encoder":
        raise ValueError("Not a trainable subtype-encoder checkpoint.")
    if checkpoint.get("test_accessed") is not False:
        raise ValueError("Subtype encoder checkpoint accessed test data.")
    if checkpoint.get("taxonomy_sha256") != taxonomy.source_sha256:
        raise ValueError("Subtype encoder taxonomy fingerprint changed.")
    stored_initialization = dict(checkpoint.get("initialization") or {})
    if stored_initialization.get("stage1_checkpoint_sha256") != initialization.get(
        "stage1_checkpoint_sha256"
    ):
        raise ValueError("Subtype encoder Stage1 initialization changed.")
    if dict(checkpoint.get("trainability") or {}) != trainability:
        raise ValueError("Subtype encoder trainability scope changed.")
    load_trainable_checkpoint_state(model, checkpoint["model"])
    model.to(device).eval()
    collator = OnlineSubtypeCollator(
        tokenizer,
        max_length=int(initialization["max_length"]),
    )
    gold_metrics = evaluate_online_gold_spans(
        model,
        dev_gold_dataset,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
        include_detailed=True,
    )
    formal = evaluate_online_formal_predictions(
        model,
        dev_formal_dataset,
        formal_payload,
        collator=collator,
        taxonomy=taxonomy,
        batch_size=config.optim.eval_batch_size,
        device=device,
    )
    result = {
        "metadata": {
            **formal["metadata"],
            "kind": "fmnerg_subtype_encoder_dev_evaluation",
            "format_version": 1,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "encoder_scope": config.model.encoder_scope,
            "trainability": trainability,
            "formal_stage1_mutated": False,
            "test_accessed": False,
        },
        "metrics": {
            **gold_metrics,
            **formal["metrics"],
        },
    }
    output = resolve_path(
        args.output or checkpoint_path.parent / "dev_metrics_recomputed.json",
        root,
    )
    save_json_atomic(result, output)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
