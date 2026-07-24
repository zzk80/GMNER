"""Export frozen dev predictions for the independent FMNERG subtype sidecar."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sidecars.fmnerg_subtype.config import load_sidecar_config
from sidecars.fmnerg_subtype.formal_chain import (
    export_evidence_visibility_predictions,
    save_formal_predictions,
)
from sidecars.fmnerg_subtype.io import resolve_path
from sidecars.fmnerg_subtype.taxonomy import SubtypeTaxonomy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_sidecar_config(args.config)
    taxonomy = SubtypeTaxonomy.from_file(
        resolve_path(config.taxonomy, root)
    )
    requested_device = args.device or config.runtime.device
    device = torch.device(
        requested_device
        if str(requested_device).startswith("cuda")
        and torch.cuda.is_available()
        else "cpu"
    )
    payload = export_evidence_visibility_predictions(
        root=root,
        taxonomy=taxonomy,
        source_file=config.data.dev_source,
        evidence_config_path=config.frozen.evidence_config,
        evidence_checkpoint_path=config.frozen.evidence_checkpoint,
        formal_cache_path=config.frozen.formal_dev_cache,
        expanded_cache_path=config.frozen.expanded_dev_cache,
        device=device,
        batch_size=args.batch_size,
    )
    output = resolve_path(
        args.output or config.data.dev_formal_predictions,
        root,
    )
    save_formal_predictions(payload, output)
    print(
        json.dumps(
            {
                "output": str(output),
                **payload["metadata"]["coarse_metrics"],
                "coarse_prediction_sha256": payload["metadata"][
                    "coarse_prediction_sha256"
                ],
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
