"""Train Stage1-F without constructing or evaluating the Test split."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--lambda-fine-subtype", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    config = load_config(config_path)
    if config.data.label_schema != "fine_hierarchical":
        raise ValueError(
            "Stage1-F wrapper requires label_schema=fine_hierarchical."
        )
    if not config.model.use_fine_subtype_head:
        raise ValueError("Stage1-F wrapper requires use_fine_subtype_head.")
    if (
        args.lambda_fine_subtype is not None
        and args.lambda_fine_subtype not in {0.5, 1.0}
    ):
        raise ValueError(
            "F1 only preregisters --lambda-fine-subtype 0.5 or 1.0."
        )
    if (
        args.lambda_fine_subtype is not None
        and args.lambda_fine_subtype
        != config.loss.lambda_fine_subtype
        and args.output_dir is None
    ):
        raise ValueError(
            "A subtype-loss override requires a separate --output-dir."
        )

    command = [
        sys.executable,
        "-u",
        str(root / "scripts" / "train.py"),
        "--config",
        str(config_path),
        "--skip-test-evaluation",
    ]
    for flag, value in (
        ("--seed", args.seed),
        ("--output-dir", args.output_dir),
        ("--lambda-fine-subtype", args.lambda_fine_subtype),
        ("--max-train-samples", args.max_train_samples),
        ("--num-epochs", args.num_epochs),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root)
    subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()
