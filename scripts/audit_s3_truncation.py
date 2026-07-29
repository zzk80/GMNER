"""Run the read-only S3 P0-B word/subword truncation audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.data import load_word_aligned_tokenizer
from gmner.diagnostics import audit_truncation, read_s3_source_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/fmnerg_twitter10000_stage1.yaml",
    )
    parser.add_argument("--split", choices=["train", "dev"], required=True)
    parser.add_argument("--input-file", default=None)
    parser.add_argument("--text-model-name", default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _resolve(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_config(_resolve(args.config, root))
    source_value = args.input_file or (
        config.data.train_file
        if args.split == "train"
        else config.data.dev_file
    )
    source = _resolve(source_value, root)
    model_name = args.text_model_name or config.model.text_model_name
    tokenizer = load_word_aligned_tokenizer(
        model_name,
        local_files_only=args.local_files_only,
    )
    report = audit_truncation(
        read_s3_source_records(source),
        tokenizer=tokenizer,
        max_length=args.max_length or config.data.max_length,
        split=args.split,
    )
    report["source"] = str(source.resolve())
    report["source_sha256"] = _sha256(source)
    report["config_sha256"] = _sha256(_resolve(args.config, root))
    report["text_model_name"] = str(model_name)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
