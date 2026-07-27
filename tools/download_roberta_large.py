"""Download the minimal official RoBERTa-large artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-id",
        default="FacebookAI/roberta-large",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/zzk/gmner/roberta-large",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoTokenizer

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=args.model_id,
        local_dir=output_dir,
        allow_patterns=[
            "config.json",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "model.safetensors",
        ],
    )
    config = AutoConfig.from_pretrained(
        output_dir,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        output_dir,
        use_fast=True,
        local_files_only=True,
    )
    weight_path = output_dir / "model.safetensors"
    if not weight_path.exists():
        raise FileNotFoundError(
            "Official model.safetensors was not downloaded."
        )
    if config.model_type != "roberta":
        raise ValueError(f"Unexpected model type: {config.model_type}")
    if int(config.hidden_size) != 1024:
        raise ValueError("Downloaded model is not RoBERTa-large.")
    if not tokenizer.is_fast:
        raise ValueError("Downloaded tokenizer is not fast.")

    manifest = {
        "kind": "roberta_large_local_model_manifest",
        "format_version": 1,
        "model_id": args.model_id,
        "model_type": config.model_type,
        "hidden_size": int(config.hidden_size),
        "num_hidden_layers": int(config.num_hidden_layers),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_is_fast": bool(tokenizer.is_fast),
        "weight_file": str(weight_path),
        "weight_bytes": weight_path.stat().st_size,
        "weight_sha256": sha256_file(weight_path),
    }
    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
