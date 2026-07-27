"""Download the minimal official RoBERTa-large artifact set."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


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


def download_with_curl(
    *,
    endpoint: str,
    model_id: str,
    filename: str,
    output_dir: Path,
) -> Path:
    destination = output_dir / filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    url = (
        f"{endpoint.rstrip('/')}/{model_id}/resolve/main/{filename}"
    )
    base_command = [
        "curl",
        "--http1.1",
        "--fail",
        "--location",
        "--retry",
        "10",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--speed-time",
        "120",
        "--speed-limit",
        "1024",
        "--output",
        str(partial),
        url,
    ]
    command = [*base_command[:13], "--continue-at", "-", *base_command[13:]]
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError:
        if not partial.exists():
            raise
        partial.unlink()
        subprocess.run(base_command, check=True)
    partial.replace(destination)
    return destination


def main() -> None:
    args = parse_args()
    from transformers import AutoConfig, AutoTokenizer

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = os.environ.get(
        "HF_ENDPOINT",
        "https://huggingface.co",
    )
    downloaded = [
        download_with_curl(
            endpoint=endpoint,
            model_id=args.model_id,
            filename=filename,
            output_dir=output_dir,
        )
        for filename in (
            "config.json",
            "merges.txt",
            "vocab.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "model.safetensors",
        )
    ]
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
        "download_endpoint": endpoint,
        "files": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
            }
            for path in downloaded
        ],
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
