#!/usr/bin/env python3
"""Build gold-free full-image CLIP global and patch caches for DVH."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPImageProcessor, CLIPVisionModel

from gmner.data.frozen_clip_cache import (
    DVH_CLIP_CACHE_KIND,
    DVH_CLIP_CACHE_VERSION,
    sha256_file,
)
from gmner.utils.io import maybe_convert_conll, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_file).resolve()
    image_dir = Path(args.image_dir).resolve()
    model_path = Path(args.model_name).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = maybe_convert_conll(source, output_dir / "converted")
    records = read_jsonl(converted)
    image_rows: dict[str, Path] = {}
    for record in records:
        image_name = str(record["image"])
        image_id = Path(image_name).stem
        image_rows.setdefault(image_id, image_dir / image_name)
    for image_id, path in image_rows.items():
        if not path.exists():
            alternatives = list(image_dir.glob(f"{image_id}.*"))
            if len(alternatives) != 1:
                raise FileNotFoundError(f"Image missing for {image_id}: {path}")
            image_rows[image_id] = alternatives[0]

    processor = CLIPImageProcessor.from_pretrained(
        model_path, local_files_only=True
    )
    model = CLIPVisionModel.from_pretrained(
        model_path, local_files_only=True
    )
    model.requires_grad_(False).eval()
    device = torch.device(args.device)
    model.to(device)
    image_size = int(model.config.image_size)
    patch_size = int(model.config.patch_size)
    patch_count = (image_size // patch_size) ** 2
    feature_dim = int(model.config.hidden_size)
    image_ids = sorted(image_rows)
    index: dict[str, dict[str, int | str]] = {}
    shards: dict[str, dict[str, int | str]] = {}
    pending: list[dict] = []
    shard_index = 0

    def flush() -> None:
        nonlocal pending, shard_index
        if not pending:
            return
        name = f"shard_{shard_index:05d}.pt"
        path = output_dir / name
        torch.save({"entries": pending}, path)
        shards[name] = {
            "records": len(pending),
            "sha256": sha256_file(path),
        }
        for offset, entry in enumerate(pending):
            index[str(entry["image_id"])] = {
                "shard": name,
                "offset": offset,
            }
        pending = []
        shard_index += 1

    for start in tqdm(range(0, len(image_ids), args.batch_size), desc="DVH CLIP"):
        batch_ids = image_ids[start : start + args.batch_size]
        images = []
        for image_id in batch_ids:
            with Image.open(image_rows[image_id]) as image:
                images.append(image.convert("RGB").resize((image_size, image_size)))
        inputs = processor(
            images=images,
            return_tensors="pt",
            do_resize=False,
            do_center_crop=False,
        )
        pixels = inputs["pixel_values"].to(device)
        with torch.inference_mode():
            outputs = model(pixel_values=pixels)
        hidden = outputs.last_hidden_state.detach().cpu()
        global_features = hidden[:, 0]
        patch_features = hidden[:, 1:]
        if patch_features.size(1) != patch_count:
            raise ValueError("Unexpected CLIP patch count.")
        if args.fp16:
            global_features = global_features.half()
            patch_features = patch_features.half()
        for row, image_id in enumerate(batch_ids):
            pending.append(
                {
                    "image_id": image_id,
                    "global_feature": global_features[row].contiguous(),
                    "patch_features": patch_features[row].contiguous(),
                    "patch_mask": torch.ones(patch_count, dtype=torch.bool),
                }
            )
            if len(pending) >= args.shard_size:
                flush()
    flush()

    model_files = []
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin", "model.safetensors"):
        path = model_path / name
        if path.exists():
            model_files.append(
                {"name": name, "sha256": sha256_file(path), "size": path.stat().st_size}
            )
    preprocessing = {
        "resize": [image_size, image_size],
        "direct_resize": True,
        "center_crop": False,
        "processor": processor.to_dict(),
    }
    manifest = {
        "kind": DVH_CLIP_CACHE_KIND,
        "format_version": DVH_CLIP_CACHE_VERSION,
        "split": args.split,
        "records": len(image_ids),
        "feature_dim": feature_dim,
        "patch_count": patch_count,
        "patch_grid_size": image_size // patch_size,
        "feature_dtype": "float16" if args.fp16 else "float32",
        "model": {
            "source": str(model_path),
            "fully_frozen": True,
            "eval_mode": True,
            "artifacts": model_files,
        },
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "gold_used_for_candidate_generation": False,
        },
        "preprocessing": preprocessing,
        "preprocessing_sha256": hashlib.sha256(
            json.dumps(preprocessing, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "image_ids_sha256": hashlib.sha256(
            "\n".join(image_ids).encode("utf-8")
        ).hexdigest(),
        "index": index,
        "shards": shards,
        "test_accessed": False,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(output_dir),
        "records": len(image_ids),
        "patch_count": patch_count,
        "feature_dim": feature_dim,
        "test_accessed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
