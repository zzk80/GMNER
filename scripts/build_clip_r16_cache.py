#!/usr/bin/env python3
"""Build frozen CLIP ViT-B/32 features for formal VinVL R16 boxes."""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from gmner.config import load_config
from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.clip_r16_cache import (
    model_artifact_fingerprint,
    require_train_or_dev_split,
    stable_json_sha256,
    write_clip_r16_cache,
)
from gmner.utils.io import maybe_convert_conll, read_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=("train", "dev"))
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def resolve(path: str, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def select_formal_r16(npz_path: Path, max_regions: int, min_score: float) -> dict[str, Any]:
    with np.load(str(npz_path), allow_pickle=True) as payload:
        num_boxes = int(payload.get("num_boxes", 0))
        raw_scores = np.asarray(
            payload.get("scores", np.ones((num_boxes,), dtype=np.float32)),
            dtype=np.float32,
        )[:num_boxes]
        indices = np.flatnonzero(raw_scores >= float(min_score))[: int(max_regions)]
        count = len(indices)
        boxes = np.zeros((max_regions, 4), dtype=np.float32)
        scores = np.zeros((max_regions,), dtype=np.float32)
        valid = np.zeros((max_regions,), dtype=np.bool_)
        if "bounding_boxes" not in payload:
            raise ValueError(f"VinVL artifact has no bounding_boxes: {npz_path}")
        boxes[:count] = np.asarray(payload["bounding_boxes"], dtype=np.float32)[indices]
        scores[:count] = raw_scores[indices]
        valid[:count] = True
        return {
            "boxes": boxes,
            "scores": scores,
            "valid": valid,
            "selected_indices": indices.astype(np.int64),
            "image_height": float(payload.get("image_h", 0)),
            "image_width": float(payload.get("image_w", 0)),
        }


def crop_regions(image: Image.Image, boxes: np.ndarray, valid: np.ndarray) -> list[Image.Image]:
    width, height = image.size
    crops: list[Image.Image] = []
    for box, is_valid in zip(boxes, valid):
        if not bool(is_valid):
            continue
        x1, y1, x2, y2 = [float(value) for value in box]
        left = max(0, min(width - 1, int(np.floor(x1))))
        top = max(0, min(height - 1, int(np.floor(y1))))
        right = max(left + 1, min(width, int(np.ceil(x2))))
        bottom = max(top + 1, min(height, int(np.ceil(y2))))
        crops.append(image.crop((left, top, right, bottom)).convert("RGB"))
    return crops


@torch.no_grad()
def encode_images(
    model,
    processor,
    images: list[Image.Image],
    device: torch.device,
    fp16: bool,
    batch_size: int,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    for start in range(0, len(images), batch_size):
        inputs = processor(images=images[start : start + batch_size], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        use_amp = bool(fp16 and device.type == "cuda")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_amp
            else contextlib.nullcontext()
        )
        with autocast:
            features = model(pixel_values=pixel_values).image_embeds
        outputs.append(torch.nn.functional.normalize(features.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def build_record(
    *,
    record: dict[str, Any],
    image_path: Path,
    vinvl_path: Path,
    model,
    processor,
    device: torch.device,
    max_regions: int,
    min_score: float,
    fp16: bool,
    batch_size: int,
) -> dict[str, Any]:
    formal = select_formal_r16(vinvl_path, max_regions, min_score)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
        crops = crop_regions(image, formal["boxes"], formal["valid"])
        encoded = encode_images(
            model,
            processor,
            [image] + crops,
            device,
            fp16,
            batch_size,
        )
        image_width, image_height = image.size
    feature_dim = int(encoded.shape[-1])
    region_features = torch.zeros((max_regions, feature_dim), dtype=torch.float32)
    region_features[torch.from_numpy(formal["valid"])] = encoded[1:]
    dtype = torch.float16 if fp16 else torch.float32
    return {
        "record_ids": [str(record["id"])],
        "image_id": image_path.stem,
        "image_path": str(image_path.resolve()),
        "image_sha256": sha256_file(image_path),
        "image_size": torch.tensor([image_height, image_width], dtype=torch.int32),
        "vinvl_path": str(vinvl_path.resolve()),
        "vinvl_sha256": sha256_file(vinvl_path),
        "selected_vinvl_indices": torch.from_numpy(formal["selected_indices"]),
        "global_feature": encoded[0].to(dtype=dtype),
        "region_features": region_features.to(dtype=dtype),
        "region_boxes": torch.from_numpy(formal["boxes"]),
        "region_valid_mask": torch.from_numpy(formal["valid"]),
        "region_detector_scores": torch.from_numpy(formal["scores"]),
    }


def main() -> None:
    args = parse_args()
    split = require_train_or_dev_split(args.split)
    if args.batch_size < 1 or args.shard_size < 1:
        raise ValueError("batch-size and shard-size must be positive.")
    root = Path(__file__).resolve().parents[1]
    config_path = resolve(args.config, root)
    config = load_config(config_path)
    output_dir = resolve(args.output_dir, root)
    manifest_path = output_dir / "manifest.json"
    if args.resume and manifest_path.exists():
        from gmner.data.clip_r16_cache import validate_clip_r16_manifest

        validate_clip_r16_manifest(manifest_path, expected_split=split)
        print(json.dumps({"status": "already_complete", "manifest": str(manifest_path)}))
        return
    source_txt = resolve(getattr(config.data, f"{split}_file"), root)
    converted = maybe_convert_conll(source_txt, output_dir / "source")
    records = read_jsonl(converted)
    image_dir = resolve(config.data.image_dir, root)
    vinvl_dir = resolve(config.data.image_feature_dir, root)
    unique: dict[str, dict[str, Any]] = {}
    for record in records:
        image_name = str(record.get("image", ""))
        image_id = Path(image_name).stem
        if not image_id:
            raise ValueError(f"Record {record.get('id')} has no image identity.")
        if image_id in unique:
            unique[image_id]["record_ids"].append(str(record["id"]))
        else:
            unique[image_id] = {"record": record, "record_ids": [str(record["id"])]}

    from transformers import AutoImageProcessor, CLIPVisionModelWithProjection

    device = torch.device(args.device)
    processor = AutoImageProcessor.from_pretrained(
        args.model_name, local_files_only=True
    )
    model = CLIPVisionModelWithProjection.from_pretrained(
        args.model_name,
        local_files_only=True,
    ).to(device)
    model.eval()
    built: list[dict[str, Any]] = []
    for image_id, group in tqdm(unique.items(), desc=f"CLIP R16 {split}"):
        record = group["record"]
        image_name = str(record.get("image", f"{image_id}.jpg"))
        image_path = image_dir / image_name
        vinvl_path = vinvl_dir / f"{image_id}.jpg.npz"
        if not image_path.exists() or not vinvl_path.exists():
            raise FileNotFoundError(f"Missing image/VinVL input for {image_id}.")
        value = build_record(
            record=record,
            image_path=image_path,
            vinvl_path=vinvl_path,
            model=model,
            processor=processor,
            device=device,
            max_regions=int(config.data.max_regions),
            min_score=float(config.data.region_min_score),
            fp16=bool(args.fp16),
            batch_size=args.batch_size,
        )
        value["record_ids"] = group["record_ids"]
        built.append(value)
    processor_payload = (
        processor.to_dict() if hasattr(processor, "to_dict") else str(processor)
    )
    processor_payload = json.loads(json.dumps(processor_payload, default=str))
    preprocessing = {
        "processor_class": type(processor).__name__,
        "processor": processor_payload,
        "feature_normalization": "l2",
        "crop": "raw_bbox_clamped_to_image",
        "global_image": True,
    }
    metadata = {
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "source_path": str(source_txt.resolve()),
        "source_sha256": sha256_file(source_txt),
        "source_record_ids_sha256": stable_id_digest(str(item["id"]) for item in records),
        "model": model_artifact_fingerprint(args.model_name),
        "preprocessing": preprocessing,
        "preprocessing_sha256": stable_json_sha256(preprocessing),
        "region_min_score": float(config.data.region_min_score),
        "formal_region_budget": int(config.data.max_regions),
        "null_included": False,
    }
    result = write_clip_r16_cache(
        output_dir=output_dir,
        split=split,
        records=built,
        shard_size=args.shard_size,
        metadata=metadata,
    )
    print(json.dumps({"split": split, "images": len(built), "manifest": str(result)}, indent=2))


if __name__ == "__main__":
    main()
