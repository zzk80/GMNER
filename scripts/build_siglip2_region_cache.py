"""Cache frozen multi-view SigLIP 2 features for M3.4A reliability."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.constants import ID2ENTITY_TYPE
from gmner.data.paired_record_candidate_dataset import (
    PairedRecordCandidateDataset,
)
from gmner.data.record_candidate_dataset import RecordCandidateDataset
from gmner.data.siglip2_region_cache import (
    IMAGE_VIEW_NAMES,
    SIGLIP2_CACHE_FORMAT_VERSION,
    TEXT_VIEW_NAMES,
    sha256_file,
)
from gmner.utils.io import read_jsonl
from scripts.convert_gmner_conll_to_jsonl import parse_conll


TYPE_PROMPTS = {
    "PER": "a photo of {mention}, a person",
    "ORG": "a photo related to {mention}, an organization",
    "LOC": "a photo of {mention}, a location",
    "OTHER": "a photo related to {mention}, an entity",
}
PROCESSOR_FILES = {
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-cache", required=True)
    parser.add_argument("--expanded-cache", required=True)
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "dev"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shard-size", type=int, default=128)
    parser.add_argument("--context-expansion", type=float, default=1.5)
    parser.add_argument("--minimum-crop-side", type=int, default=2)
    parser.add_argument("--text-max-length", type=int, default=64)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-records", type=int, default=None)
    return parser.parse_args()


def _resolve(path: str | Path, root: Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _source_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".txt":
        return parse_conll(path, image_ext=".jpg")
    return read_jsonl(path)


def _directory_fingerprint(path: Path, *, processor_only: bool = False) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        return sha256_file(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if processor_only:
        files = [item for item in files if item.name in PROCESSOR_FILES]
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _stable_tensor_hash(value: torch.Tensor, decimals: int = 3) -> str:
    scale = float(10**decimals)
    quantized = torch.round(value.detach().float().cpu() * scale).to(torch.int64)
    return hashlib.sha256(quantized.numpy().tobytes()).hexdigest()


def _safe_image(path: Path, resolution: int) -> tuple[Image.Image, bool]:
    try:
        with Image.open(path) as image:
            return image.convert("RGB"), False
    except (OSError, ValueError):
        return Image.new("RGB", (resolution, resolution), color=0), True


def _finite_box(box: Iterable[float]) -> tuple[float, float, float, float]:
    values = [float(value) for value in box]
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        return 0.0, 0.0, 0.0, 0.0
    x1, y1, x2, y2 = values
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def candidate_local_crop(
    image: Image.Image,
    box: Iterable[float],
    *,
    minimum_side: int = 2,
) -> tuple[Image.Image, bool]:
    """Clip a candidate crop to the image and flag invalid/empty boxes."""

    x1, y1, x2, y2 = _finite_box(box)
    width, height = image.size
    x1, x2 = max(0.0, x1), min(float(width), x2)
    y1, y2 = max(0.0, y1), min(float(height), y2)
    if x2 - x1 < minimum_side or y2 - y1 < minimum_side:
        return Image.new("RGB", (minimum_side, minimum_side), color=0), True
    bounds = (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))
    return image.crop(bounds), False


def candidate_context_crop(
    image: Image.Image,
    box: Iterable[float],
    *,
    expansion: float = 1.5,
    minimum_side: int = 2,
) -> tuple[Image.Image, bool]:
    """Create a square expanded crop; PIL pads out-of-bounds pixels with black."""

    x1, y1, x2, y2 = _finite_box(box)
    width = x2 - x1
    height = y2 - y1
    if width < minimum_side or height < minimum_side:
        return Image.new("RGB", (minimum_side, minimum_side), color=0), True
    side = max(width, height) * max(float(expansion), 1.0)
    side = max(side, float(minimum_side))
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    integer_side = max(int(math.ceil(side)), int(minimum_side))
    left = math.floor(center_x - integer_side / 2)
    top = math.floor(center_y - integer_side / 2)
    bounds = (left, top, left + integer_side, top + integer_side)
    return image.crop(bounds), False


def _feature_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    for name in ("pooler_output", "image_embeds", "text_embeds"):
        tensor = getattr(value, name, None)
        if isinstance(tensor, torch.Tensor):
            return tensor
    raise TypeError(f"Unsupported SigLIP 2 feature output: {type(value)!r}")


@torch.no_grad()
def _encode_images(
    model,
    processor,
    images: list[Image.Image],
    *,
    device: torch.device,
    batch_size: int,
    fp16: bool,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(images), max(1, int(batch_size))):
        encoded = processor(
            images=images[start : start + batch_size], return_tensors="pt"
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(fp16 and device.type == "cuda"),
        ):
            features = _feature_tensor(model.get_image_features(**encoded))
        outputs.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def _encode_texts(
    model,
    processor,
    texts: list[str],
    *,
    device: torch.device,
    batch_size: int,
    max_length: int,
    fp16: bool,
) -> torch.Tensor:
    outputs = []
    for start in range(0, len(texts), max(1, int(batch_size))):
        encoded = processor(
            text=texts[start : start + batch_size],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
            if isinstance(value, torch.Tensor)
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=bool(fp16 and device.type == "cuda"),
        ):
            features = _feature_tensor(model.get_text_features(**encoded))
        outputs.append(F.normalize(features.float(), dim=-1).cpu())
    return torch.cat(outputs, dim=0)


def _atomic_torch_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    os.replace(temporary, path)


def _native_scale_bias(model) -> tuple[float, float]:
    scale = getattr(model, "logit_scale", None)
    bias = getattr(model, "logit_bias", None)
    scale_value = float(scale.detach().float().exp().cpu()) if scale is not None else 1.0
    bias_value = float(bias.detach().float().cpu()) if bias is not None else 0.0
    return scale_value, bias_value


def _processor_resolution(processor) -> int:
    size = getattr(getattr(processor, "image_processor", None), "size", 224)
    if isinstance(size, dict):
        return int(size.get("height") or size.get("shortest_edge") or 224)
    return int(size)


class SeparatedSiglipProcessor:
    """Combine a Gemma tokenizer with the legacy SigLIP image processor."""

    def __init__(self, tokenizer, image_processor) -> None:
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __call__(self, *, text=None, images=None, **kwargs):
        if (text is None) == (images is None):
            raise ValueError("Pass exactly one of text or images.")
        if text is not None:
            return self.tokenizer(text=text, **kwargs)
        return self.image_processor(images=images, **kwargs)


def load_separated_processor(model_path: Path) -> SeparatedSiglipProcessor:
    """Load SigLIP2 components without the old AutoProcessor class mismatch."""

    from transformers import AutoImageProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=False,
    )
    image_processor = AutoImageProcessor.from_pretrained(
        model_path,
        local_files_only=True,
    )
    return SeparatedSiglipProcessor(tokenizer, image_processor)


def main() -> None:
    args = parse_args()
    if args.context_expansion < 1.0:
        raise ValueError("--context-expansion must be at least 1.0.")
    root = Path(__file__).resolve().parents[1]
    formal_path = _resolve(args.formal_cache, root)
    expanded_path = _resolve(args.expanded_cache, root)
    source_path = _resolve(args.source_file, root)
    image_dir = _resolve(args.image_dir, root)
    model_path = _resolve(args.model_name, root)
    output_dir = _resolve(args.output_dir, root)
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(exist_ok=True)

    formal = RecordCandidateDataset(formal_path)
    expanded = RecordCandidateDataset(expanded_path)
    paired = PairedRecordCandidateDataset(formal, expanded)
    source_records = _source_records(source_path)
    source_by_id = {str(record.get("id")): record for record in source_records}
    if len(source_by_id) != len(source_records):
        raise ValueError("Source data contains duplicate or missing record ids.")

    from transformers import AutoModel

    device = torch.device(
        args.device
        if str(args.device).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    processor = load_separated_processor(model_path)
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    resolution = _processor_resolution(processor)
    text_limit = int(
        getattr(getattr(model.config, "text_config", None), "max_position_embeddings", 64)
    )
    text_max_length = min(max(4, int(args.text_max_length)), text_limit)
    logit_scale, logit_bias = _native_scale_bias(model)

    metadata = {
        "format_version": SIGLIP2_CACHE_FORMAT_VERSION,
        "split": args.split,
        "model_name": str(model_path),
        "model_sha256": _directory_fingerprint(model_path),
        "processor_sha256": _directory_fingerprint(model_path, processor_only=True),
        "processor_use_fast": False,
        "formal_cache_sha256": sha256_file(formal_path),
        "expanded_cache_sha256": sha256_file(expanded_path),
        "formal_candidate_config_sha256": formal.metadata.get(
            "candidate_config_sha256"
        ),
        "expanded_candidate_config_sha256": expanded.metadata.get(
            "candidate_config_sha256"
        ),
        "source_sha256": sha256_file(source_path),
        "input_resolution": resolution,
        "context_expansion": float(args.context_expansion),
        "minimum_crop_side": int(args.minimum_crop_side),
        "text_max_length": text_max_length,
        "text_views": list(TEXT_VIEW_NAMES),
        "image_views": list(IMAGE_VIEW_NAMES),
        "logit_scale": logit_scale,
        "logit_bias": logit_bias,
    }
    signature = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    metadata["build_signature"] = signature
    manifest_path = output_dir / "manifest.json"
    entries: list[dict] = []
    start_index = 0
    previous: dict = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not args.resume:
            raise FileExistsError(
                f"Cache already exists at {manifest_path}; pass --resume or use a new directory."
            )
        if previous.get("build_signature") != signature:
            raise ValueError("Cannot resume: SigLIP 2 cache build signature changed.")
        entries = list(previous.get("records") or [])
        start_index = len(entries)
    target_count = len(paired)
    if args.max_records is not None:
        target_count = min(target_count, max(0, int(args.max_records)))
    if start_index > target_count:
        raise ValueError("Existing cache contains more records than requested.")

    pending: list[dict] = []
    previous_diagnostics = previous.get("diagnostics") or {}
    fallback_images = int(previous_diagnostics.get("fallback_images", 0))
    invalid_local = int(previous_diagnostics.get("invalid_local_crops", 0))
    invalid_context = int(previous_diagnostics.get("invalid_context_crops", 0))
    feature_size = int(previous.get("feature_size", 0)) or None
    progress = tqdm(range(start_index, target_count), desc=f"SigLIP2 {args.split}")
    for index in progress:
        pair = paired[index]
        expanded_record = pair["expanded"]
        record_id = str((expanded_record.get("metadata") or {}).get("record_id"))
        source = source_by_id.get(record_id)
        if source is None:
            raise ValueError(f"Source data is missing record {record_id}.")
        image_path = image_dir / str(source["image"])
        image, fallback = _safe_image(image_path, resolution)
        fallback_images += int(fallback)
        boxes = torch.as_tensor(expanded_record["region_boxes"]).float()
        valid_regions = (
            torch.as_tensor(expanded_record["region_mask"]).bool()
            & ~torch.as_tensor(expanded_record["region_is_null"]).bool()
        )
        local_images: list[Image.Image] = []
        context_images: list[Image.Image] = []
        valid_indices = torch.nonzero(valid_regions, as_tuple=False).squeeze(-1).tolist()
        for region_index in valid_indices:
            local, bad_local = candidate_local_crop(
                image,
                boxes[region_index].tolist(),
                minimum_side=args.minimum_crop_side,
            )
            context, bad_context = candidate_context_crop(
                image,
                boxes[region_index].tolist(),
                expansion=args.context_expansion,
                minimum_side=args.minimum_crop_side,
            )
            local_images.append(local)
            context_images.append(context)
            invalid_local += int(bad_local)
            invalid_context += int(bad_context)
        image_features = _encode_images(
            model,
            processor,
            [*local_images, *context_images, image],
            device=device,
            batch_size=args.batch_size,
            fp16=args.fp16,
        )
        count = len(valid_indices)
        current_feature_size = int(image_features.size(-1))
        feature_size = feature_size or current_feature_size
        if feature_size != current_feature_size:
            raise ValueError("SigLIP 2 feature size changed during caching.")
        local_features = torch.zeros(boxes.size(0), feature_size)
        context_features = torch.zeros_like(local_features)
        if count:
            indices = torch.tensor(valid_indices, dtype=torch.long)
            local_features[indices] = image_features[:count]
            context_features[indices] = image_features[count : 2 * count]
        global_feature = image_features[-1]

        tokens = list(source.get("tokens") or [])
        text = " ".join(tokens)
        spans = torch.as_tensor(expanded_record["span_candidates"]).long()
        fixed_types = torch.as_tensor(expanded_record["fixed_type_ids"]).long()
        prompts: list[str] = []
        for span, type_id in zip(spans.tolist(), fixed_types.tolist()):
            start, end = map(int, span)
            mention = " ".join(tokens[start:end]).strip() or "entity"
            entity_type = ID2ENTITY_TYPE.get(int(type_id), "OTHER")
            prompts.extend(
                [
                    mention,
                    f'The text says: "{text}". The entity is "{mention}".',
                    TYPE_PROMPTS.get(entity_type, TYPE_PROMPTS["OTHER"]).format(
                        mention=mention
                    ),
                ]
            )
        text_features = _encode_texts(
            model,
            processor,
            prompts,
            device=device,
            batch_size=args.batch_size,
            max_length=text_max_length,
            fp16=args.fp16,
        ).reshape(spans.size(0), len(TEXT_VIEW_NAMES), feature_size)
        record = {
            "span_candidates": spans,
            "span_feature_mask": torch.as_tensor(
                expanded_record["span_mask"]
            ).bool(),
            "text_features": text_features.to(torch.float16),
            "region_boxes": boxes,
            "region_feature_mask": valid_regions,
            "local_features": local_features.to(torch.float16),
            "context_features": context_features.to(torch.float16),
            "global_feature": global_feature.to(torch.float16),
            "logit_scale": logit_scale,
            "logit_bias": logit_bias,
            "metadata": {
                "record_id": record_id,
                "image_id": Path(str(source["image"])).stem,
                "image_sha256": sha256_file(image_path) if image_path.exists() else "",
                "image_fallback": bool(fallback),
                "bbox_sha256": _stable_tensor_hash(boxes),
            },
        }
        pending.append(record)
        if len(pending) >= args.shard_size or index + 1 == target_count:
            shard_number = len({entry["shard"] for entry in entries})
            shard_name = f"shards/shard_{shard_number:05d}.pt"
            _atomic_torch_save({"records": pending}, output_dir / shard_name)
            for offset, item in enumerate(pending):
                entries.append(
                    {
                        "record_id": str(item["metadata"]["record_id"]),
                        "shard": shard_name,
                        "offset": offset,
                    }
                )
            pending = []
            manifest = {
                **metadata,
                "record_count": len(entries),
                "feature_size": int(feature_size or 0),
                "records": entries,
                "diagnostics": {
                    "fallback_images": fallback_images,
                    "invalid_local_crops": invalid_local,
                    "invalid_context_crops": invalid_context,
                },
            }
            _atomic_json(manifest, manifest_path)
            progress.set_postfix(shards=shard_number + 1)

    print(
        json.dumps(
            {
                "split": args.split,
                "records": len(entries),
                "feature_size": int(feature_size or 0),
                "fallback_images": fallback_images,
                "invalid_local_crops": invalid_local,
                "invalid_context_crops": invalid_context,
                "saved_to": str(output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
