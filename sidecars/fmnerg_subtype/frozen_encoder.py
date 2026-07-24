"""Frozen Stage1 RoBERTa loading and deterministic span feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml
from tqdm.auto import tqdm
from transformers import AutoModel

from gmner.data.tokenization import load_word_aligned_tokenizer

from .io import resolve_path, sha256_file


def stage1_text_settings(
    stage1_config_path: str | Path,
    root: Path,
) -> tuple[str, int]:
    config = yaml.safe_load(
        resolve_path(stage1_config_path, root).read_text(encoding="utf-8")
    )
    return (
        str(config["model"]["text_model_name"]),
        int(config["data"]["max_length"]),
    )


def load_frozen_stage1_backbone(
    *,
    stage1_config_path: str | Path,
    stage1_checkpoint_path: str | Path,
    root: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    model_name, max_length = stage1_text_settings(stage1_config_path, root)
    tokenizer = load_word_aligned_tokenizer(model_name)
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError(
            "The formal RoBERTa subtype sidecar requires a fast tokenizer."
        )
    backbone = AutoModel.from_pretrained(model_name)
    checkpoint_path = resolve_path(stage1_checkpoint_path, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    full_state = dict(checkpoint.get("model_state_dict") or {})
    prefix = "text_encoder.backbone."
    backbone_state = {
        key[len(prefix) :]: value
        for key, value in full_state.items()
        if key.startswith(prefix)
    }
    if not backbone_state:
        raise ValueError(
            f"Stage1 checkpoint contains no {prefix!r} backbone parameters."
        )
    incompatible = backbone.load_state_dict(backbone_state, strict=False)
    allowed_missing_suffixes = ("position_ids", "token_type_ids")
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.endswith(allowed_missing_suffixes)
    ]
    if missing or incompatible.unexpected_keys:
        raise ValueError(
            "Frozen Stage1 text-backbone mismatch: "
            f"missing={missing}, unexpected={incompatible.unexpected_keys}"
        )
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False
    hidden_size = int(backbone.config.hidden_size)
    metadata = {
        "stage1_checkpoint": str(checkpoint_path),
        "stage1_checkpoint_sha256": sha256_file(checkpoint_path),
        "stage1_checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "text_model_name": model_name,
        "max_length": max_length,
        "hidden_size": hidden_size,
        "feature_size": hidden_size * 3,
        "pooling": "first_subword(start)+last_subword(end-1)+mean(all_span_subwords)",
        "base_model_training": False,
        "base_model_requires_grad": False,
    }
    return backbone, tokenizer, metadata


@torch.inference_mode()
def encode_record_spans(
    *,
    backbone: torch.nn.Module,
    tokenizer: Any,
    records: list[dict[str, Any]],
    max_length: int,
    batch_size: int,
    device: torch.device,
    fp16: bool,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    backbone.eval()
    features: list[torch.Tensor] = []
    examples: list[dict[str, Any]] = []
    amp_enabled = bool(fp16 and device.type == "cuda")

    for offset in tqdm(
        range(0, len(records), max(1, int(batch_size))),
        desc="Encoding frozen subtype span features",
    ):
        group = records[offset : offset + max(1, int(batch_size))]
        token_lists = [list(item["tokens"]) for item in group]
        encoding = tokenizer(
            token_lists,
            is_split_into_words=True,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        word_ids = [
            list(encoding.word_ids(batch_index=index))
            for index in range(len(group))
        ]
        model_inputs = {
            key: value.to(device)
            for key, value in encoding.items()
            if key in {"input_ids", "attention_mask", "token_type_ids"}
        }
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            hidden = backbone(**model_inputs).last_hidden_state

        for row, item in enumerate(group):
            row_word_ids = word_ids[row]
            for span in item.get("spans") or []:
                start = int(span["start"])
                end = int(span["end"])
                positions = [
                    index
                    for index, word_id in enumerate(row_word_ids)
                    if word_id is not None and start <= int(word_id) < end
                ]
                if not positions:
                    raise ValueError(
                        f"Span [{start}, {end}) in record {item['record_id']} "
                        f"was truncated by max_length={max_length}."
                    )
                start_positions = [
                    index
                    for index, word_id in enumerate(row_word_ids)
                    if word_id is not None and int(word_id) == start
                ]
                end_positions = [
                    index
                    for index, word_id in enumerate(row_word_ids)
                    if word_id is not None and int(word_id) == end - 1
                ]
                if not start_positions or not end_positions:
                    raise ValueError(
                        f"Boundary subwords missing for record {item['record_id']} "
                        f"span [{start}, {end})."
                    )
                token_states = hidden[row]
                pooled = torch.cat(
                    (
                        token_states[start_positions[0]],
                        token_states[end_positions[-1]],
                        token_states[positions].mean(dim=0),
                    ),
                    dim=-1,
                )
                features.append(
                    pooled.detach().cpu().to(
                        dtype=torch.float16 if fp16 else torch.float32
                    )
                )
                examples.append(
                    {
                        "record_id": str(item["record_id"]),
                        "span": [start, end],
                        "text": " ".join(item["tokens"][start:end]),
                        **{
                            key: value
                            for key, value in span.items()
                            if key not in {"start", "end"}
                        },
                    }
                )
    if not features:
        hidden_size = int(backbone.config.hidden_size)
        return torch.empty((0, hidden_size * 3)), examples
    return torch.stack(features), examples
