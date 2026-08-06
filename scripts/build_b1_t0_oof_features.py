#!/usr/bin/env python3
"""Build frozen text-only span features for the B1-T0 OOF population."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data.artifact_utils import sha256_file, stable_id_digest
from gmner.data.full_chain_oof_contract import validate_fold_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--authorization",
        default="docs/experiments/b1_t0_oof_separability_authorization.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def reject_dev_test_path(path: Path) -> None:
    normalized = str(path).replace("\\", "/").casefold()
    if "/dev" in normalized or "_dev" in normalized or "/test" in normalized or "_test" in normalized:
        raise PermissionError(f"B1-T0 cannot access Dev/Test paths: {path}")


def directory_fingerprint(path: Path) -> dict[str, Any]:
    names = (
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    )
    files = [path / name for name in names if (path / name).is_file()]
    if not files or not any(item.name in {"model.safetensors", "pytorch_model.bin"} for item in files):
        raise FileNotFoundError("Frozen RoBERTa model files are incomplete.")
    digest = hashlib.sha256()
    descriptors = []
    for file_path in files:
        sha = sha256_file(file_path)
        digest.update(file_path.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha))
        descriptors.append(
            {"name": file_path.name, "bytes": file_path.stat().st_size, "sha256": sha}
        )
    return {"path": str(path), "sha256": digest.hexdigest(), "files": descriptors}


def validate_authorization(payload: dict[str, Any]) -> None:
    if (
        payload.get("kind") != "b1_t0_oof_separability_authorization"
        or payload.get("status") != "AUTHORIZED"
    ):
        raise PermissionError("B1-T0 authorization is invalid.")
    forbidden = dict(payload.get("forbidden") or {})
    for key in ("a1_training", "b1_tv", "visual_features", "clip", "dev_access", "test_access"):
        if forbidden.get(key) is not True:
            raise PermissionError(f"B1-T0 lock is disabled: {key}")


def lexical_features(tokens: list[str]) -> dict[str, float | int | str]:
    mention = " ".join(tokens)
    characters = "".join(tokens)
    alpha = [character for character in characters if character.isalpha()]
    return {
        "mention": mention.casefold(),
        "span_word_length": len(tokens),
        "span_character_length": len(characters),
        "uppercase_ratio": (
            sum(character.isupper() for character in alpha) / len(alpha) if alpha else 0.0
        ),
        "digit_ratio": (
            sum(character.isdigit() for character in characters) / len(characters)
            if characters
            else 0.0
        ),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    authorization_path = resolve(root, args.authorization)
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    validate_authorization(authorization)
    inputs = dict(authorization["input_contract"])
    rows_path = resolve(root, inputs["gold_free_rows"])
    sidecar_path = resolve(root, inputs["supervision_sidecar"])
    manifest_path = resolve(root, inputs["fold_manifest"])
    for path in (authorization_path, rows_path, sidecar_path, manifest_path):
        reject_dev_test_path(path)
    if sha256_file(rows_path) != inputs["gold_free_rows_sha256"]:
        raise RuntimeError("B1-T0 gold-free row SHA256 changed.")
    if sha256_file(sidecar_path) != inputs["supervision_sidecar_sha256"]:
        raise RuntimeError("B1-T0 supervision sidecar SHA256 changed.")
    manifest = validate_fold_manifest(manifest_path, expected_num_folds=10)
    rows = read_jsonl(rows_path)
    sidecars = read_jsonl(sidecar_path)
    if len(rows) != 7000 or len(sidecars) != 7000:
        raise RuntimeError("B1-T0 merged population must contain 7000 records.")
    row_by_id = {str(row["record_id"]): row for row in rows}
    sidecar_by_id = {str(row["record_id"]): row for row in sidecars}
    if set(row_by_id) != set(sidecar_by_id):
        raise RuntimeError("B1-T0 row and sidecar record IDs differ.")

    records: dict[str, dict[str, Any]] = {}
    fold_ids: dict[int, list[str]] = {}
    for fold in manifest["folds"]:
        fold_id = int(fold["fold"])
        heldout_path = Path(fold["heldout_file"]).resolve()
        reject_dev_test_path(heldout_path)
        heldout = read_jsonl(heldout_path)
        ids = [str(record["id"]) for record in heldout]
        if ids != [str(value) for value in fold["heldout_record_ids"]]:
            raise RuntimeError(f"Fold {fold_id} held-out text order changed.")
        fold_ids[fold_id] = ids
        for record in heldout:
            record_id = str(record["id"])
            if record_id in records:
                raise RuntimeError("A Train record occurs in multiple folds.")
            records[record_id] = record
    if set(records) != set(row_by_id):
        raise RuntimeError("Frozen text sources do not cover the merged OOF population.")

    from transformers import AutoModel, AutoTokenizer

    model_path = Path(authorization["feature_contract"]["text_encoder_path"]).resolve()
    reject_dev_test_path(model_path)
    model_fingerprint = directory_fingerprint(model_path)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    model = AutoModel.from_pretrained(str(model_path), add_pooling_layer=False)
    model.requires_grad_(False)
    model.eval().to(args.device)
    output_root = resolve(root, authorization["output_contract"]["root"]) / "features"
    output_root.mkdir(parents=True, exist_ok=True)
    feature_descriptors = []
    total_examples = 0
    rows_sha_before = sha256_file(rows_path)
    sidecar_sha_before = sha256_file(sidecar_path)

    with torch.no_grad():
        for fold_id in range(10):
            examples: list[dict[str, Any]] = []
            ids = fold_ids[fold_id]
            for offset in range(0, len(ids), int(args.batch_size)):
                batch_ids = ids[offset : offset + int(args.batch_size)]
                batch_tokens = [list(records[record_id]["tokens"]) for record_id in batch_ids]
                encoded = tokenizer(
                    batch_tokens,
                    is_split_into_words=True,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                model_inputs = {key: value.to(args.device) for key, value in encoded.items()}
                hidden = model(**model_inputs).last_hidden_state.detach().cpu()
                for batch_index, record_id in enumerate(batch_ids):
                    row = row_by_id[record_id]
                    sidecar = sidecar_by_id[record_id]
                    labels = {
                        item["prediction_id"]: item for item in sidecar["b1_predictions"]
                    }
                    word_ids = encoded.word_ids(batch_index=batch_index)
                    tokens = batch_tokens[batch_index]
                    for prediction in row["formal_predictions"]:
                        label = labels[prediction["prediction_id"]]
                        if not label["exact_span"]:
                            continue
                        start = int(prediction["span"]["start"])
                        end = int(prediction["span"]["end"])
                        positions = [
                            index
                            for index, word_id in enumerate(word_ids)
                            if word_id is not None and start <= int(word_id) < end
                        ]
                        if not positions:
                            raise RuntimeError(
                                f"Exact prediction was truncated: {record_id}:{start}:{end}"
                            )
                        token_states = hidden[batch_index, positions].float()
                        text_embedding = torch.cat(
                            [token_states[0], token_states[-1], token_states.mean(dim=0)],
                            dim=-1,
                        ).half()
                        if not torch.isfinite(text_embedding).all():
                            raise ValueError("Non-finite frozen text embedding.")
                        lexical = lexical_features(tokens[start:end])
                        logits = [float(value) for value in prediction["type_logits"]]
                        if len(logits) != 4 or not all(math.isfinite(value) for value in logits):
                            raise ValueError("Invalid B1-T0 type logits.")
                        examples.append(
                            {
                                "record_id": record_id,
                                "fold_id": fold_id,
                                "prediction_id": prediction["prediction_id"],
                                "span_start": start,
                                "span_end": end,
                                "base_type_id": int(label["base_type_id"]),
                                "gold_type_id": int(label["gold_type_id"]),
                                "base_wrong": label["population_label"] == "base_wrong",
                                "type_logits": logits,
                                "span_base_score": float(
                                    prediction["observable_features"]["span_base_score"]
                                ),
                                "text_embedding": text_embedding,
                                **lexical,
                            }
                        )
            output = output_root / f"fold{fold_id}.pt"
            torch.save(
                {
                    "kind": "b1_t0_frozen_text_fold_features",
                    "format_version": 1,
                    "fold_id": fold_id,
                    "examples": examples,
                    "baseline_counts": {
                        "records": len(ids),
                        "prediction_count": sum(
                            len(row_by_id[record_id]["formal_predictions"])
                            for record_id in ids
                        ),
                        "gold_count": sum(
                            int(sidecar_by_id[record_id]["gold_entity_count"])
                            for record_id in ids
                        ),
                        "mner_correct": sum(
                            int(sidecar_by_id[record_id]["base_mner_correct_count"])
                            for record_id in ids
                        ),
                    },
                    "text_encoder_sha256": model_fingerprint["sha256"],
                    "gold_free_rows_sha256": rows_sha_before,
                    "supervision_sidecar_sha256": sidecar_sha_before,
                    "dev_accessed": False,
                    "test_accessed": False,
                },
                output,
            )
            feature_descriptors.append(
                {
                    "fold_id": fold_id,
                    "path": str(output),
                    "sha256": sha256_file(output),
                    "examples": len(examples),
                    "base_wrong": sum(item["base_wrong"] for item in examples),
                    "base_correct": sum(not item["base_wrong"] for item in examples),
                    "baseline_counts": {
                        "records": len(ids),
                        "prediction_count": sum(
                            len(row_by_id[record_id]["formal_predictions"])
                            for record_id in ids
                        ),
                        "gold_count": sum(
                            int(sidecar_by_id[record_id]["gold_entity_count"])
                            for record_id in ids
                        ),
                        "mner_correct": sum(
                            int(sidecar_by_id[record_id]["base_mner_correct_count"])
                            for record_id in ids
                        ),
                    },
                }
            )
            total_examples += len(examples)
            print(f"Fold {fold_id}: {len(examples)} exact-span examples", flush=True)
    if sha256_file(rows_path) != rows_sha_before or sha256_file(sidecar_path) != sidecar_sha_before:
        raise RuntimeError("B1-T0 feature extraction mutated the sealed population.")
    feature_manifest = {
        "kind": "b1_t0_frozen_text_feature_manifest",
        "format_version": 1,
        "status": "PASSED",
        "authorization": str(authorization_path),
        "authorization_sha256": sha256_file(authorization_path),
        "text_encoder": model_fingerprint,
        "text_encoder_frozen": True,
        "span_pooling": authorization["feature_contract"]["span_pooling"],
        "records": 7000,
        "record_ids_sha256": stable_id_digest([row["record_id"] for row in rows]),
        "exact_span_examples": total_examples,
        "features": feature_descriptors,
        "gold_free_rows_sha256_before": rows_sha_before,
        "gold_free_rows_sha256_after": sha256_file(rows_path),
        "supervision_sidecar_sha256_before": sidecar_sha_before,
        "supervision_sidecar_sha256_after": sha256_file(sidecar_path),
        "visual_features": False,
        "dev_accessed": False,
        "test_accessed": False,
    }
    manifest_output = output_root / "feature_manifest.json"
    manifest_output.write_text(
        json.dumps(feature_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(feature_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
