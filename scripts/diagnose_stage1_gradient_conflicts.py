#!/usr/bin/env python3
"""Run the preregistered, Train-only Stage1 D0 gradient audit."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_config
from gmner.data import (
    GMNERCollator,
    MMNERJsonDataset,
    TextGraphBuilder,
    load_word_aligned_tokenizer,
    validate_model_input_length,
)
from gmner.data.graph_builders import GraphBuilderConfig
from gmner.data.null_release_oof_cache import sha256_file, stable_id_digest
from gmner.diagnostics import (
    aggregate_gradient_observations,
    compute_gradient_observation,
    encoder_layer_parameter_groups,
    stable_probe_record_ids,
)
from gmner.engine.utils import move_batch_to_device
from gmner.models import GMNERModel
from gmner.utils.io import maybe_convert_conll
from gmner.utils.seed import set_seed


REPORT_FORMAT_VERSION = 1
REPORT_KIND = "stage1_train_only_gradient_conflict_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose Stage1 NER/Grounding/Alignment gradient conflicts on a "
            "fixed Train-only probe. This entrypoint cannot access Dev/Test."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--text-model-name",
        default=None,
        help="Optional local text-backbone path override.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--probe-records", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Optional positive cap; zero audits every selected Train sample.",
    )
    parser.add_argument(
        "--layers",
        default="0,5,11",
        help="Comma-separated shared RoBERTa encoder layer indices.",
    )
    parser.add_argument("--min-valid-batches", type=int, default=10)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast. The formal D0 command leaves this disabled.",
    )
    return parser.parse_args()


def resolve_path(value: str | Path, root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def parse_layers(value: str) -> list[int]:
    try:
        layers = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--layers must contain comma-separated integers.") from exc
    if not layers or len(set(layers)) != len(layers):
        raise ValueError("--layers must contain unique layer indices.")
    return layers


def git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_worktree_dirty(root: Path) -> bool | None:
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return bool(status.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def stable_json_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_train_dataset(
    *,
    config,
    tokenizer,
    root: Path,
    output_path: Path,
) -> tuple[MMNERJsonDataset, Path]:
    if bool(getattr(config.model, "use_fine_subtype_head", False)):
        raise ValueError(
            "D0 targets the formal coarse Stage1 and does not accept Stage1-F."
        )

    train_source = resolve_path(config.data.train_file, root).resolve()
    forbidden_sources = {
        resolve_path(config.data.dev_file, root).resolve(),
        resolve_path(config.data.test_file, root).resolve(),
    }
    if train_source in forbidden_sources:
        raise ValueError("Configured Train source aliases Dev or Test.")
    if not train_source.exists():
        raise FileNotFoundError(f"Train source not found: {train_source}")

    cache_root = output_path.parent / "cache"
    train_path = maybe_convert_conll(train_source, cache_root)
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=config.data.use_dependency_graph,
            dependency_backend=config.data.dependency_backend,
            dependency_model=config.data.dependency_model,
            window_size=config.data.graph_window_size,
        )
    )

    def optional_path(value: str) -> str | None:
        if not value:
            return None
        return str(resolve_path(value, root))

    dataset = MMNERJsonDataset(
        jsonl_path=str(train_path),
        image_dir=str(resolve_path(config.data.image_dir, root)),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=config.data.grounding_enabled,
        expand_entities_for_grounding=config.data.expand_entities_for_grounding,
        image_feature_dir=optional_path(config.data.image_feature_dir),
        image_annotation_dir=optional_path(config.data.image_annotation_dir),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=optional_path(
            config.data.groundability_type_priors
        ),
        groundability_mention_priors=optional_path(
            config.data.groundability_mention_priors
        ),
        region_min_score=config.data.region_min_score,
    )
    return dataset, train_source


def select_probe(
    dataset: MMNERJsonDataset,
    *,
    probe_records: int,
    seed: int,
) -> tuple[list[str], list[int]]:
    record_ids = [
        str(record.get("id", index))
        for index, record in enumerate(dataset.records)
    ]
    selected_ids = stable_probe_record_ids(
        record_ids,
        count=probe_records,
        seed=seed,
    )
    selected_set = set(selected_ids)
    sample_indices = [
        index
        for index, sample in enumerate(dataset.samples)
        if str(sample.get("record_id", sample.get("sample_id"))) in selected_set
    ]
    observed = {
        str(dataset.samples[index].get("record_id"))
        for index in sample_indices
    }
    missing = selected_set - observed
    if missing:
        raise ValueError(
            f"Selected Train probe records have no materialized samples: {sorted(missing)}"
        )
    return selected_ids, sample_indices


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path = resolve_path(args.config, root).resolve()
    checkpoint_path = resolve_path(args.checkpoint, root).resolve()
    output_path = resolve_path(args.output, root).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if args.probe_records <= 0 or args.batch_size <= 0:
        raise ValueError("--probe-records and --batch-size must be positive.")
    if args.max_batches < 0:
        raise ValueError("--max-batches cannot be negative.")

    config = load_config(config_path)
    if args.text_model_name:
        config.model.text_model_name = args.text_model_name
    seed = int(config.runtime.seed if args.seed is None else args.seed)
    layers = parse_layers(args.layers)
    device_name = str(args.device or config.runtime.device)
    device = torch.device(
        device_name
        if device_name.startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    set_seed(seed)

    print(f"D0 device={device} amp={amp_enabled} seed={seed}")
    print(f"Loading tokenizer and Train-only dataset from {config.data.train_file}")
    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    backbone_config = AutoConfig.from_pretrained(config.model.text_model_name)
    validate_model_input_length(
        tokenizer,
        backbone_config,
        config.data.max_length,
    )
    dataset, train_source = build_train_dataset(
        config=config,
        tokenizer=tokenizer,
        root=root,
        output_path=output_path,
    )
    selected_ids, sample_indices = select_probe(
        dataset,
        probe_records=args.probe_records,
        seed=seed,
    )
    probe_dataset = Subset(dataset, sample_indices)
    loader = DataLoader(
        probe_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=GMNERCollator(tokenizer=tokenizer),
    )
    print(
        f"Train probe: records={len(selected_ids)} samples={len(sample_indices)} "
        f"batches={len(loader)}"
    )

    model = GMNERModel(config=config, num_labels=9)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    checkpoint_metadata = dict(checkpoint.get("model_metadata") or {})
    checkpoint_epoch = checkpoint.get("epoch")
    del checkpoint
    model.to(device)
    model.eval()
    model.zero_grad(set_to_none=True)
    layer_groups = encoder_layer_parameter_groups(model, layers)
    task_weights = {
        "ner": float(config.loss.lambda_ner),
        "grounding": float(config.loss.lambda_grounding),
        "alignment": float(config.loss.lambda_alignment),
    }

    observations: list[dict[str, Any]] = []
    processed_sample_count = 0
    for batch_index, batch in enumerate(loader):
        if args.max_batches and batch_index >= args.max_batches:
            break
        batch_record_ids = [
            str(item.get("record_id", item.get("sample_id")))
            for item in batch["metadata"]
        ]
        processed_sample_count += len(batch_record_ids)
        device_batch = move_batch_to_device(batch, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            outputs = model(device_batch)
        raw_task_losses = {
            "ner": outputs.get("task_loss_ner"),
            "grounding": outputs.get("task_loss_grounding"),
            "alignment": outputs.get("task_loss_alignment"),
        }
        effective_task_losses = {
            task_name: (
                loss * task_weights[task_name]
                if isinstance(loss, torch.Tensor)
                else loss
            )
            for task_name, loss in raw_task_losses.items()
        }
        observation = compute_gradient_observation(
            effective_task_losses,
            layer_groups,
        )
        observation["effective_task_losses"] = observation.pop("task_losses")
        observation["raw_task_losses"] = {
            task_name: float(loss.detach().float().item())
            for task_name, loss in raw_task_losses.items()
            if isinstance(loss, torch.Tensor) and loss.numel() == 1
        }
        observation.update(
            {
                "batch_index": batch_index,
                "record_ids": batch_record_ids,
            }
        )
        observations.append(observation)
        del outputs, device_batch, batch
        model.zero_grad(set_to_none=True)
        if device.type == "cuda" and (batch_index + 1) % 10 == 0:
            torch.cuda.empty_cache()
        if (batch_index + 1) % 5 == 0:
            print(f"Audited {batch_index + 1} batches")

    if not observations:
        raise RuntimeError("D0 did not produce any gradient observations.")

    summary = aggregate_gradient_observations(
        observations,
        min_valid_batches=args.min_valid_batches,
    )
    report = {
        "format_version": REPORT_FORMAT_VERSION,
        "kind": REPORT_KIND,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "split": "train_probe",
            "parameter_updates": 0,
            "dev_accessed": False,
            "test_accessed": False,
            "task_losses": [
                "task_loss_ner",
                "task_loss_grounding",
                "task_loss_alignment",
            ],
            "task_weights": task_weights,
            "gradient_norm_scope": "effective_weighted_task_losses",
            "layers": layers,
            "amp": amp_enabled,
        },
        "inputs": {
            "diagnostic_script": str(Path(__file__).resolve()),
            "diagnostic_script_sha256": sha256_file(Path(__file__).resolve()),
            "diagnostic_module": str(
                root
                / "gmner"
                / "diagnostics"
                / "stage1_gradient_conflicts.py"
            ),
            "diagnostic_module_sha256": sha256_file(
                root
                / "gmner"
                / "diagnostics"
                / "stage1_gradient_conflicts.py"
            ),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_metadata": checkpoint_metadata,
            "train_source": str(train_source),
            "train_source_sha256": sha256_file(train_source),
            "text_model_name": str(config.model.text_model_name),
            "code_commit": git_commit(root),
            "git_worktree_dirty": git_worktree_dirty(root),
        },
        "probe": {
            "seed": seed,
            "requested_record_count": int(args.probe_records),
            "selected_record_count": len(selected_ids),
            "selected_record_ids": selected_ids,
            "selected_record_ids_sha256": stable_id_digest(selected_ids),
            "selection_contract_sha256": stable_json_digest(
                {
                    "seed": seed,
                    "record_ids": selected_ids,
                    "algorithm": "sha256(seed + NUL + record_id)",
                }
            ),
            "materialized_sample_count": len(sample_indices),
            "processed_sample_count": processed_sample_count,
            "batch_size": int(args.batch_size),
            "processed_batch_count": len(observations),
            "max_batches": int(args.max_batches),
        },
        "summary": summary,
        "observations": observations,
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    temporary_path.replace(output_path)

    recommendation = summary["recommendation"]
    print(
        json.dumps(
            {
                "output": str(output_path),
                "observation_count": summary["observation_count"],
                "status": recommendation["status"],
                "has_significant_conflict": recommendation[
                    "has_significant_conflict"
                ],
                "recommend_d2": recommendation["recommend_d2"],
                "most_conflicted_site": recommendation[
                    "most_conflicted_site"
                ],
                "dev_accessed": False,
                "test_accessed": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    del model, dataset, probe_dataset, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
