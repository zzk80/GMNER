"""Verify that zero-initialized PA1 exactly preserves formal Stage1 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
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
from gmner.engine import evaluate_model
from gmner.engine.utils import move_batch_to_device
from gmner.models import GMNERModel
from gmner.utils.io import maybe_convert_conll


PROTECTED_PREFIXES = (
    "protected_region_adapter.",
    "protected_bidirectional_attention.",
    "protected_visual_type_head.",
)
FORMAL_TENSORS = (
    "base_text_nodes",
    "text_graph_nodes",
    "base_ner_logits",
    "pre_prototype_fused_tokens",
    "ner_logits",
    "image_nodes",
    "alignment_score",
    "grounding_logits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/fmnerg_twitter10000_stage1.yaml",
    )
    parser.add_argument(
        "--protected-config",
        default="configs/protected_region_mner/pa1_phase1_seed42.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        default="outputs/fmnerg_stage1_roberta128/best_model.pt",
    )
    parser.add_argument(
        "--output",
        default="outputs/protected_region_mner/pa1_epoch0_equivalence.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=0.0)
    return parser.parse_args()


def resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def build_dev_loader(config, root: Path) -> DataLoader:
    tokenizer = load_word_aligned_tokenizer(config.model.text_model_name)
    backbone_config = AutoConfig.from_pretrained(config.model.text_model_name)
    validate_model_input_length(tokenizer, backbone_config, config.data.max_length)
    graph_builder = TextGraphBuilder(
        GraphBuilderConfig(
            use_dependency_graph=config.data.use_dependency_graph,
            dependency_backend=config.data.dependency_backend,
            dependency_model=config.data.dependency_model,
            window_size=config.data.graph_window_size,
        )
    )
    output_dir = resolve(config.runtime.output_dir, root)
    data_path = maybe_convert_conll(resolve(config.data.dev_file, root), output_dir)
    dataset = MMNERJsonDataset(
        jsonl_path=str(data_path),
        image_dir=str(resolve(config.data.image_dir, root)),
        tokenizer=tokenizer,
        graph_builder=graph_builder,
        max_length=config.data.max_length,
        grounding_enabled=config.data.grounding_enabled,
        expand_entities_for_grounding=config.data.expand_entities_for_grounding,
        image_feature_dir=str(resolve(config.data.image_feature_dir, root)),
        image_annotation_dir=str(resolve(config.data.image_annotation_dir, root)),
        max_regions=config.data.max_regions,
        region_feature_dim=config.model.region_feature_dim,
        grounding_iou_threshold=config.data.grounding_iou_threshold,
        add_null_region=config.data.add_null_region,
        groundability_type_priors=(
            str(resolve(config.data.groundability_type_priors, root))
            if config.data.groundability_type_priors
            else None
        ),
        groundability_mention_priors=(
            str(resolve(config.data.groundability_mention_priors, root))
            if config.data.groundability_mention_priors
            else None
        ),
        region_min_score=config.data.region_min_score,
    )
    return DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=GMNERCollator(tokenizer=tokenizer),
    )


@torch.no_grad()
def compare_forwards(
    *,
    base_model: GMNERModel,
    protected_model: GMNERModel,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, float], str, str]:
    base_model.eval()
    protected_model.eval()
    errors = {key: 0.0 for key in FORMAL_TENSORS}
    base_digest = hashlib.sha256()
    protected_digest = hashlib.sha256()
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        base_outputs = base_model(batch)
        protected_outputs = protected_model(batch)
        for key in FORMAL_TENSORS:
            if key not in base_outputs and key not in protected_outputs:
                continue
            if key not in base_outputs or key not in protected_outputs:
                errors[key] = float("inf")
                continue
            difference = (
                base_outputs[key].float() - protected_outputs[key].float()
            ).abs().max()
            errors[key] = max(errors[key], float(difference.cpu()))
        for outputs, digest in (
            (base_outputs, base_digest),
            (protected_outputs, protected_digest),
        ):
            predictions = outputs["ner_logits"].argmax(dim=-1).detach().cpu().numpy()
            digest.update(predictions.tobytes())
            if "grounding_logits" in outputs:
                regions = outputs["grounding_logits"].argmax(dim=-1).detach().cpu().numpy()
                digest.update(regions.tobytes())
    return errors, base_digest.hexdigest(), protected_digest.hexdigest()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    base_config = load_config(args.base_config)
    protected_config = load_config(args.protected_config)
    if not bool(protected_config.model.use_protected_region_mner):
        raise ValueError("Protected config does not enable PA1.")
    if Path(base_config.data.dev_file).name != Path(protected_config.data.dev_file).name:
        raise ValueError("Base and protected configs do not use the same Dev data.")

    checkpoint_path = resolve(args.checkpoint, root)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state_dict"]
    base_model = GMNERModel(base_config, num_labels=9)
    base_model.load_state_dict(state_dict, strict=True)
    protected_model = GMNERModel(protected_config, num_labels=9)
    incompatible = protected_model.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith(PROTECTED_PREFIXES)
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={invalid_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    base_model.to(device)
    protected_model.to(device)
    loader = build_dev_loader(protected_config, root)
    errors, base_digest, protected_digest = compare_forwards(
        base_model=base_model,
        protected_model=protected_model,
        loader=loader,
        device=device,
    )
    base_metrics = evaluate_model(model=base_model, dataloader=loader, device=device)
    protected_metrics = evaluate_model(model=protected_model, dataloader=loader, device=device)
    compared_metrics = {}
    for key in sorted(set(base_metrics) & set(protected_metrics)):
        if "loss" in key or key.startswith("protected_"):
            continue
        compared_metrics[key] = {
            "base": float(base_metrics[key]),
            "protected": float(protected_metrics[key]),
            "exact": float(base_metrics[key]) == float(protected_metrics[key]),
        }

    tolerance = float(args.tolerance)
    tensor_gate = all(error <= tolerance for error in errors.values())
    metric_gate = all(item["exact"] for item in compared_metrics.values())
    report = {
        "kind": "protected_region_mner_epoch0_equivalence",
        "scope": "dev",
        "checkpoint": str(checkpoint_path),
        "missing_protected_parameter_count": len(incompatible.missing_keys),
        "unexpected_parameter_count": len(incompatible.unexpected_keys),
        "tolerance": tolerance,
        "max_abs_error": errors,
        "base_prediction_digest": base_digest,
        "protected_prediction_digest": protected_digest,
        "prediction_digest_exact": base_digest == protected_digest,
        "formal_metrics": compared_metrics,
        "gate_passed": tensor_gate and metric_gate and base_digest == protected_digest,
        "test_accessed": False,
    }
    output_path = resolve(args.output, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
