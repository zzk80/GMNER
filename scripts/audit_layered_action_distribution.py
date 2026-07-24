"""Audit train/dev state drift before fitting a layered action policy."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.engine.fine_grounding_adapter_evaluator import (
    move_paired_record_batch,
)
from gmner.engine.layered_action_verifier_evaluator import (
    frozen_layered_action_features,
)
from gmner.layered_action_verifier_config import (
    load_layered_action_verifier_config,
)
from gmner.losses.layered_action_verifier_loss import (
    layered_action_supervision,
)
from gmner.models.layered_action_verifier import (
    ACTION_MODE_FULL,
    LayeredActionVerifier,
)
from gmner.siglip2_region_reliability_config import (
    load_siglip2_region_reliability_config,
)
from scripts.train_fine_grounding_adapter import (
    decode_options,
    resolve,
    validate_fingerprints,
)
from scripts.train_siglip2_region_reliability import (
    _base_paired,
    _paired_dataset,
    load_frozen_reliability_chain,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-train-records", type=int, default=None)
    parser.add_argument("--max-dev-records", type=int, default=None)
    parser.add_argument("--require-oof-train-cache", action="store_true")
    return parser.parse_args()


def _cache_metadata(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid candidate cache: {path}")
    metadata = dict(payload.get("metadata") or {})
    oof = dict(metadata.get("oof") or {})
    return {
        "path": str(path.resolve()),
        "split": metadata.get("split"),
        "oof_enabled": bool(oof.get("enabled") or metadata.get("oof_heldout")),
        "oof_num_folds": oof.get("num_folds"),
        "stage1_checkpoint_sha256": metadata.get("stage1_checkpoint_sha256"),
        "candidate_config_sha256": metadata.get("candidate_config_sha256"),
    }


def _append(
    target: list[float],
    values: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    if mask.any():
        target.extend(values.float()[mask].detach().cpu().tolist())


def _describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0.0,
            "mean": 0.0,
            "std": 0.0,
            "p10": 0.0,
            "p50": 0.0,
            "p90": 0.0,
        }
    tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "count": float(tensor.numel()),
        "mean": float(tensor.mean().item()),
        "std": float(tensor.std(unbiased=False).item()),
        "p10": float(torch.quantile(tensor, 0.10).item()),
        "p50": float(torch.quantile(tensor, 0.50).item()),
        "p90": float(torch.quantile(tensor, 0.90).item()),
    }


@torch.inference_mode()
def _audit_split(
    split: str,
    loader: DataLoader,
    model: LayeredActionVerifier,
    reliability_model: torch.nn.Module,
    evidence_model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchy: torch.nn.Module,
    device: torch.device,
    *,
    decode: dict,
    stage1_spans_only: bool,
    require_correct_type: bool,
) -> dict:
    counts = Counter()
    values: dict[str, list[float]] = defaultdict(list)
    for raw_batch in tqdm(loader, desc=f"Auditing {split}"):
        paired = move_paired_record_batch(raw_batch, device)
        context = frozen_layered_action_features(
            reliability_model,
            evidence_model,
            fine_model,
            hierarchy,
            paired,
            decode_options=decode,
        )
        expanded = context["expanded"]
        fine_outputs = context["fine_outputs"]
        hierarchy_outputs = context["hierarchy_outputs"]
        evidence_outputs = context["evidence_outputs"]
        assert isinstance(expanded, dict)
        assert isinstance(fine_outputs, dict)
        assert isinstance(hierarchy_outputs, dict)
        assert isinstance(evidence_outputs, dict)
        outputs = model(
            fine_outputs,
            hierarchy_outputs,
            evidence_outputs,
            expanded,
            current_visible_mask=context["current_visible"],
            base_is_null_mask=context["base_is_null"],
            reliability_outputs=context["reliability_outputs"],
        )
        supervision = layered_action_supervision(
            outputs,
            fine_outputs,
            hierarchy_outputs,
            expanded,
            stage1_spans_only=stage1_spans_only,
            require_correct_type=require_correct_type,
        )
        deployable = supervision["deployable_mask"]
        supervised = supervision["supervised_mask"]
        current_visible = outputs["current_visible_mask"].bool()
        comparable = (
            deployable
            & expanded["gold_span_mask"].bool()
            & expanded["visibility_targets"].float().ge(0.0)
            & supervision["type_correct_mask"]
        )
        target_visible = supervision["target_visible_mask"]
        visible_comparable = comparable & target_visible

        counts["records"] += len(expanded["metadata"])
        counts["deployable"] += int(deployable.sum().item())
        counts["current_visible"] += int((deployable & current_visible).sum().item())
        counts["current_null"] += int((deployable & ~current_visible).sum().item())
        counts["supervised"] += int(supervised.sum().item())
        for name in ("keep", "to_null", "to_visible"):
            counts[name] += int(supervision[f"{name}_mask"].sum().item())
        counts["comparable"] += int(comparable.sum().item())
        counts["formal_correct"] += int(
            (comparable & supervision["current_correct_mask"]).sum().item()
        )
        counts["visible_comparable"] += int(visible_comparable.sum().item())
        gold_in_top4 = (
            outputs["fine_top4_mask"].bool()
            & expanded["gold_region_positive_mask"].bool()
        ).any(dim=-1)
        counts["gold_in_top4"] += int(
            (visible_comparable & gold_in_top4).sum().item()
        )

        candidate_mask = fine_outputs["candidate_mask"].bool()
        candidate_count = candidate_mask.sum(dim=-1)
        fine_logits = fine_outputs["final_region_logits"].float().masked_fill(
            ~candidate_mask, -1e4
        )
        fine_probabilities = torch.softmax(fine_logits, dim=-1) * candidate_mask.float()
        fine_probabilities = fine_probabilities / fine_probabilities.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        entropy = -(
            fine_probabilities
            * fine_probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1)
        top_count = min(2, fine_logits.size(-1))
        top_values = fine_logits.topk(top_count, dim=-1).values
        margin = top_values[..., 0] - top_values[..., min(1, top_count - 1)]
        fine_top1 = fine_logits.argmax(dim=-1)
        base_top1 = fine_outputs["base_log_prior"].float().masked_fill(
            ~candidate_mask, -1e4
        ).argmax(dim=-1)
        coarse_top1 = fine_outputs["coarse_log_prior"].float().masked_fill(
            ~candidate_mask, -1e4
        ).argmax(dim=-1)
        has_candidate = deployable & candidate_count.gt(0)
        counts["ranking_denominator"] += int(has_candidate.sum().item())
        counts["base_coarse_agree"] += int(
            (has_candidate & base_top1.eq(coarse_top1)).sum().item()
        )
        counts["base_fine_agree"] += int(
            (has_candidate & base_top1.eq(fine_top1)).sum().item()
        )
        counts["coarse_fine_agree"] += int(
            (has_candidate & coarse_top1.eq(fine_top1)).sum().item()
        )
        counts["all_rankers_agree"] += int(
            (
                has_candidate
                & base_top1.eq(coarse_top1)
                & base_top1.eq(fine_top1)
            ).sum().item()
        )
        promoted_top1 = fine_outputs["promoted_candidate_mask"].bool().gather(
            -1, fine_top1.unsqueeze(-1)
        ).squeeze(-1)
        counts["promoted_top1"] += int((has_candidate & promoted_top1).sum().item())

        _append(
            values["visibility_logit"],
            evidence_outputs["final_visibility_logits"],
            deployable,
        )
        _append(
            values["fine_top1_top2_margin"],
            margin,
            deployable & candidate_count.ge(2),
        )
        _append(values["region_entropy"], entropy, has_candidate)

    deployable_count = max(counts["deployable"], 1)
    supervised_count = max(counts["supervised"], 1)
    comparable_count = max(counts["comparable"], 1)
    visible_count = max(counts["visible_comparable"], 1)
    ranking_count = max(counts["ranking_denominator"], 1)
    return {
        "records": float(counts["records"]),
        "counts": {key: float(value) for key, value in sorted(counts.items())},
        "rates": {
            "current_null_ratio": counts["current_null"] / deployable_count,
            "current_visible_ratio": counts["current_visible"] / deployable_count,
            "keep_label_ratio": counts["keep"] / supervised_count,
            "to_null_label_ratio": counts["to_null"] / supervised_count,
            "to_visible_label_ratio": counts["to_visible"] / supervised_count,
            "formal_prediction_accuracy": counts["formal_correct"]
            / comparable_count,
            "gold_in_top4_rate": counts["gold_in_top4"] / visible_count,
            "base_coarse_agreement": counts["base_coarse_agree"] / ranking_count,
            "base_fine_agreement": counts["base_fine_agree"] / ranking_count,
            "coarse_fine_agreement": counts["coarse_fine_agree"] / ranking_count,
            "all_rankers_agreement": counts["all_rankers_agree"] / ranking_count,
            "promoted_top1_ratio": counts["promoted_top1"] / ranking_count,
        },
        "continuous": {
            key: _describe(series) for key, series in sorted(values.items())
        },
    }


def _gaps(train: dict, dev: dict) -> dict:
    return {
        "rates_dev_minus_train": {
            key: float(dev["rates"][key]) - float(train["rates"][key])
            for key in train["rates"]
        },
        "continuous_mean_dev_minus_train": {
            key: float(dev["continuous"][key]["mean"])
            - float(train["continuous"][key]["mean"])
            for key in train["continuous"]
        },
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    config = load_layered_action_verifier_config(args.config)
    reliability_config = load_siglip2_region_reliability_config(
        resolve(config.frozen.reliability_config, root)
    )
    datasets = {}
    collators = {}
    for split in ("train", "dev"):
        datasets[split], collators[split] = _paired_dataset(
            reliability_config, root, split
        )
    device_name = args.device or config.runtime.device
    device = torch.device(
        device_name
        if str(device_name).startswith("cuda") and torch.cuda.is_available()
        else "cpu"
    )
    (
        reliability_model,
        evidence_model,
        fine_model,
        hierarchy,
        hierarchy_config,
        hierarchy_checkpoint,
        coarse_checkpoint,
        _,
        _,
    ) = load_frozen_reliability_chain(reliability_config, root, device)
    reliability_checkpoint = torch.load(
        resolve(config.frozen.reliability_checkpoint, root), map_location="cpu"
    )
    reliability_model.load_state_dict(reliability_checkpoint["model_state_dict"])
    reliability_model.to(device).eval()
    for frozen in (reliability_model, evidence_model, fine_model, hierarchy):
        frozen.eval()

    train_cache_path = resolve(reliability_config.data.formal_train_cache, root)
    dev_cache_path = resolve(reliability_config.data.formal_dev_cache, root)
    cache = {
        "train": _cache_metadata(train_cache_path),
        "dev": _cache_metadata(dev_cache_path),
    }
    if args.require_oof_train_cache and not cache["train"]["oof_enabled"]:
        raise RuntimeError(
            "The configured train cache is not a merged held-out OOF cache: "
            f"{train_cache_path}"
        )
    for split in ("train", "dev"):
        validate_fingerprints(
            _base_paired(datasets[split]),
            hierarchy_checkpoint=hierarchy_checkpoint,
            coarse_checkpoint=coarse_checkpoint,
            require_oof=args.require_oof_train_cache and split == "train",
        )

    limits = {
        "train": args.max_train_records,
        "dev": args.max_dev_records,
    }
    for split, limit in limits.items():
        if limit is not None:
            datasets[split] = Subset(
                datasets[split], range(min(max(1, limit), len(datasets[split])))
            )
    batch_size = args.batch_size or config.optim.batch_size
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=reliability_config.data.num_workers,
            collate_fn=collators[split],
        )
        for split, dataset in datasets.items()
    }
    audit_model_config = replace(config.model, action_mode=ACTION_MODE_FULL)
    model = LayeredActionVerifier(audit_model_config).to(device).eval()
    decode = decode_options(hierarchy_config)
    results = {
        split: _audit_split(
            split,
            loaders[split],
            model,
            reliability_model,
            evidence_model,
            fine_model,
            hierarchy,
            device,
            decode=decode,
            stage1_spans_only=config.loss.stage1_spans_only,
            require_correct_type=config.loss.require_correct_type,
        )
        for split in ("train", "dev")
    }
    report = {
        "config": str(resolve(args.config, root).resolve()),
        "test_read": False,
        "require_oof_train_cache": bool(args.require_oof_train_cache),
        "cache": cache,
        **results,
        "gaps": _gaps(results["train"], results["dev"]),
    }
    output = resolve(args.output, root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(output.resolve()),
                "test_read": False,
                "train_oof_enabled": cache["train"]["oof_enabled"],
                "train_rates": results["train"]["rates"],
                "dev_rates": results["dev"]["rates"],
                "gaps": report["gaps"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
