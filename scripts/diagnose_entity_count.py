#!/usr/bin/env python3
"""
M3.3A Entity Count Diagnostic Analysis

直接复用 Evidence Visibility 评估框架
在逐记录循环中抽取 single/multi 切片信息
"""

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.data import (
    PairedRecordCandidateDataset,
    PairedRecordCandidateCollator,
    RecordCandidateDataset,
)
from gmner.engine.evidence_visibility_evaluator import (
    _selected_span_indices,
    frozen_hierarchical_context,
    move_paired_record_batch,
)
from gmner.engine.utils import match_record_predictions
from gmner.evidence_visibility_config import load_evidence_visibility_config
from gmner.models.evidence_visibility import (
    RegionEvidenceVisibilityHead,
    decode_evidence_visibility,
)


def resolve(path_str: str, root: Path) -> Path:
    """Resolve path relative to project root"""
    path = Path(path_str)
    if path.is_absolute():
        return path
    return root / path


def load_frozen_chain(config, root, device):
    """Load frozen chain and evidence model - copied from train_evidence_visibility.py"""
    from gmner.hierarchical_record_verifier_config import (
        load_hierarchical_record_verifier_config,
    )
    from gmner.models.hierarchical_record_verifier import HierarchicalRecordVerifier
    from gmner.fine_grounding_adapter_config import load_fine_grounding_adapter_config
    from gmner.models.fine_grounding_adapter import FineGroundingAdapter

    # Load fine config first
    fine_config_path = resolve(config.frozen.fine_config, root)
    fine_config = load_fine_grounding_adapter_config(fine_config_path)

    # Load hierarchical verifier (from fine config's frozen)
    hierarchy_config = load_hierarchical_record_verifier_config(
        resolve(fine_config.frozen.hierarchical_config, root)
    )
    hierarchy = HierarchicalRecordVerifier(hierarchy_config.model)
    hierarchy_checkpoint = torch.load(
        resolve(fine_config.frozen.hierarchical_checkpoint, root),
        map_location='cpu'
    )
    hierarchy.load_state_dict(hierarchy_checkpoint['model_state_dict'])

    # Load fine grounding adapter
    fine_model = FineGroundingAdapter(fine_config.model)
    fine_checkpoint = torch.load(
        resolve(config.frozen.fine_checkpoint, root),
        map_location='cpu'
    )
    fine_model.load_state_dict(fine_checkpoint['model_state_dict'])

    # Freeze models
    hierarchy.to(device).eval()
    fine_model.to(device).eval()
    for frozen_model in (fine_model, hierarchy):
        for parameter in frozen_model.parameters():
            parameter.requires_grad = False

    # Create evidence visibility model
    model = RegionEvidenceVisibilityHead(config.model).to(device)

    return model, fine_model, hierarchy


def compute_sha256(file_path: Path) -> str:
    """计算文件 SHA-256"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_git_commit() -> str:
    """获取当前 git commit"""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except:
        return "unknown"


@torch.no_grad()
def evaluate_with_entity_count_diagnostics(
    model: torch.nn.Module,
    fine_model: torch.nn.Module,
    hierarchical_model: torch.nn.Module,
    dataloader,
    device: torch.device,
    *,
    decode_options: dict,
    output_jsonl: Path,
):
    """
    复制 evaluate_evidence_visibility 的逻辑
    在逐记录循环中抽取 gold entity count 和预测信息
    """
    model.eval()
    fine_model.eval()
    hierarchical_model.eval()

    entity_threshold = decode_options.get("entity_threshold", 0.0)
    decode_strategy = decode_options.get("strategy", "interval")
    stage1_spans_only = decode_options.get("stage1_spans_only", True)
    visible_from_null_threshold = decode_options.get(
        "visible_from_null_threshold", 0.8
    )
    null_from_visible_threshold = decode_options.get(
        "null_from_visible_threshold", 0.2
    )

    # 整体计数器
    overall_counts = Counter()
    overall_correct = {"baseline": Counter(), "final": Counter()}

    # 切片计数器
    single_counts = Counter()
    single_correct = {"baseline": Counter(), "final": Counter()}
    multi_counts = Counter()
    multi_correct = {"baseline": Counter(), "final": Counter()}

    # 逐记录输出
    records_output = []

    for batch in dataloader:
        formal, expanded = move_paired_record_batch(batch, device)

        # Frozen chain inference
        hierarchy_outputs = frozen_hierarchical_context(
            hierarchical_model, fine_model, formal, expanded
        )

        # Evidence visibility inference
        outputs = model(
            formal_span_features=formal["span_features"],
            expanded_span_features=expanded["span_features"],
            expanded_region_features=expanded["region_features"],
            expanded_span_mask=expanded["span_mask"],
            expanded_region_mask=expanded["region_mask"],
            base_region_scores=expanded["base_region_scores"],
            fine_top1_region_index=hierarchy_outputs["fine_top1_region_index"],
            fine_top1_region_score=hierarchy_outputs["fine_top1_region_score"],
            base_top1_region_score=hierarchy_outputs["base_top1_region_score"],
            base_fine_agreement=hierarchy_outputs["base_fine_agreement"],
            all_rankers_agree=hierarchy_outputs["all_rankers_agree"],
            fine_probability_margin=hierarchy_outputs["fine_probability_margin"],
        )

        # Decode visibility
        baseline_visible, final_visible = decode_evidence_visibility(
            base_visibility_logits=outputs["base_visibility_logits"],
            visibility_correction_logits=outputs[
                "unbounded_visibility_correction_logits"
            ],
            visible_from_null_threshold=visible_from_null_threshold,
            null_from_visible_threshold=null_from_visible_threshold,
        )

        # Get region indices
        fine_indices = hierarchy_outputs["fine_top1_region_index"].long()
        expanded_null = torch.tensor(
            [
                int(metadata.get("null_region_index", -1))
                for metadata in expanded["metadata"]
            ],
            device=device,
            dtype=torch.long,
        )[:, None].expand_as(fine_indices)

        baseline_indices = torch.where(
            baseline_visible, fine_indices, expanded_null
        )
        final_indices = torch.where(final_visible, fine_indices, expanded_null)

        # 逐记录循环
        for row, metadata in enumerate(expanded["metadata"]):
            record_id = metadata.get("record_id", "")
            text = metadata.get("text", "")

            spans, selected = _selected_span_indices(
                hierarchy_outputs,
                formal,
                row,
                entity_threshold=entity_threshold,
                decode_strategy=decode_strategy,
                stage1_spans_only=stage1_spans_only,
            )

            predictions = {"baseline": [], "final": []}
            for span_index in selected:
                shared = {
                    "span": list(spans[span_index]),
                    "type_id": int(
                        hierarchy_outputs["fixed_type_ids"][
                            row, span_index
                        ].item()
                    ),
                    "candidate_index": span_index,
                }
                predictions["baseline"].append(
                    {
                        **shared,
                        "region_index": int(
                            baseline_indices[row, span_index].item()
                        ),
                    }
                )
                predictions["final"].append(
                    {
                        **shared,
                        "region_index": int(
                            final_indices[row, span_index].item()
                        ),
                    }
                )

            # Gold entities
            gold = list(metadata.get("gold_entities") or [])
            gold_entity_count = len(gold)

            # Predicted entity count
            pred_entity_count = len(predictions["final"])

            # Match predictions
            matches = {
                branch: match_record_predictions(values, gold)
                for branch, values in predictions.items()
            }

            # 记录详情
            record_info = {
                "record_id": record_id,
                "text": text,
                "gold_entity_count": gold_entity_count,
                "pred_entity_count": pred_entity_count,
                "gold_entities": [
                    {
                        "span": g["span"],
                        "type_id": g["type_id"],
                        "visible": g.get("visible", False),
                    }
                    for g in gold
                ],
                "predicted_entities": predictions["final"],
                "matches": {
                    "span": len(matches["final"]["span"]),
                    "mner": len(matches["final"]["mner"]),
                    "eeg": len(matches["final"]["eeg"]),
                    "gmner": len(matches["final"]["gmner"]),
                },
            }
            records_output.append(record_info)

            # 分类到切片
            is_single = gold_entity_count == 1
            is_multi = gold_entity_count >= 2

            # 更新整体计数
            overall_counts["predicted"] += pred_entity_count
            overall_counts["gold"] += gold_entity_count
            for branch in predictions:
                for metric in ("span", "mner", "eeg", "gmner"):
                    overall_correct[branch][metric] += len(matches[branch][metric])

            # 更新切片计数
            if is_single:
                single_counts["predicted"] += pred_entity_count
                single_counts["gold"] += gold_entity_count
                single_counts["record_count"] += 1
                for branch in predictions:
                    for metric in ("span", "mner", "eeg", "gmner"):
                        single_correct[branch][metric] += len(
                            matches[branch][metric]
                        )
            elif is_multi:
                multi_counts["predicted"] += pred_entity_count
                multi_counts["gold"] += gold_entity_count
                multi_counts["record_count"] += 1
                for branch in predictions:
                    for metric in ("span", "mner", "eeg", "gmner"):
                        multi_correct[branch][metric] += len(
                            matches[branch][metric]
                        )

    # 保存逐记录输出
    with open(output_jsonl, 'w') as f:
        for record in records_output:
            f.write(json.dumps(record) + '\n')

    # 计算 F1
    def compute_f1(correct_count, pred_count, gold_count):
        p = correct_count / pred_count if pred_count > 0 else 0.0
        r = correct_count / gold_count if gold_count > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def slice_metrics(counts, correct):
        pred = counts["predicted"]
        gold = counts["gold"]
        metrics = {}
        for metric in ("span", "mner", "eeg", "gmner"):
            p, r, f1 = compute_f1(correct["final"][metric], pred, gold)
            metrics[f"{metric}_precision"] = p
            metrics[f"{metric}_recall"] = r
            metrics[f"{metric}_f1"] = f1
        metrics["record_count"] = counts.get("record_count", 0)
        metrics["predicted"] = pred
        metrics["gold"] = gold
        return metrics

    overall_metrics = slice_metrics(overall_counts, overall_correct)
    single_metrics = slice_metrics(single_counts, single_correct)
    multi_metrics = slice_metrics(multi_counts, multi_correct)

    return {
        "overall": overall_metrics,
        "gold_single": single_metrics,
        "gold_multi": multi_metrics,
        "records_output_path": str(output_jsonl),
        "total_records": len(records_output),
    }


def main():
    parser = argparse.ArgumentParser(
        description='M3.3A Entity Count Diagnostic Analysis'
    )
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--formal-cache', type=str, required=True)
    parser.add_argument('--expanded-cache', type=str, required=True)
    parser.add_argument(
        '--output-dir',
        type=str,
        default='outputs/diagnostics/m33a_entity_count',
    )
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]

    print("=" * 80)
    print("M3.3A Entity Count Diagnostic Analysis")
    print("=" * 80)

    # Load config
    print("\n[1/5] Loading config...")
    config = load_evidence_visibility_config(args.config)
    device = torch.device(args.device)

    # Load frozen chain
    print("\n[2/5] Loading frozen chain...")
    model, fine_model, hierarchical_model = load_frozen_chain(config, root, device)

    # Load evidence visibility checkpoint
    print("\n[3/5] Loading evidence visibility checkpoint...")
    checkpoint = torch.load(resolve(args.checkpoint, root), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load dataset
    print("\n[4/5] Loading dataset...")
    root = Path(__file__).resolve().parents[1]
    formal = RecordCandidateDataset(resolve(args.formal_cache, root))
    expanded = RecordCandidateDataset(resolve(args.expanded_cache, root))
    dataset = PairedRecordCandidateDataset(formal, expanded)
    collator = PairedRecordCandidateCollator()
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        collate_fn=collator,
        shuffle=False,
    )

    # Evaluate with diagnostics
    print("\n[5/5] Running evaluation with entity count diagnostics...")
    results = evaluate_with_entity_count_diagnostics(
        model=model,
        fine_model=fine_model,
        hierarchical_model=hierarchical_model,
        dataloader=dataloader,
        device=device,
        decode_options=dict(config.decode),
        output_jsonl=output_dir / 'records.jsonl',
    )

    # Save results
    print("\nSaving results...")
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Protocol
    protocol = {
        'git_commit': get_git_commit(),
        'evidence_checkpoint_sha256': compute_sha256(Path(args.checkpoint)),
        'formal_cache_sha256': compute_sha256(Path(args.formal_cache)),
        'expanded_cache_sha256': compute_sha256(Path(args.expanded_cache)),
        'test_accessed': False,
    }
    with open(output_dir / 'protocol.json', 'w') as f:
        json.dump(protocol, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    print("\nOverall Metrics:")
    print(f"  Records: {results['total_records']}")
    print(f"  GMNER F1: {results['overall']['gmner_f1']:.6f}")
    print(f"  MNER F1: {results['overall']['mner_f1']:.6f}")
    print(f"  EEG F1: {results['overall']['eeg_f1']:.6f}")

    print("\nGold Single-Entity Slice:")
    print(f"  Records: {results['gold_single']['record_count']}")
    print(f"  GMNER F1: {results['gold_single']['gmner_f1']:.6f}")
    print(f"  MNER F1: {results['gold_single']['mner_f1']:.6f}")
    print(f"  EEG F1: {results['gold_single']['eeg_f1']:.6f}")

    print("\nGold Multi-Entity Slice:")
    print(f"  Records: {results['gold_multi']['record_count']}")
    print(f"  GMNER F1: {results['gold_multi']['gmner_f1']:.6f}")
    print(f"  MNER F1: {results['gold_multi']['mner_f1']:.6f}")
    print(f"  EEG F1: {results['gold_multi']['eeg_f1']:.6f}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
