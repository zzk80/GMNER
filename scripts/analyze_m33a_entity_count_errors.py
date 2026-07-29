#!/usr/bin/env python3
"""
M3.3A Entity Count Error Analysis

诊断脚本：分析单实体 vs 多实体场景的性能差异
严格遵循 P1-P4 要求
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gmner.config import load_evidence_visibility_config
from gmner.data import MMNERJsonDataset, load_word_aligned_tokenizer
from gmner.evaluation.evidence_visibility import EvidenceVisibilityEvaluator
from gmner.models.evidence_visibility import EvidenceVisibilityModel
from gmner.utils.io import maybe_convert_conll


def compute_sha256(file_path: Path) -> str:
    """计算文件 SHA-256"""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def load_gold_data(data_file: Path) -> List[Dict]:
    """加载 gold 数据"""
    records = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                records.append(record)
    return records


def classify_by_entity_count(records: List[Dict], key: str = 'gold') -> Dict:
    """
    按实体数量分类记录

    key: 'gold' 或 'pred'
    """
    single = []
    multi = []

    for record in records:
        if key == 'gold':
            entity_count = len(record.get('entities', []))
        else:  # pred
            entity_count = len(record.get('predicted_entities', []))

        if entity_count == 1:
            single.append(record)
        elif entity_count >= 2:
            multi.append(record)

    return {
        'single': single,
        'multi': multi,
        'all': records
    }


def compute_metrics(records: List[Dict]) -> Dict:
    """
    计算一组记录的指标

    返回: span/MNER/EEG/GMNER 的 P/R/F1
    """
    # Span metrics
    span_pred = span_gold = span_correct = 0

    # MNER metrics (span + type)
    mner_pred = mner_gold = mner_correct = 0

    # EEG metrics (entity + region grounding)
    eeg_pred = eeg_gold = eeg_correct = 0

    # GMNER metrics (span + type + region)
    gmner_pred = gmner_gold = gmner_correct = 0

    for record in records:
        gold_entities = record.get('entities', [])
        pred_entities = record.get('predicted_entities', [])

        # Span: (start, end)
        gold_spans = {(e['start'], e['end']) for e in gold_entities}
        pred_spans = {(e.get('start'), e.get('end')) for e in pred_entities if e.get('start') is not None}

        span_gold += len(gold_spans)
        span_pred += len(pred_spans)
        span_correct += len(gold_spans & pred_spans)

        # MNER: (start, end, type)
        gold_mner = {(e['start'], e['end'], e['type']) for e in gold_entities}
        pred_mner = {(e.get('start'), e.get('end'), e.get('type')) for e in pred_entities
                     if e.get('start') is not None and e.get('type') is not None}

        mner_gold += len(gold_mner)
        mner_pred += len(pred_mner)
        mner_correct += len(gold_mner & pred_mner)

        # EEG: entity + region (暂时简化，需要实际 region 信息)
        # TODO: 从预测中提取 region 信息

        # GMNER: (start, end, type, region)
        # TODO: 从预测中提取完整三元组

    def safe_f1(pred, gold, correct):
        p = correct / pred if pred > 0 else 0.0
        r = correct / gold if gold > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    span_p, span_r, span_f1 = safe_f1(span_pred, span_gold, span_correct)
    mner_p, mner_r, mner_f1 = safe_f1(mner_pred, mner_gold, mner_correct)
    eeg_p, eeg_r, eeg_f1 = safe_f1(eeg_pred, eeg_gold, eeg_correct)
    gmner_p, gmner_r, gmner_f1 = safe_f1(gmner_pred, gmner_gold, gmner_correct)

    return {
        'record_count': len(records),
        'span': {
            'predicted': span_pred,
            'gold': span_gold,
            'correct': span_correct,
            'precision': span_p,
            'recall': span_r,
            'f1': span_f1,
        },
        'mner': {
            'predicted': mner_pred,
            'gold': mner_gold,
            'correct': mner_correct,
            'precision': mner_p,
            'recall': mner_r,
            'f1': mner_f1,
        },
        'eeg': {
            'predicted': eeg_pred,
            'gold': eeg_gold,
            'correct': eeg_correct,
            'precision': eeg_p,
            'recall': eeg_r,
            'f1': eeg_f1,
        },
        'gmner': {
            'predicted': gmner_pred,
            'gold': gmner_gold,
            'correct': gmner_correct,
            'precision': gmner_p,
            'recall': gmner_r,
            'f1': gmner_f1,
        }
    }


def bootstrap_delta(single_records: List[Dict], multi_records: List[Dict],
                    metric_name: str, n_bootstrap: int = 1000, seed: int = 42) -> Dict:
    """
    Bootstrap 估计 single vs multi 的指标差异

    返回: delta, 95% CI, sample counts
    """
    rng = np.random.RandomState(seed)

    def compute_metric(records):
        metrics = compute_metrics(records)
        if metric_name == 'span_f1':
            return metrics['span']['f1']
        elif metric_name == 'mner_f1':
            return metrics['mner']['f1']
        elif metric_name == 'eeg_f1':
            return metrics['eeg']['f1']
        elif metric_name == 'gmner_f1':
            return metrics['gmner']['f1']
        else:
            return 0.0

    # 原始差值
    single_metric = compute_metric(single_records)
    multi_metric = compute_metric(multi_records)
    delta = single_metric - multi_metric

    # Bootstrap
    deltas = []
    for _ in range(n_bootstrap):
        single_sample = [single_records[i] for i in rng.choice(len(single_records), len(single_records), replace=True)]
        multi_sample = [multi_records[i] for i in rng.choice(len(multi_records), len(multi_records), replace=True)]

        single_boot = compute_metric(single_sample)
        multi_boot = compute_metric(multi_sample)
        deltas.append(single_boot - multi_boot)

    deltas = np.array(deltas)
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])

    return {
        'delta': delta,
        'ci_95_low': ci_low,
        'ci_95_high': ci_high,
        'single_count': len(single_records),
        'multi_count': len(multi_records),
        'bootstrap_iterations': n_bootstrap,
        'bootstrap_seed': seed,
    }


def main():
    parser = argparse.ArgumentParser(description='M3.3A Entity Count Error Analysis')
    parser.add_argument('--config', type=str, required=True, help='Evidence Visibility config')
    parser.add_argument('--checkpoint', type=str, required=True, help='Evidence Visibility checkpoint')
    parser.add_argument('--dev-file', type=str, required=True, help='Dev gold file')
    parser.add_argument('--r16-cache', type=str, required=True, help='R16 cache')
    parser.add_argument('--r36-cache', type=str, required=True, help='R36 cache')
    parser.add_argument('--output-dir', type=str, default='outputs/diagnostics/m33a_entity_count',
                       help='Output directory')
    parser.add_argument('--bootstrap-iterations', type=int, default=1000, help='Bootstrap iterations')
    parser.add_argument('--bootstrap-seed', type=int, default=42, help='Bootstrap random seed')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("M3.3A Entity Count Error Analysis")
    print("=" * 80)

    # Load gold data
    print("\n[1/6] Loading gold data...")
    gold_records = load_gold_data(Path(args.dev_file))
    print(f"  Loaded {len(gold_records)} records")

    # TODO: Load predictions from Evidence Visibility evaluation
    # 当前版本先使用 gold 数据进行结构验证

    # Classify by entity count (gold)
    print("\n[2/6] Classifying by gold entity count...")
    gold_slices = classify_by_entity_count(gold_records, key='gold')
    print(f"  Single-entity: {len(gold_slices['single'])} records")
    print(f"  Multi-entity: {len(gold_slices['multi'])} records")

    # Compute metrics per slice
    print("\n[3/6] Computing metrics per slice...")
    slice_metrics = {}
    for slice_name, records in gold_slices.items():
        print(f"  Computing {slice_name}...")
        slice_metrics[slice_name] = compute_metrics(records)

    # Bootstrap analysis
    print("\n[4/6] Running bootstrap analysis...")
    bootstrap_results = {}
    for metric in ['span_f1', 'mner_f1', 'eeg_f1', 'gmner_f1']:
        print(f"  Bootstrap {metric}...")
        bootstrap_results[metric] = bootstrap_delta(
            gold_slices['single'],
            gold_slices['multi'],
            metric,
            n_bootstrap=args.bootstrap_iterations,
            seed=args.bootstrap_seed
        )

    # Compute protocol
    print("\n[5/6] Computing protocol...")
    protocol = {
        'git_commit': 'TODO',  # Need to get from git
        'dev_source_sha256': compute_sha256(Path(args.dev_file)),
        'evidence_checkpoint_sha256': compute_sha256(Path(args.checkpoint)),
        'r16_cache_sha256': compute_sha256(Path(args.r16_cache)),
        'r36_cache_sha256': compute_sha256(Path(args.r36_cache)),
        'slice_definitions': {
            'gold_single': 'gold entity count = 1',
            'gold_multi': 'gold entity count >= 2',
        },
        'bootstrap_seed': args.bootstrap_seed,
        'bootstrap_iterations': args.bootstrap_iterations,
        'test_accessed': False,
    }

    # Save results
    print("\n[6/6] Saving results...")

    # summary.json
    summary = {
        'slice_metrics': slice_metrics,
        'bootstrap_results': bootstrap_results,
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary.json")

    # protocol.json
    with open(output_dir / 'protocol.json', 'w') as f:
        json.dump(protocol, f, indent=2)
    print(f"  Saved protocol.json")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\nSlice Metrics:")
    for slice_name in ['single', 'multi', 'all']:
        metrics = slice_metrics[slice_name]
        print(f"\n{slice_name.upper()}:")
        print(f"  Records: {metrics['record_count']}")
        print(f"  Span F1: {metrics['span']['f1']:.4f}")
        print(f"  MNER F1: {metrics['mner']['f1']:.4f}")
        print(f"  EEG F1: {metrics['eeg']['f1']:.4f}")
        print(f"  GMNER F1: {metrics['gmner']['f1']:.4f}")

    print("\nBootstrap Delta (Single - Multi):")
    for metric, result in bootstrap_results.items():
        print(f"\n{metric}:")
        print(f"  Delta: {result['delta']:.4f}")
        print(f"  95% CI: [{result['ci_95_low']:.4f}, {result['ci_95_high']:.4f}]")
        print(f"  Crosses zero: {'Yes' if result['ci_95_low'] < 0 < result['ci_95_high'] else 'No'}")

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
