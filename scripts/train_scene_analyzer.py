#!/usr/bin/env python3
"""
训练 Scene Analyzer

功能: 训练场景分类器，区分 single-entity 和 multi-entity 记录
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# PyTorch imports are optional - only needed for neural network version
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    from gmner.models.scene_analyzer import SceneAnalyzer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger_temp = logging.getLogger(__name__)
    logger_temp.info("PyTorch not available - only baseline mode supported")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Dataset class - only needed for PyTorch version
if TORCH_AVAILABLE:
    class SceneDataset(Dataset):
        """Scene classification dataset"""

        def __init__(self, records: List[Dict], label_fn=None):
            """
            Args:
                records: 记录列表，每条包含 text, entities 等
                label_fn: 标签生成函数，默认使用 entity count
            """
            self.records = records
            self.label_fn = label_fn or self._default_label_fn

        def _default_label_fn(self, record: Dict) -> int:
            """默认标签: 0=single (<=1 entity), 1=multi (>1 entity)"""
            num_entities = len(record.get('entities', []))
            return 0 if num_entities <= 1 else 1

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            record = self.records[idx]
            label = self.label_fn(record)

            # 提取特征
            text_length = len(record['tokens'])
            num_spans = len(record['entities'])

            # 简化版本：使用统计特征
            return {
                'text_length': text_length,
                'num_spans': num_spans,
                'label': label,
                'record_idx': idx,
            }


def parse_conll_file(file_path: Path) -> List[Dict]:
    """解析 CoNLL 文件"""
    records = []
    current_record = {"tokens": [], "labels": [], "entities": []}
    current_entity = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_record["tokens"]:
                    records.append(current_record)
                    current_record = {"tokens": [], "labels": [], "entities": []}
                    current_entity = None
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token, label = parts[0], parts[1]
            current_record["tokens"].append(token)
            current_record["labels"].append(label)

            if label.startswith('B-'):
                if current_entity:
                    current_record["entities"].append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    "type": entity_type,
                    "start": len(current_record["tokens"]) - 1,
                    "end": len(current_record["tokens"]),
                }
            elif label.startswith('I-') and current_entity:
                current_entity["end"] = len(current_record["tokens"])
            elif label == 'O' and current_entity:
                current_record["entities"].append(current_entity)
                current_entity = None

        if current_entity:
            current_record["entities"].append(current_entity)
        if current_record["tokens"]:
            records.append(current_record)

    return records


def simple_feature_baseline(args):
    """
    简单基线: 使用统计特征训练 Logistic Regression
    用于快速验证场景分类的可行性
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    import numpy as np

    logger.info("=" * 80)
    logger.info("Simple Feature Baseline (Logistic Regression)")
    logger.info("=" * 80)

    # 加载数据
    logger.info(f"Loading train data: {args.train_file}")
    train_records = parse_conll_file(Path(args.train_file))
    logger.info(f"Loading dev data: {args.dev_file}")
    dev_records = parse_conll_file(Path(args.dev_file))

    logger.info(f"Train: {len(train_records)} records")
    logger.info(f"Dev: {len(dev_records)} records")

    # 提取特征
    def extract_features(records):
        features = []
        labels = []
        for record in records:
            text_len = len(record['tokens'])
            num_entities = len(record['entities'])

            # 统计特征
            entity_density = num_entities / max(text_len, 1)

            # 类型多样性
            types = set(e['type'] for e in record['entities'])
            type_diversity = len(types)

            # Span 长度统计
            span_lengths = [e['end'] - e['start'] for e in record['entities']]
            avg_span_len = np.mean(span_lengths) if span_lengths else 0

            features.append([
                text_len / 100.0,  # 归一化
                num_entities / 10.0,
                entity_density,
                type_diversity / 4.0,
                avg_span_len / 10.0,
                (num_entities ** 2) / 100.0,  # 二次项
            ])

            # 标签: 0=single, 1=multi
            label = 0 if num_entities <= 1 else 1
            labels.append(label)

        return np.array(features), np.array(labels)

    X_train, y_train = extract_features(train_records)
    X_dev, y_dev = extract_features(dev_records)

    logger.info(f"\nFeature shape: {X_train.shape}")
    logger.info(f"Train label distribution: {np.bincount(y_train)}")
    logger.info(f"Dev label distribution: {np.bincount(y_dev)}")

    # 训练模型
    logger.info("\nTraining Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # 评估
    train_pred = model.predict(X_train)
    dev_pred = model.predict(X_dev)

    train_acc = accuracy_score(y_train, train_pred)
    dev_acc = accuracy_score(y_dev, dev_pred)

    logger.info("\n" + "=" * 80)
    logger.info("Results")
    logger.info("=" * 80)
    logger.info(f"Train Accuracy: {train_acc:.4f}")
    logger.info(f"Dev Accuracy: {dev_acc:.4f}")

    logger.info("\nDev Classification Report:")
    logger.info(classification_report(y_dev, dev_pred,
                                     target_names=['Single', 'Multi'],
                                     digits=4))

    logger.info("\nDev Confusion Matrix:")
    logger.info("                Predicted")
    logger.info("                Single  Multi")
    cm = confusion_matrix(y_dev, dev_pred)
    logger.info(f"Actual Single  {cm[0,0]:6d}  {cm[0,1]:5d}")
    logger.info(f"       Multi   {cm[1,0]:6d}  {cm[1,1]:5d}")

    # 特征重要性
    logger.info("\nFeature Importance (coefficients):")
    feature_names = ['text_len', 'num_entities', 'entity_density',
                    'type_diversity', 'avg_span_len', 'num_entities^2']
    for name, coef in zip(feature_names, model.coef_[0]):
        logger.info(f"  {name:20s}: {coef:8.4f}")

    # 决策点验证
    logger.info("\n" + "=" * 80)
    logger.info("Decision Point 1 Verification")
    logger.info("=" * 80)

    if dev_acc >= 0.95:
        logger.info(f"✅ PASS: Dev accuracy {dev_acc:.4f} >= 0.95")
        logger.info("   Conclusion: Scene classification is feasible")
        logger.info("   Next: Implement full Scene Analyzer with neural network")
        result = "PASS"
    elif dev_acc >= 0.90:
        logger.info(f"⚠️  ACCEPTABLE: Dev accuracy {dev_acc:.4f} >= 0.90")
        logger.info("   Conclusion: Scene classification works but could be better")
        logger.info("   Next: Try neural network or use as-is")
        result = "ACCEPTABLE"
    else:
        logger.info(f"❌ FAIL: Dev accuracy {dev_acc:.4f} < 0.90")
        logger.info("   Conclusion: Scene classification may not be reliable")
        logger.info("   Fallback: Use entity count threshold or skip scene analyzer")
        result = "FAIL"

    # 保存结果
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "model": "LogisticRegression",
            "train_accuracy": float(train_acc),
            "dev_accuracy": float(dev_acc),
            "feature_names": feature_names,
            "feature_coefficients": model.coef_[0].tolist(),
            "intercept": float(model.intercept_[0]),
            "decision": result,
        }

        output_file = output_dir / "baseline_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n💾 Results saved to: {output_file}")

    logger.info("=" * 80)

    return dev_acc >= 0.90


def main():
    parser = argparse.ArgumentParser(description="Train Scene Analyzer")
    parser.add_argument("--train-file", type=str, required=True,
                       help="Training data file (CoNLL format)")
    parser.add_argument("--dev-file", type=str, required=True,
                       help="Dev data file (CoNLL format)")
    parser.add_argument("--output", type=str, default="outputs/scene_analyzer",
                       help="Output directory")
    parser.add_argument("--baseline-only", action="store_true",
                       help="Only run simple baseline (fast)")
    parser.add_argument(
        "--allow-gold-count-leakage-audit",
        action="store_true",
        help=(
            "Run the historical gold-count audit. This mode is not a "
            "deployable Scene Analyzer and must not be reported as a model."
        ),
    )

    args = parser.parse_args()

    if not args.allow_gold_count_leakage_audit:
        parser.error(
            "This historical script derives both features and labels from "
            "gold entity count and is therefore leakage-prone. Use "
            "scripts/generate_scene_predictions.py with formal deployed "
            "predictions. Pass --allow-gold-count-leakage-audit only to "
            "reproduce the invalid historical audit."
        )

    # Phase 1: 快速基线验证
    logger.info("Phase 1: Running simple feature baseline...")
    success = simple_feature_baseline(args)

    if args.baseline_only:
        logger.info("\nBaseline-only mode: Exiting after baseline")
        return

    if not success:
        logger.warning("\n⚠️  Baseline did not pass. Consider fallback strategies.")
        logger.warning("    You can still continue with neural network implementation.")

    # Phase 2: 神经网络实现 (TODO)
    logger.info("\nPhase 2: Neural network implementation - TODO")
    logger.info("  This will be implemented after baseline validation")


if __name__ == "__main__":
    main()
