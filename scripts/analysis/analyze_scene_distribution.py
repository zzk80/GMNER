#!/usr/bin/env python3
"""
分析 GMNER 数据集的场景分布
- 单实体 vs 多实体记录
- 实体数量分布
- 类型分布
- 区域重叠情况

用于验证 Instance-Aware 方法的可行性
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import sys


def parse_conll_file(file_path: Path) -> List[Dict]:
    """解析 CoNLL 格式的数据文件"""
    records = []
    current_record = {"tokens": [], "labels": [], "entities": []}
    current_entity = None

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()

            if not line:  # 记录结束
                if current_record["tokens"]:
                    records.append(current_record)
                    current_record = {"tokens": [], "labels": [], "entities": []}
                    current_entity = None
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            token = parts[0]
            label = parts[1]

            current_record["tokens"].append(token)
            current_record["labels"].append(label)

            # 提取实体
            if label.startswith('B-'):
                if current_entity:
                    current_record["entities"].append(current_entity)
                entity_type = label[2:]
                current_entity = {
                    "type": entity_type,
                    "start": len(current_record["tokens"]) - 1,
                    "end": len(current_record["tokens"]),
                    "tokens": [token]
                }
            elif label.startswith('I-') and current_entity:
                current_entity["end"] = len(current_record["tokens"])
                current_entity["tokens"].append(token)
            elif label == 'O':
                if current_entity:
                    current_record["entities"].append(current_entity)
                    current_entity = None

        # 处理最后一条记录
        if current_entity:
            current_record["entities"].append(current_entity)
        if current_record["tokens"]:
            records.append(current_record)

    return records


def analyze_scene_distribution(records: List[Dict]) -> Dict:
    """分析场景分布"""
    stats = {
        "total_records": len(records),
        "entity_count_distribution": Counter(),
        "single_entity_records": 0,
        "multi_entity_records": 0,
        "multi_entity_2_3": 0,
        "multi_entity_4plus": 0,
        "type_distribution": Counter(),
        "type_co_occurrence": defaultdict(Counter),
        "records_by_entity_count": defaultdict(list),
    }

    for idx, record in enumerate(records):
        num_entities = len(record["entities"])
        stats["entity_count_distribution"][num_entities] += 1
        stats["records_by_entity_count"][num_entities].append(idx)

        if num_entities <= 1:
            stats["single_entity_records"] += 1
        else:
            stats["multi_entity_records"] += 1
            if num_entities <= 3:
                stats["multi_entity_2_3"] += 1
            else:
                stats["multi_entity_4plus"] += 1

        # 类型分布
        entity_types = [e["type"] for e in record["entities"]]
        for et in entity_types:
            stats["type_distribution"][et] += 1

        # 类型共现
        if len(entity_types) > 1:
            unique_types = set(entity_types)
            for t1 in unique_types:
                for t2 in unique_types:
                    if t1 != t2:
                        stats["type_co_occurrence"][t1][t2] += 1

    return stats


def compute_token_overlap(records: List[Dict]) -> Dict:
    """计算实体在文本中的重叠情况"""
    overlap_stats = {
        "records_with_overlap": 0,
        "total_overlap_pairs": 0,
        "overlap_by_entity_count": defaultdict(int),
    }

    for record in records:
        entities = record["entities"]
        if len(entities) < 2:
            continue

        has_overlap = False
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # 检查 token span 是否重叠
                if not (e1["end"] <= e2["start"] or e2["end"] <= e1["start"]):
                    has_overlap = True
                    overlap_stats["total_overlap_pairs"] += 1

        if has_overlap:
            overlap_stats["records_with_overlap"] += 1
            overlap_stats["overlap_by_entity_count"][len(entities)] += 1

    return overlap_stats


def print_statistics(stats: Dict, overlap_stats: Dict):
    """打印统计结果"""
    print("=" * 80)
    print("GMNER Scene Distribution Analysis")
    print("=" * 80)

    print(f"\n总体统计:")
    print(f"  总记录数: {stats['total_records']}")
    print(f"  单实体记录: {stats['single_entity_records']} ({stats['single_entity_records']/stats['total_records']*100:.1f}%)")
    print(f"  多实体记录: {stats['multi_entity_records']} ({stats['multi_entity_records']/stats['total_records']*100:.1f}%)")
    print(f"    - 2-3 个实体: {stats['multi_entity_2_3']} ({stats['multi_entity_2_3']/stats['total_records']*100:.1f}%)")
    print(f"    - 4+ 个实体: {stats['multi_entity_4plus']} ({stats['multi_entity_4plus']/stats['total_records']*100:.1f}%)")

    print(f"\n实体数量分布:")
    for count in sorted(stats['entity_count_distribution'].keys()):
        num_records = stats['entity_count_distribution'][count]
        pct = num_records / stats['total_records'] * 100
        bar = "=" * int(pct / 2)
        print(f"  {count:2d} 个实体: {num_records:4d} ({pct:5.1f}%) {bar}")

    print(f"\n类型分布:")
    for entity_type, count in stats['type_distribution'].most_common(10):
        print(f"  {entity_type:15s}: {count:4d}")

    print(f"\n类型共现 (Top 10 pairs):")
    all_pairs = []
    for t1, t2_counts in stats['type_co_occurrence'].items():
        for t2, count in t2_counts.items():
            all_pairs.append((t1, t2, count))
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    for t1, t2, count in all_pairs[:10]:
        print(f"  {t1} + {t2}: {count}")

    print(f"\nToken 重叠情况:")
    print(f"  存在重叠的记录: {overlap_stats['records_with_overlap']}")
    print(f"  总重叠对数: {overlap_stats['total_overlap_pairs']}")
    print(f"  按实体数分布:")
    for count in sorted(overlap_stats['overlap_by_entity_count'].keys()):
        print(f"    {count} 个实体: {overlap_stats['overlap_by_entity_count'][count]} records")

    print("\n" + "=" * 80)
    print("结论:")
    multi_ratio = stats['multi_entity_records'] / stats['total_records']
    if multi_ratio >= 0.3:
        print(f"  多实体场景占比 {multi_ratio*100:.1f}%，足够支持 Instance-Aware 建模")
    else:
        print(f"  WARNING: 多实体场景占比仅 {multi_ratio*100:.1f}%，可能样本量不足")

    if stats['multi_entity_2_3'] > 100:
        print(f"  2-3 实体场景有 {stats['multi_entity_2_3']} 条，适合初始验证")

    if stats['multi_entity_4plus'] > 50:
        print(f"  4+ 实体场景有 {stats['multi_entity_4plus']} 条，可测试复杂情况")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="分析 GMNER 场景分布")
    parser.add_argument("--data-file", type=str, required=True,
                       help="数据文件路径 (CoNLL 格式)")
    parser.add_argument("--output", type=str, default=None,
                       help="输出 JSON 文件路径")

    args = parser.parse_args()

    # 解析数据
    print(f"读取数据: {args.data_file}")
    records = parse_conll_file(Path(args.data_file))
    print(f"读取完成: {len(records)} 条记录\n")

    # 分析分布
    stats = analyze_scene_distribution(records)
    overlap_stats = compute_token_overlap(records)

    # 打印结果
    print_statistics(stats, overlap_stats)

    # 保存结果
    if args.output:
        output_data = {
            "data_file": str(args.data_file),
            "total_records": stats["total_records"],
            "single_entity_records": stats["single_entity_records"],
            "multi_entity_records": stats["multi_entity_records"],
            "multi_entity_2_3": stats["multi_entity_2_3"],
            "multi_entity_4plus": stats["multi_entity_4plus"],
            "entity_count_distribution": dict(stats["entity_count_distribution"]),
            "type_distribution": dict(stats["type_distribution"]),
            "overlap_records": overlap_stats["records_with_overlap"],
            "overlap_pairs": overlap_stats["total_overlap_pairs"],
        }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
