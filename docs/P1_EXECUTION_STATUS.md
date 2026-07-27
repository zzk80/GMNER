# P1 执行状态 - 2026-07-27 18:00

## 当前状态

**脚本**: `scripts/diagnose_entity_count.py`  
**状态**: 后台运行中 (任务 ID: bmyot7t6j)  
**方法**: 方案 B - 直接复用 Evidence Visibility 评估框架

---

## 实现方式

### 复用的组件
- ✅ `PairedRecordCandidateDataset`
- ✅ `PairedRecordCandidateCollator`
- ✅ `frozen_hierarchical_context()`
- ✅ `_selected_span_indices()`
- ✅ `move_paired_record_batch()`
- ✅ `match_record_predictions()`
- ✅ `decode_evidence_visibility()`
- ✅ `load_frozen_chain()` (本地实现)

### Gold 实体来源
```python
gold = list(metadata.get("gold_entities") or [])
gold_entity_count = len(gold)
```

从 `expanded["metadata"][row]["gold_entities"]` 直接获取

### 诊断信息提取
在逐记录循环中提取：
- record_id, text
- gold_entity_count, pred_entity_count
- gold_entities (span, type_id, visible)
- predicted_entities (final predictions)
- matches (span/MNER/EEG/GMNER)

### 切片分类
- `gold_single`: gold_entity_count == 1
- `gold_multi`: gold_entity_count >= 2

---

## 预期输出

### 文件
```
outputs/diagnostics/m33a_entity_count/
  records.jsonl          # 逐记录详情
  summary.json           # 切片汇总指标
  protocol.json          # SHA-256 + git commit
```

### 指标
```json
{
  "overall": {
    "gmner_f1": 0.621316,  // 目标
    "mner_f1": 0.816714,   // 目标
    "eeg_f1": 0.660880,    // 目标
    "record_count": 1500
  },
  "gold_single": {
    "gmner_f1": ...,
    "mner_f1": ...,
    "eeg_f1": ...,
    "record_count": ...
  },
  "gold_multi": {
    "gmner_f1": ...,
    "mner_f1": ...,
    "eeg_f1": ...,
    "record_count": ...
  }
}
```

---

## 验收标准

### 必须通过
1. ✅ **整体指标复现**:
   - Dev GMNER: 0.621316 (容差 ±0.0001)
   - Dev MNER: 0.816714 (容差 ±0.0001)
   - Dev EEG: 0.660880 (容差 ±0.0001)

2. ⏳ **切片合理性**:
   - single + multi record_count = 1500
   - single + multi 的 gold/predicted 加总 = overall

### 如果不通过
- 停止切片分析
- 修正解码路径
- 重新验证

---

## 已修复的问题

### 问题 1: ImportError
- ❌ `from gmner.engine.fine_grounding_adapter_evaluator import load_frozen_chain`
- ✅ 本地实现 `load_frozen_chain()`

### 问题 2: Dataset 初始化
- ❌ `PairedRecordCandidateDataset(formal_cache=..., expanded_cache=...)`
- ✅ `PairedRecordCandidateDataset(formal_dataset, expanded_dataset)`

---

## 第一版功能范围

### ✅ 包含
- 逐记录 JSONL 输出
- Single/Multi 切片的 micro metrics
- Gold entity count 分类
- Span/MNER/EEG/GMNER F1

### ❌ 不包含（按要求）
- AUROC 分析
- Bootstrap 统计
- false-NULL 定位 (140个)
- misranking 定位 (97个)
- 各阶段归因 (Verifier/Fine/Visibility)
- R16/R36 覆盖率分析
- Test 集访问

---

## 下一步

### 如果整体指标复现成功
1. 分析 Single vs Multi 的差异
2. 如果差异显著，完善 P1 功能：
   - 添加 false-NULL 定位
   - 添加 misranking 定位
   - 添加错误记录 ID 列表
3. 执行 P2 (Bootstrap)
4. 执行 P3 (可审计输出)
5. 执行 P4 (决策规则)

### 如果整体指标不能复现
1. 检查解码逻辑差异
2. 对比 `evaluate_evidence_visibility.py`
3. 修正并重新验证
4. 不进行切片分析

---

## 当前时间线

```
15:00 - 开始方案 B
15:30 - 创建脚本框架
16:00 - 修复 import 错误
16:30 - 修复 dataset 初始化
17:00 - 脚本运行中...
18:00 - 等待结果
```

**预计完成时间**: 18:05 (~5 分钟推理时间)

---

**等待后台任务完成通知...**
