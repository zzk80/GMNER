# M3.3A 正式主链运行路线

**基于文档和代码的完整理解**

---

## 核心架构

M3.3A 使用 **层次化分解** 而非平坦联合空间：

```
P(span, type, region) = P_entity(span) 
                      × P_type(type | span)
                      × P_visibility(visible | span, type)
                      × P_region(region | span, type, visible)
```

---

## 完整训练链路

### Stage 1: 基础 Span + Type 预测

**模型**: RoBERTa-base (max_length=128)

**训练**:
```bash
PYTHONPATH=. python scripts/train.py \
  --config configs/fmnerg_twitter10000_stage1.yaml
```

**输出**:
- Span boundaries (CRF)
- Type predictions (4类: PER, ORG, LOC, MISC)
- 初步 region scores (Top-16)

**当前性能**: 
- Dev GMNER: 0.621316
- Test GMNER: 0.61529

---

### Stage 2: 候选缓存构建

**目的**: 为每个 span 构建候选区域集合

**步骤**:
1. **构建 record candidates**:
```bash
PYTHONPATH=. python scripts/build_record_candidate_cache.py \
  --stage1-checkpoint outputs/stage1/best_model.pt \
  --output knowledge/record_candidates/roberta128/
```

2. **生成层次化候选**:
```bash
PYTHONPATH=. python scripts/build_oof_hierarchical_candidates.py
```

**候选策略**:
- Detector Top-16: VinVL 检测器的 top-16 区域
- Stage1 Top-16: Stage1 打分的 top-16
- 添加 NULL 选项

---

### Stage 3: 层次化 Verifier 训练

**组件顺序**:

#### 3.1 Evidence Visibility
**功能**: 判断实体是否在图像中可见

```bash
PYTHONPATH=. python scripts/train_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml
```

**决策**:
- P(visible | span, type)
- 双阈值机制避免频繁翻转

#### 3.2 Coarse Region Selector (可选)
**功能**: 从 R36 粗筛到 R16

```bash
PYTHONPATH=. python scripts/train_coarse_region_selector.py \
  --config configs/fmnerg_twitter10000_coarse_selector.yaml
```

**状态**: Milestone 3.1 验证中

#### 3.3 Hierarchical Record Verifier (核心)
**功能**: 联合优化 entityness, visibility, region ranking

```bash
PYTHONPATH=. python scripts/train_hierarchical_record_verifier.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml
```

**损失函数**:
```python
L = λ_entity * L_entityness
  + λ_visibility * L_visibility  
  + λ_region_multi * L_multi_positive
  + λ_region_iou * L_iou_aware
  + λ_region_hard * L_hard_negative
  + λ_region_preserve * L_base_preservation
```

**训练设置**:
- Batch size: 8
- Learning rate: 1e-4
- Epochs: 12
- 早停: patience=3

#### 3.4 Fine Grounding Adapter (可选)
**功能**: Correction-preservation grounding 微调

```bash
PYTHONPATH=. python scripts/train_fine_grounding_adapter.py \
  --config configs/fmnerg_twitter10000_fine_grounding_adapter.yaml
```

**状态**: Milestone 3.2，取决于 3.1 结果

---

### Stage 4: 推理解码

**解码策略**: Balanced Hierarchical Decoder

```python
for each span:
    # 1. Entityness 过滤
    if P_entity(span) < threshold:
        continue
    
    # 2. Type (fixed from Stage1)
    type = stage1_type
    
    # 3. Visibility 判断
    if P_visible < 0.2:
        output (span, type, NULL)
    elif P_visible > 0.8:
        # 4. Region ranking
        region = argmax P_region(r | span, type, visible)
        output (span, type, region)
    else:
        # 保持 Stage1 决策
        output stage1_decision
```

**非重叠约束**: Interval-based decoding

---

### Stage 5: 评估

```bash
PYTHONPATH=. python scripts/evaluate.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml \
  --checkpoint outputs/.../best_model.pt \
  --split dev
```

**指标**:
- GMNER (主指标): F1 of (span, type, region) triples
- MNER: F1 of (span, type)
- EEG: Entity-region grounding accuracy

---

## 关键配置文件

### configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml

```yaml
data:
  train_cache: knowledge/record_candidates/.../train.pt
  dev_cache: knowledge/record_candidates/.../dev.pt
  test_cache: knowledge/record_candidates/.../test.pt

model:
  hidden_size: 256
  num_types: 4
  dropout: 0.2
  base_region_temperature: 1.0
  region_residual_scale: 0.25

optim:
  batch_size: 8
  learning_rate: 0.0001
  num_epochs: 12
  gradient_clip_norm: 1.0

loss:
  lambda_entity: 1.0
  lambda_visibility: 1.0
  lambda_region_multi_positive: 1.0
  lambda_region_iou: 0.2
  lambda_region_hard: 0.5
  lambda_region_preserve: 0.5
  grounding_stage1_only: true

decode:
  entity_threshold: 0.0
  strategy: interval
  stage1_spans_only: true
  enable_visibility_correction: true
  enable_region_override: true
  visible_from_null_threshold: 0.8
  null_from_visible_threshold: 0.2
```

---

## Oracle 诊断结果

**Dev 集 (991 个 visible gold)**:

| 指标 | 值 |
|------|-----|
| Gold in R16 | 827 (83.5%) |
| Gold in R36 | 891 (89.9%) |
| R36 新增覆盖 | 64 (+6.5%) |
| Stage1 错误但 gold in R16 | 219 |
| 最终 false-NULL (gold in R16) | 140 |
| R16 内真实区域排序错误 | 97 |

**结论**: R36 覆盖率提升显著，但直接用 R36 会降低性能 (0.600 vs 0.607)

---

## 当前瓶颈分析

### 1. 实体-区域匹配差距
- MNER: 0.818
- EEG: 0.652
- **Gap: 16.6%** ← 主要优化空间

### 2. 多实体场景困难
- 单实体 AUROC: 0.72
- 多人场景 AUROC: 0.52
- **Gap: 20%**

### 3. 分解误差累积
```
P_entity error × P_visibility error × P_region error
→ 多个模块的误差会累积
```

---

## 优化机会

基于以上理解，**可优化的位置**:

### 位置 1: CrossModalAligner (✓ 推荐)
- **当前**: 独立对齐每个 text node → image nodes
- **问题**: 未考虑实体间关系
- **改进**: 对齐前先聚合实体关系
- **改动**: 最小 (~100行)

### 位置 2: HierarchicalRecordVerifier
- **当前**: 逐 span 独立预测
- **问题**: 多实体场景缺乏联合建模
- **改进**: 添加 inter-entity context
- **改动**: 中等 (~200行)

### 位置 3: GroundingHead
- **当前**: 独立打分每个 (entity, region) pair
- **问题**: 同一图像的多实体未联合考虑
- **改进**: Set-level scoring
- **改动**: 大 (~300行)

---

## 我的理解是否正确？

请确认：
1. **训练链路**: Stage1 → 候选缓存 → Verifier训练 → 解码评估
2. **核心模块**: HierarchicalRecordVerifier 是主要优化目标
3. **优化位置**: CrossModalAligner 是最小改动入口点
4. **评估指标**: Dev GMNER 是主指标，需要严格 OOF

如果理解有误，请指正！
