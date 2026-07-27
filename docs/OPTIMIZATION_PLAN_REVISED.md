# 基于现有代码的优化计划重整

**日期**: 2026-07-27  
**目标**: 基于 M3.3A 现有架构进行最小改动优化

---

## 现有架构理解

### M3.3A 核心组件

```
GMNERModel (gmner_model.py)
├─ TextEncoder (RoBERTa)
├─ ImageEncoder (VinVL)
├─ TextGraphEncoder (图编码器)
├─ ImageGraphEncoder (图编码器)
├─ CrossModalAligner (跨模态对齐)
├─ NER Head (实体识别)
├─ GroundingHead (区域匹配)
└─ Optional:
   ├─ SemanticPrototypeBank
   ├─ ExternalKnowledgeBank
   ├─ MultiScaleGroundingAligner
   └─ FineSubtypeHead
```

### 训练流程

1. **数据准备**: CoNLL → JSON → Cache
2. **Stage 训练**:
   - Evidence Visibility
   - Coarse Region Selector
   - Hierarchical Record Verifier
   - Fine Grounding Adapter
3. **评估**: scripts/evaluate.py

---

## 当前问题分析

### 已识别瓶颈（来自优化报告）

1. **实体-区域匹配差距**: MNER 0.818 vs EEG 0.652 (16%)
2. **多实体场景困难**: 单实体 AUROC 0.72, 多实体 0.52
3. **独立匹配限制**: 当前是逐实体匹配，缺乏关系建模

### 数据验证（Phase 1）

- Dev 集 42.8% 多实体记录（642条）
- Person-Organization 共现最频繁（180次）
- 样本充足，支持优化

---

## 正确的优化策略

### ❌ 之前的错误方向

1. **从零搭建新模块** - 创建了独立的 entity_relation_encoder.py
2. **未考虑现有架构** - 没有查看如何集成到 GMNERModel
3. **未理解训练流程** - 不知道如何训练和评估

### ✅ 正确的方向

1. **在现有模块基础上增强**
2. **最小改动原则**
3. **保持向后兼容**
4. **逐步验证改进**

---

## 推荐的优化路径

### 方案 A: 增强 CrossModalAligner（推荐）

**原理**: 当前 CrossModalAligner 进行跨模态对齐，但是**独立处理每个实体**。

**改进**: 在对齐之前，先用关系编码增强实体表示。

**修改位置**: `gmner/models/aligner.py`

**改动量**: 最小（~100行代码）

**步骤**:
1. 在 `CrossModalAligner.forward()` 之前
2. 添加可选的 entity relation aggregation
3. 使用简化的图注意力（不需要完整 GAT）
4. 配置开关控制是否启用

**代码示例**:
```python
class CrossModalAligner(nn.Module):
    def __init__(self, ..., use_entity_relations=False):
        ...
        if use_entity_relations:
            self.entity_relation_aggregator = SimpleEntityRelationLayer(...)
    
    def forward(self, text_features, image_features, entity_mask=None):
        # 新增: 实体关系聚合
        if self.entity_relation_aggregator and entity_mask is not None:
            text_features = self.entity_relation_aggregator(
                text_features, entity_mask
            )
        
        # 原有逻辑
        return self.cross_attention(text_features, image_features)
```

**优势**:
- 改动最小
- 向后兼容（默认关闭）
- 易于消融实验
- 直接影响匹配质量

---

### 方案 B: 增强 GroundingHead

**原理**: 在最终的 grounding 打分时考虑实体间关系。

**修改位置**: `gmner/models/heads.py` 的 `GroundingHead`

**改动量**: 中等（~150行）

**步骤**:
1. 在 grounding score 计算时
2. 添加 relation-aware 权重调整
3. 多实体场景使用集合打分

---

### 方案 C: 新增 Multi-entity Branch（原报告建议）

**原理**: 场景分类 → 单/多实体分支

**改动量**: 大（需要修改 GMNERModel 主流程）

**不推荐原因**:
- 改动太大
- 增加训练复杂度
- Scene Analyzer 已证明过于简单（100%准确率）

---

## 立即执行计划（修正版）

### Week 1 剩余时间

#### Task 1: 环境修复 ✅ 进行中
```bash
# 等待 NumPy 降级完成
ssh server4090 "~/miniconda3/envs/gmner/bin/python -c 'import torch'"
```

#### Task 2: 理解现有代码（今天）
- [x] 查看 GMNERModel 结构
- [x] 查看训练脚本
- [ ] 查看 CrossModalAligner 实现
- [ ] 查看评估流程
- [ ] 运行一次完整的 train+eval

#### Task 3: 实现方案 A（1-2天）
- [ ] 在 aligner.py 添加 SimpleEntityRelationLayer
- [ ] 添加配置开关
- [ ] 本地测试前向传播
- [ ] 服务器测试训练

#### Task 4: 实验验证（2-3天）
- [ ] Baseline: 当前 M3.3A
- [ ] +Relation: 开启关系聚合
- [ ] 评估 Dev GMNER
- [ ] 分析多实体切片

### Week 2

#### Task 5: 消融与优化
- [ ] 消融：spatial vs semantic vs context
- [ ] 超参数调优
- [ ] 训练策略优化

#### Task 6: 文档与提交
- [ ] 实验报告
- [ ] 代码文档
- [ ] Git 提交

---

## 预期收益（修正）

### 保守估计（更现实）

**方案 A (Relation-enhanced Aligner)**:
- Overall GMNER: 0.621 → 0.625 (+0.4%)
- Multi-entity 切片: +1.0-1.5%

**原因**: 改动小，影响范围有限

### 如果效果好，再扩展

如果方案 A 有效（Dev +0.3%以上），再考虑：
- 方案 B: 增强 GroundingHead
- 或在 HierarchicalRecordVerifier 中添加关系建模

---

## 文件修改清单（方案 A）

### 需要修改的文件

1. **gmner/models/aligner.py**
   - 添加 `SimpleEntityRelationLayer` 类
   - 修改 `CrossModalAligner.__init__` 和 `forward`
   - ~100 行新增

2. **gmner/config.py**
   - 添加配置项: `use_entity_relations: bool = False`
   - ~5 行新增

3. **configs/xxx.yaml**
   - 新建实验配置文件
   - 基于现有配置 + `use_entity_relations: true`

4. **测试文件（可选）**
   - tests/test_relation_aligner.py
   - 验证前向传播

### 不需要修改的文件

- gmner/models/gmner_model.py（无需改动！）
- scripts/train.py（无需改动！）
- scripts/evaluate.py（无需改动！）

---

## 下一步

等待：
1. NumPy 安装完成
2. 环境验证通过

然后：
3. 阅读 CrossModalAligner 源码
4. 设计 SimpleEntityRelationLayer
5. 实现并测试

您同意这个**最小改动**的方案 A 吗？
