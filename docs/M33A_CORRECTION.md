# M3.3A 理解纠错文档

**日期**: 2026-07-27  
**状态**: 纠正之前理解中的关键错误

---

## 🚨 关键错误总结

### 错误 1: Stage1 指标混淆

**错误理解**:
- Stage1 Dev GMNER: 0.621316
- Stage1 Test GMNER: 0.61529

**正确理解**:
- **Stage1 bypass** Dev GMNER: 0.60733
- **Stage1 bypass** Test GMNER: 0.59168
- **最终 M3.3A 完整链** Dev GMNER: 0.621316
- **最终 M3.3A 完整链** Test GMNER: 0.61529

**提升**: 完整链相对 Stage1 提升 +1.4% (0.607→0.621)

---

### 错误 2: 训练顺序完全错误

**错误理解**:
```
Evidence Visibility
→ Coarse Region Selector
→ Hierarchical Record Verifier
→ Fine Grounding Adapter
```

**正确顺序**:
```
RoBERTa Stage1 (span/type/初步region)
→ 离线构建 R16 候选缓存
→ 离线构建 R36 扩展缓存
→ Hierarchical Record Verifier (核心修正)
→ Coarse Selector (Base Top-8 + Learned Top-8)
→ Fine Grounding Adapter (correction-preservation)
→ Evidence Visibility (最终可见性判断)
→ Record-level Decode
```

**关键认识**:
- 候选缓存是**离线物化**，不是训练阶段
- Evidence Visibility 是**最后一个**训练模块
- Coarse/Fine/Visibility 都是**正式主链**，不是"可选"

---

### 错误 3: 模块状态描述过期

**错误描述**:
- Coarse Region Selector: "可选，Milestone 3.1 验证中"
- Fine Grounding Adapter: "可选，取决于 3.1"

**正确状态**:
- 两者都已是**正式 M3.3A 主链组件**
- 不是"验证中"，而是已完成并进入正式结果

---

### 错误 4: CrossModalAligner 不是"最小改动"

**错误判断**:
- 认为修改 aligner.py (~100行) 是最小改动
- 认为可以直接在对齐前添加实体关系聚合

**正确认识**:
1. **接口问题**: CrossModalAligner 接收的是 **所有 token** 和 **region**，没有实体 span 或 entity mask
2. **时序问题**: 对齐时 CRF **还没输出**实体边界
3. **依赖链问题**: 修改后必须：
   ```
   重训 Stage1
   → 重建 R16 缓存
   → 重建 R36 缓存
   → 重训 Hierarchical Verifier
   → 重训 Coarse Selector
   → 重训 Fine Adapter
   → 重训 Evidence Visibility
   → 重新做 OOF
   ```
4. **结论**: 这是"代码局部改动"，**不是**"工程最小改动"

---

### 错误 5: 多实体 AUROC 数据无来源

**错误使用**:
- 单实体 AUROC 0.72
- 多实体 AUROC 0.52
- 作为实施依据

**问题**:
- 这些数字**未出现**在正式文档中
- 缺少：对应脚本、checkpoint、切片定义、输出文件
- **不能**作为实施依据

**必须补充**:
1. 诊断脚本路径
2. 输入 checkpoint
3. 切片定义（如何定义多实体？）
4. positive/negative 定义
5. AUROC 输出文件
6. 记录数量

---

### 错误 6: 实体类型名称错误

**错误**: PER, ORG, LOC, MISC

**正确**: PER, ORG, LOC, **OTHER**

---

### 错误 7: 匹配差距解释不准确

**错误表述**: "实体-区域匹配准确率相差 16.6%"

**问题**: MNER 0.818 vs EEG 0.652 不能直接相减
- 两者分母和条件不完全相同
- 不是同一概率的简单减法

**正确依据**:
```
R16 visible coverage  = 0.83451
R36 visible coverage  = 0.89909
final false-NULL      = 140
R16 real misranking   = 97
```

---

## ✅ 正确的完整链路

### 训练阶段

```
1. RoBERTa Stage1
   输入: 文本 + 图像
   输出: spans, types, 初步 region scores
   性能: Dev GMNER 0.60733

2. 候选缓存构建（离线）
   命令:
   PYTHONPATH=. python scripts/build_record_candidate_cache.py \
     --config configs/fmnerg_twitter10000_stage1.yaml \
     --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
     --split dev \
     --output knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical.pt

3. Hierarchical Record Verifier
   作用: entityness + visibility + region ranking
   提升: 0.607 → 0.615 (+0.8%)

4. Coarse Region Selector
   作用: Base Top-8 + Learned Top-8
   
5. Fine Grounding Adapter
   作用: correction-preservation grounding

6. Evidence Visibility
   作用: 最终可见性判断
   最终: Dev GMNER 0.621316 (+1.4% vs Stage1)
```

### 评估入口

**正确命令**:
```bash
PYTHONPATH=. python scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/.../best_model.pt \
  --split dev
```

**不是**: `scripts/evaluate.py` (通用评估脚本)

---

## 🎯 正确的优化入口

### 如果要做实体关系建模

**不应该改**: CrossModalAligner
- 原因: 工程依赖链太长，不是最小改动

**应该考虑**: HierarchicalRecordVerifier
- 原因:
  1. 冻结 Stage1 和缓存，只重训 Verifier 及下游
  2. Verifier 确实对每个 span 独立处理
  3. 代码中没有 span-to-span self-attention
  
**前提条件**:
1. **必须先完成正式诊断**
2. 必须有可审计的多实体切片分析
3. 必须证明多实体场景确实存在显著瓶颈
4. 不能基于无来源的 AUROC 数字

---

## 📋 当前应该做的事

### ❌ 不应该做
- ~~实现 CrossModalAligner 优化~~
- ~~基于无来源的 AUROC 数据做决策~~
- ~~修改任何模型代码~~

### ✅ 应该做

1. **补充多实体诊断** (优先级最高)
   ```bash
   # 需要创建或找到诊断脚本
   # 分析多实体 vs 单实体场景的性能差异
   # 输出可审计的结果
   ```

2. **理解 HierarchicalRecordVerifier 代码**
   - 查看如何构造 span features
   - 查看是否真的完全独立
   - 查看添加 inter-span context 的可行性

3. **Oracle 分析**
   - 理解 140 个 false-NULL
   - 理解 97 个 misranking
   - 定位具体是哪些记录

4. **修订理解文档**
   - 提交纠错 commit
   - 更新所有错误描述

---

## 🔄 下一步行动

### 立即执行

1. **提交纠错 commit**
2. **暂停所有模型改动**
3. **补充多实体诊断脚本**
4. **等待诊断结果**

### 等诊断完成后

如果多实体确实有显著瓶颈:
- 考虑在 HierarchicalRecordVerifier 添加关系建模
- 设计低依赖成本的实现方案
- 小规模实验验证

如果多实体瓶颈不显著:
- 放弃实体关系方向
- 转向其他优化点（如 Oracle 发现的 140/97 问题）

---

## 总结

**我之前的理解存在7个关键错误，不能作为实施依据。**

**纠正后的认识**:
1. M3.3A 是完整5模块链路，不是单一 Stage1
2. CrossModalAligner 不是最小改动入口
3. 必须先做正式诊断，不能基于无来源数据
4. 当前最稳妥的决定是：保留作为草稿，先纠错，不实施模型改动

**下一步**: 提交纠错 commit，补充多实体诊断
