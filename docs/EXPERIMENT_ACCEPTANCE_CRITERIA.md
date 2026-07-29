# 实验验收标准（统一版）

本文档定义所有 GMNER/FMNERG 实验的验收标准和状态分类规则，与 [EXPERIMENT_RESULTS_TABLE.md](EXPERIMENT_RESULTS_TABLE.md) 的状态标签体系保持一致。

---

## 状态标签体系

所有实验必须归入以下 7 种状态之一：

| 状态 | 含义 | 验收要求 |
|------|------|----------|
| **FORMAL** | 锁定且符合正式 Test 协议的结果 | 见"FORMAL 验收标准" |
| **VALID_DEV** | 有效 Dev 实验或消融 | 见"VALID_DEV 验收标准" |
| **VALID_AUDIT** | Train-OOF、cross-fit、分布和协议审计 | 见"VALID_AUDIT 验收标准" |
| **ORACLE** | 使用 gold 的理想理论上限 | 见"ORACLE 验收标准" |
| **ENGINEERING_ONLY** | 有信号但协议不满足正式报告要求 | 见"ENGINEERING_ONLY 验收标准" |
| **NO_GO** | 预注册 Gate 失败，分支关闭 | 见"NO_GO 验收标准" |
| **ENGINEERING_HISTORY** | 早期探索和历史 Test 结果 | 不接受新实验，仅归档已有结果 |

---

## FORMAL 验收标准

**用途：** 论文 Test 主表或正式 Test 消融表

### 必须满足（ALL）：

1. **严格一次性 Test 协议**
   - Dev 上选择所有配置和 checkpoint
   - Test 只读取一次，不再修改任何内容
   - `test_accessed` 在实验过程中始终为 `false`，直到最终提交

2. **完整链路可复现**
   - 所有 checkpoint 文件完整且可加载
   - 配置文件明确且版本控制
   - README/文档中有完整复现步骤

3. **指标精确复现**
   - 提供的 checkpoint 能精确复现报告的 Dev/Test 指标
   - 允许的误差：F1 < 1e-6

4. **协议透明**
   - 训练数据来源明确（Train/Train+Dev）
   - 训练-推理分布差异（如有）已披露
   - OOF/non-OOF 明确标注

5. **完整性**
   - Dev 和 Test 结果都已报告
   - 如果是 multi-seed 实验，报告 mean ± std
   - 关键消融或对比已完成

### 验收流程：

```
1. 提交实验报告 → README/docs 中的正式章节
2. 代码审查：checkpoint/config/评估脚本完整性
3. 复现验证：第三方从 checkpoint 复现指标
4. 协议审查：确认符合一次性 Test 协议
5. 批准进入 FORMAL，写入 EXPERIMENT_RESULTS_TABLE.md
```

### 典型示例：

- Model-G (M3.3A) Dev/Test
- Model-F (F2 Subtype Sidecar) Dev/Test
- M3.3A 分阶段 Test 消融

---

## VALID_DEV 验收标准

**用途：** 论文 Dev 消融、方法对比

### 必须满足（ALL）：

1. **仅在 Dev 上评估**
   - 不访问 Test
   - `test_accessed = false`

2. **可复现**
   - Checkpoint 和配置文件完整
   - Dev 指标可精确复现

3. **有效基线或消融**
   - 与 FORMAL 结果或其他 VALID_DEV 有明确对比关系
   - 提供有意义的方法学洞察

4. **协议清晰**
   - 训练数据、模型结构、超参数明确

### 验收流程：

```
1. 提交实验报告
2. 确认 test_accessed = false
3. 验证 Dev 指标可复现
4. 确认与已有结果的对比关系清晰
5. 批准进入 VALID_DEV
```

### 典型示例：

- Frozen F0 subtype head (F2 基线)
- RoBERTa Last-4 (编码器消融)
- 各种 FMNERG loss 变体

---

## VALID_AUDIT 验收标准

**用途：** Train-OOF、cross-fit、分布和协议审计

### 必须满足（ALL）：

1. **非 Dev/Test 主结果**
   - 在 Train OOF 或其他 held-out split 上评估
   - 或用于协议验证、分布分析

2. **严格 OOF 协议**（如适用）
   - 明确的 fold 划分
   - 每条记录的预测来自未见过该记录的模型
   - Pooled micro 和 fold-level 统计都报告

3. **审计目的明确**
   - 泛化分析
   - 协议对比
   - 分布错位检测
   - 训练信号充分性验证

4. **不与 Dev 直接对比**
   - 明确说明为什么不能与 Dev FORMAL 结果直接比较
   - Split/size/protocol 差异已披露

### 验收流程：

```
1. 提交审计报告
2. 验证 OOF 协议（如适用）
3. 确认审计目的和结论清晰
4. 确认不误导为 Dev 消融
5. 批准进入 VALID_AUDIT
```

### 典型示例：

- M3.3A 10-fold Train-OOF cross-fitted estimate
- Fold-level statistics

---

## ORACLE 验收标准

**用途：** 使用 gold 的理想理论上限，方法空间分析

### 必须满足（ALL）：

1. **使用 gold 信息**
   - 明确说明使用了哪些 gold 信息
   - 不能作为可部署方法

2. **理论上限计算清晰**
   - 假设明确（fixed-denominator, zero-damage, etc.）
   - 计算方法可验证

3. **用途限定**
   - 只用于方法空间分析
   - 不能作为预期收益
   - 不能直接与 FORMAL 结果对比

4. **完整文档**
   - Oracle 分析报告
   - Gold 信息使用说明
   - 理论上限 vs 实际可实现空间的讨论

### 验收流程：

```
1. 提交 Oracle 分析报告
2. 验证 gold 信息使用透明
3. 验证理论上限计算正确
4. 确认用途限定清晰
5. 批准进入 ORACLE
```

### 典型示例：

- P1 Visibility Gold Oracle (+235 ceiling)
- P2 Span Recovery Gold Oracle (+0.08801 ceiling)
- P3 Same-Type Assignment Oracle (+0.00969 ceiling)

---

## ENGINEERING_ONLY 验收标准

**用途：** 内部工程验证，有信号但协议不满足正式报告要求

### 适用场景：

1. **使用 in-sample/full-fit cache 训练**
   - 训练特征来自同一模型在 Train 上的预测
   - 不符合 OOF 协议

2. **快速工程信号验证**
   - 用于决定是否值得投入完整实验
   - 不用于论文报告

3. **协议不完整**
   - 缺少某些验收要素
   - 但提供有价值的工程洞察

### 验收流程：

```
1. 提交工程报告
2. 明确标注协议限制
3. 说明为什么不能升级为 VALID_DEV/FORMAL
4. 批准进入 ENGINEERING_ONLY
```

### 典型示例：

- M3.6A-r1 (in-sample cache, Dev GMNER 0.623738)

---

## NO_GO 验收标准

**用途：** 预注册 Gate 失败，分支关闭，记录阴性结果

### 必须满足（ALL）：

1. **预注册 Gate 存在**
   - 实验开始前明确 Gate 条件
   - Gate 条件合理且可验证

2. **完整执行**
   - 按预注册协议完整执行实验
   - 不因中间结果不理想而提前终止

3. **Gate 失败有记录**
   - 明确报告哪些 Gate 条件未满足
   - 提供失败原因分析

4. **阴性结果有价值**
   - 说明为什么该方向 no-go
   - 为未来研究提供参考

5. **不访问 Test**
   - Gate 失败后不读 Test
   - `test_accessed = false`

### 验收流程：

```
1. 提交 no-go 报告
2. 验证预注册 Gate 存在
3. 验证实验完整执行
4. 验证 Gate 失败判断正确
5. 确认 test_accessed = false
6. 批准进入 NO_GO
```

### 典型示例：

- M3.6A-r2 NULL Release (best epoch 0)
- P1 Observable Visibility Override (net +4, precision 0.50)
- P3 C1 Resolver MVP (GMNER delta 0)

---

## ENGINEERING_HISTORY 验收标准

**用途：** 早期探索和历史 Test 结果，仅归档已有结果

### 说明：

- **不接受新实验进入此状态**
- 仅用于归档早于严格一次性 Test 协议的历史结果
- 已有的 ENGINEERING_HISTORY 实验不可用于：
  - 正式论文 Test 消融主表
  - 模型选择的主要证据
  - 显著性统计比较

### 允许用途：

- 内部实验历史记录
- 路线追踪和定性说明
- 描述研究演进
- 说明哪些早期路线被放弃

### 典型示例：

- mBERT Stage1 及各种早期方法
- 早期 OOF Evidence Graph (Test 0.58035)

---

## 状态升级和降级规则

### 升级路径：

```
ENGINEERING_ONLY → VALID_DEV
  条件：补全协议，改用 OOF 或严格 split

VALID_DEV → FORMAL
  条件：通过 Dev Gate，执行一次性 Test

ORACLE → VALID_DEV
  条件：实现不使用 gold 的可部署版本
```

### 降级路径：

```
VALID_DEV → NO_GO
  条件：后续实验发现严重问题，Gate 失败

FORMAL → ENGINEERING_HISTORY
  条件：发现违反一次性 Test 协议（极少发生，需要严肃处理）
```

### 禁止的状态转换：

```
ENGINEERING_HISTORY → 任何其他状态
  理由：历史实验不能升级为正式结果

NO_GO → VALID_DEV/FORMAL
  理由：已关闭的分支不能重新激活，除非是全新实验
```

---

## 实验提交检查清单

每个实验提交到 EXPERIMENT_RESULTS_TABLE.md 前必须完成：

### 通用检查项：

- [ ] 目标状态明确（FORMAL/VALID_DEV/VALID_AUDIT/ORACLE/ENGINEERING_ONLY/NO_GO）
- [ ] 符合该状态的所有验收标准
- [ ] `test_accessed` 状态正确
- [ ] Checkpoint/config 文件完整
- [ ] 指标可复现
- [ ] 文档完整（README 或 docs/ 中有正式章节）

### FORMAL 额外检查项：

- [ ] Dev 上选择所有配置和 checkpoint
- [ ] Test 只读取一次
- [ ] 训练-推理协议透明披露
- [ ] Dev 和 Test 结果都已报告

### VALID_AUDIT 额外检查项：

- [ ] OOF 协议严格（如适用）
- [ ] Fold 划分明确
- [ ] Pooled 和 fold-level 统计都报告
- [ ] 与 Dev 不直接对比的原因已说明

### ORACLE 额外检查项：

- [ ] Gold 信息使用透明
- [ ] 理论上限假设明确
- [ ] 用途限定清晰（不作为预期收益）

### NO_GO 额外检查项：

- [ ] 预注册 Gate 文档存在
- [ ] 实验完整执行
- [ ] Gate 失败原因分析清晰
- [ ] 未访问 Test

---

## 与 EXPERIMENT_RESULTS_TABLE.md 的关系

- **本文档定义规则**：什么样的实验可以进入哪种状态
- **EXPERIMENT_RESULTS_TABLE.md 记录结果**：已验收的实验的统一索引

**流程：**

```
1. 实验执行
2. 按本文档验收标准检查
3. 通过验收 → 写入 EXPERIMENT_RESULTS_TABLE.md
4. 未通过验收 → 补充/修正/关闭
```

---

## 特殊情况处理

### 1. 实验部分满足 FORMAL 但 Test 未运行

**状态：** VALID_DEV

**条件：** 等待 Dev Gate 通过后才运行 Test

---

### 2. OOF 实验但在 Dev 上评估

**状态：** 不明确，需要根据目的判断

**判断：**
- 如果是 Dev 消融 → VALID_DEV
- 如果是协议审计 → VALID_AUDIT
- 如果是 OOF 训练 + Dev 配对评估（如 P3 C1）→ NO_GO（如果失败）

---

### 3. Oracle 的可部署实现失败

**Oracle 状态：** ORACLE（保持）

**可部署实现状态：** NO_GO（新增一行）

**说明：** Oracle 和实现是两个独立实验

---

### 4. 历史实验需要重新验证

**不允许**：ENGINEERING_HISTORY 实验不能重新验证并升级状态

**替代方案**：设计全新实验，按当前协议重新执行

---

## 版本信息

- **创建日期：** 2026-07-27
- **最后更新：** 2026-07-27
- **关联文档：** [EXPERIMENT_RESULTS_TABLE.md](EXPERIMENT_RESULTS_TABLE.md)
- **适用范围：** 所有 GMNER/FMNERG 相关实验

---

**本文档是实验验收的唯一标准，所有新实验必须按此文档验收后才能进入 EXPERIMENT_RESULTS_TABLE.md。**
