# 有效实验结果总表（最终版）

以下只统计具有明确配置、checkpoint、指标或审计文件的实验。OOM、中断、smoke run、损坏 checkpoint 和用 Test 反复调参的结果不算正式有效结果。

---

## 状态标签说明

| 状态 | 含义 | 用途 |
|------|------|------|
| **FORMAL** | 锁定且符合正式 Test 协议的结果 | 论文 Test 主表或正式 Test 消融表 |
| **VALID_DEV** | 有效 Dev 实验或消融 | 论文 Dev 消融、方法对比 |
| **VALID_AUDIT** | 有效但不属于 Dev/Test 主结果的审计或 cross-fit 评估 | OOF 泛化、分布和协议审计 |
| **ORACLE** | 使用 gold 的理想理论上限 | 方法空间分析，不作为可部署结果 |
| **ENGINEERING_ONLY** | 存在信号，但协议不满足正式报告要求 | 内部工程验证 |
| **NO_GO** | 完整预注册 Gate 失败，分支关闭 | 阴性结果记录 |
| **ENGINEERING_HISTORY** | 早期探索和历史 Test 结果 | 路线追踪、定性说明 |

---

## 1. 当前正式双主链（FORMAL）

### Model-G：M3.3A 层次化 GMNER

```
RoBERTa Stage1
→ R16 候选生成
→ R36 候选扩展
→ Hierarchical Record Verifier
→ Base Top-8 + Learned Top-8 候选选择
→ Fine Grounding Adapter
→ Evidence Visibility
→ Record-level Decode
→ GMNER 三元组输出
```

### Model-F：F3 Subtype Sidecar

```
冻结 Model-G 的 span / coarse type / region
→ 独立 RoBERTa 全量微调副本
→ start/end/mean span pooling
→ 51 类 subtype head
→ predicted-parent hard mask
→ lower RoBERTa LR 由 1e-6 调整为 2e-6
→ FMNERG 输出
```

**训练协议：**
- Model-G: 完整 Train 训练，Dev 选择配置与 checkpoint
- Model-F: gold span + gold parent supervision on Train；end-to-end evaluation 使用 frozen Model-G 的 predicted span + predicted parent
- 两条链都不是 OOF 训练
- F3 仅在 Dev 上完成单变量学习率选择，随后冻结三个 seed
- 一次性 Test 评估（F2 历史 Test 已知；F3 为新的冻结方法 Test）

### 正式结果

| Split | Span F1 | MNER | Fine MNER | EEG | GMNER | FMNERG |
|-------|---------|------|-----------|-----|-------|--------|
| **Dev** | 0.87283 | 0.816714 | 0.68039 ± 0.00297 | 0.660880 | 0.621316 | 0.52052 ± 0.00219 |
| **Test** | 0.86980 | 0.818431 | 0.66510 ± 0.00160 | 0.652157 | 0.615294 | 0.50431 ± 0.00111 |

**状态：** FORMAL

---

## 2. M3.3A 主链阶段结果（FORMAL）

以下均为标准分阶段训练（非 OOF），用于正式 Test 消融。

| 链路阶段 | Dev：MNER / EEG / GMNER | Test：MNER / EEG / GMNER |
|----------|-------------------------|--------------------------|
| RoBERTa Stage1 | 0.81474 / 0.64599 / 0.60733 | 0.81586 / 0.62683 / 0.59168 |
| + Hierarchical Verifier | 0.81671 / 0.65442 / 0.61526 | 0.81843 / 0.64431 / 0.60784 |
| + Coarse Selector + Fine Adapter | 0.81671 / 0.65805 / 0.61889 | 0.81843 / 0.64980 / 0.61333 |
| + Evidence Visibility | **0.81671 / 0.66088 / 0.62132** | **0.81843 / 0.65216 / 0.61529** |

### Coarse Selector 关键指标

Coarse Selector 本身不输出三元组，其有效结果为：

```
R36 unconditional visible coverage = 0.89909
  (分母：所有 visible gold entities)

Base Top-8 + Learned Top-8 conditional recall = 0.90769
  (分母：满足特定 eligibility 条件的 visible gold entities subset*
   *具体定义见 coarse_selector_evaluator.py 的 eligible mask)

原候选保护率 = 1.00000
新增覆盖的 gold entities = 57
丢失的 gold entities = 0
平均候选数 = 15.68
```

**状态：** FORMAL

---

## 3. 严格 OOF 结果（VALID_AUDIT / NO_GO）

| 实验 | 训练特征协议 | 评估协议 | 结果 | 状态 | 结论 |
|------|-------------|---------|------|------|------|
| **M3.3A 10-fold Train-OOF cross-fitted estimate** | 10-fold 整链 OOF，每折 ~700 条 | 7000 条 Train records pooled | Span 0.870900；MNER 0.811690；EEG 0.651135；GMNER 0.610849 | **VALID_AUDIT** | 有效泛化评估参考 |
| Fold-level statistics | 同上 | 10 个 fold 独立计算 | GMNER mean ± std = 0.610869 ± 0.010907 | **VALID_AUDIT** | 折间方差参考 |
| M3.6A-r2 NULL Release | Train strict 10-fold OOF features | Dev full-fit frozen chain | Best epoch 仍为 0，完全 KEEP | **NO_GO** | 不读 Test |
| M3.6A-r1 | Train in-sample/full-fit cache | Dev engineering-only | Dev GMNER 0.623738，净修正 +6 | **ENGINEERING_ONLY** | 仅工程信号，不作为正式提升 |
| 早期 OOF Evidence Graph | 5-fold Stage1 evidence OOF（非整链） | Test | MNER 0.78593；EEG 0.62542；GMNER 0.58035 | **ENGINEERING_HISTORY** | 低于 mBERT Stage1 |

### 关键说明

**10-fold Train-OOF 的两个数字：**
- **Pooled micro GMNER = 0.610849**: 7000 条 OOF predictions 合并后计算
- **Fold mean GMNER = 0.610869**: 10 个 fold F1 的非加权平均

**不与 Dev 0.621316 直接对比的原因：**
- 评估 split 不同（Train OOF vs Dev）
- 每折只使用 90% Train
- 记录难度分布可能不同
- Checkpoint 选择协议不同

**用途：** 作为严格的 Train cross-fit 泛化参考，不是与 Dev 的严格消融对照。

---

## 4. 历史 GMNER 有效对照（ENGINEERING_HISTORY）

这些是早期探索性 Test 结果，早于严格一次性 Test 协议，部分用过 Test 进行探索。

### 使用范围

**✓ 允许：**
- 内部实验历史记录
- 路线追踪和定性说明
- 描述研究演进
- 说明哪些早期路线被放弃

**✗ 禁止：**
- 正式论文 Test 消融主表
- 模型选择的主要证据
- 显著性统计比较
- "我们在 Test 上优于这些消融"的表述

### 历史结果

| 方法 | MNER | EEG | GMNER | 备注 |
|------|------|-----|-------|------|
| mBERT Stage1 | 0.78593 | 0.62582 | 0.58154 | 历史基线 |
| Prototype Gate | 0.79016 | 0.60889 | 0.57319 | 类型提升，联合指标下降 |
| Auxiliary + Contrastive | 0.79189 | 0.60266 | 0.56848 | no-go |
| Prototype Type Refine | 0.79754 | 0.60678 | 0.57069 | no-go |
| External Knowledge | 0.78498 | 0.62530 | 0.58063 | 无净收益 |
| Multiscale Grounding v2 | 0.78593 | 0.62582 | 0.58154 | 输出未改变 |
| Evidence Graph Stable | 0.78593 | 0.62582 | 0.58193 | 仅约 1 个三元组提升 |
| Flat Record Verifier | 0.78709 | 0.62943 | 0.58546 | 被层次化结构取代 |
| mBERT Hierarchical | 0.78951 | 0.63526 | 0.59034 | 有效旧主线 |

**状态：** ENGINEERING_HISTORY

**说明：** 当前正式最优 M3.3A 结果见第 1、2 节，不在此历史表中重复。

---

## 5. FMNERG 有效实验（FORMAL / VALID_DEV / NO_GO）

| 实验 | Dev FMNERG | 状态 | 备注 |
|------|-----------|------|------|
| Frozen F0 subtype head | 0.47167 ± 0.00133 | **VALID_DEV** | 编码器冻结基线 |
| RoBERTa Last-4 | 0.49926 ± 0.00238 | **VALID_DEV** | 有效提升 |
| **RoBERTa All F2** | **0.51729 ± 0.00083** | **FORMAL** | Dev 接受，正式方案 |
| **F2 Final Test** | Test **0.50144 ± 0.00133** | **FORMAL** | 三 seed 一次性 Test |
| **F3 lower-LR ×2** | Dev **0.52052 ± 0.00219** | **FORMAL** | 三 seed paired delta `+0.00323`，预注册 Gate 通过 |
| **F3 Final Test** | Test **0.50431 ± 0.00111** | **FORMAL** | 相对 F2 `+0.00288`；三 seed 均冻结 |
| Class-weighted CE | 0.46723 | **NO_GO** | 低频提升但总体下降 |
| Effective-number CE | 0.46642 | **NO_GO** | |
| Parent-specific head | 0.47490 | **NO_GO** | 低于 shared 0.47719 |
| C1 continued-F2 | 0.517292 | **NO_GO** | Best epoch 0，无收益 |
| J0 fixed-region visual | 0.517292 | **NO_GO** | 相对 C1 增量 0 |
| M3.3F lambda_f=1.0 | 0.50081（B0 为 0.50946） | **NO_GO** | 共享 Stage1 负迁移 |
| M3.3F lambda_f=0.5 | 0.49869 | **NO_GO** | 不进入 R36/Test |

### 训练-推理协议说明

**Model-F (F3 Subtype Sidecar):**
- **Training:** gold span + gold parent supervision on Train
- **Evaluation:** frozen Model-G predicted span + predicted parent on Dev/Test
- **协议类型:** teacher-forced training + predicted-input end-to-end evaluation
- **分布差异:** 存在训练-推理输入分布差异（transparent disclosure）
- **F3 唯一变化:** RoBERTa lower learning rate `1e-6 → 2e-6`

**不变性保证：**
- 三个 seed 的 MNER、EEG、GMNER 均逐记录保持不变
- 只有 Fine MNER 和 FMNERG 发生变化

---

## 6. 有效 Dev-only 诊断（ORACLE / NO_GO）

| 诊断实验 | 训练特征协议 | 评估协议 | 关键结果 | 状态 | 决策 |
|---------|-------------|---------|---------|------|------|
| **P1 Visibility Gold Oracle** | N/A（gold analysis） | Dev full-fit | Gold ceiling +235 entities | **ORACLE** | 理论空间分析 |
| **P1 Observable Visibility Override** | N/A | Dev full-fit | 最佳规则净修正仅 +4，precision 0.50 | **NO_GO** | 后处理 override 关闭 |
| **D0 Stage1 Gradient Conflict Audit** | 固定 Train-only 128-record probe；正式 Stage1 checkpoint；FP32；0 updates | 不访问 Dev/Test；层 0/5/11，58 batches | 9 个 layer/pair 均未满足强负相关与同尺度联合 Gate；`recommend_d2=false` | **VALID_AUDIT** | 不运行 progressive loss schedule，进入 D1 |
| **P2 Span Recovery Gold Oracle** | N/A（gold analysis） | Dev full-fit | R36 GMNER-compatible 218；S1a/S1b/S1c = 134/113/41；理论 ceiling +0.08801* | **ORACLE** | 理论空间大 |
| **P2 Observable Post-processing** | N/A | Dev full-fit | 未通过 Gate | **NO_GO** | 后处理规则关闭 |
| **P2/D1 Learned Selector Seed42** | Train strict 10-fold Stage1-only OOF；旧 NULL Release OOF 特征未复用 | 同候选契约的 full-fit Dev Stage1 cache | Span/MNER/EEG/Stage1-GMNER delta `+0.00430/+0.00570/+0.00166/+0.00256`；formal gold preservation `0.98381`；promoted precision `0.71429` | **NO_GO** | Span 未达 `+0.005` 且保护率低于 `0.99`；不运行 Seeds 41/43、下游重建或 Test；见 [`experiments/ARCHIVED_EXPERIMENTS.md`](experiments/ARCHIVED_EXPERIMENTS.md) |
| **P3 Same-Type Assignment Oracle** | N/A（gold analysis） | Dev full-fit | 24 个唯一可恢复实体；理论上限 +0.00969 | **ORACLE** | Oracle Gate 通过，进入 MVP |
| **P3 C1 Resolver MVP** | Train strict 10-fold OOF features | 同一 full-fit Dev 主链，Resolver disabled/enabled 配对 | Dev 仍 0.621316；0 override、0 corrected、0 damaged | **NO_GO** | 对当前工程链无增益，C2 不运行 |
| SigLIP2 Reliability | N/A | Dev full-fit | VinVL/SigLIP2/Fusion AUROC：0.5773/0.5759/0.6003；Fusion risk +9 | **NO_GO** | |
| M3.5B 动作 Oracle | N/A（gold analysis） | Dev full-fit | Top-1/2/4/8/16 净上限：+233/+326/+372/+403/+411 | **ORACLE** | 候选充足，选择困难 |
| Subtype R36 Oracle | N/A（gold analysis） | Dev full-fit | Top-4 每 seed 仅新增约 5.33 个恢复；Top-4 已等于 R36 上限 | **ORACLE** | 视觉 subtype 路线 no-go |
| Scene-conditioned routing | Train strict 10-fold OOF features；Dev formal predicted entities | Dev only | gold-count 历史结果 1.0000（泄漏，无效）；可部署 predicted-count / OOF classifier 最佳 0.8893，低于 0.95 Gate | **NO_GO** | 停在 Task 1；不接入 decoder，不搜索阈值 |

### P2 理论上限的假设标注

```
*P2 Span Recovery Oracle 的 +0.08801 基于以下假设：
  - 每个候选无损替换一个现有错误 prediction（one-for-one correction）
  - Predicted denominator 不变（fixed denominator）
  - 不考虑 interval decode 竞争关系变化
  - 不考虑现有正确 span 被挤掉
  - 不考虑新增 false positive

实际可实现收益需要通过 record-level decode 重新计算。
这是 fixed-denominator、zero-damage 假设下的理想上限，不是可部署结果的预期。
```

### P2 当前状态

- **Gold Oracle:** 空间大（+0.08801 ceiling）
- **Observable post-processing:** no-go（未通过 Gate）
- **Learned gold-free selector:** Seed42 已完成并判定 `NO_GO`；Seeds 41/43
  与下游重建未运行，紧凑结论见归档索引

---

## 7. 最终可用于论文的核心数字

### GMNER 正式 Test（FORMAL）

```
Model-G (M3.3A):
  MNER  = 0.818431
  EEG   = 0.652157
  GMNER = 0.615294
```

### FMNERG 正式 Test（FORMAL）

```
Model-F (F3 Subtype Sidecar):
  Fine MNER = 0.66510 ± 0.00160
  FMNERG    = 0.50431 ± 0.00111

Paired improvement over frozen F2:
  Fine MNER = +0.00366
  FMNERG    = +0.00288

Test access metadata:
  F3 method access count = 1
  Repository formal access count = 2 (F2 + F3)
  Select best seed on Test = false
```

### 严格 10-fold Train-OOF（VALID_AUDIT）

```
Pooled micro estimate (7000 Train records):
  MNER  = 0.811690
  EEG   = 0.651135
  GMNER = 0.610849

Fold-level mean ± std:
  GMNER = 0.610869 ± 0.010907

不与 Dev 0.621316 直接对比（split/size/protocol 不同）
```

### Oracle 上限（ORACLE）

```
P1 Visibility: +235 gold ceiling, +4 observable best (precision 0.50)
P2 Span Recovery: +0.08801 (under fixed-denominator/zero-damage assumptions)
P3 Same-Type Assignment: +0.00969 (24 entities)
```

---

## 8. 文档索引

- **正式结果来源:** [README.md](../README.md)
- **M3.3A 链路:** [HIERARCHICAL_RECORD_VERIFIER.md](HIERARCHICAL_RECORD_VERIFIER.md)
- **FMNERG F2/F3:** [sidecars/fmnerg_subtype/README.md](../sidecars/fmnerg_subtype/README.md)
- **已关闭实验归档:** [ARCHIVED_EXPERIMENTS.md](experiments/ARCHIVED_EXPERIMENTS.md)
- **实验选择规则:** [EXPERIMENT_ACCEPTANCE_CRITERIA.md](EXPERIMENT_ACCEPTANCE_CRITERIA.md)

---

## 9. 版本信息

- **最后更新:** 2026-07-28
- **Total valid experiments tracked:** 31+
- **正式 Test 协议:** 仅锁定后的 FORMAL Model-G 和 Model-F 结果用于最终 Test 报告
- **历史 Test 访问:** 早期探索性 Test 结果客观存在，均标为 ENGINEERING_HISTORY，不用于当前模型选择、正式消融或显著性结论
- **OOF protocols:** 10-fold Train-OOF cross-fitting for generalization audit

---

**这份总表是内部实验总账、仓库 README 实验总览和论文数字核对的统一事实源。**
