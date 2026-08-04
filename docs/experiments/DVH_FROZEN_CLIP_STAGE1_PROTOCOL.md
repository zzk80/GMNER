# DVH-Stage1 独立全量训练协议

**协议版本**：1.0
**冻结日期**：2026-08-02
**Git 分支**：`codex/dvh-frozen-clip-stage1`
**状态**：`ENGINEERING_IMPLEMENTED_D0_PASSED_SEED42_NOT_STARTED`
**Test 状态**：`LOCKED`

## 0. 执行状态（2026-08-02）

已完成：

```text
独立 DVH record-level 数据合同
原始全图 Frozen CLIP Train/Dev 缓存（7000 / 1500 records）
Word-level Boundary CRF
Span-level Coarse Type Head
VinVL + box-pooled CLIP Grounding
Boundary / Type / Grounding 三个独立视觉 residual 与 gate
独立 denominator 的联合损失
Dev-only checkpoint selection 与严格 GMNER evaluator
优化器全参数唯一分组审计
```

工程验证：

```text
Focused DVH tests: 3 passed
Full repository tests: 151 passed
8 Train / 8 Dev / 1 epoch end-to-end smoke: passed
Frozen CLIP cache: 49 patch tokens x 768 dimensions per image
Old checkpoint used: false
Test accessed: false
```

Smoke 只证明数据、前向、损失、反向传播、解码和指标闭环，不构成方法结果。
正式 Seed42、视觉归因控制、多随机种子、下游重建和 Test 均未启动。

## 1. 实验目标

新路线命名为：

> **DVH-Stage1：Frozen-CLIP Dual-Visual Hierarchical Stage1**

目标是独立训练一套新的 Stage1，以提高 MNER，并保持或提高完整 GMNER：

```text
RoBERTa 文本语义
+ 冻结 CLIP 全局/patch 语义
+ VinVL object region/geometry
-> 分层边界、类型与 Grounding
```

本路线不是 M3.3A 的 adapter、residual、continued training 或 checkpoint
微调。旧正式链仅作为数值基线，不参与新模型前向、损失或初始化。

## 2. 独立性契约

### 2.1 允许使用

```text
官方 RoBERTa-base 预训练权重
官方 CLIP ViT-B/32 预训练权重
原始 Train/Dev 文本、图像与标注
原始 VinVL proposal features、boxes、detector scores/object labels
固定 coarse type mapping：LOC=0, PER=1, ORG=2, OTHER=3
```

RoBERTa 是新 Student 的初始化，训练时允许更新。CLIP 只作为冻结视觉
编码器，整个实验阶段不得更新。

### 2.2 禁止使用

```text
M3.3A Stage1 checkpoint
J1/J2/J3 或 TP-CLIP checkpoint
任何旧 Teacher/KD logits
旧模型 prediction preservation loss
旧正式 span/type/region/NULL 预测
旧 R16/R36 candidate cache
旧 Hierarchical/Coarse/Fine/Evidence checkpoint
Test 数据参与训练、checkpoint 选择或阈值选择
```

### 2.3 可复用数据与不可复用派生产物

VinVL 原始 detector 特征属于冻结输入证据，可以复用。由旧模型预测产生的
候选顺序、formal mask、region logits 和 record decode 均属于派生产物，不能
复用。

## 3. CLIP 冻结合同

CLIP 第一版固定为 `ViT-B/32`，全程满足：

```text
requires_grad = false
model.eval()
无 optimizer parameter group
无 gradient checkpointing
无 CLIP fine-tuning
无 train/dev 不同的图像增强
```

推荐提前物化 Train/Dev 特征：

```text
global embedding
patch token grid
preprocessing fingerprint
model/config SHA256
image ID -> feature row mapping
```

缓存必须使用相同 resize、normalization 和 patch 顺序。训练期间只更新 CLIP
projection、attention、gate 等新模块，不更新 CLIP 本体。

## 4. 模型结构

### 4.1 三条表示路径

```text
Text path:
RoBERTa -> text graph -> token states

Semantic visual path:
Frozen CLIP -> global token + patch tokens -> trainable projection

Object visual path:
VinVL regions + boxes -> region projection -> image graph
```

CLIP 不替换 VinVL：

* CLIP 负责全局场景、局部开放语义和图文语义一致性；
* VinVL 负责对象候选、几何关系和 region/NULL Grounding；
* RoBERTa 保持实体边界与文本类型判断的主路径。

### 4.2 Type-conditioned visual retrieval

为 `PER / LOC / ORG / OTHER` 建立可训练 type query。每个 query 在 CLIP
patch tokens 上检索类型相关视觉证据：

```text
type query + sentence state
-> CLIP patch attention
-> type-specific visual state
```

第一版不使用 subtype query，不使用外部 OCR、caption 或知识检索。

### 4.3 Boundary branch

文本输出基础边界 logits，CLIP 只提供受门控的 token-level residual：

```text
boundary_logits = text_boundary_logits
                + boundary_gate * clip_boundary_delta
```

`boundary_gate` 和 residual 输出层采用零影响初始化，使训练开始时视觉不能
无条件覆盖文本边界。Boundary CRF 输出 word-level `B/I/O` 序列。

视觉边界证据允许关注完整 token 序列，但应单独报告其 corrected/damaged，
不能把类型纠正归入边界收益。

### 4.4 Span type branch

对 Boundary CRF 产生的 span 使用：

```text
span text state
+ type-conditioned CLIP state
+ pooled VinVL context
-> gated span type head
-> PER / LOC / ORG / OTHER
```

训练时使用 gold span 监督类型，并逐步加入 predicted-span replay；正式 Dev
只允许 predicted span，不允许映射到最近 gold span。

### 4.5 Grounding branch

Grounding 以 VinVL region 为候选基础，并加入对应区域内池化的 CLIP patch
语义：

```text
entity span state
<-> VinVL object state + geometry
<-> CLIP box-pooled patch state
-> real region / NULL logits
```

NULL 必须是显式候选。CLIP 全局语义不能直接迫使实体为 visible。

### 4.6 非对称门控

Boundary、Type 和 Grounding 使用三个独立 gate：

```text
token-level boundary gate
span-level type gate
span-region grounding gate
```

禁止共享一个全局 gate 控制三个任务。所有 gate 必须输出使用率、均值和按
正确/错误样本的分布。

## 5. 训练目标

第一版总损失固定为：

```text
L = 1.0 * L_boundary_crf
  + 1.0 * L_coarse_type
  + 1.0 * L_grounding_multi_positive
  + 0.1 * L_alignment
  + 0.01 * L_gate_regularization
```

约束：

* Grounding 使用 multi-positive region supervision；
* 各任务使用独立 denominator；
* 不使用旧模型 KD；
* 不使用 preservation loss；
* 不使用 class-weighted type loss；
* 不在首轮搜索损失权重。

如果梯度尺度差异超过一个数量级，只允许先做 Train-only scaling audit；不得
根据 Dev 反复调整权重。

## 6. 优化器协议

初始设置：

```text
RoBERTa all layers                 3e-6
text/image graph and aligners     1e-5
CLIP/VinVL projections            1e-5
Boundary/Type/Grounding heads     1e-4
Frozen CLIP                       no optimizer group
```

每个 `requires_grad=true` 参数必须且只能属于一个 optimizer group。启动时
输出每组名称、学习率、参数张量数和 trainable element 数。

## 7. 实验阶段

### D0：数据与缓存合同

```text
固定 CLIP preprocessing
构建 Train/Dev global + patch cache
验证 CLIP 完全冻结
验证 VinVL region/box 对齐
验证无旧 checkpoint/cache 来源
不访问 Test
```

### D1：Seed 42 独立 Stage1

从官方 RoBERTa 与 CLIP 初始化，其余模块随机初始化。只按 Stage1 Dev GMNER
选择唯一 best checkpoint，MNER 仅用于 Gate 和 tie-break，不得另选 checkpoint。

### D2：视觉归因控制

使用同一 Seed 42 训练协议比较：

```text
T: RoBERTa only
V: RoBERTa + VinVL
C: RoBERTa + frozen CLIP
D: RoBERTa + VinVL + frozen CLIP
S: D with shuffled image pairing
```

主模型为 D。只有 `D > V` 且 `D > S`，才能归因于配对 CLIP 视觉信息。

### D3：多随机种子

Seed 42 通过后运行预注册 seeds `41/42/43`。不增加新结构，不改变权重和
checkpoint 选择规则。

### D4：下游重建

仅在 D3 通过后授权：

```text
新 Stage1
-> 新 Train/Dev R16
-> 新 R36
-> 重训 Hierarchical Verifier
-> 重训 Coarse Selector
-> 重训 Fine Adapter
-> 重训 Evidence Visibility
```

旧候选 cache 与下游 checkpoint 不得混入。

## 8. 冻结基线与 Gate

Stage1 Dev 冻结基线：

```text
Span F1  = 0.8707208971
MNER F1  = 0.8147402286
EEG F1   = 0.6459927507
GMNER F1 = 0.6073298429
```

Seed 42 筛选 Gate：

```text
MNER delta  >= +0.003
GMNER delta >=  0.000
EEG delta   >= -0.002
D MNER      > V MNER
D MNER      > S MNER
test_accessed = false
```

三 Seed 正式 Gate：

```text
mean MNER delta  >= +0.005
mean GMNER delta >= +0.002
mean Span delta  >= +0.002
mean EEG delta   >= -0.002
至少 2/3 seeds 的 MNER 和 GMNER 同时提升
paired CLIP 相对 shuffled-image 的平均收益为正
test_accessed = false
```

未通过 D3 时，不重建后半链。

## 9. 必须报告的诊断

```text
Span/MNER/EEG/GMNER F1 与 correct count
Boundary corrected / damaged / net
Type corrected / damaged / net
visible 与 NULL grounding corrected / damaged / net
三个 gate 的触发率和分布
PER / LOC / ORG / OTHER 分类型指标
单实体 / 多实体场景指标
paired / shuffled / zero-CLIP 对照
CLIP cache 与 VinVL 对齐失败数
Test accessed 状态
```

## 10. Test 协议

Test 在 D1-D4 全部锁定。只有架构、训练权重、三 seeds、checkpoint 选择规则、
decode 和完整下游链在 Dev 上冻结后，才能一次性运行 Test：

```text
报告三 seed mean +/- std
不按 Test 选择 seed
不根据 Test 修改 threshold 或结构
```

## 11. 当前授权边界

```text
已授权：
协议与数据合同
冻结 CLIP cache 构建
新模型工程实现与单元测试
D0 数据预检

尚未开始：
D1 Seed42 正式训练

未授权：
旧 checkpoint 初始化
CLIP 解冻
旧 R16/R36 复用
下游重建
Test 访问
```
