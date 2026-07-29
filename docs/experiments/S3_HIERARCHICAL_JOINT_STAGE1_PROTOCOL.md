# S3 层次化联合 Stage1 最终实施协议

**状态**：P0 已实现；S3.0 按批准的数值容差修订正式通过；S3.1
工程有效但 Seed42 方法 Gate 为 `NO_GO`。根据 6.8 的预注册规则，
seeds 41/43、S3.2、S3.3/S3.4 和后半链均不运行，Test 保持锁定。

## 实现审计备注

冻结代码存在一个必须区分的旧口径：

```text
entity-expanded GMNERModel.forward：
使用 entity-specific grounding_null_prior

正式 predicted-span Dev decode / candidate cache：
未传入 grounding_null_prior，等效于 neutral prior=0.5
```

S3.0 不修复该旧行为，否则会改变已冻结的 prediction digest。等价 Gate
因此分别验证：

```text
gold entity-expanded grounding：
完整 entity-specific NULL prior 逐阶段等价

formal predicted-span grounding：
按实际旧 decode 使用 neutral prior=0.5，严格复现基线 digest
```

该差异只作为基线实现事实归档，不构成新方法。是否统一 predicted-span
NULL prior 必须另行预注册，当前批次禁止修改。

正式指标还必须沿用旧 evaluator 的论文口径：

```text
EEG / GMNER region correctness：
XML ground-truth box 与预测候选框 IoU 严格大于 0.5

NULL correctness：
实体名称在 XML 中不存在任何 GT box
```

`region_positive_mask` 只用于训练与 grounding 条件诊断。候选集中没有
IoU 合格框时，该 mask 会把 NULL 作为训练目标，但正式 EEG/GMNER 仍将
该实体视为 visible 且不可由当前候选恢复，因此不能用 positive mask
代替正式指标计算。

## 一、批准范围

当前正式批准：

```text
P0：只读数据与错误审计
S3.0：record-level 等价性重构
S3.1：Boundary CRF + Span Type Head
```

有条件批准：

```text
S3.2：Boundary-score-connected Utility Auxiliary
```

暂不批准：

```text
S3.3：Utility 控制正式 decode
S3.4：Teacher KD
动态候选刷新
完整 M3.3A 后半链重建
Test 访问
```

S3.2 只有在 S3.1 完成并通过相应分析后才开发。S3.3、S3.4 不提前实现。

---

# 二、研究目标

当前 Stage1 的主要问题不是候选池召回不足，而是：

```text
typed-BIO 同时承担：
1. 实体边界检测；
2. coarse type 判断；
3. 正式 span 集合解码。
```

D1 已证明候选 utility 中存在一定学习信号，但独立 selector 的 F1 提升主要来自删除预测，而不是恢复缺失实体。

S3 的核心研究假设是：

```text
Boundary 决定实体在哪里；
Span Type 决定实体是什么；
Grounding 决定实体对应哪个图像区域。
```

正式方法贡献限定为：

```text
边界、类型与视觉定位的层次化联合训练
```

以下内容不单独声明为创新：

```text
record-level 数据重构
vectorized grounding
旧公式等价迁移
缓存 Teacher 输出
```

这些属于支撑新方法的必要工程改造。

---

# 三、P0 只读审计

在修改模型前，先运行三个 Train/Dev 审计。

## P0-A：边界错误与类型错误分解

对正式 Stage1 输出分解：

```text
Boundary correct + Type correct
Boundary correct + Type wrong
Boundary wrong + 存在重叠预测
Boundary completely missing
Extra non-gold span
```

同时报告：

```text
PER / LOC / ORG / OTHER
span length
single-token / multi-token
start offset
end offset
overlap F1
```

目的：

```text
确认 Boundary CRF 与 Span Type Head 分离确实对应主要错误来源。
```

## P0-B：截断审计

报告：

```text
超过 max_length=128 的 records 数量
被截断 words 数量
被部分截断 gold entity 数量
被完全截断 gold entity 数量
被截断实体的 coarse type
```

首轮不实现动态 128/256 router。

若被截断实体数量很低，则正式关闭动态长度方向。

## P0-C：候选可行动性

对现有候选池报告：

```text
exact candidate coverage
typed exact candidate coverage
候选 source
候选 rank
teacher score
boundary shift
与正式 span 的 overlap
```

只做审计，不训练 selector。

---

# 四、S3.0：Record-Level Forward and Decode Equivalence

中文名称：

```text
S3.0 Record-level 前向与解码等价性重构
```

## 4.1 核心原则

S3.0 不引入任何新方法：

```text
无 Boundary CRF
无 Span Type Head
无 Utility
无 KD
无动态候选
无新 grounding projection
无新 region representation
```

目标只有一个：

```text
将旧 entity-expanded Stage1
重构成 record-level vectorized Stage1，
并精确复现旧输出。
```

---

## 4.2 新数据单位

每个 dataset item 对应一条原始 record，而不是一个实体。

```python
{
    "record_id": str,

    # 文本
    "input_ids": LongTensor[L_subword],
    "attention_mask": BoolTensor[L_subword],
    "adjacency": FloatTensor[L_subword, L_subword],

    # word/subword 坐标
    "word_count": int,
    "first_subword_indices": LongTensor[L_word],
    "word_to_subword_start": LongTensor[L_word],
    "word_to_subword_end": LongTensor[L_word],
    "subword_to_word": LongTensor[L_subword],

    # 图像区域
    "region_features": FloatTensor[R, 2048],
    "region_boxes": FloatTensor[R, 4],
    "region_mask": BoolTensor[R],
    "region_scores": FloatTensor[R],
    "region_object_labels": list[str],
    "region_object_attributes": list[str],
    "null_region_index": int,
    "region_is_null": BoolTensor[R],

    # NER
    "typed_bio_labels": LongTensor[L_word],

    # 实体，全部使用 word-space 半开区间
    "gold_spans": LongTensor[E, 2],
    "gold_type_ids": LongTensor[E],
    "gold_entity_mask": BoolTensor[E],

    # 映射后的 subword mask
    "gold_subword_masks": BoolTensor[E, L_subword],

    # Grounding
    "gold_region_labels": LongTensor[E],
    "gold_region_positive_mask": BoolTensor[E, R],
    "gold_region_iou_targets": FloatTensor[E, R],

    # 每个实体独立的正式 NULL prior
    "grounding_null_prior": FloatTensor[E],
}
```

Coarse type ID 必须统一从仓库 constants 导入：

```text
LOC   = 0
PER   = 1
ORG   = 2
OTHER = 3
O     = 4
```

文档中的 `PER / LOC / ORG / OTHER` 只表示自然语言展示顺序，
不表示 tensor 中的类别 ID 顺序。禁止在新模块中重新定义类型映射。

---

## 4.3 坐标契约

统一规定：

```text
gold_spans：
word-space [start_word, end_word)

candidate_spans：
word-space [start_word, end_word)

Boundary CRF：
word-level sequence

Span pooling：
通过 word→subword mapping，
在 subword states 上构造 mask

Region：
原始 VinVL region index

NULL：
由 null_region_index 和 region_is_null 显式表示
```

不再隐式假设最后一个区域一定是 NULL。

## 4.4 截断规则

一个实体只有在以下条件满足时才有效：

```text
start_word 和 end_word 均被完整编码；
该实体的所有 words 至少存在一个有效 subword。
```

输出：

```text
gold_entity_mask
grounding_entity_mask
type_entity_mask
```

被截断实体：

```text
不参与 Boundary/Type/Grounding loss；
保留在 protocol diagnostics 中；
不能被静默改成 NULL。
```

---

## 4.5 完整冻结 Teacher

Teacher 定义为完整旧 Stage1：

```text
旧 RoBERTa
+ 旧 Text Graph Encoder
+ 旧 Cross-modal Aligner
+ 旧 typed-BIO CRF
+ 旧 Grounding Head
```

不能只冻结旧 CRF head 并接在更新中的 Student backbone 上。

S3.0 可以采用完整冻结 Teacher 在线推理进行等价测试。

后续 KD 需要的 Teacher 输出采用缓存：

```text
teacher typed-BIO emissions/logits
teacher decoded tags
teacher formal spans
teacher grounding logits
teacher prediction digest
teacher checkpoint SHA256
```

所有 Teacher tensor 均：

```text
requires_grad = false
```

---

## 4.6 Grounding 严格等价向量化

S3.0 不只复现 raw dot-product logits，还必须复现当前正式 Stage1
加入全部 grounding priors 后的最终 score。

当前正式顺序为：

```text
mean-pooled entity state
→ GroundingHead query projection
→ entity-region dot product
→ temperature scaling
→ entity-specific NULL prior
→ global NULL logit bias
→ detector score prior
→ type-region object compatibility prior
→ invalid region masking
```

旧单实体公式：

```python
entity_state = mean(fused_tokens over target subwords)
query = old_grounding_head.proj(entity_state)
logits = query @ image_nodes.T
logits /= old_grounding_head.temperature
```

record-level 版本：

```python
entity_states = masked_mean(
    fused_tokens[:, None, :, :],
    gold_subword_masks[:, :, :, None],
)  # [B, E, H]

queries = old_grounding_head.proj(entity_states)

grounding_logits = torch.einsum(
    "beh,brh->ber",
    queries,
    image_nodes,
)

grounding_logits = (
    grounding_logits
    / old_grounding_head.temperature.clamp_min(1e-4)
)

grounding_logits = grounding_logits.masked_fill(
    ~region_mask[:, None, :],
    -1e4,
)
```

随后必须调用旧 `_apply_grounding_knowledge` 的逐实体向量化等价实现：

```python
formal_logits = apply_record_grounding_knowledge(
    logits=grounding_logits,
    entity_type_ids=gold_type_ids,
    grounding_null_prior=grounding_null_prior,
    region_scores=region_scores,
    region_object_labels=region_object_labels,
    region_object_attributes=region_object_attributes,
    region_mask=region_mask,
    null_region_index=null_region_index,
)
```

`apply_record_grounding_knowledge` 必须保持旧实现的：

```text
prior weight
NULL log-odds 公式
grounding_null_logit_bias
detector score 变换
compatibility function
mask 规则
```

不得继续隐式假设 NULL 一定位于最后一个 region index。

S3.0 禁止：

```text
start/end/mean grounding query
额外 entity projection
额外 region projection
新的 bilinear scorer
新的 temperature
```

## 4.7 S3.0 等价 Gate

等价测试只在以下条件下成立：

```text
model.eval()
dropout disabled
相同 checkpoint
相同 records
相同 tokenization
相同 region features
相同 grounding priors
```

在固定 Dev records 上逐项比较旧 entity-expanded 与新 record-level 输出。

必须满足：

```text
typed-BIO emission max abs error < 1e-6
typed-BIO decoded sequence 完全一致

每个 gold entity：
raw grounding logits max abs error < 3e-5
after-NULL-prior logits max abs error < 3e-5
after-detector-prior logits max abs error < 3e-5
after-compatibility logits max abs error < 3e-5
final masked logits max abs error < 3e-5
grounding argmax 完全一致
NULL/visible decision 完全一致
positive-set correctness 完全一致

Stage1 Span/MNER/EEG/GMNER 完全复现
prediction digest 完全一致
Test accessed = false
```

### S3.0 numerical-equivalence amendment（2026-07-29）

原始预注册 grounding 容差与正式 CUDA 观测为：

```text
Original preregistered grounding tolerance: <1e-5
Observed fully vectorized CUDA max error: 2.288818e-5
Revised tolerance: <3e-5

original_numerical_gate_passed = false
amended_numerical_gate_passed  = true
```

修订原因：

```text
FP32 reduction-order difference between scalar entity execution and
batched CUDA einsum. All discrete decisions, prediction digests,
metrics and correct counts remain exactly equal.
```

该修订仅影响连续 grounding logits 的绝对数值容差。以下 Gate 不放宽：

```text
typed-BIO emissions 与 backbone states <1e-6
region/NULL argmax exact
NULL/visible exact
positive-set correctness exact
prediction set 与 digest exact
Span/MNER/EEG/GMNER 与 correct count exact
test_accessed=false
```

正式容差写入 `s3_stage1_baseline_lock.json`。CLI 只允许在显式
`--diagnostic-only` 模式覆盖容差；该模式的结果不得标记为正式 Gate
通过。

正式 Dev 复跑结果：

```text
evaluated commit:
2f970cf252c40d7eff72448ba1e82aa24ce2d968

original_numerical_gate_passed = false
amended_numerical_gate_passed  = true
formal_gate_eligible           = true
formal_gate_passed             = true
test_accessed                  = false
```

完整原始报告归档于：

```text
docs/experiments/s3_0_forward_equivalence_dev.json
SHA256:
a0627884ce5e6274ac6f1e471fd0f6b8784af14eacfadf26227315a6809a4ae6
```

并检查：

```text
每条 record 只编码一次
NER loss 不再依赖实体展开次数
record-level grounding 与旧实体顺序对齐
NULL index 对齐
```

S3.0 只能证明：

```text
eval 模式下 backbone forward states 等价
typed-BIO emissions 与 CRF decode 等价
raw 及 formal-prior grounding logits 等价
region/NULL decode、prediction set、指标与 digest 等价
```

S3.0 不能证明：

```text
训练过程等价
总 loss 等价
梯度等价
optimizer trajectory 等价
dropout sample 等价
训练 wall-clock 等价
```

原因包括旧系统按 entity-expanded sample 重复计算 NER 和 alignment，
而新系统按 record 计算；新系统中的多实体也共享同一次带 dropout 的编码。
训练 loss 只做诊断性报告，不进入 S3.0 等价 Gate。

S3.0 的正式结论只能写为：

> 新 record-level 实现在 eval 模式下精确复现旧 Stage1 的前向输出、
> 正式 grounding 先验、解码结果和评估指标。该等价性不延伸到训练损失、
> 随机 dropout、梯度或优化轨迹。

S3.0 未通过，不得进入 S3.1。

---

# 五、S3.0 后的梯度审计

完成 record-level loss 归一化后，重新运行 Train-only 梯度审计。

正式流程必须使用同一初始化 checkpoint 的两个独立副本：

```text
副本 A：临时 scaling probe
→ Train-only 100 steps
→ 计算 task gradient norms
→ 推导并冻结 lambda
→ 完全丢弃副本 A

副本 B：正式 S3.1
→ 重新加载原始初始化 checkpoint
→ 重新初始化 optimizer / scheduler / scaler
→ 使用冻结后的 lambda 开始正式训练
```

Probe 固定 Train records、seed、batch 顺序和初始化，不读取 Dev，
不参与模型选择，也不得保存为正式 checkpoint。

## 5.1 审计时点

```text
Student 初始化后、更新前
训练 100 steps 后
第一个 epoch 结束后
```

## 5.2 任务

S3.1 初始审计：

```text
Boundary
Type
Grounding
Alignment
```

## 5.3 参数区域

```text
RoBERTa layer 0
RoBERTa layer 5
RoBERTa layer 11
Cross-modal Aligner
```

## 5.4 先检查 loss denominator

每个 loss 必须分别按以下单位归一：

```text
Boundary：有效 words
Type：有效 gold entities
Grounding：有效 gold entities
Alignment：有效 records
```

若某个任务梯度仍比其他任务高两个数量级以上，先确认：

```text
mask
denominator
重复样本
异常 logits
```

不存在实现问题后，再确定静态 loss scaling。

## 5.5 静态 scaling 协议

不执行 Dev 网格搜索。

在固定 Train-only probe 上计算每项任务的 median gradient norm：

```text
g_boundary
g_type
g_grounding
g_alignment
```

以 Boundary 为参考：

```python
lambda_i = clip(
    g_boundary / max(g_i, eps),
    min=0.05,
    max=20.0,
)
```

对多个审计层使用 log-norm ratio 的中位数。

得到的 lambda：

```text
写入配置
训练期间固定
不得根据 Dev 再调整
```

Probe 完成后禁止在其 100-step 参数上继续训练。正式训练必须从原始初始化
重新开始，并记录初始化 checkpoint 与最终 lambda 的 SHA256。

应用静态 lambda 后，在以下时点再次审计：

```text
正式 step 100
正式 epoch 1 结束
最佳 checkpoint
```

输出：

```text
raw gradient norms
raw norm ratios
derived lambda
weighted gradient norms
weighted norm ratios
clipping 是否触发
```

若关键共享层的加权任务梯度仍持续满足：

```text
max_norm / min_norm >= 100
```

只能记录为“静态 scaling 未充分解决尺度失衡”。不得在同一次实验中临时
修改 lambda；动态 balancing 必须另行预注册。

方向冲突与尺度失衡分开解释：

```text
强方向冲突：
才讨论 PCGrad 类方法

纯尺度失衡：
只使用预注册静态 scaling
```

首轮不使用 PCGrad、GradNorm 或 adversarial training。

---

# 六、S3.1：核心方法

## 6.1 架构

```text
Student record-level backbone
        │
        ├── Word-level Boundary CRF
        ├── Span-level Coarse Type Head
        ├── Legacy-equivalent Vectorized Grounding
        └── Alignment Objective
```

S3.1 不包含：

```text
Utility
Teacher KD
动态候选
hard selector
新视觉模块
```

---

## 6.2 Word-level Boundary CRF

Student fused subword states 先映射为 word states：

```python
word_states = gather(
    fused_subword_states,
    first_subword_indices,
)
```

第一版直接使用每个 word 的 first-subword state，以保持与旧标签位置一致。

输出标签：

```text
O
B
I
```

合法转移：

```text
O → O / B
B → O / B / I
I → O / B / I
```

禁止：

```text
O → I
```

训练标签由旧 coarse typed-BIO 去掉类型得到。

---

## 6.3 Span Type Head

输入使用 Student 当前 span representation：

```text
first subword
last subword
all-span-subword mean
```

结构：

```text
LayerNorm(3H)
Linear(3H, H)
GELU
Dropout
Linear(H, 4)
```

输出：

```text
PER / LOC / ORG / OTHER
```

不包含 O 类。

实现中的 tensor ID 顺序必须使用：

```text
LOC=0, PER=1, ORG=2, OTHER=3
```

不得按上述自然语言展示顺序重新编码。

训练：

```text
只在有效 gold spans 上监督
```

评估：

```text
Boundary CRF predicted spans
→ Span Type Head
```

必须额外报告：

```text
gold-span type accuracy
predicted-span type accuracy
newly recovered span type accuracy
legacy-preserved span type accuracy
```

---

## 6.4 Grounding

保持 S3.0 的旧公式等价 vectorized grounding。

训练使用有效 gold spans。

评估使用：

```text
Boundary predicted span
→ subword mask
→ old-formula mean pooling
→ grounding
```

必须报告训练—推理 span 分布差异：

```text
gold-span grounding
predicted-span grounding
newly recovered-span grounding
boundary-shift-span grounding
```

首轮不为 predicted-span exposure gap增加新 loss。

---

## 6.5 S3.1 Loss

```text
L =
lambda_boundary  * L_boundary
+ lambda_type    * L_type
+ lambda_ground  * L_grounding
+ lambda_align   * L_alignment
```

lambda 使用 S3.0 后的 Train-only 静态梯度审计结果。

不能默认：

```text
1 : 1 : 1 : 0.1
```

就是中性配置。

---

## 6.6 Checkpoint 选择

唯一 checkpoint 主选择指标：

```text
Stage1 Dev GMNER
```

不得按多个指标来回选择不同 checkpoint。

对 Stage1 GMNER 最优 checkpoint，再一次性检查 Gate：

```text
Span F1
MNER
EEG
Stage1 GMNER
正确 span 数
正确 triple 数
formal preservation
candidate coverage
```

Early stopping 同样只监控：

```text
Stage1 Dev GMNER
```

---

## 6.7 S3.1 Seed42 Gate

必须同时满足：

```text
Span F1 delta              >= +0.005
MNER delta                 >= +0.003
Stage1 GMNER delta         >= +0.003
EEG delta                  >= -0.002

正确 span 数不得下降
正确 GMNER triple 数不得下降
formal gold preservation   >= 0.99
R16 coverage delta         >= -0.002

Test accessed              = false
```

同时输出：

```text
Boundary corrected / damaged
Type corrected / damaged
新增 span 数量
删除 span 数量
新增 span 的 type accuracy
新增 span 的 grounding correctness
各 coarse type 指标
span length 分层
```

## 6.8 S3.1 决策

### 完全通过 Gate

```text
运行 seeds 41 / 43
```

### 指标有正向信号，但未完全通过

例如：

```text
Span/MNER 提升
correct count 不下降
但 GMNER delta < +0.003
```

状态：

```text
VALID_DEV
```

可以评估是否值得进入 S3.2，但不能重建下游。

### correct count 下降或 preservation 失败

状态：

```text
NO_GO
```

不运行 S3.2。

### Seed42 正式结果

修复优化器分组后，正式 run 按 Dev GMNER 选择 epoch 1：

```text
Span delta              = +0.0012807
MNER delta              = +0.0008211
EEG delta               = -0.0057992
Stage1 GMNER delta      = -0.0038225
correct span delta      = +1
correct GMNER delta     = -11
formal preservation    = 0.9529178
R16 coverage delta     = 0
Test accessed           = false
```

冻结基线及所有来源检查精确通过，因此该结果归类为方法 `NO_GO`，
不是工程失败。按照本节规则，不运行 seeds 41/43 或 S3.2。

---

# 七、S3.2：有条件 Utility Auxiliary

S3.2 在 S3.1 完成前不开发。

## 7.1 进入条件

至少满足：

```text
S3.1 correct span count 不下降
formal preservation >= 0.99
Boundary CRF 对 promotable spans 存在可测信号
候选池 exact coverage 仍充分
```

## 7.2 固定候选集

使用 S3.1 最佳 Seed42 checkpoint，对 Train 和 Dev 各生成一次固定候选 manifest。

候选来源：

```text
S3.1 Boundary 1-best
S3.1 Boundary n-best
完整冻结 Teacher spans
±1 word boundary perturbation
```

候选集合生成后固定。

首轮不进行：

```text
每 epoch 刷新
在线改变离散 candidate indices
动态 manifest
```

## 7.3 Boundary span score

不能使用未归一化整句 CRF sequence score。

为候选 span `s=[l,r)` 定义：

```text
n = r - l
n >= 1
```

Entity path score：

```text
entity(s) = e_l(B),                                      n = 1

entity(s) = e_l(B)
            + sum(e_t(I), t=l+1...r-1)
            + transition(B,I)
            + (n-2) * transition(I,I),                   n > 1
```

单词实体不包含 `B→I` transition。

Outside path score：

```text
outside(s) =
    sum(e_t(O), t=l...r-1)
    + (n-1) * transition(O,O)
```

最终可微 Boundary span score：

```text
b(s) = (entity(s) - outside(s)) / sqrt(n)
```

其中：

```text
e_t(label)：Boundary emission
τ：Boundary CRF transition
```

与同一区间全 `O` 路径作差，使分数对 emission 的公共平移更稳定；
长度归一化降低长 span 的分数偏置。

该定义：

```text
与候选所在的其他 n-best 完整序列无关
确定性
可反向传播到 Boundary emissions
可反向传播到 CRF transitions
```

## 7.4 Utility 连接

```python
utility_score = (
    boundary_span_score
    + residual_scale * torch.tanh(residual_delta)
)
```

其中：

```text
candidate indices：stop-gradient
boundary_span_score：允许反向传播
residual_delta：允许反向传播
```

Residual 最后一层可以零初始化，因为 boundary span score 从第一步开始直接提供梯度。

## 7.5 S3.2 只做 auxiliary

正式预测仍使用：

```text
Boundary CRF 1-best
→ Span Type Head
→ Grounding
```

Utility 不控制正式 decode。

Utility loss直接约束：

```text
正确 non-formal span > overlapping wrong span
正确 formal span > overlapping wrong span
wrong non-formal span 保持负监督
```

## 7.6 S3.2 Gate

除完整 Stage1 指标外，还必须报告：

```text
utility AUROC / AUPRC
promotable positive recall
wrong non-formal false-positive rate
corrected vs damaged ranking pairs
boundary emissions 的变化
```

只有同时满足：

```text
Stage1 GMNER 提升
正确 span 数不下降
正确 triple 数不下降
Utility 在可观察候选上具有稳定分离性
```

才讨论 S3.3。

---

# 八、S3.3 与 S3.4 的授权条件

## S3.3 Utility Decode

只有 S3.2 已证明：

```text
Utility 可分离
并且 auxiliary 已改善正式 Boundary 输出
```

才允许设计一次固定 hard-decode 实验。

不得预先实现 threshold scan。

## S3.4 Teacher KD

只有出现：

```text
新模型具有明确 correction 信号
但损伤集中在旧 Teacher 高置信正确样本
```

才允许使用缓存 Teacher 输出进行一次固定 KD 实验。

KD Teacher 必须是：

```text
完整冻结旧 Stage1
或其不可变缓存输出
```

不能只冻结旧 CRF head。

---

# 九、三 Seed 与下游重建

S3.1 或后续获准版本通过 Seed42 Gate 后运行：

```text
41 / 42 / 43
```

三 Seed Gate：

```text
至少 2/3 seed Stage1 GMNER 提升
mean Stage1 GMNER delta >= +0.003
mean MNER delta >= 0
mean EEG delta >= -0.002
mean correct span count 不下降
mean correct triple count 不下降
Test 全程锁定
```

通过后才允许：

```text
new R16
→ new R36
→ retrain Hierarchical Verifier
→ retrain Coarse Selector
→ retrain Fine Adapter
→ retrain Evidence Visibility
```

第一轮完整链只运行：

```text
Seed42 Dev
```

Gate：

```text
Full-chain Dev GMNER >= 0.624316
MNER delta >= 0
EEG delta >= -0.002
Test accessed = false
```

---

# 十、最终执行顺序

```text
1. 冻结 M3.3A、F3、D0、D1

2. P0-A 边界/类型错误审计
3. P0-B 截断审计
4. P0-C 候选可行动性审计

5. 实现 word/subword 映射契约
6. 实现 record-level dataset/collator
7. 实现旧 grounding 的严格 vectorization
8. 构建完整冻结 Teacher
9. 完成 S3.0 等价 Gate

10. 对新 record-level loss 运行 Train-only 梯度审计
11. 固定静态 loss scaling

12. 实现 word-level Boundary CRF
13. 实现 Span Type Head
14. 完成 S3.1 Seed42

15. 根据 S3.1 结果决定是否开发 S3.2

16. S3.1/S3.2 通过后运行 seeds 41/43

17. 三 Seed Gate 通过后重建 M3.3A 后半链

18. 新 Model-G 正式锁定后再重建 F3 successor

19. 所有配置、checkpoint、decode 和 seed 冻结后访问 Test
```

---

# 十一、最终项目状态

```text
S3.0：APPROVED
S3.1：APPROVED
S3.2：CONDITIONAL
S3.3：NOT AUTHORIZED
S3.4：NOT AUTHORIZED
Downstream rebuild：NOT AUTHORIZED
Test：LOCKED
```
