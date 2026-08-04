# Stage1 三套体系方法与结果对比

**范围**：当前正式最优 Model-G（M3.3A），以及两次独立 Stage1 重构
DVH-Stage1、TQ-DV-MNER。

---

## 结论摘要

当前最优仍是 M3.3A：

```text
RoBERTa Stage1
-> R16 formal candidates
-> R36 expanded regions
-> Hierarchical Record Verifier
-> Base Top-8 + Learned Top-8 Coarse Selector
-> Fine Grounding Adapter
-> Evidence Visibility
-> Record-level Decode
```

两次失败实验验证了两个不同假设：

```text
DVH-Stage1：
将边界、类型、Grounding 解耦，并让 Frozen CLIP 同时辅助三条分支。
结果：Grounding尚可，但先边界后类型的分解损失了大量正确span。

TQ-DV-MNER：
让四个类型查询在边界形成前参与抽取，并联合解码typed spans。
结果：条件类型准确率提高，但新的span生成器仍损失更多正确边界。
```

因此，现有证据不支持继续从零替换正式 Typed-BIO CRF。更合理的待验证方向是：

```text
保留正式 Typed-BIO CRF 的span
+
使用TQ-DV对固定span重新计算type分数
```

---



## 核心结果对比

### Dev 主结果

| 体系 | Span F1 | MNER F1 | EEG F1 | GMNER F1 | 状态 |
|---|---:|---:|---:|---:|---|
| 正式 RoBERTa Stage1 bypass | 0.870721 | 0.814740 | 0.645993 | 0.607330 | 有效基线 |
| **完整 M3.3A** | **0.872830** | **0.816714** | **0.660880** | **0.621316** | **FORMAL** |
| DVH 正式归档 checkpoint | 0.851785 | 0.799355 | 未完整归档 | 未完整归档 | NO_GO |
| DVH epoch 20 诊断 | 0.854314 | 0.800970 | 0.651041 | 0.612245 | 诊断项，非正式最优 |
| TQ-DV Seed42 | 0.852346 | 0.810275 | 不适用 | 不适用 | NO_GO |

说明：

- DVH 正式比较 checkpoint 的 Span/MNER 来自后续 TQ-DV 协议中的冻结归档记录。
- DVH epoch 20 是训练历史项；DVH 按 Dev GMNER 选择 checkpoint，因此不能用
  epoch 20 的单项最大值替代正式 checkpoint。
- TQ-DV 本阶段只训练 MNER，不训练正式 Grounding，因此不能报告可比较的
  EEG/GMNER，也不能复用旧 R16/R36 下游结果。

### 正式 M3.3A Test

| Span F1 | MNER F1 | EEG F1 | GMNER F1 |
|---:|---:|---:|---:|
| 0.869800 | 0.818431 | 0.652157 | 0.615294 |

DVH 和 TQ-DV 未访问 Test，不能与该行构成 Test 对照。

### 正确数量和误差传递

| 体系 | 预测数 | Span correct | MNER correct | 正确span条件类型准确率 |
|---|---:|---:|---:|---:|
| 正式 Stage1 | 2516 | 2162 | 2023 | 2023/2162 = 0.9357 |
| DVH epoch 20 | 2499 | 2114 | 1982 | 0.9376 |
| TQ-DV Seed42 | 未统一归档 | 2107 | 2003 | 0.9506 |

相对正式 Stage1：

```text
DVH epoch 20：Span correct -48，MNER correct -41
TQ-DV Seed42：Span correct -55，MNER correct -20
```

DVH 只带来很小的条件类型改善，无法抵消边界损失。TQ-DV 的类型判断改善更明显，
但仍不足以抵消 55 个正确span的损失。

---

## 统一讨论实例

```text
Tweet:
Jordan met Nike executives in Paris after the game.

Gold entities:
Jordan  -> PER -> person region r1
Nike    -> ORG -> logo/building region r3
Paris   -> LOC -> landmark region r5

Image candidates:
r1  basketball player A
r2  basketball player B
r3  Nike logo
r4  generic building
r5  Paris landmark
r6  crowd/background
NULL
```

这个例子包含三类困难：

1. `Jordan` 可能是人名，也可能被理解成地点或品牌，类型需要上下文。
2. 图中有两个 person region，一般场景语义不能确定哪个人对应 `Jordan`。
3. `Nike` 的视觉证据可能是 logo，也可能没有直接可见实体，需要处理 region/NULL。

三套方法的核心区别可以先概括为：

```text
正式 Stage1：
一次性产生全句 typed-BIO 序列，再对已解码实体做 Grounding。

DVH：
先产生无类型 B/I/O 边界，再对每个固定span做类型和Grounding。

TQ-DV：
分别问四次“这里有没有LOC/PER/ORG/OTHER”，再联合选择typed spans。
```

讨论时应始终区分三个问题：

```text
Boundary：实体在哪里开始和结束？
Type：这个span属于哪一个coarse type？
Grounding：它对应哪个region，还是NULL？
```

MNER 只要求前两个问题同时正确；GMNER 要求三个问题全部正确。

---

## 体系一：正式 M3.3A

### 训练协议

```text
训练：完整Train训练，各阶段在Dev选择checkpoint/config
OOF：正式Dev/Test链不使用OOF训练
Test：架构冻结后一次性正式评估
视觉输入：VinVL R16/R36；不使用CLIP
```

十折 full-chain OOF 只用于训练分布审计和部分后处理实验，不是正式 M3.3A
Dev/Test 模型的训练方式。

### Stage1 文本与图像编码

```text
文本：
RoBERTa-base
-> 3层依存/窗口文本图

图像：
VinVL 2048维region features
-> 768维投影
-> 2层区域图

跨模态：
token作为query，region作为key/value
-> 8-head cross attention
-> 融合token表示
```

正式 Stage1 边界并非纯文本；VinVL region attention 在 NER 解码前已经进入融合
token 表示。但它属于通用 token-region attention，不是边界专用或类型专用检索。

#### 张量与信息流

设文本长度为 `L`，真实区域数为 `R=16`，隐藏维度为 `d=768`：

```text
RoBERTa token states       H_bert  [L,768]
Text graph states          H_text  [L,768]
VinVL raw regions          X_img   [16,2048]
Projected region states    V       [16,768]
Cross-modal token states   H_fused [L,768]
Typed-BIO emissions        E       [L,9]
```

文本图使用依存关系和局部窗口邻接矩阵：

```text
H_text^(k+1)
= LayerNorm(
    H_text^k
    + Dropout(GELU(A_text H_text^k W_k))
  )
```

VinVL 区域先投影到 768 维，再通过区域关系图传播。跨模态层让每个 token 查询所有
有效区域：

```text
H_visual = MultiHeadAttention(
    query = H_text,
    key   = V,
    value = V
)

H_fused = LayerNorm(H_text + H_visual)
```

这里的视觉信息可以帮助判断 `Jordan` 与 person region、`Nike` 与 logo region
相关，但所有类型共享同一个 token-to-region attention。模型没有显式执行：

```text
PER query -> person regions
ORG query -> logo/building regions
```

因此它更像通用视觉上下文化，而不是类型专属检索。

### Typed-BIO CRF

标签空间：

```text
O
B/I-LOC
B/I-PER
B/I-ORG
B/I-OTHER
```

CRF 用一个序列同时决定边界和类型：

```text
P(span,type | text,image)
= Typed-BIO sequence probability
```

优点是边界、类型和合法 B/I 转移共同解码，具有很强的序列归纳偏置。缺点是类型
和边界难以独立修正。

#### CRF 的数学含义

对标签序列 `y=(y_1,...,y_L)`，CRF 的序列分数为：

```text
Score(y)
= sum_i emission(i,y_i)
 + sum_i transition(y_(i-1),y_i)
```

最终使用 Viterbi 求：

```text
y* = argmax_y Score(y)
```

所以 `B-PER I-PER` 不是两个独立 token 分类，而是一个全句最优序列的一部分。
例如：

```text
Jordan met Nike executives in Paris
B-PER  O  B-ORG      O       O B-LOC
```

如果 `Nike` 的 `B-ORG` emission 只略高于 `O`，CRF 仍可利用前后标签转移与整句
结构保留它。反过来，若把 Boundary 和 Type 完全拆开，这种联合约束就会消失。

#### Typed-BIO 的优势和限制

优势：

```text
边界与类型共享同一token证据
B/I转移合法性由CRF保证
不同类型候选在同一个全句序列中竞争
```

限制：

```text
类型证据变化可能同时改变边界
无法只把B-PER改成B-ORG而绝对保护span
对“边界正确但类型错误”的定向修正不方便
```

### Stage1 Grounding

对每个实体span池化融合token，得到实体表示，与 R16 区域做点积，并依次加入：

```text
raw entity-region score
-> entity NULL prior
-> global NULL bias
-> detector prior
-> type-object compatibility
-> invalid-region mask
-> 16 real regions / NULL argmax
```

训练时 Grounding 使用 gold span/type，推理时使用 predicted span/type，因此边界和
类型错误会继续传播到 EEG/GMNER。

#### Grounding 分数展开

对实体 `e`，先在其 subword mask 上池化：

```text
h_e = masked_mean(H_fused, entity_mask_e)
```

基础区域分数近似为：

```text
s_raw(e,r) = dot(W_e h_e, v_r) / temperature
```

正式分数继续加入可部署先验：

```text
s(e,r)
= s_raw(e,r)
 + 0.1 * log(detector_score_r)
 + 0.2 * compatibility(type_e, object_r)
 + NULL priors
```

在统一实例中：

```text
Jordan/PER：r1、r2因person compatibility同时升分
Nike/ORG： r3因logo/object语义升分
Paris/LOC：r5因location/building compatibility升分
```

这也说明类型错误为何会传递到 Grounding。若 `Nike` 被预测为 OTHER，区域兼容性
和实体表示都会改变，即使正确 logo 区域已经在 R16 中，也可能不再成为 top-1。

#### 训练与推理条件差异

```text
训练 Grounding：gold span + gold type -> region/NULL
推理 Grounding：predicted span + predicted type -> region/NULL
```

这是正式 Stage1 的 exposure gap。训练时实体池化边界准确，推理时任何边界偏移
都会改变 `h_e`。因此不能只看 gold-span grounding accuracy 判断正式 GMNER。

### M3.3A 后半链

#### Hierarchical Record Verifier

冻结 Stage1 的候选表示，分层判断 entityness、Visibility 和区域残差。它能够删除
部分错误实体，并修正 NULL/visible 和区域，但不生成新的正式 span/type。

其方法论不是一个平坦分类器，而是将决策拆开：

```text
Stage1 span候选
-> Entityness：该span是否应保留
-> Visibility：应为真实区域还是NULL
-> Region：若visible，应选择哪个真实区域
```

例如 Stage1 错误输出一个 `game/OTHER`，Verifier 可以将其删除；但若 Stage1 漏掉
`Nike`，后半链不能凭空创建该span。

#### Coarse Selector

R36 中使用：

```text
Stage1 Base Top-8
+ Learned non-duplicate Top-8
```

Dev 条件召回为 `0.90769`，原候选保护率为 `1.00000`，新增覆盖 57 个 gold
entities，平均保留 15.68 个区域。

#### Fine Grounding Adapter

只对 frozen Top-16 真实区域进行 correction-preservation 重排，不改变 span、type
和基线 NULL 实体。Dev 将 GMNER 从 `0.615260` 提高到 `0.618894`。

它同时优化两类相反目标：

```text
Correction：base top-1错误且gold在候选中时，提升gold region
Preservation：base top-1正确时，限制residual破坏原排序
```

在统一实例中，若 `Jordan` 的 r2 原始分数高于正确 r1，Fine Adapter 可以利用
实体、区域、几何和候选来源特征将 r1 提升；但它不能把 `Jordan/PER` 改为其他
类型。

#### Evidence Visibility

使用 Fine 概率、margin、entropy、候选来源、detector score、agreement 和
type-object compatibility 等部署期证据，对冻结 Visibility logit 增加有界残差：

```text
l_final = l_hierarchy + 4 * tanh(delta_visibility)
```

继续使用 `0.80/0.20` 非对称阈值，最终 Dev GMNER 达到 `0.621316`。

两个阈值形成带滞回的保守决策：

```text
原来为NULL：p_visible >= 0.80 才释放为visible
原来可见： p_visible <= 0.20 才回退为NULL
中间区间：保持原状态
```

这避免了在 `Nike` 图像证据模糊时，仅因概率从 `0.49` 变成 `0.51` 就翻转输出。

#### M3.3A 统一实例完整推理

```text
1. Typed-BIO CRF：
   Jordan/PER, Nike/ORG, Paris/LOC

2. Stage1 R16 Grounding：
   Jordan -> r2（错误人物）
   Nike   -> NULL
   Paris  -> r5（正确）

3. Hierarchical Verifier：
   保留三个span/type；判断Jordan和Paris visible，Nike证据不确定

4. R36 Coarse Selector：
   为Jordan保留base候选，并补入learned候选r1

5. Fine Adapter：
   Jordan r2 -> r1

6. Evidence Visibility：
   若Nike的fine top-1与多项证据一致且p_visible>=0.80，NULL -> r3；
   否则安全保持NULL
```

该例体现了 M3.3A 的设计原则：后半链只在明确负责的维度上修正，不重新打开
所有决策。

### 为什么它仍是最优链

M3.3A 没有显著改进 Stage1 的类型能力，而是保护已有 MNER，并逐层解决：

```text
错误实体过滤
R36候选覆盖
真实区域细排
NULL/visible协调
```

Stage1 到最终链的 MNER correct 都是 2023；MNER F1 的小幅提升主要来自预测数
`2516 -> 2504`，而 EEG/GMNER 的主要收益来自后半链。

---

## 体系二：DVH-Stage1

### 目标与训练协议

DVH 是一套完全独立的新 Stage1：

```text
初始化：官方 RoBERTa + 官方 Frozen CLIP ViT-B/32
旧checkpoint：不使用
RoBERTa：可训练
CLIP：编码器不进入模型，只读取冻结缓存
VinVL：冻结region输入
checkpoint：只按Dev Stage1 GMNER选择
OOF：不使用
Test：锁定
```

CLIP cache 每张图包含一个 global token 和 `7x7=49` 个 patch tokens，维度为
768。VinVL 提供 R16 目标框、区域外观、bbox、detector 和 object label。

### 三分支体系

```text
Trainable RoBERTa word states
├── Boundary CRF
├── Span Type Head
└── Region/NULL Grounding

Frozen CLIP + VinVL
├── boundary visual residual
├── type visual residual
└── grounding visual residual
```

三条视觉残差都有独立 gate，末层零初始化，避免视觉在初始化时覆盖文本。

#### DVH 的方法假设

DVH 假设原 Typed-BIO 把两个不同问题绑得太紧：

```text
问题A：哪些连续token构成实体？
问题B：实体属于LOC/PER/ORG/OTHER中的哪一类？
```

因此它采用层次化分解：

```text
Sentence
-> type-agnostic Boundary
-> span representation
-> coarse Type
-> region/NULL Grounding
```

理论优势是每一层可使用不同视觉证据：

```text
Boundary：物体/文字区域是否支持“这里存在实体”
Type：人物、地点、组织等类别语义
Grounding：具体实体与具体候选框的匹配
```

但这一分解隐含了一个强条件：Boundary 必须先有足够高的召回率，因为后两层只能
处理已经存在的span。

#### 双视觉证据为什么不等于重复输入

DVH 同时保留 VinVL 与 Frozen CLIP，因为二者承担不同职责：

| 视觉来源 | 主要信息 | 主要用途 |
|---|---|---|
| CLIP global | 全图场景与图文语义 | 判断推文与图像是否整体相关 |
| CLIP patches | 粗粒度局部开放语义 | 为边界和类型提供视觉 residual |
| VinVL regions | 对象候选与局部外观 | 正式 Grounding 候选 |
| VinVL boxes | 位置、大小、重叠关系 | 区域图与候选几何 |
| detector labels/scores | person/logo/building 等先验 | 类型兼容性与候选置信度 |

CLIP ViT-B/32 在 224 输入下只有 `7x7` patch 网格。它适合提供“这里大致有人或
建筑”的语义，但未必能对齐 `Jordan` 这样的精确 token 边界，也不能可靠区分图中
两个同类人物实例。

### Boundary 分支

DVH 首先预测与类型无关的：

```text
B / I / O
```

CLIP patch 通过 word-to-patch attention 产生 token-level residual：

```text
boundary_logits
= text_boundary_logits
 + boundary_gate * clip_boundary_delta
```

然后由 WordBoundaryCRF 固定最终span。

#### Boundary Gate 的行为

对 word `i`：

```text
delta_i, gate_i = VisualResidual(h_i, clip_context_i)
emission_i = text_emission_i + gate_i * delta_i
```

`gate_i` 不是硬开关，而是 `[0,1]` 的连续权重。残差末层零初始化保证 epoch 0
视觉增量为0，但独立训练后它可以逐步改变 B/I/O emission。

这个保护只保证初始化不被视觉覆盖，不保证训练后保留正式 Stage1 的边界。DVH
没有旧模型 KD，也没有 formal-span preservation loss，因此训练目标允许它重构整套
span集合。

### Type 分支

对已经由 Boundary CRF 固定的span，融合：

```text
span text state
+ type-conditioned CLIP state
+ pooled VinVL context
-> LOC/PER/ORG/OTHER
```

训练使用 gold span 类型监督，正式 Dev 使用 predicted span，不进行 gold 对齐。

Span Type Head 的训练与部署存在条件差异：

```text
Train： gold span -> type head
Dev：   boundary-predicted span -> type head
```

训练时 Type Head 总能看到完整的 `Golden Gate Bridge`；推理时如果 Boundary 只
产生 `Golden Gate`，Type Head 即使正确预测 LOC，也只能输出错误span上的 LOC。
对 MNER 而言仍然是错误。

### Grounding 分支

候选仍为 VinVL R16 + NULL。每个区域使用：

```text
VinVL object state
+ bbox geometry
+ box-pooled CLIP patch state
```

实体与区域打分后，再加入独立 grounding visual residual。NULL 是显式候选；CLIP
全局语义不能直接强制实体为 visible。

CLIP patch 会根据 VinVL bbox 做 box pooling：

```text
CLIP patch grid
-> 找到中心落入region bbox的patch
-> masked mean
-> box-level CLIP state
```

这是一种近似的局部语义对齐。由于 patch 网格较粗，小框、相邻人物和高重叠区域
可能共享大量 patch，无法提供精确实例身份。

### 损失

```text
L = 1.0 * L_boundary_crf
  + 1.0 * L_coarse_type
  + 1.0 * L_grounding_multi_positive
  + 0.1 * L_alignment
  + 0.01 * L_gate_regularization
```

各任务使用独立 denominator，但共享可训练 RoBERTa，因此三个任务仍会共同改变
文本主干。

独立 denominator 的含义是：

```text
Boundary loss  / 有效word数
Type loss      / 有效实体数
Grounding loss / 有region监督的实体数
```

它防止样本数量差异直接决定 loss 大小，但不能保证三个任务对共享 RoBERTa 的梯度
规模或方向始终平衡。

#### DVH 统一实例推理

```text
Tweet: Jordan met Nike executives in Paris after the game.

1. Boundary CRF：
   [Jordan], [Nike executives], [Paris]

2. Type Head：
   Jordan          -> PER
   Nike executives -> ORG
   Paris           -> LOC

3. Grounding：
   Jordan          -> r1
   Nike executives -> r3
   Paris           -> r5
```

虽然三个类型和区域都可能正确，但 `Nike executives` 的span比 gold `Nike` 多一个
token，所以：

```text
Span：错误
MNER：错误
EEG：错误
GMNER：错误
```

Type 和 Grounding 后续再准确，也不能弥补 Boundary 的一次偏移。

### 失败原因

其概率分解是：

```text
P(span) * P(type | span) * P(region | span,type)
```

类型视觉证据只能在 Boundary CRF 已经产生span之后介入。漏检和错误边界不能由
Type Head 回救。最终出现：

```text
条件类型准确率略升
但Span correct明显下降
因此MNER correct下降
```

这否定的是“先做无类型边界、再做视觉类型分类”这一具体分解，不是否定 Frozen
CLIP 的所有用途。

更严格地说，DVH 失败说明：

```text
Boundary召回/精度损失
>
独立Type Head与双视觉Grounding带来的收益
```

它没有证明以下命题：

```text
CLIP对固定正确span的类型判断无效
CLIP对已有region候选的语义重排无效
所有Boundary/Type解耦都必然失败
```

---

## 体系三：TQ-DV-MNER

### 目标与训练协议

TQ-DV 针对 DVH 的边界先行问题，改为在抽取前引入类型条件：

```text
初始化：官方 RoBERTa；不使用旧GMNER checkpoint
CLIP：冻结缓存
VinVL：R16冻结region输入
checkpoint：只按Dev MNER选择
Grounding：本阶段不训练正式Grounding链
OOF：不使用
Test：锁定
```

### 四个自然语言 Type Query

每条记录构造四个 query-sentence 输入：

```text
Location query     + sentence
Person query       + sentence
Organization query + sentence
Other query        + sentence
```

四路输入共享一个可训练 RoBERTa，得到类型条件化 token 表示和 query summary。

#### Query 输入示例

以统一实例为例，模型实际形成四个语义不同的编码任务：

```text
[Location query]     Jordan met Nike executives in Paris after the game.
[Person query]       Jordan met Nike executives in Paris after the game.
[Organization query] Jordan met Nike executives in Paris after the game.
[Other query]        Jordan met Nike executives in Paris after the game.
```

共享 RoBERTa 参数，但四次前向得到不同的条件化 token states：

```text
H_LOC, H_PER, H_ORG, H_OTHER  [4,L,d]
```

因此 `Jordan` 在 PER query 下可获得高 start/end 分数，在 LOC query 下被抑制；
`Nike` 则应在 ORG query 下得到最高span分数。

### 类型条件视觉检索

每个 query 分别检索：

```text
Frozen CLIP global/patch states
+ VinVL R16 region states
```

再通过零初始化 gated residual 修改该类型的 word states：

```text
h'_(t,i) = h_(t,i)
         + sigmoid(g_(t,i)) * tanh(delta_(t,i))
```

query-region 辅助目标使用 detached query，限制视觉对齐目标直接拖动 RoBERTa。

视觉检索分两路：

```text
q_t -> CLIP patch attention -> semantic context_t
q_t -> VinVL region attention -> object context_t
```

两路上下文与 query summary、sentence summary 融合，再为该类型的每个 word 产生
受门控 residual。与正式 Stage1 相比，视觉证据从“所有类型共享”变成“每种类型
分别查询”。

例如：

```text
PER query 更关注 r1/r2 两个人物区域
ORG query 更关注 r3 logo 和 r4 building
LOC query 更关注 r5 landmark
```

但 query 只能提供类别条件，不能自动知道 r1 和 r2 中谁是 `Jordan`。

### Typed-Span 生成

每个类型 query 同时预测：

```text
type existence
start logits
end logits
start-end span match
```

候选区间 `[i,j)` 的分数为：

```text
score(t,i,j)
= start(t,i)
 + end(t,j)
 + span_match(t,i,j)
 + 0.5 * log_sigmoid(existence(t))
```

每类先保留 Top-32、最大span长度为10，再把四类候选放入确定性的最大权重非重叠
区间解码。类型由生成候选的 query 决定，不再对固定span事后分类。

#### 四类候选如何联合竞争

假设模型产生以下候选：

| 候选 | 类型 | 分数 |
|---|---|---:|
| `Jordan` | PER | 8.4 |
| `Jordan` | LOC | 3.1 |
| `Nike` | ORG | 7.8 |
| `Nike executives` | ORG | 8.0 |
| `Paris` | LOC | 8.2 |
| `executives in Paris` | OTHER | 4.0 |

联合解码寻找互不重叠候选的最大总分集合：

```text
argmax_A sum_(c in A) score(c)
subject to spans in A do not overlap
```

它可能选择：

```text
Jordan/PER + Nike executives/ORG + Paris/LOC
```

总分很高、类型也合理，但 `Nike executives` 仍是错误边界。non-overlap 约束只能
阻止相互覆盖，不能判断哪个边界与 gold 完全一致。

#### 与 CRF 的关键区别

Typed-BIO CRF 比较的是全句 token 标签序列：

```text
O/B-PER/I-PER/... 的全局路径
```

TQ-DV 比较的是不同 query 独立产生的区间候选：

```text
(type,start,end,score) 的集合
```

因此 TQ-DV 有更直接的类型条件，却缺少以下共享约束：

```text
统一的token级O标签竞争
B/I连续转移
不同类型在同一个emission空间中的校准
```

四类 query 的分数还需要跨 query 可比。若 ORG query 整体分数尺度偏高，它可能
比真正的 PER/LOC 候选更容易进入最终集合。

### 损失

```text
L = 0.5 * L_existence
  + 1.0 * L_start
  + 1.0 * L_end
  + 1.0 * L_span_match
  + 0.1 * L_query_region
  + 0.01 * L_gate
```

前三轮为 text-only warmup，之后才启用视觉检索和 residual。

各项监督分别回答：

```text
L_existence：该记录是否存在此类实体
L_start：    每个word是否为该类型实体起点
L_end：      每个word是否为该类型实体终点
L_span：     start-end组合是否构成完整实体
L_query_region：query是否检索到该类型实体的正区域
L_gate：     抑制无必要的视觉残差
```

`L_span` 使用较高正例权重，因为全部合法 `[start,end)` 组合中负例远多于正例。
但这也意味着 span 分数需要在严重类别不平衡下学习校准。

#### TQ-DV 统一实例推理

```text
PER query： Jordan [0,1) 得分最高
ORG query： Nike [2,3) 与 Nike executives [2,4) 分数接近
LOC query： Paris [5,6) 得分最高
OTHER query：没有可靠候选

联合解码：
Jordan/PER + Nike executives/ORG + Paris/LOC
```

TQ-DV 正确利用了视觉和 query 判断 `Nike` 属于 ORG，但 start/end/span head 仍可能
偏向更长的 `Nike executives`。这就是“条件类型更准、MNER仍下降”的具体形式。

### 失败原因

TQ-DV 成功改善了类型条件判别：

```text
正式Stage1条件类型准确率  0.9357
TQ-DV条件类型准确率       0.9506
```

但是它用四套类型查询的 start/end/span 打分替换了 Typed-BIO CRF 的统一序列分布。
四类候选之间只在最后通过 non-overlap objective 竞争，缺少 Typed-BIO CRF 在
token 级提供的共享 B/I 转移约束。结果是：

```text
类型更准
但正确span减少55个
最终MNER correct仍减少20个
```

这说明 TQ-DV 的有效成分更可能是 type scorer，而不是新的 span generator。

另一个典型例子是：

```text
New York Times reported from Paris.
```

LOC query 容易提出 `New York/LOC`，ORG query 提出 `New York Times/ORG`。如果
跨 query 分数没有充分校准，联合区间解码可能选择前者。相比之下，Typed-BIO CRF
可以在同一序列中比较：

```text
B-LOC I-LOC O
vs
B-ORG I-ORG I-ORG
```

这正是正式 CRF 序列先验仍然有竞争力的原因。

---

## 三种概率分解对比

### 正式 Stage1

```text
P(typed BIO sequence | text, VinVL)
* P(region/NULL | decoded span,type)
```

边界和类型在一个 CRF 序列内联合决定。

### DVH-Stage1

```text
P(boundary BIO | text,CLIP)
* P(type | fixed span,text,CLIP,VinVL)
* P(region/NULL | span,type,CLIP,VinVL)
```

层次清晰，但错误边界会截断后续类型和 Grounding 的收益。

### TQ-DV-MNER

```text
score(type,start,end)
= existence + start + end + span match + visual retrieval
```

类型在span形成前介入，但丢失统一 Typed-BIO CRF 的强序列先验。

---

## 从 MNER 数学分解理解两次失败

在 exact-match MNER 中，一个预测必须同时满足：

```text
boundary exact
AND
coarse type exact
```

因此可以把 MNER correct 写成：

```text
C_mner = C_span * Accuracy(type | exact span)
```

三套体系的数量关系为：

```text
正式 Stage1：2162 * 0.9357 ~= 2023
DVH epoch20：2114 * 0.9376 ~= 1982
TQ-DV：      2107 * 0.9506 ~= 2003
```

这组等式直接揭示了失败原因：

### DVH

```text
类型条件准确率：约 +0.19个百分点
正确span：       -48
MNER correct：   -41
```

类型增益几乎可以忽略，边界损失直接主导结果。

### TQ-DV

```text
类型条件准确率：约 +1.49个百分点
正确span：       -55
MNER correct：   -20
```

类型分支明显有效，约抵消了35个边界损失带来的 MNER 错误，但仍未完全抵消55个
正确span的下降。

### 达到 0.83 的实际要求

完整 M3.3A Dev 有：

```text
C = 2023
P = 2504
G = 2450
```

若保持预测数不变：

```text
F1 = 2C / (P+G)

C_target
= 0.83 * (2504+2450) / 2
~= 2056
```

所以约需 `+33` 个净正确 typed spans。当前正确边界内仍有：

```text
2162 - 2023 = 139
```

个“边界正确但类型错误”的样本。若完全保护正式span，只需净修正其中约四分之一，
就可能接近 MNER `0.83`。这比重新生成全部span更符合当前实验数据。

---

## 视觉信息对 MNER 的能力边界

讨论中容易把“加入更强视觉编码器”直接等同于“MNER应提升”。实际上视觉对三个
决策的帮助程度并不相同。

### 对 Boundary

视觉可以回答：

```text
图中是否存在人物、地点、logo或产品
某个文本片段是否与画面主题相关
```

但通常不能直接回答：

```text
实体在tweet中从第几个token开始和结束
Nike 与 Nike executives 哪个才是标注边界
New York 与 New York Times 应如何切分
```

边界主要是语言和标注规范问题。图像适合作为弱辅助证据，不适合取代文本序列
约束。

### 对 Type

视觉最可能在以下场景提供增益：

```text
Jordan：人物画面支持PER而不是LOC
Giants：球队logo支持ORG而不是OTHER
Apple：产品/公司视觉上下文帮助区分OTHER与ORG
Paris：地标场景支持LOC
```

这与 TQ-DV 条件类型准确率提升一致。因此视觉的优先用途应是固定span上的类型
验证，而不是重新决定span边界。

### 对 Grounding

视觉是 Grounding 的必要信息，但一般语义仍不等于实体身份：

```text
CLIP知道r1和r2都是篮球运动员
不代表CLIP知道谁是Jordan
```

VinVL 提供实例候选和几何，CLIP 提供开放语义，两者互补；多人同类实例仍需要更
细的关系、OCR、身份或局部证据。

---

## 三套体系的方法论对照

| 维度 | 正式 M3.3A | DVH-Stage1 | TQ-DV-MNER |
|---|---|---|---|
| 初始化 | 正式 RoBERTa Stage1 + 分阶段下游 | 官方 RoBERTa，其他模块新建 | 官方 RoBERTa，其他模块新建 |
| 视觉 | VinVL regions | Frozen CLIP + VinVL | Frozen CLIP + VinVL |
| 边界 | 9类 Typed-BIO CRF | 3类 B/I/O CRF | type-conditioned start/end/span |
| 类型 | 与边界联合 | 固定span后分类 | query在span生成前介入 |
| 区域 | R16+NULL，后续R36细化 | R16+NULL正式训练 | 仅query-region辅助，不输出正式Grounding |
| 全局约束 | CRF序列路径 | Boundary CRF | weighted non-overlap intervals |
| 主要保护 | 后半链分层冻结与preservation | 仅残差零初始化 | 仅残差零初始化、query detach |
| checkpoint指标 | 各阶段冻结指标，最终GMNER | Stage1 Dev GMNER | Dev MNER |
| 正式训练是否OOF | 否 | 否 | 否 |
| Test | 正式一次性完成 | 未访问 | 未访问 |
| 结果 | 当前最优 | Boundary损失主导 | Type有效、span生成失败 |

---

## 讨论中可能被问到的问题

### Q1：为什么正式方法只用 VinVL，反而超过加入 CLIP 的体系？

不是因为 VinVL 一定强于 CLIP，而是正式方法保留了更强的 Typed-BIO CRF 和已经
验证的分阶段 Grounding 链。DVH/TQ-DV 同时改变了边界生成机制，因此架构变化的
损失大于 CLIP 带来的类型或视觉收益，不能把结果简单归因于视觉编码器优劣。

### Q2：DVH 的 EEG/GMNER 一度高于正式 Stage1，为什么仍判失败？

研究目标是提高 MNER 并形成新的完整链。DVH epoch20 的 MNER 为 `0.800970`，明显
低于正式 Stage1 `0.814740` 和完整链 `0.816714`。而且正式 checkpoint 按 GMNER
选择，单轮历史值不能替代预注册 checkpoint。其新span集合还需要重建全部下游，
在 MNER Gate 未通过时继续重建没有依据。

### Q3：TQ-DV 的 0.810275 已经接近基线，是否值得继续调阈值？

不建议围绕 Dev 扫描 span threshold、Top-K 或最大长度。当前主要证据已经显示类型
能力有效、边界生成不足。继续调 decode 可能得到局部 Dev 改善，但不能清楚回答收益
来自类型建模还是阈值选择。fixed-span replay 是更直接的因果检验。

### Q4：为什么不立即解冻 CLIP？

当前失败发生在边界保持，不是缺少 CLIP 容量。解冻会增加训练成本和过拟合风险，
同时使视觉收益更难归因。应先证明冻结 CLIP 特征在固定span类型验证中有可转化
收益，再决定是否微调。

### Q5：后半链为什么不能修复 MNER？

M3.3A 后半链固定 Stage1 span/type，只能：

```text
删除可疑span
调整visible/NULL
调整真实区域
```

它不能新增漏掉的span，也不能把错误类型改成正确类型。因此 MNER correct 从
Stage1 到最终链保持 2023，F1 小幅上升来自错误预测减少。

### Q6：下一步最小实验回答什么？

fixed-span type replay 固定正式 Stage1 的2162个正确span和完整预测集合，只替换类型
评分。它将直接输出：

```text
type corrected
type damaged
net typed-span correction
各LOC/PER/ORG/OTHER切片
最终exact MNER
```

若净修正达到约33个且错误类型损伤可控，TQ 类型模块才具备进入正式 Stage1 的依据。

---

## 当前方法结论

```text
保留：
M3.3A 作为正式 GMNER 最优链
Typed-BIO CRF 作为当前最强span生成器
TQ-DV 的类型条件编码和视觉检索作为可复用组件

关闭：
DVH 的 B/I/O -> post-hoc type 完整替换路线
TQ-DV 的四查询 typed-span generator 正式替换路线
两条失败分支的 Seeds 41/43、下游重建和 Test
```

下一项最小、可归因的验证是 fixed-span type replay：

```text
冻结正式 Stage1 span集合
-> 使用TQ-DV对每个固定span计算四类分数
-> 只允许type变化
-> 比较type corrected/damaged和精确MNER
```

它回答的不是“整套 TQ-DV 是否有效”，而是：

> TQ-DV 已观察到的条件类型提升，能否在完全保护正式边界的条件下转化为至少
> 33 个净正确 typed spans。

---

## 对应实现与协议

```text
正式 M3.3A：
configs/fmnerg_twitter10000_stage1.yaml
configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml
configs/fmnerg_twitter10000_coarse_selector.yaml
configs/fmnerg_twitter10000_fine_grounding_adapter.yaml
configs/fmnerg_twitter10000_evidence_visibility.yaml
docs/HIERARCHICAL_RECORD_VERIFIER.md

DVH-Stage1：
configs/dvh_stage1/frozen_clip_vit_b32_seed42.yaml
gmner/models/dvh_stage1.py
gmner/losses/dvh_stage1_loss.py
docs/experiments/DVH_FROZEN_CLIP_STAGE1_PROTOCOL.md

TQ-DV-MNER：
configs/tq_dv_mner/type_query_dual_visual_seed42.yaml
gmner/models/tq_dv_mner.py
gmner/losses/tq_dv_mner_loss.py
docs/experiments/TQ_DV_MNER_README.md
```
