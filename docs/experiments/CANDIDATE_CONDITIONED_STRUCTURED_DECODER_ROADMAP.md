# Candidate-Conditioned Risk-Aware Structured Decoder

## 文档状态

```text
性质：后续候选研究方案
版本：1.2
修订日期：2026-08-07
当前状态：J0-A PASSED；J0-B / J1 未授权
正式主链：M3.3A 保持不变
B1-T0：NO_GO / SEALED
A1-T0：NO_GO / SEALED
Observable post-hoc correction：TERMINATED
近期唯一方法闸门：J0-B feasibility -> J1
Dev / Test：LOCKED
```

本方案不再尝试在 final-chain 输出之后用冻结标量执行类型翻转或边界替换，而是把决策位置前移：

> 保留 Typed-BIO 的 span-type 联合能力，将其从“最终唯一解码器”改为强候选生成器和基础打分器；候选在最终 argmax 之前主动读取文本、视觉和 record 证据，并通过风险感知的结构化 decoder 选择最终预测集合。

这条路线检验的是一个新假设：

> B1/A1 失败可能来自最终压缩状态的信息不足；候选假设条件化的潜表示和反事实交互，可能提供表格特征中不存在的可分性。

它不推翻 M3.3A，也不能把已有 OOF 标签直接当成完整联合候选监督。

---

# 一、为什么这条路线仍有依据

现有实验已经证明：

```text
候选池中存在可恢复的 typed-span 正例；
CRF k-best 和局部边界候选具有实际 Oracle 容量；
B1-T0 能学到中等类型错误趋势和目标类型信号；
A1-T0 的 source-aware 排序优于 no-source 消融；
但两者都没有形成稳定、高精度、可迁移的后置动作尾部。
```

因此，当前被否定的是：

```text
final-chain 输出
→ 冻结 observable-tabular features
→ 独立 post-hoc correction
```

尚未被检验的是：

```text
候选生成阶段
→ 候选条件化读取上下文
→ base-candidate 潜表示比较
→ 风险监督参与最终结构化解码
```

这使候选条件化联合解码成为合理的新方向，但不代表它已经获得方法成功证据。

当前投入重点不是一次性实现完整多模态框架，而是验证两个前置命题：

```text
J0：受约束 typed-span lattice 是否有足够的完整净 Oracle 空间；
J1：潜表示、候选条件化证据和反事实比较是否产生可迁移的净收益。
```

J1 是近期唯一的方法闸门。视觉、动态 region、record GNN 和完整 set decoder
都只是条件分支，不能与 J1 同时开发并用联合结果反推其中某个机制有效。

---

# 二、总体数据流

第一阶段只做 text-only typed-span 验证；完整路线才扩展视觉和 region：

```text
文本                                      图像
 │                                         │
RoBERTa                                  VinVL
 │                                         │
token states H^T                       region states H^V
 │                                         │
 └──────────────────┬──────────────────────┘
                    ↓
       1. Typed-span candidate lattice
                    ↓
       2. Hypothesis-conditioned evidence
                    ↓
       3. Base-candidate counterfactual comparison
                    ↓
       4. Optional region/NULL expansion
                    ↓
       5. Record-level candidate interaction
                    ↓
       6. Risk-aware structured set decoding
                    ↓
       final set of (span, type, region/NULL)
```

核心候选单位始终是：

```text
typed span:       h_st  = (s, t)
full hypothesis: h_str = (s, t, r)
```

边界和类型不会被拆成两个彼此独立的最终预测任务。

---

# 三、基础编码器与冻结策略

## 3.1 文本编码

```text
输入文本 → RoBERTa → H^T ∈ R^(L×d)
```

## 3.2 图像编码

```text
输入图像 → VinVL → H^V ∈ R^(K×d)
```

第一版不进行稠密 token-region 全局融合。PA1 已表明，无条件视觉融合容易引入噪声；视觉只有在 text-only 候选排序通过后才加入。

## 3.3 首轮冻结边界

J1 首轮固定：

```text
RoBERTa：冻结
Typed-BIO CRF：冻结
正式 Stage1 基础分数：冻结
grounding 与下游模块：冻结且不参与训练

仅训练：
候选条件化文本交互
base-candidate 反事实比较
候选证据与风险头
结构化 typed-span scorer
```

这样可以先回答潜表示交互本身是否有效，避免重演 S3.1 中共享更新损坏 grounding 的问题。

若冻结版本有稳定收益，才单独研究高层 adapter 或有限解冻。

---

# 四、Typed-span Candidate Lattice

保留当前 Typed-BIO CRF：

```text
O
B/I-PER
B/I-LOC
B/I-ORG
B/I-OTHER
```

候选来源固定为：

```text
正式 Viterbi 路径
CRF k-best 路径
边界扩展或收缩
预注册的少量局部扰动
同 span 的替代 coarse-type 假设
```

每个候选必须包含完整 `(span,type)`，例如：

```text
[Jordan, PER]
[Jordan, ORG]
[Jordan Brand, ORG]
[Jordan Brand, OTHER]
```

候选生成必须完全 gold-free，并在监督附加前封存：

```text
候选来源
word-space [start,end)
type_id
CRF/path/base score
candidate_id
group_id
确定性去重与 tie-break
```

## 4.1 两类候选组

### 已有正式预测组

对每个正式预测建立：

```text
KEEP
边界替代候选
类型替代候选
边界+类型联合候选
```

正式 Typed-BIO 输出始终作为 `KEEP`，其效用基准不能被删除。

### 漏检恢复组

仅围绕非正式 k-best 或局部 proposal 建立：

```text
NONE
ADD(span,type) candidate 1
ADD(span,type) candidate 2
...
```

若没有显式 `NONE`，模型只能修正已有实体，不能恢复 pure miss。

## 4.2 候选裁剪

必须逐级报告 Oracle：

```text
原始 candidate lattice oracle
去重后 oracle
每组 Top-K 后 oracle
record 冲突约束后 oracle
最终预算下 oracle
```

不能只报告未受预算限制的理论候选上限。

当前正式 MNER 从 `0.816714` 达到 `0.83` 约需净增加 33 个正确实体。模型不可能
无损兑现全部 Oracle，因此 J0 不能只证明候选空间略高于 `+33`。本路线建议：

```text
最终预算与 record 约束后的净 Oracle（每 1500 records 等效）：
继续 J1 的建议下限     >= +66
更有说服力的目标区间   +70 到 +100
```

J0 使用 7000 条 Train OOF records，因此正式 Gate 按 record 数归一化：

```text
最低 OOF 净增 = ceil(66 * 7000 / 1500) = 308
优选 OOF 区间 = 327 到 467
```

不能直接在 7000 条 OOF 总体上使用未缩放的 `+66`，否则只相当于约 14 个 Dev
实体，低于正式目标所需的 33 个净正确实体。

若 raw Oracle 很高，但去重、Top-K 和冲突约束后只剩约 `+33` 到 `+45`，应停止，
不以“理论上仍超过目标”作为训练 J1 的理由。精确数值仍须在 J0 独立预注册中冻结；
这里给出的是投入决策基准，不构成 J0 执行授权。

---

# 五、Hypothesis-Conditioned Text Evidence

对候选 `(s_i,t_i)` 构造 span 表示：

```text
e_s = [h_start ; h_end ; MeanPool(H^T_s)]
q_i = W_q [e_s ; e_t]
```

候选使用自己的 span-type 查询全文：

```text
E_i^T = Attention(q_i, H^T, H^T)
Z_i^T = f_T(e_s, e_t, E_i^T)
```

因此同一个 span 的不同类型假设可以关注不同上下文：

```text
[Jordan, PER]   → 人物动作、称谓、个人上下文
[Jordan, ORG]   → 品牌、机构、产品或组织上下文
```

边界证据还应显式读取：

```text
span 内部 token
左右边界 token
局部 outside token
完整 record 上下文
候选新增与删除的 token
```

首轮不加入视觉，以确保 text-conditioned interaction 的收益可单独归因。

---

# 六、Base-Candidate Counterfactual Comparison

设正式候选为 `h_0=(s_0,t_0)`，替代候选为 `h_i=(s_i,t_i)`，分别得到潜表示 `Z_0` 和 `Z_i`。

构造反事实比较：

```text
D_(0,i) = [
  Z_0;
  Z_i;
  Z_i - Z_0;
  |Z_i - Z_0|;
  Z_i ⊙ Z_0
]
```

它回答：

```text
候选增加或删除了哪些边界信息？
替代 type 是否获得更强上下文支持？
候选相对 KEEP 的证据增量是多少？
候选是否会与其他正式实体冲突？
执行候选可能损坏什么？
```

模型输出至少包含：

```text
candidate evidence score
candidate advantage score
candidate damage-risk score
```

学习目标不是“候选自身是否像实体”，而是：

> 相对于当前 KEEP 或 NONE，候选是否构成安全改进。

---

# 七、监督与损失

## 7.1 现有 OOF 标签的能力边界

现有监督不能直接混成完整联合标签：

```text
A1：只覆盖 type 与 region identity 不变的边界 replacement
B1：只覆盖 exact-span 的 coarse-type correction
```

它们可以作为对应动作切片的辅助风险监督，但不能直接标注任意 `(span,type,region)` 联合候选。

新 candidate lattice 必须在 gold-free 封存后，重新后附统一监督：

```text
FIX：候选使最终正确数增加，且不损坏正式正确项
NEUTRAL：候选不产生正式净收益，也不损失正式正确项
DAMAGE：候选损坏正式正确项、制造非法冲突或使正确数下降
```

对组合动作还必须在最终 record 预测集合上重算结果，不能简单相加 A1/B1 标签。

## 7.2 分组式训练

训练单位是完整候选组，而不是随机 action：

```text
已有预测组：KEEP vs alternatives
漏检 proposal 组：NONE vs ADD alternatives
```

推荐总损失：

```text
L = L_listwise
  + λ_risk L_FIX/NEUTRAL/DAMAGE
  + λ_preserve L_KEEP_preservation
  + λ_margin L_candidate_vs_base
```

其中：

* `L_listwise`：组内选择正确候选或 KEEP/NONE；
* `L_risk`：学习 OOF 部署式 FIX/DAMAGE 风险；
* `L_preserve`：保护 base-correct 正式输出；
* `L_margin`：要求正候选相对 base 具有明确优势。

同一个 group 的全部候选必须位于同一训练或验证分区，禁止 action-level 随机拆分。

---

# 八、视觉与 Region 扩展：仅在 J1 通过后

对通过 text-only 预筛的 `(span,type)` 候选，构造视觉查询：

```text
q_(s,t)^V = W_v [Z_(s,t)^T ; e_t]
E_(s,t)^V = SparseTopKAttention(q_(s,t)^V, H^V, H^V)
```

保留三个解耦通道：

```text
Z^T：文本证据
Z^V：视觉证据
Z^C：跨模态一致性
```

候选表示：

```text
Z_(s,t) = [Z^T ; g_(s,t) Z^V ; Z^C]
```

视觉可靠性 gate 可读取：

```text
visual availability
region selector margin
attention entropy
NULL probability
text-visual compatibility
```

视觉不可用时强制自然退化为 text-only。

随后只对保留的 typed spans 展开：

```text
(span,type) × Top-M R16/R36 regions + NULL
```

区域分数条件于完整 typed span：

```text
S_region(r | s,t)
```

不能直接对全部 typed spans 与 R36 做笛卡尔积，否则候选规模和负例数量会失控。

---

# 九、Record-Level Interaction

当单组 typed-span scorer 已有稳定信号后，再加入 record-level 交互。

节点：

```text
typed-span candidates
或后续的 typed-span-region hypotheses
```

边：

```text
span overlap
候选互斥
token distance
相同 mention
type compatibility
region competition
共享 region
```

经过一到两层 GNN 或 Transformer：

```text
Z_tilde_i = RecordInteraction(Z_i, {Z_j})
```

现有 Hierarchical Record Verifier 可以在这一阶段被改造成候选集合 encoder，但不能直接假设旧 checkpoint 适用于新 span/type 候选。

---

# 十、正确的最终解码目标

单个三元组的：

```text
argmax_(s,t,r) S(s,t,r)
```

不足以表示一个 record 中的多实体输出。最终必须选择候选集合：

```text
Y* = argmax_(Y ⊆ C) [
  Σ_(h∈Y) S_unary(h)
  + Σ_(h_i,h_j∈Y) S_pair(h_i,h_j)
]
```

约束至少包括：

```text
每个候选组最多选择一个非 KEEP/NONE 动作
不允许非法 span overlap
不允许重复 typed span
region/NULL identity 合法
候选预算固定
确定性 tie-break
```

单候选分数可以写成：

```text
S_unary(h) =
    S_TypedBIO(h)
  + α S_text(h)
  + β S_visual(h)
  + γ S_cross_modal(h)
  + μ S_advantage(h,h0)
  - λ S_damage_risk(h,h0)
```

record 关系由 `S_pair` 单独表达。KEEP/NONE 应具有明确基准效用，而不是在候选列表中被隐式省略。

---

# 十一、OOF 潜表示与部署漂移

J1 需要重新物化 fold-specific 潜表示，现有 sealed tabular rows 不包含：

```text
token states
span pooled states
base/candidate latent vectors
hypothesis-conditioned attention states
```

重新物化必须单独证明：

```text
使用对应 fold-specific checkpoint
外层 held-out record 未进入训练或 checkpoint 选择
candidate_id 与现有 OOF population 一一对应
特征生成不读取 gold sidecar
双重 replay 的身份与连续值稳定
Dev/Test 未访问
```

还必须审计：

```text
fold-specific OOF latent distribution
full-fit Train/Dev deployment latent distribution
各 fold 的均值、方差和范数漂移
候选分数与错误率先验差异
```

因为一个共享 reranker 最终需要从十个 fold 模型的表示迁移到 full-fit 正式模型。若表示漂移严重，应先使用归一化或共享冻结投影解决，不能在 Dev 上事后调阈值。

跨 fold 稳定化的优先顺序固定为：

```text
1. 对各类 latent 做同口径 LayerNorm
2. 使用所有 fold 共享且冻结的 projection
3. 优先使用 candidate - base、绝对差和逐元素乘积等相对表示
4. 绝对 candidate latent 仅作为消融，不作为默认唯一输入
```

其中 `Z_candidate - Z_base` 直接描述同一 checkpoint、同一 record 内的反事实变化，
通常比跨 checkpoint 的绝对向量更稳定。必须分别报告绝对表示和相对表示的跨 fold
漂移，不能只报告合并后的总体均值。

---

# 十二、分阶段实验路线

## J0：Candidate-Lattice Oracle 与物化可行性

```text
性质：只读、OOF-only
训练：无
Dev/Test：锁定
```

J0 分成两个顺序固定的只读子阶段。

### J0-A：受约束候选 Oracle

先完成 gold-free lattice 封存和 post-seal Oracle，只回答：

```text
typed-span 正候选覆盖率是多少？
KEEP/NONE 加入后的约束集合 Oracle 是多少？
不同 Top-K 预算损失多少 Oracle？
联合边界+类型候选是否仍有达到目标的净空间？
```

若最终预算下的受约束 Oracle 未达到预注册投入门槛，立即停止，不物化 latent，
也不进入 J1。

#### J0-A 正式结果

J0-A 已于 2026-08-07 按独立预注册执行完成：

```text
数据范围                    10-fold final-chain OOF Train only
records                     7000
formal KEEP groups          11951
NONE/ADD groups              7558
raw alternatives           337996
deduplicated alternatives  278241

OOF baseline MNER F1       0.790898
Top-1 Oracle net correct       +578
Top-2 Oracle net correct       +687
Top-4 Oracle net correct      +1001
record-constrained Top-4       +988
final-budget constrained       +973
final Oracle MNER F1         0.860002
1500-record equivalent gain   +208.5
```

最终预算固定为：

```text
每组 Top-4
每 record 最多 32 个非控制候选
span 不允许重叠
每 record 最多 ADD 1 个实体
KEEP / NONE 永不裁剪
```

最终 `+973` 精确分解为：

```text
replacement corrected   617
replacement damaged       0
correct ADD              356
net                      973
```

十折均为正净增，gold-free lattice 在 supervision 前后 SHA256 完全不变，Dev/Test
均未访问。因此 J0-A 的候选容量 Gate 通过。该结果只证明**受约束候选空间足够**，
不证明 latent 可合法物化，也不证明 J1 reranker 可学习。

执行前发现 R16 与 R36 的 perturbation span proposal 在 1183 条记录上不完全相同。
该差异在读取 gold 前被记录并修订为：

```text
J0 唯一候选命名空间 = sealed R36 span candidates
R16                    = 仅做描述性身份审计
R16/R36 union          = 禁止
```

原因是 formal predictions 与最终下游状态均锚定 R36。完整机器结果见
`j0_a_candidate_lattice_oracle_result.json`。

Windows/Linux 独立重放还发现两个派生分数存在最大 `4.44e-16` 的 `libm` 末位差异，
但离散 digest、排序和全部 Oracle 指标完全一致。正式 artifact 因此只将派生的
`type_log_probability` 与 `typed_score` 使用 precision 50 的 `Decimal` 计算并按
`ROUND_HALF_EVEN` 固定为小数点后 12 位；原始 logits 和输入 score 不改。该规范
用于保证跨平台完整文件 SHA256 一致。

### J0-B：潜表示物化可行性

仅在 J0-A 通过后验证：

```text
fold-specific latent states 能否合法、确定性物化？
candidate/base identity 能否与 sealed lattice 一一对应？
绝对与相对 latent 的跨 fold 漂移是否可控？
full-fit deployment latent 是否落在 OOF 可支持范围内？
```

J0-B 仍然不训练 reranker，也不访问 Dev/Test。

## J1：Frozen Text-Only Candidate Reranker

```text
冻结 RoBERTa、Typed-BIO 和 grounding
只训练候选条件化文本交互、反事实比较和风险头
只评价 Span / MNER
不运行视觉，不重建 R16/R36，不评价最终 GMNER
```

必要消融：

```text
C1：静态 span latent pooling
C2：+ type-hypothesis-conditioned text attention
C3：+ base-candidate counterfactual comparison
C4：+ FIX/DAMAGE risk supervision
```

J1 必须针对同一候选状态联合回答：

```text
候选是否比 KEEP/NONE 获得更多证据？
执行该候选是否会损坏当前正确决策？
```

不能把一个独立 evidence 排序器和另一个 risk 分类器的总体指标相乘后声称动作安全。
正式动作必须来自同一候选的 advantage 与 DAMAGE risk 联合分数。

只有锁定 OOF 上出现稳定的：

```text
C1 static pooling
< C2 conditioned evidence
< C3 counterfactual comparison
< C4 risk-aware scoring
```

并且 C3/C4 的 corrected 明显高于 damaged，才能证明新机制优于普通 latent MLP。
若 C3/C4 仍无稳定净收益，候选纠错主线直接停止，不再用视觉或 GNN 补救。

## J2：Typed-Span-Conditioned Visual Evidence

仅在 J1 通过后加入：

```text
Sparse Top-K region interaction
visual reliability gate
text-only natural fallback
```

先验证视觉是否降低 DAMAGE、提高候选精度，而不是只提高 AUROC。

J2 的正式目标不是 visual coarse-type accuracy，而是：

```text
降低 text-only 候选的 DAMAGE
提高正确候选相对 KEEP/NONE 的 margin
视觉缺失时严格退化为冻结的 J1 text-only 行为
```

若视觉只提高 AUROC 或 region recall，却不改善动作 precision、net correction 和
KEEP preservation，则 J2 判为无效，不进入动态 region 或 record-level 解码。

## J3：Dynamic Region 与 Record-Level Joint Decoder

仅在 J2 或独立 region Gate 通过后：

```text
重新构建候选级 R16/R36
训练动态 region/NULL scorer
加入 record candidate graph
执行 constrained set decoding
重建完整下游链
评价 GMNER
```

现有下游 checkpoint 和 cache 不能直接用于新的 span/type 候选；必须重新生成或证明严格语义兼容。

---

# 十三、J1 的正式评价重点

不能只报告候选分类准确率或 AUROC。至少需要：

```text
Span F1 / MNER F1
corrected / damaged / neutral / net
KEEP preservation
NONE/ADD action precision
boundary-only / type-only / joint action分解
每个候选来源的净收益
每折和每seed稳定性
候选预算与Oracle retention
```

达到 `MNER=0.83` 需要相对正式 `0.816714` 获得约 33 个净正确实体。正式训练前应先证明受约束候选 lattice 的 Oracle 明显高于这一需求，并为模型误差留出余量。

任何 Gate、seed 数、开发折、锁定折和阈值规则都必须在训练前独立预注册，不能沿用 A1-T0 的参数后事后解释。

---

# 十四、主要风险与停止条件

## 风险 1：候选 Oracle 足够，但潜表示不可分

若 J1 的候选条件化模型仍没有稳定高精度动作区域，则说明问题不只是表格压缩，应停止继续堆叠视觉和 GNN。

## 风险 2：OOF 与 full-fit 表示漂移

若 fold-specific latent 与正式模型 latent 无法对齐，OOF 风险标签难以部署。不能通过 Dev 二次校准掩盖这一问题。

## 风险 3：候选组合爆炸

必须采用：

```text
typed-span Top-K
→ region Top-M
→ record constrained decode
```

并逐层报告 Oracle retention。

## 风险 4：模块过多导致无法归因

第一版禁止同时加入：

```text
视觉 attention
record GNN
动态 R36
全量 RoBERTa 解冻
新 grounding loss
```

## 风险 5：风险标签与动作空间不匹配

A1/B1 标签只可用于其定义内动作。任何新联合动作都必须重新进行 gold-free 封存和 post-seal supervision。

---

# 十五、最终框架定位

本方案最有价值的核心不是“再增加一个更大的 decoder”，而是三个明确机制：

```text
1. Typed hypothesis-conditioned evidence extraction
2. Base-candidate counterfactual comparison
3. Risk-aware structured set decoding
```

推荐的实际顺序是：

```text
J0 受约束联合候选 Oracle
→ J1 冻结 text-only latent reranker
→ J2 条件化稀疏视觉证据
→ J3 动态 region 与 record-level set decoder
```

只有 J1 在严格 OOF 锁定评估中获得稳定 typed-span 净收益，完整多模态联合架构才值得继续投入。

从论文方法角度，只有消融能够证明 conditioned evidence、counterfactual comparison
和 risk-aware decoding 分别带来递增且可迁移的净收益，才能形成完整创新主张。若最终
实现退化为 `CRF candidates -> MLP reranker`，则工程上可以作为基线，但不足以支撑
本路线声称的方法贡献。

当前授权边界为：

```text
J0-A gold-free lattice + post-seal Oracle  COMPLETE / PASSED
J0-B latent rematerialization              NOT AUTHORIZED
J1 training                                NOT AUTHORIZED
J2/J3                                      LOCKED
Dev/Test                                   LOCKED
```
