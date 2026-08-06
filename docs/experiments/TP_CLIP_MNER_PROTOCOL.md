# TP-CLIP-MNER：面向现有 GMNER 的受保护多模态 MNER 强化方案

## 1. 立项目标

当前路线只解决一个问题：

> **在尽量不损伤现有 EEG/Grounding 能力的前提下，引入 CLIP 视觉语义，提升 Stage1 的 typed-BIO 实体边界与类型识别能力。**

当前不再把主要精力放在新的 Grounding scorer 上。新模块首先服务于较弱的 MNER，原有 Grounding 表示路径严格保护。

DESER提供的设计依据是：

* 浅层图文交互可能不足；
* 深层直接融合容易使视觉噪声覆盖文本语义；
* 文本应作为主导模态；
* 后续可通过 Skip-Attention 和模态 Gate 控制视觉信息。

但第一阶段不直接复制完整 DESER，只验证 CLIP 视觉信息是否真的有独立价值。

---

# 2. 当前授权范围

## 立即执行

```text
M0    Train/Dev CLIP视觉缓存
M0.5  typed-BIO与Grounding保护路径审计
M1    受保护的typed-BIO视觉残差实验
```

## 暂时锁定

```text
M2    两层Text-Anchored Skip-Attention
M3    Boundary/Type解耦
M4    模态Gate
M5    cross-fitted伪标签Gate
M6    完整Model-G重建
M7    F3/FMNERG重建
```

## 当前禁止

```text
独立Type Head
新Span Proposal
P4-v2
S4.5
R16扩展为R36
解冻RoBERTa
微调CLIP
生成Test缓存
访问Test
```

---

# 3. 总体架构

```text
文本
 ↓
冻结的现有Stage1文本表示路径
 ↓
mner_base_tokens
 │
 │                         图像
 │                          ↓
 │                    VinVL正式R16框
 │                          ↓
 │              CLIP ViT-B/32区域编码
 │                          ↓
 ├────────────── Cross-Attention ──────────────┐
 │                                             │
 │                                      multimodal tokens
 │                                             │
 └──────────── typed-BIO emission residual ────┘
                         ↓
                原9类typed-BIO CRF
                         ↓
                   新span + 新type
                         ↓
              受保护Grounding replay
                         ↓
           pre_prototype_fused_tokens
                         ↓
              根据新span重新池化
                         ↓
                   原target_repr
                         ↓
        原Grounding Head + image_nodes + 先验
                         ↓
            Stage1 EEG / GMNER输出
```

必须保留两个不同入口：

```text
MNER入口：
当前typed-BIO emission真正读取的token tensor

Grounding入口：
当前正式pre_prototype_fused_tokens
```

不能用原始 RoBERTa 输出代替 `pre_prototype_fused_tokens`。

正式实现必须锁定为：

```text
mner_base_tokens = outputs["fused_tokens"]
E_old            = outputs["ner_logits"]
grounding_tokens = outputs["pre_prototype_fused_tokens"]
```

其中 `mner_base_tokens` 是当前正式 `ner_head` 真正读取的 token tensor，
`E_old` 是全部现有 refinement 完成后的正式 emission。禁止使用：

```text
base_text_nodes
base_ner_logits
任意重新计算或近似得到的中间表示
```

替代上述正式接口。M0.5 必须同时锁定 Stage1 config、checkpoint 和三个
tensor 接口的 SHA256/fingerprint。

---

# 4. MNER输出契约

当前正式模型没有独立的四类 Type Head。边界和类型共同来自9类 typed-BIO CRF：

```text
O

B-PER
I-PER

B-LOC
I-LOC

B-ORG
I-ORG

B-OTHER
I-OTHER
```

因此新模块只输出：

[
\Delta E_i\in\mathbb R^9
]

最终 emission：

[
E_i^{new}
=========

E_i^{old}
+
\rho\cdot\tanh(\Delta E_i)
]

其中：

* (E^{old})：冻结旧 typed-BIO emission；
* CRF transition matrix：冻结；
* residual最后一层：零初始化；
* 不增加独立Type Head；
* 不单独修改实体类型；
* span和type继续由同一次CRF Viterbi解码产生。

这样不会出现：

```text
CRF解码为PER
独立Type Head却预测ORG
```

---

# 5. M0：CLIP缓存构建

## 5.1 数据范围

只处理：

```text
Train
Dev
```

不得处理 Test。

## 5.2 视觉来源

继续使用正式 Stage1 的 R16 bbox。

对每张图片生成：

### 区域特征

[
c_k
===

\operatorname{CLIPVision}
\left(
\operatorname{Crop}(I,b_k)
\right)
]

### 全图特征

[
c_g
===

\operatorname{CLIPVision}(I)
]

第一轮使用原始 bbox crop：

* 不默认向外扩展；
* 不改变区域顺序；
* 不改变 valid mask；
* 不改变NULL位置；
* 使用冻结 CLIP ViT-B/32；
* 使用固定官方预处理。

DESER使用局部视觉表示是为了让文本实体关注局部图像内容，但其固定网格存在无法对应语义对象的问题；这里使用VinVL目标框代替固定网格。

## 5.3 缓存格式

```text
image_id
split
image_sha256
clip_checkpoint_sha256
preprocess_digest
region_count
region_index
original_bbox
valid_region_mask
clip_region_feature
clip_global_feature
vinvl_bbox_manifest_sha256
schema_version
```

## 5.4 M0通过条件

```text
Train/Dev图像覆盖率100%
R16区域数量与顺序完全一致
bbox digest完全一致
invalid region处理一致
缓存重复加载一致
本地/云端manifest一致
Test accessed = false
未训练模型
```

---

# 6. M0.5：保护路径与可达性审计

M0.5不训练模型。

## 6.1 Epoch-0复现

残差为0时，必须逐例满足：

```text
typed-BIO emissions完全一致
CRF路径完全一致
span/type集合完全一致
prediction count完全一致
Grounding logits完全一致
Stage1 Span/MNER/EEG/GMNER完全一致
```

离散结果必须 exact identity。

数值 Gate 分开定义：

```text
typed-BIO emission max abs error       < 1e-7
冻结的MNER输入状态 max abs error       < 1e-7

Grounding各中间阶段 max abs error      < 3e-5
raw grounding logits
after entity NULL prior
after global NULL bias
after detector prior
after type-object compatibility prior
final formal grounding logits
```

`3e-5` 沿用已验证的 CUDA FP32 向量化归约容差。以下结果不使用浮点
容差，必须完全一致：

```text
CRF Viterbi path
span/type prediction set
region/NULL argmax
NULL/visible decision
prediction digest
Span/MNER/EEG/GMNER
各指标correct count
```

## 6.2 旧span Grounding复现

```text
旧span mask
→ pre_prototype_fused_tokens
→ 原池化
→ 原target_repr
→ 原image_nodes
→ 原Grounding Head
```

输出必须与当前正式 Stage1 完全一致。

## 6.3 新span Grounding replay

对新CRF产生的span：

```text
新span坐标
→ 构造新的wordpiece/token mask
→ 从原pre_prototype_fused_tokens池化
→ 产生新target_repr
→ 使用原image_nodes
→ 使用原R16 mask
→ 使用原Grounding Head
→ 按新span和新预测type重新计算冻结先验
```

必须完整复现正式 Grounding 顺序：

```text
raw dot-product logits
→ entity NULL prior（按新mention和新预测type重算）
→ global NULL bias
→ detector-score prior
→ predicted-type/object compatibility prior
→ valid-region mask
```

禁止：

* 复用附近旧实体的 region logits；
* 使用原始RoBERTa代替正式token表示；
* 使用gold type；
* 使用旧span的fixed type；
* 复用旧实体的mention/type NULL prior；
* 修改Grounding阈值；
* 修改NULL bias。

## 6.4 Residual可达性

只用 Train 确定残差边界 (\rho)。设旧 CRF 的最佳路径和次优路径分别为
(y_1,y_2)，路径分数为 (S)，Hamming 距离为 (d_H)，定义每条有效记录的
最小翻转半径诊断：

[
r_i=
\frac{S(y_1)-S(y_2)}
{2\max(d_H(y_1,y_2),1)}
]

第一版固定：

[
\rho=\operatorname{Median}_{Train}(r_i)
]

不根据 Dev 扫描或修改 (\rho)。若结果非有限、(\rho\le10^{-6})，则停止并
重新预注册，不能进入 M1。

需要统计：

```text
Viterbi top-1与top-2 path margin
boundary-only可达修正数
type-only可达修正数
boundary+type联合可达数
在固定rho下gold path可达率
零损伤MNER Oracle
零损伤Stage1 GMNER Oracle
```

Top-1/Top-2 margin 只作为残差尺度诊断，不能直接代表 gold path 可达性。
Gold path 可达性必须使用约束 Viterbi 精确计算。对 gold path (g)，检查：

[
\max_y
\left[
S(y)-S(g)-2\rho d_H(y,g)
\right]
\le 0
]

该结果是“每个 token-label 可以独立使用界内残差”条件下的结构上限，不代表
实际残差网络一定能够达到。

## 6.5 M0.5 Gate

先仅用 Train 封存 (\rho)，随后允许执行一次固定的 Dev 只读可达性审计；不得
依据 Dev 重新计算 (\rho) 或修改任何阈值。M0.5 必须同时满足：

```text
Epoch-0离散输出与指标               exact identity
rho                                 finite and > 1e-6
固定rho下可达gold path记录数         >= 25
零损伤MNER Oracle delta             >= +0.010
零损伤Stage1 GMNER Oracle delta      >= +0.006
Grounding replay数值/离散Gate        passed
Test accessed                       false
```

Oracle 对每条记录只能在“保持 A0”与“采用界内理想残差”之间选择，并必须重新
计算完整预测集合的 F1，不能用 corrected-damaged 近似。任一 Gate 未通过即停止
M1，不通过扩大 (\rho) 补救。

---

# 7. M1：固定实验矩阵

## A0：原 Stage1

正式冻结基线。

开发基准：

```text
Stage1 Dev GMNER = 0.607330
```

---

## A-text：参数匹配 Text-only Control

与 A2 使用：

* 相同Cross-Attention层数；
* 相同hidden size；
* 相同MHA；
* 相同FFN；
* 相同residual head；
* 相近可训练参数量。

但视觉token全部由attention mask屏蔽：

[
K=V=[T;\operatorname{MaskedVisualSlots}]
]

它控制以下影响：

```text
额外参数量
额外文本计算
新增MHA/FFN容量
residual head重新训练收益
```

只有 A2 优于 A-text，才能把收益归因于视觉。

A-text 与 A2 必须分别报告：

```text
declared trainable parameter count
实际获得非零梯度的parameter count
trainable element count
每个optimizer group的参数量与学习率
```

视觉 projection 在 A-text 中可能因为 mask 而没有梯度，因此 A-text 主要控制
新增文本计算容量；A2-paired/A2-shuffled 进一步验证配对视觉信息。

---

## A1：CLIP Global

输入：

```text
mner_base_tokens
+
1个CLIP全图token
```

交互：

[
H^{global}
==========

\operatorname{MHA}
(Q=T,K=[T;c_g],V=[T;c_g])
]

输出9类typed-BIO残差。

该实验主要验证：

> 全局场景语义是否帮助 PER、LOC、ORG、OTHER 的粗类型判断。

---

## A2：CLIP R16 + 单层 Cross-Attention

CLIP区域序列：

[
V=[c_1,\ldots,c_{16}]
]

投影：

[
\tilde c_k=W_cc_k+W_bb_k+W_ss_k
]

交互：

[
H^{mm}
======

\operatorname{MHA}
\left(
Q=T,
K=[T;\tilde V],
V=[T;\tilde V]
\right)
]

经过FFN和残差归一化：

[
\hat H^{mm}
===========

\operatorname{FFNBlock}(T+H^{mm})
]

Emission residual：

[
\Delta E_i
==========

\operatorname{MLP}
[
T_i;
\hat H_i^{mm};
T_i\odot\hat H_i^{mm};
|T_i-\hat H_i^{mm}|
]
]

最终：

[
E_i^{new}
=========

E_i^{old}
+
\rho\tanh(\Delta E_i)
]

固定设置：

```text
Cross-Attention层数 = 1
视觉注入alpha = 1
residual最后一层 = 0初始化
CRF transition冻结
RoBERTa冻结
CLIP冻结
Grounding冻结
```

不同时使用 `alpha≈0`，避免上游Attention初期几乎没有梯度。

---

# 8. A1/A2图像错配诊断

A1、A2 训练完成后，分别固定唯一 checkpoint，不重新训练，执行：

```text
A1/A2-paired：
文本使用原配对图片

A1/A2-shuffled：
按预注册seed对Dev image_id做无固定点置换
```

固定使用五个置换 seed：

```text
101, 102, 103, 104, 105
```

每个置换必须满足当前 record 不获得原 image_id。若数据中存在重复 image_id，
应在唯一 image_id 层完成置换，再映射回 record。

判断：

```text
A1/A2-paired > mean(A1/A2-shuffled)
```

说明模型确实使用了文本—图片对应关系。

若：

```text
A1/A2-paired ≈ mean(A1/A2-shuffled)
```

则收益可能来自：

* 增加参数；
* 通用视觉类别先验；
* 数据集偏置；
* 模型忽略配对关系。

这一诊断不参与 checkpoint 选择，不允许根据置换结果更换 checkpoint、特征或
模型配置。

---

# 9. 参数冻结与训练范围

## 9.1 Record-level训练数据契约

M1 使用每条数据记录恰好一次的 record-level dataset：

```text
每条record每个epoch出现一次
一次计算完整typed-BIO CRF loss
一次读取对应CLIP image cache
不按gold实体展开样本
不使用expand_entities_for_grounding后的重复记录
```

M1 是直接监督的 MNER 分支训练，不是基于旧错误状态的 selector，因此不要求
full-chain OOF。只有未来 M5 使用 text/mm 相对损失监督 Gate 时，才必须执行
Train 内 cross-fitting。

## 9.2 冻结

```text
RoBERTa全部参数
旧typed-BIO emission层
CRF transition
pre_prototype_fused_tokens生成路径
旧Grounding Head
VinVL
CLIP ViT-B/32
Hierarchical Verifier
R36 Coarse
Fine Adapter
Evidence Visibility
```

## 9.3 只训练

```text
CLIP区域projection
CLIP global projection
bbox/score embedding
单层Cross-Attention
FFN
typed-BIO residual MLP
```

所有冻结模块必须：

```text
requires_grad = false
eval mode = true
dropout关闭
参数SHA256训练前后不变
```

## 9.4 固定模型与优化器配置

M1 第一轮统一使用：

```text
hidden_size                 = 768
attention_heads             = 8
ffn_intermediate_size       = 1536
dropout                     = 0.1

optimizer                   = AdamW
learning_rate               = 1e-4
weight_decay                = 0.01
batch_size                  = 8 records
epochs                      = 15
warmup_ratio                = 0.1
gradient_clip_norm          = 1.0
mixed_precision             = true
seed                        = 42
```

A-text、A1、A2 不允许分别调整优化器、训练轮数、dropout 或 hidden size，
也不进行学习率扫描。

---

# 10. 训练损失

第一阶段不使用伪标签Gate。

## 10.1 CRF损失

\[
L_{crf} = -\log P(y\mid E^{new})
\]

## 10.2 旧输出保护

\[
L_{preserve} =
\operatorname{KL}
\left(
p_{old}^{emission}
\Vert
p_{new}^{emission}
\right)
\]

第一版在全部非 padding token 上使用均匀权重，温度固定为：

```text
distillation_temperature = 1.0
```

这里的 KL 是 emission-level distillation，不宣称等价于完整 CRF sequence
distribution distillation。CRF transition 始终冻结，并通过最终 Viterbi
preservation 指标验收实际序列保护效果。

## 10.3 残差正则

\[
L_{res} =
\left|
\tanh(\Delta E)
\right|_2^2
\]

该正则作用于归一化后的有界残差，避免其尺度随 (\rho) 改变。

## 10.4 总损失

\[
L = L_{crf} + \lambda_p L_{preserve} + \lambda_r L_{res}
\]

M1 唯一预注册权重为：

```text
lambda_preserve = 1.0
lambda_residual = 0.01
```

不执行 loss-weight sweep。若该设置导致训练数值失败，只能归档为工程失败并
重新预注册，不能依据 Dev 结果改权重后继续同一实验。

M1不加入：

```text
Grounding loss
独立Type loss
CLIP对比损失
伪标签Gate
token-patch对齐
Span Proposal
错误桶专项权重
```

---

# 11. M1正式评价

M1只能正式评价：

```text
Stage1 Span
Stage1 MNER
Stage1 EEG
Stage1 GMNER
```

EEG和GMNER通过冻结Grounding replay获得。

不能将旧Hierarchical/Fine/Evidence checkpoint直接套在新span上，因为这些模块的缓存和输入绑定旧的：

```text
span_candidates
fixed_type_ids
stage1_spans_only
```

完整Model-G必须等未来M6重新生成缓存并重训。

## 11.1 唯一Checkpoint选择规则

A-text、A1、A2 全部使用同一规则：

```text
1. 只考虑 Dev EEG >= A0 EEG - 0.001 的 checkpoint
2. 在有效 checkpoint 中选择 Stage1 Dev GMNER 最高者
3. GMNER 差异 <= 1e-6 时选择 MNER 较高者
4. GMNER/MNER 均相同时选择更早 epoch
5. 如果不存在满足 EEG 约束的训练后 checkpoint，则该实验 NO_GO
```

Epoch 0 只用于等价性 Gate，不参与“训练后 checkpoint”候选。A2-shuffled、
错误分桶、单类指标和 Test 均不得用于 checkpoint 选择。

---

# 12. 必报错误分解

## 12.1 Boundary-only

在类型正确条件下：

```text
旧boundary错 → 新boundary对
旧boundary对 → 新boundary错
boundary corrected
boundary damaged
boundary net
```

## 12.2 Type-only

在exact span保持一致条件下：

```text
旧type错 → 新type对
旧type对 → 新type错
type corrected
type damaged
type net
```

## 12.3 联合MNER

```text
MNER corrected
MNER damaged
MNER net
新增实体
删除实体
prediction count变化
```

## 12.4 完整GMNER转化

[
\text{MNER-to-GMNER conversion}
===============================

\frac{
\text{新增正确MNER且Grounding正确}
}{
\text{新增正确MNER}
}
]

拆分报告：

```text
新增正确MNER + real-region正确
新增正确MNER + NULL正确
新增正确MNER但Grounding错误
```

## 12.5 EEG保护

```text
base-correct Grounding preserved
base-correct Grounding damaged
新span Grounding正确
新span Grounding错误
real-region corrected/damaged
NULL corrected/damaged
```

`base-correct EEG preservation` 固定定义为：

[
\frac{
|EEG_{correct}^{A0}\cap Predictions^{new}|
}{
|EEG_{correct}^{A0}|
}
]

其中集合元素使用 canonical `(record_id, span_start, span_end, region_or_NULL)`
标识。类型变化不改变 EEG 集合键；删除旧正确 span、改变其 region/NULL 或不再
输出该实体均计为 damaged。

---

# 13. M1正式Gate

## 13.1 Seed42主Gate

A2相对A0必须同时满足：

```text
Dev MNER delta                  >= +0.005
Dev Stage1 GMNER delta          >= +0.003
Span F1 delta                   >= 0
EEG delta                       >= -0.001
base-correct EEG preservation   >= 0.99
boundary net                    >= 0
type net                        >= 0
max(boundary net, type net)     > 0
MNER corrected                  > damaged
GMNER corrected                 > damaged
abs(prediction count delta)     <= 25
Test accessed                   false
```

## 13.2 视觉独立贡献Gate

A2必须同时优于A-text：

```text
MNER(A2 - A-text)  >= +0.002
GMNER(A2)          > GMNER(A-text)
MNER net(A2)       > MNER net(A-text)
MNER(A2-paired) - mean(MNER(A2-shuffled)) >= +0.001
GMNER(A2-paired)   > mean(GMNER(A2-shuffled))
```

如果 A2 只优于A0，却不优于A-text，就不能宣称视觉有贡献。

## 13.3 多种子Gate

Seed42 同时通过主 Gate 和视觉独立贡献 Gate 后，固定全部代码、配置、(\rho)、
loss 权重和 checkpoint 规则，再运行：

```text
41, 42, 43
```

多种子阶段至少复跑 A-text 和 A2，要求：

```text
mean MNER(A2 - A0)              >= +0.005
mean Stage1 GMNER(A2 - A0)      >= +0.003
mean MNER(A2 - A-text)          >= +0.002
至少2/3 seed的MNER(A2 - A0)      > 0
至少2/3 seed的GMNER(A2 - A0)     > 0
mean EEG(A2 - A0)               >= -0.001
mean base-correct preservation  >= 0.99
mean boundary net               >= 0
mean type net                   >= 0
至少一个mean net                > 0
paired-vs-shuffled mean gap      >= +0.001 MNER
Test accessed                   false
```

只有多种子 Gate 通过后，才能形成正式方法结论并授权 M2。Seed42 结果只用于
筛选，不构成“CLIP视觉有效”的最终结论。

---

# 14. 裁决规则

## A-text > A0，A2 ≤ A-text

结论：

> 新增文本计算有效，CLIP视觉无独立贡献。

处理：

```text
保留Text-only结果
停止CLIP路线
不开发Skip-Attention
```

## A1通过全局视觉Gate，A2没有额外收益

A1 的全局视觉 Gate 定义为：

```text
MNER(A1 - A0)                         > 0
MNER(A1 - A-text)                     >= +0.002
GMNER(A1)                             > GMNER(A-text)
MNER(A1-paired)-mean(MNER(A1-shuffled)) >= +0.001
```

结论：

> 全局场景语义有效，但R16区域级视觉交互无效。

处理：

```text
保留Global CLIP
停止区域深交互
```

若 A1 只优于 A0、但不优于 A-text 或 shuffled control，则不能宣称全局 CLIP
有效，应归入“新增文本计算或数据集先验”结果。

## A2 > A0 且 A2 > A-text

结论：

> 配对CLIP区域视觉对typed-BIO具有独立贡献。

进入 A-text/A2 多种子复跑；只有多种子 Gate 通过后才授权 M2。

## A2提升MNER但GMNER不升

结论：

> 文本识别改善无法被现有Grounding闭合。

处理：

```text
保留MNER分析结果
不重建Model-G
不作为GMNER主贡献
```

## A2提升MNER但EEG损伤明显

处理：

```text
停止后续解冻
检查新span/type的Grounding转化率
不通过增加层数补救
```

## A2全部未通过

停止整个视觉MNER方向：

```text
不增加Skip层
不增加Gate
不生成伪标签
不解冻RoBERTa
不换更大CLIP
```

---

# 15. 条件化后续路线

只有A2通过全部Gate后，才重新预注册。

## M2：Text-Anchored Skip-Attention

DESER的核心是每层多模态交互后，重新使用原文本表示作为Query和残差锚点，以减少视觉信息覆盖文本语义。

拟比较：

```text
1层普通Cross-Attention
2层普通Concat-Cross
2层Text-Anchored Skip-Attention
```

## M3：Boundary/Type解耦

当前typed-BIO仍是联合解码。只有基础视觉分支有效后，才讨论：

```text
boundary representation
type representation
```

但必须重新定义一致性契约，不能直接添加独立Type Head。

## M4：模态Gate

先验证普通Gate，不使用伪标签。

## M5：伪标签Gate

DESER通过比较文本和多模态分支的相对损失生成软模态贡献标签，并延迟伪标签生成以避免训练早期不稳定。

你的版本必须使用Train内部cross-fitting，不能直接使用训练内过拟合损失。

## M6：完整Model-G重建

重新生成：

```text
新Stage1 spans/types
R16/R36缓存
fixed type ids
span candidates
Hierarchical训练数据
Fine/Evidence训练数据
```

然后重新训练完整后半链。

## M7：F3/FMNERG

新Model-G冻结后再重训F3，并独立验收FMNERG。

---

# 16. 建议文件结构

```text
scripts/build_clip_r16_cache.py
scripts/audit_mner_grounding_replay.py
scripts/train_typed_bio_visual_residual.py
scripts/evaluate_typed_bio_visual_residual.py

gmner/data/clip_r16_cache.py
gmner/models/typed_bio_visual_residual.py
gmner/models/mner_cross_attention.py
gmner/engine/mner_visual_diagnostics.py

configs/mner_visual_a_text.yaml
configs/mner_visual_a1_global.yaml
configs/mner_visual_a2_r16_cross.yaml

docs/experiments/M0_CLIP_CACHE_PROTOCOL.md
docs/experiments/M0_CLIP_CACHE_RESULT.md
docs/experiments/M0_5_REPLAY_AUDIT_PROTOCOL.md
docs/experiments/M0_5_REPLAY_AUDIT_RESULT.md
docs/experiments/M1_TYPED_BIO_VISUAL_PROTOCOL.md
docs/experiments/M1_TYPED_BIO_VISUAL_RESULT.md
```

自动测试至少包括：

```text
CLIP缓存与R16 region顺序一致
bbox digest一致
Test access guard
正式fused_tokens/ner_logits接口锁
epoch-0 emission一致
CRF路径一致
Grounding各中间阶段replay容差
pre_prototype_fused_tokens入口一致
新span mask正确
invalid region mask正确
新span/type先验重新计算正确
冻结参数SHA256一致
A-text/A2 optimizer参数覆盖唯一且完整
A-text视觉mask完全生效
record-level dataset每条记录每epoch恰好一次
residual bound生效
约束Viterbi gold-path reachability正确
checkpoint选择规则确定性
shuffled image_id无固定点
```

---

# 17. 方法定位

第一阶段通过后，方法可以表述为：

> **在保持现有Grounding表示路径和CRF结构不变的条件下，引入基于VinVL区域框的CLIP视觉语义，通过受保护的typed-BIO emission residual学习图文交互。参数匹配的Text-only控制和图像错配实验用于验证提升是否真正来自配对视觉信息。**

只有后续M2成立，才进一步表述为：

> **文本锚定的深层多模态交互，通过反复回到纯文本Query，限制噪声视觉信息在深层融合中的累积。**

当前实际执行顺序固定为：

```text
M0
→ M0.5
→ A0
→ A-text
→ A1
→ A2
→ A1/A2固定checkpoint错配诊断
→ Seed42 Gate
→ A-text/A2三种子Gate
→ 通过后才讨论M2
```

这份方案首先回答最核心的问题：

> **在现有typed-BIO CRF和正式Grounding路径严格受保护的条件下，配对CLIP视觉信息是否比参数匹配的纯文本增容，提供独立且能够转化为完整GMNER的MNER收益。**
