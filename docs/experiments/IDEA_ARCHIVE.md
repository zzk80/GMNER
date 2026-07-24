# 原型记忆方向早期设计归档

下面给你一个**能落地到你现有框架**的版本。你的基础组件是：

[
\boxed{
\text{BERT} + \text{CLIP} + \text{CRF} + \text{VinVL}
}
]

所以不要再设计得太虚。最终架构可以叫：

[
\boxed{
\textbf{ARGP-BVC}
}
]

即：

> **Ambiguity-aware Reliability-gated Prototype BERT-VinVL-CLIP GMNER Framework**

中文可以叫：

> **歧义感知与可靠性门控的 BERT-VinVL-CLIP 原型增强 GMNER 框架**

---

# 1. 总体思路

你现在的任务是 GMNER，也就是：

[
\text{text} + \text{image}
\rightarrow
\text{entity span} + \text{entity type} + \text{grounding region}
]

现有基础框架可以这样理解：

```text
文本  → BERT → CRF → 实体识别
图像  → VinVL → 区域特征 / object labels
图文  → CLIP → 图文语义对齐 / 视觉可靠性辅助
```

现在把上面讨论的原型思想加进去，形成：

```text
BERT 表示
   ↓
基础 CRF 识别候选实体
   ↓
对候选实体判断：是否歧义？
   ↓
用 type/subtype 原型检索语义锚点
   ↓
判断原型检索是否可靠
   ↓
可靠才残差融合，不可靠就拒绝融合
   ↓
最终 CRF 解码 + VinVL grounding
```

核心不是“所有实体都用原型增强”，而是：

[
\boxed{
\text{歧义实体} + \text{可靠原型} \Rightarrow \text{增强}
}
]

否则：

[
\boxed{
\text{不增强}
}
]

---

# 2. 完整架构图

```text
输入：文本 X + 图像 I
        │
        ├──────────────────────────────┐
        │                              │
        ↓                              ↓
   BERT Text Encoder              VinVL Detector
        │                              │
        │                              ├── region features: v_j ∈ R^2048
        │                              ├── bounding boxes: b_j
        │                              └── object labels / attributes
        │
        ↓
   Token representations H
        │
        ↓
   Base CRF
        │
        ├── 基础 BIO 预测
        └── 候选实体 spans
                    │
                    ↓
          Entity representation h_e
                    │
        ┌───────────┼────────────┐
        │           │            │
        ↓           ↓            ↓
 Ambiguity     Prototype      Visual evidence
 Estimator     Retrieval      from VinVL + CLIP
        │           │            │
        ↓           ↓            ↓
      u_e       prototype      r_vis
                reliability
                    r_p
        └───────────┬────────────┘
                    ↓
      Ambiguity-Reliability Gate
                    │
                    ↓
            g_e ∈ [0,1]
                    │
                    ↓
        Residual Prototype Fusion
                    │
                    ↓
        Prototype-enhanced token/entity representation
                    │
                    ↓
              Final CRF Decoder
                    │
                    ↓
          Final entity labels + types
                    │
                    ↓
        Entity-region grounding with VinVL regions
```

---

# 3. 模块一：BERT 文本编码

输入文本：

[
X = {x_1,x_2,...,x_n}
]

经过 BERT：

[
H = \text{BERT}(X)
]

其中：

[
H = {h_1,h_2,...,h_n}, \quad h_i \in \mathbb{R}^{768}
]

这里 (h_i) 是 token 级别上下文表示。

如果你的 BERT 是 `bert-base`，维度就是 768。

---

# 4. 模块二：VinVL 图像识别

图像输入 VinVL，得到区域级视觉信息：

[
V = {v_1,v_2,...,v_m}
]

其中：

[
v_j \in \mathbb{R}^{2048}
]

同时 VinVL 还能输出：

```text
bounding box
object label
attribute label
object confidence
```

例如：

```text
region 1: person, standing, 0.92
region 2: basketball, orange, 0.87
region 3: jersey, red, 0.76
```

因为 VinVL 的视觉特征是 2048 维，而 BERT 是 768 维，所以需要投影：

[
\hat{v}_j = W_v v_j
]

其中：

[
W_v \in \mathbb{R}^{768 \times 2048}
]

得到：

[
\hat{v}_j \in \mathbb{R}^{768}
]

这样 VinVL 区域特征就可以和 BERT 实体表示对齐。

---

# 5. 模块三：CLIP 视觉语义辅助

CLIP 在这里不直接替代 VinVL，而是做两个辅助作用。

---

## 5.1 图文全局相关性

用 CLIP 编码整张图片和文本：

[
g_I = \text{CLIP}_{img}(I)
]

[
g_X = \text{CLIP}_{text}(X)
]

计算图文相关性：

[
s_{global} = \cos(g_I,g_X)
]

如果 (s_{global}) 很低，说明图文弱相关。此时视觉信息不应该强行注入。

---

## 5.2 VinVL object label 语义辅助

VinVL 会输出 object label，例如：

```text
person
basketball
building
logo
car
```

可以把 label / attribute 拼成文本：

```text
a standing person
a basketball player
a company logo
a city building
```

然后用 CLIP text encoder 编码：

[
o_j = \text{CLIP}_{text}(label_j)
]

对于实体 mention，也构造文本 prompt：

```text
the entity Jordan in the sentence: Jordan scored 30 points.
```

编码为：

[
c_e = \text{CLIP}_{text}(prompt_e)
]

然后计算实体和 VinVL object label 的语义相关性：

[
s_{clip}(e,j)=\cos(c_e,o_j)
]

这个分数可以辅助判断：

> 当前实体是否可能和某个 VinVL 检测区域相关。

注意：**VinVL 提供区域，CLIP 提供语义匹配可靠性。**

---

# 6. 模块四：基础 BERT-CRF 分支

普通 BERT-CRF 先做基础识别。

对每个 token：

[
e_i^{base} = W_o h_i + b
]

然后 CRF 解码：

[
Y^{base} = \text{CRF}(e^{base})
]

得到初始 BIO 标签和候选实体 span：

[
E^{base} = {e_1,e_2,...,e_K}
]

例如：

```text
Jordan scored 30 points in Chicago.
B-PER O O O O B-LOC
```

得到候选实体：

```text
Jordan / PER
Chicago / LOC
```

这一分支的作用有两个：

1. 提供基础 NER 结果；
2. 为后续原型模块提供候选实体。

---

# 7. 模块五：实体表示构造

对于候选实体：

[
e=(s,t)
]

从 BERT token 表示中池化：

[
h_e = \text{Pool}(h_s,h_{s+1},...,h_t)
]

推荐先用 mean pooling：

[
h_e = \frac{1}{t-s+1}\sum_{i=s}^{t}h_i
]

得到实体表示：

[
h_e \in \mathbb{R}^{768}
]

---

# 8. 模块六：构建 type/subtype 原型库

这里原型库只保留两层：

[
\boxed{
P^{type}, P^{subtype}
}
]

不单独设置“知识原型”，因为知识已经融入 type/subtype 原型的构建过程。

---

## 8.1 Type 原型

type 原型对应：

```text
PER, ORG, LOC, MISC
```

每个 type 原型由两部分组成：

[
p_c^{type}
==========

\lambda p_c^{data}
+
(1-\lambda)p_c^{desc}
]

其中：

### 数据原型

[
p_c^{data}
==========

\frac{1}{N_c}
\sum_{e:y_e=c}h_e
]

也就是训练集中同类实体表示均值。

### 描述原型

为每个类别写描述：

```text
PER: a person entity refers to a human individual.
ORG: an organization entity refers to a company, team, institution, agency, or brand.
LOC: a location entity refers to a country, city, region, landmark, or venue.
MISC: a miscellaneous entity refers to an event, product, work, nationality, language, or other named concept.
```

用 BERT 编码这些描述：

[
p_c^{desc} = \text{BERT}_{desc}(d_c)
]

推荐使用 `[CLS]` 表示或 mean pooling。

---

## 8.2 Subtype 原型

subtype 原型对应更细语义。

可以设置如下：

```text
PER:
  athlete, politician, artist, celebrity, ordinary person

ORG:
  company, sports team, institution, government agency, brand

LOC:
  country, city, region, landmark, venue

MISC:
  event, product, work, nationality, language, award
```

每个 subtype 写一句描述：

```text
athlete: a person who participates in sports or physical competitions.
company: an organization that produces goods or provides services.
city: a large human settlement and geographical location.
product: an item or service produced by an organization.
```

编码：

[
p_s^{sub} = \text{BERT}_{desc}(d_s)
]

训练过程中可以做软更新：

[
p_s^{sub}
\leftarrow
m p_s^{sub}
+
(1-m)\bar{h}_s
]

其中：

[
\bar{h}_s
=========

\frac{
\sum_e q_{e,s}h_e
}{
\sum_e q_{e,s}
}
]

[
q_{e,s}
=======

\text{softmax}*{s\in S*{y_e}}
(\cos(h_e,p_s^{sub})/\tau)
]

注意：subtype 没有 gold 标签，所以只能软对齐，不能硬监督。

---

# 9. 模块七：实体歧义估计

对每个候选实体 (e)，先用基础分类头预测实体类型：

[
p_{base}(c|e)=\text{Softmax}(W_b h_e)
]

然后计算歧义程度。

可以用熵：

[
u_e = -\sum_c p_{base}(c|e)\log p_{base}(c|e)
]

也可以用 top-2 margin：

[
u_e = 1 - (p_{top1} - p_{top2})
]

如果：

```text
PER: 0.92
LOC: 0.03
ORG: 0.03
MISC: 0.02
```

则歧义低。

如果：

```text
PER: 0.43
LOC: 0.39
ORG: 0.11
MISC: 0.07
```

则歧义高。

---

# 10. 模块八：多视角 query 构建

为了避免“用歧义实体自己检索原型”的自证陷阱，query 不只使用 (h_e)。

构造三个主要 query：

---

## 10.1 实体 query

[
q_{ent}=W_{ent}h_e
]

表示实体本身语义。

---

## 10.2 上下文 query

从实体左右上下文中池化：

[
h_{ctx}
=======

\text{Pool}(H_{\text{left}},H_{\text{right}})
]

[
q_{ctx}=W_{ctx}h_{ctx}
]

例如：

```text
Jordan scored 30 points.
```

实体 `Jordan` 的上下文是：

```text
scored 30 points
```

这能帮助检索到 athlete / PER 相关原型。

---

## 10.3 视觉 query

使用实体表示对 VinVL 区域做 attention：

[
\alpha_{e,j}
============

\text{softmax}
(\cos(W_e h_e, W_v v_j))
]

[
q_{vis}
=======

\sum_j \alpha_{e,j}\hat{v}_j
]

然后结合 CLIP 的实体-object label 语义相似度修正：

[
\alpha_{e,j}
============

\text{softmax}
(
\cos(W_e h_e, W_v v_j)
+
\eta s_{clip}(e,j)
)
]

最终 query：

[
q_e
===

W_q[q_{ent};q_{ctx};q_{vis}]
]

如果你想第一版更稳，可以先不用 (q_{vis}) 检索原型，只用：

[
q_e = W_q[q_{ent};q_{ctx}]
]

视觉信息先单独用于 grounding 和可靠性判断。

---

# 11. 模块九：原型检索

用 (q_e) 分别和 type/subtype 原型计算相似度。

---

## 11.1 Type 相似度

[
s_e^{type}(c)
=============

\cos(q_e,p_c^{type})
]

---

## 11.2 Subtype 相似度

对于每个 subtype：

[
s_e^{sub}(s)
============

\cos(q_e,p_s^{sub})
]

然后把 subtype 分数聚合回对应 type。

对于类别 (c)，其 subtype 集合是 (S_c)：

[
s_e^{sub}(c)
============

\log
\sum_{s\in S_c}
\exp(s_e^{sub}(s)/\tau)
]

也可以用 max：

[
s_e^{sub}(c)
============

\max_{s\in S_c}s_e^{sub}(s)
]

---

## 11.3 总原型分数

[
s_e^{proto}(c)
==============

\beta_1 s_e^{type}(c)
+
\beta_2 s_e^{sub}(c)
]

得到：

```text
PER score
ORG score
LOC score
MISC score
```

---

# 12. 模块十：原型可靠性判断

不能只看 top1 相似度，还要看 top1 和 top2 的差距。

[
r_e^{proto}
===========

## s_{top1}^{proto}

s_{top2}^{proto}
]

如果：

```text
PER: 0.73
LOC: 0.31
```

说明原型比较可靠。

如果：

```text
PER: 0.51
LOC: 0.49
```

说明原型本身也不确定，不应强行融合。

---

# 13. 模块十一：视觉可靠性判断

视觉信息也要判断是否可靠。

可以用三个指标：

---

## 13.1 VinVL 区域匹配最大值

[
r_e^{region}
============

\max_j
\cos(W_e h_e, W_v v_j)
]

---

## 13.2 VinVL top-2 区域 margin

[
r_e^{region-margin}
===================

## a_{top1}

a_{top2}
]

---

## 13.3 CLIP 图文全局相关性

[
s_{global}
==========

\cos(\text{CLIP}*{img}(I),\text{CLIP}*{text}(X))
]

最终视觉可靠性：

[
r_e^{vis}
=========

\mu_1 r_e^{region}
+
\mu_2 r_e^{region-margin}
+
\mu_3 s_{global}
]

如果图文弱相关，视觉相关性低，后续视觉信息的权重要降低。

---

# 14. 模块十二：歧义-可靠性门控

最终门控由三个因素决定：

1. 实体是否歧义；
2. 原型是否可靠；
3. 视觉信息是否可靠。

[
g_e
===

\sigma
(
W_g[
u_e;
r_e^{proto};
r_e^{vis};
m_e^{base}
]
)
]

其中：

[
m_e^{base}=p_{top1}^{base}-p_{top2}^{base}
]

更简单的形式：

[
g_e
===

\text{Ambiguity}(u_e)
\times
\text{Reliability}(r_e^{proto})
]

视觉可靠性可以作为辅助项：

[
g_e
===

\text{Ambiguity}(u_e)
\times
\text{Reliability}(r_e^{proto})
\times
\text{VisualReliability}(r_e^{vis})
]

门控逻辑是：

| 情况            | 原型融合     |
| ------------- | -------- |
| 实体不歧义         | 不融合      |
| 实体歧义，但原型不可靠   | 不融合      |
| 实体歧义，原型可靠     | 融合       |
| 图文弱相关         | 降低视觉分支影响 |
| VinVL 检测区域不确定 | 降低视觉注入   |

---

# 15. 模块十三：残差式原型融合

根据原型分数得到实体相关原型表示：

[
k_e
===

\sum_c \rho_c p_c^{type}
+
\sum_s \rho_s p_s^{sub}
]

其中：

[
\rho=\text{softmax}(s_e^{proto})
]

然后构造修正项：

[
\Delta_e
========

\text{MLP}
(
[h_e;k_e;h_e-k_e;h_e\odot k_e]
)
]

最终实体表示：

[
\tilde{h}_e
===========

h_e
+
g_e\Delta_e
]

这一步很关键。

不是：

[
\tilde{h}_e=k_e
]

而是：

[
\tilde{h}_e=h_e+\text{correction}
]

也就是原型只做修正，不替代 BERT 表示。

---

# 16. 模块十四：回写 token 表示并再次 CRF

CRF 是 token-level 解码，所以需要把实体级增强回写到 token。

对于实体 span：

[
e=(s,t)
]

对实体内部 token：

[
\tilde{h}_i
===========

h_i
+
g_e W_r[\Delta_e;h_i]
\quad
i\in[s,t]
]

非实体 token：

[
\tilde{h}_i=h_i
]

然后计算最终 emission score：

[
e_i^{final}
===========

W_f\tilde{h}_i+b_f
]

送入最终 CRF：

[
Y^{final}
=========

\text{CRF}(e^{final})
]

---

# 17. 模块十五：VinVL grounding

对于最终实体表示：

[
\tilde{h}_e
]

和 VinVL 区域：

[
\hat{v}_j
]

计算实体-区域匹配分数：

[
a_{e,j}
=======

\cos(W_e^g\tilde{h}*e,W_v^g v_j)
+
\eta s*{clip}(e,j)
]

其中：

* 第一项来自实体表示和 VinVL region feature；
* 第二项来自 CLIP 对实体文本和 VinVL object label 的语义匹配。

最终选择：

[
j^*=\arg\max_j a_{e,j}
]

为了避免不可见实体被强行 grounding，可以加入 NULL region：

[
v_{null}
]

区域集合变成：

[
V'={v_1,...,v_m,v_{null}}
]

如果：

[
j^*=null
]

表示该实体不进行视觉 grounding。

---

# 18. 训练流程

推荐采用二阶段训练，更容易落地。

---

## Stage 1：训练基础 BERT-CRF + VinVL/CLIP 对齐

先训练基础模型：

[
\mathcal{L}_{base}
==================

\mathcal{L}*{CRF}^{base}
+
\lambda_g \mathcal{L}*{ground}
+
\lambda_a \mathcal{L}_{align}
]

得到一个可以识别候选实体的基础 BERT-CRF。

---

## Stage 2：加入原型门控增强

用训练集 gold span 或基础 CRF 预测 span 作为实体候选，训练原型模块。

最终损失：

[
\mathcal{L}
===========

\mathcal{L}*{CRF}^{final}
+
\lambda_1\mathcal{L}*{type}
+
\lambda_2\mathcal{L}*{sub}
+
\lambda_3\mathcal{L}*{ground}
+
\lambda_4\mathcal{L}_{align}
]

---

## 18.1 Type prototype loss

[
\mathcal{L}_{type}
==================

-\log
\frac{
\exp(\cos(h_e,p_y^{type})/\tau)
}{
\sum_c
\exp(\cos(h_e,p_c^{type})/\tau)
}
]

---

## 18.2 Subtype prototype loss

因为没有 subtype gold label，所以用集合式约束：

[
\mathcal{L}_{sub}
=================

-\log
\frac{
\sum_{s\in S_y}
\exp(\cos(h_e,p_s^{sub})/\tau)
}{
\sum_c
\sum_{s\in S_c}
\exp(\cos(h_e,p_s^{sub})/\tau)
}
]

---

## 18.3 Grounding loss

如果有实体-region 标注：

[
\mathcal{L}_{ground}
====================

-\log
\frac{
\exp(a_{e,j^+}/\tau)
}{
\sum_j
\exp(a_{e,j}/\tau)
}
]

如果没有 region 标注，可以先用弱监督：

[
\mathcal{L}_{align}
===================

-\log
\frac{
\exp(\cos(\tilde{h}*e,g_I)/\tau)
}{
\sum*{I'}
\exp(\cos(\tilde{h}*e,g*{I'})/\tau)
}
]

---

# 19. 推理流程

推理时按下面流程走。

```text
1. 输入文本和图片
2. BERT 得到 token 表示
3. VinVL 得到区域特征、bbox、object labels
4. CLIP 计算图文全局相关性和实体-object label 语义分数
5. 基础 CRF 得到候选实体
6. 对每个候选实体：
   a. 计算实体表示 h_e
   b. 计算基础类型分布和歧义分数 u_e
   c. 构造 ent/context/visual query
   d. 检索 type/subtype 原型
   e. 计算原型可靠性 r_proto
   f. 计算视觉可靠性 r_vis
   g. 得到门控 g_e
   h. 残差融合原型
7. 回写增强后的 token 表示
8. 最终 CRF 解码
9. 对最终实体做 VinVL region grounding
10. 输出实体类型和对应 bbox / NULL
```

---

# 20. 最小可实现版本

第一版不要做太复杂。建议你先落地这个版本：

[
\boxed{
\text{BERT}
+
\text{VinVL region features}
+
\text{CLIP reliability}
+
\text{Type/Subtype prototype bank}
+
\text{Ambiguity-Reliability Gate}
+
\text{Residual Fusion}
+
\text{CRF}
}
]

具体删减如下：

| 模块                             | 第一版是否保留 |
| ------------------------------ | ------- |
| BERT                           | 保留      |
| CRF                            | 保留      |
| VinVL 2048 region features     | 保留      |
| CLIP global image-text score   | 保留      |
| CLIP entity-object label score | 可选      |
| type prototype                 | 保留      |
| subtype prototype              | 保留      |
| subtype 动量更新                   | 可选      |
| visual query 参与原型检索            | 可选      |
| NULL visual region             | 建议保留    |
| gate loss                      | 暂时不加    |
| Gaussian prototype             | 不加      |
| 反事实原型                          | 不加      |
| 原型图推理                          | 不加      |

最小损失：

[
\boxed{
\mathcal{L}
===========

\mathcal{L}*{CRF}^{final}
+
\lambda_1\mathcal{L}*{type}
+
\lambda_2\mathcal{L}*{sub}
+
\lambda_3\mathcal{L}*{ground}
}
]

---

# 21. 落地版本的模块命名

你可以在代码里这样组织：

```text
models/
  bert_encoder.py
  vinvl_encoder.py
  clip_reliability.py
  prototype_bank.py
  ambiguity_estimator.py
  prototype_retriever.py
  reliability_gate.py
  residual_fusion.py
  crf_decoder.py
  grounding_head.py
```

核心前向流程：

```python
H = bert_encoder(input_ids, attention_mask)

vinvl_feats, boxes, obj_labels = vinvl_encoder(image)

base_emissions = base_classifier(H)
base_tags = crf.decode(base_emissions)

entity_spans = get_candidate_spans(base_tags)  # train 阶段可用 gold spans

for e in entity_spans:
    h_e = span_pool(H, e)

    p_base = entity_type_classifier(h_e)
    u_e = ambiguity_score(p_base)

    q_e = build_query(h_e, H, e, vinvl_feats, obj_labels)

    proto_scores, k_e = prototype_retriever(q_e, prototype_bank)
    r_proto = prototype_reliability(proto_scores)

    r_vis = visual_reliability(h_e, vinvl_feats, clip_score)

    g_e = reliability_gate(u_e, r_proto, r_vis)

    delta_e = residual_fusion(h_e, k_e)

    h_e_enhanced = h_e + g_e * delta_e

H_enhanced = write_back(H, entity_spans, h_e_enhanced)

final_emissions = final_classifier(H_enhanced)
loss_crf = crf.loss(final_emissions, labels)

grounding_scores = grounding_head(h_e_enhanced, vinvl_feats, clip_scores)
```

---

# 22. 这套架构的核心创新表达

可以写成三点。

---

## 创新点一：BERT-VinVL-CLIP 的互补视觉建模

VinVL 负责区域级 object grounding：

[
v_j, b_j, label_j
]

CLIP 负责图文语义相关性和视觉可靠性判断：

[
s_{global}, s_{clip}(e,j)
]

这样避免单纯依赖 VinVL region feature 或单纯依赖 CLIP global feature。

---

## 创新点二：知识增强的 type/subtype 原型库

构建：

[
P^{type}, P^{subtype}
]

其中 type 原型提供粗粒度类别锚点，subtype 原型提供细粒度语义锚点。

它们不是单纯训练集均值，而是：

[
\text{数据统计} + \text{类别/子类型描述}
]

共同构建。

---

## 创新点三：歧义感知与可靠性门控

不是所有实体都用原型增强，而是：

[
g_e =
\text{Ambiguity}(e)
\times
\text{PrototypeReliability}(e)
\times
\text{VisualReliability}(e)
]

只有当实体歧义高、原型检索可靠、视觉证据可信时，才进行残差式原型融合。

---

# 23. 最终一句话总结

你的落地架构可以概括为：

> 以 BERT-CRF 作为文本实体识别主干，以 VinVL 提供区域级视觉对象特征，以 CLIP 评估图文和实体-object 语义相关性；在此基础上构建 type/subtype 多粒度原型库，并通过实体歧义程度、原型检索可靠性和视觉可靠性共同控制原型是否残差注入，最终由增强后的 token 表示进入 CRF 解码，同时用增强实体表示与 VinVL 区域完成 grounding。
