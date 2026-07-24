# GMNER / FMNERG 阶段性实验总结

> 本文是历史实验归档，早期“当前最优”表述不再更新。当前 RoBERTa 完整链路、
> 可运行配置和正式 test 结果以 [README.md](../README.md) 为准。

## 2026-07-24 M3.6A-r2 最终归档

- 10 个 full-chain OOF fold 已完成，7000 条 heldout 记录完整覆盖，
  `test_accessed=false`。
- M3.3A OOF Train micro F1：Span `0.870900`、MNER `0.811690`、
  EEG `0.651135`、GMNER `0.610849`。
- NULL Release 在严格 OOF 下没有超过 epoch-0 KEEP，M3.6A-r2 判定 no-go；
  正式 Dev/Test 保持 `0.621316/0.61529`。
- 完整阶段实现已封存在 Git tag `m3.6a-r2-oof-complete`；主分支只保留最小
  复现实现与 OOF 审计基础设施。

## 2026-07-22 归档补充

- Absolute Region Reliability 的 hard A/B AUROC 最高约 `0.6393`，未达到接入
  Visibility 的 `0.70` 门槛。
- 风险 checkpoint 的 dev 净修正为 `+12`，但只代表小规模高精度尾部，没有进入
  正式 test 链路。
- 该分支代码已在工作区清理；若重新研究区域绝对可靠性，应采用实体 crop、局部
  crop 和全图多尺度视觉证据，不恢复旧 MLP。

## 2026-07-23 M3.4A 归档补充

- 冻结 SigLIP 2 多尺度旁路的同口径 dev 结果为：VinVL-only AUROC `0.5773`、
  SigLIP2-only `0.5759`、Fusion `0.6003`。
- Fusion balanced accuracy 为 `0.6241`、NULL preservation 为 `0.9932`、风险净
  纠错为 `+9`，说明存在有限互补性，但未达到 `0.70/+15` 接入门槛。
- M3.4A 判定 no-go，不进入 M3.4B，不读取 test；正式 test GMNER 继续为 `0.61529`。
- 旧 `0.6393` 只冻结到 Fine Adapter；M3.4A 使用 Evidence Visibility 最终 KEEP，
  A/B 数量分别从 `106/64` 变为 `105/73`，两者不能作为同口径升降比较。

## 2026-07-23 M3.5B 集合动作 Oracle

- 冻结当前 RoBERTa R36 Fine + Evidence Visibility 链，dev 基线精确复现为
  `GMNER=0.621316`，未读取 test。
- sharing-aware 独立候选 Oracle 从 Top-1 到 Top-16 的净修正依次为
  `+233/+326/+372/+403/+411`；这是 gold-aware 上限，不是可部署结果。
- Top-16 上限含 `130` 个 TO_NULL 和 `281` 个 TO_REAL；Top-4 已覆盖 242 个
  TO_REAL，上探 Top-16 仅多 39 个，因此后续候选预算优先使用 Top-4。
- 66 条记录存在当前真实区域碰撞，但直接落在碰撞实体上的可修复 TO_REAL 只有
  24 个。集合推理存在信号，但不是全部 `+411` 的来源。
- 严格真实区域容量为 1 的匹配比允许区域共享的 Oracle 少 8 个正确三元组，说明
  不应使用全局硬 Hungarian；后续只能使用带 NULL、允许复用的软容量约束。
- 后续主线固定为 Fine Top-4 的分层动作策略：先判定 KEEP/TO_NULL/TO_VISIBLE，
  再在 TO_VISIBLE 条件下选择真实区域。旧控制器虽有固定零分 KEEP，但仍是平坦动作
  竞争；新结构的关键是可学习 KEEP、分层条件解码和保护当前正确三元组。
- Set Verifier 只作为多人/同类/碰撞高风险记录的残差模块；DINOv2 只做困难切片的
  冻结局部外观诊断，两者均不得直接读取 test 或替代普通 pair-level 主干。

## 2026-07-23 M3.6A 工程实现

- 旧 Action Controller 的归因修正为：固定且不可学习的 KEEP 效用、NULL/真实区域
  平坦竞争，以及多来源 Top-k 并集噪声；旧实现并非完全没有 KEEP。
- 新结构按 `KEEP/TO_NULL/TO_VISIBLE -> Fine Top-4` 分层，并按当前 NULL/visible
  状态屏蔽重复动作。KEEP 是显式、状态相关且可学习的类别。
- 只有 span/type 正确、gold 映射明确且目标动作可由 Fine Top-4 实现的样本参与动作
  分类；不可操作样本只用于保护冻结预测。
- Epoch 0 通过正负 bias 和零初始化实现真实全 KEEP，而非推理 bypass；训练入口会对
  动作数、逐记录预测、预测数量及 MNER/EEG/GMNER 做硬恒等检查。
- 当前本地回归为 `150 passed, 3 skipped`；云端真实缓存冒烟、epoch-0 恒等审计和完整
  dev 训练均通过工程检查。
- 完整 dev epoch 0 精确复现 `GMNER=0.6213161082`；epoch 1/2/3 分别降至
  `0.61849/0.59386/0.61163`，净纠错为 `-7/-68/-24`，best 保持 epoch 0 no-op。
- 错误主要来自 TO_NULL；TO_REAL 独立风险前缀最多 `+8`，但未转化为最终正收益。
  M3.6A 第一版因此 no-go，未读取 test，正式 test GMNER 仍为 `0.61529`。
- 当前训练缓存是 in-sample engineering cache；正式复验前应先构建对齐的 RoBERTa
  OOF R16/R36/SigLIP2 缓存，不继续基于本轮 dev 扫类别权重或阈值。

## 2026-07-23 M3.6A-r1 分支隔离

- TO_REAL-only 在 epoch 3 将 dev GMNER 从 `0.621316` 提升到 `0.623738`，18 次动作
  包含 8 FIX、2 DAMAGE、8 NEUTRAL，净修正 `+6`，KEEP 正确保护率 `0.9987`。
- 该收益全部来自 `NULL -> real` 释放；`real -> other real` 实际未执行，其独立风险
  上限仅 `+3`。因此当前正信号应定义为 NULL Release，而不是一般区域切换。
- TO_NULL-only 的 best 仍为 epoch 0 no-op；学习后出现 `-25/-14`，风险上限最多
  `+2`。新 Null-Revert 分支暂停，沿用 Evidence Visibility 的正式 NULL 决策。
- 当前 in-sample train 的 formal 正确率为 `0.8788`，dev 为 `0.7608`；TO_VISIBLE
  标签比例为 `0.0344 -> 0.1271`，KEEP 为 `0.9209 -> 0.8049`。这确认了明显的
  train/dev 状态错位。
- 下一步硬前置是 10-fold 整链 cross-fitting，而非只生成 OOF Stage1。每折必须对齐
  R16/R36、Coarse、Fine、Evidence Visibility formal state 和 SigLIP2 候选索引；
  通过分布审计后再训练状态专属二分类优势头。全程未读取 test。

## 2026-07-23 M3.6A-r2 NULL Release 收缩

- 完整 `KEEP/TO_NULL/TO_VISIBLE` 路线正式收缩为当前 NULL 实体上的
  `KEEP/RELEASE_TO_VISIBLE -> Fine Top-4`；现有 Evidence Visibility 保留既有
  NULL 回退职责，真实区域间切换暂停。
- Release Head 使用相对 KEEP 的二元 advantage；错误释放与漏释放权重为 `3:1`，
  epoch 0 由零权重和负 bias 精确保持全 KEEP。
- Policy scope 只包含当前正式解码选中的 NULL 实体，不再把所有 Stage1 span 当作
  可部署动作样本。不可修正完整三元组的当前 NULL 样本进入高代价负类。
- 正式配置强制使用 10-fold 整链 OOF 冻结特征。整链包含 Stage1、Hierarchy、
  Coarse、Fine、Evidence Visibility 和启用时的 Reliability；仅替换 Stage1 仍视为
  泄漏。
- 新增 fold proof、特征物化和十折合并校验：检查 train/heldout ID 互斥、7000 条
  完整覆盖、每折补集关系、R16/R36 fold 一致性及所有配置/缓存/checkpoint 哈希。
- 新增 Fold 0 优先的整链编排入口及 pipeline manifest。OOF 模式不再构建 Stage1 test
  Dataset，也不读取 Hierarchical test cache；六个监督模块必须共享同一 fold train ID
  摘要。Fine Top-4 在物化时显式冻结，加载后不重新排序。
- 本节记录的是 r2 启动前状态；最终十折结果见文档顶部和
  [OOF_NULL_RELEASE.md](OOF_NULL_RELEASE.md)。

本文档总结当前 GMNER/FMNERG 任务上的主线链路、已完成尝试、实验现象、方法优劣和后续判断。重点不放在具体代码细节，而放在“为什么这样做、结果如何、下一步该往哪里走”。

## 当前可靠最优链路

当前最可靠的实验链路是：

```text
FMNERG fine txt 数据
  -> multilingual BERT 文本编码
  -> CRF 做 BIO 序列标注
  -> VinVL npz 区域特征作为候选视觉区域
  -> 文本图 / 图像图 / 跨模态注意力
  -> NER + Grounding 多任务训练
  -> 输出 MNER / EEG / GMNER 三类指标
```

对应配置：

```text
configs/fmnerg_twitter10000_stage1.yaml
```

对应输出：

```text
outputs/fmnerg_twitter10000_stage1
```

当前 Stage 1 baseline 的测试集表现：

```text
MNER / Entity F1: 0.7859
EEG F1:          0.6258
GMNER / Triple:  0.5815
Grounding Acc:   0.7162
```

从目前实验看，Stage 1 虽然不是结构上最“新”的方案，但它是当前最稳、最可复现、泛化最可靠的主线。后续创新应以它为基础，而不是破坏它已经学到的文本 span、实体类型和图文对齐空间。

## 指标口径

当前使用三个核心指标：

```text
MNER  = Entity + Type
EEG   = Entity + Region
GMNER = Entity + Type + Region
```

其中：

- `entity_f1` 对应 MNER。
- `eeg_f1` 对应 EEG。
- `gmner_score` / `triple_f1` 对应 GMNER。
- `grounding_accuracy` 是 grounding 条件诊断指标，不等价于 EEG。

这点很关键：单独提高 `grounding_accuracy` 不一定能提高 EEG 或 GMNER，因为 GMNER 需要实体边界、实体类型和区域同时正确。

## 已完成尝试

### 1. 核心代码修复与 FMNERG 数据接入

完成内容：

- 支持原始 GMNER 的 `txt` 数据。
- 支持 FMNERG 的三列 `txt_fine` 数据：

```text
token coarse_label fine_label
```

- coarse label 用于主 NER。
- fine label 保留为 subtype 信息，用于后续 subtype 原型、辅助监督或分析。
- 修复训练中 invalid grounding label 导致 loss 异常的问题。
- 增加 MNER、EEG、GMNER 指标输出。
- 增加磁盘空间检查，避免 checkpoint 保存中途损坏。

优点：

- 代码基础更加稳定。
- GMNER 和 FMNERG 可以在同一套框架中切换。
- 指标口径更接近论文汇报方式。

不足：

- 这部分主要是工程基础，不直接构成模型创新。

### 2. 离线实体知识库与文本原型

最初构建的知识库包括：

```text
entity_occurrences.jsonl
mention_inventory.jsonl
ambiguous_mentions.jsonl
semantic_prototypes.pt
```

主要思想是：

- 从训练集实体中抽取 mention、上下文、实体类型。
- 对实体上下文表示进行聚类。
- 形成 type-level / subtype-level semantic prototypes。
- 希望这些原型补足模糊实体的类别判断。

实验现象：

- 原型对 MNER/type 判断有时有帮助。
- 但直接把原型 residual 加回 BERT 表示后，容易破坏原始文本-图像 grounding 空间。
- MNER 提升不能稳定转化为 EEG/GMNER 提升。

优点：

- 思路清晰，能解释为“类型语义记忆”。
- 对细粒度类别和模糊实体有一定理论价值。
- 可作为论文中的知识增强尝试或消融对象。

不足：

- 原型来自训练集统计，泛化边界有限。
- 直接融合到 token/span 表示会影响 grounding。
- 如果只是作为 bias 或 residual，作用强度有限；如果作用太强，又会伤害图文对齐。

结论：

文本原型不适合继续作为全局共享表示增强。更合理的用法是作为实体类型判断或解释模块的辅助证据，而不是直接改写用于 grounding 的文本向量。

### 3. 视觉原型与 CLIP 尝试

完成内容：

- 基于 VinVL 区域特征构造视觉原型。
- 尝试引入 CLIP subtype prompt，用文本 prompt 与实体语义进行对齐。
- 分析 `.npz` 文件结构，确认已有 VinVL 候选框、框特征、object label、attribute label。

实验判断：

- 当前代码实际视觉输入主要来自 VinVL `.npz`。
- `.npz` 已经提供候选框和 2048 维区域特征，不需要本地部署 VinVL。
- CLIP 在当前链路中没有直接替换 VinVL 的位置。
- CLIP prompt 更像外部语义解释，不是当前 grounding 主干。

优点：

- CLIP 具备开放词汇语义能力。
- 对 subtype prompt 或外部解释有一定扩展空间。

不足：

- 当前数据已有 VinVL region feature，CLIP 如果只做文本 prompt，很难直接提升 region 排序。
- 如果改成 CLIP 图像编码，需要重新处理图像、候选框裁剪或区域级编码，工程量更大。
- 当前实验中 CLIP 与原型记忆关联弱，没有形成稳定收益。

结论：

在现有框架下，CLIP 不应作为主线。除非后续明确做“区域裁剪 + CLIP region encoder”或“CLIP-based entity-region reranker”，否则保留 VinVL 更稳。

### 4. 原型门控、类型校准和辅助监督

尝试过的方向包括：

- entropy gate；
- constant gate；
- reliability score；
- type temperature calibration；
- subtype auxiliary loss；
- subtype contrastive loss；
- prototype-aware type refinement；
- 文本分支和 grounding 分支解耦。

实验现象：

- 原型参与率提高后，并没有稳定提升 GMNER。
- 辅助监督和对比学习反而使 grounding 或 GMNER 下降。
- type calibration 可以降低置信度过高的问题，但不能从根本上提升实体-区域联合判断。

优点：

- 对“基础模型过度自信”问题做了客观校准尝试。
- 能证明简单门控不是决定性瓶颈。
- 分支解耦验证了“原型影响文本判断、但不要污染 grounding”的方向。

不足：

- 门控标准本身很难准确决定哪些实体需要原型。
- 原型强介入时容易带来负迁移。
- 辅助监督优化的是中间表征，不一定优化最终 GMNER。

结论：

单纯调 gate 或 loss 权重价值有限。原型模块如果继续保留，应该服务于实体类型解释或候选证据生成，而不是作为主干表征增强。

### 5. Evidence Graph Decoder

为避免原型直接污染 BERT 表示，进一步尝试了实体级证据图：

```text
entity node
context node
type prototype nodes
region nodes
object / attribute evidence
NULL region
```

核心目标：

- 不直接改写 BERT token 表示。
- 对每个实体单独建图。
- 让实体、类型、区域、object label、attribute label 在图中交互。
- 输出 entity-region / type-region 联合分数。

实现过的训练形式：

```text
Gold-span evidence
Predicted-span evidence
Out-of-Fold predicted-span evidence
```

其中 OOF evidence 的构造流程是：

```text
train split -> 5 folds
每折用 4/5 训练 Stage 1
对 heldout 1/5 预测实体 evidence
合并 5 折 prediction
得到 train OOF predicted evidence
用 OOF evidence 训练 Stage 2
```

OOF 合并结果：

```text
records: 7000
evidence_entities: 11680
empty_records: 437
duplicate_count: 0
```

OOF Stage 2 测试表现：

```text
MNER / Entity F1: 0.7859
EEG F1:          0.6254
GMNER / Triple:  0.5804
Grounding Acc:   0.7170
```

与 Stage 1 baseline 对比：

```text
Stage 1 GMNER:   0.5815
OOF Stage 2:     0.5804
```

优点：

- 结构比简单原型融合更合理。
- 更符合“实体级解释”和“实体-区域联合推理”的论文叙事。
- OOF 方案避免了 train gold leakage，实验更严谨。
- 可以作为已验证的负结果，说明 naive evidence graph 不是充分条件。

不足：

- Dev 上有提升，但 test 不稳定。
- evidence graph 的收益没有泛化。
- 如果 evidence 节点质量来自同一个 Stage 1，可能只是重复 Stage 1 的偏差。
- 训练复杂度增加，但最终收益不足。

结论：

Evidence graph 是目前最完整的结构创新尝试，但按当前实现还不能作为最终主线。它证明了“实体级证据建模”方向有研究价值，但也说明仅靠现有 evidence 组合不能稳定突破 GMNER。

## 方法对比

| 方法 | 主要作用 | 优点 | 问题 | 当前结论 |
|---|---|---|---|---|
| Stage 1 BERT+CRF+VinVL | 基础 MNER + Grounding | 稳定、泛化最好 | 创新性有限 | 当前可靠主线 |
| 文本原型直接融合 | 增强 type 判断 | 有时提升 MNER | 破坏 grounding 空间 | 不作为主线 |
| 视觉原型 | 建模视觉类别中心 | 可解释 | 对 region 排序帮助弱 | 保留分析价值 |
| CLIP prompt | 外部语义解释 | 开放词汇 | 与 VinVL grounding 主干脱节 | 暂不作为主线 |
| 门控/校准 | 控制原型介入强度 | 能分析过度自信 | 无稳定收益 | 不继续调参 |
| 辅助监督/对比学习 | 强化 subtype 表示 | 论文叙事容易 | GMNER 下降 | 不作为主线 |
| Evidence Graph | 实体级证据推理 | 结构更完整 | test 不稳定 | 保留为尝试 |
| OOF Evidence | 避免 gold leakage | 实验严谨 | 仍未提升 | 证明负结果可信 |

## 当前瓶颈判断

当前主要瓶颈不是 NER。

原因：

- Entity F1 已经接近 0.786。
- Type 相关问题存在，但不是 GMNER 的最大损失来源。
- EEG 和 GMNER 的差距说明 region decision 仍是关键问题。

当前主要瓶颈是：

```text
实体已经找到了，但对应图像区域没有稳定选对。
```

换句话说，模型需要更强的 entity-region ranking，而不是继续在 BERT 表示上叠加原型。

## 后续建议方向

### 推荐方向：Entity-Region Reranker

下一步建议从 Stage 1 出发，冻结或半冻结主干，单独训练一个实体-区域重排序模块：

```text
predicted entity span
  + span hidden state
  + local context
  + candidate region feature
  + bbox geometry
  + VinVL object label / attribute label
  -> entity-region compatibility score
```

优化目标直接对齐：

```text
Entity 正确 + Region 正确
```

也就是优先优化 EEG，再进一步推动 GMNER。

为什么这个方向更合理：

- 不破坏 Stage 1 的 NER 能力。
- 直接作用于当前最大瓶颈：region 排序。
- 可以利用 VinVL 已有 object/attribute 信息。
- 比原型融合更贴近 GMNER 的最终判定条件。

### 可保留的论文叙事

当前工作仍然有价值，可以总结为：

```text
1. 从 GMNER 扩展到 FMNERG，支持 fine-grained entity 标签。
2. 系统验证了文本原型、视觉原型、门控、辅助监督、证据图等多种知识增强方式。
3. 实验证明直接原型融合容易造成 grounding 退化。
4. OOF evidence 验证了实体级证据图在无泄漏条件下的真实效果。
5. 当前结论指向更直接的 entity-region reranking 结构。
```

这个叙事比单纯说“原型没用”更合理：不是知识没有价值，而是知识不能粗暴进入共享表示空间；它更适合进入候选区域选择、解释或重排序阶段。

## 当前可汇报结论

可以在组会中这样总结：

```text
目前最稳的系统仍是 BERT+CRF+VinVL 的 Stage 1 baseline。
在此基础上，我们尝试了文本原型、视觉原型、CLIP prompt、门控校准、辅助监督、对比学习和实体级 evidence graph。
实验表明，原型类知识能影响实体类型判断，但容易破坏 grounding；证据图结构更合理，但 OOF predicted evidence 下没有带来 test 提升。
因此当前瓶颈不在实体识别本身，而在实体与视觉区域的精确匹配。
下一步应转向 entity-region reranker，让创新点直接服务于 EEG/GMNER 的核心错误来源。
```
