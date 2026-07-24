# Hierarchical Record Verifier

本文档只描述当前保留的层次化 verifier、复现步骤和已经确定的实验边界。

## 1. 结构

当前主链使用 `RoBERTa-base, max_length=128` 的 Stage1 产生 span、type 和
region/NULL。层次化模型不把 type、NULL 和真实
region 放进一个平坦联合空间，而是分解为：

```text
P(s, c, r)
= P_entity(s)
  P_type(c | s)
  P_visibility(v | s, c)
  P_region(r | s, c, v=visible)
```

当前正式实验固定 Stage1 type：

```text
Stage1 spans
  -> Reject / Entityness
  -> fixed Stage1 type
  -> Visibility
       -> NULL
       -> Visible -> real-region residual ranker
  -> non-overlapping interval decode
```

NULL 不进入真实区域 softmax。Visibility 使用非对称双阈值：

- Stage1 为 NULL：`p_visible >= 0.80` 才切换为 visible；
- Stage1 为 visible：`p_visible <= 0.20` 才切换为 NULL；
- 中间区域保持 Stage1 决策。

真实区域残差为：

```text
z_final = z_stage1 / temperature + residual_scale * delta_z
```

覆盖 Stage1 真实框还必须满足 logit margin 和 probability margin，避免 raw
ranker 的高诊断上限直接破坏部署结果。

## 2. 训练目标

```text
L = lambda_entity * L_entity
  + lambda_visibility * L_visibility
  + lambda_multi * L_multi_positive
  + lambda_iou * L_iou_soft
  + lambda_hard * L_base_wrong
  + lambda_preserve * L_base_correct
```

- `L_entity`：候选 span 二分类；
- `L_visibility`：visible/NULL 二分类；
- `L_multi_positive`：所有 IoU 达标框共同作为正例；
- `L_iou_soft`：连续 IoU 软排序目标；
- `L_base_wrong`：纠正 Stage1 错误 top-1；
- `L_base_correct`：保护 Stage1 已正确的真实框。

gold visible 但 VinVL 候选中没有 IoU 达标框时，只训练 visibility，不训练
region ranking。

## 3. 候选缓存

层次化缓存必须包含：

```text
fixed_type_ids
base_region_indices
base_region_scores
region_iou_targets
```

构建命令：

```bash
cd ~/gmner

for split in train dev test; do
  PYTHONPATH=. python scripts/build_record_candidate_cache.py \
    --config configs/fmnerg_twitter10000_stage1.yaml \
    --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
    --split ${split} \
    --output knowledge/record_candidates/roberta128/fmnerg_${split}_hierarchical.pt \
    --k-best 6 \
    --max-span-candidates 12 \
    --top-m-types 3 \
    --boundary-shift 0 \
    --boundary-penalty 0.25 \
    --device cuda
done
```

当前正式解码只使用 Stage1 span。缓存仍保留其他来源，供候选召回与 Oracle
诊断使用。

## 4. 训练与评估

```bash
PYTHONPATH=. python scripts/train_hierarchical_record_verifier.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml

PYTHONPATH=. python scripts/evaluate_hierarchical_record_verifier.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml \
  --checkpoint outputs/fmnerg_roberta128_hierarchical_record_verifier/best_model.pt \
  --split dev \
  --output outputs/fmnerg_roberta128_hierarchical_record_verifier/dev_metrics.json
```

所有阈值只能在 dev 上确定。正式论文实验还需要 OOF train cache、多随机种子
和均值/标准差。主配置设置 `evaluate_test_after_training: false`，只能在 dev
确定最终 checkpoint 和解码阈值后进行一次 test。

## 5. 已验证结果

旧 mBERT 层次化链路作为历史对照：

| Model | Span F1 | MNER F1 | EEG F1 | GMNER F1 |
| --- | ---: | ---: | ---: | ---: |
| Stage1 bypass | 0.84918 | 0.78593 | 0.62582 | 0.58154 |
| Hierarchical verifier | **0.85311** | **0.78951** | **0.63526** | **0.59034** |
| Delta | +0.00393 | +0.00358 | +0.00945 | +0.00880 |

Test 上 GMNER 正确三元组从 1471 增至 1485。Grounding 修正同时为正：

```text
visible: 27 corrected - 24 damaged = +3
NULL:    33 corrected - 21 damaged = +12
```

当前 RoBERTa 主链的一次性 test 为：

| Model | Span F1 | MNER F1 | EEG F1 | GMNER F1 |
| --- | ---: | ---: | ---: | ---: |
| RoBERTa-128 Stage1 bypass | 0.86702 | 0.81586 | 0.62683 | 0.59168 |
| RoBERTa hierarchical verifier | **0.86980** | **0.81843** | 0.64431 | 0.60784 |
| RoBERTa Fine Adapter baseline | **0.86980** | **0.81843** | 0.64980 | 0.61333 |
| RoBERTa Evidence Visibility | **0.86980** | **0.81843** | **0.65216** | **0.61529** |

RoBERTa 层次化 verifier 已使用独立的 `roberta128` 候选缓存完成重建。当前 dev
结果如下：

| Model | Span F1 | MNER F1 | EEG F1 | GMNER F1 |
| --- | ---: | ---: | ---: | ---: |
| RoBERTa Stage1 bypass | 0.87072 | 0.81474 | 0.64599 | 0.60733 |
| RoBERTa hierarchical verifier | **0.87283** | **0.81671** | **0.65442** | **0.61526** |
| Delta | +0.00211 | +0.00197 | +0.00843 | +0.00793 |

最佳 checkpoint 来自 epoch 3。层次解码在 dev 上净增加 16 个正确三元组；其中
NULL 修正净收益为 `66-30=+36`，可见区域修正为 `12-31=-19`。这说明总体提升
成立。最终 test 上修正 76 个、损伤 41 个，净增加 35 个正确三元组；NULL
修正净收益为 `67-21=+46`，可见区域修正为 `16-24=-8`。后续仍应优先降低
可见区域损伤。

## 6. 已停止分支

| Branch | Dev/Test finding | Decision |
| --- | --- | --- |
| Flat verifier | Test GMNER 0.58546 | 被层次化模型取代 |
| External knowledge / prototypes | 类型局部改善，最终净修正不稳定 | 停止主线接入 |
| Multiscale frozen residual | 输出几乎不改变 Stage1 | 停止 |
| Real-to-real utility | Candidate oracle 仅 +2 | 不运行 test |
| Independent action controller | Dev 净修正 -9，风险曲线最多 +3 | 停止 |
| Listwise frozen policy | 最优仍为 Epoch 0；训练后净修正 -60 | 停止 |

统一动作 Oracle 的 `+342` 只证明正确候选存在，不证明冻结表示能区分 FIX 与
DAMAGE。冻结策略已经验证失败，不再通过阈值、类别权重或更深 action head
继续调参。

## 7. 下一阶段边界

下一阶段先执行 Milestone 3.0 visible-region oracle：在不改正式 R16 配置的
前提下，用同一 RoBERTa checkpoint 构建 dev R36 诊断缓存，并输出：

```text
R_region@16 / R_region@36
gold only covered by R36
Stage1 wrong + gold-in-R16
Verifier corrected / damaged
A candidate missing / B base misrank / C verifier damage / D false-NULL
```

只有完成上述诊断后才修改 grounding 表示：

```text
冻结或低学习率：RoBERTa + CRF span/type 主干
允许更新：region projection、cross-modal interaction、visibility、region scorer
联合监督：原 grounding + multi-positive IoU + hard negative + base preservation
最终解码：继续使用已验证的 balanced hierarchical decoder
```

若 `R36-R16 >= 3` 个百分点，先实现 top36→top12/16 coarse-to-fine；否则先
实现 correction-preservation Grounding Adapter。两种情况都不再增加独立 Action
Controller。

RoBERTa dev Oracle 已完成：

| Diagnostic | Count / Recall |
| --- | ---: |
| Visible gold | 991 |
| Gold covered by R16 | 827 / 0.83451 |
| Gold covered by R36 | 891 / 0.89909 |
| Newly covered by R36 | 64 / +0.06458 |
| Stage1 wrong, gold available in R16 | 219 |
| Stage1 wrong, gold available only in R36 | 57 |
| Final false-NULL with gold available in R16 | 140 |
| Remaining real-region misranking in R16 | 97 |

R36 的覆盖增量超过阈值，但直接 R36 Stage1 的 dev GMNER 只有 `0.60004`，低于
R16 的 `0.60733`。因此不能直接把正式分支替换为 36-way softmax。Milestone 3.1
先单独验证 recall-preserving coarse selector；只有固定 Top-16 预算下的覆盖和
保护指标达标，Milestone 3.2 才训练 correction-preservation Grounding Adapter。

## 8. Milestone 3.1：召回保持型粗筛

输入为同一 RoBERTa Stage1 生成的 R36 候选缓存。粗筛使用预测 span 表示、预测
type、VinVL region feature、detector score、bbox、type-object compatibility 以及
Stage1 region score/rank。它只输出实框候选分数，不处理 NULL，也不改变最终三元组。

候选策略：

```text
detector Top-16                         诊断基线
Stage1 base-score Top-16                非学习重排对照
learned Top-16                          独立粗筛
Stage1 entity-conditioned Top-8
  + learned non-duplicate Top-8         保留式主策略
Stage1 entity-conditioned Top-10
  + learned non-duplicate Top-6         更保守策略
```

训练监督按候选覆盖而非 Stage1 top-1 正误划分：

```text
promotion:    gold 不在 detector Top-16，但存在于 R36
preservation: gold 已在 detector Top-16
```

损失由多正例候选 NLL、IoU soft target、promotion hard-negative margin 和
preservation margin 组成；两组分别求均值后平衡，避免 64 个 R36-only 样本被原
Top-16 已覆盖样本淹没。checkpoint 按 union Top-16 Recall 保存，并以原 Top-16
保留率打破平局。

验收指标固定为：

```text
raw_detector_r16_recall / raw_detector_r36_recall
learned/union recall@16
new_gold_promoted / gold_dropped
top16_preservation
base_wrong_corrected / base_correct_preservation
average_candidate_count
```

Go/no-go：union Recall@16 目标约 `0.87`，原 Top-16 正确框保留率至少 `0.98`，
平均候选数不超过 16。未达标时只调整候选训练，不运行 test，也不进入 fine adapter。

dev 实验已完成：

```text
visible gold / selector eligible / R36 recoverable = 991 / 910 / 827
detector Top-16 recall (eligible)                  = 0.84505
Stage1 base-score Top-16 recall                    = 0.88791
learned Top-16 recall                              = 0.90769
base Top-8 + learned Top-8 recall                  = 0.90769
base Top-8 + learned Top-8 Top-16 preservation     = 1.00000
new gold promoted / old gold dropped               = 57 / 0
average candidates                                 = 15.68
```

Top8+8 覆盖 826/827 个 R36 可恢复样本，同时没有丢失原 Top-16 已覆盖样本；相对
非学习的 Stage1 base-score Top-16 仍提升 1.98 个百分点，因此收益不能仅归因于
Stage1 分数重排。Milestone 3.1 判定通过，Milestone 3.2 固定采用 Top8+8。

动作标签如继续使用，只作为训练期辅助监督，不再作为独立推理控制器。

## 9. Milestone 3.2：可见区域细排

M3.2 使用已通过验收的 `Base Top-8 + Learned Top-8`，但不替换正式 R16
层次模型。R16 模型固定 span、type、Reject 和 Visibility；只有 baseline 已判为
visible 的实体允许由 fine adapter 改写 real-region index，baseline NULL 保持
不变。因此本阶段的 GMNER 增量可以明确归因于区域排序。

```text
R16 hierarchical decision (frozen) -----> KEEP span/type/visibility
R36 coarse selector (frozen) -----------> Base8 + Learned8
cached span/region features ------------> trainable adapters
explicit pair interaction --------------> bounded residual
calibrated base + coarse priors --------> final real-region rank
```

总损失为：

```text
L = L_multi_positive
  + 0.2 L_iou_soft
  + 1.0 L_correction_margin
  + 0.5 L_preservation_margin
  + 0.05 L_preservation_residual
```

dev 报告必须同时给出总体 GMNER/EEG、`base_wrong_corrected`、
`base_correct_damaged`、`visible_net_correction`、Top16 覆盖切片、promoted
恢复数、候选来源占比、预测改动率和 residual 均值。checkpoint 以 GMNER 为主；
visible 净修正、保护率和 promoted 恢复率只作为平局项。

该阶段只允许 dev 验收，不读取 test。当前非 OOF train cache 只用于工程验证；
正式实验必须以 OOF RoBERTa Stage1 分数重建成对的 R16/R36 train cache。

当前工程验证的 best epoch 为 1：

```text
R16 baseline GMNER / EEG       0.615260 / 0.654421
epoch-0 dual-prior GMNER       0.618086
fine-adapter GMNER / EEG       0.618894 / 0.658054
visible corrected / damaged    22 / 13  (net +9)
base-correct preservation      0.974659
promoted raw top-1 recovered   32 / 57
promoted deployed recovered    14 / 27
span F1 delta / MNER F1 delta  0 / 0
```

因此 M3.2 最低 go/no-go 已通过，而且 fine scorer 相对固定双先验仍有小幅独立
收益。下一步不是在当前 dev 上继续扫描 prior/residual 超参数，而是构建成对的
RoBERTa OOF R16/R36 train cache，验证 correction-preservation 学习能否跨种子
保持净正收益；在此之前不运行 test。

## 10. Milestone 3.3A：区域证据辅助 Visibility

M3.2 部署漏斗进一步拆解为：

```text
Fine top-1 correct but M3.2 final NULL              113
其中 fixed type correct                             106
其中 promoted + fixed type correct                   18
Fine top-1 wrong and M3.2 final NULL                 74
正确 NULL 且当前保持 NULL                          1108
```

因此 M3.3A 不是阈值扫描，而是冻结全部既有模块后训练独立的 Evidence Visibility
Head。其输入由 Fine Adapter 的实体/区域/type 状态和 22 维部署期可用证据组成，
包括概率 margin、归一化熵、三套 rank、base/coarse/fine agreement、候选来源、
detector score、promoted 标记和 type-object compatibility。输出为冻结 Visibility
logit 上的有界残差：

```text
l_final = l_hierarchy + 4 * tanh(F_visibility(detached_region_evidence))
```

解码继续使用原 `0.80 / 0.20` 非对称双阈值。最终 visible 时固定采用 M3.2 Fine
top-1，最终 NULL 时采用 R36 NULL；span、type、Reject 和区域排序均不会改变。
最后一层零初始化确保 epoch 0 与 M3.2 完全一致。

训练监督只强化可转化为最终指标的动作，并保护原有正确决策：

```text
L = L_grouped_BCE
  + L_visible_correction
  + L_null_preserve
  + 0.5 L_uncertain_keep_KL
  + 0.05 L_preservation_residual
```

评估必须报告 promoted 部署漏斗、Fine-correct-but-NULL、Fine-wrong switch、证据
agreement/margin 切片、visible/NULL 各自 corrected/damaged/net、NULL 正确保护率，
以及 M3.2 baseline consistency。M3.3A 仍只做 dev 工程验收，不读取 test；正式
结果仍要求成对的 OOF R16/R36 train cache。

```bash
PYTHONPATH=. python scripts/train_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml
```

最低 go/no-go 为：GMNER 超过 `0.61889`、visible net 为正、NULL net 非负、NULL
正确保护率至少 0.97，且 Span/MNER 不变。只有 M3.3A 稳定通过后，才允许以低权重
解冻 Fine Adapter 进入 M3.3B。

首轮同分布 train-cache 工程结果为：

```text
best epoch                               1
M3.2 -> M3.3A GMNER                      0.618894 -> 0.621316
M3.2 -> M3.3A EEG                        0.658054 -> 0.660880
GMNER corrected / damaged                14 / 8  (net +6)
visible corrected / damaged              4 / 3   (net +1)
NULL corrected / damaged                 12 / 6  (net +6)
NULL correct preservation                0.994585
Fine-correct baseline NULL                113
Fine-correct final NULL                   112
type-correct Fine-correct final NULL      105  (baseline 106)
promoted final triple                     13   (M3.2 14)
span F1 delta / MNER F1 delta             0 / 0
```

该轮达到 GMNER、visible net、NULL net、NULL 保护率和额外净三元组门槛，但没有
显著降低核心阻塞项，且 promoted triple 下降。因此 M3.3A 只证明 Evidence
Visibility Head 可以产生净正残差修正，尚未证明其主要利用了新增区域证据。M3.3B
继续暂停；不得把该 dev 工程结果当作 OOF 或 test 结果。

固定 checkpoint 和 dev 双阈值后完成的一次性 test 为：

```text
M3.2 -> M3.3A GMNER                      0.613333 -> 0.615294
M3.2 -> M3.3A EEG                        0.649804 -> 0.652157
M3.2 -> M3.3A MNER                       0.818431 -> 0.818431
GMNER corrected / damaged                15 / 10  (net +5)
visible corrected / damaged              3 / 9    (net -6)
NULL corrected / damaged                 13 / 1   (net +12)
NULL correct preservation                0.999101
promoted final triple                     15
```

因此当前正式最优 test GMNER 为 `0.61529`。收益仍主要来自 NULL，真实区域分支
继续净损伤；该 test 只用于最终报告，不再参与阈值、checkpoint 或结构选择。

## 11. Milestone 3.3A.1：证据释放归因

固定 M3.3A best checkpoint 后，dev 分组为：

```text
A  base NULL    + Fine top-1 correct       106
B  base NULL    + Fine top-1 wrong         122
C  base visible + Fine top-1 correct       516
D  base visible + Fine top-1 wrong         127
E  gold NULL    + base NULL               1017
F  gold NULL    + base visible             135
```

A 中只有 4 个被当前 head 释放，18 个 promoted A 中没有一个被释放。但 `residual
scale=4.0` 时，A 有 95/106 可跨越与其 Stage1 来源对应的双阈值，promoted A 有
17/18 可达；A 组所需残差中位数为 1.78。由此排除“统一残差上限过小”作为主因。

部署期完美 NULL→visible 控制器可净修正 106 个三元组，对应 dev GMNER 上限约
0.66169；再包含 F 组 TO_NULL 动作时，联合 Visibility 动作 Oracle 为 +241。因此
动作空间本身具有足够上限。

现有 22 维标量证据加候选来源的五折线性探针结果为：

```text
A vs all B                 AUROC 0.6856 / balanced accuracy 0.6328
A vs candidate-covered B   AUROC 0.6167 / balanced accuracy 0.5503
A vs B + gold-NULL         AUROC 0.7965
current head residual,
  A vs candidate-covered B AUROC 0.5611
```

包含 region-missing 的 B 较容易识别，但真正需要判别的“正确高分框 vs 错误高分框”
几乎不可分。故当前属于 Case C：不继续放大 residual、不只增加 rescue loss，也不
解冻 Fine Adapter；先新增只做绝对区域有效性学习的 Region Reliability Head，并
用 IoU 正例、高分错误框、gold NULL 框和同图其他实体框监督。其困难 A/B dev
AUROC 未明显提升前，不得接入最终 Visibility 或运行 test。

## 12. 已归档：VinVL-only 区域绝对可靠性

旧版隔离实验未接入当前链路，其只冻结到 Fine Adapter，hard A/B AUROC 最高约为
`0.6393`。新的 M3.4A 以 Evidence Visibility 最终 KEEP 重新定义 A/B，并增加冻结
SigLIP 2 的 mention/context/type 与 local/context/global 三尺度旁路。其同口径结果
为 VinVL `0.5773`、SigLIP2 `0.5759`、Fusion `0.6003`；Fusion 风险净纠错为 `+9`，
仍未达到 `0.70/+15` 门槛。M3.4A 因此 no-go，不进入 M3.4B，也不读取 test。旧版与
M3.4A 的 A/B 集合不同，不能写成直接性能回退。完整口径、结果和 dev 切片入口见
[README_SIGLIP2_REGION_RELIABILITY.md](README_SIGLIP2_REGION_RELIABILITY.md)。

## 13. Milestone 3.5：实例对应与集合级诊断

M3.4A 的 dev 切片显示，单人物和无人物场景的 hard A/B AUROC 分别约为
`0.7216/0.7368`，但多人物场景仅为 `0.5234`；Fine 与 gold 同属一个 VinVL
object 类别时仅为 `0.4874`。因此当前问题应表述为“同类实例与文本实体的对应关系
不确定”，而不是一般图像语义缺失。

在引入 DINOv2 或新的集合模型前，M3.5B 先固定当前 R36 Fine + Evidence
Visibility 链，在 dev 上测量三种口径：

```text
deployed independent decoding
sharing-aware independent candidate oracle
strict real-region capacity-1 matching diagnostic
```

独立候选 Oracle 允许真实区域复用，是同一动作集合的上限；严格匹配不可能超过它。
二者差值用于衡量无条件 Hungarian 一对一假设会损失多少合法共享区域，而不是把严格
匹配误写成更高的 Oracle。脚本同时拆分单实体、多实体、当前区域碰撞和非碰撞记录的
候选净收益；只有碰撞/多实体切片具有足够额外空间时，才支持实现 Relation-aware Set
Verifier。大量收益若来自单实体记录，则下一步仍应优化 pair ranker，而不是集合解码。

```bash
PYTHONPATH=. python scripts/analyze_record_set_assignment_oracle.py \
  --config configs/fmnerg_twitter10000_siglip2_reliability_vinvl.yaml \
  --top-k 1,2,4,8,16 \
  --device cuda \
  --output outputs/fmnerg_roberta128_evidence_visibility/dev_set_oracle.json
```

该脚本没有 `--split test` 接口，不加载 SigLIP 2 特征，也不改变正式输出。DINOv2
仅在该 Oracle 明确集合推理收益来源后作为局部外观旁路诊断；它不能被解释为实体身份
识别器，也不会直接接入正式 test 链路。

首次冻结链 dev 结果精确复现当前基线 `GMNER=0.621316`：

| Fine 动作预算 | 独立 Oracle 净修正 | 严格容量净修正 | 共享损失 | 多实体净修正 | 碰撞记录净修正 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| Top-1 | +233 | +218 | 15 | +149 | +35 |
| Top-2 | +326 | +317 | 9 | +217 | +52 |
| Top-4 | +372 | +366 | 6 | +254 | +60 |
| Top-8 | +403 | +395 | 8 | +272 | +61 |
| Top-16 | +411 | +403 | 8 | +275 | +61 |

Top-16 的 `+411` 由 `130` 个 TO_NULL 和 `281` 个 TO_REAL gold-aware 上限组成，
不能视为可部署收益。更严格的集合信号是：66 条当前区域碰撞记录中，只有 24 个碰撞
实体可由 Top-4/8/16 的 TO_REAL 动作直接修复。另有 7 条记录出现严格容量冲突，硬
一对一匹配相对允许共享的独立 Oracle 少 8 个正确三元组。

因此 M3.5B 的结论是：候选空间充足，存在值得验证的集合关系信号，但大部分 Oracle
空间仍不是碰撞特有收益。后续若实现集合模型，应采用带 NULL、允许复用且只对高风险
同类竞争施加软容量惩罚的 Set Verifier；不得使用全局硬 Hungarian。候选预算优先
Top-4，Top-8 仅作覆盖对照，Top-16 的边际收益过小。

## 14. 后续主线：分层 Top-4 动作验证

M3.5B 将后续固定为 `KEEP/TO_NULL/TO_VISIBLE`，并只在 TO_VISIBLE 下选择 Fine
Top-4。这里需要修正旧实验归因：旧 Action Controller 已通过固定 `KEEP=0` 显式
加入 KEEP；失败原因不是 KEEP 完全缺失，而是 KEEP 不可学习、NULL 与所有真实区域
平坦竞争，并且真实区域来自 fused/residual/base 三套 Top-k 并集。

M3.6A 改为两层条件策略：

```text
current deployed decision
  -> Layer 1: KEEP / TO_NULL / TO_VISIBLE
  -> Layer 2: Fine Top-4 real-region choice, only when TO_VISIBLE
```

动作必须按当前状态去重：当前为 NULL 时屏蔽 TO_NULL；当前为 visible 时，第二层屏蔽
当前区域，因为保留当前区域已经由 KEEP 表示。Span/type 错误和 gold real region 不在
Top-4 的样本只用于诊断，不作为随机切换监督。当前正确时 KEEP 是正动作；当前错误且
gold 为 NULL 时 TO_NULL 为正动作；当前错误且 Top-4 含 gold real region 时
TO_VISIBLE 与对应区域为正动作。

训练目标至少包含分层多正例策略损失、base-correct preservation、base-wrong
correction、动作 margin、正确决策蒸馏和残差幅度约束。最后层应零初始化并给 KEEP
非负初始优势，使 epoch 0 精确复现当前 `GMNER=0.621316`，而不是依靠推理阈值近似
回退。

M3.6A 已实现为独立模块，不修改或覆盖当前正式 checkpoint：

```text
gmner/models/layered_action_verifier.py
gmner/losses/layered_action_verifier_loss.py
gmner/engine/layered_action_verifier_evaluator.py
configs/fmnerg_twitter10000_layered_action_verifier.yaml
configs/fmnerg_twitter10000_layered_action_to_real_only.yaml
configs/fmnerg_twitter10000_layered_action_to_null_only.yaml
scripts/train_layered_action_verifier.py
scripts/evaluate_layered_action_verifier.py
scripts/audit_layered_action_distribution.py
```

Layer 1 直接学习三个状态相关 logits；Layer 2 在 Fine 单一排序的 Top-4 内使用多正框
目标，不再构造 fused/residual/base 并集。损失由分组 Layer-1 CE、Layer-2 多正例
NLL、KEEP margin、correction margin、不可操作样本 preservation 和 Layer-2 residual
约束组成。训练脚本会先执行完整 epoch-0 恒等审计：动作数、区域变化、预测数量变化
必须为零，逐记录预测及四类 F1 必须完全一致；完整 dev 还需复现 `0.621316`。任一
条件不满足即中止训练。

```bash
PYTHONPATH=. python scripts/train_layered_action_verifier.py \
  --config configs/fmnerg_twitter10000_layered_action_verifier.yaml

PYTHONPATH=. python scripts/evaluate_layered_action_verifier.py \
  --config configs/fmnerg_twitter10000_layered_action_verifier.yaml \
  --checkpoint outputs/fmnerg_roberta128_layered_action_verifier/best_model.pt \
  --split dev \
  --output outputs/fmnerg_roberta128_layered_action_verifier/dev_metrics.json
```

评估器分别输出 TO_NULL 与 TO_REAL 的 corrected/damaged/net、KEEP 正确保护率、两层
动作准确率、碰撞/非碰撞净修正和不依赖执行阈值的风险覆盖曲线。该入口故意不提供
test split。

首轮非 OOF 工程验收已完成。Epoch 0 精确复现 `GMNER=0.6213161082`、
`EEG=0.6608800969`、`MNER=0.8167137667`，动作数和预测变化均为 0。训练后：

| Epoch | GMNER | 总净修正 | TO_NULL net | TO_REAL net | KEEP 保护率 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.61849 | -7 | -7 | 0 | 0.9844 |
| 2 | 0.59386 | -68 | -56 | -12 | 0.9019 |
| 3 | 0.61163 | -24 | -24 | 0 | 0.9662 |

训练在 epoch 3 早停，best 为 epoch 0 no-op，`go_no_go=0`。TO_VISIBLE 的独立风险
前缀最多达到 `+8`，但 Layer 1 同时偏向错误 TO_NULL，无法形成稳定最终收益。因此
M3.6A 第一版工程通过、方法验收未通过，仍不得读取 test。当前训练使用 in-sample
engineering cache；下一次正式复验必须先解决 RoBERTa OOF R16/R36 与 SigLIP2
缓存对齐，不能围绕本次 dev 继续扫描类别权重或执行阈值。

### 14.1 M3.6A-r1 分支隔离结果

`action_mode` 现在从模型有效动作掩码、监督标签和最终解码三处同时隔离分支。禁用动作
不会作为 CE 标签进入另一分支；对应样本只保护当前正式决策。两种模式均通过 epoch 0
逐记录恒等检查，随后在相同 seed、冻结链和预算下完成 dev-only 工程消融：

| 模式 | Best epoch | GMNER | EEG | 执行 | FIX | DAMAGE | NEUTRAL | Net | KEEP 保护率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TO_REAL-only | 3 | 0.623738 | 0.662495 | 18 | 8 | 2 | 8 | +6 | 0.998700 |
| TO_NULL-only | 0 | 0.621316 | 0.660880 | 0 | 0 | 0 | 0 | 0 | 1.000000 |

TO_REAL-only 的风险曲线上限为 `+8`，达到 `+5` 工程门槛且不是极窄单动作结果。但迁移
拆分显示，实际 `+6` 全部来自 NULL Release：

| TO_VISIBLE 状态迁移 | 实际执行 | FIX/DAMAGE/NEUTRAL | 实际 Net | 风险上限 |
| --- | ---: | ---: | ---: | ---: |
| NULL -> real | 18 | 8/2/8 | +6 | +7 @ 33 |
| real -> other real | 0 | 0/0/0 | 0 | +3 @ 16 |

因此当前可保留信号是 `NULL Release Head`，不是泛化的区域切换能力。TO_NULL-only 的
epoch 2/3 分别为 `-25/-14`，风险上限最多 `+2`，新 Null-Revert Head 暂停；正式
Evidence Visibility 的 NULL 决策保持不变。

分布审计入口：

```bash
PYTHONPATH=. python scripts/audit_layered_action_distribution.py \
  --config configs/fmnerg_twitter10000_layered_action_verifier.yaml \
  --output outputs/fmnerg_roberta128_layered_action_verifier/train_dev_distribution_audit.json \
  --device cuda
```

当前非 OOF train 与 dev 的关键差异为：

| 统计量 | Train | Dev | Dev-Train |
| --- | ---: | ---: | ---: |
| formal prediction accuracy | 0.8788 | 0.7608 | -0.1180 |
| KEEP label ratio | 0.9209 | 0.8049 | -0.1159 |
| TO_NULL label ratio | 0.0448 | 0.0680 | +0.0232 |
| TO_VISIBLE label ratio | 0.0344 | 0.1271 | +0.0927 |
| Fine top1-top2 mean margin | 6.1752 | 5.0629 | -1.1123 |

这确认当前 Action Verifier 训练状态比部署 dev 明显更容易。正式 M3.6A-r2 必须进行
10-fold 整链 cross-fitting：每折依次训练 Stage1、构建 heldout R16/R36、交叉拟合
Coarse、Fine 和 Evidence Visibility，再生成 heldout formal KEEP 状态。仅替换 OOF
Stage1 而复用看过该 fold 标签的 Fine/Visibility 仍属于泄漏。SigLIP2 编码器可复用，
但特征必须按每折 span/region 索引重新对齐。新训练必须带
`--require-oof-train-cache` 运行分布审计并与正式 dev 比较后才允许启动。

M3.6A-r3 不再恢复共享三分类 CE。状态专属优先级固定为：先验证 NULL Release，
再验证 Region Switch；Null-Revert 只有在完整 OOF 下连续多 seed 非负才重新开放。三者
使用相对 KEEP 的二分类优势，且 false-null 损伤成本必须高于 missed-null。

### 14.2 M3.6A-r2 NULL Release 与整链 OOF 契约

r1 的 `+6` 全部来自 NULL 到真实区域，因此 r2 关闭 TO_NULL 和真实区域间切换，仅对
当前 formal 输出为 NULL 且最终记录解码实际保留的实体执行：

```text
KEEP / RELEASE_TO_VISIBLE -> Fine Top-4
```

实现使用独立 `NullReleaseVerifier`。KEEP logit 固定为相对基准 0，Release Head 学习
单一 advantage；末层零权重、`-4` bias 保证 epoch 0 全 KEEP。错误释放和漏释放分别
使用 `3:1` 权重，Layer 2 仍使用 Top-4 多正框 NLL。只有当前 NULL 的正式预测进入
policy scope，不能再用所有 Stage1 span 扩大表面样本量。

正式训练不再在线读取单个全局 train checkpoint，而是读取 10-fold 整链 OOF 冻结
特征缓存。每个 heldout 样本的 Stage1、Hierarchy、Coarse、Fine、Evidence 与
Reliability 输出必须来自未见过该样本标签的 fold 链。合并缓存执行以下硬检查：

1. fold id 必须完整为 `0..9`；
2. heldout ID 不得跨折重复，合计必须为 7000；
3. 每折 train ID 必须精确等于全量 ID 减去本折 heldout ID；
4. R16/R36 必须同时标记为同一 `oof_heldout` fold；
5. 全部配置、缓存和 checkpoint 必须匹配 fold proof 中的 SHA-256；
6. 启用 Reliability 时，缓存必须包含该 fold 的 Reliability 输出；
7. fold proof 必须绑定经过校验的 pipeline manifest；Stage1、Hierarchy、Coarse、Fine、
   Evidence、Reliability 六个监督阶段必须使用同一个 fold train ID 摘要并明确排除
   heldout；
8. OOF 运行期间不构建 Stage1 test Dataset，也不读取层次化 `test_cache`；
9. Fine Top-4 索引和有效位在物化时冻结，加载后禁止重新排序；
10. 缓存缺失或任一检查失败时，训练在加载 dev 和 GPU 模型前终止。

缓存只保存模型实际消费的冻结中间状态，表示张量使用 FP16，Fine/base/coarse 排序
logits 保持 FP32，并显式保存固定 Top-4，因此不会重复保存原图或 VinVL 2048 维原始
区域特征，也不会因加载后重新排序产生标签漂移。当前 r1 的
in-sample `+6` 只作为结构 go 信号，不能写成 r2 正式结果。

Fold 0 的正式入口为：

```bash
PYTHONPATH=. python scripts/run_null_release_full_chain_oof_fold.py \
  --fold-id 0 \
  --dry-run

nohup env PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u scripts/run_null_release_full_chain_oof_fold.py \
    --fold-id 0 \
  > null_release_oof_fold0.log 2>&1 &
```

该入口默认支持断点续跑，但会重新核验每个输入、输出、配置和 checkpoint 的哈希。
pipeline 封存后若任何监督阶段产物发生变化，必须重建该 fold，不能静默复用。

### 14.3 整链 OOF 折级封存与清理

十折采用流式生命周期：

```text
run fold
-> seal
-> validate
-> materialize heldout_features.pt
-> archive reports and hashes
-> delete rebuildable fold artifacts
-> reload heldout_features.pt
-> run next fold
```

清理入口为 `tools/archive_null_release_oof_fold.py`。它不属于实验源码树指纹，且默认
dry-run。执行前强制验证：pipeline 已封存、8 个阶段完整、未访问 test、fold proof
绑定未变化、proof 中 artifact SHA-256 均能对应现存文件、heldout ID 完整互斥、
固定 Top-4 合法，以及冻结 payload 只包含张量和基础值而不依赖外部路径。

```bash
PYTHONPATH=. python tools/archive_null_release_oof_fold.py \
  --fold-id 0 \
  --fold-work knowledge/null_release_oof/roberta128/fold0 \
  --output-work-root outputs/null_release_oof/roberta128/fold0

PYTHONPATH=. python tools/archive_null_release_oof_fold.py \
  --fold-id 0 \
  --fold-work knowledge/null_release_oof/roberta128/fold0 \
  --output-work-root outputs/null_release_oof/roberta128/fold0 \
  --execute
```

永久保留 heldout 冻结特征、proof、pipeline manifest、配置、日志、metrics 和
checkpoint 哈希清单；checkpoint 本体应先同步到服务器外部存储。删除范围被限制为
本折 `candidates/`、`siglip2/` 和对应 output root。清理前后状态、目录树摘要和
SHA-256 记录在 `fold_archive_manifest.json`；清理后再次从磁盘加载特征做 smoke。
默认每折保留上限为 500 MB，任何闸门失败都不会开始删除。

Stage1 若在完整写出 `train_summary.json`、best checkpoint 和 tokenizer 后，仅在
解释器退出阶段出现 `SIGSEGV: 11`，使用
`tools/recover_completed_oof_stage.py` 做受限恢复。工具默认 dry-run，并核验
checkpoint 可加载、summary 指标一致、训练完成标记、外层 SIGSEGV 记录、test-free
配置、fold ID 和当前源码树指纹；通过后显式 `--execute` 才会备份原 pipeline 并写入
恢复 receipt。它不接受普通异常、OOM、中途崩溃或缺少完成标记的产物。

Fold 2–9 的正式流式入口为：

```bash
nohup env \
  START_FOLD=2 \
  END_FOLD=9 \
  PREPARE_PREDECESSOR=1 \
  MIN_FREE_GB=5 \
  MIN_GPU_FREE_MB=10000 \
  POLL_SECONDS=300 \
  GPU_POLL_SECONDS=300 \
  ALLOW_HASH_ONLY_CHECKPOINT_RETENTION=1 \
  bash tools/run_null_release_oof_folds_streaming.sh \
  > null_release_oof_folds2_9_master.log 2>&1 &
```

入口会等待当前 Fold 1 物化完成并先封存清理，然后严格串行处理 Fold 2–9。每折只有在
sealed、八阶段完整、test-free、proof/hash、700 条 heldout、固定 Top-4、特征自包含
和清理后复载全部通过后，才进入下一折。`ALLOW_HASH_ONLY_CHECKPOINT_RETENTION=1`
明确授权删除 checkpoint 本体；服务器永久保留的是冻结特征、证明、配置、日志、指标
和 checkpoint 哈希。任意普通训练异常都会停止整条流水线，只有符合严格契约的
Stage1 完成后 `SIGSEGV: 11` 可自动恢复。Fold 9 后不会自动合并或训练 Release Head，
必须先完成人工十折覆盖与分布审计。

M3.6B 仅在多实体、同类 object、Top-1/2 冲突、高 context overlap 且 pair margin
较低时启用 Set Verifier。NULL 可无限复用，真实区域默认允许共享；只有不同 PER 等
明确竞争关系才施加软容量损失。普通记录必须绕过集合模块，以单独报告碰撞净修正、
非碰撞损伤和共享区域保留数。

M3.5A DINOv2 仍是冻结旁路诊断，只评估多人、同 object、PER、中等框和高 context
overlap 切片。通过条件为多人 AUROC 至少 `0.60`、同 object 至少 `0.58`、风险净
修正不低于 `+9`、NULL preservation 至少 `0.98`，并减少高重叠 DAMAGE。未通过则
停止增加通用视觉编码器；无论是否通过，都不能把 DINOv2 描述为实体身份识别器。

M3.6A/B 的 checkpoint、阈值和候选预算只能在 dev 上确定。必须报告 TO_NULL 与
TO_REAL 的 corrected/damaged/net、KEEP 正确保护率、Top-4 动作准确率、碰撞/非碰撞
切片和三元组净修正；未达到 KEEP 保护率 `0.97` 且 dev GMNER 稳定超过
`0.621316` 前不得读取 test。
