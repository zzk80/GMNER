# GMNER / FMNERG Core

本仓库只保留当前可复现主链及其必要诊断工具。云端自行提供图像、VinVL
`.npz`、Transformer 权重、候选缓存和 checkpoint；推理阶段不调用 LLM/MLLM，也不
要求部署 VinVL 检测器。

## 当前主链

```text
FMNERG text
  -> RoBERTa-base + dependency graph + CRF
  -> Stage1 span/type predictions

VinVL region features + boxes
  -> Stage1 grounding
  -> R16 formal + R36 expanded candidate caches
  -> hierarchical verifier
       -> span reject
       -> fixed Stage1 type
       -> base visible / NULL decision
  -> coarse recall selector
  -> Fine Adapter visible-region ranking
  -> Evidence Visibility bounded residual
       -> fixed 0.80 / 0.20 dual-threshold decode
  -> non-overlapping record decode
```

当前主 Stage1 已由 mBERT 切换为 `RoBERTa-base, max_length=128`。RoBERTa
Stage1 已完成一次性 test；层次化 verifier 也已使用 RoBERTa checkpoint 和独立
候选缓存完成重建，并完成最终 test。旧 mBERT verifier 只作为历史对照：

| Model | Status | MNER F1 | EEG F1 | GMNER F1 |
| --- | --- | ---: | ---: | ---: |
| mBERT Stage1 | historical test | 0.78593 | 0.62582 | 0.58154 |
| mBERT hierarchical verifier | historical test | 0.78951 | **0.63526** | 0.59034 |
| RoBERTa-128 Stage1 | current test | 0.81586 | 0.62683 | 0.59168 |
| RoBERTa-128 hierarchical verifier | test | 0.81843 | 0.64431 | 0.60784 |
| RoBERTa-128 Fine Adapter | fixed test | 0.81843 | 0.64980 | 0.61333 |
| RoBERTa-128 Evidence Visibility | **current best test** | **0.81843** | **0.65216** | **0.61529** |

RoBERTa 层次链路相对同一 Stage1 test bypass 的增量为：MNER `+0.00258`、
EEG `+0.01748`、GMNER `+0.01616`，净增加 35 个正确三元组。
当前完整链路相对同一 Stage1 test bypass 的 GMNER 增量为 `+0.02361`。

详细结构和复现实验见
[README_HIERARCHICAL_RECORD_VERIFIER.md](README_HIERARCHICAL_RECORD_VERIFIER.md)。
已完成并判定 no-go 的 dev-only M3.4A 见
[README_SIGLIP2_REGION_RELIABILITY.md](README_SIGLIP2_REGION_RELIABILITY.md)。
历史实验结论保留在 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)，原始研究
设想保留在 [idea.md](idea.md)。

## 有效配置

```text
configs/fmnerg_twitter10000_stage1.yaml
configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml
configs/fmnerg_twitter10000_coarse_selector.yaml
configs/fmnerg_twitter10000_fine_grounding_adapter.yaml
configs/fmnerg_twitter10000_evidence_visibility.yaml
configs/fmnerg_twitter10000_siglip2_reliability_vinvl.yaml
configs/fmnerg_twitter10000_siglip2_reliability_siglip2.yaml
configs/fmnerg_twitter10000_siglip2_reliability_fusion.yaml
configs/fmnerg_twitter10000_layered_action_verifier.yaml
configs/fmnerg_twitter10000_layered_action_to_real_only.yaml
configs/fmnerg_twitter10000_layered_action_to_null_only.yaml
configs/fmnerg_twitter10000_null_release_verifier.yaml
```

- `fmnerg_twitter10000_stage1.yaml`：当前 RoBERTa-128 Stage1。
- `fmnerg_twitter10000_hierarchical_record_verifier.yaml`：RoBERTa 主链的层次化结构。
- `fmnerg_twitter10000_coarse_selector.yaml`：M3.1 的 R36 召回保持型粗筛。
- `fmnerg_twitter10000_fine_grounding_adapter.yaml`：M3.2 的 visible-only 区域细排。
- `fmnerg_twitter10000_evidence_visibility.yaml`：M3.3A 的冻结区域证据 Visibility 残差。
- 三个 `siglip2_reliability_*` 配置：M3.4A 的 VinVL-only、SigLIP2-only 和融合
  旁路消融；只允许 train/dev，已归档为 no-go，不属于当前正式推理链。
- `fmnerg_twitter10000_layered_action_verifier.yaml`：M3.6A 的 dev-only 分层
  `KEEP/TO_NULL/TO_VISIBLE -> Fine Top-4` 动作验证；尚未进入正式 test 链。
- 两个 `layered_action_to_*_only` 配置：M3.6A-r1 的严格分支消融；TO_REAL-only
  工程通过，TO_NULL-only 失败，均未读取 test。
- `fmnerg_twitter10000_null_release_verifier.yaml`：M3.6A-r2 的正式 NULL Release
  配置；缺少 10-fold 整链 OOF 特征缓存时会拒绝训练。

已失败的 prototype、CHEG、external-knowledge、旧 token-region multiscale、flat verifier、
utility controller 和 listwise action-policy 配置不再作为运行入口保留。

## 主要入口

```text
scripts/train.py
scripts/evaluate.py
scripts/build_record_candidate_cache.py
scripts/train_hierarchical_record_verifier.py
scripts/evaluate_hierarchical_record_verifier.py
scripts/audit_layered_action_distribution.py
scripts/analyze_visible_region_oracle.py
scripts/train_coarse_region_selector.py
scripts/evaluate_coarse_region_selector.py
scripts/train_fine_grounding_adapter.py
scripts/evaluate_fine_grounding_adapter.py
scripts/train_evidence_visibility.py
scripts/evaluate_evidence_visibility.py
scripts/analyze_evidence_visibility_release.py
scripts/analyze_candidate_oracle.py
scripts/analyze_hierarchical_action_oracle.py
scripts/build_evidence_folds.py
scripts/build_oof_hierarchical_candidates.py
scripts/build_siglip2_region_cache.py
scripts/train_siglip2_region_reliability.py
scripts/evaluate_siglip2_region_reliability.py
scripts/train_layered_action_verifier.py
scripts/evaluate_layered_action_verifier.py
```

## 运行

Stage1：

```bash
cd ~/gmner
PYTHONPATH=. python scripts/train.py \
  --config configs/fmnerg_twitter10000_stage1.yaml
```

当前已训练 checkpoint：

```text
outputs/fmnerg_stage1_roberta128/best_model.pt
```

### 文本骨干受控对照

骨干对照只修改 text model、统一 `max_length=128`，并写入独立输出目录。
第一阶段跳过 test，只依据 dev 选择候选骨干：

```bash
PYTHONPATH=. python scripts/train.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --text-model-name /home/zzk/gmner/bert-base-multilingual-cased \
  --max-length 128 \
  --output-dir outputs/fmnerg_stage1_mbert128 \
  --skip-test-evaluation

PYTHONPATH=. python scripts/train.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --text-model-name /home/zzk/gmner/roberta-base \
  --max-length 128 \
  --output-dir outputs/fmnerg_stage1_roberta128 \
  --skip-test-evaluation

PYTHONPATH=. python scripts/train.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --text-model-name /home/zzk/gmner/bertweet-base \
  --max-length 128 \
  --output-dir outputs/fmnerg_stage1_bertweet128 \
  --skip-test-evaluation
```

BERTweet 当前由慢速 tokenizer 加载，数据层会显式构造 word-to-subword
alignment；其可用输入上限为 128，配置超过该值会在训练前直接报错。
RoBERTa fast tokenizer 会自动以 `add_prefix_space=True` 加载，以支持预分词
输入。CRF 标签、实体 mask 和依存图偏移继续使用原始词编号。每次训练会在
输出目录保存包含所有 CLI 覆盖项的 `resolved_config.yaml`。

文本骨干 dev 对照：

| Backbone | MNER F1 | EEG F1 | GMNER F1 |
| --- | ---: | ---: | ---: |
| mBERT-128 | 0.7830 | 0.6200 | 0.5715 |
| BERTweet-128 | **0.8180** | 0.6391 | 0.6061 |
| RoBERTa-128 | 0.8147 | **0.6460** | **0.6073** |

RoBERTa 层次化 verifier 已在同一 dev 集上完成验证：

| Model | MNER F1 | EEG F1 | GMNER F1 |
| --- | ---: | ---: | ---: |
| RoBERTa Stage1 bypass | 0.81474 | 0.64599 | 0.60733 |
| RoBERTa hierarchical verifier | **0.81671** | **0.65442** | **0.61526** |
| Delta | +0.00197 | +0.00843 | +0.00793 |

RoBERTa 层次化候选缓存：

```bash
mkdir -p knowledge/record_candidates/roberta128

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

层次化 verifier（训练后先验收 dev，不自动运行 test）：

```bash
PYTHONPATH=. python scripts/train_hierarchical_record_verifier.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml

PYTHONPATH=. python scripts/evaluate_hierarchical_record_verifier.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml \
  --checkpoint outputs/fmnerg_roberta128_hierarchical_record_verifier/best_model.pt \
  --split dev \
  --output outputs/fmnerg_roberta128_hierarchical_record_verifier/dev_metrics.json
```

Visible-region R16/R36 诊断只在 dev 上运行，不修改正式配置：

```bash
PYTHONPATH=. python scripts/build_record_candidate_cache.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
  --split dev \
  --output knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical_r36.pt \
  --k-best 6 \
  --max-span-candidates 12 \
  --top-m-types 3 \
  --boundary-shift 0 \
  --boundary-penalty 0.25 \
  --max-regions 36 \
  --device cuda

PYTHONPATH=. python scripts/analyze_visible_region_oracle.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml \
  --checkpoint outputs/fmnerg_roberta128_hierarchical_record_verifier/best_model.pt \
  --expanded-cache knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical_r36.pt \
  --split dev \
  --proposal-budgets 16,36 \
  --output outputs/fmnerg_roberta128_hierarchical_record_verifier/dev_visible_region_oracle.json
```

该报告按统一口径区分候选缺失、Stage1 排错、Verifier real-region 损伤和
false-NULL。若 `R36-R16 >= 0.03`，再考虑 coarse-to-fine 扩框；否则优先训练
Grounding Adapter。

当前 RoBERTa dev 诊断结果：

```text
visible gold                    991
R_region@16                     0.83451 (827/991)
R_region@36                     0.89909 (891/991)
new coverage                    +64 / +0.06458
base wrong + gold in R16        219
base wrong + gold only in R36    57
```

直接让 Stage1 在 36 框上分类会将 dev GMNER 从 `0.60733` 降至 `0.60004`。
因此下一结构固定为 `top36 coarse proposal -> top12/16 fine Grounding Adapter`，
不能把正式分支简单替换成 R36 softmax。

### M3.1：召回保持型粗筛

粗筛只读取冻结 Stage1 生成的 R36 缓存，不更新 RoBERTa、CRF、Visibility 或
层次化 verifier。训练分组按候选覆盖定义：原始 detector Top-16 未覆盖但 R36
覆盖的样本用于 promotion，Top-16 已覆盖的样本用于 preservation。推理同时比较
detector Top-16、Stage1 base-score Top-16、learned Top-16、Stage1 Top-8 +
learned Top-8、Stage1 Top-10 + learned Top-6。`base Top-16` 用于确认增益是否
真的来自 learned selector。

先构建 train R36 缓存：

```bash
PYTHONPATH=. python scripts/build_record_candidate_cache.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
  --split train \
  --output knowledge/record_candidates/roberta128/fmnerg_train_hierarchical_r36.pt \
  --k-best 6 \
  --max-span-candidates 12 \
  --top-m-types 3 \
  --boundary-shift 0 \
  --boundary-penalty 0.25 \
  --max-regions 36 \
  --batch-size 8 \
  --device cuda
```

训练和 dev 验收：

```bash
PYTHONPATH=. python scripts/train_coarse_region_selector.py \
  --config configs/fmnerg_twitter10000_coarse_selector.yaml

PYTHONPATH=. python scripts/evaluate_coarse_region_selector.py \
  --config configs/fmnerg_twitter10000_coarse_selector.yaml \
  --checkpoint outputs/fmnerg_roberta128_coarse_selector/best_model.pt \
  --output outputs/fmnerg_roberta128_coarse_selector/dev_metrics.json
```

M3.1 不读取 test。进入 M3.2 的最低条件是 union Recall@16 明显高于
`0.83451`、目标约 `0.87`，同时原始 Top-16 正确框保留率不低于 `0.98`，平均
候选数不超过 16。

当前 dev 最优结果（epoch 1）：

| Policy | Recall@16（eligible） | 原 Top-16 保留率 | 新找回 | 丢失 |
| --- | ---: | ---: | ---: | ---: |
| detector Top-16 | 0.84505 | 1.00000 | 0 | 0 |
| Stage1 base-score Top-16 | 0.88791 | 1.00000 | 39 | 0 |
| learned Top-16 | 0.90769 | 0.99870 | 58 | 1 |
| base Top-8 + learned Top-8 | **0.90769** | **1.00000** | 57 | 0 |
| base Top-10 + learned Top-6 | 0.90549 | 0.99870 | 56 | 1 |

这里 `eligible=910` 表示 Stage1 命中的可见 gold span，`recoverable=827` 表示
其中 R36 至少存在一个 IoU 正例；因此当前 Top8+8 已覆盖 826/827 个可恢复样本，
平均候选数为 15.68。M3.1 已通过，M3.2 使用 Top8+8 作为默认候选策略。

### M3.2：Correction-Preservation Grounding Adapter

M3.2 保留正式 R16 层次模型作为完整决策基线，只让 R36 分支处理已经被该
基线判为 visible 的实体。RoBERTa、CRF、type、Reject、Visibility、VinVL 和
M3.1 coarse selector 全部冻结；训练参数仅属于 Text Grounding Adapter、region
projection 和 entity-region fine scorer。

```text
formal R16 hierarchy -> span / type / reject / visibility
expanded R36 cache   -> Base Top-8 + Learned Top-8
                     -> calibrated base/coarse prior
                     -> bounded fine interaction residual
baseline visible     -> replace real-region index only
baseline NULL        -> keep NULL
```

细排特征包含实体与区域向量、乘积和绝对差、固定预测 type、候选来源
`base-only / learned-only / both`、base/coarse/detector rank、detector score、
type-object compatibility、bbox 和 promoted 标记。当前缓存未持久化原始 object /
attribute 字符串，因此第一版使用已有的 type-object compatibility，不为此重建缓存。

训练损失采用多正例 NLL、IoU soft target、定向 correction margin、preservation
margin 和仅作用于保护组的 residual L2。Correction / Preservation / Other 按
`0.4 / 0.4 / 0.2` 分组求均值，promoted correction 再单独平衡。候选外正例不
参加 fine loss。

工程验证可直接使用当前 train cache；论文正式结果必须改用对齐的 RoBERTa OOF
R16/R36 train cache，并把 `require_oof_train_cache` 设为 `true`。本阶段没有 test
入口，checkpoint 按 dev `gmner_score` 保存。

```bash
PYTHONPATH=. python scripts/train_fine_grounding_adapter.py \
  --config configs/fmnerg_twitter10000_fine_grounding_adapter.yaml

PYTHONPATH=. python scripts/evaluate_fine_grounding_adapter.py \
  --config configs/fmnerg_twitter10000_fine_grounding_adapter.yaml \
  --checkpoint outputs/fmnerg_roberta128_fine_grounding_adapter/best_model.pt \
  --output outputs/fmnerg_roberta128_fine_grounding_adapter/dev_metrics.json
```

最低验收条件：`visible_corrected > visible_damaged`、visible 净修正至少 `+5`、
`base_correct_preservation_rate >= 0.97`、promoted 57 个候选至少恢复 8 个、
`span_f1_delta = entity_f1_delta = 0`，且 dev GMNER 高于对应层次 baseline。

当前非 OOF 工程验证在 epoch 1 达到：

| Metric | R16 baseline | Epoch-0 prior | M3.2 fine |
| --- | ---: | ---: | ---: |
| Span F1 | 0.87283 | 0.87283 | 0.87283 |
| MNER F1 | 0.81671 | 0.81671 | 0.81671 |
| EEG F1 | 0.65442 | 0.65725 | **0.65805** |
| GMNER F1 | 0.61526 | 0.61809 | **0.61889** |

Fine 结果为 `22 corrected - 13 damaged = +9`，Stage1 正确区域保护率
`0.97466`；57 个 promoted gold 中，fine raw top-1 选对 32 个，其中受冻结
Visibility 允许部署的 27 个样本选对 14 个。相对双先验的 epoch 0，训练后的
adapter 继续增加 2 个净正确三元组。该结果通过 M3.2 最低
门槛，但仍属于同分布 train-cache 工程证据，不能替代 OOF、多随机种子和正式
test 报告。

### M3.3A：Region-Evidence Visibility Head

M3.2 诊断发现，Fine top-1 已选对但最终仍为 NULL 的样本有 113 个，其中 106 个
固定 type 正确；promoted 候选中对应有 18 个。与此同时，Fine top-1 错误且当前
保持 NULL 的样本有 74 个，因此不能通过整体降低 visible 阈值解决。

M3.3A 冻结 RoBERTa、CRF、type、Reject、层次模型、coarse selector 和完整 Fine
Adapter，只训练一个区域证据 Visibility Head：

```text
frozen hierarchy visibility logit -----------------------+
                                                         |
frozen Fine top-1 state + ranking distribution           |
  + top1/top2 margin + normalized entropy                |
  + base/coarse/fine agreement + candidate source        |
  + detector confidence + type-object compatibility      |
  -> Evidence Visibility Head -> bounded residual -------+
  -> original 0.80 / 0.20 dual-threshold decode
  -> visible uses frozen Fine top-1; NULL uses expanded NULL
```

最后一层采用零初始化，因此 epoch 0 必须逐项复现 M3.2 的 dev 输出。区域证据在
进入 Visibility Head 前显式 detach，M3.3A 不更新 Fine Adapter。训练按组平衡：

```text
visible correction: gold visible + type correct + Fine top-1 correct + current NULL
NULL correction:    gold NULL + type correct + current visible
visible preserve:   gold visible + Fine top-1 correct + current visible
NULL preserve:      gold NULL + current NULL
uncertain keep:     高熵、低 margin、排序器不一致或不可操作样本保持原概率
```

损失由分组 BCE、visible correction、NULL preserve、Bernoulli-KL keep 和保护组
residual L2 组成。checkpoint 仍以 dev GMNER 为主，NULL 正确保护率与 Visibility
净修正只作平局项；配置和脚本不提供 test cache/自动 test 入口。

```bash
PYTHONPATH=. python scripts/train_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml

PYTHONPATH=. python scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --output outputs/fmnerg_roberta128_evidence_visibility/dev_metrics.json
```

最低通过条件：GMNER 高于 `0.61889`、`fine_top1_correct_final_null_type_correct`
明显下降、visible 净修正为正、NULL 净修正非负、NULL 正确保护率不低于 0.97，
且 Span/MNER delta 均为 0。达到这些条件后才进入 M3.3B 的低权重联合适配。

首轮非 OOF 工程验证在 epoch 1 取得：

| Metric | M3.2 | M3.3A |
| --- | ---: | ---: |
| Span F1 / MNER F1 delta | - | 0 / 0 |
| EEG F1 | 0.65805 | **0.66088** |
| GMNER F1 | 0.61889 | **0.62132** |
| GMNER corrected / damaged | - | 14 / 8 |
| visible corrected / damaged | - | 4 / 3 |
| NULL corrected / damaged | - | 12 / 6 |
| NULL 正确保护率 | - | 0.99458 |
| type-correct Fine-correct-but-NULL | 106 | 105 |
| promoted final triple | 14 | 13 |

该结果净增加 6 个正确三元组，但主要来自 visible→NULL 的校正；预期的“Fine 已
找对区域却被 Visibility 阻止”只释放 1 个，而且 promoted triple 损失 1 个。因此
当前只能判定为**指标增益成立、目标机制尚未通过**，暂不进入 M3.3B。固定
checkpoint 与双阈值后已按用户要求完成一次性 test；不再使用 test 调参。

一次性 test 结果：M3.2 frozen baseline 的 MNER/EEG/GMNER 为
`0.81843/0.64980/0.61333`，M3.3A 为 `0.81843/0.65216/0.61529`。Visibility 对
GMNER 修正 15、损伤 10，净增加 5 个三元组；NULL 净修正 `+12`，真实可见区域
净修正 `-6`，NULL 正确保持率 `0.99910`。结果证明小幅总收益可以泛化，但区域
证据释放问题仍未解决。下一步应提升 null→visible 可分性和真实框可靠性，而不是
继续扫描统一阈值。

固定 test 复现命令如下；test cache 必须显式传入，训练配置不会自动读取 test：

```bash
PYTHONPATH=. python scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --split test \
  --formal-cache knowledge/record_candidates/roberta128/fmnerg_test_hierarchical.pt \
  --expanded-cache knowledge/record_candidates/roberta128/fmnerg_test_hierarchical_r36.pt \
  --output outputs/fmnerg_roberta128_evidence_visibility/test_metrics.json
```

### M3.3A.1：Evidence Release Diagnosis

诊断入口固定 M3.3A checkpoint，只读取 dev，不训练或修改主模型：

```bash
PYTHONPATH=. python scripts/analyze_evidence_visibility_release.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --output outputs/fmnerg_roberta128_evidence_visibility/dev_release_diagnostic.json
```

脚本输出 A/B/C/D 与 NULL 分组、逐样本 A 组 JSONL、双阈值距离、残差结构
Oracle、部署动作 Oracle，以及不依赖 sklearn 的五折线性可分性探针。当前结果：

| 诊断项 | 结果 |
| --- | ---: |
| A：base NULL + Fine correct | 106 |
| B：base NULL + Fine wrong | 122 |
| B 中 gold region 仍在候选集 | 64 |
| C：base visible + Fine correct | 516 |
| D：base visible + Fine wrong | 127 |
| gold NULL + base NULL / visible | 1017 / 135 |
| A 实际释放 | 4 / 106 |
| promoted A 实际释放 | 0 / 18 |
| 当前 residual bound 可达 A | 95 / 106 |
| 当前 residual bound 可达 promoted A | 17 / 18 |
| A/B 全集五折 AUROC / balanced accuracy | 0.6856 / 0.6328 |
| A/候选覆盖 B 五折 AUROC / balanced accuracy | **0.6167 / 0.5503** |
| A vs B+gold-NULL 五折 AUROC | 0.7965 |
| NULL→visible 完美动作 Oracle | +106 triples |

A 组所需正残差中位数为 `1.78`，当前上限为 `4.0`；约 89.6% 的 A 样本结构上
可达。因此不应优先放大 residual scale。A 与包含 region-missing 的全部 B 尚有中等
可分性，但去除这类容易负例后，candidate-covered A/B 接近不可分；当前 head 给该
困难切片的 residual AUROC 也只有 `0.5611`。

结论对应诊断中的 Case C：动作空间上限很高、残差参数化不是主瓶颈，缺失的是
“候选区域绝对有效性”。下一步应先训练独立 Region Reliability Head，以 IoU 正例、
Stage1/Fine 高分错误框、gold NULL 全部真实框和同图其他实体框构造监督；其输出先
只做诊断，不直接控制 Visibility。Reliability 在困难 A/B 切片显著可分后，才允许
接入 Visibility；M3.3B 继续暂停。

### 已归档：Absolute Region Reliability

该隔离实验的 hard A/B AUROC 最高约为 `0.64`，未达到 `0.70` 的接入门槛；风险
checkpoint 在 dev 上只有小规模高精度尾部，且未进入当前最优链路。因此对应配置、
训练脚本、模型、损失和测试已从工作区删除。历史数值保留在
[EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)，后续若重启该方向，应直接采用
实体 crop、局部 crop 与全图的多尺度视觉证据，而不是恢复旧 MLP。

### 已归档：M3.4A SigLIP 2 Region Reliability

冻结 SigLIP 2 的 mention/context/type 与 local/context/global 三尺度诊断已经完成。
同口径 dev hard A/B AUROC 为 VinVL `0.5773`、SigLIP2 `0.5759`、Fusion `0.6003`；
Fusion balanced accuracy 为 `0.6241`，风险净纠错为 `+9`。该分支未达到
`AUROC >= 0.70` 和 `risk net >= +15` 门槛，因此不进入 M3.4B、不接入 Visibility、
不读取 test。正式结果继续保持 GMNER F1 `0.61529`。旧 VinVL-only `0.6393` 使用
不同的 pre-Visibility A/B 定义，不能与本轮作直接升降比较。详见
[README_SIGLIP2_REGION_RELIABILITY.md](README_SIGLIP2_REGION_RELIABILITY.md)。

### 开发中：M3.6A/r1 分层 Top-4 动作验证

旧 Action Controller 已包含固定 `KEEP=0`；其失败应归因于状态相关 KEEP 不可学习、
NULL 与真实区域平坦竞争，以及 fused/residual/base 候选并集带来的冗余。M3.6A 在
当前正式 Evidence Visibility 输出之后增加独立残差策略：

```text
Layer 1: KEEP / TO_NULL / TO_VISIBLE
Layer 2: Fine Top-4 real region, only under TO_VISIBLE
```

当前为 NULL 时屏蔽 `TO_NULL`；当前为 visible 时从 Layer 2 删除当前区域，保持当前
区域只能由 `KEEP` 表达。只有 span/type 正确且 gold 动作在 Top-4 内的样本参与动作
监督；其余不可操作样本只进入 preservation loss。RoBERTa Stage1、Hierarchy、
Coarse/Fine、Evidence Visibility 和 Fusion Reliability 均冻结。

```bash
PYTHONPATH=. python scripts/train_layered_action_verifier.py \
  --config configs/fmnerg_twitter10000_layered_action_verifier.yaml

PYTHONPATH=. python scripts/evaluate_layered_action_verifier.py \
  --config configs/fmnerg_twitter10000_layered_action_verifier.yaml \
  --checkpoint outputs/fmnerg_roberta128_layered_action_verifier/best_model.pt \
  --split dev \
  --output outputs/fmnerg_roberta128_layered_action_verifier/dev_metrics.json
```

训练前会强制检查 epoch 0：执行动作数、区域变化数和预测数量变化必须为 0，所有记录
预测及 MNER/EEG/GMNER 必须与冻结链完全一致，完整 dev 基线还必须匹配
`GMNER=0.621316`。评估入口只接受 `--split dev`，当前正式 test `0.61529` 继续冻结。

首次非 OOF 工程运行在 epoch 3 早停，best 保持 epoch 0。epoch 1/2/3 的 dev
GMNER 分别为 `0.61849/0.59386/0.61163`，净纠错为 `-7/-68/-24`；损伤主要来自
错误 TO_NULL，TO_REAL 仅在单独风险排序中出现最多 `+8` 的窄前缀信号，未转化为
最终正收益。因此 M3.6A 共享三分类首轮判定 no-go，不读取 test。

M3.6A-r1 已增加模型级 `action_mode`，在完全相同的冻结链和训练预算下隔离分支：

| Dev-only 分支 | Best epoch | GMNER | 执行 | 修正/损坏/中性 | 净修正 | KEEP 保护率 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TO_REAL-only | 3 | 0.623738 | 18 | 8/2/8 | +6 | 0.9987 |
| TO_NULL-only | 0 | 0.621316 | 0 | 0/0/0 | 0 | 1.0000 |

TO_REAL-only 达到工程门槛，但全部 `+6` 来自 `NULL -> real`：该迁移 18 次中修正 8、
损坏 2；`real -> other real` 没有执行，独立风险上限仅 `+3`。TO_NULL-only 学习后
出现 `-25/-14`，best 仍为 epoch 0 no-op，因此新 Null-Revert 分支暂停，继续沿用
Evidence Visibility 的现有 NULL 决策。

`audit_layered_action_distribution.py` 进一步确认当前 train cache 是 in-sample 且
明显更容易：formal 正确率 `0.8788 -> 0.7608`（train -> dev），KEEP 标签比例
`0.9209 -> 0.8049`，TO_VISIBLE 比例 `0.0344 -> 0.1271`。下一次训练前必须对
RoBERTa Stage1、R16/R36、Coarse、Fine 和 Evidence Visibility 整链执行 10-fold
cross-fitting；只替换 OOF Stage1 不满足要求。SigLIP2 权重可复用，但候选索引必须
逐 fold 对齐。当前正式 test `0.61529` 继续冻结。

### 开发中：M3.6A-r2 NULL Release Verifier

r2 已将有效动作进一步收缩为独立策略，而不是继续训练完整三分类控制器：

```text
Evidence Visibility formal state
  -> only currently NULL, formally decoded entities
  -> KEEP / RELEASE_TO_VISIBLE
  -> Fine Top-4 real-region choice after RELEASE
```

`visible -> NULL` 继续由已有 Evidence Visibility 负责，`visible -> other real` 暂停。
Release 使用相对 KEEP 的单一 advantage，错误释放权重为漏释放的 3 倍；训练监督只
覆盖正式解码实体。当前为 NULL 但 span/type 错误、gold 为 NULL、Top-4 不含合格框或
释放后不能修正完整三元组的样本均作为高代价负例。

正式配置默认要求 10-fold 整链 OOF，不能回退到当前 in-sample train cache：

```text
configs/fmnerg_twitter10000_null_release_verifier.yaml
scripts/create_null_release_fold_proof.py
scripts/build_null_release_oof_features.py
scripts/merge_null_release_oof_features.py
scripts/train_null_release_verifier.py
scripts/evaluate_null_release_verifier.py
```

整链包含 Stage1、R16/R36、Hierarchy、Coarse、Fine、Evidence Visibility，以及启用
时的 Fusion Reliability。冻结 SigLIP2 编码特征可以复用，但每折的 span/region 索引
和 manifest 必须重新对齐。每折 proof 记录 90% train ID、10% heldout ID 和全部配置、
缓存、checkpoint 哈希；合并器强制验证十折互斥、7000 条完整覆盖和每折训练集合恰好
等于其余九折。

先只运行 Fold 0 的整链闭环。入口会自动生成或验证统一 10-fold manifest，按顺序训练
该折的 Stage1、Hierarchy、Coarse、Fine、Evidence、Fusion Reliability，构建对齐的
R16/R36 与 SigLIP2 缓存，最后生成 fold proof 和 heldout 冻结特征：

```bash
PYTHONPATH=. python scripts/run_null_release_full_chain_oof_fold.py \
  --fold-id 0 \
  --dry-run

nohup env PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -u scripts/run_null_release_full_chain_oof_fold.py \
    --fold-id 0 \
  > null_release_oof_fold0.log 2>&1 &
```

Fold 0 未完成人工验收前，入口会拒绝执行其余折。确认其 pipeline manifest、fold proof、
heldout feature cache 和分布审计均正确后，才可对折 1–9 显式加入
`--allow-nonzero-fold`。

每折物化完成后使用折级归档工具释放可重建中间产物。该工具位于 `tools/`，不会进入
实验源码树指纹；默认只做只读 dry-run：

```bash
PYTHONPATH=. python tools/archive_null_release_oof_fold.py \
  --fold-id 0 \
  --fold-work knowledge/null_release_oof/roberta128/fold0 \
  --output-work-root outputs/null_release_oof/roberta128/fold0
```

dry-run 必须先通过 sealed pipeline、全部阶段、test-free、fold proof、全部 artifact
SHA-256、heldout 数量、固定 Top-4 和特征自包含检查。确认 checkpoint 已外部备份或接受
仅保留哈希清单后，显式执行：

```bash
PYTHONPATH=. python tools/archive_null_release_oof_fold.py \
  --fold-id 0 \
  --fold-work knowledge/null_release_oof/roberta128/fold0 \
  --output-work-root outputs/null_release_oof/roberta128/fold0 \
  --checkpoint-backup-note "copied outside the server before cleanup" \
  --execute
```

永久保留 `heldout_features.pt`、SHA-256 文件、fold proof、pipeline manifest、fold
configs、训练日志和最终 metrics。只删除本折 `candidates/`、`siglip2/` 和对应
`outputs/.../foldN/`。清理后工具会从磁盘重新加载冻结特征并重复 Top-4/record
校验；完整清理账本写入 `fold_archive_manifest.json`。默认要求每折最终保留空间低于
500 MB，重复执行是幂等的。

若 Stage1 已写出 summary、完整 checkpoint 和 tokenizer，并在记录
`Skipping final test evaluation by request.` 后仅于 Python/CUDA 退出阶段发生
`SIGSEGV: 11`，不得直接重跑 20 epochs，也不得手工编辑 pipeline。先执行严格只读
恢复审计：

```bash
PYTHONPATH=. python tools/recover_completed_oof_stage.py \
  --fold-id 1 \
  --fold-work knowledge/null_release_oof/roberta128/fold1 \
  --output-work-root outputs/null_release_oof/roberta128/fold1 \
  --failure-log null_release_oof_fold1.log
```

只有 summary/checkpoint 指标一致、checkpoint 可在 CPU 加载、tokenizer 已保存、
完成日志和 SIGSEGV 外层日志同时存在、test 被禁用且源码树未变化时，才可增加
`--execute` 原子恢复 Stage1 provenance。随后用原整链命令 `--resume`，Stage1 将按
哈希跳过。其他退出错误禁止使用该恢复工具。

Fold 2–9 可使用流式控制脚本自动执行“等待 Fold 1、运行、验收、归档、清理、下一折”：

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

`ALLOW_HASH_ONLY_CHECKPOINT_RETENTION=1` 是不可省略的删除授权：每折通过自包含特征
验证后，checkpoint 本体与可重建缓存会被删除，只永久保留
`heldout_features.pt`、proof、manifest、配置、日志、metrics 和 checkpoint
SHA-256 清单。需要永久保留 checkpoint 本体时，不得使用该模式，应先将每折
checkpoint 同步到服务器外存。脚本只对已经完整训练并在退出阶段触发
`SIGSEGV: 11` 的 Stage1 执行严格恢复；其他失败立即停止，不会进入下一折。

Fold 1–9 必须逐折执行“运行、验收、归档、清理”，不能先累计十份中间产物。十折全部
保留的 `heldout_features.pt` 完成后执行：

```bash
inputs=()
for fold in {0..9}; do
  inputs+=("knowledge/null_release_oof/roberta128/fold${fold}/heldout_features.pt")
done

PYTHONPATH=. python scripts/merge_null_release_oof_features.py \
  --inputs "${inputs[@]}" \
  --output knowledge/null_release_oof/roberta128/full_chain_train_oof.pt \
  --expected-records 7000

PYTHONPATH=. python scripts/aggregate_m33a_oof_metrics.py \
  --cache knowledge/null_release_oof/roberta128/full_chain_train_oof.pt \
  --source-file GMNER-main/Twitter10000_v2.0/txt_fine/train.txt \
  --output outputs/fmnerg_roberta128_m33a_oof_train/metrics.json

PYTHONPATH=. python scripts/train_null_release_verifier.py \
  --config configs/fmnerg_twitter10000_null_release_verifier.yaml
```

`aggregate_m33a_oof_metrics.py` 不加载 checkpoint、不训练模型，也不重新推理。它直接
聚合十个 heldout cache 中冻结的 M3.3A 正式预测：`deployment_span_mask` 确定实体，
`fixed_type_ids` 确定类型，`current_visible + Fine top-1/NULL` 确定区域。原始
`train.txt` 只用于提供完整 gold 分母并验证 7000 个 record ID 精确覆盖；span、type
和 region 是否正确均来自 heldout 时已冻结的匹配掩码。

当前十折 OOF Train 微平均结果为：

| Split | Records | Pred / Gold | Span F1 | MNER F1 | EEG F1 | GMNER F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M3.3A OOF Train | 7000 | 12001 / 11779 | 0.870900 | 0.811690 | 0.651135 | 0.610849 |

十折 GMNER F1 的 fold 均值为 `0.610869`、总体标准差为 `0.010907`；论文主表应使用
上述按 7000 条样本一次性计数得到的微平均 `0.610849`。该结果是严格的 OOF Train
诊断，不是 train in-sample 指标，也不是 fold ensemble。它不会改变冻结的正式
Dev/Test 结果；若要用十个 fold 模型改善 Dev/Test，必须另行定义 ensemble 实验。
聚合报告明确记录 `model_training=false`、`model_inference=false`、
`fold_ensemble=false` 和 `test_accessed=false`。

`--skip-test-evaluation` 现在连 test Dataset/DataLoader 都不会构建；层次化训练在关闭
自动 test 时也不再读取 `test_cache`。每折 pipeline manifest 固定 train/heldout ID、
源码树、配置和六个监督 checkpoint 的 SHA-256，所有阶段都必须声明
`test_accessed=false`。Fine Top-4 的索引和有效位在特征物化时显式保存，加载后不重新
排序。

训练仍只在 dev 选择 checkpoint；评估入口仅接受 `--split dev`。正式门槛为 NULL
Release 净修正至少 `+5`、正确 KEEP 保护率至少 `0.99`，并要求多随机种子平均为正。
在满足这些条件前不读取 test。

## 指标

```text
MNER  = entity_f1 = Entity + Type
EEG   = eeg_f1    = Entity + Region
GMNER = triple_f1 = Entity + Type + Region
```

`grounding_accuracy` 仅是条件诊断指标，不能替代 EEG 或 GMNER。正式模型按
`gmner_score` 选择 checkpoint。

## 测试

```bash
PYTHONPATH=. python -m pytest -q
```
