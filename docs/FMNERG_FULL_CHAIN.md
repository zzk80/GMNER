# M3.3F Independent FMNERG Chain

本文档定义独立 FMNERG 模型 `Model-F` 的正式开发协议。现有
M3.3A `Model-G` 不修改，正式 GMNER Dev/Test 结果不变。

## 1. 当前范围

本分支第一阶段严格停在 F1：

```text
F0
  fixed taxonomy + SHA
  Fine MNER / FMNERG metrics
  fine_hierarchical config and data contract
  subtype-aware R16 cache contract

F1
  matched B0 on Dev
  Stage1-F seed 42
  Stage1-F Dev evaluation
  Stage1-F Dev R16 cache
  Dev R16 visible-region proposal Oracle
```

当前不执行：

```text
R36
Hierarchical Verifier
Coarse Selector
Fine Adapter
Evidence Visibility
OOF
Test
```

只有 F2 三 seed Gate 通过后，才允许修改或训练后半链。

## 2. Stage1-F

```text
RoBERTa text states
├── [first; last; mean] span pooling
│   └── 51-class subtype head
│       └── predicted parent hard mask
│
└── existing text graph + cross-modal aligner
    ├── 9-label parent typed-BIO CRF
    └── R16 region / NULL grounding
```

Subtype 分支第一版是 text-only。它与 NER/grounding 共享 RoBERTa，
但不直接读取跨模态 fused tokens。

损失：

```text
L = L_NER
  + lambda_f * L_subtype
  + L_grounding
  + 0.1 * L_alignment
```

F1 只预注册 `lambda_f = 1.0` 和 `0.5`。主配置
[`configs/fmnerg_fine/stage1.yaml`](../configs/fmnerg_fine/stage1.yaml)
使用 `1.0`。

训练日志必须包含：

```text
raw/weighted NER loss
raw/weighted subtype loss
raw/weighted grounding loss
raw/weighted alignment loss
shared RoBERTa total gradient norm
subtype-vs-NER gradient cosine
subtype-vs-grounding gradient cosine
```

梯度余弦每 400 个 forward step 在训练开始时固定的小批次上采样一次，只作
诊断，不使用 PCGrad。

## 3. 不变量

Parent 仍固定为四类：

```text
LOC=0, PER=1, ORG=2, OTHER=3
```

Fine cache 在原字段旁新增：

```text
fixed_parent_ids
subtype_raw_logits       [num_spans, 51]
fixed_subtype_ids        [num_spans]
subtype_confidence       [num_spans]
subtype_margin           [num_spans]
subtype_entropy          [num_spans]
gold_subtype_ids         [num_spans]
```

每个 Stage1/Viterbi/k-best/perturbation span candidate 都必须有 raw
51 类 logits。正式 subtype 由 fixed parent mask 后的 argmax 得到。

Gold subtype 只用于 CE、指标和错误分析。任何后续模块的输入都只能使用
Stage1-F predicted subtype。`positive_fine_triple_mask` 不物化。

Taxonomy 的唯一源文件仍为：

```text
sidecars/fmnerg_subtype/taxonomy_twitter10000.json
```

checkpoint、cache 和运行时 taxonomy SHA-256 必须完全一致。

## 4. F1 运行顺序

以下命令均在云端仓库根目录运行。

### 4.1 预检

```bash
cd ~/gmner

PYTHONPATH=. python scripts/preflight_fmnerg_stage1.py \
  --config configs/fmnerg_fine/stage1.yaml \
  --output outputs/fmnerg_fine/f1_preflight.json
```

预检只读取 Train/Dev、taxonomy、模型路径和旧 Stage1 checkpoint，不解析
或读取 Test。

### 4.2 构建同口径 B0

```bash
PYTHONPATH=. python scripts/evaluate_fmnerg_stage1_b0.py \
  --subtype-config sidecars/fmnerg_subtype/roberta128_encoder_all.yaml \
  --subtype-checkpoint \
    outputs/fmnerg_roberta128_subtype_encoder_ablation/all_seed42/best_model.pt \
  --stage1-dev-cache \
    knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical.pt \
  --output outputs/fmnerg_fine/b0_seed42_dev.json \
  --device cuda

PYTHONPATH=. python scripts/analyze_fmnerg_r16_oracle.py \
  --cache knowledge/record_candidates/roberta128/fmnerg_dev_hierarchical.pt \
  --output outputs/fmnerg_fine/b0_seed42_r16_oracle.json
```

B0 定义为：

```text
old Stage1 bypass span/parent/region
+ F2 seed42 subtype on the same predicted spans
```

它不是完整 Evidence Visibility 的 FMNERG。

### 4.3 训练 Stage1-F seed 42

```bash
PYTHONPATH=. python scripts/train_fmnerg_stage1.py \
  --config configs/fmnerg_fine/stage1.yaml
```

该入口始终向底层训练脚本传入 `--skip-test-evaluation`。

预注册的 `lambda_f=0.5` 诊断运行使用独立目录：

```bash
PYTHONPATH=. python scripts/train_fmnerg_stage1.py \
  --config configs/fmnerg_fine/stage1.yaml \
  --lambda-fine-subtype 0.5 \
  --output-dir outputs/fmnerg_fine/stage1_seed42_lambda05
```

不得在看到 Dev 后继续扩大 loss-weight sweep。

### 4.4 Dev 评估与 R16

```bash
PYTHONPATH=. python scripts/evaluate_fmnerg_stage1.py \
  --config configs/fmnerg_fine/stage1.yaml \
  --checkpoint outputs/fmnerg_fine/stage1_seed42/best_model.pt \
  --output outputs/fmnerg_fine/stage1_seed42/dev_metrics.json \
  --device cuda

PYTHONPATH=. python scripts/build_record_candidate_cache.py \
  --config configs/fmnerg_fine/stage1.yaml \
  --checkpoint outputs/fmnerg_fine/stage1_seed42/best_model.pt \
  --split dev \
  --output knowledge/fmnerg_fine/seed42/r16_dev.pt \
  --k-best 6 \
  --max-span-candidates 12 \
  --top-m-types 3 \
  --boundary-shift 0 \
  --boundary-penalty 0.25 \
  --device cuda

PYTHONPATH=. python scripts/analyze_fmnerg_r16_oracle.py \
  --cache knowledge/fmnerg_fine/seed42/r16_dev.pt \
  --taxonomy sidecars/fmnerg_subtype/taxonomy_twitter10000.json \
  --output outputs/fmnerg_fine/stage1_seed42/r16_oracle.json
```

Fine Test cache 需要显式 `--allow-test`，F1 禁止使用该参数。

### 4.5 自动汇总

```bash
PYTHONPATH=. python scripts/summarize_fmnerg_stage1_f1.py \
  --baseline-b0 outputs/fmnerg_fine/b0_seed42_dev.json \
  --baseline-r16-oracle outputs/fmnerg_fine/b0_seed42_r16_oracle.json \
  --stage1-dev outputs/fmnerg_fine/stage1_seed42/dev_metrics.json \
  --stage1-r16-oracle outputs/fmnerg_fine/stage1_seed42/r16_oracle.json \
  --output outputs/fmnerg_fine/stage1_seed42/f1_summary.json
```

## 5. Gate

F1 只判断 seed42 是否存在继续做 F2 的信号：

```text
Fine MNER delta             >= +0.003
Stage1 FMNERG delta         >= +0.005
Span F1 delta               >= -0.003
R16 visible-region Oracle   >= baseline -0.002
Hierarchy consistency       = 1
test_accessed               = false
```

F1 通过不等于正式方法通过。F2 仍需 seed 41/42/43 至少 2/3 的 FMNERG
提升，之后才能进入后半链。

## 6. 状态

```text
Model-G M3.3A: frozen and unchanged
M3.3F F0: engineering implementation
M3.3F F1 lambda_f=1.0: formal no-go on seed42 Dev
M3.3F F1 lambda_f=0.5: formal no-go on seed42 Dev
Fully shared Stage1-F route: closed
M3.3F F2 / R36 / downstream chain: not started and not authorized
Test: forbidden during F0/F1
```

### 6.1 F1 `lambda_f=1.0` Dev result

Checkpoint selection used `fmnerg_score`; the independent evaluator and R16
oracle were run on the saved best checkpoint from epoch 11.

| Metric | Matched B0 | Stage1-F | Delta | Gate |
| --- | ---: | ---: | ---: | --- |
| Span F1 | 0.87072 | 0.86382 | -0.00690 | fail |
| Fine MNER F1 | 0.67660 | 0.66882 | -0.00779 | fail |
| EEG F1 | 0.64599 | 0.63820 | -0.00780 | diagnostic decline |
| GMNER F1 | 0.60733 | 0.60193 | -0.00540 | diagnostic decline |
| FMNERG F1 | 0.50946 | 0.50081 | -0.00866 | fail |
| R16 visible-region oracle | 0.83451 | 0.83451 | 0.00000 | pass |
| Hierarchy consistency | 1.00000 | 1.00000 | 0.00000 | pass |

The subtype classifier itself also declined under joint training:

| Gold-span diagnostic | Matched B0 | Stage1-F | Delta |
| --- | ---: | ---: | ---: |
| Subtype accuracy | 0.78939 | 0.77796 | -0.01143 |
| Subtype macro-F1 | 0.69061 | 0.67074 | -0.01987 |
| Parent-conditioned subtype accuracy | 0.83045 | 0.82505 | -0.00540 |

The unchanged R16 oracle rules out proposal recall as the cause. The joint
shared-backbone update produced negative transfer in both the main
span/grounding path and the subtype task. Training diagnostics also recorded
negative subtype-versus-NER gradient cosine during substantial parts of
training.

Formal artifacts:

```text
outputs/fmnerg_fine/stage1_seed42/dev_metrics.json
outputs/fmnerg_fine/stage1_seed42/r16_oracle.json
outputs/fmnerg_fine/stage1_seed42/f1_summary.json
```

### 6.2 F1 `lambda_f=0.5` Dev result

The only preregistered loss-weight diagnostic stopped at epoch 10. Checkpoint
selection used `fmnerg_score`, and the saved best checkpoint was epoch 7.

| Metric | Matched B0 | `lambda_f=1.0` | `lambda_f=0.5` | `0.5` delta vs B0 | Gate |
| --- | ---: | ---: | ---: | ---: | --- |
| Span F1 | 0.87072 | 0.86382 | 0.86523 | -0.00550 | fail |
| Fine MNER F1 | 0.67660 | 0.66882 | 0.66842 | -0.00818 | fail |
| EEG F1 | 0.64599 | 0.63820 | 0.63892 | -0.00708 | diagnostic decline |
| GMNER F1 | 0.60733 | 0.60193 | 0.60174 | -0.00559 | diagnostic decline |
| FMNERG F1 | 0.50946 | 0.50081 | 0.49869 | -0.01078 | fail |
| R16 visible-region oracle | 0.83451 | 0.83451 | 0.83451 | 0.00000 | pass |
| Hierarchy consistency | 1.00000 | 1.00000 | 1.00000 | 0.00000 | pass |

Gold-span subtype diagnostics also became worse than both B0 and the
`lambda_f=1.0` run:

| Gold-span diagnostic | Matched B0 | `lambda_f=0.5` | Delta |
| --- | ---: | ---: | ---: |
| Subtype accuracy | 0.78939 | 0.76898 | -0.02041 |
| Subtype macro-F1 | 0.69061 | 0.64743 | -0.04318 |
| Parent-conditioned subtype accuracy | 0.83045 | 0.82166 | -0.00879 |

Formal artifacts:

```text
outputs/fmnerg_fine/stage1_seed42_lambda05/dev_metrics.json
outputs/fmnerg_fine/stage1_seed42_lambda05/r16_oracle.json
outputs/fmnerg_fine/stage1_seed42_lambda05/f1_summary.json
```

The lower weight slightly recovered Span and EEG relative to `lambda_f=1.0`,
but did not produce net benefit against B0 and made subtype classification
worse. This is a structural negative-transfer result rather than evidence for
further scalar tuning. Per protocol, no additional loss weights, seeds 41/43,
R36 caches, downstream modules, or Test evaluation are run for this branch.
