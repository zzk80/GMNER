# GMNER / FMNERG

本仓库当前只以 **M3.3A** 作为正式主线。历史实验、失败分支和严格 OOF
基础设施仍可复现，但不属于正式推理链。

独立 FMNERG `Model-F / M3.3F` 的完全共享 Stage1-F 已完成 Dev-only F1：
`lambda_f=1.0` 和唯一预注册诊断 `0.5` 均未通过 Gate，因此该分支正式
no-go，未运行其他 seed、R36、后半链或 Test。固定 taxonomy、Fine 指标和
subtype-aware R16 工程保留用于复现。协议与结果见
[`docs/FMNERG_FULL_CHAIN.md`](docs/FMNERG_FULL_CHAIN.md)。

## Current Status

```text
Current formal Model-G: M3.3A
Current formal Model-F: F3 lower-LR subtype sidecar
Formal Dev GMNER:      0.621316
Formal Test MNER:      0.81843
Formal Test Fine MNER: 0.66510 +/- 0.00160
Formal Test EEG:       0.65216
Formal Test GMNER:     0.61529
Formal Test FMNERG:    0.50431 +/- 0.00111
```

- Dev/Test 结果已经冻结。
- M3.6 NULL Release 没有访问 Test，也没有进入正式链路。
- GMNER 与 FMNERG 都是主任务；51 类 subtype 由独立 sidecar 评估。
- FMNERG 使用 Dev 选定的全量解冻 RoBERTa 副本；F3 仅将 lower backbone
  learning rate 从 `1e-6` 调整为 `2e-6`。
- F3 Test 固定报告三个预定 seed 的 mean/std，不按 Test 选择 seed；F2 Test
  是执行 F3 前已知的历史正式基线。

## Formal Architecture

```text
RoBERTa Stage1
  -> R16 formal span/type candidates
  -> R36 expanded region candidates
  -> Hierarchical Record Verifier
  -> Base Top-8 + Learned Top-8
  -> Fine Grounding Adapter
  -> Evidence Visibility
  -> Record-level Decode
```

边界约束：

- R16 决定正式 span 和 coarse type。
- R36 只扩展区域候选，不覆盖正式 span/type。
- 后续 grounding 模块不改变 MNER。
- FMNERG subtype sidecar 不修改 GMNER 主链的 coarse type。

## Formal Results

| Split | Span F1 | MNER F1 | Fine MNER F1 | EEG F1 | GMNER F1 | FMNERG F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OOF Train | 0.870900 | 0.811690 | - | 0.651135 | 0.610849 | - |
| Dev | 0.87283 | 0.816714 | 0.68039 ± 0.00297 | 0.660880 | 0.621316 | 0.52052 ± 0.00219 |
| Test | 0.86980 | 0.818431 | 0.66510 ± 0.00160 | 0.652157 | 0.615294 | 0.50431 ± 0.00111 |

**说明：**
- **OOF Train:** 10-fold 整链 OOF，7000 条记录严格 pooled micro F1；fold-level mean ± std = 0.610869 ± 0.010907
- **Dev/Test:** 正式锁定结果；F3 在原子 seal 后一次性访问 Test
- **Fine MNER/FMNERG:** 三 seed (41/42/43) 报告 mean ± std，其他指标逐记录保持不变
- **F3 vs F2 Test:** Fine MNER `+0.00366`，FMNERG `+0.00288`

### 完整实验结果

详细的阶段结果、历史对照、Oracle 诊断和验收标准见：

- **[实验结果总表](docs/EXPERIMENT_RESULTS_TABLE.md)** — 所有有效实验的统一索引（31+ 实验）
- **[实验验收标准](docs/EXPERIMENT_ACCEPTANCE_CRITERIA.md)** — 7 种状态标签的验收规则

**状态标签体系：**
- **FORMAL:** 锁定且符合正式 Test 协议（Test 主表或消融表）
- **VALID_DEV:** 有效 Dev 实验或消融
- **VALID_AUDIT:** Train-OOF、cross-fit、分布和协议审计
- **ORACLE:** 使用 gold 的理想理论上限（方法空间分析）
- **ENGINEERING_ONLY:** 有信号但协议不满足正式报告要求
- **NO_GO:** 预注册 Gate 失败，分支关闭
- **ENGINEERING_HISTORY:** 早期探索和历史 Test 结果（仅归档）

## Repository Layout

正式配置：

```text
configs/fmnerg_twitter10000_stage1.yaml
configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml
configs/fmnerg_twitter10000_coarse_selector.yaml
configs/fmnerg_twitter10000_fine_grounding_adapter.yaml
configs/fmnerg_twitter10000_evidence_visibility.yaml
```

正式入口位于 `scripts/`；模型、数据和评估实现位于 `gmner/`。

其他目录：

- `sidecars/fmnerg_subtype/`：独立 51 类 FMNERG subtype 评估链；包含冻结
  F0 以及“RoBERTa 最后 4 层 / 全量解冻”的隔离副本消融，详见
  [Subtype Sidecar](sidecars/fmnerg_subtype/README.md)。
- `sidecars/fmnerg_joint/`：读取冻结 M3.3A region 的 subtype-region
  联合实验；matched C1/J0 已判定 no-go 并关闭。目录当前只保留复现代码以及
  新结构实施前的 Dev-only R36 subtype 可分性 Oracle，不修改正式
  span/type/region，详见
  [Joint Experiments](sidecars/fmnerg_joint/README.md)。
- `docs/HIERARCHICAL_RECORD_VERIFIER.md`：M2 到 M3.3A 的方法细节。
- `docs/FMNERG_FULL_CHAIN.md`：独立 M3.3F 的 F0/F1 契约、Gate 和 Dev-only
  运行流程。
- `docs/EXPERIMENT_SUMMARY.md`：历史实验和负结果。
- `docs/OOF_NULL_RELEASE.md`：严格 OOF 契约及 M3.6A-r2 no-go 结论。
- `docs/experiments/`：不属于正式推理链的诊断实验。

## Reproduction

训练 Stage1：

```bash
PYTHONPATH=. python scripts/train.py \
  --config configs/fmnerg_twitter10000_stage1.yaml
```

训练后续正式模块：

```bash
PYTHONPATH=. python scripts/train_hierarchical_record_verifier.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml

PYTHONPATH=. python scripts/train_coarse_region_selector.py \
  --config configs/fmnerg_twitter10000_coarse_selector.yaml

PYTHONPATH=. python scripts/train_fine_grounding_adapter.py \
  --config configs/fmnerg_twitter10000_fine_grounding_adapter.yaml

PYTHONPATH=. python scripts/train_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml
```

正式 Dev 评估：

```bash
PYTHONPATH=. python scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --split dev
```

聚合已经物化的严格 OOF 特征：

```bash
PYTHONPATH=. python scripts/aggregate_m33a_oof_metrics.py \
  --feature-root knowledge/null_release_oof/roberta128 \
  --source-file GMNER-main/Twitter10000_v2.0/txt_fine/train.txt \
  --output outputs/fmnerg_roberta128_m33a_oof_train/metrics.json
```

完整候选缓存构建、阶段输入输出和一次性 Test 规范见
[`docs/HIERARCHICAL_RECORD_VERIFIER.md`](docs/HIERARCHICAL_RECORD_VERIFIER.md)。

## Archived Experiments

- **FMNERG J0 matched control**：C1 continued-F2 与 fixed-region visual
  fusion 的三 seed Dev 最优均为 epoch 0，J0 相对 C1 的 FMNERG 增量为
  `0.000000`；按预注册规则 no-go，未读取 Test。该结论关闭 J0/J1/J2，
  后续 F3 仅采用独立的学习率单变量优化。
- **Subtype-region successor gate**：在实现
  `Top-K regions + subtype-conditioned attention` 前，先使用 Train 视觉
  subtype centroid 对 Dev 的“GMNER 正确但 subtype 错误”切片执行只读 R36
  Oracle。Top-4 相对 formal region 每个 seed 仅新增约 `5.33` 个 pairwise
  recovery，且 Top-4 已等于完整 R36 上限；严格 sibling top-1 只有
  `22.33%`，因此该后继结构也判定 no-go，不改变正式 Dev/Test。
- **M3.6A-r1**：非 OOF Dev 达到 `0.623738`，不能作为正式提升。
- **M3.6A-r2**：严格 10-fold full-chain OOF 下，最优 checkpoint 为
  epoch-0 KEEP，无可部署收益，状态为 **archived no-go**。
- **M3.4A SigLIP2 Reliability**：冻结旁路诊断，未达到正式接入门槛。
- Prototype、external knowledge、flat verifier 和旧 action controller 均不属于
  当前正式方法。

历史完成点保存在 Git tag `m3.6a-r2-oof-complete`。

## Validation

```bash
python -m pytest
ruff check .
python -m compileall gmner scripts sidecars tools
```

正式结果复现期间不得使用 Test 选择 checkpoint、阈值或超参数。
