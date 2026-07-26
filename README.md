# GMNER / FMNERG

本仓库当前只以 **M3.3A** 作为正式主线。历史实验、失败分支和严格 OOF
基础设施仍可复现，但不属于正式推理链。

## Current Status

```text
Current formal method: M3.3A
Formal Dev GMNER:      0.621316
Formal Test MNER:      0.81843
Formal Test Fine MNER: 0.66144 +/- 0.00037
Formal Test EEG:       0.65216
Formal Test GMNER:     0.61529
Formal Test FMNERG:    0.50144 +/- 0.00133
```

- Dev/Test 结果已经冻结。
- M3.6 NULL Release 没有访问 Test，也没有进入正式链路。
- GMNER 与 FMNERG 都是主任务；51 类 subtype 由独立 sidecar 评估。
- FMNERG 使用 Dev 选定的全量解冻 RoBERTa 副本，Test 固定报告三个预定
  seed 的 mean/std，不按 Test 选择 seed。

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
| Dev | - | 0.816714 | 0.67488 +/- 0.00243 | 0.660880 | 0.621316 | 0.51729 +/- 0.00083 |
| Test | - | 0.81843 | 0.66144 +/- 0.00037 | 0.65216 | 0.61529 | 0.50144 +/- 0.00133 |

OOF Train 是 10 个 heldout fold、7000 条记录的严格 micro-average，不是
fold ensemble，也不会改变正式 Dev/Test 结果。其 fold GMNER 为
`0.610869 +/- 0.010907`，并满足：

```text
10 folds
700 records per fold
7000 unique records
no overlap / no missing records
test_accessed=false
```

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
  联合实验；使用 matched F2-continuation control，不修改正式
  span/type/region，详见
  [Joint Experiments](sidecars/fmnerg_joint/README.md)。
- `docs/HIERARCHICAL_RECORD_VERIFIER.md`：M2 到 M3.3A 的方法细节。
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
  `0.000000`；按预注册规则 no-go，未读取 Test。
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
