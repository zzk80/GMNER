# Strict OOF and NULL Release

## Status

```text
OOF infrastructure: retained, analysis-only
M3.6A-r2 NULL Release: archived no-go
Formal test accessed by M3.6A-r2: false
```

严格 OOF 用于生成无泄漏训练特征、审计泛化差距，以及验证后置模块。它不是
M3.3A 正式 Dev/Test 推理链，也不是 fold ensemble。

## Contract

固定一份 10-fold manifest。每个 heldout 样本使用的全部监督模块都只能由该折
其余 90% 训练数据得到：

```text
Fold train IDs
  -> RoBERTa Stage1
  -> R16 / R36 candidates
  -> Hierarchical Verifier
  -> Coarse Selector
  -> Fine Grounding Adapter
  -> Evidence Visibility
  -> optional Fusion Reliability
  -> heldout_features.pt
```

每折 proof 必须验证：

- train IDs 与 heldout IDs 无交集；
- heldout IDs 跨折无重复，十折并集覆盖全部 7000 条训练记录；
- 所有监督 checkpoint 均带训练来源和 SHA-256；
- R36 的正式 span/type 锚定到同折 R16 输出；
- Top-4 区域顺序在物化时固定；
- `test_accessed=false`。

每折完成后执行 `seal -> validate -> archive -> cleanup`，永久保留
`heldout_features.pt`、proof、manifest、配置和摘要日志。

## OOF Result

| Metric | Micro F1 |
| --- | ---: |
| Span | 0.870900 |
| MNER | 0.811690 |
| EEG | 0.651135 |
| GMNER | 0.610849 |

统计口径：

```text
records: 7000
folds: 10 x 700
predictions: 12001
gold entities: 11779
GMNER fold mean/std: 0.610869 +/- 0.010907
```

聚合命令：

```bash
PYTHONPATH=. python scripts/aggregate_m33a_oof_metrics.py \
  --feature-root knowledge/null_release_oof/roberta128 \
  --source-file GMNER-main/Twitter10000_v2.0/txt_fine/train.txt \
  --output outputs/fmnerg_roberta128_m33a_oof_train/metrics.json
```

## NULL Release Conclusion

M3.6A-r2 只处理 formal decode 后为 NULL 的实体：

```text
KEEP (fixed utility 0)
RELEASE_TO_VISIBLE
  -> Fine Top-4 region selection
```

严格 OOF 训练后，最优模型仍为 epoch-0 KEEP。学习分支没有稳定超过正式
M3.3A 输出，因此：

- 不接入正式推理；
- 不读取 Test；
- 不继续扫描阈值或增加控制器复杂度；
- 最小实现仅作为 no-go 复现材料保留。

## Retained Infrastructure

```text
scripts/run_null_release_full_chain_oof_fold.py
scripts/build_null_release_oof_features.py
scripts/merge_null_release_oof_features.py
scripts/aggregate_m33a_oof_metrics.py
tools/archive_null_release_oof_fold.py
tools/pull_null_release_oof_archives.ps1
tools/run_null_release_oof_folds_streaming.sh
gmner/data/full_chain_oof_contract.py
gmner/data/null_release_oof_cache.py
```

阶段完成前的完整历史实现可从 Git tag
`m3.6a-r2-oof-complete` 恢复。
