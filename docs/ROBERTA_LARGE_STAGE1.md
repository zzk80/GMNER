# RoBERTa-large Stage1 验证协议

## 状态

当前只启用 Phase 1：

```text
RoBERTa-base Stage1
vs
RoBERTa-large Stage1 Seed 42
```

使用 Train 训练、Dev 选择和验收，不构建 Test 数据集，不运行 Test。Phase 2
及后续 R16/R36 重建必须等待 Phase 1 Gate。

## 对原方案的工程修正

当前仓库的训练入口是 `scripts/train.py`，评估入口是
`scripts/evaluate.py`。不存在 `train_stage1.py`、`evaluate_stage1.py`、
`generate_r16_cache.py` 或 `generate_r36_cache.py`。

RoBERTa-large 输出维度为 1024，但当前 `GMNERModel` 会通过
`text_projector` 投影到共享的 768 维图编码和 grounding 空间。因此本实验保持：

```text
model.hidden_size = 768
model.projection_dim = 768
```

只替换文本骨干。不能同时把图编码器、图卷积和 grounding 隐层改成 1024，
否则无法区分收益来自模型容量还是整条架构扩容。

基线 Stage1 bypass 指标为：

| Metric | Dev |
| --- | ---: |
| Span F1 | 0.870721 |
| Token F1 | 0.842912 |
| MNER F1 | 0.814740 |
| EEG F1 | 0.645993 |
| GMNER F1 | 0.607330 |

`0.872830` 是 Hierarchical Verifier 后的 Span F1，不是 Phase 1 的
Stage1 基线。

## 受控训练配置

候选配置：

```text
configs/fmnerg_twitter10000_stage1_roberta_large.yaml
```

相对 RoBERTa-base 只允许以下差异：

```text
text_model_name
batch_size: 8 -> 4
gradient_accumulation_steps: 1 -> 2
output_dir
```

有效 batch size 仍为 8。Dropout、损失、学习率、20 epoch 预算和随机种子
全部保持一致。Checkpoint 继续按 `gmner_score` 保存，不能改按 Span F1
保存，否则可能选择 grounding 更差的模型。

## Phase 1 Gate

容量信号：

```text
Span F1 delta >= +0.010
或
MNER F1 delta >= +0.010
```

安全约束：

```text
EEG F1 delta >= -0.002
GMNER F1 delta >= -0.002
```

容量信号与安全约束同时成立才进入 Phase 2。完整定义位于：

```text
docs/experiments/roberta_large_stage1_phase1_protocol.yaml
```

## 云端执行

先下载最小模型文件：

```bash
cd ~/gmner

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_XET=1

PYTHONPATH=. /home/zzk/miniconda3/envs/gmner/bin/python \
  tools/download_roberta_large.py
```

前台预检：

```bash
PYTHONPATH=. /home/zzk/miniconda3/envs/gmner/bin/python \
  tools/preflight_roberta_large_stage1.py
```

后台训练：

```bash
nohup env \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  MIN_FREE_GB=4 \
  MIN_GPU_FREE_MB=22000 \
  GPU_RESERVE_GB=8 \
  bash tools/run_roberta_large_stage1_phase1.sh \
  > roberta_large_stage1_phase1.log 2>&1 &

echo $! > roberta_large_stage1_phase1.pid
tail -f roberta_large_stage1_phase1.log
```

结果文件：

```text
outputs/fmnerg_stage1_roberta_large_seed42/preflight.json
outputs/fmnerg_stage1_roberta_large_seed42/baseline_recomputed/dev_metrics_from_checkpoint.json
outputs/fmnerg_stage1_roberta_large_seed42/train_summary.json
outputs/fmnerg_stage1_roberta_large_seed42/dev_metrics_from_checkpoint.json
outputs/fmnerg_stage1_roberta_large_seed42/phase1_summary.json
outputs/fmnerg_stage1_roberta_large_seed42/phase1_report.md
```

## Phase 2 边界

Phase 1 通过后才使用现有 `scripts/build_record_candidate_cache.py`：

```text
R16: --max-regions 16
R36: --max-regions 36 --formal-anchor-cache <R16 cache>
```

然后训练 `scripts/train_hierarchical_record_verifier.py`。Phase 2 仍只使用
Train/Dev，不访问 Test，也不直接重建 Coarse、Fine、Evidence 或 OOF。
