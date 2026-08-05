# 三条 Stage1 / GMNER 链路代码目录说明

本文档定义清理后的保留边界。当前保留三条链：

1. 正式最优 Model-G：M3.3A；
2. 失败对照一：DVH Frozen-CLIP Stage1；
3. 失败对照二：TQ-DV-MNER，以及只读 fixed-span type replay。

关闭实验的结果结论可保留在已提交归档中，但其训练入口、临时缓存、checkpoint、
日志和未提交实现不属于当前代码空间。

## 顶层目录

```text
GMNER/
├── configs/                 三条链的配置
├── docs/                    正式结果、方法说明和实验归档
├── gmner/                   数据、模型、损失和评估核心包
├── scripts/                 训练、缓存构建和评估入口
├── tests/                   三条链及共享合同测试
├── tools/                   正式 FMNERG 辅助工具，非三链主入口
├── GMNER-main/              Twitter10000、VinVL、XML 和图像数据
├── knowledge/               可重建或冻结特征缓存
├── outputs/                 checkpoint 与指标产物
├── roberta-base/            云端本地 RoBERTa 权重
└── clip-vit-base-patch32/   云端本地 Frozen CLIP 权重
```

## 链路一：正式 M3.3A

### 配置

```text
configs/fmnerg_twitter10000_stage1.yaml
configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml
configs/fmnerg_twitter10000_coarse_selector.yaml
configs/fmnerg_twitter10000_fine_grounding_adapter.yaml
configs/fmnerg_twitter10000_evidence_visibility.yaml
```

### 模型与损失

```text
gmner/models/gmner_model.py
gmner/models/text_encoder.py
gmner/models/graph_encoder.py
gmner/models/aligner.py
gmner/models/heads.py
gmner/models/hierarchical_record_verifier.py
gmner/models/coarse_region_selector.py
gmner/models/fine_grounding_adapter.py
gmner/models/evidence_visibility.py

gmner/losses/multitask.py
gmner/losses/hierarchical_record_candidate_loss.py
gmner/losses/coarse_region_selector_loss.py
gmner/losses/fine_grounding_adapter_loss.py
gmner/losses/evidence_visibility_loss.py
```

### 数据与评估

```text
gmner/data/mmner_dataset.py
gmner/data/collator.py
gmner/data/record_candidate_dataset.py
gmner/data/record_candidate_collator.py
gmner/data/paired_record_candidate_dataset.py
gmner/data/hierarchical_record_candidate_collator.py
gmner/data/formal_candidate_anchor.py

gmner/engine/evaluator.py
gmner/engine/trainer.py
gmner/engine/hierarchical_record_verifier_evaluator.py
gmner/engine/coarse_region_selector_evaluator.py
gmner/engine/fine_grounding_adapter_evaluator.py
gmner/engine/evidence_visibility_evaluator.py
gmner/engine/evidence_visibility_diagnostics.py
```

### 运行入口

```text
scripts/train.py
scripts/build_record_candidate_cache.py
scripts/train_hierarchical_record_verifier.py
scripts/evaluate_hierarchical_record_verifier.py
scripts/train_coarse_region_selector.py
scripts/evaluate_coarse_region_selector.py
scripts/train_fine_grounding_adapter.py
scripts/evaluate_fine_grounding_adapter.py
scripts/train_evidence_visibility.py
scripts/evaluate_evidence_visibility.py
```

### 云端保留资产

```text
outputs/fmnerg_stage1_roberta128/
outputs/fmnerg_roberta128_hierarchical_record_verifier/
outputs/fmnerg_roberta128_coarse_selector/
outputs/fmnerg_roberta128_fine_grounding_adapter/
outputs/fmnerg_roberta128_evidence_visibility/

knowledge/record_candidates/
knowledge/grounding/
roberta-base/
```

## 链路二：DVH Frozen-CLIP Stage1

### 专用代码

```text
configs/dvh_stage1/frozen_clip_vit_b32_seed42.yaml

gmner/models/dvh_stage1.py
gmner/losses/dvh_stage1_loss.py
gmner/data/dvh_record_collator.py
gmner/data/frozen_clip_cache.py

scripts/build_dvh_frozen_clip_cache.py
scripts/train_dvh_stage1.py
tests/test_dvh_stage1.py

docs/experiments/DVH_FROZEN_CLIP_STAGE1_PROTOCOL.md
docs/experiments/dvh_frozen_clip_stage1_protocol.json
```

### 与 TQ-DV 共用的 record-level 组件

```text
gmner/data/stage1_record_contract.py
gmner/data/record_level_stage1_dataset.py
gmner/data/record_level_stage1_collator.py

gmner/models/stage1/boundary_crf.py
gmner/models/stage1/span_type_head.py
```

### 云端保留资产

```text
outputs/dvh_stage1/frozen_clip_vit_b32_seed42/
knowledge/dvh_frozen_clip/
clip-vit-base-patch32/
```

## 链路三：TQ-DV-MNER

### 专用代码

```text
configs/tq_dv_mner/type_query_dual_visual_seed42.yaml

gmner/models/tq_dv_mner.py
gmner/losses/tq_dv_mner_loss.py
gmner/data/type_query_collator.py
gmner/engine/tq_mner_evaluator.py

scripts/train_tq_dv_mner.py
scripts/evaluate_tq_fixed_span_type_replay.py
tests/test_tq_dv_mner.py

docs/experiments/TQ_DV_MNER_README.md
docs/experiments/TQ_DV_FIXED_SPAN_REPLAY_RESULT.md
```

### 云端保留资产

```text
outputs/tq_dv_mner/type_query_dual_visual_seed42/
knowledge/tq_dv_mner/
```

TQ-DV 复用 DVH 的 Frozen CLIP cache，不重复保存 CLIP patch 特征。机器可读
replay 结果保留在对应输出目录，`docs/` 只保留紧凑结果说明。

## 三链比较文档

```text
docs/experiments/STAGE1_THREE_SYSTEM_COMPARISON.md
docs/experiments/THREE_CHAIN_CODE_LAYOUT.md
docs/EXPERIMENT_RESULTS_TABLE.md
docs/HIERARCHICAL_RECORD_VERIFIER.md
```

## 清理规则

可以删除：

```text
__pycache__ / .pytest_cache / tmp
根目录历史 *.log / *.pid
关闭实验的未提交代码入口
明确获准删除的关闭实验 checkpoint、optimizer state 和可重建缓存
.codex_sync_tmp / .codex_backups / .codex_transfers
```

不得删除：

```text
上述三链代码与配置
M3.3A R16/R36 candidate caches
M3.3A 五阶段 best checkpoints
DVH best checkpoint 与 Frozen CLIP cache
TQ-DV best checkpoint
TQ-DV fixed-span replay JSON 与训练 summary
RoBERTa/CLIP本地模型目录
Twitter10000、VinVL、XML和原始图像
```

当前归档节点明确要求暂时保留 DVH/TQ-DV checkpoint 和相关 cache；只有后续实验需要
释放空间时再单独清理，不能因 `NO_GO` 状态自动删除。
