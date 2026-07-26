# FMNERG Joint Experiments

本目录用于 FMNERG subtype-region 联合实验，与冻结的 M3.3A 正式链和已完成
F2 Test 完全隔离。

## Version Boundary

正式基线冻结在：

```text
tag:    fmnerg-f2-test-frozen
commit: 9b8e7bc
branch: main
```

实验分支：

```text
experiment/fmnerg-joint-grounding
```

不得覆盖以下正式资产：

```text
outputs/fmnerg_stage1_roberta128/
outputs/fmnerg_roberta128_hierarchical_record_verifier/
outputs/fmnerg_roberta128_coarse_selector/
outputs/fmnerg_roberta128_fine_grounding_adapter/
outputs/fmnerg_roberta128_evidence_visibility/
outputs/fmnerg_roberta128_subtype_encoder_ablation/
```

## J0: Fixed-Region Visual Subtype Fusion

J0 是第一项保守实验：

```text
冻结 M3.3A span / coarse type / region / NULL
                    +
已接受的 F2 RoBERTa subtype encoder 副本
                    +
固定 region feature / geometry / detector score
                    ↓
零初始化 subtype residual
                    ↓
51 类 hierarchy-masked subtype
```

形式上：

```text
z_final(f) = z_F2(f) + scale * tanh(delta_visual(f | text, fixed_region))
```

J0 不输出 region logits，也没有 region 解码接口。因此：

```text
MNER exact identity = true
EEG exact identity = true
GMNER exact identity = true
```

epoch 0 的视觉 residual 严格为 0，必须逐例复现对应 seed 的 F2 subtype
预测。该检查失败时训练会直接终止。

## Data Contract

Train 使用 gold span，并从冻结 R36 cache 中读取：

```text
visible: 最高 IoU 的 gold-positive region
NULL:    固定 NULL evidence
missing: zero feature + visual_available=false
```

Dev 的正式 FMNERG 评估不使用 gold region，而是读取 M3.3A 已部署的
formal region/NULL。formal prediction JSON 中记录的 R36 SHA-256 必须与
实际 cache 完全一致。

因此 J0 回答的是：

> 在不改变 grounding 的条件下，正确/正式区域的视觉证据能否改善 subtype？

Train 的 gold-region 与 Dev formal-region 存在有意保留的监督差异。若 J0
有效，下一阶段再构造 OOF formal-region 训练特征；若 J0 无效，不为该方向
重建十折缓存。

## Trainable Parameters

J0 可训练：

```text
F2 独立 RoBERTa 副本
F2 subtype head
text / region / scalar projections
visual fusion residual
```

J0 永久冻结：

```text
正式 Stage1
R16 / R36 候选集合
Hierarchical Verifier
Coarse Selector
Fine Adapter
Evidence Visibility
最终 region / NULL
```

J1（Fine Adapter 副本）和 J2（Evidence Visibility 副本）尚未实现，也不能
通过配置开关启用。它们必须在 J0 通过后分别预注册。

## Dev Protocol

固定三个 seed：

```text
41 / 42 / 43
```

每个 J0 seed 使用相同 seed 的 F2 checkpoint 初始化。只报告 mean/std，不
选择最佳 seed。

预注册最低通过条件：

```text
mean FMNERG delta >= +0.005
至少 2/3 seeds 的 FMNERG delta > 0
MNER / EEG / GMNER exact identity
Test accessed = false
```

首次同步到训练环境后先执行只读预检：

```bash
PYTHONPATH=. /home/zzk/miniconda3/envs/gmner/bin/python \
  tools/train_fmnerg_joint_j0.py \
  --config sidecars/fmnerg_joint/configs/j0_visual_fusion.yaml \
  --seed 41 \
  --device cuda \
  --preflight
```

预检只验证缓存、checkpoint、epoch-0 F2 恒等性和冻结 GMNER，不执行反向
传播，也不访问 Test。

运行：

```bash
cd ~/gmner

nohup env \
  PYTHONPATH=. \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  DEVICE=cuda \
  bash tools/run_fmnerg_joint_j0.sh \
  > fmnerg_joint_j0.log 2>&1 &

echo $!
tail -f fmnerg_joint_j0.log
```

结果：

```text
outputs/fmnerg_joint_j0/seed41/
outputs/fmnerg_joint_j0/seed42/
outputs/fmnerg_joint_j0/seed43/
outputs/fmnerg_joint_j0/dev_summary.json
```

J0 的工具没有 Test 参数。F2 Test 已经冻结，不能用于 J0/J1/J2 的结构、
学习率、epoch 或阈值选择。
