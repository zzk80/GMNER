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

由于 F2 encoder 和 subtype head 也会继续更新，J0 相对冻结 F2 的总增益
不能直接归因为视觉。为此使用完全匹配的 C1：

```text
C1: F2 Continued
  - 与 J0 相同 seed、样本、batch、epoch、warmup、early stopping 和学习率
  - F2 encoder 与 subtype head 继续训练
  - visual residual 在 forward 中永久固定为 0

J0: Visual Fusion
  - 唯一额外变量是固定区域视觉 residual
```

配置：

```text
sidecars/fmnerg_joint/configs/c1_text_continuation.yaml
sidecars/fmnerg_joint/configs/j0_visual_fusion.yaml
```

测试会逐字段比较两份配置；除 `experiment_mode` 和 `output_dir` 外不允许
存在差异。

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

每个 C1/J0 seed 使用相同 seed 的 F2 checkpoint 初始化。只报告成对
mean/std，不选择最佳 seed。

整体方案通过：

```text
J0 相对初始 F2：
mean FMNERG delta >= +0.005
至少 2/3 seeds > 0
MNER / EEG / GMNER exact identity
Test accessed = false
```

视觉模块通过：

```text
J0 相对 C1：
mean FMNERG delta >= +0.003
至少 2/3 seeds > 0
Fine MNER mean 不下降
MNER / EEG / GMNER exact identity
```

`j0_visual_residual_fmnerg_delta` 表示 J0 最终视觉输出相对同一时刻文本分支
的差值；`fmnerg_delta_vs_initial_f2` 才表示整个训练方案相对原始 F2 的
差值。两者不能混用。

首次同步到训练环境后，三个 seed 分别执行只读预检：

```bash
for seed in 41 42 43; do
  PYTHONPATH=. /home/zzk/miniconda3/envs/gmner/bin/python \
    tools/train_fmnerg_joint_j0.py \
    --config sidecars/fmnerg_joint/configs/j0_visual_fusion.yaml \
    --seed "$seed" \
    --device cuda \
    --preflight
done
```

预检只验证缓存、checkpoint、epoch-0 F2 恒等性和冻结 GMNER，不执行反向
传播，也不访问 Test。正式 matched runner 会在训练前再次预检 C1/J0 的全部
seed。

正式运行顺序固定为 C1 三 seed、J0 三 seed、成对汇总：

训练前需将当前提交冻结为 tag：

```bash
git tag -a fmnerg-j0-matched-dev-preregistered -m \
  "Preregister matched C1/J0 Dev experiment"
```

runner 会拒绝 tag 未指向当前 `HEAD` 或存在 tracked file 修改的环境，并将
commit、tag 和两份配置的 SHA-256 写入
`outputs/fmnerg_joint_matched/protocol_manifest.json`。

```bash
cd ~/gmner

nohup env \
  PYTHONPATH=. \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  DEVICE=cuda \
  bash tools/run_fmnerg_joint_matched.sh \
  > fmnerg_joint_matched.log 2>&1 &

echo $!
tail -f fmnerg_joint_matched.log
```

结果：

```text
outputs/fmnerg_joint_c1/seed41..43/
outputs/fmnerg_joint_j0/seed41/
outputs/fmnerg_joint_j0/seed42/
outputs/fmnerg_joint_j0/seed43/
outputs/fmnerg_joint_matched/c1_dev_summary.json
outputs/fmnerg_joint_matched/j0_dev_summary.json
outputs/fmnerg_joint_matched/matched_dev_summary.json
```

## Formal Dev Result

预注册版本：

```text
date:   2026-07-26
commit: ae8553a
tag:    fmnerg-j0-matched-dev-preregistered
seeds:  41 / 42 / 43
```

三组匹配结果：

| Dev 指标 | 初始 F2 | C1 Continued | J0 Visual Fusion |
|---|---:|---:|---:|
| FMNERG F1 mean | 0.517292 | 0.517292 | 0.517292 |
| FMNERG F1 std | 0.000830 | 0.000830 | 0.000830 |
| Fine MNER F1 mean | 0.674876 | 0.674876 | 0.674876 |
| best epoch (41/42/43) | 0/0/0 | 0/0/0 | 0/0/0 |

配对增量：

```text
C1 - initial F2 = 0.000000
J0 - initial F2 = 0.000000
J0 - C1         = 0.000000
```

冻结主链恒等：

```text
Coarse MNER F1 = 0.816714
EEG F1         = 0.660880
GMNER F1       = 0.621316
formal prediction changed = 0
test_accessed = false
```

正式判断：

```text
整体 J0 方案：no-go
视觉模块独立贡献：no-go
C1 继续训练收益：无
```

所有已训练 epoch 的 Dev FMNERG 均未超过对应 seed 的 epoch 0，因此早停
一致保留初始 F2。该结果说明在当前“Train gold-positive region / Dev formal
region”的 J0 口径下，没有证据支持固定区域视觉 residual。按照预注册规则，
不启动 J1/J2，不为 J0 构建 OOF formal-region 特征，也不读取 Test。

机器可读归档：

```text
docs/experiments/fmnerg_joint_j0_matched_protocol_manifest.json
docs/experiments/fmnerg_joint_j0_matched_dev_summary.json
```

C1/J0 的工具没有 Test 参数。F2 Test 已经冻结，不能用于 J0/J1/J2 的
结构、学习率、epoch 或阈值选择。
