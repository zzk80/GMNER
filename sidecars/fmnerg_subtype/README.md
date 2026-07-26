# Hierarchical Subtype Sidecar

## 目标

该分支在冻结现有 GMNER 主链的前提下，为每个正式预测实体增加 51 类
细粒度 subtype。GMNER 与 FMNERG 均作为主指标报告。

F0 只建立冻结纯文本细粒度类型基线，不使用图像、原型、SigLIP2 或外部知识。

```text
冻结 RoBERTa Stage1
    -> gold span 或正式 predicted span 的冻结表示
    -> LayerNorm
    -> Linear
    -> GELU
    -> Dropout
    -> Linear(51)
    -> predicted coarse type 父类屏蔽
    -> subtype
```

span 表示固定为：

```text
[首词首 subword; 末词末 subword; span 内全部 subword 均值]
```

## 不变量

Subtype Sidecar 只允许新增：

```text
subtype_id
subtype
Fine MNER metrics
FMNERG metrics
```

它不得修改：

```text
span
coarse type
region / NULL
预测顺序
原 GMNER 指标
```

评估包含两级硬校验：

1. 推理时比较附加 subtype 前后的 coarse prediction SHA-256 和完整指标字典；
2. 独立审计逐记录比较 span/type/region/order/gold target。

任一变化都会报错，不接受仅比较四舍五入后的 GMNER。

## 标签契约

固定词表位于：

```text
sidecars/fmnerg_subtype/taxonomy_twitter10000.json
```

共 51 类：

```text
LOC   11
PER   13
ORG   10
OTHER 17
```

训练使用 gold coarse type 屏蔽；正式推理必须使用 predicted coarse type
屏蔽。父类不一致的 subtype 永远不能被选中。

## 数据口径

训练：

```text
train gold span
-> frozen RoBERTa representation
-> gold subtype
```

诊断：

```text
dev gold span
-> subtype accuracy / micro-F1 / macro-F1
```

正式 dev：

```text
冻结 GMNER 正式 predicted span/type/region
-> subtype
-> Fine MNER / FMNERG
```

错误 span 不会被强行映射到邻近 gold subtype。它在正式端到端指标中自然判错。

## 一键运行

先暂停正在使用 GPU 的 OOF 进程，再执行：

```bash
cd ~/gmner

nohup env \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  CONFIG=sidecars/fmnerg_subtype/roberta128.yaml \
  DEVICE=cuda \
  bash tools/run_fmnerg_subtype_sidecar.sh \
  > fmnerg_subtype_sidecar.log 2>&1 &

echo $!
tail -f fmnerg_subtype_sidecar.log
```

流水线依次执行：

1. 导出冻结 Evidence Visibility 的 dev 正式预测；
2. 提取 train gold-span 冻结特征；
3. 提取 dev gold-span 冻结特征；
4. 提取 dev formal predicted-span 冻结特征；
5. 训练小型 Subtype Sidecar；
6. 同时评估 GMNER 与 FMNERG；
7. 做逐记录 GMNER 恒等审计。

该流程禁止读取 test。初次方法选择、早停和错误分析均只使用 dev。

## 输出

```text
knowledge/fmnerg_subtype_sidecar/roberta128/
  train_gold.pt
  dev_gold.pt
  dev_formal.pt
  dev_formal_predictions.json

outputs/fmnerg_roberta128_subtype_sidecar/
  best_model.pt
  train.log
  history.json
  train_summary.json
  dev_metrics.json
  gmner_identity_audit.json
  dev_error_analysis.json
```

重点检查：

```text
subtype_accuracy_on_gold_spans
subtype_micro_f1_on_gold_spans
subtype_macro_f1_on_gold_spans
subtype_accuracy_on_correct_predicted_spans
parent_conditioned_subtype_accuracy
span_f1
coarse_mner_f1
fine_mner_f1
eeg_f1
gmner_f1
fmnerg_f1
hierarchy_consistency_rate
gmner_identity_exact
test_accessed
```

必须满足：

```text
hierarchy_consistency_rate = 1
gmner_identity_exact = true
test_accessed = false
冻结主链 dev GMNER = 0.621316（容差 5e-7）
Fine MNER F1 <= Coarse MNER F1
FMNERG F1 <= GMNER F1
```

## 测试

不依赖 pytest：

```bash
cd ~/gmner
PYTHONPATH=. python -m unittest discover \
  -s sidecars/fmnerg_subtype/tests -v
```

测试覆盖：

```text
51 类 taxonomy 与 4 个父类计数
父类屏蔽
正式预测附加 subtype
GMNER 精确恒等
Fine MNER / FMNERG 计算
```

错误分析按 4 个父类、51 个 subtype、训练频率、visible/NULL 和 subtype
混淆分别报告 span、coarse type、subtype、grounding、GMNER 与 FMNERG
召回，用于确定下一阶段瓶颈。

## Subtype Loss 消融

冻结特征生成后，可在 CPU 上运行三种损失、三个随机种子的严格同口径消融：

```text
CE
Class-weighted CE: 1 / sqrt(n)
Effective-number CE: beta = 0.999
```

两种加权损失都在每个 coarse parent 内部归一化到均值 1。唯一变化是
subtype loss；表示、父类 mask、优化器、batch size、epoch、早停和正式
GMNER 预测全部固定。

```bash
cd ~/gmner

nohup env \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  DEVICE=cpu \
  CPU_THREADS=2 \
  SEEDS="41 42 43" \
  bash tools/run_fmnerg_subtype_loss_ablation.sh \
  > fmnerg_subtype_loss_ablation.log 2>&1 &
```

每次训练同时保存：

```text
best_model.pt                 # 配置的主指标，正式为 FMNERG
best_fmnerg_model.pt
best_subtype_macro_model.pt
```

最终汇总：

```text
outputs/fmnerg_roberta128_subtype_loss_ablation/summary.json
```

汇总器自动报告：

```text
三个 seed 的 mean / std / min / max
Fine MNER 与 FMNERG
Gold-span Accuracy 与 Macro-F1
父类正确条件下的 subtype Accuracy
高频 / 低频 Accuracy
PER / ORG Accuracy
Visible / NULL FMNERG Recall
头部类别 corrected / damaged / net
尾部类别 corrected / damaged / net
每个 seed 是否取得 FMNERG 正增益
```

预注册通过标准：

```text
FMNERG mean delta >= +0.005
Macro-F1 mean delta >= +0.015
低频 Accuracy 提升
高频 Accuracy 下降不超过 0.02
三个 seed 的 FMNERG 均高于对应 CE seed
GMNER identity 全部严格成立
test_accessed = false
```

### Dev 结果（3 seeds）

所有结果均使用 `best_fmnerg_model.pt`，随机种子为 `41/42/43`：

| Loss | Fine MNER F1 | FMNERG F1 | Gold Macro-F1 | Low Acc. | High Acc. |
| --- | ---: | ---: | ---: | ---: | ---: |
| CE | 0.61957 ± 0.00266 | **0.47719 ± 0.00238** | 0.52998 ± 0.01117 | 0.54678 | **0.81224** |
| Class-weighted CE | 0.60651 ± 0.00356 | 0.46723 ± 0.00467 | **0.53416 ± 0.01601** | **0.60526** | 0.78697 |
| Effective-number CE | 0.60436 ± 0.00760 | 0.46642 ± 0.00710 | 0.53229 ± 0.00837 | 0.58772 | 0.78300 |

相对 CE：

```text
Class-weighted: FMNERG -0.00996, tail net +20, head net -121
Effective-number: FMNERG -0.01077, tail net +14, head net -140
```

两种重加权均提高了低频类别准确率，但损伤了更多头部样本，且三个 seed
的 FMNERG 全部低于对应 CE。二者均未通过预注册标准，不进入正式模型。
本轮最佳单次结果为 CE seed 41、epoch 17：

```text
FMNERG F1 = 0.48042
Fine MNER F1 = 0.62334
GMNER F1 = 0.621316（逐记录恒等）
test_accessed = false
```

初始 Sidecar 的 `FMNERG=0.47033` 来自旧的 Fine-MNER checkpoint 选择口径；
本表改为按最终主指标 FMNERG 选择 checkpoint，因此应以三 seed CE
均值作为当前 dev 基线，不把单次最佳值写成正式泛化结果。

## Parent-specific Head 消融

当前 `shared_hard` 已在训练阶段 mask 非法父类。对任意固定 coarse
预测 `c`，合法 subtype 都满足 `parent(f)=c`，因此：

```text
argmax_f [z_f + lambda * log p(parent(f) | x)]
= argmax_f [z_f + lambda * log p(c | x)]
= argmax_f z_f
```

所以在同时要求以下两项时，soft coarse prior 不会改变任何预测：

```text
冻结 coarse type，保持 GMNER 逐记录恒等
最终 subtype 与 coarse type 100% 层次一致
```

若允许 soft prior 跨父类改变 subtype，则必须同步修改 coarse type，否则层次
不一致；同步修改 coarse type 又会改变 GMNER。故本阶段不把
`shared + soft prior` 和 `parent heads + soft gate` 伪装成有效消融。

真正测试的结构为：

```text
shared_hard:
  2304 -> shared hidden 768 -> 51 logits -> hard parent mask

parent_specific_hard:
  LOC/PER/ORG/OTHER 各自使用独立 2304 -> 192 -> local subtype MLP
  -> 拼接为 51 logits -> hard parent mask
```

四个父分支总参数量与 shared-768 相差低于 2%，且某个父类样本只向对应
分支传播梯度。这样检验的是跨父类共享表示是否有害，而不是模型容量增加。

冻结缓存上的三 seed CPU 消融：

```bash
cd ~/gmner

nohup env \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  DEVICE=cpu \
  CPU_THREADS=2 \
  SEEDS="41 42 43" \
  bash tools/run_fmnerg_subtype_head_ablation.sh \
  > fmnerg_subtype_head_ablation.log 2>&1 &
```

脚本复用现有 CE checkpoint 作为 `shared_hard`，仅重新评估新增诊断；
只训练 `parent_specific_hard`。输出：

```text
outputs/fmnerg_roberta128_subtype_head_ablation/summary.json
```

除 FMNERG、Fine MNER、Macro-F1 和父类条件准确率外，还报告：

```text
predicted-parent subtype accuracy
gold-parent Oracle subtype accuracy
coarse-wrong exact spans
coarse-wrong spans recoverable with the gold parent
head/tail corrected, damaged, and net
parameter-count ratio
```

`gold-parent Oracle` 只用于 dev 诊断，不参与正式推理。它比直接比较跨父类
raw logits 更可靠，因为 hard-mask CE 并未校准不同父类的 logit 尺度。

### Dev 结果（3 seeds）

| Head | Parameters | Fine MNER F1 | FMNERG F1 | Gold Acc. | Gold Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared-51 hard | 1,814,067 | **0.61957 ± 0.00266** | **0.47719 ± 0.00238** | 0.69102 | 0.52998 |
| Parent-specific hard | 1,798,515 | 0.61889 ± 0.00206 | 0.47490 ± 0.00305 | **0.69306** | **0.53238** |

Parent-specific 相对 Shared：

```text
FMNERG F1                         -0.00229
Fine MNER F1                      -0.00067
Gold-span subtype Accuracy        +0.00204
Gold-span subtype Macro-F1        +0.00239
Parent-conditioned Accuracy       -0.00082
Gold-parent Oracle Accuracy       -0.00031
Coarse-wrong Oracle recoverable   +1.00 example / seed
Head-frequency net                -30 across 3 seeds
Tail-frequency net                -10 across 3 seeds
```

三个 seed 的 FMNERG 均未超过对应 Shared 基线，因此该结构未通过验收。
这说明把 shared trunk 拆成父类专属表示只能带来很小的 gold-span
分类变化，无法改善正式 predicted-span FMNERG。当前正式 Sidecar
继续保留 `Shared-51 hard + CE`。

Coarse propagation 诊断显示，每个 seed 固定有 139 个 exact predicted span
父类错误；提供 gold parent 后，Shared 平均可恢复约 42.7 个 subtype，
Parent-specific 平均约 43.7 个。该 Oracle 说明 coarse error 确实造成一部分
上限损失，但仅更换 subtype head 无法解决它。任何允许跨父类修正的后续方法
必须作为会改变 coarse GMNER 的独立联合实验，不能继续声称 GMNER 恒等。

## Trainable RoBERTa Copy 消融

当前 FMNERG 的主要缺口来自 subtype，而不是正式 grounding。新的 encoder
消融采用“任务隔离、表示继承”：

```text
M3.3A formal chain（冻结、只读）
  -> fixed span / coarse type / region / NULL

Stage1 RoBERTa 参数副本（独立）
  -> 在线编码原始文本
  -> fixed span 的 start/end/mean 表示
  -> Shared-51 subtype head
  -> predicted coarse type hard mask
```

它不加载可训练的 Hierarchy、Coarse、Fine 或 Evidence 模块，也不会将梯度写回
`outputs/fmnerg_stage1_roberta128/`。正式 Dev 评估仍逐记录检查原始
span/type/region/order 和 GMNER SHA-256 完全一致。

三个受控版本为：

| ID | 可训练参数 | 配置 |
| --- | --- | --- |
| F0 | subtype MLP | `roberta128.yaml` |
| F1 | RoBERTa 最后 4 层 + subtype head | `roberta128_encoder_last4.yaml` |
| F2 | RoBERTa 全部层 + subtype head | `roberta128_encoder_all.yaml` |

F1/F2 第一轮继续使用 train gold span 和 gold parent；Dev 使用固定 M3.3A
predicted span 和 predicted parent。唯一新增变量是 encoder 的可训练范围，不同时
加入类别重加权、图像、外部知识、prototype 或新的 coarse head。

一键脚本优先复用 `frozen_seed*/dev_metrics.json`；若云端清理时已删除旧的
逐 seed 文件，则基于现有冻结 `train_gold/dev_gold/dev_formal` 特征在 CPU
快速重建 F0 CE seed 41/42/43。该步骤不重新编码 RoBERTa。汇总器随后将 F1/F2
与对应的 F0 seed 做配对比较。

脚本使用独占运行锁；中断后再次执行会跳过已有 `dev_metrics.json` 或
`train_summary.json` 的完整 seed。只有显式设置 `FORCE=1` 才会重跑。

优化设置：

```text
RoBERTa lower layers: 1e-6（仅 F2）
RoBERTa upper 4:      5e-6
Subtype head:         1e-4
Weight decay:         0.01
Warmup:               0.1
Epochs:               15
Early stopping:       Dev FMNERG
Gradient clipping:    1.0
```

先确保冻结正式预测文件已经由 F0 流水线生成，再运行三个 seed：

```bash
cd ~/gmner

nohup env \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  DEVICE=cuda \
  SEEDS="41 42 43" \
  bash tools/run_fmnerg_subtype_encoder_ablation.sh \
  > fmnerg_subtype_encoder_ablation.log 2>&1 &

echo $!
tail -f fmnerg_subtype_encoder_ablation.log
```

输出：

```text
outputs/fmnerg_roberta128_subtype_encoder_ablation/
  frozen_seed41/
  frozen_seed42/
  frozen_seed43/
  last4_seed41/
  last4_seed42/
  last4_seed43/
  all_seed41/
  all_seed42/
  all_seed43/
  summary.json
```

checkpoint 只保存相对正式 Stage1 初始化发生训练的 backbone 参数和 subtype
head；正式 Stage1 checkpoint 继续作为只读初始化来源。单次 Dev 复评：

```bash
PYTHONPATH=. python tools/evaluate_fmnerg_subtype_encoder.py \
  --config sidecars/fmnerg_subtype/roberta128_encoder_last4.yaml \
  --checkpoint outputs/fmnerg_roberta128_subtype_encoder_ablation/last4_seed42/best_model.pt \
  --output outputs/fmnerg_roberta128_subtype_encoder_ablation/last4_seed42/dev_metrics_recomputed.json \
  --device cuda
```

第一阶段验收固定为：

```text
主选择指标 = Dev FMNERG
三 seed mean FMNERG 相对 F0 至少 +0.005
Fine MNER 不下降
三个 seed 的 FMNERG 均高于对应 F0 seed
GMNER identity exact = true
formal_stage1_mutated = false
test_accessed = false
```

F1/F2 只在 Dev 上比较。选定唯一 scope 和学习率后，才允许对最终模型执行一次
Test；当前工具没有 Test 参数，防止把 Test 用作调参集。

## 与 NULL Release OOF 的关系

该实现全部位于：

```text
sidecars/
tools/
```

不修改 OOF source fingerprint 当前覆盖的：

```text
gmner/**/*.py
scripts/**/*.py
configs/**/*.yaml
```

因此完成 sidecar 实验后可以继续现有 OOF。若未来将 subtype 特征回写 Fine、
Evidence、Release、RoBERTa 或 coarse type，则不再满足该条件，需要重新审计
甚至重建整链 OOF。

## 从 Windows 同步到云端

PowerShell 中将目标替换为实际可解析的 SSH 地址，不要继续使用未配置的
`4090server` 别名：

```powershell
cd "C:\Users\Administrator\Desktop\代码\GMNER"
$Server = "zzk@<服务器IP或可解析主机名>"

ssh $Server "mkdir -p ~/gmner/sidecars ~/gmner/tools"
scp -r sidecars/fmnerg_subtype "${Server}:~/gmner/sidecars/"
scp sidecars/__init__.py "${Server}:~/gmner/sidecars/"

scp tools/export_fmnerg_formal_predictions.py `
    tools/build_fmnerg_subtype_features.py `
    tools/train_fmnerg_subtype_sidecar.py `
    tools/evaluate_fmnerg_subtype_sidecar.py `
    tools/audit_fmnerg_subtype_identity.py `
    tools/analyze_fmnerg_subtype_errors.py `
    tools/run_fmnerg_subtype_sidecar.sh `
    tools/summarize_fmnerg_subtype_loss_ablation.py `
    tools/run_fmnerg_subtype_loss_ablation.sh `
    tools/summarize_fmnerg_subtype_head_ablation.py `
    tools/run_fmnerg_subtype_head_ablation.sh `
    "${Server}:~/gmner/tools/"
```
