# M3.4A Multi-scale SigLIP 2 Region Reliability

M3.4A 是当前最优 FMNERG 链路之外的独立诊断实验。它只验证冻结 SigLIP 2
是否提供跨样本可比较的实体-区域语义证据，不改变当前正式输出，也不读取 test。

## 1. 冻结基线

正式链路保持不变：

```text
RoBERTa-128 Stage1
  -> R16 formal / R36 expanded candidates
  -> hierarchical verifier
  -> Base Top8 + Learned Top8 coarse selection
  -> Fine Grounding Adapter
  -> Evidence Visibility
  -> record decode
```

当前一次性 test 最优结果继续冻结：

```text
MNER F1  0.81843
EEG F1   0.65216
GMNER F1 0.61529
```

M3.4A 会加载并冻结当前最优 Evidence Visibility checkpoint，用它产生正式链路的
`KEEP` 决策以及 hard A/B 标签；该 checkpoint 不参与更新。独立 Region
Reliability Head 仍只在 train 上训练、只在 dev 上选模型和阈值。

因此这里的“当前 Visibility 输出 NULL”严格指 M3.3A Evidence Visibility 的最终
双阈值输出，而不是层次模型内部尚未修正的 visibility 概率。

## 2. 为什么使用 SigLIP 2

VinVL、Coarse 和 Fine 分数主要描述同一候选集合内的相对排序。已有
旧版 VinVL-only Reliability 的 hard A/B AUROC 最高约为 `0.6393`，但它只冻结到
Fine Adapter。本轮 M3.4A 以 Evidence Visibility 的最终 `KEEP` 决策重新定义 hard
A/B，因此二者不是同一评价集合，不能写成 `0.6393 -> 0.6003` 的性能回退。两组
实验共同说明现有候选排序特征不足以稳定判断“最高分框本身是否可信”。

[SigLIP 2](https://huggingface.co/docs/transformers/model_doc/siglip2)可直接产生图像和
文本表示；本实验使用
[`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224)，
冻结全部参数，只离线缓存表示。

每个真实 R36 候选框使用三种图像视图：

```text
local    = 候选框内部
context  = 以候选框中心扩展 1.5 倍后的正方形上下文
global   = 全图
```

每个候选实体 span 使用三种文本视图：

```text
mention  = 实体表面文本
context  = 固定模板中的完整推文和实体
type     = 使用 Stage1 formal type 的固定类型模板
```

三乘三产生 9 个未经过候选 softmax 的原生匹配 logit。模型同时使用 local/global、
context/global 相似度、候选内 margin/entropy，以及 SigLIP 2 与 Base/Coarse/Fine
top-1 一致性。候选内 softmax 仅用于 margin 和 entropy，不能替代绝对匹配 logit。

## 3. 实现边界

新增入口：

```text
scripts/build_siglip2_region_cache.py
scripts/train_siglip2_region_reliability.py
scripts/evaluate_siglip2_region_reliability.py
```

归档阶段使用过的配置：

```text
configs/fmnerg_twitter10000_siglip2_reliability_fusion.yaml
```

主分支只保留 OOF 仍依赖的 Fusion 配置。VinVL-only 与 SigLIP2-only 配置已随
M3.4A 归档，可从 Git tag `m3.6a-r2-oof-complete` 恢复。

特征缓存使用分片格式：

```text
knowledge/siglip2_region_cache/roberta128_r36/
  train/
    manifest.json
    shards/shard_00000.pt
  dev/
    manifest.json
    shards/shard_00000.pt
```

`manifest.json` 固定以下指纹：

```text
SigLIP 2 model / processor SHA256
R16 / R36 candidate cache SHA256
source data SHA256
input resolution
context expansion ratio
text and image view definitions
```

每条记录还保存图像、候选框和 record id 指纹。训练前会再次验证候选文件 SHA、
span 边界和区域框；候选缓存变化后旧 SigLIP 2 缓存会被拒绝。

## 4. 环境隔离

仓库正式环境锁定的 `transformers==4.39.0`，不能稳定加载该 SigLIP 2
checkpoint。为避免升级影响当前 RoBERTa 主链，同时减少云端磁盘占用，使用正式
环境的包作为只读基础建立轻量 venv，并仅在 venv 中覆盖 Transformers：

```bash
cd ~/gmner

/home/zzk/miniconda3/envs/gmner/bin/python -m venv \
  --system-site-packages ~/venvs/gmner-siglip2-cache

PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
  ~/venvs/gmner-siglip2-cache/bin/python -m pip install \
  --no-cache-dir --upgrade -r requirements-siglip2.txt

~/venvs/gmner-siglip2-cache/bin/python - <<'PY'
import transformers
print(transformers.__version__)
assert transformers.__version__ == "4.50.3"
PY
```

`transformers==4.49.0` 会把该 FixRes checkpoint 错配到旧
`SiglipTokenizer`，因此这里固定为已实际验收的 `4.50.3`。完成特征缓存后仍使用
原 `gmner` Conda 环境训练 Reliability Head；训练阶段不会导入 SigLIP 2 模型。

## 5. 下载模型

```bash
cd ~/gmner

export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DOWNLOAD_TIMEOUT=300
export HF_HUB_ETAG_TIMEOUT=60
export HF_HUB_DISABLE_XET=1

~/venvs/gmner-siglip2-cache/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="google/siglip2-base-patch16-224",
    local_dir="/home/zzk/gmner/siglip2-base-patch16-224",
    allow_patterns=[
        "*.json",
        "*.model",
        "model.safetensors",
    ],
)
print("DOWNLOAD_OK")
PY
```

下载后检查：

```bash
ls -lh siglip2-base-patch16-224/model.safetensors

TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
~/venvs/gmner-siglip2-cache/bin/python - <<'PY'
from transformers import AutoModel, AutoProcessor

path = "/home/zzk/gmner/siglip2-base-patch16-224"
AutoProcessor.from_pretrained(path, local_files_only=True, use_fast=False)
model = AutoModel.from_pretrained(path, local_files_only=True)
print(type(model).__name__)
print("MODEL_OK")
PY
```

## 6. 构建缓存

先检查磁盘。模型约 1.5 GB，train/dev FP16 分片缓存还需要额外空间：

```bash
df -h ~/gmner
```

正式构建命令：

```bash
cd ~/gmner

for split in train dev; do
  PYTHONPATH=. ~/venvs/gmner-siglip2-cache/bin/python \
    scripts/build_siglip2_region_cache.py \
    --formal-cache knowledge/record_candidates/roberta128/fmnerg_${split}_hierarchical.pt \
    --expanded-cache knowledge/record_candidates/roberta128/fmnerg_${split}_hierarchical_r36.pt \
    --source-file GMNER-main/Twitter10000_v2.0/txt_fine/${split}.txt \
    --image-dir GMNER-main/Twitter10000_v2.0/images \
    --model-name /home/zzk/gmner/siglip2-base-patch16-224 \
    --output-dir knowledge/siglip2_region_cache/roberta128_r36/${split} \
    --split ${split} \
    --context-expansion 1.5 \
    --batch-size 64 \
    --shard-size 128 \
    --fp16 \
    --resume \
    --device cuda
done
```

`--resume` 可以安全续跑已完成的分片；模型、预处理、候选缓存或 crop 参数改变时，
签名检查会阻止错误续跑。首次工程检查可以加 `--max-records 16`，确认后删除该参数
并保留 `--resume`，脚本会继续生成剩余记录。

缓存验收：

```bash
python - <<'PY'
import json
from pathlib import Path

for split in ("train", "dev"):
    path = Path(f"knowledge/siglip2_region_cache/roberta128_r36/{split}/manifest.json")
    data = json.load(open(path, encoding="utf-8"))
    print(split, {
        "records": data["record_count"],
        "feature_size": data["feature_size"],
        "fallback_images": data["diagnostics"]["fallback_images"],
        "invalid_local": data["diagnostics"]["invalid_local_crops"],
        "invalid_context": data["diagnostics"]["invalid_context_crops"],
    })
PY
```

## 7. 三组消融

切回原训练环境：

```bash
cd ~/gmner
conda activate gmner

for mode in vinvl siglip2 fusion; do
  PYTHONPATH=. python scripts/train_siglip2_region_reliability.py \
    --config configs/fmnerg_twitter10000_siglip2_reliability_${mode}.yaml
done
```

三组实验使用相同 hard A/B 标签、损失和 dev 评价，仅改变输入特征：

| mode | 输入 |
| --- | --- |
| `vinvl_only` | 原 Fine/VinVL/Base/Coarse 特征 |
| `siglip2_only` | 19 维显式多尺度 SigLIP 2 特征 |
| `fusion` | 两组特征联合 |

每组保存三个用途不同的 checkpoint：

```text
best_ab_model.pt          hard A/B AUROC/AUPRC 最优
best_risk_model.pt        NULL 保持约束下风险净纠错最优
best_calibrated_model.pt  hard A/B Brier/ECE 最优
```

这些 checkpoint 都是 dev 研究模型，不接入正式推理。

## 8. Dev 评估

```bash
for mode in vinvl siglip2 fusion; do
  PYTHONPATH=. python scripts/evaluate_siglip2_region_reliability.py \
    --config configs/fmnerg_twitter10000_siglip2_reliability_${mode}.yaml \
    --checkpoint outputs/fmnerg_roberta128_siglip2_reliability_${mode}/best_ab_model.pt \
    --output outputs/fmnerg_roberta128_siglip2_reliability_${mode}/dev_ab.json
done
```

主报告至少比较：

```text
hard_ab_auc
hard_ab_auprc
hard_ab_best_balanced_accuracy
hard_ab_brier
hard_ab_ece
a_accept_rate
hard_b_reject_rate
null_high_reliability_false_positive_rate
fine_correct_top1_accept_rate
fine_wrong_top1_reject_rate
promoted_a_accept_rate
risk_best_net_correction
risk_best_null_preservation_rate
risk_best_promoted_fix_count
```

## 9. Go/no-go

只有 Fusion 同时满足以下条件，才进入 M3.4B：

```text
hard A/B AUROC                  >= 0.70
best balanced accuracy          >= 0.62
NULL preservation               >= 0.98
risk net correction             >= 15
promoted A fix count             > 0
```

同时必须证明 Fusion 明显优于 `vinvl_only`，否则不能把收益归因于 SigLIP 2。

M3.4A 不提供 `--split test`。若未达到门槛，则保留当前 test GMNER `0.61529`，停止
该分支；若达到门槛，先生成 OOF train 特征和多随机种子结果，再实现 M3.4B，将冻结
Reliability 证据以 `detach()` 形式输入 Evidence Visibility。

## 10. 2026-07-23 正式结论

M3.4A 已完成 train/dev 缓存、三组同口径训练和 dev 评估：

| mode | hard A/B AUROC | best balanced accuracy | risk net |
| --- | ---: | ---: | ---: |
| VinVL-only | 0.5773 | 0.5845 | 0 |
| SigLIP2-only | 0.5759 | 0.5759 | +1 |
| Fusion | **0.6003** | **0.6241** | **+9** |

Fusion 的 NULL preservation 为 `0.9932`，promoted fix 为 `2`。它证明 SigLIP 2
与 VinVL/Fine 存在有限互补性，但没有达到 `AUROC >= 0.70` 和 `risk net >= +15`
两个核心门槛。因此：

```text
M3.4A = no-go
M3.4B = 不启动
test   = 不读取
正式链路 = 保持 GMNER F1 0.61529
```

本轮结果不支持继续扫描 prompt、crop expansion、MLP 宽度或阈值，也不支持端到端
微调 SigLIP 2。多尺度 crop 主要提供少量高精度尾部证据，没有形成稳定的区域绝对
可靠性排序。

### 与旧版 0.6393 的口径差异

两轮使用相同 R16/R36 cache、Fine checkpoint、优化器主要参数和随机种子，但冻结
决策链不同：

```text
旧版 0.6393：Hierarchical -> Fine，直接以 Fine 前的 visibility 构造 A/B
本轮 0.5773：Hierarchical -> Fine -> Evidence Visibility，以最终 KEEP 构造 A/B
```

由此 hard A/B 数量也发生变化：

```text
旧版：A=106，candidate-covered B=64
本轮：A=105，candidate-covered B=73
```

旧版 checkpoint 按 hard A/B AUROC 保存；本轮也以 `best_ab_model.pt` 报告 AUROC，
但样本定义不同，所以只能作为两个独立实验设置归档。

## 11. Dev 错误切片

在决定是否尝试 DINOv2 前，只对 Fusion `best_ab_model.pt` 做一次冻结 dev 诊断：

```bash
cd ~/gmner

PYTHONPATH=. python scripts/analyze_siglip2_reliability_slices.py \
  --config configs/fmnerg_twitter10000_siglip2_reliability_fusion.yaml \
  --checkpoint outputs/fmnerg_roberta128_siglip2_reliability_fusion/best_ab_model.pt \
  --output outputs/fmnerg_roberta128_siglip2_reliability_fusion/dev_slices.json \
  --device cuda
```

脚本复用本轮 hard A/B 和风险阈值，报告：

```text
entity type
single/multi/no-person scene
Fine top1 与 gold 候选的 VinVL object 类别关系
small/medium/large box
Fine-SigLIP2 top1 agreement
four-way top1 agreement
promoted/original candidate
1.5x context box crowding
risk-tail FIX/DAMAGE/NEUTRAL 来源及样例
```

对象类别从原 VinVL NPZ 读取，并通过候选框 IoU 对齐。`context_overlap=high` 定义为
Fine top1 的 1.5 倍正方形上下文框与任一其他真实候选框的最大 IoU 不低于 `0.3`。
该脚本固定 `split=dev`，没有 test 入口。

## 12. Dev 切片结论

Fusion `best_ab_model.pt` 的切片诊断严格复现全局结果：

```text
hard A/B = 105 / 73
AUROC    = 0.60026
risk     = 16 FIX / 7 DAMAGE / 9 NEUTRAL = +9
VinVL object 与 R36 box 对齐率 = 1.0
```

主要切片如下：

| slice | samples (A/B) | AUROC |
| --- | ---: | ---: |
| multi-person | 114 (70/44) | **0.5234** |
| single-person | 27 (16/11) | 0.7216 |
| no-person | 37 (19/18) | 0.7368 |
| same VinVL object class | 122 (105/17) | **0.4874** |
| PER | 76 (49/27) | 0.5639 |
| ORG | 45 (26/19) | 0.6478 |
| medium box | 41 (20/21) | **0.4952** |
| small box | 51 (30/21) | 0.6444 |
| large box | 86 (55/31) | 0.6264 |

SigLIP 2 与 Fine top1 只在 `9/178` 个 hard A/B 样本上一致；该极小切片可分，但不能
形成广泛收益。Promoted gold 切片 AUROC 为 `0.50`，说明新增候选覆盖成功不等于
绝对可靠性可判。

风险尾部的 `+9` 主要来自 PER `+6` 和 multi-person `+8`。但所有 `7` 个 DAMAGE 都
出现在高 context overlap 中；低 overlap 尾部为 `7 FIX / 0 DAMAGE / 1 NEUTRAL`。
这说明 Fusion 找到的是少量局部高精度规则，并没有解决拥挤场景中的同类实例区分。

### 对 DINOv2 的约束

切片证据满足“错误集中在同类实例和局部外观”的必要条件，因此允许进入一个冻结的
M3.5A 旁路诊断，但不能据此预设 DINOv2 有效。DINOv2 只应提供：

```text
local/context patch 外观
候选间视觉相似度与唯一性
多人/拥挤和背景污染证据
```

它不能从实体字符串识别人名或组织身份，因此不能替代跨模态 identity evidence。
M3.5A 仍只允许 train/dev、保持正式链冻结，并沿用 `AUROC >= 0.70`、
`risk net >= +15`、`NULL preservation >= 0.98` 的门槛。若主要增益仍只来自低重叠
风险尾部，则停止继续增加通用视觉编码器，转向 OCR、logo/object 字符串匹配或显式
“无足够身份视觉证据”建模。
