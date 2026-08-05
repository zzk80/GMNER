# 当前保留链路运行与验收手册

**适用工作区**：`/home/zzk/gmner`
**Python**：`/home/zzk/miniconda3/envs/gmner/bin/python`
**更新日期**：2026-08-03

本文只覆盖清理后仍保留且可以执行的链路：

1. 正式 Model-G：M3.3A GMNER 全链；
2. 正式 Model-F：F3 FMNERG subtype 链；
3. DVH Frozen-CLIP 独立 Stage1 对照；
4. TQ-DV-MNER 独立 Stage1 对照及 fixed-span type replay。

关闭的 OOF、S3/S4、P4、TP/J3 等实验不在本文中恢复。除非另有明确授权，
日常运行只使用 Train/Dev，不重新读取 Test，也不覆盖正式 checkpoint。

## 1. 统一环境与资源检查

登录云端后执行：

```bash
cd ~/gmner

export PYTHONPATH=.
export PY=/home/zzk/miniconda3/envs/gmner/bin/python
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

$PY -V
nvidia-smi
df -h .
git status --short --branch
```

检查核心数据、模型和冻结产物：

```bash
cd ~/gmner

for path in \
  roberta-base/config.json \
  clip-vit-base-patch32/config.json \
  GMNER-main/Twitter10000_v2.0/txt_fine/train.txt \
  GMNER-main/Twitter10000_v2.0/txt_fine/dev.txt \
  GMNER-main/Twitter10000_v2.0/VinVL \
  GMNER-main/Twitter10000_v2.0/xml \
  outputs/fmnerg_stage1_roberta128/best_model.pt \
  outputs/fmnerg_roberta128_hierarchical_record_verifier/best_model.pt \
  outputs/fmnerg_roberta128_coarse_selector/best_model.pt \
  outputs/fmnerg_roberta128_fine_grounding_adapter/best_model.pt \
  outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  knowledge/dvh_frozen_clip/train/manifest.json \
  knowledge/dvh_frozen_clip/dev/manifest.json; do
  test -e "$path" && echo "OK      $path" || echo "MISSING $path"
done
```

需要等待至少 10 GiB 空闲显存和 5 GiB 空闲磁盘时，可复用：

```bash
wait_for_resources() {
  local min_gpu_mib="${1:-10240}"
  local min_disk_bytes="${2:-5368709120}"
  while true; do
    local gpu_free disk_free
    gpu_free=$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits | head -1 | tr -d ' ')
    disk_free=$(df -B1 --output=avail "$HOME/gmner" | tail -1 | tr -d ' ')
    if [ "$gpu_free" -ge "$min_gpu_mib" ] && \
       [ "$disk_free" -ge "$min_disk_bytes" ]; then
      echo "[$(date '+%F %T')] Gates passed: GPU=${gpu_free}MiB disk=${disk_free}B"
      break
    fi
    echo "[$(date '+%F %T')] Waiting: GPU=${gpu_free}MiB disk=${disk_free}B"
    sleep 300
  done
}
```

## 2. Model-G：M3.3A 正式 GMNER 全链

### 2.1 链路和训练口径

```text
RoBERTa Stage1
-> R16 formal candidates
-> R36 expanded candidates
-> Hierarchical Record Verifier
-> Coarse Region Selector
-> Fine Grounding Adapter
-> Evidence Visibility
-> record-level decode
```

正式 Dev/Test 模型采用完整 Train 正常训练，不采用 OOF。历史 full-chain OOF 只用于
训练分布和后处理研究，不是 `0.621316` Dev 或 `0.615294` Test 的训练口径。

### 2.2 直接复用冻结模型评估 Dev

Stage1 Dev：

```bash
cd ~/gmner
export PYTHONPATH=.
PY=/home/zzk/miniconda3/envs/gmner/bin/python

$PY -u scripts/evaluate.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
  --split dev \
  --output-dir outputs/fmnerg_stage1_roberta128/dev_recheck
```

完整 M3.3A Dev：

```bash
$PY -u scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --split dev \
  --output outputs/fmnerg_roberta128_evidence_visibility/dev_recheck.json \
  --device cuda
```

### 2.3 分阶段查看当前冻结结果

```bash
$PY -u scripts/evaluate_hierarchical_record_verifier.py \
  --config configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml \
  --checkpoint outputs/fmnerg_roberta128_hierarchical_record_verifier/best_model.pt \
  --split dev \
  --output outputs/fmnerg_roberta128_hierarchical_record_verifier/dev_recheck.json

$PY -u scripts/evaluate_coarse_region_selector.py \
  --config configs/fmnerg_twitter10000_coarse_selector.yaml \
  --checkpoint outputs/fmnerg_roberta128_coarse_selector/best_model.pt \
  --output outputs/fmnerg_roberta128_coarse_selector/dev_recheck.json \
  --device cuda

$PY -u scripts/evaluate_fine_grounding_adapter.py \
  --config configs/fmnerg_twitter10000_fine_grounding_adapter.yaml \
  --checkpoint outputs/fmnerg_roberta128_fine_grounding_adapter/best_model.pt \
  --output outputs/fmnerg_roberta128_fine_grounding_adapter/dev_recheck.json \
  --device cuda
```

### 2.4 独立目录完整重训

以下流程不会覆盖正式目录。先定义运行编号：

```bash
cd ~/gmner
export PYTHONPATH=.
PY=/home/zzk/miniconda3/envs/gmner/bin/python

RUN_ID="m33a_repro_$(date +%Y%m%d_%H%M%S)"
RUN_OUT="outputs/reproductions/${RUN_ID}"
RUN_CACHE="knowledge/reproductions/${RUN_ID}"
RUN_CFG="${RUN_OUT}/configs"
mkdir -p "$RUN_OUT" "$RUN_CACHE" "$RUN_CFG"
```

生成只指向本次运行目录的配置副本：

```bash
$PY - "$RUN_OUT" "$RUN_CACHE" "$RUN_CFG" <<'PY'
import copy
import sys
from pathlib import Path
import yaml

out, cache, cfg_dir = map(Path, sys.argv[1:])
cfg_dir.mkdir(parents=True, exist_ok=True)

def load(name):
    return yaml.safe_load(Path(name).read_text(encoding="utf-8"))

def save(name, payload):
    path = cfg_dir / name
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return str(path)

h = load("configs/fmnerg_twitter10000_hierarchical_record_verifier.yaml")
h["data"].update({
    "train_cache": str(cache / "r16_train.pt"),
    "dev_cache": str(cache / "r16_dev.pt"),
    "test_cache": str(cache / "r16_test.pt"),
})
h["runtime"]["output_dir"] = str(out / "hierarchical")
h_path = save("hierarchical.yaml", h)

c = load("configs/fmnerg_twitter10000_coarse_selector.yaml")
c["data"].update({
    "train_cache": str(cache / "r36_train.pt"),
    "dev_cache": str(cache / "r36_dev.pt"),
})
c["runtime"]["output_dir"] = str(out / "coarse")
c_path = save("coarse.yaml", c)

f = load("configs/fmnerg_twitter10000_fine_grounding_adapter.yaml")
f["data"].update({
    "formal_train_cache": str(cache / "r16_train.pt"),
    "expanded_train_cache": str(cache / "r36_train.pt"),
    "formal_dev_cache": str(cache / "r16_dev.pt"),
    "expanded_dev_cache": str(cache / "r36_dev.pt"),
})
f["frozen"].update({
    "hierarchical_config": h_path,
    "hierarchical_checkpoint": str(out / "hierarchical/best_model.pt"),
    "coarse_checkpoint": str(out / "coarse/best_model.pt"),
})
f["runtime"]["output_dir"] = str(out / "fine")
f_path = save("fine.yaml", f)

e = load("configs/fmnerg_twitter10000_evidence_visibility.yaml")
e["data"].update(copy.deepcopy(f["data"]))
e["frozen"].update({
    "fine_config": f_path,
    "fine_checkpoint": str(out / "fine/best_model.pt"),
})
e["runtime"]["output_dir"] = str(out / "evidence")
save("evidence.yaml", e)
PY
```

训练 Stage1，并生成本次运行自己的 R16/R36 Train/Dev cache：

```bash
$PY -u scripts/train.py \
  --config configs/fmnerg_twitter10000_stage1.yaml \
  --output-dir "$RUN_OUT/stage1" \
  --skip-test-evaluation

for split in train dev; do
  $PY -u scripts/build_record_candidate_cache.py \
    --config configs/fmnerg_twitter10000_stage1.yaml \
    --checkpoint "$RUN_OUT/stage1/best_model.pt" \
    --split "$split" \
    --output "$RUN_CACHE/r16_${split}.pt" \
    --k-best 6 \
    --max-span-candidates 12 \
    --top-m-types 3 \
    --boundary-shift 1 \
    --boundary-penalty 0.25 \
    --max-span-length 10 \
    --device cuda

  $PY -u scripts/build_record_candidate_cache.py \
    --config configs/fmnerg_twitter10000_stage1.yaml \
    --checkpoint "$RUN_OUT/stage1/best_model.pt" \
    --split "$split" \
    --output "$RUN_CACHE/r36_${split}.pt" \
    --max-regions 36 \
    --formal-anchor-cache "$RUN_CACHE/r16_${split}.pt" \
    --k-best 6 \
    --max-span-candidates 12 \
    --top-m-types 3 \
    --boundary-shift 1 \
    --boundary-penalty 0.25 \
    --max-span-length 10 \
    --device cuda
done
```

按正式顺序训练后半链并评估 Dev：

```bash
$PY -u scripts/train_hierarchical_record_verifier.py \
  --config "$RUN_CFG/hierarchical.yaml"

$PY -u scripts/train_coarse_region_selector.py \
  --config "$RUN_CFG/coarse.yaml"

$PY -u scripts/train_fine_grounding_adapter.py \
  --config "$RUN_CFG/fine.yaml"

$PY -u scripts/train_evidence_visibility.py \
  --config "$RUN_CFG/evidence.yaml"

$PY -u scripts/evaluate_evidence_visibility.py \
  --config "$RUN_CFG/evidence.yaml" \
  --checkpoint "$RUN_OUT/evidence/best_model.pt" \
  --split dev \
  --output "$RUN_OUT/evidence/dev_final.json" \
  --device cuda
```

### 2.5 长任务会话

M3.3A 重训由多个有依赖关系的阶段组成，建议在 `tmux` 中逐段执行 2.4，而不是把
所有命令压成一个不可审计的后台字符串：

```bash
tmux new -s m33a_reproduction
# 在 tmux 会话中依次执行 2.4；按 Ctrl-b d 脱离。

# 重新连接
tmux attach -t m33a_reproduction

# 查看会话
tmux ls
```

每个训练阶段自身都会把日志写入对应 `runtime.output_dir/train.log`。

### 2.6 Test 边界

日常只查看已有正式结果：

```bash
cat outputs/fmnerg_roberta128_evidence_visibility/test_metrics.json
```

只有架构、checkpoint、阈值和协议再次冻结后，才允许显式构建 Test cache并评估：

```bash
# LOCKED: 不作为日常命令执行。
for budget in 16 36; do
  if [ "$budget" -eq 16 ]; then
    output=knowledge/record_candidates/roberta128/fmnerg_test_hierarchical.pt
    extra=()
  else
    output=knowledge/record_candidates/roberta128/fmnerg_test_hierarchical_r36.pt
    extra=(--max-regions 36 --formal-anchor-cache \
      knowledge/record_candidates/roberta128/fmnerg_test_hierarchical.pt)
  fi
  $PY -u scripts/build_record_candidate_cache.py \
    --config configs/fmnerg_twitter10000_stage1.yaml \
    --checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
    --split test \
    --output "$output" \
    --allow-test \
    --device cuda \
    "${extra[@]}"
done

$PY -u scripts/evaluate_evidence_visibility.py \
  --config configs/fmnerg_twitter10000_evidence_visibility.yaml \
  --checkpoint outputs/fmnerg_roberta128_evidence_visibility/best_model.pt \
  --split test \
  --formal-cache knowledge/record_candidates/roberta128/fmnerg_test_hierarchical.pt \
  --expanded-cache knowledge/record_candidates/roberta128/fmnerg_test_hierarchical_r36.pt \
  --output outputs/fmnerg_roberta128_evidence_visibility/test_recheck.json \
  --device cuda
```

## 3. Model-F：F3 正式 FMNERG 链

### 3.1 链路和边界

```text
冻结 M3.3A formal entity
-> [first; last; mean] RoBERTa span states
-> parent-masked 51-class subtype encoder
-> Fine MNER / FMNERG
```

F3 只能增加 subtype；不能修改 span、coarse type、region/NULL、EEG 或 GMNER。

### 3.2 直接评估冻结 Dev checkpoint

```bash
cd ~/gmner
export PYTHONPATH=.
PY=/home/zzk/miniconda3/envs/gmner/bin/python

for seed in 41 42 43; do
  $PY -u tools/evaluate_fmnerg_subtype_encoder.py \
    --config sidecars/fmnerg_subtype/configs/f3_p1_lr6_lower_double.yaml \
    --checkpoint \
      "outputs/fmnerg_subtype_f3_p1/lr6_lower_double/seed${seed}/best_model.pt" \
    --output \
      "outputs/fmnerg_subtype_f3_p1/lr6_lower_double/seed${seed}/dev_recheck.json" \
    --device cuda
done
```

### 3.3 单 seed 或完整三 seed Dev 训练

独立单 seed，不覆盖正式目录：

```bash
$PY -u tools/train_fmnerg_subtype_encoder.py \
  --config sidecars/fmnerg_subtype/configs/f3_p1_lr6_lower_double.yaml \
  --seed 42 \
  --device cuda \
  --output-dir outputs/reproductions/f3_seed42
```

按预注册协议执行完整 Dev study：

```bash
nohup env \
  PYTHONPATH=. \
  PYTHON_BIN=/home/zzk/miniconda3/envs/gmner/bin/python \
  DEVICE=cuda \
  OUTPUT_ROOT=outputs/reproductions/f3_p1 \
  LOCK_DIR=knowledge/fmnerg_subtype_sidecar/.f3_reproduction.lock \
  bash tools/run_fmnerg_subtype_f3_p1.sh \
  > f3_p1_reproduction.log 2>&1 &
echo $! > f3_p1_reproduction.pid
```

### 3.4 Test 边界

正式 Test 已一次性完成。日常读取归档，不重跑：

```bash
cat sidecars/fmnerg_subtype/f3_final_test_result.json
cat outputs/fmnerg_subtype_f3_final_test/final_test_summary.json
```

历史一次性入口是：

```bash
# LOCKED: 不重复执行。
PYTHONPATH=. $PY tools/run_fmnerg_subtype_encoder_final_test.py \
  --protocol sidecars/fmnerg_subtype/f3_final_test.yaml \
  --device cuda
```

## 4. DVH Frozen-CLIP 独立 Stage1

DVH 是独立训练的研究对照，不使用旧 Stage1 checkpoint，CLIP 完全冻结；checkpoint
只按 Dev GMNER 选择，配置不包含 Test 文件。

### 4.1 构建 Frozen CLIP cache

已有 cache 时先验证，通常不需要重建：

```bash
for split in train dev; do
  test -s "knowledge/dvh_frozen_clip/${split}/manifest.json" \
    && echo "OK ${split}" || echo "MISSING ${split}"
done
```

确需重建时：

```bash
for split in train dev; do
  $PY -u scripts/build_dvh_frozen_clip_cache.py \
    --source-file "GMNER-main/Twitter10000_v2.0/txt_fine/${split}.txt" \
    --image-dir GMNER-main/Twitter10000_v2.0/images \
    --model-name /home/zzk/gmner/clip-vit-base-patch32 \
    --output-dir "knowledge/dvh_frozen_clip/${split}" \
    --split "$split" \
    --batch-size 64 \
    --shard-size 256 \
    --fp16 \
    --device cuda
done
```

### 4.2 训练

```bash
nohup bash -lc '
  set -euo pipefail
  cd ~/gmner
  export PYTHONPATH=.
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  /home/zzk/miniconda3/envs/gmner/bin/python -u \
    scripts/train_dvh_stage1.py \
    --config configs/dvh_stage1/frozen_clip_vit_b32_seed42.yaml \
    --seed 42 \
    --device cuda \
    --output-dir outputs/reproductions/dvh_seed42
' > dvh_stage1_seed42.log 2>&1 &
echo $! > dvh_stage1_seed42.pid
```

产物：

```text
outputs/reproductions/dvh_seed42/best_model.pt
outputs/reproductions/dvh_seed42/train_summary.json
outputs/reproductions/dvh_seed42/resolved_config.yaml
```

DVH 没有独立 evaluate 脚本；训练过程每个 epoch 都在 Dev 上评估，最佳结果和完整
history 位于 `train_summary.json`。

## 5. TQ-DV-MNER 与固定 span replay

TQ-DV 同样独立训练、冻结 CLIP、禁止 Test；checkpoint 只按 Dev MNER 选择。它复用
`knowledge/dvh_frozen_clip/`，不重复构建图像 patch cache。

### 5.1 训练

```bash
nohup bash -lc '
  set -euo pipefail
  cd ~/gmner
  export PYTHONPATH=.
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  /home/zzk/miniconda3/envs/gmner/bin/python -u \
    scripts/train_tq_dv_mner.py \
    --config configs/tq_dv_mner/type_query_dual_visual_seed42.yaml \
    --seed 42 \
    --device cuda \
    --output-dir outputs/reproductions/tq_dv_seed42
' > tq_dv_mner_seed42.log 2>&1 &
echo $! > tq_dv_mner_seed42.pid
```

产物：

```text
outputs/reproductions/tq_dv_seed42/best_model.pt
outputs/reproductions/tq_dv_seed42/train_summary.json
outputs/reproductions/tq_dv_seed42/resolved_config.yaml
```

### 5.2 Fixed-span type replay

该诊断冻结正式 Stage1 的 span 和 prediction count，只用 TQ-DV 为现有 span 重算四类
type 分数。它是 Dev-only，不训练、不扫描阈值、不访问 Test。

前台运行：

```bash
$PY -u scripts/evaluate_tq_fixed_span_type_replay.py \
  --formal-config configs/fmnerg_twitter10000_stage1.yaml \
  --formal-checkpoint outputs/fmnerg_stage1_roberta128/best_model.pt \
  --tq-config configs/tq_dv_mner/type_query_dual_visual_seed42.yaml \
  --tq-checkpoint outputs/tq_dv_mner/type_query_dual_visual_seed42/best_model.pt \
  --output \
    outputs/tq_dv_mner/type_query_dual_visual_seed42/dev_fixed_span_type_replay.json \
  --batch-size 4 \
  --device cuda
```

带 10 GiB GPU / 5 GiB磁盘门槛的后台运行：

```bash
nohup bash -lc '
  set -euo pipefail
  cd ~/gmner
  export PYTHONPATH=.
  export TRANSFORMERS_OFFLINE=1
  export HF_HUB_OFFLINE=1
  while true; do
    gpu=$(nvidia-smi --query-gpu=memory.free \
      --format=csv,noheader,nounits | head -1 | tr -d " ")
    disk=$(df -B1 --output=avail "$HOME/gmner" | tail -1 | tr -d " ")
    [ "$gpu" -ge 10240 ] && [ "$disk" -ge 5368709120 ] && break
    echo "[$(date "+%F %T")] Waiting: GPU=${gpu}MiB disk=${disk}B"
    sleep 300
  done
  exec /home/zzk/miniconda3/envs/gmner/bin/python -u \
    scripts/evaluate_tq_fixed_span_type_replay.py \
    --batch-size 4 \
    --device cuda
' > tq_fixed_span_type_replay.log 2>&1 &
echo $! > tq_fixed_span_type_replay.pid
```

## 6. 运行状态查看

### 6.1 通用状态

```bash
nvidia-smi
df -h ~/gmner
free -h
pgrep -af 'train.py|train_.*\.py|evaluate_.*\.py'
```

根据 PID 文件检查：

```bash
PID_FILE=tq_fixed_span_type_replay.pid
if [ -s "$PID_FILE" ]; then
  pid=$(cat "$PID_FILE")
  ps -o pid,ppid,stat,etime,%cpu,%mem,cmd -p "$pid"
else
  echo "No PID file: $PID_FILE"
fi
```

实时日志：

```bash
tail -n 100 -f m33a_reproduction.log
tail -n 100 -f f3_p1_reproduction.log
tail -n 100 -f dvh_stage1_seed42.log
tail -n 100 -f tq_dv_mner_seed42.log
tail -n 100 -f tq_fixed_span_type_replay.log
```

一次性查看最近错误：

```bash
for log in *.log; do
  [ -f "$log" ] || continue
  echo "===== $log ====="
  grep -E 'Traceback|RuntimeError|ValueError|SIGSEGV|CUDA out of memory|Killed' \
    "$log" | tail -20 || true
done
```

### 6.2 判断进程是否真的结束

`tail -f` 没有新输出不代表进程失败。联合检查：

```bash
pid=$(cat YOUR_RUN.pid)
if kill -0 "$pid" 2>/dev/null; then
  echo "RUNNING pid=$pid"
else
  echo "EXITED pid=$pid"
fi

tail -50 YOUR_RUN.log
```

## 7. 结果查看命令

### 7.1 M3.3A Dev/Test

```bash
$PY - <<'PY'
import json
from pathlib import Path

paths = [
    Path("outputs/fmnerg_stage1_roberta128/dev_recheck/dev_metrics_from_checkpoint.json"),
    Path("outputs/fmnerg_roberta128_hierarchical_record_verifier/dev_metrics.json"),
    Path("outputs/fmnerg_roberta128_fine_grounding_adapter/dev_metrics.json"),
    Path("outputs/fmnerg_roberta128_evidence_visibility/dev_metrics.json"),
    Path("outputs/fmnerg_roberta128_evidence_visibility/test_metrics.json"),
]

for path in paths:
    if not path.exists():
        print("MISSING", path)
        continue
    payload = json.load(path.open(encoding="utf-8"))
    metrics = payload.get("metrics", payload)
    print(path)
    print({
        "span_f1": metrics.get("span_f1"),
        "mner_f1": metrics.get("mner_f1", metrics.get("entity_f1")),
        "eeg_f1": metrics.get("eeg_f1"),
        "gmner_f1": metrics.get("gmner_f1", metrics.get("gmner_score", metrics.get("triple_f1"))),
        "correct": metrics.get("triple_correct"),
        "predicted": metrics.get("triple_predict", metrics.get("prediction_count")),
        "gold": metrics.get("triple_gold", metrics.get("gold_count")),
    })
PY
```

### 7.2 F3 三 seed Dev 与正式 Test

```bash
$PY - <<'PY'
import json

dev = json.load(open(
    "docs/experiments/fmnerg_subtype_f3_p1_dev_summary.json",
    encoding="utf-8",
))
test = json.load(open(
    "sidecars/fmnerg_subtype/f3_final_test_result.json",
    encoding="utf-8",
))

print("Dev runs:")
for run in dev["runs"]:
    print(run["seed"], run["fine_mner_f1"], run["fmnerg_f1"])
dev_fine = [float(x["fine_mner_f1"]) for x in dev["runs"]]
dev_fmnerg = [float(x["fmnerg_f1"]) for x in dev["runs"]]
print("Dev mean:", {
    "fine_mner_f1": sum(dev_fine) / len(dev_fine),
    "fmnerg_f1": sum(dev_fmnerg) / len(dev_fmnerg),
    "passed": dev["passed"],
})
print("Test aggregate:", test["aggregate"])
PY
```

### 7.3 DVH 与 TQ-DV 最佳 epoch

```bash
$PY - <<'PY'
import json

paths = {
    "DVH": "outputs/dvh_stage1/frozen_clip_vit_b32_seed42/train_summary.json",
    "TQ-DV": "outputs/tq_dv_mner/type_query_dual_visual_seed42/train_summary.json",
}
for name, path in paths.items():
    payload = json.load(open(path, encoding="utf-8"))
    epoch = int(payload["best_epoch"])
    entry = next(x for x in payload["history"] if int(x["epoch"]) == epoch)
    metrics = entry["metrics"]
    print(name, {
        "best_epoch": epoch,
        "span_f1": metrics.get("span_f1"),
        "mner_f1": metrics.get("mner_f1"),
        "eeg_f1": metrics.get("eeg_f1"),
        "gmner_f1": metrics.get("gmner_f1"),
        "prediction_count": metrics.get("prediction_count"),
        "test_accessed": payload.get("test_accessed"),
    })
PY
```

### 7.4 Fixed-span replay

```bash
$PY - <<'PY'
import json
from pathlib import Path

path = Path(
    "outputs/tq_dv_mner/type_query_dual_visual_seed42/"
    "dev_fixed_span_type_replay.json"
)
if not path.exists():
    raise SystemExit(f"Result not generated yet: {path}")
d = json.load(path.open(encoding="utf-8"))
print({
    "baseline_mner": d["baseline"]["mner_f1"],
    "replay_mner": d["replay"]["mner_f1"],
    "delta": d["delta"],
    "actions": d["actions"],
    "checks": d["checks"],
    "test_accessed": d["test_accessed"],
})
PY
```

## 8. 当前冻结参考结果

| 链路 | Split | Span F1 | MNER/Fine MNER F1 | EEG F1 | GMNER/FMNERG F1 | 状态 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| M3.3A Stage1 | Dev | 0.870721 | 0.814740 | 0.645993 | 0.607330 | Stage1 baseline |
| M3.3A full | Dev | 0.872830 | 0.816714 | 0.660880 | 0.621316 | 正式最优 Model-G |
| M3.3A full | Test | 0.869804 | 0.818431 | 0.652157 | 0.615294 | 冻结正式结果 |
| F3 | Dev | 固定 Model-G | 0.68039 +/- 0.00297 | 固定 | 0.52052 +/- 0.00219 | 正式 Model-F |
| F3 | Test | 固定 Model-G | 0.66510 +/- 0.00160 | 固定 | 0.50431 +/- 0.00111 | 冻结正式结果 |
| DVH best | Dev | 0.851785 | 0.799355 | 0.653761 | 0.615043 | NO_GO 对照 |
| TQ-DV best | Dev | 0.852346 | 0.810275 | 不适用 | 不适用 | NO_GO 对照 |
| TQ-DV fixed-span replay | Dev | 0.870721 | 0.817559 | 不适用 | 不适用 | +7，POSITIVE_DIAGNOSTIC / NO_GO |

说明：

- DVH 表中是按 Dev GMNER 选出的 epoch 18，而不是 epoch 20 诊断项；
- TQ-DV 不训练正式完整 Grounding，不能报告可比的 EEG/GMNER；
- fixed-span replay 保留正式 span 和预测数量，MNER 从 `0.814740` 提升到 `0.817559`，
  但净增仅 7 个正确 typed spans，不进入下游重建；
- DVH/TQ-DV 均未访问 Test。

### 8.1 B1/A1 动作可分性准备

最终 M3.3A Dev 只读审计已固定下一阶段的两个窄假设：

```text
B1：最终链 exact-span coarse-type correction
A1：保持预测数不变的一换一 boundary replacement
```

当前只允许生成 Dev 人工语义审阅队列：

```bash
cd ~/gmner
PYTHONPATH=. $PY scripts/prepare_m33a_action_review_queues.py
```

固定输出为：

```text
outputs/final_m33a_action_review/type_semantic_union_111.csv
outputs/final_m33a_action_review/boundary_replacement_positive_55.csv
outputs/final_m33a_action_review/boundary_promotion_positive_61.csv
outputs/final_m33a_action_review/review_manifest.json
```

其中 111 条类型审阅并集由 `21 text+visual + 4 visual-only + 86
text-only` 构成；另有 28 条 neither，不进入首轮人工队列。这些 Dev 行只用于解释
Oracle 正动作，不得用于训练、校准、特征选择或阈值选择。B1/A1 正式训练必须先取得
完整最终链 OOF 的全部正负动作总体；当前不生成 OOF、不训练、不访问 Test。正式约束见
`docs/experiments/B1_A1_ACTION_SEPARABILITY_PROTOCOL.md`。

## 9. 产物目录速查

```text
正式 Model-G
  configs/fmnerg_twitter10000_*.yaml
  outputs/fmnerg_stage1_roberta128/
  outputs/fmnerg_roberta128_hierarchical_record_verifier/
  outputs/fmnerg_roberta128_coarse_selector/
  outputs/fmnerg_roberta128_fine_grounding_adapter/
  outputs/fmnerg_roberta128_evidence_visibility/
  knowledge/record_candidates/roberta128/

正式 Model-F
  sidecars/fmnerg_subtype/
  outputs/fmnerg_subtype_f3_p1/
  outputs/fmnerg_subtype_f3_final_test/
  knowledge/fmnerg_subtype_sidecar/roberta128/

DVH
  configs/dvh_stage1/
  outputs/dvh_stage1/frozen_clip_vit_b32_seed42/
  knowledge/dvh_frozen_clip/

TQ-DV
  configs/tq_dv_mner/
  outputs/tq_dv_mner/type_query_dual_visual_seed42/

本地预训练模型
  roberta-base/
  clip-vit-base-patch32/
```

## 10. 最低验证与同步检查

代码静态检查：

```bash
cd ~/gmner
PYTHONPATH=. $PY -m compileall -q gmner scripts tools sidecars
```

定向测试（云端环境有 pytest 时）：

```bash
PYTHONPATH=. $PY -m pytest -q \
  tests/test_hierarchical_record_verifier.py \
  tests/test_dvh_stage1.py \
  tests/test_tq_dv_mner.py
```

本地与云端核对本文档：

```powershell
# Windows 本地 PowerShell
$local = "docs/experiments/CURRENT_CHAIN_RUNBOOK.md"
$localHash = (Get-FileHash $local -Algorithm SHA256).Hash.ToLower()
$remoteHash = ssh server4090 \
  "sha256sum ~/gmner/docs/experiments/CURRENT_CHAIN_RUNBOOK.md | cut -d' ' -f1"
"local=$localHash"
"remote=$remoteHash"
```

若 Windows CRLF 与云端 LF 导致字节 hash 不同，使用 Git blob 风格的 LF 归一化
比较，或上传后统一执行：

```bash
sed -i 's/\r$//' ~/gmner/docs/experiments/CURRENT_CHAIN_RUNBOOK.md
```
