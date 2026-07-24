# Knowledge and Cached Assets

该目录存放正式链、FMNERG sidecar 和严格 OOF 的可重建数据资产。大型二进制缓存
默认由 `.gitignore` 排除，不应提交到 Git。

```text
record_candidates/
  roberta128/                 M3.3A R16/R36、Coarse、Fine、Evidence 缓存

grounding/                    train-only grounding 统计与先验

fmnerg_subtype_sidecar/       51 类 subtype sidecar 资产

null_release_oof/
  roberta128/fold0..fold9/    heldout 特征、proof、manifest 和归档摘要
  roberta128/full_chain_train_oof.pt
                              可选的十折合并缓存

siglip2_region_cache/         已归档 M3.4/OOF 的可重建冻结特征
```

正式 M3.3A 推理不依赖 `null_release_oof/` 或 `siglip2_region_cache/`。OOF
目录仅用于无泄漏训练特征、泛化审计和已归档的 NULL Release 实验。

数据约束：

- 正式 span/type 以 R16 为准，R36 只能扩展区域；
- OOF fold proof 必须保持 `test_accessed=false`；
- 每个 heldout 样本只能由未见过该样本的监督 checkpoint 生成；
- 缓存删除前必须确认可重建来源、配置 fingerprint 和 SHA-256 已记录。

原型和 external-knowledge 分支已从当前工作区资产中移除；其历史版本可从
Git tag `m3.6a-r2-oof-complete` 恢复。
