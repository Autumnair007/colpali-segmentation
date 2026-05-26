# 何青泽退化不变 Adapter 权重

这个目录保存随 PR 提供的最佳退化不变 Adapter checkpoint，供其他组员直接复现实验或在统一评测入口中调用。

## 文件

```text
adapter_checkpoint.pt
adapter_embedding_drift_summary.json
metadata.json
```

checkpoint 内包含：

| 字段 | 含义 |
|---|---|
| `state_dict` | Adapter 权重 |
| `dim` | embedding 维度，当前为 128 |
| `hidden_dim` | MLP hidden 维度，当前为 256 |
| `best_epoch` | validation early stopping 选出的 epoch |
| `best_val_ndcg@5` | best epoch 的验证均值 |
| `training_args` | 训练参数 |
| `query_summary` | query 统计 |

## 训练配置

```text
seed=13
epochs=30
patience=6
batch_query_groups=4
train variants=CS/GN/JC/LR/MB/PD
score_weight=2.0
rank_weight=0.5
identity_weight=0.01
temperature=0.07
lr=1e-3
best_epoch=2
```

## 主要结果

主测试：

```text
test query groups x PD_MB_GN_JC_LR_CS
36 queries
111 pages
```

| 方法 | nDCG@5 | Recall@5 | MRR |
|---|---:|---:|---:|
| degraded original MaxSim | 0.4506 | 0.3727 | 0.5957 |
| degraded + adapter | 0.4997 | 0.4259 | 0.6372 |
| gain | +0.0491 | +0.0532 | +0.0414 |

Proxy restored 的 `nDCG@5` 也均为正提升：

| proxy | nDCG@5 gain |
|---|---:|
| gaussian | +0.0164 |
| nlmeans | +0.0141 |
| wiener | +0.0183 |

`adapter_embedding_drift_summary.json` 保存了 adapter 前后 page-level mean embedding 与 clean embedding 的漂移统计。需要注意的是，最佳 adapter 的提升主要来自 score distillation 和 qrels ranking，而不是让 page-level mean embedding 更接近 clean。

## 使用方式

`experiments/evaluate_invariant_adapter.py` 默认会读取这个 checkpoint：

```bash
python experiments/evaluate_invariant_adapter.py \
  --model ./colqwen2-v1.0 \
  --local-files-only \
  --eval-raw-variant PD_MB_GN_JC_LR_CS
```

如果模型或数据路径不同，请显式传入 `--model`、`--dataset-root` 和 `--cache-root`。

## 边界

该 checkpoint 对主测试组合和 proxy restored 有稳定正收益，但在额外高强度 raw held-out variant 上平均略降。因此它适合作为课程作业阶段的正式结果和可复现权重，不应被表述为“所有退化组合都提升”。
