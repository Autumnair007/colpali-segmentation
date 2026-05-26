# 退化不变 Adapter

本模块属于何青泽的退化不变特征学习方向。它不训练 ColQwen2 主干，只在已经缓存好的页面多向量 embedding 上训练一个很小的 residual adapter，用于缓解退化图像导致的检索排序下降。

## 方法定位

ColQwen2 原始检索使用多向量 Late Interaction：

```text
score(q, d) = sum_i max_j <q_i, d_j>
```

退化图像会改变 document token/Patch embedding，使同一页面的 clean/degraded 表示发生漂移，也会改变 query-page 分数矩阵。第一版只做 page-level embedding 对齐的线性 calibration 已经证明是负结果，因此本模块改为直接对齐检索分数和排序。

## 当前实现

核心文件：

| 文件 | 作用 |
|---|---|
| `adapter.py` | `ResidualEmbeddingAdapter`，128 维输入输出，LayerNorm + MLP + residual normalize |
| `losses.py` | LateInteraction 分数、score distillation loss、multi-positive ranking loss、identity loss |
| `experiments/build_invariant_splits.py` | 生成 query/variant split manifest |
| `experiments/run_invariant_adapter_training.py` | 训练入口 |
| `experiments/evaluate_invariant_adapter.py` | 评测入口 |

训练目标：

```text
L = score_weight * L_score_distill
  + rank_weight * L_qrels_rank
  + identity_weight * L_identity
```

其中 clean page embeddings 不过 adapter，只作为 teacher；degraded/restored page embeddings 过 adapter。

## 已随仓库提供的权重

最佳 checkpoint:

```text
artifacts/invariant_adapter/tune_distill6_seed13/adapter_checkpoint.pt
```

对应配置：

```text
seed=13
hidden_dim=256
score_weight=2.0
rank_weight=0.5
identity_weight=0.01
best_epoch=2
```

主测试 `PD_MB_GN_JC_LR_CS` 上：

```text
nDCG@5:   0.4506 -> 0.4997  (+0.0491)
Recall@5: 0.3727 -> 0.4259  (+0.0532)
MRR:      0.5957 -> 0.6372  (+0.0414)
```

## 使用方式

评测默认会读取随仓库提供的 checkpoint：

```bash
python experiments/evaluate_invariant_adapter.py \
  --model ./colqwen2-v1.0 \
  --local-files-only \
  --eval-raw-variant PD_MB_GN_JC_LR_CS
```

如果需要换 checkpoint：

```bash
python experiments/evaluate_invariant_adapter.py \
  --checkpoint path/to/adapter_checkpoint.pt \
  --model ./colqwen2-v1.0 \
  --local-files-only
```

训练新 adapter：

```bash
python experiments/run_invariant_adapter_training.py \
  --model ./colqwen2-v1.0 \
  --local-files-only \
  --score-weight 2.0 \
  --rank-weight 0.5 \
  --identity-weight 0.01
```

## 注意事项

- 不要提交 embedding cache；cache 可以通过训练/评测入口重新生成。
- 当前 checkpoint 主要验证主测试组合 `PD_MB_GN_JC_LR_CS`，在额外高强度 held-out raw variant 上泛化还不稳定。
- 后续如果追求泛化，应优先扩大训练 variant 覆盖，例如使用 `--variant-order hard --max-variants 12`，而不是单纯增加 epoch。
