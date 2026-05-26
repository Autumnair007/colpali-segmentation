# 2026-05-26 何青泽标准推理对齐与稳健 Late Interaction 实验总结

> 本记录只围绕何青泽的退化不变表示学习方向。退化生成、前端复原、分割、Patch 质量加权等其他成员方向没有改动。

## 1. 本次调整目的

前一轮实验已经证明两件事：

1. `multiview` 对 degraded 数据只有很小的 `nDCG@5 / MRR` 改善，`Recall@5` 不稳定，而且推理成本明显增加。
2. 第一版 `invariant_calibration` 虽然训练 loss 下降，但检索指标低于 degraded singleview，是负结果。

因此本次按“对齐标准原始推理”的原则清理旧改进代码：

```text
标准 baseline = ColQwen2 原图/退化图编码 + 原始 score_multi_vector MaxSim
```

分析代码保留，失败实验记录保留，但不再把旧方法作为当前可用改进入口。

## 2. 代码清理结果

已删除的旧改进代码：

| 文件 | 处理原因 |
| --- | --- |
| `experiments/multiview.py` | 多视图收益太小且推理成本高，不再作为主方法 |
| `tests/test_multiview.py` | 随 multiview 实现删除 |
| `experiments/invariant_calibration.py` | 第一版线性校准是负结果 |
| `experiments/run_invariant_calibration.py` | 不再提供失败校准训练入口 |
| `tests/test_invariant_calibration.py` | 随 calibration 实现删除 |

已回到标准推理的入口：

| 文件 | 当前状态 |
| --- | --- |
| `experiments/run_benchmark.py` | 删除 `--condition multiview`，只保留 clean/degraded/restored/segmented 标准流程 |
| `experiments/run_local_hr_benchmark.py` | 删除 `--use-multiview / --views / --fusion`，固定为 singleview 原始推理 |

保留的分析代码：

| 文件 | 作用 |
| --- | --- |
| `experiments/invariant_embeddings.py` | clean/degraded 页面 embedding 缓存与 mean pooling 工具 |
| `experiments/analyze_invariant_features.py` | 表示漂移、score correlation、query drop case 分析 |
| `experiments/analyze_local_hr_results.py` | 历史结果汇总；其中 multiview/calibration 只作为历史 JSON 兼容 |

## 3. 新方法：稳健 Late Interaction 聚合

本次新增：

```text
experiments/robust_late_interaction.py
experiments/run_robust_late_interaction.py
tests/test_robust_late_interaction.py
```

核心思路是不改 ColQwen2 编码器，也不改标准 baseline，只在单独实验入口中替换 late interaction 的文档 token 聚合方式。

原始 MaxSim：

```text
score(q, d) = sum_i max_j <q_i, d_j>
```

新实验支持：

```text
Top-k mean:
score(q, d) = sum_i mean(TopK_j(<q_i, d_j>, k))

SmoothMax:
score(q, d) = sum_i sum_j softmax(<q_i,d_j>/tau) * <q_i,d_j>
```

设计动机：

```text
退化页面中单个异常 patch 可能抢到 max。
Top-k mean 不只看一个最大响应，而是看前 k 个强响应的平均值，
希望减少单点噪声对排序的影响，同时保持 ColQwen2 多向量结构不变。
```

这个方法仍属于何青泽方向，因为它研究的是冻结编码器下的多向量检索表示与 late interaction 稳健性；它没有改动其他成员的退化、复原或 Patch 加权模块。

## 4. 验证命令

单元测试：

```bash
conda run -n colqwen2-test python -m pytest \
  tests/test_run_local_hr_benchmark.py \
  tests/test_robust_late_interaction.py \
  tests/test_metrics.py \
  -q
```

结果：

```text
12 passed
```

编译检查：

```bash
python -m compileall experiments tests
```

本地 111 页入口 dry-run：

```bash
conda run -n colqwen2-test python experiments/run_local_hr_benchmark.py \
  --mode degraded \
  --variant PD_MB_GN_JC_LR_CS \
  --dry-run \
  --local-files-only \
  --max-queries 2 \
  --max-docs 2
```

dry-run 显示当前入口已经是：

```json
"method": "singleview"
```

## 5. tmux 多卡运行

已按要求使用：

```text
conda activate colqwen2-test
cuda:1 / cuda:2 / cuda:3
tmux 多进程
```

三个会话命令分别为：

```bash
tmux new-session -d -s robust_li_topk2_cuda1 \
  'cd /home/qz/projects/colpali-segmentation && source /home/qz/miniconda3/etc/profile.d/conda.sh && conda activate colqwen2-test && python experiments/run_robust_late_interaction.py --variant PD_MB_GN_JC_LR_CS --device cuda:1 --model ./colqwen2-v1.0 --local-files-only --query-batch-size 1 --score-batch-size 16 --reduction topk_mean --top-k 2 --output-dir results/robust_late_interaction > results/robust_late_interaction/robust_li_topk2_cuda1_20260526.log 2>&1'
```

```bash
tmux new-session -d -s robust_li_topk3_cuda2 \
  'cd /home/qz/projects/colpali-segmentation && source /home/qz/miniconda3/etc/profile.d/conda.sh && conda activate colqwen2-test && python experiments/run_robust_late_interaction.py --variant PD_MB_GN_JC_LR_CS --device cuda:2 --model ./colqwen2-v1.0 --local-files-only --query-batch-size 1 --score-batch-size 16 --reduction topk_mean --top-k 3 --output-dir results/robust_late_interaction > results/robust_late_interaction/robust_li_topk3_cuda2_20260526.log 2>&1'
```

```bash
tmux new-session -d -s robust_li_smoothmax_cuda3 \
  'cd /home/qz/projects/colpali-segmentation && source /home/qz/miniconda3/etc/profile.d/conda.sh && conda activate colqwen2-test && python experiments/run_robust_late_interaction.py --variant PD_MB_GN_JC_LR_CS --device cuda:3 --model ./colqwen2-v1.0 --local-files-only --query-batch-size 1 --score-batch-size 16 --reduction smoothmax --temperature 0.05 --output-dir results/robust_late_interaction > results/robust_late_interaction/robust_li_smoothmax_cuda3_20260526.log 2>&1'
```

由于页面 embedding 缓存已经存在，三个任务运行较快，检查时会话已经正常结束并生成结果。

## 6. 实验结果

结果文件：

| 方法 | 设备 | 结果文件 |
| --- | --- | --- |
| Top-k mean, k=2 | `cuda:1` | `results/robust_late_interaction/20260526_102326_robust_late_interaction_topk2_PD_MB_GN_JC_LR_CS.json` |
| Top-k mean, k=3 | `cuda:2` | `results/robust_late_interaction/20260526_102355_robust_late_interaction_topk3_PD_MB_GN_JC_LR_CS.json` |
| SmoothMax, tau=0.05 | `cuda:3` | `results/robust_late_interaction/20260526_102526_robust_late_interaction_smoothmax_tau0p05_PD_MB_GN_JC_LR_CS.json` |

基线：

| 条件 | nDCG@5 | Recall@5 | MRR |
| --- | ---: | ---: | ---: |
| clean 原始 MaxSim | 0.572484 | 0.575000 | 0.662977 |
| degraded 原始 MaxSim | 0.463308 | 0.415004 | 0.628592 |

新方法：

| 方法 | nDCG@5 | Recall@5 | MRR | nDCG 增益 | Recall 增益 | MRR 增益 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Top-k mean, k=2 | 0.465520 | 0.413472 | 0.639105 | +0.002212 | -0.001533 | +0.010514 |
| Top-k mean, k=3 | 0.468793 | 0.419219 | 0.642425 | +0.005484 | +0.004215 | +0.013833 |
| SmoothMax, tau=0.05 | 0.454974 | 0.405138 | 0.620983 | -0.008334 | -0.009866 | -0.007609 |

## 7. 当前结论

第一，旧改进代码应该退出主线。

```text
multiview 有轻微收益但成本高；
linear calibration 是负结果；
标准推理入口已经重新对齐原始 singleview MaxSim。
```

第二，新方法中 `Top-k mean, k=3` 值得作为下一版主线。

```text
它在 degraded 子集上三项指标均高于原始 MaxSim：
nDCG@5 +0.005484
Recall@5 +0.004215
MRR +0.013833
```

第三，`SmoothMax tau=0.05` 是负结果。

```text
它可能把本应由最强视觉证据承担的匹配响应过度平滑，
导致相关页面和非相关页面之间的分数差变小。
```

第四，Top-k mean 的优势是成本低。

```text
它不需要生成多视图图像，
不需要重新编码文档，
只在已有 query/page embedding 上替换 scoring 聚合。
因此它比 multiview 更适合作为何青泽方向的后续主方法。
```

## 8. 下一步建议

建议下一轮只围绕稳健 late interaction 继续做，不再回到 multiview 或线性校准：

1. 继续跑 `k=4 / k=5`，确认 `k=3` 是否为最优点。
2. 用 `query_drop_cases.csv` 比较 Top-k mean 修复了哪些 degraded rank 下降案例。
3. 增加 per-query 分析：哪些 query 从 Top-k 聚合受益，哪些 query 被平均化伤害。
4. 把报告表述收束为：

```text
冻结 ColQwen2 主干的前提下，
通过稳健 late interaction 聚合减少退化 patch 的单点异常 MaxSim 响应，
在本地 111 页退化子集上获得比原始 degraded MaxSim 更稳定的排序结果。
```

