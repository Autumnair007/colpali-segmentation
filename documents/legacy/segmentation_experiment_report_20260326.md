# 文档分割模块：自适应阈值方法实验报告

> **日期**：2026-03-26  
> **模块**：`robust/segmentation/` — 自适应阈值 + 连通域分析文档分割  
> **实验环境**：ColQwen2-v1.0 + ViDoRe 基准（docvqa / infovqa 各 500 样本）

---

## 1. 概述

本次工作实现了基于**自适应阈值 + 连通域分析**的文档图像分割方法 (`adaptive_segment`)，替代了原有的全局 Otsu 方法。该方法作为 ColQwen2 编码前的预处理步骤，旨在提取文档主体、去除冗余背景区域，减少模型处理的无效 patch。

### 核心结论

| 条件 | DocVQA nDCG@5 | InfoVQA nDCG@5 | 平均 |
|------|:---:|:---:|:---:|
| Clean (基准) | 0.5529 | 0.9107 | 0.7318 |
| Segmented - Otsu (已废弃) | 0.3760 | 0.6566 | 0.5163 |
| **Segmented - Adaptive** | **0.5468** | **0.9129** | **0.7299** |

- **Otsu 方法严重损害性能**（nDCG@5 平均下降 29.5%），已从代码中移除
- **Adaptive 方法基本持平**：DocVQA 微降 1.1%，InfoVQA 微升 0.2%
- Adaptive 方法在干净 PDF 截图上几乎不引入副作用，验证了其安全性

---

## 2. 方法设计

### 2.1 为什么替换 Otsu？

原有 Otsu 方法的问题：

1. **全局阈值**：对整张图计算单一阈值，光照不均时容易将内容区域误判为背景
2. **只取最大轮廓**：无法处理多区域文档（如页面中散布的多个图表）
3. **实验结果差**：DocVQA 上 nDCG@5 从 0.5529 暴跌至 0.3760（-32%）

### 2.2 Adaptive 方法核心流程

```
输入图像 (PIL.Image)
    │
    ▼
灰度化 (cv2.cvtColor)
    │
    ▼
自适应高斯阈值 (cv2.adaptiveThreshold)
  • block_size=51, C=10
  • 局部窗口内计算阈值，对光照不均鲁棒
    │
    ▼
形态学闭操作 (cv2.morphologyEx MORPH_CLOSE)
  • 15×15 矩形核，合并邻近文本/图表区域
    │
    ▼
连通域分析 (cv2.connectedComponentsWithStats)
  • 过滤面积 < 0.1% 总面积的小噪点
  • 保留所有大型内容区域
    │
    ▼
合并 bounding box + padding
    │
    ├─ mode="whiten" → 背景置白，保持原尺寸
    └─ mode="crop"   → 裁剪到内容区域
```

### 2.3 关键参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `block_size` | 51 | 自适应阈值的邻域大小（像素） |
| `c_offset` | 10 | 阈值偏移常数 |
| `morph_kernel_size` | 15 | 形态学核大小 |
| `min_area_ratio` | 0.001 | 最小连通域面积比（过滤噪点） |
| `padding` | 10 | bounding box 外扩像素数 |
| `mode` | "whiten" | 输出模式：whiten / crop |

所有参数均可通过 PSO 寻优调整。

---

## 3. 实验结果详细分析

### 3.1 完整指标

| Condition | Subset | nDCG@5 | Recall@5 | MRR | Preprocess (s/batch) |
|-----------|--------|:------:|:--------:|:---:|:-------------------:|
| clean | docvqa_test_subsampled | 0.5529 | 0.6160 | 0.5506 | 0.0000 |
| clean | infovqa_test_subsampled | 0.9107 | 0.9420 | 0.9041 | 0.0000 |
| segmented_adaptive | docvqa_test_subsampled | 0.5468 | 0.6100 | 0.5433 | 1.0817 |
| segmented_adaptive | infovqa_test_subsampled | 0.9129 | 0.9440 | 0.9064 | 1.4720 |

### 3.2 分析

**DocVQA (表格/表单类文档)**：
- nDCG@5 微降 0.0061（-1.1%）
- 原因：多数 docvqa 样本是白底 PDF 截图，背景本身就是白色，分割几乎没有可操作空间
- 微降可能来自分割时极少数情况下切掉了文档边缘的少量内容

**InfoVQA (信息图/海报类文档)**：
- nDCG@5 微升 0.0022（+0.2%）
- 信息图背景更丰富（有色彩、纹理），分割能有效去除一些干扰 patch
- 提升幅度很小，说明干净图上分割的边际收益有限

**预处理耗时**：
- 约 1.0-1.5 秒/batch（batch_size=4），纯 CPU 操作
- 相比模型编码时间（~7s/batch），预处理开销占比约 15%

### 3.3 关于 Otsu 方法失败的原因

Otsu 在 DocVQA 上 nDCG@5 直接跌到 0.3760（-32%），原因：
- 白色背景 + 黑色文字时，Otsu 二值化后文字区域为前景、背景为白色
- 形态学闭操作把文字块合并，但**最大轮廓**可能不覆盖全部内容
- 部分图的文档区域被不当裁切，丢失关键信息

这正是改用自适应阈值 + 连通域分析的理由——保留所有显著区域而非只取最大一个。

---

## 4. 代码变更清单

### 新增文件

| 文件 | 说明 |
|------|------|
| `robust/segmentation/adaptive_seg.py` | 自适应阈值 + 连通域分析的分割实现 |
| `experiments/run_segmentation_experiment.py` | 独立的分割实验运行脚本，输出到 `outputs/<timestamp>/` |

### 修改文件

| 文件 | 变更 |
|------|------|
| `robust/segmentation/__init__.py` | 导出 `adaptive_segment`，移除 `segment_document` |
| `experiments/run_benchmark.py` | `--condition segmented` 直接使用 adaptive 方法；修复 `score_multi_vector` 调用（传 list 而非 stack） |
| `tests/test_segmentation.py` | 替换为 adaptive 方法的 6 个测试用例 |

### 删除文件

| 文件 | 原因 |
|------|------|
| `robust/segmentation/document_seg.py` | Otsu 方法实验验证效果极差，完全移除 |

---

## 5. 接口与使用

### 5.1 作为独立函数调用

```python
from robust.segmentation.adaptive_seg import adaptive_segment
from PIL import Image

img = Image.open("document.png")
result = adaptive_segment(img, mode="whiten")   # 背景置白
result = adaptive_segment(img, mode="crop")     # 裁剪到内容区域
```

### 5.2 通过实验框架调用

```bash
# 通过 run_benchmark.py
python experiments/run_benchmark.py --condition segmented

# 通过独立实验脚本（推荐，输出更完整）
python experiments/run_segmentation_experiment.py --device cuda:0
```

### 5.3 后台运行（适合长时间实验）

```bash
# 使用 nohup，关闭终端/SSH 也不会停
nohup python experiments/run_segmentation_experiment.py --device cuda:0 \
    > outputs/experiment_console.log 2>&1 &

# 查看进度
tail -f outputs/experiment_console.log
```

如需使用 HuggingFace 镜像（国内网络）：
```bash
export HF_ENDPOINT=https://hf-mirror.com

# 或在 nohup 中传入
nohup env HF_ENDPOINT=https://hf-mirror.com \
    python experiments/run_segmentation_experiment.py --device cuda:0 \
    > outputs/experiment_console.log 2>&1 &
```

如使用本地已下载的模型：
```bash
python experiments/run_segmentation_experiment.py \
    --device cuda:0 --model ./colqwen2-v1.0
```

---

## 6. 实验输出说明

每次运行会在 `outputs/<YYYYMMDD_HHMMSS>/` 下生成：

```
outputs/20260326_111355/
├── all_results.json           # 所有条件的完整指标
├── experiment_log.json        # 实验元信息（设备、模型、参数等）
├── summary.txt                # 文本格式的结果摘要
├── comparison_chart.png       # nDCG@5 对比柱状图
└── visualizations/            # 分割前后对比图
    ├── docvqa_test_subsampled/
    │   ├── sample_000.png
    │   └── ...
    └── infovqa_test_subsampled/
        ├── sample_000.png
        └── ...
```

`outputs/` 已在 `.gitignore` 中，不会被提交。

---

## 7. 单元测试

```bash
python -m pytest tests/test_segmentation.py -v
```

共 6 个测试用例：
1. `test_adaptive_returns_pil` — 返回 PIL.Image 类型
2. `test_adaptive_whiten_same_size` — whiten 模式保持原尺寸
3. `test_adaptive_crop_smaller_or_equal` — crop 模式尺寸不超过原图
4. `test_adaptive_background_white` — whiten 模式下四角为白色
5. `test_adaptive_handles_white_image` — 全白图不崩溃
6. `test_adaptive_handles_black_image` — 全黑图不崩溃

---

## 8. 后续工作方向

1. **退化场景测试**：在 heavy_noise / motion_blur 等退化条件叠加分割，观察分割是否能抵消部分退化影响
2. **参数优化**：利用 PSO 模块搜索最优的 `block_size`、`c_offset`、`padding` 等参数
3. **GrabCut 方法**：实现基于 GMM 的像素级前景分割，预期在不规则背景文档上效果更好
4. **crop 模式评估**：crop 模式可以减少 patch 数量从而加速推理，但也可能影响模型对位置信息的编码
