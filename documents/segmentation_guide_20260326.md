# 分割实验操作手册

> **最后更新**：2026-03-26  
> **作者**：同学B  
> **模块**：`robust/segmentation/` + `experiments/run_segmentation_experiment.py`

---

## 1. 快速启动（后台运行，关闭 VS Code / SSH 也不停）

```bash
# 激活环境
conda activate colqwen2-test

# 后台运行（使用本地模型，默认 cuda:1）
cd ~/projects/colpali-segmentation
nohup python experiments/run_segmentation_experiment.py --model ./colqwen2-v1.0 &

# 查看进度（日志自动写入 outputs/<时间戳>/experiment_console.log）
ls outputs/ | tail -1                          # 看最新的时间戳目录名
tail -f outputs/<时间戳>/experiment_console.log  # 实时跟踪
```

> **说明**：不需要手动重定向 `> xxx.log 2>&1`，脚本内部的 `_TeeLogger` 会自动将所有输出同时写入终端和 `outputs/<时间戳>/experiment_console.log`。

---

## 2. 常用命令

### 2.1 完整分割实验（比较所有方法）

```bash
# 运行 clean + adaptive + grabcut + edge 四个条件
nohup python experiments/run_segmentation_experiment.py --model ./colqwen2-v1.0 &
```

默认在 `cuda:1` 上运行（配置在 `experiments/config.py` 的 `DEVICE`）。

### 2.2 单独跑某一种分割方法（通过 run_benchmark.py）

```bash
# Adaptive（自适应阈值）
python experiments/run_benchmark.py --condition segmented --seg_method adaptive

# GrabCut（GMM 前景分割）
python experiments/run_benchmark.py --condition segmented --seg_method grabcut

# Edge（Canny 边缘检测）
python experiments/run_benchmark.py --condition segmented --seg_method edge
```

### 2.3 用 HuggingFace 镜像（国内网络）

```bash
export HF_ENDPOINT=https://hf-mirror.com
nohup python experiments/run_segmentation_experiment.py --model ./colqwen2-v1.0 &
```

### 2.4 切换 GPU

```bash
# 临时指定
nohup python experiments/run_segmentation_experiment.py --device cuda:0 --model ./colqwen2-v1.0 &

# 永久修改：编辑 experiments/config.py 中的 DEVICE
```

---

## 3. 输出结构

每次运行会在 `outputs/<YYYYMMDD_HHMMSS>/` 下生成：

```
outputs/20260326_175131/
├── experiment_console.log     # 完整终端输出（自动生成，不会互相覆盖）
├── all_results.json           # 所有条件的完整指标
├── experiment_log.json        # 实验元信息（设备、模型、参数）
├── summary.txt                # 文本格式结果摘要
├── comparison_chart.png       # nDCG@5 对比柱状图
└── visualizations/            # 分割前后对比图
    ├── docvqa_test_subsampled/
    └── infovqa_test_subsampled/
```

`results/` 目录结构：

```
results/
├── results_clean.json              # 最新一次的结果（兼容 visualize_results.py）
├── results_segmented_adaptive.json
├── results_segmented_grabcut.json
├── results_segmented_edge.json
└── 20260326_175131/                # 归档（不会被覆盖）
    ├── results_clean.json
    ├── results_segmented_adaptive.json
    ├── results_segmented_grabcut.json
    └── results_segmented_edge.json
```

---

## 4. 查看与管理运行中的实验

```bash
# 查看进程是否还在跑
ps aux | grep run_segmentation

# 实时看日志
tail -f outputs/$(ls outputs/ | tail -1)/experiment_console.log

# 停止实验
kill <PID>
```

---

## 5. 单元测试（无需 GPU）

```bash
conda activate colqwen2-test
python -m pytest tests/test_segmentation.py -v
# 预期：16 passed（adaptive 6 + grabcut 5 + edge 5）
```

---

## 6. 当前分割方法一览

| 方法 | 文件 | 关键参数 | 原理 |
|------|------|---------|------|
| **Adaptive** | `robust/segmentation/adaptive_seg.py` | `block_size=51, c_offset=10` | 自适应高斯阈值 + 连通域分析 |
| **GrabCut** | `robust/segmentation/grabcut_seg.py` | `margin_ratio=0.02, iter_count=5` | GMM 前景/背景像素级分割 |
| **Edge** | `robust/segmentation/edge_seg.py` | `canny_low=50, canny_high=150` | Canny 边缘检测 + 膨胀连通域 |

所有方法共享接口：
```python
from robust.segmentation import adaptive_segment, grabcut_segment, edge_segment

result = adaptive_segment(img, mode="whiten")  # 或 "crop"
result = grabcut_segment(img, mode="whiten")
result = edge_segment(img, mode="whiten")
```

所有参数均可通过 PSO 寻优调整。

---

## 7. 本次修改说明（2026-03-26）

### 7.1 修复的问题

| 问题 | 原因 | 修复方式 |
|------|------|---------|
| `experiment_console.log` 在 `outputs/` 根目录被覆盖 | 原来依赖 nohup 手动重定向到固定路径 | 脚本内部用 `_TeeLogger` 自动写入 `outputs/<时间戳>/experiment_console.log` |
| `results/` 下 JSON 文件互相覆盖 | 每次运行写同名文件 | 新增 `results/<时间戳>/` 归档目录，同时保留平坦文件兼容旧代码 |
| 默认 DEVICE 是 `"cuda"` 不指定编号 | config.py 写死 | 改为 `"cuda:1"` |

### 7.2 新增文件

| 文件 | 说明 |
|------|------|
| `robust/segmentation/grabcut_seg.py` | GrabCut 分割：基于 GMM 的前景/背景分割，适合不规则彩色背景 |
| `robust/segmentation/edge_seg.py` | Edge 分割：Canny 边缘 + 膨胀连通域，适合文本密集区域 |

### 7.3 修改文件

| 文件 | 变更 |
|------|------|
| `experiments/config.py` | `DEVICE` 改为 `"cuda:1"` |
| `robust/segmentation/__init__.py` | 导出 `grabcut_segment`、`edge_segment` |
| `experiments/run_segmentation_experiment.py` | ① 加 `_TeeLogger` 自动日志 ② 运行 4 个条件（clean + 3 种分割）③ results 存时间戳目录 ④ 可视化对比所有方法 |
| `experiments/run_benchmark.py` | 新增 `--seg_method` 参数 + results 时间戳归档 |
| `tests/test_segmentation.py` | 6 → 16 个测试用例 |

### 7.4 设计决策

- **最小化修改原则**：未修改 `colpali_engine/` 任何文件，所有新方法统一放在 `robust/segmentation/`
- **统一接口**：三种方法签名一致 `(img, ..., mode="whiten") -> PIL.Image`，方便 PSO 和实验框架统一调用
- **向后兼容**：`visualize_results.py` 仍能读取 `results/results_*.json` 平坦文件
