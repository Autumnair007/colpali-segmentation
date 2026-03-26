# 课程大作业实操指南：怎么跑通整个项目

> 面向四个人的研究生课程大作业，不需要搞大型科研，重点是展示"实验→分析→结论"的完整流程。

---

## 1. 先搞懂我们到底在做什么

### 一句话概括

> 我们用一个**已经训练好的** ColQwen2 模型，在 ViDoRe 测试集上跑推理，对比"干净图→退化图→复原/分割后的图"的检索性能差异。

**不需要训练、不需要微调**，只做推理（inference）和评估（evaluation）。

### 实验本质

```
          ┌─────────────┐
          │  ViDoRe 测试集  │ ← 从 HuggingFace 自动下载
          │  (query, image) │
          └──────┬──────┘
                 │
    ┌────────────┼────────────┐
    │            │            │
    ▼            ▼            ▼
  干净图      退化图       处理后的图
  (原图)    (加噪/模糊等)  (复原/分割后)
    │            │            │
    ▼            ▼            ▼
  ColQwen2    ColQwen2     ColQwen2
  编码+检索   编码+检索    编码+检索
    │            │            │
    ▼            ▼            ▼
 nDCG@5=X    nDCG@5=Y     nDCG@5=Z
    │            │            │
    └────────────┼────────────┘
                 ▼
         对比 X, Y, Z → 得出结论
```

就这么简单。每次实验就是换一种图像预处理方式，然后看检索分数变了多少。

---

## 2. 数据集说明

### 2.1 数据从哪来？

数据集托管在 HuggingFace Hub 上，**代码会自动下载**，不需要手动下载。

```python
# 这一行代码会自动从 HuggingFace Hub 下载数据集到本地缓存
ds = load_dataset("vidore/docvqa_test_subsampled", split="test")
```

**首次下载**需要网络。如果在国内网络不好，设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

下载后会缓存在 `~/.cache/huggingface/datasets/`，以后再运行不需要重新下载。

### 2.2 数据集长什么样？

每个数据集就是一张表，每行包含：

| 字段 | 类型 | 说明 |
|------|------|------|
| `query` | 字符串 | 一个自然语言问题，例如 "What is the total revenue?" |
| `image` | PIL 图像 | 对应的文档页面截图（PDF 的一页） |

**关键规则**：第 i 个 query 对应第 i 个 image，就是说 query[0] 的正确答案在 image[0] 里。

评估时把**所有 query 对所有 image** 计算相似度，看每个 query 能不能把自己对应的 image 排在前面。

### 2.3 我们用哪些数据集？

当前配置使用 **2 个** subsampled 子集，每个约 **500 个样本**：

| 子集 | 样本数 | 内容 |
|------|-------|------|
| `vidore/docvqa_test_subsampled` | ~500 | 文档问答（英文表格、表单等） |
| `vidore/infovqa_test_subsampled` | ~500 | 信息图问答（图表、海报等） |

完整 ViDoRe 基准有 10 个子集，但**两个就够了**——大作业不需要跑全部。如果想多跑一些增加工作量，可以加几个：

```python
# experiments/config.py 里修改
VIDORE_SUBSETS = [
    "vidore/docvqa_test_subsampled",       # 必跑
    "vidore/infovqa_test_subsampled",      # 必跑
    # 以下可选，想多跑就加
    "vidore/arxivqa_test_subsampled",      # 论文问答，500样本
    "vidore/tabfquad_test_subsampled",     # 法语表格，210样本
]
```

### 2.4 先看看数据长啥样

可以用 Python 快速预览数据集：

```python
from datasets import load_dataset

ds = load_dataset("vidore/docvqa_test_subsampled", split="test")
print(f"样本数: {len(ds)}")
print(f"第一条 query: {ds[0]['query']}")
ds[0]['image'].save("sample_doc.png")  # 保存一张看看
print(f"图片尺寸: {ds[0]['image'].size}")
```

---

## 3. 四个人的分工与完整实验方案

### 3.1 每人做什么

| 同学 | 负责 | 核心产出 |
|------|------|---------|
| A | 图像退化 + 复原 | 退化代码（已有） + 各种退化/复原条件下的 nDCG@5 结果 |
| B（你） | 文档分割 | 分割代码（需改进） + 分割条件下的 nDCG@5 结果 |
| C | 域外泛化 | 收集一些特殊文档图片（中文、手写等），测试模型能力 |
| D | 评估框架 + PSO 寻优 | 评估代码（已有） + 最优参数搜索结果 |

### 3.2 完整实验矩阵（大作业够用的规模）

```
实验条件                              谁负责跑    说明
─────────────────────────────────────────────────────
1. clean (干净原图)                    全组共用    基准线
2. degraded - heavy_noise              同学A      高斯噪声
3. degraded - motion_blur              同学A      运动模糊
4. degraded - jpeg_low                 同学A      JPEG 压缩
5. restored - heavy_noise + nlmeans    同学A      去噪后
6. restored - heavy_noise + wiener     同学A      Wiener 去模糊后
7. segmented (当前 Otsu 方法)          你(B)      现有分割 baseline
8. segmented_v2 (改进方法)             你(B)      你新的分割方法
9. seg + degraded (退化+分割)          你(B)+A    先退化再分割
10. PSO 寻优                           同学D      自动调参
```

总共大约 **10 组实验**，每组跑 2 个数据集，每个数据集 500 样本。

### 3.3 预估时间

- 每组实验（500 样本 × 1 个数据集）：约 5-15 分钟（取决于 GPU）
- 全部跑完：约 2-3 小时 GPU 时间
- PSO 寻优：约 30-60 分钟

---

## 4. 手把手操作步骤

### 第一步：确认环境能跑

```bash
cd /home/qz/projects/colpali-segmentation

# 确保 conda 环境激活
conda activate base  # 或你的环境名

# 跑一下单元测试（不需要 GPU）
python -m pytest tests/ -v
```

### 第二步：预览数据集（不需要 GPU）

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('vidore/docvqa_test_subsampled', split='test')
print(f'样本数: {len(ds)}')
print(f'字段: {ds.column_names}')
print(f'query 示例: {ds[0][\"query\"]}')
ds[0]['image'].save('sample_docvqa.png')
print('已保存 sample_docvqa.png')
"
```

### 第三步：跑 clean baseline（需要 GPU）

```bash
# 确认 GPU 可用
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 如果没有 GPU / 显存不够，改 config.py 里 DEVICE = "cpu"（会很慢）
# 跑 clean baseline
python experiments/run_benchmark.py --condition clean
```

**第一次跑会同时下载模型**（约 4-5GB），之后也会缓存。

预期输出类似：
```
Loading vidore/colqwen2-v1.0...
Evaluating: docvqa_test_subsampled
  nDCG@5=0.XXXX  Recall@5=0.XXXX  MRR=0.XXXX
Evaluating: infovqa_test_subsampled
  nDCG@5=0.XXXX  Recall@5=0.XXXX  MRR=0.XXXX
Saved to results/results_clean.json
```

### 第四步：跑分割实验

```bash
python experiments/run_benchmark.py --condition segmented
```

### 第五步：对比结果

```bash
# 查看两次结果
cat results/results_clean.json
cat results/results_segmented.json
```

比较 nDCG@5 的变化就行。

---

## 5. 你（分割部分）具体该怎么做

### 5.1 最小可行方案（保底能交作业）

1. **跑现有的 Otsu 分割** → 拿到 baseline 分数
2. **实现一个改进版** → 对比是否有提升
3. **可视化几张分割前后的对比图** → 放报告里
4. **写一段分析** → 为什么提升/没提升

### 5.2 你要实现什么代码

在 `robust/segmentation/` 下新增方法。每个方法就是一个函数：

```python
def my_segment(img: Image.Image, **params) -> Image.Image:
    """输入 PIL 图，输出分割处理后的 PIL 图"""
    # 你的分割逻辑
    return processed_img
```

然后在 `run_benchmark.py` 的 `get_preprocessor()` 里加一个分支来调用它。

### 5.3 怎么把新方法接入实验框架

修改 `experiments/run_benchmark.py`，在 `get_preprocessor()` 里扩展：

```python
if condition == "segmented":
    seg_type = kwargs.get("seg_type", "otsu")  # 默认用原版

    if seg_type == "otsu":
        from robust.segmentation.document_seg import segment_document
        return lambda imgs: [segment_document(img) for img in imgs]
    elif seg_type == "grabcut":
        from robust.segmentation.grabcut_seg import grabcut_segment
        return lambda imgs: [grabcut_segment(img) for img in imgs]
    # ... 更多方法
```

---

## 6. 最终提交的作业应该包含什么

### 6.1 实验结果表格（核心）

| 条件 | DocVQA nDCG@5 | InfoVQA nDCG@5 | 平均 |
|------|:---:|:---:|:---:|
| Clean (基准) | X.XX | X.XX | X.XX |
| Heavy Noise | X.XX | X.XX | X.XX |
| Noise + NLMeans 复原 | X.XX | X.XX | X.XX |
| Segmented (Otsu) | X.XX | X.XX | X.XX |
| Segmented (改进方法) | X.XX | X.XX | X.XX |
| Noise + Segmented | X.XX | X.XX | X.XX |

### 6.2 可视化

- 各条件的 nDCG@5 柱状图（`experiments/visualize_results.py` 已有）
- 分割前后的文档图片对比（随便用 matplotlib 画几张）
- （可选）ColQwen2 的相似度热图（interpretability 模块）

### 6.3 分析结论

预期可能的结论方向：
- "在干净 PDF 截图上，分割帮助有限，因为背景本身就是白色"
- "在有噪声/水印的退化图上，分割能去除部分干扰，nDCG@5 提升了 X%"
- "过度激进的分割切掉了文档边缘内容，反而降低了性能"
- "GrabCut 比简单 Otsu 边界更精确，但提升幅度有限 / 显著"

---

## 7. 常见问题

### Q: 显存不够怎么办？

把 `experiments/config.py` 里的 `BATCH_SIZE` 改小，比如 `BATCH_SIZE = 1` 或 `BATCH_SIZE = 2`。

### Q: 模型下载太慢？

```bash
# 方法1：用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方法2：如果你已经有本地模型（colqwen2-v1.0 目录）
# 修改 config.py:
MODEL_NAME = "./colqwen2-v1.0"
PROCESSOR_NAME = "./colqwen2-v1.0"
```

注意：你仓库里已有 `colqwen2-v1.0/` 目录和配置文件，但检查一下里面有没有完整的模型权重（`adapter_model.safetensors` 应该在）。如果是从 HuggingFace 下载的完整模型，可以直接指向本地。

### Q: 只想快速测试不想等太久？

可以只跑一个数据集，并且限制样本数：

```bash
# 只跑 docvqa
python experiments/run_benchmark.py --condition clean --subsets vidore/docvqa_test_subsampled
```

或者临时修改代码限制样本数（调试时用）：

```python
# evaluate_subset() 里加一行
n = min(len(queries), 50)  # 只用前 50 个样本做快速测试
```

### Q: 我们不需要训练对吧？

**对，完全不需要训练**。模型已经训练好了（`vidore/colqwen2-v1.0`），我们只做推理和评估。整个项目的工作量在于：
1. 实现各种图像预处理方法（退化/复原/分割）
2. 跑实验对比性能差异
3. 分析结论

---

## 8. 总结：你接下来该做的事（按优先级）

1. ✅ **跑通模型**（你说已经做了）
2. **下载并预览 ViDoRe 数据集** → 看看数据长啥样
3. **跑 clean baseline** → 确认正常出分数
4. **跑现有 segmented** → 看 Otsu 方法和 clean 的差距
5. **实现 1-2 个改进分割方法** → 参考之前的方案文档
6. **跑对比实验** → 填结果表格
7. **画图 + 写分析** → 完成报告
