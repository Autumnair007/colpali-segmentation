# Degradation-Aware Patch Weighting

负责人：郭明坤

本模块用于研究 ColPali 延迟交互阶段的退化感知 Patch 置信度加权。核心目标是在不修改主干模型参数的前提下，降低低质量 Patch 对最终检索得分的误导性贡献。

计划内容：

- Patch 质量或退化置信度估计
- 延迟交互得分的置信度加权
- 加权前后 nDCG@5、MRR、延迟和存储开销对比
- 与前端复原和退化不变表示学习的组合评估

当前目录先保留模块边界，后续实现应优先提供与 `processor.score_multi_vector` 兼容的独立评分函数。
