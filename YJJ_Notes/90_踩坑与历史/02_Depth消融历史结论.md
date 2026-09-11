# Depth 消融历史结论

## 1. 文档定位

这份文件只保留最有价值的 Depth ablation 结论。

原先大量 Zero / Shuffle / Spatial Shuffle / Flat / Constant / Scale 等流水账不再作为主笔记维护。

---

## 2. Corrected Core-4 结果

在修正关键点标签后的 same-subject test 上，历史 Core-4 结果：

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9612 | 0.9882 |
| Zero Depth | 0.8955 | 0.9950 |
| Cross-image Shuffled Depth | 0.943054 | 0.986793 |
| Spatial Shuffled Depth | 0.302566 | 0.134790 |

---

## 3. 这组实验能够说明什么

### 第四通道确实被模型使用

True Depth 与 Zero Depth 的 Box 指标出现明显变化：

`0.9612 -> 0.8955`

因此第 4 通道不是完全被忽略。

---

### 网络对 Depth 的空间组织非常敏感

Spatial Shuffle：

- Box mAP50-95：0.302566
- Pose mAP50-95：0.134790

性能大幅下降。

这说明模型依赖的不只是“Depth 通道存在”或一个全局均值，而与第 4 通道的空间组织有关。

---

### 换成其他图片的 Depth 没有像 Spatial Shuffle 那样崩溃

Cross-image Shuffle：

- Box：0.943054
- Pose：0.986793

与 True Depth 相比变化较小。

因此旧实验不能简单解释成：

> 模型已经学会了精确的 RGB-Depth 样本级几何对应。

更稳妥的解释是：

> 模型会利用 Depth 通道，并且对其空间结构非常敏感，但当时的实验还不足以证明它利用的是哪一种具体物理几何信息。

---

## 4. Zero Depth 的特殊现象

Zero Depth：

- Box 下降
- Pose mAP50-95 反而接近 RGB 的高位

这说明：

- 第 4 通道可能同时提供有用信息和干扰；
- 在 same-subject ceiling 场景下，Pose mAP 很难单独解释 Depth 贡献；
- 需要与 coverage、NME 和 Cross-subject 一起看。

---

## 5. 为什么这些结果现在只放历史

这组 Core-4：

- 使用的是修补 Depth 之前的旧 Depth 管线；
- 主要在 same-subject 12 张 test 上完成；
- test 存在 ceiling；
- 不能直接代表当前 `out_npy_filtered` 管线下的模型行为。

因此：

**它可以支持“模型确实会使用第 4 通道”这一历史机制结论，但不能替代当前 repaired Depth 的 Cross-subject 主结果。**

---

## 6. 是否需要重跑

当前方向 1 已经阶段性完成。

除非论文后期明确需要回答：

> repaired Depth 模型是否仍表现出相同的 Zero / Cross-image Shuffle / Spatial Shuffle 机制？

否则暂时不重跑完整消融。

后续优先级转向：

1. 不同网络模型对比
2. YOLO 版本 4 通道对比
