# Hand RGB vs RGBD：当前疑惑与后续验证方向

> 记录时间：2026-09-02  
> 目的：暂时冻结当前实验结论与疑惑，避免后续继续实验时把“现象”和“原因解释”混在一起。

---

## 1. 当前已经确认的实验结果

### RGB 基线

Hand 独立 test：

- Pose mAP50-95 = **0.8958**
- Box mAP50-95 = **0.9723**

### Raw-Depth RGBD

使用真实 Raw Depth NPY 构建第 4 通道后，Hand 独立 test：

- Pose mAP50-95 = **0.9349**
- Box mAP50-95 = **0.9478**

相对 RGB：

- Pose：**+0.0391（+3.91 个百分点）**
- Box：**-0.0246（-2.46 个百分点）**

所以目前可以确认：

> 当前 RGBD 训练得到的模型，在 Hand 的 Pose 指标上明显高于当前 RGB 基线模型。

但这一点本身还不能直接推出：

> “这 +3.91 个百分点全部来自真实 Depth 几何信息。”

---

## 2. 最新 True Depth / Zero Depth 消融结果

对同一个 Raw-Depth RGBD best.pt，不重新训练，只改变测试输入第 4 通道：

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9478 | 0.9349 |
| Zero Depth | 0.9330 | 0.9490 |
| True - Zero | +0.0148 | -0.0141 |

这里出现了一个重要现象：

### Box

True Depth > Zero Depth

说明真实第 4 通道对当前 Box 预测有正向影响。

### Pose

True Depth < Zero Depth

也就是说：

> 对当前 Hand Pose，测试时把 Depth 清零以后，Pose mAP50-95 反而从 0.9349 上升到 0.9490。

因此，当前不能再写：

> “Zero Depth 后 Pose 下降，所以模型依赖真实 Depth 来提高关键点定位。”

实际结果并不支持这个说法。

---

## 3. 当前最大的疑惑

现在同时存在下面三个结果：

```text
RGB 模型 + RGB 输入
Pose = 0.8958

RGBD 模型 + True Depth
Pose = 0.9349

RGBD 模型 + Zero Depth
Pose = 0.9490
```

这意味着需要把两个问题分开：

### 问题 A：RGBD 训练方式为什么得到更好的模型？

RGBD 模型即使在测试时把 Depth 清零，Pose 仍然明显高于原来的 RGB 模型：

```text
0.9490 > 0.8958
```

所以 RGBD 模型相对 RGB 模型的优势，不能简单解释为：

```text
“测试时多输入了一个 Depth 通道”
```

RGBD 在训练阶段引入第 4 通道以后，可能改变了整个网络的优化过程，包括：

- 第一层 RGB 三个通道的权重更新方式；
- 后续特征提取层的优化轨迹；
- Pose Head 的学习过程；
- best epoch 所处位置；
- 整体正则化或训练扰动效果。

也就是说：

> Depth 即使没有在最终推理阶段直接提升 Hand Pose，也可能在训练阶段改变了 RGB 特征的学习方式。

---

## 4. 一个需要特别澄清的地方

当前 RGBD 并不是：

```text
先把 RGB 模型训练完
→ 再从 RGB best.pt 继续训练 RGBD
```

当前实际流程是：

```text
                 yolov8n-pose.pt
                       │
              ┌────────┴────────┐
              │                 │
           RGB训练          转成4通道初始化权重
              │                 │
         RGB best.pt        RGBD训练
                                │
                           RGBD best.pt
```

4 通道初始化的原则是：

- RGB 三个卷积通道保留原始预训练权重；
- 第 4 通道使用 RGB 三通道卷积权重均值初始化。

因此 RGB 与 RGBD 是从同一个官方 Pose 预训练模型出发的两条不同训练路线，而不是“RGBD 比 RGB 多训练了一遍”。

---

## 5. 目前不能下的结论

当前不应该写：

> “Hand 实验已经证明真实 Depth 几何信息使 Pose 提升 3.91 个百分点。”

因为 True / Zero 消融并不支持这一因果解释。

也不应该写：

> “Depth 对 Hand Pose 完全没用。”

因为：

- RGBD 模型确实比当前 RGB 模型高很多；
- True Depth 与 Zero Depth 会改变结果；
- Box 在 True Depth 下优于 Zero Depth；
- 测试集只有 12 张，1~2 个百分点的小差异需要谨慎解释。

当前最稳妥的说法是：

> Hand 实验中，Raw-Depth RGBD 训练模型的 Pose mAP50-95 高于当前 RGB 基线；但同一 RGBD 模型的 True/Zero Depth 消融中，Zero Depth 的 Pose 指标反而略高，因此目前还不能把 RGBD 相对 RGB 的 Pose 提升直接归因于推理阶段真实 Depth 信息。需要进一步从 RGB 与 RGBD 的完整训练过程和控制实验中分析原因。

---

## 6. 现在最值得优先研究的问题

下一步先不急着继续增加新的网络或实验。

先比较现有：

- RGB 训练；
- Raw-Depth RGBD 训练；

两次完整训练过程。

重点查看各 epoch：

- train box loss
- train pose loss
- val box loss
- val pose loss
- Box mAP50-95
- Pose mAP50-95
- best epoch
- early stop epoch

重点回答：

> RGBD 是从训练早期就一直优于 RGB，还是在某个阶段才拉开？

以及：

> RGBD 的 Pose 提升，是更快收敛、更低 loss、更高稳定性，还是只是在某个 best epoch 上偶然更高？

---

## 7. 后续可能需要的控制实验

如果完整训练曲线仍无法解释现象，再考虑增加一个更严格的控制组：

```text
RGB 模型：
3 通道 RGB

True-Depth RGBD 模型：
RGB + 真实 Depth

Zero-4ch 模型：
RGB + 永远为 0 的第 4 通道
```

三组尽量保证：

- 相同 train / val / test split；
- 相同 seed；
- 相同 epoch；
- 相同增强参数；
- 相同 optimizer 逻辑；
- 相同 batch；
- 相同 imgsz；
- 4ch 模型使用一致的 4ch 初始化策略。

这个实验的目的不是看“Zero 推理”，而是看：

> 在训练阶段加入真实 Depth，是否真的比“只是多一个第 4 通道”更有效。

如果最终：

```text
True-Depth RGBD > Zero-4ch
```

才更有力地支持：

> 真实 Depth 信息本身在训练阶段提供了额外有效信息。

如果：

```text
True-Depth RGBD ≈ Zero-4ch
```

甚至：

```text
Zero-4ch > True-Depth RGBD
```

那么 Hand 当前的 RGBD 优势就不能主要解释成真实 Depth 几何收益。

---

## 8. 当前阶段的核心认识

现在需要明确区分三个概念：

```text
1. RGBD 模型是否比 RGB 模型分数高
2. RGBD 模型是否会受到第 4 通道影响
3. 真实 Depth 几何信息是否真正带来了性能提升
```

这三个问题不是一回事。

当前 Hand 实验已经较明确回答了第 1 个问题：

```text
RGBD Pose > RGB Pose
```

第 2 个问题表现为：

```text
改变第 4 通道会影响 Box / Pose 结果
```

但第 3 个问题目前仍然没有被真正证明。

后续论文分析应围绕：

> “RGBD 训练为什么产生了更好的 Pose 模型，以及这个优势中到底有多少能够归因于真实 Depth 信息。”

展开，而不是直接把 RGBD > RGB 等同于 Depth 有效。
