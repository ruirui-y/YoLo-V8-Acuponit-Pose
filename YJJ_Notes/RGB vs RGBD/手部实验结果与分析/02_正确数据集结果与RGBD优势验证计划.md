# 基于正确数据集的 RGB / RGB-D 当前结果与后续验证计划

> 项目：YOLOv8 RGB-D Acupoint Pose  
> 仓库：https://github.com/ruirui-y/YoLo-V8-Acuponit-Pose.git

---

# 1. 文档目的

本文只使用已经修复关键点 ID / Dataset Path / Cross-subject canonical 问题后的正确数据，整理：

1. 当前可靠的 RGB / RGB-D 实验结果；
2. 目前哪些结论已经成立；
3. 哪些“RGB-D 优于 RGB”的结论目前还不能成立；
4. 后续需要做哪些实验，才能公平、可重复地验证 RGB-D 是否具有优势。

这里的目标不是预设：

```text
RGB-D 一定优于 RGB
```

而是设计足够严格的实验，使最终结果无论支持还是不支持 RGB-D，都具有可信度。

---

# 2. 当前正确数据集

当前修正后的基础数据：

```text
Total : 75
Train : 52
Val   : 11
Test  : 12
Kpts  : 7
```

要求：

```text
RGB train == RGBD train
RGB val   == RGBD val
RGB test  == RGBD test
Labels    完全一致
```

RGB：

```text
3 channels
```

RGB-D：

```text
4 channels
R + G + B + Depth
```

---

# 3. Corrected In-domain Baseline

## RGB

```text
Box:
P         0.9958
R         1.0000
mAP50     0.9950
mAP50-95  0.9473

Pose:
P         0.9958
R         1.0000
mAP50     0.9950
mAP50-95  0.9950
```

## RGB-D

```text
Box:
P         1.0000
R         1.0000
mAP50     0.9950
mAP50-95  0.9612

Pose:
P         1.0000
R         1.0000
mAP50     0.9950
mAP50-95  0.9882
```

差值：

```text
RGBD - RGB

Box  mAP50-95  +0.0138
Pose mAP50-95  -0.0068
```

## 当前解释

可以说：

> RGB-D 在当前 In-domain test 上提高了 Box mAP50-95。

不能说：

> RGB-D 在 Pose 上优于 RGB。

因为：

```text
RGB Pose mAP50-95  = 0.9950
RGBD Pose mAP50-95 = 0.9882
```

而且 test 只有 12 张，存在明显 ceiling effect。

---

# 4. Corrected Core-4 Depth Ablation

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9612 | 0.9882 |
| Zero Depth | 0.8955 | 0.9950 |
| Cross-image Shuffled | 0.9431 | 0.9868 |
| Spatial-shuffled | 0.3026 | 0.1348 |

---

# 5. 当前 Depth Ablation 已经证明什么

## 5.1 第四通道不是摆设

True → Zero：

```text
Box:
0.9612 -> 0.8955
```

下降：

```text
≈ 0.0657
```

因此可以较有把握地说：

> RGB-D 模型确实利用了第四通道。

---

## 5.2 网络明显依赖 Depth 空间结构

Spatial Shuffle：

```text
Box:
0.9612 -> 0.3026

Pose:
0.9882 -> 0.1348
```

说明：

> Depth 的空间组织对当前模型预测非常重要。

---

## 5.3 目前不能过度解释

仍不能只凭 Spatial Shuffle 写：

```text
模型已经理解真实三维几何
```

因为 spatial shuffle 本身是严重 OOD 干预。

更稳妥的表述：

> 模型学习并使用了 Depth 通道中的空间结构信息。

---

# 6. 当前 Cross-subject v2

Cross-subject：

```text
Train : 52
Val   : 11
Test  : 101
```

test 来自训练阶段未见的新 Subject。

所有关键点已经 canonical reorder 到与训练体系一致。

---

# 7. Cross-subject RGB

```text
Box:
P         0.8674
R         0.8911
mAP50     0.9177
mAP50-95  0.5373

Pose:
P         0.9706
R         0.9818
mAP50     0.9945
mAP50-95  0.7608
```

---

# 8. Cross-subject RGB-D

```text
Box:
P         0.8730
R         0.9531
mAP50     0.9497
mAP50-95  0.4808

Pose:
P         0.9092
R         0.9911
mAP50     0.9624
mAP50-95  0.6868
```

---

# 9. Cross-subject 差值

```text
RGBD - RGB

Box mAP50:
+0.0320

Box Recall:
+0.0620

Box mAP50-95:
-0.0565

Pose Recall:
+0.0093

Pose mAP50-95:
-0.0740
```

---

# 10. 当前 Cross-subject 能得到什么结论

当前结果说明 RGB-D 不是“全面更强”。

RGB-D 当前优势：

```text
Box Recall
Box mAP50
Pose Recall
```

RGB 当前优势：

```text
Box mAP50-95
Pose mAP50-95
```

因此目前只能写：

> 在当前单一 Cross-subject 数据上，RGB-D 提升了检测召回率以及较宽松 IoU 条件下的 Box mAP50，但在严格定位与关键点 mAP50-95 上未超过 RGB。

---

# 11. 为什么当前还不能得出最终结论

因为目前 Cross-subject 只有：

```text
一个 external Subject
```

单一 Subject 可能受到：

- 手型差异；
- 手掌大小；
- 姿态；
- 摄像机距离；
- Depth 噪声；
- 标注习惯；
- 光照；
- 遮挡；
- 采集区间；

等因素影响。

因此：

> 一个 Subject 上 RGB 更强或 RGB-D 更强，都不能直接推广为总体结论。

---

# 12. 后续实验总原则

后续所有 RGB / RGB-D 对比必须严格控制：

```text
相同 train / val / test
相同 labels
相同增强
相同 imgsz
相同 batch
相同 epoch
相同 optimizer
相同 seed
相同 early stopping
相同评估代码
相同后处理
```

唯一核心变量：

```text
RGB 3ch
vs
RGB-D 4ch
```

不能通过改变数据划分、阈值或后处理为某一模型制造优势。

---

# 13. 第一优先级：多 Subject Cross-subject

这是下一阶段最重要的实验。

例如：

```text
Subject A：train / val

Subject B：test
Subject C：test
Subject D：test
Subject E：test
...
```

每个 Subject 单独报告：

```text
RGB Box mAP50-95
RGBD Box mAP50-95
RGB Pose mAP50-95
RGBD Pose mAP50-95
```

最后再报告：

```text
Subject-level Macro Average
```

不要只把所有图片混在一起算一个总 mAP，否则图片数量多的 Subject 会主导结果。

---

# 14. 第二优先级：Leave-One-Subject-Out

当 Subject 数量足够后，可以做：

```text
LOSO
Leave-One-Subject-Out
```

例如 5 个 Subject：

```text
Fold 1:
A/B/C/D train
E test

Fold 2:
A/B/C/E train
D test

...
```

RGB 与 RGB-D 在完全相同 fold 上训练和评估。

这会比：

```text
只训练 Subject A → 测 Subject B
```

更能反映模型对“未知新 Subject”的泛化。

---

# 15. 第三优先级：多随机种子重复训练

单次训练可能受初始化与 batch 顺序影响。

建议至少：

```text
3 seeds
```

更理想：

```text
5 seeds
```

例如：

```text
seed = 0
seed = 1
seed = 2
seed = 3
seed = 4
```

最终报告：

```text
mean ± std
```

而不是只选最好的一次。

---

# 16. 第四优先级：Cross-subject Depth Ablation

现有 Core-4 主要是在 In-domain test 上完成。

应该在正确的 Cross-subject benchmark 上重新做：

```text
True Depth
Zero Depth
Cross-image Shuffled Depth
Spatial-shuffled Depth
```

重点回答：

> 在未知 Subject 上，Depth 是否仍然提供有效信息？

如果：

```text
True > Zero
```

在多个 Subject 上重复成立，就比当前同主体 ablation 更有说服力。

---

# 17. 第五优先级：距离分层测试

Depth 最可能发挥作用的因素之一是距离信息。

可以按真实 Depth / 手部平均距离把 test 分成：

```text
Near
Mid
Far
```

然后分别比较 RGB / RGB-D。

例如：

| Distance | RGB Pose | RGB-D Pose |
|---|---:|---:|
| Near | ? | ? |
| Mid | ? | ? |
| Far | ? | ? |

如果 RGB-D 的优势主要出现在特定距离范围，这是有意义的研究结果。

---

# 18. 第六优先级：姿态 / 角度分层

将 test 按手部姿态难度划分：

```text
正面
轻微旋转
大角度旋转
倾斜
局部遮挡
```

观察：

```text
RGB degradation
vs
RGB-D degradation
```

如果 RGB-D 在困难姿态下下降更慢，可以形成比“总 mAP 高一点”更有解释力的结论。

---

# 19. 第七优先级：RGB 退化鲁棒性测试

为了测试 Depth 是否在 RGB 视觉质量下降时提供补充信息，可以设计受控干预：

```text
Brightness ↓
Contrast ↓
Blur ↑
Partial Occlusion
```

要求：

- RGB 和 RGB-D 使用同一张 RGB；
- 对 RGB 部分施加完全相同的干预；
- RGB-D 的 Depth 保持真实；
- 不修改 label；
- 不重新训练，只测试鲁棒性。

比较：

```text
性能下降幅度 Δ
```

如果：

```text
RGBD drop < RGB drop
```

则可以说明：

> Depth 在 RGB 信息退化时提供了额外鲁棒性。

这类实验往往比单纯比较 clean test 更容易体现多模态输入的价值。

---

# 20. 第八优先级：Depth 退化曲线

不要只做：

```text
True
Zero
```

还可以做逐级退化：

```text
Depth noise 10%
Depth noise 20%
Depth noise 30%
...
```

或者：

```text
Depth scale perturbation
Depth blur
Depth quantization
```

然后画：

```text
Depth quality
vs
Pose / Box performance
```

这能回答：

> RGB-D 的收益是否随着 Depth 质量连续变化？

---

# 21. 第九优先级：失败案例成对分析

对同一 test 样本分成四类：

```text
A. RGB 对，RGB-D 对
B. RGB 对，RGB-D 错
C. RGB 错，RGB-D 对
D. RGB 错，RGB-D 错
```

重点看 C 类：

```text
RGB 错
RGB-D 对
```

检查它们是否集中在：

- 手部边界不清；
- 背景接近肤色；
- RGB 光照差；
- 深度边缘明显；
- 遮挡；
- 大角度。

同时也必须分析 B 类，不能只展示 RGB-D 成功案例。

---

# 22. 第十优先级：关键点级别误差

mAP 是整体指标。

可以进一步统计每个 K：

```text
K0 error
K1 error
...
K6 error
```

例如：

```text
normalized pixel error
PCK
OKS-based AP
```

如果 RGB-D 只对某些穴位有优势，也可能与这些点的空间结构或遮挡相关。

---

# 23. 第十一优先级：Box 与 Pose 分开解释

当前 Cross-subject 已经出现：

```text
RGB-D Box Recall 更高
但 mAP50-95 更低
```

这说明：

> “找得到”与“定位得够不够精确”是两件事。

后续应该分别分析：

```text
Detection Recall
High-IoU localization
Pose localization
```

不能只用一个综合指标概括所有行为。

---

# 24. 第十二优先级：模型容量公平性

4 通道模型参数略多于 3 通道模型。

应记录：

```text
RGB parameters
RGB-D parameters
GFLOPs
Inference latency
VRAM
```

并说明：

> RGB-D 的额外收益是否值得额外输入与计算成本。

如果最终精度提升很小但成本明显增加，也需要如实报告。

---

# 25. 统计显著性

如果未来 Subject 和 seed 数量足够，应避免只比较：

```text
0.7608 vs 0.7700
```

而应增加：

```text
paired bootstrap
confidence interval
subject-level paired comparison
```

推荐核心统计单位优先使用：

```text
Subject
```

而不是把连续的每一帧当成完全独立样本。

因为连续帧之间高度相关。

---

# 26. 建议的最终主表

未来论文主表可以设计为：

| Setting | Model | Box mAP50-95 | Pose mAP50-95 |
|---|---|---:|---:|
| In-domain | RGB | 0.9473 | 0.9950 |
| In-domain | RGB-D | 0.9612 | 0.9882 |
| Cross-subject B | RGB | 0.5373 | 0.7608 |
| Cross-subject B | RGB-D | 0.4808 | 0.6868 |
| Cross-subject C | RGB | ... | ... |
| Cross-subject C | RGB-D | ... | ... |
| Cross-subject D | RGB | ... | ... |
| Cross-subject D | RGB-D | ... | ... |
| Macro Avg | RGB | ... | ... |
| Macro Avg | RGB-D | ... | ... |

这样最终结论来自：

```text
多个 Subject
```

而不是一个人。

---

# 27. 建议的 Ablation 主表

| Test Setting | Depth Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---|---:|---:|
| In-domain | True | 0.9612 | 0.9882 |
| In-domain | Zero | 0.8955 | 0.9950 |
| In-domain | Cross-image Shuffle | 0.9431 | 0.9868 |
| In-domain | Spatial Shuffle | 0.3026 | 0.1348 |
| Cross-subject | True | ... | ... |
| Cross-subject | Zero | ... | ... |
| Cross-subject | Cross-image Shuffle | ... | ... |
| Cross-subject | Spatial Shuffle | ... | ... |

---

# 28. 什么时候可以说“RGB-D 优于 RGB”

建议至少满足：

1. 多个 Subject；
2. 相同 split；
3. 多 seed；
4. 主要指标预先确定；
5. 多数 Subject / fold 上方向一致；
6. 平均优势不是由单一 Subject 拉动；
7. 有置信区间或统计检验；
8. Depth ablation 证明第四通道确实参与；
9. 可视化失败案例支持定量结果；
10. 不存在 label / canonical / dataset leakage 问题。

如果最终只在：

```text
Box Recall
```

上稳定更强，而 Pose mAP 没有更强，也应该写：

> RGB-D 在召回率方面具有优势。

而不是扩大成：

> RGB-D 全面优于 RGB。

---

# 29. 如果最终 RGB-D 仍然没有全面超过 RGB

这也不是项目失败。

仍然可能形成有价值的结论：

```text
RGB-D 模型显著使用 Depth；
Depth 空间结构对预测重要；
Depth 提升了 detection recall；
但当前网络融合方式无法把 Depth 优势稳定转化为高精度 Pose 泛化。
```

这时后续研究方向会自然变成：

- 更好的 RGB / Depth fusion；
- Depth normalization；
- mid-level fusion；
- attention；
- Depth-specific branch；
- 更大多 Subject 数据集。

这比强行寻找一个“RGB-D 必须赢”的测试更科学。

---

# 30. 建议的下一步顺序

按优先级：

```text
1. 新增 Subject C / D / E
2. 每个 Subject 第一帧人工 canonical K0~K6
3. 构建统一 Cross-subject benchmark
4. RGB / RGB-D 同模型配置评估
5. 多 seed
6. Subject-level 汇总
7. Cross-subject Core-4 Ablation
8. 距离 / 姿态分层
9. RGB degradation robustness
10. 失败案例与关键点级误差分析
11. 统计显著性
12. 最终论文表格
```

---

# 31. 当前阶段最重要的研究判断

当前已有证据支持：

```text
Depth channel is used.
```

当前证据暂不支持：

```text
RGB-D universally outperforms RGB.
```

下一阶段真正需要证明的是：

> 在预先定义、无泄漏、多 Subject、重复训练的测试条件下，RGB-D 是否在某些明确指标或困难场景中表现出稳定、可重复的优势。

如果能够做到这一点，论文结论会比单纯追求一次更高的 mAP 更可靠。
