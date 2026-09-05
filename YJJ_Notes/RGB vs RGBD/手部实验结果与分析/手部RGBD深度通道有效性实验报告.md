# 手部RGBD深度通道有效性实验报告

> 状态：阶段性实验结论，可作为论文实验章节与后续综合分析的基础材料。  
> 实验对象：Hand 单一部位 YOLOv8 Pose，RGB 3 通道 vs RGBD 4 通道。  
> 重点：验证“正确保留深度几何语义的第 4 通道”是否能够提升穴位关键点定位性能，以及模型是否实际利用了第 4 通道。

---

## 1. 实验背景

手部实验最初采用 RGB + `depth_show*.jpg` 构造 4 通道 RGBD 输入。

旧流程为：

```text
RGB
+
SDK depth_show 伪彩图
→ BGR2GRAY
→ 作为第4通道
```

正式 test 结果中，旧 RGBD 并没有超过 RGB：

| 指标 | RGB | 旧 RGBD | RGBD - RGB |
|---|---:|---:|---:|
| Box mAP50-95 | 0.9723 | 0.9270 | -0.0454 |
| Pose mAP50-95 | 0.8958 | 0.8865 | -0.0093 |

这个结果一度表现为：

> 加入第四通道以后，Hand 的 Pose 性能没有提升。

但后续分析发现，问题并不能简单归因于“手部相机的 Depth 没用”，而是旧 Hand 数据处理流程没有可靠保留真实 Depth 的远近关系。

---

## 2. 腹部与手部旧 Depth 管线为什么不同

### 2.1 不是“一个相机有深度、另一个相机没有深度”

这里最重要的结论是：

> **目前证据支持的是“处理流程不同导致 Depth 信息保留程度不同”，而不是“贵相机有用、便宜相机没用”。**

腹部旧实验的第 4 通道来自：

```text
原始 uint16 Depth
→ 逐帧 min-max
→ uint8
→ 第4通道
```

这种方式虽然丢失了绝对毫米尺度，但原始距离之间的顺序仍基本保留。

例如：

```text
1200 mm < 1500 mm < 1800 mm
```

经过单调归一化后仍大致保持：

```text
较小灰度 < 中间灰度 < 较大灰度
```

因此以下信息仍然存在：

- 像素之间的远近顺序；
- 相对深度关系；
- 人体表面的几何起伏。

手部旧实验则来自：

```text
原始 Depth
→ SDK depth_show 伪彩图
→ BGR2GRAY
→ uint8
→ 第4通道
```

`depth_show` 的主要目的通常是让人眼方便观察 Depth，而不是保存原始距离数值。

不同距离先被映射成不同颜色，再将彩色图转为灰度以后：

```text
距离大小
≠
稳定的灰度大小
```

因此真实的远近关系可能被扭曲。

所以更准确的表述应当是：

```text
腹部旧流程：
原始 Depth → 单调压缩 → 相对深度关系基本保留

手部旧流程：
原始 Depth → 伪彩显示图 → 灰度 → 深度关系被明显扭曲
```

### 2.2 手部相机本身其实保存了有效 Depth

手部相机同时保存了原始：

```text
depth_<timestamp>.npy
```

这些数据为 `uint16` 原始 Depth。

后续直接使用这些原始 NPY 重建第 4 通道以后，RGBD Pose 性能由低于 RGB 变成高于 RGB。

因此实验反而说明：

> **手部相机本身包含有用的深度信息，旧实验失败的关键问题更可能出在 Depth 表示与预处理流程，而不是相机“没有深度信息”。**

这比简单归因于相机档次更有方法学价值。

---

## 3. Depth 保真度验证

为了判断最终第 4 通道是否真实反映原始 Depth，对腹部和手部都进行了“原始 Depth vs 最终第4通道”的相关性分析。

### 3.1 腹部

腹部第 4 通道与原始 uint16 Depth 的关系：

- Pearson 均值约：**+0.998**
- Spearman 均值约：**+0.956**
- Spearman 中位数约：**+0.996**
- depth bin 单调性：**+1.000**

说明：

> 腹部最终第 4 通道虽然做了归一化，但远近排序与相对几何信息基本完整保留。

### 3.2 手部旧方案

手部旧第 4 通道与原始 Depth 的关系：

- Pearson 约：**-0.127**
- Spearman 约：**-0.720**
- depth bin 单调性约：**-0.15**
- `depth_show` 中 B=G=R 像素占比仅约 **7%**

说明：

> `depth_show` 实际是伪彩可视化，而不是普通灰度深度图；伪彩图再转灰度以后，最终第 4 通道与真实距离之间不再保持稳定、单调的对应关系。

因此旧 Hand RGBD 实验存在明显的 Depth 表示问题。

---

## 4. 原始 Hand Depth 分布分析

手部原始 Depth 位于：

```text
H:\YJJ\Yolo_RGBD\Resource\session1_200146\out_npy
```

RGB 位于：

```text
H:\YJJ\Yolo_RGBD\Resource\session1_200146\out_image
```

共 81 组 RGB / NPY，timestamp 100% 一一对应。

### 4.1 原始 Depth 统计

全部 81 张：

- shape：`(720, 1280)`
- dtype：`uint16`
- `depth == 0`：约 **12.84%**
- `depth == 65535`：约 **0.012%**
- 有效像素比例：约 **87.15%**

全局有效 Depth：

- P0.5：约 1140 mm
- P1：约 1167 mm
- P5：约 1260 mm
- P50：约 1764 mm
- P95：约 1827 mm
- P99：约 1842 mm
- P99.5：约 1850 mm
- P99.9：约 37409 mm

说明：

> 绝大多数真正有意义的 Hand 深度集中在约 1.1–1.85 m，而极少量远背景像素会延伸到数万毫米。

因此不能直接对整张 uint16 Depth 做普通全范围 min-max，否则极远背景会把人体有效深度压缩到很窄的灰度范围。

---

## 5. 新 Raw-Depth RGBD 映射方案

最终采用全数据集统一的固定映射：

```text
low  = 1100 mm
high = 1850 mm
```

规则：

```text
depth == 0
或 depth == 65535
→ ch4 = 0

有效 Depth：
depth < 1100
→ clip 到 1100

depth > 1850
→ clip 到 1850

1100 mm → 1
1850 mm → 255

中间严格线性映射
```

实际公式：

```text
ch4 = 1 + (clip(depth, 1100, 1850) - 1100) * 254 / 750
```

最终转为 `uint8`。

这样：

- 0 专门表示 invalid Depth；
- 1–255 表示有效相对深度；
- 所有帧使用同一个尺度；
- 保持严格单调关系；
- 不再使用 `depth_show*.jpg`。

---

## 6. 新第 4 通道保真度验证

新 `raw-depth` 第 4 通道生成后，与 clipped 原始 Depth 做相关性验证。

固定 seed=42 抽 20 张：

- Pearson：**0.99999**
- Spearman：**0.99972**
- depth-bin 单调性：**1.0**

具体映射检查：

| 原始深度 | 最终 ch4 |
|---:|---:|
| 1100 mm | 1 |
| 1200 mm | 34 |
| 1500 mm | 136 |
| 1764 mm | 225 |
| 1850 mm | 255 |

结论：

> 新 Hand 第 4 通道已经是真正保持远近顺序的单调相对 Depth 表示。

---

## 7. 新 Raw-Depth 数据集

新数据集：

```text
H:\YJJ\Yolo_RGBD\Resource\session1_200146\dataset\hand\rgbd_rawdepth
```

结构：

```text
rgbd_rawdepth/
├─ images/
│  ├─ train
│  ├─ val
│  └─ test
├─ labels/
│  ├─ train
│  ├─ val
│  └─ test
└─ data_rgbd_rawdepth.yaml
```

划分严格复用原有：

- train：57
- val：12
- test：12

并已确认：

- RGB / 旧 RGBD / 新 raw-depth RGBD 的 train/val/test ID 100% 一致；
- labels 100% 一致；
- 新 RGBD 图片全部为 `(720,1280,4)`、`uint8`；
- RGB 前三通道与原 RGB 逐像素一致；
- 唯一改变的是第 4 通道的 Depth 表示。

因此新实验满足公平对比条件。

---

## 8. 新 Raw-Depth RGBD 正式 Test 结果

RGB 基线模型保持不变。

RGB：

```text
runs/pose/train_pose_rgb_3ch2/weights/best.pt
```

新 Raw-Depth RGBD：

```text
runs/pose/train_pose_rgbd_4ch2/weights/best.pt
```

两者使用同一批 12 张独立 test。

### 8.1 正式结果

| 指标 | RGB | Raw-Depth RGBD | RGBD - RGB |
|---|---:|---:|---:|
| Box mAP50-95 | 0.9723 | 0.9478 | **-0.0246** |
| Pose mAP50-95 | 0.8958 | 0.9349 | **+0.0391** |

### 8.2 最关键结果

Pose mAP50-95：

```text
RGB       = 0.8958
RGBD Raw  = 0.9349
```

提升：

```text
+0.0391
```

即：

> **+3.91 个百分点**

这已经提供了第一层直接证据：

> 在相同数据划分、相同 RGB、相同 labels 和同一独立 test 条件下，加入正确构造的 Raw Depth 后，穴位关键点定位性能获得提升。

---

## 9. “RGBD 比 RGB 高”与“模型真的使用了 Depth”是两个问题

这两个结论应当分开。

### 9.1 RGBD vs RGB 回答的是

> **加入 Depth 以后，最终任务性能有没有提升？**

当前 Hand 结果：

```text
RGB Pose mAP50-95       = 0.8958
Raw RGBD Pose mAP50-95  = 0.9349
提升                     = +0.0391
```

说明：

> 正确构造的 Depth 输入对最终 Pose 任务产生了正收益。

### 9.2 Depth Ablation 回答的是

> **已经训练好的 RGBD 模型，推理时到底有没有依赖第 4 通道？**

这需要对同一个 RGBD 模型保持前三个 RGB 通道完全不变，只破坏或移除第 4 通道，然后观察性能变化。

这类实验可以排除一种可能：

> RGBD 模型虽然有四个输入通道，但训练后实际上完全忽略了第 4 通道。

因此，RGB vs RGBD 和 Depth Ablation 是互补证据。

---

## 10. 已完成的历史 Zero Depth 消融

此前已经对旧 RGBD 模型做过 Depth ablation。

### 10.1 True Depth

保持原 RGBD 输入不变。

### 10.2 Zero Depth

RGB 三通道保持不变，第 4 通道全部置 0。

结果：

| 指标 | Zero Depth |
|---|---:|
| Box mAP50-95 | 0.9010 |
| Pose mAP50-95 | 0.8563 |

相对于 True Depth：

| True - Zero | 差值 |
|---|---:|
| Box mAP50-95 | +0.0260 |
| Pose mAP50-95 | +0.0302 |

该结果说明：

> **这个 RGBD 模型对第 4 通道存在明确依赖或敏感性。**

换句话说，模型并没有完全忽略第四通道。

但这里必须保留一个限定：

> 全 0 Depth 与训练时真实第 4 通道分布明显不同，因此性能下降不能单独解释为“这 3% 全部来自正确的三维几何信息”。

Zero Depth 同时带来了明显的输入分布变化。

---

## 11. 其他历史 Depth 消融结果及解释

此前还做过：

### 11.1 Cross-image Shuffled Depth

保持 RGB 不变，把另一张图像的 Depth 作为当前图像的第 4 通道。

历史结果中，性能与 True Depth 非常接近。

这说明：

> 仅做跨图替换未必足以破坏该数据集中的整体 Depth 结构，因此不能仅靠这一项判断模型是否利用了精确的 RGB-Depth 对齐关系。

### 11.2 Spatial Shuffle Depth

将同一张图的 Depth 像素位置随机打乱，但保留其数值集合。

此前结果出现大幅下降。

这说明：

> 模型对第 4 通道的空间组织非常敏感。

但仍不能单凭这一项写成：

> “空间 shuffle 已经证明模型学会了真实 3D 几何。”

因为像素级随机打乱会制造非常强的异常噪声，同样属于明显的分布外输入。

因此这些消融的正确用途是：

> **证明模型确实在读取并利用第四通道，而不是单独作为“真实几何因果证明”。**

---

## 12. 一个重要限定：历史消融不是针对当前 Raw-Depth 新模型

前面的 Zero / Cross-image Shuffle / Spatial Shuffle 消融是在旧 RGBD 模型上完成的。

因此它能够证明：

> 旧四通道模型确实会使用第 4 通道。

但是现在论文真正关注的是新的：

```text
runs/pose/train_pose_rgbd_4ch2/weights/best.pt
```

也就是 Raw-Depth RGBD 模型。

因此如果要形成最完整、最严谨的证据链，应该：

> **把同样的 Depth Ablation 在新的 raw-depth best.pt 上重新执行一次。**

如果新模型得到：

```text
True Raw Depth
>
Zero Depth
```

并且在破坏真实 Depth 空间关系后 Pose 性能继续下降，那么就可以同时得到：

1. Raw RGBD 比独立 RGB 模型更好；
2. 新 RGBD 模型确实依赖第 4 通道；
3. 破坏 Depth 后性能下降；
4. 因而更有力地支持“有意义的 Depth 信息参与了关键点定位”。

这比单纯报告 RGBD > RGB 更完整。

---

## 13. 当前最强的证据链应该怎样组织

建议最终论文将证据分成三层。

### 第一层：输入有效性

验证：

```text
原始 uint16 Depth
→ 最终 ch4
```

仍保持单调深度关系。

当前已经完成：

```text
Pearson ≈ 0.99999
Spearman ≈ 0.99972
```

证明输入给模型的第四通道确实是可靠的相对 Depth 表示。

### 第二层：最终任务收益

独立训练：

```text
RGB
vs
Raw RGBD
```

同一 test：

```text
Pose mAP50-95
0.8958 → 0.9349
```

提升：

```text
+3.91 个百分点
```

证明加入 Depth 后最终关键点定位性能提高。

### 第三层：模型内部依赖

对同一个 Raw RGBD best.pt：

```text
True Depth
vs
Zero / Perturbed Depth
```

如果破坏 Depth 后 Pose 指标明显下降，则说明：

> 模型实际使用了第四通道，而不是仅仅因为训练随机性得到更好的结果。

三层证据组合起来，比任何单个实验都更有说服力。

---

## 14. GUI 是否应该增加“Depth 有效性测试”

建议增加。

而且它应该作为 RGBD 模型正式测试之后的标准功能，而不是临时脚本。

建议在“模型与权重 / 测试结果”区域增加：

```text
[Depth 消融测试]
```

测试对象：

```text
当前选择的 RGBD best.pt
+
当前 RGBD YAML
+
固定 split=test
```

最少应自动运行：

```text
1. True Depth
2. Zero Depth
```

推荐完整版运行：

```text
1. True Depth
2. Zero Depth
3. Cross-image Shuffled Depth
4. Spatially Shuffled Depth
```

并自动输出：

| Variant | Box mAP50-95 | Pose mAP50-95 | 相对 True Pose |
|---|---:|---:|---:|
| True Depth | - | - | 0 |
| Zero Depth | - | - | - |
| Cross-image Shuffle | - | - | - |
| Spatial Shuffle | - | - | - |

GUI 最后再给出一段只读结论，例如：

```text
Depth 通道依赖：
True Pose mAP50-95 = ...
Zero Pose mAP50-95 = ...
True - Zero = ...

结论：
当前 RGBD 模型对第4通道存在 / 不存在明显依赖。
```

但 GUI 不应自动输出过度结论：

```text
“已证明真实三维几何一定有效”
```

更稳妥的自动描述是：

> “破坏或移除第 4 通道后性能下降，表明当前 RGBD 模型实际依赖 Depth 通道。”

---

## 15. 为什么建议把 Depth Ablation 做进 GUI

因为以后腹部、腿部、手部都可以重复同一套流程：

```text
选择数据集
→ 训练 RGB
→ 训练 RGBD
→ RGB/RGBD 正式 test
→ Depth Ablation
→ 自动得到统一格式结果
```

这样实验协议不会随着人工操作发生变化，也方便论文复现。

同时 GUI 可以强制：

- 固定 `split=test`；
- 使用同一 RGBD best.pt；
- RGB 三通道始终不变；
- 只改变第 4 通道；
- 固定随机 seed；
- 自动保存 CSV/JSON/Markdown 结果；
- 自动记录 YAML、权重路径与时间。

这会比临时命令行测试更适合后续多个身体部位的正式实验。

---

## 16. 如何理解 Box 指标下降

新 Raw-Depth RGBD 的 Box mAP50-95：

```text
RGB  = 0.9723
RGBD = 0.9478
```

下降约 2.46 个百分点。

因此当前不能表述为：

> “RGBD 在所有指标上全面优于 RGB。”

但本研究核心任务是：

> 穴位关键点定位（Pose），而不是单纯目标框检测。

当前实验表现为：

- Box 定位没有获得同步收益；
- Pose 精细定位获得约 +3.91 个百分点提升。

一个合理的阶段性解释是：

> Depth 提供的几何信息对关键点精细定位更有帮助，而对已经接近高精度的目标框检测帮助有限。

该解释仍需结合腹部、腿部实验进一步验证。

---

## 17. 当前能够支持的论文级结论

现阶段可以较稳妥地写：

> **在 Hand 单一部位实验中，当第 4 通道采用保持真实深度单调关系的相对 Depth 表示后，RGBD 模型在独立测试集上的 Pose mAP50-95 从 RGB 模型的 0.8958 提升至 0.9349，提高 3.91 个百分点。该结果表明，正确保留深度几何语义的 Depth 信息能够为穴位关键点定位提供有效的补充信息。**

与此同时，历史 Depth ablation 已经说明：

> **四通道模型并非完全忽略第 4 通道，移除该通道后性能会下降，说明模型能够形成对第四通道的依赖。**

但为了把该结论严格对应到当前 Raw-Depth 模型，应当在新的 `train_pose_rgbd_4ch2/weights/best.pt` 上重新完成一次相同消融。

如果新模型也出现：

```text
True Raw Depth > Zero Depth / Perturbed Depth
```

则可以形成更加完整的结论：

> **不仅 Raw RGBD 模型在最终 Pose 指标上优于 RGB，而且在同一 RGBD 模型内部，移除或破坏 Depth 信息也会造成性能下降，从而进一步支持模型确实利用了深度通道提供的信息。**

---

## 18. 当前不能过度宣称的内容

暂时不要写：

- “RGBD 在所有情况下都优于 RGB”；
- “任何形式的第四通道都会产生收益”；
- “Zero Depth 下降的全部差值都等同于真实 3D 几何贡献”；
- “Spatial Shuffle 一项就已经证明模型学习了真实三维几何”；
- “已经证明所有身体部位均有相同提升”；
- “提升来自相机价格或相机档次”；
- “一次 12 张 test 已经具备很强统计显著性”。

当前 Hand test 只有 12 张，因此后续仍应结合腹部、腿部以及可能的重复实验共同建立更稳定的证据链。

---

## 19. 当前实验最重要的方法学发现

本轮 Hand 实验暴露了两个重要的方法学问题。

第一：

> **RGBD 实验不能只检查“有没有第四通道”，还必须验证第四通道是否真实保留 Depth 的几何语义。**

至少应检查：

- 原始 Depth 与最终第 4 通道的 Spearman；
- 是否保持单调远近关系；
- invalid Depth 如何处理；
- 是否存在极端背景压缩有效动态范围；
- 是否使用统一尺度；
- 是否错误使用伪彩可视化作为深度数值。

第二：

> **RGBD 指标提升和模型实际使用 Depth 是两个不同的问题，最好同时通过独立 RGB/RGBD 对比和 Depth Ablation 两条证据验证。**

---

## 20. 当前阶段一句话总结

> **Hand 实验最终表明：旧的“伪彩 Depth → 灰度”方案没有可靠保留真实深度关系；改用原始 uint16 Depth 并进行统一范围的单调映射后，RGBD 在同一独立 test 上将 Pose mAP50-95 从 0.8958 提升到 0.9349（+3.91 个百分点）。此前的 Depth Ablation 又证明四通道模型确实会依赖第 4 通道。下一步只需在新的 Raw-Depth best.pt 上重新执行同样的 True/Zero/Perturbed Depth 消融，即可把“最终性能提升”和“模型真实使用 Depth”两条证据完整闭环。**
