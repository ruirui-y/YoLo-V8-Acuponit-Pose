# RGBD Depth 到底学会了什么？——验证计划

> 目标：不急着证明“Depth 有用”或“Depth 没用”，而是一步一步确认：**训练好的 4 通道 RGBD Pose 模型，究竟从第 4 通道学到了什么。**

## 1. 当前已经确认的事实

### 1.1 4 通道权重来源已确认

RGBD 模型不是使用额外训练好的 4 通道模型，而是从同一个基础 3 通道 `yolov8n-pose.pt` 转换得到：

```text
yolov8n-pose.pt
        ↓
build_rgbd_pose_weights.py
        ↓
第一层输入 3ch → 4ch
        ↓
RGB 权重原样继承
Depth 权重 = RGB 三通道第一层卷积权重均值
        ↓
yolov8n-pose_4ch.pt
        ↓
RGBD 数据训练
        ↓
runs/pose/.../weights/best.pt
```

因此，RGBD 分支的第 4 通道不是随机接入，也不是来自另一个预训练 RGBD 模型。

### 1.2 当前 Hand RGBD 模型确实会受到第 4 通道影响

同一个训练完成的 RGBD `best.pt`：

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9478 | 0.9349 |
| Zero Depth | 0.9330 | 0.9490 |
| True - Zero | +0.0148 | -0.0141 |

当前可以确认：

- 第 4 通道会影响模型输出；
- True Depth 对 Box 指标有正影响；
- True Depth 对当前 Hand Test 的 Pose 指标没有表现出正收益；
- Zero Depth 后 Pose 反而更高。

因此目前不能简单写：

> “RGBD Pose 的提升来自真实 Depth 信息。”

更准确的说法是：

> **RGBD 训练改变了模型的学习结果，但真实 Depth 在推理阶段对 Pose 的直接贡献仍未被证明。**

## 2. 现在真正要回答的问题

核心问题：

> **训练好的 RGBD 模型，到底从 Depth 学会了什么？**

目前考虑四种主要可能。

### 假设 A：学到了真实空间几何

模型真正利用了手部表面的高低关系、局部相对深度、手掌或穴位附近的空间结构。

如果成立，那么破坏 Depth 的空间结构应该明显损害结果。

### 假设 B：只学到了 Depth 的整体统计特征

模型可能没有真正理解每个像素的空间几何，只利用了平均深度、前景/背景深度比例、Depth 值分布或大致距离范围。

如果成立，那么空间打乱后可能不会下降太多。

### 假设 C：学到了 RGB 与 Depth 的样本对应关系

模型可能依赖“这张 RGB 必须对应这张 Depth”。

如果把另一张图片的 Depth 配给当前 RGB，结果明显下降，则说明 RGB-D 配对关系重要。

### 假设 D：Depth 是“有用信息 + 干扰信息”的混合

当前结果很符合这种可能：

```text
Box : True > Zero
Pose: True < Zero
```

可能意味着：

- Depth 帮助整体轮廓、前景范围、BBox；
- 但局部 Depth 噪声、孔洞、量化或边缘变化干扰精细关键点定位。

这时不能简单把 Depth 定义为“有用”或“没用”。

## 3. 下一步实验：四组 Depth Ablation

先不改模型结构、不重新训练。

继续使用**同一个已经训练好的 RGBD best.pt**，只修改 Test 输入的第 4 通道。

### True Depth

```text
RGB + 原始正确 Depth
```

作为基准组，保留真实 RGB-D 对应关系以及 Depth 空间结构和统计分布。

### Zero Depth

```text
RGB + 全 0 Depth
```

完全关闭第 4 通道的有效输入。

当前结果：

```text
Box  mAP50-95 = 0.9330
Pose mAP50-95 = 0.9490
```

### Shuffled Depth

```text
当前 RGB + 另一张 Test 图片的完整 Depth
```

保留 Depth 自身空间结构和数值分布，破坏 RGB 与 Depth 的正确样本对应关系。

主要回答：

> **模型是否依赖 RGB-D 一一对应？**

### Spatial Shuffle Depth

```text
当前 RGB + 当前 Depth 的空间位置被打乱
```

尽量保留 Depth 的总体数值分布，破坏空间几何结构。

主要回答：

> **模型到底有没有利用 Depth 的空间结构？**

## 4. 四组结果怎么解释

### True > Spatial Shuffle

说明 Depth 的空间位置/几何结构具有价值。

### True ≈ Spatial Shuffle

说明模型可能没有真正利用 Depth 空间结构，更可能利用整体统计量或把第 4 通道当作普通辅助特征。

### True > Shuffled

说明正确的 RGB-D 样本对应关系具有价值。

### Shuffled ≈ True

说明模型可能不太关心“哪张 Depth 对应哪张 RGB”，需要继续怀疑它学到的是统计捷径而非真实几何。

### Zero > True / Shuffled / Spatial Shuffle

尤其如果 Pose 指标稳定出现这种趋势，则当前 Depth 通道对 Pose 更像干扰源。

但必须结合更大测试集和其它身体区域验证，不能只凭 Hand 的 12 张 Test 下最终结论。

### True > Zero / Shuffled / Spatial Shuffle

这是最理想的 Depth 利用模式：模型不仅使用了第 4 通道，而且使用的是正确 RGB-D 对应下的真实空间结构。

## 5. 实验记录表

### Hand

| Variant | Box mAP50-95 | Pose mAP50-95 | 备注 |
|---|---:|---:|---|
| True Depth | 0.9478 | 0.9349 | 已完成 |
| Zero Depth | 0.9330 | 0.9490 | 已完成 |
| Cross-image Shuffled | 0.9466 | 0.9349 | 已完成 |
| Spatial-shuffled | 0.8699 | 0.8550 | 已完成 |

### Abdomen

| Variant | Box mAP50-95 | Pose mAP50-95 | 备注 |
|---|---:|---:|---|
| True Depth | 待测 | 待测 | |
| Zero Depth | 待测 | 待测 | |
| Shuffled Depth | 待测 | 待测 | |
| Spatial Shuffle Depth | 待测 | 待测 | |

### Leg

| Variant | Box mAP50-95 | Pose mAP50-95 | 备注 |
|---|---:|---:|---|
| True Depth | 待测 | 待测 | |
| Zero Depth | 待测 | 待测 | |
| Shuffled Depth | 待测 | 待测 | |
| Spatial Shuffle Depth | 待测 | 待测 | |

## 6. 当前暂时不要做的事情

在四组消融结果出来之前，先不要：

- 为了让 RGBD 赢而调参；
- 修改 Depth 映射范围；
- 修改 4ch 初始化策略；
- 增加复杂 RGB/Depth 双分支；
- 上注意力融合；
- 改 YOLO backbone；
- 因为一次 Hand 结果就下“Depth 无用”的结论。

先回答：

> **现有模型到底已经学到了什么。**

再决定是否值得改结构。

## 7. Hand 四组消融后的新结论

四组结果：

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9478 | 0.9349 |
| Zero Depth | 0.9330 | 0.9490 |
| Cross-image Shuffled | 0.9466 | 0.9349 |
| Spatial-shuffled | 0.8699 | 0.8550 |

### 7.1 已确认：模型明显依赖第 4 通道的空间排列

True Depth 与 Spatial-shuffled 的差值：

```text
Box  : 0.9478 → 0.8699   -0.0779
Pose : 0.9349 → 0.8550   -0.0799
```

Spatial Shuffle 保留了当前 Depth 的像素值集合和整体统计分布，但破坏了像素的空间位置。

因此当前证据支持：

> **模型并非只读取 Depth 的 mean/std 等整体统计量，而是明显受到第 4 通道空间结构的影响。**

---

### 7.2 新现象：换成另一张测试图的完整 Depth 几乎没有影响

```text
True:
Box  = 0.9478
Pose = 0.9349

Cross-image Shuffled:
Box  = 0.9466
Pose = 0.9349
```

两组结果几乎一致。

这意味着当前实验还不能证明模型依赖：

> **“当前 RGB 必须对应当前样本自己的精确 Depth”。**

但这里有一个重要限制：

当前 Hand Test 只有 12 张图片，而且来自连续几秒的相邻帧。不同样本的手部姿态、距离和 Depth 空间结构可能本来就非常相似。

因此：

> **Cross-image Shuffled 目前只能说明“在这一小批高度相似的测试帧中，换另一张完整 Depth 没有明显影响”，不能直接证明 RGB-D 对应关系不重要。**

---

### 7.3 当前更合理的工作假设

当前模型可能主要利用第 4 通道中的：

- 手掌或目标区域的大体空间轮廓；
- 前景 / 背景分布；
- 连续的低频空间结构；
- 粗尺度几何信息。

而未必强依赖：

- 每一个 Depth 像素的精确毫米值；
- 当前 RGB 与当前 Depth 的严格逐像素唯一对应。

与此同时：

```text
True Depth Pose = 0.9349
Zero Depth Pose = 0.9490
```

说明真实 Depth 输入对当前 Hand Pose 仍可能包含干扰成分。

当前最合理的描述是：

> **Depth 可能同时包含“有用的粗空间结构”和“影响精细关键点定位的局部干扰”。**

这比简单说“Depth 有用”或“Depth 是干扰”更符合现有证据。

---

## 8. 新问题：模型需要的是粗空间结构，还是精细局部 Depth？

四组消融之后，下一层问题变为：

> **模型真正需要的是 Depth 的粗尺度轮廓，还是 Depth 中细粒度的局部高低变化？**

这也是解释 `Zero > True`（Pose）最关键的问题之一。

如果真实 Depth 中的小尺度噪声、孔洞、边缘跳变或量化变化在干扰关键点定位，那么：

> 保留大体空间结构、同时抑制局部变化，可能比原始 True Depth 更好。

---

## 9. 下一步实验：Blurred Depth

下一组只增加一个变体：

```text
Blurred Depth
```

做法：

```text
当前 RGB
+
对当前真实 Depth 做空间平滑后的 Depth
```

目标：

- 保留手掌/目标区域的大体空间结构；
- 保留低频 Depth 信息；
- 削弱局部噪声和细碎深度变化；
- 不改变 RGB；
- 不改变 label；
- 不重新训练；
- 仍然使用同一个 RGBD best.pt。

### 预期解释

如果：

```text
Blurred Pose > True Pose
```

则支持：

> **原始 Depth 中的局部高频变化可能正在干扰 Pose，而粗尺度 Depth 结构可能仍然有价值。**

如果：

```text
Blurred ≈ True
```

则说明局部 Depth 细节可能不是造成 True/Zero 差异的主要原因。

如果：

```text
Blurred < True
```

则说明原始 Depth 的局部细节中可能确实包含模型正在利用的信息。

---

## 10. 实验记录规范

从现在开始，每完成一组实验，都立即在本文档记录以下四类内容：

1. **实验条件**  
   使用什么权重、数据集、split、输入变体，以及哪些条件保持不变。

2. **原始结果**  
   优先记录 Box mAP50-95 与 Pose mAP50-95，不只记录口头结论。

3. **本轮能够确认的结论**  
   只写当前数据真正支持的内容，不把推测写成事实。

4. **由结果产生的新问题**  
   每次只推进一层问题，并明确下一组实验为什么能够回答它。

原则：

> **结果 → 结论 → 新问题 → 下一实验**

而不是先决定结论，再为了证明结论去调模型。

---

## 11. 当前实验主线

```text
问题 1：
第 4 通道有没有被模型使用？
        ↓
True vs Zero
        ↓
确认：会影响模型输出

问题 2：
模型是否利用 Depth 的空间结构？
        ↓
True vs Spatial Shuffle
        ↓
确认：强烈依赖第 4 通道的空间排列

问题 3：
模型是否依赖当前 RGB 对应的那一张 Depth？
        ↓
True vs Cross-image Shuffle
        ↓
当前结果：几乎无差异
但测试帧高度相似，证据不足，暂不下最终结论

问题 4：
模型需要粗尺度 Depth 结构，
还是细粒度局部 Depth？
        ↓
下一步：Blurred Depth
```

当前不要修改模型结构，也不要为了让 RGBD 指标超过 RGB 而调参。

先继续回答：

> **训练好的 RGBD 模型究竟从第 4 通道学会了什么？**


---

## 12. Blurred Depth（11×11）实验结果

实验条件：

```text
模型：同一个已训练完成的 Hand RGBD best.pt
数据：同一个 Hand test split，共 12 张
变体：blurred_depth
Gaussian Blur kernel：11×11
RGB：不变
Label：不变
Depth：仅对有效 Depth 做 masked / normalized Gaussian blur
invalid=0：保持不变
不重新训练
```

结果：

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9478 | 0.9349 |
| Blurred Depth 11×11 | 0.9478 | 0.9349 |
| Zero Depth | 0.9330 | 0.9490 |
| Cross-image Shuffled | 0.9466 | 0.9349 |
| Spatial-shuffled | 0.8699 | 0.8550 |

Blurred Depth 的 JSON 精确结果：

```text
Box mAP50-95  = 0.9477533333333333
Pose mAP50-95 = 0.9349090909090908
```

与 True Depth 的 mAP50-95 完全一致。

### 12.1 本轮能够确认的结论

原先假设：

> 如果原始 Depth 的局部高频噪声正在干扰 Pose，那么进行适度空间平滑后，Pose 可能高于 True Depth。

本轮 11×11 Blur 并没有出现这种现象：

```text
Blurred Pose = True Pose = 0.9349
Blurred Box  = True Box  = 0.9478
```

因此：

> **当前证据不支持“11×11 尺度的局部高频 Depth 变化，是 True Depth 比 Zero Depth 的 Pose 更差的主要原因”。**

同时，结合此前 Spatial Shuffle 的明显下降：

```text
True Pose     = 0.9349
Blurred Pose  = 0.9349
Spatial Pose  = 0.8550
```

现在更支持下面这个方向：

> **模型在意第 4 通道的较大尺度空间组织，但对 11×11 平滑所移除的细粒度局部变化并不敏感。**

换句话说，模型目前更像是在利用某种：

- 粗尺度空间结构；
- 连续区域；
- 手部/前景的大体 Depth 形状；
- 低频几何信息；

而不是依赖 11×11 尺度内的精细 Depth 起伏。

### 12.2 本轮不能确认的事情

本轮不能直接推出：

> “所有局部 Depth 细节都没用。”

原因有两个：

1. 11×11 只代表一种平滑强度；
2. 当前 Hand test 只有 12 张，指标分辨能力有限。

另外，从日志看，Blur 后每张图的 mean/std 变化很小，这提示：

> **当前原始 Depth 本身可能已经比较平滑，11×11 Blur 实际施加的扰动可能还不够强。**

因此还需要进一步量化 Blur 到底改动了多少像素，而不能只看 kernel 名义大小。

---

## 13. 由 Blurred Depth 结果产生的新问题

现在新的问题变成：

> **Depth 的空间结构到底需要保留到多粗的尺度，模型性能才会开始下降？**

当前已经形成一个很清晰的边界：

```text
11×11 Blur
→ 几乎不影响性能

完全 Spatial Shuffle
→ Box / Pose 均下降约 8 个百分点
```

说明“轻度平滑”和“彻底破坏空间结构”之间，存在一个尚未找到的临界区域。

下一步应验证：

> **随着平滑越来越强，模型何时开始失去它依赖的 Depth 空间结构？**

---

## 14. 下一步建议：Blur 强度阶梯实验

下一步暂时不改模型，只扩展同一种实验变量。

建议比较：

```text
True Depth
Blur 11×11   ← 已完成
Blur 31×31
Blur 51×51
```

如果图像尺寸 640×640，这三个尺度足以形成从轻度到明显低通的梯度。

重点不是追求“哪个最好”，而是观察趋势：

### 如果 11 / 31 / 51 都与 True 接近

则进一步支持：

> 模型主要依赖非常粗的 Depth 空间轮廓，而不是局部几何细节。

### 如果从 31 或 51 开始明显下降

则可以定位：

> 模型依赖的 Depth 信息大致存在于比 11×11 更大的空间尺度中。

### 如果某个更强 Blur 的 Pose 高于 True

才重新支持：

> 原始 Depth 中存在会干扰关键点定位的较小尺度信息，但 11×11 平滑强度还不足以去除它。

---

## 15. 当前实验主线（更新）

```text
问题 1：
第 4 通道有没有被模型使用？
        ↓
True vs Zero
        ↓
确认：第 4 通道会影响模型输出

问题 2：
模型是否利用 Depth 的空间结构？
        ↓
True vs Spatial Shuffle
        ↓
确认：对第 4 通道空间排列高度敏感

问题 3：
模型是否要求当前 RGB 对应当前自己的 Depth？
        ↓
True vs Cross-image Shuffle
        ↓
当前 12 张连续帧上几乎无差异
证据受样本高度相似限制

问题 4：
11×11 尺度的局部 Depth 变化是否重要？
        ↓
True vs Blurred 11×11
        ↓
结果完全一致
当前不支持“11×11 尺度局部高频是 Pose 干扰主因”

问题 5：
模型究竟依赖多大尺度的 Depth 空间结构？
        ↓
下一步：更强 Blur 的阶梯实验
```

---

## 16. Blur 强度阶梯实验结果

本轮继续使用同一个 Hand RGBD `best.pt`、同一个 Hand test split（12 张）、同样的评估参数，只改变第 4 通道的 Gaussian Blur 强度。

### 16.1 Depth 扰动强度

仅在原始有效区域 `Depth > 0` 上计算：

```text
Depth MAE = mean(abs(BlurredDepth - TrueDepth))
```

结果：

| Variant | Kernel | Depth MAE mean |
|---|---:|---:|
| Blurred Depth | 11×11 | 0.3431 |
| Blurred Depth 31x31 | 31×31 | 1.4185 |
| Blurred Depth 51x51 | 51×51 | 2.6014 |

说明随着 kernel 增大，Depth 的局部数值确实被逐步改变，但在当前 0~255 的映射尺度下，平均改变量仍然不大。

### 16.2 模型指标

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9478 | 0.9349 |
| Blur 11×11 | 0.9478 | 0.9349 |
| Blur 31×31 | 0.9478 | 0.9349 |
| Blur 51×51 | 0.9478 | 0.9349 |
| Zero Depth | 0.9330 | 0.9490 |
| Cross-image Shuffled | 0.9466 | 0.9349 |
| Spatial-shuffled | 0.8699 | 0.8550 |

三个 Blur 变体的 JSON 中，Box / Pose `mAP50-95` 与 True Depth 的精确值均一致：

```text
Box  = 0.9477533333333333
Pose = 0.9349090909090908
```

### 16.3 本轮能够确认的结论

从 11×11 增强到 51×51 后，当前 test 集指标仍完全不变。

因此当前证据进一步支持：

> **模型对这些 Blur 所削弱的局部 Depth 数值变化不敏感。**

但这里必须修正之前“51×51 可以定位模型依赖尺度”的表述：

> **kernel 大小本身不能直接等价于实际信息破坏强度。**

本轮虽然 kernel 从 11 增加到 51，但有效区域上的平均 Depth 改变量只有：

```text
0.3431 → 1.4185 → 2.6014
```

相对于 0~255 的第 4 通道范围仍然较小。因此现在还不能严谨地说：

> “模型只依赖 51 像素以上的结构。”

更准确的结论是：

> **当前 Hand Depth 本身较平滑；即使做 51×51 Gaussian Blur，实际数值扰动仍有限，而模型对这一级别的扰动完全不敏感。**

---

## 17. 一个更重要的新疑问：模型会不会主要在看 Depth 的“有效区域形状”？

目前几个实验放在一起出现了一个新的线索：

```text
True Depth
→ 正常

Blur 11 / 31 / 51
→ invalid=0 的空间位置完全保留
→ 指标完全不变

Cross-image Shuffled
→ 仍是一张完整正常的 Depth 图
→ 当前连续帧中几乎不变

Spatial Shuffle
→ Depth 数值位置被彻底打乱
→ 同时也破坏了 invalid=0 的空间分布 / 轮廓
→ 指标大幅下降
```

因此，Spatial Shuffle 的大幅下降目前不能全部归因于：

> “连续 Depth 几何被破坏了。”

因为它同时破坏了另一个可能非常重要的信息：

> **Depth 有效 / 无效区域的空间形状（valid mask / silhouette）。**

也就是说，模型有可能主要利用的是：

- 哪些位置有有效深度；
- 手部 / 前景的大体轮廓；
- Depth 有效区域边界；

而不一定是在利用精确的连续深度高低变化。

这是当前比继续无限增大 Blur kernel 更值得验证的问题。

---

## 18. 下一步实验：Flat Valid Depth

下一步只回答一个问题：

> **如果保留 Depth 的有效区域形状，但把所有局部 Depth 高低变化全部抹掉，模型还会不会保持性能？**

建议新增一个变体：

```text
flat_valid_depth
```

做法：

```text
对于每一张当前样本自己的 Depth：

invalid 像素（原始 Depth == 0）
→ 仍然保持 0

所有 valid 像素（原始 Depth > 0）
→ 全部替换成该图 valid 区域的平均 Depth 值
```

即：

```text
True Depth:
有效区域里有连续的高低变化

Flat Valid Depth:
有效区域内部全部变成同一个常数
但有效区域的空间轮廓完全保留
```

这样会保留：

- 当前样本自己的 valid mask；
- 当前样本有效区域的外形；
- 当前样本的平均 Depth 水平。

同时彻底删除：

- 局部 Depth 高低；
- Depth 梯度；
- 局部连续几何；
- 细节深度纹理。

### 18.1 结果如何解释

如果：

```text
Flat Valid ≈ True
```

则强烈支持：

> **模型主要依赖 valid mask / 粗轮廓或全局 Depth 水平，而不是连续的局部 Depth 几何。**

如果：

```text
Flat Valid < True
```

则说明：

> **连续 Depth 数值变化中确实存在模型正在使用的信息。**

如果 Flat Valid 与 True 很接近，下一步再进一步做：

```text
Fixed Valid Depth
```

即所有图片的 valid 像素都填同一个固定常数，连“每张图的平均 Depth”也删除，只留下 valid mask。

这样可以继续区分：

```text
模型使用的是：
valid mask / silhouette

还是：
valid mask + 每张图整体距离水平
```

---

## 19. 当前实验主线（再次更新）

```text
第 4 通道是否被使用？
        ↓
True vs Zero
        ↓
会影响输出

第 4 通道空间排列是否重要？
        ↓
True vs Spatial Shuffle
        ↓
Spatial Shuffle 大幅下降

模型是否依赖 11~51 尺度的局部 Depth 数值细节？
        ↓
Blur 11 / 31 / 51
        ↓
指标完全不变
且实际 Depth MAE 仍较小

新的关键疑问：
Spatial Shuffle 的下降，
到底来自“连续 Depth 几何被破坏”，
还是来自“valid mask / 前景轮廓被破坏”？
        ↓
下一步：
Flat Valid Depth
```

当前不继续机械增加更大的 Gaussian kernel。

下一步优先把：

> **Depth 连续数值信息**

和：

> **Depth 有效区域空间轮廓**

这两类信息拆开验证。

---

## 20. Flat Valid Depth 实验结果

本轮继续使用同一个 Hand RGBD `best.pt`、同一个 Hand test split（12 张）和相同评估参数。

实验定义：

```text
True Depth:
valid mask + 每张图整体 Depth 水平 + 有效区域内部连续 Depth 变化

Flat Valid Depth:
valid mask + 每张图整体平均 Depth 水平
```

`Flat Valid Depth` 中：

- 原始 `Depth == 0` 的位置严格保持 0；
- 原始 `Depth > 0` 的所有像素，被替换为该图片有效区域 Depth 的平均值；
- 因此有效区域内部不再保留局部 Depth 高低、梯度或连续几何变化。

### 20.1 输入扰动强度

全 test 集有效区域上的平均 Depth MAE：

```text
Flat Valid Depth MAE mean = 71.4935
```

相比此前 Blur：

```text
Blur 11×11  MAE = 0.3431
Blur 31×31  MAE = 1.4185
Blur 51×51  MAE = 2.6014
Flat Valid  MAE = 71.4935
```

因此，Flat Valid 对连续 Depth 数值的破坏远强于此前 Gaussian Blur，已经能够作为真正的“删除局部 Depth 几何”实验。

### 20.2 模型结果

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9478 | 0.9349 |
| Flat Valid Depth | 0.9365 | 0.9303 |
| Zero Depth | 0.9330 | 0.9490 |
| Spatial-shuffled | 0.8699 | 0.8550 |

精确差值：

```text
Box:
True - Flat = 0.9477533 - 0.9365130
            = +0.0112403

Pose:
True - Flat = 0.9349091 - 0.9303377
            = +0.0045714
```

### 20.3 本轮能够确认的结论

Flat Valid 保留了：

- 当前样本自己的 valid mask；
- invalid=0 的空间轮廓；
- 当前样本自己的平均 Depth 水平。

但删除了：

- 有效区域内部连续 Depth 数值变化；
- 局部 Depth 高低；
- Depth gradient；
- 局部几何结构。

结果出现：

```text
True > Flat Valid
```

尤其 Box mAP50-95 下降约 0.0112，Pose mAP50-95 下降约 0.0046。

因此当前证据支持：

> **模型并不只是利用 valid mask / 前景轮廓。有效区域内部的连续 Depth 数值变化本身也提供了可利用信息。**

这修正了上一轮“模型也许主要只看有效区域形状”的猜测。

更准确的当前认识是：

> **模型很可能同时利用两类 Depth 信息：**
>
> 1. `valid mask / 大体空间轮廓`
> 2. `有效区域内部的连续 Depth 变化`

### 20.4 与 Spatial Shuffle 联合解释

当前：

```text
True          Box 0.9478  Pose 0.9349
Flat Valid    Box 0.9365  Pose 0.9303
Spatial       Box 0.8699  Pose 0.8550
```

Flat Valid 删除连续 Depth 几何后，只出现中等幅度下降；

Spatial Shuffle 同时破坏：

- valid mask 空间形状；
- 连续 Depth 空间组织；
- RGB 与 Depth 的像素位置关系；

结果下降非常大。

因此目前最合理的判断是：

> **连续 Depth 几何确实有用，但 Spatial Shuffle 的巨大损失不能只由连续几何消失解释。valid mask / 大体空间轮廓本身很可能也占有重要贡献。**

---

## 21. 由 Flat Valid 结果产生的新问题

现在可以把 Depth 信息拆成三层：

```text
True Depth
=
valid mask
+ 每张图整体 Depth 水平
+ 有效区域内部相对 Depth 几何

Flat Valid
=
valid mask
+ 每张图整体 Depth 水平

Fixed Valid（下一步）
=
valid mask

Zero
=
以上全部删除
```

下一步最自然的问题是：

> **Flat Valid 里保留下来的“每张图片整体平均 Depth 水平”有没有被模型利用？**

也就是说：

> 模型到底需要“这张手整体离相机多远”，还是只需要 valid mask 的空间轮廓？

---

## 22. 下一步实验：Fixed Valid Depth

新增：

```text
fixed_valid_depth
```

定义：

先对当前 test split 的所有原始 Depth 有效像素计算一个统一的全局平均值：

```text
global_valid_mean
```

然后对每一张图片：

```text
原始 Depth == 0
→ 仍然保持 0

原始 Depth > 0
→ 全部替换成同一个 global_valid_mean
```

这样每张图片都使用完全相同的有效 Depth 常数。

它保留：

- 每张图片自己的 valid mask；
- invalid=0 的空间轮廓；
- 前景/有效区域形状。

它删除：

- 局部连续 Depth 几何；
- 每张图片自己的平均 Depth；
- 样本之间的整体距离差异。

### 22.1 关键比较

```text
Flat Valid vs Fixed Valid
```

如果：

```text
Fixed ≈ Flat
```

则支持：

> **每张图片自己的整体平均 Depth 水平基本不重要，Flat Valid 剩下的主要有效信息就是 valid mask / 空间轮廓。**

如果：

```text
Fixed < Flat
```

则支持：

> **除了 valid mask，模型还在利用“当前样本整体离相机多远”这一全局 Depth 水平。**

然后再比较：

```text
Fixed Valid vs Zero
```

如果：

```text
Fixed > Zero
```

则进一步支持：

> **即使完全没有连续 Depth 数值，只保留 valid mask / 轮廓本身，也能给模型带来信息。**

---

## 23. 当前实验主线（更新）

```text
第 4 通道是否被使用？
        ↓
True vs Zero
        ↓
会影响模型输出

是否依赖空间排列？
        ↓
True vs Spatial Shuffle
        ↓
高度敏感

局部轻度 Depth 变化是否重要？
        ↓
Blur 11 / 31 / 51
        ↓
指标完全不变

彻底删除有效区域内部连续 Depth 几何呢？
        ↓
True vs Flat Valid
        ↓
Box -0.0112
Pose -0.0046
        ↓
确认：连续 Depth 数值变化本身确实提供信息

新的问题：
Flat Valid 中保留的“每张图平均 Depth 水平”
有没有额外贡献？
        ↓
下一步：Fixed Valid Depth
```

---

## 24. Fixed Valid Depth 实验结果

本轮继续使用同一个 Hand RGBD `best.pt`、同一个 Hand test split（12 张）和相同评估参数。

当前信息剥离链为：

```text
True Depth
=
valid mask
+ 每张图整体 Depth 水平
+ 有效区域内部连续 Depth 几何

Flat Valid Depth
=
valid mask
+ 每张图自己的平均 Depth 水平

Fixed Valid Depth
=
valid mask
+ 所有图片共用同一个固定 Depth 值

Zero Depth
=
第 4 通道全部为 0
```

本轮全局有效 Depth 平均值：

```text
global_valid_mean = 169.9634
fixed_value = 170
```

### 24.1 输入扰动强度

有效区域上的 Depth MAE：

| Variant | Depth MAE mean |
|---|---:|
| Flat Valid Depth | 71.4935 |
| Fixed Valid Depth | 71.5023 |

Flat 与 Fixed 的 MAE 几乎相同。

当前 12 张连续测试帧中，每张图自己的有效区域平均值只在大约 `168~173` 之间变化，而 Fixed 统一使用 `170`。

因此，本轮确实删除了“每张图片自己的整体平均 Depth 水平”，但这批测试样本之间的整体距离差异本来就很小。

### 24.2 模型结果

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9478 | 0.9349 |
| Flat Valid Depth | 0.9365 | 0.9303 |
| Fixed Valid Depth | 0.9365 | 0.9303 |
| Zero Depth | 0.9330 | 0.9490 |

Flat Valid 与 Fixed Valid 的 `mAP50-95` 精确一致：

```text
Box:
Flat  = 0.936513
Fixed = 0.936513

Pose:
Flat  = 0.9303376623376624
Fixed = 0.9303376623376624
```

### 24.3 本轮能够确认的结论

在当前这 12 张连续 Hand 测试帧上：

> **把每张图片自己的平均 Depth 值统一成同一个全局值 170，没有产生可测量的指标变化。**

因此当前证据支持：

> **“每张图片整体离相机多远”这一全局 Depth 水平，在这批高度相似的连续测试帧中不是主要信息来源。**

但这个结论必须限定在当前测试集，因为这些帧来自连续几秒，样本间平均 Depth 本来就只变化约几个灰度级。

不能据此推广为：

> “整体距离信息在所有数据上都没有用。”

---

## 25. 当前最重要的新发现：Depth 信息对 Box 与 Pose 的作用方向不同

把四组放在一起看：

```text
True         Box 0.9478   Pose 0.9349
Flat/Fixed   Box 0.9365   Pose 0.9303
Zero         Box 0.9330   Pose 0.9490
```

可以拆出两个非常重要的现象。

### 25.1 连续 Depth 几何对 Box 和 Pose 都有正贡献

比较：

```text
True vs Fixed
```

由于 Flat 与 Fixed 完全一致，当前样本的整体平均 Depth 水平没有可测量贡献。

因此 True 与 Fixed 之间的主要区别，就是：

> **有效区域内部的连续 Depth 数值/几何。**

差值：

```text
Box:
True - Fixed
= 0.9478 - 0.9365
≈ +0.0112

Pose:
True - Fixed
= 0.9349 - 0.9303
≈ +0.0046
```

这支持：

> **连续 Depth 几何本身对 Box 和 Pose 都提供了正向信息。**

### 25.2 但“非零 valid-mask 型第 4 通道”对 Pose 可能存在负作用

比较：

```text
Fixed Valid vs Zero
```

Fixed Valid 已经没有连续 Depth 几何，只剩：

```text
valid mask × 常数 170
```

结果：

```text
Box:
Fixed - Zero
= 0.9365 - 0.9330
≈ +0.0035

Pose:
Fixed - Zero
= 0.9303 - 0.9490
≈ -0.0187
```

当前现象与下面这个解释相一致：

> **valid-mask / 前景轮廓型第 4 通道可能对 Box 有一点帮助，但对精细 Pose 定位反而形成干扰。**

然后真实连续 Depth 几何又从这个基础上把 Pose 拉回约 `+0.0046`，但仍不足以抵消这一负作用，因此最终出现：

```text
True Pose < Zero Pose
```

注意：

> 这里不能把各个差值严格当作可线性相加的“因果贡献”，因为神经网络是非线性的。

但作为当前实验现象的机制解释，这条线已经非常值得继续验证。

---

## 26. 新问题：Fixed Valid 的影响到底来自“valid mask 形状”，还是来自“非零常数本身”？

`Fixed Valid Depth` 实际上是：

```text
invalid 区域 = 0
valid 区域   = 170
```

所以它同时引入了两个因素：

1. `valid mask / 前景轮廓`
2. 一个非零的第 4 通道常数值 `170`

因此：

```text
Fixed Valid vs Zero
```

还不能单独证明：

> “是 valid mask 的空间形状导致 Pose 下降。”

有可能模型只是对第 4 通道出现大面积非零常数非常敏感。

下一步应该把这两个因素再拆开。

---

## 27. 下一步实验：Full Constant Depth

新增：

```text
full_constant_depth
```

严格定义：

```text
整张第 4 通道
所有像素
全部填同一个 fixed_value = 170
```

即：

```text
Fixed Valid Depth:
invalid = 0
valid   = 170
→ 有 valid mask 空间形状

Full Constant Depth:
全图 = 170
→ 没有 valid mask 空间形状
→ 只有统一的第 4 通道非零基线

Zero Depth:
全图 = 0
→ 没有 valid mask
→ 也没有非零基线
```

### 27.1 关键比较

#### Fixed Valid vs Full Constant

两者使用完全相同的非零值 `170`。

唯一主要差别：

```text
Fixed Valid
→ 保留 valid mask / 空间轮廓

Full Constant
→ 删除 valid mask / 空间轮廓
```

因此：

> **这组比较用于判断 valid mask 的空间形状本身是否影响模型。**

#### Full Constant vs Zero

两者都没有真实 Depth 几何，也没有 valid mask 形状。

主要差别：

```text
Full Constant = 全图 170
Zero          = 全图 0
```

因此：

> **这组比较用于判断模型是否仅仅对第 4 通道的绝对基线/非零激活敏感。**

---

## 28. 当前完整的信息剥离链

```text
True Depth
=
valid mask
+ 样本整体 Depth 水平
+ 连续局部 Depth 几何

Flat Valid
=
valid mask
+ 样本整体 Depth 水平

Fixed Valid
=
valid mask
+ 固定常数 170

Full Constant（下一步）
=
固定常数 170

Zero
=
常数 0
```

目前已经确认：

```text
True > Fixed
→ 连续 Depth 几何确实提供正向信息

Flat == Fixed
→ 当前连续测试帧中，每张图平均 Depth 水平没有可测量贡献

Fixed > Zero（Box）
→ valid-mask 型通道或非零基线对 Box 有少量正作用

Fixed < Zero（Pose）
→ valid-mask 型通道或非零基线对 Pose 有明显负作用
```

下一步通过：

```text
Fixed
vs
Full Constant
vs
Zero
```

把：

> **空间轮廓**

和：

> **非零第 4 通道基线**

彻底拆开。

---

## 29. Full Constant Depth 实验结果

本轮使用同一个 Hand RGBD `best.pt`、同一个 Hand test split（12 张）和相同评估参数。

三组定义：

```text
Zero Depth:
整张第 4 通道 = 0

Fixed Valid Depth:
原始 invalid 区域 = 0
原始 valid 区域   = 170
→ 保留 valid mask / 空间轮廓

Full Constant Depth:
整张第 4 通道 = 170
→ 删除 valid mask / 空间轮廓
→ 只保留统一的非零第 4 通道基线
```

当前统一常数：

```text
global_valid_mean = 169.9634
fixed_value = 170
```

### 29.1 结果

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| Zero Depth | 0.9330 | 0.9490 |
| Fixed Valid Depth | 0.9365 | 0.9303 |
| Full Constant Depth | 0.9563 | 0.9206 |

精确值：

```text
Zero:
Box  = 0.933000
Pose = 0.949000

Fixed Valid:
Box  = 0.936513
Pose = 0.9303376623

Full Constant:
Box  = 0.956250
Pose = 0.9206363636
```

有效区域上的 Depth MAE：

```text
Fixed Valid Depth   = 71.5023
Full Constant Depth = 71.5023
```

二者在原始有效区域上的 MAE 完全一致；它们的主要差别正是 invalid 区域是否保持为 0。

---

## 30. 本轮最重要的结论：必须修正上一轮对 valid mask 的解释

上一轮仅比较：

```text
Fixed Valid vs Zero
```

时，我们还不能区分：

- valid mask 空间形状；
- 第 4 通道非零基线；

各自造成了什么影响。

本轮通过 Full Constant 把二者拆开后，结论变得更清楚。

### 30.1 非零常数基线本身具有非常强的影响

比较：

```text
Full Constant 170 vs Zero 0
```

二者都没有真实连续 Depth 几何，也都没有 valid mask 形状。

主要差别只是：

```text
第 4 通道全图 = 170
vs
第 4 通道全图 = 0
```

结果：

```text
Box:
0.9563 - 0.9330
≈ +0.0233

Pose:
0.9206 - 0.9490
≈ -0.0284
```

这是目前非常强的信号：

> **即使第 4 通道完全没有任何空间结构，只把整张通道从 0 改成常数 170，也足以明显改变模型输出。**

而且方向非常明确：

```text
Box 变好
Pose 变差
```

因此，现在已经不能把所有第 4 通道效应都解释为“模型学到了几何”。

模型还明显对：

> **第 4 通道的绝对基线 / 整体激活水平**

敏感。

---

### 30.2 valid mask 本身的作用方向，与此前初步猜测相反

比较：

```text
Fixed Valid vs Full Constant
```

二者有效区域都等于相同常数 `170`。

差别：

```text
Fixed Valid:
invalid 区域重新变成 0
→ 恢复 valid mask / 空间轮廓

Full Constant:
整张图都为 170
→ 没有 valid mask
```

结果：

```text
Box:
Fixed - Full
= 0.9365 - 0.9563
≈ -0.0197

Pose:
Fixed - Full
= 0.9303 - 0.9206
≈ +0.0097
```

所以在当前实验条件下：

> **恢复 valid mask / 空间轮廓，会降低 Box，但会改善 Pose。**

这说明上一轮根据 `Fixed vs Zero` 得出的：

> “valid-mask 型通道可能对 Pose 有负作用”

不能继续保留为当前解释。

更准确的拆分是：

```text
非零常数基线：
Box 明显上升
Pose 明显下降

在这个非零基线之上恢复 valid mask：
Box 回落
Pose 回升
```

注意：

> 神经网络是非线性的，以上差值不能被当作严格可加的“因果贡献”，但这组受控输入已经能明确说明两种因素作用方向不同。

---

## 31. 当前对 True Depth 的认识进一步复杂化

现在已有：

```text
True Depth        Box 0.9478   Pose 0.9349
Fixed Valid       Box 0.9365   Pose 0.9303
Full Constant     Box 0.9563   Pose 0.9206
Zero              Box 0.9330   Pose 0.9490
```

这说明第 4 通道至少存在三类不同信息：

```text
A. 绝对基线 / 整体数值水平
B. valid mask / 空间轮廓
C. 有效区域内部连续 Depth 几何
```

当前实验显示：

- `A` 单独出现时，对 Box 很有利，对 Pose 很不利；
- 在 `A` 上加入 `B` 后，Box 回落、Pose 回升；
- 再从 Fixed/Flat 恢复真实连续几何 `C`，True 相比 Fixed：
  - Box 进一步提高约 `+0.0112`
  - Pose 进一步提高约 `+0.0046`

因此，当前最合理的描述是：

> **训练好的 RGBD 模型确实在使用第 4 通道，但它使用的不只是几何信息。第 4 通道的绝对数值基线、valid mask 轮廓和连续 Depth 几何都会影响模型，而且 Box 与 Pose 对这些因素的响应方向并不一致。**

这也开始解释为什么：

```text
True Depth Pose < Zero Depth Pose
```

不是因为“模型没学到 Depth”，而更可能是：

> **Depth 通道中不同类型的信息对 Pose 产生了相互竞争的作用，其中某些第 4 通道激活效应对 Pose 的负影响大于真实几何带来的正收益。**

---

## 32. 新问题：模型对“全图常数 Depth”的响应是否随数值大小变化？

现在发现：

```text
全图 0   → Pose 0.9490
全图 170 → Pose 0.9206
```

但只测两个点还无法判断模型到底是：

1. 只要第 4 通道非零就发生变化；
2. 对常数值大小近似单调敏感；
3. 在某些 Depth 数值区间最敏感；
4. 对训练数据常见的 Depth 值附近形成了特殊响应。

因此下一步不再动空间结构。

只做：

> **Full Constant Depth 数值扫描**

把第 4 通道保持为“完全无空间结构的常数图”，只改变常数值。

---

## 33. 下一步实验：Full Constant Value Sweep

建议测试：

```text
0
1
64
128
170
224
255
```

其中：

- `0` = 已有 Zero Depth；
- `170` = 已有 Full Constant Depth；
- 其余为新增控制点。

所有图片：

```text
RGB 不变
Label 不变
第 4 通道整张图只有一个常数
```

这样可以直接得到：

```text
constant value
        ↓
Box mAP50-95 / Pose mAP50-95
```

### 33.1 如何解释

如果：

```text
1 ≈ 64 ≈ 128 ≈ 170 ≈ 224 ≈ 255
但都明显不同于 0
```

则说明：

> **模型可能主要对“第 4 通道是否为零”敏感，而不是对真实 Depth 大小敏感。**

如果指标随常数值逐步变化：

> **模型确实对第 4 通道绝对数值大小敏感。**

如果在训练数据常见范围附近出现特殊峰值或谷值：

> **模型可能学到了某种与 Depth 数值分布相关的响应，而不仅仅是二值开关。**

这一步将帮助判断：

> `Full Constant 170` 的强烈效应，到底是“非零开关效应”，还是“数值幅度效应”。

---

## 34. 当前实验主线（更新）

```text
第 4 通道是否影响模型？
        ↓
True vs Zero
        ↓
确认：会影响

是否依赖空间排列？
        ↓
Spatial Shuffle
        ↓
确认：高度敏感

局部轻微 Depth 变化重要吗？
        ↓
Blur 11 / 31 / 51
        ↓
当前无可测量影响

连续 Depth 几何重要吗？
        ↓
True vs Flat/Fixed
        ↓
确认：有正向信息

每张图整体平均 Depth 重要吗？
        ↓
Flat vs Fixed
        ↓
当前连续帧中无可测量影响

valid mask 与非零基线分别怎样影响模型？
        ↓
Fixed vs Full vs Zero
        ↓
发现：
非零常数基线本身影响很强
Box ↑，Pose ↓
恢复 valid mask 后
Box ↓，Pose ↑

新的问题：
模型到底是对“非零”敏感，
还是对“常数值大小”敏感？
        ↓
下一步：
Full Constant Value Sweep
```

---

## 35. Full Constant Value Sweep 实验结果

本轮继续使用同一个 Hand RGBD `best.pt`、同一个 Hand test split（12 张）和相同评估参数。

实验只改变第 4 通道的全图常数值：

```text
0, 1, 64, 128, 170, 224, 255
```

其中：

- `0` = Zero Depth；
- `170` = 当前 test split 的 global valid mean；
- 所有 Full Constant 变体都没有真实 Depth 几何，也没有 valid mask 空间形状。

### 35.1 结果

| Full Constant Value | Box mAP50-95 | Pose mAP50-95 |
|---:|---:|---:|
| 0 | 0.9330 | 0.9490 |
| 1 | 0.9330 | 0.9490 |
| 64 | 0.9421 | 0.9490 |
| 128 | 0.9473 | 0.9336 |
| 170 | 0.9563 | 0.9206 |
| 224 | 0.9375 | 0.9129 |
| 255 | 0.9388 | 0.9080 |

### 35.2 本轮可以明确排除“只看 0 / 非 0”的解释

如果模型只关心：

```text
第 4 通道是不是非零
```

那么：

```text
1, 64, 128, 170, 224, 255
```

应该大体表现相近。

实际结果不是这样。

特别是 Pose：

```text
0   → 0.9490
1   → 0.9490
64  → 0.9490
128 → 0.9336
170 → 0.9206
224 → 0.9129
255 → 0.9080
```

因此当前证据明确支持：

> **模型对第 4 通道的绝对数值幅度敏感，而不是简单的“零 / 非零开关”。**

---

## 36. Box 与 Pose 对常数幅度的响应完全不同

### 36.1 Pose：高幅度常数越大，性能总体越差

当前 Pose 曲线表现为：

```text
0 / 1 / 64
→ 0.9490，几乎无影响

128
→ 0.9336，开始明显下降

170
→ 0.9206

224
→ 0.9129

255
→ 0.9080
```

这显示在当前 Hand 模型上存在非常明显的幅度响应：

> **当第 4 通道的统一常数升高到一定程度后，Pose 性能持续恶化。**

因此 `True Depth Pose < Zero Depth Pose` 已经有了一个非常重要的机制线索：

> **真实 Depth 的数值大多处于较高范围，而模型对较高第 4 通道绝对幅度本身就可能产生不利于精细 Pose 定位的响应。**

这并不意味着真实几何没用。此前 `True > Fixed` 已经说明连续 Depth 几何本身提供正向信息。

更可能是：

```text
真实 Depth 输入
=
有用的连续几何
+
对高幅度第 4 通道的响应
+
valid mask / 空间组织
```

这些因素共同作用，最终得到当前 Pose 结果。

### 36.2 Box：不是单调曲线，而是在中间值附近出现峰值

Box：

```text
0   → 0.9330
1   → 0.9330
64  → 0.9421
128 → 0.9473
170 → 0.9563   ← 当前最高
224 → 0.9375
255 → 0.9388
```

因此 Box 并不是“常数越高越好”。

而更像：

> **模型对第 4 通道绝对幅度存在非线性响应，在当前测试中约 170 附近出现明显高点。**

这进一步说明第 4 通道的作用不能简单解释成一个二值 mask 或开关。

---

## 37. 当前关于第 4 通道的认识

到目前为止，可以把模型对 Depth 的响应拆成至少三部分：

```text
A. 第 4 通道绝对数值幅度
B. valid mask / 空间轮廓
C. 有效区域内部连续 Depth 几何
```

现有证据分别为：

### A. 绝对幅度

Full Constant Sweep 已确认：

> 模型明显对绝对幅度敏感。

尤其 Pose 在高值区间明显恶化。

### B. valid mask / 空间轮廓

此前：

```text
Full Constant 170
vs
Fixed Valid 170
```

已经证明恢复 valid mask 会改变结果：

```text
Box 下降
Pose 上升
```

说明空间轮廓本身也在起作用。

### C. 连续 Depth 几何

此前：

```text
True
vs
Fixed / Flat
```

证明恢复真实连续 Depth 后：

```text
Box ↑
Pose ↑
```

说明真实连续几何也确实包含模型正在利用的信息。

因此当前最准确的描述是：

> **这个 RGBD 模型确实学到了真实 Depth 几何，但第 4 通道的绝对数值幅度也会强烈影响网络，而且这种幅度效应对 Box 与 Pose 的方向并不一致。**

这比简单说：

> “Depth 是干扰”

或者：

> “Depth 没学会”

都更符合现有实验。

---

## 38. 新问题：幅度效应是否依赖 valid mask / 空间轮廓？

目前 Full Constant Sweep 测的是：

```text
整张图 = 常数
```

也就是说完全没有 valid mask。

但是实际 True Depth 不是全图常数，而是：

```text
invalid = 0
valid   = 非零 Depth
```

所以现在需要回答：

> **Pose 对高幅度常数的恶化，是纯粹的绝对幅度效应，还是“绝对幅度 × valid mask 空间结构”的交互效应？**

要回答它，最自然的方法不是继续增加新的全图常数点，而是在同一组数值上做：

> **Fixed Valid Value Sweep**

---

## 39. 下一步实验：Fixed Valid Value Sweep

对每个指定常数值：

```text
1, 64, 128, 170, 224, 255
```

生成：

```text
原始 Depth == 0
→ 保持 0

原始 Depth > 0
→ 全部填指定常数
```

即保持每张图片自己的 valid mask，但改变 valid 区域内部的统一数值。

最终可以和 Full Constant Sweep 一一对应：

| Value | Full Constant | Fixed Valid |
|---:|---|---|
| 0 | 全图 0 | 等价于 Zero |
| 1 | 全图 1 | mask 内 1 |
| 64 | 全图 64 | mask 内 64 |
| 128 | 全图 128 | mask 内 128 |
| 170 | 全图 170 | mask 内 170 |
| 224 | 全图 224 | mask 内 224 |
| 255 | 全图 255 | mask 内 255 |

这样就得到一个很清晰的二维控制：

```text
变量 1：Depth 常数幅度
变量 2：是否保留 valid mask 空间结构
```

### 39.1 结果如何解释

如果：

```text
Fixed Valid 曲线
≈
Full Constant 曲线
```

则说明：

> **绝对幅度是主导因素，valid mask 只提供较小修正。**

如果：

```text
两条曲线明显不同
```

则说明：

> **第 4 通道幅度与 valid mask / 空间结构存在明显交互，不能把幅度效应单独理解。**

特别关注 Pose：

> `128~255` 高值区间在保留 valid mask 后，是否仍持续恶化。

---

## 40. 当前实验主线（更新）

```text
第 4 通道有没有被使用？
        ↓
确认：会影响模型

真实连续 Depth 几何有没有信息？
        ↓
确认：有

valid mask 有没有影响？
        ↓
确认：有

绝对数值幅度有没有影响？
        ↓
Full Constant Value Sweep
        ↓
确认：非常明显
并非简单 0 / 非 0 开关

Pose:
高幅度区间明显恶化

Box:
中间值附近出现非线性峰值

新的问题：
这种幅度响应是否依赖 valid mask / 空间轮廓？
        ↓
下一步：
Fixed Valid Value Sweep
```

---

## 41. Fixed Valid Value Sweep 实验结果

本轮继续使用同一个 Hand RGBD `best.pt`、同一个 Hand test split（12 张）和相同评估参数。

Fixed Valid Sweep 定义：

```text
原始 Depth == 0
→ 保持 0

原始 Depth > 0
→ 填指定常数值
```

因此它保留：

- 当前样本自己的 valid mask / 空间轮廓；

同时删除：

- 真实连续 Depth 几何；
- 每张图自己的平均 Depth；
- 局部 Depth 纹理。

本轮扫描值：

```text
0, 1, 64, 128, 170, 224, 255
```

### 41.1 Fixed Valid 结果

| Fixed Valid Value | Box mAP50-95 | Pose mAP50-95 |
|---:|---:|---:|
| 0 | 0.9330 | 0.9490 |
| 1 | 0.9330 | 0.9490 |
| 64 | 0.9417 | 0.9490 |
| 128 | 0.9377 | 0.9336 |
| 170 | 0.9365 | 0.9303 |
| 224 | 0.9462 | 0.9344 |
| 255 | 0.9462 | 0.9344 |

对应的 Full Constant Sweep：

| Value | Full Box | Full Pose | Fixed Box | Fixed Pose |
|---:|---:|---:|---:|---:|
| 0 | 0.9330 | 0.9490 | 0.9330 | 0.9490 |
| 1 | 0.9330 | 0.9490 | 0.9330 | 0.9490 |
| 64 | 0.9421 | 0.9490 | 0.9417 | 0.9490 |
| 128 | 0.9473 | 0.9336 | 0.9377 | 0.9336 |
| 170 | 0.9563 | 0.9206 | 0.9365 | 0.9303 |
| 224 | 0.9375 | 0.9129 | 0.9462 | 0.9344 |
| 255 | 0.9388 | 0.9080 | 0.9462 | 0.9344 |

---

## 42. 本轮结论：绝对幅度是主效应之一，但 valid mask 会显著调制高幅度响应

### 42.1 在低值区，两条曲线几乎完全一致

```text
Value 0:
Full  == Fixed

Value 1:
Full  == Fixed

Value 64:
Full  ≈ Fixed
Pose  完全一致
```

这说明在低幅度区间：

> **valid mask 的存在与否几乎没有改变模型响应，绝对幅度本身是主要因素。**

尤其 Pose：

```text
0 / 1 / 64
→ 都是 0.9490
```

说明低幅度第 4 通道基本不会破坏当前 Pose 表现。

### 42.2 从 128 开始，Pose 对幅度明显敏感

```text
Value 128:
Full Pose  = 0.9336
Fixed Pose = 0.9336
```

两者同时下降，进一步说明：

> **Pose 从约 128 这一幅度区间开始出现明显的“绝对值惩罚”，且此时 valid mask 还没有明显改变这种响应。**

### 42.3 在更高幅度区，valid mask 开始明显缓冲 Pose 的恶化

```text
Value 170:
Full  Pose = 0.9206
Fixed Pose = 0.9303

Value 224:
Full  Pose = 0.9129
Fixed Pose = 0.9344

Value 255:
Full  Pose = 0.9080
Fixed Pose = 0.9344
```

因此：

> **高幅度第 4 通道对 Pose 的负作用并不是完全独立于空间结构。保留 valid mask 后，高值带来的 Pose 恶化被明显缓冲。**

也就是说：

```text
绝对幅度
×
valid mask / 空间轮廓
```

之间存在明显交互。

### 42.4 Box 的交互更复杂，而且明显非单调

Box 在 128、170 时：

```text
Full > Fixed
```

但在 224、255 时：

```text
Fixed > Full
```

所以 Box 对第 4 通道的响应不能被简单总结成：

> “数值越大越好”

或者：

> “valid mask 一定提高 / 降低 Box”。

更准确的是：

> **Box 对“绝对幅度 + 空间 mask”存在明显非线性交互。**

---

## 43. 当前对 True Depth Pose 低于 Zero 的机制认识

现有实验已经支持下面这条更具体的机制链：

```text
真实 Depth
=
真实连续几何
+ valid mask
+ 较高绝对 Depth 幅度
```

其中：

### 正向部分

此前：

```text
True > Flat / Fixed
```

已经证明：

> **真实连续 Depth 几何对 Box 和 Pose 都提供正向信息。**

### 负向部分

常数扫描又证明：

> **第 4 通道的高绝对幅度会明显压低 Pose。**

而当前真实 Depth 的有效区域均值大约就在：

```text
170
```

附近。

因此目前最合理的工作假设变为：

> **Hand RGBD 模型不是没有学到 Depth，而是“真实几何带来的正收益”与“较高绝对幅度带来的负响应”同时存在。对于 Pose，后者目前更强，所以最终 True Depth 仍低于 Zero Depth。**

这比“Depth 是纯干扰”更准确。

---

## 44. 新问题：如果保留真实几何，只降低 Depth 的绝对幅度，会发生什么？

现在最关键的实验不再是继续用常数图。

因为常数图已经证明：

> 模型对幅度敏感。

下一步应该尽量保留：

```text
真实 Depth 的空间排列、valid mask、相对高低关系
```

同时降低：

```text
第 4 通道整体数值幅度
```

注意：乘法缩放会同时缩小局部 Depth 差值，因此它不是“物理几何完全不变”的严格实验；它保留的是空间模式与排序。后续如果结果显示明显收益，还需要再用匹配幅度的控制组继续拆分。

这样直接回答：

> **如果保留真实 Depth 的空间模式和相对排序，同时把整体幅度压到模型更“安全”的低幅值区间，Pose 能不能恢复甚至超过原始 True Depth？**

这一步将第一次直接检验：

> “幅度效应是否正在掩盖真实几何收益”。

---

## 45. 下一步实验：Scaled True Depth

新增三个变体：

```text
scaled_true_depth_25
scaled_true_depth_50
scaled_true_depth_75
```

定义：

对于原始有效 Depth：

```text
scaled = round(original_depth * scale)
```

其中：

```text
scale = 0.25
scale = 0.50
scale = 0.75
```

并要求：

```text
原始 invalid = 0
→ 仍保持 0

原始 valid > 0
→ 缩放后至少保持 1
```

即：

```text
valid_scaled = clip(round(depth * scale), 1, 255)
```

这样能够尽量保留：

- 当前样本自己的 valid mask；
- Depth 的空间排序；
- 相对高低关系；
- 连续几何形状；

同时系统性降低：

- 第 4 通道绝对幅度。

当前真实有效 Depth 平均值约为 170，因此三组大致会把平均幅度移到：

```text
25%  → ~42
50%  → ~85
75%  → ~128
100% → ~170（原始 True）
```

这正好覆盖常数扫描中：

```text
低幅度安全区
→
开始出现 Pose 恶化的区间
→
原始真实幅度区间
```

### 45.1 结果如何解释

如果：

```text
Scaled 25 / 50 Pose > True Pose
```

同时仍保留真实几何，则强烈支持：

> **真实几何是有用的，但原始 Depth 幅度过高产生了负作用；降低幅度可以释放几何信息的收益。**

如果：

```text
Scaled 越低，Pose 越接近 Zero
```

但始终不高于 Zero，则说明：

> **降低幅度确实减少了干扰，但当前真实几何收益仍不足以超过纯 RGB 路径。**

如果：

```text
Scaled 25 / 50 / 75 都明显差于 True
```

则说明：

> **原始 Depth 的绝对尺度本身也是模型训练时学到的重要条件，不能简单压缩。**

---

## 46. 当前实验主线（更新）

```text
第 4 通道会不会影响模型？
        ↓
确认：会

真实连续 Depth 几何有没有信息？
        ↓
确认：有正向贡献

绝对数值幅度有没有影响？
        ↓
Full Constant Sweep
        ↓
确认：非常明显

valid mask 会不会调制幅度效应？
        ↓
Fixed Valid Sweep
        ↓
确认：会
尤其在高幅度区明显缓冲 Pose 恶化

新的核心机制假设：
真实 Depth Pose 较低
=
真实几何正收益
+
高绝对幅度负作用
        ↓
下一步：
保留真实几何，只压低绝对幅度
        ↓
Scaled True Depth 25% / 50% / 75%
```

---

## 47. Scaled True Depth 实验结果

本轮继续使用同一个 Hand RGBD `best.pt`、同一个 Hand test split（12 张）和相同评估参数。

实验定义：

```text
原始 invalid = 0
→ 始终保持 0

原始 valid Depth
→ 分别乘以 0.25 / 0.50 / 0.75
→ 四舍五入并限制到 1~255
```

因此本轮尽量保留：

- 当前样本自己的 valid mask；
- Depth 的空间排列；
- 相对高低排序；
- 当前样本自己的空间模式；

同时降低第 4 通道整体数值幅度。

注意：

> 乘法缩放也会同比缩小局部 Depth 差值，因此它不是“物理几何完全不变”的实验。它保留的是空间模式和相对排序。

### 47.1 实际有效 Depth 平均值

| Variant | Scale | Valid Depth Mean |
|---|---:|---:|
| Scaled True 25% | 0.25 | 42.50 |
| Scaled True 50% | 0.50 | 84.99 |
| Scaled True 75% | 0.75 | 127.47 |
| True Depth | 1.00 | 约 169.96 |

### 47.2 模型结果

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| Zero Depth | 0.9330 | 0.9490 |
| Scaled True 25% | 0.9417 | 0.9490 |
| Scaled True 50% | 0.9387 | 0.9405 |
| Scaled True 75% | 0.9377 | 0.9377 |
| True Depth 100% | 0.9478 | 0.9349 |

精确值：

```text
Zero:
Box  = 0.933000
Pose = 0.949000

25%:
Box  = 0.941679
Pose = 0.949000

50%:
Box  = 0.9386655
Pose = 0.9405000

75%:
Box  = 0.937661
Pose = 0.9376667

100%:
Box  = 0.9477533
Pose = 0.9349091
```

---

## 48. 本轮最重要的结论：Pose 的“高幅度负响应”在真实空间模式下仍然存在

这次不再使用全图常数，而是保留真实 Depth 的空间模式。

Pose 结果：

```text
Valid Mean ≈ 0       → 0.9490
Valid Mean ≈ 42.5    → 0.9490
Valid Mean ≈ 85.0    → 0.9405
Valid Mean ≈ 127.5   → 0.9377
Valid Mean ≈ 170     → 0.9349
```

随着真实 Depth 空间模式的整体幅度逐步恢复，Pose 总体持续下降。

因此现在已经可以更有把握地说：

> **此前 Full Constant Sweep 看到的高幅度 Pose 恶化，不只是“全图常数这种异常输入”造成的。即使保留真实 Depth 的空间模式，高幅度第 4 通道仍然会对当前 Hand Pose 产生明显负响应。**

尤其：

```text
Scaled 25% Pose = Zero Pose = 0.9490
```

说明把真实 Depth 空间模式压到较低幅度后，原来 `True < Zero` 的 Pose 差距完全消失。

这强烈支持当前机制假设：

> **原始 Depth 绝对幅度正在贡献 Pose 的负响应。**

---

## 49. 但不能因此说“降低幅度后真实几何已经带来 Pose 增益”

虽然：

```text
Scaled 25% Pose = 0.9490
```

已经恢复到 Zero 水平，但还没有超过 Zero。

因此本轮只能确认：

> **降低幅度能够消除当前可测量的 Pose 劣势。**

还不能确认：

> **低幅度下真实 Depth 几何能够让 Pose 超过 Zero。**

Box 则表现不同：

```text
Zero Box       = 0.9330
Scaled 25% Box = 0.9417
```

低幅度真实空间模式下，Box 比 Zero 高约：

```text
+0.0087
```

因此对于 Box，低幅度第 4 通道仍然保留了可利用信息。

---

## 50. Box 曲线再次说明：模型响应不是简单的单调幅度函数

Box：

```text
0%   → 0.9330
25%  → 0.9417
50%  → 0.9387
75%  → 0.9377
100% → 0.9478
```

它没有随着幅度单调上升或下降。

所以 Box 更像是同时受：

- 绝对幅度；
- valid mask；
- 连续 Depth 空间几何；
- 网络内部非线性；

共同影响。

当前不应该为了 Box 峰值去调 scale。

---

## 51. 新问题：在相同幅度下，真实空间几何本身到底贡献了多少？

Scaled True Depth 同时包含：

```text
valid mask
+ 当前幅度
+ 真实空间变化
```

因此：

```text
Scaled 25% > Zero（Box）
```

仍然不能单独告诉我们：

> 这个增益究竟来自真实空间几何，还是来自“低幅度 non-zero + valid mask”。

最干净的下一步，是给每一个 Scaled True 做一个**同幅度、同 valid mask、但删除空间变化**的配对控制组。

---

## 52. 下一步实验：Matched Flat Scaled Depth

新增三组：

```text
flat_scaled_depth_25
flat_scaled_depth_50
flat_scaled_depth_75
```

对每一张图片：

1. 先按现有逻辑得到对应的 `Scaled True Depth`；
2. 保留该图片原始 valid mask；
3. 计算该图片 `Scaled True Depth` 有效区域的平均值；
4. 将该图片所有 valid 像素全部填成这个平均值；
5. invalid=0 继续严格保持 0。

即：

```text
Scaled True 25%
=
mask
+ 25% 幅度
+ 真实空间变化

Flat Scaled 25%
=
同一个 mask
+ 几乎相同的 25% 平均幅度
+ 无空间变化
```

50% 和 75% 同理。

### 52.1 这组比较为什么更干净

关键比较：

```text
Scaled True 25
vs
Flat Scaled 25

Scaled True 50
vs
Flat Scaled 50

Scaled True 75
vs
Flat Scaled 75
```

每一对都尽量匹配：

- 当前图片；
- RGB；
- label；
- valid mask；
- 平均第 4 通道幅度；

主要差别只剩：

> **有效区域内部是否保留真实 Depth 空间变化。**

因此它比拿 Scaled True 去和 Zero 或固定常数比较，更直接地回答：

> **在相同幅度条件下，真实 Depth 空间几何到底有没有独立贡献？**

### 52.2 结果如何解释

如果：

```text
Scaled True > Flat Scaled
```

则支持：

> **真实连续 Depth 空间变化在该幅度下提供正向信息。**

如果：

```text
Scaled True ≈ Flat Scaled
```

则说明：

> **该幅度下，真实空间变化没有产生可测量的独立收益，模型响应主要由幅度和 mask 决定。**

特别关注 Pose：

如果低幅度 25% 下：

```text
Scaled 25 > Flat Scaled 25
```

那么就第一次能证明：

> **降低幅度后，真实几何的正向贡献被释放出来。**

如果两者完全一致，则说明：

> **25% 能恢复 Pose，主要可能是因为降低了幅度干扰，而不是几何本身带来了额外收益。**

---

## 53. 当前实验主线（更新）

```text
真实 Depth 几何有没有被模型利用？
        ↓
True vs Flat / Fixed
        ↓
确认：有

第 4 通道绝对幅度有没有影响？
        ↓
Full Constant Sweep
        ↓
确认：有，Pose 对高幅度明显不利

这种幅度效应在真实空间模式下还存在吗？
        ↓
Scaled True 25 / 50 / 75
        ↓
确认：存在

Pose:
0 / ~42.5  → 0.9490
~85        → 0.9405
~127.5     → 0.9377
~170       → 0.9349

因此：
降低真实 Depth 幅度
可以消除当前 Pose 劣势

但新的关键问题是：
在相同幅度下，
真实空间几何本身到底贡献多少？
        ↓
下一步：
Matched Flat Scaled Depth
25% / 50% / 75%
```

---

## 54. Matched Flat Scaled Depth 配对实验结果

本轮继续使用同一个 Hand RGBD `best.pt`、同一个 Hand test split（12 张）和相同评估参数。

本轮每一对输入都尽量匹配：

- 同一张 RGB；
- 同一份 label；
- 同一个 valid mask；
- 几乎相同的 valid 区域平均 Depth 幅度；

主要差别只剩：

```text
Scaled True
→ 保留有效区域内部真实 Depth 空间变化

Flat Scaled
→ 有效区域内部全部拍平成当前图对应尺度下的平均值
```

实际平均幅度匹配情况：

| Scale | Scaled True Mean | Flat Scaled Mean |
|---:|---:|---:|
| 25% | 42.50 | 42.33 |
| 50% | 84.99 | 85.08 |
| 75% | 127.47 | 127.50 |

匹配良好，因此可以用于观察“真实空间变化”在不同幅度下的独立影响。

### 54.1 指标结果

| Scale | Variant | Box mAP50-95 | Pose mAP50-95 |
|---:|---|---:|---:|
| 25% | Scaled True | 0.941679 | 0.949000 |
| 25% | Flat Scaled | 0.941679 | 0.949000 |
| 50% | Scaled True | 0.9386655 | 0.940500 |
| 50% | Flat Scaled | 0.9304500 | 0.942200 |
| 75% | Scaled True | 0.9376610 | 0.9376667 |
| 75% | Flat Scaled | 0.9381337 | 0.9349091 |

配对差值（Scaled True - Flat Scaled）：

```text
25%:
Box  =  0.0000
Pose =  0.0000

50%:
Box  ≈ +0.0082
Pose ≈ -0.0017

75%:
Box  ≈ -0.0005
Pose ≈ +0.0028
```

---

## 55. 本轮能够确认什么

### 55.1 25% 低幅度下，真实空间变化没有可测量独立贡献

25% 时：

```text
Scaled True  = Flat Scaled
Box           0.941679
Pose          0.949000
```

两者完全一致。

因此此前：

```text
Scaled True 25% Pose = Zero Pose = 0.9490
```

现在可以进一步解释为：

> **25% 下 Pose 恢复到 Zero 水平，主要证据指向“幅度降低后负响应被解除”，而不是“真实 Depth 空间几何带来了额外 Pose 增益”。**

换句话说，在当前 12 张 Hand test 上：

> **低幅度下，真实 Depth 空间变化对 Pose 没有产生可测量的独立收益。**

### 55.2 50% 时，真实空间变化明显帮助 Box，但没有帮助 Pose

50%：

```text
Box:
Scaled True - Flat
≈ +0.0082

Pose:
Scaled True - Flat
≈ -0.0017
```

当前证据支持：

> **在约 85 的有效平均幅度下，真实 Depth 空间变化明显帮助了 Box。**

但 Pose 没有同步收益，反而小幅下降。

由于 test 只有 12 张，`-0.0017` 很小，不应过度解释成稳定的负贡献。

### 55.3 75% 时，Pose 出现小幅真实空间变化收益

75%：

```text
Pose:
Scaled True - Flat
≈ +0.0028
```

说明在约 127.5 的有效平均幅度附近：

> **真实 Depth 空间变化开始给 Pose 带来小幅正向贡献。**

Box 此时两者几乎一致：

```text
≈ -0.0005
```

不宜认为存在稳定差异。

---

## 56. 最重要的阶段性结论：几何贡献不是固定值，而是与 Depth 幅度发生交互

这轮配对实验说明：

```text
真实空间几何的作用
并不是一个固定的“+X mAP”
```

而是会随着第 4 通道幅度变化。

当前 Hand test 上大致呈现：

```text
低幅度 25%:
几何几乎无可测量独立作用

中等幅度 50%:
几何明显帮助 Box
Pose 无稳定收益

较高幅度 75%:
几何对 Pose 出现小幅正收益

原始 100%:
此前 True vs Flat/Fixed
Box 约 +0.0112
Pose 约 +0.0046
```

因此当前最准确的机制描述不是：

```text
Depth 几何有用
```

或者：

```text
Depth 幅度有害
```

二选一。

而是：

> **模型同时对 Depth 的绝对幅度、valid mask / 空间轮廓、连续 Depth 空间变化产生响应，而且这些因素之间存在明显交互。**

---

## 57. 为什么 True Depth Pose 仍然低于 Zero Depth Pose

现在已经有比较完整的证据链。

### 57.1 模型确实使用了第 4 通道

True / Zero、Spatial Shuffle 等实验已经确认：

> 第 4 通道会显著影响模型输出。

### 57.2 模型确实能利用真实连续 Depth 空间变化

True vs Flat/Fixed，以及本轮部分 scale 配对，说明：

> 真实连续 Depth 空间变化在部分幅度条件下会提供正向信息。

### 57.3 但第 4 通道较高绝对幅度会明显压低 Pose

Full Constant Sweep、Fixed Valid Sweep、Scaled True Sweep 都共同指向：

> Pose 对较高第 4 通道幅度存在明显负响应。

### 57.4 因此当前最合理的工作解释

```text
True Depth Pose
=
真实 Depth 空间信息带来的部分正收益
+
较高第 4 通道幅度带来的负响应
+
valid mask / 空间结构与幅度之间的交互
```

最终在当前 Hand test 上：

```text
True Pose = 0.9349
Zero Pose = 0.9490
```

所以：

> **不是“Depth 没学会”，而是模型学到/响应了多种第 4 通道因素；对 Pose 来说，当前这些因素合起来的净效果没有优于 Zero。**

---

## 58. 当前所有实验，用一句话分别回答了什么

```text
True vs Zero
→ 第 4 通道会不会影响模型？
→ 会。

Spatial Shuffle
→ 第 4 通道空间排列重要吗？
→ 很重要。

Blur 11 / 31 / 51
→ 轻度局部平滑会不会影响？
→ 当前几乎不影响。

Flat / Fixed
→ 连续 Depth 空间变化有没有信息？
→ 有，但作用大小受条件影响。

Flat vs Fixed
→ 当前连续帧之间的平均距离差异重要吗？
→ 当前没有可测量影响。

Full Constant Sweep
→ 模型是不是只看 0 / 非0？
→ 不是，绝对幅度非常重要。

Fixed Valid Sweep
→ valid mask 会不会调制幅度响应？
→ 会，尤其高幅度区。

Scaled True Sweep
→ 真实空间模式下，高幅度 Pose 负响应还存在吗？
→ 存在。

Matched Flat Scaled
→ 控制住幅度和 mask 后，真实空间变化还有多少独立价值？
→ 有，但不是稳定固定值；它与幅度发生明显交互。
```

---

## 59. 阶段性收束：现在先停止继续增加消融 Variant

当前已经从：

```text
“Depth 有没有用？”
```

一路拆到了：

```text
绝对幅度
×
valid mask / 空间轮廓
×
连续 Depth 空间变化
```

这个层级。

对于当前只有 12 张、而且还是连续几秒 Hand 帧的 test 集，如果继续无限增加 variant，很容易出现：

- 对很小的 mAP 差异过度解释；
- 因样本高度相似而得到不稳定结论；
- 实验树越来越复杂，但外部有效性没有增加。

因此当前阶段建议：

> **先停止新增消融 Variant，不再继续追更细的数值点。**

下一阶段应该从“继续拆模型”切换到：

> **验证这些现象能不能在更有代表性的样本上重复出现。**

优先方向不是再改 `ablate_rgbd_depth.py`，而是扩大验证范围，例如：

1. Hand 使用更分散的时间片 / 更多 test 图；
2. 在 Abdomen / Leg 的 RGBD 模型上复现最关键的少数控制实验；
3. 重点只保留一组最小核心实验：
   - True
   - Zero
   - Spatial Shuffle
   - Flat/Fixed
   - Scaled True（例如 25%）
4. 看“高幅度伤 Pose、真实几何有条件收益”是否跨区域重复。

在完成外部复现之前，不把当前 12 张 Hand test 的细小差值上升为一般规律。

---

## 60. 下一阶段：从“推理时消融”切换到“表示方式验证”

当前阶段已经回答：

- 第 4 通道会影响模型；
- 模型对 Depth 的空间排列、valid mask、连续空间变化和绝对数值幅度都会产生响应；
- 当前 Hand test 上，较高第 4 通道幅度会明显压低 Pose；
- 推理时把 True Depth 压到 25% 后，Pose 可以恢复到 Zero Depth 水平。

但必须明确：

> **这些 Scaled True 实验全部是在“已经用原始 1~255 Depth 训练好的模型”上，临时修改测试输入。**

因此它们只能证明：

> **这个已训练模型对第 4 通道数值尺度很敏感。**

它们还不能证明：

> **“把训练数据的 Depth 编码改成更低幅度后，重新训练一定会更好。”**

因为训练阶段改变输入尺度后，网络权重会重新适应。

---

## 61. 下一步先做什么：统计整个 Hand 数据集的 Depth 幅度分布

在重新训练之前，先停止继续增加推理消融 variant。

先扫描完整 Hand 数据集：

```text
train
val
test
```

统计第 4 通道有效像素 `Depth > 0` 的：

- valid pixel mean；
- p10 / p25 / p50 / p75 / p90；
- min / max；
- 每张图片 valid mean 的分布；
- invalid=0 像素比例。

目的：

> **确认当前 12 张 test 图平均 Depth ≈170，到底是整个 Hand 数据集的常态，还是只是这 12 张连续帧的特殊情况。**

如果 train / val / test 都主要集中在相似的高幅度区间，才能更有依据进入下一步“重新设计 Depth 编码尺度”。

---

## 62. 如果全数据分布确认后，再做真正关键的训练实验

下一阶段只新增一个受控训练组，不一次尝试很多方案：

```text
RGBD 原始编码：
1100~1850 → 1~255

RGBD Scale25 编码：
保持同样 raw Depth 映射关系和 valid mask，
但把有效第 4 通道整体压缩到约 25% 幅度。
```

可概念化为：

```text
原映射结果 d ∈ [1,255]

新训练输入：
d25 = clip(round(d * 0.25), 1, 64)
```

然后用完全相同的：

- train / val / test split；
- 模型结构；
- 4ch 初始化来源；
- epochs；
- batch；
- imgsz；
- seed；
- 其它训练超参数；

重新训练一个新的 RGBD Scale25 模型。

最终比较：

```text
RGB baseline
vs
原始 RGBD 1~255
vs
重新训练的 RGBD Scale25
```

这一组训练实验才真正回答：

> **当前 1~255 的 Depth 编码尺度是否是导致 Hand Pose 净收益不佳的原因之一。**

---

## 63. 为什么不能直接拿现在的 Scaled 25% 结果当训练结论

当前：

```text
Scaled True 25% Pose = 0.9490
True Depth Pose      = 0.9349
```

这是很强的线索，但不是最终训练结论。

因为：

```text
训练时：
模型见的是原始幅度

测试时：
突然把 Depth 压到 25%
```

属于输入分布被人为改变。

因此当前最严谨的表述是：

> **“原模型对较高 Depth 幅度存在不利 Pose 响应；低幅度测试输入可缓解这一现象。”**

下一阶段重新训练 Scale25 才能判断：

> **“低幅度 Depth 编码是否是一种更好的 RGBD 表示方式。”**

---

## 64. 完整 Hand train / val / test Depth 幅度分布

本轮不再做模型推理，而是直接扫描当前 Hand RGBD 数据集中真正送给网络的第 4 通道 `uint8 Depth`。

统计对象：

```text
RGBD image[..., 3]
valid   = Depth > 0
invalid = Depth == 0
```

数据量：

```text
train = 57 张
val   = 12 张
test  = 12 张
total = 81 张
```

### 64.1 Pixel-level valid Depth

| Split | Valid Pixel Mean | P25 | P50 | P75 | P90 | Invalid Ratio |
|---|---:|---:|---:|---:|---:|---:|
| train | 180.85 | 116 | 226 | 235 | 243 | 12.72% |
| val | 175.17 | 102 | 225 | 236 | 243 | 13.25% |
| test | 169.96 | 96 | 224 | 235 | 243 | 13.07% |
| overall | 178.40 | 111 | 225 | 235 | 243 | 12.85% |

### 64.2 Per-image valid mean

| Split | Mean | P25 | P50 | P75 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| train | 180.80 | 174.09 | 177.48 | 188.77 | 169.35 | 193.38 |
| val | 175.17 | 174.39 | 175.50 | 176.15 | 172.58 | 177.06 |
| test | 169.97 | 169.24 | 169.70 | 170.44 | 167.86 | 173.13 |
| overall | 178.36 | 172.82 | 175.86 | 185.52 | 167.86 | 193.38 |

---

## 65. 本轮能够确认的结论

### 65.1 “高幅度 Depth”不是 12 张 test 连续帧独有现象

虽然 test 的 per-image valid mean 大约只有：

```text
170
```

而 train 更高：

```text
约 181
```

但整个 Hand 数据集都处在明显的高幅度区间。

尤其 pixel-level 中位数：

```text
train P50 = 226
val   P50 = 225
test  P50 = 224
```

P75 / P90 更几乎完全一致：

```text
P75 ≈ 235~236
P90 = 243
```

因此：

> **当前第 4 通道大量有效像素长期处于约 224~243 的高值区域，不是 test 12 张图的偶然现象。**

这使得此前“高幅度可能影响 Pose”的发现具有继续做训练验证的价值。

### 65.2 test 确实比 train / val 整体更近一些

per-image valid mean：

```text
train ≈ 180.8
val   ≈ 175.2
test  ≈ 170.0
```

因此不能写成：

> “test≈170 就是整份 Hand 数据集完全相同的常态。”

更准确的是：

> **test 的整体 Depth 水平比 train 偏低，但 train / val / test 全部仍处于同一个较高幅度 regime。**

这也提示当前 split 存在一定距离分布差异，后续解释结果时需要保留这一限制。

### 65.3 invalid 比例非常稳定

```text
train 12.72%
val   13.25%
test  13.07%
```

说明三个 split 的 `Depth==0` 比例接近。

因此当前最显著的 split 差异主要不是 valid mask 总面积，而是：

> **有效 Depth 的整体数值水平。**

---

## 66. 为什么现在值得进入“重新训练 Scale25”阶段

前面的推理时 Scaled True 实验已经发现：

```text
True 100%    Pose = 0.9349
Scaled 75%   Pose = 0.9377
Scaled 50%   Pose = 0.9405
Scaled 25%   Pose = 0.9490
Zero         Pose = 0.9490
```

但那是在一个已经用原始 1~255 Depth 训练完成的模型上临时修改测试输入，因此只能证明：

> **现有模型对第 4 通道幅度敏感。**

现在完整数据分布又确认：

> **高幅度不是 test 特例，而是整个 Hand 数据集普遍存在。**

所以接下来可以进行真正有意义的训练对照：

```text
原始 RGBD：
Depth 映射结果保持 1~255

Scale25 RGBD：
在已经映射好的第 4 通道上，
valid Depth × 0.25，
并保持 invalid=0
```

然后从同一个 4ch 初始化权重重新训练。

这样才真正检验：

> **把训练阶段的 Depth 数值尺度整体降低，能不能改善最终 RGBD Pose。**

---

## 67. 下一步执行顺序

现在不直接启动训练。

先生成一个完全独立的新数据集：

```text
hand/rgbd_rawdepth_scale25
```

要求：

```text
RGB           完全不变
label         完全不变
train/val/test split 完全不变
invalid=0     完全不变
valid Depth   round(depth * 0.25)，并限制到 1~64
```

生成后先用现有：

```text
analyze_rgbd_depth_distribution.py
```

重新统计新数据集。

预期大致：

```text
train per-image mean ≈ 45
val   per-image mean ≈ 44
test  per-image mean ≈ 42
```

以及 pixel median：

```text
原始约 224~226
→ Scale25 后约 56
```

只有这些统计确认无误后，再进入新的 RGBD Scale25 训练。

当前阶段主线：

```text
推理消融
→ 找到幅度敏感性
→ 全数据分布统计
→ 确认高幅度是全数据集现象
→ 生成 Scale25 训练数据
→ 先验证数据
→ 再重新训练
```

---

## 68. Scale25 Hand RGBD 数据集生成与复核

已从原始：

```text
hand/rgbd_rawdepth
```

生成独立数据集：

```text
hand/rgbd_rawdepth_scale25
```

生成规则：

```text
invalid Depth == 0
→ 保持 0

valid Depth > 0
→ round(depth * 0.25)
→ clip 到 1~255
```

本轮实际最大有效值为 64。

生成脚本报告：

```text
train:
57 images / 57 labels
source valid mean = 180.8489
scale25 mean      = 45.2182
source min/max    = 1/255
scale25 min/max   = 1/64
invalid ratio     = 12.7237%
valid mask        = consistent
RGB               = consistent

val:
12 images / 12 labels
source valid mean = 175.1674
scale25 mean      = 43.7997
source min/max    = 24/255
scale25 min/max   = 6/64
invalid ratio     = 13.2456%
valid mask        = consistent
RGB               = consistent

test:
12 images / 12 labels
source valid mean = 169.9634
scale25 mean      = 42.4982
source min/max    = 4/255
scale25 min/max   = 1/64
invalid ratio     = 13.0691%
valid mask        = consistent
RGB               = consistent
```

---

## 69. Scale25 分布统计复核

使用独立的 Depth distribution analyzer 再次扫描生成后的完整 train / val / test。

### 69.1 Pixel-level valid Depth

| Split | 原始 Mean | Scale25 Mean | Scale25 P25 | P50 | P75 | P90 |
|---|---:|---:|---:|---:|---:|---:|
| train | 180.85 | 45.22 | 29 | 56 | 59 | 61 |
| val | 175.17 | 43.80 | 26 | 56 | 59 | 61 |
| test | 169.96 | 42.50 | 24 | 56 | 59 | 61 |
| overall | 178.40 | 44.61 | 28 | 56 | 59 | 61 |

原始 pixel median：

```text
train 226
val   225
test  224
```

Scale25 后统一约：

```text
56
```

与预期一致。

### 69.2 Per-image valid mean

```text
train mean = 45.21
val   mean = 43.80
test  mean = 42.50
```

整体：

```text
44.60
```

### 69.3 invalid ratio 完全保持原分布

```text
train 12.7237%
val   13.2456%
test  13.0691%
```

与原始 RGBD 数据集相同。

因此目前能够确认：

> **Scale25 数据集已经成功把第 4 通道整体幅度压到原来的约 1/4，同时保持 RGB、split 和 valid/invalid 空间结构。**

数据层面的目标已达到，可以进入真正的重新训练实验。

---

## 70. 下一阶段：重新训练 RGBD Scale25

这是前面所有推理消融之后，第一个真正改变“训练阶段 Depth 表示方式”的受控实验。

需要特别区分：

```text
之前：
原始 RGBD 模型训练完成
→ 只在 test 时临时 Scale25

现在：
train / val / test 从一开始全部使用 Scale25
→ 网络重新学习适应这个 Depth 数值尺度
```

本轮只改变：

```text
第 4 通道编码幅度
```

其余保持与原始 raw-depth Hand RGBD run 一致：

```text
初始化权重：
YJJ_Pose_Scripts/weights/4ch/yolov8n-pose_4ch.pt

epochs       = 120
imgsz        = 640
batch        = 4
fliplr       = 0.0
patience     = 20
seed         = 0（Ultralytics 当前原始 run 配置）
workers      = 0
amp          = False
mosaic       = 0
mixup        = 0
copy_paste   = 0
lr0          = 0.0005
warmup_epochs= 10
以及现有 train_rgbd_pose.py 的其它训练设置保持不变
```

原始 Hand raw-depth RGBD run 使用的命令参数已经记录为：

```text
--fliplr 0.0
--patience 20
--weights YJJ_Pose_Scripts/weights/4ch/yolov8n-pose_4ch.pt
```

因此 Scale25 训练不允许从原始 RGBD `best.pt` 接着训练。

必须重新从相同的：

```text
yolov8n-pose_4ch.pt
```

开始。

### 70.1 训练完成后真正要比较的三组

```text
RGB baseline
Box  mAP50-95 = 0.9723
Pose mAP50-95 = 0.8958

原始 Raw-depth RGBD
Box  mAP50-95 = 0.9478
Pose mAP50-95 = 0.9349

新的 RGBD Scale25
Box  mAP50-95 = ?
Pose mAP50-95 = ?
```

这里最重要的问题不是要求 Scale25 必须获胜，而是：

> **当训练阶段也使用低幅度 Depth 表示后，之前观察到的 Pose 幅度问题是否仍然存在，以及 RGBD Pose 的最终净效果如何变化。**

如果 Scale25 Pose 明显高于原始 RGBD：

> 支持“Depth 编码尺度是当前模型行为的重要因素”。

如果 Scale25 与原始 RGBD 接近：

> 说明模型经过重新训练后可以适应不同输入幅度，推理时的幅度敏感性不能直接转化为训练方案收益。

如果 Scale25 更差：

> 说明原始高幅度尺度可能本身也是训练过程中模型利用的重要条件，不能依据推理消融直接缩放训练表示。

---

## 71. RGBD Scale25 重新训练完成

本轮通过现有 GUI 启动训练，训练命令确认使用：

```text
data:
hand/rgbd_rawdepth_scale25/data_rgbd_scale25.yaml

weights:
YJJ_Pose_Scripts/weights/4ch/yolov8n-pose_4ch.pt

fliplr = 0.0
patience = 20
epochs = 120
batch = 4
imgsz = 640
seed = 0
resume = False
```

第一层卷积确认：

```text
in_channels = 4
```

因此本轮确实是：

```text
同一个干净 4ch 初始化权重
+
Scale25 数据集
```

而不是从原始 RGBD `best.pt` 继续训练。

### 71.0 一个需要记录的训练配置细节

日志显示：

```text
optimizer=auto
```

因此 Ultralytics 实际忽略了脚本里传入的：

```text
lr0=0.0005
momentum=0.937
```

并自动选择：

```text
AdamW(lr=0.002, momentum=0.9)
```

这一点此前容易被“脚本参数”误认为实际生效参数。

不过原始 raw-depth RGBD 训练使用的也是同一训练脚本/同一 `optimizer=auto` 行为，所以当前 Scale25 与原始 RGBD 的对照仍保持一致。后续记录实验参数时，应以 trainer 日志中的实际 optimizer 参数为准。

输出目录：

```text
runs/pose/train_pose_rgbd_4ch3
```

### 71.1 训练停止情况

EarlyStopping：

```text
Best results observed at epoch 37
patience = 20
最终在 epoch 57 停止
```

因此本轮不是训练失败，而是正常早停。

最终保存：

```text
runs/pose/train_pose_rgbd_4ch3/weights/best.pt
runs/pose/train_pose_rgbd_4ch3/weights/last.pt
```

### 71.2 best.pt 在 val split 上的最终验证

Ultralytics 自动对 `best.pt` 做的 validation：

```text
Images = 12
Instances = 12

Box:
P          = 1.000
R          = 1.000
mAP50      = 0.995
mAP50-95   = 0.983

Pose:
P          = 1.000
R          = 1.000
mAP50      = 0.995
mAP50-95   = 0.685
```

注意：

> **这里是训练过程使用的 val split，不是最终 test split。**

因此现在不能拿：

```text
Box 0.983 / Pose 0.685
```

直接与此前：

```text
RGB baseline test
Raw-depth RGBD test
```

比较。

真正的科学比较仍必须使用同一个 `test` split。

---

## 72. 当前下一步：只做 Scale25 best.pt 的 test 评估

下一步不改代码、不增加新的消融变体，也不重新训练。

只需要使用：

```text
weights:
runs/pose/train_pose_rgbd_4ch3/weights/best.pt

data:
hand/rgbd_rawdepth_scale25/data_rgbd_scale25.yaml

split:
test
```

得到最终：

```text
Scale25 Box mAP50-95
Scale25 Pose mAP50-95
```

然后与此前固定基线比较：

```text
RGB baseline:
Box  = 0.9723
Pose = 0.8958

原始 Raw-depth RGBD:
Box  = 0.9478
Pose = 0.9349

RGBD Scale25:
Box  = ?
Pose = ?
```

这一轮 test 结果才真正回答：

> **训练阶段把 Depth 编码整体压到 25% 后，最终 RGBD 模型相对于原始 RGBD 是否产生了净改善。**

在 test 结果出来之前，不根据 val 指标提前下结论。

---

## 73. RGBD Scale25 重新训练后的正式 test 结果

本轮使用：

```text
weights:
runs/pose/train_pose_rgbd_4ch3/weights/best.pt

data:
hand/rgbd_rawdepth_scale25/data_rgbd_scale25.yaml

split:
test
```

固定在与此前实验相同的 12 张 Hand test 上评估。

结果：

```text
Box:
P          = 1.0000
R          = 1.0000
mAP50      = 0.9950
mAP50-95   = 0.9464373

Pose:
P          = 1.0000
R          = 1.0000
mAP50      = 0.9950
mAP50-95   = 0.9375000
```

---

## 74. 三组最终核心对照

| Model / Input | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| RGB baseline | 0.9723 | 0.8958 |
| Raw-depth RGBD | 0.9478 | 0.9349 |
| Scale25 RGBD（重新训练） | 0.9464 | 0.9375 |

Scale25 相对原始 Raw-depth RGBD：

```text
Box:
0.9464373 - 0.9477533
≈ -0.00132

Pose:
0.9375000 - 0.9349091
≈ +0.00259
```

Scale25 相对 RGB baseline：

```text
Box:
0.9464373 - 0.9723
≈ -0.02586

Pose:
0.9375000 - 0.8958
≈ +0.04170
```

---

## 75. 本轮最重要的结论：推理时的幅度敏感性，没有转化成明显的重新训练收益

此前在“原始 RGBD 模型”上临时缩小 test Depth：

```text
True 100% Pose = 0.9349
Scaled 25% Pose = 0.9490
```

当时看起来：

> 把 Depth 幅度降到 25% 可以明显缓解 Pose 的负响应。

但是现在真正从头使用 Scale25 数据重新训练以后：

```text
Scale25 retrained Pose = 0.9375
Raw RGBD Pose          = 0.9349
```

只提高约：

```text
+0.0026
```

同时 Box 还下降约：

```text
-0.0013
```

因此当前最合理的解释是：

> **模型确实对推理输入的第 4 通道尺度很敏感，但训练阶段会重新适应新的 Depth 数值尺度。**

所以不能把之前的推理消融结论直接翻译成：

> “训练时把 Depth 压到 25% 就会明显提高 Pose。”

当前数据并不支持这个强结论。

更准确地说：

> **Depth 编码幅度是模型行为的重要因素，但单纯把整个训练数据的 Depth 统一乘 0.25，并没有带来明显的最终净收益。**

---

## 76. Scale25 结果不能被解释成“完全没作用”

虽然 Scale25 相比原始 RGBD 的变化很小，但它仍然说明：

1. 把训练 Depth 从 `1~255` 压到 `1~64` 后，模型仍然能够正常学习；
2. RGBD Pose 优于 RGB baseline 的现象仍然存在；
3. RGBD 对 Box 的损失也仍然存在；
4. 当前主要问题不是简单由“Depth 数值太大”单独决定。

尤其三组核心对照仍然是：

```text
RGB:
Box 高
Pose 低

Raw RGBD:
Box 下降
Pose 明显提高

Scale25 RGBD:
Box 与 Raw RGBD 几乎相同
Pose 也与 Raw RGBD 几乎相同
```

这意味着：

> **RGBD 与 RGB 的差异，更可能来自“加入 Depth 这一模态后网络学习到的整体表示”，而不是某一个固定的 uint8 幅度选择。**

---

## 77. 当前不能确认的东西

当前只有：

- 81 张 Hand 数据；
- 12 张 test；
- 原始 RGBD 一次训练；
- Scale25 一次训练；
- 单个 seed。

因此：

```text
Pose +0.0026
```

不能被当成稳定提升。

在当前 test 规模下，更合理的描述是：

> **Scale25 与原始 RGBD 基本处于同一水平，尚无证据证明 Scale25 是更好的训练表示。**

同样，也不能据此证明所有 Depth scaling 都没意义；这里只验证了一个简单的全局 `×0.25` 编码。

---

## 78. 下一步最有解释力的实验：Cross-Scale Test

现在不需要继续训练新的 Scale50 / Scale75。

先利用已经存在的两个模型做一个最便宜、最直接的交叉测试：

```text
模型 A：Raw-depth RGBD 训练
模型 B：Scale25 RGBD 训练

输入 1：Raw-depth test
输入 2：Scale25 test
```

目前已有三个格子：

```text
Raw-trained + Raw test
Pose = 0.9349

Raw-trained + Scale25 test
Pose = 0.9490
（此前 Scaled True 25% 推理消融，等价控制）

Scale25-trained + Scale25 test
Pose = 0.9375
```

只缺最后一个：

```text
Scale25-trained + Raw test
Pose = ?
```

### 78.1 为什么这个格子重要

如果：

```text
Scale25-trained + Raw
明显差于
Scale25-trained + Scale25
```

说明：

> 模型重新训练后确实适应了 Scale25 输入尺度，跨尺度会产生分布失配。

如果：

```text
Scale25-trained + Raw
反而接近或优于 Scale25 test
```

则说明：

> 模型对幅度的响应更复杂，不能简单解释成“训练尺度匹配最好”。

这个 Cross-Scale Test 不需要改代码、不需要重新训练，只需要把 Scale25 模型放到原始 raw-depth test YAML 上测试一次。

---

## 79. 当前阶段主线

```text
推理消融：
发现原始模型对 Depth 幅度敏感
        ↓
完整数据分布：
确认高幅度是整个 Hand 数据集普遍现象
        ↓
生成 Scale25 数据
        ↓
重新训练 Scale25 模型
        ↓
正式 test：
Raw RGBD Pose    0.9349
Scale25 RGBD Pose 0.9375
        ↓
结论：
训练会重新适应尺度，
单纯 Scale25 没有明显净收益
        ↓
下一步：
只补一个 Cross-Scale Test
Scale25-trained + Raw-depth test
```

---

## 80. Cross-Scale Test：最后一个交叉格子补齐

本轮使用：

```text
模型：
Scale25 数据重新训练得到的
runs/pose/train_pose_rgbd_4ch3/weights/best.pt

测试输入：
原始 hand/rgbd_rawdepth test

split:
test
```

正式结果：

```text
Box mAP50-95  = 0.941679
Pose mAP50-95 = 0.9348333333
```

因此 2×2 Cross-Scale Test 已完整：

| Train Scale | Test Raw | Test Scale25 |
|---|---:|---:|
| Raw-trained Box | 0.9478 | 0.9417 |
| Scale25-trained Box | 0.9417 | 0.9464 |
| Raw-trained Pose | 0.9349 | 0.9490 |
| Scale25-trained Pose | 0.9348 | 0.9375 |

其中：

```text
Raw-trained + Raw test
Box  = 0.9477533
Pose = 0.9349091

Raw-trained + Scale25 test
Box  = 0.9416790
Pose = 0.9490000

Scale25-trained + Raw test
Box  = 0.9416790
Pose = 0.9348333

Scale25-trained + Scale25 test
Box  = 0.9464373
Pose = 0.9375000
```

---

## 81. Cross-Scale Test 能确认什么

### 81.1 Box 明显更偏好“训练尺度与测试尺度匹配”

对于 Raw-trained：

```text
Raw test    = 0.9478
Scale25 test= 0.9417
```

对于 Scale25-trained：

```text
Scale25 test= 0.9464
Raw test    = 0.9417
```

两种模型都表现出：

> **训练时使用什么 Depth 尺度，Box 在匹配该尺度的 test 输入上更好。**

这支持：

> **网络在训练过程中确实会适应第 4 通道的数值尺度。**

注意两种 cross-scale Box 都恰好为 `0.941679`，但 test 只有 12 张，不应把这个精确相等解释成更深层规律；更重要的是“匹配尺度 > 不匹配尺度”的方向在两边一致。

---

### 81.2 Pose 的行为不是对称的尺度匹配问题

Scale25-trained：

```text
Scale25 test = 0.9375
Raw test     = 0.9348
```

匹配 Scale25 输入时略好：

```text
约 +0.0027
```

这与“重新训练后模型适应 Scale25”一致。

但是 Raw-trained：

```text
Raw test     = 0.9349
Scale25 test = 0.9490
```

反而在不匹配的低幅度输入上显著更高。

因此：

> **Pose 不能简单解释成“训练尺度匹配就最好”。**

更准确的是：

> **原始 Raw-trained 模型对测试时降低第 4 通道幅度存在一种特殊的有利响应；但当网络从头在 Scale25 数据上重新训练后，这个大幅提升并没有保留下来。**

也就是说：

```text
Raw-trained + 临时 Scale25
Pose = 0.9490
```

是“已训练模型面对输入尺度扰动时的响应”，而不是一个可以直接通过 Scale25 训练复制出来的稳定训练收益。

---

## 82. 目前关于“Depth 幅度”的最终阶段性判断

现在可以把前面大量实验收束成三句话：

### 82.1 第 4 通道幅度确实会影响模型

Full Constant、Fixed Valid、Scaled True 等实验已经反复确认：

> **第 4 通道绝对数值尺度不是无关变量。**

### 82.2 训练阶段会重新适应这个尺度

Cross-Scale Box 结果尤其清楚：

```text
Raw-trained 更适合 Raw test
Scale25-trained 更适合 Scale25 test
```

因此：

> **不能把推理时缩放 Depth 的表现，直接当成一种新的训练编码方案的预期收益。**

### 82.3 单纯 Scale25 不是当前 Hand RGBD 的明确改进方案

正式同尺度 test：

```text
Raw RGBD:
Box  = 0.9478
Pose = 0.9349

Scale25 RGBD:
Box  = 0.9464
Pose = 0.9375
```

差值：

```text
Box  ≈ -0.0013
Pose ≈ +0.0026
```

在只有 12 张 test、单次训练、单个 seed 的条件下：

> **应视为基本同一水平。**

因此当前没有足够证据支持：

> “把 Depth 从 1~255 压成 1~64 是更好的 Hand RGBD 训练表示。”

---

## 83. 当前最合理的总体解释

现有证据最支持下面这个模型：

```text
第 4 通道作用
=
真实 Depth 空间信息
+
valid mask / 空间轮廓
+
绝对数值尺度
+
这些因素与网络训练过程之间的适应和交互
```

所以：

> **Depth 不是“没学会”，也不能简单归结为“幅度太大”。**

更准确的是：

> **网络确实在利用第 4 通道，但它对 Depth 表示方式本身也很敏感；训练会适应这种表示，因此推理消融发现的幅度效应不等价于重新设计训练编码后一定能获得收益。**

这就是当前 Hand 实验能够支持到的最稳妥结论。

---

## 84. 当前阶段停止继续做 Hand 的尺度消融

到这里：

- 推理消融已经足够多；
- Scale25 数据已经真正重新训练；
- Cross-Scale 2×2 已补齐；
- 主要机制已经能够解释；
- 当前 test 只有 12 张连续帧。

因此不再继续增加：

```text
Scale10
Scale50
Scale75
更多常数点
更多 Hand variant
```

下一阶段如果继续研究，应优先验证：

> **这些现象能否在更多、更分散的 Hand 数据，或者 Abdomen / Leg 上重复。**

否则继续在当前 12 张 test 上追 `0.001~0.003` 的差异，科学价值已经很低。

---

## 85. 当前最终核心结果表

| 实验 | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| RGB baseline | 0.9723 | 0.8958 |
| Raw RGBD trained + Raw test | 0.9478 | 0.9349 |
| Raw RGBD trained + Scale25 test | 0.9417 | 0.9490 |
| Scale25 RGBD trained + Scale25 test | 0.9464 | 0.9375 |
| Scale25 RGBD trained + Raw test | 0.9417 | 0.9348 |

阶段性总结：

```text
RGB → Box 最好，Pose 最低

加入真实 Depth → Pose 明显提高，Box 略下降

推理时临时降低 Depth 幅度
→ Raw-trained Pose 可进一步提高

但从头用 Scale25 重新训练
→ 最终与 Raw RGBD 基本同水平

Cross-Scale
→ Box 明显表现出尺度适应
→ Pose 存在更复杂、非对称的幅度响应
```
