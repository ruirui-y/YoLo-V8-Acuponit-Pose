# YOLOv8 RGB-D Acupoint Pose

> 基于 YOLOv8 Pose 的 RGB / RGB-D 手部穴位关键点检测与跨受试者泛化实验项目。

GitHub：<https://github.com/ruirui-y/YoLo-V8-Acuponit-Pose.git>

---

## 1. 项目简介

本项目围绕 **YOLOv8 Pose + RGB-D** 展开，目标是研究深度信息是否能够提升手部穴位关键点检测的稳定性、定位能力与跨受试者泛化能力。

当前项目包含：

- RGB 3 通道 Pose 训练 / 测试；
- RGB-D 4 通道 Pose 训练 / 测试；
- RGB → RGB-D 4 通道模型初始化；
- Depth channel 消融实验；
- LaMa 辅助图像处理与标注流程；
- Pose label 点序审计；
- Cross-subject Test Set 重建；
- Canonical keypoint reorder；
- RGB / RGB-D 同步数据集生成；
- GUI 数据集准备、训练、评估与对比。

当前重点不是单纯追求某一次最高 mAP，而是保证：

1. RGB 与 RGB-D 使用完全一致的数据划分；
2. 不发生 subject leakage；
3. 关键点 ID 语义一致；
4. test set 可复现；
5. 深度通道是否真正被模型使用可以通过消融验证；
6. 结果能够区分「同主体拟合」与「跨主体泛化」。

---

## 2. 当前研究问题

项目主要回答以下问题：

### 2.1 RGB-D 是否优于 RGB？

比较：

- Box mAP50 / mAP50-95；
- Pose mAP50 / mAP50-95；
- Precision / Recall；
- In-domain 与 Cross-subject 两种测试条件。

### 2.2 模型是否真正使用了 Depth？

通过对第四通道进行：

- True Depth；
- Zero Depth；
- Cross-image Shuffled Depth；
- Spatial-shuffled Depth；

等干预，观察检测与关键点指标变化。

### 2.3 RGB-D 是否具有更好的跨受试者泛化？

训练集与验证集来自 Subject A，测试集全部来自其他受试者，禁止随机混入 train / val。

---

## 3. 项目目录概览

主要代码位于：

```text
YJJ_Pose_Scripts/
├─ 01_data/        # 数据集生成、Cross-subject rebuild、label reorder
├─ 02_model/       # 模型相关
├─ 03_train/       # 训练脚本
├─ 04_eval/        # 测试、消融、可视化
├─ 05_infer/       # 推理
├─ 06_debug/       # label audit / 调试工具
├─ gui/            # PySide6 GUI
└─ weights/        # 权重相关
```

当前关键脚本：

```text
01_data/prepare_pose_rgbd_dataset.py
01_data/rebuild_cross_subject_testset.py
01_data/pose_label_reorder.py

04_eval/test_pose_rgbd.py
04_eval/ablate_rgbd_depth.py
04_eval/visualize_rgbd_depth_ablation.py

06_debug/check_pose_label_order.py
06_debug/reorder_pose_labels_by_template.py
```

---

## 4. RGB / RGB-D 数据结构

标准数据集结构：

```text
dataset/
└─ hand/
   ├─ rgb/
   │  ├─ images/
   │  │  ├─ train/
   │  │  ├─ val/
   │  │  └─ test/
   │  ├─ labels/
   │  │  ├─ train/
   │  │  ├─ val/
   │  │  └─ test/
   │  └─ data_rgb.yaml
   │
   └─ rgbd/
      ├─ images/
      │  ├─ train/
      │  ├─ val/
      │  └─ test/
      ├─ labels/
      │  ├─ train/
      │  ├─ val/
      │  └─ test/
      └─ data_rgbd.yaml
```

要求：

```text
RGB train stems  == RGBD train stems
RGB val stems    == RGBD val stems
RGB test stems   == RGBD test stems
RGB labels       == RGBD labels
```

RGB 与 RGB-D 的唯一区别应当是输入通道，不应该由数据划分差异制造结果差异。

---

## 5. RGB-D 4 通道方式

RGB-D 输入使用 4 通道：

```text
R
G
B
Depth
```

第一层卷积由官方 3 通道权重扩展为 4 通道。

Depth 通道使用项目既定的距离映射规则生成，训练与测试必须保持相同的 low / high 参数与处理流程。

当前实验常用：

```text
Depth low  = 1100 mm
Depth high = 1850 mm
```

> 不要直接 `pip upgrade ultralytics` 覆盖项目所使用的 vendored / 定制 Ultralytics 代码。

---

## 6. 关键点标签与 Canonical ID

### 6.1 曾发现的严重问题

早期数据集中存在关键点 ID 顺序错误。

旧 81 张数据审计曾发现：

```text
valid      = 81
suspects   = 24
parseError = 0
```

其中主要表现为：

```text
K2 <-> K3
```

以及少量：

```text
K5 <-> K6
```

这类错误不是关键点坐标本身不存在，而是：

> 坐标位置正确，但对应的 K ID 语义错误。

修正标注链路后重新审计：

```text
valid      = 75
suspects   = 0
parseError = 0
```

因此，早期被点序污染的数据与结果不应作为最终论文主结果。

---

## 7. 当前关键点 ID 原则

### 7.1 最终推荐规则

未来每一个 Subject / 每一组数据都采用：

```text
第一张 Reference
    ↓
人工明确指定 K0、K1、K2 ... K(N-1)
    ↓
锁定该组 canonical ID
    ↓
后续帧只做 Stable-ID 继承
    ↓
禁止重新按 x/y 自动定义语义编号
```

也就是说：

> **Canonical ID 应由人工语义定义，而不是由几何排序自动决定。**

如果有 10 组数据：

```text
Group 1：第一张人工定义 K0~K6 → 后续全部继承
Group 2：第一张人工定义 K0~K6 → 后续全部继承
...
Group 10：第一张人工定义 K0~K6 → 后续全部继承
```

这是当前认为最稳妥的标注策略。

---

## 8. Canonical reorder 的定位

项目当前仍保留自动 canonical reorder，用于：

- 历史数据修复；
- 外部已有标签与训练体系对齐；
- 点序一致性审计；
- Cross-subject Test Set 重建时的兜底。

其核心逻辑位于：

```text
01_data/pose_label_reorder.py
```

独立 CLI：

```text
06_debug/reorder_pose_labels_by_template.py
```

当前实现使用：

```text
Shape Signature
    +
Hungarian Assignment
    +
Similarity ICP
    +
Ambiguity Guard
```

并使用 base train labels 作为 Reference Bank。

但这套算法的定位应当是：

> **历史兼容 / audit / repair 工具，而不是未来新数据的首选 ID 定义方法。**

---

## 9. Cross-subject Test Set

Cross-subject 数据集结构：

```text
dataset/
└─ hand_cross_subject_v2/
   ├─ rgb/
   └─ rgbd/
```

当前最终 Cross-subject 数据规模：

```text
Train : 52
Val   : 11
Test  : 101
Kpts  : 7
```

其中：

- train / val 保留原 Subject A；
- test 全部来自新受试者；
- external 样本全部进入 test；
- 不随机 split；
- RGB / RGB-D test 完全同步。

---

## 10. Cross-subject rebuild 流程

GUI 中的 `Cross-subject Test Set` 区域需要：

```text
External Images
External Labels
External Depth NPY
Output Dataset Name
Canonical Template Label
```

### Validate

Validate 会检查：

```text
image 数量
label 数量
depth npy 数量
完整时间戳匹配
重复 timestamp
缺失 label
缺失 npy
keypoint 数量
canonical template
reference bank
canonical reorder audit
```

时间戳匹配形式：

```text
color_<timestamp>.jpg
color_<timestamp>.txt
depth_<timestamp>.npy
```

identity 使用完整 timestamp，不依赖目录顺序。

### Rebuild

流程：

```text
base train / val
        ↓ 原样复制

external RGB + label + depth
        ↓ 全部进入 test

staging:
.prepare_cross_subject_tmp
        ↓
canonical reorder
        ↓
RGB/RGBD label byte-for-byte sync
        ↓
完整一致性检查
        ↓
publish
```

正式目标已存在时：

```text
BLOCK
```

不会静默覆盖。

---

## 11. Canonical reorder audit

当前真实 101 张 Cross-subject 数据曾完整执行 audit。

结果：

```text
Total samples              : 101
Best/second mapping same   : 101
Best/second mapping diff   : 0

Median residual            : 0.139478
P95 residual               : 0.155488
Max residual               : 0.164199
```

101 张得到的主 mapping 完全一致：

```text
K0 <- K1
K1 <- K0
K2 <- K2
K3 <- K4
K4 <- K3
K5 <- K5
K6 <- K6
```

这说明该 external subject 的标签内部并不是随机错乱，而是整组使用了另一套固定编号方式。

修正后 Cross-subject Pose 评估从原先的全 0 恢复到正常范围，确认早先的：

```text
Pose P/R/mAP = 0
```

主要来自关键点 ID 语义不一致，而不是模型完全失效。

---

# 12. 当前最终实验结果

## 12.1 修正后的 In-domain Baseline

当前 75 张修正数据集：

```text
Train : 52
Val   : 11
Test  : 12
```

### RGB

| Metric | Box | Pose |
|---|---:|---:|
| P | 0.9958 | 0.9958 |
| R | 1.0000 | 1.0000 |
| mAP50 | 0.9950 | 0.9950 |
| mAP50-95 | **0.9473** | **0.9950** |

### RGB-D

| Metric | Box | Pose |
|---|---:|---:|
| P | 1.0000 | 1.0000 |
| R | 1.0000 | 1.0000 |
| mAP50 | 0.9950 | 0.9950 |
| mAP50-95 | **0.9612** | **0.9882** |

差值：

```text
RGBD - RGB

Box  mAP50-95 : +0.0138
Pose mAP50-95 : -0.0068
```

当前 In-domain Pose 已接近 ceiling，不能仅凭这 12 张 test 宣称 RGB-D 对 Pose 普遍更优。

---

## 12.2 Corrected Core-4 Depth Ablation

使用修正后的 RGB-D 模型：

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | **0.9612** | **0.9882** |
| Zero Depth | 0.8955 | 0.9950 |
| Cross-image Shuffled | 0.9431 | 0.9868 |
| Spatial-shuffled | **0.3026** | **0.1348** |

关键观察：

### True vs Zero

```text
Box:
0.9612 -> 0.8955
下降约 6.57 个百分点
```

说明第四通道并非完全被网络忽略。

### Spatial Shuffle

```text
Box  : 0.9612 -> 0.3026
Pose : 0.9882 -> 0.1348
```

说明模型强烈依赖 Depth 的空间组织。

但需要注意：

> Spatial shuffle 是非常强的 OOD 干预，因此它可以证明模型使用了深度空间结构，但不能单独等价为“证明模型学到了通用 3D 几何”。

---

## 12.3 最终 Cross-subject v2

测试集：

```text
101 张
来自训练阶段未见的新受试者
```

### RGB

| Metric | Box | Pose |
|---|---:|---:|
| P | 0.8674 | 0.9706 |
| R | 0.8911 | 0.9818 |
| mAP50 | 0.9177 | **0.9945** |
| mAP50-95 | **0.5373** | **0.7608** |

### RGB-D

| Metric | Box | Pose |
|---|---:|---:|
| P | 0.8730 | 0.9092 |
| R | **0.9531** | **0.9911** |
| mAP50 | **0.9497** | 0.9624 |
| mAP50-95 | 0.4808 | 0.6868 |

### RGBD - RGB

```text
Box  mAP50-95 : -0.0565
Pose mAP50-95 : -0.0740
```

当前跨受试者测试中：

> **RGB 在严格的 mAP50-95 上优于 RGB-D。**

但 RGB-D 也表现出：

```text
Box Recall:
0.9531 > 0.8911

Box mAP50:
0.9497 > 0.9177
```

因此目前更准确的结论是：

> RGB-D 在当前 Cross-subject 数据上提升了 Box recall 与较宽松 IoU 下的检测能力，但没有提升严格 Box mAP50-95，也没有提升 Pose mAP50-95。

这部分结果应如实报告，不应为了“RGB-D 必须更好”而筛选数据或改变评估规则。

---

# 13. 当前实验结论

截至目前，项目已经得到三类相对明确的证据。

### 13.1 深度通道确实被使用

True → Zero 的 Box 下降，以及 Spatial Shuffle 的灾难性下降，都说明：

```text
第四通道不是无效输入
```

### 13.2 In-domain Pose 已出现 ceiling effect

当前 12 张同主体 test 上：

```text
RGB Pose mAP50-95  = 0.9950
RGBD Pose mAP50-95 = 0.9882
```

因此继续在这 12 张上堆叠大量 ablation，对证明泛化优势意义有限。

### 13.3 当前 Cross-subject 中 RGB 更强

修正关键点 ID 后：

```text
RGB Pose mAP50-95  = 0.7608
RGBD Pose mAP50-95 = 0.6868
```

因此目前没有证据支持：

> “RGB-D 在所有条件下都优于 RGB。”

当前更合理的科研表述是：

> RGB-D 模型确实使用了深度通道，并在部分检测指标上展现收益；但在当前跨受试者严格定位与关键点 mAP50-95 指标上，RGB 基线仍然更强。

---

# 14. 当前完整测试过程回顾

项目当前实验演进如下：

```text
① 旧数据训练 / 测试
        ↓
② 发现异常关键点
        ↓
③ 全数据 Pose label order audit
        ↓
④ 发现 24/81 可疑点序
        ↓
⑤ 修 Stable-ID / label 链路
        ↓
⑥ 重新标注 / 审计：75 samples，0 suspect
        ↓
⑦ 重训 RGB / RGB-D baseline
        ↓
⑧ Corrected Core-4 Depth Ablation
        ↓
⑨ 发现同主体 12 张 test Pose ceiling
        ↓
⑩ 构建 Cross-subject Test Set
        ↓
⑪ 第一次 Cross-subject Pose = 0
        ↓
⑫ 定位为 external keypoint ID 顺序与训练体系不一致
        ↓
⑬ canonical reorder audit 101/101 mapping 一致
        ↓
⑭ 重建 hand_cross_subject_v2
        ↓
⑮ 最终 Cross-subject RGB / RGB-D 对比
```

这个过程本身也是项目中非常重要的实验质量控制记录。

---

# 15. 建议的下一阶段实验

后续如果继续扩展，建议优先做：

### 15.1 多 Subject Cross-subject

不要只测试一个 external subject。

建议：

```text
Subject A -> train / val
Subject B -> test
Subject C -> test
Subject D -> test
...
```

最终报告：

```text
每个 Subject 单独指标
+
所有 Cross-subject 汇总指标
```

### 15.2 第一帧人工 canonical ID

未来采集新 Subject 时：

```text
人工指定第一帧 K0~K6
→ Stable-ID 传播
```

从源头避免事后 reorder。

### 15.3 Cross-subject Depth Ablation

在修正后的 Cross-subject 数据上重新执行：

```text
True
Zero
Cross-image Shuffle
Spatial Shuffle
```

以判断深度通道在跨受试者条件下是否仍被有效利用。

### 15.4 可视化误差分析

对：

```text
RGB 正确 / RGBD 错误
RGB 错误 / RGBD 正确
两者都错
两者都正确
```

进行定性与定量分析，比只看一个 mAP 数值更容易理解深度输入到底帮助了什么。

---

# 16. 结果使用注意事项

以下结果不要混入最终主表：

- 关键点 ID 污染前的旧 81 张实验；
- 旧 label order 错误条件下的 Pose 指标；
- Cross-subject canonical reorder 前的 Pose=0 结果。

建议最终论文 / 报告主表只使用：

```text
Corrected In-domain Baseline
Corrected Depth Ablation
Corrected Cross-subject v2
```

---

# 17. 项目当前状态

当前主要链路已经可用：

```text
数据准备
✓

RGB / RGB-D dataset 同步
✓

Pose label audit
✓

Canonical reorder
✓

Cross-subject rebuild
✓

RGB / RGB-D 训练
✓

RGB / RGB-D test
✓

Depth ablation
✓

GUI
✓
```

当前最重要的后续工作不是继续修改基础工具，而是：

> 扩展不同受试者数据，并用统一人工 canonical ID 规则生成更可靠的 Cross-subject benchmark。

---

## Repository

```text
https://github.com/ruirui-y/YoLo-V8-Acuponit-Pose.git
```

