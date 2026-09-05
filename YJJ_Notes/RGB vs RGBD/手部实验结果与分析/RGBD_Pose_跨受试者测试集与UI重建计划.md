# RGBD Pose 下一阶段计划：跨受试者测试集 + UI 重建 Test Set

## 1. 当前状态

修正 LaMa 标注中的关键点 ID 交换问题后，新标签审计结果为：

```text
valid      : 75
suspects   : 0
parse_error: 0
```

当前重新训练后的 baseline：

| 模型 | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| RGB | 0.9473 | 0.9950 |
| RGBD | 0.9612 | 0.9882 |

新的 RGBD Core 消融：

| Variant | Box mAP50-95 | Pose mAP50-95 |
|---|---:|---:|
| True Depth | 0.9612 | 0.9882 |
| Zero Depth | 0.8955 | 0.9950 |
| Cross-image Shuffled | 0.9431 | 0.9868 |
| Spatial-shuffled | 0.3026 | 0.1348 |

当前能够确认：

- 第四通道确实被网络使用；
- Depth 对 Box 定位有明显帮助；
- 打乱 Depth 空间结构后性能大幅下降，说明网络依赖 Depth 的空间组织；
- 当前 12 张 test 来自相近时间段/同一主体，Pose 已接近 0.995，存在明显天花板效应；
- 因此下一阶段不应继续只围绕这 12 张连续帧堆消融，而应先建立真正的跨受试者测试集。

---

## 2. 下一阶段主目标：Cross-subject Test

下一步计划使用“其他人的手”作为正式测试集，不再使用当前主体的连续帧作为最终 test。

推荐划分：

```text
Subject A（当前主体）
├─ train
└─ val

Subject B / C / D ...
└─ test only
```

核心原则：

- 测试主体不能出现在 train / val；
- 不允许把同一个人的一部分图片放 train、另一部分放 test；
- 外部主体数据全部进入 test，不重新随机拆分；
- RGB 与 RGBD 必须使用完全相同的 test 文件名、标签和 split。

论文最终要回答的问题改为：

> RGBD 是否能在未见过的手/未见主体上，提供比 RGB 更好的定位与关键点泛化能力？

这比当前同主体、连续帧 test 更有说服力。

---

## 3. 建议的新数据集版本

不要覆盖现有数据集。

建议新建：

```text
dataset/
└─ hand_cross_subject_v1/
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

其中：

```text
train / val = 当前主体数据
test        = 其他人的手
```

RGB / RGBD 两套数据：

```text
train filenames 完全一致
val filenames   完全一致
test filenames  完全一致
labels          完全一致
```

区别只能是：

```text
RGB  = 3 channel
RGBD = 4 channel
```

---

# 4. UI 必须新增：重建 Cross-subject Test Set

需要把“重建测试集”加入现有 Pose 数据集准备 UI。

建议新增区域：

```text
Cross-subject Test Set
────────────────────────────────

External Images:
[ ................................ ] [Browse]

External Labels:
[ ................................ ] [Browse]

External Depth NPY:
[ ................................ ] [Browse]

[ Validate ]

Images : xxx
Labels : xxx
Depth  : xxx
Matched: xxx
Keypoints: N

Status: Ready / Error

[ Rebuild Test Set ]
```

RGB 模式下 Depth NPY 可以禁用或隐藏。

RGBD 模式下 Depth NPY 必须存在且全部匹配。

---

## 5. UI：External Test Source 校验

点击 Validate 后必须检查：

### 文件数量

```text
image count
label count
npy count
```

### 时间戳匹配

必须按完整时间戳匹配，例如：

```text
color_2026_07_28_20_02_17_074608.png
color_2026_07_28_20_02_17_074608.txt
depth_2026_07_28_20_02_17_074608.npy
```

不能按照目录遍历顺序配对。

RGBD 模式下：

```text
image 有、npy 没有
→ BLOCK
```

### Label 校验

至少检查：

- 每张 image 有对应 label；
- label 格式合法；
- keypoint count 与目标数据集一致；
- 不允许空 label；
- 不允许缺失文件。

`check_pose_label_order.py` 可以继续作为额外的离线审计工具，不必强行耦合进数据集重建核心。

---

# 6. “Rebuild Test Set”按钮行为

目标：

```text
当前正式数据集
        ↓
保留 train
保留 val
        ↓
替换旧 test
        ↓
External Subject 全部写入新 test
        ↓
RGB / RGBD 使用同一批样本
        ↓
重新生成 YAML
        ↓
最终一致性验证
```

建议不要原地破坏当前数据集，而是生成新的 version，例如：

```text
hand_cross_subject_v1
```

---

## 7. RGB 与 RGBD 同步重建

重建 Test Set 时必须一次性完成 RGB / RGBD 同步。

逻辑：

```text
External RGB Images
External Labels
External Depth NPY
        ↓
按 timestamp 建立唯一 sample list
        ↓
同一个 sample list
        ├─ 生成 RGB test
        └─ 生成 RGBD test
```

最终必须自动检查：

```text
RGB test stems == RGBD test stems
RGB labels     == RGBD labels
```

任何不一致都 BLOCK。

---

# 8. YAML 路径必须再次防止临时目录 Bug

之前已经真实出现：

```text
.prepare_pose_rgbd_tmp
```

残留到正式 YAML 中，导致训练时去已经不存在的临时目录找数据。

因此新的 Rebuild Test Set 完成后必须保证：

RGB：

```yaml
path: H:/.../dataset/hand_cross_subject_v1/rgb
train: images/train
val: images/val
test: images/test
```

RGBD：

```yaml
path: H:/.../dataset/hand_cross_subject_v1/rgbd
train: images/train
val: images/val
test: images/test
```

最终 YAML 中禁止出现：

```text
.prepare_pose_rgbd_tmp
```

正确生命周期：

```text
临时目录准备
→ 原子发布到最终目录
→ 用最终正式路径生成/重写 YAML
→ 验证 train/val/test 都实际存在
```

---

# 9. UI 建议增加的最终摘要

重建前显示：

```text
Current Train : 52
Current Val   : 11

External Test
Images        : 120
Labels        : 120
Depth         : 120
Matched       : 120

Keypoints     : 7

RGB/RGBD synchronized : YES
Subject-independent   : user confirmed

Output:
dataset/hand_cross_subject_v1
```

点击：

```text
[ Rebuild Cross-subject Test Set ]
```

再执行实际生成。

---

# 10. 新测试集建立后的实验顺序

新的外部主体 test 建好后：

## 第一阶段：主实验

```text
1. RGB baseline
2. RGBD baseline
```

重点比较：

```text
Box mAP50-95
Pose mAP50-95
```

建议最终论文主表同时保留：

| Model | In-domain Test | Cross-subject Test |
|---|---:|---:|
| RGB | ... | ... |
| RGBD | ... | ... |

真正重点放在 Cross-subject Test。

---

## 第二阶段：重新跑核心消融

在新的外部主体 test 上重新跑：

```text
True Depth
Zero Depth
Cross-image Shuffle
Spatial Shuffle
```

回答：

1. 新主体上第四通道是否仍然有效；
2. Depth 是否提高泛化能力；
3. 是否依赖真实空间结构；
4. RGBD 相对 RGB 的优势是否在困难泛化场景扩大。

---

## 第三阶段：必要时再扩展消融

只有 Core 4 有意义后再考虑：

```text
Flat Valid
Full Constant
Scale 25 / 50 / 75
Blur
```

不要在当前旧的 12 张连续帧 test 上继续堆大量消融。

---

# 11. 新论文叙事方向

当前同主体 test：

```text
RGB Pose  = 0.9950
RGBD Pose = 0.9882
```

不应该为了论文强行解释成“RGBD 全面更优”。

更合理的研究路线是验证：

```text
同主体 / 简单场景：
RGB 已接近天花板，RGB ≈ RGBD

跨主体 / 困难场景：
RGBD 是否比 RGB 更稳、更准
```

如果最终得到：

```text
In-domain:
RGB ≈ RGBD

Cross-subject:
RGBD > RGB
```

论文结论会比“在同一批连续帧上高几个千分点”更有说服力。

---

# 12. 接下来开发优先级

```text
1. 完成当前 LaMa/UI 已知 bug 修复
2. 在数据集 UI 中加入 Rebuild Cross-subject Test Set
3. 采集其他人的手
4. 用修复后的 LaMa 流程重新标注
5. 运行 keypoint order audit
6. UI 重建 hand_cross_subject_v1
7. 验证 RGB/RGBD test 完全同步
8. 重新训练/评估 RGB 与 RGBD
9. 在新 test 上重新跑 Core 4 Depth 消融
10. 根据结果决定后续论文实验
```

---

# 13. 当前阶段一句话结论

下一阶段的重点不再是继续证明模型能拟合当前同主体连续帧，而是：

> 建立严格的跨受试者测试集，验证 RGBD 是否能够在未见主体上提供真正可复现的泛化优势。
