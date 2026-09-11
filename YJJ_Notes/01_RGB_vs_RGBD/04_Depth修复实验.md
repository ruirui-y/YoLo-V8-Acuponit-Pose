# Depth 修复实验

## 1. 为什么开始检查 Depth

早期 RGBD 实验中已经确认：

- 第 4 通道会影响模型输出；
- 破坏 Depth 空间结构会造成明显性能下降。

但 Cross-subject 上 RGBD 并没有稳定优于 RGB。

随后重新检查原始 Depth 数据质量，发现训练 Depth 中存在较多空洞/无效像素。

---

## 2. 原始训练 Depth 质量

训练 Depth：

`H:\YJJ\Yolo_RGBD\Resource\session1_200146\out_npy`

处理 75 / 75 文件后：

- Total：75
- Valid：75
- Invalid：0
- Invalid ratio：0
- Mean hole ratio：0.1267
- Mean valid ratio：0.9942

重要解释：

75 张全部通过质量门槛，不代表 Depth 没问题。

原始图像平均约 **12.67%** 像素属于空洞/无效区域，因此：

**Depth 文件存在且整体通过阈值 != Depth 适合未经处理直接作为第 4 通道。**

---

## 3. 当前无效 Depth 定义

统一 invalid mask：

```text
(~isfinite(depth))
OR (depth <= 0)
OR (depth == 65535)
```

不能只处理 `depth == 0`。

---

## 4. 当前修补脚本

入口：

`YJJ_Pose_Scripts\07_depth_clean\filter_depth.py`

工具：

`YJJ_Pose_Scripts\07_depth_clean\utils\fill_depth.py`

当前脚本支持：

- `--input-dir`
- `--output-dir`
- `--depth-low`
- `--depth-high`
- `--max-hole-ratio`
- `--min-valid-range-ratio`

默认输出目录：

`<input_dir>_filtered`

同时输出：

- `depth_quality.csv`
- `invalid_depths.txt`
- `valid_depths.txt`

即使某张图没有通过质量门槛，也保留修补后的输出，不自动删除样本。

---

## 5. 当前修补方法

当前 `fill_zihang_filter` 主要流程：

1. 根据统一 invalid mask 找无效点；
2. 将无效位置置为 0；
3. 使用 OpenCV Telea inpaint 修补；
4. 再进行 bilateral filter；
5. 保持原 shape / dtype；
6. 整数 Depth 做 round；
7. 输出中避免 NaN / Inf / 65535。

这份处理来自师兄给出的 Depth 修补思路，并在本项目中统一了无效值判断和 CLI。

---

## 6. 训练集 Depth 修补

命令：

```bat
H:\YJJ\Conda\python.exe filter_depth.py ^
  --input-dir "H:\YJJ\Yolo_RGBD\Resource\session1_200146\out_npy" ^
  --depth-low 1100 ^
  --depth-high 1850
```

输出：

`H:\YJJ\Yolo_RGBD\Resource\session1_200146\out_npy_filtered`

随后使用该目录重新生成当前 RGBD 训练数据集。

---

## 7. Cross-subject Depth 修补

输入：

`H:\YJJ\Yolo_RGBD\Resource\session3_200348\out_npy`

输出：

`H:\YJJ\Yolo_RGBD\Resource\session3_200348\out_npy_filtered`

当前正式 `hand_cross_subject` RGBD 测试集由修补后的 session3 Depth 重建。

训练端和测试端保持相同的 Depth 处理原则。

---

## 8. 修补前的 Cross-subject NME

修补前历史结果：

- Total：101
- RGB matched：66
- RGBD matched：71
- Common：54
- RGB-only：12
- RGBD-only：17
- Neither：18

Common 54：

- RGB NME：0.082723
- RGBD NME：0.086873
- Delta：+0.004150
- RGB wins：34
- RGBD wins：20

当时 RGBD 没有形成总体 NME 优势。

---

## 9. 当前修补后的 Cross-subject NME

当前模型：

- RGB：`train_pose_rgb_3ch5`
- RGBD：`train_pose_rgbd_4ch5`

结果：

- Total：101
- RGB matched：66
- RGBD matched：101
- Common：66
- RGB-only：0
- RGBD-only：35
- Neither：0

Common 66：

- RGB NME：0.088657
- RGBD NME：0.059819
- Delta：-0.028838
- RGBD wins：63
- RGB wins：3

当前 repaired/filtered Depth 管线下，RGBD 表现与之前原始 Depth 阶段明显不同。

---

## 10. 科学解释边界

不能简单写：

> 填洞算法直接导致了全部提升。

原因：

- 修补前后 RGBD 的训练数据发生变化；
- Cross-subject 测试的第 4 通道也发生变化；
- 当前 RGB 也重新训练为 `3ch5`；
- 两次统计的 Common matched 集合不同；
- 深度修补包含填洞和滤波，不是单独一个变量。

所以当前最稳妥的结论是：

> Depth 数据质量和预处理管线对 RGBD Pose 的跨受试者表现具有非常重要的影响；在当前 repaired/filtered Depth 管线下，RGBD 相比当前 RGB 已表现出明确优势。

如果未来论文需要单独证明“训练端修补”和“测试端修补”各自贡献，再设计严格 2x2 控制实验。

目前方向 1 不继续展开。
