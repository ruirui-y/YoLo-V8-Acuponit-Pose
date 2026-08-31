# 方向一验证方案：RGB vs RGBD 消融实验

## 📌 一句话搞懂方向一在做什么

> **问题**：我加了深度通道（第4通道），到底有没有用？
> **实验**：训练两个模型，一个只有 RGB，一个带 Depth，比谁准。
> **结论**：如果带 Depth 的准，就证明深度通道有用，这就是你的论文创新点。

---

## 🔧 第一步：准备工作（理解你手上有啥）

### 你的数据集

| 文件夹                            | 数量   | 干嘛用的                                |
| --------------------------------- | ------ | --------------------------------------- |
| `datasets/acupoint/images/train/` | 106 张 | **训练** — 给模型学习用的               |
| `datasets/acupoint/images/val/`   | 13 张  | **验证** — 训练过程中用来挑最佳模型     |
| `datasets/acupoint/images/test/`  | 14 张  | **测试** — 全部训练完后，最终打成绩用的 |

> 💡 **为什么分三个？** 就像考试：train 是平时做题，val 是模拟考（看学得怎么样，调学习策略），test 是高考（最终成绩，平时绝对不看）。

### 你的配置文件

`datasets/acupoint.yaml` 里面写了：
- 数据集的路径
- 类别（腹部 = 1 个类）
- 关键点数量（3 个：左穴位、肚脐、右穴位）
- RGBD 模式（`rgbd: true`, `channels: 4`）

---

## 🏗️ 第二步：你要训练两个模型

| 模型             | 名字      | 输入               | 预训练权重                           | 作用                 |
| ---------------- | --------- | ------------------ | ------------------------------------ | -------------------- |
| **A1（对照组）** | RGB 模型  | 3通道（只有彩色）  | 官方 `yolov8n-pose.pt`               | 看看不加深度能有多准 |
| **A2（实验组）** | RGBD 模型 | 4通道（彩色+深度） | 你做的 `yolov8n-pose_4ch_perfect.pt` | 看看加了深度能有多准 |

**两个模型除了通道数不同，其他所有设置必须一模一样。** 这样如果结果有差异，原因只能是"深度通道"。

---

## 📝 第三步：具体执行步骤

### 第 1 步：跑 RGBD 模型（A2，这是你已有的）

**做什么**：直接用你写好的 `train_rgbd_pose.py` 开始训练。

**打开文件确认一下关键内容**，大概长这样：

```python
# train_rgbd_pose.py 核心部分
from ultralytics import YOLO

def main():
    # 加载 4通道 模型
    model = YOLO('yolov8n-pose_4ch_perfect.pt', task='pose')
    
    # 开始训练
    model.train(
        data='datasets/acupoint.yaml',
        epochs=120,          # 上限120轮
        patience=20,         # 20轮没进步就自动停
        imgsz=640,
        batch=4,
        device='cuda:0',
        name='train_pose_rgbd_4ch',   # 改个名字，方便以后认
        project='runs/pose',
        workers=0,
        amp=False,
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        ...
    )

if __name__ == '__main__':
    main()
```

**如何运行**：打开终端（命令行），输入：

```bash
conda activate yolov8
cd H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection
python train_rgbd_pose.py
```

**训练过程中你会看到什么：**
```
     Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     1/120     1.08G      3.629      28.82      2.256         16        640
     2/120     1.18G      3.741      29.09      2.214         16        640
     ...
```

> 💡 **loss（损失值）** 越来越小 → 说明模型在学东西，这是好事。
> 如果 loss 开始反弹上升 → 过拟合了，早停会自动掐断。

**训练结束后会生成**：
```
runs/pose/train_pose_rgbd_4ch/weights/best.pt   ← 最佳模型（论文用的）
runs/pose/train_pose_rgbd_4ch/weights/last.pt   ← 最后一轮的模型
runs/pose/train_pose_rgbd_4ch/results.csv        ← 每轮的指标数据
runs/pose/train_pose_rgbd_4ch/results.png        ← 训练曲线图
```

---

### 第 2 步：跑 RGB 模型（A1，需要新建脚本）

**做什么**：复制 `train_rgbd_pose.py`，改一行代码后开始训练。

1. 复制文件：把 `train_rgbd_pose.py` 复制一份，重命名为 `train_rgb_pose.py`
2. 修改模型加载那一行：

```python
# 原来是（4通道RGBD）
model = YOLO('yolov8n-pose_4ch_perfect.pt', task='pose')

# 改成（3通道RGB，用官方的）
model = YOLO('yolov8n-pose.pt', task='pose')
```

> 💡 **为什么可以用官方的？** 官方 `yolov8n-pose.pt` 本来就是 3 通道的，直接下载就能用。
> 你之前做 `build_rgbd_pose_weights.py` 是因为你需要 4 通道才要转换。

3. 改一下训练名字，免得跟 RGBD 的结果混在一起：

```python
name='train_pose_rgb_3ch',   # 改成 3ch 方便区分
```

4. 其他**所有参数保持不变**（epochs=120, batch=4, 数据增强等等都一样）。

5. 运行：

```bash
conda activate yolov8
python train_rgb_pose.py
```

6. 训练结束后会生成：
```
runs/pose/train_pose_rgb_3ch/weights/best.pt
runs/pose/train_pose_rgb_3ch/results.csv
```

---

### 第 3 步：评估两个模型（打成绩）

**做什么**：用测试集（test 文件夹的 14 张图）分别评估两个模型。

**先评估 RGBD 模型：**

```bash
python eval_pipeline_smoke_test.py \
    --model runs/pose/train_pose_rgbd_4ch/weights/best.pt \
    --data datasets/acupoint.yaml \
    --split test
```

**再评估 RGB 模型：**

```bash
python eval_pipeline_smoke_test.py \
    --model runs/pose/train_pose_rgb_3ch/weights/best.pt \
    --data datasets/acupoint.yaml \
    --split test
```

> 💡 **为什么用 test 不用 val？** val 是训练过程中用来选 best.pt 的，模型已经"见过"val 了，用它算成绩不公平。test 才是真正的"高考"，模型完全没见过的题。

---

### 第 4 步：对比结果

**做什么**：把两个模型的成绩填到表格里对比。

```
┌──────────────────┬────────────┬────────────┬──────────┐
│      指标        │  RGB (3ch) │ RGBD (4ch) │  提升    │
├──────────────────┼────────────┼────────────┼──────────┤
│ Box mAP50        │     ?      │     ?      │   +?%    │
│ Box mAP50-95     │     ?      │     ?      │   +?%    │
│ Pose mAP50       │     ?      │     ?      │   +?%    │
│ Pose mAP50-95    │     ?      │     ?      │   +?%    │
│ 关键点平均误差    │    ? px    │    ? px    │   ↓? px  │
└──────────────────┴────────────┴────────────┴──────────┘
```

> 💡 **怎么才算"赢了"？**
> - 4ch 的 **mAP 更高**（数值越大越好）
> - 4ch 的**关键点误差更小**（数值越小越好）
> - 如果两个差距不大（<1%），说明深度信息帮助有限
> - 如果 4ch 明显更好（+3%以上），这就是你论文的核心证据

---

## 🔍 过程中你可能遇到的问题

### 问题 1：训练到一半停了

正常。`patience=20` 的意思是"验证集连续 20 轮没进步就自动停"。可能 60 轮就停了，这是好事，说明模型已经学到位了。

### 问题 2：报错 "CUDA out of memory"

显卡显存不够。把 `batch=4` 改成 `batch=2` 试试。

### 问题 3：下载官方 `yolov8n-pose.pt` 很慢

第一次运行会自动从外网下载。可以手动下载放到项目根目录：
- 下载地址：https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt

### 问题 4：eval_pipeline_smoke_test.py 报错

看看测试集路径对不对：
```
H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint/images/test/
```

---

## ✅ 做完方向一之后你手里有什么

做完上面 4 步，你手上就有：

- ✅ 一个训练好的 RGBD 模型（`best_4ch.pt`）
- ✅ 一个训练好的 RGB 模型（`best_3ch.pt`）
- ✅ 两个模型在测试集上的全部指标
- ✅ 一张可以放进论文的对比表格
- ✅ 证明了"深度通道有效" = 你的核心创新点

---

## 📋 你的行动清单（按顺序做）

| 序号   | 做什么                                               | 在哪里             | 预计耗时    |
| ------ | ---------------------------------------------------- | ------------------ | ----------- |
| [ ] 1  | 打开 `train_rgbd_pose.py`，确认代码没问题            | 项目根目录         | 5 分钟      |
| [ ] 2  | 确认 epochs=120, patience=20 已设好                  | train_rgbd_pose.py | 2 分钟      |
| [ ] 3  | **跑 RGBD 模型**：`python train_rgbd_pose.py`        | 终端（命令行）     | ~30-60 分钟 |
| [ ] 4  | 复制 `train_rgbd_pose.py` → 改名 `train_rgb_pose.py` | 文件管理器         | 1 分钟      |
| [ ] 5  | 把模型加载那行换成官方 `yolov8n-pose.pt`             | train_rgb_pose.py  | 1 分钟      |
| [ ] 6  | 把 name 改成 `train_pose_rgb_3ch`                    | train_rgb_pose.py  | 1 分钟      |
| [ ] 7  | **跑 RGB 模型**：`python train_rgb_pose.py`          | 终端               | ~30-60 分钟 |
| [ ] 8  | **评估 RGBD**：运行 eval_pipeline_smoke_test.py               | 终端               | 2 分钟      |
| [ ] 9  | **评估 RGB**：运行 eval_pipeline_smoke_test.py                | 终端               | 2 分钟      |
| [ ] 10 | 把两个模型的指标填到对比表格                         | 笔记/Word          | 10 分钟     |

---

好了，这就是方向一的全部流程。如果你想，我可以现在就帮你把 `train_rgb_pose.py`（RGB 3通道训练脚本）直接写出来，你只需要复制粘贴然后运行就行。要不要？