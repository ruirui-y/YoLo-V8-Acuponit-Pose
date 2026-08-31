# 🧹 第一步：清理旧文件

打开 `H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection`，删除以下内容：

| 要删的                 | 路径    | 说明                       |
| ---------------------- | ------- | -------------------------- |
| `runs/` 整个文件夹     | `runs/` | 所有旧训练结果             |
| `check_blue.png` 等    | 根目录  | 检查图片                   |
| `check_channels.py`    | 根目录  | 检查脚本                   |
| `yolov8n.pt`           | 根目录  | 旧检测权重                 |
| `yolov8n-pose_4ch.pt`  | 根目录  | 重复的，跟 `_perfect` 重复 |
| `yolov8_4ch_direct.pt` | 根目录  | 旧检测权重                 |

**注意：`yolov8n-pose.pt` 和 `yolov8n-pose_4ch_perfect.pt` 可以不删**，因为下面生成时会自动覆盖。但如果你要彻底重来，删了也没关系。

---

# 📥 第二步：打开命令行

按 `Win + R`，输入 `cmd`，回车。

然后输入以下命令，一条一条执行：

```bash
# 第1条：激活 conda 环境
conda activate yolov8

# 第2条：进入项目目录
cd /d H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection
```

> 💡 **解释**：`/d` 是 Windows 下切换到另一个盘符的写法。
> 执行成功后，命令行前面会出现 `(yolov8) H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection>`，说明环境对了。

---

# 🏗️ 第三步：生成 4 通道权重

运行这个命令：

```bash
python YJJ-Pose-Scripts/build_rgbd_pose_weights.py
```

**这个脚本会自动做三件事：**
1. 如果本地没有 `yolov8n-pose.pt`，自动从网上下载（126KB 的官方 Pose 权重）
2. 把官方 3 通道权重改成 4 通道（第4通道用RGB平均值初始化）
3. 保存为 `yolov8n-pose_4ch_perfect.pt`（在你的项目根目录下）

**成功后会看到：**
```
============================================================
🚀 终极外科手术 3.0：最稳的 4通道 Pose 模型
============================================================

1. 加载 yolov8n-pose.pt...
...
✅ 手术圆满成功！已生成: yolov8n-pose_4ch_perfect.pt
```

现在你的项目根目录下会有两个文件：

| 文件                          | 通道   | 用途                   |
| ----------------------------- | ------ | ---------------------- |
| `yolov8n-pose.pt`             | 3 通道 | 方向一的 RGB 模型要用  |
| `yolov8n-pose_4ch_perfect.pt` | 4 通道 | 方向一的 RGBD 模型要用 |

---

# 🚀 第四步：先跑 RGBD（4通道）训练

> 先跑4通道版本，因为你的 `train_rgbd_pose.py` 本来就是为它写的。

## 4.1 确认脚本内容

打开 `YJJ-Pose-Scripts/train_rgbd_pose.py`，找到训练参数那一段，确保是以下内容：

```python
results = model.train(
    task='pose',
    data='datasets/acupoint.yaml',
    epochs=120,            # 原来是10，改成120
    imgsz=640,
    batch=4,
    device='cuda:0' if torch.cuda.is_available() else 'cpu',
    name='train_pose_rgbd_4ch',  # 改个名字方便认
    project='runs/pose',
    patience=20,           # 原来是0，改成20（20轮没进步就停）
    save=True,
    plots=True,
    verbose=True,
    workers=0,
    cache=False,
    amp=False,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    fliplr=0.5,
    flipud=0.0,
    degrees=10.0,
    translate=0.1,
    scale=0.2,
    shear=0.0,
    perspective=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    lr0=0.0005,
    lrf=0.01,
    warmup_epochs=10,
)
```

## 4.2 开始训练

在命令行输入：

```bash
python YJJ-Pose-Scripts/train_rgbd_pose.py
```

## 4.3 训练过程中你会看到什么

```
(yolov8) H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection>python YJJ-Pose-Scripts/train_rgbd_pose.py

     Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     1/120     1.08G      3.629      28.82      2.256         16        640
     2/120     1.18G      3.741      29.09      2.214         16        640
     3/120      1.2G      5.262      21.22      3.052         16        640
     ...
```

> 💡 **怎么看进度？**
> - `box_loss`（框损失）和 `pose_loss`（关键点损失）越来越小 → 正常，在学东西
> - 如果 loss 开始反弹上升 → 过拟合了，patience=20 会自动停
> - 预计实际跑 **60-80 轮**左右就会自动结束

## 4.4 训练结束后会生成

```
runs/pose/train_pose_rgbd_4ch/weights/best.pt   ← 这就是你论文要用的模型
runs/pose/train_pose_rgbd_4ch/weights/last.pt   ← 最后1轮的模型
runs/pose/train_pose_rgbd_4ch/results.csv        ← 每轮数据
runs/pose/train_pose_rgbd_4ch/results.png        ← 训练曲线图
```

---

# 🔄 第五步：跑 RGB（3通道）训练

## 5.1 创建 RGB 版数据集配置

复制 `datasets/acupoint.yaml` → 重命名为 `datasets/acupoint_rgb.yaml`

打开 `datasets/acupoint_rgb.yaml`，把这两行改掉：

```yaml
# 原来是：
rgbd: true
channels: 4

# 改成：
rgbd: false
channels: 3
```

**为什么？** `rgbd: true` 会让数据加载器读 4 通道图片，但 3 通道模型只吃 3 通道，会报错。这个新文件告诉加载器"只读 RGB，不要深度通道"。

## 5.2 复制脚本

把 `YJJ-Pose-Scripts/train_rgbd_pose.py` 复制一份，重命名为 `train_rgb_pose.py`。

## 5.3 修改四处

打开 `YJJ-Pose-Scripts/train_rgb_pose.py`，找到这四处改掉：

```python
# 第1处：加载 3 通道官方权重（约第12行）
# 原来是：
model = YOLO('yolov8n-pose_4ch_perfect.pt', task='pose')
# 改成：
model = YOLO('yolov8n-pose.pt', task='pose')

# 第2处：验证模型是 3 通道（约第15-18行）
# 原来是：
assert first_layer.weight.shape[1] == 4, "❌ 模型不是4通道！"
# 改成：
assert first_layer.weight.shape[1] == 3, "❌ 模型不是3通道！"
# 再把打印信息也改一下
print("✓ 模型确认为 3 通道")

# 第3处：数据集配置指向新 YAML（约第27行）
# 原来是：
data='datasets/acupoint.yaml',
# 改成：
data='datasets/acupoint_rgb.yaml',

# 第4处：改个名字方便认（约第35行）
# 原来是：
name='train_pose_rgbd_4ch',
# 改成：
name='train_pose_rgb_3ch',
```

**其他所有参数保持完全不变**（epochs=120, batch=4, 数据增强等全部一样）。

## 5.4 开始训练

```bash
python YJJ-Pose-Scripts/train_rgb_pose.py
```

## 5.5 训练结束后会生成

```
runs/pose/train_pose_rgb_3ch/weights/best.pt
runs/pose/train_pose_rgb_3ch/results.csv
runs/pose/train_pose_rgb_3ch/results.png
```

---

# 📊 第六步：测试集评估（打最终成绩）

两个模型都跑完后，用测试集分别打分。

## 6.1 评估 RGBD（4通道）

创建 `YJJ-Pose-Scripts/eval_rgbd_pose.py`：

```python
from ultralytics import YOLO

def main():
    model = YOLO('runs/pose/train_pose_rgbd_4ch/weights/best.pt', task='pose')
    metrics = model.val(data='datasets/acupoint.yaml', split='test', imgsz=640, batch=4)

    print('=== RGBD 4通道 测试集成绩 ===')
    print(f'Box mAP50:     {metrics.box.map50:.4f}')
    print(f'Box mAP50-95:  {metrics.box.map:.4f}')
    print(f'Pose mAP50:    {metrics.pose.map50:.4f}')
    print(f'Pose mAP50-95: {metrics.pose.map:.4f}')

if __name__ == '__main__':
    main()
```

运行：

```bash
python YJJ-Pose-Scripts/eval_rgbd_pose.py
```

## 6.2 评估 RGB（3通道）

创建 `YJJ-Pose-Scripts/eval_rgb_pose.py`：

```python
from ultralytics import YOLO

def main():
    model = YOLO('runs/pose/train_pose_rgb_3ch2/weights/best.pt', task='pose')
    metrics = model.val(data='datasets/acupoint_rgb.yaml', split='test', imgsz=640, batch=4)

    print('=== RGB 3通道 测试集成绩 ===')
    print(f'Box mAP50:     {metrics.box.map50:.4f}')
    print(f'Box mAP50-95:  {metrics.box.map:.4f}')
    print(f'Pose mAP50:    {metrics.pose.map50:.4f}')
    print(f'Pose mAP50-95: {metrics.pose.map:.4f}')

if __name__ == '__main__':
    main()
```

运行：

```bash
python YJJ-Pose-Scripts/eval_rgb_pose.py
```

> 💡 **`split='test'` 什么意思？**
> 就是只用 test 文件夹里的 14 张图来评估，模拟"高考"——模型从来没见过的题。

## 6.3 方向一最终结果

| 指标          | RGB（3通道） | RGBD（4通道） |   差异   |
| ------------- | :----------: | :-----------: | :------: |
| Box mAP50     |    0.9950    |    0.9950     |   持平   |
| Box mAP50-95  |  **0.9346**  |    0.9145     | RGB 略高 |
| Pose mAP50    |    0.9950    |    0.9950     |   持平   |
| Pose mAP50-95 |    0.9950    |    0.9950     |   持平   |

> **注意：** 两个模型成绩都接近满分（0.99+），这是因为测试集只有 14 张图，任务偏简单，不足以区分两者差距。后续方向二的骨干网络对比实验可以进一步验证模型的泛化能力。



# 📋 完整的执行清单（按顺序做）

| 序号   | 做什么                         | 命令/操作                                                    | 耗时           |
| ------ | ------------------------------ | ------------------------------------------------------------ | -------------- |
| [ ] 1  | 删除旧的 `runs/` 文件夹        | 文件管理器删除                                               | 1分钟          |
| [ ] 2  | 打开 cmd 并进入环境            | `conda activate yolov8` → `cd /d H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection` | 1分钟          |
| [ ] 3  | 生成 4 通道权重                | `python YJJ-Pose-Scripts/build_rgbd_pose_weights.py`           | 1分钟          |
| [ ] 4  | 修改 `train_rgbd_pose.py` 参数 | epochs=120, patience=20, name 改掉                           | 2分钟          |
| [ ] 5  | **跑 RGBD 训练**               | `python YJJ-Pose-Scripts/train_rgbd_pose.py`                 | **~30-60分钟** |
| [ ] 6  | 创建 `train_rgb_pose.py`       | 复制 + 改 2 行                                               | 2分钟          |
| [ ] 7  | **跑 RGB 训练**                | `python YJJ-Pose-Scripts/train_rgb_pose.py`                  | **~30-60分钟** |
| [ ] 8  | **评估 RGBD**                  | 运行上面的评估命令                                           | 2分钟          |
| [ ] 9  | **评估 RGB**                   | 运行上面的评估命令                                           | 2分钟          |
| [ ] 10 | 填对比表格                     | 记下数字                                                     | 5分钟          |

---

## ⚠️ 可能会遇到的问题

**Q：提示 `conda` 不是内部命令？**
→ 先打开 Anaconda Prompt（开始菜单搜 Anaconda Prompt），在里面执行命令。

**Q：提示 `yolov8n-pose.pt` 下载很慢？**
→ 正常的，第一次运行会自动下载。也可以手动下载放到项目根目录：
→ https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-pose.pt

**Q：训练到一半 CUDA out of memory？**
→ 把 `batch=4` 改成 `batch=2` 或 `batch=1`。

**Q：训练时 loss 全是 NaN？**
→ 马上停止，检查图片是不是损坏了。

---

做完这 10 步，方向一就验证完毕了，你就有了论文里第一张核心对比表的数字。有问题随时问我，一步一步来就行。

---

# 方向一文档收尾

在文档最后加上以下内容：

---

## ✅ 方向一：完成总结

### 实验结果

| 指标          | RGB（3通道） | RGBD（4通道） |   差异   |
| ------------- | :----------: | :-----------: | :------: |
| Box mAP50     |    0.9950    |    0.9950     |   持平   |
| Box mAP50-95  |  **0.9346**  |    0.9145     | RGB 略高 |
| Pose mAP50    |    0.9950    |    0.9950     |   持平   |
| Pose mAP50-95 |    0.9950    |    0.9950     |   持平   |

### 结果分析

两个模型在测试集上均达到 0.99+ 的极高精度，**并非深度通道无效**，而是存在以下两个问题：

1. **测试集样本过少（仅 14 张）**，任务难度不足，无法拉开差距
2. **两个模型均使用了完整的数据增强**（翻转、旋转、平移、缩放），在简单任务上泛化能力都已足够强

### 下一步

重新划分数据集，扩大测试集样本量（14 张 → 20 张），然后重新训练并验证。

---

## 数据集重新划分方案

### 当前分布

| 来源前缀                       |   train   |   val    |   test   |   合计    |
| ------------------------------ | :-------: | :------: | :------: | :-------: |
| `20250915_103_0_`（0000-0067） |   54张    |   7张    |   7张    |   68张    |
| `20250930_103_2_`（0000-0064） |   52张    |   6张    |   7张    |   65张    |
| **合计**                       | **106张** | **13张** | **14张** | **133张** |

### 新划分（70% / 15% / 15%）

| 集       |   比例   |   数量    |
| -------- | :------: | :-------: |
| train    |   70%    |   93张    |
| val      |   15%    |   20张    |
| test     |   15%    |   20张    |
| **合计** | **100%** | **133张** |

### 操作方法

创建 `YJJ-Pose-Scripts/split_dataset.py`：

```python
import os
import random
import shutil
from pathlib import Path

base_dir = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# 收集所有文件
all_files = []
for split in ["train", "val", "test"]:
    for f in (images_dir / split).glob("*.png"):
        stem = f.stem
        all_files.append({
            "stem": stem,
            "img_src": str(images_dir / split / f"{stem}.png"),
            "label_src": str(labels_dir / split / f"{stem}.txt")
        })

print(f"总共找到 {len(all_files)} 张图片")

# 随机打乱
random.seed(42)
random.shuffle(all_files)

# 划分
n = len(all_files)
train_files = all_files[:int(n*0.7)]
val_files = all_files[int(n*0.7):int(n*0.85)]
test_files = all_files[int(n*0.85):]

print(f"训练集: {len(train_files)} 张")
print(f"验证集: {len(val_files)} 张")
print(f"测试集: {len(test_files)} 张")

# 创建新目录（不碰旧的）
new_base = base_dir.parent / "acupoint_new"
for split in ["train", "val", "test"]:
    (new_base / "images" / split).mkdir(parents=True, exist_ok=True)
    (new_base / "labels" / split).mkdir(parents=True, exist_ok=True)

# 复制到新目录
def copy_to(file_list, split_name):
    for item in file_list:
        dst_img = str(new_base / "images" / split_name / f"{item['stem']}.png")
        dst_label = str(new_base / "labels" / split_name / f"{item['stem']}.txt")
        shutil.copy2(item["img_src"], dst_img)
        shutil.copy2(item["label_src"], dst_label)

copy_to(train_files, "train")
copy_to(val_files, "val")
copy_to(test_files, "test")

print(f"\n✅ 新数据集已创建到: {new_base}")
print("请手动操作：")
print("1. 删除旧的 datasets/acupoint 文件夹")
print("2. 把 datasets/acupoint_new 重命名为 acupoint")
```

运行：

```bash
python YJJ-Pose-Scripts/split_dataset.py
```

### 完成后

重新依次执行文档中的 第三步到第六步：

```
第三步 → 生成 4 通道权重
第四步 → 跑 RGBD 训练
第五步 → 跑 RGB 训练
第六步 → 评估对比
```



