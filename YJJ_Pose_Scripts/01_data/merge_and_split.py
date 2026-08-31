import os
import random
import shutil
from pathlib import Path

random.seed(42)

# 路径配置
base = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets")
old_acupoint = base / "acupoint"
new_mark = base / "mark"
output = base / "acupoint_v2"

# 收集所有旧数据（从旧 acupoint 的 train/val/test 里找）
old_images = []
old_labels = []
for split in ["train", "val", "test"]:
    for f in (old_acupoint / "images" / split).glob("*.png"):
        old_images.append(f)
    for f in (old_acupoint / "labels" / split).glob("*.txt"):
        old_labels.append(f)

print(f"旧数据: {len(old_images)} 张图片, {len(old_labels)} 个标签")

# 收集所有新数据
new_images = list((new_mark / "rgbd_images").glob("*.png"))
new_labels = []
for img in new_images:
    label = new_mark / "labels" / f"{img.stem}.txt"
    if label.exists():
        new_labels.append(label)

print(f"新数据: {len(new_images)} 张图片, {len(new_labels)} 个标签")

# 合并成完整列表
all_files = []
for img in old_images:
    stem = img.stem
    # 找对应的标签
    label = None
    for split in ["train", "val", "test"]:
        p = old_acupoint / "labels" / split / f"{stem}.txt"
        if p.exists():
            label = p
            break
    if label:
        all_files.append((img, label))

for img in new_images:
    stem = img.stem
    label = new_mark / "labels" / f"{stem}.txt"
    if label.exists():
        all_files.append((img, label))

print(f"总计: {len(all_files)} 对 (图片+标签)")

# 打乱
random.shuffle(all_files)

# 按 70/15/15 划分
n = len(all_files)
train = all_files[:int(n*0.7)]
val = all_files[int(n*0.7):int(n*0.85)]
test = all_files[int(n*0.85):]

print(f"\n训练集: {len(train)} 张")
print(f"验证集: {len(val)} 张")
print(f"测试集: {len(test)} 张")

# 创建输出目录
for split in ["train", "val", "test"]:
    (output / "images" / split).mkdir(parents=True, exist_ok=True)
    (output / "labels" / split).mkdir(parents=True, exist_ok=True)

# 复制文件
def copy_files(file_list, split_name):
    for img_src, label_src in file_list:
        stem = img_src.stem
        dst_img = output / "images" / split_name / f"{stem}.png"
        dst_label = output / "labels" / split_name / f"{stem}.txt"
        shutil.copy2(img_src, dst_img)
        shutil.copy2(label_src, dst_label)

copy_files(train, "train")
copy_files(val, "val")
copy_files(test, "test")

print(f"\n✅ 合并完成！新数据集在: {output}")
print("请手动操作：")
print("1. 删除旧的 datasets/acupoint 文件夹")
print("2. 把 datasets/acupoint_v2 重命名为 acupoint")