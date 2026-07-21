"""从 mark/rgbd_images 中提取腿部图片（20250930_103_0_0000~0380）并按 70/15/15 划分"""
import os
import random
import shutil
from pathlib import Path

random.seed(42)

# 路径配置
base = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets")
src_images = base / "mark" / "rgbd_images"
src_labels = base / "mark" / "labels"
output = base / "acupoint_380"

# 收集文件：0000 ~ 0380
all_files = []
for i in range(0, 381):  # 含 0380
    stem = f"20250930_103_0_{i:04d}"
    img_path = src_images / f"{stem}.png"
    label_path = src_labels / f"{stem}.txt"
    if img_path.exists() and label_path.exists():
        all_files.append((img_path, label_path))
    else:
        missing = "图片" if not img_path.exists() else "标签"
        print(f"⚠️ 跳过：{stem}（{missing}不存在）")

print(f"\n共找到 {len(all_files)} 张腿部图片")

# 打乱并划分
random.shuffle(all_files)
n = len(all_files)
train_files = all_files[:int(n*0.7)]
val_files = all_files[int(n*0.7):int(n*0.85)]
test_files = all_files[int(n*0.85):]

print(f"训练集: {len(train_files)} 张")
print(f"验证集: {len(val_files)} 张")
print(f"测试集: {len(test_files)} 张")

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

copy_files(train_files, "train")
copy_files(val_files, "val")
copy_files(test_files, "test")

print(f"\n✅ 腿部数据集已创建到: {output}")
print("接下来需要手动操作：")
print("1. 在 datasets/ 下创建 acupoint_leg.yaml（参考 acupoint.yaml）")
print("2. 路径指向 datasets/acupoint_380")
print("3. rgbd: true, channels: 4")