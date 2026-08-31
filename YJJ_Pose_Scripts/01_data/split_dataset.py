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