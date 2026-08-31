"""将 point_10 按 70/15/15 划分到 train/val/test"""
import random
import shutil
from pathlib import Path

random.seed(42)

src = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint_leg/point_10")
dst = src  # 直接在原目录划分

# 收集所有文件
all_files = []
for f in (src / "images" / "train").glob("*.png"):
    stem = f.stem
    label = src / "labels" / "train" / f"{stem}.txt"
    if label.exists():
        all_files.append(stem)

print(f"共找到 {len(all_files)} 个文件")

# 打乱
random.shuffle(all_files)

# 划分
n = len(all_files)
train_files = all_files[:int(n*0.7)]
val_files = all_files[int(n*0.7):int(n*0.85)]
test_files = all_files[int(n*0.85):]

print(f"训练集: {len(train_files)} 张")
print(f"验证集: {len(val_files)} 张")
print(f"测试集: {len(test_files)} 张")

# 创建目标目录
for split in ["val", "test"]:
    (src / "images" / split).mkdir(parents=True, exist_ok=True)
    (src / "labels" / split).mkdir(parents=True, exist_ok=True)

# 移动文件到对应目录
def move_files(stems, split_name):
    for stem in stems:
        img_src = src / "images" / "train" / f"{stem}.png"
        label_src = src / "labels" / "train" / f"{stem}.txt"
        img_dst = src / "images" / split_name / f"{stem}.png"
        label_dst = src / "labels" / split_name / f"{stem}.txt"
        shutil.move(img_src, img_dst)
        shutil.move(label_src, label_dst)

move_files(val_files, "val")
move_files(test_files, "test")

print(f"\n✅ 划分完成！")
print(f"point_10/images/train: {len(list((src/'images'/'train').glob('*.png')))} 张")
print(f"point_10/images/val:   {len(list((src/'images'/'val').glob('*.png')))} 张")
print(f"point_10/images/test:  {len(list((src/'images'/'test').glob('*.png')))} 张")