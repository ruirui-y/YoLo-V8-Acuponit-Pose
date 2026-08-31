"""统计腿部标签的关键点数量分布，并按关键点数分类到 acupoint_leg/point_N/ 下"""
from pathlib import Path
import shutil
import os

# 路径配置
src_images = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/mark/rgbd_images")
src_labels = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/mark/labels")
output_base = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint_leg")

# ---- 第1步：统计每个标签的关键点数量 ----
from collections import Counter

counter = Counter()  # {关键点数: 文件数}
file_list = {}       # {关键点数: [(stem, img_path, label_path), ...]}

for label_file in sorted(src_labels.glob("20250930_103_0_*.txt")):
    with open(label_file) as f:
        line = f.readline().strip()
    parts = line.split()
    if len(parts) < 5:
        continue
    # 计算关键点数: 总列数 - cls(1) - box(4) = kpt_cols, 再 /3
    kpt_cols = len(parts) - 5
    if kpt_cols % 3 != 0:
        print(f"⚠️ 跳过 {label_file.name}：关键点列数 {kpt_cols} 不是3的倍数")
        continue
    num_kpts = kpt_cols // 3
    counter[num_kpts] += 1

    stem = label_file.stem
    img_path = src_images / f"{stem}.png"
    if not img_path.exists():
        print(f"⚠️ 跳过 {stem}：图片不存在")
        continue

    if num_kpts not in file_list:
        file_list[num_kpts] = []
    file_list[num_kpts].append((stem, img_path, label_file))

# 打印统计结果
print("=" * 60)
print("📊 腿部标签关键点数量分布统计")
print("=" * 60)
for kpts in sorted(counter.keys()):
    print(f"  {kpts} 个关键点（{kpts*3+5:2d} 列）: {counter[kpts]:4d} 个文件")
print("-" * 60)
print(f"  总计: {sum(counter.values())} 个文件")
print("=" * 60)

# ---- 第2步：按关键点数分类复制到不同目录 ----
print("\n📁 开始复制到分类目录...")
for kpts, files in file_list.items():
    point_dir = output_base / f"point_{kpts}"
    for split in ["train", "val", "test"]:
        (point_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (point_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for stem, img_path, label_path in files:
        dst_img = point_dir / "images" / "train" / f"{stem}.png"
        dst_label = point_dir / "labels" / "train" / f"{stem}.txt"
        shutil.copy2(img_path, dst_img)
        shutil.copy2(label_path, dst_label)

    count = len(files)
    print(f"  ✅ point_{kpts}: {count} 个文件已复制到 {point_dir}")

print("\n✅ 全部完成！")
print("\n提示：后续需要重新划分 train/val/test（当前全部在 train 下），"
      "可以用 split_dataset.py 模板修改后运行。")