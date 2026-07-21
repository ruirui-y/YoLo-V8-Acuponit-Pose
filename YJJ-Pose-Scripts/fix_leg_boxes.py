"""根据关键点坐标自动生成边框，修复 point_N 所有标签中无效的 0 0 0 0"""
import numpy as np
from pathlib import Path

# 遍历所有 point_N 目录
base = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint_leg")

MARGIN_RATIO = 0.2  # 边框比关键点范围放大 20%
fixed_count = 0
skipped_count = 0

# 扫描所有 point_N/labels/train、val、test
label_dirs = list(base.glob("point_*/labels/*"))

for label_dir in sorted(label_dirs):
    if not label_dir.is_dir():
        continue
    point_name = label_dir.parent.parent.name
    split_name = label_dir.name
    for f in sorted(label_dir.glob("*.txt")):
        with open(f) as fh:
            line = fh.readline().strip()
        parts = list(map(float, line.split()))

        cls = int(parts[0])
        cx, cy, w, h = parts[1:5]

        # 边框已有效则跳过
        if w > 0 and h > 0:
            skipped_count += 1
            continue

        # 从关键点算边框
        kpts = np.array(parts[5:]).reshape(-1, 3)
        visible_kpts = kpts[kpts[:, 2] > 0]

        if len(visible_kpts) == 0:
            print(f"⚠️ 跳过 {f.name} ({point_name}/{split_name})：没有可见关键点")
            skipped_count += 1
            continue

        xs = visible_kpts[:, 0]
        ys = visible_kpts[:, 1]
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()

        box_w = x_max - x_min
        box_h = y_max - y_min
        margin_x = box_w * MARGIN_RATIO
        margin_y = box_h * MARGIN_RATIO

        new_cx = (x_min + x_max) / 2
        new_cy = (y_min + y_max) / 2
        new_w = box_w + margin_x * 2
        new_h = box_h + margin_y * 2

        new_parts = [str(cls), f"{new_cx:.6f}", f"{new_cy:.6f}", f"{new_w:.6f}", f"{new_h:.6f}"]
        new_parts.extend(f"{v:.6f}" for v in parts[5:])

        with open(f, "w") as fh:
            fh.write(" ".join(new_parts) + "\n")
        fixed_count += 1

print(f"\n✅ 修复边框: {fixed_count} 个")
print(f"⏭️  跳过（已有有效边框）: {skipped_count} 个")
print("\n完成记得清理缓存：")
print("del /q \"H:\\YJJ\\Yolo_RGBD\\yolov8-rgbd-detection\\datasets\\acupoint_leg\\point_10\\labels\\*.cache\"")