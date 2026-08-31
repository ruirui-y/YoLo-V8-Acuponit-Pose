"""将修复后的标签（边框+关键点）绘制到图片上并保存，用于验证"""
import cv2
import numpy as np
from pathlib import Path

# 配置：要验证哪个 point_N
POINT = "point_10"
SPLIT = "train"  # 可以改 val / test

base = Path("H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint_leg")
img_dir = base / POINT / "images" / SPLIT
label_dir = base / POINT / "labels" / SPLIT
output_dir = base / POINT / "visualize" / SPLIT
output_dir.mkdir(parents=True, exist_ok=True)

# 关键点颜色列表（最多支持 13 个）
COLORS = [
    (0, 0, 255),    # 0: 红
    (255, 0, 0),    # 1: 蓝
    (0, 255, 255),  # 2: 黄
    (0, 255, 0),    # 3: 绿
    (255, 0, 255),  # 4: 紫
    (255, 255, 0),  # 5: 青
    (128, 0, 255),  # 6
    (255, 128, 0),  # 7
    (0, 128, 255),  # 8
    (128, 128, 0),  # 9
    (128, 0, 128),  # 10
    (0, 128, 128),  # 11
    (128, 128, 128),# 12
]

print(f"可视化 {POINT}/{SPLIT} ...")

img_files = sorted(img_dir.glob("*.png"))
for img_path in img_files:
    stem = img_path.stem
    label_path = label_dir / f"{stem}.txt"

    if not label_path.exists():
        print(f"  ⚠️ 跳过 {stem}：无标签")
        continue

    # 读取图片（保持原始尺寸）
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        continue
    H, W = img.shape[:2]
    vis = img[..., :3].copy()  # 取 RGB 用于显示

    # 读取标签
    with open(label_path) as f:
        line = f.readline().strip()
    parts = list(map(float, line.split()))

    cls = int(parts[0])
    cx, cy, bw, bh = parts[1:5]

    # ---- 画边框 ----
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # ---- 画关键点 ----
    kpts = np.array(parts[5:]).reshape(-1, 3)
    for i, (kx, ky, v) in enumerate(kpts):
        px = int(kx * W)
        py = int(ky * H)
        color = COLORS[i % len(COLORS)]
        # 可见（v=2）画实心圆，不可见（v=0）画空心圆
        if v > 0:
            cv2.circle(vis, (px, py), 5, color, -1)
        else:
            cv2.circle(vis, (px, py), 5, color, 2)
        cv2.putText(vis, str(i), (px + 6, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # ---- 保存可视化图 ----
    out_path = output_dir / f"{stem}.jpg"
    cv2.imwrite(str(out_path), vis)

print(f"\n✅ 完成！可视化图保存在: {output_dir}")
print(f"共生成 {len(list(output_dir.glob('*.jpg')))} 张")