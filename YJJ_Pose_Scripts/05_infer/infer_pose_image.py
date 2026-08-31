import cv2
import numpy as np
from ultralytics import YOLO

# 加载训练好的4通道模型
model = YOLO('runs/pose/train_pose_rgbd_10ep_test2/weights/best.pt', task='pose')

# 读取4通道RGBD测试图
img_path = "H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint/images/val/20250930_103_2_0045.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
assert img.shape[2] == 4, "必须是4通道RGBD图片"

# 推理
results = model(img, conf=0.5)

# 可视化结果
result = results[0]
vis_img = img[..., :3].copy()  # 取RGB通道显示
H, W = vis_img.shape[:2]

# 画腹部大框
for box in result.boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# 画穴位关键点
for kpts in result.keypoints:
    for i, (kx, ky, conf) in enumerate(kpts.data[0]):
        kx_px, ky_px = int(kx), int(ky)
        # 不同点用不同颜色
        color = (0, 0, 255) if i == 0 else (255, 0, 0) if i == 1 else (0, 255, 255)
        cv2.circle(vis_img, (kx_px, ky_px), 5, color, -1)
        cv2.putText(vis_img, f"pt{i}", (kx_px+5, ky_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# 保存结果
cv2.imwrite("pose_result.jpg", vis_img)
print("✅ 推理完成，结果已保存为 pose_result.jpg")