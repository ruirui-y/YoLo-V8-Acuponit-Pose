import cv2
import numpy as np

# 替换为你合成的图片路径
img_path = "H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint/images/val/20250930_103_2_0045.png"

# 【关键】必须用 IMREAD_UNCHANGED 读取，否则 OpenCV 会自动转成 3 通道
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print(f"错误：找不到图片 {img_path}")
else:
    # 打印基本信息
    print(f"图片形状 (H, W, C)：{img.shape}")
    print(f"通道数：{img.shape[2] if img.ndim == 3 else 1}")
    print(f"数据类型：{img.dtype}")

    # 拆分通道并保存（方便你确认每个通道的内容）
    if img.ndim == 3 and img.shape[2] == 4:
        print("\n✅ 这是 4 通道 RGBD 图片！")
        # 拆分通道（注意：OpenCV 读取的是 BGR 顺序，第 4 通道是深度）
        b, g, r, d = cv2.split(img)
        # 保存每个通道的可视化图
        cv2.imwrite("check_blue.png", b)
        cv2.imwrite("check_green.png", g)
        cv2.imwrite("check_red.png", r)
        cv2.imwrite("check_depth.png", d)
        print("已保存各通道的可视化图片：check_blue/green/red/depth.png")
    elif img.ndim == 3 and img.shape[2] == 3:
        print("\n❌ 这是 3 通道 RGB 图片，不是 4 通道 RGBD！")
    else:
        print(f"\n❌ 通道数异常：{img.shape[2] if img.ndim == 3 else 1}")