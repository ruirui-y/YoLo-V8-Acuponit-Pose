import cv2
from ultralytics import YOLO

# 1. 加载你的最佳权重
model = YOLO(r"runs\detect\train_rgbd_python_api4\weights\best.pt")

# 2. 读取 4 通道图像
img_path = r"H:\YJJ\Yolo_RGBD\yolov8-rgbd-detection\datasets\acupoint\images\val\20250930_103_2_0045.png"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

if img is None:
    print("图没读到，检查下路径对不对！")
else:
    print(f"成功读取图像，当前矩阵形状为: {img.shape}")

    # 3. 直接喂给模型，并强制最低及格线 conf=0.01
    results = model(img, conf=0.01)

    # 4. 🔑 核心排查点：获取框的数量！
    detected_boxes = results[0].boxes
    num_boxes = len(detected_boxes)

    print("\n" + "=" * 50)
    print(f"⚠️ 诊断报告：在 conf=0.01 的极低标准下，模型一共画了 【{num_boxes}】 个框！")
    print("=" * 50 + "\n")

    if num_boxes > 0:
        results[0].save("predict_result.png")
        print("🎉 图片已保存，快去项目根目录下看看 predict_result.png 里画成了什么鬼样子！")
    else:
        print("❌ 依然是 0 个框！这说明连瞎蒙都蒙不出来，网络的权重极大概率已经崩塌或者特征彻底丢失了。")