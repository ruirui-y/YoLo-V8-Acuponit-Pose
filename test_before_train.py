import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO


def test_simple():
    print("=" * 60)
    print("🧪 简化版测试：2分钟确认核心环节")
    print("=" * 60)

    # ================== 1. 测试模型 ==================
    print("\n[1/3] 测试模型...")
    model = YOLO('yolov8n-pose_4ch_perfect.pt', task='pose')
    first_layer = model.model.model[0].conv
    in_channels = first_layer.weight.shape[1]
    print(f"   模型输入通道数: {in_channels}")
    assert in_channels == 4, "❌ 模型不是4通道！"
    print("   ✅ 模型验证通过")

    # ================== 2. 测试图片和标签 ==================
    print("\n[2/3] 测试图片和标签...")
    # 找一张测试图和对应的标签
    img_dir = Path('datasets/acupoint/images/val')
    label_dir = Path('datasets/acupoint/labels/val')

    img_path = next(img_dir.glob("*.png"))
    label_path = label_dir / (img_path.stem + ".txt")

    print(f"   测试图片: {img_path.name}")
    print(f"   测试标签: {label_path.name}")

    # 读取图片（必须用 IMREAD_UNCHANGED）
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    print(f"   图片形状: {img.shape}")
    assert img.ndim == 3 and img.shape[2] == 4, "❌ 图片不是4通道！"
    print("   ✅ 图片是4通道 RGBD")

    # 读取标签
    with open(label_path) as f:
        line = f.readline().strip()
    parts = list(map(float, line.split()))
    print(f"   标签长度: {len(parts)}")

    # 解析标签
    if len(parts) > 5:
        kpt_part = parts[5:]
        num_kpts = len(kpt_part) // 3
        print(f"   关键点数量: {num_kpts}")
        assert num_kpts == 3, f"❌ 关键点数量不对！期望3，实际{num_kpts}"
    print("   ✅ 标签验证通过")

    # ================== 3. 可视化验证（最直观！） ==================
    print("\n[3/3] 生成可视化验证图...")
    # 拆分 RGB 和深度
    b, g, r, d = cv2.split(img)
    rgb = cv2.merge([r, g, b])  # 转成RGB显示
    depth_colored = cv2.applyColorMap(cv2.normalize(d, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U),
                                      cv2.COLORMAP_JET)

    # 画标签
    vis_img = rgb.copy()
    H, W = vis_img.shape[:2]

    # 画大框
    cls, cx, cy, bw, bh = parts[:5]
    x1 = int((cx - bw / 2) * W)
    y1 = int((cy - bh / 2) * H)
    x2 = int((cx + bw / 2) * W)
    y2 = int((cy + bh / 2) * H)
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 画关键点
    kpts = np.array(kpt_part).reshape(-1, 3)
    for i, (kx, ky, v) in enumerate(kpts):
        kx_px = int(kx * W)
        ky_px = int(ky * H)
        color = (0, 0, 255) if i == 0 else (255, 0, 0) if i == 1 else (0, 255, 255)
        cv2.circle(vis_img, (kx_px, ky_px), 5, color, -1)
        cv2.putText(vis_img, str(i), (kx_px + 5, ky_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 拼接保存
    top_row = np.hstack([vis_img, depth_colored])
    cv2.imwrite('test_visualization.jpg', top_row)
    print("   ✅ 可视化图已生成: test_visualization.jpg")
    print("   👉 请打开这张图，确认：")
    print("      1. 绿色框是否框住了腹部")
    print("      2. 三个彩色点是否在正确的穴位位置")

    print("\n" + "=" * 60)
    print("🎉 所有测试通过！可以放心试训10轮了！")
    print("=" * 60)


if __name__ == '__main__':
    try:
        test_simple()
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback

        traceback.print_exc()