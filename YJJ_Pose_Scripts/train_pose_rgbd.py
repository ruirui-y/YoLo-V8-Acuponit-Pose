"""直接使用Python API训练 RGBD-Pose (关键点) 模型"""
from ultralytics import YOLO
import torch

def main():
    print("=" * 60)
    print("🚀 【试训版】RGBD 4通道 Pose(关键点) 模型 🚀")
    print("=" * 60)

    # 1. 加载模型
    print("\n1. 加载模型...")
    model = YOLO('yolov8n-pose_4ch_perfect.pt', task='pose')

    # 验证模型是 4 通道
    first_layer = model.model.model[0].conv
    print(f"第一层卷积输入通道: {first_layer.weight.shape[1]}")
    assert first_layer.weight.shape[1] == 4, "❌ 模型不是4通道！"
    print("✓ 模型确认为 4 通道")

    # 2. 开始试训 (先跑10轮确认没问题)
    print("\n2. 开始试训...")

    try:
        results = model.train(
            task='pose',
            #data='datasets/acupoint.yaml',
            data='datasets/acupoint_leg.yaml',

            # 【关键】先试训10轮！没问题再改成300
            epochs=120,

            imgsz=640,
            batch=4,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            name='train_pose_rgbd_4ch',
            project='runs/pose',

            patience=0,
            save=True,
            plots=True,
            verbose=True,
            workers=0,
            cache=False,
            amp=False,

            # 【开启安全的数据增强】对抗过拟合！
            mosaic=0.0,  # 绝对保持 0！(拼图会破坏腹部的整体拓扑结构)
            mixup=0.0,  # 绝对保持 0！(图像混合会干扰深度图通道)
            copy_paste=0.0,  # 绝对保持 0！

            # --- 下面这些可以安全开启，逼模型泛化 ---
            fliplr=0.5,  # 开启50%概率的左右翻转！(因为你已经写了 flip_idx，现在非常安全)
            flipud=0.0,  # 保持 0 (人不会倒着长)
            degrees=10.0,  # 允许正负 10 度的微小旋转 (模拟人躺得有点歪)
            translate=0.1,  # 允许 10% 的平移 (防止模型死记硬背肚脐在正中心)
            scale=0.2,  # 允许放大缩小 20% (模拟镜头远近变化)
            shear=0.0,  # 保持 0 (避免严重形变)
            perspective=0.0,  # 保持 0 (避免3D透视形变破坏深度计算)

            # 色彩增强保持 0 (防止底层库在处理 4 通道色彩空间转换时把深度图搞坏)
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,

            # 优化器 (小目标用稍小的学习率更稳)
            lr0=0.0005,     # 从0.001降到0.0005
            lrf=0.01,
            warmup_epochs=10, # 热身稍长一点
        )

        print("\n" + "=" * 60)
        print("✅ 试训完成！")
        print("=" * 60)
        print("\n👉 下一步：")
        print("   1. 查看 runs/pose/train_pose_rgbd_10ep_test/results.png")
        print("   2. 确认 box_loss 和 pose_loss 正常下降")
        print("   3. 没问题的话，把 epochs 改成 300 正式训练！")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()