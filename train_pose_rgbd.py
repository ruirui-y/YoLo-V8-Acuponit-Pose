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
            data='datasets/acupoint.yaml',

            # 【关键】先试训10轮！没问题再改成300
            epochs=10,

            imgsz=640,
            batch=4,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            name='train_pose_rgbd_10ep_test',  # 试训专用名字
            project='runs/pose',

            patience=0,
            save=True,
            plots=True,
            verbose=True,
            workers=0,
            cache=False,
            amp=False,

            # 【核心】100%禁用所有破坏小目标/关键点的增强
            mosaic=0.0,
            copy_paste=0.0,
            mixup=0.0,
            fliplr=0.0,      # 先完全关闭翻转
            flipud=0.0,
            degrees=0.0,
            translate=0.0,
            scale=0.0,
            shear=0.0,
            perspective=0.0,
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