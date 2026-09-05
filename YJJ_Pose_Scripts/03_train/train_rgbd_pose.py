"""RGBD 4通道 Pose(关键点) 训练脚本 (参数化: --data/--fliplr/--patience)"""
import argparse
from ultralytics import YOLO
import torch


def parse_args():
    p = argparse.ArgumentParser(description="训练 RGBD Pose (4通道) 模型")
    p.add_argument('--data', required=True, help='数据配置 yaml 绝对路径')
    p.add_argument('--fliplr', required=True, type=float, help='左右翻转概率 (0.0 关闭)')
    p.add_argument('--patience', required=True, type=int, help='早停耐心轮数 (0 关闭早停)')
    p.add_argument('--weights', required=True, help='模型初始化权重 .pt 路径')
    p.add_argument('--name', default='train_pose_rgbd_4ch',
                   help='训练输出实验名 (默认 train_pose_rgbd_4ch)')
    return p.parse_args()


def main():
    args = parse_args()
    print("=" * 60)
    print("RGBD 4通道 Pose(关键点) 训练")
    print("=" * 60)

    # 1. 加载模型
    print("\n1. 加载模型...")
    model = YOLO(args.weights, task='pose')

    # 验证模型是 4 通道
    first_layer = model.model.model[0].conv
    print(f"第一层卷积输入通道: {first_layer.weight.shape[1]}")
    assert first_layer.weight.shape[1] == 4, "❌ 模型不是4通道！"
    print("✓ 模型确认为 4 通道")

    # 2. 开始训练
    print("\n2. 开始训练...")

    try:
        results = model.train(
            task='pose',
            data=args.data,

            # 训练轮数
            epochs=120,

            imgsz=640,
            batch=4,
            device='cuda:0' if torch.cuda.is_available() else 'cpu',
            name=args.name,
            project='runs/pose',

            patience=args.patience,
            save=True,
            plots=True,
            verbose=True,
            workers=0,
            cache=False,
            amp=False,

            # 【开启安全的数据增强】对抗过拟合！
            mosaic=0.0,  # 保持 0 (拼图会破坏人体/目标整体拓扑结构)
            mixup=0.0,  # 绝对保持 0！(图像混合会干扰深度图通道)
            copy_paste=0.0,  # 绝对保持 0！

            # --- 下面这些可以安全开启，逼模型泛化 ---
            fliplr=args.fliplr,  # 左右翻转概率 (hand 无 flip_idx 时设 0.0)
            flipud=0.0,  # 保持 0 (人不会倒着长)
            degrees=10.0,  # 允许正负 10 度的微小旋转 (模拟人躺得有点歪)
            translate=0.1,  # 允许 10% 的平移 (避免模型死记固定位置)
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

        print("\n训练完成")
        print(f"结果目录: {model.trainer.save_dir}")

    except Exception as e:
        print(f"\n❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == '__main__':
    main()