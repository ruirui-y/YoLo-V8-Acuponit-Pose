"""从 last.pt 续训 RGBD 模型到 120 轮（patience=0，跑满）"""
from ultralytics import YOLO
import torch

def main():
    # 加载上次训练的第 31 轮权重（不是 best.pt）
    model = YOLO('runs/pose/train_pose_rgbd_4ch/weights/last.pt', task='pose')

    # 验证仍然是 4 通道
    first_layer = model.model.model[0].conv
    print(f"第一层卷积输入通道: {first_layer.weight.shape[1]}")
    assert first_layer.weight.shape[1] == 4, "❌ 模型不是4通道！"
    print("✓ 模型确认为 4 通道，继续训练...")

    # 续训 -- patience=0 表示跑满 epochs
    results = model.train(
        task='pose',
        data='datasets/acupoint.yaml',

        # 跑满 120 轮（续训 = 从第 31 轮继续，直到 120）
        epochs=120,

        imgsz=640,
        batch=4,
        device='cuda:0' if torch.cuda.is_available() else 'cpu',

        # 改个新名字，不覆盖旧结果
        name='train_pose_rgbd_4ch_full120',
        project='runs/pose',

        patience=0,          # 🔑 关键改动：不早停，跑满！
        save=True,
        plots=True,
        verbose=True,
        workers=0,
        cache=False,
        amp=False,

        # 数据增强和上次完全一致
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        fliplr=0.5,
        flipud=0.0,
        degrees=10.0,
        translate=0.1,
        scale=0.2,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        lr0=0.0005,
        lrf=0.01,
        warmup_epochs=10,
    )

    print(f"\n✅ 续训完成！结果保存在: runs/pose/train_pose_rgbd_4ch_full120/")

if __name__ == '__main__':
    main()