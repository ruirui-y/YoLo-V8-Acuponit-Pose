from ultralytics import YOLO

def main():
    model = YOLO('runs/pose/train_pose_rgb_3ch/weights/best.pt', task='pose')
    # metrics = model.val(data='datasets/acupoint_rgb.yaml', split='test', imgsz=640, batch=4)
    metrics = model.val(data='datasets/acupoint_leg_rgb.yaml', split='test', imgsz=640, batch=4)

    print('=== RGB 3通道 测试集成绩 ===')
    print(f'Box mAP50:     {metrics.box.map50:.4f}')
    print(f'Box mAP50-95:  {metrics.box.map:.4f}')
    print(f'Pose mAP50:    {metrics.pose.map50:.4f}')
    print(f'Pose mAP50-95: {metrics.pose.map:.4f}')

if __name__ == '__main__':
    main()