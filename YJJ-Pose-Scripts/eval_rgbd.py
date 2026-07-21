from ultralytics import YOLO

def main():
    model = YOLO('runs/pose/train_pose_rgbd_4ch/weights/best.pt', task='pose')
    #metrics = model.val(data='datasets/acupoint.yaml', split='test', imgsz=640, batch=4)
    metrics = model.val(data='datasets/acupoint_leg.yaml', split='test', imgsz=640, batch=4)

    print('=== RGBD 4通道 测试集成绩 ===')
    print(f'Box mAP50:     {metrics.box.map50:.4f}')
    print(f'Box mAP50-95:  {metrics.box.map:.4f}')
    print(f'Pose mAP50:    {metrics.pose.map50:.4f}')
    print(f'Pose mAP50-95: {metrics.pose.map:.4f}')

if __name__ == '__main__':
    main()