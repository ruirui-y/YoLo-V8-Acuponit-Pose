from ultralytics import YOLO

def run_final_test():
    print("============================================================")
    print(" 终极摸底考试：测试集绝密测验 ")
    print("============================================================")

    # 1. 加载你 300 轮训练出来的最终 best.pt
    # ⚠️ 注意：一定要把下面的路径换成你 300 轮跑完后实际生成的路径！
    # 比如可能是 'runs/pose/train_pose_rgbd_300ep/weights/best.pt'
    model = YOLO('runs/pose/train_pose_rgbd_10ep_test3/weights/best.pt', task='pose')

    # 2. 启动验证模式，并强行指定只考 test 集
    print("\n 开始进行测试集评估...")
    metrics = model.val(
        data='datasets/acupoint.yaml',
        split='test',  # 🔑 核心指令：只跑 test 文件夹！
        imgsz=640,
        batch=4,
        device='cuda:0'
    )

    # 3. 打印最终成绩单
    print("\n============================================================")
    print(" 考试成绩单出炉！")
    print(f" 大肚子框准确率 (Box mAP50): {metrics.box.map50:.4f}")
    print(f" 穴位点准确率 (Pose mAP50): {metrics.pose.map50:.4f}")
    print("============================================================")

if __name__ == '__main__':
    run_final_test()