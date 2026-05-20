import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def calculate_iou(box1, box2):
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def predict_with_exact_training_match(
        model_path: str,
        img_path: str,
        gt_label_path: str = None,
        conf_threshold: float = 0.5,
        save_path: str = "pose_final_match.jpg"
) -> dict:
    """
    完美匹配版：和你的训练脚本100%一致的预处理
    """
    model = YOLO(model_path, task='pose')
    img_path = Path(img_path)

    # 读取4通道RGBD图片
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    assert img.ndim == 3 and img.shape[2] == 4, "必须是4通道RGBD图片"
    orig_H, orig_W = img.shape[:2]
    vis_img = img[..., :3].copy()
    result_dict = {
        "image_size": {"width": orig_W, "height": orig_H},
        "ground_truth": {"box": [], "keypoints": []},
        "prediction": {"box": [], "keypoints": []},
        "metrics": {"box_iou": 0.0, "keypoint_errors": [], "mean_keypoint_error": 0.0}
    }

    # ================== 读取真实标签 ==================
    if gt_label_path is not None:
        label_path = Path(gt_label_path)
    else:
        label_path = img_path.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")

    gt_box = None
    gt_kpts = None

    if label_path.exists():
        with open(label_path, 'r') as f:
            line = f.readline().strip()
        parts = list(map(float, line.split()))
        cls, cx, cy, bw, bh = parts[:5]
        kpts = np.array(parts[5:]).reshape(-1, 3)

        # 归一化转像素
        x1 = int((cx - bw / 2) * orig_W)
        y1 = int((cy - bh / 2) * orig_H)
        x2 = int((cx + bw / 2) * orig_W)
        y2 = int((cy + bh / 2) * orig_H)
        gt_box = [x1, y1, x2, y2]

        gt_kpts = []
        for i, (kx, ky, v) in enumerate(kpts):
            kx_px = int(kx * orig_W)
            ky_px = int(ky * orig_H)
            gt_kpts.append([kx_px, ky_px, v])
            cv2.circle(vis_img, (kx_px, ky_px), 7, (255, 0, 255), -1)
            cv2.putText(vis_img, f"GT{i}", (kx_px + 8, ky_px), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 255, 255), 3)

        result_dict["ground_truth"]["box"] = gt_box
        result_dict["ground_truth"]["keypoints"] = gt_kpts
        print("=" * 60)
        print(f"✅ 读取到真实标签")
        print(f"真实大框: {gt_box}")
        for i, kpt in enumerate(gt_kpts):
            print(f"真实关键点{i}: {kpt[:2]}")

    # ================== 【核心】完美匹配训练的推理 ==================
    results = model(
        img,
        conf=conf_threshold,
        # 🔑 完美匹配你的训练脚本
        imgsz=640,  # 和训练时一致
        rect=False,  # 强制正方形，和训练默认一致
        augment=False,  # 推理不开增强
    )
    result = results[0]

    # ================== 解析预测结果 ==================
    pred_box = None
    pred_kpts = None
    if result.boxes is not None and len(result.boxes) > 0:
        x1, y1, x2, y2 = map(int, result.boxes.xyxy[0])
        pred_box = [x1, y1, x2, y2]
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        pred_kpts = []
        if result.keypoints is not None and len(result.keypoints) > 0:
            for i, (kx, ky, conf) in enumerate(result.keypoints.data[0]):
                kx_px, ky_px = int(kx), int(ky)
                pred_kpts.append([kx_px, ky_px, float(conf)])
                color = (0, 0, 255) if i == 0 else (255, 0, 0) if i == 1 else (0, 255, 255)
                cv2.circle(vis_img, (kx_px, ky_px), 5, color, -1)
                cv2.putText(vis_img, f"pt{i}", (kx_px + 5, ky_px - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        result_dict["prediction"]["box"] = pred_box
        result_dict["prediction"]["keypoints"] = pred_kpts
        print("=" * 60)
        print(f"✅ 模型预测结果 (完美匹配训练)")
        print(f"预测大框: {pred_box}")
        for i, kpt in enumerate(pred_kpts):
            print(f"预测关键点{i}: {kpt[:2]}, 置信度: {kpt[2]:.4f}")

    # ================== 计算量化差距 ==================
    if gt_box is not None and pred_box is not None:
        print("\n" + "=" * 60)
        print("📊 完美匹配后量化差距分析")
        print("=" * 60)

        iou = calculate_iou(gt_box, pred_box)
        result_dict["metrics"]["box_iou"] = iou
        print(f"腹部大框 IOU: {iou:.4f}")

        if gt_kpts is not None and pred_kpts is not None and len(gt_kpts) == len(pred_kpts):
            kpt_errors = []
            for i in range(len(gt_kpts)):
                gt_x, gt_y = gt_kpts[i][:2]
                pred_x, pred_y = pred_kpts[i][:2]
                error = np.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
                kpt_errors.append(error)
                result_dict["metrics"]["keypoint_errors"].append(error)
                print(f"关键点{i} 像素误差: {error:.2f} px")

            mean_error = np.mean(kpt_errors)
            result_dict["metrics"]["mean_keypoint_error"] = mean_error
            print(f"\n关键点平均像素误差: {mean_error:.2f} px")
            if mean_error < 5:
                print("  🎉 顶级精度！")

    cv2.imwrite(save_path, vis_img)
    print("\n" + "=" * 60)
    print(f"✅ 完美匹配对比图已保存: {save_path}")

    return result_dict


if __name__ == '__main__':
    compare_result = predict_with_exact_training_match(
        model_path='runs/pose/train_pose_rgbd_10ep_test2/weights/best.pt',
        img_path="H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint/images/train/20250915_103_0_0043.png",
        gt_label_path="H:/YJJ/Yolo_RGBD/yolov8-rgbd-detection/datasets/acupoint/labels/train/20250915_103_0_0043.txt",
        conf_threshold=0.5,
        save_path="pose_final_match.jpg"
    )