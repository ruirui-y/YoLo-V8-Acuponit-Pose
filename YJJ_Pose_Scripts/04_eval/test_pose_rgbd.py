"""通用 Pose test 集评估脚本 (参数化: --weights/--data/--split)

用于论文最终比较 RGB vs RGBD。
- 强制 split='test'（不允许 val），保证 RGB 与 RGBD 基于同一批测试样本对比。
- 输出人类可读成绩 + 一行机器可读 JSON（前缀 POSE_EVAL_JSON）供 GUI 解析。
- 不修改 best.pt，不重新训练。

调用示例（GUI 内部已封装）:
    python YJJ_Pose_Scripts/04_eval/test_pose_rgbd.py \
        --weights runs/pose/train_pose_rgb_3ch2/weights/best.pt \
        --data    dataset/hand/rgb/data_rgb.yaml \
        --split   test
"""
import argparse
import json

from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser(description="Pose test 集评估 (RGB vs RGBD 对比)")
    p.add_argument("--weights", required=True, help="待评估权重 .pt 路径")
    p.add_argument("--data", required=True, help="数据配置 yaml（rgb 或 rgbd 的 data_*.yaml）")
    p.add_argument("--split", default="test", help="评估划分，固定为 test")
    p.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    p.add_argument("--batch", type=int, default=4, help="batch size")
    p.add_argument("--device", default=None, help="设备，默认自动选择 (cuda 优先)")
    return p.parse_args()


def main():
    args = parse_args()
    # 本脚本仅用于 test 集评估，禁止用 val 混淆对比结论
    if args.split != "test":
        raise SystemExit("本脚本仅用于 test 集评估，split 必须为 'test'")

    print("=" * 60)
    print("Pose test 集评估")
    print(f"weights : {args.weights}")
    print(f"data    : {args.data}")
    print(f"split   : {args.split}  (固定 test，不使用 val)")
    print("=" * 60)

    model = YOLO(args.weights, task="pose")

    val_kwargs = dict(
        data=args.data,
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        verbose=False,
        plots=False,
        save_json=False,
    )
    if args.device is not None:
        val_kwargs["device"] = args.device

    metrics = model.val(**val_kwargs)
    d = metrics.results_dict

    box_p = float(d["metrics/precision(B)"])
    box_r = float(d["metrics/recall(B)"])
    box_map50 = float(d["metrics/mAP50(B)"])
    box_map = float(d["metrics/mAP50-95(B)"])
    pose_p = float(d["metrics/precision(P)"])
    pose_r = float(d["metrics/recall(P)"])
    pose_map50 = float(d["metrics/mAP50(P)"])
    pose_map = float(d["metrics/mAP50-95(P)"])

    # 人类可读
    print("\n=== Box (test) ===")
    print(f"P:         {box_p:.4f}")
    print(f"R:         {box_r:.4f}")
    print(f"mAP50:     {box_map50:.4f}")
    print(f"mAP50-95:  {box_map:.4f}")
    print("\n=== Pose (test) ===")
    print(f"P:         {pose_p:.4f}")
    print(f"R:         {pose_r:.4f}")
    print(f"mAP50:     {pose_map50:.4f}")
    print(f"mAP50-95:  {pose_map:.4f}")

    # 机器可读（单行，供 GUI stdout 解析）
    payload = {
        "box_p": box_p,
        "box_r": box_r,
        "box_map50": box_map50,
        "box_map": box_map,
        "pose_p": pose_p,
        "pose_r": pose_r,
        "pose_map50": pose_map50,
        "pose_map": pose_map,
    }
    print("POSE_EVAL_JSON " + json.dumps(payload))


if __name__ == "__main__":
    main()
