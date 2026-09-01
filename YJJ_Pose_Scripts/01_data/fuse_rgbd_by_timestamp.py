# 用于将 RGB 图像与 uint8 Depth 图像按时间戳配对，
# 合成为 YOLO RGBD 使用的 4 通道 PNG。
#
# 配对规则（按时间戳，非按整文件名）：
#   color_<timestamp>.jpg
#   对应
#   depth_show<timestamp>.jpg
#
# 复用 scripts/fuse_rgb_depth.py 的 fuse_pair() 完成实际的 4 通道合成。
#
# 用法:
# python YJJ_Pose_Scripts/01_data/fuse_rgbd_by_timestamp.py \
#     --rgb_dir    "H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_image" \
#     --depth_dir  "H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_depth" \
#     --output_dir "H:/YJJ/Yolo_RGBD/Resource/session1_200146/out"
#
# 可选: --label_dir 指定标签目录时，只融合 RGB+Depth+Label 三者时间戳齐备的样本。
import argparse
import os
import sys
from pathlib import Path

# 复用原脚本的融合函数（RGB 读 + 深度读 + 4 通道构造 + uint8 归一化）
# 定位 scripts/fuse_rgb_depth.py：本文件位于 <repo>/YJJ_Pose_Scripts/01_data/，
# 向上两级为项目根，scripts 在其下。
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../YJJ_Pose_Scripts/01_data
_REPO_ROOT = _SCRIPT_DIR.parents[1]                   # .../yolov8_rgbd_detection
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from fuse_rgb_depth import fuse_pair  # noqa: E402

# 时间戳配对用的文件名前缀（数据集约定，非绝对路径）
RGB_PREFIX = "color_"
DEPTH_PREFIX = "depth_show"


def ts_of(name: str, prefix: str):
    """从文件名提取时间戳部分：去掉 prefix 与扩展名。"""
    stem = Path(name).stem
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return None


def collect(dir_path: Path, prefix: str):
    return {ts_of(p.name, prefix): p for p in dir_path.iterdir()
            if p.is_file() and ts_of(p.name, prefix) is not None}


def main():
    parser = argparse.ArgumentParser(
        description="将 RGB 与 uint8 Depth 按时间戳配对，合成为 YOLO RGBD 的 4 通道 PNG")
    parser.add_argument("--rgb_dir", required=True, help="RGB 图像目录（color_<timestamp>.jpg）")
    parser.add_argument("--depth_dir", required=True, help="Depth 图像目录（depth_show<timestamp>.jpg）")
    parser.add_argument("--output_dir", required=True, help="4 通道 PNG 输出目录")
    parser.add_argument("--label_dir", default=None,
                        help="可选：标签目录（color_<timestamp>.txt）。提供时只融合 RGB+Depth+Label 齐备的样本")
    parser.add_argument("--depth_type", choices=["uint8", "uint16"], default="uint8",
                        help="深度通道保存类型，默认 uint8（4 通道 PNG）")
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_map = collect(rgb_dir, RGB_PREFIX)
    depth_map = collect(depth_dir, DEPTH_PREFIX)
    label_map = collect(Path(args.label_dir), RGB_PREFIX) if args.label_dir else {}

    print(f"[统计] RGB   : {len(rgb_map)}")
    print(f"[统计] Depth : {len(depth_map)}")
    if args.label_dir:
        print(f"[统计] Label : {len(label_map)}")

    # 候选 = RGB ∩ Depth；若提供 label 则进一步取交集
    usable = set(rgb_map) & set(depth_map)
    if args.label_dir:
        usable = usable & set(label_map)

    missing_depth = set(rgb_map) - set(depth_map)
    if missing_depth:
        print(f"[警告] 缺 Depth 的时间戳数: {len(missing_depth)}")
    if args.label_dir:
        missing_label = (set(rgb_map) & set(depth_map)) - set(label_map)
        if missing_label:
            print(f"[警告] 缺 Label 的时间戳数: {len(missing_label)}")

    print(f"[统计] 将融合样本数: {len(usable)}")

    done, failed = 0, []
    for ts in sorted(usable):
        out_path = out_dir / (RGB_PREFIX + ts + ".png")
        try:
            fuse_pair(rgb_map[ts], depth_map[ts], out_path, depth_type=args.depth_type)
            done += 1
        except Exception as e:  # noqa: BLE001
            failed.append((ts, str(e)))

    print(f"[融合] 已写出 4 通道 PNG: {done} 张 -> {out_dir}")
    if failed:
        print(f"[融合] 失败 {len(failed)} 张:")
        for ts, err in failed:
            print(f"   - {ts}: {err}")


if __name__ == "__main__":
    main()
