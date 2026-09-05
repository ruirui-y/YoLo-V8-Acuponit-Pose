"""
session1_200146 手部数据 RGBD 四通道融合脚本
- 复用 scripts/fuse_rgb_depth.py 的 fuse_pair() 融合逻辑
- 配对方式改为【按文件时间戳】匹配（out_image 前缀 color_，out_depth 前缀 depth_show）
- 仅当 RGB + Depth + Label 三者时间戳都存在时才视为“可用样本”并融合
- 不修改原始 out_image / out_depth / labels，只向 out/ 写 4 通道 PNG

用法:
python scripts/fuse_rgb_depth_session1.py
"""
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# 复用原脚本的融合函数（RGB 读 + 深度读 + 4 通道构造 + uint8 归一化）
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from fuse_rgb_depth import fuse_pair  # noqa: E402

RGB_DIR = Path(r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_image")
DEPTH_DIR = Path(r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_depth")
LABEL_DIR = Path(r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/labels")
OUT_DIR = Path(r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/out")

RGB_PREFIX = "color_"
DEPTH_PREFIX = "depth_show"


def ts_of(name: str, prefix: str) -> str | None:
    """从文件名提取时间戳部分；去掉 prefix 与扩展名。"""
    stem = Path(name).stem
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return None


def collect(dir_path: Path, prefix: str):
    return {ts_of(p.name, prefix): p for p in dir_path.iterdir()
            if p.is_file() and ts_of(p.name, prefix) is not None}


def main():
    rgb_map = collect(RGB_DIR, RGB_PREFIX)
    depth_map = collect(DEPTH_DIR, DEPTH_PREFIX)
    label_map = collect(LABEL_DIR, RGB_PREFIX)  # labels 前缀也是 color_

    print(f"[统计] out_image(RGB) : {len(rgb_map)}")
    print(f"[统计] out_depth(Depth): {len(depth_map)}")
    print(f"[统计] labels         : {len(label_map)}")

    # 实际可用样本 = RGB ∩ Depth ∩ Label
    rgb_depth = set(rgb_map) & set(depth_map)
    usable = rgb_depth & set(label_map)
    missing_depth = set(rgb_map) - set(depth_map)
    missing_label = rgb_depth - set(label_map)

    print(f"[统计] RGB∩Depth      : {len(rgb_depth)}")
    print(f"[统计] 实际可用样本   : {len(usable)}  (RGB+Depth+Label 三者齐备)")
    if missing_depth:
        print(f"[警告] 缺 Depth 的时间戳数: {len(missing_depth)} -> {sorted(missing_depth)[:5]} ...")
    if missing_label:
        print(f"[警告] 缺 Label 的时间戳数: {len(missing_label)} -> {sorted(missing_label)[:5]} ...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done, failed = 0, []
    for ts in sorted(usable):
        rgb_path = rgb_map[ts]
        depth_path = depth_map[ts]
        out_path = OUT_DIR / (RGB_PREFIX + ts + ".png")
        try:
            fuse_pair(rgb_path, depth_path, out_path, depth_type="uint8")
            done += 1
        except Exception as e:  # noqa: BLE001
            failed.append((ts, str(e)))

    print(f"\n[融合] 已写出 4 通道 PNG: {done} 张 -> {OUT_DIR}")
    if failed:
        print(f"[融合] 失败 {len(failed)} 张:")
        for ts, err in failed:
            print(f"   - {ts}: {err}")

    # 70/15/15 划分预览（仅展示，不写文件）
    n = len(usable)
    n_train = round(n * 0.70)
    n_val = round(n * 0.15)
    n_test = n - n_train - n_val
    print("\n[划分预览 70/15/15]")
    print(f"  train={n_train}  val={n_val}  test={n_test}  (合计 {n})")


if __name__ == "__main__":
    main()
