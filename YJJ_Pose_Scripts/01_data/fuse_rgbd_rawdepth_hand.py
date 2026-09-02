"""
生成「raw npy Depth 版」手部 RGBD 4 通道 PNG。

- RGB 三通道: 直接取自原始 color_*.jpg (cv2.imread -> BGR), 像素逐字节不变
- 第4通道 (alpha=Depth): 取自原始 depth_*.npy (uint16), 不再经过 depth_show*.jpg
  - invalid: depth == 0 或 depth == 65535 -> 第4通道 = 0 (0 专门保留给 invalid)
  - valid  : clip 到 [low, high], 再线性映射  low->1, high->255 (严格单调)
- 严格按 timestamp 与原始 RGB 一一对应, 文件名保持 color_<ts>.png
- 输出到独立目录, 不覆盖任何旧数据

映射:
    c   = clip(d, low, high)
    ch4 = 1 + (c - low) * 254 / (high - low)     # low->1, high->255, 中间线性
"""
import argparse
import numpy as np
import cv2
from pathlib import Path


def ts_of(name: str, prefix: str):
    s = Path(name).stem
    return s[len(prefix):] if s.startswith(prefix) else None


def main():
    ap = argparse.ArgumentParser(description="raw npy Depth 版手部 RGBD 生成")
    ap.add_argument("--image-dir", default=r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_image")
    ap.add_argument("--npy-dir", default=r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_npy")
    ap.add_argument("--out-dir", default=r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_rawdepth")
    ap.add_argument("--low", type=float, default=1100.0, help="mm, 低于此值 clip 到 low")
    ap.add_argument("--high", type=float, default=1850.0, help="mm, 高于此值 clip 到 high")
    ap.add_argument("--invalid0", type=int, default=0)
    ap.add_argument("--invalid65535", type=int, default=65535)
    args = ap.parse_args()

    img_dir = Path(args.image_dir)
    npy_dir = Path(args.npy_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    img_map = {ts_of(p.name, "color_"): p for p in img_dir.iterdir()
               if p.is_file() and ts_of(p.name, "color_")}
    npy_map = {ts_of(p.name, "depth_"): p for p in npy_dir.iterdir()
               if p.is_file() and p.suffix == ".npy" and ts_of(p.name, "depth_")}
    matched = sorted(set(img_map) & set(npy_map))
    missing_rgb = sorted(set(npy_map) - set(img_map))
    missing_npy = sorted(set(img_map) - set(npy_map))
    print(f"[匹配] RGB={len(img_map)} NPY={len(npy_map)} 匹配={len(matched)} "
          f"缺RGB的NPY={len(missing_rgb)} 缺NPY的RGB={len(missing_npy)}")

    low, high = args.low, args.high
    scale = 254.0 / (high - low)
    ok = 0
    for ts in matched:
        color = cv2.imread(str(img_map[ts]), cv2.IMREAD_UNCHANGED)
        if color is None:
            print("  跳过 读图失败:", ts)
            continue
        if color.ndim == 2:
            color = cv2.cvtColor(color, cv2.COLOR_GRAY2BGR)
        bgr = color[:, :, :3].copy()  # 原始 RGB 像素逐字节保留 (BGR 顺序, 与既有 hand RGBD 一致)

        d = np.load(npy_map[ts])
        if d.ndim == 3:
            d = d[..., 0]
        d = d.astype(np.int64)
        if d.shape[:2] != bgr.shape[:2]:
            d = cv2.resize(d.astype(np.float32), (bgr.shape[1], bgr.shape[0])).astype(np.int64)

        invalid = (d == args.invalid0) | (d == args.invalid65535)
        c = np.clip(d.astype(np.float64), low, high)
        ch4 = np.zeros(d.shape, dtype=np.float64)
        ch4[~invalid] = 1.0 + (c[~invalid] - low) * scale
        ch4 = np.clip(ch4, 0, 255).astype(np.uint8)

        out = np.dstack([bgr, ch4])  # BGRA, shape (H,W,4), uint8
        out_path = out_dir / (img_map[ts].stem + ".png")
        cv2.imwrite(str(out_path), out)
        ok += 1

    print(f"[生成] 成功 {ok}/{len(matched)} -> {out_dir}")


if __name__ == "__main__":
    main()
