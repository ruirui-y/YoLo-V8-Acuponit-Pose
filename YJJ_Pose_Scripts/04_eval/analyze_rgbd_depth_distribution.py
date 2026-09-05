"""统计完整 RGBD 数据集（train / val / test + overall）中"真正送进模型的第4通道 uint8 Depth"数值分布。

目的：确认当前 Hand test 12 张连续帧中"True Depth 有效区域均值约 170"是整份数据集的常态，
还是仅这 12 张的特殊情况。直接统计模型实际看到的第4通道幅度（已映射后的 uint8 Depth，
非原始 16bit raw / 毫米值 / 1100~1850 映射前数据）。

不做模型推理 / 训练 / 修改数据 / 修改 yaml / 重新生成 RGBD / 修改 1100~1850 映射。

读取：cv2.imread(path, cv2.IMREAD_UNCHANGED)，要求 shape==(H,W,4) 且 dtype==uint8，
否则明确报错并指出文件路径。仅取 depth = image[:, :, 3]，不转 RGB / 不 normalize / 不 resize / 不 blur。

valid / invalid 定义（与消融实验一致）：
  invalid : depth == 0
  valid   : depth > 0

内存控制：Depth 为 uint8(0~255)，逐 split 累积 256-bin histogram（不把所有 valid pixel 存进巨型 list），
可精确计算 min/max/mean/std/percentile。per-image 的 valid mean / valid ratio 数量仅图片级，正常存 float list。
不引入 pandas / scipy 等新依赖；仅用 numpy / cv2 / yaml / json / pathlib / argparse。

输出：终端紧凑统计 + <out>/depth_distribution.json（所有值为 JSON 可序列化 Python 类型）。

调用示例：
  python YJJ_Pose_Scripts/04_eval/analyze_rgbd_depth_distribution.py
  python YJJ_Pose_Scripts/04_eval/analyze_rgbd_depth_distribution.py --rgbd-yaml <path> --splits train val test --out runs/depth_distribution
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---- 默认路径（本机 rawdepth 4ch 数据集；如换机用 --rgbd-yaml 指定）----
_DEFAULT_RGBD_YAML = (
    r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand/rgbd_rawdepth/data_rgbd_rawdepth.yaml"
)
_DEFAULT_OUT = r"runs/depth_distribution"  # runs/ 已被 .gitignore 忽略，作为安全输出目录

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")  # 当前项目实际使用的扩展名（大小写不敏感）


# --------------------------------------------------------------------------- 工具
def _LoadYaml(rgbd_yaml: Path) -> dict:
    """读取 RGBD data yaml，返回解析后的 dict。"""
    if not rgbd_yaml.exists():
        raise FileNotFoundError(f"RGBD yaml 不存在: {rgbd_yaml}")
    return yaml.safe_load(rgbd_yaml.read_text(encoding="utf-8"))


def _ResolveSplitDir(cfg: dict, split: str) -> Path:
    """解析 train / val / test 对应图片目录（遵循本仓库 yaml 约定：path + 各 split 子目录）。

    优先读 yaml 中 train/val/test 字段；若该 split 字段缺失则回退为 images/<split>。
    """
    if "path" not in cfg:
        raise ValueError(f"yaml 缺少 path: 字段，无法解析 split 目录: {cfg}")
    ds_root = Path(cfg["path"])
    rel = cfg.get(split, f"images/{split}")  # 真实 Hand yaml 已显式给出 train/val/test 子目录
    return ds_root / rel


def _ReadRgba(path: Path) -> np.ndarray:
    """读取 4 通道融合图（BGRA/4ch RGBD），返回 (H, W, 4) uint8；不满足则明确报错并指出路径。"""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    if im.ndim != 3 or im.shape[2] != 4:
        raise ValueError(f"图像不是 4 通道（shape={im.shape if im.ndim == 3 else im.shape}）: {path}")
    if im.dtype != np.uint8:
        raise ValueError(f"图像 dtype 非 uint8（dtype={im.dtype}）: {path}")
    return im


def _PercentileFromHistogram(hist: np.ndarray, q: float) -> float:
    """从 256-bin histogram（bin0=invalid，已被 _AnalyzeSplit 排除）计算 valid Depth 分位数（CDF 下侧反函数）。

    返回满足"至少 (q/100) 比例 valid 像素 <= 该值"的最小整数 bin 值（单位宽度分箱，误差 <=1）。
    valid 像素总数 0 时返回 0.0（调用方需在此之前保证 split 存在 valid 像素）。
    """
    total = int(hist[1:].sum())
    if total == 0:
        return 0.0
    target = (q / 100.0) * total
    cum = 0
    for v in range(1, 256):
        cnt = int(hist[v])
        if cnt == 0:
            continue
        cum += cnt
        if cum >= target:
            return float(v)
    return 255.0


def _StatsFromHistogram(hist: np.ndarray) -> dict:
    """由 256-bin histogram（bin0=invalid 不参与）精确计算 valid Depth 像素级统计。"""
    total = int(hist[1:].sum())
    if total == 0:
        return {"min": 0, "max": 0, "mean": 0.0, "std": 0.0,
                "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    # min / max：最小 / 最大有效 bin
    nonzero = np.nonzero(hist[1:])[0] + 1  # bin 下标 1..255
    vmin = int(nonzero[0])
    vmax = int(nonzero[-1])
    # mean：sum(v * count) / total
    weighted = sum(v * int(hist[v]) for v in range(1, 256))
    mean = float(weighted / total)
    # std：总体标准差（ddof=0，与 np.std 默认一致）
    var = sum(int(hist[v]) * (v - mean) ** 2 for v in range(1, 256)) / total
    std = float(math.sqrt(var))
    return {
        "min": vmin,
        "max": vmax,
        "mean": mean,
        "std": std,
        "p10": _PercentileFromHistogram(hist, 10),
        "p25": _PercentileFromHistogram(hist, 25),
        "p50": _PercentileFromHistogram(hist, 50),
        "p75": _PercentileFromHistogram(hist, 75),
        "p90": _PercentileFromHistogram(hist, 90),
    }


def _StatsFromValues(values) -> dict:
    """由图片级 float 列表（per-image valid mean / per-image valid ratio）计算统计。"""
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
                "p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),  # np.std 默认 ddof=0
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
    }


# --------------------------------------------------------------------------- 单 split 分析
def _AnalyzeSplit(split_img_dir: Path, split_name: str) -> dict:
    """读取某 split 全部 4 通道图，累积 256-bin histogram + 图片级 valid mean/ratio 列表。

    返回内部 dict（含 hist / per_image_valid_mean / per_image_valid_ratio，供 overall 合并），
    同时完成整 split 无 valid Depth 的报错。
    """
    if not split_img_dir.exists():
        raise FileNotFoundError(f"split 图片目录不存在: {split_img_dir} (split={split_name})")

    img_paths = sorted(p for p in split_img_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
    if not img_paths:
        raise FileNotFoundError(f"split 未找到图片（{_IMAGE_SUFFIXES}）: {split_img_dir} (split={split_name})")

    hist = np.zeros(256, dtype=np.int64)          # 256-bin histogram（bin0 = invalid）
    per_image_valid_mean: list[float] = []        # 仅含有效图片（无 valid Depth 的图不加入）
    per_image_valid_ratio: list[float] = []       # 含全部图片（无 valid Depth 的图 ratio=0）
    images_without_valid = 0
    total_pixel_count = 0

    for p in img_paths:
        im = _ReadRgba(p)                          # (H,W,4) uint8，不满足直接报错退出
        depth = im[:, :, 3]                        # 第4通道 = 模型实际看到的 uint8 Depth
        h, _ = np.histogram(depth.ravel(), bins=256, range=(0, 256))
        hist += h.astype(np.int64)

        n_pixels = int(depth.size)
        valid_count = int(h[1:].sum())
        total_pixel_count += n_pixels

        if valid_count == 0:
            # 无有效 Depth：打印 warning（含路径），仍计入 image_count，ratio=0，不加入 per-image valid mean
            print(f"[警告] 无有效 Depth(depth>0) 的图片: {p}")
            images_without_valid += 1
            per_image_valid_ratio.append(0.0)
            continue

        vmean = float(depth[depth > 0].mean())
        vratio = valid_count / n_pixels
        per_image_valid_mean.append(vmean)
        per_image_valid_ratio.append(vratio)

    valid_pixel_count = int(hist[1:].sum())
    invalid_pixel_count = int(hist[0])  # 全 split invalid 像素总量
    if valid_pixel_count == 0:
        raise RuntimeError(
            f"split '{split_name}' 全部图片均无有效 Depth(depth>0)，无法输出有效统计: {split_img_dir}")

    image_count = len(img_paths)
    return {
        "image_count": image_count,
        "images_without_valid_depth": images_without_valid,
        "total_pixel_count": total_pixel_count,
        "valid_pixel_count": valid_pixel_count,
        "invalid_pixel_count": invalid_pixel_count,
        "hist": hist,
        "per_image_valid_mean": per_image_valid_mean,
        "per_image_valid_ratio": per_image_valid_ratio,
    }


def _BuildSplitResult(internal: dict, split_name: str) -> dict:
    """把 _AnalyzeSplit 内部结果整理成对外 JSON 结构（去掉 hist，展开统计字段）。"""
    total_pixel = internal["total_pixel_count"]
    valid_pixel = internal["valid_pixel_count"]
    invalid_pixel = internal["invalid_pixel_count"]
    valid_ratio = (valid_pixel / total_pixel) if total_pixel else 0.0
    invalid_ratio = (invalid_pixel / total_pixel) if total_pixel else 0.0
    return {
        "image_count": internal["image_count"],
        "images_without_valid_depth": internal["images_without_valid_depth"],
        "total_pixel_count": total_pixel,
        "valid_pixel_count": valid_pixel,
        "invalid_pixel_count": invalid_pixel,
        "valid_ratio": valid_ratio,
        "invalid_ratio": invalid_ratio,
        "valid_pixel_depth": _StatsFromHistogram(internal["hist"]),
        "per_image_valid_mean": _StatsFromValues(internal["per_image_valid_mean"]),
        "per_image_valid_ratio": _StatsFromValues(internal["per_image_valid_ratio"]),
    }


# --------------------------------------------------------------------------- 输出
def _PrintSummary(split_name: str, result: dict):
    """打印单个 split 的紧凑统计。"""
    vp = result["valid_pixel_depth"]
    pm = result["per_image_valid_mean"]
    pr = result["per_image_valid_ratio"]
    print("=" * 64)
    print(f"Depth Distribution: {split_name}")
    print(f"images              : {result['image_count']}")
    print(f"images w/o valid    : {result['images_without_valid_depth']}")
    print(f"valid pixels        : {result['valid_pixel_count']:,}")
    print(f"invalid ratio       : {result['invalid_ratio'] * 100:.4f}%")
    print()
    print("Valid Depth Pixels:")
    print(f"  min/max           : {vp['min']} / {vp['max']}")
    print(f"  mean/std          : {vp['mean']:.4f} / {vp['std']:.4f}")
    print(f"  p10/p25/p50/p75/p90 : {vp['p10']:.2f} / {vp['p25']:.2f} / {vp['p50']:.2f} "
          f"/ {vp['p75']:.2f} / {vp['p90']:.2f}")
    print()
    print("Per-image Valid Mean:")
    print(f"  min/max           : {pm['min']:.4f} / {pm['max']:.4f}")
    print(f"  mean/std          : {pm['mean']:.4f} / {pm['std']:.4f}")
    print(f"  p10/p25/p50/p75/p90 : {pm['p10']:.2f} / {pm['p25']:.2f} / {pm['p50']:.2f} "
          f"/ {pm['p75']:.2f} / {pm['p90']:.2f}")
    print()
    print("Per-image Valid Ratio:")
    print(f"  min/max           : {pr['min']:.6f} / {pr['max']:.6f}")
    print(f"  mean/std          : {pr['mean']:.6f} / {pr['std']:.6f}")
    print(f"  p10/p25/p50/p75/p90 : {pr['p10']:.4f} / {pr['p25']:.4f} / {pr['p50']:.4f} "
          f"/ {pr['p75']:.4f} / {pr['p90']:.4f}")


def _PrintCompactTable(rows: dict):
    """打印最重要的简洁对照表（Split / Images / PixelMean / P25 / P50 / P75 / P90 / ImageMeanAvg / InvalidRatio）。"""
    print()
    print("=" * 78)
    hdr = (f"{'Split':10s} {'Images':>7s} {'PixelMean':>10s} {'P25':>6s} {'P50':>6s} "
           f"{'P75':>6s} {'P90':>6s} {'ImageMeanAvg':>12s} {'InvalidRatio':>12s}")
    print(hdr)
    print("-" * 78)
    for name in ("train", "val", "test", "overall"):
        if name not in rows:
            continue
        r = rows[name]
        vp = r["valid_pixel_depth"]
        pm = r["per_image_valid_mean"]
        print(f"{name:10s} {r['image_count']:>7d} {vp['mean']:>10.2f} {vp['p25']:>6.1f} {vp['p50']:>6.1f} "
              f"{vp['p75']:>6.1f} {vp['p90']:>6.1f} {pm['mean']:>12.2f} {r['invalid_ratio'] * 100:>11.4f}%")
    print("=" * 78)


def _PrintKeyQuestion(splits_result: dict):
    """额外打印最关键对照（不自动下科学结论）：train/val/test 的 valid pixel mean，
    以及每张图 valid mean 的 p25/p50/p75，便于直接判断 test≈170 是否具有代表性。"""
    print()
    print("[关键对照] 各 split valid pixel mean / per-image valid mean 分位数：")
    for name in ("train", "val", "test"):
        if name not in splits_result:
            continue
        r = splits_result[name]
        vp = r["valid_pixel_depth"]
        pm = r["per_image_valid_mean"]
        print(f"  {name:6s} valid pixel mean = {vp['mean']:.4f} | "
              f"per-image valid mean p25/p50/p75 = {pm['p25']:.2f}/{pm['p50']:.2f}/{pm['p75']:.2f}")
    print("  （若 test 与 train/val 的 pixel mean 及 per-image mean 分位数接近，")
    print("    则说明 test≈170 是整份 Hand 数据集的常态，而非 12 张连续帧特例。）")


# --------------------------------------------------------------------------- main
def Main():
    p = argparse.ArgumentParser(description="RGBD 第4通道 uint8 Depth 数值分布统计（train/val/test + overall）")
    p.add_argument("--rgbd-yaml", default=_DEFAULT_RGBD_YAML,
                   help="RGBD data yaml（提供 path / nc / kpt_shape / channels / rgbd 与 train/val/test 子目录）")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                   help="要统计的 split（默认 train val test）")
    p.add_argument("--out", default=_DEFAULT_OUT, help="结果输出目录（depth_distribution.json 写入此处）")
    args = p.parse_args()

    rgbd_yaml = Path(args.rgbd_yaml)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    cfg = _LoadYaml(rgbd_yaml)
    print("=" * 64)
    print("RGBD Depth Distribution Analyzer")
    print(f"yaml   : {rgbd_yaml}")
    print(f"out    : {out_root}")
    print(f"splits : {args.splits}")
    print("=" * 64)

    splits_internal: dict[str, dict] = {}
    splits_result: dict[str, dict] = {}
    for split in args.splits:
        split_dir = _ResolveSplitDir(cfg, split)
        print(f"\n[扫描] split={split} 目录={split_dir}")
        internal = _AnalyzeSplit(split_dir, split)
        # 打印图片目录与数量
        img_paths = sorted(p for p in split_dir.iterdir()
                           if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
        print(f"        图片数量 = {len(img_paths)}")
        splits_internal[split] = internal
        splits_result[split] = _BuildSplitResult(internal, split)
        _PrintSummary(split, splits_result[split])

    # overall：三组合并（hist 累加 + per-image 列表拼接）
    if len(args.splits) > 1:
        ov_hist = np.zeros(256, dtype=np.int64)
        ov_mean: list[float] = []
        ov_ratio: list[float] = []
        ov_image_count = 0
        ov_without_valid = 0
        ov_total_pixel = 0
        ov_valid_pixel = 0
        ov_invalid_pixel = 0
        for split in args.splits:
            it = splits_internal[split]
            ov_hist += it["hist"]
            ov_mean.extend(it["per_image_valid_mean"])
            ov_ratio.extend(it["per_image_valid_ratio"])
            ov_image_count += it["image_count"]
            ov_without_valid += it["images_without_valid_depth"]
            ov_total_pixel += it["total_pixel_count"]
            ov_valid_pixel += it["valid_pixel_count"]
            ov_invalid_pixel += it["invalid_pixel_count"]
        ov_internal = {
            "image_count": ov_image_count,
            "images_without_valid_depth": ov_without_valid,
            "total_pixel_count": ov_total_pixel,
            "valid_pixel_count": ov_valid_pixel,
            "invalid_pixel_count": ov_invalid_pixel,
            "hist": ov_hist,
            "per_image_valid_mean": ov_mean,
            "per_image_valid_ratio": ov_ratio,
        }
        splits_result["overall"] = _BuildSplitResult(ov_internal, "overall")
        _PrintSummary("overall", splits_result["overall"])

    _PrintCompactTable(splits_result)
    _PrintKeyQuestion(splits_result)

    payload = {
        "rgbd_yaml": str(rgbd_yaml),
        "splits": {name: splits_result[name] for name in args.splits if name in splits_result},
    }
    if "overall" in splits_result:
        payload["overall"] = splits_result["overall"]
    out_json = out_root / "depth_distribution.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[结果] 已写出 {out_json}")


if __name__ == "__main__":
    Main()
