"""从已生成的 Hand RGBD 4 通道数据集（rgbd_rawdepth）生成一个完全独立的新数据集（rgbd_rawdepth_scale25）。

目标：保持 RGB / train-val-test split / label / valid mask 完全不变，仅把"模型当前真正看到的第4通道
uint8 Depth"按 scale（本轮 0.25）整体缩放，用于构造低幅度 Depth 配对数据集。

严格定义（Scale25 / 一般 scale）：
  depth = image[:, :, 3]                       # 已是 1100~1850 → 1~255 映射后的 uint8 Depth，不再重新映射
  valid_mask = depth > 0
  depth == 0  -> 继续严格保持 0（invalid 不污染、不缩放）
  depth > 0   -> scaled = clip(rint(depth * scale), 1, 255)   # scale=0.25 时有效值最大应为 64
  output_image[:, :, 3] = scaled_depth        # 前三通道逐像素不变

不重新读取 16bit raw depth；不重新执行 1100~1850 映射；不改 RGB / label / split / 模型；不训练 / 不跑 YOLO。
输出 PNG（无损），label 用 shutil.copy2 原样复制。每张图生成时强制校验，失败直接抛异常（不静默继续）。

调用示例：
  python YJJ_Pose_Scripts/01_data/scale_rgbd_depth_dataset.py
  python YJJ_Pose_Scripts/01_data/scale_rgbd_depth_dataset.py \
      --rgbd-yaml H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand/rgbd_rawdepth/data_rgbd_rawdepth.yaml \
      --scale 0.25 \
      --out H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand/rgbd_rawdepth_scale25
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

# ---- 默认路径（与本轮实际运行一致）----
_DEFAULT_RGBD_YAML = (
    r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand/rgbd_rawdepth/data_rgbd_rawdepth.yaml"
)
_DEFAULT_SCALE = 0.25
_DEFAULT_OUT = r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand/rgbd_rawdepth_scale25"

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")  # 当前项目实际使用的扩展名（大小写不敏感）
_SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- 工具
def _LoadYaml(rgbd_yaml: Path) -> dict:
    """读取源 RGBD data yaml，返回解析后的 dict。"""
    if not rgbd_yaml.exists():
        raise FileNotFoundError(f"源 RGBD yaml 不存在: {rgbd_yaml}")
    return yaml.safe_load(rgbd_yaml.read_text(encoding="utf-8"))


def _ScaleDepth(depth: np.ndarray, scale: float) -> np.ndarray:
    """按 scale 缩放已映射的 uint8 Depth：invalid(depth==0) 保持 0，valid 缩放后 clip 到 1~255。

    与消融实验 scaled_true_depth_channel 同一语义：valid_mask = depth>0，先转 float 再 rint 防 uint8
    乘法截断，最后 clip 下限 1 防止有效像素归零（不污染 invalid）。输出 dtype=uint8，shape 一致。
    """
    scaled = np.zeros_like(depth, dtype=np.uint8)
    valid_mask = depth > 0
    if np.any(valid_mask):
        vals = np.rint(depth[valid_mask].astype(np.float32) * scale)
        vals = np.clip(vals, 1, 255)
        scaled[valid_mask] = vals.astype(np.uint8)
    return scaled


def _WriteOutputYaml(src_yaml: Path, out_yaml: Path, out_root: Path):
    """以源 yaml 为基础生成新 yaml：保留任务配置（names / kpt_shape / flip_idx / nc / channels / rgbd 等），
    仅改写 path / train / val / test 指向新数据集目录。
    """
    text = src_yaml.read_text(encoding="utf-8")
    new_path = str(out_root).replace("\\", "/")
    lines = []
    path_replaced = False
    for line in text.splitlines():
        m = re.match(r"^(\s*path\s*:\s*)(.+?)(\s*#.*)?$", line)
        if m:
            lines.append(f"{m.group(1)}{new_path}{m.group(3) or ''}")
            path_replaced = True
            continue
        m2 = re.match(r"^(\s*(?:train|val|test)\s*:\s*)(.+?)(\s*#.*)?$", line)
        if m2:
            key = re.match(r"^\s*(train|val|test)", m2.group(1)).group(1)
            lines.append(f"{m2.group(1)}images/{key}{m2.group(3) or ''}")
            continue
        lines.append(line)
    if not path_replaced:
        raise ValueError("源 yaml 未找到 path: 字段，无法改写")
    out_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- 单 split 处理
def _ProcessSplit(source_img_dir: Path, target_img_dir: Path,
                  source_label_dir: Path, target_label_dir: Path,
                  scale: float, split_name: str) -> dict:
    """处理单个 split：缩放第4通道 + 原样复制 label，逐图强制校验，累积统计。"""
    if not source_img_dir.exists():
        raise FileNotFoundError(f"源图片目录不存在: {source_img_dir} (split={split_name})")
    target_img_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(p for p in source_img_dir.iterdir()
                       if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES)
    if not img_paths:
        raise FileNotFoundError(f"split 未找到图片（{_IMAGE_SUFFIXES}）: {source_img_dir} (split={split_name})")

    max_allowed = int(np.rint(255 * scale))  # scale=0.25 -> 64；用于校验缩放后有效上限

    # 统计累积
    src_valid_sum = 0.0
    src_valid_count = 0
    scl_valid_sum = 0.0
    scl_valid_count = 0
    src_valid_min = 255
    src_valid_max = 0
    scl_valid_min = 255
    scl_valid_max = 0
    total_pixels = 0
    invalid_pixels = 0
    rgb_ok_all = True
    mask_ok_all = True

    for p in img_paths:
        src = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if src is None:
            raise FileNotFoundError(f"图像读取失败: {p}")
        if src.ndim != 3 or src.shape[2] != 4:
            raise ValueError(f"图像不是 4 通道（shape={src.shape if src.ndim == 3 else src.shape}）: {p}")
        if src.dtype != np.uint8:
            raise ValueError(f"图像 dtype 非 uint8（dtype={src.dtype}）: {p}")

        depth = src[:, :, 3]
        scaled_depth = _ScaleDepth(depth, scale)

        out_img = src.copy()
        out_img[:, :, 3] = scaled_depth

        # ---- 强制校验（失败直接抛异常，不 warning 后继续）----
        # 1) shape 完全一致
        if out_img.shape != src.shape:
            raise AssertionError(f"shape 不一致: {p} src={src.shape} out={out_img.shape}")
        # 2) dtype uint8
        if out_img.dtype != np.uint8:
            raise AssertionError(f"dtype 非 uint8: {p} dtype={out_img.dtype}")
        # 3) 前三通道逐像素完全不变
        if not np.array_equal(src[:, :, :3], out_img[:, :, :3]):
            raise AssertionError(f"前三通道被修改（RGB 未逐像素保持）: {p}")
        # 4) invalid mask 完全一致
        if not np.array_equal(depth == 0, scaled_depth == 0):
            raise AssertionError(f"invalid mask 改变（0=invalid 被污染）: {p}")
        valid_mask = depth > 0
        # 5) 有效像素缩放后仍 > 0
        if np.any(valid_mask) and scaled_depth[valid_mask].min() < 1:
            raise AssertionError(f"有效像素缩放后存在 <1（归零）: {p} "
                                  f"min={int(scaled_depth[valid_mask].min())}")
        # 6) scale=0.25 时 scaled.max() <= 64（一般：<= rint(255*scale)）
        if scaled_depth.max() > max_allowed:
            raise AssertionError(f"缩放后有效值上限超出预期: {p} max={int(scaled_depth.max())} "
                                  f"允许上限={max_allowed} (scale={scale})")
        # 7) 对正常含非零 Depth 的图，输出必须确实与源 Depth 不完全一致（scale<1 才有意义）
        if scale < 1.0 and np.any(valid_mask):
            if np.array_equal(scaled_depth[valid_mask], depth[valid_mask]):
                raise AssertionError(f"缩放后有效区域与源 Depth 完全相同（未缩放）: {p}")

        # 写图（PNG 无损，保证 RGB 精确不变）
        dst = target_img_dir / p.name
        ok = cv2.imwrite(str(dst), out_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if not ok:
            raise RuntimeError(f"写图失败: {dst}")

        # 累积统计（仅 valid 区域）
        n = int(depth.size)
        vcount = int(np.count_nonzero(valid_mask))
        total_pixels += n
        invalid_pixels += (n - vcount)
        if vcount > 0:
            s_vals = depth[valid_mask].astype(np.float64)
            t_vals = scaled_depth[valid_mask].astype(np.float64)
            src_valid_sum += float(s_vals.sum())
            src_valid_count += vcount
            scl_valid_sum += float(t_vals.sum())
            scl_valid_count += vcount
            src_valid_min = min(src_valid_min, int(s_vals.min()))
            src_valid_max = max(src_valid_max, int(s_vals.max()))
            scl_valid_min = min(scl_valid_min, int(t_vals.min()))
            scl_valid_max = max(scl_valid_max, int(t_vals.max()))
        rgb_ok_all = rgb_ok_all and np.array_equal(src[:, :, :3], out_img[:, :, :3])
        mask_ok_all = mask_ok_all and np.array_equal(depth == 0, scaled_depth == 0)

    # 原样复制 label（文件名 / 内容 / split 不变）
    label_paths = sorted(source_label_dir.glob("*")) if source_label_dir.exists() else []
    for lp in label_paths:
        if lp.is_file():
            shutil.copy2(lp, target_label_dir / lp.name)
    # 至少检查 source label 数量 == target label 数量
    target_labels = sorted(target_label_dir.glob("*"))
    if len(label_paths) != len(target_labels):
        raise RuntimeError(
            f"split '{split_name}' label 数量不一致: source={len(label_paths)} target={len(target_labels)}")

    src_valid_mean = (src_valid_sum / src_valid_count) if src_valid_count else 0.0
    scl_valid_mean = (scl_valid_sum / scl_valid_count) if scl_valid_count else 0.0
    invalid_ratio = (invalid_pixels / total_pixels) if total_pixels else 0.0

    return {
        "images": len(img_paths),
        "labels": len(label_paths),
        "source_valid_mean": float(src_valid_mean),
        "scaled_valid_mean": float(scl_valid_mean),
        "source_valid_min": int(src_valid_min) if src_valid_count else 0,
        "source_valid_max": int(src_valid_max) if src_valid_count else 0,
        "scaled_valid_min": int(scl_valid_min) if scl_valid_count else 0,
        "scaled_valid_max": int(scl_valid_max) if scl_valid_count else 0,
        "invalid_ratio": float(invalid_ratio),
        "valid_mask_consistent": bool(mask_ok_all),
        "rgb_consistent": bool(rgb_ok_all),
    }


# --------------------------------------------------------------------------- main
def Main():
    p = argparse.ArgumentParser(description="RGBD 第4通道 Depth 按 scale 缩放生成独立数据集（RGB/label/split/mask 不变）")
    p.add_argument("--rgbd-yaml", default=_DEFAULT_RGBD_YAML,
                   help="源 RGBD data yaml（已生成好的 uint8 Depth 4ch 数据集）")
    p.add_argument("--scale", type=float, default=_DEFAULT_SCALE,
                   help="Depth 缩放比例（0 < scale <= 1；本轮 0.25）")
    p.add_argument("--out", default=_DEFAULT_OUT, help="输出数据集目录（与源完全独立）")
    args = p.parse_args()

    scale = args.scale
    if not (0 < scale <= 1):
        raise ValueError(f"scale 必须满足 0 < scale <= 1，当前={scale}")

    src_yaml = Path(args.rgbd_yaml)
    out_root = Path(args.out)
    cfg = _LoadYaml(src_yaml)

    # 覆盖保护：--out 不能等于源数据集目录（不修改、不删除源）
    source_dataset_dir = Path(cfg["path"]).resolve()
    if out_root.resolve() == source_dataset_dir:
        raise RuntimeError(f"--out 不能等于源数据集目录（会破坏源数据）: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    out_yaml = out_root / "data_rgbd_scale25.yaml"

    print("=" * 64)
    print("Scale RGBD Depth Dataset")
    print(f"source yaml : {src_yaml}")
    print(f"scale       : {scale}")
    print(f"out root    : {out_root}")
    print(f"out yaml    : {out_yaml}")
    print("=" * 64)

    splits_result: dict[str, dict] = {}
    for split in _SPLITS:
        rel = cfg.get(split, f"images/{split}")
        src_img_dir = source_dataset_dir / rel
        src_label_dir = source_dataset_dir / "labels" / split
        tgt_img_dir = out_root / "images" / split
        tgt_label_dir = out_root / "labels" / split
        print(f"\n[处理] split={split}  源图目录={src_img_dir}  源label目录={src_label_dir}")
        stats = _ProcessSplit(src_img_dir, tgt_img_dir, src_label_dir, tgt_label_dir, scale, split)
        splits_result[split] = stats
        print(f"  图片={stats['images']}  label={stats['labels']}  "
              f"source_valid_mean={stats['source_valid_mean']:.4f}  "
              f"scaled_valid_mean={stats['scaled_valid_mean']:.4f}")
        print(f"  source min/max={stats['source_valid_min']}/{stats['source_valid_max']}  "
              f"scaled min/max={stats['scaled_valid_min']}/{stats['scaled_valid_max']}")
        print(f"  invalid_ratio={stats['invalid_ratio'] * 100:.4f}%  "
              f"valid_mask_consistent={stats['valid_mask_consistent']}  "
              f"rgb_consistent={stats['rgb_consistent']}")

    # overall：三组合并
    ov_images = sum(s["images"] for s in splits_result.values())
    ov_labels = sum(s["labels"] for s in splits_result.values())
    # 用加权（按 valid 像素数）合并 mean 需要各 split valid 像素数 —— 这里仅给简单合计均值便于查看
    ov_source_valid_mean = float(np.mean([s["source_valid_mean"] for s in splits_result.values()]))
    ov_scaled_valid_mean = float(np.mean([s["scaled_valid_mean"] for s in splits_result.values()]))
    ov_invalid_ratio = float(np.mean([s["invalid_ratio"] for s in splits_result.values()]))
    splits_result["overall"] = {
        "images": ov_images,
        "labels": ov_labels,
        "source_valid_mean": ov_source_valid_mean,
        "scaled_valid_mean": ov_scaled_valid_mean,
        "invalid_ratio": ov_invalid_ratio,
        "valid_mask_consistent": all(s["valid_mask_consistent"] for s in splits_result.values()),
        "rgb_consistent": all(s["rgb_consistent"] for s in splits_result.values()),
    }

    # 写输出 yaml
    _WriteOutputYaml(src_yaml, out_yaml, out_root)
    print(f"\n[写出] 输出 yaml: {out_yaml}")

    # 写结果 JSON
    payload = {
        "source_yaml": str(src_yaml),
        "output_yaml": str(out_yaml),
        "scale": scale,
        "splits": {sp: splits_result[sp] for sp in _SPLITS},
        "overall": splits_result["overall"],
    }
    out_json = out_root / "scale_depth_result.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[写出] 结果 JSON: {out_json}")


if __name__ == "__main__":
    Main()
