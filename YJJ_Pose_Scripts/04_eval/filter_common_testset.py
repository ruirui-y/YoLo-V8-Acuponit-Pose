"""基于权重的公共测试集筛选（GUI 调用 / 命令行均可）。

功能
----
给定一组权重，各自对源测试集做一次推理并按 IoU 判定命中，
取「全部权重都命中」的图片，导出为一个新的【公共测试集】目录。

为什么需要
----------
不同模型的命中集不同（例如 RGB 仅 66/101、RGBD 101/101）。
若各自在不同子集上算 NME，分母不同 → 数字不可比。
先筛出「人人都有分」的公共子集并固化到磁盘，之后所有 NME 都在
该公共集上算 → 分母天然一致。

跨模态
------
源数据集可能同时含 rgb/ 与 rgbd/（同一批 stem，扩展名 .jpg / .png 不同）。
- 3 通道权重读 rgb/images/<split>
- 4 通道权重读 rgbd/images/<split>
命中判定按 stem 对齐；交集按 stem 计算；导出时把入选 stem 的
rgb + rgbd + label 一并复制。

口径一致性
----------
命中判定完全复用 eval_rgb_vs_rgbd_nme.py 的
load_pose_label / run_pose_model / find_best_prediction，
不另写一套算法（IoU 阈值默认 0.50，conf 0.001，imgsz 640）。

用法
----
    python filter_common_testset.py ^
        --dataset-root H:\\YJJ\\Yolo_RGBD\\Resource\\session1_200146\\dataset\\hand_cross_subject ^
        --weights a.pt b.pt c.pt ^
        --output-dir H:\\YJJ\\Yolo_RGBD\\Resource\\session1_200146\\dataset\\hand_common

输出
----
    <output-dir>/
        rgb/images/test/   rgb/labels/test/
        rgbd/images/test/  rgbd/labels/test/
        filter_report.csv      每张图：各权重命中情况 / 是否入选
        filter_summary.json    各权重命中数、公共集大小、剔除清单
        data_rgb.yaml          指向新目录（若源含 rgb/）
        data_rgbd.yaml         指向新目录（若源含 rgbd/）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

# ============================================================
# 路径推导：本脚本位于 YJJ_Pose_Scripts/04_eval/ 下
# ============================================================
_HERE: Path = Path(__file__).resolve()
_REPO_ROOT: Path = _HERE.parents[2]
_EVAL_SCRIPT: Path = _HERE.parent / "eval_rgb_vs_rgbd_nme.py"

# ============================================================
# 关键：在任何 ultralytics import 之前，先把项目根从 sys.path 剥离。
#
# GUI 的 ProcessRunner 会把项目根前置到 PYTHONPATH
# （core/process_runner.py:56-60，"确保本地 ultralytics 可被找到"），
# 于是 sys.path 里出现项目根，import ultralytics 会命中项目内 vendored
# 的 8.3.176 —— 即使子进程用的是 .venv 的 python 也一样，
# 因为 PYTHONPATH 与解释器无关。
#
# 本脚本需要 venv 的 8.4.x（支持 YOLO26），故先剥离项目根。
# 同时移除空串条目（'' 表示当前工作目录，而 cwd 恰是项目根）。
# ============================================================
_REPO_NORM: str = os.path.normcase(os.path.abspath(str(_REPO_ROOT)))
sys.path[:] = [
    entry for entry in sys.path
    if entry and os.path.normcase(os.path.abspath(entry)) != _REPO_NORM
]

# ============================================================
# 再 import venv 的 ultralytics 并锁定版本，然后加载第一阶段脚本。
# 第一阶段脚本开头也会 sys.path.insert(0, 项目根)，但那发生在 ultralytics
# 已进入 sys.modules 之后，不会再被替换。
# ============================================================
import ultralytics  # noqa: E402

_LOCKED_VERSION: str = ultralytics.__version__

if not _EVAL_SCRIPT.is_file():
    raise FileNotFoundError(f"第一阶段评估脚本不存在: {_EVAL_SCRIPT}")

sys.path.insert(0, str(_EVAL_SCRIPT.parent))
import eval_rgb_vs_rgbd_nme as ev  # noqa: E402

if ultralytics.__version__ != _LOCKED_VERSION:
    raise RuntimeError(
        f"ultralytics 被换掉了: {_LOCKED_VERSION} -> {ultralytics.__version__}"
    )

# 清掉第一阶段脚本插进来的项目根，避免后续 import 命中 vendored 版本
_LEAKED: str = str(_REPO_ROOT)
while _LEAKED in sys.path:
    sys.path.remove(_LEAKED)

from ultralytics import YOLO  # noqa: E402


IMAGE_EXTS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
)
DEFAULT_META: dict[str, Any] = {"nc": 1, "names": {0: "hand"}, "kpt_shape": [7, 3]}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="基于权重的公共测试集筛选")
    p.add_argument("--dataset-root", required=True,
                   help="源数据集根目录（含 rgb/ 和/或 rgbd/）")
    p.add_argument("--weights", nargs="+", required=True,
                   help="参与筛选的权重路径（可传多个，空格分隔）")
    p.add_argument("--output-dir", required=True,
                   help="公共测试集输出目录（不存在会自动创建）")
    p.add_argument("--split", default="test", choices=["test"],
                   help="筛选哪个 split，目前仅支持 test")
    p.add_argument("--match-iou", type=float, default=0.50,
                   help="命中判定 IoU 阈值，默认 0.50")
    p.add_argument("--conf", type=float, default=0.001,
                   help="推理置信度阈值，默认 0.001")
    p.add_argument("--imgsz", type=int, default=640,
                   help="推理尺寸，默认 640")
    p.add_argument("--min-common", type=int, default=0,
                   help="公共集低于该张数时告警（0 = 不告警）")
    p.add_argument("--allow-skip", action="store_true",
                   help="允许跳过加载失败的权重（默认严格：任一权重失败即中止，"
                        "因为「全部权重都命中」的定义会被破坏）")
    return p.parse_args()


# ============================================================
# 基础工具
# ============================================================
def stem_map(directory: Path) -> dict[str, Path]:
    """目录内图片 stem -> 路径（保持一一对应，不按文件名直接配）。"""
    if not directory.is_dir():
        return {}
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    }


def read_image(path: Path, channels: int) -> np.ndarray:
    """3 通道用 IMREAD_COLOR；4 通道必须 IMREAD_UNCHANGED，否则 Depth 被静默丢弃。"""
    flag = cv2.IMREAD_UNCHANGED if channels == 4 else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), flag)
    if image is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    if image.ndim != 3:
        raise ValueError(f"图像维度异常: {path} shape={image.shape}")
    return image


def load_model(weights: Path) -> tuple[YOLO, int, str]:
    """加载权重并返回 (model, 第一层输入通道数, head 类名)。"""
    model = YOLO(str(weights))
    channels = int(model.model.model[0].conv.weight.shape[1])
    head = type(model.model.model[-1]).__name__
    return model, channels, head


def make_unique_names(paths: list[Path]) -> list[str]:
    """为每个权重生成互不重复的显示名。

    训练产物惯例是 `.../<run-name>/weights/best.pt`，直接用 stem 会让所有权重
    都叫 "best"，导致报表列名与图表互相覆盖。这里优先取 weights 的父目录名
    （如 yolo11n-pose / train_pose_rgbd_4ch5），重名时追加序号。
    """
    names: list[str] = []
    used: dict[str, int] = {}
    for path in paths:
        if path.parent.name.lower() == "weights" and path.parent.parent.name:
            base = path.parent.parent.name
        elif path.parent.name:
            base = f"{path.parent.name}_{path.stem}"
        else:
            base = path.stem
        count = used.get(base, 0) + 1
        used[base] = count
        names.append(base if count == 1 else f"{base}_{count}")
    return names


def load_src_meta(src_yaml: Path) -> dict[str, Any]:
    """从源 yaml 读取 nc / names / kpt_shape，用于生成新 yaml。"""
    if src_yaml.is_file():
        try:
            data: Any = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return {
                    "nc": data.get("nc", DEFAULT_META["nc"]),
                    "names": data.get("names", DEFAULT_META["names"]),
                    "kpt_shape": data.get("kpt_shape", DEFAULT_META["kpt_shape"]),
                }
        except (OSError, yaml.YAMLError):
            pass
    return dict(DEFAULT_META)


def write_target_yaml(dest_yaml: Path, dataset_root: Path, split: str,
                      channels: int, meta: dict[str, Any]) -> None:
    """生成指向新目录的 data_*.yaml（train/val/test 均指向筛选后的 test）。"""
    content = {
        "path": str(dataset_root),
        "train": f"images/{split}",
        "val": f"images/{split}",
        "test": f"images/{split}",
        "nc": meta["nc"],
        "names": meta["names"],
        "kpt_shape": meta["kpt_shape"],
        "channels": channels,
    }
    dest_yaml.write_text(
        yaml.safe_dump(content, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    args = parse_args()
    root = Path(args.dataset_root).expanduser()
    out = Path(args.output_dir).expanduser()
    weight_paths = [Path(w).expanduser() for w in args.weights]
    split = args.split

    print("=" * 92)
    print("基于权重的公共测试集筛选")
    print(f"源数据集 : {root}")
    print(f"输出目录 : {out}")
    print(f"split    : {split}")
    print(f"判定口径 : IoU >= {args.match_iou}   conf={args.conf}   imgsz={args.imgsz}")
    print(f"评估实现 : {_EVAL_SCRIPT.name}（ultralytics {_LOCKED_VERSION}）")
    print("=" * 92)

    # ---- 1. 收集源图 ----
    rgb_img_dir = root / "rgb" / "images" / split
    rgbd_img_dir = root / "rgbd" / "images" / split
    rgb_lbl_dir = root / "rgb" / "labels" / split
    rgbd_lbl_dir = root / "rgbd" / "labels" / split

    rgb_map = stem_map(rgb_img_dir)
    rgbd_map = stem_map(rgbd_img_dir)
    if not rgb_map and not rgbd_map:
        print(f"[STOP] 源数据集没有可用图片:\n  {rgb_img_dir}\n  {rgbd_img_dir}")
        return

    # 公共 stem：两模态都有则取交集；只有单模态时以其自身为准
    mismatched: set[str] = set()
    if rgb_map and rgbd_map:
        stems = sorted(set(rgb_map) & set(rgbd_map))
        mismatched = set(rgb_map) ^ set(rgbd_map)
    else:
        stems = sorted(rgb_map or rgbd_map)
    if not stems:
        print("[STOP] rgb/ 与 rgbd/ 没有共同的 stem，无法对齐")
        return

    label_dir = rgb_lbl_dir if rgb_lbl_dir.is_dir() else rgbd_lbl_dir
    if not label_dir.is_dir():
        print(f"[STOP] 找不到标签目录:\n  {rgb_lbl_dir}\n  {rgbd_lbl_dir}")
        return

    print(f"源图片   : rgb={len(rgb_map)} 张，rgbd={len(rgbd_map)} 张，"
          f"可用 stem={len(stems)} 张")
    if mismatched:
        print(f"[提示] 两模态 stem 不一致 {len(mismatched)} 个，已取交集")

    # ---- 2. 加载权重 ----
    names_all = make_unique_names(weight_paths)
    usable: list[dict[str, Any]] = []
    failures: list[str] = []
    for weights, name in zip(weight_paths, names_all):
        if not weights.is_file():
            print(f"[错误] 权重不存在: {weights}")
            failures.append(f"{name}（文件不存在）")
            continue
        try:
            model, channels, head = load_model(weights)
        except Exception as exc:  # noqa: BLE001
            hint = ""
            if "Pose26" in f"{type(exc).__name__}{exc}":
                hint = ("\n        该权重是 YOLO26，需要 ultralytics >= 8.4；"
                        "当前解释器加载的是项目内 vendored 8.3.176。\n"
                        "        请在 GUI 中把 Python 解释器指向 "
                        "MultiYoloRGB\\.venv\\Scripts\\python.exe")
            print(f"[错误] {name} 加载失败: {type(exc).__name__}: {exc}{hint}")
            failures.append(f"{name}（{type(exc).__name__}）")
            continue
        if channels == 3 and not rgb_map:
            print(f"[错误] {name} 是 3 通道权重，但源缺少 rgb/{split}")
            failures.append(f"{name}（缺少 rgb/{split}）")
            continue
        if channels == 4 and not rgbd_map:
            print(f"[错误] {name} 是 4 通道权重，但源缺少 rgbd/{split}")
            failures.append(f"{name}（缺少 rgbd/{split}）")
            continue
        if channels not in (3, 4):
            print(f"[错误] {name} 第一层通道={channels}（非 3/4）")
            failures.append(f"{name}（通道数 {channels}）")
            continue
        usable.append({"name": name, "path": weights,
                       "model": model, "channels": channels, "head": head})
        print(f"已加载: {name:26s} {channels}ch   head={head}")

    if failures and not args.allow_skip:
        print("\n" + "!" * 92)
        print("[STOP] 有权重未能参与筛选，「全部权重都命中」的定义已不成立，本次结果无效。")
        for item in failures:
            print(f"         - {item}")
        print("        修复后重跑；确需忽略请显式加 --allow-skip")
        print("!" * 92)
        raise SystemExit(2)

    if not usable:
        print("[STOP] 没有可用的权重")
        return

    names = [m["name"] for m in usable]

    # ---- 3. 逐权重推理，记录命中 ----
    hit: dict[str, dict[str, bool]] = {stem: {} for stem in stems}
    iou: dict[str, dict[str, float]] = {stem: {} for stem in stems}

    for m in usable:
        src_dir = rgb_map if m["channels"] == 3 else rgbd_map
        print(f"\n>>> 推理 {m['name']}（{m['channels']}ch）")
        started = time.perf_counter()
        for index, stem in enumerate(stems, start=1):
            image_path = src_dir.get(stem)
            label_path = label_dir / f"{stem}.txt"
            if image_path is None or not label_path.is_file():
                hit[stem][m["name"]] = False
                iou[stem][m["name"]] = 0.0
                continue
            try:
                image = read_image(image_path, m["channels"])
                height, width = image.shape[:2]
                gt = ev.load_pose_label(label_path, width, height)
                result = ev.run_pose_model(m["model"], image, args.conf, args.imgsz)
                pred = ev.find_best_prediction(result, gt["box"], args.match_iou)
                hit[stem][m["name"]] = bool(pred["matched"])
                iou[stem][m["name"]] = float(pred["iou"])
            except Exception as exc:  # noqa: BLE001
                hit[stem][m["name"]] = False
                iou[stem][m["name"]] = 0.0
                print(f"  [跳过] {stem}: {type(exc).__name__}: {exc}")
            if index % 20 == 0 or index == len(stems):
                print(f"  进度 {index}/{len(stems)}", flush=True)
        matched_n = sum(1 for stem in stems if hit[stem][m["name"]])
        print(f"  命中 {matched_n}/{len(stems)}   用时 {time.perf_counter() - started:.1f}s")

    # ---- 4. 求交集 ----
    common = [stem for stem in stems
              if all(hit[stem][name] for name in names)]
    dropped = [stem for stem in stems if stem not in set(common)]

    # ---- 5. 导出 ----
    out_rgb_img = out / "rgb" / "images" / split
    out_rgb_lbl = out / "rgb" / "labels" / split
    out_rgbd_img = out / "rgbd" / "images" / split
    out_rgbd_lbl = out / "rgbd" / "labels" / split

    # 先清空这 4 个目标子目录，否则上一轮产物会残留，
    # 出现「本次导出 100 张、目录里却有 101 个」的假象。
    # 只动这 4 个子目录，不碰输出目录下的其他内容。
    purged = 0
    for directory in (out_rgb_img, out_rgb_lbl, out_rgbd_img, out_rgbd_lbl):
        if not directory.is_dir():
            continue
        for child in directory.iterdir():
            if child.is_file() or child.is_symlink():
                child.unlink()
                purged += 1
            elif child.is_dir():
                shutil.rmtree(child)
                purged += 1
    if purged:
        print(f"[提示] 已清理输出目录中的旧产物 {purged} 项")

    for directory in (out_rgb_img, out_rgb_lbl, out_rgbd_img, out_rgbd_lbl):
        directory.mkdir(parents=True, exist_ok=True)

    copied_rgb = copied_rgbd = copied_label = 0
    for stem in common:
        if stem in rgb_map:
            shutil.copy2(rgb_map[stem], out_rgb_img / rgb_map[stem].name)
            copied_rgb += 1
        if stem in rgbd_map:
            shutil.copy2(rgbd_map[stem], out_rgbd_img / rgbd_map[stem].name)
            copied_rgbd += 1
        label_path = label_dir / f"{stem}.txt"
        if label_path.is_file():
            shutil.copy2(label_path, out_rgb_lbl / label_path.name)
            shutil.copy2(label_path, out_rgbd_lbl / label_path.name)
            copied_label += 1

    # ---- 6. 报表 ----
    report_csv = out / "filter_report.csv"
    with report_csv.open("w", encoding="utf-8-sig", newline="") as f:
        fieldnames = (["image"]
                      + [f"{name}_matched" for name in names]
                      + [f"{name}_iou" for name in names]
                      + ["selected"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for stem in stems:
            row: dict[str, Any] = {"image": stem}
            for name in names:
                row[f"{name}_matched"] = str(hit[stem][name])
                row[f"{name}_iou"] = round(iou[stem][name], 6)
            row["selected"] = str(stem in set(common))
            writer.writerow(row)

    summary: dict[str, Any] = {
        "dataset_root": str(root),
        "output_dir": str(out),
        "split": split,
        "match_iou": args.match_iou,
        "conf": args.conf,
        "imgsz": args.imgsz,
        "eval_impl": _EVAL_SCRIPT.name,
        "ultralytics": _LOCKED_VERSION,
        "total_stems": len(stems),
        "weights": [
            {"name": m["name"], "path": str(m["path"]),
             "channels": m["channels"], "head": m["head"],
             "matched": sum(1 for stem in stems if hit[stem][m["name"]])}
            for m in usable
        ],
        "common_count": len(common),
        "dropped_count": len(dropped),
        "dropped": [
            {"image": stem,
             "missing": [name for name in names if not hit[stem][name]]}
            for stem in dropped
        ],
        "copied": {"rgb": copied_rgb, "rgbd": copied_rgbd, "label": copied_label},
    }
    (out / "filter_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if rgb_map:
        meta = load_src_meta(root / "rgb" / "data_rgb.yaml")
        write_target_yaml(out / "rgb" / "data_rgb.yaml", out / "rgb", split, 3, meta)
    if rgbd_map:
        meta = load_src_meta(root / "rgbd" / "data_rgbd.yaml")
        write_target_yaml(out / "rgbd" / "data_rgbd.yaml", out / "rgbd", split, 4, meta)

    # ---- 7. 汇总输出 ----
    print("\n" + "=" * 92)
    print("汇总")
    print("=" * 92)
    for m in usable:
        matched_n = sum(1 for stem in stems if hit[stem][m["name"]])
        print(f"  {m['name']:26s} {m['channels']}ch  命中 {matched_n:>4d}/{len(stems)}")
    print(f"\n  公共测试集（全部权重都命中）: {len(common)} / {len(stems)}")
    print(f"  已导出: rgb={copied_rgb} 张，rgbd={copied_rgbd} 张，label={copied_label} 个")
    print(f"  输出目录: {out}")

    if args.min_common and len(common) < args.min_common:
        print(f"\n  [警告] 公共集 {len(common)} 张，低于设定下限 {args.min_common} 张，请检查权重")
    if not common:
        print("\n  [警告] 公共集为空：没有任何一张图被全部权重命中")

    print(f"\n结果文件:\n  {report_csv}\n  {out / 'filter_summary.json'}")


if __name__ == "__main__":
    main()
