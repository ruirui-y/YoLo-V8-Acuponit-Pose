#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Use ONE canonical/template YOLO-Pose label to reorder another label set.

说明（本文件已改为薄 CLI wrapper）：
- 核心几何匹配逻辑已抽取到 ``01_data/pose_label_reorder.py``（可 import、无 Qt、
  无 subprocess），本文件只负责 argparse / 目录合法性检查 / 输出 report 与摘要，
  保持原有独立运行能力与 CLI 兼容。

用法（与原版一致）：
    python reorder_pose_labels_by_template.py \\
        --template <canonical label.txt> \\
        --src <目标 label 目录> \\
        --dst <输出目录(必须 != src)> \\
        [--max-residual 0.17] [--dry-run]

行为：忽略目标 label 自带 K 编号，按整体几何（平移/旋转/scale）匹配 template，
仅重排 (x,y,v) 三元组顺序；class/bbox 原样保留；源文件从不覆盖。
新增歧义护栏：best 与次优过近（近似对称/重合点）-> AMBIGUOUS BLOCK，不猜 ID。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让 01_data/pose_label_reorder.py 可被 import（Python 3 隐式命名空间包）
_01DATA = Path(__file__).resolve().parents[1] / "01_data"
if str(_01DATA) not in sys.path:
    sys.path.insert(0, str(_01DATA))

import pose_label_reorder as plr  # noqa: E402


def parse_args(argv: list[str] | None = None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="Canonical/template label .txt")
    ap.add_argument("--src", required=True, help="Target label directory")
    ap.add_argument("--dst", required=True, help="Output directory (must differ from src)")
    ap.add_argument(
        "--max-residual",
        type=float,
        default=plr.DEFAULT_MAX_RESIDUAL,
        help="BLOCK if normalized similarity RMS is above this value (default 0.17)",
    )
    ap.add_argument("--dry-run", action="store_true", help="Only validate/report; write nothing")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    template = Path(args.template).expanduser().resolve()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()

    if not template.is_file():
        print(f"[ERROR] template not found: {template}")
        return 1
    if not src.is_dir():
        print(f"[ERROR] src directory not found: {src}")
        return 1
    if src == dst:
        print("[ERROR] src and dst must be different")
        return 1
    try:
        dst.relative_to(src)
    except ValueError:
        pass
    else:
        print("[ERROR] dst must not be inside src")
        return 1
    if dst.exists() and any(dst.iterdir()):
        print(f"[ERROR] dst already exists and is not empty: {dst}")
        return 1

    try:
        template_k = plr.template_keypoint_count(template)
        files = sorted(src.rglob("*.txt"))
    except plr.ReorderError as e:
        print(f"[ERROR] {e}")
        return 2
    if not files:
        print(f"[ERROR] no .txt labels under: {src}")
        return 1

    try:
        rows, ok, fail = plr.reorder_directory(
            src, dst, template, max_residual=args.max_residual, dry_run=args.dry_run)
    except plr.ReorderError as e:
        print(f"[ERROR] {e}")
        return 2

    report_dir = src if args.dry_run else dst
    report = report_dir / "reorder_by_template_report.csv"
    plr.write_report(rows, report)

    print("=== Reorder Pose Labels By Template ===")
    print(f"template   : {template}")
    print(f"src        : {src}")
    print(f"dst        : {dst}")
    print(f"keypoints  : {template_k}")
    print(f"files      : {len(files)}")
    print(f"ok         : {ok}")
    print(f"blocked    : {fail}")
    print(f"dry_run    : {args.dry_run}")
    print(f"report     : {report}")

    if fail:
        print("[BLOCK] some labels were not reordered; review report.")
        return 2

    print("[OK] all labels matched the template geometry.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
