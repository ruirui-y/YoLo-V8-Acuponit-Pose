"""Depth NPY 批量清洗：质量评估 + 空洞修补 + 报告输出。

所有路径与深度阈值均来自命令行参数，代码中不写死任何数据路径。

示例：
    python filter_depth.py --input-dir "H:\\xxx\\out_npy"

    python filter_depth.py ^
        --input-dir "H:\\xxx\\out_npy" ^
        --output-dir "H:\\xxx\\out_npy_filtered" ^
        --depth-low 1100 ^
        --depth-high 1850
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.fill_depth import (
    DEFAULT_DEPTH_HIGH,
    DEFAULT_DEPTH_LOW,
    DEFAULT_MAX_HOLE_RATIO,
    DEFAULT_MIN_VALID_RANGE_RATIO,
    evaluate_depth,
    fill_zihang_filter,
)

# ------------------------------------------------------------
# 固定常量（均为文件名 / 后缀，不含任何数据路径）
# ------------------------------------------------------------

# 未传 --output-dir 时，输出目录 = <input-dir 同名> + 此后缀
DEFAULT_OUTPUT_SUFFIX = "_filtered"

QUALITY_CSV_NAME = "depth_quality.csv"
INVALID_TXT_NAME = "invalid_depths.txt"
VALID_TXT_NAME = "valid_depths.txt"

QUALITY_CSV_FIELDNAMES = [
    "filename",
    "is_valid",
    "hole_ratio",
    "valid_range_ratio",
    "valid_pixels",
    "median_depth_mm",
    "total_pixels",
    "invalid_pixels",
    "valid_range_pixels",
]


# ============================================================
# 参数
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量清洗 Depth NPY：质量评估 + 空洞修补 + 输出报告",
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="原始 Depth NPY 文件夹（必填，只扫描该目录下的 *.npy）",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "清洗后 NPY 输出文件夹（可选）；"
            f"不传时自动使用 <input-dir>{DEFAULT_OUTPUT_SUFFIX}"
        ),
    )

    parser.add_argument(
        "--depth-low",
        type=float,
        default=DEFAULT_DEPTH_LOW,
        help=f"有效深度下限（mm），默认 {DEFAULT_DEPTH_LOW:g}",
    )

    parser.add_argument(
        "--depth-high",
        type=float,
        default=DEFAULT_DEPTH_HIGH,
        help=f"有效深度上限（mm），默认 {DEFAULT_DEPTH_HIGH:g}",
    )

    parser.add_argument(
        "--max-hole-ratio",
        type=float,
        default=DEFAULT_MAX_HOLE_RATIO,
        help=f"允许的最大空洞率，默认 {DEFAULT_MAX_HOLE_RATIO:g}",
    )

    parser.add_argument(
        "--min-valid-range-ratio",
        type=float,
        default=DEFAULT_MIN_VALID_RANGE_RATIO,
        help=(
            "有效像素中落在 [depth-low, depth-high] 的最低占比，"
            f"默认 {DEFAULT_MIN_VALID_RANGE_RATIO:g}"
        ),
    )

    return parser.parse_args()


def resolve_output_dir(input_dir: Path, output_dir) -> Path:
    """确定输出目录；未显式指定时为 <input-dir><DEFAULT_OUTPUT_SUFFIX>。"""
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()

    return input_dir.with_name(input_dir.name + DEFAULT_OUTPUT_SUFFIX)


def collect_npy_files(input_dir: Path) -> list:
    """扫描 input-dir 下的 *.npy，按文件名排序。"""
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".npy"
    )


# ============================================================
# 报告输出
# ============================================================

def format_metric(value):
    """None -> 空字符串，其余保持原值（交给 csv 处理）。"""
    if value is None:
        return ""

    return value


def write_quality_csv(csv_path: Path, rows: list) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=QUALITY_CSV_FIELDNAMES)

        writer.writeheader()
        writer.writerows(rows)


def write_name_list(txt_path: Path, names: list) -> None:
    with open(txt_path, "w", encoding="utf-8") as file:
        for name in names:
            file.write(f"{name}\n")


# ============================================================
# 主流程
# ============================================================

def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir).expanduser().resolve()

    if not input_dir.is_dir():
        print(f"[ERROR] --input-dir 不存在或不是目录: {input_dir}")
        return 1

    output_dir = resolve_output_dir(input_dir, args.output_dir)

    if output_dir == input_dir:
        print(
            "[ERROR] --output-dir 不能与 --input-dir 相同，"
            "否则会覆盖原始 NPY"
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    npy_files = collect_npy_files(input_dir)

    if not npy_files:
        print(f"[WARN] --input-dir 下没有 *.npy: {input_dir}")

    quality_rows = []
    invalid_names = []
    valid_names = []

    hole_ratios = []
    valid_range_ratios = []

    load_failed = 0

    for npy_path in tqdm(npy_files, desc="depth clean", unit="npy"):
        try:
            depth = np.load(npy_path)
        except Exception as error:
            load_failed += 1

            tqdm.write(f"[WARN] 读取失败，已跳过: {npy_path.name} ({error})")

            continue

        is_valid, metrics = evaluate_depth(
            depth,
            args.depth_low,
            args.depth_high,
            args.max_hole_ratio,
            args.min_valid_range_ratio,
        )

        # 即使 evaluate_depth 判定为 invalid，也继续保存修补版本
        depth_filled = fill_zihang_filter(depth)

        np.save(output_dir / npy_path.name, depth_filled)

        if is_valid:
            valid_names.append(npy_path.name)
        else:
            invalid_names.append(npy_path.name)

        hole_ratios.append(metrics["hole_ratio"])
        valid_range_ratios.append(metrics["valid_range_ratio"])

        quality_rows.append({
            "filename": npy_path.name,
            "is_valid": is_valid,
            "hole_ratio": metrics["hole_ratio"],
            "valid_range_ratio": metrics["valid_range_ratio"],
            "valid_pixels": metrics["valid_pixels"],
            "median_depth_mm": format_metric(metrics["median_depth_mm"]),
            "total_pixels": metrics["total_pixels"],
            "invalid_pixels": metrics["invalid_pixels"],
            "valid_range_pixels": metrics["valid_range_pixels"],
        })

    # --------------------------------------------------------
    # 报告
    # --------------------------------------------------------

    quality_csv_path = output_dir / QUALITY_CSV_NAME
    invalid_txt_path = output_dir / INVALID_TXT_NAME
    valid_txt_path = output_dir / VALID_TXT_NAME

    write_quality_csv(quality_csv_path, quality_rows)
    write_name_list(invalid_txt_path, invalid_names)
    write_name_list(valid_txt_path, valid_names)

    # --------------------------------------------------------
    # 摘要
    # --------------------------------------------------------

    total_count = len(npy_files)
    valid_count = len(valid_names)
    invalid_count = len(invalid_names)

    invalid_ratio = (
        invalid_count / total_count * 100.0
        if total_count > 0
        else 0.0
    )

    processed_count = len(quality_rows)

    mean_hole_ratio = (
        float(np.mean(hole_ratios))
        if processed_count > 0
        else 0.0
    )

    mean_valid_range_ratio = (
        float(np.mean(valid_range_ratios))
        if processed_count > 0
        else 0.0
    )

    print()
    print("=" * 70)
    print("DEPTH CLEAN SUMMARY")
    print("=" * 70)
    print(f"Input dir        : {input_dir}")
    print(f"Output dir       : {output_dir}")
    print(f"Depth range      : {args.depth_low:g} ~ {args.depth_high:g} mm")
    print(f"Total depth NPY  : {total_count}")
    print(f"Valid depth      : {valid_count}")
    print(f"Invalid depth    : {invalid_count}")
    print(f"Invalid ratio    : {invalid_ratio:.2f} %")
    print(f"Mean hole ratio  : {mean_hole_ratio:.4f}")
    print(f"Mean valid ratio : {mean_valid_range_ratio:.4f}")

    if load_failed > 0:
        print(f"Load failed      : {load_failed}")

    print()
    print(f"depth_quality.csv  : {quality_csv_path}")
    print(f"invalid_depths.txt : {invalid_txt_path}")
    print(f"valid_depths.txt   : {valid_txt_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
