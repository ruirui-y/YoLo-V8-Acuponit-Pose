#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit YOLO Pose label keypoint ordering for the current Hand dataset.

Goal
----
Detect frames where keypoint IDs were likely swapped during labeling/export.
The script does NOT modify any label file.

Method
------
1. Read all YOLO Pose .txt labels recursively.
2. Convert each keypoint to bbox-local coordinates, removing global image
   translation/scale as much as possible.
3. Build a robust canonical keypoint template using the median position of
   each keypoint ID across the dataset.
4. For every label, test every single pair swap (K0<->K1, ..., K5<->K6).
5. If one swap makes the two involved points much closer to the canonical
   template, mark that file as SUSPECT.

This is intended for the current Hand session where pose/orientation changes
are small and most labels are assumed correct. A SUSPECT result means
"manual review strongly recommended", not an automatic rewrite.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path


NUM_KEYPOINTS = 7
EXPECTED_VALUES = 5 + NUM_KEYPOINTS * 3  # class + xywh + 7*(x,y,v)


@dataclass(frozen=True)
class Sample:
    path: Path
    bbox: tuple[float, float, float, float]
    points: tuple[tuple[float, float, float], ...]
    local_points: tuple[tuple[float, float], ...]


def _ParseArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit 7-keypoint YOLO Pose labels for likely keypoint-ID swaps."
    )
    parser.add_argument(
        "--labels-dir",
        required=True,
        help="Directory containing YOLO Pose label .txt files; searched recursively.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="CSV output path. Default: <labels-dir>/pose_label_order_report.csv",
    )
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=0.35,
        help=(
            "Flag when swapped pair error / original pair error is <= this value. "
            "Default: 0.35"
        ),
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.20,
        help=(
            "Minimum bbox-local pair-error reduction required to flag. "
            "Default: 0.20"
        ),
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Print every parsed label, not only suspects/errors.",
    )
    return parser.parse_args()


def _ParseLabel(path: Path) -> Sample:
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(lines) != 1:
        raise ValueError(f"expected exactly 1 object line, got {len(lines)}")

    parts = lines[0].split()
    if len(parts) != EXPECTED_VALUES:
        raise ValueError(
            f"expected {EXPECTED_VALUES} values for 7-keypoint pose label, got {len(parts)}"
        )

    values = [float(x) for x in parts]

    # values[0] = class id
    cx, cy, bw, bh = values[1:5]
    if bw <= 0.0 or bh <= 0.0:
        raise ValueError(f"invalid bbox size: w={bw}, h={bh}")

    points = []
    local_points = []
    offset = 5
    for k in range(NUM_KEYPOINTS):
        x = values[offset + k * 3]
        y = values[offset + k * 3 + 1]
        v = values[offset + k * 3 + 2]
        if v <= 0:
            raise ValueError(f"K{k} is not visible (v={v}); current audit requires all 7")
        points.append((x, y, v))

        # bbox-local coordinates make the audit less sensitive to the whole hand
        # moving/scaling in the image.
        local_x = (x - cx) / bw
        local_y = (y - cy) / bh
        local_points.append((local_x, local_y))

    return Sample(
        path=path,
        bbox=(cx, cy, bw, bh),
        points=tuple(points),
        local_points=tuple(local_points),
    )


def _MedianTemplate(samples: list[Sample]) -> tuple[tuple[float, float], ...]:
    template = []
    for k in range(NUM_KEYPOINTS):
        xs = [s.local_points[k][0] for s in samples]
        ys = [s.local_points[k][1] for s in samples]
        template.append((statistics.median(xs), statistics.median(ys)))
    return tuple(template)


def _Distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _MeanIdentityError(
    sample: Sample,
    template: tuple[tuple[float, float], ...],
) -> float:
    return sum(
        _Distance(sample.local_points[k], template[k])
        for k in range(NUM_KEYPOINTS)
    ) / NUM_KEYPOINTS


def _BestSingleSwap(
    sample: Sample,
    template: tuple[tuple[float, float], ...],
) -> tuple[int, int, float, float, float]:
    """
    Return:
      i, j,
      original_pair_error,
      swapped_pair_error,
      improvement
    """
    best = (-1, -1, 0.0, 0.0, 0.0)

    for i in range(NUM_KEYPOINTS):
        for j in range(i + 1, NUM_KEYPOINTS):
            pi = sample.local_points[i]
            pj = sample.local_points[j]

            original = _Distance(pi, template[i]) + _Distance(pj, template[j])
            swapped = _Distance(pi, template[j]) + _Distance(pj, template[i])
            improvement = original - swapped

            if improvement > best[4]:
                best = (i, j, original, swapped, improvement)

    return best


def _FormatTemplate(template: tuple[tuple[float, float], ...]) -> str:
    chunks = []
    for k, (x, y) in enumerate(template):
        chunks.append(f"K{k}=({x:+.3f},{y:+.3f})")
    return "  ".join(chunks)


def main() -> int:
    args = _ParseArgs()

    labels_dir = Path(args.labels_dir).expanduser().resolve()
    if not labels_dir.is_dir():
        raise SystemExit(f"[ERROR] labels directory not found: {labels_dir}")

    if not (0.0 < args.ratio_threshold < 1.0):
        raise SystemExit("[ERROR] --ratio-threshold must be between 0 and 1")
    if args.improvement_threshold < 0.0:
        raise SystemExit("[ERROR] --improvement-threshold must be >= 0")

    txt_files = sorted(labels_dir.rglob("*.txt"))
    if not txt_files:
        raise SystemExit(f"[ERROR] no .txt labels found under: {labels_dir}")

    samples: list[Sample] = []
    parse_errors: list[tuple[Path, str]] = []

    for path in txt_files:
        try:
            samples.append(_ParseLabel(path))
        except Exception as exc:
            parse_errors.append((path, str(exc)))

    if len(samples) < 5:
        raise SystemExit(
            "[ERROR] fewer than 5 valid labels. "
            "This detector needs a majority of correct frames to build a robust template."
        )

    template = _MedianTemplate(samples)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else labels_dir / "pose_label_order_report.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    suspects = []

    for sample in samples:
        mean_error = _MeanIdentityError(sample, template)
        i, j, original, swapped, improvement = _BestSingleSwap(sample, template)

        if original <= 1e-12:
            ratio = 1.0
        else:
            ratio = swapped / original

        is_suspect = (
            i >= 0
            and ratio <= args.ratio_threshold
            and improvement >= args.improvement_threshold
        )

        status = "SUSPECT_SWAP" if is_suspect else "OK"
        suggested_swap = f"K{i}<->K{j}" if is_suspect else ""

        try:
            rel_path = sample.path.relative_to(labels_dir)
        except ValueError:
            rel_path = sample.path

        row = {
            "status": status,
            "file": str(rel_path),
            "suggested_swap": suggested_swap,
            "mean_template_error": f"{mean_error:.6f}",
            "pair_original_error": f"{original:.6f}",
            "pair_swapped_error": f"{swapped:.6f}",
            "swap_ratio": f"{ratio:.6f}",
            "pair_improvement": f"{improvement:.6f}",
        }
        rows.append(row)

        if is_suspect:
            suspects.append(row)

        if args.show_all:
            suffix = f"  {suggested_swap}" if suggested_swap else ""
            print(f"[{status:12s}] {rel_path}{suffix}")

    for path, message in parse_errors:
        try:
            rel_path = path.relative_to(labels_dir)
        except ValueError:
            rel_path = path
        rows.append(
            {
                "status": "PARSE_ERROR",
                "file": str(rel_path),
                "suggested_swap": "",
                "mean_template_error": "",
                "pair_original_error": "",
                "pair_swapped_error": "",
                "swap_ratio": "",
                "pair_improvement": message,
            }
        )

    fieldnames = [
        "status",
        "file",
        "suggested_swap",
        "mean_template_error",
        "pair_original_error",
        "pair_swapped_error",
        "swap_ratio",
        "pair_improvement",
    ]
    with output_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("=== Pose label order audit ===")
    print(f"labels_dir : {labels_dir}")
    print(f"valid      : {len(samples)}")
    print(f"suspects   : {len(suspects)}")
    print(f"parse_error: {len(parse_errors)}")
    print(f"template   : {_FormatTemplate(template)}")
    print(f"report     : {output_path}")

    if suspects:
        print()
        print("=== Suspected swapped labels ===")
        for row in suspects:
            print(
                f"[SUSPECT] {row['file']}  "
                f"{row['suggested_swap']}  "
                f"ratio={row['swap_ratio']}  "
                f"improvement={row['pair_improvement']}"
            )
    else:
        print()
        print("No strong single-pair swap was detected.")

    if parse_errors:
        print()
        print("=== Parse errors ===")
        for path, message in parse_errors:
            print(f"[PARSE_ERROR] {path.name}: {message}")

    print()
    print("NOTE: This is a dataset-consistency audit, not an automatic ground-truth rewrite.")
    print("Review every SUSPECT visually before changing labels.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
