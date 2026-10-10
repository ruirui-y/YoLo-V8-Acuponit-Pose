"""读取现有 NME 评估脚本写出的 CSV，生成 GUI 展示摘要。

不重新计算预测坐标、bbox 匹配或 NME；仅对 per_image.csv
按评估脚本相同的 common matched 规则做简单聚合。
"""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


class NmeResultService:
    _IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
    _REQUIRED_COLUMNS = frozenset({
        "image", "rgb_matched", "rgbd_matched", "common_matched",
        "rgb_nme", "rgbd_nme", "winner",
    })

    @staticmethod
    def validate_dataset_root(dataset_root: Path) -> tuple[bool, str, int]:
        """检查 NME 必需的 test 图片/标签是否存在且以 stem 一一对应。"""
        rgb_dir = dataset_root / "rgb" / "images" / "test"
        rgbd_dir = dataset_root / "rgbd" / "images" / "test"
        label_dir = dataset_root / "rgb" / "labels" / "test"
        for directory in (rgb_dir, rgbd_dir, label_dir):
            if not directory.is_dir():
                return False, f"缺少目录: {directory}", 0

        def stems(directory: Path, suffixes: frozenset[str]) -> set[str]:
            return {p.stem for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in suffixes}

        rgb_ids = stems(rgb_dir, NmeResultService._IMAGE_EXTS)
        rgbd_ids = stems(rgbd_dir, frozenset({".png", ".tif", ".tiff"}))
        label_ids = stems(label_dir, frozenset({".txt"}))
        if not rgb_ids:
            return False, "RGB test 中没有图片", 0
        if rgb_ids != rgbd_ids:
            return False, (f"RGB / RGBD test 图片 ID 不一致：RGB={len(rgb_ids)}，"
                           f"RGBD={len(rgbd_ids)}"), 0
        missing_labels = rgb_ids - label_ids
        if missing_labels:
            sample = sorted(missing_labels)[0]
            return False, (f"{len(missing_labels)} 张 test 图片缺少 GT 标签，"
                           f"例如 {sample}.txt"), 0
        return True, "", len(rgb_ids)

    @staticmethod
    def _is_true(text: str | None) -> bool:
        return str(text or "").strip().lower() in ("true", "1")

    @staticmethod
    def _valid_number(value: str | None) -> float | None:
        if value is None or not str(value).strip():
            return None
        try:
            number = float(value)
        except ValueError:
            return None
        return number if math.isfinite(number) else None

    @classmethod
    def read_summary(cls, csv_path: Path) -> dict[str, Any]:
        """与评估脚本的 FINAL NME / KPOINT SUMMARY 在同一 CSV 结果上统计。"""
        csv_path = Path(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            missing = cls._REQUIRED_COLUMNS - set(columns)
            if missing:
                raise ValueError(f"NME CSV 缺少字段: {', '.join(sorted(missing))}")
            rows = list(reader)

        if not rows:
            raise ValueError("NME CSV 没有图片结果")
        matched_rgb = sum(cls._is_true(row.get("rgb_matched")) for row in rows)
        matched_rgbd = sum(cls._is_true(row.get("rgbd_matched")) for row in rows)
        rgb_only = sum(cls._is_true(row.get("rgb_matched")) and
                       not cls._is_true(row.get("rgbd_matched")) for row in rows)
        rgbd_only = sum(cls._is_true(row.get("rgbd_matched")) and
                        not cls._is_true(row.get("rgb_matched")) for row in rows)
        neither = len(rows) - matched_rgb - rgbd_only  # 两侧均没有 matched
        error_count = sum(row.get("winner") == "ERROR" for row in rows)

        common_rows = [row for row in rows
                       if cls._is_true(row.get("common_matched"))]
        # Eval 写入的 common_matched 表示双方 NME 均有效，非仅 bbox matched。
        rgb_nmes = [cls._valid_number(row.get("rgb_nme")) for row in common_rows]
        rgbd_nmes = [cls._valid_number(row.get("rgbd_nme")) for row in common_rows]
        if any(v is None for v in rgb_nmes + rgbd_nmes):
            raise ValueError("NME CSV 中 common_matched 行含空值/非法 NME")

        rgb_mean = mean(rgb_nmes) if rgb_nmes else None
        rgbd_mean = mean(rgbd_nmes) if rgbd_nmes else None
        k_indices = sorted({int(m.group(1)) for col in columns
                            if (m := re.fullmatch(r"rgb_k(\d+)_px", col))
                            and f"rgbd_k{m.group(1)}_px" in columns})
        kpoints: list[dict[str, Any]] = []
        for k in k_indices:
            rgb_key, rgbd_key = f"rgb_k{k}_px", f"rgbd_k{k}_px"
            pairs = [(a, b) for row in common_rows
                     if (a := cls._valid_number(row.get(rgb_key))) is not None
                     and (b := cls._valid_number(row.get(rgbd_key))) is not None]
            kpoints.append({
                "keypoint": k,
                "count": len(pairs),
                "rgb_mean": mean(a for a, _ in pairs) if pairs else None,
                "rgbd_mean": mean(b for _, b in pairs) if pairs else None,
                "rgb_wins": sum(a < b for a, b in pairs),
                "rgbd_wins": sum(b < a for a, b in pairs),
                "ties": sum(a == b for a, b in pairs),
            })

        return {
            "total": len(rows), "rgb_matched": matched_rgb,
            "rgbd_matched": matched_rgbd, "common": len(common_rows),
            "rgb_only": rgb_only, "rgbd_only": rgbd_only, "neither": neither,
            "error_count": error_count,
            "rgb_nme": rgb_mean, "rgbd_nme": rgbd_mean,
            "delta_nme": rgbd_mean - rgb_mean if rgb_mean is not None else None,
            "rgb_wins": sum(r.get("winner") == "RGB" for r in common_rows),
            "rgbd_wins": sum(r.get("winner") == "RGBD" for r in common_rows),
            "ties": sum(r.get("winner") == "TIE" for r in common_rows),
            "keypoints": kpoints,
        }
