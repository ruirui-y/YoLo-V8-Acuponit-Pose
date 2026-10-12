"""公共测试集筛选结果的服务（不依赖 Qt）。

风格同 NmeResultService：只读筛选脚本的产物，不重算预测；
GUI 只负责展示与校验。
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class TestsetFilterService:

    _IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})

    # ---------------------------------------------------------------- 校验
    @staticmethod
    def validate_dataset_root(root: str | Path) -> tuple[bool, str, int]:
        """检查源数据集是否含可用的 test 图片，并返回按 stem 对齐后的张数。"""
        text = str(root).strip()
        if not text:
            return False, "请选择源测试集目录（含 rgb/ 与/或 rgbd/）", 0
        base = Path(text).expanduser()
        if not base.is_dir():
            return False, f"源数据集目录不存在：{base}", 0

        def stems(directory: Path) -> set[str]:
            if not directory.is_dir():
                return set()
            return {p.stem for p in directory.iterdir()
                    if p.is_file() and p.suffix.lower() in TestsetFilterService._IMAGE_EXTS}

        rgb = stems(base / "rgb" / "images" / "test")
        rgbd = stems(base / "rgbd" / "images" / "test")
        if not rgb and not rgbd:
            return False, "源数据集缺少 rgb/images/test 与 rgbd/images/test", 0
        if rgb and rgbd:
            common = rgb & rgbd
            if not common:
                return False, f"rgb 与 rgbd 的 test stem 无交集（rgb={len(rgb)}，rgbd={len(rgbd)}）", 0
            return True, "", len(common)
        return True, "", len(rgb or rgbd)

    @staticmethod
    def validate_weights(paths: list[str]) -> tuple[bool, str]:
        if not paths:
            return False, "请至少添加一个权重（点击「+ 添加权重」）"
        missing = [p for p in paths if not Path(p).expanduser().is_file()]
        if missing:
            return False, "以下权重不存在：" + "；".join(missing[:3])
        return True, f"共 {len(paths)} 个权重"

    @staticmethod
    def validate_output_dir(path: str) -> tuple[bool, str]:
        text = str(path).strip()
        if not text:
            return False, "请指定公共测试集的输出目录"
        target = Path(text).expanduser()
        if target.exists() and not target.is_dir():
            return False, f"输出路径已存在且不是目录：{target}"
        return True, ""

    # ---------------------------------------------------------------- 读产物
    @staticmethod
    def read_summary(json_path: Path) -> dict[str, Any]:
        """读取 filter_summary.json（筛选脚本写出的唯一摘要来源）。"""
        json_path = Path(json_path)
        data: Any = json.loads(json_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"筛选摘要格式异常（应为 JSON 对象）: {json_path}")
        return data

    @staticmethod
    def read_report(csv_path: Path) -> list[dict[str, str]]:
        """读取 filter_report.csv（每张图 × 每个权重的命中矩阵）。"""
        csv_path = Path(csv_path)
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
