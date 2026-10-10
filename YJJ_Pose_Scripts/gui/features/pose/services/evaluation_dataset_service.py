"""统一评估数据集入口（不依赖 Qt；mAP/NME/Depth 消融共用）。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .dataset_service import DatasetService


class EvaluationDatasetService:
    """解析一个数据集根目录，并确保 RGB/RGBD test 真正来自该目录。"""

    @staticmethod
    def inspect(root: str | Path) -> dict[str, Any]:
        if not str(root).strip():
            return {"ok": False, "message": "请选择评估数据集目录（包含 rgb/、rgbd/）"}
        base = Path(root).expanduser()
        rgb_yaml = base / "rgb" / "data_rgb.yaml"
        rgbd_yaml = base / "rgbd" / "data_rgbd.yaml"
        info: dict[str, Any] = {
            "ok": False, "root": base,
            "rgb_yaml": rgb_yaml, "rgbd_yaml": rgbd_yaml,
            "rgb_cnt": 0, "rgbd_cnt": 0,
        }
        if not base.is_dir():
            info["message"] = f"数据集目录不存在：{base}"
            return info
        for name, yaml in (("RGB", rgb_yaml), ("RGBD", rgbd_yaml)):
            if not yaml.is_file():
                info["message"] = f"缺少 {name} YAML：{yaml}"
                return info

        rgb_dir, rgb_cnt, rgb_ids = DatasetService.resolve_test_set(rgb_yaml)
        rgbd_dir, rgbd_cnt, rgbd_ids = DatasetService.resolve_test_set(rgbd_yaml)
        info.update(rgb_dir=rgb_dir, rgbd_dir=rgbd_dir,
                    rgb_cnt=rgb_cnt, rgbd_cnt=rgbd_cnt)
        expected_rgb = base / "rgb" / "images" / "test"
        expected_rgbd = base / "rgbd" / "images" / "test"
        # 防止 YAML 的 path: 仍是旧数据集路径，导致界面选 A 实际跑 B。
        for name, observed, expected in (
            ("RGB", rgb_dir, expected_rgb), ("RGBD", rgbd_dir, expected_rgbd)
        ):
            if observed is None or observed.resolve() != expected.resolve():
                info["message"] = (f"{name} YAML 的 test 路径不是当前数据集："
                                   f"{observed}；应为 {expected}。请检查 YAML 的 path 字段。")
                return info
        if rgb_ids is None or rgbd_ids is None or not rgb_ids or not rgbd_ids:
            info["message"] = "RGB / RGBD test 图片缺失或为空"
            return info
        if rgb_ids != rgbd_ids:
            info["message"] = (f"RGB / RGBD test 图片 ID 不一致："
                               f"RGB={rgb_cnt}、RGBD={rgbd_cnt}")
            return info
        info["ok"] = True
        info["message"] = f"test：RGB {rgb_cnt} 张 | RGBD {rgbd_cnt} 张 | ID {len(rgb_ids)}/{len(rgb_ids)} 一致"
        return info
