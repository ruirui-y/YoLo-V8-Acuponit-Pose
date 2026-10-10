"""统一评估目录路径验证：不加载 Qt/YOLO/模型。"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from features.pose.services.evaluation_dataset_service import EvaluationDatasetService


class EvalDatasetSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "hand_cross_subject"
        for modality in ("rgb", "rgbd"):
            (self.root / modality / "images" / "test").mkdir(parents=True)
            (self.root / modality / "labels" / "test").mkdir(parents=True)
            yaml = self.root / modality / f"data_{modality}.yaml"
            yaml.write_text(
                f"path: {self.root / modality}\n"
                "train: images/train\nval: images/val\ntest: images/test\n",
                encoding="utf-8",
            )
        for stem in ("color_a", "color_b"):
            (self.root / "rgb" / "images" / "test" / f"{stem}.jpg").write_bytes(b"abc")
            (self.root / "rgbd" / "images" / "test" / f"{stem}.png").write_bytes(b"def")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_valid_shared_root(self) -> None:
        info = EvaluationDatasetService.inspect(self.root)
        self.assertTrue(info["ok"], info.get("message"))
        self.assertEqual((info["rgb_cnt"], info["rgbd_cnt"]), (2, 2))
        self.assertEqual(info["root"], self.root)
        self.assertEqual(info["rgb_yaml"], self.root / "rgb" / "data_rgb.yaml")
        self.assertEqual(info["rgbd_yaml"], self.root / "rgbd" / "data_rgbd.yaml")

    def test_old_yaml_path_is_rejected(self) -> None:
        rgb_yaml = self.root / "rgb" / "data_rgb.yaml"
        rgb_yaml.write_text(
            f"path: {self.root.parent / 'old_hand' / 'rgb'}\ntest: images/test\n",
            encoding="utf-8",
        )
        info = EvaluationDatasetService.inspect(self.root)
        self.assertFalse(info["ok"])
        self.assertIn("YAML 的 test 路径不是当前数据集", info["message"])

    def test_mismatched_test_ids_are_rejected(self) -> None:
        (self.root / "rgbd" / "images" / "test" / "color_b.png").unlink()
        info = EvaluationDatasetService.inspect(self.root)
        self.assertFalse(info["ok"])
        self.assertIn("图片 ID 不一致", info["message"])

    def test_missing_yaml_is_rejected(self) -> None:
        (self.root / "rgbd" / "data_rgbd.yaml").unlink()
        info = EvaluationDatasetService.inspect(self.root)
        self.assertFalse(info["ok"])
        self.assertIn("缺少 RGBD YAML", info["message"])

    def test_missing_test_directory_is_rejected(self) -> None:
        (self.root / "rgbd" / "images" / "test" / "color_a.png").unlink()
        (self.root / "rgbd" / "images" / "test" / "color_b.png").unlink()
        info = EvaluationDatasetService.inspect(self.root)
        self.assertFalse(info["ok"])
        self.assertIn("test 图片缺失或为空", info["message"])


if __name__ == "__main__":
    unittest.main()
