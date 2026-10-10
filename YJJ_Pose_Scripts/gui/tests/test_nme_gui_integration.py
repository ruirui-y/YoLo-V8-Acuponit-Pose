"""NME GUI 后端轻量单测：不加载 YOLO、不执行推理、无需 PySide6。"""
from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))
from features.pose.services.nme_result_service import NmeResultService
from features.pose.services.command_service import CommandService


class NmeSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_csv(self, rows: list[dict], *, kcount: int = 2) -> Path:
        base = ["image", "rgb_matched", "rgbd_matched", "common_matched",
                "rgb_nme", "rgbd_nme", "winner"]
        keys = [f"{leg}_k{k}_px" for leg in ("rgb", "rgbd") for k in range(kcount)]
        path = self.root / "per_image.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=base + keys)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_common_matched_and_missing_keypoint(self) -> None:
        csv_path = self.make_csv([
            {"image":"a.jpg", "rgb_matched":True, "rgbd_matched":True,
             "common_matched":True, "rgb_nme":0.1, "rgbd_nme":0.05,
             "winner":"RGBD", "rgb_k0_px":20, "rgbd_k0_px":10,
             "rgb_k1_px":10, "rgbd_k1_px":15},
            {"image":"b.jpg", "rgb_matched":False, "rgbd_matched":True,
             "common_matched":False, "winner":"", "rgbd_k0_px":7},
            {"image":"c.jpg", "rgb_matched":True, "rgbd_matched":True,
             "common_matched":True, "rgb_nme":0.2, "rgbd_nme":0.1,
             "winner":"RGBD", "rgb_k0_px":40, "rgbd_k0_px":30,
             "rgb_k1_px":"", "rgbd_k1_px":30},
            {"image":"d.jpg", "rgb_matched":False, "rgbd_matched":False,
             "common_matched":False, "winner":"ERROR"},
        ])
        s = NmeResultService.read_summary(csv_path)
        self.assertEqual((s["total"], s["rgb_matched"], s["rgbd_matched"],
                          s["common"], s["rgbd_only"], s["neither"]),
                         (4, 2, 3, 2, 1, 1))
        self.assertEqual(s["rgbd_wins"], 2)
        self.assertEqual(s["error_count"], 1)
        self.assertAlmostEqual(s["rgb_nme"], 0.15)
        self.assertAlmostEqual(s["rgbd_nme"], 0.075)
        self.assertAlmostEqual(s["delta_nme"], -0.075)
        kp0, kp1 = s["keypoints"]
        self.assertEqual((kp0["count"], kp0["rgbd_wins"], kp0["rgb_wins"]), (2, 2, 0))
        self.assertEqual((kp0["rgb_mean"], kp0["rgbd_mean"]), (30, 20))
        self.assertEqual((kp1["count"], kp1["rgb_wins"], kp1["rgbd_wins"]), (1, 1, 0))
        self.assertEqual((kp1["rgb_mean"], kp1["rgbd_mean"]), (10, 15))

    def test_no_common_matched_is_valid(self) -> None:
        p = self.make_csv([
            {"image":"a.jpg", "rgb_matched":False, "rgbd_matched":True,
             "common_matched":False, "winner":""},
        ])
        s = NmeResultService.read_summary(p)
        self.assertEqual(s["common"], 0)
        self.assertIsNone(s["rgb_nme"])
        self.assertEqual(s["keypoints"][0]["count"], 0)

    def test_empty_or_missing_csv_raises(self) -> None:
        p = self.root / "missing.csv"
        with self.assertRaises(FileNotFoundError):
            NmeResultService.read_summary(p)
        with self.assertRaisesRegex(ValueError, "没有图片结果"):
            NmeResultService.read_summary(self.make_csv([]))

    def test_validate_dataset_root(self) -> None:
        r = self.root / "hand_cross_subject"
        rgb = r / "rgb" / "images" / "test"
        rgbd = r / "rgbd" / "images" / "test"
        labels = r / "rgb" / "labels" / "test"
        for d in (rgb, rgbd, labels):
            d.mkdir(parents=True)
        (rgb / "color_abc.jpg").touch()
        (rgbd / "color_abc.png").touch()
        (labels / "color_abc.txt").touch()
        valid, _, count = NmeResultService.validate_dataset_root(r)
        self.assertTrue(valid)
        self.assertEqual(count, 1)
        (rgbd / "color_abc.png").unlink()
        valid, msg, _ = NmeResultService.validate_dataset_root(r)
        self.assertFalse(valid)
        self.assertIn("不一致", msg)

    def test_command_matches_existing_evaluator(self) -> None:
        cmd = CommandService.build_nme_command(
            Path("C:/test/hand"), Path("C:/best_rgb.pt"),
            Path("C:/best_rgbd.pt"), Path("C:/results"), Path("C:/depth"))
        self.assertIn("--all", cmd)
        self.assertIn("--dataset-root", cmd)
        self.assertEqual(cmd[cmd.index("--conf") + 1], "0.001")
        self.assertEqual(cmd[cmd.index("--match-iou") + 1], "0.50")
        self.assertEqual(cmd[cmd.index("--output-dir") + 1], "C:/results")
        self.assertEqual(cmd[cmd.index("--depth-dir") + 1], "C:/depth")
        no_depth = CommandService.build_nme_command(
            Path("a"), Path("b"), Path("c"), Path("d"))
        self.assertNotIn("--depth-dir", no_depth)


if __name__ == "__main__":
    unittest.main()
