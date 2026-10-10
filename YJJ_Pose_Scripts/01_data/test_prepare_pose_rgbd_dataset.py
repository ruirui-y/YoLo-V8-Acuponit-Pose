"""prepare_pose_rgbd_dataset 端到端数据集准备测试（输出目录与 YAML 路径回归）。

覆盖用户需求“八、测试要求 / Dataset YAML”全部 7 项：
21. 使用正式目录旁的临时 staging 准备数据集
22. 直接发布到 --output_dir（不再自动嵌套 dataset/hand 或 out）
23. RGB data_rgb.yaml 的 path 指向正式 rgb 目录
24. RGBD data_rgbd.yaml 的 path 指向正式 rgbd 目录
25. 两个 YAML 中都不存在 ".prepare_pose_rgbd_tmp"
26. train/val/test 的 images/labels 路径实际存在
27. RGB 与 RGBD 的 train/val/test 文件名集合一致，labels 与 images 配对一致

全程使用临时目录，不触碰真实数据。
启动：
    python YJJ_Pose_Scripts/01_data/test_prepare_pose_rgbd_dataset.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_pose_rgbd_dataset as prep  # noqa: E402


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        failures.append(name)


def _make_rgb_jpg(path: Path, w: int = 64, h: int = 48, seed: int = 0) -> None:
    """生成一张带纹理的 RGB jpg（纯色即可，像素内容对流程无影响）。"""
    rng = np.random.RandomState(seed)
    img = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)  # BGR 顺序写入 jpg


def _make_depth_npy(path: Path, w: int = 64, h: int = 48, seed: int = 0) -> None:
    """生成 uint16 raw depth：有效值域 [1150, 1800]，避开 invalid(0/65535)。"""
    rng = np.random.RandomState(seed + 1000)
    d = rng.randint(1150, 1800, (h, w), dtype=np.uint16)
    np.save(path, d)


def _make_label_txt(path: Path, kpt: int = 5) -> None:
    """生成一行合法 YOLO-Pose 标签：class cx cy w h + kpt 组 (x,y,v)。"""
    pts = []
    for k in range(kpt):
        x = 0.2 + 0.1 * k
        y = 0.3 + 0.08 * (k % 3)
        pts.append(f"{x:.6f} {y:.6f} 2")
    line = "0 0.500000 0.500000 0.900000 0.900000 " + " ".join(pts)
    path.write_text(line + "\n", encoding="utf-8")


def _read_yaml_path(yaml_path: Path) -> str:
    """读取 YAML 的 path: 行值（纯文本解析，避免引入 yaml 依赖）。"""
    for ln in yaml_path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("path:"):
            return ln.split(":", 1)[1].strip()
    raise AssertionError(f"{yaml_path} 缺少 path 行")


def main() -> None:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="prep_pose_test_") as td:
        base = Path(td)
        rgb_dir = base / "rgb_in"
        npy_dir = base / "npy_in"
        lab_dir = base / "lab_in"
        out_root = base / "hand_test"  # --output_dir 就是最终数据集目录
        for d in (rgb_dir, npy_dir, lab_dir):
            d.mkdir(parents=True, exist_ok=True)

        n_total = 12
        for i in range(n_total):
            ts = f"2026_{i:04d}"
            _make_rgb_jpg(rgb_dir / f"color_{ts}.jpg", seed=i)
            _make_depth_npy(npy_dir / f"depth_{ts}.npy", seed=i)
            _make_label_txt(lab_dir / f"color_{ts}.txt", kpt=5)

        # ---- 记录 shutil.move 的 staging -> 正式 发布轨迹 ----
        moves: list[tuple[str, str]] = []
        orig_move = prep.shutil.move

        def _spy_move(src, dst):
            moves.append((str(src), str(dst)))
            return orig_move(src, dst)

        prep.shutil.move = _spy_move
        try:
            sys.argv = [
                "prepare_pose_rgbd_dataset.py",
                "--rgb_dir", str(rgb_dir),
                "--depth_npy_dir", str(npy_dir),
                "--label_dir", str(lab_dir),
                "--output_dir", str(out_root),
                "--class_name", "hand",
                "--depth_low", "1100",
                "--depth_high", "1850",
            ]
            prep.main()
        finally:
            prep.shutil.move = orig_move

        print("[21] 使用正式目录旁的临时 staging 准备数据集")
        staged_src = [s for (s, _d) in moves if ".prepare_pose_rgbd_tmp" in s]
        _check("21a. 产物先写入 staging（存在 .prepare_pose_rgbd_tmp 源路径 move）",
               len(staged_src) >= 1, failures, f"staged moves={len(staged_src)}")
        _check("21b. 发布完成后 staging 目录被清理",
               not (base / ".hand_test.prepare_pose_rgbd_tmp").exists(), failures)

        print("[22] 最终发布到正式目录")
        final_ds = out_root
        _check("22a. --output_dir 本身就是最终数据集目录", final_ds.is_dir(), failures)
        _check("22b. hand_test/rgb 与 rgbd 存在",
               (final_ds / "rgb").is_dir() and (final_ds / "rgbd").is_dir(), failures)
        _check("22c. 有从 staging 目录 move 到正式位置的记录",
               any(".prepare_pose_rgbd_tmp" in s and "dataset" in s
                   for (s, _d) in moves), failures)
        _check("22d. 最终目录没有多余 out/ 或 dataset/hand 嵌套",
               not (out_root / "out").exists() and not (out_root / "dataset").exists(), failures)
        _check("22e. backup 目录被清理",
               not (base / ".hand_test.prepare_pose_rgbd_backup").exists(), failures)

        print("[23/24] YAML path 指向正式目录")
        rgb_y = final_ds / "rgb" / "data_rgb.yaml"
        rgbd_y = final_ds / "rgbd" / "data_rgbd.yaml"
        _check("23a. data_rgb.yaml 存在", rgb_y.is_file(), failures)
        _check("23b. RGB path == 正式 rgb 目录",
               Path(_read_yaml_path(rgb_y)).resolve() == (final_ds / "rgb").resolve(),
               failures, f"path={_read_yaml_path(rgb_y)}")
        _check("24a. data_rgbd.yaml 存在", rgbd_y.is_file(), failures)
        _check("24b. RGBD path == 正式 rgbd 目录",
               Path(_read_yaml_path(rgbd_y)).resolve() == (final_ds / "rgbd").resolve(),
               failures, f"path={_read_yaml_path(rgbd_y)}")

        print("[25] YAML 不含 .prepare_pose_rgbd_tmp")
        _check("25a. data_rgb.yaml 无 tmp 残留",
               ".prepare_pose_rgbd_tmp" not in rgb_y.read_text(encoding="utf-8"),
               failures)
        _check("25b. data_rgbd.yaml 无 tmp 残留",
               ".prepare_pose_rgbd_tmp" not in rgbd_y.read_text(encoding="utf-8"),
               failures)

        print("[26] train/val/test 路径实际存在且非空")
        all_ok_26 = True
        for split in ("train", "val", "test"):
            for sub in ("images", "labels"):
                for side in ("rgb", "rgbd"):
                    d = final_ds / side / sub / split
                    cnt = len(list(d.glob("*"))) if d.is_dir() else 0
                    if not (d.is_dir() and cnt > 0):
                        all_ok_26 = False
                        print(f"      缺失/为空: {d} (count={cnt})")
        _check("26a. rgb/rgbd 的 images/labels 全部 train/val/test 存在且非空",
               all_ok_26, failures)

        print("[27] RGB/RGBD split 文件集合一致，labels 与 images 配对一致")
        all_ok_27 = True
        for split in ("train", "val", "test"):
            def _stems(side: str, sub: str) -> set[str]:
                d = final_ds / side / sub / split
                return {p.stem for p in d.glob("*")} if d.is_dir() else set()

            rgb_img = _stems("rgb", "images")
            rgbd_img = _stems("rgbd", "images")
            rgb_lab = _stems("rgb", "labels")
            rgbd_lab = _stems("rgbd", "labels")
            if not (rgb_img == rgbd_img and rgb_lab == rgbd_lab):
                all_ok_27 = False
                print(f"      {split} RGB/RGBD 集合不一致: "
                      f"img={rgb_img ^ rgbd_img} lab={rgb_lab ^ rgbd_lab}")
            if not (rgb_img == rgb_lab and rgbd_img == rgbd_lab):
                all_ok_27 = False
                print(f"      {split} images 与 labels stem 不一致")
        _check("27a. train/val/test 的 RGB==RGBD 集合且 labels==images",
               all_ok_27, failures)

        print("[28] 已有目标目录安全保护")
        before = (final_ds / "dataset_report.json").read_bytes()
        old_argv = sys.argv
        try:
            sys.argv = [
                "prepare_pose_rgbd_dataset.py",
                "--rgb_dir", str(rgb_dir),
                "--depth_npy_dir", str(npy_dir),
                "--label_dir", str(lab_dir),
                "--output_dir", str(final_ds),
                "--class_name", "hand",
            ]
            try:
                prep.main()
            except RuntimeError as err:
                blocked = "拒绝覆盖" in str(err)
            else:
                blocked = False
        finally:
            sys.argv = old_argv
        _check("28a. 默认不覆盖已有非空数据集", blocked, failures)
        _check("28b. 拒绝后旧数据报告不变",
               before == (final_ds / "dataset_report.json").read_bytes(), failures)

        print("[29] 已有数据集 --force 校验失败后回滚")
        # 人为制造发布后校验失败，确保恢复原有数据集且不会遗留 backup。
        original_verifier = prep.verify_published_yaml
        old_argv = sys.argv
        def _fail_verify(_path):
            raise RuntimeError("simulated post-publish validation failure")
        try:
            prep.verify_published_yaml = _fail_verify
            sys.argv = [
                "prepare_pose_rgbd_dataset.py",
                "--rgb_dir", str(rgb_dir),
                "--depth_npy_dir", str(npy_dir),
                "--label_dir", str(lab_dir),
                "--output_dir", str(final_ds),
                "--class_name", "hand",
                "--depth_low", "1100",
                "--depth_high", "1850",
                "--force",
            ]
            try:
                prep.main()
            except RuntimeError as err:
                rolled_back = "回滚" in str(err) or "恢复" in str(err)
            else:
                rolled_back = False
        finally:
            prep.verify_published_yaml = original_verifier
            sys.argv = old_argv
        _check("29a. 发布后校验失败会报错", rolled_back, failures)
        _check("29b. 旧数据报告恢复成功",
               before == (final_ds / "dataset_report.json").read_bytes(), failures)
        _check("29c. 故障后临时目录与备份目录已清理",
               not (base / ".hand_test.prepare_pose_rgbd_tmp").exists()
               and not (base / ".hand_test.prepare_pose_rgbd_backup").exists(), failures)

    print("-" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
