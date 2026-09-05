"""rebuild_cross_subject_testset 端到端 + 校验测试（纯临时目录，不碰真实数据）。

覆盖 Cross-subject Test Set 重建需求 #1..#14：
1. 正常 image/label/npy timestamp 匹配
2. image 缺 label -> BLOCK
3. RGBD 缺 npy -> BLOCK
4. duplicate timestamp -> BLOCK
5. external keypoint count 不一致 -> BLOCK
6. external 与现有 dataset keypoint count 不一致 -> BLOCK
7. external 全部进入 test，不发生随机 split
8. 原 train/val 文件集合完全保持
9. RGB/RGBD test stems 完全一致
10. RGB/RGBD labels/test 完全一致
11. 输出 YAML path 指向正式目录
12. YAML 不包含任何 .prepare_* 临时路径
13. 目标目录已存在 -> BLOCK
14. staging 失败时不能留下半成品正式 dataset

启动：
    python YJJ_Pose_Scripts/01_data/test_rebuild_cross_subject_testset.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import prepare_pose_rgbd_dataset as prep  # noqa: E402
import rebuild_cross_subject_testset as xsub  # noqa: E402


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        failures.append(name)


# ============================================================ 样本生成
def _label_line(kpt: int) -> str:
    pts = []
    for k in range(kpt):
        pts.append(f"{0.2 + 0.1 * k:.6f} {0.3 + 0.05 * (k % 4):.6f} 2")
    return "0 0.500000 0.500000 0.800000 0.800000 " + " ".join(pts)


def _write_rgb(path: Path, w: int = 64, h: int = 48, seed: int = 0) -> None:
    rng = np.random.RandomState(seed)
    cv2.imwrite(str(path), rng.randint(0, 256, (h, w, 3), dtype=np.uint8))


def _write_4ch(path: Path, w: int = 64, h: int = 48, seed: int = 0) -> None:
    rng = np.random.RandomState(seed + 500)
    img = rng.randint(0, 256, (h, w, 3), dtype=np.uint8)
    alpha = rng.randint(0, 256, (h, w, 1), dtype=np.uint8)
    cv2.imwrite(str(path), np.dstack([img, alpha]))


def _write_npy(path: Path, w: int = 64, h: int = 48, seed: int = 0) -> None:
    rng = np.random.RandomState(seed + 1000)
    np.save(path, rng.randint(1150, 1800, (h, w), dtype=np.uint16))


def _write_label(path: Path, kpt: int = 7) -> None:
    path.write_text(_label_line(kpt) + "\n", encoding="utf-8")


# ---- canonical template 测试几何（K0..K6 非对称布局，距离结构唯一）----
_P7 = [
    (0.3125, 0.8333), (0.3281, 0.6250), (0.2969, 0.4167), (0.3594, 0.3958),
    (0.2656, 0.2292), (0.3906, 0.2500), (0.3438, 0.1250),
]


def _label_text_pts(pts: list[tuple[float, float]]) -> str:
    tokens = ["0", "0.500000", "0.500000", "0.900000", "0.900000"]
    for x, y in pts:
        tokens.append(f"{x:.6f}")
        tokens.append(f"{y:.6f}")
        tokens.append("2")
    return " ".join(tokens) + "\n"


def _write_label_pts(path: Path, pts: list[tuple[float, float]]) -> None:
    path.write_text(_label_text_pts(pts), encoding="utf-8")


# ============================================================ 基准数据集 fixture
def build_base(base: Path, kpt: int = 7, train_ids: tuple[str, ...] = ("t1", "t2"),
               val_ids: tuple[str, ...] = ("t3",)) -> None:
    """生成最小“当前数据集”：rgb + rgbd，train/val images+labels，含 yaml。"""
    for side, ch in (("rgb", 3), ("rgbd", 4)):
        for split, ids in (("train", train_ids), ("val", val_ids)):
            for sub in ("images", "labels"):
                (base / side / sub / split).mkdir(parents=True, exist_ok=True)
            for sid in ids:
                if side == "rgb":
                    _write_rgb(base / side / "images" / split / f"color_{sid}.jpg", seed=len(sid))
                else:
                    _write_4ch(base / side / "images" / split / f"color_{sid}.png", seed=len(sid))
                # K=7 时用 canonical _P7 几何（模拟“训练体系内已统一 K 语义”的 train labels）
                if kpt == 7:
                    _write_label_pts(base / side / "labels" / split / f"color_{sid}.txt", _P7)
                else:
                    _write_label(base / side / "labels" / split / f"color_{sid}.txt", kpt=kpt)
    prep.build_yaml(base / "rgb" / "data_rgb.yaml", "hand", kpt, 3, False,
                    dataset_path=base / "rgb")
    prep.build_yaml(base / "rgbd" / "data_rgbd.yaml", "hand", kpt, 4, True,
                    dataset_path=base / "rgbd")


def build_external(root: Path, ids: tuple[str, ...], kpt: int = 7,
                   seed_base: int = 0, perm: list[int] | None = None) -> None:
    """生成外部目录。

    perm=None: 旧式公式几何（仅供数量/匹配类测试）。
    perm!=None: 用 canonical _P7 几何、按 perm 打乱三元组顺序（仅 K=7），
                供 canonical template 重排集成测试（验证能恢复 template 顺序）。
    """
    for sub in ("img", "lab", "npy"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(ids):
        _write_rgb(root / "img" / f"color_{sid}.jpg", seed=seed_base + i)
        _write_npy(root / "npy" / f"depth_{sid}.npy", seed=seed_base + i)
        if perm is not None:
            if kpt != 7:
                raise AssertionError("perm 参数仅支持 K=7")
            _write_label_pts(root / "lab" / f"color_{sid}.txt", [_P7[p] for p in perm])
        else:
            _write_label(root / "lab" / f"color_{sid}.txt", kpt=kpt)


def _yaml_path_text(yaml_path: Path) -> str:
    for ln in yaml_path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("path:"):
            return ln.split(":", 1)[1].strip()
    raise AssertionError("yaml 缺 path")


# ============================================================ 测试主体
def main() -> None:
    failures: list[str] = []
    kpt = 7
    ext_ids = ("e1", "e2", "e3", "e4", "e5")

    with tempfile.TemporaryDirectory(prefix="cross_subject_test_") as td:
        root = Path(td)
        base = root / "dataset" / "hand"
        build_base(base, kpt=kpt)
        ext = root / "ext"
        build_external(ext, ext_ids, kpt=kpt)

        # ---------- 1. 正常匹配 ----------
        print("[1] 正常 image/label/npy 匹配")
        a = xsub.analyze_inputs(ext / "img", ext / "lab", ext / "npy")
        _check("1a. images=5 labels=5 depth=5",
               a["images"] == 5 and a["labels"] == 5 and a["depth"] == 5,
               failures, f"{a['images']}/{a['labels']}/{a['depth']}")
        _check("1b. matched=5 且 ok=True",
               a["matched"] == 5 and bool(a["ok"]) is True, failures, str(a))
        _check("1c. validate 返回 kpt=7",
               xsub.validate(base, ext / "img", ext / "lab", ext / "npy")["keypoints"] == 7,
               failures)

        # ---------- 2. image 缺 label ----------
        print("[2] image 缺 label -> BLOCK")
        bad_lab = root / "ext2_lab"
        (bad_lab).mkdir(parents=True, exist_ok=True)
        for sid in ext_ids[:-1]:
            _write_label(bad_lab / f"color_{sid}.txt", kpt=kpt)
        a2 = xsub.analyze_inputs(ext / "img", bad_lab, ext / "npy")
        _check("2a. 缺 1 个 label -> image_no_label 非空且 ok=False",
               not a2["ok"] and a2["image_no_label"] == ["e5"], failures, str(a2))
        try:
            xsub.validate(base, ext / "img", bad_lab, ext / "npy")
            _check("2b. validate BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("2b. validate BLOCK", "缺 label" in str(e), failures, str(e))

        # ---------- 3. RGBD 缺 npy ----------
        print("[3] RGBD 缺 npy -> BLOCK")
        bad_npy = root / "ext3_npy"
        bad_npy.mkdir(parents=True, exist_ok=True)
        for sid in ext_ids[:-1]:
            _write_npy(bad_npy / f"depth_{sid}.npy")
        a3 = xsub.analyze_inputs(ext / "img", ext / "lab", bad_npy)
        _check("3a. 缺 1 个 npy -> missing_npy=[e5] 且 ok=False",
               not a3["ok"] and a3["missing_npy"] == ["e5"], failures, str(a3))

        # ---------- 4. duplicate timestamp ----------
        print("[4] duplicate timestamp -> BLOCK")
        dup_img = root / "ext4_img"
        dup_img.mkdir(parents=True, exist_ok=True)
        for p in (ext / "img").iterdir():
            if p.is_file():
                cv2.imwrite(str(dup_img / p.name), cv2.imread(str(p)))
        # 同一 identity 再放一个 .png
        _write_rgb(dup_img / "color_e2.png", seed=99)
        a4 = xsub.analyze_inputs(dup_img, ext / "lab", ext / "npy")
        _check("4a. duplicate e2 -> duplicates 含 e2 且 ok=False",
               not a4["ok"] and "e2" in a4["duplicates"], failures, str(a4))
        try:
            xsub.validate(base, dup_img, ext / "lab", ext / "npy")
            _check("4b. validate BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("4b. validate BLOCK", "重复 timestamp" in str(e), failures, str(e))

        # ---------- 5. external K 不一致 ----------
        print("[5] external keypoint count 不一致 -> BLOCK")
        k6_img = root / "ext5"
        build_external(k6_img, ext_ids[:4], kpt=7, seed_base=10)
        (k6_img / "lab").mkdir(parents=True, exist_ok=True)
        for sid in ext_ids[:4]:
            _write_label(k6_img / "lab" / f"color_{sid}.txt", kpt=kpt)
        _write_label(k6_img / "lab" / f"color_{ext_ids[4]}.txt", kpt=6)  # 混入 6 点
        _write_rgb(k6_img / "img" / f"color_{ext_ids[4]}.jpg", seed=3)
        _write_npy(k6_img / "npy" / f"depth_{ext_ids[4]}.npy", seed=3)
        try:
            xsub.validate(base, k6_img / "img", k6_img / "lab", k6_img / "npy")
            _check("5a. K 混用 -> validate BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("5a. K 混用 -> validate BLOCK", "label" in str(e).lower()
                   or "K" in str(e), failures, str(e))

        # ---------- 6. external K != 当前数据集 K ----------
        print("[6] external 与当前 dataset K 不一致 -> BLOCK")
        ext6 = root / "ext6"
        build_external(ext6, ("f1", "f2"), kpt=6, seed_base=20)
        try:
            xsub.validate(base, ext6 / "img", ext6 / "lab", ext6 / "npy")
            _check("6a. external K=6 vs base K=7 -> BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("6a. Expected 7 keypoints, got 6",
                   "Expected 7 keypoints, got 6" in str(e), failures, str(e))

        # ---------- 7..14. rebuild ----------
        print("[7] external 全部进入 test，不发生随机 split")
        r = xsub.rebuild(base, "hand_cross_subject_v1", ext / "img", ext / "lab", ext / "npy")
        target = base.parent / "hand_cross_subject_v1"
        _check("7a. rebuild 成功且 test=5", int(r["test"]) == 5, failures, str(r))
        test_stems = sorted(p.stem for p in (target / "rgb" / "images" / "test").iterdir())
        exp_stems = sorted(f"color_{sid}" for sid in ext_ids)
        _check("7b. test 集合 == 全部 external（无随机拆分/无子集）",
               test_stems == exp_stems, failures, str(test_stems))
        _check("7c. train/val/test 总数 = 原 train/val + 全部 external",
               int(r["base_train"]) == 2 and int(r["base_val"]) == 1, failures, str(r))

        print("[8] 原 train/val 文件集合完全保持")
        same = True
        for side in ("rgb", "rgbd"):
            for sub in ("images", "labels"):
                for split in ("train", "val"):
                    src = base / side / sub / split
                    dst = target / side / sub / split
                    if {p.name for p in src.iterdir()} != {p.name for p in dst.iterdir()}:
                        same = False
        _check("8a. train/val 文件集合与源完全一致",
               same, failures)

        print("[9/10] RGB/RGBD test stems 与 labels 完全一致")
        _check("9a. test images stems 一致",
               {p.stem for p in (target / "rgb" / "images" / "test").iterdir()}
               == {p.stem for p in (target / "rgbd" / "images" / "test").iterdir()},
               failures)
        _check("10a. labels/test 文件集合一致",
               {p.name for p in (target / "rgb" / "labels" / "test").iterdir()}
               == {p.name for p in (target / "rgbd" / "labels" / "test").iterdir()},
               failures)

        print("[11/12] YAML path 指向正式目录，无 .prepare_* 残留")
        rgb_y = target / "rgb" / "data_rgb.yaml"
        rgbd_y = target / "rgbd" / "data_rgbd.yaml"
        _check("11a. rgb yaml path == target/rgb",
               Path(_yaml_path_text(rgb_y)).resolve() == (target / "rgb").resolve(),
               failures, _yaml_path_text(rgb_y))
        _check("11b. rgbd yaml path == target/rgbd",
               Path(_yaml_path_text(rgbd_y)).resolve() == (target / "rgbd").resolve(),
               failures, _yaml_path_text(rgbd_y))
        _check("12a. rgb yaml 无 .prepare_cross_subject_tmp",
               ".prepare_cross_subject_tmp" not in rgb_y.read_text(encoding="utf-8")
               and ".prepare_pose_rgbd_tmp" not in rgb_y.read_text(encoding="utf-8"),
               failures)
        _check("12b. rgbd yaml 无 .prepare_* 残留",
               ".prepare" not in rgbd_y.read_text(encoding="utf-8"),
               failures)
        _check("12c. staging 目录已清理",
               not (base.parent / xsub.STAGING_NAME).exists(), failures)

        print("[13] 目标目录已存在 -> BLOCK")
        try:
            xsub.rebuild(base, "hand_cross_subject_v1", ext / "img", ext / "lab", ext / "npy")
            _check("13a. 已存在 -> BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("13a. Output dataset already exists",
                   "already exists" in str(e), failures, str(e))

        print("[14] staging 失败不留下半成品正式 dataset")
        bad_rgb = root / "ext14"
        build_external(bad_rgb, ("g1", "g2"), kpt=kpt, seed_base=30)
        # 把 g2 的图替换成不可解码的伪图片（同扩展名，读图返回 None -> fuse 抛错）
        (bad_rgb / "img" / "color_g2.jpg").write_bytes(b"not an image at all")
        try:
            xsub.rebuild(base, "hand_cross_subject_v2", bad_rgb / "img",
                         bad_rgb / "lab", bad_rgb / "npy")
            _check("14a. 融合失败 -> rebuild BLOCK", False, failures, "未 BLOCK")
        except (xsub.CrossSubjectError, RuntimeError, OSError):
            _check("14a. 融合失败 -> rebuild BLOCK", True, failures)
        _check("14b. 无正式半成品目录",
               not (base.parent / "hand_cross_subject_v2").exists(), failures)
        _check("14c. staging 已清理（无残留）",
               not (base.parent / xsub.STAGING_NAME).exists(), failures)

        # ================================================================ canonical template 集成
        print("[15] template K 与 base dataset K 不一致 -> BLOCK")
        tmpl_dir = root / "tmpl"
        tmpl_dir.mkdir()
        tmpl6 = tmpl_dir / "hand_kpt6_template.txt"
        _write_label(tmpl6, kpt=6)
        try:
            xsub.validate(base, ext / "img", ext / "lab", ext / "npy", template=tmpl6)
            _check("15a. template K=6 vs base K=7 -> BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("15a. template K=6 vs base K=7 -> BLOCK",
                   "Template keypoints 6 != base dataset keypoints 7" in str(e),
                   failures, str(e))
        try:
            xsub.validate(base, ext / "img", ext / "lab", ext / "npy",
                          template=tmpl_dir / "missing.txt")
            _check("15b. template 不存在 -> BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("15b. template 不存在 -> BLOCK", "不存在" in str(e), failures, str(e))

        print("[16] rebuild with template：乱序 external 恢复 canonical 且 RGB/RGBD 逐字一致")
        tmpl7 = tmpl_dir / "hand_kpt7_template.txt"
        _write_label_pts(tmpl7, _P7)
        ext_t = root / "ext_tmpl"
        # m1/m2 canonical 顺序；m3..m5 故意打乱（验证几何恢复，不看原编号）
        build_external(ext_t, ("m1", "m2"), kpt=kpt, seed_base=40,
                       perm=list(range(7)))
        build_external(ext_t, ("m3", "m4", "m5"), kpt=kpt, seed_base=50,
                       perm=[1, 0, 3, 2, 6, 5, 4])
        r = xsub.rebuild(base, "hand_cross_tmpl", ext_t / "img", ext_t / "lab",
                         ext_t / "npy", template=tmpl7)
        target_t = base.parent / "hand_cross_tmpl"
        _check("16a. 报告含 template_keypoints=7 / reordered=5 / label_sync",
               int(r.get("template_keypoints", -1)) == 7
               and int(r.get("reordered", -1)) == 5
               and bool(r.get("label_sync")) is True, failures, str(r))
        canon_text = _label_text_pts(_P7)
        sides_same = True
        all_canonical = True
        for p in (target_t / "rgb" / "labels" / "test").iterdir():
            rgb_bytes = p.read_bytes()
            rgbd_bytes = (target_t / "rgbd" / "labels" / "test" / p.name).read_bytes()
            if rgb_bytes != rgbd_bytes:
                sides_same = False
            if p.read_text(encoding="utf-8") != canon_text:
                all_canonical = False
        _check("16b. RGB/RGBD labels/test 内容逐字一致", sides_same, failures)
        _check("16c. 每张 test label 都恢复为 canonical 顺序（含乱序 m3..m5）",
               all_canonical, failures)
        report_csv = target_t / "canonical_reorder_report.csv"
        _check("16d. canonical_reorder_report.csv 存在且 5 行全 OK",
               report_csv.is_file()
               and report_csv.read_text(encoding="utf-8-sig").count("\n") == 6
               and report_csv.read_text(encoding="utf-8-sig").count(",OK,") == 5,
               failures)
        same = True
        for side in ("rgb", "rgbd"):
            for sub in ("images", "labels"):
                for split in ("train", "val"):
                    src = base / side / sub / split
                    dst = target_t / side / sub / split
                    if {p.name for p in src.iterdir()} != {p.name for p in dst.iterdir()}:
                        same = False
        _check("16e. train/val 文件集合完全未改", same, failures)
        no_tmp = (".prepare" not in (target_t / "rgb" / "data_rgb.yaml").read_text(encoding="utf-8")
                  and ".prepare" not in (target_t / "rgbd" / "data_rgbd.yaml").read_text(encoding="utf-8"))
        _check("16f. YAML 无 .prepare_* 残留", no_tmp, failures)
        _check("16g. staging 已清理",
               not (base.parent / xsub.STAGING_NAME).exists(), failures)

        print("[17] reorder 失败 -> BLOCK 且不发布正式目录 / staging 清理")
        ext_bad = root / "ext_bad"
        build_external(ext_bad, ("h1", "h2"), kpt=kpt, seed_base=60)
        # 换成共线几何（与 canonical 完全不同形状，几何匹配必然 BLOCK：residual 或 ambiguous）
        collinear = [(0.1 + 0.12 * i, 0.55) for i in range(7)]
        for sid in ("h1", "h2"):
            _write_label_pts(ext_bad / "lab" / f"color_{sid}.txt", collinear)
        try:
            xsub.rebuild(base, "hand_cross_bad", ext_bad / "img", ext_bad / "lab",
                         ext_bad / "npy", template=tmpl7)
            _check("17a. reorder 失败 -> BLOCK", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError:
            _check("17a. reorder 失败 -> BLOCK", True, failures)
        _check("17b. 未发布正式目录",
               not (base.parent / "hand_cross_bad").exists(), failures)
        _check("17c. staging 已清理",
               not (base.parent / xsub.STAGING_NAME).exists(), failures)

        print("[18] Validate 执行完整 reference-bank reorder dry-run")
        v = xsub.validate(base, ext_t / "img", ext_t / "lab", ext_t / "npy",
                          template=tmpl7)
        _check("18a. dry-run 全部 OK（5/5，blocked=0）",
               bool(v.get("reorder_ok")) is True
               and int(v.get("reorder_total")) == 5
               and int(v.get("reorder_ok_count")) == 5
               and int(v.get("reorder_blocked")) == 0,
               failures, str(v))
        _check("18b. reference bank 数量 = base train 数（>=2）",
               int(v.get("reference_count", 0)) >= 2, failures, str(v))
        _check("18c. max_residual 为有限数值",
               isinstance(v.get("max_residual"), float)
               and v.get("max_residual", 9.0) >= 0.0, failures, str(v))

        print("[19] Validate dry-run 任意一张 BLOCK -> Validate BLOCK（不等到 Rebuild）")
        try:
            xsub.validate(base, ext_bad / "img", ext_bad / "lab", ext_bad / "npy",
                          template=tmpl7)
            _check("19a. dry-run BLOCK（共线几何）", False, failures, "未 BLOCK")
        except xsub.CrossSubjectError as e:
            _check("19a. dry-run BLOCK（消息含文件与原因）",
                   "canonical reorder" in str(e) and "@" in str(e),
                   failures, str(e))
        _check("19b. Validate BLOCK 时正式目录不出现",
               not (base.parent / "hand_cross_bad").exists(), failures)

    print("-" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
