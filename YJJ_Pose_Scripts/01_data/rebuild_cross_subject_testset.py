"""Cross-subject Test Set 重建（纯逻辑，无 Qt 依赖）。

功能：用“其他受试者”的外部 RGB / Label / Depth NPY 重建一套新的 test set，
train / val 原样沿用当前数据集（dataset/<class>/rgb 与 rgbd），不做任何随机拆分。

典型调用：
    python rebuild_cross_subject_testset.py validate \\
        --base <dataset/hand> --name hand_cross_subject_v1 \\
        --rgb-dir <subject_b/out_image> --label-dir <subject_b/image/labels> --npy-dir <subject_b/out_npy>
    python rebuild_cross_subject_testset.py rebuild ...（同上）

流程：
1. 按完整时间戳匹配三个目录（color_<ts>.<ext> / color_<ts>.txt / depth_<ts>.npy）
2. 只读校验（validate / rebuild 前均执行）：
   - label 非空 / YOLO Pose 格式合法 / 外部 K 一致
   - 外部 K == 当前数据集 K（读 base/rgb/data_rgb.yaml 的 kpt_shape）
   - 提供 --template 时：template 存在、格式合法、template K == base K
   - image 缺 label / label 缺 image / RGBD 缺 npy / 重复 timestamp -> BLOCK
3. rebuild：
   - staging = <dataset>/.prepare_cross_subject_tmp/<name>（先完整生成）
     * RGB train/val images+labels <- 原 rgb 目录复制
     * RGBD train/val images+labels <- 原 rgbd 目录复制
     * external 全部 -> rgb|rgbd 的 images/test + labels/test（不随机拆分）
     * RGBD test 4ch 图 = 外部 RGB + 对应 depth npy 用项目 fuse_pair_rawdepth 逻辑融合
   - 提供 --template 时：publish 前在 staging 内对 external test labels 做一次
     canonical template 几何重排（只排一次），同一份修正结果写入 rgb/labels/test
     与 rgbd/labels/test（逐字一致）；任何 reorder BLOCK（residual 超限 / 歧义 /
     K 不一致）都中止，绝不猜 ID
   - RGB/RGBD 各 split stem 集合、test labels 集合必须完全一致，否则发布失败
   - 全部成功后才 move 到正式目录 <dataset>/<name>，并把只读
     canonical_reorder_report.csv 写入新数据集根
   - 目标目录已存在 -> BLOCK（不自动覆盖）
4. 最终 YAML path 指向正式目录，禁止含任何 .prepare_* 临时路径

与 prepare_pose_rgbd_dataset.py 共享：RGB_PREFIX/DEPTH_PREFIX、fuse_pair_rawdepth、
build_yaml、verify_published_yaml、validate_labels；重排复用 pose_label_reorder.py。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pose_label_reorder as plr  # canonical template 重排核心（同 01_data 目录）

from prepare_pose_rgbd_dataset import (  # noqa: E402
    DEPTH_PREFIX,
    RGB_PREFIX,
    build_yaml,
    fuse_pair_rawdepth,
    validate_labels,
    verify_published_yaml,
)

# 与 GUI 默认 / prepare 默认一致的 depth 映射参数
DEFAULT_DEPTH_LOW = 1100.0
DEFAULT_DEPTH_HIGH = 1850.0

# staging 目录名（与正式目录同级，发布后删除）
STAGING_NAME = ".prepare_cross_subject_tmp"

IMG_EXTS = (".png", ".jpg", ".jpeg")

_JSON_TAG = "CROSS_SUBJECT_JSON"
_BLOCK_TAG = "CROSS_SUBJECT_BLOCK"


class CrossSubjectError(Exception):
    """业务级 BLOCK：message 即对用户的提示。"""


# ============================================================ 扫描 / 匹配
def scan_files(directory: str | Path, prefix: str, exts: tuple[str, ...]) -> tuple[dict[str, Path], list[str]]:
    """扫描目录中带指定前缀与扩展名的文件，按 sample identity 分组。

    返回 (id -> Path, duplicates)。重复 timestamp 会进 duplicates（同一 identity 多个文件）。
    """
    d = Path(directory)
    id_map: dict[str, Path] = {}
    duplicates: list[str] = []
    if not d.is_dir():
        return id_map, duplicates
    for p in sorted(d.iterdir()):
        if not p.is_file() or p.suffix.lower() not in exts:
            continue
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        sid = stem[len(prefix):]
        if not sid:
            continue
        if sid in id_map:
            duplicates.append(sid)
        else:
            id_map[sid] = p
    return id_map, duplicates


def _sorted_dups(dups: list[str]) -> list[str]:
    return sorted(set(dups))


def analyze_inputs(rgb_dir: str | Path, label_dir: str | Path,
                   npy_dir: str | Path) -> dict[str, object]:
    """只读分析外部输入，返回统计 + 各种不一致清单（不抛业务异常）。

    返回 dict（全部 JSON 可序列化）：
        images/labels/depth: 各自文件数
        matched: image∩label∩npy 的样本数
        image_no_label / label_no_image / missing_npy: 各自缺失 ID 清单
        duplicates: 重复 timestamp 清单
        ok: 是否可进入下一步
        error: ok=False 时的第一句提示
    """
    imgs, dup_i = scan_files(rgb_dir, RGB_PREFIX, IMG_EXTS)
    labs, dup_l = scan_files(label_dir, RGB_PREFIX, (".txt",))
    npys, dup_n = scan_files(npy_dir, DEPTH_PREFIX, (".npy",))

    img_ids = set(imgs)
    lab_ids = set(labs)
    npy_ids = set(npys)

    duplicates = _sorted_dups(dup_i + dup_l + dup_n)

    image_no_label = sorted(img_ids - lab_ids)
    label_no_image = sorted(lab_ids - img_ids)
    missing_npy = sorted((img_ids & lab_ids) - npy_ids)

    ok = True
    error = ""
    if duplicates:
        ok = False
        error = "重复 timestamp: " + ", ".join(duplicates[:10])
    elif image_no_label:
        ok = False
        error = "image 缺 label: " + ", ".join(image_no_label[:10])
    elif label_no_image:
        ok = False
        error = "label 没有对应 image: " + ", ".join(label_no_image[:10])
    elif missing_npy:
        ok = False
        error = "RGBD 缺 NPY: " + ", ".join(missing_npy[:10])

    matched = sorted(img_ids & lab_ids & npy_ids)
    return {
        "images": len(imgs),
        "labels": len(labs),
        "depth": len(npys),
        "matched": len(matched),
        "matched_ids": matched,
        "image_no_label": image_no_label,
        "label_no_image": label_no_image,
        "missing_npy": missing_npy,
        "duplicates": duplicates,
        "ok": ok,
        "error": error,
    }


def parse_kpt_from_yaml(yaml_path: str | Path) -> int | None:
    """极简解析 kpt_shape 第一个数字（兼容内联 / 多行块，不依赖 PyYAML）。"""
    try:
        lines = Path(yaml_path).read_text(encoding="utf-8").splitlines()
    except Exception:  # noqa: BLE001
        return None
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if not s.startswith("kpt_shape:"):
            continue
        rest = s[len("kpt_shape:"):].split("#", 1)[0].strip()
        if rest.startswith("["):
            nums = [x.strip() for x in rest[1:].split("]")[0].split(",") if x.strip()]
            if nums:
                try:
                    return int(nums[0])
                except ValueError:
                    return None
        elif rest.isdigit():
            return int(rest)
    return None


def base_keypoint_count(base: str | Path) -> int:
    """当前数据集关键点数量 = base/rgb/data_rgb.yaml 的 kpt_shape[0]。"""
    y = Path(base) / "rgb" / "data_rgb.yaml"
    k = parse_kpt_from_yaml(y)
    if k is None or k <= 0:
        raise CrossSubjectError(f"无法从当前数据集解析关键点数量: {y}")
    return k


# ============================================================ 校验
def validate(base: str | Path, rgb_dir: str | Path, label_dir: str | Path,
             npy_dir: str | Path,
             template: str | Path | None = None,
             audit_dir: str | Path | None = None) -> dict[str, object]:
    """完整只读校验，返回统计 dict；任何失败抛 CrossSubjectError。

    template: canonical label 路径。提供时额外要求：文件存在、格式合法、
              template K == 当前数据集 K == 外部 K，并对全部 external label
              执行 canonical reference-bank 全量 audit（不是第一张失败就退出）。
    audit_dir: 非 None 时把全量 canonical_reorder_audit.csv 写到此目录（不改正式 dataset）。
    """
    b = Path(base)
    if not b.is_dir():
        raise CrossSubjectError(f"当前数据集目录不存在: {b}")
    if not (b / "rgb" / "images" / "train").is_dir() or not (b / "rgbd" / "images" / "train").is_dir():
        raise CrossSubjectError(
            f"当前数据集不完整（缺少 rgb/images/train 或 rgbd/images/train）: {b}")
    if not (b / "rgb" / "data_rgb.yaml").is_file() or not (b / "rgbd" / "data_rgbd.yaml").is_file():
        raise CrossSubjectError(f"当前数据集缺少 data yaml: {b}")

    base_k = base_keypoint_count(b)

    # canonical template：存在 / 格式合法 / K 与 base 一致
    template_k = None
    if template is not None:
        tp = Path(template)
        if not tp.is_file():
            raise CrossSubjectError(f"Canonical Template Label 不存在: {tp}")
        try:
            template_k = plr.template_keypoint_count(tp)
        except plr.ReorderError as e:
            raise CrossSubjectError(f"Canonical Template Label 格式非法: {e}") from e
        if template_k != base_k:
            raise CrossSubjectError(
                f"Template keypoints {template_k} != base dataset keypoints {base_k}")

    analysis = analyze_inputs(rgb_dir, label_dir, npy_dir)
    if not analysis["ok"]:
        raise CrossSubjectError(str(analysis["error"]))

    # label 非空 + YOLO Pose 格式 + 外部 K 一致（不一致时 validate_labels 抛错）
    lab_map = {sid: p for sid, p in
               scan_files(label_dir, RGB_PREFIX, (".txt",))[0].items()}
    if not lab_map:
        raise CrossSubjectError("未找到任何外部 label（color_*.txt）")
    try:
        ext_k = validate_labels(lab_map)  # 数量/格式/K 一致性严格校验
    except RuntimeError as e:
        raise CrossSubjectError(f"外部 label 校验失败: {e}") from e
    if ext_k != base_k:
        raise CrossSubjectError(f"Expected {base_k} keypoints, got {ext_k}")

    result = {
        **{k: v for k, v in analysis.items() if k != "matched_ids"},
        "keypoints": ext_k,
        "base_keypoints": base_k,
    }
    if template_k is not None:
        result["template_keypoints"] = template_k

    # ---- Validate 即做全量 canonical reorder audit（reference bank）----
    # 扫描全部 external（residual/ambiguous/error 都不提前退出），最后统一 PASS/BLOCK。
    if template is not None:
        matched_ids = [sid for sid in analysis["matched_ids"]]
        label_files = [(f"{RGB_PREFIX}{sid}.txt", lab_map[sid]) for sid in matched_ids]
        result.update(_run_reorder_audit(b, label_files, base_k, audit_dir))

    return result


def _run_reorder_audit(base: Path, label_files: list[tuple[str, Path]],
                       expected_k: int,
                       audit_dir: str | Path | None) -> dict[str, object]:
    """用 base train labels 作为 canonical reference bank，对全部 external 全量 audit。

    - 逐张判定 OK/RESIDUAL_BLOCK/AMBIGUOUS/ERROR，记录 best/second
      reference、mapping、residual、gap
    - 全部扫完才统一决定：production_blocked>0 -> 抛 CrossSubjectError（BLOCK）
    - audit_dir 非 None 时写 canonical_reorder_audit.csv（不改正式 dataset）
    """
    refs = plr.load_reference_bank(base / "rgb" / "labels" / "train",
                                   expected_k=expected_k)
    audits = plr.scan_bank_audit(refs, label_files)
    stats = plr.audit_statistics(audits)
    blocked = int(stats["production_blocked"])
    total = int(stats["total"])

    audit_file = ""
    if audit_dir:
        ad = Path(audit_dir)
        ad.mkdir(parents=True, exist_ok=True)
        audit_file = str(ad / "canonical_reorder_audit.csv")
        plr.write_audit_report(audits, audit_file)

    if blocked > 0:
        first = next(a for a in audits if a.status != "OK")
        suffix = f" (see audit {audit_file})" if audit_file else ""
        raise CrossSubjectError(
            f"canonical reorder BLOCK: {blocked}/{total} "
            f"(first @ {first.file}: {first.status} {first.message}) "
            f"| max residual {stats.get('max_residual')}{suffix}")

    return {
        "reorder_ok": True,
        "reorder_total": total,
        "reorder_ok_count": int(stats["production_ok"]),
        "reorder_blocked": 0,
        "max_residual": stats.get("max_residual"),
        "reference_count": len(refs),
        "audit_total": total,
        "audit_min_residual": stats.get("min_residual"),
        "audit_median_residual": stats.get("median_residual"),
        "audit_p95_residual": stats.get("p95_residual"),
        "audit_le_015": stats.get("le_015"),
        "audit_le_016": stats.get("le_016"),
        "audit_le_018": stats.get("le_018"),
        "audit_best_second_same": stats.get("best_second_same"),
        "audit_best_second_different": stats.get("best_second_different"),
        "audit_most_common_mapping": stats.get("most_common_mapping"),
        "audit_most_common_count": stats.get("most_common_count"),
        "audit_file": audit_file,
    }


# ============================================================ 重建
def _copy_split_images(src_root: Path, dst_root: Path, split: str) -> None:
    """复制一个 split 的 images 全部文件（保留原名），供 RGB/RGBD train|val 复用。"""
    src = src_root / "images" / split
    dst = dst_root / "images" / split
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        raise CrossSubjectError(f"源目录缺失: {src}")
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def _copy_split_labels(src_root: Path, dst_root: Path, split: str) -> None:
    """复制一个 split 的 labels 全部文件（保留原名）。"""
    src = src_root / "labels" / split
    dst = dst_root / "labels" / split
    dst.mkdir(parents=True, exist_ok=True)
    if not src.is_dir():
        raise CrossSubjectError(f"源目录缺失: {src}")
    for p in src.iterdir():
        if p.is_file():
            shutil.copy2(p, dst / p.name)


def _stems(directory: Path) -> set[str]:
    if not directory.is_dir():
        return set()
    return {p.stem for p in directory.iterdir() if p.is_file()}


def _check_consistency(target: Path, base: Path) -> None:
    """发布前最终一致性：train/val 与 base 一致；RGB/RGBD 各 split、test labels 完全同步。"""
    for split in ("train", "val"):
        for side in ("rgb", "rgbd"):
            if (_stems(target / side / "images" / split) != _stems(base / side / "images" / split)
                    or _stems(target / side / "labels" / split) != _stems(base / side / "labels" / split)):
                raise CrossSubjectError(
                    f"一致性失败：{side}/{split} 与源数据集文件集合不一致，拒绝发布")
    for split in ("train", "val", "test"):
        rgb_imgs = _stems(target / "rgb" / "images" / split)
        rgbd_imgs = _stems(target / "rgbd" / "images" / split)
        if rgb_imgs != rgbd_imgs:
            raise CrossSubjectError(
                f"一致性失败：RGB 与 RGBD 的 {split} 图像 stem 不一致，拒绝发布")
    for split in ("train", "val", "test"):
        if (_stems(target / "rgb" / "labels" / split)
                != _stems(target / "rgbd" / "labels" / split)):
            raise CrossSubjectError(
                f"一致性失败：RGB 与 RGBD 的 {split} labels 文件集合不一致，拒绝发布")
    # labels 与 images 配对（YOLO 需要同 stem）
    for split in ("train", "val", "test"):
        for side in ("rgb", "rgbd"):
            if (_stems(target / side / "images" / split)
                    != _stems(target / side / "labels" / split)):
                raise CrossSubjectError(
                    f"一致性失败：{side}/{split} 的 images 与 labels stem 不一致，拒绝发布")


def rebuild(base: str | Path, name: str, rgb_dir: str | Path,
            label_dir: str | Path, npy_dir: str | Path,
            template: str | Path | None = None,
            depth_low: float = DEFAULT_DEPTH_LOW,
            depth_high: float = DEFAULT_DEPTH_HIGH) -> dict[str, object]:
    """重建 cross-subject 数据集（validate -> staging -> publish）。

    目标 = <base 上级>/<name>（如 .../dataset/hand_cross_subject_v1）。
    template: canonical label 路径。提供时在 publish 前对 external test labels 做
    一次 canonical 几何重排（同一结果写入 rgb/rgbd 两份），并生成只读
    canonical_reorder_report.csv；任何 reorder BLOCK 都中止且不发布。
    返回摘要 dict；任何失败抛 CrossSubjectError，且不留下半成品正式目录。
    """
    b = Path(base).resolve()
    name = (name or "").strip()
    if not name:
        raise CrossSubjectError("Output Dataset Name 不能为空")
    if "/" in name or "\\" in name or name in (".", ".."):
        raise CrossSubjectError(f"Output Dataset Name 非法: {name}")
    if name == b.name:
        raise CrossSubjectError("不能覆盖当前数据集（Output Dataset Name 与当前类别同名）")

    summary = validate(b, rgb_dir, label_dir, npy_dir, template=template)
    k = int(summary["keypoints"])
    template_k = summary.get("template_keypoints")

    dataset_parent = b.parent
    target = dataset_parent / name
    if target.exists():
        raise CrossSubjectError(f"Output dataset already exists: {target}")

    # ---- 输入地图（analyze 已保证无重复/无缺失，这里重扫取文件路径）----
    imgs, _ = scan_files(rgb_dir, RGB_PREFIX, IMG_EXTS)
    labs, _ = scan_files(label_dir, RGB_PREFIX, (".txt",))
    npys, _ = scan_files(npy_dir, DEPTH_PREFIX, (".npy",))
    matched_ids = sorted(set(imgs) & set(labs) & set(npys))
    if not matched_ids:
        raise CrossSubjectError("没有可用的 external test 样本")

    # canonical reference bank = base 全部 train labels（validate 已 dry-run 通过）
    refs: list[plr.Reference] | None = None
    if template is not None:
        try:
            refs = plr.load_reference_bank(b / "rgb" / "labels" / "train", expected_k=k)
        except plr.ReorderError as e:
            raise CrossSubjectError(f"canonical reference bank 读取失败: {e}") from e

    # ---- staging：先完整生成并验证 ----
    staging = dataset_parent / STAGING_NAME / name
    if staging.exists():
        shutil.rmtree(staging)
    reorder_rows: list[plr.ReorderRow] = []
    try:
        for side in ("rgb", "rgbd"):
            for sub in ("images", "labels"):
                for split in ("train", "val", "test"):
                    (staging / side / sub / split).mkdir(parents=True, exist_ok=True)
        for side in ("rgb", "rgbd"):
            _copy_split_images(b / side, staging / side, "train")
            _copy_split_images(b / side, staging / side, "val")
            _copy_split_labels(b / side, staging / side, "train")
            _copy_split_labels(b / side, staging / side, "val")

        # 2. external 全部进入 test（不随机拆分）
        #    有 template 时，RGB labels/test 先放“原始副本”，rgbd 那份由 reorder 统一写；
        #    无 template（旧行为）时两边都直接复制原始 label。
        for sid in matched_ids:
            src_img = imgs[sid]
            src_lab = labs[sid]
            shutil.copy2(src_img, staging / "rgb" / "images" / "test" / src_img.name)
            shutil.copy2(src_lab, staging / "rgb" / "labels" / "test" / src_lab.name)
            if template is None:
                shutil.copy2(src_lab, staging / "rgbd" / "labels" / "test" / src_lab.name)
            # RGBD test 4ch 图 = RGB + depth npy 融合（复用 prepare 的 fuse 逻辑）
            out_png = staging / "rgbd" / "images" / "test" / f"{RGB_PREFIX}{sid}.png"
            fuse_pair_rawdepth(src_img, npys[sid], out_png, depth_low, depth_high)

        # 2.5 canonical reference-bank 几何重排（publish 前，只排一次；两侧逐字一致）
        if template is not None and refs is not None:
            reorder_rows = _reorder_test_labels(staging, matched_ids, refs)
            _check_test_label_sync(staging)

        # 3. YAML（path 写正式目录，先按发布后位置生成）
        build_yaml(staging / "rgb" / "data_rgb.yaml", b.name, k, 3, False,
                   dataset_path=target / "rgb")
        build_yaml(staging / "rgbd" / "data_rgbd.yaml", b.name, k, 4, True,
                   dataset_path=target / "rgbd")

        # 4. 一致性校验（发布前）
        _check_consistency(staging, b)

        # 5. 事务发布：target 已确认不存在 -> move
        shutil.move(str(staging), str(target))
    except Exception:
        # staging 整根清理（含残留空壳），绝不留下半成品 staging
        staging_root = dataset_parent / STAGING_NAME
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    # 发布成功：清理 staging 根（内容已全部 move 到正式目录）
    staging_root = dataset_parent / STAGING_NAME
    if staging_root.exists():
        shutil.rmtree(staging_root)

    # 6. 最终路径验证（path 指向正式目录 + train/val/test 存在 + 无 .prepare_* 残留）
    verify_published_yaml(target / "rgb" / "data_rgb.yaml")
    verify_published_yaml(target / "rgbd" / "data_rgbd.yaml")
    _check_consistency(target, b)
    if template is not None:
        _check_test_label_sync(target)
        # 只读追溯报告：记录每张 test label 命中的 reference 与 permutation
        plr.write_report_with_reference(reorder_rows,
                                        target / "canonical_reorder_report.csv")

    result = {
        "images": int(summary["images"]),
        "labels": int(summary["labels"]),
        "depth": int(summary["depth"]),
        "matched": int(summary["matched"]),
        "keypoints": k,
        "base_keypoints": int(summary["base_keypoints"]),
        "base_train": len(_stems(b / "rgb" / "images" / "train")),
        "base_val": len(_stems(b / "rgb" / "images" / "val")),
        "test": int(summary["matched"]),
        "target_dir": str(target),
    }
    if template_k is not None:
        result["template_keypoints"] = int(template_k)
        result["reordered"] = len(reorder_rows)
        result["label_sync"] = True
        result["max_residual"] = summary.get("max_residual")
    return result


# ============================================================ canonical reference-bank 重排
def _reorder_test_labels(staging: Path, matched_ids: list[str],
                         refs: list[plr.Reference]) -> list[plr.ReorderRow]:
    """对 staging 内 rgb/labels/test 的 external labels 做一次 canonical bank 重排。

    - 忽略 external label 当前 K 编号，与 base 全部 train references 比较
      （shape signature + Hungarian + ICP + ambiguity guard），选 residual 最小者
    - 同一份重排结果写入 rgb 与 rgbd 的 labels/test（逐字一致）
    - 任一文件 residual 超限 / 歧义 / 无可用 reference -> CrossSubjectError（整次中止）
    返回每文件的 ReorderRow（供 canonical_reorder_report.csv，含 reference_file）。
    """
    rows: list[plr.ReorderRow] = []
    for sid in matched_ids:
        name = f"{RGB_PREFIX}{sid}.txt"
        rgb_path = staging / "rgb" / "labels" / "test" / name
        rgbd_path = staging / "rgbd" / "labels" / "test" / name
        try:
            head, triples, pts = plr.read_label(rgb_path)
            m = plr.bank_best_match(refs, pts)
            text = plr.build_reordered_text(head, triples, m.assignment) + "\n"
            rgb_path.write_text(text, encoding="utf-8")
            rgbd_path.write_text(text, encoding="utf-8")
            rows.append(plr.ReorderRow(name, "OK", plr.mapping_text(m.assignment),
                                       f"{m.normalized_rms:.6f}", "",
                                       reference=m.reference.name))
        except plr.ReorderAmbiguousError as e:
            raise CrossSubjectError(
                f"canonical reorder AMBIGUOUS @ {name}: {e}（不猜 ID）") from e
        except plr.ReorderError as e:
            raise CrossSubjectError(f"canonical reorder BLOCK @ {name}: {e}") from e
    return rows


def _check_test_label_sync(ds_root: Path) -> None:
    """逐字校验 rgb 与 rgbd 的 labels/test 内容完全一致（同一份重排结果）。"""
    rgb_dir = ds_root / "rgb" / "labels" / "test"
    rgbd_dir = ds_root / "rgbd" / "labels" / "test"
    rgb_names = {p.name for p in rgb_dir.iterdir() if p.is_file()} if rgb_dir.is_dir() else set()
    rgbd_names = {p.name for p in rgbd_dir.iterdir() if p.is_file()} if rgbd_dir.is_dir() else set()
    if rgb_names != rgbd_names:
        raise CrossSubjectError(
            f"RGB/RGBD labels/test 文件集合不一致（多出: "
            f"{sorted(rgb_names ^ rgbd_names)[:5]}），拒绝发布")
    for name in sorted(rgb_names):
        a = (rgb_dir / name).read_bytes()
        b = (rgbd_dir / name).read_bytes()
        if a != b:
            raise CrossSubjectError(
                f"RGB/RGBD labels/test 内容不一致: {name}，拒绝发布")


# ============================================================ CLI
def _print_json(payload: dict[str, object]) -> None:
    print(f"{_JSON_TAG} {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cross-subject test set 重建")
    ap.add_argument("mode", choices=("validate", "rebuild"),
                    help="validate: 只读校验；rebuild: 校验+staging+发布")
    ap.add_argument("--base", required=True, help="当前数据集目录（含 rgb/ rgbd/）")
    ap.add_argument("--name", required=True, help="Output Dataset Name（如 hand_cross_subject_v1）")
    ap.add_argument("--rgb-dir", required=True, help="外部 RGB 目录（color_<ts>.*）")
    ap.add_argument("--label-dir", required=True, help="外部 Labels 目录（color_<ts>.txt）")
    ap.add_argument("--npy-dir", required=True, help="外部 Raw Depth NPY 目录（depth_<ts>.npy）")
    ap.add_argument("--template", required=True,
                    help="Canonical Template Label 路径（定义 K0..K(N-1) 语义顺序）")
    ap.add_argument("--audit-dir", default="",
                    help="（可选）validate 全量 audit CSV 输出目录（默认不写文件）")
    args = ap.parse_args(argv)

    try:
        if args.mode == "validate":
            r = validate(args.base, args.rgb_dir, args.label_dir, args.npy_dir,
                         template=args.template,
                         audit_dir=(args.audit_dir or None))
            r = {k: v for k, v in r.items() if k != "matched_ids"}
            r["mode"] = "validate"
            r["base_train"] = len(_stems(Path(args.base) / "rgb" / "images" / "train"))
            r["base_val"] = len(_stems(Path(args.base) / "rgb" / "images" / "val"))
            r["target_dir"] = str(Path(args.base).parent / args.name)
            _print_json(r)
            return 0
        r = rebuild(args.base, args.name, args.rgb_dir, args.label_dir, args.npy_dir,
                    template=args.template)
        r["mode"] = "rebuild"
        # 成功日志（供 GUI 共享日志直接展示）
        print(f"Canonical keypoint reorder: OK")
        print(f"Template keypoints: {r.get('template_keypoints')}")
        print(f"Reordered test labels: {r.get('reordered')}")
        print("RGB/RGBD label sync: OK")
        _print_json(r)
        return 0
    except CrossSubjectError as e:
        print(f"{_BLOCK_TAG} {e}", file=sys.stderr)
        return 1
    except Exception as e:  # noqa: BLE001
        print(f"{_BLOCK_TAG} {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
