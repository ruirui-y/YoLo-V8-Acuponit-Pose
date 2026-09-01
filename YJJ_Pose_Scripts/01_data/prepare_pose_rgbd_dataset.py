"""
一键准备 RGB / RGBD Pose 数据集：

  1. 融合：将 RGB 与 uint8 Depth 按时间戳配对，合成为 YOLO RGBD 的 4 通道 PNG
     （复用 scripts/fuse_rgb_depth.py 的 fuse_pair，第 4 通道 = RGB 三通道均值）
  2. 划分：按时间戳从早到晚排序，70/15/15 切分为 train/val/test（不随机）
  3. 生成 RGB 与 RGBD 共用的 splits.json（单一数据源，保证两边样本编号完全一致）
  4. 复制图像/标签到 dataset/<class>/{rgb,rgbd}/{images,labels}/{train,val,test}
  5. 生成 data_rgb.yaml 与 data_rgbd.yaml（nc / kpt_shape 由数据推断，不写死）

RGB 与 RGBD 严格共用同一份 splits.json 与同一份 labels，仅图像通道不同（3ch vs 4ch），
满足「相同 train/val/test 样本、只允许输入通道不同」的对照实验约束。

关键点维度 (kpt_shape) 自动从标签文件推断：每行的字段数 = 5(bbox) + 3*kpt。

用法:
python YJJ_Pose_Scripts/01_data/prepare_pose_rgbd_dataset.py \
    --rgb_dir    "H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_image" \
    --depth_dir  "H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_depth" \
    --label_dir  "H:/YJJ/Yolo_RGBD/Resource/session1_200146/labels" \
    --output_dir "H:/YJJ/Yolo_RGBD/Resource/session1_200146" \
    --class_name hand \
    --force            # 可选：允许成功后覆盖已有输出（默认拒绝覆盖非空结果）
"""
import argparse
import json
import math
import shutil
import sys
from pathlib import Path

# 复用原融合脚本的核心函数（RGB 读 + 深度读 + 4 通道构造 + uint8 归一化）
# 本文件位于 <repo>/YJJ_Pose_Scripts/01_data/，向上两级为项目根，scripts 在其下。
_SCRIPT_DIR = Path(__file__).resolve().parent          # .../YJJ_Pose_Scripts/01_data
_REPO_ROOT = _SCRIPT_DIR.parents[1]                   # .../yolov8_rgbd_detection
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from fuse_rgb_depth import fuse_pair  # noqa: E402

RGB_PREFIX = "color_"
DEPTH_PREFIX = "depth_show"


def ts_of(name: str, prefix: str):
    """从文件名提取时间戳部分：去掉 prefix 与扩展名。"""
    stem = Path(name).stem
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return None


def collect(dir_path: Path, prefix: str):
    return {ts_of(p.name, prefix): p for p in dir_path.iterdir()
            if p.is_file() and ts_of(p.name, prefix) is not None}


def validate_labels(label_map: dict):
    """严格校验所有匹配到的 YOLO-Pose 标签，自动推断统一关键点数量 K。

    校验项（任意违规立即打印文件名/行号并 raise RuntimeError）：
      - 每行字段数必须 >= 8 且 (字段数 - 5) % 3 == 0，自动得 K = (字段数 - 5) / 3
      - 所有样本、所有行的 K 必须完全一致（不一致即报错并打印 期望/实际）
      - 所有字段都能转换成数字
      - class 必须是非负整数
      - bbox：x,y ∈ [0,1]，w,h ∈ (0,1]
      - 每个关键点：x,y ∈ [0,1]，visibility ∈ {0,1,2}
      - 空 txt 视为错误

    返回：统一的 K（int）。标签非法时不应生成任何融合图片。
    """
    if not label_map:
        raise RuntimeError("未匹配到任何标签文件，无法推断关键点数量，已中止数据准备")

    expected_k = None
    for ts in sorted(label_map):
        p = label_map[ts]
        raw = p.read_text(encoding="utf-8")
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        if not lines:
            print(f"[错误] 标签文件为空: {p.name}")
            raise RuntimeError(f"标签文件 {p.name} 为空（无任何标注行），已中止数据准备")

        for idx, ln in enumerate(lines, start=1):
            cols = ln.split()
            ncols = len(cols)

            # 1. YOLO-Pose 格式：字段数 = 5(bbox) + 3*K，K >= 1
            if ncols < 8 or (ncols - 5) % 3 != 0:
                print(f"[错误] 标签格式非法: {p.name} 第 {idx} 行")
                print(f"  实际字段数={ncols} (要求 >= 8 且 (字段数-5) 能被 3 整除)")
                raise RuntimeError(
                    f"标签 {p.name} 第 {idx} 行字段数={ncols} 不符合 YOLO-Pose 格式"
                    f"（需 5(bbox)+3*K 且 K>=1），已中止数据准备")

            k = (ncols - 5) // 3

            # 2. 关键点数量一致性
            if expected_k is None:
                expected_k = k
            elif k != expected_k:
                print(f"[错误] 关键点数量不一致: {p.name} 第 {idx} 行")
                print(f"  实际字段数={ncols}")
                print(f"  实际 K={k}")
                print(f"  期望 K={expected_k}")
                raise RuntimeError(
                    f"标签 {p.name} 第 {idx} 行 K={k} 与期望 K={expected_k} 不一致，已中止数据准备")

            # 3. 所有字段可转换为数字
            try:
                vals = [float(c) for c in cols]
            except ValueError as e:
                print(f"[错误] 标签含非数字字段: {p.name} 第 {idx} 行: {e}")
                raise RuntimeError(
                    f"标签 {p.name} 第 {idx} 行存在无法解析为数字的字段，已中止数据准备") from e

            # 3b. 所有字段必须为有限数（NaN / inf / -inf 非法，立即中止）
            for ci, v in enumerate(vals):
                if not math.isfinite(v):
                    print(f"[错误] 标签含非有限数: {p.name} 第 {idx} 行 第 {ci + 1} 个字段={cols[ci]}")
                    raise RuntimeError(
                        f"标签 {p.name} 第 {idx} 行第 {ci + 1} 个字段 {cols[ci]} 为 NaN/inf/-inf，"
                        f"已中止数据准备")

            # 4. class 必须严格等于 0（YAML 固定 nc=1，names 仅 0:class_name）
            cls_val = vals[0]
            if cls_val != int(cls_val) or int(cls_val) != 0:
                print(f"[错误] class 非法: {p.name} 第 {idx} 行 class={cols[0]}")
                raise RuntimeError(
                    f"标签 {p.name} 第 {idx} 行 class={cols[0]} 必须严格等于 0，已中止数据准备")

            # 5. bbox 归一化范围
            x, y, w, h = vals[1], vals[2], vals[3], vals[4]
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                print(f"[错误] bbox 中心越界: {p.name} 第 {idx} 行 x={x} y={y}")
                raise RuntimeError(
                    f"标签 {p.name} 第 {idx} 行 bbox 中心 (x,y)=({x},{y}) 必须 ∈ [0,1]，已中止数据准备")
            if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                print(f"[错误] bbox 尺寸非法: {p.name} 第 {idx} 行 w={w} h={h}")
                raise RuntimeError(
                    f"标签 {p.name} 第 {idx} 行 bbox 尺寸 (w,h)=({w},{h}) 必须 w>0,h>0 且 <=1，已中止数据准备")

            # 6. 每个关键点：x,y ∈ [0,1]，visibility 必须严格为 0/1/2 整数
            kpts = vals[5:]
            for j in range(0, len(kpts), 3):
                kx, ky, vis = kpts[j], kpts[j + 1], kpts[j + 2]
                if not (0.0 <= kx <= 1.0 and 0.0 <= ky <= 1.0):
                    print(f"[错误] 关键点越界: {p.name} 第 {idx} 行 第 {j // 3 + 1} 个点 x={kx} y={ky}")
                    raise RuntimeError(
                        f"标签 {p.name} 第 {idx} 行关键点 {j // 3 + 1} (x,y)=({kx},{ky}) 必须 ∈ [0,1]，"
                        f"已中止数据准备")
                if vis != int(vis) or int(vis) not in (0, 1, 2):
                    print(f"[错误] visibility 非法: {p.name} 第 {idx} 行 第 {j // 3 + 1} 个点 vis={vis}")
                    raise RuntimeError(
                        f"标签 {p.name} 第 {idx} 行关键点 {j // 3 + 1} visibility={vis} 必须严格为 0/1/2 整数，"
                        f"已中止数据准备")

    if expected_k is None:
        raise RuntimeError("未能推断关键点数量（无有效标签行），已中止数据准备")
    return expected_k


def build_yaml(yaml_path: Path, class_name: str, kpt: int, channels: int, rgbd: bool):
    """生成 YOLO-Pose data yaml（格式对齐项目现有 acupoint yaml）。"""
    tag = "RGBD" if rgbd else "RGB"
    lines = [
        f"# {tag} 手部关键点数据集配置 (YOLO-Pose)",
        f"# path: 数据集根目录（{tag} 实验，输入 {channels} 通道）",
        f"path: {yaml_path.parent}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "# 类别配置",
        "nc: 1",
        "names:",
        f"  0: {class_name}",
        "",
        f"# 输入通道配置（{tag}）",
        f"rgbd: {'true' if rgbd else 'false'}",
        f"channels: {channels}",
        "",
        "# Pose 关键点配置",
        f"kpt_shape: [{kpt}, 3]  # {kpt} 个关键点，每个点 (x, y, visibility)",
        "",
        "# flip_idx 暂未配置：关键点语义与左右翻转对应关系尚未确认",
    ]
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="一键准备 RGB/RGBD Pose 数据集")
    ap.add_argument("--rgb_dir", required=True, help="RGB 图像目录（color_<timestamp>.jpg）")
    ap.add_argument("--depth_dir", required=True, help="Depth 图像目录（depth_show<timestamp>.jpg）")
    ap.add_argument("--label_dir", required=True, help="标签目录（color_<timestamp>.txt）")
    ap.add_argument("--output_dir", required=True, help="输出根目录（内部生成 out/ 与 dataset/<class>/）")
    ap.add_argument("--class_name", default="hand", help="类别名，默认 hand")
    ap.add_argument("--force", action="store_true",
                    help="允许成功后覆盖已有输出；不传则在检测到非空输出时拒绝并停止")
    args = ap.parse_args()

    rgb_dir = Path(args.rgb_dir)
    depth_dir = Path(args.depth_dir)
    label_dir = Path(args.label_dir)
    out_root = Path(args.output_dir)
    cls = args.class_name.strip() or "hand"
    force = bool(args.force)

    # === 输出目录保护：默认不允许覆盖非空结果 ===
    final_out = out_root / "out"
    final_ds = out_root / "dataset" / cls
    if not force:
        blockers = []
        if final_out.exists() and any(final_out.iterdir()):
            blockers.append(str(final_out))
        if final_ds.exists() and any(final_ds.iterdir()):
            blockers.append(str(final_ds))
        if blockers:
            print("[错误] 已有输出目录且非空，拒绝覆盖：")
            for b in blockers:
                print("  ", b)
            print("提示：已有输出，如需重新生成请使用 --force")
            raise RuntimeError("检测到已有非空输出目录，已中止（未传 --force）")

    # === 临时 staging 目录：所有产物先写入此处，全部校验通过后才替换正式目录 ===
    staging_root = out_root / ".prepare_pose_rgbd_tmp"
    if staging_root.exists():
        shutil.rmtree(staging_root)  # 清理上次崩溃残留
    staging_root.mkdir(parents=True, exist_ok=True)

    try:
        # 1. 融合 4 通道 PNG（写入 staging）
        fused_dir = staging_root / "out"
        fused_dir.mkdir(parents=True, exist_ok=True)
        rgb_map = collect(rgb_dir, RGB_PREFIX)
        depth_map = collect(depth_dir, DEPTH_PREFIX)
        label_map = collect(label_dir, RGB_PREFIX)

        # 严格样本配对检查：RGB / Depth / Label 必须一一对应，任何缺失立即失败
        rgb_ids = set(rgb_map)
        depth_ids = set(depth_map)
        label_ids = set(label_map)
        missing_rgb = sorted((depth_ids | label_ids) - rgb_ids)
        missing_depth = sorted((rgb_ids | label_ids) - depth_ids)
        missing_label = sorted((rgb_ids | depth_ids) - label_ids)
        if missing_rgb or missing_depth or missing_label:
            print("[错误] RGB / Depth / Label 时间戳未一一对应：")
            if missing_rgb:
                print("  缺 RGB 的 ID:", ", ".join(missing_rgb))
            if missing_depth:
                print("  缺 Depth 的 ID:", ", ".join(missing_depth))
            if missing_label:
                print("  缺 Label 的 ID:", ", ".join(missing_label))
            raise RuntimeError(
                f"样本配对失败：RGB={len(rgb_ids)} Depth={len(depth_ids)} Label={len(label_ids)}，"
                f"存在缺失样本，已中止数据准备")

        # === 标签严格校验（必须在 RGBD 融合之前，非法则不生成任何融合图片）===
        kpt = validate_labels(label_map)
        print(f"[标签] 严格校验通过，统一关键点数量 K={kpt}")

        usable = sorted(set(rgb_map) & set(depth_map) & set(label_map))
        print(f"[统计] RGB={len(rgb_map)} Depth={len(depth_map)} Label={len(label_map)} 三集合一致={len(usable)}")
        for ts in usable:
            try:
                fuse_pair(rgb_map[ts], depth_map[ts], fused_dir / f"{RGB_PREFIX}{ts}.png", depth_type="uint8")
            except Exception as e:  # noqa: BLE001
                print(f"[错误] 融合失败，样本 ID={ts}: {e}")
                raise RuntimeError(f"样本 {ts} 融合失败，已中止数据准备") from e
        print(f"[融合] 已写出 4 通道 PNG: {len(usable)}/{len(usable)} -> {fused_dir}")

        # 融合产物强制校验：期望 RGBD ID 集合 == 实际生成的 PNG ID 集合
        expected_ids = set(usable)
        actual_ids = {p.stem for p in fused_dir.glob(f"{RGB_PREFIX}*.png")}
        missing_png = sorted(expected_ids - actual_ids)
        extra_png = sorted(actual_ids - expected_ids)
        if missing_png or extra_png:
            print("[错误] 融合产物与期望样本不一致：")
            if missing_png:
                print("  缺少的 ID:", ", ".join(missing_png))
            if extra_png:
                print("  多出的 ID:", ", ".join(extra_png))
            raise RuntimeError(
                f"融合产物校验失败：期望={len(expected_ids)} 实际={len(actual_ids)}，"
                f"存在缺失/多出样本，已中止数据准备")

        # 2. 划分 70/15/15（按时间戳升序，不随机）
        ds_root = staging_root / "dataset" / cls
        rgbd_ids = sorted(p.stem for p in fused_dir.glob(f"{RGB_PREFIX}*.png"))
        label_ids = sorted(p.stem for p in label_dir.glob(f"{RGB_PREFIX}*.txt"))
        usable = sorted(set(rgbd_ids) & set(label_ids))
        n = len(usable)
        train_n = round(n * 0.70)
        val_n = round(n * 0.15)
        test_n = n - train_n - val_n  # 余数归 test，保证总量守恒
        train = usable[:train_n]
        val = usable[train_n:train_n + val_n]
        test = usable[train_n + val_n:train_n + val_n + test_n]
        splits = {
            "class": cls,
            "kpt_shape": [kpt, 3],
            "n_total": n,
            "n": {"train": len(train), "val": len(val), "test": len(test)},
            "train": train,
            "val": val,
            "test": test,
        }
        ds_root.mkdir(parents=True, exist_ok=True)
        (ds_root / "splits.json").write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[划分] 总={n} train={len(train)} val={len(val)} test={len(test)} kpt={kpt}")

        # 3. 复制图像 + 标签（RGB 3ch / RGBD 4ch 各自 images，labels 同一份双写）
        for split, ids in (("train", train), ("val", val), ("test", test)):
            rgb_img_d = ds_root / "rgb" / "images" / split
            rgbd_img_d = ds_root / "rgbd" / "images" / split
            rgb_lab_d = ds_root / "rgb" / "labels" / split
            rgbd_lab_d = ds_root / "rgbd" / "labels" / split
            for d in (rgb_img_d, rgbd_img_d, rgb_lab_d, rgbd_lab_d):
                d.mkdir(parents=True, exist_ok=True)
            for sid in ids:
                # RGB 3ch 源（优先 jpg，回退 png）
                src_rgb = rgb_dir / f"{sid}.jpg"
                if not src_rgb.exists():
                    src_rgb = rgb_dir / f"{sid}.png"
                if src_rgb.exists():
                    shutil.copy2(src_rgb, rgb_img_d / f"{sid}{src_rgb.suffix}")
                # RGBD 4ch 源（融合产物）
                src_rgbd = fused_dir / f"{sid}.png"
                if src_rgbd.exists():
                    shutil.copy2(src_rgbd, rgbd_img_d / f"{sid}.png")
                # 同一份标签双写
                src_lab = label_dir / f"{sid}.txt"
                if src_lab.exists():
                    shutil.copy2(src_lab, rgb_lab_d / f"{sid}.txt")
                    shutil.copy2(src_lab, rgbd_lab_d / f"{sid}.txt")
        print(f"[复制] 完成 dataset/{cls}/")

        # 4. 生成 data yaml
        build_yaml(ds_root / "rgb" / "data_rgb.yaml", cls, kpt, 3, False)
        build_yaml(ds_root / "rgbd" / "data_rgbd.yaml", cls, kpt, 4, True)
        print(f"[yaml] 已生成 dataset/{cls}/rgb/data_rgb.yaml 与 rgbd/data_rgbd.yaml")

        # 5. 校验并写出 dataset_report.json（GUI 数据集状态读取源；splits.json 仅负责样本划分）
        def _count(d):
            return len(list(d.glob("*"))) if d.exists() else 0
        per = {"train": len(train), "val": len(val), "test": len(test)}
        rgb_img_ok = all(_count(ds_root / "rgb" / "images" / s) == per[s] for s in per)
        rgbd_img_ok = all(_count(ds_root / "rgbd" / "images" / s) == per[s] for s in per)
        lab_ok = all(_count(ds_root / "rgb" / "labels" / s) == per[s] for s in per)
        yaml_ok = (ds_root / "rgb" / "data_rgb.yaml").exists() and \
                  (ds_root / "rgbd" / "data_rgbd.yaml").exists()
        validation_passed = bool(rgb_img_ok and rgbd_img_ok and lab_ok and yaml_ok and kpt > 0)
        report = {
            "total_samples": n,
            "train_samples": len(train),
            "val_samples": len(val),
            "test_samples": len(test),
            "keypoint_count": kpt,
            "rgb_channels": 3,
            "rgbd_channels": 4,
            "validation_passed": validation_passed,
        }
        (ds_root / "dataset_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[report] 已写出 dataset_report.json (validation_passed={validation_passed})")

        if not validation_passed:
            raise RuntimeError("最终校验未通过 (validation_passed=false)，已中止，不写入正式目录")

        # === 全部校验通过：单事务替换 ===
        # A: final_out -> backup_out   B: final_ds -> backup_ds
        # C: staging/out -> final_out  D: staging/ds -> final_ds
        # 任一步失败都回滚到运行前状态；backup 非空时禁止自动删除（可能是旧数据唯一副本）
        backup_root = out_root / ".prepare_pose_rgbd_backup"
        if backup_root.exists() and any(backup_root.iterdir()):
            print("[错误] 检测到上次替换可能留下的 backup 且非空，禁止自动删除：")
            print("  ", str(backup_root))
            print("提示：请人工检查该 backup（可能是旧正式数据的唯一副本），确认无误后再重新运行")
            raise RuntimeError("检测到非空 .prepare_pose_rgbd_backup，已中止以避免误删旧数据")
        if backup_root.exists():
            backup_root.rmdir()  # 仅当为空才移除，绝不自动删非空 backup
        backup_root.mkdir(parents=True, exist_ok=True)
        backup_out = backup_root / "out"
        backup_ds = backup_root / ("dataset_" + cls)

        # 记录旧正式目录是否存在（仅这些需要回滚恢复）
        old_out_existed = final_out.exists()
        old_ds_existed = final_ds.exists()

        out_backed_up = False
        ds_backed_up = False
        out_installed = False
        ds_installed = False
        try:
            # A/B：先把旧正式结果 move 到 backup 暂存（移动保存，非删除）
            if old_out_existed:
                shutil.move(str(final_out), str(backup_out))
                out_backed_up = True
            if old_ds_existed:
                shutil.move(str(final_ds), str(backup_ds))
                ds_backed_up = True
            # C/D：再把 staging 中的新结果 move 到正式位置
            shutil.move(str(staging_root / "out"), str(final_out))
            out_installed = True
            shutil.move(str(staging_root / "dataset" / cls), str(final_ds))
            ds_installed = True
        except Exception as e:  # noqa: BLE001
            # 事务失败：仅删本次新装入的正式目录；再从 backup 恢复旧结果；
            # 只有恢复成功后才删除 backup（staging 残留始终可自动删除）
            if out_installed and final_out.exists():
                shutil.rmtree(final_out)
            if ds_installed and final_ds.exists():
                shutil.rmtree(final_ds)
            restored = True
            try:
                if out_backed_up and backup_out.exists():
                    shutil.move(str(backup_out), str(final_out))
                if ds_backed_up and backup_ds.exists():
                    shutil.move(str(backup_ds), str(final_ds))
            except Exception as re:  # noqa: BLE001
                restored = False
                print(f"[错误] 回滚恢复旧结果失败，请人工处理 backup: {re}")
            if restored and (out_backed_up or ds_backed_up):
                if backup_root.exists():
                    shutil.rmtree(backup_root)
            if staging_root.exists():
                shutil.rmtree(staging_root)
            raise RuntimeError("正式结果替换失败，已尝试回滚到原有输出") from e

        # 两个新目录均安装成功：清理 backup 与 staging 残留
        if backup_root.exists():
            shutil.rmtree(backup_root)
        if staging_root.exists():
            shutil.rmtree(staging_root)
        print(f"[READY] total={n} train={len(train)} val={len(val)} test={len(test)} "
              f"kpt={kpt} rgb_ch=3 rgbd_ch=4 (已替换为正式目录)")
    except Exception:
        # 失败时清理 staging，保证正式结果（若有）不被破坏，异常继续向上抛出
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise
    else:
        # 成功：清理 staging 残留（out 与 dataset/<cls> 已移走，目录应已空）
        if staging_root.exists():
            shutil.rmtree(staging_root)


if __name__ == "__main__":
    main()
