"""RGBD Depth 消融评估 (验证已训练的 RGBD Pose 模型是否真实利用第 4 通道 Depth)

仅用于已训练好的 RGBD 4ch 模型：在同一批 RGBD test 图片上，构造三种 Depth 输入，
比较检测/关键点指标，判断模型是否真的在用 Depth。

四种变体（共用同一批 12 个 test ID 与同一份 labels，同一权重，同一评估参数）：
  1. true_depth           原始 4 通道 RGBD 不变
  2. zero_depth           RGB 三通道不变，第 4 通道(Depth)全部置 0
  3. shuffled_depth       RGB 三通道不变，第 4 通道(Depth)取自其它 test 图（跨图随机，
                          derangement 不允许回配自己），可复现 —— Cross-image Shuffle
  4. spatial_shuffle_depth RGB 三通道不变，第 4 通道用「本图自身 Depth」，仅对像素位置做
                          固定 seed 的随机 permutation（flatten→permute→reshape），破坏与 RGB
                          的空间对应关系，但保留 min/max/mean/std/histogram —— Spatial-shuffled

输出：
  - 每个变体 Box / Pose 的 P / R / mAP50 / mAP50-95
  - 对比表 + True-Zero / True-CrossShuffle / True-SpatialShuffle 差值

生成期逐图校验（打印 + 断言）：
  - shape 必须为 (H, W, 4)
  - dtype 必须为 uint8
  - 第 4 通道 min / max / mean / std
  - zero_depth 第 4 通道必须全部为 0
  - shuffled_depth 第 4 通道必须与原样本 Depth 不同、且来自其它图
  - spatial_shuffle_depth 第 4 通道必须与原样本 Depth 不同，但排序后像素集合必须完全一致

约束：
  - 不重新训练；不覆盖原始 dataset；在临时目录生成消融数据
  - 不改任何 train 脚本
  - 原始 RGB / RGBD test 结果不受影响（只读取，不写入）

调用示例：
  python YJJ_Pose_Scripts/04_eval/ablate_rgbd_depth.py
  python YJJ_Pose_Scripts/04_eval/ablate_rgbd_depth.py --no-eval   # 仅生成临时数据 + 校验，不跑 val
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# ---- 默认路径（本机 Resource 数据集；如换机用 --rgbd-yaml 指定）----
_DEFAULT_RGBD_YAML = (
    r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand/rgbd/data_rgbd.yaml"
)
_DEFAULT_WEIGHTS = r"runs/pose/train_pose_rgbd_4ch/weights/best.pt"
_DEFAULT_OUT = r"runs/ablation/rgbd_depth"  # runs/ 已被 .gitignore 忽略，作为安全临时目录

VARIANTS = ("true_depth", "zero_depth", "shuffled_depth", "spatial_shuffle_depth")

# 变体显示名（最终表格列头用）：shuffled_depth = 跨图随机 Depth；
# spatial_shuffle_depth = 同图 Depth 像素位置随机打乱（保留统计量，仅破坏空间结构）。
VARIANT_LABELS = {
    "true_depth": "True Depth",
    "zero_depth": "Zero Depth",
    "shuffled_depth": "Cross-image Shuffled",
    "spatial_shuffle_depth": "Spatial-shuffled",
}


# --------------------------------------------------------------------------- 参数
def parse_args():
    p = argparse.ArgumentParser(description="RGBD Depth 消融评估 (true / zero / cross-image shuffled / spatial-shuffled depth)")
    p.add_argument("--weights", default=_DEFAULT_WEIGHTS, help="RGBD 4ch 评估权重 .pt")
    p.add_argument("--rgbd-yaml", default=_DEFAULT_RGBD_YAML,
                   help="原始 RGBD data yaml（提供 path / nc / kpt_shape / channels / rgbd）")
    p.add_argument("--out", default=_DEFAULT_OUT, help="临时消融数据输出目录")
    p.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸（三组必须一致）")
    p.add_argument("--batch", type=int, default=4, help="batch size（三组必须一致）")
    p.add_argument("--device", default=None, help="设备，默认自动选择 (cuda 优先)")
    p.add_argument("--seed", type=int, default=42, help="shuffled_depth 的 derangement 随机种子")
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS),
                   help="指定要执行的 Depth 变体子集（默认全部 4 个）；GUI 调用时传 true_depth zero_depth")
    p.add_argument("--split", default="test", help="评估切分（消融固定 test）")
    p.add_argument("--no-eval", action="store_true",
                   help="只生成临时数据 + 逐图校验，不调用 model.val（快速自检用）")
    return p.parse_args()


# --------------------------------------------------------------------------- 工具
def read_rgba(path: Path) -> np.ndarray:
    """读取 4 通道融合 PNG（BGRA），返回 (H, W, 4) uint8。"""
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(f"图像读取失败: {path}")
    if im.ndim != 3 or im.shape[2] != 4:
        raise ValueError(f"图像不是 4 通道: {path} shape={im.shape}")
    if im.dtype != np.uint8:
        raise ValueError(f"图像 dtype 非 uint8: {path} dtype={im.dtype}")
    return im


def ch4_stats(ch4: np.ndarray) -> dict:
    return {
        "min": float(ch4.min()),
        "max": float(ch4.max()),
        "mean": round(float(ch4.mean()), 4),
        "std": round(float(ch4.std()), 4),
    }


def derangement(n: int, seed: int) -> np.ndarray:
    """返回 [0, n) 的一个排列，且无固定点 (perm[i] != i)。固定 seed 可复现。"""
    rng = np.random.default_rng(seed)
    # 真实随机排列，循环直到得到 derangement（n=12 时几乎一次命中，且完全可复现）
    while True:
        perm = rng.permutation(n)
        if not np.any(perm == np.arange(n)):
            break
    assert set(perm.tolist()) == set(range(n)), "derangement 必须是合法排列"
    assert not np.any(perm == np.arange(n)), "derangement 不能有固定点"
    return perm


def spatial_shuffle_depth_channel(ch4: np.ndarray, seed: int) -> np.ndarray:
    """同图 Depth 通道：仅对像素位置做固定 seed 的随机 permutation，保留全部像素值。

    flatten -> rng.permutation -> reshape 回 (H, W)。因此 min/max/mean/std/histogram
    与原 Depth 完全相同，唯一被破坏的是像素的空间排列（与 RGB 的对应关系）。
    固定 seed 可复现；seed = 全局 seed + 图片下标。
    """
    rng = np.random.default_rng(seed)
    flat = ch4.ravel()
    perm = rng.permutation(flat.size)  # flat.size 个像素位置的随机排列
    return flat[perm].reshape(ch4.shape).astype(ch4.dtype)


def build_variant_yaml(src_yaml: Path, variant_root: Path) -> Path:
    """复制源 yaml，仅把 path 改写为变体目录（其余 nc/kpt_shape/channels/rgbd 不变）。"""
    text = src_yaml.read_text(encoding="utf-8")
    new_path = str(variant_root).replace("\\", "/")
    test_split = "images/test"  # 消融仅生成 test 拆分
    # 替换 path: 的值（保留行内注释），并把 train/val/test 统一指向已生成的 test 拆分，
    # 否则 check_det_dataset 会因缺 train/val 目录而报 FileNotFoundError。
    lines = []
    path_replaced = False
    for line in text.splitlines():
        m = re.match(r"^(\s*path\s*:\s*)(.+?)(\s*#.*)?$", line)
        if m and not path_replaced:
            lines.append(f"{m.group(1)}{new_path}{m.group(3) or ''}")
            path_replaced = True
            continue
        m2 = re.match(r"^(\s*(?:train|val|test)\s*:\s*)(.+?)(\s*#.*)?$", line)
        if m2:
            lines.append(f"{m2.group(1)}{test_split}{m2.group(3) or ''}")
            continue
        lines.append(line)
    if not path_replaced:
        raise ValueError("源 yaml 未找到 path: 字段，无法改写")
    out = variant_root / "data_rgbd_ablation.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- 生成 + 校验
def generate_ablation_data(src_yaml: Path, out_root: Path, seed: int, selected):
    """读取源 RGBD test 集，生成指定变体子集的临时数据，返回 (variants_info, checks)。"""
    import yaml  # 延迟导入，--no-eval 之外的路径也安全

    cfg = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    ds_root = Path(cfg["path"])
    test_img_dir = ds_root / cfg.get("test", "images/test")
    test_lbl_dir = ds_root / "labels" / "test"

    img_paths = sorted(test_img_dir.glob("*"))
    img_paths = [p for p in img_paths if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
    if not img_paths:
        raise FileNotFoundError(f"未找到 test 图片: {test_img_dir}")
    n = len(img_paths)
    print(f"[生成] test 图片数 = {n}，来源: {test_img_dir}")

    # 读取所有 4 通道原图 + 对应 labels
    orig = [read_rgba(p) for p in img_paths]
    depths = [im[:, :, 3].copy() for im in orig]  # 第 4 通道 = Depth
    label_paths = []
    for p in img_paths:
        lp = test_lbl_dir / (p.stem + ".txt")
        if not lp.exists():
            raise FileNotFoundError(f"缺少 label: {lp}")
        label_paths.append(lp)

    # shuffled_depth 的 derangement（固定 seed）
    perm = derangement(n, seed)

    checks = []  # 逐图校验记录
    variant_roots = {}
    for vname in selected:
        vroot = out_root / vname / "rgbd"
        img_out = vroot / "images" / "test"
        lbl_out = vroot / "labels" / "test"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        variant_roots[vname] = vroot

        if vname == "true_depth":
            src_depths = depths
        elif vname == "zero_depth":
            src_depths = [np.zeros_like(d) for d in depths]
        elif vname == "spatial_shuffle_depth":
            # 同图自身 Depth，仅随机打乱像素位置（seed = 全局 seed + 图片下标，可复现）
            src_depths = [spatial_shuffle_depth_channel(depths[i], seed + i) for i in range(n)]
        else:  # shuffled_depth：跨图随机 Depth（derangement，不取自己）
            src_depths = [depths[perm[i]] for i in range(n)]

        for i, p in enumerate(img_paths):
            im = orig[i].copy()
            new_ch4 = src_depths[i]
            # 形状对齐（极少数尺寸不一致时 resize 到原图尺寸，保持公平）
            if new_ch4.shape != im[:, :, 3].shape:
                new_ch4 = cv2.resize(new_ch4, (im.shape[1], im.shape[0]),
                                     interpolation=cv2.INTER_NEAREST)
            im[:, :, 3] = new_ch4.astype(np.uint8)
            dst = img_out / p.name
            ok = cv2.imwrite(str(dst), im, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            if not ok:
                raise RuntimeError(f"写图失败: {dst}")
            shutil.copy2(label_paths[i], lbl_out / (p.stem + ".txt"))

            # 逐图校验
            ch4 = im[:, :, 3]
            rec = {
                "variant": vname,
                "image": p.name,
                "shape": list(im.shape),
                "dtype": str(im.dtype),
                "ch4": ch4_stats(ch4),
            }
            if vname == "zero_depth":
                assert ch4.max() == 0 and ch4.min() == 0, f"zero_depth 第4通道非全0: {p.name}"
            if vname == "shuffled_depth":
                assert not np.array_equal(ch4, depths[i]), \
                    f"shuffled_depth 第4通道与原样本相同(回配自己): {p.name}"
                # 必须确实来自某张其它图
                assert any(np.array_equal(ch4, d) for d in depths), \
                    f"shuffled_depth 第4通道不匹配任何样本: {p.name}"
            if vname == "spatial_shuffle_depth":
                # 必须与原样本 Depth 不同（确实被打乱）
                assert not np.array_equal(ch4, depths[i]), \
                    f"spatial_shuffle_depth 第4通道与原样本相同(未被打乱): {p.name}"
                # 排序后像素集合必须与原 Depth 完全一致 -> 统计量必然一致
                assert np.array_equal(np.sort(ch4.ravel()), np.sort(depths[i].ravel())), \
                    f"spatial_shuffle_depth 像素集合与原 Depth 不一致(非纯 permutation): {p.name}"
                # 双重校验：逐图统计相等（permutation 必然成立）
                assert ch4_stats(ch4) == ch4_stats(depths[i]), \
                    f"spatial_shuffle_depth 第4通道统计与原 Depth 不一致: {p.name}"
            checks.append(rec)

        build_variant_yaml(src_yaml, vroot)
        print(f"[生成] 变体完成: {vname} -> {vroot}")

    # 控制台汇总校验
    print("\n=== 逐图生成校验（第4通道 Depth 统计）===")
    for i, p in enumerate(img_paths):
        line = f"  {p.name:40s}"
        for vname in selected:
            rec = next(c for c in checks if c["variant"] == vname and c["image"] == p.name)
            s = rec["ch4"]
            line += f" | {vname[:4]}:{s['min']:5.0f}/{s['max']:5.0f}/{s['mean']:6.1f}/{s['std']:5.1f}"
        print(line)
    print("\n[校验] 全部 shape=(H,W,4) / dtype=uint8 / zero 全0 / shuffled 异于原样本 / spatial 排序一致：通过")

    (out_root / "_ablation_checks.json").write_text(
        json.dumps({"n": n, "seed": seed, "checks": checks}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    return variant_roots, n


# --------------------------------------------------------------------------- 评估
def evaluate_variant(yaml_path: Path, weights: str, imgsz: int, batch: int, device, split="test"):
    # 4 通道 RGBD 加载逻辑只存在于仓库根目录的 vendored ultralytics 副本中，
    # 必须把仓库根插入 sys.path 最前，避免误用 conda site-packages 里的原生 ultralytics。
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    import ultralytics  # noqa: F401  (触发路径生效后再 import YOLO)
    from ultralytics import YOLO

    print(f"[ultralytics] {ultralytics.__file__}")
    model = YOLO(weights, task="pose")
    val_kwargs = dict(
        data=str(yaml_path),
        split=split,
        imgsz=imgsz,
        batch=batch,
        verbose=False,
        plots=False,
        save_json=False,
    )
    if device is not None:
        val_kwargs["device"] = device
    metrics = model.val(**val_kwargs)
    d = metrics.results_dict
    return {
        "box_p": float(d["metrics/precision(B)"]),
        "box_r": float(d["metrics/recall(B)"]),
        "box_map50": float(d["metrics/mAP50(B)"]),
        "box_map": float(d["metrics/mAP50-95(B)"]),
        "pose_p": float(d["metrics/precision(P)"]),
        "pose_r": float(d["metrics/recall(P)"]),
        "pose_map50": float(d["metrics/mAP50(P)"]),
        "pose_map": float(d["metrics/mAP50-95(P)"]),
    }


def _ablation_conclusion(results: dict, selected: list) -> str:
    """保守自动结论：仅当 true/zero 同时可用时比较 Pose mAP50-95。

    明确不推断“全部性能提升都来自真实三维几何信息”，因为 Zero Depth 已明显改变
    输入分布（第4通道全0），属分布外消融。
    """
    if "true_depth" not in results or "zero_depth" not in results:
        return "本次未同时执行 True/Zero 变体，跳过自动结论。"
    td = results["true_depth"]
    zd = results["zero_depth"]
    if td["pose_map"] > zd["pose_map"]:
        return (f"Zero Depth 后 Pose mAP50-95 下降（True={td['pose_map']:.4f} > "
                f"Zero={zd['pose_map']:.4f}），表明当前 RGBD 模型实际利用了第4通道提供的 "
                f"Depth 信息，而非完全忽略 Depth 通道。")
    if td["pose_map"] < zd["pose_map"]:
        return (f"Zero Depth 后 Pose mAP50-95 不降反升（True={td['pose_map']:.4f} < "
                f"Zero={zd['pose_map']:.4f}），属非预期，建议检查消融数据生成与评估一致性。")
    return (f"True Depth 与 Zero Depth 的 Pose mAP50-95 基本持平 "
            f"（True={td['pose_map']:.4f} ≈ Zero={zd['pose_map']:.4f}），"
            f"无法据此判断 Depth 是否被利用。")


def print_table(results: dict, out_root: Path, selected: list):
    """按 selected 变体（保持 VARIANTS 顺序）输出逐指标表 + True-Zero 差值 + 简洁对照表
    + 保守自动结论，并写出 ablation_result.json 与一行 DEPTH_ABLATION_JSON（供 GUI 捕获）。"""
    keys = ("box_p", "box_r", "box_map50", "box_map", "pose_p", "pose_r", "pose_map50", "pose_map")
    metric_labels = {
        "box_p": "Box P", "box_r": "Box R", "box_map50": "Box mAP50",
        "box_map": "Box mAP50-95", "pose_p": "Pose P", "pose_r": "Pose R",
        "pose_map50": "Pose mAP50", "pose_map": "Pose mAP50-95",
    }
    col_w = 20
    hdr = f"{'Metric':14s}"
    for v in selected:
        hdr += f" | {VARIANT_LABELS[v]:>{col_w}s}"
    sep = "-" * (14 + 3 + col_w * len(selected))
    print("\n" + "=" * len(sep))
    print(hdr)
    print(sep)
    for k in keys:
        line = f"{metric_labels[k]:14s}"
        for v in selected:
            line += f" | {results[v][k]:{col_w}.4f}"
        print(line)
    print(sep)

    td = results.get("true_depth")
    zd = results.get("zero_depth")
    if td is not None and zd is not None:
        print("\n=== True Depth vs Zero Depth ===")
        print(f"Box  mAP50-95 : True={td['box_map']:.4f}  Zero={zd['box_map']:.4f}  "
              f"True-Zero={td['box_map'] - zd['box_map']:+.4f}")
        print(f"Pose mAP50-95 : True={td['pose_map']:.4f}  Zero={zd['pose_map']:.4f}  "
              f"True-Zero={td['pose_map'] - zd['pose_map']:+.4f}")

    # 简洁对照表
    print("\n=== 简洁对照表 ===")
    print(f"{'Variant':20s} {'Box mAP50-95':>14s} {'Pose mAP50-95':>16s}")
    print("-" * 52)
    for v in selected:
        print(f"{VARIANT_LABELS[v]:20s} {results[v]['box_map']:14.4f} {results[v]['pose_map']:16.4f}")
    if td is not None and zd is not None:
        print(f"{'True-Zero':20s} {td['box_map'] - zd['box_map']:14.4f} "
              f"{td['pose_map'] - zd['pose_map']:16.4f}")

    conclusion = _ablation_conclusion(results, selected)
    print("\n[结论] " + conclusion)
    print("=" * len(sep))

    (out_root / "ablation_result.json").write_text(
        json.dumps({"results": results, "selected": selected, "conclusion": conclusion},
                   indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[结果] 已写出 {out_root / 'ablation_result.json'}")
    # 机器可读摘要（供 GUI 实时捕获）
    print("DEPTH_ABLATION_JSON " + json.dumps(
        {"selected": selected, "results": results, "conclusion": conclusion},
        ensure_ascii=False))


# --------------------------------------------------------------------------- main
def main():
    args = parse_args()
    src_yaml = Path(args.rgbd_yaml)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # 选定变体（保持 VARIANTS 原有顺序，仅保留请求子集）
    selected = [v for v in VARIANTS if v in set(args.variants)]
    if not selected:
        print("[错误] --variants 为空或不含合法变体")
        sys.exit(2)

    print("=" * 64)
    print("RGBD Depth 消融评估")
    print(f"weights : {args.weights}")
    print(f"rgbd    : {src_yaml}")
    print(f"out     : {out_root}")
    print(f"imgsz  : {args.imgsz}   batch: {args.batch}   seed: {args.seed}")
    print(f"split   : {args.split}")
    print(f"variants: {selected}")
    print("=" * 64)

    variant_roots, n = generate_ablation_data(src_yaml, out_root, args.seed, selected)

    if args.no_eval:
        print("\n[no-eval] 已生成临时数据并完成校验，跳过 model.val。运行去掉 --no-eval 执行评估。")
        return

    results = {}
    for vname in selected:
        yaml_path = variant_roots[vname] / "data_rgbd_ablation.yaml"
        print(f"\n>>> 评估变体: {vname}  ({yaml_path})")
        results[vname] = evaluate_variant(
            yaml_path, args.weights, args.imgsz, args.batch, args.device, args.split)

    print_table(results, out_root, selected)


if __name__ == "__main__":
    main()
