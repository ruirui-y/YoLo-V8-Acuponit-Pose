"""
按时间戳从早到晚排序，将手部 RGB / RGBD 数据集划分为 train / val / test，
并生成 splits.json 作为 RGB 与 RGBD 共享的单一数据源，
确保两者 train/val/test 样本编号完全一致，仅图像通道不同（3ch vs 4ch）。

用法（默认指向 session1_200146，可用参数覆盖）:
python YJJ_Pose_Scripts/01_data/split_dataset_by_timestamp.py
"""
import argparse
import json
import shutil
from pathlib import Path

BASE = Path(r"H:/YJJ/Yolo_RGBD/Resource/session1_200146")
PREFIX = "color_"  # RGB / 标签文件名前缀；RGBD 输出 PNG 也用同一 stem


def main():
    ap = argparse.ArgumentParser(description="按时间戳升序划分手部 RGB/RGBD 数据集")
    ap.add_argument("--rgb_img_dir", default=str(BASE / "out_image"))
    ap.add_argument("--rgbd_img_dir", default=str(BASE / "out"))
    ap.add_argument("--label_dir", default=str(BASE / "labels"))
    ap.add_argument("--out_root", default=str(BASE / "dataset" / "hand"))
    ap.add_argument("--train_n", type=int, default=57)
    ap.add_argument("--val_n", type=int, default=12)
    ap.add_argument("--test_n", type=int, default=12)
    args = ap.parse_args()

    rgb_ids = sorted(p.stem for p in Path(args.rgb_img_dir).glob(f"{PREFIX}*.jpg"))
    rgbd_ids = sorted(p.stem for p in Path(args.rgbd_img_dir).glob(f"{PREFIX}*.png"))
    label_ids = sorted(p.stem for p in Path(args.label_dir).glob(f"{PREFIX}*.txt"))

    usable = sorted(set(rgb_ids) & set(rgbd_ids) & set(label_ids))
    print(f"[统计] RGB={len(rgb_ids)} RGBD={len(rgbd_ids)} Label={len(label_ids)} 交集可用={len(usable)}")

    n = len(usable)
    assert args.train_n + args.val_n + args.test_n == n, \
        f"划分数量之和({args.train_n+args.val_n+args.test_n}) != 可用样本数({n})"

    # 按时间戳升序依次切片（不随机 shuffle）
    train = usable[0:args.train_n]
    val = usable[args.train_n:args.train_n + args.val_n]
    test = usable[args.train_n + args.val_n:args.train_n + args.val_n + args.test_n]

    splits = {
        "class": "hand",
        "kpt_shape": [7, 3],
        "n_total": n,
        "n": {"train": len(train), "val": len(val), "test": len(test)},
        "train": train,
        "val": val,
        "test": test,
    }
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "splits.json").write_text(json.dumps(splits, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[写入] {out_root / 'splits.json'}")

    # 按 splits 复制：RGB(3ch jpg) / RGBD(4ch png) 各自 images，labels 共用同一份
    rgb_img_dir = Path(args.rgb_img_dir)
    rgbd_img_dir = Path(args.rgbd_img_dir)
    label_dir = Path(args.label_dir)

    counter = {"rgb": 0, "rgbd": 0, "label": 0}
    for split, ids in (("train", train), ("val", val), ("test", test)):
        rgb_imgs = out_root / "rgb" / "images" / split
        rgbd_imgs = out_root / "rgbd" / "images" / split
        rgb_labels = out_root / "rgb" / "labels" / split
        rgbd_labels = out_root / "rgbd" / "labels" / split
        for d in (rgb_imgs, rgbd_imgs, rgb_labels, rgbd_labels):
            d.mkdir(parents=True, exist_ok=True)

        for sid in ids:
            # RGB 3ch
            shutil.copy2(rgb_img_dir / f"{sid}.jpg", rgb_imgs / f"{sid}.jpg")
            counter["rgb"] += 1
            # RGBD 4ch
            shutil.copy2(rgbd_img_dir / f"{sid}.png", rgbd_imgs / f"{sid}.png")
            counter["rgbd"] += 1
            # 同一份 label 复制到两个数据集
            src_label = label_dir / f"{sid}.txt"
            shutil.copy2(src_label, rgb_labels / f"{sid}.txt")
            shutil.copy2(src_label, rgbd_labels / f"{sid}.txt")
            counter["label"] += 2  # 两份（rgb + rgbd）

    print(f"[复制] RGB 图={counter['rgb']}  RGBD 图={counter['rgbd']}  Label 文件(双份)={counter['label']}")

    # 校验：RGB / RGBD / Label 在三个 split 的样本编号必须完全一致
    def ids_in(d):
        return sorted(p.stem for p in Path(d).iterdir() if p.is_file())

    ok = True
    for split, ids in (("train", train), ("val", val), ("test", test)):
        r = set(ids_in(out_root / "rgb" / "images" / split))
        d = set(ids_in(out_root / "rgbd" / "images" / split))
        l_r = set(ids_in(out_root / "rgb" / "labels" / split))
        l_d = set(ids_in(out_root / "rgbd" / "labels" / split))
        same = (r == d == l_r == l_d == set(ids))
        ok = ok and same
        print(f"[校验] {split}: RGB={len(r)} RGBD={len(d)} Label(rgb/rgbd)={len(l_r)}/{len(l_d)} 完全一致={same}")
    print("RGB 与 RGBD 是否完全对应(同一 splits.json 驱动):", ok)


if __name__ == "__main__":
    main()
