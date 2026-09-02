"""
构建「hand raw-depth RGBD」正式数据集 (不训练, 不覆盖旧数据)。

- 严格读取 dataset/hand/splits.json 的 train/val/test 划分 (不重新随机)
- 图片来源: out_rawdepth/color_<id>.png (raw npy depth 版 4ch)
- labels 复用: dataset/hand/rgbd/labels (并断言与 rgb/labels 逐文件一致)
- 输出:   dataset/hand/rgbd_rawdepth/
           images/{train,val,test}
           labels/{train,val,test}
           data_rgbd_rawdepth.yaml
"""
import json
import shutil
from pathlib import Path

BASE = Path(r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand")
SPLITS = BASE / "splits.json"
SRC_IMG = Path(r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_rawdepth")
SRC_LBL_RGB = BASE / "rgb/labels"
SRC_LBL_RGBD = BASE / "rgbd/labels"
OUT = BASE / "rgbd_rawdepth"


def main():
    sp = json.loads(SPLITS.read_text(encoding="utf-8"))
    assert sp["n"]["train"] == 57 and sp["n"]["val"] == 12 and sp["n"]["test"] == 12, "split 数量异常"

    # ---- 0. 断言两套 labels 逐文件一致 (决定复用源) ----
    all_ids = sp["train"] + sp["val"] + sp["test"]
    lbl_diff = 0
    for split in ("train", "val", "test"):
        for sid in sp[split]:
            a = SRC_LBL_RGB / split / f"{sid}.txt"
            b = SRC_LBL_RGBD / split / f"{sid}.txt"
            if a.read_bytes() != b.read_bytes():
                lbl_diff += 1
    print(f"[labels 一致性] rgb vs rgbd 不一致文件数 = {lbl_diff} (期望 0)")
    assert lbl_diff == 0, "rgb/labels 与 rgbd/labels 不一致, 停止以免复制错误源"

    # ---- 1. 构建目录 + 复制 ----
    copied_img = copied_lbl = 0
    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        for sid in sp[split]:
            # 图片
            src_png = SRC_IMG / f"{sid}.png"
            assert src_png.exists(), f"源图缺失: {src_png}"
            dst_png = OUT / "images" / split / f"{sid}.png"
            if not dst_png.exists() or dst_png.read_bytes() != src_png.read_bytes():
                shutil.copy2(src_png, dst_png)
            copied_img += 1
            # label (复用 rgbd/labels)
            src_lbl = SRC_LBL_RGBD / split / f"{sid}.txt"
            assert src_lbl.exists(), f"源 label 缺失: {src_lbl}"
            dst_lbl = OUT / "labels" / split / f"{sid}.txt"
            if not dst_lbl.exists() or dst_lbl.read_bytes() != src_lbl.read_bytes():
                shutil.copy2(src_lbl, dst_lbl)
            copied_lbl += 1
    print(f"[复制] 图片={copied_img}  label={copied_lbl}")

    # ---- 2. YAML ----
    yaml_text = (
        "path: H:/YJJ/Yolo_RGBD/Resource/session1_200146/dataset/hand/rgbd_rawdepth\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n"
        "nc: 1\n"
        "names:\n"
        "  0: hand\n"
        "rgbd: true\n"
        "channels: 4\n"
        "kpt_shape: [7, 3]\n"
    )
    (OUT / "data_rgbd_rawdepth.yaml").write_text(yaml_text, encoding="utf-8")
    print(f"[YAML] 已写出 {OUT / 'data_rgbd_rawdepth.yaml'}")

    # ---- 3. 三套 split ID 一致性 ----
    # 从实际目录反推 ID, 与 splits.json 及 rgb/rgbd 旧数据集比对
    def ids_in(root, split):
        d = root / "images" / split
        # rgb 用 .jpg, rgbd/rawdepth 用 .png, 按 stem 取 ID 比较
        return sorted(p.stem for p in d.glob("*") if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    new_ids = {s: ids_in(OUT, s) for s in ("train", "val", "test")}
    rgb_ids = {s: ids_in(BASE / "rgb", s) for s in ("train", "val", "test")}
    rgbd_ids = {s: ids_in(BASE / "rgbd", s) for s in ("train", "val", "test")}
    sp_ids = {s: sorted(sp[s]) for s in ("train", "val", "test")}
    print("\n[三套 split 一致性]")
    for s in ("train", "val", "test"):
        a = (new_ids[s] == sp_ids[s])
        b = (new_ids[s] == rgb_ids[s])
        c = (new_ids[s] == rgbd_ids[s])
        print(f"  {s}: new==splits.json={a}  new==rgb={b}  new==rgbd={c}  "
              f"(count new={len(new_ids[s])} splits={len(sp_ids[s])})")
    assert all(new_ids[s] == sp_ids[s] and new_ids[s] == rgb_ids[s] and new_ids[s] == rgbd_ids[s]
               for s in ("train", "val", "test")), "split ID 不一致!"

    # ---- 4. 每张图 4ch + label 合法性 ----
    import cv2
    import numpy as np
    bad_shape = bad_dtype = no_lbl = bad_fmt = 0
    for split in ("train", "val", "test"):
        for sid in sp[split]:
            im = cv2.imread(str(OUT / "images" / split / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
            if im is None or im.shape != (720, 1280, 4):
                bad_shape += 1
            elif im.dtype != np.uint8:
                bad_dtype += 1
            lbl = OUT / "labels" / split / f"{sid}.txt"
            if not lbl.exists():
                no_lbl += 1
            else:
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    toks = line.split()
                    if len(toks) != 26:   # 5 + 7*3
                        bad_fmt += 1
    print(f"\n[4ch+label 检查] bad_shape={bad_shape} bad_dtype={bad_dtype} "
          f"no_label={no_lbl} bad_format(≠26字段)={bad_fmt}")
    assert bad_shape == bad_dtype == no_lbl == bad_fmt == 0, "存在非法文件!"

    # ---- 5. 随机 20 张: 新图前三通道 vs 原始 out_image RGB ----
    rng = np.random.default_rng(42)
    orig_rgb_dir = Path(r"H:/YJJ/Yolo_RGBD/Resource/session1_200146/out_image")
    all_new = [(s, sid) for s in ("train", "val", "test") for sid in sp[s]]
    sub = rng.choice(len(all_new), size=20, replace=False)
    mism = 0
    for idx in sub:
        s, sid = all_new[idx]
        new_im = cv2.imread(str(OUT / "images" / s / f"{sid}.png"), cv2.IMREAD_UNCHANGED)
        orig = cv2.imread(str(orig_rgb_dir / f"{sid}.jpg"), cv2.IMREAD_UNCHANGED)
        if not np.array_equal(new_im[:, :, :3], orig[:, :, :3]):
            mism += 1
    print(f"[RGB 前三通道一致性] 20 张中不一致 = {mism} (期望 0)")

    print("\n=== 构建完成 ===")
    print(f"  新数据集路径: {OUT}")
    print(f"  train/val/test = {len(new_ids['train'])}/{len(new_ids['val'])}/{len(new_ids['test'])}")
    print(f"  三套 split 100% 一致: True")
    print(f"  labels 100% 一致 (rgb==rgbd, 且已逐文件复用): True")
    print(f"  4ch 检查通过: True")
    print(f"  RGB 前三通道 100% 一致: {mism == 0}")


if __name__ == "__main__":
    main()
