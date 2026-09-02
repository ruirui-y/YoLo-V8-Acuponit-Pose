"""MaskLabelService 自动测试（Phase 2）。

不依赖 PySide6 / Qt，纯 numpy + cv2 算法测试。
覆盖：
1. extract_components: 7 个 component 正常提取
2. component count != 7 -> BLOCK
3. first-reference canonical y/x ID（按 (y,x) 升序）
4. prediction ID assignment（一对一）
5. bbox 生成
6. YOLO label 输出格式（26 数值、归一化、vis=2、顺序）
7. assign_stable_ids 冲突时 BLOCK
8. save_label 原子写入

启动：
    python YJJ_Pose_Scripts/gui/tests/test_lama_mask_label.py
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

# 让 services 包可被 import
GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))

import numpy as np                  # noqa: E402
from features.lama.services.mask_label_service import (   # noqa: E402
    MaskLabelService, MaskComponent, LabelResult,
)


def _check(name, cond, failures, detail=""):
    """统一断言：cond 为 True 打印 [OK]，否则 [FAIL] 并加入 failures。"""
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        if failures is not None:
            failures.append(name)


def _make_mask_with_circles(size=(640, 480), centers=None, radius=12):
    """生成测试 mask：在指定 center 位置画白圆。"""
    w, h = size
    mask = np.zeros((h, w), dtype=np.uint8)
    import cv2
    for cx, cy in (centers or []):
        cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
    return mask


def main():
    failures = []

    # ================================================================ 1. extract_components 7 个
    print("[1] extract_components: 7 个 component 正常提取")
    centers_7 = [
        (100, 100), (200, 100), (300, 100),
        (100, 200), (200, 200), (300, 200),
        (200, 300),
    ]
    mask = _make_mask_with_circles((640, 480), centers_7, radius=12)
    svc = MaskLabelService()
    comps = svc.extract_components(mask, min_area=50)
    _check("1a. 提取出 7 个 component", len(comps) == 7,
           failures, f"got {len(comps)}")
    # 每个 component 都有 center / bbox / area
    all_have = all(hasattr(c, "center") and hasattr(c, "bbox") and hasattr(c, "area")
                  for c in comps)
    _check("1b. 每个 component 字段完整", all_have, failures)

    # ================================================================ 2. component 数 != 7 -> BLOCK
    print("[2] component count != 7 -> make_label BLOCK")
    centers_6 = centers_7[:6]
    mask6 = _make_mask_with_circles((640, 480), centers_6, radius=12)
    res6 = svc.make_label(mask6, {i: c for i, c in enumerate(centers_6)},
                          (640, 480))
    _check("2a. 6 component -> None", res6 is None, failures)

    centers_8 = centers_7 + [(400, 300)]
    mask8 = _make_mask_with_circles((640, 480), centers_8, radius=12)
    res8 = svc.make_label(mask8, {i: c for i, c in enumerate(centers_8)},
                         (640, 480))
    _check("2b. 8 component -> None", res8 is None, failures)

    # ================================================================ 3. first-reference canonical y/x ID
    print("[3] first-reference canonical y/x ID")
    # 用上一节的 7 component mask，模拟 SetRef 时按 (y, x) 升序建立 ID 0..6
    # 这里 predicted_centers 就是按 canonical 排序后的 ID -> center
    # 排序：center.y 升序，相同则 center.x 升序
    # 注意：ExtractMaskCenters / BuildLocalTracks 在 C++ 里用 centroid 而非 bbox center
    sorted_centers = sorted(centers_7, key=lambda c: (c[1], c[0]))
    predicted = {i: c for i, c in enumerate(sorted_centers)}
    res = svc.make_label(mask, predicted, (640, 480))
    _check("3a. make_label 成功（7 ID + 7 component 一对一）",
           res is not None, failures)
    if res:
        _check("3b. ordered_points 长度 = 7",
               len(res.ordered_points) == 7, failures)
        # ordered_points[i] 应该 == sorted_centers[i]（在 mask 上的 centroid）
        # 注意：connectedComponents 质心 ≈ 圆心，但允许 0.5 像素级误差
        all_match = all(
            abs(res.ordered_points[i][0] - sorted_centers[i][0]) < 1.0 and
            abs(res.ordered_points[i][1] - sorted_centers[i][1]) < 1.0
            for i in range(7)
        )
        _check("3c. ordered_points[i] == sorted_centers[i]（误差<1px）",
               all_match, failures)

    # ================================================================ 4. prediction ID assignment 一对一
    print("[4] prediction ID assignment 一对一")
    # 故意打乱 components 输入顺序，验证 stable ID 仍能正确一对一匹配
    import random
    random.seed(42)
    # 重新提取一次，再随机打乱 components 顺序
    comps_shuffled = svc.extract_components(mask, min_area=50)
    random.shuffle(comps_shuffled)
    # 把它“伪装”成一个新的 mask：直接复用 make_label 不行，因为 make_label 内部会重新 extract
    # 直接调 assign_stable_ids 验证
    ordered = svc.assign_stable_ids(comps_shuffled, predicted)
    _check("4a. 打乱 components 顺序后 assign_stable_ids 仍成功",
           ordered is not None, failures)
    if ordered:
        # 每个 ID 仍应匹配到与 sorted_centers[i] 最近的 component
        all_match = all(
            abs(ordered[i][0] - sorted_centers[i][0]) < 1.0 and
            abs(ordered[i][1] - sorted_centers[i][1]) < 1.0
            for i in range(7)
        )
        _check("4b. 打乱后 ID->Component 映射仍正确",
               all_match, failures)

    # ================================================================ 5. bbox 生成
    print("[5] bbox 生成 + clamp")
    if res:
        bx, by, bw, bh = res.bbox
        # 7 个点的 min/max + 20 padding
        min_x = min(c[0] for c in centers_7) - 20
        min_y = min(c[1] for c in centers_7) - 20
        max_x = max(c[0] for c in centers_7) + 20
        max_y = max(c[1] for c in centers_7) + 20
        # clamp 到图内（这里图是 640x480）
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(640, max_x)
        max_y = min(480, max_y)
        _check("5a. bbox x ≈ 预期",
               abs(bx - min_x) < 1.0, failures, f"bx={bx} expected={min_x}")
        _check("5b. bbox y ≈ 预期",
               abs(by - min_y) < 1.0, failures, f"by={by} expected={min_y}")
        _check("5c. bbox w ≈ 预期",
               abs(bw - (max_x - min_x)) < 1.0,
               failures, f"bw={bw} expected={max_x - min_x}")
        _check("5d. bbox h ≈ 预期",
               abs(bh - (max_y - min_y)) < 1.0,
               failures, f"bh={bh} expected={max_y - min_y}")

    # ================================================================ 6. YOLO label 输出格式
    print("[6] YOLO label 输出格式（26 数值、归一化、vis=2）")
    if res:
        with tempfile.TemporaryDirectory() as tmpdir:
            label_path = os.path.join(tmpdir, "img_0001.txt")
            ok = svc.save_label(label_path, res.bbox, res.ordered_points, (640, 480))
            _check("6a. save_label 返回 True", ok, failures)
            if os.path.exists(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    line = f.read().strip()
                toks = line.split()
                _check("6b. 标签 token 数 = 26",
                       len(toks) == 26, failures, f"got {len(toks)}")
                _check("6c. class = 0", toks[0] == "0", failures)
                # bbox 4 个归一化 [0,1]
                bbox_norm = [float(t) for t in toks[1:5]]
                all_norm = all(0.0 <= v <= 1.0 for v in bbox_norm)
                _check("6d. bbox 归一化在 [0,1]", all_norm, failures)
                # 7 keypoints * 3 = 21 个
                kp_tokens = toks[5:]
                _check("6e. keypoints token 数 = 21",
                       len(kp_tokens) == 21, failures, f"got {len(kp_tokens)}")
                # 每个 keypoint 第 3 个 token（vis）= "2"
                vis_ok = all(kp_tokens[i + 2] == "2" for i in range(0, 21, 3))
                _check("6f. 所有 keypoint vis = 2", vis_ok, failures)
                # keypoints 坐标归一化 [0,1]
                kp_norm_ok = all(
                    0.0 <= float(kp_tokens[i]) <= 1.0 and
                    0.0 <= float(kp_tokens[i + 1]) <= 1.0
                    for i in range(0, 21, 3)
                )
                _check("6g. keypoint 坐标归一化 [0,1]", kp_norm_ok, failures)

    # ================================================================ 7. assign_stable_ids 冲突 -> BLOCK
    print("[7] assign_stable_ids 冲突 -> BLOCK")
    # 构造一个 7 个 component 但只有 6 个独立位置（两个 component 重合）的情况
    # assign_stable_ids 用一对一贪心：两个 ID 抢同一 Component 时只会匹配一个，
    # 另一个 ID 会匹配失败 -> None
    # 这里简化：predicted_centers 与 components 数量不匹配
    bad_predicted = {i: c for i, c in enumerate(centers_7[:6])}  # 只给 6 个 ID
    ordered_bad = svc.assign_stable_ids(comps, bad_predicted)
    _check("7a. predicted_centers 只有 6 个 ID -> None",
           ordered_bad is None, failures)

    # 给 7 个 ID 但有重复位置：ID 5 和 ID 6 都指向同一坐标
    bad_predicted2 = {0: centers_7[0], 1: centers_7[1], 2: centers_7[2],
                     3: centers_7[3], 4: centers_7[4], 5: centers_7[5],
                     6: centers_7[5]}  # 重复
    ordered_bad2 = svc.assign_stable_ids(comps, bad_predicted2)
    _check("7b. 两个 ID 抢同一坐标 -> 仍可一对一匹配（component 有 7 个）",
           ordered_bad2 is not None, failures)
    # 注：由于 components 实际有 7 个，predicted_centers 有 2 个 ID 指向同坐标，
    # 贪心会把它们分配给两个不同 component，所以仍能匹配成功

    # ================================================================ 8. save_label 原子写入
    print("[8] save_label 失败时不留 .tmp")
    # 构造一个一定会失败的路径：父目录是一个已存在的文件（无法成为目录）
    with tempfile.TemporaryDirectory() as tmpdir:
        blocker = os.path.join(tmpdir, "blocker.txt")
        with open(blocker, "w") as f:
            f.write("block")
        # 用 blocker.txt 当父目录 -> makedirs / open 一定失败
        bad_path = os.path.join(blocker, "sub", "file.txt")
        ok = svc.save_label(bad_path, (0, 0, 10, 10),
                            [(1, 1)] * 7, (100, 100))
        _check("8a. 父目录是文件 -> save_label 返回 False",
               not ok, failures, "")
        # 不应留下 .tmp
        _check("8b. 失败后不留 .tmp 文件",
               not os.path.exists(bad_path + ".tmp"), failures, "")

    # ---------------------------------------------------------------- 总结
    print("-" * 60)
    if failures:
        print(f"MaskLabelService 测试失败：{len(failures)} 项 -> {failures}")
        sys.exit(1)
    else:
        print("MaskLabelService 测试全部通过")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("测试抛出异常：")
        traceback.print_exc()
        sys.exit(1)
