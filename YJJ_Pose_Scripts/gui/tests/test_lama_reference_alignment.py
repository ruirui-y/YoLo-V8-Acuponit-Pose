"""ReferenceAlignmentService 自动测试（Phase 3）。

不依赖 PySide6 / Qt，纯 numpy + cv2 算法测试。
覆盖：
1. set_reference：7 个 component 正常构建 local tracks
2. stable ID：第一次 SetRef 按 (y,x) 升序固定
3. predict strict：相同图 -> 7/7 成功，mask 与参考 mask 一致
4. predict strict：位移图 -> mask 应跟随位移
5. predict assist：允许部分成功，画圆
6. extract_mask_centers 静态工具
7. set_reference 失败 -> is_ready() == False
8. predict 未 set_reference -> 返回空结果

启动：
    python YJJ_Pose_Scripts/gui/tests/test_lama_reference_alignment.py
"""
import os
import sys
import traceback
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))

import numpy as np                  # noqa: E402
import cv2                          # noqa: E402
from features.lama.services.reference_alignment_service import (   # noqa: E402
    ReferenceAlignmentService, PredictionResult,
)


def _check(name, cond, failures, detail=""):
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        if failures is not None:
            failures.append(name)


def _make_test_image_and_mask(size=(640, 480), centers=None, patch_size=30, shift=(0, 0)):
    """生成测试图 + mask：在指定 center 周围放置随机噪声 patch。

    返回 (image_rgb HxWx3 uint8, mask HxW uint8 0/255)
    shift: 整体位移（dx, dy），用于模拟目标图相对参考图的位移
    """
    w, h = size
    rng = np.random.RandomState(42)
    img = np.full((h, w, 3), 30, dtype=np.uint8)        # 灰色背景
    mask = np.zeros((h, w), dtype=np.uint8)

    half = patch_size // 2
    for i, (cx, cy) in enumerate(centers or []):
        # 每个 patch 用不同 seed 的随机噪声，保证可区分
        patch_rng = np.random.RandomState(i + 1)
        patch = patch_rng.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
        # 整体位移
        tx = int(cx + shift[0])
        ty = int(cy + shift[1])
        x0 = max(0, tx - half)
        y0 = max(0, ty - half)
        x1 = min(w, tx - half + patch_size)
        y1 = min(h, ty - half + patch_size)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = patch[0:y1 - y0, 0:x1 - x0]
            # mask 画在原 patch 中心位置（半径 12）
            cv2.circle(mask, (tx, ty), 12, 255, -1)
    return img, mask


def main():
    failures = []
    rng = np.random.RandomState(0)

    # ---- 7 个固定 center（按 (y, x) 升序排好的，方便验证 stable ID）----
    # 注意：必须保证 (y, x) 升序后 ID 0..6 与构造顺序一致
    centers_7_sorted = [
        (100, 100), (200, 100), (300, 100),       # y=100 行
        (100, 200), (200, 200), (300, 200),       # y=200 行
        (200, 300),                                # y=300 行
    ]

    # ================================================================ 1. set_reference + 7 个 track
    print("[1] set_reference：7 个 component -> 7 local tracks")
    ref_img, ref_mask = _make_test_image_and_mask(
        size=(640, 480), centers=centers_7_sorted, patch_size=30)
    svc = ReferenceAlignmentService()
    ok = svc.set_reference(ref_img, ref_mask,
                           ordered_points=centers_7_sorted,
                           ref_rect=None)
    _check("1a. set_reference 返回 True", ok, failures)
    _check("1b. is_ready() == True", svc.is_ready(), failures)
    _check("1c. local_tracks 数 == 7",
           len(svc.local_tracks()) == 7, failures,
           f"got {len(svc.local_tracks())}")

    # ================================================================ 2. stable ID：按 (y, x) 排序
    print("[2] stable ID：按 (y, x) 升序固定 ID 0..6")
    tracks = svc.local_tracks()
    # 验证 ID 顺序
    ids_ok = all(tracks[i].id == i for i in range(7))
    _check("2a. track.id == 0..6 顺序", ids_ok, failures)
    # 验证 ref_center 按 (y, x) 升序
    centers_ok = all(
        (tracks[i].ref_center[1], tracks[i].ref_center[0])
        < (tracks[i + 1].ref_center[1], tracks[i + 1].ref_center[0])
        for i in range(6)
    )
    _check("2b. ref_center 按 (y, x) 升序", centers_ok, failures)
    # 验证 ID 0 对应第一个 (y, x) 最小的点
    _check("2c. ID 0 的 ref_center ≈ (100, 100)",
           abs(tracks[0].ref_center[0] - 100) < 5 and
           abs(tracks[0].ref_center[1] - 100) < 5, failures,
           f"got {tracks[0].ref_center}")

    # ================================================================ 3. predict strict：相同图 -> 7/7 成功
    print("[3] predict strict：相同图 -> 7/7 成功")
    res = svc.predict(ref_img, mode="strict")
    _check("3a. mask 不为 None", res.mask is not None, failures)
    _check("3b. success == 7", res.success == 7, failures,
           f"got {res.success}")
    _check("3c. total == 7", res.total == 7, failures)
    _check("3d. 所有 track.ok == True",
           all(t.ok for t in res.tracks), failures)
    # mask 形状与原图一致
    _check("3e. mask 尺寸与参考一致",
           res.mask.shape == ref_mask.shape, failures,
           f"got {res.mask.shape}")
    # 每个预测中心应与 ref_center 一致（相同图，位移为 0）
    # 注意：predict 返回的 tracks 顺序与 local_tracks 一致（ID 0..6）
    dx_dy_zero = all(
        abs(t.dx) <= svc.max_local_deviation_x and
        abs(t.dy) <= svc.max_local_deviation_y
        for t in res.tracks
    )
    _check("3f. 相同图位移接近 0（在容差内）", dx_dy_zero, failures,
           f"dx/dy: {[(t.dx, t.dy) for t in res.tracks]}")

    # ================================================================ 4. predict strict：位移图 -> mask 跟随
    print("[4] predict strict：位移图 -> mask 跟随位移")
    shift = (50, 30)
    target_img, _ = _make_test_image_and_mask(
        size=(640, 480), centers=centers_7_sorted, patch_size=30, shift=shift)
    res2 = svc.predict(target_img, mode="strict")
    _check("4a. mask 不为 None", res2.mask is not None, failures)
    _check("4b. success == 7", res2.success == 7, failures,
           f"got {res2.success}")
    # 全局位移应接近 shift
    _check("4c. global_dx ≈ 50", abs(res2.global_dx - 50) <= 5, failures,
           f"got {res2.global_dx}")
    _check("4d. global_dy ≈ 30", abs(res2.global_dy - 30) <= 5, failures,
           f"got {res2.global_dy}")

    # ================================================================ 5. predict assist：允许部分成功
    print("[5] predict assist：允许部分成功，画圆")
    res3 = svc.predict(target_img, mode="assist")
    _check("5a. mask 不为 None", res3.mask is not None, failures)
    _check("5b. assist mask 形状正确",
           res3.mask.shape == ref_mask.shape, failures)
    # assist mask 应有非零像素（圆）
    _check("5c. assist mask 非空", np.any(res3.mask > 0), failures)
    # 在 strict 模式下应该 7/7 成功，所以 assist 也至少 7/7
    _check("5d. assist success == 7", res3.success == 7, failures,
           f"got {res3.success}")

    # ================================================================ 6. extract_mask_centers 静态
    print("[6] extract_mask_centers 静态工具")
    centers = ReferenceAlignmentService.extract_mask_centers(ref_mask, min_component_area=50)
    _check("6a. 提取出 7 个 center", len(centers) == 7, failures,
           f"got {len(centers)}")
    # 验证每个 center 接近 centers_7_sorted
    matched = 0
    for c in centers:
        for ref_c in centers_7_sorted:
            if abs(c[0] - ref_c[0]) < 5 and abs(c[1] - ref_c[1]) < 5:
                matched += 1
                break
    _check("6b. 所有 center 与预期对应", matched == 7, failures,
           f"matched {matched}/7")

    # ================================================================ 7. set_reference 失败 -> not ready
    print("[7] set_reference 失败 -> not ready")
    svc2 = ReferenceAlignmentService()
    # 空 mask
    empty_mask = np.zeros((480, 640), dtype=np.uint8)
    ok = svc2.set_reference(ref_img, empty_mask)
    _check("7a. 空 mask -> set_reference False", not ok, failures)
    _check("7b. is_ready() == False", not svc2.is_ready(), failures)

    # ================================================================ 8. predict 未 set_reference
    print("[8] predict 未 set_reference -> 空结果")
    res8 = svc2.predict(ref_img, mode="strict")
    _check("8a. mask is None", res8.mask is None, failures)
    _check("8b. success == 0", res8.success == 0, failures)
    _check("8c. total == 0", res8.total == 0, failures)

    # ================================================================ 9. 尺寸不匹配 -> 空
    print("[9] 目标图尺寸不匹配 -> predict 失败")
    small_img = np.zeros((300, 300, 3), dtype=np.uint8)
    res9 = svc.predict(small_img, mode="strict")
    _check("9a. 尺寸不匹配 mask is None", res9.mask is None, failures)

    # ================================================================ 10. stable ID 跨调用保持
    print("[10] stable ID 跨调用保持：第二次 predict 同一图，ID 不变")
    res10a = svc.predict(ref_img, mode="strict")
    res10b = svc.predict(ref_img, mode="strict")
    id_consistent = all(
        res10a.tracks[i].id == res10b.tracks[i].id == i
        for i in range(7)
    )
    _check("10a. 两次 predict 的 track.id 一致", id_consistent, failures)

    # ---------------------------------------------------------------- 总结
    print("-" * 60)
    if failures:
        print(f"ReferenceAlignmentService 测试失败：{len(failures)} 项 -> {failures}")
        sys.exit(1)
    else:
        print("ReferenceAlignmentService 测试全部通过")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("测试抛出异常：")
        traceback.print_exc()
        sys.exit(1)
