"""Phase 4 自动测试：A + TestOne 走同一 predict API + QImage 转换。

不依赖完整 GUI（不需要 LamaPage / LamaController 实例），
直接测试：
1. LamaController 静态方法：QImage <-> numpy 转换 round-trip
2. ReferenceAlignmentService.set_reference -> predict(assist) -> predict(strict)
   验证 A 与 TestOne 走同一 predict()，返回类型一致
3. _get_predicted_centers 逻辑（从 PredictionResult 提取 ID -> center）
4. stable ID 跨 A / TestOne 调用保持
5. mask overlay 生成不报错

启动：
    python YJJ_Pose_Scripts/gui/tests/test_lama_phase4_predict.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# 让 services / widgets 包可被 import
GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))

import numpy as np                  # noqa: E402
import cv2                          # noqa: E402

# PySide6 QImage 不需要 QApplication 即可创建
from PySide6.QtGui import QImage    # noqa: E402

from features.lama.services import (    # noqa: E402
    ReferenceAlignmentService, PredictionResult,
)
from features.lama.lama_controller import LamaController    # noqa: E402


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        if failures is not None:
            failures.append(name)


def _make_test_image_and_mask(size: tuple[int, int] = (640, 480), centers: list[tuple[int, int]] | None = None, patch_size: int = 30, shift: tuple[int, int] = (0, 0)) -> tuple[np.ndarray, np.ndarray]:
    """生成测试图 + mask：在指定 center 周围放置随机噪声 patch。"""
    w, h = size
    img = np.full((h, w, 3), 30, dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    half = patch_size // 2
    for i, (cx, cy) in enumerate(centers or []):
        patch_rng = np.random.RandomState(i + 1)
        patch = patch_rng.randint(0, 256, (patch_size, patch_size, 3), dtype=np.uint8)
        tx = int(cx + shift[0])
        ty = int(cy + shift[1])
        x0 = max(0, tx - half)
        y0 = max(0, ty - half)
        x1 = min(w, tx - half + patch_size)
        y1 = min(h, ty - half + patch_size)
        if x1 > x0 and y1 > y0:
            img[y0:y1, x0:x1] = patch[0:y1 - y0, 0:x1 - x0]
            cv2.circle(mask, (tx, ty), 12, 255, -1)
    return img, mask


def _np_to_qimage_rgb(np_rgb: np.ndarray) -> QImage:
    """numpy HxWx3 RGB -> QImage RGB888（测试辅助）。"""
    h, w = np_rgb.shape[:2]
    np_rgb = np.ascontiguousarray(np_rgb)
    img = QImage(np_rgb.tobytes(), w, h, w * 3, QImage.Format.Format_RGB888)
    return img.copy()


def _np_to_qimage_gray(np_gray: np.ndarray) -> QImage:
    """numpy HxW uint8 -> QImage Grayscale8（测试辅助）。"""
    h, w = np_gray.shape[:2]
    np_gray = np.ascontiguousarray(np_gray)
    img = QImage(np_gray.tobytes(), w, h, w, QImage.Format.Format_Grayscale8)
    return img.copy()


def main() -> None:
    failures = []

    # 7 个固定 center（按 (y, x) 升序排好的）
    centers_7_sorted = [
        (100, 100), (200, 100), (300, 100),
        (100, 200), (200, 200), (300, 200),
        (200, 300),
    ]

    # ================================================================ 1. QImage <-> numpy 转换
    print("[1] QImage <-> numpy 转换 round-trip")
    ref_img_np, ref_mask_np = _make_test_image_and_mask(
        size=(640, 480), centers=centers_7_sorted, patch_size=30
    )

    # numpy -> QImage -> numpy
    qimg_rgb = _np_to_qimage_rgb(ref_img_np)
    rgb_back = LamaController._qimage_to_rgb_np(qimg_rgb)
    _check("1a. RGB QImage -> numpy 形状一致",
           rgb_back.shape == ref_img_np.shape, failures,
           f"got {rgb_back.shape} vs {ref_img_np.shape}")
    _check("1b. RGB 数据一致（像素完全相同）",
           np.array_equal(rgb_back, ref_img_np), failures)

    qimg_mask = _np_to_qimage_gray(ref_mask_np)
    mask_back = LamaController._mask_qimage_to_np(qimg_mask)
    _check("1c. Gray QImage -> numpy 形状一致",
           mask_back.shape == ref_mask_np.shape, failures,
           f"got {mask_back.shape} vs {ref_mask_np.shape}")
    _check("1d. Gray 数据一致（0/255 完全相同）",
           np.array_equal(mask_back, ref_mask_np), failures)

    # numpy -> QImage（反向）
    qimg_back = LamaController._np_mask_to_qimage(ref_mask_np)
    _check("1e. _np_mask_to_qimage 不为空",
           not qimg_back.isNull(), failures)
    _check("1f. _np_mask_to_qimage 尺寸正确",
           qimg_back.width() == 640 and qimg_back.height() == 480, failures)

    # 空 QImage 边界
    empty_rgb = LamaController._qimage_to_rgb_np(QImage())
    _check("1g. 空 QImage -> 空数组",
           empty_rgb.size == 0, failures)

    # ================================================================ 2. SetRef + A predict + TestOne predict 同一 API
    print("[2] SetRef + A predict(assist) + TestOne predict(strict) 同一 API")
    svc = ReferenceAlignmentService()
    ok = svc.set_reference(ref_img_np, ref_mask_np,
                           ordered_points=centers_7_sorted, ref_rect=None)
    _check("2a. set_reference 成功", ok, failures)
    _check("2b. is_ready() == True", svc.is_ready(), failures)

    # A 键：predict(assist)
    result_a = svc.predict(ref_img_np, mode="assist")
    _check("2c. A predict 返回 PredictionResult",
           isinstance(result_a, PredictionResult), failures)
    _check("2d. A predict mask 不为 None", result_a.mask is not None, failures)
    _check("2e. A predict success == 7", result_a.success == 7, failures,
           f"got {result_a.success}")

    # TestOne：predict(strict)
    result_t = svc.predict(ref_img_np, mode="strict")
    _check("2f. TestOne predict 返回 PredictionResult",
           isinstance(result_t, PredictionResult), failures)
    _check("2g. TestOne predict mask 不为 None", result_t.mask is not None, failures)
    _check("2h. TestOne predict success == 7", result_t.success == 7, failures,
           f"got {result_t.success}")

    # 验证 A 与 TestOne 走同一 predict 方法（通过 type(result) 一致判断）
    _check("2i. A 与 TestOne 返回类型完全一致",
           type(result_a) == type(result_t) == PredictionResult, failures)

    # ================================================================ 3. _get_predicted_centers 逻辑
    print("[3] _get_predicted_centers: ID -> predictedCenter 映射")
    # 模拟 Controller 的 _current_prediction = result_a
    # 然后用 _get_predicted_centers 提取（这里直接复用逻辑，不创建 Controller）

    # Controller._get_predicted_centers 是实例方法，但只读 self._current_prediction
    # 这里直接复刻其逻辑做测试
    def get_predicted_centers(prediction: "PredictionResult | None") -> dict[int, tuple[float, float]]:
        if prediction is None:
            return {}
        centers = {}
        for t in prediction.tracks:
            if t.ok:
                centers[t.id] = t.current_center
            else:
                centers[t.id] = t.ref_center
        return centers

    centers_a = get_predicted_centers(result_a)
    _check("3a. predicted_centers 长度 == 7",
           len(centers_a) == 7, failures, f"got {len(centers_a)}")
    _check("3b. predicted_centers keys == {0..6}",
           set(centers_a.keys()) == set(range(7)), failures,
           f"got {set(centers_a.keys())}")
    # 每个 ID 都有合理的 center（非 (0,0)）
    all_valid = all(
        c[0] != 0.0 or c[1] != 0.0
        for c in centers_a.values()
    )
    _check("3c. 所有 predicted centers 非零", all_valid, failures)

    # 空 prediction -> 空 dict
    empty_centers = get_predicted_centers(None)
    _check("3d. None prediction -> 空 dict",
           empty_centers == {}, failures)

    # ================================================================ 4. stable ID 跨 A / TestOne 调用保持
    print("[4] stable ID 跨 A / TestOne 调用保持")
    # 同一参考图 + 同一目标图，两次 predict 的 ID 必须一致
    id_a = sorted([t.id for t in result_a.tracks])
    id_t = sorted([t.id for t in result_t.tracks])
    _check("4a. A 与 TestOne 的 track ID 集合一致",
           id_a == id_t == list(range(7)), failures,
           f"A={id_a} T={id_t}")

    # A predict 多次调用，ID 不变
    result_a2 = svc.predict(ref_img_np, mode="assist")
    id_a2 = sorted([t.id for t in result_a2.tracks])
    _check("4b. 两次 A predict 的 track ID 一致",
           id_a == id_a2, failures)

    # ID 0 对应的 ref_center 应接近 (100, 100)
    track0 = next(t for t in result_a.tracks if t.id == 0)
    _check("4c. ID 0 的 ref_center ≈ (100, 100)",
           abs(track0.ref_center[0] - 100) < 5 and
           abs(track0.ref_center[1] - 100) < 5, failures,
           f"got {track0.ref_center}")

    # ================================================================ 5. 位移图 predict + ID 保持
    print("[5] 位移图 predict + stable ID 保持")
    shift = (50, 30)
    target_img, _ = _make_test_image_and_mask(
        size=(640, 480), centers=centers_7_sorted, patch_size=30, shift=shift
    )
    result_shift = svc.predict(target_img, mode="assist")
    _check("5a. 位移图 predict mask 不为 None",
           result_shift.mask is not None, failures)
    _check("5b. 位移图 predict success == 7",
           result_shift.success == 7, failures,
           f"got {result_shift.success}")

    # ID 集合不变
    id_shift = sorted([t.id for t in result_shift.tracks])
    _check("5c. 位移图 predict 的 track ID 集合不变",
           id_shift == list(range(7)), failures)

    # 全局位移应接近 shift
    _check("5d. global_dx ≈ 50",
           abs(result_shift.global_dx - 50) <= 5, failures,
           f"got {result_shift.global_dx}")
    _check("5e. global_dy ≈ 30",
           abs(result_shift.global_dy - 30) <= 5, failures,
           f"got {result_shift.global_dy}")

    # ================================================================ 6. mask overlay 生成不报错
    print("[6] mask overlay 生成不报错（_make_mask_overlay_rgba）")
    try:
        from features.lama.widgets.inpaint_canvas import _make_mask_overlay_rgba
        mask_q = _np_to_qimage_gray(ref_mask_np)
        overlay = _make_mask_overlay_rgba(mask_q, alpha=120)
        _check("6a. overlay 不为空", not overlay.isNull(), failures)
        _check("6b. overlay 尺寸正确",
               overlay.width() == 640 and overlay.height() == 480, failures)
    except Exception as e:
        _check("6. mask overlay 生成", False, failures, str(e))

    # ---------------------------------------------------------------- 总结
    print("-" * 60)
    if failures:
        print(f"Phase 4 测试失败：{len(failures)} 项 -> {failures}")
        sys.exit(1)
    else:
        print("Phase 4 测试全部通过")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("测试抛出异常：")
        traceback.print_exc()
        sys.exit(1)
