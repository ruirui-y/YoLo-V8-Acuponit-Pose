"""Phase 6 自动测试：Q 原子 Commit 状态机。

不依赖真实 ONNX 模型，测试 Q 流程的验证逻辑和原子写辅助方法：
1. Q BLOCK：模型未加载
2. Q BLOCK：canvas 无图
3. Q BLOCK：mask 为空
4. Q BLOCK：mask component 数 != 7
5. _CommitSnapshot 数据结构
6. _InpaintWorker 创建（不启动）
7. _cleanup_tmp 静态方法
8. Q 不会提前切图（busy 期间 Prev/Next 被拒绝）

需要 QApplication（创建 LamaPage/Controller），但不需要事件循环。
启动：
    python YJJ_Pose_Scripts/gui/tests/test_lama_phase6_commit.py
"""
import os
import sys
import tempfile
import traceback
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))

import numpy as np                  # noqa: E402
import cv2                          # noqa: E402

# 创建 QApplication（QImage/QWidget 需要）
from PySide6.QtWidgets import QApplication   # noqa: E402
app = QApplication.instance() or QApplication([])

from features.lama.lama_page import LamaPage             # noqa: E402
from features.lama.lama_controller import (                # noqa: E402
    LamaController, _CommitSnapshot, _InpaintWorker,
)


def _check(name, cond, failures, detail=""):
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        if failures is not None:
            failures.append(name)


def _make_test_image(path, size=(640, 480), centers=None):
    """生成测试图片并保存到 path。"""
    w, h = size
    img = np.full((h, w, 3), 100, dtype=np.uint8)
    for i, (cx, cy) in enumerate(centers or []):
        cv2.circle(img, (int(cx), int(cy)), 20, (200, 200, 200), -1)
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def _draw_7_masks(canvas, centers, radius=12):
    """在 canvas 上画 7 个 mask 圆。"""
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QPainter, QPen, QBrush, QColor
    mask = canvas.mask_image()
    p = QPainter(mask)
    p.setPen(QPen(QColor(255, 255, 255)))
    p.setBrush(QBrush(QColor(255, 255, 255)))
    for cx, cy in centers:
        p.drawEllipse(QPointF(cx, cy), float(radius), float(radius))
    p.end()
    canvas.set_mask(mask)


def main():
    failures = []
    tmpdir = tempfile.mkdtemp(prefix="lama_phase6_test_")

    # ================================================================ 1. 创建 Page + Controller
    print("[1] 创建 LamaPage + LamaController")
    page = LamaPage()
    ctrl = page.controller
    _check("1a. LamaPage 创建成功", page is not None, failures)
    _check("1b. LamaController 创建成功", ctrl is not None, failures)
    _check("1c. busy == False", not ctrl._busy, failures)
    _check("1d. _commit_snapshot == None", ctrl._commit_snapshot is None, failures)
    _check("1e. _inpaint_worker == None", ctrl._inpaint_worker is None, failures)

    # ================================================================ 2. Q BLOCK：模型未加载
    print("[2] Q BLOCK：模型未加载")
    # 先打开一张图
    img_path = os.path.join(tmpdir, "test001.png")
    _make_test_image(img_path, (640, 480), [(200, 200)])
    # 模拟 Open（直接调内部方法）
    ctrl._image_files = [img_path]
    ctrl._current_index = 0
    ctrl._load_current_image()
    _check("2a. canvas 有图", not page.canvas.source_image().isNull(), failures)

    # 先画 7 个 mask（让 mask 检查通过），才会到模型未加载检查
    centers_7_for_2b = [
        (100, 100), (200, 100), (300, 100),
        (100, 200), (200, 200), (300, 200),
        (200, 300),
    ]
    page.canvas.clear_mask()
    _draw_7_masks(page.canvas, centers_7_for_2b, radius=12)

    # Q -> 应该 BLOCK（模型未加载，mask 已通过 7 component 检查）
    ctrl._on_commit()
    status_after = page._status_label.text()
    _check("2b. Q 模型未加载 -> BLOCK",
           "model not loaded" in status_after.lower() or "LoadModel" in status_after,
           failures, f"status='{status_after}'")
    _check("2c. BLOCK 后 busy 仍为 False",
           not ctrl._busy, failures)
    _check("2d. BLOCK 后 current_index 不变",
           ctrl._current_index == 0, failures)

    # ================================================================ 3. Q BLOCK：mask 为空
    print("[3] Q BLOCK：mask 为空")
    # 注入 mock 模型（is_loaded() 返回 True）
    ctrl._lama_inference_service._session = True      # hack: is_loaded 检查 _session is not None
    _check("3a. mock 模型 is_loaded() == True",
           ctrl._lama_inference_service.is_loaded(), failures)

    # mask 为空 -> Q BLOCK
    page.canvas.clear_mask()
    ctrl._on_commit()
    status = page._status_label.text()
    _check("3b. Q mask 为空 -> BLOCK",
           "empty" in status.lower() or "mask" in status.lower(),
           failures, f"status='{status}'")

    # ================================================================ 4. Q BLOCK：mask component != 7
    print("[4] Q BLOCK：mask component != 7")
    # 画 6 个圆（不是 7 个）—— 必须 clear 再画，避免与上一轮 mask 叠加
    centers_6 = [
        (100, 100), (200, 100), (300, 100),
        (100, 200), (200, 200), (300, 200),
    ]
    page.canvas.clear_mask()
    _draw_7_masks(page.canvas, centers_6, radius=12)
    ctrl._on_commit()
    status = page._status_label.text()
    _check("4a. Q 6 component -> BLOCK",
           "7" in status or "mask" in status.lower(),
           failures, f"status='{status}'")
    _check("4b. BLOCK 后 current_index 不变",
           ctrl._current_index == 0, failures)

    # 画 8 个圆（不是 7 个）—— 必须先 clear 再画，避免与上一次叠加
    centers_8_real = [
        (100, 100), (200, 100), (300, 100), (400, 100),
        (100, 200), (200, 200), (300, 200),
        (200, 300),
    ]
    page.canvas.clear_mask()
    _draw_7_masks(page.canvas, centers_8_real, radius=12)
    ctrl._on_commit()
    status = page._status_label.text()
    _check("4c. Q 8 component -> BLOCK",
           "7" in status or "mask" in status.lower(),
           failures, f"status='{status}'")
    _check("4d. BLOCK 后 current_index 不变",
           ctrl._current_index == 0, failures)

    # ================================================================ 5. _CommitSnapshot 数据结构
    print("[5] _CommitSnapshot 数据结构")
    snap = _CommitSnapshot(
        current_index=3,
        current_path="/tmp/img003.png",
        original_rgb=np.zeros((10, 10, 3), dtype=np.uint8),
        final_mask=np.zeros((10, 10), dtype=np.uint8),
        ordered_points=[(1, 2), (3, 4)],
        bbox=(0, 0, 10, 10),
        image_size=(10, 10),
        output_path="/tmp/out/img003.png",
        label_path="/tmp/labels/img003.txt",
    )
    _check("5a. current_index == 3", snap.current_index == 3, failures)
    _check("5b. output_path 正确",
           snap.output_path == "/tmp/out/img003.png", failures)
    _check("5c. label_path 正确",
           snap.label_path == "/tmp/labels/img003.txt", failures)
    _check("5d. ordered_points 长度 == 2",
           len(snap.ordered_points) == 2, failures)

    # ================================================================ 6. _InpaintWorker 创建（不启动）
    print("[6] _InpaintWorker 创建（不启动）")
    dummy_img = np.zeros((10, 10, 3), dtype=np.uint8)
    dummy_mask = np.zeros((10, 10), dtype=np.uint8)
    worker = _InpaintWorker(
        ctrl._lama_inference_service, dummy_img, dummy_mask, page
    )
    _check("6a. Worker 创建成功", worker is not None, failures)
    _check("6b. Worker isRunning == False（未启动）",
           not worker.isRunning(), failures)

    # ================================================================ 7. _cleanup_tmp 静态方法
    print("[7] _cleanup_tmp 静态方法")
    # 创建临时文件
    tmp1 = os.path.join(tmpdir, "test_tmp1.tmp")
    tmp2 = os.path.join(tmpdir, "test_tmp2.tmp")
    tmp3 = os.path.join(tmpdir, "nonexistent.tmp")
    with open(tmp1, "w") as f:
        f.write("test")
    with open(tmp2, "w") as f:
        f.write("test")
    _check("7a. 临时文件已创建",
           os.path.exists(tmp1) and os.path.exists(tmp2), failures)

    # 清理
    LamaController._cleanup_tmp(tmp1, tmp2, tmp3)
    _check("7b. tmp1 已清理", not os.path.exists(tmp1), failures)
    _check("7c. tmp2 已清理", not os.path.exists(tmp2), failures)
    _check("7d. 不存在的路径不报错", True, failures)    # 如果执行到这里说明没报错

    # ================================================================ 8. busy 期间 Prev/Next 被拒绝
    print("[8] busy 期间 Prev/Next 被拒绝")
    ctrl._busy = True
    idx_before = ctrl._current_index
    ctrl._on_prev()
    _check("8a. busy 时 Prev 不切图",
           ctrl._current_index == idx_before, failures)
    ctrl._on_next()
    _check("8b. busy 时 Next 不切图",
           ctrl._current_index == idx_before, failures)
    ctrl._on_open()
    _check("8c. busy 时 Open 被拒绝",
           ctrl._current_index == idx_before, failures)
    ctrl._busy = False     # 恢复

    # ================================================================ 9. Q BLOCK：无 canvas 图
    print("[9] Q BLOCK：无 canvas 图")
    # 新建一个空 Page（canvas 无图）
    page2 = LamaPage()
    ctrl2 = page2.controller
    ctrl2._lama_inference_service._session = True
    page2.canvas.clear_mask()
    ctrl2._on_commit()
    status = page2._status_label.text()
    _check("9a. Q 无图 -> BLOCK",
           "no image" in status.lower(),
           failures, f"status='{status}'")

    # ---------------------------------------------------------------- 总结
    print("-" * 60)
    if failures:
        print(f"Phase 6 测试失败：{len(failures)} 项 -> {failures}")
        sys.exit(1)
    else:
        print("Phase 6 测试全部通过")
        sys.exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("测试抛出异常：")
        traceback.print_exc()
        sys.exit(1)
