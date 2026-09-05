"""LaMa 关键点数量 UI 可配置 + bbox 比例 padding 专项测试（对应改动需求）。

覆盖用户需求“五、测试”全部 10 条最小验证：
1. 默认 count=7 行为不变
2. count=5：5 components 正常 / 4·6 components BLOCK
3. count=8：ordered_points 输出 8 个 / label 含 8 组 x/y/v
4. predicted_centers ID 必须严格为 0..N-1
5. reference_points 数量错误 BLOCK
6. SetRef 后 UI keypoint count 被锁定（且 7→5 中途切换被 BLOCK）
7. Reset / Clear Reference 后重新允许修改
8. N=7 的 K2/K3 predicted swap：Reference geometry 防线仍能恢复正确 ID
9. N>7 不允许走 factorial permutations（匈牙利 O(N^3) 处理 N=10）
10. bbox：小范围用 min padding / 大范围用比例 padding / image boundary clamp

纯算法部分不依赖 Qt；UI 锁定逻辑走 LamaController + MockPage（headless offscreen）。

启动：
    python YJJ_Pose_Scripts/gui/tests/test_lama_keypoint_count_configurable.py
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # 必须早于 PySide6 import

import sys
import time
import traceback
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))

import numpy as np                                                   # noqa: E402
import cv2                                                          # noqa: E402
from PySide6.QtCore import QObject, Signal                          # noqa: E402
from PySide6.QtGui import QImage                                    # noqa: E402
from PySide6.QtWidgets import QApplication                          # noqa: E402

from features.lama.services.mask_label_service import (             # noqa: E402
    MaskLabelService, MaskComponent,
)
from features.lama.services.assignment_solver import solve_linear_assignment  # noqa: E402
from features.lama.services.lama_inference_service import LamaInferenceService  # noqa: E402
from features.lama.lama_controller import LamaController           # noqa: E402


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        failures.append(name)


def _make_mask(w: int, h: int, centers: list[tuple[int, int]], radius: int = 15) -> np.ndarray:
    """生成测试 mask：在指定 center 画白圆（每个圆 = 一个 component）。"""
    mask = np.zeros((h, w), dtype=np.uint8)
    for cx, cy in centers:
        cv2.circle(mask, (int(cx), int(cy)), radius, 255, -1)
    return mask


def _qimg_gray(np_mask: np.ndarray) -> QImage:
    h, w = np_mask.shape[:2]
    return QImage(np.ascontiguousarray(np_mask).tobytes(), w, h, w,
                  QImage.Format.Format_Grayscale8).copy()


def _qimg_rgb(np_rgb: np.ndarray) -> QImage:
    h, w = np_rgb.shape[:2]
    return QImage(np.ascontiguousarray(np_rgb).tobytes(), w, h, w * 3,
                  QImage.Format.Format_RGB888).copy()


# ============================================================ 纯算法部分
def _service_tests(failures: list[str]) -> None:
    svc = MaskLabelService()

    print("[1] 默认 count=7 行为不变")
    c7 = [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200),
          (300, 200), (200, 300)]
    m7 = _make_mask(640, 480, c7)
    r7 = svc.make_label(m7, {i: c for i, c in enumerate(c7)}, (640, 480), expected_count=7)
    _check("1a. 默认 N=7 且 7 components -> 成功", r7 is not None, failures)
    _check("1b. 默认行为 ordered_points 长度 7",
           r7 is not None and len(r7.ordered_points) == 7, failures)

    print("[2] count=5：5 正常 / 4·6 BLOCK")
    c5 = c7[:5]
    m5 = _make_mask(640, 480, c5)
    r5 = svc.make_label(m5, {i: c for i, c in enumerate(c5)}, (640, 480), expected_count=5)
    _check("2a. 5 components + expected_count=5 -> 成功", r5 is not None, failures)
    _check("2b. ordered_points 长度 = 5",
           r5 is not None and len(r5.ordered_points) == 5, failures)
    m4 = _make_mask(640, 480, c5[:4])
    _check("2c. 4 components + expected_count=5 -> BLOCK",
           svc.make_label(m4, {i: c for i, c in enumerate(c5[:4])}, (640, 480),
                          expected_count=5) is None, failures)
    m6 = _make_mask(640, 480, c7[:6])
    _check("2d. 6 components + expected_count=5 -> BLOCK",
           svc.make_label(m6, {i: c for i, c in enumerate(c7[:6])}, (640, 480),
                          expected_count=5) is None, failures)

    print("[3] count=8：ordered_points 8 个 / label 含 8 组 x/y/v")
    c8 = c7 + [(400, 300)]      # 7 + 1 = 8 个中心
    m8 = _make_mask(640, 480, c8)
    r8 = svc.make_label(m8, {i: c for i, c in enumerate(c8)}, (640, 480), expected_count=8)
    _check("3a. 8 components + expected_count=8 -> 成功", r8 is not None, failures)
    _check("3b. ordered_points 长度 = 8",
           r8 is not None and len(r8.ordered_points) == 8, failures)
    if r8 is not None:
        content = svc.make_label_content(r8.bbox, r8.ordered_points, (640, 480))
        toks = content.split()
        # 1(class) + 4(bbox) + 8*3(kp) = 29
        _check("3c. label token 数 = 29（8 组三元组）",
               len(toks) == 29, failures, f"got {len(toks)}")
        _check("3d. 前 5 个 token = class + bbox(4)",
               toks[0] == "0" and len(toks[1:5]) == 4, failures)
        _check("3e. 所有 keypoint vis = 2",
               all(toks[5 + i + 2] == "2" for i in range(0, 24, 3)), failures)

    print("[4] predicted_centers ID 必须严格为 0..N-1")
    comps5 = svc.extract_components(m5, min_area=50)
    # 缺一个 key（只有 0..3）
    pred_missing = {i: c5[i] for i in range(4)}
    o_miss = svc.assign_stable_ids(comps5, pred_missing, expected_count=5, reference_points=None)
    _check("4a. predicted 缺 key 4 -> None（BLOCK）", o_miss is None, failures)
    # 非连续 key（缺 1，含 5）
    pred_noncontig = {0: c5[0], 2: c5[2], 3: c5[3], 4: c5[4], 5: c5[1]}
    o_nc = svc.assign_stable_ids(comps5, pred_noncontig, expected_count=5, reference_points=None)
    _check("4b. predicted key 非连续 {0,2,3,4,5} -> None（BLOCK）",
           o_nc is None, failures)
    # 严格 0..N-1 -> 成功
    pred_ok = {i: c5[i] for i in range(5)}
    o_ok = svc.assign_stable_ids(comps5, pred_ok, expected_count=5, reference_points=None)
    _check("4c. predicted key 严格 0..4 -> 成功", o_ok is not None, failures)

    print("[5] reference_points 数量错误 BLOCK")
    ref_short = c5[:4]
    o_rs = svc.assign_stable_ids(comps5, pred_ok, expected_count=5, reference_points=ref_short)
    _check("5a. reference_points 长度 4 != 5 -> None", o_rs is None, failures)
    ref_long = c5 + [c5[0]]
    o_rl = svc.assign_stable_ids(comps5, pred_ok, expected_count=5, reference_points=ref_long)
    _check("5b. reference_points 长度 6 != 5 -> None", o_rl is None, failures)

    print("[8] N=7 的 K2/K3 predicted swap：Reference geometry 防线恢复正确 ID")
    REF = [(200, 400), (210, 300), (190, 200), (230, 190), (170, 110),
           (250, 120), (220, 60)]
    cand = [tuple(p) for p in REF]
    pred_swap = {i: REF[i] for i in range(7)}
    pred_swap[2], pred_swap[3] = pred_swap[3], pred_swap[2]   # 故意交换 K2/K3
    o_swap = svc.assign_stable_ids([MaskComponent(center=t, bbox=(0, 0, 1, 1), area=1)
                                    for t in cand], pred_swap,
                                   expected_count=7, reference_points=REF)
    _check("8a. geometry 恢复 correct K2/K3（ordered==REF）",
           o_swap is not None and
           all(abs(o_swap[i][0] - REF[i][0]) < 1e-6 and
               abs(o_swap[i][1] - REF[i][1]) < 1e-6 for i in range(7)),
           failures, str(o_swap))

    print("[9] N>7 不允许走 factorial permutations（匈牙利 O(N^3)）")
    rng = np.random.default_rng(0)
    cost10 = rng.random((10, 10)) * 100.0
    perm10 = solve_linear_assignment(cost10)
    ok_perm = (perm10 is not None and len(perm10) == 10 and
               len(set(perm10)) == 10)
    sel = sum(cost10[i, perm10[i]] for i in range(10)) if perm10 else -1.0
    # 暴力验证匈牙利给出的是最小代价（N=10 可暴力比对）
    from itertools import permutations as _perm
    best = min(sum(cost10[i, p[i]] for i in range(10)) for p in _perm(range(10)))
    _check("9a. N=10 匈牙利给出合法一一分配", ok_perm, failures, str(perm10))
    _check("9b. N=10 匈牙利 = 暴力最小总代价（非 factorial 路径但结果正确）",
           abs(sel - best) < 1e-6, failures, f"hungarian={sel} brute={best}")
    c10 = [(i * 40 + 20, (i % 3) * 90 + 20) for i in range(10)]
    m10 = _make_mask(640, 480, c10)
    t0 = time.perf_counter()
    r10 = svc.make_label(m10, {i: c for i, c in enumerate(c10)}, (640, 480), expected_count=10)
    dt = time.perf_counter() - t0
    _check("9c. N=10 make_label 成功（ordered 长度 10）",
           r10 is not None and len(r10.ordered_points) == 10, failures)
    _check("9d. N=10 在 2s 内完成（无阶乘爆炸）", dt < 2.0, failures, f"{dt:.3f}s")
    # N>7 几何 assignment 正确性：参考平移后仍能恢复
    ref10 = c10
    cand10 = [(x + 37, y - 19) for (x, y) in c10]
    pred_wrong = {i: c10[i] for i in range(10)}   # 给未平移的（错误）预测
    o10 = svc.assign_stable_ids([MaskComponent(center=t, bbox=(0, 0, 1, 1), area=1)
                                 for t in cand10], pred_wrong,
                                expected_count=10, reference_points=ref10)
    _check("9e. N=10 geometry 恢复正确（ordered==cand10）",
           o10 is not None and
           all(abs(o10[i][0] - cand10[i][0]) < 1e-6 and
               abs(o10[i][1] - cand10[i][1]) < 1e-6 for i in range(10)),
           failures, str(o10))

    print("[10] bbox 规则：小范围 min padding / 大范围比例 / 边界 clamp / 中心在内")
    # 半径 8 的圆、点间距 >= 32px：保证 5 个 component 互不粘连，bbox 每边外扩 8px。
    # 预期一律按“真实提取出的 component bbox union + pad 公式”计算，
    # 不能用 keypoint 中心 min/max 代替（那正是被废弃的贴边估算）。

    def _expected_bbox(comps, w_img: int, h_img: int) -> tuple[int, int, int, int]:
        # 与 build_bbox 逐行一致：先浮点 clamp，再对 w/h 做 int()（不对端点提前取整）
        min_x = float(min(c.bbox[0] for c in comps))
        min_y = float(min(c.bbox[1] for c in comps))
        max_x = float(max(c.bbox[0] + c.bbox[2] for c in comps))
        max_y = float(max(c.bbox[1] + c.bbox[3] for c in comps))
        uw, uh = max_x - min_x, max_y - min_y
        pad_x = max(float(svc.DEFAULT_MIN_BBOX_PADDING), uw * svc.DEFAULT_BBOX_PADDING_RATIO_X)
        pad_y = max(float(svc.DEFAULT_MIN_BBOX_PADDING), uh * svc.DEFAULT_BBOX_PADDING_RATIO_Y)
        x0 = max(0.0, min_x - pad_x)
        y0 = max(0.0, min_y - pad_y)
        x1 = min(float(w_img), max_x + pad_x)
        y1 = min(float(h_img), max_y + pad_y)
        return (int(x0), int(y0), int(x1 - x0), int(y1 - y0))

    # 小范围：5 点聚在 ~130px 内 -> union_w*0.12 < 20 -> pad 取 MIN_PADDING=20
    c_small = [(200, 200), (245, 200), (200, 248), (245, 248), (222, 224)]
    m_small = _make_mask(640, 480, c_small, radius=8)
    r_small = svc.make_label(m_small, {i: c for i, c in enumerate(c_small)},
                             (640, 480), expected_count=5)
    _check("10a. 小范围 5 个独立 component 正常标注",
           r_small is not None and len(r_small.components) == 5, failures,
           f"components={len(r_small.components) if r_small else 'None'}")
    if r_small is not None:
        comp_min_x = min(c.bbox[0] for c in r_small.components)
        uw = max(c.bbox[0] + c.bbox[2] for c in r_small.components) - comp_min_x
        exp = _expected_bbox(r_small.components, 640, 480)
        _check("10b. 小范围 union_w*0.12 < 20（走 min padding 分支的前提）",
               uw * svc.DEFAULT_BBOX_PADDING_RATIO_X < svc.DEFAULT_MIN_BBOX_PADDING,
               failures, f"uw={uw}")
        _check("10c. 小范围 bbox.x == component 最小 x - MIN_PADDING(20)",
               r_small.bbox[0] == max(0, comp_min_x - svc.DEFAULT_MIN_BBOX_PADDING),
               failures, f"bbox={r_small.bbox} comp_min_x={comp_min_x}")
        _check("10d. 小范围 bbox 与公式预期一致",
               r_small.bbox == exp, failures, f"bbox={r_small.bbox} expected={exp}")

    # 大范围：spread 400 -> union_w*0.12 > 20 -> 比例 padding
    c_big = [(100, 100), (500, 120), (120, 460), (480, 440), (300, 280)]
    m_big = _make_mask(640, 480, c_big, radius=8)
    r_big = svc.make_label(m_big, {i: c for i, c in enumerate(c_big)},
                           (640, 480), expected_count=5)
    _check("10e. 大范围 5 个独立 component 正常标注",
           r_big is not None and len(r_big.components) == 5, failures,
           f"components={len(r_big.components) if r_big else 'None'}")
    if r_big is not None:
        comp_min_x = min(c.bbox[0] for c in r_big.components)
        comp_max_x = max(c.bbox[0] + c.bbox[2] for c in r_big.components)
        uw = comp_max_x - comp_min_x
        pad_x = max(svc.DEFAULT_MIN_BBOX_PADDING, uw * svc.DEFAULT_BBOX_PADDING_RATIO_X)
        exp = _expected_bbox(r_big.components, 640, 480)
        _check("10f. 大范围比例 padding 生效（pad_x > 20）", pad_x > 20, failures, f"pad_x={pad_x}")
        _check("10g. 大范围 bbox.x == component 最小 x - 比例 padding",
               r_big.bbox[0] == max(0, int(comp_min_x - pad_x)),
               failures, f"bbox={r_big.bbox} comp_min_x={comp_min_x} pad_x={pad_x}")
        _check("10h. 大范围 bbox 与公式预期一致",
               r_big.bbox == exp, failures, f"bbox={r_big.bbox} expected={exp}")

    # 边界 clamp：最左 component bbox x=7，padding 后 < 0 -> clamp 到 0
    c_edge = [(15, 240), (120, 100), (300, 460), (500, 300), (240, 60)]
    m_edge = _make_mask(640, 480, c_edge, radius=8)
    r_edge = svc.make_label(m_edge, {i: c for i, c in enumerate(c_edge)},
                           (640, 480), expected_count=5)
    _check("10i. 靠近左边界时 bbox.x clamp 到 0",
           r_edge is not None and r_edge.bbox[0] == 0, failures,
           str(r_edge.bbox if r_edge else None))

    # 关键点中心必须位于最终 bbox 内（#20）
    for tag, r in (("小范围", r_small), ("大范围", r_big)):
        if r is None:
            continue
        bx, by, bw, bh = r.bbox
        inside = all(bx <= px <= bx + bw and by <= py <= by + bh
                     for (px, py) in r.ordered_points)
        _check(f"10j. {tag} 所有 keypoint 中心在 bbox 内", inside, failures,
               f"bbox={r.bbox} points={r.ordered_points}")


# ============================================================ UI 锁定逻辑（Controller + MockPage）
class _MockCanvas:
    def __init__(self) -> None:
        self._src = QImage()
        self._mask = QImage()

    def source_image(self) -> QImage:
        return self._src

    def mask_image(self) -> QImage:
        return self._mask

    def load_image(self, _p: str) -> bool:
        return True

    def clear_mask(self) -> None:
        pass

    def set_mask(self, _q: QImage) -> None:
        pass

    def set_brush_radius(self, _r: int) -> None:
        pass

    def isEnabled(self) -> bool:
        return True

    def setEnabled(self, _v: bool) -> None:
        pass


class _MockPage(QObject):
    openRequested = Signal()
    clearMaskRequested = Signal()
    setRefRequested = Signal()
    testOneRequested = Signal()
    assistMaskRequested = Signal()
    commitRequested = Signal()
    helpRequested = Signal()
    prevRequested = Signal()
    nextRequested = Signal()
    brushRadiusChanged = Signal(int)

    def __init__(self, kp: int = 7) -> None:
        super().__init__()
        self._kp = kp
        self._locked = False
        self.canvas = _MockCanvas()

    def set_status_text(self, _t: str) -> None:
        pass

    def set_current_info(self, *_a, **_k) -> None:
        pass

    def set_reference_info(self, *_a, **_k) -> None:
        pass

    def set_reference_ready(self, *_a, **_k) -> None:
        pass

    def set_prediction_info(self, *_a, **_k) -> None:
        pass

    def set_busy(self, _b: bool) -> None:
        pass

    def KeypointCount(self) -> int:
        return self._kp

    def SetKeypointCountEditable(self, editable: bool) -> None:
        # editable=False -> 锁定
        self._locked = not bool(editable)


def _controller_lock_tests(failures: list[str]) -> None:
    print("[6][7] Controller：SetRef 锁定 / 中途切换 BLOCK / ClearReference 解锁")
    app = QApplication.instance() or QApplication(["t"])
    # 避免 ONNX GPU 模型加载阻塞/失败影响锁定逻辑测试
    LamaInferenceService.load_model = lambda self, p: True          # type: ignore
    LamaInferenceService.is_loaded = lambda self: True              # type: ignore
    LamaInferenceService.available_providers = lambda self: []      # type: ignore
    LamaInferenceService.active_providers = lambda self: []         # type: ignore
    LamaInferenceService.last_error = lambda self: ""               # type: ignore

    page = _MockPage(kp=7)
    # 构造 7 个 component 的 mask + 一张非空原图
    c7 = [(100, 100), (200, 100), (300, 100), (100, 200), (200, 200),
          (300, 200), (200, 300)]
    img = _qimg_rgb(np.zeros((480, 640, 3), dtype=np.uint8))
    mask = _qimg_gray(_make_mask(640, 480, c7))
    page.canvas._src = img
    page.canvas._mask = mask

    ctrl = LamaController(page, log_sink=lambda _: None)
    app.processEvents()  # 让模型加载 worker 信号落地

    # 未 SetRef：未锁定，数量等于 UI 当前值
    _check("6a. SetRef 前未锁定（_expected_keypoint_count==0）",
           ctrl._expected_keypoint_count == 0, failures)
    _check("6b. SetRef 前 UI 可编辑（page._locked==False）",
           page._locked is False, failures)
    _check("6c. SetRef 前 ExpectedKeypointCount 取 UI 值 7",
           ctrl.ExpectedKeypointCount() == 7, failures)

    # SetRef 成功 -> 锁定 7
    ctrl._on_set_ref()
    _check("6d. SetRef 后 _expected_keypoint_count==7",
           ctrl._expected_keypoint_count == 7, failures)
    _check("6e. SetRef 后 UI 被锁定（page._locked==True）",
           page._locked is True, failures)
    _check("6f. SetRef 后 ExpectedKeypointCount 仍为 7",
           ctrl.ExpectedKeypointCount() == 7, failures)

    # 禁止 7->5 中途切换：把 UI 改成 5，再次 SetRef 必须 BLOCK（mask 仍是 7 点）
    page._kp = 5
    ctrl._on_set_ref()
    _check("6g. 中途把 UI 改 5 仍被锁定在 7（不切换）",
           ctrl._expected_keypoint_count == 7 and page._locked is True, failures)
    _check("6h. 锁定后改 UI 值不影响生效数量（仍 7）",
           ctrl.ExpectedKeypointCount() == 7, failures)

    # ClearReference / Reset -> 解锁
    ctrl.ClearReference()
    _check("7a. ClearReference 后解锁（_expected_keypoint_count==0）",
           ctrl._expected_keypoint_count == 0, failures)
    _check("7b. ClearReference 后 UI 重新可编辑（page._locked==False）",
           page._locked is False, failures)
    _check("7c. ClearReference 后生效数量回到 UI 当前值 5",
           ctrl.ExpectedKeypointCount() == 5, failures)

    ctrl.cleanup_worker()


def main() -> None:
    failures: list[str] = []
    _service_tests(failures)
    print()
    try:
        _controller_lock_tests(failures)
    except Exception:
        print("Controller 锁定测试抛出异常：")
        traceback.print_exc()
        failures.append("controller_lock_tests_exception")

    print("-" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
