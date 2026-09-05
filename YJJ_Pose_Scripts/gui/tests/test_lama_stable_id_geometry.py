"""MaskLabelService.assign_stable_ids —— Option 2 几何一致性定 ID 测试。

验证“Reference 7 点整体几何一致性 assignment”能防止 A/tracking 中
K2/K3、K5/K6 等 stable ID 交换污染最终 YOLO label。

不依赖 PySide6 / Qt，纯 numpy + cv2。

启动：
    python YJJ_Pose_Scripts/gui/tests/test_lama_stable_id_geometry.py
"""
from __future__ import annotations

import sys
from pathlib import Path

GUI_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GUI_DIR))

import numpy as np                                                   # noqa: E402
from features.lama.services.mask_label_service import (             # noqa: E402
    MaskLabelService, MaskComponent,
)

TOL = 1e-2  # 几何方法 residual≈0，浮点误差在 1e-9 级，用宽松 tolerant 比对位置


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        failures.append(name)


def _comps(centers: list[tuple[float, float]]) -> list[MaskComponent]:
    """由中心列表构造 MaskComponent（测试用，bbox/area 占位）。"""
    out = []
    for (x, y) in centers:
        out.append(MaskComponent(
            center=(float(x), float(y)),
            bbox=(int(round(x)) - 5, int(round(y)) - 5, 10, 10),
            area=100,
        ))
    return out


def _swap(d: dict[int, tuple[float, float]], i: int, j: int) -> dict[int, tuple[float, float]]:
    d = dict(d)
    d[i], d[j] = d[j], d[i]
    return d


def _translate(pts: list[tuple[float, float]], dx: float, dy: float) -> list[tuple[float, float]]:
    return [(x + dx, y + dy) for (x, y) in pts]


def _rotate(pts: list[tuple[float, float]], deg: float, about: tuple[float, float]) -> list[tuple[float, float]]:
    a = np.deg2rad(deg)
    ca, sa = float(np.cos(a)), float(np.sin(a))
    ox, oy = about
    out = []
    for (x, y) in pts:
        xx, yy = x - ox, y - oy
        out.append((xx * ca - yy * sa + ox, xx * sa + yy * ca + oy))
    return out


def _scale(pts: list[tuple[float, float]], s: float, about: tuple[float, float]) -> list[tuple[float, float]]:
    ox, oy = about
    return [((x - ox) * s + ox, (y - oy) * s + oy) for (x, y) in pts]


def _centroid(pts: list[tuple[float, float]]) -> tuple[float, float]:
    arr = np.array(pts, dtype=float)
    return (float(arr[:, 0].mean()), float(arr[:, 1].mean()))


# 非对称手部布局（canonical ID 0..6），用于正常/变换/交换场景
REF = [
    (200, 400),  # K0 wrist
    (210, 300),  # K1
    (190, 200),  # K2
    (230, 190),  # K3
    (170, 110),  # K4
    (250, 120),  # K5
    (220, 60),   # K6 fingertip
]


def _matches(ordered: list[tuple[float, float]] | None,
             expected: list[tuple[float, float]]) -> bool:
    if ordered is None:
        return False
    if len(ordered) != len(expected):
        return False
    return all(abs(ordered[i][0] - expected[i][0]) < TOL and
               abs(ordered[i][1] - expected[i][1]) < TOL
               for i in range(len(expected)))


def main() -> None:
    failures: list[str] = []
    svc = MaskLabelService()

    print("[1] 正常 7 点布局 -> ID 保持不变")
    cand = [tuple(p) for p in REF]
    predicted = {i: REF[i] for i in range(7)}
    ordered = svc.assign_stable_ids(_comps(cand), predicted, reference_points=REF, expected_count=7)
    _check("1a. 返回 7 点且 ID 不变（ordered[i]==REF[i]）",
           _matches(ordered, REF), failures, str(ordered))

    print("[2] 当前整体平移 -> 正常")
    cand = _translate([tuple(p) for p in REF], 35, -22)
    predicted = {i: REF[i] for i in range(7)}   # 故意给错误（未平移）预测
    ordered = svc.assign_stable_ids(_comps(cand), predicted, reference_points=REF, expected_count=7)
    _check("2a. 平移后仍能正确定 ID（ordered[i]==cand[i]）",
           _matches(ordered, cand), failures, str(ordered))

    print("[3] 当前整体旋转 -> 正常")
    c = _centroid(REF)
    cand = _rotate([tuple(p) for p in REF], 25.0, c)
    predicted = {i: REF[i] for i in range(7)}
    ordered = svc.assign_stable_ids(_comps(cand), predicted, reference_points=REF, expected_count=7)
    _check("3a. 旋转后仍能正确定 ID（ordered[i]==cand[i]）",
           _matches(ordered, cand), failures, str(ordered))

    print("[4] 当前整体缩放 -> 正常")
    c = _centroid(REF)
    cand = _scale([tuple(p) for p in REF], 1.3, c)
    predicted = {i: REF[i] for i in range(7)}
    ordered = svc.assign_stable_ids(_comps(cand), predicted, reference_points=REF, expected_count=7)
    _check("4a. 缩放后仍能正确定 ID（ordered[i]==cand[i]）",
           _matches(ordered, cand), failures, str(ordered))

    print("[5] predicted_centers 故意交换 K2/K3，component 几何正确 -> 必须恢复")
    cand = [tuple(p) for p in REF]
    predicted_swap = _swap({i: REF[i] for i in range(7)}, 2, 3)
    # 辅助信号：纯 predicted 路径（兜底）确实会被交换
    predicted_only = svc._assign_by_predicted(_comps(cand), predicted_swap, expected_count=7)
    ordered = svc.assign_stable_ids(_comps(cand), predicted_swap, reference_points=REF, expected_count=7)
    _check("5a. 几何方法恢复正确 K2/K3（ordered[2]==REF[2], ordered[3]==REF[3]）",
           ordered is not None and
           abs(ordered[2][0] - REF[2][0]) < TOL and abs(ordered[2][1] - REF[2][1]) < TOL and
           abs(ordered[3][0] - REF[3][0]) < TOL and abs(ordered[3][1] - REF[3][1]) < TOL,
           failures, str(ordered))
    _check("5b. 几何结果与（错误）predicted 不一致，证明已纠正",
           predicted_only != ordered, failures)
    _check("5c. 整体 ordered 仍为 REF（ID 未污染）",
           _matches(ordered, REF), failures, str(ordered))

    print("[6] K5/K6 同类交换 -> 恢复")
    cand = [tuple(p) for p in REF]
    predicted_swap = _swap({i: REF[i] for i in range(7)}, 5, 6)
    ordered = svc.assign_stable_ids(_comps(cand), predicted_swap, reference_points=REF, expected_count=7)
    _check("6a. 几何方法恢复正确 K5/K6（ordered[5]==REF[5], ordered[6]==REF[6]）",
           ordered is not None and
           abs(ordered[5][0] - REF[5][0]) < TOL and abs(ordered[5][1] - REF[5][1]) < TOL and
           abs(ordered[6][0] - REF[6][0]) < TOL and abs(ordered[6][1] - REF[6][1]) < TOL,
           failures, str(ordered))
    _check("6b. 整体 ordered 仍为 REF（ID 未污染）",
           _matches(ordered, REF), failures, str(ordered))

    print("[6b] 任意其它 pair（K1/K4）交换 -> 恢复")
    cand = [tuple(p) for p in REF]
    predicted_swap = _swap({i: REF[i] for i in range(7)}, 1, 4)
    ordered = svc.assign_stable_ids(_comps(cand), predicted_swap, reference_points=REF, expected_count=7)
    _check("6b1. 几何方法恢复 K1/K4（ordered[1]==REF[1], ordered[4]==REF[4]）",
           ordered is not None and
           abs(ordered[1][0] - REF[1][0]) < TOL and abs(ordered[1][1] - REF[1][1]) < TOL and
           abs(ordered[4][0] - REF[4][0]) < TOL and abs(ordered[4][1] - REF[4][1]) < TOL,
           failures, str(ordered))
    _check("6b2. 整体 ordered 仍为 REF（无 K1/K4 特判，几何一致性兜底）",
           _matches(ordered, REF), failures, str(ordered))

    print("[7] 两个关键点无法区分（退化重合）-> BLOCK")
    # ambiguity guard 仅在“单次 swap 后几何残差 ≈ best”时判定歧义。
    # 真实会触发的是两个 canonical 点完全/几乎重合（单次 swap 残差≈0），
    # 此时不应猜 ID，返回 None -> Q BLOCK；其余点拉散保证整体几何不退化。
    REF_AMB = [
        (0, 0),
        (100, 100), (100, 100),   # K1/K2 完全重合：swap 残差 == best == 0 -> 歧义
        (200, 50), (50, 200),
        (150, 150), (180, 30),
    ]
    cand_amb = [tuple(p) for p in REF_AMB]
    predicted = {i: REF_AMB[i] for i in range(7)}
    ordered = svc.assign_stable_ids(_comps(cand_amb), predicted, reference_points=REF_AMB, expected_count=7)
    _check("7a. 两关键点重合 -> 返回 None（Q BLOCK，不猜 ID）",
           ordered is None, failures, str(ordered))

    print("[8] component != 7 -> 保持原 BLOCK 行为")
    # 6 个 component，reference_points 给 7 个
    cand6 = _comps([REF[i] for i in range(6)])
    ordered6 = svc.assign_stable_ids(cand6, {i: REF[i] for i in range(6)},
                                     reference_points=REF, expected_count=7)
    _check("8a. 6 component -> None（数量不对）", ordered6 is None, failures)
    # 8 个 component
    cand8 = _comps([tuple(p) for p in REF] + [(300, 350)])
    ordered8 = svc.assign_stable_ids(cand8, {i: (REF + [(300, 350)])[i] for i in range(8)},
                                     reference_points=REF, expected_count=7)
    _check("8b. 8 component -> None（数量不对）", ordered8 is None, failures)

    print("[9] 无 reference_points（兜底）-> 退回原 predicted assignment，保持旧行为")
    # 即便 components 乱序，predicted 正确时仍能一一对应
    comps_shuf = _comps([tuple(p) for p in REF])
    import random
    random.seed(7)
    random.shuffle(comps_shuf)
    predicted = {i: REF[i] for i in range(7)}
    ordered = svc.assign_stable_ids(comps_shuf, predicted, reference_points=None, expected_count=7)
    _check("9a. reference_points=None 时退回 predicted 兜底且成功",
           _matches(ordered, REF), failures, str(ordered))

    print()
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
