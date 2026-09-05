"""pose_label_reorder 核心模块测试（canonical template 重排）。

覆盖需求 #1..#10：
1. template 与 target 顺序完全一致
2. K0/K1 swap 能恢复
3. 多组 permutation 能恢复
4. 平移后仍能恢复
5. scale 后仍能恢复
6. rotation 后仍能恢复
7. template K 与 external K 不一致 BLOCK
8. template K 与 base dataset K 不一致 BLOCK（在 rebuild 集成测试中）
9. residual 超阈值 BLOCK
10. ambiguous mapping BLOCK

启动：
    python YJJ_Pose_Scripts/01_data/test_pose_label_reorder.py
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pose_label_reorder as plr  # noqa: E402


def _check(name: str, cond: bool, failures: list[str], detail: str = "") -> None:
    if cond:
        print(f"  [OK]   {name}")
    else:
        print(f"  [FAIL] {name}  {detail}")
        failures.append(name)


# 非对称 7 点 canonical 布局（K0..K6，参照 lama REF 归一化），距离结构唯一
CANON = [
    (0.3125, 0.8333), (0.3281, 0.6250), (0.2969, 0.4167), (0.3594, 0.3958),
    (0.2656, 0.2292), (0.3906, 0.2500), (0.3438, 0.1250),
]


def _label_text(pts: list[tuple[float, float]]) -> str:
    tokens = ["0", "0.500000", "0.500000", "0.900000", "0.900000"]
    for x, y in pts:
        tokens.append(f"{x:.6f}")
        tokens.append(f"{y:.6f}")
        tokens.append("2")
    return " ".join(tokens) + "\n"


def _write_label(path: Path, pts: list[tuple[float, float]]) -> None:
    path.write_text(_label_text(pts), encoding="utf-8")


def _permute(pts: list[tuple[float, float]], perm: list[int]) -> list[tuple[float, float]]:
    return [pts[p] for p in perm]


def _inverse(perm: list[int]) -> list[int]:
    """assignment[i]=j 表示 dest K_i 取自源第 j 个三元组（源 j 位是 canonical P_perm[j]）。"""
    return [perm.index(i) for i in range(len(perm))]


def _rotate(pts: list[tuple[float, float]], deg: float) -> list[tuple[float, float]]:
    ox = sum(p[0] for p in pts) / len(pts)
    oy = sum(p[1] for p in pts) / len(pts)
    a = math.radians(deg)
    out = []
    for x, y in pts:
        xx, yy = x - ox, y - oy
        out.append((ox + xx * math.cos(a) - yy * math.sin(a),
                    oy + xx * math.sin(a) + yy * math.cos(a)))
    return out


def main() -> None:
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="reorder_test_") as td:
        root = Path(td)

        def make_tmpl(name: str, pts: list[tuple[float, float]]) -> Path:
            p = root / name
            _write_label(p, pts)
            return p

        tmpl7 = make_tmpl("tmpl7.txt", CANON)

        print("[1] template 与 target 顺序完全一致 -> 恒等映射")
        t = root / "id.txt"
        _write_label(t, CANON)
        asg, res = plr.reorder_file(tmpl7, t)
        _check("1a. 恒等映射 K0<-K0...",
               asg == list(range(7)), failures, str(asg))
        _check("1b. residual≈0", res < 1e-6, failures, f"{res:.6f}")

        print("[2] K0/K1 swap 能恢复")
        t = root / "swap01.txt"
        _write_label(t, _permute(CANON, [1, 0, 2, 3, 4, 5, 6]))
        asg, _res = plr.reorder_file(tmpl7, t, root / "swap01_out.txt")
        _check("2a. mapping 恢复（K0<-K1; K1<-K0）",
               asg == [1, 0, 2, 3, 4, 5, 6], failures, str(asg))
        _check("2b. 输出 == template 顺序内容",
               (root / "swap01_out.txt").read_text(encoding="utf-8")
               == _label_text(CANON), failures)

        print("[3] 多组 permutation 能恢复")
        perm = [5, 1, 6, 0, 3, 2, 4]
        t = root / "multi.txt"
        _write_label(t, _permute(CANON, perm))
        asg, _res = plr.reorder_file(tmpl7, t, root / "multi_out.txt")
        _check("3a. mapping == perm 的逆（dest K_i <- 源 perm^-1(i)）",
               asg == _inverse(perm), failures, str(asg))
        _check("3b. 输出 == template 顺序内容",
               (root / "multi_out.txt").read_text(encoding="utf-8")
               == _label_text(CANON), failures)

        print("[4] 平移后仍能恢复")
        trans = [(x + 0.06, y - 0.04) for x, y in CANON]
        perm4 = [2, 0, 5, 1, 6, 4, 3]
        t = root / "trans.txt"
        _write_label(t, _permute(trans, perm4))
        asg, res = plr.reorder_file(tmpl7, t)
        _check("4a. 平移 + 乱序恢复（inverse mapping）", asg == _inverse(perm4),
               failures, str(asg))
        _check("4b. residual 低于阈值", res < plr.DEFAULT_MAX_RESIDUAL,
               failures, f"{res:.6f}")

        print("[5] scale 后仍能恢复")
        ox = sum(p[0] for p in CANON) / 7
        oy = sum(p[1] for p in CANON) / 7
        scaled = [(ox + (x - ox) * 1.25, oy + (y - oy) * 1.25) for x, y in CANON]
        perm5 = [3, 5, 0, 2, 6, 1, 4]
        t = root / "scale.txt"
        _write_label(t, _permute(scaled, perm5))
        asg, res = plr.reorder_file(tmpl7, t)
        _check("5a. scale + 乱序恢复（inverse mapping）", asg == _inverse(perm5),
               failures, str(asg))
        _check("5b. residual 低于阈值", res < plr.DEFAULT_MAX_RESIDUAL,
               failures, f"{res:.6f}")

        print("[6] rotation 后仍能恢复")
        rot = _rotate(CANON, 27.0)
        perm6 = [6, 4, 1, 3, 0, 2, 5]
        t = root / "rot.txt"
        _write_label(t, _permute(rot, perm6))
        asg, res = plr.reorder_file(tmpl7, t)
        _check("6a. rotation + 乱序恢复（inverse mapping）", asg == _inverse(perm6),
               failures, str(asg))
        _check("6b. residual 低于阈值", res < plr.DEFAULT_MAX_RESIDUAL,
               failures, f"{res:.6f}")

        print("[7] template K 与 external K 不一致 -> BLOCK")
        t6 = root / "t6.txt"
        _write_label(t6, CANON[:6])
        try:
            plr.reorder_file(tmpl7, t6)
            _check("7a. 7 vs 6 -> BLOCK", False, failures, "未 BLOCK")
        except plr.ReorderError as e:
            _check("7a. 7 vs 6 -> BLOCK", "template has 7" in str(e)
                   or "target has 6" in str(e), failures, str(e))

        print("[9] residual 超阈值 -> BLOCK（微扰动 + 极小阈值确定性触发）")
        noisy = [(x + (0.004 * (i % 3)) - 0.002, y - 0.005)
                 for i, (x, y) in enumerate(CANON)]
        t = root / "noisy.txt"
        _write_label(t, noisy)
        try:
            plr.reorder_file(tmpl7, t, max_residual=1e-4)
            _check("9a. 噪声 residual > 1e-4 -> BLOCK", False, failures, "未 BLOCK")
        except plr.ReorderError as e:
            _check("9a. 噪声 residual > 1e-4 -> BLOCK", "residual" in str(e),
                   failures, str(e))

        print("[10] ambiguous mapping -> BLOCK（两个点完全重合，不猜 ID）")
        dup_canon = list(CANON)
        dup_canon[3] = dup_canon[4]  # K3/K4 完全重合
        tmpl_dup = make_tmpl("tmpl_dup.txt", dup_canon)
        t = root / "dup.txt"
        _write_label(t, dup_canon)
        try:
            plr.reorder_file(tmpl_dup, t)
            _check("10a. 重合点 -> AMBIGUOUS BLOCK", False, failures, "未 BLOCK")
        except plr.ReorderAmbiguousError as e:
            _check("10a. 重合点 -> AMBIGUOUS BLOCK", "ambiguous" in str(e),
                   failures, str(e))

        # 补充：reorder_directory 整目录路径可跑（dry-run 不出报告也可正常返回）
        print("[11] 单 template 残差高；train bank 内存在更近 reference -> 成功")
        # 构造一个“姿态微变”位移场：external 相对 canonical 位移 Δ，
        # bank 里含较接近的 reference（同种“手型变化”），单模板 canonical 则超阈值。
        def _compr(pts: list[tuple[float, float]], cx: float, cy: float,
                   ox: float, oy: float) -> list[tuple[float, float]]:
            return [(ox + (x - ox) * cx, oy + (y - oy) * cy) for x, y in pts]

        def _find_ratio(axis: str) -> float:
            """找各向异性压缩比：让 canonical 单模板对该 pose 残差 >0.15。"""
            ox = sum(p[0] for p in CANON) / 7
            oy = sum(p[1] for p in CANON) / 7
            ratio = 0.95
            while ratio > 0.1:
                pts = _compr(CANON, ratio, 1.0, ox, oy) if axis == "x" \
                    else _compr(CANON, 1.0, ratio, ox, oy)
                try:
                    plr.compute_assignment(CANON, pts, max_residual=0.15)
                except plr.ReorderError:
                    return ratio
                ratio -= 0.05
            return 0.1

        cx = _find_ratio("x")
        ox = sum(p[0] for p in CANON) / 7
        oy = sum(p[1] for p in CANON) / 7
        b_pts = _compr(CANON, cx, 1.0, ox, oy)          # “手型变化”后的 external 基序
        ref_pose = plr.Reference("t1_pose.txt", b_pts)  # bank 里的同 pose train label
        refs11 = [plr.Reference("t0_canon.txt", CANON), ref_pose]
        perm11 = [2, 0, 5, 1, 6, 4, 3]
        t11 = root / "bank1.txt"
        _write_label(t11, _permute(b_pts, perm11))
        try:
            plr.reorder_file(tmpl7, t11)
            _check("11a. 单 template 残差超限/歧义 -> BLOCK（证明单模板脆弱）",
                   False, failures, "未 BLOCK")
        except plr.ReorderError as e:
            _check("11a. 单 template 残差超限/歧义 -> BLOCK（证明单模板脆弱）",
                   "residual" in str(e) or "ambiguous" in str(e),
                   failures, str(e))
        m = plr.bank_best_match(refs11, _permute(b_pts, perm11))
        _check("11b. bank 命中同 pose 的 t1_pose.txt（而非 canonical）",
               m.reference.name == "t1_pose.txt", failures, m.reference.name)
        _check("11c. best residual <= 0.15",
               m.normalized_rms <= plr.DEFAULT_MAX_RESIDUAL,
               failures, f"{m.normalized_rms:.6f}")
        head11, tri11, _pts11 = plr.read_label(t11)
        text11 = plr.build_reordered_text(head11, tri11, m.assignment) + "\n"
        _check("11d. 输出统一到 canonical K0..K6（内容 = b_pts 顺序）",
               text11 == _label_text(b_pts), failures)

        print("[12] 不同 external 各自命中不同 train reference")
        cy_ratio = _find_ratio("y")
        refX = plr.Reference("ref_x.txt", _compr(CANON, cx, 1.0, ox, oy))
        refY = plr.Reference("ref_y.txt", _compr(CANON, 1.0, cy_ratio, ox, oy))
        bank12 = [refX, refY]
        extA = _compr(CANON, cx, 1.0, ox, oy)     # “瘦长”手型
        extB = _compr(CANON, 1.0, cy_ratio, ox, oy)  # “扁宽”手型
        permA = [3, 1, 0, 6, 4, 2, 5]
        permB = [0, 2, 4, 1, 3, 6, 5]
        mA = plr.bank_best_match(bank12, _permute(extA, permA))
        mB = plr.bank_best_match(bank12, _permute(extB, permB))
        _check("12a. external A（瘦长）命中 ref_x.txt", mA.reference.name == "ref_x.txt",
               failures, mA.reference.name)
        _check("12b. external B（扁宽）命中 ref_y.txt", mB.reference.name == "ref_y.txt",
               failures, mB.reference.name)
        tA = root / "bankA.txt"
        _write_label(tA, _permute(extA, permA))
        headA, triA, _ = plr.read_label(tA)
        _check("12c. A 输出统一到 canonical 顺序（= extA 基序文本）",
               plr.build_reordered_text(headA, triA, mA.assignment) + "\n"
               == _label_text(extA), failures)

        print("[13] bank_dry_run 整批只读试跑统计")
        refs13 = [plr.Reference("r0.txt", CANON)]
        files13 = []
        for i, sid in enumerate(("x1", "x2", "x3")):
            pts = _permute(CANON, perm11) if i else CANON
            fp = root / f"{sid}.txt"
            _write_label(fp, pts)
            files13.append((f"color_{sid}.txt", fp))
        rows13, max13 = plr.bank_dry_run(refs13, files13)
        _check("13a. 3/3 OK", len(rows13) == 3
               and all(r.status == "OK" for r in rows13), failures)
        _check("13b. max_residual 返回有限值",
               isinstance(max13, float) and max13 >= 0.0, failures, f"{max13}")

        src_dir = root / "src_dir"
        src_dir.mkdir()
        _write_label(src_dir / "a.txt", _permute(CANON, [1, 0, 2, 3, 4, 5, 6]))
        _write_label(src_dir / "b.txt", CANON)
        dst_dir = root / "dst_dir"
        rows, ok, fail = plr.reorder_directory(src_dir, dst_dir, tmpl7, dry_run=True)
        _check("dir. dry-run ok=2 fail=0", ok == 2 and fail == 0,
               failures, f"ok={ok} fail={fail}")
        _check("dir. 每行 status=OK", all(r.status == "OK" for r in rows), failures)

    print("-" * 60)
    if failures:
        print(f"RESULT: {len(failures)} FAILED -> {failures}")
        sys.exit(1)
    print("RESULT: ALL PASSED")


if __name__ == "__main__":
    main()
